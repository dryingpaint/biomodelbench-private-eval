"""Provider-side node configuration (config.yaml, never committed)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .contract import Contract

PKG = Path(__file__).resolve().parent  # the bmb_eval package directory


@dataclass
class TaskBinding:
    id: str
    dir: Path          # contract.yaml + score.py
    inputs: Path       # samples table the model sees (parquet)
    answers: Path      # private answers parquet (only the scorer sees this)
    contract: Contract
    inputs_dir: Path | None = None  # folder the model sees, holding inputs + its files
    answers_dir: Path | None = None  # scorer-only files (data.private)
    test_inputs_dir: Path | None = None  # held-out test half, when the contract has `split`

    @property
    def test_inputs(self) -> Path | None:
        return self.test_inputs_dir / "samples.parquet" if self.test_inputs_dir else None


@dataclass
class ModalSettings:
    """The `modal:` section of a Modal node's config.yaml (sandbox.backend: modal)."""
    environment: str = "bmb-eval"     # Modal environment holding the evaluation volumes and app
    app: str = "bmb-eval-sandbox"     # Modal app the sandboxes belong to
    profile: str | None = None        # ~/.modal.toml profile (MODAL_PROFILE), e.g. biomodelbench
    gpu: str = "H100"
    max_gpus: int = 64                # GPUs this node may hold at once, across all jobs
    max_jobs: int = 16                # evaluations at once
    max_gpus_per_job: int = 8         # what one Modal container can have
    gpu_memory_gb: int = 80
    volume_prefix: str = ""           # names this node's volumes; default: the node name
    retries: int = 2                  # retries of a stage after a Modal (not model) failure
    memory_headroom_mb: int = 2048    # container memory above the contract's, for the launcher

    @classmethod
    def parse(cls, raw: dict | None, node_name: str) -> "ModalSettings":
        known = {f for f in cls.__dataclass_fields__}
        raw = dict(raw or {})
        if bad := set(raw) - known:
            raise SystemExit(f"modal: unknown setting(s) {sorted(bad)}")
        s = cls(**raw)
        if not s.volume_prefix:
            s.volume_prefix = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in node_name)[:24]
        return s


@dataclass
class Config:
    path: Path
    node_name: str
    state_dir: Path
    backend: str
    image: str
    runtime: str | None
    allow_unisolated: bool
    webhook_url: str | None
    infra_failure_threshold: int
    retain_bundles: bool
    cache_gb: int      # model-file cache size before least-recently-used files go
    # Evaluations run at once (each admitted only when its contract's CPUs, memory and GPUs
    # fit what is free). 1 (the default) runs them one at a time, as nodes always have.
    max_parallel: int
    hub: dict          # {url, node_id, secret} once connected; {} before
    blob_dir: Path | None = None   # model-file cache; default state_dir/blobs
    modal: ModalSettings | None = None   # sandbox.backend: modal
    # The systemd user unit the node runs as. A second node on the same machine (e.g. a
    # Modal node beside a container node) needs its own.
    service_name: str = "bmb-eval"
    tasks: dict[str, TaskBinding] = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str = "config.yaml") -> "Config":
        path = Path(path).resolve()
        if not path.exists():
            raise SystemExit(f"{path} not found. Run `bmb-eval init` first.")
        raw = yaml.safe_load(path.read_text()) or {}
        base = path.parent

        def rel(p: str) -> Path:
            q = Path(p).expanduser()
            return q if q.is_absolute() else (base / q).resolve()

        sb = raw.get("sandbox", {})
        al = raw.get("alerts", {})
        cfg = cls(
            path=path,
            node_name=raw.get("node_name", "bmb-eval-node"),
            state_dir=rel(raw.get("state_dir", "state")),
            backend=sb.get("backend", "docker"),
            image=sb.get("image", "bmb-eval-sandbox:latest"),
            runtime=sb.get("runtime"),
            allow_unisolated=bool(sb.get("i_understand_local_backend_is_not_isolated", False)),
            webhook_url=al.get("webhook_url"),
            infra_failure_threshold=int(al.get("infra_failure_threshold", 3)),
            retain_bundles=bool(raw.get("retain_bundles", False)),
            cache_gb=int(raw.get("cache_gb", 300)),
            max_parallel=max(1, int(raw.get("max_parallel", 1))),
            hub=dict(raw.get("hub") or {}),
            blob_dir=rel(raw["blob_dir"]) if raw.get("blob_dir") else None,
            service_name=str(raw.get("service_name") or "bmb-eval"),
            raw=raw,
        )
        if not cfg.service_name.replace("-", "").replace("_", "").isalnum():
            raise SystemExit("service_name may hold only letters, digits, - and _")
        cfg.blob_dir = cfg.blob_dir or cfg.state_dir / "blobs"
        for tid, t in (raw.get("tasks") or {}).items():
            d = rel(t["dir"])
            contract = Contract.load(d / "contract.yaml")
            if contract.id != tid:
                raise SystemExit(f"task {tid!r}: contract id is {contract.id!r}")
            cfg.tasks[tid] = TaskBinding(tid, d, rel(t["inputs"]), rel(t["answers"]), contract,
                                         rel(t["inputs_dir"]) if t.get("inputs_dir") else None,
                                         rel(t["answers_dir"]) if t.get("answers_dir") else None,
                                         rel(t["test_inputs_dir"]) if t.get("test_inputs_dir") else None)
        if cfg.backend not in ("docker", "podman", "bwrap", "local", "modal"):
            raise SystemExit("sandbox.backend must be docker, podman, bwrap, modal or local")
        if cfg.backend == "modal":
            cfg.modal = ModalSettings.parse(raw.get("modal"), cfg.node_name)
            # Evaluations run side by side, each in its own Modal container.
            cfg.max_parallel = max(1, cfg.modal.max_jobs)
        if cfg.backend == "local" and not cfg.allow_unisolated:
            raise SystemExit("sandbox.backend local is NOT isolated; set "
                             "sandbox.i_understand_local_backend_is_not_isolated: true "
                             "only for tests on synthetic data")
        cfg.state_dir.mkdir(parents=True, exist_ok=True)
        return cfg

    @property
    def sealed(self) -> bool:
        """The confidential-computing node (Confidential Space, one H100)."""
        return self.backend == "bwrap"

    def save_raw(self) -> None:
        save_yaml(self.path, self.raw)


def save_yaml(path: Path, raw: dict) -> None:
    """Write config.yaml readable only by the operator (it holds the hub secret)."""
    path.write_text("# bmb-eval node configuration. Provider-local; never commit.\n"
                    + yaml.safe_dump(raw, sort_keys=False))
    os.chmod(path, 0o600)
