"""Modal backend: each stage runs in its own Modal Sandbox (gVisor) on Modal's GPUs,
while the node loop (queue, quotas, audit log, approval lock) stays on the node's VM.

    stage sandbox   block_network=True, no secrets, gpu "H100:<n>", cpu/memory/timeout
                    from the contract; the trusted launcher (modal_entry) runs the
                    stage as uid 65534 with no capabilities and pid/file/memory limits
    data            Modal Volumes in the evaluation environment, mounted read-only,
                    one per task and kind, so a stage can be given exactly its own:
                      <p>-<task>-val / -test / -in    inputs the model sees, one per half
                      <p>-<task>-ans                  answers (score stage only)
                      <p>-<task>-task                 contract.yaml + score.py
                      <p>-sh-<sha16>                  one large shared file (the genome),
                                                      stored once, linked into inputs
                      <p>-job-<id>                    one job's model files (predict only),
                                                      deleted after the stage
                      <p>-blobs                       model files by sha256; only the trusted
                                                      stager mounts it
                      <p>-meta                        MANIFEST.json (what `modal-sync` uploaded)
    pipes           stdin is a file written into the sandbox; stdout is byte-capped in
                    the sandbox and read back after a size check (modal_entry)

Mount.host paths map to (volume, path) through state/modal/mounts.json, which
`bmb-eval modal-sync` writes (modal_sync.py). Each stage's mounts are checked
against an allowlist before anything starts: predict never gets answers, the
task folder, or the other half of the data.

A Modal failure (an API error, the sandbox gone without its launcher's report)
is retried here, and after that raised as SandboxLost: the node then hands the
job back to the hub without counting it. The model's own outcomes (exit codes,
timeouts, too much output, out of memory) are never retried.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import audit
from .contract import Limits
from .sandbox import LOG_CAP, Mount, Result, SandboxLost, _bytes

RUN = "/run/bmb"                  # launcher spec, stdin, stdout and stderr files in the sandbox
CONTRACT = "/contract.json"
INPUTS = "/data/inputs"
MOUNT_AT = {"bundle": "/bundle", "inputs": INPUTS + "/.vol", "inputs_validation": INPUTS + "/.vol",
            "inputs_test": INPUTS + "/.vol", "task": "/task", "answers": "/data/answers"}
SHARED_AT = INPUTS + "/.shared"
INPUT_KINDS = {"inputs", "inputs_validation", "inputs_test"}
# What each stage may mount. Anything else refuses the stage before it starts.
ALLOWED = {"predict": {"bundle", "contract"} | INPUT_KINDS,
           "validate": {"contract"} | INPUT_KINDS,
           "score": {"task", "answers"} | INPUT_KINDS}
COLD_START_S = 300               # sandbox timeout = stage timeout + this
WATCHDOG_GRACE_S = 120           # the launcher enforces the stage timeout; this is the backstop
OOM_CODE = 137                   # what Modal reports for a sandbox killed for memory
EXIT_POLLS = 20                  # polls (0.5 s apart) for a vanished sandbox's exit code
RETRY_BACKOFF_S = (2, 6, 18, 30)  # between attempts of a stage after a Modal failure
# Modal has no GPU (or container) free for this workspace: the stage waits for one, polling,
# rather than failing. Without the wait, a node allowed to run every queued job at once
# turned capacity refusals into hand-backs, half-hour pauses and, on a third, a counted
# infra_error (2026-10-31).
CAPACITY_WAIT_S = 6 * 3600
CAPACITY_POLL_S = (15, 30, 60, 120)


def _no_capacity(e: Exception) -> bool:
    return "Insufficient capacity" in str(e)
MODAL_IMAGE_FILES = ("sandbox/modal/Dockerfile", "sandbox/modal/requirements.lock")


class _Infra(Exception):
    """A failure of Modal, not of the model: retried."""


# ----------------------------------------------------------------- helpers

def import_modal(settings=None):
    if settings is not None and settings.profile:
        os.environ.setdefault("MODAL_PROFILE", settings.profile)
    import modal
    return modal


def vol_name(prefix: str, *parts: str) -> str:
    """A valid Modal volume name (letters, digits, - _ .; under 64 characters)."""
    raw = "-".join([prefix, *parts])
    name = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in raw)
    return name if len(name) <= 63 else name[:46] + "-" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def modal_dir(cfg) -> Path:
    d = cfg.state_dir / "modal"
    d.mkdir(parents=True, exist_ok=True)
    return d


def image_record(cfg) -> dict:
    try:
        return json.loads((modal_dir(cfg) / "image.json").read_text())
    except (OSError, ValueError):
        return {}


def image_stamp(root: Path) -> str | None:
    """Digest of what goes into the Modal image: its Dockerfile and lock, and the package."""
    h = hashlib.sha256()
    for f in MODAL_IMAGE_FILES:
        if not (root / f).is_file():
            return None
        h.update(f.encode())
        h.update((root / f).read_bytes())
    pkg = root / "src" / "bmb_eval"
    for p in sorted(pkg.rglob("*.py")):
        if "__pycache__" not in p.parts:
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _repo_root() -> Path | None:
    return next((d for d in Path(__file__).resolve().parents
                 if (d / "sandbox" / "modal" / "Dockerfile").is_file()), None)


def stale_message(cfg) -> str | None:
    """Why this Modal node's image can't run its code, or None."""
    rec = image_record(cfg)
    if not rec.get("image_id"):
        return "the Modal sandbox image hasn't been built; run `bmb-eval modal-deploy`"
    root = _repo_root()
    want = image_stamp(root) if root else None
    if want and rec.get("stamp") != want:
        return (f"the Modal sandbox image ({rec.get('stamp')}) is older than this node's code ({want}); "
                "run `bmb-eval modal-deploy`, then `bmb-eval approve`")
    if rec.get("environment") != cfg.modal.environment:
        return (f"the Modal sandbox image was built in environment {rec.get('environment')!r}, "
                f"not {cfg.modal.environment!r}; run `bmb-eval modal-deploy`")
    return None


def _context_filter(root: Path):
    """Image build context: only the image's own files and the package. The install
    directory also holds private data and the hub secret; none of it may go to Modal."""
    keep = [root / "sandbox" / "modal", root / "src" / "bmb_eval"]

    def ignore(p: Path) -> bool:
        p = p if p.is_absolute() else root / p
        if "__pycache__" in p.parts or p.suffix == ".pyc":
            return True
        return not any(p == k or k in p.parents or p in k.parents for k in keep)
    return ignore


def deploy(cfg, modal_mod=None, root: Path | None = None) -> dict:
    """Build the Modal image in the evaluation environment and record its id (it
    enters this node's approval fingerprint)."""
    m = modal_mod or import_modal(cfg.modal)
    root = root or _repo_root()
    if root is None:
        raise SystemExit("cannot find sandbox/modal/Dockerfile")
    s = cfg.modal
    app = m.App.lookup(s.app, create_if_missing=True, environment_name=s.environment)
    img = m.Image.from_dockerfile(root / "sandbox" / "modal" / "Dockerfile", context_dir=root,
                                  ignore=_context_filter(root))
    img = img.build(app) or img
    rec = {"image_id": img.object_id, "stamp": image_stamp(root), "environment": s.environment,
           "app": s.app, "built": time.time()}
    (modal_dir(cfg) / "image.json").write_text(json.dumps(rec, indent=2) + "\n")
    audit.record(cfg.state_dir, "modal_deploy", image_id=rec["image_id"], stamp=rec["stamp"],
                 environment=s.environment)
    return rec


# ----------------------------------------------------------------- planning

@dataclass
class Plan:
    entry: str
    volumes: dict = field(default_factory=dict)    # mount path in the sandbox -> volume name
    env: dict = field(default_factory=dict)        # BMB_* -> path in the sandbox (no root prefix)
    contract: Path | None = None
    bundle: Path | None = None
    mirror: bool = False
    kinds: set = field(default_factory=set)
    task: str | None = None


def load_mounts(cfg) -> list[dict]:
    try:
        return json.loads((modal_dir(cfg) / "mounts.json").read_text())["entries"]
    except (OSError, ValueError, KeyError):
        return []


def _lookup(entries: list[dict], host: Path) -> tuple[dict, str] | None:
    best = None
    for e in entries:
        k = Path(e["host"])
        if host == k or (e.get("is_dir") and k in host.parents):
            if best is None or len(str(k)) > len(best[0]["host"]):
                best = (e, "" if host == k else host.relative_to(k).as_posix())
    return best


def _join(*parts: str) -> str:
    return "/".join(p.strip("/") for p in parts if p and p.strip("/"))


# ------------------------------------------------------------------ sandbox

class ModalSandbox:
    def __init__(self, cfg, modal_mod=None):
        self.cfg = cfg
        self.s = cfg.modal
        self._m = modal_mod
        self._app = None
        self._image = None
        self._lock = threading.Lock()

    # Modal handles, made on first use (the node can start, and be approved, offline).
    @property
    def m(self):
        if self._m is None:
            self._m = import_modal(self.s)
        return self._m

    def app(self):
        with self._lock:
            if self._app is None:
                self._app = self.m.App.lookup(self.s.app, create_if_missing=True,
                                              environment_name=self.s.environment)
            return self._app

    def image(self):
        with self._lock:
            if self._image is None:
                iid = image_record(self.cfg).get("image_id")
                if not iid:
                    raise RuntimeError("no Modal image; run `bmb-eval modal-deploy`")
                self._image = self.m.Image.from_id(iid)
            return self._image

    def volume(self, name: str, create: bool = False):
        return self.m.Volume.from_name(name, environment_name=self.s.environment, create_if_missing=create)

    def tags(self) -> dict:
        return {"bmb-eval": "1", "node": str(self.cfg.hub.get("node_id") or self.cfg.node_name)}

    # ------------------------------------------------------------- interface

    def image_id(self) -> str:
        return image_record(self.cfg).get("image_id") or "missing"

    def gpus_available(self) -> list[int]:
        """GPUs one job can have (memory, GB, each): a Modal container's maximum."""
        return [self.s.gpu_memory_gb] * self.s.max_gpus_per_job

    def kill_all(self) -> int:
        n = 0
        for sb in self.m.Sandbox.list(app_id=self.app().app_id, tags=self.tags()):
            try:
                sb.terminate()
                n += 1
            except Exception:
                pass
        return n

    def plan(self, entry: str, mounts: list[Mount]) -> Plan:
        """Where each mount comes from, checked against the stage's allowlist."""
        p = Plan(entry)
        entries = load_mounts(self.cfg)
        bundle_roots = [self.cfg.blob_dir.parent / "bundles", self.cfg.state_dir / "runs"]
        tasks = set()
        for mt in mounts:
            host = mt.host.resolve()
            if mt.env == "BMB_CONTRACT_FILE":
                p.kinds.add("contract")
                p.contract = host
                p.env[mt.env] = CONTRACT
                continue
            if mt.env == "BMB_BUNDLE" and any(r.resolve() in host.parents for r in bundle_roots):
                p.kinds.add("bundle")
                p.bundle = host
                p.env[mt.env] = MOUNT_AT["bundle"]
                continue
            hit = _lookup(entries, host)
            if hit is None:
                raise PermissionError(f"{entry}: {mt.env} is not on a synced volume (run `bmb-eval modal-sync`)")
            e, rel = hit
            p.kinds.add(e["kind"])
            tasks.add(e["task"])
            base = MOUNT_AT[e["kind"]]
            p.volumes[base] = e["volume"]
            path = _join(e.get("path", ""), rel)
            if e["kind"] in INPUT_KINDS and entry == "predict" and e.get("links"):
                for sha in sorted(set(e["links"].values())):
                    p.volumes[f"{SHARED_AT}/{sha[:16]}"] = vol_name(self.s.volume_prefix, "sh", sha[:16])
                if not path:                         # the inputs folder itself: mirrored
                    p.mirror = True
                    p.env[mt.env] = INPUTS
                    continue
            p.env[mt.env] = "/" + _join(base, path)
        extra = p.kinds - ALLOWED.get(entry, set())
        if extra:
            raise PermissionError(f"{entry} may not mount {sorted(extra)}")
        if len(p.kinds & INPUT_KINDS) > 1 or len(tasks) > 1:
            raise PermissionError(f"{entry} mounts more than one task's or half's data")
        if entry == "predict" and "bundle" not in p.kinds:
            raise PermissionError("predict has no bundle")
        p.task = next(iter(tasks), None)
        return p

    def run(self, *, name: str, entry: str, mounts: list[Mount], env: dict[str, str],
            limits: Limits, timeout_s: int, stdout_cap: int, stdin: bytes = b"",
            gpus: int = 0) -> Result:
        plan = self.plan(entry, mounts)          # PermissionError: never retried
        if gpus > self.s.max_gpus_per_job:
            raise PermissionError(f"{gpus} GPUs is more than one Modal container has")
        t0 = time.time()
        job_vol = None
        errors = []
        attempt = waits = 0
        try:
            while True:
                try:
                    if plan.bundle is not None and job_vol is None:
                        job_vol = self._stage_bundle(plan.bundle, name)
                    r = self._attempt(plan, job_vol, entry, env, limits, timeout_s, stdout_cap, stdin, gpus)
                    if waits:
                        audit.record(self.cfg.state_dir, "modal_capacity_ready", stage=name,
                                     waited_s=round(time.time() - t0))
                    r.seconds = round(time.time() - t0, 2)
                    return r
                except PermissionError:
                    raise
                except Exception as e:       # Modal's errors, connection failures, a lost sandbox
                    if _no_capacity(e) and time.time() - t0 < CAPACITY_WAIT_S:
                        if not waits:
                            audit.record(self.cfg.state_dir, "modal_capacity_wait", stage=name, gpus=gpus)
                        time.sleep(CAPACITY_POLL_S[min(waits, len(CAPACITY_POLL_S) - 1)])
                        waits += 1
                        continue
                    errors.append(f"{type(e).__name__}: {str(e)[:200]}")
                    audit.record(self.cfg.state_dir, "modal_infra_failure", stage=name, attempt=attempt + 1,
                                 error=type(e).__name__)
                    if attempt >= self.s.retries:
                        raise SandboxLost(f"{name}: " + " | ".join(errors))
                    time.sleep(RETRY_BACKOFF_S[min(attempt, len(RETRY_BACKOFF_S) - 1)])
                    attempt += 1
        finally:
            if job_vol is not None:
                self._delete_volume(job_vol)

    # -------------------------------------------------------------- one try

    def _attempt(self, plan: Plan, job_vol: str | None, entry: str, env: dict, limits: Limits,
                 timeout_s: int, stdout_cap: int, stdin: bytes, gpus: int) -> Result:
        vols = {path: self.volume(v).read_only() for path, v in plan.volumes.items()}
        if job_vol is not None:
            vols[MOUNT_AT["bundle"]] = self.volume(job_vol).read_only()
        mem_mib = (_bytes(limits.memory) >> 20) + self.s.memory_headroom_mb
        sb_timeout = timeout_s + COLD_START_S
        try:
            sb = self.m.Sandbox.create(
                "sleep", str(sb_timeout + 60), app=self.app(), image=self.image(), volumes=vols,
                gpu=f"{self.s.gpu}:{gpus}" if gpus else None,
                cpu=(float(limits.cpus), float(limits.cpus)), memory=(mem_mib, mem_mib),
                timeout=sb_timeout, block_network=True, tags=self.tags())
        except Exception as e:
            raise _Infra(f"create: {type(e).__name__}: {e}") from e
        stopped = threading.Event()          # set when the backstop watchdog terminated it

        def stop() -> None:
            stopped.set()
            try:
                sb.terminate()
            except Exception:
                pass

        watchdog = threading.Timer(timeout_s + WATCHDOG_GRACE_S + 60, stop)
        try:
            root = getattr(sb, "fake_root", "")       # tests' FakeModal runs sandboxes in a folder
            fs = sb.filesystem
            fs.make_directory(root + RUN)
            if plan.contract is not None:
                fs.write_bytes(plan.contract.read_bytes(), root + CONTRACT)
            if stdin:
                fs.write_bytes(stdin, root + RUN + "/in.bin")
            threads = str(max(1, int(limits.cpus)))
            spec = {
                "entry": entry, "run_dir": root + RUN, "timeout_s": timeout_s, "stdout_cap": stdout_cap,
                "memory_bytes": _bytes(limits.memory), "cpus": limits.cpus, "pids": limits.pids,
                "gpus": gpus, "stdin": bool(stdin), "cwd": root + "/tmp" if root else "/tmp",
                "env": {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": root + "/tmp" if root else "/tmp",
                        "TMPDIR": root + "/tmp" if root else "/tmp", "PYTHONHASHSEED": "0",
                        "PYTHONDONTWRITEBYTECODE": "1", "PIP_NO_INDEX": "1", "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1",
                        **{v: threads for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                        **{k: root + v for k, v in plan.env.items()}, **env},
                "mirror": ({"src": root + MOUNT_AT["inputs"], "dst": root + INPUTS, "shared": root + SHARED_AT}
                           if plan.mirror else None),
            }
            fs.write_bytes(json.dumps(spec).encode(), root + RUN + "/spec.json")
            watchdog.start()
            proc = sb.exec("python", "-m", "bmb_eval.modal_entry", root + RUN + "/spec.json", text=False)
            out = proc.stdout.read()
            proc.wait()
            status = json.loads(out.decode().strip().splitlines()[-1])
        except Exception as e:
            if stopped.is_set():             # the launcher never reported within the time limit
                return Result(-9, True, b"", False, "", 0.0)
            code = self._exit_code(sb)
            if code == OOM_CODE:             # Modal killed the sandbox for memory: the model's doing
                return Result(OOM_CODE, False, b"", False, "sandbox killed for memory", 0.0)
            raise _Infra(f"{type(e).__name__}: {e}; sandbox exit {code}") from e
        finally:
            watchdog.cancel()
        try:
            err = b""
            try:
                err = fs.read_bytes(root + RUN + "/err.log")[-LOG_CAP:]
            except Exception:
                pass
            data, overflow = b"", bool(status["overflow"])
            if not overflow:
                size = fs.stat(root + RUN + "/out.bin").size
                if size > stdout_cap:
                    overflow = True
                elif size:
                    data = fs.read_bytes(root + RUN + "/out.bin")
                    if len(data) != size:
                        raise _Infra(f"read {len(data)} of {size} output bytes")
            return Result(exit_code=int(status["exit_code"]), timed_out=bool(status["timed_out"]),
                          stdout=data, stdout_overflow=overflow,
                          stderr=err.decode(errors="replace"), seconds=0.0)
        except _Infra:
            raise
        except Exception as e:
            raise _Infra(f"reading results: {type(e).__name__}: {e}") from e
        finally:
            try:
                sb.terminate()
            except Exception:
                pass

    def _exit_code(self, sb) -> int | None:
        """The sandbox's own exit code once it has gone (OOM shows as 137)."""
        for _ in range(EXIT_POLLS):
            try:
                code = sb.poll()
            except Exception:
                return None
            if code is not None:
                return code
            time.sleep(0.5)
        return None

    # -------------------------------------------------------------- bundles

    def _bundle_files(self, bundle: Path) -> list[dict]:
        """The bundle's files with their sha256: files hard-linked from the node's
        model-file cache are named by it; anything else is hashed."""
        from .blobs import SHA_RE, file_sha256
        by_inode = {}
        blob_dir = self.cfg.blob_dir
        if blob_dir.is_dir():
            for b in blob_dir.iterdir():
                if SHA_RE.match(b.name):
                    st = b.stat()
                    by_inode[(st.st_dev, st.st_ino)] = b.name
        files = []
        for f in sorted(bundle.rglob("*")):
            if f.is_symlink() or not f.is_file():
                continue
            st = f.stat()
            sha = by_inode.get((st.st_dev, st.st_ino)) or file_sha256(f)
            e = {"path": f.relative_to(bundle).as_posix(), "sha256": sha, "size": st.st_size, "local": str(f)}
            if st.st_mode & 0o111:
                e["executable"] = True
            files.append(e)
        return files

    def _upload_blobs(self, files: list[dict], only: set[str] | None = None) -> None:
        vol = self.volume(vol_name(self.s.volume_prefix, "blobs"), create=True)
        have = {} if only else {e.path.strip("/"): e.size for e in vol.listdir("/")}
        todo = {f["sha256"]: f for f in files
                if (only and f["sha256"] in only) or (not only and have.get(f["sha256"]) != f["size"])}
        if todo:
            with vol.batch_upload(force=True) as up:
                for sha, f in todo.items():
                    up.put_file(f["local"], "/" + sha)
            audit.record(self.cfg.state_dir, "modal_blobs_uploaded", files=len(todo),
                         bytes=sum(f["size"] for f in todo.values()))

    def _stage_bundle(self, bundle: Path, name: str) -> str:
        """Copy one job's files into a volume of its own, in a trusted sandbox."""
        files = self._bundle_files(bundle)
        self._upload_blobs(files)
        job = vol_name(self.s.volume_prefix, "job", hashlib.sha256(name.encode()).hexdigest()[:20])
        self.volume(job, create=True).hydrate()
        manifest = [{k: f[k] for k in ("path", "sha256", "size", "executable") if k in f} for f in files]
        for attempt in range(2):
            sb = self.m.Sandbox.create(
                "sleep", "7200", app=self.app(), image=self.image(),
                volumes={"/blobs": self.volume(vol_name(self.s.volume_prefix, "blobs")).read_only(),
                         "/out": self.volume(job)},
                cpu=(2.0, 4.0), memory=(4096, 8192), timeout=7200, block_network=True, tags=self.tags())
            try:
                root = getattr(sb, "fake_root", "")
                sb.filesystem.make_directory(root + RUN)
                sb.filesystem.write_bytes(json.dumps({"blobs": root + "/blobs", "out": root + "/out",
                                                      "files": manifest}).encode(), root + RUN + "/stage.json")
                proc = sb.exec("python", "-m", "bmb_eval.modal_stage", root + RUN + "/stage.json", text=False)
                out = proc.stdout.read()
                code = proc.wait()
                status = json.loads(out.decode().strip().splitlines()[-1]) if out.strip() else {}
            finally:
                try:
                    sb.terminate()
                    sb.wait(raise_on_termination=False)    # its volume writes are committed on exit
                except Exception:
                    pass
            if status.get("ok"):
                self._wait_visible(job, manifest)
                return job
            if status.get("missing") and attempt == 0:
                self._upload_blobs(files, only=set(status["missing"]))
                continue
            raise RuntimeError(f"staging failed (exit {code}): {status}")
        raise RuntimeError("staging failed")

    def _wait_visible(self, job: str, manifest: list[dict], timeout_s: float = 120) -> None:
        """Wait until the job volume shows every staged file. Seen live: a sandbox started
        right after the stager could find the volume still empty, which would charge the
        model for a missing predict.py."""
        want = {f["path"]: f["size"] for f in manifest}
        deadline = time.time() + timeout_s
        while True:
            have = {e.path.strip("/"): e.size for e in self.volume(job).listdir("/", recursive=True)
                    if getattr(getattr(e, "type", None), "name", "FILE") == "FILE"}
            if all(have.get(p) == n for p, n in want.items()):
                return
            if time.time() > deadline:
                raise RuntimeError(f"staged files not visible on {job} after {timeout_s:.0f}s")
            time.sleep(2)

    def _delete_volume(self, name: str) -> None:
        try:
            self.m.Volume.objects.delete(name, environment_name=self.s.environment, allow_missing=True)
        except Exception as e:
            audit.record(self.cfg.state_dir, "modal_volume_delete_failed", volume=name, error=type(e).__name__)


def cleanup_jobs(cfg, modal_mod=None) -> int:
    """Delete job volumes a crash left behind (at node start)."""
    sb = ModalSandbox(cfg, modal_mod)
    prefix = vol_name(cfg.modal.volume_prefix, "job", "")
    n = 0
    for v in sb.m.Volume.objects.list(environment_name=cfg.modal.environment):
        name = getattr(v, "name", None) or ""
        if name.startswith(prefix):
            sb._delete_volume(name)
            n += 1
    return n
