"""The NVIDIA device spec (CDI) the sandbox can use with Podman.

Podman gives a container the GPU through a CDI spec that `nvidia-ctk cdi
generate` writes. Two things in a current toolkit's spec (1.18+) break our
sandbox:

- Its format: CDI 0.7 adds `additionalGids`, which Podman before 5 can't parse
  (Ubuntu 24.04 ships 4.9), so every GPU container fails with "unresolvable
  CDI devices".
- Two of its hooks write into the container's filesystem (library symlinks
  such as libcuda.so.1, and the dynamic linker's cache). The sandbox's root
  filesystem is read-only, so those hooks fail and the container doesn't start.

So this writes a spec without those hooks, with each symlink the hook would
have made provided as a read-only bind mount of the file it points to, and in
CDI 0.6 format. The toolkit's refresh service would put its own spec back (in
/var/run/cdi, which takes precedence), and a udev rule starts it whenever the
driver loads, so disabling it isn't enough: it is masked. Setup re-runs this
after a driver upgrade (re-running the install command does).

Run as root on the provider's machine: `python -m bmb_eval.gpu_spec`.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

WRITING_HOOKS = ("create-symlinks", "update-ldcache")
OUT = Path("/etc/cdi/nvidia.json")
STALE = [Path("/etc/cdi/nvidia.yaml"), Path("/var/run/cdi/nvidia.yaml"), Path("/var/run/cdi/nvidia.json")]
REFRESH_UNITS = ["nvidia-cdi-refresh.path", "nvidia-cdi-refresh.service"]


def _hook_name(hook: dict) -> str | None:
    args = hook.get("args") or []
    return args[1] if len(args) > 1 else None


def _links(hook: dict) -> list[tuple[str, str]]:
    """(target, link) pairs from a create-symlinks hook's `--link target::link` arguments."""
    args = hook.get("args") or []
    out = []
    for i, a in enumerate(args):
        v = args[i + 1] if a == "--link" and i + 1 < len(args) else (a[7:] if a.startswith("--link=") else None)
        if v and "::" in v:
            target, link = v.split("::", 1)
            out.append((target, link))
    return out


def _fix_edits(edits: dict, host_root: Path) -> None:
    edits.pop("additionalGids", None)
    kept, mounts = [], list(edits.get("mounts") or [])
    have = {m.get("containerPath") for m in mounts}
    root = host_root.resolve()
    for hook in edits.get("hooks") or []:
        name = _hook_name(hook)
        if name not in WRITING_HOOKS:
            kept.append(hook)
            continue
        for target, link in (_links(hook) if name == "create-symlinks" else []):
            if link.startswith("/dev/") or link in have:
                continue        # /dev/dri/by-path names: display devices, not needed for CUDA
            real = Path(os.path.realpath(root / Path(link).parent.relative_to("/") / target))
            if not real.is_file() or root not in real.parents:
                continue
            mounts.append({"hostPath": "/" + str(real.relative_to(root)), "containerPath": link,
                           "options": ["ro", "nosuid", "nodev", "bind"]})
            have.add(link)
    if "hooks" in edits:
        edits["hooks"] = kept
    if mounts:
        edits["mounts"] = mounts


def readonly_spec(spec: dict, host_root: Path = Path("/")) -> dict:
    """The spec `nvidia-ctk cdi generate` wrote, made usable by the read-only sandbox (see above)."""
    spec = json.loads(json.dumps(spec))
    _fix_edits(spec.setdefault("containerEdits", {}), host_root)
    for dev in spec.get("devices") or []:
        _fix_edits(dev.setdefault("containerEdits", {}), host_root)
    spec["cdiVersion"] = "0.6.0"
    return spec


def main() -> None:
    if os.geteuid() != 0:
        raise SystemExit("run as root (it writes /etc/cdi)")
    r = subprocess.run(["nvidia-ctk", "cdi", "generate", "--format", "json"], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"nvidia-ctk cdi generate failed: {r.stderr.strip()[-300:]}")
    spec = readonly_spec(json.loads(r.stdout))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(spec, indent=2) + "\n")
    tmp.chmod(0o644)
    tmp.replace(OUT)
    for p in STALE:
        p.unlink(missing_ok=True)
    # Masked, not disabled: the toolkit's udev rule starts the service on every driver load.
    subprocess.run(["systemctl", "stop", *REFRESH_UNITS], capture_output=True)
    subprocess.run(["systemctl", "mask", *REFRESH_UNITS], capture_output=True)
    edits = [spec["containerEdits"], *(d.get("containerEdits", {}) for d in spec.get("devices") or [])]
    print(f"wrote {OUT}: {len(spec.get('devices') or [])} devices, "
          f"{sum(len(e.get('mounts') or []) for e in edits)} mounts", file=sys.stderr)


if __name__ == "__main__":
    main()
