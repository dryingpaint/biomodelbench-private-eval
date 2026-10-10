"""Approved-configuration lock.

`bmb-eval approve` records a fingerprint of everything that decides what
can leave the node: the node code, each task's contract and scorer, the
sandbox image and backend settings, and the private data files. While
the live fingerprint differs from approved.lock.json, the node refuses
to run submissions and alerts the operator.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import PKG, Config
from . import sandbox


_hash_cache: dict[tuple, str] = {}


def _file_hash(p: Path) -> str:
    st = p.stat()
    k = (str(p.resolve()), st.st_mtime_ns, st.st_size)
    if k not in _hash_cache:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _hash_cache[k] = h.hexdigest()
    return _hash_cache[k]


def _dir_hash(d: Path, pattern: str = "*") -> str:
    h = hashlib.sha256()
    for p in sorted(d.rglob(pattern)):
        if p.is_file() and "__pycache__" not in p.parts:
            h.update(str(p.relative_to(d)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def fingerprint(cfg: Config) -> dict:
    tasks = {}
    for tid, t in sorted(cfg.tasks.items()):
        tasks[tid] = {
            "contract": _file_hash(t.dir / "contract.yaml"),
            "task_dir": _dir_hash(t.dir),  # scorer and anything it imports
            "inputs": _file_hash(t.inputs) if t.inputs.exists() else "missing",
            "answers": _file_hash(t.answers) if t.answers.exists() else "missing",
        }
    return {
        "node_code": _dir_hash(PKG, "*.py"),
        "sandbox": {
            "backend": cfg.backend,
            "image_id": sandbox.make(cfg).image_id(),
            "runtime": cfg.runtime,
        },
        "tasks": tasks,
    }


def lock_path(cfg: Config) -> Path:
    return cfg.path.parent / "approved.lock.json"


def diff(cfg: Config) -> list[str]:
    """Return human-readable differences between live config and the lock."""
    p = lock_path(cfg)
    if not p.exists():
        return ["no approved.lock.json (run `bmb-eval approve`)"]
    approved = json.loads(p.read_text())
    live = fingerprint(cfg)
    out = []

    def walk(a, b, prefix=""):
        for k in sorted(set(a) | set(b)):
            av, bv = a.get(k), b.get(k)
            if isinstance(av, dict) and isinstance(bv, dict):
                walk(av, bv, f"{prefix}{k}.")
            elif av != bv:
                out.append(f"{prefix}{k}: approved={_short(av)} live={_short(bv)}")
    walk(approved, live)
    return out


def approve(cfg: Config) -> dict:
    fp = fingerprint(cfg)
    lock_path(cfg).write_text(json.dumps(fp, indent=2) + "\n")
    return fp


def _short(v):
    return v[:12] if isinstance(v, str) and len(v) == 64 else v
