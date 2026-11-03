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
import os
from pathlib import Path

from .config import PKG, Config
from . import sandbox


_hash_cache: dict[tuple, str] = {}
_seen: set[tuple] = set()          # keys used by the fingerprint being computed
HASH_CACHE = "hash_cache.json"     # in the node's state dir


def _file_hash(p: Path) -> str:
    st = p.stat()
    k = (str(p.resolve()), st.st_mtime_ns, st.st_size)
    _seen.add(k)
    if k not in _hash_cache:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _hash_cache[k] = h.hexdigest()
    return _hash_cache[k]


def _load_hashes(state_dir: Path) -> None:
    """The per-file hashes of earlier processes, by (path, mtime, size) as in memory. Without
    them every start, `status`, `approve` and `connect` re-read every data file: 2 min 43 s of
    CPU on the public Modal node with 93 tasks (2026-11-03), all of it before a restarted node
    took any work."""
    try:
        for row in json.loads((state_dir / HASH_CACHE).read_text()):
            _hash_cache.setdefault((row[0], row[1], row[2]), row[3])
    except (OSError, ValueError, TypeError, IndexError):
        pass


def _save_hashes(state_dir: Path) -> None:
    """Keep the hashes of the files this fingerprint read (so the file stays bounded);
    written whole and renamed into place, as the node and a CLI command may both save."""
    rows = [[*k, _hash_cache[k]] for k in sorted(_seen) if k in _hash_cache]
    try:
        tmp = state_dir / f".{HASH_CACHE}.{os.getpid()}"
        tmp.write_text(json.dumps(rows))
        tmp.replace(state_dir / HASH_CACHE)
    except OSError:
        pass


def _dir_hash(d: Path, pattern: str = "*") -> str:
    h = hashlib.sha256()
    for p in sorted(d.rglob(pattern)):
        if p.is_file() and "__pycache__" not in p.parts:
            h.update(str(p.relative_to(d)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def _tree_hash(d: Path) -> str:
    """Hash of a data folder: per-file hashes are cached by (path, mtime, size),
    so re-checking a large image folder costs one stat per file."""
    if not d.exists():
        return "missing"
    h = hashlib.sha256()
    for p in sorted(d.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(d)).encode())
            h.update(_file_hash(p).encode())
    return h.hexdigest()


def fingerprint_tasks(cfg: Config) -> dict:
    """Each task's contract, scorer and data hashes (also what a Modal node's MANIFEST records)."""
    tasks = {}
    for tid, t in sorted(cfg.tasks.items()):
        tasks[tid] = {
            "contract": _file_hash(t.dir / "contract.yaml"),
            "task_dir": _dir_hash(t.dir),  # scorer and anything it imports
            "inputs_dir": _tree_hash(t.inputs_dir) if t.inputs_dir else None,
            "answers_dir": _tree_hash(t.answers_dir) if t.answers_dir else None,
            "test_inputs_dir": _tree_hash(t.test_inputs_dir) if t.test_inputs_dir else None,
            "inputs": _file_hash(t.inputs) if t.inputs.exists() else "missing",
            "answers": _file_hash(t.answers) if t.answers.exists() else "missing",
        }
    return tasks


def fingerprint(cfg: Config) -> dict:
    _load_hashes(cfg.state_dir)
    _seen.clear()
    sb = {"backend": cfg.backend, "image_id": sandbox.make(cfg).image_id(), "runtime": cfg.runtime}
    if cfg.backend == "modal":
        # Where the data lives, and exactly what was uploaded there (modal_sync's manifest).
        from .modal_sync import local_manifest_hash
        sb["modal"] = {"environment": cfg.modal.environment, "app": cfg.modal.app,
                       "manifest": local_manifest_hash(cfg)}
    if cfg.batch_families:
        # A batched model sees several tasks' inputs at once: turning it on is approved like
        # any other change to what a model sees. Absent when off, so existing locks still match.
        sb["batch_families"] = True
    out = {
        "node_code": _dir_hash(PKG, "*.py"),
        "sandbox": sb,
        "tasks": fingerprint_tasks(cfg),
    }
    _save_hashes(cfg.state_dir)
    return out


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
