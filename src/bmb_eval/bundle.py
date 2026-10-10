"""Safe unpacking of submitted model bundles.

A bundle is a .tar.gz containing at least predict.py, either at the root
or under a single top-level directory. Optional: weights/ (anything),
wheels/ (*.whl installed offline inside the sandbox), manifest.json.

Only regular files and directories are accepted. Links, devices, absolute
paths and `..` components are rejected as bundle_unsafe.
"""
from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path, PurePosixPath

from .errors import EvalError

MAX_FILES = 20_000


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def unpack(data: bytes, dest: Path, max_bytes: int) -> Path:
    """Extract bundle bytes into dest and return the directory containing predict.py."""
    if len(data) > max_bytes:
        raise EvalError("bundle_too_large")
    try:
        tar = tarfile.open(fileobj=io.BytesIO(data), mode="r:*")
        members = tar.getmembers()
    except Exception as e:
        raise EvalError("bundle_invalid", internal=f"open: {type(e).__name__}: {e}")
    if len(members) > MAX_FILES:
        raise EvalError("bundle_invalid", internal=f"{len(members)} members")

    total = 0
    for m in members:
        p = PurePosixPath(m.name)
        if p.is_absolute() or ".." in p.parts:
            raise EvalError("bundle_unsafe", internal=f"path {m.name!r}")
        if not (m.isfile() or m.isdir()):
            raise EvalError("bundle_unsafe", internal=f"member type {m.type!r} at {m.name!r}")
        total += m.size
    if total > max_bytes * 4:
        raise EvalError("bundle_too_large", internal=f"uncompressed {total}")

    dest.mkdir(parents=True, exist_ok=True)
    try:
        # The "data" filter re-checks paths/links and strips setuid bits/ownership.
        tar.extractall(dest, filter="data")
    except Exception as e:
        raise EvalError("bundle_unsafe", internal=f"extract: {type(e).__name__}: {e}")

    if (dest / "predict.py").is_file():
        return dest
    subdirs = [p for p in dest.iterdir() if p.is_dir()]
    if len(subdirs) == 1 and (subdirs[0] / "predict.py").is_file():
        return subdirs[0]
    raise EvalError("bundle_invalid", internal="no predict.py at root or in single subdir")


def pack_dir(src: Path) -> bytes:
    """Tar a bundle directory (client-side helper)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in sorted(src.rglob("*")):
            if "__pycache__" in p.parts:
                continue
            tar.add(p, arcname=str(p.relative_to(src)), recursive=False)
    return buf.getvalue()
