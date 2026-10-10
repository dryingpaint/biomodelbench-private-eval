"""Model bundles as content-addressed files.

A bundle travels as a manifest: every file's relative path, sha256 and size.
Files move only if the other side doesn't already have them, so a 20-50 GB
base model is uploaded once to the hub and downloaded once per provider;
later submissions that reuse it send only what changed (e.g. a LoRA adapter
or a new head).

The provider's node keeps a cache of verified files (state/blobs/<sha256>,
read-only) and assembles each run's bundle from it with hard links.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath

from .errors import EvalError

SHA_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_FILES = 20_000
CHUNK = 8 << 20


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def manifest_for(bundle_dir: Path) -> dict:
    """Agent side: describe a bundle directory."""
    bundle_dir = Path(bundle_dir)
    files = []
    for p in sorted(bundle_dir.rglob("*")):
        if "__pycache__" in p.parts or p.suffix == ".pyc" or p.is_dir():
            continue
        if p.is_symlink() or not p.is_file():
            raise ValueError(f"{p}: bundles may contain only regular files")
        files.append({"path": p.relative_to(bundle_dir).as_posix(),
                      "sha256": file_sha256(p), "size": p.stat().st_size})
    return {"files": files}


def manifest_id(m: dict) -> str:
    return hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()


def check_manifest(m, max_bytes: int) -> list[dict]:
    """Validate an untrusted manifest; return its file entries."""
    files = m.get("files") if isinstance(m, dict) else None
    if not isinstance(files, list) or not files or len(files) > MAX_FILES:
        raise EvalError("bundle_invalid", internal="manifest files")
    seen, total = set(), 0
    for f in files:
        if not isinstance(f, dict) or set(f) != {"path", "sha256", "size"}:
            raise EvalError("bundle_invalid", internal="manifest entry")
        p = PurePosixPath(str(f["path"]))
        if p.is_absolute() or ".." in p.parts or not p.parts or str(p) in seen:
            raise EvalError("bundle_unsafe", internal=f"path {f['path']!r}")
        if not SHA_RE.match(str(f["sha256"])) or not isinstance(f["size"], int) or f["size"] < 0:
            raise EvalError("bundle_invalid", internal="manifest hash/size")
        seen.add(str(p))
        total += f["size"]
    if total > max_bytes:
        raise EvalError("bundle_too_large", internal=f"{total} bytes")
    if "predict.py" not in seen:
        raise EvalError("bundle_invalid", internal="no predict.py at bundle root")
    return files


class BlobCache:
    """Verified, read-only files keyed by sha256, evicted least-recently-used."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, sha: str) -> Path:
        assert SHA_RE.match(sha)
        return self.root / sha

    def has(self, sha: str, size: int) -> bool:
        p = self.path(sha)
        return p.is_file() and p.stat().st_size == size

    def put(self, sha: str, size: int, read) -> None:
        """Store bytes from read(n) after checking size and hash."""
        h, n = hashlib.sha256(), 0
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=".incoming-")
        try:
            with os.fdopen(fd, "wb") as out:
                # Read exactly `size` bytes: reading further would block on a
                # kept-alive HTTP connection. The hash catches wrong content.
                while n < size and (chunk := read(min(CHUNK, size - n))):
                    n += len(chunk)
                    h.update(chunk)
                    out.write(chunk)
            if n != size or h.hexdigest() != sha:
                raise EvalError("bundle_invalid", internal=f"{sha}: size/hash mismatch")
            os.chmod(tmp, 0o444)
            os.replace(tmp, self.path(sha))
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def assemble(self, files: list[dict], dest: Path) -> Path:
        """Hard-link a bundle's files from the cache into dest."""
        dest.mkdir(parents=True, exist_ok=True)
        for f in files:
            src, dst = self.path(f["sha256"]), dest / f["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(src, dst)
            except OSError:
                shutil.copyfile(src, dst)
            os.utime(src)  # mark recently used
        return dest

    def evict(self, max_bytes: int) -> int:
        """Delete least-recently-used files until the cache fits; returns bytes freed."""
        entries = sorted((p.stat().st_mtime, p.stat().st_size, p)
                         for p in self.root.iterdir() if SHA_RE.match(p.name))
        total = sum(e[1] for e in entries)
        freed = 0
        for _, size, p in entries:
            if total - freed <= max_bytes:
                break
            p.unlink(missing_ok=True)
            freed += size
        return freed
