"""Append-only, hash-chained audit log (state/audit.jsonl).

Every run, export, alert, approval and stop/resume is recorded. Each line
carries the hash of the previous line, so `bmb-eval audit verify` detects
edits or deletions.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

_lock = threading.Lock()
GENESIS = "0" * 64


def _digest(prev: str, body: dict) -> str:
    return hashlib.sha256((prev + json.dumps(body, sort_keys=True)).encode()).hexdigest()


def _last_hash(path: Path) -> tuple[str, int]:
    """The last entry's hash, and how many bytes of a cut-off last entry were set aside.

    A write cut off part-way (a full disk on 2026-11-03) left a last line that wasn't JSON,
    and every start then crashed reading it, for hours. That tail is moved to
    audit.jsonl.partial-<time> and the file truncated after the last whole entry, so the
    chain goes on from there (`verify` still checks every kept line)."""
    if not path.exists() or path.stat().st_size == 0:
        return GENESIS, 0
    with path.open("rb+") as f:
        f.seek(0, 2)
        size = f.tell()
        start = max(0, size - 65536)
        f.seek(start)
        tail = f.read()
        lines = tail.splitlines(keepends=True)
        last = lines[-1] if lines else b""
        try:
            return json.loads(last)["hash"], 0
        except (ValueError, KeyError, TypeError):
            pass
        if last.endswith(b"\n"):
            # A whole line that isn't an entry: an edit, not a cut-off write. Leave it to verify.
            raise ValueError(f"{path}: its last line isn't an audit entry")
        (path.parent / f"{path.name}.partial-{int(time.time())}").write_bytes(last)
        f.truncate(size - len(last))
    return _last_hash(path)[0], len(last)


def record(state_dir: Path, event: str, **fields) -> dict:
    path = state_dir / "audit.jsonl"
    with _lock:
        prev, dropped = _last_hash(path)
        if dropped:
            body = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "event": "audit_tail_repaired", "bytes_set_aside": dropped}
            with path.open("a") as f:
                f.write(json.dumps({**body, "prev": prev, "hash": _digest(prev, body)}) + "\n")
            prev = _digest(prev, body)
        body = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": event, **fields}
        line = {**body, "prev": prev, "hash": _digest(prev, body)}
        with path.open("a") as f:
            f.write(json.dumps(line) + "\n")
    return line


def verify(state_dir: Path) -> tuple[bool, int, str]:
    path = state_dir / "audit.jsonl"
    if not path.exists():
        return True, 0, "empty"
    prev = GENESIS
    n = 0
    for n, raw in enumerate(path.read_text().splitlines(), 1):
        line = json.loads(raw)
        body = {k: v for k, v in line.items() if k not in ("prev", "hash")}
        if line["prev"] != prev or line["hash"] != _digest(prev, body):
            return False, n, f"chain broken at line {n}"
        prev = line["hash"]
    return True, n, "ok"
