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


def _last_hash(path: Path) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return GENESIS
    with path.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 65536))
        last = f.read().splitlines()[-1]
    return json.loads(last)["hash"]


def record(state_dir: Path, event: str, **fields) -> dict:
    path = state_dir / "audit.jsonl"
    with _lock:
        prev = _last_hash(path)
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
