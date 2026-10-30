"""Trusted bundle stager inside a Modal sandbox (no submitted code runs here).

Copies one job's model files from the node's content-addressed blob volume
(mounted read-only at <blobs>) into that job's own bundle volume (<out>), by the
manifest the controller wrote, then flushes the volume. Predict then mounts the
job's volume read-only and never sees the blob volume, so a model can't read
other submissions' files.

Usage: python -m bmb_eval.modal_stage <stage.json>
stage.json: {"blobs": dir, "out": dir, "files": [{"path", "sha256", "size", "executable"?}]}
Prints {"ok": true} or {"missing": [sha256, ...]} (blobs the controller must upload first).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath


def main(spec_path: str) -> int:
    spec = json.loads(Path(spec_path).read_text())
    blobs, out = Path(spec["blobs"]), Path(spec["out"])
    files = spec["files"]
    missing = sorted({f["sha256"] for f in files
                      if not ((blobs / f["sha256"]).is_file()
                              and (blobs / f["sha256"]).stat().st_size == f["size"])})
    if missing:
        print(json.dumps({"missing": missing}), flush=True)
        return 0
    for f in files:
        rel = PurePosixPath(f["path"])
        if rel.is_absolute() or ".." in rel.parts:      # checked on the node too
            raise SystemExit(f"unsafe path {f['path']!r}")
        dst = out.joinpath(*rel.parts)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not (dst.is_file() and dst.stat().st_size == f["size"]):
            shutil.copyfile(blobs / f["sha256"], dst)
        os.chmod(dst, 0o555 if f.get("executable") else 0o444)
    # Volumes mounted in a sandbox are persisted on sync (and when the sandbox ends).
    subprocess.run(["sync", str(out)], check=False)
    print(json.dumps({"ok": True, "files": len(files)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
