"""Which release this code is.

A release is one git tag (deploy/release.sh) naming one commit, from which the
hub builds the three things providers download: install.sh, bmb-eval.tar.gz and
sandbox-image.tar. The hub publishes the tag, the commit and the sha256 of each
download together at GET /release (and /SHA256SUMS), so "the same version" means
one checkable thing for every party.

A git checkout (the hub, developers) knows its release from git. The code
providers install is a tarball without git, so the hub writes release.json
into it when it builds the tarball.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

UNKNOWN = {"version": "unknown", "commit": "unknown"}


def from_git(root: Path) -> dict:
    """{"version": tag (or "untagged"), "commit": full sha} of a checkout; {} if it isn't one."""
    try:
        describe = _git(root, "describe", "--tags", "--always", "--dirty")
        commit = _git(root, "rev-parse", "HEAD")
    except (OSError, subprocess.SubprocessError):
        return {}
    if describe is None or commit is None:
        return {}
    base, dirty = (describe[:-6], True) if describe.endswith("-dirty") else (describe, False)
    if commit.startswith(base):          # no tag reachable: describe fell back to the commit id
        describe = "untagged" + ("-dirty" if dirty else "")
    return {"version": describe, "commit": commit}


def info(root: Path) -> dict:
    """{"version", "commit"} of the code at root: release.json (a tarball install),
    else git (a checkout), else unknown."""
    f = root / "release.json"
    if f.is_file():
        try:
            d = json.loads(f.read_text())
            if isinstance(d, dict) and d.get("version") and d.get("commit"):
                return {"version": str(d["version"]), "commit": str(d["commit"])}
        except ValueError:
            pass
    return from_git(root) or dict(UNKNOWN)


def describe(root: Path) -> str:
    """One line, e.g. "v0.2.0 (467bff1)"."""
    r = info(root)
    return f"{r['version']} ({r['commit'][:7]})"


def _git(root: Path, *args: str) -> str | None:
    r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=10)
    return r.stdout.strip() if r.returncode == 0 else None
