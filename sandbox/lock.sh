#!/usr/bin/env bash
# Re-pin the sandbox image's Python packages. Resolves sandbox/requirements.txt for the
# image's platform (Linux x86_64, Python 3.12, what the hub builds on) and writes
# sandbox/requirements.lock: every package at one version, with a hash for each wheel.
# Run after editing requirements.txt, or to take updates; then commit both files.
set -euo pipefail
cd "$(dirname "$0")/.."
uv pip compile sandbox/requirements.txt --python-version 3.12 \
  --python-platform x86_64-manylinux_2_28 --generate-hashes --upgrade \
  -o sandbox/requirements.lock
