#!/usr/bin/env bash
# Pin every package of the Modal sandbox image, with hashes (see requirements.txt).
set -euo pipefail
cd "$(dirname "$0")"
uv pip compile requirements.txt --python-version 3.11 --python-platform x86_64-manylinux_2_28 \
  --generate-hashes --no-header -o requirements.lock
