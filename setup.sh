#!/usr/bin/env bash
# Installs bmb-eval into ./.venv and prepares the sandbox image. Called by
# install.sh; also works from a git checkout. Safe to re-run (updates).
set -euo pipefail
cd "$(dirname "$0")"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

say "Installing"
PY=""
for c in python3.13 python3.12 python3.11 python3; do
  if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
    PY="$c"; break
  fi
done
if command -v uv >/dev/null; then
  [ -d .venv ] || uv venv -q --python 3.12 .venv
  uv pip install -q --python .venv/bin/python -e .
elif [ -n "$PY" ]; then
  [ -d .venv ] || "$PY" -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -e .
else
  # No Python 3.11+ (e.g. Ubuntu 22.04, RHEL 9): uv brings its own, into ~/.local.
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
  export PATH="$HOME/.local/bin:$PATH"
  uv venv -q --python 3.12 .venv
  uv pip install -q --python .venv/bin/python -e .
fi

[ -f config.yaml ] || ./bmb-eval init --engine "${BMB_ENGINE:-auto}"

say "Preparing the sandbox"
./bmb-eval build-image ${BMB_HUB:+--from "$BMB_HUB/sandbox-image.tar"}

# On an update, re-test and re-approve the already-connected task(s).
# A first install skips this: `connect` runs the self-test once, on your data.
if [ -z "${BMB_SKIP_SELFTEST:-}" ]; then
  ./bmb-eval selftest && ./bmb-eval approve >/dev/null
fi
