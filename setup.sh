#!/usr/bin/env bash
# Installs bmb-eval into ./.venv and prepares the sandbox image. Called by
# install.sh; also works from a git checkout. Safe to re-run (updates).
set -euo pipefail
cd "$(dirname "$0")"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

say "Installing"
# uv installs without sudo and brings its own Python, so this works the same on
# every distribution (stock Ubuntu's python3 can't create virtualenvs).
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
[ -x .venv/bin/python ] || { rm -rf .venv; uv venv -q --python 3.12 .venv; }
uv pip install -q --python .venv/bin/python -e .

[ -f config.yaml ] || ./bmb-eval init --engine "${BMB_ENGINE:-auto}"

say "Preparing the sandbox"
./bmb-eval build-image ${BMB_HUB:+--from "$BMB_HUB/sandbox-image.tar"}

# On an update, re-test and re-approve the already-connected task(s).
# A first install skips this: `connect` runs the self-test once, on your data.
if [ -z "${BMB_SKIP_SELFTEST:-}" ]; then
  ./bmb-eval selftest && ./bmb-eval approve >/dev/null
fi
