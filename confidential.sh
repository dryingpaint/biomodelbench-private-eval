#!/bin/sh
# BioModelBench, sealed mode: the one command providers run.
#   curl -fsSL https://<hub>/confidential.sh | sh -s -- <CODE> --data <file> --project <gcp project>
# Runs on Linux or macOS, wherever your data and the Google Cloud CLI are.
# Checks and splits your data here, shows what you're approving, then encrypts
# it with your own key and uploads it to your own bucket. Nothing to keep running.
set -eu
HUB="${BMB_HUB:-__BMB_HUB__}"
DIR="${BMB_DIR:-$HOME/bmb-confidential}"
say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

if ! command -v gcloud >/dev/null; then
  echo "This needs the Google Cloud CLI: https://cloud.google.com/sdk/docs/install" >&2
  exit 1
fi
if ! command -v uv >/dev/null; then
  say "Installing uv (Python package manager)"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
  PATH="$HOME/.local/bin:$PATH"
fi
say "Downloading the BioModelBench tool to $DIR"
mkdir -p "$DIR/app"
chmod 700 "$DIR"
curl -fsSL "$HUB/bmb-eval.tar.gz" | tar -xz -C "$DIR/app"
[ -x "$DIR/.venv/bin/python" ] || uv venv -q --python 3.12 "$DIR/.venv"
VIRTUAL_ENV="$DIR/.venv" uv pip install -q "$DIR/app[seal]"
BMB_CALLER_CWD="$(pwd)" "$DIR/.venv/bin/bmb-eval" connect-confidential --hub "$HUB" --dir "$DIR" "$@"
