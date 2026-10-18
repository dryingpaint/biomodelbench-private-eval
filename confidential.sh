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
# Everything downloads from the hub or PyPI: networks that allow those often block GitHub,
# where uv and the Python it installs come from (the hub passes those files through).
UV_VERSION=0.13.1
case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) UV_TARGET=aarch64-apple-darwin ;;
  Darwin-x86_64) UV_TARGET=x86_64-apple-darwin ;;
  Linux-x86_64) UV_TARGET=x86_64-unknown-linux-musl ;;
  Linux-aarch64|Linux-arm64) UV_TARGET=aarch64-unknown-linux-musl ;;
  *) echo "This runs on macOS or Linux (x86_64 or ARM); this computer is $(uname -s) $(uname -m)." >&2; exit 1 ;;
esac
say "Downloading the BioModelBench tool to $DIR"
mkdir -p "$DIR/app" "$DIR/bin"
chmod 700 "$DIR"
if ! curl -fsSL "$HUB/bmb-eval.tar.gz" | tar -xz -C "$DIR/app"; then
  echo "Couldn't download from $HUB. This computer needs to reach it over HTTPS." >&2
  exit 1
fi
if [ ! -x "$DIR/bin/uv" ] || [ "$("$DIR/bin/uv" --version 2>/dev/null)" != "uv $UV_VERSION" ]; then
  if ! curl -fsSL "$HUB/dl/uv/$UV_VERSION/uv-$UV_TARGET.tar.gz" | tar -xz -C "$DIR/bin" --strip-components 1; then
    echo "Couldn't download uv (the Python installer) from $HUB." >&2
    exit 1
  fi
fi
# UV_NATIVE_TLS: trust this computer's certificates (as curl does), so networks that
# inspect HTTPS with their own certificate authority work.
export UV_PYTHON_INSTALL_MIRROR="$HUB/dl/python" UV_PYTHON_INSTALL_DIR="$DIR/python" \
       UV_CACHE_DIR="$DIR/cache" UV_NATIVE_TLS=1
# A PyPI mirror your network uses (pip's setting) works too.
if [ -z "${UV_DEFAULT_INDEX:-}" ]; then
  INDEX="${PIP_INDEX_URL:-$(cat "$HOME/.config/pip/pip.conf" "$HOME/.pip/pip.conf" \
    "$HOME/Library/Application Support/pip/pip.conf" /etc/pip.conf /etc/xdg/pip/pip.conf 2>/dev/null |
    sed -n 's/^[[:space:]]*index-url[[:space:]]*=[[:space:]]*//p' | head -n 1)}"
  if [ -n "$INDEX" ]; then export UV_DEFAULT_INDEX="$INDEX"; fi
fi
if [ ! -x "$DIR/.venv/bin/python" ] && ! "$DIR/bin/uv" venv -q --python 3.12 "$DIR/.venv"; then
  echo "Couldn't install Python 3.12 (downloaded through $HUB)." >&2
  exit 1
fi
if ! VIRTUAL_ENV="$DIR/.venv" "$DIR/bin/uv" pip install -q "$DIR/app[seal]"; then
  echo "Couldn't download Python packages from ${UV_DEFAULT_INDEX:-PyPI (pypi.org, files.pythonhosted.org)}." >&2
  echo "If your network uses a PyPI mirror, run: export PIP_INDEX_URL=<its URL>, then the same command again." >&2
  exit 1
fi
BMB_CALLER_CWD="$(pwd)" "$DIR/.venv/bin/bmb-eval" connect-confidential --hub "$HUB" --dir "$DIR" "$@"
