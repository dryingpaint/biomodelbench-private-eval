#!/usr/bin/env bash
# Installs bmb-eval into ./.venv and prepares the sandbox image. Called by
# install.sh; also works from a git checkout. Safe to re-run (updates).
set -euo pipefail
cd "$(dirname "$0")"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

say "Installing"
# uv installs without sudo and brings its own Python, so this works the same on
# every distribution (stock Ubuntu's python3 can't create virtualenvs).
# From the hub (install.sh): uv, and the Python it installs, come through the hub, since
# networks that allow it often block GitHub; PyPI or pip's configured mirror for packages.
if [ -n "${BMB_HUB:-}" ]; then
  UV_VERSION=0.13.1
  case "$(uname -m)" in
    x86_64) UV_TARGET=x86_64-unknown-linux-musl ;;
    aarch64|arm64) UV_TARGET=aarch64-unknown-linux-musl ;;
    *) echo "This runs on x86_64 or ARM Linux; this machine is $(uname -m)." >&2; exit 1 ;;
  esac
  mkdir -p bin
  if [ "$(bin/uv --version 2>/dev/null | cut -d' ' -f2)" != "$UV_VERSION" ] &&
     ! curl -fsSL "$BMB_HUB/dl/uv/$UV_VERSION/uv-$UV_TARGET.tar.gz" | tar -xz -C bin --strip-components 1; then
    echo "Couldn't download uv (the Python installer) from $BMB_HUB." >&2
    exit 1
  fi
  UV="$PWD/bin/uv"
  export UV_PYTHON_INSTALL_MIRROR="$BMB_HUB/dl/python" UV_PYTHON_INSTALL_DIR="$PWD/python" \
         UV_CACHE_DIR="$PWD/cache" UV_NATIVE_TLS=1
  if [ -z "${UV_DEFAULT_INDEX:-}" ]; then
    # (no pip.conf is the usual case: cat fails, which pipefail would turn into an exit)
    INDEX="${PIP_INDEX_URL:-$({ cat "$HOME/.config/pip/pip.conf" "$HOME/.pip/pip.conf" /etc/pip.conf \
      /etc/xdg/pip/pip.conf 2>/dev/null || true; } | sed -n 's/^[[:space:]]*index-url[[:space:]]*=[[:space:]]*//p' | head -n 1)}"
    if [ -n "$INDEX" ]; then export UV_DEFAULT_INDEX="$INDEX"; fi
  fi
else
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
  UV=uv
fi
if [ ! -x .venv/bin/python ]; then
  rm -rf .venv
  "$UV" venv -q --python 3.12 .venv || { echo "Couldn't install Python 3.12." >&2; exit 1; }
fi
if ! "$UV" pip install -q --python .venv/bin/python -e .; then
  echo "Couldn't download Python packages from ${UV_DEFAULT_INDEX:-PyPI (pypi.org, files.pythonhosted.org)}." >&2
  echo "If your network uses a PyPI mirror, run: export PIP_INDEX_URL=<its URL>, then the same command again." >&2
  exit 1
fi

[ -f config.yaml ] || ./bmb-eval init --engine "${BMB_ENGINE:-auto}"

say "Preparing the sandbox"
./bmb-eval build-image ${BMB_HUB:+--from "$BMB_HUB/sandbox-image.tar"}

# On an update, re-test and re-approve the already-connected task(s).
# A first install skips this: `connect` runs the self-test once, on your data.
if [ -z "${BMB_SKIP_SELFTEST:-}" ]; then
  ./bmb-eval selftest && ./bmb-eval approve >/dev/null
  # Updating: restart the background service so it runs the new code.
  if systemctl --user is-enabled bmb-eval >/dev/null 2>&1; then
    systemctl --user restart bmb-eval && echo "Updated and restarted."
  fi
fi
