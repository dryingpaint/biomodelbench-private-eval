#!/bin/sh
# BioModelBench evaluation node: the one command providers run.
#   curl -fsSL https://<hub>/install.sh | sh -s -- <CODE> --data <file>
# Installs Podman (and GPU container support on GPU machines) if needed,
# downloads bmb-eval to ~/bmb-eval, then connects.
# Without arguments it only installs or updates. Safe to re-run.
set -eu
HUB="${BMB_HUB:-__BMB_HUB__}"
DIR="${BMB_DIR:-$HOME/bmb-eval}"
ENGINE="${BMB_ENGINE:-auto}"     # tests only: "local" skips the sandbox engine
say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

if [ "$ENGINE" != "local" ] && [ "$(uname -s)" != "Linux" ]; then
  echo "This needs a Linux machine. See $HUB/guide for how to get one." >&2
  exit 1
fi

engine_ok() {
  [ "$ENGINE" = "local" ] ||
  { command -v podman >/dev/null && podman info >/dev/null 2>&1; } ||
  { command -v docker >/dev/null && docker info >/dev/null 2>&1; }
}
if ! engine_ok; then
  say "Installing Podman, the sandbox software (asks for your password once)"
  if command -v apt-get >/dev/null; then
    sudo apt-get update -qq && sudo apt-get install -y -qq podman
  elif command -v dnf >/dev/null; then
    sudo dnf install -y -q podman
  elif command -v yum >/dev/null; then
    sudo yum install -y -q podman
  else
    echo "Please install Podman with your package manager, then run this again." >&2
    exit 1
  fi
  # Keep the evaluator running after you log out.
  sudo loginctl enable-linger "$(id -un)" 2>/dev/null || true
fi

# GPU machines: let containers use the GPU (NVIDIA container toolkit).
if [ "$ENGINE" != "local" ] && command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1 &&
   ! command -v nvidia-ctk >/dev/null; then
  say "Setting up GPU access for containers"
  if command -v apt-get >/dev/null; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |
      sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' |
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    sudo apt-get update -qq && sudo apt-get install -y -qq nvidia-container-toolkit
  elif command -v dnf >/dev/null; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo |
      sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo >/dev/null
    sudo dnf install -y -q nvidia-container-toolkit
  fi
fi
if command -v nvidia-ctk >/dev/null; then
  if command -v podman >/dev/null; then
    sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml >/dev/null 2>&1 || true
  elif command -v docker >/dev/null && ! docker info 2>/dev/null | grep -q nvidia; then
    sudo nvidia-ctk runtime configure --runtime=docker >/dev/null && sudo systemctl restart docker
  fi
fi

say "Downloading the evaluator to $DIR"
mkdir -p "$DIR"
curl -fsSL "$HUB/bmb-eval.tar.gz" | tar -xz -C "$DIR"

CALLER_CWD="$(pwd)"
cd "$DIR"
if [ $# -gt 0 ]; then
  BMB_HUB="$HUB" BMB_ENGINE="$ENGINE" BMB_SKIP_SELFTEST=1 ./setup.sh
  say "Connecting"
  BMB_HUB="$HUB" BMB_CALLER_CWD="$CALLER_CWD" ./bmb-eval connect "$@"
else
  BMB_HUB="$HUB" BMB_ENGINE="$ENGINE" ./setup.sh
fi
