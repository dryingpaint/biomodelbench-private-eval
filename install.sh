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

# Rootless Podman limits CPU through the user's systemd cgroup, and some systemd defaults
# (Ubuntu 22.04, RHEL/Rocky 9) don't hand users the CPU controller: every sandbox would
# fail with "cgroup controller cpu is not available". Delegate it (Podman's documented fix).
UCG="/sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/cgroup.controllers"
if [ "$ENGINE" != "local" ] && [ "$(id -u)" != 0 ] && command -v podman >/dev/null &&
   [ -f /sys/fs/cgroup/cgroup.controllers ] && [ -f "$UCG" ] && ! grep -qw cpu "$UCG"; then
  say "Letting the sandbox limit CPU use (a systemd setting)"
  sudo mkdir -p /etc/systemd/system/user@.service.d
  printf '[Service]\nDelegate=cpu cpuset io memory pids\n' |
    sudo tee /etc/systemd/system/user@.service.d/delegate.conf >/dev/null
  sudo systemctl daemon-reload
  sudo systemctl restart "user@$(id -u).service"
  if ! grep -qw cpu "$UCG" 2>/dev/null; then
    echo "Couldn't give the sandbox CPU limits. Log out and back in, then run the same command again." >&2
    exit 1
  fi
fi

# Without a storage setting, some rootless Podman versions (Debian 12's 4.3) use "vfs",
# which copies the whole multi-GB sandbox image for every evaluation (minutes each).
# Switch to overlay, but only while nothing besides our image is stored.
if [ "$ENGINE" != "local" ] && [ "$(id -u)" != 0 ] && command -v podman >/dev/null &&
   [ "$(podman info --format '{{.Store.GraphDriverName}}' 2>/dev/null)" = vfs ]; then
  if podman images --format '{{.Repository}}' 2>/dev/null | grep -qv 'bmb-eval-sandbox'; then
    echo "Note: Podman here uses slow \"vfs\" storage, and has other images, so it was left as is." >&2
    echo "  Evaluations will be slow. To fix it, see https://github.com/containers/podman/blob/main/docs/tutorials/rootless_tutorial.md" >&2
  else
    say "Setting up fast container storage"
    podman system reset -f >/dev/null 2>&1 || true
    mkdir -p "$HOME/.config/containers"
    KV="$(uname -r | cut -d. -f1-2)"
    if [ "$(printf '%s\n5.13\n' "$KV" | sort -V | head -n 1)" != 5.13 ] && command -v fuse-overlayfs >/dev/null; then
      MP='mount_program = "'"$(command -v fuse-overlayfs)"'"'      # kernels before 5.13
    else
      MP=""
    fi
    printf '[storage]\ndriver = "overlay"\n[storage.options.overlay]\n%s\n' "$MP" \
      > "$HOME/.config/containers/storage.conf"
  fi
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
