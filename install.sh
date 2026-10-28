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
START="$(date +%s)"

if [ "$ENGINE" != "local" ] && [ "$(uname -s)" != "Linux" ]; then
  echo "This needs a Linux machine. See $HUB/guide for how to get one." >&2
  exit 1
fi

if [ $# -gt 0 ]; then
  cat <<'EOF'
Connecting this machine to BioModelBench. What happens, and how long each part usually takes:
  1. Install the sandbox software (Podman), if needed      1-3 minutes, asks for your password once
  2. Install the evaluator and its Python packages          1-2 minutes
  3. Download the sandbox image models run in (a few GB)    2-10 minutes, depending on your connection
  4. Check your data and prepare it on this machine         seconds to a few minutes, by data size
  5. Test this machine's security                           1-3 minutes
  6. Show you exactly what's shared, and ask you to approve
Nothing is shared until you approve in step 6.
EOF
fi

engine_ok() {
  [ "$ENGINE" = "local" ] ||
  { command -v podman >/dev/null && podman info >/dev/null 2>&1; } ||
  { command -v docker >/dev/null && docker info >/dev/null 2>&1; }
}
if ! engine_ok; then
  say "Installing Podman, the sandbox software (usually 1-3 minutes; asks for your password once)"
  if command -v apt-get >/dev/null; then
    # dbus-user-session: rootless Podman applies limits through the user's systemd, which
    # it reaches over the user's D-Bus (Debian's cloud images don't have it).
    sudo apt-get update -qq && sudo apt-get install -y -qq podman dbus-user-session
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
  # Right after installing Podman (its user units), the first restart can fail: retry.
  sudo systemctl restart "user@$(id -u).service" 2>/dev/null ||
    { sleep 5; sudo systemctl restart "user@$(id -u).service" 2>/dev/null; } || true
  sleep 1
  if ! grep -qw cpu "$UCG" 2>/dev/null; then
    echo "Couldn't give the sandbox CPU limits. Log out and back in, then run the same command again." >&2
    exit 1
  fi
fi

# Rootless Podman without the user's systemd (no D-Bus user session) falls back to
# "cgroupfs", where memory and process limits aren't applied: the self-test then refuses
# to start. Install the D-Bus user session (Debian, older Ubuntu) and restart systemd.
if [ "$ENGINE" != "local" ] && [ "$(id -u)" != 0 ] && command -v podman >/dev/null &&
   [ -f /sys/fs/cgroup/cgroup.controllers ] &&
   [ "$(podman info --format '{{.Host.CgroupManager}}' 2>/dev/null)" != systemd ]; then
  say "Setting up the user session the sandbox needs for its limits"
  if command -v apt-get >/dev/null; then
    sudo apt-get install -y -qq dbus-user-session
  elif command -v dnf >/dev/null; then
    sudo dnf install -y -q dbus-daemon
  fi
  sudo loginctl enable-linger "$(id -un)" 2>/dev/null || true
  sudo systemctl restart "user@$(id -u).service" 2>/dev/null ||
    { sleep 5; sudo systemctl restart "user@$(id -u).service" 2>/dev/null; } || true
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
  sleep 2
  if [ "$(podman info --format '{{.Host.CgroupManager}}' 2>/dev/null)" != systemd ]; then
    echo "The sandbox can't set its limits yet. Log out and back in, then run the same command again." >&2
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
    :   # setup.sh writes the device spec Podman's sandbox needs (src/bmb_eval/gpu_spec.py)
  elif command -v docker >/dev/null && ! docker info 2>/dev/null | grep -q nvidia; then
    sudo nvidia-ctk runtime configure --runtime=docker >/dev/null && sudo systemctl restart docker
  fi
fi

say "Downloading the evaluator to $DIR (a few seconds)"
mkdir -p "$DIR"
curl -fsSL "$HUB/bmb-eval.tar.gz" | tar -xz -C "$DIR"

CALLER_CWD="$(pwd)"
cd "$DIR"
if [ $# -gt 0 ]; then
  BMB_HUB="$HUB" BMB_ENGINE="$ENGINE" BMB_SKIP_SELFTEST=1 ./setup.sh
  say "Connecting"
  BMB_HUB="$HUB" BMB_CALLER_CWD="$CALLER_CWD" ./bmb-eval connect "$@"
  echo "  (whole setup took $(( ($(date +%s) - START) / 60 )) min)"
else
  BMB_HUB="$HUB" BMB_ENGINE="$ENGINE" ./setup.sh
fi
