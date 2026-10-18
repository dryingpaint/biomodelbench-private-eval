#!/bin/bash
# Runs as root on every boot of the hub VM (installed by hub-setup.sh): system packages,
# uv and the code, then hub-deploy.sh, which installs the hub, builds the sandbox image
# providers download, and runs the hub behind Caddy (which obtains the HTTPS certificate).
# To update the code, don't reboot: `deploy/gcp/hub deploy` runs hub-deploy.sh on the
# live VM, and the hub is unavailable for about a second, with requests held by Caddy.
set -euo pipefail
md() { curl -fsS -H Metadata-Flavor:Google \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
DOMAIN="$(md domain)"; BUCKET="$(md bucket)"; REPO="$(md repo)"
HOME_DIR=/home/bmb; CODE="$HOME_DIR/bmb-eval"; STATE="$HOME_DIR/hub_state"

id bmb >/dev/null 2>&1 || useradd --create-home --shell /bin/bash bmb
export DEBIAN_FRONTEND=noninteractive
# A reset can interrupt Ubuntu's automatic updates; finish them first, and wait
# for the package lock rather than failing. Install only what's missing.
dpkg --configure -a || true
missing=""
for pkg in git podman caddy curl; do dpkg -s "$pkg" >/dev/null 2>&1 || missing="$missing $pkg"; done
if [ -n "$missing" ]; then
  apt-get -o DPkg::Lock::Timeout=600 update -qq
  apt-get -o DPkg::Lock::Timeout=600 install -y -qq $missing >/dev/null
fi

as_bmb() { sudo -u bmb -H bash -lc "$*"; }
as_bmb 'command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null'
if [ -d "$CODE/.git" ]; then as_bmb "git -C $CODE pull -q"; else as_bmb "git clone -q $REPO $CODE"; fi
# The VM runs the copy of this script stored in its metadata at setup time. Run the
# repo's current copy instead, so "push, then reboot" also deploys changes to this script.
if [ -z "${BMB_STARTUP_REEXEC:-}" ] && ! cmp -s "$0" "$CODE/deploy/gcp/hub-vm-startup.sh"; then
  BMB_STARTUP_REEXEC=1 exec bash "$CODE/deploy/gcp/hub-vm-startup.sh"
fi
exec bash "$CODE/deploy/gcp/hub-deploy.sh"
