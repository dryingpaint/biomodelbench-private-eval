#!/bin/bash
# Runs as root on every boot of the hub VM (installed by hub-setup.sh).
# Installs or updates the hub, builds the sandbox image providers download,
# and runs the hub behind Caddy, which obtains the HTTPS certificate.
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
as_bmb "cd $CODE && ~/.local/bin/uv venv -q --python 3.12 .venv 2>/dev/null || true; \
        ~/.local/bin/uv pip install -q --python .venv/bin/python -e '.[hub]'"
as_bmb "mkdir -p $STATE"

# Sandbox image providers download (rebuilt only when its inputs change).
# Only what runs inside the sandbox: contract checks and the three entry points.
SANDBOX_FILES="sandbox/Dockerfile sandbox/requirements.txt src/bmb_eval/__init__.py src/bmb_eval/contract.py src/bmb_eval/errors.py src/bmb_eval/sandbox_entry/__init__.py src/bmb_eval/sandbox_entry/predict.py src/bmb_eval/sandbox_entry/validate.py src/bmb_eval/sandbox_entry/score.py"
STAMP="$(cd "$CODE" && cat $SANDBOX_FILES | sha256sum | cut -c1-16)"
# Built as root: a boot script has no user login session, which rootless
# Podman needs. The image is only exported as a file for providers.
# Published only once complete, flushed to disk and readable, with a checksum
# providers verify (a reset right after writing once truncated it).
if [ "$(cat "$STATE/sandbox-image.stamp" 2>/dev/null)" != "$STAMP" ] ||
   ! sha256sum --status -c "$STATE/sandbox-image.tar.sha256" 2>/dev/null; then
  rm -f "$STATE/sandbox-image.stamp"
  (cd "$CODE" && podman build -q --build-arg BMB_SANDBOX_STAMP="$STAMP" -t bmb-eval-sandbox:latest \
     -f sandbox/Dockerfile . >/dev/null)
  podman save -o "$STATE/sandbox-image.tar.tmp" bmb-eval-sandbox:latest
  sync
  tar -tf "$STATE/sandbox-image.tar.tmp" >/dev/null
  mv "$STATE/sandbox-image.tar.tmp" "$STATE/sandbox-image.tar"
  (cd "$STATE" && sha256sum sandbox-image.tar > sandbox-image.tar.sha256)
  echo "$STAMP" > "$STATE/sandbox-image.stamp"
  sync
  chown bmb "$STATE"/sandbox-image.*
fi

cat > /etc/systemd/system/bmb-hub.service <<UNIT
[Unit]
Description=BioModelBench hub
After=network-online.target

[Service]
User=bmb
WorkingDirectory=$CODE
ExecStart=$CODE/.venv/bin/bmb-hub --state $STATE serve --port 8790 --public-url https://$DOMAIN --bucket $BUCKET --launch-sealed
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

# Model files go straight to Cloud Storage, so the hub only sees small JSON
# bodies plus the provider downloads it serves (install script, image).
cat > /etc/caddy/Caddyfile <<CADDY
$DOMAIN {
    request_body {
        max_size 32MB
    }
    reverse_proxy 127.0.0.1:8790
}
CADDY

systemctl daemon-reload
systemctl enable --now bmb-hub >/dev/null
systemctl restart bmb-hub caddy
