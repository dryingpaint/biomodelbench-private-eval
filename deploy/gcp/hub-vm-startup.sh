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
apt-get update -qq
apt-get install -y -qq git podman caddy curl >/dev/null

as_bmb() { sudo -u bmb -H bash -lc "$*"; }
as_bmb 'command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null'
if [ -d "$CODE/.git" ]; then as_bmb "git -C $CODE pull -q"; else as_bmb "git clone -q $REPO $CODE"; fi
as_bmb "cd $CODE && ~/.local/bin/uv venv -q --python 3.12 .venv 2>/dev/null || true; \
        ~/.local/bin/uv pip install -q --python .venv/bin/python -e '.[hub]'"
as_bmb "mkdir -p $STATE"

# Sandbox image providers download (rebuilt only when its inputs change).
STAMP="$(cd "$CODE" && cat sandbox/Dockerfile sandbox/requirements.txt $(find src/bmb_eval -name '*.py' | sort) | sha256sum | cut -c1-16)"
if [ "$(cat "$STATE/sandbox-image.stamp" 2>/dev/null)" != "$STAMP" ]; then
  as_bmb "cd $CODE && podman build -q -t bmb-eval-sandbox:latest -f sandbox/Dockerfile . && \
          podman save -o $STATE/sandbox-image.tar.tmp bmb-eval-sandbox:latest && \
          mv $STATE/sandbox-image.tar.tmp $STATE/sandbox-image.tar && echo $STAMP > $STATE/sandbox-image.stamp"
fi

cat > /etc/systemd/system/bmb-hub.service <<UNIT
[Unit]
Description=BioModelBench hub
After=network-online.target

[Service]
User=bmb
WorkingDirectory=$CODE
ExecStart=$CODE/.venv/bin/bmb-hub --state $STATE serve --port 8790 --public-url https://$DOMAIN --bucket $BUCKET
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
