#!/bin/bash
# Installs the checked-out code and (re)starts the hub. Runs as root, from
# hub-vm-startup.sh on boot and from `deploy/gcp/hub deploy` on the live VM. Everything
# slow (dependencies, the sandbox image) happens while the old hub keeps serving; only
# the final restart interrupts it, for about a second, and Caddy holds requests meanwhile.
set -euo pipefail
md() { curl -fsS -H Metadata-Flavor:Google \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
DOMAIN="$(md domain)"; BUCKET="$(md bucket)"
HOME_DIR=/home/bmb; CODE="$HOME_DIR/bmb-eval"; STATE="$HOME_DIR/hub_state"
as_bmb() { sudo -u bmb -H bash -lc "$*"; }

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
  # A build interrupted after this point leaves the .tmp behind, and podman save won't
  # overwrite it ("docker-archive doesn't support modifying existing images"): from
  # 2026-10-17 every boot failed here and the published image stopped updating.
  rm -f "$STATE/sandbox-image.tar.tmp"
  podman save -o "$STATE/sandbox-image.tar.tmp" bmb-eval-sandbox:latest
  sync
  tar -tf "$STATE/sandbox-image.tar.tmp" >/dev/null
  mv "$STATE/sandbox-image.tar.tmp" "$STATE/sandbox-image.tar"
  (cd "$STATE" && sha256sum sandbox-image.tar > sandbox-image.tar.sha256)
  echo "$STAMP" > "$STATE/sandbox-image.stamp"
  sync
  chown bmb "$STATE"/sandbox-image.*
fi

UNIT_FILE=/etc/systemd/system/bmb-hub.service
cat > "$UNIT_FILE.new" <<UNIT
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
if ! cmp -s "$UNIT_FILE.new" "$UNIT_FILE"; then mv "$UNIT_FILE.new" "$UNIT_FILE"; systemctl daemon-reload
else rm -f "$UNIT_FILE.new"; fi

# Model files go straight to Cloud Storage, so the hub only sees small JSON
# bodies plus the provider downloads it serves (install script, image).
# While the hub restarts (about a second on a deploy), Caddy holds requests and retries
# the hub for up to 2 minutes rather than answering 502: a deploy doesn't fail anyone's
# call. Requests are only retried when the hub wasn't reachable, so none runs twice.
cat > /etc/caddy/Caddyfile.new <<CADDY
$DOMAIN {
    request_body {
        max_size 32MB
    }
    reverse_proxy 127.0.0.1:8790 {
        lb_try_duration 2m
        lb_try_interval 250ms
    }
}
CADDY
caddy validate --adapter caddyfile --config /etc/caddy/Caddyfile.new >/dev/null
mv /etc/caddy/Caddyfile.new /etc/caddy/Caddyfile

# Warm the page cache with the new code before the swap. The VM's standard persistent disk
# reads the venv's thousands of files slowly: on 2026-10-17 a restart right after an
# install took 3 minutes to start listening, against half a second once the files are
# cached. Importing the hub here, while the old one serves, is what makes the restart quick.
as_bmb "cd $CODE && time .venv/bin/python -c 'import bmb_eval.hub, bmb_eval.sealed_launcher'" 2>&1 | grep real || true

systemctl enable bmb-hub caddy >/dev/null
# Everything above ran with the old hub still serving. Now swap: restart the hub, and
# reload Caddy's config in place (a reload drops no connections).
systemctl restart bmb-hub
if systemctl is-active -q caddy; then systemctl reload caddy; else systemctl restart caddy; fi
for i in $(seq 1 60); do
  curl -fsS -o /dev/null http://127.0.0.1:8790/v1/tasks && { echo "hub is serving the new code"; exit 0; }
  sleep 1
done
echo "the hub didn't answer within 60s after the restart; see: journalctl -u bmb-hub" >&2
exit 1
