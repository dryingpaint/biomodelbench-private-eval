#!/usr/bin/env bash
# Check the image BioModelBench asks you to approve: that Google Cloud Build built
# exactly that digest from the public source commit it names (signed SLSA
# provenance), so you can read the code your key will open for.
#   curl -fsSL https://eval.biomodelbench.org/verify-image.sh | sh
# Needs slsa-verifier: https://github.com/slsa-framework/slsa-verifier
set -eu
HUB="${BMB_HUB:-https://eval.biomodelbench.org}"
conf=$(curl -fsSL "$HUB/v1/confidential")
field() { printf '%s' "$conf" | python3 -c "import json,sys; print(json.load(sys.stdin).get('$1') or '')"; }
IMAGE=$(field image); SOURCE=$(field source); DIGEST=$(field digest)
[ -n "$SOURCE" ] || { echo "The hub didn't publish a source commit for $DIGEST." >&2; exit 1; }
REPO="${SOURCE%@*}"; COMMIT="${SOURCE##*@}"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
curl -fsSL "$HUB/v1/confidential/provenance" -o "$TMP/provenance.json"
command -v slsa-verifier >/dev/null || { echo "Install slsa-verifier first: go install github.com/slsa-framework/slsa-verifier/v2/cli/slsa-verifier@latest" >&2; exit 1; }
slsa-verifier verify-image "$IMAGE" --provenance-path "$TMP/provenance.json" \
  --source-uri "${REPO#https://}" --builder-id=https://cloudbuild.googleapis.com/GoogleHostedWorker
echo
echo "Verified: $DIGEST was built by Google Cloud Build from $REPO at commit $COMMIT."
echo "Read that code: $REPO/tree/$COMMIT (sealed node: src/bmb_eval/confidential.py,"
echo "sandbox: src/bmb_eval/sandbox.py, image: deploy/confidential/Dockerfile)."
