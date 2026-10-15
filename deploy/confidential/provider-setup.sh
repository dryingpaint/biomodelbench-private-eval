#!/usr/bin/env bash
# Provider side (prototype): encrypt your data and let ONLY BioModelBench's
# attested evaluation image decrypt it. Run in your own Google Cloud project:
#
#   ./provider-setup.sh --project my-project --data data.csv \
#       --digest sha256:<image digest from BioModelBench> \
#       --workload-sa bmb-cs-workload@biomodelbench.iam.gserviceaccount.com
#
# What it creates in YOUR project:
#   - a KMS key (you hold it; you can revoke access any time),
#   - a bucket with your file, encrypted on this machine before upload,
#   - a workload identity pool that trusts Google's Confidential Space
#     attestation, accepting only: production Confidential Space, debugging
#     off, running exactly --digest, as BioModelBench's workload account.
# Re-run with a new --digest to approve an updated image (the old one stops
# working). --allow-gpu-preview also accepts the GPU preview image (H100 in
# confidential mode). Requires gcloud and python3 with `cryptography`.
set -euo pipefail
NAME=bmb; POOL_ID=bmb-eval; LOC=us-central1
while [ $# -gt 0 ]; do case "$1" in
  --project) PROJECT=$2; shift 2;; --data) DATA=$2; shift 2;;
  --digest) DIGEST=$2; shift 2;; --workload-sa) SA=$2; shift 2;;
  --allow-gpu-preview) GPU_PREVIEW=1; shift;;
  *) echo "unknown argument $1" >&2; exit 2;; esac; done
: "${PROJECT:?--project}" "${DATA:?--data}" "${DIGEST:?--digest}" "${SA:?--workload-sa}"
g() { gcloud --project "$PROJECT" --quiet "$@"; }
NUM=$(g projects describe "$PROJECT" --format='value(projectNumber)')
BUCKET="gs://$PROJECT-$NAME-data"; KEY="projects/$PROJECT/locations/$LOC/keyRings/$NAME/cryptoKeys/data"
POOL="principalSet://iam.googleapis.com/projects/$NUM/locations/global/workloadIdentityPools/$POOL_ID/*"

echo "1/4 Key and bucket"
g services enable cloudkms.googleapis.com storage.googleapis.com iam.googleapis.com sts.googleapis.com >/dev/null
g kms keyrings describe "$NAME" --location "$LOC" >/dev/null 2>&1 || g kms keyrings create "$NAME" --location "$LOC"
g kms keys describe data --keyring "$NAME" --location "$LOC" >/dev/null 2>&1 ||
  g kms keys create data --keyring "$NAME" --location "$LOC" --purpose encryption
g storage buckets describe "$BUCKET" >/dev/null 2>&1 ||
  g storage buckets create "$BUCKET" --location "$LOC" --uniform-bucket-level-access --public-access-prevention

echo "2/4 Encrypt on this machine, upload only ciphertext"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
python3 - "$DATA" "$TMP" <<'EOF'
import os, sys
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
data, out = open(sys.argv[1], "rb").read(), sys.argv[2]
dek, nonce = AESGCM.generate_key(256), os.urandom(12)
open(f"{out}/dek", "wb").write(dek)
open(f"{out}/data.enc", "wb").write(nonce + AESGCM(dek).encrypt(nonce, data, None))
EOF
g kms encrypt --key "$KEY" --plaintext-file "$TMP/dek" --ciphertext-file "$TMP/dek.enc"
g storage cp "$TMP/data.enc" "$BUCKET/data.csv.enc"
g storage cp "$TMP/dek.enc" "$BUCKET/data.dek.enc"

echo "3/4 Trust only the attested BioModelBench image"
SUPPORT="'STABLE' in assertion.submods.confidential_space.support_attributes"
# Confidential Space on GPUs is a preview ('EXPERIMENTAL' image). Opt in only
# for it, and only with the H100 attested in confidential-computing mode.
[ -n "${GPU_PREVIEW:-}" ] && SUPPORT="($SUPPORT || ('EXPERIMENTAL' in assertion.submods.confidential_space.support_attributes \
&& assertion.submods.nvidia_gpu.cc_mode == 'ON'))"
COND="assertion.swname == 'CONFIDENTIAL_SPACE' && $SUPPORT \
&& assertion.dbgstat == 'disabled-since-boot' && assertion.submods.container.image_digest == '$DIGEST' \
&& '$SA' in assertion.google_service_accounts"
g iam workload-identity-pools describe "$POOL_ID" --location global >/dev/null 2>&1 ||
  g iam workload-identity-pools create "$POOL_ID" --location global --display-name "BioModelBench evaluation"
if g iam workload-identity-pools providers describe attestation --location global \
     --workload-identity-pool "$POOL_ID" >/dev/null 2>&1; then verb=update-oidc; else verb=create-oidc; fi
g iam workload-identity-pools providers "$verb" attestation --location global --workload-identity-pool "$POOL_ID" \
  --issuer-uri https://confidentialcomputing.googleapis.com/ --allowed-audiences https://sts.googleapis.com \
  --attribute-mapping "google.subject=assertion.sub" --attribute-condition "$COND" >/dev/null

echo "4/4 Grant decrypt and read to that pool only"
g kms keys add-iam-policy-binding data --keyring "$NAME" --location "$LOC" \
  --member "$POOL" --role roles/cloudkms.cryptoKeyDecrypter >/dev/null
g storage buckets add-iam-policy-binding "$BUCKET" --member "$POOL" --role roles/storage.objectViewer >/dev/null

cat <<EOF

Done. Send BioModelBench these (none are secret):
  PROVIDER_PROJECT_NUMBER=$NUM
  POOL=$POOL_ID
  PROVIDER_ID=attestation
  KEY=$KEY
  DATA_URI=$BUCKET/data.csv.enc
  DEK_URI=$BUCKET/data.dek.enc
Approved image: $DIGEST
To revoke at any time: gcloud kms keys disable or delete the pool '$POOL_ID'.
EOF
