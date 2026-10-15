#!/usr/bin/env bash
# Build the sealed-node image from a public commit and publish it on the hub.
#   deploy/confidential/build.sh [commit]        (default: origin/main's HEAD)
# Needs the commit pushed to github.com/dryingpaint/biomodelbench-private-eval.
set -euo pipefail
REPO=https://github.com/dryingpaint/biomodelbench-private-eval
PROJECT="${PROJECT:-biomodelbench}"
SHA="${1:-$(git rev-parse origin/main)}"
TAG="${SHA:0:12}"
IMG="us-central1-docker.pkg.dev/$PROJECT/bmb/sealed-node"
gcloud builds submit "$REPO" --git-source-revision="$SHA" --project "$PROJECT" --region us-central1 \
  --config deploy/confidential/cloudbuild.yaml --substitutions "_TAG=$TAG"
DIGEST=$(gcloud artifacts docker images describe "$IMG:$TAG" --project "$PROJECT" --format='value(image_summary.digest)')
OUT="$(mktemp -d)/provenance.json"
gcloud artifacts docker images describe "$IMG@$DIGEST" --project "$PROJECT" --show-provenance \
  --format=json > "$OUT"
echo "built $IMG@$DIGEST from $REPO@$SHA"
echo "publish:  scp $OUT to the hub, then"
echo "  bmb-hub confidential-image $IMG@$DIGEST --workload-sa bmb-cs-workload@$PROJECT.iam.gserviceaccount.com \\"
echo "    --gpu-preview --source $REPO@$SHA --provenance <path on hub>"
