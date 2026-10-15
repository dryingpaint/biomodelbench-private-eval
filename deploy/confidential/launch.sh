#!/usr/bin/env bash
# Our side (prototype): run an image as a Confidential Space workload.
#   IMAGE=us-central1-docker.pkg.dev/biomodelbench/bmb/cs-probe@sha256:... \
#   ENV_FILE=provider.env ./launch.sh [cpu|h100]
# ENV_FILE holds the provider's KEY=VALUE lines (bmb-hub confidential-env <node>),
# or 'none' for a self-test run on synthetic data (debug image, logs on).
# Normally the hub starts sealed nodes itself (bmb-hub serve --launch-sealed).
# cpu:  n2d-standard-2, AMD SEV (Confidential Space GA)
# cpu-tdx: c3-standard-4, Intel TDX (Confidential Space GA)
# h100: a3-highgpu-1g, Intel TDX + H100 in confidential mode (Confidential Space on GPU: preview)
set -euo pipefail
KIND="${1:-cpu}"
PROJECT="${PROJECT:-biomodelbench}"; ZONE="${ZONE:-us-central1-a}"
NAME="${NAME:-bmb-cs-$KIND}"; SA="${SA:-bmb-cs-workload@$PROJECT.iam.gserviceaccount.com}"
: "${IMAGE:?IMAGE=<registry path>@sha256:<digest>}" "${ENV_FILE:?ENV_FILE=<provider values>}"
# Decrypted data lives only in memory: a tmpfs at /run/tmp (the node refuses anything else).
META="tee-image-reference=$IMAGE~tee-restart-policy=Never"
[ "$ENV_FILE" = none ] && META="$META~tee-container-log-redirect=true"
META="$META~tee-mount=type=tmpfs,source=tmpfs,destination=/run/tmp,size=${TMPFS_BYTES:-34359738368}"
# No ENV_FILE (ENV_FILE=none): the node runs its security self-test on synthetic data and exits.
if [ "$ENV_FILE" != none ]; then
  while IFS='=' read -r k v; do [ -n "$k" ] && META="$META~tee-env-$k=$v"; done < <(grep -E '^[A-Z_]+=' "$ENV_FILE")
fi
# Self-tests run on Google's debug image: production images have no logs, and a
# debug VM can never open provider data (providers' rules require dbgstat disabled).
DBG=""; [ "$ENV_FILE" = none ] && DBG="-debug"
if [ "$KIND" = h100 ]; then
  HW=(--machine-type a3-highgpu-1g --confidential-compute-type TDX --maintenance-policy TERMINATE
      --boot-disk-size 300GB --image-family confidential-space$DBG-preview-cgpu
      ${SPOT:+--provisioning-model SPOT --instance-termination-action DELETE})
  META="$META~tee-install-gpu-driver=true"
elif [ "$KIND" = cpu-tdx ]; then
  HW=(--machine-type c3-standard-4 --confidential-compute-type TDX --maintenance-policy TERMINATE
      --image-family confidential-space$DBG)
else
  HW=(--machine-type n2d-standard-2 --confidential-compute-type SEV --maintenance-policy MIGRATE
      --image-family confidential-space$DBG)
fi
gcloud compute instances create "$NAME" --project "$PROJECT" --zone "$ZONE" "${HW[@]}" \
  --image-project confidential-space-images --shielded-secure-boot \
  --service-account "$SA" --scopes cloud-platform --metadata "^~^$META"
echo "Logs: gcloud logging read 'logName=\"projects/$PROJECT/logs/confidential-space-launcher\" AND textPayload:BMB-PROBE' --project $PROJECT"
