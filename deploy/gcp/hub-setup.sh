#!/usr/bin/env bash
# Bring up the BioModelBench hub on Google Cloud. Run from your laptop:
#
#   deploy/gcp/hub-setup.sh                       # hub at https://<ip>.sslip.io
#   deploy/gcp/hub-setup.sh eval.your-domain.org  # or on your own domain
#
# Creates (or reuses): a Cloud Storage bucket for model files, a service
# account that can sign download links, a static IP, firewall rules for
# HTTPS, and a small VM that installs and runs the hub on boot (Caddy gets
# the HTTPS certificate automatically). With your own domain, point its DNS
# at the printed IP. Safe to re-run.
#
# Settings (environment): PROJECT, REGION, ZONE, NAME, BUCKET, REPO.
set -euo pipefail
cd "$(dirname "$0")"

DOMAIN="${1:-}"
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-central1}"
ZONE="${ZONE:-$REGION-a}"
NAME="${NAME:-bmb-hub}"
BUCKET="${BUCKET:-$PROJECT-bmb-models}"
REPO="${REPO:-https://github.com/dryingpaint/biomodelbench-private-eval}"
SA="$NAME@$PROJECT.iam.gserviceaccount.com"
say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
exists() { "$@" >/dev/null 2>&1; }

say "Project $PROJECT, zone $ZONE"
gcloud services enable compute.googleapis.com storage.googleapis.com iamcredentials.googleapis.com \
  --project "$PROJECT"

say "Service account"
exists gcloud iam service-accounts describe "$SA" --project "$PROJECT" ||
  gcloud iam service-accounts create "$NAME" --display-name "BioModelBench hub" --project "$PROJECT"
# The hub signs upload/download links as itself. (Editors may not be allowed
# to set a service account's own policy; a project-level grant also works.)
gcloud iam service-accounts add-iam-policy-binding "$SA" --project "$PROJECT" \
  --member "serviceAccount:$SA" --role roles/iam.serviceAccountTokenCreator >/dev/null 2>&1 ||
  gcloud projects add-iam-policy-binding "$PROJECT" --condition None \
    --member "serviceAccount:$SA" --role roles/iam.serviceAccountTokenCreator >/dev/null

say "Sealed nodes (confidential computing)"
# The account sealed VMs run as (providers' attestation rules name it), and the
# hub's right to start those VMs when jobs are queued (bmb-hub serve --launch-sealed).
WL="bmb-cs-workload@$PROJECT.iam.gserviceaccount.com"
exists gcloud iam service-accounts describe "$WL" --project "$PROJECT" ||
  gcloud iam service-accounts create bmb-cs-workload --display-name "BMB sealed node" --project "$PROJECT"
for role in roles/confidentialcomputing.workloadUser roles/logging.logWriter roles/artifactregistry.reader; do
  gcloud projects add-iam-policy-binding "$PROJECT" --condition None \
    --member "serviceAccount:$WL" --role "$role" >/dev/null
done
gcloud projects add-iam-policy-binding "$PROJECT" --condition None \
  --member "serviceAccount:$SA" --role roles/compute.instanceAdmin.v1 >/dev/null
gcloud iam service-accounts add-iam-policy-binding "$WL" --project "$PROJECT" \
  --member "serviceAccount:$SA" --role roles/iam.serviceAccountUser >/dev/null

say "Bucket gs://$BUCKET for model files"
exists gcloud storage buckets describe "gs://$BUCKET" ||
  gcloud storage buckets create "gs://$BUCKET" --project "$PROJECT" --location "$REGION" \
    --uniform-bucket-level-access --public-access-prevention
# Model files are kept 30 days; a later submission re-uploads anything expired.
gcloud storage buckets update "gs://$BUCKET" --lifecycle-file lifecycle.json >/dev/null
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
  --member "serviceAccount:$SA" --role roles/storage.objectAdmin >/dev/null 2>&1 ||
  gcloud projects add-iam-policy-binding "$PROJECT" --condition None \
    --member "serviceAccount:$SA" --role roles/storage.objectAdmin >/dev/null

say "Static IP and firewall"
exists gcloud compute addresses describe "$NAME" --region "$REGION" --project "$PROJECT" ||
  gcloud compute addresses create "$NAME" --region "$REGION" --project "$PROJECT"
IP="$(gcloud compute addresses describe "$NAME" --region "$REGION" --project "$PROJECT" --format 'value(address)')"
# No domain: sslip.io maps <a-b-c-d>.sslip.io to a.b.c.d, and Caddy can get a
# real certificate for it.
DNS_STEP=1
if [ -z "$DOMAIN" ]; then DOMAIN="${IP//./-}.sslip.io"; DNS_STEP=; fi
exists gcloud compute firewall-rules describe "$NAME-https" --project "$PROJECT" ||
  gcloud compute firewall-rules create "$NAME-https" --project "$PROJECT" \
    --allow tcp:80,tcp:443 --target-tags "$NAME"

say "VM $NAME"
if exists gcloud compute instances describe "$NAME" --zone "$ZONE" --project "$PROJECT"; then
  gcloud compute instances add-metadata "$NAME" --zone "$ZONE" --project "$PROJECT" \
    --metadata "domain=$DOMAIN,bucket=$BUCKET,repo=$REPO" --metadata-from-file startup-script=hub-vm-startup.sh
  echo "VM exists; settings updated. Apply with: gcloud compute ssh $NAME --zone $ZONE -- sudo reboot"
else
  gcloud compute instances create "$NAME" --project "$PROJECT" --zone "$ZONE" \
    --machine-type e2-standard-2 --boot-disk-size 100GB --boot-disk-type pd-balanced \
    --image-family ubuntu-2404-lts-amd64 --image-project ubuntu-os-cloud \
    --address "$IP" --tags "$NAME" --service-account "$SA" --scopes cloud-platform \
    --metadata "domain=$DOMAIN,bucket=$BUCKET,repo=$REPO" \
    --metadata-from-file startup-script=hub-vm-startup.sh
fi

say "Almost done: the hub will be at https://$DOMAIN"
[ -n "$DNS_STEP" ] && echo "Add this DNS record where your domain is managed:   $DOMAIN  A  $IP"
cat <<MSG
The VM finishes setting itself up in about 15 minutes (it builds the sandbox
image providers download). Then this prints the installer:
    curl -fsS https://$DOMAIN/install.sh | head -3
Admin commands run from here, e.g.:
    deploy/gcp/hub agent-token my-agent
    deploy/gcp/hub enroll examples/tasks/sequence_regression
MSG
