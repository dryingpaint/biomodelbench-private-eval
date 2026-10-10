#!/usr/bin/env bash
# Delete everything hub-setup.sh created: VM, static IP, firewall rule,
# bucket (with all model files) and service account. Same settings as setup.
set -uo pipefail
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-central1}"; ZONE="${ZONE:-$REGION-a}"; NAME="${NAME:-bmb-hub}"
BUCKET="${BUCKET:-$PROJECT-bmb-models}"; SA="$NAME@$PROJECT.iam.gserviceaccount.com"
read -r -p "Delete the hub '$NAME' and gs://$BUCKET in $PROJECT? [y/N] " ok
[ "$ok" = y ] || exit 1
gcloud compute instances delete "$NAME" --zone "$ZONE" --project "$PROJECT" --quiet
gcloud compute addresses delete "$NAME" --region "$REGION" --project "$PROJECT" --quiet
gcloud compute firewall-rules delete "$NAME-https" --project "$PROJECT" --quiet
gcloud storage rm --recursive "gs://$BUCKET" --quiet
gcloud iam service-accounts delete "$SA" --project "$PROJECT" --quiet
