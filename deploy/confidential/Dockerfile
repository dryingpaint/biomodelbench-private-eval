# The sealed node: runs as a Confidential Space workload on BioModelBench's
# machine. Providers approve this image's digest; their key opens only for it.
# Build from the repo root:
#   gcloud builds submit --config deploy/confidential/cloudbuild.yaml
FROM docker.io/library/python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends bubblewrap procps \
 && rm -rf /var/lib/apt/lists/*
# What submitted models can import (same as the provider-run sandbox), plus
# what the node needs to reach the provider's key and bucket.
COPY sandbox/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
      cryptography google-auth google-cloud-kms google-cloud-storage requests \
 && rm /tmp/requirements.txt
COPY src/bmb_eval /opt/bmb/bmb_eval
ENV PYTHONPATH=/opt/bmb PYTHONDONTWRITEBYTECODE=1 \
    PATH=/usr/local/nvidia/bin:/usr/local/bin:/usr/bin:/bin \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib64 BMB_ROOT=/run/tmp/bmb
# Logs carry only the node's fixed status lines ("BMB {...}"), never data.
LABEL "tee.launch_policy.log_redirect"="always"
LABEL "tee.launch_policy.allow_env_override"="PROVIDER_PROJECT_NUMBER,POOL,PROVIDER_ID,KEY,BUCKET,BMB_IDLE_EXIT_S"
LABEL "tee.launch_policy.allow_mount_destinations"="/run/tmp"
CMD ["python3", "-c", "from bmb_eval.cli import main; main(['confidential-node'])"]
