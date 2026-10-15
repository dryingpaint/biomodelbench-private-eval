# The sealed node: runs as a Confidential Space workload on BioModelBench's
# machine. Providers approve this image's digest; their key opens only for it.
#
# Built by Google Cloud Build from a public git commit, with signed SLSA
# provenance, so anyone can check which source a digest came from
# (deploy/confidential/build.sh, verify-image.sh). Inputs are pinned: the base
# image by digest, Debian packages by snapshot date, Python packages by hash.
FROM docker.io/library/python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1
ARG DEBIAN_SNAPSHOT=20261015T000000Z
RUN printf 'Types: deb\nURIs: http://snapshot.debian.org/archive/debian/%s\nSuites: trixie trixie-updates\nComponents: main\nSigned-By: /usr/share/keyrings/debian-archive-keyring.pgp\n\nTypes: deb\nURIs: http://snapshot.debian.org/archive/debian-security/%s\nSuites: trixie-security\nComponents: main\nSigned-By: /usr/share/keyrings/debian-archive-keyring.pgp\n' \
      "$DEBIAN_SNAPSHOT" "$DEBIAN_SNAPSHOT" > /etc/apt/sources.list.d/debian.sources \
 && apt-get -o Acquire::Check-Valid-Until=false update \
 && apt-get install -y --no-install-recommends bubblewrap procps libseccomp2 \
 && rm -rf /var/lib/apt/lists/*
# What submitted models can import (the provider-run sandbox's set), plus what the
# node needs to reach the provider's key and bucket. Every wheel is hash-checked.
# PyTorch is its CUDA 12.1 build: Confidential Space's GPU driver is 535 (CUDA 12.2).
COPY deploy/confidential/requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir --require-hashes --no-deps -r /tmp/requirements.lock \
      --extra-index-url https://download.pytorch.org/whl/cu121 \
 && rm /tmp/requirements.lock
COPY src/bmb_eval /opt/bmb/bmb_eval
ENV PYTHONPATH=/opt/bmb PYTHONDONTWRITEBYTECODE=1 \
    PATH=/usr/local/nvidia/bin:/usr/local/bin:/usr/bin:/bin \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib64 BMB_ROOT=/run/tmp/bmb
# No logs in production: Confidential Space refuses to redirect them unless the VM
# runs the debug image, which providers' keys never open. The node reports fixed
# status fields to the hub instead (bmb-hub status).
LABEL "tee.launch_policy.log_redirect"="debugonly"
LABEL "tee.launch_policy.allow_env_override"="PROVIDER_PROJECT_NUMBER,POOL,PROVIDER_ID,KEY,BUCKET,BMB_IDLE_EXIT_S"
LABEL "tee.launch_policy.allow_mount_destinations"="/run/tmp"
CMD ["python3", "-c", "from bmb_eval.cli import main; main(['confidential-node'])"]
