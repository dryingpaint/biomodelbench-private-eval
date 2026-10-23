# Sandbox image: runs predict (untrusted bundle code), validate and score.
# One image for CPU and GPU machines: PyTorch's standard Linux wheels bundle
# the CUDA libraries and fall back to CPU when no GPU is passed in. The GPU
# driver comes from the host through the NVIDIA container toolkit.
#
# The image id is part of each node's approved fingerprint. Operators build
# it once and publish it from the hub (docs/OPERATOR.md); providers download
# it instead of building.
#
# Every input is pinned, so one commit names one image: the base image by
# digest, Debian packages by snapshot date, and Python packages by version and
# hash (sandbox/requirements.lock, made from requirements.txt by sandbox/lock.sh).
FROM docker.io/library/python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1
# Digest of the code below (sandbox.SANDBOX_FILES); nodes refuse work when it isn't theirs.
ARG BMB_SANDBOX_STAMP=unstamped
LABEL bmb.sandbox=$BMB_SANDBOX_STAMP

# libgomp: OpenMP runtime for LightGBM/XGBoost; libxrender/libxext: RDKit drawing deps.
ARG DEBIAN_SNAPSHOT=20261015T000000Z
RUN printf 'Types: deb\nURIs: http://snapshot.debian.org/archive/debian/%s\nSuites: trixie trixie-updates\nComponents: main\nSigned-By: /usr/share/keyrings/debian-archive-keyring.pgp\n\nTypes: deb\nURIs: http://snapshot.debian.org/archive/debian-security/%s\nSuites: trixie-security\nComponents: main\nSigned-By: /usr/share/keyrings/debian-archive-keyring.pgp\n' \
      "$DEBIAN_SNAPSHOT" "$DEBIAN_SNAPSHOT" > /etc/apt/sources.list.d/debian.sources \
 && apt-get -o Acquire::Check-Valid-Until=false update \
 && apt-get install -y --no-install-recommends libgomp1 libxrender1 libxext6 \
 && rm -rf /var/lib/apt/lists/*
# The lock lists every package, so nothing is resolved at build time and each wheel is hash-checked.
COPY sandbox/requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir --require-hashes --no-deps -r /tmp/requirements.lock \
 && rm /tmp/requirements.lock

COPY src/bmb_eval /opt/bmb/bmb_eval
ENV PYTHONPATH=/opt/bmb PYTHONDONTWRITEBYTECODE=1 PIP_NO_INDEX=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
USER 65534:65534
WORKDIR /tmp
