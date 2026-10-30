# Sandbox image for the Modal backend (sandbox.backend: modal, src/bmb_eval/modal_backend.py).
#
# The same stages and entrypoints as sandbox/Dockerfile, plus what the genomic tasks
# need: Python 3.11, tabix and samtools, PyTorch 2.5.1 (CUDA 12.4), transformers
# 4.46.3, pysam, pyfaidx, pyBigWig and biopython (sandbox/modal/requirements.txt).
# A separate image, so the container nodes' image and its stamp don't change.
#
# Built in the evaluation Modal environment by `bmb-eval modal-deploy`. The id
# Modal gives the built image is part of the Modal node's approved fingerprint.
#
# No USER line: on Modal the trusted launcher (bmb_eval.modal_entry) starts as
# root, then runs each stage as uid 65534 with no capabilities, no new privileges,
# and process, file and memory limits.
FROM docker.io/library/python:3.11-slim-bookworm@sha256:94a8fee35252d06ab75856f15f12c7bae67b10e8a6da93c7fded82b2610926cd

ARG DEBIAN_SNAPSHOT=20261015T000000Z
RUN printf 'Types: deb\nURIs: http://snapshot.debian.org/archive/debian/%s\nSuites: bookworm bookworm-updates\nComponents: main\nSigned-By: /usr/share/keyrings/debian-archive-keyring.gpg\n\nTypes: deb\nURIs: http://snapshot.debian.org/archive/debian-security/%s\nSuites: bookworm-security\nComponents: main\nSigned-By: /usr/share/keyrings/debian-archive-keyring.gpg\n' \
      "$DEBIAN_SNAPSHOT" "$DEBIAN_SNAPSHOT" > /etc/apt/sources.list.d/debian.sources \
 && apt-get -o Acquire::Check-Valid-Until=false update \
 && apt-get install -y --no-install-recommends libgomp1 libxrender1 libxext6 tabix samtools \
 && rm -rf /var/lib/apt/lists/*
COPY sandbox/modal/requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir --require-hashes --no-deps -r /tmp/requirements.lock \
 && rm /tmp/requirements.lock

COPY src/bmb_eval /opt/bmb/bmb_eval
ENV PYTHONPATH=/opt/bmb PYTHONDONTWRITEBYTECODE=1 PIP_NO_INDEX=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
WORKDIR /tmp
