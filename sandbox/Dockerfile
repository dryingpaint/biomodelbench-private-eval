# Sandbox image: runs predict (untrusted bundle code), validate and score.
# One image for CPU and GPU machines: PyTorch's standard Linux wheels bundle
# the CUDA libraries and fall back to CPU when no GPU is passed in. The GPU
# driver comes from the host through the NVIDIA container toolkit.
#
# The image id is part of each node's approved fingerprint. Operators build
# it once and publish it from the hub (docs/OPERATOR.md); providers download
# it instead of building.
FROM docker.io/library/python:3.12-slim

# libgomp: OpenMP runtime for LightGBM/XGBoost; libxrender/libxext: RDKit drawing deps.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libxrender1 libxext6 \
 && rm -rf /var/lib/apt/lists/*
COPY sandbox/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY src/bmb_eval /opt/bmb/bmb_eval
ENV PYTHONPATH=/opt/bmb PYTHONDONTWRITEBYTECODE=1 PIP_NO_INDEX=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
USER 65534:65534
WORKDIR /tmp
