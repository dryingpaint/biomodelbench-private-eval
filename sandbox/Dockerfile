# Sandbox image: runs predict (untrusted bundle code), validate and score.
# Rebuild with `bmb-eval build-image`, then `bmb-eval approve`: the image id
# is part of the approved fingerprint.
#
# Packages here are what bundles can import without shipping wheels. Agents
# can add pure-Python or binary wheels in their bundle's wheels/ directory.
FROM docker.io/library/python:3.12-slim

# CPU-only torch from the PyTorch index (much smaller than the CUDA build).
COPY sandbox/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && rm /tmp/requirements.txt

COPY src/bmb_eval /opt/bmb/bmb_eval
ENV PYTHONPATH=/opt/bmb PYTHONDONTWRITEBYTECODE=1 PIP_NO_INDEX=1
USER 65534:65534
WORKDIR /tmp
