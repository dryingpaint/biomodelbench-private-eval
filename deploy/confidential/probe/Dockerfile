FROM python:3.12-slim
ARG GVISOR=20261005
RUN apt-get update && apt-get install -y --no-install-recommends bubblewrap util-linux curl bzip2 ca-certificates \
 && URL=https://storage.googleapis.com/gvisor/releases/release/${GVISOR}/$(uname -m) \
 && curl -fsSLO ${URL}/gvisor.tar.bz2 && curl -fsSLO ${URL}/gvisor.tar.bz2.sha512 && sha512sum -c gvisor.tar.bz2.sha512 \
 && mkdir /tmp/g && tar -xjf gvisor.tar.bz2 -C /tmp/g && find /tmp/g -maxdepth 3 | head -20 \
 && cp -a "$(dirname "$(find /tmp/g -name runsc -type f | head -1)")"/. /usr/local/bin/ \
 && rm -rf /tmp/g gvisor.tar.bz2* && apt-get purge -y curl bzip2 && rm -rf /var/lib/apt/lists/* && runsc --version
RUN pip install --no-cache-dir google-auth google-cloud-kms google-cloud-storage cryptography
COPY probe.py /probe.py
# Confidential Space: let the operator choose where logs go, nothing else.
LABEL "tee.launch_policy.log_redirect"="always"
LABEL "tee.launch_policy.allow_env_override"="PROVIDER_PROJECT_NUMBER,POOL,PROVIDER_ID,KEY,DATA_URI,DEK_URI"
CMD ["python3", "/probe.py"]
