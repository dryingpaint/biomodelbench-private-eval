"""Confidential Space probe (prototype).

Runs as the Confidential Space workload. Proves two things without ever
printing the provider's data:
  1. key release: the provider's KMS key decrypts their file only for this
     attested image (prints row count and a hash, never content);
  2. sandboxing: which ways of isolating a submitted model from the data and
     the network work inside the Confidential Space container.

Config comes from launch metadata (tee-env-*):
  PROVIDER_PROJECT_NUMBER, POOL, PROVIDER_ID, KEY (full KMS key name),
  DATA_URI (gs://bucket/data.csv.enc), DEK_URI (gs://bucket/data.dek.enc)
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import tempfile

TOKEN = "/run/container_launcher/attestation_verifier_claims_token"


def log(**kv) -> None:
    print("BMB-PROBE " + json.dumps(kv), flush=True)


def credentials():
    """Federated credentials: the attestation token, exchanged in the
    provider's workload identity pool. Works only if the provider's policy
    accepts this image's attested claims."""
    from google.auth import identity_pool
    audience = (f"//iam.googleapis.com/projects/{os.environ['PROVIDER_PROJECT_NUMBER']}/locations/global/"
                f"workloadIdentityPools/{os.environ['POOL']}/providers/{os.environ['PROVIDER_ID']}")
    return identity_pool.Credentials.from_info({
        "type": "external_account",
        "audience": audience,
        "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
        "token_url": "https://sts.googleapis.com/v1/token",
        "credential_source": {"file": TOKEN},
    }).with_scopes(["https://www.googleapis.com/auth/cloud-platform"])


def claims() -> dict:
    """The attested claims (unverified decode, for the log only)."""
    try:
        payload = open(TOKEN).read().split(".")[1]
        c = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        sub = c.get("submods", {})
        return {"submods": sorted(sub), "gpu": {k: v for k, v in sub.items() if "gpu" in k.lower()},
                "swname": c.get("swname"), "dbgstat": c.get("dbgstat"), "hwmodel": c.get("hwmodel"),
                "image_digest": sub.get("container", {}).get("image_digest"),
                "support": sub.get("confidential_space", {}).get("support_attributes")}
    except Exception as e:
        return {"error": type(e).__name__}


def fetch(creds, uri: str) -> bytes:
    from google.cloud import storage
    bucket, name = uri.removeprefix("gs://").split("/", 1)
    return storage.Client(project="-", credentials=creds).bucket(bucket).blob(name).download_as_bytes()


def decrypt_data() -> None:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from google.cloud import kms
    try:
        creds = credentials()
        wrapped, blob = fetch(creds, os.environ["DEK_URI"]), fetch(creds, os.environ["DATA_URI"])
        dek = kms.KeyManagementServiceClient(credentials=creds).decrypt(
            name=os.environ["KEY"], ciphertext=wrapped).plaintext
        data = AESGCM(dek).decrypt(blob[:12], blob[12:], None)
        log(step="decrypt", ok=True, bytes=len(data), rows=data.count(b"\n") - 1,
            sha256=hashlib.sha256(data).hexdigest())
    except Exception as e:
        log(step="decrypt", ok=False, error=f"{type(e).__name__}: {str(e)[:300]}")


def try_cmd(name: str, cmd: list[str]) -> None:
    if not shutil.which(cmd[0]):
        return log(step="sandbox", method=name, ok=False, error="not installed")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        log(step="sandbox", method=name, ok=r.returncode == 0, rc=r.returncode,
            out=(r.stdout + r.stderr)[-300:])
    except Exception as e:
        log(step="sandbox", method=name, ok=False, error=f"{type(e).__name__}: {e}")


def probe_sandboxes() -> None:
    # The isolated program tries to reach the network; success = it can't.
    netcheck = ("import socket,sys\ntry:\n socket.create_connection(('1.1.1.1',443),timeout=3)\n"
                " sys.exit('NETWORK OPEN')\nexcept OSError:\n print('no network')")
    root = tempfile.mkdtemp()
    log(step="env", uid=os.getuid(), kernel=os.uname().release,
        caps=open("/proc/self/status").read().split("CapEff:")[1].split()[0],
        userns_max=open("/proc/sys/user/max_user_namespaces").read().strip()
        if os.path.exists("/proc/sys/user/max_user_namespaces") else None)
    try_cmd("unshare-user-net", ["unshare", "-Urn", "python3", "-c", netcheck])
    try_cmd("bubblewrap", ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--ro-bind", "/proc", "/proc",
                           "--unshare-user", "--unshare-net", "--unshare-ipc", "--unshare-uts",
                           "--die-with-parent", "python3", "-c", netcheck])
    try_cmd("gvisor-systrap", ["runsc", "--rootless", "--network=none", "--ignore-cgroups",
                               f"--root={root}/runsc1", "do", "python3", "-c", netcheck])
    try_cmd("gvisor-ptrace", ["runsc", "--rootless", "--network=none", "--ignore-cgroups",
                              "--platform=ptrace", f"--root={root}/runsc2", "do", "python3", "-c", netcheck])
    # Can a sandboxed program read the decrypted data? It must not (only the scorer may).
    secret = "/run/bmb-secret"
    open(secret, "w").write("x")
    os.chmod(secret, 0o600)
    readcheck = f"import sys\ntry:\n open('{secret}').read()\n sys.exit('DATA READABLE')\nexcept OSError:\n print('data hidden')"
    try_cmd("unshare-data-hidden", ["unshare", "-Urn", "--map-user=65534", "python3", "-c", readcheck])
    smi = shutil.which("nvidia-smi") or next((p for p in ("/usr/local/nvidia/bin/nvidia-smi",
                                                            "/var/lib/nvidia/bin/nvidia-smi")
                                               if os.path.exists(p)), "nvidia-smi")
    os.environ["LD_LIBRARY_PATH"] = "/usr/local/nvidia/lib64"   # where Confidential Space mounts the driver
    log(step="gpu-env", smi=smi, devices=sorted(d for d in os.listdir("/dev") if d.startswith("nvidia")),
        ld=os.environ.get("LD_LIBRARY_PATH"))
    try_cmd("nvidia-smi", [smi, "-L"])
    try_cmd("gpu-confidential-mode", [smi, "conf-compute", "-f"])
    try_cmd("gpu-ready-state", [smi, "conf-compute", "-grs"])
    # The model must reach the GPU from inside the sandbox (gVisor's nvproxy).
    try_cmd("gvisor-nvproxy", ["runsc", "--rootless", "--network=none", "--ignore-cgroups", "--nvproxy",
                               "--nvproxy-docker", f"--root={root}/runsc4", "do", smi, "-L"])


if __name__ == "__main__":
    log(step="start", claims=claims())
    decrypt_data()
    probe_sandboxes()
    log(step="done")
