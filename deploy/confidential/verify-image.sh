#!/bin/sh
# Check the image BioModelBench asks you to approve: Google Cloud Build signed
# that it built exactly this digest from the public source commit the hub names.
# Then read that code: it's what your key will open for.
#   curl -fsSL https://eval.biomodelbench.org/verify-image.sh | sh
# Needs python3 and openssl.
#
# Google's signing key ("google-hosted-worker", verified-builder project). The
# same key ships with slsa-verifier (verifiers/internal/gcb/keys); fetch it
# yourself to cross-check:
#   gcloud kms keys versions get-public-key 1 --location global --keyring attestor \
#     --key google-hosted-worker --project verified-builder
set -eu
HUB="${BMB_HUB:-https://eval.biomodelbench.org}"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
cat > "$TMP/google.pem" <<'KEY'
-----BEGIN PUBLIC KEY-----
MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEg9KII7kzr/30HBluf00y9WwtMFkE
qc3oCcFVH3QJ37IBLUv/MUApbnNHFfD75ayJ/a0F45xa+MLv5zoep+GxsA==
-----END PUBLIC KEY-----
KEY
curl -fsSL "$HUB/v1/confidential" -o "$TMP/conf.json"
curl -fsSL "$HUB/v1/confidential/provenance" -o "$TMP/provenance.json"
python3 - "$TMP" <<'PY'
import base64, json, subprocess, sys
from pathlib import Path
tmp = Path(sys.argv[1])
conf = json.loads((tmp / "conf.json").read_text())
digest, source = conf["digest"], conf.get("source") or ""
repo, _, commit = source.rpartition("@")
if not commit:
    sys.exit("The hub didn't publish the source commit for this image.")
KEYID = "projects/verified-builder/locations/global/keyRings/attestor/cryptoKeys/google-hosted-worker/cryptoKeyVersions/1"
prov = json.loads((tmp / "provenance.json").read_text())
for entry in prov["provenance_summary"]["provenance"]:
    env = entry["envelope"]
    body = base64.b64decode(env["payload"])
    statement = json.loads(body)
    if statement.get("predicateType") != "https://slsa.dev/provenance/v1":
        continue
    sig = next((s for s in env["signatures"] if s.get("keyid") == KEYID), None)
    if not sig:
        continue
    ptype = env["payloadType"].encode()
    pae = b"DSSEv1 %d %s %d %s" % (len(ptype), ptype, len(body), body)   # DSSE signing input
    (tmp / "pae").write_bytes(pae)
    (tmp / "sig").write_bytes(base64.b64decode(sig["sig"].replace("-", "+").replace("_", "/") + "=="))
    r = subprocess.run(["openssl", "dgst", "-sha256", "-verify", str(tmp / "google.pem"),
                        "-signature", str(tmp / "sig"), str(tmp / "pae")], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("FAILED: the provenance signature is not Google Cloud Build's.")
    subjects = {"sha256:" + s["digest"]["sha256"] for s in statement["subject"]}
    if digest not in subjects:
        sys.exit(f"FAILED: the signed provenance is for {sorted(subjects)}, not {digest}.")
    deps = statement["predicate"]["buildDefinition"].get("resolvedDependencies", [])
    if not any(d.get("uri") == f"git+{repo}" and d.get("digest", {}).get("gitCommit") == commit
               for d in deps):
        sys.exit(f"FAILED: the signed provenance doesn't name {repo} at {commit}.")
    print(f"Verified: Google Cloud Build signed that {digest}\n"
          f"was built from {repo} at commit {commit}.\n"
          f"Read that code: {repo}/tree/{commit}\n"
          f"  sealed node:  src/bmb_eval/confidential.py\n"
          f"  sandbox:      src/bmb_eval/sandbox.py\n"
          f"  image:        deploy/confidential/Dockerfile")
    break
else:
    sys.exit("FAILED: no Google-signed SLSA v1 provenance for this image.")
PY
