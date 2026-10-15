"""Confidential mode: BioModelBench runs the node, the provider keeps the key.

The provider runs `bmb-eval connect-confidential` once per task, on any
machine with their data and Google Cloud access. It checks and splits the
data exactly like `connect`, shows the same approval summary, then seals
everything the node needs into one encrypted bundle in the provider's own
bucket:

    task/      contract.yaml, score.py (what the provider approved)
    data/      inputs_validation/, inputs_test/, answers.parquet, answers_files/
    node.json  hub url, node id and secret (only the sealed node can post results)

The bundle key is wrapped by the provider's Cloud KMS key, which Google
releases only to a Confidential Space workload whose hardware-attested claims
match the provider's rule: production image, debugging off, exactly the
BioModelBench image digest they approved (and, for the GPU preview, the H100
in confidential-computing mode).

BioModelBench runs that image (`bmb-eval confidential-node`) on its own
machine. It unseals the provider's tasks into memory (tmpfs, never disk),
runs the security self-test, then works like any node: pull jobs, sandbox
each stage (bubblewrap), post gated scores. Quotas and final-evaluation
counts are kept in the provider's bucket, encrypted, where only the attested
workload can write. It exits when idle; the VM then stops.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import yaml

PREFIX = "bmb"                    # object prefix in the provider's bucket
POOL_ID = "bmb-eval"              # workload identity pool in the provider's project
PROVIDER_ID = "attestation"
LOCATION = "us-central1"
TOKEN = "/run/container_launcher/attestation_verifier_claims_token"
IDLE_EXIT_S = int(os.environ.get("BMB_IDLE_EXIT_S", "1200"))


# ------------------------------------------------------------- sealing

def encrypt(dek: bytes, plaintext: bytes, aad: str) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    nonce = os.urandom(12)
    return nonce + AESGCM(dek).encrypt(nonce, plaintext, aad.encode())


def decrypt(dek: bytes, blob: bytes, aad: str) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    return AESGCM(dek).decrypt(blob[:12], blob[12:], aad.encode())


def pack(task_dir: Path, data_dir: Path, node: dict) -> bytes:
    """The sealed bundle's plaintext: a gzipped tar of task/, data/ and node.json."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in sorted(task_dir.iterdir()):
            if p.is_file():
                tar.add(p, f"task/{p.name}")
        for name in ("inputs_validation", "inputs_test", "inputs", "answers.parquet", "answers_files"):
            if (data_dir / name).exists():
                tar.add(data_dir / name, f"data/{name}")
        raw = json.dumps(node).encode()
        ti = tarfile.TarInfo("node.json")
        ti.size = len(raw)
        tar.addfile(ti, io.BytesIO(raw))
    return buf.getvalue()


def unpack(plaintext: bytes, task_dest: Path, data_dest: Path) -> dict:
    """Extract a sealed bundle; returns node.json. Rejects links and paths
    outside task/ and data/."""
    node = None
    with tarfile.open(fileobj=io.BytesIO(plaintext), mode="r:gz") as tar:
        for m in tar.getmembers():
            parts = Path(m.name).parts
            if m.name.startswith("/") or ".." in parts or not (m.isfile() or m.isdir()):
                raise ValueError(f"unsafe member in sealed bundle: {m.name!r}")
            if m.name == "node.json":
                node = json.loads(tar.extractfile(m).read())
                continue
            top, rest = parts[0], Path(*parts[1:]) if len(parts) > 1 else Path()
            base = {"task": task_dest, "data": data_dest}.get(top)
            if base is None:
                raise ValueError(f"unexpected member in sealed bundle: {m.name!r}")
            out = base / rest
            if m.isdir():
                out.mkdir(parents=True, exist_ok=True)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(m) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
            os.chmod(out, 0o600)
    if node is None:
        raise ValueError("sealed bundle has no node.json")
    return node


def attestation_condition(digest: str, workload_sa: str, gpu_preview: bool) -> str:
    """The provider's rule: which attested workloads may use their key."""
    support = "'STABLE' in assertion.submods.confidential_space.support_attributes"
    if gpu_preview:
        # Confidential Space on GPUs is a preview ('EXPERIMENTAL' image): accept
        # it only with the H100 attested in confidential-computing mode.
        support = (f"({support} || ('EXPERIMENTAL' in assertion.submods.confidential_space."
                   "support_attributes && assertion.submods.nvidia_gpu.cc_mode == 'ON'))")
    return (f"assertion.swname == 'CONFIDENTIAL_SPACE' && {support} "
            f"&& assertion.dbgstat == 'disabled-since-boot' "
            f"&& assertion.submods.container.image_digest == '{digest}' "
            f"&& '{workload_sa}' in assertion.google_service_accounts")


# ------------------------------------------------------- provider side

class GcloudError(SystemExit):
    pass


def _gcloud(*args: str, project: str | None = None, input: bytes | None = None,
            check: bool = True) -> str:
    cmd = ["gcloud", *args, "--quiet", *(["--project", project] if project else [])]
    r = subprocess.run(cmd, input=input, capture_output=True)
    if check and r.returncode != 0:
        msg = r.stderr.decode(errors="replace").strip().splitlines()
        raise GcloudError("Google Cloud said:\n  " + "\n  ".join(msg[-4:]))
    return r.stdout.decode().strip()


def _exists(*args: str, project: str) -> bool:
    return subprocess.run(["gcloud", *args, "--quiet", "--project", project],
                          capture_output=True).returncode == 0


def check_gcloud() -> None:
    if not shutil.which("gcloud"):
        raise SystemExit("This needs the Google Cloud CLI (gcloud): https://cloud.google.com/sdk/docs/install")
    if subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True).returncode:
        raise SystemExit("Log in to Google Cloud first:  gcloud auth login")


def setup_provider_cloud(project: str, digest: str, workload_sa: str, gpu_preview: bool) -> dict:
    """Key, bucket and attestation rule in the provider's project (idempotent).
    Returns the non-secret pointers the node needs."""
    num = _gcloud("projects", "describe", project, "--format=value(projectNumber)")
    bucket = f"gs://{project}-bmb-data"
    key = f"projects/{project}/locations/{LOCATION}/keyRings/bmb/cryptoKeys/data"
    member = (f"principalSet://iam.googleapis.com/projects/{num}/locations/global/"
              f"workloadIdentityPools/{POOL_ID}/*")
    _gcloud("services", "enable", "cloudkms.googleapis.com", "storage.googleapis.com",
            "iam.googleapis.com", "sts.googleapis.com", project=project)
    if not _exists("kms", "keyrings", "describe", "bmb", "--location", LOCATION, project=project):
        _gcloud("kms", "keyrings", "create", "bmb", "--location", LOCATION, project=project)
    if not _exists("kms", "keys", "describe", "data", "--keyring", "bmb", "--location", LOCATION,
                   project=project):
        _gcloud("kms", "keys", "create", "data", "--keyring", "bmb", "--location", LOCATION,
                "--purpose", "encryption", project=project)
    if not _exists("storage", "buckets", "describe", bucket, project=project):
        _gcloud("storage", "buckets", "create", bucket, "--location", LOCATION,
                "--uniform-bucket-level-access", "--public-access-prevention", project=project)
    if not _exists("iam", "workload-identity-pools", "describe", POOL_ID, "--location", "global",
                   project=project):
        _gcloud("iam", "workload-identity-pools", "create", POOL_ID, "--location", "global",
                "--display-name", "BioModelBench evaluation", project=project)
    verb = ("update-oidc" if _exists("iam", "workload-identity-pools", "providers", "describe",
                                     PROVIDER_ID, "--location", "global",
                                     "--workload-identity-pool", POOL_ID, project=project)
            else "create-oidc")
    _gcloud("iam", "workload-identity-pools", "providers", verb, PROVIDER_ID,
            "--location", "global", "--workload-identity-pool", POOL_ID,
            "--issuer-uri", "https://confidentialcomputing.googleapis.com/",
            "--allowed-audiences", "https://sts.googleapis.com",
            "--attribute-mapping", "google.subject=assertion.sub",
            "--attribute-condition", attestation_condition(digest, workload_sa, gpu_preview),
            project=project)
    # Decrypt the bundle; encrypt + decrypt the node's own state (quotas, final counts).
    _gcloud("kms", "keys", "add-iam-policy-binding", "data", "--keyring", "bmb",
            "--location", LOCATION, "--member", member,
            "--role", "roles/cloudkms.cryptoKeyEncrypterDecrypter", project=project)
    _gcloud("storage", "buckets", "add-iam-policy-binding", bucket, "--member", member,
            "--role", "roles/storage.objectAdmin", project=project)
    return {"PROVIDER_PROJECT_NUMBER": num, "POOL": POOL_ID, "PROVIDER_ID": PROVIDER_ID,
            "KEY": key, "BUCKET": bucket}


def upload_sealed(project: str, key: str, bucket: str, task: str, plaintext: bytes) -> None:
    dek = os.urandom(32)
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        (t / "dek").write_bytes(dek)
        os.chmod(t / "dek", 0o600)
        _gcloud("kms", "encrypt", "--key", key, "--plaintext-file", str(t / "dek"),
                "--ciphertext-file", str(t / "dek.enc"), project=project)
        (t / "bundle.enc").write_bytes(encrypt(dek, plaintext, aad=task))
        _gcloud("storage", "cp", str(t / "bundle.enc"), f"{bucket}/{PREFIX}/{task}.tar.enc",
                project=project)
        _gcloud("storage", "cp", str(t / "dek.enc"), f"{bucket}/{PREFIX}/{task}.dek.enc",
                project=project)


def cmd_connect_confidential(args) -> None:
    from .cli import _ask, _summary, _user_path, data_problems
    from .config import TaskBinding
    from .contract import Contract
    from .node import HubAPI, HubError, parse_connect
    from .prepare import DataError, apply_split, prepare

    home = Path(args.dir).expanduser().resolve()
    home.mkdir(parents=True, exist_ok=True)
    os.chmod(home, 0o700)
    state_path = home / "provider.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    hub_url, code = parse_connect(args.code, args.hub or state.get("hub"))
    project = args.project or state.get("project")
    if not project:
        raise SystemExit("Add --project <your Google Cloud project id>.")
    check_gcloud()

    api = HubAPI(hub_url, state.get("secret"))
    try:
        pkg = api.enrollment(code)
        conf = api._req("GET", "/v1/confidential")
    except HubError as e:
        raise SystemExit(f"Couldn't use that connect link: {e}")
    task = pkg["task"]
    print(f"Connecting to BioModelBench for task '{task}' (sealed: your data stays encrypted)...")
    task_dir = home / "tasks" / task
    task_dir.mkdir(parents=True, exist_ok=True)
    for name, text in pkg["files"].items():
        if "/" in name or name.startswith("."):
            raise SystemExit("unexpected file in task package")
        (task_dir / name).write_text(text)

    data_dir = home / "private" / task
    try:
        contract = Contract.load(task_dir / "contract.yaml")
        inputs_dir, answers, n_rows = prepare(contract, _user_path(args.data), data_dir)
        test_dir = None
        if contract.split:
            inputs_dir, test_dir, n_val, n_test = apply_split(contract, data_dir)
        binding = TaskBinding(task, task_dir, inputs_dir / "samples.parquet", answers, contract,
                              inputs_dir, data_dir / "answers_files"
                              if (data_dir / "answers_files").exists() else None, test_dir)
        if problems := data_problems(binding):
            raise DataError(problems)
    except DataError as e:
        print("\nThere's a problem with your data:")
        for p in e.problems:
            print(f"  • {p}")
        raise SystemExit("\nFix it and run the same command again.")
    print(f"✓ Your data looks good ({n_rows:,} rows)")
    if contract.split:
        print(f"✓ Split at random on this machine: {n_val:,} validation rows (scored during the "
              f"benchmark), {n_test:,} test rows (held out for final evaluations)")

    print("\n" + _summary(contract, n_rows))
    print(f"  Runs on:             BioModelBench's sealed machine (Google Confidential Space, "
          f"{conf.get('hardware', 'NVIDIA H100')}). Your data is encrypted with your own key in\n"
          f"                       your own Google Cloud project, and only this exact software can "
          f"open it:\n                       {conf['digest']}")
    if conf.get("source"):
        repo, _, commit = conf["source"].rpartition("@")
        print(f"                       built by Google Cloud Build from {repo}/tree/{commit}\n"
              f"                       (check it: curl -fsSL {hub_url}/verify-image.sh | sh)")
    print()
    if not args.yes and _ask("Approve, encrypt and upload? [y/N] ") not in ("y", "yes"):
        raise SystemExit("Nothing was uploaded or shared. Run the same command again whenever you're ready.")

    try:
        creds = HubAPI(hub_url, state.get("secret")).activate(code, f"sealed-{project}")
    except HubError as e:
        raise SystemExit(f"Couldn't finish connecting: {e}")
    state.update(hub=hub_url, project=project, node_id=creds["node_id"])
    if "secret" in creds:
        state["secret"] = creds["secret"]
    state.setdefault("tasks", {})[task] = {"code": code, "data": str(_user_path(args.data)),
                                           "version": int(pkg.get("version", 1))}
    state_path.write_text(json.dumps(state, indent=2))
    os.chmod(state_path, 0o600)

    print("  Setting up your key and storage in Google Cloud...")
    ptr = setup_provider_cloud(project, conf["digest"], conf["workload_sa"],
                               bool(conf.get("gpu_preview")))
    print("  Encrypting on this machine and uploading...")
    node = {"hub": hub_url, "node_id": state["node_id"], "secret": state["secret"], "task": task,
            "package_version": int(pkg.get("version", 1))}
    upload_sealed(project, ptr["KEY"], ptr["BUCKET"], task, pack(task_dir, data_dir, node))
    try:
        HubAPI(hub_url, state["secret"])._req("POST", "/v1/node/confidential",
                                              {**ptr, "digest": conf["digest"]})
    except HubError as e:
        raise SystemExit(f"Uploaded, but couldn't tell BioModelBench: {e}. Run the same command again.")
    print(f"\n✓ Connected. Task '{task}' is sealed in {ptr['BUCKET']} and goes live on "
          "BioModelBench's sealed machine.")
    print("  Nothing runs on your side. To stop everything at any time, disable your key:\n"
          f"    gcloud kms keys versions disable 1 --key data --keyring bmb --location {LOCATION} "
          f"--project {project}")


# --------------------------------------------------------- enclave side

def _no_dump() -> None:
    """Make this process non-dumpable: sandboxed code can't read its memory
    through /proc even if it shared a uid."""
    try:
        import ctypes
        ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)   # PR_SET_DUMPABLE, 0
    except Exception:
        pass


def _is_tmpfs(path: Path) -> bool:
    best, fstype = "", ""
    for line in Path("/proc/mounts").read_text().splitlines():
        _, mnt, fs, *_ = line.split()
        if str(path).startswith(mnt) and len(mnt) > len(best):
            best, fstype = mnt, fs
    return fstype == "tmpfs"


def _attested_credentials(env: dict):
    from google.auth import identity_pool
    audience = (f"//iam.googleapis.com/projects/{env['PROVIDER_PROJECT_NUMBER']}/locations/global/"
                f"workloadIdentityPools/{env['POOL']}/providers/{env['PROVIDER_ID']}")
    return identity_pool.Credentials.from_info({
        "type": "external_account", "audience": audience,
        "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
        "token_url": "https://sts.googleapis.com/v1/token",
        "credential_source": {"file": TOKEN},
    }).with_scopes(["https://www.googleapis.com/auth/cloud-platform"])


STATUS: dict = {}


def say(**kv) -> None:
    """The node's only output: fixed status fields, never data. Printed (visible
    only on the debug image, which can't open provider data) and kept in STATUS,
    which the node reports to the hub."""
    kv["at"] = round(time.time())
    STATUS[kv.get("step", "note")] = kv
    print("BMB " + json.dumps(kv), flush=True)


def _report(cfg) -> None:
    """Send STATUS to the hub now (e.g. before refusing work and exiting)."""
    from .node import HubAPI, HubError
    try:
        HubAPI(cfg.hub["url"], cfg.hub["secret"]).hello({}, False, [], STATUS)
    except (HubError, KeyError):
        pass


class SealedState:
    """The node's store (quotas, final counts, undelivered results), kept
    encrypted in the provider's bucket so a restart can't reset it."""

    def __init__(self, bucket, kms, key: str):
        self.bucket, self.kms, self.key = bucket, kms, key
        self.dek = None

    def _dek(self) -> bytes:
        if self.dek is None:
            blob = self.bucket.blob(f"{PREFIX}/state.dek.enc")
            if blob.exists():
                self.dek = self.kms.decrypt(name=self.key, ciphertext=blob.download_as_bytes()).plaintext
            else:
                self.dek = os.urandom(32)
                blob.upload_from_string(self.kms.encrypt(name=self.key, plaintext=self.dek).ciphertext)
        return self.dek

    def load(self, db_path: Path) -> bool:
        blob = self.bucket.blob(f"{PREFIX}/state.enc")
        if not blob.exists():
            return False
        db_path.write_bytes(decrypt(self._dek(), blob.download_as_bytes(), aad="state"))
        return True

    def save(self, store) -> None:
        with store.lock:
            raw = store.db.serialize()
        self.bucket.blob(f"{PREFIX}/state.enc").upload_from_string(encrypt(self._dek(), raw, aad="state"))


def unseal(env: dict, root: Path):
    """Decrypt every sealed task in the provider's bucket into root (tmpfs).
    Returns (node.json, {task: (task_dir, data_dir, node)}, bucket, kms)."""
    from google.cloud import kms as kms_mod, storage
    creds = _attested_credentials(env)
    bucket_name = env["BUCKET"].removeprefix("gs://")
    bucket = storage.Client(project="-", credentials=creds).bucket(bucket_name)
    kms = kms_mod.KeyManagementServiceClient(credentials=creds)
    tasks = {}
    for blob in bucket.list_blobs(prefix=f"{PREFIX}/"):
        if not blob.name.endswith(".tar.enc"):
            continue
        task = blob.name[len(PREFIX) + 1:-len(".tar.enc")]
        wrapped = bucket.blob(f"{PREFIX}/{task}.dek.enc").download_as_bytes()
        dek = kms.decrypt(name=env["KEY"], ciphertext=wrapped).plaintext
        plaintext = decrypt(dek, blob.download_as_bytes(), aad=task)
        task_dir, data_dir = root / "tasks" / task, root / "private" / task
        node = unpack(plaintext, task_dir, data_dir)
        if node.get("task") != task:
            raise ValueError(f"sealed bundle {task} names task {node.get('task')}")
        tasks[task] = (task_dir, data_dir, node)
    return tasks, bucket, kms


def write_config(root: Path, tasks: dict, image: str, blob_dir: Path) -> Path:
    nodes = {(n["hub"], n["node_id"], n["secret"]) for _, _, n in tasks.values()}
    if len(nodes) != 1:
        raise ValueError("sealed tasks disagree about the node identity")
    hub, node_id, secret = nodes.pop()
    raw = {"node_name": "sealed", "state_dir": str(root / "state"), "blob_dir": str(blob_dir),
           "sandbox": {"backend": "bwrap", "image": image}, "retain_bundles": False,
           "hub": {"url": hub, "node_id": node_id, "secret": secret}, "tasks": {}}
    for task, (task_dir, data_dir, node) in tasks.items():
        t = {"dir": str(task_dir), "answers": str(data_dir / "answers.parquet"),
             "package_version": node.get("package_version", 1)}
        val = data_dir / ("inputs_validation" if (data_dir / "inputs_validation").exists() else "inputs")
        t.update(inputs=str(val / "samples.parquet"), inputs_dir=str(val))
        if (data_dir / "inputs_test").exists():
            t["test_inputs_dir"] = str(data_dir / "inputs_test")
        if (data_dir / "answers_files").exists():
            t["answers_dir"] = str(data_dir / "answers_files")
        raw["tasks"][task] = t
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    os.chmod(path, 0o600)
    return path


def _traversable(root: Path) -> None:
    """The sandbox user binds mounted files by path: let it traverse (not list)
    the folders above them. File contents stay root-only until a stage grants them."""
    for d in [root, *[p for p in root.rglob("*") if p.is_dir()]]:
        os.chmod(d, 0o711)


def synthetic_tasks(root: Path) -> dict:
    """A made-up task (random data) for checking the sandbox on this machine
    before any provider's data is involved. Its finals use the GPU if there is one."""
    import numpy as np
    import pandas as pd
    from .contract import Contract
    from .sandbox import gpus_here
    from .scaffold import synth_inputs, write_task
    task_dir = write_task("sealed_selftest", root / "tasks",
                          [("id", "string", False), ("sequence", "string", False)],
                          [("id", "string", False), ("value", "float", False)],
                          ["id"], "regression", "value")
    text = (task_dir / "contract.yaml").read_text()
    text += "split: {validation_fraction: 0.5, final_evaluations: 3}\n"
    if gpus_here():
        text += "final_limits: {gpus: 1}\n"
    (task_dir / "contract.yaml").write_text(text)
    df = synth_inputs(Contract.load(task_dir / "contract.yaml"), 200)
    data_dir = root / "private" / "sealed_selftest"
    for name, part in (("inputs_validation", df.iloc[:100]), ("inputs_test", df.iloc[100:])):
        (data_dir / name).mkdir(parents=True, exist_ok=True)
        part.to_parquet(data_dir / name / "samples.parquet", index=False)
    pd.DataFrame({"id": df["id"], "value": np.random.default_rng(0).normal(size=len(df))}).to_parquet(
        data_dir / "answers.parquet", index=False)
    node = {"hub": "http://unused.invalid", "node_id": "selftest", "secret": "selftest",
            "task": "sealed_selftest"}
    return {"sealed_selftest": (task_dir, data_dir, node)}


def _gpu_diag() -> None:
    """Why CUDA isn't usable inside the sandbox (synthetic self-test only)."""
    from .contract import Limits
    from .sandbox import SANDBOX_UID, BwrapSandbox
    code = ("import os, ctypes, glob\n"
            "print('uid', os.getuid(), 'ld', os.environ.get('LD_LIBRARY_PATH'))\n"
            "print('libcuda', glob.glob('/usr/local/nvidia/lib64/libcuda.so*')[:3])\n"
            "print('proc', os.path.exists('/proc/driver/nvidia/version'))\n"
            "try:\n c = ctypes.CDLL('libcuda.so.1'); n = ctypes.c_int()\n"
            " print('cuInit', c.cuInit(0), 'cuDeviceGetCount', c.cuDeviceGetCount(ctypes.byref(n)), n.value)\n"
            "except Exception as e: print('libcuda error', e)\n"
            "import torch\n"
            "try: torch.cuda.init(); print('torch ok', torch.cuda.device_count())\n"
            "except Exception as e: print('torch error', e)\n")
    for label, as_sandbox in (("sandbox", True), ("node", False)):
        cmd = (BwrapSandbox().cmd(entry="predict", mounts=[], env={}, limits=Limits(), gpus=1)[:-3]
               + ["python3", "-c", code]) if as_sandbox else ["python3", "-c", code]

        def drop():
            os.setgroups([])
            os.setgid(SANDBOX_UID)
            os.setuid(SANDBOX_UID)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                           preexec_fn=drop if as_sandbox else None)
        print(f"BMB-DIAG gpu[{label}] rc={r.returncode}: "
              + (r.stdout + r.stderr)[-1500:].replace("\n", " | "), flush=True)


def cmd_confidential_node(args) -> None:
    from . import approval
    from .cli import _selftest
    from .config import Config
    from .node import HubError, Worker

    _no_dump()
    env = dict(os.environ)
    root = Path(env.get("BMB_ROOT", "/run/tmp/bmb"))
    root.mkdir(parents=True, exist_ok=True)
    if not _is_tmpfs(root):
        say(step="refused", reason="decrypted data would land on disk; BMB_ROOT must be tmpfs")
        sys.exit(2)
    os.chmod(root, 0o711)
    selftest_only = "BUCKET" not in env
    if selftest_only:
        tasks, bucket, kms = synthetic_tasks(root), None, None
    else:
        tasks, bucket, kms = unseal(env, root)
    if not tasks:
        say(step="idle", reason="no sealed tasks in the provider's bucket")
        return
    blob_dir = Path(env.get("BMB_BLOB_DIR", "/var/bmb/blobs"))   # agents' model files: disk is fine
    cfg_path = write_config(root, tasks, env.get("BMB_IMAGE", "confidential"), blob_dir)
    (root / "state").mkdir(exist_ok=True)
    sealed = None if selftest_only else SealedState(bucket, kms, env["KEY"])
    restored = bool(sealed and sealed.load(root / "state" / "node.db"))
    _traversable(root)
    cfg = Config.load(cfg_path)
    say(step="unsealed", tasks=sorted(tasks), state_restored=restored)

    approval.approve(cfg)   # the provider approved by sealing this exact package
    for task in sorted(cfg.tasks):
        failed = _selftest(cfg, task, verbose=True)   # prints attack names and PASS/FAIL only
        say(step="selftest", task=task, failed=failed)
        if failed and selftest_only:
            # Synthetic data only: safe to show why (never done with a provider's data).
            devs = subprocess.run("ls -l /dev/nvidia* /dev/nvidia-caps 2>&1 | head", shell=True,
                                  capture_output=True, text=True).stdout
            print("BMB-DIAG devices: " + devs.replace("\n", " | "), flush=True)
            _gpu_diag()
            for log in sorted((root / "state" / "runs").glob("*/predict.log")):
                tail = [ln for ln in log.read_text().splitlines() if ln.strip()][-6:]
                print(f"BMB-DIAG {log.parent.name}: " + " | ".join(tail)[-1500:], flush=True)
        if failed:
            say(step="refused", reason="security self-test failed; not taking work")
            if not selftest_only:
                _report(cfg)
            sys.exit(3)
    if selftest_only:
        say(step="exit", reason="self-test only (no provider configured)")
        return

    worker = Worker(cfg)
    worker.status = STATUS
    lost = worker.drop_lost_jobs()     # interrupted by a VM replacement: the hub re-runs them
    n = worker.store.requeue_running()
    say(step="running", requeued=n, lost_on_restart=lost)
    idle_since = time.time()
    while True:
        try:
            busy = worker.step()
        except HubError:
            busy = False
        if busy:
            sealed.save(worker.store)
            idle_since = time.time()
        elif time.time() - idle_since > IDLE_EXIT_S:
            sealed.save(worker.store)
            say(step="exit", reason="idle")
            return
        else:
            time.sleep(5)


def add_parsers(sub) -> None:
    p = sub.add_parser("connect-confidential",
                       help="seal a task's data for BioModelBench's sealed machine (no node to run)")
    p.add_argument("code", help="the code from BioModelBench's email")
    p.add_argument("--data", required=True, help="your data file or folder")
    p.add_argument("--project", help="your Google Cloud project id (remembered)")
    p.add_argument("--dir", default=os.environ.get("BMB_CONFIDENTIAL_DIR", "~/bmb-confidential"))
    p.add_argument("--hub")
    p.add_argument("--yes", action="store_true", help=argparse.SUPPRESS)
    p.set_defaults(fn=cmd_connect_confidential)
    p = sub.add_parser("confidential-node", help=argparse.SUPPRESS)   # the sealed VM's entrypoint
    p.set_defaults(fn=cmd_confidential_node)
