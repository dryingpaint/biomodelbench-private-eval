"""Start sealed nodes on demand (runs inside the hub, `bmb-hub serve --launch-sealed`).

A sealed node (Confidential Space VM, one H100) exits when idle and its VM
stops. When jobs are queued for a sealed node's tasks and the node hasn't
checked in recently, the hub creates its VM again, trying each place in
`PLACES` in order (zone x on-demand/Spot) until one has capacity and quota.
The VM gets the provider's (non-secret) pointers and the approved image; it
has no logs (production Confidential Space image).

Uses the hub VM's service account through the metadata server; it needs
roles/compute.instanceAdmin.v1 and roles/iam.serviceAccountUser on the
sealed workload account (deploy/gcp/hub-setup.sh grants both).
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

PLACES = [  # where a3-highgpu-1g Confidential VMs exist; on-demand first, then Spot
    ("us-central1-a", "STANDARD"), ("us-east5-a", "STANDARD"), ("europe-west4-c", "STANDARD"),
    ("us-central1-a", "SPOT"), ("us-east5-a", "SPOT"), ("europe-west4-c", "SPOT"),
]
OFFLINE_S = 180          # node silent this long = not running
COOLDOWN_S = 20 * 60     # after a launch, give the VM this long to boot and check in
TMPFS_BYTES = 32 << 30
API = "https://compute.googleapis.com/compute/v1"


def _token() -> str:
    req = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())["access_token"]


def _project() -> str:
    req = urllib.request.Request("http://metadata.google.internal/computeMetadata/v1/project/project-id",
                                 headers={"Metadata-Flavor": "Google"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.read().decode()


def _call(method: str, url: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": f"Bearer {_token()}",
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return {"error": json.loads(e.read() or b"{}").get("error", {"code": e.code})}


def _wait(op: dict, timeout: float = 300) -> dict:
    """Follow a zonal operation until DONE; returns its final state."""
    deadline = time.time() + timeout
    while op.get("status") != "DONE" and "error" not in op and time.time() < deadline:
        time.sleep(5)
        op = _call("GET", op["selfLink"])
    return op


def instance_name(node_id: str) -> str:
    return "bmb-sealed-" + node_id.replace("_", "-").lower()


def instance_body(name: str, zone: str, model: str, project: str, conf: dict, pointers: dict) -> dict:
    meta = {"tee-image-reference": conf["image"], "tee-restart-policy": "Never",
            "tee-install-gpu-driver": "true",
            "tee-mount": f"type=tmpfs,source=tmpfs,destination=/run/tmp,size={TMPFS_BYTES}",
            **{f"tee-env-{k}": v for k, v in pointers.items() if k != "digest"}}
    sched = {"onHostMaintenance": "TERMINATE", "provisioningModel": model, "automaticRestart": False}
    if model == "SPOT":
        sched["instanceTerminationAction"] = "DELETE"
    family = "confidential-space-preview-cgpu" if conf.get("gpu_preview") else "confidential-space"
    return {
        "name": name, "machineType": f"zones/{zone}/machineTypes/a3-highgpu-1g",
        "confidentialInstanceConfig": {"confidentialInstanceType": "TDX"},
        "scheduling": sched,
        "shieldedInstanceConfig": {"enableSecureBoot": True},
        "disks": [{"boot": True, "autoDelete": True, "initializeParams": {
            "sourceImage": f"projects/confidential-space-images/global/images/family/{family}",
            "diskSizeGb": "300", "diskType": f"zones/{zone}/diskTypes/pd-balanced"}}],
        "networkInterfaces": [{"network": "global/networks/default",
                               "accessConfigs": [{"type": "ONE_TO_ONE_NAT", "name": "External NAT"}]}],
        "serviceAccounts": [{"email": conf["workload_sa"],
                             "scopes": ["https://www.googleapis.com/auth/cloud-platform"]}],
        "metadata": {"items": [{"key": k, "value": v} for k, v in meta.items()]},
        "labels": {"bmb-role": "sealed-node"},
    }


def launch(node_id: str, pointers: dict, conf: dict, project: str | None = None) -> dict:
    """(Re)create a sealed node's VM wherever there's capacity. Returns {zone, model} or {error}."""
    project = project or _project()
    name = instance_name(node_id)
    errors = []
    for zone, _ in dict.fromkeys(PLACES):               # remove any stopped copy first
        _wait(_call("DELETE", f"{API}/projects/{project}/zones/{zone}/instances/{name}"))
    for zone, model in PLACES:
        op = _call("POST", f"{API}/projects/{project}/zones/{zone}/instances",
                   instance_body(name, zone, model, project, conf, pointers))
        op = _wait(op)
        err = op.get("error")
        if not err:
            return {"zone": zone, "model": model, "name": name}
        msg = json.dumps(err)[:300]
        errors.append(f"{zone}/{model}: {msg}")
        # A failed create can leave a stopped instance that still holds GPU quota:
        # remove it before trying the next place.
        _wait(_call("DELETE", f"{API}/projects/{project}/zones/{zone}/instances/{name}"))
        if not any(k in msg for k in ("QUOTA", "Quota", "STOCKOUT", "stockout", "unavailable",
                                      "does not have enough resources", "ZONE_RESOURCE_POOL_EXHAUSTED")):
            break                                         # not a capacity problem: don't spray zones
    return {"error": errors}


def due(store, now: float | None = None) -> list[tuple[str, dict]]:
    """Sealed nodes with queued work that are offline and not launched recently."""
    now = now or time.time()
    out = []
    for nid, seen, ptr in store.q("SELECT id, last_seen, confidential FROM nodes WHERE confidential IS NOT NULL"):
        queued = store.q("SELECT COUNT(*) FROM submissions s JOIN node_tasks t ON t.task = s.task "
                         "WHERE t.node_id=? AND s.status='queued'", (nid,), one=True)[0]
        last = (store.setting(f"sealed_launch:{nid}") or {}).get("at", 0)
        if queued and (not seen or now - seen > OFFLINE_S) and now - last > COOLDOWN_S:
            out.append((nid, json.loads(ptr)))
    return out


def loop(store, every_s: int = 60) -> threading.Thread:
    def run():
        while True:
            try:
                conf = store.setting("confidential")
                for nid, ptr in (due(store) if conf else []):
                    store.set_setting(f"sealed_launch:{nid}", {"at": time.time(), "state": "launching"})
                    res = launch(nid, ptr, conf)
                    store.set_setting(f"sealed_launch:{nid}", {"at": time.time(), **res})
            except Exception as e:              # never take the hub down
                store.set_setting("sealed_launch:last_error", {"at": time.time(), "error": repr(e)[:300]})
            time.sleep(every_s)
    t = threading.Thread(target=run, daemon=True, name="sealed-launcher")
    t.start()
    return t
