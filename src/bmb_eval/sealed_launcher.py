"""Start sealed nodes on demand (runs inside the hub, `bmb-hub serve --launch-sealed`).

A sealed node (Confidential Space VM, one H100) exits when idle and its VM
stops. When jobs are queued for a sealed node's tasks and the node hasn't
checked in recently, the hub creates its VM again.

Confidential a3-highgpu-1g VMs come only as Flex-start or Spot (no on-demand).
Flex-start comes first: the request waits for capacity (up to
`FLEX_WAIT_S`), and once running the VM isn't preempted (up to
`MAX_RUN_S`). Spot is the fallback; if it's reclaimed mid-job, the hub hands
the job out again and the node, restarted, drops what it lost. Attempts
rotate through `PLACES`, starting after the last place tried.
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

PLACES = [  # where a3-highgpu-1g Confidential VMs exist: Flex-start first, then Spot
    ("us-central1-a", "FLEX_START"), ("us-east5-a", "FLEX_START"), ("europe-west4-c", "FLEX_START"),
    ("us-central1-a", "SPOT"), ("us-east5-a", "SPOT"), ("europe-west4-c", "SPOT"),
]
OFFLINE_S = 180          # node silent this long = not running
COOLDOWN_S = 20 * 60     # after a launch, give the VM this long to boot and check in
RETRY_S = 5 * 60         # after a failed launch (no capacity anywhere), try again this soon
FLEX_WAIT_S = 20 * 60    # how long a Flex-start request waits for capacity
MAX_RUN_S = 3 * 24 * 3600   # Flex-start VMs end after this (the node normally exits when idle)
LIVE = ("PROVISIONING", "STAGING", "RUNNING", "PENDING", "REPAIRING")
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
    sched = {"onHostMaintenance": "TERMINATE", "provisioningModel": model, "automaticRestart": False,
             "instanceTerminationAction": "DELETE"}
    extra = {}
    if model == "FLEX_START":
        sched["maxRunDuration"] = {"seconds": str(MAX_RUN_S)}
        extra["params"] = {"requestValidForDuration": {"seconds": str(FLEX_WAIT_S)}}
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
        **extra,
    }


def launch(node_id: str, pointers: dict, conf: dict, project: str | None = None,
           start_at: int = 0) -> dict:
    """(Re)create a sealed node's VM wherever there's capacity, trying PLACES from
    `start_at`. Returns {zone, model, place} or {error}, or {state: "waiting"} if
    a VM (or a Flex-start request still waiting for capacity) already exists."""
    project = project or _project()
    name = instance_name(node_id)
    errors = []
    for zone in dict.fromkeys(z for z, _ in PLACES):
        inst = _call("GET", f"{API}/projects/{project}/zones/{zone}/instances/{name}")
        if inst.get("status") in LIVE:
            return {"state": "waiting", "zone": zone, "status": inst["status"]}
        if "error" not in inst:                            # a stopped copy: replace it
            _wait(_call("DELETE", f"{API}/projects/{project}/zones/{zone}/instances/{name}"))
    order = PLACES[start_at % len(PLACES):] + PLACES[:start_at % len(PLACES)]
    last = start_at % len(PLACES)
    for zone, model in order:
        last = PLACES.index((zone, model))
        op = _call("POST", f"{API}/projects/{project}/zones/{zone}/instances",
                   instance_body(name, zone, model, project, conf, {**pointers, "BMB_NODE_ID": node_id}))
        op = _wait(op)
        err = op.get("error")
        if not err:
            return {"zone": zone, "model": model, "name": name, "place": PLACES.index((zone, model))}
        msg = json.dumps(err)[:300]
        errors.append(f"{zone}/{model}: {msg}")
        # A failed create can leave a stopped instance that still holds GPU quota:
        # remove it before trying the next place.
        _wait(_call("DELETE", f"{API}/projects/{project}/zones/{zone}/instances/{name}"))
        if not any(k in msg for k in ("QUOTA", "Quota", "STOCKOUT", "stockout", "unavailable",
                                      "does not have enough resources", "ZONE_RESOURCE_POOL_EXHAUSTED")):
            break                                         # not a capacity problem: don't spray zones
    return {"error": errors, "place": last}


def remove_stopped(store, now: float | None = None, project: str | None = None) -> int:
    """Delete the VMs of sealed nodes that exited (idle, done or refused). A stopped VM
    keeps its 300 GB boot disk, which counts against the region's SSD quota, so a few
    of them block every other sealed node from starting. The disk holds only cached
    model files; the node's state is in the provider's bucket."""
    now = now or time.time()
    n = 0
    for (nid,) in store.q("SELECT id FROM nodes WHERE confidential IS NOT NULL AND "
                          "(last_seen IS NULL OR last_seen < ?)", (now - OFFLINE_S,)):
        rec = store.setting(f"sealed_launch:{nid}") or {}
        if not rec.get("zone") or rec.get("removed") or now - rec.get("at", 0) < COOLDOWN_S:
            continue
        project = project or _project()
        url = f"{API}/projects/{project}/zones/{rec['zone']}/instances/{instance_name(nid)}"
        inst = _call("GET", url)
        status = inst.get("status")
        if status in LIVE or (not status and inst.get("error", {}).get("code") != 404):
            continue                                   # running, or the API call failed: later
        if status:                                     # TERMINATED, STOPPED, SUSPENDED...
            _wait(_call("DELETE", url))
            n += 1
        store.set_setting(f"sealed_launch:{nid}", {**rec, "removed": now})
    return n


def reclaim(store, now: float | None = None) -> int:
    """Jobs held by a sealed node that has gone silent (its VM was preempted or
    died): back to the queue, so they run again when the node restarts."""
    now = now or time.time()
    n = 0
    for (nid,) in store.q("SELECT id FROM nodes WHERE confidential IS NOT NULL AND "
                          "(last_seen IS NULL OR last_seen < ?)", (now - OFFLINE_S,)):
        held = store.q("SELECT COUNT(*) FROM submissions WHERE node_id=? AND status='assigned'",
                       (nid,), one=True)[0]
        if held:
            store.q("UPDATE submissions SET status='queued', node_id=NULL "
                    "WHERE node_id=? AND status='assigned'", (nid,))
            n += held
    return n


def due(store, now: float | None = None, digest: str | None = None) -> list[tuple[str, dict]]:
    """Sealed nodes with queued work that are offline and not launched recently.
    A node that refused work (e.g. its self-test failed) on the current image
    isn't relaunched: it would only refuse again, on an H100, every cooldown."""
    now = now or time.time()
    out = []
    for nid, seen, ptr, st in store.q("SELECT id, last_seen, confidential, status FROM nodes "
                                      "WHERE confidential IS NOT NULL"):
        queued = store.q("SELECT COUNT(*) FROM submissions s JOIN node_tasks t ON t.task = s.task "
                         "WHERE t.node_id=? AND s.status='queued'", (nid,), one=True)[0]
        rec = store.setting(f"sealed_launch:{nid}") or {}
        wait = RETRY_S if rec.get("error") else COOLDOWN_S
        if "refused" in json.loads(st or "{}") and digest and rec.get("digest") == digest:
            continue
        if queued and (not seen or now - seen > OFFLINE_S) and now - rec.get("at", 0) > wait:
            out.append((nid, json.loads(ptr)))
    return out


def loop(store, every_s: int = 60) -> threading.Thread:
    def run():
        while True:
            try:
                conf = store.setting("confidential")
                if conf:
                    reclaim(store)
                    remove_stopped(store)
                for nid, ptr in (due(store, digest=conf.get("digest")) if conf else []):
                    prev = store.setting(f"sealed_launch:{nid}") or {}
                    store.set_setting(f"sealed_launch:{nid}", {**prev, "at": time.time(), "state": "launching"})
                    # Reuse the place that last worked; after a failure, start after it.
                    start = prev.get("place", 0) + (1 if prev.get("error") else 0)
                    res = launch(nid, ptr, conf, start_at=start)
                    store.set_setting(f"sealed_launch:{nid}", {"at": time.time(), **res,
                                                               "digest": conf.get("digest")})
            except Exception as e:              # never take the hub down
                store.set_setting("sealed_launch:last_error", {"at": time.time(), "error": repr(e)[:300]})
            time.sleep(every_s)
    t = threading.Thread(target=run, daemon=True, name="sealed-launcher")
    t.start()
    return t
