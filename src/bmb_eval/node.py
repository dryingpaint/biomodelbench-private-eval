"""Provider node: polls the BioModelBench hub for work over outbound HTTPS.

No listening port, no TLS certificate, no firewall rule. The node decides
everything that protects the data locally: quotas, approval lock, pause
switch, and when a result may be released. The hub only ever receives the
gated payload, and only once its release time has passed.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import urllib.error
import urllib.request

from . import alerts, approval, audit
from .blobs import BlobCache, check_manifest
from .config import Config
from .errors import EvalError
from .runner import Runner
from .sandbox import gpus_here, stale_image_message
from .store import Store

POLL_S = 5
HELLO_S = 60
HUB_ALERT_AFTER = 60   # consecutive failed hub calls (~5 minutes) before alerting


class HubError(Exception):
    """A hub call failed: `status` is the HTTP status the hub answered with, or None
    when it couldn't be reached."""

    def __init__(self, msg: str, status: int | None = None):
        super().__init__(msg)
        self.status = status


def parse_connect(link: str, hub: str | None = None) -> tuple[str, str]:
    """'https://hub/c/ABCD-EFGH-JKLM', or a bare 'ABCD-EFGH-JKLM' plus the hub
    url (install.sh passes it) -> (hub url, code)."""
    link = link.strip()
    base, sep, code = link.rpartition("/c/")
    if not sep and hub:
        base, code = hub, link
    if not base.startswith(("http://", "https://")) or len(code) != 14:
        raise SystemExit("That doesn't look like a connect code. Copy the command from "
                         "the BioModelBench email exactly.")
    return base.rstrip("/"), code.upper()


class HubAPI:
    def __init__(self, url: str, secret: str | None = None, timeout: float = 60):
        self.url = url.rstrip("/")
        self.secret = secret
        self.timeout = timeout

    def _req(self, method: str, path: str, body: dict | None = None, raw: bool = False,
             cap: int | None = None):
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"}
        if self.secret:
            headers["Authorization"] = f"Bearer {self.secret}"
        req = urllib.request.Request(self.url + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                if r.status == 204:
                    return None
                if raw:
                    out = r.read(cap + 1 if cap else -1)
                    if cap and len(out) > cap:
                        raise EvalError("bundle_too_large")
                    return out
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read())["error"]["message"]
            except Exception:
                msg = f"HTTP {e.code}"
            raise HubError(msg, e.code)
        except (urllib.error.URLError, OSError) as e:
            raise HubError(f"cannot reach {self.url}: {e}")

    # enrollment
    def enrollment(self, code: str) -> dict:
        return self._req("GET", f"/v1/enroll/{code}")

    def activate(self, code: str, name: str) -> dict:
        return self._req("POST", f"/v1/enroll/{code}", {"name": name})

    # work
    def hello(self, tasks: dict, accepting: bool, held: list[str], status: dict | None = None) -> dict:
        body = {"tasks": tasks, "accepting": accepting, "held": held}
        if status is not None:
            body["status"] = status      # fixed fields only (sealed node; it has no logs)
        return self._req("POST", "/v1/node/hello", body) or {}

    def next(self) -> dict | None:
        return self._req("GET", "/v1/node/next")

    def download(self, info: dict, cache: BlobCache, sha: str, size: int) -> None:
        """Fetch one model file (signed storage URL, or the hub itself) into the cache."""
        headers = dict(info.get("headers") or {})
        if info.get("auth"):
            headers["Authorization"] = f"Bearer {self.secret}"
        req = urllib.request.Request(info["url"], headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                cache.put(sha, size, r.read)
        except (urllib.error.URLError, OSError) as e:
            raise HubError(f"download failed: {e}")

    def result(self, sid: str, payload: dict) -> None:
        self._req("POST", f"/v1/node/results/{sid}", payload)


def mem_bytes(spec: str | int) -> int:
    """A contract's memory limit ("24g", "512m") in bytes."""
    s = str(spec).strip().lower()
    units = {"k": 1 << 10, "m": 1 << 20, "g": 1 << 30, "t": 1 << 40}
    return int(float(s[:-1]) * units[s[-1]]) if s and s[-1] in units else int(float(s or 0))


def machine_capacity() -> tuple[float, int, int]:
    """This machine's CPUs, memory (bytes) and GPUs."""
    try:
        mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        mem = 0
    return float(os.cpu_count() or 1), int(mem), len(gpus_here())


def paused_path(cfg: Config):
    return cfg.state_dir / "PAUSED"


def updates_path(cfg: Config):
    return cfg.state_dir / "updates.json"


def pending_updates(cfg: Config) -> dict[str, int]:
    """Task updates BioModelBench has published that the provider hasn't approved."""
    try:
        return json.loads(updates_path(cfg).read_text())
    except (OSError, ValueError):
        return {}


class Worker:
    def __init__(self, cfg: Config, api: HubAPI | None = None):
        if not cfg.hub and api is None:
            raise SystemExit("This node is not connected yet. Run the connect command we sent you.")
        self.cfg = cfg
        self.api = api or HubAPI(cfg.hub["url"], cfg.hub["secret"])
        self.store = Store(cfg.state_dir)
        self.runner = Runner(cfg, store=self.store)
        self.queue_dir = cfg.state_dir / "queue"      # one <id>.json per job held locally
        self.queue_dir.mkdir(exist_ok=True)
        self.cache = BlobCache(cfg.blob_dir)
        self._last_hello = 0.0
        self._hub_failures = 0
        self._unapproved_alerted = False
        self._accepting = (0.0, False)
        self._hello_rejected: str | None = None
        self._stale_alerted = False
        self.status: dict | None = None   # set by the sealed node: reported to the hub
        # Evaluations in flight (max_parallel > 1): sid -> (thread, cpus, memory bytes, gpus).
        self._jobs: dict[str, tuple[threading.Thread, float, int, int]] = {}
        self.parallel = 1 if cfg.sealed else cfg.max_parallel
        self.capacity = machine_capacity()

    # ------------------------------------------------------------- gates

    def accepting(self) -> bool:
        if paused_path(self.cfg).exists():
            return False
        checked, ok = self._accepting
        if time.time() - checked < 30:
            return ok
        if stale := stale_image_message(self.cfg):
            if not self._stale_alerted:
                audit.record(self.cfg.state_dir, "sandbox_stale")
                alerts.send(self.cfg, "infra", stale)
                self._stale_alerted = True
            self._accepting = (time.time(), False)
            return False
        self._stale_alerted = False
        drift = approval.diff(self.cfg)
        self._accepting = (time.time(), not drift)
        if drift and not self._unapproved_alerted:
            alerts.send(self.cfg, "config", "configuration differs from what you approved; "
                        "no new work until you run `bmb-eval approve`", changes=len(drift))
        self._unapproved_alerted = bool(drift)
        return not drift

    # -------------------------------------------------------------- loop

    def step(self) -> bool:
        """One iteration. Returns True if it did anything."""
        did = False
        accepting = self.accepting()
        if time.time() - self._last_hello > HELLO_S:
            # Held = anything not yet delivered, including finished results the
            # hub hasn't received; otherwise the hub would re-queue them.
            held = sorted({p.stem for p in self.queue_dir.glob("*.json")} | set(self.store.unposted()))
            try:
                reply = self.api.hello({t: b.contract.public() for t, b in self.cfg.tasks.items()},
                                       accepting, held, self.status)
                self._note_updates(reply.get("versions") or {})
                if self._hello_rejected:
                    audit.record(self.cfg.state_dir, "hello_accepted")
                    self._hello_rejected = None
            except HubError as e:
                if e.status is None or e.status >= 500:
                    raise          # hub unreachable: retry the whole step later
                # The hub refused the hello itself (e.g. too large). Keep evaluating:
                # jobs and results use other calls. But the hub's view of this node
                # (new contracts, accepting) is now stale, so tell the operator once.
                if self._hello_rejected != str(e):
                    audit.record(self.cfg.state_dir, "hello_rejected", status=e.status, error=str(e)[:200])
                    alerts.send(self.cfg, "infra", f"the hub rejected this node's check-in (HTTP {e.status}: "
                                f"{e}); evaluations continue, but task changes won't register until "
                                "BioModelBench fixes it", status=e.status)
                    self._hello_rejected = str(e)
            self._last_hello = time.time()
        for sid, payload in self.store.released_unposted():
            try:
                self.api.result(sid, payload)
            except HubError as e:
                if "No such job" not in str(e):
                    raise          # hub unreachable: retry the whole step later
                # The hub no longer knows this job (e.g. it was deleted); don't
                # let one undeliverable result block all other work.
                audit.record(self.cfg.state_dir, "result_dropped", run_id=sid, error=str(e)[:200])
            self.store.mark_posted(sid)
            did = True
        did = self._reap() or did
        if not self._room_for_any():
            return did
        # Finish anything interrupted by a restart before taking new work.
        local = self.store.next_queued()
        if local:
            self._dispatch(*local, self._need(local[0], local[1]))
            return True
        if not accepting:
            return did
        job = self.api.next()
        if job:
            self._take(job)
            return True
        return did

    # ------------------------------------------------------ parallel jobs

    def _need(self, sid: str, task: str) -> tuple[float, int, int]:
        """The CPUs, memory (bytes) and GPUs a job's contract allows it."""
        t = self.cfg.tasks.get(task)
        try:
            split = json.loads((self.queue_dir / f"{sid}.json").read_text()).get("split", "validation")
        except (OSError, ValueError):
            split = "validation"
        if t is None:
            return (1.0, 0, 0)
        lim = t.contract.limits_for(split, sealed=self.cfg.sealed)
        return (float(lim.cpus), mem_bytes(lim.memory), int(lim.gpus))

    def _free(self) -> tuple[float, int, int]:
        cpus, mem, gpus = self.capacity
        for _, c, m, g in self._jobs.values():
            cpus, mem, gpus = cpus - c, mem - m, gpus - g
        return cpus, mem, gpus

    def _fits(self, need: tuple[float, int, int]) -> bool:
        if not self._jobs:
            return True              # always run one job, whatever this machine is
        c, m, g = self._free()
        gpu_busy = any(j[3] for j in self._jobs.values())
        # The sandbox gives a GPU job every GPU, so GPU jobs never share.
        return need[0] <= c and need[1] <= m and (not need[2] or (not gpu_busy and need[2] <= g))

    def _room_for_any(self) -> bool:
        """Whether a job of any connected task could start now: taken from the hub only
        then, so a job never waits here for resources once it is ours."""
        if self.parallel <= 1 or not self._jobs:
            return not self._jobs
        if len(self._jobs) >= self.parallel:
            return False
        needs = [self._need("", tid) for tid in self.cfg.tasks] or [(1.0, 0, 0)]
        return self._fits(tuple(max(n[i] for n in needs) for i in range(3)))

    def _dispatch(self, sid: str, task: str, submitter: str, need: tuple[float, int, int]) -> None:
        if self.parallel <= 1:
            self._run(sid, task, submitter)
            self.cache.evict(self.cfg.cache_gb << 30)
            return
        th = threading.Thread(target=self._run, args=(sid, task, submitter), daemon=True,
                              name=f"eval-{sid}")
        self._jobs[sid] = (th, *need)
        th.start()

    def _reap(self) -> bool:
        done = [sid for sid, j in self._jobs.items() if not j[0].is_alive()]
        for sid in done:
            self._jobs.pop(sid)
        if done and not self._jobs:
            # Evict only when nothing runs: a job assembling its bundle needs its files cached.
            self.cache.evict(self.cfg.cache_gb << 30)
        return bool(done)

    def wait_jobs(self, timeout: float | None = None) -> None:
        for th, *_ in list(self._jobs.values()):
            th.join(timeout)
        self._reap()

    def _note_updates(self, versions: dict) -> None:
        """BioModelBench changed a task: tell the provider (once per version).
        Nothing changes here until they review and approve it with `bmb-eval update`."""
        pending = pending_updates(self.cfg)
        for task, version in versions.items():
            have = int((self.cfg.raw.get("tasks", {}).get(task) or {}).get("package_version", 1))
            if version > have and pending.get(task) != version:
                pending[task] = version
                audit.record(self.cfg.state_dir, "task_update_available", task=task, version=version)
                alerts.send(self.cfg, "config", f"BioModelBench updated task {task} (version {version}); "
                            "run `bmb-eval update` to review and approve it", task=task)
        updates_path(self.cfg).write_text(json.dumps(pending))

    def drop_lost_jobs(self) -> int:
        """Jobs the store says are unfinished but whose job file is gone (a sealed
        node's VM was replaced, e.g. Spot preemption: its queue lived in memory).
        Forgetting them, rather than failing them, lets the hub re-run them
        without using up a quota slot or a final evaluation."""
        lost = [sid for sid in self.store.unfinished() if not (self.queue_dir / f"{sid}.json").exists()]
        for sid in lost:
            self.store.forget(sid)
            audit.record(self.cfg.state_dir, "job_lost", run_id=sid)
        return len(lost)

    def run_forever(self) -> None:
        self.drop_lost_jobs()
        n = self.store.requeue_running()
        audit.record(self.cfg.state_dir, "node_start", requeued=n)
        print(f"bmb-eval node running; connected to {self.api.url}. Ctrl-C to stop.")
        while True:
            try:
                busy = self.step()
                self._hub_failures = 0
            except HubError as e:
                self._hub_failures += 1
                audit.record(self.cfg.state_dir, "hub_unreachable", error=str(e)[:200])
                if self._hub_failures == HUB_ALERT_AFTER:
                    alerts.send(self.cfg, "infra", "cannot reach the BioModelBench hub",
                                failures=self._hub_failures)
                busy = False
            if not busy:
                time.sleep(POLL_S)

    def drain(self, max_s: float = 300) -> None:
        """Process until idle (tests and one-shot use)."""
        deadline = time.time() + max_s
        self.store.requeue_running()
        while time.time() < deadline:
            if not self.step() and not self.store.released_unposted() and not self._jobs:
                break
            if self._jobs:
                self.wait_jobs(timeout=0.05)

    # -------------------------------------------------------------- jobs

    def _take(self, job: dict) -> None:
        sid, task, submitter = job["submission_id"], job["task"], job["submitter"]
        if self.store.has(sid):
            return
        t = self.cfg.tasks.get(task)
        split = job.get("split", "validation")
        lim = t.contract.limits_for(split, sealed=self.cfg.sealed) if t else None
        refusal, files = None, []
        if t is None or split not in ("validation", "test"):
            refusal = EvalError("task_unknown")
        elif split == "test" and not (t.contract.split and t.test_inputs_dir):
            refusal = EvalError("no_test_split")
        elif split == "test" and self.store.count_final(task) >= t.contract.split.final_evaluations:
            # Enforced here, not at the hub: even a compromised hub can't run the test set again.
            refusal = EvalError("final_quota_exceeded")
        elif split == "test" and lim.gpus and len(gpus_here()) < lim.gpus:
            # Refused before it counts, so the final can run once GPUs are added.
            refusal = EvalError("final_hardware_missing")
        elif split == "validation" and (
                self.store.count_today(submitter, task) >= lim.submissions_per_day or (
                lim.submissions_total and self.store.count_total(task) >= lim.submissions_total)):
            refusal = EvalError("quota_exceeded")
        else:
            try:
                files = check_manifest(job.get("manifest"), lim.max_bundle_gb << 30)
                # Fetch only what this machine hasn't cached. A network failure
                # raises HubError before anything is recorded, so the hub
                # re-queues the job and it is retried later.
                fetched = 0
                for f in {f["sha256"]: f for f in files}.values():
                    if not self.cache.has(f["sha256"], f["size"]):
                        self.api.download(job["downloads"][f["sha256"]], self.cache,
                                          f["sha256"], f["size"])
                        fetched += f["size"]
            except EvalError as e:   # bad manifest, or a file that fails its hash
                refusal = e
        if refusal:
            # Data-independent refusal: answer immediately, never run anything.
            # A refused test job isn't counted as a final evaluation.
            self.store.add(sid, task, submitter, 0, "test_refused" if split == "test" else split)
            self.store.finish(sid, {"run_id": sid, "task": task, "status": "error",
                                    "error": refusal.public()})
            return
        (self.queue_dir / f"{sid}.json").write_text(json.dumps(
            {"files": files, "bundle_id": job.get("bundle_id"), "split": split}))
        self.store.add(sid, task, submitter, lim.release_after_s, split)
        audit.record(self.cfg.state_dir, "job_received", run_id=sid, task=task, split=split,
                     submitter=submitter, bundle_id=job.get("bundle_id"),
                     bytes=sum(f["size"] for f in files), downloaded=fetched)
        self.store.start(sid)
        self._dispatch(sid, task, submitter, (float(lim.cpus), mem_bytes(lim.memory), int(lim.gpus)))

    def _run(self, sid: str, task: str, submitter: str) -> None:
        path = self.queue_dir / f"{sid}.json"
        try:
            job = json.loads(path.read_text())
            if not all(self.cache.has(f["sha256"], f["size"]) for f in job["files"]):
                raise EvalError("infra_error", internal="cached files missing")
            # Assembled next to the cache (hard links, same filesystem): the
            # confidential node keeps its state in memory but models on disk.
            bundle = self.cache.assemble(job["files"], self.cfg.blob_dir.parent / "bundles" / sid)
            try:
                result = self.runner.evaluate(task, bundle, run_id=sid, submitter=submitter,
                                              bundle_id=job.get("bundle_id"),
                                              split=job.get("split", "validation"))
            finally:
                if not self.cfg.retain_bundles:
                    shutil.rmtree(bundle, ignore_errors=True)
        except Exception as e:
            audit.record(self.cfg.state_dir, "worker_error", run_id=sid, error=type(e).__name__)
            result = {"run_id": sid, "status": "error", "error": EvalError("infra_error").public()}
        self.store.finish(sid, result)
        path.unlink(missing_ok=True)
