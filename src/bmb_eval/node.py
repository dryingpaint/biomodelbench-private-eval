"""Provider node: polls the BioModelBench hub for work over outbound HTTPS.

No listening port, no TLS certificate, no firewall rule. The node decides
everything that protects the data locally: quotas, approval lock, pause
switch, and when a result may be released. The hub only ever receives the
gated payload, and only once its release time has passed.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from . import alerts, approval, audit
from .blobs import BlobCache, check_manifest
from .config import Config
from .errors import EvalError
from .runner import Runner
from .store import Store

POLL_S = 5
HELLO_S = 60
HUB_ALERT_AFTER = 60   # consecutive failed hub calls (~5 minutes) before alerting


class HubError(Exception):
    pass


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
            raise HubError(msg)
        except (urllib.error.URLError, OSError) as e:
            raise HubError(f"cannot reach {self.url}: {e}")

    # enrollment
    def enrollment(self, code: str) -> dict:
        return self._req("GET", f"/v1/enroll/{code}")

    def activate(self, code: str, name: str) -> dict:
        return self._req("POST", f"/v1/enroll/{code}", {"name": name})

    # work
    def hello(self, tasks: dict, accepting: bool, held: list[str]) -> None:
        self._req("POST", "/v1/node/hello", {"tasks": tasks, "accepting": accepting, "held": held})

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


def paused_path(cfg: Config):
    return cfg.state_dir / "PAUSED"


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
        self.cache = BlobCache(cfg.state_dir / "blobs")
        self._last_hello = 0.0
        self._hub_failures = 0
        self._unapproved_alerted = False
        self._accepting = (0.0, False)

    # ------------------------------------------------------------- gates

    def accepting(self) -> bool:
        if paused_path(self.cfg).exists():
            return False
        checked, ok = self._accepting
        if time.time() - checked < 30:
            return ok
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
            self.api.hello({t: b.contract.public() for t, b in self.cfg.tasks.items()},
                           accepting, held)
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
        # Finish anything interrupted by a restart before taking new work.
        local = self.store.next_queued()
        if local:
            self._run(*local)
            return True
        if not accepting:
            return did
        job = self.api.next()
        if job:
            self._take(job)
            return True
        return did

    def run_forever(self) -> None:
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
            if not self.step() and not self.store.released_unposted():
                break

    # -------------------------------------------------------------- jobs

    def _take(self, job: dict) -> None:
        sid, task, submitter = job["submission_id"], job["task"], job["submitter"]
        if self.store.has(sid):
            return
        t = self.cfg.tasks.get(task)
        lim = t.contract.limits if t else None
        split = job.get("split", "validation")
        refusal, files = None, []
        if t is None or split not in ("validation", "test"):
            refusal = EvalError("task_unknown")
        elif split == "test" and not (t.contract.split and t.test_inputs_dir):
            refusal = EvalError("no_test_split")
        elif split == "test" and self.store.count_final(task) >= t.contract.split.final_evaluations:
            # Enforced here, not at the hub: even a compromised hub can't run the test set again.
            refusal = EvalError("final_quota_exceeded")
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
        self._run(sid, task, submitter)

    def _run(self, sid: str, task: str, submitter: str) -> None:
        path = self.queue_dir / f"{sid}.json"
        try:
            job = json.loads(path.read_text())
            if not all(self.cache.has(f["sha256"], f["size"]) for f in job["files"]):
                raise EvalError("infra_error", internal="cached files missing")
            bundle = self.cache.assemble(job["files"], self.cfg.state_dir / "runs" / sid / "bundle")
            result = self.runner.evaluate(task, bundle, run_id=sid, submitter=submitter,
                                          bundle_id=job.get("bundle_id"),
                                          split=job.get("split", "validation"))
        except Exception as e:
            audit.record(self.cfg.state_dir, "worker_error", run_id=sid, error=type(e).__name__)
            result = {"run_id": sid, "status": "error", "error": EvalError("infra_error").public()}
        self.store.finish(sid, result)
        path.unlink(missing_ok=True)
        self.cache.evict(self.cfg.cache_gb << 30)
