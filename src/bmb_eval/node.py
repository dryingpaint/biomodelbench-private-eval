"""Provider node: polls the BioModelBench hub for work over outbound HTTPS.

No listening port, no TLS certificate, no firewall rule. The node decides
everything that protects the data locally: quotas, approval lock, pause
switch, and when a result may be released. The hub only ever receives the
gated payload, and only once its release time has passed.

With batch_families, a job the hub says has queued siblings (the same bundle,
split and submitter on other tasks of this node: a family submission or its
finals) is taken together with them, and they share one predict sandbox
(Runner.predict_batch). Each job keeps its own quota, result, release time and
audit record; a job the batch didn't predict runs alone afterwards.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import alerts, approval, audit
from .blobs import BlobCache, check_manifest
from .config import Config
from .errors import EvalError
from .runner import Runner
from .sandbox import SandboxLost, SandboxRefused, stale_image_message
from .store import Store

POLL_S = 5
HELLO_S = 60
HUB_ALERT_AFTER = 60   # consecutive failed hub calls (~5 minutes) before alerting
# A job whose sandbox infrastructure failed goes back to the hub uncounted at most this
# many times; after that it is finished as a counted infra_error. Requeues are not
# reported to the agent, but bounding them bounds what a model that could make its
# own sandbox look lost (rather than fail) could signal through them.
MAX_INFRA_REQUEUES = 2
INFRA_BACKOFF_S = (60, 300, 900, 1800)   # no new work for this long after each lost sandbox in a row
MODAL_VERIFY_S = 3600  # full check of a Modal node's volumes against its manifest
MODAL_DRIFT_S = 300    # the manifest on Modal against the approved one (one API call)
MODAL_DRIFT_RETRY_S = 60   # sooner while drifted, to notice the fix
MODAL_CHECK_BACKOFF_S = (60, 120, 300, 600, 1800)   # after checks Modal couldn't answer
MODAL_CHECK_ALERT_AFTER = 3   # failed checks in a row before alerting
# Model files fetched at once across all jobs. Each job downloads in its own thread, so a
# node taking hundreds of jobs would otherwise open hundreds of multi-GB streams.
MAX_DOWNLOADS = 32


BATCH_MARKER = "BMB_BATCHABLE"   # a bundle file: the model may predict several tasks in one process


def _batchable(job: dict) -> bool:
    """Whether a job's bundle opted in to family batching. A batched model is imported and
    loaded once for several tasks; one that reads its contract or inputs only at import would
    predict every task as the first, so batching is the model's choice, not ours."""
    files = (job.get("manifest") or {}).get("files") or []
    return any(f.get("path") == BATCH_MARKER for f in files if isinstance(f, dict))


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
        # format=2: this client writes subfolders and binary files (see write_task_files).
        return self._req("GET", f"/v1/enroll/{code}?format=2")

    def activate(self, code: str, name: str) -> dict:
        return self._req("POST", f"/v1/enroll/{code}", {"name": name})

    def report_setup(self, code: str, step: str, reason: str, detail: str = "") -> None:
        """Tell BioModelBench a connect failed: its step and a fixed reason only.
        Best effort: a report that can't be sent never hides the real error."""
        try:
            HubAPI(self.url, timeout=10)._req("POST", f"/v1/enroll/{code}/report",
                                              {"step": step, "reason": reason, "detail": detail})
        except Exception:
            pass

    # work
    def hello(self, tasks: dict, accepting: bool, held: list[str], status: dict | None = None) -> dict:
        body = {"tasks": tasks, "accepting": accepting, "held": held}
        if status is not None:
            body["status"] = status      # fixed fields only (sealed node; it has no logs)
        return self._req("POST", "/v1/node/hello", body) or {}

    def next(self, **only) -> dict | None:
        """Claim a job. `only` (bundle_id, split, submitter, tasks) claims a sibling of a job
        already taken: a hub before family batching ignores it, so ask only a hub that
        reported siblings."""
        if only.get("tasks") is not None:
            only = {**only, "tasks": ",".join(only["tasks"])}
        q = urllib.parse.urlencode({k: v for k, v in only.items() if v is not None})
        return self._req("GET", "/v1/node/next" + (f"?{q}" if q else ""))

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


def machine_capacity(cfg: Config | None = None, sandbox=None) -> tuple[float, int, int]:
    """CPUs, memory (bytes) and GPUs this node's jobs share: this machine's, or for a
    Modal node, the GPU budget in its config (CPUs and memory come with each container)."""
    if cfg is not None and cfg.backend == "modal":
        return float("inf"), 1 << 62, int(cfg.modal.max_gpus)
    try:
        mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        mem = 0
    if sandbox is None:
        from .sandbox import gpus_here
        gpus = gpus_here()
    else:
        gpus = sandbox.gpus_available()
    return float(os.cpu_count() or 1), int(mem), len(gpus)


def paused_path(cfg: Config):
    return cfg.state_dir / "PAUSED"


def updates_path(cfg: Config):
    return cfg.state_dir / "updates.json"


def checkin_path(cfg: Config):
    """Touched after every successful check-in with the hub; `connect` waits for the first."""
    return cfg.state_dir / "last_checkin"


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
        self.capacity = machine_capacity(cfg, self.runner.sandbox)
        self._modal_verified = 0.0
        # The hourly volume-by-volume check runs in its own thread (_start_verify): it lists
        # every volume (~360 on the public Modal node), and in the loop it held up the node
        # for minutes and, rate-limited, kept it from taking work for hours (2026-11-02).
        self._verify_thread: threading.Thread | None = None
        self._verify_drift: list[str] = []      # what the last finished volume check found
        # Last answer of the Modal data check: None = not known yet, [] = matches, else the
        # problems. A check Modal can't answer (rate limit, API error) leaves it as it was.
        self._modal_state: list[str] | None = None
        self._modal_next_check = 0.0
        self._modal_check_failures = 0
        self._modal_lock_mtime: float | None = None
        self._infra_pause_until = 0.0     # after a lost sandbox: don't take new work until then
        # Model files jobs this node took still have to download (sha -> size): counted as used
        # disk when deciding whether another job fits (_room_on_disk).
        self._pending_files: dict[str, int] = {}
        self._disk_low = False
        # Size of the last job turned away for lack of disk: no work is asked for until it
        # fits, or the hub would hand the same (oldest) job back at every poll.
        self._deferred_need = 0
        self._modal_alerted: str | None = None
        # Model files being downloaded: sha256 -> set once that download ends, so a second
        # job needing the same file waits for it rather than fetching it again.
        self._fetching: dict[str, threading.Event] = {}
        self._fetch_lock = threading.Lock()
        self._download_slots = threading.Semaphore(MAX_DOWNLOADS)
        self._download_failures = 0       # in a row, across jobs
        # Set by SIGTERM/SIGINT (run_forever): when draining began. A second signal forces exit.
        self._draining: float | None = None
        self._force_exit = False

    # ------------------------------------------------------------- gates

    def accepting(self) -> bool:
        if self._draining is not None or paused_path(self.cfg).exists():
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
        if drift and self._reload_approved_tasks():
            drift = approval.diff(self.cfg)
        why = self._modal_drift() if not drift and self.cfg.backend == "modal" else []
        if why is None:
            # Not checked successfully yet (Modal hasn't answered): no work, but no drift either.
            self._accepting = (time.time(), False)
            return False
        if why:
            # The data on Modal must be what was approved. A local config that drifts gets
            # the usual alert below; data drift on Modal gets its own.
            if self._modal_alerted != why[0]:
                audit.record(self.cfg.state_dir, "modal_drift", problems=why[:5])
                alerts.send(self.cfg, "config", "the task data on Modal differs from what you approved; "
                            "no new work until it matches", first=why[0][:200])
                self._modal_alerted = why[0]
            self._accepting = (time.time(), False)
            return False
        self._modal_alerted = None
        self._accepting = (time.time(), not drift)
        if drift and not self._unapproved_alerted:
            alerts.send(self.cfg, "config", "configuration differs from what you approved; "
                        "no new work until you run `bmb-eval approve`", changes=len(drift))
        self._unapproved_alerted = bool(drift)
        return not drift

    def _reload_approved_tasks(self) -> bool:
        """Take up tasks added (or removed) on disk and approved, e.g. by `bmb-eval connect`
        while the service runs. The service compared the approval against the config it
        loaded at start, so every connect stopped it taking work until a restart: 7 hours on
        2026-11-02, while 63 tasks were added. Only the task list may differ; any other
        change still waits for a restart."""
        try:
            fresh = Config.load(self.cfg.path)
        except Exception:
            return False
        others = lambda c: {k: v for k, v in c.raw.items() if k != "tasks"}
        if approval.diff(fresh) or others(fresh) != others(self.cfg):
            return False
        self.cfg = self.runner.cfg = fresh
        if getattr(self.runner.sandbox, "cfg", None) is not None:
            self.runner.sandbox.cfg = fresh
        audit.record(self.cfg.state_dir, "config_reloaded", tasks=sorted(fresh.tasks))
        self._last_hello = 0.0        # publish the new tasks' contracts on the next step
        return True

    def _modal_drift(self) -> list[str] | None:
        """The data on Modal against what was approved: [] when it matches, the problems
        when it doesn't, None while it has never been checked. The manifest is checked every
        MODAL_DRIFT_S (at once after an approval), every volume hourly. When Modal can't
        answer, the last answer stands and the next try backs off."""
        import random
        from . import modal_sync
        now = time.time()
        try:
            lock = approval.lock_path(self.cfg).stat().st_mtime
        except OSError:
            lock = None
        if lock != self._modal_lock_mtime:
            # A new approval: the volume check's last answer was for other data. Check again.
            self._verify_drift, self._modal_verified = [], 0.0
        if now < self._modal_next_check and lock == self._modal_lock_mtime:
            return None if self._modal_state is None else self._modal_state + self._verify_drift
        try:
            out = modal_sync.drift(self.cfg)          # the manifest: one call, in the loop
        except modal_sync.ModalCheckError as e:
            self._modal_check_failures += 1
            n = self._modal_check_failures
            wait = MODAL_CHECK_BACKOFF_S[min(n, len(MODAL_CHECK_BACKOFF_S)) - 1]
            self._modal_next_check = now + wait * (0.75 + random.random() / 2)
            self._modal_lock_mtime = lock
            if n == 1:
                audit.record(self.cfg.state_dir, "modal_check_failed", error=str(e)[:300])
            if n == MODAL_CHECK_ALERT_AFTER:
                alerts.send(self.cfg, "infra", f"can't check the task data on Modal ({n} tries in a row); "
                            + ("still accepting on the last good check" if self._modal_state == []
                               else "no new work until a check succeeds"), error=str(e)[:200])
            return self._modal_state
        if self._modal_check_failures:
            audit.record(self.cfg.state_dir, "modal_check_recovered", failures=self._modal_check_failures)
        self._modal_check_failures = 0
        self._modal_state = out
        self._modal_lock_mtime = lock
        self._modal_next_check = now + (MODAL_DRIFT_RETRY_S if out else MODAL_DRIFT_S)
        if not out and now - self._modal_verified > MODAL_VERIFY_S:
            self._start_verify()
        return out + self._verify_drift

    def _start_verify(self) -> None:
        """Every volume against the manifest, in the background. Drift it finds stops new work
        (through _modal_drift) until a later check or approval clears it; a check Modal can't
        answer stops nothing and is tried again in ten minutes."""
        from . import modal_sync
        if self._verify_thread is not None and self._verify_thread.is_alive():
            return
        cfg, started = self.cfg, time.time()
        self._modal_verified = started             # due again only after this one
        def run() -> None:
            try:
                # Volumes listed in the last half hour (a check cut short) aren't listed again.
                self._verify_drift = modal_sync.verify(cfg, max_age_s=MODAL_VERIFY_S / 2)
                if self._verify_drift:
                    self._accepting = (0.0, False)   # take no more work from the next step on
            except modal_sync.ModalCheckError as e:
                self._modal_verified = started - MODAL_VERIFY_S + 600
                audit.record(cfg.state_dir, "modal_verify_failed", error=str(e)[:300])
            except Exception as e:
                self._modal_verified = started - MODAL_VERIFY_S + 600
                audit.record(cfg.state_dir, "modal_verify_failed", error=type(e).__name__)
        self._verify_thread = threading.Thread(target=run, daemon=True, name="modal-verify")
        self._verify_thread.start()

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
                checkin_path(self.cfg).write_text(str(time.time()))
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
        # Reaped before results are posted, so a job seen finished here has its result
        # delivered in this same step (a draining node exits once nothing is left running).
        did = self._reap()
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
        if self._draining is not None or not self._room_for_any():
            return did
        # Finish anything interrupted by a restart before taking new work.
        local = self.store.next_queued()
        if local:
            # No download links: a file evicted since is fetched by handing the job back.
            self._dispatch(*local, self._need(local[0], local[1]), {})
            return True
        if (not accepting or time.time() < self._infra_pause_until
                or not self._room_on_disk(self._deferred_need)):
            return did
        job = self.api.next()
        if job:
            jobs = [job, *self._claim_siblings(job)]
            self._start([(j, need) for j in jobs if (need := self._take(j)) is not None])
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
        if self.cfg.backend == "modal":
            # Each job gets its own containers with exactly its GPUs: they share the budget.
            return need[2] <= g
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

    def _dispatch(self, sid: str, task: str, submitter: str, need: tuple[float, int, int],
                  downloads: dict) -> None:
        # Every job runs in its own thread, a serial node's too (it just runs one at a
        # time): the loop keeps checking in and posting results while a job downloads its
        # model or runs, and can stop taking work at any moment (SIGTERM drains).
        th = threading.Thread(target=self._run, args=(sid, task, submitter, downloads), daemon=True,
                              name=f"eval-{sid}")
        self._jobs[sid] = (th, *need)
        th.start()

    def _claim_siblings(self, job: dict) -> list[dict]:
        """With batch_families: the queued siblings the hub reported for a job just taken (its
        bundle, split and submitter, on other tasks of this node), claimed at once so they
        share its predict. Only tasks whose predict needs GPUs exactly when the job's does, and
        at most batch_max_tasks jobs in all. The hub counts the siblings as it hands out the
        job, so nothing waits for more: a family's jobs are queued together when it is
        committed, and siblings claimed meanwhile by another node or loop are simply not here."""
        t = self.cfg.tasks.get(job.get("task"))
        want = min(int(job.get("siblings") or 0), self.cfg.batch_max_tasks - 1)
        if not (self.cfg.batch_families and t and want > 0 and job.get("bundle_id") and _batchable(job)):
            return []
        split = job.get("split", "validation")

        def gpu(b) -> bool:
            return bool(b.contract.limits_for(split, sealed=self.cfg.sealed).gpus)
        tasks = [tid for tid, b in self.cfg.tasks.items() if tid != t.id and gpu(b) == gpu(t)]
        out: list[dict] = []
        while tasks and len(out) < want:
            try:
                j = self.api.next(bundle_id=job["bundle_id"], split=split, submitter=job["submitter"],
                                  tasks=tasks)
            except HubError:
                break                    # run what we hold; the hub hands the rest out again
            if not j:
                break
            out.append(j)
            if j.get("task") in tasks:
                tasks.remove(j["task"])
        return out

    def _start(self, taken: list[tuple[dict, tuple[float, int, int]]]) -> None:
        """Run jobs just taken (with what each needs): those that can share a predict
        (one bundle, split and submitter, different tasks, all on GPUs or none) as one
        batch, any other alone."""
        def key(j: dict, need) -> tuple:
            return j.get("bundle_id"), j.get("split", "validation"), j["submitter"], bool(need[2])
        batch, alone, seen = [], [], set()
        for j, need in taken:
            if self.cfg.batch_families and key(j, need) == key(*taken[0]) and j["task"] not in seen:
                batch.append((j, need))
                seen.add(j["task"])
            else:
                alone.append((j, need))
        if len(batch) < 2:
            alone, batch = batch + alone, []
        if batch:
            members = [(j["submission_id"], j["task"], j["submitter"]) for j, _ in batch]
            need = tuple(max(n[i] for _, n in batch) for i in range(3))
            th = threading.Thread(target=self._run_batch, args=(members, batch[0][0].get("downloads") or {}),
                                  daemon=True, name=f"batch-{members[0][0]}")
            # One slot of max_parallel and the most any member needs: they predict one after another.
            self._jobs[f"batch-{members[0][0]}"] = (th, *need)
            audit.record(self.cfg.state_dir, "jobs_batched", runs=[m[0] for m in members],
                         bundle_id=batch[0][0].get("bundle_id"))
            th.start()
        for j, need in alone:
            self._dispatch(j["submission_id"], j["task"], j["submitter"], need, j.get("downloads") or {})

    def _reap(self) -> bool:
        done = [sid for sid, j in self._jobs.items() if not j[0].is_alive()]
        for sid in done:
            self._jobs.pop(sid)
        if done:
            self._evict()
        return bool(done)

    def _evict(self) -> None:
        """Keep the cache within cache_gb, sparing every file of a job this node still
        holds (running or waiting). Evicting only when no job ran never happened on a
        busy node: on 2026-10-30 its 100 GB cache had grown to 176 GB and the disk to 99%."""
        keep: set[str] = set()
        for p in self.queue_dir.glob("*.json"):
            try:
                keep |= {f["sha256"] for f in json.loads(p.read_text())["files"]}
            except (OSError, ValueError, KeyError, TypeError):
                continue
        self.cache.evict(self.cfg.cache_gb << 30, keep=keep)

    def wait_jobs(self, timeout: float | None = None) -> None:
        for th, *_ in list(self._jobs.values()):
            th.join(timeout)
        self._reap()

    def _note_updates(self, versions: dict) -> None:
        """BioModelBench changed a task: tell the provider (once per version).
        Nothing changes here until they review and approve it with `bmb-eval update`."""
        def have(task: str) -> int:
            return int((self.cfg.raw.get("tasks", {}).get(task) or {}).get("package_version", 1))
        # Applied since (`bmb-eval update`): no longer pending. A notice added back by a node
        # that hadn't reloaded the updated config stayed in `status` for good (2026-11-03).
        pending = {t: v for t, v in pending_updates(self.cfg).items() if v > have(t)}
        for task, version in versions.items():
            if version > have(task) and pending.get(task) != version:
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

    def run_forever(self) -> bool:
        if self.cfg.backend == "modal":
            # Sandboxes and job volumes of a previous run of this node are orphans now.
            from .modal_backend import cleanup_jobs
            try:
                killed = self.runner.sandbox.kill_all()
                volumes = cleanup_jobs(self.cfg)
                audit.record(self.cfg.state_dir, "modal_cleanup", sandboxes=killed, volumes=volumes)
            except Exception as e:
                audit.record(self.cfg.state_dir, "modal_cleanup_failed", error=type(e).__name__)
        self.drop_lost_jobs()
        n = self.store.requeue_running()
        audit.record(self.cfg.state_dir, "node_start", requeued=n)
        print(f"bmb-eval node running; connected to {self.api.url}. Ctrl-C to stop "
              "(after running evaluations finish; twice to stop at once).", flush=True)
        prev = {s: signal.signal(s, self._on_signal) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            return self.serve()
        finally:
            for s, h in prev.items():
                signal.signal(s, h)

    def _on_signal(self, signum, frame) -> None:
        """SIGTERM/SIGINT: drain. Every deploy (systemctl restart) used to throw away the
        evaluations in flight, up to an hour of GPU time each; now the node finishes them
        first. A second signal stops it at once, as before."""
        if self._draining is None:
            self._draining = time.time()
            self._last_hello = 0.0        # tell the hub now that this node takes no more work
            print(f"stopping: no new work; waiting for {len(self._jobs)} running evaluation(s), "
                  f"at most {self.cfg.drain_timeout_s} s. Signal again to stop at once.", flush=True)
        else:
            self._force_exit = True

    def serve(self) -> bool:
        """The node's loop, until a drain started by _on_signal ends: True once every job
        it was running has finished and its result is stored (posted if released), False
        when a second signal or drain_timeout_s cut it short. Jobs cut short run again when
        the node next starts, as after any interruption."""
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
            if self._draining is not None:
                if not self._jobs:
                    audit.record(self.cfg.state_dir, "node_drained",
                                 seconds=round(time.time() - self._draining))
                    return True
                if self._force_exit or time.time() - self._draining > self.cfg.drain_timeout_s:
                    audit.record(self.cfg.state_dir, "node_stop_forced", running=sorted(self._jobs),
                                 reason="signal" if self._force_exit else "drain_timeout")
                    return False
                busy = False              # nothing to take: wait for the running jobs
            if not busy:
                time.sleep(POLL_S)

    def drain(self, max_s: float = 300) -> None:
        """Process until idle (tests and one-shot use)."""
        deadline = time.time() + max_s
        if not self._jobs:
            # Only jobs no thread is running: requeuing one in flight would run it twice.
            self.store.requeue_running()
        while time.time() < deadline:
            if not self.step() and not self.store.released_unposted() and not self._jobs:
                break
            if self._jobs:
                self.wait_jobs(timeout=0.05)

    # -------------------------------------------------------------- jobs

    def _take(self, job: dict) -> tuple[float, int, int] | None:
        """Record a job from the hub, or refuse it. Returns what it needs to run (CPUs,
        memory, GPUs), or None when there is nothing to run."""
        sid, task, submitter = job["submission_id"], job["task"], job["submitter"]
        if self.store.has(sid):
            return None
        t = self.cfg.tasks.get(task)
        split = job.get("split", "validation")
        # Only "baseline" (BioModelBench's reference models, a baseline token at the hub)
        # has its own final cap; anything else counts as an agent's, as before kinds existed.
        kind = "baseline" if job.get("kind") == "baseline" else "agent"
        lim = t.contract.limits_for(split, sealed=self.cfg.sealed) if t else None
        refusal, files = None, []
        if t is None or split not in ("validation", "test"):
            refusal = EvalError("task_unknown")
        elif split == "test" and not (t.contract.split and t.test_inputs_dir):
            refusal = EvalError("no_test_split")
        elif split == "test" and self.store.count_final(task, kind) >= (
                t.contract.split.baseline_final_evaluations if kind == "baseline"
                else t.contract.split.final_evaluations):
            # Enforced here, not at the hub: even a compromised hub can't run the test set again.
            # A hub that mislabels jobs gets at most both caps together, which is what the
            # provider approved (leakage counts both). A node on code from before kinds
            # ignores the field and counts every final against final_evaluations.
            refusal = EvalError("final_quota_exceeded")
        elif split == "test" and lim.gpus and len(self.runner.sandbox.gpus_available()) < lim.gpus:
            # Refused before it counts, so the final can run once GPUs are added.
            refusal = EvalError("final_hardware_missing")
        elif split == "validation" and (
                self.store.count_today(submitter, task) >= lim.submissions_per_day or (
                lim.submissions_total and self.store.count_total(submitter, task) >= lim.submissions_total)):
            refusal = EvalError("quota_exceeded")
        else:
            try:
                files = check_manifest(job.get("manifest"), lim.max_bundle_gb << 30)
            except EvalError as e:   # bad manifest
                refusal = e
        if refusal:
            self._refuse(sid, task, submitter, split, kind, refusal, new=True)
            return None
        # A job whose model files don't fit isn't recorded: the next hello doesn't report it
        # held, so the hub hands it out again (uncounted) once this node, or another, has room.
        with self._fetch_lock:
            missing = {f["sha256"]: f["size"] for f in files
                       if not self.cache.has(f["sha256"], f["size"]) and f["sha256"] not in self._pending_files}
        if not self._room_on_disk(sum(missing.values())):
            audit.record(self.cfg.state_dir, "job_deferred_disk", run_id=sid, task=task,
                         need_gb=round(sum(missing.values()) / 2**30, 1))
            self._deferred_need = sum(missing.values())
            self._last_hello = 0.0
            return None
        self._deferred_need = 0
        with self._fetch_lock:
            self._pending_files.update(missing)
        # Taking a job is cheap: record it (so the next hello reports it held); its thread
        # (_start) downloads what this machine hasn't cached and then runs it. A
        # multi-GB download here used to stall the whole node: no new jobs, no results
        # posted, no check-ins.
        (self.queue_dir / f"{sid}.json").write_text(json.dumps(
            {"files": files, "bundle_id": job.get("bundle_id"), "split": split, "kind": kind}))
        self.store.add(sid, task, submitter, lim.release_after_s, split, kind)
        audit.record(self.cfg.state_dir, "job_received", run_id=sid, task=task, split=split, kind=kind,
                     submitter=submitter, bundle_id=job.get("bundle_id"),
                     bytes=sum(f["size"] for f in files))
        self.store.start(sid)
        return float(lim.cpus), mem_bytes(lim.memory), int(lim.gpus)

    def _free_bytes(self) -> int:
        """Free disk at the cache, less what taken jobs still have to download."""
        with self._fetch_lock:
            self._pending_files = {s: n for s, n in self._pending_files.items() if not self.cache.has(s, n)}
            pending = sum(self._pending_files.values())
        return shutil.disk_usage(self.cfg.blob_dir).free - pending

    def _room_on_disk(self, need: int = 0) -> bool:
        """Whether `need` more bytes of model files leave min_free_gb free. The node used to
        take jobs whatever the disk held: on 2026-11-03 a Modal node holding 156 jobs (10
        bundles, waiting for GPUs) filled the shared 484 GB disk, a half-written audit log
        then crashed it at every start, and both nodes took no work for about 14 hours."""
        ok = self._free_bytes() - need >= self.cfg.min_free_gb << 30
        if not ok and not self._disk_low:
            audit.record(self.cfg.state_dir, "disk_low", free_gb=round(self._free_bytes() / 2**30, 1),
                         need_gb=round(need / 2**30, 1), min_free_gb=self.cfg.min_free_gb)
            alerts.send(self.cfg, "infra", "not taking work: the disk is too full for more model files "
                        f"(keeping {self.cfg.min_free_gb} GB free)")
        self._disk_low = not ok
        return ok

    def _refuse(self, sid: str, task: str, submitter: str, split: str, kind: str,
                refusal: EvalError, new: bool) -> None:
        """Data-independent refusal: answer immediately, never run anything. A refused
        test job isn't counted as a final evaluation. `new`: not recorded yet; otherwise
        the recorded job (taken, then a file failed its hash) becomes the refusal."""
        refused = "test_refused" if split == "test" else split
        result = {"run_id": sid, "task": task, "status": "error", "error": refusal.public()}
        if new:
            self.store.add(sid, task, submitter, 0, refused, kind)
            self.store.finish(sid, result)
        else:
            self.store.refuse(sid, refused, result)
            (self.queue_dir / f"{sid}.json").unlink(missing_ok=True)

    def _fetch(self, files: list[dict], downloads: dict) -> int:
        """Download every file of a job that isn't cached; returns the bytes fetched.
        HubError when one can't be fetched; EvalError when one fails its hash.

        A job's files download concurrently (up to MAX_DOWNLOADS across the node). One at a
        time, a genomic bundle of thousands of small files took over 20 minutes on
        2026-11-02, a new HTTPS connection per file, while its family's other 25 jobs waited
        on the same files."""
        uniq = list({f["sha256"]: f for f in files}.values())
        if len(uniq) <= 1:
            return sum(self._fetch_one(f, downloads) for f in uniq)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(min(MAX_DOWNLOADS, len(uniq)), thread_name_prefix="fetch") as ex:
            # Leaving the pool waits for every download, so a failure is raised only once
            # nothing is still writing into the cache for this job.
            futures = [ex.submit(self._fetch_one, f, downloads) for f in uniq]
            return sum(f.result() for f in futures)

    def _fetch_one(self, f: dict, downloads: dict) -> int:
        fetched = 0
        sha, size = f["sha256"], f["size"]
        while not self.cache.has(sha, size):
            with self._fetch_lock:
                pending = self._fetching.get(sha)
                if pending is None:
                    mine = self._fetching[sha] = threading.Event()
            if pending is not None:
                # Another job is fetching this file: wait, then look again (and fetch it
                # ourselves if that download failed).
                pending.wait()
                continue
            try:
                if sha not in downloads:
                    # A job resumed after a restart has no links (signed, short-lived).
                    raise HubError(f"no download link for {sha[:12]}")
                with self._download_slots:
                    self.api.download(downloads[sha], self.cache, sha, size)
                fetched += size
            finally:
                with self._fetch_lock:
                    del self._fetching[sha]
                mine.set()
        return fetched

    def _hand_back(self, sid: str, task: str, e: HubError, count: bool = True) -> None:
        """A model file couldn't be downloaded: give the job back to the hub uncounted, as
        when the download ran before the job was recorded and the hub re-queued it. `count`:
        a failed download (once for all the jobs of a batch that shared it)."""
        self.store.forget(sid)
        (self.queue_dir / f"{sid}.json").unlink(missing_ok=True)
        with self._fetch_lock:
            self._download_failures += count
            n = self._download_failures
        audit.record(self.cfg.state_dir, "job_download_failed", run_id=sid, task=task,
                     error=str(e)[:300])
        if count and n == HUB_ALERT_AFTER:
            alerts.send(self.cfg, "infra", f"{n} model downloads in a row failed; jobs go back to "
                        "the hub uncounted", error=str(e)[:200])
        # Don't take the job straight back: the hub hands it out again at once.
        self._infra_pause_until = max(self._infra_pause_until, time.time() + POLL_S)
        self._last_hello = 0.0        # tell the hub on the next step, so it hands the job out again

    def _bundle(self, members: list[tuple[str, str, str]], job: dict, downloads: dict) -> Path | None:
        """The model's files for held jobs (sid, task, submitter) of one bundle, downloaded
        if need be and assembled; None when the jobs were handed back (a download failed)
        or refused (a file failed its hash) instead."""
        t0 = time.time()
        try:
            if fetched := self._fetch(job["files"], downloads):
                audit.record(self.cfg.state_dir, "job_downloaded", run_id=members[0][0], bytes=fetched,
                             seconds=round(time.time() - t0, 1))
        except HubError as e:
            for i, (sid, task, _) in enumerate(members):
                self._hand_back(sid, task, e, count=i == 0)
            return None
        except EvalError as e:   # a file that fails its hash
            for sid, task, submitter in members:
                self._refuse(sid, task, submitter, job.get("split", "validation"), job.get("kind", "agent"),
                             e, new=False)
            return None
        with self._fetch_lock:
            self._download_failures = 0
        # Assembled next to the cache (hard links, same filesystem): the
        # confidential node keeps its state in memory but models on disk.
        return self.cache.assemble(job["files"], self.cfg.blob_dir.parent / "bundles" / members[0][0])

    def _run(self, sid: str, task: str, submitter: str, downloads: dict) -> None:
        def evaluate() -> dict | None:
            job = json.loads((self.queue_dir / f"{sid}.json").read_text())
            bundle = self._bundle([(sid, task, submitter)], job, downloads)
            if bundle is None:
                return None
            try:
                return self.runner.evaluate(task, bundle, run_id=sid, submitter=submitter,
                                            bundle_id=job.get("bundle_id"),
                                            split=job.get("split", "validation"))
            finally:
                if not self.cfg.retain_bundles:
                    shutil.rmtree(bundle, ignore_errors=True)
        self._finish(sid, task, evaluate)

    def _run_batch(self, members: list[tuple[str, str, str]], downloads: dict) -> None:
        """Held jobs (sid, task, submitter) of one bundle and split: one predict for all of
        them, then each job's validate and score from its own predictions. A job the batch
        didn't predict goes back into this node's queue, and the loop runs it alone."""
        predicted: dict = {}
        bundle = None
        try:
            job = json.loads((self.queue_dir / f"{members[0][0]}.json").read_text())
            if (bundle := self._bundle(members, job, downloads)) is None:
                return
            split = job.get("split", "validation")
            predicted = self.runner.predict_batch([(sid, task) for sid, task, _ in members], bundle, split)

            def one(m: tuple[str, str, str]) -> None:
                sid, task, submitter = m
                res, info = predicted[sid]
                self._finish(sid, task, lambda: self.runner.evaluate(
                    task, bundle, run_id=sid, submitter=submitter, bundle_id=job.get("bundle_id"),
                    split=split, predicted=res, batch=info))
            done = [m for m in members if m[0] in predicted]
            # On Modal every stage has its own container, so the jobs validate and score side
            # by side; a container node runs them one at a time, in what the batch reserved.
            with ThreadPoolExecutor(max(1, len(done) if self.cfg.backend == "modal" else 1)) as ex:
                list(ex.map(one, done))
        except Exception as e:
            audit.record(self.cfg.state_dir, "worker_error", run_id=members[0][0], error=type(e).__name__)
        finally:
            if bundle is not None and not self.cfg.retain_bundles:
                shutil.rmtree(bundle, ignore_errors=True)
            for sid, task, _ in members:
                if sid not in predicted and self.store.requeue(sid):
                    audit.record(self.cfg.state_dir, "job_unbatched", run_id=sid, task=task)

    def _finish(self, sid: str, task: str, evaluate) -> None:
        """Run a held job (evaluate() gives its result, or None when it was already handed
        back or refused) and record the result, or hand the job back."""
        path = self.queue_dir / f"{sid}.json"
        try:
            result = evaluate()
            if result is None:
                return
        except SandboxLost as e:
            if self._requeue_lost(sid, task, e):
                return
            result = {"run_id": sid, "status": "error", "error": EvalError("infra_error").public()}
        except SandboxRefused as e:
            # A configuration problem (e.g. a contract asking Modal for more than a container
            # can have): every retry would be refused the same way, so it isn't handed back.
            # Finished as an infra_error the agent isn't charged for; the operator fixes it.
            self.store.finish(sid, {"run_id": sid, "task": task, "status": "error",
                                    "error": EvalError("infra_error").public()})
            self.store.uncount(sid)
            path.unlink(missing_ok=True)
            audit.record(self.cfg.state_dir, "job_refused_by_sandbox", run_id=sid, task=task,
                         error=str(e)[:300])
            alerts.send(self.cfg, "infra", f"the sandbox backend refused run {sid} as configured "
                        "(not retried, not counted); fix the task or node config", run_id=sid, task=task,
                        error=str(e)[:200])
            return
        except Exception as e:
            audit.record(self.cfg.state_dir, "worker_error", run_id=sid, error=type(e).__name__)
            result = {"run_id": sid, "status": "error", "error": EvalError("infra_error").public()}
        self.store.finish(sid, result)
        path.unlink(missing_ok=True)
        if result.get("status") == "ok" or (result.get("error") or {}).get("code") != "infra_error":
            self.store.bump("consecutive_sandbox_lost", reset=True)

    def _requeue_lost(self, sid: str, task: str, e: SandboxLost) -> bool:
        """The sandbox infrastructure failed (after the backend's retries): hand the job back
        to the hub without counting it against the agent, like a job a restart lost. True if
        it was handed back; False once this job has used up its requeues."""
        n = self.store.bump(f"infra_requeues:{sid}")
        lost = self.store.bump("consecutive_sandbox_lost")
        # Modal is failing: stop taking work for a while rather than burn the job's requeues.
        self._infra_pause_until = time.time() + INFRA_BACKOFF_S[min(lost, len(INFRA_BACKOFF_S)) - 1]
        if lost == self.cfg.infra_failure_threshold:
            alerts.send(self.cfg, "infra", f"{lost} evaluations in a row lost to sandbox infrastructure "
                        "failures (Modal); jobs go back to the hub uncounted", last_run_id=sid, task=task)
        if n > MAX_INFRA_REQUEUES:
            audit.record(self.cfg.state_dir, "job_infra_failed", run_id=sid, task=task, requeues=n - 1)
            return False
        self.store.forget(sid)
        (self.queue_dir / f"{sid}.json").unlink(missing_ok=True)
        audit.record(self.cfg.state_dir, "job_requeued_infra", run_id=sid, task=task, attempt=n,
                     error=str(e)[:300])
        self._last_hello = 0.0        # tell the hub on the next step, so it hands the job out again
        return True
