"""BioModelBench hub: the one service we run. Providers' nodes connect *out*
to it, so providers open no ports and run no web server.

Holds agent submissions (model bundles, as content-addressed files) and the
gated results nodes send back. Never sees provider data.

    agents   POST /v1/tasks/<task|family>/submissions   {manifest} -> {submission_id, uploads}
             PUT  <upload url>                    each file the hub doesn't have yet
             POST /v1/submissions/<id>/commit     queue it once every file is present
             GET  /v1/submissions/<id>            {status: pending|done, result?} (validation runs only)
             GET  /v1/tasks                       public contracts + node status
    nodes    POST /v1/node/hello                  report tasks/contracts, recover jobs
             GET  /v1/node/next                   claim a job: manifest + download urls
             POST /v1/node/results/<id>           deliver the gated payload
    enroll   GET  /v1/enroll/<code>               task package (contract, scorer, ...)
             POST /v1/enroll/<code>               activate: bind task to a node
    public   GET  /install.sh, /bmb-eval.tar.gz, /sandbox-image.tar,
                  /release, /SHA256SUMS (the release tag, commit and sha256 of each download)
                  /guide (the provider guide as a web page)

Model files live in Google Cloud Storage (`--bucket`), moved with signed URLs
so they never pass through this server; without a bucket the hub stores
them on its own disk (development and tests). Run behind TLS: see
deploy/gcp/ and docs/OPERATOR.md. Admin commands: `bmb-hub --help`.
"""
from __future__ import annotations

import argparse
import base64
import datetime
import gzip
import hashlib
import io
import json
import re
import secrets
import sqlite3
import tarfile
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import release
from .blobs import BlobCache, check_manifest, manifest_id
from .contract import Contract
from .errors import ERRORS, EvalError
from .sandbox import sandbox_stamp

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS agents (name TEXT PRIMARY KEY, token_sha256 TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS families (name TEXT PRIMARY KEY, tasks TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY, secret_sha256 TEXT NOT NULL, name TEXT NOT NULL,
    created REAL NOT NULL, last_seen REAL, accepting INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS node_tasks (
    task TEXT PRIMARY KEY, node_id TEXT NOT NULL, contract TEXT);
CREATE TABLE IF NOT EXISTS enrollments (
    code TEXT PRIMARY KEY, task TEXT NOT NULL, files TEXT NOT NULL,
    created REAL NOT NULL, used_by TEXT);
CREATE TABLE IF NOT EXISTS submissions (
    id TEXT PRIMARY KEY, task TEXT NOT NULL, agent TEXT NOT NULL, created REAL NOT NULL,
    status TEXT NOT NULL,          -- uploading | queued | assigned | done
    node_id TEXT, result TEXT, manifest TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS setup_reports (   -- a provider's connect command that failed
    code TEXT NOT NULL, step TEXT NOT NULL, reason TEXT NOT NULL, detail TEXT NOT NULL,
    at REAL NOT NULL);
"""
ONLINE_S = 120          # a node that polled within this window is "online"
STALL_S = 3600          # a code used this long ago whose setup never finished is "stalled"
# What a failed connect may report: its step and a fixed reason, plus an exception's
# class name for unexpected errors. Never data, paths or error text.
SETUP_STEPS = {"start", "gcloud_check", "fetch_task", "sandbox_image", "data_check",
               "hardware_check", "selftest", "review", "activate", "cloud_setup", "upload",
               "notify", "start_service"}
SETUP_REASONS = {"data_problem", "declined", "machine_not_ready", "selftest_failed",
                 "gcloud_not_ready", "gcloud_refused", "region_blocked", "tls", "hub_error",
                 "interrupted", "stopped", "crashed"}
SETUP_DETAIL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,39}")
# Release files the provider install needs from GitHub (uv, and the Python uv installs),
# passed through and cached: providers' networks often allow this hub but not GitHub.
# Only these two repositories' release assets; uv checks Python downloads' hashes itself.
DOWNLOADS = {"uv": "https://github.com/astral-sh/uv/releases/download",
             "python": "https://github.com/astral-sh/python-build-standalone/releases/download"}
DOWNLOAD_RE = re.compile(r"/dl/(uv|python)/([A-Za-z0-9._+-]{1,40})/([A-Za-z0-9._+-]{1,160})")
MAX_DOWNLOAD = 512 << 20
MAX_JSON_BYTES = 16 << 20   # manifests of large bundles can be a few MB
MAX_HELLO_BYTES = 4 << 20   # a node's hello carries every task's public contract (~2-6 KB each)
MAX_RESULT_BYTES = 64 << 10
MAX_PACKAGE_BYTES = 16 << 20   # a task folder: contract, scorer, labels, small reference files
URL_TTL = datetime.timedelta(hours=24)
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O/1/I
AGENT_KINDS = ("agent", "baseline")


def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def new_code() -> str:
    return "-".join("".join(secrets.choice(CODE_ALPHABET) for _ in range(4)) for _ in range(3))


# ------------------------------------------------------------- file storage

class DiskBlobs:
    """Model files on the hub's own disk; the hub serves uploads and downloads
    itself (with the caller's bearer token). Verified by hash on upload."""

    def __init__(self, root: Path, public_url: str):
        self.cache = BlobCache(root)
        self.public_url = public_url

    def has(self, agent: str, sha: str, size: int) -> bool:
        return self.cache.has(sha, size)

    def upload(self, agent: str, sha: str, size: int) -> dict:
        return {"url": f"{self.public_url}/v1/blobs/{sha}?size={size}", "method": "PUT",
                "headers": {}, "auth": True}

    def download(self, agent: str, sha: str, size: int) -> dict:
        return {"url": f"{self.public_url}/v1/blobs/{sha}", "headers": {}, "auth": True}


class GCSBlobs:
    """Model files in a Cloud Storage bucket, moved with signed URLs, so
    multi-GB uploads and downloads never pass through the hub. Stored per
    agent (an agent can't plant a bad file under someone else's hash); nodes
    verify every hash anyway. Needs google-cloud-storage and a service
    account allowed to sign (deploy/gcp/hub-setup.sh grants it)."""

    def __init__(self, bucket: str, credentials=None, project: str | None = None):
        import google.auth
        from google.auth.transport import requests as gauth_requests
        from google.cloud import storage
        if credentials is None:
            credentials, project = google.auth.default()
        self._creds = credentials
        self._request = gauth_requests.Request()
        self._refresh_lock = threading.Lock()
        self.bucket = storage.Client(project=project, credentials=self._creds).bucket(bucket)

    def _name(self, agent: str, sha: str) -> str:
        return f"blobs/{_h(agent)[:16]}/{sha}"

    def _sign(self, agent: str, sha: str, method: str, **kw) -> str:
        signer = {}
        if hasattr(self._creds, "service_account_email") and not hasattr(self._creds, "signer"):
            # Compute Engine credentials sign through the IAM API. Signing runs in parallel
            # (sign_all): refresh the token under a lock, once.
            with self._refresh_lock:
                if not self._creds.valid:
                    self._creds.refresh(self._request)
            signer = {"service_account_email": self._creds.service_account_email,
                      "access_token": self._creds.token}
        return self.bucket.blob(self._name(agent, sha)).generate_signed_url(
            version="v4", expiration=URL_TTL, method=method, **signer, **kw)

    def has(self, agent: str, sha: str, size: int) -> bool:
        b = self.bucket.get_blob(self._name(agent, sha))
        return b is not None and b.size == size

    def upload(self, agent: str, sha: str, size: int) -> dict:
        ct = "application/octet-stream"
        return {"url": self._sign(agent, sha, "PUT", content_type=ct), "method": "PUT",
                "headers": {"Content-Type": ct}, "auth": False}

    def download(self, agent: str, sha: str, size: int) -> dict:
        return {"url": self._sign(agent, sha, "GET"), "headers": {}, "auth": False}


# -------------------------------------------------------------------- state

def _public_status(status: str) -> str:
    """done, registered (a final candidate: kept, not run on validation), or pending."""
    return status if status in ("done", "registered") else "pending"


def _summary(sid: str, rows) -> dict:
    """(task, status, result) rows of one submission -> its status and, once every
    dataset is done, an equal-weight combined headline over the datasets."""
    status = {_public_status(r[1]) for r in rows}
    out = {"submission_id": sid, "tasks": [r[0] for r in rows],
           "status": status.pop() if len(status) == 1 else "pending"}
    if out["status"] == "done":
        datasets = {t: json.loads(r) for t, _, r in rows}
        ok = [d for d in datasets.values() if d.get("status") == "ok"]
        # Each dataset counts with its own headline metric (a family can mix metrics).
        heads = {d["headline"] for d in ok}
        headline = heads.pop() if len(heads) == 1 else ("mean of headlines" if heads else None)
        vals = [d["metrics"][d["headline"]] for d in ok if d["metrics"].get(d["headline"]) is not None]
        out["result"] = {"status": "ok" if vals else "error", "headline": headline,
                         "combined": round(sum(vals) / len(vals), 4) if vals else None,
                         "datasets_scored": len(vals), "datasets": datasets}
    return out


class HubStore:
    def __init__(self, state: Path):
        self.state = state
        state.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(state / "hub.db", check_same_thread=False)
        self.db.executescript(SCHEMA)
        # Columns added after the first release; older hub.db files gain them here.
        for table, col, decl in [("enrollments", "version", "INTEGER NOT NULL DEFAULT 1"),
                                 ("enrollments", "activated", "REAL"),
                                 ("submissions", "split", "TEXT NOT NULL DEFAULT 'validation'"),
                                 ("submissions", "group_id", "TEXT"),
                                 ("agents", "eval_budget", "INTEGER"),
                                 ("nodes", "confidential", "TEXT"),
                                 ("nodes", "status", "TEXT"),
                                 # "baseline": BioModelBench's reference models, whose finals
                                 # count against the contract's baseline_final_evaluations.
                                 ("agents", "kind", "TEXT NOT NULL DEFAULT 'agent'"),
                                 ("submissions", "kind", "TEXT NOT NULL DEFAULT 'agent'")]:
            if col not in {r[1] for r in self.db.execute(f"PRAGMA table_info({table})")}:
                self.db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
        self.db.commit()
        self.lock = threading.Lock()

    def q(self, sql: str, args=(), one=False):
        with self.lock, self.db:
            cur = self.db.execute(sql, args)
            return cur.fetchone() if one else cur.fetchall()

    # ---------------------------------------------------------------- auth

    def setting(self, key: str) -> dict | None:
        row = self.q("SELECT value FROM settings WHERE key=?", (key,), one=True)
        return json.loads(row[0]) if row else None

    def set_setting(self, key: str, value: dict) -> None:
        self.q("INSERT OR REPLACE INTO settings VALUES (?,?)", (key, json.dumps(value)))

    def agent(self, token: str) -> str | None:
        row = self.q("SELECT name FROM agents WHERE token_sha256=?", (_h(token),), one=True)
        return row[0] if row else None

    def operator_status(self, recent: int = 50) -> dict:
        """Nodes, the submission queue by task/split/status, and the latest submissions."""
        nodes = []
        for nid, name, seen, acc in self.q("SELECT id, name, last_seen, accepting FROM nodes"):
            nodes.append({"id": nid, "name": name, "last_seen": seen, "accepting": bool(acc),
                          "tasks": [r[0] for r in self.q("SELECT task FROM node_tasks WHERE node_id=?", (nid,))]})
        subs = [{"task": t, "split": sp, "status": st, "n": n, "oldest_created": oc}
                for t, sp, st, n, oc in self.q("SELECT task, split, status, COUNT(*), MIN(created) "
                                                "FROM submissions GROUP BY 1, 2, 3")]
        rec = [{"id": i, "task": t, "split": sp, "status": st, "agent": a, "created": c}
               for i, t, sp, st, a, c in self.q("SELECT id, task, split, status, agent, created FROM submissions "
                                                "ORDER BY created DESC LIMIT ?", (recent,))]
        return {"now": time.time(), "nodes": nodes, "submissions": subs, "recent": rec,
                "setup_problems": self.setup_problems()}

    def is_operator(self, token: str) -> bool:
        op = self.setting("operator")
        return bool(token) and op is not None and op["token_sha256"] == _h(token)

    def set_operator(self) -> str:
        """Create or rotate the operator token (bmbo_…): issues agent tokens and requests
        and reads final evaluations over the API, for automated runs."""
        token = "bmbo_" + secrets.token_urlsafe(32)
        self.set_setting("operator", {"token_sha256": _h(token)})
        return token

    def node(self, secret: str) -> str | None:
        row = self.q("SELECT id FROM nodes WHERE secret_sha256=?", (_h(secret),), one=True)
        return row[0] if row else None

    def _finals_of(self, sid: str) -> list[str]:
        """Final evaluations of one submission row (ids "<sid>-final<n>"), oldest first.
        Matched on the exact prefix: LIKE would treat "_" in an id as a wildcard."""
        pre = sid + "-final"
        return [r[0] for r in self.q("SELECT id FROM submissions WHERE split='test' "
                                     "AND substr(id, 1, ?) = ? ORDER BY created", (len(pre), pre))]

    def request_final(self, sid: str, again: bool = True) -> str:
        """Queue a final (test set) evaluation of an existing submission's model.
        Admin only. The provider's node decides whether it may run (its cap). With
        again=False, an existing final is returned instead of spending another, unless
        the node refused it: a refused final never ran and wasn't counted."""
        row = self.q("SELECT task, agent, manifest, kind FROM submissions WHERE id=? AND split='validation' "
                     "AND status NOT IN ('uploading', 'uploading_final')", (sid,), one=True)
        if not row:
            raise KeyError(f"no committed submission {sid}")
        done = self._finals_of(sid)
        if done and not again and not self._refused(done[-1]):
            return done[-1]
        fid = f"{sid}-final{len(done) + 1}"
        # The final keeps its submission's kind: a baseline's counts against the baseline cap.
        self.q("INSERT INTO submissions (id, task, agent, created, status, node_id, result, manifest, split, "
               "kind) VALUES (?,?,?,?, 'queued', NULL, NULL, ?, 'test', ?)",
               (fid, row[0], row[1], time.time(), row[2], row[3]))
        return fid

    def _refused(self, fid: str) -> bool:
        """Refused by the node's policy (e.g. no test split yet, before the provider approved
        one): answered without running, and not counted against the task's finals."""
        row = self.q("SELECT result FROM submissions WHERE id=?", (fid,), one=True)
        code = ((json.loads(row[0]) if row and row[0] else {}).get("error") or {}).get("code")
        return code in ERRORS and ERRORS[code][1] == "policy"

    def request_finals(self, sid: str, again: bool = True) -> list[str]:
        """Final evaluations for a task submission, or for every dataset of a
        family submission."""
        children = [r[0] for r in self.q("SELECT id FROM submissions WHERE group_id=? "
                                         "AND split='validation' ORDER BY task", (sid,))]
        return [self.request_final(c, again) for c in (children or [sid])]

    def add_agent(self, name: str, evaluations: int | None = None, kind: str = "agent") -> str:
        """Token for one attempt. `evaluations` caps its submissions across all
        tasks and families (a family submission counts once); None = no cap.
        kind="baseline" (operator only) is for BioModelBench's own reference
        models: their submissions, and so their finals, are marked as baselines,
        which providers' nodes cap separately (split.baseline_final_evaluations)."""
        if kind not in AGENT_KINDS:
            raise ValueError(f"kind must be one of {AGENT_KINDS}")
        token = "bmba_" + secrets.token_urlsafe(32)
        self.q("INSERT OR REPLACE INTO agents (name, token_sha256, eval_budget, kind) VALUES (?,?,?,?)",
               (name, _h(token), evaluations, kind))
        return token

    def agent_kind(self, agent: str) -> str:
        row = self.q("SELECT kind FROM agents WHERE name=?", (agent,), one=True)
        return row[0] if row else "agent"

    def budget(self, agent: str) -> tuple[int | None, int]:
        """(evaluation budget, evaluations used) for an attempt."""
        cap = self.q("SELECT eval_budget FROM agents WHERE name=?", (agent,), one=True)[0]
        used = self.q("SELECT COUNT(DISTINCT COALESCE(group_id, id)) FROM submissions WHERE agent=? "
                      "AND split='validation' AND status IN ('queued', 'assigned', 'done')", (agent,), one=True)[0]
        return cap, used

    # ----------------------------------------------------------- families

    def set_family(self, name: str, tasks: list[str]) -> None:
        self.q("INSERT OR REPLACE INTO families VALUES (?,?)", (name, json.dumps(tasks)))

    def families(self) -> dict[str, list[str]]:
        return {n: json.loads(t) for n, t in self.q("SELECT name, tasks FROM families ORDER BY name")}

    def group_view(self, gid: str, agent: str | None = None) -> dict | None:
        """A family submission: per-dataset results plus an equal-weight combined
        headline. Pending until every dataset has reported."""
        q = ("SELECT task, status, result FROM submissions WHERE group_id=? AND split='validation'"
             + (" AND agent=?" if agent else "") + " ORDER BY task")
        rows = self.q(q, (gid, agent) if agent else (gid,))
        return _summary(gid, rows) if rows else None

    def final_view(self, sid: str) -> dict | None:
        """The latest final (test set) evaluation of each dataset of a submission, in
        group_view's shape. Operator only."""
        children = [r[0] for r in self.q("SELECT id FROM submissions WHERE group_id=? "
                                         "AND split='validation' ORDER BY task", (sid,))]
        latest = [f[-1] for f in map(self._finals_of, children or [sid]) if f]
        rows = [self.q("SELECT task, status, result FROM submissions WHERE id=?", (f,), one=True) for f in latest]
        return _summary(sid, sorted(rows)) if rows else None

    # ----------------------------------------------------------- enrollment

    def code_for(self, task: str) -> str | None:
        """The task's connect code (one per task, for good), unless revoked."""
        row = self.q("SELECT code FROM enrollments WHERE task=? AND (used_by IS NULL OR used_by <> 'revoked') "
                     "ORDER BY created DESC LIMIT 1", (task,), one=True)
        return row[0] if row else None

    def add_enrollment(self, task_dir: Path) -> tuple[str, Contract]:
        """Package a task folder: contract, scorer, and any other files it uses, in
        subfolders too (e.g. a gene-panel labels file, a reference .npy). Example
        bundles are not shipped.

        A task keeps one code: enrolling an existing task id again updates its
        package (version + 1) behind the same code. A connected provider's node
        then reports an update, which the provider reviews and approves with
        `bmb-eval update`; until then it keeps serving the approved version."""
        c = Contract.load(task_dir / "contract.yaml")
        files = self.package_files(task_dir)
        code = self.code_for(c.id)
        if code:
            self.q("UPDATE enrollments SET files=?, version=version+1 WHERE code=?",
                   (json.dumps(files), code))
        else:
            code = new_code()
            self.q("INSERT INTO enrollments (code, task, files, created, used_by, version) "
                   "VALUES (?,?,?,?,NULL,1)", (code, c.id, json.dumps(files), time.time()))
        return code, c

    @staticmethod
    def package_files(task_dir: Path) -> dict[str, str | dict]:
        """Every file in the task folder, subfolders included, keyed by its path
        relative to the folder. Text files as text; anything else (.npy, .parquet,
        ...) as {"base64": ...}. Hidden files, caches and the example bundle/ are left out."""
        files, total = {}, 0
        for p in sorted(task_dir.rglob("*")):
            rel = p.relative_to(task_dir)
            if (not p.is_file() or rel.parts[0] == "bundle" or "__pycache__" in rel.parts
                    or any(part.startswith(".") for part in rel.parts)):
                continue
            raw = p.read_bytes()
            total += len(raw)
            if total > MAX_PACKAGE_BYTES:
                raise ValueError(f"The task folder {task_dir} holds more than {MAX_PACKAGE_BYTES >> 20} MB "
                                 "of files; task packages are for contracts, scorers and small reference files.")
            try:
                files[rel.as_posix()] = raw.decode()
            except UnicodeDecodeError:
                files[rel.as_posix()] = {"base64": base64.b64encode(raw).decode()}
        return files

    def enrollment(self, code: str) -> tuple[str, dict, str | None, int] | None:
        row = self.q("SELECT task, files, used_by, version FROM enrollments WHERE code=?", (code,), one=True)
        return (row[0], json.loads(row[1]), row[2], row[3]) if row else None

    def task_versions(self, node_id: str) -> dict[str, int]:
        """Current package version of each task this node serves."""
        return {t: v for t, v in self.q(
            "SELECT e.task, MAX(e.version) FROM enrollments e JOIN node_tasks n ON n.task = e.task "
            "WHERE n.node_id=? AND e.used_by=? GROUP BY e.task", (node_id, node_id))}

    def activate(self, code: str, node_id: str | None, name: str) -> dict:
        """Bind the code's task to a node, creating the node if needed."""
        e = self.enrollment(code)
        # Reusable by the node it's bound to (re-runs, updates); any other
        # machine is refused once it's been used.
        if e is None or e[2] == "revoked" or (e[2] is not None and e[2] != node_id):
            raise KeyError("code unknown or already used by another machine")
        task = e[0]
        owner = self.q("SELECT node_id FROM node_tasks WHERE task=?", (task,), one=True)
        out = {"task": task}
        if node_id is None:
            node_id = "node_" + secrets.token_hex(6)
            secret = "bmbn_" + secrets.token_urlsafe(32)
            self.q("INSERT INTO nodes (id, secret_sha256, name, created, last_seen, accepting) "
                   "VALUES (?,?,?,?,NULL,0)", (node_id, _h(secret), name, time.time()))
            out["secret"] = secret
        if owner and owner[0] != node_id:
            raise KeyError("task is already served by another node")
        self.q("INSERT OR REPLACE INTO node_tasks VALUES (?,?,NULL)", (task, node_id))
        # Each activation (a provider re-running their command) restarts the stall clock
        # and re-arms the stall alert, so a second stalled attempt is reported too.
        self.q("UPDATE enrollments SET used_by=?, activated=? WHERE code=?", (node_id, time.time(), code))
        alerted = self.setting("stall_alerted") or {}
        if alerted.pop(code, None) is not None:
            self.set_setting("stall_alerted", alerted)
        out["node_id"] = node_id
        return out

    def retire(self, task: str, apply: bool = False) -> dict[str, int]:
        """Remove a task from the hub: its codes, its node binding, unfinished
        submissions, family membership, setup reports, and any node left serving
        nothing. Finished results stay as history. Dry run unless `apply`; with it,
        hub.db is backed up first. Returns how many of each would go (or went)."""
        codes = [r[0] for r in self.q("SELECT code FROM enrollments WHERE task=?", (task,))]
        nodes = [r[0] for r in self.q("SELECT node_id FROM node_tasks WHERE task=?", (task,))]
        orphans = [n for n in nodes if self.q("SELECT COUNT(*) FROM node_tasks WHERE node_id=? AND task<>?",
                                              (n, task), one=True)[0] == 0]
        marks = ",".join("?" * len(codes)) or "NULL"
        counts = {
            "codes": len(codes),
            "node bindings": len(nodes),
            "unfinished submissions": self.q("SELECT COUNT(*) FROM submissions WHERE task=? AND status<>'done'",
                                             (task,), one=True)[0],
            "families": sum(task in t for t in self.families().values()),
            "setup reports": self.q(f"SELECT COUNT(*) FROM setup_reports WHERE code IN ({marks})", codes,
                                    one=True)[0],
            "nodes left serving nothing": len(orphans),
        }
        if not apply:
            return counts
        backup = sqlite3.connect(self.state / f"hub.db.bak-{time.strftime('%Y%m%d-%H%M%S')}-retire-{task}")
        with self.lock:
            self.db.backup(backup)
        backup.close()
        with self.lock, self.db:
            self.db.execute(f"DELETE FROM setup_reports WHERE code IN ({marks})", codes)
            self.db.execute("DELETE FROM enrollments WHERE task=?", (task,))
            self.db.execute("DELETE FROM node_tasks WHERE task=?", (task,))
            self.db.execute("DELETE FROM submissions WHERE task=? AND status<>'done'", (task,))
        for name, tasks in self.families().items():
            if task in tasks:
                self.set_family(name, [t for t in tasks if t != task])
        alerted = self.setting("stall_alerted") or {}
        if set(alerted) & set(codes):
            self.set_setting("stall_alerted", {c: t for c, t in alerted.items() if c not in codes})
        for n in orphans:
            self.q("DELETE FROM nodes WHERE id=?", (n,))
            self.q("DELETE FROM settings WHERE key=?", (f"sealed_launch:{n}",))
        return counts

    def is_sealed(self, node_id: str) -> bool:
        """A sealed (confidential) node: it has bucket pointers, or was named by
        connect-confidential (which names it sealed-<project>) before uploading."""
        row = self.q("SELECT name, confidential FROM nodes WHERE id=?", (node_id,), one=True)
        return bool(row and (row[1] or row[0].startswith("sealed-")))

    # ------------------------------------------------------- setup problems

    def report_setup(self, code: str, step: str, reason: str, detail: str = "") -> str:
        """Record a provider's failed connect. Returns the task id. Raises KeyError
        for an unknown or revoked code, ValueError for anything off the fixed lists."""
        e = self.enrollment(code)
        if e is None or e[2] == "revoked":
            raise KeyError("code unknown")
        if step not in SETUP_STEPS or reason not in SETUP_REASONS or (
                detail and not SETUP_DETAIL_RE.fullmatch(detail)):
            raise ValueError("step, reason or detail not recognized")
        self.q("INSERT INTO setup_reports VALUES (?,?,?,?,?)", (code, step, reason, detail, time.time()))
        # A code's reports are capped, so a leaked code can't fill the disk.
        self.q("DELETE FROM setup_reports WHERE code=? AND rowid NOT IN (SELECT rowid FROM "
               "setup_reports WHERE code=? ORDER BY at DESC LIMIT 20)", (code, code))
        return e[0]

    def setup_problems(self, now: float | None = None) -> list[dict]:
        """Setups that need a look: each code's latest failure report (last 7 days),
        and codes used over STALL_S ago whose machine never finished connecting
        (no upload for a sealed one, never polled for a provider's own machine)."""
        now = now or time.time()
        out = []
        for code, task, step, reason, detail, at, used, seen, sealed in self.q(
                "SELECT r.code, e.task, r.step, r.reason, r.detail, MAX(r.at), e.used_by, n.last_seen, "
                "n.confidential FROM setup_reports r JOIN enrollments e ON e.code = r.code "
                "LEFT JOIN nodes n ON n.id = e.used_by WHERE r.at > ? GROUP BY r.code", (now - 7 * 86400,)):
            done = bool(seen or sealed)
            out.append({"kind": "failed", "code": code, "task": task, "step": step, "reason": reason,
                        "detail": detail, "at": at, "resolved": done})
        reported = {p["code"] for p in out}
        for code, task, at, name in self.q(
                "SELECT e.code, e.task, COALESCE(e.activated, n.created), n.name FROM enrollments e "
                "JOIN nodes n ON n.id = e.used_by WHERE n.last_seen IS NULL AND n.confidential IS NULL "
                "AND COALESCE(e.activated, n.created) < ?", (now - STALL_S,)):
            if code not in reported:
                out.append({"kind": "stalled", "code": code, "task": task, "at": at,
                            "sealed": (name or "").startswith("sealed-")})
        return out


def cached_download(cache: Path, kind: str, release: str, name: str) -> Path | None:
    """A GitHub release file (see DOWNLOADS), fetched once and kept. None if it doesn't exist."""
    import urllib.error
    import urllib.request
    if ".." in (release, name):
        return None
    path = cache / kind / release / name
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{name}.{secrets.token_hex(4)}")
    try:
        url = f"{DOWNLOADS[kind]}/{urllib.parse.quote(release)}/{urllib.parse.quote(name)}"
        with urllib.request.urlopen(url, timeout=60) as r, \
                tmp.open("wb") as f:
            n = 0
            while chunk := r.read(1 << 20):
                n += len(chunk)
                if n > MAX_DOWNLOAD:
                    raise ValueError("too large")
                f.write(chunk)
        tmp.replace(path)
        return path
    except (urllib.error.URLError, ValueError, OSError):
        tmp.unlink(missing_ok=True)
        return None


def code_tarball(root: Path, rel: dict | None = None) -> bytes | None:
    """The node code providers install, built from this checkout, with release.json
    saying which release it is. Byte-for-byte the same for the same files (no
    timestamps or owners), so its sha256 identifies the release (GET /release)."""
    if not (root / "setup.sh").is_file():
        return None
    include = ["pyproject.toml", "README.md", "setup.sh", "bmb-eval", "sandbox", "src", "docs"]

    def clean(ti: tarfile.TarInfo) -> tarfile.TarInfo | None:
        if "__pycache__" in ti.name or ti.name.endswith(".pyc"):
            return None
        ti.mtime = 0
        ti.uid = ti.gid = 0
        ti.uname = ti.gname = ""
        ti.mode = 0o755 if ti.isdir() or ti.mode & 0o100 else 0o644
        return ti

    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w") as tar:
        for name in include:            # tarfile walks directories in sorted order
            p = root / name
            if p.exists():
                tar.add(p, arcname=name, filter=clean)
        if rel:
            data = (json.dumps(rel) + "\n").encode()
            ti = tarfile.TarInfo("release.json")
            ti.size, ti.mode = len(data), 0o644
            tar.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def published_image(state: Path, expected_stamp: str | None) -> dict | None:
    """The sandbox image this hub publishes: its sha256 and size, the stamp it was
    built with (deploy/gcp/hub-deploy.sh), and whether that is this code's stamp.
    None until an image has been published."""
    tar = state / "sandbox-image.tar"
    sums = tar.with_suffix(".tar.sha256")
    if not tar.is_file() or not sums.is_file():
        return None
    stamp_file = state / "sandbox-image.stamp"
    stamp = stamp_file.read_text().strip() if stamp_file.is_file() else None
    return {"sha256": sums.read_text().split()[0], "bytes": tar.stat().st_size,
            "stamp": stamp, "current": stamp is not None and stamp == expected_stamp}


def _digest(body: bytes) -> dict:
    return {"sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}


# --------------------------------------------------------------------- HTTP

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ --fg:#1d2330; --muted:#5b6475; --bg:#ffffff; --code:#f4f6f9; --line:#e3e7ee; --link:#1f5fbf; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --fg:#e6e9ef; --muted:#a3abb9; --bg:#14171c; --code:#1e232b; --line:#2c323c; --link:#7fb0ff; }}
  }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
         font:16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
  main {{ max-width:760px; margin:0 auto; padding:40px 20px 80px; }}
  h1 {{ font-size:1.9em; line-height:1.25; margin:0 0 .4em; }}
  h2 {{ font-size:1.3em; margin:2em 0 .5em; padding-top:.4em; border-top:1px solid var(--line); }}
  a {{ color:var(--link); }}
  code {{ background:var(--code); padding:.1em .35em; border-radius:4px; font-size:.92em; }}
  pre {{ background:var(--code); padding:14px 16px; border-radius:8px; overflow-x:auto; line-height:1.45; }}
  pre code {{ background:none; padding:0; }}
  blockquote {{ margin:1em 0; padding:.2em 1em; border-left:3px solid var(--line); color:var(--muted); }}
  table {{ border-collapse:collapse; width:100%; margin:1em 0; display:block; overflow-x:auto; }}
  th, td {{ border:1px solid var(--line); padding:6px 10px; text-align:left; vertical-align:top; }}
  th {{ background:var(--code); }}
</style></head>
<body><main>
{body}
</main></body></html>"""


def render_page(md_text: str, public_url: str) -> bytes:
    """A docs/ Markdown file as a web page, with the hub's real address filled in."""
    text = md_text.replace("https://<hub address>", public_url)
    try:
        import markdown
        body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    except ImportError:   # hub installed without the [hub] extra: still readable
        import html
        body = f"<pre>{html.escape(text)}</pre>"
    title = next((ln[2:].strip() for ln in text.splitlines() if ln.startswith("# ")), "BioModelBench")
    return PAGE.format(title=title, body=body).encode()


SIGN_THREADS = 32


def _concurrently(fn, items: list) -> list:
    from concurrent.futures import ThreadPoolExecutor
    if len(items) <= 4:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(min(SIGN_THREADS, len(items))) as ex:
        return list(ex.map(fn, items))


def plan_uploads(blobs, agent: str, files: list[dict]) -> list[dict]:
    """An upload URL for each distinct file the hub doesn't hold yet. Existence checks and
    signatures are GCS/IAM calls (~0.1 s each): one by one, a 1,482-file bundle took minutes,
    the client timed out and retried, and each retry left another submission stuck uploading."""
    uniq = list({f["sha256"]: f for f in files}.values())
    def one(f):
        if blobs.has(agent, f["sha256"], f["size"]):
            return None
        return {"sha256": f["sha256"], "size": f["size"], **blobs.upload(agent, f["sha256"], f["size"])}
    return [u for u in _concurrently(one, uniq) if u]


def missing_blobs(blobs, agent: str, files: list[dict]) -> list[str]:
    """Distinct files of a submission the hub doesn't hold (checked concurrently)."""
    uniq = list({f["sha256"]: f for f in files}.values())
    held = _concurrently(lambda f: blobs.has(agent, f["sha256"], f["size"]), uniq)
    return sorted(f["sha256"] for f, h in zip(uniq, held) if not h)


def sign_all(blobs, agent: str, files: list[dict]) -> dict[str, str]:
    """A download URL per distinct file, signed concurrently. On GCS each signature is an
    IAM API call (~0.1 s); a 1,482-file bundle took minutes one by one, past the node's 60 s
    read timeout, so the job was handed out, timed out and requeued over and over."""
    from concurrent.futures import ThreadPoolExecutor
    uniq = {f["sha256"]: f for f in files}
    if len(uniq) <= 4:
        return {sha: blobs.download(agent, sha, f["size"]) for sha, f in uniq.items()}
    with ThreadPoolExecutor(min(SIGN_THREADS, len(uniq))) as ex:
        urls = list(ex.map(lambda f: blobs.download(agent, f["sha256"], f["size"]), uniq.values()))
    return dict(zip(uniq, urls))


def flat_package(files: dict) -> bool:
    """True when a task package is only top-level text files (what old clients read)."""
    return all("/" not in name and isinstance(body, str) for name, body in files.items())


# ------------------------------------------------------------ operator alerts

SETUP_STEP_TEXT = {
    "start": "starting", "gcloud_check": "checking the Google Cloud CLI",
    "fetch_task": "fetching the task", "sandbox_image": "getting the sandbox image",
    "data_check": "checking their data", "hardware_check": "checking the machine",
    "selftest": "the security self-test", "review": "the approval screen",
    "activate": "activating the code", "cloud_setup": "setting up their Google Cloud key and storage",
    "upload": "encrypting and uploading", "notify": "telling the hub about the upload",
    "start_service": "starting the background service"}


def alert(store: HubStore, text: str) -> None:
    """Post to BioModelBench's alert webhook (`bmb-hub alerts --webhook URL`), if set.
    Slack-compatible ({"text": ...}); Teams and most incident tools accept it too."""
    print(f"[bmb-hub alert] {text}", flush=True)
    url = (store.setting("alerts") or {}).get("webhook")
    if not url:
        return
    import urllib.request
    req = urllib.request.Request(url, data=json.dumps({"text": f"[bmb-hub] {text}"}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        urllib.request.urlopen(req, timeout=10).read()
    except Exception as e:
        print(f"[bmb-hub alert] webhook failed: {type(e).__name__}", flush=True)


def alert_setup_failure(store: HubStore, code: str, task: str, step: str, reason: str,
                        detail: str) -> None:
    text = (f"setup failed for task {task} (code {code}) at {SETUP_STEP_TEXT.get(step, step)}: "
            f"{reason}" + (f" ({detail})" if detail else ""))
    threading.Thread(target=alert, args=(store, text), daemon=True).start()


def stalled_hint(prob: dict) -> str:
    """A stalled setup got past the approval screen (the code was activated), so what's
    missing is the step after it: the sealed upload, or the provider's background node."""
    if prob.get("sealed"):
        return ("the provider approved it, but the encrypted upload never completed; ask them to run "
                "the same command again")
    return ("the provider approved it, but their machine's background service never checked in "
            "(not started, or no way out to the hub: proxy, firewall, certificate); ask them for the "
            "output of `~/bmb-eval/bmb-eval status` and `journalctl --user -u bmb-eval -n 30`, or to "
            "run the same command again")


def check_stalled(store: HubStore, now: float | None = None) -> None:
    """Alert once per code whose setup was started but never finished (STALL_S)."""
    now = now or time.time()
    seen = store.setting("stall_alerted") or {}
    for prob in store.setup_problems(now):
        if prob["kind"] == "stalled" and prob["code"] not in seen:
            alert(store, f"setup for task {prob['task']} (code {prob['code']}) was started "
                         f"{int((now - prob['at']) // 60)} min ago and never finished; "
                         f"{stalled_hint(prob)}")
            seen[prob["code"]] = now
    store.set_setting("stall_alerted", seen)


def watch_stalled(store: HubStore, every_s: int = 300) -> threading.Thread:
    def run():
        while True:
            try:
                check_stalled(store)
            except Exception as e:
                print(f"[bmb-hub] stall check failed: {type(e).__name__}: {e}", flush=True)
            time.sleep(every_s)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def make_handler(store: HubStore, blobs, public_url: str, repo_root: Path, max_bundle_gb: int):
    max_bundle = max_bundle_gb << 30
    rel = release.info(repo_root)
    expected_stamp = sandbox_stamp(repo_root)
    tarball = code_tarball(repo_root, rel)
    install_sh = repo_root / "install.sh"
    confidential_sh = repo_root / "confidential.sh"
    verify_sh = repo_root / "deploy" / "confidential" / "verify-image.sh"
    pages = {"/guide": repo_root / "docs" / "PROVIDER_SETUP.md"}
    image_tar = store.state / "sandbox-image.tar"

    def release_view() -> dict:
        """GET /release: which release this is, and the sha256 of each download. The
        image's entry is None until deploy/gcp/hub-deploy.sh has published one."""
        files = {}
        if install_sh.is_file():
            files["install.sh"] = _digest(install_sh.read_text().replace("__BMB_HUB__", public_url).encode())
        if tarball:
            files["bmb-eval.tar.gz"] = _digest(tarball)
        files["sandbox-image.tar"] = published_image(store.state, expected_stamp)
        return {**rel, "sandbox_stamp": expected_stamp, "files": files}

    class Handler(BaseHTTPRequestHandler):
        server_version = "bmb-hub"
        sys_version = ""

        def log_message(self, fmt, *args):
            pass

        # ------------------------------------------------------- plumbing

        _head = False   # HEAD request: send headers only

        def _send(self, code: int, obj=None, body: bytes | None = None,
                  ctype: str = "application/json") -> None:
            if obj is not None:
                body = json.dumps(obj).encode()
            body = body or b""
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if not self._head:
                self.wfile.write(body)

        def _err(self, status: int, code: str, message: str, **extra) -> None:
            self._send(status, {"error": {"code": code, "message": message, **extra}})

        def _bearer(self) -> str:
            h = self.headers.get("Authorization", "")
            return h[7:] if h.startswith("Bearer ") else ""

        def _json_body(self, cap: int = MAX_JSON_BYTES):
            n = int(self.headers.get("Content-Length", "0") or 0)
            if n > cap:
                raise ValueError("body too large")
            return json.loads(self.rfile.read(n) or b"{}")

        def _send_file(self, path: Path, ctype: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
            if self._head:
                return
            with path.open("rb") as f:
                while chunk := f.read(8 << 20):
                    self.wfile.write(chunk)

        def _path(self) -> tuple[str, dict]:
            p, _, qs = self.path.partition("?")
            return p, dict(x.split("=", 1) for x in qs.split("&") if "=" in x)

        # ---------------------------------------------------------- routes

        def do_GET(self):
            p, qs = self._path()
            if p == "/install.sh" and install_sh.is_file():
                body = install_sh.read_text().replace("__BMB_HUB__", public_url).encode()
                return self._send(200, body=body, ctype="text/x-shellscript")
            if p == "/confidential.sh" and confidential_sh.is_file():
                body = confidential_sh.read_text().replace("__BMB_HUB__", public_url).encode()
                return self._send(200, body=body, ctype="text/x-shellscript")
            if p == "/v1/confidential/provenance":
                prov = store.setting("confidential_provenance")
                return self._send(200, prov) if prov else self._err(404, "not_found", "No provenance.")
            if p == "/verify-image.sh" and verify_sh.is_file():
                body = verify_sh.read_text().replace(
                    '${BMB_HUB:-https://eval.biomodelbench.org}', f"${{BMB_HUB:-{public_url}}}").encode()
                return self._send(200, body=body, ctype="text/x-shellscript")
            if p == "/v1/confidential":
                # The sealed-node image providers approve (public: its digest is the point).
                conf = store.setting("confidential")
                return self._send(200, conf) if conf else self._err(
                    404, "not_found", "Confidential mode isn't set up on this hub.")
            if p.startswith("/dl/") and (m := DOWNLOAD_RE.fullmatch(urllib.parse.unquote(p))):
                path = cached_download(store.state / "downloads", *m.groups())
                return (self._send_file(path, "application/octet-stream") if path
                        else self._err(404, "not_found", "No such release file."))
            if p == "/bmb-eval.tar.gz" and tarball:
                return self._send(200, body=tarball, ctype="application/gzip")
            if p in pages and pages[p].is_file():
                return self._send(200, body=render_page(pages[p].read_text(), public_url),
                                  ctype="text/html; charset=utf-8")
            if p.removesuffix(".md") in pages and p.endswith(".md"):
                return self._send(200, body=pages[p.removesuffix(".md")].read_bytes(),
                                  ctype="text/markdown; charset=utf-8")
            if p == "/sandbox-image.tar" and image_tar.is_file():
                return self._send_file(image_tar, "application/x-tar")
            if p == "/sandbox-image.tar.sha256" and image_tar.with_suffix(".tar.sha256").is_file():
                return self._send(200, body=image_tar.with_suffix(".tar.sha256").read_bytes(),
                                  ctype="text/plain")
            if p == "/release":
                return self._send(200, release_view())
            if p == "/SHA256SUMS":           # the same checksums, for `sha256sum -c`
                lines = [f"{v['sha256']}  {k}\n" for k, v in release_view()["files"].items() if v]
                return self._send(200, body="".join(lines).encode(), ctype="text/plain")
            if m := re.fullmatch(r"/v1/enroll/([A-Z0-9-]{14})", p):
                e = store.enrollment(m.group(1))
                me = store.node(self._bearer()) if self._bearer() else None
                if e is None or e[2] == "revoked" or (e[2] is not None and e[2] != me):
                    return self._err(404, "code_invalid",
                                     "This connect code is unknown or already connected from another "
                                     "computer. Run the command on that computer, or ask "
                                     "BioModelBench for a new code.")
                if qs.get("format") != "2" and not flat_package(e[1]):
                    # Clients before task folders write each file as top-level text.
                    return self._err(409, "client_outdated",
                                     "This task needs a newer BioModelBench tool. Run the install "
                                     "command from our email again (it updates the tool), then this "
                                     "command.")
                return self._send(200, {"task": e[0], "files": e[1], "version": e[3]})
            if m := re.fullmatch(r"/v1/blobs/([0-9a-f]{64})", p):
                return self._blob_get(m.group(1))
            if p.startswith("/v1/node/"):
                return self._node_get(p)
            if p.startswith("/v1/operator/"):
                return self._operator("GET", p)
            agent = store.agent(self._bearer())
            if agent is None:
                return self._err(401, "unauthorized", "Missing or invalid token.")
            if p == "/v1/tasks":
                now = time.time()
                rows = store.q("SELECT t.task, t.contract, n.last_seen, n.accepting FROM node_tasks t "
                               "JOIN nodes n ON n.id = t.node_id WHERE t.contract IS NOT NULL")
                cap, used = store.budget(agent)
                return self._send(200, {"tasks": [{
                    "contract": json.loads(c),
                    "node": ("offline" if not seen or now - seen > ONLINE_S
                             else "online" if acc else "paused"),
                } for _, c, seen, acc in rows],
                    "families": [{"name": n, "tasks": t} for n, t in store.families().items()],
                    "budget": {"evaluations": cap, "used": used}})
            if m := re.fullmatch(r"/v1/submissions/([\w-]{1,128})", p):
                if group := store.group_view(m.group(1), agent):
                    return self._send(200, group)
                # Final (test set) results are for BioModelBench only, never the agent.
                row = store.q("SELECT id, task, status, result FROM submissions WHERE id=? AND agent=? "
                              "AND split='validation'", (m.group(1), agent), one=True)
                if not row:
                    return self._err(404, "not_found", "No such submission.")
                out = {"submission_id": row[0], "task": row[1], "status": _public_status(row[2])}
                if row[2] == "done":
                    out["result"] = json.loads(row[3])
                return self._send(200, out)
            self._err(404, "not_found", "No such endpoint.")

        def do_HEAD(self):
            self._head = True
            self.do_GET()

        def do_PUT(self):
            p, qs = self._path()
            m = re.fullmatch(r"/v1/blobs/([0-9a-f]{64})", p)
            if not m or not isinstance(blobs, DiskBlobs):
                return self._err(404, "not_found", "No such endpoint.")
            if store.agent(self._bearer()) is None:
                return self._err(401, "unauthorized", "Missing or invalid token.")
            try:
                size = int(qs.get("size", "-1"))
                n = int(self.headers.get("Content-Length", "-1"))
            except ValueError:
                size = n = -1
            if size < 0 or n != size or size > max_bundle:
                return self._err(400, "bundle_invalid", "Bad upload size.")
            try:
                blobs.cache.put(m.group(1), size, self.rfile.read)
            except EvalError:
                return self._err(400, "bundle_invalid", "Upload does not match its hash.")
            self._send(200, {"ok": True})

        def do_POST(self):
            p, _ = self._path()
            if m := re.fullmatch(r"/v1/enroll/([A-Z0-9-]{14})/report", p):
                # A provider's connect failed: which step, and a reason from a fixed list.
                try:
                    body = self._json_body(MAX_RESULT_BYTES)
                    task = store.report_setup(m.group(1), str(body.get("step", "")),
                                              str(body.get("reason", "")), str(body.get("detail", "")))
                except KeyError:
                    return self._err(404, "not_found", "Unknown code.")
                except ValueError as e:
                    return self._err(400, "invalid", str(e))
                alert_setup_failure(store, m.group(1), task, body["step"], body["reason"],
                                    str(body.get("detail", "")))
                return self._send(204, None)
            if m := re.fullmatch(r"/v1/enroll/([A-Z0-9-]{14})", p):
                try:
                    body = self._json_body()
                    node_id = store.node(self._bearer()) if self._bearer() else None
                    out = store.activate(m.group(1), node_id, str(body.get("name", ""))[:64])
                except (KeyError, ValueError) as e:
                    return self._err(409, "code_invalid", str(e).strip("'\""))
                print(f"[bmb-hub] code {m.group(1)} activated for task {out['task']} by node "
                      f"{out['node_id']} ({str(body.get('name', ''))[:64]})", flush=True)
                return self._send(200, out)
            if p.startswith("/v1/node/"):
                return self._node_post(p)
            if p.startswith("/v1/operator/"):
                return self._operator("POST", p)
            agent = store.agent(self._bearer())
            if agent is None:
                return self._err(401, "unauthorized", "Missing or invalid token.")
            if m := re.fullmatch(r"/v1/tasks/([\w.-]{1,64})/submissions", p):
                return self._submit(agent, m.group(1))
            if m := re.fullmatch(r"/v1/submissions/([\w-]{1,128})/commit", p):
                return self._commit(agent, m.group(1))
            self._err(404, "not_found", "No such endpoint.")

        # ---------------------------------------------------------- operator

        def _operator(self, method: str, p: str) -> None:
            """Automated runs (BioModelBench's harness): issue an attempt's agent token,
            request final evaluations and read their results."""
            # Read the body first, even to refuse: a reply sent before the client has
            # finished sending resets the connection, and the client sees no reply.
            try:
                body = self._json_body() if method == "POST" else {}
            except ValueError:
                body = None
            if not store.is_operator(self._bearer()):
                return self._err(401, "unauthorized", "Missing or invalid token.")
            if method == "POST" and p == "/v1/operator/agent-tokens":
                # Always a budget, and an existing name (a live attempt's token) is only
                # replaced on request: a leaked or buggy caller can't mint unlimited tokens
                # or silently cut off a running attempt.
                try:
                    name, evals = str(body["name"])[:128], body["evaluations"]
                    kind = body.get("kind", "agent")
                    if not name or type(evals) is not int or evals < 0 or kind not in AGENT_KINDS:
                        raise ValueError
                except (ValueError, KeyError, TypeError):
                    return self._err(400, "bad_request",
                                     'Body must be {"name": str, "evaluations": int >= 0, "rotate"?: bool, '
                                     '"kind"?: "agent" | "baseline"}.')
                if store.q("SELECT 1 FROM agents WHERE name=?", (name,), one=True) and not body.get("rotate"):
                    return self._err(409, "name_taken", f"An agent token named {name!r} exists; "
                                     'send "rotate": true to replace it.')
                return self._send(200, {"name": name, "kind": kind,
                                        "token": store.add_agent(name, evals, kind)})
            if method == "GET" and p == "/v1/operator/status":
                # Read-only: what `bmb-hub status` prints, for the ops page.
                return self._send(200, store.operator_status())
            if m := re.fullmatch(r"/v1/operator/submissions/([\w-]{1,128})/final", p):
                if method == "GET":
                    view = store.final_view(m.group(1))
                    return self._send(200, view) if view else self._err(
                        404, "not_found", "No final evaluation for this submission.")
                if not isinstance(body, dict):
                    return self._err(400, "bad_request", 'Body must be {} or {"again": true}.')
                try:
                    return self._send(202, {"finals": store.request_finals(m.group(1), bool(body.get("again")))})
                except KeyError:
                    return self._err(404, "not_found", "No such committed submission.")
            self._err(404, "not_found", "No such endpoint.")

        # ------------------------------------------------------------ agents

        def _budget_left(self, agent: str) -> bool:
            cap, used = store.budget(agent)
            if cap is not None and used >= cap:
                self._err(429, "budget_exhausted",
                          f"This attempt has used all {cap} of its evaluations.")
                return False
            return True

        def _submit(self, agent: str, name: str) -> None:
            """Submit to one task, or to a family (fans out to each dataset)."""
            fam = store.families().get(name)
            enrolled = {r[0] for r in store.q("SELECT task FROM node_tasks")}
            tasks = [t for t in fam if t in enrolled] if fam else ([name] if name in enrolled else [])
            if not tasks:
                return self._err(404, "task_unknown", "Unknown task id.")
            try:
                body = self._json_body()
                manifest, final = body["manifest"], bool(body.get("final_candidate"))
                files = check_manifest(manifest, max_bundle)
            except (ValueError, KeyError, TypeError):
                return self._err(400, "bundle_invalid", "Body must be {\"manifest\": ...}.")
            except EvalError as e:
                return self._err(413 if e.code == "bundle_too_large" else 400, e.code,
                                 e.public()["message"])
            # A final candidate isn't run on validation, so it doesn't spend the budget.
            if not final and not self._budget_left(agent):
                return
            sid = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + secrets.token_hex(4)
            kind = store.agent_kind(agent)
            for task in tasks:
                # A family submission is one row per dataset, sharing a group id.
                rid, gid = (f"{sid}-{task}", sid) if fam else (sid, None)
                store.q("INSERT INTO submissions (id, task, agent, created, status, node_id, result, "
                        "manifest, group_id, kind) VALUES (?,?,?,?, ?, NULL, NULL, ?, ?, ?)",
                        (rid, task, agent, time.time(), "uploading_final" if final else "uploading",
                         json.dumps(manifest), gid, kind))
            uploads = plan_uploads(blobs, agent, files)
            self._send(201, {"submission_id": sid, "tasks": tasks, "uploads": uploads})

        def _commit(self, agent: str, sid: str) -> None:
            row = store.q("SELECT status, manifest FROM submissions WHERE (id=? OR group_id=?) AND agent=? "
                          "LIMIT 1", (sid, sid, agent), one=True)
            if not row:
                return self._err(404, "not_found", "No such submission.")
            if row[0] not in ("uploading", "uploading_final"):
                return self._send(200, {"submission_id": sid, "status": _public_status(row[0])})
            final = row[0] == "uploading_final"
            if not final and not self._budget_left(agent):
                return
            missing = missing_blobs(blobs, agent, json.loads(row[1])["files"])
            if missing:
                return self._err(409, "files_missing", "Some files were not uploaded.",
                                 missing=missing)
            store.q("UPDATE submissions SET status=? WHERE (id=? OR group_id=?) AND agent=? "
                    "AND status=?", ("registered" if final else "queued", sid, sid, agent, row[0]))
            self._send(202, {"submission_id": sid, "status": "registered" if final else "pending"})

        def _blob_get(self, sha: str) -> None:
            if not isinstance(blobs, DiskBlobs) or store.node(self._bearer()) is None:
                return self._err(404, "not_found", "No such file.")
            path = blobs.cache.path(sha)
            if not path.is_file():
                return self._err(404, "not_found", "No such file.")
            self._send_file(path, "application/octet-stream")

        # ------------------------------------------------------------- nodes

        def _node(self) -> str | None:
            node = store.node(self._bearer())
            if node is None:
                self._err(401, "unauthorized", "Missing or invalid node secret.")
            else:
                if store.q("SELECT last_seen FROM nodes WHERE id=?", (node,), one=True)[0] is None:
                    print(f"[bmb-hub] node {node} checked in for the first time", flush=True)
                store.q("UPDATE nodes SET last_seen=? WHERE id=?", (time.time(), node))
            return node

        def _node_get(self, p: str) -> None:
            node = self._node()
            if node is None:
                return
            if p != "/v1/node/next":
                return self._err(404, "not_found", "No such endpoint.")
            with store.lock, store.db:
                row = store.db.execute(
                    "SELECT s.id, s.task, s.agent, s.manifest, s.split, s.kind FROM submissions s "
                    "JOIN node_tasks t "
                    "ON t.task = s.task WHERE t.node_id=? AND s.status='queued' "
                    "ORDER BY s.created LIMIT 1", (node,)).fetchone()
                if row:
                    store.db.execute("UPDATE submissions SET status='assigned', node_id=? "
                                     "WHERE id=?", (node, row[0]))
            if not row:
                return self._send(204)
            manifest = json.loads(row[3])
            downloads = sign_all(blobs, row[2], manifest["files"])
            self._send(200, {"submission_id": row[0], "task": row[1], "submitter": row[2],
                             "split": row[4], "kind": row[5],
                             "manifest": manifest, "bundle_id": manifest_id(manifest),
                             "downloads": downloads})

        def _node_post(self, p: str) -> None:
            node = self._node()
            if node is None:
                return
            try:
                body = self._json_body(MAX_HELLO_BYTES if p == "/v1/node/hello" else MAX_RESULT_BYTES)
            except ValueError:
                return self._err(413, "too_large", "Body too large.")
            if p == "/v1/node/hello":
                owned = {r[0] for r in store.q("SELECT task FROM node_tasks WHERE node_id=?", (node,))}
                for task, contract in (body.get("tasks") or {}).items():
                    if task in owned:
                        store.q("UPDATE node_tasks SET contract=? WHERE task=?",
                                (json.dumps(contract), task))
                store.q("UPDATE nodes SET accepting=? WHERE id=?",
                        (1 if body.get("accepting") else 0, node))
                st = body.get("status")
                if isinstance(st, dict) and len(json.dumps(st)) < 8192:
                    store.q("UPDATE nodes SET status=? WHERE id=?", (json.dumps(st), node))
                # Jobs this node claimed but no longer holds (crash before saving) go back.
                held = set(body.get("held") or [])
                for (sid,) in store.q("SELECT id FROM submissions WHERE node_id=? AND status='assigned'",
                                      (node,)):
                    if sid not in held:
                        store.q("UPDATE submissions SET status='queued', node_id=NULL WHERE id=?", (sid,))
                return self._send(200, {"ok": True, "versions": store.task_versions(node)})
            if p == "/v1/node/confidential":
                # Where a sealed node's data lives (non-secret pointers), for launching it.
                keys = ("PROVIDER_PROJECT_NUMBER", "POOL", "PROVIDER_ID", "KEY", "BUCKET", "digest")
                if not all(isinstance(body.get(k), str) and len(body[k]) < 256 for k in keys):
                    return self._err(400, "invalid", "Missing pointers.")
                store.q("UPDATE nodes SET confidential=? WHERE id=?",
                        (json.dumps({k: body[k] for k in keys}), node))
                return self._send(200, {"ok": True})
            if m := re.fullmatch(r"/v1/node/results/([\w-]{1,128})", p):
                sid = m.group(1)
                # Accept from the node that serves the task, even if the job was
                # re-queued while its result was in flight.
                if not store.q("SELECT 1 FROM submissions s JOIN node_tasks t ON t.task = s.task "
                               "WHERE s.id=? AND t.node_id=?", (sid, node), one=True):
                    return self._err(404, "not_found", "No such job.")
                store.q("UPDATE submissions SET status='done', node_id=?, result=? WHERE id=?",
                        (node, json.dumps(body), sid))
                return self._send(200, {"ok": True})
            self._err(404, "not_found", "No such endpoint.")

    return Handler


def make_blobs(state: Path, public_url: str, bucket: str | None):
    return GCSBlobs(bucket) if bucket else DiskBlobs(state / "blobs", public_url)


# ---------------------------------------------------------------- admin CLI

def _repo_root() -> Path:
    for d in [Path.cwd(), *Path(__file__).resolve().parents]:
        try:
            if (d / "setup.sh").is_file() and (d / "src" / "bmb_eval").is_dir():
                return d
        except OSError:      # e.g. run as the hub user from another user's home (deploy/gcp/hub)
            continue
    return Path.cwd()


def provider_message(public_url: str, code: str, c: Contract) -> str:
    """The email we send a provider. The same for everyone except the code and,
    for GPU tasks, the machine line: their data's format was agreed during
    intake, and `connect` itself shows them exactly what the model will and
    won't see before they approve."""
    lim = c.limits_for("test")   # the most any run of this task needs
    gpus = f"two {lim.gpu_memory_gb or 24} GB NVIDIA GPUs" if lim.gpus == 2 else (
        f"{lim.gpus} NVIDIA GPUs ({lim.gpu_memory_gb or 24} GB each)" if lim.gpus > 1 else
        f"an NVIDIA GPU ({lim.gpu_memory_gb or 24} GB or more of GPU memory)")
    machine = (f"a Linux machine that stays on, with {gpus}, "
               f"{max(64, 128 * lim.gpus)} GB RAM and {max(300, lim.max_bundle_gb * 3)} GB free disk"
               if lim.gpus else
               "a Linux machine that stays on (4 CPUs, 16 GB RAM, 150 GB free disk)")
    return f"""\
Subject: Connecting your data to BioModelBench

Hi! Thanks for taking part. Connecting your data takes one command.

You need {machine}. The server your data is already on is ideal.

On that machine, run this, replacing the last part with the location of the
data you described to us (a file or a folder):

    curl -fsSL {public_url}/install.sh | sh -s -- {code} --data /path/to/your/data

It takes about 15 minutes and may ask for your password once. It checks your
data, tests its own security, then shows you exactly what the model will and
won't see and asks you to type y. That's it. Only scores ever leave your
machine.

If you ever need to:
    ~/bmb-eval/bmb-eval status     check it's running
    ~/bmb-eval/bmb-eval stop       stop everything immediately
    ~/bmb-eval/bmb-eval resume     start again

Guide and FAQ for your IT team: {public_url}/guide
"""


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="bmb-hub", description="BioModelBench hub (operator tool)")
    ap.add_argument("--state", default="hub_state", help="hub data directory")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8790)
    p.add_argument("--public-url", required=True, help="https URL providers and agents use")
    p.add_argument("--bucket", help="Cloud Storage bucket for model files (default: hub disk)")
    p.add_argument("--max-bundle-gb", type=int, default=160)
    p.add_argument("--launch-sealed", action="store_true",
                   help="start sealed nodes' VMs when they have queued work (on GCP)")

    p = sub.add_parser("new-task", help="write a contract + scorer from a column list")
    p.add_argument("task")
    p.add_argument("--inputs", required=True, help="e.g. variant_id:string,sequence:string")
    p.add_argument("--target", required=True, help="e.g. fitness:float")
    p.add_argument("--key", required=True, nargs="+")
    p.add_argument("--kind", default="regression", choices=["regression", "classification"])
    p.add_argument("--tasks-dir", default="tasks")
    p.add_argument("--split", type=float, metavar="FRACTION",
                   help="hold out a test set, scoring this fraction of rows during the benchmark "
                        "(e.g. 0.5); off by default")

    p = sub.add_parser("enroll", help="make a connect code for a task; prints the provider email")
    p.add_argument("task_dir")
    p.add_argument("--public-url", required=True)
    p.add_argument("--replace", action="store_true",
                   help="replace an existing task's package even though it differs")

    p = sub.add_parser("alerts", help="where operator alerts go (failed or stalled provider setups)")
    p.add_argument("--webhook", help="Slack-compatible incoming webhook URL ('' to turn off)")
    p.add_argument("--test", action="store_true", help="send a test alert")

    p = sub.add_parser("retire", help="remove a task from the hub (dry run unless --yes)")
    p.add_argument("task")
    p.add_argument("--yes", action="store_true", help="really remove it (hub.db is backed up first)")

    p = sub.add_parser("revoke", help="cancel an unused connect code")
    p.add_argument("code")

    sub.add_parser("operator-token", help="create or rotate the operator token (API for automated runs)")
    p = sub.add_parser("agent-token", help="create or rotate an agent token (one per attempt)")
    p.add_argument("name")
    p.add_argument("--evaluations", type=int, help="evaluation budget for this attempt, across all datasets")
    p.add_argument("--baseline", action="store_true",
                   help="a token for BioModelBench's own reference model: its finals count against "
                        "the task's baseline_final_evaluations, not the agents' final_evaluations")

    p = sub.add_parser("family", help="group tasks so one submission is scored on all of them")
    p.add_argument("action", choices=["set", "list"])
    p.add_argument("name", nargs="?")
    p.add_argument("tasks", nargs="*")

    sub.add_parser("status", help="nodes, tasks and queue")

    p = sub.add_parser("confidential-image", help="publish the sealed-node image providers approve")
    p.add_argument("image", help="registry/path@sha256:<digest>")
    p.add_argument("--workload-sa", required=True, help="service account the sealed VMs run as")
    p.add_argument("--hardware", default="NVIDIA H100")
    p.add_argument("--gpu-preview", action="store_true",
                   help="ask providers to accept the GPU preview image (H100 in confidential mode)")
    p.add_argument("--source", help="repo@commit the image was built from (deploy/confidential/build.sh)")
    p.add_argument("--provenance", help="Cloud Build SLSA provenance JSON for the image")

    p = sub.add_parser("confidential-launch", help="start a sealed node's VM now (normally automatic)")
    p.add_argument("node_id")

    p = sub.add_parser("confidential-env", help="launch settings for a sealed node (KEY=value lines)")
    p.add_argument("node_id")

    p = sub.add_parser("final", help="evaluate a submission's model on the held-out test set")
    p.add_argument("submission_id")

    p = sub.add_parser("results", help="scores for a task (validation and final)")
    p.add_argument("task")
    p.add_argument("--agent")
    args = ap.parse_args(argv)
    state = Path(args.state)

    if args.cmd == "new-task":
        from . import scaffold
        inputs = scaffold.parse_columns(args.inputs)
        tname, ttype, tnull = scaffold.parse_columns(args.target)[0]
        keys = [c for c in inputs if c[0] in args.key]
        d = scaffold.write_task(args.task, Path(args.tasks_dir), inputs, keys + [(tname, ttype, tnull)],
                                args.key, args.kind, tname, split=args.split)
        print(f"wrote {d}/contract.yaml and score.py: fill in descriptions, then `bmb-hub enroll {d}`")
        return

    store = HubStore(state)
    if args.cmd == "serve":
        url = args.public_url.rstrip("/")
        root = _repo_root()
        handler = make_handler(store, make_blobs(state, url, args.bucket), url, root,
                               args.max_bundle_gb)
        httpd = ThreadingHTTPServer((args.host, args.port), handler)
        if args.launch_sealed:
            from .sealed_launcher import loop
            loop(store)
        watch_stalled(store)
        print(f"bmb-hub {release.describe(root)} on http://{args.host}:{args.port} (public: {url}; "
              f"model files: {'gs://' + args.bucket if args.bucket else 'hub disk'})")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
    elif args.cmd == "enroll":
        task_id = Contract.load(Path(args.task_dir) / "contract.yaml").id
        try:
            new = store.package_files(Path(args.task_dir))
        except ValueError as e:
            raise SystemExit(str(e))
        existing = store.code_for(task_id)
        used = existing and store.enrollment(existing)[2]
        if existing:
            _, old, _, old_version = store.enrollment(existing)
            if new == old:
                print(f"Task {task_id} is already enrolled with this exact package (version "
                      f"{old_version}); nothing changed. The code stays {existing}.")
                return
            if not args.replace:
                changed = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
                raise SystemExit(
                    f"Task {task_id} is already enrolled (version {old_version}"
                    f"{', connected to a provider' if used else ''}) with a different package; "
                    f"these files differ: {', '.join(changed)}.\nIf you mean to replace it, run "
                    f"again with --replace. Use another task id for a separate task.")
        code, c = store.add_enrollment(Path(args.task_dir))
        version = store.enrollment(code)[3]
        if not existing or not used:
            if existing:
                print(f"(updated task {c.id} to version {version}; same code)\n")
            print(provider_message(args.public_url.rstrip("/"), code, c))
        elif store.is_sealed(used):
            print(f"Updated task {c.id} to version {version}. The code stays {code}.\n"
                  f"This provider uses sealed evaluation: ask them to run their original command again\n"
                  f"(the confidential.sh one from our email). It shows the change for them to approve,\n"
                  f"then re-encrypts and uploads. Until then the sealed machine uses the version they approved.")
        else:
            print(f"Updated task {c.id} to version {version}. The code stays {code}.\n"
                  f"The provider's machine will show an update; they review and approve it with:\n"
                  f"    ~/bmb-eval/bmb-eval update\n"
                  f"Until then it keeps evaluating with the version they approved.")
    elif args.cmd == "alerts":
        if args.webhook is not None:
            store.set_setting("alerts", {"webhook": args.webhook} if args.webhook else {})
        url = (store.setting("alerts") or {}).get("webhook")
        print(f"alerts go to {url[:40] + '...' if url else 'nowhere (set --webhook URL)'}")
        if args.test:
            alert(store, "test alert from bmb-hub alerts --test")
    elif args.cmd == "retire":
        if not store.q("SELECT 1 FROM enrollments WHERE task=? UNION SELECT 1 FROM node_tasks WHERE task=?",
                       (args.task, args.task)):
            raise SystemExit(f"No task {args.task!r} on this hub.")
        counts = store.retire(args.task, apply=args.yes)
        print(("Removed" if args.yes else "Would remove") + f" task {args.task}: "
              + ", ".join(f"{n} {what}" for what, n in counts.items()))
        if args.yes:
            print("Finished results are kept. The provider's copy of the data is theirs to delete:\n"
                  "  own machine: ~/bmb-eval/bmb-eval stop, then remove ~/bmb-eval\n"
                  "  sealed: disable their key (the command printed when they connected)")
        else:
            print("Run again with --yes to do it.")
    elif args.cmd == "revoke":
        with store.lock, store.db:
            n = store.db.execute("UPDATE enrollments SET used_by='revoked' WHERE code=? AND used_by IS NULL",
                                 (args.code,)).rowcount
        print(f"revoked {args.code}" if n else f"{args.code} is unknown or already used")
    elif args.cmd == "operator-token":
        print(store.set_operator())
    elif args.cmd == "agent-token":
        print(store.add_agent(args.name, args.evaluations, "baseline" if args.baseline else "agent"))
    elif args.cmd == "family":
        if args.action == "set":
            if not args.name or not args.tasks:
                raise SystemExit("usage: bmb-hub family set <name> <task> [<task> ...]")
            store.set_family(args.name, args.tasks)
            print(f"family {args.name}: {', '.join(args.tasks)}")
        else:
            for n, t in store.families().items():
                print(f"{n}: {', '.join(t)}")
    elif args.cmd == "confidential-image":
        if "@sha256:" not in args.image:
            raise SystemExit("give the image by digest: registry/path@sha256:...")
        store.set_setting("confidential", {
            "image": args.image, "digest": args.image.split("@", 1)[1],
            "workload_sa": args.workload_sa, "hardware": args.hardware,
            "gpu_preview": args.gpu_preview, "source": args.source})
        if args.provenance:
            store.set_setting("confidential_provenance", json.loads(Path(args.provenance).read_text()))
        print(f"providers now approve {args.image.split('@', 1)[1]}; existing sealed tasks need "
              "their providers to re-run their connect command")
    elif args.cmd == "confidential-launch":
        from .sealed_launcher import launch
        row = store.q("SELECT confidential FROM nodes WHERE id=?", (args.node_id,), one=True)
        if not row or not row[0]:
            raise SystemExit(f"node {args.node_id} has no sealed data")
        res = launch(args.node_id, json.loads(row[0]), store.setting("confidential"))
        store.set_setting(f"sealed_launch:{args.node_id}", {"at": time.time(), **res})
        print(json.dumps(res, indent=2))
    elif args.cmd == "confidential-env":
        row = store.q("SELECT confidential FROM nodes WHERE id=?", (args.node_id,), one=True)
        if not row or not row[0]:
            raise SystemExit(f"node {args.node_id} has no sealed data")
        for k, v in json.loads(row[0]).items():
            if k != "digest":
                print(f"{k}={v}")
    elif args.cmd == "status":
        now = time.time()
        root = _repo_root()
        img = published_image(state, sandbox_stamp(root))
        print(f"release {release.describe(root)}; sandbox image: " + (
            "not published yet" if img is None else
            f"stamp {img['stamp'] or 'unstamped'}" + (
                " (current)" if img["current"] else
                f" (STALE: this code expects {sandbox_stamp(root)}; run deploy/gcp/hub deploy)")))
        for nid, name, seen, acc, st in store.q("SELECT id, name, last_seen, accepting, status FROM nodes"):
            age = f"{int(now - seen)}s ago" if seen else "never"
            tasks = [r[0] for r in store.q("SELECT task FROM node_tasks WHERE node_id=?", (nid,))]
            print(f"node {nid} ({name}) last seen {age}, {'accepting' if acc else 'NOT accepting'}: {tasks}")
            for step, kv in sorted((json.loads(st) if st else {}).items(), key=lambda x: x[1].get("at", 0)):
                fields = ", ".join(f"{k}={v}" for k, v in kv.items() if k not in ("step", "at"))
                print(f"    {time.strftime('%m-%d %H:%M', time.gmtime(kv.get('at', 0)))} {step}: {fields}")
        for task, split, status, n in store.q(
                "SELECT task, split, status, COUNT(*) FROM submissions GROUP BY 1, 2, 3"):
            print(f"task {task}: {split} {status} {n}")
        for prob in store.setup_problems(now):
            when = time.strftime('%m-%d %H:%M', time.gmtime(prob["at"]))
            if prob["kind"] == "failed":
                print(f"setup {prob['task']} ({prob['code']}): failed {when} at "
                      f"{SETUP_STEP_TEXT.get(prob['step'], prob['step'])}: {prob['reason']}"
                      + (f" ({prob['detail']})" if prob["detail"] else "")
                      + (" [since completed]" if prob["resolved"] else ""))
            else:
                print(f"setup {prob['task']} ({prob['code']}): STALLED, code used {when} "
                      f"but setup never finished; {stalled_hint(prob)}")
    elif args.cmd == "final":
        for fid in store.request_finals(args.submission_id):
            print(f"queued final evaluation {fid}")
        print("see `bmb-hub results <task or family>` when they're done")
    elif args.cmd == "results" and args.task in store.families():
        tasks = store.families()[args.task]
        marks = ",".join("?" * len(tasks))
        groups = store.q(f"SELECT DISTINCT group_id, agent FROM submissions WHERE task IN ({marks}) "
                         "AND group_id IS NOT NULL" + (" AND agent=?" if args.agent else "")
                         + " ORDER BY created", (*tasks, *([args.agent] if args.agent else [])))
        for gid, agent in groups:
            g = store.group_view(gid) or {}
            r = g.get("result") or {}
            per = " ".join(f"{t}={(d.get('metrics') or {}).get(r.get('headline'))}"
                           for t, d in (r.get("datasets") or {}).items())
            print(f"{gid:28s} {agent:20s} {g.get('status', ''):8s} combined={r.get('combined')}  {per}")
            for split, sid, res in store.q("SELECT split, id, result FROM submissions WHERE id LIKE ? "
                                           "AND split='test' ORDER BY id", (gid + "-%final%",)):
                d = json.loads(res) if res else {}
                score = (d.get("metrics") or {}).get(d.get("headline")) if d.get("status") == "ok" else None
                print(f"    final {sid}: {d.get('headline')}={score}" if score is not None
                      else f"    final {sid}: {(d.get('error') or {}).get('code', 'pending')}")
    elif args.cmd == "results":
        q = ("SELECT id, agent, split, status, result FROM submissions WHERE task=?"
             + (" AND agent=?" if args.agent else "") + " ORDER BY created")
        for sid, agent, split, status, res in store.q(q, (args.task, args.agent) if args.agent else (args.task,)):
            r = json.loads(res) if res else {}
            score = (r.get("metrics") or {}).get(r.get("headline")) if r.get("status") == "ok" else None
            what = (f"{r.get('headline')}={score}" if r.get("status") == "ok"
                    else (r.get("error") or {}).get("code", status))
            print(f"{split:10s} {sid:40s} {agent:24s} {what}")
        for code, task, used in store.q("SELECT code, task, used_by FROM enrollments"):
            print(f"code {code} for {task}: {'used by ' + used if used else 'unused'}")


if __name__ == "__main__":
    main()
