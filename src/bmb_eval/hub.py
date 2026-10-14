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
                  /guide (the provider guide as a web page)

Model files live in Google Cloud Storage (`--bucket`), moved with signed URLs
so they never pass through this server; without a bucket the hub stores
them on its own disk (development and tests). Run behind TLS: see
deploy/gcp/ and docs/OPERATOR.md. Admin commands: `bmb-hub --help`.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import re
import secrets
import sqlite3
import tarfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .blobs import BlobCache, check_manifest, manifest_id
from .contract import Contract
from .errors import EvalError

SCHEMA = """
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
"""
ONLINE_S = 120          # a node that polled within this window is "online"
MAX_JSON_BYTES = 16 << 20   # manifests of large bundles can be a few MB
URL_TTL = datetime.timedelta(hours=24)
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O/1/I


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
        self.bucket = storage.Client(project=project, credentials=self._creds).bucket(bucket)

    def _name(self, agent: str, sha: str) -> str:
        return f"blobs/{_h(agent)[:16]}/{sha}"

    def _sign(self, agent: str, sha: str, method: str, **kw) -> str:
        signer = {}
        if hasattr(self._creds, "service_account_email") and not hasattr(self._creds, "signer"):
            # Compute Engine credentials sign through the IAM API.
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


class HubStore:
    def __init__(self, state: Path):
        self.state = state
        state.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(state / "hub.db", check_same_thread=False)
        self.db.executescript(SCHEMA)
        # Columns added after the first release; older hub.db files gain them here.
        for table, col, decl in [("enrollments", "version", "INTEGER NOT NULL DEFAULT 1"),
                                 ("submissions", "split", "TEXT NOT NULL DEFAULT 'validation'"),
                                 ("submissions", "group_id", "TEXT"),
                                 ("agents", "eval_budget", "INTEGER")]:
            if col not in {r[1] for r in self.db.execute(f"PRAGMA table_info({table})")}:
                self.db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
        self.db.commit()
        self.lock = threading.Lock()

    def q(self, sql: str, args=(), one=False):
        with self.lock, self.db:
            cur = self.db.execute(sql, args)
            return cur.fetchone() if one else cur.fetchall()

    # ---------------------------------------------------------------- auth

    def agent(self, token: str) -> str | None:
        row = self.q("SELECT name FROM agents WHERE token_sha256=?", (_h(token),), one=True)
        return row[0] if row else None

    def node(self, secret: str) -> str | None:
        row = self.q("SELECT id FROM nodes WHERE secret_sha256=?", (_h(secret),), one=True)
        return row[0] if row else None

    def request_final(self, sid: str) -> str:
        """Queue a final (test set) evaluation of an existing submission's model.
        Admin only. The provider's node decides whether it may run (its cap)."""
        row = self.q("SELECT task, agent, manifest FROM submissions WHERE id=? AND split='validation' "
                     "AND status NOT IN ('uploading', 'uploading_final')", (sid,), one=True)
        if not row:
            raise KeyError(f"no committed submission {sid}")
        n = self.q("SELECT COUNT(*) FROM submissions WHERE id LIKE ?", (sid + "-final%",), one=True)[0]
        fid = f"{sid}-final{n + 1}"
        self.q("INSERT INTO submissions (id, task, agent, created, status, node_id, result, manifest, split) "
               "VALUES (?,?,?,?, 'queued', NULL, NULL, ?, 'test')", (fid, row[0], row[1], time.time(), row[2]))
        return fid

    def request_finals(self, sid: str) -> list[str]:
        """Final evaluations for a task submission, or for every dataset of a
        family submission."""
        children = [r[0] for r in self.q("SELECT id FROM submissions WHERE group_id=? "
                                         "AND split='validation' ORDER BY task", (sid,))]
        return [self.request_final(c) for c in (children or [sid])]

    def add_agent(self, name: str, evaluations: int | None = None) -> str:
        """Token for one attempt. `evaluations` caps its submissions across all
        tasks and families (a family submission counts once); None = no cap."""
        token = "bmba_" + secrets.token_urlsafe(32)
        self.q("INSERT OR REPLACE INTO agents (name, token_sha256, eval_budget) VALUES (?,?,?)",
               (name, _h(token), evaluations))
        return token

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
        if not rows:
            return None
        status = {_public_status(r[1]) for r in rows}
        out = {"submission_id": gid, "tasks": [r[0] for r in rows],
               "status": status.pop() if len(status) == 1 else "pending"}
        if out["status"] == "done":
            datasets = {t: json.loads(r) for t, _, r in rows}
            ok = [d for d in datasets.values() if d.get("status") == "ok"]
            headline = ok[0]["headline"] if ok else None
            vals = [d["metrics"].get(headline) for d in ok if d["metrics"].get(headline) is not None]
            out["result"] = {"status": "ok" if vals else "error", "headline": headline,
                             "combined": round(sum(vals) / len(vals), 4) if vals else None,
                             "datasets_scored": len(vals), "datasets": datasets}
        return out

    # ----------------------------------------------------------- enrollment

    def code_for(self, task: str) -> str | None:
        """The task's connect code (one per task, for good), unless revoked."""
        row = self.q("SELECT code FROM enrollments WHERE task=? AND (used_by IS NULL OR used_by <> 'revoked') "
                     "ORDER BY created DESC LIMIT 1", (task,), one=True)
        return row[0] if row else None

    def add_enrollment(self, task_dir: Path) -> tuple[str, Contract]:
        """Package a task: contract, scorer, and any small text files it uses
        (e.g. a gene-panel labels file). Example bundles are not shipped.

        A task keeps one code: enrolling an existing task id again updates its
        package (version + 1) behind the same code. A connected provider's node
        then reports an update, which the provider reviews and approves with
        `bmb-eval update`; until then it keeps serving the approved version."""
        c = Contract.load(task_dir / "contract.yaml")
        files = {p.name: p.read_text() for p in sorted(task_dir.iterdir())
                 if p.is_file() and p.suffix in (".yaml", ".py", ".txt", ".json", ".csv", ".tsv")}
        code = self.code_for(c.id)
        if code:
            self.q("UPDATE enrollments SET files=?, version=version+1 WHERE code=?",
                   (json.dumps(files), code))
        else:
            code = new_code()
            self.q("INSERT INTO enrollments (code, task, files, created, used_by, version) "
                   "VALUES (?,?,?,?,NULL,1)", (code, c.id, json.dumps(files), time.time()))
        return code, c

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
            self.q("INSERT INTO nodes VALUES (?,?,?,?,NULL,0)", (node_id, _h(secret), name, time.time()))
            out["secret"] = secret
        if owner and owner[0] != node_id:
            raise KeyError("task is already served by another node")
        self.q("INSERT OR REPLACE INTO node_tasks VALUES (?,?,NULL)", (task, node_id))
        self.q("UPDATE enrollments SET used_by=? WHERE code=?", (node_id, code))
        out["node_id"] = node_id
        return out


def code_tarball(root: Path) -> bytes | None:
    """The node code providers install, built from this checkout."""
    if not (root / "setup.sh").is_file():
        return None
    include = ["pyproject.toml", "README.md", "setup.sh", "bmb-eval", "sandbox", "src", "docs"]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in include:
            p = root / name
            if p.exists():
                tar.add(p, arcname=name, filter=lambda ti: None
                        if "__pycache__" in ti.name or ti.name.endswith(".pyc") else ti)
    return buf.getvalue()


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


def make_handler(store: HubStore, blobs, public_url: str, repo_root: Path, max_bundle_gb: int):
    max_bundle = max_bundle_gb << 30
    tarball = code_tarball(repo_root)
    install_sh = repo_root / "install.sh"
    pages = {"/guide": repo_root / "docs" / "PROVIDER_SETUP.md"}
    image_tar = store.state / "sandbox-image.tar"

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
            if m := re.fullmatch(r"/v1/enroll/([A-Z0-9-]{14})", p):
                e = store.enrollment(m.group(1))
                me = store.node(self._bearer()) if self._bearer() else None
                if e is None or e[2] == "revoked" or (e[2] is not None and e[2] != me):
                    return self._err(404, "code_invalid",
                                     "This connect code is unknown or used by another machine.")
                return self._send(200, {"task": e[0], "files": e[1], "version": e[3]})
            if m := re.fullmatch(r"/v1/blobs/([0-9a-f]{64})", p):
                return self._blob_get(m.group(1))
            if p.startswith("/v1/node/"):
                return self._node_get(p)
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
            if m := re.fullmatch(r"/v1/enroll/([A-Z0-9-]{14})", p):
                try:
                    body = self._json_body()
                    node_id = store.node(self._bearer()) if self._bearer() else None
                    out = store.activate(m.group(1), node_id, str(body.get("name", ""))[:64])
                except (KeyError, ValueError) as e:
                    return self._err(409, "code_invalid", str(e).strip("'\""))
                return self._send(200, out)
            if p.startswith("/v1/node/"):
                return self._node_post(p)
            agent = store.agent(self._bearer())
            if agent is None:
                return self._err(401, "unauthorized", "Missing or invalid token.")
            if m := re.fullmatch(r"/v1/tasks/([\w.-]{1,64})/submissions", p):
                return self._submit(agent, m.group(1))
            if m := re.fullmatch(r"/v1/submissions/([\w-]{1,128})/commit", p):
                return self._commit(agent, m.group(1))
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
            for task in tasks:
                # A family submission is one row per dataset, sharing a group id.
                rid, gid = (f"{sid}-{task}", sid) if fam else (sid, None)
                store.q("INSERT INTO submissions (id, task, agent, created, status, node_id, result, "
                        "manifest, group_id) VALUES (?,?,?,?, ?, NULL, NULL, ?, ?)",
                        (rid, task, agent, time.time(), "uploading_final" if final else "uploading",
                         json.dumps(manifest), gid))
            uploads, seen = [], set()
            for f in files:
                if f["sha256"] in seen or blobs.has(agent, f["sha256"], f["size"]):
                    continue
                seen.add(f["sha256"])
                uploads.append({"sha256": f["sha256"], "size": f["size"],
                                **blobs.upload(agent, f["sha256"], f["size"])})
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
            missing = sorted({f["sha256"] for f in json.loads(row[1])["files"]
                              if not blobs.has(agent, f["sha256"], f["size"])})
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
                    "SELECT s.id, s.task, s.agent, s.manifest, s.split FROM submissions s JOIN node_tasks t "
                    "ON t.task = s.task WHERE t.node_id=? AND s.status='queued' "
                    "ORDER BY s.created LIMIT 1", (node,)).fetchone()
                if row:
                    store.db.execute("UPDATE submissions SET status='assigned', node_id=? "
                                     "WHERE id=?", (node, row[0]))
            if not row:
                return self._send(204)
            manifest = json.loads(row[3])
            downloads = {f["sha256"]: blobs.download(row[2], f["sha256"], f["size"])
                         for f in manifest["files"]}
            self._send(200, {"submission_id": row[0], "task": row[1], "submitter": row[2],
                             "split": row[4],
                             "manifest": manifest, "bundle_id": manifest_id(manifest),
                             "downloads": downloads})

        def _node_post(self, p: str) -> None:
            node = self._node()
            if node is None:
                return
            try:
                body = self._json_body(64 << 10)
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
                # Jobs this node claimed but no longer holds (crash before saving) go back.
                held = set(body.get("held") or [])
                for (sid,) in store.q("SELECT id FROM submissions WHERE node_id=? AND status='assigned'",
                                      (node,)):
                    if sid not in held:
                        store.q("UPDATE submissions SET status='queued', node_id=NULL WHERE id=?", (sid,))
                return self._send(200, {"ok": True, "versions": store.task_versions(node)})
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
        if (d / "setup.sh").is_file() and (d / "src" / "bmb_eval").is_dir():
            return d
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

    p = sub.add_parser("new-task", help="write a contract + scorer from a column list")
    p.add_argument("task")
    p.add_argument("--inputs", required=True, help="e.g. variant_id:string,sequence:string")
    p.add_argument("--target", required=True, help="e.g. fitness:float")
    p.add_argument("--key", required=True, nargs="+")
    p.add_argument("--kind", default="regression", choices=["regression", "classification"])
    p.add_argument("--tasks-dir", default="tasks")

    p = sub.add_parser("enroll", help="make a connect code for a task; prints the provider email")
    p.add_argument("task_dir")
    p.add_argument("--public-url", required=True)

    p = sub.add_parser("revoke", help="cancel an unused connect code")
    p.add_argument("code")

    p = sub.add_parser("agent-token", help="create or rotate an agent token (one per attempt)")
    p.add_argument("name")
    p.add_argument("--evaluations", type=int, help="evaluation budget for this attempt, across all datasets")

    p = sub.add_parser("family", help="group tasks so one submission is scored on all of them")
    p.add_argument("action", choices=["set", "list"])
    p.add_argument("name", nargs="?")
    p.add_argument("tasks", nargs="*")

    sub.add_parser("status", help="nodes, tasks and queue")

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
                                args.key, args.kind, tname)
        print(f"wrote {d}/contract.yaml and score.py: fill in descriptions, then `bmb-hub enroll {d}`")
        return

    store = HubStore(state)
    if args.cmd == "serve":
        url = args.public_url.rstrip("/")
        handler = make_handler(store, make_blobs(state, url, args.bucket), url, _repo_root(),
                               args.max_bundle_gb)
        httpd = ThreadingHTTPServer((args.host, args.port), handler)
        print(f"bmb-hub on http://{args.host}:{args.port} (public: {url}; "
              f"model files: {'gs://' + args.bucket if args.bucket else 'hub disk'})")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
    elif args.cmd == "enroll":
        existing = store.code_for(Contract.load(Path(args.task_dir) / "contract.yaml").id)
        used = existing and store.enrollment(existing)[2]
        code, c = store.add_enrollment(Path(args.task_dir))
        version = store.enrollment(code)[3]
        if not existing or not used:
            if existing:
                print(f"(updated task {c.id} to version {version}; same code)\n")
            print(provider_message(args.public_url.rstrip("/"), code, c))
        else:
            print(f"Updated task {c.id} to version {version}. The code stays {code}.\n"
                  f"The provider's machine will show an update; they review and approve it with:\n"
                  f"    ~/bmb-eval/bmb-eval update\n"
                  f"Until then it keeps evaluating with the version they approved.")
    elif args.cmd == "revoke":
        with store.lock, store.db:
            n = store.db.execute("UPDATE enrollments SET used_by='revoked' WHERE code=? AND used_by IS NULL",
                                 (args.code,)).rowcount
        print(f"revoked {args.code}" if n else f"{args.code} is unknown or already used")
    elif args.cmd == "agent-token":
        print(store.add_agent(args.name, args.evaluations))
    elif args.cmd == "family":
        if args.action == "set":
            if not args.name or not args.tasks:
                raise SystemExit("usage: bmb-hub family set <name> <task> [<task> ...]")
            store.set_family(args.name, args.tasks)
            print(f"family {args.name}: {', '.join(args.tasks)}")
        else:
            for n, t in store.families().items():
                print(f"{n}: {', '.join(t)}")
    elif args.cmd == "status":
        now = time.time()
        for nid, name, seen, acc in store.q("SELECT id, name, last_seen, accepting FROM nodes"):
            age = f"{int(now - seen)}s ago" if seen else "never"
            tasks = [r[0] for r in store.q("SELECT task FROM node_tasks WHERE node_id=?", (nid,))]
            print(f"node {nid} ({name}) last seen {age}, {'accepting' if acc else 'NOT accepting'}: {tasks}")
        for task, split, status, n in store.q(
                "SELECT task, split, status, COUNT(*) FROM submissions GROUP BY 1, 2, 3"):
            print(f"task {task}: {split} {status} {n}")
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
