"""BioModelBench hub: the one service we run. Providers' nodes connect *out*
to it, so providers open no ports and run no web server.

Holds only agent submissions (model bundles) and the gated results nodes
send back. Never sees provider data. Standard library only.

    agents   POST /v1/tasks/<task>/submissions   bundle .tar.gz -> {submission_id}
             GET  /v1/submissions/<id>           {status: pending|done, result?}
             GET  /v1/tasks                      public contracts + node status
    nodes    POST /v1/node/hello                 report tasks/contracts, recover jobs
             GET  /v1/node/next                  claim the next job (204 if none)
             GET  /v1/node/bundles/<id>          download its bundle
             POST /v1/node/results/<id>          deliver the gated payload
    enroll   GET  /v1/enroll/<code>              task package (contract + scorer)
             POST /v1/enroll/<code>              activate: bind task to a node
    public   GET  /install.sh, /bmb-eval.tar.gz, /sandbox-image.tar, /guide

Run behind TLS (see docs/OPERATOR.md). Admin commands: `bmb-hub --help`.
"""
from __future__ import annotations

import argparse
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

from .contract import Contract

SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (name TEXT PRIMARY KEY, token_sha256 TEXT NOT NULL);
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
    status TEXT NOT NULL,          -- queued | assigned | done
    node_id TEXT, result TEXT);
"""
ONLINE_S = 120          # a node that polled within this window is "online"
MAX_RESULT_BYTES = 64 << 10
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O/1/I


def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def new_code() -> str:
    return "-".join("".join(secrets.choice(CODE_ALPHABET) for _ in range(4)) for _ in range(3))


class HubStore:
    def __init__(self, state: Path):
        self.state = state
        (state / "bundles").mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(state / "hub.db", check_same_thread=False)
        self.db.executescript(SCHEMA)
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

    def add_agent(self, name: str) -> str:
        token = "bmba_" + secrets.token_urlsafe(32)
        self.q("INSERT OR REPLACE INTO agents VALUES (?,?)", (name, _h(token)))
        return token

    # ----------------------------------------------------------- enrollment

    def add_enrollment(self, task_dir: Path) -> tuple[str, Contract]:
        c = Contract.load(task_dir / "contract.yaml")
        files = {p.name: p.read_text() for p in sorted(task_dir.iterdir())
                 if p.is_file() and p.suffix in (".yaml", ".py")}
        code = new_code()
        self.q("INSERT INTO enrollments VALUES (?,?,?,?,NULL)",
               (code, c.id, json.dumps(files), time.time()))
        return code, c

    def enrollment(self, code: str) -> tuple[str, dict, str | None] | None:
        row = self.q("SELECT task, files, used_by FROM enrollments WHERE code=?", (code,), one=True)
        return (row[0], json.loads(row[1]), row[2]) if row else None

    def activate(self, code: str, node_id: str | None, name: str) -> dict:
        """Bind the code's task to a node, creating the node if needed."""
        e = self.enrollment(code)
        if e is None or e[2] is not None:
            raise KeyError("code unknown or already used")
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
    include = ["pyproject.toml", "README.md", "setup.sh", "bmb-eval", "sandbox", "src",
               "docs", "examples", "tasks/example_regression"]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in include:
            p = root / name
            if p.exists():
                tar.add(p, arcname=name, filter=lambda ti: None
                        if "__pycache__" in ti.name or ti.name.endswith(".pyc") else ti)
    return buf.getvalue()


def make_handler(store: HubStore, public_url: str, repo_root: Path, max_upload_mb: int):
    max_upload = max_upload_mb << 20
    tarball = code_tarball(repo_root)
    install_sh = (repo_root / "install.sh")
    image_tar = store.state / "sandbox-image.tar"

    class Handler(BaseHTTPRequestHandler):
        server_version = "bmb-hub"
        sys_version = ""

        def log_message(self, fmt, *args):
            pass

        # ------------------------------------------------------- plumbing

        def _send(self, code: int, obj=None, body: bytes | None = None,
                  ctype: str = "application/json") -> None:
            if obj is not None:
                body = json.dumps(obj).encode()
            body = body or b""
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _err(self, status: int, code: str, message: str) -> None:
            self._send(status, {"error": {"code": code, "message": message}})

        def _bearer(self) -> str:
            h = self.headers.get("Authorization", "")
            return h[7:] if h.startswith("Bearer ") else ""

        def _json_body(self, cap: int = MAX_RESULT_BYTES):
            n = int(self.headers.get("Content-Length", "0") or 0)
            if n > cap:
                raise ValueError("body too large")
            return json.loads(self.rfile.read(n) or b"{}")

        def _send_file(self, path: Path, ctype: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
            with path.open("rb") as f:
                while chunk := f.read(1 << 20):
                    self.wfile.write(chunk)

        # ---------------------------------------------------------- routes

        def do_GET(self):
            p = self.path.split("?")[0]
            if p == "/install.sh" and install_sh.is_file():
                body = install_sh.read_text().replace("__BMB_HUB__", public_url).encode()
                return self._send(200, body=body, ctype="text/x-shellscript")
            if p == "/guide" and (repo_root / "docs" / "PROVIDER_SETUP.md").is_file():
                return self._send(200, body=(repo_root / "docs" / "PROVIDER_SETUP.md").read_bytes(),
                                  ctype="text/markdown; charset=utf-8")
            if p == "/bmb-eval.tar.gz" and tarball:
                return self._send(200, body=tarball, ctype="application/gzip")
            if p == "/sandbox-image.tar" and image_tar.is_file():
                return self._send_file(image_tar, "application/x-tar")
            if m := re.fullmatch(r"/v1/enroll/([A-Z0-9-]{14})", p):
                e = store.enrollment(m.group(1))
                if e is None or e[2] is not None:
                    return self._err(404, "code_invalid", "This connect code is unknown or already used.")
                return self._send(200, {"task": e[0], "files": e[1]})
            if p.startswith("/v1/node/"):
                return self._node_get(p)
            agent = store.agent(self._bearer())
            if agent is None:
                return self._err(401, "unauthorized", "Missing or invalid token.")
            if p == "/v1/tasks":
                now = time.time()
                rows = store.q("SELECT t.task, t.contract, n.last_seen, n.accepting FROM node_tasks t "
                               "JOIN nodes n ON n.id = t.node_id WHERE t.contract IS NOT NULL")
                return self._send(200, {"tasks": [{
                    "contract": json.loads(c),
                    "node": ("offline" if not seen or now - seen > ONLINE_S
                             else "online" if acc else "paused"),
                } for _, c, seen, acc in rows]})
            if m := re.fullmatch(r"/v1/submissions/([\w-]{1,64})", p):
                row = store.q("SELECT id, task, status, result FROM submissions WHERE id=? AND agent=?",
                              (m.group(1), agent), one=True)
                if not row:
                    return self._err(404, "not_found", "No such submission.")
                out = {"submission_id": row[0], "task": row[1],
                       "status": "done" if row[2] == "done" else "pending"}
                if row[2] == "done":
                    out["result"] = json.loads(row[3])
                return self._send(200, out)
            self._err(404, "not_found", "No such endpoint.")

        def do_POST(self):
            p = self.path.split("?")[0]
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
            self._err(404, "not_found", "No such endpoint.")

        def _submit(self, agent: str, task: str) -> None:
            if not store.q("SELECT 1 FROM node_tasks WHERE task=?", (task,), one=True):
                return self._err(404, "task_unknown", "Unknown task id.")
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                n = 0
            if n <= 0:
                return self._err(400, "bundle_invalid", "Empty upload.")
            if n > max_upload:
                self.close_connection = True
                return self._err(413, "bundle_too_large", "Bundle exceeds the upload limit.")
            sid = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + secrets.token_hex(4)
            path = store.state / "bundles" / f"{sid}.tar.gz"
            with path.open("wb") as f:
                left = n
                while left > 0:
                    chunk = self.rfile.read(min(left, 1 << 20))
                    if not chunk:
                        break
                    f.write(chunk)
                    left -= len(chunk)
            store.q("INSERT INTO submissions VALUES (?,?,?,?, 'queued', NULL, NULL)",
                    (sid, task, agent, time.time()))
            self._send(202, {"submission_id": sid, "status": "pending"})

        # ----------------------------------------------------------- nodes

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
            if p == "/v1/node/next":
                with store.lock, store.db:
                    row = store.db.execute(
                        "SELECT s.id, s.task, s.agent FROM submissions s JOIN node_tasks t "
                        "ON t.task = s.task WHERE t.node_id=? AND s.status='queued' "
                        "ORDER BY s.created LIMIT 1", (node,)).fetchone()
                    if row:
                        store.db.execute("UPDATE submissions SET status='assigned', node_id=? "
                                         "WHERE id=?", (node, row[0]))
                if not row:
                    return self._send(204)
                return self._send(200, {"submission_id": row[0], "task": row[1], "submitter": row[2]})
            if m := re.fullmatch(r"/v1/node/bundles/([\w-]{1,64})", p):
                ok = store.q("SELECT 1 FROM submissions WHERE id=? AND node_id=? AND status='assigned'",
                             (m.group(1), node), one=True)
                path = store.state / "bundles" / f"{m.group(1)}.tar.gz"
                if not ok or not path.is_file():
                    return self._err(404, "not_found", "No such job.")
                return self._send_file(path, "application/gzip")
            self._err(404, "not_found", "No such endpoint.")

        def _node_post(self, p: str) -> None:
            node = self._node()
            if node is None:
                return
            try:
                body = self._json_body()
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
                return self._send(200, {"ok": True})
            if m := re.fullmatch(r"/v1/node/results/([\w-]{1,64})", p):
                sid = m.group(1)
                if not store.q("SELECT 1 FROM submissions WHERE id=? AND node_id=?", (sid, node), one=True):
                    return self._err(404, "not_found", "No such job.")
                store.q("UPDATE submissions SET status='done', result=? WHERE id=?",
                        (json.dumps(body), sid))
                (store.state / "bundles" / f"{sid}.tar.gz").unlink(missing_ok=True)
                return self._send(200, {"ok": True})
            self._err(404, "not_found", "No such endpoint.")

    return Handler


# ---------------------------------------------------------------- admin CLI

def _repo_root() -> Path:
    for d in [Path.cwd(), *Path(__file__).resolve().parents]:
        if (d / "setup.sh").is_file() and (d / "src" / "bmb_eval").is_dir():
            return d
    return Path.cwd()


def provider_message(public_url: str, code: str, c: Contract) -> str:
    """What we send the provider: everything they need, nothing else."""
    model_sees = [f.name for f in c.inputs]
    answer = [f.name for f in c.answers if f.name not in c.key]
    cols = ", ".join(dict.fromkeys(model_sees + [f.name for f in c.answers]))
    return f"""\
Subject: Connecting your data to BioModelBench ({c.title})

Hi! Here's how to connect your data. It takes one command.

You need:
  • A Linux machine that stays on (4 CPUs, 16 GB RAM). The server your data
    is already on is ideal.
  • Your data as one file (CSV is fine) with these columns:
        {cols}
    Other columns are fine; they're ignored. The model will only ever see
    {", ".join(model_sees)}. It never sees {", ".join(answer)}.

On that machine, run this (put in the path to your file):

    curl -fsSL {public_url}/install.sh | sh -s -- {code} --data /path/to/your_data.csv

It takes about 10 minutes and may ask for your password once. It checks your
file, tests its own security, then shows you exactly what you're agreeing to
and asks you to type y. That's it. Only scores ever leave your machine.

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
    p.add_argument("--max-upload-mb", type=int, default=2048)

    p = sub.add_parser("new-task", help="write a contract + scorer from a column list")
    p.add_argument("task")
    p.add_argument("--inputs", required=True, help="e.g. variant_id:string,sequence:string")
    p.add_argument("--target", required=True, help="e.g. fitness:float")
    p.add_argument("--key", required=True, nargs="+")
    p.add_argument("--kind", default="regression", choices=["regression", "classification"])
    p.add_argument("--tasks-dir", default="tasks")

    p = sub.add_parser("enroll", help="make a connect code for a task; prints the provider message")
    p.add_argument("task_dir")
    p.add_argument("--public-url", required=True)

    p = sub.add_parser("agent-token", help="create or rotate an agent token")
    p.add_argument("name")

    sub.add_parser("status", help="nodes, tasks and queue")
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
        httpd = ThreadingHTTPServer((args.host, args.port), make_handler(
            store, args.public_url.rstrip("/"), _repo_root(), args.max_upload_mb))
        print(f"bmb-hub on http://{args.host}:{args.port} (public: {args.public_url})")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
    elif args.cmd == "enroll":
        code, c = store.add_enrollment(Path(args.task_dir))
        print(provider_message(args.public_url.rstrip("/"), code, c))
    elif args.cmd == "agent-token":
        print(store.add_agent(args.name))
    elif args.cmd == "status":
        now = time.time()
        for nid, name, seen, acc in store.q("SELECT id, name, last_seen, accepting FROM nodes"):
            age = f"{int(now - seen)}s ago" if seen else "never"
            tasks = [r[0] for r in store.q("SELECT task FROM node_tasks WHERE node_id=?", (nid,))]
            print(f"node {nid} ({name}) last seen {age}, {'accepting' if acc else 'NOT accepting'}: {tasks}")
        for task, status, n in store.q("SELECT task, status, COUNT(*) FROM submissions GROUP BY 1, 2"):
            print(f"task {task}: {status} {n}")
        for code, task, used in store.q("SELECT code, task, used_by FROM enrollments"):
            print(f"code {code} for {task}: {'used by ' + used if used else 'unused'}")


if __name__ == "__main__":
    main()
