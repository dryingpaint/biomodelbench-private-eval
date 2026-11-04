"""Hub storage and job hand-out at scale: manifests stored once, links signed before
a node asks, abandoned uploads dropped, concurrent polls, and the move of manifests
out of older hub.db files."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from bmb_eval import hub as hub_mod
from bmb_eval.blobs import manifest_id

REPO = Path(__file__).resolve().parents[1]
NODE_SECRET = "bmbn_test"


class SigningBlobs(hub_mod.DiskBlobs):
    """Disk storage with GCS's signed-link cache: download() signs (and counts it), signed()
    only looks a link up."""

    def __init__(self, *a):
        super().__init__(*a)
        self.links, self.signatures = {}, []

    def download(self, agent, sha, size, ahead=0):
        self.signatures.append(sha)
        self.links[(agent, sha)] = link = {"url": f"https://signed/{agent}/{sha}/{len(self.signatures)}",
                                           "headers": {}, "auth": False}
        return link

    def signed(self, agent, sha):
        return self.links.get((agent, sha))


def _serve(tmp_path, blobs_cls=hub_mod.DiskBlobs):
    store = hub_mod.HubStore(tmp_path / "hub_state")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), None)
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    blobs = blobs_cls(tmp_path / "hub_state" / "blobs", url)
    httpd.RequestHandlerClass = handler = hub_mod.make_handler(store, blobs, url, REPO, 64)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    store.q("INSERT INTO nodes (id, secret_sha256, name, created, last_seen, accepting) "
            "VALUES ('node_1', ?, 'n', 0, NULL, 1)", (hub_mod._h(NODE_SECRET),))
    return store, url, blobs, handler, httpd


@pytest.fixture
def hub(tmp_path):
    store, url, blobs, handler, httpd = _serve(tmp_path)
    yield store, url, blobs, handler
    httpd.shutdown()


@pytest.fixture
def signing_hub(tmp_path):
    store, url, blobs, handler, httpd = _serve(tmp_path, SigningBlobs)
    yield store, url, blobs, handler
    httpd.shutdown()


def _req(url, path, token, body=None, method=None):
    req = urllib.request.Request(url + path, data=None if body is None else json.dumps(body).encode(),
                                 method=method or ("POST" if body is not None else "GET"),
                                 headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, (json.loads(raw) if (raw := r.read()) else None)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _serve_tasks(store, *tasks):
    for t in tasks:
        store.q("INSERT INTO node_tasks VALUES (?, 'node_1', NULL)", (t,))


def _bundle(n_files=3, tag="a"):
    """A manifest and the bytes of each of its files."""
    data = {f"w{i}.bin": f"{tag}{i}".encode() * 10 for i in range(n_files)}
    data["predict.py"] = f"# {tag}\n".encode()
    files = [{"path": p, "sha256": hashlib.sha256(b).hexdigest(), "size": len(b)} for p, b in data.items()]
    return {"files": files}, {f["sha256"]: data[f["path"]] for f in files}


def _upload(blobs, contents):
    for sha, b in contents.items():
        blobs.cache.put(sha, len(b), io.BytesIO(b).read)


def _drain(presigner):
    presigner._bundles.submit(lambda: None).result(timeout=30)


def test_family_manifest_is_stored_once(hub):
    store, url, blobs, _ = hub
    tasks = [f"ds{i}" for i in range(45)]
    _serve_tasks(store, *tasks)
    store.set_family("fam", tasks)
    token = store.add_agent("a1")
    manifest, contents = _bundle()
    _upload(blobs, contents)
    code, sub = _req(url, "/v1/tasks/fam/submissions", token, {"manifest": manifest})
    assert code == 201 and len(sub["tasks"]) == 45
    _req(url, "/v1/tasks/fam/submissions", token, {"manifest": manifest})      # same bundle again
    assert store.q("SELECT COUNT(*) FROM manifests", one=True)[0] == 1
    assert store.q("SELECT COUNT(*), COUNT(DISTINCT manifest_id) FROM submissions", one=True) == (90, 1)
    assert _req(url, f"/v1/submissions/{sub['submission_id']}/commit", token, {})[0] == 202
    code, job = _req(url, "/v1/node/next", NODE_SECRET)
    assert code == 200 and job["manifest"] == manifest and job["bundle_id"] == manifest_id(manifest)
    assert set(job["downloads"]) == set(contents)


def test_concurrent_polls_never_get_the_same_job(hub):
    store, url, *_ = hub
    _serve_tasks(store, "t")
    store.add_agent("a1")
    manifest, _ = _bundle()
    for i in range(60):
        store.add_submission(f"s{i:02d}", ["t"], "a1", manifest, family=False, final=False)
    store.q("UPDATE submissions SET status='queued'")
    got, lock = [], threading.Lock()

    def poll():
        while True:
            code, job = _req(url, "/v1/node/next", NODE_SECRET)
            if code == 204:
                return
            with lock:
                got.append(job["submission_id"])

    threads = [threading.Thread(target=poll) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(got) == [f"s{i:02d}" for i in range(60)]          # each job exactly once
    assert store.q("SELECT COUNT(*) FROM submissions WHERE status='assigned' AND node_id='node_1'",
                   one=True)[0] == 60


def test_links_are_signed_at_commit_not_at_hand_out(signing_hub):
    store, url, blobs, handler = signing_hub
    _serve_tasks(store, "t")
    token = store.add_agent("a1")
    manifest, contents = _bundle(n_files=40)
    _upload(blobs, contents)
    sid = _req(url, "/v1/tasks/t/submissions", token, {"manifest": manifest})[1]["submission_id"]
    assert blobs.signatures == []                          # nothing signed for an upload in progress
    _req(url, f"/v1/submissions/{sid}/commit", token, {})
    _drain(handler.presigner)
    assert sorted(blobs.signatures) == sorted(contents)   # each distinct file, once, at commit
    signed_at_commit = len(blobs.signatures)
    code, job = _req(url, "/v1/node/next", NODE_SECRET)
    assert code == 200 and len(blobs.signatures) == signed_at_commit   # the hand-out only looked up
    assert job["downloads"] == {sha: blobs.links[("a1", sha)] for sha in contents}

    # Not signed ahead (say the hub restarted): the hand-out still works, signing what's missing.
    sid2 = _req(url, "/v1/tasks/t/submissions", token, {"manifest": manifest})[1]["submission_id"]
    store.q("UPDATE submissions SET status='queued' WHERE id=?", (sid2,))
    blobs.links.clear()
    code, job = _req(url, "/v1/node/next", NODE_SECRET)
    assert code == 200 and job["submission_id"] == sid2 and set(job["downloads"]) == set(contents)
    assert len(blobs.signatures) == 2 * signed_at_commit


def test_final_candidates_are_not_signed_until_their_final_is_queued(signing_hub):
    store, url, blobs, handler = signing_hub
    _serve_tasks(store, "t")
    token = store.add_agent("a1")
    manifest, contents = _bundle()
    _upload(blobs, contents)
    sid = _req(url, "/v1/tasks/t/submissions", token,
               {"manifest": manifest, "final_candidate": True})[1]["submission_id"]
    _req(url, f"/v1/submissions/{sid}/commit", token, {})
    handler.presigner.sweep()
    _drain(handler.presigner)
    assert blobs.signatures == []
    store.request_final(sid)
    handler.presigner.sweep()                              # what serve runs every minute
    _drain(handler.presigner)
    assert sorted(blobs.signatures) == sorted(contents)


def test_presigner_renews_links_of_jobs_that_wait():
    """A link reused at hand-out keeps at least URL_REUSE_MIN; links of a job that waits in
    the queue are renewed before they get that close, so the hand-out never signs."""
    pytest.importorskip("google.cloud.storage")
    g = object.__new__(hub_mod.GCSBlobs)
    g._signed, g._signed_lock = {}, threading.Lock()
    calls = []
    g._sign = lambda agent, sha, method: calls.append(sha) or f"https://s/{sha}/{len(calls)}"
    manifest, contents = _bundle()

    class Store:
        def q(self, sql, args=(), one=False):
            return [("a1", manifest_id(manifest))]

        def manifest(self, mid):
            return manifest

    p = hub_mod.Presigner(Store(), g)
    p.sweep()
    _drain(p)
    assert len(calls) == len(contents)
    p.sweep()                                              # checked within RENEW_S: nothing to do
    _drain(p)
    assert len(calls) == len(contents)
    sha = next(iter(contents))
    first = g.signed("a1", sha)["url"]
    # Later: one link has under 2 * RENEW_S of reuse left, the others plenty.
    until, url = g._signed[("a1", sha)]
    g._signed[("a1", sha)] = (time.time() + hub_mod.RENEW_S, url)
    p._checked.clear()                                     # RENEW_S has passed
    p.sweep()
    _drain(p)
    assert len(calls) == len(contents) + 1 and g.signed("a1", sha)["url"] != first
    left = g._signed[("a1", sha)][0] - time.time()
    assert left > 2 * hub_mod.RENEW_S                      # good past the next check
    # A link past its reuse window is never handed out as it is.
    g._signed[("a1", sha)] = (time.time() - 1, url)
    assert g.signed("a1", sha) is None


def test_abandoned_uploads_are_dropped_and_never_counted(hub):
    store, url, blobs, _ = hub
    _serve_tasks(store, "t")
    token = store.add_agent("a1", evaluations=1)
    old_m, _ = _bundle(tag="old")
    new_m, contents = _bundle(tag="new")
    _upload(blobs, contents)
    for i in range(3):        # a client that kept timing out and retrying: never committed
        assert _req(url, "/v1/tasks/t/submissions", token, {"manifest": old_m})[0] == 201
    _req(url, "/v1/tasks/t/submissions", token, {"manifest": old_m, "final_candidate": True})
    assert store.budget("a1") == (1, 0)                    # uncommitted uploads spend nothing
    store.q("UPDATE submissions SET created = created - ?", (hub_mod.ABANDON_S + 60,))
    sid = _req(url, "/v1/tasks/t/submissions", token, {"manifest": new_m})[1]["submission_id"]
    assert store.drop_abandoned() == 4
    assert [r[0] for r in store.q("SELECT id FROM submissions")] == [sid]   # the fresh one stays
    assert [r[0] for r in store.q("SELECT id FROM manifests")] == [manifest_id(new_m)]
    assert _req(url, f"/v1/submissions/{sid}/commit", token, {})[0] == 202
    assert store.budget("a1") == (1, 1)
    store.q("UPDATE submissions SET created = created - ?", (hub_mod.ABANDON_S + 60,))
    assert store.drop_abandoned() == 0                     # committed work is never dropped
    assert store.q("SELECT status FROM submissions WHERE id=?", (sid,), one=True)[0] == "queued"


def test_older_hub_db_moves_manifests_out_once(tmp_path):
    """A hub.db from before the manifests table: every row its own copy of the manifest.
    They move to one row per bundle; ids, hand-outs and finals work as before."""
    h = tmp_path / "hub"
    h.mkdir()
    db = sqlite3.connect(h / "hub.db")
    db.execute("CREATE TABLE agents (name TEXT PRIMARY KEY, token_sha256 TEXT NOT NULL, eval_budget INTEGER, "
               "kind TEXT NOT NULL DEFAULT 'agent')")
    db.execute("CREATE TABLE submissions (id TEXT PRIMARY KEY, task TEXT NOT NULL, agent TEXT NOT NULL, "
               "created REAL NOT NULL, status TEXT NOT NULL, node_id TEXT, result TEXT, manifest TEXT NOT NULL, "
               "split TEXT NOT NULL DEFAULT 'validation', group_id TEXT, kind TEXT NOT NULL DEFAULT 'agent')")
    db.execute("CREATE INDEX submissions_group ON submissions (group_id, split)")
    fam_m, solo_m = _bundle(tag="fam")[0], _bundle(tag="solo")[0]
    ok = json.dumps({"status": "ok", "headline": "r", "metrics": {"r": 0.5}})
    rows = [(f"g-ds{i}", f"ds{i}", "a1", i, "done", None, ok, json.dumps(fam_m), "g") for i in range(5)]
    rows.append(("s", "ds0", "a1", 9, "queued", None, None, json.dumps(solo_m), None))
    db.executemany("INSERT INTO submissions (id, task, agent, created, status, node_id, result, manifest, "
                   "group_id) VALUES (?,?,?,?,?,?,?,?,?)", rows)
    db.commit()
    db.close()

    store = hub_mod.HubStore(h)
    assert "manifest" not in hub_mod._columns(sqlite3.connect(h / "hub.db"), "submissions")
    assert sorted(r[0] for r in store.q("SELECT id FROM manifests")) == sorted(
        [manifest_id(fam_m), manifest_id(solo_m)])
    assert store.q("SELECT COUNT(*) FROM submissions WHERE manifest_id=?", (manifest_id(fam_m),),
                   one=True)[0] == 5
    assert store.manifest(store.q("SELECT manifest_id FROM submissions WHERE id='s'", one=True)[0]) == solo_m
    assert store.group_view("g")["status"] == "done"
    fid = store.request_final("g-ds2")
    assert store.q("SELECT manifest_id FROM submissions WHERE id=?", (fid,), one=True)[0] == manifest_id(fam_m)
    assert len(list(h.glob("hub.db.bak-*-manifests"))) == 1          # backed up before the move
    assert sqlite3.connect(next(h.glob("hub.db.bak-*-manifests"))).execute(
        "SELECT COUNT(*) FROM submissions WHERE manifest <> ''").fetchone()[0] == 6
    hub_mod.HubStore(h)                                    # opening again changes nothing
    assert len(list(h.glob("hub.db.bak-*-manifests"))) == 1
    assert store.q("SELECT COUNT(*) FROM manifests", one=True)[0] == 2


def test_reads_are_not_held_up_by_a_write(tmp_path):
    """A request reads while another holds the write lock (WAL, a connection each): on
    2026-10-30 one lock around one connection queued every request behind the slowest."""
    store = hub_mod.HubStore(tmp_path)
    store.add_agent("a1")
    held, release = threading.Event(), threading.Event()

    def writer():
        with store.tx() as db:
            db.execute("UPDATE agents SET eval_budget=5")
            held.set()
            release.wait(10)

    t = threading.Thread(target=writer)
    t.start()
    held.wait(10)
    started = time.time()
    assert store.q("SELECT eval_budget FROM agents", one=True)[0] is None   # the committed state
    assert time.time() - started < 1
    release.set()
    t.join()
    assert store.q("SELECT eval_budget FROM agents", one=True)[0] == 5


# ------------------------------------------------------------------ backups

def test_local_backups_keep_only_the_newest(tmp_path):
    store = hub_mod.HubStore(tmp_path)
    old = []
    for i in range(hub_mod.LOCAL_BACKUPS + 2):
        p = tmp_path / f"hub.db.bak-2026100{i}-old"
        p.write_bytes(b"x")
        os.utime(p, (1000 + i, 1000 + i))
        old.append(p)
    with store._conn() as db:
        new = hub_mod.local_backup(db, tmp_path, "retire-t")
    left = sorted(tmp_path.glob("hub.db.bak-*"))
    assert len(left) == hub_mod.LOCAL_BACKUPS and new in left
    assert set(left) - {new} == set(old[-(hub_mod.LOCAL_BACKUPS - 1):])
    assert sqlite3.connect(new).execute("SELECT COUNT(*) FROM nodes").fetchone() == (0,)


class _Bucket:
    def __init__(self):
        self.objects = {}

    def bucket(self, name):
        outer = self

        class B:
            def blob(self, key):
                class O:
                    def upload_from_string(self, data, content_type=None):
                        outer.objects[f"{name}/{key}"] = data
                return O()
        return B()


def test_offsite_backup_uploads_a_checked_copy(tmp_path):
    store = hub_mod.HubStore(tmp_path)
    store.set_setting("alerts", {"webhook": "https://example.invalid/x"})
    gcs = _Bucket()
    url = hub_mod.offsite_backup(store, "bkt", client=gcs)
    (key, data), = gcs.objects.items()
    assert url == f"gs://{key}" and key.startswith("bkt/hub-backups/hub-") and key.endswith(".db.gz")
    restored = tmp_path / "restored.db"
    restored.write_bytes(gzip.decompress(data))
    db = sqlite3.connect(restored)
    assert db.execute("SELECT value FROM settings WHERE key='alerts'").fetchone()
    assert not (tmp_path / "hub.db.offsite.tmp").exists()
    assert store.setting("last_offsite_backup")["url"] == url
    assert not list(tmp_path.glob("hub.db.bak-*"))          # local copies untouched
