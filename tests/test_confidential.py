"""Confidential mode: sealing on the provider side, unsealing in the sealed node.

Google Cloud (KMS, Storage, attestation) is replaced by in-memory fakes; the
sealed bundle, the node it configures and the scores it produces are real.
"""
from __future__ import annotations

import io
import json
import tarfile

import pytest
import yaml

from bmb_eval import cli, confidential as conf
from bmb_eval.client import Client
from bmb_eval.config import Config
from bmb_eval.contract import Limits
from bmb_eval.node import Worker
from bmb_eval.sandbox import BwrapSandbox, Mount

from test_node import EXAMPLE_BUNDLE, _split_task, hub, provider_files  # noqa: F401 (fixtures)

DIGEST = "sha256:" + "ab" * 32


def test_seal_round_trip_and_tamper():
    dek = bytes(32)
    blob = conf.encrypt(dek, b"secret rows", aad="task_a")
    assert conf.decrypt(dek, blob, aad="task_a") == b"secret rows"
    with pytest.raises(Exception):
        conf.decrypt(dek, blob, aad="task_b")          # bound to its task
    with pytest.raises(Exception):
        conf.decrypt(dek, blob[:-1] + bytes([blob[-1] ^ 1]), aad="task_a")


@pytest.mark.parametrize("name", ["../escape", "/abs", "other/x"])
def test_unpack_rejects_unsafe_members(tmp_path, name):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        ti = tarfile.TarInfo(name)
        ti.size = 1
        tar.addfile(ti, io.BytesIO(b"x"))
    with pytest.raises(ValueError):
        conf.unpack(buf.getvalue(), tmp_path / "t", tmp_path / "d")


def test_attestation_rule():
    rule = conf.attestation_condition(DIGEST, "wl@p.iam.gserviceaccount.com", gpu_preview=False)
    for part in ("CONFIDENTIAL_SPACE", "'STABLE'", "disabled-since-boot", DIGEST, "wl@p.iam"):
        assert part in rule
    assert "EXPERIMENTAL" not in rule
    gpu = conf.attestation_condition(DIGEST, "wl@p", gpu_preview=True)
    assert "'EXPERIMENTAL'" in gpu and "nvidia_gpu.cc_mode == 'ON'" in gpu


def test_bwrap_sandbox_command(tmp_path):
    (tmp_path / "bundle").mkdir()
    cmd = BwrapSandbox().cmd(entry="predict", mounts=[Mount("BMB_BUNDLE", tmp_path / "bundle", "/bundle")],
                             env={}, limits=Limits(), gpus=0)
    assert "--unshare-net" in cmd and "--clearenv" in cmd
    assert ["--ro-bind", str((tmp_path / "bundle").resolve()), "/bundle"] == cmd[cmd.index("/bundle") - 2:
                                                                               cmd.index("/bundle") + 1]
    assert not any("nvidia" in c for c in cmd)
    assert "/" not in [cmd[i + 1] for i, c in enumerate(cmd) if c == "--ro-bind"]   # never the whole root


def test_gpu_stage_memory_is_watched(tmp_path):
    from bmb_eval.sandbox import sandbox_anon_bytes
    for pid, uid, anon in (("1", 0, 900000), ("2", 65534, 300000), ("3", 65534, 400000)):
        (tmp_path / pid).mkdir()
        (tmp_path / pid / "status").write_text(f"Name:\tpython3\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n"
                                               f"RssAnon:\t{anon} kB\nRssFile:\t999999 kB\n")
    (tmp_path / "4").mkdir()
    (tmp_path / "4" / "status").write_text("Name:\tkthreadd\nUid:\t65534\n")   # kernel thread: no Rss
    assert sandbox_anon_bytes(tmp_path) == 700000 << 10


def test_sealed_node_end_to_end(hub, provider_files, tmp_path, monkeypatch):  # noqa: F811
    """Provider seals (fake cloud) -> sealed node unseals -> agent gets a score."""
    store, url = hub
    data, _ = provider_files
    store.set_setting("confidential", {"image": f"r/sealed@{DIGEST}", "digest": DIGEST,
                                       "workload_sa": "wl@p", "hardware": "test"})
    code, _ = store.add_enrollment(_split_task(tmp_path, name="sealed"))
    uploaded = {}
    monkeypatch.setattr(conf, "check_gcloud", lambda: None)
    monkeypatch.setattr(conf, "setup_provider_cloud", lambda project, digest, sa, gpu: {
        "PROVIDER_PROJECT_NUMBER": "1", "POOL": "bmb-eval", "PROVIDER_ID": "attestation",
        "KEY": "k", "BUCKET": "gs://b"})
    monkeypatch.setattr(conf, "upload_sealed",
                        lambda project, key, bucket, task, plaintext: uploaded.update({task: plaintext}))
    cli.main(["connect-confidential", f"{url}/c/{code}", "--data", str(data), "--project", "p",
              "--dir", str(tmp_path / "provider"), "--yes"])
    assert set(uploaded) == {"sealed"}
    with tarfile.open(fileobj=io.BytesIO(uploaded["sealed"])) as tar:
        names = tar.getnames()
    assert "node.json" in names and "task/score.py" in names
    assert "data/inputs_test/samples.parquet" in names and not any("split.json" in n for n in names)
    node_id = store.q("SELECT id FROM nodes", one=True)[0]
    assert json.loads(store.q("SELECT confidential FROM nodes WHERE id=?", (node_id,), one=True)[0])["BUCKET"] == "gs://b"

    # The sealed node: unseal into its memory folder and serve (local sandbox stands in for bwrap).
    root = tmp_path / "sealed_root"
    node = conf.unpack(uploaded["sealed"], root / "tasks" / "sealed", root / "private" / "sealed")
    cfg_path = conf.write_config(root, {"sealed": (root / "tasks" / "sealed", root / "private" / "sealed",
                                                   node)}, "test", tmp_path / "blobs")
    raw = yaml.safe_load(cfg_path.read_text())
    assert raw["sandbox"]["backend"] == "bwrap" and raw["hub"]["url"] == url
    raw["sandbox"] = {"backend": "local", "i_understand_local_backend_is_not_isolated": True}
    cfg_path.write_text(yaml.safe_dump(raw))
    cfg = Config.load(cfg_path)
    from bmb_eval import approval
    approval.approve(cfg)
    agent = Client(url, store.add_agent("agent-sealed"))
    sid = agent.submit_async("sealed", EXAMPLE_BUNDLE)["submission_id"]
    Worker(cfg).drain()
    res = agent.status(sid)["result"]
    assert res["status"] == "ok" and res["split"] == "validation"


def test_next_bare_code_works_after_a_task_stops_at_bad_data(hub, provider_files, tmp_path,  # noqa: F811
                                                            monkeypatch):
    """As in a multi-task email: the first command (with the hub) hits a data problem;
    the next task's command has just the code."""
    import pandas as pd
    store, url = hub
    data, _ = provider_files
    store.set_setting("confidential", {"image": f"r/sealed@{DIGEST}", "digest": DIGEST,
                                       "workload_sa": "wl@p", "hardware": "test"})
    first, _ = store.add_enrollment(_split_task(tmp_path, name="first"))
    second, _ = store.add_enrollment(_split_task(tmp_path, name="second"))
    monkeypatch.setattr(conf, "check_gcloud", lambda: None)
    pd.read_csv(data).drop(columns=["sequence"]).to_csv(tmp_path / "bad.csv", index=False)
    home = str(tmp_path / "provider")
    with pytest.raises(SystemExit, match="Fix it"):
        cli.main(["connect-confidential", first, "--hub", url, "--data", str(tmp_path / "bad.csv"),
                  "--project", "p", "--dir", home, "--yes"])
    uploaded = {}
    monkeypatch.setattr(conf, "setup_provider_cloud", lambda project, digest, sa, gpu: {
        "PROVIDER_PROJECT_NUMBER": "1", "POOL": "bmb-eval", "PROVIDER_ID": "attestation",
        "KEY": "k", "BUCKET": "gs://b"})
    monkeypatch.setattr(conf, "upload_sealed",
                        lambda project, key, bucket, task, plaintext: uploaded.update({task: plaintext}))
    cli.main(["connect-confidential", second, "--data", str(data), "--dir", home, "--yes"])
    assert set(uploaded) == {"second"}


def test_hub_confidential_endpoints(hub):  # noqa: F811
    import urllib.request
    store, url = hub
    with pytest.raises(Exception):
        urllib.request.urlopen(f"{url}/v1/confidential")          # not set up: 404
    store.set_setting("confidential", {"digest": DIGEST})
    assert json.loads(urllib.request.urlopen(f"{url}/v1/confidential").read())["digest"] == DIGEST


def test_sealed_node_status_reaches_the_hub(hub, provider_files, tmp_path):  # noqa: F811
    from bmb_eval.node import HubAPI
    store, url = hub
    data, task_dir = provider_files
    code, _ = store.add_enrollment(task_dir)
    creds = HubAPI(url).activate(code, "sealed-test")
    conf.STATUS.clear()
    conf.say(step="selftest", task="binding", failed=0)
    HubAPI(url, creds["secret"]).hello({}, False, [], conf.STATUS)
    st = json.loads(store.q("SELECT status FROM nodes WHERE id=?", (creds["node_id"],), one=True)[0])
    assert st["selftest"]["failed"] == 0 and "at" in st["selftest"]


def test_launcher_starts_only_offline_sealed_nodes_with_work(hub, provider_files):  # noqa: F811
    import time as _t
    from bmb_eval import sealed_launcher as L
    from bmb_eval.node import HubAPI
    store, url = hub
    data, task_dir = provider_files
    code, _ = store.add_enrollment(task_dir)
    creds = HubAPI(url).activate(code, "sealed-test")
    nid = creds["node_id"]
    ptr = {"PROVIDER_PROJECT_NUMBER": "1", "POOL": "p", "PROVIDER_ID": "a", "KEY": "k",
           "BUCKET": "gs://b", "digest": DIGEST}
    store.q("UPDATE nodes SET confidential=? WHERE id=?", (json.dumps(ptr), nid))
    assert L.due(store) == []                                     # no queued work
    task = store.q("SELECT task FROM node_tasks WHERE node_id=?", (nid,), one=True)[0]
    store.q("INSERT INTO submissions (id, task, agent, created, status, manifest) VALUES "
            "('s1', ?, 'a', ?, 'queued', '{}')", (task, _t.time()))
    store.q("UPDATE nodes SET last_seen=? WHERE id=?", (_t.time(), nid))
    assert L.due(store) == []                                     # node is online
    store.q("UPDATE nodes SET last_seen=? WHERE id=?", (_t.time() - 3600, nid))
    assert [n for n, _ in L.due(store)] == [nid]                  # offline + queued -> launch
    store.set_setting(f"sealed_launch:{nid}", {"at": _t.time()})
    assert L.due(store) == []                                     # cooldown after a launch
    # It refused work (self-test failed) on this image: not relaunched until a new image.
    store.set_setting(f"sealed_launch:{nid}", {"at": _t.time() - 3600, "digest": DIGEST})
    store.q("UPDATE nodes SET status=? WHERE id=?", (json.dumps({"refused": {"reason": "x"}}), nid))
    assert L.due(store, digest=DIGEST) == []
    assert [n for n, _ in L.due(store, digest="sha256:" + "cd" * 32)] == [nid]
    body = L.instance_body("bmb-sealed-x", "us-central1-a", "SPOT", "proj",
                           {"image": "img@sha256:1", "workload_sa": "wl@p", "gpu_preview": True}, ptr)
    meta = {i["key"]: i["value"] for i in body["metadata"]["items"]}
    assert meta["tee-env-BUCKET"] == "gs://b" and "tee-env-digest" not in meta
    assert "tee-container-log-redirect" not in meta                # production: no logs
    assert body["scheduling"]["provisioningModel"] == "SPOT"


def test_preempted_sealed_node_loses_nothing(hub, provider_files, tmp_path):  # noqa: F811
    """A job held by a node whose VM vanished goes back to the queue (hub), and the
    restarted node forgets it instead of failing it (no quota or final used)."""
    import time as _t
    from bmb_eval import sealed_launcher as L
    from bmb_eval.node import HubAPI
    from bmb_eval.store import Store
    store, url = hub
    data, task_dir = provider_files
    code, _ = store.add_enrollment(task_dir)
    creds = HubAPI(url).activate(code, "sealed-test")
    nid = creds["node_id"]
    store.q("UPDATE nodes SET confidential=?, last_seen=? WHERE id=?",
            (json.dumps({"BUCKET": "gs://b"}), _t.time() - 3600, nid))
    task = store.q("SELECT task FROM node_tasks WHERE node_id=?", (nid,), one=True)[0]
    store.q("INSERT INTO submissions (id, task, agent, created, status, node_id, manifest, split) VALUES "
            "('f1', ?, 'a', ?, 'assigned', ?, '{}', 'test')", (task, _t.time(), nid))
    assert L.reclaim(store) == 1
    assert store.q("SELECT status, node_id FROM submissions WHERE id='f1'", one=True) == ("queued", None)
    assert [n for n, _ in L.due(store)] == [nid]

    node_store = Store(tmp_path)                    # the node's store, restored from the bucket
    node_store.add("f1", task, "a", 0, "test")
    node_store.start("f1")                          # it was running when the VM died
    assert node_store.count_final(task) == 1
    from bmb_eval.node import Worker
    w = Worker.__new__(Worker)
    w.store, w.queue_dir, w.cfg = node_store, tmp_path / "queue", type("C", (), {"state_dir": tmp_path})()
    w.queue_dir.mkdir()
    assert w.drop_lost_jobs() == 1
    assert node_store.count_final(task) == 0 and not node_store.has("f1")


def test_flex_start_request_shape():
    from bmb_eval import sealed_launcher as L
    body = L.instance_body("n", "us-central1-a", "FLEX_START", "p",
                           {"image": "i@sha256:1", "workload_sa": "wl@p", "gpu_preview": True}, {})
    assert body["scheduling"]["provisioningModel"] == "FLEX_START"
    assert body["params"]["requestValidForDuration"]["seconds"] == str(L.FLEX_WAIT_S)
    assert "maxRunDuration" in body["scheduling"]
    assert all(m in ("FLEX_START", "SPOT") for _, m in L.PLACES)    # no on-demand for this machine


def test_bucket_with_two_nodes_serves_only_the_launched_one(tmp_path, monkeypatch):
    """A provider who connected tasks from two machines has two node identities in
    one bucket: the VM the hub launches for one node serves only that node's tasks
    (before, every VM refused to start)."""
    from bmb_eval import sealed_launcher as L
    tasks = {"a": (tmp_path, tmp_path, {"node_id": "node_1"}),
             "b": (tmp_path, tmp_path, {"node_id": "node_1"}),
             "old": (tmp_path, tmp_path, {"node_id": "node_0"})}
    assert sorted(conf.for_node(tasks, "node_1")) == ["a", "b"]
    assert sorted(conf.for_node(tasks, "node_0")) == ["old"]
    assert conf.for_node(tasks, "node_9") == {}
    with pytest.raises(ValueError):
        conf.for_node(tasks, None)                       # unnamed VM: still refuses to guess
    assert sorted(conf.for_node({k: tasks[k] for k in "ab"}, None)) == ["a", "b"]

    calls = []
    monkeypatch.setattr(L, "_call", lambda m, url, body=None: calls.append((m, body)) or
                        ({"error": {"code": 404}} if m == "GET" else {"status": "DONE"}))
    monkeypatch.setattr(L, "_wait", lambda op, timeout=300: op)
    L.launch("node_1", {"BUCKET": "gs://b", "digest": "sha256:1"},
             {"image": "i@sha256:1", "workload_sa": "wl@p"}, project="p")
    body = next(b for m, b in calls if m == "POST")
    meta = {i["key"]: i["value"] for i in body["metadata"]["items"]}
    assert meta["tee-env-BMB_NODE_ID"] == "node_1"


def test_sealed_state_is_per_node_and_keeps_legacy_counts(tmp_path):
    class Blob:
        def __init__(self, store, name): self.store, self.name = store, name
        def exists(self): return self.name in self.store
        def download_as_bytes(self): return self.store[self.name]
        def upload_from_string(self, b): self.store[self.name] = b

    class Bucket(dict):
        def blob(self, name): return Blob(self, name)

    class Kms:
        def encrypt(self, name, plaintext): return type("R", (), {"ciphertext": plaintext})()
        def decrypt(self, name, ciphertext): return type("R", (), {"plaintext": ciphertext})()

    class Store:
        def __init__(self, raw): self.lock, self.db = __import__("threading").Lock(), type("D", (), {"serialize": lambda s: raw})()

    bucket = Bucket()
    legacy = conf.SealedState(bucket, Kms(), "k")                  # a node started before this change
    legacy.save(Store(b"legacy counts"))
    one = conf.SealedState(bucket, Kms(), "k", "node_1")
    assert one.load(tmp_path / "a.db") and (tmp_path / "a.db").read_bytes() == b"legacy counts"
    one.save(Store(b"node_1 counts"))
    two = conf.SealedState(bucket, Kms(), "k", "node_2")
    two.save(Store(b"node_2 counts"))
    assert conf.SealedState(bucket, Kms(), "k", "node_1").load(tmp_path / "b.db")
    assert (tmp_path / "b.db").read_bytes() == b"node_1 counts"     # not overwritten by node_2
    assert sorted(k for k in bucket if "state-" in k) == ["bmb/state-node_1.enc", "bmb/state-node_2.enc"]


def test_selftest_skips_a_task_with_no_validation_rows(tmp_path, monkeypatch):
    """A published-test-set task can have an empty validation half. Its self-test used
    to crash (first row of an empty table), which stopped the whole sealed node."""
    import types
    import pandas as pd
    pd.DataFrame({"row_id": pd.Series([], dtype=str)}).to_parquet(tmp_path / "samples.parquet")
    field = types.SimpleNamespace(name="row_id")
    t = types.SimpleNamespace(inputs=tmp_path / "samples.parquet",
                              contract=types.SimpleNamespace(inputs=[field]))
    cfg = types.SimpleNamespace(tasks={"empty": t}, state_dir=tmp_path, backend="bwrap")
    monkeypatch.setattr("bmb_eval.runner.Runner", lambda *a, **k: pytest.fail("ran attacks"))
    assert cli._selftest(cfg, "empty", verbose=False) == 0
    assert '"skipped": true' in (tmp_path / "audit.jsonl").read_text()


def test_exited_sealed_vm_is_deleted_so_its_disk_frees_quota(hub, provider_files, monkeypatch):  # noqa: F811
    import time as _t
    from bmb_eval import sealed_launcher as L
    from bmb_eval.node import HubAPI
    store, url = hub
    _, task_dir = provider_files
    code, _ = store.add_enrollment(task_dir)
    nid = HubAPI(url).activate(code, "sealed-test")["node_id"]
    store.q("UPDATE nodes SET confidential=?, last_seen=? WHERE id=?",
            (json.dumps({"BUCKET": "gs://b"}), _t.time() - 3600, nid))
    store.set_setting(f"sealed_launch:{nid}", {"at": _t.time() - 3600, "zone": "z1"})
    status = {"value": "RUNNING"}
    calls = []
    monkeypatch.setattr(L, "_call", lambda m, u, b=None: calls.append(m) or
                        ({"status": status["value"]} if m == "GET" else {"status": "DONE"}))
    monkeypatch.setattr(L, "_wait", lambda op, timeout=300: op)
    assert L.remove_stopped(store, project="p") == 0 and "DELETE" not in calls   # still running
    status["value"] = "TERMINATED"
    assert L.remove_stopped(store, project="p") == 1 and calls[-1] == "DELETE"
    assert L.remove_stopped(store, project="p") == 0                              # done once


def test_hub_passes_through_only_uv_and_python_release_files(hub, monkeypatch):  # noqa: F811
    import io as _io
    import urllib.request
    store, url = hub
    fetched = []

    def fake_urlopen(u, timeout=60):
        fetched.append(u)
        return _io.BytesIO(b"release bytes")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)    # stands in for GitHub

    def get(path):
        import http.client
        from urllib.parse import urlsplit
        c = http.client.HTTPConnection(urlsplit(url).hostname, urlsplit(url).port)
        c.request("GET", path)
        r = c.getresponse()
        return r.status, r.read()
    assert get("/dl/uv/0.13.1/uv-x86_64-unknown-linux-musl.tar.gz") == (200, b"release bytes")
    assert fetched == ["https://github.com/astral-sh/uv/releases/download/0.13.1/"
                       "uv-x86_64-unknown-linux-musl.tar.gz"]
    assert get("/dl/uv/0.13.1/uv-x86_64-unknown-linux-musl.tar.gz")[0] == 200
    assert len(fetched) == 1                                   # served from the cache
    assert get("/dl/other/1/x")[0] in (401, 404)               # only uv and python
    assert get("/dl/uv/../../etc/passwd")[0] in (401, 404)
    assert get("/dl/python/../x")[0] in (401, 404)
