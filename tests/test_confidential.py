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
    body = L.instance_body("bmb-sealed-x", "us-central1-a", "SPOT", "proj",
                           {"image": "img@sha256:1", "workload_sa": "wl@p", "gpu_preview": True}, ptr)
    meta = {i["key"]: i["value"] for i in body["metadata"]["items"]}
    assert meta["tee-env-BUCKET"] == "gs://b" and "tee-env-digest" not in meta
    assert "tee-container-log-redirect" not in meta                # production: no logs
    assert body["scheduling"]["provisioningModel"] == "SPOT"
