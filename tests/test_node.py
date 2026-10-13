"""End-to-end and unit tests. Run on the local (unisolated) backend, so they
cover the pipeline, gates, hub and node logic; container isolation itself is
covered by `bmb-eval selftest` on a docker/podman node."""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import tarfile
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from bmb_eval import approval, audit, leakage, scaffold
from conftest import sequence_data
from bmb_eval.blobs import BlobCache, check_manifest, manifest_for
from bmb_eval.bundle import pack_dir, unpack
from bmb_eval.client import Client
from bmb_eval.config import Config
from bmb_eval.contract import Contract, gate_metrics
from bmb_eval.errors import EvalError
from bmb_eval.redteam import ATTACKS, bundle_bytes
from bmb_eval.runner import Runner
from bmb_eval.sandbox import Mount, container_cmd
from bmb_eval import cli, hub as hub_mod
from bmb_eval.node import HubAPI, HubError, Worker, paused_path

REPO = Path(__file__).resolve().parents[1]
TASK = "sequence_regression"
EXAMPLE_TASK = REPO / "examples" / "tasks" / TASK
EXAMPLE_BUNDLE = EXAMPLE_TASK / "bundle"


@pytest.fixture
def node_dir(tmp_path):
    shutil.copytree(EXAMPLE_TASK, tmp_path / "tasks" / TASK, ignore=shutil.ignore_patterns("bundle"))
    inp, ans = sequence_data(tmp_path / "private_data" / TASK)
    (tmp_path / "config.yaml").write_text(json.dumps({
        "node_name": "test",
        "sandbox": {"backend": "local", "i_understand_local_backend_is_not_isolated": True},
        "tasks": {TASK: {"dir": f"tasks/{TASK}", "inputs": str(inp), "answers": str(ans)}},
    }))
    return tmp_path


@pytest.fixture
def cfg(node_dir):
    return Config.load(node_dir / "config.yaml")


def attack(name):
    return next(a for a in ATTACKS if a.name == name)


def run_attack(cfg, name):
    a = attack(name)
    return Runner(cfg, send_alerts=False).evaluate(TASK, bundle_bytes(a), limits_override=a.limits)


# ------------------------------------------------------------------ pipeline

def test_example_bundle_scores(cfg):
    p = Runner(cfg).evaluate(TASK, pack_dir(EXAMPLE_BUNDLE))
    assert p["status"] == "ok", p
    assert set(p["metrics"]) == {"spearman", "pearson", "rmse"}
    assert p["metrics"]["spearman"] > 0.5


@pytest.mark.parametrize("name", ["baseline", "timeout", "flood_output", "leak_in_error",
                                  "parser_attack", "wrong_schema", "symlink_tar"])
def test_attacks_contained(cfg, name):
    p = run_attack(cfg, name)
    got = "ok" if p["status"] == "ok" else p["error"]["code"]
    assert got == attack(name).expect, p


def test_private_values_never_exported(cfg):
    import pandas as pd
    seq = pd.read_parquet(cfg.tasks[TASK].inputs)["sequence"].iloc[0]
    p = run_attack(cfg, "leak_in_error")
    assert seq not in json.dumps(p)
    # ...but the operator can still debug from the provider-side log.
    assert seq in (cfg.state_dir / "runs" / p["run_id"] / "predict.log").read_text()


@pytest.mark.parametrize("name", ["network", "write_host"])
def test_canaries_detect_escape_when_unisolated(cfg, name):
    """The local backend is not a sandbox, so these attacks must report success.
    That proves selftest would catch the same failure on a misconfigured container."""
    p = run_attack(cfg, name)
    assert p["status"] == "error"
    log = (cfg.state_dir / "runs" / p["run_id"] / "predict.log").read_text()
    assert "ESCAPED" in log


def test_bundle_deleted_after_run(cfg):
    p = Runner(cfg).evaluate(TASK, pack_dir(EXAMPLE_BUNDLE))
    assert not (cfg.state_dir / "runs" / p["run_id"] / "bundle").exists()


def test_validation_error_detail_is_gated(cfg):
    p = run_attack(cfg, "wrong_schema")
    assert p["error"]["detail"] == {"column": "fitness_pred", "reason": "wrong_type"}


# ---------------------------------------------------------------------- gate

def _contract():
    return Contract.load(EXAMPLE_TASK / "contract.yaml")


@pytest.mark.parametrize("raw", [
    {"spearman": 0.5, "pearson": 0.5},                                    # missing key
    {"spearman": 0.5, "pearson": 0.5, "rmse": 1, "extra": 1},             # extra key
    {"spearman": "0.5", "pearson": 0.5, "rmse": 1},                       # string
    {"spearman": 0.5, "pearson": 0.5, "rmse": 11},                        # out of bounds
    {"spearman": 0.5, "pearson": 0.5, "rmse": None},                      # null not allowed
    {"spearman": True, "pearson": 0.5, "rmse": 1},                        # bool
])
def test_gate_rejects(raw):
    with pytest.raises(EvalError) as e:
        gate_metrics(_contract(), json.dumps(raw).encode())
    assert e.value.code == "output_contract_violation"


def test_gate_rounds_and_allows_declared_null():
    out = gate_metrics(_contract(), b'{"spearman": null, "pearson": 0.123456, "rmse": 1.0004}')
    assert out == {"spearman": None, "pearson": 0.123, "rmse": 1.0}


def test_gate_rejects_nan():
    with pytest.raises(EvalError):
        gate_metrics(_contract(), b'{"spearman": NaN, "pearson": 0.1, "rmse": 1}')


def test_unknown_error_code_refused():
    with pytest.raises(ValueError):
        EvalError("anything_with_free_text")


# -------------------------------------------------------------------- bundle

def _tar(add):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        add(t)
    return buf.getvalue()


@pytest.mark.parametrize("name", ["../evil.py", "/abs.py"])
def test_unpack_rejects_bad_paths(tmp_path, name):
    def add(t):
        ti = tarfile.TarInfo(name)
        ti.size = 1
        t.addfile(ti, io.BytesIO(b"x"))
    with pytest.raises(EvalError) as e:
        unpack(_tar(add), tmp_path / "b", 1 << 20)
    assert e.value.code == "bundle_unsafe"


# ------------------------------------------------------------------ contract

def test_public_contract_hides_answers_and_round_trips():
    c = _contract()
    pub = c.public()
    assert "answers" not in pub and "fitness" not in json.dumps(pub["inputs"])
    c2 = Contract.from_dict(json.loads(json.dumps(pub)), public=True)
    assert c2.public() == pub


def test_synth_inputs_match_contract():
    from bmb_eval.contract import check_frame
    c = _contract()
    df = scaffold.synth_inputs(c, n=50)
    assert check_frame(c.inputs, df, strict_columns=True, key=c.key) is None


# ------------------------------------------------------------------- leakage

def test_leakage_budget():
    b = leakage.budget(_contract())
    assert 30 < b.metric_bits < 40
    assert b.timing_bits > 0 and b.lifetime == pytest.approx(b.per_submission * 200)


def test_release_delay_closes_timing_channel():
    import dataclasses
    c = _contract()
    c = dataclasses.replace(c, limits=dataclasses.replace(c.limits, release_after_s=10_000))
    assert leakage.budget(c).timing_bits == 0


def test_unbounded_metric_is_infinite():
    import dataclasses
    c = _contract()
    c.metrics[2] = dataclasses.replace(c.metrics[2], max=None)
    assert leakage.budget(c).per_submission == float("inf")


# --------------------------------------------------------------- docker cmd

def test_container_cmd_hardening():
    c = _contract()
    cmd = container_cmd("docker", "img", "runsc", name="n", entry="predict",
                        mounts=[Mount("BMB_INPUTS", Path("/tmp/x.parquet"), "/data/inputs.parquet")],
                        env={}, limits=c.limits)
    s = " ".join(cmd)
    for flag in ["--network none", "--read-only", "--cap-drop ALL", "--log-driver none",
                 "no-new-privileges", "--runtime runsc", "--memory-swap"]:
        assert flag in s
    vols = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-v"]
    assert vols == ["/private/tmp/x.parquet:/data/inputs.parquet:ro"] or \
        vols == ["/tmp/x.parquet:/data/inputs.parquet:ro"]
    pod = container_cmd("podman", "img", None, name="n", entry="predict", mounts=[], env={},
                        limits=c.limits)
    assert "keep-id" in pod


# ------------------------------------------------------- approval and audit

def test_approval_detects_scorer_change(cfg):
    approval.approve(cfg)
    assert approval.diff(cfg) == []
    score = cfg.tasks[TASK].dir / "score.py"
    score.write_text(score.read_text() + "\n# tampered\n")
    assert any("task_dir" in d for d in approval.diff(cfg))


def test_audit_chain_detects_edit(cfg):
    for i in range(3):
        audit.record(cfg.state_dir, "e", i=i)
    assert audit.verify(cfg.state_dir)[0]
    p = cfg.state_dir / "audit.jsonl"
    p.write_text(p.read_text().replace('"i": 1', '"i": 9'))
    assert not audit.verify(cfg.state_dir)[0]


# ------------------------------------------------------------------- ladder

def test_ladder_hides_non_improvements(node_dir):
    path = node_dir / "tasks" / TASK / "contract.yaml"
    path.write_text(path.read_text().replace("export:\n", "export:\n  ladder_step: 0.05\n"))
    cfg = Config.load(node_dir / "config.yaml")
    from bmb_eval.store import Store
    r = Runner(cfg, store=Store(cfg.state_dir))
    first = r.evaluate(TASK, pack_dir(EXAMPLE_BUNDLE), submitter="a")
    again = r.evaluate(TASK, pack_dir(EXAMPLE_BUNDLE), submitter="a")
    assert first["improved"] is True
    assert again["improved"] is False and again["metrics"] == first["metrics"]


# ------------------------------------------------------------- hub + node

@pytest.fixture
def hub(tmp_path):
    store = hub_mod.HubStore(tmp_path / "hub_state")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), None)
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    blobs = hub_mod.DiskBlobs(tmp_path / "hub_state" / "blobs", url)
    httpd.RequestHandlerClass = hub_mod.make_handler(store, blobs, url, REPO, 64)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield store, url
    httpd.shutdown()


@pytest.fixture
def provider_files(tmp_path):
    """The provider's one CSV (with an extra column we never asked for), and the
    task we wrote for them without seeing the data."""
    import pandas as pd
    inp, ans = sequence_data(tmp_path / "raw")
    df = pd.read_parquet(inp).merge(pd.read_parquet(ans), on="variant_id")
    df["lab_notes"] = "do not share"
    df.to_csv(tmp_path / "my_data.csv", index=False)
    task_dir = scaffold.write_task(
        "binding", tmp_path / "authored",
        [("variant_id", "string", False), ("sequence", "string", False)],
        [("variant_id", "string", False), ("fitness", "float", False)],
        ["variant_id"], "regression", "fitness")
    return tmp_path / "my_data.csv", task_dir


def _connect(node_dir, url, code, data):
    cli.main(["--config", str(node_dir / "config.yaml"), "connect", f"{url}/c/{code}",
              "--data", str(data), "--yes", "--no-service"])
    return Config.load(node_dir / "config.yaml")


@pytest.fixture
def connected(hub, node_dir, provider_files):
    store, url = hub
    data, task_dir = provider_files
    code, _ = store.add_enrollment(task_dir)
    cfg = _connect(node_dir, url, code, data)
    agent = Client(url, store.add_agent("agent-1"))
    return cfg, store, url, agent


def test_provider_connects_and_agent_gets_scores(connected):
    cfg, store, url, agent = connected
    assert cfg.hub["url"] == url and cfg.hub["secret"].startswith("bmbn_")
    assert (cfg.path.stat().st_mode & 0o777) == 0o600       # secret not world-readable
    assert not approval.diff(cfg)                            # connect approved it
    w = Worker(cfg)
    w.step()                                                 # hello: publishes the contract
    assert [t["contract"]["id"] for t in agent.tasks()["tasks"]] == ["binding"]
    sid = agent.submit_async("binding", EXAMPLE_BUNDLE)["submission_id"]
    assert agent.status(sid)["status"] == "pending"
    w.drain()
    res = agent.status(sid)
    assert res["status"] == "done" and res["result"]["metrics"]["spearman"] > 0.5
    # The hub holds only the model's files (by hash), nothing from the provider.
    held = {p.name for p in (store.state / "blobs").iterdir() if not p.name.startswith(".")}
    assert held == {f["sha256"] for f in manifest_for(EXAMPLE_BUNDLE)["files"]}


def test_code_reusable_by_its_own_machine_only(hub, connected, provider_files, tmp_path):
    cfg, store, url, _ = connected
    code = store.q("SELECT code FROM enrollments", one=True)[0]
    _connect(cfg.path.parent, url, code, provider_files[0])      # same machine: fine (re-run)
    other = _fresh_node(tmp_path / "someone_else")
    with pytest.raises(SystemExit, match="used by another machine"):
        _connect(other, url, code, provider_files[0])


def test_connect_explains_bad_file_and_keeps_code(hub, node_dir, provider_files, tmp_path, capsys):
    import pandas as pd
    store, url = hub
    data, task_dir = provider_files
    pd.read_csv(data).rename(columns={"sequence": "seq"}).to_csv(tmp_path / "bad.csv", index=False)
    code, _ = store.add_enrollment(task_dir)
    with pytest.raises(SystemExit):
        _connect(node_dir, url, code, tmp_path / "bad.csv")
    assert "It has no column 'sequence'" in capsys.readouterr().out
    assert HubAPI(url).enrollment(code)["task"] == "binding"   # code still usable
    _connect(node_dir, url, code, data)                         # ...and works once fixed


def test_one_file_is_split_and_extra_columns_dropped(connected):
    import pandas as pd
    cfg, *_ = connected
    t = cfg.tasks["binding"]
    assert list(pd.read_parquet(t.inputs).columns) == ["variant_id", "sequence"]
    assert list(pd.read_parquet(t.answers).columns) == ["variant_id", "fitness"]


def test_connect_with_bare_code_and_relative_path(hub, node_dir, provider_files, monkeypatch):
    store, url = hub
    data, task_dir = provider_files
    code, _ = store.add_enrollment(task_dir)
    monkeypatch.setenv("BMB_CALLER_CWD", str(data.parent))     # what install.sh sets
    cli.main(["--config", str(node_dir / "config.yaml"), "connect", code.lower(),
              "--hub", url, "--data", data.name, "--yes", "--no-service"])
    assert Config.load(node_dir / "config.yaml").hub["url"] == url


def test_second_task_attaches_to_same_node(hub, connected, provider_files, tmp_path):
    cfg, store, url, _ = connected
    data, _ = provider_files
    d = scaffold.write_task("binding2", tmp_path / "authored2",
                            [("variant_id", "string", False), ("sequence", "string", False)],
                            [("variant_id", "string", False), ("fitness", "float", False)],
                            ["variant_id"], "regression", "fitness")
    code, _ = store.add_enrollment(d)
    cfg2 = _connect(cfg.path.parent, url, code, data)
    assert cfg2.hub["node_id"] == cfg.hub["node_id"] and cfg2.hub["secret"] == cfg.hub["secret"]
    assert store.q("SELECT COUNT(*) FROM nodes", one=True)[0] == 1


def test_node_takes_no_work_when_stopped_or_changed(connected):
    cfg, store, url, agent = connected
    w = Worker(cfg)
    sid = agent.submit_async("binding", EXAMPLE_BUNDLE)["submission_id"]
    paused_path(cfg).write_text("x")
    w.step()
    assert store.q("SELECT status FROM submissions WHERE id=?", (sid,), one=True)[0] == "queued"
    paused_path(cfg).unlink()
    score = cfg.tasks["binding"].dir / "score.py"
    score.write_text(score.read_text() + "\n# edited\n")
    w.step()
    assert store.q("SELECT status FROM submissions WHERE id=?", (sid,), one=True)[0] == "queued"


def test_quota_enforced_by_node(connected):
    cfg, store, url, agent = connected
    cfg.tasks["binding"].contract.limits.submissions_per_day = 1
    w = Worker(cfg)
    a = agent.submit_async("binding", EXAMPLE_BUNDLE)["submission_id"]
    b = agent.submit_async("binding", EXAMPLE_BUNDLE)["submission_id"]
    w.drain()
    assert agent.status(a)["result"]["status"] == "ok"
    assert agent.status(b)["result"]["error"]["code"] == "quota_exceeded"


def test_result_held_on_node_until_release(connected):
    cfg, store, url, agent = connected
    cfg.tasks["binding"].contract.limits.release_after_s = 4
    w = Worker(cfg)
    sid = agent.submit_async("binding", EXAMPLE_BUNDLE)["submission_id"]
    w.step()                                     # runs it (well under 4s)
    w.step()
    assert agent.status(sid)["status"] == "pending"
    time.sleep(4)
    w.step()
    assert agent.status(sid)["status"] == "done"


def test_hub_requeues_job_a_node_lost(connected):
    cfg, store, url, agent = connected
    sid = agent.submit_async("binding", EXAMPLE_BUNDLE)["submission_id"]
    api = HubAPI(url, cfg.hub["secret"])
    assert api.next()["submission_id"] == sid    # claimed, then the node "crashes"
    api.hello({}, True, held=[])
    assert store.q("SELECT status FROM submissions WHERE id=?", (sid,), one=True)[0] == "queued"


def test_hub_auth(connected):
    cfg, store, url, agent = connected
    assert Client(url, "bmba_wrong").tasks()["error"]["code"] == "unauthorized"
    with pytest.raises(HubError):
        HubAPI(url, "bmbn_wrong").next()
    with pytest.raises(HubError):                # agents can't act as nodes
        HubAPI(url, agent.token).next()


def test_node_cannot_publish_contract_for_task_it_doesnt_own(connected):
    cfg, store, url, agent = connected
    HubAPI(url, cfg.hub["secret"]).hello({"someone_else": {"id": "x"}}, True, [])
    assert store.q("SELECT 1 FROM node_tasks WHERE task='someone_else'", one=True) is None


def test_install_script_and_code_tarball(hub):
    import urllib.request
    store, url = hub
    sh = urllib.request.urlopen(f"{url}/install.sh").read().decode()
    assert f'HUB="${{BMB_HUB:-{url}}}"' in sh
    tar = tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(f"{url}/bmb-eval.tar.gz").read()))
    names = tar.getnames()
    assert "setup.sh" in names and "src/bmb_eval/node.py" in names
    assert not any(n.startswith(("config.yaml", "state", "private_data", "hub_state", ".venv"))
                   or "__pycache__" in n for n in names)


def test_hub_new_task_and_provider_message(tmp_path, capsys):
    hub_mod.main(["--state", str(tmp_path / "h"), "new-task", "kd", "--tasks-dir", str(tmp_path / "t"),
                  "--inputs", "id:string,seq:string", "--target", "kd:float", "--key", "id"])
    hub_mod.main(["--state", str(tmp_path / "h"), "enroll", str(tmp_path / "t" / "kd"),
                  "--public-url", "https://hub.example.org"])
    msg = capsys.readouterr().out
    code = re.search(r"sh -s -- ([A-Z0-9-]{14}) --data", msg).group(1)
    assert f"curl -fsSL https://hub.example.org/install.sh | sh -s -- {code} --data" in msg
    email = msg[msg.index("Subject:"):]
    assert "/guide" in email and "kd" not in email    # same email for every provider


# --------------------------------------------------------- big-model transfer

def _big_bundle(tmp_path, name="m", head="0"):
    d = tmp_path / name
    (d / "weights").mkdir(parents=True)
    (d / "predict.py").write_text((EXAMPLE_BUNDLE / "predict.py").read_text())
    (d / "weights" / "model.json").write_text((EXAMPLE_BUNDLE / "weights" / "model.json").read_text())
    (d / "weights" / "base.bin").write_bytes(b"\x07" * (3 << 20))   # stands in for a 20 GB base
    (d / "weights" / "head.bin").write_bytes(head.encode() * 1024)
    return d


def test_resubmission_uploads_and_downloads_only_changes(connected, tmp_path, monkeypatch):
    cfg, store, url, agent = connected
    w = Worker(cfg)
    puts, gets = [], []
    real_put, real_dl = agent._put, w.api.download
    monkeypatch.setattr(agent, "_put", lambda up, p: (puts.append(p.name), real_put(up, p)))
    monkeypatch.setattr(w.api, "download", lambda info, c, sha, size: (gets.append(size), real_dl(info, c, sha, size)))
    a = agent.submit_async("binding", _big_bundle(tmp_path, "v1", "a"))["submission_id"]
    w.drain()
    assert sorted(puts) == ["base.bin", "head.bin", "model.json", "predict.py"]
    puts.clear(); gets.clear()
    b = agent.submit_async("binding", _big_bundle(tmp_path, "v2", "b"))["submission_id"]
    w.drain()
    assert puts == ["head.bin"] and gets == [1024]           # the 3 MB "base" never moved again
    assert agent.status(a)["result"]["status"] == agent.status(b)["result"]["status"] == "ok"


def test_corrupted_model_file_is_rejected(connected, tmp_path):
    cfg, store, url, agent = connected
    sid = agent.submit_async("binding", _big_bundle(tmp_path))["submission_id"]
    # Tamper with a stored file after upload (e.g. a storage fault).
    sha = manifest_for(_big_bundle(tmp_path, "again"))["files"][-1]["sha256"]
    p = store.state / "blobs" / sha
    p.chmod(0o644)
    p.write_bytes(b"tampered" + p.read_bytes()[8:])
    Worker(cfg).drain()
    assert agent.status(sid)["result"]["error"]["code"] == "bundle_invalid"


def test_commit_requires_every_file(connected, tmp_path):
    cfg, store, url, agent = connected
    d = _big_bundle(tmp_path)
    r = agent._req("POST", "/v1/tasks/binding/submissions", {"manifest": manifest_for(d)})
    res = agent._req("POST", f"/v1/submissions/{r['submission_id']}/commit")
    assert res["error"]["code"] == "files_missing" and len(res["error"]["missing"]) == 4


@pytest.mark.parametrize("path", ["../x.py", "/etc/passwd", "a/../../b"])
def test_manifest_paths_cannot_escape(path):
    m = {"files": [{"path": "predict.py", "sha256": "0" * 64, "size": 1},
                   {"path": path, "sha256": "1" * 64, "size": 1}]}
    with pytest.raises(EvalError) as e:
        check_manifest(m, 1 << 30)
    assert e.value.code == "bundle_unsafe"


def test_manifest_size_limit():
    m = {"files": [{"path": "predict.py", "sha256": "0" * 64, "size": 2 << 30}]}
    with pytest.raises(EvalError) as e:
        check_manifest(m, 1 << 30)
    assert e.value.code == "bundle_too_large"


def test_cache_evicts_least_recently_used(tmp_path):
    import hashlib
    c = BlobCache(tmp_path)
    shas = []
    for i in range(3):
        data = bytes([i]) * 1000
        sha = hashlib.sha256(data).hexdigest()
        c.put(sha, 1000, io.BytesIO(data).read)
        os.utime(c.path(sha), (i, i))
        shas.append(sha)
    c.evict(2000)
    assert not c.path(shas[0]).exists() and c.path(shas[2]).exists()


def test_gcs_signed_urls_offline():
    """The Cloud Storage path signs per-agent PUT/GET links (checked offline
    with a throwaway service-account key; no network)."""
    pytest.importorskip("google.cloud.storage")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from google.oauth2 import service_account
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048).private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption()).decode()
    creds = service_account.Credentials.from_service_account_info({
        "type": "service_account", "client_email": "hub@p.iam.gserviceaccount.com",
        "private_key": key, "token_uri": "https://oauth2.googleapis.com/token", "project_id": "p"})
    g = hub_mod.GCSBlobs("p-bmb-models", credentials=creds, project="p")
    sha = "a" * 64
    up, down = g.upload("agent-1", sha, 10), g.download("agent-1", sha, 10)
    assert up["method"] == "PUT" and up["auth"] is False and not down["auth"]
    assert "p-bmb-models/blobs/" in up["url"] and sha in up["url"] and "X-Goog-Signature" in up["url"]
    assert g._name("agent-1", sha) != g._name("agent-2", sha)   # per-agent storage


def test_guide_page(hub):
    import urllib.error
    import urllib.request
    store, url = hub
    r = urllib.request.urlopen(url + "/guide")
    page = r.read().decode()
    assert r.headers["Content-Type"].startswith("text/html")
    assert "<h1>Connecting your data to BioModelBench</h1>" in page and "<h2>Technical details</h2>" in page
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(url + "/security")
    guide = urllib.request.urlopen(url + "/guide").read().decode()
    assert f"curl -fsSL {url}/install.sh" in guide and "&lt;hub address&gt;" not in guide
    head = urllib.request.urlopen(urllib.request.Request(url + "/install.sh", method="HEAD"))
    assert head.status == 200 and int(head.headers["Content-Length"]) > 100 and head.read() == b""


def test_result_in_flight_survives_hub_restart(connected, monkeypatch):
    """A finished result the hub hasn't received yet is still 'held', so the hub
    doesn't re-queue it; and if a job was re-queued anyway, the owning node's
    result is still accepted and nothing blocks new work."""
    cfg, store, url, agent = connected
    w = Worker(cfg)
    sid = agent.submit_async("binding", EXAMPLE_BUNDLE)["submission_id"]
    real = w.api.result
    monkeypatch.setattr(w.api, "result", lambda *a: (_ for _ in ()).throw(HubError("cannot reach hub")))
    with pytest.raises(HubError):
        w.drain()                                  # ran the job; delivery failed
    assert sid in w.store.unposted()
    w._last_hello = 0
    monkeypatch.setattr(w.api, "result", lambda *a: None)
    w.step()                                       # hello reports it as held
    assert store.q("SELECT status FROM submissions WHERE id=?", (sid,), one=True)[0] == "assigned"
    store.q("UPDATE submissions SET status='queued', node_id=NULL WHERE id=?", (sid,))   # force a re-queue
    monkeypatch.setattr(w.api, "result", real)
    w.store.db.execute("UPDATE submissions SET posted=0 WHERE id=?", (sid,))
    w.store.db.commit()
    w.drain()
    assert agent.status(sid)["status"] == "done"
    sid2 = agent.submit_async("binding", EXAMPLE_BUNDLE)["submission_id"]
    w.drain()
    assert agent.status(sid2)["result"]["status"] == "ok"   # new work still flows


# ------------------------------------------------- held-out test set (split)

SPLIT_SCORER = """
import numpy as np
from scipy import stats

def score(df):
    y, p = df["fitness"].to_numpy(float), df["fitness_pred"].to_numpy(float)
    rho = float(stats.spearmanr(y, p)[0]) if np.std(p) > 0 and np.std(y) > 0 else None
    return {"spearman": rho, "n": int(len(df))}
"""


def _split_task(tmp_path, final_evaluations=2, name="split_task"):
    d = scaffold.write_task(
        name, tmp_path / "authored_split",
        [("variant_id", "string", False), ("sequence", "string", False)],
        [("variant_id", "string", False), ("fitness", "float", False)],
        ["variant_id"], "regression", "fitness")
    text = (d / "contract.yaml").read_text()
    start = text.index("metrics:")
    end = text.index("export:")
    text = (text[:start] + "metrics:\n"
            "  - {name: spearman, type: float, nullable: true, min: -1, max: 1}\n"
            "  - {name: n, type: int, min: 0, max: 100000}\n"
            "headline: spearman\n\n"
            f"split: {{validation_fraction: 0.5, final_evaluations: {final_evaluations}}}\n\n"
            + text[end:])
    (d / "contract.yaml").write_text(text)
    (d / "score.py").write_text(SPLIT_SCORER)
    return d


@pytest.fixture
def split_connected(hub, node_dir, provider_files, tmp_path):
    store, url = hub
    data, _ = provider_files
    code, _ = store.add_enrollment(_split_task(tmp_path))
    cfg = _connect(node_dir, url, code, data)
    return cfg, store, url, Client(url, store.add_agent("agent-split"))


def test_split_is_random_disjoint_and_hidden(split_connected):
    import pandas as pd
    cfg, *_ = split_connected
    t = cfg.tasks["split_task"]
    val = set(pd.read_parquet(t.inputs)["variant_id"])
    test = set(pd.read_parquet(t.test_inputs)["variant_id"])
    assert val and test and not (val & test)                     # disjoint
    assert len(val | test) == 500 and 150 < len(val) < 350       # all rows, roughly half
    assert not (t.inputs_dir.parent / "inputs").exists()        # no combined folder to mount
    assert (t.inputs_dir.parent / "split.json").stat().st_mode & 0o077 == 0


def test_agent_only_ever_scored_on_validation(split_connected):
    import pandas as pd
    cfg, store, url, agent = split_connected
    n_val = len(pd.read_parquet(cfg.tasks["split_task"].inputs))
    sid = agent.submit_async("split_task", EXAMPLE_BUNDLE)["submission_id"]
    Worker(cfg).drain()
    r = agent.status(sid)["result"]
    assert r["split"] == "validation" and r["metrics"]["n"] == n_val


def test_final_runs_on_test_set_and_is_hidden_from_agent(split_connected):
    import pandas as pd
    cfg, store, url, agent = split_connected
    n_test = len(pd.read_parquet(cfg.tasks["split_task"].test_inputs))
    sid = agent.submit_async("split_task", EXAMPLE_BUNDLE)["submission_id"]
    w = Worker(cfg)
    w.drain()
    fid = store.request_final(sid)
    w.drain()
    final = json.loads(store.q("SELECT result FROM submissions WHERE id=?", (fid,), one=True)[0])
    assert final["split"] == "test" and final["metrics"]["n"] == n_test
    assert agent.status(fid)["error"]["code"] == "not_found"    # agents never see final scores
    # Final runs don't use the agent's validation quota.
    assert w.store.count_total("split_task") == 1


def test_node_enforces_final_cap_and_refusals_dont_count(hub, node_dir, provider_files, tmp_path):
    store, url = hub
    data, _ = provider_files
    code, _ = store.add_enrollment(_split_task(tmp_path, final_evaluations=1, name="capped"))
    cfg = _connect(node_dir, url, code, data)
    agent = Client(url, store.add_agent("agent-cap"))
    sid = agent.submit_async("capped", EXAMPLE_BUNDLE)["submission_id"]
    w = Worker(cfg)
    w.drain()
    first, second, third = (store.request_final(sid) for _ in range(3))
    w.drain()
    res = {f: json.loads(store.q("SELECT result FROM submissions WHERE id=?", (f,), one=True)[0])
           for f in (first, second, third)}
    assert res[first]["status"] == "ok"
    assert res[second]["error"]["code"] == res[third]["error"]["code"] == "final_quota_exceeded"
    assert w.store.count_final("capped") == 1


def test_final_on_task_without_split_is_refused(connected):
    cfg, store, url, agent = connected
    sid = agent.submit_async("binding", EXAMPLE_BUNDLE)["submission_id"]
    w = Worker(cfg)
    w.drain()
    fid = store.request_final(sid)
    w.drain()
    r = json.loads(store.q("SELECT result FROM submissions WHERE id=?", (fid,), one=True)[0])
    assert r["error"]["code"] == "no_test_split"


def test_resplit_keeps_existing_assignment(tmp_path, provider_files):
    from bmb_eval.prepare import apply_split, prepare
    import pandas as pd
    data, _ = provider_files
    c = Contract.load(_split_task(tmp_path) / "contract.yaml")
    dest = tmp_path / "pd"
    prepare(c, data, dest)
    v1, t1, *_ = apply_split(c, dest)
    before = set(pd.read_parquet(t1 / "samples.parquet")["variant_id"])
    prepare(c, data, dest)                                       # provider re-runs connect
    v2, t2, *_ = apply_split(c, dest)
    assert set(pd.read_parquet(t2 / "samples.parquet")["variant_id"]) == before


def test_existing_databases_gain_split_column(tmp_path):
    import sqlite3
    from bmb_eval.store import Store
    db = sqlite3.connect(tmp_path / "node.db")
    db.execute("CREATE TABLE submissions (id TEXT PRIMARY KEY, task TEXT NOT NULL, submitter TEXT NOT NULL, "
               "day TEXT NOT NULL, status TEXT NOT NULL, created REAL NOT NULL, release_at REAL NOT NULL, "
               "finished REAL, result TEXT, posted INTEGER NOT NULL DEFAULT 0)")
    db.execute("INSERT INTO submissions VALUES ('old', 't', 'a', '2026-01-01', 'done', 0, 0, 0, '{}', 1)")
    db.commit()
    db.close()
    s = Store(tmp_path)
    assert s.count_total("t") == 1                               # old rows count as validation
    h = tmp_path / "hub"
    h.mkdir()
    hdb = sqlite3.connect(h / "hub.db")
    hdb.execute("CREATE TABLE submissions (id TEXT PRIMARY KEY, task TEXT NOT NULL, agent TEXT NOT NULL, "
                "created REAL NOT NULL, status TEXT NOT NULL, node_id TEXT, result TEXT, manifest TEXT NOT NULL)")
    hdb.commit()
    hdb.close()
    assert "split" in {r[1] for r in hub_mod.HubStore(h).db.execute("PRAGMA table_info(submissions)")}


def test_group_split_keeps_each_molecule_in_one_half(tmp_path):
    from bmb_eval.prepare import apply_split, prepare
    import pandas as pd
    d = scaffold.write_task("grp", tmp_path / "t",
                            [("row_id", "string", False), ("smiles", "string", False), ("endpoint", "string", False)],
                            [("row_id", "string", False), ("y", "float", False)], ["row_id"], "regression", "y")
    text = (d / "contract.yaml").read_text().replace("export:", "split: {validation_fraction: 0.5, group_by: smiles}\n\nexport:")
    (d / "contract.yaml").write_text(text)
    c = Contract.load(d / "contract.yaml")
    rows = [{"row_id": f"r{i}", "smiles": f"C{'C' * (i % 40)}O", "endpoint": f"e{i % 3}", "y": float(i)}
            for i in range(240)]                     # 40 molecules x 3 endpoints x 2
    pd.DataFrame(rows).to_csv(tmp_path / "d.csv", index=False)
    prepare(c, tmp_path / "d.csv", tmp_path / "out")
    v, t, nv, nt = apply_split(c, tmp_path / "out")
    vs = set(pd.read_parquet(v / "samples.parquet")["smiles"])
    ts = set(pd.read_parquet(t / "samples.parquet")["smiles"])
    assert vs and ts and not (vs & ts) and nv + nt == 240


def test_revoked_code_cannot_connect(tmp_path, capsys):
    store = hub_mod.HubStore(tmp_path / "h")
    d = scaffold.write_task("t1", tmp_path / "t", [("id", "string", False), ("x", "float", False)],
                            [("id", "string", False), ("y", "float", False)], ["id"], "regression", "y")
    code, _ = store.add_enrollment(d)
    hub_mod.main(["--state", str(tmp_path / "h"), "revoke", code])
    assert "revoked" in capsys.readouterr().out
    with pytest.raises(KeyError):
        store.activate(code, None, "n")


# ---------------------------------------- families: one attempt, many datasets

def _fresh_node(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "config.yaml").write_text(json.dumps({
        "node_name": root.name,
        "sandbox": {"backend": "local", "i_understand_local_backend_is_not_isolated": True}}))
    return root


def _provider_csv(path: Path, seed: int, n: int = 300) -> Path:
    import pandas as pd
    inp, ans = sequence_data(path.parent / f"raw{seed}", n=n, seed=seed)
    pd.read_parquet(inp).merge(pd.read_parquet(ans), on="variant_id").to_csv(path, index=False)
    return path


@pytest.fixture
def family(hub, tmp_path):
    """Two providers on two machines, each with its own dataset and split."""
    store, url = hub
    workers = {}
    for name, seed in (("lab_a", 1), ("lab_b", 2)):
        d = _split_task(tmp_path / name, final_evaluations=2, name=f"fit_{name}")
        code, _ = store.add_enrollment(d)
        node = _fresh_node(tmp_path / f"node_{name}")
        cfg = _connect(node, url, code, _provider_csv(tmp_path / f"{name}.csv", seed))
        workers[f"fit_{name}"] = Worker(cfg)
    store.set_family("fitness", ["fit_lab_a", "fit_lab_b"])
    agent = Client(url, store.add_agent("attempt-1", evaluations=3))
    return store, url, agent, workers


def _drain(workers):
    for w in workers.values():
        w.drain()


def test_family_submission_scored_on_every_dataset(family, monkeypatch):
    store, url, agent, workers = family
    listing = agent.tasks()
    assert listing["families"] == [{"name": "fitness", "tasks": ["fit_lab_a", "fit_lab_b"]}]
    assert listing["budget"] == {"evaluations": 3, "used": 0}
    puts = []
    real = agent._put
    monkeypatch.setattr(agent, "_put", lambda up, p: (puts.append(p.name), real(up, p)))
    r = agent.submit_async("fitness", EXAMPLE_BUNDLE)
    assert r["status"] == "pending" and len(puts) == 2       # files upload once for both datasets
    assert agent.status(r["submission_id"])["status"] == "pending"
    _drain(workers)
    g = agent.status(r["submission_id"])
    res = g["result"]
    assert g["status"] == "done" and set(res["datasets"]) == {"fit_lab_a", "fit_lab_b"}
    assert all(d["split"] == "validation" for d in res["datasets"].values())
    scores = [d["metrics"]["spearman"] for d in res["datasets"].values()]
    assert res["datasets_scored"] == 2 and res["combined"] == round(sum(scores) / 2, 4)
    assert agent.tasks()["budget"]["used"] == 1               # one evaluation, not two


def test_attempt_budget_is_shared_across_datasets_and_tasks(family):
    store, url, agent, workers = family
    agent.submit_async("fitness", EXAMPLE_BUNDLE)             # 1
    agent.submit_async("fitness", EXAMPLE_BUNDLE)             # 2
    agent.submit_async("fit_lab_a", EXAMPLE_BUNDLE)           # 3 (a single dataset counts too)
    r = agent.submit_async("fitness", EXAMPLE_BUNDLE)
    assert r["error"]["code"] == "budget_exhausted"
    assert agent.tasks()["budget"] == {"evaluations": 3, "used": 3}


def test_family_final_runs_every_test_set_hidden_from_agent(family):
    store, url, agent, workers = family
    gid = agent.submit_async("fitness", EXAMPLE_BUNDLE)["submission_id"]
    _drain(workers)
    finals = store.request_finals(gid)
    assert len(finals) == 2
    _drain(workers)
    for fid in finals:
        r = json.loads(store.q("SELECT result FROM submissions WHERE id=?", (fid,), one=True)[0])
        assert r["split"] == "test" and r["status"] == "ok"
        assert agent.status(fid)["error"]["code"] == "not_found"
    assert agent.status(gid)["result"]["datasets_scored"] == 2   # agent's view: validation only
    assert agent.tasks()["budget"]["used"] == 1                  # finals don't use the budget


def test_family_results_command(family, capsys):
    store, url, agent, workers = family
    gid = agent.submit_async("fitness", EXAMPLE_BUNDLE)["submission_id"]
    _drain(workers)
    store.request_finals(gid)
    _drain(workers)
    hub_mod.main(["--state", str(store.state), "results", "fitness"])
    out = capsys.readouterr().out
    assert gid in out and "combined=" in out and out.count("final ") == 2


def test_unbudgeted_tokens_still_work(connected):
    cfg, store, url, agent = connected
    assert agent.tasks()["budget"] == {"evaluations": None, "used": 0}



# ------------------------------------------- one code per task, updates in place

def test_enrolling_again_keeps_the_code(tmp_path):
    store = hub_mod.HubStore(tmp_path / "h")
    d = scaffold.write_task("t1", tmp_path / "t", [("id", "string", False), ("x", "float", False)],
                            [("id", "string", False), ("y", "float", False)], ["id"], "regression", "y")
    code1, _ = store.add_enrollment(d)
    code2, _ = store.add_enrollment(d)
    assert code1 == code2 and store.enrollment(code1)[3] == 2


def test_task_update_reaches_provider_who_approves_it(connected, provider_files, capsys):
    cfg, store, url, agent = connected
    _, task_dir = provider_files
    w = Worker(cfg)
    w.step()                                          # hello: publishes version 1
    assert "update" not in agent.contract("binding")["description"]

    # BioModelBench edits the task and enrolls again: same code, version 2.
    text = (task_dir / "contract.yaml").read_text().replace('description: ""', 'description: "update 2"')
    (task_dir / "contract.yaml").write_text(text)
    code = store.code_for("binding")
    assert store.add_enrollment(task_dir)[0] == code

    # The provider's node notices and tells them, but keeps the approved version.
    w._last_hello = 0
    w.step()
    from bmb_eval.node import pending_updates
    assert pending_updates(cfg) == {"binding": 2}
    assert "update 2" not in agent.contract("binding")["description"]
    cli.main(["--config", str(cfg.path), "status"])
    assert "run `bmb-eval update`" in capsys.readouterr().out

    # The provider reviews and approves it with one command: no code to look up.
    cli.main(["--config", str(cfg.path), "update", "--yes", "--no-service"])
    cfg2 = Config.load(cfg.path)
    assert cfg2.raw["tasks"]["binding"]["package_version"] == 2 and pending_updates(cfg2) == {}
    assert not approval.diff(cfg2)
    w2 = Worker(cfg2)
    w2.step()
    assert agent.contract("binding")["description"] == "update 2"
    sid = agent.submit_async("binding", EXAMPLE_BUNDLE)["submission_id"]
    w2.drain()
    assert agent.status(sid)["result"]["status"] == "ok"


def test_update_with_nothing_pending(connected, capsys):
    cfg, *_ = connected
    cli.main(["--config", str(cfg.path), "update", "--yes", "--no-service"])
    assert "up to date" in capsys.readouterr().out
