"""End-to-end and unit tests. Run on the local (unisolated) backend, so they
cover the pipeline, gates, hub and node logic; container isolation itself is
covered by `bmb-eval selftest` on a docker/podman node."""
from __future__ import annotations

import io
import json
import re
import shutil
import tarfile
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from bmb_eval import approval, audit, leakage, scaffold
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
TASK = "example_regression"
EXAMPLE_BUNDLE = REPO / "examples" / "bundles" / "example_baseline"


@pytest.fixture
def node_dir(tmp_path):
    shutil.copytree(REPO / "tasks", tmp_path / "tasks")
    inp, ans = scaffold.example_private_data(tmp_path / "private_data" / TASK)
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
    return Contract.load(REPO / "tasks" / TASK / "contract.yaml")


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


def test_scaffold_does_not_publish_private_bounds(tmp_path):
    inp, ans = scaffold.example_private_data(tmp_path / "d")
    d = scaffold.add_task("t", tmp_path / "tasks", inp, ans, ["variant_id"], "classification",
                          target="fitness")
    c = Contract.load(d / "contract.yaml")
    assert all(f.min is None and f.max is None and f.allowed is None for f in c.inputs)


def test_scaffold_refuses_target_in_inputs(tmp_path):
    import pandas as pd
    pd.DataFrame({"id": ["a"], "y": [1.0]}).to_parquet(tmp_path / "i.parquet")
    pd.DataFrame({"id": ["a"], "y": [1.0]}).to_parquet(tmp_path / "a.parquet")
    with pytest.raises(SystemExit):
        scaffold.add_task("t", tmp_path / "tasks", tmp_path / "i.parquet",
                          tmp_path / "a.parquet", ["id"], "regression")


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
    httpd.RequestHandlerClass = hub_mod.make_handler(store, url, REPO, 64)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield store, url
    httpd.shutdown()


@pytest.fixture
def provider_files(tmp_path):
    """The provider's one CSV (with an extra column we never asked for), and the
    task we wrote for them without seeing the data."""
    import pandas as pd
    inp, ans = scaffold.example_private_data(tmp_path / "raw")
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
    assert not (store.state / "bundles" / f"{sid}.tar.gz").exists()


def test_connect_code_is_single_use(hub, connected, provider_files):
    cfg, store, url, _ = connected
    code = store.q("SELECT code FROM enrollments", one=True)[0]
    with pytest.raises(SystemExit, match="unknown or already used"):
        _connect(cfg.path.parent, url, code, provider_files[0])


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
    assert "id, seq, kd" in msg and "never sees kd" in msg
