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
import yaml

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


def test_gate_clamps_floating_point_past_a_bound():
    # sklearn can return a perfect score as 1.0000000000000002; the contract's max is 1.
    out = gate_metrics(_contract(), b'{"spearman": 1.0000000000000002, "pearson": -1.0000000000000002, "rmse": 1}')
    assert out["spearman"] == 1.0 and out["pearson"] == -1.0
    with pytest.raises(EvalError):                     # a real excess still fails
        gate_metrics(_contract(), b'{"spearman": 1.001, "pearson": 0.1, "rmse": 1}')


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


def test_leakage_stated_as_rows_of_hidden_values():
    import dataclasses
    c = _contract()                                   # one float answer: ~10 bits per row
    b = leakage.budget(c)
    assert leakage.bits_per_row(c) == pytest.approx(leakage.VALUE_BITS)
    assert leakage.rows_recoverable(c, b.lifetime) == pytest.approx(b.lifetime / leakage.VALUE_BITS)
    text = " ".join(cli._summary(c, 100_000).split())
    assert "recover the fitness values of about" in text and "of your 100,000 rows" in text
    assert "of all 50 rows" in " ".join(cli._summary(c, 50).split())   # capped at the data size
    # A yes/no answer costs one bit per row, so many more rows are exposed.
    c2 = dataclasses.replace(c, answers=[c.answers[0], dataclasses.replace(c.answers[1], type="bool")])
    assert leakage.rows_recoverable(c2, b.lifetime) == pytest.approx(b.lifetime)
    # Answers kept in private files: no per-row estimate, so it counts measured values instead.
    c3 = dataclasses.replace(c, answers=[c.answers[0]])
    assert leakage.rows_recoverable(c3, b.lifetime) is None
    assert "hidden measured values in your data" in " ".join(cli._summary(c3, 100).split())


def test_new_task_has_no_split_unless_asked(tmp_path):
    cols = [("id", "string", False), ("x", "float", False)], [("id", "string", False), ("y", "float", False)]
    d = scaffold.write_task("t_off", tmp_path, *cols, ["id"], "regression", "y")
    assert Contract.load(d / "contract.yaml").split is None
    d = scaffold.write_task("t_on", tmp_path, *cols, ["id"], "regression", "y", split=0.5)
    assert Contract.load(d / "contract.yaml").split.validation_fraction == 0.5


def test_progress_bar_off_a_terminal_prints_every_tenth(capsys):
    from bmb_eval import progress
    bar = progress.Bar("Sandbox image", 1000)
    for _ in range(100):
        bar.update(10)
    bar.close()
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 11 and "100%" in lines[-1] and "1.0 KB of 1.0 KB" in lines[-1]
    assert progress.transfer_estimate(3_000_000_000) == "about 2 minutes"
    assert progress.duration(65) == "1m 05s"


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
    with pytest.raises(SystemExit, match="already connected from another"):
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


def test_bad_value_in_the_test_half_is_found(tmp_path):
    """A provider's mistake is reported whichever half the split put it in."""
    import pandas as pd
    from bmb_eval.config import TaskBinding
    from bmb_eval.contract import Contract
    d = scaffold.write_task("assays", tmp_path / "authored",
                            [("id", "string", False), ("assay", "string", False)],
                            [("id", "string", False), ("y", "float", False)],
                            ["id"], "regression", "y")
    text = (d / "contract.yaml").read_text()
    (d / "contract.yaml").write_text(text.replace("name: assay, type: string",
                                                  "name: assay, type: string, allowed: [a, b]"))
    c = Contract.load(d / "contract.yaml")
    for half, assays in (("validation", ["a", "b"]), ("test", ["a", "typo"])):
        (tmp_path / half).mkdir()
        pd.DataFrame({"id": [f"{half}{i}" for i in range(2)], "assay": assays}).to_parquet(
            tmp_path / half / "samples.parquet", index=False)
    pd.DataFrame({"id": ["validation0", "validation1", "test0", "test1"], "y": [1.0, 2, 3, 4]}).to_parquet(
        tmp_path / "answers.parquet", index=False)
    t = TaskBinding("assays", d, tmp_path / "validation" / "samples.parquet", tmp_path / "answers.parquet",
                    c, tmp_path / "validation", None, tmp_path / "test")
    assert cli.data_problems(t) == ["Column 'assay' in your data file has values that aren't in "
                                    "the allowed list."]


def test_windows_excel_csv_is_read(tmp_path):
    from bmb_eval.prepare import read_table
    (tmp_path / "w.csv").write_bytes("id,note\nr1,IC50 in µM at 37 °C\n".encode("cp1252"))
    assert read_table(tmp_path / "w.csv")["note"][0] == "IC50 in µM at 37 °C"


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


def test_total_quota_counts_per_submitter(connected):
    cfg, store, url, agent = connected
    cfg.tasks["binding"].contract.limits.submissions_total = 1
    other = Client(url, store.add_agent("agent-2"))
    w = Worker(cfg)
    a = agent.submit_async("binding", EXAMPLE_BUNDLE)["submission_id"]
    b = agent.submit_async("binding", EXAMPLE_BUNDLE)["submission_id"]
    c = other.submit_async("binding", EXAMPLE_BUNDLE)["submission_id"]
    w.drain()
    assert agent.status(a)["result"]["status"] == "ok"
    assert agent.status(b)["result"]["error"]["code"] == "quota_exceeded"
    assert other.status(c)["result"]["status"] == "ok"      # another run has its own cap


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


def test_hello_with_many_contracts_publishes_them(connected):
    # A node serving dozens of tasks sends every public contract in each hello;
    # that grows past 64 KB (36 tasks reached it), and a rejected hello silently
    # stops contract updates.
    cfg, store, url, agent = connected
    big = {"id": "binding", "description": "x" * (80 << 10)}
    HubAPI(url, cfg.hub["secret"]).hello({"binding": big}, True, [])
    assert [t["contract"]["description"] for t in agent.tasks()["tasks"]] == [big["description"]]


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


def test_gcs_download_links_reused():
    """A download link is signed once per (agent, file) and reused while plenty of it
    remains: a family's datasets and a re-offered job cost no new IAM calls."""
    pytest.importorskip("google.cloud.storage")
    g = object.__new__(hub_mod.GCSBlobs)
    g._signed, g._signed_lock = {}, threading.Lock()
    calls = []
    g._sign = lambda agent, sha, method: calls.append((agent, sha)) or f"https://s/{agent}/{sha}/{len(calls)}"
    first = g.download("a", "1" * 64, 5)["url"]
    assert g.download("a", "1" * 64, 5)["url"] == first and len(calls) == 1
    g.download("b", "1" * 64, 5)                       # another agent's copy: its own link
    assert len(calls) == 2
    g._signed[("a", "1" * 64)] = (time.time() - 1, first)   # under URL_REUSE_MIN left: re-sign
    assert g.download("a", "1" * 64, 5)["url"] != first and len(calls) == 3


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


def _split_task(tmp_path, final_evaluations=2, name="split_task", limits="", final_limits="",
                baseline_final_evaluations=None):
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
            f"split: {{validation_fraction: 0.5, final_evaluations: {final_evaluations}"
            + (f", baseline_final_evaluations: {baseline_final_evaluations}"
               if baseline_final_evaluations is not None else "") + "}\n\n"
            + (f"final_limits: {{{final_limits}}}\n\n" if final_limits else "")
            + text[end:])
    if limits:
        text = text.replace("limits:\n", f"limits:\n  {limits}\n", 1)
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
    assert w.store.count_total("agent-split", "split_task") == 1


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


def _final_results(store, fids):
    return {f: json.loads(store.q("SELECT result FROM submissions WHERE id=?", (f,), one=True)[0])
            for f in fids}


def test_baseline_finals_have_their_own_cap(hub, node_dir, provider_files, tmp_path):
    """One test evaluation per agent run; our baselines are evaluated separately and
    don't use up the agents' finals (nor can agents use the baselines')."""
    store, url = hub
    data, _ = provider_files
    code, _ = store.add_enrollment(_split_task(tmp_path, final_evaluations=1, name="sep",
                                               baseline_final_evaluations=2))
    cfg = _connect(node_dir, url, code, data)
    op = Client(url, store.set_operator())
    tok = op.agent_token("ref:score", 1, kind="baseline")
    assert tok["kind"] == "baseline"
    agent, base = Client(url, store.add_agent("run-a")), Client(url, tok["token"])
    a_sid = agent.submit_async("sep", EXAMPLE_BUNDLE)["submission_id"]
    b_sid = base.submit_async("sep", EXAMPLE_BUNDLE)["submission_id"]
    assert store.q("SELECT kind FROM submissions WHERE id=?", (b_sid,), one=True)[0] == "baseline"
    w = Worker(cfg)
    w.drain()
    b = [store.request_final(b_sid) for _ in range(3)]
    a = [store.request_final(a_sid) for _ in range(2)]
    w.drain()
    res = _final_results(store, a + b)
    assert res[a[0]]["status"] == "ok" and res[a[1]]["error"]["code"] == "final_quota_exceeded"
    assert res[b[0]]["status"] == res[b[1]]["status"] == "ok"      # baselines ran first, agent's still ran
    assert res[b[2]]["error"]["code"] == "final_quota_exceeded"
    assert w.store.count_final("sep") == 1 and w.store.count_final("sep", "baseline") == 2
    assert {r[0] for r in w.store.db.execute(
        "SELECT kind FROM submissions WHERE id IN (?, ?)", (b[0], b[1]))} == {"baseline"}


def test_baseline_finals_default_to_none(hub, node_dir, provider_files, tmp_path):
    """A contract without baseline_final_evaluations allows exactly what it did before:
    agents' finals up to final_evaluations, and no baseline finals at all."""
    store, url = hub
    data, _ = provider_files
    d = _split_task(tmp_path, final_evaluations=1, name="dflt")
    c = Contract.load(d / "contract.yaml")
    assert c.split.baseline_final_evaluations == 0 and c.split.total_finals == 1
    assert "baseline_final_evaluations" not in c.public()["split"]   # public contract unchanged
    code, _ = store.add_enrollment(d)
    cfg = _connect(node_dir, url, code, data)
    agent = Client(url, store.add_agent("run-d"))
    base = Client(url, store.add_agent("ref-d", 1, kind="baseline"))
    a_sid = agent.submit_async("dflt", EXAMPLE_BUNDLE)["submission_id"]
    b_sid = base.submit_async("dflt", EXAMPLE_BUNDLE)["submission_id"]
    w = Worker(cfg)
    w.drain()
    bf, af, af2 = store.request_final(b_sid), store.request_final(a_sid), store.request_final(a_sid)
    w.drain()
    res = _final_results(store, [bf, af, af2])
    assert res[bf]["error"]["code"] == "final_quota_exceeded"
    assert res[af]["status"] == "ok" and res[af2]["error"]["code"] == "final_quota_exceeded"
    assert w.store.count_final("dflt") == 1 and w.store.count_final("dflt", "baseline") == 0


def test_unknown_kind_counts_as_an_agents_final(hub, node_dir, provider_files, tmp_path):
    store, url = hub
    data, _ = provider_files
    code, _ = store.add_enrollment(_split_task(tmp_path, final_evaluations=1, name="unk",
                                               baseline_final_evaluations=5))
    cfg = _connect(node_dir, url, code, data)
    agent = Client(url, store.add_agent("run-u"))
    sid = agent.submit_async("unk", EXAMPLE_BUNDLE)["submission_id"]
    w = Worker(cfg)
    w.drain()
    f1, f2 = store.request_final(sid), store.request_final(sid)
    store.q("UPDATE submissions SET kind='sota' WHERE id IN (?, ?)", (f1, f2))
    w.drain()
    res = _final_results(store, [f1, f2])
    assert res[f1]["status"] == "ok" and res[f2]["error"]["code"] == "final_quota_exceeded"
    assert w.store.count_final("unk") == 1 and w.store.count_final("unk", "baseline") == 0


def test_baseline_final_evaluations_validated_and_counted(tmp_path):
    import dataclasses
    raw = yaml.safe_load((_split_task(tmp_path, final_evaluations=2) / "contract.yaml").read_text())
    for bad in (-1, "2", 1.5, True):
        raw["split"]["baseline_final_evaluations"] = bad
        with pytest.raises(ValueError, match="baseline_final_evaluations"):
            Contract.from_dict(raw)
    raw["split"]["baseline_final_evaluations"] = 3
    c = Contract.from_dict(raw)
    assert c.split.total_finals == 5
    assert Contract.from_dict(c.public(), public=True).split.baseline_final_evaluations == 3
    without = dataclasses.replace(c, split=dataclasses.replace(c.split, baseline_final_evaluations=0))
    b, b0 = leakage.budget(c), leakage.budget(without)
    assert b.lifetime - b0.lifetime == pytest.approx(3 * b.per_final)
    assert "3 of BioModelBench's reference models (5 in total)" in leakage.describe(c)
    summary = cli._summary(c, 500)
    assert "at most 2 final evaluations of agents' models and 3 of BioModelBench's" in summary
    assert "across one agent's 205 evaluations" in summary
    assert "reference models" not in cli._summary(without, 500)


def test_final_on_task_without_split_is_refused(connected):
    cfg, store, url, agent = connected
    sid = agent.submit_async("binding", EXAMPLE_BUNDLE)["submission_id"]
    w = Worker(cfg)
    w.drain()
    fid = store.request_final(sid)
    w.drain()
    r = json.loads(store.q("SELECT result FROM submissions WHERE id=?", (fid,), one=True)[0])
    assert r["error"]["code"] == "no_test_split"
    # Refused, so not kept in place of a real final: once the task has a split, asking
    # without `again` queues a new one rather than returning the refusal.
    assert store.request_final(sid, again=False) == f"{sid}-final2"
    assert store.request_final(sid, again=False) == f"{sid}-final2"   # queued: kept


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
    assert s.count_total("a", "t") == 1                          # old rows count as validation
    h = tmp_path / "hub"
    h.mkdir()
    hdb = sqlite3.connect(h / "hub.db")
    hdb.execute("CREATE TABLE submissions (id TEXT PRIMARY KEY, task TEXT NOT NULL, agent TEXT NOT NULL, "
                "created REAL NOT NULL, status TEXT NOT NULL, node_id TEXT, result TEXT, manifest TEXT NOT NULL)")
    hdb.commit()
    hdb.close()
    assert "split" in {r[1] for r in hub_mod.HubStore(h).db.execute("PRAGMA table_info(submissions)")}


def test_existing_databases_gain_kind_columns(tmp_path):
    """Nodes and hubs from before baseline finals: every existing final is an agent's."""
    import sqlite3
    from bmb_eval.store import Store
    db = sqlite3.connect(tmp_path / "node.db")
    db.execute("CREATE TABLE submissions (id TEXT PRIMARY KEY, task TEXT NOT NULL, submitter TEXT NOT NULL, "
               "day TEXT NOT NULL, status TEXT NOT NULL, created REAL NOT NULL, release_at REAL NOT NULL, "
               "finished REAL, result TEXT, posted INTEGER NOT NULL DEFAULT 0, "
               "split TEXT NOT NULL DEFAULT 'validation')")
    db.execute("INSERT INTO submissions VALUES ('f', 't', 'a', '2026-01-01', 'done', 0, 0, 0, '{}', 1, 'test')")
    db.commit()
    db.close()
    s = Store(tmp_path)
    assert s.count_final("t") == 1 and s.count_final("t", "baseline") == 0
    s.add("b", "t", "ref", 0, "test", "baseline")
    assert s.count_final("t") == 1 and s.count_final("t", "baseline") == 1
    h = tmp_path / "hub"
    h.mkdir()
    hdb = sqlite3.connect(h / "hub.db")
    hdb.execute("CREATE TABLE agents (name TEXT PRIMARY KEY, token_sha256 TEXT NOT NULL, eval_budget INTEGER)")
    hdb.execute("INSERT INTO agents VALUES ('old-run', 'x', 3)")
    hdb.execute("CREATE TABLE submissions (id TEXT PRIMARY KEY, task TEXT NOT NULL, agent TEXT NOT NULL, "
                "created REAL NOT NULL, status TEXT NOT NULL, node_id TEXT, result TEXT, manifest TEXT NOT NULL, "
                "split TEXT NOT NULL DEFAULT 'validation', group_id TEXT)")
    hdb.execute("INSERT INTO submissions VALUES ('s', 't', 'old-run', 0, 'done', NULL, NULL, '{}', "
                "'validation', NULL)")
    hdb.commit()
    hdb.close()
    hs = hub_mod.HubStore(h)
    assert hs.agent_kind("old-run") == "agent"
    assert hs.q("SELECT kind FROM submissions WHERE id='s'", one=True)[0] == "agent"
    assert hs.q("SELECT kind FROM submissions WHERE id=?", (hs.request_final("s"),), one=True)[0] == "agent"
    hs.add_agent("ref", 1, kind="baseline")
    assert hs.agent_kind("ref") == "baseline"
    with pytest.raises(ValueError):
        hs.add_agent("x", 1, kind="sota")


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


def test_task_folder_with_subfolders_and_binary_files_reaches_the_node(hub, node_dir, provider_files):
    store, url = hub
    data, task_dir = provider_files
    (task_dir / "refs").mkdir()
    (task_dir / "refs" / "panel.txt").write_text("GENE1\nGENE2\n")
    (task_dir / "refs" / "table.bin").write_bytes(bytes(range(256)))
    (task_dir / "bundle").mkdir()
    (task_dir / "bundle" / "predict.py").write_text("# example model: not shipped\n")
    (task_dir / ".secret").write_text("not shipped")
    code, _ = store.add_enrollment(task_dir)
    assert set(store.enrollment(code)[1]) == {"contract.yaml", "score.py", "refs/panel.txt", "refs/table.bin"}
    cfg = _connect(node_dir, url, code, data)
    got = cfg.tasks["binding"].dir
    assert (got / "refs" / "panel.txt").read_text() == "GENE1\nGENE2\n"
    assert (got / "refs" / "table.bin").read_bytes() == bytes(range(256))
    assert not (got / "bundle").exists() and not (got / ".secret").exists()


def test_task_package_paths_stay_inside_the_task_folder(tmp_path):
    for bad in ("../escape.py", "/etc/x", "a/../../b", ".hidden", "a\\..\\b"):
        with pytest.raises(SystemExit):
            cli.write_task_files({bad: "x"}, tmp_path / "t")
    assert not (tmp_path / "escape.py").exists()


def test_task_folder_too_big_to_package(tmp_path, monkeypatch):
    d = scaffold.write_task("t1", tmp_path / "t", [("id", "string", False), ("x", "float", False)],
                            [("id", "string", False), ("y", "float", False)], ["id"], "regression", "y")
    monkeypatch.setattr(hub_mod, "MAX_PACKAGE_BYTES", 100)
    (d / "big.bin").write_bytes(b"\0" * 200)
    with pytest.raises(ValueError, match="task packages are for"):
        hub_mod.HubStore.package_files(d)


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



def test_optional_numeric_output_may_be_left_empty():
    import pandas as pd
    from bmb_eval.contract import Field, _check_column
    f = Field(name="sd", type="float", nullable=True, min=0)
    assert _check_column(f, pd.Series([None, None], dtype=object)) is None
    assert _check_column(f, pd.Series([0.5, None], dtype=object)) is None
    assert _check_column(f, pd.Series(["x", None], dtype=object)) == "wrong_type"
    assert _check_column(f, pd.Series([-1.0, None], dtype=object)) == "below_min"



def test_second_task_connects_with_bare_code_on_same_machine(hub, connected, provider_files, tmp_path):
    cfg, store, url, _ = connected
    data, _ = provider_files
    d = scaffold.write_task("second", tmp_path / "authored_second",
                            [("variant_id", "string", False), ("sequence", "string", False)],
                            [("variant_id", "string", False), ("fitness", "float", False)],
                            ["variant_id"], "regression", "fitness")
    code, _ = store.add_enrollment(d)
    # As in a provider email: just the code, no hub address.
    cli.main(["--config", str(cfg.path), "connect", code, "--data", str(data), "--yes", "--no-service"])
    assert "second" in Config.load(cfg.path).tasks


# ------------------------------------------------------- final-only metrics

def _final_only_task(tmp_path, name="fo_task"):
    d = _split_task(tmp_path, name=name)
    text = (d / "contract.yaml").read_text().replace(
        "  - {name: n, type: int, min: 0, max: 100000}\n",
        "  - {name: n, type: int, min: 0, max: 100000}\n"
        "  - {name: mae, type: float, min: 0, max: 10, final_only: true}\n")
    (d / "contract.yaml").write_text(text)
    (d / "score.py").write_text(SPLIT_SCORER.replace(
        '"n": int(len(df))}', '"n": int(len(df)), "mae": min(float(np.mean(np.abs(y - p))), 10.0)}'))
    return d


def test_final_only_metrics_leave_only_in_final_evaluations(hub, node_dir, provider_files, tmp_path):
    store, url = hub
    data, _ = provider_files
    code, _ = store.add_enrollment(_final_only_task(tmp_path))
    cfg = _connect(node_dir, url, code, data)
    agent = Client(url, store.add_agent("agent-fo"))
    w = Worker(cfg)
    w.step()                                       # check-in publishes the contract
    assert any(m.get("final_only") for m in agent.contract("fo_task")["metrics"])   # agents know
    sid = agent.submit_async("fo_task", EXAMPLE_BUNDLE)["submission_id"]
    w.drain()
    assert set(agent.status(sid)["result"]["metrics"]) == {"spearman", "n"}      # no mae in feedback
    fid = store.request_final(sid)
    w.drain()
    final = json.loads(store.q("SELECT result FROM submissions WHERE id=?", (fid,), one=True)[0])
    assert set(final["metrics"]) == {"spearman", "n", "mae"}


def test_final_only_needs_split_and_cant_be_headline(tmp_path):
    d = _final_only_task(tmp_path)
    text = (d / "contract.yaml").read_text()
    (d / "contract.yaml").write_text(text.replace("headline: spearman", "headline: mae"))
    with pytest.raises(ValueError, match="headline"):
        Contract.load(d / "contract.yaml")
    (d / "contract.yaml").write_text(text[:text.index("split:")] + text[text.index("export:"):])
    with pytest.raises(ValueError, match="need a held-out test set"):
        Contract.load(d / "contract.yaml")


def test_final_only_metrics_cost_leakage_only_in_finals(tmp_path):
    with_fo = leakage.budget(Contract.load(_final_only_task(tmp_path / "a") / "contract.yaml"))
    plain = leakage.budget(Contract.load(_split_task(tmp_path / "b") / "contract.yaml"))
    assert with_fo.per_submission == plain.per_submission and with_fo.per_final > plain.per_final


def test_final_limits_apply_only_to_final_runs(tmp_path):
    c = Contract.load(_split_task(tmp_path, final_limits="gpus: 2, gpu_memory_gb: 80, max_bundle_gb: 150")
                      / "contract.yaml")
    assert c.limits_for("validation").gpus == 0
    final = c.limits_for("test")
    assert (final.gpus, final.gpu_memory_gb, final.max_bundle_gb) == (2, 80, 150)
    assert final.cpus == c.limits.cpus                      # unset fields inherit
    assert c.public()["final_limits"]["gpus"] == 2          # agents see the final hardware
    assert Contract.from_dict(c.public(), public=True).limits_for("test").gpus == 2


@pytest.mark.parametrize("final_limits, why", [
    ({"submissions_total": 5}, "quotas"),
    ({"gpu": 2}, "unexpected keyword"),
])
def test_final_limits_rejected(tmp_path, final_limits, why):
    raw = yaml.safe_load((_split_task(tmp_path) / "contract.yaml").read_text())
    raw["final_limits"] = final_limits
    with pytest.raises((ValueError, TypeError), match=why):
        Contract.from_dict(raw)
    raw.pop("split")
    raw["final_limits"] = {"gpus": 1}
    with pytest.raises(ValueError, match="split"):
        Contract.from_dict(raw)


def test_final_candidate_skips_validation_and_runs_under_final_limits(hub, node_dir, provider_files,
                                                                      tmp_path, monkeypatch):
    # Validation can't take any bundle (0 GB); finals can. Like a 70B model on a CPU-only
    # validation tier: register it, then run it only as a final.
    store, url = hub
    data, _ = provider_files
    code, _ = store.add_enrollment(_split_task(tmp_path, name="big", limits="max_bundle_gb: 0",
                                               final_limits="max_bundle_gb: 1"))
    monkeypatch.setattr(cli, "_selftest", lambda *a, **k: 0)   # its probes can't fit in 0 GB either
    cfg = _connect(node_dir, url, code, data)
    agent = Client(url, store.add_agent("agent-big", evaluations=1))
    r = agent.register_final("big", EXAMPLE_BUNDLE)
    assert r["status"] == "registered"
    w = Worker(cfg)
    w.drain()
    assert agent.status(r["submission_id"])["status"] == "registered"   # never run on validation
    assert agent.tasks()["budget"] == {"evaluations": 1, "used": 0}
    assert w.store.count_total("agent-big", "big") == 0
    fid = store.request_final(r["submission_id"])
    w.drain()
    final = json.loads(store.q("SELECT result FROM submissions WHERE id=?", (fid,), one=True)[0])
    assert final["status"] == "ok" and final["split"] == "test"
    # The same model as a normal submission is refused by validation's size limit.
    sid = agent.submit_async("big", EXAMPLE_BUNDLE)["submission_id"]
    w.drain()
    assert agent.status(sid)["result"]["error"]["code"] == "bundle_too_large"


def test_final_refused_until_the_gpus_are_there(hub, node_dir, provider_files, tmp_path, monkeypatch):
    from bmb_eval import sandbox as sandbox_mod
    store, url = hub
    data, _ = provider_files
    code, _ = store.add_enrollment(_split_task(tmp_path, name="gpu_final", final_limits="gpus: 2"))
    cfg = _connect(node_dir, url, code, data)     # connects fine without GPUs (validation is CPU)
    agent = Client(url, store.add_agent("agent-gpu"))
    sid = agent.submit_async("gpu_final", EXAMPLE_BUNDLE)["submission_id"]
    w = Worker(cfg)
    w.drain()
    assert agent.status(sid)["result"]["status"] == "ok"
    monkeypatch.setattr(sandbox_mod, "gpus_here", lambda: [80])
    fid = store.request_final(sid)
    w.drain()
    r = json.loads(store.q("SELECT result FROM submissions WHERE id=?", (fid,), one=True)[0])
    assert r["error"]["code"] == "final_hardware_missing"
    assert w.store.count_final("gpu_final") == 0      # doesn't use up a final evaluation


def test_sealed_final_limits_apply_only_on_the_sealed_node(tmp_path):
    raw = yaml.safe_load((_split_task(tmp_path, final_limits="gpus: 2, max_bundle_gb: 150")
                          / "contract.yaml").read_text())
    raw["final_limits_sealed"] = {"gpus": 1, "max_bundle_gb": 80}
    c = Contract.from_dict(raw)
    assert (c.limits_for("test").gpus, c.limits_for("test").max_bundle_gb) == (2, 150)
    sealed = c.limits_for("test", sealed=True)
    assert (sealed.gpus, sealed.max_bundle_gb) == (1, 80)
    assert c.limits_for("validation", sealed=True).gpus == 0
    assert Contract.from_dict(c.public(), public=True).limits_for("test", sealed=True).gpus == 1


def test_fixed_split_by_column(tmp_path):
    """A published test set: the provider's own column decides validation vs test,
    and the column is never shown to the model."""
    import pandas as pd
    from bmb_eval.prepare import apply_split, prepare
    d = scaffold.write_task("fixed", tmp_path / "t", [("id", "string", False), ("seq", "string", False)],
                            [("id", "string", False), ("y", "float", False),
                             ("half", "string", False)], ["id"], "regression", "y")
    raw = yaml.safe_load((d / "contract.yaml").read_text())
    raw["split"] = {"by_column": "half", "final_evaluations": 1}
    c = Contract.from_dict(raw)
    rows = pd.DataFrame({"id": [f"r{i}" for i in range(10)], "seq": ["ACGT"] * 10,
                         "y": range(10), "half": ["validation"] * 4 + ["test"] * 6})
    rows.to_csv(tmp_path / "data.csv", index=False)
    inputs_dir, answers, n = prepare(c, tmp_path / "data.csv", tmp_path / "out")
    val, test, n_val, n_test = apply_split(c, tmp_path / "out")
    assert (n_val, n_test) == (4, 6)
    assert set(pd.read_parquet(val / "samples.parquet")["id"]) == {"r0", "r1", "r2", "r3"}
    assert "half" not in pd.read_parquet(val / "samples.parquet").columns
    assert Contract.from_dict(c.public(), public=True).split.by_column == "half"
    raw["split"] = {"final_evaluations": 1}
    with pytest.raises(ValueError, match="by_column"):
        Contract.from_dict(raw)


def test_container_caps_math_thread_pools_to_the_cpu_limit():
    """On a 32-CPU host, OpenBLAS started 32 threads in every stage and crashed under
    the fork_bomb attack's pid limit (a provider's self-test failed on this)."""
    from bmb_eval.contract import Limits
    from bmb_eval.sandbox import container_cmd
    cmd = container_cmd("podman", "img", None, name="n", entry="score", mounts=[], env={},
                        limits=Limits(cpus=2))
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        assert f"{v}=2" in cmd


def test_flood_output_accepts_truncated_delivery():
    from bmb_eval.redteam import ATTACKS
    flood = next(a for a in ATTACKS if a.name == "flood_output")
    assert flood.expect == "predictions_too_large" and "predictions_invalid" in flood.also
    assert all(not a.also for a in ATTACKS if a.name != "flood_output")


# ------------------------------------------- robustness (found stress-testing a 36-task node)

def test_rejected_hello_keeps_the_node_working(connected):
    # A node's check-in can be refused (it outgrew the hub's size cap). That used to
    # abort every step, so the node silently stopped taking work.
    from bmb_eval.node import HubError
    cfg, store, url, agent = connected
    w = Worker(cfg)
    w.step()                                                 # publishes the contract
    sid = agent.submit_async("binding", EXAMPLE_BUNDLE)["submission_id"]

    def refused(*a, **k):
        raise HubError("Body too large.", 413)
    w.api.hello = refused
    w._last_hello = 0
    w.drain()
    assert agent.status(sid)["status"] == "done"
    events = [json.loads(l)["event"] for l in (cfg.state_dir / "audit.jsonl").read_text().splitlines()]
    assert events.count("hello_rejected") == 1               # recorded (and alerted) once


def test_unreachable_hub_still_raises(connected):
    from bmb_eval.node import HubError
    cfg, *_ = connected
    w = Worker(cfg)

    def unreachable(*a, **k):
        raise HubError("cannot reach the hub")
    w.api.hello = unreachable
    with pytest.raises(HubError):
        w.step()


def test_job_reassembles_after_an_interrupted_run(tmp_path):
    # A restart mid-job leaves the bundle's hard links behind; the re-run used to fail
    # with SameFileError (copyfile onto its own link) and report infra_error.
    import hashlib
    cache = BlobCache(tmp_path / "blobs")
    blob = b"print('hi')\n"
    sha = hashlib.sha256(blob).hexdigest()
    cache.put(sha, len(blob), io.BytesIO(blob).read)
    files = [{"path": "predict.py", "sha256": sha, "size": len(blob)}]
    dest = tmp_path / "bundles" / "sid1"
    cache.assemble(files, dest)
    cache.assemble(files, dest)                               # interrupted job, run again
    assert (dest / "predict.py").read_bytes() == blob


def test_stale_sandbox_image_is_reported(monkeypatch):
    import subprocess
    from types import SimpleNamespace
    from bmb_eval import sandbox
    root = Path(sandbox.__file__).resolve().parents[2]
    stamp = sandbox.sandbox_stamp(root)
    # Same digest as the hub's `cat $SANDBOX_FILES | sha256sum | cut -c1-16`.
    shell = subprocess.run("cat " + " ".join(sandbox.SANDBOX_FILES) + " | sha256sum | cut -c1-16",
                           shell=True, cwd=root, capture_output=True, text=True).stdout.strip()
    assert stamp == shell
    cfg = SimpleNamespace(backend="podman", image="bmb-eval-sandbox:latest", runtime=None)
    monkeypatch.setattr(sandbox.ContainerSandbox, "image_label", lambda self, key: "0123456789abcdef")
    assert "older than this node's code" in sandbox.stale_image_message(cfg)
    monkeypatch.setattr(sandbox.ContainerSandbox, "image_label", lambda self, key: stamp)
    assert sandbox.stale_image_message(cfg) is None
    assert sandbox.stale_image_message(SimpleNamespace(backend="local")) is None


def test_enrolling_a_different_package_needs_replace(tmp_path, capsys):
    state = str(tmp_path / "h")
    hub_mod.main(["--state", state, "new-task", "kd", "--tasks-dir", str(tmp_path / "t"),
                  "--inputs", "id:string,seq:string", "--target", "kd:float", "--key", "id"])
    task = tmp_path / "t" / "kd"
    enroll = ["--state", state, "enroll", str(task), "--public-url", "https://hub.example.org"]
    hub_mod.main(enroll)
    store = hub_mod.HubStore(Path(state))
    code = store.code_for("kd")
    hub_mod.main(enroll)                                      # same package: no new version
    assert store.enrollment(code)[3] == 1 and "nothing changed" in capsys.readouterr().out
    text = (task / "contract.yaml").read_text().replace('description: ""', 'description: "other"')
    (task / "contract.yaml").write_text(text)
    with pytest.raises(SystemExit, match="contract.yaml"):    # another package under the same id
        hub_mod.main(enroll)
    assert store.enrollment(code)[3] == 1
    hub_mod.main(enroll + ["--replace"])
    assert store.code_for("kd") == code and store.enrollment(code)[3] == 2


def test_connect_names_a_fixed_split(hub, node_dir, provider_files, capsys):
    # A split taken from the provider's column was announced as "Split at random".
    import pandas as pd
    store, url = hub
    data, task_dir = provider_files
    df = pd.read_csv(data)
    df["half"] = ["validation" if i % 2 else "test" for i in range(len(df))]
    df.to_csv(data, index=False)
    raw = yaml.safe_load((task_dir / "contract.yaml").read_text())
    raw["answers"].append({"name": "half", "type": "string"})
    raw["split"] = {"by_column": "half", "final_evaluations": 1}
    (task_dir / "contract.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    code, _ = store.add_enrollment(task_dir)
    _connect(node_dir, url, code, data)
    out = capsys.readouterr().out
    assert "Split by the 'half' column" in out and "at random" not in out

def test_operator_issues_tokens_and_runs_finals_over_the_api(family):
    store, url, agent, workers = family
    op = Client(url, store.set_operator())
    assert op.tasks()["error"]["code"] == "unauthorized"                 # not an agent token
    assert agent.agent_token("x", 1)["error"]["code"] == "unauthorized"  # agents can't issue tokens
    for bad in (-1, None, 1.5, True):
        assert op._req("POST", "/v1/operator/agent-tokens",
                       {"name": "x", "evaluations": bad})["error"]["code"] == "bad_request"
    for bad in ("sota", "", None, 1):
        assert op._req("POST", "/v1/operator/agent-tokens",
                       {"name": "x", "evaluations": 1, "kind": bad})["error"]["code"] == "bad_request"
    tok = op.agent_token("run-1", 2)
    assert tok["kind"] == "agent" and store.agent_kind("run-1") == "agent"
    # Read-only status for the ops page: operator only.
    assert agent._req("GET", "/v1/operator/status")["error"]["code"] == "unauthorized"
    st = op._req("GET", "/v1/operator/status")
    assert {"nodes", "submissions", "recent", "now"} <= set(st)
    assert all({"id", "name", "last_seen", "accepting", "tasks"} <= set(n) for n in st["nodes"])
    run = Client(url, tok["token"])
    assert run.tasks()["budget"] == {"evaluations": 2, "used": 0}
    assert op.agent_token("run-1", 2)["error"]["code"] == "name_taken"   # a live token isn't cut off
    sid = run.register_final("fitness", EXAMPLE_BUNDLE)["submission_id"]
    assert op.final(sid)["error"]["code"] == "not_found"                 # nothing requested yet
    assert run.request_final(sid)["error"]["code"] == "unauthorized"     # only the operator asks
    finals = op.request_final(sid)["finals"]
    assert len(finals) == 2 and op.final(sid)["status"] == "pending"
    assert op.request_final(sid)["finals"] == finals                     # a retry spends no final
    _drain(workers)
    view = op.final(sid)
    assert view["status"] == "done" and view["tasks"] == ["fit_lab_a", "fit_lab_b"]
    assert view["result"]["datasets_scored"] == 2 and view["result"]["combined"] is not None
    assert all(d["split"] == "test" for d in view["result"]["datasets"].values())
    assert run.status(sid)["status"] == "registered"                     # the attempt never sees finals
    assert run.tasks()["budget"]["used"] == 0                            # finals cost no budget
    again = op.request_final(sid, again=True)["finals"]
    assert len(again) == 2 and not set(again) & set(finals)
    assert op.final(sid.replace("-", "_", 1))["error"]["code"] == "not_found"   # no wildcard match
    rotated = op.agent_token("run-1", 2, rotate=True)["token"]
    assert run.tasks()["error"]["code"] == "unauthorized" and Client(url, rotated).tasks()["budget"]
    new_op = store.set_operator()
    assert op.final(sid)["error"]["code"] == "unauthorized" and Client(url, new_op).final(sid)["tasks"]


def test_executable_bundle_files_stay_executable(tmp_path):
    # bacdive_multitrait 20261017T193938-codex-high-hub shipped bin/diamond and failed with
    # "Permission denied": bundle files were assembled read-only and not executable.
    import hashlib
    b = tmp_path / "bundle"
    (b / "bin").mkdir(parents=True)
    (b / "predict.py").write_text("def predict(df): return df")
    (b / "bin" / "tool").write_text("#!/bin/sh\necho ok\n")
    os.chmod(b / "bin" / "tool", 0o755)
    m = manifest_for(b)
    by_path = {f["path"]: f for f in m["files"]}
    assert by_path["bin/tool"].get("executable") is True and "executable" not in by_path["predict.py"]
    files = check_manifest(m, 1 << 30)
    c = BlobCache(tmp_path / "cache")
    for f in files:
        data = (b / f["path"]).read_bytes()
        c.put(f["sha256"], len(data), io.BytesIO(data).read)
    out = c.assemble(files, tmp_path / "assembled")
    assert os.access(out / "bin" / "tool", os.X_OK) and not os.access(out / "predict.py", os.X_OK)
    assert not os.access(c.path(by_path["bin/tool"]["sha256"]), os.X_OK)   # the shared blob is untouched
    with pytest.raises(EvalError):                                          # only `true` is accepted
        check_manifest({"files": [{**by_path["predict.py"], "executable": "yes"}]}, 1 << 30)


def test_parallel_node_runs_jobs_side_by_side_within_capacity(connected):
    # max_parallel > 1: jobs run in threads while the loop keeps checking in; each starts
    # only when its contract's CPUs, memory and GPUs fit what's free.
    cfg, store, url, agent = connected
    cfg.max_parallel = 4
    w = Worker(cfg)
    lim = cfg.tasks["binding"].contract.limits
    w.capacity = (lim.cpus * 2, 1 << 50, 0)                 # room for exactly two at once
    running, peak, lock = [0], [0], threading.Lock()

    def evaluate(task, bundle, run_id, submitter, bundle_id=None, split="validation"):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.3)
        with lock:
            running[0] -= 1
        return {"run_id": run_id, "status": "ok", "metrics": {"spearman": 0.9}}

    w.runner.evaluate = evaluate
    sids = [agent.submit_async("binding", EXAMPLE_BUNDLE)["submission_id"] for _ in range(4)]
    w.drain(max_s=30)
    assert peak[0] == 2                                      # never more than fits
    assert all(agent.status(s)["result"]["status"] == "ok" for s in sids)


def test_serial_node_is_unchanged_by_default(connected):
    cfg, store, url, agent = connected
    assert cfg.max_parallel == 1
    w = Worker(cfg)
    sid = agent.submit_async("binding", EXAMPLE_BUNDLE)["submission_id"]
    w.drain()
    assert not w._jobs and agent.status(sid)["result"]["status"] == "ok"

def test_next_bare_code_works_after_a_connect_stops_at_bad_data(hub, node_dir, provider_files,
                                                                 tmp_path):
    """Multi-task email, own machine: the first command (with the hub) hits a data problem;
    the next task's command has just the code."""
    import pandas as pd
    store, url = hub
    data, task_dir = provider_files
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    first, _ = store.add_enrollment(task_dir)
    pd.read_csv(data).drop(columns=["sequence"]).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(SystemExit):
        cli.main(["--config", str(fresh / "config.yaml"), "connect", first, "--hub", url,
                  "--data", str(tmp_path / "bad.csv"), "--yes", "--no-service"])
    second = scaffold.write_task("second", tmp_path / "authored_second",
                                 [("variant_id", "string", False), ("sequence", "string", False)],
                                 [("variant_id", "string", False), ("fitness", "float", False)],
                                 ["variant_id"], "regression", "fitness")
    code, _ = store.add_enrollment(second)
    # Must get past parsing the bare code (it would say "doesn't look like a connect code").
    try:
        cli.main(["--config", str(fresh / "config.yaml"), "connect", code, "--data", str(data),
                  "--yes", "--no-service"])
    except SystemExit as e:
        assert "connect code" not in str(e)


def test_sign_all_gives_one_url_per_distinct_file():
    from bmb_eval.hub import sign_all

    class Blobs:
        def download(self, agent, sha, size):
            time.sleep(0.01)
            return f"https://x/{agent}/{sha}/{size}"

    files = [{"sha256": f"{i:064x}", "size": i} for i in range(200)] + [{"sha256": f"{0:064x}", "size": 0}]
    t = time.time()
    urls = sign_all(Blobs(), "a", files)
    assert len(urls) == 200 and urls[f"{5:064x}"] == f"https://x/a/{5:064x}/5"
    assert time.time() - t < 1.0                      # concurrent, not 2 s one by one


def test_upload_plan_and_missing_check_run_concurrently():
    from bmb_eval.hub import missing_blobs, plan_uploads

    class Blobs:
        def has(self, agent, sha, size):
            time.sleep(0.01)
            return int(sha, 16) % 2 == 0
        def upload(self, agent, sha, size):
            time.sleep(0.01)
            return {"url": f"https://x/{sha}", "method": "PUT"}

    files = [{"sha256": f"{i:064x}", "size": i} for i in range(200)] + [{"sha256": f"{1:064x}", "size": 1}]
    t = time.time()
    ups = plan_uploads(Blobs(), "a", files)
    miss = missing_blobs(Blobs(), "a", files)
    assert len(ups) == 100 and all(int(u["sha256"], 16) % 2 for u in ups)
    assert miss == sorted(f"{i:064x}" for i in range(200) if i % 2)
    assert time.time() - t < 1.5                         # concurrent: ~6 s one by one


# ------------------------------------------- failed and stalled setups reach BioModelBench

def _reports(store):
    return [r[1:4] for r in store.q("SELECT code, step, reason, detail FROM setup_reports ORDER BY at")]


def test_failed_connect_reports_step_and_reason_and_alerts(hub, node_dir, provider_files, tmp_path,
                                                           monkeypatch):
    import pandas as pd
    store, url = hub
    data, task_dir = provider_files
    alerts = []
    monkeypatch.setattr(hub_mod, "alert", lambda store, text: alerts.append(text))
    code, _ = store.add_enrollment(task_dir)
    pd.read_csv(data).drop(columns=["sequence"]).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(SystemExit):
        cli.main(["--config", str(node_dir / "config.yaml"), "connect", code, "--hub", url,
                  "--data", str(tmp_path / "bad.csv"), "--yes", "--no-service"])
    assert _reports(store) == [("data_check", "data_problem", "")]
    # Declining at the approval screen is reported as such.
    monkeypatch.setattr(cli, "_ask", lambda prompt: "n")
    with pytest.raises(SystemExit):
        cli.main(["--config", str(node_dir / "config.yaml"), "connect", code, "--hub", url,
                  "--data", str(data), "--no-service"])
    assert _reports(store)[-1] == ("review", "declined", "")
    # An unexpected error reports only its class name.
    monkeypatch.setattr(cli, "_summary", lambda c, n: (_ for _ in ()).throw(RuntimeError("/secret/path")))
    with pytest.raises(RuntimeError):
        cli.main(["--config", str(node_dir / "config.yaml"), "connect", code, "--hub", url,
                  "--data", str(data), "--yes", "--no-service"])
    assert _reports(store)[-1] == ("review", "crashed", "RuntimeError")
    for _ in range(50):
        if len(alerts) == 3:
            break
        time.sleep(0.05)
    assert len(alerts) == 3 and "at checking their data: data_problem" in alerts[0]
    assert not any("secret" in a for a in alerts)
    probs = store.setup_problems()
    assert [p["reason"] for p in probs] == ["crashed"] and not probs[0]["resolved"]


def test_setup_report_accepts_only_fixed_fields(hub, provider_files):
    import urllib.error
    import urllib.request
    store, url = hub
    _, task_dir = provider_files
    code, _ = store.add_enrollment(task_dir)

    def post(c, body):
        req = urllib.request.Request(f"{url}/v1/enroll/{c}/report", data=json.dumps(body).encode(),
                                     method="POST", headers={"Content-Type": "application/json"})
        try:
            return urllib.request.urlopen(req).status
        except urllib.error.HTTPError as e:
            return e.code
    assert post(code, {"step": "upload", "reason": "gcloud_refused"}) == 204
    assert post(code, {"step": "upload", "reason": "my data is /home/x.csv"}) == 400
    assert post(code, {"step": "upload", "reason": "crashed", "detail": "a b/c"}) == 400
    assert post("AAAA-BBBB-CCCC", {"step": "upload", "reason": "crashed"}) == 404
    for _ in range(30):
        post(code, {"step": "upload", "reason": "crashed"})
    assert store.q("SELECT COUNT(*) FROM setup_reports", one=True)[0] == 20     # capped per code


def test_used_code_whose_setup_never_finished_is_stalled(hub, provider_files, monkeypatch):
    from bmb_eval.node import HubAPI
    store, url = hub
    _, task_dir = provider_files
    code, _ = store.add_enrollment(task_dir)
    HubAPI(url).activate(code, "sealed-p")                        # approved, then nothing
    assert store.setup_problems() == []                           # not yet an hour
    later = time.time() + hub_mod.STALL_S + 60
    [p] = store.setup_problems(later)
    assert p["kind"] == "stalled" and p["code"] == code
    alerts = []
    monkeypatch.setattr(hub_mod, "alert", lambda store, text: alerts.append(text))
    hub_mod.check_stalled(store, later)
    hub_mod.check_stalled(store, later + 600)                     # alerts once per code
    assert len(alerts) == 1 and "never finished" in alerts[0]
    assert "upload never completed" in alerts[0]                  # a sealed node: what's missing is the upload


def _fake_service(monkeypatch, start=None):
    """Stand in for systemd: the service "starts" by running start(cfg), or nothing at all."""
    def install(cfg):
        if start:
            start(cfg)
        return True, ""
    monkeypatch.setattr(cli, "_install_service", install)


def _connect_with_service(node_dir, url, code, data):
    cli.main(["--config", str(node_dir / "config.yaml"), "connect", code, "--hub", url,
              "--data", str(data), "--yes"])


def test_connect_is_done_only_when_the_background_node_checks_in(hub, node_dir, provider_files,
                                                                 monkeypatch, capsys):
    store, url = hub
    data, task_dir = provider_files
    code, _ = store.add_enrollment(task_dir)
    threads = []

    def start(cfg):                                               # what systemd would run
        t = threading.Thread(target=lambda: Worker(cfg).step(), daemon=True)
        t.start()
        threads.append(t)
    _fake_service(monkeypatch, start)
    _connect_with_service(node_dir, url, code, data)
    out = capsys.readouterr().out
    assert "has checked in with BioModelBench" in out and "is live" in out
    assert _reports(store) == []
    node_id = Config.load(node_dir / "config.yaml").hub["node_id"]
    assert store.q("SELECT last_seen FROM nodes WHERE id=?", (node_id,), one=True)[0]
    assert store.setup_problems(time.time() + hub_mod.STALL_S + 60) == []   # not a stall, ever
    for t in threads:
        t.join(5)


def test_connect_reports_a_node_that_never_checks_in(hub, node_dir, provider_files, monkeypatch, capsys):
    store, url = hub
    data, task_dir = provider_files
    code, _ = store.add_enrollment(task_dir)
    monkeypatch.setattr(cli, "CHECKIN_WAIT_S", 1)
    monkeypatch.setattr(cli, "_service_state", lambda name="bmb-eval": "active")
    # The service "started" but nothing ever reached the hub.
    _fake_service(monkeypatch)
    with pytest.raises(SystemExit) as e:
        _connect_with_service(node_dir, url, code, data)
    assert "hasn't checked in" in str(e.value) and "bmb-eval status" in str(e.value)
    assert _reports(store)[-1] == ("start_service", "stopped", "no_checkin")
    # The node ran and said why: the provider sees the error, the hub a fixed reason.
    _fake_service(monkeypatch, lambda cfg: audit.record(
        cfg.state_dir, "hub_unreachable", error="cannot reach https://hub: [Errno -2] Name or service not known"))
    with pytest.raises(SystemExit) as e:
        _connect_with_service(node_dir, url, code, data)
    assert "Name or service not known" in str(e.value) and "HTTPS_PROXY" in str(e.value)
    assert _reports(store)[-1] == ("start_service", "hub_error", "unreachable")
    _fake_service(monkeypatch, lambda cfg: audit.record(
        cfg.state_dir, "hub_unreachable", error="[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"))
    with pytest.raises(SystemExit) as e:
        _connect_with_service(node_dir, url, code, data)
    assert _reports(store)[-1] == ("start_service", "tls", "certificate")
    # The service kept dying.
    _fake_service(monkeypatch)
    monkeypatch.setattr(cli, "_service_state", lambda name="bmb-eval": "failed")
    monkeypatch.setattr(cli, "_service_log", lambda n=20, name="bmb-eval": "Traceback: boom")
    with pytest.raises(SystemExit) as e:
        _connect_with_service(node_dir, url, code, data)
    assert "isn't running" in str(e.value) and "boom" in str(e.value)
    assert _reports(store)[-1] == ("start_service", "crashed", "service_exited")
    # No user systemd at all: no "Connected" either, and how to run it by hand.
    monkeypatch.setattr(cli, "_install_service", lambda cfg: (False, "Failed to connect to bus: No medium found"))
    with pytest.raises(SystemExit) as e:
        _connect_with_service(node_dir, url, code, data)
    assert "No medium found" in str(e.value) and "bmb-eval run" in str(e.value)
    assert _reports(store)[-1] == ("start_service", "machine_not_ready", "service_install_failed")
    assert "✓ Connected" not in capsys.readouterr().out
    # Every one of those is a failure the operator sees, never a silent stall.
    probs = store.setup_problems(time.time() + hub_mod.STALL_S + 60)
    assert [p["kind"] for p in probs] == ["failed"] and probs[0]["step"] == "start_service"


def test_repo_root_survives_an_unreadable_working_directory(tmp_path, monkeypatch):
    # deploy/gcp/hub runs bmb-hub as the hub user from the operator's home, which it can't read.
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0)
    try:
        if os.access(locked / "x", os.R_OK):
            pytest.skip("running as root: nothing is unreadable")
        monkeypatch.setattr(hub_mod.Path, "cwd", classmethod(lambda cls: locked))
        assert hub_mod._repo_root() == REPO
    finally:
        locked.chmod(0o700)


def test_service_unit_carries_the_shells_proxy_and_certificate_settings(cfg, tmp_path, monkeypatch):
    import subprocess
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")
    monkeypatch.setenv("SSL_CERT_FILE", "/etc/ssl/corp.pem")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/usr/bin/{name}")
    calls = []
    monkeypatch.setattr(cli.subprocess, "run",
                        lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", ""))
    ok, why = cli._install_service(cfg)
    unit = (tmp_path / "home" / ".config/systemd/user/bmb-eval.service").read_text()
    assert ok and why == ""
    assert 'Environment="HTTPS_PROXY=http://proxy.example:3128"' in unit
    assert 'Environment="SSL_CERT_FILE=/etc/ssl/corp.pem"' in unit and "NO_PROXY" not in unit
    assert ["systemctl", "--user", "enable", "--now", "bmb-eval"] in calls


def test_reactivating_a_stalled_code_restarts_the_clock_and_rearms_the_alert(hub, provider_files):
    from bmb_eval.node import HubAPI
    store, url = hub
    _, task_dir = provider_files
    code, _ = store.add_enrollment(task_dir)
    creds = HubAPI(url).activate(code, "sealed-p")
    first = store.q("SELECT activated FROM enrollments WHERE code=?", (code,), one=True)[0]
    alerts = []
    import bmb_eval.hub as H
    orig, H.alert = H.alert, lambda store, text: alerts.append(text)
    try:
        H.check_stalled(store, first + hub_mod.STALL_S + 60)
        assert len(alerts) == 1
        time.sleep(0.01)
        HubAPI(url, creds["secret"]).activate(code, "sealed-p")       # the provider re-runs
        second = store.q("SELECT activated FROM enrollments WHERE code=?", (code,), one=True)[0]
        assert second > first
        assert store.setup_problems(second + 60) == []                # a fresh attempt isn't stalled
        H.check_stalled(store, second + hub_mod.STALL_S + 60)
        assert len(alerts) == 2                                       # the second stall is alerted too
    finally:
        H.alert = orig


def test_retire_removes_a_task_but_keeps_results(connected, tmp_path, capsys):
    cfg, store, url, agent = connected
    w = Worker(cfg)
    done = agent.submit_async("binding", EXAMPLE_BUNDLE)["submission_id"]
    w.drain()
    pending = agent.submit_async("binding", EXAMPLE_BUNDLE)["submission_id"]
    store.set_family("fam", ["binding", "other"])
    code = store.code_for("binding")
    dry = store.retire("binding")
    assert dry["codes"] == 1 and dry["unfinished submissions"] == 1 and dry["nodes left serving nothing"] == 1
    assert store.code_for("binding") == code                          # dry run changed nothing
    store.retire("binding", apply=True)
    assert store.code_for("binding") is None and store.enrollment(code) is None
    assert store.q("SELECT COUNT(*) FROM node_tasks WHERE task='binding'", one=True)[0] == 0
    assert store.q("SELECT status FROM submissions WHERE id=?", (done,), one=True)[0] == "done"
    assert store.q("SELECT 1 FROM submissions WHERE id=?", (pending,), one=True) is None
    assert store.families() == {"fam": ["other"]}
    assert store.q("SELECT COUNT(*) FROM nodes", one=True)[0] == 0
    assert list(store.state.glob("hub.db.bak-*-retire-binding"))      # backed up first


def test_enroll_tells_sealed_providers_to_rerun_their_command(hub, provider_files, capsys):
    from bmb_eval.node import HubAPI
    store, url = hub
    _, task_dir = provider_files
    state = str(store.state)
    hub_mod.main(["--state", state, "enroll", str(task_dir), "--public-url", url])
    code = store.code_for("binding")
    HubAPI(url).activate(code, "sealed-myproject")
    (task_dir / "contract.yaml").write_text((task_dir / "contract.yaml").read_text()
                                            .replace('description: ""', 'description: "v2"'))
    capsys.readouterr()
    hub_mod.main(["--state", state, "enroll", str(task_dir), "--public-url", url, "--replace"])
    out = capsys.readouterr().out
    assert "run their original command again" in out and "bmb-eval update" not in out


def test_old_clients_get_an_update_message_for_folder_packages(hub, provider_files):
    import urllib.error
    import urllib.request
    from bmb_eval.node import HubAPI
    store, url = hub
    _, task_dir = provider_files
    code, _ = store.add_enrollment(task_dir)
    assert json.loads(urllib.request.urlopen(f"{url}/v1/enroll/{code}").read())["task"] == "binding"
    (task_dir / "refs").mkdir()
    (task_dir / "refs" / "panel.txt").write_text("A\n")
    store.add_enrollment(task_dir)
    with pytest.raises(urllib.error.HTTPError) as e:                  # an old client: no format=2
        urllib.request.urlopen(f"{url}/v1/enroll/{code}")
    assert e.value.code == 409 and "install command" in json.loads(e.value.read())["error"]["message"]
    assert "refs/panel.txt" in HubAPI(url).enrollment(code)["files"]   # a current client


# ------------------------------------------- the node's own failures cost the agent nothing

def test_truncated_predictions_are_reported_as_infra_error(cfg):
    import dataclasses
    r = Runner(cfg, send_alerts=False)
    real = r.sandbox.run

    def dropping(*, entry, **kw):                              # conmon before 2.1 (Ubuntu 22.04)
        res = real(entry=entry, **kw)
        return dataclasses.replace(res, stdout=res.stdout[: len(res.stdout) // 2]) if entry == "predict" else res
    r.sandbox.run = dropping
    p = r.evaluate(TASK, pack_dir(EXAMPLE_BUNDLE))
    assert p["status"] == "error" and p["error"]["code"] == "infra_error"
    r.sandbox.run = real
    assert r.evaluate(TASK, pack_dir(EXAMPLE_BUNDLE))["status"] == "ok"



def test_a_validator_crash_on_the_models_file_is_invalid_predictions(cfg):
    # The parquet reader aborts on some malformed files (SIGABRT on 48 of 1,600 parser_attack
    # runs on Linux). That is the submission's fault: as infra_error it paged the operator and
    # failed providers' self-tests at random.
    import dataclasses
    from bmb_eval.runner import CRASH_EXITS
    r = Runner(cfg, send_alerts=False)
    real = r.sandbox.run
    for code in (-6, -11, 134, 139):            # direct, and through a container engine
        assert code in CRASH_EXITS

        def crashing(*, entry, code=code, **kw):
            res = real(entry=entry, **kw)
            return dataclasses.replace(res, exit_code=code, stdout=b"") if entry == "validate" else res
        r.sandbox.run = crashing
        p = r.evaluate(TASK, pack_dir(EXAMPLE_BUNDLE))
        assert p["status"] == "error" and p["error"]["code"] == "predictions_invalid", (code, p)
    r.sandbox.run = real
    assert r.evaluate(TASK, pack_dir(EXAMPLE_BUNDLE))["status"] == "ok"

def test_connect_refuses_a_conmon_that_drops_output(monkeypatch):
    monkeypatch.setattr(cli, "conmon_version", lambda: (2, 0, 25))   # Ubuntu 22.04's
    [p] = cli.engine_problems("podman")
    assert "2.0.25" in p and "2.1" in p and "containers.conf" in p
    assert cli.engine_problems("docker") == []
    monkeypatch.setattr(cli, "conmon_version", lambda: (2, 1, 12))
    assert cli.engine_problems("podman") == []
