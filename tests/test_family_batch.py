"""Family batching (batch_families): the jobs of one family submission on one node
share a predict sandbox, which loads the model once; validate and score stay per job.
Local backend and the FakeModal (tests/fake_modal.py): neither is isolated, so the
bundles here can count their own calls in a file."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

import fake_modal
from bmb_eval import approval, cli, modal_backend, modal_sync
from bmb_eval.client import Client
from bmb_eval.config import Config
from bmb_eval.node import BATCH_MARKER, Worker
from bmb_eval.runner import Runner, parse_frames
from bmb_eval.sandbox import Mount, check_mounts
from bmb_eval.sandbox_entry.predict import EXIT_RUNTIME, EXIT_TIMEOUT, FRAME, FRAME_MAGIC
from test_modal_backend import _modal_node, _stage_records
from test_node import _connect, _fresh_node, _provider_csv, _split_task, hub  # noqa: F401

FAMILY = "fam"

# Counts its own calls in mode["log"] (one line per load_model / predict, with the pid),
# and misbehaves on the n-th predict() of its process as mode.json says.
MODEL = '''
import json, os, time
from pathlib import Path

MODE = json.loads((Path(__file__).parent / "mode.json").read_text())
CALLS = [0]


def _log(what):
    with open(MODE["log"], "a") as f:
        f.write(f"{what} {os.getpid()}\\n")


def load_model():
    _log("load")


def predict(df):
    CALLS[0] += 1
    _log("predict")
    # As agents' models do: which task is this? (A batch once handed them the list of all.)
    _log("contract:" + json.loads(Path(os.environ["BMB_CONTRACT_FILE"]).read_text())["id"])
    if CALLS[0] == MODE.get("fail_call"):
        raise RuntimeError("this task's predict fails")
    if CALLS[0] == MODE.get("sleep_call"):
        time.sleep(MODE["sleep_s"])
    return [sum(ch in "AILMFVW" for ch in s) / len(s) for s in df["sequence"]]
'''


def _bundle(tmp_path: Path, batchable: bool = True, **mode) -> tuple[Path, Path]:
    b = tmp_path / f"bundle-{len(list(tmp_path.glob('bundle-*')))}"
    b.mkdir()
    (b / "predict.py").write_text(MODEL)
    if batchable:
        (b / BATCH_MARKER).write_text("")
    log = b.parent / f"{b.name}.log"
    (b / "mode.json").write_text(json.dumps({"log": str(log), **mode}))
    return b, log


def _calls(log: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"load": [], "predict": []}
    for line in (log.read_text().splitlines() if log.exists() else []):
        what, pid = line.split()
        out.setdefault(what, []).append(pid)
    return out


def _family(hub, tmp_path, node_cfg: Path | None = None, n: int = 3, batch: bool = True,  # noqa: F811
            predict_timeout_s: int = 1800):
    """n datasets of one family, all served by one node."""
    store, url = hub
    node = node_cfg.parent if node_cfg else _fresh_node(tmp_path / "node")
    data = _provider_csv(tmp_path / "data.csv", 1)
    tasks = [f"fam_{i}" for i in range(n)]
    for i, t in enumerate(tasks):
        d = _split_task(tmp_path / f"author{i}", name=t)
        c = d / "contract.yaml"
        c.write_text(c.read_text().replace("predict_timeout_s: 1800", f"predict_timeout_s: {predict_timeout_s}"))
        code, _ = store.add_enrollment(d)
        cfg = _connect(node, url, code, data)
    if batch:
        cfg.raw["batch_families"] = True
        cfg.save_raw()
        cfg = Config.load(cfg.path)
        approval.approve(cfg)
    store.set_family(FAMILY, tasks)
    return cfg, store, Client(url, store.add_agent("agent-fam")), tasks


def _count_stages(w: Worker) -> list[tuple[str, bool]]:
    """(entry, batched) of every sandbox the node's runner starts."""
    seen, real, lock = [], w.runner.sandbox.run, threading.Lock()

    def run(**kw):
        with lock:
            seen.append((kw["entry"], "BMB_BATCH" in kw["env"]))
        return real(**kw)
    w.runner.sandbox.run = run
    return seen


def _results(store, gid: str) -> dict[str, dict]:
    rows = store.q("SELECT task, result FROM submissions WHERE group_id=?", (gid,))
    return {t: json.loads(r) for t, r in rows}


def _events(cfg, runs) -> list[str]:
    """The audit events about these runs (connect's self-test records runs too)."""
    out = []
    for line in (cfg.state_dir / "audit.jsonl").read_text().splitlines():
        e = json.loads(line)
        if e.get("run_id") in runs:
            out.append(e["event"])
    return out


def _internal(cfg, run_id: str) -> dict:
    return json.loads((cfg.state_dir / "runs" / run_id / "internal.json").read_text())


# ------------------------------------------------------------- local backend

def test_family_predicts_once_and_scores_each_dataset(hub, tmp_path):  # noqa: F811
    cfg, store, agent, tasks = _family(hub, tmp_path)
    w = Worker(cfg)
    stages = _count_stages(w)
    b, log = _bundle(tmp_path)
    gid = agent.submit_async(FAMILY, b)["submission_id"]
    w.drain()
    res = _results(store, gid)
    assert sorted(res) == tasks and all(r["status"] == "ok" for r in res.values()), res
    assert sorted(stages) == [("predict", True)] + [("score", False)] * 3 + [("validate", False)] * 3
    calls = _calls(log)
    assert len(calls["load"]) == 1 and len(set(calls["predict"])) == 1 and len(calls["predict"]) == 3
    assert sorted(k for k in calls if k.startswith("contract:")) == [f"contract:{t}" for t in tasks]
    # Each job keeps its own record, quota count and audit entry.
    for t, r in res.items():
        rec = _internal(cfg, r["run_id"])
        assert rec["batched"]["size"] == 3 and rec["task"] == t and rec["exported"] == r
        assert w.store.count_total("agent-fam", t) == 1
    assert _events(cfg, [r["run_id"] for r in res.values()]).count("run") == 3
    assert agent.status(gid)["result"]["datasets_scored"] == 3

    # The same model alone on each dataset scores the same: batching changes nothing but cost.
    cfg.raw["batch_families"] = False
    cfg.save_raw()
    cfg = Config.load(cfg.path)
    approval.approve(cfg)
    w = Worker(cfg)
    stages = _count_stages(w)
    gid2 = agent.submit_async(FAMILY, b)["submission_id"]
    w.drain()
    res2 = _results(store, gid2)
    assert sorted(stages) == [("predict", False)] * 3 + [("score", False)] * 3 + [("validate", False)] * 3
    assert {t: r["metrics"] for t, r in res2.items()} == {t: r["metrics"] for t, r in res.items()}
    assert len(_calls(log)["load"]) == 1 + 3


def test_a_bundle_that_did_not_opt_in_runs_alone(hub, tmp_path):  # noqa: F811
    """Batching is the model's choice (a BMB_BATCHABLE file): one that reads its contract only
    at import would predict every task of a batch as the first."""
    cfg, store, agent, tasks = _family(hub, tmp_path)
    w = Worker(cfg)
    stages = _count_stages(w)
    b, log = _bundle(tmp_path, batchable=False)
    gid = agent.submit_async(FAMILY, b)["submission_id"]
    w.drain()
    assert all(r["status"] == "ok" for r in _results(store, gid).values())
    assert sorted(stages) == [("predict", False)] * 3 + [("score", False)] * 3 + [("validate", False)] * 3
    assert len(set(_calls(log)["predict"])) == 3


def test_node_that_did_not_opt_in_never_claims_siblings(hub, tmp_path):  # noqa: F811
    cfg, store, agent, tasks = _family(hub, tmp_path, batch=False)
    w = Worker(cfg)
    asked = []
    real = w.api.next
    w.api.next = lambda **only: (asked.append(only), real(**only))[1]
    b, log = _bundle(tmp_path)
    gid = agent.submit_async(FAMILY, b)["submission_id"]
    w.drain()
    assert all(r["status"] == "ok" for r in _results(store, gid).values())
    assert asked and not any(asked)                      # only plain claims
    assert len(set(_calls(log)["predict"])) == 3


def test_a_failing_task_does_not_fail_the_others(hub, tmp_path):  # noqa: F811
    cfg, store, agent, tasks = _family(hub, tmp_path)
    w = Worker(cfg)
    b, log = _bundle(tmp_path, fail_call=2)
    gid = agent.submit_async(FAMILY, b)["submission_id"]
    w.drain()
    codes = sorted(r["status"] if r["status"] == "ok" else r["error"]["code"]
                   for r in _results(store, gid).values())
    assert codes == ["ok", "ok", "predict_runtime_error"]
    assert len(set(_calls(log)["predict"])) == 1           # one process, not re-run


def test_a_task_out_of_time_in_a_batch_is_that_tasks_timeout(hub, tmp_path):  # noqa: F811
    cfg, store, agent, tasks = _family(hub, tmp_path, predict_timeout_s=4)
    w = Worker(cfg)
    b, log = _bundle(tmp_path, sleep_call=2, sleep_s=30)
    gid = agent.submit_async(FAMILY, b)["submission_id"]
    w.drain()
    codes = sorted(r["status"] if r["status"] == "ok" else r["error"]["code"]
                   for r in _results(store, gid).values())
    assert codes == ["ok", "ok", "predict_timeout"]
    assert len(set(_calls(log)["predict"])) == 1


def test_a_batch_cut_off_runs_the_rest_alone_counted_once(hub, tmp_path):  # noqa: F811
    cfg, store, agent, tasks = _family(hub, tmp_path)
    cfg.batch_max_predict_s = 3                          # the whole batch's limit (the config's floor is 60)
    w = Worker(cfg)
    stages = _count_stages(w)
    b, log = _bundle(tmp_path, sleep_call=2, sleep_s=30)  # the 2nd task outlasts the batch
    gid = agent.submit_async(FAMILY, b)["submission_id"]
    w.drain(max_s=120)
    res = _results(store, gid)
    assert all(r["status"] == "ok" for r in res.values()), res
    assert sorted(stages).count(("predict", True)) == 1 and sorted(stages).count(("predict", False)) == 2
    batched = [t for t, r in res.items() if "batched" in _internal(cfg, r["run_id"])]
    assert len(batched) == 1                             # finished before the batch was cut off
    for t, r in res.items():
        assert w.store.count_total("agent-fam", t) == 1  # each job ran and counted once
    events = _events(cfg, [r["run_id"] for r in res.values()])
    assert events.count("job_unbatched") == 2 and events.count("run") == 3


def test_family_finals_batch_on_the_test_half(hub, tmp_path):  # noqa: F811
    cfg, store, agent, tasks = _family(hub, tmp_path)
    w = Worker(cfg)
    b, log = _bundle(tmp_path)
    gid = agent.submit_async(FAMILY, b)["submission_id"]
    w.drain()
    stages = _count_stages(w)
    fids = store.request_finals(gid)
    w.drain()
    finals = [json.loads(store.q("SELECT result FROM submissions WHERE id=?", (f,), one=True)[0]) for f in fids]
    assert all(r["status"] == "ok" and r["split"] == "test" for r in finals)
    assert stages.count(("predict", True)) == 1
    assert all(w.store.count_final(t) == 1 for t in tasks)


# ---------------------------------------------------- what may be batched

def test_hub_hands_out_only_the_named_siblings(hub, tmp_path):  # noqa: F811
    cfg, store, agent, tasks = _family(hub, tmp_path)
    w = Worker(cfg)
    b, _ = _bundle(tmp_path)
    other, _ = _bundle(tmp_path)
    agent.submit_async(FAMILY, b)
    agent.submit_async("fam_1", other)                   # another bundle: never a sibling
    first = w.api.next()
    assert first["siblings"] == 2 and first["group_id"]
    only = dict(bundle_id=first["bundle_id"], split="validation", submitter="agent-fam")
    assert w.api.next(**only, tasks=["nope"]) is None
    assert w.api.next(**{**only, "split": "test"}, tasks=tasks) is None
    got = [w.api.next(**only, tasks=tasks) for _ in range(3)]
    assert got[2] is None and {g["bundle_id"] for g in got[:2]} == {first["bundle_id"]}
    assert w.api.next()["bundle_id"] != first["bundle_id"]


def _job(sid, task, bundle="b1", split="validation", submitter="a"):
    return {"submission_id": sid, "task": task, "submitter": submitter, "bundle_id": bundle, "split": split}


def test_batches_never_mix_bundles_splits_submitters_or_gpus(tmp_path):
    (tmp_path / "config.yaml").write_text(json.dumps({
        "node_name": "c", "sandbox": {"backend": "local", "i_understand_local_backend_is_not_isolated": True},
        "batch_families": True, "hub": {"url": "http://unused", "secret": "x"}, "tasks": {}}))
    w = Worker(Config.load(tmp_path / "config.yaml"))
    batches, alone = [], []
    w._run_batch = lambda members, downloads: batches.append([m[0] for m in members])
    w._dispatch = lambda sid, *a: alone.append(sid)
    cpu, gpu = (1.0, 1, 0), (1.0, 1, 1)
    w._start([(_job("1", "a"), cpu), (_job("2", "b"), cpu), (_job("3", "c", bundle="b2"), cpu),
              (_job("4", "d", split="test"), cpu), (_job("5", "e", submitter="z"), cpu),
              (_job("6", "f"), gpu), (_job("7", "a"), cpu)])
    w.wait_jobs(5)
    assert batches == [["1", "2"]] and alone == ["3", "4", "5", "6", "7"]
    batches.clear(), alone.clear()
    w._start([(_job("8", "a"), cpu), (_job("9", "b", bundle="b2"), cpu)])
    assert batches == [] and alone == ["8", "9"]
    w.cfg.batch_families = False
    w._start([(_job("10", "a"), cpu), (_job("11", "b"), cpu)])
    assert batches == [] and alone == ["8", "9", "10", "11"]


def test_runner_refuses_a_batch_that_would_mix(hub, tmp_path):  # noqa: F811
    cfg, store, agent, tasks = _family(hub, tmp_path)
    r = Runner(cfg, send_alerts=False)
    b, _ = _bundle(tmp_path)
    assert r.predict_batch([("x", "fam_0"), ("y", "fam_0")], b) == {}          # one task twice
    assert r.predict_batch([("x", "fam_0")], b) == {}                           # nothing to share
    cfg.batch_families = False
    with pytest.raises(PermissionError):
        r.predict_batch([("x", "fam_0"), ("y", "fam_1")], b)


def test_only_a_batched_predict_on_an_opted_in_node_mounts_several_tasks(tmp_path):
    two = [Mount("BMB_INPUTS_DIR_0", tmp_path, ""), Mount("BMB_INPUTS_DIR_1", tmp_path, "")]
    check_mounts("predict", two, {"BMB_BATCH": "2"}, batch_ok=True)
    with pytest.raises(PermissionError):
        check_mounts("predict", two, {}, batch_ok=True)                 # not batched
    with pytest.raises(PermissionError):
        check_mounts("predict", two, {"BMB_BATCH": "2"}, batch_ok=False)  # node didn't opt in
    with pytest.raises(PermissionError):
        check_mounts("score", two, {"BMB_BATCH": "2"}, batch_ok=True)
    with pytest.raises(PermissionError):
        check_mounts("predict", two + [Mount("BMB_ANSWERS", tmp_path, "")], {"BMB_BATCH": "2"}, batch_ok=True)
    with pytest.raises(PermissionError):
        check_mounts("predict", [Mount("BMB_INPUTS_DIR", tmp_path, ""), Mount("BMB_INPUTS", tmp_path, "")],
                     {}, batch_ok=True)


def test_frames_are_parsed_only_while_whole():
    def frame(status, data=b""):
        return FRAME.pack(FRAME_MAGIC, status, len(data)) + data
    ok = frame(0, b"x" * 10 + b"PAR1")
    assert parse_frames(ok + frame(EXIT_RUNTIME) + frame(EXIT_TIMEOUT), [100] * 3) == [
        (0, b"x" * 10 + b"PAR1"), (EXIT_RUNTIME, b""), (EXIT_TIMEOUT, b"")]
    assert parse_frames(ok + ok[:-3], [100, 100]) == [(0, ok[FRAME.size:])]          # cut off
    assert parse_frames(ok + b"JUNK" + ok[4:], [100, 100]) == [(0, ok[FRAME.size:])]  # bad magic
    assert parse_frames(ok, [5]) == []                                                # over its cap
    assert parse_frames(ok + ok + ok, [100, 100]) == [(0, ok[FRAME.size:])] * 2      # no extra tasks


def test_sealed_node_cannot_batch(tmp_path):
    (tmp_path / "config.yaml").write_text(json.dumps({
        "node_name": "s", "sandbox": {"backend": "bwrap"}, "batch_families": True, "tasks": {}}))
    with pytest.raises(SystemExit, match="batch_families"):
        Config.load(tmp_path / "config.yaml")


def test_turning_batching_on_needs_approval(hub, tmp_path):  # noqa: F811
    cfg, *_ = _family(hub, tmp_path, n=1, batch=False)
    assert not approval.diff(cfg)
    cfg.raw["batch_families"] = True
    cfg.save_raw()
    assert any("batch_families" in d for d in approval.diff(Config.load(cfg.path)))


# ------------------------------------------------------------------- Modal

@pytest.fixture
def fm(tmp_path, monkeypatch):
    monkeypatch.setattr(modal_backend, "EXIT_POLLS", 1)
    monkeypatch.setattr(modal_backend, "RETRY_BACKOFF_S", (0,))
    monkeypatch.setattr(modal_sync, "LIST_BACKOFF_S", 0.0)
    monkeypatch.setattr(cli, "_selftest", lambda *a, **k: 0)   # the attacks need real isolation
    return fake_modal.install(monkeypatch, tmp_path / "modal")


def test_modal_family_takes_one_predict_container(hub, tmp_path, fm):  # noqa: F811
    cfg, store, agent, tasks = _family(hub, tmp_path, node_cfg=_modal_node(tmp_path / "mnode"))
    w = Worker(cfg)
    b, log = _bundle(tmp_path)
    gid = agent.submit_async(FAMILY, b)["submission_id"]
    w.drain()
    res = _results(store, gid)
    assert all(r["status"] == "ok" for r in res.values()), res
    predicts = _stage_records(fm, "predict")
    assert len(predicts) == 1 and len(_stage_records(fm, "validate")) == len(_stage_records(fm, "score")) == 3
    vols = predicts[0]["volumes"]
    inputs = sorted(p for p in vols if p.startswith("/data/inputs/"))
    assert inputs == ["/data/inputs/0/.vol", "/data/inputs/1/.vol", "/data/inputs/2/.vol"]
    assert sorted(vols[p][0].rsplit("-", 2)[-2] for p in inputs) == tasks       # each task's own
    assert all(ro for _, ro in vols.values())
    assert not [n for n, _ in vols.values() if n.endswith(("-ans", "-task", "-blobs", "-meta", "-test"))]
    calls = _calls(log)
    assert len(calls["load"]) == 1 and len(calls["predict"]) == 3


def test_modal_plan_allows_several_tasks_only_in_a_batch(hub, tmp_path, fm):  # noqa: F811
    cfg, *_ = _family(hub, tmp_path, node_cfg=_modal_node(tmp_path / "mnode"), batch=False)
    sb = Runner(cfg).sandbox
    bundle = cfg.blob_dir.parent / "bundles" / "x"
    bundle.mkdir(parents=True)
    t0, t1 = cfg.tasks["fam_0"], cfg.tasks["fam_1"]
    base = [Mount("BMB_BUNDLE", bundle, "/bundle")]
    two = [Mount("BMB_INPUTS_DIR_0", t0.inputs_dir, ""), Mount("BMB_INPUTS_DIR_1", t1.inputs_dir, "")]
    with pytest.raises(PermissionError):                         # this node didn't opt in
        sb.plan("predict", base + two, batched=True)
    with pytest.raises(PermissionError):                         # unbatched: one task only
        sb.plan("predict", base + [Mount("BMB_INPUTS_DIR", t0.inputs_dir, ""),
                                   Mount("BMB_INPUTS", t1.inputs, "")])
    sb.cfg.batch_families = True
    p = sb.plan("predict", base + two, batched=True)
    assert p.volumes["/data/inputs/1/.vol"] == modal_backend.vol_name(cfg.modal.volume_prefix, "fam_1", "val")
    assert p.env["BMB_INPUTS_DIR_1"] == "/data/inputs/1/.vol"
    for bad in (two + [Mount("BMB_ANSWERS", t0.answers, "")],                       # answers
                [two[0], Mount("BMB_INPUTS_DIR_1", t1.test_inputs_dir, "")],         # two halves
                [two[0], Mount("BMB_INPUTS_DIR_1", t0.inputs_dir, "")],              # a task twice
                [two[0], Mount("BMB_INPUTS_DIR", t1.inputs_dir, "")]):               # unnumbered
        with pytest.raises(PermissionError):
            sb.plan("predict", base + bad, batched=True)
    with pytest.raises(PermissionError):
        sb.plan("validate", two, batched=True)
