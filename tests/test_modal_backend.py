"""The Modal backend (sandbox.backend: modal) against an in-process FakeModal: volumes
are folders and the real launcher runs as a local subprocess (NOT isolated). These
cover what the node and the backend decide: which volumes each stage gets, output
caps, timeouts, retries and requeues, cleanup, the data manifest, GPU admission.
Isolation itself is checked by `bmb-eval selftest` on a real Modal node."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
import yaml

import fake_modal
from bmb_eval import approval, cli, modal_backend, modal_sync
from bmb_eval.bundle import pack_dir
from bmb_eval.client import Client
from bmb_eval.config import Config
from bmb_eval.node import MAX_INFRA_REQUEUES, Worker
from bmb_eval.redteam import ATTACKS, bundle_bytes
from bmb_eval.runner import Runner
from bmb_eval.sandbox import Mount, SandboxLost
from test_node import EXAMPLE_BUNDLE, _connect, _split_task, hub, provider_files  # noqa: F401

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def fm(tmp_path, monkeypatch):
    monkeypatch.setattr(modal_backend, "EXIT_POLLS", 1)
    monkeypatch.setattr(modal_backend, "RETRY_BACKOFF_S", (0,))
    monkeypatch.setattr(modal_sync, "LIST_BACKOFF_S", 0.0)
    return fake_modal.install(monkeypatch, tmp_path / "modal")


def _modal_node(root: Path, **modal) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.yaml").write_text(yaml.safe_dump({
        "node_name": "mnode", "service_name": "bmb-eval-modal",
        "sandbox": {"backend": "modal"},
        "modal": {"environment": "bmb-eval", "app": "bmb-eval-sandbox", "max_gpus": 64, "max_jobs": 4,
                  **modal},
        "tasks": {}}))
    modal_backend.deploy(Config.load(root / "config.yaml"), root=REPO)
    return root / "config.yaml"


@pytest.fixture
def modal_connected(hub, provider_files, tmp_path, fm, monkeypatch):  # noqa: F811
    """A provider connected with the Modal backend, on a split task, through the real hub."""
    store, url = hub
    data, _ = provider_files
    cfg_path = _modal_node(tmp_path / "mnode")
    monkeypatch.setattr(cli, "_selftest", lambda *a, **k: 0)   # the attacks need real isolation
    code, _ = store.add_enrollment(_split_task(tmp_path, final_limits="gpus: 8"))
    cfg = _connect(cfg_path.parent, url, code, data)
    return cfg, store, url, Client(url, store.add_agent("agent-modal")), fm


def _attack(name):
    a = next(a for a in ATTACKS if a.name == name)
    return bundle_bytes(a), a.limits


def _stage_records(fm, stage):
    return [r for r in fm.created if fm.stage_of(r) == stage]


# ------------------------------------------------------------- full flow

def test_connect_syncs_and_an_agent_gets_scores(modal_connected):
    cfg, store, url, agent, fm = modal_connected
    assert cfg.backend == "modal" and cfg.max_parallel == 4 and not approval.diff(cfg)
    fp = json.loads(approval.lock_path(cfg).read_text())
    assert fp["sandbox"]["modal"]["environment"] == "bmb-eval"
    assert fp["sandbox"]["image_id"] == "im-fake0001"
    assert fp["sandbox"]["modal"]["manifest"] == modal_sync.local_manifest_hash(cfg) != "missing"
    w = Worker(cfg)
    assert w.accepting()
    w.step()
    sid = agent.submit_async("split_task", EXAMPLE_BUNDLE)["submission_id"]
    w.drain()
    res = agent.status(sid)
    assert res["status"] == "done" and res["result"]["status"] == "ok", res
    assert res["result"]["metrics"]["spearman"] > 0.3
    # Every stage ran without network, with this node's tags; job volumes are gone afterwards.
    assert fm.created and all(r["block_network"] for r in fm.created)
    assert all(r["tags"] == {"bmb-eval": "1", "node": cfg.hub["node_id"]} for r in fm.created)
    assert not [p for p in (fm.root / "volumes" / "bmb-eval").iterdir() if "-job-" in p.name]


def test_predict_never_gets_answers_or_the_other_half(modal_connected):
    cfg, store, url, agent, fm = modal_connected
    w = Worker(cfg)
    sid = agent.submit_async("split_task", EXAMPLE_BUNDLE)["submission_id"]
    w.drain()
    fid = store.request_final(sid)
    w.drain()
    final = json.loads(store.q("SELECT result FROM submissions WHERE id=?", (fid,), one=True)[0])
    assert final["status"] == "ok" and final["split"] == "test"
    predicts = _stage_records(fm, "predict")
    assert len(predicts) == 2
    halves = []
    for r in predicts:
        names = [n for n, _ in r["volumes"].values()]
        assert all(ro for _, ro in r["volumes"].values())             # all read-only
        assert not [n for n in names if n.endswith(("-ans", "-task", "-blobs", "-meta"))]
        halves.append(sorted(n.rsplit("-", 1)[1] for n in names if "-job-" not in n))
    assert halves == [["val"], ["test"]]
    assert predicts[0]["gpu"] is None and predicts[1]["gpu"] == "H100:8"   # finals: 8 GPUs
    score = _stage_records(fm, "score")
    assert any(n.endswith("-ans") for n, _ in score[0]["volumes"].values())
    # The stager alone mounts the blob volume, and writes only the job's volume.
    for r in _stage_records(fm, "stage"):
        assert r["volumes"]["/blobs"][1] is True and "-job-" in r["volumes"]["/out"][0]


def test_stage_allowlist_refuses_before_anything_starts(modal_connected):
    cfg, store, url, agent, fm = modal_connected
    t = cfg.tasks["split_task"]
    sb = Runner(cfg).sandbox
    bundle = cfg.blob_dir.parent / "bundles" / "x"
    bundle.mkdir(parents=True)
    contract = Mount("BMB_CONTRACT_FILE", t.dir / "contract.yaml", "/contract.json")
    base = [Mount("BMB_BUNDLE", bundle, "/bundle"), contract]
    ok = sb.plan("predict", base + [Mount("BMB_INPUTS_DIR", t.inputs_dir, "/data/inputs")])
    assert ok.kinds == {"bundle", "contract", "inputs_validation"}
    for extra in ([Mount("BMB_INPUTS_DIR", t.inputs_dir, ""), Mount("BMB_ANSWERS", t.answers, "")],
                  [Mount("BMB_INPUTS_DIR", t.inputs_dir, ""), Mount("BMB_X", t.test_inputs_dir, "")],
                  [Mount("BMB_TASK", t.dir, "")],
                  [Mount("BMB_X", Path("/etc"), "")]):
        with pytest.raises(PermissionError):
            sb.plan("predict", base + extra)
    with pytest.raises(PermissionError):
        sb.plan("validate", [Mount("BMB_INPUTS", t.inputs, ""), Mount("BMB_ANSWERS", t.answers, "")])
    n = len(fm.created)
    r = Runner(cfg, send_alerts=False)
    real = r.sandbox.plan
    r.sandbox.plan = lambda entry, mounts: real(entry, mounts + [Mount("BMB_ANSWERS", t.answers, "")])
    p = r.evaluate("split_task", pack_dir(EXAMPLE_BUNDLE))
    assert p["error"]["code"] == "infra_error" and len(fm.created) == n      # nothing ran


# ------------------------------------------------------- the model's outcomes

def test_stdout_cap_stops_the_stage(modal_connected):
    cfg, *_ = modal_connected
    b, lim = _attack("flood_output")
    p = Runner(cfg, send_alerts=False).evaluate("split_task", b, limits_override=lim)
    assert p["error"]["code"] == "predictions_too_large"


def test_timeout_is_the_models(modal_connected):
    cfg, *_, fm = modal_connected
    b, lim = _attack("timeout")
    p = Runner(cfg, send_alerts=False).evaluate("split_task", b, limits_override=lim)
    assert p["error"]["code"] == "predict_timeout"
    assert len(_stage_records(fm, "predict")) == 1                 # not retried


def test_modal_error_is_retried_but_the_models_exit_is_not(modal_connected):
    cfg, *_, fm = modal_connected
    fm.fail_creates = 1                                            # e.g. no capacity, once
    p = Runner(cfg, send_alerts=False).evaluate("split_task", pack_dir(EXAMPLE_BUNDLE))
    assert p["status"] == "ok"
    events = [json.loads(line) for line in (cfg.state_dir / "audit.jsonl").read_text().splitlines()]
    assert any(e["event"] == "modal_infra_failure" for e in events)
    before = len(_stage_records(fm, "predict"))
    b, lim = _attack("leak_in_error")                              # predict() raises: exit 11
    p = Runner(cfg, send_alerts=False).evaluate("split_task", b, limits_override=lim)
    assert p["error"]["code"] == "predict_runtime_error"
    assert len(_stage_records(fm, "predict")) == before + 1


def test_lost_sandbox_after_retries_raises(modal_connected):
    cfg, *_, fm = modal_connected
    fm.lose_execs = 3                                              # gone, no exit code, every try
    with pytest.raises(SandboxLost):
        Runner(cfg, send_alerts=False).evaluate("split_task", pack_dir(EXAMPLE_BUNDLE))


def test_infra_failures_requeue_the_job_uncounted_then_alert(modal_connected, monkeypatch):
    cfg, store, url, agent, fm = modal_connected
    sent = []
    from bmb_eval import alerts
    monkeypatch.setattr(alerts, "send", lambda cfg, kind, summary, **f: sent.append((kind, summary)))
    cfg.infra_failure_threshold = 2
    w = Worker(cfg)
    w.step()
    sid = agent.submit_async("split_task", EXAMPLE_BUNDLE)["submission_id"]
    fm.fail_creates = 10 ** 6                                      # Modal is down
    for n in range(MAX_INFRA_REQUEUES):
        w._infra_pause_until = 0                                    # (the back-off has passed)
        w.drain(max_s=60)
        # Handed back, uncounted: no quota used, the hub has it queued again.
        assert w.store.count_today("agent-modal", "split_task") == 0 and not w.store.has(sid)
        assert store.q("SELECT status FROM submissions WHERE id=?", (sid,), one=True)[0] == "queued"
        assert agent.status(sid)["status"] == "pending"
        assert w._infra_pause_until > time.time()                  # no new work for a while
        w.drain(max_s=5)
        assert not w.store.has(sid)
    assert [k for k, _ in sent] == ["infra"]                        # alerted at the threshold
    fm.fail_creates = 0                                             # Modal is back
    w._infra_pause_until = 0
    w.drain(max_s=60)
    assert agent.status(sid)["result"]["status"] == "ok"
    assert w.store.count_today("agent-modal", "split_task") == 1


def test_requeues_are_bounded_per_job(modal_connected):
    cfg, store, url, agent, fm = modal_connected
    w = Worker(cfg)
    w.step()
    sid = agent.submit_async("split_task", EXAMPLE_BUNDLE)["submission_id"]
    fm.fail_creates = 10 ** 6
    for _ in range(MAX_INFRA_REQUEUES + 1):
        w._infra_pause_until = 0
        w.drain(max_s=60)
    res = agent.status(sid)
    assert res["status"] == "done" and res["result"]["error"]["code"] == "infra_error"
    assert w.store.count_today("agent-modal", "split_task") == 1    # now it counts


# ------------------------------------------------------------- operations

def test_kill_all_by_tag(modal_connected):
    cfg, *_, fm = modal_connected
    sb = Runner(cfg).sandbox
    mine = [fm.Sandbox.create("sleep", "9", tags=sb.tags()) for _ in range(2)]
    other = fm.Sandbox.create("sleep", "9", tags={"bmb-eval": "1", "node": "someone-else"})
    assert sb.kill_all() == 2
    assert all(s.poll() is not None for s in mine) and other.poll() is None


def test_manifest_drift_stops_accepting(modal_connected):
    cfg, *_, fm = modal_connected
    w = Worker(cfg)
    assert w.accepting()
    meta = fm.volume_dir("bmb-eval", modal_backend.vol_name(cfg.modal.volume_prefix, "meta")) / "MANIFEST.json"
    m = json.loads(meta.read_text())
    m["tasks"]["split_task"]["answers"] = "0" * 64
    meta.write_text(json.dumps(m))
    w._accepting, w._modal_next_check = (0.0, False), 0.0
    assert not w.accepting()
    assert modal_sync.drift(cfg)
    # Restored by a sync of the approved data.
    modal_sync.sync(cfg, say=lambda *a: None)
    w._accepting, w._modal_next_check = (0.0, False), 0.0
    assert w.accepting()


def test_volume_changed_behind_the_manifest_is_found(modal_connected):
    cfg, *_, fm = modal_connected
    assert modal_sync.verify(cfg) == []
    ans = fm.volume_dir("bmb-eval", modal_backend.vol_name(cfg.modal.volume_prefix, "split_task", "ans"))
    (ans / "extra.csv").write_text("x")
    assert any("1 unexpected" in p for p in modal_sync.verify(cfg, max_age_s=0))
    w = Worker(cfg)
    assert not w.accepting()


def test_no_final_hardware_missing_on_modal(modal_connected):
    cfg, store, url, agent, fm = modal_connected
    assert len(Runner(cfg).sandbox.gpus_available()) == 8
    w = Worker(cfg)
    sid = agent.submit_async("split_task", EXAMPLE_BUNDLE)["submission_id"]
    w.drain()
    fid = store.request_final(sid)
    w.drain()
    r = json.loads(store.q("SELECT result FROM submissions WHERE id=?", (fid,), one=True)[0])
    assert r["status"] == "ok"
    assert cli.gpu_problems(cfg.tasks["split_task"].contract.limits_for("test"), cfg) == []


def test_gpu_jobs_share_the_modal_budget(tmp_path, fm):
    cfg = Config.load(_modal_node(tmp_path / "n", max_gpus=64))
    cfg.hub = {"url": "http://unused", "secret": "x", "node_id": "n"}
    w = Worker(cfg)
    assert w.capacity[2] == 64
    for i in range(8):                                   # eight 8-GPU finals: 64 GPUs
        assert w._fits((4.0, 64 << 30, 8))
        w._jobs[f"j{i}"] = (threading.Thread(), 4.0, 64 << 30, 8)
    assert not w._fits((4.0, 1 << 30, 1))                # the 65th GPU waits
    assert w._fits((4.0, 1 << 30, 0))                    # CPU-only work still runs


def test_container_gpu_jobs_still_never_share(tmp_path):
    (tmp_path / "config.yaml").write_text(json.dumps({
        "node_name": "c", "sandbox": {"backend": "local", "i_understand_local_backend_is_not_isolated": True},
        "max_parallel": 4, "hub": {"url": "http://unused", "secret": "x"}, "tasks": {}}))
    w = Worker(Config.load(tmp_path / "config.yaml"))
    w.capacity = (64.0, 1 << 40, 8)
    w._jobs["a"] = (threading.Thread(), 1.0, 1, 1)
    assert not w._fits((1.0, 1, 1))


def test_shared_file_is_stored_once_and_linked_into_inputs(tmp_path, fm, monkeypatch):
    """A large file both halves see (the genome) goes into its own volume, mounted only
    into predict, and appears in the model's inputs folder at its usual path."""
    from bmb_eval.modal_entry import mirror_inputs
    monkeypatch.setattr(modal_sync, "SHARED_MIN_BYTES", 10)
    cfg = Config.load(_modal_node(tmp_path / "n"))
    priv = tmp_path / "n" / "private_data" / "g"
    for half in ("inputs_validation", "inputs_test"):
        (priv / half / "ref").mkdir(parents=True)
        (priv / half / "ref" / "genome.fa").write_text(">chr1\n" + "ACGT" * 100)
        (priv / half / "ref" / "small.txt").write_text("hi")
        (priv / half / "samples.parquet").write_text(half)
    (priv / "answers.parquet").write_text("a")
    task = tmp_path / "n" / "tasks" / "g"
    import shutil
    shutil.copytree(_split_task(tmp_path, name="g"), task)
    raw = yaml.safe_load(cfg.path.read_text())
    raw["tasks"] = {"g": {"dir": "tasks/g", "inputs": str(priv / "inputs_validation" / "samples.parquet"),
                          "answers": str(priv / "answers.parquet"),
                          "inputs_dir": str(priv / "inputs_validation"),
                          "test_inputs_dir": str(priv / "inputs_test")}}
    cfg.path.write_text(yaml.safe_dump(raw))
    cfg = Config.load(cfg.path)
    manifest = modal_sync.sync(cfg, say=lambda *a: None)
    shared = [v for v in manifest["volumes"] if "-sh-" in v]
    assert len(shared) == 1
    for half in ("val", "test"):
        files = manifest["volumes"][modal_backend.vol_name(cfg.modal.volume_prefix, "g", half)]
        assert "ref/genome.fa" not in files and "ref/small.txt" in files
    sb = Runner(cfg).sandbox
    bundle = cfg.blob_dir.parent / "bundles" / "x"
    bundle.mkdir(parents=True)
    p = sb.plan("predict", [Mount("BMB_BUNDLE", bundle, ""), Mount("BMB_INPUTS_DIR", cfg.tasks["g"].test_inputs_dir, "")])
    assert p.mirror and p.env["BMB_INPUTS_DIR"] == "/data/inputs"
    assert sorted(p.volumes) == ["/data/inputs/.shared/" + shared[0].rsplit("-", 1)[1], "/data/inputs/.vol"]
    v = sb.plan("validate", [Mount("BMB_INPUTS", cfg.tasks["g"].inputs, "")])
    assert list(v.volumes) == ["/data/inputs/.vol"]                # validate gets no shared volume
    # What the launcher builds in the sandbox from those mounts:
    root = tmp_path / "sbx"
    vol = fm.volume_dir("bmb-eval", modal_backend.vol_name(cfg.modal.volume_prefix, "g", "test"))
    (root / "data" / "inputs" / ".shared").mkdir(parents=True)
    (root / "data" / "inputs" / ".vol").symlink_to(vol)
    (root / "data" / "inputs" / ".shared" / shared[0].rsplit("-", 1)[1]).symlink_to(fm.volume_dir("bmb-eval", shared[0]))
    mirror_inputs({"src": str(root / "data/inputs/.vol"), "dst": str(root / "data/inputs"),
                   "shared": str(root / "data/inputs/.shared")})
    inp = root / "data" / "inputs"
    assert (inp / "ref" / "genome.fa").read_text().startswith(">chr1")
    assert (inp / "ref" / "small.txt").read_text() == "hi"
    assert (inp / "samples.parquet").read_text() == "inputs_test"
    assert not (inp / ".bmb_links.json").exists()


def test_image_build_context_holds_no_private_files(tmp_path, fm):
    root = tmp_path / "install"
    for f in ("sandbox/modal/Dockerfile", "src/bmb_eval/node.py", "private_data/t/answers.parquet",
              "config.yaml", "state/node.db", "src/bmb_eval/__pycache__/node.cpython-312.pyc"):
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text("x")
    ignore = modal_backend._context_filter(root)
    kept = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and not ignore(p))
    assert kept == ["sandbox/modal/Dockerfile", "src/bmb_eval/node.py"]


def test_stale_modal_image_stops_the_node(modal_connected, monkeypatch):
    cfg, *_ = modal_connected
    assert modal_backend.stale_message(cfg) is None
    monkeypatch.setattr(modal_backend, "image_stamp", lambda root: "different")
    assert "modal-deploy" in modal_backend.stale_message(cfg)
    w = Worker(cfg)
    monkeypatch.setattr("bmb_eval.alerts.send", lambda *a, **k: None)
    assert not w.accepting()


def test_second_node_has_its_own_service_unit(modal_connected, tmp_path, monkeypatch):
    import subprocess
    cfg, *_ = modal_connected
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MODAL_PROFILE", "biomodelbench")
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/usr/bin/{name}")
    calls = []
    monkeypatch.setattr(cli.subprocess, "run",
                        lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", ""))
    assert cli._install_service(cfg) == (True, "")
    unit = (tmp_path / "home" / ".config/systemd/user/bmb-eval-modal.service").read_text()
    assert f"--config {cfg.path} run" in unit and 'Environment="MODAL_PROFILE=biomodelbench"' in unit
    assert ["systemctl", "--user", "enable", "--now", "bmb-eval-modal"] in calls
    assert not (tmp_path / "home" / ".config/systemd/user/bmb-eval.service").exists()


# ------------------------------------------- Modal's rate limits aren't drift

def _events(cfg, name):
    return [e for e in (json.loads(x) for x in (cfg.state_dir / "audit.jsonl").read_text().splitlines())
            if e["event"] == name]


def _due(w):
    """Let the cached accepting answer and the check schedule expire."""
    w._accepting, w._modal_next_check = (0.0, False), 0.0


def test_rate_limits_keep_the_last_good_check_and_back_off(modal_connected):
    cfg, *_, fm = modal_connected
    w = Worker(cfg)
    assert w.accepting()                                   # a full, successful check
    fm.rate_limit = 10 ** 6                                # Modal says ResourceExhausted to everything
    w._modal_verified = 0.0                                # the hourly volume check is due too
    _due(w)
    assert w.accepting()                                   # couldn't check: the last answer stands
    assert not _events(cfg, "modal_drift") and len(_events(cfg, "modal_check_failed")) == 1
    assert w._modal_next_check - time.time() >= 60 * 0.75  # backs off rather than retrying at once
    calls = fm.api_calls
    for _ in range(5):                                     # the node loop asks again and again
        w._accepting = (0.0, False)
        assert w.accepting()
    assert fm.api_calls == calls                           # without asking Modal before the back-off ends
    _due(w)
    w.accepting()
    first, second = w._modal_check_failures, w._modal_next_check - time.time()
    assert first == 2 and second >= 120 * 0.75             # longer each time
    fm.rate_limit = 0                                      # Modal answers again
    _due(w)
    assert w.accepting() and w._modal_check_failures == 0
    assert _events(cfg, "modal_check_recovered") and not _events(cfg, "modal_drift")
    assert w._modal_next_check - time.time() > 200         # then the regular interval
    calls = fm.api_calls
    w._accepting = (0.0, False)
    assert w.accepting() and fm.api_calls == calls


def test_no_check_yet_means_no_work_but_no_drift(modal_connected):
    cfg, *_, fm = modal_connected
    fm.rate_limit = 10 ** 6
    w = Worker(cfg)
    assert not w.accepting()                               # never checked: don't run anything
    assert not _events(cfg, "modal_drift")
    fm.rate_limit = 0
    _due(w)
    assert w.accepting()


def test_rate_limited_listings_are_retried_and_paced(modal_connected, monkeypatch):
    cfg, *_, fm = modal_connected
    seen, peak, lock = [0], [0], threading.Lock()
    real = fake_modal.FakeVolume.listdir

    def counting(self, path, recursive=False):
        with lock:
            seen[0] += 1
            peak[0] = max(peak[0], seen[0])
        try:
            time.sleep(0.02)
            return real(self, path, recursive)
        finally:
            with lock:
                seen[0] -= 1
    monkeypatch.setattr(fake_modal.FakeVolume, "listdir", counting)
    fm.rate_limit = 5                                      # a burst of rate limits, then answers
    assert modal_sync.verify(cfg, max_age_s=0) == []
    assert fm.rate_limit == 0 and peak[0] <= modal_sync.LIST_CONCURRENCY


def test_a_cut_short_verification_resumes_and_real_drift_still_counts(modal_connected, monkeypatch):
    cfg, *_, fm = modal_connected
    monkeypatch.setattr(modal_sync, "LIST_TRIES", 1)
    monkeypatch.setattr(modal_sync, "LIST_CONCURRENCY", 1)
    vols = json.loads((modal_backend.modal_dir(cfg) / "MANIFEST.json").read_text())["volumes"]
    fm.rate_limit = 2                                      # the first two listings fail
    with pytest.raises(modal_sync.ModalCheckError):
        modal_sync.verify(cfg, max_age_s=0)
    calls = fm.api_calls
    assert modal_sync.verify(cfg) == []                    # only the two that failed are listed again
    assert fm.api_calls - calls == 2 < len(vols)
    # A listing that answers and differs is drift, even when others couldn't be listed
    # (here the first volume, -ans, is rate-limited; -val has changed).
    val = fm.volume_dir("bmb-eval", modal_backend.vol_name(cfg.modal.volume_prefix, "split_task", "val"))
    (val / "extra.csv").write_text("x")
    fm.rate_limit = 1
    problems = modal_sync.verify(cfg, max_age_s=0)
    assert problems == [f"{modal_backend.vol_name(cfg.modal.volume_prefix, 'split_task', 'val')}: "
                        "1 unexpected, 0 missing, 0 changed file(s)"]
    w = Worker(cfg)
    fm.rate_limit = 0
    assert not w.accepting() and _events(cfg, "modal_drift")


def test_modal_sync_verify_reports_couldnt_check_not_drift(modal_connected, capsys):
    cfg, *_, fm = modal_connected
    fm.rate_limit = 10 ** 6
    with pytest.raises(SystemExit) as e:
        cli.main(["--config", str(cfg.path), "modal-sync", "--verify"])
    assert e.value.code == 2 and "Couldn't check" in capsys.readouterr().out


def test_sandbox_pythonpath_drops_entries_the_sandbox_user_cannot_list(tmp_path):
    """Modal's runtime appends root-only /pkg and /root to PYTHONPATH; pip, which lists
    every sys.path entry, failed every bundle with wheels/ until they were dropped."""
    import os
    from bmb_eval.modal_entry import _sandbox_pythonpath
    shared, private = tmp_path / "shared", tmp_path / "private"
    shared.mkdir(mode=0o755)
    private.mkdir(mode=0o700)
    shared.chmod(0o755)
    private.chmod(0o700)
    value = os.pathsep.join([str(shared), str(private), str(tmp_path / "missing"), ""])
    assert _sandbox_pythonpath(value) == str(shared)
