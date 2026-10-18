"""bmb-eval command line.

Providers need two commands: `connect` (once) and, rarely, `stop`.
Everything else is for troubleshooting, operators, or agents.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import yaml

from . import approval, audit, leakage, scaffold
from .config import save_yaml
from .sandbox import ENGINES, detect_engine, gpus_here, make as make_sandbox, sandbox_stamp, stale_image_message

IMAGE = "bmb-eval-sandbox:latest"


def _cfg(args):
    from .config import Config
    return Config.load(args.config)


def _repo_root() -> Path:
    """Directory holding sandbox/Dockerfile (the install directory)."""
    for d in [Path(os.environ.get("BMB_CONFIG", "config.yaml")).resolve().parent, Path.cwd(),
              *Path(__file__).resolve().parents]:
        if (d / "sandbox" / "Dockerfile").is_file():
            return d
    raise SystemExit("cannot find sandbox/Dockerfile; run from the bmb-eval directory")


# ------------------------------------------------------------------- setup

def cmd_init(args) -> None:
    cfg_path = Path(args.config)
    if cfg_path.exists() and not args.force:
        raise SystemExit(f"{cfg_path} exists (use --force to overwrite)")
    engine = args.engine
    if engine == "auto":
        engine = detect_engine()
        if engine is None:
            raise SystemExit(
                "No running container engine found. Install Podman:\n"
                "  Ubuntu/Debian:  sudo apt-get install -y podman\n"
                "  RHEL/Fedora:    sudo dnf install -y podman")
    runtime = "runsc" if engine in ENGINES and _has_gvisor(engine) else None

    raw = {
        "node_name": socket.gethostname().split(".")[0],
        "sandbox": {"backend": engine, "image": IMAGE, "runtime": runtime},
        "alerts": {"webhook_url": None, "infra_failure_threshold": 3},
        "retain_bundles": False,
        "tasks": {},
    }
    if engine == "local":
        raw["sandbox"]["i_understand_local_backend_is_not_isolated"] = True
    save_yaml(cfg_path, raw)
    print(f"wrote {cfg_path}  (sandbox: {engine}" + (", gVisor" if runtime else "") + ")")


def _has_gvisor(engine: str) -> bool:
    if engine == "docker":
        r = subprocess.run(["docker", "info", "--format", "{{json .Runtimes}}"],
                           capture_output=True, text=True)
        return "runsc" in r.stdout
    return shutil.which("runsc") is not None


def cmd_build_image(args) -> None:
    """Load the prebuilt sandbox image from the hub if offered, else build it."""
    cfg = _cfg(args)
    engine = cfg.backend
    if engine not in ENGINES:
        print("local backend: no image needed")
        return
    if args.from_url:
        try:
            import hashlib
            with urllib.request.urlopen(args.from_url + ".sha256", timeout=60) as r:
                want = r.read().decode().split()[0]
            with tempfile.NamedTemporaryFile(suffix=".tar", dir=Path.home()) as f:
                print("downloading the sandbox image (several GB)...")
                h = hashlib.sha256()
                with urllib.request.urlopen(args.from_url, timeout=600) as r:
                    while chunk := r.read(8 << 20):
                        h.update(chunk)
                        f.write(chunk)
                f.flush()
                if h.hexdigest() != want:
                    raise ValueError("checksum mismatch")
                subprocess.run([engine, "load", "-i", f.name], check=True, capture_output=True)
            print("sandbox image ready")
            return
        except Exception as e:
            print(f"prebuilt image unavailable ({type(e).__name__}: {e}); building locally instead")
    root = _repo_root()
    print("building the sandbox image (a few minutes the first time)...")
    subprocess.run([engine, "build", "-q", "--build-arg", f"BMB_SANDBOX_STAMP={sandbox_stamp(root)}",
                    "-t", IMAGE, "-f", str(root / "sandbox" / "Dockerfile"), str(root)], check=True)
    print("sandbox image ready")


# ----------------------------------------------------------------- connect

REASON_TEXT = {
    "null_not_allowed": "has empty values",
    "wrong_type": "has values of the wrong type (expected {type})",
    "non_finite": "has infinite or NaN values",
    "below_min": "has values below the allowed range",
    "above_max": "has values above the allowed range",
    "not_in_allowed_values": "has values that aren't in the allowed list",
}


def data_problems(t) -> list[str]:
    """Plain-language problems with a task's data files, or []."""
    import pandas as pd

    from .contract import check_frame
    c = t.contract
    out = []
    frames = {}
    for label, path, fields in (("inputs", t.inputs, c.inputs), ("answers", t.answers, c.answers)):
        if not path.exists():
            if out:
                continue
            out.append(f"Can't find your data file at {path}.")
            continue
        df = pd.read_parquet(path)
        if label == "inputs" and t.test_inputs and t.test_inputs.exists():
            # Split task: check both halves, or a bad value's detection depends on the split.
            df = pd.concat([df, pd.read_parquet(t.test_inputs)], ignore_index=True)
        frames[label] = df
        bad = check_frame(fields, df, strict_columns=False, key=c.key)
        if not bad:
            continue
        col, reason = bad
        if reason == "missing_column":
            out.append(f"Your data file has no column '{col}'. "
                       f"Columns found: {', '.join(map(str, df.columns))}.")
        elif reason == "duplicate_keys":
            out.append(f"Some IDs ('{col}') appear more than once in your data file.")
        else:
            ftype = next((f.type for f in fields if f.name == col), "")
            out.append(f"Column '{col}' in your data file "
                       + REASON_TEXT.get(reason, reason).format(type=ftype) + ".")
    if len(frames) == 2 and not out:
        ik = set(map(tuple, frames["inputs"][c.key].astype(str).to_numpy()))
        ak = set(map(tuple, frames["answers"][c.key].astype(str).to_numpy()))
        if ik != ak:
            out.append(f"The inputs and answers must list the same IDs: {len(ik - ak)} input "
                       f"rows have no answer and {len(ak - ik)} answers have no input row.")
    return out


def _user_path(p: str) -> Path:
    """Resolve a path the way the user typed it (install.sh changes directory)."""
    q = Path(p).expanduser()
    return q if q.is_absolute() else (Path(os.environ.get("BMB_CALLER_CWD", ".")) / q).resolve()


def _bind_task(cfg_path: Path, task: str, task_dir: Path, inputs: Path, answers: Path,
               inputs_dir: Path | None = None, answers_dir: Path | None = None,
               test_inputs_dir: Path | None = None) -> None:
    raw = yaml.safe_load(cfg_path.read_text()) or {}
    raw.setdefault("tasks", {})[task] = {
        "dir": str(task_dir.relative_to(cfg_path.parent)),
        "inputs": str(inputs), "answers": str(answers)}
    if inputs_dir:
        raw["tasks"][task]["inputs_dir"] = str(inputs_dir)
    if answers_dir:
        raw["tasks"][task]["answers_dir"] = str(answers_dir)
    if test_inputs_dir:
        raw["tasks"][task]["test_inputs_dir"] = str(test_inputs_dir)
    save_yaml(cfg_path, raw)


def _selftest(cfg, task: str, verbose: bool) -> int:
    from .redteam import ATTACKS, bundle_bytes
    from .runner import Runner
    import pandas as pd
    t = cfg.tasks[task]
    rows = pd.read_parquet(t.inputs, columns=[f.name for f in t.contract.inputs])
    if rows.empty:
        # No validation rows (e.g. a published test set with no validation half): the
        # attacks need rows to run on, and test rows are reserved for finals. The
        # sandbox itself is the same for every task, so it's checked on the others.
        if verbose:
            print("  SKIP  (no validation rows to test with)")
        audit.record(cfg.state_dir, "selftest", task=task, backend=cfg.backend, failed=0, skipped=True)
        return 0
    first_row = " ".join(map(str, rows.iloc[0].tolist()))
    runner = Runner(cfg, send_alerts=False)
    local = cfg.backend == "local"
    failed = 0
    for a in ATTACKS:
        limits = a.limits
        if a.gpu_only:
            gpus = t.contract.limits_for("test", sealed=cfg.sealed).gpus
            if not (gpus and gpus_here()):
                continue
            limits = {**limits, "gpus": gpus}
        if a.backends and cfg.backend not in a.backends:
            continue
        if a.container_only and local:
            if verbose:
                print(f"  SKIP  {a.name:14s} {a.what} (local backend has no isolation)")
            continue
        p = runner.evaluate(task, bundle_bytes(a), submitter="selftest", limits_override=limits)
        got = "ok" if p["status"] == "ok" else p["error"]["code"]
        leaked = first_row in json.dumps(p)
        ok = (got == a.expect or got in a.also) and not leaked
        failed += not ok
        if verbose or not ok:
            why = "" if ok else f"  expected {a.expect}, got {got}" + (
                ", PRIVATE DATA IN PAYLOAD" if leaked else "")
            print(f"  {'PASS' if ok else 'FAIL'}  {a.name:14s} {a.what}{why}")
    audit.record(cfg.state_dir, "selftest", task=task, backend=cfg.backend, failed=failed)
    return failed


def _kb(bits: float) -> str:
    b = bits / 8
    return f"{b:,.0f} bytes" if b < 1000 else f"{b / 1000:.1f} KB"


def hardware_problems(lim, root: Path) -> list[str]:
    out = gpu_problems(lim)
    if (shutil.disk_usage(root).free >> 30) < 20:
        out.append("Less than 20 GB of disk is free; models need room to download.")
    return out


def gpu_problems(lim) -> list[str]:
    out = []
    if lim.gpus:
        mem = gpus_here()
        if len(mem) < lim.gpus:
            out.append(f"This task needs {lim.gpus} NVIDIA GPU(s); found {len(mem)}."
                       + ("" if mem else " (If it has one, its driver may not be installed.)"))
        elif lim.gpu_memory_gb and min(sorted(mem)[-lim.gpus:]) < lim.gpu_memory_gb:
            out.append(f"This task needs {lim.gpu_memory_gb} GB of GPU memory; the GPU here has "
                       f"{max(mem)} GB.")
        if (v := podman_version()) and v < (4, 1) and not shutil.which("docker"):
            out.append(f"GPU tasks need Podman 4.1 or later (this machine has {'.'.join(map(str, v))}), "
                       "or Docker with the NVIDIA container toolkit.")
    return out


def podman_version() -> tuple | None:
    if not shutil.which("podman"):
        return None
    r = subprocess.run(["podman", "--version"], capture_output=True, text=True)
    m = re.search(r"(\d+)\.(\d+)", r.stdout)
    return (int(m.group(1)), int(m.group(2))) if m else None


def disk_note(lim, root: Path) -> str | None:
    """Advice (not a blocker): large models need room to cache."""
    free_gb, want_gb = shutil.disk_usage(root).free >> 30, 2 * lim.max_bundle_gb
    if free_gb < want_gb:
        return (f"Note: {free_gb} GB of disk is free. Models for this task can be up to "
                f"{lim.max_bundle_gb} GB, so about {want_gb} GB free is recommended.")
    return None


def _summary(c, n_rows: int) -> str:
    from .prepare import describe_hidden, describe_inputs
    b = leakage.budget(c)
    lim = c.limits
    total = f", {lim.submissions_total} in total" if lim.submissions_total else ""
    n_all = (lim.submissions_total or 0) + (c.split.total_finals if c.split else 0)
    worst = (f"at most ~{_kb(b.lifetime)} of information about your data, "
             f"across all {n_all} evaluations"
             if b.lifetime else f"at most ~{b.per_submission:.0f} bits per evaluation")
    return "\n".join([
        f"  Task:                {c.title}",
        f"  Rows:                {n_rows:,}",
        f"  The model will see:  {describe_inputs(c)}",
        f"  Never shared:        {describe_hidden(c)}",
        f"  Evaluations:         at most {lim.submissions_per_day}/day per agent{total}",
        f"  What leaves:         only scores ({', '.join(m.name for m in c.metrics)}), "
        f"{c.decimals} decimals",
        f"  Worst case:          a model built to extract data could learn {worst}",
        *([(f"  Held-out test set:   rows marked 'validation' in '{c.split.by_column}' are scored "
            f"during the benchmark; 'test' rows only for final evaluations "
            f"({c.split.describe_finals()})") if c.split.by_column else
           (f"  Held-out test set:   {c.split.validation_fraction:.0%} of rows scored during the "
            f"benchmark; the rest only for final evaluations ({c.split.describe_finals()})")]
          if c.split else []),
    ])


def cmd_connect(args) -> None:
    from .config import Config
    from .node import HubAPI, HubError, parse_connect
    cfg_path = Path(args.config).resolve()
    root = cfg_path.parent
    # A bare code works on a machine that's already connected: use its hub.
    saved_hub = ((yaml.safe_load(cfg_path.read_text()) or {}).get("hub") or {}).get("url") \
        if cfg_path.exists() else None
    # ...or the hub of an earlier connect here that stopped (bad data, not approved).
    hub_file = root / "hub_url"
    if not saved_hub and hub_file.exists():
        saved_hub = hub_file.read_text().strip() or None
    hub_url, code = parse_connect(args.code, args.hub or saved_hub)

    # 1. Fetch the task package. A node that already used this code (re-runs,
    #    updates) identifies itself, so the code keeps working for it.
    existing = (yaml.safe_load(cfg_path.read_text()) or {}) if cfg_path.exists() else {}
    secret = (existing.get("hub") or {}).get("secret")
    try:
        pkg = HubAPI(hub_url, secret).enrollment(code)
    except HubError as e:
        raise SystemExit(f"Couldn't use that connect link: {e}")
    task = pkg["task"]
    print(f"Connecting to BioModelBench for task '{task}'...")
    root.mkdir(parents=True, exist_ok=True)
    hub_file.write_text(hub_url + "\n")

    # 2. Node config, sandbox image and task files.
    if not cfg_path.exists():
        cmd_init(argparse.Namespace(config=str(cfg_path), engine="auto", force=False))
    cfg = Config.load(cfg_path)
    if cfg.backend in ENGINES and (make_sandbox(cfg).image_id() == "missing" or stale_image_message(cfg)):
        cmd_build_image(argparse.Namespace(config=str(cfg_path), from_url=f"{hub_url}/sandbox-image.tar"))
    if stale := stale_image_message(cfg):
        raise SystemExit(f"\nCan't start: {stale}.\nIf the hub is still publishing a new image, "
                         "wait a few minutes and run the same command again.")
    task_dir = root / "tasks" / task
    task_dir.mkdir(parents=True, exist_ok=True)
    for name, text in pkg["files"].items():
        if "/" in name or name.startswith("."):
            raise SystemExit("unexpected file in task package")
        (task_dir / name).write_text(text)

    # 3. Their data (one table or a folder), split locally into what the model
    #    sees and the answers.
    from .contract import Contract
    from .prepare import DataError, apply_split, prepare
    try:
        contract = Contract.load(task_dir / "contract.yaml")
        inputs_dir, answers, n_rows = prepare(contract, _user_path(args.data),
                                              root / "private_data" / task)
        test_dir, split_counts = None, None
        if contract.split:
            # Held-out test set, chosen here on the provider's machine with a secret seed.
            inputs_dir, test_dir, n_val, n_test = apply_split(contract, answers.parent)
            split_counts = (n_val, n_test)
        _bind_task(cfg_path, task, task_dir, inputs_dir / "samples.parquet", answers, inputs_dir,
                   answers.parent / "answers_files" if (answers.parent / "answers_files").exists() else None,
                   test_dir)
        cfg = Config.load(cfg_path)
        if problems := data_problems(cfg.tasks[task]):
            raise DataError(problems)
    except DataError as e:
        print("\nThere's a problem with your data:")
        for p in e.problems:
            print(f"  • {p}")
        raise SystemExit("\nFix it and run the same command again.")
    print(f"✓ Your data looks good ({n_rows:,} rows)")
    if split_counts:
        how = (f"by the '{contract.split.by_column}' column" if contract.split.by_column
               else "at random on this machine")
        print(f"✓ Split {how}: {split_counts[0]:,} validation rows (scored during "
              f"the benchmark), {split_counts[1]:,} test rows (held out for final evaluations)")

    if problems := hardware_problems(cfg.tasks[task].contract.limits, root):
        print("\nThis machine isn't quite ready for this task:")
        for p in problems:
            print(f"  • {p}")
        raise SystemExit("\nPlease send this message to BioModelBench; we'll help sort it out.")

    final_lim = cfg.tasks[task].contract.limits_for("test")
    for p in gpu_problems(final_lim):
        # Not a blocker: validation runs on CPU; GPUs are needed by the final evaluations.
        print(f"  Note: {p} Final evaluations need it; everything else runs now.")
    if note := disk_note(final_lim, root):
        print(f"  {note}")

    # 4. Prove the sandbox holds on this machine, with this data.
    print("  Testing this machine's security (about a minute)...")
    if _selftest(cfg, task, verbose=False):
        raise SystemExit("\nThe security self-test failed, so nothing was started. "
                         "Please send the lines above to BioModelBench.")
    print("✓ Security self-test passed")

    # 5. Informed approval.
    print("\n" + _summary(cfg.tasks[task].contract, n_rows) + "\n")
    if not args.yes and _ask("Approve and start? [y/N] ") not in ("y", "yes"):
            raise SystemExit("Not started. Nothing was shared. "
                             "Run the same command again whenever you're ready.")

    # 6. Lock, activate, run.
    fp = approval.approve(cfg)
    audit.record(cfg.state_dir, "approved", fingerprint=fp)
    api = HubAPI(hub_url, (cfg.hub or {}).get("secret"))
    try:
        creds = api.activate(code, cfg.node_name)
    except HubError as e:
        raise SystemExit(f"Couldn't finish connecting: {e}")
    raw = yaml.safe_load(cfg_path.read_text())
    hub = raw.get("hub") or {"url": hub_url}
    hub["node_id"] = creds["node_id"]
    if "secret" in creds:
        hub["secret"] = creds["secret"]
    raw["hub"] = hub
    # Remember how this task was connected, so `bmb-eval update` can re-run it.
    raw["tasks"][task].update(code=code, data=str(_user_path(args.data)),
                              package_version=int(pkg.get("version", 1)))
    save_yaml(cfg_path, raw)
    audit.record(cfg.state_dir, "connected", hub=hub_url, task=task, node_id=creds["node_id"],
                 version=int(pkg.get("version", 1)))
    from .node import pending_updates, updates_path
    pending = pending_updates(Config.load(cfg_path))
    if pending.pop(task, None) is not None:
        updates_path(Config.load(cfg_path)).write_text(json.dumps(pending))

    started = not args.no_service and _install_service(Config.load(cfg_path))
    print(f"\n✓ Connected. Task '{task}' is live.")
    if started:
        print("  The node runs in the background and restarts on reboot. You can log out.")
    else:
        print(f"  Start the node with:  {root}/bmb-eval run")
    print(f"  Check on it:  {root}/bmb-eval status      Emergency stop:  {root}/bmb-eval stop")


def _ask(prompt: str) -> str:
    """Prompt on the terminal, even when stdin is the piped install script."""
    try:
        if sys.stdin.isatty():
            return input(prompt).strip().lower()
        # Separate handles: a tty isn't seekable, so "r+" fails.
        with open("/dev/tty", "w") as out, open("/dev/tty") as inp:
            out.write(prompt)
            out.flush()
            return inp.readline().strip().lower()
    except (OSError, EOFError):
        raise SystemExit("Run this in a terminal so you can approve it.")


def cmd_update(args) -> None:
    """Review and approve task updates from BioModelBench (no new code needed):
    re-runs the connect check for each updated task, with its saved code and data."""
    from .node import pending_updates
    cfg = _cfg(args)
    tasks = [args.task] if args.task else sorted(pending_updates(cfg))
    if not tasks:
        print("No updates from BioModelBench. Everything is up to date.")
        return
    for task in tasks:
        t = (cfg.raw.get("tasks") or {}).get(task) or {}
        if not t.get("code") or not t.get("data"):
            raise SystemExit(f"Task {task} was connected before updates were supported; "
                             "run the original connect command from BioModelBench's email once.")
        print(f"Reviewing BioModelBench's update to task '{task}'...\n")
        cmd_connect(argparse.Namespace(config=args.config, code=t["code"], data=t["data"],
                                       hub=cfg.hub.get("url"), yes=args.yes,
                                       no_service=args.no_service))


def _install_service(cfg) -> bool:
    if platform.system() != "Linux" or not shutil.which("systemctl"):
        return False
    exe = shutil.which("bmb-eval") or f"{sys.executable} -m bmb_eval.cli"
    unit = Path.home() / ".config/systemd/user/bmb-eval.service"
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text(f"""[Unit]
Description=BioModelBench evaluation node
After=network-online.target

[Service]
WorkingDirectory={cfg.path.parent}
Environment=BMB_CONFIG={cfg.path}
ExecStart={exe} --config {cfg.path} run
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
""")
    subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
    r = subprocess.run(["systemctl", "--user", "enable", "--now", "bmb-eval"], capture_output=True)
    if r.returncode != 0:
        return False
    subprocess.run(["systemctl", "--user", "restart", "bmb-eval"], capture_output=True)
    user = os.environ.get("USER", "")
    if subprocess.run(["loginctl", "enable-linger", user], capture_output=True).returncode != 0:
        print(f"  NOTE: so the node keeps running after you log out, run once:\n"
              f"    sudo loginctl enable-linger {user}")
    return True


def cmd_install_service(args) -> None:
    if not _install_service(_cfg(args)):
        raise SystemExit("could not install a systemd user service; use `bmb-eval run`")
    print("service installed; logs: journalctl --user -u bmb-eval -f")


def cmd_run(args) -> None:
    from .node import Worker
    Worker(_cfg(args)).run_forever()


# ------------------------------------------------------------- day to day

def cmd_status(args) -> None:
    from .node import paused_path
    from .store import Store
    cfg = _cfg(args)
    ok, n, msg = audit.verify(cfg.state_dir)
    drift = approval.diff(cfg)
    running = subprocess.run(["systemctl", "--user", "is-active", "bmb-eval"],
                             capture_output=True, text=True).stdout.strip() if shutil.which("systemctl") else "?"
    print(f"connected   {cfg.hub.get('url', 'NO (run the connect command)')}")
    print(f"service     {running}")
    from .node import pending_updates
    for task, version in pending_updates(cfg).items():
        print(f"update      task {task} has a new version ({version}) from BioModelBench: "
              "run `bmb-eval update` to review it")
    print(f"accepting   {'NO: stopped (bmb-eval resume)' if paused_path(cfg).exists() else 'NO: settings changed (bmb-eval approve)' if drift else 'yes'}")
    print(f"sandbox     {cfg.backend}{' + gVisor' if cfg.runtime else ''}")
    print(f"audit log   {n} entries, {msg}")
    counts = Store(cfg.state_dir).summary()
    for task in cfg.tasks:
        c = counts.get(task, {})
        print(f"task        {task}: {c.get('done', 0)} evaluated, "
              f"{c.get('queued', 0) + c.get('running', 0)} in progress")


def cmd_stop(args) -> None:
    from . import alerts
    from .node import paused_path
    cfg = _cfg(args)
    paused_path(cfg).write_text("paused\n")
    n = make_sandbox(cfg).kill_all()
    audit.record(cfg.state_dir, "stopped", killed=n)
    alerts.send(cfg, "control", f"stopped by operator; {n} running evaluation(s) killed")
    print(f"Stopped. No new evaluations will run ({n} running one(s) killed).\n"
          f"Resume any time with: bmb-eval resume")


def cmd_resume(args) -> None:
    from .node import paused_path
    cfg = _cfg(args)
    paused_path(cfg).unlink(missing_ok=True)
    audit.record(cfg.state_dir, "resumed")
    print("Resumed.")


def cmd_approve(args) -> None:
    cfg = _cfg(args)
    for d in approval.diff(cfg):
        print("  changed:", d)
    fp = approval.approve(cfg)
    audit.record(cfg.state_dir, "approved", fingerprint=fp)
    print("Approved. The node will accept evaluations with these settings.")


# ------------------------------------------------------------ troubleshoot

def cmd_check(args) -> None:
    cfg = _cfg(args)
    errors, warnings = [], []
    if cfg.backend == "local":
        errors.append("sandbox.backend is local: submitted code would NOT be isolated")
    elif subprocess.run([cfg.backend, "info"], capture_output=True).returncode != 0:
        errors.append(f"{cfg.backend} is not running")
    elif make_sandbox(cfg).image_id() == "missing":
        errors.append(f"sandbox image {cfg.image} missing (bmb-eval build-image)")
    elif stale := stale_image_message(cfg):
        errors.append(stale)
    elif not cfg.runtime:
        warnings.append("gVisor not configured (optional hardening; see docs/PROVIDER_SETUP.md)")
    for tid, t in cfg.tasks.items():
        print(f"\n== {tid}")
        errors += [f"{tid}: {p}" for p in data_problems(t)]
        print("   " + leakage.describe(t.contract).replace("\n", "\n   "))
        if leakage.budget(t.contract).per_submission == float("inf"):
            errors.append(f"{tid}: a metric has no min/max bounds")
    print()
    for w in warnings:
        print(f"WARN  {w}")
    for e in errors:
        print(f"FAIL  {e}")
    if approval.diff(cfg):
        print("NOTE  settings not yet approved (bmb-eval approve)")
    if errors:
        sys.exit(1)
    print("check passed")


def cmd_selftest(args) -> None:
    cfg = _cfg(args)
    task = args.task or next(iter(cfg.tasks), None)
    if task is None:
        print("no task connected yet; the self-test runs when you connect")
        return
    print(f"security self-test (task {task!r}, sandbox {cfg.backend}"
          + (" + gVisor" if cfg.runtime else "") + ")")
    failed = _selftest(cfg, task, verbose=True)
    if cfg.backend == "local":
        print("\nlocal backend: isolation NOT tested and NOT present.")
    if failed:
        print(f"\n{failed} check(s) failed. Please send this output to BioModelBench.")
        sys.exit(1)
    print("\nall checks passed")


def cmd_eval(args) -> None:
    from .bundle import pack_dir
    from .runner import Runner
    cfg = _cfg(args)
    p = Path(args.bundle)
    data = pack_dir(p) if p.is_dir() else p.read_bytes()
    print(json.dumps(Runner(cfg).evaluate(args.task, data, submitter="operator"), indent=2))


def cmd_contract(args) -> None:
    print(json.dumps(_cfg(args).tasks[args.task].contract.public(), indent=2))


def cmd_audit(args) -> None:
    ok, n, msg = audit.verify(_cfg(args).state_dir)
    print(f"{n} entries: {msg}")
    sys.exit(0 if ok else 1)


# ------------------------------------------------------------------- agent

def _load_public(path: str):
    from .contract import Contract
    return Contract.from_dict(json.loads(Path(path).read_text()), public=True)


def cmd_submit(args) -> None:
    from .client import Client
    token = args.token or os.environ.get("BMB_TOKEN")
    if not token:
        raise SystemExit("pass --token or set BMB_TOKEN")
    c = Client(args.hub, token)
    res = c.submit(args.task, args.bundle) if args.wait else c.submit_async(args.task, args.bundle)
    print(json.dumps(res, indent=2))


def cmd_synth(args) -> None:
    from .prepare import synth_files
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = synth_files(_load_public(args.contract), args.n, out)
    print(f"wrote a synthetic inputs folder to {out} ({len(df)} samples)")


def cmd_dryrun(args) -> None:
    """Run predict + validate on synthetic inputs, unisolated, with full logs.
    For agents iterating on a bundle before spending a private evaluation."""
    from .prepare import synth_files
    from .sandbox import LocalSandbox, Mount
    c = _load_public(args.contract)
    bundle = Path(args.bundle).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        inputs_dir = Path(tmp) / "inputs"
        inputs_dir.mkdir()
        synth_files(c, args.n, inputs_dir)
        if c.data.include:
            print("note: pass-through files are empty placeholders in a dry run: "
                  + ", ".join(i["path"] for i in c.data.include), file=sys.stderr)
        contract_file = Path(tmp) / "contract.json"
        contract_file.write_text(json.dumps(c.public()))
        inputs = inputs_dir / "samples.parquet"
        cm = Mount("BMB_CONTRACT_FILE", contract_file, "")
        env = {}
        lim = c.limits_for("test" if args.final else "validation")
        sb = LocalSandbox()
        r = sb.run(name="dryrun-predict", entry="predict", env=env, limits=lim,
                   mounts=[Mount("BMB_BUNDLE", bundle, ""), Mount("BMB_INPUTS_DIR", inputs_dir, ""), cm],
                   timeout_s=lim.predict_timeout_s, stdout_cap=lim.max_predictions_mb << 20)
        sys.stderr.write(r.stderr)
        if r.exit_code != 0 or r.timed_out or r.stdout_overflow or not r.stdout:
            raise SystemExit(f"predict failed: exit={r.exit_code} timed_out={r.timed_out} "
                             f"overflow={r.stdout_overflow} bytes={len(r.stdout)}")
        v = sb.run(name="dryrun-validate", entry="validate", env=env, limits=lim,
                   mounts=[Mount("BMB_INPUTS", inputs, ""), cm], stdin=r.stdout,
                   timeout_s=lim.validate_timeout_s, stdout_cap=lim.max_predictions_mb << 20)
        if v.exit_code != 0:
            sys.stderr.write(v.stderr)
            raise SystemExit(f"predictions rejected: {v.stdout.decode(errors='replace')}")
    print(f"ok: predict ran in {r.seconds}s and its predictions pass the contract "
          f"({args.n} synthetic rows). Metrics need the private node.")


# -------------------------------------------------------------------- main

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        prog="bmb-eval",
        description="BioModelBench evaluation node. Providers: `connect` once; "
                    "`status`, `stop`, `resume` as needed.")
    ap.add_argument("--config", default=os.environ.get("BMB_CONFIG", "config.yaml"))
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="command")

    # Providers
    p = sub.add_parser("connect", help="connect this machine using the code BioModelBench sent you")
    p.add_argument("code", help="XXXX-XXXX-XXXX, or https://<hub>/c/XXXX-XXXX-XXXX")
    p.add_argument("--data", required=True, help="your data: one table (.csv, .tsv, .parquet) or a folder")
    p.add_argument("--hub", default=os.environ.get("BMB_HUB"), help=argparse.SUPPRESS)
    p.add_argument("--yes", action="store_true", help="approve without asking")
    p.add_argument("--no-service", action="store_true", help="don't install the background service")
    p.set_defaults(fn=cmd_connect)
    sub.add_parser("status", help="is everything running?").set_defaults(fn=cmd_status)
    sub.add_parser("stop", help="emergency stop: refuse new work, kill running evaluations"
                   ).set_defaults(fn=cmd_stop)
    sub.add_parser("resume", help="undo stop").set_defaults(fn=cmd_resume)
    sub.add_parser("approve", help="accept changed settings").set_defaults(fn=cmd_approve)
    p = sub.add_parser("update", help="review and approve task updates from BioModelBench")
    p.add_argument("task", nargs="?", help="only this task (default: every task with an update)")
    p.add_argument("--yes", action="store_true", help="approve without asking")
    p.add_argument("--no-service", action="store_true", help=argparse.SUPPRESS)
    p.set_defaults(fn=cmd_update)

    # Setup and troubleshooting
    p = sub.add_parser("init", help="(setup) write config.yaml")
    p.add_argument("--engine", default="auto", choices=["auto", *ENGINES, "local"])
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_init)
    p = sub.add_parser("build-image", help="(setup) get or build the sandbox image")
    p.add_argument("--from", dest="from_url", help="URL of a prebuilt image tarball")
    p.set_defaults(fn=cmd_build_image)
    sub.add_parser("run", help="(service) run the node in the foreground").set_defaults(fn=cmd_run)
    sub.add_parser("install-service", help="(service) reinstall the background service"
                   ).set_defaults(fn=cmd_install_service)
    sub.add_parser("check", help="(troubleshoot) check data, sandbox, leakage budget"
                   ).set_defaults(fn=cmd_check)
    p = sub.add_parser("selftest", help="(troubleshoot) run attack bundles against the sandbox")
    p.add_argument("--task")
    p.set_defaults(fn=cmd_selftest)
    p = sub.add_parser("eval", help="(troubleshoot) evaluate a bundle locally")
    p.add_argument("task")
    p.add_argument("bundle")
    p.set_defaults(fn=cmd_eval)
    p = sub.add_parser("audit", help="(troubleshoot) verify the audit log")
    p.add_argument("action", choices=["verify"])
    p.set_defaults(fn=cmd_audit)
    p = sub.add_parser("contract", help="(operator) print a task's public contract")
    p.add_argument("task")
    p.set_defaults(fn=cmd_contract)

    # Agents
    p = sub.add_parser("submit", help="(agent) submit a bundle to the hub")
    p.add_argument("--hub", required=True)
    p.add_argument("--token")
    p.add_argument("--no-wait", dest="wait", action="store_false")
    p.add_argument("task")
    p.add_argument("bundle")
    p.set_defaults(fn=cmd_submit)
    p = sub.add_parser("synth", help="(agent) synthetic inputs from a public contract")
    p.add_argument("contract")
    p.add_argument("--out", default="synthetic_inputs")
    p.add_argument("-n", type=int, default=200)
    p.set_defaults(fn=cmd_synth)
    p = sub.add_parser("dryrun", help="(agent) run a bundle on synthetic inputs with full logs")
    p.add_argument("contract")
    p.add_argument("bundle")
    p.add_argument("-n", type=int, default=200)
    p.add_argument("--final", action="store_true",
                   help="use the final-evaluation limits (e.g. GPUs) instead of validation's")
    p.set_defaults(fn=cmd_dryrun)

    from .confidential import add_parsers
    add_parsers(sub)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
