"""Runs INSIDE the predict sandbox, alongside untrusted bundle code.

Sees: the bundle and the task's inputs folder (both read-only). Never sees
answers. No network. No writable host path.

Predictions leave as parquet bytes on the original stdout; fd 1 is pointed
at stderr before bundle code loads, so the bundle's prints go to the
provider-only log. Nothing this process emits is trusted: the controller
caps stdout bytes, re-validates them in a separate sandbox, and maps the
exit code onto a fixed enum.

    BMB_BUNDLE         dir containing predict.py
    BMB_INPUTS_DIR     inputs folder: samples.parquet plus any files the task provides
    BMB_INPUTS         (alternative) path to the samples table when there is no folder
    BMB_CONTRACT_FILE  public contract JSON

predict.py may define predict(df) or predict(df, inputs_dir), and optionally
load_model() or load_model(inputs_dir). File columns arrive as absolute paths.

Batch mode (BMB_BATCH=<n>, family batching on nodes that opt in): one process
predicts n tasks with the same bundle, so the model is imported and loaded once.
BMB_CONTRACT_FILE then holds a list of {"contract", "timeout_s", "max_bytes"},
one per task, and task i's inputs are BMB_INPUTS_DIR_<i> (or BMB_INPUTS_<i>).
load_model() runs once; load_model(inputs_dir) depends on the task, so it runs
before each task's predict. Each task's outcome leaves as one frame on stdout,
in task order and flushed as it completes, so tasks finished before the process
dies keep their predictions:

    b"BMBF" | status (uint32, 0 = ok, else this module's exit code) | length (uint64) | bytes

A failing task gets its own status and the next task still runs. Each task's
time limit (its timeout_s, less the shared load) is enforced here with a timer,
which the model could defeat; the controller's limit on the whole batch is the
real bound.
"""
from __future__ import annotations

import importlib.util
import inspect
import io
import json
import os
import random
import signal
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path

EXIT_DEPS = 12
EXIT_IMPORT = 10
EXIT_RUNTIME = 11
EXIT_MEMORY = 13       # MemoryError: hit the memory limit (where it's a process limit)
# Batch mode only, as a task's frame status: in single mode the controller sees these itself.
EXIT_TIMEOUT = 14
EXIT_TOO_LARGE = 15

FRAME = struct.Struct(">4sIQ")
FRAME_MAGIC = b"BMBF"


class _TaskTimeout(BaseException):
    """A batched task ran out of time. BaseException, so a model's `except Exception`
    doesn't swallow it."""


def _seed() -> None:
    random.seed(0)
    try:
        import numpy as np
        np.random.seed(0)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def _install_wheels(bundle: Path) -> None:
    wheels = bundle / "wheels"
    if not wheels.is_dir():
        return
    target = Path(os.environ.get("TMPDIR", "/tmp")) / "bmb_site"
    whls = sorted(str(p) for p in wheels.glob("*.whl"))
    if not whls:
        return
    # Wheels only (no sdists), so installing runs no package code.
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
         "--disable-pip-version-check", "--target", str(target), *whls],
        capture_output=True, text=True,
    )
    sys.stderr.write(r.stdout + r.stderr)
    if r.returncode != 0:
        sys.exit(EXIT_DEPS)
    sys.path.insert(0, str(target))


def _call(fn, *args):
    """Call fn with as many of args as it accepts (so predict(df) still works)."""
    try:
        n = len(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        n = len(args)
    return fn(*args[:n])


def _takes_args(fn) -> bool:
    try:
        return len(inspect.signature(fn).parameters) > 0
    except (TypeError, ValueError):
        return True


def _load_module(bundle: Path):
    """Import the bundle's predict.py (raises what the import raises)."""
    sys.path.insert(0, str(bundle))
    os.chdir(bundle)
    spec = importlib.util.spec_from_file_location("predict", bundle / "predict.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "predict", None)):
        raise AttributeError("predict.py defines no predict()")
    return module


def _predict_task(module, contract: dict, inputs_path: Path, inputs_dir: Path | None):
    """Predictions for one task's inputs, as a DataFrame of key + output columns."""
    import pandas as pd
    key = contract["key"]
    output_names = [f["name"] for f in contract["outputs"]]
    input_names = [f["name"] for f in contract["inputs"]]
    file_cols = [f["name"] for f in contract["inputs"] if f["type"] == "file"]
    batch_size = int(contract.get("prediction", {}).get("batch_size", 1024))
    inputs = pd.read_parquet(inputs_path, columns=input_names)
    for c in file_cols:
        inputs[c] = inputs[c].map(lambda p: str(inputs_dir / p) if isinstance(p, str) else p)
    parts = []
    for start in range(0, len(inputs), batch_size):
        batch = inputs.iloc[start:start + batch_size].reset_index(drop=True)
        res = _call(module.predict, batch.copy(), inputs_dir)
        if isinstance(res, pd.DataFrame):
            res = res.reset_index(drop=True)
            missing = [c for c in output_names if c not in res.columns]
            if missing:
                raise ValueError(f"predict() output lacks columns {missing}")
            res = res[output_names]
        elif len(output_names) == 1:
            res = pd.DataFrame({output_names[0]: list(res)})
        else:
            raise TypeError("predict() must return a DataFrame for multi-output tasks")
        if len(res) != len(batch):
            raise ValueError(f"predict() returned {len(res)} rows for {len(batch)} inputs")
        parts.append(pd.concat([batch[key], res], axis=1))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=key + output_names)


def _parquet(df) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def main() -> None:
    pred_out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    import pandas  # noqa: F401  before the bundle's wheels: the image's pandas reads the inputs

    bundle = Path(os.environ["BMB_BUNDLE"])
    if os.environ.get("BMB_BATCH"):
        return main_batch(pred_out, bundle, int(os.environ["BMB_BATCH"]))
    inputs_dir = Path(os.environ["BMB_INPUTS_DIR"]) if os.environ.get("BMB_INPUTS_DIR") else None
    inputs_path = Path(os.environ.get("BMB_INPUTS") or inputs_dir / "samples.parquet")
    contract = json.loads(Path(os.environ["BMB_CONTRACT_FILE"]).read_text())

    _seed()
    _install_wheels(bundle)
    try:
        module = _load_module(bundle)
        if callable(getattr(module, "load_model", None)):
            _call(module.load_model, inputs_dir)
    except MemoryError:
        sys.exit(EXIT_MEMORY)
    except BaseException:
        traceback.print_exc()
        sys.exit(EXIT_IMPORT)

    try:
        preds = _predict_task(module, contract, inputs_path, inputs_dir)
    except MemoryError:
        sys.exit(EXIT_MEMORY)
    except BaseException:
        traceback.print_exc()
        sys.exit(EXIT_RUNTIME)
    pred_out.write(_parquet(preds))
    pred_out.flush()


def main_batch(pred_out, bundle: Path, n: int) -> None:
    specs = json.loads(Path(os.environ["BMB_CONTRACT_FILE"]).read_text())
    if len(specs) != n:
        raise SystemExit(f"batch of {n} tasks with {len(specs)} contracts")
    done = 0

    def frame(status: int, data: bytes = b"") -> None:
        nonlocal done
        pred_out.write(FRAME.pack(FRAME_MAGIC, status, len(data)) + data)
        pred_out.flush()
        done += 1

    def fail_rest(status: int) -> None:
        while done < n:
            frame(status)

    def expire(signum, frame_) -> None:
        raise _TaskTimeout()

    _seed()
    try:
        _install_wheels(bundle)
    except SystemExit as e:              # every task would fail to install them alike
        return fail_rest(int(e.code))
    t0 = time.monotonic()
    try:
        module = _load_module(bundle)
        load = getattr(module, "load_model", None)
        per_task_load = callable(load) and _takes_args(load)
        if callable(load) and not per_task_load:
            load()
    except MemoryError:
        return fail_rest(EXIT_MEMORY)
    except BaseException:
        traceback.print_exc()
        return fail_rest(EXIT_IMPORT)
    # The shared load counts against every task's limit, as it would have run alone.
    shared_load_s = time.monotonic() - t0
    signal.signal(signal.SIGALRM, expire)

    for i, spec in enumerate(specs):
        inputs_dir = Path(os.environ[f"BMB_INPUTS_DIR_{i}"]) if os.environ.get(f"BMB_INPUTS_DIR_{i}") else None
        inputs_path = Path(os.environ.get(f"BMB_INPUTS_{i}") or inputs_dir / "samples.parquet")
        print(f"# ---- batch task {i}: {spec['contract']['id']}", file=sys.stderr, flush=True)
        _seed()                          # each task's predictions don't depend on the ones before
        stage = EXIT_IMPORT
        try:
            budget = spec["timeout_s"] - shared_load_s
            if budget <= 0:
                raise _TaskTimeout()
            signal.setitimer(signal.ITIMER_REAL, budget)
            if per_task_load:
                load(inputs_dir)
            stage = EXIT_RUNTIME
            data = _parquet(_predict_task(module, spec["contract"], inputs_path, inputs_dir))
            signal.setitimer(signal.ITIMER_REAL, 0)
        except _TaskTimeout:
            frame(EXIT_TIMEOUT)
            continue
        except MemoryError:
            signal.setitimer(signal.ITIMER_REAL, 0)
            frame(EXIT_MEMORY)
            continue
        except BaseException:
            signal.setitimer(signal.ITIMER_REAL, 0)
            traceback.print_exc()
            frame(stage)
            continue
        frame(EXIT_TOO_LARGE if len(data) > spec["max_bytes"] else 0,
              b"" if len(data) > spec["max_bytes"] else data)


if __name__ == "__main__":
    try:
        main()
    except MemoryError:            # e.g. importing torch under a tight memory limit
        sys.exit(EXIT_MEMORY)
