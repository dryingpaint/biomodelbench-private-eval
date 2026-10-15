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
"""
from __future__ import annotations

import importlib.util
import inspect
import io
import json
import os
import random
import subprocess
import sys
import traceback
from pathlib import Path

EXIT_DEPS = 12
EXIT_IMPORT = 10
EXIT_RUNTIME = 11
EXIT_MEMORY = 13       # MemoryError: hit the memory limit (where it's a process limit)


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


def main() -> None:
    pred_out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    import pandas as pd

    bundle = Path(os.environ["BMB_BUNDLE"])
    inputs_dir = Path(os.environ["BMB_INPUTS_DIR"]) if os.environ.get("BMB_INPUTS_DIR") else None
    inputs_path = Path(os.environ.get("BMB_INPUTS") or inputs_dir / "samples.parquet")
    contract = json.loads(Path(os.environ["BMB_CONTRACT_FILE"]).read_text())
    key = contract["key"]
    output_names = [f["name"] for f in contract["outputs"]]
    input_names = [f["name"] for f in contract["inputs"]]
    file_cols = [f["name"] for f in contract["inputs"] if f["type"] == "file"]
    batch_size = int(contract.get("prediction", {}).get("batch_size", 1024))

    _seed()
    _install_wheels(bundle)

    sys.path.insert(0, str(bundle))
    os.chdir(bundle)
    try:
        spec = importlib.util.spec_from_file_location("predict", bundle / "predict.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not callable(getattr(module, "predict", None)):
            raise AttributeError("predict.py defines no predict()")
        if callable(getattr(module, "load_model", None)):
            _call(module.load_model, inputs_dir)
    except MemoryError:
        sys.exit(EXIT_MEMORY)
    except BaseException:
        traceback.print_exc()
        sys.exit(EXIT_IMPORT)

    inputs = pd.read_parquet(inputs_path, columns=input_names)
    for c in file_cols:
        inputs[c] = inputs[c].map(lambda p: str(inputs_dir / p) if isinstance(p, str) else p)
    parts = []
    try:
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
    except MemoryError:
        sys.exit(EXIT_MEMORY)
    except BaseException:
        traceback.print_exc()
        sys.exit(EXIT_RUNTIME)

    preds = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=key + output_names)
    buf = io.BytesIO()
    preds.to_parquet(buf, index=False)
    pred_out.write(buf.getvalue())
    pred_out.flush()


if __name__ == "__main__":
    try:
        main()
    except MemoryError:            # e.g. importing torch under a tight memory limit
        sys.exit(EXIT_MEMORY)
