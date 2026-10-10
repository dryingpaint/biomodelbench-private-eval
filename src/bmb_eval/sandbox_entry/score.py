"""Runs INSIDE the score sandbox. No submitted code runs here, and no
submitted bytes are parsed except canonical predictions (stdin, numpy arrays
loaded with allow_pickle=False, produced by the validate sandbox).

Joins predictions to the answers (and the samples table's scalar input
columns, for grouping), runs the task scorer, and writes the metrics JSON to
stdout for the controller's output gate. No network.

    BMB_TASK      dir with contract.yaml + score.py
    BMB_INPUTS    samples table (only key columns are read)
    BMB_ANSWERS      answers.parquet
    BMB_ANSWERS_DIR  (optional) scorer-only files; passed as score(df, answers_dir)
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path

EXIT_SCORER = 20


def main() -> None:
    import pandas as pd

    from bmb_eval.contract import Contract, from_canonical

    task = Path(os.environ["BMB_TASK"])
    contract = Contract.load(task / "contract.yaml")
    # Keys plus scalar inputs (e.g. endpoint, assay) so scorers can group by them.
    scalar_inputs = [f.name for f in contract.inputs if f.type in ("string", "int", "float", "bool")]
    inputs = pd.read_parquet(os.environ["BMB_INPUTS"], columns=scalar_inputs)
    preds = from_canonical(contract, sys.stdin.buffer.read(), len(inputs))
    preds = pd.concat([inputs.reset_index(drop=True), preds], axis=1)

    answers = pd.read_parquet(os.environ["BMB_ANSWERS"], columns=[f.name for f in contract.answers])
    df = preds.merge(answers, on=contract.key, how="inner", validate="one_to_one")
    if len(df) != len(inputs):
        raise RuntimeError("answers do not cover every input key")

    spec = importlib.util.spec_from_file_location("task_score", task / "score.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    answers_dir = Path(os.environ["BMB_ANSWERS_DIR"]) if os.environ.get("BMB_ANSWERS_DIR") else None
    takes_dir = len(inspect.signature(module.score).parameters) > 1
    metrics = module.score(df, answers_dir) if takes_dir else module.score(df)
    sys.stdout.write(json.dumps(
        metrics, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.exit(EXIT_SCORER)
