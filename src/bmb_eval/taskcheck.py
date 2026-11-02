"""`bmb-eval check-task <task_dir>`: try a task's contract.yaml and score.py before any
provider connects.

A scorer whose output doesn't match the contract (a metric name spelled differently from
its `per:` group, a group skipped when it has no rows, a None on a non-nullable metric)
otherwise surfaces only at the first real submission, as output_contract_violation, on
the provider's machine. Here the scorer runs exactly as the score stage runs it (the
sandbox entry, unisolated) on fake answers and predictions that satisfy the contract,
including edge cases, and each output goes through the node's output gate for both
validation and final (test) evaluations.
"""
from __future__ import annotations

import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import leakage
from .contract import SCALARS, Contract, gate_metrics, to_canonical, validate_predictions
from .errors import EvalError
from .prepare import DataError, prepare, synth_column
from .sandbox import LocalSandbox, Mount
from .scaffold import synth_inputs

SHOW = 6           # metric names listed per problem; the rest are counted
SUBSET_ROWS = 3


@dataclass
class Case:
    name: str
    inputs: pd.DataFrame
    answers: pd.DataFrame
    constant_preds: bool = False


def load(task_dir: Path) -> tuple[Contract | None, list[str], dict | None]:
    """The contract, or None and why it can't be used. Also flags what loads but is
    wrong anyway (a version written as a string). The third value is the contract with
    its version replaced, when it had to be, for the scorer run to read instead."""
    path = task_dir / "contract.yaml"
    if not path.is_file():
        return None, [f"no contract.yaml in {task_dir}"], None
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        return None, [f"contract.yaml isn't valid YAML: {e}"], None
    if not isinstance(raw, dict):
        return None, ["contract.yaml should be a mapping (id:, version:, inputs:, ...)"], None
    problems, patched = [], None
    if "version" in raw and (isinstance(raw["version"], bool) or not isinstance(raw["version"], int)):
        # Contract.load still reads a quoted "2" (older contracts have it); new ones write the
        # number. Reported, then checked as version 0 so the scorer's problems show too.
        problems.append(f"version must be written as an integer, e.g. `version: 2` (got {raw['version']!r})")
        raw["version"] = 0
        patched = raw
    try:
        c = Contract.from_dict(raw, base=task_dir)
    except KeyError as e:
        return None, problems + [f"contract.yaml is missing {e}"], None
    except (ValueError, TypeError, OSError) as e:
        return None, problems + [f"contract.yaml: {e}"], None
    c.path = path
    if not (task_dir / "score.py").is_file():
        problems.append(f"no score.py in {task_dir}")
    return c, problems, patched


def synth_data(c: Contract, n: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fake inputs and answers satisfying the contract (answers' key columns match)."""
    inputs = synth_inputs(c, n=n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    answers = inputs[c.key].copy()
    for f in c.answers:
        if f.name in c.key:
            continue
        if c.split and f.name == c.split.by_column:
            answers[f.name] = rng.choice(["validation", "test"], n)
        else:
            answers[f.name] = synth_column(f, inputs, rng)
    return inputs, answers


def real_data(c: Contract, data: Path, tmp: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path | None]:
    """A provider-style table or folder, prepared the way `connect` prepares it."""
    inputs_dir, answers, _ = prepare(c, data, tmp / "prepared")
    private = answers.parent / "answers_files"
    return (pd.read_parquet(inputs_dir / "samples.parquet"), pd.read_parquet(answers),
            private if private.exists() else None)


def cases(c: Contract, inputs: pd.DataFrame, answers: pd.DataFrame) -> list[Case]:
    out = [Case("random predictions", inputs, answers),
           Case("constant predictions", inputs, answers, constant_preds=True)]
    rng = np.random.default_rng(0)
    one_class = answers.copy()
    for f in c.answers:
        if f.name not in c.key and not (c.split and f.name == c.split.by_column):
            one_class[f.name] = synth_column(f, inputs, rng, constant=True)
    out.append(Case("answers all one value (e.g. one class)", inputs, one_class))
    # A group with no rows: scorers that skip it emit fewer metrics than the contract
    # declares, which the gate rejects. Real validation halves and small tasks have them.
    for f in c.inputs:
        if f.type in SCALARS and f.allowed and len(f.allowed) > 1 and f.name not in c.key:
            gone = inputs[f.name].dropna().iloc[0] if inputs[f.name].notna().any() else None
            rest = inputs[inputs[f.name] != gone].reset_index(drop=True)
            if gone is not None and len(rest):
                out.append(Case(f"no rows with {f.name}={gone}", rest, answers.merge(rest[c.key], on=c.key)))
    few = inputs.head(SUBSET_ROWS).reset_index(drop=True)
    out.append(Case(f"only {SUBSET_ROWS} rows", few, answers.merge(few[c.key], on=c.key)))
    return out


def predictions(c: Contract, inputs: pd.DataFrame, constant: bool, seed: int) -> bytes:
    """Fake predictions, checked by the validate stage's own code and returned in the
    canonical form the score stage reads."""
    rng = np.random.default_rng(seed)
    preds = inputs[c.key].copy()
    for f in c.outputs:
        preds[f.name] = synth_column(f, inputs, rng, constant=constant)
    try:
        ordered = validate_predictions(c, preds, inputs)
    except EvalError as e:
        raise RuntimeError(f"check-task made predictions the contract rejects ({e.detail}); "
                           "this is a bug in check-task") from None
    return to_canonical(c, ordered)


def run_scorer(c: Contract, task_dir: Path, case: Case, answers_dir: Path | None, tmp: Path,
               seed: int, synthetic: bool) -> tuple[bytes | None, str]:
    """The score stage's entry on this case: (stdout, "") or (None, why it failed)."""
    d = Path(tempfile.mkdtemp(dir=tmp))
    scalar_inputs = [f.name for f in c.inputs if f.type in SCALARS]
    case.inputs[scalar_inputs].to_parquet(d / "samples.parquet", index=False)
    case.answers.to_parquet(d / "answers.parquet", index=False)
    lim = c.limits
    r = LocalSandbox().run(
        name="check-task-score", entry="score", env={}, limits=lim, timeout_s=lim.score_timeout_s,
        stdout_cap=c.max_export_bytes * 4,
        stdin=predictions(c, case.inputs, case.constant_preds, seed),
        mounts=[Mount("BMB_TASK", task_dir, ""), Mount("BMB_INPUTS", d / "samples.parquet", ""),
                Mount("BMB_ANSWERS", d / "answers.parquet", ""),
                *([Mount("BMB_ANSWERS_DIR", answers_dir, "")] if answers_dir else [])])
    if r.timed_out:
        return None, f"the scorer ran past limits.score_timeout_s ({lim.score_timeout_s} s)"
    if r.stdout_overflow:
        return None, f"the scorer wrote more than {c.max_export_bytes * 4} bytes (export.max_bytes x 4)"
    if r.exit_code != 0:
        tail = "\n".join(r.stderr.strip().splitlines()[-6:])
        why = f"the scorer failed (exit {r.exit_code}):\n" + _indent(tail, 4)
        if "ModuleNotFoundError" in r.stderr:
            why += ("\n    (The score sandbox has the packages in sandbox/requirements.txt; install the "
                    "missing one in this environment to check the scorer here.)")
        elif synthetic:
            why += ("\n    (Synthetic rows combine input values at random. If the scorer accepts only "
                    "real combinations, pass --data with a few real-shaped rows.)")
        return None, why
    return r.stdout, ""


def _names(names) -> str:
    names = sorted(names)
    return ", ".join(names[:SHOW]) + (f", ... ({len(names) - SHOW} more)" if len(names) > SHOW else "")


def output_problems(c: Contract, raw: bytes) -> list[str]:
    """Everything wrong with one scorer output, named (the gate stops at the first)."""
    try:
        obj = json.loads(raw)
    except ValueError:
        return [f"the scorer's output isn't JSON: {raw[:200]!r}"]
    if not isinstance(obj, dict):
        return [f"score() should return a dict of metric name -> value, not {type(obj).__name__}"]
    declared = {m.name: m for m in c.metrics}
    out = []
    if extra := set(obj) - set(declared):
        out.append(f"{len(extra)} metric(s) emitted but not declared: {_names(extra)}")
    if missing := set(declared) - set(obj):
        out.append(f"{len(missing)} declared metric(s) not emitted: {_names(missing)}"
                   " (a metric with nothing to score should be None, and declared nullable)")
    nulls, bad_type, non_finite, oob, not_int = [], [], [], [], []
    for name, v in obj.items():
        m = declared.get(name)
        if m is None:
            continue
        if v is None:
            if not m.nullable:
                nulls.append(name)
        elif isinstance(v, bool) or not isinstance(v, (int, float)):
            bad_type.append(f"{name}={v!r}")
        elif not math.isfinite(v):
            non_finite.append(name)
        elif (m.min is not None and v < m.min - 1e-9) or (m.max is not None and v > m.max + 1e-9):
            oob.append(f"{name}={v:.4g} (bounds {m.min}..{m.max})")
        elif m.type == "int" and float(v) != int(v):
            not_int.append(f"{name}={v}")
    for label, xs in (("None but not declared nullable", nulls), ("not a number", bad_type),
                      ("NaN or infinite", non_finite), ("outside the declared bounds", oob),
                      ("not an integer (type: int)", not_int)):
        if xs:
            out.append(f"{len(xs)} metric(s) {label}: {_names(xs)}")
    return out


def check(task_dir: Path, data: Path | None = None, n: int = 300, seed: int = 0) -> int:
    """Print the report; return the number of problems."""
    task_dir = task_dir.resolve()
    print(f"Checking task {task_dir}")
    c, problems, patched = load(task_dir)
    for p in problems:
        print(f"  FAIL  {p}")
    if c is None or not (task_dir / "score.py").is_file():
        return len(problems)
    print(f"  contract {c.id} v{c.version}: {len(c.metrics)} metrics "
          f"({sum(m.final_only for m in c.metrics)} final-only), headline {c.headline}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if patched:
            # The score entry reads contract.yaml itself: give it the loadable version.
            shutil.copytree(task_dir, tmp / "task")
            task_dir = tmp / "task"
            (task_dir / "contract.yaml").write_text(yaml.safe_dump(patched, sort_keys=False))
        answers_dir = None
        if data is not None:
            try:
                inputs, answers, answers_dir = real_data(c, data, tmp)
            except DataError as e:
                for p in e.problems:
                    print(f"  FAIL  --data: {p}")
                return len(problems) + len(e.problems)
            print(f"  data: {len(inputs):,} rows from {data}; predictions are synthetic")
        else:
            inputs, answers = synth_data(c, n, seed)
            print(f"  data: {n} synthetic rows (pass --data <table or folder> to use real-shaped rows)")
            if c.data.private:
                answers_dir = tmp / "answers_files"
                answers_dir.mkdir()
                print("  note: the scorer-only files (data.private) are missing in synthetic data; "
                      "pass --data <folder> to check a scorer that reads them")
        seen = {}
        for case in cases(c, inputs, answers):
            out, why = run_scorer(c, task_dir, case, answers_dir, tmp, seed, synthetic=data is None)
            found = [why] if out is None else output_problems(c, out)
            if out is not None and not found:
                # The gate itself, as the node runs it, for both kinds of evaluation.
                for split in ("validation", "test"):
                    try:
                        gate_metrics(c, out, split)
                    except EvalError as e:
                        found.append(f"the output gate rejects it for split={split}: {e.internal}")
            if found and tuple(found) in seen:
                # e.g. a scorer that fails on import fails every case the same way
                print(f"  FAIL  {case.name}: the same as for {seen[tuple(found)]}")
                problems.append(f"{case.name}: as {seen[tuple(found)]}")
            elif found:
                seen[tuple(found)] = case.name
                print(f"  FAIL  {case.name}")
                for p in found:
                    print(_indent(p, 8))
                problems += [f"{case.name}: {p}" for p in found]
            else:
                exported = {s: len(gate_metrics(c, out, s)) for s in ("validation", "test")}
                print(f"  ok    {case.name}: validation exports {exported['validation']} metrics, "
                      f"final evaluations {exported['test']}")
    print()
    print(_indent(leakage.describe(c), 2))
    if leakage.budget(c).per_submission == float("inf"):
        problems.append("a metric has no min/max bounds, so the leakage budget is unbounded")
        print("  FAIL  a metric has no min/max bounds, so the leakage budget is unbounded")
    print()
    print(f"check-task: {len(problems)} problem(s)" if problems else "check-task passed")
    return len(problems)


def _indent(text: str, n: int) -> str:
    return "\n".join(" " * n + line for line in text.splitlines())
