"""Generate task contracts, scorers and synthetic data.

add_task() infers a contract from the provider's parquet files so setup is
one command plus a review. It records names, types and nullability only.
It never copies min/max, category sets or row counts out of private data:
the contract is shown to the agent, and a column's max is a private value.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .contract import Contract

AMINO = list("ACDEFGHIKLMNPQRSTVWY")


def _type_of(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    if pd.api.types.is_integer_dtype(s):
        return "int"
    if pd.api.types.is_float_dtype(s):
        return "float"
    return "string"


def _field_yaml(name: str, type_: str, nullable: bool, indent: str = "  ") -> str:
    line = f"{indent}- {{name: {name}, type: {type_}"
    if nullable:
        line += ", nullable: true"
    return line + ", description: \"\"}"


def parse_columns(spec: str) -> list[tuple[str, str, bool]]:
    """'id:string,sequence:string,ph:float?' -> [(name, type, nullable)]; '?' = nullable."""
    out = []
    for part in filter(None, (x.strip() for x in spec.split(","))):
        name, _, t = part.partition(":")
        t = t or "string"
        out.append((name, t.rstrip("?"), t.endswith("?")))
    return out


SCORERS = {
    "regression": textwrap.dedent('''\
        """Scorer for {task}. Runs only inside the score sandbox, next to the answers.

        df has the key column(s), the prediction column {pred!r} and the answer
        column {target!r}, one row per test example. Return exactly the metrics
        declared in contract.yaml, inside their declared bounds.
        """
        import numpy as np
        from scipy import stats

        RMSE_CAP = 1000.0  # keep in sync with metrics.rmse.max in contract.yaml


        def score(df):
            y = df[{target!r}].to_numpy(float)
            p = df[{pred!r}].to_numpy(float)

            def corr(fn):
                if np.std(y) == 0 or np.std(p) == 0:
                    return None  # undefined for constant vectors
                return float(fn(y, p)[0])

            return {{
                "spearman": corr(stats.spearmanr),
                "pearson": corr(stats.pearsonr),
                "rmse": min(float(np.sqrt(np.mean((y - p) ** 2))), RMSE_CAP),
            }}
        '''),
    "classification": textwrap.dedent('''\
        """Scorer for {task}. Runs only inside the score sandbox, next to the answers.

        df has the key column(s), the predicted probability {pred!r} (of the
        positive class) and the binary answer {target!r}. Return exactly the
        metrics declared in contract.yaml, inside their declared bounds.
        """
        import numpy as np
        from scipy.stats import rankdata


        def score(df):
            y = df[{target!r}].to_numpy().astype(bool)
            p = df[{pred!r}].to_numpy(float)
            n_pos, n_neg = int(y.sum()), int((~y).sum())
            if n_pos == 0 or n_neg == 0:
                auroc = None
            else:  # Mann-Whitney U
                r = rankdata(p)
                auroc = float((r[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
            return {{
                "auroc": auroc,
                "accuracy": float(np.mean((p >= 0.5) == y)),
            }}
        '''),
}

METRICS = {
    "regression": textwrap.dedent("""\
        metrics:
          # Bounds are required: they cap how much a single result can carry.
          - {name: spearman, type: float, nullable: true, min: -1, max: 1, goal: max}
          - {name: pearson, type: float, nullable: true, min: -1, max: 1, goal: max}
          - {name: rmse, type: float, min: 0, max: 1000, goal: min}
        headline: spearman
        """),
    "classification": textwrap.dedent("""\
        metrics:
          # Bounds are required: they cap how much a single result can carry.
          - {name: auroc, type: float, nullable: true, min: 0, max: 1, goal: max}
          - {name: accuracy, type: float, min: 0, max: 1, goal: max}
        headline: auroc
        """),
}


def add_task(task_id: str, tasks_dir: Path, inputs: Path, answers: Path, key: list[str],
             kind: str, target: str | None = None) -> Path:
    """Infer columns from the provider's files and write the task."""
    inp = pd.read_parquet(inputs)
    ans = pd.read_parquet(answers)
    for k in key:
        if k not in inp.columns or k not in ans.columns:
            raise SystemExit(f"key column {k!r} must be in both files")
    if target is None:
        rest = [c for c in ans.columns if c not in key]
        if len(rest) != 1:
            raise SystemExit(f"answers has columns {rest}; pick one with --target")
        target = rest[0]
    cols = lambda df, names: [(c, _type_of(df[c]), bool(df[c].isna().any())) for c in names]
    return write_task(task_id, tasks_dir, cols(inp, list(inp.columns)),
                      cols(ans, key + [target]), key, kind, target)


def write_task(task_id: str, tasks_dir: Path, inputs: list[tuple[str, str, bool]],
               answers: list[tuple[str, str, bool]], key: list[str], kind: str,
               target: str) -> Path:
    """Write tasks/<id>/contract.yaml and score.py from column lists."""
    if kind not in SCORERS:
        raise SystemExit(f"--kind must be one of {sorted(SCORERS)}")
    if target in {c[0] for c in inputs}:
        raise SystemExit(f"target {target!r} also appears in inputs: the model would see it")
    for k in key:
        if k not in {c[0] for c in inputs} or k not in {c[0] for c in answers}:
            raise SystemExit(f"key column {k!r} must be in both inputs and answers")
    pred = f"{target}_pred"

    d = tasks_dir / task_id
    if d.exists():
        raise SystemExit(f"{d} already exists")
    d.mkdir(parents=True)

    out_field = (f"  - {{name: {pred}, type: float, min: 0, max: 1, "
                 f"description: \"P({target} is true)\"}}" if kind == "classification"
                 else f"  - {{name: {pred}, type: float, description: \"predicted {target}\"}}")
    contract = "\n".join([
        f"# Task contract for {task_id}. Review every line.",
        "# Everything except `answers` is shown to the agent. Do not add min/max/allowed",
        "# values taken from the private data unless you are comfortable publishing them.",
        f"id: {task_id}",
        "version: 1",
        f"title: {task_id}",
        "description: \"\"",
        f"key: {yaml.safe_dump(key, default_flow_style=True).strip()}",
        "",
        "# Columns predict() receives. Drop any the model must not see.",
        "inputs:",
        *[_field_yaml(*c) for c in inputs],
        "",
        "# Columns predict() must return (plus the key).",
        "outputs:",
        out_field,
        "",
        "# Private. Only the scorer sees these; never shown to the agent.",
        "answers:",
        *[_field_yaml(*c) for c in answers],
        "",
        METRICS[kind].rstrip(),
        "",
        "export:",
        "  decimals: 3",
        "  # ladder_step: 0.01   # report only real improvements (recommended for final splits)",
        "",
        "limits:",
        "  predict_timeout_s: 1800",
        "  memory: 8g",
        "  cpus: 2",
        "  submissions_per_day: 20",
        "  submissions_total: 200     # lifetime cap across all agents; see `bmb-eval check`",
        "  release_after_s: 0         # >= sum of timeouts removes the timing channel",
        "",
    ])
    (d / "contract.yaml").write_text(contract)
    (d / "score.py").write_text(SCORERS[kind].format(task=task_id, pred=pred, target=target))
    Contract.load(d / "contract.yaml")  # fail now, not at first submission
    return d


def synth_inputs(c: Contract, n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Fake inputs that satisfy the public contract and nothing else. Safe to
    hand to anyone: generated from declared types and bounds only."""
    rng = np.random.default_rng(seed)
    cols = {}
    for f in c.inputs:
        if f.name in c.key:
            cols[f.name] = (np.arange(n) if f.type == "int" else [f"id{i:06d}" for i in range(n)])
            continue
        if f.allowed:
            v = rng.choice(np.array(f.allowed, dtype=object), n)
        elif f.type == "string":
            v = ["".join(rng.choice(AMINO, rng.integers(20, 60))) for _ in range(n)]
        elif f.type == "bool":
            v = rng.random(n) < 0.5
        else:
            lo = f.min if f.min is not None else 0.0
            hi = f.max if f.max is not None else (100.0 if f.type == "int" else 1.0)
            v = rng.uniform(lo, hi, n)
            v = np.round(v).astype(int) if f.type == "int" else v
        s = pd.Series(v)
        if f.nullable:
            s = s.astype(object).where(rng.random(n) > 0.05, None)
        cols[f.name] = s
    return pd.DataFrame(cols)


def example_private_data(dest: Path, n: int = 500, seed: int = 1) -> tuple[Path, Path]:
    """Stand-in 'private' data for the bundled example task."""
    rng = np.random.default_rng(seed)
    seqs = ["".join(rng.choice(AMINO, rng.integers(30, 80))) for _ in range(n)]
    hydro = np.array([sum(ch in "AILMFVW" for ch in s) / len(s) for s in seqs])
    fitness = 3 * hydro + rng.normal(0, 0.2, n)
    ids = [f"v{i:05d}" for i in range(n)]
    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"variant_id": ids, "sequence": seqs}).to_parquet(dest / "inputs.parquet")
    pd.DataFrame({"variant_id": ids, "fitness": fitness}).to_parquet(dest / "answers.parquet")
    return dest / "inputs.parquet", dest / "answers.parquet"
