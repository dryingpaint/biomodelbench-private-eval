"""Shared test helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples" / "tasks"
AMINO = list("ACDEFGHIKLMNPQRSTVWY")


def sequence_data(dest: Path, n: int = 500, seed: int = 1) -> tuple[Path, Path]:
    """Stand-in private data for examples/tasks/sequence_regression."""
    rng = np.random.default_rng(seed)
    seqs = ["".join(rng.choice(AMINO, rng.integers(30, 80))) for _ in range(n)]
    hydro = np.array([sum(ch in "AILMFVW" for ch in s) / len(s) for s in seqs])
    fitness = 3 * hydro + rng.normal(0, 0.2, n)
    ids = [f"v{i:05d}" for i in range(n)]
    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"variant_id": ids, "sequence": seqs}).to_parquet(dest / "inputs.parquet")
    pd.DataFrame({"variant_id": ids, "fitness": fitness}).to_parquet(dest / "answers.parquet")
    return dest / "inputs.parquet", dest / "answers.parquet"
