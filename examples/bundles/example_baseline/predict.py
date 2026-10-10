"""Example bundle for the example_regression task.

A bundle is a directory (sent as .tar.gz) containing:
    predict.py   required: predict(df) -> DataFrame or list, one row per input row
                 optional: load_model(), called once before any predict()
    weights/     optional: anything predict.py loads (paths relative to this file)
    wheels/      optional: *.whl installed offline before predict.py is imported

predict() gets batches of the task's input columns (see the public contract)
and must return the output columns. No network is available.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
W = {}


def load_model():
    W.update(json.loads((HERE / "weights" / "model.json").read_text()))


def predict(df: pd.DataFrame) -> pd.DataFrame:
    frac = df["sequence"].map(lambda s: sum(ch in W["residues"] for ch in s) / len(s))
    return pd.DataFrame({"fitness_pred": W["slope"] * frac.to_numpy(float) + W["intercept"]})
