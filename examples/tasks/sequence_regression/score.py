"""Scorer for example_regression. Runs only inside the score sandbox, next to the answers.

df has the key column(s), the prediction column 'fitness_pred' and the answer
column 'fitness', one row per test example. Return exactly the metrics
declared in contract.yaml, inside their declared bounds.
"""
import numpy as np
from scipy import stats

RMSE_CAP = 10.0  # keep in sync with metrics.rmse.max in contract.yaml


def score(df):
    y = df['fitness'].to_numpy(float)
    p = df['fitness_pred'].to_numpy(float)

    def corr(fn):
        if np.std(y) == 0 or np.std(p) == 0:
            return None  # undefined for constant vectors
        return float(fn(y, p)[0])

    return {
        "spearman": corr(stats.spearmanr),
        "pearson": corr(stats.pearsonr),
        "rmse": min(float(np.sqrt(np.mean((y - p) ** 2))), RMSE_CAP),
    }
