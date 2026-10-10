import numpy as np
from scipy.stats import spearmanr

ENDPOINTS = ["solubility", "logD", "clearance"]


def score(df):
    out, rhos = {}, []
    for e in ENDPOINTS:
        g = df[df["endpoint"] == e]
        y, p = g["measured"].to_numpy(float), g["value"].to_numpy(float)
        out[f"mae[{e}]"] = min(float(np.mean(np.abs(y - p))), 100.0) if len(g) else 0.0
        if len(g) > 2 and np.std(y) > 0 and np.std(p) > 0:
            rhos.append(spearmanr(y, p)[0])
    out["spearman_macro"] = float(np.mean(rhos)) if rhos else None
    return out
