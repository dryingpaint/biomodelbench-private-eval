import numpy as np


def score(df):
    p, y = np.stack(df["delta"]), np.stack(df["delta_true"])
    rs = [np.corrcoef(a, b)[0, 1] for a, b in zip(p, y) if a.std() > 0 and b.std() > 0]
    return {"pearson_mean": float(np.mean(rs)) if rs else None,
            "mae": min(float(np.mean(np.abs(p - y))), 50.0)}
