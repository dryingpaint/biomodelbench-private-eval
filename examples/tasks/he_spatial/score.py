import numpy as np


def score(df):
    p, y = np.stack(df["expr"]), np.stack(df["expr_true"])
    rs = [np.corrcoef(p[:, g], y[:, g])[0, 1] for g in range(y.shape[1])
          if p[:, g].std() > 0 and y[:, g].std() > 0]
    return {"pearson_per_gene_mean": float(np.mean(rs)) if rs else None,
            "mae": min(float(np.mean(np.abs(p - y))), 20.0)}
