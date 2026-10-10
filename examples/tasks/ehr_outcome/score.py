import numpy as np

HORIZONS = {"d30": 30, "d90": 90, "d365": 365}


def score(df):
    risk = np.stack(df["risk"])            # patients x 3
    t = df["followup_days"].to_numpy(float)
    e = df["had_event"].to_numpy(int)
    out = {}
    for j, (name, h) in enumerate(HORIZONS.items()):
        known = (e == 1) | (t >= h)         # drop patients censored before the horizon
        y = ((e == 1) & (t <= h))[known].astype(float)
        out[f"brier[{name}]"] = float(np.mean((risk[known, j] - y) ** 2)) if known.any() else 0.0
    out["brier_mean"] = float(np.mean(list(out.values())))
    return out
