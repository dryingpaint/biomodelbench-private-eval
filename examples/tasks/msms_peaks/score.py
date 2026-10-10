import numpy as np

BIN = 0.5      # Da; fixed tolerance for the binned cosine
TOP = 5000.0   # highest m/z considered


def _vec(peaks, top):
    v = np.zeros(int(top / BIN) + 2)
    np.add.at(v, (np.asarray(peaks["mz"], float) / BIN).astype(int).clip(0, len(v) - 1),
              np.asarray(peaks["intensity"], float))
    return v


def score(df):
    cos = []
    for p, t in zip(df["peaks"], df["true_peaks"]):
        a, b = _vec(p, TOP), _vec(t, TOP)
        n = np.linalg.norm(a) * np.linalg.norm(b)
        cos.append(float(a @ b / n) if n > 0 else 0.0)
    return {"cosine_mean": float(np.mean(cos))}
