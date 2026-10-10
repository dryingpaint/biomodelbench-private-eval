import numpy as np


def score(df):
    ndcg, hit = [], []
    for pred, truth in zip(df["ranking"], df["true_tfs"]):
        order = [t for _, t in sorted(zip(-np.asarray(pred["score"], float), pred["tf"]))]
        true = set(truth["tf"])
        gains = [1 / np.log2(i + 2) for i, t in enumerate(order[:10]) if t in true]
        ideal = sum(1 / np.log2(i + 2) for i in range(min(10, len(true))))
        ndcg.append(sum(gains) / ideal if ideal else 0.0)
        hit.append(float(any(t in true for t in order[:5])))
    return {"ndcg_at_10": float(np.mean(ndcg)), "hit_at_5": float(np.mean(hit))}
