import numpy as np
import pandas as pd

def predict(df):
    rows = []
    for path in df["patch"]:          # absolute paths inside the inputs folder
        m = np.load(path).mean(axis=(0, 1))   # mean R, G, B
        rows.append([m[0], m[1], m[2], m[0] - m[2], m[1] * 2])
    return pd.DataFrame({"expr": rows})
