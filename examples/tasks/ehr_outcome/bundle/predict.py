import numpy as np
import pandas as pd

COUNTS = {}

def load_model(inputs_dir):
    ev = pd.read_parquet(inputs_dir / "events.parquet")
    COUNTS.update(ev.groupby("patient_id").size().to_dict())

def predict(df):
    base = np.clip(np.array([COUNTS.get(p, 0) for p in df["patient_id"]]) / 20, 0, 1)
    return pd.DataFrame({"risk": [[b * 0.2, b * 0.5, b] for b in base]})
