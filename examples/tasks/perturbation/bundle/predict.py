import numpy as np
import pandas as pd

MEAN = {}

def load_model(inputs_dir):
    MEAN["control"] = np.load(inputs_dir / "control_expression.npy").mean(axis=0)

def predict(df):
    sign = df["modality"].map({"KO": -1, "CRISPRi": -1, "CRISPRa": 1, "OE": 1}).to_numpy()
    return pd.DataFrame({"delta": [s * 0.1 * MEAN["control"] for s in sign]})
