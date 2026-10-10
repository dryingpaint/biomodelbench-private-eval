import pandas as pd

def predict(df):
    # Toy model: the precursor plus two fixed neutral losses.
    return pd.DataFrame({"peaks": [
        {"mz": [m, m - 18.0, m - 44.0], "intensity": [1.0, 0.5, 0.2]} for m in df["precursor_mz"]]})
