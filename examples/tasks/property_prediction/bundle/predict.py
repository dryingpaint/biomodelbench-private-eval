import pandas as pd

def predict(df):
    # Toy model: longer molecules -> larger value.
    return pd.DataFrame({"value": df["smiles"].str.len() / 10.0})
