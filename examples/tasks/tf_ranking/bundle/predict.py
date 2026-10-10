import pandas as pd

def predict(df):
    # Toy model: alphabetical order of the permitted candidates.
    return pd.DataFrame({"ranking": [
        {"tf": sorted(c)[:10], "score": [1.0 / (i + 1) for i in range(min(10, len(c)))]}
        for c in df["candidates"]]})
