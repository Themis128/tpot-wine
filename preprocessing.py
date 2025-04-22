import pandas as pd

def load_and_clean_csv(path: str) -> pd.DataFrame:
    """Load CSV and drop all-empty or all-blank columns."""
    df = pd.read_csv(path)
    original_cols = df.columns.tolist()

    # Drop all-NaN columns
    df = df.dropna(axis=1, how='all')

    # Drop all-empty-string columns
    df = df.loc[:, ~(df == "").all()]

    # Optional: log dropped columns
    dropped = set(original_cols) - set(df.columns)
    if dropped:
        print(f"[INFO] Dropped empty columns: {dropped}")

    return df
