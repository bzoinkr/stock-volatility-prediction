import pandas as pd
from pathlib import Path


def read_csv(path):
    return pd.read_csv(Path(path))


def write_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def read_parquet(path):
    return pd.read_parquet(Path(path))


def write_parquet(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
