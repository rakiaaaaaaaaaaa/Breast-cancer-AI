import numpy as np
import pandas as pd
from typing import Tuple, Optional

def load_wdbc_csv_or_sklearn(fpath: str) -> pd.DataFrame:
    """
    Load Kaggle WDBC CSV if present; otherwise build a compatible DataFrame
    from sklearn.datasets.load_breast_cancer().
    """
    try:
        df = pd.read_csv(fpath)
        return df
    except Exception:
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        # Align column names to Kaggle style (approximate mapping)
        rename_map = {c: c.replace(" (mean)", "_mean")
                        .replace(" (worst)", "_worst")
                        .replace(" (se)", "_se")
                        .replace("mean radius", "radius_mean")
                        .replace("mean texture", "texture_mean") for c in X.columns}
        X = X.rename(columns=rename_map)
        y = pd.Series(data.target).map({0:"malignant", 1:"benign"})
        df = pd.concat([pd.Series(range(1, len(X)+1), name="id"),
                        y.rename("diagnosis"),
                        X], axis=1)
        # Add the empty column to match Kaggle sample
        df["Unnamed: 32"] = np.nan
        # Match Kaggle's label encoding (M/B)
        df["diagnosis"] = df["diagnosis"].map({"malignant":"M","benign":"B"})
        return df

def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Drop non-feature columns and NaNs-only column if present
    drops = [c for c in ["id", "Unnamed: 32"] if c in df.columns]
    df = df.drop(columns=drops, errors="ignore")
    df = df.dropna(axis=1, how="all")
    # Encode target
    if "diagnosis" not in df.columns:
        raise ValueError("Expected 'diagnosis' column.")
    y = df["diagnosis"].map({"M":1, "B":0}).astype(int)
    X = df.drop(columns=["diagnosis"])
    # Ensure consistent column order (sort for reproducibility)
    X = X.reindex(sorted(X.columns), axis=1)
    return X, y
