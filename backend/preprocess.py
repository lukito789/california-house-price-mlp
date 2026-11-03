"""Preprocessing bersama untuk training & inference (supaya identik)."""
import numpy as np
import pandas as pd

NUMERIC = ["longitude", "latitude", "housing_median_age", "total_rooms",
           "total_bedrooms", "population", "households", "median_income"]
LOG_COLS = ["total_rooms", "total_bedrooms", "population", "households"]
CATEGORIES = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
TARGET = "median_house_value"
TARGET_SCALE = 100_000.0  # target dilatih dalam satuan $100k


def _raw(df: pd.DataFrame, fill: dict) -> pd.DataFrame:
    X = df[NUMERIC].astype(float).copy()
    for c, v in fill.items():
        X[c] = X[c].fillna(v)
    for c in LOG_COLS:
        X[c] = np.log1p(X[c])
    for cat in CATEGORIES:
        X["ocean_" + cat] = (df["ocean_proximity"] == cat).astype(float)
    return X


def fit_prep(train_df: pd.DataFrame) -> dict:
    fill = {c: float(train_df[c].median()) for c in NUMERIC}
    X = _raw(train_df, fill)
    std = X.std().replace(0, 1.0).fillna(1.0)
    return {"fill": fill, "columns": list(X.columns),
            "mean": X.mean().tolist(), "std": std.tolist()}


def transform(df: pd.DataFrame, prep: dict) -> np.ndarray:
    X = _raw(df, prep["fill"])[prep["columns"]]
    return ((X.values - np.array(prep["mean"])) / np.array(prep["std"])).astype("float32")
