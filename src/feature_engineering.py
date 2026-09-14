"""Turn the cleaned hourly DataFrame into a model-ready feature set:
cyclical time encodings, lag features and rolling statistics of the
target, built strictly from past values so no future information leaks
into a given row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hour = df.index.hour
    dow = df.index.dayofweek
    month = df.index.month

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["is_weekend"] = (dow >= 5).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, target_col: str = config.TARGET_COL, lags=config.LAG_HOURS) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, target_col: str = config.TARGET_COL, windows=config.ROLLING_WINDOWS) -> pd.DataFrame:
    df = df.copy()
    shifted = df[target_col].shift(1)  # exclude current row from its own rolling stat
    for w in windows:
        df[f"{target_col}_roll_mean_{w}"] = shifted.rolling(window=w, min_periods=w).mean()
        df[f"{target_col}_roll_std_{w}"] = shifted.rolling(window=w, min_periods=w).std()
    return df


def build_feature_set(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Apply all feature engineering steps and return the feature-engineered
    DataFrame plus the ordered list of feature-column names (target excluded).
    """
    out = df.copy()
    out = add_time_features(out)
    out = add_lag_features(out)
    out = add_rolling_features(out)

    # Rows at the start of the series don't have enough history for the
    # longest lag/rolling window — drop them rather than fill with 0/NaN.
    out = out.dropna()

    feature_cols = (
        list(config.WEATHER_FEATURE_COLS)
        + ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend"]
        + [f"{config.TARGET_COL}_lag_{lag}" for lag in config.LAG_HOURS]
        + [f"{config.TARGET_COL}_roll_mean_{w}" for w in config.ROLLING_WINDOWS]
        + [f"{config.TARGET_COL}_roll_std_{w}" for w in config.ROLLING_WINDOWS]
    )
    feature_cols = [c for c in feature_cols if c in out.columns]
    return out, feature_cols


def feature_pipeline(save: bool = True) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(config.CLEANED_DATA_PATH, index_col="timestamp", parse_dates=True)
    featured, feature_cols = build_feature_set(df)
    if save:
        featured.to_csv(config.FEATURED_DATA_PATH)
    return featured, feature_cols


if __name__ == "__main__":
    featured, feature_cols = feature_pipeline()
    print(f"Featured dataset: {featured.shape[0]} rows x {featured.shape[1]} cols")
    print(f"{len(feature_cols)} feature columns:")
    for c in feature_cols:
        print(" -", c)
    print(f"Saved to {config.FEATURED_DATA_PATH}")
