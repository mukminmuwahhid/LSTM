"""Chronological train/val/test split, scaling (fit on train only) and
sliding-window sequence creation for the LSTM.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src import config


def chronological_split(df: pd.DataFrame, train_frac: float = config.TRAIN_FRAC, val_frac: float = config.VAL_FRAC):
    """Split a time-ordered DataFrame into train/val/test without shuffling,
    so no future information ever leaks into training or validation."""
    n = len(df)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    return train_df, val_df, test_df


def fit_scalers(train_df: pd.DataFrame, feature_cols: list[str], target_col: str = config.TARGET_COL):
    feature_scaler = MinMaxScaler()
    feature_scaler.fit(train_df[feature_cols])

    target_scaler = MinMaxScaler()
    target_scaler.fit(train_df[[target_col]])
    return feature_scaler, target_scaler


def apply_scalers(df: pd.DataFrame, feature_cols: list[str], feature_scaler, target_scaler, target_col: str = config.TARGET_COL):
    scaled_features = feature_scaler.transform(df[feature_cols])
    scaled_target = target_scaler.transform(df[[target_col]])
    return scaled_features, scaled_target.ravel()


def create_sequences(features: np.ndarray, target: np.ndarray, lookback: int = config.LOOKBACK, horizon: int = config.HORIZON):
    """Build sliding windows: X[i] = features[i : i+lookback],
    y[i] = target[i+lookback+horizon-1] (the value `horizon` steps after
    the end of the window)."""
    X, y = [], []
    last_start = len(features) - lookback - horizon + 1
    for i in range(last_start):
        X.append(features[i : i + lookback])
        y.append(target[i + lookback + horizon - 1])
    return np.array(X), np.array(y)


def prepare_datasets(featured_df: pd.DataFrame, feature_cols: list[str], save_scalers: bool = True):
    """Full orchestration: split -> fit scalers on train -> scale each split
    -> window into sequences. Returns a dict with X/y arrays for each split
    plus the fitted scalers."""
    train_df, val_df, test_df = chronological_split(featured_df)

    feature_scaler, target_scaler = fit_scalers(train_df, feature_cols)

    splits = {}
    for name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        feats, tgt = apply_scalers(split_df, feature_cols, feature_scaler, target_scaler)
        X, y = create_sequences(feats, tgt)
        splits[f"X_{name}"] = X
        splits[f"y_{name}"] = y
        splits[f"index_{name}"] = split_df.index[config.LOOKBACK + config.HORIZON - 1 :]

    if save_scalers:
        joblib.dump(feature_scaler, config.FEATURE_SCALER_PATH)
        joblib.dump(target_scaler, config.TARGET_SCALER_PATH)
        with open(config.FEATURE_LIST_PATH, "w") as f:
            json.dump(feature_cols, f, indent=2)

    splits["feature_scaler"] = feature_scaler
    splits["target_scaler"] = target_scaler
    splits["feature_cols"] = feature_cols
    return splits


if __name__ == "__main__":
    from src import feature_engineering

    featured, feature_cols = feature_engineering.feature_pipeline(save=False)
    data = prepare_datasets(featured, feature_cols)
    for split in ("train", "val", "test"):
        print(f"{split}: X={data[f'X_{split}'].shape} y={data[f'y_{split}'].shape}")
