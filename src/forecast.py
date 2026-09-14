"""Recursive multi-step forecasting built on top of the single-step LSTM.

The trained model predicts exactly 1 hour ahead from a 24-hour lookback
window. To forecast an arbitrary horizon ("any hour"), this module feeds
each prediction back in as if it were an observed value, recomputing lag
and rolling features from a hybrid actual+predicted history so later steps
never see true future electricity — only what a real deployment would have
at that point (its own prior forecasts).

Weather and calendar features for the forecast window are taken from the
historical record, standing in for a live weather-forecast feed (this
dataset has no forecast weather). That means valid forecasts are limited to
timestamps inside the historical range — this module doubles as an
interactive backtesting tool as well as a forecaster.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from src import config

_cache: dict = {}


def _load_artifacts():
    if not _cache:
        from tensorflow import keras

        _cache["model"] = keras.models.load_model(config.MODEL_PATH)
        _cache["feature_scaler"] = joblib.load(config.FEATURE_SCALER_PATH)
        _cache["target_scaler"] = joblib.load(config.TARGET_SCALER_PATH)
        with open(config.FEATURE_LIST_PATH) as f:
            _cache["feature_cols"] = json.load(f)
        _cache["cleaned"] = pd.read_csv(config.CLEANED_DATA_PATH, index_col="timestamp", parse_dates=True)
        _cache["featured"] = pd.read_csv(config.FEATURED_DATA_PATH, index_col="timestamp", parse_dates=True)
    return _cache


def _time_features(ts: pd.Timestamp) -> dict:
    hour, dow, month = ts.hour, ts.dayofweek, ts.month
    return {
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "is_weekend": 1 if dow >= 5 else 0,
    }


def get_bounds(max_horizon: int = 72):
    """Return (min_start, max_end, max_horizon): the range of timestamps a
    forecast can legally start/end at, given the historical data on hand."""
    art = _load_artifacts()
    min_start = art["featured"].index.min() + pd.Timedelta(hours=config.LOOKBACK)
    max_end = art["cleaned"].index.max()
    return min_start, max_end, max_horizon


def recursive_forecast(start: str | pd.Timestamp, horizon: int) -> list[dict]:
    art = _load_artifacts()
    model = art["model"]
    feature_scaler = art["feature_scaler"]
    target_scaler = art["target_scaler"]
    feature_cols = art["feature_cols"]
    cleaned = art["cleaned"]
    featured = art["featured"]

    start_ts = pd.Timestamp(start)
    min_start = featured.index.min() + pd.Timedelta(hours=config.LOOKBACK)
    max_end = cleaned.index.max()

    if start_ts < min_start or start_ts > max_end:
        raise ValueError(f"start must be between {min_start} and {max_end}")

    end_ts = start_ts + pd.Timedelta(hours=horizon - 1)
    if end_ts > max_end:
        max_horizon_here = int((max_end - start_ts) / pd.Timedelta(hours=1)) + 1
        raise ValueError(
            f"horizon too large for this start time — historical data (used as the weather-forecast "
            f"stand-in) only extends to {max_end}. Max horizon from this start is {max_horizon_here}h."
        )

    elec_actual = cleaned[config.TARGET_COL]
    predicted: dict[pd.Timestamp, float] = {}
    feat_cache: dict[pd.Timestamp, np.ndarray] = {}

    def elec_value(ts: pd.Timestamp) -> float:
        if ts in predicted:
            return predicted[ts]
        return float(elec_actual.loc[ts])

    def get_feature_row(ts: pd.Timestamp) -> np.ndarray:
        if ts in feat_cache:
            return feat_cache[ts]
        if ts < start_ts:
            row = featured.loc[ts, feature_cols].to_numpy(dtype=float)
            feat_cache[ts] = row
            return row

        row_dict = {c: float(cleaned.loc[ts, c]) for c in config.WEATHER_FEATURE_COLS}
        row_dict.update(_time_features(ts))
        for lag in config.LAG_HOURS:
            row_dict[f"{config.TARGET_COL}_lag_{lag}"] = elec_value(ts - pd.Timedelta(hours=lag))
        for w in config.ROLLING_WINDOWS:
            vals = [elec_value(ts - pd.Timedelta(hours=k)) for k in range(1, w + 1)]
            row_dict[f"{config.TARGET_COL}_roll_mean_{w}"] = float(np.mean(vals))
            row_dict[f"{config.TARGET_COL}_roll_std_{w}"] = float(np.std(vals, ddof=1))

        row = np.array([row_dict[c] for c in feature_cols], dtype=float)
        feat_cache[ts] = row
        return row

    results = []
    for h in range(horizon):
        target_ts = start_ts + pd.Timedelta(hours=h)
        window_ts = [target_ts - pd.Timedelta(hours=k) for k in range(config.LOOKBACK, 0, -1)]
        window_feats = pd.DataFrame(np.stack([get_feature_row(t) for t in window_ts]), columns=feature_cols)
        scaled = feature_scaler.transform(window_feats)
        X = scaled[np.newaxis, :, :]
        y_scaled = float(np.asarray(model(X, training=False)).ravel()[0])
        y_pred = float(target_scaler.inverse_transform([[y_scaled]])[0, 0])
        predicted[target_ts] = y_pred

        actual_val = float(elec_actual.loc[target_ts]) if target_ts in elec_actual.index else None
        results.append({"timestamp": target_ts.isoformat(), "predicted": round(y_pred, 4), "actual": actual_val})

    return results


if __name__ == "__main__":
    min_start, max_end, max_h = get_bounds()
    demo_start = max_end - pd.Timedelta(hours=48)
    print(f"valid range: {min_start} -> {max_end}")
    out = recursive_forecast(demo_start, horizon=24)
    for row in out[:5]:
        print(row)
    print("...")
    errs = [abs(r["predicted"] - r["actual"]) for r in out if r["actual"] is not None]
    print(f"MAE over demo window: {np.mean(errs):.4f}")
