"""Export the trained model's weights, scalers, historical data, and
precomputed dashboard aggregates as static JSON files for the GitHub
Pages build (docs/), which runs everything client-side via TensorFlow.js
instead of hitting a Flask backend.

Run from the project root with:
    python -m src.export_web
"""
from __future__ import annotations

import json

import joblib
import pandas as pd
from tensorflow import keras

from src import config

DOCS_DIR = config.ROOT_DIR / "docs"
DATA_OUT_DIR = DOCS_DIR / "data"
MODEL_OUT_DIR = DOCS_DIR / "model"


def export_model_weights():
    model = keras.models.load_model(config.MODEL_PATH)
    layers_out = []
    for layer in model.layers:
        cfg = layer.get_config()
        weights = [w.tolist() for w in layer.get_weights()]
        entry = {"class_name": layer.__class__.__name__, "name": layer.name, "weights": weights}
        if entry["class_name"] == "LSTM":
            entry["units"] = cfg["units"]
            entry["return_sequences"] = cfg["return_sequences"]
        elif entry["class_name"] == "Dense":
            entry["units"] = cfg["units"]
            entry["activation"] = cfg["activation"]
        layers_out.append(entry)

    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {"input_shape": list(model.input_shape[1:]), "layers": layers_out}
    with open(MODEL_OUT_DIR / "weights.json", "w") as f:
        json.dump(out, f)
    size_kb = (MODEL_OUT_DIR / "weights.json").stat().st_size / 1024
    print(f"Exported {len(layers_out)} layers ({size_kb:.0f} KB) to {MODEL_OUT_DIR / 'weights.json'}")


def export_scalers_and_features():
    feature_scaler = joblib.load(config.FEATURE_SCALER_PATH)
    target_scaler = joblib.load(config.TARGET_SCALER_PATH)
    with open(config.FEATURE_LIST_PATH) as f:
        feature_cols = json.load(f)

    DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_OUT_DIR / "scalers.json", "w") as f:
        json.dump({
            "feature_cols": feature_cols,
            "feature_scale": feature_scaler.scale_.tolist(),
            "feature_min": feature_scaler.min_.tolist(),
            "target_scale": target_scaler.scale_.tolist(),
            "target_min": target_scaler.min_.tolist(),
        }, f)
    print(f"Exported scalers to {DATA_OUT_DIR / 'scalers.json'}")
    return feature_cols


def export_history():
    cleaned = pd.read_csv(config.CLEANED_DATA_PATH, index_col="timestamp", parse_dates=True)
    cols = [config.TARGET_COL] + list(config.WEATHER_FEATURE_COLS)
    records = {"timestamps": [t.isoformat() for t in cleaned.index]}
    for c in cols:
        records[c] = [round(float(v), 4) for v in cleaned[c]]

    with open(DATA_OUT_DIR / "history.json", "w") as f:
        json.dump(records, f)
    size_kb = (DATA_OUT_DIR / "history.json").stat().st_size / 1024
    print(f"Exported {len(cleaned)} hourly rows ({size_kb:.0f} KB) to {DATA_OUT_DIR / 'history.json'}")


def export_dashboard_aggregates():
    cleaned = pd.read_csv(config.CLEANED_DATA_PATH, index_col="timestamp", parse_dates=True)
    featured = pd.read_csv(config.FEATURED_DATA_PATH, index_col="timestamp", parse_dates=True)
    tgt = cleaned[config.TARGET_COL]

    with open(config.METRICS_DIR / "test_metrics.json") as f:
        test_metrics = json.load(f)

    overview = {
        "n_rows": int(len(cleaned)),
        "date_start": cleaned.index.min().isoformat(),
        "date_end": cleaned.index.max().isoformat(),
        "target_mean": float(tgt.mean()),
        "target_std": float(tgt.std()),
        "target_min": float(tgt.min()),
        "target_max": float(tgt.max()),
        "test_metrics": test_metrics,
    }
    with open(DATA_OUT_DIR / "overview.json", "w") as f:
        json.dump(overview, f)

    daily = tgt.resample("D").agg(["mean", "max", "min"])
    timeseries = {
        "timestamps": [d.date().isoformat() for d in daily.index],
        "mean": [None if pd.isna(v) else round(float(v), 4) for v in daily["mean"]],
        "max": [None if pd.isna(v) else round(float(v), 4) for v in daily["max"]],
        "min": [None if pd.isna(v) else round(float(v), 4) for v in daily["min"]],
    }
    with open(DATA_OUT_DIR / "timeseries.json", "w") as f:
        json.dump(timeseries, f)

    hourly = cleaned.groupby(cleaned.index.hour)[config.TARGET_COL].mean()
    dow = cleaned.groupby(cleaned.index.dayofweek)[config.TARGET_COL].mean()
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    month_period = cleaned.index.to_period("M")
    month_avg = cleaned.groupby(month_period)[config.TARGET_COL].mean().sort_index()
    seasonality = {
        "hourly": {"labels": [f"{h:02d}h" for h in range(24)], "values": [round(float(hourly.get(h, 0)), 4) for h in range(24)]},
        "dow": {"labels": dow_names, "values": [round(float(dow.get(d, 0)), 4) for d in range(7)]},
        "month": {"labels": [p.strftime("%b %y") for p in month_avg.index], "values": [round(float(v), 4) for v in month_avg.values]},
    }
    with open(DATA_OUT_DIR / "seasonality.json", "w") as f:
        json.dump(seasonality, f)

    weather_time_cols = list(config.WEATHER_FEATURE_COLS) + [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend",
    ]
    lag_cols = [c for c in featured.columns if "_lag_" in c or "_roll_" in c]
    corr_wt = featured[weather_time_cols + [config.TARGET_COL]].corr()[config.TARGET_COL].drop(config.TARGET_COL)
    corr_wt = corr_wt.reindex(corr_wt.abs().sort_values(ascending=False).index)
    corr_lag = featured[lag_cols + [config.TARGET_COL]].corr()[config.TARGET_COL].drop(config.TARGET_COL)
    corr_lag = corr_lag.reindex(corr_lag.abs().sort_values(ascending=False).index)
    correlation = {
        "weather_time": {"labels": corr_wt.index.tolist(), "values": [round(float(v), 4) for v in corr_wt.values]},
        "lag": {"labels": corr_lag.index.tolist(), "values": [round(float(v), 4) for v in corr_lag.values]},
    }
    with open(DATA_OUT_DIR / "correlation.json", "w") as f:
        json.dump(correlation, f)

    imp_path = config.METRICS_DIR / "permutation_importance.json"
    if imp_path.exists():
        with open(imp_path) as f:
            imp = json.load(f)
        items = list(imp["importances"].items())[:15]
        importance = {
            "available": True,
            "baseline_mae": imp["baseline_mae"],
            "labels": [k for k, _ in items],
            "values": [round(v, 5) for _, v in items],
        }
    else:
        importance = {"available": False}
    with open(DATA_OUT_DIR / "importance.json", "w") as f:
        json.dump(importance, f)

    test_pred_df = pd.read_csv(config.METRICS_DIR / "test_predictions.csv", parse_dates=["timestamp"])
    n = min(24 * 14, len(test_pred_df))
    subset = test_pred_df.iloc[:n]
    test_predictions = {
        "timestamps": [t.isoformat() for t in subset["timestamp"]],
        "actual": [round(float(v), 4) for v in subset["actual"]],
        "predicted": [round(float(v), 4) for v in subset["predicted"]],
        "scatter_actual": [round(float(v), 4) for v in test_pred_df["actual"]],
        "scatter_predicted": [round(float(v), 4) for v in test_pred_df["predicted"]],
    }
    with open(DATA_OUT_DIR / "test_predictions.json", "w") as f:
        json.dump(test_predictions, f)

    print(f"Exported dashboard aggregates to {DATA_OUT_DIR}")


def export_meta(feature_cols):
    meta = {
        "lookback": config.LOOKBACK,
        "horizon": config.HORIZON,
        "target_col": config.TARGET_COL,
        "weather_cols": list(config.WEATHER_FEATURE_COLS),
        "lag_hours": list(config.LAG_HOURS),
        "rolling_windows": list(config.ROLLING_WINDOWS),
        "feature_cols": feature_cols,
        "max_forecast_horizon": 72,
    }
    with open(DATA_OUT_DIR / "meta.json", "w") as f:
        json.dump(meta, f)
    print(f"Exported meta to {DATA_OUT_DIR / 'meta.json'}")


if __name__ == "__main__":
    export_model_weights()
    feature_cols = export_scalers_and_features()
    export_history()
    export_dashboard_aggregates()
    export_meta(feature_cols)
    print("\nAll web export files written under docs/")
