"""Flask web app: dashboard + interactive multi-hour prediction UI for the
LSTM electricity-consumption model.

Run from the project root with:
    python app.py
Then open http://127.0.0.1:5000
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from src import config, forecast

app = Flask(__name__)

_state: dict = {}


def load_state():
    if not _state:
        _state["cleaned"] = pd.read_csv(config.CLEANED_DATA_PATH, index_col="timestamp", parse_dates=True)
        _state["featured"] = pd.read_csv(config.FEATURED_DATA_PATH, index_col="timestamp", parse_dates=True)
        with open(config.METRICS_DIR / "test_metrics.json") as f:
            _state["test_metrics"] = json.load(f)
        imp_path = config.METRICS_DIR / "permutation_importance.json"
        _state["importance"] = json.load(open(imp_path)) if imp_path.exists() else None
    return _state


@app.route("/")
def dashboard():
    return render_template("dashboard.html", active="dashboard")


@app.route("/predict")
def predict_page():
    return render_template("predict.html", active="predict")


@app.route("/api/overview")
def api_overview():
    s = load_state()
    cleaned = s["cleaned"]
    tgt = cleaned[config.TARGET_COL]
    return jsonify({
        "n_rows": int(len(cleaned)),
        "date_start": cleaned.index.min().isoformat(),
        "date_end": cleaned.index.max().isoformat(),
        "target_mean": float(tgt.mean()),
        "target_std": float(tgt.std()),
        "target_min": float(tgt.min()),
        "target_max": float(tgt.max()),
        "test_metrics": s["test_metrics"],
    })


@app.route("/api/timeseries")
def api_timeseries():
    s = load_state()
    daily = s["cleaned"][config.TARGET_COL].resample("D").agg(["mean", "max", "min"])
    return jsonify({
        "timestamps": [d.date().isoformat() for d in daily.index],
        "mean": [None if pd.isna(v) else round(float(v), 4) for v in daily["mean"]],
        "max": [None if pd.isna(v) else round(float(v), 4) for v in daily["max"]],
        "min": [None if pd.isna(v) else round(float(v), 4) for v in daily["min"]],
    })


@app.route("/api/seasonality")
def api_seasonality():
    s = load_state()
    df = s["cleaned"]

    hourly = df.groupby(df.index.hour)[config.TARGET_COL].mean()
    dow = df.groupby(df.index.dayofweek)[config.TARGET_COL].mean()
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    month_period = df.index.to_period("M")
    month_avg = df.groupby(month_period)[config.TARGET_COL].mean().sort_index()

    return jsonify({
        "hourly": {"labels": [f"{h:02d}h" for h in range(24)], "values": [round(float(hourly.get(h, 0)), 4) for h in range(24)]},
        "dow": {"labels": dow_names, "values": [round(float(dow.get(d, 0)), 4) for d in range(7)]},
        "month": {"labels": [p.strftime("%b %y") for p in month_avg.index], "values": [round(float(v), 4) for v in month_avg.values]},
    })


@app.route("/api/correlation")
def api_correlation():
    s = load_state()
    df = s["featured"]

    weather_time_cols = list(config.WEATHER_FEATURE_COLS) + [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend",
    ]
    lag_cols = [c for c in df.columns if "_lag_" in c or "_roll_" in c]

    corr_wt = df[weather_time_cols + [config.TARGET_COL]].corr()[config.TARGET_COL].drop(config.TARGET_COL)
    corr_wt = corr_wt.reindex(corr_wt.abs().sort_values(ascending=False).index)

    corr_lag = df[lag_cols + [config.TARGET_COL]].corr()[config.TARGET_COL].drop(config.TARGET_COL)
    corr_lag = corr_lag.reindex(corr_lag.abs().sort_values(ascending=False).index)

    return jsonify({
        "weather_time": {"labels": corr_wt.index.tolist(), "values": [round(float(v), 4) for v in corr_wt.values]},
        "lag": {"labels": corr_lag.index.tolist(), "values": [round(float(v), 4) for v in corr_lag.values]},
    })


@app.route("/api/importance")
def api_importance():
    s = load_state()
    if not s["importance"]:
        return jsonify({"available": False})
    imp = s["importance"]["importances"]
    items = list(imp.items())[:15]
    return jsonify({
        "available": True,
        "baseline_mae": s["importance"]["baseline_mae"],
        "labels": [k for k, _ in items],
        "values": [round(v, 5) for _, v in items],
    })


@app.route("/api/test-predictions")
def api_test_predictions():
    df = pd.read_csv(config.METRICS_DIR / "test_predictions.csv", parse_dates=["timestamp"])
    n = min(24 * 14, len(df))
    subset = df.iloc[:n]
    return jsonify({
        "timestamps": [t.isoformat() for t in subset["timestamp"]],
        "actual": [round(float(v), 4) for v in subset["actual"]],
        "predicted": [round(float(v), 4) for v in subset["predicted"]],
        "scatter_actual": [round(float(v), 4) for v in df["actual"]],
        "scatter_predicted": [round(float(v), 4) for v in df["predicted"]],
    })


@app.route("/api/valid-range")
def api_valid_range():
    min_start, max_end, max_horizon = forecast.get_bounds()
    return jsonify({"min_start": min_start.isoformat(), "max_end": max_end.isoformat(), "max_horizon": max_horizon})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    body = request.get_json(force=True, silent=True) or {}
    start = body.get("start")
    horizon = body.get("horizon", 1)
    if not start:
        return jsonify({"error": "missing 'start' timestamp"}), 400
    try:
        horizon = max(1, min(int(horizon), 72))
    except (TypeError, ValueError):
        return jsonify({"error": "'horizon' must be an integer"}), 400

    try:
        points = forecast.recursive_forecast(start, horizon)
    except (ValueError, KeyError) as e:
        return jsonify({"error": str(e)}), 400

    actuals = [p["actual"] for p in points]
    metrics = None
    if all(a is not None for a in actuals):
        a = np.array(actuals)
        p = np.array([pt["predicted"] for pt in points])
        metrics = {"mae": round(float(np.mean(np.abs(a - p))), 4), "rmse": round(float(np.sqrt(np.mean((a - p) ** 2))), 4)}

    return jsonify({"points": points, "metrics": metrics})


if __name__ == "__main__":
    load_state()
    forecast._load_artifacts()
    app.run(debug=False, host="127.0.0.1", port=5000)
