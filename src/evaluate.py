"""Evaluate the trained model on the held-out test split.

Run from the project root with:
    python -m src.evaluate
"""
from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras

from src import config, dataset


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    # avoid divide-by-zero: exclude near-zero true values from MAPE
    nonzero = np.abs(y_true) > 1e-6
    mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE_%": float(mape), "R2": float(r2)}


def plot_predictions(index, y_true, y_pred, out_path, n_points: int = 24 * 14):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(index[:n_points], y_true[:n_points], label="actual")
    ax.plot(index[:n_points], y_pred[:n_points], label="predicted", alpha=0.8)
    ax.set_title("Test set: actual vs predicted electricity consumption (first 14 days)")
    ax.set_xlabel("timestamp")
    ax.set_ylabel(config.TARGET_COL)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scatter(y_true, y_pred, out_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=6, alpha=0.3)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("actual")
    ax.set_ylabel("predicted")
    ax.set_title("Predicted vs actual")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    model = keras.models.load_model(config.MODEL_PATH)
    feature_scaler = joblib.load(config.FEATURE_SCALER_PATH)
    target_scaler = joblib.load(config.TARGET_SCALER_PATH)
    with open(config.FEATURE_LIST_PATH) as f:
        feature_cols = json.load(f)

    featured = pd.read_csv(config.FEATURED_DATA_PATH, index_col="timestamp", parse_dates=True)
    _, _, test_df = dataset.chronological_split(featured)

    feats, tgt = dataset.apply_scalers(test_df, feature_cols, feature_scaler, target_scaler)
    X_test, y_test_scaled = dataset.create_sequences(feats, tgt)
    test_index = test_df.index[config.LOOKBACK + config.HORIZON - 1 :]

    y_pred_scaled = model.predict(X_test, verbose=0).ravel()

    y_test = target_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    metrics = compute_metrics(y_test, y_pred)
    print("Test set metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    with open(config.METRICS_DIR / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    results_df = pd.DataFrame({"timestamp": test_index, "actual": y_test, "predicted": y_pred})
    results_df.to_csv(config.METRICS_DIR / "test_predictions.csv", index=False)

    plot_predictions(test_index, y_test, y_pred, config.FIGURES_DIR / "test_predictions.png")
    plot_scatter(y_test, y_pred, config.FIGURES_DIR / "test_scatter.png")

    print(f"\nMetrics saved to {config.METRICS_DIR / 'test_metrics.json'}")
    print(f"Predictions saved to {config.METRICS_DIR / 'test_predictions.csv'}")
    print(f"Plots saved to {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
