"""Permutation feature importance on the trained model, evaluated on the
test split: for each feature, shuffle it across samples (breaking its link
to the target while preserving every other feature) and measure how much
test MAE degrades. A bigger degradation means the model leans on that
feature more — this is what actually drives the model's predictions,
as opposed to a raw correlation which only reflects the data.

Run from the project root with:
    python -m src.importance
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from tensorflow import keras

from src import config, dataset


def main(n_repeats: int = 5, seed: int = config.RANDOM_SEED):
    rng = np.random.default_rng(seed)

    model = keras.models.load_model(config.MODEL_PATH)
    feature_scaler = joblib.load(config.FEATURE_SCALER_PATH)
    target_scaler = joblib.load(config.TARGET_SCALER_PATH)
    with open(config.FEATURE_LIST_PATH) as f:
        feature_cols = json.load(f)

    featured = pd.read_csv(config.FEATURED_DATA_PATH, index_col="timestamp", parse_dates=True)
    _, _, test_df = dataset.chronological_split(featured)
    feats, tgt = dataset.apply_scalers(test_df, feature_cols, feature_scaler, target_scaler)
    X_test, y_test_scaled = dataset.create_sequences(feats, tgt)

    y_test = target_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()
    base_pred = target_scaler.inverse_transform(
        np.asarray(model.predict(X_test, verbose=0)).reshape(-1, 1)
    ).ravel()
    baseline_mae = mean_absolute_error(y_test, base_pred)

    n = X_test.shape[0]
    importances = {}
    print(f"baseline test MAE: {baseline_mae:.4f}\n")
    for j, col in enumerate(feature_cols):
        deltas = []
        for _ in range(n_repeats):
            perm = rng.permutation(n)
            X_perm = X_test.copy()
            X_perm[:, :, j] = X_test[perm, :, j]
            pred_scaled = model.predict(X_perm, verbose=0).ravel()
            pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
            deltas.append(mean_absolute_error(y_test, pred) - baseline_mae)
        importances[col] = float(np.mean(deltas))
        print(f"  {col}: {importances[col]:+.4f}")

    ranked = dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))
    out = {"baseline_mae": float(baseline_mae), "n_repeats": n_repeats, "importances": ranked}

    out_path = config.METRICS_DIR / "permutation_importance.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nTop factor: {next(iter(ranked))}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
