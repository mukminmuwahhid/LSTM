"""End-to-end training entry point.

Run from the project root with:
    python -m src.train
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src import config, data_cleaning, dataset, feature_engineering, model as model_module


def set_seed(seed: int = config.RANDOM_SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def plot_training_history(history: keras.callbacks.History, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss (MSE)")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history.history["mae"], label="train")
    axes[1].plot(history.history["val_mae"], label="val")
    axes[1].set_title("MAE")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    set_seed()

    print("1/4 Cleaning raw data...")
    cleaned = data_cleaning.clean_pipeline()

    print("2/4 Engineering features...")
    featured, feature_cols = feature_engineering.build_feature_set(cleaned)
    featured.to_csv(config.FEATURED_DATA_PATH)

    print("3/4 Building sequences and scaling (fit on train split only)...")
    data = dataset.prepare_datasets(featured, feature_cols)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    print(f"   train={X_train.shape}, val={X_val.shape}, test={data['X_test'].shape}")

    print("4/4 Training LSTM...")
    model = model_module.build_lstm_model(n_timesteps=X_train.shape[1], n_features=X_train.shape[2])
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=config.EARLY_STOPPING_PATIENCE, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=config.REDUCE_LR_PATIENCE, min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(config.MODEL_PATH), monitor="val_loss", save_best_only=True
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config.MAX_EPOCHS,
        batch_size=config.BATCH_SIZE,
        shuffle=True,  # safe: each window already encodes its own temporal order,
        # and the chronological split (not this shuffle) is what prevents leakage
        callbacks=callbacks,
        verbose=2,
    )

    plot_training_history(history, config.FIGURES_DIR / "training_history.png")

    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(config.METRICS_DIR / "training_history.json", "w") as f:
        json.dump(history_dict, f, indent=2)

    print(f"\nBest model saved to {config.MODEL_PATH}")
    print(f"Scalers saved to {config.FEATURE_SCALER_PATH} / {config.TARGET_SCALER_PATH}")
    print(f"Training curves saved to {config.FIGURES_DIR / 'training_history.png'}")


if __name__ == "__main__":
    main()
