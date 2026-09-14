"""Stacked LSTM model for next-step electricity consumption forecasting."""
from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from src import config


def build_lstm_model(
    n_timesteps: int,
    n_features: int,
    units: list[int] = config.LSTM_UNITS,
    dropout_rate: float = config.DROPOUT_RATE,
    learning_rate: float = config.LEARNING_RATE,
) -> keras.Model:
    model = keras.Sequential(name="lstm_electricity_forecaster")
    model.add(layers.Input(shape=(n_timesteps, n_features)))

    for i, n_units in enumerate(units):
        return_sequences = i < len(units) - 1
        model.add(layers.LSTM(n_units, return_sequences=return_sequences))
        model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dense(1))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


if __name__ == "__main__":
    m = build_lstm_model(n_timesteps=config.LOOKBACK, n_features=30)
    m.summary()
