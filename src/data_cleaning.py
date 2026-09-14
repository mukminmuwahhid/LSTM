"""Load raw electricity + weather data, clean it, and merge into one
hourly-indexed DataFrame ready for feature engineering.

Cleaning steps applied:
  - parse timestamps, sort, drop exact-duplicate timestamps
  - reindex to a complete hourly range so gaps become explicit NaN rows
    instead of being silently skipped (important for an LSTM, which needs
    an unbroken time step)
  - normalize the inconsistent ``site_id`` naming found in the weather file
  - drop constant / uninformative columns (snow, snowdepth are always 0
    in this tropical location)
  - time-based interpolation for short gaps, cap physically impossible
    values (negative electricity/irradiance etc.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def load_electricity(path=config.ELECTRICITY_RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="first")
    df = df.set_index("timestamp")

    full_range = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full_range)
    df.index.name = "timestamp"

    # Electricity can't be negative; treat as missing and interpolate.
    df.loc[df[config.TARGET_COL] < 0, config.TARGET_COL] = np.nan
    df[config.TARGET_COL] = df[config.TARGET_COL].interpolate(method="time", limit=6)
    df[config.TARGET_COL] = df[config.TARGET_COL].ffill().bfill()
    return df


def load_weather(path=config.WEATHER_RAW_PATH) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Normalize the inconsistent site naming (e.g. "bukit mahkota" vs
    # "bukit mahkota bangi") — this is a single-site dataset.
    df["site_id"] = config.SITE_NAME

    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="first")
    df = df.set_index("timestamp")

    full_range = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full_range)
    df.index.name = "timestamp"

    # Drop constant / non-predictive columns.
    drop_cols = [c for c in ("snow", "snowdepth", "site_id") if c in df.columns]
    df = df.drop(columns=drop_cols)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method="time", limit=6)
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    # Physical floors: negative radiation/precip/wind are sensor errors.
    for col in ("precip", "windgust", "windspeed", "solarradiation", "solarenergy", "cloudcover", "humidity"):
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    return df


def merge_electricity_weather(elec: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    merged = elec.join(weather, how="inner")
    merged = merged.dropna(subset=[config.TARGET_COL])
    return merged


def clean_pipeline(save: bool = True) -> pd.DataFrame:
    elec = load_electricity()
    weather = load_weather()
    merged = merge_electricity_weather(elec, weather)

    if save:
        merged.to_csv(config.CLEANED_DATA_PATH)
    return merged


if __name__ == "__main__":
    df = clean_pipeline()
    print(f"Cleaned merged dataset: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"Range: {df.index.min()} -> {df.index.max()}")
    print(f"Remaining NaNs:\n{df.isna().sum()[df.isna().sum() > 0]}")
    print(f"Saved to {config.CLEANED_DATA_PATH}")
