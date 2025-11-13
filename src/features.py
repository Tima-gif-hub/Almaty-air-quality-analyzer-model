"""Feature engineering for PM2.5 forecasting."""

from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

from .config import get_settings
from .db import FeatureRow, upsert_batch
from .io_data import load_parquet, save_parquet
from .utils_time import (
    add_calendar_features,
    lagged_features,
    make_fourier_features,
    rolling_stats,
    seasonal_lag,
)

logger = logging.getLogger(__name__)


def _select_lags(freq: str) -> Iterable[int]:
    if freq.upper().startswith("H"):
        return [1, 24, 168]
    return [1, 7, 28]


def build_features() -> pd.DataFrame:
    settings = get_settings()
    df = load_parquet("data/processed/clean.parquet")
    feats = pd.DataFrame(index=df.index)

    lags = _select_lags(settings.freq)
    feats = feats.join(lagged_features(df, "pm25", lags))
    feats = feats.join(rolling_stats(df, "pm25", [24, 168]))
    feats = feats.join(add_calendar_features(pd.DataFrame(index=df.index), settings.tz))

    daily_fourier = make_fourier_features(df.index, period=24, order=2, prefix="daily")
    weekly_fourier = make_fourier_features(df.index, period=24 * 7, order=2, prefix="weekly")
    feats = feats.join(daily_fourier).join(weekly_fourier)

    for weather_col in ["temp", "humidity", "wind_speed"]:
        if weather_col in df.columns:
            feats[weather_col] = df[weather_col]
            for lag in [1, 24]:
                feats[f"{weather_col}_lag_{lag}"] = df[weather_col].shift(lag)

    feats["pm25_target"] = df["pm25"]
    feats.dropna(inplace=True)

    save_parquet(feats, "data/processed/features.parquet")

    upsert_rows = [
        {"timestamp": idx.to_pydatetime(), "data": row.to_dict()} for idx, row in feats.iterrows()
    ]
    upsert_batch(FeatureRow, upsert_rows)
    logger.info("Features saved with %s rows.", len(feats))
    return feats


if __name__ == "__main__":
    build_features()
