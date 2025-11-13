"""ETL pipeline for cleaning PM2.5 data."""

from __future__ import annotations

import logging

import pandas as pd

from .config import BASE_DIR, get_settings
from .db import Measurement, upsert_batch
from .io_data import load_raw_data, save_parquet
from .utils_time import hampel_filter, iqr_clip, resample_and_interpolate

logger = logging.getLogger(__name__)


def clean_data() -> pd.DataFrame:
    settings = get_settings()
    df = load_raw_data()
    df = resample_and_interpolate(df, settings.freq)
    df["pm25"] = hampel_filter(df["pm25"])
    df["pm25"] = iqr_clip(df["pm25"])
    clean_path = "data/processed/clean.parquet"
    save_parquet(df, clean_path)
    logger.info("Clean data stored at %s", clean_path)

    upsert_rows = [
        {
            "timestamp": idx.to_pydatetime(),
            "pm25": row.get("pm25"),
            "temp": row.get("temp"),
            "humidity": row.get("humidity"),
            "wind_speed": row.get("wind_speed"),
        }
        for idx, row in df.iterrows()
    ]
    upsert_batch(Measurement, upsert_rows)
    return df


if __name__ == "__main__":
    clean_data()
