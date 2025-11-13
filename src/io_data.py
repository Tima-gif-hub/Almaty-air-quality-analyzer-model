"""Raw data ingestion helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import re

from .config import BASE_DIR, get_settings
from .utils_time import ensure_datetime_index

logger = logging.getLogger(__name__)


def ensure_raw_data(path: Optional[Path] = None, freq: str | None = None, days: int = 180) -> Path:
    """Create synthetic data if the expected CSV is missing."""
    settings = get_settings()
    path = Path(path or settings.data_path)
    freq = freq or settings.freq

    if path.exists():
        logger.info("Found raw data at %s", path)
        return path

    logger.warning("Raw file %s not found; generating synthetic dataset.", path)
    path.parent.mkdir(parents=True, exist_ok=True)

    periods = days * (24 if freq.upper().startswith("H") else 1)
    idx = pd.date_range(
        end=pd.Timestamp.now(tz=settings.tz),
        periods=periods,
        freq=freq,
        tz=settings.tz,
    )

    hours = np.arange(len(idx))
    seasonal = 20 * np.sin(2 * np.pi * hours / 24) + 5 * np.cos(2 * np.pi * hours / (24 * 7))
    trend = np.linspace(5, 35, len(idx))
    noise = np.random.normal(0, 5, len(idx))

    temp = 15 + 10 * np.sin(2 * np.pi * hours / (24 * 30)) + np.random.normal(0, 1.5, len(idx))
    humidity = 50 + 20 * np.sin(2 * np.pi * hours / (24 * 14)) + np.random.normal(0, 5, len(idx))
    wind = 3 + np.random.gamma(2, 0.5, len(idx))

    df = pd.DataFrame(
        {
            "timestamp": idx,
            "pm25": np.clip(trend + seasonal + noise, 5, 250),
            "temp": temp,
            "humidity": np.clip(humidity, 10, 95),
            "wind_speed": wind,
        }
    )
    df.to_csv(path, index=False)
    logger.info("Synthetic dataset saved to %s", path)
    return path


def load_raw_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load raw CSV (ensuring presence)."""
    settings = get_settings()
    csv_path = ensure_raw_data(path or settings.data_path, settings.freq)
    df = pd.read_csv(csv_path)
    df = _normalise_raw_columns(df)
    if "timestamp" not in df.columns:
        raise ValueError(
            "Raw CSV must have a timestamp column (e.g. 'timestamp', 'date', 'datetime')."
        )
    if "pm25" not in df.columns:
        raise ValueError(
            "Raw CSV must contain a PM2.5 column (accepted aliases: pm25, PM2.5, pm_25)."
        )
    df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")
    df = ensure_datetime_index(df, "timestamp", settings.tz)
    return df


def save_parquet(df: pd.DataFrame, relative_path: str) -> Path:
    """Persist dataframe under data/processed."""
    out_path = BASE_DIR / relative_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    logger.info("Saved %s rows to %s", len(df), out_path)
    return out_path


def load_parquet(relative_path: str) -> pd.DataFrame:
    """Load dataset from parquet relative to project root."""
    path = BASE_DIR / relative_path
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def _clean_column_name(name: str) -> str:
    """Convert messy header names to snake_case."""
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower())
    return normalized.strip("_")


ALIASES = {
    "date": "timestamp",
    "datetime": "timestamp",
    "time": "timestamp",
    "pm_25": "pm25",
    "pm2_5": "pm25",
    "pm2.5": "pm25",
    "pm_2_5": "pm25",
}


def _normalise_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe with trimmed column names and canonical aliases."""
    df = df.copy()
    df.columns = [_clean_column_name(col) for col in df.columns]
    for alias, target in ALIASES.items():
        if alias in df.columns and target not in df.columns:
            df.rename(columns={alias: target}, inplace=True)
    return df
