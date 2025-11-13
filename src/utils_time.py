"""Time helpers for feature engineering."""

from __future__ import annotations

from datetime import timedelta
from typing import Iterable

import numpy as np
import pandas as pd
import pytz
from holidays import country_holidays


def ensure_datetime_index(
    df: pd.DataFrame, timestamp_col: str, tz_name: str
) -> pd.DataFrame:
    """Return dataframe indexed by timezone-aware timestamps."""
    tz = pytz.timezone(tz_name)
    df = df.copy()
    stamps = pd.to_datetime(df[timestamp_col], errors="coerce")
    if stamps.dt.tz is None:
        stamps = stamps.dt.tz_localize(tz)
    else:
        stamps = stamps.dt.tz_convert(tz)
    df[timestamp_col] = stamps
    df.set_index(timestamp_col, inplace=True)
    df.sort_index(inplace=True)
    return df


def resample_and_interpolate(
    df: pd.DataFrame, freq: str, agg: str = "mean"
) -> pd.DataFrame:
    """Resample to target frequency and interpolate missing points."""
    df = df.resample(freq).agg(agg)
    df.interpolate(method="time", inplace=True, limit_direction="both")
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df


def hampel_filter(series: pd.Series, window_size: int = 24, n_sigmas: float = 3.0) -> pd.Series:
    """Clip outliers using a Hampel-like filter."""
    if series.isna().all():
        return series
    rolling_median = series.rolling(window=window_size, center=True, min_periods=1).median()
    diff = np.abs(series - rolling_median)
    mad = diff.rolling(window=window_size, center=True, min_periods=1).median()
    threshold = n_sigmas * 1.4826 * mad
    clipped = series.where(diff <= threshold, rolling_median)
    return clipped


def iqr_clip(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """Clip outliers based on the interquartile range."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return series.clip(lower, upper)


def make_fourier_features(
    index: pd.DatetimeIndex, period: int, order: int = 2, prefix: str = "daily"
) -> pd.DataFrame:
    """Construct sine/cosine seasonal features."""
    seconds = (index.view("int64") // 10**9).astype(float)
    features = {}
    for k in range(1, order + 1):
        angle = 2 * np.pi * k * seconds / (period * 3600)
        features[f"{prefix}_sin_{k}"] = np.sin(angle)
        features[f"{prefix}_cos_{k}"] = np.cos(angle)
    return pd.DataFrame(features, index=index)


def add_calendar_features(df: pd.DataFrame, tz_name: str) -> pd.DataFrame:
    """Append calendar-based categorical features."""
    tz = pytz.timezone(tz_name)
    localized = df.index.tz_convert(tz)
    df = df.copy()
    df["hour"] = localized.hour
    df["dayofweek"] = localized.dayofweek
    df["month"] = localized.month
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    years = list(range(localized.min().year, localized.max().year + 1))
    kz_holidays = country_holidays("KZ", years=years)
    holiday_dates = set(kz_holidays.keys())
    df["is_holiday"] = localized.normalize().isin(holiday_dates).astype(int)
    return df


def lagged_features(
    df: pd.DataFrame, column: str, lags: Iterable[int]
) -> pd.DataFrame:
    """Create lag features for specified lags."""
    lagged = {f"{column}_lag_{lag}": df[column].shift(lag) for lag in lags}
    return pd.DataFrame(lagged, index=df.index)


def rolling_stats(
    df: pd.DataFrame, column: str, windows: Iterable[int]
) -> pd.DataFrame:
    """Compute rolling mean/std stats."""
    feats = {}
    for window in windows:
        feats[f"{column}_roll_mean_{window}"] = df[column].rolling(window=window).mean()
        feats[f"{column}_roll_std_{window}"] = df[column].rolling(window=window).std()
    return pd.DataFrame(feats, index=df.index)


def seasonal_lag(freq: str) -> int:
    """Return seasonal lag used by seasonal naive model."""
    if freq.upper().startswith("H"):
        return 24
    if freq.upper().startswith("D"):
        return 7
    return 1


def horizon_timedelta(freq: str, steps: int) -> timedelta:
    """Translate horizon steps into timedelta."""
    freq = freq.upper()
    if freq.startswith("H"):
        return timedelta(hours=steps)
    if freq.startswith("D"):
        return timedelta(days=steps)
    return timedelta(hours=steps)
