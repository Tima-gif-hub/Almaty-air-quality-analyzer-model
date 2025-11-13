"""Model training and forecasting utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from .config import BASE_DIR, get_settings
from .db import Forecast, upsert_batch
from .io_data import load_parquet
from .utils_time import seasonal_lag

logger = logging.getLogger(__name__)

MODEL_DIR = BASE_DIR / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_test_split(df: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test


@dataclass
class BaseModel:
    name: str

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    def save(self) -> Path:
        raise NotImplementedError

    def load(self) -> None:
        raise NotImplementedError


class NaiveLast(BaseModel):
    def __init__(self):
        super().__init__("naive_last")
        self.last_value: float | None = None

    def fit(self, X, y):
        self.last_value = float(y.iloc[-1])

    def predict(self, X):
        if self.last_value is None:
            raise ValueError("Model not trained.")
        return np.full(len(X), self.last_value)


class SeasonalNaive(BaseModel):
    def __init__(self, freq: str):
        super().__init__("seasonal_naive")
        self.freq = freq
        self.lag = seasonal_lag(freq)
        self.history: pd.Series | None = None

    def fit(self, X, y):
        self.history = y

    def predict(self, X):
        if self.history is None:
            raise ValueError("Model not trained.")
        preds = self.history.shift(self.lag).reindex(X.index, method="nearest")
        return preds.fillna(method="bfill").fillna(method="ffill").to_numpy()


class RandomForestModel(BaseModel):
    def __init__(self, random_state: int):
        super().__init__("rf")
        self.model = RandomForestRegressor(
            n_estimators=300, max_depth=12, random_state=random_state, n_jobs=-1
        )
        self.feature_names: List[str] = []

    def fit(self, X, y):
        self.feature_names = list(X.columns)
        tscv = TimeSeriesSplit(n_splits=5)
        # Optional cross-val - we just fit final model
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self):
        path = MODEL_DIR / "rf_model.joblib"
        joblib.dump({"model": self.model, "features": self.feature_names}, path)
        logger.info("Saved RF model to %s", path)
        return path

    def load(self):
        path = MODEL_DIR / "rf_model.joblib"
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names = data["features"]
        return path


class ProphetModel(BaseModel):
    def __init__(self):
        super().__init__("prophet")
        try:
            self.model = Prophet(
                weekly_seasonality=True,
                daily_seasonality=True,
                yearly_seasonality=False,
                changepoint_range=0.9,
            )
        except AttributeError as exc:  # missing backend (e.g., cmdstanpy download blocked)
            raise RuntimeError(
                "Prophet backend is unavailable. Ensure cmdstanpy is installed and "
                "CmdStan binaries are downloaded (run `python -m prophet.models.download`)."
            ) from exc

    def fit(self, X, y):
        idx = X.index.tz_convert(None) if X.index.tz else X.index
        df = pd.DataFrame({"ds": idx, "y": y.to_numpy()})
        self.model.fit(df)

    def predict(self, X):
        idx = X.index.tz_convert(None) if X.index.tz else X.index
        future = pd.DataFrame({"ds": idx})
        forecast = self.model.predict(future)
        return forecast["yhat"].values

    def save(self):
        path = MODEL_DIR / "prophet_model.json"
        self.model.save(str(path))
        return path

    def load(self):
        path = MODEL_DIR / "prophet_model.json"
        self.model = Prophet.load(str(path))
        return path


def _build_model(name: str) -> BaseModel:
    settings = get_settings()
    if name == "naive_last":
        return NaiveLast()
    if name == "seasonal_naive":
        return SeasonalNaive(settings.freq)
    if name == "rf":
        return RandomForestModel(settings.random_state)
    if name == "prophet":
        return ProphetModel()
    raise ValueError(f"Unknown model '{name}'")


def train_model(model_name: str = "rf") -> dict[str, float]:
    feats = load_parquet("data/processed/features.parquet")
    train, test = train_test_split(feats)
    X_train, y_train = train.drop(columns=["pm25_target"]), train["pm25_target"]
    X_test, y_test = test.drop(columns=["pm25_target"]), test["pm25_target"]

    model = _build_model(model_name)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": mean_squared_error(y_test, preds, squared=False),
    }

    if model_name in {"rf", "prophet"}:
        model.save()

    logger.info("%s -> MAE %.2f RMSE %.2f", model_name, metrics["mae"], metrics["rmse"])
    return metrics


def forecast(model_name: str, horizon: int) -> pd.DataFrame:
    if model_name not in {"rf", "prophet"}:
        raise ValueError("Unsupported model for forecasting.")
    feats = load_parquet("data/processed/features.parquet")
    horizon = min(horizon, len(feats))
    model = _build_model(model_name)
    model.load()

    last = feats.iloc[-horizon:]
    X = last.drop(columns=["pm25_target"])
    preds = model.predict(X)

    forecast_df = pd.DataFrame({"timestamp": X.index, "model": model_name, "yhat": preds})
    forecast_df["horizon"] = range(1, len(forecast_df) + 1)

    upsert_rows = [
        {
            "timestamp": row.timestamp.to_pydatetime(),
            "model": model_name,
            "horizon": int(row.horizon),
            "yhat": float(row.yhat),
        }
        for row in forecast_df.itertuples()
    ]
    upsert_batch(Forecast, upsert_rows)
    return forecast_df
