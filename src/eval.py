"""Model evaluation utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import BASE_DIR, get_settings
from .db import Metric, upsert_batch
from .io_data import load_parquet
from .models import _build_model, train_test_split

logger = logging.getLogger(__name__)


def evaluate_models() -> pd.DataFrame:
    settings = get_settings()
    feats = load_parquet("data/processed/features.parquet")
    train, test = train_test_split(feats)
    X_train, y_train = train.drop(columns=["pm25_target"]), train["pm25_target"]
    X_test, y_test = test.drop(columns=["pm25_target"]), test["pm25_target"]

    model_names = ["naive_last", "seasonal_naive", "rf", "prophet"]

    rows = []
    for name in model_names:
        try:
            model = _build_model(name)
        except RuntimeError as exc:
            logger.warning("Skipping %s: %s", name, exc)
            continue
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)
        rows.append({"model": name, "mae": mae, "rmse": rmse})
        logger.info("Model %s -> MAE %.3f RMSE %.3f", name, mae, rmse)

    metrics_df = pd.DataFrame(rows)
    out_path = BASE_DIR / "data" / "processed" / "metrics.csv"
    metrics_df.to_csv(out_path, index=False)

    upsert_rows = [
        {
            "model": row.model,
            "metric": "mae",
            "value": float(row.mae),
            "details": {"rmse": float(row.rmse)},
        }
        for row in metrics_df.itertuples()
    ]
    upsert_batch(Metric, upsert_rows)
    return metrics_df


if __name__ == "__main__":
    evaluate_models()
