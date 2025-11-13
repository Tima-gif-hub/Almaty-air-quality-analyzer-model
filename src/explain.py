"""Explainability utilities using SHAP."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import shap

from .config import BASE_DIR, get_settings
from .io_data import load_parquet
from .models import RandomForestModel, train_test_split

logger = logging.getLogger(__name__)


def explain_rf() -> Path | None:
    settings = get_settings()
    feats = load_parquet("data/processed/features.parquet")
    train, test = train_test_split(feats)
    X_train, y_train = train.drop(columns=["pm25_target"]), train["pm25_target"]
    X_test = test.drop(columns=["pm25_target"])

    model = RandomForestModel(settings.random_state)
    try:
        model.load()
    except FileNotFoundError:
        logger.warning("RF model not found, training quickly for SHAP.")
        model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(X_test)

    out_dir = BASE_DIR / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "shap_summary.png"
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(summary_path)
    plt.close()
    logger.info("SHAP summary saved to %s", summary_path)
    return summary_path


if __name__ == "__main__":
    explain_rf()
