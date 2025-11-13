import pandas as pd

from src.features import build_features
from src.io_data import ensure_raw_data
from src.etl_clean import clean_data


def test_feature_columns(tmp_path, monkeypatch):
    ensure_raw_data()
    clean_data()
    feats = build_features()
    required_cols = ["pm25_lag_1", "pm25_lag_24", "pm25_lag_168", "daily_sin_1", "daily_cos_1"]
    for col in required_cols:
        assert col in feats.columns
