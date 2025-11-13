from src.eval import evaluate_models
from src.features import build_features
from src.etl_clean import clean_data
from src.io_data import ensure_raw_data


def test_eval_metrics_non_negative():
    ensure_raw_data()
    clean_data()
    build_features()
    metrics = evaluate_models()
    assert (metrics["rmse"] >= 0).all()
