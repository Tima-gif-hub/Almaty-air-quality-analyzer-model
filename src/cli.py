"""Command line interface for the project."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys

from .config import get_settings
from .etl_clean import clean_data
from .eval import evaluate_models
from .explain import explain_rf
from .features import build_features
from .io_data import ensure_raw_data
from .models import forecast as run_forecast
from .models import train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_ingest(path: str | None) -> None:
    ensure_raw_data(path)
    logger.info("Data ready at %s", path)


def run_clean() -> None:
    clean_data()


def run_features() -> None:
    build_features()


def run_train(model: str) -> None:
    metrics = train_model(model)
    logger.info("Train metrics: %s", metrics)


def run_eval() -> None:
    df = evaluate_models()
    logger.info("Evaluation completed:\n%s", df)


def run_forecast_cli(model: str, horizon: int) -> None:
    df = run_forecast(model, horizon)
    logger.info("Forecast saved:\n%s", df.tail())


def run_explain(model: str) -> None:
    if model != "rf":
        logger.warning("Explain currently supports only RF.")
        return
    path = explain_rf()
    logger.info("SHAP artifacts at %s", path)


def run_dashboard() -> None:
    cmd = [
        "streamlit",
        "run",
        "src/app_streamlit.py",
        "--server.port",
        "8501",
        "--server.address",
        "0.0.0.0",
    ]
    subprocess.run(cmd, check=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Almaty Air Quality Forecast CLI")
    sub = parser.add_subparsers(dest="command")

    ingest = sub.add_parser("ingest")
    ingest.add_argument("--path", default=get_settings().data_path, help="Path to CSV file.")

    sub.add_parser("clean")
    sub.add_parser("make-features")

    train = sub.add_parser("train")
    train.add_argument("--model", choices=["rf", "prophet"], default="rf")

    sub.add_parser("eval")

    forecast = sub.add_parser("forecast")
    forecast.add_argument("--model", choices=["rf", "prophet"], default="rf")
    forecast.add_argument("--h", type=int, default=get_settings().forecast_horizon)

    explain = sub.add_parser("explain")
    explain.add_argument("--model", default="rf")

    sub.add_parser("dashboard")

    args = parser.parse_args(argv)

    if args.command == "ingest":
        run_ingest(args.path)
    elif args.command == "clean":
        run_clean()
    elif args.command == "make-features":
        run_features()
    elif args.command == "train":
        run_train(args.model)
    elif args.command == "eval":
        run_eval()
    elif args.command == "forecast":
        run_forecast_cli(args.model, args.h)
    elif args.command == "explain":
        run_explain(args.model)
    elif args.command == "dashboard":
        run_dashboard()
    else:
        parser.print_help()


if __name__ == "__main__":
    main(sys.argv[1:])
