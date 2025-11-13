"""Streamlit dashboard for Almaty air quality."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import BASE_DIR, get_settings
from src.eval import evaluate_models
from src.explain import explain_rf
from src.features import build_features
from src.io_data import ensure_raw_data
from src.models import forecast as run_forecast
from src.models import train_model


st.set_page_config(page_title="Almaty Air Quality Forecast", layout="wide")


def load_artifact(path: str) -> pd.DataFrame | None:
    full = BASE_DIR / path
    if not full.exists():
        return None
    if full.suffix == ".parquet":
        return pd.read_parquet(full)
    if full.suffix == ".csv":
        return pd.read_csv(full)
    return None


def main():
    settings = get_settings()
    st.sidebar.header("Controls")
    model = st.sidebar.selectbox("Model", ["rf", "prophet"])
    horizon = st.sidebar.slider("Forecast horizon", 6, 72, settings.forecast_horizon, step=6)
    st.sidebar.checkbox("Use weather covariates", value=True, disabled=True)
    retrain = st.sidebar.checkbox("Retrain model on demand", value=False)
    st.sidebar.button("Generate synthetic data", on_click=lambda: ensure_raw_data())

    tabs = st.tabs(["Data & QC", "Features", "Train & Metrics", "Forecast", "Explainability"])

    with tabs[0]:
        st.subheader("Raw & Clean data")
        raw_path = settings.data_path
        st.write(f"Raw CSV: `{raw_path}`")
        try:
            raw_df = pd.read_csv(raw_path)
            st.dataframe(raw_df.tail())
        except FileNotFoundError:
            st.warning("Raw data missing, click sidebar to generate synthetic data.")
        clean_df = load_artifact("data/processed/clean.parquet")
        if clean_df is not None:
            st.line_chart(clean_df["pm25"])
        else:
            st.info("Clean dataset not available yet. Run the cleaning step.")

    with tabs[1]:
        st.subheader("Features")
        if st.button("Build features"):
            build_features()
        feats = load_artifact("data/processed/features.parquet")
        if feats is not None:
            st.write("Feature sample", feats.head())
            st.bar_chart(feats.drop(columns=["pm25_target"]).std().sort_values(ascending=False).head(10))
        else:
            st.info("No feature dataset found. Click 'Build features' above.")

    with tabs[2]:
        st.subheader("Training & Metrics")
        if retrain and st.button("Train now"):
            try:
                train_model(model)
                st.success("Training finished.")
            except (FileNotFoundError, RuntimeError) as exc:
                st.error(f"Training failed: {exc}")
        metrics = load_artifact("data/processed/metrics.csv")
        if metrics is None:
            if st.button("Evaluate models"):
                try:
                    metrics = evaluate_models()
                except FileNotFoundError:
                    st.error("Features missing. Build features before evaluation.")
                except RuntimeError as exc:
                    st.error(f"Evaluation skipped: {exc}")
        if metrics is not None:
            st.table(metrics)

    with tabs[3]:
        st.subheader("Forecast")
        if st.button("Generate forecast"):
            try:
                forecast_df = run_forecast(model, horizon)
                st.line_chart(forecast_df.set_index("timestamp")["yhat"])
                st.dataframe(forecast_df.tail())
            except FileNotFoundError:
                st.error("Model artifacts not found. Build features and train a model first.")
            except RuntimeError as exc:
                st.error(f"{exc}")
        else:
            st.info("Click to generate forecast.")

    with tabs[4]:
        st.subheader("Explainability (SHAP)")
        if st.button("Compute SHAP for RF"):
            try:
                path = explain_rf()
                if path:
                    st.image(str(path))
            except FileNotFoundError:
                st.error("Train the Random Forest model before running SHAP.")
        shap_path = BASE_DIR / "data" / "processed" / "shap_summary.png"
        if shap_path.exists():
            st.image(str(shap_path))
        else:
            st.info("No SHAP summary yet.")


if __name__ == "__main__":
    main()
