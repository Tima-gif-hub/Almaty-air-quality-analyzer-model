# Almaty Air Quality Forecast

End-to-end ML pipeline for hourly PM2.5 forecasting in Almaty: ETL → cleaning → feature engineering → benchmarks + models → forecasting → explainability (SHAP) → Streamlit dashboard. Stack: Python 3.11, pandas, scikit-learn, Prophet, SHAP, SQLAlchemy, Streamlit, Postgres (default) / SQLite (local dev).

## 0. Virtual Environment
All Python commands assume a local virtual environment stored in `.venv/`. Create/activate it once per machine:

```bash
cp .env.example .env                   # Windows CMD: copy .env.example .env
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt    # or simply run `make install`
```

The repo already contains a populated `.venv/` folder in this workspace; re-use it by running the activation command above before invoking any CLI or tests.

## 1. Project & Stack
- **Languages / libs:** Python 3.11, pandas, numpy, scikit-learn, Prophet, SHAP, SQLAlchemy, psycopg2, Streamlit, matplotlib, seaborn, holidays, pyarrow.
- **Persistence:** Postgres via Docker Compose; SQLite fallback for quick local runs.
- **Artifacts:** All datasets stored under `data/`, model objects in `data/models/`, metrics + SHAP plots in `data/processed/`.

## 2. Data Format & Synthetic Generation
Expected CSV: `data/raw/almaty_pm25.csv`

```
timestamp,pm25,temp?,humidity?,wind_speed?
```

- Column aliases such as `date`, `datetime`, `pm 2.5`, `pm_25`, `PM2.5` are auto-normalised.
- Additional pollutant columns (pm10, no2, so2, co, etc.) are preserved and can later be engineered manually.
- Timestamps should already be in the Asia/Almaty timezone; naive timestamps are assumed to be local and are localised automatically.

If the file is missing, run `make data` (or `python -m src.cli ingest`) to generate 180 days of synthetic hourly data with trend + seasonality + weather covariates. For real daily aggregates (e.g., official PM2.5 reports), set `FREQ=D` in `.env` before running the pipeline so resampling/lag logic uses daily steps instead of hourly interpolation.

### Using your own dataset
1. Drop the CSV into `data/raw/almaty_pm25.csv` (or point `DATA_PATH` to another file).
2. Adjust `.env` if needed:
   - `FREQ=H` for hourly data (default); `FREQ=D` for daily aggregates.
   - Keep `TZ=Asia/Almaty` unless the timestamps are already timezone aware.
3. Run the standard CLI to build every artefact:
   ```bash
   python -m src.cli clean          # ETL + parquet + DB ingest
   python -m src.cli make-features  # features.parquet
   python -m src.cli train --model rf   # or prophet
   python -m src.cli eval
   python -m src.cli forecast --model prophet --h 30
   ```
   These commands work identically inside Docker via `docker compose exec app bash`.

## 3. Quick Local Start (SQLite, no Docker)
Ensure `.env` already exists (copied from `.env.example` or crafted manually) and tweak values as needed, then:

```bash
cp .env.example .env                              # skip if already configured
source .venv/bin/activate                     # Windows: .\.venv\Scripts\Activate.ps1
make install                                  # installs/refreshes deps inside .venv
export USE_POSTGRES=false DB_URL=sqlite:///data/app.db   # or edit .env
make data
make clean features train eval
streamlit run src/app_streamlit.py
```

## 4. Docker + Postgres
Ensure `.env` is present next to `docker/`. Then run:

```bash
cp .env.example .env                     # run once, or ensure .env already exists
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up
# open http://localhost:8501
```

Compose spins up Postgres plus the app container. `app` waits for DB readiness via `src.wait_for_db`, creates tables automatically, then launches Streamlit on port 8501.

## 5. CLI Usage
```bash
python -m src.cli ingest --path data/raw/almaty_pm25.csv
python -m src.cli clean
python -m src.cli make-features
python -m src.cli train --model rf        # or prophet
python -m src.cli eval
python -m src.cli forecast --model prophet --h 48
python -m src.cli explain --model rf
python -m src.cli dashboard
```

Each command logs a concise report and writes artifacts (parquet/csv/shap) plus DB rows via SQLAlchemy models (`measurements`, `features`, `forecasts`, `metrics`).

## 6. Models, Benchmarks & Metrics
- **NaiveLast:** `ŷ_t = y_{t-1}`
- **SeasonalNaive:** `ŷ_t = y_{t-24}` for hourly data (fallbacks for daily frequencies)
- **RandomForestRegressor:** 300 trees, depth 12, TimeSeriesSplit(n=5)
- **Prophet:** daily + weekly seasonality with changepoint_range 0.9

Metrics (saved to `data/processed/metrics.csv` + DB):
```
MAE  = (1/n) * Σ |y_i - ŷ_i|
RMSE = sqrt( (1/n) * Σ (y_i - ŷ_i)^2 )
```
Baselines (NaiveLast / SeasonalNaive) are always included for benchmarking.

## 7. SHAP Explainability
`python -m src.cli explain --model rf` (or dashboard button) loads the RF model, computes SHAP values on the hold-out set with `shap.TreeExplainer`, and saves `data/processed/shap_summary.png`. Interpretation: bar length = magnitude of impact; color = feature value.

## 8. Pipeline Architecture
```
Raw CSV ──► ETL (src/etl_clean.py) ──► clean.parquet
            │
            └──► Feature builder (src/features.py) ──► features.parquet
                   │
                   ├──► Models & benchmarks (src/models.py)
                   │       │
                   │       └──► Metrics (src/eval.py → data/processed/metrics.csv)
                   └──► Forecasts + SHAP (src/explain.py)

SQLAlchemy tables: measurements, features, forecasts, metrics
Streamlit dashboard (src/app_streamlit.py) surfaces QC, features, training, forecast, explainability tabs
```

## 9. License
[MIT License](LICENSE).
