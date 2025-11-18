# Almaty Air Quality Forecast

End-to-end ML pipeline for hourly PM2.5 forecasting in Almaty.

This project trains and serves time-series models that predict PM2.5 concentrations 1–48 hours ahead for Almaty using historical air-quality and weather data. The goal is to provide reproducible, inspection-friendly forecasts that can support personal exposure planning and basic air-quality monitoring use cases.

**Demonstrates:**  
ETL with pandas/SQLAlchemy, time-series forecasting (baselines + RandomForest + Prophet), SHAP-based explainability, Dockerized Streamlit dashboard, and Postgres-backed persistence.

---

## 0. Quickstart

### Local (SQLite, no Docker)

```bash
# 1. Clone and create environment
git clone <this-repo-url>
cd almaty-air-quality-forecast
cp .env.example .env
python -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# 2. Generate data and train models (SQLite)
export USE_POSTGRES=false DB_URL=sqlite:///data/app.db   # or set in .env
make data
make clean features train eval forecast

# 3. Launch dashboard
streamlit run src/app_streamlit.py
# open http://localhost:8501
```

### Docker + Postgres

```bash
cp .env.example .env                     # ensure .env exists and is configured
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up
# open http://localhost:8501
```

Compose spins up Postgres plus the app container. `app` waits for DB readiness via `src.wait_for_db`, creates tables automatically, then launches Streamlit on port 8501.

## 1. Virtual Environment
All Python commands assume a local virtual environment stored in `.venv/`. Create/activate it once per machine:

```bash
cp .env.example .env                   # Windows CMD: copy .env.example .env
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt    # or simply run `make install`
```

The repo may already contain a populated `.venv/` folder in some environments; re-use it by running the activation command above before invoking any CLI or tests.

## 2. Project & Stack
- **Languages / libs:** Python 3.11, pandas, numpy, scikit-learn, Prophet, SHAP, SQLAlchemy, psycopg2, Streamlit, matplotlib, seaborn, holidays, pyarrow.
- **Persistence:** Postgres via Docker Compose; SQLite fallback for quick local runs.
- **Artifacts:** All datasets stored under `data/`, model objects in `data/models/`, metrics + SHAP plots in `data/processed/`.

## 3. Data Format & Synthetic Generation
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

## 4. Quick Local Start (SQLite, no Docker)
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

## 5. Docker + Postgres
Ensure `.env` is present next to `docker/` (or in project root if mounted there). Then run:

```bash
cp .env.example .env                     # run once, or ensure .env already exists
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up
# open http://localhost:8501
```

Compose spins up Postgres plus the app container. `app` waits for DB readiness via `src.wait_for_db`, creates tables automatically, then launches Streamlit on port 8501.

## 6. CLI Usage
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

## 7. Models, Benchmarks & Metrics
- **NaiveLast:** `ŷ_t = y_{t-1}`
- **SeasonalNaive:** `ŷ_t = y_{t-24}` for hourly data (fallbacks for daily frequencies)
- **RandomForestRegressor:** 300 trees, depth 12, TimeSeriesSplit(n=5)
- **Prophet:** daily + weekly seasonality with `changepoint_range=0.9`

Metrics (saved to `data/processed/metrics.csv` + DB):
```
MAE  = (1/n) * Σ |y_i - ŷ_i|
RMSE = sqrt( (1/n) * Σ (y_i - ŷ_i)^2 )
```
Baselines (NaiveLast / SeasonalNaive) are always included for benchmarking.

## 8. Results (snapshot)
Below is a template for test-set performance on an hourly hold-out split (e.g., last 6–8 weeks). Fill in actual numbers from `data/processed/metrics.csv`:

```
# Example (replace TBD with real values)
Model              MAE (µg/m³)    RMSE (µg/m³)
NaiveLast          TBD            TBD
SeasonalNaive 24h  TBD            TBD
RandomForest       TBD            TBD
Prophet            TBD            TBD
```

Example visual outputs to include under `reports/`:
- `reports/figures/forecast_vs_actual.png` – forecast vs. actual PM2.5 for the last N days.
- `data/processed/shap_summary.png` – SHAP summary plot for the RandomForest model.
- `reports/figures/feature_importance.png` – feature importance for tree-based models.

## 9. SHAP Explainability
`python -m src.cli explain --model rf` (or dashboard button) loads the RF model, computes SHAP values on the hold-out set with `shap.TreeExplainer`, and saves `data/processed/shap_summary.png`.

Interpretation: bar length = magnitude of impact; color = feature value.

## 10. Pipeline Architecture & Repository Structure

### Data & model flow
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

### Repository structure
```
.
├── src/
│   ├── cli/                # CLI entrypoints (ingest, clean, features, train, eval, forecast, explain, dashboard)
│   ├── etl_clean.py        # Raw CSV → clean parquet + DB ingest
│   ├── features.py         # Feature engineering and lag creation
│   ├── models.py           # Baselines + RandomForest + Prophet wrappers
│   ├── eval.py             # Metric computation and reporting
│   ├── explain.py          # SHAP explainability
│   └── app_streamlit.py    # Streamlit dashboard
├── data/
│   ├── raw/                # Raw CSV (almaty_pm25.csv)
│   ├── processed/          # clean.parquet, features.parquet, metrics.csv, SHAP plots
│   └── models/             # Serialized model artifacts
├── docker/
│   └── docker-compose.yml  # App + Postgres
├── reports/                # Figures, tables, notebooks for reporting
├── requirements.txt
├── Makefile
└── README.md
```

## 11. Reproducibility & Operations
- **Deterministic environments:** Use the `.venv/` instructions above or `make install`. For Docker, runtime dependencies are pinned via `docker/docker-compose.yml`. Synchronise new dependencies with `requirements.txt` / `requirements-dev.txt` and run `pip install -r ...` to stay in lockstep across machines.
- **One-command builds:** Invoke `make data`, `make clean`, `make features`, `make train`, `make eval`, and `make forecast` (or `make all`) to recreate artifacts from scratch. `make test` executes `pytest tests/` plus lint checks, and is the command wired into CI.
- **Data provenance:** Synthetic hourly PM2.5 data is generated via `make data`. For real datasets, document the source in `data/README.md`, add download scripts under `data/scripts/`, and reference hashes in commit messages so teammates can fetch identical bits.
- **Automation hooks:** `.github/workflows/ci.yml` (create if missing) should run `make install`, `make test`, and `make forecast --dry-run` to guarantee pull requests remain reproducible. Local developers can mirror this pipeline with `make ci`.
- **Experiment tracking:** Each training run writes timestamps + params to `data/processed/metrics.csv` and the SQL database. Commit these artifacts (or summaries) when publishing reports so results can be regenerated with identical CLI flags.

## 12. Documentation & Knowledge Base
- **README + wiki:** This README captures the high-level workflow. Create a `docs/` directory (MkDocs/Sphinx) for deeper dives: architecture, data dictionary, troubleshooting. Link notebooks or ADRs there for new contributors.
- **API & module docs:** Ensure every CLI entrypoint (`src/cli/*.py`) and complex helper has docstrings describing inputs, outputs, side effects. Add a `make docs` target to build HTML docs and publish them (e.g., GitHub Pages).
- **Setup cookbook:** Keep `.env.example`, `.vscode/launch.json`, and `docker/README.md` accurate. When configuration changes, update both the documentation and inline comments to prevent drift.
- **Knowledge capture:** Record FAQs, architecture decisions, and data caveats either in the repo wiki or `docs/adr/`. This keeps tribal knowledge accessible when the project scales or hands off.

## 13. Reporting & Visualization Artifacts
- **Benchmark reporting:** Every `python -m src.cli eval` writes MAE/RMSE for NaiveLast, SeasonalNaive, RandomForest, and Prophet to `data/processed/metrics.csv`. Export plots/tables to `reports/` and describe them in `docs/results.md`.
- **Visualization pack:** Generate and version charts: temporal PM2.5 trends, forecast vs. actual overlays, feature importance, SHAP summary, error histograms. Place rendered PNG/SVG into `reports/figures/`.
- **Comparison & discussion:** Document how models compare to baselines, when forecasts degrade (season transitions, missing data), and what accuracy thresholds are acceptable for stakeholders. Include runtime/performance notes for each model.
- **Reproduction scripts:** Provide notebooks like `notebooks/eval_report.ipynb` or scripts `scripts/build_report.py` that rebuild every figure/table using committed data so reviewers can validate claims.

## 14. User Experience & Usability
- **Design references:** Store UX flows, wireframes, and annotated screenshots in [`design docs/`](design%20docs/). Reference these assets inside feature PRs to keep engineers, designers, and analysts synchronized.
- **Product tours:** In the README (or `docs/usage.md`), include CLI snippets, GIFs of the Streamlit dashboard, and example API payloads. Make sure each user role (analyst, operator, developer) has a quick-start path.
- **Configuration ergonomics:** Provide `.env.example` defaults with sensible fallbacks, CLI `--help` text, and sample config files (e.g., `config/postgres.sample.yml`). Consider wrapping frequent workflows in Make targets or scripts.
- **Structure & naming:** Maintain the `src/` package as installable (`pip install -e .`) and group modules logically (ETL, features, models, dashboards). This lowers the entry barrier for external contributors.

## 15. Ethics, Data & Responsible AI
- **Licensing:** The project ships with [MIT License](LICENSE). Ensure any upstream datasets or code snippets comply with MIT-compatible licenses and credit them in `data/SOURCES.md`.
- **Data handling:** When ingesting real-world measurements, document privacy guarantees (e.g., aggregated public AQ data). If personal data ever appears, anonymize it and describe safeguards in `docs/privacy.md`.
- **Model limitations:** Clearly state in the README and dashboard that forecasts are advisory, not for critical health or safety decisions. Describe known failure modes (sensor outages, extreme events) and encourage manual validation.
- **Bias & fairness:** Track how data coverage might bias predictions (urban vs. rural stations, seasonal gaps). Note mitigation strategies—resampling, additional features, uncertainty estimates—and flag open risks for future work.
- **Transparency:** Add changelog entries for major model updates, keep SHAP explanations accessible, and document review procedures for new datasets or deployment environments.

## 16. Contributing
Pull requests are welcome. If you spot a bug or want to add a new baseline/model (e.g., XGBoost, LSTM), feel free to open an issue and describe your proposal.

Please keep PRs small and include a short note on how to reproduce your results (CLI commands and environment).

## 17. Contact
For questions about this project, open an issue on GitHub or contact the maintainer at `<your-email-here>`.

## 18. License
MIT License.
