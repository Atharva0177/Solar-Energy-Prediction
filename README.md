# UNISOLAR Solar Power Generation Prediction Platform

Multi-site solar photovoltaic (PV) power forecasting platform built on the **UNISOLAR** dataset. Compares statistical baselines, classical machine learning, and deep-learning models across multiple forecast horizons with leakage-safe temporal validation, unseen-site generalization testing, explainable AI, and uncertainty estimation — served through a REST API and an interactive dashboard.

> Full specification: [PRD.md](PRD.md) · Task tracking: [TASKS.md](TASKS.md) ·
> Decision log: [DECISIONS.md](DECISIONS.md) · Measured outcomes: [RESULTS.md](RESULTS.md)

## Project Status

**All 16 phases complete** — see [TASKS.md](TASKS.md) for detailed progress.

- Phase 0: Repository initialization ✅
- Phase 1: Dataset acquisition & inspection ✅
- Phase 2: Data engineering ✅
- Phase 3: Exploratory data analysis ✅
- Phase 4: Baselines ✅
- Phase 5: Feature engineering ✅
- Phase 6: XGBoost ✅
- Phase 7: LSTM/GRU ✅
- Phase 8: Transformer ✅
- Phase 9: Cross-site evaluation ✅
- Phase 10: Explainability (SHAP) ✅
- Phase 11: Uncertainty estimation (conformal prediction) ✅
- Phase 12: Backend (FastAPI) ✅
- Phase 13: Frontend (React dashboard) ✅
- Phase 14: Integration (model serving) ✅
- Phase 15: Testing (unit/API/leakage) ✅
- Phase 16: Docker deployment ✅
- Post-PRD enhancements: Training page + richer visuals (D-024/D-025) ✅

## Overview

| | |
|---|---|
| **Objective** | Predict PV power generation from historical generation, weather, irradiance, and temporal features |
| **Dataset** | UNISOLAR multi-site PV generation + weather CSVs (`unisolar/`) |
| **Horizons** | 15 min · 1 h · 6 h · 12 h · 24 h (adapted to dataset frequency) |
| **Models** | Zero/Mean/Persistence baselines → Random Forest → XGBoost → LSTM → GRU → Transformer |
| **Validation** | Chronological splits only + held-out-site cross-site protocol; automated leakage tests |
| **Serving** | FastAPI backend (`/api/v1`), React + TypeScript dashboard |
| **Key Metrics** | Test MAE: 1.056 kWh (XGBoost), R²: 0.951 (see [RESULTS.md](RESULTS.md)) |

## Features

- **Leakage-safe validation**: Strict chronological splits (70/15/15) per site; cross-site held-out evaluation
- **Comprehensive modeling**: 6 model families from simple baselines to state-of-the-art transformers
- **Feature engineering**: 51 features including temporal, lag, rolling, weather, and solar position
- **Explainability**: SHAP values for XGBoost model (global and local interpretations)
- **Uncertainty quantification**: Conformal prediction with Mondrian splits (day/night, lag present/missing)
- **Production-ready API**: REST endpoints for forecasting, model metrics, and historical data
- **Interactive dashboard**: 6 pages for exploration, forecasting, model comparison, and explainability
- **Dockerized deployment**: Containerized services for backend, frontend, MLflow, and monitoring
- **Rigorous testing**: 170+ unit, API, and leakage tests with 100% pass rate

## Dataset Description

The UNISOLAR dataset consists of 4 CSV files in `unisolar/`:

- `Solar_Energy_Generation.csv`: Power generation (kWh) at 15-minute intervals for 42 sites across 5 campuses
- `Weather_Data_reordered_all.csv`: Weather variables (temperature, humidity, wind speed/direction, pressure)
- `Solar_Site_Details.csv`: Site metadata (capacity, installation date, latitude/longitude, elevation)
- `Monthly_Summary_Solar.csv`: Aggregated monthly generation and weather statistics

**Key characteristics**:
- 42 sites, 5 campuses, 15-minute resolution
- Date range: 2020-01-01 to 2022-04-30 (27 months)
- Target variable: `SolarGeneration` (56.2% missing values, structurally missing at night)
- Weather data: 80% missing for 2021-08 to 2022-04 (requires imputation strategy)
- No solar irradiance columns (solar position computed via pvlib using site coordinates)
- Timestamp timezone: Australia/Melbourne (consistent across dataset)

## Project Structure

```
E:\Solar_gemini
├── data/                  # Raw / interim / processed / external datasets (gitignored except processed)
│   ├── processed/         # Parquet files: solar/ (2.7M rows × 14 cols), site_details.parquet
│   └── ... 
├── notebooks/             # Numbered exploration & analysis notebooks
├── src/
│   ├── data/              # Loader, validator, cleaner, schema mapping
│   ├── features/          # Temporal, solar-position, lag, weather features
│   ├── models/            # Baseline, RF, XGBoost, LSTM, GRU, Transformer
│   ├── training/          # Train/evaluate/experiment pipelines
│   ├── explainability/    # SHAP explainers
│   └── forecasting/       # Predictor service, uncertainty estimation
├── api/                   # FastAPI app (routes, schemas)
├── frontend/              # React dashboard (TypeScript + Vite + Tailwind)
├── scripts/               # CLI entry points (inspect_dataset.py, train_model.py, ...)
├── tests/                 # Unit, integration, leakage, API tests
├── artifacts/             # Data profiles, EDA figures, model bundles, evaluation reports
├── configs/               # YAML configuration (data, training, models)
├── docker/                # Containerization assets (Dockerfiles, compose, configs)
└── unisolar/              # Raw dataset files (source data, not modified)
```

## Setup Instructions

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) (or Python 3.11–3.13 venv)
- [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/) (for full deployment)
- [Git](https://git-scm.com/)

### Development Setup (Conda)

```bash
# Clone repository
git clone <repository-url>
cd E:\Solar_gemini

# Create and activate conda environment
conda create -n solar python=3.13 -y
conda activate solar

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Non-Conda Alternative

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.\.venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Dataset Inspection (Phase 1)
```bash
python scripts/inspect_dataset.py
# Output: artifacts/data_profile.json and artifacts/data_profile.md
```

### 2. Data Engineering (Phase 2)
```bash
python scripts/build_features.py  # Runs full pipeline: schema mapping → cleaning → parquet conversion
# Output: data/processed/solar/ (partitioned parquet) and data/processed/site_details.parquet
```

### 3. Exploratory Data Analysis (Phase 3)
```bash
python scripts/run_eda.py
# Output: artifacts/eda/ (9 figures + eda_summary.md)
```

### 4. Baseline Models (Phase 4)
```bash
python scripts/run_baselines.py
# Output: artifacts/baselines/ (baseline_metrics.csv + baselines_report.md)
```

### 5. Feature Engineering (Phase 5)
```bash
python scripts/build_features.py
# Output: data/processed/features/ (2.7M rows × 51 cols) + artifacts/features/
```

### 6. Model Training & Evaluation
#### XGBoost (Phase 6)
```bash
python scripts/train_xgboost.py
# Output: models/xgboost_site_all_h1_v1.json + artifacts/xgboost/
```

#### LSTM/GRU (Phase 7)
```bash
python scripts/train_sequence.py --model lstm  # or --model gru
# Output: models/{lstm,gru}_site_all_h1_v1.pt + artifacts/{lstm,gru}/
```

#### Transformer (Phase 8)
```bash
python scripts/train_sequence.py --model transformer
# Output: artifacts/transformer/ (no separate checkpoint; weights in sequence_model.py)
```

#### Cross-Site Evaluation (Phase 9)
```bash
python scripts/run_cross_site.py
# Output: artifacts/cross_site/ (metrics.csv, report.md, metadata.json)
```

#### Explainability (Phase 10)
```bash
python scripts/run_shap.py
# Output: artifacts/shap/ (global CSV, PNGs, local contributions CSV)
```

#### Uncertainty Estimation (Phase 11)
```bash
python scripts/run_conformal.py
# Output: artifacts/uncertainty/ (conformal predictions and evaluation)
```

#### Model Comparison (All models)
```bash
python scripts/compare_models.py
# Output: artifacts/evaluation/ (comparison CSV, JSON, report.md)
```

### 7. API Backend (Phase 12)
```bash
# Start the FastAPI server
python scripts/run_api.py
# API available at: http://localhost:8000
# Interactive docs: http://localhost:8000/docs
```

### 8. Frontend Dashboard (Phase 13)
```bash
# Build frontend (for development)
cd frontend
npm install
npm run dev  # Vite dev server at http://localhost:5173

# For production build
npm run build  # Output to frontend/dist/
```

### 9. Testing (Phase 15)
```bash
# Run all tests
pytest

# Run specific test suites
pytest tests/unit/          # Unit tests
pytest tests/api/           # API tests
pytest tests/test_leakage.py # Leakage tests
```

### 10. Docker Deployment (Phase 16)
```bash
# Start all services (backend, frontend, mlflow, redis, postgres, prometheus, grafana)
docker-compose up -d

# Access points:
#   Frontend: http://localhost
#   Backend API: http://localhost:8000
#   MLflow UI: http://localhost:5000
#   Prometheus: http://localhost:9090
#   Grafana: http://localhost:3000 (admin/admin or set GF_SECURITY_ADMIN_PASSWORD)

# Stop services
docker-compose down
```

## Technology Stack

| Category      | Technologies                                                                 |
|---------------|------------------------------------------------------------------------------|
| **ML**        | Python 3.13, NumPy, Pandas, SciPy, scikit-learn, XGBoost, PyTorch, SHAP, MLflow |
| **Data**      | Parquet via PyArrow, CSV                                                     |
| **Backend**   | FastAPI, Pydantic, Uvicorn                                                   |
| **Frontend**  | React 19, TypeScript, Vite (rolldown), Tailwind CSS 4, shadcn/ui, Recharts   |
| **Infra**     | Docker, Docker Compose, Nginx, PostgreSQL, Redis, MLflow, Prometheus, Grafana |
| **Testing**   | Pytest                                                                       |
| **Monitoring**| Prometheus, Grafana                                                          |

## Model Training Details

### XGBoost
- **Configuration**: `configs/models.yaml` and `configs/training.yaml`
- **Training**: Single-step (15-min forecast), early stopping (iter 109/2000)
- **Hardware**: CPU training (10.6 seconds per run)
- **Performance**: 
  - Test MAE: 1.056 kWh
  - Test RMSE: 2.530 kWh
  - Test R²: 0.951
- **Artifacts**: 
  - Model: `models/xgboost_site_all_h1_v1.json`
  - Metrics: `artifacts/xgboost/metrics.csv`
  - Predictions: `artifacts/xgboost/predictions_test.parquet`
  - Feature importance: `artifacts/xgboost/feature_importance.csv`

### LSTM/GRU
- **Architecture**: 24-hour lookback × 13 channels (including power observed mask)
- **Training**: Early stopping, LR scheduling, gradient clipping
- **Performance**:
  - LSTM: Test MAE 1.140, RMSE 2.603, R² 0.948
  - GRU: Test MAE 1.147, RMSE 2.551, R² 0.950
- **Artifacts**:
  - Checkpoints: `models/{lstm,gru}_site_all_h1_v1.pt`
  - Metrics: `artifacts/{lstm,gru}/metrics.csv`
  - Predictions: `artifacts/{lstm,gru}/predictions_test.pt`

### Transformer
- **Architecture**: Pre-LN encoder, sinusoidal positional encoding, no causal mask
- **Training**: Same windowing as LSTM/GRU with leak guard
- **Performance**:
  - Test MAE: 1.124 kWh
  - Test RMSE: 2.544 kWh
  - Test R²: 0.950
- **Artifacts**: 
  - Metrics: `artifacts/transformer/metrics.csv`
  - Predictions: `artifacts/transformer/predictions_test.parquet`

## API Endpoints

The FastAPI backend (`/api/v1`) provides:

| Endpoint                          | Method | Description                                                                 |
|-----------------------------------|--------|-----------------------------------------------------------------------------|
| `/health`                         | GET    | Health check                                                                |
| `/dataset`                        | GET    | Dataset metadata (shape, columns, missingness)                              |
| `/sites`                          | GET    | List of all site IDs                                                        |
| `/models/metrics`                 | GET    | Test-split metrics for all models                                           |
| `/history/{site_id}`              | GET    | Historical generation data for a site (with filtering/resampling)           |
| `/forecast`                       | POST   | Single-site forecast (recursive multi-step, horizon ≤96 steps)              |
| `/forecast/batch`                 | POST   | Batch forecast (≤10 sites)                                                  |
| `/models/{model_id}/metrics`      | GET    | Per-model metrics (from phase artifact CSVs)                                |

## Frontend Dashboard

The React dashboard includes 6 pages:

1. **Dashboard**: Overview of system status, recent forecasts, and key metrics
2. **Forecast**: Interactive forecaster with model selection, horizon slider, and uncertainty bounds
3. **Sites**: Site metadata explorer with maps and capacity comparisons
4. **Model Comparison**: Side-by-side model performance metrics and prediction visualizations
5. **Explainability**: SHAP values (global feature importance, local explanations)
6. **Data Quality**: Missingness templates, monthly gap timelines, and weather/power correlations

**Key features**:
- Model selection (XGBoost, LSTM, GRU, Transformer, Persistence)
- Uncertainty visualization (conformal prediction bounds)
- Night/day shading based on solar elevation
- Recursive multi-step forecasting (up to 24 hours)
- Caching for improved responsiveness
- Dark/light theme toggle

## Testing Summary

- **Total tests**: 170+ (unit, API, leakage)
- **Pass rate**: 100% (all suites green)
- **Test categories**:
  - Data validation: timestamps, missing values, duplicates, schema
  - Feature engineering: lag/rolling/temporal/solar feature correctness
  - Data splitting: chronological and cross-site split ordering
  - Model I/O: dimension checking for all model types
  - API: health, dataset, sites, forecast, history, models, metrics endpoints
  - Leakage prevention: future data in lags/rolling, test set in scaling/tuning

## Deployment with Docker

The `docker-compose.yml` defines 7 services:

| Service     | Description                                                                 |
|-------------|-----------------------------------------------------------------------------|
| `backend`   | FastAPI API server (port 8000)                                              |
| `frontend`  | Nginx serving React build (ports 80, 443)                                   |
| `db`        | PostgreSQL 16-alpine (for potential future use)                             |
| `mlflow`    | MLflow tracking server (port 5000)                                          |
| `redis`     | Redis cache (port 6379)                                                     |
| `prometheus`| Monitoring server (port 9090)                                               |
| `grafana`   | Dashboarding platform (port 3000)                                           |

**Volumes**:
- `postgres_data`: PostgreSQL persistent storage
- `redis_data`: Redis persistent storage
- `prometheus_data`: Prometheus TSDB storage
- `grafana_data`: Grafana persistent storage

**Networks**:
- `solar-network`: Internal bridge service network

## Results and Artifacts

Key measured outcomes are documented in [RESULTS.md](RESULTS.md):

- **Baselines**: Persistence MAE 2.783 kWh (best baseline)
- **XGBoost**: Test MAE 1.056 kWh (best overall model)
- **LSTM**: Test MAE 1.140 kWh
- **GRU**: Test MAE 1.147 kWh
- **Transformer**: Test MAE 1.124 kWh (best deep model)
- **Cross-site**: Transformer best unseen-site MAE 0.764 kWh
- **Explainability**: Top 2 features (lag_1: 34.9%, roll_mean_1h: 23.1%) ≈ 64% of SHAP attribution
- **Uncertainty**: XGBoost 90% coverage ±2.92 kWh (Mondrian night_lag: 0.88 kWh width)

All model artifacts, metrics, and predictions are stored in the `artifacts/` directory and tracked via MLflow.

## Project Guidelines

1. **Correctness over complexity**: Prefer validated, leakage-free models over untested complex architectures
2. **Reproducibility**: All training runs use fixed seeds and config-driven hyperparameters
3. **Data integrity**: Never impute power generation values (structural missingness respected)
4. **Temporal safety**: All features use strictly historical information (no future leakage)
5. **Site-aware processing**: Features computed per-site to respect spatial heterogeneity
6. **Transparent reporting**: Results.md contains only measured outcomes (no fabrication)
7. **Continuous testing**: 170+ automated tests validate all phases

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your contributions:
- Pass all existing tests (`pytest`)
- Follow the existing code style
- Update documentation as needed
- Include unit tests for new functionality

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The UNISOLAR dataset providers
- open-source libraries: XGBoost, PyTorch, SHAP, FastAPI, React, etc.
- NVIDIA/NeMo for platform support