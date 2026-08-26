# UNISOLAR Solar Power Generation Prediction Platform

Multi-site solar photovoltaic (PV) power forecasting platform built on the **UNISOLAR** dataset.
Compares statistical baselines, classical machine learning, and deep-learning models across
multiple forecast horizons with leakage-safe temporal validation, unseen-site generalization
testing, explainable AI, and uncertainty estimation — served through a REST API and an
interactive dashboard.

> Full specification: [PRD.md](PRD.md) · Task tracking: [TASKS.md](TASKS.md) ·
> Decision log: [DECISIONS.md](DECISIONS.md) · Measured outcomes: [RESULTS.md](RESULTS.md)

## Project status

**Phase 0 — Repository initialization.** No models trained yet; see TASKS.md for progress.

## Overview

| | |
|---|---|
| **Objective** | Predict PV power generation from historical generation, weather, irradiance, and temporal features |
| **Dataset** | UNISOLAR multi-site PV generation + weather CSVs (`unisolar/`) |
| **Horizons** | 15 min · 1 h · 6 h · 12 h · 24 h (adapted to dataset frequency) |
| **Models** | Zero/Mean/Persistence baselines → Random Forest → XGBoost → LSTM → GRU → Transformer |
| **Validation** | Chronological splits only + held-out-site cross-site protocol; automated leakage tests |
| **Serving** | FastAPI backend (`/api/v1`), React + TypeScript dashboard |

## Project structure

```text
├── data/            raw / interim / processed / external datasets (gitignored except processed)
├── notebooks/       numbered exploration & analysis notebooks
├── src/
│   ├── data/        loader, validator, cleaner, schema mapping
│   ├── features/    temporal, solar-position, lag, weather features
│   ├── models/      baseline, RF, XGBoost, LSTM, GRU, Transformer
│   ├── training/    train/evaluate/experiment pipelines
│   ├── explainability/  SHAP explainers
│   └── forecasting/ predictor service, uncertainty estimation
├── api/             FastAPI app (routes, schemas)
├── frontend/        React dashboard
├── scripts/         CLI entry points (inspect_dataset.py, train_model.py, ...)
├── tests/           unit, integration, leakage, API tests
├── artifacts/       data profiles, EDA figures, model bundles, evaluation reports
├── configs/         YAML configuration (data, training, models)
├── docker/          containerization assets
└── unisolar/        raw dataset files (source data, not modified)
```

## Setup

Development uses a conda environment named `solar` with Python 3.13.

```bash
conda create -n solar python=3.13 -y
conda activate solar
pip install -r requirements.txt
```

Non-conda alternative: any Python 3.11–3.13 venv works with the same `requirements.txt`.

## Usage (planned)

```bash
# Inspect and profile the dataset (Phase 1)
python scripts/inspect_dataset.py

# Train a model (Phase 6+)
python scripts/train_model.py --model xgboost

# Serve the API (Phase 12)
uvicorn api.main:app --reload
```

## Technology stack

- **ML:** Python 3.11+, NumPy, Pandas, SciPy, scikit-learn, XGBoost, PyTorch, SHAP, MLflow
- **Data:** Parquet via PyArrow
- **Backend:** FastAPI, Pydantic, Uvicorn
- **Frontend:** React, TypeScript, Vite, Tailwind CSS, shadcn/ui, Recharts
- **Infra:** Docker, Docker Compose, Nginx
- **Testing:** Pytest (+ frontend test framework)

## Guiding principle

Correctness and reproducibility over model complexity. A reliable XGBoost model with rigorous,
leakage-free evaluation is preferable to an unvalidated Transformer.
