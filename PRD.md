# Product Requirements Document

## UNISOLAR Solar Power Generation Prediction Platform

**Version:** 1.0
**Status:** Ready for implementation
**Project Type:** Pure Software / Machine Learning / Time-Series Forecasting
**Primary Dataset:** UNISOLAR
**Primary Objective:** Predict photovoltaic (PV) power generation from historical PV generation, weather, irradiance, temporal, and site-related information.

---

# 1. Executive Summary

Build a production-quality software platform for **solar photovoltaic power generation forecasting** using the UNISOLAR dataset.

The system will ingest historical multi-site photovoltaic generation and environmental data, perform automated data validation and preprocessing, engineer temporal and solar features, train and compare multiple forecasting models, evaluate them using leakage-safe time-series methodologies, and expose the best model through an API and interactive web dashboard.

The platform must support:

1. Dataset ingestion and validation
2. Exploratory data analysis
3. Data cleaning
4. Temporal feature engineering
5. Solar/weather feature engineering
6. Time-series window generation
7. Baseline forecasting
8. Traditional ML models
9. Deep-learning forecasting models
10. Multi-site forecasting
11. Cross-site generalization experiments
12. Model evaluation
13. Forecast visualization
14. Prediction uncertainty
15. Explainable AI
16. Model/version tracking
17. REST API
18. Interactive dashboard
19. Reproducible experiments
20. Dockerized deployment

The final system should be suitable for a **final-year engineering project, portfolio project, or research-oriented prototype**.

---

# 2. Problem Statement

Solar photovoltaic generation is highly variable because of:

* Solar irradiance
* Cloud cover
* Temperature
* Humidity
* Wind
* Time of day
* Season
* Weather conditions
* Site-specific characteristics

Accurate solar generation forecasting is useful for:

* Grid management
* Battery scheduling
* Energy trading
* Demand planning
* Solar farm operations
* Maintenance planning
* Renewable-energy integration

The system must answer:

> Given historical PV generation and available environmental/time information, how much solar power will a PV installation generate in the future?

The project must evaluate both:

* **single-site forecasting**
* **multi-site/cross-site forecasting**

---

# 3. Project Goals

## 3.1 Primary Goal

Develop a forecasting system capable of predicting future PV power generation at multiple forecasting horizons.

## 3.2 Secondary Goals

The system should:

* Establish strong statistical/naive baselines
* Compare classical ML against deep learning
* Prevent temporal data leakage
* Quantify forecasting uncertainty
* Explain predictions using XAI
* Visualize actual vs predicted generation
* Support multiple PV sites
* Evaluate generalization to unseen sites
* Provide reproducible training pipelines
* Expose predictions through an API

---

# 4. Non-Goals

The initial version must NOT attempt to:

* Control physical solar equipment
* Control inverters
* Implement physical MPPT hardware
* Collect data from physical sensors
* Build an IoT device
* Perform real-time hardware control
* Guarantee grid-level production accuracy
* Replace professional energy-management systems

This is a **100% software project**.

---

# 5. Dataset

## 5.1 Primary Dataset

Use:

**UNISOLAR**

The dataset is intended to provide multi-site photovoltaic generation and environmental/weather information suitable for solar forecasting.

The implementation agent MUST verify the downloaded dataset's:

* File names
* File formats
* Number of records
* Number of sites
* Sampling interval
* Timestamp format
* Target/output columns
* Weather columns
* Irradiance columns
* Missing-value patterns
* Units
* Site identifiers
* License/usage terms

Do NOT assume column names.

Create an automatic dataset-inspection script before implementing the ML pipeline.

---

# 6. Dataset Discovery Requirement

The first implementation task must be:

```text
scripts/inspect_dataset.py
```

It must:

1. Locate dataset files.
2. List all files.
3. Identify CSV/Parquet/other tabular files.
4. Print shape.
5. Print columns.
6. Print dtypes.
7. Print first five rows.
8. Print last five rows.
9. Detect timestamp columns.
10. Detect possible site-ID columns.
11. Detect target/PV generation columns.
12. Calculate missing-value percentages.
13. Calculate unique values for categorical columns.
14. Determine sampling frequency.
15. Detect duplicated timestamps.
16. Detect duplicated rows.
17. Identify obvious unit metadata.
18. Generate a dataset profile.

Output:

```text
artifacts/data_profile.json
artifacts/data_profile.md
```

No downstream model development should begin until the dataset schema is verified.

---

# 7. Expected Data Model

The canonical internal representation should be normalized to:

```text
timestamp
site_id
power
irradiance
temperature
humidity
wind_speed
wind_direction
pressure
cloud_cover
other_weather_features...
```

Actual source column names may differ.

The preprocessing layer must map source columns to the canonical schema.

---

# 8. Data Pipeline

Implement:

```text
Raw Dataset
    ↓
Dataset Inspection
    ↓
Schema Mapping
    ↓
Validation
    ↓
Cleaning
    ↓
Timestamp Normalization
    ↓
Site Normalization
    ↓
Missing-Value Handling
    ↓
Outlier Detection
    ↓
Feature Engineering
    ↓
Window Generation
    ↓
Train/Validation/Test
```

---

# 9. Data Quality Requirements

The system must detect:

### Missing timestamps

Identify gaps in each site's time series.

### Duplicate timestamps

Detect:

```text
site_id + timestamp
```

duplicates.

### Missing values

Report missingness by:

* site
* variable
* date
* percentage

### Impossible values

Examples:

```text
negative solar power
negative irradiance
humidity > 100%
invalid timestamps
negative physical measurements where impossible
```

Do not blindly delete observations.

Every cleaning operation must be logged.

---

# 10. Solar-Specific Data Handling

Night-time solar generation must be handled carefully.

The pipeline should identify night periods using:

1. irradiance when available
2. solar elevation/position when calculable
3. generation value

Do NOT simply remove all zero-power observations.

Night-time periods are valid observations and can be important for forecasting.

However, evaluation should provide both:

* all-period metrics
* daylight-only metrics

This prevents trivial night-time zeros from artificially improving model performance.

---

# 11. Train/Validation/Test Strategy

This is a critical requirement.

## 11.1 No random row splitting

Do NOT use:

```python
train_test_split(...)
```

on the full time-series dataset.

That would create temporal leakage.

## 11.2 Chronological split

For each site:

```text
Historical data
       ↓
Train
       ↓
Validation
       ↓
Test
```

Recommended starting split:

```text
70% Train
15% Validation
15% Test
```

The split must preserve temporal ordering.

The exact proportions may be configurable.

---

# 12. Cross-Site Generalization

Because UNISOLAR is multi-site, implement a second evaluation protocol.

Example:

```text
Sites 01–30 → Training
Sites 31–36 → Validation
Sites 37–42 → Test
```

The exact site counts must be determined dynamically from the dataset.

The system must ensure:

> Test sites are completely unseen during training.

This experiment answers:

> Can the model generalize to a PV site that was not present during training?

This should be a major project result.

---

# 13. Forecasting Horizons

The platform must support configurable forecasting horizons.

Primary horizons:

```text
15 minutes
1 hour
6 hours
12 hours
24 hours
```

If the source resolution differs, automatically convert horizons to the corresponding number of time steps.

For example, with 15-minute data:

```text
15 min = 1 step
1 hour = 4 steps
6 hours = 24 steps
12 hours = 48 steps
24 hours = 96 steps
```

The exact horizon configuration must be determined from the actual dataset frequency.

---

# 14. Forecasting Tasks

Support:

## Task A — Single-step forecasting

Predict:

```text
P(t+1)
```

## Task B — Multi-step forecasting

Predict:

```text
P(t+1), P(t+2), ..., P(t+N)
```

## Task C — Day-ahead forecasting

Predict the next 24-hour PV generation profile.

## Task D — Cross-site forecasting

Train on some sites and predict unseen sites.

---

# 15. Feature Engineering

## 15.1 Temporal Features

Create:

```text
hour
minute
day
day_of_week
day_of_year
week_of_year
month
quarter
season
is_weekend
```

Use cyclical encoding for periodic features:

```text
sin_hour
cos_hour
sin_day_of_year
cos_day_of_year
```

---

# 16. Lag Features

Create configurable lag variables.

Examples:

```text
power_lag_1
power_lag_2
power_lag_4
power_lag_8
power_lag_24
power_lag_48
power_lag_96
```

Only create lags appropriate to the dataset sampling frequency.

---

# 17. Rolling Features

Examples:

```text
power_rolling_mean
power_rolling_std
power_rolling_min
power_rolling_max
```

with configurable windows.

IMPORTANT:

Rolling features must only use historical observations.

Future observations must never enter a rolling feature.

---

# 18. Solar Position Features

If site latitude/longitude and timestamp are available, calculate:

```text
solar_elevation
solar_azimuth
solar_zenith
day_length
```

Use a reliable solar-position library.

These features should be optional if site metadata is unavailable.

---

# 19. Weather Features

Use available weather variables such as:

```text
irradiance
temperature
humidity
wind_speed
wind_direction
pressure
cloud_cover
```

The preprocessing pipeline must dynamically adapt to available variables.

---

# 20. Feature Selection

Implement:

1. Correlation analysis
2. Mutual information
3. Feature importance
4. SHAP importance
5. Ablation experiments

The system must determine which features contribute most to forecasting accuracy.

---

# 21. Baseline Models

Before deep learning, implement:

## Baseline 1 — Zero baseline

Predict:

```text
0
```

Useful primarily as a sanity check.

## Baseline 2 — Mean baseline

Predict historical mean.

## Baseline 3 — Persistence

For the same time-of-day:

```text
P(t+h) ≈ P(t+h-previous_day)
```

Persistence must be treated as the primary baseline.

---

# 22. Machine Learning Models

Implement at least:

### Random Forest

Use for baseline tree-based comparison.

### XGBoost

Primary traditional ML model.

### LightGBM

Optional but recommended.

Models must be configurable.

---

# 23. Deep Learning Models

Implement:

### LSTM

Input:

```text
lookback window × features
```

Output:

```text
future power
```

### GRU

Use as a lighter recurrent model.

### Transformer

Implement a time-series Transformer only after the simpler models are working.

Do NOT begin with Transformer.

---

# 24. Recommended Development Order

The AI agent MUST implement in this order:

```text
1. Dataset inspection
2. Data validation
3. Cleaning
4. EDA
5. Baselines
6. Feature engineering
7. XGBoost
8. LSTM
9. GRU
10. Transformer
11. Cross-site evaluation
12. Uncertainty estimation
13. SHAP
14. API
15. Dashboard
16. Deployment
```

Do not implement everything simultaneously.

---

# 25. Evaluation Metrics

Required:

### MAE

Mean Absolute Error.

### RMSE

Root Mean Squared Error.

### R²

Coefficient of determination.

### nRMSE

Normalize RMSE using an explicitly documented denominator such as site capacity or observed range.

### Daylight MAE

MAE calculated only during daylight periods.

### Daylight nRMSE

Normalized daylight error.

Do NOT rely exclusively on MAPE because PV output contains zero or near-zero values.

---

# 26. Evaluation Report

Automatically generate:

```text
artifacts/evaluation/
```

containing:

```text
model_comparison.csv
model_comparison.json
evaluation_report.md
```

Include:

```text
Model
Site
Forecast horizon
MAE
RMSE
R2
nRMSE
Daylight MAE
Training time
Inference time
```

---

# 27. Visualization Requirements

Generate:

### Actual vs predicted

```text
Time → Power
Actual
Predicted
```

### Daily generation

Compare complete daily profiles.

### Forecast error

Plot:

```text
Actual - Prediction
```

### Error distribution

Histogram/KDE or equivalent.

### Hourly error

Show error by hour of day.

### Site comparison

Compare model performance across sites.

### Forecast horizon comparison

Show performance degradation as horizon increases.

---

# 28. Uncertainty Estimation

Implement prediction intervals in the advanced stage.

Target output:

```json
{
  "timestamp": "...",
  "prediction": 4.82,
  "lower_bound": 4.21,
  "upper_bound": 5.36,
  "confidence_level": 0.90
}
```

Possible approaches:

* Quantile regression
* Conformal prediction
* Monte Carlo dropout

Prefer **conformal prediction** for a model-agnostic implementation if practical.

---

# 29. Explainable AI

Use SHAP.

The system should answer:

> Why did the model predict this power value?

Provide:

```text
Global feature importance
Local prediction explanation
Feature contribution plots
```

Example:

```text
Prediction = 4.82 kW

Positive contributors:
+ irradiance
+ historical power
+ solar elevation

Negative contributors:
- cloud cover
- humidity
```

---

# 30. Experiment Tracking

Use MLflow.

Track:

```text
experiment
model
dataset version
site
forecast horizon
features
hyperparameters
metrics
training duration
model artifact
```

Each model must have a unique version.

Example:

```text
xgboost-site-all-horizon-24h-v1
lstm-site-all-horizon-24h-v1
```

---

# 31. Model Registry

The system must maintain:

```text
models/
```

and/or MLflow Model Registry.

Only models that pass validation can be promoted to production.

---

# 32. Model Selection

The production model should NOT simply be the model with the highest training score.

Selection must be based on validation performance.

Primary ranking metric:

```text
RMSE
```

Secondary:

```text
MAE
R²
Daylight MAE
Inference time
```

---

# 33. API Requirements

Backend:

**FastAPI**

Base:

```text
/api/v1
```

## Health

```http
GET /api/v1/health
```

## Dataset information

```http
GET /api/v1/dataset
```

## Sites

```http
GET /api/v1/sites
```

## Available models

```http
GET /api/v1/models
```

## Forecast

```http
POST /api/v1/forecast
```

Request:

```json
{
  "site_id": "site_01",
  "forecast_horizon": 24,
  "model": "xgboost"
}
```

Response:

```json
{
  "site_id": "site_01",
  "model": "xgboost",
  "forecast_horizon": 24,
  "predictions": [
    {
      "timestamp": "2026-01-01T10:00:00Z",
      "prediction": 4.82
    }
  ]
}
```

---

# 34. Batch Forecast API

```http
POST /api/v1/forecast/batch
```

Allow forecasting multiple sites.

---

# 35. Historical Data API

```http
GET /api/v1/sites/{site_id}/history
```

Parameters:

```text
start
end
resolution
```

---

# 36. Model Evaluation API

```http
GET /api/v1/models/{model_id}/metrics
```

Return:

```text
MAE
RMSE
R²
nRMSE
daylight metrics
```

---

# 37. Frontend

Use:

```text
React
TypeScript
Vite
Tailwind CSS
shadcn/ui
Recharts
```

The frontend must be responsive.

---

# 38. Dashboard Pages

## Dashboard

Display:

```text
Current/Latest Power
Today's Energy
Predicted Next 24h
Best Model
Model Accuracy
Selected Site
```

## Forecast

Interactive:

```text
Actual vs Forecast
```

Controls:

```text
Site
Date
Forecast horizon
Model
```

## Sites

Display:

```text
All sites
Site performance
Average generation
Data availability
```

## Model Comparison

Display:

```text
Persistence
Random Forest
XGBoost
LSTM
GRU
Transformer
```

with metric comparisons.

## Explainability

Display SHAP feature importance.

## Data Quality

Display:

```text
Missing values
Duplicate records
Time gaps
Outliers
```

---

# 39. Dashboard UX

The dashboard should be designed for technical users.

Use:

* Clear charts
* Tooltips
* Date-range selection
* Site selection
* Model selection
* Forecast horizon selection
* Metric cards
* Downloadable predictions
* Downloadable evaluation results

Do not overload the dashboard with unnecessary animations.

---

# 40. Database

PostgreSQL is optional for the first ML prototype.

If persistence is required, create:

```text
sites
datasets
models
experiments
forecasts
evaluation_metrics
```

Do NOT store the entire raw dataset in PostgreSQL unless there is a demonstrated requirement.

Large historical datasets should remain in Parquet files/object storage.

---

# 41. Recommended Storage Format

After ingestion, convert normalized data to:

```text
Parquet
```

Partition by:

```text
site_id
year
month
```

Example:

```text
data/
  raw/
  processed/
    site_id=site_01/
    site_id=site_02/
```

---

# 42. Project Structure

Use:

```text
solar-power-prediction/
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_baselines.ipynb
│   ├── 04_xgboost.ipynb
│   ├── 05_lstm.ipynb
│   └── 06_model_comparison.ipynb
│
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   ├── validator.py
│   │   ├── cleaner.py
│   │   └── schema.py
│   │
│   ├── features/
│   │   ├── temporal.py
│   │   ├── solar.py
│   │   ├── lag.py
│   │   └── weather.py
│   │
│   ├── models/
│   │   ├── baseline.py
│   │   ├── random_forest.py
│   │   ├── xgboost_model.py
│   │   ├── lstm.py
│   │   ├── gru.py
│   │   └── transformer.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── experiment.py
│   │
│   ├── explainability/
│   │   └── shap_explainer.py
│   │
│   └── forecasting/
│       ├── predictor.py
│       └── uncertainty.py
│
├── api/
│   ├── main.py
│   ├── routes/
│   └── schemas/
│
├── frontend/
│
├── scripts/
│   ├── inspect_dataset.py
│   ├── validate_dataset.py
│   ├── prepare_dataset.py
│   └── train_model.py
│
├── tests/
│
├── artifacts/
│
├── configs/
│   ├── data.yaml
│   ├── training.yaml
│   └── models.yaml
│
├── docker/
│
├── requirements.txt
├── pyproject.toml
├── docker-compose.yml
├── README.md
├── PRD.md
├── TASKS.md
├── DECISIONS.md
└── RESULTS.md
```

---

# 43. Configuration

Do not hard-code:

* Dataset path
* Site IDs
* Forecast horizon
* Lookback window
* Model hyperparameters
* Train/test ratio
* Random seeds

Use YAML configuration.

Example:

```yaml
forecast:
  horizons:
    - 1
    - 4
    - 24
    - 96

training:
  seed: 42
  validation_ratio: 0.15
  test_ratio: 0.15

models:
  xgboost:
    enabled: true

  lstm:
    enabled: true

  gru:
    enabled: true

  transformer:
    enabled: true
```

The exact horizon values must be adapted to the verified dataset frequency.

---

# 44. Reproducibility

Every experiment must record:

```text
random seed
Python version
package versions
dataset version/hash
configuration
model parameters
training timestamp
```

Set deterministic seeds where practical.

---

# 45. Testing

Implement unit tests for:

### Data

* Timestamp parsing
* Missing values
* Duplicate detection
* Schema validation

### Features

* Lag calculations
* Rolling features
* Cyclic features
* Solar features

### Splitting

Verify:

```text
max(train_timestamp) < min(validation_timestamp)
max(validation_timestamp) < min(test_timestamp)
```

### Models

Verify model input/output dimensions.

### API

Test:

```text
health
sites
forecast
history
models
metrics
```

---

# 46. Data Leakage Tests

Create explicit automated tests.

The pipeline must detect if:

* Future power enters lag features
* Future weather enters features
* Rolling windows include future observations
* Test data is used during scaling
* Test data is used during feature selection
* Test sites appear in training
* Hyperparameter tuning uses the test set

Any leakage must fail CI.

---

# 47. Scaling

For neural networks:

* Fit scalers only on training data.
* Apply fitted scalers to validation/test.

For tree models:

* Scaling is generally unnecessary.

Save preprocessing artifacts with the model.

---

# 48. Training Pipeline

Command:

```bash
python scripts/train_model.py --model xgboost
```

Expected workflow:

```text
Load configuration
        ↓
Load processed data
        ↓
Create temporal split
        ↓
Create features
        ↓
Train
        ↓
Validate
        ↓
Evaluate
        ↓
Save model
        ↓
Save metrics
        ↓
Register experiment
```

---

# 49. Deep Learning Training

Use PyTorch.

Required:

* GPU support
* Mixed precision when CUDA is available
* Early stopping
* Learning-rate scheduling
* Checkpointing
* Best-model restoration
* Gradient clipping
* TensorBoard/MLflow logging

CPU fallback must work.

---

# 50. Hardware Acceleration

The software should automatically detect:

```text
CUDA GPU
```

and use it for deep learning.

If unavailable:

```text
CPU
```

must remain supported.

Do not require a GPU to run preprocessing, XGBoost, inference, or the API.

---

# 51. Model Artifacts

Save:

```text
artifacts/models/
```

Each model should contain:

```text
model
config
feature list
scaler
dataset version
metrics
training metadata
```

Example:

```text
artifacts/models/xgboost_v1/
    model.json
    features.json
    config.yaml
    metrics.json
```

---

# 52. Results Tracking

Create:

```text
RESULTS.md
```

The file must contain actual measured results only.

Do NOT fabricate metrics.

A model cannot be marked successful until:

* Training completes
* Evaluation completes
* Artifacts exist
* Metrics are recorded

---

# 53. Experiment Matrix

The final report should compare:

```text
                     Horizon
Model          15m   1h   6h   12h   24h
------------------------------------------------
Persistence
Random Forest
XGBoost
LSTM
GRU
Transformer
```

Also compare:

```text
Single-site
vs
Multi-site
vs
Cross-site
```

---

# 54. Ablation Study

At minimum perform:

### Experiment A

Historical power only.

### Experiment B

Historical power + temporal features.

### Experiment C

Historical power + weather.

### Experiment D

Historical power + weather + solar features.

### Experiment E

All features.

This determines how much each feature group contributes.

---

# 55. Key Research Questions

The project should answer:

1. How accurately can PV generation be predicted?
2. Which ML model performs best?
3. Does deep learning outperform XGBoost?
4. How does accuracy change as forecast horizon increases?
5. How important are weather variables?
6. How important are historical PV values?
7. Does a multi-site model outperform independent site models?
8. Can a model generalize to unseen PV sites?
9. Which features contribute most to predictions?
10. How reliable are prediction intervals?

---

# 56. Acceptance Criteria

The project is considered complete only when all of the following are satisfied:

* [ ] UNISOLAR dataset successfully ingested
* [ ] Dataset schema automatically inspected
* [ ] Data-quality report generated
* [ ] Timestamp integrity validated
* [ ] Missing data analyzed
* [ ] Temporal leakage tests implemented
* [ ] Chronological train/validation/test split implemented
* [ ] Cross-site split implemented
* [ ] Persistence baseline implemented
* [ ] Random Forest implemented
* [ ] XGBoost implemented
* [ ] LSTM implemented
* [ ] GRU implemented
* [ ] Transformer implemented
* [ ] Metrics calculated
* [ ] Daylight metrics calculated
* [ ] Model comparison generated
* [ ] Forecast visualization generated
* [ ] SHAP implemented
* [ ] Uncertainty estimation implemented
* [ ] Best model selected using validation data
* [ ] Model artifact saved
* [ ] FastAPI API implemented
* [ ] React dashboard implemented
* [ ] Tests implemented
* [ ] Docker deployment implemented
* [ ] README completed
* [ ] RESULTS.md contains actual results
* [ ] No fabricated metrics
* [ ] No test-set leakage
* [ ] Reproducible training confirmed

---

# 57. Implementation Phases

## Phase 0 — Repository Initialization

Create:

```text
README.md
PRD.md
TASKS.md
DECISIONS.md
RESULTS.md
```

Set up Python environment and dependency management.

---

## Phase 1 — Dataset Acquisition & Inspection

Tasks:

* Obtain UNISOLAR
* Store raw data
* Inspect files
* Determine schema
* Determine frequency
* Identify site IDs
* Identify target
* Generate data profile

Deliverable:

```text
artifacts/data_profile.*
```

---

## Phase 2 — Data Engineering

Implement:

* Cleaning
* Validation
* Timestamp normalization
* Missing-value analysis
* Duplicate detection
* Outlier analysis
* Parquet conversion

Deliverable:

```text
data/processed/
```

---

## Phase 3 — EDA

Generate:

* Generation distributions
* Daily profiles
* Monthly profiles
* Site comparisons
* Weather correlations
* Missingness plots
* Time-series plots

Deliverable:

```text
artifacts/eda/
```

---

## Phase 4 — Baselines

Implement:

* Zero
* Mean
* Persistence

Deliverable:

```text
baseline_metrics.csv
```

---

## Phase 5 — Feature Engineering

Implement:

* Temporal features
* Lag features
* Rolling features
* Weather features
* Solar-position features

Deliverable:

```text
feature_metadata.json
```

---

## Phase 6 — XGBoost

Train and evaluate XGBoost.

Deliver:

* Model
* Metrics
* Predictions
* Feature importance

---

## Phase 7 — LSTM / GRU

Implement sequence generation and recurrent models.

Deliver:

* Training pipeline
* Checkpoints
* Metrics
* Predictions

---

## Phase 8 — Transformer

Implement only after LSTM/GRU pipelines are stable.

Compare against previous models.

---

## Phase 9 — Cross-Site Evaluation

Implement:

```text
seen-site evaluation
unseen-site evaluation
```

Generate dedicated report.

---

## Phase 10 — Explainability

Implement SHAP for the selected model(s).

Generate:

```text
global importance
local explanations
```

---

## Phase 11 — Uncertainty

Implement prediction intervals.

Evaluate coverage and interval width.

---

## Phase 12 — Backend

Implement FastAPI.

---

## Phase 13 — Frontend

Implement React dashboard.

---

## Phase 14 — Integration

Connect:

```text
React
 ↓
FastAPI
 ↓
Forecasting Service
 ↓
Model
 ↓
Processed Dataset
```

---

## Phase 15 — Testing

Run:

```text
unit tests
integration tests
data leakage tests
API tests
model inference tests
```

---

## Phase 16 — Docker

Provide:

```text
docker-compose.yml
```

Services:

```text
frontend
backend
ml service
```

PostgreSQL may be added if required.

---

# 58. AI Agent Operating Rules

The coding agent MUST follow these rules.

## Rule 1

Inspect the repository before modifying it.

## Rule 2

Inspect the actual UNISOLAR dataset before assuming its schema.

## Rule 3

Never fabricate dataset columns.

## Rule 4

Never fabricate model metrics.

## Rule 5

Never mark a task DONE unless the artifact exists.

## Rule 6

Do not randomly split time-series rows.

## Rule 7

Do not leak test data into preprocessing.

## Rule 8

Do not begin with Transformer.

## Rule 9

Keep every experiment reproducible.

## Rule 10

Document important architecture decisions in:

```text
DECISIONS.md
```

## Rule 11

Update:

```text
TASKS.md
```

after every completed implementation unit.

## Rule 12

Run tests after meaningful changes.

## Rule 13

Prefer small, verifiable implementation steps.

## Rule 14

Do not rewrite working code unnecessarily.

## Rule 15

If a requirement is ambiguous, inspect the dataset/project state first and make the smallest defensible assumption. Record the assumption.

---

# 59. Definition of Done

A phase is DONE only if:

1. Code exists.
2. Tests exist where applicable.
3. Tests pass.
4. Expected artifacts exist.
5. Results are documented.
6. TASKS.md is updated.
7. No known leakage exists.
8. No placeholder implementation remains.

A model-training task is NOT DONE merely because training code exists.

It is DONE only when:

```text
Dataset
+
Training
+
Evaluation
+
Model artifact
+
Metrics
```

all exist successfully.

---

# 60. Final Deliverable

The final application should allow a user to:

1. Select a PV site.
2. Select a forecast horizon.
3. Select a model.
4. Generate a forecast.
5. View historical generation.
6. View predicted generation.
7. Compare actual vs predicted.
8. View uncertainty intervals.
9. Inspect model performance.
10. View feature importance.
11. Compare sites.
12. Download predictions.
13. View data-quality information.

The final system should demonstrate that the forecasting pipeline is:

**reproducible, leakage-safe, multi-site capable, explainable, measurable, and deployable.**

---

# 61. Final Technology Stack

## Machine Learning

* Python 3.11+
* NumPy
* Pandas
* SciPy
* Scikit-learn
* XGBoost
* PyTorch
* SHAP
* MLflow

## Data

* CSV/Parquet
* PyArrow

## Backend

* FastAPI
* Pydantic
* Uvicorn

## Frontend

* React
* TypeScript
* Vite
* Tailwind CSS
* shadcn/ui
* Recharts

## Infrastructure

* Docker
* Docker Compose
* Nginx

## Testing

* Pytest
* Playwright or equivalent frontend testing framework

---

# 62. Final Product Definition

The completed product is:

> **A web-based, multi-site solar photovoltaic power forecasting platform that uses the UNISOLAR dataset to compare statistical, machine-learning, and deep-learning forecasting approaches across multiple time horizons, with leakage-safe temporal validation, unseen-site generalization testing, explainable AI, uncertainty estimation, REST APIs, and an interactive analytics dashboard.**

The implementation must prioritize **correctness and reproducibility over model complexity**. A reliable XGBoost model with rigorous evaluation is preferable to an unvalidated Transformer.
