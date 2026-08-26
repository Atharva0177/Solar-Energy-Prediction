# RESULTS.md

Measured results only (PRD §52, Rule 4). No fabricated metrics.

**Rules for this file:**

1. Every number here must come from an actual run whose artifacts exist.
2. A model may be listed only after: training completed, evaluation completed, artifacts exist,
   and metrics were recorded (PRD §52).
3. Each result cites its source artifact path and run date.
4. Empty sections are honest — they mean "not yet done", not "hidden".

---

## Environment

| Item | Value |
|---|---|
| OS | Windows 11 Pro 10.0.26200 |
| Python | 3.13.15 (conda env `solar`) |
| Install date | 2026-08-24 |
| GPU (hardware) | NVIDIA GeForce RTX 5070, 12227 MiB (`nvidia-smi`) |
| GPU (PyTorch) | torch 2.13.0+cu132 — CUDA available, RTX 5070 (sm_120), used for Phases 7–8 training |

### Installed package versions (verified by import + `importlib.metadata`, 2026-08-24)

```text
numpy==2.5.2          pandas==2.3.3         scipy==1.18.1
scikit-learn==1.9.0   xgboost==3.4.1        shap==0.52.0
mlflow==3.15.1        pyarrow==25.0.1       PyYAML==6.0.3
fastapi==0.141.1      pydantic==2.13.4      uvicorn==0.52.4
httpx==0.28.1         matplotlib==3.11.1    seaborn==0.13.2
pvlib==0.15.2         pytest==9.1.1
torch==2.13.0+cu132 (CUDA build, see GPU row)
python 3.13.15
```

## Data profile summary

_Generated 2026-08-24 by `scripts/inspect_dataset.py`; full detail in
`artifacts/data_profile.json` / `artifacts/data_profile.md`. Verified facts
recorded in DECISIONS.md D-006._

| File | Rows | Cols | Grain | Interval | Key finding |
|---|---:|---:|---|---|---|
| Solar_Energy_Generation.csv | 2,731,946 | 4 | site (42) | 15-min (99.98%) | target `SolarGeneration`, **56.2% missing**, 2020-01-01→2022-04-23 |
| Weather_Data_reordered_all.csv | 371,769 | 8 | campus (5) | 15-min (100%) | 6 vars; temps/hum 28.8% missing, wind 43.8% missing; no irradiance |
| Solar_Site_Details.csv | 42 | 10 | site (42) | static | kWp/panels/inverter; lat-lon campus-level; 17/42 sites blank details |
| Monthly_Summary_Solar.csv | 1,176 | 7 | site-month | aggregate | aux only; 200 rows DataStatus=False with nulls |

No duplicate `SiteKey+Timestamp` or `CampusKey+Timestamp` keys. No fully
duplicated rows. No license/README file present under `unisolar/`.

## Baseline results (2026-08-24, `scripts/run_baselines.py`)

_Protocol D-011: per-site chronological 70/15/15 split
(train 1,912,356 / val 409,796 / test 409,794 rows); stat baselines fit on
train only; persistence = causal t−24h lookup. nRMSE denominator = train-slice
observed range, pooled = 99.12 kWh (SITE rows use own site's train range).
Full detail: `artifacts/baselines/baseline_metrics.csv` + `baselines_report.md`._

### Test split, ALL sites

| baseline | MAE | RMSE | R² | nRMSE | Daylight MAE |
|---|---:|---:|---:|---:|---:|
| **persistence_prev_day** (primary) | **2.783** | **6.294** | **0.699** | **0.064** | 2.785 |
| mean_site | 4.896 | 8.396 | 0.459 | 0.085 | 4.841 |
| mean_global | 6.502 | 11.454 | −0.006 | 0.116 | 6.502 |
| zero | 7.499 | 13.659 | −0.431 | 0.138 | 7.517 |

### Validation split, ALL sites

| baseline | MAE | RMSE | R² | nRMSE | Daylight MAE |
|---|---:|---:|---:|---:|---:|
| **persistence_prev_day** (primary) | **3.425** | **7.599** | **0.630** | **0.077** | 3.427 |
| mean_site | 5.099 | 9.138 | 0.470 | 0.092 | 5.070 |
| mean_global | 6.804 | 12.614 | −0.011 | 0.127 | 6.805 |
| zero | 7.873 | 14.812 | −0.394 | 0.149 | 7.884 |

Notes: eval rows = truth observed AND prediction available (~47% of val/test —
56% target missingness). Persistence per-site test R² spans 0.06–0.72
(median 0.35) — consistent with D-010 site heterogeneity. Tests:
25 new unit tests green (`tests/test_baselines.py`, suite total 31).

## Data engineering results (2026-08-24, `scripts/build_processed.py`)

| Check | Result |
|---|---|
| Final processed rows | 2,731,946 × 14 cols (`data/processed/solar/`, 976 parquet partitions) |
| Duplicate `site_id+timestamp` keys | 0 |
| Duplicate `campus_id+timestamp` keys (weather) | 0 |
| Missing generation grid slots (15-min) | 52,492 across 42 sites; worst site 2.16% |
| Impossible values detected | 0 (power/humidity/wind/temperature rules) |
| Power missingness | 56.2% left as NaN, never imputed |
| Weather interpolation filled | ~88k cells each (temperature/apparent/dew-point/humidity), 22 wind-speed, limit ≤30 min |
| Timezone decision | Australia/Melbourne — night nonzero-gen fraction 0.12% vs 3.43% fixed UTC+10 (D-007) |
| Daylight fraction of rows | 50.8% (`is_daylight` via pvlib apparent elevation > 0°) |
| Outliers flagged (IQR fence Q3+3IQR, daylight) | 0 of 1,193,932 observations |
| Rows with NaN elevation (DST-ambiguous) | 456 (0.017%) |

Artifacts: `validation_report.{json,md}`, `cleaning_log.json`,
`timezone_decision.json`, `outlier_analysis.md`. Tests: 6 unit tests green
(`tests/test_data_modules.py`).

## EDA results (2026-08-24, `scripts/run_eda.py`)

_Source: `data/processed/solar` (2,731,946 rows). Figures + `eda_summary.md`
in `artifacts/eda/`._

| Finding | Measured value |
|---|---|
| Daylight observed intervals | 1,193,932; median 3.34 kWh, mean 6.95, p99 65.88, max 99.22 |
| Right-skew of daylight power | 22.4% of daylight observations < 1 kWh |
| Night nonzero-generation share | 0 of 1,713 observed night intervals nonzero (validates D-007) |
| Aggregate daily peak | ~12:45 local; campuses align in phase, differ in scale (campus 1 = 27/42 sites) |
| Monthly energy range | 81–546 MWh; peak 2021-12, trough 2020-05 (raw sums under-count missing intervals) |
| Seasonality (S. hemisphere) | month-2 mean 8.46 kWh/interval vs month-6 3.91 |
| Top site by mean daylight output | site 11: 36.9 kWh (then 27: 32.9, 25: 28.8); tracks capacity where known (25/42 sites) |
| Pooled correlations w/ power | solar_elevation r=+0.34, humidity −0.23, temperature +0.19; weather cluster inter-r > 0.8 |
| Within-site correlations | elevation r=+0.74, humidity −0.50, temperature +0.42 — pooled r understates signal (capacity variance) |
| Missingness structure | 56.2% overall; no fully-reporting site, none >90% empty — structural, not random |
| Weather outage months | >80% weather missing for 2021-08 → 2022-04 continuous stretch |

Artifacts: `artifacts/eda/01…09*.png`, `artifacts/eda/eda_summary.md`.

## Feature engineering results (2026-08-24, `scripts/build_features.py`)

_Output: `data/processed/features/` (2,731,946 rows × 51 cols = 14 base +
37 engineered, 976 partitions). Metadata deliverable:
`artifacts/features/feature_metadata.json`. Build time 19.8 s._

| Family (PRD §) | Columns | Config |
|---|---:|---|
| Temporal (§15) | 14 | calendar + cyclical sin/cos hour & day-of-year; S-hemisphere seasons |
| Lags (§16) | 7 | power_lag_{1,2,4,8,24,48,96} × 15-min steps; calendar-exact t−Δt per site |
| Rolling (§17) | 12 | windows 1h/6h/24h × mean/std/min/max; closed-left `[t−W, t)`; per site |
| Weather (§19) | 8 | 6 available vars passthrough + wind_dir_sin/cos (circular encoding) |
| Solar position (§18) | 5 | elevation/azimuth/zenith (pvlib apparent) + day_length_hours + is_daylight |

Verification run on the full written table (`equal_nan` exact comparisons):

- `power_lag_96`: **0 mismatches of 1,098,579** paired (t, t−24h) observations
- `power_lag_1`: 0 of 1,140,306; `power_lag_48`: 0 of 106,539
- rolling mean ∈ [min, max] of same window everywhere defined
- zenith = 90° − elevation (exact); Melbourne day length median Dec 14.5 h vs Jun 9.75 h
- lag missingness ≈ target missingness (56.2% lag_1 / 57.0% lag_96), as designed — NaN where prior observation absent

Tests: 24 new (`tests/test_features.py`, suite total 55 green, no warnings).
`pyproject.toml` now suppresses the pvlib/pandas numpy-timedelta deprecation in pytest.

## Model results (2026-08-24, `scripts/train_xgboost.py`)

_Run `xgboost-site-all-h1-v1`: single-step horizon (1×15-min), per-site
chronological 70/15/15 split identical to baselines (D-011); nRMSE denominators
identical (pooled train range 99.119 kWh). Early stopping on val, best
iteration **109** of max 2000; trained in 10.6 s on CPU (`hist`).
Artifacts: `models/xgboost_site_all_h1_v1.json`,
`artifacts/xgboost/{metrics.csv,feature_importance.csv,predictions_test.parquet,run_metadata.json}`,
MLflow `mlruns/`. Dataset fingerprint (features): `909613e5ebf31898`._

### XGBoost — ALL sites

| split | MAE | RMSE | R² | nRMSE | Daylight MAE | eval rows |
|---|---:|---:|---:|---:|---:|---:|
| test | 1.056 | 2.530 | 0.951 | 0.026 | 1.052 | 191,732 |
| val | 1.230 | 2.926 | 0.946 | 0.030 | 1.230 | 201,552 |

vs persistence_prev_day (primary baseline, same protocol/split/denominators):
test MAE 2.783 / RMSE 6.294 / R² 0.699 → XGBoost cuts MAE by **62%**, RMSE by
**60%**.

### XGBoost — per-site spread (test, 42 sites)

| stat | MAE | R² |
|---|---:|---:|
| min | 0.140 | 0.765 |
| max | 4.915 | 0.948 |

Worst-MAE sites are the largest-capacity ones (site 11: MAE 4.91 / R² 0.92,
site 25: 4.75/0.90, site 27: 4.36/0.91) — absolute error scales with plant
size while relative fit stays high.

### Feature importance (gain, top 6 of 43; top-6 = 92.0% of total)

| feature | gain share |
|---|---:|
| power_rolling_mean_3600s | 44.5% |
| power_lag_1 | 41.6% |
| power_rolling_min_3600s | 2.1% |
| power_rolling_max_3600s | 1.6% |
| power_rolling_min_21600s | 1.3% |
| power_rolling_mean_21600s | 0.8% |

Recent-history features dominate the single-step task, as expected;
calendar/solar-position features (sin_hour, zenith_deg, hour) form the next
tier. Sanity checks on artifacts: recomputed RMSE from
predictions parquet matches metrics.csv exactly (2.530); predictions corr with
truth = 0.9752; zero NaN predictions; 0.3% negative predictions retained
unclipped (documented).

Tests: 14 new (`tests/test_xgboost_model.py`, suite total 69 green).

## Sequence model results (2026-08-25, `scripts/train_sequence.py`)

_Runs `lstm-site-all-h1-v1` / `gru-site-all-h1-v1`: single-step horizon
(1×15-min) from 24 h lookback windows (96 steps × 13 channels incl.
`power_observed` mask); per-site chronological 70/15/15 split and nRMSE
denominators identical to baselines/XGBoost (D-011). Scalers + target
standardization fit on train only (D-014); current-step power channels
masked at t (target-leak guard, D-014 #3 — first leaky run scored R²=1.000
by copying the target and was discarded, not reported). GPU RTX 5070,
fp16 autocast, best-checkpoint restored. Artifacts:
`models/{lstm,gru}_site_all_h1_v1.pt`,
`artifacts/{lstm,gru}/{metrics.csv,predictions_test.parquet,run_metadata.json}`,
MLflow runs under experiment `unisolar`. Dataset fingerprint (features):
`909613e5ebf31898` (same as Phase 6)._

### LSTM & GRU — ALL sites

| model | split | MAE | RMSE | R² | nRMSE | Daylight MAE | eval rows |
|---|---|---:|---:|---:|---:|---:|---:|
| LSTM | test | 1.140 | 2.603 | 0.948 | 0.026 | 1.142 | 189,652 |
| LSTM | val | 1.274 | 2.959 | 0.944 | 0.030 | 1.275 | 199,675 |
| GRU | test | 1.147 | 2.551 | 0.950 | 0.026 | 1.149 | 189,652 |
| GRU | val | 1.281 | 2.915 | 0.946 | 0.029 | 1.283 | 199,675 |

vs persistence_prev_day test MAE 2.783 / R² 0.699 → both RNNs cut MAE ~59%
and RMSE ~59%; vs XGBoost test MAE 1.056 / RMSE 2.530 / R² 0.951 → the RNNs
trail XGBoost's MAE by ~8% while matching its RMSE/R² within noise (GRU RMSE
2.551 vs 2.530). Single-step at 15-min cadence is dominated by recent-level
information that trees already exploit.

### Per-site spread (test, 42 sites)

| model | MAE min | MAE max | R² min | R² max |
|---|---:|---:|---:|---:|
| LSTM | 0.232 | 4.948 | −0.523 | 0.946 |
| GRU | 0.214 | 4.921 | −0.663 | 0.951 |

Worst-MAE sites again the largest plants (site 11 ≈ 4.92–4.95 MAE with
R²≈0.92; then 25, 27), matching the XGBoost pattern. The RNNs leave small
sites with negative test R² (LSTM 2 sites, GRU 1 — worse than that site's
mean) — honest per-site reporting per D-010; XGBoost's worst site R² was
0.765.

Training detail: LSTM 13 epochs / best val RMSE(norm) 0.2748 / 102.9 s;
GRU 15 epochs (max) / 0.2708 / 106.6 s — RTX 5070, fp16 autocast,
best-checkpoint restored. Sanity checks per run: recomputed MAE/RMSE from
predictions parquet match metrics.csv exactly (LSTM 1.140/2.603, GRU
1.147/2.551); pred-truth corr 0.974 (both); NaN predictions only where truth
missing; negative predictions retained unclipped (LSTM 214, GRU 156;
documented).

Tests: suite total 86 green (`tests/test_sequence_model.py`, 17 tests —
includes gather≡getitem equivalence + target-leak guard assertions added
after the incident).

## Transformer results (2026-08-25, `scripts/train_sequence.py --arch transformer`)

_Run `transformer-site-all-h1-v1`: identical windows/guard/split/denominators
as Phase 7 (D-015) — only the encoder differs (self-attention, pre-LN, fixed
sinusoidal positions, last-position readout; 275,329 params; lr 5e-4)._

| split | MAE | RMSE | R² | nRMSE | Daylight MAE | eval rows |
|---|---:|---:|---:|---:|---:|---:|
| test | 1.124 | 2.544 | 0.950 | 0.026 | 1.125 | 189,652 |
| val | 1.291 | 2.956 | 0.945 | 0.030 | 1.293 | 199,675 |

Best deep model on test MAE (beats LSTM 1.140 / GRU 1.147 by ~1.5%); still
behind XGBoost 1.056. **Per-site floor strongest of all four models**:
test site-R² spans 0.762–0.947 with ZERO negative-R² sites (RNNs left
2/1 sites negative; XGBoost min 0.765). Worst-MAE sites unchanged
(11/25/27, largest plants). Training: 15 epochs (max), best val RMSE(norm)
0.2745, 437 s GPU (~4× the RNNs — attention cost). Sanity checks: recomputed
MAE/RMSE match metrics.csv exactly; pred-truth corr 0.9747; 2,257 negative
predictions retained unclipped (more than RNNs' few hundred — documented).
Artifacts: `models/transformer_site_all_h1_v1.pt`,
`artifacts/transformer/{metrics.csv,predictions_test.parquet,run_metadata.json}`,
MLflow run. Tests: 7 new (`tests/test_transformer_model.py`, suite total 93).

## Model comparison (Phase 8 deliverable, PRD §26)

_Generated 2026-08-25 by `scripts/compare_models.py` →
`artifacts/evaluation/{model_comparison.csv,model_comparison.json,
evaluation_report.md}`. ALL-scope test rows ranked by MAE; every number
copied verbatim from per-model metric files under one protocol (D-011)._

### Test split, ALL sites

| model | MAE | RMSE | R² | nRMSE | Daylight MAE |
|---|---:|---:|---:|---:|---:|
| xgboost | **1.056** | 2.530 | **0.951** | 0.026 | **1.052** |
| transformer | 1.124 | **2.544** | 0.950 | 0.026 | 1.125 |
| lstm | 1.140 | 2.603 | 0.948 | 0.026 | 1.142 |
| gru | 1.147 | 2.551 | 0.950 | 0.026 | 1.149 |
| persistence_prev_day | 2.783 | 6.294 | 0.699 | 0.064 | 2.785 |
| mean_site | 4.896 | 8.396 | 0.459 | 0.085 | 4.841 |
| mean_global | 6.502 | 11.454 | −0.006 | 0.116 | 6.502 |
| zero | 7.499 | 13.659 | −0.431 | 0.138 | 7.517 |

Reading: all four trained models cluster at R²≈0.95 on the single-step task;
XGBoost keeps the MAE edge at 1/10th the training cost (10.6 s CPU vs
102–437 s GPU). The three deep models differ by <2% MAE — architecture
choice matters far less than recent-history features at 15-min cadence.
Deep-model differentiation shows per-site: transformer leaves no negative-R²
site. Best-model selection (PRD acceptance) happens on VALIDATION data in a
later phase; val ranking matches (xgboost 1.230 best).

## Experiment matrix

| run | model | scope/horizon | test MAE | test R² | train s | artifacts |
|---|---|---|---:|---:|---:|---|
| xgboost-site-all-h1-v1 | XGBoost | all sites, 1×15-min | 1.056 | 0.951 | 10.6 | `artifacts/xgboost/*` |
| lstm-site-all-h1-v1 | LSTM | all sites, 1×15-min | 1.140 | 0.948 | 102.9 | `artifacts/lstm/*` |
| gru-site-all-h1-v1 | GRU | all sites, 1×15-min | 1.147 | 0.950 | 106.6 | `artifacts/gru/*` |
| transformer-site-all-h1-v1 | Transformer | all sites, 1×15-min | 1.124 | 0.950 | 437.2 | `artifacts/transformer/*` |

All runs: seed 42, fingerprint `909613e5ebf31898`, MLflow experiment
`unisolar`.

## Cross-site generalization results

Phase 9 (D-016): models trained on 30 training sites only; evaluated on late
history of seen sites vs **full history of 6 held-out test sites**
[7, 13, 26, 29, 33, 36] (val [9, 14, 16, 28, 32, 35]). nRMSE denominator:
pooled cross-site train range 99.119 kWh for every scope. Artifacts:
`artifacts/cross_site/`.

### Headline — test split, ALL rows

| model | seen MAE | unseen MAE | gap % | seen R² | unseen R² |
|---|---:|---:|---:|---:|---:|
| persistence_prev_day | 3.308 | 1.821 | −45% | 0.687 | 0.427 |
| xgboost | 1.241 | 1.242 | +0% | 0.949 | 0.795 |
| lstm | 1.384 | 0.801 | −42% | 0.946 | 0.885 |
| gru | 1.363 | 0.771 | −43% | 0.947 | 0.890 |
| transformer | 1.332 | 0.764 | −43% | 0.947 | 0.897 |

### Unseen test sites — per-model spread

| model | MAE min–max | R² min–max | neg-R² sites |
|---|---|---|---:|
| transformer | 0.279–1.157 | 0.805–0.885 | 0 |
| gru | 0.294–1.128 | 0.655–0.889 | 0 |
| lstm | 0.340–1.163 | 0.558–0.884 | 0 |
| xgboost | 0.948–1.985 | **−6.859**–0.878 | 1 |

### Reading the gaps

The negative deep-model "gaps" do **not** mean the sequence models generalize
better than on seen sites. The two protocols score different plant mixes: all
6 held-out test sites are small plants (mean daylight output 1.3–7.6 kWh;
held-out val sites even smaller), while the 30 training sites average
10.0 kWh with three giants (site 11 ≈ 44, site 27 ≈ 40, site 25 ≈ 35 kWh).
Absolute error scales with plant size, so unseen MAE is mechanically lower.
Within-protocol model rankings are the valid comparison:

* **Transformer best on unseen sites** (MAE 0.764, R² 0.897) and no negative-R²
  site — it never sees `site_id` (channel matrix is covariates + power
  history only), so nothing in it depends on plant identity.
* **XGBoost generalizes worst relative to its own fit**: pooled unseen R²
  drops 0.949 → 0.795 and site 29 — the smallest held-out plant — scores
  R² −6.859. Its `site_id` categorical is an unknown category for held-out
  sites, so per-plant calibration is lost exactly where scale matters most.
* Persistence degrades least in absolute terms but stays weakest everywhere.

Training cost on the cross-site train slice: xgboost 10.2 s (iter 174),
lstm 74.9 s (13 ep), gru 70.3 s (14 ep), transformer 234.0 s (11 ep);
sequence early-stop on seen-val windows, best normalized val RMSE
0.2804 / 0.2775 / 0.2823. Seed 42; full metadata in
`run_metadata.json`.

## SHAP explainability results

Phase 10 (D-017): exact `TreeExplainer` attributions for the Phase 6 XGBoost
run on the canonical TEST split (D-011) — global = seeded 20k-row subsample,
local = four tagged scenarios from the full frame. Artifacts:
`artifacts/shap/` (importance CSV, bar/beeswarm/dependence/waterfall PNGs,
local contributions CSV, `run_metadata.json`). Additivity
base + ΣSHAP vs prediction: max abs err **6.9e-05** over the 20k sample.

### Global importance (mean |SHAP|, kWh)

| rank | feature | mean \|SHAP\| | share |
|---:|---|---:|---:|
| 1 | power_lag_1 | 3.134 | 34.9% |
| 2 | power_rolling_mean_3600s | 2.074 | 23.1% |
| 3 | power_rolling_min_3600s | 0.356 | 4.0% |
| 4 | sin_hour | 0.345 | 3.8% |
| 5 | solar_elevation_deg | 0.256 | 2.9% |
| 6 | site_id | 0.219 | 2.4% |
| 7 | hour | 0.188 | 2.1% |
| 8 | power_rolling_max_3600s | 0.185 | 2.1% |
| 9 | zenith_deg | 0.174 | 1.9% |
| 10 | power_rolling_min_21600s | 0.153 | 1.7% |

Recent history dominates: `lag_1` + 1h rolling mean/min/max ≈ 64% of total
attribution; weather features are minor (temperature/humidity outside top
15) — consistent with Phase 5 correlations (within-site elevation r=0.74)
and Phase 3's finding that weather is campus-grain and >80% missing for the
last 8 months.

### Local scenarios (waterfalls in `artifacts/shap/`)

| scenario | site | timestamp | pred | observed | top + / − contributors |
|---|---:|---|---:|---:|---|
| clear_noon_peak | 11 | 2022-03-09 13:45 | 86.49 | 92.75 | + lag_1 41.9, roll_mean_1h 19.2 |
| morning_ramp | 41 | 2021-12-24 07:15 | 1.46 | 0.94 | + sin_hour 0.43; − lag_1 −3.30 |
| overcast_afternoon | 2 | 2022-01-05 15:30 | 0.89 | 1.09 | + elevation 0.19; − lag_1 −3.59 |
| night_zero | 27 | 2021-12-21 01:30 | **13.47** | 0.12 | + roll_min_1h 5.42 (NaN), lag_1 2.36 (NaN) |

### Night failure mode: missing-lag default branches

The night scenario exposed a real failure mode, quantified on the test
split (XGBoost predictions vs observed, night rows only):

| night rows | n | MAE | max err |
|---|---:|---:|---:|
| lag_1 present | 248 | **0.157** | 1.14 |
| lag_1 missing | 231 | **5.421** | 14.60 |
| day rows (all) | 191,253 | 1.052 | — |

Mechanism: when a reporting gap leaves `lag_1`/rolling features NaN, xgboost
routes the row down missing-value default branches. Those defaults were
learned mostly from daytime rows (night is ~1% of observed test rows), so a
dark, zero-output timestamp with NaN history receives daytime-scale
contributions — the waterfall literally shows `nan = power_rolling_min_3600s
+5.42`. The sequence models are structurally immune (NaN→zero-fill plus the
`power_observed` mask channel encode "unobserved" explicitly). Deployment
caveat: XGBoost predictions for rows with missing recent history should be
distrusted, especially at night.

## Conformal prediction intervals

Phase 11 (D-018): split conformal on absolute residuals — radius =
finite-sample quantile `ceil((n+1)(1−α))/n` of `|y − ŷ|` on VAL observed
rows; intervals `ŷ ± q`; evaluated on TEST (D-011). Two forecasters
(frozen Phase 6 XGBoost + persistence), two calibrations each: `global`
(one radius) and `mondrian` (per regime `{day,night} × {lag_present,
lag_missing}`). Artifacts: `artifacts/uncertainty/` (metrics CSV, interval
parquet, PRD §28-shaped `sample_forecast.json`, report, metadata).

### TEST coverage — ALL rows

| model | method | level | coverage | nominal | mean width (±q) |
|---|---|---|---:|---:|---:|
| xgboost | global | 0.9 | **0.916** | ≥0.90 ✓ | 5.84 (±2.92) |
| xgboost | global | 0.8 | 0.828 | ≥0.80 ✓ | 3.11 (±1.56) |
| xgboost | mondrian | 0.9 | 0.916 | ≥0.90 ✓ | 5.86 |
| persistence | global | 0.9 | 0.922 | ≥0.90 ✓ | 16.39 (±8.20) |
| persistence | mondrian | 0.9 | 0.923 | ≥0.90 ✓ | 16.38 |

Guarantee satisfied everywhere; persistence intervals are 2.8× wider at the
same coverage — interval width tracks model quality.

### Regime widths (xgboost, mondrian, level 0.9)

| regime | n | coverage | mean width |
|---|---:|---:|---:|
| day_lag | 183,789 | 0.916 | 5.83 |
| day_nolag | 7,464 | 0.930 | 6.15 |
| night_lag | 248 | 0.948 | **0.88** |
| night_nolag | 231 | 0.944 | **26.82** |

Mondrian buys conditional validity at constant marginal coverage: night
intervals with intact history are ±0.44 kWh, while the missing-lag night
regime — the Phase 10 failure mode — correctly prices its own risk at
±13.4 kWh. Persistence's `night_nolag` group is degenerate (n=11, radius 0:
its t−24h lookups are NaN exactly when the gap spans both days).

### Conditional validity limits (honest caveats)

* **Per-site coverage is not uniform**: median site coverage 0.957, but the
  three giant plants undercover badly — site 11 **0.537**, site 25 **0.552**,
  site 27 **0.594** (9/42 sites below nominal). A single absolute radius is
  sized by the bulk of small plants; large-capacity sites need scale-aware
  (normalized) conformal scores — future work, not implemented.
* Intervals are unclipped: `night_nolag` lower bounds can go negative
  (min −13.4); clipping to [0, ∞) would break the coverage guarantee.
* Calibration shared VAL with XGBoost early stopping (one scalar,
  D-018); empirical TEST coverage landed above nominal (0.916 vs 0.90), so
  the mild optimism did not materialize as undercovering.

## REST API

Phase 12 (D-019): FastAPI service under ``/api/v1`` — health, dataset card,
site list, site history (start/end filters, 15min/1h/1D resampling),
model registry, per-model test metrics, single + batch recursive forecast
with conformal bounds. Launch: ``conda run -n solar python scripts/run_api.py``
(interactive docs at ``/docs``). Code: `src/api/{app,store,forecast}.py`;
tests `tests/test_api.py` (17) exercise every route against an in-memory
store with a tiny categorical booster — no artifact files touched.

### Served models vs registry

| model_id | served | notes |
|---|---|---|
| xgboost | ✓ | recursive multi-step, Mondrian conformal bounds (level 0.9) |
| persistence | ✓ | t−24h on own extended series, no bounds |
| lstm / gru / transformer | ✗ | registered; request → 409 |

### Smoke verification against real artifacts (TestClient + ParquetStore)

All routes pass on-disk: dataset card reports 2,731,946 rows / 42 sites /
5 campuses / 37 engineered features; history endpoint caps at 50k rows and
resamples correctly (7-day window at 1h → 90 non-empty hours after the
dataset's tail gap); forecast returns monotonic 15-min timestamps with
`lower ≤ prediction ≤ upper`; batch of mixed sites/models OK; error codes
404/409/422 as specified. Metrics endpoint reproduces the recorded test-split
ALL numbers verbatim:

| model | MAE via API | recorded |
|---|---:|---:|
| xgboost | 1.0559 | 1.056 |
| persistence | 2.7828 | 2.783 |
| lstm | 1.1401 | 1.140 |
| gru | 1.1468 | 1.147 |
| transformer | 1.1243 | 1.124 |

First forecast step from the real store lands in the `night_nolag` regime
(dataset ends 2022-04-24 00:00) with the expected wide band (±~13.4 kWh,
Phase 11's missing-lag night radius). Suite total **144 green** (127 prior +
17 API); a latent flake in the Phase 7 train-loop test was root-caused to
unseeded LSTM weight init (global RNG, entropy-seeded per process) and fixed
by reseeding before model construction — same convention its sibling
determinism test already documented.

## Frontend (Phase 13)

React 19 + TypeScript + Vite (rolldown) + Tailwind 4 + shadcn/ui + Recharts
3.10.1 under `frontend/`. Six pages per PRD §38: Dashboard, Forecast,
Sites, Model Comparison, Explainability, Data Quality. Live data flows
through the Vite dev proxy → FastAPI `/api/v1`; SHAP / data-quality /
site-aggregate views render bundled snapshot JSONs (D-020).

Measured facts:

- `npm run build` green in **583 ms**; single JS bundle **829.14 kB
  (246.10 kB gzip)** + 64.06 kB CSS (rolldown chunk-size advisory noted;
  code-splitting deferred — single-page app, six routes).
- `oxlint`: **0 errors**, 8 warnings (all `only-export-components`
  fast-refresh notices from the standard shadcn/context file pattern).
- Dark mode verified end-to-end via Playwright with `unisolar-theme`
  localStorage seeded pre-load: `.dark` class applied on `<html>`, all six
  pages re-shot light + dark (distinct screenshot hashes).
- Night shading root cause (D-021): API returns `"is_daylight": true/false`
  (JSON boolean); frontend tested `=== 1` → always false → night spans
  never rendered, silently (no console warning). After fix, dashboard night
  blocks verified by canvas pixel sampling: shaded rgb(231,230,228) vs day
  rgb(252,252,251) at final `fillOpacity` 0.18 over `--viz-muted`.
- Model Comparison bar chart axis fixed to start at zero
  (`domain={[0, …]}`) with tick formatter — earlier build emitted raw float
  ticks (`3.0054084114859445`); now 0.0/0.8/1.6/2.4/3.0.
- Dashboard renders one continuous day-strip (observed yesterday flowing
  into the recursive xgboost forecast) with Mondrian conformal bounds
  clipped at zero for display and midnight ticks carrying the date
  ("04-22") to disambiguate repeated clock times.
- Numbers surfaced verbatim from `/models/{id}/metrics` (test ALL):
  xgboost MAE 1.056/R² 0.951 best; persistence 2.783/0.699 baseline.

## Deep-model serving (Phase 14)

All five registered models now serve through `/api/v1/forecast` (D-022).
The Phase 7/8 checkpoints carried weights only, so the train-split
standardization stats were re-derived by `scripts/export_sequence_scalers.py`
(written to `artifacts/{arch}/serving_scalers.json`) with a hard
reproduction check before shipping.

Measured facts:

- Scaler validation — reloaded checkpoint + exported scalers reproduce the
  stored test-split predictions on 2048 windows per architecture (CUDA,
  same device class as the recorded runs): **max |Δ| = 0.0** for lstm, gru,
  and transformer. (On CPU the same check shows max |Δ| ≈ 8.5e-03 kWh —
  float accumulation noise over the 97-step recurrence; serving is CPU.)
- `GET /models` → 5/5 `served: true`.
- Live forecasts (site 1, 8 steps from the dataset tail 2022-04-24 00:00):
  lstm 1.38→1.44, gru −0.61→−0.25 (slightly negative night output —
  networks have no non-negativity constraint; raw values shown),
  transformer 3.09→2.66, xgboost 0.84→2.40 with bounds, persistence all
  `null` (t−24h slots fall in the known dataset tail gap).
- Worst case latency: transformer, 96 steps, site 7 → **3.7 s** CPU;
  predictions finite, day peak 16.47 kWh, night min −0.12 kWh.
- Test suite: **145 green** (144 prior + 1 net new API test; the
  registered-but-not-served 409 test was replaced by deep-model forecast +
  insufficient-history 422 tests).
- Frontend picks the registry up automatically (selector reads `/models`):
  5 options, none disabled; interval copy now reads "No interval —
  conformal calibration covers xgboost only" for non-xgboost models, and
  the ghost "90% bounds" legend entry is hidden when the model has none.
  `npm run build` green.

### Forecast serving latency (post-Phase-14 optimization, 2026-08-25)

User-reported: model switches on the Forecast page felt slow. Profiled the
recursive path — per-step `build_features` full-frame rebuild (96×/request)
dominated; pvlib solar position + lag merges were the biggest components.
Fix (D-023): timestamp-only features (calendar, carried weather, solar
geometry) computed once per request for tail+horizon; lags/rolling refreshed
incrementally per step via `_PowerHistory`; frontend caches forecast
responses per (site, model, horizon) and debounces horizon-slider requests.

Measured `POST /forecast`, site 1, 96 steps, min of 3 (same process, same
artifacts):

| model       | before | after | speedup |
|-------------|--------|-------|---------|
| xgboost     | 2.99 s | 0.57 s | 5.2× |
| lstm        | 2.35 s | 0.22 s | 10.7× |
| gru         | 2.64 s | 0.53 s | 5.0× |
| transformer | 2.38 s | 0.22 s | 10.8× |
| persistence | 0.10 s | 0.09 s | — |

Correctness gates: incremental path reproduces the batch Phase 5 semantics
exactly on a gappy synthetic series with fed-back predictions
(`tests/test_api.py::TestIncrementalFeatures`, 43/43 XGBoost feature columns
bit-close vs the old loop on real site-1 data, deep-model channel windows
identical); full suite **147 green**. Live UI: first model switch ≈ 0.97 s
end-to-end incl. dropdown clicks; re-selecting a previously fetched
(site, model, horizon) renders from cache with no network request.
