---
name: project-state
description: Current phase progress of UNISOLAR solar forecasting build in E:\Solar_gemini — read first every session
metadata:
  type: project
---

Project: UNISOLAR solar power prediction platform (`E:\Solar_gemini`, PRD-driven, 16 phases).

Authoritative trackers (repo files, always win over this snapshot):
- `TASKS.md` — per-task checkboxes
- `DECISIONS.md` — D-001…D-022 architecture/data decisions
- `RESULTS.md` — measured results only (no fabrication allowed, PRD Rule 4)

**Snapshot as of 2026-09-01 (post Phase 16):**
- Phase 0 (repo init) ✅ — conda env `solar`, Python 3.13, torch 2.13.0+cu132 GPU verified
- Phase 1 (inspection) ✅ — schema verified, see D-006; key facts: 42 sites / 5 campuses, 15-min data
  2020-01→2022-04, target SolarGeneration 56.2% missing, weather at CAMPUS grain, NO irradiance columns,
  no license file found (user still to confirm provenance)
- Phase 2 (data engineering) ✅ — processed parquet at `data/processed/solar/`
  (2,731,946 rows × 14 cols, partitions site_id/year/month), tz=Australia/Melbourne (D-007),
  power never imputed (D-008), 0 outliers (D-009), tests green
- Phase 3 (EDA) ✅ — `artifacts/eda/` 9 figures + `eda_summary.md` via `scripts/run_eda.py`;
  key findings: within-site correlations ≫ pooled (elevation r=0.74 vs 0.34) → D-010 per-site
  normalization + within-site metrics; night nonzero share ~0% validates D-007; missingness structural;
  weather >80% missing for 2021-08→2022-04
- Phase 4 (baselines) ✅ — `artifacts/baselines/` via `scripts/run_baselines.py`;
  persistence primary: test MAE 2.783, R² 0.699 (beats mean_site 4.896 / mean_global 6.502 / zero 7.499).
  Protocol D-011: per-site chronological 70/15/15 in `src/data/splits.py` (reused by Phases 5–8,
  extended Phase 9); stat baselines fit train-only; persistence lag table = full series but lookups
  strictly causal (t−24h < t, unit-tested); nRMSE denom = train observed range 99.12 kWh.
  Gotchas: parquet loads site_id as Categorical (baselines normalize); pd.Timedelta keyword
  constructors (days=) trip numpy 2.5 deprecation — use positional + unit="s".
- Phase 5 (features) ✅ — `src/features/{temporal,lag,rolling,weather,solar}.py` +
  `scripts/build_features.py` → `data/processed/features/` (2.73M × 51, 37 engineered) +
  `artifacts/features/feature_metadata.json`. Conventions D-012: calendar-exact lags (NOT positional
  shift), rolling closed-left [t−W,t) min_periods=1, S-hemisphere seasons, wind→sin/cos,
  solar recomputed via pvlib per campus. Verified exact on 1.1M+ lag pairs; 55 tests green.
  pytest filterwarnings suppresses pvlib/pandas numpy-timedelta deprecation (pyproject.toml).
- Phase 6 (XGBoost) ✅ — `src/models/xgboost_model.py` + `src/config.py` +
  `configs/{models,training}.yaml` + `scripts/train_xgboost.py`. Run
  `xgboost-site-all-h1-v1`: single-step (1×15-min), early stop iter 109/2000,
  10.6 s CPU. Test ALL: MAE 1.056 / RMSE 2.530 / R² 0.951 (persistence 2.783/6.294/0.699).
  Artifacts: `models/xgboost_site_all_h1_v1.json`, `artifacts/xgboost/*` (metrics val+test ×
  ALL+42 sites, importance top-2 = rolling_mean_1h 44.5% + lag_1 41.6%, predictions parquet),
  MLflow `mlruns/`. Conventions D-013: CPU hist, site_id native categorical, NaN features native,
  config-driven hyperparams, `MLFLOW_ALLOW_FILE_STORE=true` needed for mlruns file backend.
  69 tests green.
- Phase 7 (LSTM/GRU) ✅ — `src/models/sequence_model.py` + `scripts/train_sequence.py`.
  Runs `lstm-site-all-h1-v1` / `gru-site-all-h1-v1`: 24 h lookback × 13 channels
  (incl. `power_observed` mask), single-step, same D-011 split/denominators.
  Test ALL: LSTM MAE 1.140/RMSE 2.603/R² 0.948; GRU 1.147/2.551/0.950;
  XGBoost still best MAE 1.056/2.530/0.951; persistence 2.783/6.294/0.699.
  **Leak incident**: first run had power(t) in the input window → R²=1.000;
  purged artifacts/mlrun, added `_mask_current_step` guard (zeroes power+mask
  at step t) + unit tests; details in D-014. Artifacts:
  `models/{lstm,gru}_site_all_h1_v1.pt`, `artifacts/{lstm,gru}/*`, MLflow.
  Gotchas: torch scripts exit code **9 after success** on this box (teardown;
  check artifacts not exit codes); gather must copy before in-place masking
  (strided view aliases dataset matrix). Suite total 86 green.
- Phase 8 (Transformer) ✅ — `TransformerForecaster` in `sequence_model.py`
  (same windows/guard/loop, pre-LN encoder, sinusoidal PE buffer, no causal
  mask, last-position readout; D-015). Run `transformer-site-all-h1-v1`:
  test ALL MAE 1.124 / RMSE 2.544 / R² 0.950 — best deep model on MAE, no
  negative-R² sites (min 0.762); XGBoost still best overall (1.056/2.530/
  0.951 at 10.6 s CPU). Comparison deliverable:
  `scripts/compare_models.py` → `artifacts/evaluation/*` (8 models).
  Suite total 93 green.
- Phase 9 (Cross-site) ✅ — `cross_site_split` in `src/data/splits.py` (D-016:
  42 sites → 30/6/6, seeded permutation, held-out = FULL history, seen =
  D-011 chrono split of train sites; pooled nRMSE denom only). Run via
  `scripts/run_cross_site.py` → `artifacts/cross_site/{cross_site_metrics.csv
  (380 rows), cross_site_report.md, run_metadata.json}`. Headline test ALL:
  transformer best unseen MAE 0.764 / R² 0.897 (no neg-R² site); XGBoost
  generalizes worst (R² 0.949→0.795, unseen site 29 R² −6.859, unknown
  site_id category); persistence weakest everywhere. **Negative deep-model
  gaps are a plant-size artifact**: all 6 held-out test sites small (1.3–7.6
  kWh mean daylight output) vs training avg 10.0 with giants 35–44 — compare
  within protocol, not across. Leakage tests `tests/test_leakage.py` (9,
  PRD §46 mapping; feature-selection/tuning stubbed until those stages).
- Phase 10 (Explainability) ✅ — `src/explain/shap_explain.py` +
  `scripts/run_shap.py` + `tests/test_shap.py` (D-017: SHAP on XGBoost only,
  exact TreeExplainer; sequence models excluded — no exact fast path).
  Artifacts `artifacts/shap/` (global CSV, bar/beeswarm/dependence/waterfall
  PNGs, local contributions CSV). Global: lag_1 34.9% + roll_mean_1h 23.1%
  ≈ 64% of attribution; weather minor. Local scenarios incl. night_zero
  pred 13.47 vs obs 0.12 → **quantified night failure mode**: night MAE
  0.157 (lag_1 present, n=248) vs 5.421 (lag_1 missing, n=231) — xgboost
  NaN default branches learned from daytime rows; sequence models immune
  (mask channel). Additivity err 6.9e-05. Suite total 117 green.
  Gotcha: shap.Explanation needs base_values as ARRAY not scalar;
  xgboost validates categorical dtype positionally inside shap's internal
  predict → always rebuild inputs via prepare_matrix with stored layout.
- Phase 11 (Uncertainty) ✅ — split conformal (D-018): `src/models/conformal.py`
  + `scripts/run_conformal.py` + `tests/test_conformal.py` (10 tests).
  Calibrated on VAL observed rows, evaluated on TEST; xgboost + persistence;
  global + Mondrian regimes {day,night}×{lag_present,lag_missing}.
  Coverage ≥ nominal everywhere (xgb 0.916 @0.9, ±2.92; persistence ±8.20,
  2.8× wider). Mondrian: night_lag width 0.88 vs night_nolag 26.8 (prices
  the Phase 10 failure mode). Caveats: giant sites 11/25/27 undercover
  (0.54–0.59; absolute radius sized by small-plant bulk — normalized
  conformal = future work); unclipped night_nolag bounds negative.
  Artifacts `artifacts/uncertainty/`. Gotcha: persistence must fit on FULL
  table (causal t−24h lookups) or eval rows go NaN-starved (D-011 #3).
  Suite total 127 green.
- Phase 12 (Backend) ✅ — `src/api/{app,store,forecast}.py` + `scripts/run_api.py`
  (D-019): /api/v1 health/dataset/sites/models-metrics/history/forecast(+batch).
  Multi-step by recursion (predict→append→rebuild Phase 5 features), weather
  carried forward, horizon ≤96 (24 h), batch ≤10; served = xgboost (+Mondrian
  conformal bounds 0.9) + persistence; lstm/grp/transformer registered-only →
  409. Store reads partition-filtered parquet only; MemStore twin for tests
  (`tests/test_api.py`, 17). Smoke vs real artifacts: all routes pass, metrics
  endpoint reproduces recorded MAEs verbatim. Fixed latent flake in Phase 7
  train-loop test: LSTM init drew from unseeded global RNG (entropy per
  process) — reseed before model construction (convention already documented
  by sibling determinism test); margin now rmse 2.18 vs spread 3.10.
  Suite total 144 green.
- Phase 13 (Frontend) ✅ — `frontend/` React 19 + TS + Vite(rolldown) +
  Tailwind 4 + shadcn/ui + Recharts. Six PRD §38 pages; live via Vite proxy
  → FastAPI, SHAP/quality/site aggregates as bundled JSONs (`frontend/src/
  data/`, refresh `conda run -n solar python scripts/export_frontend_data.py`,
  D-020). Build green 583 ms, oxlint 0 errors. **Key bug (D-021)**: API
  serializes `is_daylight` as JSON boolean; frontend tested `=== 1` → night
  shading silently never rendered — verify raw JSON bytes, not Python-
  coerced comparisons. Visual verification: Playwright shots all 6 pages
  light+dark+mobile; night shading pixel-verified at opacity 0.18; bar
  axes start at zero; midnight ticks carry date.
- Phase 14 (Integration) ✅ — deep models served (D-022): checkpoints were
  weights-only, so `scripts/export_sequence_scalers.py` re-derived train-split
  y/channel stats → `artifacts/{arch}/serving_scalers.json`, validated bit-exact
  (max|Δ|=0 vs stored test preds on CUDA; CPU ≈8.5e-03 = float noise).
  `recursive_forecast_sequence` reuses Phase 7 window contract (97-row window,
  exported channel stats, `_mask_current_step`, pred feeds back mask=1); no
  conformal bounds for deep models (Phase 11 covered xgboost+persistence only).
  Serving CPU; transformer 96 steps ≈3.7 s. Registry 5/5 served; suite 145 green.
  Frontend auto-picks-up via /models; interval copy + ghost-legend fixed.
- Post-Phase-14 latency fix (D-023, 2026-08-25): recursive serving rebuilt
  features 96×/request (2.4–3.0 s). Now `_static_frame` computes timestamp-only
  features once, `_PowerHistory` does incremental lag/rolling per step
  (semantics pinned vs Phase 5 batch by `TestIncrementalFeatures`); frontend
  caches forecasts per (site,model,horizon) + 300 ms slider debounce.
  Measured: xgboost 2.99→0.57 s, lstm/transformer ~2.4→0.22 s; suite 147 green.
- Post-PRD enhancement shipped (D-024/D-025, 2026-08-26): Train page hardened
  + first live end-to-end run (xgboost full 56.2 s, test MAE 1.056 kW —
  `artifacts/train_live_run/summary.json`; marker-based job success, D-024);
  four new export bundles + six charts on Dashboard/Quality/Comparison
  (D-025). Suite 147 → 170 green.
- Phase 15 (Testing) ✅ — unit/API/leakage suites per PRD §45; all tests green
  (170 passed). Coverage gaps audited, no missing PRD §45 cases found.
- Phase 16 (Docker) ✅ — docker-compose.yml: frontend, backend, ml service (+ Postgres if required)

**Next: — ** (All phases complete)

Workflow rules from user: [[phase-gate-confirmations]], keep [[env-conda-run-newline]] in mind when running snippets.