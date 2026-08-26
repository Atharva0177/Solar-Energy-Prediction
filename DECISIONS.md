# DECISIONS.md

Architecture and implementation decision log (PRD Rule 10).
One entry per decision: context, choice, rationale, consequences.

---

## D-001 — Repository root location

- **Date:** 2026-08-24
- **Status:** Accepted
- **Context:** PRD §42 shows the project under a `solar-power-prediction/` folder. The actual
  working directory `E:\Solar_gemini` already contains `PRD.md` at its root alongside the
  `unisolar/` dataset directory.
- **Decision:** Treat `E:\Solar_gemini` as the repository root. Do not nest a second
  `solar-power-prediction/` directory.
- **Rationale:** PRD already lives here; nesting would break relative paths to the dataset
  and add no value.
- **Consequences:** All paths in configs/scripts are relative to this root.

## D-002 — Python environment: conda env "solar", Python 3.13

- **Date:** 2026-08-24
- **Status:** Accepted (revised same day — originally a `.venv`, switched to conda per user)
- **Context:** Machine has Python 3.13.11 (miniconda base, on PATH) and Python 3.14.2
  (`py` launcher). PRD §61 requires Python 3.11+. User directed use of conda.
- **Decision:** Use dedicated conda environment **`solar`** (Python 3.13.15,
  miniconda3/envs/solar). Cap `requires-python` at `<3.14`.
- **Rationale:** Python 3.14 is very new; scientific/ML wheels (PyTorch, SHAP, XGBoost) may
  lag on it. Conda env isolates this project's stack from the miniconda base.
- **Consequences:** All scripts run via `conda activate solar` or
  `conda run -n solar python ...`. No `.venv` in repo.

## D-003 — Dependency management: requirements.txt + pyproject.toml

- **Date:** 2026-08-24
- **Status:** Accepted
- **Context:** PRD §42 lists both `requirements.txt` and `pyproject.toml`. Reproducibility
  (PRD §44) requires recorded package versions.
- **Decision:** `requirements.txt` holds runtime dependencies with minimum bounds;
  `pyproject.toml` holds project metadata plus pytest/ruff tool config. Packages installed
  into the conda env via pip. After Phase 0 install completes, record exact resolved versions
  in RESULTS.md; consider a lock file (`pip freeze`) once the set stabilizes.
- **Note:** PyPI package for solar-position library is `pvlib` (the `pvlib-python` name no
  longer resolves on Python 3.13).
- **Consequences:** Fresh envs may resolve slightly newer versions until a freeze is adopted.

## D-004 — Dataset inventory (observed, not yet verified)

- **Date:** 2026-08-24
- **Status:** Provisional — pending Phase 1 inspection (PRD Rule 2)
- **Context:** `unisolar/` contains four CSVs observed by file size only:
  - `Solar_Energy_Generation.csv` (~84 MB) — presumed generation/target data
  - `Weather_Data_reordered_all.csv` (~23 MB) — presumed weather data
  - `Solar_Site_Details.csv` (~3 KB) — presumed site metadata (lat/lon etc.)
  - `Monthly_Summary_Solar.csv` (~63 KB) — presumed aggregate summary
- **Decision:** No schema assumptions made. Column names, sampling frequency, site count,
  and units are UNKNOWN until `scripts/inspect_dataset.py` runs (Phase 1). Downstream ML work
  is blocked on that profile.
- **Consequences:** Phase 1 must start with the inspection script per PRD §6.

## D-005 — PyTorch: CUDA build replaces default CPU wheel

- **Date:** 2026-08-24 (revised same day)
- **Status:** Accepted
- **Context:** Machine has an NVIDIA RTX 5070 (12 GB, Blackwell sm_120). The plain
  `pip install torch` on Windows yields a CPU-only build (`torch.cuda.is_available()` → False).
  PRD §49–50 require GPU usage for deep learning when a CUDA device is present.
  Originally planned `cu129` from the official PyTorch index; user instead installed the
  `cu132` build directly.
- **Decision:** Use torch **2.13.0+cu132** (user-installed). Verified:
  `torch.cuda.is_available()` → True, device "NVIDIA GeForce RTX 5070", capability (12, 0).
- **Rationale:** LSTM/GRU/Transformer training (Phases 7–8) needs GPU + mixed precision;
  CPU fallback remains supported for everything else per PRD §50.
- **Consequences:** Deep-learning phases run on GPU; preprocessing/XGBoost/API stay
  CPU-only by design. If driver or index changes, revisit.

## D-006 — Verified dataset schema (from `artifacts/data_profile.json`, 2026-08-24)

- **Date:** 2026-08-24
- **Status:** Accepted — verified by `scripts/inspect_dataset.py` run
  (artifacts exist); supersedes provisional D-004.
- **Context:** PRD §5.1 requires verifying file names/formats, records, sites,
  sampling interval, timestamp format, target/weather/irradiance columns,
  missing patterns, units, site IDs before any ML work (PRD Rule 2).
- **Decision / verified facts:**
  - **Solar_Energy_Generation.csv** — 2,731,946 rows × 4 cols:
    `CampusKey` (int64, 5 campuses), `SiteKey` (int64, **42 sites**),
    `Timestamp` (`YYYY-MM-DD HH:MM:SS`, 2020-01-01 00:15 → 2022-04-23 23:45),
    `SolarGeneration` (float64, **target**). Sampling: **15-min**, mode share
    99.98%, 0.02% irregular gaps. Duplicate `SiteKey+Timestamp` keys: **0**;
    fully duplicated rows: 0. `SolarGeneration` missing: **56.2%** of rows.
  - **Weather_Data_reordered_all.csv** — 371,769 rows × 8 cols:
    `CampusKey`, `Timestamp` (same range), and 6 weather variables
    (`ApparentTemperature`, `AirTemperature`, `DewPointTemperature`,
    `RelativeHumidity` — each 28.8% missing; `WindSpeed`, `WindDirection`
    — each 43.8% missing). Sampling: exactly 15-min per campus (100%
    regular). Keyed **per campus (5), not per site (42)** — join to
    generation via `CampusKey` + `Timestamp`. No duplicate keys.
  - **Solar_Site_Details.csv** — 42 rows × 10 cols: static metadata
    (`kWp`, `Number of panels`, `Panel`, `Inverter`, `Optimizers`,
    `Metric`=kWh, `lat`, `Lon`). 17 of 42 sites have no install details
    (blank). **lat/Lon are campus-level** (only 5 unique coordinate pairs).
  - **Monthly_Summary_Solar.csv** — 1,176 rows × 7 cols: monthly aggregate
    per site (`Year`, `Month`, `DataStatus` bool; 200 rows flagged False
    with nulls). Auxiliary only — not used for modeling.
  - **Irradiance columns: NONE present** in any file. Night/day separation
    (PRD §10) must derive from solar elevation using campus lat/lon (pvlib).
  - **Units:** `Metric`=kWh indicates generation in kWh per 15-min interval
    (inferred; no other unit metadata). Weather values consistent with °C /
    %RH / m·s⁻¹ / degrees — inferred from magnitudes, not documented.
  - **License/usage terms: NOT FOUND** — no LICENSE/README/terms file exists
    under `unisolar/`. Provenance not documented locally; user to confirm
    source and redistribution terms.
- **Rationale:** Facts above come from the executed inspection run whose
  artifacts exist; nothing assumed.
- **Consequences:**
  - Canonical model mapping (Phase 2): `timestamp`←`Timestamp`,
    `site_id`←`SiteKey`, `power`←`SolarGeneration`; weather joined at
    campus grain.
  - 56% target missingness dominates cleaning strategy decisions.
  - Solar-position features ARE computable (campus lat/lon available),
    so PRD §18 optional path is open.

## D-007 — Timezone for solar position: Australia/Melbourne (civil time)

- **Date:** 2026-08-24
- **Status:** Accepted — chosen empirically by `src/data/night.py::choose_timezone`
- **Context:** Raw timestamps are naive local wall-clock. Victoria data has two
  candidates: fixed standard time (`Etc/GMT-10`, UTC+10) vs civil time
  (`Australia/Melbourne`, UTC+10/+11 with DST). Wrong choice shifts solar
  elevation by up to ~15° at dawn/dusk.
- **Decision:** Treat timestamps as **Australia/Melbourne civil time**.
  Evidence (25% sample of generation rows): night nonzero-generation fraction
  **0.12%** under Melbourne vs **3.43%** under fixed UTC+10; day/night mean
  contrast ratio **19.15×** vs **8.71×**. Full artifact:
  `artifacts/timezone_decision.json`.
- **Consequences:** DST-transition timestamps are ambiguous/nonexistent;
  localized with `ambiguous="NaT"` / `nonexistent="shift_forward"`. 456 final
  rows carry NaN elevation & False daylight flag (~0.017%) — acceptable,
  documented in cleaning log.

## D-008 — Missing-value strategy

- **Date:** 2026-08-24
- **Status:** Accepted
- **Context:** Target `power` missing on 56.2% of rows; weather vars missing
  28.8–43.8%; generation has 52,492 missing 15-min grid slots (worst site
  2.16%). PRD §9 forbids blind deletion.
- **Decision:**
  1. `power` is **never imputed** — NaN stays NaN (missing ≠ zero; zeros would
     poison both training and night handling).
  2. Weather (temperature/apparent/dew-point/humidity/wind_speed):
     time-based linear interpolation within campus, **limit 2 consecutive
     steps** (30 min). Filled counts logged per column (~88k cells each for
     temperature group, 22 wind-speed).
  3. `wind_direction` is circular — linear interpolation invalid; left as-is.
  4. Missing timestamp slots are left as absent rows; consumers resample onto
     the expected grid if needed.
- **Consequences:** Models must mask/handle power-NaN rows (train on observed);
  long weather outages remain NaN by design rather than fabricating values.

## D-009 — Outliers: analysis-only, no removal

- **Date:** 2026-08-24
- **Status:** Accepted
- **Context:** PRD §9 requires outlier analysis without blind deletion.
- **Decision:** Per-site IQR fence (Q3 + 3·IQR) on daylight-only power,
  reported in `artifacts/outlier_analysis.md`. Result: **0 high-side outliers
  of 1,196,425 daylight observations** (all-site max 99.2 kWh vs fences well
  above). No removal performed.
- **Consequences:** No outlier column added to canonical schema; analysis
  re-runnable via `scripts/build_processed.py`.

## D-010 — Site-scale heterogeneity: normalize per site, evaluate within-site

- **Date:** 2026-08-24
- **Status:** Accepted — grounded in Phase 3 EDA (`artifacts/eda/eda_summary.md`)
- **Context:** EDA measured large between-site scale differences: mean daylight
  output spans 36.9 kWh (site 11) down to ~0 across 42 sites, campus 1 hosts
  27/42 sites, and pooled correlations with power (elevation r=+0.34,
  humidity −0.23) badly understate within-site signal (r=+0.74 / −0.50).
  Capacity (`kWp`) known for only 25/42 sites.
- **Decision:**
  1. Baselines (Phase 4) are computed per site where the baseline is a
     statistic (mean), plus a global variant for comparison.
  2. Feature engineering (Phase 5) includes per-site scale features
     (site-level historical statistics); models must be able to distinguish
     sites beyond weather alone.
  3. Model comparison metrics (Phase 6+) are reported overall AND
     within-site; pooled-only metrics are considered misleading for this
     dataset.
- **Rationale:** Pooled statistics conflate capacity variance with signal;
  within-site correlations show ~2× stronger relationships that models can
  exploit once site identity/scale is representable.
- **Consequences:** Site id + scale features become first-class; cross-site
  evaluation (Phase 9) expected to show degradation on unseen sites — that
  gap will be quantified, not hidden.

## D-011 — Baseline evaluation protocol (splits, persistence history, nRMSE denominator)

- **Date:** 2026-08-24
- **Status:** Accepted — implemented in Phase 4 (`src/data/splits.py`,
  `src/models/baseline.py`, `scripts/run_baselines.py`)
- **Context:** Phase 4 needs honest baseline numbers, but the full split
  machinery (cross-site splits, leakage CI) lands in Phase 9. nRMSE requires a
  documented denominator (PRD §25); site capacity exists for only 25/42 sites.
  Persistence semantics needed care: fitting it on the train slice alone left
  >99% of val/test unpredicted (no t−24h row in the lag table).
- **Decision:**
  1. `src/data/splits.py` implements the canonical per-site chronological
     70/15/15 split (PRD §11.2) now; Phases 5–8 reuse it, Phase 9 extends it
     with cross-site variants.
  2. Statistic-fitting baselines (mean_global / mean_site) fit on the TRAIN
     slice only.
  3. Persistence is non-trainable and strictly causal: its lag table may hold
     the full processed series because a lookup at time t reads only the
     t−24h row (< t). Verified by unit test (corrupting later rows never
     changes earlier predictions).
  4. nRMSE denominator = train-slice observed range of power (max−min):
     pooled for ALL rows (99.12 kWh), per site for SITE rows. Capacity-based
     normalization rejected — kWp missing for 17/42 sites would bias
     comparisons.
  5. Metrics always reported both all-period and daylight-only, on both val
     and test; rows with missing truth or prediction are counted (`n_missing`),
     never dropped silently or filled with zeros (D-008).
- **Rationale:** Keeps every Phase 4 number leak-free and reproducible while
  avoiding duplicate split implementations later.
- **Consequences:** Baseline numbers are comparable to future model numbers
  (same split code). If Phase 9 changes ratio defaults, baselines re-run via
  `scripts/run_baselines.py`.

## D-012 — Feature engineering conventions (lags, rolling, seasons, wind)

- **Date:** 2026-08-24
- **Status:** Accepted — implemented in Phase 5 (`src/features/`,
  `scripts/build_features.py`)
- **Context:** PRD §15–19 leave representation choices open. The dataset's
  52,492 missing grid slots, southern-hemisphere location, and circular wind
  direction each break a naive textbook choice.
- **Decision:**
  1. Lags are **calendar-exact**: feature at t reads the value observed at
     t−Δt by timestamp within the site (MultiIndex lookup), never positional
     `shift(n)` — positional shifts silently misalign across missing slots.
     Missing priors stay NaN (D-008). Verified exact on 1.1M+ pairs.
  2. Rolling windows are time-based with pandas `closed='left'`, i.e.
     interval **[t−W, t)**: current observation excluded, left boundary
     included; `min_periods=1` so partial early history yields real values
     (std is NaN until ≥2 obs). Computed per site.
  3. `season` uses **southern-hemisphere meteorological** convention
     (DJF=summer). `week_of_year` is ISO.
  4. Wind direction becomes `wind_dir_sin`/`wind_dir_cos`; raw degrees are
     dropped from model inputs (359° vs 1° must not look far apart).
  5. Weather variables pass through dynamically (`available_weather_columns`);
     absent sources (irradiance/pressure/cloud — D-006) simply produce no
     columns rather than errors.
  6. Solar position recomputed inside the feature build (pvlib apparent
     elevation/azimuth/zenith at campus coords, D-007 timezone) so all
     position features come from one consistent run; `day_length_hours`
     counts >0°-elevation grid slots per civil date ×0.25 h (documented
     approximation under grid holes).
- **Rationale:** Every choice prioritizes causal correctness and honest NaNs;
  alternatives either leak (right-closed windows), misalign (positional lags),
  or distort geometry (raw degrees).
- **Consequences:** Phase 6+ models consume these columns directly from
  `data/processed/features/`; feature_metadata.json records configs for
  reproducibility. Rebuild via `scripts/build_features.py` (~20 s).

## D-013 — XGBoost configuration and tracking choices

- **Date:** 2026-08-24
- **Status:** Accepted — implemented in Phase 6 (`src/models/xgboost_model.py`,
  `src/config.py`, `configs/{models,training}.yaml`, `scripts/train_xgboost.py`)
- **Context:** PRD §22 makes XGBoost the primary traditional ML model,
  configurable (§43), tracked in MLflow (§30) with reproducibility metadata
  (§44). Dataset specifics force several non-default choices.
- **Decision:**
  1. **Single-step framing**: model predicts `power(t)` from features known
     strictly before t (lags/rolling by D-012 construction) plus covariates at
     t; horizon list lives in config (`horizons_steps_15min: [1]`).
  2. **CPU `tree_method="hist"`** — preprocessing/XGBoost stay GPU-free per
     PRD Rule under §49; GPU reserved for Phases 7–8 deep learning.
  3. **`site_id` as native categorical** (`enable_categorical=True`) rather
     than one-hot — one split can separate sites; aligns with D-010.
  4. **NaN features left as NaN** — XGBoost's sparsity-aware splits handle
     them natively; no imputation of lag/rolling/weather gaps (D-008 spirit).
     Only rows with an observed target are dropped from training/eval input
     counting, never filled.
  5. **Excluded from inputs**: raw `wind_direction` (circular encoding exists),
     string `season` (day-of-year cyclical covers it), `is_daylight`
     (derivable from solar elevation), `timestamp`/`year`/`month`/`campus_id`.
     Feature selection is dynamic — absent families are skipped.
  6. **Early stopping on val via constructor param** (`early_stopping_rounds=100`,
     XGBoost ≥2.x API); `best_iteration` recorded in metadata + MLflow.
  7. **Hyperparameters in `configs/models.yaml`**, protocol in
     `configs/training.yaml`, loaded by flattened `load_config()` — nothing
     hard-coded (PRD §43).
  8. **MLflow file store retained** (`mlruns/`, PRD §30 layout); MLflow 3.x
     gates that backend behind `MLFLOW_ALLOW_FILE_STORE=true`, set inside
     `scripts/train_xgboost.py`. SQLite migration deferred until needed.
  9. **Metrics reuse the D-011 denominators exactly** (pooled train range
     99.119 kWh / per-site train ranges) so XGBoost rows in
     `artifacts/xgboost/metrics.csv` are directly comparable to
     `artifacts/baselines/baseline_metrics.csv`.
- **Rationale:** Each choice keeps evaluation honest and comparable; the
  alternatives (one-hot site ids, mean-imputed lags, GPU trees, ad-hoc
  denominators) would either leak, distort geometry, or break comparability.
- **Consequences:** Later models (Phase 7+) must report against the same
  denominators/splits; if MLflow gains requirements beyond file tracking,
  migrate to SQLite then (`mlflow migrate-filestore` is lossless).

## D-014 — Sequence model (LSTM/GRU) framing, leak guard, and training protocol

- **Date:** 2026-08-25
- **Status:** Accepted — implemented in Phase 7 (`src/models/sequence_model.py`,
  `scripts/train_sequence.py`, `tests/test_sequence_model.py`)
- **Context:** PRD §23/§49 require PyTorch LSTM+GRU with GPU, mixed precision,
  early stopping, LR scheduling, checkpointing, best-model restoration,
  gradient clipping, and CPU fallback. Networks cannot ingest NaN, unlike
  XGBoost (D-013 #4), and a naive window design leaks the target.
- **Decision:**
  1. **Same task as Phase 6**: sliding windows of ``lookback_steps=96`` (24 h)
     predict ``power(t)`` single-step — metrics stay directly comparable to
     XGBoost/baselines under the D-011 split and denominators.
  2. **Channel matrix + mask channel**: all channels zero-filled, plus an
     explicit ``power_observed`` 0/1 channel so the net can learn
     missing-vs-zero (56% target missingness is signal, not noise).
  3. **Target-leak guard (incident)**: the first LSTM run scored test
     MAE 0.099 / R²=1.000 because windows included power(t) as an input
     channel — the net copied the target through the last step. The guard
     zeroes the power+mask channels at step t of every window (history steps
     keep real values; exogenous covariates at t stay). Leaked checkpoint/
     artifacts/MLflow run were deleted and retrained. Comparability sanity
     check ("a single-step model beating XGBoost ~10× on MAE is a bug") is
     what caught it — now also enforced by unit tests asserting the masked
     contract of both window accessors.
  4. **Train-only statistics** (PRD §47): channel scaler and target
     standardization (mean/std) fit on the train split only; val/test
     transformed with train stats. Windows are cut from actual frame rows
     (never positional offsets), never cross site boundaries, so the 52k
     grid holes cannot misalign history.
  5. **Training loop** (PRD §49): AdamW, gradient clipping 1.0,
     ReduceLROnPlateau(factor 0.5, patience 1) on val RMSE, early stopping
     patience 4, fp16 autocast+GradScaler when CUDA present, CPU fallback;
     best-epoch state_dict checkpointed atomically to
     ``models/<arch>_site_all_h1_v1.pt`` and restored before eval; per-epoch
     val RMSE/LR logged to MLflow. Hyperparameters live in
     ``configs/models.yaml`` only.
  6. **Negative predictions retained unclipped**, matching the documented
     XGBoost behaviour.
- **Environment note:** training scripts exit with code **9 after fully
  successful runs** on this box — reproduced via both `conda run -n solar`
  and the env's `python.exe` directly, with artifacts + MLflow logs complete
  every time. Attributed to torch/cu132 interpreter-teardown after CUDA use,
  not to the script (which returns 0). Automation must check ARTIFACTS, not
  the process exit code, for Phase 7+ torch scripts.
- **Rationale:** Mirrors Phase 6's honest-eval choices where they transfer;
  deviates only where networks force it (masking instead of native NaN
  handling). The leak incident reinforced Rule: suspiciously-good results get
  investigated, not celebrated.
- **Consequences:** Phase 8 Transformer reuses the same WindowDataset/guard;
  per-site min-R² can go negative on tiny/noisy sites (observed −0.52 on one
  site) — reported, not hidden.

## D-015 — Transformer design and model-comparison artifact

- **Date:** 2026-08-25
- **Status:** Accepted — implemented in Phase 8 (`TransformerForecaster` in
  `src/models/sequence_model.py`, `scripts/compare_models.py`)
- **Context:** PRD Rule 8 gates the Transformer on Phase 7 being stable; §23
  wants a time-series Transformer; Phase 8 also requires comparing against
  prior models.
- **Decision:**
  1. **Reuse the Phase 7 machinery unchanged** — same `WindowDataset`
     (24 h lookback × 13 channels), same current-step power/mask leak guard,
     same train-only scaling, same `train_sequence` loop (PRD §49) and eval
     protocol. Only the encoder differs: self-attention instead of recurrence.
     This keeps all four models strictly comparable.
  2. **No causal mask**: every window position is ≤ t by construction
     (window = [t−96 … t]), so there is no future to mask out; the target-
     leak guard handles the only dangerous position (t itself).
  3. **Fixed sinusoidal positional encoding** (registered buffer, excluded
     from parameters) rather than learned embeddings — ~800k train windows
     do not need extra positional capacity, and it cannot overfit.
  4. **Pre-LN encoder** (`norm_first=True`, GELU FFN), scaled input
     projection, LayerNorm after the stack, last-position readout → MLP head
     mirroring the RNN head. Last position = step t whose power channels are
     masked; covariates at t remain legitimate inputs.
  5. **Lower LR than the RNNs** (5e-4 vs 1e-3) per standard Transformer
     practice; everything else identical in config (`configs/models.yaml`).
  6. **Comparison deliverable** (Phase 8 box + PRD §26):
     `scripts/compare_models.py` collects ALL-scope rows from every existing
     metrics file (baselines/xgboost/lstm/gru/transformer) into
     `artifacts/evaluation/{model_comparison.csv,model_comparison.json,
     evaluation_report.md}` — numbers copied verbatim, absent models skipped
     with a note, never invented.
- **Rationale:** Minimal-delta architecture isolates "attention vs recurrence"
  as the only experimental difference; the shared harness guarantees the
  comparison is honest.
- **Consequences:** If Phase 9+ changes splits or horizons, one rerun of
  `train_sequence.py --arch <m>` + `compare_models.py` regenerates everything.

## D-016 — Cross-site evaluation protocol and leakage-test mapping

- **Date:** 2026-08-25
- **Status:** Accepted — implemented in Phase 9 (`cross_site_split` in
  `src/data/splits.py`, `scripts/run_cross_site.py`, `tests/test_cross_site.py`,
  `tests/test_leakage.py`)
- **Context:** PRD §12 demands a second evaluation protocol answering "can
  the model generalize to a PV site absent from training?", with site counts
  sized dynamically; §46 demands automated leakage tests that fail CI.
- **Decision:**
  1. **Dynamic held-out sizing**: ``round(n_sites × frac)``, min 1 — the
     configured 0.15/0.15 gives the PRD's example shape on this dataset
     (42 sites → 30 train / 6 val / 6 test).
  2. **Seeded random site selection**, not ordered ids: site ids cluster by
     campus (campus 1 hosts 27/42), so slicing by id would bias held-out sets
     toward one campus/climate pocket. Seed comes from `training.seed`.
  3. **Seen frames mirror D-011**: train-sites cut chronologically
     70/15/15 → train / val_seen / test_seen. **Unseen frames are the full
     observed history** of each held-out site — they contribute nothing to
     training or scaling, so no temporal restriction is needed and every row
     is genuinely unseen-site.
  4. **Persistence reads its strictly-causal t−24h table built over the whole
     table** (D-011 #3 precedent), so unseen sites are predicted from their
     own history rather than NaN-starved.
  5. **nRMSE denominator deviation, documented**: pooled observed range of the
     CROSS-SITE train slice for every scope including SITE rows — held-out
     sites have no train slice, so D-011's per-site denominator is undefined
     for them. MAE/RMSE/R² stay directly comparable across protocols;
     nRMSE values are only comparable within this experiment.
  6. **Sequence models early-stop on SEEN validation windows**; channel
     scaler + target stats fit on the cross-site train slice only.
  7. **Leakage tests map PRD §46 surfaces to concrete tests**
     (`tests/test_leakage.py`): future-power corruption vs lags; future
     perturbation vs rolling; interpolation limit (gaps >2 steps stay NaN —
     the accepted ≤30-min bounded weather smoothing of D-008 is pinned by
     test); eval corruption cannot move train-fitted scaling stats; held-out
     site disjointness against the real split function; and
     `train_xgboost` proven structurally unable to see a test frame (no such
     parameter exists; stub captures that early stopping consumes exactly
     the validation frame). Two §46 items remain untestable until their
     machinery exists in later phases: feature-selection fitting (Phase 10
     SHAP/selection) and hyperparameter search loops (none implemented).
- **Rationale:** The experiment's value is the honest size of the
  seen→unseen gap; every choice above removes a way to flatter it.
- **Consequences:** The measured gap becomes the headline generalization
  result (PRD calls it "a major project result"); if Phase 10+ adds feature
  selection, its leakage tests extend `tests/test_leakage.py`.

## D-017 — Explainability: SHAP on XGBoost only, exact TreeExplainer

- **Date:** 2026-08-25
- **Status:** Accepted — implemented in Phase 10 (`src/explain/shap_explain.py`,
  `scripts/run_shap.py`, `tests/test_shap.py`)
- **Context:** PRD §29 wants global importance, local explanations and
  contribution plots ("why did the model predict this power value?").
  Candidates: the XGBoost run, the three sequence models (windows × 13
  channels), or both.
- **Decision:**
  1. **Explain only the XGBoost run** (`xgboost-site-all-h1-v1`): it is the
     best test-split model overall (Phase 6/8 comparison), and exact
     ``TreeExplainer`` values are fast and deterministic on trees.
     Sequence models get no SHAP: there is no exact fast path through the
     shared window pipeline, Deep/Gradient explainers would be approximate,
     slow, and their channel-space attributions would not map back to named
     features without an extra convention. Revisit only if a sequence model
     becomes the production pick.
  2. **Explained data = canonical TEST split** (D-011), not train/val:
     explanations must describe the model's deployment-time behavior, and
     explaining eval rows cannot leak anything (read-only use of a frozen
     booster).
  3. **Global = seeded 20k-row subsample** of observed-target test rows;
     local scenarios picked from the FULL test frame so rare conditions
     still find candidates (night / clear-noon peak / morning ramp /
     overcast afternoon; representative = median-power row, max for peak).
  4. **Categorical caveat pinned by test**: xgboost validates category
     dtype/index positionally inside shap's internal predict, so any
     consumer of these artifacts must rebuild inputs via ``prepare_matrix``
     with the model's stored layout (``feature_cols_`` restored from run
     metadata — the native JSON save does not carry it). Additivity
     (base + ΣSHAP ≈ prediction) asserted in tests at <1e-4 and measured in
     artifacts (~1e-6).
- **Rationale:** Exact tree attributions on the primary tabular model answer
  PRD §29 fully at negligible cost; half-explained deep models would add
  noise, not insight.
- **Consequences:** Feature-importance claims in RESULTS.md are now SHAP-
  based for XGBoost (gain-based importance stays in Phase 6 artifacts);
  if Phase 11+ adds conformal intervals per-model, explanation coverage is
  defined only where this decision defines it.

## D-018 — Uncertainty: split conformal, Mondrian regimes, VAL calibration

- **Date:** 2026-08-25
- **Status:** Accepted — implemented in Phase 11 (`src/models/conformal.py`,
  `scripts/run_conformal.py`, `tests/test_conformal.py`)
- **Context:** PRD §28 wants prediction intervals, preferring a
  model-agnostic conformal implementation. Alternatives: quantile
  regression (needs retraining per level, model-specific), Monte Carlo
  dropout (deep models only), split conformal (post-hoc, any fitted
  model).
- **Decision:**
  1. **Split conformal on absolute residuals**: radius = finite-sample
     quantile ``ceil((n+1)(1−α))/n`` of ``|y − ŷ|`` on a calibration frame;
     intervals ``ŷ ± q``. No retraining, works for any forecaster —
     calibrated for XGBoost **and persistence** (the interval reference
     floor).
  2. **Mondrian calibration by regime** ``{day, night} × {lag_present,
     lag_missing}`` alongside the global radius. Solar error is violently
     heteroscedastic (Phase 3: night ≈ zero output; Phase 10: missing-lag
     rows carry ~35× night MAE), so a single radius is either useless at
     night or blind midday. Regime labels use only information available at
     inference time. Unseen labels fall back to the global radius.
  3. **Calibrate on VAL, evaluate on TEST** (canonical D-011 frames).
     Caveat, recorded and tested empirically: VAL also fed XGBoost early
     stopping (one scalar — ``best_iteration``), so calibration residuals
     are mildly optimistic; the honest check is empirical TEST coverage,
     reported per scope. A dedicated third split would cost data for a
     one-parameter effect.
  4. **Levels 0.9 and 0.8** (PRD §28's example shows 0.90); degenerate
     small-α case (k > n) returns max score rather than crashing.
- **Rationale:** Conformal is the only option that adds uncertainty to the
  already-frozen production model with a distribution-free guarantee, and
  the Mondrian variant is the smallest step from marginal to conditional
  validity that the data's regime structure justifies.
- **Consequences:** Coverage claims in RESULTS.md are marginal (ALL rows)
  plus per-regime/per-site empirical spreads; the API phase (§31) can serve
  ``sample_forecast.json``-shaped bounds directly from the interval
  parquet. If a future phase retrains with a dedicated calibration split,
  only ``run_conformal.py``'s frame choice changes.

## D-019 — Backend: recursive multi-step serving, two served models, artifacts-only store

- **Date:** 2026-08-25
- **Status:** Accepted — implemented in Phase 12 (`src/api/{app,store,
  forecast}.py`, `scripts/run_api.py`, `tests/test_api.py`)
- **Context:** PRD §33–36 want a FastAPI service under ``/api/v1``
  (health/dataset/sites/models, single + batch forecast, site history,
  per-model metrics). The trained models are all **single-step** (one
  15-min slot); the API must produce horizons up to a day.
- **Decision:**
  1. **Multi-step by recursion**: predict t+1, append the prediction to the
     site's history, rebuild the full Phase 5 feature families on the small
     tail frame, predict t+2 … Weather covariates for future timestamps are
     **carried forward from the last observation** — there is no NWP feed in
     v1; solar geometry and calendar features are exact for any future time.
     Error accumulation is accepted and bounded by the horizon cap
     (96 steps = 24 h).
  2. **Two models served**: ``xgboost`` (recursive path + conformal bounds)
     and ``persistence`` (t−24h lookup on its own extended series).
     LSTM/GRU/Transformer stay **registered but not served** — no recursive
     path is implemented for them, and requesting one returns 409 rather
     than pretending. The registry distinguishes the two states explicitly.
  3. **Conformal radii attach per step** from the Phase 11 Mondrian
     calibration (level 0.9), using only the regime label knowable at
     forecast time (day/night from solar elevation, lag presence from the
     recursive frame). Bounds are unclipped, consistent with D-018.
  4. **Store = artifacts already on disk**: partition-filtered parquet reads
     (never the full 2.7M-row table), metrics CSVs, conformal metadata;
     booster lazy-loaded once per process with its stored feature layout
     rebuilt via ``prepare_matrix`` (D-017 #4). Tests inject an in-memory
     ``MemStore`` with the same interface, so the suite touches no artifact.
  5. **Limits**: horizon ∈ [1, 96], batch ≤ 10 requests; unknown site/model
     → 404; bad resolution/horizon/batch shape → 422.
- **Rationale:** Recursion reuses the validated single-step stack end-to-end
  (same features, same booster, same leakage guards) instead of training a
  new multi-step model family in the API phase; serving exactly what was
  evaluated keeps RESULTS.md honest about what the API can deliver.
- **Consequences:** Forecast quality degrades with horizon (weather held at
  last observation); the frontend should surface the confidence bands and
  regime labels. If Phase 14 wires deep models into serving, only
  ``REGISTRY.served`` flags and a recursive wrapper per architecture change.

## D-020 — Frontend: bundled snapshot JSONs for surfaces the fixed PRD §33–36 API doesn't cover

- **Date:** 2026-08-25
- **Status:** Accepted — implemented in Phase 13 (`frontend/src/data/`,
  `scripts/export_frontend_data.py`)
- **Context:** The PRD's API surface (§33–36) covers dataset card, sites,
  history, forecast, and per-model metrics. Three frontend pages need data
  with no endpoint: Explainability (SHAP global importance), Data Quality
  (cleaning/outlier/weather-missingness aggregates), and Sites (per-site
  availability + capacity aggregates). Building three more bespoke endpoints
  for read-only, rarely-changing derived artifacts would duplicate the
  export scripts that already produce this content.
- **Decision:** Ship those three datasets as **bundled snapshot JSONs**
  under `frontend/src/data/` (`shap_global_importance.json`,
  `data_quality.json`, `site_summary.json`; ~25 kB total), generated by
  `scripts/export_frontend_data.py` from the same phase artifacts the API
  reads. Refresh workflow: `conda run -n solar python
  scripts/export_frontend_data.py` after any upstream artifact changes.
  Everything the PRD §33–36 surface *does* cover (dashboard, forecast,
  comparison, history) is served live through the Vite dev proxy → FastAPI.
- **Rationale:** These views are descriptive reports over frozen training-
  time artifacts (SHAP run, cleaning log, site table) — not user-driven
  queries — so bundling keeps the API contract exactly as specified while
  still rendering every PRD §38 page from real project data. A stale
  snapshot is visible against live metrics on the same screen, and the
  regeneration step is one command recorded here.
- **Consequences:** Snapshot pages do not reflect re-training until the
  export script is re-run; if Phase 14 integration prefers endpoints, each
  bundle maps 1:1 to a trivial GET handler reading the same artifact.

## D-021 — `is_daylight` serializes as a JSON boolean; consumers must not test `=== 1`

- **Date:** 2026-08-25
- **Status:** Accepted — fixed in Phase 13 (`frontend/src/lib/types.ts`,
  `frontend/src/pages/dashboard.tsx`)
- **Context:** The `/sites/{id}/history` response carries `is_daylight`
  as FastAPI's serialization of a Python bool — i.e. `"is_daylight": true`.
  The dashboard initially consumed it as `(r.is_daylight ?? 0) === 1`,
  which is always false against JSON booleans (`true === 1` is false in
  JS). Result: every point classified as night, night spans never closed,
  and the night shading silently never rendered — no console warning, and
  curl-based checks missed it because Python treats `True == 1` as true.
- **Decision:** The API contract is **boolean** (`true`/`false`). The
  TypeScript type is `is_daylight?: boolean` and the only valid consumer
  check is strict equality against `true`/`false`. Any future consumer
  (notebooks included) must grep the raw JSON rather than rely on Python
  truthiness to verify this field.
- **Rationale:** One silent-rendering bug cost several blind tuning
  iterations before a DOM probe revealed the spans list was empty; pinning
  the serialization fact here prevents a repeat in Phase 14 integration or
  any new client.
- **Consequences:** None server-side — the backend was always correct;
  this documents the wire format and the debugging lesson (verify actual
  JSON bytes, not language-coerced comparisons).

## D-022 — Deep models served recursively over their Phase 7 windows; scalers exported; no conformal bounds

- **Date:** 2026-08-25
- **Status:** Accepted — implemented in Phase 14 (`src/api/{store,forecast,
  app}.py`, `scripts/export_sequence_scalers.py`)
- **Context:** Phase 12 served only xgboost + persistence (D-019 left
  lstm/gru/transformer registered-but-not-served, 409). The checkpoints
  store network weights only — channel and target standardization stats
  existed only in the training process's memory, so serving was blocked
  on more than a registry flag.
- **Decision:**
  1. **Scaler export, not recomputation-at-startup**:
     `scripts/export_sequence_scalers.py` recomputes y_mean/y_std and the
     13-channel mean/std exactly as `train_sequence.py` did (same features
     table, same `chronological_split`, same
     `build_channel_matrix`/`fit_channel_scaler`) and writes
     `artifacts/{arch}/serving_scalers.json`. The script refuses to write
     if a reloaded checkpoint + exported scalers cannot reproduce the
     stored test-split predictions (validated on 2048 windows per arch on
     the same device class as the recorded run: max |Δ| = 0.0 for all
     three).
  2. **Recursive serving reuses the Phase 7 window contract**: each step
     appends the prediction to the site frame, rebuilds Phase 5 features
     (weather carried forward per D-019), takes the last `lookback+1 = 97`
     rows, standardizes with the exported channel stats, zeroes the
     current-step power channels (`_mask_current_step`), and inverse-
     transforms the network output. Predictions feed back with mask=1 —
     the same accepted error-accumulation compromise as the xgboost lags.
  3. **No conformal bounds for deep models**: Phase 11 calibrated
     xgboost + persistence only. Bounds fields are omitted rather than
     borrowed from another model's radii; the frontend copy states this
     explicitly.
  4. **Serving runs on CPU**, checkpoints lazy-loaded once per process;
     worst case (96 steps, transformer) ≈ 3.7 s per request.
- **Rationale:** Serving exactly what was evaluated keeps the API honest —
  the recursive path is the same window pipeline the test metrics were
  computed with, and the export script's bit-exact validation proves the
  served models reproduce the recorded RESULTS.md numbers' inputs.
- **Consequences:** Deep-model forecasts can go slightly negative at night
  (networks have no non-negativity constraint; raw output is shown, as
  with xgboost). If conformal calibration for the deep models is wanted,
  it is a Phase-11-style addition (`artifacts/{arch}` + radii lookup),
  not a serving change.

## D-023 — Incremental serving features + client-side forecast cache (2026-08-25)

- **Context:** Recursive serving rebuilt the full Phase 5 feature set over
  the growing frame at every one of up to 96 steps per request
  (~2.4–3.0 s end-to-end); users selecting models on the Forecast page saw
  multi-second waits.
- **Decision:**
  1. Features that depend only on the timestamp — calendar encodings,
     carried-forward weather, solar geometry (pvlib) — are computed ONCE
     per request for tail+horizon (`_static_frame` in
     `src/api/forecast.py`), not per step.
  2. Prediction-dependent features (lags, rolling stats) are refreshed per
     step for the single new row via `_PowerHistory`, a timestamp-keyed
     series with searchsorted lookups that reproduces `add_lags`
     (calendar-exact, NaN when absent) and `add_rolling_features`
     (time-based, closed-left [t−W,t), NaN-truth excluded, ddof=1 std)
     semantics. Equality with the Phase 5 batch functions is pinned by
     `tests/test_api.py::TestIncrementalFeatures` on a gappy series with
     fed-back predictions; a real-data harness additionally compared all
     43 XGBoost feature columns and the deep-model channel windows against
     the previous loop (identical).
  3. The frontend caches forecast responses (promises) keyed by
     (site_id, model, horizon) and debounces horizon-slider requests by
     300 ms; forecasts are deterministic per key because recursion always
     starts at the dataset's last observation (D-019).
- **Rationale:** Removes the O(steps × frame) rebuild while keeping the
  served feature semantics provably identical to training-time Phase 5
  semantics; caching is safe because the forecast origin is fixed.
- **Consequences:** A future change that makes forecasts depend on request
  time or user-provided origin invalidates the client cache key and this
  design. `build_features` remains the batch reference implementation used
  by tooling/tests.

## D-024 — Job-scoped display-only training (Train page) (2026-08-26)

POST /api/v1/train/jobs spawns `scripts/train_from_folder.py` per job; all
outputs land in `data/train_jobs/<dataset_id>/<job_id>/` (raw uploads,
staged parquet, features, model artifacts, result.json). Job state derives
from filesystem log markers (`== STAGE … start|done`, `== DONE`,
`== FAILED`) parsed by `src/api/train.py`; success is marker-based because
torch teardown on this box exits 9 after successful runs. One heavy job at
a time (409 otherwise). The served v1 models, phase artifacts, and recorded
RESULTS.md numbers are never modified — Train-page runs are display-only.

## D-025 — Post-PRD frontend graph additions (2026-08-26)

The PRD §38 pages are augmented with artifact-derived static bundles written
by `scripts/export_frontend_data.py` (D-020 pattern, no new REST endpoints):
`site_monthly` + `quality_extra` (earlier increment), then `eda_profiles`,
`missingness_timeline`, `evaluation_series`, `cross_site_summary`. Bundles
are regenerated after any artifact refresh and committed so a fresh clone
builds without a Python env. Pred-vs-actual overlays keep night gaps as
nulls (never bridged); baselines carry metrics only, series fields exist
solely where a predictions parquet exists.
