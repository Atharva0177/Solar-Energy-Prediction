# Design — Train-page hardening + richer dashboard visuals

Date: 2026-08-26
Status: approved (brainstorming session, same day)
Project: UNISOLAR solar forecasting platform (`E:\Solar_gemini`)

## Context

Two user asks, agreed as one spec:

1. **More graphs / data on the website.** The Train feature from the previous
   session is ~90 % built (`scripts/train_from_folder.py`,
   `src/api/train.py`, `frontend/src/pages/train.tsx`, route + nav + API
   client) but was never run live and has no tests; `DECISIONS.md` lacks the
   D-024/D-025 entries its code references.
2. **A Train page** — already built; remaining work is hardening only.

User decisions taken during brainstorming:

- Both work items in one spec.
- Extra visuals = **model evaluation visuals** + **dataset/EDA visuals**
  (not site-level content, not inline PNG reports).
- Placement = **enrich existing pages** (no new page/nav).
- Data delivery = **Approach A — extend the offline export script**
  (D-020/D-025 pattern); no new backend endpoints.
- Train page = **harden only** (no past-runs browser, no editable
  hyperparameters).

## Goals

1. Four new static data bundles + charts on Dashboard, Data Quality, and
   Model Comparison pages.
2. Train feature verified end-to-end: tests, one live real run, docs/trackers
   backfilled.
3. Fix the discovered exit-code-9 job-status bug.

## Non-goals

- No new REST endpoints; no changes to served v1 models or phase artifacts.
- No Forecast/Sites/Explainability page changes; no new nav items.
- No PNG embedding, site-level scatter/sparklines, job history UI, or
  hyperparameter editing.

## Part 1 — New visuals (Approach A)

### Architecture & data flow

```
phase artifacts (frozen)          scripts/export_frontend_data.py (extended)
data/processed/solar/       ──►   eda_profiles.json
artifacts/validation_report ──►   missingness_timeline.json     frontend/src/data/
artifacts/*/predictions_    ──►   evaluation_series.json        (static imports)
  test.parquet ×8 models    ──►   cross_site_summary.json
```

The existing export script gains four bundle builders; pages import the JSONs
directly like every existing bundle. Refresh stays one command:
`conda run -n solar python scripts/export_frontend_data.py`.

### Bundle specs

**`eda_profiles.json`** — Dashboard additions

```jsonc
{
  "hour_of_day": {
    "slots": ["00:00", "00:15", "..."],            // 96 labels
    "mean_kw": { "ALL": [...96], "1": [...], ... } // per campus_id, mean daylight-observed power per slot
  },
  "correlation": {
    "campuses": [1, 2, 3, 4, 5],
    "vars": ["temperature", "humidity", "wind_speed", "solar_elevation_deg"],
    "power_corr": [[...5×4]]                        // Pearson r, daylight-observed rows per campus
  }
}
```

Computed from `data/processed/solar/` partition reads only.

**`missingness_timeline.json`** — Data Quality addition

```jsonc
{
  "months": ["2020-01", "..."],                    // full span
  "generation_missing_slot_pct": [...]             // absent rows vs expected 15-min grid per month
}
```

Expected grid derived from each site's first/last timestamp at 15-min cadence;
complements the weather monthly timeline already in `data_quality.json`.

**`evaluation_series.json`** — Model Comparison additions

```jsonc
{
  "test_window": {"start": "...", "end": "..."},
  "models": {
    "<run_name>": {
      // present for ALL 8 runs of model_comparison.csv (baselines included)
      "metrics": {"mae": ..., "rmse": ..., "r2": ..., "nrmse": ...},
      // series fields ONLY where a predictions_test.parquet exists
      // (xgboost, lstm, gru, transformer); omitted/null for pure baselines
      "hourly_all": {"t": ["2021-12-21T01:00", ...],                   // hourly means, ~670 pts
                     "actual": [...], "predicted": [...]},
      "daily_by_site": {"<site_id>": {"actual": [...~29], "predicted": [...]}},
      "scatter_sample": {"actual": [...≤2000], "predicted": [...]},    // seeded uniform sample
      "residual_hist": {"edges": [...41], "counts": [...40]} | null
    }
  }
}
```

Metric tiles/bars read `metrics` for all runs; the pred-vs-actual explorer
lists only runs with series data. Per-site detail is daily resolution to keep
the bundle < ~1 MB total; ALL-scope detail is hourly.

**`cross_site_summary.json`** — Model Comparison addition

```jsonc
{
  "models": {
    "<model>": {
      "seen_val_all": {...metrics}, "unseen_test_all": {...metrics},
      "unseen_site_r2": [{"site_id": n, "r2": x}, ...]                // flags e.g. site 29 R² −6.859
    }
  }
}
```

Copied verbatim from `artifacts/cross_site/cross_site_metrics.csv`
(`scope == ALL` rollups + unseen-test SITE rows).

### Page changes (frontend only)

| Page | Chart | Source |
|---|---|---|
| Dashboard | Hourly generation profile (per-campus lines) | `eda_profiles.json` |
| Dashboard | Weather↔power correlation heatmap (5 campuses × 4 vars) | `eda_profiles.json` |
| Quality | Generation gap timeline (% missing slots/month) | `missingness_timeline.json` |
| Comparison | All-models grouped metric bars (MAE/RMSE/R²/nRMSE, 8 runs incl. baselines) | `evaluation_series.json` |
| Comparison | Pred-vs-actual explorer: model + site selectors, hourly overlay + residual histogram | `evaluation_series.json` |
| Comparison | Cross-site seen-vs-unseen paired bars + unseen-R² strip | `cross_site_summary.json` |

Charts follow the existing `components/viz.tsx` conventions (VizCard,
StatTile, axis ticks, zero-based bar axes, tooltips).

### Size budget

Hourly-ALL ≈ 670 pts × 3 arrays × 4 series-runs; daily-per-site ≈ 29 × 2 ×
42 × 4; scatter ≤ 2 000 pts/run. Target: each new bundle ≤ ~300 kB, total
addition < ~1 MB raw JSON (gzip in build where applicable).

## Part 2 — Train hardening

### Bug fix (must ship)

`src/api/train.py::_watch` treats `returncode != 0` as failure, but torch
scripts on this machine **exit code 9 after success** (Phase 7 gotcha,
documented in memory). Deep-model jobs would complete yet render as failed.

Fix: success = `== DONE` marker present in the log; returncode recorded for
diagnostics only.

### Tests (~18 new)

**`tests/test_train_api.py`**

- Path verification: synthetic mini-dataset (2 sites × a few days, 15-min)
  in tmp dir → 200 with profile; bad folder → 422 with structured
  `{message, files}` detail; missing dir → 422.
- Upload verification: multipart upload of the three CSVs → registered
  dataset; wrong/extra filenames rejected; incomplete set → 422 with
  per-file checklist.
- Job lifecycle with monkeypatched subprocess (fake script writes
  `== STAGE … start|done` markers then `result.json`): start returns job id;
  status exposes stages/log_tail/result; artifact download serves the three
  allowed names and rejects others (422); unknown dataset/job → 404.
- One-heavy-job rule: second concurrent start → 409.
- `/train/config` snapshot shape.
- Regression: log ending `== DONE` with exit code 9 → status `done`.

**`tests/test_train_pipeline.py`**

- Full `train_from_folder.py --fast-test` on the synthetic mini-dataset:
  exits through all five stages, writes `result.json` with the documented key
  set, metrics CSV has val+test × ALL+SITE rows, model/prediction artifacts
  land inside the job dir only, and repo `models/` + phase artifacts are
  byte-identical before/after (display-only guarantee, D-024).

Synthetic fixture builder shared by both files (deterministic, tiny —
fast-test keeps the whole chain to seconds).

### Live verification (first-ever run)

1. Start FastAPI + Vite dev server.
2. Verify folder `E:\Solar_gemini\unisolar`; screenshot checklist + profile tiles.
3. Start a real XGBoost job; capture progress stages + live log.
4. Confirm results card renders measured numbers; download artifacts.
5. Run one deep-model fast-test end-to-end (exercises the exit-code-9 path).
6. Playwright screenshots of verify/running/results states (Phase 13 pattern).

Measured live-run numbers go to RESULTS.md (post-PRD subsection).

### Docs / tracker backfill

- `DECISIONS.md`: D-024 (job-scoped display-only training via
  `scripts/train_from_folder.py`; served v1 untouched), D-025 (post-PRD
  frontend graph additions — existing bundles + this spec's four).
- `TASKS.md`: new post-PRD enhancement section listing shipped items.
- `RESULTS.md`: measured fast/live-run timings + test counts.
- `memories/project-state.md`: refresh snapshot.

## Definition of done

- pytest suite green: 147 existing + ~18 new tests.
- `npm run build` + oxlint clean with the six new charts mounted.
- Export script re-run produces the four bundles within size budget.
- Live XGBoost train run verified visually end-to-end; deep-model fast-test
  passes through the fixed success path.
- Docs/trackers updated as above.

## Risks / notes

- Prediction parquets carry NaN actuals (night rows) — aggregation must
  drop them from means/histograms but keep timestamps continuous enough for
  honest overlays (gaps visible, not bridged by fake zeros).
- `is_daylight` serializes as JSON boolean — reuse D-021 contract when
  filtering daylight rows client-side.
- Torch teardown exit-code quirk also affects the pipeline test if it loads
  torch — assert via result.json/artifacts, not process exit codes alone.
- Bundle drift: export script is the single refresh path; README note added
  alongside D-025 entry.
