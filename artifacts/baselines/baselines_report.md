# Baselines Report (auto-generated)

_Protocol (D-011): per-site chronological 70/15/15 split (train 1,912,356 / val 409,796 / test 409,794 rows). Fit on train only. nRMSE denominator = train-slice observed range of power = 99.12 kWh pooled (SITE rows use each site's own train range). All figures from this run's `baseline_metrics.csv`._

## Test-split summary (ALL sites)

| baseline | MAE | RMSE | R² | nRMSE | Daylight MAE | Daylight nRMSE | n_eval | preds missing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| persistence_prev_day | 2.783 | 6.294 | 0.699 | 0.064 | 2.785 | 0.064 | 182,029 | 227,765 |
| mean_site | 4.896 | 8.396 | 0.459 | 0.085 | 4.841 | 0.084 | 191,732 | 218,062 |
| mean_global | 6.502 | 11.454 | -0.006 | 0.116 | 6.502 | 0.116 | 191,732 | 218,062 |
| zero | 7.499 | 13.659 | -0.431 | 0.138 | 7.517 | 0.138 | 191,732 | 218,062 |

## Validation-split summary (ALL sites)

| baseline | MAE | RMSE | R² | nRMSE | Daylight MAE | Daylight nRMSE | n_eval | preds missing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| persistence_prev_day | 3.425 | 7.599 | 0.630 | 0.077 | 3.427 | 0.077 | 186,080 | 223,716 |
| mean_site | 5.099 | 9.138 | 0.470 | 0.092 | 5.070 | 0.092 | 201,552 | 208,244 |
| mean_global | 6.804 | 12.614 | -0.011 | 0.127 | 6.805 | 0.127 | 201,552 | 208,244 |
| zero | 7.873 | 14.812 | -0.394 | 0.149 | 7.884 | 0.150 | 201,552 | 208,244 |

## Notes

- Persistence is the primary baseline (PRD §21): prediction at t uses the observation at t−24h. The lookup table holds the full processed series but every read is strictly backward (t−24h < t), so no future information is used (D-011). Missing prior-day observations yield NaN predictions, counted under `preds missing`, never zeros (D-008).
- Per-site rows (`scope=SITE`) live in the CSV and quantify across-site spread (D-010); this summary shows pooled ALL rows.
- Daylight-only columns use the Phase-2 `is_daylight` flag so night zeros cannot flatter a baseline (PRD §10).