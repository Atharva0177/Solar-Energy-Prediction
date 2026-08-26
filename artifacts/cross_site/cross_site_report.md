# Cross-site evaluation report (PRD §12, Phase 9)

_Generated 2026-08-25T03:43:07+00:00; seed 42. Held-out sites: val [9, 14, 16, 28, 32, 35] / test [7, 13, 26, 29, 33, 36]. nRMSE denominators: pooled cross-site train range 99.119 kWh for every scope (D-016)._ 

## Headline — test split, ALL sites

| model | seen MAE | unseen MAE | gap % | seen RMSE | unseen RMSE | seen R² | unseen R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| persistence_prev_day | 3.308 | 1.821 | -45% | 7.197 | 2.978 | 0.687 | 0.427 |
| xgboost | 1.241 | 1.242 | +0% | 2.887 | 1.812 | 0.949 | 0.795 |
| lstm | 1.384 | 0.801 | -42% | 2.973 | 1.355 | 0.946 | 0.885 |
| gru | 1.363 | 0.771 | -43% | 2.948 | 1.325 | 0.947 | 0.890 |
| transformer | 1.332 | 0.764 | -43% | 2.938 | 1.286 | 0.947 | 0.897 |

Reading: 'seen' = late history of training sites; 'unseen' = full history of held-out sites. The gap measures how much of each model's accuracy depends on having seen the plant during training (site identity/scale features, D-010).

## seen sites — test ALL (ranked by MAE)

| model | MAE | RMSE | R² | nRMSE | Daylight MAE | n_eval |
|---|---:|---:|---:|---:|---:|---:|
| xgboost | 1.241 | 2.887 | 0.949 | 0.029 | 1.238 | 140492 |
| transformer | 1.332 | 2.938 | 0.947 | 0.030 | 1.333 | 138960 |
| gru | 1.363 | 2.948 | 0.947 | 0.030 | 1.367 | 138960 |
| lstm | 1.384 | 2.973 | 0.946 | 0.030 | 1.388 | 138960 |
| persistence_prev_day | 3.308 | 7.197 | 0.687 | 0.073 | 3.312 | 134416 |

## unseen sites — test ALL (ranked by MAE)

| model | MAE | RMSE | R² | nRMSE | Daylight MAE | n_eval |
|---|---:|---:|---:|---:|---:|---:|
| transformer | 0.764 | 1.286 | 0.897 | 0.013 | 0.763 | 162186 |
| gru | 0.771 | 1.325 | 0.890 | 0.013 | 0.770 | 162186 |
| lstm | 0.801 | 1.355 | 0.885 | 0.014 | 0.801 | 162186 |
| xgboost | 1.242 | 1.812 | 0.795 | 0.018 | 1.242 | 162456 |
| persistence_prev_day | 1.821 | 2.978 | 0.427 | 0.030 | 1.821 | 145044 |

## Unseen test sites — per-site spread

| model | sites | MAE min | MAE max | R² min | R² max | neg-R² sites |
|---|---:|---:|---:|---:|---:|---:|
| gru | 6 | 0.294 | 1.128 | 0.655 | 0.889 | 0 |
| lstm | 6 | 0.340 | 1.163 | 0.558 | 0.884 | 0 |
| persistence_prev_day | 6 | 0.497 | 2.974 | 0.175 | 0.450 | 0 |
| transformer | 6 | 0.279 | 1.157 | 0.805 | 0.885 | 0 |
| xgboost | 6 | 0.948 | 1.985 | -6.859 | 0.878 | 1 |