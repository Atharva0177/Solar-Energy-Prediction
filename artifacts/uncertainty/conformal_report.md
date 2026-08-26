# Conformal prediction intervals (PRD §28, Phase 11)

_Generated 2026-08-25T04:39:13+00:00; split conformal, absolute residuals, calibrated on VAL observed rows, evaluated on TEST (D-011). Mondrian regimes = {day,night} × {lag_present,lag_missing} (Phase 10 night failure mode)._

## Calibration radii (VAL, kWh)

| model | method | level | global radius | regime radii |
|---|---|---|---:|---|
| xgboost | global | 0.9 | 2.922 | - |
| xgboost | global | 0.8 | 1.556 | - |
| xgboost | mondrian | 0.9 | 2.922 | day_nolag 3.08, day_lag 2.91, night_lag 0.44, night_nolag 13.41 |
| xgboost | mondrian | 0.8 | 1.556 | day_nolag 1.59, day_lag 1.56, night_lag 0.29, night_nolag 11.74 |
| persistence | global | 0.9 | 8.195 | - |
| persistence | global | 0.8 | 4.699 | - |
| persistence | mondrian | 0.9 | 8.195 | day_nolag 6.59, day_lag 8.25, night_lag 0.50, night_nolag 0.00 |
| persistence | mondrian | 0.8 | 4.699 | day_nolag 3.25, day_lag 4.75, night_lag 0.31, night_nolag 0.00 |

## TEST coverage — ALL rows

Nominal: level 0.9 → ≥0.90, level 0.8 → ≥0.80 (marginal, exchangeability assumed).

| model | method | level | coverage | MAE | mean width | median width | p90 width | n |
|---|---|---|---:|---:|---:|---:|---:|---:|
| xgboost | global | 0.9 | 0.916 | 1.056 | 5.843 | 5.843 | 5.843 | 191,732 |
| xgboost | global | 0.8 | 0.828 | 1.056 | 3.112 | 3.112 | 3.112 | 191,732 |
| xgboost | mondrian | 0.9 | 0.916 | 1.056 | 5.860 | 5.829 | 5.829 | 191,732 |
| xgboost | mondrian | 0.8 | 0.828 | 1.056 | 3.135 | 3.111 | 3.111 | 191,732 |
| persistence | global | 0.9 | 0.922 | 2.783 | 16.391 | 16.391 | 16.391 | 182,029 |
| persistence | global | 0.8 | 0.841 | 2.783 | 9.398 | 9.398 | 9.398 | 182,029 |
| persistence | mondrian | 0.9 | 0.923 | 2.783 | 16.384 | 16.500 | 16.500 | 182,029 |
| persistence | mondrian | 0.8 | 0.842 | 2.783 | 9.400 | 9.500 | 9.500 | 182,029 |

## TEST coverage — regimes (mondrian, level 0.9)

| model | regime | n | coverage | mean width |
|---|---|---:|---:|---:|
| xgboost | day_lag | 183,789 | 0.916 | 5.829 |
| xgboost | day_nolag | 7,464 | 0.930 | 6.152 |
| xgboost | night_lag | 248 | 0.948 | 0.875 |
| xgboost | night_nolag | 231 | 0.944 | 26.822 |
| persistence | day_lag | 176,263 | 0.922 | 16.500 |
| persistence | day_nolag | 5,601 | 0.938 | 13.188 |
| persistence | night_lag | 154 | 0.935 | 1.000 |
| persistence | night_nolag | 11 | 0.909 | 0.000 |

## Per-site coverage spread (mondrian, level 0.9)

| model | sites | coverage min | coverage max | width min | width max | below-nominal sites |
|---|---:|---:|---:|---:|---:|---:|
| xgboost | 42 | 0.537 | 1.000 | 5.768 | 6.486 | 9 |
| persistence | 42 | 0.551 | 1.000 | 15.685 | 16.421 | 8 |
