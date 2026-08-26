"""Regression metrics for forecast evaluation (PRD §25).

Reports both all-period and daylight-only numbers (PRD §10: night zeros must
not flatter a model). ``nRMSE`` needs an explicitly documented denominator —
callers pass it (train-period observed range or site capacity); this module
never invents one.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def regression_metrics(
    y_true,
    y_pred,
    daylight: Optional[np.ndarray] = None,
    denom: Optional[float] = None,
) -> dict:
    """MAE / RMSE / R² / nRMSE (+ daylight variants) between truth and preds.

    Rows where either side is NaN are counted in ``n_missing`` and excluded
    from every statistic. ``denom`` is the documented nRMSE denominator.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: {y.shape} vs {p.shape}")

    valid = ~(np.isnan(y) | np.isnan(p))
    yv, pv = y[valid], p[valid]
    out = {
        "n_eval": int(valid.sum()),
        "n_missing": int((~valid).sum()),
        "mae": np.nan,
        "rmse": np.nan,
        "r2": np.nan,
        "nrmse": np.nan,
        "daylight_n": 0,
        "daylight_mae": np.nan,
        "daylight_nrmse": np.nan,
    }
    if out["n_eval"] == 0:
        return out

    err = pv - yv
    out["mae"] = float(np.mean(np.abs(err)))
    out["rmse"] = float(np.sqrt(np.mean(err**2)))
    sst = float(np.sum((yv - yv.mean()) ** 2))
    if sst > 0:
        out["r2"] = float(1.0 - np.sum(err**2) / sst)
    if denom:
        out["nrmse"] = out["rmse"] / float(denom)

    if daylight is not None:
        # full-length daylight mask -> align to the valid-filtered arrays
        day = (np.asarray(daylight, dtype=bool) & valid)[valid]
        out["daylight_n"] = int(day.sum())
        if out["daylight_n"]:
            derr = pv[day] - yv[day]
            out["daylight_mae"] = float(np.mean(np.abs(derr)))
            if denom:
                out["daylight_nrmse"] = float(np.sqrt(np.mean(derr**2)) / float(denom))
    return out


def metrics_frame(
    df: pd.DataFrame,
    pred_col: str,
    truth_col: str = "power",
    daylight_col: str = "is_daylight",
    denom: Optional[float] = None,
) -> dict:
    """Convenience wrapper computing metrics over a DataFrame."""
    return regression_metrics(
        df[truth_col],
        df[pred_col],
        daylight=df[daylight_col].to_numpy() if daylight_col in df else None,
        denom=denom,
    )
