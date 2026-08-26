"""Split-conformal prediction intervals (PRD §28, preferred approach).

Model-agnostic marginal intervals from absolute residuals:

1. score calibration rows ``s_i = |y_i − ŷ_i|`` on a held-out calibration
   frame (never the training frame — train residuals are optimistically
   small and would undercover);
2. take the finite-sample-corrected quantile
   ``q = ceil((n+1)(1−α))/n`` of the scores;
3. predict ``[ŷ − q, ŷ + q]`` — under exchangeability
   ``P(y ∈ interval) ≥ 1 − α`` (Vovk et al.; Papadopoulos et al.).

``mondrian`` mode calibrates one ``q`` per group label (e.g. daylight ×
lag-availability, Phase 11) so intervals adapt to heteroscedastic regimes
while keeping the per-group guarantee.

Caveat recorded in D-018: the XGBoost calibration frame (VAL) also fed
early stopping (one scalar, ``best_iteration``), so its residuals are
mildly optimistic; empirical TEST coverage is the honest check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected conformal quantile of |residual| scores.

    ``k = ceil((n+1)(1−α))`` must lie in ``[1, n]``; for ``α`` so small that
    ``k > n`` the interval is infinite (no valid guarantee) — we return the
    max score and the caller reports the degenerate level.
    """
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    if n == 0:
        raise ValueError("no finite calibration scores")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float(scores.max())
    return float(np.sort(scores)[k - 1])


def fit_conformal(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float,
    groups: Optional[np.ndarray] = None,
) -> dict:
    """Calibrate global (and optional per-group) radii.

    Returns ``{"alpha", "global": q, "groups": {label: q}, "n": {...}}``.
    Rows with NaN predictions or targets are dropped before scoring.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    scores = np.abs(y_true[ok] - y_pred[ok])
    out = {"alpha": float(alpha), "global": conformal_quantile(scores, alpha),
           "groups": {}, "n": {"global": int(len(scores))}}
    if groups is not None:
        g = np.asarray(groups)[ok]
        for label in pd.unique(g):
            m = g == label
            out["groups"][str(label)] = conformal_quantile(scores[m], alpha)
            out["n"][str(label)] = int(m.sum())
    return out


def interval_widths(
    y_pred: np.ndarray, calib: dict, groups: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Lower/upper bounds per prediction; Mondrian radii when groups given.

    Group labels unseen at calibration fall back to the global radius.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    if groups is None:
        q = np.full(len(y_pred), calib["global"])
    else:
        g = pd.Series(np.asarray(groups)).astype(str)
        q = g.map(calib["groups"]).astype(float).fillna(calib["global"]).to_numpy()
    return y_pred - q, y_pred + q


def coverage_metrics(
    y_true: np.ndarray, y_pred: np.ndarray,
    lower: np.ndarray, upper: np.ndarray,
) -> dict:
    """Empirical coverage + width stats on rows with an observed target."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(lower)
    y, p, lo, hi = y_true[ok], y_pred[ok], lower[ok], upper[ok]
    n = int(len(y))
    if n == 0:
        return {"n": 0}
    covered = (y >= lo) & (y <= hi)
    width = hi - lo
    return {
        "n": n,
        "coverage": float(covered.mean()),
        "mae": float(np.abs(y - p).mean()),
        "mean_width": float(width.mean()),
        "median_width": float(np.median(width)),
        "p90_width": float(np.quantile(width, 0.9)),
        "n_missing": int(len(y_true) - n),
    }
