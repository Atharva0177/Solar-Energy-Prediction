"""SHAP explanations for the XGBoost forecaster (PRD §29).

Answers "why did the model predict this power value?" on three levels:

* **Global importance** — mean |SHAP| per feature over an evaluation sample
  (``global_importance``).
* **Local explanations** — per-prediction contribution tables
  (``contribution_table``) for tagged scenarios (clear-noon peak, night,
  morning ramp, overcast afternoon) picked by ``select_local_examples``.
* **Contribution plots** — waterfall/beeswarm/dependence rendering lives in
  ``scripts/run_shap.py``; this module supplies the numbers.

The explained model is the Phase 6 XGBoost run (``xgboost-site-all-h1-v1``):
exact ``TreeExplainer`` values, additivity verified to ~1e-6 including the
categorical ``site_id`` splits. Sequence models are not explained — SHAP has
no exact fast path through the shared window pipeline, and PRD §27's SHAP
item targets tabular feature importance.

All sampling is seeded: same inputs → identical artifacts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.xgboost_model import load_xgboost_model, prepare_matrix  # noqa: F401


def explain_matrix(reg, df: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Exact TreeExplainer values for ``df`` → (shap_values, base_value)."""
    import shap

    X = prepare_matrix(df, reg.feature_cols_)
    explainer = shap.TreeExplainer(reg)
    sv = np.asarray(explainer.shap_values(X))
    return sv, float(np.asarray(explainer.expected_value).ravel()[0])


def global_importance(shap_values: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Per-feature mean |SHAP| / signed mean / max |SHAP|, ranked descending."""
    imp = pd.DataFrame({
        "feature": names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        "mean_shap": shap_values.mean(axis=0),
        "max_abs_shap": np.abs(shap_values).max(axis=0),
    })
    imp = imp.sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    imp.insert(0, "rank", np.arange(1, len(imp) + 1))
    imp["share_of_total"] = imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()
    return imp


def sample_rows(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Deterministic subsample of observed-target rows (for global SHAP)."""
    obs = df.loc[df["power"].notna()]
    n = min(n, len(obs))
    idx = np.random.default_rng(seed).choice(len(obs), size=n, replace=False)
    return obs.iloc[np.sort(idx)]


def select_local_examples(frame: pd.DataFrame) -> dict[str, int]:
    """One representative row index per scenario tag, deterministic.

    Tags (PRD §29 example mixes positive and negative contributors):

    * ``night_zero``        — after dark; tests that contributions cancel.
    * ``clear_noon_peak``   — high sun, highest output in candidates.
    * ``morning_ramp``      — sun low but rising, first-light behavior.
    * ``overcast_afternoon``— humid daylight hours, depressed output.

    Representative = row whose target is closest to the candidate median
    (peak uses the max); ties broken by earliest timestamp.
    """
    hour = frame["timestamp"].dt.hour
    dayl = frame["is_daylight"].astype(bool)

    def pick(mask: pd.Series, how: str = "median") -> int:
        cand = frame.loc[mask]
        cand = cand.loc[cand["power"].notna()]
        if cand.empty:
            raise ValueError("no rows satisfy scenario conditions")
        if how == "median":
            target = cand["power"].median()
            key = (cand["power"] - target).abs()
            best = key[key == key.min()].index
        else:  # max
            best = cand.index[cand["power"] == cand["power"].max()]
        # earliest timestamp among tied rows → stable across runs
        ts = frame.loc[best, "timestamp"]
        return int(ts.idxmin())

    elevation = frame["solar_elevation_deg"]
    humidity = frame.get("humidity")
    out = {
        "night_zero": pick(~dayl),
        "clear_noon_peak": pick(
            dayl & (elevation > 45) & hour.between(11, 14), how="max"),
        "morning_ramp": pick(
            dayl & hour.between(6, 9) & elevation.between(5, 25)),
    }
    if humidity is not None:
        out["overcast_afternoon"] = pick(
            dayl & (humidity > 80) & (hour >= 13))
    return out


def contribution_table(
    row_X: pd.Series, shap_row: np.ndarray, base_value: float,
    top_k: int = 10,
) -> pd.DataFrame:
    """Signed contributions for one prediction, |SHAP|-descending top-k.

    Rows carry feature value + SHAP contribution; sign convention is the
    model's (+ raises the prediction, − lowers it).
    """
    t = pd.DataFrame({
        "feature": row_X.index,
        "value": row_X.to_numpy(dtype=float),
        "shap": shap_row,
    })
    t["direction"] = np.where(t["shap"] > 0, "+", "-")
    t = t.reindex(t["shap"].abs().sort_values(ascending=False).index)
    head = t.head(top_k).copy()
    parts = [head]
    if len(t) > top_k:
        tail_sum = float(t["shap"].iloc[top_k:].sum())
        parts.append(pd.DataFrame([{
            "feature": f"<remaining {len(t) - top_k} features>",
            "value": np.nan, "shap": tail_sum, "direction":
            "+" if tail_sum > 0 else "-",
        }]))
    out = pd.concat(parts, ignore_index=True)
    out.insert(0, "base_value", round(base_value, 4))
    return out.reset_index(drop=True)
