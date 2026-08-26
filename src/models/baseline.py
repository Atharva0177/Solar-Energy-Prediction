"""Naive baseline forecasters (PRD §21).

* ``ZeroBaseline`` — sanity floor.
* ``MeanBaseline`` — historical mean, global or per-site (D-010: site scale
  dominates this dataset, so the per-site variant is the meaningful one).
* ``PersistenceBaseline`` — same-time-previous-day (primary baseline).

All fit statistics come strictly from the training slice; evaluation-period
targets are never read at fit time. Persistence maps t → t−24 h exactly; when
the prior-day observation is missing the prediction is NaN — never zero
(missing ≠ zero, D-008).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# NOTE: pd.Timedelta(days=1) (keyword form) trips a numpy>=2.5 generic-unit
# deprecation inside pandas 2.3; positional value + unit does not.
LAG = pd.Timedelta(86400, unit="s")


class ZeroBaseline:
    """Predicts 0 for every interval."""

    def fit(self, _df: Optional[pd.DataFrame] = None) -> "ZeroBaseline":
        return self

    def predict(self, eval_df: pd.DataFrame) -> pd.Series:
        return pd.Series(0.0, index=eval_df.index, name="prediction")


class MeanBaseline:
    """Predicts the training-period mean power.

    ``scope='site'`` uses each site's own mean with fallback to the global
    mean for sites unseen in training; ``scope='global'`` uses one pooled
    mean.
    """

    def __init__(self, scope: str = "global"):
        if scope not in ("global", "site"):
            raise ValueError(f"scope must be 'global' or 'site', got {scope!r}")
        self.scope = scope
        self.global_mean_: Optional[float] = None
        self.site_means_: dict = {}

    def fit(self, train_df: pd.DataFrame) -> "MeanBaseline":
        pw = train_df["power"].dropna()
        self.global_mean_ = float(pw.mean())
        if self.scope == "site":
            self.site_means_ = {
                sid: float(g.dropna().mean())
                for sid, g in train_df.groupby("site_id", observed=True)["power"]
            }
        return self

    def predict(self, eval_df: pd.DataFrame) -> pd.Series:
        if self.global_mean_ is None:
            raise RuntimeError("fit() before predict()")
        if self.scope == "global":
            return pd.Series(self.global_mean_, index=eval_df.index, name="prediction")
        # site_id may be Categorical (parquet partition col) — materialize raw
        # values first so the mapped result is a plain float Series.
        sids = pd.Series(eval_df["site_id"].to_numpy(dtype=object), index=eval_df.index)
        pred = sids.map(self.site_means_).astype("float64")
        return pred.fillna(self.global_mean_).rename("prediction")


class PersistenceBaseline:
    """P(t) ≈ P(t − 24h), same site and time-of-day (PRD §21, primary)."""

    def __init__(self):
        # MultiIndex (site_id, timestamp+24h) -> observed power from history.
        self._lagged: Optional[pd.Series] = None

    def fit(self, history_df: pd.DataFrame) -> "PersistenceBaseline":
        obs = history_df.loc[history_df["power"].notna(), ["site_id", "timestamp", "power"]]
        # site_id may be Categorical (parquet partition col) — materialize raw
        # values or MultiIndex lookups silently miss across category dtypes.
        idx = pd.MultiIndex.from_arrays(
            [obs["site_id"].to_numpy(dtype=object),
             obs["timestamp"] + LAG],
            names=["site_id", "timestamp"],
        )
        self._lagged = pd.Series(obs["power"].to_numpy(dtype=float), index=idx)
        return self

    def predict(self, eval_df: pd.DataFrame) -> pd.Series:
        if self._lagged is None:
            raise RuntimeError("fit() before predict()")
        keys = pd.MultiIndex.from_arrays(
            [eval_df["site_id"].to_numpy(dtype=object), eval_df["timestamp"]],
            names=self._lagged.index.names,
        )
        mapped = np.asarray(keys.map(self._lagged), dtype=float)
        return pd.Series(mapped, index=eval_df.index, name="prediction")
