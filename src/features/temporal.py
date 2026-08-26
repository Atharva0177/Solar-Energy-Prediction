"""Temporal calendar features (PRD §15).

Pure functions of the timestamp — no leakage surface. Cyclical encodings use
the civil-local hour (data is Australia/Melbourne civil time, D-007) and
day-of-year; seasons follow SOUTHERN-hemisphere meteorological convention
(this dataset is Victoria, AU).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_SEASONS = {
    "summer": (12, 1, 2),
    "autumn": (3, 4, 5),
    "winter": (6, 7, 8),
    "spring": (9, 10, 11),
}
# {(12,1,2): "summer", ...} — tuple-keyed lookup used by tests + mapping below
SEASON_BY_MONTH = {months: name for name, months in _SEASONS.items()}
_MONTH_TO_SEASON = {m: s for months, s in SEASON_BY_MONTH.items() for m in months}


def add_temporal_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """Return a copy of ``df`` with calendar + cyclical feature columns."""
    out = df.copy()
    ts = out[ts_col]
    hour, minute = ts.dt.hour, ts.dt.minute

    out["hour"] = hour
    out["minute"] = minute
    out["day"] = ts.dt.day
    out["day_of_week"] = ts.dt.dayofweek  # Monday=0
    out["day_of_year"] = ts.dt.dayofyear
    out["week_of_year"] = ts.dt.isocalendar().week.astype("int64")
    out["month"] = ts.dt.month
    out["quarter"] = ts.dt.quarter
    out["season"] = ts.dt.month.map(_MONTH_TO_SEASON)
    out["is_weekend"] = out["day_of_week"].ge(5)

    # cyclical encodings — continuous at wrap-around boundaries
    day_frac = (hour * 60 + minute) / (24 * 60)
    angle = 2 * np.pi * day_frac
    out["sin_hour"] = np.sin(angle)
    out["cos_hour"] = np.cos(angle)

    doy_angle = 2 * np.pi * ts.dt.dayofyear / 365.25
    out["sin_day_of_year"] = np.sin(doy_angle)
    out["cos_day_of_year"] = np.cos(doy_angle)
    return out
