"""Rolling window features (PRD §17) — strictly historical.

Windows are time-based and evaluated with ``closed='left'``: the observation
at t is EXCLUDED from its own rolling statistics, so no future (or current)
value can enter a feature. Computed per site; rows keep their input order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _window_suffix(window: pd.Timedelta) -> str:
    return f"{int(window.total_seconds())}s"


def add_rolling_features(
    df: pd.DataFrame,
    windows: list,
    stats: list,
    min_periods: int = 1,
    group_col: str = "site_id",
    ts_col: str = "timestamp",
    value_col: str = "power",
) -> pd.DataFrame:
    """Add ``{value_col}_rolling_{stat}_{suffix}`` per window × stat.

    ``suffix`` renders the window in whole seconds (e.g. ``3600s``). Rows with
    NaN truth contribute nothing (never zero-filled, D-008).
    """
    out = df.copy()
    for _, labels in df.groupby(group_col, observed=True).groups.items():
        sub = df.loc[labels].sort_values(ts_col)
        series = pd.Series(
            sub[value_col].to_numpy(dtype=float), index=pd.DatetimeIndex(sub[ts_col])
        )
        for window in windows:
            roller = series.rolling(window=window, closed="left", min_periods=min_periods)
            suffix = _window_suffix(window)
            for stat in stats:
                name = f"{value_col}_rolling_{stat}_{suffix}"
                out.loc[sub.index, name] = getattr(roller, stat)().to_numpy()
    return out
