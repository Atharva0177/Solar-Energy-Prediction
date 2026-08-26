"""Lag features (PRD §16) — calendar-exact, never forward-looking.

A lag of Δt at row t reads the value observed at t−Δt **by timestamp**, not by
row position. Positional shifts silently misalign when grid slots are missing
(52,492 such slots exist in this dataset). Missing prior observations stay
NaN (D-008), and a lag can only ever read strictly older rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_lags(
    df: pd.DataFrame,
    lag_specs: dict,
    group_col: str = "site_id",
    ts_col: str = "timestamp",
    value_col: str = "power",
) -> pd.DataFrame:
    """Add one column per ``{name: Timedelta}`` in ``lag_specs``.

    Column ``name`` at row t = ``value_col`` observed at t − Timedelta within
    the same site; NaN when that observation is absent.
    """
    out = df.copy()
    keys_base = pd.MultiIndex.from_arrays(
        [out[group_col].to_numpy(dtype=object), pd.DatetimeIndex(out[ts_col])]
    )
    values = pd.Series(
        out[value_col].to_numpy(dtype=float), index=keys_base
    )
    for name, delta in lag_specs.items():
        lagged_keys = pd.MultiIndex.from_arrays(
            [out[group_col].to_numpy(dtype=object), pd.DatetimeIndex(out[ts_col]) - delta]
        )
        out[name] = pd.Series(
            np.asarray(lagged_keys.map(values), dtype=float), index=out.index
        )
    return out
