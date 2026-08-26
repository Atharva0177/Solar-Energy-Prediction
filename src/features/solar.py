"""Solar-position features (PRD §18) via pvlib at campus coordinates.

Extends the Phase-2 daylight tagging (``src/data/night.py``) with azimuth,
zenith and day length. Coordinates are campus-level — 5 unique pairs for 42
sites (D-006) — so positions are computed once per campus over unique
timestamps and merged back.

``day_length_hours`` counts 15-min grid slots with elevation > 0 on that
civil date at that campus × 0.25 h. Grid holes make it a slight under-count;
documented approximation, consistent across sites/dates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib

SLOT_HOURS = 0.25


def _solar_positions_for_campus(
    ts_unique: pd.DatetimeIndex, lat: float, lon: float, tz: str
) -> pd.DataFrame:
    """apparent elevation / azimuth / zenith for one coordinate, naive-local index."""
    idx = ts_unique.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT").dropna()
    sp = pvlib.solarposition.get_solarposition(idx, lat, lon)
    out = pd.DataFrame(
        {
            "timestamp": sp.index.tz_localize(None),
            "solar_elevation_deg": sp["apparent_elevation"].to_numpy(),
            "azimuth_deg": sp["azimuth"].to_numpy(),
            "zenith_deg": sp["apparent_zenith"].to_numpy(),
        }
    )
    return out


def add_solar_position_features(
    df: pd.DataFrame,
    coords: pd.DataFrame,
    tz: str,
    group_col: str = "campus_id",
    ts_col: str = "timestamp",
    daylight_col: str = "is_daylight",
) -> pd.DataFrame:
    """Return a copy of ``df`` with solar elevation/azimuth/zenith/day-length.

    Rows whose campus has no coordinates keep NaN positions (day length NaN).
    """
    parts = []
    coord_rows = {r["campus_id"]: r for _, r in coords.iterrows()}
    for cid, sub in df.groupby(group_col, observed=True):
        row = coord_rows.get(cid)
        if row is None:
            continue
        ts_unique = pd.DatetimeIndex(sub[ts_col].dropna().unique())
        pos = _solar_positions_for_campus(ts_unique, row["latitude"], row["longitude"], tz)

        # day length per civil date from this campus's own grid
        pos["date"] = pos["timestamp"].dt.normalize()
        lit = pos[pos["solar_elevation_deg"] > 0].groupby("date").size()
        day_len = lit.mul(SLOT_HOURS).rename("day_length_hours")
        pos = pos.merge(day_len, left_on="date", right_index=True, how="left")
        pos = pos.drop(columns="date")

        merged = sub.merge(pos, on=ts_col, how="left")
        merged[daylight_col] = merged["solar_elevation_deg"] > 0
        parts.append(merged)

    missing = df[~df[group_col].isin(coords[group_col])].copy() if len(parts) else df.iloc[:0]
    if not missing.empty:
        for col in ("solar_elevation_deg", "azimuth_deg", "zenith_deg", "day_length_hours"):
            missing[col] = float("nan")
        missing[daylight_col] = False
        parts.append(missing)

    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["site_id", ts_col]).reset_index(drop=True) if "site_id" in out else out
