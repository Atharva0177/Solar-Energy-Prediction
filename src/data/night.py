"""Night-period identification via solar position (PRD §10).

No irradiance exists in UNISOLAR (D-006), so daylight comes from solar
elevation computed with pvlib at campus coordinates (lat/lon are
campus-level, 5 unique pairs).

Timestamps in the raw data carry no timezone. Two candidates exist for
Victoria/Australia data: fixed standard time ``Etc/GMT-10`` (UTC+10) or civil
``Australia/Melbourne`` (UTC+10/+11 with DST).
:func:`choose_timezone` picks empirically by comparing how well each
candidate separates daytime generation from nighttime generation.
"""

from __future__ import annotations

import pvlib
import pandas as pd

DAYLIGHT_ELEVATION_DEG = 0.0

TZ_CANDIDATES = ["Etc/GMT-10", "Australia/Melbourne"]


def campus_coordinates(site_details: pd.DataFrame) -> pd.DataFrame:
    """One row per campus with its representative lat/lon."""
    coords = (
        site_details.groupby("campus_id", observed=True)
        .agg(latitude=("latitude", "median"), longitude=("longitude", "median"))
        .reset_index()
    )
    return coords


def _elevation_for_campus(
    ts_unique: pd.DatetimeIndex, lat: float, lon: float, tz: str
) -> pd.Series:
    """Apparent elevation for one site coordinate over given naive-local times."""
    idx = ts_unique.tz_localize(
        tz, nonexistent="shift_forward", ambiguous="NaT"
    ).dropna()
    sp = pvlib.solarposition.get_solarposition(idx, lat, lon)
    elev = sp["apparent_elevation"]
    elev.index = elev.index.tz_localize(None)
    return elev


def add_solar_position(
    df: pd.DataFrame,
    coords: pd.DataFrame,
    tz: str,
    elevation_col: str = "solar_elevation_deg",
    daylight_col: str = "is_daylight",
) -> pd.DataFrame:
    """Attach solar elevation + daylight flag per row using campus coordinates."""
    parts = []
    for _, row in coords.iterrows():
        sub = df[df["campus_id"] == row["campus_id"]]
        if sub.empty:
            continue
        ts_unique = pd.DatetimeIndex(sub["timestamp"].dropna().unique())
        elev = _elevation_for_campus(ts_unique, row["latitude"], row["longitude"], tz)
        mapping = pd.DataFrame(
            {
                "timestamp": elev.index,
                elevation_col: elev.to_numpy(),
            }
        )
        sub = sub.merge(mapping, on="timestamp", how="left")
        sub[daylight_col] = sub[elevation_col] > DAYLIGHT_ELEVATION_DEG
        parts.append(sub)
    out = pd.concat(parts, ignore_index=True)
    # Rows from campuses without coordinates keep NaN elevation / False flag.
    missing = df[~df["campus_id"].isin(coords["campus_id"])].copy()
    if not missing.empty:
        missing[elevation_col] = float("nan")
        missing[daylight_col] = False
        out = pd.concat([out, missing], ignore_index=True)
    return out.sort_values(["site_id", "timestamp"]).reset_index(drop=True)


def choose_timezone(
    gen_df: pd.DataFrame, coords: pd.DataFrame, sample_every: int = 3
) -> dict:
    """Score TZ candidates by day/night generation contrast.

    Returns the chosen timezone plus the evidence numbers for the decision.
    """
    sampled = gen_df.iloc[::sample_every]
    results = {}
    for tz in TZ_CANDIDATES:
        tagged = add_solar_position(sampled, coords, tz)
        day = tagged.loc[tagged["is_daylight"], "power"]
        night = tagged.loc[~tagged["is_daylight"], "power"]
        night_nonzero_frac = float((night.fillna(0) > 0).mean())
        day_mean = float(day.mean()) if len(day) else 0.0
        night_mean_excl_na = float(night.dropna().mean()) if night.notna().any() else 0.0
        contrast = day_mean / night_mean_excl_na if night_mean_excl_na > 0 else float("inf")
        results[tz] = {
            "day_mean_power": round(day_mean, 4),
            "night_mean_power_observed": round(night_mean_excl_na, 4),
            "night_nonzero_fraction": round(night_nonzero_frac, 4),
            "contrast_ratio": None if contrast == float("inf") else round(contrast, 2),
            "ambiguous_nat_rows": int(tagged["solar_elevation_deg"].isna().sum()),
        }
    def score(r):
        # Lower night nonzero fraction is the primary signal; NaT rows tiebreak.
        return (r["night_nonzero_fraction"], r["ambiguous_nat_rows"])

    best = min(results, key=lambda tz: score(results[tz]))
    return {"chosen_timezone": best, "candidates": results}
