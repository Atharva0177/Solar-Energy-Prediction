"""Weather features (PRD §19) — dynamically adapt to available variables.

UNISOLAR has no irradiance/pressure/cloud cover (D-006); whatever weather
columns exist are passed through untouched by this module. The one derived
encoding here: wind direction is circular, so raw degrees become sin/cos
components — a model must not see 359° vs 1° as far apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KNOWN_WEATHER_VARS = [
    "temperature",
    "apparent_temperature",
    "dew_point_temperature",
    "humidity",
    "wind_speed",
    "wind_direction",
    "pressure",
    "cloud_cover",
]


def available_weather_columns(df: pd.DataFrame) -> list:
    """Subset of known weather variables present in ``df``, in canonical order."""
    return [c for c in KNOWN_WEATHER_VARS if c in df.columns]


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Copy ``df`` adding circular wind-direction encodings when present."""
    out = df.copy()
    if "wind_direction" in out.columns:
        rad = np.deg2rad(out["wind_direction"].to_numpy(dtype=float))
        out["wind_dir_sin"] = np.sin(rad)
        out["wind_dir_cos"] = np.cos(rad)
    return out
