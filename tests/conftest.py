"""Shared synthetic-data builders for train-page hardening tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

TS_FMT = "%Y-%m-%d %H:%M:%S"


def make_unisolar_folder(root, days=10, sites=(1, 2), campus=7, seed=0):
    """Write the three UNISOLAR CSVs under root/raw with a learnable
    daylight signal. Returns the raw dir. Schema matches
    src/data/schema_map.py exactly."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2022-01-01", periods=days * 96, freq="15min")
    hour = ts.hour + ts.minute / 60
    elev = np.clip(np.sin((hour - 6) / 24 * 2 * np.pi) * 60, 0, None)

    wx = pd.DataFrame({
        "CampusKey": campus,
        "Timestamp": ts.strftime(TS_FMT),
        "AirTemperature": 18 + 6 * np.sin((hour - 9) / 24 * 2 * np.pi)
                          + rng.normal(0, 0.5, len(ts)),
        "ApparentTemperature": 17 + 6 * np.sin((hour - 9) / 24 * 2 * np.pi),
        "DewPointTemperature": 12.0,
        "RelativeHumidity": 60 - 10 * np.sin((hour - 9) / 24 * 2 * np.pi),
        "WindSpeed": 3 + rng.normal(0, 0.5, len(ts)),
        "WindDirection": 180 + rng.normal(0, 30, len(ts)),
    })

    gen_frames, site_rows = [], []
    for sid in sites:
        # Night rows report NaN like the real feed (D-008: missing ≠ zero);
        # daylight power is a clean site-scaled elevation signal clipped at 0
        # so cleaning never nulls a boundary slot (keeps lookups complete).
        daylight = elev > 0
        power = np.where(
            daylight,
            np.maximum(elev * (0.15 + 0.05 * sid) + rng.normal(0, 0.02, len(ts)), 0.0),
            np.nan)
        power = np.round(power, 4)
        gen_frames.append(pd.DataFrame({
            "SiteKey": sid, "CampusKey": campus,
            "Timestamp": ts.strftime(TS_FMT),
            "SolarGeneration": power.round(4),
        }))
        site_rows.append({"SiteKey": sid, "CampusKey": campus,
                          "kWp": 10 * sid, "lat": -36.1, "Lon": 146.8})

    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    pd.concat(gen_frames, ignore_index=True).to_csv(
        raw / "Solar_Energy_Generation.csv", index=False)
    wx.to_csv(raw / "Weather_Data_reordered_all.csv", index=False)
    pd.DataFrame(site_rows).to_csv(raw / "Solar_Site_Details.csv", index=False)
    return raw
