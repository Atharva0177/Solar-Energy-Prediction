"""Schema mapping: source UNISOLAR columns -> canonical model (PRD §7).

Verified source schema lives in DECISIONS.md D-006 and
``artifacts/data_profile.json``. No column names are invented here.

Canonical naming:

* ``timestamp``      - naive local wall-clock datetime
* ``site_id``        - int, source ``SiteKey``
* ``campus_id``      - int, source ``CampusKey`` (weather is campus-grain)
* ``power``          - kWh generated per 15-min interval (source ``Metric`` = kWh)

Weather variables keep descriptive canonical names; irradiance does NOT exist
in this dataset (D-006).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = REPO_ROOT / "unisolar"

GENERATION_FILE = RAW_DIR / "Solar_Energy_Generation.csv"
WEATHER_FILE = RAW_DIR / "Weather_Data_reordered_all.csv"
SITE_DETAILS_FILE = RAW_DIR / "Solar_Site_Details.csv"

# Source -> canonical maps (documented contract, verified in D-006).
GENERATION_COLUMNS = {
    "SiteKey": "site_id",
    "CampusKey": "campus_id",
    "Timestamp": "timestamp",
    "SolarGeneration": "power",
}

WEATHER_COLUMNS = {
    "CampusKey": "campus_id",
    "Timestamp": "timestamp",
    "AirTemperature": "temperature",
    "ApparentTemperature": "apparent_temperature",
    "DewPointTemperature": "dew_point_temperature",
    "RelativeHumidity": "humidity",
    "WindSpeed": "wind_speed",
    "WindDirection": "wind_direction",
}

SITE_DETAIL_COLUMNS = {
    "SiteKey": "site_id",
    "CampusKey": "campus_id",
    "kWp": "capacity_kwp",
    "Number of panels": "n_panels",
    "Panel": "panel_model",
    "Inverter": "inverter",
    "Optimizers": "optimizers",
    "Metric": "metric_unit",
    "lat": "latitude",
    "Lon": "longitude",
}


def load_generation(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Raw generation rows mapped onto canonical names."""
    df = pd.read_csv(Path(raw_dir) / "Solar_Energy_Generation.csv")
    return df.rename(columns=GENERATION_COLUMNS)


def load_weather(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Raw campus-grain weather rows mapped onto canonical names."""
    df = pd.read_csv(Path(raw_dir) / "Weather_Data_reordered_all.csv")
    return df.rename(columns=WEATHER_COLUMNS)


def load_site_details(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Static per-site metadata mapped onto canonical names."""
    df = pd.read_csv(Path(raw_dir) / "Solar_Site_Details.csv")
    return df.rename(columns=SITE_DETAIL_COLUMNS)
