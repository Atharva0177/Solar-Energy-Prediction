"""Phase 5 orchestrator: processed parquet -> engineered feature table.

Reads ``data/processed/solar``, applies the PRD §15-19 feature families and
writes:

* ``data/processed/features/`` — full feature table, partitioned
  ``site_id/year/month`` like its source (PRD §41 pattern)
* ``artifacts/features/feature_metadata.json`` — the Phase 5 deliverable:
  families, columns, configs, missingness, leakage notes

Leakage posture: lags read strictly older timestamps by calendar time; rolling
windows are closed-left (``[t-W, t)``); temporal features are pure functions
of the timestamp. Target ``power`` is carried through untouched (never imputed,
D-008).
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.features import lag as lag_mod
from src.features import rolling as roll_mod
from src.features import solar as solar_mod
from src.features import temporal as temp_mod
from src.features import weather as wx_mod

SOURCE_DIR = REPO_ROOT / "data" / "processed" / "solar"
FEATURES_DIR = REPO_ROOT / "data" / "processed" / "features"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "features"

CADENCE_S = 900  # dataset is 15-min (D-006)
LAG_STEPS = [1, 2, 4, 8, 24, 48, 96]  # PRD §16 examples, in 15-min steps
ROLLING_WINDOWS_S = [3600, 21600, 86400]  # 1 h / 6 h / 24 h
ROLLING_STATS = ["mean", "std", "min", "max"]
TIMEZONE = "Australia/Melbourne"  # D-007


def main() -> int:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    df = pd.read_parquet(SOURCE_DIR)
    base_cols = list(df.columns)
    print(f"loaded {df.shape[0]:,} rows x {df.shape[1]} cols from {SOURCE_DIR.name}")

    # ---- temporal (§15) -----------------------------------------------------
    df = temp_mod.add_temporal_features(df)
    print("temporal features added")

    # ---- lags (§16) — calendar-exact ---------------------------------------
    lag_specs = {f"power_lag_{s}": pd.Timedelta(s * CADENCE_S, unit="s") for s in LAG_STEPS}
    df = lag_mod.add_lags(df, lag_specs)
    print(f"lag features added: {list(lag_specs)}")

    # ---- rolling (§17) — history only --------------------------------------
    windows = [pd.Timedelta(w, unit="s") for w in ROLLING_WINDOWS_S]
    df = roll_mod.add_rolling_features(
        df, windows=windows, stats=ROLLING_STATS, min_periods=1
    )
    print("rolling features added")

    # ---- weather (§19) — dynamic + circular wind ---------------------------
    passthrough = wx_mod.available_weather_columns(df)
    df = wx_mod.add_weather_features(df)
    print(f"weather passthrough: {passthrough}")

    # ---- solar position (§18) — pvlib at campus coords ---------------------
    sites = pd.read_parquet(REPO_ROOT / "data" / "processed" / "site_details.parquet")
    coords = sites.groupby("campus_id", observed=True).agg(
        latitude=("latitude", "median"), longitude=("longitude", "median")
    ).reset_index()
    # recomputed here so elevation/azimuth/zenith come from one consistent run
    df = df.drop(columns=[c for c in ("solar_elevation_deg", "is_daylight") if c in df.columns])
    df = solar_mod.add_solar_position_features(df, coords, TIMEZONE)
    print("solar position features added")

    new_cols = [c for c in df.columns if c not in base_cols]

    # ---- persist partitioned -------------------------------------------------
    out = df.copy()
    out["year"] = out["timestamp"].dt.year
    out["month"] = out["timestamp"].dt.month
    if FEATURES_DIR.exists():
        shutil.rmtree(FEATURES_DIR)
    out.to_parquet(FEATURES_DIR, engine="pyarrow",
                   partition_cols=["site_id", "year", "month"], index=False)
    n_parts = sum(1 for _ in FEATURES_DIR.rglob("*.parquet"))
    print(f"wrote {FEATURES_DIR} ({out.shape[0]:,} rows x {out.shape[1]} cols, {n_parts} partitions)")

    # ---- metadata deliverable -------------------------------------------------
    engineered = [c for c in new_cols if c not in ("year", "month")]
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "scripts/build_features.py",
        "source": str(SOURCE_DIR.relative_to(REPO_ROOT)),
        "output": str(FEATURES_DIR.relative_to(REPO_ROOT)),
        "n_rows": int(out.shape[0]),
        "n_columns_total": int(out.shape[1]),
        "cadence_minutes": CADENCE_S // 60,
        "timezone": TIMEZONE,
        "target": {"column": "power", "imputed": False},
        "engineered_columns": engineered,
        "n_engineered_columns": len(engineered),
        "feature_families": [
            {
                "family": "temporal",
                "prd_section": 15,
                "columns": [
                    "hour", "minute", "day", "day_of_week", "day_of_year",
                    "week_of_year", "month", "quarter", "season", "is_weekend",
                    "sin_hour", "cos_hour", "sin_day_of_year", "cos_day_of_year",
                ],
                "config": {
                    "season_convention": "southern_hemisphere_meteorological",
                    "cyclical": ["sin_hour", "cos_hour", "sin_day_of_year", "cos_day_of_year"],
                },
            },
            {
                "family": "lags",
                "prd_section": 16,
                "columns": list(lag_specs),
                "config": {
                    "lag_steps_15min": LAG_STEPS,
                    "alignment": "calendar_exact_by_timestamp_not_positional",
                    "missing_prior_observation": "nan_never_zero",
                },
            },
            {
                "family": "rolling",
                "prd_section": 17,
                "columns": [
                    f"power_rolling_{stat}_{w}s"
                    for w in ROLLING_WINDOWS_S for stat in ROLLING_STATS
                ],
                "config": {
                    "windows_seconds": ROLLING_WINDOWS_S,
                    "stats": ROLLING_STATS,
                    "closed": "left_interval_t_minus_W_to_t",
                    "min_periods": 1,
                    "per_site": True,
                },
            },
            {
                "family": "weather",
                "prd_section": 19,
                "passthrough_columns": passthrough,
                "derived_columns": [c for c in engineered if c.startswith("wind_dir_")],
                "config": {"dynamic_to_available_variables": True},
            },
            {
                "family": "solar_position",
                "prd_section": 18,
                "columns": ["solar_elevation_deg", "azimuth_deg", "zenith_deg",
                            "day_length_hours", "is_daylight"],
                "config": {
                    "library": "pvlib apparent solar position",
                    "coordinates_grain": "campus (5 pairs for 42 sites)",
                    "day_length_definition": "count of >0-degree-elevation 15-min grid slots per civil date * 0.25h",
                },
            },
        ],
        "missing_pct_engineered": {
            c: round(float(df[c].isna().mean() * 100), 2) for c in engineered
        },
        "leakage_notes": [
            "lags read only strictly older timestamps (calendar-exact, per site)",
            "rolling windows closed='left' -> interval [t-W, t), current obs excluded",
            "temporal features are pure functions of the timestamp",
            "power target untouched; NaN never imputed (D-008)",
        ],
    }
    meta_path = ARTIFACTS_DIR / "feature_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {meta_path}")

    elapsed = time.perf_counter() - t_start
    print(f"done in {elapsed:.1f}s | engineered columns: {len(engineered)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
