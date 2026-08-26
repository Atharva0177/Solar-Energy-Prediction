"""Unit tests for the post-PRD export bundle builders (D-025).

Each builder takes a root path; tests construct a miniature replica of the
real directory layout (partitioned processed parquet + tiny CSV artifacts).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _mini_root(tmp_path: Path) -> Path:
    """2 sites × 1 campus, 3 days of 15-min rows, partitioned parquet."""
    ts = pd.date_range("2022-01-01", periods=96 * 3, freq="15min")
    hour = ts.hour + ts.minute / 60
    sine = np.sin((hour - 6) / 24 * 2 * np.pi)
    frames = []
    for sid in (1, 2):
        power = np.clip(sine, 0, None) * (2 * sid)
        df = pd.DataFrame({
            "timestamp": ts, "site_id": sid, "campus_id": 7,
            "power": power,
            "is_daylight": sine > 0,
            "temperature": 20 - 8 * sine, "humidity": 55.0,
            "wind_speed": 3.0,
            "solar_elevation_deg": sine * 60,
        })
        df["year"], df["month"] = df["timestamp"].dt.year, df["timestamp"].dt.month
        frames.append(df)
    out = tmp_path / "data" / "processed" / "solar"
    out.mkdir(parents=True)
    big = pd.concat(frames, ignore_index=True)
    big.to_parquet(out, engine="pyarrow",
                   partition_cols=["site_id", "year", "month"], index=False)
    det = tmp_path / "data" / "processed"
    pd.DataFrame([{"site_id": 1, "campus_id": 7},
                  {"site_id": 2, "campus_id": 7}]).to_parquet(
        det / "site_details.parquet", index=False)
    return tmp_path


class TestEdaProfiles:
    def test_slots_campuses_and_correlation(self, tmp_path):
        from scripts.export_frontend_data import eda_profiles_bundle

        b = eda_profiles_bundle(_mini_root(tmp_path))
        assert len(b["hour_of_day"]["slots"]) == 96
        assert b["hour_of_day"]["slots"][36] == "09:00"
        mean_kw = b["hour_of_day"]["mean_kw"]
        assert set(mean_kw) == {"ALL", "7"}
        # site 1 peaks 2 kW, site 2 peaks 4 kW -> campus mean peak = 3 kW
        assert max(v for v in mean_kw["7"] if v is not None) == 3.0
        corr = b["correlation"]
        assert corr["campuses"] == [7]
        assert corr["vars"] == ["temperature", "humidity", "wind_speed",
                                "solar_elevation_deg"]
        # temperature is anti-phase diurnal (negative r); humidity and wind
        # are constant -> None; power tracks elevation. Pooling two sites with
        # different power scales caps |r| around 0.8 — sign and rough strength
        # are the meaningful pins.
        r_temp, r_hum, r_wind, r_elev = corr["power_corr"][0]
        assert r_temp is not None and r_temp < -0.7
        assert r_hum is None and r_wind is None
        assert r_elev is not None and 0.75 < r_elev < 1.0


class TestMissingnessTimeline:
    def test_pct_bounds_and_month_span(self, tmp_path):
        from scripts.export_frontend_data import missingness_timeline_bundle

        b = missingness_timeline_bundle(_mini_root(tmp_path))
        assert b["months"][0] == "2022-01"
        assert len(b["months"]) == len(b["generation_missing_slot_pct"]) == 1
        pct = b["generation_missing_slot_pct"][0]
        assert pct == 0.0          # synthetic grid has no gaps

    def test_gap_counts_as_missing(self, tmp_path):
        import shutil

        from scripts.export_frontend_data import missingness_timeline_bundle

        root = _mini_root(tmp_path)
        # interior gap: drop site 2's rows 25-50 h after the global start —
        # expected grid (per-site first/last ts) keeps those slots
        d = root / "data" / "processed" / "solar"
        df = pd.read_parquet(d)
        lo = df["timestamp"].min() + pd.Timedelta(90000, unit="s")
        hi = lo + pd.Timedelta(90000, unit="s")
        keep = df[~((df.site_id == 2) & (df.timestamp >= lo)
                    & (df.timestamp < hi))]
        shutil.rmtree(d)
        keep["year"] = keep["timestamp"].dt.year
        keep["month"] = keep["timestamp"].dt.month
        keep.to_parquet(d, engine="pyarrow",
                        partition_cols=["site_id", "year", "month"], index=False)
        b = missingness_timeline_bundle(root)
        assert b["generation_missing_slot_pct"][0] > 0
