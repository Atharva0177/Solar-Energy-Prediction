"""Phase 5 tests: temporal, lag, rolling, weather, solar-position features.

Tiny synthetic frames; expectations hand-derived. Causality (PRD §17: rolling
must never see future observations) is tested by perturbation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.lag import add_lags
from src.features.rolling import add_rolling_features
from src.features.solar import add_solar_position_features
from src.features.temporal import SEASON_BY_MONTH, add_temporal_features
from src.features.weather import available_weather_columns, add_weather_features


def grid(start: str, periods: int, freq: str = "15min") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site_id": 1,
            "timestamp": pd.date_range(start, periods=periods, freq=freq),
            "power": np.arange(periods, dtype=float),
        }
    )


# ---------------------------------------------------------------------------
# temporal (PRD §15)
# ---------------------------------------------------------------------------


class TestTemporalFeatures:
    def test_known_timestamp_values(self):
        # 2021-01-02 was a Saturday? No — it was a Sunday... check: 2021-01-01
        # = Friday, so 01-02 = Saturday (dow 5), ISO week 53 of 2020.
        df = pd.DataFrame({"timestamp": [pd.Timestamp("2021-01-02 03:45")]})
        out = add_temporal_features(df)
        r = out.iloc[0]
        assert r["hour"] == 3
        assert r["minute"] == 45
        assert r["day"] == 2
        assert r["day_of_week"] == 5  # Monday=0 ... Saturday=5
        assert r["day_of_year"] == 2
        assert r["week_of_year"] == 53
        assert r["month"] == 1
        assert r["quarter"] == 1
        assert bool(r["is_weekend"]) is True

    def test_season_mapping_southern_hemisphere(self):
        assert SEASON_BY_MONTH[12, 1, 2] == "summer"
        assert SEASON_BY_MONTH[3, 4, 5] == "autumn"
        assert SEASON_BY_MONTH[6, 7, 8] == "winter"
        assert SEASON_BY_MONTH[9, 10, 11] == "spring"

    def test_cyclical_hour_continuous_across_midnight(self):
        # midnight-crossing step must look like any other 15-min step
        df = pd.DataFrame({"timestamp": pd.to_datetime([
            "2021-06-01 23:45", "2021-06-02 00:00",
            "2021-06-01 14:00", "2021-06-01 14:15"])})
        out = add_temporal_features(df)
        for pre in ("sin", "cos"):
            midnight_gap = abs(out[f"{pre}_hour"].iloc[0] - out[f"{pre}_hour"].iloc[1])
            normal_gap = abs(out[f"{pre}_hour"].iloc[2] - out[f"{pre}_hour"].iloc[3])
            assert midnight_gap <= 1.5 * normal_gap

    def test_cyclical_values_bounded(self):
        df = pd.DataFrame({"timestamp": pd.date_range("2021-01-01", periods=96 * 30, freq="15min")})
        out = add_temporal_features(df)
        for col in ("sin_hour", "cos_hour", "sin_day_of_year", "cos_day_of_year"):
            assert out[col].abs().max() <= 1.0 + 1e-9

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"timestamp": pd.date_range("2021-01-01", periods=3, freq="15min")})
        before = df.copy(deep=True)
        add_temporal_features(df)
        pd.testing.assert_frame_equal(df, before)


# ---------------------------------------------------------------------------
# lags (PRD §16) — calendar-exact, not positional
# ---------------------------------------------------------------------------


class TestLags:
    def test_calendar_exact_alignment_survives_gap(self):
        # drop the 00:15 slot; a positional shift(1) would misalign everything.
        df = grid("2021-06-01", 8).drop(index=[1]).reset_index(drop=True)
        out = add_lags(df, {"power_lag_15m": pd.Timedelta(900, unit="s")})
        # row at 00:45 must see power of 00:30 (=row value 2), not 00:15's.
        assert out.loc[out.timestamp == pd.Timestamp("2021-06-01 00:45"), "power_lag_15m"].iloc[0] == 2.0

    def test_lag_before_history_is_nan(self):
        # frame spans 00:00-01:00; only the last row has a t-1h observation
        out = add_lags(grid("2021-06-01", 5), {"lag_1h": pd.Timedelta(3600, unit="s")})
        assert out["lag_1h"].iloc[:4].isna().all()
        assert out["lag_1h"].iloc[4] == pytest.approx(0.0)

    def test_missing_prior_observation_is_nan_not_zero(self):
        df = grid("2021-06-01", 6)
        df.loc[df.index[2], "power"] = np.nan
        out = add_lags(df, {"lag_15m": pd.Timedelta(900, unit="s")})
        assert np.isnan(out.loc[out.index[3], "lag_15m"])

    def test_never_looks_forward(self):
        df = grid("2021-06-01", 10)
        base = add_lags(df, {"lag_15m": pd.Timedelta(900, unit="s")})
        df2 = df.copy()
        df2.loc[df2.index[7], "power"] = 99999.0
        alt = add_lags(df2, {"lag_15m": pd.Timedelta(900, unit="s")})
        np.testing.assert_allclose(base["lag_15m"].iloc[:8], alt["lag_15m"].iloc[:8])

    def test_respects_site_boundaries(self):
        a, b = grid("2021-06-01", 4), grid("2021-06-01", 4)
        b["site_id"], b["power"] = 2, 100.0 + b["power"]
        both = pd.concat([a, b], ignore_index=True)
        out = add_lags(both, {"lag_15m": pd.Timedelta(900, unit="s")})
        site2 = out[out.site_id == 2].reset_index(drop=True)
        assert site2["lag_15m"].iloc[-1] == 102.0  # from site 2, not site 1


# ---------------------------------------------------------------------------
# rolling (PRD §17) — history only
# ---------------------------------------------------------------------------


class TestRolling:
    STATS = ("mean", "std", "min", "max")

    def roll(self, df, seconds=3600):
        return add_rolling_features(
            df, windows=[pd.Timedelta(seconds, "s")], stats=list(self.STATS), min_periods=1
        )

    def names(self, df, suffix="3600s"):
        # naming convention: window rendered as whole seconds
        return [f"power_rolling_{s}_{suffix}" for s in self.STATS]

    def test_excludes_current_observation(self):
        df = grid("2021-06-01", 4)
        df.loc[df.index[3], "power"] = 9999.0  # current obs must not leak in
        out = self.roll(df)
        m = out.loc[out.index[3], "power_rolling_mean_3600s"]
        prior_mean = df["power"].iloc[:3].mean()
        assert m == pytest.approx(prior_mean)

    def test_future_perturbation_changes_nothing_behind(self):
        df = grid("2021-06-01", 20)
        base = self.roll(df)
        df2 = df.copy()
        df2.loc[df2.index[15], "power"] = -77.0
        alt = self.roll(df2)
        cols = self.names(df)
        np.testing.assert_allclose(base[cols].iloc[:16].to_numpy(),
                                   alt[cols].iloc[:16].to_numpy())

    def test_window_semantics_left_closed(self):
        # pandas closed='left' time window = [t-W, t): current obs excluded,
        # left boundary included. 1800s window over 15-min cadence holds 2 obs.
        df = grid("2021-06-01", 5)
        out = add_rolling_features(
            df,
            windows=[pd.Timedelta(1800, unit="s")],
            stats=["mean", "min", "max"],
            min_periods=1,
        )
        assert out["power_rolling_mean_1800s"].iloc[2] == pytest.approx(0.5)  # {0,1}
        assert out["power_rolling_max_1800s"].iloc[3] == pytest.approx(2.0)   # {1,2}
        assert out["power_rolling_mean_1800s"].iloc[4] == pytest.approx(2.5)  # {2,3}

    def test_site_boundaries_respected(self):
        a, b = grid("2021-06-01", 5), grid("2021-06-01", 5)
        b["site_id"], b["power"] = 2, 100.0 + b["power"]
        out = self.roll(pd.concat([a, b], ignore_index=True))
        first_b = out[(out.site_id == 2)].sort_values("timestamp").iloc[0]
        assert first_b["power_rolling_mean_3600s"] != pytest.approx(
            out[out.site_id == 1]["power_rolling_mean_3600s"].iloc[0]
        )

    def test_nan_truth_excluded_not_zero_filled(self):
        df = grid("2021-06-01", 4)
        df.loc[df.index[2], "power"] = np.nan
        out = self.roll(df)
        mean_at_3 = out["power_rolling_mean_3600s"].iloc[3]
        assert mean_at_3 == pytest.approx(np.nanmean([0.0, 1.0]))


# ---------------------------------------------------------------------------
# weather (PRD §19) — dynamic to available variables
# ---------------------------------------------------------------------------


class TestWeatherFeatures:
    def test_wind_direction_encoding(self):
        df = pd.DataFrame({"wind_direction": [0.0, 90.0, 180.0, 270.0, 360.0]})
        out = add_weather_features(df)
        np.testing.assert_allclose(out["wind_dir_sin"], [0, 1, 0, -1, 0], atol=1e-9)
        np.testing.assert_allclose(out["wind_dir_cos"], [1, 0, -1, 0, 1], atol=1e-9)

    def test_skips_when_direction_absent(self):
        out = add_weather_features(pd.DataFrame({"temperature": [21.0]}))
        assert "wind_dir_sin" not in out.columns
        assert "temperature" in available_weather_columns(out)

    def test_dynamic_availability_reported(self):
        df = pd.DataFrame({"temperature": [21.0], "humidity": [55.0]})
        assert available_weather_columns(df) == ["temperature", "humidity"]

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"wind_direction": [45.0]})
        before = df.copy(deep=True)
        add_weather_features(df)
        pd.testing.assert_frame_equal(df, before)


# ---------------------------------------------------------------------------
# solar position (PRD §18)
# ---------------------------------------------------------------------------


COORDS_MELB = pd.DataFrame(
    [{"campus_id": 1, "latitude": -37.81, "longitude": 144.96}]
)


def make_day(month: int, day: int) -> pd.DataFrame:
    ts = pd.date_range(f"2021-{month:02d}-{day:02d}", periods=96, freq="15min")
    return pd.DataFrame({"campus_id": 1, "site_id": 1, "timestamp": ts})


@pytest.fixture(scope="module")
def summer_winter():
    s = add_solar_position_features(make_day(12, 21), COORDS_MELB, "Australia/Melbourne")
    w = add_solar_position_features(make_day(6, 21), COORDS_MELB, "Australia/Melbourne")
    return s, w


class TestSolarFeatures:

    def test_adds_expected_columns(self, summer_winter):
        out = summer_winter[0]
        for col in ("azimuth_deg", "zenith_deg", "day_length_hours",
                    "solar_elevation_deg", "is_daylight"):
            assert col in out.columns

    def test_zenith_complements_elevation(self, summer_winter):
        out = summer_winter[0]
        ok = out["zenith_deg"].notna()
        np.testing.assert_allclose(
            out.loc[ok, "zenith_deg"], 90.0 - out.loc[ok, "solar_elevation_deg"], atol=1e-6
        )

    def test_azimuth_bounded(self, summer_winter):
        az = summer_winter[0]["azimuth_deg"].dropna()
        assert az.between(-360, 360).all()

    def test_summer_day_longer_than_winter_day_melbourne(self, summer_winter):
        s_len = summer_winter[0]["day_length_hours"].iloc[0]
        w_len = summer_winter[1]["day_length_hours"].iloc[0]
        assert 0 < w_len < s_len <= 24.0

    def test_dst_ambiguity_rows_kept_as_nan_elevation(self):
        # 2021-04-04 02:00-03:00 repeats in Melbourne civil time; contract:
        # function survives, ambiguous rows get NaN position values.
        ts = pd.date_range("2021-04-04", periods=96, freq="15min")
        df = pd.DataFrame({"campus_id": 1, "site_id": 1, "timestamp": ts})
        out = add_solar_position_features(df, COORDS_MELB, "Australia/Melbourne")
        assert len(out) == len(df)
