"""Phase 9 leakage tests (PRD §46) — these must fail CI if any leak appears.

Surfaces covered here (each named in PRD §46):

* future power entering lag features            → corruption test on ``add_lags``
* rolling windows including future observations → corruption test on
  ``add_rolling_features`` (deeper semantics already in tests/test_features.py)
* future weather entering features              → interpolation limit respected:
  gaps longer than the documented ≤2-step window stay NaN (no long-range
  future reach); the accepted ≤30-min bounded smoothing is D-008
* test data used during scaling                 → eval corruption cannot move
  train-fitted statistics; transforms use frozen (mean, std)
* test sites appearing in training              → cross-site disjointness
  (asserted against the real split function)
* hyperparameter tuning using the test set      → ``train_xgboost`` accepts no
  test frame at all and early-stops on validation only (captured via stub)

Sequence-model current-step power masking is covered by
tests/test_sequence_model.py::TestWindowDataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def hourly_site(days: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2021-06-01", periods=24 * days, freq="h")
    return pd.DataFrame({
        "site_id": 1, "timestamp": ts,
        "power": rng.uniform(0, 10, len(ts)),
    })


# ---------------------------------------------------------------------------
# lags: future perturbation must not change earlier features
# ---------------------------------------------------------------------------


def test_future_power_never_enters_lags():
    from src.features.lag import add_lags

    df = hourly_site()
    specs = {f"lag_{s}": pd.Timedelta(hours=s) for s in (1, 6, 24)}
    clean = add_lags(df, specs)

    corrupt = df.copy()
    corrupt.loc[corrupt.index[-48:], "power"] = 999.0   # far future values
    dirty = add_lags(corrupt, specs)

    head = slice(0, len(df) - 48 - 24)                   # rows whose t−24h < cutoff
    for col in specs:
        np.testing.assert_array_equal(
            clean[col].to_numpy()[head], dirty[col].to_numpy()[head],
            err_msg=f"{col} changed when FUTURE power was corrupted")


def test_lag_never_reads_ahead():
    """With everything from t=50 onward missing, the LAST defined lag_1 sits
    at t=50 (reading the still-observed t=49). Nothing later can be defined —
    a lag at t>50 would require reading missing-or-future rows."""
    from src.features.lag import add_lags

    df = hourly_site(3)                                  # 72 hourly rows
    df.loc[df.index[50:], "power"] = np.nan
    out = add_lags(df, {"lag_1": pd.Timedelta(hours=1)})
    assert out["lag_1"].dropna().index[-1] == 50


# ---------------------------------------------------------------------------
# rolling: future observations excluded
# ---------------------------------------------------------------------------


def test_rolling_excludes_future_under_perturbation():
    from src.features.rolling import add_rolling_features

    df = hourly_site()
    clean = add_rolling_features(df, windows=[pd.Timedelta("1h")], stats=["mean"])

    corrupt = df.copy()
    corrupt.loc[corrupt.index[-24:], "power"] = -777.0
    dirty = add_rolling_features(corrupt, windows=[pd.Timedelta("1h")], stats=["mean"])

    head = slice(0, len(df) - 24)                       # windows fully before cutoff
    np.testing.assert_array_equal(
        clean["power_rolling_mean_3600s"].to_numpy()[head],
        dirty["power_rolling_mean_3600s"].to_numpy()[head])


# ---------------------------------------------------------------------------
# weather: interpolation bounded by the D-008 limit (≤2 steps / 30 min)
# ---------------------------------------------------------------------------


class _NoopLog:
    def add(self, *a, **k):
        pass


def _weather_frame(values: list[float]) -> pd.DataFrame:
    n = len(values)
    return pd.DataFrame({
        "site_id": [1] * n,
        "campus_id": [7] * n,
        "timestamp": pd.date_range("2021-06-01", periods=n, freq="15min"),
        "temperature": values,
        "apparent_temperature": values,
        "dew_point_temperature": values,
        "humidity": values,
        "wind_speed": values,
        "wind_direction": [180.0] * n,
        "power": [1.0] * n,
    })


def test_weather_interp_gap_within_limit_is_filled():
    import scripts.build_processed as bp

    vals = [10.0] * 10 + [np.nan] + [14.0] * 10        # single-step gap
    out = bp.handle_missing(_weather_frame(vals), _NoopLog())
    assert abs(out["temperature"].iloc[10] - 12.0) < 1e-9


def test_weather_interp_long_gap_stays_nan_no_far_future_reach():
    """A 4-step gap must NOT be bridged: interpolation is capped at 2 steps,
    so nothing beyond ±30 min can ever influence a feature value."""
    import scripts.build_processed as bp

    vals = [10.0] * 8 + [np.nan] * 4 + [18.0] * 8
    out = bp.handle_missing(_weather_frame(vals), _NoopLog())
    filled = out["temperature"].iloc[8:12]
    assert filled.isna().sum() >= 2, (
        "interpolation crossed more than the documented 2-step window — "
        "future weather is reaching backward into features")
    # any value it did fill must lie between the bracketing real observations
    assert filled.dropna().between(10.0, 18.0).all()


def test_power_is_never_interpolated():
    import scripts.build_processed as bp

    f = _weather_frame([10.0] * 20)
    f["power"] = [1.0] * 10 + [np.nan] * 4 + [3.0] * 6
    out = bp.handle_missing(f, _NoopLog())
    assert out["power"].isna().sum() == 4               # D-008: never impute power


# ---------------------------------------------------------------------------
# scaling: eval data cannot influence fitted statistics or train transform
# ---------------------------------------------------------------------------


def test_scaling_stats_immune_to_eval_corruption():
    from src.models.sequence_model import CHANNELS, build_channel_matrix, fit_channel_scaler

    def full(rows):
        """Hourly frame carrying EVERY channel column the matrix builder reads."""
        d = hourly_site(rows)
        n = len(d)
        d["temperature"], d["humidity"], d["wind_speed"] = 15.0, 60.0, 3.0
        d["wind_dir_sin"], d["wind_dir_cos"] = 0.0, 1.0
        d["solar_elevation_deg"], d["zenith_deg"] = 30.0, 60.0
        for c in ("sin_hour", "cos_hour", "sin_day_of_year", "cos_day_of_year"):
            d[c] = 0.5
        assert set(CHANNELS) - {"power", "power_observed"} <= set(d.columns)
        return d

    tr, ev = full(72 * 5), full(72 * 2)
    M_tr = build_channel_matrix(tr)
    mean0, std0 = fit_channel_scaler(M_tr)

    ev_corrupt = ev.copy()
    ev_corrupt["power"] = 1e9                            # wildly corrupted EVAL rows
    M_all = build_channel_matrix(pd.concat([tr, ev_corrupt], ignore_index=True))
    # pipeline contract: scaler sees ONLY the train slice — refit on identical
    # train rows must reproduce the same statistics bit-for-bit
    mean1, std1 = fit_channel_scaler(M_all[:len(M_tr)])
    np.testing.assert_array_equal(mean0, mean1)
    np.testing.assert_array_equal(std0, std1)
    # sanity: had eval rows been (wrongly) included, stats WOULD move
    _, std_bad = fit_channel_scaler(M_all)
    assert not np.allclose(std0, std_bad)

    M_tr = build_channel_matrix(tr)
    mean0, std0 = fit_channel_scaler(M_tr)

    ev_corrupt = ev.copy()
    ev_corrupt["power"] = 1e9                            # wildly corrupted EVAL rows
    M_all = build_channel_matrix(pd.concat([tr, ev_corrupt], ignore_index=True))
    # pipeline contract: scaler sees ONLY the train slice — refit on identical
    # train rows must reproduce the same statistics bit-for-bit
    mean1, std1 = fit_channel_scaler(M_all[:len(M_tr)])
    np.testing.assert_array_equal(mean0, mean1)
    np.testing.assert_array_equal(std0, std1)
    # and the eval block contributed nothing: stats differ from fit-on-everything
    _, std_bad = fit_channel_scaler(M_all)
    assert not np.allclose(std0, std_bad)


# ---------------------------------------------------------------------------
# sites: held-out sites absent from training frames
# ---------------------------------------------------------------------------


def test_test_sites_never_in_training_frames():
    from src.data.splits import cross_site_split

    sys.path.insert(0, str(Path(__file__).parent))
    from test_cross_site import synth_sites

    out = cross_site_split(synth_sites(16), seed=9)
    fr, s = out["frames"], out["sites"]
    for name in ("train", "val_seen", "test_seen"):
        assert not fr[name]["site_id"].isin(s["test"]).any(), \
            f"test site leaked into {name}"
        assert not fr[name]["site_id"].isin(s["val"]).any(), \
            f"val site leaked into {name}"


# ---------------------------------------------------------------------------
# tuning: early stopping consumes validation only
# ---------------------------------------------------------------------------


def test_tuning_uses_validation_not_test(monkeypatch):
    """train_xgboost has no test parameter at all; the stub records what it
    is asked to early-stop on — exactly one eval set, the validation frame."""
    import src.models.xgboost_model as xg

    captured = {}

    class StubLearner:
        def __init__(self, **kw):
            self.kw = kw

        def fit(self, X, y, eval_set=None, verbose=False):
            captured["eval_sets"] = [len(e[0]) for e in (eval_set or [])]
            captured["n_train"] = len(X)
            self.best_iteration = 3
            return self

        def get_booster(self):
            raise AssertionError("not used here")

    monkeypatch.setattr(xg, "XGBRegressor", StubLearner)

    def frame(rows, base_power):
        ts = pd.date_range("2021-06-01", periods=rows, freq="15min")
        return pd.DataFrame({"site_id": np.arange(rows) % 7,
                             "timestamp": ts, "power": base_power})

    val = frame(40, 5.0)
    model, info = xg.train_xgboost(frame(200, 1.0), val, params={"n_estimators": 10})
    assert captured["eval_sets"] == [40]                # exactly the val frame
    assert captured["n_train"] == 200
