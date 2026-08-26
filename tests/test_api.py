"""Phase 12/14 tests: REST API (PRD §33-36).

Exercises every route against a ``MemStore`` (tiny synthetic data + a tiny
categorical XGBoost model + a tiny LSTM package), so no parquet/artifact
files are touched:

* health/dataset/sites/models shapes;
* recursive forecast: horizon length, monotonic 15-min timestamps,
  finite predictions, conformal bounds present for xgboost and absent for
  persistence and the deep models, unknown site/model errors;
* deep models (lstm/gru/transformer): served recursively since Phase 14;
* batch: multiple sites in one call, >MAX_BATCH rejected;
* history: filters (start/end), resolution resampling, unknown site 404;
* metrics endpoint mirrors the store's numbers, unknown model 404.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from src.api.app import MAX_BATCH, create_app
from src.api.store import MemStore
from src.models.xgboost_model import prepare_matrix


def synth(n_steps: int = 96 * 4, seed: int = 0):
    """Two sites, 4 days of 15-min rows with a learnable power signal."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2022-01-01", periods=n_steps, freq="15min")
    hour = ts.hour + ts.minute / 60
    frames_p, frames_h = [], []
    for sid in (1, 2):
        elev = np.sin((hour - 6) / 24 * 2 * np.pi) * 60  # pseudo elevation
        power = np.clip(elev, 0, None) * (0.2 * sid) + rng.normal(0, 0.05, n_steps)
        base = pd.DataFrame({
            "timestamp": ts, "site_id": sid, "campus_id": 2,
            "power": power,
            "temperature": 20 + rng.normal(0, 1, n_steps),
            "humidity": 60 + rng.normal(0, 2, n_steps),
        })
        frames_h.append(base)
        # features frame: same + the covariates prepare_matrix expects
        frames_p.append(base.assign(
            solar_elevation_deg=elev, wind_speed=1.0, wind_direction=180.0,
            apparent_temperature=base.temperature, dew_point_temperature=15.0))
    sites = pd.DataFrame({"site_id": [1, 2], "campus_id": [2, 2],
                          "latitude": [-36.1, -36.1],
                          "longitude": [146.8, 146.8]})
    return pd.concat(frames_p, ignore_index=True), \
        pd.concat(frames_h, ignore_index=True), sites


def tiny_booster(features: pd.DataFrame):
    from xgboost import XGBRegressor

    cols = {"categorical": ["site_id"],
            "numeric": ["solar_elevation_deg", "humidity"]}
    X = prepare_matrix(features, cols)
    y = features["power"]
    model = XGBRegressor(n_estimators=40, max_depth=4, tree_method="hist",
                         enable_categorical=True, random_state=0)
    model.fit(X, y)
    model.feature_cols_ = cols  # forecast path reads the stored layout
    return model


def tiny_sequence() -> dict:
    """Deep-model serving package mirroring ParquetStore.load_sequence."""
    from src.models.sequence_model import RecurrentForecaster

    model = RecurrentForecaster("lstm", input_size=13, hidden_size=8,
                                num_layers=1, dropout=0.0)
    model.eval()
    return {"model": model, "architecture": "lstm", "lookback": 8,
            "y_mean": 0.0, "y_std": 1.0,
            "channel_mean": [0.0] * 13, "channel_std": [1.0] * 13}


@pytest.fixture(scope="module")
def client():
    features, processed, sites = synth()
    store = MemStore(
        features, processed, sites,
        metrics={"xgboost": {"model_id": "xgboost", "split": "test",
                             "scope": "ALL", "mae": 1.0, "rmse": 2.0,
                             "r2": 0.9, "nrmse": 0.02, "daylight_mae": 1.1,
                             "daylight_nrmse": 0.02, "n_eval": 100}},
        radii={"global": 2.9, "regimes": {"day_lag": 2.9, "night_lag": 0.4}},
        booster=tiny_booster(features),
        sequence=tiny_sequence(),
    )
    return TestClient(create_app(store))


class TestMetaRoutes:
    def test_health(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        b = r.json()
        assert b["status"] == "ok" and b["n_sites"] == 2

    def test_dataset(self, client):
        b = client.get("/api/v1/dataset").json()
        assert b["n_rows"] > 0 and b["cadence_minutes"] == 15

    def test_sites(self, client):
        b = client.get("/api/v1/sites").json()
        assert [s["site_id"] for s in b["sites"]] == [1, 2]

    def test_models_registry(self, client):
        b = client.get("/api/v1/models").json()
        ids = {m["model_id"]: m["served"] for m in b["models"]}
        # Phase 14: every registered model is served
        assert set(ids) == {"persistence", "xgboost", "lstm", "gru",
                            "transformer"}
        assert all(ids.values())

    def test_metrics(self, client):
        b = client.get("/api/v1/models/xgboost/metrics").json()
        assert b["mae"] == 1.0 and b["r2"] == 0.9 and b["split"] == "test"

    def test_metrics_unknown(self, client):
        assert client.get("/api/v1/models/nope/metrics").status_code == 404


class TestForecast:
    def test_xgboost_recursive_with_bounds(self, client):
        req = {"site_id": 1, "forecast_horizon": 8, "model": "xgboost"}
        b = client.post("/api/v1/forecast", json=req).json()
        assert len(b["predictions"]) == 8
        ts = [p["timestamp"] for p in b["predictions"]]
        pd_ts = pd.to_datetime(ts)
        assert (pd_ts.to_series().diff().dropna() ==
                pd.Timedelta("15min")).all()
        preds = [p["prediction"] for p in b["predictions"]]
        assert all(np.isfinite(preds))
        first = b["predictions"][0]
        assert first["upper_bound"] > first["prediction"] > first["lower_bound"]
        assert first["confidence_level"] == 0.9

    def test_persistence_no_bounds(self, client):
        req = {"site_id": 1, "forecast_horizon": 4, "model": "persistence"}
        b = client.post("/api/v1/forecast", json=req).json()
        assert len(b["predictions"]) == 4
        assert "lower_bound" not in b["predictions"][0]

    def test_unknown_site_and_model(self, client):
        r = client.post("/api/v1/forecast",
                        json={"site_id": 99, "forecast_horizon": 2})
        assert r.status_code == 404
        r = client.post("/api/v1/forecast",
                        json={"site_id": 1, "forecast_horizon": 2,
                              "model": "nope"})
        assert r.status_code == 404

    def test_deep_model_recursive_no_bounds(self, client):
        for model in ("lstm", "gru", "transformer"):
            r = client.post("/api/v1/forecast",
                            json={"site_id": 2, "forecast_horizon": 6,
                                  "model": model})
            assert r.status_code == 200, model
            preds = r.json()["predictions"]
            assert len(preds) == 6
            ts = pd.to_datetime([p["timestamp"] for p in preds])
            assert (ts.to_series().diff().dropna() ==
                    pd.Timedelta("15min")).all()
            values = [p["prediction"] for p in preds]
            assert all(np.isfinite(values))
            # no conformal calibration exists for the deep models
            assert "lower_bound" not in preds[0]

    def test_deep_model_insufficient_history(self, client):
        """A lookback longer than the stored tail must be a 422, not a crash."""
        features, processed, sites = synth()
        store = MemStore(
            features, processed, sites, booster=None,
            sequence=dict(tiny_sequence(), lookback=10_000))
        short = TestClient(create_app(store))
        r = short.post("/api/v1/forecast",
                       json={"site_id": 1, "forecast_horizon": 2,
                             "model": "lstm"})
        assert r.status_code == 422

    def test_horizon_validation(self, client):
        r = client.post("/api/v1/forecast",
                        json={"site_id": 1, "forecast_horizon": 0})
        assert r.status_code == 422

    def test_batch(self, client):
        req = {"requests": [
            {"site_id": 1, "forecast_horizon": 2, "model": "persistence"},
            {"site_id": 2, "forecast_horizon": 2, "model": "gru"},
        ]}
        b = client.post("/api/v1/forecast/batch", json=req).json()
        assert [r["site_id"] for r in b["results"]] == [1, 2]

    def test_batch_too_many(self, client):
        req = {"requests": [{"site_id": 1, "forecast_horizon": 1}
                            for _ in range(MAX_BATCH + 1)]}
        assert client.post("/api/v1/forecast/batch", json=req).status_code == 422


class TestHistory:
    def test_full_and_filtered(self, client):
        full = client.get("/api/v1/sites/1/history").json()
        assert full["n_rows"] == 96 * 4
        part = client.get("/api/v1/sites/1/history",
                          params={"start": "2022-01-02",
                                  "end": "2022-01-02"}).json()
        assert 0 < part["n_rows"] < full["n_rows"]
        assert all(r["timestamp"].startswith("2022-01-02")
                   for r in part["rows"])

    def test_resolution_resample(self, client):
        b = client.get("/api/v1/sites/1/history",
                       params={"resolution": "1h"}).json()
        assert b["n_rows"] == 96  # 4 days × 24 h

    def test_unknown_site(self, client):
        assert client.get("/api/v1/sites/99/history").status_code == 404

    def test_bad_resolution(self, client):
        r = client.get("/api/v1/sites/1/history",
                       params={"resolution": "7min"})
        assert r.status_code == 422


class TestIncrementalFeatures:
    """Serving-path lag/rolling queries must reproduce the Phase 5 batch
    semantics (calendar-exact lags; closed-left time windows; ddof=1 std)
    that the models were trained on — including grid holes and fed-back
    predictions (the recursion contract of D-019)."""

    @staticmethod
    def _gappy_tail(seed=7):
        rng = np.random.default_rng(seed)
        ts = pd.date_range("2022-03-01", periods=96 * 3, freq="15min")
        keep = rng.random(len(ts)) > 0.15            # 15% missing grid slots
        hour = (ts.hour + ts.minute / 60).to_numpy(dtype=float)[keep]
        power = np.clip(np.sin((hour - 6) / 24 * 2 * np.pi), 0, None) * 8
        power[rng.random(power.size) < 0.10] = np.nan  # observed-NaN holes
        ts = ts[keep]
        return pd.DataFrame({"timestamp": ts, "site_id": 1, "campus_id": 2,
                             "power": power})

    def test_matches_batch_with_feedback(self):
        from src.api.forecast import CADENCE_S, LAG_STEPS, ROLLING_WINDOWS_S
        from src.api.forecast import _PowerHistory
        from src.features.lag import add_lags
        from src.features.rolling import add_rolling_features

        tail = self._gappy_tail()
        horizon, pred_const = 40, 3.25
        t_last = tail["timestamp"].max()

        # OLD semantics: extend + full batch rebuild per step, feed preds back
        frame = tail.copy()
        old_rows = []
        for i in range(1, horizon + 1):
            t = t_last + pd.Timedelta(i * CADENCE_S, unit="s")
            frame = pd.concat([frame, pd.DataFrame(
                [{"timestamp": t, "site_id": 1, "campus_id": 2,
                  "power": np.nan}])], ignore_index=True)
            feat = add_lags(frame, {f"l{s}": pd.Timedelta(s * CADENCE_S,
                                                          unit="s")
                                    for s in LAG_STEPS})
            feat = add_rolling_features(
                feat, windows=[pd.Timedelta(w, unit="s")
                               for w in ROLLING_WINDOWS_S],
                stats=["mean", "std", "min", "max"], min_periods=1)
            old_rows.append(feat.iloc[-1])
            frame.loc[frame.index[-1], "power"] = pred_const

        # NEW serving path: timestamp-keyed incremental queries
        static = pd.concat([tail, pd.DataFrame({
            "timestamp": [t_last + pd.Timedelta(i * CADENCE_S, unit="s")
                          for i in range(1, horizon + 1)],
            "site_id": 1, "campus_id": 2, "power": np.nan})],
            ignore_index=True)
        hist = _PowerHistory(static["timestamp"].to_numpy(),
                             static["power"].to_numpy())
        n_obs = len(static) - horizon
        for i in range(1, horizon + 1):
            k = n_obs + i - 1
            t = static["timestamp"].iloc[k]
            new_vals, old = {}, old_rows[i - 1]
            for s in LAG_STEPS:
                new_vals[f"l{s}"] = hist.lag(t, pd.Timedelta(s * CADENCE_S,
                                                             unit="s"))
            for w in ROLLING_WINDOWS_S:
                r = hist.roll(t, pd.Timedelta(w, unit="s"))
                for stat, val in r.items():
                    new_vals[f"power_rolling_{stat}_{w}s"] = val
            for name, v in new_vals.items():
                assert np.isnan(v) == bool(pd.isna(old[name])), name
                if not np.isnan(v):
                    assert abs(v - float(old[name])) < 1e-9, name
            hist.powers[k] = pred_const

    def test_single_value_window_std_is_nan(self):
        from src.api.forecast import _PowerHistory

        ts = pd.date_range("2022-01-01", periods=3, freq="15min")
        # only the MIDDLE slot observed — the current step (last) stays
        # excluded by closed-left, so the window sees exactly one value
        hist = _PowerHistory(ts.to_numpy(), np.array([np.nan, 5.0, np.nan]))
        r = hist.roll(ts[2], pd.Timedelta(3600, unit="s"))
        assert r["mean"] == 5.0 and np.isnan(r["std"])  # ddof=1, n=1
        assert r["min"] == 5.0 and r["max"] == 5.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
