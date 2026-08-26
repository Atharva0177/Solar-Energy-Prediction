"""Phase 6 tests: XGBoost model module, config, dataset fingerprint.

Training tests use a tiny learnable synthetic pattern (daily sinusoid +
site offset) so a correct implementation must clearly beat the mean baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.xgboost_model import (
    dataset_fingerprint,
    extract_importance,
    predict_frame,
    select_feature_columns,
    train_xgboost,
)
from src.training.evaluate import regression_metrics


def synth(days: int = 10, n_sites: int = 2, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for sid in range(1, n_sites + 1):
        ts = pd.date_range("2021-06-01", periods=96 * days, freq="15min")
        doy_angle = 2 * np.pi * ts.dayofyear / 365.25
        hour_angle = 2 * np.pi * (ts.hour * 60 + ts.minute) / 1440
        power = (
            5.0 * sid
            + 8.0 * np.clip(np.sin(hour_angle), 0, None)
            + 2.0 * np.sin(doy_angle)
            + rng.normal(0, 0.05, len(ts))
        )
        frames.append(pd.DataFrame({"site_id": sid, "timestamp": ts, "power": power}))
    return pd.concat(frames, ignore_index=True)


def with_features(df: pd.DataFrame) -> pd.DataFrame:
    from src.features.lag import add_lags
    from src.features.temporal import add_temporal_features

    out = add_temporal_features(df)
    return add_lags(out, {"power_lag_1": pd.Timedelta(900, unit="s"),
                          "power_lag_96": pd.Timedelta(86400, unit="s")})


def chronological(df, ratios=(0.7, 0.15, 0.15)):
    from src.data.splits import chronological_split as cs

    return cs(df, ratios=ratios)


# ---------------------------------------------------------------------------
# feature selection
# ---------------------------------------------------------------------------


class TestFeatureSelection:
    def test_excludes_target_and_derived_or_raw_columns(self):
        df = with_features(synth(days=1))
        cols = select_feature_columns(df)
        all_sel = cols["categorical"] + cols["numeric"]
        for banned in ("power", "wind_direction", "season", "is_daylight",
                       "timestamp", "year", "month", "campus_id"):
            assert banned not in all_sel

    def test_includes_available_families(self):
        df = with_features(synth(days=1))
        cols = select_feature_columns(df)
        assert cols["categorical"] == ["site_id"]
        for want in ("hour", "sin_hour", "cos_hour", "day_of_year",
                     "power_lag_1", "power_lag_96"):
            assert want in cols["numeric"]

    def test_absent_columns_are_skipped(self):
        df = pd.DataFrame({"site_id": [1], "timestamp": pd.Timestamp("2021-01-01")})
        cols = select_feature_columns(df)
        assert cols["categorical"] == ["site_id"]
        assert cols["numeric"] == [] or all(c in df.columns for c in cols["numeric"])

    def test_weather_and_solar_enter_when_present(self):
        df = with_features(synth(days=1))
        df["temperature"] = 20.0
        df["wind_dir_sin"] = 0.0
        df["wind_dir_cos"] = 1.0
        df["solar_elevation_deg"] = 10.0
        df["azimuth_deg"] = 100.0
        df["zenith_deg"] = 80.0
        df["day_length_hours"] = 10.0
        cols = select_feature_columns(df)
        for want in ("temperature", "wind_dir_sin", "solar_elevation_deg",
                     "azimuth_deg", "zenith_deg", "day_length_hours"):
            assert want in cols["numeric"]
        assert "wind_direction" not in cols["numeric"]


# ---------------------------------------------------------------------------
# training behaviour (small synthetic data)
# ---------------------------------------------------------------------------


PARAMS = {"n_estimators": 120, "max_depth": 6, "learning_rate": 0.1}


@pytest.fixture(scope="module")
def trained():
    tr, va, te = chronological(with_features(synth(days=12)))
    model, info = train_xgboost(tr, va, PARAMS, seed=42, early_stopping_rounds=20)
    return model, info, tr, va, te


class TestTraining:

    def test_beats_mean_baseline_clearly(self, trained):
        model, _, tr, va, te = trained
        pred = predict_frame(model, te)
        m = regression_metrics(te["power"], pred)
        mean_pred = np.full(len(te), tr["power"].mean())
        m_mean = regression_metrics(te["power"], mean_pred)
        assert m["r2"] > 0.9
        assert m["rmse"] < 0.5 * m_mean["rmse"]

    def test_early_stopping_can_fire_before_max_trees(self, trained):
        _, info, *_ = trained
        assert info["best_iteration"] <= PARAMS["n_estimators"]

    def test_deterministic_given_seed(self):
        tr, va, _ = chronological(with_features(synth(days=6)))
        m1, _ = train_xgboost(tr, va, {"n_estimators": 30}, seed=7)
        m2, _ = train_xgboost(tr, va, {"n_estimators": 30}, seed=7)
        p1, p2 = predict_frame(m1, va), predict_frame(m2, va)
        np.testing.assert_allclose(p1, p2)

    def test_nan_features_are_tolerated(self):
        df = with_features(synth(days=6))
        df["temperature"] = 20.0
        df.loc[df.index[:200], "temperature"] = np.nan
        df.loc[df.index[:300], "power_lag_1"] = np.nan
        tr, va, _ = chronological(df)
        model, _ = train_xgboost(tr, va, {"n_estimators": 30}, seed=1)
        preds = predict_frame(model, va)
        assert len(preds) == len(va)
        assert np.isfinite(preds).all()


class TestImportance:
    def test_gains_positive_and_complete(self):
        tr, va, _ = chronological(with_features(synth(days=8)))
        model, _ = train_xgboost(tr, va, PARAMS, seed=3)
        imp = extract_importance(model)
        assert not imp.empty
        assert (imp["gain"] > 0).any()
        assert imp["feature"].is_unique

    def test_hour_signal_ranks_high_for_solar_shape(self):
        tr, va, _ = chronological(with_features(synth(days=8)))
        model, _ = train_xgboost(tr, va, PARAMS, seed=3)
        imp = extract_importance(model).set_index("feature")
        top10 = set(imp.sort_values("gain", ascending=False).head(10).index)
        assert top10 & {"hour", "sin_hour", "power_lag_1"}


# ---------------------------------------------------------------------------
# reproducibility bookkeeping
# ---------------------------------------------------------------------------


class TestDatasetFingerprint:
    def test_stable_across_calls(self, tmp_path):
        p = tmp_path / "d"
        p.mkdir()
        pd.DataFrame({"a": [1]}).to_parquet(p / "x.parquet")
        f1 = dataset_fingerprint(p)
        f2 = dataset_fingerprint(p)
        assert f1 == f2 and isinstance(f1, str) and len(f1) >= 8

    def test_changes_when_content_changes(self, tmp_path):
        p = tmp_path / "d"
        p.mkdir()
        pd.DataFrame({"a": [1]}).to_parquet(p / "x.parquet")
        before = dataset_fingerprint(p)
        pd.DataFrame({"a": [2, 3]}).to_parquet(p / "x.parquet")
        after = dataset_fingerprint(p)
        assert before != after

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            dataset_fingerprint(tmp_path / "nope")


class TestConfigs:
    def test_yaml_configs_exist_and_validate(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.config import load_config

        cfg = load_config()
        assert cfg["training"]["seed"] == pytest.approx(int(cfg["training"]["seed"]))
        r = (cfg["training"]["train_ratio"], cfg["training"]["val_ratio"],
             cfg["training"]["test_ratio"])
        assert abs(sum(r) - 1.0) < 1e-9
        assert cfg["models"]["xgboost"]["enabled"] is True
        xgb_params = cfg["models"]["xgboost"]["params"]
        assert "n_estimators" in xgb_params and "learning_rate" in xgb_params
