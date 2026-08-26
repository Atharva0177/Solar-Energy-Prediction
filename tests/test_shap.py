"""Phase 10 tests: SHAP explainability (PRD §29).

Pins the contract the artifacts rely on:

* TreeExplainer additivity on a categorical XGBoost model — base + ΣSHAP
  must equal predictions (the Phase 6 booster uses enable_categorical=True;
  if a shap/xgboost upgrade breaks that path, this fails loudly).
* ``global_importance`` ranking/shape invariants.
* ``sample_rows`` determinism and observed-target filtering.
* ``select_local_examples`` scenario predicates + determinism.
* ``contribution_table`` sorting, sign convention, remainder row.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.explain.shap_explain import (
    contribution_table,
    explain_matrix,
    global_importance,
    sample_rows,
    select_local_examples,
)
from src.models.xgboost_model import prepare_matrix

rng = np.random.default_rng(7)


def synth_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    ts = pd.date_range("2021-01-01", periods=n, freq="15min")
    elevation = np.clip(r.normal(30, 20, n), -90, 90)
    daylight = (elevation > 0).astype(int)
    power = np.clip(elevation, 0, None) * 0.3 * r.uniform(0.5, 1.5, n)
    return pd.DataFrame({
        "site_id": r.integers(1, 4, n),
        "timestamp": ts,
        "power": power,
        "is_daylight": daylight,
        "solar_elevation_deg": elevation,
        "humidity": r.uniform(30, 100, n),
        "temperature": r.normal(18, 6, n),
        "hour": ts.hour,
    })


def tiny_model(df: pd.DataFrame):
    """Fit a small categorical XGBRegressor through the production path.

    Training MUST use ``prepare_matrix``'s layout (categorical columns
    first): xgboost validates category dtypes positionally inside shap's
    internal predict, so a different column order fails even when names
    match — exactly what the real Phase 6 run does correctly.
    """
    from xgboost import XGBRegressor

    cols = {"categorical": ["site_id"],
            "numeric": ["solar_elevation_deg", "humidity"]}
    X = prepare_matrix(df, cols)
    y = df["power"]
    model = XGBRegressor(n_estimators=30, max_depth=4, tree_method="hist",
                         enable_categorical=True, random_state=0)
    model.fit(X, y)
    return model, X


def test_tree_explainer_additivity_with_categorical():
    """base_value + Σ SHAP == prediction, incl. categorical splits."""
    df = synth_frame()
    model, X = tiny_model(df)
    # explain_matrix path: prepare_matrix via the model's stored layout
    model.feature_cols_ = {"categorical": ["site_id"],
                           "numeric": ["solar_elevation_deg", "humidity"]}
    sv, base = explain_matrix(model, X)
    assert sv.shape == (len(X), 3)
    # X is already the exact training layout — no re-preparation needed
    err = np.abs(base + sv.sum(axis=1) - model.predict(X))
    assert err.max() < 1e-4


def test_global_importance_ranks_descending():
    sv = rng.normal(size=(500, 5)) * np.array([10, 5, 1, 0.1, 0])
    names = [f"f{i}" for i in range(5)]
    imp = global_importance(sv, names)
    assert list(imp.columns[:2]) == ["rank", "feature"]
    assert imp["mean_abs_shap"].is_monotonic_decreasing
    assert list(imp["rank"]) == [1, 2, 3, 4, 5]
    assert imp.iloc[0]["feature"] == "f0"
    assert imp["share_of_total"].between(0, 1).all()
    assert pytest.approx(imp["share_of_total"].sum()) == pytest.approx(1.0)


def test_sample_rows_deterministic_and_observed_only():
    df = synth_frame(400, seed=1)
    df.loc[df.index[:50], "power"] = np.nan
    s1 = sample_rows(df, n=100, seed=42)
    s2 = sample_rows(df, n=100, seed=42)
    s3 = sample_rows(df, n=100, seed=7)
    assert len(s1) == 100 and len(s3) == 100
    pd.testing.assert_frame_equal(s1, s2)
    assert not s1["power"].isna().any()
    assert set(s3.index) != set(s1.index)


def _scenario_frame() -> pd.DataFrame:
    rows = [
        # night: elevation<0, no humidity filter needed
        dict(site_id=1, timestamp=pd.Timestamp("2021-01-01 22:00"),
             power=0.0, is_daylight=0, solar_elevation_deg=-15.0,
             humidity=60.0),
        # clear noon peak candidates (max wins)
        dict(site_id=1, timestamp=pd.Timestamp("2021-01-01 12:00"),
             power=9.0, is_daylight=1, solar_elevation_deg=62.0,
             humidity=40.0),
        dict(site_id=1, timestamp=pd.Timestamp("2021-01-02 12:00"),
             power=12.0, is_daylight=1, solar_elevation_deg=64.0,
             humidity=35.0),
        # morning ramp: hour 6-9, elevation 5..25 → median-power pick
        dict(site_id=2, timestamp=pd.Timestamp("2021-01-01 07:00"),
             power=1.0, is_daylight=1, solar_elevation_deg=8.0,
             humidity=70.0),
        dict(site_id=2, timestamp=pd.Timestamp("2021-01-01 08:00"),
             power=2.5, is_daylight=1, solar_elevation_deg=15.0,
             humidity=65.0),
        dict(site_id=2, timestamp=pd.Timestamp("2021-01-01 09:00"),
             power=5.0, is_daylight=1, solar_elevation_deg=22.0,
             humidity=55.0),
        # overcast afternoon: daylight, humidity>80, hour>=13
        dict(site_id=3, timestamp=pd.Timestamp("2021-01-01 14:00"),
             power=3.0, is_daylight=1, solar_elevation_deg=40.0,
             humidity=88.0),
    ]
    return pd.DataFrame(rows)


class TestSelectLocalExamples:
    def test_tags_satisfy_predicates(self):
        frame = _scenario_frame()
        tags = select_local_examples(frame)
        assert set(tags) == {"night_zero", "clear_noon_peak", "morning_ramp",
                             "overcast_afternoon"}
        f = frame
        i_night = tags["night_zero"]
        assert not bool(f.at[i_night, "is_daylight"])
        i_peak = tags["clear_noon_peak"]
        assert f.at[i_peak, "power"] == f.loc[
            f.is_daylight.astype(bool)
            & (f.solar_elevation_deg > 45)]["power"].max()
        i_ramp = tags["morning_ramp"]
        assert 6 <= f.at[i_ramp, "timestamp"].hour <= 9
        assert 5 <= f.at[i_ramp, "solar_elevation_deg"] <= 25
        i_ovc = tags["overcast_afternoon"]
        assert f.at[i_ovc, "humidity"] > 80
        assert f.at[i_ovc, "timestamp"].hour >= 13

    def test_deterministic_and_median_pick(self):
        frame = _scenario_frame()
        t1 = select_local_examples(frame)
        t2 = select_local_examples(frame)
        assert t1 == t2
        # ramp median of [1.0, 2.5, 5.0] is 2.5 → the 08:00 row
        assert frame.at[t1["morning_ramp"], "timestamp"].hour == 8


class TestContributionTable:
    def _row(self):
        feats = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0})
        shap_row = np.array([0.5, -2.0, 0.05])
        return feats, shap_row, 1.5

    def test_sorted_by_abs_shap_with_sign_and_remainder(self):
        feats, shap_row, base = self._row()
        t = contribution_table(feats, shap_row, base, top_k=2)
        assert list(t["feature"][:2]) == ["b", "a"]
        assert list(t["direction"][:2]) == ["-", "+"]
        rem = t.iloc[-1]
        assert rem["feature"].startswith("<remaining 1 features>")
        assert rem["shap"] == pytest.approx(0.05)
        assert rem["direction"] == "+"
        assert (t["base_value"] == base).all()

    def test_no_remainder_when_top_k_ge_features(self):
        feats, shap_row, base = self._row()
        t = contribution_table(feats, shap_row, base, top_k=10)
        assert len(t) == 3
        assert not t["feature"].str.startswith("<").any()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
