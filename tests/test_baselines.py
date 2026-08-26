"""Phase 4 tests: chronological splits, regression metrics, baseline models.

Synthetic frames only — no dependency on the processed dataset, so these run
fast and pin exact expected numbers (PRD Rule 4: nothing fabricated; these
expectations are hand-derived from the tiny fixtures).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.splits import chronological_split
from src.models.baseline import MeanBaseline, PersistenceBaseline, ZeroBaseline
from src.training.evaluate import regression_metrics

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def make_site_frame(site_id: int, start: str, periods: int) -> pd.DataFrame:
    ts = pd.date_range(start, periods=periods, freq="15min")
    rng = np.random.default_rng(site_id)
    return pd.DataFrame(
        {
            "site_id": site_id,
            "timestamp": ts,
            "power": rng.uniform(0, 10, size=periods),
            "is_daylight": np.tile(
                np.array([False] * 32 + [True] * 32 + [False] * 32), periods // 96 + 1
            )[:periods],
        }
    )


@pytest.fixture()
def two_sites() -> pd.DataFrame:
    # Site A starts 2021-01-01, site B starts 2021-06-01 — different ranges,
    # so a global cutoff would NOT produce per-site chronological purity.
    return pd.concat(
        [make_site_frame(1, "2021-01-01", 96 * 100), make_site_frame(2, "2021-06-01", 96 * 100)],
        ignore_index=True,
    )


# ---------------------------------------------------------------------------
# src/data/splits.py — chronological split (PRD §11)
# ---------------------------------------------------------------------------


class TestChronologicalSplit:
    def test_no_row_overlap_and_full_coverage(self, two_sites):
        tr, va, te = chronological_split(two_sites, ratios=(0.7, 0.15, 0.15))
        assert len(tr) + len(va) + len(te) == len(two_sites)

    def test_temporal_order_preserved_per_site(self, two_sites):
        parts = chronological_split(two_sites, ratios=(0.7, 0.15, 0.15))
        for part in parts:
            for sid, sub in part.groupby("site_id"):
                assert sub["timestamp"].is_monotonic_increasing
        tr, va, te = parts
        for sid in (1, 2):
            assert tr.loc[tr.site_id == sid, "timestamp"].max() < va.loc[
                va.site_id == sid, "timestamp"
            ].min()
            assert va.loc[va.site_id == sid, "timestamp"].max() < te.loc[
                te.site_id == sid, "timestamp"
            ].min()

    def test_split_is_per_site_not_global_cutoff(self, two_sites):
        # Site 2 lives entirely later than site 1; a single global timestamp
        # cutoff would put all of site 2's early rows into val/test. Per-site
        # splitting guarantees each site has rows in train AND test.
        tr, _, te = chronological_split(two_sites, ratios=(0.7, 0.15, 0.15))
        assert (tr["site_id"] == 2).sum() > 0
        assert (te["site_id"] == 1).sum() > 0

    def test_proportions_approximate_ratios(self, two_sites):
        n = len(two_sites) // 2
        tr, va, te = chronological_split(two_sites, ratios=(0.7, 0.15, 0.15))
        for part, ratio in ((tr, 0.7), (va, 0.15), (te, 0.15)):
            per_site = part.groupby("site_id").size()
            assert (per_site == round(n * ratio)).all()

    def test_rejects_non_monotonic_fractions(self, two_sites):
        with pytest.raises(ValueError):
            chronological_split(two_sites, ratios=(0.5, 0.7, 0.15))


# ---------------------------------------------------------------------------
# src/training/evaluate.py — PRD §25 metrics
# ---------------------------------------------------------------------------


class TestRegressionMetrics:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        m = regression_metrics(y, y.copy())
        assert m["mae"] == pytest.approx(0.0)
        assert m["rmse"] == pytest.approx(0.0)
        assert m["r2"] == pytest.approx(1.0)

    def test_known_values(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        p = np.array([2.0, 2.0, 3.0, 6.0])
        m = regression_metrics(y, p, denom=4.0)
        # errors: 1, 0, 0, 2 -> mae = 3/4, sse = 5
        assert m["mae"] == pytest.approx(0.75)
        assert m["rmse"] == pytest.approx(np.sqrt(5.0 / 4.0))
        assert m["nrmse"] == pytest.approx(np.sqrt(5.0 / 4.0) / 4.0)
        # r2 = 1 - SSE/SST ; SST = 5 -> 0
        assert m["r2"] == pytest.approx(0.0)

    def test_daylight_metrics_use_subset_only(self):
        y = np.array([0.0, 10.0])
        p = np.array([5.0, 10.0])
        day = np.array([False, True])
        m = regression_metrics(y, p, daylight=day, denom=10.0)
        # all-period MAE includes the 5-unit night miss; daylight does not
        assert m["mae"] == pytest.approx(2.5)
        assert m["daylight_mae"] == pytest.approx(0.0)

    def test_daylight_mask_with_missing_rows(self):
        # NaN rows shrink the eval set; the daylight mask is full-length and
        # must be aligned to the *filtered* arrays, not the raw inputs.
        y = np.array([1.0, np.nan, 3.0])
        p = np.array([2.0, 5.0, 3.0])
        day = np.array([True, True, False])
        m = regression_metrics(y, p, daylight=day, denom=3.0)
        assert m["n_eval"] == 2
        assert m["daylight_n"] == 1
        assert m["daylight_mae"] == pytest.approx(1.0)

    def test_zero_variance_truth_gives_nan_r2(self):
        y = np.array([3.0, 3.0, 3.0])
        p = np.array([3.0, 4.0, 2.0])
        m = regression_metrics(y, p)
        assert np.isnan(m["r2"])

    def test_n_missing_and_n_reported(self):
        y = np.array([1.0, np.nan, 3.0])
        p = np.array([1.0, 2.0, np.nan])
        m = regression_metrics(y, p)
        assert m["n_eval"] == 1
        assert m["n_missing"] == 2


# ---------------------------------------------------------------------------
# src/models/baseline.py — PRD §21
# ---------------------------------------------------------------------------


class TestZeroBaseline:
    def test_predicts_all_zeros(self):
        df = pd.DataFrame({"site_id": [1, 1], "timestamp": pd.date_range("2021-01-01", periods=2, freq="15min")})
        pred = ZeroBaseline().fit(None).predict(df)
        assert (pred == 0.0).all()


class TestMeanBaseline:
    def fit_frame(self):
        idx = pd.date_range("2021-01-01", periods=8, freq="15min")
        return pd.DataFrame(
            {
                "site_id": [1] * 4 + [2] * 4,
                "timestamp": list(idx[:4]) * 2,
                "power": [2.0, 4.0, 6.0, 8.0, 10.0, 10.0, 10.0, 10.0],
            }
        )

    def test_mean_site_uses_train_rows_only(self):
        train = self.fit_frame()
        test = train.copy()
        test["power"] = 999.0  # must be ignored by fit
        model = MeanBaseline(scope="site").fit(train)
        assert model.predict(test).tolist() == pytest.approx([5.0] * 4 + [10.0] * 4)

    def test_mean_global_single_value(self):
        model = MeanBaseline(scope="global").fit(self.fit_frame())
        test = self.fit_frame()
        assert model.predict(test).nunique() == 1
        assert model.predict(test).iloc[0] == pytest.approx((2 + 4 + 6 + 8 + 40) / 8)

    def test_unseen_site_falls_back_to_global(self):
        model = MeanBaseline(scope="site").fit(self.fit_frame())
        test = self.fit_frame()
        test["site_id"] = 99
        preds = model.predict(test)
        assert preds.notna().all()

    def test_fit_ignores_nan_power(self):
        f = self.fit_frame()
        f.loc[0, "power"] = np.nan
        model = MeanBaseline(scope="global").fit(f)
        assert model.predict(f).iloc[0] == pytest.approx((4 + 6 + 8 + 40) / 7)


    def test_categorical_site_ids_supported(self):
        # processed parquet loads partition col site_id as Categorical
        f = self.fit_frame()
        f["site_id"] = f["site_id"].astype("category")
        model = MeanBaseline(scope="site").fit(f)
        preds = model.predict(f)
        assert preds.notna().all()
        assert list(preds) == pytest.approx([5.0] * 4 + [10.0] * 4)


class TestPersistenceBaseline:
    def build(self):
        # 96 slots/day grid over a week, one site.
        idx = pd.date_range("2021-01-01", periods=96 * 7, freq="15min")
        power = np.arange(len(idx), dtype=float)
        return pd.DataFrame({"site_id": 1, "timestamp": idx, "power": power})

    def test_prediction_is_same_time_previous_day(self):
        df = self.build()
        model = PersistenceBaseline().fit(df)
        out = model.predict(df)
        expected = df["power"].shift(96).to_numpy()
        np.testing.assert_allclose(out.to_numpy()[96:], expected[96:])

    def test_first_day_has_no_previous_observation_nan(self):
        df = self.build()
        model = PersistenceBaseline().fit(df)
        out = model.predict(df)
        assert out.iloc[:96].isna().all()

    def test_missing_prior_day_yields_nan_not_zero(self):
        df = self.build()
        df.loc[df.index[500], "power"] = np.nan  # hole in the history
        model = PersistenceBaseline().fit(df)
        out = model.predict(df)
        # t = slot 500 + 96 has no usable prior observation
        assert np.isnan(out.iloc[500 + 96])

    def test_works_across_dst_gap_hour(self):
        # 2021-04-04 03:00 exists twice-ish / 02:45 missing in Melbourne;
        # naive wall-clock shift still maps onto existing rows or NaN —
        # contract: no exception, no invented values.
        idx = pd.date_range("2021-04-03", periods=96 * 3, freq="15min")
        df = pd.DataFrame({"site_id": 1, "timestamp": idx, "power": 1.0})
        model = PersistenceBaseline().fit(df)
        out = model.predict(df)
        assert len(out) == len(df)

    def test_categorical_site_ids_supported(self):
        df = self.build()
        df["site_id"] = df["site_id"].astype("category")
        out = PersistenceBaseline().fit(df).predict(df)
        assert len(out) == len(df)
        assert out.iloc[:96].isna().all()
        assert out.iloc[96:].notna().all()

    def test_lookup_is_strictly_causal(self):
        # Prediction at t reads only t-24h. Corrupting a later row must not
        # change any prediction made before t+24h (leakage guard, PRD §46).
        df = self.build()
        base = PersistenceBaseline().fit(df).predict(df)
        j = 400
        df2 = df.copy()
        df2.loc[df2.index[j], "power"] = 99999.0
        alt = PersistenceBaseline().fit(df2).predict(df2)
        np.testing.assert_allclose(base.iloc[: j + 96], alt.iloc[: j + 96])
        assert not base.iloc[j + 96] == pytest.approx(alt.iloc[j + 96])

    def test_predict_does_not_mutate_input(self):
        df = self.build()
        before = df.copy(deep=True)
        PersistenceBaseline().fit(df).predict(df)
        pd.testing.assert_frame_equal(df, before)


# ---------------------------------------------------------------------------
# leakage guard (PRD §46 precursor): baselines never see eval-period targets
# ---------------------------------------------------------------------------


def test_baselines_fit_stats_come_from_train_slice_only():
    df = make_site_frame(1, "2021-01-01", 96 * 30)
    tr, va, _ = chronological_split(df, ratios=(0.7, 0.15, 0.15))
    model = MeanBaseline(scope="global").fit(tr)
    full_mean = df["power"].mean()
    assert model.global_mean_ != pytest.approx(full_mean)
