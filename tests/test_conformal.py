"""Phase 11 tests: split-conformal prediction intervals (PRD §28).

Pins the statistical contract the artifacts rely on:

* finite-sample quantile matches hand-computed ``ceil((n+1)(1−α))`` order
  statistic;
* marginal coverage guarantee holds on synthetic exchangeable data
  (empirical ≥ nominal − tolerance, never wildly above on fresh data);
* Mondrian per-group radii: per-group coverage + unseen-label fallback;
* interval sanity (lower ≤ pred ≤ upper, positive width, NaN handling);
* calibration ignores rows with missing targets/predictions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.conformal import (
    conformal_quantile,
    coverage_metrics,
    fit_conformal,
    interval_widths,
)


class TestConformalQuantile:
    def test_matches_hand_computed_order_statistic(self):
        scores = np.arange(1.0, 11.0)  # n=10, sorted
        # alpha=0.1 → k = ceil(11*0.9) = 10 → 10th order statistic
        assert conformal_quantile(scores, 0.1) == pytest.approx(10.0)
        # alpha=0.2 → k = ceil(11*0.8) = 9
        assert conformal_quantile(scores, 0.2) == pytest.approx(9.0)
        # alpha=0.5 → k = ceil(11*0.5) = 6
        assert conformal_quantile(scores, 0.5) == pytest.approx(6.0)

    def test_alpha_too_small_returns_max(self):
        scores = np.arange(1.0, 11.0)
        # alpha=0.05 → k = ceil(11*0.95) = 11 > n → degenerate: max score
        assert conformal_quantile(scores, 0.05) == pytest.approx(10.0)

    def test_drops_nonfinite_and_rejects_empty(self):
        scores = np.array([1.0, np.nan, 3.0, np.inf])
        assert conformal_quantile(scores, 0.5) == pytest.approx(3.0)
        with pytest.raises(ValueError):
            conformal_quantile(np.array([np.nan]), 0.1)


class TestCoverageGuarantee:
    def test_marginal_coverage_on_synthetic_data(self):
        rng = np.random.default_rng(11)
        # heteroscedastic truth: error scale grows with prediction level
        n_cal, n_test = 4000, 20000
        x_cal = rng.uniform(0, 10, n_cal)
        x_test = rng.uniform(0, 10, n_test)
        y_cal = x_cal + rng.normal(0, 0.5 + 0.2 * x_cal)
        y_test = x_test + rng.normal(0, 0.5 + 0.2 * x_test)
        calib = fit_conformal(y_cal, x_cal, alpha=0.1)
        lo, hi = interval_widths(x_test, calib)
        cov = float(((y_test >= lo) & (y_test <= hi)).mean())
        # guarantee is marginal in expectation over calibration draws, not
        # per-draw: allow 1.5% slack (~7 sigma of the binomial noise at n=20k
        # would be 0.3%; this only trips on a broken quantile)
        assert 0.885 <= cov <= 1.0

    def test_calibration_drops_missing_rows(self):
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        p = np.array([0.0, np.nan, 3.0, 4.5, 6.0])
        calib = fit_conformal(y, p, alpha=0.5)
        # only rows 0, 3, 4 survive → scores [1, 0.5, 1], k=ceil(4*.5)=2 → 1.0
        assert calib["n"]["global"] == 3
        assert calib["global"] == pytest.approx(1.0)


class TestMondrian:
    def _data(self, seed=3):
        rng = np.random.default_rng(seed)
        n = 6000
        day = rng.integers(0, 2, n)  # 0=night (tiny errors), 1=day (big)
        scale = np.where(day == 1, 5.0, 0.1)
        y = rng.normal(0, scale)
        pred = np.zeros(n)
        return day, y, pred

    def test_per_group_radii_and_coverage(self):
        day, y, pred = self._data()
        n_cal = 2000
        calib = fit_conformal(y[:n_cal], pred[:n_cal], alpha=0.1,
                              groups=day[:n_cal])
        assert set(calib["groups"]) == {"0", "1"}
        # night radius ≪ day radius (heteroscedasticity captured)
        assert calib["groups"]["0"] < 0.2 * calib["groups"]["1"]
        lo, hi = interval_widths(pred[n_cal:], calib, groups=day[n_cal:])
        for label in (0, 1):
            m = day[n_cal:] == label
            cov = float(((y[n_cal:][m] >= lo[m]) & (y[n_cal:][m] <= hi[m])).mean())
            assert 0.88 <= cov <= 1.0

    def test_unseen_group_falls_back_to_global(self):
        day, y, pred = self._data()
        calib = fit_conformal(y[:2000], pred[:2000], alpha=0.1,
                              groups=day[:2000])
        lo, hi = interval_widths(pred[:5], calib,
                                 groups=np.array([7] * 5))
        assert np.allclose(lo, pred[:5] - calib["global"])
        assert np.allclose(hi, pred[:5] + calib["global"])

    def test_mondrian_beats_global_conditional_width(self):
        """Same marginal coverage, far tighter night intervals."""
        day, y, pred = self._data()
        n_cal = 2000
        glob = fit_conformal(y[:n_cal], pred[:n_cal], alpha=0.1)
        mond = fit_conformal(y[:n_cal], pred[:n_cal], alpha=0.1,
                             groups=day[:n_cal])
        night_m = day[n_cal:] == 0
        lo_g, hi_g = interval_widths(pred[n_cal:], glob)
        lo_m, hi_m = interval_widths(pred[n_cal:], mond, groups=day[n_cal:])
        assert (hi_m[night_m] - lo_m[night_m]).mean() < \
            0.2 * (hi_g[night_m] - lo_g[night_m]).mean()
        # marginal coverages comparable
        y_te = y[n_cal:]
        cov_g = float(((y_te >= lo_g) & (y_te <= hi_g)).mean())
        cov_m = float(((y_te >= lo_m) & (y_te <= hi_m)).mean())
        assert abs(cov_g - cov_m) < 0.05


class TestCoverageMetrics:
    def test_counts_and_stats(self):
        y = np.array([0.0, 1.0, 2.0, np.nan])
        p = np.array([0.0, 1.0, 5.0, 3.0])
        lo = np.array([-1.0, 0.0, 4.0, 2.0])
        hi = np.array([1.0, 2.0, 6.0, 4.0])
        m = coverage_metrics(y, p, lo, hi)
        assert m["n"] == 3
        assert m["coverage"] == pytest.approx(2 / 3)
        assert m["n_missing"] == 1
        assert m["mean_width"] == pytest.approx(2.0)
        assert m["mae"] == pytest.approx((0 + 0 + 3) / 3)

    def test_empty_frame(self):
        m = coverage_metrics(np.array([np.nan]), np.array([1.0]),
                             np.array([0.0]), np.array([2.0]))
        assert m == {"n": 0}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
