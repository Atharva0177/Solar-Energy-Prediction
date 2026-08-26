"""Phase 7 tests: sequence (LSTM/GRU) dataset construction + forecaster.

Covers the leakage surfaces unique to the sequence pipeline: channel matrix
building (mask must mirror observed power), train-only scaling, window
alignment (window ends AT t, target IS power(t), never crosses sites),
and unit inversion at prediction time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.sequence_model import (
    CHANNELS,
    RecurrentForecaster,
    WindowDataset,
    build_channel_matrix,
    fit_channel_scaler,
    predict_windows,
    train_sequence,
)


def synth(days: int = 6, n_sites: int = 2, seed: int = 0) -> pd.DataFrame:
    """Feature-table-shaped frame: temporal + weather + solar + power."""
    rng = np.random.default_rng(seed)
    frames = []
    for sid in range(1, n_sites + 1):
        ts = pd.date_range("2021-06-01", periods=96 * days, freq="15min")
        hour_angle = 2 * np.pi * (ts.hour * 60 + ts.minute) / 1440
        doy_angle = 2 * np.pi * ts.dayofyear / 365.25
        power = 5.0 * sid + 8.0 * np.clip(np.sin(hour_angle), 0, None) \
            + 2.0 * np.sin(doy_angle) + rng.normal(0, 0.05, len(ts))
        elev = 80.0 * np.clip(np.sin(hour_angle), 0, None)
        frames.append(pd.DataFrame({
            "site_id": sid, "timestamp": ts, "power": power,
            "temperature": 15 + rng.normal(0, 1, len(ts)),
            "humidity": np.clip(60 + rng.normal(0, 5, len(ts)), 0, 100),
            "wind_speed": np.abs(rng.normal(3, 1, len(ts))),
            "wind_dir_sin": rng.uniform(-1, 1, len(ts)),
            "wind_dir_cos": rng.uniform(-1, 1, len(ts)),
            "solar_elevation_deg": elev,
            "zenith_deg": 90 - elev,
        }))
    out = pd.concat(frames, ignore_index=True)
    out["hour"] = out.timestamp.dt.hour
    out["minute"] = out.timestamp.dt.minute
    day_frac = (out.hour * 60 + out.minute) / 1440
    out["sin_hour"] = np.sin(2 * np.pi * day_frac)
    out["cos_hour"] = np.cos(2 * np.pi * day_frac)
    out["day_of_year"] = out.timestamp.dt.dayofyear
    out["sin_day_of_year"] = np.sin(2 * np.pi * out.day_of_year / 365.25)
    out["cos_day_of_year"] = np.cos(2 * np.pi * out.day_of_year / 365.25)
    return out


def punch_holes(df: pd.DataFrame, frac: float = 0.2, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    drop = rng.random(len(out)) < frac
    out.loc[drop, "power"] = np.nan
    return out


# ---------------------------------------------------------------------------
# channel matrix
# ---------------------------------------------------------------------------


class TestChannelMatrix:
    def test_shape_and_channel_order(self):
        df = synth(days=2)
        m = build_channel_matrix(df)
        assert m.shape == (len(df), len(CHANNELS))  # 2 sites x 192
        assert CHANNELS[0] == "power" and CHANNELS[1] == "power_observed"

    def test_power_channel_zero_filled_and_mask_mirrors_observed(self):
        df = punch_holes(synth(days=2))
        m = build_channel_matrix(df)
        obs = df["power"].notna().to_numpy()
        np.testing.assert_array_equal(m[:, 1].astype(bool), obs)
        assert np.isfinite(m).all(), "NaN must be filled (network cannot ingest)"
        # zero-filled power rows carry 0 exactly
        np.testing.assert_array_equal(m[~obs, 0], 0.0)
        np.testing.assert_allclose(m[obs, 0], df.loc[obs, "power"].to_numpy())

    def test_weather_nan_filled(self):
        df = synth(days=2)
        df.loc[df.index[:10], "temperature"] = np.nan
        m = build_channel_matrix(df)
        assert np.isfinite(m).all()


# ---------------------------------------------------------------------------
# scaling (train-only statistics)
# ---------------------------------------------------------------------------


class TestScaler:
    def test_fit_on_train_applies_to_eval(self):
        tr, ev = synth(days=4)[:384], synth(days=4)[384:]
        M_tr = build_channel_matrix(tr)
        mean, std = fit_channel_scaler(M_tr)
        assert mean.shape == std.shape == (M_tr.shape[1],)
        Z_tr = (M_tr - mean) / std
        Z_ev = (build_channel_matrix(ev) - mean) / std
        # mask channel untouched by scaling
        np.testing.assert_array_equal(Z_tr[:, 1], M_tr[:, 1])
        np.testing.assert_array_equal(Z_ev[:, 1], M_ev_mask := build_channel_matrix(ev)[:, 1])
        # standardized train power has ~unit variance over observed rows
        assert Z_tr[tr["power"].notna().to_numpy(), 0].std() == pytest.approx(1.0, abs=1e-3)
        assert np.isfinite(Z_ev).all()

    def test_constant_channel_gets_floor_std(self):
        M = np.zeros((10, len(CHANNELS)), dtype=np.float32)
        _, std = fit_channel_scaler(M)
        assert (std > 0).all()


# ---------------------------------------------------------------------------
# window dataset alignment
# ---------------------------------------------------------------------------


class TestWindowDataset:
    def _ds(self, lookback=8):
        df = punch_holes(synth(days=3))
        M = build_channel_matrix(df)
        y = df["power"].to_numpy(dtype=np.float32)
        bounds = [(0, 288), (288, 576)]
        return WindowDataset(M, y, site_bounds=bounds, lookback=lookback), M, y, df

    def test_only_valid_indices_sampled(self):
        ds, *_ = self._ds()
        assert len(ds) > 0
        y = self._ds()[2]
        for i in range(min(len(ds), 20)):
            g = ds.global_pos[i]
            assert not np.isnan(y[g])

    def test_window_alignment_last_step_is_t(self):
        ds, M, y, df = self._ds(lookback=8)
        X, tgt = ds[0]
        assert X.shape == (9, len(CHANNELS))  # lookback + current step
        g = ds.global_pos[0]
        # covariate channels match the frame at t ...
        np.testing.assert_allclose(X[-1, 2:], M[g, 2:], rtol=1e-6)
        # ... but power/mask are zeroed at t (target-leak guard)
        assert X[-1, 0].item() == 0.0
        assert X[-1, 1].item() == 0.0
        np.testing.assert_allclose(X[-2], M[g - 1], rtol=1e-6)  # history intact
        assert tgt == pytest.approx(y[g])
        assert g == 8  # first valid = site start + lookback

    def test_gather_matches_getitem_and_masks_current_power(self):
        """gather ≡ stacked __getitem__, incl. the current-step power masking."""
        ds, M, y, _ = self._ds(lookback=8)
        idx = list(range(0, len(ds), 7))[:25]
        Xg, yg = ds.gather(idx)
        assert Xg.shape == (len(idx), 9, len(CHANNELS))
        np.testing.assert_array_equal(Xg[:, -1, 0].numpy(), 0.0)   # power(t)
        np.testing.assert_array_equal(Xg[:, -1, 1].numpy(), 0.0)   # mask(t)
        # guard must not blank history: observed-marker channel still fires
        assert bool((Xg[:, :-1, 1] == 1.0).any())
        for k, i in enumerate(idx):
            Xs, ys = ds[i]
            np.testing.assert_allclose(Xg[k].numpy(), Xs.numpy(), rtol=1e-6)
            assert float(yg[k]) == pytest.approx(float(ys))

    def test_history_rows_keep_real_power(self):
        """Leak guard must not blank the history channels (regression test for
        the strided-view aliasing trap: gather mutates a COPY, never matrix)."""
        ds, M, y, _ = self._ds(lookback=8)
        g = int(ds.global_pos[3])
        X, _ = ds.gather([3])
        np.testing.assert_allclose(X[0, :-1, 0].numpy(), M[g - 8:g, 0], rtol=1e-6)
        # dataset matrix untouched by gather
        np.testing.assert_allclose(ds.matrix[g, 0], M[g, 0], rtol=1e-6)

    def test_never_crosses_site_boundary(self):
        ds, M, *_ = self._ds(lookback=8)
        for i in range(len(ds)):
            g = ds.global_pos[i]
            if g < 288:  # first site
                assert g - 8 >= 0
            else:
                assert g - 8 >= 288


# ---------------------------------------------------------------------------
# forecaster module
# ---------------------------------------------------------------------------


class TestRecurrentForecaster:
    @pytest.mark.parametrize("rnn", ["lstm", "gru"])
    def test_forward_shape_scalar(self, rnn):
        model = RecurrentForecaster(rnn_type=rnn, input_size=len(CHANNELS),
                                    hidden_size=16, num_layers=1, dropout=0.0)
        X = np.random.default_rng(0).normal(size=(4, 9, len(CHANNELS))).astype(np.float32)
        out = model(__import__("torch").from_numpy(X))
        assert out.shape == (4,)

    def test_gru_lighter_than_lstm(self):
        kw = dict(input_size=len(CHANNELS), hidden_size=32, num_layers=1, dropout=0.0)
        lstm = RecurrentForecaster(rnn_type="lstm", **kw)
        gru = RecurrentForecaster(rnn_type="gru", **kw)
        n_lstm = sum(p.numel() for p in lstm.parameters())
        n_gru = sum(p.numel() for p in gru.parameters())
        assert n_gru < n_lstm

    def test_unknown_rnn_rejected(self):
        with pytest.raises(ValueError):
            RecurrentForecaster(rnn_type="transformer", input_size=4,
                                hidden_size=8, num_layers=1, dropout=0.0)


# ---------------------------------------------------------------------------
# training loop (tiny, CPU)
# ---------------------------------------------------------------------------

TINY_PARAMS = {"lr": 5e-3, "weight_decay": 0.0, "batch_size": 64,
               "max_epochs": 20, "patience": 5}


@pytest.fixture(scope="module")
def tiny_data():
    import torch

    df = punch_holes(synth(days=14, seed=11), frac=0.25)
    M = build_channel_matrix(df).astype(np.float32)
    y = df["power"].to_numpy(dtype=np.float32)
    cut = int(len(df) * 0.7)
    bounds_tr = [(0, cut // 2), (cut // 2, cut)]
    bounds_va = [(cut, len(df))]
    mean, std = fit_channel_scaler(M[:cut])
    Z = ((M - mean) / std).astype(np.float32)
    y_mean = float(np.nanmean(y[:cut]))
    y_std = float(np.nanstd(y[:cut]))
    yn = (y - y_mean) / y_std
    tr = WindowDataset(Z, yn, bounds_tr, lookback=16)
    va = WindowDataset(Z, yn, bounds_va, lookback=16)
    return tr, va, y_mean, y_std, torch


class TestTrainLoop:
    def test_loss_decreases_and_predict_inverts_units(self, tiny_data):
        tr, va, y_mean, y_std, torch = tiny_data
        torch.manual_seed(7)  # weight init draws from global RNG — reseed per build
        model, info = train_sequence(
            RecurrentForecaster("lstm", len(CHANNELS), 24, 1, 0.0),
            tr, va, params=TINY_PARAMS, seed=7, device="cpu")
        assert info["epochs_ran"] <= TINY_PARAMS["max_epochs"]
        assert info["best_val_rmse"] < info.get("first_val_rmse", np.inf)
        pred = predict_windows(model, va, batch_size=256, device="cpu",
                               y_mean=y_mean, y_std=y_std)
        assert len(pred) == len(va)
        # back in kWh space, sane magnitudes
        assert np.isfinite(pred).all()
        assert np.abs(pred).max() < 100
        truth = np.array([va[i][1] for i in range(len(va))]) * y_std + y_mean
        rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
        spread = float(np.std(truth))
        assert rmse < spread, "must beat climatological mean on learnable sinusoid"

    def test_deterministic_given_seed(self, tiny_data):
        tr, va, *_ , torch = tiny_data
        kw = dict(input_size=len(CHANNELS), hidden_size=8, num_layers=1, dropout=0.0)
        torch.manual_seed(3)  # weight init draws from global RNG — reseed per build
        m1, _ = train_sequence(RecurrentForecaster("gru", **kw), tr, va,
                               params={**TINY_PARAMS, "max_epochs": 1}, seed=3, device="cpu")
        torch.manual_seed(3)
        m2, _ = train_sequence(RecurrentForecaster("gru", **kw), tr, va,
                               params={**TINY_PARAMS, "max_epochs": 1}, seed=3, device="cpu")
        p1 = predict_windows(m1, va, 256, "cpu", y_mean=0.0, y_std=1.0)
        p2 = predict_windows(m2, va, 256, "cpu", y_mean=0.0, y_std=1.0)
        np.testing.assert_allclose(p1, p2, atol=1e-5)


# ---------------------------------------------------------------------------
# config wiring
# ---------------------------------------------------------------------------


class TestConfigs:
    def test_sequence_blocks_present_when_enabled(self):
        from src.config import load_config

        cfg = load_config()
        for name in ("lstm", "gru"):
            block = cfg["models"][name]
            assert isinstance(block, dict)
            if block.get("enabled"):
                p = block["params"]
                for key in ("lookback_steps", "hidden_size", "num_layers",
                            "dropout", "lr", "batch_size", "max_epochs", "patience"):
                    assert key in p, f"{name}.{key} missing"
