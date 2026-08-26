"""Phase 8 tests: Transformer forecaster over the shared Phase 7 windows.

The leakage-critical machinery (channel matrix, mask channel, current-step
power guard, train-only scaling) is shared with LSTM/GRU and already covered
by ``tests/test_sequence_model.py``. Here we verify the new encoder: shapes,
the sinusoidal position buffer, integration with masked ``gather`` batches,
learning on the tiny CPU task, seed determinism, and config wiring.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.sequence_model import (
    CHANNELS,
    TransformerForecaster,
    WindowDataset,
)
from test_sequence_model import TINY_PARAMS, punch_holes, synth


def test_forward_shape_scalar():
    import torch

    model = TransformerForecaster(input_size=len(CHANNELS), d_model=32,
                                  nhead=4, num_layers=1, dim_feedforward=64,
                                  max_len=9)
    X = torch.from_numpy(
        np.random.default_rng(0).normal(size=(4, 9, len(CHANNELS))).astype(np.float32))
    out = model(X)
    assert out.shape == (4,)


def test_d_model_nhead_validated():
    with pytest.raises(ValueError):
        TransformerForecaster(input_size=13, d_model=30, nhead=8)


def test_positional_encoding_sinusoidal_properties():
    import torch

    model = TransformerForecaster(input_size=13, d_model=16, nhead=4,
                                  max_len=97)
    pe = model.pos_enc.squeeze(0)                     # (97, 16)
    assert pe.shape == (97, 16)
    assert torch.isfinite(pe).all()
    # adjacent positions differ; values bounded in [-1, 1]
    assert float((pe[1:] - pe[:-1]).abs().max()) > 0
    assert float(pe.abs().max()) <= 1.0 + 1e-6


def test_position_buffer_not_trained():
    """Sinusoidal encoding is a registered buffer — excluded from optimizer."""
    model = TransformerForecaster(input_size=13, d_model=16, nhead=4)
    names = [n for n, _ in model.named_parameters()]
    assert not any("pos_enc" in n for n in names)


@pytest.fixture(scope="module")
def tiny_data():
    import torch

    df = punch_holes(synth(days=14, seed=11), frac=0.25)
    from src.models.sequence_model import build_channel_matrix, fit_channel_scaler

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


class TestTransformerTrainLoop:
    def test_learns_and_inverts_units(self, tiny_data):
        from src.models.sequence_model import predict_windows, train_sequence

        tr, va, y_mean, y_std, torch = tiny_data
        params = {**TINY_PARAMS, "batch_size": 128}
        model, info = train_sequence(
            TransformerForecaster(len(CHANNELS), d_model=24, nhead=4,
                                  num_layers=1, dim_feedforward=48,
                                  max_len=17),
            tr, va, params=params, seed=7, device="cpu")
        assert info["epochs_ran"] <= params["max_epochs"]
        assert info["best_val_rmse"] < info.get("first_val_rmse", np.inf)
        pred = predict_windows(model, va, batch_size=256, device="cpu",
                               y_mean=y_mean, y_std=y_std)
        truth = np.array([va[i][1] for i in range(len(va))]) * y_std + y_mean
        rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
        assert rmse < float(np.std(truth)), \
            "must beat climatological mean on learnable sinusoid"

    def test_deterministic_given_seed(self, tiny_data):
        from src.models.sequence_model import predict_windows, train_sequence

        tr, va, *_ , torch = tiny_data
        kw = dict(d_model=16, nhead=4, num_layers=1, dim_feedforward=32,
                  max_len=17)

        def run():
            torch.manual_seed(3)  # weight init draws global RNG before train reseeds
            m, _ = train_sequence(TransformerForecaster(len(CHANNELS), **kw),
                                  tr, va, params={**TINY_PARAMS, "max_epochs": 1},
                                  seed=3, device="cpu")
            return predict_windows(m, va, 256, "cpu", y_mean=0.0, y_std=1.0)

        np.testing.assert_allclose(run(), run(), atol=1e-5)


class TestConfigWiring:
    def test_transformer_params_present_when_enabled(self):
        from src.config import load_config

        block = load_config()["models"]["transformer"]
        if block.get("enabled"):
            p = block["params"]
            for key in ("lookback_steps", "d_model", "nhead", "num_layers",
                        "dim_feedforward", "dropout", "lr", "batch_size",
                        "max_epochs", "patience"):
                assert key in p, f"transformer.{key} missing"
