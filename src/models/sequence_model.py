"""Sequence forecasters — LSTM / GRU (Phase 7) + Transformer (Phase 8), PRD §23.

Framing: sliding windows of ``lookback`` steps (+ current step covariates)
per site → predict ``power(t)`` (same single-step target as Phase 6 XGBoost,
so metrics stay comparable).

Design points:

* **NaN strategy** differs from XGBoost (D-013): networks cannot ingest NaN,
  so the channel matrix zero-fills everything and adds an explicit
  ``power_observed`` 0/1 mask channel — the net learns missing-vs-zero.
* Scalers and target normalization statistics are fit on the TRAIN split
  only (leakage guard); eval frames are transformed with train stats.
* Windows never cross site boundaries (``site_bounds``); a window ending at
  t covers [t−lookback, t] and its target IS power(t) — no positional drift
  across the 52k grid holes because positions come from the actual frame.
* **Target-leak guard**: the power + observation-mask channels are zeroed at
  the FINAL step (t) of every window — generation at t is what we predict and
  cannot be observed when the forecast is issued. History steps [t−L, t−1]
  keep real values (equivalent to XGBoost's lag/rolling inputs). First run
  without this guard scored R²=1.000 by copying the target through step t;
  caught because it beat XGBoost ~10× on MAE.
* Training loop (PRD §49): AdamW + gradient clipping, ReduceLROnPlateau LR
  scheduling on val RMSE, per-epoch checkpointing of the best model to disk,
  best-weight restoration before returning, mixed precision when CUDA is
  available, plain-CPU fallback for tests/small runs. Per-epoch history
  (val RMSE + LR) is returned so the orchestrator can stream it to MLflow.
* Phase 8 adds ``TransformerForecaster`` on the SAME windows/guard/loop —
  only the encoder differs (self-attention vs recurrence).
* Batches are gathered with strided views (``WindowDataset.gather``) —
  per-item ``__getitem__`` collate would dominate wall-clock at ~1.9M windows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# channel layout of the model input matrix (column order is contract:
# tests assert index 0/1 are power / power_observed)
CHANNELS = [
    "power", "power_observed",
    "sin_hour", "cos_hour", "sin_day_of_year", "cos_day_of_year",
    "temperature", "humidity", "wind_speed", "wind_dir_sin", "wind_dir_cos",
    "solar_elevation_deg", "zenith_deg",
]
MASK_IDX = 1
POWER_IDX = 0
_STD_FLOOR = 1e-6


def build_channel_matrix(df) -> np.ndarray:
    """(N, F) float32 matrix; NaN→0 everywhere, mask channel mirrors power."""
    M = np.zeros((len(df), len(CHANNELS)), dtype=np.float32)
    for j, name in enumerate(CHANNELS):
        if name == "power_observed":
            # derived from the target column — never expected as a df column
            M[:, j] = np.isfinite(pd_to_float(df["power"])).astype(np.float32)
            continue
        vals = pd_to_float(df[name])
        ok = np.isfinite(vals)
        M[ok, j] = vals[ok]
    return M


def pd_to_float(col) -> np.ndarray:
    import pandas as pd

    return pd.to_numeric(col, errors="coerce").to_numpy(dtype=np.float64)


def fit_channel_scaler(M_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std from TRAIN rows only; mask channel passes through."""
    mean = np.zeros(M_train.shape[1], dtype=np.float64)
    std = np.ones(M_train.shape[1], dtype=np.float64)
    for j in range(M_train.shape[1]):
        if j == MASK_IDX:
            continue
        v = M_train[:, j]
        v = v[np.isfinite(v)]
        if len(v):
            mean[j] = v.mean()
            std[j] = max(v.std(), _STD_FLOOR)
    return mean, std


def _mask_current_step(W: np.ndarray) -> None:
    """Zero power + mask channels at the LAST step in place (leak guard).

    ``W`` is (..., L+1, F) with the current step last along axis −2. Only
    channels POWER_IDX / MASK_IDX are touched — exogenous covariates at t
    (calendar, weather, solar position) stay, matching XGBoost's inputs.
    """
    W[..., -1, POWER_IDX] = 0.0
    W[..., -1, MASK_IDX] = 0.0


class WindowDataset(Dataset):
    """Sliding windows over a standardized channel matrix.

    ``site_bounds`` lists (start, end) row spans per site so windows never
    mix sites. Samples exist only where ``g - start >= lookback`` AND the
    target is observed (non-NaN).
    """

    def __init__(self, matrix: np.ndarray, targets: np.ndarray,
                 site_bounds: list[tuple[int, int]], lookback: int):
        self.matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.lookback = int(lookback)
        pos = []
        for start, end in site_bounds:
            lo = start + self.lookback
            for g in range(lo, end):
                if not np.isnan(self.targets[g]):
                    pos.append(g)
        self.global_pos = np.asarray(pos, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.global_pos)

    def __getitem__(self, i: int):
        g = int(self.global_pos[i])
        X = self.matrix[g - self.lookback:g + 1].copy()   # (L+1, F)
        _mask_current_step(X)
        return torch.from_numpy(X), float(self.targets[g])

    def gather(self, indices) -> tuple[torch.Tensor, torch.Tensor]:
        """Vectorized fetch of many windows: (B, L+1, F) + (B,) tensors.

        Equivalent to stacking ``__getitem__`` over ``indices`` (unit-tested)
        but one strided view + transpose instead of a Python loop. The
        ascontiguousarray copy happens BEFORE masking so the leak guard can
        edit in place safely (strided view aliases ``self.matrix``) and both
        accessors share the same (...steps, channels) layout for the guard.
        """
        pos = self.global_pos[np.asarray(indices, dtype=np.int64)]
        # view i covers rows [i, i+L]; window at g starts at g - L
        windows = np.lib.stride_tricks.sliding_window_view(
            self.matrix, self.lookback + 1, axis=0)[pos - self.lookback]
        # contiguous copy BEFORE masking — strided view aliases self.matrix
        X = np.ascontiguousarray(windows.transpose(0, 2, 1))   # (B, L+1, F)
        _mask_current_step(X)
        return torch.from_numpy(X), torch.from_numpy(self.targets[pos].copy())


class RecurrentForecaster(nn.Module):
    """LSTM or GRU encoder → MLP head → scalar power."""

    def __init__(self, rnn_type: str, input_size: int, hidden_size: int,
                 num_layers: int, dropout: float):
        super().__init__()
        rnn_type = rnn_type.lower()
        if rnn_type not in ("lstm", "gru"):
            raise ValueError(f"unsupported rnn_type: {rnn_type!r}")
        cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = cls(input_size=input_size, hidden_size=hidden_size,
                       num_layers=num_layers, batch_first=True,
                       dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:  # X: (B, L+1, F)
        out, _ = self.rnn(X)
        return self.head(out[:, -1]).squeeze(-1)          # (B,)


class TransformerForecaster(nn.Module):
    """Transformer encoder over the same windows → scalar power (PRD §23).

    Same contract as ``RecurrentForecaster``: input (B, L+1, F) from
    ``WindowDataset`` (current-step power already masked), output (B,)
    predicted power. Readout is the LAST position (= step t, whose power
    channels are zeroed — covariates at t are legitimate inputs). Fixed
    sinusoidal positional encoding (no learned positions to overfit on
    ~800k windows); pre-LN blocks; no causal mask needed because every
    position in a window is ≤ t by construction.
    """

    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 2, dim_feedforward: int = 256,
                 dropout: float = 0.1, max_len: int = 97):
        super().__init__()
        if d_model % nhead:
            raise ValueError(f"d_model {d_model} not divisible by nhead {nhead}")
        self.d_model = d_model
        self.input_proj = nn.Linear(input_size, d_model)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pos_enc", pe.unsqueeze(0))          # (1, max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        # enable_nested_tensor=False: incompatible with norm_first and only
        # produces a UserWarning otherwise
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, 1),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:           # X: (B, L+1, F)
        h = self.input_proj(X) * (self.d_model ** 0.5)
        h = h + self.pos_enc[:, : X.size(1)]
        h = self.norm(self.encoder(h))
        return self.head(h[:, -1]).squeeze(-1)                    # (B,)


def _save_checkpoint(path: Path, model: nn.Module, opt, amp_scaler,
                     epoch: int, val_rmse: float) -> None:
    """Atomic-ish checkpoint of the current best state (PRD §49)."""
    payload = {
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state_dict": opt.state_dict(),
        "amp_scaler_state_dict": amp_scaler.state_dict(),
        "epoch": epoch,
        "val_rmse_normalized": float(val_rmse),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def train_sequence(model: RecurrentForecaster, train_ds: Dataset, val_ds: Dataset,
                   params: dict, seed: int = 42, device: str = "cuda",
                   verbose: bool = False,
                   checkpoint_path: Optional[Path] = None) -> tuple:
    """AdamW + MSE on standardized targets; early-stop on val RMSE (PRD §49).

    Gradient clipping + ReduceLROnPlateau scheduling each epoch; whenever val
    improves, weights are checkpointed to ``checkpoint_path`` (if given) and
    kept in memory; best weights are restored before returning. ``info``
    carries epochs_ran / first_val_rmse / best_val_rmse (normalized space),
    per-epoch history [{epoch, lr, val_rmse}] for MLflow streaming, and the
    final LR after scheduling.
    """
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = model.to(dev)

    gen = torch.Generator().manual_seed(seed)
    bs = int(params["batch_size"])
    opt = torch.optim.AdamW(model.parameters(), lr=float(params["lr"]),
                            weight_decay=float(params.get("weight_decay", 0.0)))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min",
        factor=float(params.get("scheduler_factor", 0.5)),
        patience=int(params.get("scheduler_patience", 1)))
    clip = float(params.get("grad_clip_norm", 1.0))
    loss_fn = nn.MSELoss()
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    patience, best_rmse, best_state = int(params["patience"]), np.inf, None
    history, stale = [], 0
    epochs_ran = first_rmse = 0
    for epoch in range(int(params["max_epochs"])):
        model.train()
        perm = torch.randperm(len(train_ds), generator=gen).numpy()
        for s in range(0, len(perm), bs):
            Xb, yb = train_ds.gather(perm[s:s + bs])
            Xb, yb = Xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(Xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt)
            scaler.update()

        rmse = _val_rmse(model, val_ds, dev)
        first_rmse = first_rmse or rmse
        epochs_ran = epoch + 1
        lr_now = float(opt.param_groups[0]["lr"])
        history.append({"epoch": epochs_ran, "lr": lr_now, "val_rmse": round(rmse, 6)})
        if verbose:
            print(f"epoch {epochs_ran}: val_rmse(norm)={rmse:.4f} lr={lr_now:.2e}")
        if rmse < best_rmse - 1e-5:
            best_rmse = rmse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                _save_checkpoint(checkpoint_path, model, opt, scaler, epochs_ran, rmse)
        else:
            stale += 1
            if stale >= patience:
                break
        sched.step(rmse)

    if best_state is not None:
        model.load_state_dict(best_state)
    info = {"epochs_ran": epochs_ran, "best_val_rmse": float(best_rmse),
            "first_val_rmse": float(first_rmse), "history": history,
            "final_lr": float(opt.param_groups[0]["lr"]),
            "grad_clip_norm": clip}
    return model, info


def _val_rmse(model, ds: WindowDataset, dev: torch.device) -> float:
    model.eval()
    se = n = 0
    with torch.no_grad():
        for s in range(0, len(ds), 8192):
            Xb, yb = ds.gather(np.arange(s, min(s + 8192, len(ds))))
            Xb, yb = Xb.to(dev), yb.to(dev)
            se += float(((model(Xb) - yb) ** 2).sum())
            n += len(yb)
    return (max(se, 0.0) / max(n, 1)) ** 0.5


@torch.no_grad()
def predict_windows(model, ds: WindowDataset, batch_size: int, device: str,
                    y_mean: float, y_std: float) -> np.ndarray:
    """Predictions in kWh (inverse of target standardization)."""
    dev = torch.device("cuda" if device != "cpu" and torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()
    out = []
    for s in range(0, len(ds), batch_size):
        Xb, _ = ds.gather(np.arange(s, min(s + batch_size, len(ds))))
        out.append(model(Xb.to(dev)).float().cpu().numpy())
    return np.concatenate(out) * y_std + y_mean if out else np.zeros(0)
