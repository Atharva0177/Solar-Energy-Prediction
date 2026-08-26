"""Export train-split scaler stats for serving the deep models (Phase 14).

The Phase 7/8 checkpoints (``models/{lstm,gru,transformer}_site_all_h1_v1.pt``)
store network weights only — the channel standardization and target
standardization were computed in memory during training and never persisted.
The API's recursive serving path needs them, so this script recomputes them
EXACTLY as ``scripts/train_sequence.py`` did: same features table, same
chronological split, same ``build_channel_matrix``/``fit_channel_scaler``
— and writes ``artifacts/{arch}/serving_scalers.json`` per architecture.

The stats are identical across the three architectures (same split, same
channel list); each arch gets its own copy so its artifact directory stays
self-contained. As a correctness check the script also reloads each
checkpoint and reproduces stored test-split predictions on a sample of
windows — max abs difference must be ~0 (identical weights + identical
inputs ⇒ identical outputs), which proves the exported scalers match what
the metrics in RESULTS.md were computed with.

Usage::

    conda run -n solar python scripts/export_sequence_scalers.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.data.splits import chronological_split
from src.models.sequence_model import (
    CHANNELS,
    RecurrentForecaster,
    TransformerForecaster,
    WindowDataset,
    build_channel_matrix,
    fit_channel_scaler,
)

ARCHS = ("lstm", "gru", "transformer")
VALIDATION_WINDOWS = 2048


def site_bounds_of(part: pd.DataFrame) -> list[tuple[int, int]]:
    """Same definition as train_sequence.py — contiguous per-site spans."""
    sid = part["site_id"].to_numpy()
    bounds, start = [], 0
    for i in range(1, len(sid) + 1):
        if i == len(sid) or sid[i] != sid[start]:
            bounds.append((start, i))
            start = i
    return bounds


def build_model(arch: str, params: dict, n_channels: int, lookback: int):
    if arch == "transformer":
        return TransformerForecaster(
            input_size=n_channels,
            d_model=int(params["d_model"]), nhead=int(params["nhead"]),
            num_layers=int(params["num_layers"]),
            dim_feedforward=int(params["dim_feedforward"]),
            dropout=float(params["dropout"]), max_len=lookback + 1)
    return RecurrentForecaster(arch, input_size=n_channels,
                               hidden_size=int(params["hidden_size"]),
                               num_layers=int(params["num_layers"]),
                               dropout=float(params["dropout"]))


def main() -> int:
    cfg = load_config()
    ratios = (cfg["training"]["train_ratio"], cfg["training"]["val_ratio"],
              cfg["training"]["test_ratio"])

    print(f"reading features table …", flush=True)
    df = pd.read_parquet(REPO_ROOT / cfg["paths"]["features_dir"])
    train, _val, test = chronological_split(df, ratios=ratios)
    del df

    tr_obs = train.loc[train["power"].notna()]
    y_mean = float(tr_obs["power"].mean())
    y_std = float(tr_obs["power"].std())
    del tr_obs

    M_tr = build_channel_matrix(train)
    ch_mean, ch_std = fit_channel_scaler(M_tr)
    del M_tr

    payload_common = {
        "y_mean": y_mean,
        "y_std": y_std,
        "channel_mean": [round(float(v), 8) for v in ch_mean],
        "channel_std": [round(float(v), 8) for v in ch_std],
        "channels": CHANNELS,
        "n_train_rows": int(len(train)),
        "fit": "train split only (recomputed identically to "
               "scripts/train_sequence.py; see Phase 14 D-022)",
    }

    # ---- validation: checkpoints + exported scalers must reproduce stored preds
    yn_test = ((test["power"].to_numpy(dtype=np.float64)) - y_mean) / y_std
    M_te = build_channel_matrix(test)
    ds_val_check = {}

    for arch in ARCHS:
        meta = json.loads((REPO_ROOT / "artifacts" / arch /
                           "run_metadata.json").read_text(encoding="utf-8"))
        params = meta["config"]["models"][arch]["params"]
        lb = int(meta["lookback_steps"])
        payload = dict(payload_common, lookback_steps=lb)
        out = REPO_ROOT / "artifacts" / arch / "serving_scalers.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[{arch}] wrote {out.relative_to(REPO_ROOT)} (lookback={lb})")

        ckpt = torch.load(REPO_ROOT / "models" / f"{arch}_site_all_h1_v1.pt",
                          map_location="cpu")
        model = build_model(arch, params, len(CHANNELS), lb)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        if lb not in ds_val_check:
            ds_val_check[lb] = WindowDataset(
                ((M_te - ch_mean) / ch_std).astype(np.float32),
                yn_test.astype(np.float32),
                site_bounds_of(test), lb)
        ds = ds_val_check[lb]
        n = min(VALIDATION_WINDOWS, len(ds))
        # validate on the same device class the stored predictions were made
        # with (CUDA fp32) — CPU differs by float accumulation noise over the
        # 97-step recurrence (~1e-2 kWh max), which is serving-irrelevant
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        from src.models.sequence_model import predict_windows
        reproduced = predict_windows(model, ds, batch_size=2048, device=dev,
                                     y_mean=y_mean, y_std=y_std)
        stored_path = REPO_ROOT / "artifacts" / arch / "predictions_test.parquet"
        stored = pd.read_parquet(stored_path)["prediction"].to_numpy()
        pos = ds.global_pos[:n]
        diff = np.abs(reproduced[:n] - stored[pos])
        ok = bool(np.nanmax(diff) < 1e-3)
        print(f"[{arch}] validation vs stored test predictions "
              f"(n={n}): max|Δ|={np.nanmax(diff):.2e} → "
              f"{'OK' if ok else 'MISMATCH'}")
        if not ok:
            print(f"[{arch}] FATAL: scalers do not reproduce recorded "
                   "predictions; refusing to ship.")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
