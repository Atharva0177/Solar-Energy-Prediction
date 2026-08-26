"""Phase 7/8 orchestrator: train + evaluate LSTM/GRU/Transformer sequence models.

Usage: ``conda run -n solar python scripts/train_sequence.py --arch lstm|gru|transformer``

Protocol identical to Phase 6 (D-011/D-013): canonical per-site chronological
split, single-step target, same nRMSE denominators — numbers directly
comparable to `artifacts/baselines/baseline_metrics.csv` and
`artifacts/xgboost/metrics.csv`. Sequence specifics (D-014): lookback window
of covariate+power-history channels with an observation mask channel;
scalers/target stats fit on train only; GPU + AMP; best checkpoint restored.

Artifacts per arch (``lstm`` example):

* ``models/lstm_site_all_h1_v1.pt``          — best state_dict checkpoint
* ``artifacts/lstm/metrics.csv``             — ALL + per-site, val/test
* ``artifacts/lstm/predictions_test.parquet``
* ``artifacts/lstm/run_metadata.json``
* MLflow run ``lstm-site-all-h1-v1``
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# PRD §30 mandates local mlruns/ file tracking; recent MLflow blocks that
# backend unless explicitly opted in.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import torch

from src.config import load_config
from src.data.splits import chronological_split
from src.models.sequence_model import (
    CHANNELS,
    WindowDataset,
    build_channel_matrix,
    fit_channel_scaler,
    predict_windows,
    train_sequence,
)
from src.training.evaluate import regression_metrics

KEEP_COLS_IN_PREDICTIONS = ["site_id", "timestamp", "power", "is_daylight"]


def fmt(v) -> str:
    return "n/a" if pd.isna(v) else f"{v:.3f}"


def site_bounds(part: pd.DataFrame) -> list[tuple[int, int]]:
    """Contiguous (start, end) row spans per site — raises if interleaved."""
    sid = part["site_id"].to_numpy()
    bounds, start = [], 0
    for i in range(1, len(sid) + 1):
        if i == len(sid) or sid[i] != sid[start]:
            bounds.append((start, i))
            start = i
    return bounds


def score_partitions(model, parts: dict[str, pd.DataFrame], datasets: dict,
                     denom_all, site_ranges, y_mean, y_std, device):
    rows, preds_by_split = [], {}
    for split_name, part in parts.items():
        ds = datasets[split_name]
        pred_valid = predict_windows(model, ds, batch_size=2048, device=device,
                                     y_mean=y_mean, y_std=y_std)
        # scatter window predictions back onto frame positions
        pred = np.full(len(part), np.nan, dtype=np.float64)
        pred[ds.global_pos] = pred_valid
        preds_by_split[split_name] = pred

        m = regression_metrics(part["power"].to_numpy(), pred,
                               daylight=part["is_daylight"].to_numpy(),
                               denom=denom_all)
        rows.append({"split": split_name, "scope": "ALL", "site_id": "", **m})
        for sid, sub in part.groupby("site_id", observed=True):
            ms = regression_metrics(sub["power"].to_numpy(), pred[sub.index],
                                    daylight=sub["is_daylight"].to_numpy(),
                                    denom=site_ranges.get(sid))
            rows.append({"split": split_name, "scope": "SITE", "site_id": sid, **ms})
    return pd.DataFrame(rows), preds_by_split


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=("lstm", "gru", "transformer"), required=True)
    args = ap.parse_args()
    arch = args.arch

    cfg = load_config()
    tcfg, mcfg = cfg["training"], cfg["models"][arch]
    assert mcfg["enabled"], f"{arch} disabled in configs/models.yaml"
    params = dict(mcfg["params"])

    out_dir = REPO_ROOT / "artifacts" / arch
    model_path = REPO_ROOT / "models" / f"{arch}_site_all_h1_v1.pt"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = int(tcfg["seed"])
    ratios = (tcfg["train_ratio"], tcfg["val_ratio"], tcfg["test_ratio"])

    # ---- data -----------------------------------------------------------------
    df = pd.read_parquet(REPO_ROOT / cfg["paths"]["features_dir"])
    print(f"features: {df.shape[0]:,} x {df.shape[1]} | device={device} "
          f"| torch {torch.__version__}")
    if device == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    train, val, test = chronological_split(df, ratios=ratios)
    del df
    parts = {"val": val, "test": test}
    print(f"split: train={len(train):,} val={len(val):,} test={len(test):,}")

    # nRMSE denominators — identical definition to run_baselines.py (D-011)
    tr_obs = train.loc[train["power"].notna()]
    denom_all = float(tr_obs["power"].max() - tr_obs["power"].min())
    per_range = tr_obs.groupby("site_id", observed=True)["power"].agg(
        lambda s: float(s.max() - s.min()))
    site_ranges = {sid: (v if v > 0 else None) for sid, v in per_range.items()}
    y_mean = float(tr_obs["power"].mean())
    y_std = float(tr_obs["power"].std())
    del tr_obs

    # ---- channels + scaling (train-only stats) ----------------------------------
    M_tr = build_channel_matrix(train)
    mean, std = fit_channel_scaler(M_tr)
    y_tr = train["power"].to_numpy(dtype=np.float64)
    yn_tr = (y_tr - y_mean) / y_std
    lookback = int(params["lookback_steps"])
    ds_train = WindowDataset(((M_tr - mean) / std).astype(np.float32),
                             yn_tr.astype(np.float32), site_bounds(train), lookback)
    print(f"train windows: {len(ds_train):,} | lookback={lookback} "
          f"channels={len(CHANNELS)}")

    datasets, frames_std = {}, {}
    for split_name, part in parts.items():
        M = build_channel_matrix(part)
        frames_std[split_name] = M
        yn = ((part["power"].to_numpy(dtype=np.float64)) - y_mean) / y_std
        datasets[split_name] = WindowDataset(
            ((M - mean) / std).astype(np.float32), yn.astype(np.float32),
            site_bounds(part), lookback)

    # ---- train ------------------------------------------------------------------
    from src.models.sequence_model import RecurrentForecaster, TransformerForecaster

    torch.manual_seed(seed)
    if arch == "transformer":
        model = TransformerForecaster(
            input_size=len(CHANNELS),
            d_model=int(params["d_model"]), nhead=int(params["nhead"]),
            num_layers=int(params["num_layers"]),
            dim_feedforward=int(params["dim_feedforward"]),
            dropout=float(params["dropout"]), max_len=lookback + 1)
    else:
        model = RecurrentForecaster(arch, input_size=len(CHANNELS),
                                    hidden_size=int(params["hidden_size"]),
                                    num_layers=int(params["num_layers"]),
                                    dropout=float(params["dropout"]))
    print(f"model: {type(model).__name__} | params="
          f"{sum(p.numel() for p in model.parameters()):,}")
    t0 = time.perf_counter()
    model, info = train_sequence(model, ds_train, datasets["val"], params=params,
                                 seed=seed, device=device, verbose=True,
                                 checkpoint_path=model_path)
    fit_s = time.perf_counter() - t0
    print(f"trained in {fit_s:.1f}s | epochs={info['epochs_ran']} "
          f"best_val_rmse(norm)={info['best_val_rmse']:.4f}")
    # best-epoch checkpoint already written by train_sequence (PRD §49)

    # ---- evaluate ------------------------------------------------------------------
    metrics, preds = score_partitions(model, parts, datasets, denom_all,
                                      site_ranges, y_mean, y_std, device)
    cols = ["split", "scope", "site_id", "n_eval", "n_missing", "mae", "rmse",
            "r2", "nrmse", "daylight_n", "daylight_mae", "daylight_nrmse"]
    metrics_path = out_dir / "metrics.csv"
    metrics[cols].to_csv(metrics_path, index=False)

    test_part = test[KEEP_COLS_IN_PREDICTIONS].copy()
    test_part["prediction"] = preds["test"]
    test_part.to_parquet(out_dir / "predictions_test.parquet",
                         engine="pyarrow", index=False)

    # ---- metadata -------------------------------------------------------------------
    from src.models.xgboost_model import dataset_fingerprint

    meta = {
        "run_name": f"{arch}-site-all-h1-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": "single-step forecast, horizon = 1 x 15-min, sequence model",
        "architecture": arch,
        "config": cfg,
        "seed": seed,
        "device": str(torch.cuda.get_device_name(0)) if device == "cuda" else "cpu",
        "mixed_precision": device == "cuda",
        "channels": CHANNELS,
        "lookback_steps": lookback,
        "n_windows_train": len(ds_train),
        "n_train_val_test": [len(train), len(val), len(test)],
        "epochs_ran": info["epochs_ran"],
        "best_val_rmse_normalized": round(info["best_val_rmse"], 5),
        "training_history": info["history"],
        "final_lr": info["final_lr"],
        "grad_clip_norm": info["grad_clip_norm"],
        "target_standardization": {"mean": y_mean, "std": y_std},
        "channel_scaler_fit_on": "train split only",
        "nrmse_denominator_pooled_train_range_kwh": round(denom_all, 3),
        "dataset_fingerprint_features": dataset_fingerprint(
            REPO_ROOT / cfg["paths"]["features_dir"]),
        "training_seconds": round(fit_s, 2),
        "python_version": sys.version.split()[0],
        "package_versions": _pkg_versions("torch", "pandas", "numpy", "mlflow"),
        "artifacts": {
            "checkpoint": str(model_path.relative_to(REPO_ROOT)),
            "metrics": str(metrics_path.relative_to(REPO_ROOT)),
            "predictions": str((out_dir / "predictions_test.parquet").relative_to(REPO_ROOT)),
        },
    }
    meta_path = out_dir / "run_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote artifacts under {out_dir.name}/ + {model_path.name}")

    # ---- MLflow (PRD §30) -------------------------------------------------------------
    mlflow.set_tracking_uri((REPO_ROOT / "mlruns").as_uri())
    mlflow.set_experiment("unisolar")
    with mlflow.start_run(run_name=f"{arch}-site-all-h1-v1"):
        mlflow.log_params({
            "model": arch, "site_scope": "all", "horizon_steps": 1,
            "cadence_minutes": cfg["forecast"]["cadence_minutes"],
            "lookback_steps": lookback, "input_channels": len(CHANNELS),
            "seed": seed, **{f"param_{k}": v for k, v in params.items()},
            "dataset_fingerprint_features": meta["dataset_fingerprint_features"],
        })
        test_all = metrics[(metrics.scope == "ALL") & (metrics.split == "test")].iloc[0]
        val_all = metrics[(metrics.scope == "ALL") & (metrics.split == "val")].iloc[0]
        mlflow.log_metrics({
            **{f"test_{k}": float(test_all[k]) for k in ("mae", "rmse", "r2", "nrmse")},
            **{f"val_{k}": float(val_all[k]) for k in ("mae", "rmse", "r2", "nrmse")},
            "epochs_ran": info["epochs_ran"],
            "training_seconds": fit_s,
        })
        # per-epoch training history (PRD §49 logging requirement)
        for h in info["history"]:
            mlflow.log_metrics({"val_rmse_normalized": h["val_rmse"], "lr": h["lr"]},
                               step=h["epoch"])
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")
        mlflow.log_artifact(str(meta_path), artifact_path="")

    # ---- console summary ----------------------------------------------------------------
    row = metrics[(metrics.scope == "ALL") & (metrics.split == "test")].iloc[0]
    print(f"\ntest ALL: MAE={fmt(row.mae)} RMSE={fmt(row.rmse)} R²={fmt(row.r2)} "
          f"nRMSE={fmt(row.nrmse)} DayMAE={fmt(row.daylight_mae)}")
    for name, path in (("xgboost", "artifacts/xgboost/metrics.csv"),):
        p = REPO_ROOT / path
        if p.exists():
            ref = pd.read_csv(p)
            r = ref[(ref.split == "test") & (ref.scope == "ALL")].iloc[0]
            print(f"{name} baseline test: MAE={r.mae:.3f} RMSE={r.rmse:.3f} R²={r.r2:.3f}")
    return 0


def _pkg_versions(*names: str) -> dict:
    import importlib.metadata as im

    return {p: im.version(p) for p in names}


if __name__ == "__main__":
    raise SystemExit(main())
