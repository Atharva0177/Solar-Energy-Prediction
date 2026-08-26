"""Phase 6 orchestrator: train + evaluate XGBoost on the feature table.

Protocol:
* canonical per-site chronological split from ``configs/training.yaml``
  (same ``src/data/splits`` code as baselines — D-011);
* single-step horizon (one 15-min slot), early stopping on validation;
* metrics scored with the SAME nRMSE denominators as Phase 4 so numbers are
  directly comparable to `artifacts/baselines/baseline_metrics.csv`;
* MLflow tracking to local ``mlruns/`` (PRD §30), run name follows the
  ``xgboost-site-all-h<N>-v<N>`` convention.

Artifacts:

* ``models/xgboost_site_all_h1_v1.json``   — booster (native JSON)
* ``artifacts/xgboost/metrics.csv``        — ALL + per-site, val/test, PRD §25
* ``artifacts/xgboost/feature_importance.csv``
* ``artifacts/xgboost/predictions_test.parquet``
* ``artifacts/xgboost/run_metadata.json``
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# PRD §30 mandates local mlruns/ file tracking; recent MLflow blocks that
# backend unless explicitly opted in.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow

from src.config import load_config
from src.data.splits import chronological_split
from src.models.xgboost_model import (
    dataset_fingerprint,
    extract_importance,
    predict_frame,
    select_feature_columns,
    train_xgboost,
)
from src.training.evaluate import regression_metrics

RUN_NAME = "xgboost-site-all-h1-v1"
MODEL_PATH = REPO_ROOT / "models" / "xgboost_site_all_h1_v1.json"
OUT_DIR = REPO_ROOT / "artifacts" / "xgboost"
KEEP_COLS_IN_PREDICTIONS = ["site_id", "timestamp", "power", "is_daylight"]


def fmt(v) -> str:
    return "n/a" if pd.isna(v) else f"{v:.3f}"


def score_rows(model, part, split_name, denom_all, site_ranges):
    pred = pd.Series(predict_frame(model, part), index=part.index)
    rows = []
    m = regression_metrics(part["power"], pred,
                           daylight=part["is_daylight"].to_numpy(), denom=denom_all)
    rows.append({"split": split_name, "scope": "ALL", "site_id": "", **m})
    for sid, sub in part.groupby("site_id", observed=True):
        ms = regression_metrics(sub["power"], pred.loc[sub.index],
                                daylight=sub["is_daylight"].to_numpy(),
                                denom=site_ranges.get(sid))
        rows.append({"split": split_name, "scope": "SITE", "site_id": sid, **ms})
    out = pd.DataFrame(rows)
    return out, pred


def main() -> int:
    cfg = load_config()
    tcfg = cfg["training"]
    xgb_cfg = cfg["models"]["xgboost"]
    assert xgb_cfg["enabled"], "xgboost disabled in configs/models.yaml"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    seed = int(tcfg["seed"])
    ratios = (tcfg["train_ratio"], tcfg["val_ratio"], tcfg["test_ratio"])
    es_rounds = int(tcfg["early_stopping_rounds"])

    # ---- data -----------------------------------------------------------------
    features_dir = REPO_ROOT / cfg["paths"]["features_dir"]
    fp_features = dataset_fingerprint(features_dir)
    fp_source = dataset_fingerprint(REPO_ROOT / cfg["paths"]["processed_dir"])
    df = pd.read_parquet(features_dir)
    print(f"features: {df.shape[0]:,} x {df.shape[1]} | fingerprint {fp_features}")

    train, val, test = chronological_split(df, ratios=ratios)
    print(f"split: train={len(train):,} val={len(val):,} test={len(test):,} {ratios}")

    # nRMSE denominators — identical definition to run_baselines.py (D-011)
    tr_obs = train.loc[train["power"].notna()]
    denom_all = float(tr_obs["power"].max() - tr_obs["power"].min())
    per_range = tr_obs.groupby("site_id", observed=True)["power"].agg(
        lambda s: float(s.max() - s.min()))
    site_ranges = {sid: (v if v > 0 else None) for sid, v in per_range.items()}
    del tr_obs

    # ---- train ------------------------------------------------------------------
    t0 = time.perf_counter()
    model, info = train_xgboost(
        train, val, params=xgb_cfg["params"], seed=seed,
        early_stopping_rounds=es_rounds,
    )
    fit_s = time.perf_counter() - t0
    print(f"trained in {fit_s:.1f}s | best_iteration={info['best_iteration']} "
          f"| train rows used={info['n_train_rows']:,}")

    # ---- evaluate -----------------------------------------------------------------
    all_rows = []
    for split_name, part in (("val", val), ("test", test)):
        rows, _ = score_rows(model, part, split_name, denom_all, site_ranges)
        all_rows.append(rows)
    metrics = pd.concat(all_rows, ignore_index=True)
    cols = ["split", "scope", "site_id", "n_eval", "n_missing", "mae", "rmse",
            "r2", "nrmse", "daylight_n", "daylight_mae", "daylight_nrmse"]
    metrics_path = OUT_DIR / "metrics.csv"
    metrics[cols].to_csv(metrics_path, index=False)

    # ---- artifacts -------------------------------------------------------------------
    imp = extract_importance(model)
    imp.to_csv(OUT_DIR / "feature_importance.csv", index=False)

    _, test_pred = score_rows(model, test, "test", denom_all, site_ranges)
    preds = test[KEEP_COLS_IN_PREDICTIONS].copy()
    preds["prediction"] = test_pred
    preds.to_parquet(OUT_DIR / "predictions_test.parquet", engine="pyarrow", index=False)

    model.get_booster().save_model(MODEL_PATH)

    meta = {
        "run_name": RUN_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": "single-step forecast, horizon = 1 x 15-min",
        "config": cfg,
        "seed": seed,
        "best_iteration": info["best_iteration"],
        "early_stopping_rounds": es_rounds,
        "n_train_val_test": [len(train), len(val), len(test)],
        "rows_used_for_training": info["n_train_rows"],
        "feature_columns": info["feature_columns"],
        "nrmse_denominator_pooled_train_range_kwh": round(denom_all, 3),
        "dataset_fingerprints": {"features": fp_features, "processed_source": fp_source},
        "training_seconds": round(fit_s, 2),
        "python_versions": {"python": sys.version.split()[0]},
        "artifacts": {
            "model": str(MODEL_PATH.relative_to(REPO_ROOT)),
            "metrics": str(metrics_path.relative_to(REPO_ROOT)),
            "importance": str((OUT_DIR / "feature_importance.csv").relative_to(REPO_ROOT)),
            "predictions": str((OUT_DIR / "predictions_test.parquet").relative_to(REPO_ROOT)),
        },
    }
    import importlib.metadata as im
    meta["package_versions"] = {
        p: im.version(p) for p in ("xgboost", "pandas", "numpy", "scikit-learn", "mlflow")
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote artifacts under {OUT_DIR.name}/ + {MODEL_PATH.name}")

    # ---- MLflow (PRD §30) -------------------------------------------------------------
    mlflow.set_tracking_uri((REPO_ROOT / "mlruns").as_uri())
    mlflow.set_experiment("unisolar")
    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.log_params({
            "model": "xgboost", "site_scope": "all", "horizon_steps": 1,
            "cadence_minutes": cfg["forecast"]["cadence_minutes"],
            "seed": seed, **{f"param_{k}": v for k, v in xgb_cfg["params"].items()},
            "best_iteration": info["best_iteration"],
            "dataset_fingerprint_features": fp_features,
            "dataset_fingerprint_processed": fp_source,
        })
        test_all = metrics[(metrics.scope == "ALL") & (metrics.split == "test")].iloc[0]
        val_all = metrics[(metrics.scope == "ALL") & (metrics.split == "val")].iloc[0]
        mlflow.log_metrics({
            **{f"test_{k}": float(test_all[k]) for k in ("mae", "rmse", "r2", "nrmse", "daylight_mae")},
            **{f"val_{k}": float(val_all[k]) for k in ("mae", "rmse", "r2", "nrmse")},
            "training_seconds": fit_s,
        })
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")
        mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")
        mlflow.log_artifact(str(OUT_DIR / "feature_importance.csv"), artifact_path="evaluation")
        mlflow.log_artifact(str(OUT_DIR / "run_metadata.json"), artifact_path="")

    # ---- console summary ----------------------------------------------------------------
    row = metrics[(metrics.scope == "ALL") & (metrics.split == "test")].iloc[0]
    print(f"\ntest ALL: MAE={fmt(row.mae)} RMSE={fmt(row.rmse)} R²={fmt(row.r2)} "
          f"nRMSE={fmt(row.nrmse)} DayMAE={fmt(row.daylight_mae)}")
    base = pd.read_csv(REPO_ROOT / "artifacts" / "baselines" / "baseline_metrics.csv")
    bp = base[(base.baseline == "persistence_prev_day") & (base.split == "test") & (base.scope == "ALL")]
    if not bp.empty:
        b = bp.iloc[0]
        print(f"persistence baseline test: MAE={b['mae']:.3f} RMSE={b['rmse']:.3f} R²={b['r2']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
