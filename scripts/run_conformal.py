"""Phase 11 orchestrator: split-conformal prediction intervals (PRD §28).

Calibrates absolute-residual conformal radii on the VAL split and evaluates
coverage/width on TEST (both canonical D-011), for two forecasters:

* ``xgboost``     — the frozen Phase 6 run (best overall model)
* ``persistence`` — the primary baseline (intervals as reference floor)

Two calibrations per model × level:

* ``global``   — one radius for every row (marginal guarantee only)
* ``mondrian`` — one radius per regime label
  ``{day,night} × {lag_present, lag_missing}``, the heteroscedasticity the
  data actually has (Phase 3: night ≈ zero output; Phase 10: missing-lag
  default branches inflate night errors ~35×)

Protocol notes (D-018):

* Calibration uses VAL observed-target rows only. VAL also fed XGBoost
  early stopping (one scalar) — mildly optimistic residuals; empirical TEST
  coverage is the honest check and is reported per scope.
* Persistence is fit on the FULL table's history (D-011 #3 precedent: its
  t−24h lookups are strictly causal, so val/test targets are never read),
  otherwise eval rows would be NaN-starved.
* Intervals are ``ŷ ± q`` (absolute-residual split conformal); the
  guarantee is marginal under exchangeability, per-regime for Mondrian.

Artifacts → ``artifacts/uncertainty/``:

* ``conformal_metrics.csv``            — model × method × level × scope
* ``predictions_with_intervals.parquet`` — TEST rows + bounds (long format)
* ``sample_forecast.json``             — PRD §28 output shape, 6 sample rows
* ``conformal_report.md``              — coverage/width tables
* ``run_metadata.json``

Usage: ``conda run -n solar python scripts/run_conformal.py``
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow

from src.config import load_config
from src.data.splits import chronological_split
from src.models.baseline import PersistenceBaseline
from src.models.conformal import (
    coverage_metrics,
    fit_conformal,
    interval_widths,
)
from src.models.xgboost_model import load_xgboost_model, predict_frame

RUN_NAME = "conformal-h1-test-v1"
MODEL_PATH = REPO_ROOT / "models" / "xgboost_site_all_h1_v1.json"
META_PATH = REPO_ROOT / "artifacts" / "xgboost" / "run_metadata.json"
OUT_DIR = REPO_ROOT / "artifacts" / "uncertainty"
LEVELS = {"0.9": 0.10, "0.8": 0.20}  # confidence_level → alpha


def regime_labels(frame: pd.DataFrame) -> np.ndarray:
    """{day,night} × {lag_present,lag_missing} — known at inference time."""
    lag = frame["power_lag_1"].to_numpy(dtype=float)
    day = frame["is_daylight"].to_numpy(dtype=bool)
    nolag = np.isnan(lag)
    return np.where(day,
                    np.where(nolag, "day_nolag", "day_lag"),
                    np.where(nolag, "night_nolag", "night_lag"))


def main() -> int:
    cfg = load_config()
    tcfg = cfg["training"]
    seed = int(tcfg["seed"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(REPO_ROOT / cfg["paths"]["features_dir"])
    train, val, test = chronological_split(
        df, ratios=(tcfg["train_ratio"], tcfg["val_ratio"], tcfg["test_ratio"]))
    del df
    print(f"split: train={len(train):,} val={len(val):,} test={len(test):,}")

    # ---- model predictions -------------------------------------------------
    reg, xgb_meta = load_xgboost_model(MODEL_PATH, META_PATH)
    preds = {
        "xgboost": {"val": predict_frame(reg, val),
                    "test": predict_frame(reg, test)},
    }
    base = PersistenceBaseline().fit(
        pd.concat([train, val, test], ignore_index=True)[
            ["site_id", "timestamp", "power"]])
    preds["persistence"] = {"val": base.predict(val).to_numpy(dtype=float),
                            "test": base.predict(test).to_numpy(dtype=float)}

    groups = {"val": regime_labels(val), "test": regime_labels(test)}
    y = {"val": val["power"].to_numpy(dtype=float),
         "test": test["power"].to_numpy(dtype=float)}

    # ---- calibrate on VAL, evaluate on TEST ---------------------------------
    calibrations, metric_rows = {}, []
    for model in ("xgboost", "persistence"):
        for method in ("global", "mondrian"):
            for lvl_name, alpha in LEVELS.items():
                g = groups["val"] if method == "mondrian" else None
                cal = fit_conformal(y["val"], preds[model]["val"],
                                    alpha=alpha, groups=g)
                calibrations[(model, method, lvl_name)] = cal
                gt = groups["test"] if method == "mondrian" else None
                lo, hi = interval_widths(preds[model]["test"], cal, groups=gt)
                m = coverage_metrics(y["test"], preds[model]["test"], lo, hi)
                metric_rows.append({
                    "model": model, "method": method, "level": lvl_name,
                    "scope": "ALL", "label": "", **m})

    cal_df = None  # radii live in meta['calibration']; no separate frame

    # per-regime coverage (mondrian) + per-site spread (headline 0.9 mondrian)
    for model in ("xgboost", "persistence"):
        cal = calibrations[(model, "mondrian", "0.9")]
        lo, hi = interval_widths(preds[model]["test"], cal,
                                 groups=groups["test"])
        for label in ("day_lag", "day_nolag", "night_lag", "night_nolag"):
            m = groups["test"] == label
            mm = coverage_metrics(y["test"][m], preds[model]["test"][m],
                                  lo[m], hi[m])
            metric_rows.append({"model": model, "method": "mondrian",
                                "level": "0.9", "scope": "REGIME",
                                "label": label, **mm})
        site_ids = test["site_id"].to_numpy()
        for sid in pd.unique(site_ids):
            m = site_ids == sid
            mm = coverage_metrics(y["test"][m], preds[model]["test"][m],
                                  lo[m], hi[m])
            metric_rows.append({"model": model, "method": "mondrian",
                                "level": "0.9", "scope": "SITE",
                                "label": f"site_{sid}", **mm})

    metrics = pd.DataFrame(metric_rows)
    cols = ["model", "method", "level", "scope", "label", "n", "n_missing",
            "coverage", "mae", "mean_width", "median_width", "p90_width"]
    metrics_path = OUT_DIR / "conformal_metrics.csv"
    metrics[cols].to_csv(metrics_path, index=False)

    # ---- interval predictions (long format, TEST) ----------------------------
    out_parts = []
    keep = test[["site_id", "timestamp", "power", "is_daylight"]].copy()
    keep["regime"] = groups["test"]
    for model in ("xgboost", "persistence"):
        for method in ("global", "mondrian"):
            for lvl_name in LEVELS:
                cal = calibrations[(model, method, lvl_name)]
                lo, hi = interval_widths(
                    preds[model]["test"], cal,
                    groups=groups["test"] if method == "mondrian" else None)
                out_parts.append(pd.DataFrame({
                    "site_id": keep["site_id"].to_numpy(),
                    "timestamp": keep["timestamp"].to_numpy(),
                    "power": keep["power"].to_numpy(),
                    "is_daylight": keep["is_daylight"].to_numpy(),
                    "regime": keep["regime"].to_numpy(),
                    "model": model, "method": method,
                    "confidence_level": float(lvl_name),
                    "prediction": preds[model]["test"],
                    "lower_bound": lo, "upper_bound": hi,
                }))
    intervals = pd.concat(out_parts, ignore_index=True)
    intervals_path = OUT_DIR / "predictions_with_intervals.parquet"
    intervals.to_parquet(intervals_path, engine="pyarrow", index=False)

    # ---- PRD §28 sample output ----------------------------------------------
    xg_mond = intervals[(intervals.model == "xgboost")
                        & (intervals.method == "mondrian")
                        & (intervals.confidence_level == 0.9)]
    sample = xg_mond.dropna(subset=["prediction"]).groupby(
        "regime", observed=True).head(2).sort_values("timestamp")
    sample_json = [{
        "site_id": int(r.site_id),
        "timestamp": str(r.timestamp),
        "prediction": round(float(r.prediction), 3),
        "lower_bound": round(float(r.lower_bound), 3),
        "upper_bound": round(float(r.upper_bound), 3),
        "confidence_level": 0.9,
    } for r in sample.itertuples()]
    (OUT_DIR / "sample_forecast.json").write_text(
        json.dumps(sample_json, indent=2), encoding="utf-8")

    # ---- metadata ------------------------------------------------------------
    meta = {
        "run_name": RUN_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": "split conformal, absolute residuals, finite-sample quantile",
        "calibration_split": "val (observed targets; also fed xgboost early "
                             "stopping — see D-018 caveat)",
        "evaluation_split": "test",
        "explained_model_run": xgb_meta.get("run_name"),
        "levels": LEVELS,
        "regimes": ["day_lag", "day_nolag", "night_lag", "night_nolag"],
        "calibration": {f"{m}|{me}|{lv}": {
            "global_radius": round(c["global"], 4),
            "regime_radii": {k: round(v, 4)
                             for k, v in c["groups"].items()},
            "n": c["n"],
        } for (m, me, lv), c in calibrations.items()},
        "seed": seed,
        "n_rows": {"train": len(train), "val": len(val), "test": len(test)},
    }
    meta_path = OUT_DIR / "run_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    write_report(OUT_DIR, metrics, meta)

    # ---- MLflow ---------------------------------------------------------------
    mlflow.set_tracking_uri((REPO_ROOT / "mlruns").as_uri())
    mlflow.set_experiment("unisolar")
    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.log_params({
            "method": "split_conformal", "models": "xgboost+persistence",
            "calibration_split": "val", "evaluation_split": "test",
            "levels": ",".join(LEVELS), "mondrian_regimes": 4, "seed": seed,
        })
        allrows = metrics[(metrics.scope == "ALL")]
        mlflow.log_metrics({
            **{f"cov_{r.model}_{r.method}_{r.level}": float(r.coverage)
               for r in allrows.itertuples()},
            **{f"width_{r.model}_{r.method}_{r.level}": float(r.mean_width)
               for r in allrows.itertuples()},
        })
        for p in OUT_DIR.iterdir():
            if p.suffix in (".csv", ".json", ".md", ".parquet"):
                mlflow.log_artifact(str(p), artifact_path="uncertainty")

    print(f"wrote artifacts under artifacts/uncertainty/ "
          f"({len(metrics)} metric rows, {len(intervals):,} interval rows)")
    return 0


def write_report(out_dir: Path, metrics: pd.DataFrame, meta: dict) -> None:
    def cell(v):
        return "n/a" if pd.isna(v) else f"{v:.3f}"

    lines = ["# Conformal prediction intervals (PRD §28, Phase 11)", "",
             f"_Generated {meta['generated_at']}; split conformal, absolute "
             "residuals, calibrated on VAL observed rows, evaluated on TEST "
             "(D-011). Mondrian regimes = {day,night} × {lag_present,"
             "lag_missing} (Phase 10 night failure mode)._", "",
             "## Calibration radii (VAL, kWh)", "",
             "| model | method | level | global radius | regime radii |",
             "|---|---|---|---:|---|"]
    for k, c in meta["calibration"].items():
        model, method, lvl = k.split("|")
        regs = ", ".join(f"{k2} {v2:.2f}"
                         for k2, v2 in c["regime_radii"].items()) or "-"
        lines.append(f"| {model} | {method} | {lvl} | "
                     f"{c['global_radius']:.3f} | {regs} |")
    lines += ["", "## TEST coverage — ALL rows", "",
              "Nominal: level 0.9 → ≥0.90, level 0.8 → ≥0.80 "
              "(marginal, exchangeability assumed).", "",
              "| model | method | level | coverage | MAE | mean width | "
              "median width | p90 width | n |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    sub = metrics[(metrics.scope == "ALL")]
    for r in sub.itertuples():
        lines.append(f"| {r.model} | {r.method} | {r.level} | "
                     f"{cell(r.coverage)} | {cell(r.mae)} | "
                     f"{cell(r.mean_width)} | {cell(r.median_width)} | "
                     f"{cell(r.p90_width)} | {int(r.n):,} |")
    lines += ["", "## TEST coverage — regimes (mondrian, level 0.9)", "",
              "| model | regime | n | coverage | mean width |",
              "|---|---|---:|---:|---:|"]
    sub = metrics[(metrics.scope == "REGIME") & (metrics.level == "0.9")]
    for r in sub.itertuples():
        lines.append(f"| {r.model} | {r.label} | {int(r.n):,} | "
                     f"{cell(r.coverage)} | {cell(r.mean_width)} |")
    lines += ["", "## Per-site coverage spread (mondrian, level 0.9)", "",
              "| model | sites | coverage min | coverage max | "
              "width min | width max | below-nominal sites |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for model in ("xgboost", "persistence"):
        s = metrics[(metrics.scope == "SITE") & (metrics.model == model)
                    & (metrics.level == "0.9")]
        lines.append(f"| {model} | {len(s)} | {s.coverage.min():.3f} | "
                     f"{s.coverage.max():.3f} | {s.mean_width.min():.3f} | "
                     f"{s.mean_width.max():.3f} | "
                     f"{int((s.coverage < 0.90).sum())} |")
    lines.append("")
    (out_dir / "conformal_report.md").write_text("\n".join(lines),
                                                 encoding="utf-8")
    print("wrote conformal_report.md")


if __name__ == "__main__":
    raise SystemExit(main())
