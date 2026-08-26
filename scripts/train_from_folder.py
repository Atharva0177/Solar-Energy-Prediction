"""Folder-to-model trainer backing POST /api/v1/train/jobs.

Composes the SAME library entry points the phase scripts use — schema_map,
validate/clean/night (Phase 2 flow), the Phase 5 feature families,
chronological_split (D-011), PersistenceBaseline, train_xgboost /
train_sequence — but everything lands in a job-scoped directory:

    <dataset-dir>/
      raw/                      # uploaded CSVs (path mode reads in place)
      processed/solar/          # staged cleaned parquet (+ site_details)
      features/                 # staged Phase 5 feature table
      artifacts/<model>/        # model file, metrics.csv, predictions
      result.json               # everything the Train page displays

The served v1 models and phase artifacts are never touched (D-024).

Stage progress goes to stdout as ``== STAGE <name> start|done`` markers so
the API can parse a live stage list from the log tail without shared state.

``--fast-test`` shrinks hyperparameters (tiny models, 1 epoch) so the test
suite can exercise the whole chain in seconds on synthetic data.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

os.environ.pop("MLFLOW_TRACKING_URI", None)  # job runs never touch mlruns/

CADENCE_S = 900
LAG_STEPS = [1, 2, 4, 8, 24, 48, 96]
ROLLING_WINDOWS_S = [3600, 21600, 86400]
ROLLING_STATS = ["mean", "std", "min", "max"]
TIMEZONE = "Australia/Melbourne"  # D-007 default; verified empirically below
METRIC_COLS = ["split", "scope", "site_id", "n_eval", "n_missing", "mae",
               "rmse", "r2", "nrmse", "daylight_n", "daylight_mae",
               "daylight_nrmse"]

REQUIRED_FILES = {
    "Solar_Energy_Generation.csv": ["SiteKey", "CampusKey", "Timestamp",
                                    "SolarGeneration"],
    "Weather_Data_reordered_all.csv": ["CampusKey", "Timestamp",
                                       "AirTemperature", "RelativeHumidity"],
    "Solar_Site_Details.csv": ["SiteKey", "CampusKey", "kWp", "lat", "Lon"],
}


def stage(name: str) -> None:
    print(f"== STAGE {name} start", flush=True)


def stage_done(name: str, payload: dict | None = None) -> None:
    print(f"== STAGE {name} done {json.dumps(payload or {}, default=str)}",
          flush=True)


def fail(reason: str) -> int:
    print(f"== FAILED {json.dumps({'reason': reason})}", flush=True)
    return 1


def score_partitions(pred_by_split: dict[str, pd.Series],
                     parts: dict[str, pd.DataFrame],
                     denom_all: float, site_ranges: dict) -> pd.DataFrame:
    """ALL + per-site PRD §25 metric rows, D-011 denominators."""
    rows = []
    for split_name, part in parts.items():
        pred = pred_by_split[split_name]
        m = _metrics(part["power"].to_numpy(), np.asarray(pred, dtype=float),
                     part["is_daylight"].to_numpy(), denom_all)
        rows.append({"split": split_name, "scope": "ALL", "site_id": "", **m})
        for sid, sub in part.groupby("site_id", observed=True):
            ms = _metrics(sub["power"].to_numpy(),
                          np.asarray(pred.loc[sub.index], dtype=float),
                          sub["is_daylight"].to_numpy(),
                          site_ranges.get(sid))
            rows.append({"split": split_name, "scope": "SITE",
                         "site_id": int(sid), **ms})
    return pd.DataFrame(rows)


def _metrics(y, p, daylight, denom) -> dict:
    from src.training.evaluate import regression_metrics

    return regression_metrics(y, p, daylight=daylight, denom=denom)


def _jsonable(obj):
    """numpy scalars → Python; NaN floats → None. pandas .to_dict() output is
    numpy-typed and json's default=str would stringify int64s."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        return None if np.isnan(f) else f
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


# ---------------------------------------------------------------- stages ----

def do_verify(raw: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three CSVs, check required columns, build the profile block."""
    from src.data import schema_map

    files = []
    for fname, req_cols in REQUIRED_FILES.items():
        p = raw / fname
        if not p.exists():
            return_fail = f"missing file: {fname}"
            raise SystemExit(fail(f"{raw} -> {return_fail}"))
        head = pd.read_csv(p, nrows=5)
        missing = [c for c in req_cols if c not in head.columns]
        files.append({"name": fname, "ok": not missing,
                      "detail": ("columns ok" if not missing
                                 else f"missing columns: {missing}")})
        if missing:
            raise SystemExit(fail(f"{fname}: missing columns {missing}"))

    gen = schema_map.load_generation(raw)
    wx = schema_map.load_weather(raw)
    sites = schema_map.load_site_details(raw)

    # loaders return CANONICAL columns (schema_map renames) — profile reads them
    ts = pd.to_datetime(gen["timestamp"], errors="coerce")
    cadence_min = None
    if ts.notna().sum() > 10:
        diffs = ts.dropna().sort_values().diff().dropna()
        mode_delta = diffs.mode().iloc[0] if len(diffs) else None
        if mode_delta is not None and mode_delta.total_seconds() > 0:
            cadence_min = int(mode_delta.total_seconds() // 60)

    profile = {
        "generation_rows": int(len(gen)),
        "weather_rows": int(len(wx)),
        "sites": int(sites["site_id"].nunique()),
        "campuses": int(gen["campus_id"].nunique()),
        "start": str(ts.min()), "end": str(ts.max()),
        "cadence_minutes": cadence_min,
        "target_missing_pct": round(float(pd.to_numeric(
            gen["power"], errors="coerce").isna().mean() * 100), 2),
    }
    # Downstream night-tagging merges solar-position timestamps (datetime64)
    # against these frames — CSV strings here would fail the merge
    # (Phase 2 flow parses before tagging; this orchestrator must too).
    for frame in (gen, wx):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    return {"files": files, "profile": profile}, gen, wx, sites


def do_prepare(job: Path, gen, wx, sites, t_choice: dict) -> dict:
    """Phase 2 flow into <job>/processed + Phase 5 features into <job>/features."""
    from src.data import night, validate
    from src.data.clean import CleanLog
    from src.data import clean as cln
    from build_processed import handle_missing, outlier_analysis
    from src.features import lag as lag_mod
    from src.features import rolling as roll_mod
    from src.features import solar as solar_mod
    from src.features import temporal as temp_mod
    from src.features import weather as wx_mod

    log = CleanLog()
    numeric_expected = {
        "generation": ["power"],
        "weather": ["temperature", "apparent_temperature",
                    "dew_point_temperature", "humidity", "wind_speed",
                    "wind_direction"],
        "site_details": ["capacity_kwp", "n_panels", "latitude", "longitude"],
    }
    for name, df in (("generation", gen), ("weather", wx),
                     ("site_details", sites)):
        df = cln.parse_timestamps(df, log, name)
        df = cln.coerce_numeric(df, numeric_expected[name], log, name)
        df = cln.drop_exact_duplicates(df, log, name)
        if name == "generation":
            gen = df
        elif name == "weather":
            wx = df
        else:
            sites = df

    gen = cln.null_impossible_values(
        gen, {"power": validate.IMPOSSIBLE_RULES["power"]}, log, "generation")
    wx_rules = {c: validate.IMPOSSIBLE_RULES[c] for c in wx.columns
                if c in validate.IMPOSSIBLE_RULES}
    wx = cln.null_impossible_values(wx, wx_rules, log, "weather")

    validation = {
        "generation": {
            "duplicate_keys": validate.check_duplicate_keys(gen, "site_id"),
            "gaps": validate.check_gaps(gen, "site_id"),
            "impossible": validate.check_impossible_values(gen),
        },
        "weather": {
            "duplicate_keys": validate.check_duplicate_keys(wx, "campus_id"),
            "gaps": validate.check_gaps(wx, "campus_id"),
            "impossible": validate.check_impossible_values(wx),
        },
    }

    chosen_tz = t_choice["chosen_timezone"]
    coords = night.campus_coordinates(sites)
    gen = night.add_solar_position(gen, coords, chosen_tz)

    merged = gen.merge(wx, on=["campus_id", "timestamp"], how="left",
                       suffixes=("", "_wx"))
    merged = handle_missing(merged, log)

    processed_dir = job / "processed"
    parquet_dir = processed_dir / "solar"
    if parquet_dir.exists():
        shutil.rmtree(parquet_dir)
    out = merged.copy()
    out["year"] = out["timestamp"].dt.year
    out["month"] = out["timestamp"].dt.month
    out.to_parquet(parquet_dir, engine="pyarrow",
                   partition_cols=["site_id", "year", "month"], index=False)
    sites_out = sites.copy()
    sites_out.to_parquet(processed_dir / "site_details.parquet",
                         engine="pyarrow", index=False)

    # ---- Phase 5 features -------------------------------------------------
    feats_dir = job / "features"
    if feats_dir.exists():
        shutil.rmtree(feats_dir)
    df = pd.read_parquet(parquet_dir)
    base_cols = list(df.columns)
    df = temp_mod.add_temporal_features(df)
    lag_specs = {f"power_lag_{s}": pd.Timedelta(s * CADENCE_S, unit="s")
                 for s in LAG_STEPS}
    df = lag_mod.add_lags(df, lag_specs)
    df = roll_mod.add_rolling_features(
        df, windows=[pd.Timedelta(w, unit="s") for w in ROLLING_WINDOWS_S],
        stats=ROLLING_STATS, min_periods=1)
    df = wx_mod.add_weather_features(df)
    coords_f = sites.groupby("campus_id", observed=True).agg(
        latitude=("latitude", "median"), longitude=("longitude", "median")
    ).reset_index()
    df = df.drop(columns=[c for c in ("solar_elevation_deg", "is_daylight")
                          if c in df.columns])
    df = solar_mod.add_solar_position_features(df, coords_f, TIMEZONE)
    fout = df.copy()
    fout["year"] = fout["timestamp"].dt.year
    fout["month"] = fout["timestamp"].dt.month
    fout.to_parquet(feats_dir, engine="pyarrow",
                    partition_cols=["site_id", "year", "month"], index=False)

    return {
        "merged_rows": int(len(out)),
        "features_rows": int(len(fout)),
        "features_cols": int(fout.shape[1]),
        "engineered_columns": int(len([c for c in fout.columns
                                       if c not in base_cols]) - 2),
        "timezone_chosen": chosen_tz,
        "timezone_candidates": t_choice.get("candidates"),
        "cleaning_ops": log.to_dict()["operations"],
        "validation": json.loads(json.dumps(validation, default=str)),
    }


def do_baseline(feats_dir: Path, ratios, seed: int) -> tuple[dict, dict, float]:
    """Persistence baseline on the canonical split — comparison anchor."""
    from src.models.baseline import PersistenceBaseline

    df = pd.read_parquet(feats_dir)
    train, val, test = chronological_split(df, ratios)
    tr_obs = train.loc[train["power"].notna()]
    denom_all = float(tr_obs["power"].max() - tr_obs["power"].min())
    per_range = tr_obs.groupby("site_id", observed=True)["power"].agg(
        lambda s: float(s.max() - s.min()))
    site_ranges = {sid: (v if v > 0 else None) for sid, v in per_range.items()}
    del tr_obs

    # Fit on the FULL table (D-011 #3): the causal t−24h lookups reach back
    # into val/test history, so fitting train-only NaN-starves val/test preds.
    base = PersistenceBaseline().fit(df)
    preds = {name: base.predict(part) for name, part in
             (("val", val), ("test", test))}
    metrics = score_partitions(preds, {"val": val, "test": test},
                               denom_all, site_ranges)
    split_info = _split_info(train, val, test)
    return metrics, split_info, denom_all, site_ranges, train, val, test


def _split_info(*parts) -> dict:
    out = {}
    for name, p in zip(("train", "val", "test"), parts):
        obs = p.loc[p["power"].notna(), "timestamp"]
        out[name] = {"rows": int(len(p)),
                     "observed_rows": int(len(obs)),
                     "start": str(p["timestamp"].min()),
                     "end": str(p["timestamp"].max())}
    return out


def chronological_split(df, ratios):
    from src.data.splits import chronological_split as cs

    return cs(df, ratios=ratios)


def do_train_xgboost(job: Path, train, val, test, cfg, denom_all,
                     site_ranges, fast: bool) -> dict:
    from src.models.xgboost_model import (
        dataset_fingerprint,
        extract_importance,
        predict_frame,
        train_xgboost,
    )

    params = dict(cfg["models"]["xgboost"]["params"])
    es_rounds = int(cfg["training"]["early_stopping_rounds"])
    if fast:
        params.update(n_estimators=15, max_depth=3)
        es_rounds = 10

    t0 = time.perf_counter()
    model, info = train_xgboost(train, val, params=params, seed=int(cfg["training"]["seed"]),
                                early_stopping_rounds=es_rounds)
    fit_s = time.perf_counter() - t0

    preds = {}
    for name, part in (("val", val), ("test", test)):
        preds[name] = pd.Series(predict_frame(model, part), index=part.index)
    metrics = score_partitions(preds, {"val": val, "test": test},
                               denom_all, site_ranges)

    art = job / "artifacts" / "xgboost"
    art.mkdir(parents=True, exist_ok=True)
    imp = extract_importance(model)
    imp.head(20).to_csv(art / "feature_importance_top20.csv", index=False)
    test_pred = test[["site_id", "timestamp", "power", "is_daylight"]].copy()
    test_pred["prediction"] = preds["test"]
    test_pred.to_parquet(art / "predictions_test.parquet", engine="pyarrow",
                         index=False)
    model.get_booster().save_model(str(art / "model.json"))
    metrics[METRIC_COLS].to_csv(art / "metrics.csv", index=False)

    imp20 = [{"feature": r.feature, "gain": round(float(r.gain), 6)}
             for r in imp.head(20).itertuples()]
    return {
        "model_name": "xgboost",
        "fit_seconds": round(fit_s, 2),
        "best_iteration": info["best_iteration"],
        "params_used": params,
        "feature_importance_top20": imp20,
        "dataset_fingerprint_features": dataset_fingerprint(job / "features"),
        "artifacts": {"model": str((art / "model.json").relative_to(job)),
                      "metrics": "artifacts/xgboost/metrics.csv",
                      "predictions": "artifacts/xgboost/predictions_test.parquet"},
        "_metrics_df": metrics[METRIC_COLS],
    }


def do_train_sequence(job: Path, arch: str, train, val, test, cfg,
                      denom_all, site_ranges, fast: bool) -> dict:
    import torch

    from src.models.sequence_model import (
        CHANNELS,
        RecurrentForecaster,
        TransformerForecaster,
        WindowDataset,
        build_channel_matrix,
        fit_channel_scaler,
        predict_windows,
        train_sequence,
    )

    params = dict(cfg["models"][arch]["params"])
    if fast:
        params.update(lookback_steps=8, hidden_size=8, num_layers=1,
                      batch_size=512, max_epochs=1, patience=1)
        if arch == "transformer":
            params.update(d_model=16, nhead=2, dim_feedforward=32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = int(cfg["training"]["seed"])

    parts = {"val": val, "test": test}
    tr_obs = train.loc[train["power"].notna()]
    y_mean = float(tr_obs["power"].mean())
    y_std = float(tr_obs["power"].std())
    del tr_obs

    M_tr = build_channel_matrix(train)
    mean, std = fit_channel_scaler(M_tr)
    yn_tr = ((train["power"].to_numpy(dtype=np.float64)) - y_mean) / y_std
    lookback = int(params["lookback_steps"])

    def bounds_of(part):
        sid = part["site_id"].to_numpy()
        b, start = [], 0
        for i in range(1, len(sid) + 1):
            if i == len(sid) or sid[i] != sid[start]:
                b.append((start, i))
                start = i
        return b

    ds_train = WindowDataset(((M_tr - mean) / std).astype(np.float32),
                             yn_tr.astype(np.float32),
                             bounds_of(train), lookback)
    datasets = {}
    for split_name, part in parts.items():
        M = build_channel_matrix(part)
        yn = ((part["power"].to_numpy(dtype=np.float64)) - y_mean) / y_std
        datasets[split_name] = WindowDataset(
            ((M - mean) / std).astype(np.float32), yn.astype(np.float32),
            bounds_of(part), lookback)

    torch.manual_seed(seed)
    if arch == "transformer":
        model = TransformerForecaster(
            input_size=len(CHANNELS), d_model=int(params["d_model"]),
            nhead=int(params["nhead"]), num_layers=int(params["num_layers"]),
            dim_feedforward=int(params["dim_feedforward"]),
            dropout=float(params["dropout"]), max_len=lookback + 1)
    else:
        model = RecurrentForecaster(arch, input_size=len(CHANNELS),
                                    hidden_size=int(params["hidden_size"]),
                                    num_layers=int(params["num_layers"]),
                                    dropout=float(params["dropout"]))

    art = job / "artifacts" / arch
    art.mkdir(parents=True, exist_ok=True)
    ckpt = art / "model.pt"
    t0 = time.perf_counter()
    model, info = train_sequence(model, ds_train, datasets["val"],
                                 params=params, seed=seed, device=device,
                                 verbose=True, checkpoint_path=ckpt)
    fit_s = time.perf_counter() - t0

    pred_by_split = {}
    for split_name, part in parts.items():
        ds = datasets[split_name]
        pv = predict_windows(model, ds, batch_size=2048, device=device,
                             y_mean=y_mean, y_std=y_std)
        pred = np.full(len(part), np.nan)
        pred[np.asarray(ds.global_pos)] = pv
        pred_by_split[split_name] = pd.Series(pred, index=part.index)
    metrics = score_partitions(pred_by_split, parts, denom_all, site_ranges)

    test_pred = test[["site_id", "timestamp", "power", "is_daylight"]].copy()
    test_pred["prediction"] = pred_by_split["test"]
    test_pred.to_parquet(art / "predictions_test.parquet", engine="pyarrow",
                         index=False)
    metrics[METRIC_COLS].to_csv(art / "metrics.csv", index=False)

    return {
        "model_name": arch,
        "device": device,
        "fit_seconds": round(fit_s, 2),
        "epochs_ran": info["epochs_ran"],
        "best_val_rmse_normalized": round(info["best_val_rmse"], 5),
        "training_history": info["history"],
        "lookback_steps": lookback,
        "channels": list(CHANNELS),
        "params_used": params,
        "artifacts": {"model": str(ckpt.relative_to(job)),
                      "metrics": f"artifacts/{arch}/metrics.csv",
                      "predictions": f"artifacts/{arch}/predictions_test.parquet"},
        "_metrics_df": metrics[METRIC_COLS],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--model", required=True,
                    choices=("xgboost", "lstm", "gru", "transformer"))
    ap.add_argument("--fast-test", action="store_true")
    args = ap.parse_args()
    job, raw, arch = args.dataset_dir.resolve(), args.raw.resolve(), args.model

    timings = {}
    t_all = time.perf_counter()

    stage("verify")
    t0 = time.perf_counter()
    verify, gen, wx, sites = do_verify(raw)
    timings["verify_s"] = round(time.perf_counter() - t0, 2)
    stage_done("verify", verify)

    from src.config import load_config
    from src.data.night import choose_timezone, campus_coordinates

    cfg = load_config()
    if args.fast_test:
        cfg = json.loads(json.dumps(cfg))  # deep copy before shrinking
    tcfg = cfg["training"]
    ratios = (float(tcfg["train_ratio"]), float(tcfg["val_ratio"]),
              float(tcfg["test_ratio"]))
    coords = campus_coordinates(sites)  # schema_map loaders already canonicalized

    stage("prepare")
    t0 = time.perf_counter()
    t_choice = choose_timezone(gen.sample(frac=min(1.0, 400_000 / max(len(gen), 1)),
                                          random_state=42), coords)
    prep = do_prepare(job, gen, wx, sites, t_choice)
    timings["prepare_s"] = round(time.perf_counter() - t0, 2)
    stage_done("prepare", {"rows": prep["features_rows"],
                           "cols": prep["features_cols"],
                           "tz": prep["timezone_chosen"]})

    stage("baseline")
    t0 = time.perf_counter()
    base_metrics, split_info, denom_all, site_ranges, train, val, test = \
        do_baseline(job / "features", ratios, int(tcfg["seed"]))
    timings["baseline_s"] = round(time.perf_counter() - t0, 2)
    stage_done("baseline", {"test_mae": _safe(base_metrics, "test", "mae")})

    stage("train")
    t0 = time.perf_counter()
    if arch == "xgboost":
        trained = do_train_xgboost(job, train, val, test, cfg, denom_all,
                                   site_ranges, args.fast_test)
    else:
        trained = do_train_sequence(job, arch, train, val, test, cfg,
                                    denom_all, site_ranges, args.fast_test)
    timings["train_s"] = round(time.perf_counter() - t0, 2)
    stage_done("train", {"model": trained["model_name"],
                         "fit_seconds": trained["fit_seconds"]})

    stage("evaluate")
    metrics_df = trained.pop("_metrics_df")
    test_all = metrics_df[(metrics_df.split == "test") & (metrics_df.scope == "ALL")].iloc[0].to_dict()
    val_all = metrics_df[(metrics_df.split == "val") & (metrics_df.scope == "ALL")].iloc[0].to_dict()
    timings["evaluate_s"] = round(time.perf_counter() - t0, 2)
    timings["total_s"] = round(time.perf_counter() - t_all, 2)
    stage_done("evaluate", {"test_all": test_all})

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": verify,
        "cleaning_and_prepare": prep,
        "split": split_info,
        "config_used": {"ratios": ratios,
                        "seed": int(tcfg["seed"]),
                        "fast_test": args.fast_test},
        "timing": timings,
        "persistence": {
            "val_all": base_metrics[(base_metrics.split == "val") & (base_metrics.scope == "ALL")].iloc[0].to_dict(),
            "test_all": base_metrics[(base_metrics.split == "test") & (base_metrics.scope == "ALL")].iloc[0].to_dict(),
        },
        "model": trained,
        "metrics_per_site": metrics_df.to_dict("records"),
        "test_all": test_all,
        "val_all": val_all,
    }
    result = _jsonable(result)
    (job / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print("== DONE", flush=True)
    return 0


def _safe(df, split, col) -> float | None:
    row = df[(df.split == split) & (df.scope == "ALL")]
    v = float(row.iloc[0][col]) if len(row) else np.nan
    return None if pd.isna(v) else round(v, 4)


if __name__ == "__main__":
    raise SystemExit(main())
