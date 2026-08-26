"""Phase 9 orchestrator: seen-site vs unseen-site evaluation (PRD §12).

Trains every enabled model on the cross-site split's TRAIN frame only, then
evaluates on four frames: val/test of SEEN sites (their late-history rows,
D-011 style) and the full histories of held-out VAL/TEST sites. The headline
result is the seen→unseen degradation gap per model.

Protocol notes (D-016):

* Persistence is fit on the FULL feature table — its lookups are strictly
  causal (t−24h < t), so unseen sites are predicted from their own history.
* nRMSE uses the pooled observed range of the CROSS-SITE train slice for
  every scope, including SITE rows: held-out sites have no train slice of
  their own, so D-011's per-site denominator is undefined for them and a
  single pooled denominator keeps rows comparable.
* Sequence models early-stop on SEEN validation windows; scalers + target
  stats come from the cross-site train slice only.

Artifacts → ``artifacts/cross_site/``:

* ``cross_site_metrics.csv``  — model × protocol(seen/unseen) × split × ALL/SITE
* ``cross_site_report.md``    — ranked tables + generalization gaps
* ``run_metadata.json``       — sites chosen, seeds, versions, timing

Usage: ``conda run -n solar python scripts/run_cross_site.py [--models xgboost,lstm]``
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
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import torch

from src.config import load_config
from src.data.splits import cross_site_split
from src.models.baseline import PersistenceBaseline
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
from src.training.evaluate import regression_metrics


def fmt(v) -> str:
    return "n/a" if pd.isna(v) else f"{v:.3f}"


def site_bounds(part: pd.DataFrame) -> list[tuple[int, int]]:
    sid = part["site_id"].to_numpy()
    bounds, start = [], 0
    for i in range(1, len(sid) + 1):
        if i == len(sid) or sid[i] != sid[start]:
            bounds.append((start, i))
            start = i
    return bounds


def score_frame(frame: pd.DataFrame, pred: np.ndarray, denom: float) -> dict:
    m = regression_metrics(frame["power"].to_numpy(), pred,
                           daylight=frame["is_daylight"].to_numpy(), denom=denom)
    return m


def all_rows(model_name: str, protocol_frames: dict[str, pd.DataFrame],
             preds: dict[str, np.ndarray], denom: float) -> pd.DataFrame:
    """ALL + SITE metric rows across the four eval frames."""
    rows = []
    for fname, part in protocol_frames.items():
        protocol = "unseen" if fname.endswith("unseen") else "seen"
        split = "val" if fname.startswith("val") else "test"
        rows.append({"model": model_name, "protocol": protocol, "split": split,
                     "scope": "ALL", "site_id": "",
                     **score_frame(part, preds[fname], denom)})
        for sid, sub in part.groupby("site_id", observed=True):
            rows.append({"model": model_name, "protocol": protocol, "split": split,
                         "scope": "SITE", "site_id": sid,
                         **score_frame(sub, preds[fname][sub.index], denom)})
    return pd.DataFrame(rows)


def eval_sequence_arch(arch: str, params: dict, seed: int, device: str,
                       train_df: pd.DataFrame, eval_frames: dict,
                       lookback: int) -> tuple[pd.DataFrame, dict]:
    """Train one RNN/Transformer on the cross-site train frame; evaluate."""
    tr_obs = train_df.loc[train_df["power"].notna()]
    y_mean, y_std = float(tr_obs["power"].mean()), float(tr_obs["power"].std())
    del tr_obs

    M_tr = build_channel_matrix(train_df)
    mean, std = fit_channel_scaler(M_tr)
    yn_tr = ((train_df["power"].to_numpy(dtype=np.float64)) - y_mean) / y_std
    ds_train = WindowDataset(((M_tr - mean) / std).astype(np.float32),
                             yn_tr.astype(np.float32),
                             site_bounds(train_df), lookback)
    del M_tr

    datasets, preds = {}, {}
    for name in eval_frames:
        part = eval_frames[name]
        M = build_channel_matrix(part)
        yn = ((part["power"].to_numpy(dtype=np.float64)) - y_mean) / y_std
        datasets[name] = WindowDataset(
            ((M - mean) / std).astype(np.float32), yn.astype(np.float32),
            site_bounds(part), lookback)

    torch.manual_seed(seed)
    common = dict(input_size=len(CHANNELS), num_layers=int(params["num_layers"]),
                  dropout=float(params["dropout"]))
    if arch == "transformer":
        model = TransformerForecaster(
            d_model=int(params["d_model"]), nhead=int(params["nhead"]),
            dim_feedforward=int(params["dim_feedforward"]),
            max_len=lookback + 1, **common)
    else:
        model = RecurrentForecaster(arch, hidden_size=int(params["hidden_size"]),
                                    **common)

    t0 = time.perf_counter()
    model, info = train_sequence(model, ds_train, datasets["val_seen"],
                                 params=params, seed=seed, device=device)
    fit_s = time.perf_counter() - t0
    print(f"[{arch}] trained in {fit_s:.1f}s | epochs={info['epochs_ran']} "
          f"best_val_rmse(norm)={info['best_val_rmse']:.4f}")

    for name, ds in datasets.items():
        pv = predict_windows(model, ds, batch_size=2048, device=device,
                             y_mean=y_mean, y_std=y_std)
        p = np.full(len(eval_frames[name]), np.nan)
        p[ds.global_pos] = pv
        preds[name] = p

    meta = {"training_seconds": round(fit_s, 2), "epochs_ran": info["epochs_ran"],
            "best_val_rmse_normalized": round(info["best_val_rmse"], 5)}
    return preds, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="persistence,xgboost,lstm,gru,transformer",
                    help="comma list subset of persistence,xgboost,lstm,gru,transformer")
    args = ap.parse_args()
    wanted = [m.strip() for m in args.models.split(",") if m.strip()]

    cfg = load_config()
    tcfg = cfg["training"]
    seed = int(tcfg["seed"])
    xcfg = tcfg.get("cross_site", {})
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = REPO_ROOT / "artifacts" / "cross_site"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(REPO_ROOT / cfg["paths"]["features_dir"])
    print(f"features: {df.shape[0]:,} x {df.shape[1]} | device={device}")

    split = cross_site_split(df, val_site_frac=float(xcfg.get("val_site_frac", 0.15)),
                             test_site_frac=float(xcfg.get("test_site_frac", 0.15)),
                             seed=seed)
    del df
    sites = split["sites"]
    fr = split["frames"]
    print(f"sites: train={len(sites['train'])} val={len(sites['val'])} "
          f"test={len(sites['test'])}")
    for k, v in fr.items():
        print(f"  {k}: {len(v):,} rows")

    tr_obs = fr["train"].loc[fr["train"]["power"].notna()]
    denom_all = float(tr_obs["power"].max() - tr_obs["power"].min())
    del tr_obs

    # full feature frames — sequence channel matrix needs covariates too
    eval_keys = ("val_seen", "test_seen", "val_unseen", "test_unseen")
    eval_frames = {k: fr[k] for k in eval_keys}
    metrics_parts, model_meta = [], {}

    # ---- persistence ---------------------------------------------------------
    if "persistence" in wanted:
        t0 = time.perf_counter()
        base = PersistenceBaseline()
        # causal table over the WHOLE table (D-011 #3): lookups read only
        # t−24h < t, so unseen sites are predicted from their own history
        history = pd.concat([fr["train"][["site_id", "timestamp", "power"]]
                             ] + [ef[["site_id", "timestamp", "power"]]
                                  for ef in (fr[k] for k in eval_keys)],
                            ignore_index=True)
        base.fit(history)
        del history
        preds_p = {name: base.predict(part).to_numpy(dtype=float)
                   for name, part in eval_frames.items()}
        model_meta["persistence"] = {"training_seconds":
                                     round(time.perf_counter() - t0, 3)}

    # ---- xgboost ---------------------------------------------------------------
    if "xgboost" in wanted:
        from src.models.xgboost_model import predict_frame, train_xgboost

        t0 = time.perf_counter()
        xb, _info = train_xgboost(
            fr["train"], fr["val_seen"],
            params=dict(cfg["models"]["xgboost"]["params"]),
            seed=seed, early_stopping_rounds=int(tcfg["early_stopping_rounds"]))
        print(f"[xgboost] trained in {time.perf_counter() - t0:.1f}s | "
              f"best_iter={_info['best_iteration']}")
        preds_x = {name: predict_frame(xb, part) for name, part in eval_frames.items()}
        model_meta["xgboost"] = {"training_seconds":
                                 round(time.perf_counter() - t0, 2),
                                 "best_iteration": _info["best_iteration"]}

    # ---- sequence archs ----------------------------------------------------------
    seq_wanted = [a for a in ("lstm", "gru", "transformer") if a in wanted]
    for arch in seq_wanted:
        block = cfg["models"][arch]
        assert block.get("enabled"), f"{arch} disabled in configs/models.yaml"
        preds_s, meta_s = eval_sequence_arch(
            arch, params=dict(block["params"]), seed=seed, device=device,
            train_df=fr["train"], eval_frames=eval_frames,
            lookback=int(block["params"]["lookback_steps"]))
        metrics_parts.append(all_rows(arch, eval_frames, preds_s, denom_all))
        model_meta[arch] = meta_s

    if "persistence" in wanted:
        metrics_parts.append(all_rows("persistence_prev_day", eval_frames,
                                      preds_p, denom_all))
    if "xgboost" in wanted:
        metrics_parts.append(all_rows("xgboost", eval_frames, preds_x, denom_all))

    metrics = pd.concat(metrics_parts, ignore_index=True)
    cols = ["model", "protocol", "split", "scope", "site_id", "n_eval",
            "n_missing", "mae", "rmse", "r2", "nrmse", "daylight_n",
            "daylight_mae", "daylight_nrmse"]
    metrics_path = out_dir / "cross_site_metrics.csv"
    metrics[cols].to_csv(metrics_path, index=False)
    print(f"wrote {metrics_path.name} ({len(metrics)} rows)")

    # ---- metadata -----------------------------------------------------------------
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "device": str(torch.cuda.get_device_name(0)) if device == "cuda" else "cpu",
        "sites": {k: [int(s) for s in v] for k, v in sites.items()},
        "n_rows": {k: len(v) for k, v in fr.items()},
        "nrmse_denominator_pooled_train_range_kwh": round(denom_all, 3),
        "models_requested": wanted,
        "model_meta": model_meta,
        "python_version": sys.version.split()[0],
        "package_versions": {"torch": __import__("torch").__version__,
                             "pandas": pd.__version__, "numpy": np.__version__},
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2),
                                               encoding="utf-8")
    print(f"sites: train={sites['train']}\n       val={sites['val']}\n       "
          f"test={sites['test']}")

    # ---- report ----------------------------------------------------------------------
    write_report(out_dir, metrics, meta)
    return 0


def write_report(out_dir: Path, metrics: pd.DataFrame, meta: dict) -> None:
    def cell(r, col):
        v = getattr(r, col)
        return "n/a" if pd.isna(v) else f"{v:.3f}"

    lines = ["# Cross-site evaluation report (PRD §12, Phase 9)", "",
             f"_Generated {meta['generated_at']}; seed {meta['seed']}. Held-out "
             f"sites: val {meta['sites']['val']} / test {meta['sites']['test']}. "
             f"nRMSE denominators: pooled cross-site train range "
             f"{meta['nrmse_denominator_pooled_train_range_kwh']} kWh for every "
             f"scope (D-016)._ ", ""]
    lines += ["## Headline — test split, ALL sites", "",
              "| model | seen MAE | unseen MAE | gap % | seen RMSE | unseen RMSE | seen R² | unseen R² |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]

    def get(model, protocol):
        r = metrics[(metrics.model == model) & (metrics.protocol == protocol)
                    & (metrics.split == "test") & (metrics.scope == "ALL")]
        return r.iloc[0] if len(r) else None

    models = [m for m in metrics.model.unique()]
    order = ["persistence_prev_day", "xgboost", "lstm", "gru", "transformer"]
    models.sort(key=lambda m: order.index(m) if m in order else 99)
    for m in models:
        s, u = get(m, "seen"), get(m, "unseen")
        if s is None or u is None:
            continue
        gap = (u.mae - s.mae) / s.mae * 100 if s.mae else float("nan")
        lines.append(f"| {m} | {cell(s,'mae')} | {cell(u,'mae')} | {gap:+.0f}% "
                     f"| {cell(s,'rmse')} | {cell(u,'rmse')} "
                     f"| {cell(s,'r2')} | {cell(u,'r2')} |")
    lines.append("")
    lines += ["Reading: 'seen' = late history of training sites; 'unseen' = full "
              "history of held-out sites. The gap measures how much of each "
              "model's accuracy depends on having seen the plant during "
              "training (site identity/scale features, D-010).", ""]
    for proto in ("seen", "unseen"):
        lines += [f"## {proto} sites — test ALL (ranked by MAE)", "",
                  "| model | MAE | RMSE | R² | nRMSE | Daylight MAE | n_eval |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        sub = metrics[(metrics.protocol == proto) & (metrics.split == "test")
                      & (metrics.scope == "ALL")].sort_values("mae")
        for r in sub.itertuples():
            lines.append(f"| {r.model} | {cell(r,'mae')} | {cell(r,'rmse')} "
                         f"| {cell(r,'r2')} | {cell(r,'nrmse')} "
                         f"| {cell(r,'daylight_mae')} | {int(r.n_eval)} |")
        lines.append("")
    # unseen per-site spread for trained models
    lines += ["## Unseen test sites — per-site spread", "",
              "| model | sites | MAE min | MAE max | R² min | R² max | neg-R² sites |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    un = metrics[(metrics.protocol == "unseen") & (metrics.scope == "SITE")
                 & (metrics.split == "test")]
    for m, g in un.groupby("model"):
        lines.append(f"| {m} | {len(g)} | {g.mae.min():.3f} | {g.mae.max():.3f} "
                     f"| {g.r2.min():.3f} | {g.r2.max():.3f} "
                     f"| {int((g.r2 < 0).sum())} |")
    (out_dir / "cross_site_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote cross_site_report.md")


if __name__ == "__main__":
    raise SystemExit(main())
