"""Phase 4 orchestrator: naive baselines -> baseline_metrics.csv (PRD §21).

Protocol (D-011):
* per-site chronological 70/15/15 split of the processed dataset (PRD §11.2);
* baselines fit on the TRAIN slice only — val/test targets never seen at fit;
* metrics reported on validation and test, overall AND per site (D-010),
  all-period AND daylight-only (PRD §10/§25);
* nRMSE denominator = train-slice observed range (max-min) of power, pooled
  for ALL rows and per site for SITE rows — documented here and in
  DECISIONS.md D-011.

Artifacts written:

* ``artifacts/baselines/baseline_metrics.csv``
* ``artifacts/baselines/baselines_report.md``

Every number in the report comes from this run's computed rows.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.splits import chronological_split
from src.models.baseline import MeanBaseline, PersistenceBaseline, ZeroBaseline
from src.training.evaluate import regression_metrics

PROCESSED = REPO_ROOT / "data" / "processed" / "solar"
OUT_DIR = REPO_ROOT / "artifacts" / "baselines"

COLS = ["site_id", "timestamp", "power", "is_daylight"]


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED, columns=COLS)
    df = df.sort_values(["site_id", "timestamp"]).reset_index(drop=True)
    print(f"loaded {df.shape[0]:,} rows, {df['site_id'].nunique()} sites")
    return df


def fmt(v) -> str:
    return "n/a" if pd.isna(v) else f"{v:.3f}"


def score(name, split_name, scope, site_id, eval_df, pred, denom,
          fit_s=None, pred_s=None) -> dict:
    m = regression_metrics(
        eval_df["power"], pred,
        daylight=eval_df["is_daylight"].to_numpy() if "is_daylight" in eval_df else None,
        denom=denom,
    )
    return {
        "baseline": name, "split": split_name, "scope": scope,
        "site_id": "" if site_id is None else site_id,
        **m,
        "fit_seconds": "" if fit_s is None else round(fit_s, 4),
        "predict_seconds": "" if pred_s is None else round(pred_s, 4),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    train, val, test = chronological_split(df)

    # nRMSE denominators come from the TRAIN slice only.
    tr_obs = train.loc[train["power"].notna(), "power"]
    denom_all = float(tr_obs.max() - tr_obs.min())
    per_site_range = (
        train.loc[tr_obs.index]
        .groupby("site_id", observed=True)["power"]
        .agg(lambda s: float(s.max() - s.min()))
    )
    site_ranges = {sid: (v if v > 0 else None) for sid, v in per_site_range.items()}
    print(f"train={len(train):,} val={len(val):,} test={len(test):,} "
          f"| train range (nRMSE denominator) = {denom_all:.3f} kWh")

    # Stat-based baselines fit on TRAIN only. Persistence is non-trainable
    # and strictly causal (lookup at t reads only the t-24h row), so it is
    # given the full processed series as its lag table — see D-011.
    models = {
        "zero": lambda: (ZeroBaseline(), train),
        "mean_global": lambda: (MeanBaseline(scope="global"), train),
        "mean_site": lambda: (MeanBaseline(scope="site"), train),
        "persistence_prev_day": lambda: (PersistenceBaseline(), df),
    }

    rows: list[dict] = []
    for name, make in models.items():
        t0 = time.perf_counter()
        model, fit_frame = make()
        model.fit(fit_frame)
        fit_s = time.perf_counter() - t0

        for split_name, part in (("val", val), ("test", test)):
            t0 = time.perf_counter()
            pred = model.predict(part)
            pred_s = time.perf_counter() - t0

            rows.append(score(name, split_name, "ALL", None, part, pred,
                              denom_all, fit_s, pred_s))
            for sid, sub in part.groupby("site_id", observed=True):
                rows.append(score(name, split_name, "SITE", sid, sub,
                                  pred.loc[sub.index], site_ranges.get(sid)))

    out = pd.DataFrame(rows)[[
        "baseline", "split", "scope", "site_id", "n_eval", "n_missing",
        "mae", "rmse", "r2", "nrmse", "daylight_n", "daylight_mae",
        "daylight_nrmse", "fit_seconds", "predict_seconds",
    ]]
    csv_path = OUT_DIR / "baseline_metrics.csv"
    out.to_csv(csv_path, index=False)
    print(f"wrote {csv_path} ({len(out)} rows)")

    write_report(out, denom_all, len(train), len(val), len(test))
    return 0


def write_report(out: pd.DataFrame, denom: float, n_tr: int, n_va: int, n_te: int) -> None:
    test_rows = out[(out["scope"] == "ALL") & (out["split"] == "test")].set_index("baseline")
    val_rows = out[(out["scope"] == "ALL") & (out["split"] == "val")].set_index("baseline")
    order = ["persistence_prev_day", "mean_site", "mean_global", "zero"]

    lines = [
        "# Baselines Report (auto-generated)",
        "",
        f"_Protocol (D-011): per-site chronological 70/15/15 split "
        f"(train {n_tr:,} / val {n_va:,} / test {n_te:,} rows). Fit on train only. "
        f"nRMSE denominator = train-slice observed range of power = "
        f"{denom:.2f} kWh pooled (SITE rows use each site's own train range). "
        f"All figures from this run's `baseline_metrics.csv`._",
        "",
        "## Test-split summary (ALL sites)",
        "",
        "| baseline | MAE | RMSE | R² | nRMSE | Daylight MAE | Daylight nRMSE | n_eval | preds missing |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for b in [x for x in order if x in test_rows.index]:
        m = test_rows.loc[b]
        lines.append(
            f"| {b} | {fmt(m['mae'])} | {fmt(m['rmse'])} | {fmt(m['r2'])} | {fmt(m['nrmse'])} "
            f"| {fmt(m['daylight_mae'])} | {fmt(m['daylight_nrmse'])} | {int(m['n_eval']):,} "
            f"| {int(m['n_missing']):,} |"
        )

    lines += [
        "",
        "## Validation-split summary (ALL sites)",
        "",
        "| baseline | MAE | RMSE | R² | nRMSE | Daylight MAE | Daylight nRMSE | n_eval | preds missing |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for b in [x for x in order if x in val_rows.index]:
        m = val_rows.loc[b]
        lines.append(
            f"| {b} | {fmt(m['mae'])} | {fmt(m['rmse'])} | {fmt(m['r2'])} | {fmt(m['nrmse'])} "
            f"| {fmt(m['daylight_mae'])} | {fmt(m['daylight_nrmse'])} | {int(m['n_eval']):,} "
            f"| {int(m['n_missing']):,} |"
        )

    lines += [
        "",
        "## Notes",
        "",
        "- Persistence is the primary baseline (PRD §21): prediction at t uses "
        "the observation at t−24h. The lookup table holds the full processed "
        "series but every read is strictly backward (t−24h < t), so no future "
        "information is used (D-011). Missing prior-day observations yield NaN "
        "predictions, counted under `preds missing`, never zeros (D-008).",
        "- Per-site rows (`scope=SITE`) live in the CSV and quantify across-site "
        "spread (D-010); this summary shows pooled ALL rows.",
        "- Daylight-only columns use the Phase-2 `is_daylight` flag so night "
        "zeros cannot flatter a baseline (PRD §10).",
    ]
    (OUT_DIR / "baselines_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote baselines_report.md")


if __name__ == "__main__":
    raise SystemExit(main())
