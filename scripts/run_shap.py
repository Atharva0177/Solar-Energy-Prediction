"""Phase 10 orchestrator: SHAP explainability for the XGBoost forecaster.

Explains the Phase 6 run ``xgboost-site-all-h1-v1`` on the canonical TEST
split (D-011), answering PRD §29:

* global importance      → mean |SHAP| table + summary bar/beeswarm PNGs
* local explanations     → tagged scenarios (night / clear noon / ramp /
                           overcast afternoon) with waterfall PNGs and a
                           contribution CSV per prediction
* contribution plots     → dependence scatter of the top-3 features,
                           colored by automatic interaction pick

Protocol notes:

* Global sample = seeded subsample (default 20k) of observed-target test
  rows — exact TreeExplainer values on a fixed slice, not approximations.
* Local scenarios are picked from the FULL test frame so rare conditions
  still find candidates; their SHAP rows come from the same explainer.
* Nothing here retrains or retunes the model; artifacts are read-only views
  of ``models/xgboost_site_all_h1_v1.json``.

Artifacts → ``artifacts/shap/``:

* ``shap_global_importance.csv``   — ranked mean |SHAP| table
* ``shap_summary_bar.png``         — global bar plot
* ``shap_beeswarm.png``            — sample-level value/impact beeswarm
* ``shap_dependence_<rank>_<feature>.png``
* ``shap_waterfall_<tag>.png``     — one per local scenario
* ``shap_local_contributions.csv`` — top-k contributions per scenario
* ``run_metadata.json``

Usage: ``conda run -n solar python scripts/run_shap.py [--sample 20000]``
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import shap

from src.config import load_config
from src.data.splits import chronological_split
from src.explain.shap_explain import (
    contribution_table,
    explain_matrix,
    global_importance,
    sample_rows,
    select_local_examples,
)
from src.models.xgboost_model import load_xgboost_model, prepare_matrix

RUN_NAME = "shap-xgboost-test-v1"
MODEL_PATH = REPO_ROOT / "models" / "xgboost_site_all_h1_v1.json"
META_PATH = REPO_ROOT / "artifacts" / "xgboost" / "run_metadata.json"
OUT_DIR = REPO_ROOT / "artifacts" / "shap"

SURFACE = "#fcfcfb"


def save_current_fig(path: Path) -> None:
    fig = plt.gcf()
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight", dpi=150)
    plt.close("all")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=20000,
                    help="test rows for global SHAP (seeded)")
    args = ap.parse_args()

    cfg = load_config()
    tcfg = cfg["training"]
    seed = int(tcfg["seed"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    reg, xgb_meta = load_xgboost_model(MODEL_PATH, META_PATH)
    names = reg.feature_cols_["categorical"] + reg.feature_cols_["numeric"]

    df = pd.read_parquet(REPO_ROOT / cfg["paths"]["features_dir"])
    _, _, test = chronological_split(
        df, ratios=(tcfg["train_ratio"], tcfg["val_ratio"], tcfg["test_ratio"]))
    del df
    print(f"test frame: {len(test):,} rows")

    # ---- global ------------------------------------------------------------
    sample = sample_rows(test, n=args.sample, seed=seed)
    print(f"global sample: {len(sample):,} rows (seed {seed})")
    sv, base = explain_matrix(reg, sample)
    imp = global_importance(sv, names)
    imp_path = OUT_DIR / "shap_global_importance.csv"
    imp.to_csv(imp_path, index=False)

    X_sample = prepare_matrix(sample, reg.feature_cols_)
    explanation = shap.Explanation(
        values=sv, base_values=np.full(len(sample), base),
        data=X_sample.to_numpy(),
        feature_names=list(X_sample.columns))

    shap.plots.bar(explanation, max_display=15, show=False)
    save_current_fig(OUT_DIR / "shap_summary_bar.png")
    shap.plots.beeswarm(explanation, max_display=15, show=False)
    save_current_fig(OUT_DIR / "shap_beeswarm.png")

    dep_paths = []
    for i in range(min(3, len(imp))):
        feat = imp.iloc[i]["feature"]
        shap.plots.scatter(explanation[:, feat], color=explanation, show=False)
        p = OUT_DIR / f"shap_dependence_{i+1}_{feat}.png"
        save_current_fig(p)
        dep_paths.append(p.name)

    top15 = ", ".join(f"{r.feature} ({r.mean_abs_shap:.3f})"
                      for r in imp.head(15).itertuples())
    print(f"top features by mean |SHAP|:\n  {top15}")

    # ---- local -------------------------------------------------------------
    tags = select_local_examples(test)
    contrib_parts, local_meta = [], {}
    for tag, idx in tags.items():
        row_df = test.loc[[idx]]
        sv_row, base_row = explain_matrix(reg, row_df)
        X_row = prepare_matrix(row_df, reg.feature_cols_)
        pred = float(reg.predict(X_row)[0])
        expl_row = shap.Explanation(
            values=sv_row[0], base_values=base_row,
            data=X_row.iloc[0].to_numpy(),
            feature_names=list(X_row.columns))
        shap.plots.waterfall(expl_row, max_display=10, show=False)
        save_current_fig(OUT_DIR / f"shap_waterfall_{tag}.png")

        ct = contribution_table(X_row.iloc[0], sv_row[0], base_row)
        ct.insert(0, "scenario", tag)
        ct.insert(1, "site_id", int(test.at[idx, "site_id"]))
        ct.insert(2, "timestamp", str(test.at[idx, "timestamp"]))
        ct["prediction_kwh"] = round(pred, 3)
        contrib_parts.append(ct)
        pos = ct[(ct.shap > 0) & ~ct.feature.str.startswith("<")].head(3)
        neg = ct[(ct.shap < 0) & ~ct.feature.str.startswith("<")].head(3)
        fmt = lambda g: "; ".join(f"{r.feature} {r.shap:+.2f}"
                                  for r in g.itertuples()) or "-"
        local_meta[tag] = {
            "site_id": int(test.at[idx, "site_id"]),
            "timestamp": str(test.at[idx, "timestamp"]),
            "power_observed": round(float(test.at[idx, "power"]), 3),
            "prediction": round(pred, 3),
            "base_value": round(base_row, 4),
            "top_positive": fmt(pos),
            "top_negative": fmt(neg),
        }
        print(f"[{tag}] site {local_meta[tag]['site_id']} "
              f"{local_meta[tag]['timestamp']} pred={pred:.2f} "
              f"(observed {local_meta[tag]['power_observed']:.2f})\n"
              f"    + {fmt(pos)} | - {fmt(neg)}")

    contrib = pd.concat(contrib_parts, ignore_index=True)
    contrib_path = OUT_DIR / "shap_local_contributions.csv"
    contrib.to_csv(contrib_path, index=False)

    # ---- metadata + tracking -------------------------------------------------
    meta = {
        "run_name": RUN_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "explained_model_run": xgb_meta.get("run_name"),
        "explainer": "shap.TreeExplainer (exact, tree_path_dependent)",
        "shap_version": shap.__version__,
        "seed": seed,
        "split_protocol": "D-011 per-site chronological (TEST split)",
        "global_sample_rows": int(len(sample)),
        "n_features_explained": len(names),
        "additivity_max_abs_err": float(np.abs(
            base + sv.sum(axis=1) - reg.predict(X_sample)).max()),
        "top10_mean_abs_shap": imp.head(10)[
            ["rank", "feature", "mean_abs_shap", "share_of_total"]
        ].round(5).to_dict(orient="records"),
        "local_scenarios": local_meta,
        "artifacts": sorted(p.name for p in OUT_DIR.iterdir()
                            if p.suffix in (".png", ".csv")),
    }
    meta_path = OUT_DIR / "run_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote artifacts under artifacts/shap/ "
          f"(additivity err {meta['additivity_max_abs_err']:.2e})")

    mlflow.set_tracking_uri((REPO_ROOT / "mlruns").as_uri())
    mlflow.set_experiment("unisolar")
    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.log_params({
            "explained_model": xgb_meta.get("run_name"),
            "explainer": "TreeExplainer",
            "split": "test", "seed": seed,
            "global_sample_rows": len(sample),
        })
        mlflow.log_metrics({
            **{f"mean_abs_shap_rank{i+1}": float(r.mean_abs_shap)
               for i, r in enumerate(imp.head(10).itertuples())},
            "additivity_max_abs_err": meta["additivity_max_abs_err"],
        })
        for p in OUT_DIR.iterdir():
            if p.suffix in (".png", ".csv", ".json"):
                mlflow.log_artifact(str(p), artifact_path="shap")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
