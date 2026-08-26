"""Export frontend data bundles (PRD §38 pages the REST API does not cover).

The API surface is fixed by PRD §33-36; pages that need artifact-derived
aggregates (SHAP importance, per-site performance, data-quality numbers)
read snapshot JSONs instead of new endpoints. Run after any artifact
refresh:  ``conda run -n solar python scripts/export_frontend_data.py``

Outputs ``frontend/src/data/{shap_global_importance,site_summary,
data_quality,site_monthly,quality_extra}.json``. Numbers are copied
verbatim from Phase artifacts — no recomputation beyond aggregation of the
processed parquet (post-PRD graph additions, D-025).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "frontend" / "src" / "data"
CADENCE = pd.Timedelta(900, unit="s")


def shap_bundle() -> dict:
    meta = json.loads(
        (ROOT / "artifacts" / "shap" / "run_metadata.json").read_text(encoding="utf-8"))
    df = pd.read_csv(ROOT / "artifacts" / "shap" / "shap_global_importance.csv")
    return {
        "explained_run": meta["explained_model_run"],
        "sample_rows": int(meta["global_sample_rows"]),
        "additivity_max_abs_err": float(meta["additivity_max_abs_err"]),
        "features": [
            {"rank": int(r.rank), "feature": str(r.feature),
             "mean_abs_shap": round(float(r.mean_abs_shap), 4),
             "share": float(r.share_of_total)}
            for r in df.itertuples()
        ],
    }


def site_summary_bundle() -> list[dict]:
    details = pd.read_parquet(ROOT / "data" / "processed" / "site_details.parquet")
    processed_dir = ROOT / "data" / "processed" / "solar"
    out = []
    for det in details.sort_values("site_id").itertuples():
        df = pd.read_parquet(processed_dir, filters=[("site_id", "=", int(det.site_id))],
                             columns=["timestamp", "power", "is_daylight"])
        df = df.sort_values("timestamp")
        span_slots = int((df["timestamp"].iloc[-1] - df["timestamp"].iloc[0])
                         / CADENCE) + 1
        obs = df["power"].notna()
        day = obs & df["is_daylight"].astype(bool)
        p = df.loc[day, "power"]
        out.append({
            "site_id": int(det.site_id),
            "campus_id": int(det.campus_id),
            "latitude": None if pd.isna(det.latitude) else float(det.latitude),
            "longitude": None if pd.isna(det.longitude) else float(det.longitude),
            "capacity_kwp": None if pd.isna(det.capacity_kwp) else float(det.capacity_kwp),
            "first_ts": str(df["timestamp"].iloc[0]),
            "last_ts": str(df["timestamp"].iloc[-1]),
            "n_rows": int(len(df)),
            "expected_slots": span_slots,
            "row_availability_pct": round(100 * len(df) / span_slots, 2),
            "power_observed_pct": round(100 * float(obs.mean()), 2),
            "mean_daylight_kwh": round(float(p.mean()), 3) if len(p) else None,
            "max_daylight_kwh": round(float(p.max()), 3) if len(p) else None,
        })
    return out


def data_quality_bundle() -> dict:
    val = json.loads(
        (ROOT / "artifacts" / "validation_report.json").read_text(encoding="utf-8"))
    clean = json.loads(
        (ROOT / "artifacts" / "cleaning_log.json").read_text(encoding="utf-8"))
    gen, weather = val["generation"], val["weather"]
    months = weather["missing"]["by_month_pct"]
    vars_ = ["temperature", "humidity", "wind_speed"]
    timeline = [{"month": m,
                 "temperature_pct": months[m].get("temperature"),
                 "humidity_pct": months[m].get("humidity"),
                 "wind_speed_pct": months[m].get("wind_speed")}
                for m in sorted(months)]
    interp_rows = sum(op["rows_affected"] for op in clean["operations"]
                      if op["operation"].startswith("interpolate_weather"))
    return {
        "generation": {
            "duplicate_keys": int(gen["duplicate_keys"]["duplicate_key_rows"]),
            "impossible_values": len(gen["impossible"]) if isinstance(gen["impossible"], dict) else 0,
            "missing_slots": int(gen["gaps"]["total_missing_slots"]),
            "groups_checked": int(gen["gaps"]["groups_checked"]),
            "worst_sites": [
                {"site_id": w["site_id"], "missing_pct": w["missing_pct"]}
                for w in gen["gaps"]["worst_groups"][:5]
            ],
            # raw NaN share incl. night rows — night generation reports as NaN by design
            "power_missing_raw_pct": float(gen["missing"]["by_variable_pct"]["power"]),
        },
        "weather_overall_pct": {k: float(v)
                                for k, v in weather["missing"]["by_variable_pct"].items()},
        "weather_monthly": timeline,
        "cleaning": {
            "total_operations": int(clean["total_operations"]),
            "weather_interpolated_rows": int(interp_rows),
        },
        "outliers_flagged": 0,  # D-009: daylight IQR analysis flagged none
    }


def site_monthly_bundle() -> dict:
    """Monthly energy per site + campus means (Dashboard graphs, D-025).

    Energy per month = sum of mean daylight-slot power × 15-min kWh
    conversion over *observed* slots only; night rows are NaN by design and
    contribute nothing. Sites with zero observed slots in a month get null.
    """
    details = pd.read_parquet(ROOT / "data" / "processed" / "site_details.parquet")
    processed_dir = ROOT / "data" / "processed" / "solar"
    kwh_per_slot = 0.25  # 15 min at 1 kW

    per_site: dict[int, dict[str, float]] = {}
    campus_of: dict[int, int] = {}
    for det in details.sort_values("site_id").itertuples():
        sid = int(det.site_id)
        campus_of[sid] = int(det.campus_id)
        df = pd.read_parquet(processed_dir,
                             filters=[("site_id", "=", sid)],
                             columns=["timestamp", "power"])
        obs = df.loc[df["power"].notna()].copy()
        if obs.empty:
            continue
        obs["month"] = obs["timestamp"].dt.strftime("%Y-%m")
        g = obs.groupby("month")["power"].sum() * kwh_per_slot
        per_site[sid] = {m: round(float(v), 2) for m, v in g.items()}

    months = sorted({m for vals in per_site.values() for m in vals})
    by_site = [{"site_id": sid,
                "campus_id": campus_of[sid],
                "monthly_kwh": [per_site.get(sid, {}).get(m) for m in months]}
               for sid in sorted(per_site)]

    by_campus: dict[int, list[float | None]] = {}
    for row in by_site:
        acc = by_campus.setdefault(row["campus_id"], [None] * len(months))
        for i, v in enumerate(row["monthly_kwh"]):
            if v is not None:
                acc[i] = v if acc[i] is None else acc[i] + v
    n_per_campus = {c: sum(1 for r in by_site if r["campus_id"] == c)
                    for c in by_campus}
    campus_rows = [{"campus_id": c,
                    "n_sites": n_per_campus[c],
                    "monthly_kwh_mean": [None if v is None else round(v / n_per_campus[c], 2)
                                         for v in by_campus[c]]}
                   for c in sorted(by_campus)]

    return {"months": months, "sites": by_site, "campuses": campus_rows}


def quality_extra_bundle() -> dict:
    """Daylight-power histogram + per-site availability series (Quality page
    additions, D-025). Computed from the processed parquet."""
    processed_dir = ROOT / "data" / "processed" / "solar"
    details = pd.read_parquet(ROOT / "data" / "processed" / "site_details.parquet")

    hist_ids, hist_counts = [], []
    avail_rows = []
    edges = np.arange(0.0, 10.5, 0.5)  # 0-10 kW in 0.5 kW bins, fixed scale
    for det in details.sort_values("site_id").itertuples():
        sid = int(det.site_id)
        df = pd.read_parquet(processed_dir, filters=[("site_id", "=", sid)],
                             columns=["timestamp", "power", "is_daylight"])
        day_obs = df.loc[df["is_daylight"].astype(bool) & df["power"].notna(), "power"]
        counts, _ = np.histogram(day_obs, bins=edges)
        hist_ids.append(sid)
        hist_counts.append([int(c) for c in counts])
        span_slots = int((df["timestamp"].iloc[-1] - df["timestamp"].iloc[0])
                         / CADENCE) + 1
        avail_rows.append({
            "site_id": sid,
            "row_availability_pct": round(100 * len(df) / max(span_slots, 1), 2),
            "daylight_power_obs_pct": round(100 * float(
                df.loc[df["is_daylight"].astype(bool), "power"].notna().mean()), 2),
        })

    bin_labels = [f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(len(edges) - 1)]
    total = (np.sum(np.array(hist_counts), axis=0) if hist_counts
             else np.zeros(len(edges) - 1))
    return {
        "hist_bin_edges_kw": [float(e) for e in edges],
        "hist_bin_labels": bin_labels,
        # per-site matrix kept row-oriented for compactness
        "hist_by_site": [{"site_id": sid, "counts": c}
                         for sid, c in zip(hist_ids, hist_counts)],
        "hist_total": [int(c) for c in total],
        "availability": avail_rows,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    bundles = {
        "shap_global_importance.json": shap_bundle(),
        "site_summary.json": site_summary_bundle(),
        "data_quality.json": data_quality_bundle(),
        "site_monthly.json": site_monthly_bundle(),
        "quality_extra.json": quality_extra_bundle(),
    }
    for name, payload in bundles.items():
        path = OUT / name
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)} "
              f"({path.stat().st_size / 1024:.1f} kB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
