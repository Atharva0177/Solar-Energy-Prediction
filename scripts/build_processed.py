"""Phase 2 orchestrator: raw UNISOLAR CSVs -> validated, cleaned, enriched
Parquet dataset under ``data/processed/`` (PRD §§7-10, 41).

Pipeline: load -> validate (raw) -> clean (logged) -> validate (clean) ->
timezone selection (empirical) -> solar position/daylight -> campus-weather
merge -> missing-value handling -> parquet partitioned by site_id/year/month.

Artifacts written:

* ``artifacts/validation_report.json`` / ``.md``
* ``artifacts/cleaning_log.json``
* ``artifacts/timezone_decision.json``
* ``artifacts/outlier_analysis.md``

No fabricated numbers: every figure below comes from the loaded frames.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data import clean as cln
from src.data import night, schema_map, validate
from src.data.clean import CleanLog

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

WEATHER_INTERP_COLS = [
    "temperature",
    "apparent_temperature",
    "dew_point_temperature",
    "humidity",
    "wind_speed",
]

INTERP_LIMIT_STEPS = 2  # <= 30 minutes at 15-min cadence


def main() -> int:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    log = CleanLog()

    # ---- Load + map schema -------------------------------------------------
    gen = schema_map.load_generation()
    wx = schema_map.load_weather()
    sites = schema_map.load_site_details()
    print(f"loaded: gen={gen.shape}, weather={wx.shape}, sites={sites.shape}")

    # ---- Clean --------------------------------------------------------------
    NUMERIC_EXPECTED = {
        "generation": ["power"],
        "weather": [
            "temperature",
            "apparent_temperature",
            "dew_point_temperature",
            "humidity",
            "wind_speed",
            "wind_direction",
        ],
        "site_details": ["capacity_kwp", "n_panels", "latitude", "longitude"],
    }
    for name, df in (
        ("generation", gen),
        ("weather", wx),
        ("site_details", sites),
    ):
        df = cln.parse_timestamps(df, log, name)
        df = cln.coerce_numeric(df, NUMERIC_EXPECTED[name], log, name)
        df = cln.drop_exact_duplicates(df, log, name)
        if name == "generation":
            gen = df
        elif name == "weather":
            wx = df
        else:
            sites = df

    gen = cln.null_impossible_values(gen, {"power": validate.IMPOSSIBLE_RULES["power"]}, log, "generation")
    wx_rules = {c: validate.IMPOSSIBLE_RULES[c] for c in wx.columns if c in validate.IMPOSSIBLE_RULES}
    wx = cln.null_impossible_values(wx, wx_rules, log, "weather")

    # ---- Validation (post-clean) -------------------------------------------
    validation = {
        "generation": {
            "duplicate_keys": validate.check_duplicate_keys(gen, "site_id"),
            "gaps": validate.check_gaps(gen, "site_id"),
            "missing": validate.check_missing_values(gen, "site_id"),
            "impossible": validate.check_impossible_values(gen),
        },
        "weather": {
            "duplicate_keys": validate.check_duplicate_keys(wx, "campus_id"),
            "gaps": validate.check_gaps(wx, "campus_id"),
            "missing": validate.check_missing_values(wx, "campus_id"),
            "impossible": validate.check_impossible_values(wx),
        },
    }

    # ---- Timezone selection (PRD §10 prerequisite) --------------------------
    coords = night.campus_coordinates(sites)
    tz_choice = night.choose_timezone(gen.sample(frac=0.25, random_state=42), coords)
    chosen_tz = tz_choice["chosen_timezone"]
    print(f"timezone chosen: {chosen_tz} | evidence: {tz_choice['candidates']}")
    (ARTIFACTS_DIR / "timezone_decision.json").write_text(
        json.dumps(tz_choice, indent=2), encoding="utf-8"
    )
    log.add(
        "generation",
        "select_timezone_for_solar_position",
        0,
        f"chosen={chosen_tz}; evidence=timezone_decision.json",
    )

    # ---- Solar elevation + daylight ----------------------------------------
    gen = night.add_solar_position(gen, coords, chosen_tz)

    # ---- Merge campus weather onto site generation --------------------------
    n_before_merge = len(gen)
    merged = gen.merge(wx, on=["campus_id", "timestamp"], how="left", suffixes=("", "_wx"))
    assert len(merged) == n_before_merge, "campus-weather join changed row count"
    print(f"merged: {merged.shape}")

    # ---- Missing-value handling --------------------------------------------
    merged = handle_missing(merged, log)

    # ---- Outlier analysis (flag-only, artifact) -----------------------------
    outlier_md = outlier_analysis(merged, sites)

    # ---- Persist ------------------------------------------------------------
    out = merged.copy()
    out["year"] = out["timestamp"].dt.year
    out["month"] = out["timestamp"].dt.month
    parquet_dir = PROCESSED_DIR / "solar"
    # Idempotent rebuilds: legacy partitioned writer mints fresh GUID
    # filenames per run, so stale partitions must be removed first.
    if parquet_dir.exists():
        shutil.rmtree(parquet_dir)
    out.to_parquet(
        parquet_dir,
        engine="pyarrow",
        partition_cols=["site_id", "year", "month"],
        index=False,
    )
    sites.to_parquet(PROCESSED_DIR / "site_details.parquet", engine="pyarrow", index=False)
    print(f"wrote parquet under {parquet_dir} ({out.shape[0]:,} rows)")

    # ---- Reports ------------------------------------------------------------
    (ARTIFACTS_DIR / "cleaning_log.json").write_text(
        json.dumps(log.to_dict(), indent=2), encoding="utf-8"
    )
    write_validation_md(validation, chosen_tz, out)
    (ARTIFACTS_DIR / "validation_report.json").write_text(
        json.dumps(validation, indent=2, default=str), encoding="utf-8"
    )
    (ARTIFACTS_DIR / "outlier_analysis.md").write_text(outlier_md, encoding="utf-8")

    print("done.")
    return 0


def handle_missing(df: pd.DataFrame, log: CleanLog) -> pd.DataFrame:
    """Time-interpolate short weather gaps inside each campus; power untouched."""
    filled_counts = {c: 0 for c in WEATHER_INTERP_COLS}
    parts = []
    for _, sub in df.groupby("campus_id", observed=True):
        sub = sub.sort_values("timestamp").set_index("timestamp")
        for col in WEATHER_INTERP_COLS:
            na_before = int(sub[col].isna().sum())
            if na_before:
                sub[col] = sub[col].interpolate(method="time", limit=INTERP_LIMIT_STEPS)
                filled_counts[col] += na_before - int(sub[col].isna().sum())
        parts.append(sub.reset_index())
    df = pd.concat(parts, ignore_index=True).sort_values(["site_id", "timestamp"])
    for col, n in filled_counts.items():
        if n:
            log.add("merged", f"interpolate_weather:{col}", n,
                    f"method=time, limit={INTERP_LIMIT_STEPS} steps within campus")
    log.add("merged", "leave_power_missing_as_nan", int(df["power"].isna().sum()),
            "missing generation is never imputed with zeros (PRD §10)")
    log.add("merged", "no_interpolation_wind_direction", int(df["wind_direction"].isna().sum()),
            "circular quantity - linear interpolation invalid")
    return df


def outlier_analysis(df: pd.DataFrame, sites: pd.DataFrame) -> str:
    """Per-site daylight generation outliers via IQR fence. Analysis only."""
    day = df[df["is_daylight"] & df["power"].notna()]
    cap = sites.set_index("site_id")["capacity_kwp"]
    lines = [
        "# Outlier Analysis (auto-generated)",
        "",
        "Method: per-site IQR fence (Q3 + 3*IQR) on daylight-only power.",
        "Flag-only: no rows removed or altered (PRD §9).",
        "",
        "| site_id | capacity_kwp | n_day_obs | Q1 | Q3 | upper_fence | outliers | max_power |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    total_outliers = 0
    for sid, s in day.groupby("site_id"):
        q1, q3 = s["power"].quantile([0.25, 0.75])
        iqr = q3 - q1
        fence = q3 + 3 * iqr
        n_out = int((s["power"] > fence).sum())
        total_outliers += n_out
        lines.append(
            f"| {sid} | {cap.get(sid)} | {len(s):,} | {q1:.3f} | {q3:.3f} "
            f"| {fence:.3f} | {n_out:,} | {s['power'].max():.3f} |"
        )
    lines += [
        "",
        f"**Total daylight observations flagged as high-side outliers: {total_outliers:,}"
        f" of {len(day):,}**",
    ]
    return "\n".join(lines)


def write_validation_md(validation: dict, chosen_tz: str, final_df: pd.DataFrame) -> None:
    lines = [
        "# Data Quality Report (auto-generated)",
        "",
        f"_Post-cleaning state. Timezone used for solar position: **{chosen_tz}**._",
        "",
        f"- Final merged rows: {final_df.shape[0]:,}",
        f"- Final columns: `{', '.join(final_df.columns)}`",
        "",
    ]
    for section, checks in validation.items():
        lines += [f"## {section}", ""]
        dupes = checks["duplicate_keys"]
        lines.append(f"- Duplicate keys ({dupes['key']}): **{dupes['duplicate_key_rows']}**")
        gaps = checks["gaps"]
        lines.append(
            f"- Missing timestamps ({gaps['frequency']}): "
            f"**{gaps['total_missing_slots']}** slots across {gaps['groups_checked']} groups"
        )
        worst = gaps.get("worst_groups", [])[:3]
        for g in worst:
            key = [k for k in g if k not in ("expected_slots", "actual_rows", "missing_slots", "missing_pct")][0]
            lines.append(
                f"  - worst: {key}={g[key]} missing {g['missing_pct']}% "
                f"({g['missing_slots']}/{g['expected_slots']} slots)"
            )
        miss = checks["missing"]["by_variable_pct"]
        if miss:
            lines.append("- Missing % by variable: " + ", ".join(f"`{k}`={v}" for k, v in miss.items()))
        imp = checks["impossible"]
        if imp:
            lines.append("- Impossible values (nulled during cleaning): ")
            for rule, info in imp.items():
                lines.append(f"  - `{rule}`: {info.get('count')} ({info})")
        else:
            lines.append("- Impossible values: none detected")
        lines.append("")
    (ARTIFACTS_DIR / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
