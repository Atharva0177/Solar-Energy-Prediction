"""Automatic UNISOLAR dataset inspection (PRD §6).

Locates tabular dataset files under ``unisolar/``, profiles each one, and writes:

* ``artifacts/data_profile.json`` - machine-readable profile
* ``artifacts/data_profile.md``   - human-readable report

Makes no schema assumptions: timestamp / site / target columns are *detected*
heuristically and reported as candidates with evidence, never presumed.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "unisolar"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

TABULAR_EXTENSIONS = {".csv", ".parquet", ".pq", ".feather", ".tsv"}

# --- Heuristic keyword tables (detection only; verified against real data) ---

TIMESTAMP_NAME_RE = re.compile(
    r"(time|timestamp|date|datetime|utc|local)", re.IGNORECASE
)

SITE_ID_NAME_RE = re.compile(
    r"(site|plant|station|system|asset|location|loc)_?(id|name|code)?", re.IGNORECASE
)

TARGET_NAME_RE = re.compile(
    r"(power|generation|generated|energy|yield|output|production|pv)", re.IGNORECASE
)

# Obvious unit metadata embedded in column names, e.g. "power_kW" or "temp(C)".
UNIT_HINT_RE = re.compile(
    r"[\(_\-\s](?P<unit>"
    r"k?W(?:h)?|M?Wh|mwh|kw|w|"
    r"celsius|fahrenheit|deg[cfk]|c|f|"
    r"kpa|hpa|pa|mbar|bar|mm|cm|km|h|kmh|km/h|mph|ms|deg|%"
    r")[\)\]_\-\s]?$",
    re.IGNORECASE,
)

DATETIME_PARSE_SAMPLE_ROWS = 2000


def find_tabular_files(root: Path) -> list[Path]:
    """Requirement 1-3: locate and list all tabular dataset files."""
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in TABULAR_EXTENSIONS and p.is_file())


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet" or path.suffix.lower() == ".pq":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".feather":
        return pd.read_feather(path)
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(path, sep=sep)


def try_parse_datetimes(series: pd.Series) -> tuple[pd.Series | None, float]:
    """Attempt to parse a Series as datetimes on a sample.

    Returns (parsed_series_or_None, parse_success_rate).
    """
    sample = series.dropna().astype(str).head(DATETIME_PARSE_SAMPLE_ROWS)
    if sample.empty:
        return None, 0.0
    parsed = pd.to_datetime(sample, errors="coerce", utc=False, format="mixed")
    rate = float(parsed.notna().mean())
    if rate < 0.9:
        return None, rate
    full = pd.to_datetime(series, errors="coerce", utc=False, format="mixed")
    return full, rate


def detect_timestamp_columns(df: pd.DataFrame) -> dict[str, dict]:
    """Requirement 9: detect candidate timestamp columns."""
    candidates = {}
    for col in df.columns:
        name_match = bool(TIMESTAMP_NAME_RE.search(str(col)))
        already_dt = pd.api.types.is_datetime64_any_dtype(df[col])
        parsed = None
        rate = 0.0
        if already_dt:
            parsed, rate = df[col], 1.0
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            parsed, rate = try_parse_datetimes(df[col])
        numeric_like_time = (
            pd.api.types.is_numeric_dtype(df[col])
            and TIMESTAMP_NAME_RE.search(str(col)) is not None
        )
        if parsed is not None and rate >= 0.9:
            candidates[col] = {
                "parse_success_rate": round(rate, 4),
                "matched_by_name": name_match,
                "min": str(parsed.min()),
                "max": str(parsed.max()),
                "dtype": "datetime64",
            }
        elif numeric_like_time:
            candidates[col] = {
                "parse_success_rate": None,
                "matched_by_name": True,
                "note": "numeric column with time-like name (epoch/unix?)",
                "dtype": str(df[col].dtype),
            }
    return candidates


def detect_site_columns(df: pd.DataFrame) -> dict[str, dict]:
    """Requirement 10: detect candidate site/grouping-ID columns."""
    suffix_id_re = re.compile(r"(key|id|code|num)$", re.IGNORECASE)
    candidates = {}
    for col in df.columns:
        nunique = int(df[col].nunique(dropna=True))
        name_match = bool(SITE_ID_NAME_RE.search(str(col)))
        is_low_card = 1 < nunique <= max(1000, int(len(df) * 0.5))
        is_suffix_id = bool(suffix_id_re.search(str(col)))
        if name_match or (nunique <= 500 and not pd.api.types.is_float_dtype(df[col])):
            candidates[col] = {
                "unique_values": nunique,
                "matched_by_name": name_match,
                "matched_by_id_suffix": is_suffix_id,
                # Plausible grouping key: semantic ID name OR Key/ID-style
                # suffix on a low-cardinality column (e.g. CampusKey).
                "plausible_id_column": bool(is_low_card and (name_match or is_suffix_id)),
                "sample_values": [
                    _json_safe(v) for v in df[col].dropna().unique()[:10]
                ],
            }
    return candidates


def detect_target_columns(df: pd.DataFrame) -> dict[str, dict]:
    """Requirement 11: detect candidate PV-generation/target columns."""
    candidates = {}
    for col in df.columns:
        name_match = TARGET_NAME_RE.search(str(col))
        if name_match and pd.api.types.is_numeric_dtype(df[col]):
            s = df[col]
            zero_frac = float((s == 0).mean()) if len(s) else 0.0
            candidates[col] = {
                "matched_keyword": name_match.group(0),
                "dtype": str(s.dtype),
                "min": _json_safe(float(s.min())),
                "max": _json_safe(float(s.max())),
                "mean": _json_safe(float(s.mean())),
                "zero_fraction": round(zero_frac, 4),
            }
    return candidates


def detect_units(df: pd.DataFrame) -> dict[str, list[str]]:
    """Requirement 17: identify obvious unit metadata in column names."""
    found: dict[str, list[str]] = {}
    for col in df.columns:
        m = UNIT_HINT_RE.search(str(col))
        if m:
            found.setdefault(m.group("unit").lower(), []).append(str(col))
    return found


def sampling_frequency(ts: pd.Series) -> dict:
    """Requirement 14: determine sampling frequency from a parsed timestamp series."""
    ordered = ts.sort_values().dropna()
    diffs = ordered.diff().dropna()
    if diffs.empty:
        return {"median_interval": None, "mode_interval": None}
    mode_delta = diffs.mode().iloc[0]
    mode_count = int((diffs == mode_delta).sum())
    irregular_frac = round(float((diffs != mode_delta).mean()), 4)
    return {
        "median_interval": str(diffs.median()),
        "mode_interval": str(mode_delta),
        "mode_interval_share": round(mode_count / len(diffs), 4),
        "irregular_gap_fraction": irregular_frac,
        "max_gap": str(diffs.max()),
        "min_gap": str(diffs.min()),
    }


def sampling_frequency_grouped(ts: pd.Series, group: pd.Series | None = None) -> dict:
    """Requirement 14: sampling frequency computed *within* each site/group.

    Global diffs are meaningless when many sites share one timestamp grid.
    """
    if group is None:
        return sampling_frequency(ts)

    frame = pd.DataFrame({"ts": ts, "g": group}).dropna().sort_values("ts")
    diffs = frame.groupby("g", observed=True)["ts"].diff().dropna()
    if diffs.empty:
        return {"mode_interval": None}

    mode_delta = diffs.mode().iloc[0]
    per_group_mode = (
        frame.groupby("g", observed=True)["ts"]
        .apply(lambda s: s.diff().dropna().mode().iloc[0] if len(s) > 1 else None)
        .dropna()
        .astype(str)
        .to_dict()
    )
    return {
        "grouped_by": True,
        "median_interval": str(diffs.median()),
        "mode_interval": str(mode_delta),
        "mode_interval_share": round(float((diffs == mode_delta).mean()), 4),
        "irregular_gap_fraction": round(float((diffs != mode_delta).mean()), 4),
        "max_gap": str(diffs.max()),
        "min_gap": str(diffs.min()),
        "per_group_mode_intervals": per_group_mode,
    }


def missing_percentages(df: pd.DataFrame) -> dict[str, float]:
    """Requirement 12."""
    pct = (df.isna().mean() * 100.0).round(3)
    return {str(k): float(v) for k, v in pct.items()}


def categorical_summary(df: pd.DataFrame, max_cardinality: int = 50) -> dict[str, dict]:
    """Requirement 13: unique values for categorical-looking columns."""
    out = {}
    for col in df.columns:
        nunique = int(df[col].nunique(dropna=True))
        if nunique <= max_cardinality:
            vc = df[col].value_counts(dropna=False).head(max_cardinality)
            out[str(col)] = {
                "n_unique": nunique,
                "value_counts": {_json_safe(k): int(v) for k, v in vc.items()},
            }
    return out


def duplicate_summary(df: pd.DataFrame, ts_col: str | None, group_col: str | None) -> dict:
    """Requirements 15-16. Key duplicates use the detected ID column + timestamp."""
    summary = {
        "fully_duplicated_rows": int(df.duplicated().sum()),
        "total_rows": int(len(df)),
    }
    if ts_col is not None:
        if group_col:
            dup_mask = df.groupby(group_col, observed=True)[ts_col].transform(
                lambda s: s.duplicated(keep=False)
            )
            summary["duplicated_timestamp_values_within_group"] = int(dup_mask.sum())
        else:
            summary["duplicated_timestamp_values"] = int(df[ts_col].duplicated(keep=False).sum())
        keys = ([group_col] if group_col else []) + [ts_col]
        summary["duplicated_keys"] = {
            "+".join(keys): int(df.duplicated(subset=keys, keep=False).sum())
        }
    return summary


def _json_safe(value):
    """Convert numpy/pandas scalars into JSON-serializable Python objects."""
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def profile_file(path: Path) -> dict:
    """Requirements 4-18 for a single file."""
    df = load_table(path)
    print(f"\n=== {path.name} ===")
    print(f"shape: {df.shape[0]} rows x {df.shape[1]} cols")

    ts_candidates = detect_timestamp_columns(df)
    primary_ts = next((c for c, meta in ts_candidates.items() if meta.get("parse_success_rate")), None)

    site_candidates = detect_site_columns(df)
    target_candidates = detect_target_columns(df)

    # Finest-grained plausible ID column wins (site beats campus).
    plausible = [
        (c, m["unique_values"])
        for c, m in site_candidates.items()
        if m.get("plausible_id_column")
    ]
    group_col = max(plausible, key=lambda t: t[1])[0] if plausible else None

    # Re-parse to datetime for interval math; detection only proves parsability.
    freq: dict = {}
    ts_series = None
    if primary_ts:
        ts_series = df[primary_ts]
        if not pd.api.types.is_datetime64_any_dtype(ts_series):
            ts_series = pd.to_datetime(ts_series, errors="coerce", format="mixed")
        freq = sampling_frequency_grouped(
            ts_series, df[group_col] if group_col else None
        )

    dupes = duplicate_summary(df, primary_ts, group_col)

    profile = {
        "file": str(path.relative_to(REPO_ROOT)),
        "size_bytes": path.stat().st_size,
        "format": path.suffix.lower().lstrip("."),
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "head_5": [_row_to_json(r) for r in df.head(5).to_dict(orient="records")],
        "tail_5": [_row_to_json(r) for r in df.tail(5).to_dict(orient="records")],
        "timestamp_columns": ts_candidates,
        "primary_timestamp_column": primary_ts,
        "grouping_column": group_col,
        "site_id_candidates": site_candidates,
        "target_candidates": target_candidates,
        "missing_percentage": missing_percentages(df),
        "categorical_summary": categorical_summary(df),
        "sampling_frequency": freq,
        "duplicates": dupes,
        "unit_metadata_in_names": detect_units(df),
        "memory_usage_mb": round(float(df.memory_usage(deep=True).sum()) / 1e6, 2),
    }
    return profile


def _row_to_json(record: dict) -> dict:
    return {str(k): _json_safe(v) for k, v in record.items()}


def write_markdown(profiles: list[dict], out_path: Path) -> None:
    lines = [
        "# Data Profile (auto-generated)",
        "",
        f"_Generated by `scripts/inspect_dataset.py` on "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. "
        f"No schema assumptions made; detections are heuristic candidates._",
        "",
        "## File inventory",
        "",
        "| File | Format | Rows | Columns | Size |",
        "|---|---|---:|---:|---:|",
    ]
    for p in profiles:
        size_mb = p["size_bytes"] / 1e6
        lines.append(
            f"| `{p['file']}` | {p['format']} | {p['shape']['rows']:,} "
            f"| {p['shape']['columns']} | {size_mb:.2f} MB |"
        )

    for p in profiles:
        lines += ["", f"## `{p['file']}`", ""]
        lines += ["### Columns & dtypes", "", "| Column | Dtype | Missing % |", "|---|---|---:|"]
        miss = p["missing_percentage"]
        for col, dt in p["dtypes"].items():
            lines.append(f"| `{col}` | {dt} | {miss.get(col, 0.0)} |")

        lines += ["", "### Detected roles", ""]
        lines.append(f"- Primary timestamp column: `{p['primary_timestamp_column']}`")
        lines.append(f"- Grouping (ID) column: `{p['grouping_column']}`")
        if p["sampling_frequency"]:
            sf = p["sampling_frequency"]
            lines.append(f"- Sampling interval (mode, within-group): **{sf['mode_interval']}** "
                         f"(share {sf['mode_interval_share']:.1%}, irregular gaps {sf['irregular_gap_fraction']:.2%})")
            per_group = sf.get("per_group_mode_intervals") or {}
            if per_group:
                lines.append(f"- Per-group mode intervals: "
                             + ", ".join(f"`{g}`: {v}" for g, v in sorted(per_group.items())))
        sites = [c for c, m in p["site_id_candidates"].items() if m.get("plausible_id_column")]
        lines.append(f"- Plausible site-ID columns: {[f'`{c}`' for c in sites] or 'none'}")
        targets = list(p["target_candidates"].keys())
        lines.append(f"- Candidate target columns: {[f'`{t}`' for t in targets] or 'none'}")

        lines += ["", "### Duplicates", ""]
        d = p["duplicates"]
        lines.append(f"- Fully duplicated rows: {d['fully_duplicated_rows']:,}")
        for key, cnt in d.get("duplicated_keys", {}).items():
            lines.append(f"- Duplicate `{key}` keys (any occurrence counted): {cnt:,}")

        units = p["unit_metadata_in_names"]
        if units:
            lines += ["", "### Unit hints in column names", ""]
            for unit, cols in sorted(units.items()):
                lines.append(f"- `{unit}`: {[f'`{c}`' for c in cols]}")

        cats = p["categorical_summary"]
        small = {k: v for k, v in cats.items() if v["n_unique"] <= 20}
        if small:
            lines += ["", "### Low-cardinality columns", ""]
            for col, info in small.items():
                vals = ", ".join(f"`{v}` ({n})" for v, n in list(info["value_counts"].items())[:20])
                lines.append(f"- `{col}` ({info['n_unique']} unique): {vals}")

        lines += ["", "### First 5 rows", "", "```", _rows_as_text(p["head_5"]), "```"]
        lines += ["", "### Last 5 rows", "", "```", _rows_as_text(p["tail_5"]), "```"]

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _rows_as_text(rows: list[dict], max_cols: int = 8) -> str:
    if not rows:
        return "(empty)"
    cols = list(rows[0].keys())[:max_cols]
    header = " | ".join(cols)
    sep = "-+-".join("-" * max(len(c), 3) for c in cols)
    body = "\n".join(
        " | ".join(str(r.get(c, ""))[:24] for c in cols) for r in rows
    )
    return "\n".join([header, sep, body])


def main() -> int:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    files = find_tabular_files(DATA_DIR)
    print(f"Tabular files found under {DATA_DIR}:")
    for f in files:
        print(f"  - {f.name} ({f.stat().st_size / 1e6:.2f} MB)")
    if not files:
        print("ERROR: no tabular files found.", file=sys.stderr)
        return 1

    profiles = [profile_file(f) for f in files]

    license_files = [
        p.name for p in DATA_DIR.rglob("*")
        if p.is_file() and re.search(r"(license|licence|readme|terms)", p.name, re.IGNORECASE)
    ]

    bundle = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_directory": str(DATA_DIR.relative_to(REPO_ROOT)),
        "files_found": [str(f.relative_to(REPO_ROOT)) for f in files],
        "license_or_readme_files_present": license_files,
        "profiles": profiles,
        "_notes": [
            "Detections are heuristic candidates, not verified facts.",
            "License/usage terms must still be confirmed manually (PRD §5.1).",
        ],
    }

    json_path = ARTIFACTS_DIR / "data_profile.json"
    md_path = ARTIFACTS_DIR / "data_profile.md"
    json_path.write_text(json.dumps(bundle, indent=2, default=_json_safe), encoding="utf-8")
    write_markdown(profiles, md_path)

    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
