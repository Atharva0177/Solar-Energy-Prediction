"""Data-quality validation (PRD §9).

Pure checks — no mutation. Every function returns plain dicts/lists so the
orchestrator can serialize them into ``artifacts/validation_report.json``.

Checks: missing timestamps (gaps), duplicate keys, missing values by
site/variable/date, impossible values.
"""

from __future__ import annotations

import pandas as pd

IMPOSSIBLE_RULES = {
    # canonical column -> (predicate name, boolean Series factory)
    "power": ("negative_power", lambda s: s < 0),
    "humidity": ("humidity_out_of_range", lambda s: (s < 0) | (s > 100)),
    "wind_speed": ("negative_wind_speed", lambda s: s < 0),
    "wind_direction": (
        "wind_direction_out_of_range",
        lambda s: (s < 0) | (s > 360),
    ),
    "temperature": ("temperature_implausible", lambda s: (s < -30) | (s > 60)),
    "apparent_temperature": (
        "apparent_temperature_implausible",
        lambda s: (s < -30) | (s > 60),
    ),
    "dew_point_temperature": (
        "dew_point_implausible",
        lambda s: (s < -30) | (s > 60),
    ),
}


def check_duplicate_keys(df: pd.DataFrame, id_col: str, ts_col: str = "timestamp") -> dict:
    dup_mask = df.duplicated(subset=[id_col, ts_col], keep=False)
    return {
        "key": f"{id_col}+{ts_col}",
        "duplicate_key_rows": int(dup_mask.sum()),
        "affected_groups": int(df.loc[dup_mask, id_col].nunique()),
    }


def check_gaps(df: pd.DataFrame, id_col: str, freq: str = "15min") -> dict:
    """Missing timestamps: expected grid slots vs actual rows per group."""
    per_group = []
    for gid, sub in df.groupby(id_col, observed=True):
        ts = pd.DatetimeIndex(sub["timestamp"].dropna())
        if ts.empty:
            continue
        grid = pd.date_range(ts.min(), ts.max(), freq=freq)
        n_missing = len(grid.difference(ts))
        per_group.append(
            {
                str(id_col): _scalar(gid),
                "expected_slots": int(len(grid)),
                "actual_rows": int(len(pd.unique(ts))),
                "missing_slots": int(n_missing),
                "missing_pct": round(100.0 * n_missing / max(len(grid), 1), 3),
            }
        )
    total_missing = sum(g["missing_slots"] for g in per_group)
    return {
        "frequency": freq,
        "groups_checked": len(per_group),
        "total_missing_slots": total_missing,
        "worst_groups": sorted(per_group, key=lambda g: -g["missing_pct"])[:10],
        "per_group": per_group,
    }


def check_missing_values(df: pd.DataFrame, id_col: str, ts_col: str = "timestamp") -> dict:
    """Missingness by variable, by group, and by month."""
    by_variable = {c: int(df[c].isna().sum()) for c in df.columns}
    by_variable_pct = {
        c: round(100.0 * v / len(df), 3) for c, v in by_variable.items() if v
    }

    value_cols = [c for c in df.columns if df[c].isna().any() and c not in (id_col, ts_col)]
    by_group = {}
    for gid, sub in df.groupby(id_col, observed=True):
        row = {c: round(100.0 * sub[c].isna().mean(), 3) for c in value_cols}
        if any(v > 0 for v in row.values()):
            by_group[_scalar(gid)] = row

    tmp = df.assign(_month=df[ts_col].dt.to_period("M").astype(str))
    by_month = tmp.groupby("_month")[value_cols].apply(lambda g: g.isna().mean() * 100.0)
    by_month_records = {
        str(month): {c: round(float(v), 4) for c, v in row.items() if v > 0}
        for month, row in by_month.iterrows()
    }

    return {
        "by_variable_count": by_variable,
        "by_variable_pct": by_variable_pct,
        f"by_{id_col}_pct": by_group,
        "by_month_pct": by_month_records,
    }


def check_impossible_values(df: pd.DataFrame) -> dict:
    findings = {}
    for col, (rule_name, predicate) in IMPOSSIBLE_RULES.items():
        if col not in df.columns:
            continue
        mask = predicate(df[col]) & df[col].notna()
        if mask.any():
            findings[rule_name] = {
                "column": col,
                "count": int(mask.sum()),
                "min": _maybe_float(df.loc[mask, col].min()),
                "max": _maybe_float(df.loc[mask, col].max()),
            }
    if "timestamp" in df.columns:
        invalid_ts = int(df["timestamp"].isna().sum())
        if invalid_ts:
            findings["invalid_timestamps"] = {"count": invalid_ts}
    return findings


def _scalar(value):
    try:
        return value.item()  # numpy scalar
    except AttributeError:
        return value


def _maybe_float(value):
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None
