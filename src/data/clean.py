"""Cleaning operations with full operation logging (PRD §9).

Rule: no blind row deletion. Every mutation goes through :class:`CleanLog`,
and destructive steps are limited to exact-duplicate removal. Impossible
values are nulled (not dropped) so missing-value handling sees them
explicitly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd


class CleanLog:
    """Accumulates one record per cleaning operation."""

    def __init__(self) -> None:
        self.operations: list[dict[str, Any]] = []

    def add(self, dataset: str, op: str, rows_affected: int, detail: str = "") -> None:
        self.operations.append(
            {
                "dataset": dataset,
                "operation": op,
                "rows_affected": int(rows_affected),
                "detail": detail,
            }
        )

    def to_dict(self) -> dict:
        return {
            "logged_at_utc": datetime.now(timezone.utc).isoformat(),
            "total_operations": len(self.operations),
            "operations": self.operations,
        }


def parse_timestamps(df: pd.DataFrame, log: CleanLog, name: str) -> pd.DataFrame:
    """Coerce ``timestamp`` to datetime64; unparsable -> NaT (counted, kept).

    Datasets without a timestamp column pass through unchanged.
    """
    if "timestamp" not in df.columns:
        return df
    before_na = df["timestamp"].isna().sum()
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", format="mixed")
    new_na = df["timestamp"].isna().sum()
    if new_na > before_na:
        log.add(name, "coerce_unparsable_timestamps_to_nat", int(new_na - before_na))
    return df


def coerce_numeric(df: pd.DataFrame, cols: list[str], log: CleanLog, name: str) -> pd.DataFrame:
    """Ensure value columns are numeric; coercion failures become NaN."""
    df = df.copy()
    for col in cols:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            coerced = pd.to_numeric(df[col], errors="coerce")
            n_new_na = int((coerced.isna() & df[col].notna()).sum())
            log.add(name, f"coerce_numeric:{col}", n_new_na)
            df[col] = coerced
    return df


def drop_exact_duplicates(df: pd.DataFrame, log: CleanLog, name: str) -> pd.DataFrame:
    """Only sanctioned row deletion: fully identical rows."""
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = n_before - len(df)
    if removed:
        log.add(name, "drop_exact_duplicate_rows", int(removed))
    return df


def null_impossible_values(
    df: pd.DataFrame,
    rules: dict,
    log: CleanLog,
    name: str,
) -> pd.DataFrame:
    """Null out impossible values per validation rules; keep rows.

    The original values are recorded in the log summary counts only;
    rows survive so downstream imputation/masking decides their fate.
    """
    from .validate import IMPOSSIBLE_RULES

    rules = rules or IMPOSSIBLE_RULES
    df = df.copy()
    for col, (rule_name, predicate) in rules.items():
        if col not in df.columns:
            continue
        mask = predicate(df[col]) & df[col].notna()
        if mask.any():
            df.loc[mask, col] = pd.NA if df[col].dtype == object else float("nan")
            log.add(
                name,
                f"null_impossible:{rule_name}",
                int(mask.sum()),
                f"column={col}",
            )
    return df
