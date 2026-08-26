"""XGBoost forecaster (PRD §22: primary traditional ML model).

Single-step framing (horizon = one 15-min slot): predict ``power`` at t from
features known strictly before t (lags/rolling by D-012 construction), plus
calendar/weather/solar/site-identity covariates at t.

Design points:

* Feature selection is dynamic — families absent from the frame are skipped
  (mirrors PRD §19's adapt-to-available principle).
* Raw ``wind_direction`` excluded (replaced by circular sin/cos, D-012);
  string ``season`` excluded (day-of-year cyclical covers it).
* NaN features are left for XGBoost's native sparsity handling; only rows
  with an observed target enter training (D-008: never impute power).
* CPU ``hist`` tree method — preprocessing/XGBoost stay GPU-free by design
  (PRD Rule under §49).
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

TARGET_COL = "power"

TEMPORAL_FEATURES = [
    "hour", "minute", "day", "day_of_week", "day_of_year", "week_of_year",
    "month", "quarter", "is_weekend",
    "sin_hour", "cos_hour", "sin_day_of_year", "cos_day_of_year",
]
LAG_FEATURES = [f"power_lag_{s}" for s in (1, 2, 4, 8, 24, 48, 96)]
ROLLING_FEATURES = [
    f"power_rolling_{stat}_{w}s"
    for w in (3600, 21600, 86400)
    for stat in ("mean", "std", "min", "max")
]
WEATHER_FEATURES = [
    "temperature", "apparent_temperature", "dew_point_temperature",
    "humidity", "wind_speed", "wind_dir_sin", "wind_dir_cos",
]
SOLAR_FEATURES = ["solar_elevation_deg", "azimuth_deg", "zenith_deg", "day_length_hours"]

CATEGORICAL_FEATURES = ["site_id"]

# never fed to the model raw: target, strings/bools redundant with numeric
# encodings, partition helpers, identifiers beyond site_id
EXCLUDED_COLUMNS = {
    TARGET_COL, "season", "is_daylight", "wind_direction",
    "timestamp", "year", "month", "campus_id",
}


def select_feature_columns(df: pd.DataFrame) -> dict:
    """Present-only feature columns, grouped by XGBoost dtype treatment."""
    candidates = (
        TEMPORAL_FEATURES + LAG_FEATURES + ROLLING_FEATURES
        + WEATHER_FEATURES + SOLAR_FEATURES
    )
    return {
        "categorical": [c for c in CATEGORICAL_FEATURES if c in df.columns],
        "numeric": [
            c for c in candidates
            if c in df.columns and c not in EXCLUDED_COLUMNS
        ],
    }


def prepare_matrix(df: pd.DataFrame, cols: Optional[dict] = None) -> pd.DataFrame:
    """Model input matrix: categorical dtype for cats, numeric floats else."""
    cols = cols or select_feature_columns(df)
    order = cols["categorical"] + cols["numeric"]
    X = df[order].copy()
    for c in cols["categorical"]:
        X[c] = X[c].astype("category")
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype("int8")
        elif X[c].dtype == object:
            # defensive: object columns (e.g. None-filled) must become floats
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    return X


def train_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    params: dict,
    seed: int = 42,
    early_stopping_rounds: Optional[int] = 50,
    target: str = TARGET_COL,
) -> tuple:
    """Fit on observed targets, early-stop on val; returns (model, info)."""
    cols = select_feature_columns(pd.concat([train_df, val_df], ignore_index=True))

    def xy(frame):
        X = prepare_matrix(frame, cols)
        y = frame[target].astype(float)
        ok = y.notna()
        return X.loc[ok], y.loc[ok]

    X_tr, y_tr = xy(train_df)
    X_va, y_va = xy(val_df)

    model = XGBRegressor(
        random_state=seed,
        tree_method="hist",
        enable_categorical=True,
        n_jobs=-1,
        **({"early_stopping_rounds": early_stopping_rounds} if early_stopping_rounds else {}),
        **params,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    model.feature_cols_ = cols  # consumed by predict_frame

    best_it = getattr(model, "best_iteration", None)
    info = {
        "best_iteration": int(best_it) + 1 if best_it is not None else params.get("n_estimators"),
        "n_train_rows": int(len(X_tr)),
        "n_val_rows": int(len(X_va)),
        "feature_columns": cols,
    }
    return model, info


def predict_frame(model, df: pd.DataFrame) -> np.ndarray:
    """Predictions for ``df`` using the feature layout the model was fit on."""
    cols = getattr(model, "feature_cols_", None) or select_feature_columns(df)
    return model.predict(prepare_matrix(df, cols))


def extract_importance(model) -> pd.DataFrame:
    """gain/weight/cover per feature (0-filled for unused features)."""
    booster = model.get_booster()
    names = model.feature_cols_["categorical"] + model.feature_cols_["numeric"] \
        if hasattr(model, "feature_cols_") else booster.feature_names
    rows = []
    for imp_type in ("gain", "weight", "cover"):
        scores = booster.get_score(importance_type=imp_type)
        rows.append(pd.Series({n: scores.get(n, 0.0) for n in names}, name=imp_type))
    imp = pd.concat(rows, axis=1).reset_index().rename(columns={"index": "feature"})
    return imp.sort_values("gain", ascending=False, ignore_index=True)


def dataset_fingerprint(path) -> str:
    """Stable short hash of a dataset directory: rel name + size + mtime_ns."""
    p = __import__("pathlib").Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    h = hashlib.sha256()
    for f in sorted(p.rglob("*")):
        if f.is_file():
            st = f.stat()
            h.update(f"{f.relative_to(p)}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()[:16]


def load_xgboost_model(model_path, metadata_path: Optional["object"] = None) -> tuple:
    """Reload a saved booster as an XGBRegressor with its feature layout.

    ``feature_cols_`` is restored from the run metadata because the native
    JSON save does not carry it; without it ``prepare_matrix`` could reorder
    columns relative to training. Used by the explainability (Phase 10) and
    uncertainty (Phase 11) consumers of the frozen Phase 6 model.
    """
    import json
    from pathlib import Path

    import xgboost

    reg = xgboost.XGBRegressor()
    reg.load_model(str(model_path))
    meta = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    reg.feature_cols_ = meta["feature_columns"]
    return reg, meta
