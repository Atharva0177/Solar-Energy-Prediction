"""Data store backing the REST API (PRD §33-36).

``ParquetStore`` serves everything from artifacts already on disk — the
processed/features parquet datasets, per-phase metrics CSVs and the
conformal calibration — with per-site partition-filtered reads so no
endpoint loads the full 2.7M-row table. Tests inject an in-memory store
with the same interface instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

HISTORY_COLS = [
    "timestamp", "campus_id", "power", "is_daylight",
    "temperature", "humidity", "wind_speed",
]
# weather carried forward for future timestamps (no NWP feed in v1, D-019)
WEATHER_COLS = [
    "temperature", "apparent_temperature", "dew_point_temperature",
    "humidity", "wind_speed", "wind_direction",
]
RESOLUTIONS = {"15min": "15min", "1h": "1h", "1D": "1D"}
MAX_HISTORY_ROWS = 50_000


def _resample_history(df: pd.DataFrame, resolution: str) -> pd.DataFrame:
    """Mean-aggregate numeric history cols to ``resolution``; daylight = max."""
    df = df.set_index("timestamp")
    agg = {c: "mean" for c in df.columns
           if c not in ("timestamp", "is_daylight", "campus_id", "site_id")
           and pd.api.types.is_numeric_dtype(df[c])}
    if "is_daylight" in df.columns:
        agg["is_daylight"] = "max"
    out = df.resample(resolution).agg(agg)
    if "power" in out.columns:
        out = out.dropna(subset=["power"], how="all")
    return out.reset_index()

# model_id → (family, artifact path, served)
REGISTRY = {
    "persistence": ("naive", None, True),
    "xgboost": ("boosted_trees", "models/xgboost_site_all_h1_v1.json", True),
    "lstm": ("recurrent", "models/lstm_site_all_h1_v1.pt", True),
    "gru": ("recurrent", "models/gru_site_all_h1_v1.pt", True),
    "transformer": ("transformer", "models/transformer_site_all_h1_v1.pt", True),
}

_METRIC_SOURCES = {
    "xgboost": "artifacts/xgboost/metrics.csv", "lstm": "artifacts/lstm/metrics.csv",
    "gru": "artifacts/gru/metrics.csv",
    "transformer": "artifacts/transformer/metrics.csv",
    "persistence": "artifacts/baselines/baseline_metrics.csv",
}


class SiteNotFound(LookupError):
    pass


class ModelNotFound(LookupError):
    pass


class ModelNotServed(LookupError):
    pass


class ParquetStore:
    """Disk-backed store; one process-wide instance per app."""

    def __init__(self, root: Path = REPO_ROOT):
        self.root = Path(root)
        self.processed_dir = self.root / "data" / "processed" / "solar"
        self.features_dir = self.root / "data" / "processed" / "features"
        self._coords: pd.DataFrame | None = None
        self._site_details: pd.DataFrame | None = None
        self._seq_cache: dict = {}

    # ---- dataset / sites ---------------------------------------------------
    def dataset_info(self) -> dict:
        meta_path = self.root / "artifacts" / "features" / "feature_metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sites = self.site_details()
        return {
            "n_rows": int(meta["n_rows"]),
            "n_sites": int(len(sites)),
            "n_campuses": int(sites["campus_id"].nunique()),
            "cadence_minutes": int(meta["cadence_minutes"]),
            "timezone": meta["timezone"],
            "n_features_engineered": int(meta["n_engineered_columns"]),
            "target": meta.get("target", {}).get("column"),
            "target_imputed": meta.get("target", {}).get("imputed"),
        }

    def site_details(self) -> pd.DataFrame:
        if self._site_details is None:
            self._site_details = pd.read_parquet(
                self.root / "data" / "processed" / "site_details.parquet")
        return self._site_details

    def coords(self) -> pd.DataFrame:
        if self._coords is None:
            self._coords = (self.site_details()
                            .groupby("campus_id", observed=True)
                            .agg(latitude=("latitude", "median"),
                                 longitude=("longitude", "median"))
                            .reset_index())
        return self._coords

    def sites(self) -> list[dict]:
        sd = self.site_details().sort_values("site_id")
        return [{
            "site_id": int(r.site_id), "campus_id": int(r.campus_id),
            "latitude": None if pd.isna(r.latitude) else float(r.latitude),
            "longitude": None if pd.isna(r.longitude) else float(r.longitude),
        } for r in sd.itertuples()]

    # ---- history -------------------------------------------------------------
    def site_history(self, site_id: int, start=None, end=None,
                     resolution: str = "15min") -> pd.DataFrame:
        if resolution not in RESOLUTIONS:
            raise ValueError(f"resolution must be one of {sorted(RESOLUTIONS)}")
        df = pd.read_parquet(self.processed_dir,
                             filters=[("site_id", "=", site_id)])
        if df.empty:
            raise SiteNotFound(str(site_id))
        df = df[HISTORY_COLS].sort_values("timestamp")
        if start is not None:
            df = df[df["timestamp"] >= pd.Timestamp(start)]
        if end is not None:
            end_ts = pd.Timestamp(end)
            if end_ts == end_ts.normalize():  # date-only → whole day
                end_ts = end_ts + pd.Timedelta(hours=23, minutes=59,
                                               seconds=59)
            df = df[df["timestamp"] <= end_ts]
        if resolution != "15min":
            df = _resample_history(df, resolution)
        if len(df) > MAX_HISTORY_ROWS:
            df = df.iloc[-MAX_HISTORY_ROWS:]
        return df

    # ---- forecast inputs -------------------------------------------------------
    def site_feature_tail(self, site_id: int, hours: int = 48) -> pd.DataFrame:
        df = pd.read_parquet(self.features_dir,
                             filters=[("site_id", "=", site_id)])
        if df.empty:
            raise SiteNotFound(str(site_id))
        df = df.sort_values("timestamp")
        return df.tail(hours * 4).reset_index(drop=True)

    def conformal_radii(self, model_id: str = "xgboost", level: str = "0.9") -> dict:
        path = self.root / "artifacts" / "uncertainty" / "run_metadata.json"
        meta = json.loads(path.read_text(encoding="utf-8"))
        cal = meta["calibration"][f"{model_id}|mondrian|{level}"]
        return {"global": cal["global_radius"], "regimes": cal["regime_radii"]}

    def load_booster(self):
        """The served XGBoost model, with its stored feature layout."""
        from src.models.xgboost_model import load_xgboost_model

        return load_xgboost_model(
            self.root / "models" / "xgboost_site_all_h1_v1.json",
            self.root / "artifacts" / "xgboost" / "run_metadata.json")[0]

    def load_sequence(self, model_id: str) -> dict:
        """Served deep model (lstm|gru|transformer) + serving scalers.

        Checkpoints hold weights only; the standardization stats live in
        ``artifacts/{arch}/serving_scalers.json`` (Phase 14, D-022) —
        exported by ``scripts/export_sequence_scalers.py`` and validated
        there to reproduce the stored test predictions bit-exactly.
        Loaded once per process, on CPU (serving needs no GPU).
        """
        if model_id not in self._seq_cache:
            import torch

            from src.models.sequence_model import (
                RecurrentForecaster,
                TransformerForecaster,
            )

            meta = json.loads((self.root / "artifacts" / model_id /
                               "run_metadata.json").read_text(encoding="utf-8"))
            sc = json.loads((self.root / "artifacts" / model_id /
                             "serving_scalers.json").read_text(encoding="utf-8"))
            params = meta["config"]["models"][model_id]["params"]
            lookback = int(meta["lookback_steps"])
            n_ch = len(sc["channels"])
            if meta["architecture"] == "transformer":
                model = TransformerForecaster(
                    input_size=n_ch,
                    d_model=int(params["d_model"]),
                    nhead=int(params["nhead"]),
                    num_layers=int(params["num_layers"]),
                    dim_feedforward=int(params["dim_feedforward"]),
                    dropout=float(params["dropout"]), max_len=lookback + 1)
            else:
                model = RecurrentForecaster(
                    meta["architecture"], input_size=n_ch,
                    hidden_size=int(params["hidden_size"]),
                    num_layers=int(params["num_layers"]),
                    dropout=float(params["dropout"]))
            ckpt = torch.load(
                self.root / "models" / f"{model_id}_site_all_h1_v1.pt",
                map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            self._seq_cache[model_id] = {
                "model": model, "architecture": meta["architecture"],
                "lookback": lookback,
                "y_mean": float(sc["y_mean"]), "y_std": float(sc["y_std"]),
                "channel_mean": sc["channel_mean"],
                "channel_std": sc["channel_std"],
                "channels": sc["channels"],
            }
        return self._seq_cache[model_id]

    # ---- models ------------------------------------------------------------------
    def model_registry(self) -> list[dict]:
        out = []
        for mid, (family, artifact, served) in REGISTRY.items():
            out.append({
                "model_id": mid, "family": family,
                "artifact": artifact,
                "served": served and (artifact is None
                                      or (self.root / artifact).exists()),
            })
        return out

    def model_metrics(self, model_id: str) -> dict:
        if model_id not in _METRIC_SOURCES:
            raise ModelNotFound(model_id)
        path = self.root / _METRIC_SOURCES[model_id]
        if not path.exists():
            raise ModelNotFound(model_id)
        m = pd.read_csv(path)
        if "baseline" in m.columns:  # baselines CSV layout
            row = m[(m.baseline == "persistence_prev_day")
                    & (m.split == "test") & (m.scope == "ALL")]
        else:
            row = m[(m.split == "test") & (m.scope == "ALL")]
        if row.empty:
            raise ModelNotFound(model_id)
        r = row.iloc[0]
        return {
            "model_id": model_id, "split": "test", "scope": "ALL",
            "mae": float(r.mae), "rmse": float(r.rmse), "r2": float(r.r2),
            "nrmse": None if pd.isna(r.nrmse) else float(r.nrmse),
            "daylight_mae": None if pd.isna(r.daylight_mae) else float(r.daylight_mae),
            "daylight_nrmse": (None if pd.isna(r.daylight_nrmse)
                               else float(r.daylight_nrmse)),
            "n_eval": int(r.n_eval),
        }


class MemStore(ParquetStore):
    """In-memory store for tests: same interface, tiny synthetic data."""

    def __init__(self, features: pd.DataFrame, processed: pd.DataFrame,
                 sites: pd.DataFrame, metrics: dict | None = None,
                 radii: dict | None = None, booster=None, sequence=None):
        super().__init__(root=Path("."))
        self._features = features
        self._processed = processed
        self._sites = sites
        self._metrics = metrics or {}
        self._radii = radii or {"global": 1.0, "regimes": {}}
        self._booster = booster
        self._sequence = sequence

    def dataset_info(self) -> dict:
        return {"n_rows": int(len(self._processed)), "n_sites": 2,
                "n_campuses": 1, "cadence_minutes": 15,
                "timezone": "Australia/Melbourne", "date_range": None,
                "target_missing_share": None, "n_features_engineered": None}

    def site_details(self) -> pd.DataFrame:
        return self._sites

    def site_history(self, site_id: int, start=None, end=None,
                     resolution: str = "15min") -> pd.DataFrame:
        df = self._processed[self._processed.site_id == site_id]
        if df.empty:
            raise SiteNotFound(str(site_id))
        df = df.sort_values("timestamp")
        if start is not None:
            df = df[df.timestamp >= pd.Timestamp(start)]
        if end is not None:
            df = df[df.timestamp <= pd.Timestamp(end)]
        if resolution != "15min":
            df = _resample_history(df, resolution)
        return df

    def site_feature_tail(self, site_id: int, hours: int = 48) -> pd.DataFrame:
        df = self._features[self._features.site_id == site_id]
        if df.empty:
            raise SiteNotFound(str(site_id))
        return df.sort_values("timestamp").tail(hours * 4).reset_index(drop=True)

    def conformal_radii(self, model_id="xgboost", level="0.9") -> dict:
        return self._radii

    def load_booster(self):
        return self._booster

    def load_sequence(self, model_id: str) -> dict:
        if self._sequence is None:
            raise ModelNotFound(model_id)
        return self._sequence

    def model_metrics(self, model_id: str) -> dict:
        if model_id not in self._metrics:
            raise ModelNotFound(model_id)
        return self._metrics[model_id]
