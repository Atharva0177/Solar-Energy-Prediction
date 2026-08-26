"""FastAPI app factory (PRD §33-36).

Routes under ``/api/v1``:

* ``GET  /health``                     — liveness + store sanity
* ``GET  /dataset``                    — dataset card (PRD §33)
* ``GET  /sites``                      — site list
* ``GET  /sites/{site_id}/history``    — observed power, start/end/resolution
* ``GET  /models``                     — registry (served vs registered-only)
* ``GET  /models/{model_id}/metrics``  — test-split MAE/RMSE/R²/nRMSE/daylight
* ``POST /forecast``                   — recursive multi-step, one site
* ``POST /forecast/batch``             — same, multiple sites
* ``/train/*``                          — Train-page job API + config
  (post-PRD enhancement, D-024/D-025); plus read-only ``/static`` mount of
  ``artifacts/`` for the frontend image galleries.

Served models: ``xgboost`` (recursive, with conformal bounds),
``persistence``, and — since Phase 14 — ``lstm`` / ``gru`` / ``transformer``
on the same recursive path over their Phase 7 windows (no conformal radii:
Phase 11 calibrated xgboost + persistence only, so bounds are omitted).

Create with ``create_app(store)``; tests inject a ``MemStore``, production
uses ``ParquetStore`` (see ``scripts/run_api.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import json

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .forecast import (
    recursive_forecast_persistence,
    recursive_forecast_sequence,
    recursive_forecast_xgboost,
)
from .store import (
    MemStore,
    ModelNotFound,
    ModelNotServed,
    ParquetStore,
    SiteNotFound,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

API_VERSION = "1.0.0"
MAX_HORIZON_STEPS = 96  # 24 h at 15-min cadence
MAX_BATCH = 10
CONFIDENCE_LEVEL = 0.9

REG, XGB = "src.models.xgboost_model", None  # lazy-loaded booster cache
_xgb_cache: dict = {}


class ForecastRequest(BaseModel):
    site_id: int
    forecast_horizon: int = Field(default=MAX_HORIZON_STEPS, ge=1,
                                  le=MAX_HORIZON_STEPS)
    model: str = "xgboost"


class BatchForecastRequest(BaseModel):
    requests: list[ForecastRequest] = Field(min_length=1, max_length=MAX_BATCH)


def _get_booster(store):
    """The served booster, loaded once per process."""
    if "reg" not in _xgb_cache:
        _xgb_cache["reg"] = store.load_booster()
    return _xgb_cache["reg"]


def _resolve_model(store, model_id: str) -> str:
    entry = next((m for m in store.model_registry()
                  if m["model_id"] == model_id), None)
    if entry is None:
        raise HTTPException(404, f"unknown model {model_id!r}")
    if not entry["served"]:
        raise HTTPException(409, f"model {model_id!r} is registered "
                                 "but not served by this API")
    return model_id


DEEP_MODELS = ("lstm", "gru", "transformer")


def _run_one(store, req: ForecastRequest) -> dict:
    _resolve_model(store, req.model)
    try:
        tail = store.site_feature_tail(req.site_id, hours=48)
    except SiteNotFound:
        raise HTTPException(404, f"unknown site_id {req.site_id}")
    if req.model == "persistence":
        preds = recursive_forecast_persistence(tail, req.forecast_horizon)
    elif req.model in DEEP_MODELS:
        pkg = store.load_sequence(req.model)
        if len(tail) < pkg["lookback"] + 1:
            raise HTTPException(
                422, f"{req.model} needs {pkg['lookback'] + 1} history "
                     f"slots for site {req.site_id}, found {len(tail)}")
        preds = recursive_forecast_sequence(pkg, tail, req.forecast_horizon,
                                            store.coords())
    else:
        radii = store.conformal_radii("xgboost", "0.9")
        preds = recursive_forecast_xgboost(
            _get_booster(store), tail, req.forecast_horizon,
            store.coords(), radii=radii, confidence_level=CONFIDENCE_LEVEL)
    return {"site_id": req.site_id, "model": req.model,
            "forecast_horizon": req.forecast_horizon, "predictions": preds}


def create_app(store: ParquetStore | MemStore) -> FastAPI:
    app = FastAPI(title="UNISOLAR forecasting API",
                  version=API_VERSION,
                  description="Solar power forecasting (PRD §33-36)")

    # Train page (post-PRD enhancement, D-024/D-025) + read-only artifact
    # images for the frontend galleries (EDA / SHAP PNGs).
    from fastapi.staticfiles import StaticFiles

    from .train import router as train_router

    app.include_router(train_router, prefix="/api/v1")
    artifacts_dir = REPO_ROOT / "artifacts"
    app.mount("/static", StaticFiles(directory=str(artifacts_dir)),
              name="artifacts")

    @app.get("/api/v1/health")
    def health():
        return {"status": "ok", "api_version": API_VERSION,
                "n_sites": len(store.sites())}

    @app.get("/api/v1/dataset")
    def dataset():
        return store.dataset_info()

    @app.get("/api/v1/sites")
    def sites():
        return {"sites": store.sites()}

    @app.get("/api/v1/sites/{site_id}/history")
    def site_history(
        site_id: int,
        start: Optional[str] = Query(None, description="ISO timestamp"),
        end: Optional[str] = Query(None, description="ISO timestamp"),
        resolution: str = Query("15min", description="15min | 1h | 1D"),
    ):
        from .store import RESOLUTIONS
        if resolution not in RESOLUTIONS:
            raise HTTPException(422, f"resolution must be one of "
                                     f"{sorted(RESOLUTIONS)}")
        try:
            df = store.site_history(site_id, start, end, resolution)
        except SiteNotFound:
            raise HTTPException(404, f"unknown site_id {site_id}")
        except ValueError as e:
            raise HTTPException(422, str(e))
        df = df.copy()
        # to_json handles numpy scalars/NaN → null; .to_dict would leak
        # np.float64 into JSON encoding and 500
        rows = json.loads(df.to_json(orient="records", date_format="iso"))
        return {"site_id": site_id, "resolution": resolution,
                "n_rows": int(len(df)), "rows": rows}

    @app.get("/api/v1/models")
    def models():
        return {"models": store.model_registry()}

    @app.get("/api/v1/models/{model_id}/metrics")
    def model_metrics(model_id: str):
        try:
            return store.model_metrics(model_id)
        except ModelNotFound:
            raise HTTPException(404, f"no metrics for model {model_id!r}")

    @app.post("/api/v1/forecast")
    def forecast(req: ForecastRequest):
        return _run_one(store, req)

    @app.post("/api/v1/forecast/batch")
    def forecast_batch(batch: BatchForecastRequest):
        return {"results": [_run_one(store, r) for r in batch.requests]}

    return app
