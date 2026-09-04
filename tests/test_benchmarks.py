"""Performance benchmarks (PRD Phase 16 CI performance workflow).

Self-contained pytest-benchmark cases on synthetic data — no parquet or
model artifacts required, so CI runners measure real inference paths:

* xgboost single-step predict (matrix prep + booster);
* API recursive forecast over HTTP via TestClient (xgboost + lstm);
* API history endpoint.

Selected in CI via ``pytest -k benchmark``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.store import MemStore
from src.models.xgboost_model import prepare_matrix
from test_api import synth, tiny_booster, tiny_sequence


@pytest.fixture(scope="module")
def bench_env():
    features, processed, sites = synth()
    store = MemStore(features, processed, sites,
                     radii={"global": 2.9, "regimes": {"day_lag": 2.9}},
                     booster=tiny_booster(features),
                     sequence=tiny_sequence())
    return features, TestClient(create_app(store))


def test_benchmark_xgboost_single_step(benchmark, bench_env):
    """One-step predict: matrix build + booster inference on 768 rows."""
    features, _ = bench_env
    booster = tiny_booster(features)

    def predict():
        X = prepare_matrix(features, booster.feature_cols_)
        return booster.predict(X)

    preds = benchmark(predict)
    assert len(preds) == len(features)


def test_benchmark_api_forecast_xgboost(benchmark, bench_env):
    """Recursive 8-step xgboost forecast through the full HTTP stack."""
    _, client = bench_env
    req = {"site_id": 1, "forecast_horizon": 8, "model": "xgboost"}

    def call():
        r = client.post("/api/v1/forecast", json=req)
        assert r.status_code == 200
        return r

    body = benchmark(call).json()
    assert len(body["predictions"]) == 8


def test_benchmark_api_forecast_lstm(benchmark, bench_env):
    """Recursive 8-step LSTM forecast (deep-model serving path)."""
    _, client = bench_env
    req = {"site_id": 1, "forecast_horizon": 8, "model": "lstm"}

    def call():
        r = client.post("/api/v1/forecast", json=req)
        assert r.status_code == 200
        return r

    body = benchmark(call).json()
    assert len(body["predictions"]) == 8


def test_benchmark_api_history(benchmark, bench_env):
    """History endpoint: filter + serialize 4 days of 15-min rows."""

    _, client = bench_env

    def call():
        r = client.get("/api/v1/sites/1/history")
        assert r.status_code == 200
        return r

    benchmark(call)
