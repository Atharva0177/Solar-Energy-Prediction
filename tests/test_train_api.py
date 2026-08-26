"""Train-page hardening tests (post-PRD enhancement, D-024).

Covers the /train router surface: dataset verification (path/upload,
structured 422s), job lifecycle against a fake subprocess, artifact
downloads, config snapshot — plus unit tests for marker-based success
semantics (torch teardown exits 9 after success on this box).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.train import _final_state


class TestFinalState:
    """== DONE marker decides success; returncode is diagnostic only."""

    def test_done_marker_beats_exit_code_9(self):
        text = "== STAGE verify start\n== STAGE verify done {}\n" \
               "== STAGE train start\n== DONE\n"
        st = _final_state(9, text)
        assert st["status"] == "done"
        assert st["error"] is None

    def test_failed_marker_wins_even_with_done(self):
        text = "== STAGE verify start\n== FAILED {\"reason\": \"missing columns\"}\n"
        st = _final_state(1, text)
        assert st["status"] == "failed"
        assert "missing columns" in st["error"]

    def test_crash_without_markers_reports_exit_code(self):
        st = _final_state(-1073741819, "some partial stdout")
        assert st["status"] == "failed"
        assert "exit code" in st["error"]


from fastapi.testclient import TestClient

from conftest import make_unisolar_folder


class _NoStore:
    """The /train router never touches the forecast store."""


@pytest.fixture(scope="module")
def client():
    from src.api.app import create_app

    return TestClient(create_app(_NoStore()))


class TestVerifyPath:
    def test_good_folder_returns_profile(self, client, tmp_path):
        raw = make_unisolar_folder(tmp_path)
        r = client.post("/api/v1/train/datasets/path", json={"path": str(raw)})
        assert r.status_code == 200
        body = r.json()
        assert body["mode"] == "path"
        assert body["profile"]["sites"] == 2
        assert body["profile"]["cadence_minutes"] == 15   # per-site diffs, not cross-site zeros
        # night slots report NaN by design -> half the 15-min grid unobserved
        assert body["profile"]["target_missing_pct"] == pytest.approx(50.0, abs=0.1)
        assert all(f["ok"] for f in body["files"])

    def test_missing_directory_422(self, client):
        r = client.post("/api/v1/train/datasets/path",
                        json={"path": "Z:/no/such/dir"})
        assert r.status_code == 422

    def test_bad_columns_structured_422(self, client, tmp_path):
        raw = make_unisolar_folder(tmp_path)
        p = raw / "Solar_Energy_Generation.csv"
        p.write_text("SiteKey,CampusKey,Timestamp\n1,7,2022-01-01\n",
                     encoding="utf-8")
        r = client.post("/api/v1/train/datasets/path", json={"path": str(raw)})
        assert r.status_code == 422
        detail = r.json()["detail"]
        assert "files" in detail
        gen = next(f for f in detail["files"]
                   if f["name"] == "Solar_Energy_Generation.csv")
        assert not gen["ok"] and "SolarGeneration" in gen["detail"]


class TestVerifyUpload:
    def _three_files(self):
        import io

        import numpy as np
        import pandas as pd

        from conftest import TS_FMT
        ts = pd.date_range("2022-01-01", periods=96 * 3, freq="15min")
        hour = ts.hour + ts.minute / 60
        elev = np.clip(np.sin((hour - 6) / 24 * 2 * np.pi) * 60, 0, None)
        power = np.where(elev > 0, np.maximum(elev * 0.2, 0.0), np.nan)
        gen = pd.DataFrame({
            "SiteKey": 1, "CampusKey": 7, "Timestamp": ts.strftime(TS_FMT),
            "SolarGeneration": power.round(4)})
        wx = pd.DataFrame({
            "CampusKey": 7, "Timestamp": ts.strftime(TS_FMT),
            "AirTemperature": 20.0, "RelativeHumidity": 60.0,
            "WindSpeed": 3.0, "WindDirection": 180.0})
        sites = pd.DataFrame(
            [{"SiteKey": 1, "CampusKey": 7, "kWp": 10.0,
              "lat": -36.1, "Lon": 146.8}])
        return [
            ("files", ("Solar_Energy_Generation.csv",
                       gen.to_csv(index=False), "text/csv")),
            ("files", ("Weather_Data_reordered_all.csv",
                       wx.to_csv(index=False), "text/csv")),
            ("files", ("Solar_Site_Details.csv",
                       sites.to_csv(index=False), "text/csv")),
        ]

    def test_upload_registers_dataset(self, client):
        r = client.post("/api/v1/train/datasets/upload",
                        files=self._three_files())
        assert r.status_code == 200
        body = r.json()
        assert body["mode"] == "upload"
        assert body["profile"]["generation_rows"] == 96 * 3

    def test_incomplete_set_lists_whats_missing(self, client):
        files = [f for f in self._three_files() if "Weather" not in f[1][0]]
        r = client.post("/api/v1/train/datasets/upload", files=files)
        assert r.status_code == 422
        names = {f["name"]: f for f in r.json()["detail"]["files"]}
        assert "not uploaded" in names["Weather_Data_reordered_all.csv"]["detail"]

    def test_unexpected_filename_rejected(self, client):
        files = self._three_files()
        files.append(("files", ("evil.csv", "x", "text/csv")))
        r = client.post("/api/v1/train/datasets/upload", files=files)
        assert r.status_code == 422
        assert "unexpected file" in r.json()["detail"]["message"]


import json
import threading
import time

import src.api.train as train_mod


def _verify_dataset(client, tmp_path):
    raw = make_unisolar_folder(tmp_path)
    r = client.post("/api/v1/train/datasets/path", json={"path": str(raw)})
    return r.json()["dataset_id"]


def _fake_result(job_dir) -> None:
    (job_dir / "result.json").write_text(json.dumps({
        "generated_at": "2026-08-26T00:00:00+00:00",
        "dataset": {}, "split": {}, "timing": {},
        "persistence": {}, "model": {"model_name": "xgboost"},
        "metrics_per_site": [], "test_all": {}, "val_all": {},
    }), encoding="utf-8")


class _Proc:
    """Instant fake child: writes markers + result.json, optional hang."""

    def __init__(self, cmd, release=None):
        self.cmd = cmd
        self._release = release
        job_dir = Path(cmd[cmd.index("--dataset-dir") + 1])
        (job_dir / "log.txt").write_text(
            "== STAGE verify start\n== STAGE verify done {}\n== DONE\n",
            encoding="utf-8")
        _fake_result(job_dir)
        self.returncode = 9  # torch-teardown style exit AFTER success

    def poll(self):
        return 0 if self._release is None or self._release.is_set() else None


def _fake_popen_factory(release=None, hang_first=None):
    def fake_popen(cmd, **kw):
        p = _Proc(cmd, release=release)
        if hang_first is not None and not hang_first["used"]:
            hang_first["used"] = True
            p.poll = lambda: None if not release.is_set() else 0
        return p
    return fake_popen


class TestJobLifecycle:
    def test_start_status_artifacts(self, client, tmp_path, monkeypatch):
        ds = _verify_dataset(client, tmp_path)
        seen_cmds = []
        monkeypatch.setattr(train_mod.subprocess, "Popen",
                            lambda cmd, **kw: seen_cmds.append(cmd)
                            or _Proc(cmd))
        r = client.post("/api/v1/train/jobs",
                        json={"dataset_id": ds, "model": "xgboost"})
        assert r.status_code == 200
        body = r.json()
        assert body["model"] == "xgboost" and body["status"] == "running"
        job_id = body["job_id"]
        assert "--fast-test" not in " ".join(seen_cmds[0])

        # watcher thread needs a beat to flip running -> done
        s = None
        for _ in range(50):
            s = client.get(f"/api/v1/train/jobs/{job_id}").json()
            if s["status"] != "running":
                break
            time.sleep(0.1)
        assert s is not None and s["status"] == "done"  # exit code 9 ≠ failed
        assert s["returncode"] == 9  # diagnostic field surfaces the teardown exit
        assert [st["name"] for st in s["stages"]] == ["verify"]
        assert s["result"]["model"]["model_name"] == "xgboost"
        assert s["error"] is None

        a = client.get(f"/api/v1/train/jobs/{job_id}/artifacts/result.json")
        assert a.status_code == 200
        assert a.json()["model"]["model_name"] == "xgboost"

    def test_unknown_dataset_404_before_model_check(self, client):
        # dataset existence is validated first (train.py order)
        r = client.post("/api/v1/train/jobs",
                        json={"dataset_id": "nope", "model": "not-a-model"})
        assert r.status_code == 404

    def test_bad_model_422(self, client, tmp_path):
        ds = _verify_dataset(client, tmp_path)
        r = client.post("/api/v1/train/jobs",
                        json={"dataset_id": ds, "model": "not-a-model"})
        assert r.status_code == 422

    def test_bad_artifact_name_422_and_unknown_job_404(self, client):
        r = client.get("/api/v1/train/jobs/x/artifacts/secrets.pem")
        assert r.status_code == 422
        r = client.get("/api/v1/train/jobs/no-such-job")
        assert r.status_code == 404
        r = client.get("/api/v1/train/jobs/no-such-job/artifacts/result.json")
        assert r.status_code == 404

    def test_one_job_at_a_time_409_then_slot_frees(self, client, tmp_path,
                                                   monkeypatch):
        ds = _verify_dataset(client, tmp_path)
        release = threading.Event()
        hang = {"used": False}
        monkeypatch.setattr(
            train_mod.subprocess, "Popen", _fake_popen_factory(release, hang))

        j1 = client.post("/api/v1/train/jobs",
                         json={"dataset_id": ds, "model": "lstm"}).json()["job_id"]
        r = client.post("/api/v1/train/jobs",
                        json={"dataset_id": ds, "model": "gru"})
        assert r.status_code == 409

        release.set()  # let the watcher reap job 1
        deadline = time.time() + 10
        while time.time() < deadline:
            ok = client.post("/api/v1/train/jobs",
                             json={"dataset_id": ds, "model": "gru",
                                   "fast_test": True})
            if ok.status_code == 200:
                break
            time.sleep(0.1)
        assert ok.status_code == 200, "slot never freed after job finished"


class TestConfigEndpoint:
    def test_config_shape(self, client):
        r = client.get("/api/v1/train/config")
        assert r.status_code == 200
        body = r.json()
        assert {"seed", "train_ratio"} <= set(body["training"])
        assert set(body["models"]) == {"xgboost", "lstm", "gru", "transformer"}
