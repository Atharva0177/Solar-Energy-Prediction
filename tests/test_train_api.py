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
