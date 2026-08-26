# Train Hardening + Rich Visuals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the built-but-never-run Train feature (tests + live run + docs) and add six new charts fed by four new offline export bundles.

**Architecture:** Two workstreams. (1) Train page: fix two latent bugs (exit-code-9 success semantics; persistence baseline fitted on train-only → NaN-starved test metrics), then test the API and pipeline on a synthetic UNISOLAR folder, run one live job, backfill docs. (2) Visuals: extend `scripts/export_frontend_data.py` with four bundle builders (pure functions, unit-tested), then mount charts on Dashboard, Quality, Comparison pages reading static JSONs (D-020 pattern).

**Tech Stack:** Python 3.13 conda env `solar` (pandas, pyarrow, xgboost, torch, FastAPI TestClient), pytest, React 19 + TS + Vite + Tailwind 4 + Recharts, Playwright for screenshots.

**Spec:** `docs/superpowers/specs/2026-08-26-train-hardening-and-rich-visuals-design.md`

## Global Constraints

- Python commands run via `conda run -n solar …` (repo convention). Multiline `python -c` breaks under conda run — use temp script files ([[env-conda-run-newline]]).
- NEVER modify served v1 models (`models/`), phase artifacts under `artifacts/` (except adding `artifacts/train_live_run/`), or recorded RESULTS.md numbers. Training jobs are display-only (D-024).
- Timezone = Australia/Melbourne (D-007); splits via `chronological_split` 70/15/15 (D-011); nRMSE denominator = train observed range.
- Frontend: `is_daylight` arrives as JSON boolean — compare `=== true`, never `=== 1` (D-021).
- New bundles total < ~1 MB raw JSON.
- All new tests join the existing suite (`conda run -n solar python -m pytest -q`); suite currently 147 green.
- Repo becomes git in Task 0; every task ends with a commit on `main`.
- Windows box; use the Bash tool syntax shown (Git Bash) for shell steps.

---

### Task 0: Initialize git repository

**Files:**
- Modify: `.gitignore`

**Interfaces:**
- Produces: git repo on `main`; all later tasks commit against it.

- [ ] **Step 1: Extend .gitignore** — append these lines (large/regenerable paths must not enter VCS):

```gitignore

# Large processed data & raw dataset (regenerable via scripts)
data/processed/
data/train_jobs/
unisolar/

# Trained model artifacts at repo root (regenerable via phase scripts)
models/
```

Note: `frontend/src/data/*.json` stay tracked — the frontend build imports them, so a fresh clone builds without a Python env.

- [ ] **Step 2: Init and commit baseline**

```bash
cd /e/Solar_gemini && git init -b main \
  && git add -A \
  && git commit -m "chore: baseline before train-hardening + visuals work"
```

Expected: commit succeeds; `git status` clean afterwards. If `git add` warns about huge files, check `git ls-files -s | awk '{print $4}' | xargs -I{} du -k {} 2>/dev/null | sort -rn | head` — nothing over ~5 MB should be staged.

---

### Task 1: Restore missing export bundles → frontend build green

The prior session added imports of `@/data/site_monthly.json` and `@/data/quality_extra.json` (dashboard.tsx:33, quality.tsx:19) but never re-ran the export script — those files don't exist and **the frontend build is currently broken**.

**Files:**
- Create (generated): `frontend/src/data/site_monthly.json`, `frontend/src/data/quality_extra.json`

**Interfaces:**
- Consumes: existing `scripts/export_frontend_data.py` as-is.
- Produces: green `npm run build`; baseline for all frontend tasks.

- [ ] **Step 1: Run the export script**

```bash
cd /e/Solar_gemini && conda run -n solar python scripts/export_frontend_data.py
```

Expected output includes `wrote frontend\src\data\site_monthly.json (...)` and `wrote frontend\src\data\quality_extra.json (...)`. All seven bundle files now exist in `frontend/src/data/`.

- [ ] **Step 2: Verify frontend builds**

```bash
cd /e/Solar_gemini/frontend && npm run build
```

Expected: build completes with no errors (bundle size grows slightly from new JSONs).

- [ ] **Step 3: Commit**

```bash
cd /e/Solar_gemini && git add frontend/src/data .gitignore \
  && git commit -m "fix: regenerate missing site_monthly/quality_extra bundles; restore frontend build"
```

---

### Task 2: Fix exit-code-9 success semantics in the job watcher

Torch scripts on this machine exit code 9 AFTER a successful run (teardown crash — Phase 7 gotcha). `src/api/train.py:_watch` treats any nonzero returncode as failure, so deep-model jobs would finish fine yet display "failed". Success must be decided by log markers only.

**Files:**
- Modify: `src/api/train.py` (~line 230 `_watch`)
- Test: `tests/test_train_api.py` (create)

**Interfaces:**
- Produces: `_final_state(returncode: int | None, text: str) -> dict` module-level function returning `{"status": "done"|"failed", "error": str|None}`. Task 5's lifecycle tests rely on watcher behavior via this function.

- [ ] **Step 1: Write failing tests**

Create `tests/test_train_api.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /e/Solar_gemini && conda run -n solar python -m pytest tests/test_train_api.py -v
```

Expected: FAIL/ERROR — `ImportError: cannot import name '_final_state'`.

- [ ] **Step 3: Implement**

In `src/api/train.py`, add above `_watch`:

```python
def _final_state(returncode: int | None, text: str) -> dict:
    """Success = '== DONE' marker present and no '== FAILED' marker.

    Returncode is diagnostic only: torch scripts on this machine can exit
    code 9 AFTER a successful run (teardown crash — Phase 7 gotcha), so a
    nonzero exit alone must not flip a finished job to failed.
    """
    failed = FAILED_RE.search(text, re.MULTILINE)
    ok = "== DONE" in text and failed is None
    return {
        "status": "done" if ok else "failed",
        "error": None if ok else (
            failed.group(1)[:300] if failed else f"exit code {returncode}"),
    }
```

Then replace the tail of `_watch` (the block from `ok = proc.returncode == 0 ...` through the `job.update({...})` call) with:

```python
    state = _final_state(proc.returncode, text)
    job.update({
        **state,
        "stages": stages,
        "stage": "",
        "finished_at": time.time(),
        "returncode": proc.returncode,
    })
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n solar python -m pytest tests/test_train_api.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: Run full suite (no regressions)**

```bash
conda run -n solar python -m pytest -q
```

Expected: 150 passed (147 + 3).

- [ ] **Step 6: Commit**

```bash
git add src/api/train.py tests/test_train_api.py \
  && git commit -m "fix(train): marker-based job success — torch teardown exits 9 after success"
```

---

### Task 3: Synthetic UNISOLAR fixture + pipeline e2e test + persistence fit-scope fix

Two things in one task because the e2e test EXPOSES the bug it also pins:

**Bug:** `scripts/train_from_folder.py::do_baseline` fits `PersistenceBaseline()` on the TRAIN split only (`base = PersistenceBaseline().fit(train)`). The persistence lookup table therefore contains no val/test rows, so any prediction needing t−24h from inside val/test history → NaN. Val rows survive (their t−24h lands in late-train); TEST rows beyond the first day are NaN-starved — test MAE silently computed on ≤1 day of data. D-011 #3 (Phase 11 gotcha): persistence must fit the FULL table; t−24h lookups are strictly causal so this leaks nothing.

**Files:**
- Create: `tests/conftest.py` (shared synthetic-folder builder)
- Create: `tests/test_train_pipeline.py`
- Modify: `scripts/train_from_folder.py::do_baseline` (~line 275)

**Interfaces:**
- Produces: `make_unisolar_folder(root: Path, days: int = 10, sites=(1, 2)) -> Path` in `tests/conftest.py` — writes `root/raw/` with the three schema-valid CSVs, returns raw dir. Task 4/5 reuse it.
- Consumes: `src.data.schema_map` raw column names `AirTemperature, ApparentTemperature, DewPointTemperature, RelativeHumidity, WindSpeed, WindDirection` (verified in schema_map.py WEATHER_COLUMNS).

- [ ] **Step 1: Write the fixture builder** — create `tests/conftest.py`:

```python
"""Shared synthetic-data builders for train-page hardening tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

TS_FMT = "%Y-%m-%d %H:%M:%S"


def make_unisolar_folder(root, days=10, sites=(1, 2), campus=7, seed=0):
    """Write the three UNISOLAR CSVs under root/raw with a learnable
    daylight signal. Returns the raw dir. Schema matches
    src/data/schema_map.py exactly."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2022-01-01", periods=days * 96, freq="15min")
    hour = ts.hour + ts.minute / 60
    elev = np.clip(np.sin((hour - 6) / 24 * 2 * np.pi) * 60, 0, None)

    wx = pd.DataFrame({
        "CampusKey": campus,
        "Timestamp": ts.strftime(TS_FMT),
        "AirTemperature": 18 + 6 * np.sin((hour - 9) / 24 * 2 * np.pi)
                          + rng.normal(0, 0.5, len(ts)),
        "ApparentTemperature": 17 + 6 * np.sin((hour - 9) / 24 * 2 * np.pi),
        "DewPointTemperature": 12.0,
        "RelativeHumidity": 60 - 10 * np.sin((hour - 9) / 24 * 2 * np.pi),
        "WindSpeed": 3 + rng.normal(0, 0.5, len(ts)),
        "WindDirection": 180 + rng.normal(0, 30, len(ts)),
    })

    gen_frames, site_rows = [], []
    for sid in sites:
        power = elev * (0.15 + 0.05 * sid) + rng.normal(0, 0.05, len(ts))
        gen_frames.append(pd.DataFrame({
            "SiteKey": sid, "CampusKey": campus,
            "Timestamp": ts.strftime(TS_FMT),
            "SolarGeneration": power.round(4),
        }))
        site_rows.append({"SiteKey": sid, "CampusKey": campus,
                          "kWp": 10 * sid, "lat": -36.1, "Lon": 146.8})

    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    pd.concat(gen_frames, ignore_index=True).to_csv(
        raw / "Solar_Energy_Generation.csv", index=False)
    wx.to_csv(raw / "Weather_Data_reordered_all.csv", index=False)
    pd.DataFrame(site_rows).to_csv(raw / "Solar_Site_Details.csv", index=False)
    return raw
```

- [ ] **Step 2: Write the pipeline e2e test** — create `tests/test_train_pipeline.py`:

```python
"""End-to-end test of scripts/train_from_folder.py --fast-test on a
synthetic UNISOLAR folder. Pins:

* the five stages run and result.json carries the documented key set;
* persistence is fit on the FULL table (D-011 #3) — test-split coverage
  is complete, not NaN-starved beyond the first day;
* artifacts land inside the job dir only; repo models/ and key phase
  artifacts stay byte-identical (display-only guarantee, D-024);
* success/failure is judged by result.json presence, not process exit
  code (torch teardown exits 9 after success on this box).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conftest import make_unisolar_folder

REPO_ROOT = Path(__file__).resolve().parent.parent


def _hash_tree(p: Path) -> dict:
    return {str(f.relative_to(p)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(p.rglob("*")) if f.is_file()}


def _run(model: str, raw: Path, tmp: Path) -> tuple[Path, subprocess.CompletedProcess]:
    job = tmp / f"job-{model}"
    before_models = _hash_tree(REPO_ROOT / "models") if (REPO_ROOT / "models").exists() else {}
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "train_from_folder.py"),
         "--dataset-dir", str(job), "--raw", str(raw),
         "--model", model, "--fast-test"],
        capture_output=True, text=True, timeout=600, cwd=str(REPO_ROOT))
    after_models = _hash_tree(REPO_ROOT / "models") if (REPO_ROOT / "models").exists() else {}
    assert before_models == after_models, "train job touched served models/"
    return job, proc


@pytest.mark.parametrize("model", ["xgboost", "transformer"])
def test_fast_test_pipeline_end_to_end(tmp_path, model):
    raw = make_unisolar_folder(tmp_path)
    job, proc = _run(model, raw, tmp_path)

    # judge by artifacts, not exit code (torch teardown may exit 9)
    assert "== DONE" in proc.stdout, f"pipeline failed:\n{proc.stdout[-3000:]}"
    res = json.loads((job / "result.json").read_text(encoding="utf-8"))
    assert set(res) >= {"dataset", "cleaning_and_prepare", "split",
                        "persistence", "model", "metrics_per_site",
                        "test_all", "val_all", "timing"}
    assert res["model"]["model_name"] == model
    assert res["cleaning_and_prepare"]["features_cols"] > 14

    # FULL-table persistence: zero missing predictions across val AND test
    for split in ("val_all", "test_all"):
        assert res["persistence"][split]["n_missing"] == 0, (
            f"{split}: persistence predictions NaN-starved — fit-scope bug")

    # model artifacts confined to the job dir
    assert (job / "artifacts" / model / "metrics.csv").exists()
    assert (job / "artifacts" / model / "predictions_test.parquet").exists()

    # metrics grid: val+test × ALL+SITE rows present
    mps = res["metrics_per_site"]
    scopes = {(r["split"], r["scope"]) for r in mps}
    assert {("test", "ALL"), ("test", "SITE"), ("val", "ALL")} <= scopes
```

- [ ] **Step 3: Run to verify — expect the persistence assertion to FAIL**

```bash
conda run -n solar python -m pytest tests/test_train_pipeline.py -v
```

Expected: both parametrized cases run the real pipeline (seconds each in fast-test mode); the `n_missing == 0` assertion FAILS on `test_all` (NaN-starved test predictions). If instead everything passes, the bug doesn't reproduce — stop and investigate `PersistenceBaseline.predict` before proceeding.

- [ ] **Step 4: Fix do_baseline fit scope** — in `scripts/train_from_folder.py`, replace:

```python
    base = PersistenceBaseline().fit(train)
```

with:

```python
    # Fit on the FULL table (D-011 #3): the causal t−24h lookups reach back
    # into val/test history, so fitting train-only NaN-starves test preds.
    base = PersistenceBaseline().fit(df)
```

and delete the now-wrong `del df` line right below it (the baseline keeps `df` alive via its lookup table).

- [ ] **Step 5: Re-run pipeline tests — expect PASS**

```bash
conda run -n solar python -m pytest tests/test_train_pipeline.py -v
```

Expected: 2 PASS.

- [ ] **Step 6: Full suite + commit**

```bash
conda run -n solar python -m pytest -q
git add tests/conftest.py tests/test_train_pipeline.py scripts/train_from_folder.py \
  && git commit -m "fix(train): persistence baseline fits full table (D-011 #3); pin pipeline e2e on synthetic folder"
```

Expected: all green (152 passed).

---

### Task 4: Train API tests — dataset verification endpoints

**Files:**
- Modify: `tests/test_train_api.py`
- Uses: `src/api/app.create_app(store)` — the train router mounts regardless of store type, so pass a minimal fake store object (the router never touches it).

**Interfaces:**
- Consumes: `make_unisolar_folder` from Task 3; `create_app` signature `create_app(store)` (app.py:118).
- Produces: nothing downstream yet; Task 5 appends to this file.

- [ ] **Step 1: Write failing tests** — append to `tests/test_train_api.py`:

```python
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
        assert body["profile"]["cadence_minutes"] == 15
        assert body["profile"]["target_missing_pct"] == 0.0
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

        import pandas as pd

        from conftest import TS_FMT
        import numpy as np

        ts = pd.date_range("2022-01-01", periods=96 * 3, freq="15min")
        hour = ts.hour + ts.minute / 60
        elev = np.clip(np.sin((hour - 6) / 24 * 2 * np.pi) * 60, 0, None)
        gen = pd.DataFrame({
            "SiteKey": 1, "CampusKey": 7, "Timestamp": ts.strftime(TS_FMT),
            "SolarGeneration": elev * 0.2})
        wx = pd.DataFrame({
            "CampusKey": 7, "Timestamp": ts.strftime(TS_FMT),
            "AirTemperature": 20.0, "RelativeHumidity": 60.0})
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
```

- [ ] **Step 2: Run to verify**

```bash
conda run -n solar python -m pytest tests/test_train_api.py -v
```

Expected: the three earlier `_final_state` tests still pass; ALL new tests in `TestVerifyPath`/`TestVerifyUpload` PASS already (implementation exists — these characterize the built endpoints). If any fails, that's a real endpoint bug — fix `src/api/train.py` minimally, note it in the commit message.

- [ ] **Step 3: Full suite + commit**

```bash
conda run -n solar python -m pytest -q
git add tests/test_train_api.py && git commit -m "test(train): dataset verify endpoints (path/upload, structured 422s)"
```

Expected: 159 passed.

---

### Task 5: Train API tests — job lifecycle, artifacts, config

Fake subprocess: monkeypatch `src.api.train.subprocess.Popen` so "training" instantly writes stage markers + a minimal `result.json` into the job dir, then a controllable `poll()` for the one-at-a-time rule.

**Files:**
- Modify: `tests/test_train_api.py`
- Modify: `src/api/train.py` — only if a test exposes a genuine defect (same policy as Task 4)

**Interfaces:**
- Consumes: `client` fixture, `_final_state` (Task 2), `JOBS_ROOT` layout `data/train_jobs/<dataset_id>/<job_id>/`.
- Produces: confidence that Tasks 11's live run exercises tested code paths.

- [ ] **Step 1: Write failing tests** — append to `tests/test_train_api.py`:

```python
import json

import src.api.train as train_mod


def _verify_dataset(client, tmp_path):
    raw = make_unisolar_folder(tmp_path)
    r = client.post("/api/v1/train/datasets/path", json={"path": str(raw)})
    return r.json()["dataset_id"]


RESULT_KEYS = {"generated_at", "dataset", "split", "persistence",
               "model", "metrics_per_site", "test_all", "val_all", "timing"}


def _fake_result(job_dir: Path) -> None:
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


class TestJobLifecycle:
    def test_start_status_artifacts(self, client, tmp_path, monkeypatch):
        ds = _verify_dataset(client, tmp_path)
        seen_cmds = []

        def fake_popen(cmd, **kw):
            seen_cmds.append(cmd)
            return _Proc(cmd)

        monkeypatch.setattr(train_mod.subprocess, "Popen", fake_popen)
        r = client.post("/api/v1/train/jobs",
                        json={"dataset_id": ds, "model": "xgboost"})
        assert r.status_code == 200
        job_id = r.json()["job_id"]
        assert "--fast-test" not in " ".join(seen_cmds[0])

        s = client.get(f"/api/v1/train/jobs/{job_id}").json()
        assert s["status"] == "done"          # exit code 9 must NOT mean failed
        assert [st["name"] for st in s["stages"]] == ["verify"]
        assert s["result"]["model"]["model_name"] == "xgboost"

        a = client.get(f"/api/v1/train/jobs/{job_id}/artifacts/result.json")
        assert a.status_code == 200 and a.json()["model"]["model_name"] == "xgboost"

    def test_unknown_dataset_404_and_bad_model_422(self, client):
        r = client.post("/api/v1/train/jobs",
                        json={"dataset_id": "nope", "model": "xgboost"})
        assert r.status_code == 404
        ds_body = {"dataset_id": "x", "model": "not-a-model"}
        # unknown dataset checked first? model validated first in handler —
        # either 404 or 422 acceptable here; pin exact behavior:
        r2 = client.post("/api/v1/train/jobs", json=ds_body)
        assert r2.status_code in (404, 422)

    def test_bad_artifact_name_422_and_unknown_job_404(self, client):
        r = client.get("/api/v1/train/jobs/x/artifacts/secrets.pem")
        assert r.status_code == 422
        r = client.get("/api/v1/train/jobs/no-such-job")
        assert r.status_code == 404

    def test_one_job_at_a_time_409(self, client, tmp_path, monkeypatch):
        import threading

        ds = _verify_dataset(client, tmp_path)
        release = threading.Event()
        state = {"first": True}

        def fake_popen(cmd, **kw):
            p = _Proc(cmd, release=release)
            if state["first"]:      # first job hangs until released
                state["first"] = False
                p.poll = lambda: None if not release.is_set() else 0
            return p

        monkeypatch.setattr(train_mod.subprocess, "Popen", fake_popen)
        j1 = client.post("/api/v1/train/jobs",
                         json={"dataset_id": ds, "model": "lstm"}).json()["job_id"]
        r = client.post("/api/v1/train/jobs",
                        json={"dataset_id": ds, "model": "gru"})
        assert r.status_code == 409
        release.set()               # let the watcher reap job 1
        deadline = __import__("time").time() + 10
        while __import__("time").time() < deadline:
            if train_mod._active_job_id is None:
                break
            __import__("time").sleep(0.05)


class TestConfigEndpoint:
    def test_config_shape(self, client):
        r = client.get("/api/v1/train/config")
        assert r.status_code == 200
        body = r.json()
        assert {"seed", "train_ratio"} <= set(body["training"])
        assert set(body["models"]) == {"xgboost", "lstm", "gru", "transformer"}
```

- [ ] **Step 2: Run to verify**

```bash
conda run -n solar python -m pytest tests/test_train_api.py -v
```

Expected: all new tests PASS against the existing implementation. Watch `test_start_status_artifacts` closely — it pins the Task 2 fix through the real watcher thread. If the 409 test flakes on watcher timing, increase the deadline loop granularity but keep the assertion.

- [ ] **Step 3: Full suite + commit**

```bash
conda run -n solar python -m pytest -q
git add tests/test_train_api.py && git commit -m "test(train): job lifecycle w/ fake subprocess, artifacts, config, one-at-a-time"
```

Expected: 166 passed.

---

### Task 6: Export builders — `eda_profiles` + `missingness_timeline`

**Files:**
- Modify: `scripts/export_frontend_data.py`
- Test: `tests/test_export_bundles.py` (create)

**Interfaces:**
- Produces:
  - `eda_profiles_bundle(root: Path = ROOT) -> dict` — `{"hour_of_day": {"slots": list[str] (96 "%H:%M"), "mean_kw": {"ALL": [...], "<campus_id>": [...]}}, "correlation": {"campuses": [int], "vars": ["temperature","humidity","wind_speed","solar_elevation_deg"], "power_corr": [[r|null] × campuses × vars]}}`
  - `missingness_timeline_bundle(root: Path = ROOT) -> dict` — `{"months": ["2020-01", ...], "generation_missing_slot_pct": [float|null]}`
  - Both accept a `root` override for testing.
- Consumers: Task 8 (Dashboard), Task 9 (Quality).

- [ ] **Step 1: Write failing tests** — create `tests/test_export_bundles.py`:

```python
"""Unit tests for the post-PRD export bundle builders (D-25 follow-up).

Each builder takes a root path; tests construct a miniature replica of the
real directory layout (partitioned processed parquet + tiny CSV artifacts).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _mini_root(tmp_path: Path) -> Path:
    """2 sites × 1 campus, 3 days of 15-min rows, partitioned parquet."""
    ts = pd.date_range("2022-01-01", periods=96 * 3, freq="15min")
    hour = ts.hour + ts.minute / 60
    frames = []
    for sid in (1, 2):
        power = np.clip(np.sin((hour - 6) / 24 * 2 * np.pi), 0, None) * (2 * sid)
        df = pd.DataFrame({
            "timestamp": ts, "site_id": sid, "campus_id": 7,
            "power": power,
            "temperature": 20 + sid, "humidity": 55.0,
            "wind_speed": 3.0, "solar_elevation_deg":
                np.sin((hour - 6) / 24 * 2 * np.pi) * 60,
        })
        df["year"], df["month"] = df["timestamp"].dt.year, df["timestamp"].dt.month
        frames.append(df)
    out = tmp_path / "data" / "processed" / "solar"
    out.mkdir(parents=True)
    big = pd.concat(frames, ignore_index=True)
    big.to_parquet(out, engine="pyarrow",
                   partition_cols=["site_id", "year", "month"], index=False)
    det = tmp_path / "data" / "processed"
    pd.DataFrame([{"site_id": 1, "campus_id": 7},
                  {"site_id": 2, "campus_id": 7}]).to_parquet(
        det / "site_details.parquet", index=False)
    return tmp_path


class TestEdaProfiles:
    def test_slots_campuses_and_correlation(self, tmp_path):
        from scripts.export_frontend_data import eda_profiles_bundle

        b = eda_profiles_bundle(_mini_root(tmp_path))
        assert len(b["hour_of_day"]["slots"]) == 96
        assert b["hour_of_day"]["slots"][36] == "09:00"
        mean_kw = b["hour_of_day"]["mean_kw"]
        assert set(mean_kw) == {"ALL", "7"}
        # site 1 peaks 2 kW, site 2 peaks 4 kW → campus peak ≈ 3 kW
        assert max(v for v in mean_kw["7"] if v is not None) == 3.0
        corr = b["correlation"]
        assert corr["campuses"] == [7]
        assert corr["vars"] == ["temperature", "humidity", "wind_speed",
                                "solar_elevation_deg"]
        r_elev = corr["power_corr"][0][3]
        assert r_elev is not None and r_elev > 0.95   # power ∝ elevation sine


class TestMissingnessTimeline:
    def test_pct_bounds_and_month_span(self, tmp_path):
        from scripts.export_frontend_data import missingness_timeline_bundle

        b = missingness_timeline_bundle(_mini_root(tmp_path))
        assert b["months"][0] == "2022-01"
        assert len(b["months"]) == len(b["generation_missing_slot_pct"]) == 1
        pct = b["generation_missing_slot_pct"][0]
        assert pct == 0.0          # synthetic grid has no gaps

    def test_gap_counts_as_missing(self, tmp_path):
        from scripts.export_frontend_data import missingness_timeline_bundle

        root = _mini_root(tmp_path)
        # drop 100 rows of site 2 → expected grid unchanged, actual shrinks
        d = root / "data" / "processed" / "solar"
        df = pd.read_parquet(d)
        keep = df[~((df.site_id == 2) & (df.timestamp < df.timestamp.min()
                                         + pd.Timedelta(hours=25)))]
        import shutil
        shutil.rmtree(d)
        keep["year"], keep["month"] = keep["timestamp"].dt.year, keep["timestamp"].dt.month
        keep.to_parquet(d, engine="pyarrow",
                        partition_cols=["site_id", "year", "month"], index=False)
        b = missingness_timeline_bundle(root)
        assert b["generation_missing_slot_pct"][0] > 0
```

- [ ] **Step 2: Run to verify they fail**

```bash
conda run -n solar python -m pytest tests/test_export_bundles.py -v
```

Expected: ImportError — `eda_profiles_bundle` doesn't exist. (If pytest can't import `scripts.export_frontend_data` because `scripts/` lacks `__init__.py`, add an empty `scripts/__init__.py`.)

- [ ] **Step 3: Implement both builders** — add to `scripts/export_frontend_data.py`:

```python
EDA_VARS = ["temperature", "humidity", "wind_speed", "solar_elevation_deg"]


def eda_profiles_bundle(root: Path = ROOT) -> dict:
    """Hourly-of-day generation profile per campus + weather↔power Pearson
    correlations on daylight-observed rows (Dashboard additions)."""
    processed = root / "data" / "processed" / "solar"
    details = pd.read_parquet(root / "data" / "processed" / "site_details.parquet")
    campus_of = {int(r.site_id): int(r.campus_id) for r in details.itertuples()}

    slot_sum: dict[int, np.ndarray] = {}
    slot_cnt: dict[int, np.ndarray] = {}
    corr_frames: dict[int, list[pd.DataFrame]] = {}
    cols = ["timestamp", "power", "is_daylight", "temperature", "humidity",
            "wind_speed", "solar_elevation_deg"]
    for sid in sorted(campus_of):
        df = pd.read_parquet(processed, filters=[("site_id", "=", sid)],
                             columns=cols)
        day = df["is_daylight"].astype(bool) & df["power"].notna()
        slots = (df.loc[day, "timestamp"].dt.hour * 4
                 + df.loc[day, "timestamp"].dt.minute // 15).to_numpy()
        c = campus_of[sid]
        slot_sum[c] = (slot_sum.get(c, np.zeros(96))
                       + np.bincount(slots, weights=df.loc[day, "power"],
                                     minlength=96))
        slot_cnt[c] = slot_cnt.get(c, np.zeros(96)) + np.bincount(slots, minlength=96)
        sub = df.loc[day, ["power"] + EDA_VARS].dropna()
        if len(sub):
            corr_frames.setdefault(c, []).append(sub)

    def means(sums, cnts):
        return [round(float(s / n), 4) if n > 0 else None
                for s, n in zip(sums, cnts)]

    tot_sum = np.sum(list(slot_sum.values()), axis=0) if slot_sum else np.zeros(96)
    tot_cnt = np.sum(list(slot_cnt.values()), axis=0) if slot_cnt else np.zeros(96)

    corr_out = {}
    for c in sorted(corr_frames):
        pooled = pd.concat(corr_frames[c], ignore_index=True)
        cr = pooled.corr(numeric_only=True)["power"].drop("power")
        corr_out[c] = [None if pd.isna(cr.get(v)) else round(float(cr[v]), 4)
                       for v in EDA_VARS]

    return {
        "hour_of_day": {
            "slots": [(t := i * 15) // 60 and f"{i * 15 // 60:02d}:{i * 15 % 60:02d}"
                      for i in range(96)],
            "mean_kw": {"ALL": means(tot_sum, tot_cnt),
                        **{str(c): means(slot_sum[c], slot_cnt[c])
                           for c in sorted(slot_sum)}},
        },
        "correlation": {
            "campuses": sorted(corr_frames),
            "vars": EDA_VARS,
            "power_corr": [corr_out[c] for c in sorted(corr_frames)],
        },
    }


def missingness_timeline_bundle(root: Path = ROOT) -> dict:
    """Share of each month's expected 15-min grid that carries no row
    (generation grain), across all sites — Quality page timeline."""
    processed = root / "data" / "processed" / "solar"
    details = pd.read_parquet(root / "data" / "processed" / "site_details.parquet")
    exp: dict[str, int] = {}
    act: dict[str, int] = {}
    for det in details.itertuples():
        df = pd.read_parquet(processed,
                             filters=[("site_id", "=", int(det.site_id))],
                             columns=["timestamp"])
        if df.empty:
            continue
        t0, t1 = df["timestamp"].min(), df["timestamp"].max()
        grid = pd.date_range(t0, t1, freq="15min")
        gexp = pd.Series(grid).dt.to_period("M").value_counts()
        gact = df["timestamp"].dt.to_period("M").value_counts()
        for m, n in gexp.items():
            k = str(m)
            exp[k] = exp.get(k, 0) + int(n)
            act[k] = act.get(k, 0) + int(gact.get(m, 0))
    months = sorted(exp)
    return {"months": months,
            "generation_missing_slot_pct": [
                round(100 * (1 - act[m] / exp[m]), 3) for m in months]}
```

Fix the awkward walrus in the slots comprehension — write it plainly:

```python
            "slots": [f"{(i * 15) // 60:02d}:{(i * 15) % 60:02d}"
                      for i in range(96)],
```

- [ ] **Step 4: Run builders' tests to green**

```bash
conda run -n solar python -m pytest tests/test_export_bundles.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: Wire into main()** — in `export_frontend_data.py::main`, extend the dict:

```python
    bundles = {
        "shap_global_importance.json": shap_bundle(),
        "site_summary.json": site_summary_bundle(),
        "data_quality.json": data_quality_bundle(),
        "site_monthly.json": site_monthly_bundle(),
        "quality_extra.json": quality_extra_bundle(),
        "eda_profiles.json": eda_profiles_bundle(),
        "missingness_timeline.json": missingness_timeline_bundle(),
    }
```

Run the full export and eyeball sizes:

```bash
conda run -n solar python scripts/export_frontend_data.py
ls -la frontend/src/data/
```

Expected: 7 bundles written; `eda_profiles.json` and `missingness_timeline.json` each well under 300 kB.

- [ ] **Step 6: Full suite + commit**

```bash
conda run -n solar python -m pytest -q
git add scripts/export_frontend_data.py tests/test_export_bundles.py scripts/__init__.py frontend/src/data \
  && git commit -m "feat(export): eda_profiles + missingness_timeline bundles"
```

Expected: all green (169 passed).

---

### Task 7: Export builders — `evaluation_series` + `cross_site_summary`

**Files:**
- Modify: `scripts/export_frontend_data.py`
- Modify: `tests/test_export_bundles.py`

**Interfaces:**
- Produces:
  - `evaluation_series_bundle(root: Path = ROOT, seed: int = 0) -> dict` — `{"test_window": {"start","end"}, "models": {"<run>": {"metrics": {...}, "hourly_all": {"t","actual","predicted"} | omitted, "daily_by_site": {...} | omitted, "scatter_sample": {...} | omitted, "residual_hist": {"edges","counts"} | omitted}}}`. Series fields only for runs with `artifacts/<name>/predictions_test.parquet` (xgboost/lstm/gru/transformer).
  - `cross_site_summary_bundle(root: Path = ROOT) -> dict` — `{"models": {"<model>": {"seen_val_all", "unseen_test_all", "unseen_site_r2"}}}`
- Consumer: Task 10 (Comparison page).

- [ ] **Step 1: Confirm protocol labels** (one-off, temp script — no multiline `python -c`):

Write `tmp_check.py` at repo root:

```python
import pandas as pd
df = pd.read_csv("artifacts/cross_site/cross_site_metrics.csv")
print(sorted(df.protocol.unique()), sorted(df.split.unique()), sorted(df.scope.unique()))
print(sorted(df.model.unique()))
comp = pd.read_csv("artifacts/evaluation/model_comparison.csv")
print(sorted(comp.model.unique()))
```

Run `conda run -n solar python tmp_check.py`. Expected: protocols `['seen','unseen']`, splits `['val','test']`, models include `xgboost lstm gru transformer persistence_prev_day zero mean_global mean_site`. Delete `tmp_check.py` afterwards. If labels differ, adjust the literals in Step 3.

- [ ] **Step 2: Write failing tests** — append to `tests/test_export_bundles.py`:

```python
def _mini_artifacts(root: Path) -> None:
    """Fake evaluation + cross-site + prediction artifacts alongside _mini_root."""
    art = root / "artifacts"
    comp = pd.DataFrame([
        {"model": "xgboost", "split": "test", "scope": "ALL", "site_id": "",
         "n_eval": 100, "mae": 1.0, "rmse": 2.0, "r2": 0.9, "nrmse": 0.02,
         "daylight_mae": 1.1, "training_seconds": 1.0},
        {"model": "zero", "split": "test", "scope": "ALL", "site_id": "",
         "n_eval": 100, "mae": 3.0, "rmse": 4.0, "r2": 0.1, "nrmse": 0.05,
         "daylight_mae": 3.1, "training_seconds": 0.0},
    ])
    (art / "evaluation").mkdir(parents=True, exist_ok=True)
    comp.to_csv(art / "evaluation" / "model_comparison.csv", index=False)

    ts = pd.date_range("2022-01-03", periods=96, freq="15min")
    for name in ("xgboost",):
        d = pd.DataFrame({"site_id": 1, "timestamp": ts,
                          "power": np.where(ts.hour.between(6, 18), 2.0, np.nan),
                          "is_daylight": ts.hour.between(6, 18),
                          "prediction": np.full(96, 1.5)})
        (art / name).mkdir(parents=True, exist_ok=True)
        d.to_parquet(art / name / "predictions_test.parquet", index=False)

    cs = pd.DataFrame([
        {"model": "xgboost", "protocol": "seen", "split": "val", "scope": "ALL",
         "site_id": "", "n_eval": 10, "n_missing": 0, "mae": 1.0, "rmse": 1.5,
         "r2": 0.92, "nrmse": 0.02, "daylight_n": 10, "daylight_mae": 1.0,
         "daylight_nrmse": 0.02},
        {"model": "xgboost", "protocol": "unseen", "split": "test", "scope": "ALL",
         "site_id": "", "n_eval": 10, "n_missing": 0, "mae": 2.0, "rmse": 2.5,
         "r2": 0.80, "nrmse": 0.03, "daylight_n": 10, "daylight_mae": 2.0,
         "daylight_nrmse": 0.03},
        {"model": "xgboost", "protocol": "unseen", "split": "test", "scope": "SITE",
         "site_id": 29, "n_eval": 10, "n_missing": 0, "mae": 2.2, "rmse": 2.7,
         "r2": -1.5, "nrmse": 0.04, "daylight_n": 10, "daylight_mae": 2.2,
         "daylight_nrmse": 0.04},
    ])
    (art / "cross_site").mkdir(parents=True, exist_ok=True)
    cs.to_csv(art / "cross_site" / "cross_site_metrics.csv", index=False)


class TestEvaluationSeries:
    def test_metrics_for_all_runs_series_only_where_parquet(self, tmp_path):
        from scripts.export_frontend_data import evaluation_series_bundle

        root = _mini_root(tmp_path)
        _mini_artifacts(root)
        b = evaluation_series_bundle(root)
        assert set(b["models"]) == {"xgboost", "zero"}
        assert b["models"]["zero"]["metrics"]["mae"] == 3.0
        assert "hourly_all" not in b["models"]["zero"]

        ser = b["models"]["xgboost"]
        assert ser["metrics"]["mae"] == 1.0
        assert len(ser["hourly_all"]["t"]) == 24          # 96 × 15-min → 24 h buckets
        assert len(ser["residual_hist"]["edges"]) == 42   # −10..10 step 0.5
        assert sum(ser["residual_hist"]["counts"]) == 48  # daylight rows only
        assert len(ser["scatter_sample"]["actual"]) <= 2000

    def test_test_window_from_predictions(self, tmp_path):
        from scripts.export_frontend_data import evaluation_series_bundle

        root = _mini_root(tmp_path)
        _mini_artifacts(root)
        b = evaluation_series_bundle(root)
        assert b["test_window"]["start"].startswith("2022-01-03")


class TestCrossSiteSummary:
    def test_rollups_and_negative_r2_preserved(self, tmp_path):
        from scripts.export_frontend_data import cross_site_summary_bundle

        root = _mini_root(tmp_path)
        _mini_artifacts(root)
        b = cross_site_summary_bundle(root)
        m = b["models"]["xgboost"]
        assert m["seen_val_all"]["mae"] == 1.0
        assert m["unseen_test_all"]["r2"] == 0.8
        assert m["unseen_site_r2"] == [{"site_id": 29, "r2": -1.5}]
```

- [ ] **Step 3: Implement** — add to `scripts/export_frontend_data.py`:

```python
SERIES_RUNS = ("xgboost", "lstm", "gru", "transformer")
RESID_EDGES = np.arange(-10.0, 10.01, 0.5)


def _metric4(row) -> dict:
    return {"mae": round(float(row.mae), 4), "rmse": round(float(row.rmse), 4),
            "r2": round(float(row.r2), 4), "nrmse": round(float(row.nrmse), 4)}


def evaluation_series_bundle(root: Path = ROOT, seed: int = 0) -> dict:
    """Per-run test-window aggregates for the Model Comparison page.

    Metrics come from artifacts/evaluation/model_comparison.csv (all runs);
    series fields only where artifacts/<run>/predictions_test.parquet exists.
    Night rows carry NaN actuals — dropped from means/histograms, and gaps
    stay null in overlays (never bridged with zeros).
    """
    comp = pd.read_csv(root / "artifacts" / "evaluation" / "model_comparison.csv")
    rng = np.random.default_rng(seed)
    models: dict = {}
    window: dict | None = None

    for _, row in comp[(comp.scope == "ALL") & (comp.split == "test")].iterrows():
        entry: dict = {"metrics": _metric4(row)}
        models[str(row.model)] = entry
        pq = root / "artifacts" / str(row.model) / "predictions_test.parquet"
        if str(row.model) not in SERIES_RUNS or not pq.exists():
            continue
        df = pd.read_parquet(pq)
        if window is None:
            window = {"start": str(df["timestamp"].min()),
                      "end": str(df["timestamp"].max())}
        obs = df.loc[df["power"].notna()]

        hourly_a = obs.set_index("timestamp")["power"].resample("1h").mean()
        hourly_p = df.set_index("timestamp")["prediction"].resample("1h").mean()
        idx = hourly_a.index.union(hourly_p.index)
        hourly_a, hourly_p = hourly_a.reindex(idx), hourly_p.reindex(idx)
        entry["hourly_all"] = {
            "t": [str(t) for t in idx],
            "actual": [None if pd.isna(v) else round(float(v), 4)
                       for v in hourly_a],
            "predicted": [None if pd.isna(v) else round(float(v), 4)
                          for v in hourly_p],
        }

        daily_a = (obs.assign(day=obs["timestamp"].dt.date)
                   .groupby(["site_id", "day"]))[["power"]].mean()
        daily_p = (df.assign(day=df["timestamp"].dt.date)
                    .groupby(["site_id", "day"]))[["prediction"]].mean()
        j = daily_a.join(daily_p, how="outer").reset_index()
        daily: dict[int, dict] = {}
        for sid, g in j.groupby("site_id"):
            daily[int(sid)] = {
                "actual": [None if pd.isna(v) else round(float(v), 4)
                           for v in g.sort_values("day")["power"]],
                "predicted": [None if pd.isna(v) else round(float(v), 4)
                              for v in g.sort_values("day")["prediction"]]}
        entry["daily_by_site"] = {str(k): v for k, v in sorted(daily.items())}

        take = min(2000, len(obs))
        sel = rng.choice(len(obs), size=take, replace=False) if take else []
        s = obs.iloc[sorted(sel)] if len(sel) else obs.iloc[0:0]
        entry["scatter_sample"] = {
            "actual": [round(float(v), 4) for v in s["power"]],
            "predicted": [round(float(v), 4) for v in s["prediction"]]}

        resid = (obs["prediction"] - obs["power"]).to_numpy()
        counts, _ = np.histogram(resid, bins=RESID_EDGES)
        entry["residual_hist"] = {
            "edges": [round(float(e), 2) for e in RESID_EDGES],
            "counts": [int(c) for c in counts]}

    return {"test_window": window or {}, "models": models}


def cross_site_summary_bundle(root: Path = ROOT) -> dict:
    """Seen-val vs unseen-test rollups per model + unseen per-site R² strip
    (verbatim from Phase 9 cross-site_metrics.csv)."""
    df = pd.read_csv(root / "artifacts" / "cross_site" / "cross_site_metrics.csv")

    def grab(g, protocol, split):
        r = g[(g.protocol == protocol) & (g.split == split) & (g.scope == "ALL")]
        return _metric4(r.iloc[0]) if len(r) else None

    out = {}
    for model, g in df.groupby("model"):
        unseen_sites = g[(g.protocol == "unseen") & (g.split == "test")
                         & (g.scope == "SITE")]
        out[str(model)] = {
            "seen_val_all": grab(g, "seen", "val"),
            "unseen_test_all": grab(g, "unseen", "test"),
            "unseen_site_r2": [{"site_id": int(r.site_id),
                                "r2": round(float(r.r2), 4)}
                               for r in unseen_sites.itertuples()],
        }
    return {"models": out}
```

- [ ] **Step 4: Run to green + wire main()**

```bash
conda run -n solar python -m pytest tests/test_export_bundles.py -v
```

Expected: 6 PASS. Then extend `bundles` in `main()` with:

```python
        "evaluation_series.json": evaluation_series_bundle(),
        "cross_site_summary.json": cross_site_summary_bundle(),
```

Re-run export; confirm 9 bundles, sizes sane (<300 kB each new):

```bash
conda run -n solar python scripts/export_frontend_data.py
```

- [ ] **Step 5: Full suite + commit**

```bash
conda run -n solar python -m pytest -q
rm -f tmp_check.py
git add scripts/export_frontend_data.py tests/test_export_bundles.py frontend/src/data \
  && git commit -m "feat(export): evaluation_series + cross_site_summary bundles"
```

Expected: 175 passed.

---

### Task 8: Dashboard — hourly profile + correlation heatmap

**Files:**
- Modify: `frontend/src/lib/types.ts` (add all new bundle types once — later tasks consume them)
- Modify: `frontend/src/pages/dashboard.tsx`

**Interfaces:**
- Consumes: `eda_profiles.json` (Task 6).
- Produces: types `EdaProfilesBundle`, `MissingnessTimelineBundle`, `EvalSeriesBundle`, `CrossSiteSummaryBundle` exported from types.ts.

- [ ] **Step 1: Add types** — append to `frontend/src/lib/types.ts`:

```typescript
/* ---- Post-PRD visual bundles (export_frontend_data.py) ---- */

export interface EdaProfilesBundle {
  hour_of_day: {
    slots: string[]
    mean_kw: Record<string, (number | null)[]>   // "ALL" + campus ids
  }
  correlation: {
    campuses: number[]
    vars: string[]
    power_corr: (number | null)[][]              // campuses × vars
  }
}

export interface MissingnessTimelineBundle {
  months: string[]
  generation_missing_slot_pct: number[]
}

export interface EvalSeriesMetrics {
  mae: number
  rmse: number
  r2: number
  nrmse: number
}

export interface HourlySeries {
  t: string[]
  actual: (number | null)[]
  predicted: (number | null)[]
}

export interface ResidualHist {
  edges: number[]     // 41 edges → 40 bins
  counts: number[]
}

export interface EvalSeriesEntry {
  metrics: EvalSeriesMetrics
  hourly_all?: HourlySeries
  daily_by_site?: Record<string, { actual: (number | null)[]; predicted: (number | null)[] }>
  scatter_sample?: { actual: number[]; predicted: number[] }
  residual_hist?: ResidualHist
}

export interface EvalSeriesBundle {
  test_window: { start: string; end: string }
  models: Record<string, EvalSeriesEntry>
}

export interface CrossSiteEntry {
  seen_val_all: EvalSeriesMetrics | null
  unseen_test_all: EvalSeriesMetrics | null
  unseen_site_r2: { site_id: number; r2: number }[]
}

export interface CrossSiteSummaryBundle {
  models: Record<string, CrossSiteEntry>
}
```

- [ ] **Step 2: Mount charts on Dashboard** — in `dashboard.tsx`, add import near the other JSON import:

```tsx
import edaJson from '@/data/eda_profiles.json'
import type { EdaProfilesBundle } from '@/lib/types'
```

and at module level:

```tsx
const eda = edaJson as EdaProfilesBundle

const CORR_VAR_LABELS: Record<string, string> = {
  temperature: 'Temperature',
  humidity: 'Humidity',
  wind_speed: 'Wind speed',
  solar_elevation_deg: 'Solar elevation',
}
```

Append two components at the bottom of the file (after `CampusChart`):

```tsx
/** Mean daylight-slot kW per time-of-day — ALL bold, campuses thin. */
function HourlyProfileChart() {
  const { slots, mean_kw } = eda.hour_of_day
  const rows = slots.map((slot, i) => {
    const pt: Record<string, string | number | null> = { slot }
    for (const [key, arr] of Object.entries(mean_kw))
      pt[key === 'ALL' ? 'all' : `c${key}`] = arr[i]
    return pt
  })
  const campusKeys = Object.keys(mean_kw).filter((k) => k !== 'ALL')
  return (
    <figure className="h-72">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={rows} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid {...gridProps} />
          <XAxis dataKey="slot" tick={{ ...axisTick }} tickLine={false}
                 axisLine={{ stroke: 'var(--viz-axis)' }} minTickGap={28} />
          <YAxis width={44} tick={{ ...axisTick }} tickLine={false} axisLine={false} />
          <RTooltip content={<ChartTooltip fmtValue={(v) => `${fmtNum(v, 2)} kW`} />} />
          <Legend wrapperStyle={legendStyle()} />
          {campusKeys.map((k, i) => (
            <Line key={k} type="monotone" dataKey={`c${k}`}
                  name={`campus ${k}`}
                  stroke={CAMPUS_PALETTE[i % CAMPUS_PALETTE.length]}
                  strokeWidth={1} strokeOpacity={0.55} dot={false}
                  isAnimationActive={false} />
          ))}
          <Line type="monotone" dataKey="all" name="all sites"
                stroke="var(--series-observed)" strokeWidth={2.5} dot={false}
                isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </figure>
  )
}

/** CSS-grid correlation heatmap — color-mix intensity encodes |r|. */
function CorrHeatmap() {
  const { campuses, vars, power_corr } = eda.correlation
  const cellBg = (r: number | null) =>
    r === null ? 'var(--muted)'
      : `color-mix(in oklab, var(--chart-1) ${Math.round(Math.abs(r) * 100)}%, transparent)`
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-left text-muted-foreground">
            <th className="px-2 py-1 font-medium">Campus</th>
            {vars.map((v) => (
              <th key={v} className="px-2 py-1 font-medium">{CORR_VAR_LABELS[v] ?? v}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {campuses.map((c, ri) => (
            <tr key={c} className="border-t">
              <td className="px-2 py-1.5 font-mono">campus {c}</td>
              {vars.map((_v, ci) => {
                const r = power_corr[ri]?.[ci] ?? null
                return (
                  <td key={ci} className="px-1.5 py-1.5">
                    <div
                      className="rounded px-2 py-1 text-center font-mono tabular-nums"
                      style={{ background: cellBg(r) }}
                      title={`campus ${c} · ${vars[ci]} · r=${r ?? '—'}`}
                    >
                      {r === null ? '—' : fmtNum(r, 2)}
                    </div>
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-[11px] text-muted-foreground">
        Pearson r vs observed daylight power, pooled over each campus's sites.
      </p>
    </div>
  )
}
```

And inside the main return, extend the second grid section (after the Campus comparison VizCard's closing `</div>` at line ~340) with a new full-width block before `{models.data && …}`:

```tsx
      <div className="grid gap-4 lg:grid-cols-2">
        <VizCard
          title="Generation profile by time of day"
          description="Mean observed daylight-slot power per 15-min slot — all sites bold, campuses thin."
        >
          <HourlyProfileChart />
        </VizCard>

        <VizCard
          title="Weather ↔ power correlation"
          description="Which covariates actually move generation, per campus."
        >
          <CorrHeatmap />
        </VizCard>
      </div>
```

- [ ] **Step 3: Verify visually + lint**

```bash
cd /e/Solar_gemini/frontend && npm run build && npx oxlint src/pages/dashboard.tsx
```

Expected: build green, 0 errors. Then `npm run dev` and eyeball `/` — profile lines render, heatmap cells tinted.

- [ ] **Step 4: Screenshot + commit**

Playwright screenshot of the dashboard bottom section; save to `docs/superpowers/plans/shots/dashboard-new-cards.png` (dir `shots/` is gitignored? No — commit shots; they're small evidence).

```bash
git add frontend/src/lib/types.ts frontend/src/pages/dashboard.tsx docs/superpowers/plans/shots \
  && git commit -m "feat(frontend): dashboard hourly-profile + correlation-heatmap cards"
```

---

### Task 9: Quality — generation gap timeline

**Files:**
- Modify: `frontend/src/pages/quality.tsx`
- Types: already added in Task 8.

**Interfaces:**
- Consumes: `missingness_timeline.json` (Task 6), `MissingnessTimelineBundle`.

- [ ] **Step 1: Mount the chart** — add imports:

```tsx
import missingnessJson from '@/data/missingness_timeline.json'
import type { MissingnessTimelineBundle } from '@/lib/types'
```

module level:

```tsx
const mt = missingnessJson as MissingnessTimelineBundle
```

Insert a new VizCard directly after the "Weather coverage by month" card (before the first `grid gap-4 lg:grid-cols-2`):

```tsx
      <VizCard
        title="Generation slot gaps by month"
        description={
          `Missing 15-min slots vs each site's expected grid, summed across ` +
          `${mt.months.length} months. Flat at zero except where reporting drops.`
        }
      >
        <figure className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={mt.months.map((m, i) => ({
                month: m.slice(2),
                pct: mt.generation_missing_slot_pct[i],
              }))}
              margin={{ top: 8, right: 8, left: 0, bottom: 0 }}
            >
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="month" tick={{ ...axisTick }} tickLine={false}
                     axisLine={{ stroke: 'var(--viz-axis)' }} minTickGap={24} />
              <YAxis width={44} domain={[0, (dataMax: number) => Math.max(1, dataMax * 1.1)]}
                     tickFormatter={(v: number) => `${fmtNum(v, 1)}%`}
                     tick={{ ...axisTick }} tickLine={false} axisLine={false} />
              <RTooltip
                content={({ active, payload, label }) => {
                  if (!active || !payload?.length) return null
                  return (
                    <div className="rounded-lg border bg-popover px-3 py-2 text-xs shadow-md">
                      <div className="mb-1 font-mono text-muted-foreground">20{label}</div>
                      <div className="font-mono tabular-nums">
                        {(payload[0].value as number).toFixed(2)}% slots missing
                      </div>
                    </div>
                  )
                }}
              />
              <Bar dataKey="pct" name="missing slots" fill="var(--chart-5)"
                   radius={[2, 2, 0, 0]} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </figure>
      </VizCard>
```

(`--chart-5` is the palette's warning tone — already used by quality.tsx availability bars.)

- [ ] **Step 2: Build + screenshot + commit**

```bash
cd /e/Solar_gemini/frontend && npm run build && npx oxlint src/pages/quality.tsx
```

Screenshot `/quality` → `docs/superpowers/plans/shots/quality-gap-timeline.png`.

```bash
git add frontend/src/pages/quality.tsx docs/superpowers/plans/shots \
  && git commit -m "feat(frontend): quality page generation-gap monthly timeline"
```

---

### Task 10: Comparison — all-runs bars, pred-vs-actual explorer, cross-site card

**Files:**
- Modify: `frontend/src/pages/comparison.tsx`

**Interfaces:**
- Consumes: `evaluation_series.json` + `cross_site_summary.json` (Task 7), types from Task 8.

- [ ] **Step 1: Add imports + module constants**:

```tsx
import evalJson from '@/data/evaluation_series.json'
import crossJson from '@/data/cross_site_summary.json'
import type { CrossSiteSummaryBundle, EvalSeriesBundle } from '@/lib/types'
import {
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  Tooltip as RTooltip,
} from 'recharts'
import { ChartTooltip, gridProps, legendStyle, VizCard, axisTick } from '@/components/viz'
```

(reconcile with the existing recharts/viz imports — merge, don't duplicate.)

Module level:

```tsx
const ev = evalJson as EvalSeriesBundle
const cs = crossJson as CrossSiteSummaryBundle

const RUN_ORDER = ['zero', 'mean_global', 'mean_site', 'persistence_prev_day',
                   'xgboost', 'lstm', 'gru', 'transformer']
const SERIES_RUNS = ['xgboost', 'lstm', 'gru', 'transformer']
const COMPACT_METRICS = [
  { key: 'mae', label: 'MAE (kWh)', lowerBetter: true },
  { key: 'rmse', label: 'RMSE (kWh)', lowerBetter: true },
  { key: 'r2', label: 'R²', lowerBetter: false },
  { key: 'nrmse', label: 'nRMSE', lowerBetter: true },
] as const
type CompactKey = (typeof COMPACT_METRICS)[number]['key']
```

- [ ] **Step 2: Add three components** (bottom of file):

```tsx
/** All 8 evaluated runs on one selectable compact metric. */
function AllRunsBars() {
  const [metric, setMetric] = useState<CompactKey>('mae')
  const spec = COMPACT_METRICS.find((m) => m.key === metric)!
  const rows = RUN_ORDER
    .filter((r) => ev.models[r])
    .map((r) => ({ run: r, value: ev.models[r].metrics[metric] as number }))
    .filter((r) => Number.isFinite(r.value))
    .sort((a, b) => (spec.lowerBetter ? a.value - b.value : b.value - a.value))
  const best = rows[0]?.run
  return (
    <VizCard
      title={`All evaluated runs — ${spec.label}`}
      description={`${rows.length} runs · Phase 8 comparison artifact · ${spec.lowerBetter ? 'lower' : 'higher'} is better.`}
      action={
        <Select value={metric} onValueChange={(v) => setMetric(v as CompactKey)}>
          <SelectTrigger className="h-7 w-32 text-xs" aria-label="Compact metric">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {COMPACT_METRICS.map((m) => (
              <SelectItem key={m.key} value={m.key}>{m.label}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      }
    >
      <figure className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 48, left: 8, bottom: 0 }}>
            <XAxis type="number" tick={{ ...axisTick }} tickLine={false}
                   axisLine={{ stroke: 'var(--viz-axis)' }}
                   domain={[0, (dataMax: number) => dataMax * 1.08]} />
            <YAxis type="category" dataKey="run" width={128}
                   tick={{ ...axisTick, fontSize: 10 }} tickLine={false} axisLine={false} />
            <RTooltip content={<ChartTooltip fmtValue={(v) => fmtNum(v, 3)} />} />
            <Bar dataKey="value" name={spec.label} barSize={14} radius={[0, 3, 3, 0]}
                 isAnimationActive={false}>
              {rows.map((d) => (
                <Cell key={d.run} fill={d.run === best ? 'var(--series-forecast)' : 'var(--viz-axis)'} />
              ))}
              <LabelList dataKey="value" position="right"
                         formatter={(v) => fmtNum(Number(v), 3)}
                         style={{ fontSize: 10, fontFamily: 'var(--font-mono)',
                                  fill: 'var(--muted-foreground)' }} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </figure>
    </VizCard>
  )
}
```

```tsx
/** Pred-vs-actual overlay + residual histogram for runs with stored test preds. */
function PredActualExplorer() {
  const [run, setRun] = useState('xgboost')
  const [site, setSite] = useState('ALL')
  const entry = ev.models[run]
  const perSiteKeys = Object.keys(entry?.daily_by_site ?? {})

  const overlayRows = useMemo(() => {
    if (!entry) return []
    if (site === 'ALL' || !entry.daily_by_site?.[site]) {
      const h = entry.hourly_all!
      return h.t.map((t, i) => ({
        t: t.slice(5, 16),
        actual: h.actual[i],
        predicted: h.predicted[i],
      }))
    }
    const d = entry.daily_by_site[site]
    return d.actual.map((_, i) => ({
      t: String(i), // day index within test window
      actual: d.actual[i],
      predicted: d.predicted[i],
    }))
  }, [entry, site])

  const histRows = useMemo(() => {
    const h = entry?.residual_hist
    if (!h) return []
    return h.edges.slice(0, -1).map((e, i) => ({
      bin: e.toFixed(1),
      count: h.counts[i],
    }))
  }, [entry])

  return (
    <VizCard
      title="Predicted vs actual — held-out test window"
      description={
        `${ev.test_window.start.slice(0, 10)} → ${ev.test_window.end.slice(0, 10)}. ` +
        'ALL scope shows hourly means; a single site shows daily means. Nulls are honest gaps (night), never bridged.'
      }
      action={
        <div className="flex items-center gap-2">
          <Select value={run} onValueChange={setRun}>
            <SelectTrigger className="h-7 w-32 text-xs" aria-label="Run">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {SERIES_RUNS.filter((r) => ev.models[r]?.hourly_all).map((r) => (
                <SelectItem key={r} value={r}>{r}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Select value={site} onValueChange={setSite}>
            <SelectTrigger className="h-7 w-28 text-xs" aria-label="Site scope">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="ALL">ALL</SelectItem>
              {perSiteKeys.map((s) => (
                <SelectItem key={s} value={s}>site {s}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      }
    >
      <figure className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={overlayRows} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
            <CartesianGrid {...gridProps} />
            <XAxis dataKey="t" tick={{ ...axisTick }} tickLine={false}
                   axisLine={{ stroke: 'var(--viz-axis)' }} minTickGap={40} />
            <YAxis width={44} tick={{ ...axisTick }} tickLine={false} axisLine={false} />
            <RTooltip content={<ChartTooltip fmtValue={(v) => `${fmtNum(v, 2)} kW`} />} />
            <Legend wrapperStyle={legendStyle()} />
            <Line type="monotone" dataKey="actual" name="Observed"
                  stroke="var(--series-observed)" strokeWidth={2} dot={false}
                  connectNulls={false} isAnimationActive={false} />
            <Line type="monotone" dataKey="predicted" name={`Predicted (${run})`}
                  stroke="var(--series-forecast)" strokeWidth={2} dot={false}
                  connectNulls={false} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </figure>
      <figure className="mt-2 h-40">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={histRows} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
            <CartesianGrid {...gridProps} />
            <XAxis dataKey="bin" tick={{ ...axisTick, fontSize: 9 }} tickLine={false}
                   axisLine={{ stroke: 'var(--viz-axis)' }} interval={4} />
            <YAxis width={40} tick={{ ...axisTick }} tickLine={false} axisLine={false} />
            <RTooltip content={<ChartTooltip fmtLabel={(l) => `${l} kW residual`}
                                              fmtValue={(v) => v.toLocaleString()} />} />
            <Bar dataKey="count" name="residuals" fill="var(--chart-3)"
                 radius={[2, 2, 0, 0]} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </figure>
      <p className="mt-1 text-[11px] text-muted-foreground">
        Residual = predicted − actual, daylight-observed rows, ±10 kW bins.
      </p>
    </VizCard>
  )
}
```

```tsx
/** Seen-val vs unseen-test MAE + unseen R² strip (Phase 9). */
function CrossSiteCard() {
  const entries = RUN_ORDER
    .filter((r) => cs.models[r]?.seen_val_all && cs.models[r]?.unseen_test_all)
    .map((r) => ({
      model: r,
      seen: cs.models[r].seen_val_all!.mae,
      unseen: cs.models[r].unseen_test_all!.mae,
    }))
  const r2Strip = Object.entries(cs.models).flatMap(([model, e]) =>
    e.unseen_site_r2.map((s) => ({ model, ...s })))
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <VizCard
        title="Cross-site generalization — MAE"
        description="Seen protocol = chronological val on training sites. Unseen = fully held-out sites' test windows (Phase 9, D-016)."
      >
        <figure className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={entries} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="model" tick={{ ...axisTick, fontSize: 9 }} tickLine={false}
                     axisLine={{ stroke: 'var(--viz-axis)' }} interval={0} angle={-30}
                     textAnchor="end" height={58} />
              <YAxis width={44} tick={{ ...axisTick }} tickLine={false} axisLine={false} />
              <RTooltip content={<ChartTooltip fmtValue={(v) => `${fmtNum(v, 2)} kW`} />} />
              <Legend wrapperStyle={legendStyle()} />
              <Bar dataKey="seen" name="seen (val)" fill="var(--chart-2)"
                   radius={[2, 2, 0, 0]} isAnimationActive={false} />
              <Bar dataKey="unseen" name="unseen (test)" fill="var(--chart-1)"
                   radius={[2, 2, 0, 0]} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </figure>
      </VizCard>

      <VizCard
        title="Unseen-site R² per model"
        description="Each dot is one held-out site. Negative R² (red) = worse than predicting the mean — plant-size mix drives the extremes (compare within protocol)."
      >
        <ul className="max-h-64 space-y-1 overflow-auto text-xs">
          {r2Strip.sort((a, b) => a.r2 - b.r2).map((s) => (
            <li key={`${s.model}-${s.site_id}`} className="flex items-center gap-2">
              <span className="w-28 shrink-0 truncate font-mono">{s.model}</span>
              <span className="w-14 shrink-0 text-right font-mono text-muted-foreground">
                site {s.site_id}
              </span>
              <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
                <div
                  className={'h-full rounded-full ' + (s.r2 < 0 ? 'bg-status-bad-text' : 'bg-chart-2')}
                  style={{ width: `${Math.min(100, Math.max(2, Math.abs(s.r2) * 100))}%` }}
                />
              </div>
              <span className={'w-14 shrink-0 text-right font-mono tabular-nums ' +
                               (s.r2 < 0 ? 'text-status-bad-text' : '')}>
                {fmtNum(s.r2, 3)}
              </span>
            </li>
          ))}
        </ul>
      </VizCard>
    </div>
  )
}
```

- [ ] **Step 3: Mount** — in the main return, after the existing single-metric VizCard:

```tsx
      <PredActualExplorer />
      <AllRunsBars />
      <CrossSiteCard />
```

- [ ] **Step 4: Build + lint + screenshot + commit**

```bash
cd /e/Solar_gemini/frontend && npm run build && npx oxlint src/pages/comparison.tsx
```

Screenshot `/models` (top + scrolled sections) → `docs/superpowers/plans/shots/comparison-new-cards.png`.

```bash
git add frontend/src/pages/comparison.tsx docs/superpowers/plans/shots \
  && git commit -m "feat(frontend): comparison pred-vs-actual explorer, all-runs bars, cross-site card"
```

---

### Task 11: Live verification run (first-ever end-to-end)

**Files:**
- Create: `artifacts/train_live_run/summary.json` (measured numbers; committed)
- Screenshots: `docs/superpowers/plans/shots/train-{verify,running,results}.png`

**Interfaces:**
- Consumes: everything Tasks 2–5 fixed/tested; real dataset at `E:\Solar_gemini\unisolar`.

- [ ] **Step 1: Start backend + frontend** (two background shells):

```bash
cd /e/Solar_gemini && conda run -n solar python scripts/run_api.py &
```

```bash
cd /e/Solar_gemini/frontend && npm run dev &
```

Wait for `Uvicorn running` / Vite ready lines. Confirm `curl -s http://localhost:8000/api/v1/health` returns ok (adjust port to what run_api.py prints).

- [ ] **Step 2: Verify the real folder via the API**

```bash
curl -s -X POST http://localhost:8000/api/v1/train/datasets/path \
  -H 'Content-Type: application/json' \
  -d '{"path": "E:\\Solar_gemini\\unisolar"}'
```

Record `dataset_id` from response. Expected: 42 sites, cadence 15, ~2.73M generation rows.

- [ ] **Step 3: Start a real XGBoost job and poll**

```bash
curl -s -X POST http://localhost:8000/api/v1/train/jobs \
  -H 'Content-Type: application/json' \
  -d '{"dataset_id": "<DATASET_ID>", "model": "xgboost"}'
# poll until status != running:
curl -s http://localhost:8000/api/v1/train/jobs/<JOB_ID> | python -c "import json,sys; d=json.load(sys.stdin); print(d['status'], d.get('error'))"
```

Expected: reaches `done` in ~1–3 min (verify+prepare dominate). While running, screenshot the Train page progress section → `shots/train-running.png`. After done, screenshot results → `shots/train-results.png`. Also capture the verify checklist BEFORE starting the job → `shots/train-verify.png`.

- [ ] **Step 4: Deep-model fast-test through the exit-9 path**

```bash
curl -s -X POST http://localhost:8000/api/v1/train/jobs \
  -H 'Content-Type: application/json' \
  -d '{"dataset_id": "<DATASET_ID>", "model": "transformer", "fast_test": true}'
```

Poll to completion. Expected: `status == "done"` even if the child exited 9 — this validates Task 2 live.

- [ ] **Step 5: Record measured numbers** — write `artifacts/train_live_run/summary.json` with the ACTUAL values from the two jobs (copy from `result.json` responses; never invent):

```json
{
  "xgboost_full": {"job_id": "...", "test_mae": ..., "test_rmse": ...,
                   "test_r2": ..., "fit_seconds": ..., "total_seconds": ...},
  "transformer_fast_test": {"job_id": "...", "status": "done",
                            "child_exit_code": ..., "total_seconds": ...},
  "note": "display-only runs (D-024); served v1 untouched"
}
```

- [ ] **Step 6: Sanity-check isolation** — `git status` must show NO changes under `models/` or `artifacts/` except the new `train_live_run/` dir; `data/train_jobs/` is gitignored.

```bash
git status --porcelain
```

- [ ] **Step 7: Commit**

```bash
git add artifacts/train_live_run docs/superpowers/plans/shots \
  && git commit -m "verify(train): first live end-to-end run — xgboost full + transformer fast-test"
```

---

### Task 12: Docs/tracker backfill + final gates

**Files:**
- Modify: `DECISIONS.md`, `TASKS.md`, `RESULTS.md`, `memories/project-state.md`

**Interfaces:**
- Consumes: measured numbers from Task 11's summary.json (copy verbatim — RESULTS.md Rule 4 forbids fabrication).

- [ ] **Step 1: DECISIONS.md** — append:

```markdown
## D-024 — Job-scoped display-only training (Train page) (2026-08-26)

POST /api/v1/train/jobs spawns `scripts/train_from_folder.py` per job; all
outputs land in `data/train_jobs/<dataset_id>/<job_id>/` (raw uploads,
staged parquet, features, model artifacts, result.json). Job state derives
from filesystem log markers (`== STAGE … start|done`, `== DONE`,
`== FAILED`) parsed by `src/api/train.py`; success is marker-based because
torch teardown on this box exits 9 after successful runs. One heavy job at
a time (409 otherwise). The served v1 models, phase artifacts, and recorded
RESULTS.md numbers are never modified — Train-page runs are display-only.

## D-025 — Post-PRD frontend graph additions (2026-08-26)

The PRD §38 pages are augmented with artifact-derived static bundles written
by `scripts/export_frontend_data.py` (D-020 pattern, no new REST endpoints):
`site_monthly` + `quality_extra` (earlier increment), then `eda_profiles`,
`missingness_timeline`, `evaluation_series`, `cross_site_summary`. Bundles
are regenerated after any artifact refresh and committed so a fresh clone
builds without a Python env. Pred-vs-actual overlays keep night gaps as
nulls (never bridged); baselines carry metrics only, series fields exist
solely where a predictions parquet exists.
```

- [ ] **Step 2: TASKS.md** — insert before `## Acceptance criteria`:

```markdown
## Post-PRD Enhancement — Train page + richer visuals (D-024/D-025)

- [x] Train page: folder verify (path/upload) → configure → job progress → results;
      job-scoped dirs, one heavy job at a time, served v1 untouched
- [x] Persistence fit-scope fix in train_from_folder (D-011 #3) + pipeline e2e tests
- [x] Marker-based job success (torch exit-9 teardown) + regression tests
- [x] Train API test coverage (verify endpoints, job lifecycle, artifacts, config)
- [x] Export bundles: eda_profiles, missingness_timeline, evaluation_series,
      cross_site_summary (+ unit tests)
- [x] Dashboard: hourly profile + weather↔power heatmap; Quality: monthly gap
      timeline; Comparison: pred-vs-actual explorer, all-runs bars, cross-site card
- [x] First live end-to-end training run verified (see RESULTS.md)
```

- [ ] **Step 3: RESULTS.md** — append a subsection using ONLY numbers from `artifacts/train_live_run/summary.json`:

```markdown
## Post-PRD: Train-page live verification (2026-08-26)

Display-only jobs against the unmodified UNISOLAR folder (D-024):

| run | model | test MAE | test RMSE | test R² | fit s | total s |
|-----|-------|----------|-----------|---------|-------|---------|
| <job_id> | xgboost (full) | … | … | … | … | … |
| <job_id> | transformer fast-test | — | — | — | … | … |

Fast-test numbers are smoke-test only. Suite grew 147 → <N> tests, all green.
```

- [ ] **Step 4: memory refresh** — update `memories/project-state.md`: change the "**Next:**" bullet to reflect Phase 15 remaining scope (leakage/CI audit still pending) and add a line noting D-024/D-025 shipped with the live-run pointer.

- [ ] **Step 5: Final gates**

```bash
conda run -n solar python -m pytest -q
cd frontend && npm run build && npx oxlint src
```

Expected: all tests pass; build green; oxlint 0 errors.

- [ ] **Step 6: Final commit**

```bash
cd /e/Solar_gemini && git add DECISIONS.md TASKS.md RESULTS.md memories \
  && git commit -m "docs: backfill D-024/D-025, trackers, live-run results; refresh project-state memory"
```

---

## Self-review notes (already applied)

- Spec coverage: Part 1 bundles → Tasks 6–7; page charts → Tasks 8–10; size budget → export-run checks in 6–7; Part 2 bug fixes → Tasks 2–3; API/pipeline tests → Tasks 3–5; live run → Task 11; docs → Task 12. Non-goals respected (no new endpoints, no nav changes).
- Baseline runs lack prediction parquets → `evaluation_series` emits metrics-only entries (Task 7 handles explicitly, tested).
- Type consistency: `_final_state` defined Task 2, reused implicitly by Task 5's lifecycle assertions; `EvalSeriesMetrics` shared by eval + cross-site types; builder signatures all take `root` for testability.
