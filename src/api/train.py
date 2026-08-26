"""Training-job orchestration behind the Train page (post-PRD enhancement,
D-024/D-025).

One heavy job at a time. Each POST /train/jobs spawns
``scripts/train_from_folder.py`` as a subprocess writing everything into a
job-scoped directory under ``data/train_jobs/<dataset_id>/<job_id>/`` — the
served v1 models and phase artifacts are never touched. Job state is derived
from the filesystem (log markers + result.json), so the API process holds no
authoritative state beyond the child handle.

Endpoints:

* ``POST /train/datasets/path``      — verify a server-side folder
* ``POST /train/datasets/upload``    — save uploaded CSVs, verify them
* ``POST /train/jobs``               — start training a selected model
* ``GET  /train/jobs/{job_id}``      — status, stages, log tail, result
* ``GET  /train/jobs/{job_id}/artifacts/{name}`` — download artifacts
* ``GET  /train/config``             — hyperparameter snapshot for display
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
JOBS_ROOT = REPO_ROOT / "data" / "train_jobs"

REQUIRED_FILES = {
    "Solar_Energy_Generation.csv": ["SiteKey", "CampusKey", "Timestamp",
                                    "SolarGeneration"],
    "Weather_Data_reordered_all.csv": ["CampusKey", "Timestamp",
                                       "AirTemperature", "RelativeHumidity",
                                       "WindSpeed", "WindDirection"],
    "Solar_Site_Details.csv": ["SiteKey", "CampusKey", "kWp", "lat", "Lon"],
}
TRAINABLE_MODELS = ("xgboost", "lstm", "gru", "transformer")
MODEL_HINTS = {
    "xgboost": "~1 min on this machine (full dataset)",
    "lstm": "~10 min GPU / longer CPU (15 epochs)",
    "gru": "~10 min GPU / longer CPU (15 epochs)",
    "transformer": "~10-20 min GPU (15 epochs)",
}

STAGE_RE = re.compile(r"^== STAGE (\w+) (start|done)", re.MULTILINE)
FAILED_RE = re.compile(r"^== FAILED (.*)$", re.MULTILINE)
ARTIFACT_FILES = {"metrics.csv", "result.json", "predictions_test.parquet"}

router = APIRouter(prefix="/train", tags=["train"])


class PathDatasetRequest(BaseModel):
    path: str


class TrainJobRequest(BaseModel):
    dataset_id: str
    model: str
    fast_test: bool = False


class DatasetOut(BaseModel):
    dataset_id: str
    mode: str
    raw_dir: str
    files: list[dict]
    profile: dict


_jobs: dict[str, dict] = {}
_lock = threading.Lock()
_active_job_id: str | None = None


# ---- dataset verification ---------------------------------------------------

def _check_raw(raw: Path, saved_names: set[str] | None = None) -> list[dict]:
    """Per-file presence + header check against the verified UNISOLAR schema."""
    out = []
    bad = False
    for fname, req_cols in REQUIRED_FILES.items():
        p = raw / fname
        info: dict = {"name": fname, "ok": False, "rows": None, "detail": ""}
        if saved_names is not None and fname not in saved_names:
            info["detail"] = "not uploaded"
            bad = True
            out.append(info)
            continue
        if not p.exists():
            info["detail"] = "file not found"
            bad = True
            out.append(info)
            continue
        try:
            head = pd.read_csv(p, nrows=5)
            with p.open("rb") as fh:
                rows = sum(1 for _ in fh) - 1
        except Exception as exc:  # unreadable / binary
            info["detail"] = f"unreadable CSV: {exc}"
            bad = True
            out.append(info)
            continue
        missing = [c for c in req_cols if c not in head.columns]
        info["ok"] = not missing
        info["rows"] = max(0, int(rows))
        info["detail"] = ("columns ok" if not missing
                          else f"missing columns: {missing}")
        bad = bad or bool(missing)
        out.append(info)
    return out


def _quick_profile(raw: Path, files: list[dict]) -> dict:
    """Cheap profile for the verify response — generation CSV scan only
    (usecols), row counts reused from the header check. No cleaning."""
    gen = pd.read_csv(raw / "Solar_Energy_Generation.csv",
                      usecols=["SiteKey", "CampusKey", "Timestamp",
                               "SolarGeneration"])
    ts = pd.to_datetime(gen["Timestamp"], errors="coerce")
    cadence_min = None
    # within-site diffs only — cross-site same-timestamp rows would make
    # 0 s the modal delta on any multi-site folder
    ordered = gen.assign(_ts=ts).sort_values(["SiteKey", "_ts"])
    diffs = ordered.groupby("SiteKey", observed=True)["_ts"].diff().dropna()
    if len(diffs):
        mode_delta = diffs.mode().iloc[0]
        if mode_delta.total_seconds() > 0:
            cadence_min = int(mode_delta.total_seconds() // 60)
    wx_rows = next((f["rows"] for f in files
                    if f["name"] == "Weather_Data_reordered_all.csv"), None)
    return {
        "generation_rows": int(len(gen)),
        "weather_rows": wx_rows,
        "sites": int(gen["SiteKey"].nunique()),
        "campuses": int(gen["CampusKey"].nunique()),
        "start": str(ts.min()),
        "end": str(ts.max()),
        "cadence_minutes": cadence_min,
        "target_missing_pct": round(float(pd.to_numeric(
            gen["SolarGeneration"], errors="coerce").isna().mean() * 100), 2),
    }


def _register_dataset(mode: str, raw_dir: Path,
                      files: list[dict]) -> DatasetOut:
    dataset_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    ds_dir = JOBS_ROOT / dataset_id
    ds_dir.mkdir(parents=True, exist_ok=True)
    meta = {"dataset_id": dataset_id, "mode": mode,
            "raw_dir": str(raw_dir), "files": files,
            "profile": _quick_profile(raw_dir, files)}
    (ds_dir / "dataset.json").write_text(json.dumps(meta, indent=2),
                                         encoding="utf-8")
    return DatasetOut(**meta)


@router.post("/datasets/path")
def verify_path_dataset(req: PathDatasetRequest):
    raw = Path(req.path).expanduser().resolve()
    if not raw.is_dir():
        raise HTTPException(422, f"not a directory: {raw}")
    files = _check_raw(raw)
    if not all(f["ok"] for f in files):
        raise HTTPException(422, detail={"message": "folder failed verification",
                                         "files": files})
    return _register_dataset("path", raw, files)


@router.post("/datasets/upload")
async def verify_uploaded_dataset(files: list[UploadFile]):
    dataset_id = time.strftime("%Y%m%d-%H%M%S") + "-upload-" + uuid.uuid4().hex[:6]
    raw = JOBS_ROOT / dataset_id / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    allowed = set(REQUIRED_FILES)
    saved: set[str] = set()
    for f in files:
        name = Path(f.filename or "").name
        if name not in allowed:
            raise HTTPException(422, detail={
                "message": (f"unexpected file {name!r}; expected one of "
                            f"{sorted(allowed)}"),
                "files": []})
        dest = raw / name
        with dest.open("wb") as fh:
            while chunk := await f.read(1 << 20):
                fh.write(chunk)
        saved.add(name)
    checked = _check_raw(raw, saved_names=saved)
    if not all(c["ok"] for c in checked):
        raise HTTPException(422, detail={"message": "uploaded folder failed verification",
                                         "files": checked})
    return _register_dataset("upload", raw, checked)


# ---- jobs --------------------------------------------------------------------

def _load_result(job_dir: Path) -> dict | None:
    p = job_dir / "result.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"error": "result.json unreadable"}


def _parse_stages(log_text: str) -> tuple[list[dict], str]:
    """Stage list + current stage from `== STAGE <name> start|done` markers."""
    order: list[str] = []
    state: dict[str, str] = {}
    for line in log_text.splitlines():
        m = STAGE_RE.match(line.strip())
        if m:
            name, ev = m.group(1), m.group(2)
            if name not in state:
                order.append(name)
                state[name] = "start"
            state[name] = "done" if ev == "done" else "running"
    stages = [{"name": n, "status": state[n]} for n in order]
    current = next((n for n in order if state[n] != "done"), None)
    return stages, current or ""


def _final_state(returncode: int | None, text: str) -> dict:
    """Success = '== DONE' marker present and no '== FAILED' marker.

    Returncode is diagnostic only: torch scripts on this machine can exit
    code 9 AFTER a successful run (teardown crash — Phase 7 gotcha), so a
    nonzero exit alone must not flip a finished job to failed.
    """
    failed = FAILED_RE.search(text)
    ok = "== DONE" in text and failed is None
    return {
        "status": "done" if ok else "failed",
        "error": None if ok else (
            failed.group(1)[:300] if failed else f"exit code {returncode}"),
    }


def _watch(job_id: str) -> None:
    """Background watcher: follow the child until exit; derive final status."""
    global _active_job_id
    job = _jobs[job_id]
    proc, log_path = job["proc"], job["log_path"]
    seen = 0
    while proc.poll() is None:
        time.sleep(1.0)
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(text) > seen or seen == 0:
            seen = len(text)
            stages, current = _parse_stages(text)
            job["stages"], job["stage"] = stages, current

    text = log_path.read_text(encoding="utf-8", errors="replace") \
        if log_path.exists() else ""
    stages, current = _parse_stages(text)
    job.update({
        **_final_state(proc.returncode, text),
        "stages": stages,
        "stage": "",
        "finished_at": time.time(),
        "returncode": proc.returncode,
    })
    with _lock:
        if _active_job_id == job_id:
            _active_job_id = None


@router.post("/jobs")
def start_job(req: TrainJobRequest):
    global _active_job_id
    ds_dir = JOBS_ROOT / req.dataset_id
    meta_p = ds_dir / "dataset.json"
    if not meta_p.exists():
        raise HTTPException(404, f"unknown dataset {req.dataset_id!r} — verify it first")
    if req.model not in TRAINABLE_MODELS:
        raise HTTPException(422, f"model must be one of {list(TRAINABLE_MODELS)}")

    with _lock:
        if _active_job_id is not None and _jobs.get(_active_job_id, {}).get("proc") is not None \
                and _jobs[_active_job_id]["proc"].poll() is None:
            raise HTTPException(409, f"job {_active_job_id} still running — one at a time")
        _active_job_id = None

    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    job_id = time.strftime("%H%M%S") + "-" + req.model + "-" + uuid.uuid4().hex[:5]
    job_dir = ds_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    log_path = job_dir / "log.txt"

    cmd = [sys.executable, str(SCRIPTS_DIR / "train_from_folder.py"),
           "--dataset-dir", str(job_dir), "--raw", meta["raw_dir"],
           "--model", req.model]
    if req.fast_test:
        cmd.append("--fast-test")
    log_fh = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT,
                            cwd=str(REPO_ROOT))
    _jobs[job_id] = {
        "job_id": job_id, "dataset_id": req.dataset_id, "model": req.model,
        "dir": job_dir, "log_path": log_path, "proc": proc,
        "status": "running", "stages": [], "stage": "",
        "started_at": time.time(), "error": None, "returncode": None,
    }
    with _lock:
        _active_job_id = job_id
    threading.Thread(target=_watch, args=(job_id,), daemon=True).start()
    return {"job_id": job_id, "model": req.model,
            "hint": MODEL_HINTS[req.model], "status": "running"}


@router.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        # maybe from an earlier process run — reconstruct minimal view
        for p in JOBS_ROOT.glob(f"*/*/{job_id}/log.txt"):
            text = p.read_text(encoding="utf-8", errors="replace")
            stages, current = _parse_stages(text)
            done = "== DONE" in text
            return {"job_id": job_id,
                    "status": "done" if done else "unknown",
                    "stages": stages, "stage": "" if done else current,
                    "log_tail": text.splitlines()[-200:],
                    "result": _load_result(p.parent)}
        raise HTTPException(404, f"unknown job {job_id!r}")

    text = job["log_path"].read_text(encoding="utf-8", errors="replace") \
        if job["log_path"].exists() else ""
    out = {
        "job_id": job_id,
        "status": job["status"],
        "stages": job["stages"] or _parse_stages(text)[0],
        "stage": job["stage"] or _parse_stages(text)[1],
        "elapsed_s": round((job.get("finished_at") or time.time())
                           - job["started_at"], 1),
        "log_tail": text.splitlines()[-200:],
        "error": job["error"],
        "result": _load_result(job["dir"]),
    }
    return out


@router.get("/jobs/{job_id}/artifacts/{name}")
def job_artifact(job_id: str, name: str):
    if name not in ARTIFACT_FILES:
        raise HTTPException(422, f"artifact must be one of {sorted(ARTIFACT_FILES)}")
    candidates = []
    job = _jobs.get(job_id)
    if job is not None:
        candidates = sorted((job["dir"]).glob(f"artifacts/*/{name}"))
        if name == "result.json":
            candidates = [job["dir"] / "result.json"]
    else:
        candidates = [p for p in JOBS_ROOT.glob(f"*/*/{job_id}/artifacts/*/{name}")]
    media = {"metrics.csv": "text/csv",
             "predictions_test.parquet": "application/octet-stream",
             "result.json": "application/json"}[name]
    for p in candidates:
        if p.exists():
            return FileResponse(p, filename=f"{job_id}_{name}",
                                media_type=media)
    raise HTTPException(404, f"{name} not available for job {job_id}")


_config_cache: dict | None = None


@router.get("/config")
def train_config():
    global _config_cache
    if _config_cache is None:
        from src.config import load_config

        cfg = load_config()
        _config_cache = {
            "training": cfg["training"],
            "models": {k: v for k, v in cfg["models"].items()
                       if k in TRAINABLE_MODELS},
        }
    return _config_cache
