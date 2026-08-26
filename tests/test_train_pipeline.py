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

    # FULL-table persistence (D-011 #3): the ONLY allowed missing eval rows are
    # night slots whose target itself is NaN (missing ≠ zero). Any more means
    # the baseline was fit train-only and its lookups starved.
    for split in ("val", "test"):
        info = res["split"][split]
        night = info["rows"] - info["observed_rows"]
        got = res["persistence"][f"{split}_all"]["n_missing"]
        assert got == night, (
            f"{split}: persistence n_missing {got} != night rows {night} "
            "— fit-scope or lookup bug")

    # model artifacts confined to the job dir
    assert (job / "artifacts" / model / "metrics.csv").exists()
    assert (job / "artifacts" / model / "predictions_test.parquet").exists()

    # metrics grid: val+test × ALL+SITE rows present
    mps = res["metrics_per_site"]
    scopes = {(r["split"], r["scope"]) for r in mps}
    assert {("test", "ALL"), ("test", "SITE"), ("val", "ALL")} <= scopes
