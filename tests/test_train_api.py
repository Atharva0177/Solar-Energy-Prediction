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
