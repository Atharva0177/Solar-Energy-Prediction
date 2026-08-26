"""Launch the UNISOLAR REST API (PRD §33).

Usage: ``conda run -n solar python scripts/run_api.py [--host 127.0.0.1]
[--port 8000] [--reload]``

Serves ``/api/v1`` from on-disk artifacts; interactive docs at ``/docs``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.api.app import create_app
from src.api.store import ParquetStore


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()

    app = create_app(ParquetStore(REPO_ROOT))
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
