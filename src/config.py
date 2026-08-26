"""YAML configuration loader (PRD §43: nothing hard-coded)."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


def load_config(config_dir: Path = CONFIG_DIR) -> dict:
    """Load ``models.yaml`` + ``training.yaml`` into one flat dict.

    Returns ``{"models": {...}, "training": {...}, "forecast": {...},
    "paths": {...}}`` — i.e. ``cfg["training"]["seed"]``.
    """
    models = yaml.safe_load((config_dir / "models.yaml").read_text(encoding="utf-8")) or {}
    training = yaml.safe_load((config_dir / "training.yaml").read_text(encoding="utf-8")) or {}
    return {"models": models.get("models", {}), **training}
