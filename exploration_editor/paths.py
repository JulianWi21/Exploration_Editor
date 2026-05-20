from __future__ import annotations

from pathlib import Path


EXPLORATION_ROOT = Path(r"C:\Projekte\YouTube\GEOPANDA\exploration")


def exploration_examples_dir() -> Path:
    return EXPLORATION_ROOT / "examples"


def exploration_exports_dir() -> Path:
    return EXPLORATION_ROOT / "exports"