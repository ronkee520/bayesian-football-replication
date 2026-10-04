"""Validate the tracked EPL data and write the on-field league table."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bayes_football.cli import validate_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(validate_main())
