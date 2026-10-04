"""Run a chronological final-round holdout experiment."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bayes_football.cli import holdout_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(holdout_main())
