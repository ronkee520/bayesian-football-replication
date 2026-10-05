"""Generate comparison tables and figures from completed model fits."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bayes_football.reporting import report_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(report_main())
