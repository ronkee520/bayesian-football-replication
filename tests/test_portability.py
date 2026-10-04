from __future__ import annotations

import re
from pathlib import Path

import yaml

from bayes_football.config import load_config, save_resolved_config

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".cff", ".csv", ".json", ".md", ".py", ".toml", ".yaml", ".yml"}
WINDOWS_ABSOLUTE = re.compile(r"(?<![A-Za-z])\b[A-Za-z]:[\\/]")
UNIX_USER_ABSOLUTE = re.compile(
    r"(?<![:A-Za-z0-9])/(?:Users|home|opt|tmp|var|mnt|workspace|root)/"
)
LOCAL_FILE_URL = re.compile(r"file:///(?:[A-Za-z]:|Users/|home/)", re.IGNORECASE)


def test_repository_and_saved_configs_are_portable() -> None:
    """Prevent machine-specific absolute paths from entering publishable files."""
    scan_roots = [
        ROOT / ".github",
        ROOT / "configs",
        ROOT / "data",
        ROOT / "docs",
        ROOT / "notebooks",
        ROOT / "results",
        ROOT / "scripts",
        ROOT / "src",
        ROOT / "tests",
    ]
    files = [
        ROOT / ".gitattributes",
        ROOT / ".gitignore",
        ROOT / "CITATION.cff",
        ROOT / "LICENSE",
        ROOT / "README.md",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    ]
    for scan_root in scan_roots:
        files.extend(
            path
            for path in scan_root.rglob("*")
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES
        )

    failures: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        if (
            WINDOWS_ABSOLUTE.search(text)
            or UNIX_USER_ABSOLUTE.search(text)
            or LOCAL_FILE_URL.search(text)
        ):
            failures.append(path.relative_to(ROOT).as_posix())
    assert not failures, f"Machine-specific absolute paths found in: {failures}"

    config = load_config(ROOT / "configs" / "quick_smoke.yaml")
    saved = save_resolved_config(config, ROOT / "data" / "processed")
    try:
        portable = yaml.safe_load(saved.read_text(encoding="utf-8"))
        assert portable["data_path"] == "data/raw/epl_2023_24.csv"
        assert portable["output_dir"] == "results/quick_smoke"
        assert str(ROOT) not in saved.read_text(encoding="utf-8")
    finally:
        saved.unlink(missing_ok=True)
