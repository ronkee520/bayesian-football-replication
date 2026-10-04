"""Configuration loading and path resolution."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _project_path(value: str | Path, key: str) -> Path:
    """Resolve a path and require it to stay inside this self-contained project."""
    candidate = Path(value).expanduser()
    resolved = (candidate if candidate.is_absolute() else PROJECT_ROOT / candidate).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"{key} must point inside the project directory")
    return resolved


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration and resolve project-relative paths."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a mapping: {config_path}")

    for key in ("data_path", "output_dir"):
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
        config[key] = str(_project_path(config[key], key))

    if config.get("model") not in {"basic", "mixture"}:
        raise ValueError("model must be 'basic' or 'mixture'")
    if config.get("profile") not in {"paper_replication", "modernized"}:
        raise ValueError("profile must be 'paper_replication' or 'modernized'")
    return config


def save_resolved_config(config: dict[str, Any], output_dir: str | Path) -> Path:
    """Save a reproducible configuration without machine-specific absolute paths."""
    destination = Path(output_dir) / "config_resolved.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    portable = deepcopy(config)
    for key in ("data_path", "output_dir"):
        if key in portable:
            portable[key] = _project_path(portable[key], key).relative_to(PROJECT_ROOT).as_posix()
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(portable, handle, sort_keys=False, allow_unicode=True)
    return destination
