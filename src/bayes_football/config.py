"""Configuration loading and path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
        candidate = Path(config[key])
        config[key] = str(candidate if candidate.is_absolute() else PROJECT_ROOT / candidate)

    if config.get("model") not in {"basic", "mixture"}:
        raise ValueError("model must be 'basic' or 'mixture'")
    if config.get("profile") not in {"paper_replication", "modernized"}:
        raise ValueError("profile must be 'paper_replication' or 'modernized'")
    return config


def save_resolved_config(config: dict[str, Any], output_dir: str | Path) -> Path:
    """Save the exact configuration used by a model run."""
    destination = Path(output_dir) / "config_resolved.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    return destination
