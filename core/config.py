"""
Configuration loader for Kinbridge-Sync experiments.

Loads YAML config files and provides a typed accessor interface.
All experiment parameters live in config/ — never hard-coded.

Usage:
    from core.config import load_config
    cfg = load_config("pilot/pilot_config.yaml")
    print(cfg["ollama"]["model_large"])
"""

import os
from pathlib import Path
from typing import Any

import yaml

# Project root is always resolved relative to this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


def load_config(relative_path: str, base_dir: Path | None = None) -> dict[str, Any]:
    """Load a YAML config file and return its contents as a dict.

    Args:
        relative_path: Path relative to the config/ directory,
                       e.g. "pilot/pilot_config.yaml".
        base_dir: Override the base config directory (for testing).

    Returns:
        Parsed configuration dictionary.
    """
    if base_dir is None:
        base_dir = CONFIG_DIR
    path = base_dir / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def results_dir() -> Path:
    """Return the canonical results directory, creating it if needed."""
    d = PROJECT_ROOT / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def experiment_output_path(name: str, ext: str = "csv") -> Path:
    """Return a full path under results/ for an experiment output file.

    Args:
        name: Stem of the filename (e.g. "pilot_results").
        ext: File extension without dot (e.g. "csv", "json").

    Returns:
        Absolute Path for the output file.
    """
    return results_dir() / f"{name}.{ext}"
