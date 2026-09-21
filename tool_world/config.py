"""Phase 2.1 configuration for the FastAPI tool world.

Sensible defaults so the tool world can be tested locally without
additional setup.
"""

from __future__ import annotations

import os
from pathlib import Path

# Project root (same convention as core/config.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# SQLite database path — default to project root for local development
DEFAULT_DB_PATH = PROJECT_ROOT / "tool_world" / "effects.db"

# Server defaults
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


def get_db_path() -> Path:
    """Return the SQLite database path from env or default."""
    env_path = os.environ.get("KINBRIDGE_DB_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_DB_PATH


def get_server_config() -> dict[str, str | int]:
    """Return server host/port from env or defaults."""
    return {
        "host": os.environ.get("KINBRIDGE_HOST", DEFAULT_HOST),
        "port": int(os.environ.get("KINBRIDGE_PORT", str(DEFAULT_PORT))),
    }
