"""C3 compose-override mechanism (B4).

The frozen ``docker-compose.yml`` hardcodes the replicator lag
(``REPLICATOR_LAG_MS=0`` / ``--lag-ms 0``).  C3 requires the harness to
select 0 ms / 500 ms / unreplicated (-1) WITHOUT editing the frozen base
file.

This module generates a small compose override (single service:
``replicator``) that can be applied as:

    docker compose -f docker-compose.yml -f <override> up -d

Mechanism per C3 condition:
    lag=0    -> the existing replicator configuration (compose override
                still pinned to 0 for later C3 determinism).
    lag=500  -> override ``--lag-ms 500`` and ``REPLICATOR_LAG_MS=500``.
    lag=-1   -> override ``--lag-ms -1`` / ``REPLICATOR_LAG_MS=-1``,
                which uses the existing replicator's documented "never"
                behavior (the forwarder does not start).

The value is the CONFIGURED replication lag, never claimed to be a
measured delivery latency.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml


def generate_compose_override(lag_ms: int) -> str:
    """Return the YAML text of a compose override for a replicator lag.

    Raises:
        ValueError: if ``lag_ms`` is not one of {0, 500, -1}.
    """
    lag = int(lag_ms)
    if lag not in (0, 500, -1):
        raise ValueError(
            f"unsupported replication lag {lag}; C3 options are {[0, 500, -1]}"
        )
    data = {
        "services": {
            "replicator": {
                "command": [f"--lag-ms {lag}"],
                "environment": {
                    "REPLICATOR_LAG_MS": str(lag),
                },
            }
        }
    }
    return yaml.safe_dump(data, sort_keys=False)


def write_compose_override(lag_ms: int, out_path: str | Path) -> Path:
    """Write the C3 override for a lag to ``out_path`` and return it."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(generate_compose_override(lag_ms), encoding="utf-8")
    return p


def compose_command(lag_ms: int, override_path: str | Path) -> list[str]:
    """Return the ``docker compose`` argv that applies a C3 lag override.

    The base compose file is passed first and is never modified.
    """
    return [
        "docker", "compose",
        "-f", "docker-compose.yml",
        "-f", str(override_path),
    ]


def validate_lag_option(lag_ms: int) -> bool:
    """True when the lag is a valid C3 option."""
    try:
        generate_compose_override(lag_ms)
        return True
    except ValueError:
        return False


def _lag_from_override(text: str) -> int:
    """Parse the lag value back out of an override (for tests)."""
    data = yaml.safe_load(text)
    cmd = data["services"]["replicator"]["command"][0]
    m = re.search(r"--lag-ms (-?\d+)", cmd)
    if not m:
        raise ValueError(f"no --lag-ms in command {cmd!r}")
    return int(m.group(1))