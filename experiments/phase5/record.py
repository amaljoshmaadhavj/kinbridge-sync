"""Episode record building and validation (§11).

Each episode produces exactly one machine-readable JSON record with the
documented fields.  ``validate_episode_record`` fails loudly on
malformed records rather than silently dropping trials; the analyzer
never silently omits failed trials.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

# Top-level episode metadata fields required by the research spec.
REQUIRED_TOP_FIELDS = (
    "episode_id",
    "method",
    "outage_profile",
    "outage_duration",
    "seed",
    "episode_seed",
    "temperature",
    "key_scheme",
    "replication_lag",
    "topology",
    "source_node",
    "reconnect_node",
    "epoch_before",
    "epoch_after",
    "epoch_guard_result",
    "actions",
    "ttr_seconds",
    "tau_star",
    "tau_star_source",
    "scenario_id",
)

# Per-action fields.
REQUIRED_ACTION_FIELDS = (
    "action_id",
    "tool",
    "args",
    "intent_id",
    "session_id",
    "turn_seq",
    "ttl",
    "effect_identity",
    "ledger_status",
    "decision",
    "duplicate",
    "stale_execution",
    "escalated",
    "invalidated",
    "committed",
    "v_pre_outcome",
    "v_post_outcome",
    "absent_after_epoch_guard",
    "replication_ambiguity",
    "safety_decision",
)

REQUIRED_DECISION_FIELDS = (
    "decision",
    "duplicate",
    "stale_execution",
    "escalated",
    "invalidated",
    "committed",
)


def _has_all(record: dict[str, Any], fields: tuple[str, ...], label: str) -> None:
    missing = [f for f in fields if f not in record]
    if missing:
        raise ValueError(f"{label} record missing fields: {sorted(missing)}")


class RecordValidationError(ValueError):
    """Raised when an episode record is malformed."""


def validate_episode_record(record: dict[str, Any]) -> dict[str, Any]:
    """Validate one episode record; returns it unchanged on success."""
    _has_all(record, REQUIRED_TOP_FIELDS, "episode")
    if isinstance(record["actions"], list):
        for action in record["actions"]:
            _has_all(action, REQUIRED_ACTION_FIELDS, "action")

    _validate_flag(record, "epoch_guard_result")
    if not _is_number(record["ttr_seconds"]) or record["ttr_seconds"] < 0:
        raise RecordValidationError(
            f"non-negative numeric ttr_seconds required, got {record['ttr_seconds']!r}"
        )
    for f in ("episode_id",):
        if not isinstance(record[f], int):
            raise RecordValidationError(f"{f} must be an int")

    for action in record["actions"]:
        if not isinstance(action, dict):
            raise RecordValidationError("action entry must be a dict")
        for flag in ("duplicate", "stale_execution", "escalated", "invalidated",
                     "committed", "absent_after_epoch_guard", "replication_ambiguity"):
            _validate_flag(action, flag)
        if not _is_number(action["ttl"]):
            raise RecordValidationError(
                f"action {action.get('action_id')!r}: ttl must be numeric"
            )

    return record


def _validate_flag(record: dict[str, Any], field: str) -> None:
    if not isinstance(record[field], bool):
        raise RecordValidationError(f"{field} must be a bool, got {record[field]!r}")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def write_episode_records(records: list[dict[str, Any]], path: str | Path) -> Path:
    """Append records as JSONL (append-only experiment log)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        for rec in records:
            validate_episode_record(rec)
            f.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")
    return p


def read_episode_records(path: str | Path) -> list[dict[str, Any]]:
    """Read and validate a JSONL episode-record file.

    Raises:
        RecordValidationError: if any line is malformed (fails loudly).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"episode records not found: {p}")
    records = []
    with open(p, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RecordValidationError(
                    f"malformed JSON at {p}:{lineno}: {exc}"
                ) from exc
            try:
                validate_episode_record(rec)
            except (ValueError, RecordValidationError) as exc:
                raise RecordValidationError(
                    f"invalid episode record at {p}:{lineno}: {exc}"
                ) from exc
            records.append(rec)
    return records