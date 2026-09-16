"""
Three idempotency key schemes from the Kinbridge-Sync paper.

k_payload  = H(tool || canonical(args))
k_bucket   = H(agent_id || action_type || payload || floor(t / Delta))
k_intent   = H(session_id || turn_seq || intent_id)

All keys are hex-encoded SHA-256 hashes.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def _sha256_hex(data: str) -> str:
    """Return the hex-encoded SHA-256 hash of a UTF-8 string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def canonical_serialise(args: dict[str, Any]) -> str:
    """Produce a deterministic JSON serialisation of tool arguments.

    Sorts keys recursively so that equivalent dicts with different
    insertion orders produce identical serialisations.
    """
    return json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def action_fingerprint(tool: str, args: dict[str, Any]) -> str:
    """Return the canonical payload string for a tool call: tool||canonical(args)."""
    return tool + "||" + canonical_serialise(args)


# ---- k_payload ---------------------------------------------------------------

def k_payload(tool: str, args: dict[str, Any]) -> str:
    """Payload key: H(tool || canonical(args))."""
    return _sha256_hex(action_fingerprint(tool, args))


# ---- k_bucket ----------------------------------------------------------------

def k_bucket(
    agent_id: str,
    action_type: str,
    args: dict[str, Any],
    timestamp_s: float,
    delta_s: float = 60.0,
) -> str:
    """Timestamp-bucket key: H(agent_id || action_type || payload || floor(t/Delta)).

    Args:
        agent_id:    Stable agent identifier.
        action_type: Tool name.
        args:        Tool arguments.
        timestamp_s: Wall-clock time in seconds (float).
        delta_s:     Bucket width in seconds (default 60).
    """
    bucket = math.floor(timestamp_s / delta_s)
    payload = action_fingerprint(action_type, args)
    raw = f"{agent_id}||{action_type}||{payload}||{bucket}"
    return _sha256_hex(raw)


def k_bucket_from_fingerprint(
    agent_id: str,
    action_type: str,
    fingerprint: str,
    timestamp_s: float,
    delta_s: float = 60.0,
) -> str:
    """Bucket key using a pre-computed payload fingerprint (avoids re-serialisation)."""
    bucket = math.floor(timestamp_s / delta_s)
    raw = f"{agent_id}||{action_type}||{fingerprint}||{bucket}"
    return _sha256_hex(raw)


# ---- k_intent ----------------------------------------------------------------

def k_intent(
    session_id: str,
    turn_seq: int,
    intent_id: str,
) -> str:
    """Intent key: H(session_id || turn_seq || intent_id).

    The intent_id comes from the ground-truth dataset — the LLM never
    creates or modifies it.  This key therefore always matches for the
    same ground-truth intent regardless of how the model rephrases.
    """
    raw = f"{session_id}||{turn_seq}||{intent_id}"
    return _sha256_hex(raw)


# ---- Bucket-boundary helper ---------------------------------------------------

def bucket_index(timestamp_s: float, delta_s: float = 60.0) -> int:
    """Return floor(t / Delta) — useful for bucket-boundary tests."""
    return math.floor(timestamp_s / delta_s)
