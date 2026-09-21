"""FastAPI tool world application for Phase 2.1.

Exposes the three Kinbridge-Sync tool actions as POST endpoints with
SQLite-backed idempotency via the effect ledger.  Each request flows:

  validate → canonicalise → compute key → check ledger → execute → commit

Fault injection is available at the request level for testing.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from tool_world.config import get_db_path, get_server_config
from tool_world.ledger import EffectLedger
from tool_world.models import (
    EffectStatus,
    LedgerLookupResponse,
    LogInspectionRequest,
    NavigateToRequest,
    RequestSupplyDropRequest,
    ToolResponse,
)

# ---------------------------------------------------------------------------
# Reuse existing pilot tool definitions (read-only import)
# ---------------------------------------------------------------------------
from pilot.tools import TOOL_FUNCTIONS  # noqa: E402

# ---------------------------------------------------------------------------
# Global ledger instance (initialised at startup)
# ---------------------------------------------------------------------------

_ledger: EffectLedger | None = None


def get_ledger() -> EffectLedger:
    """Return the global ledger instance (for testing access)."""
    assert _ledger is not None, "Ledger not initialised"
    return _ledger


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise and tear down the effect ledger."""
    global _ledger
    _ledger = EffectLedger(db_path=str(get_db_path()))
    yield
    if _ledger is not None:
        _ledger.close()
        _ledger = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Kinbridge-Sync Tool World",
    description="Phase 2.1 — FastAPI tool endpoints + SQLite effect ledger",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _canonical_args(args: dict[str, Any]) -> str:
    """Deterministic JSON serialisation of tool arguments."""
    return json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _args_hash(args: dict[str, Any]) -> str:
    """SHA-256 hex digest of canonical args."""
    return hashlib.sha256(_canonical_args(args).encode("utf-8")).hexdigest()


def _compute_key(
    tool: str,
    args: dict[str, Any],
    intent_id: str | None = None,
    session_id: str | None = None,
    turn_seq: int | None = None,
) -> str:
    """Compute the effect idempotency key.

    Prefers intent key when intent_id + session_id + turn_seq are supplied.
    Falls back to payload key.
    """
    if intent_id and session_id and turn_seq is not None:
        from pilot.key_schemes import k_intent
        return k_intent(session_id, turn_seq, intent_id)
    from pilot.key_schemes import k_payload
    return k_payload(tool, args)


def _execute_tool(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    """Execute a tool action via the existing pilot function stubs.

    In Phase 2.1 this is a local function call.  Later phases may
    route through Docker/HTTP to the actual tool world service.
    """
    func = TOOL_FUNCTIONS.get(tool)
    if func is None:
        raise ValueError(f"Unknown tool: {tool}")
    return func(**args)


# ---------------------------------------------------------------------------
# Tool endpoints
# ---------------------------------------------------------------------------

@app.post("/tools/navigate_to", response_model=ToolResponse)
def navigate_to(req: NavigateToRequest) -> ToolResponse:
    """Navigate the agent to a geographic location."""
    args = {"lat": req.lat, "lon": req.lon, "mode": req.mode}
    return _handle_tool("navigate_to", args, req)


@app.post("/tools/log_inspection", response_model=ToolResponse)
def log_inspection(req: LogInspectionRequest) -> ToolResponse:
    """Log the result of a site inspection."""
    args = {"site_id": req.site_id, "status": req.status, "notes": req.notes}
    return _handle_tool("log_inspection", args, req)


@app.post("/tools/request_supply_drop", response_model=ToolResponse)
def request_supply_drop(req: RequestSupplyDropRequest) -> ToolResponse:
    """Request a supply drop at a location."""
    args = {
        "lat": req.lat,
        "lon": req.lon,
        "payload": req.payload,
        "priority": req.priority,
    }
    return _handle_tool("request_supply_drop", args, req)


# ---------------------------------------------------------------------------
# Core handler (shared logic for all tools)
# ---------------------------------------------------------------------------

def _handle_tool(
    tool: str,
    args: dict[str, Any],
    req: NavigateToRequest | LogInspectionRequest | RequestSupplyDropRequest,
) -> ToolResponse:
    """Shared request→ledger→execute→commit flow."""
    ledger = get_ledger()

    # 1. Compute effect key and args hash
    intent_id = getattr(req, "intent_id", None)
    session_id = getattr(req, "session_id", None)
    turn_seq = getattr(req, "turn_seq", None)
    key = _compute_key(tool, args, intent_id, session_id, turn_seq)
    a_hash = _args_hash(args)

    # 2. Check ledger for existing record
    existing = ledger.lookup(key)
    if existing is not None and existing["status"] == EffectStatus.COMMITTED.value:
        return ToolResponse(
            tool=tool,
            result=existing,
            key=key,
            status=EffectStatus.COMMITTED,
            committed=False,
            epoch=existing["epoch"],
            fault_injected=False,
        )

    # 3. Create ledger record (if new)
    if existing is None:
        epoch = ledger.increment_epoch()
        record = ledger.create(
            key=key,
            tool=tool,
            args_hash=a_hash,
            ttl=0.0,
            intent_id=intent_id,
        )
        # Set the epoch on the newly created record
        ledger.set_epoch(key, epoch)
    else:
        epoch = existing["epoch"]
        record = existing

    # 4. Fault injection — force execution failure
    fault_injected = getattr(req, "fault_inject", False)
    if fault_injected:
        ledger.mark_unknown(key)
        return ToolResponse(
            tool=tool,
            result={"error": "fault_injected"},
            key=key,
            status=EffectStatus.UNKNOWN,
            committed=False,
            epoch=epoch,
            fault_injected=True,
        )

    # 5. Execute tool action
    try:
        result = _execute_tool(tool, args)
    except Exception as exc:
        ledger.mark_unknown(key)
        return ToolResponse(
            tool=tool,
            result={"error": str(exc)},
            key=key,
            status=EffectStatus.UNKNOWN,
            committed=False,
            epoch=epoch,
            fault_injected=False,
        )

    # 6. Mark committed
    ledger.mark_committed(key)

    return ToolResponse(
        tool=tool,
        result=result,
        key=key,
        status=EffectStatus.COMMITTED,
        committed=True,
        epoch=epoch,
        fault_injected=False,
    )


# ---------------------------------------------------------------------------
# Ledger lookup endpoint
# ---------------------------------------------------------------------------

@app.get("/ledger/{key}", response_model=LedgerLookupResponse)
def ledger_lookup(key: str) -> LedgerLookupResponse:
    """Look up an effect record by key."""
    ledger = get_ledger()
    record = ledger.lookup(key)
    if record is None:
        raise HTTPException(status_code=404, detail="Effect not found")
    return LedgerLookupResponse(
        key=record["key"],
        tool=record["tool"],
        args_hash=record["args_hash"],
        status=EffectStatus(record["status"]),
        created_at=record["created_at"],
        committed_at=record["committed_at"],
        epoch=record["epoch"],
        ttl=record["ttl"],
        intent_id=record["intent_id"],
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
