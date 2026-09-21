"""Pydantic v2 request/response models for the Kinbridge-Sync tool world.

Each tool has a request model and a response model.  The response models
include ledger metadata so that tests and the future Reintegration Service
can inspect the effect status.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EffectStatus(str, Enum):
    """Ledger status for an effect record."""
    ABSENT = "ABSENT"
    COMMITTED = "COMMITTED"
    UNKNOWN = "UNKNOWN"
    ESCALATED = "ESCALATED"
    INVALIDATED = "INVALIDATED"


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class NavigateToRequest(BaseModel):
    """Request to navigate the agent to a geographic location."""
    lat: float = Field(..., description="Latitude in decimal degrees")
    lon: float = Field(..., description="Longitude in decimal degrees")
    mode: str = Field(
        ...,
        description="Routing mode",
        pattern="^(fastest|shortest|standard|autonomous)$",
    )
    intent_id: str | None = Field(
        None, description="Optional explicit intent ID for the intent key"
    )
    session_id: str | None = Field(None, description="Session ID for intent key")
    turn_seq: int | None = Field(None, description="Turn sequence for intent key")
    fault_inject: bool = Field(
        False, description="Force execution failure for testing"
    )


class LogInspectionRequest(BaseModel):
    """Request to log the result of a site inspection."""
    site_id: str = Field(..., description="Unique site identifier")
    status: str = Field(
        ...,
        description="Inspection outcome",
        pattern="^(passed|warning|failed)$",
    )
    notes: str = Field(..., description="Free-text observation notes")
    intent_id: str | None = Field(
        None, description="Optional explicit intent ID for the intent key"
    )
    session_id: str | None = Field(None, description="Session ID for intent key")
    turn_seq: int | None = Field(None, description="Turn sequence for intent key")
    fault_inject: bool = Field(
        False, description="Force execution failure for testing"
    )


class RequestSupplyDropRequest(BaseModel):
    """Request to request a supply drop at a location."""
    lat: float = Field(..., description="Latitude in decimal degrees")
    lon: float = Field(..., description="Longitude in decimal degrees")
    payload: str = Field(..., description="Description of the supply payload")
    priority: str = Field(
        ...,
        description="Dispatch priority",
        pattern="^(high|medium|low)$",
    )
    intent_id: str | None = Field(
        None, description="Optional explicit intent ID for the intent key"
    )
    session_id: str | None = Field(None, description="Session ID for intent key")
    turn_seq: int | None = Field(None, description="Turn sequence for intent key")
    fault_inject: bool = Field(
        False, description="Force execution failure for testing"
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class ToolResponse(BaseModel):
    """Generic response from a tool execution, including ledger metadata."""
    tool: str = Field(..., description="Tool name that was executed")
    result: dict[str, Any] = Field(..., description="Tool execution result")
    key: str = Field(..., description="Effect idempotency key used")
    status: EffectStatus = Field(..., description="Ledger status after execution")
    committed: bool = Field(
        ..., description="True if this execution committed a new effect"
    )
    epoch: int = Field(..., description="Ledger epoch at execution time")
    fault_injected: bool = Field(
        False, description="True if execution was forced to fail"
    )


class LedgerLookupResponse(BaseModel):
    """Response from a ledger lookup."""
    key: str
    tool: str
    args_hash: str
    status: EffectStatus
    created_at: float
    committed_at: float | None
    epoch: int
    ttl: float
    intent_id: str | None
