"""
Tool definitions for the Kinbridge-Sync pilot study.

Three tool classes with JSON-schema signatures compatible with Ollama's
tool-calling API.  Each tool is represented as:
  - a Python function (for type hints and documentation)
  - a JSON-schema dict  (for the Ollama /api/chat tools parameter)

The LLM is asked to call one of these tools for every prompt.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# JSON-schema definitions (Ollama tool-calling format)
# ---------------------------------------------------------------------------

NAVIGATE_TO_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "navigate_to",
        "description": "Navigate the agent to a geographic location.",
        "parameters": {
            "type": "object",
            "properties": {
                "lat": {
                    "type": "number",
                    "description": "Latitude in decimal degrees",
                },
                "lon": {
                    "type": "number",
                    "description": "Longitude in decimal degrees",
                },
                "mode": {
                    "type": "string",
                    "enum": ["fastest", "shortest", "standard", "autonomous"],
                    "description": "Routing mode",
                },
            },
            "required": ["lat", "lon", "mode"],
        },
    },
}

LOG_INSPECTION_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "log_inspection",
        "description": "Log the result of a site inspection.",
        "parameters": {
            "type": "object",
            "properties": {
                "site_id": {
                    "type": "string",
                    "description": "Unique site identifier",
                },
                "status": {
                    "type": "string",
                    "enum": ["passed", "warning", "failed"],
                    "description": "Inspection outcome",
                },
                "notes": {
                    "type": "string",
                    "description": "Free-text observation notes",
                },
            },
            "required": ["site_id", "status", "notes"],
        },
    },
}

REQUEST_SUPPLY_DROP_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "request_supply_drop",
        "description": "Request a supply drop at a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "lat": {
                    "type": "number",
                    "description": "Latitude in decimal degrees",
                },
                "lon": {
                    "type": "number",
                    "description": "Longitude in decimal degrees",
                },
                "payload": {
                    "type": "string",
                    "description": "Description of the supply payload",
                },
                "priority": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "Dispatch priority",
                },
            },
            "required": ["lat", "lon", "payload", "priority"],
        },
    },
}

# All tool schemas indexed by tool name
TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "navigate_to": NAVIGATE_TO_SCHEMA,
    "log_inspection": LOG_INSPECTION_SCHEMA,
    "request_supply_drop": REQUEST_SUPPLY_DROP_SCHEMA,
}

# Ordered list for the Ollama API call
TOOL_LIST: list[dict[str, Any]] = [
    NAVIGATE_TO_SCHEMA,
    LOG_INSPECTION_SCHEMA,
    REQUEST_SUPPLY_DROP_SCHEMA,
]


# ---------------------------------------------------------------------------
# Python function stubs (documentation / type-checking only)
# ---------------------------------------------------------------------------

def navigate_to(lat: float, lon: float, mode: str) -> dict[str, Any]:
    """Navigate the agent to a geographic location."""
    return {"lat": lat, "lon": lon, "mode": mode}


def log_inspection(site_id: str, status: str, notes: str) -> dict[str, Any]:
    """Log the result of a site inspection."""
    return {"site_id": site_id, "status": status, "notes": notes}


def request_supply_drop(
    lat: float, lon: float, payload: str, priority: str
) -> dict[str, Any]:
    """Request a supply drop at a location."""
    return {"lat": lat, "lon": lon, "payload": payload, "priority": priority}


# Map from tool name to Python function
TOOL_FUNCTIONS: dict[str, Any] = {
    "navigate_to": navigate_to,
    "log_inspection": log_inspection,
    "request_supply_drop": request_supply_drop,
}
