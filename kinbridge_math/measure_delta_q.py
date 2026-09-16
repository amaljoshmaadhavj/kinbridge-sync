"""
Measure Delta-q — quality gap between qwen2.5:7b-instruct and qwen2.5:1.5b-instruct.

Phase 16F: The degradation-gate equation requires a measured quality gap
between the large and small models.  This module runs both models on a
held-out evaluation set, scores each response, and reports the gap.

Quality metric (operational definition):
  For each evaluation task, the model must call the correct tool with
  plausible arguments.  Score is 1.0 (correct tool + plausible args),
  0.5 (correct tool but implausible args), or 0.0 (wrong tool / no call).

Note: this metric is an operational choice documented here explicitly,
because the paper does not fully specify the quality scoring rubric.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from core.config import load_config
from pilot.tools import TOOL_LIST

# ---------------------------------------------------------------------------
# Held-out evaluation set (independent of Phase 1 pilot prompts)
# ---------------------------------------------------------------------------
# These are 8 additional prompts NOT in prompts.json, covering all three
# tool classes, used only for Delta-q measurement.

EVAL_SET: list[dict[str, Any]] = [
    {
        "task_id": "eval_nav_01",
        "tool_class": "navigation",
        "expected_tool": "navigate_to",
        "prompt": "Set a course for 52.3676, 4.9041 at standard speed.",
    },
    {
        "task_id": "eval_nav_02",
        "tool_class": "navigation",
        "expected_tool": "navigate_to",
        "prompt": "Move the unit to latitude 1.3521, longitude 103.8198.",
    },
    {
        "task_id": "eval_insp_01",
        "tool_class": "inspection",
        "expected_tool": "log_inspection",
        "prompt": "Record inspection for site kappa-4: passed, all readings normal.",
    },
    {
        "task_id": "eval_insp_02",
        "tool_class": "inspection",
        "expected_tool": "log_inspection",
        "prompt": "Log inspection data at site lambda-2, status warning, thermal anomaly.",
    },
    {
        "task_id": "eval_sdrop_01",
        "tool_class": "supply_drop",
        "expected_tool": "request_supply_drop",
        "prompt": "Request supply drop at 34.0522, -118.2437, payload: water, priority high.",
    },
    {
        "task_id": "eval_sdrop_02",
        "tool_class": "supply_drop",
        "expected_tool": "request_supply_drop",
        "prompt": "Dispatch fuel to 55.7558, 37.6173, priority medium.",
    },
    {
        "task_id": "eval_nav_03",
        "tool_class": "navigation",
        "expected_tool": "navigate_to",
        "prompt": "Go to coordinates -34.6037, -58.3816 using the fastest route.",
    },
    {
        "task_id": "eval_insp_03",
        "tool_class": "inspection",
        "expected_tool": "log_inspection",
        "prompt": "Write inspection for mu-6: failed, sensor malfunction detected.",
    },
]


@dataclass
class EvalResult:
    """Result of evaluating a single task on a single model."""
    task_id: str
    tool_class: str
    expected_tool: str
    model: str
    prompt: str
    response: dict[str, Any]
    extracted_tool: str
    extracted_args: dict[str, Any]
    score: float          # 0.0, 0.5, or 1.0
    success: bool         # whether the model produced a tool call
    timestamp_iso: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "tool_class": self.tool_class,
            "expected_tool": self.expected_tool,
            "model": self.model,
            "prompt": self.prompt,
            "extracted_tool": self.extracted_tool,
            "extracted_args": self.extracted_args,
            "score": self.score,
            "success": self.success,
            "timestamp_iso": self.timestamp_iso,
        }


def _extract_tool_call(response: dict[str, Any]) -> tuple[str, dict, bool]:
    """Extract tool call from Ollama response. Returns (tool, args, success)."""
    message = response.get("message", {})
    tool_calls = message.get("tool_calls", [])
    if not tool_calls:
        return "", {}, False
    tc = tool_calls[0].get("function", {})
    return tc.get("name", ""), tc.get("arguments", {}), True


def _score_response(
    expected_tool: str,
    extracted_tool: str,
    extracted_args: dict[str, Any],
    success: bool,
) -> float:
    """Score a single response.

    1.0 = correct tool + non-empty plausible args
    0.5 = correct tool but empty or clearly wrong args
    0.0 = wrong tool or no tool call
    """
    if not success or extracted_tool != expected_tool:
        return 0.0

    # Basic plausibility: args must be non-empty and contain expected keys
    if not extracted_args:
        return 0.5

    # Check that required keys are present (basic plausibility)
    required_keys = {
        "navigate_to": {"lat", "lon", "mode"},
        "log_inspection": {"site_id", "status", "notes"},
        "request_supply_drop": {"lat", "lon", "payload", "priority"},
    }
    expected_keys = required_keys.get(expected_tool, set())
    provided_keys = set(extracted_args.keys())

    if expected_keys.issubset(provided_keys):
        return 1.0
    elif provided_keys:
        return 0.5
    else:
        return 0.0


def measure_delta_q(
    base_url: str | None = None,
    model_large: str | None = None,
    model_small: str | None = None,
    temperature: float = 0.0,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run both models on the held-out eval set and compute Delta-q.

    Returns a dict with per-task results and aggregate quality scores.
    """
    cfg = load_config("pilot/pilot_config.yaml")
    base_url = base_url or cfg["ollama"]["base_url"]
    model_large = model_large or cfg["ollama"]["model_large"]
    model_small = model_small or cfg["ollama"]["model_small"]

    results_large: list[EvalResult] = []
    results_small: list[EvalResult] = []

    for task in EVAL_SET:
        for model, results_list in [(model_large, results_large), (model_small, results_small)]:
            messages = [
                {"role": "system", "content": "You are a field operations agent. Use the provided tools to fulfill the user's request."},
                {"role": "user", "content": task["prompt"]},
            ]
            try:
                url = f"{base_url}/api/chat"
                payload = {
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "tools": TOOL_LIST,
                    "options": {"temperature": temperature},
                }
                resp = requests.post(url, json=payload, timeout=120)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                data = {"error": str(exc)}

            tool, args, success = _extract_tool_call(data)
            score = _score_response(task["expected_tool"], tool, args, success)

            results_list.append(EvalResult(
                task_id=task["task_id"],
                tool_class=task["tool_class"],
                expected_tool=task["expected_tool"],
                model=model,
                prompt=task["prompt"],
                response=data,
                extracted_tool=tool,
                extracted_args=args,
                score=score,
                success=success,
                timestamp_iso=datetime.now(timezone.utc).isoformat(),
            ))

    # Aggregate
    scores_large = [r.score for r in results_large]
    scores_small = [r.score for r in results_small]
    mean_large = sum(scores_large) / len(scores_large) if scores_large else 0.0
    mean_small = sum(scores_small) / len(scores_small) if scores_small else 0.0
    delta_q = mean_large - mean_small

    output = {
        "metadata": {
            "model_large": model_large,
            "model_small": model_small,
            "temperature": temperature,
            "n_tasks": len(EVAL_SET),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "results_large": [r.to_dict() for r in results_large],
        "results_small": [r.to_dict() for r in results_small],
        "aggregate": {
            "mean_quality_large": mean_large,
            "mean_quality_small": mean_small,
            "delta_q": delta_q,
        },
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    return output
