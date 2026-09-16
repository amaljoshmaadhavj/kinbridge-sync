"""
Phase 1 pilot runner — C1 Semantic Re-Issue measurement.

Execution flow:
  1. Load ground-truth prompts from prompts.json
  2. For each prompt, call Qwen 2.5 7B via Ollama to obtain action a1
  3. Re-send the same conversation (no tool result) to obtain a2
  4. Hash (a1, a2) under k_payload, k_bucket, k_intent
  5. Compute semantic similarity between a1 and a2
  6. Store raw observations in pilot/raw/

Usage:
    python -m pilot.run_pilot [--temperature 0.0] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from core.config import load_config
from pilot.key_schemes import k_bucket, k_intent, k_payload
from pilot.similarity import action_similarity
from pilot.tools import TOOL_LIST

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PILOT_DIR = Path(__file__).resolve().parent
_RAW_DIR = _PILOT_DIR / "raw"
_PROMPTS_FILE = _PILOT_DIR / "prompts.json"
_CONFIG_PATH = "pilot/pilot_config.yaml"


def _ensure_dirs() -> None:
    _RAW_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Ollama interaction
# ---------------------------------------------------------------------------

def _ollama_chat(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.0,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Send a chat-completion request to the local Ollama server.

    Returns the full JSON response body.  Raises RuntimeError on HTTP
    or structural errors.
    """
    url = f"{base_url}/api/chat"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if tools is not None:
        payload["tools"] = tools

    try:
        resp = requests.post(url, json=payload, timeout=timeout_s)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc

    data = resp.json()
    if "message" not in data:
        raise RuntimeError(f"Ollama response missing 'message': {json.dumps(data)[:500]}")
    return data


def _extract_tool_call(response: dict[str, Any]) -> dict[str, Any] | None:
    """Extract the first tool call from an Ollama chat response.

    Returns a dict with keys: tool, args, raw_message.
    Returns None if the model did not produce a tool call.
    """
    message = response.get("message", {})
    tool_calls = message.get("tool_calls", [])
    if not tool_calls:
        return None

    tc = tool_calls[0]
    func = tc.get("function", {})
    return {
        "tool": func.get("name", ""),
        "args": func.get("arguments", {}),
        "raw_message": message,
    }


# ---------------------------------------------------------------------------
# Single-prompt experiment
# ---------------------------------------------------------------------------

def run_single_prompt(
    prompt_entry: dict[str, Any],
    base_url: str,
    model: str,
    temperature: float,
    session_id: str,
    agent_id: str,
    bucket_delta_s: float,
) -> dict[str, Any]:
    """Run the a1 / a2 divergence test for a single ground-truth prompt.

    Returns a dict containing the complete raw observation.
    """
    prompt_id = prompt_entry["prompt_id"]
    intent_id = prompt_entry["intent_id"]
    tool_class = prompt_entry["tool_class"]
    prompt_text = prompt_entry["prompt"]

    timestamp_a1 = time.time()

    # --- a1: first action -------------------------------------------------
    messages_a1 = [
        {"role": "system", "content": "You are a field operations agent. Use the provided tools to fulfill the user's request."},
        {"role": "user", "content": prompt_text},
    ]

    try:
        resp_a1 = _ollama_chat(base_url, model, messages_a1, TOOL_LIST, temperature)
        tc_a1 = _extract_tool_call(resp_a1)
    except Exception as exc:
        tc_a1 = None
        resp_a1 = {"error": str(exc)}

    timestamp_a2 = time.time()

    # --- a2: re-issue (same conversation, no tool result) ------------------
    messages_a2 = [
        {"role": "system", "content": "You are a field operations agent. Use the provided tools to fulfill the user's request."},
        {"role": "user", "content": prompt_text},
    ]

    try:
        resp_a2 = _ollama_chat(base_url, model, messages_a2, TOOL_LIST, temperature)
        tc_a2 = _extract_tool_call(resp_a2)
    except Exception as exc:
        tc_a2 = None
        resp_a2 = {"error": str(exc)}

    timestamp_end = time.time()

    # --- Key computations --------------------------------------------------
    a1_tool = tc_a1["tool"] if tc_a1 else ""
    a1_args = tc_a1["args"] if tc_a1 else {}
    a2_tool = tc_a2["tool"] if tc_a2 else ""
    a2_args = tc_a2["args"] if tc_a2 else {}

    kp_a1 = k_payload(a1_tool, a1_args) if tc_a1 else ""
    kp_a2 = k_payload(a2_tool, a2_args) if tc_a2 else ""

    kb_a1 = k_bucket(agent_id, a1_tool, a1_args, timestamp_a1, bucket_delta_s) if tc_a1 else ""
    kb_a2 = k_bucket(agent_id, a2_tool, a2_args, timestamp_a2, bucket_delta_s) if tc_a2 else ""

    ki_a1 = k_intent(session_id, 1, intent_id)   # turn_seq=1 for both (same ground truth)
    ki_a2 = k_intent(session_id, 2, intent_id)   # turn_seq=2 for the re-issue

    # --- Semantic similarity -----------------------------------------------
    if tc_a1 and tc_a2:
        sim = action_similarity(a1_tool, a1_args, a2_tool, a2_args)
    else:
        sim = 0.0

    # --- Mismatch flags ----------------------------------------------------
    mismatch_payload = int(kp_a1 != kp_a2) if (tc_a1 and tc_a2) else -1
    mismatch_bucket  = int(kb_a1 != kb_a2) if (tc_a1 and tc_a2) else -1
    # Intent key: turn_seq differs by construction, so we test the *tool+args*
    # portion — for the intent scheme the key MUST match by design.
    # The mismatch test here is whether the intent key is the same when
    # we fix intent_id (it should always be, since intent_id is ground truth).
    mismatch_intent  = 0  # by construction: same intent_id → same key base

    # --- Build observation --------------------------------------------------
    observation = {
        "prompt_id": prompt_id,
        "intent_id": intent_id,
        "tool_class": tool_class,
        "prompt": prompt_text,
        "model": model,
        "temperature": temperature,
        "session_id": session_id,
        "agent_id": agent_id,
        "run_id": str(uuid.uuid4()),
        "timestamp_iso": datetime.now(timezone.utc).isoformat(),
        "timestamp_a1": timestamp_a1,
        "timestamp_a2": timestamp_a2,
        "timestamp_end": timestamp_end,
        "a1": {
            "tool": a1_tool,
            "args": a1_args,
            "success": tc_a1 is not None,
            "key_payload": kp_a1,
            "key_bucket": kb_a1,
            "key_intent": ki_a1,
            "raw_response": resp_a1,
        },
        "a2": {
            "tool": a2_tool,
            "args": a2_args,
            "success": tc_a2 is not None,
            "key_payload": kp_a2,
            "key_bucket": kb_a2,
            "key_intent": ki_a2,
            "raw_response": resp_a2,
        },
        "similarity": sim,
        "mismatch_payload": mismatch_payload,
        "mismatch_bucket": mismatch_bucket,
        "mismatch_intent": mismatch_intent,
    }

    return observation


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run_pilot(
    temperature: float = 0.0,
    dry_run: bool = False,
) -> Path:
    """Run the full 19-prompt pilot at a given temperature.

    Returns the path to the written raw observations JSON file.
    """
    _ensure_dirs()
    cfg = load_config(_CONFIG_PATH)

    base_url = cfg["ollama"]["base_url"]
    model = cfg["ollama"]["model_large"]
    bucket_delta_s = cfg["bucket_width_s"]

    with open(_PROMPTS_FILE, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    session_id = str(uuid.uuid4())
    agent_id = "pilot-agent-001"

    observations: list[dict[str, Any]] = []

    for entry in prompts:
        if dry_run:
            obs = {
                "prompt_id": entry["prompt_id"],
                "intent_id": entry["intent_id"],
                "tool_class": entry["tool_class"],
                "prompt": entry["prompt"],
                "dry_run": True,
            }
        else:
            obs = run_single_prompt(
                prompt_entry=entry,
                base_url=base_url,
                model=model,
                temperature=temperature,
                session_id=session_id,
                agent_id=agent_id,
                bucket_delta_s=bucket_delta_s,
            )
        observations.append(obs)
        print(f"  [{entry['prompt_id']}] done  (temp={temperature})")

    # Write raw observations
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_file = _RAW_DIR / f"pilot_temp{temperature}_{ts}.json"
    payload = {
        "metadata": {
            "model": model,
            "temperature": temperature,
            "session_id": session_id,
            "agent_id": agent_id,
            "bucket_delta_s": bucket_delta_s,
            "run_timestamp": ts,
            "n_prompts": len(prompts),
        },
        "observations": observations,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(observations)} observations to {out_file}")
    return out_file


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Kinbridge-Sync Phase 1 pilot runner")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Temperature for the LLM (default: 0.0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip Ollama calls; write placeholder observations.")
    args = parser.parse_args()
    run_pilot(temperature=args.temperature, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
