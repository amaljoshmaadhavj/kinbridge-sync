"""
Dedicated negative-trial generator for Phase 1b FMR estimation.

Generates independent different-intent action pairs for measuring
False Match Rate (FMR) at each similarity threshold.

Each negative trial:
  1. Selects two distinct intents (intent_A, intent_B)
  2. Selects one prompt for each intent
  3. Makes a fresh Ollama model call for each prompt
  4. Parses the resulting tool action
  5. Computes cosine similarity between the two actions
  6. Labels as same_intent=False

--n-trials N means N SUCCESSFUL negative trials. Failed attempts are
retried with fresh model calls until N valid observations are obtained.

Usage:
    python -m pilot.run_negative_trials \
        --temperature 0.0 \
        --n-trials 200 \
        --seed 42 \
        --model qwen2.5:7b-instruct

    python -m pilot.run_negative_trials \
        --temperature 0.7 \
        --n-trials 200 \
        --seed 42 \
        --model qwen2.5:7b-instruct
"""

from __future__ import annotations

import argparse
import json
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from core.config import load_config
from pilot.similarity import action_similarity
from pilot.tools import TOOL_LIST

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PILOT_DIR = Path(__file__).resolve().parent
_PROMPTS_FILE = _PILOT_DIR / "prompts.json"
_RAW_FMR_DIR = Path(__file__).resolve().parent.parent / "results" / "raw_fmr"
_CONFIG_PATH = "pilot/pilot_config.yaml"


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def _load_prompts() -> list[dict[str, Any]]:
    """Load ground-truth prompts from prompts.json."""
    with open(_PROMPTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _group_prompts_by_intent(
    prompts: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group prompts by intent_id."""
    by_intent: dict[str, list[dict[str, Any]]] = {}
    for p in prompts:
        by_intent.setdefault(p["intent_id"], []).append(p)
    return by_intent


# ---------------------------------------------------------------------------
# Intent-pair selection strategy
# ---------------------------------------------------------------------------

def select_intent_pairs(
    intent_ids: list[str],
    n_trials: int,
    rng: random.Random,
) -> list[tuple[str, str]]:
    """Select ordered intent pairs for negative trials.

    Strategy: distribute trials as evenly as possible across all
    unordered distinct-intent pairs, then randomize order.

    Returns a list of (intent_A, intent_B) tuples, length = n_trials.
    """
    from itertools import combinations

    all_pairs = list(combinations(intent_ids, 2))
    n_pairs = len(all_pairs)

    # Calculate trials per pair: floor(n_trials / n_pairs) with remainder
    base_per_pair = n_trials // n_pairs
    remainder = n_trials % n_pairs

    selected: list[tuple[str, str]] = []
    for i, pair in enumerate(all_pairs):
        # First `remainder` pairs get one extra trial
        count = base_per_pair + (1 if i < remainder else 0)
        for _ in range(count):
            # Randomly order within the pair to avoid directional bias
            if rng.random() < 0.5:
                selected.append(pair)
            else:
                selected.append((pair[1], pair[0]))

    rng.shuffle(selected)
    return selected


def _next_random_pair(
    intent_ids: list[str],
    rng: random.Random,
) -> tuple[str, str]:
    """Generate a single random intent pair using the RNG.

    Used for retry attempts when a trial fails. Advances the RNG state.
    """
    from itertools import combinations
    all_pairs = list(combinations(intent_ids, 2))
    pair = all_pairs[rng.randint(0, len(all_pairs) - 1)]
    if rng.random() < 0.5:
        return pair
    else:
        return (pair[1], pair[0])


# ---------------------------------------------------------------------------
# Ollama interaction (reuses pattern from run_pilot.py)
# ---------------------------------------------------------------------------

def _ollama_chat(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.0,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Send a chat-completion request to the local Ollama server."""
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
    """Extract the first tool call from an Ollama chat response."""
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
        "call_id": tc.get("id", ""),
        "created_at": response.get("created_at", ""),
        "total_duration": response.get("total_duration", 0),
        "eval_count": response.get("eval_count", 0),
    }


# ---------------------------------------------------------------------------
# Single negative trial
# ---------------------------------------------------------------------------

def run_single_negative_trial(
    intent_a: str,
    intent_b: str,
    prompts_by_intent: dict[str, list[dict[str, Any]]],
    base_url: str,
    model: str,
    temperature: float,
) -> dict[str, Any]:
    """Run a single negative trial: generate one action for each of two
    distinct intents and compute their similarity.

    Returns a dict containing the complete trial observation.
    """
    # Select one prompt for each intent (first available)
    prompt_a = prompts_by_intent[intent_a][0]
    prompt_b = prompts_by_intent[intent_b][0]

    # --- Generate action for intent A ---
    messages_a = [
        {"role": "system", "content": "You are a field operations agent. Use the provided tools to fulfill the user's request."},
        {"role": "user", "content": prompt_a["prompt"]},
    ]

    timestamp_a = time.time()
    try:
        resp_a = _ollama_chat(base_url, model, messages_a, TOOL_LIST, temperature)
        tc_a = _extract_tool_call(resp_a)
    except Exception as exc:
        tc_a = None
        resp_a = {"error": str(exc)}

    # --- Generate action for intent B ---
    messages_b = [
        {"role": "system", "content": "You are a field operations agent. Use the provided tools to fulfill the user's request."},
        {"role": "user", "content": prompt_b["prompt"]},
    ]

    timestamp_b = time.time()
    try:
        resp_b = _ollama_chat(base_url, model, messages_b, TOOL_LIST, temperature)
        tc_b = _extract_tool_call(resp_b)
    except Exception as exc:
        tc_b = None
        resp_b = {"error": str(exc)}

    timestamp_end = time.time()

    # --- Extract actions ---
    tool_a = tc_a["tool"] if tc_a else ""
    args_a = tc_a["args"] if tc_a else {}
    tool_b = tc_b["tool"] if tc_b else ""
    args_b = tc_b["args"] if tc_b else {}

    # --- Compute similarity ---
    if tc_a and tc_b:
        sim = action_similarity(tool_a, args_a, tool_b, args_b)
    else:
        sim = 0.0

    # --- Build trial observation ---
    trial = {
        "trial_id": f"neg_{uuid.uuid4().hex[:12]}",
        "temperature": temperature,
        "intent_id_a": intent_a,
        "intent_id_b": intent_b,
        "prompt_id_a": prompt_a["prompt_id"],
        "prompt_id_b": prompt_b["prompt_id"],
        "tool_class_a": prompt_a["tool_class"],
        "tool_class_b": prompt_b["tool_class"],
        "action_a": {
            "tool": tool_a,
            "args": args_a,
            "success": tc_a is not None,
            "raw_response": resp_a,
            "timestamp": timestamp_a,
        },
        "action_b": {
            "tool": tool_b,
            "args": args_b,
            "success": tc_b is not None,
            "raw_response": resp_b,
            "timestamp": timestamp_b,
        },
        "similarity": sim,
        "same_intent": False,  # ground truth: different intents
        "model": model,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return trial


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_trial(trial: dict[str, Any]) -> list[str]:
    """Validate a negative trial. Returns list of error messages (empty = valid)."""
    errors: list[str] = []

    # Ground truth: different intents
    if trial["intent_id_a"] == trial["intent_id_b"]:
        errors.append(f"Same intent: {trial['intent_id_a']} == {trial['intent_id_b']}")

    # Actions must be present
    if not trial["action_a"]["success"]:
        errors.append("Action A failed (no tool call)")
    if not trial["action_b"]["success"]:
        errors.append("Action B failed (no tool call)")

    # Similarity must be in valid range
    sim = trial["similarity"]
    if sim < -1.0 or sim > 1.0:
        errors.append(f"Similarity out of range: {sim}")

    # Check for reuse: a1 and a2 must come from different model calls
    tc_a_list = trial["action_a"]["raw_response"].get("message", {}).get("tool_calls", [])
    tc_b_list = trial["action_b"]["raw_response"].get("message", {}).get("tool_calls", [])
    id_a = tc_a_list[0].get("id", "") if tc_a_list else ""
    id_b = tc_b_list[0].get("id", "") if tc_b_list else ""
    if id_a and id_b and id_a == id_b:
        errors.append(f"Same model call ID: {id_a}")

    return errors


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def write_trials(
    successful_trials: list[dict[str, Any]],
    failed_attempts: list[dict[str, Any]],
    temperature: float,
    seed: int,
    model: str,
    n_requested: int,
    n_attempted: int,
    intent_pair_counts: dict[tuple[str, str], int],
    out_dir: Path,
) -> Path:
    """Write trials to a JSON file and return the path.

    Only successful trials are included in the `trials` array (FMR observations).
    Failed attempts are recorded in `failed_attempts` for audit trail.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"neg_trials_temp{temperature:.1f}_{timestamp}.json"
    out_path = out_dir / filename

    # Build intent pair distribution summary (successful trials only)
    pair_dist = {}
    for (ia, ib), count in sorted(intent_pair_counts.items()):
        key = f"{ia}|{ib}"
        pair_dist[key] = count

    n_successful = len(successful_trials)

    data = {
        "metadata": {
            "temperature": temperature,
            "n_requested": n_requested,
            "n_attempted": n_attempted,
            "n_successful": n_successful,
            "n_failed": n_attempted - n_successful,
            "model": model,
            "seed": seed,
            "intent_pair_strategy": "even_distribution_across_all_78_pairs",
            "total_action_generations": n_attempted * 2,
            "timestamp": timestamp,
            "software_version": "phase1b_design_c_retry",
        },
        "intent_pair_distribution": {
            "unique_pairs": len(pair_dist),
            "trials_per_pair": pair_dist,
            "min_per_pair": min(pair_dist.values()) if pair_dist else 0,
            "max_per_pair": max(pair_dist.values()) if pair_dist else 0,
        },
        "trials": successful_trials,
        "failed_attempts": failed_attempts,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_negative_experiment(
    temperature: float,
    n_trials: int,
    seed: int,
    model: str,
    base_url: str,
    out_dir: Path,
) -> dict[str, Any]:
    """Run the full negative-trial experiment for one temperature.

    n_trials = N means N SUCCESSFUL negative trials. Failed attempts are
    retried with fresh model calls until N valid observations are obtained.

    RNG behavior under retries:
    - Initial N intent pairs are pre-generated deterministically from seed.
    - If a trial fails, a NEW intent pair is generated via RNG (advancing it).
    - The RNG state after the experiment depends on the number of failures.
    - The initial pair sequence is deterministic; retry pairs are not.

    Returns a summary dict.
    """
    prompts = _load_prompts()
    prompts_by_intent = _group_prompts_by_intent(prompts)
    intent_ids = sorted(prompts_by_intent.keys())

    rng = random.Random(seed)

    # Pre-generate N intent pairs (deterministic under seed)
    initial_pairs = select_intent_pairs(intent_ids, n_trials, rng)

    successful_trials: list[dict[str, Any]] = []
    failed_attempts: list[dict[str, Any]] = []
    intent_pair_counts: dict[tuple[str, str], int] = {}

    n_attempted = 0
    pair_index = 0  # index into initial_pairs

    while len(successful_trials) < n_trials:
        # Get next intent pair
        if pair_index < len(initial_pairs):
            intent_a, intent_b = initial_pairs[pair_index]
            pair_index += 1
        else:
            # Exhausted pre-generated pairs; generate new ones via RNG
            intent_a, intent_b = _next_random_pair(intent_ids, rng)

        n_attempted += 1
        pair_key = (intent_a, intent_b)

        try:
            trial = run_single_negative_trial(
                intent_a, intent_b, prompts_by_intent,
                base_url, model, temperature,
            )
            trial_errors = validate_trial(trial)

            if trial_errors:
                # Failed attempt — record for audit, do NOT count toward N
                failed_attempts.append({
                    "attempt_index": n_attempted - 1,
                    "intent_a": intent_a,
                    "intent_b": intent_b,
                    "errors": trial_errors,
                    "trial_id": trial["trial_id"],
                })
            else:
                # Successful trial — count toward N
                successful_trials.append(trial)
                intent_pair_counts[pair_key] = intent_pair_counts.get(pair_key, 0) + 1

            # Progress
            n_done = len(successful_trials)
            if n_done % 10 == 0 or n_done == n_trials:
                print(f"  [{n_done}/{n_trials}] successful "
                      f"(attempted={n_attempted}, failed={len(failed_attempts)}) "
                      f"(temp={temperature})")

        except Exception as exc:
            # Exception during trial — record for audit, do NOT count toward N
            failed_attempts.append({
                "attempt_index": n_attempted - 1,
                "intent_a": intent_a,
                "intent_b": intent_b,
                "errors": [str(exc)],
                "trial_id": None,
            })

    # Write output
    out_path = write_trials(
        successful_trials, failed_attempts,
        temperature, seed, model, n_trials, n_attempted,
        intent_pair_counts, out_dir,
    )

    # Summary
    summary = {
        "temperature": temperature,
        "n_trials_requested": n_trials,
        "n_trials_attempted": n_attempted,
        "n_trials_completed": len(successful_trials),
        "n_trials_failed": len(failed_attempts),
        "n_action_generations": n_attempted * 2,
        "unique_intent_pairs": len(intent_pair_counts),
        "min_trials_per_pair": min(intent_pair_counts.values()) if intent_pair_counts else 0,
        "max_trials_per_pair": max(intent_pair_counts.values()) if intent_pair_counts else 0,
        "output_path": str(out_path),
        "errors": failed_attempts,
    }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate dedicated negative trials for FMR estimation"
    )
    parser.add_argument(
        "--temperature", type=float, required=True,
        help="Temperature for model generation (0.0 or 0.7)",
    )
    parser.add_argument(
        "--n-trials", type=int, default=200,
        help="Number of SUCCESSFUL negative trials (default: 200)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Ollama model name (default: from config)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(_RAW_FMR_DIR),
        help=f"Output directory (default: {_RAW_FMR_DIR})",
    )
    args = parser.parse_args()

    cfg = load_config(_CONFIG_PATH)
    base_url = cfg["ollama"]["base_url"]
    model = args.model or cfg["ollama"]["model_large"]

    print(f"Phase 1b Negative Trial Generator")
    print(f"  Temperature:  {args.temperature}")
    print(f"  N trials:     {args.n_trials} (successful)")
    print(f"  Seed:         {args.seed}")
    print(f"  Model:        {model}")
    print(f"  Output:       {args.output_dir}")
    print()

    summary = run_negative_experiment(
        temperature=args.temperature,
        n_trials=args.n_trials,
        seed=args.seed,
        model=model,
        base_url=base_url,
        out_dir=Path(args.output_dir),
    )

    print(f"\nCompleted: {summary['n_trials_completed']}/{summary['n_trials_requested']} successful trials")
    print(f"Attempted: {summary['n_trials_attempted']}")
    print(f"Failed:    {summary['n_trials_failed']}")
    print(f"Actions:   {summary['n_action_generations']} generations")
    print(f"Intent pairs covered: {summary['unique_intent_pairs']}")
    print(f"Output:    {summary['output_path']}")

    if summary["errors"]:
        print(f"\nFailed attempts:")
        for e in summary["errors"]:
            print(f"  Attempt {e['attempt_index']}: {e['errors']}")


if __name__ == "__main__":
    main()
