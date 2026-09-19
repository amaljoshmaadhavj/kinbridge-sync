"""Diagnostic: trace one negative trial end-to-end without Ollama."""
import sys, json
sys.path.insert(0, ".")
from pilot.run_negative_trials import (
    _load_prompts, _group_prompts_by_intent, validate_trial,
    run_single_negative_trial, select_intent_pairs
)
import random

# Load prompts
prompts = _load_prompts()
by_intent = _group_prompts_by_intent(prompts)
intent_ids = sorted(by_intent.keys())

# Get first intent pair (same as seed=42 would produce)
rng = random.Random(42)
pairs = select_intent_pairs(intent_ids, 200, rng)
intent_a, intent_b = pairs[0]
print(f"First pair: ({intent_a}, {intent_b})")
print(f"Prompt A: {by_intent[intent_a][0]}")
print(f"Prompt B: {by_intent[intent_b][0]}")

# Simulate a SUCCESSFUL Ollama response
fake_response_a = {
    "model": "qwen2.5:7b-instruct",
    "created_at": "2026-09-17T00:00:00Z",
    "message": {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_diag_a1",
                "function": {
                    "index": 0,
                    "name": "navigate_to",
                    "arguments": {"lat": 45.0, "lon": 6.0, "mode": "fastest"}
                }
            }
        ]
    }
}
fake_response_b = {
    "model": "qwen2.5:7b-instruct",
    "created_at": "2026-09-17T00:00:01Z",
    "message": {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_diag_b1",
                "function": {
                    "index": 0,
                    "name": "request_supply_drop",
                    "arguments": {"lat": 51.0, "lon": 0.0, "payload": "test", "priority": "high"}
                }
            }
        ]
    }
}

# Manually construct a trial (simulating what run_single_negative_trial does)
from pilot.similarity import action_similarity

prompt_a = by_intent[intent_a][0]
prompt_b = by_intent[intent_b][0]

tc_a_tool = fake_response_a["message"]["tool_calls"][0]["function"]["name"]
tc_a_args = fake_response_a["message"]["tool_calls"][0]["function"]["arguments"]
tc_b_tool = fake_response_b["message"]["tool_calls"][0]["function"]["name"]
tc_b_args = fake_response_b["message"]["tool_calls"][0]["function"]["arguments"]

sim = action_similarity(tc_a_tool, tc_a_args, tc_b_tool, tc_b_args)
print(f"\nSimilarity: {sim}")

trial = {
    "trial_id": "neg_diag_001",
    "temperature": 0.0,
    "intent_id_a": intent_a,
    "intent_id_b": intent_b,
    "prompt_id_a": prompt_a["prompt_id"],
    "prompt_id_b": prompt_b["prompt_id"],
    "tool_class_a": prompt_a["tool_class"],
    "tool_class_b": prompt_b["tool_class"],
    "action_a": {
        "tool": tc_a_tool,
        "args": tc_a_args,
        "success": True,
        "raw_response": fake_response_a,
        "timestamp": 0.0,
    },
    "action_b": {
        "tool": tc_b_tool,
        "args": tc_b_args,
        "success": True,
        "raw_response": fake_response_b,
        "timestamp": 0.0,
    },
    "similarity": sim,
    "same_intent": False,
    "model": "qwen2.5:7b-instruct",
    "run_timestamp": "2026-09-17T00:00:00Z",
}

# Validate
errors = validate_trial(trial)
print(f"\nValidation errors: {errors}")
print(f"Trial is valid: {len(errors) == 0}")

# Now simulate what happens if Ollama returns NO tool calls (content instead)
print("\n--- Simulating NO tool calls ---")
fake_response_no_tc = {
    "model": "qwen2.5:7b-instruct",
    "created_at": "2026-09-17T00:00:00Z",
    "message": {
        "role": "assistant",
        "content": "I'll navigate to the coordinates.",
        "tool_calls": []
    }
}
tc = fake_response_no_tc["message"]["tool_calls"]
print(f"tool_calls list: {tc}")
print(f"bool(tc): {bool(tc)}")
print(f"_extract_tool_call would return: None (empty tool_calls)")

# Now simulate what happens if Ollama returns tool_calls as a KEY but with weird format
print("\n--- Simulating tool_calls with dict instead of list ---")
fake_response_dict_tc = {
    "model": "qwen2.5:7b-instruct",
    "message": {
        "role": "assistant",
        "content": "",
        "tool_calls": {
            "id": "call_123",
            "function": {"name": "navigate_to", "arguments": {}}
        }
    }
}
tc_dict = fake_response_dict_tc["message"]["tool_calls"]
print(f"tool_calls type: {type(tc_dict)}")
print(f"bool(tc_dict): {bool(tc_dict)}")
try:
    first = tc_dict[0]
    print(f"tc_dict[0]: {first}")
except Exception as e:
    print(f"tc_dict[0] ERROR: {type(e).__name__}: {e}")
