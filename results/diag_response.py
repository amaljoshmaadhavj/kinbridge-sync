"""Check actual Ollama response structure from pilot data."""
import sys
sys.path.insert(0, ".")
import json

with open('pilot/raw/pilot_temp0.0_20260916T160848Z.json') as f:
    data = json.load(f)

# Check first observation's raw response structure
obs = data['observations'][0]
print(f"Prompt: {obs['prompt_id']} ({obs['intent_id']})")
print(f"a1 success: {obs['a1']['success']}")
print(f"a1 raw_response keys: {list(obs['a1']['raw_response'].keys())}")

resp = obs['a1']['raw_response']
print(f"\nFull raw_response:")
print(json.dumps(resp, indent=2)[:2000])

# Now check what _extract_tool_call would do with this
from pilot.run_negative_trials import _extract_tool_call
tc = _extract_tool_call(resp)
print(f"\n_extract_tool_call result: {tc}")
