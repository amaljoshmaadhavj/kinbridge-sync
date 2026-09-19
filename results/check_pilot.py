import json
with open('pilot/raw/pilot_temp0.0_20260916T160848Z.json') as f:
    data = json.load(f)
obs = data['observations']
print(f'Total observations: {len(obs)}')
success_count = sum(1 for o in obs if o['a1']['success'] and o['a2']['success'])
print(f'Both a1+a2 successful: {success_count}')
a1_tc = sum(1 for o in obs if o['a1']['raw_response'].get('message',{}).get('tool_calls',[]))
a2_tc = sum(1 for o in obs if o['a2']['raw_response'].get('message',{}).get('tool_calls',[]))
print(f'a1 tool calls present: {a1_tc}')
print(f'a2 tool calls present: {a2_tc}')
print()
for o in obs[:3]:
    tc_a = o['a1']['raw_response'].get('message',{}).get('tool_calls',[])
    tc_b = o['a2']['raw_response'].get('message',{}).get('tool_calls',[])
    print(f'{o["prompt_id"]} ({o["intent_id"]}): a1_success={o["a1"]["success"]}, a2_success={o["a2"]["success"]}, a1_tool_calls={len(tc_a)}, a2_tool_calls={len(tc_b)}')
    if tc_a:
        print(f'  a1 tool: {tc_a[0]["function"]["name"]}, id: {tc_a[0].get("id","")}')
    if tc_b:
        print(f'  a2 tool: {tc_b[0]["function"]["name"]}, id: {tc_b[0].get("id","")}')
