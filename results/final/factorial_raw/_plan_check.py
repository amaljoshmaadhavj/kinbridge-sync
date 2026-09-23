from experiments.phase5.scenario import (
    METHODS,
    OUTAGES_S,
    load_scenario,
    resolve_episode,
)

t = load_scenario()
jobs = []
order = []
for method in METHODS:
    for outage in OUTAGES_S:
        order.append((method, outage))
        for ep in range(100):
            jobs.append(resolve_episode(t, method, outage, ep))

assert len(jobs) == 1600, len(jobs)
keys = [(j["method"], j["outage_duration_s"], j["episode_id"]) for j in jobs]
assert len(set(keys)) == 1600, "duplicate job keys"
assert all(j["topology"] == "same-node" for j in jobs)
assert all(j["replication_lag_ms"] == 0 for j in jobs)
assert all(j["scenario_id"] == "replan-dup-2" for j in jobs)
assert all(j["master_seed"] == 42 for j in jobs)

# deterministic cell order: method then outage then episode
assert keys[0] == ("cold_restart", 0, 0)
assert keys[-1] == ("kinbridge_sync", 300, 99)

# no ablation / c3 job fields
assert all("ablation_row" not in j for j in jobs)

print("cell order:", order)
print("jobs", len(jobs), "unique", len(set(keys)))
print(
    "all same-node lag0:",
    all(j["topology"] == "same-node" and j["replication_lag_ms"] == 0 for j in jobs),
)
print("seed samples:", jobs[0]["episode_seed"], jobs[-1]["episode_seed"])
print("PLAN OK 4x4x100=1600")
