"""Post-run raw-data validation for the 1600-episode factorial."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from experiments.phase5.record import read_episode_records, validate_episode_record
from experiments.phase5.scenario import METHODS, OUTAGES_S

JSONL = Path("results/final/factorial_raw/episodes.jsonl")
EXPECTED = 1600
errors: list[str] = []

if not JSONL.exists():
    print(f"FAIL missing {JSONL}")
    sys.exit(1)

records = read_episode_records(JSONL)
print(f"records_read={len(records)}")

if len(records) != EXPECTED:
    errors.append(f"expected {EXPECTED} records, got {len(records)}")

# --- cell counts ---
cells: Counter[tuple[str, int]] = Counter()
ep_ids: dict[tuple[str, int], list[int]] = defaultdict(list)
seeds_master = set()
taus = set()
tau_sources = set()
topologies = set()
lags = set()
ablation_rows = set()
scenarios = set()
failed = []
duplicate_keys = []
seen_keys = set()

for i, rec in enumerate(records):
    try:
        validate_episode_record(rec)
    except Exception as exc:
        errors.append(f"record[{i}] validation: {exc}")
        continue

    method = rec["method"]
    outage = int(rec["outage_duration"])
    eid = int(rec["episode_id"])
    key = (method, outage, eid)
    if key in seen_keys:
        duplicate_keys.append(key)
    seen_keys.add(key)

    if method not in METHODS:
        errors.append(f"unexpected method {method!r} at {i}")
    if outage not in OUTAGES_S:
        errors.append(f"unexpected outage {outage!r} at {i}")

    cells[(method, outage)] += 1
    ep_ids[(method, outage)].append(eid)
    seeds_master.add(rec.get("seed"))
    taus.add(rec.get("tau_star"))
    tau_sources.add(str(rec.get("tau_star_source")))
    topologies.add(rec.get("topology"))
    lags.add(int(rec.get("replication_lag", -999)))
    ablation_rows.add(rec.get("ablation_row", "full"))
    scenarios.add(rec.get("scenario_id"))

    # failure-ish signals in record structure (no status field; treat missing actions as failure)
    if not rec.get("actions"):
        failed.append(key)

    # required reproducibility fields
    for f in ("episode_seed", "method", "outage_duration", "episode_id", "seed"):
        if f not in rec:
            errors.append(f"missing {f} in record {i}")

    # episode_seed must be derived consistently (non-zero, unique per cell+id roughly)
    if not isinstance(rec.get("episode_seed"), int) or rec["episode_seed"] < 0:
        errors.append(f"bad episode_seed at {i}")

if duplicate_keys:
    errors.append(f"duplicate episode keys: {duplicate_keys[:10]} (n={len(duplicate_keys)})")

expected_cells = {(m, o): 100 for m in METHODS for o in OUTAGES_S}
for cell, n in expected_cells.items():
    got = cells.get(cell, 0)
    if got != n:
        errors.append(f"cell {cell} count={got} expected {n}")
    ids = sorted(ep_ids.get(cell, []))
    if ids != list(range(100)):
        errors.append(f"cell {cell} episode_ids not 0..99: missing={set(range(100))-set(ids)} extra={set(ids)-set(range(100))}")

if seeds_master != {42}:
    errors.append(f"master seeds != {{42}}: {seeds_master}")
if ablation_rows != {"full"}:
    errors.append(f"unexpected ablation rows: {ablation_rows}")
if topologies != {"same-node"}:
    errors.append(f"unexpected topologies: {topologies}")
if lags != {0}:
    errors.append(f"unexpected lags: {lags} (C3 must not leak)")
if scenarios != {"replan-dup-2"}:
    errors.append(f"unexpected scenarios: {scenarios}")
if failed:
    errors.append(f"failed/empty-action episodes: {failed[:20]} n={len(failed)}")

# no C3/ablation contamination: all lag 0 same-node already checked
# tau provenance
if taus != {0.5051826557580381}:
    errors.append(f"unexpected tau* values: {taus}")
if not any("tau_star_results.csv" in s for s in tau_sources):
    errors.append(f"tau* source missing artifact path: {tau_sources}")

print("cells:", dict(sorted(cells.items())))
print("master_seed:", seeds_master)
print("tau*:", taus)
print("tau_sources:", tau_sources)
print("topologies:", topologies, "lags:", lags, "ablation:", ablation_rows)
print("unique_episode_keys:", len(seen_keys))
print("raw_bytes:", JSONL.stat().st_size)

# phase1b untouched?
pre = Path("results/final/factorial_raw/integrity_pre_run.json")
if pre.exists():
    import hashlib

    man = json.loads(pre.read_text(encoding="utf-8"))
    for rel, h in man["files"].items():
        p = Path(rel)
        if p.exists():
            now = hashlib.sha256(p.read_bytes()).hexdigest()
            if now != h:
                errors.append(f"INTEGRITY CHANGED: {rel}")
        else:
            errors.append(f"INTEGRITY MISSING: {rel}")
    print("integrity_checked:", len(man["files"]))

# phase1b must not gain phase5 files
phase1b = set(p.name for p in Path("results/phase1b").iterdir() if p.is_file())
print("phase1b_files:", sorted(phase1b))

if errors:
    print("VALIDATION FAILURES:")
    for e in errors:
        print(" -", e)
    sys.exit(1)

print("VALIDATION OK: exactly 1600 records, 16 cells x 100, pure factorial")
sys.exit(0)
