# Phase 5 — Implementation Report

**Status:** Implementation complete. Tests green. Smoke test run only.
**Not run:** full 1600-episode factorial (explicitly out of scope for this delivery).

---

## 1. Scope and integrity

| Item | Value |
|---|---|
| Scenario | `replan-dup-2` (`config/experiment/phase5_scenario.yaml`) |
| Master seed | 42 |
| Calibration | T=0.7 → τ\* read from `results/phase1b/tau_star_results.csv` (never hardcoded) |
| Factorial grid | 4 methods × 4 outages × 100 eps/cell (configured; **not executed**) |
| C3 grid | lag {0, 500, −1} × {same-node, cross-node} (configured; **not executed** at paper scale) |
| Ablation ladder | 6 rows @ 90 s outage × 200 eps/row (configured; **not executed** at paper scale) |
| Smoke executed | `kinbridge_sync` × outage 90 s × **2 episodes** → `results/final/raw/` |
| Full test suite | **493 passed, 31 skipped, 0 failed** (Windows / Python 3.13 / pytest 7.4.3) |
| Phase 5 unit tests | `tests/test_phase5.py` — **40 passed** |

Frozen surfaces were not modified by Phase 5 logic: `client/`, `tool_world/`, `pilot/key_schemes.py`, `pilot/wilson.py`, `reintegration/` (Algorithm 1 + baselines), `kinbridge_math/`, `core/`, `Markdown/`, `config/experiment/factorial.yaml`, `config/experiment/ablation.yaml`, `results/phase1b/*`, `experiments/netctl.py`, `experiments/outage_model.py`.

Working-tree changes to `Dockerfile` / `docker-compose.yml` / Phase 3b test files pre-date Phase 5 (Phase 3b replication work, uncommitted). Phase 5 did not edit them.

---

## 2. B1–B5 resolutions

### B1 — Scenario (`phase5_scenario.yaml` + `scenario.py`)
- Two pre-gap intents: **a1** `navigate_to` (delivery `pre_gap`, ttl 0, effect E1) and **a2** `request_supply_drop` (delivery `in_flight`, ttl 30, effect E2).
- Replanned **a1p/a2p** at turns 99/100 with **identical args** (fresh intent keys diverge → cold-restart duplicate construction).
- V_pre true, V_post false; `client_policy: decision_based`.
- Defaults: same-node, reconnect edge-a, lag 0; C3 grid and ablation ladder declared in the same file.
- Child seed: `sha256(master:scenario:method:outage:episode) mod 2^63`, applied to numpy as `mod 2^32`.

### B2 — τ\* loader (`tau_star.py`)
- Reads `results/phase1b/tau_star_results.csv`.
- T=0.7 → **0.5051826557580381** (method `bisection`); T=0.0 → `None` (unestimable) → fail closed / ESCALATE.
- `calibration_provenance(temperature, path=…)` supplies record provenance; `phi_tau_star_provider` mismatches temperature → `None` (no substitution).
- Test `test_F_no_hardcoded_tau_star_in_phase5_sources` asserts the numeric literal does not appear in Phase 5 code/config.

### B3/B5 — Ablation (`ablation.py` + yaml `ablation_ladder`)
Authoritative six Table VI rows (frozen `ablation.yaml` collapses hash/intent into identical booleans and lacks the degradation-gate row):

| Row | key_scheme | intent | staleness | epoch guard | degradation gate |
|---|---|---|---|---|---|
| buffer_only | **none** | off | off | off | off |
| plus_hash_keys | **payload** | off | off | off | off |
| plus_intent_keys | **intent** | on | off | off | off |
| plus_staleness | intent | on | **on** | off | off |
| plus_epoch_guard | intent | on | on | **on** | off |
| plus_degradation_gate | intent | on | on | on | **on** |

- `buffer_only` uses `key_scheme: none` (server never dedupes — every call re-applies) so it is distinguishable from `plus_hash_keys` (k_payload).
- Degradation gate is **harness-level** termination (ESCALATED/DEGRADED); it never overrides epoch guard, τ\*, F1 ABSENT, TTL/V_pre, or UNKNOWN.

### B4 — C3 compose override (`compose_override.py`)
- Generates a single-service override for `replicator` with `--lag-ms` ∈ {0, 500, −1} and `REPLICATOR_LAG_MS`.
- Applied as `docker compose -f docker-compose.yml -f <override> up -d`; base file never edited.
- Lag −1 = replicator “never”; value is **configured lag**, not measured latency.

### B5 — Baselines (`baselines.py`)
All four methods share the same harness (tool server, ledgers, replication, ground truth):

| Method | Behavior |
|---|---|
| `cold_restart` | Buffer discarded → all DROPPED; client executes replans only |
| `naive_retry` | Blind execute (skips buffer COMMITTED/INVALIDATED); no TTL/V_pre/epoch |
| `verify_before_retry` | Ledger lookup; ABSENT/UNKNOWN → execute; no TTL/V_pre/epoch |
| `kinbridge_sync` | Frozen `ReintegrationService` + flags from ablation row |

---

## 3. Harness semantics (`harness.py`)

In-process deterministic simulation against **frozen** components:

1. **Pre-gap:** append to `ActionBuffer` (no session/turn on WAL — resolved from template when needed); `pre_gap` delivered/ACK’d; `in_flight` dropped when outage > 0; outage 0 delivers both.
2. **Replication:** edge-a → edge-b with configured lag; lag −1 never forwards; upsert mirrors frozen `sqlite_writer` (**ledger_meta epoch is never bumped**).
3. **Reintegrate:** baseline driver + epoch guard / F1 / staleness / semantic via frozen service.
4. **Client policy `decision_based`:** COMMITTED/REPLAYED → skip; MERGED → execute once; ESCALATED/INVALIDATED → skip; DROPPED → execute (cold restart); no counterpart → execute.
5. **Ground truth:** applications by `effect_identity`; duplicate = \>1 application; stale = `(t − t0) > ttl` with ttl > 0 (t0 = 1.0 for in_flight, else 0.0).
6. **Unsafe rate** = (duplicate + stale) / eligible_actions; eligible = 2 buffered actions/episode; escalation and F1 escalate are **not** unsafe.

**Key scheme** is a client/episode property applied uniformly (buffer key = server key authority):

- `intent` → k_intent when session/turn/intent present, else payload;
- `payload` → k_payload only;
- `none` → buffer-local key; server **always re-applies** (no dedupe).

Service `_resolve_key` only rebuilds intent keys when rec has session/turn; harness supplies those from the template so buffer/server keys align.

---

## 4. Runner (`experiments/run_experiment.py`)

```
python -m experiments.run_experiment --method kinbridge_sync --outage-duration 90 \
  --episodes 2 --seed 42 --output-dir results/final/raw

python -m experiments.run_experiment --grid factorial --episodes 100
python -m experiments.run_experiment --grid c3 --replication-lag-ms 500 --topology cross-node --episodes 20
python -m experiments.run_experiment --grid ablation --ablation-row plus_epoch_guard --episodes 200
python -m experiments.run_experiment --harness docker --replication-lag-ms 500 ...
```

- Writes append-only `episodes.jsonl` + flat `results.csv`.
- **`--episodes` defaults to 0** and factorial with 0 episodes exits **2** (prevents accidental overnight runs).
- `--harness docker` also materialises the C3 compose override (still runs the deterministic in-process model; full Docker factorial remains an explicit later run).
- Output grid: `--grid factorial|c3|ablation`.

---

## 5. Analyzer (`experiments/analyze_results.py`)

```
python -m experiments.analyze_results --input results/final/raw/episodes.jsonl \
  --output-dir results/final --figures
```

- Fail-loud on missing/malformed JSONL.
- Wilson 95% CI via frozen `pilot/wilson.py` (z = 1.96) on every rate.
- Cohen’s h = 2asin(√p₁) − 2asin(√p₂) vs `cold_restart` (Table IV) and vs `buffer_only` (Table VI).
- No method is ranked “best”.
- Outputs: `results/final/results.csv`, `table_III.csv` … `table_VI.csv`, optional `figures/unsafe_rate_vs_outage.png`.
- Table III always materialises the full 4×4 grid (empty cells remain 0-episode placeholders until a full run fills them).

---

## 6. Smoke test (only experiment run)

Command:

```powershell
python -m experiments.run_experiment --method kinbridge_sync --outage-duration 90 `
  --episodes 2 --seed 42 --output-dir results/final/raw
python -m experiments.analyze_results --input results/final/raw/episodes.jsonl `
  --output-dir results/final --figures
```

Observed (n = 2 episodes, **not** paper results):

| metric | value |
|---|---|
| episodes | 2 (ids 0, 1) |
| method / outage / topology | kinbridge_sync / 90 s / same-node |
| epoch_guard | True |
| a1 decision | COMMITTED (effect E1 applied once) |
| a2 decision | INVALIDATED (TTL 30 < elapsed 90) |
| unsafe actions | 0 / 4 eligible |
| median TTR | 0.0 s |
| τ\* in records | 0.5051826557580381 from Phase 1b CSV |
| Wilson CI (unsafe 0/4) | [0, 0.4899] |

Determinism: re-running the same cell with the same seed reproduces byte-identical episode records (`test_R_fixed_seed_determinism`).

**Honest expectation trace (validation only — not claimed as results):**

| method | out0 | out10 | out90 | out300 | notes |
|---|---|---|---|---|---|
| cold_restart | 1.0 | 0.5 | 1.0 | 1.0 | intent-key replan duplicates a1; long outage also stales a2 |
| naive_retry | 0 | 0 | 0.5 | 0.5 | skips buffer COMMITTED; no TTL → stale a2 when outage > 30 |
| verify_before_retry | 0 | 0 | 0.5 | 0.5 | same as naive on this scenario |
| kinbridge_sync (same-node) | 0 | 0 | 0 | 0 | COMMITTED / MERGED or INVALIDATED |
| kinbridge_sync (cross-node) | 0 | 0 | 0 | 0 | guard fails (meta epoch 0); a1 COMMITTED, a2 ESCALATED, TTR ≈ 2.0 s |

Documented risks (may be indistinguishable at paper scale):

1. naive ≡ verify on this scenario (both skip buffer COMMITTED; neither checks TTL).
2. C3 lag 500 vs 0 may match for same-node (a1 committed at t=0 arrives before reconnect).
3. Cross-node kinbridge is lag-invariant because the frozen writer never bumps `ledger_meta`.
4. buffer_only vs plus_hash_keys may coincide where no payload collision is exercised by kinbridge decisions.

---

## 7. Tests A–X (`tests/test_phase5.py`, 40 tests)

| ID | Coverage |
|---|---|
| A | Scenario generation + resolve_episode validation |
| B | cold_restart distinct from other methods |
| C | Baseline behaviors across outages |
| D | Deterministic child seeds |
| E | Episode record schema + JSONL round-trip + rejects |
| F | τ\* CSV / T=0 unestimable / no hardcoded τ\* |
| G | Six ablation rows + key_scheme distinction |
| H | Compose overrides lag 0 / 500 / −1 |
| I | Same-node reconnect |
| J | Cross-node epoch guard + TTR ≈ 2.0 (± one 50 ms poll) |
| K | F1 ABSENT after guard / replication ambiguity |
| L | Wilson CI |
| M | Cohen’s h |
| N | Unsafe metric = duplicate ∪ stale; escalation excluded |
| O | Table shapes III (16), IV (4), V (6), VI (6) |
| P | Malformed input fails loud; runner rejects bad args |
| Q | Frozen file digests unchanged after full harness surface |
| R | Fixed-seed determinism |
| S | CLI `--episodes 0` → exit 2; smoke CLI path |
| T | Degradation gate does not override guard/F1 |
| U | Docker harness writes compose override |
| V | eligible_actions denominator = 2 |
| W | Outage 0 delivers both actions |
| X | Frozen factorial/ablation configs still parse unchanged |

Known pre-existing failure `tests/test_phase3_compose.py::TestComposeCLI::test_compose_up_and_down` was **not** reproduced as a failure in this run (compose CLI tests skipped in this environment). No Phase 5 change touched that file.

---

## 8. How to run the real factorial (when ready)

```powershell
# Full paper grid — 1600 episodes (4×4×100). Overnight class run.
python -m experiments.run_experiment --grid factorial --episodes 100 `
  --output-dir results/final/raw

# C3 (Table V)
foreach ($lag in 0, 500, -1) {
  foreach ($topo in "same-node", "cross-node") {
    python -m experiments.run_experiment --grid c3 --episodes 20 `
      --replication-lag-ms $lag --topology $topo `
      --output-dir results/final/raw
  }
}

# Ablation (Table VI)
python -m experiments.run_experiment --grid ablation --episodes 200 `
  --output-dir results/final/raw

python -m experiments.analyze_results --input results/final/raw/episodes.jsonl `
  --output-dir results/final --figures
```

Only numbers produced by those commands (with Wilson CIs) may be pasted into Tables III–VI.

---

## 9. Files added / changed (Phase 5)

| Path | Role |
|---|---|
| `config/experiment/phase5_scenario.yaml` | B1 scenario / C3 / ablation config (new) |
| `experiments/phase5/__init__.py` | Package |
| `experiments/phase5/tau_star.py` | B2 τ\* bridge |
| `experiments/phase5/scenario.py` | Episode / C3 resolution + child seeds |
| `experiments/phase5/ablation.py` | B3/B5 ladder resolution |
| `experiments/phase5/compose_override.py` | B4 C3 overrides |
| `experiments/phase5/record.py` | §11 schema + JSONL I/O |
| `experiments/phase5/baselines.py` | B5 four methods |
| `experiments/phase5/harness.py` | In-process episode harness |
| `experiments/run_experiment.py` | CLI runner (replaced stub) |
| `experiments/analyze_results.py` | Tables III–VI (replaced stub) |
| `tests/test_phase5.py` | Tests A–X |
| `results/final/**` | Smoke raw + tables + figure (n=2; **not** paper results) |
| `Phase5-Implementation-Report.md` | This report |
| `README.md` | Phase 4/5 status refresh |

---

## 10. Stop condition

Implementation, tests, and the **small deterministic smoke test only** are done.

- Do **not** treat `results/final/**` as experimental findings for the paper.
- Do **not** run the 1600-episode factorial unless explicitly requested later.
- Deliverable for commit (when asked):

```powershell
git add config/experiment/phase5_scenario.yaml experiments/phase5/ `
  experiments/run_experiment.py experiments/analyze_results.py `
  tests/test_phase5.py Phase5-Implementation-Report.md README.md
git commit -m "Phase 5: experiment harness, runner, analyzer, tests A–X, smoke only"
```
