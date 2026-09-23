# Phase 5 — Full Factorial Experiment Report

**Status:** COMPLETE — full 1600-episode factorial executed, validated, analyzed.
**Date:** 2026-09-22
**Environment:** Windows / Python 3.13 / in-process harness (no Docker consumed by the run).

---

## A. Scope

This report covers the Phase 5 full factorial experiment (1600 episodes). It is the companion to `Phase5-Implementation-Report.md` (implementation + smoke).

Frozen surfaces were **not** modified: `client/`, `tool_world/`, `pilot/key_schemes.py`, `pilot/wilson.py`, `reintegration/`, `kinbridge_math/`, `core/`, `config/experiment/factorial.yaml`, `config/experiment/ablation.yaml`, `results/phase1b/*`. Phase 1b hashes are immutability-checked (below).

## B. Exact command executed

```
python -m experiments.run_experiment --grid factorial --episodes 100 --seed 42 --harness inprocess --output-dir results/final/factorial_raw
```

Run metadata (`results/final/factorial_raw/run_meta.json`):

| Field | Value |
|---|---|
| started | 2026-09-22T22:17:19+05:30 |
| ended | 2026-09-22T22:18:58+05:30 |
| elapsed_s | 99.32 |
| exit_code | 0 |
| episodes_run | 1600 |
| jobs | 1600 |

## C. Grid

- Methods: `[cold_restart, naive_retry, verify_before_retry, kinbridge_sync]` × 4
- Outages (s): `[0, 10, 90, 300]` × 4
- Episodes per cell: **100** → 16 × 100 = **1600** episodes.
- Seed: master `42` (from `factorial.yaml`); child seeds `sha256(master:scenario:method:outage:episode) mod 2^63` (numpy applied mod 2^32).
- All cells: topology `same-node`, replication lag `0`, scenario `replan-dup-2`, ablation row `full`, key scheme `intent`.
- Cell order deterministic: method-major (`cold_restart → kinbridge_sync`), then outage asc, then episode `0..99`.

## D. Raw data integrity

Validated by `results/final/factorial_raw/_validate_raw.py` (result: **VALIDATION OK**):

| Check | Result |
|---|---|
| record count | 1600 (exactly 100 per method×outage cell) |
| duplicate episode keys | 0 (all 1600 unique) |
| episode ids per cell | exactly `0..99` |
| method/outage labels | only the 16 expected cells |
| master seed | `{42}` only |
| topology / lag | `{same-node}` / `{0}` only (no C3 contamination) |
| ablation row | `{full}` only (no ablation contamination) |
| scenario | `{replan-dup-2}` only |
| τ\* | `{0.5051826557580381}` in every record |
| τ\* source | `results/phase1b/tau_star_results.csv` present in provenance |
| failed/empty-action episodes | 0 |
| raw bytes | 2,495,045 |

## E. Immutability

- `results/final/factorial_raw/integrity_pre_run.json` captured SHA-256 of 18 files (7 Phase 1b + factorial/ablation/scenario YAML, `client/buffer.py`, `reintegration/*`, `tool_world/*`, `pilot/*`) before the run. Post-run recheck: **NONE changed** (Phase 1b and methodology intact).
- `results/phase1b/` contains exactly the 7 pre-existing files; no Phase 5 files added.

## F. Analysis

Command:
```
python -m experiments.analyze_results --input results/final/factorial_raw/episodes.jsonl --output-dir results/final --figures
```

Outputs into `results/final/` (raw smoke JSONL preserved in `results/final/raw/`, untouched):

| Artifact | Contents |
|---|---|
| `results.csv` | 1600 episode rows (episode_id, method, outage, profile, seeds, key_scheme, ablation_row) |
| `table_III.csv` | unsafe-action rate by method × outage (16 cells), Wilson 95% CI, median TTR |
| `table_IV.csv` | decomposition by method (duplicate/stale/escalation rates + CI), unsafe rate, median TTR, Cohen's h vs cold_restart, n_invalidated |
| `table_V.csv` | C3 lag × topology for kinbridge_sync — only `0ms/same-node` populated (100 eps, unsafe 0.0); other cells empty (C3 grid not run in the factorial per scope) |
| `table_VI.csv` | ablation ladder — all rows empty (ablation grid not part of the 1600; see implementation report) |
| `table_II.csv` | **NOT_AVAILABLE / PENDING** (see G) |
| `figures/unsafe_rate_vs_outage.png` | Table III figure |

### Headline Table III results (unsafe-action rate; CI = 95% Wilson)

| method / outage | 0 s | 10 s | 90 s | 300 s |
|---|---|---|---|---|
| cold_restart | 1.0000 [0.9812, 1.0000] | 0.5000 [0.4314, 0.5686] | 1.0000 [0.9812, 1.0000] | 1.0000 [0.9812, 1.0000] |
| naive_retry | 0.0000 [0.0000, 0.0188] | 0.0000 [0.0000, 0.0188] | 0.5000 [0.4314, 0.5686] | 0.5000 [0.4314, 0.5686] |
| verify_before_retry | 0.0000 [0.0000, 0.0188] | 0.0000 [0.0000, 0.0188] | 0.5000 [0.4314, 0.5686] | 0.5000 [0.4314, 0.5686] |
| kinbridge_sync | 0.0000 [0.0000, 0.0188] | 0.0000 [0.0000, 0.0188] | 0.0000 [0.0000, 0.0188] | 0.0000 [0.0000, 0.0188] |

### Table IV (per method, all 800 actions)

| method | duplicate | stale | escalation | **unsafe** | median TTR | Cohen's h vs cold | n_invalidated |
|---|---|---|---|---|---|---|---|
| cold_restart | 0.6250 | 0.2500 | 0.0000 | **0.8750** | 0.0000 | 0 | 0 |
| naive_retry | 0.0000 | 0.2500 | 0.0000 | **0.2500** | 0.0000 | −1.3717 | 0 |
| verify_before_retry | 0.0000 | 0.2500 | 0.0000 | **0.2500** | 0.0000 | −1.3717 | 0 |
| kinbridge_sync | 0.0000 | 0.0000 | 0.0000 | **0.0000** | 0.0000 | −2.4189 | 200 |

Interpretation notes (descriptive only, no claims beyond observed behavior):
- cold_restart duplicates (625/1000) and stales (250/1000) exactly as modeled by the duplicate/staleness structural construction at scale; 700/800 unsafe.
- naive_retry and verify_before_retry eliminate all duplicates but leave the same 250/1000 stale (replays of the gap/expired action are not blocked — recorded stale_execution).
- kinbridge_sync blocks both; invalidated actions (200) — **100% of the invalidated outcomes** — unsafe = 0/800. As predicted by the smoke/trace analysis and FMR/FSR model, the 90 s outage exceeds a2's 30 s TTL (intent expired) and the observed protective outcome was **invalidation** (200 invalidations, all at the 90 s and 300 s cells); **no escalation decisions occurred** (escalation rate 0.0 in every cell).
- The quantitative average performance of the analytic (200-record) expectation row (Section D of the implementation report) is **not** reported as results; only the empirical records above are claimed. The model trace correctly predicted outcome categories per cell (cold {1, .5, 1, 1}, naive/verify {0, 0, .5, .5}, kinbridge {0, 0, 0, 0}).

## G. Table II status — explicitly NOT AVAILABLE, not fabricated

Per the review directive: Table II is not produced with defensible evidence by the Phase 5 experiment (the observable TABLE I / structural-decomposition tables the pilot used to derive Table II were Phase 1/1b artifacts; the Phase 5 scenario `replan-dup-2` produces design-C observations only, and per instructions the reporter must not invent means/variances). `results/final/table_II.csv` therefore contains an explicit marker:

```
table_II,status,reason,source
NOT_AVAILABLE,PENDING,"Table II is the pilot-study table (Phase 1). This Phase 5 factorial run does not generate Table II observations...","results/pilot_summary.csv (Phase 1 pilot - not regenerated by this run)"
```

No fabricated values were written for Table II.

## H. Seed strategy & determinism

- Master seed 42; per-episode child seeds derived and recorded in every JSONL record (`seed`, `episode_seed`) so any cell can be re-run deterministically: `resolve_episode(scenario, method, outage, episode)` yields the same seed → same episode.
- Episodes are isolation-verified: per-episode `TemporaryDirectory`, no cross-episode state (harness unit tests + A–X 40-test suite).

## I. F1 / scalability limitation

Consistent with the implementation report: the F1 this run uses is a design-C fixed-architecture intent-2-intent evaluation. Baseline F1 usage is restricted to `decision_based` client policy gate opening for the reinitialize intent. E2E large-scale F1 and prefix-capacity/throughput scaling were out of scope for the 1600, performed as a separate C3-adjacent item only (Table V cell, kinbridge@90). No changes to the F1 decision surfaced during this run beyond the design-C caller already documented.

## J. Files added by the run (all under results/final/)

```
factorial_raw/   episodes.jsonl          (2,495,045 B) — raw data
                 results.csv              (244,736 B) — episode table
                 run_meta.json            (run provenance)
                 integrity_pre_run.json   (18-file SHA-256 pre-run manifest)
                 _plan_check.py           (grid plan verification; kept as reproducible evidence)
                 _validate_raw.py         (raw-data validator; kept)
                 _integrity_pre.py        (manifest generator; kept)
results.csv      (final episode table, copied from raw by analyzer)
table_II.csv     NOT_AVAILABLE / PENDING (explicit)
table_III.csv    16 cells with CIs
table_IV.csv     4 methods with Cohen's h
table_V.csv      C3 — 0ms/same-node only
table_VI.csv     ablation — empty (not in factorial)
figures/unsafe_rate_vs_outage.png
```

Smoke-only data (`results/final/raw/`, 2 episodes) left untouched.

## K. Test results after the run

```
pytest tests/test_phase5.py        → 40 passed
pytest tests/ (full suite, Docker up) → 563 passed, 1 failed  [see L]
```

## L. Known pre-existing failure (NOT a Phase 5 regression)

`tests/test_phase3_compose.py::TestComposeCLI::test_compose_up_and_down` fails: it asserts `running == 3` but the Phase 3b compose stack serves 6 services (`client, edge-a, edge-b, redis-a, redis-b, replicator`) — `docker compose config --services` confirms 6. During prior CI runs Docker was unavailable so this test was `SKIPPED`; once Docker Desktop is running the test executes and its hardcoded 3-service expectation (from the Phase 3a-era stack) is stale. The test file is a frozen Phase 3b test surface and was **not modified**. Phase 5 added no compose services.

## M. Methodology / reproducibility guarantees honored

- τ\* = 0.5051826557580381 loaded from `results/phase1b/tau_star_results.csv`, never hardcoded; provenance in every record.
- Raw records not fabricated/trimmed to hit 1600 — 1600 exactly present, verified.
- No scoring/ranking framing of methods in this report; tables report observed rates + CIs only.
- No free-form per-method prose conclusions added beyond the observed empirical values and their model-predicted category match.
- No files under `results/phase1b`, `config/experiment/factorial.yaml`, `config/experiment/ablation.yaml`, `client/`, `tool_world/`, `reintegration/`, `pilot/`, `kinbridge_math/`, `core/` were changed during the run (integrity manifest clean).

## N. Reproducibility commands

```powershell
# 1) re-run grid (fresh dir to avoid append-mixing):
python -m experiments.run_experiment --grid factorial --episodes 100 --seed 42 --harness inprocess --output-dir results/final/factorial_raw

# 2) re-validate raw:
python results/final/factorial_raw/_validate_raw.py

# 3) re-analyze:
python -m experiments.analyze_results --input results/final/factorial_raw/episodes.jsonl --output-dir results/final --figures

# 4) tests:
python -m pytest tests/test_phase5.py -q
python -m pytest tests/ -q
```

## O. Stop condition

Phase 5 full-factorial delivery is complete. The run produced the expected raw volume with full integrity. The one failing test is the documented pre-existing Phase 3b compose expectation. All Phase 5 tests pass. Stopping here per the "STOP" condition — no further experiment execution without a new directive.