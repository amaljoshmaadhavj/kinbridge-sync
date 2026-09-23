# Kinbridge-Sync

## Replay Is Not Retry: Action Safety for Stateful LLM Agents Across Connectivity Gaps at the Mobile Edge

### Research Objective

Kinbridge-Sync investigates the safe execution and reconciliation of stateful Large Language Model (LLM) agent actions across intermittent connectivity gaps at the mobile edge. When network partitions or high-latency outages occur, standard retry policies risk duplicate execution of non-idempotent actions, while naive local replay without state synchronization risks safety violations. This research formalizes and empirically evaluates replay and retry safety, semantic intent merging, action buffering, and stateful reintegration mechanisms under realistic edge connectivity degradation.

---

## Current Implementation Status

| Phase | Description | Status |
|---|---|---|
| **Phase 1** | Pilot study (19 prompt pairs, key schemes, LLM divergence) | **COMPLETE / FROZEN** |
| **Phase 1b** | Mathematical calibration pipeline (Design C negatives, $\tau^*$ calibration) | **COMPLETE / FROZEN** |
| **Phase 2.1** | Tool-world FastAPI service + SQLite effect ledger | **COMPLETE / FROZEN** |
| **Phase 2.2** | Client Action Buffer (JSONL WAL) + Link Monitor (EWMA & hysteresis FSM) | **COMPLETE / FROZEN** |
| **Phase 3a** | Containerization and network emulation testbed (Docker Compose, `netctl`, Gilbert–Elliott) | **COMPLETE / FROZEN** |
| **Phase 3b** | Replication-lag forwarder between edge nodes | **COMPLETE / FROZEN** |
| **Phase 4** | Reintegration service (Algorithm 1 + baselines) | **COMPLETE / FROZEN** |
| **Phase 5** | Full factorial experiment (1600 episodes) | **COMPLETE / VERIFIED** — full 1600-episode grid run and independently audited (Phase 6A: PASS WITH QUALIFICATIONS) |

Phase 5 delivers the in-process episode harness, `run_experiment` / `analyze_results`, scenario/τ*/ablation/C3 modules, and tests A–X (`tests/test_phase5.py`). The full 1600-episode factorial was executed (`results/final/factorial_raw/episodes.jsonl`) and independently recomputed in Phase 6A. Implementation details: `Phase5-Implementation-Report.md`; results and analysis: `Phase5-Factorial-Experiment-Report.md`, `results/final/table_III.csv` … `table_VI.csv`.

---

## Phase 1 & 1b: Empirical Pilot and Mathematical Calibration

### Pilot Study Methodology (Phase 1)
- **Action Prompts**: 19 curated prompt pairs across three tool domains (`navigate_to`, `log_inspection`, `request_supply_drop`), forming 19 positive reissue pairs ($a_1 \leftrightarrow a_2$).
- **Key Schemes Evaluated**: Payload hash ($k_{\text{payload}}$), Time-bucket hash ($k_{\text{bucket}}$), and Intent key ($k_{\text{intent}}$).
- **Execution Conditions**: Temperatures $T \in \{0.0, 0.7\}$ with deterministic seed `42`.
- **Inference Model**: `qwen2.5:7b-instruct` served via Ollama.

### Design C Negative-Trial Methodology (Phase 1b)
The original 19 reissue pairs contain only same-intent executions. To establish an unbiased False Merge Rate without data contamination, Phase 1b introduces **Design C (dedicated independent negatives)**:
- **Dedicated Negative Trials**: 200 negative trials per temperature (400 total trials across $T=0.0$ and $T=0.7$, comprising 800 distinct model actions).
- **Intent Pair Balancing**: Distributed evenly across all $\binom{13}{2} = 78$ possible distinct-intent combinations.
- **Trial Validation**: Enforces different intents, verified execution success, unique model invocation IDs (no response reuse), and retry-until-success logic to guarantee exactly 200 valid observations per temperature.
- **Independent Data Sources**: False Split Rate (FSR) and False Merge Rate (FMR) are computed from completely disjoint datasets and never conflated.

### Error Metric Definitions
- **False Split Rate ($\text{FSR}(\tau)$)**: Proportion of same-intent reissue pairs whose semantic cosine similarity falls below threshold $\tau$ (incorrectly classified as distinct intents):
  $$\text{FSR}(\tau) = \frac{|\{\text{same-intent pairs with similarity} < \tau\}|}{19}$$
  Evaluated strictly against the 19 reissue pairs (denominator = 19).
- **False Merge Rate ($\text{FMR}(\tau)$)**: Proportion of distinct-intent negative pairs whose semantic cosine similarity meets or exceeds threshold $\tau$ (incorrectly merged):
  $$\text{FMR}(\tau) = \frac{|\{\text{different-intent pairs with similarity} \ge \tau\}|}{200}$$
  Evaluated strictly against the 200 dedicated negative trials (denominator = 200).
- **Confidence Intervals**: Wilson 95% binomial score intervals computed for each point across the 101-point threshold sweep ($\tau \in [0.00, 1.00]$).

### Corrected Power-Law Formulation
Empirical calibration fits the following monotonic power-law relationships:
$$\text{FSR}(\tau) \approx \tau^\beta \quad (\text{monotonically increasing in } \tau)$$
$$\text{FMR}(\tau) \approx (1 - \tau)^\alpha \quad (\text{monotonically decreasing in } \tau)$$

Parameters $\alpha$ and $\beta$ are fitted via linear regression in log-log space ($\log \text{FMR}$ vs. $\log(1-\tau)$, and $\log \text{FSR}$ vs. $\log \tau$).

### Risk Formulation and Optimal Threshold $\tau^*$
The total decision risk $R(\tau)$ balances duplicate action costs against missed intent merge opportunities:
$$R(\tau) = w_d \cdot \text{FSR}(\tau) + w_m \cdot \text{FMR}(\tau) + w_s \cdot P_{\text{stale}}$$
Standard paper weights configure $w_d = 4.0$ (duplicate execution penalty), $w_m = 1.0$ (missed merge penalty), and $w_s = 0.0$ ($P_{\text{stale}} = 0.0$).

The first derivative is:
$$R'(\tau) = w_d \cdot \beta \cdot \tau^{\beta - 1} - w_m \cdot \alpha \cdot (1 - \tau)^{\alpha - 1}$$

- **Closed-Form Solution** (applicable when $|\alpha - \beta| < 0.05$):
  $$\tau^* = \frac{1}{1 + \rho}, \quad \text{where } \rho = \left(\frac{w_d}{w_m}\right)^{\frac{1}{\alpha - 1}}$$
- **Bisection Search**: When $\alpha \neq \beta$, $\tau^*$ is located by bisecting $R'(\tau) = 0$ over the interval $[0.001, 0.999]$.

### Verified Calibration Results (from `results/phase1b/`)
- **Temperature 0.0**: $\alpha = 1.665$ ($R^2 = 0.954$). Parameter $\beta$ is unestimable because zero non-identical reissues occurred ($\text{FSR} = 0$ across all $\tau$), rendering $\tau^*$ unestimable at $T=0.0$.
- **Temperature 0.7**: $\alpha = 1.591$ ($R^2 = 0.958$), $\beta = 5.440$ ($R^2 = 0.206$). Bisection yields an optimal operating threshold of **$\tau^* = 0.505$** with risk $R(\tau^*) = 0.334$.

---

## Phase 5: Verified Full-Factorial Experiment (1600 Episodes)

### Experiment configuration

| Parameter | Value |
|---|---|
| Methods | `cold_restart`, `naive_retry`, `verify_before_retry`, `kinbridge_sync` |
| Outages | deterministic 0 / 10 / 90 / 300 s fixed-duration conditions |
| Episodes | 100 per method × outage cell = **1600 total** |
| Seed | master `42`, deterministic per-episode child seeds |
| Scenario | `replan-dup-2` |
| Topology | same-node |
| Replication lag | 0 ms |
| Key scheme | `intent` |
| Temperature | 0.7 |
| τ\* | imported from Phase 1b calibration (0.5051826557580381); **Phase 5 does not calibrate τ\* itself** |

Raw records: `results/final/factorial_raw/episodes.jsonl`.

### Table III — unsafe-action rate (method × outage), Wilson 95% CI

`unsafe = duplicate OR stale_execution` (escalation alone is not unsafe).

| method / outage | 0 s | 10 s | 90 s | 300 s |
|---|---|---|---|---|
| `cold_restart` | 1.0000 [0.9812, 1.0000] | 0.5000 [0.4314, 0.5686] | 1.0000 [0.9812, 1.0000] | 1.0000 [0.9812, 1.0000] |
| `naive_retry` | 0.0000 [0.0000, 0.0188] | 0.0000 [0.0000, 0.0188] | 0.5000 [0.4314, 0.5686] | 0.5000 [0.4314, 0.5686] |
| `verify_before_retry` | 0.0000 [0.0000, 0.0188] | 0.0000 [0.0000, 0.0188] | 0.5000 [0.4314, 0.5686] | 0.5000 [0.4314, 0.5686] |
| `kinbridge_sync` | 0.0000 [0.0000, 0.0188] | 0.0000 [0.0000, 0.0188] | 0.0000 [0.0000, 0.0188] | 0.0000 [0.0000, 0.0188] |

### Table IV — decomposition by method (800 actions per method)

| method | duplicate | stale | escalation | **unsafe** | median TTR | n_invalidated |
|---|---|---|---|---|---|---|
| `cold_restart` | 0.6250 | 0.2500 | 0.0000 | **0.8750** | 0.0 | 0 |
| `naive_retry` | 0.0000 | 0.2500 | 0.0000 | **0.2500** | 0.0 | 0 |
| `verify_before_retry` | 0.0000 | 0.2500 | 0.0000 | **0.2500** | 0.0 | 0 |
| `kinbridge_sync` | 0.0000 | 0.0000 | 0.0000 | **0.0000** | 0.0 | 200 |

Cohen's h vs `cold_restart`: `cold_restart` 0.0, `naive_retry` −1.3716608546, `verify_before_retry` −1.3716608546, `kinbridge_sync` −2.4188584058.

### Observed empirical findings

- In the tested configuration and scenario, `kinbridge_sync` recorded **zero duplicate and zero stale executions** across all four deterministic outage conditions (0/800 unsafe, 95% Wilson CI [0.0000, 0.0188]), with 200 invalidations.
- In the factorial experiment, the observed protective outcome was **invalidation**; **no escalation decisions occurred** (escalation rate 0.0000 in every cell), so the escalation path remains unexercised by this experiment.
- `cold_restart` recorded 700/800 unsafe (500 duplicates, 200 stale executions); `naive_retry` and `verify_before_retry` each recorded 200/800 unsafe (all stale executions, no duplicates).
- Under the tested same-node, zero-replication-lag configuration, invalidation of `kinbridge_sync` actions occurred only at the 90 s and 300 s cells (100 each, all action `a2`, which carries TTL 30 s), matching the expiry explanation.
- **TTR was not meaningfully exercised by the in-process harness; all recorded TTR values were 0.0.** The median TTR of 0.0 is therefore a harness artifact of unexercised recovery-time measurement, not evidence of instant recovery.
- Outages were **deterministic 0/10/90/300-second fixed-duration conditions**; the factorial did not use stochastic Gilbert–Elliott outage sampling.

### Model vs empirical distinction

Quantitative rates above are empirical (recomputed from the 1600 records in Phase 6A). Model-derived artifacts — the Phase 1b τ\* calibration and model-trace predictions — are distinct. The empirical cell categories matched the model trace predictions (cold {1, .5, 1, 1}; naive/verify {0, 0, .5, .5}; kinbridge {0, 0, 0, 0}); this is reported as agreement with the model trace, not independent validation of every theoretical assumption.

### Tables II, V, VI — not fabricated

- **Table II**: retained as **explicitly unavailable/pending** (`results/final/table_II.csv`); not produced by this experiment.
- **Table V** (C3): only the **0 ms / same-node** factorial cell is populated (100 eps, unsafe 0.0). The broader C3 lag × topology grid remains **unmeasured**.
- **Table VI** (ablation): the ablation ladder was **not part of the 1600-episode factorial**; all rows remain empty. No results were substituted.

### Limitations

1. Same-node topology only.
2. Replication lag = 0 ms.
3. C3 lag/topology grid not completed by the factorial.
4. Ablation grid not included in the 1600 episodes.
5. Table II unavailable from this experiment.
6. TTR not meaningfully measured (all 0.0; harness artifact).
7. Escalation path not exercised (no escalation decisions occurred).
8. Outages were deterministic fixed-duration conditions, not stochastic Gilbert–Elliott sampling.
9. τ\* was imported from Phase 1b calibration rather than independently calibrated in Phase 5.
10. Results are for the `replan-dup-2` scenario and should not be generalized beyond the tested scenario without additional evidence.

---

## Phase 2: Plumbed Edge and Client Components

### Phase 2.1 — Tool World (`tool_world/`)
- **FastAPI Application**: Implements standard operational endpoints (`/tools/navigate_to`, `/tools/log_inspection`, `/tools/request_supply_drop`), administrative health probes (`/health`), and ledger query interfaces (`/ledger/lookup`, `/ledger/effects`).
- **SQLite Effect Ledger (`tool_world/ledger.py`)**: ACID-compliant persistence tracking action life cycles:
  - Strict state machine: `INIT` $\to$ `COMMITTED`, `ABORTED`, `INVALIDATED`.
  - Idempotency enforcement via unique action keys preventing duplicate effect application.
  - Explicit epoch isolation (`set_epoch`) supporting reconnect validation and split-brain mitigation.

### Phase 2.2 — Action Buffer and Link Monitor (`client/`)
- **Action Buffer (`client/buffer.py`)**:
  - Append-only JSONL Write-Ahead Log (WAL) accompanied by an atomic `.idx.json` status index.
  - In-place crash recovery across restarts without mutating historical log entries.
  - Thread-safe access via internal synchronization locks.
  - Monotonic epoch tracking (`max_epoch`) and status updating (`PENDING`, `COMMITTED`, `INVALIDATED`).
- **Link Monitor (`client/link_monitor.py`)**:
  - Jacobson/Karels EWMA estimator computing smoothed round-trip time ($SRTT$) and round-trip variance ($RTTVAR$).
  - Three-state hysteresis Finite State Machine (FSM): `ONLINE`, `DEGRADED`, and `DISCONNECTED`.
  - Probing frequency and consecutive failure thresholds configured to prevent rapid state flapping during transient link degradation.

---

## Phase 3a: Containerization and Network Emulation Testbed

Phase 3a establishes a reproducible Linux networking environment for injecting controllable network anomalies. Docker-based validation has been completed.

- **Docker Container Image (`Dockerfile`)**:
  - Multi-stage build based on `python:3.11-slim`.
  - Packages essential Linux traffic-control and inspection utilities (`tc`, `iptables`, `iproute2`, `iputils-ping`, `curl`).
  - Bundles frozen Phase 2 packages (`tool_world`, `client`) and shared modules.
  - Image built and validated as `kinbridge-sync:phase3a`.
- **Docker Compose Network Testbed (`docker-compose.yml`)**:
  - Deploys a three-node topology: `client`, `edge-a` (primary edge), and `edge-b` (secondary edge).
  - Attached to an isolated bridge network (`kinbridge-net`).
  - Containers configured with `NET_ADMIN` Linux capabilities required for kernel traffic manipulation.
- **Network Controls (`experiments/netctl.py`)**:
  - Programmatic and CLI controls driving `tc netem` to inject asymmetric delay, jitter, packet loss, and bandwidth constraints.
  - Packet filtering and interface partitioning via `iptables`.
  - Automatic rollback and reset functionality.
- **Outage Model (`experiments/outage_model.py`)**:
  - Discrete-time Gilbert–Elliott two-state Markov chain (GOOD and BAD link states).
  - Configured via transition probabilities $p_{GB}$ and $p_{BG}$ to emulate empirical mobile edge disconnection intervals.
  - Directly drives `netctl.py` for automated outage injection.

---

## Architecture Status: Phase 3b — SQLite→Redis Replication (Validated)

Phase 3b (Replication-Lag Forwarder between edge nodes) is **implemented and end-to-end validated** in Docker (`Phase3b-Validation-Report.md`; compose stack of 6 services: `client`, `redis-a`, `redis-b`, `edge-a`, `edge-b`, `replicator`).

Architecture:
- **SQLite remains authoritative on both nodes** (edged ledgers).
- **Redis is asynchronous event transport** only — not a source of truth.
- **`sqlite_watcher`** publishes edge-A SQLite changes to Redis.
- **`replicator`** forwards Redis events to edge-B with **configurable lag** (0 ms, 500 ms, or "never").
- **`sqlite_writer`** applies received events to edge-B SQLite (idempotent ON CONFLICT).

Known replication-loss limitation (documented, not production-grade guarantees):
- If **Redis-B is unavailable while an event is produced on edge-A**, the event can be lost: the current forwarder acknowledges (ACKs) on Redis-A at read time, before successful forwarding to Redis-B is confirmed. This is an experimental-testbed limitation; a production system would require a transactional outbox or XCLAIM-based retry.
- Do not interpret Phase 3b or the Phase 5 factorial as providing production-grade replication guarantees.

---

## Test Suite & Verification

All reported test counts are directly verified from the current repository test suite (553 tests total):

| Component / Subsystem | Test Module | Test Count |
|---|---|---|
| **Phase 1 & 1b: Pilot & Math** | `tests/test_pilot.py` | 112 |
| **Phase 1b: Negative Trials Mock** | `pilot/tests/test_negative_trials_mock.py` | 27 |
| **Phase 1: Key Schemes** | `pilot/test_bucket_key.py` | 2 |
| **Phase 2.1: Tool World & Ledger** | `tests/test_phase2_1.py` | 39 |
| **Phase 2.2: Action Buffer** | `tests/test_buffer.py` | 18 |
| **Phase 2.2: Link Monitor** | `tests/test_link_monitor.py` | 42 |
| **Phase 3a: Docker Container Image** | `tests/test_docker_image.py` | 26 |
| **Phase 3a: Docker Compose Testbed** | `tests/test_phase3_compose.py` | 49 |
| **Phase 3a: Network Control (`netctl`)** | `tests/test_netctl.py` | 63 |
| **Phase 3a: Outage Model** | `tests/test_outage_model.py` | 58 |
| **Phase 3b: Replicator** | `tests/test_replicator.py` | 14 |
| **Phase 3b: SQLite Watcher** | `tests/test_sqlite_watcher.py` | 14 |
| **Phase 3b: SQLite Writer** | `tests/test_sqlite_writer.py` | 15 |
| **Phase 4: Reintegration Service** | `tests/test_reintegration_service.py` | 34 |
| **Phase 5: Factorial Harness** | `tests/test_phase5.py` | 40 |
| **Total Test Suite** | *15 test modules* | **553** |

---

## Technology Stack

The repository exclusively incorporates the following technologies:
- **Core Runtime**: Python 3.11+
- **API & Web Service**: FastAPI, Uvicorn, Pydantic, HTTPX
- **Data Persistence**: SQLite (ACID WAL mode), JSONL (append-only write-ahead log), PyYAML
- **Containerization & Orchestration**: Docker, Docker Compose
- **Network Emulation**: Linux Traffic Control (`tc` / `netem`), `iptables`, `iproute2`
- **Scientific Computing & Statistics**: NumPy, SciPy, Pandas, Statsmodels
- **Machine Learning & NLP**: PyTorch, Sentence-Transformers (`all-MiniLM-L6-v2`)
- **LLM Inference (Pilot)**: Ollama (`qwen2.5:7b-instruct`, `qwen2.5:1.5b-instruct`)
- **Testing Framework**: Pytest

---

## Repository Structure

```
kinbridge-sync/
├── client/                     # Phase 2.2 client-side edge components
│   ├── buffer.py               # Action Buffer: append-only JSONL WAL + index
│   └── link_monitor.py         # Link Monitor: Jacobson/Karels EWMA + FSM
├── config/                     # Configuration parameters (YAML)
│   ├── experiment/             # Factorial and ablation configuration
│   ├── outage/                 # Gilbert–Elliott model parameters
│   ├── pilot/                  # Pilot prompts and study parameters
│   ├── testbed/                # Docker network topology configuration
│   └── tools/                  # Tool schemas
├── core/                       # Shared utilities (config loader, seed control)
├── experiments/                # Phase 3a network emulation & testbed tools
│   ├── netctl.py               # Linux tc/netem and iptables control
│   ├── outage_model.py         # Gilbert–Elliott Markov model
│   ├── run_experiment.py       # Experiment runner entrypoint
│   └── analyze_results.py      # Results analysis
├── kinbridge_math/             # Phase 1b mathematical modeling & calibration
│   ├── calibrate.py            # Design C calibration entrypoint
│   ├── fmr_from_negatives.py   # Dedicated negative trial FMR/FSR processing
│   ├── fit_alpha_beta.py       # Power-law regression fitting
│   ├── tau_star.py             # Risk function, bisection, and closed-form tau*
│   ├── degradation_threshold.py# Quality gap thresholding
│   ├── measure_delta_q.py      # SLM quality gap evaluation
│   └── validate_tau.py         # Threshold sweep evaluation
├── pilot/                      # Phase 1 pilot study implementation
│   ├── analyze_pilot.py        # Pilot divergence analysis
│   ├── fmr_fsr.py              # Error rate calculation
│   ├── key_schemes.py          # Idempotency key generators (payload, bucket, intent)
│   ├── pairs.py                # Pairwise comparison generation
│   ├── prompts.json            # Pilot prompts
│   ├── run_negative_trials.py  # Design C negative trials generator
│   ├── run_pilot.py            # Pilot prompt runner
│   ├── similarity.py           # Sentence-transformers action cosine similarity
│   ├── tools.py                # Tool definitions
│   └── wilson.py               # Wilson score confidence intervals
├── results/                    # Preserved experimental outputs & figures
│   ├── phase1b/                # Calibrated curves, summary JSON, and fits
│   ├── raw/                    # Raw pilot execution outputs
│   └── raw_fmr/                # Raw negative trials outputs
├── tests/                      # Repository test suite (553 tests / 15 modules)
│   ├── test_buffer.py
│   ├── test_docker_image.py
│   ├── test_link_monitor.py
│   ├── test_netctl.py
│   ├── test_outage_model.py
│   ├── test_phase2_1.py
│   ├── test_phase3_compose.py
│   ├── test_phase5.py
│   ├── test_pilot.py
│   ├── test_reintegration_service.py
│   ├── test_replicator.py
│   ├── test_sqlite_watcher.py
│   ├── test_sqlite_writer.py
├── Dockerfile                  # Unified Phase 3a container image definition
├── docker-compose.yml          # Three-node network testbed topology
├── requirements.txt            # Project dependencies
└── README.md
```

---

## Research Integrity & Reproducibility Guarantees

- **Frozen Empirical Baselines**: Raw experimental outputs from Phase 1 and Phase 1b stored in `results/` and `pilot/raw/` are preserved artifacts. Downstream implementation phases must not alter frozen experimental methodology, raw trial logs, or fitted mathematical constants.
- **Deterministic Randomness**: All stochastic operations (negative trial sampling; the Phase 3a Gilbert–Elliott outage model's Markov transitions) are governed by explicit seeds (`core.seeds.seed_everything(seed=42)`). The Phase 5 factorial did **not** sample stochastic GE trajectories: it used deterministic 0/10/90/300-second fixed-duration outage conditions instead.
- **Configuration Driven**: All execution parameters reside exclusively in `config/*.yaml` files, avoiding hard-coded parameters.
- **Decoupled Workflows**: Execution runners and analysis scripts remain strictly separated; analysis tools never mutate raw experiment data.

---

## Setup & Testing

### Environment Setup
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Running Tests
```powershell
pytest
```

To run specific subsystem test suites:
```powershell
# Phase 1 & 1b tests
pytest tests/test_pilot.py pilot/tests/

# Phase 2 tests
pytest tests/test_phase2_1.py tests/test_buffer.py tests/test_link_monitor.py

# Phase 3a testbed tests
pytest tests/test_docker_image.py tests/test_phase3_compose.py tests/test_netctl.py tests/test_outage_model.py
```

### Configuration Loading
Configuration files are accessed through the shared loader:
```python
from core.config import load_config
cfg = load_config("pilot/pilot_config.yaml")  # relative to config/
```

---

## License

Research use only.

