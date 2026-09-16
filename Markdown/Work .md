# PART A — AMAL
## Your responsibility

You will own:
1. Environment/setup
2. Pilot experiment
3. Semantic re-issue/C1
4. Similarity measurement
5. FMR/FSR calculation
6. Mathematical calibration
7. **Novel τ* derivation and implementation**
8. Risk function implementation
9. τ* validation
10. Δq measurement
11. Degradation threshold
12. Statistical analysis
13. Final result generation
14. Updating the mathematical/results portions of the paper

You will use:

* **OpenCode** → implementation, scripts, mathematics, experiments
* **Antigravity** → review/debug/verify your implementation where useful

Santhosh will independently build the systems side using **Antigravity**.

---

# AMAL — PHASE 0

# Environment + repository

## Goal

Create the common repository and verify that your machine can run the entire LLM/pilot/math stack.

---

## 0.1 Create repository

```powershell
mkdir kinbridge-sync
cd kinbridge-sync

git init
```

Recommended initial structure:

```text
kinbridge-sync/
│
├── pilot/
├── math/
├── client/
├── tool_world/
├── reintegration/
├── testbed/
├── experiments/
├── results/
├── paper/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 0.2 Python

Create environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install:

```powershell
pip install fastapi uvicorn
pip install numpy pandas scipy statsmodels
pip install matplotlib
pip install sentence-transformers
pip install requests pytest
```

Then:

```powershell
pip freeze > requirements.txt
```

---

# 0.3 Ollama

Install the two models required by the paper:

```powershell
ollama pull qwen2.5:7b-instruct
ollama pull qwen2.5:1.5b-instruct
```

Verify:

```powershell
ollama run qwen2.5:7b-instruct
```

and:

```powershell
ollama run qwen2.5:1.5b-instruct
```

---

# 0.4 OpenCode

Configure OpenCode to use:

```text
Ollama
↓
qwen2.5:7b-instruct
```

Do not use a cloud API for the core pilot if the experiment is supposed to reproduce the paper's local setup.

---

# 0.5 First OpenCode prompt

Give OpenCode this:

```text
You are implementing the experimental validation pipeline for the Kinbridge-Sync research paper.

IMPORTANT:
Complete-Research.md is the authoritative specification.
Guide.md and Runbook.md are implementation references.

Do NOT invent experimental results.
Do NOT hard-code expected results.
Do NOT simplify or replace the mathematical formulation.

Create the repository structure for:

pilot/
math/
results/
experiments/

The implementation must preserve reproducibility:
- deterministic random seeds where applicable
- configuration files instead of hard-coded experiment parameters
- CSV/JSON raw outputs
- separate analysis scripts
- no mock experimental values

Create:
requirements.txt
.gitignore
README.md

Do not implement the research algorithm yet.
First create only the project skeleton and configuration system.
```

---

# AMAL — PHASE 1

# C1 Semantic Re-Issue Pilot

This is your first **real experiment**.

The paper's C1 claim is that an LLM can regenerate the same real-world intent with different serialized arguments, causing ordinary byte/payload-based idempotency to fail. 

---

# 1.1 Implement the three tools

Create:

```text
pilot/tools.py
```

Tools:

```python
navigate_to(...)
```

```python
log_inspection(...)
```

```python
request_supply_drop(...)
```

Use the exact three tool classes specified by the research plan.

---

# 1.2 Create ground-truth intent dataset

Create:

```text
pilot/prompts.json
```

Each entry should contain:

```json
{
    "prompt_id": "NAV_001",
    "tool_class": "navigation",
    "intent_id": "intent_nav_001",
    "prompt": "Navigate to ..."
}
```

The crucial thing:

### `intent_id` is ground truth.

The LLM must **not** generate it.

It tells your evaluation code whether two calls represent the same underlying intent.

---

# 1.3 Generate first action

For each prompt:

```text
prompt
 ↓
Qwen 2.5 7B
 ↓
tool call a1
```

Save:

```text
a1
```

---

# 1.4 Simulate the gap

Then generate:

```text
same task/context
+
re-issued request
 ↓
Qwen 2.5 7B
 ↓
a2
```

Save:

```text
a2
```

Do not simply duplicate `a1`.

The purpose is to allow the model to regenerate the action.

---

# 1.5 Record the raw model output

Never overwrite the raw response.

Store:

```text
pilot/raw/
```

For example:

```text
pilot/raw/NAV_001_run_001.json
pilot/raw/NAV_001_run_002.json
```

This gives you an audit trail.

---

# AMAL — 1.6 Implement the three key schemes

This is important because these keys are part of the C1 experiment.

## Payload key

```python
k_payload = hash(
    tool + canonical_args
)
```

The paper's conceptual construction is based on the serialized action/payload.

---

## Timestamp bucket key

Implement:

$$
k_{bucket}
=
H(agent\_id\Vert action\_type\Vert payload
\Vert \lfloor t/\Delta\rfloor)
$$

with:

$$
\Delta=60s
$$

---

## Intent key

Implement:

$$
k_{intent}
=
H(session\_id\Vert turn\_seq\Vert intent\_id)
$$

This is the proposed semantic/intent-scoped mechanism. 

---

# AMAL — 1.7 Deterministic bucket proof test

Create:

```text
pilot/test_bucket_key.py
```

Test:

```text
Δ = 60
G = 90
```

Show that:

$$
\left\lfloor\frac{t+90}{60}\right\rfloor
\neq
\left\lfloor\frac{t}{60}\right\rfloor
$$

for the selected test condition.

Then:

$$
k_{bucket}(t+90)
\neq
k_{bucket}(t)
$$

even when the payload is identical.

This is a mathematical/deterministic sanity check, not an LLM result. The paper explicitly describes this as a consequence of timestamp bucketing. 

---

# AMAL — 1.8 Semantic similarity

This is where I want you to make the implementation stronger than a simplistic pilot.

Use:

```text
sentence-transformers
all-MiniLM-L6-v2
```

Represent an action in normalized form.

For example:

```text
tool=request_supply_drop
lat=45.231
lon=6.887
payload=medical
priority=urgent
```

Generate:

$$
\phi(a)
$$

Then calculate:

$$
sim(a_1,a_2)
=
\frac{\phi(a_1)\cdot\phi(a_2)}
{\|\phi(a_1)\|\|\phi(a_2)\|}
$$

Store the actual similarity.

---

# AMAL — 1.9 Build positive and negative pairs

You need two classes.

### Positive

```text
same intent
```

Example:

```text
a1(intent_001)
a2(intent_001)
```

### Negative

```text
different intent
```

Example:

```text
intent_001
vs
intent_002
```

This allows you to calculate both sides of the threshold tradeoff.

---

# AMAL — 1.10 FMR

For a threshold τ:

$$
FMR(\tau)
=
\frac{
\#\{\text{different intents with similarity}\geq\tau\}
}{
\#\{\text{different-intent pairs}\}
}
$$

Interpretation:

> False Match Rate — two different actions are incorrectly considered the same.

---

# AMAL — 1.11 FSR

For a threshold τ:

$$
FSR(\tau)
=
\frac{
\#\{\text{same intents with similarity}<\tau\}
}{
\#\{\text{same-intent pairs}\}
}
$$

Interpretation:

> False Separation Rate — two representations of the same intent fail to match.

---

# AMAL — 1.12 Threshold sweep

Do not test only one τ.

Run:

```text
0.00
0.01
0.02
...
0.99
1.00
```

or an equivalent sufficiently fine grid.

Produce:

```text
tau
FMR
FSR
```

This gives the empirical tradeoff curve.

---

# AMAL — 1.13 Wilson CI

For every measured proportion:

$$
\hat p=\frac{x}{n}
$$

use:

$$
z=1.96
$$

and:

$$
center=
\frac{\hat p+z^2/(2n)}
{1+z^2/n}
$$

$$
margin=
\frac{
z\sqrt{
\frac{\hat p(1-\hat p)}{n}
+
\frac{z^2}{4n^2}
}
}{
1+z^2/n
}
$$

Then:

$$
CI=[center-margin,\ center+margin]
$$

This formula is explicitly required by the research plan. 

Implement it yourself in:

```text
math/statistics.py
```

rather than relying blindly on a library.

Then validate your implementation against `statsmodels`.

---

# AMAL — 1.14 Pilot output

You should have:

```text
pilot/
├── tools.py
├── prompts.json
├── run_pilot.py
├── similarity.py
├── analyze_pilot.py
├── test_bucket_key.py
│
├── raw/
│
└── pilot_results.csv
```

And:

```text
results/
├── pilot_table.csv
└── fmr_fsr.csv
```

---

# AMAL — PHASE 1 DECISION RULE

The paper has a predefined rule:

```text
failure rate > 15%
        ↓
C1 headline

5–15%
        ↓
C1 secondary

<5%
        ↓
C1 not headline
```

Follow the measured result.

Do not modify the experiment because the result is inconvenient. The paper explicitly instructs the team to use this predetermined rule. 

---

# AMAL — PHASE 1b

# THE NOVEL MATHEMATICAL CONTRIBUTION

This is one of the most important parts of your responsibility.

The paper doesn't merely say:

> "choose a threshold."

It proposes a **risk-minimizing threshold**.

You must implement that mathematical model.

---

# 2.1 Fit FSR

The model assumes:

$$
FSR(\tau)\approx(1-\tau)^\beta
$$

Take logs:

$$
\log FSR
=
\beta\log(1-\tau)
$$

Therefore:

$$
\beta
=
\text{slope}
$$

of:

$$
x=\log(1-\tau)
$$

versus:

$$
y=\log(FSR)
$$

---

# 2.2 Fit FMR

Similarly:

$$
FMR(\tau)\approx\tau^\alpha
$$

Therefore:

$$
\log FMR
=
\alpha\log\tau
$$

so:

$$
\alpha
=
\text{slope}
$$

of:

$$
x=\log\tau
$$

versus:

$$
y=\log(FMR)
$$

This is the exact procedure specified by Algorithm 2. 

---

# 2.3 Don't hide bad fits

For each tool class calculate:

```text
alpha
beta
R²(alpha)
R²(beta)
```

Also record:

```text
alpha > 1
beta > 1
```

The paper's convexity argument assumes α and β are greater than 1. 

Therefore your code must explicitly check that condition.

---

# 2.4 Implement the actual risk function

Create:

```text
math/risk_model.py
```

Implement:

$$
R(\tau)
=
w_dFSR(\tau)
+
w_mFMR(\tau)
+
w_sP_{stale}
$$

Substitute:

$$
FSR(\tau)=(1-\tau)^\beta
$$

and:

$$
FMR(\tau)=\tau^\alpha
$$

giving:

$$
R(\tau)
=
w_d(1-\tau)^\beta
+
w_m\tau^\alpha
+
w_sP_{stale}
$$

The risk formulation is explicitly part of the paper's mathematical contribution. 

---

# 2.5 Implement derivative

Your code must explicitly implement:

$$
R'(\tau)
=
-w_d\beta(1-\tau)^{\beta-1}
+
w_m\alpha\tau^{\alpha-1}
$$

Do not numerically approximate the derivative unless you use it as a separate validation.

The analytical derivative should be the primary implementation.

---

# 2.6 Implement second derivative

Also implement:

$$
R''(\tau)
=
w_d\beta(\beta-1)(1-\tau)^{\beta-2}
+
w_m\alpha(\alpha-1)\tau^{\alpha-2}
$$

This lets you experimentally validate the convexity condition when:

$$
\alpha>1,\qquad\beta>1.
$$

The paper uses this expression in the mathematical argument. 

---

# 2.7 Closed-form τ*

When the paper's condition:

$$
|\alpha-\beta|<0.05
$$

is satisfied, use the closed-form solution.

Define:

$$
A=
\left(
\frac{w_d\beta}
{w_m\alpha}
\right)^{1/(\alpha-1)}
$$

Then:

$$
\boxed{
\tau^*
=
\frac{A}{1+A}
}
$$

This must exist as an actual Python function.

For example:

```python
def tau_star_closed_form(alpha, beta, wd, wm):
    A = ((wd * beta) / (wm * alpha)) ** (1 / (alpha - 1))
    return A / (1 + A)
```

Do **not** simply use a hard-coded `0.87` or similar number.

---

# 2.8 General α ≠ β case

If:

$$
|\alpha-\beta|\geq0.05
$$

your implementation must not pretend the closed-form approximation applies.

Instead solve:

$$
R'(\tau)=0
$$

using bisection on:

$$
[0.001,0.999].
$$

The paper specifies 40 iterations. 

Implement:

```python
def tau_star_bisection(...):
    ...
```

with exactly:

```text
40 iterations
```

unless you additionally make the iteration count configurable.

---

# 2.9 Validate the root

After finding:

```text
tau_star
```

calculate:

```text
R'(tau_star)
```

and report it.

It should be close to zero.

Then calculate:

```text
R(tau_star)
```

and compare against nearby thresholds.

For example:

```text
tau_star - 0.05
tau_star - 0.02
tau_star
tau_star + 0.02
tau_star + 0.05
```

---

# 2.10 Empirical vs theoretical τ*

This distinction is very important for the paper.

You will have:

### Empirical risk

Directly calculated from:

```text
measured FMR
measured FSR
```

and:

### Fitted-model risk

Calculated from:

$$
(1-\tau)^\beta
$$

and:

$$
\tau^\alpha.
$$

Then compare them.

This validates whether the mathematical model actually approximates your measurements.

---

# 2.11 Sensitivity analysis

The paper uses:

$$
w_d/w_m=4.
$$

Do not report only that one configuration.

Run:

```text
wd/wm = 1
wd/wm = 2
wd/wm = 4
wd/wm = 8
wd/wm = 16
```

and calculate:

$$
\tau^*
$$

for each.

This will give you:

```text
cost ratio → optimal threshold
```

and demonstrate how deployment costs affect the proposed threshold.

---

# AMAL — PHASE 1b DELIVERABLE

You should produce:

```text
math/
├── statistics.py
├── calibration.py
├── risk_model.py
├── tau_solver.py
├── validate_tau.py
└── measure_delta_q.py
```

and:

```text
results/
├── fitted_parameters.csv
├── tau_star.csv
├── risk_curve.csv
├── tau_sensitivity.csv
└── delta_q.csv
```

---

# AMAL — Δq

Run:

```text
qwen2.5:7b-instruct
```

and:

```text
qwen2.5:1.5b-instruct
```

on the same 19 prompts.

Score:

```text
0 = incorrect tool / implausible arguments

1 = correct tool / plausible arguments
```

Then:

$$
Q_{7B}=
\frac{\sum score_{7B}}{N}
$$

$$
Q_{1.5B}=
\frac{\sum score_{1.5B}}{N}
$$

and:

$$
\boxed{
\Delta q=Q_{7B}-Q_{1.5B}
}
$$

The paper explicitly calls for this measurement. 

---

# AMAL — Degradation threshold

Implement the paper's formulation:

$$
\tau^*
=
\frac{\lambda_q\Delta q}{\lambda_\ell}
$$

with the chosen cost parameters documented in the experiment configuration.

Do not hard-code the final threshold into the application.

The application should read it from the calculated result/configuration.

---

# AMAL — OPEN CODE PROMPT FOR MATHEMATICS

Give OpenCode this **after Phase 1 data exists**:

```text
Read Complete-Research.md, Guide.md and Runbook.md.

Implement the mathematical contribution exactly as specified in the paper.

Do NOT replace the proposed mathematical model with a generic threshold search.

Input:
pilot_results.csv
fmr_fsr.csv

Implement:

1. Empirical FMR(tau)
2. Empirical FSR(tau)
3. Log-log regression:
       log(FSR) = beta * log(1-tau) + c
4. Log-log regression:
       log(FMR) = alpha * log(tau) + c
5. R² for both fits
6. Check alpha > 1 and beta > 1
7. Check |alpha-beta| < 0.05
8. If condition holds, calculate:

       A = ((wd*beta)/(wm*alpha))^(1/(alpha-1))
       tau_star = A/(1+A)

9. Otherwise solve:

       R'(tau)=0

   using bisection on [0.001, 0.999]
   for exactly 40 iterations.

10. Implement:
       R(tau)
       R'(tau)
       R''(tau)

11. Validate:
       R'(tau_star) ≈ 0
       tau_star is a minimum
       compare fitted risk and empirical risk

12. Run sensitivity analysis for wd/wm:
       1, 2, 4, 8, 16

13. Generate:
       fitted_parameters.csv
       tau_star.csv
       risk_curve.csv
       tau_sensitivity.csv

14. Generate the real FMR/FSR figure.

Every number must come from actual pilot data.
Never generate mock results.
```

---

# AMAL — PHASE 5

# Statistical analysis

When Santhosh gives you:

```text
results.csv
```

you calculate all final statistics.

---

## Table III

Unsafe action rate:

```text
Cold restart
Naive retry
Verify-before-retry
Kinbridge-Sync
```

against:

```text
0
10
90
300 sec
```

The paper defines this exact comparison. 

---

# Table IV

Calculate:

```text
duplicate rate
stale-execution rate
escalation rate
median TTR
```

for each method.

The paper expects this table to expose **why** each baseline fails, rather than merely reporting aggregate failure. 

---

# Table V

Calculate:

```text
replication lag
same-node
cross-node
```

for:

```text
0 ms
500 ms
unreplicated
```

This isolates C3.

---

# Table VI

Calculate:

```text
Buffer only
+ hash keys
+ intent keys
+ staleness
+ epoch guard
+ degradation gate
```

The paper's ablation is designed so that each mechanism can be evaluated independently. 

---

# Cohen's h

Implement:

$$
h=
2\arcsin(\sqrt{p_1})
-
2\arcsin(\sqrt{p_2})
$$

against the cold-restart baseline as specified.

---

# AMAL FINAL DELIVERABLE

Your side should ultimately contain:

```text
results/
│
├── pilot/
│   ├── pilot_results.csv
│   ├── fmr_fsr.csv
│   └── pilot_statistics.csv
│
├── mathematics/
│   ├── fitted_parameters.csv
│   ├── tau_star.csv
│   ├── risk_curve.csv
│   ├── tau_sensitivity.csv
│   └── delta_q.csv
│
├── final/
│   ├── results.csv
│   ├── table_II.csv
│   ├── table_III.csv
│   ├── table_IV.csv
│   ├── table_V.csv
│   └── table_VI.csv
│
└── figures/
    └── fmr_fsr_curve.pdf
```

---

# PART B — SANTHOSH

Santhosh uses **Antigravity** as the primary implementation agent.

His responsibility is the actual distributed systems/testbed implementation.

---

# SANTHOSH — PHASE 0

Install:

```text
WSL2
Docker Desktop
Python
Git
Redis
```

Verify:

```powershell
docker run hello-world
```

and:

```powershell
docker compose version
```

---

# SANTHOSH — PHASE 2

# Tool World

Create:

```text
tool_world/
```

Build:

```text
FastAPI
+
SQLite
+
effect ledger
```

The ledger schema must contain:

```text
key TEXT PRIMARY KEY
tool TEXT
args_hash TEXT
status TEXT
committed_at REAL
```

This is specified directly in the research plan. 

---

# Santhosh — effect ledger

Implement:

```text
PUT/POST action
      ↓
check key
      ↓
already committed?
   /        \
 YES        NO
  ↓          ↓
return      execute
committed     ↓
             ledger
```

This is the basis for testing P1.

---

# Santhosh — fault injection

Implement:

```text
delay
timeout
hang
response loss
```

The client must be able to experience:

```text
server executes action
      ↓
client doesn't receive confirmation
```

This produces the uncertain-status condition needed by the reintegration algorithm.

---

# Santhosh — action buffer

Create:

```text
client/buffer.py
```

WAL record:

```text
intent_id
key
tool
args
t0
ttl
v_pre_spec
```

The Runbook explicitly defines this structure. 

---

# Santhosh — link monitor

Implement:

$$
\hat r_t
=
\alpha r_t+(1-\alpha)\hat r_{t-1}
$$

with:

$$
\alpha=0.125
$$

Then:

$$
\hat d_t
=
\beta |r_t-\hat r_t|
+
(1-\beta)\hat d_{t-1}
$$

with:

$$
\beta=0.25.
$$

Then:

$$
RTO=\hat r+4\hat d.
$$

These equations are part of the specified design. 

---

# Santhosh — PHASE 3

Create:

```text
client
edge-a
edge-b
```

with Docker Compose.

Architecture:

```text
                CLIENT
                   |
              network gap
                   |
          ┌────────┴────────┐
          │                 │
       EDGE-A            EDGE-B
       FastAPI            FastAPI
       Redis              Redis
       Ledger             Ledger
```

The paper specifies this three-container evaluation setup. 

---

# Santhosh — Network control

Implement:

```text
netctl.py
```

with:

### Delay

```bash
tc qdisc add dev eth0 root netem delay Xms
```

### Loss

```bash
tc qdisc add dev eth0 root netem loss X%
```

### Blackout

```bash
iptables -I OUTPUT -j DROP
```

### Restore

```bash
iptables -F
```

The Runbook explicitly calls for these mechanisms. 

---

# Santhosh — Gilbert–Elliott model

Implement:

```text
GOOD
  |
 pGB
  ↓
BAD
  |
 pBG
  ↓
GOOD
```

with:

$$
\pi_B=
\frac{p_{GB}}
{p_{GB}+p_{BG}}
$$

and configure it to generate the target outage behavior.

---

# Santhosh — PHASE 3b

# C3 replication lag

Implement:

```text
replicator.py
```

Flow:

```text
Edge-A ledger
      ↓
replicator
      ↓
sleep(lag)
      ↓
Edge-B ledger
```

Support:

```text
--lag-ms 0
--lag-ms 500
--lag never
```

The paper explicitly requires these three conditions. 

---

# Santhosh — PHASE 4

# Algorithm 1

This is the **main systems contribution**.

Do not implement a generic retry mechanism.

Implement the paper's actual reintegration procedure.

---

# Algorithm 1 — exact flow

## STEP 1 — Epoch guard

On reconnect:

```text
client reconnects to edge B
        ↓
compare:
ledger_epoch(B)
buffer.max_epoch
```

If:

$$
ledger\_epoch(B)<max\_epoch(buffer)
$$

then:

```text
wait sync window
```

If it catches up:

```text
continue
```

Otherwise:

```text
ESCALATE
STOP
```

The paper explicitly says never replay blindly in this condition. 

---

# STEP 2 — Check ledger

For every buffered action:

```text
status(k)
```

Possible:

```text
COMMITTED
ABSENT
UNKNOWN
```

---

# STEP 3 — COMMITTED

If:

```text
COMMITTED
```

then:

```text
mark action COMMITTED
continue
```

Do **not execute it again**.

---

# STEP 4 — UNKNOWN

Call:

```text
V_post
```

which is read-only.

If:

```text
V_post == TRUE
```

then:

```text
COMMITTED
```

No replay.

If still unknown:

```text
ESCALATED
```

This directly implements the paper's uncertain-status handling. 

---

# STEP 5 — TTL

If:

$$
t_{now}-t_0 > TTL
$$

then:

```text
INVALIDATED
```

Never replay.

---

# STEP 6 — V_pre

Evaluate:

$$
V_{pre}(a,S_{now})
$$

If false:

```text
INVALIDATED
```

---

# STEP 7 — Semantic similarity

Compare the buffered action against the newly planned action.

Use:

```text
all-MiniLM-L6-v2
```

and:

$$
cosine(a,b)
$$

Then compare with **the τ*** generated by Amal's mathematical pipeline.

This is crucial.

### Do NOT put:

```python
TAU = 0.8
```

inside the algorithm.

Instead:

```text
calibration result
       ↓
tau_star
       ↓
configuration
       ↓
Reintegration Service
```

That connects your novel mathematical contribution to the actual algorithm.

---

# STEP 8 — Intent merge

If:

$$
similarity\geq\tau^*
$$

then merge the intent.

Otherwise follow the appropriate safety path rather than blindly treating the action as equivalent.

---

# STEP 9 — Execute using key

Execute using:

```text
intent key
```

so the server-side effect ledger can prevent duplicate effects.

---

# Santhosh — feature flags

Implement:

```text
--enable-intent-keys
--enable-staleness-check
--enable-epoch-guard
--enable-degradation-gate
```

The paper explicitly requires independent ablation of the mechanisms. 

---

# Santhosh — baselines

Implement all three through the **same underlying infrastructure**.

## Cold restart

```text
no retained state
```

## Naive retry

```text
always execute
```

## Verify-before-retry

```text
ledger status
+
V_post
```

but:

```text
no TTL
no V_pre
no epoch guard
```

The research plan explicitly specifies these baseline variants. 

---

# Santhosh — mandatory tests

Before Phase 5:

```text
[ ] committed action isn't executed twice
[ ] unknown + V_post true
[ ] unknown + V_post unknown
[ ] TTL expired
[ ] V_pre false
[ ] valid replay
[ ] epoch behind
[ ] epoch catches up
[ ] epoch remains stale
[ ] semantic match
[ ] semantic non-match
[ ] cross-node replication lag
[ ] same-node reconnect
[ ] blackout recovery
```

---

# SANThOSH — PHASE 5

# Run the actual experiment

He runs:

$$
4\ methods
\times
4\ outage durations
\times
100
$$

Therefore:

$$
\boxed{1600\ episodes}
$$

The explicit experiment grid is:

```text
Methods:
1. cold_restart
2. naive_retry
3. verify_before_retry
4. kinbridge_sync

Outages:
0
10
90
300 seconds
```

The Runbook describes these cells and 100 episodes per cell. 

---

# Santhosh — C3

Run:

```text
0 ms
500 ms
unreplicated
```

against:

```text
same-node
cross-node
```

for Kinbridge-Sync.

---

# Santhosh — ablation

Run:

```text
buffer only
+ hash keys
+ intent keys
+ staleness
+ epoch guard
+ degradation gate
```

at:

```text
90-second outage
```

with:

```text
200 episodes / configuration
```

as specified in the paper. 

---

# Santhosh — experiment log

Every episode must produce a machine-readable record.

Example:

```json
{
  "episode_id": 1,
  "method": "kinbridge_sync",
  "outage_duration": 90,
  "reconnect_node": "edge-b",
  "replication_lag_ms": 500,

  "duplicate": false,
  "stale_execution": false,
  "escalated": true,

  "status": "ESCALATED",

  "ttr_seconds": 2.41
}
```

Then aggregate this into:

```text
results.csv
```

---

# SANThOSH — ANTIGRAVITY MASTER PROMPT

Give Antigravity this:

```text
Read these files first:

1. Complete-Research.md — authoritative specification
2. Guide.md — implementation guide
3. Runbook.md — operational runbook

You are responsible for the Santhosh systems implementation.

Do not invent research results.
Do not simplify Algorithm 1.
Do not replace any mechanism with a generic retry implementation.

Implement:

PHASE 2:
- FastAPI tool world
- SQLite effect ledger
- fault injection
- action buffer/WAL
- link monitor
- EWMA RTT
- RTT deviation
- RTO
- hysteresis FSM

PHASE 3:
- Docker Compose
- client
- edge-A
- edge-B
- tc netem
- iptables blackout
- Gilbert-Elliott outage model

PHASE 3b:
- dual Redis
- replication lag:
  0 ms
  500 ms
  unreplicated

PHASE 4:
Implement the Reintegration Service exactly as Algorithm 1.

Required sequence:

1. ledger epoch guard
2. wait sync window if replica is behind
3. escalate if still behind
4. check ledger status
5. COMMITTED -> mark committed
6. UNKNOWN -> V_post
7. V_post true -> committed
8. unresolved UNKNOWN -> escalate
9. TTL check
10. V_pre check
11. semantic similarity check
12. compare against externally calibrated tau_star
13. merge equivalent intent
14. execute using intent key

Feature flags:

--enable-intent-keys
--enable-staleness-check
--enable-epoch-guard
--enable-degradation-gate

Implement:
- naive_retry.py
- verify_before_retry.py

All baselines must use the same underlying tool world and experiment harness.

Write comprehensive pytest coverage for every Algorithm 1 branch.

Do NOT hard-code tau_star.
Read it from the mathematical calibration output produced by Amal.

Do NOT hard-code expected experimental rates.

Every episode must produce structured JSON/CSV data containing:
- method
- outage duration
- reconnect node
- replication lag
- duplicate
- stale execution
- escalation
- final status
- TTR
- relevant algorithm decisions

Before declaring Phase 4 complete:
run the complete test suite and demonstrate each branch of Algorithm 1 with tests.
```

---

# THE MOST IMPORTANT INTEGRATION BETWEEN YOUR TWO PARTS

This is where I want you and Santhosh to be especially careful.

Your work is **not two unrelated implementations**.

It must connect like this:

```text
                    AMAL
                     │
                     │ Pilot
                     ▼
              similarity data
                     │
                     ▼
                FMR / FSR
                     │
                     ▼
                 α / β
                     │
                     ▼
             Risk minimization
                     │
                     ▼
                  τ*
                     │
                     │
                     ▼
             ┌───────────────┐
             │ CONFIG / JSON │
             └───────┬───────┘
                     │
                     ▼
                  SANTHOSH
                     │
                     ▼
          Reintegration Service
                     │
                     ▼
          semantic similarity
                     │
                     ▼
                similarity
                     │
              ┌──────┴──────┐
              │             │
          ≥ τ*             < τ*
              │             │
        intent merge     safety path
```

So **τ*** is not just a number that appears in the paper.

It must actually control the semantic matching decision in Algorithm 1.

That is one of the most important things to preserve if you want the claimed mathematical contribution to be genuinely implemented rather than merely described.

---

# FINAL OWNERSHIP

| Component                     |                             Amal |           Santhosh |
| ----------------------------- | -------------------------------: | -----------------: |
| Environment                   |                                ✅ |                  ✅ |
| Ollama                        |                                ✅ |                  — |
| Pilot                         |                                ✅ |                  — |
| C1                            |                                ✅ |                  — |
| Three key schemes             |                                ✅ |                  — |
| Similarity measurement        |                                ✅ | Shared integration |
| FMR                           |                                ✅ |                  — |
| FSR                           |                                ✅ |                  — |
| α                             |                                ✅ |                  — |
| β                             |                                ✅ |                  — |
| Risk function                 |                                ✅ |                  — |
| \(R'\)                        |                                ✅ |                  — |
| \(R''\)                       |                                ✅ |                  — |
| Closed-form \(\tau^*\)        |                                ✅ |                  — |
| Bisection \(\tau^*\)          |                                ✅ |                  — |
| τ sensitivity                 |                                ✅ |                  — |
| Δq                            |                                ✅ |                  — |
| Degradation threshold         |                                ✅ |        Integration |
| FastAPI tool world            |                                — |                  ✅ |
| SQLite ledger                 |                                — |                  ✅ |
| WAL                           |                                — |                  ✅ |
| Link monitor                  |                                — |                  ✅ |
| Docker testbed                |                                — |                  ✅ |
| Network outage                |                                — |                  ✅ |
| Gilbert–Elliott               |                                — |                  ✅ |
| Redis replication             |                                — |                  ✅ |
| C3                            |                                — |                  ✅ |
| Algorithm 1                   | Specification/integration review |                  ✅ |
| Intent matcher implementation |         Mathematical calibration |                  ✅ |
| Epoch guard                   |                                — |                  ✅ |
| Staleness                     |                                — |                  ✅ |
| V_pre                         |                                — |                  ✅ |
| V_post                        |                                — |                  ✅ |
| Baselines                     |                                — |                  ✅ |
| Full experiment execution     |                         Analysis |                  ✅ |
| `results.csv`                 |                          Receive |            Produce |
| Wilson CI                     |                                ✅ |                  — |
| Cohen's h                     |                                ✅ |                  — |
| Tables II–VI                  |                                ✅ |           Raw data |
| Figures                       |                                ✅ |           Raw data |
| Sections V–VI                 |                                ✅ |                  — |
| Sections IV/VII               |                                — |                  ✅ |

---

## Two synchronization points

### SYNC 1 — after your Phase 1

You hand Santhosh:

```text
pilot_results.csv
similarity_results.csv
FMR/FSR
decision rule
calibrated tau* if available
```

### SYNC 2 — after Santhosh Phase 5

He hands you:

```text
results.csv
experiment configuration
raw logs
seeds
```

You then produce:

```text
Wilson CIs
Cohen's h
Tables III–VI
final figures
final mathematical validation
```

This preserves the division specified in the research plan while ensuring the novel mathematics is **actually connected to the executable Kinbridge-Sync algorithm**, rather than being mathematics that exists only in the paper. 

And importantly, if the real results contradict the paper's predictions, **we update the paper to the measured result**. The research document explicitly says not to force the results to match the prediction. 
