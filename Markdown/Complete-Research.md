Understood — no file this time, everything inline. This is the full, consolidated, from-scratch version: every math formula the paper uses (including a few that were referenced but never spelled out), every setup step for Windows, and the two-person task split. I've also added a section at the end specifically about what to do when a result doesn't come out the way the paper predicted — that's a normal part of research, not a problem to hide.

---

# PART 0 — Ground rules

Three honesty notes before the steps, so nothing here misleads you later:

1. **The math in this paper is standard decision-theory technique applied to a new problem** (LLM tool-call idempotency). That is real, legitimate novelty for a systems paper — but it is not "brand new mathematics." Don't claim more than that in the paper.
2. **Every result table starts empty for a reason.** You're going to run real experiments. Some will confirm the paper's predictions, some might not. Both outcomes are fine — Part 6 below tells you exactly how to write up either case.
3. **Nothing about Sections I–IV of the paper (the model, the five properties, the impossibility proof, the architecture) depends on any of this.** Those are proven on paper already. Everything below only fills in Sections V and VI.

---

# PART 1 — Windows 11 setup, completely from scratch (all free)

### 1.1 Enable WSL2
Open **PowerShell as Administrator**:
```powershell
wsl --install
```
Reboot when asked. Open "Ubuntu" from the Start menu once to finish setup (create a Linux username/password — anything works). Confirm:
```powershell
wsl -l -v
```
You should see `Ubuntu`, `VERSION 2`. This matters because `tc netem` and `iptables` (the network-outage tools the paper's testbed needs) are Linux-only — WSL2 is what makes Docker Desktop run genuine Linux containers on Windows so those commands work.

### 1.2 Install Docker Desktop (free for personal/education use)
Download from docker.com/products/docker-desktop, run installer, keep "Use WSL 2" checked (default on Win11). After install: **Settings → Resources → WSL Integration** → enable your Ubuntu distro → Apply & Restart.
Confirm:
```powershell
docker run hello-world
```

### 1.3 Install Ollama (free, native Windows, runs as a background service)
```powershell
winget install --id Ollama.Ollama
```
Pull two models (both free downloads):
```powershell
ollama pull qwen2.5:7b-instruct
ollama pull qwen2.5:1.5b-instruct
```
The 7B model is the "agent under test" in the pilot. The 1.5B model is the paper's local-SLM stand-in (needed for Phase 1b's Δq measurement). If your machine has under ~8GB free RAM, use `qwen2.5:1.5b-instruct` / `phi3:mini` for both — the pilot's divergence measurement doesn't depend on model size.

### 1.4 Install Python and Git (free)
```powershell
winget install --id Python.Python.3.12
winget install --id Git.Git
```
Set up the project:
```powershell
mkdir C:\kinbridge-sync
cd C:\kinbridge-sync
git init
python -m venv .venv
.venv\Scripts\activate
pip install fastapi uvicorn sentence-transformers scipy statsmodels numpy matplotlib redis requests
```

---

# PART 2 — Configure your two AI coding tools (both free)

### 2.1 opencode → local Ollama (unlimited, $0)
Create `%USERPROFILE%\.config\opencode\opencode.json`:
```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://localhost:11434/v1" },
      "models": { "qwen2.5:7b-instruct": {} }
    }
  }
}
```
Run `opencode` in the project folder, pick the `ollama` provider.

### 2.2 Antigravity → its own free Gemini/Claude quota
Antigravity does **not** accept a local model (checked directly — it only offers a fixed curated list, no custom endpoint option). Just open the project folder in Antigravity and pick a model in the Agent Panel. Free up to Google's daily quota. Use it for the one or two hardest files (Algorithm 1); use opencode/Ollama for everything repetitive, since that has no limit.

---

# PART 3 — Every math formula you need, in one place

### 3.1 The buffered action (what gets stored)
$$a = (k,\ \text{tool},\ \text{args},\ t_0,\ \text{ttl},\ V_{\text{pre}},\ V_{\text{post}})$$
Plain English: each pending action carries its idempotency key, which tool it calls, the arguments, when it was created, how long it's valid for, and two check functions — one to test if it's still correct to run ($V_{\text{pre}}$), one to ask the server if it already ran ($V_{\text{post}}$).

### 3.2 Link detection (Jacobson/Karels estimator)
$$\hat{r} \leftarrow \alpha r + (1-\alpha)\hat{r}, \quad \alpha = 0.125$$
$$\hat{d} \leftarrow \beta |r - \hat{r}| + (1-\beta)\hat{d}, \quad \beta = 0.25$$
$$\text{RTO} \leftarrow \hat{r} + 4\hat{d}$$
Plain English: track a moving average of round-trip time ($\hat r$) and how much it jitters ($\hat d$); if you don't hear back within RTO, that's a timeout. Link goes DOWN after several timeouts in a row (not just one), and back UP only after several successful probes in a row — this hysteresis stops a single lucky/unlucky packet from flipping the state.

### 3.3 The three idempotency key schemes
$$k_{\text{payload}} = H(\text{tool} \,\|\, \text{canonical}(\text{args}))$$
$$k_{\text{bucket}} = H(\text{agent\_id} \,\|\, \text{action\_type} \,\|\, \text{payload} \,\|\, \lfloor t/\Delta \rfloor)$$
$$k_{\text{intent}} = H(\text{session\_id} \,\|\, \text{turn\_seq} \,\|\, \text{intent\_id})$$
Plain English: the first two hash the actual request (so any wording change breaks them); the third hashes a stable ID assigned once at planning time, so it survives the agent rewording its request.

**Provable failure of the bucket scheme** — for outage length $G$ longer than bucket width $\Delta$:
$$\lfloor (t_0+G)/\Delta \rfloor \neq \lfloor t_0/\Delta \rfloor \implies k_{\text{bucket}} \text{ mismatches even on an identical payload}$$
This is arithmetic, not model behavior — a good sanity check that your pilot harness is wired correctly (this row should measure ~100% failure once $G>\Delta$).

### 3.4 Matching error rates
$$\text{FMR}(\tau) = \frac{|\{\text{distinct intents merged}\}|}{|\text{intent pairs}|}, \qquad \text{FSR}(\tau) = \frac{|\{\text{same intent not merged}\}|}{|\text{re-issued intents}|}$$
Plain English: turn the similarity threshold τ up and you catch more true matches but also wrongly merge some distinct ones (FMR↑); turn it down and the reverse (FSR↑, more duplicates slip through).

### 3.5 Risk-minimization threshold (the paper's main "novel formula")
$$R(\tau) = w_d \cdot \text{FSR}(\tau) + w_m \cdot \text{FMR}(\tau) + w_s \cdot P_{\text{stale}}$$
With the curvature-aware approximation $\text{FSR}(\tau)\approx(1-\tau)^\beta$, $\text{FMR}(\tau)\approx\tau^\alpha$:
$$\tau^\star = \left(1 + \left(\frac{w_d\beta}{w_m\alpha}\right)^{\frac{1}{\alpha-1}}\right)^{-1}\left(\frac{w_d\beta}{w_m\alpha}\right)^{\frac{1}{\alpha-1}} \quad \text{(valid when } \alpha=\beta\text{)}$$

**Lemma 1 (uniqueness — the extension that covers the general case, α≠β):**
$$R''(\tau) = w_d\beta(\beta-1)(1-\tau)^{\beta-2} + w_m\alpha(\alpha-1)\tau^{\alpha-2} \geq 0 \text{ on } (0,1)$$
so $R$ is strictly convex; since $R'(0^+) = -w_d\beta<0$ and $R'(1^-)=w_m\alpha>0$, there is exactly one root — the global minimum, guaranteed.

**Algorithm 2 (turns the formula into something you run):**
```
Input: measured (τ_i, FMR_i, FSR_i) triples from the pilot, cost ratio w_d/w_m
1. Fit β = slope of log(FSR_i) vs log(1-τ_i)      [numpy.polyfit]
2. Fit α = slope of log(FMR_i) vs log(τ_i)
3. If |α-β| < 0.05: use the closed form above
4. Else: bisect on R'(τ)=0 over [0.001, 0.999] for ~40 iterations
         (guaranteed to converge — Lemma 1 proves R' is monotonic)
5. Return τ*, α, β
```

### 3.6 Degradation-gate stopping rule
$$\lambda_q \Delta q < \lambda_\ell \int_0^{T_{\max}}(1-F(\tau))\,d\tau + \lambda_{\text{miss}}(1-F(T_{\max})) \;\Rightarrow\; \tau^* = \frac{\lambda_q \Delta q}{\lambda_\ell}$$
Plain English: if the outage is predicted to run longer than this threshold, switch to the smaller local model rather than wait silently. $\Delta q$ (the quality gap between your two Ollama models) needs to be *measured*, not assumed — see Phase 1b.

### 3.7 Statistics you'll compute on every result (not in the paper explicitly, needed for all tables)
**Wilson 95% confidence interval** on a proportion $\hat p = x/n$:
$$\text{center} = \frac{\hat p + z^2/2n}{1+z^2/n}, \quad \text{margin} = \frac{z\sqrt{\hat p(1-\hat p)/n + z^2/4n^2}}{1+z^2/n}, \quad z=1.96$$
**Cohen's h** (effect size between two proportions):
$$h = 2\arcsin(\sqrt{p_1}) - 2\arcsin(\sqrt{p_2})$$

### 3.8 Gilbert–Elliott outage model (mentioned in the paper, formula not previously given — needed to actually build it)
Two states, Good (G) and Bad (B). Transition probabilities $p_{GB}$ (link fails) and $p_{BG}$ (link recovers). In state B, packets drop with probability $\approx 1$ (hard outage); in state G, drop probability $\approx 0$. The long-run fraction of time in the Bad state is
$$\pi_B = \frac{p_{GB}}{p_{GB}+p_{BG}}$$
Pick $p_{GB}, p_{BG}$ so that $\pi_B \times (\text{total sim time})$ matches your target outage duration (10s/90s/300s cells).

### 3.9 The impossibility proof (Two Generals, restated for completeness)
No finite exchange of messages over an unreliable channel lets two parties both become certain an action occurred — proven by contradiction (the last message in any finite exchange could always be lost, and the sender can't tell). This is why the paper claims P1 (no duplicates) unconditionally but P2 (no silent loss) only "modulo escalation" — you cannot buy both guarantees at once, ever, regardless of how good your code is. Nothing to implement here; just know it's why Kinbridge-Sync has an escalation state instead of pretending to solve an unsolvable problem.

---

# PART 4 — Build phases, in order, with exact opencode/Antigravity prompts

### Phase 0 — Verify environment (you do this, no agent)
```powershell
docker run hello-world
ollama run qwen2.5:7b-instruct "say hi"
```

### Phase 1 — Pilot study → Table II
Prompt for **opencode**:
```
Create pilot/ with:
- tools.py: 3 mock tool schemas as Python functions with JSON-schema
  signatures for Ollama tool-calling:
    navigate_to(lat: float, lon: float, mode: str)
    log_inspection(site_id: str, status: str, notes: str)
    request_supply_drop(lat: float, lon: float, payload: str, priority: str)
- prompts.json: 19 short task prompts, ~6-7 per tool class, e.g.
  "Navigate to the coordinates 45.231N, 6.887E using the fastest route."
- run_pilot.py: connect to http://localhost:11434. For each prompt, call
  the model to get tool call a1, then re-send the SAME conversation
  without a tool result (simulating the gap) to get a2. Compute
  hash(a1)!=hash(a2) under k_payload, k_bucket (bucket=60s), k_intent
  (a UUID intent_id generated once, carried through both calls — this
  key can never mismatch by construction, it's the control condition).
  Sweep temperature in [0.0, 0.7]. Write pilot_results.csv:
  columns = prompt_id, tool_class, scheme, temperature, mismatch (0/1).
- analyze_pilot.py: read the CSV, compute per-(scheme,temperature)
  failure rate and Wilson 95% CI using the formula:
  center=(p+z^2/2n)/(1+z^2/n), margin=z*sqrt(p*(1-p)/n+z^2/4n^2)/(1+z^2/n),
  z=1.96. Print a table matching Table II's row structure. Print which
  branch of the decision rule fired (>15%, 5-15%, <5%).
```
```powershell
python pilot\run_pilot.py
python pilot\analyze_pilot.py
```

### Phase 1b — Validate the math (Section 3.5–3.6 above) → real Fig. 2, real τ*, real Δq
Prompt for **opencode**:
```
Create calibrate.py implementing Algorithm 2:
1. Read pilot_results.csv, bucket mismatches by similarity-threshold-
   equivalent tau (use the semantic-equivalence normalizer's score as tau
   for each pair, not just a fixed value).
2. Fit beta = numpy.polyfit(log(1-tau), log(FSR), 1)[0] per tool class.
3. Fit alpha = numpy.polyfit(log(tau), log(FMR), 1)[0] per tool class.
4. If abs(alpha-beta) < 0.05: use the closed form for tau_star.
   Else: bisect R'(tau)=0 over [0.001,0.999], 40 iterations.
5. Print tau_star, alpha, beta per tool class.
6. Regenerate the FMR/FSR figure (matplotlib) using the REAL fitted
   curves per tool class, replacing the illustrative one.

Create measure_delta_q.py: run the same 19 pilot prompts through both
qwen2.5:7b-instruct and qwen2.5:1.5b-instruct, score each response with
a simple rubric (did it call the right tool with plausible arguments,
0-1), and report the mean score gap as Delta_q.
```
```powershell
python calibrate.py
python measure_delta_q.py
```

### Phase 2 — Tool world, buffer, link monitor
Prompt for **opencode**:
```
Create tool_world/ — FastAPI + SQLite app exposing the 3 tools from
pilot/tools.py. Add effect_ledger table: (key TEXT PRIMARY KEY, tool
TEXT, args_hash TEXT, status TEXT, committed_at REAL). Add a
fault-injection query param that delays or hangs a response.

Create client/buffer.py — append-only JSONL WAL storing
(intent_id, key, tool, args, t0, ttl, v_pre_spec).

Create client/link_monitor.py implementing the EWMA/RTO/hysteresis FSM
from Section 3.2 above exactly, with a pytest test using a synthetic
RTT trace with an injected dropout.
```
```powershell
cd tool_world
uvicorn main:app --reload
pytest client\
```

### Phase 3 — Network testbed (Docker does the Linux networking for you)
Prompt for **opencode**:
```
Create docker-compose.yml: 3 services (client, edge-a, edge-b) on a
bridge network. edge-a/edge-b run tool_world + Redis, base image
python:3.12-slim + apt-installed iproute2 and iptables, with
cap_add: [NET_ADMIN, NET_RAW].

Create netctl.py: docker-exec into a container, run
`tc qdisc add dev eth0 root netem delay {ms}ms loss {pct}%` for shaping,
`iptables -I OUTPUT -j DROP` / `iptables -F` for a full blackout.

Create outage_model.py implementing the Gilbert-Elliott model from
Section 3.8 above (two states, p_GB/p_BG transition probabilities,
pi_B = p_GB/(p_GB+p_BG) calibrated to hit target average outage length),
driving netctl.py.
```
```powershell
docker compose up -d
```

### Phase 3b — Replication-lag forwarder (needed for Table V / C3)
Prompt for **opencode**:
```
Create replicator.py: an async script that watches edge-a's Redis for
new ledger writes and copies each one to edge-b's Redis after
asyncio.sleep(lag_ms/1000). Make lag_ms a CLI argument so it can be set
to 0, 500, or "never" (unreplicated) per experiment cell.
```

### Phase 4 — Algorithm 1 + baselines (hand this file to Antigravity — worth spending quota on)
Prompt for **Antigravity**:
```
Implement reintegration_service.py per this exact procedure:
1. If reconnect node's ledger_epoch < buffer's max_epoch: wait up to a
   sync window; if still behind, escalate everything and stop (never
   replay blind).
2. For each buffered action, in order: check ledger status.
   - Committed -> mark committed, next action.
   - Unknown -> call V_post (read-only). True -> committed. Still
     unknown -> escalate.
   - If ttl expired or V_pre fails -> invalidate, notify, next action.
   - Check similarity (sentence-transformers all-MiniLM-L6-v2, cosine)
     against the freshly re-planned actions; if above tau, merge intent.
   - Execute with the action's key.
Build it with feature flags: --enable-intent-keys, --enable-staleness-
check, --enable-epoch-guard, each independently toggleable, so the
ablation ladder in Phase 5 is just flag combinations, not six separate
programs. Write one pytest per branch (6 tests total).

Then create naive_retry.py (all flags off, always execute blind) and
verify_before_retry.py (only the ledger-status/V_post check on, ttl/
V_pre and epoch-guard flags off) as thin wrappers around the same code.
```
```powershell
pip install sentence-transformers
pytest reintegration_service\
```

### Phase 5 — Full factorial run + ablation → Tables III–VI
Prompt for **opencode**:
```
Create run_experiment.py: for each (method, outage_duration) in
{cold_restart, naive_retry, verify_before_retry, kinbridge_sync} x
{0,10,90,300}s, 100 episodes each, drive the Docker testbed via
netctl.py/outage_model.py, log dup/stale/escalation/committed to
results.csv. Separately, for the ablation ladder, run 200 episodes per
flag-combination (buffer-only, +hash-keys, +intent-keys, +staleness,
+epoch-guard, +full) at a fixed 90s outage.

Create analyze_results.py: compute Wilson CIs (Section 3.7 formula) and
Cohen's h per cell, print tables matching the exact structure of Tables
III-VI in the paper.
```
```powershell
python run_experiment.py --episodes 100
python analyze_results.py
```
Expect roughly one overnight run for the full grid, matching the paper's stated budget.

---

# PART 5 — Task split: Amal and Santhosh

**Amal — pilot, math validation, statistics**
- Phase 0 (own machine setup)
- Phase 1 (pilot harness + run)
- Phase 1b (Algorithm 2, Δq measurement — this is the part that makes Section V a real result)
- Phase 5, stats half (Wilson CIs, Cohen's h, regenerate Fig. 2 once Santhosh's `results.csv` lands)
- Owns writing up Sections V and VI in the paper

**Santhosh — systems build**
- Phase 2 (tool world, buffer, link monitor)
- Phase 3 + 3b (Docker testbed, replication-lag forwarder)
- Phase 4 (Algorithm 1 + baselines, feature-flagged) — use Antigravity here
- Phase 5, run half (execute the overnight factorial run, hand `results.csv` to Amal)
- Owns writing up Sections IV and VII in the paper

**Only two hard sync points:** after Phase 1 (Amal's decision-rule result should inform how Santhosh frames Phase 4's ablation priorities), and after Phase 5's run (Santhosh → Amal handoff of `results.csv`). Everything else runs in parallel on two machines.

---

# PART 6 — When a result doesn't come out the way the paper predicts

This will probably happen somewhere — that's normal, not a failure. Here's what to do in each case, honestly:

- **Pilot failure rate lands under 5%** (paper predicted >15%): don't force it. Follow the paper's own decision rule — drop the semantic-key claim as the headline and lead Section VI with C2/C3 instead. Update the abstract's opening sentence accordingly. This is a legitimate paper, just a differently-shaped one.
- **Kinbridge-Sync doesn't clearly beat a baseline in some cell**: report the actual number with its Wilson CI. If the CI overlaps the baseline's CI, say so explicitly — "no significant difference at this cell" is a true, useful, publishable sentence. Don't round in your own favor.
- **α ≈ β doesn't hold, and the general bisection (Algorithm 2, step 4) gives a weird τ\***: check the fitted α, β first — if the log-log regression's R² is poor, the power-law approximation itself may not fit your tool classes well; say that in Limitations rather than reporting a τ\* you don't trust.
- **The ablation ladder doesn't show monotonic improvement** (e.g., adding the epoch guard doesn't reduce unsafe-action rate): this usually means two flags interact — worth a sentence in Threats to Validity noting the mechanisms aren't perfectly independent, rather than hiding the row.
- **General rule:** every number that goes into the paper should come with its Wilson CI or it isn't a real result yet — a single run's raw percentage without an interval is not enough to update a table.

Once each phase's script has printed its table, the mechanical last step is: replace the dashes in the corresponding LaTeX table with the real numbers and CIs, swap in the regenerated `fmr_fsr_curve.pdf`, and change the "planned"/"we expect" language in the surrounding prose to plain statements of what was measured — including saying so plainly if a prediction didn't hold.