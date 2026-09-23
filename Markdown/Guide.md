# Kinbridge-Sync: Windows 11 Build Guide (100% Free Tools)

This is the from-scratch version of the runbook, written specifically for
Windows 11, using the tools you already have: **opencode** and
**Antigravity**. Everything here costs $0 — no API keys to pay for, no
trial periods, no credit card.

One thing to understand up front, since it decides how the two tools get used:

| Tool | What it's good for here | Cost |
|---|---|---|
| **Ollama** (runs on Windows natively) | The LLM *being tested* inside the pilot and the factorial experiment. You need to call it hundreds/thousands of times with controlled temperature — a local model has no rate limit and no bill. | Free, unlimited, offline |
| **opencode** | Your coding agent for scaffolding files, writing tests, wiring the Docker setup. Point it at your local Ollama so it also costs nothing. | Free when pointed at Ollama |
| **Antigravity** | A second coding agent, backed by Google's free Gemini/Claude quota (it does **not** support local models — confirmed, it only offers a fixed curated model list). Good for the trickier single files (Algorithm 1) where a stronger cloud model helps, still $0 up to the free quota. | Free, quota-based |

So: Ollama plays two roles (the subject of the experiment, *and* optionally
opencode's brain). Antigravity is a separate, cloud-backed but free, second
pair of hands for harder files.

---

## Part A — One-time machine setup

### A1. Enable WSL2 (gives you real Linux, needed for `tc netem` / `iptables`)
Windows containers can't easily do Linux-style traffic shaping, but Docker
Desktop's WSL2 backend runs your containers as genuine Linux containers, so
`tc` and `iptables` work perfectly *inside* them regardless of the Windows
host. You don't need to hand-run Linux commands on the host at all.

Open **PowerShell as Administrator** and run:
```powershell
wsl --install
```
This installs WSL2 and Ubuntu by default. Reboot when prompted. After reboot, open the "Ubuntu" app once from the Start menu to finish setup (it'll ask you to create a Linux username/password — anything works).

Confirm it worked:
```powershell
wsl -l -v
```
You should see `Ubuntu` with `VERSION 2`.

### A2. Install Docker Desktop (free for personal/education use)
Download from **docker.com/products/docker-desktop** and run the installer.
- During install, keep "Use WSL 2 instead of Hyper-V" checked (this is the default on Windows 11).
- After install, open Docker Desktop → **Settings → Resources → WSL Integration** → enable integration with your Ubuntu distro → Apply & Restart.

Confirm it worked (PowerShell or Ubuntu terminal):
```powershell
docker run hello-world
```

### A3. Install Ollama (free, native Windows app)
```powershell
winget install --id Ollama.Ollama
```
It installs as a background Windows service on `http://localhost:11434` — no window to keep open. Then pull the models you'll need (all free downloads):
```powershell
ollama pull qwen2.5:7b-instruct
ollama pull qwen2.5:1.5b-instruct
```
The 7B model is the "subject" LLM the pilot tests. The 1.5B model doubles as the paper's "local SLM" stand-in for the degradation gate (Section IV-C) — using a genuinely smaller model here makes the quality-gap term Δq in Eq. 15 something you can actually measure instead of assume.

> If your machine has less than ~8GB free RAM, use `qwen2.5:1.5b-instruct` or `phi3:mini` for both roles — everything in the pilot still works, the divergence-rate measurement doesn't depend on model size.

### A4. Install Python and Git for Windows (both free)
```powershell
winget install --id Python.Python.3.12
winget install --id Git.Git
```
Then create your project folder and virtual environment:
```powershell
mkdir C:\kinbridge-sync
cd C:\kinbridge-sync
git init
python -m venv .venv
.venv\Scripts\activate
pip install fastapi uvicorn sentence-transformers scipy statsmodels numpy matplotlib redis requests
```

---

## Part B — Point your two coding agents at free backends

### B1. opencode → Ollama
Edit (or create) `%USERPROFILE%\.config\opencode\opencode.json`:
```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://localhost:11434/v1" },
      "models": {
        "qwen2.5:7b-instruct": {}
      }
    }
  }
}
```
Run `opencode` in your project folder and select the `ollama` provider / `qwen2.5:7b-instruct` model when prompted. Every call opencode makes from here is free and local.

### B2. Antigravity → its built-in free quota
Antigravity does not accept a local endpoint, so there's nothing to configure — just open the project folder in Antigravity and use its Agent Panel model selector (Gemini 3 Flash for quick edits, Gemini 3 Pro or Claude for the harder file). It's free up to Google's quota; if you hit a daily limit, switch back to opencode+Ollama for the rest of the session.

**Rule of thumb for which tool to use where:** opencode/Ollama for anything repetitive or scaffolding-heavy (it's unlimited); Antigravity for the one or two trickiest files where you want a stronger model's reasoning (Algorithm 1, the risk-minimization derivation check).

---

## Part C — Build order (mirrors Table VII in the paper, Windows-adjusted)

Do these in order. Each phase is a prompt you can hand to opencode or
Antigravity almost verbatim.

### Phase 0 — Verify the environment (do this yourself, no agent needed)
```powershell
docker run hello-world
ollama run qwen2.5:7b-instruct "say hi"
```
If both respond, you're ready.

### Phase 1 — Pilot study (produces Table II, real Fig. 2)
Prompt for opencode, run from `C:\kinbridge-sync`:
```
Create a folder pilot/ with:
- tools.py: 3 mock tool schemas (navigation, data-write, payment/dispatch)
  as Python functions with JSON-schema signatures compatible with Ollama's
  tool-calling API (see https://ollama.com/blog/tool-support for the format).
- prompts.json: 19 short task prompts, ~6-7 per tool class.
- run_pilot.py: connects to http://localhost:11434, for each prompt calls
  the model once to get tool call a1, then re-sends the conversation
  without a tool result (simulating a gap) to get a2. Compute whether
  hash(a1) != hash(a2) under three schemes: k_payload (hash of tool+args),
  k_bucket (hash of tool+args+floor(timestamp/60)), k_intent (a stable
  intent_id generated once and carried through, so it always matches by
  construction). Sweep temperature in [0.0, 0.7]. Write results to
  pilot_results.csv, one row per (prompt, scheme, temperature).
- analyze_pilot.py: read pilot_results.csv, compute per-condition failure
  rate + Wilson 95% CI (manual formula, no extra dependency needed beyond
  scipy), print a table matching Table II's exact row structure in the
  paper, and print which branch of the decision rule (>15%, 5-15%, <5%)
  was hit.
```
Run it:
```powershell
python pilot\run_pilot.py
python pilot\analyze_pilot.py
```
Apply the paper's decision rule (Section VI-A) to whatever comes back — this determines whether C1 stays the headline claim.

### Phase 2 — Non-novel plumbing (tool world, buffer, link monitor)
Prompt for opencode:
```
Create tool_world/ — FastAPI app (Uvicorn) exposing the 3 tool endpoints
from pilot/tools.py, backed by SQLite. Add an append-only effect_ledger
table: (key TEXT PRIMARY KEY, tool TEXT, args_hash TEXT, status TEXT,
committed_at REAL). Add a fault-injection query param that can force a
request to hang before responding.

Create client/buffer.py — append-only JSONL write-ahead log storing
(intent_id, key, tool, args, t0, ttl, v_pre_spec).

Create client/link_monitor.py implementing Eqs 3-5 exactly: EWMA RTT,
EWMA deviation, RTO = r_hat + 4*d_hat, and the UP/DOWN hysteresis FSM.
Include a pytest test file that feeds it a synthetic RTT trace with an
injected dropout and asserts the FSM transitions correctly.
```
Test locally (no Docker needed yet):
```powershell
cd tool_world
uvicorn main:app --reload
# in a second terminal:
pytest client\
```

### Phase 3 — The network testbed (Docker + WSL2 does the Linux networking)
Prompt for opencode:
```
Create docker-compose.yml with 3 services on a bridge network: client,
edge-a, edge-b. edge-a and edge-b each run the tool_world FastAPI app
and a Redis container. Give edge-a and edge-b cap_add: [NET_ADMIN,
NET_RAW] so tc and iptables work inside them. Base images should be
python:3.12-slim with iptables and iproute2 installed via apt.

Create netctl.py — a small script that docker-execs into a container and
runs `tc qdisc add dev eth0 root netem delay {ms}ms loss {pct}%` to shape
latency, and `iptables -I OUTPUT -j DROP` / `iptables -F` to toggle a
true blackout.

Create outage_model.py implementing the Gilbert-Elliott two-state Markov
chain described in the paper, driving netctl.py's blackout toggle.
```
Bring it up (this is where WSL2/Docker Desktop does the real work):
```powershell
docker compose up -d
docker compose ps
```

### Phase 4 — The actual contribution (Algorithm 1 + baselines)
This is the file worth handing to **Antigravity** instead of opencode, since
it's the one place a stronger model's reasoning is most useful — the
five nested conditions in Algorithm 1 need to be gotten exactly right.

Prompt for Antigravity:
```
Implement reintegration_service.py exactly per Algorithm 1 in the attached
paper: epoch guard first (escalate everything if the replica is behind and
doesn't catch up within a sync window), then per buffered action in issue
order: check ledger status -> if unknown, call a read-only V_post check ->
if still unknown, escalate -> if the ttl has expired or V_pre fails,
invalidate and notify -> otherwise check intent-similarity against the
newly re-planned actions using sentence-transformers all-MiniLM-L6-v2
(local, free, runs on CPU) with cosine similarity above threshold tau ->
execute. Write six unit tests, one per branch, using pytest.

Then create two baseline variants as clearly-marked subsets of the same
code: naive_retry.py (delete every check, always execute) and
verify_before_retry.py (keep only the V_post check, delete the ttl/V_pre
check and the epoch guard — this is the fair reimplementation of Mansoor
et al. described in the paper).
```
```powershell
pip install sentence-transformers
pytest reintegration_service\
```

### Phase 5 — Run the real factorial experiment
Prompt for opencode:
```
Create run_experiment.py: for each (method, outage_duration) in the
factorial grid from the paper (4 methods x 4 durations, 100 episodes
each), drive the Docker testbed via netctl.py and outage_model.py,
log dup/stale/escalation/committed outcomes to results.csv.

Create analyze_results.py: read results.csv, compute Wilson 95% CIs and
Cohen's h per cell using scipy.stats, and print/export tables matching
the exact structure of Tables III-VI in the paper. Also regenerate the
FMR/FSR figure (matplotlib) from real per-tool-class similarity-score
data collected during the run, replacing the illustrative curve.
```
```powershell
python run_experiment.py --episodes 100
python analyze_results.py
```
This is an unattended run — expect roughly one overnight session for the full 1,600-episode grid on a single laptop, matching the budget already stated in the paper (4 methods × 4 outage durations × 100 episodes).

---

## Part D — Getting the numbers back into the paper

Once each phase's script has run:
1. `analyze_pilot.py`'s printed table → **Table II remains unavailable/pending** (per the final research decision). Do not fabricate or substitute values; leave it as NOT_AVAILABLE / PENDING unless a defensible source is subsequently obtained.
2. `analyze_results.py`'s output → replace the dashes in **Tables III–VI** (`tab:headline`, `tab:decomp`, `tab:c3`, `tab:ablation`).
3. The regenerated `fmr_fsr_curve.pdf` → overwrite the existing file referenced by `\includegraphics` in the `.tex` — the LaTeX doesn't need to change, just the PDF it points to.
4. Change the wording: every place the paper currently says "planned," "template," "we expect," or "\todo{To be filled in during Phase X}" should become a plain statement of what was measured, and the Limitations section's "has not been run at the time of writing" sentence should be deleted or updated to describe what *was* run.

Nothing else in the paper (the model, the proofs, the architecture, the two
closed-form derivations) needs to change regardless of what the numbers turn
out to be — that content was correct before any experiment ran.

---

## Quick troubleshooting

| Problem | Fix |
|---|---|
| `docker run hello-world` fails | Docker Desktop isn't running, or WSL integration isn't enabled for your distro — check Settings → Resources → WSL Integration |
| `ollama run` hangs or is very slow | Model is too big for your RAM — switch to `qwen2.5:1.5b-instruct` or `phi3:mini` |
| `tc` / `iptables` "command not found" inside a container | Base image is missing `iproute2` / `iptables` — add `RUN apt-get update && apt-get install -y iproute2 iptables` to that service's Dockerfile |
| `tc` / `iptables` "operation not permitted" inside a container | Missing `cap_add: [NET_ADMIN, NET_RAW]` in docker-compose.yml for that service |
| opencode can't reach Ollama | Confirm the Ollama tray icon is running, then `curl http://localhost:11434/api/tags` in PowerShell to sanity-check the API is up |