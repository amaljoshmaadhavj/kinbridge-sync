# Kinbridge-Sync: Windows 11 Build Guide, v2
### (Now with the math-validation gap closed, and tasks split for two: Amal & Santhosh)

---

## 0. What changed from the first version, and why

The first runbook got you real numbers for Tables II–VI. It did **not**
include any task that actually tests the two "novel formula" claims in
Section V of the paper (the risk-minimization threshold τ\*, and the
degradation stopping rule). A paper that derives a formula but never
checks it against its own experiment is an easy target for a reviewer.
This version adds that missing piece — **Phase 1b** below — plus a new,
genuinely provable extension to the math itself (Lemma 1 + Algorithm 2),
so the closed-form claim covers the general case, not just the special
case the original derivation handled.

**Honesty check, stated plainly:** the underlying technique (a
cost-weighted threshold, and an optimal-stopping rule) is standard
decision theory — it is not new to the world. What is novel is deriving
the specific closed form for *this* problem (tool-call idempotency
thresholding) and proving it's the unique global optimum, which nobody
has published for this setting before, as far as we've checked. That is
a legitimate, normal-sized claim for a systems paper's theory
contribution — no bigger and no smaller than that.

---

## 1. Lemma 1 (Uniqueness) — the new math to add to the paper

**Claim.** For $R(\tau) = w_d \tau^\beta + w_m (1-\tau)^\alpha$ (dropping
the constant $w_s P_{\text{stale}}$ term, which doesn't depend on $\tau$),
with $\alpha, \beta > 1$ and $w_d, w_m > 0$, on the open interval
$\tau \in (0,1)$:

$$R''(\tau) = w_d\beta(\beta-1)\tau^{\beta-2} + w_m\alpha(\alpha-1)(1-\tau)^{\alpha-2} \;\geq\; 0$$

Both terms are non-negative (in fact strictly positive, since $\alpha,\beta>1$
makes $(\alpha-1),(\beta-1)>0$ and the power terms are positive on the open
interval), so $R$ is **strictly convex** on $(0,1)$.

Boundary behavior of the first derivative:
$$R'(0^+) = -w_m\alpha < 0, \qquad R'(1^-) = +w_d\beta > 0$$

$R'$ is continuous and strictly increasing (since $R''\geq0$), so by the
Intermediate Value Theorem it crosses zero exactly once. **Conclusion:**
Eq. 14's stationary point (special case $\alpha=\beta$) — or the
numerically-solved root for $\alpha\neq\beta$ — is the unique **global**
minimum of $R(\tau)$, not merely a local one. This also means a plain
bisection search is guaranteed to converge, no fancy solver needed.

## 2. Algorithm 2 (new) — turning the formula into a runnable procedure

This is the piece that was missing: a concrete, implementable algorithm
that takes real pilot data and produces a calibrated $\tau^\star$,
instead of leaving the formula as a standalone equation nobody runs.

```
Algorithm 2: Calibrate τ* from measured FMR/FSR data
Input: (τ_i, FMR_i, FSR_i) samples from the Phase 1 pilot sweep,
       cost ratio w_d/w_m (a deployment choice, default 4)
Output: τ*, the calibrated threshold, plus fitted α, β

1. Fit β: linear-regress log(FSR_i) against log(τ_i)           → slope = β
2. Fit α: linear-regress log(FMR_i) against log(1 - τ_i)   → slope = α
3. If |α - β| < ε (e.g. ε = 0.05):
       τ* = closed_form(w_d, w_m, α, β)        # Eq. 14, the α=β case
4. Else (general case, uses Lemma 1's guarantee):
       lo, hi = 0.001, 0.999
       repeat ~40 times (bisection on R'(τ)):
           mid = (lo + hi) / 2
           if R'(mid) < 0:  lo = mid   # still descending
           else:            hi = mid   # already ascending
       τ* = (lo + hi) / 2
5. Return τ*, α, β
```

This is a dozen lines of Python (`numpy.polyfit` on log-log data for
steps 2–3, a `while` loop for step 4) — small, but it's the thing that
makes the formula an actual contribution you tested rather than a
formula you derived and left alone.

---

## 3. Updated phase list (adds Phase 1b, details the two tasks that were vague before)

| Phase | What | Produces |
|---|---|---|
| 0 | Environment setup (WSL2, Docker Desktop, Ollama, Python) | Working dev environment |
| 1 | Pilot study: 3 tools × 19 prompts × key schemes | Table II, raw (τ, FMR, FSR) samples |
| **1b (new)** | **Fit α, β via Algorithm 2 on Phase 1's data; separately, measure Δq (quality gap between the 7B and 1.5B Ollama models on a small held-out task set) for Eq. 15/16** | **Validated τ\*, validated degradation threshold — this is what makes Section V an empirical result, not just algebra** |
| 2 | Tool world, effect ledger, action buffer, link monitor | Testable components |
| 3 | Docker network testbed + outage injection | Runnable testbed |
| 3b (new detail) | Replication-lag forwarder: a small async script that copies edge-A's ledger writes to edge-B's Redis after `time.sleep(lag_ms/1000)` — this is what makes the C3 lag knob in Table V real instead of assumed | Working C3 isolation harness |
| 4 | Algorithm 1 (Reintegration Service) + 2 baselines, **built with feature flags** (`--enable-intent-keys`, `--enable-staleness-check`, `--enable-epoch-guard`) so Phase 5's ablation ladder is just flag combinations of one codebase, not six separate implementations | Systems under test, ablation-ready |
| 5 | Full factorial run + ablation ladder + stats + figures, **including running Algorithm 2 for real per-tool-class τ\* and regenerating Fig. 2 from measured, not assumed, data** | Tables III–VI, final Fig. 2, validated Eq. 14 |

Everything else from the first runbook (Part A machine setup, Part B
tool config for opencode/Antigravity, Part D "plugging numbers back into
the paper") is unchanged and still applies — see the original
`WINDOWS_BUILD_RUNBOOK.md`. This file adds to it, doesn't replace it.

---

## 4. Task split: Amal and Santhosh

Two people, roughly even effort, minimal blocking dependencies between
them after Phase 1.

### Amal — pilot, math validation, statistics
| Phase | Task | Tools |
|---|---|---|
| 0 | Set up own machine: WSL2, Docker Desktop, Ollama, Python venv | winget, Docker Desktop |
| 1 | Build and run the pilot harness (3 tools × 19 prompts × key schemes); apply the paper's decision rule | Ollama, opencode |
| **1b** | **Implement Algorithm 2 (α/β fitting + bisection); measure Δq between the two Ollama model sizes; validate τ\* against the measured FMR/FSR curve** | numpy, scipy, opencode |
| 5 (stats half) | Compute Wilson CIs and Cohen's h once Santhosh's experiment run produces `results.csv`; regenerate Fig. 2 from real per-tool-class data using the fitted α, β | scipy, statsmodels, matplotlib |
| — | Own the paper's Section V and Section VI writing — turn Amal's own output directly into the LaTeX table/figure updates | — |

### Santhosh — systems build, testbed, algorithm implementation
| Phase | Task | Tools |
|---|---|---|
| 2 | Build tool-world API, effect ledger, action buffer, link monitor (+ unit tests) | FastAPI, SQLite, pytest |
| 3 | Docker Compose testbed (client, edge-A, edge-B), `tc netem` / `iptables` scripting inside containers | Docker Desktop (WSL2 backend) |
| **3b** | **Build the replication-lag forwarder for C3** | Redis, asyncio |
| 4 | Implement Algorithm 1 (Reintegration Service) **with feature flags**, plus the two baseline variants as flag subsets | Antigravity (for this file specifically), sentence-transformers, pytest |
| 5 (run half) | Run the full factorial experiment + ablation ladder (unattended overnight), hand `results.csv` to Amal | Docker, Python |
| — | Own the paper's Sections IV and VII (architecture, task table) updates | — |

**Sync points** (the only two places one person blocks the other):
1. After Phase 1 — Amal's decision rule result determines whether Section
   V leads with C1 or C2/C3; Santhosh should know this before starting
   Phase 4's feature-flag design, since it affects which ablation rows
   matter most.
2. After Phase 5's run — Santhosh hands `results.csv` to Amal for the
   stats pass; this is the only hard handoff in the whole plan.

Everything else (Phase 2/3 for Santhosh, Phase 1b for Amal) can run in
parallel on two separate machines with no coordination needed.

---

## 5. What "the paper is complete" actually means once this is done

- Every dashed template table (II–VI) has real numbers.
- Fig. 2 is a measured curve, not an illustration.
- Section V's τ\* formula has been run against real data (Phase 1b),
  not just derived on paper — and now covers the general α≠β case via
  Lemma 1 + Algorithm 2, not just the α=β special case the first draft
  handled.
- Section VII's task table can be updated from "planned" to "completed,"
  with Amal and Santhosh named as the people who ran each phase.
- Nothing in Sections I–IV (the model, the impossibility proof, the
  architecture) needs to change — none of that depended on the
  experiment, and it was already correct.

What this does **not** do: turn a course/project-scale evaluation into a
venue-grade one. Single-laptop Docker emulation, three tool classes, and
a 1,200-episode budget are appropriate for what this is — don't oversell
it as more than that in the abstract once real numbers go in.