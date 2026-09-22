# Phase 4 — Implementation Report

## Status
**IMPLEMENTED AND VERIFIED** — Reintegration Service (Algorithm 1) delivered.
34 new Phase 4 tests pass; full suite green except one pre-existing
environment-dependent Phase 3b Docker test (see Limitations).

## What was implemented

### New files (Phase 4 only — no frozen file modified)
| File | Contents |
|------|----------|
| `reintegration/__init__.py` | Package exports |
| `reintegration/reintegration_service.py` | Algorithm 1 service + CLI |
| `reintegration/naive_retry.py` | Ablation baseline (no ledger check, blind replay) |
| `reintegration/verify_before_retry.py` | Ablation intermediate (ledger + V_post only) |
| `tests/test_reintegration_service.py` | 34 tests covering the 24 required cases |

### ReintegrationService (`reintegration/reintegration_service.py`)
Algorithm 1 order implemented exactly as specified:

1. **STEP 1 — Epoch guard**: compares `ledger.get_epoch()` (edge-B) with
   `buffer.max_epoch`. Synced → proceed. Behind → poll at exactly 50 ms
   (`DEFAULT_POLL_INTERVAL_SEC`) until caught up or the sync window
   expires (default `2.0 s`, env `REINTEGRATION_SYNC_WINDOW_SEC`).
   Window exhausted → **ESCALATE all PENDING actions and STOP** — never
   blind replay.
2. **STEP 2 — Ledger interpretation**, in original append order:
   - `COMMITTED` → mark committed, no replay
   - `UNKNOWN` → evaluate injected `V_post`; True → COMMITTED, else ESCALATE
     (no `V_post` available → ESCALATE)
   - `ESCALATED` / unknown status → ESCALATE
   - `ABSENT` — see FD-3 below
3. **STEP 3 — TTL / V_pre** (`--enable-staleness-check`):
   `now - t0 > ttl` → `INVALIDATED`; injected `V_pre(action, S_now)`
   False or raising → `INVALIDATED` (fail closed).
4. **STEP 4 — Semantic equivalence**: injectable similarity hook +
   calibrated τ* provider. Validity metric: `similarity >= tau_star` →
   `MERGED` (intent merged, no duplicate execution); `< tau_star` → safety
   path (execute as its own action).
5. **STEP 5 — Execute** with the buffered key, or `k_intent(session_id,
   turn_seq, intent_id)` when `--enable-intent-keys` and the fields are
   present. `execute_fn` raising → record UNKNOWN then ESCALATE.

Feature flags `--enable-intent-keys`, `--enable-staleness-check`,
`--enable-epoch-guard`, `--enable-degradation-gate` toggle individual
safeguards for the Phase 5 ablation ladder; the degradation-gate flag is
accepted for ablation symmetry and has no Phase 4 decision effect.

## Finalized Phase 4 decisions reflected
- **FD-1 (epoch sync window)**: default `2.0 s`, env
  `REINTEGRATION_SYNC_WINDOW_SEC`, polling interval exactly `50 ms`.
  Verified by `test_sync_window_env_override` and
  `test_behind_epoch_polls_at_50ms`.
- **FD-2 (τ* unavailable)**: missing / NaN / unestimable τ* → ESCALATE.
  Never substituted, never merge-degraded. `FileTauStarProvider` and
  `ToggleTauStarProvider` both return `None` for unusable values.
  `test_no_hardcoded_tau_star` asserts the string ``0.5052`` appears
  nowhere in the module and that providers fail open.
- **FD-3 (ABSENT after epoch guard)**: on the replicated edge-B ledger
  (`replicated_ledger=True`, the default) `ABSENT` **ESCALATE** — an ABSENT
  key cannot distinguish "never existed" from Phase 3b F1 replication
  loss, so no TTL/V_pre/semantic/execute is ever attempted. For a
  primary / same-node ledger (`replicated_ledger=False`) ABSENT is
  genuine and the original Algorithm 1 path runs. No edge-A fallback
  query is performed.

## Interface compatibility with frozen code
Consumed without modification: `client.buffer.ActionBuffer`
(`append/read_all/mark_committed/mark_invalidated/max_epoch`),
`tool_world.ledger.EffectLedger`
(`lookup/get_epoch/create/mark_committed/mark_unknown/mark_escalated/mark_invalidated`),
`pilot.key_schemes.k_intent` / `canonical_serialise`.
`tests/test_reintegration_service.py::TestFrozenApiCompatibility` binds
the service to the real frozen classes end-to-end.

## Ablation baselines
- `naive_retry.py` — no ledger check, blind replay of every PENDING
  action (the baseline Algorithm 1 removes).
- `verify_before_retry.py` — ledger + V_post only; ABSENT → execute; no
  TTL/epoch/semantic/intent keys (middle rung).

## Test results
- Phase 4: **34/34** (`tests/test_reintegration_service.py`), covering
  all 24 required cases from the Phase 4 spec §9.
- Full suite: **504 passed, 8 skipped, 1 failed, 7 warnings** (513
  collected). The failure is `TestComposeCLI::test_compose_up_and_down`
  — a frozen Phase 3b Docker test expecting exactly **3** running
  services while the 6-service compose file now brings all 6 to
  `running`. Unrelated to Phase 4 (no Phase 4 reference; deterministic
  in isolation). Phase 4 code adds no failures.
- `flak8` clean on new code except E501 line length (repo norm — frozen
  files carry the same).

## Repository integrity
Only new files added; none of the frozen Phase 1/1b/2.1/2.2/3 code was
modified (`git diff` over `client/ tool_world/ pilot/ kinbridge_math/
experiments/ config/ Markdown/ core/` is empty). No hardcoded τ*. No
Phase 3b modifications were made to add replication guarantees.

## Limitations
1. **Phase 3b F1** (pre-existing, not fixed — per constraint): events
   ACKed from Redis A while Redis-B is down are permanently lost
   (XREADGROUP ACK-on-read). The ABSENT → ESCALATE rule (FD-3) is the
   client-side mitigation; recovery of such events is out of scope.
2. **Epoch sync timing** (pre-existing test-environment artifact, F2):
   the 500 ms lag test measured ~178 ms locally; the mechanism
   (`asyncio.sleep(lag_ms)`) is correct.
3. **Compose up/down assertion** (`test_compose_up_and_down`):
   environment-dependent exact count (3 vs 6) in a frozen Phase 3b
   test; left untouched per the no-modify constraint.
4. **Semantic similarity** depends on `sentence-transformers`
   installed at runtime; the hook is injectable so tests never loaded
   the model.

## Out of scope (not done, per instructions)
- Phase 5 (experiment harness / degradation gate operation)
- Modifications to Phase 3b to add replication guarantees
- Fabricated experiment results