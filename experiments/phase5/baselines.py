"""Baseline drivers for Phase 5 (B5).

Faithful in-process implementations of the four methods evaluated in
the factorial (method x outage) grid:

  cold_restart
      The client retains NO buffered action state across the gap.
      ActionBuffer state is discarded at the restart boundary; the
      client does not replay buffered pre-gap actions.  After restart
      the client performs the newly planned post-gap action.  Whether
      that post-gap action duplicates an already-committed pre-gap
      effect is decided by ground truth (effect-identity accounting in
      the harness).  Distinct from naive_retry: nothing is replayed.

  naive_retry
      Every buffered action is executed regardless of ledger state,
      staleness, or connectivity (mirrors the frozen
      reintegration/naive_retry.py CLI logic).

  verify_before_retry
      Checks the ledger + V_post only; ABSENT/UNKNOWN is treated as
      permission to execute (mirrors the frozen
      reintegration/verify_before_retry.py CLI logic — no TTL, no
      V_pre, no epoch guard).

  kinbridge_sync
      The full Phase 4 ReintegrationService with the feature flags
      chosen from the ablation row (frozen service, injected
      hooks/clock/tau*).

All four methods run through the SAME harness (tool server, ledger
fabric, replication engine, ground truth), as required by the
research specification ("all baselines must use the same underlying
tool world and experiment harness").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from client.buffer import ActionBuffer
from experiments.phase5.scenario import AblationRow, EpisodeTemplate
from reintegration.reintegration_service import ReintegrationService
from tool_world.ledger import EffectLedger


@dataclass(frozen=True)
class BaselineDecisions:
    """Outcome of a baseline run over the buffered actions.

    ``by_key`` maps the buffer key -> per-action decision string.
    ``epoch_guard_ok`` reflects the Algorithm 1 epoch guard when used.
    ``actions`` is the service ordering (buffer append order).
    """

    by_key: dict[str, str]
    epoch_guard_ok: bool = True
    degraded: bool = False


class ColdRestartBaseline:
    """B5: cold restart — discard the buffer, replay nothing."""

    method = "cold_restart"

    def run(self, buffer: ActionBuffer, ledger: EffectLedger) -> BaselineDecisions:
        # The buffer is deliberately NOT replayed; statuses remain as the
        # pre-gap client wrote them.  Every buffered action is dropped.
        decisions = {
            rec["key"]: "DROPPED" for rec in buffer.read_all()
        }
        return BaselineDecisions(by_key=decisions, epoch_guard_ok=True)


class NaiveRetryBaseline:
    """Naive replay: execute every buffered action, ledger unconsulted."""

    method = "naive_retry"

    def __init__(self, execute_fn: Callable[[dict[str, Any]], Any]) -> None:
        self._execute_fn = execute_fn
        self._replayed: list[str] = []

    @property
    def replayed(self) -> list[str]:
        return list(self._replayed)

    def run(self, buffer: ActionBuffer, ledger: EffectLedger) -> BaselineDecisions:
        decisions: dict[str, str] = {}
        for rec in buffer.read_all():
            key = rec["key"]
            if rec.get("status") in ("COMMITTED", "INVALIDATED"):
                decisions[key] = "COMMITTED"
                continue
            self._execute_fn(rec)          # blindly replay
            ledger.mark_committed(key)
            buffer.mark_committed(key)
            self._replayed.append(key)
            decisions[key] = "REPLAYED"
        return BaselineDecisions(by_key=decisions, epoch_guard_ok=True)


class VerifyBeforeRetryBaseline:
    """Verify-then-retry: ledger status only; ABSENT/UNKNOWN executes."""

    method = "verify_before_retry"

    def __init__(self, execute_fn: Callable[[dict[str, Any]], Any]) -> None:
        self._execute_fn = execute_fn
        self._replayed: list[str] = []

    @property
    def replayed(self) -> list[str]:
        return list(self._replayed)

    def run(self, buffer: ActionBuffer, ledger: EffectLedger) -> BaselineDecisions:
        decisions: dict[str, str] = {}
        for rec in buffer.read_all():
            key = rec["key"]
            if rec.get("status") in ("COMMITTED", "INVALIDATED"):
                decisions[key] = "COMMITTED"
                continue
            ledger_rec = ledger.lookup(key)
            status = (ledger_rec or {}).get("status", "ABSENT")
            if status == "COMMITTED":
                buffer.mark_committed(key)
                decisions[key] = "COMMITTED"
                continue
            # ABSENT / UNKNOWN -> execute (no TTL, no V_pre, no semantic).
            self._execute_fn(rec)
            ledger.mark_committed(key)
            buffer.mark_committed(key)
            self._replayed.append(key)
            decisions[key] = "REPLAYED"
        return BaselineDecisions(by_key=decisions, epoch_guard_ok=True)


class KinbridgeSyncBaseline:
    """Full Algorithm 1 (Phase 4 service) with flags from the ablation row."""

    method = "kinbridge_sync"

    def __init__(
        self,
        template: EpisodeTemplate,
        row: Optional[AblationRow],
        tau_star_provider,
        v_pre,
        v_post,
        semantic_similarity,
        execute_fn: Callable[[dict[str, Any]], Any],
        clock: Callable[[], float],
        sleep_fn: Callable[[float], None],
        replicated_ledger: bool,
        use_v_post: bool,
        sync_window_s: Optional[float] = None,
    ) -> None:
        self._template = template
        self._row = row
        self._flags = AblationRow(
            name=row.name if row is not None else "full",
            key_scheme=row.key_scheme if row is not None else "intent",
            enable_intent_keys=(row.enable_intent_keys if row is not None else True),
            enable_staleness_check=(row.enable_staleness_check if row is not None else True),
            enable_epoch_guard=(row.enable_epoch_guard if row is not None else True),
            enable_degradation_gate=(row.enable_degradation_gate if row is not None else True),
        ) if row is not None else None
        self._tau_star_provider = tau_star_provider
        self._v_pre = v_pre
        self._v_post = v_post
        self._semantic_similarity = semantic_similarity
        self._execute_fn = execute_fn
        self._clock = clock
        self._sleep_fn = sleep_fn
        self._sync_window_s = sync_window_s
        self._replicated_ledger = replicated_ledger
        self._use_v_post = use_v_post

    @property
    def flags(self) -> Optional[AblationRow]:
        return self._flags

    def run(self, buffer: ActionBuffer, ledger: EffectLedger) -> BaselineDecisions:
        service = ReintegrationService(
            buffer=buffer,
            ledger=ledger,
            v_pre=self._v_pre,
            v_post=self._v_post,
            semantic_similarity=self._semantic_similarity,
            tau_star_provider=self._tau_star_provider,
            replanned_actions=[
                {
                    "tool": r.tool,
                    "args": dict(r.args),
                    "intent_id": r.intent_id,
                    "session_id": r.session_id,
                    "turn_seq": r.turn_seq,
                }
                for r in self._template.replanned_actions
            ],
            sync_window_sec=self._sync_window_s,
            enable_intent_keys=self._flags.enable_intent_keys if self._flags else True,
            enable_staleness_check=self._flags.enable_staleness_check if self._flags else True,
            enable_epoch_guard=self._flags.enable_epoch_guard if self._flags else True,
            enable_degradation_gate=self._flags.enable_degradation_gate if self._flags else True,
            replicated_ledger=self._replicated_ledger,
            use_v_post=self._use_v_post,
            execute_fn=self._execute_fn,
            clock=self._clock,
            sleep_fn=self._sleep_fn,
        )
        result = service.run()
        decisions = {}
        for rec, action in zip(buffer.read_all(), result.actions):
            decisions[rec["key"]] = action.value
        return BaselineDecisions(
            by_key=decisions,
            epoch_guard_ok=result.epoch_guard_ok,
        )


def build_baseline(
    method: str,
    template: EpisodeTemplate,
    row: Optional[AblationRow],
    tau_star_provider,
    v_pre,
    v_post,
    semantic_similarity,
    execute_fn: Callable[[dict[str, Any]], Any],
    clock: Callable[[], float],
    sleep_fn: Callable[[float], None],
    replicated_ledger: bool,
    use_v_post: bool,
    sync_window_s: Optional[float] = None,
) -> Any:
    """Factory selecting the baseline driver for a method name.

    Raises:
        ValueError: for an unknown method.
    """
    if method == "cold_restart":
        return ColdRestartBaseline()
    if method == "naive_retry":
        return NaiveRetryBaseline(execute_fn=execute_fn)
    if method == "verify_before_retry":
        return VerifyBeforeRetryBaseline(execute_fn=execute_fn)
    if method == "kinbridge_sync":
        return KinbridgeSyncBaseline(
            template=template,
            row=row,
            tau_star_provider=tau_star_provider,
            v_pre=v_pre,
            v_post=v_post,
            semantic_similarity=semantic_similarity,
            execute_fn=execute_fn,
            clock=clock,
            sleep_fn=sleep_fn,
            replicated_ledger=replicated_ledger,
            use_v_post=use_v_post,
            sync_window_s=sync_window_s,
        )
    raise ValueError(f"unknown method {method!r}")