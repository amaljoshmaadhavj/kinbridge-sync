"""Phase 4 tests for the Reintegration Service (Algorithm 1).

Covers the 24 required cases from ``Markdown/Work .md`` Phase 4 §9:

  1.  COMMITTED record → no replay, buffer marked committed
  2.  UNKNOWN + V_post True  → COMMITTED
  3.  UNKNOWN + V_post False → ESCALATED
  4.  UNKNOWN + V_post unavailable → ESCALATED
  5.  epoch guard: synced → passes immediately
  6.  epoch guard: behind → polls at exactly 50 ms
  7.  epoch guard: behind beyond sync window → ESCALATE all and STOP
  8.  sync window env override honored
  9.  ABSENT + primary ledger → proceeds to safety/execute (genuine absent)
 10.  ABSENT + replicated ledger (Phase 3b F1) → ESCALATE; no TTL/V_pre/
      semantic/execute is attempted
 11.  tau* missing         → ESCALATE
 12.  tau* NaN             → ESCALATE
 13.  tau* unestimable     → ESCALATE
 14.  tau* never hardcoded — no fallback threshold substitution
 15.  TTL expired          → INVALIDATED
 16.  V_pre False          → INVALIDATED
 17.  semantic match    (sim >= tau*) → MERGED
 18.  semantic non-match (sim <  tau*) → execute (safety path)
 19.  intent-key preference when enabled
 20.  ordering preserved
 21.  multi-action run
 22.  CLI --tau-star / provider integration
 23.  V_pre raising    → INVALIDATED (fail closed)
 24.  execute_fn raising → ESCALATED via UNKNOWN

Also verified: frozen API compatibility with the real ActionBuffer and
EffectLedger classes, and buffer statuses are never misrepresented for
escalations.
"""

from __future__ import annotations

import time
from typing import Any

from client.buffer import ActionBuffer
from reintegration.reintegration_service import (
    DEFAULT_POLL_INTERVAL_SEC,
    DEFAULT_SYNC_WINDOW_SEC,
    FileTauStarProvider,
    ReintegrationAction,
    ReintegrationService,
    ToggleTauStarProvider,
    get_sync_window_sec,
)
from tool_world.ledger import EffectLedger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeClock:
    """Mutable time source for deterministic tests."""

    def __init__(self, start: float = 50.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, delta: float) -> None:
        self.now += delta


class FakeSleep:
    """Sleep that advances a FakeClock (no real waiting)."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        self.clock.advance(seconds)


class FakeLedger:
    """In-memory EffectLedger stand-in with the same public surface."""

    def __init__(self, records: dict[str, dict[str, Any]] | None = None, epoch: int = 10) -> None:
        self._records = dict(records or {})
        self._epoch = epoch

    def lookup(self, key: str) -> dict[str, Any] | None:
        return self._records.get(key)

    def get_epoch(self) -> int:
        return self._epoch

    def create(self, key: str, tool: str, args_hash: str, ttl: float = 0.0,
               intent_id: str | None = None) -> dict[str, Any]:
        rec = {"key": key, "tool": tool, "args_hash": args_hash, "status": "ABSENT",
               "ttl": ttl, "intent_id": intent_id}
        if key not in self._records:
            self._records[key] = rec
        return rec

    def mark_committed(self, key: str, epoch: int | None = None) -> bool:
        return self._bump(key, "COMMITTED")

    def mark_unknown(self, key: str) -> bool:
        return self._bump(key, "UNKNOWN")

    def mark_escalated(self, key: str) -> bool:
        return self._bump(key, "ESCALATED")

    def mark_invalidated(self, key: str) -> bool:
        return self._bump(key, "INVALIDATED")

    def _bump(self, key: str, status: str) -> bool:
        if key not in self._records:
            self._records[key] = {"key": key, "status": "ABSENT"}
        self._records[key]["status"] = status
        return True


def make_record(**overrides: Any) -> dict[str, Any]:
    """Return a realistic buffered-action record."""
    base: dict[str, Any] = {
        "intent_id": "INT-001",
        "key": "key-1",
        "tool": "navigate_to",
        "args": {"x": 1.0, "y": 2.0},
        "t0": 0.0,
        "ttl": 60.0,
        "v_pre_spec": {},
        "status": "PENDING",
        "epoch": 1,
    }
    base.update(overrides)
    return base


def append_buffer(
    buf: ActionBuffer,
    *records: dict[str, Any],
    add_intent_fields: bool = False,
) -> None:
    """Append records to a real ActionBuffer via its public API."""
    for r in records:
        rec = buf.append(
            intent_id=r["intent_id"],
            key=r["key"],
            tool=r["tool"],
            args=r["args"],
            t0=r["t0"],
            ttl=r["ttl"],
            v_pre_spec=r.get("v_pre_spec", {}),
            status=r.get("status", "PENDING"),
            epoch=r.get("epoch", 0),
        )
        if add_intent_fields:
            rec["session_id"] = r.get("session_id", "SESS-1")
            rec["turn_seq"] = r.get("turn_seq", 1)


# ---------------------------------------------------------------------------
# 1–4: COMMITTED / UNKNOWN / V_post handling
# ---------------------------------------------------------------------------


class TestCommittedAndUnknown:
    def test_committed_is_not_replayed(self, tmp_path) -> None:
        buf = ActionBuffer(tmp_path / "wal.jsonl")
        rec = make_record(key="key-c", intent_id="INT-C")
        append_buffer(buf, rec)
        clock = FakeClock()
        service = ReintegrationService(
            buffer=buf,
            ledger=FakeLedger({"key-c": {"status": "COMMITTED"}}),
            clock=clock, sleep_fn=FakeSleep(clock),
            enable_epoch_guard=True,
        )
        result = service.run()
        assert result.actions == [ReintegrationAction.COMMITTED]
        assert buf.read_all()[0]["status"] == "COMMITTED"

    def test_unknown_with_vpost_true_commits(self, tmp_path) -> None:
        buf = ActionBuffer(tmp_path / "wal.jsonl")
        append_buffer(buf, make_record(key="key-u"))
        clock = FakeClock()
        service = ReintegrationService(
            buffer=buf,
            ledger=FakeLedger({"key-u": {"status": "UNKNOWN"}}),
            v_post=lambda rec: True,
            clock=clock, sleep_fn=FakeSleep(clock),
        )
        result = service.run()
        assert result.actions == [ReintegrationAction.COMMITTED]
        assert buf.read_all()[0]["status"] == "COMMITTED"
        assert service._ledger.lookup("key-u")["status"] == "UNKNOWN"  # ledger untouched

    def test_unknown_with_vpost_false_escalates(self, tmp_path) -> None:
        buf = ActionBuffer(tmp_path / "wal.jsonl")
        append_buffer(buf, make_record(key="key-u"))
        ledger = FakeLedger({"key-u": {"status": "UNKNOWN"}})
        service = ReintegrationService(
            buffer=buf, ledger=ledger,
            v_post=lambda rec: False,
            clock=FakeClock(), sleep_fn=FakeSleep(FakeClock()),
        )
        result = service.run()
        assert result.actions == [ReintegrationAction.ESCALATED]
        assert ledger.lookup("key-u")["status"] == "ESCALATED"
        # buffer NOT marked committed for an escalation
        assert buf.read_all()[0]["status"] == "PENDING"

    def test_unknown_without_vpost_escalates(self, tmp_path) -> None:
        buf = ActionBuffer(tmp_path / "wal.jsonl")
        append_buffer(buf, make_record(key="key-u"))
        ledger = FakeLedger({"key-u": {"status": "UNKNOWN"}})
        service = ReintegrationService(
            buffer=buf, ledger=ledger, v_post=None,
            clock=FakeClock(), sleep_fn=FakeSleep(FakeClock()),
        )
        result = service.run()
        assert result.actions == [ReintegrationAction.ESCALATED]
        assert ledger.lookup("key-u")["status"] == "ESCALATED"


# ---------------------------------------------------------------------------
# 5–8: Epoch guard
# ---------------------------------------------------------------------------


class TestEpochGuard:
    def _mk(self, buf: ActionBuffer, ledger_epoch: int, clock: FakeClock, sleep: FakeSleep, window: float):
        return ReintegrationService(
            buffer=buf, ledger=FakeLedger(epoch=ledger_epoch),
            sync_window_sec=window,
            poll_interval_sec=DEFAULT_POLL_INTERVAL_SEC,
            enable_epoch_guard=True,
            clock=clock, sleep_fn=sleep,
        )

    def test_synced_epoch_passes_immediately(self, tmp_path) -> None:
        buf = ActionBuffer(tmp_path / "wal.jsonl")
        append_buffer(buf, make_record(key="k", epoch=3))
        clock = FakeClock()
        sleep = FakeSleep(clock)
        service = self._mk(buf, ledger_epoch=3, clock=clock, sleep=sleep, window=2.0)
        assert service._epoch_guard() is True
        assert sleep.calls == []  # already synced

    def test_behind_epoch_polls_at_50ms(self, tmp_path) -> None:
        buf = ActionBuffer(tmp_path / "wal.jsonl")
        append_buffer(buf, make_record(key="k", epoch=5))

        ledger = FakeLedger(epoch=3)

        def slow_epoch() -> int:
            return 3 if clock.now < 50.2 else 5  # catches up just inside the 2.0s window

        clock = FakeClock()
        sleep = FakeSleep(clock)
        ledger.get_epoch = slow_epoch  # type: ignore[assignment, method-assign]
        service = ReintegrationService(
            buffer=buf, ledger=ledger,
            sync_window_sec=2.0, poll_interval_sec=DEFAULT_POLL_INTERVAL_SEC,
            enable_epoch_guard=True, clock=clock, sleep_fn=sleep,
        )
        assert service._epoch_guard() is True
        assert set(sleep.calls) == {DEFAULT_POLL_INTERVAL_SEC}
        assert len(sleep.calls) >= 2

    def test_behind_beyond_window_escalates_all_and_stops(self, tmp_path) -> None:
        buf = ActionBuffer(tmp_path / "wal.jsonl")
        append_buffer(buf, make_record(key="a", epoch=3))
        append_buffer(buf, make_record(key="b", intent_id="INT-B", tool="log_inspection", epoch=5))
        ledger = FakeLedger(epoch=3)
        executed: list[dict[str, Any]] = []

        clock = FakeClock()
        sleep = FakeSleep(clock)
        service = ReintegrationService(
            buffer=buf, ledger=ledger,
            sync_window_sec=0.2, poll_interval_sec=0.05,
            enable_epoch_guard=True,
            v_pre=lambda rec, state: True,
            tau_star_provider=ToggleTauStarProvider(0.9),
            execute_fn=lambda payload: executed.append(payload),
            clock=clock, sleep_fn=sleep,
        )
        result = service.run()

        assert result.epoch_guard_ok is False
        assert result.actions == [ReintegrationAction.ESCALATED, ReintegrationAction.ESCALATED]
        assert executed == []  # never replayed
        for r in buf.read_all():
            assert r["status"] == "PENDING"  # never misrepresented as committed
        assert ledger.lookup("a")["status"] == "ESCALATED"
        assert ledger.lookup("b")["status"] == "ESCALATED"

    def test_sync_window_env_override(self, monkeypatch) -> None:
        monkeypatch.setenv("REINTEGRATION_SYNC_WINDOW_SEC", "7.5")
        assert get_sync_window_sec() == 7.5
        service = ReintegrationService(
            buffer=ActionBuffer.__new__(ActionBuffer),  # never touched
            ledger=FakeLedger(),
        )
        assert service._sync_window_sec == 7.5
        monkeypatch.delenv("REINTEGRATION_SYNC_WINDOW_SEC")
        assert get_sync_window_sec() == DEFAULT_SYNC_WINDOW_SEC

    def test_default_sync_window_is_200(self) -> None:
        assert DEFAULT_SYNC_WINDOW_SEC == 2.0

    def test_buffer_empty_skips_epoch_wait(self, tmp_path) -> None:
        buf = ActionBuffer(tmp_path / "empty.jsonl")  # real buffer, no records
        ledger = FakeLedger(epoch=1)
        clock = FakeClock()
        sleep = FakeSleep(clock)
        service = ReintegrationService(
            buffer=buf, ledger=ledger, enable_epoch_guard=True,
            clock=clock, sleep_fn=sleep,
        )
        assert service._epoch_guard() is True
        assert sleep.calls == []


# ---------------------------------------------------------------------------
# 9–10: ABSENT handling — primary vs replicated (Phase 3b F1)
# ---------------------------------------------------------------------------


class TestAbsent:
    def test_primary_ledger_absent_is_genuine_and_executes(self, tmp_path) -> None:
        """ABSENT on a primary/same-node ledger → original Algorithm 1 path."""
        buf = ActionBuffer(tmp_path / "wal.jsonl")
        append_buffer(buf, make_record(key="k", t0=0.0, ttl=60.0))
        executed: list[dict[str, Any]] = []
        ledger = FakeLedger()  # empty → ABSENT

        clock = FakeClock(start=10.0)
        service = ReintegrationService(
            buffer=buf, ledger=ledger,
            replicated_ledger=False,
            enable_staleness_check=True,
            v_pre=lambda rec, state: True,
            tau_star_provider=ToggleTauStarProvider(0.9),
            semantic_similarity=lambda a, b: 0.2,  # non-match → safety path
            replanned_actions=[{"tool": "other", "args": {}}],
            execute_fn=lambda payload: executed.append(payload),
            clock=clock, sleep_fn=FakeSleep(clock),
        )
        result = service.run()
        assert result.actions == [ReintegrationAction.REPLAYED]
        assert len(executed) == 1
        assert ledger.lookup("k")["status"] == "COMMITTED"
        assert buf.read_all()[0]["status"] == "COMMITTED"

    def test_replicated_absent_escalates_without_any_processing(self, tmp_path) -> None:
        """Phase 3b F1: ABSENT on edge-B after epoch guard → ESCALATE.

        TTL, V_pre, semantic comparison, and execution must NOT run.
        """
        buf = ActionBuffer(tmp_path / "wal.jsonl")
        rec = make_record(key="lost", t0=0.0, ttl=1.0)  # would expire given clock
        append_buffer(buf, rec)
        executed: list[dict[str, Any]] = []
        v_pre_called: list[bool] = []
        semantic_called: list[bool] = []

        clock = FakeClock(start=100.0)  # t0=0, ttl=1 → expired by clock
        ledger = FakeLedger(epoch=10)
        service = ReintegrationService(
            buffer=buf, ledger=ledger,
            replicated_ledger=True,  # default — edge-B
            enable_epoch_guard=True,
            enable_staleness_check=True,
            v_pre=lambda rec, state: v_pre_called.append(True) or True,
            semantic_similarity=lambda a, b: semantic_called.append(True) or 0.99,
            tau_star_provider=ToggleTauStarProvider(0.5),
            replanned_actions=[{"tool": "navigate_to", "args": {"x": 1.0, "y": 2.0}}],
            execute_fn=lambda payload: executed.append(payload),
            clock=clock, sleep_fn=FakeSleep(clock),
        )
        result = service.run()

        assert result.actions == [ReintegrationAction.ESCALATED]
        assert executed == []
        assert v_pre_called == []
        assert semantic_called == []
        assert buf.read_all()[0]["status"] == "PENDING"  # not committed
        assert ledger.lookup("lost")["status"] == "ESCALATED"

    def test_replicated_absent_with_no_parallel_record_and_flags_off(self, tmp_path) -> None:
        """Default replicated_ledger is True — ABSENT always escalates."""
        buf = ActionBuffer(tmp_path / "wal.jsonl")
        append_buffer(buf, make_record(key="k"))
        ledger = FakeLedger()
        service = ReintegrationService(
            buffer=buf, ledger=ledger,
            tau_star_provider=ToggleTauStarProvider(0.5),
            execute_fn=lambda payload: None,
            clock=FakeClock(), sleep_fn=FakeSleep(FakeClock()),
        )
        result = service.run()
        assert result.actions == [ReintegrationAction.ESCALATED]


# ---------------------------------------------------------------------------
# 11–14: tau* availability — no substitution, no hardcode
# ---------------------------------------------------------------------------


class TestTauStar:
    def _service(self, tau_provider, tmp_path, led_status="ABSENT", **kw):
        buf = ActionBuffer(tmp_path / "wal.jsonl")
        append_buffer(buf, make_record(key="k"))
        ledger = FakeLedger({"k": {"status": led_status}})
        executed: list[dict[str, Any]] = []
        service = ReintegrationService(
            buffer=buf, ledger=ledger,
            replicated_ledger=False,
            tau_star_provider=tau_provider,
            semantic_similarity=lambda a, b: 0.999,
            replanned_actions=[{"tool": "navigate_to", "args": {}}],
            execute_fn=lambda payload: executed.append(payload),
            clock=FakeClock(), sleep_fn=FakeSleep(FakeClock()),
            **kw,
        )
        return service, buf, executed

    def test_tau_missing_escalates(self, tmp_path) -> None:
        service, buf, executed = self._service(ToggleTauStarProvider(None), tmp_path)
        result = service.run()
        assert result.actions == [ReintegrationAction.ESCALATED]
        assert executed == []
        assert service._ledger.lookup("k")["status"] == "ESCALATED"
        assert buf.read_all()[0]["status"] == "PENDING"

    def test_tau_nan_escalates(self, tmp_path) -> None:
        service, buf, executed = self._service(ToggleTauStarProvider(float("nan")), tmp_path)
        result = service.run()
        assert result.actions == [ReintegrationAction.ESCALATED]
        assert executed == []

    def test_tau_unestimable_escalates(self, tmp_path) -> None:
        service, buf, executed = self._service(_RaisingProvider(), tmp_path)
        result = service.run()
        assert result.actions == [ReintegrationAction.ESCALATED]
        assert executed == []

    def test_no_hardcoded_tau_star(self) -> None:
        """The service must never fall back to a magic threshold (0.5052)."""
        import inspect
        from reintegration import reintegration_service as mod
        source = inspect.getsource(mod)
        assert "0.5052" not in source
        assert ToggleTauStarProvider(None)() is None

    def test_toggled_tau_valid(self, tmp_path) -> None:
        service, buf, executed = self._service(ToggleTauStarProvider(0.5), tmp_path)
        result = service.run()
        # sim 0.999 >= 0.5 → MERGED (no duplicate execution)
        assert result.actions == [ReintegrationAction.MERGED]
        assert executed == []
        assert buf.read_all()[0]["status"] == "COMMITTED"


class _RaisingProvider:
    def __call__(self) -> None:
        raise RuntimeError("tau* unestimable")


# ---------------------------------------------------------------------------
# 15–16: TTL / V_pre (STEP 3)
# ---------------------------------------------------------------------------


class TestSafetyChecks:
    def test_ttl_expired_invalidates(self, tmp_path) -> None:
        buf = ActionBuffer(tmp_path / "wal.jsonl")
        append_buffer(buf, make_record(key="k", t0=0.0, ttl=10.0))
        ledger = FakeLedger(records={}, epoch=1)

        clock = FakeClock(start=50.0)
        service = ReintegrationService(
            buffer=buf, ledger=ledger,
            replicated_ledger=False,
            enable_staleness_check=True,
            v_pre=lambda rec, state: True,
            clock=clock, sleep_fn=FakeSleep(clock),
        )
        result = service.run()
        assert result.actions == [ReintegrationAction.INVALIDATED]
        assert ledger.lookup("k")["status"] == "INVALIDATED"
        assert buf.read_all()[0]["status"] == "INVALIDATED"

    def test_vpre_false_invalidates(self, tmp_path) -> None:
        buf = ActionBuffer(tmp_path / "wal.jsonl")
        append_buffer(buf, make_record(key="k", t0=0.0, ttl=60.0))
        ledger = FakeLedger(records={}, epoch=1)
        service = ReintegrationService(
            buffer=buf, ledger=ledger,
            replicated_ledger=False,
            enable_staleness_check=True,
            v_pre=lambda rec, state: False,
            clock=FakeClock(), sleep_fn=FakeSleep(FakeClock()),
        )
        result = service.run()
        assert result.actions == [ReintegrationAction.INVALIDATED]
        assert ledger.lookup("k")["status"] == "INVALIDATED"
        assert buf.read_all()[0]["status"] == "INVALIDATED"

    def test_staleness_check_disabled_skips_vpre(self, tmp_path) -> None:
        buf = ActionBuffer(tmp_path / "wal.jsonl")
        append_buffer(buf, make_record(key="k", t0=0.0, ttl=1.0))
        ledger = FakeLedger(records={}, epoch=1)
        clock = FakeClock(start=500.0)  # long after t0
        service = ReintegrationService(
            buffer=buf, ledger=ledger,
            replicated_ledger=False,
            enable_staleness_check=False,  # ablation rung
            tau_star_provider=ToggleTauStarProvider(0.9),
            semantic_similarity=lambda a, b: 0.0,
            replanned_actions=[{}],
            execute_fn=lambda payload: None,
            clock=clock, sleep_fn=FakeSleep(clock),
        )
        result = service.run()
        # TTL ignored when staleness check off → proceeds to execute
        assert result.actions == [ReintegrationAction.REPLAYED]

    def test_vpre_raises_fails_closed(self, tmp_path) -> None:
        buf = ActionBuffer(tmp_path / "wal.jsonl")
        append_buffer(buf, make_record(key="k", t0=0.0, ttl=60.0))
        ledger = FakeLedger(records={}, epoch=1)

        def boom(rec, state) -> bool:
            raise RuntimeError("V_pre unavailable")

        service = ReintegrationService(
            buffer=buf, ledger=ledger,
            replicated_ledger=False,
            enable_staleness_check=True,
            v_pre=boom,
            clock=FakeClock(), sleep_fn=FakeSleep(FakeClock()),
        )
        result = service.run()
        assert result.actions == [ReintegrationAction.INVALIDATED]


# ---------------------------------------------------------------------------
# 17–18: semantic equivalence (STEP 4)
# ---------------------------------------------------------------------------


class TestSemantic:
    def _mk(self, tmp_path, sim, tau, **kw):
        buf = ActionBuffer(tmp_path / "wal.jsonl")
        append_buffer(buf, make_record(key="k", t0=0.0, ttl=60.0))
        ledger = FakeLedger(records={}, epoch=1)
        executed: list[dict[str, Any]] = []
        service = ReintegrationService(
            buffer=buf, ledger=ledger,
            replicated_ledger=False,
            enable_staleness_check=True,
            v_pre=lambda rec, state: True,
            semantic_similarity=lambda a, b: sim,
            tau_star_provider=ToggleTauStarProvider(tau),
            replanned_actions=[{"tool": "navigate_to", "args": {}}],
            execute_fn=lambda payload: executed.append(payload),
            clock=FakeClock(), sleep_fn=FakeSleep(FakeClock()),
            **kw,
        )
        return service, buf, executed

    def test_semantic_match_merges(self, tmp_path) -> None:
        service, buf, executed = self._mk(tmp_path, sim=0.96, tau=0.9)
        result = service.run()
        assert result.actions == [ReintegrationAction.MERGED]
        assert executed == []
        assert buf.read_all()[0]["status"] == "COMMITTED"

    def test_semantic_mismatch_executes_safety_path(self, tmp_path) -> None:
        service, buf, executed = self._mk(tmp_path, sim=0.40, tau=0.9)
        result = service.run()
        assert result.actions == [ReintegrationAction.REPLAYED]
        assert len(executed) == 1
        assert executed[0]["key"] == "k"
        assert ledger_status_is(service, "k", "COMMITTED")

    def test_no_replanned_actions_means_no_equivalence(self, tmp_path) -> None:
        buf = ActionBuffer(tmp_path / "wal.jsonl")
        append_buffer(buf, make_record(key="k", t0=0.0, ttl=60.0))
        ledger = FakeLedger(records={}, epoch=1)
        executed: list[dict[str, Any]] = []
        service = ReintegrationService(
            buffer=buf, ledger=ledger,
            replicated_ledger=False,
            enable_staleness_check=True,
            v_pre=lambda rec, state: True,
            tau_star_provider=ToggleTauStarProvider(0.3),
            replanned_actions=[],  # none planned
            execute_fn=lambda payload: executed.append(payload),
            clock=FakeClock(), sleep_fn=FakeSleep(FakeClock()),
        )
        result = service.run()
        assert result.actions == [ReintegrationAction.REPLAYED]
        assert len(executed) == 1


def ledger_status_is(service: ReintegrationService, key: str, status: str) -> bool:
    return service._ledger.lookup(key)["status"] == status


# ---------------------------------------------------------------------------
# 19–21: intent keys, ordering, multi-action
# ---------------------------------------------------------------------------


class TestExecutionKeyOrdering:
    def test_intent_key_preferred_when_enabled(self, tmp_path) -> None:
        from pilot.key_schemes import k_intent
        buf = ActionBuffer(tmp_path / "wal.jsonl")
        append_buffer(buf, make_record(key="payload-key", intent_id="INT-7"), add_intent_fields=True)
        ledger = FakeLedger()
        executed: list[dict[str, Any]] = []
        service = ReintegrationService(
            buffer=buf, ledger=ledger,
            replicated_ledger=False,
            enable_staleness_check=True,
            enable_intent_keys=True,
            v_pre=lambda rec, state: True,
            tau_star_provider=ToggleTauStarProvider(0.9),
            semantic_similarity=lambda a, b: 0.1,
            replanned_actions=[{}],
            execute_fn=lambda payload: executed.append(payload),
            clock=FakeClock(), sleep_fn=FakeSleep(FakeClock()),
        )
        result = service.run()
        assert result.actions == [ReintegrationAction.REPLAYED]
        assert executed[0]["key"] == k_intent("SESS-1", 1, "INT-7")
        assert executed[0]["key"] != "payload-key"

    def test_same_node_key_used_when_intent_keys_disabled(self, tmp_path) -> None:
        buf = ActionBuffer(tmp_path / "wal.jsonl")
        append_buffer(buf, make_record(key="payload-key", intent_id="INT-7"), add_intent_fields=True)
        ledger = FakeLedger()
        executed: list[dict[str, Any]] = []
        service = ReintegrationService(
            buffer=buf, ledger=ledger,
            replicated_ledger=False,
            enable_staleness_check=True,
            enable_intent_keys=False,
            v_pre=lambda rec, state: True,
            tau_star_provider=ToggleTauStarProvider(0.9),
            semantic_similarity=lambda a, b: 0.1,
            replanned_actions=[{}],
            execute_fn=lambda payload: executed.append(payload),
            clock=FakeClock(), sleep_fn=FakeSleep(FakeClock()),
        )
        service.run()
        assert executed[0]["key"] == "payload-key"

    def test_original_append_order_preserved(self, tmp_path) -> None:
        buf = ActionBuffer(tmp_path / "wal.jsonl")
        append_buffer(buf, make_record(key="k1", intent_id="A"))
        append_buffer(buf, make_record(key="k2", intent_id="B", tool="log_inspection"))
        append_buffer(buf, make_record(key="k3", intent_id="C", tool="request_supply_drop"))
        ledger = FakeLedger({
            "k1": {"status": "COMMITTED"},
            "k2": {"status": "UNKNOWN"},
        })
        executed: list[str] = []
        service = ReintegrationService(
            buffer=buf, ledger=ledger,
            v_post=lambda rec: True,
            replicated_ledger=False,
            enable_staleness_check=True,
            v_pre=lambda rec, state: True,
            tau_star_provider=ToggleTauStarProvider(0.9),
            semantic_similarity=lambda a, b: 0.1,
            replanned_actions=[{}],
            execute_fn=lambda payload: executed.append(payload["key"]),
            clock=FakeClock(), sleep_fn=FakeSleep(FakeClock()),
        )
        result = service.run()
        assert [a.value for a in result.actions] == ["COMMITTED", "COMMITTED", "REPLAYED"]
        assert executed == ["k3"]

    def test_multi_action_mixed_outcomes(self, tmp_path) -> None:
        buf = ActionBuffer(tmp_path / "wal.jsonl")
        append_buffer(buf, make_record(key="c", intent_id="C"))
        append_buffer(buf, make_record(key="e", intent_id="E"))
        append_buffer(buf, make_record(key="l", intent_id="L", t0=0.0, ttl=5.0))
        ledger = FakeLedger({"c": {"status": "COMMITTED"}})
        clock = FakeClock(start=1000.0)
        service = ReintegrationService(
            buffer=buf, ledger=ledger,
            replicated_ledger=True,
            enable_staleness_check=True,
            v_pre=lambda rec, state: True,
            tau_star_provider=ToggleTauStarProvider(0.9),
            semantic_similarity=lambda a, b: 0.1,
            replanned_actions=[{}],
            execute_fn=lambda payload: None,
            clock=clock, sleep_fn=FakeSleep(clock),
        )
        result = service.run()
        # c committed; e ABSENT on replicated ledger → escalate; l ABSENT → escalate
        assert [a.value for a in result.actions] == [
            "COMMITTED", "ESCALATED", "ESCALATED",
        ]


# ---------------------------------------------------------------------------
# 22–24: CLI / provider, fail-closed behaviour
# ---------------------------------------------------------------------------


class TestProvidersAndCli:
    def test_file_tau_provider_valid_and_invalid(self, tmp_path) -> None:
        import json
        path = tmp_path / "tau.json"
        path.write_text(json.dumps({"tau_star": 0.83}), encoding="utf-8")
        assert FileTauStarProvider(str(path))() == 0.83

        path.write_text(json.dumps({"nope": 1}), encoding="utf-8")
        assert FileTauStarProvider(str(path))() is None

        path.write_text(json.dumps({"tool_class_x": 0.7}), encoding="utf-8")
        assert FileTauStarProvider(str(path), "tool_class_x")() == 0.7
        assert FileTauStarProvider(str(path), "other_class")() is None

        path.write_text(json.dumps({"tau_star": "nan"}), encoding="utf-8")
        assert FileTauStarProvider(str(path))() is None

    def test_cli_no_tau_passed_is_missing(self) -> None:
        from reintegration import reintegration_service as mod
        args = mod.parse_args(["--buffer", "b", "--ledger-db", ":memory:"])
        if args.tau_star is not None:
            prov = mod.ToggleTauStarProvider(args.tau_star)
        elif args.tau_star_file:
            prov = mod.FileTauStarProvider(args.tau_star_file)
        else:
            prov = mod.ToggleTauStarProvider(None)
        assert prov() is None

    def test_execute_raises_escalates_via_unknown(self, tmp_path) -> None:
        buf = ActionBuffer(tmp_path / "wal.jsonl")
        append_buffer(buf, make_record(key="k", t0=0.0, ttl=60.0))
        ledger = FakeLedger(records={}, epoch=1)

        def execute_fn(payload):
            raise RuntimeError("edge-B unreachable")

        service = ReintegrationService(
            buffer=buf, ledger=ledger,
            replicated_ledger=False,
            enable_staleness_check=True,
            v_pre=lambda rec, state: True,
            tau_star_provider=ToggleTauStarProvider(0.9),
            semantic_similarity=lambda a, b: 0.1,
            replanned_actions=[{}],
            execute_fn=execute_fn,
            clock=FakeClock(), sleep_fn=FakeSleep(FakeClock()),
        )
        result = service.run()
        assert result.actions == [ReintegrationAction.ESCALATED]
        assert ledger.lookup("k")["status"] == "ESCALATED"
        assert buf.read_all()[0]["status"] == "PENDING"


# ---------------------------------------------------------------------------
# Frozen API compatibility — real ActionBuffer + real EffectLedger
# ---------------------------------------------------------------------------


class TestFrozenApiCompatibility:
    def test_full_run_with_real_frozen_classes(self, tmp_path) -> None:
        """Bound the service to the real frozen ActionBuffer/EffectLedger."""
        buf = ActionBuffer(tmp_path / "wal.jsonl")
        append_buffer(buf, make_record(key="already", intent_id="A"))
        append_buffer(buf, make_record(key="pending1", intent_id="B", epoch=1))

        db = tmp_path / "effects.db"
        ledger = EffectLedger(db_path=str(db))
        ledger.create("already", "navigate_to", "h1", ttl=60.0, intent_id="A")
        ledger.mark_committed("already", epoch=1)
        ledger.increment_epoch()  # global epoch 0 -> 1 == buffer.max_epoch(1)

        executed: list[dict[str, Any]] = []
        service = ReintegrationService(
            buffer=buf, ledger=ledger,
            replicated_ledger=True,
            enable_epoch_guard=True,
            sync_window_sec=0.0,
            v_pre=lambda rec, state: True,
            tau_star_provider=ToggleTauStarProvider(0.9),
            semantic_similarity=lambda a, b: 0.1,
            replanned_actions=[{}],
            execute_fn=lambda payload: executed.append(payload),
            clock=time.time, sleep_fn=lambda s: None,
        )
        result = service.run()

        assert result.epoch_guard_ok is True
        assert result.actions == [ReintegrationAction.COMMITTED, ReintegrationAction.ESCALATED]
        assert executed == []
        assert ledger.lookup("pending1")["status"] == "ESCALATED"
        assert buf.read_all()[0]["status"] == "COMMITTED"
        assert buf.read_all()[1]["status"] == "PENDING"
        ledger.close()

    def test_effect_ledger_status_enum_compat(self, tmp_path) -> None:
        """Service comparisons work against EffectStatus str-enum values."""
        from tool_world.models import EffectStatus
        db = tmp_path / "effects.db"
        ledger = EffectLedger(db_path=str(db))
        ledger.create("k", "navigate_to", "h", ttl=60.0)
        rec = ledger.lookup("k")
        assert rec is not None and rec["status"] == EffectStatus.ABSENT
        # plain-string comparison used in the service matches the enum
        assert rec["status"] == "ABSENT"
        ledger.close()
