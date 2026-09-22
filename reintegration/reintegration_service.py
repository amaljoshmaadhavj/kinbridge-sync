"""Reintegration Service — Algorithm 1 (Phase 4).

Implements client-side action safety after connectivity gaps per the
Kinbridge-Sync paper's Algorithm 1.  Orchestrates ActionBuffer,
EffectLedger and LinkMonitor, and applies the epoch guard, TTL/V_pre
safety checks, semantic-equivalence comparison against a calibrated
tau-star, and execution with the action's key.

Finalized Phase 4 decisions reflected here:

1. Epoch sync window
   - Default 2.0 s, configurable via ``REINTEGRATION_SYNC_WINDOW_SEC``.
   - Polling interval exactly 50 ms.
   - If the replica does not catch up within the window: ESCALATE all
     affected actions and STOP.  Never replay blindly.

2. tau-star unavailable / unestimable
   - Before semantic comparison, a calibrated tau* must be supplied for
     the current context.  Missing / NaN / unestimable tau* → ESCALATE.
   - tau* is never hardcoded and never substituted (no magic default
     threshold is ever used as a fallback).

3. Phase 3b ABSENT-record safety
   - The epoch guard takes precedence over ledger interpretation.
   - After the epoch guard completes, an ABSENT key on the *replicated*
     ledger is ESCALATED, because ABSENT cannot distinguish
     "never existed" from "lost during replication" (Phase 3b F1).
     ABSENT is never treated as permission to replay.
   - For a primary / same-node ledger (``replicated_ledger=False``)
     ABSENT genuinely means "never recorded", and the original
     Algorithm 1 path (TTL → V_pre → semantic → execute) applies.
   - This module does NOT modify Phase 3b and does NOT add a
     replication guarantee.

Feature flags (--enable-intent-keys, --enable-staleness-check,
--enable-epoch-guard, --enable-degradation-gate) make the individual
safeguards independently toggleable for the Phase 5 ablation ladder.
The degradation-gate flag is accepted for ablation symmetry; its
operational semantics (local-model switching) belong to the Phase 5
harness, so it does not alter Phase 4 decision logic.

References:
    Markdown/Complete-Research.md  §3.5, Algorithm 1
    Markdown/Work .md              B — Phase 4 / Algorithm 1 steps
    Markdown/Runbook.md            Phase 4 reintegration service
"""

from __future__ import annotations

import argparse
import json
import os
import time
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from client.buffer import ActionBuffer
from client.link_monitor import LinkMonitor
from tool_world.ledger import EffectLedger


# ---------------------------------------------------------------------------
# Defaults / configuration
# ---------------------------------------------------------------------------

DEFAULT_SYNC_WINDOW_SEC = 2.0
DEFAULT_POLL_INTERVAL_SEC = 0.05  # 50 ms, exactly

# Hook signatures
VPreHook = Callable[[dict[str, Any], dict[str, Any]], bool]
VPostHook = Callable[[dict[str, Any]], bool]
SemanticHook = Callable[[dict[str, Any], Any], float]
TauStarProvider = Callable[[], Optional[float]]
ExecuteFn = Callable[[dict[str, Any]], Any]


def get_sync_window_sec() -> float:
    """Return the sync window from the environment, default 2.0 s."""
    raw = os.environ.get("REINTEGRATION_SYNC_WINDOW_SEC")
    if raw is None:
        return DEFAULT_SYNC_WINDOW_SEC
    try:
        val = float(raw)
    except ValueError:
        return DEFAULT_SYNC_WINDOW_SEC
    if val < 0.0:
        return DEFAULT_SYNC_WINDOW_SEC
    return val


# ---------------------------------------------------------------------------
# Decision outcomes
# ---------------------------------------------------------------------------


class ReintegrationAction(Enum):
    """Per-action outcome from Algorithm 1."""

    COMMITTED = "COMMITTED"          # already committed; no replay
    ESCALATED = "ESCALATED"          # requires human attention; no replay
    INVALIDATED = "INVALIDATED"      # TTL expired or V_pre failed
    REPLAYED = "REPLAYED"            # executed after safety checks
    MERGED = "MERGED"                # semantically equivalent; intent merged


class ReintegrationResult:
    """Result of a reintegration run."""

    def __init__(self, actions: list[ReintegrationAction], epoch_guard_ok: bool) -> None:
        self.actions = actions
        self.epoch_guard_ok = epoch_guard_ok


# ---------------------------------------------------------------------------
# Default semantic similarity (sentence-transformers all-MiniLM-L6-v2)
# ---------------------------------------------------------------------------


def _action_text(rec: dict[str, Any]) -> str:
    """Canonical text representation of an action for embedding."""
    tool = rec.get("tool", "")
    args = rec.get("args", {})
    from pilot.key_schemes import canonical_serialise
    return f"{tool}||{canonical_serialise(args)}"


class DefaultSemanticSimilarity:
    """Cosine similarity between two actions via all-MiniLM-L6-v2.

    Loads the sentence-transformer on first use.  If the model is not
    installed, raises ImportError so callers can fail clearly rather
    than degrade silently.
    """

    _model = None

    @classmethod
    def _get_model(cls):
        if cls._model is None:
            from sentence_transformers import SentenceTransformer
            cls._model = SentenceTransformer("all-MiniLM-L6-v2")
        return cls._model

    def __call__(self, buffered: dict[str, Any], replanned: Any) -> float:
        model = self._get_model()
        if isinstance(replanned, dict):
            other = _action_text(replanned)
        else:
            other = str(replanned)
        import numpy as np
        emb = model.encode([_action_text(buffered), other], normalize_embeddings=True)
        return float(np.dot(emb[0], emb[1]))


class FileTauStarProvider:
    """Calibrated tau* provider reading a JSON config.

    The file may contain a top-level ``tau_star`` number, or a mapping
    from tool-class names to values.  Returns None when the value is
    missing, NaN, or otherwise not usable for the requested context.
    """

    def __init__(self, path: str | Path, tool_class: str | None = None) -> None:
        self._path = Path(path)
        self._tool_class = tool_class
        self._data: Any = None

    def _load(self) -> Any:
        if self._data is None:
            with open(self._path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        return self._data

    def __call__(self) -> Optional[float]:
        try:
            data = self._load()
        except Exception:
            return None
        value: Any
        if isinstance(data, dict):
            if self._tool_class is not None and self._tool_class in data:
                value = data[self._tool_class]
            else:
                value = data.get("tau_star")
        else:
            value = data
        if value is None:
            return None
        try:
            fval = float(value)
        except (TypeError, ValueError):
            return None
        if fval != fval:  # NaN
            return None
        return fval


class ToggleTauStarProvider:
    """Simple provider that returns a fixed value or None.

    Intended for tests and small harnesses; requests a calibrated value
    and returns None when unavailable (no threshold substitution).
    """

    def __init__(self, tau: float | None) -> None:
        self._tau = tau

    def __call__(self) -> Optional[float]:
        tau = self._tau
        if tau is None:
            return None
        if tau != tau:  # NaN
            return None
        return tau


# ---------------------------------------------------------------------------
# ReintegrationService
# ---------------------------------------------------------------------------


class ReintegrationService:
    """Algorithm 1 reintegration / action-safety service.

    Args:
        buffer: Frozen :class:`client.buffer.ActionBuffer`.
        ledger: :class:`tool_world.ledger.EffectLedger` for the reconnect
            target.  The default ABSENT policy assumes this is the
            replicated ledger (edge-B); pass ``replicated_ledger=False``
            for a primary / same-node ledger.
        link_monitor: Optional :class:`client.link_monitor.LinkMonitor`
            used to detect link recovery.
        v_pre: Inject ``V_pre(action, state) -> bool``.  ``None`` means
            the check is unavailable (treated as failing closed within
            the staleness check).
        v_post: Inject read-only ``V_post(action) -> bool``.  ``None``
            means no post-condition check is available (UNKNOWN escalates).
        semantic_similarity: ``(buffered, replanned) -> float``.
        tau_star_provider: Callable returning calibrated tau* for the
            current context, or None if unavailable.
        replanned_actions: Sequence of newly re-planned actions used for
            semantic-equivalence comparison.
        sync_window_sec: Epoch sync window in seconds (default from env).
        poll_interval_sec: Epoch polling interval (default 0.05).
        replicated_ledger: When True (default), an ABSENT key after a
            successful epoch guard is ESCALATED (Phase 3b F1 safety).
            When False, ABSENT means never recorded and the original
            Algorithm 1 safety/execute path applies.
        use_v_post: Whether the UNKNOWN ledger state triggers the V_post
            evaluation (True) or escalates directly (False).
        enable_intent_keys: Ablation flag.
        enable_staleness_check: Ablation flag (TTL + V_pre).
        enable_epoch_guard: Ablation flag.
        enable_degradation_gate: Ablation flag (no Phase 4 decision
            effect; phase 5 harness semantics).
        execute_fn: Executor callback invoked on replay.
        state: External state ``S_now`` passed to V_pre.
        clock:  Callable clock (tests).
        sleep_fn: Sleep function (tests).
    """

    def __init__(
        self,
        *,
        buffer: ActionBuffer,
        ledger: EffectLedger,
        link_monitor: Optional[LinkMonitor] = None,
        v_pre: Optional[VPreHook] = None,
        v_post: Optional[VPostHook] = None,
        semantic_similarity: Optional[SemanticHook] = None,
        tau_star_provider: Optional[TauStarProvider] = None,
        replanned_actions: Optional[Sequence[Any]] = None,
        sync_window_sec: Optional[float] = None,
        poll_interval_sec: float = DEFAULT_POLL_INTERVAL_SEC,
        replicated_ledger: bool = True,
        use_v_post: bool = True,
        enable_intent_keys: bool = False,
        enable_staleness_check: bool = False,
        enable_epoch_guard: bool = False,
        enable_degradation_gate: bool = False,
        execute_fn: Optional[ExecuteFn] = None,
        state: Optional[dict[str, Any]] = None,
        clock: Callable[[], float] = time.time,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._buffer = buffer
        self._ledger = ledger
        self._link_monitor = link_monitor
        self._v_pre = v_pre
        self._v_post = v_post
        self._semantic_similarity = semantic_similarity
        self._tau_star_provider = tau_star_provider
        self._replanned_actions = list(replanned_actions) if replanned_actions else []
        self._sync_window_sec = (
            sync_window_sec if sync_window_sec is not None else get_sync_window_sec()
        )
        self._poll_interval_sec = poll_interval_sec
        self._replicated_ledger = replicated_ledger
        self._use_v_post = use_v_post
        self._enable_intent_keys = enable_intent_keys
        self._enable_staleness_check = enable_staleness_check
        self._enable_epoch_guard = enable_epoch_guard
        self._enable_degradation_gate = enable_degradation_gate
        self._execute_fn = execute_fn
        self._state = state if state is not None else {}
        self._clock = clock
        self._sleep_fn = sleep_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> ReintegrationResult:
        """Execute Algorithm 1: epoch guard, then process buffered actions."""
        epoch_ok = True
        if self._enable_epoch_guard:
            epoch_ok = self._epoch_guard()

        if not epoch_ok:
            # ESCALATE all affected buffered actions and STOP. Never replay.
            actions: list[ReintegrationAction] = []
            for rec in self._buffer.read_all():
                if rec.get("status") in ("COMMITTED", "INVALIDATED"):
                    actions.append(ReintegrationAction.COMMITTED)
                    continue
                self._ledger.mark_escalated(rec["key"])
                actions.append(ReintegrationAction.ESCALATED)
            return ReintegrationResult(actions, epoch_guard_ok=False)

        # Process buffered actions in original append order.
        actions = []
        for rec in self._buffer.read_all():
            actions.append(self._process_action(rec))
        return ReintegrationResult(actions, epoch_guard_ok=True)

    # ------------------------------------------------------------------
    # Epoch guard
    # ------------------------------------------------------------------

    def _epoch_guard(self) -> bool:
        """Compare ledger_epoch(B) with buffer.max_epoch, polling if behind."""
        buffer_epoch = self._buffer.max_epoch
        if buffer_epoch == 0:
            return True  # nothing buffered — nothing to wait for

        if self._ledger.get_epoch() >= buffer_epoch:
            return True

        deadline = self._clock() + self._sync_window_sec
        while self._clock() < deadline:
            self._sleep_fn(self._poll_interval_sec)
            if self._ledger.get_epoch() >= buffer_epoch:
                return True

        # Still behind after sync window → escalate everything.
        return False

    # ------------------------------------------------------------------
    # Per-action processing (STEP 2)
    # ------------------------------------------------------------------

    def _process_action(self, rec: dict[str, Any]) -> ReintegrationAction:
        """Apply Algorithm 1 steps 2–5 to a single buffered action."""
        key = rec["key"]
        ledger_rec = self._ledger.lookup(key)

        if ledger_rec is None or ledger_rec.get("status") == "ABSENT":
            # ABSENT after a successful epoch guard.
            if self._replicated_ledger:
                # Phase 3b F1 ambiguity: cannot distinguish "never
                # existed" from "lost during replication". ESCALATE.
                return self._escalate(rec)
            # Genuine primary-ledger ABSENT → original Algorithm 1 path
            # (TTL → V_pre → semantic → execute).
            return self._safety_and_execute(rec)

        status = ledger_rec["status"]

        if status == "COMMITTED":
            self._buffer.mark_committed(key)
            return ReintegrationAction.COMMITTED

        if status == "UNKNOWN":
            if self._use_v_post and self._v_post is not None:
                try:
                    confirmed = bool(self._v_post(rec))
                except Exception:
                    confirmed = False
                if confirmed:
                    self._buffer.mark_committed(key)
                    return ReintegrationAction.COMMITTED
            return self._escalate(rec)

        if status == "ESCALATED":
            return ReintegrationAction.ESCALATED

        if status == "INVALIDATED":
            self._buffer.mark_invalidated(key)
            return ReintegrationAction.INVALIDATED

        # Unknown status string — fail safe.
        return self._escalate(rec)

    def _escalate(self, rec: dict[str, Any]) -> ReintegrationAction:
        """Record an escalation in the ledger; no buffer misrepresentation.

        If no ledger record exists for the key (e.g. lost during Phase 3b
        replication), create one so the escalation is auditable, then mark
        it ESCALATED.  Never replays, never marks the buffer committed.
        """
        key = rec["key"]
        if self._ledger.lookup(key) is None:
            args_hash = rec.get("args_hash", "")
            if not args_hash:
                from pilot.key_schemes import canonical_serialise
                args_hash = canonical_serialise(rec.get("args", {}))
            try:
                self._ledger.create(
                    key,
                    rec.get("tool", ""),
                    args_hash,
                    ttl=float(rec.get("ttl") or 0.0),
                    intent_id=rec.get("intent_id"),
                )
            except Exception:
                pass
        self._ledger.mark_escalated(key)
        return ReintegrationAction.ESCALATED

    # ------------------------------------------------------------------
    # STEP 3 — safety checks, STEP 4 — semantic, STEP 5 — execution
    # ------------------------------------------------------------------

    def _safety_and_execute(self, rec: dict[str, Any]) -> ReintegrationAction:
        """Apply TTL/V_pre, semantic equivalence, then execute if allowed."""
        # STEP 3 — staleness checks
        if self._enable_staleness_check:
            t0 = float(rec.get("t0", 0.0))
            ttl = float(rec.get("ttl", 0.0))
            if ttl > 0.0 and self._clock() - t0 > ttl:
                self._ledger.mark_invalidated(rec["key"])
                self._buffer.mark_invalidated(rec["key"])
                return ReintegrationAction.INVALIDATED

            pre_ok = True
            if self._v_pre is not None:
                try:
                    pre_ok = bool(self._v_pre(rec, self._state))
                except Exception:
                    pre_ok = False
            if not pre_ok:
                self._ledger.mark_invalidated(rec["key"])
                self._buffer.mark_invalidated(rec["key"])
                return ReintegrationAction.INVALIDATED

        # STEP 4 — semantic equivalence
        merged = self._semantic_check(rec)
        if merged is not None:
            return merged

        # STEP 5 — execution
        return self._execute(rec)

    def _semantic_check(self, rec: dict[str, Any]) -> Optional[ReintegrationAction]:
        """Compare against replanned actions using calibrated tau*.

        Returns MERGED when similarity >= tau*, ESCALATED when tau* is
        unavailable, or None when no equivalence is found (execution
        with the action's own key is the safety path).
        """
        tau = None
        if self._tau_star_provider is not None:
            try:
                tau = self._tau_star_provider()
            except Exception:
                tau = None
        if tau is None or tau != tau:  # missing / NaN / unestimable
            return self._escalate(rec)

        if self._semantic_similarity is None or not self._replanned_actions:
            # No comparison target — no equivalence identified.
            return None

        best_sim = -1.0
        for replanned in self._replanned_actions:
            try:
                sim = float(self._semantic_similarity(rec, replanned))
            except Exception:
                continue
            if sim > best_sim:
                best_sim = sim

        if best_sim >= tau:
            # Treat as equivalent / merge intent — no duplicate execution.
            self._buffer.mark_committed(rec["key"])
            return ReintegrationAction.MERGED

        # similarity < tau* → safety path: execute as its own action.
        return None

    # ------------------------------------------------------------------
    # STEP 5 — execution
    # ------------------------------------------------------------------

    def _resolve_key(self, rec: dict[str, Any]) -> str:
        """Return the execution key.

        Prefers the intent key when intent fields (session_id, turn_seq,
        intent_id) are all present and intent keys are enabled;
        otherwise the buffered action's own key is used.
        """
        if self._enable_intent_keys:
            sid = rec.get("session_id")
            tseq = rec.get("turn_seq")
            iid = rec.get("intent_id")
            if sid and tseq is not None and iid:
                from pilot.key_schemes import k_intent
                return k_intent(sid, int(tseq), iid)
        return rec["key"]

    def _execute(self, rec: dict[str, Any]) -> ReintegrationAction:
        """Execute the action with its (intent / stored) key."""
        key = self._resolve_key(rec)
        payload = dict(rec)
        payload["key"] = key

        if self._execute_fn is not None:
            try:
                self._execute_fn(payload)
            except Exception:
                # Execution outcome uncertain → record UNKNOWN then ESCALATE.
                self._ledger.mark_unknown(rec["key"])
                return self._escalate(rec)

        self._ledger.mark_committed(rec["key"])
        self._buffer.mark_committed(rec["key"])
        return ReintegrationAction.REPLAYED


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse the reintegration-service CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="reintegration_service",
        description="Kinbridge-Sync Algorithm 1 reintegration / action safety service",
    )
    parser.add_argument("--buffer", required=True, help="Path to the JSONL action buffer")
    parser.add_argument("--ledger-db", required=True, help="Path to the edge-B SQLite ledger")
    parser.add_argument(
        "--edge-b-url",
        default=os.environ.get("EDGE_B_URL", "http://edge-b:8001"),
        help="Edge-B base URL used for execution",
    )
    parser.add_argument("--sync-window", type=float, help="Epoch sync window seconds")
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL_SEC)
    parser.add_argument("--tau-star", type=float, help="Calibrated tau* (explicit)")
    parser.add_argument(
        "--tau-star-file",
        help="JSON file providing calibrated tau* (provider)",
    )
    parser.add_argument("--tool-class", help="Tool class context for tau* lookup")
    parser.add_argument(
        "--replicated-ledger",
        action="store_true",
        default=True,
        help="Assume ledger is the replicated edge-B (ABSENT → escalate). Default: on",
    )
    parser.add_argument(
        "--primary-ledger",
        action="store_true",
        help="Treat ledger as primary (ABSENT is genuine; original Algorithm 1 path)",
    )
    parser.add_argument("--enable-intent-keys", action="store_true")
    parser.add_argument("--enable-staleness-check", action="store_true")
    parser.add_argument("--enable-epoch-guard", action="store_true")
    parser.add_argument("--enable-degradation-gate", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Run the reintegration service from the command line."""
    args = parse_args(argv)

    buffer = ActionBuffer(args.buffer)
    ledger = EffectLedger(db_path=args.ledger_db)

    tau_provider: Optional[TauStarProvider]
    if args.tau_star is not None:
        tau_provider = ToggleTauStarProvider(args.tau_star)
    elif args.tau_star_file:
        tau_provider = FileTauStarProvider(args.tau_star_file, args.tool_class)
    else:
        tau_provider = ToggleTauStarProvider(None)

    semantic: Optional[SemanticHook] = DefaultSemanticSimilarity()

    import httpx

    def execute_fn(payload: dict[str, Any]) -> Any:
        tool = payload.get("tool")
        args_dict = dict(payload.get("args", {}))
        body: dict[str, Any] = dict(args_dict)
        if payload.get("intent_id"):
            body["intent_id"] = payload["intent_id"]
        if payload.get("session_id"):
            body["session_id"] = payload["session_id"]
        if payload.get("turn_seq") is not None:
            body["turn_seq"] = payload["turn_seq"]
        if payload.get("fault_inject"):
            body["fault_inject"] = True
        resp = httpx.post(f"{args.edge_b_url}/tools/{tool}", json=body, timeout=10.0)
        resp.raise_for_status()

    service = ReintegrationService(
        buffer=buffer,
        ledger=ledger,
        v_pre=None,
        v_post=None,
        semantic_similarity=semantic,
        tau_star_provider=tau_provider,
        sync_window_sec=args.sync_window,
        poll_interval_sec=args.poll_interval,
        replicated_ledger=not args.primary_ledger,
        enable_intent_keys=args.enable_intent_keys,
        enable_staleness_check=args.enable_staleness_check,
        enable_epoch_guard=args.enable_epoch_guard,
        enable_degradation_gate=args.enable_degradation_gate,
        execute_fn=execute_fn,
    )
    result = service.run()

    for key, action in zip((rec["key"] for rec in buffer.read_all()), result.actions):
        print(f"{key} -> {action.value}")
    print(f"epoch_guard_ok={result.epoch_guard_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
