"""In-process episode harness (Phase 5).

Deterministic simulation of one episode against the REAL frozen
components: ``client.buffer.ActionBuffer``, ``tool_world.ledger.EffectLedger``,
the ``tool_world.main._handle_tool`` idempotency flow, the Phase 3b
replication semantics (watcher -> lagged replicator -> writer), and the
Phase 4 Algorithm 1 ReintegrationService + baselines.

Components
----------
ToolWorldServer
    Mirrors ``tool_world.main._handle_tool``: computes the effect key
    from the client request (intent-preferring), looks up the ledger,
    creates + commits a new record on a miss, and de-duplicates on a
    COMMITTED hit.  The client can be configured to send intent fields
    or not; the key scheme therefore matches the marker the client
    actually uses (B3 ablation rows differ by key_scheme).
ReplicationEngine
    Faithful to Phase 3b: a commit on edge-a publishes a replication
    event; the event arrives at edge-b after the configured lag; the
    edge-b sink applies the frozen ``sqlite_writer`` upsert semantics
    (record row + per-record epoch — the ``ledger_meta`` epoch counter
    is NEVER bumped, exactly like the frozen writer).  Replication is
    one-directional (edge-a -> edge-b), mirroring the frozen
    replication deployment.
Outage / network
    Client->server deliveries are dropped inside the outage window.
    Server<->server replication is NOT interrupted (only the client link
    is blacked out in the testbed).
Ground truth
    The harness records every executed effect by effect_identity and
    computes duplicate / stale per action over the whole episode.  The
    original pre-gap committed effect counts as one application; a
    second application (from a blind replay or a divergent-key replan)
    is a duplicate; an application after the action TTL elapsed
    (t - t0 > ttl) is a stale execution.

All time is simulated: the harness clock is injected into the Phase 4
service via ``clock``/``sleep_fn``.  ``ttr_seconds`` is measured from
the reconnect instant to reintegration completion on the harness clock.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

from client.buffer import ActionBuffer
from experiments.phase5.record import validate_episode_record
from experiments.phase5.scenario import AblationRow, EpisodeTemplate, ScenarioAction
from pilot.key_schemes import canonical_serialise, k_intent, k_payload
from tool_world.ledger import EffectLedger
from tool_world.models import EffectStatus

OUTAGE_PROFILE_NAMES = {0: "no_outage", 10: "short_outage", 90: "baseline", 300: "long_outage"}


def action_key(
    tool: str,
    args: dict[str, Any],
    key_scheme: str,
    *,
    intent_id: Optional[str] = None,
    session_id: Optional[str] = None,
    turn_seq: Optional[int] = None,
) -> str:
    """Client/ledger key for an action under a key scheme (B3).

    ``key_scheme`` is:
      - ``intent``: intent-preferring when intent fields are present
        (mirrors ``tool_world.main._compute_key`` when the client sends
        intent fields), else payload key.
      - ``payload``: payload-only key (k_payload).
      - ``none``: no client idempotency — a one-shot buffer-local key;
        the server issues a fresh ledger key on every call so replays
        can never dedupe (Table VI ``buffer_only``).
    """
    if key_scheme == "none":
        # Stable within one buffer append only; never matches across
        # a cold replan or a blind second send under a fresh buffer key.
        return "buf:" + hashlib.sha256(
            f"{tool}|{canonical_serialise(args)}|{intent_id or ''}".encode("utf-8")
        ).hexdigest()
    if key_scheme == "intent" and intent_id and session_id and turn_seq is not None:
        return k_intent(session_id, int(turn_seq), intent_id)
    return k_payload(tool, args)


def args_hash(args: dict[str, Any]) -> str:
    """SHA-256 hex digest of canonical args (matches tool_world)."""
    return hashlib.sha256(canonical_serialise(args).encode("utf-8")).hexdigest()


class SimClock:
    """Deterministic mutable clock."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, delta: float) -> None:
        if delta < 0:
            raise ValueError("cannot advance the clock backwards")
        self.now += delta


class SimSleep:
    """Sleep that advances a SimClock (no real waiting)."""

    def __init__(self, clock: SimClock) -> None:
        self.clock = clock

    def __call__(self, seconds: float) -> None:
        self.clock.advance(seconds)


class ToolWorldServer:
    """Mirror of ``tool_world.main._handle_tool`` for one ledger."""

    def __init__(self, ledger: EffectLedger) -> None:
        self._ledger = ledger
        self.effects_applied: list[dict[str, Any]] = []

    @property
    def ledger(self) -> EffectLedger:
        return self._ledger

    def execute(
        self,
        tool: str,
        args: dict[str, Any],
        *,
        intent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        turn_seq: Optional[int] = None,
        key_scheme: str = "intent",
        clock: Optional[Callable[[], float]] = None,
        force_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """Execute a tool call with the frozen key/idempotency semantics.

        Mirror of ``tool_world.main._handle_tool``:
            1. compute key (intent-preferring under "intent" scheme),
               unless ``force_key`` pins the ledger key (buffer scheme
               authority for naive/verify replay).
            2. lookup; COMMITTED record -> no re-execution (dedupe)
            3. create + commit a new effect
        """
        if force_key is not None:
            key = force_key
        else:
            key = action_key(tool, args, key_scheme, intent_id=intent_id,
                             session_id=session_id, turn_seq=turn_seq)
        a_hash = args_hash(args)

        # key_scheme "none": never dedupe — every call re-applies the
        # effect (Table VI buffer_only demonstrates missing idempotency).
        if key_scheme == "none":
            existing = self._ledger.lookup(key)
            if existing is None:
                epoch = self._ledger.increment_epoch()
                self._ledger.create(
                    key=key, tool=tool, args_hash=a_hash, ttl=0.0, intent_id=intent_id
                )
                self._ledger.set_epoch(key, epoch)
                self._ledger.mark_committed(key)
            else:
                epoch = existing["epoch"]
                self._ledger.mark_committed(key)
            now = clock() if clock is not None else epoch
            self.effects_applied.append(
                {
                    "key": key,
                    "tool": tool,
                    "args": dict(args),
                    "intent_id": intent_id,
                    "committed_at": now,
                    "epoch": epoch,
                }
            )
            return {
                "key": key,
                "status": EffectStatus.COMMITTED.value,
                "committed": True,
                "epoch": epoch,
                "dedup": False,
            }

        existing = self._ledger.lookup(key)
        if existing is not None and existing["status"] == EffectStatus.COMMITTED.value:
            return {
                "key": key,
                "status": EffectStatus.COMMITTED.value,
                "committed": False,
                "epoch": existing["epoch"],
                "dedup": True,
            }

        if existing is None:
            epoch = self._ledger.increment_epoch()
            self._ledger.create(
                key=key, tool=tool, args_hash=a_hash, ttl=0.0, intent_id=intent_id
            )
            # Mirror main.py: set_epoch then mark_committed(no epoch arg).
            self._ledger.set_epoch(key, epoch)
            self._ledger.mark_committed(key)
        else:
            epoch = existing["epoch"]
            self._ledger.mark_committed(key)

        now = clock() if clock is not None else epoch
        self.effects_applied.append(
            {
                "key": key,
                "tool": tool,
                "args": dict(args),
                "intent_id": intent_id,
                "committed_at": now,
                "epoch": epoch,
            }
        )
        return {"key": key, "status": EffectStatus.COMMITTED.value,
                "committed": True, "epoch": epoch, "dedup": False}


class ReplicationEngine:
    """Phase 3b-faithful edge-a -> edge-b record replication with lag."""

    def __init__(self, ledger_b: EffectLedger, clock: SimClock, lag_ms: int) -> None:
        self._ledger_b = ledger_b
        self._clock = clock
        self._lag_ms = int(lag_ms)
        self._queue: list[tuple[float, dict[str, Any]]] = []

    def publish(self, record: dict[str, Any]) -> None:
        """Publish a committed edge-a record (like the SQLite watcher)."""
        if not record:
            return
        if self._lag_ms == -1:
            # replicator "never" behavior: nothing is forwarded.
            return
        arrival = self._clock.now + (self._lag_ms / 1000.0)
        self._queue.append((arrival, dict(record)))

    def drain(self) -> None:
        """Deliver due events to edge-b (like the sqlite writer upsert)."""
        now = self._clock.now
        due = [e for e in self._queue if e[0] <= now]
        for _, record in due:
            self._upsert_into_b(record)
        self._queue = [e for e in self._queue if e[0] > now]

    def _upsert_into_b(self, record: dict[str, Any]) -> None:
        """Mirror the frozen writer upsert (record row, no meta-epoch bump)."""
        key = record["key"]
        existing = self._ledger_b.lookup(key)
        if existing is None:
            try:
                self._ledger_b.create(
                    key=key,
                    tool=record["tool"],
                    args_hash=record["args_hash"],
                    ttl=float(record.get("ttl") or 0.0),
                    intent_id=record.get("intent_id"),
                )
            except Exception:
                return
        if record.get("status") == EffectStatus.COMMITTED.value:
            self._ledger_b.mark_committed(key, epoch=int(record.get("epoch") or 0))
        elif record.get("status") == EffectStatus.ESCALATED.value:
            self._ledger_b.mark_escalated(key)
        elif record.get("status") == EffectStatus.INVALIDATED.value:
            self._ledger_b.mark_invalidated(key)
        elif record.get("status") == EffectStatus.UNKNOWN.value:
            self._ledger_b.mark_unknown(key)

    @property
    def lag_ms(self) -> int:
        return self._lag_ms

    @property
    def pending_events(self) -> int:
        return len(self._queue)


class GroundTruth:
    """Per-episode ground-truth effect accounting (duplicate / stale)."""

    def __init__(self) -> None:
        # effect_identity -> list of effect application times
        self._applications: dict[str, list[float]] = {}

    def record_application(self, effect_identity: str, at: float) -> None:
        self._applications.setdefault(effect_identity, []).append(at)

    def applications(self, effect_identity: str) -> list[float]:
        return list(self._applications.get(effect_identity, []))


class EpisodeHarness:
    """Runs one episode and produces a validated episode record."""

    def __init__(
        self,
        template: EpisodeTemplate,
        semantic_similarity: Optional[Callable[[dict[str, Any], Any], float]] = None,
        semantic_similarity_label: str = "injected",
    ) -> None:
        self._template = template
        self._semantic_similarity = semantic_similarity
        self._semantic_similarity_label = semantic_similarity_label

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_episode(
        self,
        method: str,
        outage_s: int,
        episode_id: int,
        *,
        topology: str | None = None,
        reconnect_node: str | None = None,
        source_node: str | None = None,
        replication_lag_ms: int | None = None,
        ablation_row: Optional[AblationRow] = None,
        key_scheme: str | None = None,
        calibration_temperature: float | None = None,
        tau_star_path: str | Path | None = None,
        tmp_root: Optional[Path] = None,
    ) -> dict[str, Any]:
        """Run one deterministic episode; return the validated record.

        ``ablation_row`` selects the kinbridge_sync feature flags
        (key_scheme + enable_*).  ``None`` means full Algorithm 1 with
        the scenario's default key scheme.
        """
        src = source_node or self._template.default_source_node
        re = reconnect_node or self._template.default_reconnect_node
        lag = self._template.default_replication_lag_ms if replication_lag_ms is None else int(replication_lag_ms)
        topo = self._template.default_topology if topology is None else topology

        if topo == "cross-node":
            re = "edge-b"
        else:
            re = "edge-a"

        scheme = key_scheme or (ablation_row.key_scheme if ablation_row else "intent")

        clock = SimClock(start=0.0)
        sleep_fn = SimSleep(clock)

        seed = episode_seed_of(self._template, method, int(outage_s), int(episode_id))
        from core.seeds import seed_everything
        # episode_seed is mod 2**63 (documented); numpy requires 32-bit.
        seed_everything(int(seed) % (2 ** 32))

        with tempfile.TemporaryDirectory(prefix="phase5_ep_", dir=str(tmp_root) if tmp_root else None) as td:
            td_path = Path(td)
            buffer = ActionBuffer(td_path / "wal.jsonl")

            ledger_a = EffectLedger(db_path=":memory:")
            server_a = ToolWorldServer(ledger_a)
            ledger_b = EffectLedger(db_path=":memory:")
            server_b = ToolWorldServer(ledger_b)
            repl = ReplicationEngine(ledger_b=ledger_b, clock=clock, lag_ms=lag)

            global_truth = GroundTruth()

            epoch_before_a = ledger_a.get_epoch()
            epoch_before_b = ledger_b.get_epoch()

            # ---- pre-gap phase --------------------------------------
            outage_start = 1.0
            outage_end = outage_start + float(outage_s)
            last_known_epoch = 0

            buffered_keys: dict[str, ScenarioAction] = {}
            for action in self._template.pre_gap_actions:
                key = action_key(
                    action.tool, dict(action.args), scheme,
                    intent_id=action.intent_id,
                    session_id=action.session_id,
                    turn_seq=action.turn_seq,
                )
                buf_rec = buffer.append(
                    intent_id=action.intent_id,
                    key=key,
                    tool=action.tool,
                    args=dict(action.args),
                    t0=outage_start if action.delivery == "in_flight" else 0.0,
                    ttl=action.ttl,
                    epoch=last_known_epoch,
                    status="PENDING",
                )
                buffered_keys[key] = action

                if action.delivery == "pre_gap":
                    resp = server_a.execute(
                        action.tool, dict(action.args),
                        intent_id=action.intent_id,
                        session_id=action.session_id,
                        turn_seq=action.turn_seq,
                        key_scheme=scheme,
                        clock=clock,
                    )
                    last_known_epoch = max(last_known_epoch, int(resp["epoch"]))
                    if resp["committed"]:
                        global_truth.record_application(action.effect_identity, clock())
                        repl.publish(_record_row(ledger_a, resp["key"]))
                    buffer.mark_committed(key)
                elif int(outage_s) == 0:
                    # zero outage: the in-flight action is delivered too.
                    resp = server_a.execute(
                        action.tool, dict(action.args),
                        intent_id=action.intent_id,
                        session_id=action.session_id,
                        turn_seq=action.turn_seq,
                        key_scheme=scheme,
                        clock=clock,
                    )
                    last_known_epoch = max(last_known_epoch, int(resp["epoch"]))
                    if resp["committed"]:
                        global_truth.record_application(action.effect_identity, clock())
                        repl.publish(_record_row(ledger_a, resp["key"]))
                    buffer.mark_committed(key)
                # else: in_flight during a nonzero outage -> dropped, stays PENDING.

            # replication continues through the outage (client link only)
            clock.advance(outage_end - clock() if outage_end > clock() else 0.0)
            repl.drain()
            reconnect_time = clock()

            # ---- reintegration ---------------------------------------
            reconnect_ledger = ledger_b if topo == "cross-node" else ledger_a
            reconnect_server = server_b if topo == "cross-node" else server_a

            decisions, epoch_guard_ok = self._reintegrate(
                method=method,
                buffer=buffer,
                ledger=reconnect_ledger,
                reconnect_server=reconnect_server,
                source_server=server_a,
                repl=repl,
                global_truth=global_truth,
                clock=clock,
                sleep_fn=sleep_fn,
                outage_s=int(outage_s),
                episode_id=int(episode_id),
                lag=lag,
                topo=topo,
                ablation_row=ablation_row,
                key_scheme=scheme,
                calibration_temperature=calibration_temperature,
                tau_star_path=tau_star_path,
            )

            # ---- post-gap client policy ------------------------------
            self._apply_client_policy(
                buffer=buffer,
                decisions=decisions,
                reconnect_server=reconnect_server,
                source_server=server_a,
                repl=repl,
                clock=clock,
                global_truth=global_truth,
                key_scheme=scheme,
            )

            epoch_after_a = ledger_a.get_epoch()
            epoch_after_b = ledger_b.get_epoch()
            ttr = max(0.0, clock() - reconnect_time)

            record = build_record(
                template=self._template,
                method=method,
                outage_s=int(outage_s),
                episode_id=int(episode_id),
                seed=seed,
                clock=clock,
                ttr_seconds=ttr,
                epoch_before_a=epoch_before_a,
                epoch_before_b=epoch_before_b,
                epoch_after_a=epoch_after_a,
                epoch_after_b=epoch_after_b,
                epoch_guard_ok=epoch_guard_ok,
                decisions=decisions,
                source_node=src,
                reconnect_node=re,
                topology=topo,
                lag=lag,
                ablation_row=ablation_row,
                key_scheme=scheme,
                calibration_temperature=calibration_temperature,
                tau_star_path=tau_star_path,
                global_truth=global_truth,
                reconnect_ledger=reconnect_ledger,
                buffer=buffer,
            )
            return validate_episode_record(record)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reintegrate(
        self,
        *,
        method: str,
        buffer: ActionBuffer,
        ledger: EffectLedger,
        reconnect_server: ToolWorldServer,
        source_server: ToolWorldServer,
        repl: ReplicationEngine,
        global_truth: GroundTruth,
        clock: SimClock,
        sleep_fn: SimSleep,
        outage_s: int,
        episode_id: int,
        lag: int,
        topo: str,
        ablation_row: Optional[AblationRow],
        key_scheme: str,
        calibration_temperature: Optional[float],
        tau_star_path: Optional[str | Path],
    ) -> tuple[dict[str, str], bool]:
        from experiments.phase5.tau_star import phi_tau_star_provider

        trust_src = topo == "same-node"  # genuine ABSENT only on primary ledger

        def v_pre(rec, state):
            return self._template.v_pre_outcome

        def v_post(rec):
            return self._template.v_post_outcome

        semantic = self._semantic_similarity

        # Intent fields for pre-gap buffer records (buffer itself does
        # NOT persist session_id/turn_seq — resolve from the template).
        intent_fields = {
            a.intent_id: (a.session_id, a.turn_seq, a.tool)
            for a in self._template.pre_gap_actions
        }

        def execute_fn(payload: dict[str, Any]) -> Any:
            tool = payload.get("tool")
            args = dict(payload.get("args", {}) or {})
            sid = payload.get("session_id")
            tseq = payload.get("turn_seq")
            iid = payload.get("intent_id")
            if sid is None and tseq is None and iid in intent_fields:
                sid, tseq, tool = intent_fields[iid]
            # Authority: the (scheme-resolved) key already on the payload
            # when present; otherwise fall back to server-side compute.
            force_key = payload.get("key")
            resp = reconnect_server.execute(
                tool, args,
                intent_id=iid,
                session_id=sid,
                turn_seq=tseq,
                key_scheme=key_scheme,
                clock=clock,
                force_key=force_key if key_scheme != "none" else None,
            )
            if resp["committed"] and iid:
                global_truth.record_application(
                    _effect_identity_of(self._template, iid),
                    clock(),
                )
                # Only source-node (edge-a) commits replicate onward.
                if reconnect_server is source_server:
                    repl.publish(_record_row(reconnect_server.ledger, resp["key"]))
            return resp

        from experiments.phase5.baselines import build_baseline

        row = ablation_row
        baseline = build_baseline(
            method=method,
            template=self._template,
            row=row,
            tau_star_provider=phi_tau_star_provider(
                temperature=(calibration_temperature
                             if calibration_temperature is not None
                             else self._template.calibration_temperature),
                scenario_temperature=self._template.calibration_temperature,
                path=tau_star_path,
            ),
            v_pre=v_pre,
            v_post=v_post,
            semantic_similarity=semantic,
            execute_fn=execute_fn,
            clock=clock,
            sleep_fn=sleep_fn,
            replicated_ledger=(topo == "cross-node"),
            use_v_post=(method in ("verify_before_retry", "kinbridge_sync")),
            sync_window_s=self._template.sync_window_s,
        )
        result = baseline.run(buffer, ledger)
        return dict(result.by_key), result.epoch_guard_ok

    def _apply_client_policy(
        self,
        buffer: ActionBuffer,
        decisions: dict[str, str],
        reconnect_server: ToolWorldServer,
        source_server: ToolWorldServer,
        repl: ReplicationEngine,
        clock: SimClock,
        global_truth: GroundTruth,
        key_scheme: str,
    ) -> None:
        """decision_based policy (documented Phase 5 harness semantics)."""
        for r in self._template.replanned_actions:
            buffered = [
                rec for rec in buffer.read_all()
                if rec.get("intent_id") == r.intent_id
            ]
            if not buffered:
                resp = reconnect_server.execute(
                    r.tool, dict(r.args),
                    intent_id=r.intent_id,
                    session_id=r.session_id,
                    turn_seq=r.turn_seq,
                    key_scheme=key_scheme,
                    clock=clock,
                )
                if resp["committed"]:
                    global_truth.record_application(
                        _effect_identity_of(self._template, r.intent_id), clock()
                    )
                    if reconnect_server is source_server:
                        repl.publish(_record_row(reconnect_server.ledger, resp["key"]))
                continue

            decision = decisions.get(buffered[0]["key"], "UNKNOWN")
            if decision in ("COMMITTED", "REPLAYED"):
                continue  # effect already exists; do not resend
            if decision == "MERGED":
                resp = reconnect_server.execute(
                    r.tool, dict(r.args),
                    intent_id=r.intent_id,
                    session_id=r.session_id,
                    turn_seq=r.turn_seq,
                    key_scheme=key_scheme,
                    clock=clock,
                )
                if resp["committed"]:
                    global_truth.record_application(
                        _effect_identity_of(self._template, r.intent_id), clock()
                    )
                    if reconnect_server is source_server:
                        repl.publish(_record_row(reconnect_server.ledger, resp["key"]))
                continue
            if decision == "DROPPED":
                # cold restart: client has no buffered state; it executes
                # the freshly planned action normally.  Availability loss,
                # and duplication is decided by ground truth afterwards.
                resp = reconnect_server.execute(
                    r.tool, dict(r.args),
                    intent_id=r.intent_id,
                    session_id=r.session_id,
                    turn_seq=r.turn_seq,
                    key_scheme=key_scheme,
                    clock=clock,
                )
                if resp["committed"]:
                    global_truth.record_application(
                        _effect_identity_of(self._template, r.intent_id), clock()
                    )
                    if reconnect_server is source_server:
                        repl.publish(_record_row(reconnect_server.ledger, resp["key"]))
                continue
            # ESCALATED / INVALIDATED / UNKNOWN -> skip (availability loss)


# ---------------------------------------------------------------------------
# record building / helpers
# ---------------------------------------------------------------------------


def _effect_identity_of(template: EpisodeTemplate, intent_id: str) -> str:
    for a in template.pre_gap_actions:
        if a.intent_id == intent_id:
            return a.effect_identity
    return f"UNKNOWN-{intent_id}"


def _record_row(ledger: EffectLedger, key: str) -> dict[str, Any]:
    rec = ledger.lookup(key)
    if rec is None:
        return {}
    return {
        "key": rec["key"],
        "tool": rec["tool"],
        "args_hash": rec["args_hash"],
        "status": rec["status"],
        "created_at": rec["created_at"],
        "committed_at": rec["committed_at"],
        "epoch": rec["epoch"],
        "ttl": rec["ttl"],
        "intent_id": rec["intent_id"],
    }


def episode_seed_of(template: EpisodeTemplate, method: str, outage_s: int, episode_id: int) -> int:
    from experiments.phase5.scenario import episode_seed
    return episode_seed(template.master_seed, template.scenario_id, method, int(outage_s), int(episode_id))


def _stale_of(template: EpisodeTemplate, action: ScenarioAction, app_times: list[float]) -> bool:
    if action.ttl <= 0 or not app_times:
        return False
    t0 = 1.0 if action.delivery == "in_flight" else 0.0
    return any((t - t0) > action.ttl for t in app_times)


def _ledger_status(ledger: EffectLedger, key: str, decision: str) -> str:
    rec = ledger.lookup(key)
    if rec is None:
        return "ABSENT" if decision != "DROPPED" else "DROPPED"
    return rec["status"]


def build_record(
    template: EpisodeTemplate,
    method: str,
    outage_s: int,
    episode_id: int,
    seed: int,
    clock: SimClock,
    ttr_seconds: float,
    epoch_before_a: int,
    epoch_before_b: int,
    epoch_after_a: int,
    epoch_after_b: int,
    epoch_guard_ok: bool,
    decisions: dict[str, str],
    source_node: str,
    reconnect_node: str,
    topology: str,
    lag: int,
    ablation_row: Optional[AblationRow],
    key_scheme: str,
    calibration_temperature: Optional[float],
    tau_star_path: Optional[str | Path],
    global_truth: GroundTruth,
    reconnect_ledger: EffectLedger,
    buffer: ActionBuffer,
) -> dict[str, Any]:
    from experiments.phase5.tau_star import calibration_provenance

    temp = calibration_temperature if calibration_temperature is not None else template.calibration_temperature
    prov = calibration_provenance(temp, path=tau_star_path)

    epoch_before = epoch_before_a if topology == "same-node" else epoch_before_b
    epoch_after = epoch_after_a if topology == "same-node" else epoch_after_b

    actions = []
    for action in template.pre_gap_actions:
        # Resolve the buffer key from the persisted buffer record
        # (authoritative for decision lookup after status merges).
        buffer_key = None
        for br in buffer.read_all():
            if br.get("intent_id") == action.intent_id:
                buffer_key = br["key"]
                break
        if buffer_key is None:
            buffer_key = action_key(
                action.tool, dict(action.args), key_scheme,
                intent_id=action.intent_id,
                session_id=action.session_id,
                turn_seq=action.turn_seq,
            )
        app_times = global_truth.applications(action.effect_identity)
        dup = len(app_times) > 1
        stale = _stale_of(template, action, app_times)
        decision = decisions.get(buffer_key, "DROPPED" if method == "cold_restart" else "UNKNOWN")
        escalated = decision in ("ESCALATED",)
        invalidated = decision in ("INVALIDATED",)
        committed = len(app_times) >= 1
        ledger_status = _ledger_status(reconnect_ledger, buffer_key, decision)

        absent_after = bool(escalated and ledger_status != "COMMITTED")

        actions.append({
            "action_id": action.action_id,
            "tool": action.tool,
            "args": dict(action.args),
            "intent_id": action.intent_id,
            "session_id": action.session_id,
            "turn_seq": action.turn_seq,
            "ttl": action.ttl,
            "effect_identity": action.effect_identity,
            "ledger_status": ledger_status,
            "decision": decision,
            "duplicate": dup,
            "stale_execution": stale,
            "escalated": escalated,
            "invalidated": invalidated,
            "committed": committed,
            "v_pre_outcome": template.v_pre_outcome,
            "v_post_outcome": template.v_post_outcome,
            "absent_after_epoch_guard": absent_after,
            "replication_ambiguity": bool(
                escalated and topology == "cross-node" and ledger_status != "COMMITTED"
            ),
            "safety_decision": "ESCALATE" if escalated else ("INVALIDATE" if invalidated else decision),
            "effect_applications": len(app_times),
        })

    return {
        "episode_id": int(episode_id),
        "method": method,
        "outage_profile": OUTAGE_PROFILE_NAMES.get(int(outage_s), f"custom-{int(outage_s)}"),
        "outage_duration": int(outage_s),
        "seed": template.master_seed,
        "episode_seed": int(seed),
        "temperature": template.temperature,
        "key_scheme": key_scheme,
        "replication_lag": int(lag),
        "topology": topology,
        "source_node": source_node,
        "reconnect_node": reconnect_node,
        "epoch_before": int(epoch_before),
        "epoch_after": int(epoch_after),
        "epoch_guard_result": bool(epoch_guard_ok),
        "actions": actions,
        "ttr_seconds": float(ttr_seconds),
        "tau_star": prov["tau_star"],
        "tau_star_source": prov["tau_star_source"],
        "scenario_id": template.scenario_id,
        "ablation_row": ablation_row.name if ablation_row is not None else "full",
    }
