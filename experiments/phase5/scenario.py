"""Episode scenario resolution (B1).

Loads ``config/experiment/phase5_scenario.yaml`` and resolves a
concrete, fully-specified episode definition.  Each episode is derived
from the scenario template plus the (method, outage_duration,
episode_id, seed) cell parameters, and carries a deterministic child
seed (B9/B10).

Deterministic child seeds are derived as:

    sha256(f"{master_seed}:{method}:{outage}:{episode_id}:{scenario_id}")
        -> int modulo 2**63

so the same configuration always produces the same episode seed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from core.config import CONFIG_DIR

SCENARIO_CONFIG = CONFIG_DIR / "experiment" / "phase5_scenario.yaml"

METHODS = ("cold_restart", "naive_retry", "verify_before_retry", "kinbridge_sync")
OUTAGES_S = (0, 10, 90, 300)


@dataclass(frozen=True)
class ScenarioAction:
    """A deterministic synthetic tool action with ground-truth identity."""

    action_id: str
    tool: str
    args: dict[str, Any]
    intent_id: str
    session_id: str
    turn_seq: int
    ttl: float
    delivery: str
    effect_identity: str


@dataclass(frozen=True)
class ReplannedAction:
    """A post-gap re-planned action (same intents, fresh turns)."""

    action_id: str
    tool: str
    args: dict[str, Any]
    intent_id: str
    session_id: str
    turn_seq: int


@dataclass(frozen=True)
class AblationRow:
    """One Table VI ablation row with explicit key scheme + flags (B3)."""

    name: str
    key_scheme: str
    enable_intent_keys: bool
    enable_staleness_check: bool
    enable_epoch_guard: bool
    enable_degradation_gate: bool


@dataclass(frozen=True)
class EpisodeTemplate:
    """The fully-resolved per-episode definition."""

    scenario_id: str
    master_seed: int
    temperature: float
    calibration_temperature: float
    semantic_model: str
    sim_horizon_s: float
    sync_window_s: float
    epoch_poll_interval_s: float
    pre_gap_actions: tuple[ScenarioAction, ...]
    replanned_actions: tuple[ReplannedAction, ...]
    v_pre_outcome: bool
    v_post_outcome: bool
    default_topology: str
    default_reconnect_node: str
    default_source_node: str
    default_replication_lag_ms: int
    c3_lag_options: tuple[int, ...] = field(default_factory=lambda: (0, 500, -1))
    c3_topologies: tuple[str, ...] = field(default_factory=lambda: ("same-node", "cross-node"))
    ablation_rows: tuple[AblationRow, ...] = field(default_factory=tuple)

    @property
    def eligible_actions(self) -> int:
        """Number of buffered actions per episode (unsafe-rate denominator)."""
        return len(self.pre_gap_actions)


def load_scenario(path: str | Path | None = None) -> EpisodeTemplate:
    """Load and validate the Phase 5 scenario configuration."""
    p = Path(path) if path is not None else SCENARIO_CONFIG
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ep = cfg["episode"]
    pre_gap = tuple(
        ScenarioAction(
            action_id=a["id"],
            tool=a["tool"],
            args=dict(a["args"]),
            intent_id=a["intent_id"],
            session_id=a["session_id"],
            turn_seq=int(a["turn_seq"]),
            ttl=float(a["ttl"]),
            delivery=a["delivery"],
            effect_identity=a["effect_identity"],
        )
        for a in ep["actions"]
    )
    replanned = tuple(
        ReplannedAction(
            action_id=r["id"],
            tool=r["tool"],
            args=dict(r["args"]),
            intent_id=r["intent_id"],
            session_id=r["session_id"],
            turn_seq=int(r["turn_seq"]),
        )
        for r in ep["replanned_actions"]
    )
    default = cfg.get("defaults", {})
    c3 = cfg.get("c3_grid", {})
    ablation = cfg.get("ablation_ladder", {})
    rows = tuple(
        AblationRow(
            name=str(r["row"]),
            key_scheme=str(r["key_scheme"]),
            enable_intent_keys=bool(r.get("enable_intent_keys", False)),
            enable_staleness_check=bool(r.get("enable_staleness_check", False)),
            enable_epoch_guard=bool(r.get("enable_epoch_guard", False)),
            enable_degradation_gate=bool(r.get("enable_degradation_gate", False)),
        )
        for r in ablation.get("rows", [])
    )
    template = EpisodeTemplate(
        scenario_id=str(cfg["scenario_id"]),
        master_seed=int(cfg["master_seed"]),
        temperature=float(cfg["temperature"]),
        calibration_temperature=float(cfg["calibration"]["temperature"]),
        semantic_model=str(cfg["semantic_model"]),
        sim_horizon_s=float(ep["sim_horizon_s"]),
        sync_window_s=float(ep["sync_window_s"]),
        epoch_poll_interval_s=float(ep["epoch_poll_interval_s"]),
        pre_gap_actions=pre_gap,
        replanned_actions=replanned,
        v_pre_outcome=bool(ep.get("v_pre_outcome", True)),
        v_post_outcome=bool(ep.get("v_post_outcome", False)),
        default_topology=str(default.get("topology", "same-node")),
        default_reconnect_node=str(default.get("reconnect_node", "edge-a")),
        default_source_node=str(default.get("source_node", "edge-a")),
        default_replication_lag_ms=int(default.get("replication_lag_ms", 0)),
        c3_lag_options=tuple(int(x) for x in c3.get("replication_lag_ms", [0, 500, -1])),
        c3_topologies=tuple(str(x) for x in c3.get("topologies", ["same-node", "cross-node"])),
        ablation_rows=rows,
    )
    _validate_template(template)
    return template


def _validate_template(t: EpisodeTemplate) -> None:
    if t.master_seed < 0:
        raise ValueError("master_seed must be non-negative")
    if not t.pre_gap_actions:
        raise ValueError("episode must define at least one pre-gap action")
    if not t.replanned_actions:
        raise ValueError("episode must define at least one replanned action")
    if t.temperature != t.calibration_temperature:
        raise ValueError(
            "scenario temperature must equal calibration temperature "
            "(tau* transfer requires a matching calibration context)"
        )
    intents = [a.intent_id for a in t.pre_gap_actions]
    if len(set(intents)) != len(intents):
        raise ValueError("pre-gap action intent_ids must be distinct within an episode")
    if t.sync_window_s < 0 or t.epoch_poll_interval_s <= 0:
        raise ValueError("invalid epoch-guard timing parameters")


def episode_seed(master_seed: int, scenario_id: str, method: str,
                 outage_s: int, episode_id: int) -> int:
    """Deterministic child seed for one episode (B10).

    Same inputs -> same child seed; different episodes -> different
    child seeds.
    """
    raw = f"{int(master_seed)}:{scenario_id}:{method}:{int(outage_s)}:{int(episode_id)}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2 ** 63)


def resolve_episode(
    template: EpisodeTemplate,
    method: str,
    outage_s: int,
    episode_id: int,
) -> dict[str, Any]:
    """Resolve the concrete parameters of one episode.

    Returns a dict consumed by the harness (B9 selection support):
    method, outage_s, episode_id, episode_seed, topology, nodes, lag,
    and a copy of the scenario action template.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if int(outage_s) not in OUTAGES_S:
        raise ValueError(f"unknown outage_duration {outage_s!r}; expected {OUTAGES_S}")
    if int(episode_id) < 0:
        raise ValueError("episode_id must be non-negative")

    return {
        "scenario_id": template.scenario_id,
        "method": method,
        "outage_duration_s": int(outage_s),
        "episode_id": int(episode_id),
        "episode_seed": episode_seed(
            template.master_seed, template.scenario_id, method, int(outage_s), int(episode_id)
        ),
        "master_seed": template.master_seed,
        "topology": template.default_topology,
        "source_node": template.default_source_node,
        "reconnect_node": template.default_reconnect_node,
        "replication_lag_ms": template.default_replication_lag_ms,
        "temperature": template.temperature,
    }


def resolve_c3_episode(
    template: EpisodeTemplate,
    lag_ms: int,
    topology: str,
    episode_id: int,
) -> dict[str, Any]:
    """Resolve a single C3-grid episode (Table V) for Kinbridge-Sync.

    ``lag_ms`` may be one of the artifact's C3 options; ``topology`` one
    of the C3 topologies.  Latency/lag values are the CONFIGURED
    replication lag, not a measured delivery latency.
    """
    if int(lag_ms) not in template.c3_lag_options:
        raise ValueError(
            f"unknown replication lag {lag_ms!r}; expected {template.c3_lag_options}"
        )
    if topology not in template.c3_topologies:
        raise ValueError(f"unknown topology {topology!r}; expected {template.c3_topologies}")

    source_node = "edge-a"
    if topology == "same-node":
        reconnect_node = "edge-a"
    else:
        reconnect_node = "edge-b"

    return {
        "scenario_id": template.scenario_id,
        "method": "kinbridge_sync",
        "outage_duration_s": 90,
        "episode_id": int(episode_id),
        "episode_seed": episode_seed(
            template.master_seed, template.scenario_id,
            "kinbridge_sync", 90, int(episode_id),
        ),
        "master_seed": template.master_seed,
        "topology": topology,
        "source_node": source_node,
        "reconnect_node": reconnect_node,
        "replication_lag_ms": int(lag_ms),
        "temperature": template.temperature,
    }