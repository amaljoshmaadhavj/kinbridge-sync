"""Ablation ladder resolution (B3 / B5).

The frozen ``config/experiment/ablation.yaml`` cannot express the six
Table VI rows (it collapses hash-vs-intent into booleans and lacks the
degradation-gate row).  The authoritative row definitions live in the
Phase 5 scenario configuration (``config/experiment/phase5_scenario.yaml``
``ablation_ladder``) and are resolved here.

Each row explicitly sets:
  - the key-generation scheme (payload | intent)
  - intent-key enabled/disabled
  - staleness check enabled/disabled
  - epoch guard enabled/disabled
  - degradation gate enabled/disabled

The six rows are distinguishable in configuration and in episode
metadata (``ablation_row`` field).

Degradation gate (Phase 5 HARNESS semantics, B5): a harness-level
termination gate.  If the connectivity/reintegration environment
remains in a state where safe reintegration cannot proceed (epoch
guard behind, tau* unavailable, F1 ABSENT ambiguity), the episode
entering the gate terminates the affected action as ESCALATED /
DEGRADED instead of continuing to execute unsafe actions.  It does NOT
override the epoch guard, tau* unavailability, F1 ABSENT ambiguity,
TTL/V_pre invalidation, or unresolved UNKNOWN handling — it is an
additional experimental termination mechanism, never a replacement for
Algorithm 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from experiments.phase5.scenario import AblationRow, EpisodeTemplate


@dataclass(frozen=True)
class ResolvedAblation:
    """One resolved ablation row with its flags and key scheme."""

    row: AblationRow
    method: str
    outage_s: int
    topology: str
    reconnect_node: str
    source_node: str
    replication_lag_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "ablation_row": self.row.name,
            "key_scheme": self.row.key_scheme,
            "enable_intent_keys": self.row.enable_intent_keys,
            "enable_staleness_check": self.row.enable_staleness_check,
            "enable_epoch_guard": self.row.enable_epoch_guard,
            "enable_degradation_gate": self.row.enable_degradation_gate,
            "method": self.method,
            "outage_duration_s": self.outage_s,
            "topology": self.topology,
            "reconnect_node": self.reconnect_node,
            "source_node": self.source_node,
            "replication_lag_ms": self.replication_lag_ms,
        }


def resolve_ablation_row(template: EpisodeTemplate, row_name: str) -> ResolvedAblation:
    """Resolve a single ablation row by name (e.g. ``plus_epoch_guard``).

    Raises:
        ValueError: if the row name is not one of the six Table VI rows.
    """
    for row in template.ablation_rows:
        if row.name == row_name:
            return ResolvedAblation(
                row=row,
                method="kinbridge_sync",
                outage_s=90,
                topology=template.default_topology,
                reconnect_node=template.default_reconnect_node,
                source_node=template.default_source_node,
                replication_lag_ms=template.default_replication_lag_ms,
            )
    raise ValueError(
        f"unknown ablation row {row_name!r}; expected one of "
        f"{[r.name for r in template.ablation_rows]}"
    )


def resolve_all_ablation_rows(template: EpisodeTemplate) -> list[ResolvedAblation]:
    """Resolve every ablation row in ladder order."""
    return [resolve_ablation_row(template, r.name) for r in template.ablation_rows]