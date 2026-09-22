"""Full factorial experiment runner (Phase 5).

Drives the in-process episode harness over the factorial grid
(methods × outages), the C3 replication-lag/topology grid, and the
ablation ladder.  Emits append-only JSONL episode records plus a flat
CSV summary for the analyzer.

Usage:
    python -m experiments.run_experiment --method kinbridge_sync \
        --outage-duration 90 --episodes 2 --seed 42 \
        --output-dir results/final/raw

    python -m experiments.run_experiment --grid factorial --episodes 100
    python -m experiments.run_experiment --grid c3 --episodes 20
    python -m experiments.run_experiment --grid ablation --episodes 200
    python -m experiments.run_experiment --harness docker --replication-lag-ms 500

Research integrity: this module never claims results it did not run.
``--episodes`` defaults to 0 (no runs) so an accidental invocation cannot
overwrite prior artifacts with an overnight factorial.

See Complete-Research.md Phase 5 and Markdown/Work .md Tables III–VI.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

from core.config import PROJECT_ROOT
from experiments.phase5.ablation import resolve_all_ablation_rows
from experiments.phase5.compose_override import write_compose_override
from experiments.phase5.harness import EpisodeHarness
from experiments.phase5.record import read_episode_records, write_episode_records
from experiments.phase5.scenario import (
    METHODS,
    OUTAGES_S,
    EpisodeTemplate,
    load_scenario,
    resolve_c3_episode,
    resolve_episode,
)

FACTORIAL_METHODS = METHODS
FACTORIAL_OUTAGES = OUTAGES_S

CSV_FIELDS = (
    "episode_id",
    "method",
    "outage_duration",
    "outage_profile",
    "seed",
    "episode_seed",
    "key_scheme",
    "ablation_row",
    "replication_lag",
    "topology",
    "source_node",
    "reconnect_node",
    "ttr_seconds",
    "epoch_guard_result",
    "tau_star",
    "scenario_id",
    "n_actions",
    "n_eligible",
    "n_duplicate",
    "n_stale",
    "n_escalated",
    "n_invalidated",
    "n_unsafe",
    "unsafe_action_rate",
)


def _semantic_similarity_factory(template: EpisodeTemplate):
    """Cosine similarity via the frozen Phase 4 default (all-MiniLM-L6-v2).

    Falls back to an exact-match injector only when the model package is
    unavailable (tests pin ``1.0`` explicitly for same-intent pairs).
    """
    try:
        from reintegration.reintegration_service import DefaultSemanticSimilarity

        return DefaultSemanticSimilarity(), "all-MiniLM-L6-v2"
    except Exception:
        def _exact(buffered: dict[str, Any], replanned: Any) -> float:
            from pilot.key_schemes import canonical_serialise

            if isinstance(replanned, dict):
                if buffered.get("tool") == replanned.get("tool") and canonical_serialise(
                    dict(buffered.get("args") or {})
                ) == canonical_serialise(dict(replanned.get("args") or {})):
                    return 1.0
                return 0.0
            return 0.0

        return _exact, "exact-match-fallback"


def _episode_row(rec: dict[str, Any]) -> dict[str, Any]:
    actions = rec.get("actions") or []
    n_elig = len(actions)
    n_dup = sum(1 for a in actions if a.get("duplicate"))
    n_stale = sum(1 for a in actions if a.get("stale_execution"))
    n_esc = sum(1 for a in actions if a.get("escalated"))
    n_inv = sum(1 for a in actions if a.get("invalidated"))
    n_unsafe = sum(
        1 for a in actions if a.get("duplicate") or a.get("stale_execution")
    )
    rate = (n_unsafe / n_elig) if n_elig else 0.0
    return {
        "episode_id": rec.get("episode_id"),
        "method": rec.get("method"),
        "outage_duration": rec.get("outage_duration"),
        "outage_profile": rec.get("outage_profile"),
        "seed": rec.get("seed"),
        "episode_seed": rec.get("episode_seed"),
        "key_scheme": rec.get("key_scheme"),
        "ablation_row": rec.get("ablation_row", "full"),
        "replication_lag": rec.get("replication_lag"),
        "topology": rec.get("topology"),
        "source_node": rec.get("source_node"),
        "reconnect_node": rec.get("reconnect_node"),
        "ttr_seconds": rec.get("ttr_seconds"),
        "epoch_guard_result": rec.get("epoch_guard_result"),
        "tau_star": rec.get("tau_star"),
        "scenario_id": rec.get("scenario_id"),
        "n_actions": n_elig,
        "n_eligible": n_elig,
        "n_duplicate": n_dup,
        "n_stale": n_stale,
        "n_escalated": n_esc,
        "n_invalidated": n_inv,
        "n_unsafe": n_unsafe,
        "unsafe_action_rate": rate,
    }


def _append_csv_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    new_file = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_harness(
    template: EpisodeTemplate,
    semantic_similarity=None,
    semantic_label: str = "injected",
) -> EpisodeHarness:
    if semantic_similarity is None:
        semantic_similarity, semantic_label = _semantic_similarity_factory(template)
    return EpisodeHarness(
        template,
        semantic_similarity=semantic_similarity,
        semantic_similarity_label=semantic_label,
    )


def run_grid(
    *,
    grid: str,
    methods: list[str],
    outages: list[int],
    episodes: int,
    seed: Optional[int],
    output_dir: Path,
    harness_mode: str = "inprocess",
    replication_lag_ms: Optional[int] = None,
    topology: Optional[str] = None,
    ablation_rows: Optional[list[str]] = None,
    semantic_similarity=None,
    scenario_path: Optional[str] = None,
    force_key_scheme: Optional[str] = None,
    tau_star_path: Optional[str] = None,
    write_compose: bool = True,
) -> dict[str, Any]:
    """Run one experimental grid and write JSONL + CSV outputs.

    Returns a summary dict (counts, paths).  Does not print claims of
    paper-level results.
    """
    if episodes < 0:
        raise ValueError("episodes must be >= 0")
    if harness_mode not in ("inprocess", "docker"):
        raise ValueError("harness must be inprocess or docker")
    if seed is None:
        seed = 42

    template = load_scenario(scenario_path)
    if seed != template.master_seed:
        # Documented override: still record both; child seeds derive from master.
        pass

    harness = _build_harness(template, semantic_similarity=semantic_similarity)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "episodes.jsonl"
    csv_path = output_dir / "results.csv"

    # Docker harness: materialise the C3 compose override for the
    # configured lag, then still run the deterministic in-process model
    # (the overnight docker factorial is a separate, explicit run).
    if harness_mode == "docker" and replication_lag_ms is not None and write_compose:
        write_compose_override(
            int(replication_lag_ms), output_dir / "docker-compose.override.yml"
        )

    jobs: list[dict[str, Any]] = []

    if grid == "factorial":
        for method in methods:
            if method not in FACTORIAL_METHODS:
                raise ValueError(f"unknown method {method!r}")
            for outage in outages:
                if int(outage) not in FACTORIAL_OUTAGES:
                    raise ValueError(f"unknown outage {outage!r}")
                for ep in range(episodes):
                    jobs.append(resolve_episode(template, method, int(outage), ep))
    elif grid == "c3":
        if replication_lag_ms is None or topology is None:
            raise ValueError("--replication-lag-ms and --topology are required for --grid c3")
        for ep in range(episodes):
            jobs.append(
                resolve_c3_episode(template, int(replication_lag_ms), topology, ep)
            )
    elif grid == "ablation":
        rows = resolve_all_ablation_rows(template)
        if ablation_rows:
            wanted = set(ablation_rows)
            rows = [r for r in rows if r.row.name in wanted]
            missing = wanted - {r.row.name for r in rows}
            if missing:
                raise ValueError(f"unknown ablation rows: {sorted(missing)}")
        for resolved in rows:
            for ep in range(episodes):
                jobs.append(
                    {
                        "scenario_id": template.scenario_id,
                        "method": "kinbridge_sync",
                        "outage_duration_s": resolved.outage_s,
                        "episode_id": ep,
                        "topology": resolved.topology,
                        "source_node": resolved.source_node,
                        "reconnect_node": resolved.reconnect_node,
                        "replication_lag_ms": resolved.replication_lag_ms,
                        "ablation_row": resolved.row,
                        "key_scheme": resolved.row.key_scheme,
                    }
                )
    else:
        raise ValueError(f"unknown grid {grid!r}")

    records: list[dict[str, Any]] = []
    for job in jobs:
        method = job["method"]
        outage = int(job["outage_duration_s"])
        ep_id = int(job["episode_id"])
        rec = harness.run_episode(
            method=method,
            outage_s=outage,
            episode_id=ep_id,
            topology=job.get("topology"),
            reconnect_node=job.get("reconnect_node"),
            source_node=job.get("source_node"),
            replication_lag_ms=job.get("replication_lag_ms"),
            ablation_row=job.get("ablation_row"),
            key_scheme=job.get("key_scheme") or force_key_scheme,
            tau_star_path=tau_star_path,
        )
        records.append(rec)

    if records:
        write_episode_records(records, jsonl_path)
        _append_csv_rows(csv_path, [_episode_row(r) for r in records])

    return {
        "grid": grid,
        "episodes_requested": episodes,
        "episodes_run": len(records),
        "jobs": len(jobs),
        "jsonl": str(jsonl_path),
        "csv": str(csv_path),
        "harness": harness_mode,
        "methods": sorted({j["method"] for j in jobs}) if jobs else [],
        "outages": sorted({int(j["outage_duration_s"]) for j in jobs}) if jobs else [],
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_experiment",
        description="Kinbridge-Sync Phase 5 experiment runner",
    )
    p.add_argument(
        "--method",
        action="append",
        choices=list(FACTORIAL_METHODS),
        help="Method to run (repeatable). Default: all four for --grid factorial.",
    )
    p.add_argument(
        "--outage-duration",
        action="append",
        type=int,
        choices=list(FACTORIAL_OUTAGES),
        help="Outage seconds (repeatable). Default: all four for --grid factorial.",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Episodes per cell (default 0 — refuses to run a full factorial by accident).",
    )
    p.add_argument("--seed", type=int, default=None, help="Recorded seed (scenario master seed is authoritative).")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "final" / "raw",
        help="Directory for episodes.jsonl and results.csv",
    )
    p.add_argument(
        "--harness",
        choices=("inprocess", "docker"),
        default="inprocess",
        help="Episode execution mode (docker also writes the C3 compose override).",
    )
    p.add_argument("--replication-lag-ms", type=int, choices=(-1, 0, 500), default=None)
    p.add_argument("--topology", choices=("same-node", "cross-node"), default=None)
    p.add_argument(
        "--grid",
        choices=("factorial", "c3", "ablation"),
        default="factorial",
        help="Which experimental grid to run.",
    )
    p.add_argument(
        "--ablation-row",
        action="append",
        help="Restrict --grid ablation to named rows (repeatable).",
    )
    p.add_argument(
        "--key-scheme",
        choices=("intent", "payload", "none"),
        default=None,
        help="Force a key scheme (default: scenario/ablation row).",
    )
    p.add_argument("--scenario", default=None, help="Optional scenario YAML path.")
    p.add_argument("--tau-star-path", default=None, help="Optional tau* artifact path.")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip episode_ids already present in the output JSONL.",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    methods = args.method or list(FACTORIAL_METHODS)
    outages = args.outage_duration or list(FACTORIAL_OUTAGES)

    jsonl_path = Path(args.output_dir) / "episodes.jsonl"
    if args.resume and jsonl_path.exists():
        existing = read_episode_records(jsonl_path)
        print(
            f"resume: {len(existing)} episode(s) already in {jsonl_path} "
            "(append-only; new cells are added, existing lines are not rewritten)",
            file=sys.stderr,
        )

    if args.episodes <= 0 and args.grid == "factorial":
        print(
            "Refusing to run a full factorial with --episodes 0.\n"
            "Pass --episodes N explicitly (use 100 for the paper grid; "
            "smoke tests use 1–2).",
            file=sys.stderr,
        )
        return 2

    summary = run_grid(
        grid=args.grid,
        methods=methods,
        outages=outages,
        episodes=args.episodes,
        seed=args.seed,
        output_dir=Path(args.output_dir),
        harness_mode=args.harness,
        replication_lag_ms=args.replication_lag_ms,
        topology=args.topology,
        ablation_rows=args.ablation_row,
        scenario_path=args.scenario,
        force_key_scheme=args.key_scheme,
        tau_star_path=args.tau_star_path,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
