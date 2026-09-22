"""Experiment results analysis — Tables III–VI generation (Phase 5).

Reads append-only episode JSONL (never mutates raw inputs), aggregates
per-cell metrics with Wilson 95% CIs, computes Cohen's h against the
cold-restart baseline, and writes:

    results/final/results.csv
    results/final/table_III.csv
    results/final/table_IV.csv
    results/final/table_V.csv
    results/final/table_VI.csv

Fail-loud on malformed records.  Every paper-facing proportion carries a
Wilson interval (Complete-Research.md §3.7).  No method is ranked
"best"; effect sizes are reported as Cohen's h vs cold_restart only.

Usage:
    python -m experiments.analyze_results --input results/final/raw/episodes.jsonl
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

from core.config import PROJECT_ROOT
from experiments.phase5.record import read_episode_records
from pilot.wilson import wilson_ci

TABLE_METHOD_ORDER = (
    "cold_restart",
    "naive_retry",
    "verify_before_retry",
    "kinbridge_sync",
)
TABLE_OUTAGE_ORDER = (0, 10, 90, 300)
ABLATION_ROW_ORDER = (
    "buffer_only",
    "plus_hash_keys",
    "plus_intent_keys",
    "plus_staleness",
    "plus_epoch_guard",
    "plus_degradation_gate",
)
C3_LAG_ORDER = (0, 500, -1)
C3_TOPOLOGY_ORDER = ("same-node", "cross-node")


class AnalysisError(ValueError):
    """Raised when inputs are missing, malformed, or internally inconsistent."""


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h between two proportions (arcsine transform).

    h = 2*arcsin(sqrt(p1)) - 2*arcsin(sqrt(p2))
    """
    if not (0.0 <= p1 <= 1.0 and 0.0 <= p2 <= 1.0):
        raise AnalysisError(f"proportions must be in [0,1], got p1={p1!r} p2={p2!r}")
    return 2.0 * math.asin(math.sqrt(p1)) - 2.0 * math.asin(math.sqrt(p2))


def _action_flags(rec: dict[str, Any]) -> tuple[int, int, int, int, int, int, float]:
    """Return (n, n_dup, n_stale, n_esc, n_inv, n_unsafe, ttr)."""
    actions = rec.get("actions") or []
    if not isinstance(actions, list):
        raise AnalysisError(f"episode {rec.get('episode_id')}: actions must be a list")
    n = len(actions)
    n_dup = 0
    n_stale = 0
    n_esc = 0
    n_inv = 0
    n_unsafe = 0
    for a in actions:
        if not isinstance(a, dict):
            raise AnalysisError(f"episode {rec.get('episode_id')}: action must be a dict")
        dup = bool(a.get("duplicate"))
        stale = bool(a.get("stale_execution"))
        if dup:
            n_dup += 1
        if stale:
            n_stale += 1
        if a.get("escalated"):
            n_esc += 1
        if a.get("invalidated"):
            n_inv += 1
        if dup or stale:
            n_unsafe += 1
    ttr = rec.get("ttr_seconds")
    if not isinstance(ttr, (int, float)) or isinstance(ttr, bool) or ttr < 0:
        raise AnalysisError(f"episode {rec.get('episode_id')}: bad ttr_seconds {ttr!r}")
    return n, n_dup, n_stale, n_esc, n_inv, n_unsafe, float(ttr)


def _wilson_row(numerator: int, denominator: int) -> dict[str, Any]:
    ci = wilson_ci(numerator, denominator, z=1.96)
    return {
        "numerator": ci.numerator,
        "denominator": ci.denominator,
        "rate": ci.estimate,
        "ci_low": ci.ci_low,
        "ci_high": ci.ci_high,
    }


def _median(values: list[float]) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    m = len(xs) // 2
    if len(xs) % 2:
        return xs[m]
    return (xs[m - 1] + xs[m]) / 2.0


def aggregate_factorial(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (method, outage) with unsafe/duplicate/stale/escalation rates."""
    cells: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        method = rec.get("method")
        outage = rec.get("outage_duration")
        if method not in TABLE_METHOD_ORDER:
            raise AnalysisError(f"unexpected method {method!r}")
        if outage not in TABLE_OUTAGE_ORDER:
            raise AnalysisError(f"unexpected outage_duration {outage!r}")
        if rec.get("ablation_row", "full") not in (None, "full"):
            # Ablation episodes are not part of Tables III/IV.
            continue
        if rec.get("topology", "same-node") != "same-node":
            continue
        cells[(method, int(outage))].append(rec)

    rows: list[dict[str, Any]] = []
    for method in TABLE_METHOD_ORDER:
        for outage in TABLE_OUTAGE_ORDER:
            group = cells.get((method, outage), [])
            n_eps = len(group)
            sum_n = sum_dup = sum_stale = sum_esc = sum_inv = sum_unsafe = 0
            ttrs: list[float] = []
            for rec in group:
                n, d, s, e, i, u, ttr = _action_flags(rec)
                sum_n += n
                sum_dup += d
                sum_stale += s
                sum_esc += e
                sum_inv += i
                sum_unsafe += u
                ttrs.append(ttr)
            unsafe = _wilson_row(sum_unsafe, sum_n)
            dup = _wilson_row(sum_dup, sum_n)
            stale = _wilson_row(sum_stale, sum_n)
            esc = _wilson_row(sum_esc, sum_n)
            rows.append({
                "method": method,
                "outage_duration": outage,
                "episodes": n_eps,
                "eligible_actions": sum_n,
                "n_duplicate": sum_dup,
                "n_stale": sum_stale,
                "n_escalated": sum_esc,
                "n_invalidated": sum_inv,
                "n_unsafe": sum_unsafe,
                "unsafe_rate": unsafe["rate"],
                "unsafe_ci_low": unsafe["ci_low"],
                "unsafe_ci_high": unsafe["ci_high"],
                "duplicate_rate": dup["rate"],
                "duplicate_ci_low": dup["ci_low"],
                "duplicate_ci_high": dup["ci_high"],
                "stale_rate": stale["rate"],
                "stale_ci_low": stale["ci_low"],
                "stale_ci_high": stale["ci_high"],
                "escalation_rate": esc["rate"],
                "escalation_ci_low": esc["ci_low"],
                "escalation_ci_high": esc["ci_high"],
                "median_ttr": _median(ttrs),
            })
    return rows


def aggregate_table_iv(factorial_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Method-level decomposition (Table IV): pool outages within method."""
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in factorial_rows:
        by_method[row["method"]].append(row)

    rows: list[dict[str, Any]] = []
    for method in TABLE_METHOD_ORDER:
        group = by_method.get(method, [])
        sum_n = sum(
            r["eligible_actions"] for r in group
        )
        sum_dup = sum(r["n_duplicate"] for r in group)
        sum_stale = sum(r["n_stale"] for r in group)
        sum_esc = sum(r["n_escalated"] for r in group)
        sum_inv = sum(r["n_invalidated"] for r in group)
        sum_unsafe = sum(r["n_unsafe"] for r in group)
        # Re-read TTRs from cell rows is not possible (medians only);
        # recompute from factorial_rows medians weighted equally per cell
        # for the published median-of-cell-medians, AND keep cell-level
        # medians in table_III.  Table IV reports pooled rates + mean of
        # cell medians is wrong — recompute properly below from episodes
        # is preferred; here we store None and fill from records in
        # analyze() when episode-level TTRs are available.
        dup = _wilson_row(sum_dup, sum_n)
        stale = _wilson_row(sum_stale, sum_n)
        esc = _wilson_row(sum_esc, sum_n)
        unsafe = _wilson_row(sum_unsafe, sum_n)
        rows.append({
            "method": method,
            "episodes": sum(r["episodes"] for r in group),
            "eligible_actions": sum_n,
            "duplicate_rate": dup["rate"],
            "duplicate_ci_low": dup["ci_low"],
            "duplicate_ci_high": dup["ci_high"],
            "stale_rate": stale["rate"],
            "stale_ci_low": stale["ci_low"],
            "stale_ci_high": stale["ci_high"],
            "escalation_rate": esc["rate"],
            "escalation_ci_low": esc["ci_low"],
            "escalation_ci_high": esc["ci_high"],
            "unsafe_rate": unsafe["rate"],
            "unsafe_ci_low": unsafe["ci_low"],
            "unsafe_ci_high": unsafe["ci_high"],
            "n_invalidated": sum_inv,
            "median_ttr": None,
            "cohens_h_vs_cold": None,
        })
    return rows


def attach_table_iv_ttr_and_h(
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fill median TTR and Cohen's h (vs cold_restart) on Table IV rows."""
    ttr_by_method: dict[str, list[float]] = defaultdict(list)
    unsafe_num: dict[str, int] = defaultdict(int)
    unsafe_den: dict[str, int] = defaultdict(int)
    for rec in records:
        method = rec.get("method")
        if method not in TABLE_METHOD_ORDER:
            continue
        if rec.get("ablation_row", "full") not in (None, "full"):
            continue
        if rec.get("topology", "same-node") != "same-node":
            continue
        _, _, _, _, _, u, ttr = _action_flags(rec)
        ttr_by_method[method].append(ttr)
        unsafe_num[method] += u
        unsafe_den[method] += len(rec.get("actions") or [])

    cold_p = (
        unsafe_num["cold_restart"] / unsafe_den["cold_restart"]
        if unsafe_den.get("cold_restart")
        else 0.0
    )
    for row in rows:
        method = row["method"]
        row["median_ttr"] = _median(ttr_by_method.get(method, []))
        den = unsafe_den.get(method, 0)
        p = unsafe_num[method] / den if den else 0.0
        row["cohens_h_vs_cold"] = cohens_h(p, cold_p)
        row["unsafe_rate"] = p
        wi = _wilson_row(unsafe_num[method], den)
        row["unsafe_ci_low"] = wi["ci_low"]
        row["unsafe_ci_high"] = wi["ci_high"]
        row["eligible_actions"] = den
        row["n_unsafe"] = unsafe_num[method]
    return rows


def aggregate_c3(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Table V: kinbridge_sync × lag × topology (C3)."""
    cells: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        if rec.get("method") != "kinbridge_sync":
            continue
        if rec.get("ablation_row", "full") not in (None, "full"):
            continue
        # C3 episodes are outage 90 with explicit lag/topology.
        if int(rec.get("outage_duration", -1)) != 90:
            continue
        lag = int(rec.get("replication_lag", 0))
        topo = str(rec.get("topology", "same-node"))
        if (lag, topo) not in {
            (l, t) for l in C3_LAG_ORDER for t in C3_TOPOLOGY_ORDER
        }:
            continue
        # Skip default factorial same-node lag0 rows that are not C3-tagged
        # when the grid mixed them — C3 rows share lag/topology keys with
        # factorial defaults, so only include if lag differs from a pure
        # factorial marker.  We include all kinbridge@90 same-node lag0 and
        # any cross-node/nonzero-lag kinbridge@90 rows.
        cells[(lag, topo)].append(rec)

    rows: list[dict[str, Any]] = []
    for lag in C3_LAG_ORDER:
        for topo in C3_TOPOLOGY_ORDER:
            group = cells.get((lag, topo), [])
            sum_n = sum_dup = sum_stale = sum_esc = sum_unsafe = 0
            ttrs: list[float] = []
            guards_ok = 0
            for rec in group:
                n, d, s, e, _, u, ttr = _action_flags(rec)
                sum_n += n
                sum_dup += d
                sum_stale += s
                sum_esc += e
                sum_unsafe += u
                ttrs.append(ttr)
                if rec.get("epoch_guard_result"):
                    guards_ok += 1
            unsafe = _wilson_row(sum_unsafe, sum_n)
            rows.append({
                "replication_lag_ms": lag,
                "topology": topo,
                "method": "kinbridge_sync",
                "episodes": len(group),
                "eligible_actions": sum_n,
                "unsafe_rate": unsafe["rate"],
                "unsafe_ci_low": unsafe["ci_low"],
                "unsafe_ci_high": unsafe["ci_high"],
                "n_duplicate": sum_dup,
                "n_stale": sum_stale,
                "n_escalated": sum_esc,
                "n_unsafe": sum_unsafe,
                "median_ttr": _median(ttrs),
                "epoch_guard_pass_episodes": guards_ok,
            })
    return rows


def aggregate_ablation(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Table VI: ablation ladder rows (kinbridge_sync @ 90s)."""
    by_row: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        row_name = rec.get("ablation_row", "full")
        if row_name not in ABLATION_ROW_ORDER:
            continue
        if rec.get("method") != "kinbridge_sync":
            continue
        by_row[str(row_name)].append(rec)

    # Reference: plus_intent_keys or full — use buffer_only-independent
    # cold-style baseline is not in ablation; report absolute rates and
    # h vs buffer_only (first rung) as secondary, plus h vs cold when
    # cold records exist at the same outage.
    rows: list[dict[str, Any]] = []
    buffer_p = None
    if by_row.get("buffer_only"):
        num = den = 0
        for rec in by_row["buffer_only"]:
            n, _, _, _, _, u, _ = _action_flags(rec)
            num += u
            den += n
        buffer_p = num / den if den else 0.0

    for name in ABLATION_ROW_ORDER:
        group = by_row.get(name, [])
        sum_n = sum_dup = sum_stale = sum_esc = sum_inv = sum_unsafe = 0
        ttrs: list[float] = []
        for rec in group:
            n, d, s, e, i, u, ttr = _action_flags(rec)
            sum_n += n
            sum_dup += d
            sum_stale += s
            sum_esc += e
            sum_inv += i
            sum_unsafe += u
            ttrs.append(ttr)
        unsafe = _wilson_row(sum_unsafe, sum_n)
        p = unsafe["rate"]
        rows.append({
            "ablation_row": name,
            "episodes": len(group),
            "eligible_actions": sum_n,
            "unsafe_rate": p,
            "unsafe_ci_low": unsafe["ci_low"],
            "unsafe_ci_high": unsafe["ci_high"],
            "n_duplicate": sum_dup,
            "n_stale": sum_stale,
            "n_escalated": sum_esc,
            "n_invalidated": sum_inv,
            "n_unsafe": sum_unsafe,
            "median_ttr": _median(ttrs),
            "cohens_h_vs_buffer_only": (
                cohens_h(p, buffer_p) if buffer_p is not None and group else None
            ),
        })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _print_table(title: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    print(f"\n=== {title} ===")
    if not rows:
        print("(no rows)")
        return
    print(" | ".join(columns))
    print("-|-".join("-" * len(c) for c in columns))
    for row in rows:
        cells = []
        for c in columns:
            v = row.get(c)
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append("" if v is None else str(v))
        print(" | ".join(cells))


def analyze(
    input_jsonl: Path,
    output_dir: Path,
    *,
    make_figures: bool = False,
) -> dict[str, Any]:
    """Read raw episode records and write results + Tables III–VI."""
    records = read_episode_records(input_jsonl)
    if not records:
        raise AnalysisError(f"no episode records in {input_jsonl}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Flat results.csv (one row per episode).
    from experiments.run_experiment import CSV_FIELDS, _episode_row

    results_path = output_dir / "results.csv"
    _write_csv(results_path, [_episode_row(r) for r in records], list(CSV_FIELDS))

    factorial_rows = aggregate_factorial(records)
    table_iii = _write_csv(
        output_dir / "table_III.csv",
        factorial_rows,
        [
            "method", "outage_duration", "episodes", "eligible_actions",
            "n_unsafe", "unsafe_rate", "unsafe_ci_low", "unsafe_ci_high",
            "n_duplicate", "duplicate_rate", "duplicate_ci_low", "duplicate_ci_high",
            "n_stale", "stale_rate", "stale_ci_low", "stale_ci_high",
            "n_escalated", "escalation_rate", "escalation_ci_low", "escalation_ci_high",
            "median_ttr",
        ],
    )

    table_iv_rows = attach_table_iv_ttr_and_h(
        aggregate_table_iv(factorial_rows), records
    )
    table_iv = _write_csv(
        output_dir / "table_IV.csv",
        table_iv_rows,
        [
            "method", "episodes", "eligible_actions",
            "n_duplicate", "duplicate_rate", "duplicate_ci_low", "duplicate_ci_high",
            "n_stale", "stale_rate", "stale_ci_low", "stale_ci_high",
            "n_escalated", "escalation_rate", "escalation_ci_low", "escalation_ci_high",
            "n_unsafe", "unsafe_rate", "unsafe_ci_low", "unsafe_ci_high",
            "median_ttr", "cohens_h_vs_cold", "n_invalidated",
        ],
    )

    c3_rows = aggregate_c3(records)
    table_v = _write_csv(
        output_dir / "table_V.csv",
        c3_rows,
        [
            "replication_lag_ms", "topology", "method", "episodes",
            "eligible_actions", "n_unsafe", "unsafe_rate", "unsafe_ci_low",
            "unsafe_ci_high", "n_duplicate", "n_stale", "n_escalated",
            "median_ttr", "epoch_guard_pass_episodes",
        ],
    )

    ablation_rows = aggregate_ablation(records)
    table_vi = _write_csv(
        output_dir / "table_VI.csv",
        ablation_rows,
        [
            "ablation_row", "episodes", "eligible_actions", "n_unsafe",
            "unsafe_rate", "unsafe_ci_low", "unsafe_ci_high",
            "n_duplicate", "n_stale", "n_escalated", "n_invalidated",
            "median_ttr", "cohens_h_vs_buffer_only",
        ],
    )

    if make_figures:
        _maybe_write_figures(output_dir, factorial_rows)

    summary = {
        "input": str(input_jsonl),
        "episodes": len(records),
        "results_csv": str(results_path),
        "table_III": str(table_iii),
        "table_IV": str(table_iv),
        "table_V": str(table_v),
        "table_VI": str(table_vi),
        "factorial_cells": len(factorial_rows),
        "c3_cells": len(c3_rows),
        "ablation_cells": len(ablation_rows),
    }

    _print_table(
        "Table III — unsafe action rate (method × outage)",
        factorial_rows,
        ["method", "outage_duration", "episodes", "unsafe_rate",
         "unsafe_ci_low", "unsafe_ci_high", "median_ttr"],
    )
    _print_table(
        "Table IV — decomposition by method",
        table_iv_rows,
        ["method", "duplicate_rate", "stale_rate", "escalation_rate",
         "unsafe_rate", "median_ttr", "cohens_h_vs_cold"],
    )
    _print_table(
        "Table V — C3 lag × topology (kinbridge_sync)",
        c3_rows,
        ["replication_lag_ms", "topology", "episodes", "unsafe_rate",
         "median_ttr", "epoch_guard_pass_episodes"],
    )
    _print_table(
        "Table VI — ablation ladder",
        ablation_rows,
        ["ablation_row", "episodes", "unsafe_rate", "n_duplicate",
         "n_stale", "n_escalated", "cohens_h_vs_buffer_only"],
    )
    return summary


def _maybe_write_figures(output_dir: Path, factorial_rows: list[dict[str, Any]]) -> None:
    """Optional matplotlib figure (unsafe rate vs outage per method)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable — skipping figures", file=sys.stderr)
        return

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in factorial_rows:
        by_method[row["method"]].append(row)

    fig, ax = plt.subplots(figsize=(7, 4))
    for method in TABLE_METHOD_ORDER:
        xs = [r["outage_duration"] for r in by_method.get(method, [])]
        ys = [r["unsafe_rate"] for r in by_method.get(method, [])]
        if xs:
            ax.plot(xs, ys, marker="o", label=method)
    ax.set_xlabel("Outage duration (s)")
    ax.set_ylabel("Unsafe action rate")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Unsafe action rate by method and outage")
    fig.tight_layout()
    fig.savefig(fig_dir / "unsafe_rate_vs_outage.png", dpi=150)
    plt.close(fig)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="analyze_results",
        description="Kinbridge-Sync Phase 5 analysis (Tables III–VI)",
    )
    p.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "results" / "final" / "raw" / "episodes.jsonl",
        help="Episode JSONL from run_experiment",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "final",
        help="Directory for results.csv and table_*.csv",
    )
    p.add_argument(
        "--figures",
        action="store_true",
        help="Also write PNG figures under <output-dir>/figures/",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        summary = analyze(args.input, args.output_dir, make_figures=args.figures)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except AnalysisError as exc:
        print(f"analysis error: {exc}", file=sys.stderr)
        return 1
    print(json_dumps(summary))
    return 0


def json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, indent=2, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())
