"""
Phase 1 analysis — Table II generation and decision rule.

Reads ONLY measured experiment output from pilot/raw/*.json.
Computes per-(scheme, temperature) divergence rate + Wilson 95 % CI.
Applies the paper's decision rule.

Usage:
    python -m pilot.analyze_pilot [--input pilot/raw/pilot_temp0.0_*.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from pilot.wilson import wilson_ci

_RAW_DIR = Path(__file__).resolve().parent / "raw"


def _load_observations(raw_path: Path) -> list[dict[str, Any]]:
    """Load observations from a raw JSON file."""
    with open(raw_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["observations"]


def _load_all_observations() -> list[dict[str, Any]]:
    """Load all raw observation files from pilot/raw/."""
    obs: list[dict[str, Any]] = []
    for p in sorted(_RAW_DIR.glob("pilot_temp*.json")):
        obs.extend(_load_observations(p))
    return obs


def compute_divergence_table(
    observations: list[dict[str, Any]],
) -> pd.DataFrame:
    """Compute per-(scheme, temperature, tool_class) divergence rate + Wilson CI.

    Returns a DataFrame matching Table II's row structure.
    """
    rows: list[dict[str, Any]] = []

    for obs in observations:
        temp = obs["temperature"]
        tc = obs["tool_class"]

        for scheme, key_a, key_b in [
            ("k_payload", "a1.key_payload", "a2.key_payload"),
            ("k_bucket",  "a1.key_bucket",  "a2.key_bucket"),
            ("k_intent",  "a1.key_intent",  "a2.key_intent"),
        ]:
            # Extract nested keys
            a_val = obs.get("a1", {}).get(key_a.split(".")[-1], "")
            b_val = obs.get("a2", {}).get(key_b.split(".")[-1], "")

            if not a_val or not b_val:
                mismatch = -1  # failure
            else:
                mismatch = int(a_val != b_val)

            rows.append({
                "prompt_id": obs["prompt_id"],
                "intent_id": obs["intent_id"],
                "tool_class": tc,
                "scheme": scheme,
                "temperature": temp,
                "mismatch": mismatch,
            })

    return pd.DataFrame(rows)


def summarise_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mismatch rate per (scheme, temperature, tool_class)."""
    # Exclude mismatch == -1 (failures) from rate calculation
    valid = df[df["mismatch"] >= 0].copy()

    summary_rows: list[dict[str, Any]] = []
    for (scheme, temp, tc), grp in valid.groupby(["scheme", "temperature", "tool_class"]):
        n = len(grp)
        n_mismatch = int(grp["mismatch"].sum())
        ci = wilson_ci(n_mismatch, n)
        summary_rows.append({
            "scheme": scheme,
            "temperature": temp,
            "tool_class": tc,
            "failure_rate": ci.estimate,
            "ci_low": ci.ci_low,
            "ci_high": ci.ci_high,
            "n_mismatch": n_mismatch,
            "n_total": n,
        })

    return pd.DataFrame(summary_rows)


def decision_rule(failure_rate: float) -> str:
    """Apply the paper's decision rule to a measured failure rate."""
    if failure_rate > 0.15:
        return "STRONG (>15%): C1 is the headline result"
    elif failure_rate >= 0.05:
        return "MODERATE (5-15%): C1 present but not dominant"
    else:
        return "WEAK (<5%): C1 may not be the headline result"


def print_table_ii(df_summary: pd.DataFrame) -> None:
    """Print the Table II summary to stdout."""
    print("\n" + "=" * 80)
    print("TABLE II — Pilot Key-Scheme Divergence Results")
    print("=" * 80)
    print(f"{'Scheme':<12} {'Temp':>5} {'Tool Class':<14} {'Fail Rate':>10} "
          f"{'95% CI':>18} {'Mismatch':>8} {'Total':>6}")
    print("-" * 80)

    for _, row in df_summary.iterrows():
        ci_str = f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
        print(f"{row['scheme']:<12} {row['temperature']:>5.1f} {row['tool_class']:<14} "
              f"{row['failure_rate']:>10.4f} {ci_str:>18} "
              f"{row['n_mismatch']:>8} {row['n_total']:>6}")

    print("-" * 80)


def print_decision_rules(df_summary: pd.DataFrame) -> None:
    """Print the decision rule assessment per condition."""
    print("\n" + "=" * 80)
    print("DECISION RULE ASSESSMENT")
    print("=" * 80)
    for _, row in df_summary.iterrows():
        rule = decision_rule(row["failure_rate"])
        print(f"  {row['scheme']:<12} temp={row['temperature']:.1f}  "
              f"{row['tool_class']:<14} -> {rule}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Phase 1 pilot results")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to a specific raw JSON file (default: all in pilot/raw/)")
    args = parser.parse_args()

    if args.input:
        obs = _load_observations(Path(args.input))
    else:
        obs = _load_all_observations()

    if not obs:
        print("No observations found. Run `python -m pilot.run_pilot` first.")
        return

    df = compute_divergence_table(obs)
    df_summary = summarise_by_condition(df)
    print_table_ii(df_summary)
    print_decision_rules(df_summary)

    # Save processed results
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "pilot_raw_divergence.csv", index=False)
    df_summary.to_csv(out_dir / "pilot_summary.csv", index=False)
    print(f"\nProcessed results written to {out_dir}/")


if __name__ == "__main__":
    main()
