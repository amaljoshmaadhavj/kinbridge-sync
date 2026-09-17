"""
FMR estimation from dedicated negative trials (Phase 1b Design C).

Computes FMR(tau) using ONLY the dedicated independent negative trials,
not the exploratory action-level or intent-level comparisons.

FMR(tau) = (# negative trials with similarity >= tau) / N

where N = total number of negative trials for the temperature.

Usage:
    python -m kinbridge_math.fmr_from_negatives \
        --input results/raw_fmr/neg_trials_temp0.0_*.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pilot.wilson import wilson_ci


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FMRPoint:
    """FMR at a single threshold, from dedicated negative trials."""
    tau: float
    fmr: float
    n_false_match: int
    n_total: int
    ci_low: float
    ci_high: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "tau": self.tau,
            "fmr": self.fmr,
            "n_false_match": self.n_false_match,
            "n_total": self.n_total,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
        }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_negative_trials(path: Path) -> dict[str, Any]:
    """Load a negative trials JSON file.

    Returns the full data dict including metadata and trials.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_similarities(data: dict[str, Any]) -> tuple[float, list[float], dict[str, Any]]:
    """Extract similarities and metadata from loaded negative trials.

    Returns (temperature, similarities, metadata).

    Only the `trials` array is used (successful trials only).
    Failed attempts are in `failed_attempts` and excluded from FMR.
    The denominator is metadata["n_successful"], NOT len(trials).
    """
    metadata = data["metadata"]
    temperature = metadata["temperature"]
    trials = data["trials"]

    # All trials in the `trials` array are successful by construction
    similarities = [t["similarity"] for t in trials]

    return temperature, similarities, metadata


# ---------------------------------------------------------------------------
# FMR computation
# ---------------------------------------------------------------------------

def compute_fmr_curve(
    similarities: list[float],
    n_total: int,
    tau_min: float = 0.0,
    tau_max: float = 1.0,
    tau_step: float = 0.01,
) -> list[FMRPoint]:
    """Compute FMR at each threshold from dedicated negative trials.

    Args:
        similarities: List of similarity values from negative trials.
        n_total: Total number of negative trials (denominator).
        tau_min: Minimum threshold.
        tau_max: Maximum threshold.
        tau_step: Step size.

    Returns:
        List of FMRPoint objects, one per threshold.
    """
    thresholds = np.arange(tau_min, tau_max + tau_step / 2, tau_step)
    sim_array = np.array(similarities)

    points: list[FMRPoint] = []
    for tau in thresholds:
        n_false_match = int(np.sum(sim_array >= tau))
        fmr = n_false_match / n_total if n_total > 0 else 0.0

        ci = wilson_ci(n_false_match, n_total)

        points.append(FMRPoint(
            tau=float(tau),
            fmr=fmr,
            n_false_match=n_false_match,
            n_total=n_total,
            ci_low=ci.ci_low,
            ci_high=ci.ci_high,
        ))

    return points


# ---------------------------------------------------------------------------
# FSR from original reissue pairs (kept separate)
# ---------------------------------------------------------------------------

def load_fsr_from_reissues(
    pilot_path: Path,
) -> tuple[float, list[float], int]:
    """Load FSR data from the original pilot observations.

    Returns (temperature, similarities, n_pairs).
    Only the 19 a1<->a2 reissue pairs are used.
    """
    with open(pilot_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    obs = data["observations"]
    temperature = obs[0]["temperature"] if obs else 0.0
    similarities = [o["similarity"] for o in obs]

    return temperature, similarities, len(similarities)


def compute_fsr_curve(
    similarities: list[float],
    n_total: int,
    tau_min: float = 0.0,
    tau_max: float = 1.0,
    tau_step: float = 0.01,
) -> list[dict[str, Any]]:
    """Compute FSR at each threshold from original reissue pairs.

    FSR(tau) = (# reissue pairs with sim < tau) / N
    """
    thresholds = np.arange(tau_min, tau_max + tau_step / 2, tau_step)
    sim_array = np.array(similarities)

    points: list[dict[str, Any]] = []
    for tau in thresholds:
        n_false_sep = int(np.sum(sim_array < tau))
        fsr = n_false_sep / n_total if n_total > 0 else 0.0

        ci = wilson_ci(n_false_sep, n_total)

        points.append({
            "tau": float(tau),
            "fsr": fsr,
            "n_false_separation": n_false_sep,
            "n_total": n_total,
            "ci_low": ci.ci_low,
            "ci_high": ci.ci_high,
        })

    return points


# ---------------------------------------------------------------------------
# Combined sweep for calibration
# ---------------------------------------------------------------------------

@dataclass
class CombinedSweep:
    """Combined FMR+FSR sweep for one temperature, ready for calibration."""
    temperature: float
    fmr_points: list[FMRPoint]
    fsr_points: list[dict[str, Any]]
    n_fmr: int
    n_fsr: int
    fmr_source: str
    fsr_source: str


def build_combined_sweep(
    fmr_path: Path,
    pilot_path: Path,
) -> CombinedSweep:
    """Build a combined FMR+FSR sweep from dedicated negatives and pilot reissues.

    This is the primary data structure for Phase 1b calibration.
    """
    # Load FMR from dedicated negatives
    fmr_data = load_negative_trials(fmr_path)
    fmr_temp, fmr_sims, fmr_meta = extract_similarities(fmr_data)
    fmr_points = compute_fmr_curve(fmr_sims, len(fmr_sims))

    # Load FSR from original reissues
    fsr_temp, fsr_sims, fsr_n = load_fsr_from_reissues(pilot_path)
    fsr_points = compute_fsr_curve(fsr_sims, fsr_n)

    # Verify temperatures match
    if abs(fmr_temp - fsr_temp) > 0.01:
        raise ValueError(
            f"Temperature mismatch: FMR={fmr_temp}, FSR={fsr_n}"
        )

    return CombinedSweep(
        temperature=fmr_temp,
        fmr_points=fmr_points,
        fsr_points=fsr_points,
        n_fmr=len(fmr_sims),
        n_fsr=fsr_n,
        fmr_source=str(fmr_path),
        fsr_source=str(pilot_path),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute FMR from dedicated negative trials"
    )
    parser.add_argument(
        "--input", type=str, nargs="+", required=True,
        help="Negative trial JSON files",
    )
    parser.add_argument(
        "--pilot", type=str, nargs="+", required=True,
        help="Original pilot JSON files for FSR",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/phase1b",
        help="Output directory",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fmr_path_str in args.input:
        fmr_path = Path(fmr_path_str)
        data = load_negative_trials(fmr_path)
        temp, sims, meta = extract_similarities(data)

        print(f"\nTemperature: {temp}")
        print(f"  Negative trials: {len(sims)}")
        print(f"  FMR range: [{min(sims):.4f}, {max(sims):.4f}]")

        fmr_points = compute_fmr_curve(sims, len(sims))

        # Print FMR at key thresholds
        print(f"\n  FMR at key thresholds:")
        for p in fmr_points:
            if p.tau in [0.0, 0.25, 0.5, 0.75, 1.0]:
                print(f"    tau={p.tau:.2f}: FMR={p.fmr:.4f} "
                      f"[{p.ci_low:.4f}, {p.ci_high:.4f}] "
                      f"({p.n_false_match}/{p.n_total})")

    print(f"\nOutputs written to {out_dir}/")


if __name__ == "__main__":
    main()
