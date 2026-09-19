"""
Phase 1b calibration pipeline -- Algorithm 2 (tau-star) from pilot data.

Corrected formulation (2026-09-18):
  FSR(tau) ≈ tau^beta           [increasing in tau]
  FMR(tau) ≈ (1 - tau)^alpha    [decreasing in tau]

  R(tau) = w_d * tau^beta + w_m * (1-tau)^alpha + w_s * P_stale

Design C data sources:
  - FSR: 19 original same-observation a1<->a2 reissue pairs from pilot
  - FMR: dedicated independent negative trials (separate experiment)

Usage:
    python -m kinbridge_math.calibrate \
        --input pilot/raw/pilot_temp0.0_20260916T160848Z.json \
               pilot/raw/pilot_temp0.7_20260916T162421Z.json \
        --fmr-input results/raw_fmr/neg_trials_temp0.0_*.json \
                     results/raw_fmr/neg_trials_temp0.7_*.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pilot.wilson import wilson_ci
from kinbridge_math.fit_alpha_beta import FitResult, fit_alpha, fit_beta
from kinbridge_math.tau_star import TauStarResult, compute_tau_star, risk_function
from kinbridge_math.fmr_from_negatives import (
    load_negative_trials,
    extract_similarities,
    compute_fmr_curve,
    FMRPoint,
    load_fsr_from_reissues,
    compute_fsr_curve,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TemperatureResult:
    """Complete calibration results for one temperature condition."""
    temperature: float
    pilot_source: str
    fmr_source: str
    n_fsr_pairs: int          # 19 reissue pairs
    n_fmr_trials: int         # 200 negative trials
    fmr_points: list[FMRPoint]
    fsr_points: list[dict[str, Any]]
    fit_alpha: FitResult | None
    fit_beta: FitResult | None
    tau_star_result: TauStarResult | None
    cannot_estimate_alpha: bool
    cannot_estimate_beta: bool
    cannot_estimate_tau_star: bool
    limitation_reason: str


# ---------------------------------------------------------------------------
# FSR from original reissue pairs
# ---------------------------------------------------------------------------

def load_fsr(
    pilot_path: Path,
) -> tuple[float, list[float], int]:
    """Load FSR data from original pilot observations.

    Returns (temperature, similarities, n_pairs).
    Only the 19 a1<->a2 reissue pairs are used.
    """
    return load_fsr_from_reissues(pilot_path)


# ---------------------------------------------------------------------------
# FMR from dedicated negative trials
# ---------------------------------------------------------------------------

def load_fmr(
    fmr_path: Path,
) -> tuple[float, list[float], dict[str, Any]]:
    """Load FMR data from dedicated negative trials.

    Returns (temperature, similarities, metadata).
    """
    data = load_negative_trials(fmr_path)
    return extract_similarities(data)


# ---------------------------------------------------------------------------
# Alpha / beta fitting
# ---------------------------------------------------------------------------

def fit_params_design_c(
    fmr_points: list[FMRPoint],
    fsr_points: list[dict[str, Any]],
) -> tuple[FitResult | None, FitResult | None, str]:
    """Fit alpha and beta from separate FMR and FSR data sources.

    Returns (fit_alpha_result, fit_beta_result, limitation_reason).
    """
    # FMR data from dedicated negatives
    fmr_tau = np.array([p.tau for p in fmr_points])
    fmr_values = np.array([p.fmr for p in fmr_points])
    n_fmr = fmr_points[0].n_total if fmr_points else 0

    # FSR data from original reissues
    fsr_tau = np.array([p["tau"] for p in fsr_points])
    fsr_values = np.array([p["fsr"] for p in fsr_points])
    n_fsr = fsr_points[0]["n_total"] if fsr_points else 0

    fit_a = None
    fit_b = None
    reasons = []

    # Fit beta from FSR (19 reissue pairs)
    if n_fsr == 0:
        reasons.append("beta cannot be fitted: no FSR data (no reissue pairs)")
    else:
        try:
            fit_b = fit_beta(fsr_tau, fsr_values)
        except Exception as exc:
            reasons.append(f"beta fitting failed: {exc}")

    # Fit alpha from FMR (200 dedicated negative trials)
    if n_fmr == 0:
        reasons.append("alpha cannot be fitted: no FMR data (no negative trials)")
    else:
        try:
            fit_a = fit_alpha(fmr_tau, fmr_values)
        except Exception as exc:
            reasons.append(f"alpha fitting failed: {exc}")

    reason = "; ".join(reasons) if reasons else ""
    return fit_a, fit_b, reason


# ---------------------------------------------------------------------------
# Tau-star / Algorithm 2
# ---------------------------------------------------------------------------

def calibrate_tau_star(
    fit_a: FitResult | None,
    fit_b: FitResult | None,
    fmr_points: list[FMRPoint],
    fsr_points: list[dict[str, Any]],
    w_d: float = 4.0,
    w_m: float = 1.0,
) -> TauStarResult | None:
    """Calibrate tau-star via Algorithm 2 if both fits are available."""
    if fit_a is None or fit_b is None:
        return None

    # Use FMR curve from dedicated negatives for tau-star computation
    tau_values = np.array([p.tau for p in fmr_points])
    fmr_values = np.array([p.fmr for p in fmr_points])
    # FSR values at the same tau grid (from reissues)
    fsr_tau = np.array([p["tau"] for p in fsr_points])
    fsr_values = np.array([p["fsr"] for p in fsr_points])

    # Interpolate FSR at FMR tau points if needed
    if not np.allclose(tau_values, fsr_tau):
        fsr_interp = np.interp(tau_values, fsr_tau, fsr_values)
    else:
        fsr_interp = fsr_values

    return compute_tau_star(
        tau_values, fmr_values, fsr_interp,
        w_d=w_d, w_m=w_m,
    )


# ---------------------------------------------------------------------------
# Per-temperature analysis
# ---------------------------------------------------------------------------

def analyze_temperature_design_c(
    pilot_path: Path,
    fmr_path: Path,
    w_d: float = 4.0,
    w_m: float = 1.0,
) -> TemperatureResult:
    """Run the full Design C calibration pipeline for one temperature."""
    # Load FSR from original reissues
    fsr_temp, fsr_sims, fsr_n = load_fsr(pilot_path)
    fsr_points = compute_fsr_curve(fsr_sims, fsr_n)

    # Load FMR from dedicated negatives
    fmr_temp, fmr_sims, fmr_meta = load_fmr(fmr_path)
    fmr_points = compute_fmr_curve(fmr_sims, len(fmr_sims))

    # Verify temperatures match
    if abs(fsr_temp - fmr_temp) > 0.01:
        raise ValueError(
            f"Temperature mismatch: pilot={fsr_temp}, negatives={fmr_temp}"
        )

    # Fit alpha and beta
    fit_a, fit_b, limitation_reason = fit_params_design_c(fmr_points, fsr_points)

    cannot_alpha = fit_a is None or fit_a.n_observations == 0
    cannot_beta = fit_b is None or fit_b.n_observations == 0
    cannot_tau_star = (
        fit_a is None or fit_b is None
        or fit_a.n_observations == 0 or fit_b.n_observations == 0
        or not np.isfinite(fit_a.slope) or not np.isfinite(fit_b.slope)
    )

    # Tau-star calibration
    tau_result = calibrate_tau_star(fit_a, fit_b, fmr_points, fsr_points, w_d, w_m)

    return TemperatureResult(
        temperature=fsr_temp,
        pilot_source=str(pilot_path),
        fmr_source=str(fmr_path),
        n_fsr_pairs=fsr_n,
        n_fmr_trials=len(fmr_sims),
        fmr_points=fmr_points,
        fsr_points=fsr_points,
        fit_alpha=fit_a,
        fit_beta=fit_b,
        tau_star_result=tau_result,
        cannot_estimate_alpha=cannot_alpha,
        cannot_estimate_beta=cannot_beta,
        cannot_estimate_tau_star=cannot_tau_star,
        limitation_reason=limitation_reason,
    )


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def write_outputs(
    results: list[TemperatureResult],
    out_dir: Path,
    w_d: float,
    w_m: float,
) -> None:
    """Write all Phase 1b output files."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- calibration_summary.json ---
    summary: dict[str, Any] = {
        "metadata": {
            "design": "C_dedicated_independent_negatives",
            "weights": {"w_d": w_d, "w_m": w_m, "w_s": 0.0, "p_stale": 0.0},
            "tau_sweep": {"min": 0.0, "max": 1.0, "step": 0.01, "n_values": 101},
            "fsr_source": "19 original a1<->a2 reissue pairs from pilot",
            "fmr_source": "200 dedicated independent negative trials per temperature",
        },
        "temperatures": [],
    }
    for r in results:
        temp_entry: dict[str, Any] = {
            "temperature": r.temperature,
            "pilot_source": r.pilot_source,
            "fmr_source": r.fmr_source,
            "n_fsr_pairs": r.n_fsr_pairs,
            "n_fmr_trials": r.n_fmr_trials,
            "cannot_estimate_alpha": r.cannot_estimate_alpha,
            "cannot_estimate_beta": r.cannot_estimate_beta,
            "cannot_estimate_tau_star": r.cannot_estimate_tau_star,
            "limitation_reason": r.limitation_reason,
        }
        if r.fit_beta is not None:
            temp_entry["fit_beta"] = r.fit_beta.to_dict()
        if r.fit_alpha is not None:
            temp_entry["fit_alpha"] = r.fit_alpha.to_dict()
        if r.tau_star_result is not None:
            temp_entry["tau_star"] = r.tau_star_result.to_dict()
        summary["temperatures"].append(temp_entry)

    with open(out_dir / "calibration_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # --- fmr_sweep.csv (from dedicated negatives) ---
    fmr_rows: list[dict[str, Any]] = []
    for r in results:
        for p in r.fmr_points:
            fmr_rows.append({
                "temperature": r.temperature,
                "tau": p.tau,
                "fmr": p.fmr,
                "n_false_match": p.n_false_match,
                "n_total": p.n_total,
                "ci_low": p.ci_low,
                "ci_high": p.ci_high,
                "source": "dedicated_negatives",
            })
    pd.DataFrame(fmr_rows).to_csv(out_dir / "fmr_sweep.csv", index=False)

    # --- fsr_sweep.csv (from original reissues) ---
    fsr_rows: list[dict[str, Any]] = []
    for r in results:
        for p in r.fsr_points:
            fsr_rows.append({
                "temperature": r.temperature,
                "tau": p["tau"],
                "fsr": p["fsr"],
                "n_false_separation": p["n_false_separation"],
                "n_total": p["n_total"],
                "ci_low": p["ci_low"],
                "ci_high": p["ci_high"],
                "source": "original_reissues",
            })
    pd.DataFrame(fsr_rows).to_csv(out_dir / "fsr_sweep.csv", index=False)

    # --- alpha_beta_fits.csv ---
    fit_rows: list[dict[str, Any]] = []
    for r in results:
        if r.fit_beta is not None:
            fit_rows.append({
                "temperature": r.temperature,
                "param": "beta",
                "source": "fsr_from_reissues",
                "slope": r.fit_beta.slope,
                "intercept": r.fit_beta.intercept,
                "r_squared": r.fit_beta.r_squared,
                "n_observations": r.fit_beta.n_observations,
                "n_excluded": r.fit_beta.n_excluded,
            })
        if r.fit_alpha is not None:
            fit_rows.append({
                "temperature": r.temperature,
                "param": "alpha",
                "source": "fmr_from_dedicated_negatives",
                "slope": r.fit_alpha.slope,
                "intercept": r.fit_alpha.intercept,
                "r_squared": r.fit_alpha.r_squared,
                "n_observations": r.fit_alpha.n_observations,
                "n_excluded": r.fit_alpha.n_excluded,
            })
    if fit_rows:
        pd.DataFrame(fit_rows).to_csv(out_dir / "alpha_beta_fits.csv", index=False)

    # --- tau_star_results.csv ---
    ts_rows: list[dict[str, Any]] = []
    for r in results:
        if r.tau_star_result is not None:
            ts = r.tau_star_result
            ts_rows.append({
                "temperature": r.temperature,
                "tau_star": ts.tau_star,
                "alpha": ts.alpha,
                "beta": ts.beta,
                "method_used": ts.method_used,
                "risk_at_tau_star": ts.risk_at_tau_star,
                "w_d": ts.w_d,
                "w_m": ts.w_m,
            })
    if ts_rows:
        pd.DataFrame(ts_rows).to_csv(out_dir / "tau_star_results.csv", index=False)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1b Design C calibration: FSR from reissues + FMR from dedicated negatives"
    )
    parser.add_argument(
        "--input", type=str, nargs="+", required=True,
        help="Original pilot JSON files (for FSR from reissue pairs)",
    )
    parser.add_argument(
        "--fmr-input", type=str, nargs="+", required=True,
        help="Dedicated negative trial JSON files (for FMR)",
    )
    parser.add_argument(
        "--w-d", type=float, default=4.0,
        help="Cost weight for false separation (default: 4.0)",
    )
    parser.add_argument(
        "--w-m", type=float, default=1.0,
        help="Cost weight for false match (default: 1.0)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/phase1b",
        help="Output directory (default: results/phase1b)",
    )
    args = parser.parse_args()

    pilot_paths = [Path(p) for p in args.input]
    fmr_paths = [Path(p) for p in args.fmr_input]

    # Group by temperature
    results: list[TemperatureResult] = []

    for pilot_path in pilot_paths:
        # Find matching FMR file for this temperature
        pilot_data = json.loads(pilot_path.read_text())
        pilot_temp = pilot_data["observations"][0]["temperature"]

        # Find FMR file with matching temperature
        matching_fmr = None
        for fmr_path in fmr_paths:
            fmr_data = json.loads(fmr_path.read_text())
            fmr_temp = fmr_data["metadata"]["temperature"]
            if abs(fmr_temp - pilot_temp) < 0.01:
                matching_fmr = fmr_path
                break

        if matching_fmr is None:
            print(f"WARNING: No FMR file found for temperature {pilot_temp}")
            continue

        result = analyze_temperature_design_c(
            pilot_path, matching_fmr, w_d=args.w_d, w_m=args.w_m,
        )
        results.append(result)

    # Write outputs
    out_dir = Path(args.output_dir)
    write_outputs(results, out_dir, args.w_d, args.w_m)

    # Print summary
    print("=" * 80)
    print("PHASE 1b DESIGN C CALIBRATION SUMMARY")
    print("=" * 80)
    for r in results:
        print(f"\nTemperature: {r.temperature}")
        print(f"  FSR pairs (reissues):  {r.n_fsr_pairs}")
        print(f"  FMR trials (negatives): {r.n_fmr_trials}")
        if r.fit_beta is not None:
            print(f"  Beta fit:    slope={r.fit_beta.slope:.4f}, R²={r.fit_beta.r_squared:.4f}")
        else:
            print(f"  Beta fit:    CANNOT FIT")
        if r.fit_alpha is not None:
            print(f"  Alpha fit:   slope={r.fit_alpha.slope:.4f}, R²={r.fit_alpha.r_squared:.4f}")
        else:
            print(f"  Alpha fit:   CANNOT FIT")
        if r.tau_star_result is not None:
            ts = r.tau_star_result
            print(f"  Tau-star:    {ts.tau_star:.4f} (method: {ts.method_used})")
            print(f"  Risk:        {ts.risk_at_tau_star:.6f}")
        else:
            print(f"  Tau-star:    CANNOT COMPUTE")
        if r.limitation_reason:
            print(f"  Limitation:  {r.limitation_reason}")

    print(f"\nOutputs written to {out_dir}/")


if __name__ == "__main__":
    main()
