"""
Fit alpha and beta from measured FMR/FSR data (Phase 16A-B).

The paper's power-law approximation:
  FSR(tau) ≈ (1 - tau)^beta
  FMR(tau) ≈ tau^alpha

Taking logs:
  log(FSR) = beta * log(1 - tau) + const
  log(FMR) = alpha * log(tau) + const

We fit these via linear regression on the non-zero, non-one observations.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats


@dataclass
class FitResult:
    """Result of a power-law log-log fit."""
    slope: float
    intercept: float
    r_squared: float
    n_observations: int
    n_excluded: int
    param_name: str   # "alpha" or "beta"

    def to_dict(self) -> dict[str, Any]:
        return {
            "slope": self.slope,
            "intercept": self.intercept,
            "r_squared": self.r_squared,
            "n_observations": self.n_observations,
            "n_excluded": self.n_excluded,
            "param_name": self.param_name,
        }


def fit_beta(
    tau_values: np.ndarray,
    fsr_values: np.ndarray,
    min_valid_points: int = 5,
) -> FitResult:
    """Fit beta from log(FSR) vs log(1 - tau).

    Excludes points where FSR <= 0 or tau >= 1 (log domain violations).

    Args:
        tau_values:       Array of threshold values.
        fsr_values:       Corresponding FSR measurements.
        min_valid_points: Minimum non-excluded points required for a fit.

    Returns:
        FitResult with slope = beta.
    """
    tau = np.asarray(tau_values, dtype=float)
    fsr = np.asarray(fsr_values, dtype=float)

    # Filter: FSR > 0 and tau < 1 (so 1-tau > 0)
    valid = (fsr > 0) & (tau < 1.0)
    n_excluded = int(np.sum(~valid))

    if np.sum(valid) < min_valid_points:
        warnings.warn(
            f"fit_beta: only {np.sum(valid)} valid points "
            f"(need {min_valid_points}). Fit may be unreliable."
        )

    log_tau = np.log(1.0 - tau[valid])
    log_fsr = np.log(fsr[valid])

    slope, intercept, r_value, _, _ = stats.linregress(log_tau, log_fsr)

    return FitResult(
        slope=slope,
        intercept=intercept,
        r_squared=r_value ** 2,
        n_observations=int(np.sum(valid)),
        n_excluded=n_excluded,
        param_name="beta",
    )


def fit_alpha(
    tau_values: np.ndarray,
    fmr_values: np.ndarray,
    min_valid_points: int = 5,
) -> FitResult:
    """Fit alpha from log(FMR) vs log(tau).

    Excludes points where FMR <= 0 or tau <= 0 (log domain violations).

    Args:
        tau_values:       Array of threshold values.
        fmr_values:       Corresponding FMR measurements.
        min_valid_points: Minimum non-excluded points required for a fit.

    Returns:
        FitResult with slope = alpha.
    """
    tau = np.asarray(tau_values, dtype=float)
    fmr = np.asarray(fmr_values, dtype=float)

    # Filter: FMR > 0 and tau > 0
    valid = (fmr > 0) & (tau > 0.0)
    n_excluded = int(np.sum(~valid))

    if np.sum(valid) < min_valid_points:
        warnings.warn(
            f"fit_alpha: only {np.sum(valid)} valid points "
            f"(need {min_valid_points}). Fit may be unreliable."
        )

    log_tau = np.log(tau[valid])
    log_fmr = np.log(fmr[valid])

    slope, intercept, r_value, _, _ = stats.linregress(log_tau, log_fmr)

    return FitResult(
        slope=slope,
        intercept=intercept,
        r_squared=r_value ** 2,
        n_observations=int(np.sum(valid)),
        n_excluded=n_excluded,
        param_name="alpha",
    )
