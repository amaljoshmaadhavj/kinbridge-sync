"""
Risk function R(tau) and tau-star calibration (Phase 16C-D).

Corrected formulation (2026-09-18):
  R(tau) = w_d * FSR(tau) + w_m * FMR(tau) + w_s * P_stale

  FSR(tau) ≈ tau^beta           [increasing in tau]
  FMR(tau) ≈ (1 - tau)^alpha    [decreasing in tau]

  R(tau) = w_d * tau^beta + w_m * (1-tau)^alpha + w_s * P_stale

  R'(tau) = w_d * beta * tau^(beta-1) - w_m * alpha * (1-tau)^(alpha-1)

Algorithm 2:
  1. Fit beta from log(FSR) vs log(tau)
  2. Fit alpha from log(FMR) vs log(1-tau)
  3. If |alpha - beta| < 0.05: use closed-form (alpha == beta case)
  4. Else: bisect R'(tau) = 0 over [0.001, 0.999]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kinbridge_math.fit_alpha_beta import FitResult, fit_alpha, fit_beta


@dataclass
class TauStarResult:
    """Result of Algorithm 2."""
    tau_star: float
    alpha: float
    beta: float
    method_used: str          # "closed_form", "bisection", or "unestimable"
    risk_at_tau_star: float
    fit_alpha: FitResult
    fit_beta: FitResult
    w_d: float
    w_m: float
    w_s: float
    p_stale: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "tau_star": self.tau_star,
            "alpha": self.alpha,
            "beta": self.beta,
            "method_used": self.method_used,
            "risk_at_tau_star": self.risk_at_tau_star,
            "w_d": self.w_d,
            "w_m": self.w_m,
            "w_s": self.w_s,
            "p_stale": self.p_stale,
            "fit_alpha": self.fit_alpha.to_dict(),
            "fit_beta": self.fit_beta.to_dict(),
        }


def risk_function(
    tau: float,
    fmr_at_tau: float,
    fsr_at_tau: float,
    w_d: float = 4.0,
    w_m: float = 1.0,
    w_s: float = 0.0,
    p_stale: float = 0.0,
) -> float:
    """Compute R(tau) = w_d * FSR + w_m * FMR + w_s * P_stale."""
    return w_d * fsr_at_tau + w_m * fmr_at_tau + w_s * p_stale


def risk_derivative_from_params(
    tau: float,
    alpha: float,
    beta: float,
    w_d: float = 4.0,
    w_m: float = 1.0,
) -> float:
    """Compute R'(tau) using the fitted power-law parameters.

    R'(tau) = w_d * beta * tau^(beta-1) - w_m * alpha * (1-tau)^(alpha-1)

    This is the derivative used for bisection.
    """
    eps = 1e-15
    tau_c = max(eps, min(1.0 - eps, tau))
    term1 = w_d * beta * tau_c ** (beta - 1.0)
    term2 = -w_m * alpha * (1.0 - tau_c) ** (alpha - 1.0)
    return term1 + term2


def _closed_form_tau_star(alpha: float, beta: float, w_d: float, w_m: float) -> float:
    """Closed-form tau* when alpha ≈ beta.

    From R'(tau) = 0 with alpha = beta:
      w_d * tau^(alpha-1) = w_m * (1-tau)^(alpha-1)
      tau/(1-tau) = (w_m/w_d)^(1/(alpha-1))
      tau* = 1/(1+rho) where rho = (w_d/w_m)^(1/(alpha-1))
    """
    eps = 1e-15
    a = max(alpha, eps)
    ratio = w_d / w_m
    exponent = 1.0 / (a - 1.0) if abs(a - 1.0) > eps else 10.0
    rho = ratio ** exponent
    return 1.0 / (1.0 + rho)


def _validate_params(
    alpha: float,
    beta: float,
    w_d: float,
    w_m: float,
) -> tuple[bool, str]:
    """Validate fitted parameters before attempting tau-star computation.

    Returns (is_valid, reason_if_invalid).
    """
    if not math.isfinite(alpha):
        return False, f"alpha is not finite: {alpha}"
    if not math.isfinite(beta):
        return False, f"beta is not finite: {beta}"
    if alpha <= 0:
        return False, f"alpha must be positive: {alpha}"
    if beta <= 0:
        return False, f"beta must be positive: {beta}"
    if w_d <= 0 or w_m <= 0:
        return False, f"weights must be positive: w_d={w_d}, w_m={w_m}"
    return True, ""


def _bisection_tau_star(
    alpha: float,
    beta: float,
    w_d: float,
    w_m: float,
    lo: float = 0.001,
    hi: float = 0.999,
    n_iter: int = 40,
) -> float | None:
    """Bisection search for the root of R'(tau) = 0.

    Returns None if no valid sign change exists (derivative does not
    cross zero in the search interval).
    """
    d_lo = risk_derivative_from_params(lo, alpha, beta, w_d, w_m)
    d_hi = risk_derivative_from_params(hi, alpha, beta, w_d, w_m)

    # Require opposite signs for bisection to converge to an interior root
    if not (math.isfinite(d_lo) and math.isfinite(d_hi)):
        return None
    if d_lo * d_hi > 0:
        # No sign change: derivative does not cross zero in [lo, hi]
        return None

    for _ in range(n_iter):
        mid = (lo + hi) / 2.0
        d_mid = risk_derivative_from_params(mid, alpha, beta, w_d, w_m)
        if not math.isfinite(d_mid):
            return None
        if d_mid * d_lo > 0:
            lo = mid
            d_lo = d_mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def compute_tau_star(
    tau_values: np.ndarray,
    fmr_values: np.ndarray,
    fsr_values: np.ndarray,
    w_d: float = 4.0,
    w_m: float = 1.0,
    w_s: float = 0.0,
    p_stale: float = 0.0,
    epsilon: float = 0.05,
) -> TauStarResult:
    """Algorithm 2: Calibrate tau* from measured FMR/FSR data.

    Args:
        tau_values: Array of threshold values from the sweep.
        fmr_values: Corresponding FMR measurements.
        fsr_values: Corresponding FSR measurements.
        w_d:        Cost weight for false separation (missed duplicate).
        w_m:        Cost weight for false match (incorrect merge).
        w_s:        Cost weight for staleness (constant, doesn't affect opt).
        p_stale:    Staleness probability (constant term).
        epsilon:    Threshold for |alpha - beta| to use closed form.

    Returns:
        TauStarResult with the calibrated threshold.
    """
    fit_a = fit_alpha(tau_values, fmr_values)
    fit_b = fit_beta(tau_values, fsr_values)

    alpha = fit_a.slope
    beta = fit_b.slope

    # Validate parameters before any optimization
    is_valid, reason = _validate_params(alpha, beta, w_d, w_m)
    if not is_valid:
        # Return unestimable result
        nan = float("nan")
        return TauStarResult(
            tau_star=nan,
            alpha=alpha,
            beta=beta,
            method_used="unestimable",
            risk_at_tau_star=nan,
            fit_alpha=fit_a,
            fit_beta=fit_b,
            w_d=w_d,
            w_m=w_m,
            w_s=w_s,
            p_stale=p_stale,
        )

    # Attempt tau-star computation
    tau_star: float | None = None
    method: str = "unestimable"

    if abs(alpha - beta) < epsilon:
        tau_star = _closed_form_tau_star(alpha, beta, w_d, w_m)
        method = "closed_form"
    else:
        tau_star = _bisection_tau_star(alpha, beta, w_d, w_m)
        if tau_star is not None:
            method = "bisection"

    if tau_star is None or not math.isfinite(tau_star):
        nan = float("nan")
        return TauStarResult(
            tau_star=nan,
            alpha=alpha,
            beta=beta,
            method_used="unestimable",
            risk_at_tau_star=nan,
            fit_alpha=fit_a,
            fit_beta=fit_b,
            w_d=w_d,
            w_m=w_m,
            w_s=w_s,
            p_stale=p_stale,
        )

    # Evaluate risk at tau_star using the corrected fitted curves
    fmr_star = max(0.0, (1.0 - tau_star) ** alpha) * math.exp(fit_a.intercept)
    fsr_star = max(0.0, tau_star ** beta) * math.exp(fit_b.intercept)
    risk = risk_function(tau_star, fmr_star, fsr_star, w_d, w_m, w_s, p_stale)

    return TauStarResult(
        tau_star=tau_star,
        alpha=alpha,
        beta=beta,
        method_used=method,
        risk_at_tau_star=risk,
        fit_alpha=fit_a,
        fit_beta=fit_b,
        w_d=w_d,
        w_m=w_m,
        w_s=w_s,
        p_stale=p_stale,
    )
