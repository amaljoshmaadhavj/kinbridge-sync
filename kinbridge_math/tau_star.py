"""
Risk function R(tau) and tau-star calibration (Phase 16C-D).

R(tau) = w_d * FSR(tau) + w_m * FMR(tau) + w_s * P_stale

Algorithm 2:
  1. Fit beta from log(FSR) vs log(1-tau)
  2. Fit alpha from log(FMR) vs log(tau)
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
    method_used: str          # "closed_form" or "bisection"
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

    R'(tau) = -w_d * beta * (1-tau)^(beta-1) + w_m * alpha * tau^(alpha-1)

    This is the derivative used for bisection.
    """
    eps = 1e-15
    tau_c = max(eps, min(1.0 - eps, tau))
    term1 = -w_d * beta * (1.0 - tau_c) ** (beta - 1.0)
    term2 = w_m * alpha * tau_c ** (alpha - 1.0)
    return term1 + term2


def _closed_form_tau_star(alpha: float, beta: float, w_d: float, w_m: float) -> float:
    """Closed-form tau* when alpha ≈ beta (paper Eq. 14).

    tau* = (1 + (w_d * beta / (w_m * alpha))^(1/(alpha-1)))^(-1)
           * (w_d * beta / (w_m * alpha))^(1/(alpha-1))

    Simplifies when alpha == beta to:
    ratio = w_d / w_m
    tau* = ratio / (1 + ratio)
    """
    eps = 1e-15
    a = max(alpha, eps)
    ratio = (w_d * beta) / (w_m * a)
    exponent = 1.0 / (a - 1.0) if abs(a - 1.0) > eps else 10.0
    r_exp = ratio ** exponent
    return r_exp / (1.0 + r_exp)


def _bisection_tau_star(
    alpha: float,
    beta: float,
    w_d: float,
    w_m: float,
    lo: float = 0.001,
    hi: float = 0.999,
    n_iter: int = 40,
) -> float:
    """Bisection search for the root of R'(tau) = 0."""
    for _ in range(n_iter):
        mid = (lo + hi) / 2.0
        if risk_derivative_from_params(mid, alpha, beta, w_d, w_m) < 0:
            lo = mid
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

    if abs(alpha - beta) < epsilon:
        tau_star = _closed_form_tau_star(alpha, beta, w_d, w_m)
        method = "closed_form"
    else:
        tau_star = _bisection_tau_star(alpha, beta, w_d, w_m)
        method = "bisection"

    # Evaluate risk at tau_star using the fitted curves
    fmr_star = max(0.0, tau_star ** alpha) * math.exp(fit_a.intercept)
    fsr_star = max(0.0, (1.0 - tau_star) ** beta) * math.exp(fit_b.intercept)
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
