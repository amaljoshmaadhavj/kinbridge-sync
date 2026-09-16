"""
Empirical validation of the calculated tau-star (Phase 16E).

Compares the analytically calibrated tau* against the measured FMR/FSR
tradeoff curve.  Generates data suitable for the paper's Figure 2.

No smoothing or curve fitting is applied to the measured data — any
disagreement between the fitted model and the raw measurements is
preserved and reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kinbridge_math.tau_star import TauStarResult, risk_function


@dataclass
class ValidationPoint:
    """One point on the measured FMR/FSR curve with risk overlay."""
    tau: float
    measured_fmr: float
    measured_fsr: float
    risk: float
    is_tau_star: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "tau": self.tau,
            "measured_fmr": self.measured_fmr,
            "measured_fsr": self.measured_fsr,
            "risk": self.risk,
            "is_tau_star": self.is_tau_star,
        }


def validate_tau_star(
    tau_values: np.ndarray,
    fmr_values: np.ndarray,
    fsr_values: np.ndarray,
    tau_star_result: TauStarResult,
    w_d: float | None = None,
    w_m: float | None = None,
) -> list[ValidationPoint]:
    """Build validation data by evaluating risk at each measured (tau, FMR, FSR).

    The is_tau_star flag marks the point closest to the calculated tau*.
    """
    w_d = w_d if w_d is not None else tau_star_result.w_d
    w_m = w_m if w_m is not None else tau_star_result.w_m

    points: list[ValidationPoint] = []
    tau_star = tau_star_result.tau_star
    min_dist = float("inf")
    closest_idx = 0

    for i, (t, fmr, fsr) in enumerate(zip(tau_values, fmr_values, fsr_values)):
        r = risk_function(t, fmr, fsr, w_d, w_m)
        dist = abs(t - tau_star)
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
        points.append(ValidationPoint(
            tau=float(t),
            measured_fmr=float(fmr),
            measured_fsr=float(fsr),
            risk=r,
            is_tau_star=False,
        ))

    if points:
        points[closest_idx].is_tau_star = True

    return points


def validation_summary(points: list[ValidationPoint], tau_star: float) -> dict[str, Any]:
    """Summarise validation: best measured point, risk at tau*, diagnostics."""
    ts_point = next((p for p in points if p.is_tau_star), None)

    risk_values = [p.risk for p in points]
    min_risk_idx = int(np.argmin(risk_values))

    return {
        "tau_star_calculated": tau_star,
        "tau_star_closest_measured": points[min_risk_idx].tau if points else None,
        "risk_at_calculated_tau_star": ts_point.risk if ts_point else None,
        "min_measured_risk": risk_values[min_risk_idx] if risk_values else None,
        "min_measured_risk_tau": points[min_risk_idx].tau if points else None,
        "n_points": len(points),
    }
