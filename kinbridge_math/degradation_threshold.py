"""
Degradation-gate threshold (Phase 16G).

Implements the paper's degradation-gate formulation (Eq. 15-16):

  lambda_q * Delta_q  <  lambda_l * integral(1-F(tau)) dtau + lambda_miss * (1-F(T_max))
  =>  tau* = lambda_q * Delta_q / lambda_l

where:
  Delta_q    = measured quality gap between large and small models
  F(tau)     = CDF of the similarity distribution
  lambda_q   = cost of quality degradation
  lambda_l   = cost of latency / waiting
  lambda_miss = cost of missed action

The implementation stores all intermediate quantities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class DegradationThreshold:
    """Result of the degradation-gate calculation."""
    tau_star_degradation: float
    delta_q: float
    lambda_q: float
    lambda_l: float
    lambda_miss: float
    t_max: float
    integral_term: float
    miss_term: float
    left_hand_side: float
    right_hand_side: float
    gate_fires: bool       # True if switching to small model is warranted

    def to_dict(self) -> dict[str, Any]:
        return {
            "tau_star_degradation": self.tau_star_degradation,
            "delta_q": self.delta_q,
            "lambda_q": self.lambda_q,
            "lambda_l": self.lambda_l,
            "lambda_miss": self.lambda_miss,
            "t_max": self.t_max,
            "integral_term": self.integral_term,
            "miss_term": self.miss_term,
            "left_hand_side": self.left_hand_side,
            "right_hand_side": self.right_hand_side,
            "gate_fires": self.gate_fires,
        }


def compute_degradation_threshold(
    delta_q: float,
    similarity_values: np.ndarray,
    lambda_q: float = 1.0,
    lambda_l: float = 1.0,
    lambda_miss: float = 0.5,
    t_max: float = 1.0,
    n_bins: int = 1000,
) -> DegradationThreshold:
    """Compute the degradation-gate threshold using measured Delta-q.

    Args:
        delta_q:           Measured quality gap (large - small model).
        similarity_values: Array of observed similarity scores from the pilot.
        lambda_q:          Cost weight for quality degradation.
        lambda_l:          Cost weight for latency / waiting.
        lambda_miss:       Cost weight for missed action.
        t_max:             Maximum similarity threshold for integration.
        n_bins:            Number of bins for numerical integration.

    Returns:
        DegradationThreshold with all intermediate quantities.
    """
    sim = np.asarray(similarity_values, dtype=float)

    # Numerical integration of (1 - F(tau)) from 0 to T_max
    # Using the empirical CDF
    sorted_sim = np.sort(sim)
    n = len(sorted_sim)
    if n == 0:
        integral_term = 0.0
        miss_term = 0.0
    else:
        tau_grid = np.linspace(0.0, t_max, n_bins)
        # Empirical CDF: F(tau) = fraction of values <= tau
        ecdf = np.array([np.mean(sorted_sim <= t) for t in tau_grid])
        integrand = 1.0 - ecdf

        # Trapezoidal integration
        integral_term = float(np.trapezoid(integrand, tau_grid))

        # Miss term: (1 - F(T_max))
        miss_term = float(1.0 - np.mean(sorted_sim <= t_max))

    left = lambda_q * delta_q
    right = lambda_l * integral_term + lambda_miss * miss_term

    gate_fires = left < right

    if gate_fires and lambda_l > 0:
        tau_star_deg = left / lambda_l
    else:
        tau_star_deg = t_max  # no degradation switch warranted

    return DegradationThreshold(
        tau_star_degradation=tau_star_deg,
        delta_q=delta_q,
        lambda_q=lambda_q,
        lambda_l=lambda_l,
        lambda_miss=lambda_miss,
        t_max=t_max,
        integral_term=integral_term,
        miss_term=miss_term,
        left_hand_side=left,
        right_hand_side=right,
        gate_fires=gate_fires,
    )
