"""
Wilson 95 % confidence intervals for proportions.

center = (p_hat + z^2 / (2n)) / (1 + z^2 / n)
margin = z * sqrt( p_hat*(1-p_hat)/n + z^2/(4n^2) ) / (1 + z^2 / n)

where z = 1.96 for 95 % confidence.

Uses the actual numerator and denominator — never a blind normal
approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math


Z_95 = 1.96


@dataclass
class WilsonCI:
    """Wilson confidence interval for a binomial proportion."""
    estimate: float       # sample proportion
    ci_low: float
    ci_high: float
    numerator: int
    denominator: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimate": self.estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "numerator": self.numerator,
            "denominator": self.denominator,
        }


def wilson_ci(numerator: int, denominator: int, z: float = Z_95) -> WilsonCI:
    """Compute Wilson 95 % CI for a proportion.

    Args:
        numerator:   Number of successes (e.g. false matches).
        denominator: Number of trials   (e.g. different-intent pairs).
        z:           Z-score (default 1.96 for 95 %).

    Returns:
        WilsonCI with the point estimate and interval bounds.

    Raises:
        ValueError: If denominator is negative.
    """
    if denominator < 0:
        raise ValueError(f"denominator must be >= 0, got {denominator}")
    if denominator == 0:
        return WilsonCI(estimate=0.0, ci_low=0.0, ci_high=0.0,
                        numerator=0, denominator=0)

    p_hat = numerator / denominator
    n = denominator
    z2 = z * z

    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    margin = z * math.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n)) / denom

    return WilsonCI(
        estimate=p_hat,
        ci_low=max(0.0, center - margin),
        ci_high=min(1.0, center + margin),
        numerator=numerator,
        denominator=denominator,
    )
