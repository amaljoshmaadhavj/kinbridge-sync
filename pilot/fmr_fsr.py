"""
FMR (False Match Rate) and FSR (False Separation Rate) calculations.

FMR(tau) = |{different-intent pairs with sim >= tau}| / |{different-intent pairs}|
FSR(tau) = |{same-intent pairs with sim < tau}|     / |{same-intent pairs}|

Both are computed over a configurable threshold sweep and optionally
per tool class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pilot.pairs import EvalPair, split_pairs


@dataclass
class FMRFSR:
    """Result for a single threshold value."""
    tau: float
    tool_class: str
    fmr: float
    fsr: float
    false_match_count: int
    different_intent_pair_count: int
    false_separation_count: int
    same_intent_pair_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "tau": self.tau,
            "tool_class": self.tool_class,
            "fmr": self.fmr,
            "fsr": self.fsr,
            "false_match_count": self.false_match_count,
            "different_intent_pair_count": self.different_intent_pair_count,
            "false_separation_count": self.false_separation_count,
            "same_intent_pair_count": self.same_intent_pair_count,
        }


def compute_fmr_fsr(
    pairs: list[EvalPair],
    tau: float,
    tool_class: str = "all",
) -> FMRFSR:
    """Compute FMR and FSR at a single threshold for one tool class.

    Args:
        pairs:        All evaluation pairs (with similarities already set).
        tau:          Threshold value.
        tool_class:   Filter to this tool class, or "all" for the full set.
    """
    if tool_class != "all":
        filtered = [p for p in pairs if p.tool_class == tool_class]
    else:
        filtered = pairs

    pos, neg = split_pairs(filtered)

    n_neg = len(neg)
    n_pos = len(pos)

    if n_neg == 0:
        fmr = 0.0
        false_match_count = 0
    else:
        false_match_count = sum(1 for p in neg if p.similarity >= tau)
        fmr = false_match_count / n_neg

    if n_pos == 0:
        fsr = 0.0
        false_separation_count = 0
    else:
        false_separation_count = sum(1 for p in pos if p.similarity < tau)
        fsr = false_separation_count / n_pos

    return FMRFSR(
        tau=tau,
        tool_class=tool_class,
        fmr=fmr,
        fsr=fsr,
        false_match_count=false_match_count,
        different_intent_pair_count=n_neg,
        false_separation_count=false_separation_count,
        same_intent_pair_count=n_pos,
    )


def threshold_sweep(
    pairs: list[EvalPair],
    tool_class: str = "all",
    tau_min: float = 0.0,
    tau_max: float = 1.0,
    tau_step: float = 0.01,
) -> list[FMRFSR]:
    """Sweep tau from tau_min to tau_max (inclusive) and return FMR/FSR at each.

    Returns a list of FMRFSR objects, one per threshold value.
    """
    thresholds = np.arange(tau_min, tau_max + tau_step / 2, tau_step)
    results: list[FMRFSR] = []
    for tau in thresholds:
        results.append(compute_fmr_fsr(pairs, float(tau), tool_class))
    return results
