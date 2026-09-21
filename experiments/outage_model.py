"""Gilbert–Elliott two-state Markov outage model (Phase 3a Step 4).

Two-state Markov chain: Good (G) ↔ Bad (B).

Transition probabilities per tick:
    G → B with probability p_GB
    B → G with probability p_BG

Stationary bad-state probability:
    π_B = p_GB / (p_GB + p_BG)

Calibration (from config/outage/gilbert_elliott.yaml):
    π_B × T_sim = T_out   =>   p_GB = p_BG × T_out / (T_sim - T_out)

This module produces state sequences and outage intervals.  It does NOT
call Docker, netctl, or any network-control utility.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional

import yaml


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "outage" / "gilbert_elliott.yaml"


# ─────────────────────────────────────────────────────────────
# State representation
# ─────────────────────────────────────────────────────────────

class State(str, Enum):
    """Markov chain state."""
    GOOD = "G"
    BAD = "B"


# ─────────────────────────────────────────────────────────────
# Profile data
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Profile:
    """A named Gilbert–Elliott configuration profile."""
    name: str
    total_sim_time_s: float
    target_outage_s: float
    p_bg: float
    tick_s: float

    @property
    def pi_b(self) -> float:
        """Stationary bad-state probability: π_B = T_out / T_sim."""
        if self.total_sim_time_s == 0:
            return 0.0
        return self.target_outage_s / self.total_sim_time_s

    @property
    def p_gb(self) -> float:
        """Calibrated G→B transition probability.

        Derived from:
            π_B = p_GB / (p_GB + p_BG)
            π_B = T_out / T_sim

        Solving for p_GB:
            p_GB = p_BG × T_out / (T_sim - T_out)
        """
        denom = self.total_sim_time_s - self.target_outage_s
        if denom == 0:
            raise ValueError(
                f"target_outage_s ({self.target_outage_s}) equals "
                f"total_sim_time_s ({self.total_sim_time_s}); "
                "cannot calibrate p_GB"
            )
        return self.p_bg * self.target_outage_s / denom

    @property
    def num_ticks(self) -> int:
        """Number of simulation ticks."""
        return int(self.total_sim_time_s / self.tick_s)

    @property
    def mean_bad_duration_ticks(self) -> float:
        """Expected number of consecutive ticks in Bad state: 1/p_BG."""
        if self.p_bg == 0:
            return float("inf")
        return 1.0 / self.p_bg

    @property
    def mean_bad_duration_s(self) -> float:
        """Expected outage duration in seconds: (1/p_BG) × tick_s."""
        return self.mean_bad_duration_ticks * self.tick_s


# ─────────────────────────────────────────────────────────────
# Configuration loading
# ─────────────────────────────────────────────────────────────

def load_profiles(config_path: Optional[Path] = None) -> dict[str, Profile]:
    """Load profiles from the YAML configuration file.

    Args:
        config_path: Path to YAML config.  Defaults to
            config/outage/gilbert_elliott.yaml.

    Returns:
        Dictionary mapping profile name to Profile.
    """
    path = config_path or _CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    tick_s = float(raw.get("tick_s", 1.0))
    profiles = {}
    for name, cfg in raw.get("profiles", {}).items():
        profiles[name] = Profile(
            name=name,
            total_sim_time_s=float(cfg["total_sim_time_s"]),
            target_outage_s=float(cfg["target_outage_s"]),
            p_bg=float(cfg["p_bg"]),
            tick_s=tick_s,
        )
    return profiles


# ─────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────

def validate_profile(profile: Profile) -> None:
    """Validate a profile's parameters.

    Raises:
        ValueError: If parameters are inconsistent.
    """
    if profile.p_bg < 0 or profile.p_bg > 1:
        raise ValueError(f"p_bg must be in [0, 1], got {profile.p_bg}")
    if profile.total_sim_time_s < 0:
        raise ValueError(
            f"total_sim_time_s must be non-negative, got {profile.total_sim_time_s}"
        )
    if profile.target_outage_s < 0:
        raise ValueError(
            f"target_outage_s must be non-negative, got {profile.target_outage_s}"
        )
    if profile.target_outage_s > profile.total_sim_time_s:
        raise ValueError(
            f"target_outage_s ({profile.target_outage_s}) cannot exceed "
            f"total_sim_time_s ({profile.total_sim_time_s})"
        )
    if profile.tick_s <= 0:
        raise ValueError(f"tick_s must be positive, got {profile.tick_s}")


# ─────────────────────────────────────────────────────────────
# Outage event representation
# ─────────────────────────────────────────────────────────────

@dataclass
class OutageEvent:
    """A contiguous interval in the Bad state."""
    start_tick: int
    end_tick: int  # exclusive

    @property
    def duration_ticks(self) -> int:
        return self.end_tick - self.start_tick

    def duration_s(self, tick_s: float) -> float:
        return self.duration_ticks * tick_s


# ─────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    """Result of a Gilbert–Elliott simulation."""
    profile_name: str
    states: List[State]
    outage_events: List[OutageEvent]
    p_gb: float
    p_bg: float
    pi_b: float
    num_ticks: int
    observed_pi_b: float

    def to_dict(self) -> dict:
        return {
            "profile": self.profile_name,
            "p_gb": self.p_gb,
            "p_bg": self.p_bg,
            "pi_b_target": self.pi_b,
            "pi_b_observed": self.observed_pi_b,
            "num_ticks": self.num_ticks,
            "num_outages": len(self.outage_events),
            "total_bad_ticks": sum(e.duration_ticks for e in self.outage_events),
            "outage_events": [
                {"start": e.start_tick, "end": e.end_tick,
                 "duration_ticks": e.duration_ticks}
                for e in self.outage_events
            ],
        }


def simulate(
    profile: Profile,
    seed: int,
    num_ticks: Optional[int] = None,
    initial_state: State = State.GOOD,
) -> SimulationResult:
    """Run a Gilbert–Elliott simulation.

    Args:
        profile: Configuration profile.
        seed: Random seed for reproducibility.
        num_ticks: Override number of ticks (defaults to profile.num_ticks).
        initial_state: Starting state.

    Returns:
        SimulationResult with full state sequence and outage events.
    """
    validate_profile(profile)
    n = num_ticks if num_ticks is not None else profile.num_ticks
    if n < 0:
        raise ValueError(f"num_ticks must be non-negative, got {n}")

    rng = random.Random(seed)
    p_gb = profile.p_gb
    p_bg = profile.p_bg

    states: List[State] = []
    current = initial_state

    for _ in range(n):
        states.append(current)
        if current == State.GOOD:
            if rng.random() < p_gb:
                current = State.BAD
        else:
            if rng.random() < p_bg:
                current = State.GOOD

    # Extract outage events (contiguous Bad intervals)
    events: List[OutageEvent] = []
    in_bad = False
    start = 0
    for i, s in enumerate(states):
        if s == State.BAD and not in_bad:
            in_bad = True
            start = i
        elif s == State.GOOD and in_bad:
            in_bad = False
            events.append(OutageEvent(start_tick=start, end_tick=i))
    if in_bad:
        events.append(OutageEvent(start_tick=start, end_tick=n))

    total_bad = sum(e.duration_ticks for e in events)
    observed_pi_b = total_bad / n if n > 0 else 0.0

    return SimulationResult(
        profile_name=profile.name,
        states=states,
        outage_events=events,
        p_gb=p_gb,
        p_bg=p_bg,
        pi_b=profile.pi_b,
        num_ticks=n,
        observed_pi_b=observed_pi_b,
    )


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="outage_model",
        description="Gilbert–Elliott two-state Markov outage model",
    )
    parser.add_argument(
        "--profile", required=True,
        help="Profile name (baseline, short_outage, long_outage, no_outage)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--ticks", type=int, default=None,
        help="Number of ticks (default: from profile total_sim_time_s / tick_s)",
    )
    parser.add_argument(
        "--initial-state", choices=["G", "B"], default="G",
        help="Initial state (default: G)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the outage_model CLI."""
    args = parse_args(argv)

    profiles = load_profiles()
    if args.profile not in profiles:
        print(
            f"Error: Unknown profile '{args.profile}'. "
            f"Available: {', '.join(sorted(profiles))}",
            file=sys.stderr,
        )
        return 1

    profile = profiles[args.profile]
    initial = State.GOOD if args.initial_state == "G" else State.BAD

    try:
        result = simulate(profile, seed=args.seed, num_ticks=args.ticks,
                          initial_state=initial)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"Profile:  {result.profile_name}")
        print(f"p_GB:     {result.p_gb:.6f}")
        print(f"p_BG:     {result.p_bg:.6f}")
        print(f"π_B target:    {result.pi_b:.6f}")
        print(f"π_B observed:  {result.observed_pi_b:.6f}")
        print(f"Ticks:    {result.num_ticks}")
        print(f"Outages:  {len(result.outage_events)}")
        total_bad = sum(e.duration_ticks for e in result.outage_events)
        print(f"Bad ticks: {total_bad}/{result.num_ticks}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
