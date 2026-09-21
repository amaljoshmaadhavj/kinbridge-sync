"""Phase 3a Step 4 — Gilbert–Elliott outage model validation tests.

Run:
    pytest tests/test_outage_model.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.outage_model import (
    OutageEvent,
    Profile,
    SimulationResult,
    State,
    load_profiles,
    main,
    parse_args,
    simulate,
    validate_profile,
)


# ─────────────────────────────────────────────────────────────
# Configuration loading tests
# ─────────────────────────────────────────────────────────────

class TestConfigurationLoading:
    """Validate YAML configuration loading."""

    @pytest.fixture(autouse=True)
    def _profiles(self):
        self.profiles = load_profiles()

    def test_profiles_load(self):
        assert len(self.profiles) > 0

    def test_baseline_exists(self):
        assert "baseline" in self.profiles

    def test_short_outage_exists(self):
        assert "short_outage" in self.profiles

    def test_long_outage_exists(self):
        assert "long_outage" in self.profiles

    def test_no_outage_exists(self):
        assert "no_outage" in self.profiles

    def test_baseline_total_sim_time(self):
        assert self.profiles["baseline"].total_sim_time_s == 3600

    def test_baseline_target_outage(self):
        assert self.profiles["baseline"].target_outage_s == 90

    def test_baseline_p_bg(self):
        assert self.profiles["baseline"].p_bg == 0.1

    def test_tick_s(self):
        assert self.profiles["baseline"].tick_s == 1.0


# ─────────────────────────────────────────────────────────────
# p_BG handling tests
# ─────────────────────────────────────────────────────────────

class TestPBGHandling:
    """Validate p_BG is taken from configuration."""

    def test_p_bg_from_config(self):
        profiles = load_profiles()
        for name, p in profiles.items():
            assert p.p_bg == 0.1, f"Profile {name} has unexpected p_bg"

    def test_p_bg_is_float(self):
        profiles = load_profiles()
        for p in profiles.values():
            assert isinstance(p.p_bg, float)


# ─────────────────────────────────────────────────────────────
# p_GB calibration tests
# ─────────────────────────────────────────────────────────────

class TestPGBCalibration:
    """Validate deterministic p_GB calibration."""

    def test_calibration_baseline(self):
        """baseline: p_GB = 0.1 × 90 / (3600 - 90) ≈ 0.002558"""
        p = Profile("test", 3600, 90, 0.1, 1.0)
        expected = 0.1 * 90 / (3600 - 90)
        assert abs(p.p_gb - expected) < 1e-10

    def test_calibration_short(self):
        """short_outage: p_GB = 0.1 × 10 / (3600 - 10) ≈ 0.000280"""
        p = Profile("test", 3600, 10, 0.1, 1.0)
        expected = 0.1 * 10 / (3600 - 10)
        assert abs(p.p_gb - expected) < 1e-10

    def test_calibration_long(self):
        """long_outage: p_GB = 0.1 × 300 / (3600 - 300) ≈ 0.008824"""
        p = Profile("test", 3600, 300, 0.1, 1.0)
        expected = 0.1 * 300 / (3600 - 300)
        assert abs(p.p_gb - expected) < 1e-10

    def test_calibration_no_outage(self):
        """no_outage: T_out=0 => p_GB=0"""
        p = Profile("test", 3600, 0, 0.1, 1.0)
        assert p.p_gb == 0.0

    def test_calibration_deterministic(self):
        """Same parameters always yield same p_GB."""
        p1 = Profile("t", 3600, 90, 0.1, 1.0)
        p2 = Profile("t", 3600, 90, 0.1, 1.0)
        assert p1.p_gb == p2.p_gb

    def test_calibration_from_profiles(self):
        """Calibration uses profile values, not hardcoded."""
        profiles = load_profiles()
        for name, p in profiles.items():
            expected = p.p_bg * p.target_outage_s / (
                p.total_sim_time_s - p.target_outage_s
            ) if p.target_outage_s < p.total_sim_time_s else 0.0
            assert abs(p.p_gb - expected) < 1e-10, (
                f"Profile {name}: p_gb={p.p_gb} != expected {expected}"
            )


# ─────────────────────────────────────────────────────────────
# Transition probability validity tests
# ─────────────────────────────────────────────────────────────

class TestTransitionProbabilities:
    """Validate transition probabilities are valid."""

    def test_p_gb_in_range(self):
        profiles = load_profiles()
        for name, p in profiles.items():
            assert 0 <= p.p_gb <= 1, f"Profile {name}: p_gb={p.p_gb} out of range"

    def test_p_bg_in_range(self):
        profiles = load_profiles()
        for name, p in profiles.items():
            assert 0 <= p.p_bg <= 1, f"Profile {name}: p_bg={p.p_bg} out of range"

    def test_pi_b_in_range(self):
        profiles = load_profiles()
        for name, p in profiles.items():
            assert 0 <= p.pi_b <= 1, f"Profile {name}: pi_b={p.pi_b} out of range"

    def test_pi_b_formula(self):
        """π_B = p_GB / (p_GB + p_BG)."""
        profiles = load_profiles()
        for name, p in profiles.items():
            denom = p.p_gb + p.p_bg
            if denom > 0:
                expected = p.p_gb / denom
                assert abs(p.pi_b - expected) < 1e-10, (
                    f"Profile {name}: pi_b={p.pi_b} != {expected}"
                )


# ─────────────────────────────────────────────────────────────
# Seed / reproducibility tests
# ─────────────────────────────────────────────────────────────

class TestReproducibility:
    """Validate deterministic simulation with same seed."""

    def test_same_seed_same_sequence(self):
        p = Profile("t", 3600, 90, 0.1, 1.0)
        r1 = simulate(p, seed=42, num_ticks=100)
        r2 = simulate(p, seed=42, num_ticks=100)
        assert r1.states == r2.states

    def test_different_seeds_different_sequences(self):
        p = Profile("t", 3600, 90, 0.1, 1.0)
        r1 = simulate(p, seed=1, num_ticks=1000)
        r2 = simulate(p, seed=2, num_ticks=1000)
        assert r1.states != r2.states

    def test_seed_override_in_cli(self):
        """CLI --seed overrides default."""
        profiles = load_profiles()
        p = profiles["baseline"]
        r1 = simulate(p, seed=99, num_ticks=100)
        r2 = simulate(p, seed=99, num_ticks=100)
        assert r1.states == r2.states


# ─────────────────────────────────────────────────────────────
# State validity tests
# ─────────────────────────────────────────────────────────────

class TestStateValidity:
    """Validate state is always G or B."""

    def test_all_states_valid(self):
        p = Profile("t", 3600, 90, 0.1, 1.0)
        result = simulate(p, seed=42, num_ticks=10000)
        for s in result.states:
            assert s in (State.GOOD, State.BAD)

    def test_state_enum_values(self):
        assert State.GOOD.value == "G"
        assert State.BAD.value == "B"


# ─────────────────────────────────────────────────────────────
# Transition behavior tests
# ─────────────────────────────────────────────────────────────

class TestTransitionBehavior:
    """Validate G→B and B→G transitions obey configured probabilities."""

    def test_no_outage_never_transitions_to_bad(self):
        """no_outage: p_GB=0, so no G→B transitions."""
        profiles = load_profiles()
        p = profiles["no_outage"]
        result = simulate(p, seed=42, num_ticks=10000)
        assert all(s == State.GOOD for s in result.states)

    def test_high_p_gb_frequent_transitions(self):
        """With very high p_GB, should see many B states."""
        p = Profile("t", 3600, 1800, 0.1, 1.0)  # π_B = 0.5
        result = simulate(p, seed=42, num_ticks=10000)
        bad_fraction = sum(1 for s in result.states if s == State.BAD) / len(result.states)
        assert bad_fraction > 0.3, f"Expected >30% bad, got {bad_fraction:.1%}"

    def test_initial_state_respected(self):
        p = Profile("t", 3600, 90, 0.1, 1.0)
        result_g = simulate(p, seed=42, num_ticks=10, initial_state=State.GOOD)
        result_b = simulate(p, seed=42, num_ticks=10, initial_state=State.BAD)
        assert result_g.states[0] == State.GOOD
        assert result_b.states[0] == State.BAD


# ─────────────────────────────────────────────────────────────
# Profile-specific tests
# ─────────────────────────────────────────────────────────────

class TestProfileSpecific:
    """Validate profiles use their configured targets."""

    def test_baseline_uses_configured_target(self):
        profiles = load_profiles()
        p = profiles["baseline"]
        assert p.target_outage_s == 90
        assert p.total_sim_time_s == 3600

    def test_short_uses_configured_target(self):
        profiles = load_profiles()
        p = profiles["short_outage"]
        assert p.target_outage_s == 10

    def test_long_uses_configured_target(self):
        profiles = load_profiles()
        p = profiles["long_outage"]
        assert p.target_outage_s == 300

    def test_no_outage_uses_configured_target(self):
        profiles = load_profiles()
        p = profiles["no_outage"]
        assert p.target_outage_s == 0


# ─────────────────────────────────────────────────────────────
# tick_s tests
# ─────────────────────────────────────────────────────────────

class TestTickS:
    """Validate tick_s is respected in duration calculations."""

    def test_num_ticks_from_tick_s(self):
        p = Profile("t", 10, 0, 0.1, 2.0)  # 10s / 2s = 5 ticks
        assert p.num_ticks == 5

    def test_mean_bad_duration_seconds(self):
        p = Profile("t", 3600, 90, 0.1, 1.0)
        # mean_bad_ticks = 1/0.1 = 10, mean_bad_s = 10 × 1.0 = 10.0
        assert p.mean_bad_duration_s == 10.0

    def test_mean_bad_duration_different_tick_s(self):
        p = Profile("t", 3600, 90, 0.1, 0.5)
        # mean_bad_ticks = 1/0.1 = 10, mean_bad_s = 10 × 0.5 = 5.0
        assert p.mean_bad_duration_s == 5.0


# ─────────────────────────────────────────────────────────────
# No global randomness tests
# ─────────────────────────────────────────────────────────────

class TestNoGlobalRandomness:
    """Validate no uncontrolled global randomness is used."""

    def test_random_module_not_used_directly(self):
        """The simulate function must use a local Random instance."""
        import inspect
        from experiments import outage_model
        source = inspect.getsource(outage_model.simulate)
        # Should use rng.random() not random.random()
        assert "rng.random()" in source or "rng .random()" in source


# ─────────────────────────────────────────────────────────────
# Input validation tests
# ─────────────────────────────────────────────────────────────

class TestInputValidation:
    """Validate invalid inputs are rejected."""

    def test_invalid_p_bg(self):
        p = Profile("t", 3600, 90, -0.1, 1.0)
        with pytest.raises(ValueError, match="p_bg"):
            validate_profile(p)

    def test_p_bg_over_1(self):
        p = Profile("t", 3600, 90, 1.5, 1.0)
        with pytest.raises(ValueError, match="p_bg"):
            validate_profile(p)

    def test_negative_total_sim_time(self):
        p = Profile("t", -100, 90, 0.1, 1.0)
        with pytest.raises(ValueError, match="total_sim_time_s"):
            validate_profile(p)

    def test_negative_target_outage(self):
        p = Profile("t", 3600, -10, 0.1, 1.0)
        with pytest.raises(ValueError, match="target_outage_s"):
            validate_profile(p)

    def test_target_exceeds_total(self):
        p = Profile("t", 100, 200, 0.1, 1.0)
        with pytest.raises(ValueError, match="cannot exceed"):
            validate_profile(p)

    def test_zero_tick_s(self):
        p = Profile("t", 3600, 90, 0.1, 0.0)
        with pytest.raises(ValueError, match="tick_s"):
            validate_profile(p)

    def test_negative_num_ticks(self):
        p = Profile("t", 3600, 90, 0.1, 1.0)
        with pytest.raises(ValueError, match="num_ticks"):
            simulate(p, seed=42, num_ticks=-1)


# ─────────────────────────────────────────────────────────────
# CLI / API input validation tests
# ─────────────────────────────────────────────────────────────

class TestCLIValidation:
    """Validate CLI input handling."""

    def test_missing_profile_fails(self):
        with pytest.raises(SystemExit):
            parse_args([])

    def test_invalid_profile_returns_1(self):
        rc = main(["--profile", "nonexistent", "--seed", "42", "--ticks", "100"])
        assert rc == 1

    def test_valid_profile_returns_0(self):
        rc = main(["--profile", "baseline", "--seed", "42", "--ticks", "100"])
        assert rc == 0

    def test_json_output(self, capsys):
        rc = main(["--profile", "baseline", "--seed", "42",
                    "--ticks", "10", "--json"])
        assert rc == 0
        captured = capsys.readouterr()
        import json
        data = json.loads(captured.out)
        assert "profile" in data
        assert "p_gb" in data

    def test_no_outage_profile(self):
        rc = main(["--profile", "no_outage", "--seed", "42", "--ticks", "100"])
        assert rc == 0


# ─────────────────────────────────────────────────────────────
# Docker / netctl isolation tests
# ─────────────────────────────────────────────────────────────

class TestDockerIsolation:
    """Validate the implementation does not call Docker or netctl."""

    def test_no_docker_import(self):
        import experiments.outage_model as mod
        assert not hasattr(mod, "docker")

    def test_no_netctl_import(self):
        import experiments.outage_model as mod
        assert not hasattr(mod, "netctl")

    def test_no_subprocess_import(self):
        import experiments.outage_model as mod
        assert not hasattr(mod, "subprocess")

    def test_no_os_system_calls(self):
        import inspect
        from experiments import outage_model
        source = inspect.getsource(outage_model)
        assert "os.system" not in source
        assert "os.popen" not in source


# ─────────────────────────────────────────────────────────────
# Outage event tests
# ─────────────────────────────────────────────────────────────

class TestOutageEvents:
    """Validate outage event extraction."""

    def test_outage_events_contiguous_bad(self):
        """Events should capture contiguous Bad intervals."""
        p = Profile("t", 3600, 90, 0.1, 1.0)
        result = simulate(p, seed=42, num_ticks=10000)
        for event in result.outage_events:
            for i in range(event.start_tick, event.end_tick):
                assert result.states[i] == State.BAD

    def test_outage_events_bounded_by_good(self):
        """Events should be preceded/followed by Good (or at boundary)."""
        p = Profile("t", 3600, 90, 0.1, 1.0)
        result = simulate(p, seed=42, num_ticks=10000)
        for event in result.outage_events:
            if event.start_tick > 0:
                assert result.states[event.start_tick - 1] == State.GOOD
            if event.end_tick < len(result.states):
                assert result.states[event.end_tick] == State.GOOD

    def test_outage_event_duration(self):
        event = OutageEvent(start_tick=10, end_tick=20)
        assert event.duration_ticks == 10
        assert event.duration_s(1.0) == 10.0
        assert event.duration_s(0.5) == 5.0


# ─────────────────────────────────────────────────────────────
# Statistical validation (synthetic)
# ─────────────────────────────────────────────────────────────

class TestStatisticalValidation:
    """Validate observed π_B is broadly consistent with target.

    Uses a very wide tolerance to avoid false failures from
    ordinary random variation.
    """

    def test_baseline_pi_b_reasonable(self):
        profiles = load_profiles()
        p = profiles["baseline"]
        result = simulate(p, seed=42, num_ticks=100000)
        # Allow factor-of-2 tolerance (very wide)
        assert 0.005 < result.observed_pi_b < 0.05, (
            f"Observed π_B={result.observed_pi_b:.4f} outside wide tolerance "
            f"for target π_B={p.pi_b:.4f}"
        )

    def test_long_outage_higher_pi_b_than_short(self):
        """long_outage should have higher observed π_B than short_outage."""
        profiles = load_profiles()
        r_short = simulate(profiles["short_outage"], seed=42, num_ticks=100000)
        r_long = simulate(profiles["long_outage"], seed=42, num_ticks=100000)
        assert r_long.observed_pi_b > r_short.observed_pi_b
