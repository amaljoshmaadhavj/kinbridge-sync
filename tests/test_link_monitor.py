"""Tests for the Link Monitor (Phase 2.2 Step 2).

Covers EWMA equations, RTO computation, hysteresis FSM,
initialisation semantics, input validation, and a synthetic trace.
"""

from __future__ import annotations

import math

import pytest

from client.link_monitor import LinkMonitor


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

_ALPHA = 0.125
_BETA = 0.25


def _ewma_rhat(r_hat_prev: float, r: float) -> float:
    """Expected r_hat after a successful sample."""
    return _ALPHA * r + (1.0 - _ALPHA) * r_hat_prev


def _ewma_dhat(d_hat_prev: float, r: float, r_hat_new: float) -> float:
    """Expected d_hat after a successful sample (uses NEW r_hat)."""
    return _BETA * abs(r - r_hat_new) + (1.0 - _BETA) * d_hat_prev


# -------------------------------------------------------------------
# 1. Initial state
# -------------------------------------------------------------------

class TestInitialState:
    def test_initial_state_is_up(self):
        lm = LinkMonitor()
        assert lm.state == "UP"

    def test_initial_r_hat_is_zero(self):
        lm = LinkMonitor()
        assert lm.r_hat_ms == 0.0

    def test_initial_d_hat_is_zero(self):
        lm = LinkMonitor()
        assert lm.d_hat_ms == 0.0

    def test_initial_rto_is_zero(self):
        lm = LinkMonitor()
        assert lm.rto_ms == 0.0

    def test_initial_consecutive_counters_are_zero(self):
        lm = LinkMonitor()
        assert lm.consecutive_timeouts == 0
        assert lm.consecutive_successes == 0


# -------------------------------------------------------------------
# 2. RTO computation
# -------------------------------------------------------------------

class TestRTOComputation:
    def test_rto_formula(self):
        lm = LinkMonitor()
        # Manually set internal state
        lm._r_hat = 100.0
        lm._d_hat = 15.0
        lm._first_sample = False
        # RTO = r_hat + 4 * d_hat = 100 + 60 = 160
        assert lm.rto_ms == 160.0

    def test_rto_after_first_sample(self):
        lm = LinkMonitor()
        lm.record_sample(80.0)
        # r_hat = 0.125 * 80 = 10.0
        # d_hat = 0.25 * |80 - 10| = 0.25 * 70 = 17.5
        # RTO = 10.0 + 4 * 17.5 = 10.0 + 70.0 = 80.0
        assert lm.r_hat_ms == pytest.approx(10.0)
        assert lm.d_hat_ms == pytest.approx(17.5)
        assert lm.rto_ms == pytest.approx(80.0)


# -------------------------------------------------------------------
# 3. EWMA update
# -------------------------------------------------------------------

class TestEWMAUpdate:
    def test_first_sample_uses_standard_equations(self):
        """First sample: r_hat = alpha * r, d_hat = beta * |r - r_hat|."""
        lm = LinkMonitor()
        lm.record_sample(100.0)
        # r_hat = 0.125 * 100 = 12.5
        # d_hat = 0.25 * |100 - 12.5| = 0.25 * 87.5 = 21.875
        assert lm.r_hat_ms == pytest.approx(12.5)
        assert lm.d_hat_ms == pytest.approx(21.875)

    def test_second_sample_ewma(self):
        lm = LinkMonitor()
        lm.record_sample(100.0)  # first sample
        lm.record_sample(100.0)  # second sample
        # r_hat = 0.125 * 100 + 0.875 * 12.5 = 12.5 + 10.9375 = 23.4375
        # d_hat = 0.25 * |100 - 23.4375| + 0.75 * 21.875
        #       = 0.25 * 76.5625 + 0.75 * 21.875
        #       = 19.140625 + 16.40625 = 35.546875
        assert lm.r_hat_ms == pytest.approx(23.4375)
        assert lm.d_hat_ms == pytest.approx(35.546875)

    def test_ewma_converges_toward_constant(self):
        """Feeding RTT=100ms repeatedly should converge r_hat toward 100."""
        lm = LinkMonitor()
        for _ in range(50):
            lm.record_sample(100.0)
        # After 50 samples of 100ms, r_hat should be close to 100
        assert lm.r_hat_ms == pytest.approx(100.0, abs=1.0)

    def test_ewma_exact_values_first_five(self):
        """Verify exact EWMA values for 5 consecutive samples of 100ms."""
        lm = LinkMonitor()
        expected_r = 0.0
        expected_d = 0.0

        for i in range(5):
            r = 100.0
            lm.record_sample(r)
            # Compute expected values
            expected_r_new = _ALPHA * r + (1.0 - _ALPHA) * expected_r
            expected_d_new = _BETA * abs(r - expected_r_new) + (1.0 - _BETA) * expected_d
            assert lm.r_hat_ms == pytest.approx(expected_r_new), f"r_hat mismatch at sample {i+1}"
            assert lm.d_hat_ms == pytest.approx(expected_d_new), f"d_hat mismatch at sample {i+1}"
            expected_r = expected_r_new
            expected_d = expected_d_new

    def test_timeout_does_not_update_ewma(self):
        lm = LinkMonitor()
        lm.record_sample(100.0)  # first sample, success
        r_hat_before = lm.r_hat_ms
        d_hat_before = lm.d_hat_ms
        lm.record_sample(None)    # timeout
        assert lm.r_hat_ms == r_hat_before
        assert lm.d_hat_ms == d_hat_before

    def test_timeout_greater_than_rto_does_not_update_ewma(self):
        lm = LinkMonitor()
        lm.record_sample(50.0)   # first sample
        r_hat_before = lm.r_hat_ms
        d_hat_before = lm.d_hat_ms
        # RTO after first sample = 50.0; feed 51.0 which is > RTO
        lm.record_sample(51.0)
        # Should be treated as timeout — EWMA unchanged
        assert lm.r_hat_ms == r_hat_before
        assert lm.d_hat_ms == d_hat_before

    def test_success_at_rto_updates_ewma(self):
        lm = LinkMonitor()
        lm.record_sample(50.0)   # first sample, RTO becomes 50.0
        r_hat_before = lm.r_hat_ms
        # Feed exactly RTO — should be success
        lm.record_sample(50.0)
        assert lm.r_hat_ms != r_hat_before  # EWMA updated


# -------------------------------------------------------------------
# 4. DOWN after timeouts
# -------------------------------------------------------------------

class TestDownAfterTimeouts:
    def test_down_after_threshold(self):
        lm = LinkMonitor()
        lm.record_sample(100.0)  # first sample (success)
        for i in range(5):
            lm.record_sample(None)
            if i < 4:
                assert lm.state == "UP", f"Should still be UP after {i+1} timeouts"
        assert lm.state == "DOWN"

    def test_down_with_timeout_threshold_from_config(self):
        lm = LinkMonitor()
        lm.record_sample(100.0)  # first sample
        # Default timeout_threshold = 5
        for _ in range(4):
            lm.record_sample(None)
        assert lm.state == "UP"
        lm.record_sample(None)
        assert lm.state == "DOWN"


# -------------------------------------------------------------------
# 5. UP after recoveries
# -------------------------------------------------------------------

class TestUpAfterRecoveries:
    def _drive_down(self, lm: LinkMonitor):
        lm.record_sample(100.0)  # first sample
        for _ in range(5):
            lm.record_sample(None)
        assert lm.state == "DOWN"

    def test_up_after_recovery_threshold(self):
        lm = LinkMonitor()
        self._drive_down(lm)
        for i in range(3):
            lm.record_sample(100.0)
            if i < 2:
                assert lm.state == "DOWN", f"Should still be DOWN after {i+1} recoveries"
        assert lm.state == "UP"

    def test_up_with_recovery_threshold_from_config(self):
        lm = LinkMonitor()
        self._drive_down(lm)
        # Default recovery_threshold = 3
        for _ in range(2):
            lm.record_sample(100.0)
        assert lm.state == "DOWN"
        lm.record_sample(100.0)
        assert lm.state == "UP"


# -------------------------------------------------------------------
# 6. Hysteresis: no flip on single timeout
# -------------------------------------------------------------------

class TestHysteresis:
    def test_single_timeout_does_not_flip_up(self):
        lm = LinkMonitor()
        lm.record_sample(100.0)  # first sample
        lm.record_sample(None)   # one timeout
        assert lm.state == "UP"
        lm.record_sample(100.0)  # one success resets timeout counter
        assert lm.state == "UP"

    def test_single_success_does_not_flip_down_to_up_immediately(self):
        lm = LinkMonitor()
        lm.record_sample(100.0)
        # Drive DOWN
        for _ in range(5):
            lm.record_sample(None)
        assert lm.state == "DOWN"
        # One success is not enough
        lm.record_sample(100.0)
        assert lm.state == "DOWN"


# -------------------------------------------------------------------
# 7. Consecutive timeout counter
# -------------------------------------------------------------------

class TestConsecutiveTimeoutCounter:
    def test_counter_increments(self):
        lm = LinkMonitor()
        lm.record_sample(100.0)
        for i in range(1, 6):
            lm.record_sample(None)
            assert lm.consecutive_timeouts == i

    def test_counter_resets_on_success(self):
        lm = LinkMonitor()
        lm.record_sample(100.0)
        lm.record_sample(None)
        lm.record_sample(None)
        assert lm.consecutive_timeouts == 2
        lm.record_sample(100.0)
        assert lm.consecutive_timeouts == 0
        assert lm.consecutive_successes == 1


# -------------------------------------------------------------------
# 8. Synthetic trace: 10 good → 6 bad → 5 good
# -------------------------------------------------------------------

class TestSyntheticTrace:
    def test_transition_up_down_up(self):
        lm = LinkMonitor()
        states = []

        # 10 good samples
        for _ in range(10):
            states.append(lm.record_sample(50.0))
        assert all(s == "UP" for s in states)

        # 6 bad samples — DOWN after 5th
        for i in range(6):
            s = lm.record_sample(None)
            if i < 4:
                assert s == "UP", f"Should be UP after {i+1} timeouts"
            elif i == 4:
                assert s == "DOWN", "Should transition to DOWN at 5th timeout"
            else:
                assert s == "DOWN"

        # 5 good samples — UP after 3rd
        for i in range(5):
            s = lm.record_sample(50.0)
            if i < 2:
                assert s == "DOWN", f"Should be DOWN after {i+1} recoveries"
            elif i == 2:
                assert s == "UP", "Should transition to UP at 3rd recovery"
            else:
                assert s == "UP"

    def test_trace_counters(self):
        lm = LinkMonitor()

        for _ in range(10):
            lm.record_sample(50.0)
        assert lm.consecutive_successes == 10
        assert lm.consecutive_timeouts == 0

        for _ in range(6):
            lm.record_sample(None)
        assert lm.consecutive_timeouts == 6
        assert lm.consecutive_successes == 0

        for _ in range(5):
            lm.record_sample(50.0)
        assert lm.consecutive_successes == 5
        assert lm.consecutive_timeouts == 0


# -------------------------------------------------------------------
# 9. None timeout
# -------------------------------------------------------------------

class TestNoneTimeout:
    def test_none_is_timeout(self):
        lm = LinkMonitor()
        lm.record_sample(100.0)
        lm.record_sample(None)
        assert lm.consecutive_timeouts == 1
        assert lm.consecutive_successes == 0

    def test_none_before_first_sample(self):
        """None before any numeric sample is a timeout (EWMA unchanged)."""
        lm = LinkMonitor()
        lm.record_sample(None)
        assert lm.state == "UP"
        assert lm.consecutive_timeouts == 1
        assert lm.r_hat_ms == 0.0
        assert lm.d_hat_ms == 0.0


# -------------------------------------------------------------------
# 10. Numeric RTT greater than RTO
# -------------------------------------------------------------------

class TestNumericRTTGreaterThanRTO:
    def test_rtt_above_rto_is_timeout(self):
        lm = LinkMonitor()
        lm.record_sample(50.0)  # RTO becomes 50.0
        lm.record_sample(50.1)  # > RTO → timeout
        assert lm.consecutive_timeouts == 1
        assert lm.consecutive_successes == 0

    def test_rtt_above_rto_does_not_update_ewma(self):
        lm = LinkMonitor()
        lm.record_sample(50.0)
        r_hat_before = lm.r_hat_ms
        d_hat_before = lm.d_hat_ms
        lm.record_sample(50.1)
        assert lm.r_hat_ms == r_hat_before
        assert lm.d_hat_ms == d_hat_before


# -------------------------------------------------------------------
# 11. Successful RTT at or below RTO
# -------------------------------------------------------------------

class TestSuccessfulRTTAtOrBelowRTO:
    def test_rtt_equal_to_rto_is_success(self):
        lm = LinkMonitor()
        lm.record_sample(50.0)  # RTO becomes 50.0
        lm.record_sample(50.0)  # == RTO → success
        assert lm.consecutive_successes == 2  # first + this one
        assert lm.consecutive_timeouts == 0

    def test_rtt_below_rto_is_success(self):
        lm = LinkMonitor()
        lm.record_sample(50.0)  # RTO becomes 50.0
        lm.record_sample(30.0)  # < RTO → success
        assert lm.consecutive_successes == 2
        assert lm.consecutive_timeouts == 0


# -------------------------------------------------------------------
# 12. Counter reset behaviour
# -------------------------------------------------------------------

class TestCounterReset:
    def test_success_resets_timeout_counter(self):
        lm = LinkMonitor()
        lm.record_sample(100.0)
        for _ in range(3):
            lm.record_sample(None)
        assert lm.consecutive_timeouts == 3
        lm.record_sample(100.0)
        assert lm.consecutive_timeouts == 0

    def test_timeout_resets_success_counter(self):
        lm = LinkMonitor()
        for _ in range(5):
            lm.record_sample(100.0)
        assert lm.consecutive_successes == 5
        lm.record_sample(None)
        assert lm.consecutive_successes == 0


# -------------------------------------------------------------------
# 13. Invalid input
# -------------------------------------------------------------------

class TestInvalidInput:
    def test_negative_rtt_raises(self):
        lm = LinkMonitor()
        with pytest.raises(ValueError, match="non-negative"):
            lm.record_sample(-1.0)

    def test_nan_rtt_raises(self):
        lm = LinkMonitor()
        with pytest.raises(ValueError, match="finite"):
            lm.record_sample(float("nan"))

    def test_inf_rtt_raises(self):
        lm = LinkMonitor()
        with pytest.raises(ValueError, match="finite"):
            lm.record_sample(float("inf"))

    def test_negative_inf_rtt_raises(self):
        lm = LinkMonitor()
        with pytest.raises(ValueError, match="finite"):
            lm.record_sample(float("-inf"))

    def test_string_rtt_raises(self):
        lm = LinkMonitor()
        with pytest.raises(TypeError, match="number or None"):
            lm.record_sample("100")  # type: ignore[arg-type]

    def test_zero_rtt_is_valid(self):
        lm = LinkMonitor()
        lm.record_sample(0.0)
        assert lm.r_hat_ms == pytest.approx(0.0)
        assert lm.d_hat_ms == pytest.approx(0.0)


# -------------------------------------------------------------------
# 14. Constructor / config defaults
# -------------------------------------------------------------------

class TestConfigDefaults:
    def test_defaults_from_config(self):
        lm = LinkMonitor()
        assert lm._alpha == 0.125
        assert lm._beta == 0.25
        assert lm._timeout_threshold == 5
        assert lm._recovery_threshold == 3

    def test_constructor_overrides(self):
        lm = LinkMonitor(alpha=0.5, beta=0.5, timeout_threshold=2, recovery_threshold=1)
        assert lm._alpha == 0.5
        assert lm._beta == 0.5
        assert lm._timeout_threshold == 2
        assert lm._recovery_threshold == 1

    def test_custom_thresholds_fsm(self):
        lm = LinkMonitor(timeout_threshold=2, recovery_threshold=1)
        lm.record_sample(100.0)
        lm.record_sample(None)
        lm.record_sample(None)
        assert lm.state == "DOWN"  # after 2 timeouts
        lm.record_sample(100.0)
        assert lm.state == "UP"  # after 1 recovery


# -------------------------------------------------------------------
# 15. Return value of record_sample
# -------------------------------------------------------------------

class TestReturnValue:
    def test_returns_state(self):
        lm = LinkMonitor()
        assert lm.record_sample(100.0) == "UP"
        for _ in range(5):
            lm.record_sample(None)
        assert lm.record_sample(None) == "DOWN"
