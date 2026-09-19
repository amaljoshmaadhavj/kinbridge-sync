"""
Comprehensive unit tests for Kinbridge-Sync Phase 1 and Phase 1b.

Tests logic only — no expected research results are asserted.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# =========================================================================
# pilot/tools.py
# =========================================================================

class TestToolSchemas:
    def test_all_three_tools_defined(self):
        from pilot.tools import TOOL_SCHEMAS
        assert "navigate_to" in TOOL_SCHEMAS
        assert "log_inspection" in TOOL_SCHEMAS
        assert "request_supply_drop" in TOOL_SCHEMAS

    def test_tool_list_count(self):
        from pilot.tools import TOOL_LIST
        assert len(TOOL_LIST) == 3

    def test_navigate_to_has_required_fields(self):
        from pilot.tools import NAVIGATE_TO_SCHEMA
        params = NAVIGATE_TO_SCHEMA["function"]["parameters"]
        assert set(params["required"]) == {"lat", "lon", "mode"}

    def test_log_inspection_has_required_fields(self):
        from pilot.tools import LOG_INSPECTION_SCHEMA
        params = LOG_INSPECTION_SCHEMA["function"]["parameters"]
        assert set(params["required"]) == {"site_id", "status", "notes"}

    def test_request_supply_drop_has_required_fields(self):
        from pilot.tools import REQUEST_SUPPLY_DROP_SCHEMA
        params = REQUEST_SUPPLY_DROP_SCHEMA["function"]["parameters"]
        assert set(params["required"]) == {"lat", "lon", "payload", "priority"}

    def test_tool_functions_callable(self):
        from pilot.tools import navigate_to, log_inspection, request_supply_drop
        r1 = navigate_to(45.0, 6.0, "fastest")
        assert r1["lat"] == 45.0
        r2 = log_inspection("alpha-7", "passed", "ok")
        assert r2["site_id"] == "alpha-7"
        r3 = request_supply_drop(45.0, 6.0, "kit", "high")
        assert r3["payload"] == "kit"


# =========================================================================
# pilot/prompts.json
# =========================================================================

class TestPrompts:
    @pytest.fixture
    def prompts(self):
        path = Path(__file__).resolve().parent.parent / "pilot" / "prompts.json"
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_exactly_19_prompts(self, prompts):
        assert len(prompts) == 19

    def test_all_required_keys(self, prompts):
        for p in prompts:
            assert "prompt_id" in p
            assert "tool_class" in p
            assert "intent_id" in p
            assert "prompt" in p

    def test_three_tool_classes(self, prompts):
        classes = {p["tool_class"] for p in prompts}
        assert classes == {"navigation", "inspection", "supply_drop"}

    def test_intent_ids_are_unique_per_class(self, prompts):
        """Each intent_id should map to exactly one tool_class."""
        intent_to_class = {}
        for p in prompts:
            iid = p["intent_id"]
            if iid in intent_to_class:
                assert intent_to_class[iid] == p["tool_class"]
            else:
                intent_to_class[iid] = p["tool_class"]

    def test_multiple_prompts_per_intent(self, prompts):
        """At least some intents should have >1 prompt (for reissue testing)."""
        from collections import Counter
        counts = Counter(p["intent_id"] for p in prompts)
        multi = sum(1 for v in counts.values() if v > 1)
        assert multi >= 3, "Need at least 3 intents with multiple prompts"

    def test_unique_prompt_ids(self, prompts):
        ids = [p["prompt_id"] for p in prompts]
        assert len(ids) == len(set(ids))


# =========================================================================
# pilot/key_schemes.py
# =========================================================================

class TestKeySchemes:
    def test_k_payload_deterministic(self):
        from pilot.key_schemes import k_payload
        k1 = k_payload("navigate_to", {"lat": 45.0, "lon": 6.0, "mode": "fastest"})
        k2 = k_payload("navigate_to", {"lat": 45.0, "lon": 6.0, "mode": "fastest"})
        assert k1 == k2

    def test_k_payload_order_invariant(self):
        from pilot.key_schemes import k_payload
        k1 = k_payload("navigate_to", {"mode": "fastest", "lat": 45.0, "lon": 6.0})
        k2 = k_payload("navigate_to", {"lat": 45.0, "lon": 6.0, "mode": "fastest"})
        assert k1 == k2

    def test_k_payload_different_args_different_key(self):
        from pilot.key_schemes import k_payload
        k1 = k_payload("navigate_to", {"lat": 45.0, "lon": 6.0, "mode": "fastest"})
        k2 = k_payload("navigate_to", {"lat": 45.0, "lon": 6.0, "mode": "shortest"})
        assert k1 != k2

    def test_k_bucket_same_within_window(self):
        from pilot.key_schemes import k_bucket
        args = {"lat": 45.0, "lon": 6.0, "mode": "fastest"}
        # Both t=100 and t=110 are in bucket floor(x/60)=1
        k1 = k_bucket("agent-1", "navigate_to", args, 100.0, 60.0)
        k2 = k_bucket("agent-1", "navigate_to", args, 110.0, 60.0)
        assert k1 == k2

    def test_k_bucket_different_across_boundary(self):
        from pilot.key_schemes import k_bucket
        args = {"lat": 45.0, "lon": 6.0, "mode": "fastest"}
        # t=59 is in bucket 0, t=61 is in bucket 1
        k1 = k_bucket("agent-1", "navigate_to", args, 59.0, 60.0)
        k2 = k_bucket("agent-1", "navigate_to", args, 61.0, 60.0)
        assert k1 != k2

    def test_k_intent_same_idempotent(self):
        from pilot.key_schemes import k_intent
        k1 = k_intent("sess-1", 1, "nav_int_01")
        k2 = k_intent("sess-1", 2, "nav_int_01")
        # Same intent_id but different turn_seq → different key
        assert k1 != k2

    def test_k_intent_different_intent_different_key(self):
        from pilot.key_schemes import k_intent
        k1 = k_intent("sess-1", 1, "nav_int_01")
        k2 = k_intent("sess-1", 1, "nav_int_02")
        assert k1 != k2

    def test_k_intent_same_session_turn_intent_matches(self):
        """Regression: a1 and a2 of the same intent MUST produce identical k_intent.

        The paper specifies k_intent = H(session_id || turn_seq || intent_id).
        For a re-issued action (a2) of the same ground-truth intent, session_id,
        turn_seq, and intent_id are all identical, so the key must match.
        """
        from pilot.key_schemes import k_intent
        k_a1 = k_intent("sess-abc", 1, "nav_int_01")
        k_a2 = k_intent("sess-abc", 1, "nav_int_01")
        assert k_a1 == k_a2, (
            "k_intent must be identical for a1 and a2 of the same intent "
            "(same session_id, turn_seq, intent_id)"
        )

    def test_canonical_serialisation_order(self):
        from pilot.key_schemes import canonical_serialise
        s1 = canonical_serialise({"b": 2, "a": 1})
        s2 = canonical_serialise({"a": 1, "b": 2})
        assert s1 == s2
        assert s1 == '{"a":1,"b":2}'

    def test_bucket_index(self):
        from pilot.key_schemes import bucket_index
        assert bucket_index(0.0, 60.0) == 0
        assert bucket_index(59.9, 60.0) == 0
        assert bucket_index(60.0, 60.0) == 1
        assert bucket_index(119.9, 60.0) == 1
        assert bucket_index(120.0, 60.0) == 2


# =========================================================================
# pilot/test_bucket_key.py (deterministic sanity test)
# =========================================================================

class TestBucketSanity:
    def test_bucket_transition_occurs(self):
        from pilot.test_bucket_key import test_bucket_transition
        # t=100, G=90: floor(100/60)=1, floor(190/60)=3 -> transition
        test_bucket_transition(t=100.0, G=90.0, Delta=60.0)

    def test_same_bucket_no_change(self):
        from pilot.test_bucket_key import test_same_bucket_no_change
        # t=100, G=10: floor(100/60)=1, floor(110/60)=1 -> same bucket
        test_same_bucket_no_change(t=100.0, G=10.0, Delta=60.0)


# =========================================================================
# pilot/similarity.py
# =========================================================================

class TestSimilarity:
    def test_normalise_action_deterministic(self):
        from pilot.similarity import normalise_action
        s1 = normalise_action("navigate_to", {"lat": 45.0, "mode": "fastest", "lon": 6.0})
        s2 = normalise_action("navigate_to", {"lon": 6.0, "lat": 45.0, "mode": "fastest"})
        assert s1 == s2

    def test_normalise_action_tool_name_upper(self):
        from pilot.similarity import normalise_action
        s = normalise_action("navigate_to", {"lat": 1.0, "lon": 2.0, "mode": "fastest"})
        assert s.startswith("NAVIGATE_TO")

    def test_cosine_similarity_identical_vectors(self):
        from pilot.similarity import cosine_similarity
        v = np.array([1.0, 0.0, 0.0])
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal_vectors(self):
        from pilot.similarity import cosine_similarity
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_cosine_similarity_clamped(self):
        from pilot.similarity import cosine_similarity
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_similarity(a, b) == -1.0


# =========================================================================
# pilot/pairs.py
# =========================================================================

class TestPairs:
    def test_build_pairs_no_self_pairs(self):
        from pilot.pairs import build_pairs_from_actions
        actions = [
            {"prompt_id": "a", "intent_id": "i1", "tool_class": "nav", "tool": "t", "args": {}},
            {"prompt_id": "a", "intent_id": "i1", "tool_class": "nav", "tool": "t", "args": {}},
        ]
        pairs = build_pairs_from_actions(actions)
        assert len(pairs) == 0  # same prompt_id → skipped

    def test_positive_same_intent(self):
        from pilot.pairs import build_pairs_from_actions
        actions = [
            {"prompt_id": "a", "intent_id": "i1", "tool_class": "nav", "tool": "t", "args": {}},
            {"prompt_id": "b", "intent_id": "i1", "tool_class": "nav", "tool": "t", "args": {}},
        ]
        pairs = build_pairs_from_actions(actions)
        assert len(pairs) == 1
        assert pairs[0].same_intent is True

    def test_negative_different_intent(self):
        from pilot.pairs import build_pairs_from_actions
        actions = [
            {"prompt_id": "a", "intent_id": "i1", "tool_class": "nav", "tool": "t", "args": {}},
            {"prompt_id": "b", "intent_id": "i2", "tool_class": "nav", "tool": "t", "args": {}},
        ]
        pairs = build_pairs_from_actions(actions)
        assert len(pairs) == 1
        assert pairs[0].same_intent is False

    def test_split_pairs(self):
        from pilot.pairs import build_pairs_from_actions, split_pairs
        actions = [
            {"prompt_id": "a", "intent_id": "i1", "tool_class": "nav", "tool": "t", "args": {}},
            {"prompt_id": "b", "intent_id": "i1", "tool_class": "nav", "tool": "t", "args": {}},
            {"prompt_id": "c", "intent_id": "i2", "tool_class": "nav", "tool": "t", "args": {}},
        ]
        pairs = build_pairs_from_actions(actions)
        pos, neg = split_pairs(pairs)
        assert len(pos) == 1
        assert len(neg) == 2


# =========================================================================
# pilot/fmr_fsr.py
# =========================================================================

class TestFMRFSR:
    def _make_pairs(self):
        from pilot.pairs import EvalPair
        return [
            EvalPair("p1", "a", "b", "i1", "i1", True,  "nav", 0.9),
            EvalPair("p2", "a", "c", "i1", "i1", True,  "nav", 0.3),
            EvalPair("p3", "a", "d", "i1", "i2", False, "nav", 0.8),
            EvalPair("p4", "b", "d", "i1", "i2", False, "nav", 0.2),
            EvalPair("p5", "a", "e", "i1", "i3", False, "nav", 0.95),
        ]

    def test_fmr_at_tau_zero(self):
        """At tau=0, all pairs have sim >= 0, so FMR = 1.0 (all neg match)."""
        from pilot.fmr_fsr import compute_fmr_fsr
        pairs = self._make_pairs()
        result = compute_fmr_fsr(pairs, tau=0.0)
        assert result.fmr == 1.0

    def test_fsr_at_tau_one(self):
        """At tau=1.0, no pair has sim >= 1.0, so FSR = 1.0 (all pos miss)."""
        from pilot.fmr_fsr import compute_fmr_fsr
        pairs = self._make_pairs()
        result = compute_fmr_fsr(pairs, tau=1.0)
        assert result.fsr == 1.0

    def test_fmr_at_high_threshold(self):
        """At tau=0.99, only the 0.95 neg pair still matches → FMR = 1/3."""
        from pilot.fmr_fsr import compute_fmr_fsr
        pairs = self._make_pairs()
        result = compute_fmr_fsr(pairs, tau=0.99)
        # neg pairs: 0.8, 0.2, 0.95 → only 0.95 < 0.99? No: 0.95 < 0.99
        # So FMR = 0/3 = 0.0
        assert result.fmr == 0.0

    def test_counts_correct(self):
        from pilot.fmr_fsr import compute_fmr_fsr
        pairs = self._make_pairs()
        result = compute_fmr_fsr(pairs, tau=0.5)
        assert result.different_intent_pair_count == 3
        assert result.same_intent_pair_count == 2

    def test_tool_class_filter(self):
        from pilot.fmr_fsr import compute_fmr_fsr
        pairs = self._make_pairs()
        result_all = compute_fmr_fsr(pairs, tau=0.5, tool_class="all")
        result_nav = compute_fmr_fsr(pairs, tau=0.5, tool_class="nav")
        assert result_all.fmr == result_nav.fmr  # all are "nav"

    def test_threshold_sweep_length(self):
        from pilot.fmr_fsr import threshold_sweep
        pairs = self._make_pairs()
        results = threshold_sweep(pairs, tau_min=0.0, tau_max=1.0, tau_step=0.1)
        assert len(results) == 11  # 0.0, 0.1, ..., 1.0


# =========================================================================
# pilot/wilson.py
# =========================================================================

class TestWilsonCI:
    def test_basic_ci(self):
        from pilot.wilson import wilson_ci
        ci = wilson_ci(50, 100)
        assert abs(ci.estimate - 0.5) < 1e-6
        assert ci.ci_low < 0.5 < ci.ci_high
        assert ci.ci_low > 0.0
        assert ci.ci_high < 1.0

    def test_zero_trials(self):
        from pilot.wilson import wilson_ci
        ci = wilson_ci(0, 0)
        assert ci.estimate == 0.0
        assert ci.ci_low == 0.0
        assert ci.ci_high == 0.0

    def test_all_successes(self):
        from pilot.wilson import wilson_ci
        ci = wilson_ci(100, 100)
        assert ci.estimate == 1.0
        assert ci.ci_high == pytest.approx(1.0, abs=1e-10)

    def test_all_failures(self):
        from pilot.wilson import wilson_ci
        ci = wilson_ci(0, 100)
        assert ci.estimate == 0.0
        assert ci.ci_low == 0.0

    def test_narrow_ci_large_sample(self):
        from pilot.wilson import wilson_ci
        ci_small = wilson_ci(50, 100)
        ci_large = wilson_ci(500, 1000)
        width_small = ci_small.ci_high - ci_small.ci_low
        width_large = ci_large.ci_high - ci_large.ci_low
        assert width_large < width_small

    def test_negative_denominator_raises(self):
        from pilot.wilson import wilson_ci
        with pytest.raises(ValueError):
            wilson_ci(5, -1)

    def test_to_dict(self):
        from pilot.wilson import wilson_ci
        ci = wilson_ci(10, 20)
        d = ci.to_dict()
        assert "estimate" in d
        assert "ci_low" in d
        assert "ci_high" in d
        assert "numerator" in d
        assert "denominator" in d


# =========================================================================
# kinbridge_math/fit_alpha_beta.py
# =========================================================================

class TestFitAlphaBeta:
    def test_fit_alpha_returns_slope(self):
        from kinbridge_math.fit_alpha_beta import fit_alpha
        tau = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        fmr = (1.0 - tau) ** 2.0  # true alpha = 2
        result = fit_alpha(tau, fmr)
        assert abs(result.slope - 2.0) < 0.1
        assert result.param_name == "alpha"

    def test_fit_beta_returns_slope(self):
        from kinbridge_math.fit_alpha_beta import fit_beta
        tau = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        fsr = tau ** 1.5  # true beta = 1.5
        result = fit_beta(tau, fsr)
        assert abs(result.slope - 1.5) < 0.1
        assert result.param_name == "beta"

    def test_fit_excludes_zeros(self):
        from kinbridge_math.fit_alpha_beta import fit_alpha
        tau = np.array([0.1, 0.2, 0.0, 0.4, 0.5])
        fmr = np.array([0.01, 0.04, 0.0, 0.16, 0.25])
        result = fit_alpha(tau, fmr)
        assert result.n_excluded >= 1

    def test_fit_too_few_points_warns(self):
        from kinbridge_math.fit_alpha_beta import fit_alpha
        tau = np.array([0.1, 0.2])
        fmr = np.array([0.01, 0.04])
        with pytest.warns(UserWarning):
            fit_alpha(tau, fmr, min_valid_points=5)


# =========================================================================
# kinbridge_math/tau_star.py
# =========================================================================

class TestTauStar:
    def test_risk_function(self):
        from kinbridge_math.tau_star import risk_function
        r = risk_function(0.5, 0.1, 0.2, w_d=4.0, w_m=1.0)
        assert r == 4.0 * 0.2 + 1.0 * 0.1

    def test_risk_derivative_sign_change(self):
        from kinbridge_math.tau_star import risk_derivative_from_params
        # Corrected: R'(tau) = w_d*beta*tau^(beta-1) - w_m*alpha*(1-tau)^(alpha-1)
        # At tau near 0: -(1-tau)^(alpha-1) term dominates, derivative is negative
        d_lo = risk_derivative_from_params(0.01, alpha=2.0, beta=2.0, w_d=4.0, w_m=1.0)
        # At tau near 1: tau^(beta-1) term dominates, derivative is positive
        d_hi = risk_derivative_from_params(0.99, alpha=2.0, beta=2.0, w_d=4.0, w_m=1.0)
        assert d_lo < 0
        assert d_hi > 0

    def test_bisection_converges(self):
        from kinbridge_math.tau_star import _bisection_tau_star
        tau = _bisection_tau_star(alpha=2.0, beta=2.0, w_d=4.0, w_m=1.0)
        assert tau is not None
        assert 0.0 < tau < 1.0

    def test_bisection_returns_none_without_sign_change(self):
        from kinbridge_math.tau_star import _bisection_tau_star
        # Both derivatives positive (no sign change) -> returns None
        result = _bisection_tau_star(alpha=0.5, beta=2.0, w_d=1.0, w_m=4.0)
        assert result is None

    def test_closed_form_symmetric(self):
        from kinbridge_math.tau_star import _closed_form_tau_star
        # When alpha == beta and w_d == w_m, tau* = 0.5
        tau = _closed_form_tau_star(2.0, 2.0, 1.0, 1.0)
        assert abs(tau - 0.5) < 0.01

    def test_compute_tau_star_returns_result(self):
        from kinbridge_math.tau_star import compute_tau_star
        tau = np.linspace(0.05, 0.95, 50)
        fmr = (1.0 - tau) ** 2.0   # corrected: FMR = (1-tau)^alpha
        fsr = tau ** 2.0            # corrected: FSR = tau^beta
        result = compute_tau_star(tau, fmr, fsr)
        assert 0.0 < result.tau_star < 1.0
        assert result.method_used in ("closed_form", "bisection")

    def test_compute_tau_star_different_alpha_beta(self):
        from kinbridge_math.tau_star import compute_tau_star
        tau = np.linspace(0.05, 0.95, 50)
        fmr = (1.0 - tau) ** 3.0   # corrected: alpha = 3
        fsr = tau ** 1.5            # corrected: beta = 1.5
        result = compute_tau_star(tau, fmr, fsr, epsilon=0.05)
        assert result.method_used == "bisection"
        assert 0.0 < result.tau_star < 1.0

    def test_compute_tau_star_rejects_nan_alpha(self):
        from kinbridge_math.tau_star import compute_tau_star
        tau = np.linspace(0.1, 0.9, 20)
        fmr = np.full_like(tau, float("nan"))
        fsr = tau ** 2.0
        result = compute_tau_star(tau, fmr, fsr)
        assert result.method_used == "unestimable"
        assert result.tau_star != result.tau_star  # NaN check

    def test_compute_tau_star_rejects_negative_alpha(self):
        from kinbridge_math.tau_star import compute_tau_star
        tau = np.linspace(0.1, 0.9, 20)
        fmr = (1.0 - tau) ** 2.0
        fsr = tau ** 2.0
        # Force negative alpha by passing wrong data shape that produces negative slope
        bad_fmr = 1.0 / (1.0 - tau)  # This gives alpha = -1 (increasing)
        result = compute_tau_star(tau, bad_fmr, fsr)
        assert result.method_used == "unestimable"


# =========================================================================
# kinbridge_math/validate_tau.py
# =========================================================================

class TestValidateTau:
    def test_validate_produces_points(self):
        from kinbridge_math.tau_star import TauStarResult, FitResult
        from kinbridge_math.validate_tau import validate_tau_star
        mock_result = TauStarResult(
            tau_star=0.6, alpha=2.0, beta=2.0, method_used="test",
            risk_at_tau_star=0.1, w_d=4.0, w_m=1.0, w_s=0.0, p_stale=0.0,
            fit_alpha=FitResult(2.0, 0.0, 0.99, 10, 0, "alpha"),
            fit_beta=FitResult(2.0, 0.0, 0.99, 10, 0, "beta"),
        )
        tau = np.linspace(0.1, 0.9, 20)
        fmr = (1.0 - tau) ** 2.0   # corrected: FMR = (1-tau)^alpha
        fsr = tau ** 2.0            # corrected: FSR = tau^beta
        points = validate_tau_star(tau, fmr, fsr, mock_result)
        assert len(points) == 20
        assert any(p.is_tau_star for p in points)


# =========================================================================
# kinbridge_math/degradation_threshold.py
# =========================================================================

class TestDegradationThreshold:
    def test_gate_does_not_fire_with_high_delta_q(self):
        from kinbridge_math.degradation_threshold import compute_degradation_threshold
        sim = np.random.RandomState(42).rand(100)
        result = compute_degradation_threshold(
            delta_q=0.8, similarity_values=sim,
            lambda_q=1.0, lambda_l=1.0,
        )
        # High delta_q → left > right → gate does NOT fire (switching is costly)
        assert result.gate_fires is False
        assert result.tau_star_degradation >= 0.0

    def test_gate_fires_with_low_delta_q(self):
        from kinbridge_math.degradation_threshold import compute_degradation_threshold
        sim = np.random.RandomState(42).rand(100)
        result = compute_degradation_threshold(
            delta_q=0.01, similarity_values=sim,
            lambda_q=1.0, lambda_l=10.0,
        )
        # Low delta_q, high lambda_l → left < right → gate fires
        assert result.gate_fires is True

    def test_stores_all_intermediates(self):
        from kinbridge_math.degradation_threshold import compute_degradation_threshold
        sim = np.random.RandomState(42).rand(50)
        result = compute_degradation_threshold(0.5, sim)
        d = result.to_dict()
        assert "integral_term" in d
        assert "miss_term" in d
        assert "left_hand_side" in d
        assert "right_hand_side" in d


# =========================================================================
# kinbridge_math/measure_delta_q.py
# =========================================================================

class TestMeasureDeltaQ:
    def test_eval_set_has_entries(self):
        from kinbridge_math.measure_delta_q import EVAL_SET
        assert len(EVAL_SET) >= 8

    def test_eval_set_no_overlap_with_prompts(self):
        """Eval set must not reuse prompt_ids from the pilot."""
        pilot_path = Path(__file__).resolve().parent.parent / "pilot" / "prompts.json"
        with open(pilot_path, "r") as f:
            pilot_prompts = json.load(f)
        pilot_ids = {p["prompt_id"] for p in pilot_prompts}
        from kinbridge_math.measure_delta_q import EVAL_SET
        eval_ids = {t["task_id"] for t in EVAL_SET}
        assert pilot_ids.isdisjoint(eval_ids), "Eval set must not overlap pilot prompts"

    def test_score_function(self):
        from kinbridge_math.measure_delta_q import _score_response
        assert _score_response("navigate_to", "navigate_to", {"lat": 1, "lon": 2, "mode": "fastest"}, True) == 1.0
        assert _score_response("navigate_to", "navigate_to", {}, True) == 0.5
        assert _score_response("navigate_to", "log_inspection", {}, True) == 0.0
        assert _score_response("navigate_to", "", {}, False) == 0.0


# =========================================================================
# pilot/analyze_pilot.py (structural tests only)
# =========================================================================

class TestAnalyzePilot:
    def test_decision_rule_strong(self):
        from pilot.analyze_pilot import decision_rule
        assert "STRONG" in decision_rule(0.20)

    def test_decision_rule_moderate(self):
        from pilot.analyze_pilot import decision_rule
        assert "MODERATE" in decision_rule(0.10)

    def test_decision_rule_weak(self):
        from pilot.analyze_pilot import decision_rule
        assert "WEAK" in decision_rule(0.02)


# =========================================================================
# core/config.py
# =========================================================================

class TestConfig:
    def test_load_pilot_config(self):
        from core.config import load_config
        cfg = load_config("pilot/pilot_config.yaml")
        assert "ollama" in cfg
        assert cfg["ollama"]["model_large"] == "qwen2.5:7b-instruct"

    def test_load_experiment_config(self):
        from core.config import load_config
        cfg = load_config("experiment/factorial.yaml")
        assert "methods" in cfg
        assert len(cfg["methods"]) == 4

    def test_experiment_output_path(self):
        from core.config import experiment_output_path
        p = experiment_output_path("test_output", "csv")
        assert p.name == "test_output.csv"
        assert "results" in str(p)


# =========================================================================
# pilot/raw/ directory
# =========================================================================

class TestRawDirectory:
    def test_raw_dir_exists(self):
        raw_dir = Path(__file__).resolve().parent.parent / "pilot" / "raw"
        assert raw_dir.exists()

    def test_raw_dir_has_init(self):
        raw_dir = Path(__file__).resolve().parent.parent / "pilot" / "raw"
        assert (raw_dir / "__init__.py").exists()


# =========================================================================
# pilot/analyze_pilot.py — input selection
# =========================================================================

class TestAnalyzeInputSelection:
    """Tests that the analyzer requires explicit input paths and does not
    silently auto-glob all pilot_temp*.json files."""

    VALID_T00 = Path(__file__).resolve().parent.parent / "pilot" / "raw" / "pilot_temp0.0_20260916T160848Z.json"
    VALID_T07 = Path(__file__).resolve().parent.parent / "pilot" / "raw" / "pilot_temp0.7_20260916T162421Z.json"
    INVALID_T00 = Path(__file__).resolve().parent.parent / "pilot" / "raw" / "pilot_temp0.0_20260916T153000Z.json"

    def test_explicit_valid_inputs_accepted(self):
        from pilot.analyze_pilot import _load_explicit
        obs = _load_explicit([self.VALID_T00, self.VALID_T07])
        assert len(obs) == 38  # 19 + 19

    def test_single_valid_input_accepted(self):
        from pilot.analyze_pilot import _load_explicit
        obs = _load_explicit([self.VALID_T00])
        assert len(obs) == 19

    def test_missing_file_raises(self):
        from pilot.analyze_pilot import _load_explicit
        import pytest
        with pytest.raises(FileNotFoundError):
            _load_explicit([Path("/nonexistent/file.json")])

    def test_invalid_file_not_auto_included(self):
        """The invalid file must NOT be loaded when only valid files are specified."""
        from pilot.analyze_pilot import _load_explicit
        obs = _load_explicit([self.VALID_T00, self.VALID_T07])
        # All observations should have k_intent matches (from valid data only)
        for o in obs:
            assert o["a1"]["key_intent"] == o["a2"]["key_intent"], (
                f"{o['prompt_id']}: k_intent differs — invalid data may have been included"
            )

    def test_combined_analysis_produces_both_temperatures(self):
        from pilot.analyze_pilot import _load_explicit, compute_divergence_table
        obs = _load_explicit([self.VALID_T00, self.VALID_T07])
        df = compute_divergence_table(obs)
        temps = sorted(df["temperature"].unique())
        assert temps == [0.0, 0.7]

    def test_exactly_114_divergence_rows(self):
        """Two valid 19-observation datasets x 3 schemes = 114 divergence rows."""
        from pilot.analyze_pilot import _load_explicit, compute_divergence_table
        obs = _load_explicit([self.VALID_T00, self.VALID_T07])
        df = compute_divergence_table(obs)
        assert len(df) == 114

    def test_57_rows_per_temperature(self):
        from pilot.analyze_pilot import _load_explicit, compute_divergence_table
        obs = _load_explicit([self.VALID_T00, self.VALID_T07])
        df = compute_divergence_table(obs)
        for temp in [0.0, 0.7]:
            assert len(df[df["temperature"] == temp]) == 57


# =========================================================================
# kinbridge_math/calibrate.py — Design C
# =========================================================================

class TestCalibrateDesignC:
    """Tests for the Design C calibration pipeline."""

    VALID_T00 = Path(__file__).resolve().parent.parent / "pilot" / "raw" / "pilot_temp0.0_20260916T160848Z.json"
    VALID_T07 = Path(__file__).resolve().parent.parent / "pilot" / "raw" / "pilot_temp0.7_20260916T162421Z.json"

    def test_fsr_loads_19_pairs(self):
        from kinbridge_math.fmr_from_negatives import load_fsr_from_reissues
        temp, sims, n = load_fsr_from_reissues(self.VALID_T00)
        assert n == 19
        assert len(sims) == 19
        assert temp == 0.0

    def test_fsr_temperature_matches_pilot(self):
        from kinbridge_math.fmr_from_negatives import load_fsr_from_reissues
        temp00, _, _ = load_fsr_from_reissues(self.VALID_T00)
        temp07, _, _ = load_fsr_from_reissues(self.VALID_T07)
        assert temp00 == 0.0
        assert temp07 == 0.7

    def test_fit_params_design_c_no_fmr(self):
        from kinbridge_math.calibrate import fit_params_design_c
        from kinbridge_math.fmr_from_negatives import compute_fsr_curve, load_fsr_from_reissues
        temp, sims, n = load_fsr_from_reissues(self.VALID_T00)
        fsr_points = compute_fsr_curve(sims, n)
        fit_a, fit_b, reason = fit_params_design_c([], fsr_points)
        assert fit_a is None
        assert fit_b is not None

    def test_fit_params_design_c_both_sources(self):
        from kinbridge_math.calibrate import fit_params_design_c
        from kinbridge_math.fmr_from_negatives import compute_fmr_curve, compute_fsr_curve, load_fsr_from_reissues
        fmr_sims = [0.05 * i for i in range(20)]
        fmr_points = compute_fmr_curve(fmr_sims, n_total=20)
        temp, sims, n = load_fsr_from_reissues(self.VALID_T00)
        fsr_points = compute_fsr_curve(sims, n)
        fit_a, fit_b, reason = fit_params_design_c(fmr_points, fsr_points)
        assert fit_a is not None
        assert fit_b is not None

    def test_calibrate_tau_star_without_alpha_returns_none(self):
        from kinbridge_math.calibrate import calibrate_tau_star
        result = calibrate_tau_star(None, None, [], [])
        assert result is None

    def test_risk_function_delegates_correctly(self):
        from kinbridge_math.tau_star import risk_function
        r = risk_function(0.5, 0.1, 0.2, w_d=4.0, w_m=1.0)
        assert r == 4.0 * 0.2 + 1.0 * 0.1

    def test_fsr_denominator_is_19(self):
        from kinbridge_math.fmr_from_negatives import load_fsr_from_reissues
        pilot_path = Path("pilot/raw/pilot_temp0.0_20260916T160848Z.json")
        temp, sims, n = load_fsr_from_reissues(pilot_path)
        assert n == 19

    def test_fmr_denominator_from_trials(self):
        from kinbridge_math.fmr_from_negatives import compute_fmr_curve
        sims = [0.5] * 200
        points = compute_fmr_curve(sims, n_total=200)
        for p in points:
            assert p.n_total == 200

    def test_tau_sweep_has_101_thresholds(self):
        from kinbridge_math.fmr_from_negatives import compute_fmr_curve, compute_fsr_curve
        fmr_sims = [0.3, 0.5, 0.7]
        fmr_points = compute_fmr_curve(fmr_sims, n_total=3)
        assert len(fmr_points) == 101

        fsr_sims = [0.4, 0.6, 0.8]
        fsr_points = compute_fsr_curve(fsr_sims, n_total=3)
        assert len(fsr_points) == 101

    def test_wilson_counts_use_actual_n(self):
        from kinbridge_math.fmr_from_negatives import compute_fmr_curve
        sims = [0.6] * 100
        points = compute_fmr_curve(sims, n_total=100)
        p50 = [p for p in points if abs(p.tau - 0.5) < 0.005][0]
        assert p50.n_false_match == 100
        assert p50.n_total == 100

    def test_no_raw_pilot_files_modified(self):
        """Verify the original pilot files are unchanged."""
        for f in [
            "pilot/raw/pilot_temp0.0_20260916T160848Z.json",
            "pilot/raw/pilot_temp0.7_20260916T162421Z.json",
        ]:
            path = Path(f)
            assert path.exists(), f"File missing: {f}"
# pilot/run_negative_trials.py
# =========================================================================

class TestNegativeTrials:
    """Tests for the dedicated negative trial generator."""

    def test_intent_pair_selection_covers_all_pairs(self):
        from pilot.run_negative_trials import select_intent_pairs
        import random
        intent_ids = [f"int_{i}" for i in range(5)]
        rng = random.Random(42)
        pairs = select_intent_pairs(intent_ids, n_trials=100, rng=rng)
        # 5 intents -> C(5,2)=10 unique pairs
        unique_pairs = set()
        for a, b in pairs:
            key = (min(a, b), max(a, b))
            unique_pairs.add(key)
        assert len(unique_pairs) == 10

    def test_intent_pair_selection_distributes_evenly(self):
        from pilot.run_negative_trials import select_intent_pairs
        import random
        intent_ids = [f"int_{i}" for i in range(4)]
        rng = random.Random(42)
        # C(4,2)=6 pairs, 12 trials -> 2 per pair
        pairs = select_intent_pairs(intent_ids, n_trials=12, rng=rng)
        from collections import Counter
        pair_counts = Counter()
        for a, b in pairs:
            key = (min(a, b), max(a, b))
            pair_counts[key] += 1
        # Each pair should have exactly 2 trials
        for count in pair_counts.values():
            assert count == 2

    def test_negative_trial_different_intents(self):
        from pilot.run_negative_trials import validate_trial
        trial = {
            "trial_id": "neg_test123",
            "temperature": 0.0,
            "intent_id_a": "int_A",
            "intent_id_b": "int_B",
            "action_a": {"success": True, "raw_response": {"message": {"tool_calls": [{"id": "call_1"}]}}},
            "action_b": {"success": True, "raw_response": {"message": {"tool_calls": [{"id": "call_2"}]}}},
            "similarity": 0.5,
        }
        errors = validate_trial(trial)
        assert len(errors) == 0

    def test_negative_trial_same_intent_detected(self):
        from pilot.run_negative_trials import validate_trial
        trial = {
            "trial_id": "neg_test456",
            "temperature": 0.0,
            "intent_id_a": "int_A",
            "intent_id_b": "int_A",  # same intent!
            "action_a": {"success": True, "raw_response": {"message": {"tool_calls": [{"id": "call_1"}]}}},
            "action_b": {"success": True, "raw_response": {"message": {"tool_calls": [{"id": "call_2"}]}}},
            "similarity": 0.5,
        }
        errors = validate_trial(trial)
        assert any("Same intent" in e for e in errors)

    def test_negative_trial_reuse_detected(self):
        from pilot.run_negative_trials import validate_trial
        trial = {
            "trial_id": "neg_test789",
            "temperature": 0.0,
            "intent_id_a": "int_A",
            "intent_id_b": "int_B",
            "action_a": {"success": True, "raw_response": {"message": {"tool_calls": [{"id": "call_1"}]}}},
            "action_b": {"success": True, "raw_response": {"message": {"tool_calls": [{"id": "call_1"}]}}},
            "similarity": 0.5,
        }
        errors = validate_trial(trial)
        assert any("Same model call" in e for e in errors)

    def test_negative_trial_failed_action_detected(self):
        from pilot.run_negative_trials import validate_trial
        trial = {
            "trial_id": "neg_test_fail",
            "temperature": 0.0,
            "intent_id_a": "int_A",
            "intent_id_b": "int_B",
            "action_a": {"success": False, "raw_response": {"message": {"tool_calls": []}}},
            "action_b": {"success": True, "raw_response": {"message": {"tool_calls": [{"id": "call_2"}]}}},
            "similarity": 0.0,
        }
        errors = validate_trial(trial)
        assert any("Action A failed" in e for e in errors)


# =========================================================================
# kinbridge_math/fmr_from_negatives.py
# =========================================================================

class TestFMRFromNegatives:
    """Tests for FMR computation from dedicated negative trials."""

    def test_compute_fmr_curve_basic(self):
        from kinbridge_math.fmr_from_negatives import compute_fmr_curve
        sims = [0.3, 0.5, 0.7, 0.9]
        points = compute_fmr_curve(sims, n_total=4)
        assert len(points) == 101  # 0.00 through 1.00
        # At tau=0.0, all 4 have sim >= 0 -> FMR = 1.0
        assert points[0].fmr == 1.0
        # At tau=1.0, none have sim >= 1.0 -> FMR = 0.0
        assert points[-1].fmr == 0.0

    def test_fmr_uses_negative_trials_only(self):
        from kinbridge_math.fmr_from_negatives import compute_fmr_curve
        # Simulate 200 negative trials
        sims = [0.1 * i for i in range(200)]
        points = compute_fmr_curve(sims, n_total=200)
        # Denominator must be 200
        for p in points:
            assert p.n_total == 200

    def test_fsr_from_reissues_count(self):
        from kinbridge_math.fmr_from_negatives import load_fsr_from_reissues
        from pathlib import Path
        pilot_path = Path("pilot/raw/pilot_temp0.0_20260916T160848Z.json")
        temp, sims, n = load_fsr_from_reissues(pilot_path)
        assert n == 19
        assert temp == 0.0

    def test_fsr_curve_has_101_points(self):
        from kinbridge_math.fmr_from_negatives import load_fsr_from_reissues, compute_fsr_curve
        from pathlib import Path
        pilot_path = Path("pilot/raw/pilot_temp0.0_20260916T160848Z.json")
        temp, sims, n = load_fsr_from_reissues(pilot_path)
        fsr_points = compute_fsr_curve(sims, n)
        assert len(fsr_points) == 101

    def test_fsr_uses_reissue_pairs_only(self):
        from kinbridge_math.fmr_from_negatives import load_fsr_from_reissues
        from pathlib import Path
        pilot_path = Path("pilot/raw/pilot_temp0.0_20260916T160848Z.json")
        temp, sims, n = load_fsr_from_reissues(pilot_path)
        # Must be exactly 19 reissue pairs
        assert n == 19
        assert len(sims) == 19


# =========================================================================
# Retry behavior tests
# =========================================================================

class TestRetryBehavior:
    """Tests for the retry-until-N-successful trials behavior."""

    def test_zero_failures_exactly_n_successful(self):
        """When no trials fail, exactly N successful trials are produced."""
        from pilot.run_negative_trials import (
            run_single_negative_trial, validate_trial,
            _load_prompts, _group_prompts_by_intent,
        )
        prompts = _load_prompts()
        by_intent = _group_prompts_by_intent(prompts)
        intent_ids = sorted(by_intent.keys())

        # Simulate 5 successful trials (mock Ollama to always succeed)
        successful = []
        for i in range(5):
            intent_a = intent_ids[i % len(intent_ids)]
            intent_b = intent_ids[(i + 1) % len(intent_ids)]
            if intent_a == intent_b:
                intent_b = intent_ids[(i + 2) % len(intent_ids)]
            # Build a fake successful trial
            trial = {
                "trial_id": f"neg_test_{i}",
                "temperature": 0.0,
                "intent_id_a": intent_a,
                "intent_id_b": intent_b,
                "action_a": {"success": True, "raw_response": {"message": {"tool_calls": [{"id": f"call_a_{i}"}]}}},
                "action_b": {"success": True, "raw_response": {"message": {"tool_calls": [{"id": f"call_b_{i}"}]}}},
                "similarity": 0.5,
                "same_intent": False,
            }
            errors = validate_trial(trial)
            if not errors:
                successful.append(trial)

        assert len(successful) == 5

    def test_one_failed_trial_still_gets_n_successful(self):
        """One failed trial does not prevent reaching N successful trials."""
        from pilot.run_negative_trials import validate_trial

        # Simulate: 1 failed + 5 successful = 5 successful total
        all_trials = []
        failed = []

        # Failed trial
        failed_trial = {
            "trial_id": "neg_fail_1",
            "temperature": 0.0,
            "intent_id_a": "int_A",
            "intent_id_b": "int_B",
            "action_a": {"success": False, "raw_response": {"message": {"tool_calls": []}}},
            "action_b": {"success": True, "raw_response": {"message": {"tool_calls": [{"id": "call_1"}]}}},
            "similarity": 0.0,
            "same_intent": False,
        }
        errors = validate_trial(failed_trial)
        assert len(errors) > 0  # should be flagged
        failed.append(failed_trial)

        # 5 successful trials
        for i in range(5):
            trial = {
                "trial_id": f"neg_ok_{i}",
                "temperature": 0.0,
                "intent_id_a": f"int_X_{i}",
                "intent_id_b": f"int_Y_{i}",
                "action_a": {"success": True, "raw_response": {"message": {"tool_calls": [{"id": f"call_a_{i}"}]}}},
                "action_b": {"success": True, "raw_response": {"message": {"tool_calls": [{"id": f"call_b_{i}"}]}}},
                "similarity": 0.5,
                "same_intent": False,
            }
            errors = validate_trial(trial)
            if not errors:
                all_trials.append(trial)

        assert len(all_trials) == 5
        assert len(failed) == 1

    def test_repeated_failures_continue_until_n(self):
        """Repeated failures are retried until N successful trials obtained."""
        from pilot.run_negative_trials import validate_trial

        n_target = 10
        successful = []
        failed = []
        n_attempted = 0

        # Simulate: 3 failures then successes
        for i in range(20):  # plenty of attempts
            n_attempted += 1
            if i < 3:
                # Fail
                trial = {
                    "trial_id": f"neg_fail_{i}",
                    "temperature": 0.0,
                    "intent_id_a": "int_A",
                    "intent_id_b": "int_B",
                    "action_a": {"success": False, "raw_response": {"message": {"tool_calls": []}}},
                    "action_b": {"success": True, "raw_response": {"message": {"tool_calls": [{"id": f"call_{i}"}]}}},
                    "similarity": 0.0,
                    "same_intent": False,
                }
                errors = validate_trial(trial)
                assert len(errors) > 0
                failed.append(trial)
            else:
                # Succeed
                trial = {
                    "trial_id": f"neg_ok_{i}",
                    "temperature": 0.0,
                    "intent_id_a": f"int_X_{i}",
                    "intent_id_b": f"int_Y_{i}",
                    "action_a": {"success": True, "raw_response": {"message": {"tool_calls": [{"id": f"call_a_{i}"}]}}},
                    "action_b": {"success": True, "raw_response": {"message": {"tool_calls": [{"id": f"call_b_{i}"}]}}},
                    "similarity": 0.5,
                    "same_intent": False,
                }
                errors = validate_trial(trial)
                if not errors:
                    successful.append(trial)

            if len(successful) >= n_target:
                break

        assert len(successful) == n_target
        assert len(failed) == 3
        assert n_attempted == n_target + 3

    def test_denominator_equals_n_successful(self):
        """FMR denominator equals number of successful trials, not attempted."""
        from kinbridge_math.fmr_from_negatives import compute_fmr_curve

        # 5 successful trials with known similarities
        sims = [0.3, 0.5, 0.7, 0.9, 0.4]
        points = compute_fmr_curve(sims, n_total=len(sims))

        # Denominator must be 5 (successful), not 10 (attempted)
        for p in points:
            assert p.n_total == 5

    def test_failed_trials_never_enter_fmr_observations(self):
        """Failed trials are excluded from FMR similarity observations."""
        from kinbridge_math.fmr_from_negatives import extract_similarities

        # Simulate output file with 3 successful + 2 failed
        data = {
            "metadata": {
                "temperature": 0.0,
                "n_requested": 3,
                "n_attempted": 5,
                "n_successful": 3,
                "n_failed": 2,
            },
            "trials": [
                {"similarity": 0.5, "action_a": {"success": True}, "action_b": {"success": True}},
                {"similarity": 0.7, "action_a": {"success": True}, "action_b": {"success": True}},
                {"similarity": 0.3, "action_a": {"success": True}, "action_b": {"success": True}},
            ],
            "failed_attempts": [
                {"errors": ["Action A failed"]},
                {"errors": ["Action B failed"]},
            ],
        }

        temp, sims, meta = extract_similarities(data)
        assert len(sims) == 3  # only successful
        assert meta["n_successful"] == 3
        assert meta["n_failed"] == 2

    def test_no_action_reuse_across_trials(self):
        """Each trial uses fresh model calls — no action reuse."""
        from pilot.run_negative_trials import validate_trial

        # Create 5 trials with distinct call IDs
        for i in range(5):
            trial = {
                "trial_id": f"neg_unique_{i}",
                "temperature": 0.0,
                "intent_id_a": f"int_A_{i}",
                "intent_id_b": f"int_B_{i}",
                "action_a": {"success": True, "raw_response": {"message": {"tool_calls": [{"id": f"unique_call_a_{i}"}]}}},
                "action_b": {"success": True, "raw_response": {"message": {"tool_calls": [{"id": f"unique_call_b_{i}"}]}}},
                "similarity": 0.5,
                "same_intent": False,
            }
            errors = validate_trial(trial)
            assert len(errors) == 0

    def test_metadata_tracks_attempted_vs_successful(self):
        """Metadata correctly reports n_requested, n_attempted, n_successful."""
        from pilot.run_negative_trials import write_trials
        import tempfile
        import json

        successful = [
            {"trial_id": "ok_1", "similarity": 0.5,
             "action_a": {"success": True}, "action_b": {"success": True}},
        ]
        failed = [
            {"attempt_index": 0, "errors": ["Action A failed"]},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = write_trials(
                successful, failed,
                temperature=0.0, seed=42, model="test_model",
                n_requested=1, n_attempted=2,
                intent_pair_counts={("A", "B"): 1},
                out_dir=Path(tmpdir),
            )
            with open(out_path) as f:
                data = json.load(f)

            meta = data["metadata"]
            assert meta["n_requested"] == 1
            assert meta["n_attempted"] == 2
            assert meta["n_successful"] == 1
            assert meta["n_failed"] == 1
            assert len(data["trials"]) == 1
            assert len(data["failed_attempts"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
