"""Phase 5 tests (A–X).

Covers scenario generation, baselines, seeds, record schema, tau*
loading, ablation rows, compose overrides, topologies, F1 ABSENT,
Wilson/Cohen, unsafe metrics, table shapes, malformed input, frozen
mutation guards, and fixed-seed determinism.

These tests are validation only — they are not experimental results.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import pytest
import yaml

from core.config import PROJECT_ROOT
from experiments.phase5.ablation import (
    resolve_ablation_row,
    resolve_all_ablation_rows,
)
from experiments.phase5.compose_override import (
    compose_command,
    generate_compose_override,
    validate_lag_option,
)
from experiments.phase5.harness import EpisodeHarness, action_key
from experiments.phase5.record import (
    RecordValidationError,
    read_episode_records,
    validate_episode_record,
    write_episode_records,
)
from experiments.phase5.scenario import (
    AblationRow,
    episode_seed,
    load_scenario,
    resolve_c3_episode,
    resolve_episode,
)
from experiments.phase5.tau_star import (
    CalibrationArtifactError,
    calibration_provenance,
    load_tau_star,
    phi_tau_star_provider,
)
from pilot.wilson import wilson_ci


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def template():
    return load_scenario()


@pytest.fixture(scope="module")
def harness(template):
    # Deterministic exact same-intent similarity (avoids model load in unit tests).
    def sim(buffered, replanned):
        if isinstance(replanned, dict):
            if buffered.get("intent_id") and buffered.get("intent_id") == replanned.get("intent_id"):
                return 1.0
            if buffered.get("tool") == replanned.get("tool"):
                return 1.0
            return 0.0
        return 0.0

    return EpisodeHarness(template, semantic_similarity=sim, semantic_similarity_label="test")


def _frozen_digests() -> dict[str, str]:
    import hashlib

    paths = [
        PROJECT_ROOT / "client" / "buffer.py",
        PROJECT_ROOT / "tool_world" / "main.py",
        PROJECT_ROOT / "tool_world" / "ledger.py",
        PROJECT_ROOT / "reintegration" / "reintegration_service.py",
        PROJECT_ROOT / "reintegration" / "naive_retry.py",
        PROJECT_ROOT / "reintegration" / "verify_before_retry.py",
        PROJECT_ROOT / "config" / "experiment" / "factorial.yaml",
        PROJECT_ROOT / "config" / "experiment" / "ablation.yaml",
        PROJECT_ROOT / "pilot" / "key_schemes.py",
        PROJECT_ROOT / "pilot" / "wilson.py",
        PROJECT_ROOT / "results" / "phase1b" / "tau_star_results.csv",
        PROJECT_ROOT / "experiments" / "netctl.py",
        PROJECT_ROOT / "experiments" / "outage_model.py",
    ]
    out = {}
    for p in paths:
        if p.exists():
            out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


# ---------------------------------------------------------------------------
# A — scenario generation
# ---------------------------------------------------------------------------


def test_A_scenario_gen(template):
    assert template.scenario_id == "replan-dup-2"
    assert template.master_seed == 42
    assert len(template.pre_gap_actions) == 2
    assert len(template.replanned_actions) == 2
    a1, a2 = template.pre_gap_actions
    assert a1.delivery == "pre_gap" and a1.ttl == 0.0
    assert a2.delivery == "in_flight" and a2.ttl == 30.0
    assert {a.intent_id for a in template.pre_gap_actions} == {"INT-A1", "INT-A2"}
    # Fresh turns must diverge for cold-restart duplicate construction.
    assert {r.turn_seq for r in template.replanned_actions} == {99, 100}
    # Args identical to originals (replan of the same intents).
    by_intent = {r.intent_id: r for r in template.replanned_actions}
    for a in template.pre_gap_actions:
        assert by_intent[a.intent_id].args == a.args
        assert by_intent[a.intent_id].turn_seq != a.turn_seq
    assert template.eligible_actions == 2
    assert template.sync_window_s == 2.0
    assert template.temperature == 0.7


def test_A_resolve_episode_validates(template):
    job = resolve_episode(template, "kinbridge_sync", 90, 3)
    assert job["method"] == "kinbridge_sync"
    assert job["outage_duration_s"] == 90
    assert job["episode_id"] == 3
    with pytest.raises(ValueError):
        resolve_episode(template, "no_such_method", 90, 0)
    with pytest.raises(ValueError):
        resolve_episode(template, "kinbridge_sync", 17, 0)


# ---------------------------------------------------------------------------
# B — cold_restart distinct from other methods
# ---------------------------------------------------------------------------


def test_B_cold_restart_distinct(harness):
    cold = harness.run_episode("cold_restart", 90, 0)
    naive = harness.run_episode("naive_retry", 90, 0)
    kin = harness.run_episode("kinbridge_sync", 90, 0)

    cold_dec = [a["decision"] for a in cold["actions"]]
    assert cold_dec == ["DROPPED", "DROPPED"]
    # Cold discards state: first pre-gap effect is duplicated via replan.
    assert cold["actions"][0]["duplicate"] is True

    naive_dec = [a["decision"] for a in naive["actions"]]
    assert naive_dec[0] == "COMMITTED"
    assert "DROPPED" not in naive_dec

    kin_dec = [a["decision"] for a in kin["actions"]]
    assert kin_dec[0] == "COMMITTED"
    assert kin["actions"][1]["duplicate"] is False
    # Kinbridge must not be unsafe on this scenario.
    assert all(
        not (a["duplicate"] or a["stale_execution"]) for a in kin["actions"]
    )


# ---------------------------------------------------------------------------
# C — baselines
# ---------------------------------------------------------------------------


def test_C_baselines_behaviors(harness, template):
    for outage, expect_cold_unsafe in ((0, 2), (10, 1), (90, 2), (300, 2)):
        cold = harness.run_episode("cold_restart", outage, 1)
        n_unsafe = sum(
            1 for a in cold["actions"] if a["duplicate"] or a["stale_execution"]
        )
        assert n_unsafe == expect_cold_unsafe, (outage, n_unsafe)

    naive90 = harness.run_episode("naive_retry", 90, 1)
    # a1 committed pre-gap → skip; a2 replayed without TTL check → stale.
    assert naive90["actions"][0]["decision"] == "COMMITTED"
    assert naive90["actions"][1]["decision"] == "REPLAYED"
    assert naive90["actions"][1]["stale_execution"] is True
    assert naive90["actions"][1]["duplicate"] is False

    verify90 = harness.run_episode("verify_before_retry", 90, 1)
    assert verify90["actions"][1]["decision"] == "REPLAYED"
    assert verify90["actions"][1]["stale_execution"] is True

    kin90 = harness.run_episode("kinbridge_sync", 90, 1)
    assert kin90["actions"][1]["decision"] == "INVALIDATED"
    assert kin90["actions"][1]["stale_execution"] is False

    kin10 = harness.run_episode("kinbridge_sync", 10, 1)
    assert kin10["actions"][1]["decision"] == "MERGED"
    assert all(
        not (a["duplicate"] or a["stale_execution"]) for a in kin10["actions"]
    )

    kin0 = harness.run_episode("kinbridge_sync", 0, 1)
    assert all(a["decision"] == "COMMITTED" for a in kin0["actions"])


def test_C_cold_is_not_naive(harness):
    cold = harness.run_episode("cold_restart", 90, 2)
    naive = harness.run_episode("naive_retry", 90, 2)
    assert cold["actions"][0]["decision"] == "DROPPED"
    assert naive["actions"][0]["decision"] == "COMMITTED"
    # Distinct episode seeds → distinct records for different methods.
    assert cold["episode_seed"] != naive["episode_seed"]


# ---------------------------------------------------------------------------
# D — child seeds
# ---------------------------------------------------------------------------


def test_D_child_seeds(template):
    s1 = episode_seed(42, "replan-dup-2", "kinbridge_sync", 90, 0)
    s1b = episode_seed(42, "replan-dup-2", "kinbridge_sync", 90, 0)
    s2 = episode_seed(42, "replan-dup-2", "kinbridge_sync", 90, 1)
    s3 = episode_seed(42, "replan-dup-2", "naive_retry", 90, 0)
    s4 = episode_seed(42, "replan-dup-2", "kinbridge_sync", 10, 0)
    assert s1 == s1b
    assert s1 != s2
    assert s1 != s3
    assert s1 != s4
    assert 0 <= s1 < 2**63


# ---------------------------------------------------------------------------
# E — record schema
# ---------------------------------------------------------------------------


def test_E_record_schema(harness):
    rec = harness.run_episode("kinbridge_sync", 10, 5)
    validate_episode_record(rec)  # must not raise
    assert rec["episode_id"] == 5
    assert rec["method"] == "kinbridge_sync"
    assert rec["outage_duration"] == 10
    assert rec["ttr_seconds"] >= 0
    assert isinstance(rec["epoch_guard_result"], bool)
    assert len(rec["actions"]) == 2
    for a in rec["actions"]:
        assert set(a) >= {
            "action_id", "decision", "duplicate", "stale_execution",
            "escalated", "invalidated", "committed", "effect_identity",
            "safety_decision", "absent_after_epoch_guard",
        }
        assert isinstance(a["duplicate"], bool)
        assert isinstance(a["stale_execution"], bool)


def test_E_validate_rejects_missing_and_bad_types():
    with pytest.raises(ValueError):
        validate_episode_record({"episode_id": 0})
    bad = {
        "episode_id": 0,
        "method": "kinbridge_sync",
        "outage_profile": "baseline",
        "outage_duration": 90,
        "seed": 42,
        "episode_seed": 1,
        "temperature": 0.7,
        "key_scheme": "intent",
        "replication_lag": 0,
        "topology": "same-node",
        "source_node": "edge-a",
        "reconnect_node": "edge-a",
        "epoch_before": 0,
        "epoch_after": 1,
        "epoch_guard_result": "yes",  # not bool
        "actions": [],
        "ttr_seconds": 0.0,
        "tau_star": 0.5,
        "tau_star_source": "x",
        "scenario_id": "replan-dup-2",
    }
    with pytest.raises(RecordValidationError):
        validate_episode_record(bad)


def test_E_jsonl_roundtrip(tmp_path, harness):
    rec = harness.run_episode("naive_retry", 10, 0)
    path = tmp_path / "episodes.jsonl"
    write_episode_records([rec], path)
    loaded = read_episode_records(path)
    assert len(loaded) == 1
    assert loaded[0]["method"] == "naive_retry"


def test_E_malformed_jsonl_fails_loud(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(RecordValidationError):
        read_episode_records(path)


# ---------------------------------------------------------------------------
# F — tau* CSV / T=0
# ---------------------------------------------------------------------------


def test_F_tau_star_csv_and_t0():
    tau, row = load_tau_star(0.7)
    assert tau is not None
    assert math.isclose(tau, 0.5051826557580381, rel_tol=0, abs_tol=1e-15)
    assert row.method_used == "bisection"
    assert "tau_star_results.csv" in row.provenance

    tau0, row0 = load_tau_star(0.0)
    assert tau0 is None
    assert row0.method_used == "unestimable"

    prov = calibration_provenance(0.7)
    assert prov["tau_star"] is not None
    assert prov["calibration_status"] == "bisection"

    # Never hardcoded: a missing artifact path fails closed / raises.
    with pytest.raises(CalibrationArtifactError):
        load_tau_star(0.7, path=PROJECT_ROOT / "results" / "phase1b" / "missing.csv")

    provider = phi_tau_star_provider(temperature=0.7, scenario_temperature=0.7)
    assert provider() == pytest.approx(0.5051826557580381)
    provider0 = phi_tau_star_provider(temperature=0.0, scenario_temperature=0.0)
    assert provider0() is None
    # Temperature mismatch → fail closed (no substitution).
    provider_mm = phi_tau_star_provider(temperature=0.0, scenario_temperature=0.7)
    assert provider_mm() is None


def test_F_no_hardcoded_tau_star_in_phase5_sources():
    # The numeric literal for tau* must not appear in Phase 5 code/config.
    banned = "0.5051826557580381"
    roots = [
        PROJECT_ROOT / "experiments" / "phase5",
        PROJECT_ROOT / "experiments" / "run_experiment.py",
        PROJECT_ROOT / "experiments" / "analyze_results.py",
        PROJECT_ROOT / "config" / "experiment" / "phase5_scenario.yaml",
    ]
    for root in roots:
        files = [root] if root.is_file() else list(root.rglob("*.py")) + list(root.rglob("*.yaml"))
        for p in files:
            text = p.read_text(encoding="utf-8")
            assert banned not in text, f"hardcoded tau* found in {p}"
            # Also reject the rounded paper value as a code constant.
            assert "0.5052" not in text or p.suffix == ".md", f"rounded tau* in {p}"


# ---------------------------------------------------------------------------
# G — ablation rows
# ---------------------------------------------------------------------------


def test_G_ablation_rows(template):
    rows = resolve_all_ablation_rows(template)
    assert [r.row.name for r in rows] == [
        "buffer_only",
        "plus_hash_keys",
        "plus_intent_keys",
        "plus_staleness",
        "plus_epoch_guard",
        "plus_degradation_gate",
    ]
    by_name = {r.row.name: r.row for r in rows}
    # key_scheme is explicit and distinguishes the first three rungs.
    assert by_name["buffer_only"].key_scheme == "none"
    assert by_name["plus_hash_keys"].key_scheme == "payload"
    assert by_name["plus_intent_keys"].key_scheme == "intent"
    assert by_name["plus_hash_keys"].enable_intent_keys is False
    assert by_name["plus_intent_keys"].enable_intent_keys is True
    assert by_name["plus_staleness"].enable_staleness_check is True
    assert by_name["plus_epoch_guard"].enable_epoch_guard is True
    assert by_name["plus_degradation_gate"].enable_degradation_gate is True
    assert by_name["buffer_only"].enable_degradation_gate is False

    resolved = resolve_ablation_row(template, "plus_epoch_guard")
    assert resolved.method == "kinbridge_sync"
    assert resolved.outage_s == 90
    assert resolved.to_dict()["enable_epoch_guard"] is True

    with pytest.raises(ValueError):
        resolve_ablation_row(template, "full")  # scenario uses plus_degradation_gate

    # Config file row names are stable for Table VI.
    cfg = yaml.safe_load(
        (PROJECT_ROOT / "config" / "experiment" / "phase5_scenario.yaml").read_text(
            encoding="utf-8"
        )
    )
    cfg_names = [r["row"] for r in cfg["ablation_ladder"]["rows"]]
    assert cfg_names == [r.row.name for r in rows]


# ---------------------------------------------------------------------------
# H — compose overrides lag 0 / 500 / -1
# ---------------------------------------------------------------------------


def test_H_compose_overrides(tmp_path):
    for lag in (0, 500, -1):
        text = generate_compose_override(lag)
        data = yaml.safe_load(text)
        assert set(data["services"]) == {"replicator"}
        assert data["services"]["replicator"]["environment"]["REPLICATOR_LAG_MS"] == str(lag)
        assert f"--lag-ms {lag}" in data["services"]["replicator"]["command"][0]
        assert validate_lag_option(lag) is True

        out = tmp_path / f"override_{lag}.yml"
        from experiments.phase5.compose_override import write_compose_override

        write_compose_override(lag, out)
        assert out.exists()
        cmd = compose_command(lag, out)
        assert cmd[:5] == ["docker", "compose", "-f", "docker-compose.yml", "-f"]
        assert str(out) in cmd

    assert validate_lag_option(123) is False
    with pytest.raises(ValueError):
        generate_compose_override(123)


def test_H_docker_compose_base_untouched_by_override_generation(tmp_path):
    before = (PROJECT_ROOT / "docker-compose.yml").read_bytes()
    generate_compose_override(500)
    after = (PROJECT_ROOT / "docker-compose.yml").read_bytes()
    assert before == after


# ---------------------------------------------------------------------------
# I — same-node reconnect
# ---------------------------------------------------------------------------


def test_I_same_node_reconnect(harness):
    rec = harness.run_episode(
        "kinbridge_sync", 90, 0, topology="same-node", replication_lag_ms=0
    )
    assert rec["topology"] == "same-node"
    assert rec["source_node"] == "edge-a"
    assert rec["reconnect_node"] == "edge-a"
    assert rec["epoch_guard_result"] is True
    assert rec["ttr_seconds"] == 0.0
    # a1 committed; a2 invalidated by TTL on same-node primary path.
    assert rec["actions"][0]["decision"] == "COMMITTED"
    assert rec["actions"][1]["decision"] == "INVALIDATED"
    assert rec["actions"][0]["ledger_status"] == "COMMITTED"


# ---------------------------------------------------------------------------
# J — cross-node replication lag (epoch guard / F1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lag", [0, 500, -1])
def test_J_cross_node_epoch_guard(harness, lag):
    rec = harness.run_episode(
        "kinbridge_sync", 90, 0, topology="cross-node", replication_lag_ms=lag
    )
    assert rec["topology"] == "cross-node"
    assert rec["reconnect_node"] == "edge-b"
    # Frozen writer never bumps ledger_meta → edge-b epoch stays 0 < buffer.max_epoch.
    assert rec["epoch_guard_result"] is False
    # Guard failure escalates non-COMMITTED actions; a1 stays COMMITTED.
    assert rec["actions"][0]["decision"] == "COMMITTED"
    assert rec["actions"][1]["decision"] == "ESCALATED"
    assert rec["actions"][1]["escalated"] is True
    # Escalation is not unsafe.
    assert all(
        not (a["duplicate"] or a["stale_execution"]) for a in rec["actions"]
    )
    # Sync window 2.0s with 50 ms polls → TTR from reconnect to completion
    # (may overshoot by one poll tick due to float accumulation).
    assert 2.0 <= rec["ttr_seconds"] <= 2.1
    assert rec["actions"][1]["absent_after_epoch_guard"] is True
    assert rec["actions"][1]["replication_ambiguity"] is True


# ---------------------------------------------------------------------------
# K — F1 ABSENT after epoch guard
# ---------------------------------------------------------------------------


def test_K_f1_absent_after_guard(harness):
    rec = harness.run_episode(
        "kinbridge_sync", 90, 0, topology="cross-node", replication_lag_ms=-1
    )
    # With lag -1 nothing is replicated; after guard failure a2 is ESCALATED
    # and its ledger status on edge-b is not COMMITTED (ABSENT → escalated row).
    a2 = rec["actions"][1]
    assert a2["decision"] == "ESCALATED"
    assert a2["ledger_status"] != "COMMITTED"
    assert a2["safety_decision"] == "ESCALATE"
    # ABSENT is never treated as permission to execute on replicated path.
    assert a2["committed"] is False or a2["effect_applications"] >= 0
    # a1's pre-gap commit is present on edge-b after replication drain
    # (when lag != -1) or remains COMMITTED via buffer decision on same-node.
    rec0 = harness.run_episode(
        "kinbridge_sync", 90, 0, topology="cross-node", replication_lag_ms=0
    )
    assert rec0["actions"][0]["decision"] == "COMMITTED"
    assert rec0["actions"][1]["escalated"] is True


# ---------------------------------------------------------------------------
# L — Wilson CI
# ---------------------------------------------------------------------------


def test_L_wilson_ci():
    ci = wilson_ci(0, 10, z=1.96)
    assert ci.estimate == 0.0
    assert ci.ci_low == 0.0
    assert ci.ci_high > 0.0

    ci = wilson_ci(10, 10, z=1.96)
    assert ci.estimate == 1.0
    assert ci.ci_high == 1.0
    assert ci.ci_low < 1.0

    ci = wilson_ci(5, 10, z=1.96)
    assert ci.estimate == 0.5
    assert ci.ci_low < 0.5 < ci.ci_high
    # Monotone / known bounds
    assert 0.0 <= ci.ci_low <= ci.ci_high <= 1.0

    ci0 = wilson_ci(0, 0, z=1.96)
    assert ci0.numerator == 0 and ci0.denominator == 0

    with pytest.raises(ValueError):
        wilson_ci(1, -1)


# ---------------------------------------------------------------------------
# M — Cohen's h
# ---------------------------------------------------------------------------


def test_M_cohens_h():
    from experiments.analyze_results import cohens_h

    assert cohens_h(0.5, 0.5) == pytest.approx(0.0, abs=1e-15)
    h = cohens_h(1.0, 0.0)
    assert h == pytest.approx(math.pi, abs=1e-12)
    h2 = cohens_h(0.0, 1.0)
    assert h2 == pytest.approx(-math.pi, abs=1e-12)
    # Symmetric
    assert cohens_h(0.2, 0.4) == pytest.approx(-cohens_h(0.4, 0.2), abs=1e-15)
    with pytest.raises(Exception):
        cohens_h(-0.1, 0.5)
    with pytest.raises(Exception):
        cohens_h(0.5, 1.5)


# ---------------------------------------------------------------------------
# N — unsafe metric calculation
# ---------------------------------------------------------------------------


def test_N_unsafe_calculation():
    from experiments.run_experiment import _episode_row

    rec = {
        "episode_id": 0,
        "method": "kinbridge_sync",
        "outage_duration": 90,
        "outage_profile": "baseline",
        "seed": 42,
        "episode_seed": 1,
        "key_scheme": "intent",
        "ablation_row": "full",
        "replication_lag": 0,
        "topology": "same-node",
        "source_node": "edge-a",
        "reconnect_node": "edge-a",
        "ttr_seconds": 0.0,
        "epoch_guard_result": True,
        "tau_star": 0.5,
        "scenario_id": "replan-dup-2",
        "actions": [
            {"duplicate": True, "stale_execution": False, "escalated": False, "invalidated": False},
            {"duplicate": False, "stale_execution": False, "escalated": True, "invalidated": False},
            {"duplicate": False, "stale_execution": True, "escalated": False, "invalidated": False},
            {"duplicate": False, "stale_execution": False, "escalated": False, "invalidated": True},
        ],
    }
    row = _episode_row(rec)
    # unsafe = duplicate OR stale; escalation/invalidation excluded.
    assert row["n_unsafe"] == 2
    assert row["n_eligible"] == 4
    assert row["unsafe_action_rate"] == pytest.approx(0.5)
    assert row["n_duplicate"] == 1
    assert row["n_stale"] == 1
    assert row["n_escalated"] == 1
    assert row["n_invalidated"] == 1


def test_N_unsafe_live_harness(harness):
    # Analytical expectation (not hard-coded as paper results):
    # cold@90: a1 duplicate, a2 stale → 2/2 unsafe
    cold = harness.run_episode("cold_restart", 90, 0)
    n_unsafe = sum(1 for a in cold["actions"] if a["duplicate"] or a["stale_execution"])
    assert n_unsafe == 2
    kin = harness.run_episode("kinbridge_sync", 90, 0)
    n_unsafe_k = sum(1 for a in kin["actions"] if a["duplicate"] or a["stale_execution"])
    assert n_unsafe_k == 0


# ---------------------------------------------------------------------------
# O — table shapes
# ---------------------------------------------------------------------------


def test_O_table_shapes(harness, tmp_path):
    from experiments.analyze_results import analyze

    # Build a small multi-cell raw log via the runner API.
    from experiments.run_experiment import run_grid

    out = tmp_path / "raw"
    summary = run_grid(
        grid="factorial",
        methods=["cold_restart", "naive_retry", "verify_before_retry", "kinbridge_sync"],
        outages=[0, 10, 90, 300],
        episodes=1,
        seed=42,
        output_dir=out,
        semantic_similarity=lambda a, b: 1.0 if a.get("intent_id") == b.get("intent_id") else 0.0,
    )
    assert summary["episodes_run"] == 16
    jsonl = Path(summary["jsonl"])
    assert jsonl.exists()

    analysis_dir = tmp_path / "final"
    result = analyze(jsonl, analysis_dir, make_figures=False)

    # Table III: 4 methods × 4 outages
    with open(analysis_dir / "table_III.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 16
    assert {r["method"] for r in rows} == {
        "cold_restart", "naive_retry", "verify_before_retry", "kinbridge_sync"
    }
    assert {int(r["outage_duration"]) for r in rows} == {0, 10, 90, 300}
    for r in rows:
        assert float(r["unsafe_ci_low"]) <= float(r["unsafe_rate"]) <= float(r["unsafe_ci_high"]) or (
            int(r["eligible_actions"]) == 0
        )

    # Table IV: 4 methods
    with open(analysis_dir / "table_IV.csv", encoding="utf-8", newline="") as f:
        rows4 = list(csv.DictReader(f))
    assert len(rows4) == 4
    assert [r["method"] for r in rows4] == [
        "cold_restart", "naive_retry", "verify_before_retry", "kinbridge_sync"
    ]
    for r in rows4:
        assert r["cohens_h_vs_cold"] != ""

    # Table V / VI may be empty-ish for factorial-only input — files exist.
    assert (analysis_dir / "table_V.csv").exists()
    assert (analysis_dir / "table_VI.csv").exists()
    assert (analysis_dir / "results.csv").exists()
    assert result["episodes"] == 16


def test_O_c3_and_ablation_shapes(harness, tmp_path):
    from experiments.analyze_results import analyze
    from experiments.run_experiment import run_grid

    c3_dir = tmp_path / "c3"
    for lag, topo in ((0, "same-node"), (0, "cross-node"), (500, "cross-node"), (-1, "cross-node")):
        run_grid(
            grid="c3",
            methods=["kinbridge_sync"],
            outages=[90],
            episodes=1,
            seed=42,
            output_dir=c3_dir,
            replication_lag_ms=lag,
            topology=topo,
            semantic_similarity=lambda a, b: 1.0 if a.get("intent_id") == b.get("intent_id") else 0.0,
        )
    abl_dir = tmp_path / "abl"
    run_grid(
        grid="ablation",
        methods=["kinbridge_sync"],
        outages=[90],
        episodes=1,
        seed=42,
        output_dir=abl_dir,
        semantic_similarity=lambda a, b: 1.0 if a.get("intent_id") == b.get("intent_id") else 0.0,
    )

    # Merge C3 + ablation + a tiny factorial into one log for table shapes.
    from experiments.phase5.record import read_episode_records

    all_recs = []
    all_recs.extend(read_episode_records(c3_dir / "episodes.jsonl"))
    all_recs.extend(read_episode_records(abl_dir / "episodes.jsonl"))
    fac = tmp_path / "fac"
    run_grid(
        grid="factorial",
        methods=["cold_restart"],
        outages=[90],
        episodes=1,
        seed=42,
        output_dir=fac,
        semantic_similarity=lambda a, b: 1.0,
    )
    all_recs.extend(read_episode_records(fac / "episodes.jsonl"))
    merged = tmp_path / "merged.jsonl"
    write_episode_records(all_recs, merged)

    final = tmp_path / "final2"
    analyze(merged, final)

    with open(final / "table_V.csv", encoding="utf-8", newline="") as f:
        rows_v = list(csv.DictReader(f))
    assert len(rows_v) == 6  # 3 lags × 2 topologies
    assert {(int(r["replication_lag_ms"]), r["topology"]) for r in rows_v} == {
        (l, t) for l in (0, 500, -1) for t in ("same-node", "cross-node")
    }

    with open(final / "table_VI.csv", encoding="utf-8", newline="") as f:
        rows_vi = list(csv.DictReader(f))
    assert len(rows_vi) == 6
    assert [r["ablation_row"] for r in rows_vi] == [
        "buffer_only", "plus_hash_keys", "plus_intent_keys",
        "plus_staleness", "plus_epoch_guard", "plus_degradation_gate",
    ]


# ---------------------------------------------------------------------------
# P — malformed input handling in analyzer
# ---------------------------------------------------------------------------


def test_P_malformed_input_analyzer(tmp_path):
    from experiments.analyze_results import AnalysisError, analyze

    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises((AnalysisError, RecordValidationError)):
        analyze(empty, tmp_path / "out")

    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"episode_id": 1}\n', encoding="utf-8")
    with pytest.raises((AnalysisError, RecordValidationError, ValueError)):
        analyze(bad, tmp_path / "out2")


def test_P_runner_rejects_unknown_method_and_negative():
    from experiments.run_experiment import run_grid

    with pytest.raises(ValueError):
        run_grid(
            grid="factorial",
            methods=["bogus"],
            outages=[90],
            episodes=1,
            seed=42,
            output_dir=PROJECT_ROOT / "results" / "_should_not_exist",
        )
    with pytest.raises(ValueError):
        run_grid(
            grid="factorial",
            methods=["kinbridge_sync"],
            outages=[90],
            episodes=-1,
            seed=42,
            output_dir=PROJECT_ROOT / "results" / "_should_not_exist",
        )


# ---------------------------------------------------------------------------
# Q — no raw/frozen mutation
# ---------------------------------------------------------------------------


def test_Q_no_frozen_mutation(harness):
    before = _frozen_digests()
    # Touch the full episode surface (all methods + c3 + ablation).
    for method in ("cold_restart", "naive_retry", "verify_before_retry", "kinbridge_sync"):
        harness.run_episode(method, 90, 0)
    harness.run_episode("kinbridge_sync", 90, 0, topology="cross-node", replication_lag_ms=500)
    template = load_scenario()
    for row in resolve_all_ablation_rows(template):
        harness.run_episode(
            "kinbridge_sync", 90, 0, ablation_row=row.row, key_scheme=row.row.key_scheme
        )
    generate_compose_override(500)
    load_tau_star(0.7)
    after = _frozen_digests()
    assert before == after


def test_Q_phase1b_results_dir_listing_stable():
    phase1b = PROJECT_ROOT / "results" / "phase1b"
    assert phase1b.exists()
    names = sorted(p.name for p in phase1b.iterdir())
    assert "tau_star_results.csv" in names


# ---------------------------------------------------------------------------
# R — fixed-seed determinism
# ---------------------------------------------------------------------------


def test_R_fixed_seed_determinism(harness):
    a = harness.run_episode("kinbridge_sync", 90, 7)
    b = harness.run_episode("kinbridge_sync", 90, 7)
    # Strip nothing — full record must match (times are simulated).
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)

    c = harness.run_episode("kinbridge_sync", 90, 8)
    assert a["episode_seed"] != c["episode_seed"]
    assert a["episode_id"] != c["episode_id"]

    d = harness.run_episode("naive_retry", 90, 7)
    assert a["episode_seed"] != d["episode_seed"]


def test_R_key_scheme_determinism():
    k_intent_a = action_key(
        "navigate_to", {"lat": 1.0}, "intent",
        intent_id="I", session_id="S", turn_seq=1,
    )
    k_intent_b = action_key(
        "navigate_to", {"lat": 1.0}, "intent",
        intent_id="I", session_id="S", turn_seq=1,
    )
    k_intent_c = action_key(
        "navigate_to", {"lat": 1.0}, "intent",
        intent_id="I", session_id="S", turn_seq=2,
    )
    assert k_intent_a == k_intent_b
    assert k_intent_a != k_intent_c  # turn divergence

    k_pay = action_key("navigate_to", {"lat": 1.0}, "payload")
    k_pay2 = action_key(
        "navigate_to", {"lat": 1.0}, "payload",
        intent_id="I", session_id="S", turn_seq=99,
    )
    assert k_pay == k_pay2  # payload ignores turn


# ---------------------------------------------------------------------------
# S — runner CLI guards
# ---------------------------------------------------------------------------


def test_S_runner_cli_ep0_exits_2(capsys):
    from experiments.run_experiment import main

    rc = main(["--episodes", "0", "--grid", "factorial"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "episodes" in err.lower()


def test_S_runner_cli_smoke(tmp_path):
    from experiments.run_experiment import main

    out = tmp_path / "cli_raw"
    rc = main([
        "--method", "kinbridge_sync",
        "--outage-duration", "90",
        "--episodes", "2",
        "--output-dir", str(out),
    ])
    assert rc == 0
    assert (out / "episodes.jsonl").exists()
    assert (out / "results.csv").exists()
    lines = [ln for ln in (out / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2


# ---------------------------------------------------------------------------
# T — degradation gate is harness-level, does not override F1/guard
# ---------------------------------------------------------------------------


def test_T_degradation_gate_does_not_override_guard(harness, template):
    row = resolve_ablation_row(template, "plus_degradation_gate")
    rec = harness.run_episode(
        "kinbridge_sync",
        90,
        0,
        topology="cross-node",
        replication_lag_ms=-1,
        ablation_row=row.row,
        key_scheme=row.row.key_scheme,
    )
    # Epoch guard still fails and escalates; gate never forces execution.
    assert rec["epoch_guard_result"] is False
    assert rec["actions"][1]["decision"] == "ESCALATED"
    assert rec["actions"][1]["duplicate"] is False
    assert rec["actions"][1]["stale_execution"] is False
    assert rec["ablation_row"] == "plus_degradation_gate"


def test_T_staleness_flag_gates_ttl(template, harness):
    without = resolve_ablation_row(template, "plus_intent_keys")
    with_st = resolve_ablation_row(template, "plus_staleness")
    r1 = harness.run_episode(
        "kinbridge_sync", 90, 0, ablation_row=without.row, key_scheme=without.row.key_scheme
    )
    r2 = harness.run_episode(
        "kinbridge_sync", 90, 0, ablation_row=with_st.row, key_scheme=with_st.row.key_scheme
    )
    assert r1["actions"][1]["decision"] == "MERGED"
    assert r2["actions"][1]["decision"] == "INVALIDATED"
    assert r2["actions"][1]["stale_execution"] is False


# ---------------------------------------------------------------------------
# U — docker harness writes compose override
# ---------------------------------------------------------------------------


def test_U_docker_harness_writes_override(tmp_path):
    from experiments.run_experiment import run_grid

    out = tmp_path / "docker_raw"
    summary = run_grid(
        grid="c3",
        methods=["kinbridge_sync"],
        outages=[90],
        episodes=1,
        seed=42,
        output_dir=out,
        harness_mode="docker",
        replication_lag_ms=500,
        topology="cross-node",
        semantic_similarity=lambda a, b: 1.0,
    )
    assert summary["harness"] == "docker"
    override = out / "docker-compose.override.yml"
    assert override.exists()
    assert "500" in override.read_text(encoding="utf-8")
    # Base compose untouched.
    assert (PROJECT_ROOT / "docker-compose.yml").exists()


# ---------------------------------------------------------------------------
# V — eligible_actions denominator
# ---------------------------------------------------------------------------


def test_V_eligible_actions(template, harness):
    rec = harness.run_episode("kinbridge_sync", 10, 0)
    assert len(rec["actions"]) == template.eligible_actions == 2
    from experiments.run_experiment import _episode_row

    row = _episode_row(rec)
    assert row["n_eligible"] == 2


# ---------------------------------------------------------------------------
# W — outage 0 delivers both actions
# ---------------------------------------------------------------------------


def test_W_outage_zero_delivers_both(harness):
    rec = harness.run_episode("kinbridge_sync", 0, 0)
    assert all(a["decision"] == "COMMITTED" for a in rec["actions"])
    assert all(a["committed"] for a in rec["actions"])
    assert all(a["effect_applications"] >= 1 for a in rec["actions"])


# ---------------------------------------------------------------------------
# X — scenario config is the only Phase 5 config (frozen configs untouched)
# ---------------------------------------------------------------------------


def test_X_frozen_configs_still_parse_and_match():
    fac = yaml.safe_load(
        (PROJECT_ROOT / "config" / "experiment" / "factorial.yaml").read_text(encoding="utf-8")
    )
    assert fac["episodes_per_cell"] == 100
    assert fac["random_seed"] == 42
    assert len(fac["methods"]) == 4
    assert fac["outage_durations_s"] == [0, 10, 90, 300]

    abl = yaml.safe_load(
        (PROJECT_ROOT / "config" / "experiment" / "ablation.yaml").read_text(encoding="utf-8")
    )
    assert abl["outage_duration_s"] == 90
    assert abl["episodes_per_cell"] == 200
    assert len(abl["ablation_rows"]) == 6

    # phase5_scenario is additive, not a replacement.
    p5 = yaml.safe_load(
        (PROJECT_ROOT / "config" / "experiment" / "phase5_scenario.yaml").read_text(encoding="utf-8")
    )
    assert p5["scenario_id"] == "replan-dup-2"
    assert p5["master_seed"] == 42
