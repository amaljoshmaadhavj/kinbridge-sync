"""
Deterministic bucket-boundary sanity check.

Demonstrates the paper's provable failure of the bucket scheme:
  floor((t + G) / Delta) != floor(t / Delta)
when the outage length G exceeds the bucket width Delta.

This is a mathematical assertion, NOT an LLM experiment.

Usage:
    python -m pilot.test_bucket_key
"""

from __future__ import annotations

import sys

from pilot.key_schemes import bucket_index, k_bucket


def test_bucket_transition(
    t: float = 100.0,
    G: float = 90.0,
    Delta: float = 60.0,
    agent_id: str = "test-agent",
    tool: str = "navigate_to",
    args: dict | None = None,
) -> None:
    """Verify that a bucket transition occurs across the outage gap.

    Preconditions ( caller must ensure ):
        floor(t / Delta) != floor((t + G) / Delta)
    """
    if args is None:
        args = {"lat": 45.0, "lon": 6.0, "mode": "fastest"}

    bucket_before = bucket_index(t, Delta)
    bucket_after = bucket_index(t + G, Delta)

    key_before = k_bucket(agent_id, tool, args, t, Delta)
    key_after = k_bucket(agent_id, tool, args, t + G, Delta)

    print(f"t = {t}s,  G = {G}s,  Delta = {Delta}s")
    print(f"floor(t / Delta)         = {bucket_before}")
    print(f"floor((t + G) / Delta)   = {bucket_after}")
    print(f"Bucket changed: {bucket_before} -> {bucket_after}  (transition: {bucket_before != bucket_after})")
    print(f"Key before: {key_before}")
    print(f"Key after:  {key_after}")
    print(f"Keys match: {key_before == key_after}")
    print()

    assert bucket_before != bucket_after, (
        f"Expected bucket transition: floor({t}/{Delta})={bucket_before} "
        f"!= floor({t+G}/{Delta})={bucket_after}"
    )
    assert key_before != key_after, (
        "Expected different keys when bucket changes"
    )

    print("PASS: Bucket transition assertion holds.")
    print("  This demonstrates the paper's claim that k_bucket mismatches")
    print("  even on identical payloads when G > Delta.")


def test_same_bucket_no_change(
    t: float = 100.0,
    G: float = 10.0,
    Delta: float = 60.0,
    agent_id: str = "test-agent",
    tool: str = "navigate_to",
    args: dict | None = None,
) -> None:
    """Verify that an outage fully contained in one bucket does NOT change the key.

    Preconditions ( caller must ensure ):
        floor(t / Delta) == floor((t + G) / Delta)
    """
    if args is None:
        args = {"lat": 45.0, "lon": 6.0, "mode": "fastest"}

    bucket_before = bucket_index(t, Delta)
    bucket_after = bucket_index(t + G, Delta)

    key_before = k_bucket(agent_id, tool, args, t, Delta)
    key_after = k_bucket(agent_id, tool, args, t + G, Delta)

    print(f"t = {t}s,  G = {G}s,  Delta = {Delta}s")
    print(f"floor(t / Delta)       = {bucket_before}")
    print(f"floor((t + G) / Delta) = {bucket_after}")
    print(f"Bucket changed: {bucket_before != bucket_after}")
    print(f"Keys match: {key_before == key_after}")
    print()

    assert bucket_before == bucket_after, (
        f"Expected same bucket: floor({t}/{Delta})={bucket_before} "
        f"== floor({t+G}/{Delta})={bucket_after}"
    )
    assert key_before == key_after, "Expected same keys within same bucket"

    print("PASS: Same-bucket consistency holds.")


def main() -> None:
    print("=" * 70)
    print("BUCKET KEY SANITY CHECK -- deterministic mathematical test")
    print("=" * 70)
    print()

    test_bucket_transition()
    print()
    test_same_bucket_no_change()

    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
