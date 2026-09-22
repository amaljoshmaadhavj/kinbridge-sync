"""Naive Retry — Phase 5 ablation baseline (Algorithm 1 with no safeguards).

``check_ledger=False``, all feature flags off.  Every buffered action is
executed regardless of ledger state, staleness, or connectivity.  This
is the "replay everything blindly" baseline that Algorithm 1 improves on.

Execution uses the action's own key (no intent-key reconstruction), and
the ledger is not consulted before replay.
"""

from __future__ import annotations

import argparse
from typing import Any, Optional

from client.buffer import ActionBuffer
from tool_world.ledger import EffectLedger


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="naive_retry",
        description="Naive replay baseline: execute every buffered action.",
    )
    parser.add_argument("--buffer", required=True, help="Path to the JSONL action buffer")
    parser.add_argument("--ledger-db", required=True, help="Path to the SQLite ledger")
    parser.add_argument("--edge-b-url", default="http://edge-b:8001")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    buffer = ActionBuffer(args.buffer)
    ledger = EffectLedger(db_path=args.ledger_db)

    import httpx

    def execute_fn(payload: dict[str, Any]) -> None:
        tool = payload.get("tool")
        body = dict(payload.get("args", {}))
        resp = httpx.post(f"{args.edge_b_url}/tools/{tool}", json=body, timeout=10.0)
        resp.raise_for_status()

    count = 0
    for rec in buffer.read_all():
        if rec.get("status") in ("COMMITTED", "INVALIDATED"):
            continue
        execute_fn(rec)  # blindly replay, ledger not consulted
        ledger.mark_committed(rec["key"])
        buffer.mark_committed(rec["key"])
        count += 1
    print(f"replayed={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
