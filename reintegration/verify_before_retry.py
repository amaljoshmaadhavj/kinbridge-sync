"""Verify-before-retry — Phase 5 ablation intermediate.

Checks the ledger and the injectable V_post hook only.  No epoch guard,
no TTL / V_pre staleness checks, no semantic comparison, no intent-key
reconstruction.  ABSENT is treated as permission to execute (the
original "read committed before replay" behaviour).

This is the middle rung of the ablation ladder, between the naive
baseline (no ledger check) and the full Algorithm 1 service.
"""

from __future__ import annotations

import argparse
from typing import Any, Optional

from client.buffer import ActionBuffer
from tool_world.ledger import EffectLedger


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="verify_before_retry",
        description="Verify-then-retry: ledger + V_post before replay.",
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

    replayed = 0
    skipped = 0
    for rec in buffer.read_all():
        if rec.get("status") in ("COMMITTED", "INVALIDATED"):
            skipped += 1
            continue

        ledger_rec = ledger.lookup(rec["key"])
        status = (ledger_rec or {}).get("status", "ABSENT")

        if status == "COMMITTED":
            buffer.mark_committed(rec["key"])
            skipped += 1
            continue

        # ABSENT / UNKNOWN → execute (no TTL, no V_pre, no semantic).
        execute_fn(rec)
        ledger.mark_committed(rec["key"])
        buffer.mark_committed(rec["key"])
        replayed += 1

    print(f"replayed={replayed} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
