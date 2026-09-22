"""SQLite-to-Redis bridge: polls edge-a's ledger and publishes replication events.

Opens a read-only SQLite connection to the edge-a ``effects.db`` ledger,
polls for new or updated records, and publishes each as a JSON event to
Redis A's ``kinbridge:ledger`` stream.

Usage:
    python -m experiments.replication.sqlite_watcher

Requires:
    - REDIS_A_URL env var (default: ``redis://redis-a:6379/0``)
    - KINBRIDGE_DB_PATH env var (default: ``tool_world/effects.db``)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sqlite3
import sys
import time
from typing import Any

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

STREAM_KEY = "kinbridge:ledger"
DEFAULT_POLL_INTERVAL_S = 0.05
DEFAULT_DB_PATH = "tool_world/effects.db"

# All columns in effect_ledger, in insertion order
_COLUMNS = (
    "key", "tool", "args_hash", "status",
    "created_at", "committed_at", "epoch", "ttl", "intent_id",
)


class SQLiteWatcher:
    """Poll an SQLite ledger and publish new/updated records to Redis.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
    redis_url : str
        Redis connection URL.
    poll_interval_s : float
        Seconds between polling cycles.
    """

    def __init__(
        self,
        db_path: str,
        redis_url: str,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    ) -> None:
        self._db_path = db_path
        self._redis_url = redis_url
        self._poll_interval_s = poll_interval_s
        self._redis: aioredis.Redis | None = None
        self._conn: sqlite3.Connection = self._open_sqlite(db_path)
        self._last_created_at: float = 0.0
        self._last_committed_at: float = 0.0
        self._running = False

    @staticmethod
    def _open_sqlite(db_path: str) -> sqlite3.Connection:
        conn = sqlite3.connect(db_path, timeout=10.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS effect_ledger (
                key          TEXT PRIMARY KEY,
                tool         TEXT NOT NULL,
                args_hash    TEXT NOT NULL,
                status       TEXT NOT NULL DEFAULT 'ABSENT',
                created_at   REAL NOT NULL,
                committed_at REAL,
                epoch        INTEGER NOT NULL DEFAULT 0,
                ttl          REAL NOT NULL DEFAULT 0.0,
                intent_id    TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_effect_ledger_tool ON effect_ledger(tool)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_effect_ledger_status ON effect_ledger(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_effect_ledger_intent ON effect_ledger(intent_id)")
        conn.commit()
        return conn

    async def start(self) -> None:
        """Open Redis connection and enter the polling loop."""
        logger.info(
            "Starting watcher: db=%s, redis=%s, poll=%.3fs",
            self._db_path,
            self._redis_url,
            self._poll_interval_s,
        )

        self._redis = aioredis.from_url(self._redis_url, decode_responses=True)

        self._running = True
        await self._loop()

    async def stop(self) -> None:
        """Gracefully shut down the watcher."""
        self._running = False
        if self._redis is not None:
            await self._redis.aclose()
        if self._conn is not None:
            self._conn.close()
        logger.info("Watcher stopped.")

    async def _loop(self) -> None:
        """Main polling loop: detect new records, publish to Redis."""
        while self._running:
            try:
                new_records = self._poll_new_records()
                update_records = self._poll_status_updates()

                all_records = new_records + update_records

                if all_records:
                    for record in all_records:
                        await self._publish_event(record)
                    logger.info(
                        "Published %d events (new=%d, updates=%d)",
                        len(all_records),
                        len(new_records),
                        len(update_records),
                    )

                await asyncio.sleep(self._poll_interval_s)
            except sqlite3.OperationalError as exc:
                logger.warning("SQLite error during poll: %s", exc)
                await asyncio.sleep(1.0)
            except Exception as exc:
                logger.error("Unexpected watcher error: %s", exc)
                await asyncio.sleep(1.0)

    def _poll_new_records(self) -> list[dict[str, Any]]:
        """Query for records created since the last poll."""
        rows = self._conn.execute(
            "SELECT * FROM effect_ledger WHERE created_at > ? ORDER BY created_at ASC",
            (self._last_created_at,),
        ).fetchall()

        records = [dict(row) for row in rows]

        if records:
            max_ts = max(r["created_at"] for r in records)
            if max_ts > self._last_created_at:
                self._last_created_at = max_ts

        return records

    def _poll_status_updates(self) -> list[dict[str, Any]]:
        """Query for records whose status changed since the last poll.

        Detects status transitions (e.g. ABSENT → COMMITTED) by checking
        committed_at, which is set only on transition to COMMITTED.
        """
        rows = self._conn.execute(
            """SELECT * FROM effect_ledger
               WHERE committed_at > ? AND status = 'COMMITTED'
               ORDER BY committed_at ASC""",
            (self._last_committed_at,),
        ).fetchall()

        records = []
        for row in rows:
            record = dict(row)
            # Only include if this is genuinely a new update
            # (committed_at > last_seen AND created_at <= last_seen)
            if record["created_at"] <= self._last_created_at:
                records.append(record)

        if records:
            max_ts = max(r["committed_at"] for r in records)
            if max_ts > self._last_committed_at:
                self._last_committed_at = max_ts

        return records

    async def _publish_event(self, record: dict[str, Any]) -> None:
        """Serialize a ledger record and publish it to Redis A's stream."""
        event = {
            "key": record["key"],
            "tool": record["tool"],
            "args_hash": record["args_hash"],
            "status": record["status"],
            "created_at": str(record["created_at"]),
            "committed_at": str(record["committed_at"]) if record["committed_at"] is not None else "",
            "epoch": str(record["epoch"]),
            "ttl": str(record["ttl"]),
            "intent_id": record["intent_id"] or "",
        }

        try:
            await self._redis.xadd(STREAM_KEY, event, id="*")
            logger.debug("Published event for key=%s status=%s", event["key"], event["status"])
        except Exception as exc:
            logger.error("Failed to publish event for key=%s: %s", event["key"], exc)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kinbridge-Sync SQLite watcher (Phase 3b)"
    )
    parser.add_argument(
        "--db-path",
        default=os.environ.get("KINBRIDGE_DB_PATH", DEFAULT_DB_PATH),
        help="Path to the SQLite database file",
    )
    parser.add_argument(
        "--redis-url",
        default=os.environ.get("REDIS_A_URL", "redis://redis-a:6379/0"),
        help="Redis A connection URL",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=float(os.environ.get("WATCHER_POLL_INTERVAL", str(DEFAULT_POLL_INTERVAL_S))),
        help="Polling interval in seconds",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        help="Logging level",
    )
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    watcher = SQLiteWatcher(
        db_path=args.db_path,
        redis_url=args.redis_url,
        poll_interval_s=args.poll_interval,
    )

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(watcher.stop()))

    try:
        await watcher.start()
    except KeyboardInterrupt:
        await watcher.stop()


if __name__ == "__main__":
    asyncio.run(main())
