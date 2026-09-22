"""Redis-to-SQLite bridge: consumes replication events and upserts into edge-b's ledger.

Reads replication events from Redis B's ``kinbridge:ledger`` stream and
reconstructs the corresponding ledger record in edge-b's ``effects.db``.

Uses ``INSERT ... ON CONFLICT(key) DO UPDATE`` for idempotent upserts.

Usage:
    python -m experiments.replication.sqlite_writer

Requires:
    - REDIS_B_URL env var (default: ``redis://redis-b:6379/0``)
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
from typing import Any

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

STREAM_KEY = "kinbridge:ledger"
CONSUMER_GROUP = "writer-group"
CONSUMER_NAME = "writer-1"
BATCH_SIZE = 10
POLL_INTERVAL_S = 0.05

_UPSERT_SQL = """
INSERT INTO effect_ledger
    (key, tool, args_hash, status, created_at, committed_at, epoch, ttl, intent_id)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(key) DO UPDATE SET
    status = excluded.status,
    committed_at = excluded.committed_at,
    epoch = excluded.epoch
"""


class SQLiteWriter:
    """Consume replication events from Redis B and upsert into edge-b's SQLite ledger.

    Parameters
    ----------
    db_path : str
        Path to the edge-b SQLite database file.
    redis_url : str
        Redis B connection URL.
    """

    def __init__(self, db_path: str, redis_url: str) -> None:
        self._db_path = db_path
        self._redis_url = redis_url
        self._redis: aioredis.Redis | None = None
        self._conn: sqlite3.Connection = self._open_sqlite(db_path)
        self._running = False

    @staticmethod
    def _open_sqlite(db_path: str) -> sqlite3.Connection:
        conn = sqlite3.connect(db_path, timeout=10.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    async def start(self) -> None:
        """Open Redis connection and enter the consumption loop."""
        logger.info(
            "Starting writer: db=%s, redis=%s",
            self._db_path,
            self._redis_url,
        )

        self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
        await self._ensure_consumer_group()

        self._running = True
        await self._loop()

    async def stop(self) -> None:
        """Gracefully shut down the writer."""
        self._running = False
        if self._redis is not None:
            await self._redis.aclose()
        if self._conn is not None:
            self._conn.close()
        logger.info("Writer stopped.")

    async def _ensure_consumer_group(self) -> None:
        """Create the consumer group if it doesn't exist."""
        try:
            await self._redis.xgroup_create(
                STREAM_KEY, CONSUMER_GROUP, id="0", mkstream=True
            )
            logger.info("Created consumer group %s", CONSUMER_GROUP)
        except aioredis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def _loop(self) -> None:
        """Main consumption loop: read events from Redis B, upsert into SQLite."""
        while self._running:
            try:
                entries = await self._redis.xreadgroup(
                    CONSUMER_GROUP,
                    CONSUMER_NAME,
                    {STREAM_KEY: ">"},
                    count=BATCH_SIZE,
                    block=int(POLL_INTERVAL_S * 1000),
                )
            except aioredis.ConnectionError:
                logger.warning("Redis B unavailable, retrying in 1s...")
                await asyncio.sleep(1.0)
                continue
            except aioredis.ResponseError as exc:
                if "NOGROUP" in str(exc):
                    logger.warning("Consumer group lost, recreating...")
                    await self._ensure_consumer_group()
                    continue
                raise

            if not entries:
                continue

            for _stream, messages in entries:
                for entry_id, fields in messages:
                    await self._process_event(entry_id, fields)

    async def _process_event(self, entry_id: str, fields: dict[str, str]) -> None:
        """Deserialize and upsert a single replication event into SQLite."""
        try:
            record = self._deserialize_event(fields)
        except (KeyError, ValueError) as exc:
            logger.warning("Malformed event %s, skipping: %s", entry_id, exc)
            await self._acknowledge(entry_id)
            return

        try:
            self._upsert_record(record)
            await self._acknowledge(entry_id)
            logger.debug("Upserted key=%s status=%s", record["key"], record["status"])
        except sqlite3.Error as exc:
            logger.error("SQLite upsert failed for key=%s: %s", record["key"], exc)

    def _deserialize_event(self, fields: dict[str, str]) -> dict[str, Any]:
        """Convert Redis stream fields back to a ledger record dict."""
        committed_at = fields.get("committed_at", "")
        intent_id = fields.get("intent_id", "")

        return {
            "key": fields["key"],
            "tool": fields["tool"],
            "args_hash": fields["args_hash"],
            "status": fields["status"],
            "created_at": float(fields["created_at"]),
            "committed_at": float(committed_at) if committed_at else None,
            "epoch": int(fields["epoch"]),
            "ttl": float(fields["ttl"]),
            "intent_id": intent_id if intent_id else None,
        }

    def _upsert_record(self, record: dict[str, Any]) -> None:
        """Insert or update a ledger record in edge-b's SQLite database."""
        with self._conn:
            self._conn.execute(
                _UPSERT_SQL,
                (
                    record["key"],
                    record["tool"],
                    record["args_hash"],
                    record["status"],
                    record["created_at"],
                    record["committed_at"],
                    record["epoch"],
                    record["ttl"],
                    record["intent_id"],
                ),
            )

    async def _acknowledge(self, entry_id: str) -> None:
        """Acknowledge the processed event in Redis B's consumer group."""
        try:
            await self._redis.xack(STREAM_KEY, CONSUMER_GROUP, entry_id)
        except Exception as exc:
            logger.warning("Failed to ack %s: %s", entry_id, exc)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kinbridge-Sync SQLite writer (Phase 3b)"
    )
    parser.add_argument(
        "--db-path",
        default=os.environ.get("KINBRIDGE_DB_PATH", "tool_world/effects.db"),
        help="Path to the SQLite database file",
    )
    parser.add_argument(
        "--redis-url",
        default=os.environ.get("REDIS_B_URL", "redis://redis-b:6379/0"),
        help="Redis B connection URL",
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

    writer = SQLiteWriter(
        db_path=args.db_path,
        redis_url=args.redis_url,
    )

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(writer.stop()))

    try:
        await writer.start()
    except KeyboardInterrupt:
        await writer.stop()


if __name__ == "__main__":
    asyncio.run(main())
