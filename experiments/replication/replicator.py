"""Redis-to-Redis replication forwarder with configurable lag.

Reads replication events from Redis A (kinbridge:ledger stream),
applies an optional artificial delay, and writes them to Redis B.

Usage:
    python -m experiments.replication.replicator --lag-ms 0
    python -m experiments.replication.replicator --lag-ms 500
    python -m experiments.replication.replicator --lag never

Stream: kinbridge:ledger (Redis Streams, XADD/XREADGROUP)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from typing import Any

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

STREAM_KEY = "kinbridge:ledger"
CONSUMER_GROUP = "replicator-group"
CONSUMER_NAME = "replicator-1"
BATCH_SIZE = 10
POLL_INTERVAL_S = 0.05


class Replicator:
    """Forward replication events from Redis A to Redis B with configurable lag.

    Parameters
    ----------
    redis_a_url : str
        Redis A connection URL (e.g. ``redis://redis-a:6379/0``).
    redis_b_url : str
        Redis B connection URL (e.g. ``redis://redis-b:6380/0``).
    lag_ms : int
        Artificial replication delay in milliseconds.
        ``0`` = synchronous, ``-1`` = never (do not start).
    """

    def __init__(self, redis_a_url: str, redis_b_url: str, lag_ms: int) -> None:
        self._redis_a_url = redis_a_url
        self._redis_b_url = redis_b_url
        self._lag_ms = lag_ms
        self._redis_a: aioredis.Redis | None = None
        self._redis_b: aioredis.Redis | None = None
        self._last_id: str = "0-0"
        self._running = False

    async def start(self) -> None:
        """Connect to both Redis instances and enter the forwarding loop."""
        if self._lag_ms == -1:
            logger.info("Replication disabled (--lag never). Exiting.")
            return

        logger.info(
            "Starting replicator: A=%s → B=%s, lag=%dms",
            self._redis_a_url,
            self._redis_b_url,
            self._lag_ms,
        )

        self._redis_a = aioredis.from_url(
            self._redis_a_url, decode_responses=True
        )
        self._redis_b = aioredis.from_url(
            self._redis_b_url, decode_responses=True
        )

        await self._ensure_stream(self._redis_a)
        await self._ensure_stream(self._redis_b)

        self._running = True
        await self._loop()

    async def stop(self) -> None:
        """Gracefully shut down the replicator."""
        self._running = False
        if self._redis_a is not None:
            await self._redis_a.aclose()
        if self._redis_b is not None:
            await self._redis_b.aclose()
        logger.info("Replicator stopped.")

    async def _ensure_stream(self, r: aioredis.Redis) -> None:
        """Create the stream and consumer group if they don't exist."""
        try:
            await r.xgroup_create(
                STREAM_KEY, CONSUMER_GROUP, id="0", mkstream=True
            )
            logger.info("Created consumer group %s on %s", CONSUMER_GROUP, r.connection_pool)
        except aioredis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def _loop(self) -> None:
        """Main forwarding loop: read from A, lag, write to B."""
        while self._running:
            try:
                entries = await self._redis_a.xreadgroup(
                    CONSUMER_GROUP,
                    CONSUMER_NAME,
                    {STREAM_KEY: ">"},
                    count=BATCH_SIZE,
                    block=int(POLL_INTERVAL_S * 1000),
                )
            except aioredis.ConnectionError:
                logger.warning("Redis A unavailable, retrying in 1s...")
                await asyncio.sleep(1.0)
                continue
            except aioredis.ResponseError as exc:
                if "NOGROUP" in str(exc):
                    logger.warning("Consumer group lost, recreating...")
                    await self._ensure_stream(self._redis_a)
                    await self._ensure_stream(self._redis_b)
                    continue
                raise

            if not entries:
                continue

            for _stream, messages in entries:
                for entry_id, fields in messages:
                    await self._apply_lag()
                    await self._forward_event(entry_id, fields)
                    await self._acknowledge(entry_id)

    async def _apply_lag(self) -> None:
        """Sleep for the configured replication lag."""
        if self._lag_ms > 0:
            await asyncio.sleep(self._lag_ms / 1000.0)

    async def _forward_event(self, entry_id: str, fields: dict[str, str]) -> None:
        """Write an event to Redis B's stream."""
        try:
            await self._redis_b.xadd(
                STREAM_KEY,
                fields,
                id="*",
            )
            logger.debug("Forwarded event %s to Redis B", entry_id)
        except aioredis.ConnectionError:
            logger.warning("Redis B unavailable, will retry on next batch")

    async def _acknowledge(self, entry_id: str) -> None:
        """Acknowledge the processed event in Redis A's consumer group."""
        try:
            await self._redis_a.xack(STREAM_KEY, CONSUMER_GROUP, entry_id)
            self._last_id = entry_id
        except aioredis.ConnectionError:
            logger.warning("Failed to ack %s in Redis A", entry_id)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kinbridge-Sync replication forwarder (Phase 3b)"
    )
    parser.add_argument(
        "--lag-ms",
        type=int,
        default=0,
        help="Replication lag in ms (0=synchronous, -1=never)",
    )
    parser.add_argument(
        "--redis-a-url",
        default=os.environ.get("REDIS_A_URL", "redis://redis-a:6379/0"),
        help="Redis A connection URL",
    )
    parser.add_argument(
        "--redis-b-url",
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

    replicator = Replicator(
        redis_a_url=args.redis_a_url,
        redis_b_url=args.redis_b_url,
        lag_ms=args.lag_ms,
    )

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(replicator.stop()))

    try:
        await replicator.start()
    except KeyboardInterrupt:
        await replicator.stop()


if __name__ == "__main__":
    asyncio.run(main())
