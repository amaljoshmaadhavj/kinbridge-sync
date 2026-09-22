"""Tests for the Redis-to-Redis replicator (Phase 3b).

Uses unittest.mock to avoid requiring a running Redis instance.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from experiments.replication.replicator import Replicator, parse_args


# ---------------------------------------------------------------------------
# Replicator — unit tests
# ---------------------------------------------------------------------------


class TestReplicatorInit:
    def test_default_lag_ms(self) -> None:
        r = Replicator("redis://a:6379", "redis://b:6380", lag_ms=0)
        assert r._lag_ms == 0

    def test_never_lag(self) -> None:
        r = Replicator("redis://a:6379", "redis://b:6380", lag_ms=-1)
        assert r._lag_ms == -1

    def test_500ms_lag(self) -> None:
        r = Replicator("redis://a:6379", "redis://b:6380", lag_ms=500)
        assert r._lag_ms == 500


class TestReplicatorNeverLag:
    @pytest.mark.asyncio
    async def test_never_lag_exits_immediately(self) -> None:
        r = Replicator("redis://a:6379", "redis://b:6380", lag_ms=-1)
        # start() should return without connecting
        await r.start()
        assert r._redis_a is None
        assert r._redis_b is None


class TestReplicatorForward:
    @pytest.mark.asyncio
    async def test_forward_event(self) -> None:
        r = Replicator("redis://a:6379", "redis://b:6380", lag_ms=0)
        r._redis_a = AsyncMock()
        r._redis_b = AsyncMock()
        r._running = True

        fields = {"key": "k1", "tool": "navigate_to", "status": "COMMITTED"}
        await r._forward_event("1-0", fields)

        r._redis_b.xadd.assert_called_once()
        call_args = r._redis_b.xadd.call_args
        assert call_args[0][0] == "kinbridge:ledger"
        assert call_args[0][1] == fields

    @pytest.mark.asyncio
    async def test_acknowledge(self) -> None:
        r = Replicator("redis://a:6379", "redis://b:6380", lag_ms=0)
        r._redis_a = AsyncMock()
        r._running = True

        await r._acknowledge("1-0")

        r._redis_a.xack.assert_called_once_with(
            "kinbridge:ledger", "replicator-group", "1-0"
        )
        assert r._last_id == "1-0"

    @pytest.mark.asyncio
    async def test_lag_zero_no_sleep(self) -> None:
        r = Replicator("redis://a:6379", "redis://b:6380", lag_ms=0)
        with patch("experiments.replication.replicator.asyncio.sleep") as mock_sleep:
            await r._apply_lag()
            mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_lag_500_sleeps(self) -> None:
        r = Replicator("redis://a:6379", "redis://b:6380", lag_ms=500)
        with patch("experiments.replication.replicator.asyncio.sleep") as mock_sleep:
            await r._apply_lag()
            mock_sleep.assert_called_once_with(0.5)

    @pytest.mark.asyncio
    async def test_stop_closes_connections(self) -> None:
        r = Replicator("redis://a:6379", "redis://b:6380", lag_ms=0)
        r._redis_a = AsyncMock()
        r._redis_b = AsyncMock()
        r._running = True

        await r.stop()

        assert r._running is False
        r._redis_a.aclose.assert_called_once()
        r._redis_b.aclose.assert_called_once()


class TestReplicatorEnsureStream:
    @pytest.mark.asyncio
    async def test_creates_group(self) -> None:
        r = Replicator("redis://a:6379", "redis://b:6380", lag_ms=0)
        mock_redis = AsyncMock()
        await r._ensure_stream(mock_redis)
        mock_redis.xgroup_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_existing_group_no_error(self) -> None:
        r = Replicator("redis://a:6379", "redis://b:6380", lag_ms=0)
        mock_redis = AsyncMock()
        import redis.asyncio as aioredis
        mock_redis.xgroup_create.side_effect = aioredis.ResponseError(
            "BUSYGROUP Consumer Group name already exists"
        )
        # Should not raise
        await r._ensure_stream(mock_redis)


class TestReplicatorParseArgs:
    def test_defaults(self) -> None:
        args = parse_args([])
        assert args.lag_ms == 0
        assert "redis-a" in args.redis_a_url
        assert "redis-b" in args.redis_b_url

    def test_custom_lag(self) -> None:
        args = parse_args(["--lag-ms", "500"])
        assert args.lag_ms == 500

    def test_never_lag(self) -> None:
        args = parse_args(["--lag-ms", "-1"])
        assert args.lag_ms == -1
