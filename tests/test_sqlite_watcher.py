"""Tests for the SQLite-to-Redis watcher (Phase 3b).

Uses a real in-memory SQLite database and unittest.mock for Redis.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from experiments.replication.sqlite_watcher import SQLiteWatcher, parse_args


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def edge_a_db(tmp_path: object) -> str:
    """Create a fresh SQLite database with the effect_ledger schema."""
    import pathlib
    db_path = str(pathlib.Path(str(tmp_path)) / "effects.db")
    conn = sqlite3.connect(db_path)
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ledger_meta (
            key TEXT PRIMARY KEY,
            value INTEGER NOT NULL
        )
    """)
    conn.execute("INSERT INTO ledger_meta (key, value) VALUES ('epoch', 0)")
    conn.commit()
    conn.close()
    return db_path


def _insert_record(
    db_path: str,
    key: str,
    tool: str = "navigate_to",
    args_hash: str = "abc123",
    status: str = "ABSENT",
    created_at: float | None = None,
    committed_at: float | None = None,
    epoch: int = 1,
    ttl: float = 0.0,
    intent_id: str | None = None,
) -> None:
    """Insert a record into the effect_ledger for testing."""
    conn = sqlite3.connect(db_path)
    now = created_at or time.time()
    conn.execute(
        """INSERT INTO effect_ledger
           (key, tool, args_hash, status, created_at, committed_at, epoch, ttl, intent_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (key, tool, args_hash, status, now, committed_at, epoch, ttl, intent_id),
    )
    conn.commit()
    conn.close()


def _update_status(db_path: str, key: str, status: str) -> None:
    """Update a record's status for testing."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE effect_ledger SET status = ?, committed_at = ? WHERE key = ?",
        (status, time.time(), key),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# SQLiteWatcher — unit tests
# ---------------------------------------------------------------------------


class TestWatcherPollNewRecords:
    def test_detects_new_record(self, edge_a_db: str) -> None:
        _insert_record(edge_a_db, "k1")
        w = SQLiteWatcher(edge_a_db, "redis://x:0")
        records = w._poll_new_records()
        assert len(records) == 1
        assert records[0]["key"] == "k1"

    def test_no_records_returns_empty(self, edge_a_db: str) -> None:
        w = SQLiteWatcher(edge_a_db, "redis://x:0")
        records = w._poll_new_records()
        assert records == []

    def test_does_not_duplicate_seen_records(self, edge_a_db: str) -> None:
        _insert_record(edge_a_db, "k1")
        w = SQLiteWatcher(edge_a_db, "redis://x:0")
        records1 = w._poll_new_records()
        assert len(records1) == 1

        # Second poll — no new records
        records2 = w._poll_new_records()
        assert records2 == []

    def test_multiple_records_in_batch(self, edge_a_db: str) -> None:
        _insert_record(edge_a_db, "k1", created_at=1.0)
        _insert_record(edge_a_db, "k2", created_at=2.0)
        _insert_record(edge_a_db, "k3", created_at=3.0)
        w = SQLiteWatcher(edge_a_db, "redis://x:0")
        records = w._poll_new_records()
        assert len(records) == 3
        keys = {r["key"] for r in records}
        assert keys == {"k1", "k2", "k3"}

    def test_preserves_all_fields(self, edge_a_db: str) -> None:
        _insert_record(
            edge_a_db, "k1",
            tool="log_inspection",
            args_hash="def456",
            status="COMMITTED",
            created_at=100.0,
            committed_at=200.0,
            epoch=5,
            ttl=30.0,
            intent_id="intent-42",
        )
        w = SQLiteWatcher(edge_a_db, "redis://x:0")
        records = w._poll_new_records()
        assert len(records) == 1
        r = records[0]
        assert r["key"] == "k1"
        assert r["tool"] == "log_inspection"
        assert r["args_hash"] == "def456"
        assert r["status"] == "COMMITTED"
        assert r["created_at"] == 100.0
        assert r["committed_at"] == 200.0
        assert r["epoch"] == 5
        assert r["ttl"] == 30.0
        assert r["intent_id"] == "intent-42"

    def test_handles_null_committed_at(self, edge_a_db: str) -> None:
        _insert_record(edge_a_db, "k1", status="ABSENT", committed_at=None)
        w = SQLiteWatcher(edge_a_db, "redis://x:0")
        records = w._poll_new_records()
        assert records[0]["committed_at"] is None


class TestWatcherPollStatusUpdates:
    def test_detects_status_change(self, edge_a_db: str) -> None:
        _insert_record(edge_a_db, "k1", created_at=1.0)
        w = SQLiteWatcher(edge_a_db, "redis://x:0")
        w._poll_new_records()  # mark as seen

        _update_status(edge_a_db, "k1", "COMMITTED")
        updates = w._poll_status_updates()
        assert len(updates) == 1
        assert updates[0]["status"] == "COMMITTED"

    def test_no_update_returns_empty(self, edge_a_db: str) -> None:
        _insert_record(edge_a_db, "k1")
        w = SQLiteWatcher(edge_a_db, "redis://x:0")
        w._poll_new_records()
        updates = w._poll_status_updates()
        assert updates == []


class TestWatcherPublishEvent:
    @pytest.mark.asyncio
    async def test_publishes_to_redis(self, edge_a_db: str) -> None:
        w = SQLiteWatcher(edge_a_db, "redis://x:0")
        w._redis = AsyncMock()

        record = {
            "key": "k1",
            "tool": "navigate_to",
            "args_hash": "abc",
            "status": "COMMITTED",
            "created_at": 100.0,
            "committed_at": 200.0,
            "epoch": 1,
            "ttl": 0.0,
            "intent_id": "i1",
        }
        await w._publish_event(record)

        w._redis.xadd.assert_called_once()
        call_args = w._redis.xadd.call_args
        assert call_args[0][0] == "kinbridge:ledger"
        fields = call_args[0][1]
        assert fields["key"] == "k1"
        assert fields["status"] == "COMMITTED"
        assert fields["epoch"] == "1"
        assert fields["created_at"] == "100.0"
        assert fields["committed_at"] == "200.0"

    @pytest.mark.asyncio
    async def test_publishes_null_committed_at(self, edge_a_db: str) -> None:
        w = SQLiteWatcher(edge_a_db, "redis://x:0")
        w._redis = AsyncMock()

        record = {
            "key": "k1",
            "tool": "navigate_to",
            "args_hash": "abc",
            "status": "ABSENT",
            "created_at": 100.0,
            "committed_at": None,
            "epoch": 1,
            "ttl": 0.0,
            "intent_id": None,
        }
        await w._publish_event(record)

        fields = w._redis.xadd.call_args[0][1]
        assert fields["committed_at"] == ""
        assert fields["intent_id"] == ""


class TestWatcherStop:
    @pytest.mark.asyncio
    async def test_stop_closes_connections(self) -> None:
        w = SQLiteWatcher(":memory:", "redis://x:0")
        w._redis = AsyncMock()
        w._conn = sqlite3.connect(":memory:")
        w._running = True

        await w.stop()

        assert w._running is False
        w._redis.aclose.assert_called_once()


class TestWatcherParseArgs:
    def test_defaults(self) -> None:
        args = parse_args([])
        assert "redis-a" in args.redis_url
        assert args.poll_interval == 0.05

    def test_custom_db_path(self) -> None:
        args = parse_args(["--db-path", "/tmp/test.db"])
        assert args.db_path == "/tmp/test.db"

    def test_custom_poll_interval(self) -> None:
        args = parse_args(["--poll-interval", "0.1"])
        assert args.poll_interval == 0.1
