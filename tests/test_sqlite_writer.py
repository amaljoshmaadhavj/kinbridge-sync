"""Tests for the Redis-to-SQLite writer (Phase 3b).

Uses a real in-memory SQLite database and unittest.mock for Redis.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from experiments.replication.sqlite_writer import SQLiteWriter, _UPSERT_SQL, parse_args


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def edge_b_db(tmp_path: object) -> str:
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
    conn.commit()
    conn.close()
    return db_path


def _make_event(
    key: str = "k1",
    tool: str = "navigate_to",
    args_hash: str = "abc123",
    status: str = "COMMITTED",
    created_at: float = 100.0,
    committed_at: float | None = 200.0,
    epoch: int = 1,
    ttl: float = 0.0,
    intent_id: str | None = "intent-1",
) -> dict[str, str]:
    """Create a Redis stream fields dict from event parameters."""
    return {
        "key": key,
        "tool": tool,
        "args_hash": args_hash,
        "status": status,
        "created_at": str(created_at),
        "committed_at": str(committed_at) if committed_at is not None else "",
        "epoch": str(epoch),
        "ttl": str(ttl),
        "intent_id": intent_id or "",
    }


def _read_all_records(db_path: str) -> list[dict]:
    """Read all records from the effect_ledger."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM effect_ledger").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# SQLiteWriter — deserialization tests
# ---------------------------------------------------------------------------


class TestDeserializeEvent:
    def test_normal_event(self) -> None:
        w = SQLiteWriter(":memory:", "redis://x:0")
        fields = _make_event(
            key="k1",
            tool="log_inspection",
            args_hash="def456",
            status="COMMITTED",
            created_at=100.0,
            committed_at=200.0,
            epoch=5,
            ttl=30.0,
            intent_id="intent-42",
        )
        record = w._deserialize_event(fields)
        assert record["key"] == "k1"
        assert record["tool"] == "log_inspection"
        assert record["args_hash"] == "def456"
        assert record["status"] == "COMMITTED"
        assert record["created_at"] == 100.0
        assert record["committed_at"] == 200.0
        assert record["epoch"] == 5
        assert record["ttl"] == 30.0
        assert record["intent_id"] == "intent-42"

    def test_null_committed_at(self) -> None:
        w = SQLiteWriter(":memory:", "redis://x:0")
        fields = _make_event(committed_at=None, intent_id=None)
        record = w._deserialize_event(fields)
        assert record["committed_at"] is None
        assert record["intent_id"] is None

    def test_missing_key_raises(self) -> None:
        w = SQLiteWriter(":memory:", "redis://x:0")
        fields = {"tool": "navigate_to", "args_hash": "abc"}
        with pytest.raises(KeyError):
            w._deserialize_event(fields)


# ---------------------------------------------------------------------------
# SQLiteWriter — upsert tests
# ---------------------------------------------------------------------------


class TestUpsertRecord:
    def test_inserts_new_record(self, edge_b_db: str) -> None:
        w = SQLiteWriter(edge_b_db, "redis://x:0")
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
        w._upsert_record(record)

        records = _read_all_records(edge_b_db)
        assert len(records) == 1
        assert records[0]["key"] == "k1"
        assert records[0]["status"] == "COMMITTED"
        assert records[0]["epoch"] == 1

    def test_idempotent_duplicate(self, edge_b_db: str) -> None:
        w = SQLiteWriter(edge_b_db, "redis://x:0")
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
        w._upsert_record(record)
        w._upsert_record(record)  # duplicate

        records = _read_all_records(edge_b_db)
        assert len(records) == 1  # no duplicate row

    def test_updates_status_on_conflict(self, edge_b_db: str) -> None:
        w = SQLiteWriter(edge_b_db, "redis://x:0")

        # First: ABSENT
        w._upsert_record({
            "key": "k1", "tool": "t", "args_hash": "a",
            "status": "ABSENT", "created_at": 1.0,
            "committed_at": None, "epoch": 1, "ttl": 0.0, "intent_id": None,
        })

        # Then: COMMITTED
        w._upsert_record({
            "key": "k1", "tool": "t", "args_hash": "a",
            "status": "COMMITTED", "created_at": 1.0,
            "committed_at": 2.0, "epoch": 2, "ttl": 0.0, "intent_id": None,
        })

        records = _read_all_records(edge_b_db)
        assert len(records) == 1
        assert records[0]["status"] == "COMMITTED"
        assert records[0]["epoch"] == 2
        assert records[0]["committed_at"] == 2.0

    def test_preserves_all_statuses(self, edge_b_db: str) -> None:
        w = SQLiteWriter(edge_b_db, "redis://x:0")
        for status in ("ABSENT", "COMMITTED", "UNKNOWN", "ESCALATED", "INVALIDATED"):
            w._upsert_record({
                "key": f"k-{status}", "tool": "t", "args_hash": "a",
                "status": status, "created_at": 1.0,
                "committed_at": None, "epoch": 1, "ttl": 0.0, "intent_id": None,
            })

        records = _read_all_records(edge_b_db)
        statuses = {r["status"] for r in records}
        assert statuses == {"ABSENT", "COMMITTED", "UNKNOWN", "ESCALATED", "INVALIDATED"}

    def test_preserves_epoch(self, edge_b_db: str) -> None:
        w = SQLiteWriter(edge_b_db, "redis://x:0")
        w._upsert_record({
            "key": "k1", "tool": "t", "args_hash": "a",
            "status": "COMMITTED", "created_at": 1.0,
            "committed_at": 2.0, "epoch": 42, "ttl": 0.0, "intent_id": None,
        })
        records = _read_all_records(edge_b_db)
        assert records[0]["epoch"] == 42

    def test_preserves_intent_id(self, edge_b_db: str) -> None:
        w = SQLiteWriter(edge_b_db, "redis://x:0")
        w._upsert_record({
            "key": "k1", "tool": "t", "args_hash": "a",
            "status": "COMMITTED", "created_at": 1.0,
            "committed_at": 2.0, "epoch": 1, "ttl": 0.0, "intent_id": "intent-99",
        })
        records = _read_all_records(edge_b_db)
        assert records[0]["intent_id"] == "intent-99"


# ---------------------------------------------------------------------------
# SQLiteWriter — process event tests
# ---------------------------------------------------------------------------


class TestProcessEvent:
    @pytest.mark.asyncio
    async def test_processes_valid_event(self, edge_b_db: str) -> None:
        w = SQLiteWriter(edge_b_db, "redis://x:0")
        w._redis = AsyncMock()

        fields = _make_event(key="k1", status="COMMITTED")
        await w._process_event("1-0", fields)

        records = _read_all_records(edge_b_db)
        assert len(records) == 1
        assert records[0]["key"] == "k1"
        w._redis.xack.assert_called_once()

    @pytest.mark.asyncio
    async def test_malformed_event_skipped(self, edge_b_db: str) -> None:
        w = SQLiteWriter(edge_b_db, "redis://x:0")
        w._redis = AsyncMock()

        fields = {"bad": "data"}  # missing required fields
        await w._process_event("1-0", fields)

        records = _read_all_records(edge_b_db)
        assert len(records) == 0
        # Still acks the event to avoid reprocessing
        w._redis.xack.assert_called_once()


class TestWriterStop:
    @pytest.mark.asyncio
    async def test_stop_closes_connections(self) -> None:
        w = SQLiteWriter(":memory:", "redis://x:0")
        w._redis = AsyncMock()
        w._conn = sqlite3.connect(":memory:")
        w._running = True

        await w.stop()

        assert w._running is False
        w._redis.aclose.assert_called_once()


class TestWriterParseArgs:
    def test_defaults(self) -> None:
        args = parse_args([])
        assert "redis-b" in args.redis_url
        assert args.db_path is not None

    def test_custom_db_path(self) -> None:
        args = parse_args(["--db-path", "/tmp/test.db"])
        assert args.db_path == "/tmp/test.db"

    def test_custom_redis_url(self) -> None:
        args = parse_args(["--redis-url", "redis://custom:6379/0"])
        assert args.redis_url == "redis://custom:6379/0"
