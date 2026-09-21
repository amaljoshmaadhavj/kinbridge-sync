"""Phase 2.2 tests for the Action Buffer (client-side JSONL WAL).

Tests cover:
  - append and read round-trip
  - ordering preservation
  - status changes (mark_committed, mark_invalidated)
  - max_epoch computation
  - persistence across instances
  - record field completeness
  - thread safety
  - nonexistent key handling
  - WAL immutability (status changes do not rewrite JSONL)
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

from client.buffer import ActionBuffer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def buf(tmp_path: Path) -> ActionBuffer:
    """Return a fresh ActionBuffer backed by a temp directory."""
    return ActionBuffer(tmp_path / "wal.jsonl")


def _make_record(
    key: str = "key-1",
    tool: str = "navigate_to",
    intent_id: str = "INT-001",
    epoch: int = 1,
    **overrides,
) -> dict:
    """Helper to build a minimal valid record dict."""
    rec = {
        "intent_id": intent_id,
        "key": key,
        "tool": tool,
        "args": {"lat": 45.0, "lon": 7.0, "mode": "fastest"},
        "t0": 1_700_000_000.0,
        "ttl": 60.0,
        "v_pre_spec": {"type": "none"},
        "epoch": epoch,
    }
    rec.update(overrides)
    return rec


# ---------------------------------------------------------------------------
# 1. test_append_and_read
# ---------------------------------------------------------------------------

def test_append_and_read(buf: ActionBuffer):
    rec = _make_record()
    buf.append(**rec)
    records = buf.read_all()
    assert len(records) == 1
    r = records[0]
    assert r["intent_id"] == "INT-001"
    assert r["key"] == "key-1"
    assert r["tool"] == "navigate_to"
    assert r["args"] == {"lat": 45.0, "lon": 7.0, "mode": "fastest"}
    assert r["t0"] == 1_700_000_000.0
    assert r["ttl"] == 60.0
    assert r["v_pre_spec"] == {"type": "none"}
    assert r["status"] == "PENDING"
    assert r["epoch"] == 1


# ---------------------------------------------------------------------------
# 2. test_append_multiple
# ---------------------------------------------------------------------------

def test_append_multiple(buf: ActionBuffer):
    keys = ["k1", "k2", "k3"]
    for k in keys:
        buf.append(**_make_record(key=k))
    records = buf.read_all()
    assert [r["key"] for r in records] == keys


# ---------------------------------------------------------------------------
# 3. test_mark_committed
# ---------------------------------------------------------------------------

def test_mark_committed(buf: ActionBuffer):
    buf.append(**_make_record())
    assert buf.mark_committed("key-1") is True
    records = buf.read_all()
    assert len(records) == 1
    assert records[0]["status"] == "COMMITTED"


# ---------------------------------------------------------------------------
# 4. test_mark_invalidated
# ---------------------------------------------------------------------------

def test_mark_invalidated(buf: ActionBuffer):
    buf.append(**_make_record())
    assert buf.mark_invalidated("key-1") is True
    records = buf.read_all()
    assert records[0]["status"] == "INVALIDATED"


# ---------------------------------------------------------------------------
# 5. test_max_epoch
# ---------------------------------------------------------------------------

def test_max_epoch(buf: ActionBuffer):
    buf.append(**_make_record(key="a", epoch=1))
    buf.append(**_make_record(key="b", epoch=3))
    buf.append(**_make_record(key="c", epoch=2))
    assert buf.max_epoch == 3


# ---------------------------------------------------------------------------
# 6. test_max_epoch_empty
# ---------------------------------------------------------------------------

def test_max_epoch_empty(buf: ActionBuffer):
    assert buf.max_epoch == 0


# ---------------------------------------------------------------------------
# 7. test_persistence
# ---------------------------------------------------------------------------

def test_persistence(tmp_path: Path):
    path = tmp_path / "wal.jsonl"

    # First instance: append + mark
    b1 = ActionBuffer(path)
    b1.append(**_make_record(key="k1", epoch=5))
    b1.append(**_make_record(key="k2", epoch=2))
    b1.mark_committed("k1")

    # Second instance on the same path
    b2 = ActionBuffer(path)
    records = b2.read_all()
    assert len(records) == 2
    assert records[0]["key"] == "k1"
    assert records[0]["status"] == "COMMITTED"
    assert records[0]["epoch"] == 5
    assert records[1]["key"] == "k2"
    assert records[1]["status"] == "PENDING"
    assert b2.max_epoch == 5


# ---------------------------------------------------------------------------
# 8. test_record_fields
# ---------------------------------------------------------------------------

def test_record_fields(buf: ActionBuffer):
    buf.append(
        intent_id="INT-99",
        key="k-full",
        tool="log_inspection",
        args={"site_id": "S1", "status": "passed", "notes": "ok"},
        t0=100.5,
        ttl=30.0,
        v_pre_spec={"check": "site_exists"},
        epoch=7,
    )
    r = buf.read_all()[0]
    assert r["intent_id"] == "INT-99"
    assert r["key"] == "k-full"
    assert r["tool"] == "log_inspection"
    assert r["args"]["site_id"] == "S1"
    assert r["t0"] == 100.5
    assert r["ttl"] == 30.0
    assert r["v_pre_spec"] == {"check": "site_exists"}
    assert r["epoch"] == 7
    assert r["status"] == "PENDING"


# ---------------------------------------------------------------------------
# Concurrent access / thread safety
# ---------------------------------------------------------------------------

def test_concurrent_appends(tmp_path: Path):
    path = tmp_path / "wal.jsonl"
    buf = ActionBuffer(path)
    n_threads = 4
    n_records = 25
    errors: list[Exception] = []

    def worker(thread_id: int):
        try:
            for i in range(n_records):
                buf.append(
                    intent_id=f"T{thread_id}-{i}",
                    key=f"k-{thread_id}-{i}",
                    tool="navigate_to",
                    args={"lat": 0.0, "lon": 0.0, "mode": "fastest"},
                    t0=time.time(),
                    ttl=10.0,
                    v_pre_spec={},
                    epoch=thread_id,
                )
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    records = buf.read_all()
    assert len(records) == n_threads * n_records


def test_concurrent_marks(tmp_path: Path):
    path = tmp_path / "wal.jsonl"
    buf = ActionBuffer(path)
    n = 50
    for i in range(n):
        buf.append(
            intent_id=f"I{i}",
            key=f"k{i}",
            tool="navigate_to",
            args={"lat": 0.0, "lon": 0.0, "mode": "fastest"},
            t0=time.time(),
            ttl=10.0,
            v_pre_spec={},
            epoch=i,
        )
    errors: list[Exception] = []

    def mark_all():
        try:
            for i in range(n):
                if i % 2 == 0:
                    buf.mark_committed(f"k{i}")
                else:
                    buf.mark_invalidated(f"k{i}")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=mark_all) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    records = buf.read_all()
    committed = [r for r in records if r["status"] == "COMMITTED"]
    invalidated = [r for r in records if r["status"] == "INVALIDATED"]
    assert len(committed) + len(invalidated) == n


# ---------------------------------------------------------------------------
# Nonexistent key handling
# ---------------------------------------------------------------------------

def test_mark_committed_nonexistent(buf: ActionBuffer):
    assert buf.mark_committed("no-such-key") is False


def test_mark_invalidated_nonexistent(buf: ActionBuffer):
    assert buf.mark_invalidated("no-such-key") is False


# ---------------------------------------------------------------------------
# WAL immutability: status changes do not rewrite historical records
# ---------------------------------------------------------------------------

def test_wal_not_rewritten_on_status_change(tmp_path: Path):
    path = tmp_path / "wal.jsonl"
    buf = ActionBuffer(path)
    buf.append(**_make_record(key="k1"))
    buf.mark_committed("k1")

    # Read the raw JSONL file and parse it
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f if line.strip()]
    assert len(raw_lines) == 1
    raw_record = __import__("json").loads(raw_lines[0])
    # Original WAL record still says PENDING — it was not rewritten
    assert raw_record["status"] == "PENDING"


# ---------------------------------------------------------------------------
# Multiple status changes do not create duplicate logical records
# ---------------------------------------------------------------------------

def test_multiple_status_changes_no_duplicates(buf: ActionBuffer):
    buf.append(**_make_record(key="k1"))
    buf.mark_committed("k1")
    buf.mark_invalidated("k1")  # second change
    buf.mark_committed("k1")    # third change
    records = buf.read_all()
    assert len(records) == 1
    assert records[0]["status"] == "COMMITTED"


# ---------------------------------------------------------------------------
# Existing WAL loaded by a new instance
# ---------------------------------------------------------------------------

def test_existing_wal_loaded(tmp_path: Path):
    path = tmp_path / "wal.jsonl"
    b1 = ActionBuffer(path)
    b1.append(**_make_record(key="a", epoch=1))
    b1.append(**_make_record(key="b", epoch=2))
    b1.mark_committed("a")

    b2 = ActionBuffer(path)
    records = b2.read_all()
    assert len(records) == 2
    assert records[0]["status"] == "COMMITTED"
    assert records[1]["status"] == "PENDING"
    assert b2.max_epoch == 2


# ---------------------------------------------------------------------------
# Default values for optional fields
# ---------------------------------------------------------------------------

def test_defaults_applied(buf: ActionBuffer):
    buf.append(
        intent_id="I1",
        key="k1",
        tool="navigate_to",
        args={},
        t0=0.0,
        ttl=0.0,
    )
    r = buf.read_all()[0]
    assert r["status"] == "PENDING"
    assert r["epoch"] == 0
    assert r["v_pre_spec"] == {}
