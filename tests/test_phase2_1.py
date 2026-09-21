"""Phase 2.1 tests for the FastAPI tool world + SQLite effect ledger.

Tests cover:
  A. FastAPI application starts
  B. Each endpoint accepts valid requests
  C. Invalid request data is rejected
  D. Successful action creates a ledger record
  E. Successful action ends in COMMITTED
  F. Repeating the same committed request does not re-execute
  G. Ledger lookup returns expected status
  H. All fields persisted correctly
  I. Forced execution failure does not produce false COMMITTED
  J. Ledger state transitions
"""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio

import tool_world.main as main_mod
from tool_world.ledger import EffectLedger
from tool_world.models import EffectStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_ledger(tmp_path):
    """Replace the global ledger with an in-memory/test instance."""
    db = str(tmp_path / "test_effects.db")
    ledger = EffectLedger(db_path=db)

    original = main_mod._ledger
    main_mod._ledger = ledger

    yield ledger

    main_mod._ledger = original
    ledger.close()


@pytest_asyncio.fixture()
async def client(_patch_ledger):
    """Yield an httpx AsyncClient wired to the FastAPI app."""
    transport = httpx.ASGITransport(app=main_mod.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


# ---------------------------------------------------------------------------
# A. FastAPI application starts
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_endpoint(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# B. Each endpoint accepts valid requests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_navigate_to(client):
    resp = await client.post("/tools/navigate_to", json={
        "lat": 45.0, "lon": 7.0, "mode": "fastest"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["tool"] == "navigate_to"
    assert data["status"] == "COMMITTED"


@pytest.mark.asyncio
async def test_log_inspection(client):
    resp = await client.post("/tools/log_inspection", json={
        "site_id": "S001", "status": "passed", "notes": "All clear"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["tool"] == "log_inspection"
    assert data["status"] == "COMMITTED"


@pytest.mark.asyncio
async def test_request_supply_drop(client):
    resp = await client.post("/tools/request_supply_drop", json={
        "lat": 48.0, "lon": 2.0, "payload": "water", "priority": "high"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["tool"] == "request_supply_drop"
    assert data["status"] == "COMMITTED"


# ---------------------------------------------------------------------------
# C. Invalid request data is rejected
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_navigate_to_missing_lat(client):
    resp = await client.post("/tools/navigate_to", json={
        "lon": 7.0, "mode": "fastest"
    })
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_navigate_to_invalid_mode(client):
    resp = await client.post("/tools/navigate_to", json={
        "lat": 45.0, "lon": 7.0, "mode": "INVALID"
    })
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_log_inspection_invalid_status(client):
    resp = await client.post("/tools/log_inspection", json={
        "site_id": "S001", "status": "unknown_status", "notes": "x"
    })
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_request_supply_drop_invalid_priority(client):
    resp = await client.post("/tools/request_supply_drop", json={
        "lat": 48.0, "lon": 2.0, "payload": "water", "priority": "urgent"
    })
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_navigate_to_empty_body(client):
    resp = await client.post("/tools/navigate_to", json={})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# D. Successful action creates a ledger record
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_record_exists_after_execution(client, _patch_ledger):
    resp = await client.post("/tools/navigate_to", json={
        "lat": 45.0, "lon": 7.0, "mode": "fastest"
    })
    key = resp.json()["key"]
    record = _patch_ledger.lookup(key)
    assert record is not None
    assert record["tool"] == "navigate_to"


# ---------------------------------------------------------------------------
# E. Successful action ends in COMMITTED
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_navigate_to_committed(client):
    resp = await client.post("/tools/navigate_to", json={
        "lat": 45.0, "lon": 7.0, "mode": "fastest"
    })
    assert resp.json()["status"] == "COMMITTED"
    assert resp.json()["committed"] is True


@pytest.mark.asyncio
async def test_log_inspection_committed(client):
    resp = await client.post("/tools/log_inspection", json={
        "site_id": "S001", "status": "passed", "notes": "OK"
    })
    assert resp.json()["status"] == "COMMITTED"


@pytest.mark.asyncio
async def test_request_supply_drop_committed(client):
    resp = await client.post("/tools/request_supply_drop", json={
        "lat": 48.0, "lon": 2.0, "payload": "food", "priority": "medium"
    })
    assert resp.json()["status"] == "COMMITTED"


# ---------------------------------------------------------------------------
# F. Repeating the same committed request does not re-execute
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_duplicate_navigate_returns_existing(client, _patch_ledger):
    args = {"lat": 45.0, "lon": 7.0, "mode": "fastest"}
    resp1 = await client.post("/tools/navigate_to", json=args)
    assert resp1.json()["committed"] is True

    resp2 = await client.post("/tools/navigate_to", json=args)
    data2 = resp2.json()
    assert data2["status"] == "COMMITTED"
    assert data2["committed"] is False
    assert data2["key"] == resp1.json()["key"]


@pytest.mark.asyncio
async def test_duplicate_log_inspection(client):
    args = {"site_id": "S001", "status": "passed", "notes": "OK"}
    resp1 = await client.post("/tools/log_inspection", json=args)
    resp2 = await client.post("/tools/log_inspection", json=args)
    assert resp2.json()["committed"] is False
    assert resp2.json()["key"] == resp1.json()["key"]


@pytest.mark.asyncio
async def test_duplicate_supply_drop(client):
    args = {"lat": 48.0, "lon": 2.0, "payload": "water", "priority": "low"}
    resp1 = await client.post("/tools/request_supply_drop", json=args)
    resp2 = await client.post("/tools/request_supply_drop", json=args)
    assert resp2.json()["committed"] is False


# ---------------------------------------------------------------------------
# G. Ledger lookup returns expected status
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lookup_returns_record(client, _patch_ledger):
    resp = await client.post("/tools/navigate_to", json={
        "lat": 45.0, "lon": 7.0, "mode": "fastest"
    })
    key = resp.json()["key"]
    resp2 = await client.get(f"/ledger/{key}")
    assert resp2.status_code == 200
    data = resp2.json()
    assert data["status"] == "COMMITTED"
    assert data["tool"] == "navigate_to"


@pytest.mark.asyncio
async def test_lookup_nonexistent_returns_404(client):
    resp = await client.get("/ledger/nonexistent_key")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# H. All fields persisted correctly
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_all_fields_persisted(client, _patch_ledger):
    resp = await client.post("/tools/navigate_to", json={
        "lat": 45.0, "lon": 7.0, "mode": "fastest"
    })
    key = resp.json()["key"]
    record = _patch_ledger.lookup(key)

    assert record["key"] == key
    assert record["tool"] == "navigate_to"
    assert record["args_hash"] is not None
    assert record["status"] == "COMMITTED"
    assert record["created_at"] > 0
    assert record["committed_at"] is not None
    assert record["epoch"] >= 0
    assert record["ttl"] == 0.0


@pytest.mark.asyncio
async def test_intent_id_persisted(client, _patch_ledger):
    resp = await client.post("/tools/navigate_to", json={
        "lat": 45.0, "lon": 7.0, "mode": "fastest",
        "intent_id": "INT-001", "session_id": "S1", "turn_seq": 1
    })
    key = resp.json()["key"]
    record = _patch_ledger.lookup(key)
    assert record["intent_id"] == "INT-001"


@pytest.mark.asyncio
async def test_epoch_increments(client, _patch_ledger):
    resp1 = await client.post("/tools/navigate_to", json={
        "lat": 45.0, "lon": 7.0, "mode": "fastest"
    })
    epoch1 = resp1.json()["epoch"]

    resp2 = await client.post("/tools/log_inspection", json={
        "site_id": "S001", "status": "passed", "notes": "OK"
    })
    epoch2 = resp2.json()["epoch"]

    assert epoch2 > epoch1


# ---------------------------------------------------------------------------
# I. Forced execution failure does not produce false COMMITTED
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fault_inject_returns_unknown(client, _patch_ledger):
    resp = await client.post("/tools/navigate_to", json={
        "lat": 45.0, "lon": 7.0, "mode": "fastest",
        "fault_inject": True
    })
    data = resp.json()
    assert data["status"] == "UNKNOWN"
    assert data["committed"] is False
    assert data["fault_injected"] is True


@pytest.mark.asyncio
async def test_fault_inject_ledger_not_committed(client, _patch_ledger):
    resp = await client.post("/tools/navigate_to", json={
        "lat": 45.0, "lon": 7.0, "mode": "fastest",
        "fault_inject": True
    })
    key = resp.json()["key"]
    record = _patch_ledger.lookup(key)
    assert record["status"] == "UNKNOWN"


@pytest.mark.asyncio
async def test_fault_inject_log_inspection(client, _patch_ledger):
    resp = await client.post("/tools/log_inspection", json={
        "site_id": "S001", "status": "passed", "notes": "OK",
        "fault_inject": True
    })
    assert resp.json()["status"] == "UNKNOWN"
    assert resp.json()["committed"] is False


@pytest.mark.asyncio
async def test_fault_inject_supply_drop(client, _patch_ledger):
    resp = await client.post("/tools/request_supply_drop", json={
        "lat": 48.0, "lon": 2.0, "payload": "water", "priority": "high",
        "fault_inject": True
    })
    assert resp.json()["status"] == "UNKNOWN"
    assert resp.json()["committed"] is False


# ---------------------------------------------------------------------------
# J. Ledger state transitions
# ---------------------------------------------------------------------------

def test_absent_to_committed(_patch_ledger):
    key = "test-key-1"
    _patch_ledger.create(key=key, tool="navigate_to", args_hash="abc")
    assert _patch_ledger.get_status(key) == EffectStatus.ABSENT
    _patch_ledger.mark_committed(key)
    assert _patch_ledger.get_status(key) == EffectStatus.COMMITTED


def test_absent_to_unknown(_patch_ledger):
    key = "test-key-2"
    _patch_ledger.create(key=key, tool="navigate_to", args_hash="abc")
    _patch_ledger.mark_unknown(key)
    assert _patch_ledger.get_status(key) == EffectStatus.UNKNOWN


def test_unknown_to_escalated(_patch_ledger):
    key = "test-key-3"
    _patch_ledger.create(key=key, tool="navigate_to", args_hash="abc")
    _patch_ledger.mark_unknown(key)
    _patch_ledger.mark_escalated(key)
    assert _patch_ledger.get_status(key) == EffectStatus.ESCALATED


def test_unknown_to_invalidated(_patch_ledger):
    key = "test-key-4"
    _patch_ledger.create(key=key, tool="navigate_to", args_hash="abc")
    _patch_ledger.mark_unknown(key)
    _patch_ledger.mark_invalidated(key)
    assert _patch_ledger.get_status(key) == EffectStatus.INVALIDATED


def test_list_by_status(_patch_ledger):
    _patch_ledger.create(key="k1", tool="navigate_to", args_hash="a")
    _patch_ledger.create(key="k2", tool="navigate_to", args_hash="b")
    _patch_ledger.mark_committed("k1")

    committed = _patch_ledger.list_by_status(EffectStatus.COMMITTED)
    assert len(committed) == 1
    assert committed[0]["key"] == "k1"

    absent = _patch_ledger.list_by_status(EffectStatus.ABSENT)
    assert len(absent) == 1
    assert absent[0]["key"] == "k2"


def test_set_intent(_patch_ledger):
    key = "test-key-5"
    _patch_ledger.create(key=key, tool="navigate_to", args_hash="abc")
    _patch_ledger.set_intent(key, "INT-999")
    record = _patch_ledger.lookup(key)
    assert record["intent_id"] == "INT-999"


# ---------------------------------------------------------------------------
# Ledger unit tests (direct API, not via HTTP)
# ---------------------------------------------------------------------------

def test_ledger_lookup_returns_none_for_missing():
    ledger = EffectLedger()
    assert ledger.lookup("nonexistent") is None
    assert ledger.get_status("nonexistent") == EffectStatus.ABSENT


def test_ledger_create_and_lookup():
    ledger = EffectLedger()
    ledger.create(key="k1", tool="navigate_to", args_hash="abc")
    record = ledger.lookup("k1")
    assert record is not None
    assert record["tool"] == "navigate_to"


def test_ledger_epoch_counter():
    ledger = EffectLedger()
    e1 = ledger.increment_epoch()
    e2 = ledger.increment_epoch()
    assert e2 > e1
    assert ledger.get_epoch() == e2


def test_ledger_is_expired():
    ledger = EffectLedger()
    ledger.create(key="k1", tool="navigate_to", args_hash="abc", ttl=0.0)
    assert ledger.is_expired("k1") is True


def test_ledger_is_expired_future_ttl():
    ledger = EffectLedger()
    ledger.create(key="k1", tool="navigate_to", args_hash="abc", ttl=3600.0)
    assert ledger.is_expired("k1") is False


def test_ledger_mark_returns_false_for_missing():
    ledger = EffectLedger()
    assert ledger.mark_committed("missing") is False
    assert ledger.mark_unknown("missing") is False
    assert ledger.mark_escalated("missing") is False
    assert ledger.mark_invalidated("missing") is False
    assert ledger.set_intent("missing", "x") is False
    assert ledger.set_epoch("missing", 1) is False


def test_set_epoch_updates_record():
    ledger = EffectLedger()
    ledger.create(key="k1", tool="navigate_to", args_hash="abc")
    assert ledger.set_epoch("k1", 42) is True
    record = ledger.lookup("k1")
    assert record["epoch"] == 42


def test_set_epoch_returns_false_for_missing():
    ledger = EffectLedger()
    assert ledger.set_epoch("nonexistent", 99) is False
