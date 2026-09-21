"""SQLite-backed effect ledger for Phase 2.1.

Stores tool execution records with idempotency keys, status tracking,
epoch versioning, and TTL support.  Designed so the future Reintegration
Service (Phase 4) can perform status lookups, escalation, and intent
association without implementing those decisions here.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from typing import Any

from tool_world.models import EffectStatus


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
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
);
"""

_CREATE_INDEX_TOOL = """
CREATE INDEX IF NOT EXISTS idx_effect_ledger_tool ON effect_ledger(tool);
"""

_CREATE_INDEX_STATUS = """
CREATE INDEX IF NOT EXISTS idx_effect_ledger_status ON effect_ledger(status);
"""

_CREATE_INDEX_INTENT = """
CREATE INDEX IF NOT EXISTS idx_effect_ledger_intent ON effect_ledger(intent_id);
"""


# ---------------------------------------------------------------------------
# Ledger class
# ---------------------------------------------------------------------------

class EffectLedger:
    """Thread-safe SQLite effect ledger.

    Each instance owns a connection to a single SQLite database file.
    Writes use explicit transactions for atomicity.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or ":memory:"
        self._local = threading.local()
        self._schema_lock = threading.Lock()
        self._ensure_schema()

    # -- connection management ------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        """Return a thread-local connection, creating if needed."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, timeout=10.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def _ensure_schema(self) -> None:
        """Create tables/indexes if they don't exist (idempotent)."""
        with self._schema_lock:
            conn = self._get_conn()
            conn.executescript(_CREATE_TABLE)
            conn.execute(_CREATE_INDEX_TOOL)
            conn.execute(_CREATE_INDEX_STATUS)
            conn.execute(_CREATE_INDEX_INTENT)
            conn.commit()

    # -- lookup ---------------------------------------------------------------

    def lookup(self, key: str) -> dict[str, Any] | None:
        """Look up an effect record by key.

        Returns:
            dict with record fields, or None if no record exists (ABSENT).
        """
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM effect_ledger WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def get_status(self, key: str) -> EffectStatus:
        """Return the status of an effect key, or ABSENT if not found."""
        record = self.lookup(key)
        if record is None:
            return EffectStatus.ABSENT
        return EffectStatus(record["status"])

    # -- creation -------------------------------------------------------------

    def create(
        self,
        key: str,
        tool: str,
        args_hash: str,
        ttl: float = 0.0,
        intent_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a new effect record.

        The record is created with status ABSENT.  The caller should
        execute the tool action and then mark it COMMITTED or UNKNOWN.

        Raises:
            sqlite3.IntegrityError: if the key already exists.
        """
        now = time.time()
        conn = self._get_conn()
        with conn:
            conn.execute(
                """INSERT INTO effect_ledger
                   (key, tool, args_hash, status, created_at, ttl, intent_id)
                   VALUES (?, ?, ?, 'ABSENT', ?, ?, ?)""",
                (key, tool, args_hash, now, ttl, intent_id),
            )
        return self.lookup(key)  # type: ignore[return-value]

    # -- status transitions ---------------------------------------------------

    def mark_committed(self, key: str, epoch: int | None = None) -> bool:
        """Mark an effect as COMMITTED.

        Returns:
            True if the record was updated, False if not found.
        """
        conn = self._get_conn()
        now = time.time()
        with conn:
            if epoch is not None:
                cursor = conn.execute(
                    """UPDATE effect_ledger
                       SET status = 'COMMITTED', committed_at = ?, epoch = ?
                       WHERE key = ?""",
                    (now, epoch, key),
                )
            else:
                cursor = conn.execute(
                    """UPDATE effect_ledger
                       SET status = 'COMMITTED', committed_at = ?
                       WHERE key = ?""",
                    (now, key),
                )
        return cursor.rowcount > 0

    def mark_unknown(self, key: str) -> bool:
        """Mark an effect as UNKNOWN (execution state uncertain).

        Returns:
            True if the record was updated, False if not found.
        """
        conn = self._get_conn()
        with conn:
            cursor = conn.execute(
                "UPDATE effect_ledger SET status = 'UNKNOWN' WHERE key = ?",
                (key,),
            )
        return cursor.rowcount > 0

    def mark_escalated(self, key: str) -> bool:
        """Mark an effect as ESCALATED (needs human/reintegration attention).

        Returns:
            True if the record was updated, False if not found.
        """
        conn = self._get_conn()
        with conn:
            cursor = conn.execute(
                "UPDATE effect_ledger SET status = 'ESCALATED' WHERE key = ?",
                (key,),
            )
        return cursor.rowcount > 0

    def mark_invalidated(self, key: str) -> bool:
        """Mark an effect as INVALIDATED (TTL expired or V_pre failed).

        Returns:
            True if the record was updated, False if not found.
        """
        conn = self._get_conn()
        with conn:
            cursor = conn.execute(
                "UPDATE effect_ledger SET status = 'INVALIDATED' WHERE key = ?",
                (key,),
            )
        return cursor.rowcount > 0

    def set_intent(self, key: str, intent_id: str) -> bool:
        """Associate an intent_id with an effect record.

        Returns:
            True if the record was updated, False if not found.
        """
        conn = self._get_conn()
        with conn:
            cursor = conn.execute(
                "UPDATE effect_ledger SET intent_id = ? WHERE key = ?",
                (intent_id, key),
            )
        return cursor.rowcount > 0

    def set_epoch(self, key: str, epoch: int) -> bool:
        """Set the epoch for an effect record.

        Returns:
            True if the record was updated, False if not found.
        """
        conn = self._get_conn()
        with conn:
            cursor = conn.execute(
                "UPDATE effect_ledger SET epoch = ? WHERE key = ?",
                (epoch, key),
            )
        return cursor.rowcount > 0

    def increment_epoch(self) -> int:
        """Increment and return the global epoch counter.

        The epoch is stored in a separate meta table so it can be
        compared against buffer.max_epoch in Phase 4.
        """
        conn = self._get_conn()
        with conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS ledger_meta (
                    key TEXT PRIMARY KEY,
                    value INTEGER NOT NULL
                )"""
            )
            conn.execute(
                """INSERT INTO ledger_meta (key, value)
                   VALUES ('epoch', 0)
                   ON CONFLICT(key) DO NOTHING"""
            )
            conn.execute(
                """UPDATE ledger_meta
                   SET value = value + 1
                   WHERE key = 'epoch'"""
            )
            row = conn.execute(
                "SELECT value FROM ledger_meta WHERE key = 'epoch'"
            ).fetchone()
        return row["value"] if row else 0

    def get_epoch(self) -> int:
        """Return the current global epoch value."""
        conn = self._get_conn()
        conn.execute(
            """CREATE TABLE IF NOT EXISTS ledger_meta (
                key TEXT PRIMARY KEY,
                value INTEGER NOT NULL
            )"""
        )
        conn.execute(
            """INSERT INTO ledger_meta (key, value)
               VALUES ('epoch', 0)
               ON CONFLICT(key) DO NOTHING"""
        )
        row = conn.execute(
            "SELECT value FROM ledger_meta WHERE key = 'epoch'"
        ).fetchone()
        return row["value"] if row else 0

    # -- queries --------------------------------------------------------------

    def list_by_status(self, status: EffectStatus) -> list[dict[str, Any]]:
        """Return all records with the given status."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM effect_ledger WHERE status = ?",
            (status.value,),
        ).fetchall()
        return [dict(r) for r in rows]

    def is_expired(self, key: str) -> bool:
        """Check if a record's TTL has expired."""
        record = self.lookup(key)
        if record is None:
            return True
        return time.time() > record["created_at"] + record["ttl"]

    def close(self) -> None:
        """Close the thread-local connection."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
