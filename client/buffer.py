"""Client-side append-only JSONL write-ahead log (Action Buffer).

Stores every action before it is sent to the tool world.  If the client
crashes or the network partitions, the buffer preserves the action record
so the Reintegration Service (Phase 4) can retry or escalate it.

Design:
  - The JSONL file is append-only.  Each line is one immutable record.
  - Status changes (COMMITTED / INVALIDATED) are tracked in a separate
    in-memory index and persisted to a companion ``.idx.json`` file.
  - ``read_all()`` merges the WAL records with the latest status index,
    returning each logical record exactly once in original append order.
  - Thread safety is provided by a single ``threading.Lock``.

References:
  Markdown/Work .md  — WAL record fields (intent_id, key, tool, args,
                       t0, ttl, v_pre_spec)
  Markdown/Complete-Research.md — Phase 2 specification
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any


class ActionBuffer:
    """Append-only JSONL action buffer with external status index.

    Args:
        path: Filesystem path for the JSONL WAL file.  A companion
              ``.idx.json`` file is created alongside it to persist
              status changes.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._idx_path = self._path.with_suffix(self._path.suffix + ".idx.json")
        self._lock = threading.Lock()
        self._records: list[dict[str, Any]] = []
        self._status_index: dict[str, str] = {}  # key → status
        self._load()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Reconstruct state from the JSONL WAL and the status index."""
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing WAL records (append-only, never rewritten)
        if self._path.exists():
            with open(self._path, "r", encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        # Malformed line: skip silently (conservative).
                        # The record is lost but the rest of the WAL
                        # remains intact.  A production system might log
                        # this; for the research testbed we simply skip.
                        continue
                    self._records.append(record)

        # Load status index (key → status)
        if self._idx_path.exists():
            with open(self._idx_path, "r", encoding="utf-8") as f:
                self._status_index = json.load(f)

    def _save_index(self) -> None:
        """Persist the status index to disk (atomic-ish via truncate+write)."""
        tmp = self._idx_path.with_suffix(self._idx_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._status_index, f, separators=(",", ":"))
        os.replace(tmp, self._idx_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(
        self,
        *,
        intent_id: str,
        key: str,
        tool: str,
        args: dict[str, Any],
        t0: float,
        ttl: float,
        v_pre_spec: dict[str, Any] | None = None,
        status: str = "PENDING",
        epoch: int = 0,
    ) -> dict[str, Any]:
        """Append a new action record to the WAL.

        Args:
            intent_id:    Ground-truth intent identifier.
            key:          Idempotency key for the action.
            tool:         Tool name.
            args:         Tool arguments (opaque dict).
            t0:           Creation timestamp (seconds since epoch).
            ttl:          Time-to-live in seconds.
            v_pre_spec:   Pre-condition specification (opaque JSON).
            status:       Initial status (default ``"PENDING"``).
            epoch:        Ledger epoch at creation time (default ``0``).

        Returns:
            The full record dict that was written.
        """
        record: dict[str, Any] = {
            "intent_id": intent_id,
            "key": key,
            "tool": tool,
            "args": args,
            "t0": t0,
            "ttl": ttl,
            "v_pre_spec": v_pre_spec if v_pre_spec is not None else {},
            "status": status,
            "epoch": epoch,
        }
        line = json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n"

        with self._lock:
            # Append to WAL (the only write to the JSONL file)
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            self._records.append(record)

        return record

    def read_all(self) -> list[dict[str, Any]]:
        """Return all buffered records in append order.

        Status changes are reflected in the returned records.  Each
        logical action appears exactly once (the most recent status
        wins).
        """
        with self._lock:
            result = []
            for rec in self._records:
                out = dict(rec)
                key = out["key"]
                if key in self._status_index:
                    out["status"] = self._status_index[key]
                result.append(out)
            return result

    def mark_committed(self, key: str) -> bool:
        """Mark the action with the given key as COMMITTED.

        Does not rewrite the original WAL record.  Updates the
        in-memory index and persists it.

        Returns:
            ``True`` if the key was found, ``False`` otherwise.
        """
        with self._lock:
            if not any(r["key"] == key for r in self._records):
                return False
            self._status_index[key] = "COMMITTED"
            self._save_index()
            return True

    def mark_invalidated(self, key: str) -> bool:
        """Mark the action with the given key as INVALIDATED.

        Same behavior as :meth:`mark_committed` with status
        ``"INVALIDATED"``.

        Returns:
            ``True`` if the key was found, ``False`` otherwise.
        """
        with self._lock:
            if not any(r["key"] == key for r in self._records):
                return False
            self._status_index[key] = "INVALIDATED"
            self._save_index()
            return True

    @property
    def max_epoch(self) -> int:
        """Highest epoch among all buffered records, or 0 if empty."""
        with self._lock:
            if not self._records:
                return 0
            return max(r.get("epoch", 0) for r in self._records)
