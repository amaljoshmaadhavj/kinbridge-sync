"""Phase 3b — SQLite ↔ Redis replication bridge.

Components:
- replicator: Reads events from Redis A, writes to Redis B after configurable lag.
- sqlite_watcher: Polls edge-a SQLite, publishes new records to Redis A.
- sqlite_writer: Consumes events from Redis B, upserts into edge-b SQLite.
"""
