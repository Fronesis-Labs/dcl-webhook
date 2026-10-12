"""Persist canonical DCL Audit Event v1.0 documents.

Sibling of the tamper-evident chain table. This module does not alter
chain rows, chain hashing, or chain_payments. Callers pass the connection
already opened for their own database. Webhook and bazaar do not share a
database file.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any, Optional


class AuditEventPersistenceError(Exception):
    """A canonical audit event was not stored.

    An already committed chain row is left in place. Callers must not turn
    this into a successful API response.
    """


class AuditEventTableMissing(Exception):
    """dcl_audit_events has not been created in this database yet."""


_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS dcl_audit_events (
        event_id TEXT PRIMARY KEY,
        event_type TEXT NOT NULL,
        schema_version TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        trace_id TEXT NOT NULL,
        tx_hash TEXT,
        event_json TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_dcl_audit_events_trace_id
        ON dcl_audit_events (trace_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_dcl_audit_events_tx_hash
        ON dcl_audit_events (tx_hash)
    """,
)


def ensure_schema(
    conn: sqlite3.Connection,
    lock: Optional[threading.Lock] = None,
) -> None:
    """Create dcl_audit_events beside chain. Idempotent."""

    def _run() -> None:
        for statement in _SCHEMA_STATEMENTS:
            conn.execute(statement)
        conn.commit()

    if lock is None:
        _run()
    else:
        with lock:
            _run()


def _tx_hash(event: dict[str, Any]) -> Optional[str]:
    proof = event.get("proof")
    if isinstance(proof, dict):
        value = proof.get("tx_hash")
        if isinstance(value, str):
            return value
    return None


def persist_audit_event(
    conn: sqlite3.Connection,
    lock: threading.Lock,
    event: dict[str, Any],
) -> None:
    """Store the exact create_audit_event() document.

    event_id uniqueness is enforced by the primary key. Failure raises
    AuditEventPersistenceError and does not report success.
    """
    try:
        columns = (
            event["event_id"],
            event["event_type"],
            event["schema_version"],
            event["timestamp"],
            event["trace_id"],
        )
    except KeyError as exc:
        raise AuditEventPersistenceError(
            f"canonical event missing {exc.args[0]}"
        ) from exc

    payload = json.dumps(event, ensure_ascii=False)
    try:
        with lock:
            conn.execute(
                """
                INSERT INTO dcl_audit_events (
                    event_id, event_type, schema_version, timestamp,
                    trace_id, tx_hash, event_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (*columns, _tx_hash(event), payload),
            )
            conn.commit()
    except AuditEventPersistenceError:
        raise
    except Exception as exc:
        raise AuditEventPersistenceError(
            f"failed to persist audit event {event.get('event_id')}"
        ) from exc


def _load(row: sqlite3.Row | tuple) -> dict[str, Any]:
    return json.loads(row[0])


def get_audit_event(
    conn: sqlite3.Connection, event_id: str
) -> Optional[dict[str, Any]]:
    row = conn.execute(
        "SELECT event_json FROM dcl_audit_events WHERE event_id = ?",
        (event_id,),
    ).fetchone()
    return _load(row) if row else None


def find_by_trace_id(
    conn: sqlite3.Connection, trace_id: str
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT event_json FROM dcl_audit_events
        WHERE trace_id = ? ORDER BY timestamp
        """,
        (trace_id,),
    ).fetchall()
    return [_load(row) for row in rows]


def find_by_tx_hash(
    conn: sqlite3.Connection, tx_hash: str
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT event_json FROM dcl_audit_events
        WHERE tx_hash = ? ORDER BY timestamp
        """,
        (tx_hash,),
    ).fetchall()
    return [_load(row) for row in rows]


def list_audit_events(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT event_json FROM dcl_audit_events ORDER BY timestamp"
    ).fetchall()
    return [_load(row) for row in rows]


def list_audit_events_from_path(db_path: str) -> list[dict[str, Any]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        exists = conn.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = 'dcl_audit_events'
            """
        ).fetchone()
        if not exists:
            raise AuditEventTableMissing(db_path)
        return list_audit_events(conn)
    finally:
        conn.close()
