"""Canonical DCL Audit Event v1.0 builder.

Inspection of this repository found no in-tree `create_audit_event`.
Production evaluation emits a *different* artifact: a dcl-core ChainState
row (`tx_hash`, `verdict`, `agent_id`, `policy_hash`, ...) via
`ChainState.append` inside webhook_server / mcp_server / bazaar_server.

This module is the local builder for the existing Canonical DCL Audit Event
v1.0 *contract* used by the agent-control architecture:

  - event_type = dcl.audit.evaluated
  - schema_version = 1.0
  - trace_id required
  - event_id distinct from payment_id / tx_hash / receipt_id
  - policy_id + policy_version identify the policy actually applied
  - agent_id is optional/contextual and is NOT implicitly verified
  - session_id_fingerprint is optional opaque context (not hashed here)
  - payment / proof / integration / metadata remain optional
  - timestamp is a required UTC ISO-8601 string

It does not modify production evaluation, does not replace ChainState, and
does not invent a second competing schema. Local hard-policy blocks must
NOT call this builder.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

CANONICAL_EVENT_TYPE = "dcl.audit.evaluated"
CANONICAL_SCHEMA_VERSION = "1.0"

_UNKNOWN_POLICY_LITERAL = "unknown"


def _is_utc_iso8601(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if "T" not in value:
        return False
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return False
    if parsed.tzinfo is None:
        return False
    offset = parsed.utcoffset()
    return offset is not None and offset == timedelta(0)


def _require_applied_policy_field(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must identify the policy actually evaluated by DCL")
    if value == _UNKNOWN_POLICY_LITERAL:
        raise ValueError(
            f"{name} must not be the literal fallback {_UNKNOWN_POLICY_LITERAL!r}"
        )
    return value


def _require_utc_iso8601_timestamp(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        raise TypeError(
            "timestamp must be a UTC ISO-8601 string, not a Unix numeric timestamp"
        )
    if not _is_utc_iso8601(value):
        raise ValueError("timestamp must be a UTC ISO-8601 string")
    return value


def canonical_bytes(event: Mapping[str, Any]) -> bytes:
    """Serialize a canonical audit event to deterministic UTF-8 bytes.

    Object keys are ordered lexicographically (including nested objects).
    Insignificant whitespace is omitted. Absent optional fields are not
    emitted as null. This utility does not compute an event hash.
    """
    return json.dumps(
        event,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def create_audit_event(
    *,
    trace_id: str,
    policy_id: str,
    policy_version: str,
    verdict: str,
    agent_id: str | None = None,
    reason: str | None = None,
    confidence: float | None = None,
    event_id: str | None = None,
    payment_id: str | None = None,
    tx_hash: str | None = None,
    receipt_id: str | None = None,
    session_id_fingerprint: str | None = None,
    payment: Mapping[str, Any] | None = None,
    proof: Mapping[str, Any] | None = None,
    integration: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build a Canonical DCL Audit Event v1.0 for a completed DCL evaluation.

    Call for both COMMIT and NO_COMMIT. Do not call when DCL was not evaluated.
    """
    if not trace_id:
        raise ValueError("trace_id is required")
    if verdict not in ("COMMIT", "NO_COMMIT"):
        raise ValueError("verdict must be COMMIT or NO_COMMIT")
    applied_policy_id = _require_applied_policy_field("policy_id", policy_id)
    applied_policy_version = _require_applied_policy_field(
        "policy_version", policy_version
    )

    if timestamp is None:
        resolved_timestamp = datetime.now(timezone.utc).isoformat()
    else:
        resolved_timestamp = _require_utc_iso8601_timestamp(timestamp)

    resolved_event_id = event_id or str(uuid.uuid4())
    colliding = {
        name: value
        for name, value in (
            ("payment_id", payment_id),
            ("tx_hash", tx_hash),
            ("receipt_id", receipt_id),
        )
        if value is not None and value == resolved_event_id
    }
    if colliding:
        names = ", ".join(colliding)
        raise ValueError(f"event_id must be distinct from {names}")

    event: dict[str, Any] = {
        "event_type": CANONICAL_EVENT_TYPE,
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "event_id": resolved_event_id,
        "trace_id": trace_id,
        "policy_id": applied_policy_id,
        "policy_version": applied_policy_version,
        "verdict": verdict,
        "timestamp": resolved_timestamp,
    }
    if agent_id is not None:
        event["agent_id"] = agent_id
    if session_id_fingerprint is not None:
        event["session_id_fingerprint"] = session_id_fingerprint
    if reason is not None:
        event["reason"] = reason
    if confidence is not None:
        event["confidence"] = confidence
    if payment_id is not None:
        event["payment_id"] = payment_id
    if tx_hash is not None:
        event["tx_hash"] = tx_hash
    if receipt_id is not None:
        event["receipt_id"] = receipt_id
    if payment is not None:
        event["payment"] = dict(payment)
    if proof is not None:
        event["proof"] = dict(proof)
    if integration is not None:
        event["integration"] = dict(integration)
    if metadata is not None:
        event["metadata"] = dict(metadata)
    return event
