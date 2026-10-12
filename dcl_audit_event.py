"""
DCL Audit Event v1.0
Canonical audit event contract for DCL integrations.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional
import uuid


SCHEMA_VERSION = "1.0"
EVENT_TYPE = "dcl.audit.evaluated"


def create_audit_event(
    *,
    service: str,
    producer: str,
    route: str,
    agent_id: str,
    identity_source: str,
    identity_confidence: str,
    task_type: str,
    policy_id: str,
    policy_version: str,
    verdict: str,
    trace_id: Optional[str] = None,
    session_id_fingerprint: Optional[str] = None,
    payment: Optional[Dict[str, Any]] = None,
    proof: Optional[Dict[str, Any]] = None,
    integration: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create one canonical DCL Audit Event.

    trace_id is mandatory at the event level. If the producer does not
    provide one, this function creates it before evaluation is recorded.
    """

    if not trace_id:
        trace_id = f"tr_{uuid.uuid4().hex}"

    event = {
        "schema_version": SCHEMA_VERSION,
        "event_id": f"evt_{uuid.uuid4().hex}",
        "event_type": EVENT_TYPE,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "service": service,
        "producer": producer,
        "route": route,
        "agent_id": agent_id,
        "identity_source": identity_source,
        "identity_confidence": identity_confidence,
        "trace_id": trace_id,
        "task_type": task_type,
        "policy_id": policy_id,
        "policy_version": policy_version,
        "verdict": verdict,
    }

    if session_id_fingerprint is not None:
        event["session_id_fingerprint"] = session_id_fingerprint

    if payment is not None:
        event["payment"] = payment

    if proof is not None:
        event["proof"] = proof

    if integration is not None:
        event["integration"] = integration

    if metadata is not None:
        event["metadata"] = metadata

    return event
