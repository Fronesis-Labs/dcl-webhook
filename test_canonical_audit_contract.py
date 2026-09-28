"""Canonical DCL Audit Event v1.0 contract tests (FROZEN semantics).

Asserts the CURRENT implementation against
docs/CANONICAL_AUDIT_EVENT_V1_CONTRACT.md. Mismatches must fail; these tests
do not weaken assertions to match the builder.

Event hash algorithm is OPEN and is intentionally untested. Do not compute
sha256 here. Concrete event_id / trace_id formats, global uniqueness,
session_id_fingerprint algorithm, fractional-second rules, and which instant
timestamp denotes are OPEN and are not tested. Inner schemas of payment /
proof / integration / metadata are not invented here.
"""

from __future__ import annotations

import inspect
from datetime import datetime, timedelta

import pytest

from agent_control import (
    Agent,
    AgentControlOrchestrator,
    AgentProposal,
    ControlContext,
    DCLAvailabilityPolicy,
    FakeDCLGuard,
    LocalHardPolicy,
    LocalHardPolicyConfig,
    MockActionExecutor,
    ProposedAction,
    create_audit_event,
)
from agent_control.dcl import DCLEvaluation
import agent_control.canonical_audit as canonical_audit


TRACE_ID = "trace-contract-001"
AGENT_ID = "agent-contract-1"
VALID_UTC_ISO8601 = "2026-09-28T09:44:00Z"
DESTINATION = "0x1111111111111111111111111111111111111111"

_CANONICAL_SERIALIZER_NAMES = (
    "canonical_bytes",
    "canonical_serialize",
    "serialize_canonical",
    "serialize_canonical_event",
    "serialize_canonical_audit_event",
    "to_canonical_bytes",
    "dump_canonical_bytes",
    "canonicalize",
    "canonical_utf8_bytes",
)


def _base_kwargs(**overrides):
    kwargs = dict(
        trace_id=TRACE_ID,
        agent_id=AGENT_ID,
        policy_id="evaluated-policy",
        policy_version="1.2.3",
        verdict="COMMIT",
        timestamp=VALID_UTC_ISO8601,
    )
    kwargs.update(overrides)
    return kwargs


def _event(**overrides):
    return create_audit_event(**_base_kwargs(**overrides))


def _transfer():
    return ProposedAction(
        action_type="transfer",
        payload={
            "asset": "USDC",
            "amount": 25,
            "chain": "base",
            "destination": DESTINATION,
        },
    )


def _policy():
    return LocalHardPolicy(
        LocalHardPolicyConfig(
            max_amount=100.0,
            allowed_chains=frozenset({"base"}),
            allowed_action_types=frozenset({"transfer", "tool_call"}),
            allowed_destinations=frozenset({DESTINATION}),
            require_dcl=True,
        )
    )


def _orchestrator(dcl):
    return AgentControlOrchestrator(
        local_policy=_policy(),
        dcl=dcl,
        executor=MockActionExecutor(),
        availability_policy=DCLAvailabilityPolicy.FAIL_CLOSED,
        audit_event_builder=create_audit_event,
    )


def _orchestrator_tracking_audit_builder(dcl):
    """Same wiring as _orchestrator, plus a spy so tests can prove no event was built."""
    builder_calls = []

    def spy_builder(**kwargs):
        builder_calls.append(kwargs)
        return create_audit_event(**kwargs)

    orch = AgentControlOrchestrator(
        local_policy=_policy(),
        dcl=dcl,
        executor=MockActionExecutor(),
        availability_policy=DCLAvailabilityPolicy.FAIL_CLOSED,
        audit_event_builder=spy_builder,
    )
    return orch, builder_calls


class _DCLWithoutEvaluatedPolicy:
    """Fake DCL that returns a verdict but no evaluated policy identifiers."""

    def __init__(self, *, policy_id, policy_version, verdict="COMMIT"):
        self.policy_id = policy_id
        self.policy_version = policy_version
        self.verdict = verdict

    def evaluate(self, action, context):
        _ = action, context
        return DCLEvaluation(
            available=True,
            verdict=self.verdict,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
        )


def _is_utc_iso8601(value):
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


def _implementation_canonical_serializer():
    for name in _CANONICAL_SERIALIZER_NAMES:
        fn = getattr(canonical_audit, name, None)
        if callable(fn):
            return fn
    for name, obj in vars(canonical_audit).items():
        if name.startswith("_") or not callable(obj):
            continue
        if obj is create_audit_event:
            continue
        lowered = name.lower()
        if "serialize" in lowered or (
            "canonical" in lowered and "event_type" not in lowered and "schema" not in lowered
        ):
            return obj
    return None


def test_event_type_exactly_dcl_audit_evaluated():
    event = _event()
    assert event["event_type"] == "dcl.audit.evaluated"


def test_schema_version_exactly_1_0():
    event = _event()
    assert event["schema_version"] == "1.0"


def test_event_id_required_and_distinct_from_payment_id_tx_hash_receipt_id():
    event = _event(
        event_id="audit-event-1",
        payment_id="payment-1",
        tx_hash="tx-1",
        receipt_id="receipt-1",
    )
    assert event.get("event_id"), "event_id is required when a canonical event is emitted"
    assert event["event_id"] != event.get("payment_id")
    assert event["event_id"] != event.get("tx_hash")
    assert event["event_id"] != event.get("receipt_id")
    assert event["event_id"] != "payment-1"
    assert event["event_id"] != "tx-1"
    assert event["event_id"] != "receipt-1"

    generated = _event()
    assert generated.get("event_id"), "event_id is required even when the caller omits it"

    with pytest.raises(ValueError, match="distinct"):
        _event(event_id="same-id", payment_id="same-id")
    with pytest.raises(ValueError, match="distinct"):
        _event(event_id="same-id", tx_hash="same-id")
    with pytest.raises(ValueError, match="distinct"):
        _event(event_id="same-id", receipt_id="same-id")


def test_trace_id_required():
    event = _event(trace_id=TRACE_ID)
    assert event.get("trace_id"), "trace_id is required when a canonical event is emitted"

    with pytest.raises(ValueError):
        _event(trace_id="")

    kwargs = _base_kwargs()
    del kwargs["trace_id"]
    with pytest.raises((TypeError, ValueError)):
        create_audit_event(**kwargs)


def test_agent_id_is_optional():
    kwargs = _base_kwargs()
    del kwargs["agent_id"]
    try:
        event = create_audit_event(**kwargs)
    except TypeError as exc:
        pytest.fail(
            "FROZEN contract: agent_id is OPTIONAL; "
            f"create_audit_event rejected omission: {exc}"
        )
    assert "agent_id" not in event, (
        "FROZEN contract: optional agent_id must be omitted when absent, "
        f"not emitted as {event.get('agent_id')!r}"
    )


def test_agent_id_presence_does_not_imply_verified_identity():
    event = _event(agent_id="agent-present")
    assert event.get("agent_id") == "agent-present"
    verification_keys = {
        "verified",
        "identity_verified",
        "agent_verified",
        "verified_identity",
        "is_verified",
        "agent_id_verified",
    }
    assert verification_keys.isdisjoint(event.keys())
    for key in event:
        assert "verif" not in key.lower(), (
            f"presence of agent_id must not create a verification claim; found {key!r}"
        )
    meta = event.get("metadata")
    if isinstance(meta, dict):
        assert verification_keys.isdisjoint(meta.keys())
        assert meta.get("verified") is not True
        assert meta.get("identity_verified") is not True


def test_session_id_fingerprint_is_optional():
    try:
        event = create_audit_event(
            **_base_kwargs(),
            session_id_fingerprint="opaque-session-fp",
        )
    except TypeError as exc:
        pytest.fail(
            "FROZEN contract: session_id_fingerprint is OPTIONAL; "
            f"create_audit_event cannot accept the field: {exc}"
        )
    assert event.get("session_id_fingerprint") == "opaque-session-fp"


def test_session_id_fingerprint_omitted_when_absent():
    event = _event()
    assert "session_id_fingerprint" not in event


def test_create_audit_event_rejects_literal_unknown_policy_id():
    try:
        event = _event(policy_id="unknown")
    except (TypeError, ValueError):
        return
    pytest.fail(
        "FROZEN contract: literal policy_id 'unknown' is not a valid fallback; "
        f"create_audit_event emitted policy_id={event.get('policy_id')!r}"
    )


def test_orchestrator_does_not_synthesize_policy_id_from_requested():
    # DCL returns policy_id=None; fail-closed must refuse rather than emit
    # a canonical event that claims the requested/context policy was applied.
    requested = "requested-policy"
    dcl = _DCLWithoutEvaluatedPolicy(policy_id=None, policy_version="3.0")
    proposal = AgentProposal(
        action=_transfer(),
        context=ControlContext(
            trace_id=TRACE_ID,
            agent_id=AGENT_ID,
            policy_id=requested,
            policy_version="requested-ver",
        ),
    )
    orch, builder_calls = _orchestrator_tracking_audit_builder(dcl)
    with pytest.raises(ValueError) as excinfo:
        orch.handle(proposal)
    message = str(excinfo.value)
    assert builder_calls == [], (
        "a canonical audit event MUST NOT claim an applied policy that DCL "
        "did not actually identify."
    )
    assert "cannot emit a canonical audit event without the policy actually evaluated by DCL" in message
    assert "policy_id=None" in message
    assert requested not in message, (
        "a canonical audit event MUST NOT claim an applied policy that DCL "
        "did not actually identify."
    )
    assert "unknown" not in message


def test_create_audit_event_rejects_literal_unknown_policy_version():
    try:
        event = _event(policy_version="unknown")
    except (TypeError, ValueError):
        return
    pytest.fail(
        "FROZEN contract: literal policy_version 'unknown' is not a valid fallback; "
        f"create_audit_event emitted policy_version={event.get('policy_version')!r}"
    )


def test_orchestrator_does_not_fallback_policy_version_to_unknown():
    # DCL returns policy_version=None; fail-closed must refuse rather than
    # substitute "unknown" or the requested/context version.
    dcl = _DCLWithoutEvaluatedPolicy(policy_id="actually-evaluated", policy_version=None)
    proposal = AgentProposal(
        action=_transfer(),
        context=ControlContext(
            trace_id=TRACE_ID,
            agent_id=AGENT_ID,
            policy_id="requested-policy",
            policy_version=None,
        ),
    )
    orch, builder_calls = _orchestrator_tracking_audit_builder(dcl)
    with pytest.raises(ValueError) as excinfo:
        orch.handle(proposal)
    message = str(excinfo.value)
    assert builder_calls == [], (
        "a canonical audit event MUST NOT claim an applied policy that DCL "
        "did not actually identify."
    )
    assert "cannot emit a canonical audit event without the policy actually evaluated by DCL" in message
    assert "policy_version=None" in message
    assert "unknown" not in message
    assert "requested-policy" not in message, (
        "a canonical audit event MUST NOT claim an applied policy that DCL "
        "did not actually identify."
    )


def test_orchestrator_does_not_synthesize_policy_version_from_requested():
    # DCL returns policy_version=""; fail-closed must refuse rather than
    # substitute requested-9 or "unknown".
    dcl = _DCLWithoutEvaluatedPolicy(policy_id="actually-evaluated", policy_version="")
    proposal = AgentProposal(
        action=_transfer(),
        context=ControlContext(
            trace_id=TRACE_ID,
            agent_id=AGENT_ID,
            policy_id="requested-policy",
            policy_version="requested-9",
        ),
    )
    orch, builder_calls = _orchestrator_tracking_audit_builder(dcl)
    with pytest.raises(ValueError) as excinfo:
        orch.handle(proposal)
    message = str(excinfo.value)
    assert builder_calls == [], (
        "a canonical audit event MUST NOT claim an applied policy that DCL "
        "did not actually identify."
    )
    assert "cannot emit a canonical audit event without the policy actually evaluated by DCL" in message
    assert "policy_version=''" in message
    assert "requested-9" not in message, (
        "a canonical audit event MUST NOT claim an applied policy that DCL "
        "did not actually identify."
    )
    assert "unknown" not in message


def test_verdict_exactly_commit_or_no_commit():
    commit = _event(verdict="COMMIT")
    assert commit["verdict"] == "COMMIT"
    no_commit = _event(verdict="NO_COMMIT")
    assert no_commit["verdict"] == "NO_COMMIT"

    for invalid in ("PASS", "FAIL", "ALLOW", "BLOCK", "pass", "fail"):
        with pytest.raises(ValueError):
            _event(verdict=invalid)


def test_timestamp_required_utc_iso8601_string():
    event = _event(timestamp=VALID_UTC_ISO8601)
    assert "timestamp" in event
    assert event["timestamp"] == VALID_UTC_ISO8601
    assert _is_utc_iso8601(event["timestamp"])

    kwargs = _base_kwargs()
    del kwargs["timestamp"]
    defaulted = create_audit_event(**kwargs)
    assert "timestamp" in defaulted
    assert _is_utc_iso8601(defaulted["timestamp"]), (
        "timestamp must be a UTC ISO-8601 string when a canonical event is emitted; "
        f"got {defaulted.get('timestamp')!r}"
    )


def test_timestamp_rejects_unix_numeric():
    try:
        event = _event(timestamp=1759052640.0)
    except (TypeError, ValueError):
        return
    pytest.fail(
        "FROZEN: a Unix numeric timestamp is not a valid v1.0 representation; "
        f"builder accepted timestamp={event.get('timestamp')!r}"
    )


def test_timestamp_rejects_non_iso8601_string():
    try:
        event = _event(timestamp="not-an-iso8601-timestamp")
    except (TypeError, ValueError):
        return
    pytest.fail(
        "FROZEN: timestamp must be UTC ISO-8601; "
        f"builder accepted timestamp={event.get('timestamp')!r}"
    )


def test_optional_extensions_omitted_when_absent_not_null():
    event = _event()
    for name in ("payment", "proof", "integration", "metadata"):
        assert name not in event, (
            f"FROZEN omit-when-absent: optional extension {name!r} must be omitted, "
            f"not emitted as {event.get(name)!r}"
        )


def test_optional_extensions_present_as_opaque_mappings():
    payment = {"opaque": "payment-blob"}
    proof = {"opaque": "proof-blob"}
    integration = {"opaque": "integration-blob"}
    metadata = {"opaque": "metadata-blob"}
    event = _event(
        payment=payment,
        proof=proof,
        integration=integration,
        metadata=metadata,
    )
    assert isinstance(event.get("payment"), dict)
    assert isinstance(event.get("proof"), dict)
    assert isinstance(event.get("integration"), dict)
    assert isinstance(event.get("metadata"), dict)


def test_orchestrator_omits_metadata_when_absent():
    dcl = FakeDCLGuard(verdict="COMMIT", policy_id="evaluated-policy", policy_version="1.2.3")
    agent = Agent(AGENT_ID)
    result = _orchestrator(dcl).handle(agent.propose(_transfer(), trace_id=TRACE_ID))
    assert result.audit_event is not None
    event = result.audit_event
    assert "metadata" not in event, (
        "FROZEN: metadata is an optional extension and must be omitted when absent; "
        "orchestrator always-on metadata is a contract mismatch"
    )


def test_canonical_serialization_deterministic_utf8_lexicographic_bytes():
    serialize = _implementation_canonical_serializer()
    assert serialize is not None, (
        "FROZEN section 11 requires deterministic canonical UTF-8 bytes; "
        "agent_control.canonical_audit exposes no canonical serializer "
        "(create_audit_event returns a Python dict only)"
    )

    event = _event(event_id="evt-canonical-1")
    raw = serialize(event)
    assert isinstance(raw, (bytes, bytearray)), (
        "canonical serialization must produce UTF-8 bytes, "
        f"got {type(raw).__name__}"
    )
    text = bytes(raw).decode("utf-8")
    assert "\n" not in text
    assert ": " not in text
    assert ", " not in text

    again = serialize(event)
    assert bytes(again) == bytes(raw), (
        "same semantic event must produce identical serialized bytes"
    )

    omitted = _event(event_id="evt-canonical-1")
    omitted_bytes = serialize(omitted)
    decoded_keys_ok = True
    try:
        import json

        parsed = json.loads(bytes(omitted_bytes).decode("utf-8"))
    except Exception:
        parsed = None
        decoded_keys_ok = False
    if decoded_keys_ok and isinstance(parsed, dict):
        for name in ("payment", "proof", "integration", "metadata", "session_id_fingerprint"):
            assert name not in parsed
        assert list(parsed.keys()) == sorted(parsed.keys()), (
            "object keys must be in lexicographic order"
        )


def test_canonical_module_does_not_use_chainstate_pipe_hash_as_event_hash():
    event = _event()
    assert "event_hash" not in event
    assert "_content_for_hash" not in event
    assert not hasattr(canonical_audit, "_content_for_hash")
    source = inspect.getsource(create_audit_event)
    assert "sha256" not in source.lower()
    for key, value in event.items():
        if key == "tx_hash":
            continue
        if "hash" in key.lower():
            assert "|" not in str(value), (
                "ChainState pipe-delimited sha256 is not the canonical audit-event hash"
            )
