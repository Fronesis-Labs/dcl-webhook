"""Local end-to-end reference traces for Agent Control Architecture v0.1.

Full chain (EvaluatePolicyDCLGuard → EvaluatePolicyOracle →
audit_logic.evaluate_policy — no network, no production DCL, no ChainState):

    Agent proposal
    → Behavior signal
    → Local Hard Policy
    → EvaluatePolicyDCLGuard
    → DCL Adapter (EvaluatePolicyOracle)
    → existing audit_logic.evaluate_policy Oracle
    → COMMIT / NO_COMMIT (engine verdict)
    → Canonical Audit Event v1.0
    → canonical_bytes()
    → Mock Executor

Three traces must remain obvious:
  1. COMMIT     — real engine COMMIT through the adapter, action executes, canonical event exists
  2. NO_COMMIT  — real engine NO_COMMIT through the adapter, action does not execute, canonical event exists
  3. LOCAL_BLOCK — DCL / oracle is not called, action does not execute, no canonical event
"""

from __future__ import annotations

from agent_control import (
    Agent,
    AgentControlOrchestrator,
    CANONICAL_EVENT_TYPE,
    CANONICAL_SCHEMA_VERSION,
    ControlContext,
    DCLAvailabilityPolicy,
    EvaluatePolicyDCLGuard,
    EvaluatePolicyOracle,
    FakeDCLGuard,
    LocalHardPolicy,
    LocalHardPolicyConfig,
    MockActionExecutor,
    PolicyVerdict,
    ProposedAction,
    StaticBehaviorSignalProvider,
    create_audit_event,
)
from agent_control.behavior import BehaviorSignal
from agent_control.canonical_audit import canonical_bytes
from agent_control.dcl import COMMIT, NO_COMMIT


TRACE_ID = "trace-e2e-ref-001"
AGENT_ID = "agent-e2e-ref-1"
DESTINATION = "0x1111111111111111111111111111111111111111"
# Requested id exists in audit_logic.BUILTIN_POLICIES; engine version is "1.0.0".
REQUESTED_POLICY_ID = "default"
FORBIDDEN_PHRASE_DEFAULT = "jailbreak"
MAX_AMOUNT = 100.0


class RecordingAudit:
    """Wraps create_audit_event so tests can prove it was or was not called."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def __call__(self, **kwargs):
        event = create_audit_event(**kwargs)
        self.events.append(event)
        return event


class CallRecordingDCL:
    """Spy around a DCLGuard. Records evaluate() calls; does not invent a verdict."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls: list = []

    def evaluate(self, action, context):
        self.calls.append((action, context))
        return self.inner.evaluate(action, context)


class RecordingOracle:
    """Spy around EvaluatePolicyOracle. Records evaluate(); does not invent a verdict."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls: list = []
        self.results: list = []

    def evaluate(self, response: str, policy_yaml: str):
        self.calls.append((response, policy_yaml))
        result = self.inner.evaluate(response, policy_yaml)
        self.results.append(result)
        return result


def _transfer(*, amount: float = 25.0, chain: str = "base", **payload_extra) -> ProposedAction:
    payload = {
        "asset": "USDC",
        "amount": amount,
        "chain": chain,
        "destination": DESTINATION,
    }
    payload.update(payload_extra)
    return ProposedAction(
        action_type="transfer",
        payload=payload,
    )


def _local_policy() -> LocalHardPolicy:
    return LocalHardPolicy(
        LocalHardPolicyConfig(
            max_amount=MAX_AMOUNT,
            allowed_chains=frozenset({"base"}),
            allowed_action_types=frozenset({"transfer"}),
            allowed_destinations=frozenset({DESTINATION}),
            require_dcl=True,
        )
    )


def _agent_proposal(*, amount: float = 25.0, chain: str = "base", **payload_extra):
    return Agent(AGENT_ID).propose(
        _transfer(amount=amount, chain=chain, **payload_extra),
        trace_id=TRACE_ID,
        policy_id=REQUESTED_POLICY_ID,
    )


def _orchestrator(*, dcl, executor, behavior, audit=None):
    return AgentControlOrchestrator(
        local_policy=_local_policy(),
        dcl=dcl,
        executor=executor,
        behavior=behavior,
        availability_policy=DCLAvailabilityPolicy.FAIL_CLOSED,
        audit_event_builder=audit or create_audit_event,
    )


def _guard_with_recording_oracle():
    oracle = RecordingOracle(EvaluatePolicyOracle())
    dcl = CallRecordingDCL(EvaluatePolicyDCLGuard(oracle=oracle))
    return dcl, oracle


def _assert_behavior_collected_and_advisory(provider, result, *, expect_dcl_saw_signal: bool, dcl):
    """Signal is collected and attached before local policy / DCL; it cannot authorize."""
    assert provider.calls, "behavior signal must be collected before local policy and DCL"
    assert result.context.behavior_signal is provider.signal
    assert result.local_policy is not None
    if expect_dcl_saw_signal:
        assert dcl.calls, "local ALLOW must still call DCL; a signal must not skip it"
        assert dcl.calls[0][1].behavior_signal is provider.signal
    else:
        assert dcl.calls == [], "local BLOCK must not call DCL; a signal must not skip local policy"


def _assert_canonical_event_matches_dcl_evaluation(event, proposal, dcl_evaluation) -> None:
    assert event is not None
    assert event["event_type"] == CANONICAL_EVENT_TYPE == "dcl.audit.evaluated"
    assert event["schema_version"] == CANONICAL_SCHEMA_VERSION == "1.0"
    assert proposal.context.trace_id == event["trace_id"] == TRACE_ID
    assert event["verdict"] == dcl_evaluation.verdict
    assert event["policy_id"] == dcl_evaluation.policy_id
    assert event["policy_version"] == dcl_evaluation.policy_version
    assert event["policy_id"] not in ("", "unknown")
    assert event["policy_version"] not in ("", "unknown")
    _assert_canonical_bytes_deterministic(event)


def _assert_canonical_bytes_deterministic(event) -> bytes:
    raw = canonical_bytes(event)
    assert isinstance(raw, bytes)
    assert raw
    assert raw.decode("utf-8")
    assert canonical_bytes(event) == raw
    return raw


def _assert_matches_oracle_result(dcl_eval, oracle, *, expected_policy_id: str) -> None:
    assert len(oracle.calls) == 1
    assert len(oracle.results) == 1
    verdict, _confidence, _reason, policy_version = oracle.results[0]
    assert dcl_eval.verdict == verdict
    assert dcl_eval.policy_version == policy_version
    assert dcl_eval.policy_id == expected_policy_id
    assert dcl_eval.policy_id not in ("", "unknown")
    assert dcl_eval.policy_version not in ("", "unknown", None)


def test_e2e_commit_action_executes_and_canonical_event_exists():
    """Trace 1: ALLOW → adapter → real evaluate_policy COMMIT → executor → canonical event.

    BUILTIN_POLICIES['default'] has no required_patterns; a transfer payload without
    forbidden phrases ("ignore previous instructions", "jailbreak", "bypass safety")
    must COMMIT. Decision arrives through EvaluatePolicyOracle, not a direct
    evaluate_policy call inside the guard.
    """
    high_risk = BehaviorSignal(
        risk_score=0.92,
        reason="anomalous but advisory only",
        source="static-e2e",
    )
    provider = StaticBehaviorSignalProvider(high_risk)
    dcl, oracle = _guard_with_recording_oracle()
    executor = MockActionExecutor()
    proposal = _agent_proposal()

    result = _orchestrator(dcl=dcl, executor=executor, behavior=provider).handle(proposal)

    _assert_behavior_collected_and_advisory(
        provider, result, expect_dcl_saw_signal=True, dcl=dcl
    )
    assert result.local_policy.verdict is PolicyVerdict.ALLOW
    dcl_eval = result.dcl_evaluation
    assert dcl_eval is not None
    assert dcl_eval.available is True
    assert dcl_eval.verdict == COMMIT
    _assert_matches_oracle_result(dcl_eval, oracle, expected_policy_id=REQUESTED_POLICY_ID)
    assert result.outcome == "EXECUTED"
    assert result.executed is True
    assert len(executor.calls) == 1
    assert result.execution is not None
    assert result.execution.executed is True
    _assert_canonical_event_matches_dcl_evaluation(result.audit_event, proposal, dcl_eval)


def test_e2e_no_commit_action_does_not_execute_and_canonical_event_exists():
    """Trace 2: ALLOW → adapter → real evaluate_policy NO_COMMIT → no execute → event.

    Same builtin YAML as COMMIT. The adapter json.dumps the action, so a payload
    string containing forbidden phrase 'jailbreak' is visible to evaluate_policy.
    Local hard policy still ALLOWs (amount/chain/destination unchanged).
    """
    low_risk = BehaviorSignal(
        risk_score=0.02,
        reason="looks safe but still advisory",
        source="static-e2e",
    )
    provider = StaticBehaviorSignalProvider(low_risk)
    dcl, oracle = _guard_with_recording_oracle()
    executor = MockActionExecutor()
    proposal = _agent_proposal(note=FORBIDDEN_PHRASE_DEFAULT)

    result = _orchestrator(dcl=dcl, executor=executor, behavior=provider).handle(proposal)

    _assert_behavior_collected_and_advisory(
        provider, result, expect_dcl_saw_signal=True, dcl=dcl
    )
    assert result.local_policy.verdict is PolicyVerdict.ALLOW
    dcl_eval = result.dcl_evaluation
    assert dcl_eval is not None
    assert dcl_eval.available is True
    assert dcl_eval.verdict == NO_COMMIT
    _assert_matches_oracle_result(dcl_eval, oracle, expected_policy_id=REQUESTED_POLICY_ID)
    assert result.outcome == "DCL_NO_COMMIT"
    assert result.executed is False
    assert executor.calls == []
    assert result.execution is None
    _assert_canonical_event_matches_dcl_evaluation(result.audit_event, proposal, dcl_eval)


def test_e2e_local_hard_policy_block_skips_dcl_and_does_not_create_canonical_event():
    """Trace 3: local BLOCK → DCL / oracle is not called → no execute → no dcl.audit.evaluated."""
    low_risk = BehaviorSignal(
        risk_score=0.01,
        reason="quiet; must not skip local hard policy",
        source="static-e2e",
    )
    provider = StaticBehaviorSignalProvider(low_risk)
    # Zero-call spy: this trace must not reach the DCL guard or the oracle.
    dcl = FakeDCLGuard(verdict=COMMIT)
    executor = MockActionExecutor()
    audit = RecordingAudit()
    # Amount over max_amount is a deterministic local hard-policy violation.
    proposal = _agent_proposal(amount=MAX_AMOUNT + 400.0)

    result = _orchestrator(
        dcl=dcl, executor=executor, behavior=provider, audit=audit
    ).handle(proposal)

    _assert_behavior_collected_and_advisory(
        provider, result, expect_dcl_saw_signal=False, dcl=dcl
    )
    assert result.local_policy.verdict is PolicyVerdict.BLOCK
    assert result.local_policy.rule == "max_amount"
    assert result.outcome == "LOCAL_BLOCK"
    assert result.executed is False
    assert executor.calls == []
    assert result.execution is None
    assert result.dcl_evaluation is None
    assert result.dcl_called is False
    assert result.audit_event is None
    assert audit.events == []
    assert result.local_block is not None
    assert result.local_block.record_type == "local.policy.blocked"
    assert result.local_block.record_type != CANONICAL_EVENT_TYPE
    assert result.local_block.trace_id == proposal.context.trace_id == TRACE_ID


def test_evaluate_policy_dcl_guard_calls_injected_oracle_not_evaluate_policy():
    """The guard must call the injected oracle; it does not need evaluate_policy itself."""

    class FakeOracle:
        def __init__(self) -> None:
            self.calls: list = []

        def evaluate(self, response: str, policy_yaml: str):
            self.calls.append((response, policy_yaml))
            return "NO_COMMIT", 0.11, "injected fake oracle", "injected-9.9.9"

    oracle = FakeOracle()
    guard = EvaluatePolicyDCLGuard(oracle=oracle)
    action = _transfer()
    context = ControlContext(
        trace_id=TRACE_ID,
        agent_id=AGENT_ID,
        policy_id=REQUESTED_POLICY_ID,
    )

    result = guard.evaluate(action, context)

    assert len(oracle.calls) == 1
    assert isinstance(oracle.calls[0][0], str)
    assert isinstance(oracle.calls[0][1], str)
    assert result.available is True
    assert result.verdict == NO_COMMIT
    assert result.reason == "injected fake oracle"
    assert result.confidence == 0.11
    assert result.policy_id == REQUESTED_POLICY_ID
    assert result.policy_version == "injected-9.9.9"
    # A real evaluate_policy on a clean transfer would COMMIT with version 1.0.0.
    assert result.verdict != COMMIT
    assert result.policy_version != "1.0.0"
