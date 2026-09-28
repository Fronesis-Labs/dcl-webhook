"""Material local proof: DCL COMMIT/NO_COMMIT actually gates a filesystem side effect.

Chain (existing code only; no second policy engine):

    Agent proposal
    → behavior signal (advisory; cannot authorize)
    → Local Hard Policy (ALLOW on both traces)
    → EvaluatePolicyDCLGuard
    → EvaluatePolicyOracle
    → audit_logic.evaluate_policy
    → COMMIT or NO_COMMIT
    → executor only when the existing orchestrator calls it on COMMIT
    → canonical audit event via create_audit_event / canonical_bytes

The executor lives only in this test file. It writes a file because it was
invoked; it does not inspect the DCL verdict.
"""

from __future__ import annotations

from pathlib import Path

from agent_control import (
    Agent,
    AgentControlOrchestrator,
    DCLAvailabilityPolicy,
    EvaluatePolicyDCLGuard,
    EvaluatePolicyOracle,
    LocalHardPolicy,
    LocalHardPolicyConfig,
    PolicyVerdict,
    ProposedAction,
    StaticBehaviorSignalProvider,
    create_audit_event,
)
from agent_control.behavior import BehaviorSignal
from agent_control.canonical_audit import canonical_bytes
from agent_control.dcl import COMMIT, NO_COMMIT
from agent_control.executor import ExecutionResult


TRACE_ID = "trace-real-action-e2e-001"
AGENT_ID = "agent-real-action-1"
REQUESTED_POLICY_ID = "default"
FORBIDDEN_PHRASE_DEFAULT = "jailbreak"
MAX_AMOUNT = 100.0
ACTION_TYPE = "record"
SIDE_EFFECT_FILENAME = "record.txt"


class FileSideEffectExecutor:
    """Test-only executor: write one file when execute() is called.

    Does not read DCL verdict, orchestrator outcome, or policy. A write means
    the orchestrator invoked execute().
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.path = self.directory / SIDE_EFFECT_FILENAME
        self.calls: list = []

    def execute(self, action, context) -> ExecutionResult:
        self.calls.append((action, context))
        self.path.write_text("side-effect recorded\n", encoding="utf-8")
        return ExecutionResult(
            executed=True,
            detail="wrote side-effect file",
            payload={"path": str(self.path), "action_type": action.action_type},
        )


def _record(*, amount: float = 25.0, chain: str = "base", **payload_extra) -> ProposedAction:
    payload = {
        "amount": amount,
        "chain": chain,
    }
    payload.update(payload_extra)
    return ProposedAction(action_type=ACTION_TYPE, payload=payload)


def _local_policy() -> LocalHardPolicy:
    return LocalHardPolicy(
        LocalHardPolicyConfig(
            max_amount=MAX_AMOUNT,
            allowed_chains=frozenset({"base"}),
            allowed_action_types=frozenset({ACTION_TYPE}),
            require_dcl=True,
        )
    )


def _orchestrator(*, executor: FileSideEffectExecutor) -> AgentControlOrchestrator:
    behavior = StaticBehaviorSignalProvider(
        BehaviorSignal(
            risk_score=0.4,
            reason="advisory only; must not authorize",
            source="static-real-action-e2e",
        )
    )
    return AgentControlOrchestrator(
        local_policy=_local_policy(),
        dcl=EvaluatePolicyDCLGuard(oracle=EvaluatePolicyOracle()),
        executor=executor,
        behavior=behavior,
        availability_policy=DCLAvailabilityPolicy.FAIL_CLOSED,
        audit_event_builder=create_audit_event,
    )


def _assert_canonical_bytes_deterministic(event) -> bytes:
    raw = canonical_bytes(event)
    assert isinstance(raw, bytes)
    assert raw
    assert raw.decode("utf-8")
    assert canonical_bytes(event) == raw
    return raw


def _assert_shared_audit_fields(event, dcl_eval) -> None:
    assert event is not None
    assert event["verdict"] == dcl_eval.verdict
    assert event["policy_id"] == "default" == dcl_eval.policy_id
    assert event["policy_version"] == "1.0.0" == dcl_eval.policy_version
    assert event["event_type"] == "dcl.audit.evaluated"
    assert event["schema_version"] == "1.0"
    assert event["confidence"] == dcl_eval.confidence
    assert event["reason"] == dcl_eval.reason
    _assert_canonical_bytes_deterministic(event)


def _fmt_bool(value: bool) -> str:
    return "true" if value else "false"


def _fmt_side_effect(exists: bool) -> str:
    return "CREATED" if exists else "NOT_CREATED"


def _print_block(title: str, *, action, result, executor: FileSideEffectExecutor) -> None:
    dcl_eval = result.dcl_evaluation
    event = result.audit_event
    print(f"{title}")
    print(f"  proposed_action: {action.action_type} {dict(action.payload)}")
    print(f"  DCL: {dcl_eval.verdict if dcl_eval is not None else None}")
    print(f"  confidence: {dcl_eval.confidence if dcl_eval is not None else None}")
    print(f"  reason: {dcl_eval.reason if dcl_eval is not None else None}")
    print(f"  executor_called: {_fmt_bool(len(executor.calls) > 0)}")
    print(f"  side_effect: {_fmt_side_effect(executor.path.exists())}")
    print(f"  audit_event: {event['event_id'] if event is not None else None}")


def test_agent_control_real_action_commit_writes_no_commit_does_not(tmp_path):
    """COMMIT writes record.txt; NO_COMMIT never calls execute() so the file stays absent."""
    commit_dir = tmp_path / "commit"
    no_commit_dir = tmp_path / "no_commit"
    commit_dir.mkdir()
    no_commit_dir.mkdir()

    commit_action = _record()
    no_commit_action = _record(note=FORBIDDEN_PHRASE_DEFAULT)

    commit_executor = FileSideEffectExecutor(commit_dir)
    no_commit_executor = FileSideEffectExecutor(no_commit_dir)

    assert not commit_executor.path.exists()
    assert not no_commit_executor.path.exists()

    commit_proposal = Agent(AGENT_ID).propose(
        commit_action, trace_id=TRACE_ID, policy_id=REQUESTED_POLICY_ID
    )
    commit_result = _orchestrator(executor=commit_executor).handle(commit_proposal)

    no_commit_proposal = Agent(AGENT_ID).propose(
        no_commit_action,
        trace_id=TRACE_ID + "-nocommit",
        policy_id=REQUESTED_POLICY_ID,
    )
    no_commit_result = _orchestrator(executor=no_commit_executor).handle(no_commit_proposal)

    # COMMIT: local ALLOW, real engine COMMIT, executor ran, file created.
    assert commit_result.local_policy.verdict is PolicyVerdict.ALLOW
    commit_eval = commit_result.dcl_evaluation
    assert commit_eval is not None
    assert commit_eval.verdict == COMMIT
    assert len(commit_executor.calls) == 1
    assert commit_executor.path.exists()
    assert commit_executor.path.is_file()
    commit_event = commit_result.audit_event
    _assert_shared_audit_fields(commit_event, commit_eval)
    assert commit_event["verdict"] == "COMMIT"

    # NO_COMMIT: local ALLOW (same amount/chain/action type); engine forbids
    # via default policy phrase. Executor never ran; file still absent.
    assert no_commit_result.local_policy.verdict is PolicyVerdict.ALLOW
    no_commit_eval = no_commit_result.dcl_evaluation
    assert no_commit_eval is not None
    assert no_commit_eval.verdict == NO_COMMIT
    assert len(no_commit_executor.calls) == 0
    assert not no_commit_executor.path.exists()
    no_commit_event = no_commit_result.audit_event
    _assert_shared_audit_fields(no_commit_event, no_commit_eval)
    assert no_commit_event["verdict"] == "NO_COMMIT"
    if "jailbreak" in str(no_commit_eval.reason).lower():
        assert "jailbreak" in str(no_commit_event["reason"]).lower()

    print()
    print("=== FRONESIS AGENT CONTROL E2E ===")
    print()
    _print_block(
        "COMMIT",
        action=commit_action,
        result=commit_result,
        executor=commit_executor,
    )
    print()
    _print_block(
        "NO_COMMIT",
        action=no_commit_action,
        result=no_commit_result,
        executor=no_commit_executor,
    )
    print()
    print("RESULT:")
    print("  Agent proposal was evaluated by DCL.")
    print("  COMMIT produced the side effect.")
    print("  NO_COMMIT prevented the side effect.")
    print("  Both decisions produced canonical audit events.")
    print()
