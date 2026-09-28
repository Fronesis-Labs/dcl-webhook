"""Control-flow orchestrator — wires layers in order, fail closed by default."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agent_control.actions import AgentProposal, ProposedAction
from agent_control.behavior import BehaviorSignalProvider
from agent_control.canonical_audit import create_audit_event
from agent_control.context import ControlContext
from agent_control.dcl import (
    DCLAvailabilityPolicy,
    DCLEvaluation,
    DCLGuard,
    DCLUnavailableError,
)
from agent_control.executor import ActionExecutor, ExecutionResult
from agent_control.policy import (
    InMemorySpendLedger,
    LocalBlockRecord,
    LocalHardPolicy,
    LocalPolicyDecision,
    PolicyVerdict,
)

AuditEventBuilder = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class ControlFlowResult:
    """Outcome of one proposal. `audit_event` is set only after a DCL verdict."""

    trace_id: str
    outcome: str
    executed: bool
    local_policy: LocalPolicyDecision
    context: ControlContext
    dcl_evaluation: DCLEvaluation | None = None
    audit_event: dict[str, Any] | None = None
    local_block: LocalBlockRecord | None = None
    execution: ExecutionResult | None = None

    @property
    def dcl_called(self) -> bool:
        return self.dcl_evaluation is not None or self.outcome == "DCL_UNAVAILABLE"


class AgentControlOrchestrator:
    """Accept a proposal, gate it, evaluate DCL, audit, then maybe execute."""

    def __init__(
        self,
        *,
        local_policy: LocalHardPolicy,
        dcl: DCLGuard,
        executor: ActionExecutor,
        behavior: BehaviorSignalProvider | None = None,
        availability_policy: DCLAvailabilityPolicy = DCLAvailabilityPolicy.FAIL_CLOSED,
        audit_event_builder: AuditEventBuilder = create_audit_event,
        spend_ledger: InMemorySpendLedger | None = None,
    ) -> None:
        if availability_policy is not DCLAvailabilityPolicy.FAIL_CLOSED:
            raise NotImplementedError(
                f"DCL availability policy {availability_policy!r} is not implemented in v0.1; "
                "only fail-closed is supported."
            )
        self.local_policy = local_policy
        self.dcl = dcl
        self.executor = executor
        self.behavior = behavior
        self.availability_policy = availability_policy
        self.audit_event_builder = audit_event_builder
        self.spend_ledger = spend_ledger or local_policy.spend_ledger

    def handle(self, proposal: AgentProposal) -> ControlFlowResult:
        action = proposal.action
        context = proposal.context

        if self.behavior is not None:
            signal = self.behavior.collect(action, context)
            context = context.with_behavior_signal(signal)

        local = self.local_policy.evaluate(action, context)
        if local.verdict is PolicyVerdict.BLOCK:
            return ControlFlowResult(
                trace_id=context.trace_id,
                outcome="LOCAL_BLOCK",
                executed=False,
                local_policy=local,
                context=context,
                dcl_evaluation=None,
                audit_event=None,
                local_block=self.local_policy.build_block_record(action, context, local),
            )

        try:
            dcl_result = self.dcl.evaluate(action, context)
        except DCLUnavailableError:
            return self._unavailable(action, context, local)

        if not dcl_result.available or dcl_result.verdict is None:
            return self._unavailable(action, context, local, dcl_result=dcl_result)

        applied_policy_id = dcl_result.policy_id
        applied_policy_version = dcl_result.policy_version
        if (
            not isinstance(applied_policy_id, str)
            or not applied_policy_id
            or applied_policy_id == "unknown"
            or not isinstance(applied_policy_version, str)
            or not applied_policy_version
            or applied_policy_version == "unknown"
        ):
            raise ValueError(
                "cannot emit a canonical audit event without the policy actually "
                f"evaluated by DCL (policy_id={applied_policy_id!r}, "
                f"policy_version={applied_policy_version!r})"
            )

        audit_kwargs: dict[str, Any] = {
            "trace_id": context.trace_id,
            "agent_id": context.agent_id,
            "policy_id": applied_policy_id,
            "policy_version": applied_policy_version,
            "verdict": dcl_result.verdict,
            "reason": dcl_result.reason,
            "confidence": dcl_result.confidence,
            "tx_hash": dcl_result.tx_hash,
        }
        audit_event = self.audit_event_builder(**audit_kwargs)

        if dcl_result.verdict != "COMMIT":
            return ControlFlowResult(
                trace_id=context.trace_id,
                outcome="DCL_NO_COMMIT",
                executed=False,
                local_policy=local,
                context=context,
                dcl_evaluation=dcl_result,
                audit_event=audit_event,
            )

        execution = self.executor.execute(action, context)
        self._record_spend(action, context)
        return ControlFlowResult(
            trace_id=context.trace_id,
            outcome="EXECUTED",
            executed=True,
            local_policy=local,
            context=context,
            dcl_evaluation=dcl_result,
            audit_event=audit_event,
            execution=execution,
        )

    def _unavailable(
        self,
        action: ProposedAction,
        context: ControlContext,
        local: LocalPolicyDecision,
        dcl_result: DCLEvaluation | None = None,
    ) -> ControlFlowResult:
        # Fail closed: DCL unavailable ≠ COMMIT. Do not invent a verdict or event.
        _ = action
        return ControlFlowResult(
            trace_id=context.trace_id,
            outcome="DCL_UNAVAILABLE",
            executed=False,
            local_policy=local,
            context=context,
            dcl_evaluation=dcl_result,
            audit_event=None,
        )

    def _record_spend(self, action: ProposedAction, context: ControlContext) -> None:
        if self.spend_ledger is None:
            return
        raw = action.get(self.local_policy.config.amount_field)
        try:
            amount = float(raw)
        except (TypeError, ValueError):
            return
        key = self.local_policy.spend_key or context.agent_id
        self.spend_ledger.add(key, amount)
