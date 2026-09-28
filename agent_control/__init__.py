"""Fronesis Agent Control Architecture v0.1 — local reference control flow.

This package is a testable, offline reference for how an autonomous agent
should be gated BEFORE an action executes. It does not replace DCL Trust
Oracle, production servers, x402, ERC-8004, or Canonical DCL Audit Event v1.0.
"""

from agent_control.actions import Agent, AgentProposal, ProposedAction
from agent_control.behavior import BehaviorSignal, BehaviorSignalProvider, StaticBehaviorSignalProvider
from agent_control.canonical_audit import (
    CANONICAL_EVENT_TYPE,
    CANONICAL_SCHEMA_VERSION,
    create_audit_event,
)
from agent_control.context import ControlContext
from agent_control.dcl import (
    DCLAvailabilityPolicy,
    DCLEvaluation,
    DCLGuard,
    DCLUnavailableError,
    EvaluatePolicyDCLGuard,
    FakeDCLGuard,
    UnavailableDCLGuard,
)
from agent_control.dcl_oracle import EvaluatePolicyOracle
from agent_control.executor import ActionExecutor, MockActionExecutor
from agent_control.orchestrator import AgentControlOrchestrator, ControlFlowResult
from agent_control.policy import (
    InMemorySpendLedger,
    LocalBlockRecord,
    LocalHardPolicy,
    LocalHardPolicyConfig,
    LocalPolicyDecision,
    PolicyVerdict,
)

__all__ = [
    "Agent",
    "AgentProposal",
    "ProposedAction",
    "BehaviorSignal",
    "BehaviorSignalProvider",
    "StaticBehaviorSignalProvider",
    "CANONICAL_EVENT_TYPE",
    "CANONICAL_SCHEMA_VERSION",
    "create_audit_event",
    "ControlContext",
    "DCLAvailabilityPolicy",
    "DCLEvaluation",
    "DCLGuard",
    "DCLUnavailableError",
    "EvaluatePolicyDCLGuard",
    "EvaluatePolicyOracle",
    "FakeDCLGuard",
    "UnavailableDCLGuard",
    "ActionExecutor",
    "MockActionExecutor",
    "AgentControlOrchestrator",
    "ControlFlowResult",
    "InMemorySpendLedger",
    "LocalBlockRecord",
    "LocalHardPolicy",
    "LocalHardPolicyConfig",
    "LocalPolicyDecision",
    "PolicyVerdict",
]
