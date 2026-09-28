"""Layer 4 — DCL adapter.

Do not implement a second DCL. This module is a thin interface around the
existing evaluation API already in this repository:

    audit_logic.evaluate_policy(response, policy_yaml)
        -> (verdict, confidence, reason, policy_version)
        verdict is "COMMIT" or "NO_COMMIT"

Production servers (webhook_server / mcp_server / bazaar_server) wrap that
function and append a dcl-core ChainState record. Those servers are NOT
called from this reference architecture.

Conceptual mapping
------------------
    adapter.evaluate(action, context) -> DCLEvaluation(COMMIT | NO_COMMIT)

    maps to EvaluatePolicyOracle.evaluate(serialized_action, policy_yaml)
    which calls audit_logic.evaluate_policy, or to a remote client that
    already speaks the same verdicts:

    * REST POST /evaluate/{tier} with {response, agent_id, policy?, task_type?}
    * MCP dcl_evaluate_fast / dcl_evaluate_strict / ...
    * TS @fronesis-labs/dcl-sdk DclClient.evaluate(tier, {response, agent_id, task_type})

There is no Python DCLClient / DCLGuard class in this repository. The
authoritative local engine is evaluate_policy; remote enforcement remains
an external DCL Trust Oracle. Tests inject FakeDCLGuard / UnavailableDCLGuard
and must not call production services.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

from agent_control.actions import ProposedAction
from agent_control.context import ControlContext
from agent_control.dcl_oracle import EvaluatePolicyOracle

COMMIT: Literal["COMMIT"] = "COMMIT"
NO_COMMIT: Literal["NO_COMMIT"] = "NO_COMMIT"
DCLVerdict = Literal["COMMIT", "NO_COMMIT"]


class DCLUnavailableError(Exception):
    """DCL could not be reached or did not return a verdict. Never treat as COMMIT."""


class DCLAvailabilityPolicy(str, Enum):
    """Extension point for unavailable-DCL behavior.

    v0.1 implements FAIL_CLOSED only. Additional members may be added later
    but must not be treated as COMMIT unless explicitly implemented.
    """

    FAIL_CLOSED = "fail_closed"


@dataclass(frozen=True)
class DCLEvaluation:
    """Result of a completed DCL evaluation. `available=False` means no verdict."""

    available: bool
    verdict: DCLVerdict | None = None
    reason: str = ""
    confidence: float | None = None
    policy_id: str | None = None
    policy_version: str | None = None
    tx_hash: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw", dict(self.raw))
        if self.available:
            if self.verdict not in (COMMIT, NO_COMMIT):
                raise ValueError("available DCL evaluation must carry COMMIT or NO_COMMIT")
        elif self.verdict is not None:
            raise ValueError("unavailable DCL evaluation must not invent a verdict")

    @property
    def is_commit(self) -> bool:
        return self.available and self.verdict == COMMIT


@runtime_checkable
class DCLGuard(Protocol):
    """Adapter around existing DCL evaluation. Implementable by wrapping evaluate_policy
    or a remote DCL client; tests inject fakes instead.
    """

    def evaluate(self, action: ProposedAction, context: ControlContext) -> DCLEvaluation:
        """Return COMMIT / NO_COMMIT, or available=False / raise DCLUnavailableError."""


class FakeDCLGuard:
    """In-memory DCL for tests. Records every call so context propagation can be asserted."""

    def __init__(
        self,
        *,
        verdict: DCLVerdict = COMMIT,
        reason: str = "fake DCL checks passed",
        confidence: float = 0.95,
        policy_id: str = "default",
        policy_version: str = "1.0.0",
        tx_hash: str | None = None,
    ) -> None:
        self.verdict = verdict
        self.reason = reason
        self.confidence = confidence
        self.policy_id = policy_id
        self.policy_version = policy_version
        self.tx_hash = tx_hash
        self.calls: list[tuple[ProposedAction, ControlContext]] = []

    def evaluate(self, action: ProposedAction, context: ControlContext) -> DCLEvaluation:
        self.calls.append((action, context))
        return DCLEvaluation(
            available=True,
            verdict=self.verdict,
            reason=self.reason,
            confidence=self.confidence,
            policy_id=self.policy_id or context.policy_id,
            policy_version=self.policy_version,
            tx_hash=self.tx_hash,
        )


class UnavailableDCLGuard:
    """DCL that cannot produce a verdict. Orchestrator must fail closed."""

    def __init__(self, message: str = "DCL unavailable") -> None:
        self.message = message
        self.calls: list[tuple[ProposedAction, ControlContext]] = []
        self.raise_error = True

    def evaluate(self, action: ProposedAction, context: ControlContext) -> DCLEvaluation:
        self.calls.append((action, context))
        if self.raise_error:
            raise DCLUnavailableError(self.message)
        return DCLEvaluation(available=False, reason=self.message)


class EvaluatePolicyDCLGuard:
    """Serializes the action, selects builtin YAML, and calls the DCL Oracle.

    Default oracle is EvaluatePolicyOracle (audit_logic.evaluate_policy).
    This guard does not import or call evaluate_policy itself. Does not
    append to ChainState, does not call production HTTP/MCP endpoints,
    and does not take payment.
    """

    def __init__(
        self,
        policy_yaml: str | None = None,
        *,
        oracle: EvaluatePolicyOracle | None = None,
    ) -> None:
        self.policy_yaml = policy_yaml
        self.oracle = oracle if oracle is not None else EvaluatePolicyOracle()

    def evaluate(self, action: ProposedAction, context: ControlContext) -> DCLEvaluation:
        from audit_logic import BUILTIN_POLICIES

        policy_yaml = self.policy_yaml
        if policy_yaml is None:
            requested = context.policy_id
            if requested in BUILTIN_POLICIES:
                applied_policy_id = requested
                policy_yaml = BUILTIN_POLICIES[applied_policy_id]
            else:
                applied_policy_id = "default"
                policy_yaml = BUILTIN_POLICIES["default"]
        else:
            applied_policy_id = context.policy_id

        response = _serialize_for_evaluate_policy(action, context)
        verdict, confidence, reason, policy_version = self.oracle.evaluate(
            response, policy_yaml
        )
        if verdict not in (COMMIT, NO_COMMIT):
            raise DCLUnavailableError(f"evaluate_policy returned unexpected verdict {verdict!r}")
        return DCLEvaluation(
            available=True,
            verdict=verdict,
            reason=reason,
            confidence=confidence,
            policy_id=applied_policy_id,
            policy_version=policy_version,
            raw={"engine": "audit_logic.evaluate_policy"},
        )


def _serialize_for_evaluate_policy(action: ProposedAction, context: ControlContext) -> str:
    body: dict[str, Any] = {
        "action_type": action.action_type,
        "payload": dict(action.payload),
        "agent_id": context.agent_id,
        "trace_id": context.trace_id,
    }
    if context.behavior_signal is not None:
        body["behavior_signal"] = context.behavior_signal.as_dict()
    return json.dumps(body, sort_keys=True, default=str)
