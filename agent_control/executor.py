"""Layer 5 — Action execution.

The executor runs only after local policy ALLOW and DCL COMMIT.
The reference implementation is a mock: no chain transactions, no payments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agent_control.actions import ProposedAction
from agent_control.context import ControlContext


@dataclass(frozen=True)
class ExecutionResult:
    executed: bool
    detail: str
    payload: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ActionExecutor(Protocol):
    def execute(self, action: ProposedAction, context: ControlContext) -> ExecutionResult:
        """Run the proposed action. Callers must already have ALLOW + COMMIT."""


class MockActionExecutor:
    """Records calls; never talks to wallets, payment rails, or production APIs."""

    def __init__(self) -> None:
        self.calls: list[tuple[ProposedAction, ControlContext]] = []

    def execute(self, action: ProposedAction, context: ControlContext) -> ExecutionResult:
        self.calls.append((action, context))
        return ExecutionResult(
            executed=True,
            detail="mock executor ran; no production side effects",
            payload={"action_type": action.action_type, "trace_id": context.trace_id},
        )
