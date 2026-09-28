"""Layer 1 — Agent proposal.

The agent may PROPOSE an action. It must not execute one.
`action_type` plus a flexible payload keeps this usable for transfers today
and MCP/API/tool calls later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from agent_control.context import ControlContext


@dataclass(frozen=True)
class ProposedAction:
    """Generic proposed action. Example transfer payload:

    {"asset": "USDC", "amount": 25, "chain": "base", "destination": "..."}
    """

    action_type: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", dict(self.payload))

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)


@dataclass(frozen=True)
class AgentProposal:
    """An action plus the control-plane context that must travel with it."""

    action: ProposedAction
    context: ControlContext


class Agent:
    """Minimal agent: it can propose, it cannot execute."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id

    def propose(
        self,
        action: ProposedAction,
        *,
        trace_id: str,
        policy_id: str = "default",
        extra: Mapping[str, Any] | None = None,
    ) -> AgentProposal:
        return AgentProposal(
            action=action,
            context=ControlContext(
                trace_id=trace_id,
                agent_id=self.agent_id,
                policy_id=policy_id,
                extra=dict(extra or {}),
            ),
        )
