"""Shared control-flow context. `trace_id` is required and must propagate."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from agent_control.behavior import BehaviorSignal


@dataclass(frozen=True)
class ControlContext:
    """Contextual facts for one proposal. `agent_id` is not implicitly verified."""

    trace_id: str
    agent_id: str
    policy_id: str = "default"
    policy_version: str | None = None
    behavior_signal: BehaviorSignal | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.trace_id:
            raise ValueError("trace_id is required")
        if not self.agent_id:
            raise ValueError("agent_id is required")
        object.__setattr__(self, "extra", dict(self.extra))

    def with_behavior_signal(self, signal: BehaviorSignal | None) -> ControlContext:
        return replace(self, behavior_signal=signal)
