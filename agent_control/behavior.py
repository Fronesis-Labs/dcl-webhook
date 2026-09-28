"""Layer 2 — Behavioral Security Signal.

Advisory only. A signal must never authorize execution, skip local policy,
or skip DCL. This module is an interface plus a deterministic mock — not an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class BehaviorSignal:
    risk_score: float | None = None
    reason: str | None = None
    source: str | None = None

    def as_dict(self) -> dict[str, float | str | None]:
        return {
            "risk_score": self.risk_score,
            "reason": self.reason,
            "source": self.source,
        }


@runtime_checkable
class BehaviorSignalProvider(Protocol):
    def collect(self, action: object, context: object) -> BehaviorSignal | None:
        """Return an advisory signal, or None if no observation is available."""


class StaticBehaviorSignalProvider:
    """Deterministic mock for tests. Always returns the configured signal."""

    def __init__(self, signal: BehaviorSignal | None) -> None:
        self.signal = signal
        self.calls: list[tuple[object, object]] = []

    def collect(self, action: object, context: object) -> BehaviorSignal | None:
        self.calls.append((action, context))
        return self.signal
