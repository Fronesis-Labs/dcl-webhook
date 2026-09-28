"""Layer 3 — Local Hard Policy.

Deterministic constraints that run with no network call. This is the
emergency/safety boundary: BLOCK means DCL is not called, the action is not
executed, and no Canonical DCL Audit Event is fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from agent_control.actions import ProposedAction
from agent_control.context import ControlContext


class PolicyVerdict(str, Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"


@dataclass(frozen=True)
class LocalPolicyDecision:
    verdict: PolicyVerdict
    reason: str
    rule: str | None = None

    @property
    def allowed(self) -> bool:
        return self.verdict is PolicyVerdict.ALLOW


@dataclass(frozen=True)
class LocalBlockRecord:
    """How a local hard-policy block is represented.

    This is NOT a Canonical DCL Audit Event v1.0 and MUST NOT be passed
    through create_audit_event. DCL was not called, so no
    `dcl.audit.evaluated` event exists for this proposal.
    """

    record_type: str = "local.policy.blocked"
    trace_id: str = ""
    agent_id: str = ""
    action_type: str = ""
    reason: str = ""
    rule: str | None = None
    policy_verdict: str = PolicyVerdict.BLOCK.value

    def as_dict(self) -> dict[str, Any]:
        return {
            "record_type": self.record_type,
            "trace_id": self.trace_id,
            "agent_id": self.agent_id,
            "action_type": self.action_type,
            "reason": self.reason,
            "rule": self.rule,
            "policy_verdict": self.policy_verdict,
        }


class InMemorySpendLedger:
    """Optional daily-budget accumulator. Keys are caller-defined (usually agent_id)."""

    def __init__(self) -> None:
        self._spent: dict[str, float] = {}

    def spent(self, key: str) -> float:
        return self._spent.get(key, 0.0)

    def add(self, key: str, amount: float) -> None:
        self._spent[key] = self.spent(key) + amount


@dataclass(frozen=True)
class LocalHardPolicyConfig:
    max_amount: float | None = None
    daily_budget: float | None = None
    allowed_chains: frozenset[str] | None = None
    allowed_action_types: frozenset[str] | None = None
    allowed_destinations: frozenset[str] | None = None
    require_dcl: bool = True
    max_advisory_risk: float | None = None
    amount_field: str = "amount"
    chain_field: str = "chain"
    destination_field: str = "destination"


class LocalHardPolicy:
    """Executable without an external network call."""

    def __init__(
        self,
        config: LocalHardPolicyConfig | None = None,
        *,
        spend_ledger: InMemorySpendLedger | None = None,
        spend_key: str | None = None,
    ) -> None:
        self.config = config or LocalHardPolicyConfig()
        self.spend_ledger = spend_ledger
        self.spend_key = spend_key

    def evaluate(self, action: ProposedAction, context: ControlContext) -> LocalPolicyDecision:
        cfg = self.config

        if cfg.allowed_action_types is not None and action.action_type not in cfg.allowed_action_types:
            return self._block(
                f"action_type {action.action_type!r} is not in the allowlist",
                rule="allowed_action_types",
            )

        chain = action.get(cfg.chain_field)
        if cfg.allowed_chains is not None:
            if chain is None:
                return self._block("chain is required by the allowlist and was missing", rule="allowed_chains")
            if str(chain) not in cfg.allowed_chains:
                return self._block(f"chain {chain!r} is not in the allowlist", rule="allowed_chains")

        destination = action.get(cfg.destination_field)
        if cfg.allowed_destinations is not None:
            if destination is None:
                return self._block(
                    "destination is required by the allowlist and was missing",
                    rule="allowed_destinations",
                )
            if str(destination) not in cfg.allowed_destinations:
                return self._block(
                    f"destination {destination!r} is not in the allowlist",
                    rule="allowed_destinations",
                )

        amount = self._parse_amount(action.get(cfg.amount_field))
        if cfg.max_amount is not None:
            if amount is None:
                return self._block(
                    "amount is required by max_amount and was missing or invalid",
                    rule="max_amount",
                )
            if amount > cfg.max_amount:
                return self._block(
                    f"amount {amount} exceeds max_amount {cfg.max_amount}",
                    rule="max_amount",
                )

        if cfg.daily_budget is not None:
            if amount is None:
                return self._block(
                    "amount is required by daily_budget and was missing or invalid",
                    rule="daily_budget",
                )
            key = self.spend_key or context.agent_id
            spent = self.spend_ledger.spent(key) if self.spend_ledger is not None else 0.0
            if spent + amount > cfg.daily_budget:
                return self._block(
                    f"amount {amount} plus spent {spent} exceeds daily_budget {cfg.daily_budget}",
                    rule="daily_budget",
                )

        # Advisory risk may tighten the local bound; it cannot produce ALLOW by itself.
        if cfg.max_advisory_risk is not None and context.behavior_signal is not None:
            score = context.behavior_signal.risk_score
            if score is not None and score > cfg.max_advisory_risk:
                return self._block(
                    f"advisory risk_score {score} exceeds max_advisory_risk {cfg.max_advisory_risk}",
                    rule="max_advisory_risk",
                )

        if cfg.require_dcl:
            # Recorded as a constraint the orchestrator already enforces: ALLOW never skips DCL.
            extra_note = "require_dcl"
        else:
            extra_note = "dcl_optional_not_honored_in_v0.1"

        return LocalPolicyDecision(
            verdict=PolicyVerdict.ALLOW,
            reason="local hard policy checks passed",
            rule=extra_note,
        )

    def _parse_amount(self, raw: Any) -> float | None:
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    def _block(self, reason: str, rule: str) -> LocalPolicyDecision:
        return LocalPolicyDecision(verdict=PolicyVerdict.BLOCK, reason=reason, rule=rule)

    def build_block_record(
        self,
        action: ProposedAction,
        context: ControlContext,
        decision: LocalPolicyDecision,
    ) -> LocalBlockRecord:
        return LocalBlockRecord(
            trace_id=context.trace_id,
            agent_id=context.agent_id,
            action_type=action.action_type,
            reason=decision.reason,
            rule=decision.rule,
        )
