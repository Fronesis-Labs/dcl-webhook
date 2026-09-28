"""Thin adapter around the existing DCL Oracle: audit_logic.evaluate_policy.

Matches the production interface used by webhook_server / mcp_server /
bazaar_server:

    evaluate_policy(response, policy_yaml)
        -> (verdict, confidence, reason, policy_version)

This module only calls that function. It does not copy policy logic, does
not append to ChainState, does not take payment, and does not call
production HTTP/MCP servers.
"""

from __future__ import annotations

from typing import Tuple

from audit_logic import evaluate_policy

EvaluatePolicyResult = Tuple[str, float, str, str]


class EvaluatePolicyOracle:
    """Existing in-process DCL Oracle. Does not invent a verdict."""

    def evaluate(self, response: str, policy_yaml: str) -> EvaluatePolicyResult:
        return evaluate_policy(response, policy_yaml)
