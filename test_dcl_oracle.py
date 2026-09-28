"""EvaluatePolicyOracle boundary: transparent bridge to audit_logic.evaluate_policy.

Does not change the oracle signature, policy semantics, or production servers.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from agent_control import (
    ControlContext,
    EvaluatePolicyDCLGuard,
    EvaluatePolicyOracle,
    ProposedAction,
)
from audit_logic import BUILTIN_POLICIES, evaluate_policy


REPO_ROOT = Path(__file__).resolve().parent
AGENT_CONTROL = REPO_ROOT / "agent_control"
# Same builtin YAML and forbidden phrase as test_agent_control_e2e (read-only).
DEFAULT_POLICY_YAML = BUILTIN_POLICIES["default"]
JAILBREAK_PHRASE = "jailbreak"
COMMIT_RESPONSE = "ordinary transfer of 25 USDC on base"


def test_evaluate_policy_oracle_passthrough_without_rewrite(monkeypatch):
    """Guard passes (response, policy_yaml); adapter forwards those objects unchanged."""
    params = list(inspect.signature(EvaluatePolicyOracle.evaluate).parameters)
    assert params == ["self", "response", "policy_yaml"]

    engine_calls = []
    known = ("NO_COMMIT", 0.123, "recorded-stub", "stub-9.9.9")

    def fake_evaluate_policy(response, policy_yaml):
        engine_calls.append((response, policy_yaml))
        return known

    monkeypatch.setattr("agent_control.dcl_oracle.evaluate_policy", fake_evaluate_policy)

    class RecordingOracle:
        def __init__(self) -> None:
            self.inner = EvaluatePolicyOracle()
            self.calls: list = []
            self.results: list = []

        def evaluate(self, response: str, policy_yaml: str):
            self.calls.append((response, policy_yaml))
            result = self.inner.evaluate(response, policy_yaml)
            self.results.append(result)
            return result

    oracle = RecordingOracle()
    guard = EvaluatePolicyDCLGuard(policy_yaml=DEFAULT_POLICY_YAML, oracle=oracle)
    action = ProposedAction(
        action_type="transfer",
        payload={"asset": "USDC", "amount": 25.0, "chain": "base"},
    )
    context = ControlContext(
        trace_id="trace-oracle-boundary",
        agent_id="agent-oracle-boundary",
        policy_id="default",
    )

    guard.evaluate(action, context)

    assert len(oracle.calls) == 1
    response, policy_yaml = oracle.calls[0]
    assert policy_yaml == DEFAULT_POLICY_YAML
    assert engine_calls == [(response, policy_yaml)]
    assert engine_calls[0][0] is response
    assert engine_calls[0][1] is policy_yaml
    assert oracle.results[0] == known
    assert len(oracle.results[0]) == 4
    verdict, confidence, reason, policy_version = oracle.results[0]
    assert verdict == known[0]
    assert confidence == known[1]
    assert reason == known[2]
    assert policy_version == known[3]


def test_evaluate_policy_oracle_matches_engine_commit_and_no_commit():
    oracle = EvaluatePolicyOracle()

    direct_commit = evaluate_policy(COMMIT_RESPONSE, DEFAULT_POLICY_YAML)
    adapted_commit = oracle.evaluate(COMMIT_RESPONSE, DEFAULT_POLICY_YAML)
    assert adapted_commit == direct_commit
    assert len(adapted_commit) == 4
    assert adapted_commit[0] == "COMMIT"
    assert adapted_commit[0] == direct_commit[0]
    assert adapted_commit[1] == direct_commit[1]
    assert adapted_commit[2] == direct_commit[2]
    assert adapted_commit[3] == direct_commit[3]

    direct_no = evaluate_policy(JAILBREAK_PHRASE, DEFAULT_POLICY_YAML)
    adapted_no = oracle.evaluate(JAILBREAK_PHRASE, DEFAULT_POLICY_YAML)
    assert adapted_no == direct_no
    assert len(adapted_no) == 4
    assert adapted_no[0] == "NO_COMMIT"
    assert adapted_no[0] == direct_no[0]
    assert adapted_no[1] == direct_no[1]
    assert adapted_no[2] == direct_no[2]
    assert adapted_no[3] == direct_no[3]


def test_evaluate_policy_dcl_guard_calls_injected_oracle_not_engine():
    class FakeOracle:
        def __init__(self) -> None:
            self.calls: list = []

        def evaluate(self, response: str, policy_yaml: str):
            self.calls.append((response, policy_yaml))
            return "COMMIT", 0.99, "injected", "injected-1"

    oracle = FakeOracle()
    guard = EvaluatePolicyDCLGuard(policy_yaml=DEFAULT_POLICY_YAML, oracle=oracle)
    result = guard.evaluate(
        ProposedAction(action_type="transfer", payload={"amount": 1}),
        ControlContext(trace_id="t", agent_id="a", policy_id="default"),
    )
    assert len(oracle.calls) == 1
    assert result.verdict == "COMMIT"
    assert result.reason == "injected"

    imported_by = []
    called_by = []
    for path in sorted(AGENT_CONTROL.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "audit_logic":
                for alias in node.names:
                    if alias.name == "evaluate_policy":
                        imported_by.append(path.name)
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "evaluate_policy":
                    called_by.append(path.name)
                elif isinstance(func, ast.Attribute) and func.attr == "evaluate_policy":
                    called_by.append(path.name)

    assert imported_by == ["dcl_oracle.py"]
    assert called_by == ["dcl_oracle.py"]
    assert "dcl.py" not in imported_by
    assert "dcl.py" not in called_by
