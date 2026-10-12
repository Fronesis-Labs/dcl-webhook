"""Run the E2 corpus through the existing Agent Control pipeline.

Each case is a fixed tool call. The LangChain proof stack maps that call to a
ProposedAction and hands it to AgentControlOrchestrator. DCL is
EvaluatePolicyDCLGuard over EvaluatePolicyOracle (audit_logic.evaluate_policy)
with builtin policy ``default``. The executor is CreateRecordExecutor: a local
JSON file, and only when the orchestrator calls execute after COMMIT.

This module does not call Nemotron, Pliny, or the production HTTP/MCP/Bazaar
servers, and it does not change policy rules.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent_control import AgentControlOrchestrator  # noqa: E402
from benchmarks.e2.report import (  # noqa: E402
    build_report,
    is_valid_canonical_audit_event,
    write_report,
)

_EXAMPLE_MODULE = None
_CANARY_TOOL = "create_record"
_AGENT_ID = "e2-benchmark-agent"
_HERE = Path(__file__).resolve().parent
DEFAULT_CASES = _HERE / "cases.yaml"
DEFAULT_RESULTS = _HERE / "results.json"


def _example_module():
    global _EXAMPLE_MODULE
    if _EXAMPLE_MODULE is not None:
        return _EXAMPLE_MODULE
    path = _REPO_ROOT / "examples" / "langchain_agent_control.py"
    spec = importlib.util.spec_from_file_location("e2_langchain_agent_control", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load CreateRecordExecutor from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _EXAMPLE_MODULE = module
    return module


def load_cases(path: Path | None = None) -> list[dict[str, Any]]:
    cases_path = Path(path) if path is not None else DEFAULT_CASES
    document = yaml.safe_load(cases_path.read_text(encoding="utf-8"))
    cases = document.get("cases") if isinstance(document, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{cases_path} must contain a non-empty cases list")

    seen: set[str] = set()
    for case in cases:
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("every case needs a case_id")
        if case_id in seen:
            raise ValueError(f"duplicate case_id {case_id}")
        seen.add(case_id)
        if case.get("group") not in ("adversarial", "benign"):
            raise ValueError(f"{case_id} group must be adversarial or benign")
        for key in ("category", "description"):
            if not isinstance(case.get(key), str) or not case[key]:
                raise ValueError(f"{case_id} missing {key}")
        tool_call = case.get("tool_call")
        if not isinstance(tool_call, dict):
            raise ValueError(f"{case_id} missing tool_call")
        if tool_call.get("name") != _CANARY_TOOL:
            raise ValueError(
                f"{case_id} tool must be {_CANARY_TOOL}; "
                "the benchmark only submits the existing canary"
            )
        args = tool_call.get("args")
        if not isinstance(args, dict):
            raise ValueError(f"{case_id} tool_call.args must be a mapping")
        for key in ("amount", "chain", "filename", "note"):
            if key not in args:
                raise ValueError(f"{case_id} tool args missing {key}")
        expected = case.get("expected_verdict", None)
        if expected not in (None, "COMMIT", "NO_COMMIT"):
            raise ValueError(f"{case_id} expected_verdict must be COMMIT, NO_COMMIT, or omitted")
    return cases


def _tool_args(case: dict[str, Any]) -> dict[str, Any]:
    raw = dict(case["tool_call"]["args"])
    return {
        "amount": float(raw["amount"]),
        "chain": str(raw["chain"]),
        "filename": str(raw["filename"]),
        "note": str(raw["note"]),
    }


def run_case(case: dict[str, Any], work_dir: Path) -> dict[str, Any]:
    """One case, one orchestrator, one canary directory."""
    example = _example_module()
    case_id = case["case_id"]
    policy_id = str(case.get("policy_id") or "default")
    trace_id = f"e2-{case_id}"
    output_dir = work_dir / case_id
    output_dir.mkdir(parents=True, exist_ok=True)
    args = _tool_args(case)

    _agent, controlled, writer, executor = example.make_langchain_control_stack(
        output_dir=output_dir,
        trace_id=trace_id,
        agent_id=_AGENT_ID,
        policy_id=policy_id,
        local_policy=example.build_local_hard_policy(),
    )
    if not isinstance(controlled.orchestrator, AgentControlOrchestrator):
        raise TypeError("canary stack did not build AgentControlOrchestrator")

    _agent.invoke_tool(args, call_id=f"tool-call-{case_id}")
    result = controlled.last_result
    if result is None:
        raise RuntimeError(f"{case_id} produced no control-flow result")

    expected_path = output_dir / args["filename"]
    written = [Path(path) for path in writer.written_paths]
    executor_called = len(executor.calls) > 0
    side_effect_present = expected_path.exists() or any(path.exists() for path in written)

    dcl = result.dcl_evaluation
    event = result.audit_event
    actual_verdict = None if dcl is None else dcl.verdict
    confidence = None if dcl is None else dcl.confidence
    if dcl is not None:
        reason = dcl.reason
    elif result.local_block is not None:
        reason = result.local_block.reason
    else:
        reason = result.outcome

    canonical_ok = False
    if actual_verdict in ("COMMIT", "NO_COMMIT"):
        canonical_ok = is_valid_canonical_audit_event(
            event,
            trace_id=trace_id,
            verdict=actual_verdict,
        )

    proposed_action = {
        "action_type": case["tool_call"]["name"],
        "payload": dict(args),
    }
    expected = case.get("expected_verdict", None)

    return {
        "case_id": case_id,
        "group": case["group"],
        "category": case["category"],
        "description": case["description"],
        "proposed_action": proposed_action,
        "expected_verdict": expected,
        "expected_verdict_basis": case.get("expected_verdict_basis"),
        "actual_verdict": actual_verdict,
        "confidence": confidence,
        "reason": reason,
        "executor_called": executor_called,
        "side_effect_present": side_effect_present,
        "audit_event_id": None if event is None else event.get("event_id"),
        "outcome": result.outcome,
        "local_policy_verdict": result.local_policy.verdict.value,
        "trace_id": trace_id,
        "policy_id": None if dcl is None else dcl.policy_id,
        "policy_version": None if dcl is None else dcl.policy_version,
        "canonical_audit_valid": canonical_ok,
        "audit_event": event,
        "canary_write_count": writer.call_count,
    }


def run_benchmark(
    *,
    cases_path: Path | None = None,
    results_path: Path | None = None,
    work_dir: Path,
) -> dict[str, Any]:
    cases = load_cases(cases_path)
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    rows = [run_case(case, work) for case in cases]
    report = build_report(rows)
    if results_path is not None:
        write_report(report, Path(results_path))
    return report


def main() -> None:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="e2-canary-") as directory:
        report = run_benchmark(
            cases_path=DEFAULT_CASES,
            results_path=DEFAULT_RESULTS,
            work_dir=Path(directory),
        )
    metrics = report["metrics"]
    print(f"wrote {DEFAULT_RESULTS}")
    print(
        "cases={total_cases} adversarial={adversarial_cases} benign={benign_cases} "
        "no_commit={no_commit_count} commit={commit_count}".format(**metrics)
    )
    print(
        "der={der} ser={ser} ebr={ebr} ac={audit_coverage} bpr={benign_pass_rate}".format(
            **metrics
        )
    )
    print(
        "policy_detection_failure_count={policy_detection_failure_count} "
        "enforcement_failure_count={enforcement_failure_count} "
        "executor_bypass_count={executor_bypass_count} "
        "side_effect_after_no_commit_count={side_effect_after_no_commit_count}".format(
            **metrics
        )
    )


if __name__ == "__main__":
    main()
