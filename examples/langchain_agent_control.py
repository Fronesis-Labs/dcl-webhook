"""LangChain tool-call proof on top of existing Agent Control (v0.1).

Chain (no second policy engine; agent_control/ is imported, not edited):

    LangChain agent
    → tool call create_record(...)
    → ControlledTool (no policy logic, no filesystem write)
    → ProposedAction
    → existing AgentControlOrchestrator
    → Local Hard Policy
    → EvaluatePolicyDCLGuard
    → EvaluatePolicyOracle
    → audit_logic.evaluate_policy
    → COMMIT: orchestrator calls the executor, which runs create_record
    → NO_COMMIT: create_record is never invoked, so the file cannot appear
    → canonical audit event from the existing orchestrator

Run (from repo root, no LLM, no network):

    python examples/langchain_agent_control.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from langchain_core.messages import ToolCall
from langchain_core.tools import StructuredTool

from agent_control import (
    Agent,
    AgentControlOrchestrator,
    DCLAvailabilityPolicy,
    EvaluatePolicyDCLGuard,
    EvaluatePolicyOracle,
    LocalHardPolicy,
    LocalHardPolicyConfig,
    ProposedAction,
    create_audit_event,
)
from agent_control.executor import ExecutionResult


TOOL_NAME = "create_record"
DEFAULT_FILENAME = "record.json"
DEFAULT_POLICY_ID = "default"
AGENT_ID = "langchain-agent-control-1"
MAX_AMOUNT = 100.0


def create_record(
    directory: str | Path,
    filename: str = DEFAULT_FILENAME,
    **content: Any,
) -> Path:
    """Real filesystem write of record.json (or ``filename``) under ``directory``.

    This is the only function that creates the file. It must be called from the
    executor after the existing orchestrator has already decided to execute.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(
        json.dumps(content, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return path


class CountingCreateRecord:
    """Counts real ``create_record`` invocations. Contains no policy rules."""

    def __init__(self) -> None:
        self.call_count = 0
        self.written_paths: list[Path] = []

    def __call__(
        self,
        directory: str | Path,
        filename: str = DEFAULT_FILENAME,
        **content: Any,
    ) -> Path:
        self.call_count += 1
        path = create_record(directory, filename=filename, **content)
        self.written_paths.append(path)
        return path


class CreateRecordExecutor:
    """Privileged executor: runs ``create_record`` because the orchestrator called it.

    Does not inspect DCL verdicts or apply policy. A write means execute() ran.
    """

    def __init__(self, directory: str | Path, writer: CountingCreateRecord) -> None:
        self.directory = Path(directory)
        self.writer = writer
        self.calls: list = []

    def execute(self, action: ProposedAction, context) -> ExecutionResult:
        filename = str(action.get("filename") or DEFAULT_FILENAME)
        fields = {k: v for k, v in dict(action.payload).items() if k != "filename"}
        path = self.writer(self.directory, filename=filename, **fields)
        self.calls.append((action, context, path))
        return ExecutionResult(
            executed=True,
            detail=f"wrote {path}",
            payload={"path": str(path), "action_type": action.action_type},
        )


class ControlledTool:
    """LangChain tool entrypoint: tool_name + tool_args → ProposedAction → orchestrator.

    No policy rules. Does not write files. The writer runs only if the existing
    orchestrator calls the executor after a DCL COMMIT.
    """

    def __init__(
        self,
        orchestrator: AgentControlOrchestrator,
        *,
        agent_id: str = AGENT_ID,
        policy_id: str = DEFAULT_POLICY_ID,
        trace_id: str,
    ) -> None:
        self.orchestrator = orchestrator
        self.agent_id = agent_id
        self.policy_id = policy_id
        self.trace_id = trace_id
        self.last_result = None

    def run(self, tool_name: str, tool_args: dict[str, Any]):
        action = ProposedAction(action_type=tool_name, payload=dict(tool_args))
        proposal = Agent(self.agent_id).propose(
            action,
            trace_id=self.trace_id,
            policy_id=self.policy_id,
        )
        result = self.orchestrator.handle(proposal)
        self.last_result = result
        return result

    def _langchain_entry(
        self,
        amount: float,
        chain: str,
        filename: str = DEFAULT_FILENAME,
        note: str | None = None,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {
            "amount": amount,
            "chain": chain,
            "filename": filename,
        }
        if note is not None:
            args["note"] = note
        result = self.run(TOOL_NAME, args)
        dcl = result.dcl_evaluation
        event = result.audit_event
        return {
            "verdict": None if dcl is None else dcl.verdict,
            "confidence": None if dcl is None else dcl.confidence,
            "reason": None if dcl is None else dcl.reason,
            "executed": result.executed,
            "event_id": None if event is None else event.get("event_id"),
        }

    def as_langchain_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            func=self._langchain_entry,
            name=TOOL_NAME,
            description=(
                "Create a JSON record. The call is gated by Fronesis agent control; "
                "the file is written only after DCL COMMIT."
            ),
        )


class DeterministicLangChainAgent:
    """LangChain agent stand-in: emits a ToolCall and dispatches it. No LLM, no network."""

    def __init__(self, tool: StructuredTool) -> None:
        self.tool = tool

    def invoke_tool(self, args: dict[str, Any], *, call_id: str) -> Any:
        tool_call: ToolCall = {
            "name": self.tool.name,
            "args": dict(args),
            "id": call_id,
            "type": "tool_call",
        }
        return self.tool.invoke(tool_call)


def build_local_hard_policy() -> LocalHardPolicy:
    """ALLOW create_record on chain=base with amount 25 (wired outside ControlledTool)."""
    return LocalHardPolicy(
        LocalHardPolicyConfig(
            max_amount=MAX_AMOUNT,
            allowed_chains=frozenset({"base"}),
            allowed_action_types=frozenset({TOOL_NAME}),
            require_dcl=True,
        )
    )


def make_langchain_control_stack(
    *,
    output_dir: str | Path,
    trace_id: str,
    agent_id: str = AGENT_ID,
    policy_id: str = DEFAULT_POLICY_ID,
    local_policy: LocalHardPolicy | None = None,
) -> tuple[
    DeterministicLangChainAgent,
    ControlledTool,
    CountingCreateRecord,
    CreateRecordExecutor,
]:
    writer = CountingCreateRecord()
    executor = CreateRecordExecutor(output_dir, writer=writer)
    orchestrator = AgentControlOrchestrator(
        local_policy=local_policy or build_local_hard_policy(),
        dcl=EvaluatePolicyDCLGuard(oracle=EvaluatePolicyOracle()),
        executor=executor,
        availability_policy=DCLAvailabilityPolicy.FAIL_CLOSED,
        audit_event_builder=create_audit_event,
    )
    controlled = ControlledTool(
        orchestrator,
        agent_id=agent_id,
        policy_id=policy_id,
        trace_id=trace_id,
    )
    agent = DeterministicLangChainAgent(controlled.as_langchain_tool())
    return agent, controlled, writer, executor


def _fmt_bool(value: bool) -> str:
    return "true" if value else "false"


def _format_agent_action(args: dict[str, Any]) -> str:
    shown = {k: v for k, v in args.items() if k != "filename"}
    inner = ", ".join(f"{k}={v}" for k, v in shown.items())
    return f"{TOOL_NAME}({inner})"


def _rel_record_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _print_trace(
    *,
    args: dict[str, Any],
    controlled: ControlledTool,
    writer: CountingCreateRecord,
) -> None:
    result = controlled.last_result
    dcl = result.dcl_evaluation
    event = result.audit_event
    tool_called = writer.call_count > 0
    if tool_called and writer.written_paths:
        record_line = _rel_record_path(writer.written_paths[-1])
    else:
        record_line = "false"

    print("AGENT ACTION")
    print(_format_agent_action(args))
    print()
    print("DCL")
    if dcl is None:
        print("unavailable")
        print("")
    else:
        print(f"{dcl.verdict} / {dcl.confidence}")
        print(dcl.reason)
    print()
    print("EXECUTION")
    print(f"tool_called: {_fmt_bool(tool_called)}")
    print(f"record created: {record_line}")
    print()
    print("AUDIT")
    print(f"event_id: {event['event_id'] if event is not None else None}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass

    demo_root = Path("demo_output")
    commit_dir = demo_root / "commit"
    no_commit_dir = demo_root / "no_commit"
    commit_dir.mkdir(parents=True, exist_ok=True)
    no_commit_dir.mkdir(parents=True, exist_ok=True)

    commit_args = {"amount": 25, "chain": "base", "filename": DEFAULT_FILENAME}
    no_commit_args = {
        "amount": 25,
        "chain": "base",
        "filename": DEFAULT_FILENAME,
        "note": "jailbreak",
    }

    commit_agent, commit_controlled, commit_writer, _commit_exec = (
        make_langchain_control_stack(
            output_dir=commit_dir,
            trace_id="trace-langchain-commit",
        )
    )
    commit_agent.invoke_tool(commit_args, call_id="tool-call-commit")

    no_commit_agent, no_commit_controlled, no_commit_writer, _no_commit_exec = (
        make_langchain_control_stack(
            output_dir=no_commit_dir,
            trace_id="trace-langchain-no-commit",
        )
    )
    no_commit_agent.invoke_tool(no_commit_args, call_id="tool-call-no-commit")

    print("=== FRONESIS AGENT CONTROL ===")
    print()
    _print_trace(args=commit_args, controlled=commit_controlled, writer=commit_writer)
    print()
    print("--------------------------------")
    print()
    _print_trace(
        args=no_commit_args,
        controlled=no_commit_controlled,
        writer=no_commit_writer,
    )
    print()
    print("RESULT")
    print("COMMIT  → action executed")
    print("NO_COMMIT → action blocked")


if __name__ == "__main__":
    main()
