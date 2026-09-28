"""LangChain tool-call integration over existing agent_control.

The action source is a LangChain ToolCall dispatched through StructuredTool.invoke.
Tests do not call AgentControlOrchestrator.handle or the executor directly.
No LLM. No network.
"""

from __future__ import annotations

from examples.langchain_agent_control import (
    DEFAULT_FILENAME,
    DEFAULT_POLICY_ID,
    TOOL_NAME,
    make_langchain_control_stack,
)
from agent_control.dcl import COMMIT, NO_COMMIT


FORBIDDEN_PHRASE_DEFAULT = "jailbreak"


def test_langchain_tool_call_commit_writes_and_no_commit_does_not(tmp_path):
    """COMMIT writes record.json; NO_COMMIT never calls create_record so the file is absent."""
    commit_dir = tmp_path / "commit"
    no_commit_dir = tmp_path / "no_commit"
    commit_dir.mkdir()
    no_commit_dir.mkdir()

    commit_file = commit_dir / DEFAULT_FILENAME
    no_commit_file = no_commit_dir / DEFAULT_FILENAME
    assert not commit_file.exists()
    assert not no_commit_file.exists()

    commit_args = {
        "amount": 25,
        "chain": "base",
        "filename": DEFAULT_FILENAME,
    }
    no_commit_args = {
        "amount": 25,
        "chain": "base",
        "filename": DEFAULT_FILENAME,
        "note": FORBIDDEN_PHRASE_DEFAULT,
    }

    commit_agent, commit_controlled, commit_writer, _commit_executor = (
        make_langchain_control_stack(
            output_dir=commit_dir,
            trace_id="trace-langchain-test-commit",
        )
    )
    no_commit_agent, no_commit_controlled, no_commit_writer, _no_commit_executor = (
        make_langchain_control_stack(
            output_dir=no_commit_dir,
            trace_id="trace-langchain-test-no-commit",
        )
    )

    commit_tool_message = commit_agent.invoke_tool(
        commit_args, call_id="tool-call-commit"
    )
    no_commit_tool_message = no_commit_agent.invoke_tool(
        no_commit_args, call_id="tool-call-no-commit"
    )

    assert getattr(commit_tool_message, "tool_call_id", None) == "tool-call-commit"
    assert getattr(no_commit_tool_message, "tool_call_id", None) == "tool-call-no-commit"
    assert commit_agent.tool.name == TOOL_NAME
    assert no_commit_agent.tool.name == TOOL_NAME

    commit_result = commit_controlled.last_result
    assert commit_result is not None
    commit_eval = commit_result.dcl_evaluation
    assert commit_eval is not None
    assert commit_eval.verdict == COMMIT
    assert commit_writer.call_count == 1
    assert commit_file.exists()
    assert commit_file.is_file()
    commit_event = commit_result.audit_event
    assert commit_event is not None
    assert commit_event["verdict"] == commit_eval.verdict == COMMIT
    assert commit_event["policy_id"] == DEFAULT_POLICY_ID == "default"
    assert commit_event["policy_version"] == "1.0.0"

    no_commit_result = no_commit_controlled.last_result
    assert no_commit_result is not None
    no_commit_eval = no_commit_result.dcl_evaluation
    assert no_commit_eval is not None
    assert no_commit_eval.verdict == NO_COMMIT
    assert no_commit_writer.call_count == 0
    assert not no_commit_file.exists()
    no_commit_event = no_commit_result.audit_event
    assert no_commit_event is not None
    assert no_commit_event["verdict"] == no_commit_eval.verdict == NO_COMMIT
    assert no_commit_event["policy_id"] == DEFAULT_POLICY_ID == "default"
    assert no_commit_event["policy_version"] == "1.0.0"
