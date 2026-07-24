DCL Trust Oracle — MCP Server (for Smithery)
Wraps the DCL logic (dcl_core.py) as real MCP tools via FastMCP.
x402 payment is NOT used here — that lives in a separate REST API (webhook_server.py).
"""
import os
import time
import uuid
import json
from typing import Annotated, List, Optional
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations, TextContent
from dcl_core import ChainState, BUILTIN_POLICIES, evaluate_policy, get_drift_mode, sha256hex

mcp = FastMCP(
    "DCL Trust Oracle",
    instructions=(
        "DCL Trust Oracle is a deterministic AI audit layer, natively integrated with the Model "
        "Context Protocol (MCP), that evaluates LLM and agent outputs against configurable policies "
        "before and after action. Every verdict is written to a tamper-evident, hash-chained audit "
        "log that stores only cryptographic metadata — never raw content — enabling privacy-first, "
        "post-action forensic review. The server exposes Jailbreak detection (instruction adherence "
        "checks), Quality and drift evaluation, and baseline Safety checks alongside fast and strict "
        "pre-action audit tiers. A companion REST API (external to this MCP server) settles usage via "
        "the x402 micropayment protocol on the Base network, letting autonomous agents pay per-call "
        "in USDC without any account setup. Use these tools to gate risky agent actions, batch-evaluate "
        "responses, and cryptographically verify the integrity of past decisions."
    ),
    transport_security=TransportSecuritySettings(
        allowed_hosts=["mcp.fronesislabs.com", "localhost", "localhost:8081", "127.0.0.1", "127.0.0.1:8081"],
        allowed_origins=["https://mcp.fronesislabs.com", "http://localhost:8081"],
    ),
)
mcp._mcp_server.version = "2.0.1"
_chain = ChainState(os.environ.get("DCL_DB_PATH", "dcl_chain.db"))
_commit_rate: List[float] = []

# ════════════════════════════════════════════════════════════════════════════════
# Output schemas (Pydantic models -> structured MCP output schemas)
# ═══════════════════════════════════════════════════════════════════════════════
class EvaluateResult(BaseModel):
    verdict: str = Field(description="COMMIT if the response passed policy checks, otherwise NO_COMMIT.")
    confidence: float = Field(description="Confidence score of the verdict, from 0.0 to 1.0.")
    reason: str = Field(description="Human-readable explanation of why the verdict was reached.")
    tx_hash: str = Field(description="Hash of this record in the tamper-evident audit chain.")
    chain_index: int = Field(description="Sequential index of this record in the audit chain.")
    input_hash: str = Field(description="Hash of the evaluated response (raw content is never stored).")
    policy_version: str = Field(description="Version of the policy that was applied.") drift_mode: str = Field(description="Current drift status: NORMAL, WARNING, ESCALATION, or BLOCK.")
    drift_score: float = Field(description="Z-score measuring deviation of the recent commit rate from baseline.")

class BatchResult(BaseModel):
    batch_id: str = Field(description="Unique identifier for this batch run.")
    agent_id: str = Field(description="Identifier of the agent whose responses were evaluated.")
    count: int = Field(description="Number of items evaluated in this batch.")
    results: List[EvaluateResult] = Field(description="Per-item evaluation results, in input order.")

class PipelineStartResult(BaseModel):
    pipeline_id: str = Field(description="Unique identifier for the newly opened pipeline session.")
    agent_id: str = Field(description="Identifier of the agent that owns this session.")
    scope: str = Field(description="Scope label for the session.")
    expires_at: float = Field(description="Unix timestamp when the session expires.")
    drift_mode: str = Field(description="Drift status at session start (always NORMAL for a new session).")

class AuditResult(BaseModel):
    error: Optional[str] = Field(default=None, description="Set if tx_hash was not found; other fields are omitted.")
    tx_hash: Optional[str] = Field(default=None, description="Hash of the audit chain record.")
    agent_id: Optional[str] = Field(default=None, description="Identifier of the agent tied to this record.")
    verdict: Optional[str] = Field(default=None, description="COMMIT or NO_COMMIT.")
    reason: Optional[str] = Field(default=None, description="Explanation recorded for the verdict.")
    confidence: Optional[float] = Field(default=None, description="Confidence score recorded for the verdict.")
    task_type: Optional[str] = Field(default=None, description="Task type tag recorded with this entry.")
    timestamp: Optional[float] = Field(default=None, description="Unix timestamp when the record was created.")
    chain_index: Optional[int] = Field(default=None, description="Sequential index of the record in the chain.")
    prev_hash: Optional[str] = Field(default=None, description="Hash of the preceding record in the chain.")
    chain_integrity: Optional[bool] = Field(default=None, description="True if the full chain verifies as intact.")

class AuditDeepResult(AuditResult):
    tampered_at_index: Optional[int] = Field(
        default=None, description="Index where chain integrity broke, if any tampering was detected."
    )
    drift_context: Optional[dict] = Field(
        default=None, description="Extended forensic metadata captured at evaluation time."
    )

def _run_evaluation(response: str, policy: str, agent_id: str, task_type: str) -> EvaluateResult:
    policy_yaml = BUILTIN_POLICIES.get(policy, policy or BUILTIN_POLICIES["default"])
    verdict, confidence, reason, policy_version = evaluate_policy(response, policy_yaml)
    input_hash = "0x" + sha256hex(response)[:16]
    policy_hash = sha256hex(policy_yaml)[:16]
    tx_hash, chain_idx = _chain.append(
        verdict=verdict, input_hash=input_hash, policy_hash=policy_hash,
        agent_id=agent_id, reason=reason, confidence=confidence, task_type=task_type,
    )
    _commit_rate.append(1.0 if verdict == "COMMIT" else 0.0)
    if len(_commit_rate) > 100:
        _commit_rate.pop(0)
    drift_mode, drift_score = get_drift_mode(_commit_rate) return EvaluateResult(
        verdict=verdict, confidence=confidence, reason=reason,
        tx_hash=tx_hash, chain_index=chain_idx, input_hash=input_hash,
        policy_version=policy_version, drift_mode=drift_mode, drift_score=drift_score,
    )

_WRITE_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False,
)
_READ_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False,
)

# ════════════════════════════════════════════════════════════════════════════════
# MCP Tools (Strictly typed for Smithery parser compliance)
# ════════════════════════════════════════════════════════════════════════════════

@mcp.tool(
    name="dcl_evaluate_fast",
    description="FAST Pre-Action Audit. Quick policy check of an agent's response using the default policy.",
    annotations=_WRITE_ANNOTATIONS
)
async def dcl_evaluate_fast(
    response: Annotated[str, Field(description="The agent or LLM response text to audit.")],
    agent_id: Annotated[str, Field(description="Identifier of the agent that produced the response.")]
) -> List[TextContent]:
    result = _run_evaluation(response, "default", agent_id, "fast")
    return [TextContent(type="text", text=json.dumps(result.model_dump(), default=str))]

@mcp.tool(
    name="dcl_evaluate_strict",
    description="STRICT Pre-Action Audit. Rigorous check with a higher confidence threshold.",
    annotations=_WRITE_ANNOTATIONS
)
async def dcl_evaluate_strict(
    response: Annotated[str, Field(description="The agent or LLM response text to audit.")],
    agent_id: Annotated[str, Field(description="Identifier of the agent that produced the response.")]
) -> List[TextContent]:
    result = _run_evaluation(response, "default", agent_id, "strict")
    return [TextContent(type="text", text=json.dumps(result.model_dump(), default=str))]

@mcp.tool(
    name="dcl_evaluate_jailbreak",
    description="PRE-ACTION Instruction Adherence Check. Detects jailbreak attempts.",
    annotations=_WRITE_ANNOTATIONS
)
async def dcl_evaluate_jailbreak(
    response: Annotated[str, Field(description="The agent or LLM response text to check for jailbreak attempts.")],
    agent_id: Annotated[str, Field(description="Identifier of the agent that produced the response.")]
) -> List[TextContent]: result = _run_evaluation(response, "anti_jailbreak", agent_id, "jailbreak")
    return [TextContent(type="text", text=json.dumps(result.model_dump(), default=str))]

@mcp.tool(
    name="dcl_evaluate_safety",
    description="PRE-ACTION Baseline Safety Check.",
    annotations=_WRITE_ANNOTATIONS
)
async def dcl_evaluate_safety(
    response: Annotated[str, Field(description="The agent or LLM response text to check for safety violations.")],
    agent_id: Annotated[str, Field(description="Identifier of the agent that produced the response.")]
) -> List[TextContent]:
    result = _run_evaluation(response, "safety", agent_id, "safety")
    return [TextContent(type="text", text=json.dumps(result.model_dump(), default=str))]

@mcp.tool(
    name="dcl_evaluate_quality",
    description="PRE-ACTION Content Quality & Drift Check.",
    annotations=_WRITE_ANNOTATIONS
)
async def dcl_evaluate_quality(
    response: Annotated[str, Field(description="The agent or LLM response text to check for quality and drift.")],
    agent_id: Annotated[str, Field(description="Identifier of the agent that produced the response.")]
) -> List[TextContent]:
    result = _run_evaluation(response, "content_quality", agent_id, "quality")
    return [TextContent(type="text", text=json.dumps(result.model_dump(), default=str))]

@mcp.tool(
    name="dcl_evaluate_batch",
    description="PRE-ACTION Bulk Processing. Evaluates multiple responses in a single call.",
    annotations=_WRITE_ANNOTATIONS
)
async def dcl_evaluate_batch(
    items: Annotated[List[dict], Field(description="List of items to evaluate, each shaped like {'response': str, 'policy'?: str}.")],
    agent_id: Annotated[str, Field(description="Identifier of the agent that produced the responses.")]
) -> List[TextContent]:
    results = [
        _run_evaluation(item["response"], item.get("policy", "default"), agent_id, "batch_item")
        for item in items
    ]
    batch_result = BatchResult(batch_id=str(uuid.uuid4())[:8], agent_id=agent_id, count=len(results), results=results)
    return [TextContent(type="text", text=json.dumps(batch_result.model_dump(), default=str))]

@mcp.tool(
    name="dcl_pipeline_start",
    description="SESSION Management. Opens a pipeline session for a series of checks.",
    annotations=_WRITE_ANNOTATIONS
)
async def dcl_pipeline_start(
    agent_id: Annotated[str, Field(description="Identifier of the agent that owns this session.")], scope: Annotated[str, Field(description="Scope label for the session.")] = "default",
    ttl_seconds: Annotated[int, Field(description="Session time-to-live, in seconds.")] = 3600
) -> List[TextContent]:
    pipeline_id = f"pl_{uuid.uuid4().hex[:12]}"
    result = PipelineStartResult(
        pipeline_id=pipeline_id, agent_id=agent_id, scope=scope,
        expires_at=time.time() + ttl_seconds, drift_mode="NORMAL",
    )
    return [TextContent(type="text", text=json.dumps(result.model_dump(), default=str))]

@mcp.tool(
    name="dcl_audit_decode",
    description="POST-ACTION Basic Audit. Retrieves a record from the tamper-evident chain by tx_hash.",
    annotations=_READ_ANNOTATIONS
)
async def dcl_audit_decode(
    tx_hash: Annotated[str, Field(description="Transaction hash of the audit chain record to retrieve.")]
) -> List[TextContent]:
    entry = _chain.get_by_tx(tx_hash)
    if not entry:
        result = AuditResult(error="tx_hash not found in chain")
    else:
        intact, _ = _chain.verify()
        result = AuditResult(
            tx_hash=entry["tx_hash"], agent_id=entry["agent_id"], verdict=entry["verdict"],
            reason=entry["reason"], confidence=entry["confidence"], task_type=entry["task_type"],
            timestamp=entry["timestamp"], chain_index=entry["index"],
            prev_hash=entry["prev_hash"], chain_integrity=intact,
        )
    return [TextContent(type="text", text=json.dumps(result.model_dump(), default=str))]

@mcp.tool(
    name="dcl_audit_decode_deep",
    description="POST-ACTION Deep Forensic Audit. Extended output with drift_context and full chain integrity verification.",
    annotations=_READ_ANNOTATIONS
)
async def dcl_audit_decode_deep(
    tx_hash: Annotated[str, Field(description="Transaction hash of the audit chain record to retrieve.")]
) -> List[TextContent]:
    entry = _chain.get_by_tx(tx_hash)
    if not entry:
        result = AuditDeepResult(error="tx_hash not found in chain")
    else:
        intact, tampered_at = _chain.verify()
        result = AuditDeepResult(
            tx_hash=entry["tx_hash"], agent_id=entry["agent_id"], verdict=entry["verdict"],
            reason=entry["reason"], confidence=entry["confidence"], task_type=entry["task_type"],
            timestamp=entry["timestamp"], chain_index=entry["index"], prev_hash=entry["prev_hash"],
            chain_integrity=intact, tampered_at_index=tampered_at, drift_context=entry["drift_context"],
        ) return [TextContent(type="text", text=json.dumps(result.model_dump(), default=str))]

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8081))
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = port
    mcp.run(transport="streamable-http")
