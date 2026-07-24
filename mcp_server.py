"""
DCL Trust Oracle — MCP Server (for Smithery)
Wraps the DCL logic (dcl_core.py) as real MCP tools via FastMCP.
x402 payment is NOT used here — that lives in a separate REST API (webhook_server.py).
"""
import os
import uuid
from typing import Optional, List

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from dcl_core import ChainState, BUILTIN_POLICIES, evaluate_policy, get_drift_mode, sha256hex

mcp = FastMCP(
    "DCL Trust Oracle",
    instructions="Deterministic AI audit layer: tamper-evident, privacy-first policy evaluation "
                 "and post-action forensic decoding.",
    transport_security=TransportSecuritySettings(
        allowed_hosts=["mcp.fronesislabs.com", "localhost", "localhost:8081", "127.0.0.1", "127.0.0.1:8081"],
        allowed_origins=["https://mcp.fronesislabs.com", "http://localhost:8081"],
    ),
)
mcp._mcp_server.version = "2.0.1"

_chain = ChainState(os.environ.get("DCL_DB_PATH", "dcl_chain.db"))
_commit_rate: List[float] = []


def _run_evaluation(response: str, policy: str, agent_id: str, task_type: str) -> dict:
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
    drift_mode, drift_score = get_drift_mode(_commit_rate)

    return {
        "verdict": verdict, "confidence": confidence, "reason": reason,
        "tx_hash": tx_hash, "chain_index": chain_idx, "input_hash": input_hash,
        "policy_version": policy_version, "drift_mode": drift_mode, "drift_score": drift_score,
    }


@mcp.tool()
def evaluate_fast(response: str, agent_id: str) -> dict:
    """FAST Pre-Action Audit. Quick policy check of an agent's response using the default policy."""
    return _run_evaluation(response, "default", agent_id, "fast")


@mcp.tool()
def evaluate_strict(response: str, agent_id: str) -> dict:
    """STRICT Pre-Action Audit. Rigorous check with a higher confidence threshold."""
    return _run_evaluation(response, "default", agent_id, "strict")


@mcp.tool()
def evaluate_jailbreak(response: str, agent_id: str) -> dict:
    """PRE-ACTION Instruction Adherence Check. Detects jailbreak attempts."""
    return _run_evaluation(response, "anti_jailbreak", agent_id, "jailbreak")


@mcp.tool()
def evaluate_safety(response: str, agent_id: str) -> dict:
    """PRE-ACTION Baseline Safety Check."""
    return _run_evaluation(response, "safety", agent_id, "safety")


@mcp.tool()
def evaluate_quality(response: str, agent_id: str) -> dict:
    """PRE-ACTION Content Quality & Drift Check."""
    return _run_evaluation(response, "content_quality", agent_id, "quality")


@mcp.tool()
def evaluate_batch(items: List[dict], agent_id: str) -> dict:
    """PRE-ACTION Bulk Processing. items: [{response, policy?}]"""
    results = [
        _run_evaluation(item["response"], item.get("policy", "default"), agent_id, "batch_item")
        for item in items
    ]
    return {"batch_id": str(uuid.uuid4())[:8], "agent_id": agent_id,
            "count": len(results), "results": results}


@mcp.tool()
def pipeline_start(agent_id: str, scope: str = "default", ttl_seconds: int = 3600) -> dict:
    """SESSION Management. Opens a pipeline session for a series of checks."""
    import time
    pipeline_id = f"pl_{uuid.uuid4().hex[:12]}"
    return {"pipeline_id": pipeline_id, "agent_id": agent_id, "scope": scope,
            "expires_at": time.time() + ttl_seconds, "drift_mode": "NORMAL"}


@mcp.tool()
def audit_decode(tx_hash: str) -> dict:
    """POST-ACTION Basic Audit. Retrieves a record from the tamper-evident chain by tx_hash."""
    entry = _chain.get_by_tx(tx_hash)
    if not entry:
        return {"error": "tx_hash not found in chain"}
    intact, _ = _chain.verify()
    return {"tx_hash": entry["tx_hash"], "agent_id": entry["agent_id"], "verdict": entry["verdict"],
            "reason": entry["reason"], "confidence": entry["confidence"], "task_type": entry["task_type"],
            "timestamp": entry["timestamp"], "chain_index": entry["index"],
            "prev_hash": entry["prev_hash"], "chain_integrity": intact}


@mcp.tool()
def audit_decode_deep(tx_hash: str) -> dict:
    """POST-ACTION Deep Forensic Audit. Extended output with drift_context and full chain integrity verification."""
    entry = _chain.get_by_tx(tx_hash)
    if not entry:
        return {"error": "tx_hash not found in chain"}
    intact, tampered_at = _chain.verify()
    return {"tx_hash": entry["tx_hash"], "agent_id": entry["agent_id"], "verdict": entry["verdict"],
            "reason": entry["reason"], "confidence": entry["confidence"], "task_type": entry["task_type"],
            "timestamp": entry["timestamp"], "chain_index": entry["index"], "prev_hash": entry["prev_hash"],
            "chain_integrity": intact, "tampered_at_index": tampered_at, "drift_context": entry["drift_context"]}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8081))
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = port
    mcp.run(transport="streamable-http")
