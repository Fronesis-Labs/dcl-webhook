"""
DCL Trust Oracle — MCP Server (for Smithery)

Wraps the DCL logic as real MCP tools via FastMCP. Chain/consensus protocol
logic now comes from the published dcl-core package (pip install dcl-core).
Policy evaluation, drift detection, and secret/PII scanning stay in
audit_logic.py (closed), mirroring webhook_server.py exactly. The crypto
suite (wallet/trade/MEV/signal/jailbreak-crypto/output-sanitizer) lives in
dcl_crypto.py.

x402 payments are gated directly on the MCP tools via PayMCP, so this
server charges the same per-call prices as the REST API in webhook_server.py.
"""

import os
import json
import time
import uuid
from collections import defaultdict
from functools import wraps

from dotenv import load_dotenv
load_dotenv()

from typing import Annotated, List, Optional
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations

from dcl_core import ChainState, sha256hex
from audit_logic import (
    BUILTIN_POLICIES, evaluate_policy, get_drift_mode,
    detect_secrets, detect_pii, format_seal,
)
from dcl_crypto import (
    detect_wallet, detect_jailbreak_crypto, detect_mev, detect_trade, detect_signal,
    detect_output_sanitizer,
)

from paymcp import PayMCP, Mode, price
from paymcp.providers import X402Provider

mcp = FastMCP(
    "DCL Trust Oracle",
    instructions=(
        "DCL Trust Oracle is a deterministic AI audit layer, natively integrated with the Model "
        "Context Protocol (MCP), that evaluates LLM and agent outputs against configurable policies "
        "before and after action. Every verdict is written to a tamper-evident, hash-chained audit "
        "log that stores only cryptographic metadata — never raw content — enabling privacy-first, "
        "post-action forensic review. The server exposes Jailbreak detection (instruction adherence "
        "checks), Quality and drift evaluation, and baseline Safety checks alongside fast and strict "
        "pre-action audit tiers. Tool calls are metered and settled via the x402 micropayment "
        "protocol (USDC on Base), letting autonomous agents pay per-call without any account setup. "
        "Use these tools to gate risky agent actions, batch-evaluate responses, and cryptographically "
        "verify the integrity of past decisions."
    ),
    transport_security=TransportSecuritySettings(
        allowed_hosts=["mcp.fronesislabs.com", "localhost", "localhost:8081", "127.0.0.1", "127.0.0.1:8081"],
        allowed_origins=["https://mcp.fronesislabs.com", "http://localhost:8081"],
    ),
)
mcp._mcp_server.version = "2.3.1"

# ════════════════════════════════════════════════════════════════════════════════
# Payments — PayMCP / x402 (mirrors pricing in webhook_server.py)
# ════════════════════════════════════════════════════════════════════════════════
# X402_WALLET must be the same env var/wallet used by webhook_server.py so revenue
# settles to one place. Use "eip155:84532" (Base Sepolia) instead of "eip155:8453"
# for testing before going live.
PayMCP(
    mcp,
    providers=[
        X402Provider(
            pay_to=[{
                "address": os.environ.get("X402_WALLET", "0x0000000000000000000000000000000000000000"),
                "network": os.environ.get("X402_NETWORK", "eip155:8453"),  # Base mainnet
            }]
        ),
    ],
    # AUTO: uses x402 for clients that support automatic on-chain payment,
    # falls back to a compatible guided flow for clients that don't.
    mode=Mode.X402,
)

_chain = ChainState(os.environ.get("DCL_DB_PATH", "dcl_chain.db"))
_commit_rate: List[float] = []

# ════════════════════════════════════════════════════════════════════════════════
# Output schemas (Pydantic models -> structured MCP output schemas)
# ════════════════════════════════════════════════════════════════════════════════
class EvaluateResult(BaseModel):
    verdict: str = Field(description="COMMIT if the response passed policy checks, otherwise NO_COMMIT.")
    confidence: float = Field(description="Confidence score of the verdict, from 0.0 to 1.0.")
    reason: str = Field(description="Human-readable explanation of why the verdict was reached.")
    tx_hash: str = Field(description="Hash of this record in the tamper-evident audit chain.")
    chain_index: int = Field(description="Sequential index of this record in the audit chain.")
    input_hash: str = Field(description="Hash of the evaluated response (raw content is never stored).")
    policy_version: str = Field(description="Version of the policy that was applied.")
    drift_mode: str = Field(description="Current drift status: NORMAL, WARNING, ESCALATION, or BLOCK.")
    drift_score: float = Field(description="Z-score measuring deviation of the recent commit rate from baseline.")
    timestamp: float = Field(description="Unix timestamp when this record was sealed.")
    seal_text: str = Field(description="Human-readable Leibniz Layer verification seal.")
    verify_url: str = Field(description="Public URL to independently verify this seal.")


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


class DetectionFinding(BaseModel):
    type: str = Field(description="The specific pattern category matched (e.g. 'api_key', 'email').")
    position: int = Field(description="Character offset of the match in the submitted text.")
    redacted_sample: str = Field(description="Masked version of the match — first 2 and last 4 chars only.")
    severity: str = Field(description="critical, major, or minor.")
    category: str = Field(description="Checklist code, e.g. S1-S8 for secrets or T1-T8 for PII.")
    provider: Optional[str] = Field(default=None, description="Identified provider/service, if known.")


class ScanResult(BaseModel):
    verdict: str = Field(description="COMMIT if nothing was found, otherwise NO_COMMIT.")
    risk_score: float = Field(description="0.0-1.0 risk score based on number and severity of findings.")
    findings: List[DetectionFinding] = Field(description="All matches found. Empty list if verdict is COMMIT.")
    detection_count: int = Field(description="Number of findings.")
    categories_checked: List[str] = Field(description="All checklist categories that were scanned.")
    categories_clear: List[str] = Field(description="Categories with no findings.")
    tx_hash: str = Field(description="Hash of this record in the tamper-evident audit chain.")
    chain_index: int = Field(description="Sequential index of this record in the audit chain.")
    input_hash: str = Field(description="Hash of the scanned text (raw content is never stored).")
    timestamp: float = Field(description="Unix timestamp when this record was sealed.")
    seal_text: str = Field(description="Human-readable Leibniz Layer verification seal.")
    verify_url: str = Field(description="Public URL to independently verify this seal.")


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
    seal_text: Optional[str] = Field(default=None, description="Human-readable Leibniz Layer verification seal.")
    verify_url: Optional[str] = Field(default=None, description="Public URL to independently verify this seal.")


class AuditDeepResult(AuditResult):
    tampered_at_index: Optional[int] = Field(
        default=None, description="Index where chain integrity broke, if any tampering was detected."
    )
    tamper_reason: Optional[str] = Field(
        default=None, description="Why chain_integrity is False — a broken prev_hash link or an edited row "
                                   "whose stored tx_hash no longer matches its recomputed content hash."
    )
    drift_context: Optional[dict] = Field(
        default=None, description="Extended forensic metadata captured at evaluation time."
    )


# ════════════════════════════════════════════════════════════════════════════════
# Crypto Suite output schemas (dcl_commit, dcl_evaluate_mev,
# dcl_evaluate_jailbreak_crypto, dcl_evaluate_signal, dcl_evaluate_trade,
# dcl_evaluate_wallet) — field shapes mirror dcl-skills/skills/crypto/*/references/mcp-tools.md
# ════════════════════════════════════════════════════════════════════════════════
class CommitResult(BaseModel):
    tx_hash: str = Field(description="Tamper-evident proof of this specific commit.")
    chain_hash: str = Field(description="Hash of the previous commit in the append-only chain that this one links to.")
    chain_depth: int = Field(description="This commit's position (index) in the chain.")
    input_hash: str = Field(description="Hash of the committed decision text (raw content is never stored).")
    timestamp: float = Field(description="Unix timestamp when this record was sealed.")


class CryptoFinding(BaseModel):
    type: str = Field(description="The specific pattern category matched.")
    severity: str = Field(description="critical or major.")
    regulatory_reference: Optional[str] = Field(
        default=None, description="Illustrative regulatory tag for this finding type (e.g. MiFID II, FCA), if applicable."
    )


class MevResult(BaseModel):
    verdict: str = Field(description="COMMIT if the response passed the MEV/compliance screen, otherwise NO_COMMIT.")
    confidence: float = Field(description="Confidence score of the verdict, from 0.0 to 1.0.")
    findings: List[CryptoFinding] = Field(description="All matched patterns. Empty list if verdict is COMMIT.")
    tx_hash: str = Field(description="Hash of this record in the tamper-evident audit chain.")
    chain_index: int = Field(description="Sequential index of this record in the audit chain.")
    input_hash: str = Field(description="Hash of the screened text (raw content is never stored).")
    timestamp: float = Field(description="Unix timestamp when this record was sealed.")


class JailbreakCryptoResult(BaseModel):
    verdict: str = Field(description="COMMIT if no injection pattern matched, otherwise NO_COMMIT.")
    confidence: float = Field(description="Confidence score of the verdict, from 0.0 to 1.0.")
    reason: str = Field(description="Human-readable explanation of the verdict.")
    findings: List[CryptoFinding] = Field(description="All matched patterns. Empty list if verdict is COMMIT.")
    tx_hash: str = Field(description="Hash of this record in the tamper-evident audit chain.")
    chain_index: int = Field(description="Sequential index of this record in the audit chain.")
    input_hash: str = Field(description="Hash of the screened text (raw content is never stored).")
    policy_version: str = Field(description="Version of the crypto jailbreak policy that was applied.")


class SignalResult(BaseModel):
    verdict: str = Field(description="COMMIT if no fabrication/overconfidence pattern matched, otherwise NO_COMMIT.")
    confidence: float = Field(description="Confidence score of the verdict, from 0.0 to 1.0.")
    reason: str = Field(description="Human-readable explanation of the verdict.")
    findings: List[CryptoFinding] = Field(description="All matched patterns. Empty list if verdict is COMMIT.")
    tx_hash: str = Field(description="Hash of this record in the tamper-evident audit chain.")
    chain_index: int = Field(description="Sequential index of this record in the audit chain.")
    input_hash: str = Field(description="Hash of the screened text (raw content is never stored).")
    timestamp: float = Field(description="Unix timestamp when this record was sealed.")


class TradeReceipt(BaseModel):
    tx_hash: str = Field(description="Tamper-evident proof of this specific trade-verification record.")
    chain_hash: str = Field(description="Hash of the previous commit in the append-only chain that this one links to.")
    chain_depth: int = Field(description="This record's position (index) in the chain.")


class TradeResult(BaseModel):
    verdict: str = Field(description="COMMIT if the trade decision's language passed the screen, otherwise NO_COMMIT.")
    confidence: float = Field(description="Confidence score of the verdict, from 0.0 to 1.0.")
    reason: str = Field(description="Human-readable explanation of the verdict.")
    findings: List[CryptoFinding] = Field(description="All matched patterns. Empty list if verdict is COMMIT.")
    trade_receipt: TradeReceipt = Field(description="Immutable receipt for this trade-verification record.")
    input_hash: str = Field(description="Hash of the screened text (raw content is never stored).")
    timestamp: float = Field(description="Unix timestamp when this record was sealed.")


class WalletFinding(BaseModel):
    type: str = Field(description="seed_phrase, private_key, wallet_address, or wallet_api_credential.")
    position: int = Field(description="Character offset of the match in the submitted text.")
    redacted_sample: str = Field(description="Masked version of the match — never the real value.")
    severity: str = Field(description="critical or major.")


class WalletResult(BaseModel):
    verdict: str = Field(description="COMMIT if nothing was found, otherwise NO_COMMIT. Wallet secrets have no safe threshold.")
    confidence: float = Field(description="Confidence score of the verdict, from 0.0 to 1.0.")
    reason: str = Field(description="Human-readable explanation of the verdict.")
    findings: List[WalletFinding] = Field(description="All matches found. Empty list if verdict is COMMIT.")
    sanitized_output: Optional[str] = Field(default=None, description="Input text with all matches redacted. Null if verdict is COMMIT.")
    risk_score: float = Field(description="0.0-1.0 risk score based on number and severity of findings.")
    tx_hash: str = Field(description="Hash of this record in the tamper-evident audit chain.")
    chain_index: int = Field(description="Sequential index of this record in the audit chain.")
    input_hash: str = Field(description="Hash of the scanned text (raw content is never stored).")
    policy_version: str = Field(description="Version of the wallet-guardian policy that was applied.")
    timestamp: float = Field(description="Unix timestamp when this record was sealed.")


class SanitizerFinding(BaseModel):
    type: str = Field(description="e.g. api_key, email, seed_phrase, internal_ip, mac_address, internal_hostname, self_harm_instruction, targeted_harassment, shell_injection, sql_injection, path_traversal, and others.")
    position: int = Field(description="Character offset of the match in the submitted text.")
    redacted_sample: str = Field(description="Masked version of the match — never the real value.")
    severity: str = Field(description="critical, major, or minor.")
    category: str = Field(description="secrets, pii, crypto, network, toxic, or unsafe_instructions.")


class OutputSanitizerResult(BaseModel):
    verdict: str = Field(description="COMMIT if the response was clean, otherwise NO_COMMIT.")
    confidence: float = Field(description="Confidence score of the verdict, from 0.0 to 1.0.")
    violations: List[str] = Field(description="Distinct finding types matched (e.g. ['api_key', 'internal_ip']). Empty list if verdict is COMMIT.")
    findings: List[SanitizerFinding] = Field(description="All matches found, with position/severity/category detail. Empty list if verdict is COMMIT.")
    sanitized_output: Optional[str] = Field(default=None, description="Input text with every match replaced by [REDACTED]. Null if verdict is COMMIT.")
    redaction_count: int = Field(description="Total number of items redacted.")
    risk_score: float = Field(description="0.0-1.0 composite severity score.")
    tx_hash: str = Field(description="Hash of this record in the tamper-evident audit chain.")
    chain_index: int = Field(description="Sequential index of this record in the audit chain.")
    input_hash: str = Field(description="Hash of the sanitized text (raw content is never stored).")
    timestamp: float = Field(description="Unix timestamp when this record was sealed.")


def _run_evaluation(response: str, policy: str, agent_id: str, task_type: str) -> EvaluateResult:
    policy_yaml = BUILTIN_POLICIES.get(policy, policy or BUILTIN_POLICIES["default"])
    verdict, confidence, reason, policy_version = evaluate_policy(response, policy_yaml)
    input_hash = "0x" + sha256hex(response)[:16]
    policy_hash = sha256hex(policy_yaml)[:16]

    tx_hash, chain_idx = _chain.append(
        verdict=verdict, input_hash=input_hash, policy_hash=policy_hash,
        agent_id=agent_id, reason=reason, confidence=confidence, task_type=task_type,
        drift_context={"environment": "production-edge", "policy_version_hash": policy_hash},
    )

    _commit_rate.append(1.0 if verdict == "COMMIT" else 0.0)
    if len(_commit_rate) > 100:
        _commit_rate.pop(0)
    drift_mode, drift_score = get_drift_mode(_commit_rate)

    ts = time.time()
    seal = format_seal(tx_hash, input_hash, ts)
    return EvaluateResult(
        verdict=verdict, confidence=confidence, reason=reason,
        tx_hash=tx_hash, chain_index=chain_idx, input_hash=input_hash,
        policy_version=policy_version, drift_mode=drift_mode, drift_score=drift_score,
        timestamp=ts, seal_text=seal["seal_text"], verify_url=seal["verify_url"],
    )


def _run_detection(text: str, detector, policy_label: str, agent_id: str, task_type: str) -> ScanResult:
    result = detector(text)
    input_hash = "0x" + sha256hex(text)[:16]
    policy_hash = sha256hex(policy_label)[:16]
    reason = (
        "; ".join(f"{f['category']}.{f['type']}" for f in result["findings"])
        if result["findings"] else "No patterns matched"
    )
    tx_hash, chain_idx = _chain.append(
        verdict=result["verdict"], input_hash=input_hash, policy_hash=policy_hash,
        agent_id=agent_id, reason=reason, confidence=1.0 - result["risk_score"], task_type=task_type,
        drift_context={"environment": "production-edge", "policy_version_hash": policy_hash},
    )
    ts = time.time()
    seal = format_seal(tx_hash, input_hash, ts)
    return ScanResult(
        verdict=result["verdict"], risk_score=result["risk_score"],
        findings=[DetectionFinding(**f) for f in result["findings"]],
        detection_count=result["detection_count"],
        categories_checked=result["categories_checked"], categories_clear=result["categories_clear"],
        tx_hash=tx_hash, chain_index=chain_idx, input_hash=input_hash,
        timestamp=ts, seal_text=seal["seal_text"], verify_url=seal["verify_url"],
    )


def _commit_to_chain(input_text: str, verdict: str, reason: str, confidence: float,
                      agent_id: str, task_type: str, policy_label: str):
    """Shared chain-append helper for the crypto suite tools — same pattern as
    _run_evaluation/_run_detection above, factored out since all six crypto
    tools need the tx_hash/chain_index/input_hash triple but each has a
    different result shape."""
    input_hash = "0x" + sha256hex(input_text)[:16]
    policy_hash = sha256hex(policy_label)[:16]
    tx_hash, chain_idx = _chain.append(
        verdict=verdict, input_hash=input_hash, policy_hash=policy_hash,
        agent_id=agent_id, reason=reason, confidence=confidence, task_type=task_type,
    )
    return tx_hash, chain_idx, input_hash


def _run_commit(decision: str, agent_id: str, prior_checks: Optional[dict]) -> CommitResult:
    reason = f"crypto_commit; prior_checks={json.dumps(prior_checks)}" if prior_checks else "crypto_commit"
    tx_hash, chain_idx, input_hash = _commit_to_chain(
        decision, "COMMIT", reason, 1.0, agent_id, "crypto_commit", "crypto_commit_v1"
    )
    chain_hash = _chain.get_by_tx(tx_hash)["prev_hash"]
    return CommitResult(
        tx_hash=tx_hash, chain_hash=chain_hash, chain_depth=chain_idx,
        input_hash=input_hash, timestamp=time.time(),
    )


def _run_mev(response: str, agent_id: str) -> MevResult:
    result = detect_mev(response)
    reason = "; ".join(f["type"] for f in result["findings"]) or "No patterns matched"
    tx_hash, chain_idx, input_hash = _commit_to_chain(
        response, result["verdict"], reason, result["confidence"], agent_id, "mev", "mev_compliance_v1"
    )
    return MevResult(
        verdict=result["verdict"], confidence=result["confidence"],
        findings=[CryptoFinding(**f) for f in result["findings"]],
        tx_hash=tx_hash, chain_index=chain_idx, input_hash=input_hash, timestamp=time.time(),
    )


def _run_jailbreak_crypto(response: str, agent_id: str) -> JailbreakCryptoResult:
    result = detect_jailbreak_crypto(response)
    tx_hash, chain_idx, input_hash = _commit_to_chain(
        response, result["verdict"], result["reason"], result["confidence"],
        agent_id, "jailbreak_crypto", result["policy_version"],
    )
    return JailbreakCryptoResult(
        verdict=result["verdict"], confidence=result["confidence"], reason=result["reason"],
        findings=[CryptoFinding(type=f["type"], severity=f["severity"]) for f in result["findings"]],
        tx_hash=tx_hash, chain_index=chain_idx, input_hash=input_hash,
        policy_version=result["policy_version"],
    )


def _run_signal(response: str, agent_id: str) -> SignalResult:
    result = detect_signal(response)
    tx_hash, chain_idx, input_hash = _commit_to_chain(
        response, result["verdict"], result["reason"], result["confidence"], agent_id, "signal", "semantic_drift_crypto_v1"
    )
    return SignalResult(
        verdict=result["verdict"], confidence=result["confidence"], reason=result["reason"],
        findings=[CryptoFinding(type=f["type"], severity=f["severity"]) for f in result["findings"]],
        tx_hash=tx_hash, chain_index=chain_idx, input_hash=input_hash, timestamp=time.time(),
    )


def _run_trade(response: str, agent_id: str) -> TradeResult:
    result = detect_trade(response)
    tx_hash, chain_idx, input_hash = _commit_to_chain(
        response, result["verdict"], result["reason"], result["confidence"], agent_id, "trade", "trade_verifier_v1"
    )
    chain_hash = _chain.get_by_tx(tx_hash)["prev_hash"]
    return TradeResult(
        verdict=result["verdict"], confidence=result["confidence"], reason=result["reason"],
        findings=[CryptoFinding(type=f["type"], severity=f["severity"]) for f in result["findings"]],
        trade_receipt=TradeReceipt(tx_hash=tx_hash, chain_hash=chain_hash, chain_depth=chain_idx),
        input_hash=input_hash, timestamp=time.time(),
    )


def _run_wallet(response: str, agent_id: str) -> WalletResult:
    result = detect_wallet(response)
    tx_hash, chain_idx, input_hash = _commit_to_chain(
        response, result["verdict"], result["reason"], result["confidence"], agent_id, "wallet", "wallet_guardian_v1"
    )
    return WalletResult(
        verdict=result["verdict"], confidence=result["confidence"], reason=result["reason"],
        findings=[WalletFinding(**f) for f in result["findings"]],
        sanitized_output=result["sanitized_output"], risk_score=result["risk_score"],
        tx_hash=tx_hash, chain_index=chain_idx, input_hash=input_hash,
        policy_version="wallet_guardian_v1", timestamp=time.time(),
    )


def _run_output_sanitizer(response: str, agent_id: str) -> OutputSanitizerResult:
    result = detect_output_sanitizer(response)
    reason = "; ".join(result["violations"]) or "No sensitive content detected"
    tx_hash, chain_idx, input_hash = _commit_to_chain(
        response, result["verdict"], reason, result["confidence"], agent_id, "output_sanitizer", "output_sanitizer_v1"
    )
    return OutputSanitizerResult(
        verdict=result["verdict"], confidence=result["confidence"], violations=result["violations"],
        findings=[SanitizerFinding(**f) for f in result["findings"]],
        sanitized_output=result["sanitized_output"], redaction_count=result["redaction_count"],
        risk_score=result["risk_score"], tx_hash=tx_hash, chain_index=chain_idx,
        input_hash=input_hash, timestamp=time.time(),
    )


_WRITE_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False,
)
_READ_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False,
)

# ════════════════════════════════════════════════════════════════════════════════
# Rate limiting — the REST API in webhook_server.py already has slowapi; these
# MCP tools had no limiter at all, which combined with dcl_evaluate_batch's
# unbounded item count is a cost/resource-exhaustion vector even behind x402
# (an attacker willing to pay per call could still hammer the server). Keyed by
# client IP from the underlying Starlette request (honoring X-Forwarded-For,
# since nginx proxies mcp.fronesislabs.com), falling back to agent_id for
# transports without an HTTP request (e.g. stdio). Simple in-memory sliding
# window — sufficient for a single PM2 process; swap for Redis if this ever
# scales to multiple workers.
# ════════════════════════════════════════════════════════════════════════════════
_RATE_LIMIT_WINDOW_SECONDS = 60
_RATE_LIMIT_MAX_CALLS = 30
_MAX_BATCH_ITEMS = 200
_rate_limit_hits: dict[str, list[float]] = defaultdict(list)


def _rate_limit_key(ctx: "Context | None", agent_id: str | None) -> str:
    if ctx is not None:
        try:
            request = ctx.request_context.request
            if request is not None:
                forwarded = request.headers.get("x-forwarded-for")
                if forwarded:
                    return forwarded.split(",")[0].strip()
                if request.client:
                    return request.client.host
        except Exception:
            pass
    return f"agent:{agent_id}" if agent_id else "unknown"


def _rate_limited(fn):
    """Reject a call before payment is collected if the caller is over the
    per-window limit. Applied above @price so a rate-limited call never gets
    charged."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        ctx = kwargs.get("ctx")
        if ctx is None:
            for a in args:
                if isinstance(a, Context):
                    ctx = a
                    break
        key = _rate_limit_key(ctx, kwargs.get("agent_id"))
        now = time.time()
        hits = _rate_limit_hits[key]
        cutoff = now - _RATE_LIMIT_WINDOW_SECONDS
        while hits and hits[0] < cutoff:
            hits.pop(0)
        if len(hits) >= _RATE_LIMIT_MAX_CALLS:
            raise ValueError(
                f"Rate limit exceeded: max {_RATE_LIMIT_MAX_CALLS} calls per "
                f"{_RATE_LIMIT_WINDOW_SECONDS}s per client. Try again shortly."
            )
        hits.append(now)
        return fn(*args, **kwargs)
    return wrapper


# ════════════════════════════════════════════════════════════════════════════════
# PAID TOOLS — pricing mirrors webhook_server.py exactly
# ════════════════════════════════════════════════════════════════════════════════
@mcp.tool(title="Fast Pre-Action Audit", annotations=_WRITE_ANNOTATIONS)
@_rate_limited
@price(price=0.01, currency="USD")
def dcl_evaluate_fast(
    response: Annotated[str, Field(description="The agent or LLM response text to audit.")],
    agent_id: Annotated[str, Field(description="Identifier of the agent that produced the response.")],
    ctx: Context,
) -> EvaluateResult:
    """FAST Pre-Action Audit ($0.01). Runs the response through the server's "default" policy: a substring check against 3 forbidden phrases ("ignore previous instructions", "jailbreak", "bypass safety") with a 0.7 minimum-confidence threshold. Each forbidden match found costs 0.4 confidence; if confidence falls below 0.7, or any match is found, the verdict is NO_COMMIT and `reason` lists which phrase triggered it. Otherwise COMMIT. Use this as the default low-cost first-pass gate before a risky agent action; switch to dcl_evaluate_strict for a broader, higher-bar check, or to dcl_evaluate_jailbreak / dcl_evaluate_safety / dcl_evaluate_quality for a narrower, single-topic check instead of the general-purpose default policy."""
    return _run_evaluation(response, "default", agent_id, "fast")


@mcp.tool(title="Strict Pre-Action Audit", annotations=_WRITE_ANNOTATIONS)
@_rate_limited
@price(price=0.05, currency="USD")
def dcl_evaluate_strict(
    response: Annotated[str, Field(description="The agent or LLM response text to audit.")],
    agent_id: Annotated[str, Field(description="Identifier of the agent that produced the response.")],
    ctx: Context,
) -> EvaluateResult:
    """STRICT Pre-Action Audit ($0.05). Runs the response against a broader, higher-bar "strict" policy: the union of all forbidden phrases from the default, anti-jailbreak, and safety policies (8 phrases total), with a 0.85 minimum-confidence threshold instead of the default policy's 0.7. Each matched phrase costs 0.4 confidence; if confidence falls below 0.85, or any phrase matches, the verdict is NO_COMMIT with `reason` listing every match found. Use this instead of dcl_evaluate_fast when the cost of a false COMMIT is high — e.g. before an irreversible or high-stakes agent action — since it catches jailbreak- and safety-adjacent phrasing that the plain default policy would miss."""
    return _run_evaluation(response, "strict", agent_id, "strict")


@mcp.tool(title="Jailbreak Detection Check", annotations=_WRITE_ANNOTATIONS)
@_rate_limited
@price(price=0.02, currency="USD")
def dcl_evaluate_jailbreak(
    response: Annotated[str, Field(description="The agent or LLM response text to check for jailbreak attempts.")],
    agent_id: Annotated[str, Field(description="Identifier of the agent that produced the response.")],
    ctx: Context,
) -> EvaluateResult:
    """PRE-ACTION Instruction Adherence Check ($0.02). Runs the "anti_jailbreak" policy: a substring check against 6 forbidden phrases ("ignore previous instructions", "jailbreak", "bypass safety", "pretend you are", "act as if", "DAN") with a 0.8 minimum-confidence threshold — each match costs 0.4 confidence. Returns COMMIT if no phrase matches and confidence stays at or above 0.8, otherwise NO_COMMIT with `reason` listing the matched phrase(s). Use this as a targeted, cheaper check when the concern is specifically prompt-injection / persona-hijack risk; use dcl_evaluate_strict instead when you also want safety- and default-policy phrases covered in the same call."""
    return _run_evaluation(response, "anti_jailbreak", agent_id, "jailbreak")


@mcp.tool(title="Baseline Safety Check", annotations=_WRITE_ANNOTATIONS)
@_rate_limited
@price(price=0.01, currency="USD")
def dcl_evaluate_safety(
    response: Annotated[str, Field(description="The agent or LLM response text to check for safety violations.")],
    agent_id: Annotated[str, Field(description="Identifier of the agent that produced the response.")],
    ctx: Context,
) -> EvaluateResult:
    """PRE-ACTION Baseline Safety Check ($0.01). Runs the "safety" policy: flags 2 forbidden disclaimers ("I cannot be held responsible", "no guarantees") and additionally REQUIRES the substring "AI" to appear somewhere in the response — missing it costs 0.2 confidence even with no forbidden phrase present. Minimum confidence is 0.75. Returns NO_COMMIT if confidence drops below 0.75, with `reason` naming the forbidden phrase found or the missing required pattern. Use this when you specifically need to confirm an AI-disclosure marker is present and the two disclaimer phrases are absent — not as a general-purpose safety net; for broader coverage use dcl_evaluate_fast or dcl_evaluate_strict instead."""
    return _run_evaluation(response, "safety", agent_id, "safety")


@mcp.tool(title="Content Quality & Drift Check", annotations=_WRITE_ANNOTATIONS)
@_rate_limited
@price(price=0.03, currency="USD")
def dcl_evaluate_quality(
    response: Annotated[str, Field(description="The agent or LLM response text to check for quality and drift.")],
    agent_id: Annotated[str, Field(description="Identifier of the agent that produced the response.")],
    ctx: Context,
) -> EvaluateResult:
    """PRE-ACTION Content Quality & Drift Check ($0.03). Runs the "content_quality" policy: flags 12 absolutist or unverifiable-claim phrases (e.g. "guaranteed returns", "100% accurate", "studies show", "without a doubt") with a 0.85 minimum-confidence threshold — the highest bar of any single-policy tool. Returns NO_COMMIT if any phrase matches or confidence falls below 0.85, with `reason` listing the matched phrase(s). Use this to catch overconfident or unsubstantiated claims in generated content — a different concern from jailbreak or safety phrasing — e.g. before publishing agent-written copy or reports."""
    return _run_evaluation(response, "content_quality", agent_id, "quality")


@mcp.tool(title="Secret & Credential Leak Scan", annotations=_WRITE_ANNOTATIONS)
@_rate_limited
@price(price=0.02, currency="USD")
def dcl_evaluate_secrets(
    response: Annotated[str, Field(description="The text to scan for exposed API keys, tokens, private keys, DB URLs, and other credentials.")],
    agent_id: Annotated[str, Field(description="Identifier of the agent that produced the response.")],
    ctx: Context,
) -> ScanResult:
    """POST-ACTION Secret & Credential Leak Scan ($0.02). Regex-based scan across 8 categories (API keys, cloud credentials, tokens/JWTs, private keys, DB URLs, connection strings, env assignments, webhook secrets, internal endpoints with auth). Any finding results in NO_COMMIT."""
    return _run_detection(response, detect_secrets, "secret_leak_v1", agent_id, "secrets")


@mcp.tool(title="PII Detection Scan", annotations=_WRITE_ANNOTATIONS)
@_rate_limited
@price(price=0.02, currency="USD")
def dcl_evaluate_pii(
    response: Annotated[str, Field(description="The text to scan for personal data: emails, phone numbers, national IDs, bank cards, IBANs, crypto addresses, IP addresses, passport numbers.")],
    agent_id: Annotated[str, Field(description="Identifier of the agent that produced the response.")],
    ctx: Context,
) -> ScanResult:
    """POST-ACTION PII Detection Scan ($0.02). Regex-based scan across 8 personal-data categories, with a Luhn checksum on card numbers to reduce false positives. Any finding results in NO_COMMIT."""
    return _run_detection(response, detect_pii, "pii_v1", agent_id, "pii")


@mcp.tool(title="Batch Evaluation", annotations=_WRITE_ANNOTATIONS)
@_rate_limited
@price(price=0.10, currency="USD")
def dcl_evaluate_batch(
    items: Annotated[
        List[dict],
        Field(description="List of items to evaluate, each shaped like {'response': str, 'policy'?: str}."),
    ],
    agent_id: Annotated[str, Field(description="Identifier of the agent that produced the responses.")],
    ctx: Context,
) -> BatchResult:
    """PRE-ACTION Bulk Processing ($0.10). Evaluates a list of items in one call; each item is a dict shaped {"response": str, "policy"?: str}, where policy defaults to "default" if omitted and may be any built-in policy name (default, strict, anti_jailbreak, safety, content_quality). Each item gets its own independent COMMIT/NO_COMMIT verdict via the same logic as the matching single-item evaluate_* tool; results are returned in input order under `results`, plus a shared `batch_id`. Capped at 200 items per call — oversized batches are rejected. Use this instead of multiple single-item evaluate_* calls when checking several responses — optionally against different policies — in one priced call rather than paying per item separately."""
    if len(items) > _MAX_BATCH_ITEMS:
        raise ValueError(f"Batch too large: {len(items)} items exceeds the {_MAX_BATCH_ITEMS}-item cap.")
    results = [
        _run_evaluation(item["response"], item.get("policy", "default"), agent_id, "batch_item")
        for item in items
    ]
    return BatchResult(batch_id=str(uuid.uuid4())[:8], agent_id=agent_id, count=len(results), results=results)


@mcp.tool(title="Start Pipeline Session", annotations=_WRITE_ANNOTATIONS)
@_rate_limited
@price(price=0.05, currency="USD")
def dcl_pipeline_start(
    agent_id: Annotated[str, Field(description="Identifier of the agent that owns this session.")],
    ctx: Context,
    scope: Annotated[str, Field(description="Scope label for the session.")] = "default",
    ttl_seconds: Annotated[int, Field(description="Session time-to-live, in seconds.")] = 3600,
) -> PipelineStartResult:
    """SESSION Management ($0.05). Generates a new `pipeline_id` and returns session metadata (scope, expiry, initial drift_mode) for organizing a series of related checks under one identifier. Note: this call does not currently link the returned pipeline_id to later evaluate_* calls — there is no server-side session state that ties subsequent audits back to it; it is an identifier/timestamp issuer, not an active tracking session. Use this to obtain a shared reference ID for your own client-side grouping of a multi-step audit sequence; do not rely on it to automatically aggregate drift across calls."""
    pipeline_id = f"pl_{uuid.uuid4().hex[:12]}"
    return PipelineStartResult(
        pipeline_id=pipeline_id, agent_id=agent_id, scope=scope,
        expires_at=time.time() + ttl_seconds, drift_mode="NORMAL",
    )


@mcp.tool(title="Basic Audit Decode", annotations=_READ_ANNOTATIONS)
@_rate_limited
@price(price=0.10, currency="USD")
def dcl_audit_decode(
    tx_hash: Annotated[str, Field(description="Transaction hash of the audit chain record to retrieve.")],
    ctx: Context,
) -> AuditResult:
    """POST-ACTION Basic Audit ($0.10). Retrieves a record from the tamper-evident chain by tx_hash."""
    entry = _chain.get_by_tx(tx_hash)
    if not entry:
        return AuditResult(error="tx_hash not found in chain")
    intact, _, _ = _chain.verify()
    seal = format_seal(entry["tx_hash"], entry["input_hash"], entry["timestamp"])
    return AuditResult(
        tx_hash=entry["tx_hash"], agent_id=entry["agent_id"], verdict=entry["verdict"],
        reason=entry["reason"], confidence=entry["confidence"], task_type=entry["task_type"],
        timestamp=entry["timestamp"], chain_index=entry["index"],
        prev_hash=entry["prev_hash"], chain_integrity=intact,
        seal_text=seal["seal_text"], verify_url=seal["verify_url"],
    )


@mcp.tool(title="Deep Forensic Audit Decode", annotations=_READ_ANNOTATIONS)
@_rate_limited
@price(price=0.50, currency="USD")
def dcl_audit_decode_deep(
    tx_hash: Annotated[str, Field(description="Transaction hash of the audit chain record to retrieve.")],
    ctx: Context,
) -> AuditDeepResult:
    """POST-ACTION Deep Forensic Audit ($0.50). Extended output with drift_context and full chain integrity verification."""
    entry = _chain.get_by_tx(tx_hash)
    if not entry:
        return AuditDeepResult(error="tx_hash not found in chain")
    intact, tampered_at, tamper_reason = _chain.verify()
    seal = format_seal(entry["tx_hash"], entry["input_hash"], entry["timestamp"])
    return AuditDeepResult(
        tx_hash=entry["tx_hash"], agent_id=entry["agent_id"], verdict=entry["verdict"],
        reason=entry["reason"], confidence=entry["confidence"], task_type=entry["task_type"],
        timestamp=entry["timestamp"], chain_index=entry["index"], prev_hash=entry["prev_hash"],
        chain_integrity=intact, tampered_at_index=tampered_at, tamper_reason=tamper_reason,
        drift_context=entry["drift_context"],
        seal_text=seal["seal_text"], verify_url=seal["verify_url"],
    )


# ════════════════════════════════════════════════════════════════════════════════
# DCL CRYPTO SUITE — implements the 6 tools documented (but not yet wired up) in
# Fronesis-Labs/dcl-skills' skills/crypto/*/references/mcp-tools.md. Same
# pattern as the tools above: @mcp.tool + @price + a typed pydantic return.
# Suggested pipeline order: firewall -> wallet -> trade -> mev -> commit (last).
# ════════════════════════════════════════════════════════════════════════════════
@mcp.tool(title="Crypto Jailbreak & Injection Detection", annotations=_WRITE_ANNOTATIONS)
@_rate_limited
@price(price=0.02, currency="USD")
def dcl_evaluate_jailbreak_crypto(
    response: Annotated[str, Field(description="The incoming prompt or agent response to screen for crypto-specialized jailbreak/injection attempts.")],
    agent_id: Annotated[str, Field(description="Identifier of the agent that produced or received the text.")],
    ctx: Context,
) -> JailbreakCryptoResult:
    """PRE-ACTION Crypto Jailbreak & Injection Detection ($0.02). Crypto-specialized instruction-override/jailbreak/injection screen: standard role-switch and instruction-override patterns, plus crypto-specific drain-wallet injection (e.g. "transfer all funds to...", fake "test transaction" requesting full balance) and unlimited-approval injection (e.g. type(uint256).max, "approve unlimited allowance", skip-slippage-confirmation framing). Any match returns NO_COMMIT with `reason` and `findings` naming the matched category/categories; run this FIRST in the DCL crypto pipeline, before wallet/trade/MEV checks, since it screens the input itself rather than a decision built on top of it."""
    return _run_jailbreak_crypto(response, agent_id)


@mcp.tool(title="Wallet Secret Guardian", annotations=_WRITE_ANNOTATIONS)
@_rate_limited
@price(price=0.02, currency="USD")
def dcl_evaluate_wallet(
    response: Annotated[str, Field(description="The text to scan for seed phrases, private keys, wallet addresses, and wallet-context API credentials.")],
    agent_id: Annotated[str, Field(description="Identifier of the agent that produced the response.")],
    ctx: Context,
) -> WalletResult:
    """POST-ACTION Wallet Secret Guardian ($0.02). Scans for BIP-39 seed phrases (12 or 24 consecutive wordlist words), raw hex or WIF-format private keys, Ethereum/Bitcoin wallet addresses, and API keys/bearer tokens appearing near wallet/custody/signing terminology. Any finding results in NO_COMMIT — wallet secrets have no safe threshold, unlike other DCL evaluators. Returns a `sanitized_output` with all matches redacted (null if nothing was found) and a masked `redacted_sample` per finding — the real value is never returned or stored server-side."""
    return _run_wallet(response, agent_id)


@mcp.tool(title="Trade Decision Verifier", annotations=_WRITE_ANNOTATIONS)
@_rate_limited
@price(price=0.02, currency="USD")
def dcl_evaluate_trade(
    response: Annotated[str, Field(description="The trade decision or recommendation text to screen.")],
    agent_id: Annotated[str, Field(description="Identifier of the agent that produced the response.")],
    ctx: Context,
) -> TradeResult:
    """PRE-ACTION Trade Decision Verifier ($0.02). Screens trade-decision language for guaranteed-return claims, zero-risk/"can't lose" framing, and unqualified "buy/sell X now" directives — any match is NO_COMMIT. If no unsafe language is found, COMMIT additionally requires the word "risk" to appear anywhere in the text as a minimum disclosure marker; its absence alone triggers NO_COMMIT with `reason` noting the missing disclosure. Produces an immutable `trade_receipt` (tx_hash/chain_hash/chain_depth) distinct from the top-level audit hash, for downstream systems that specifically need a trade-shaped receipt object."""
    return _run_trade(response, agent_id)


@mcp.tool(title="MEV & Market-Abuse Compliance Screen", annotations=_WRITE_ANNOTATIONS)
@_rate_limited
@price(price=0.03, currency="USD")
def dcl_evaluate_mev(
    response: Annotated[str, Field(description="The agent or LLM response text describing or proposing an on-chain/trading action, to screen for MEV and market-abuse language.")],
    agent_id: Annotated[str, Field(description="Identifier of the agent that produced the response.")],
    ctx: Context,
) -> MevResult:
    """POST-ACTION MEV & Market-Abuse Compliance Screen ($0.03). Text-level screen (not a mempool/transaction analyzer) for front-running/sandwich-attack language, wash trading/layering/spoofing, KYC/AML red flags (mixers, structuring, obscuring fund origin), and pump-and-dump/rug-pull language. Any critical-severity finding, or two or more major-severity findings, returns NO_COMMIT; a single major-severity finding is also returned as NO_COMMIT but with a distinctly higher `confidence` (~0.55 vs ~0.05-0.2 for harder violations) so downstream callers can tell a soft single flag apart from a hard multi-finding block. Each finding includes an illustrative `regulatory_reference` tag (MiFID II, FCA, or an EU AI Act article)."""
    return _run_mev(response, agent_id)


@mcp.tool(title="Market Signal Fabrication Screen", annotations=_WRITE_ANNOTATIONS)
@_rate_limited
@price(price=0.03, currency="USD")
def dcl_evaluate_signal(
    response: Annotated[str, Field(description="The market signal, analysis, or price-prediction text to screen.")],
    agent_id: Annotated[str, Field(description="Identifier of the agent that produced the response.")],
    ctx: Context,
) -> SignalResult:
    """POST-ACTION Market Signal Fabrication Screen ($0.03). Pattern-based heuristic on the output text alone (no source price feed) — flags guaranteed-price-prediction language ("will definitely hit $X"), absolute-certainty claims ("100% certain", "cannot go down"), a fabricated-price flag when a specific dollar figure co-occurs with a guaranteed-outcome claim, and an invented-token flag when a "$TICKER" cashtag doesn't match a small set of well-known symbols (false positives are possible for legitimate lesser-known tickers — this is a heuristic pre-check, not ground truth). For a full claim-by-claim check against an actual price-feed snapshot, use the local grounding workflow instead of this live tool. Verdict/confidence collapsing follows the same rule as dcl_evaluate_mev: any critical finding or 2+ major findings is a hard NO_COMMIT; exactly one major finding is a softer NO_COMMIT at ~0.55 confidence."""
    return _run_signal(response, agent_id)


@mcp.tool(title="Output Sanitizer — Final Gate", annotations=_WRITE_ANNOTATIONS)
@_rate_limited
@price(price=0.02, currency="USD")
def dcl_evaluate_output_sanitizer(
    response: Annotated[str, Field(description="The raw LLM/agent response to sanitize before it is delivered to a user, downstream agent, or external system.")],
    agent_id: Annotated[str, Field(description="Identifier of the agent that produced the response.")],
    ctx: Context,
) -> OutputSanitizerResult:
    """FINAL-GATE Output Sanitizer ($0.02). Post-processing checkpoint that strips secrets/credentials, PII, crypto material (seed phrases, private keys, wallet addresses), internal network details (private IPs, MAC addresses, .internal/.local/.corp hostnames), and unsafe shell/SQL/path-traversal fragments from a raw model response — plus a narrow, high-precision safety net for direct self-harm-instruction-seeking and targeted-harassment phrasing (not a general toxicity classifier). Returns a single `sanitized_output` with every match replaced by `[REDACTED]`; use that instead of the original whenever verdict is NO_COMMIT. Run this as the LAST gate before a response reaches its destination — after `dcl_evaluate_jailbreak_crypto`/other input-side checks have already run, and immediately before `dcl_commit` seals the final decision. Internally re-uses the same detection tables as `dcl_evaluate_secrets`/`dcl_evaluate_pii` for the secrets/PII categories, so results stay consistent with those tools."""
    return _run_output_sanitizer(response, agent_id)


@mcp.tool(title="Leibniz Layer Crypto Commit", annotations=_WRITE_ANNOTATIONS)
@_rate_limited
@price(price=0.01, currency="USD")
def dcl_commit(
    decision: Annotated[str, Field(description="The final trading/agent decision text to commit to the audit chain.")],
    agent_id: Annotated[str, Field(description="Identifier of the agent whose decision is being committed.")],
    ctx: Context,
    prior_checks: Annotated[
        Optional[dict],
        Field(description="Optional dict of tx_hashes from earlier pipeline steps (e.g. {'prompt_firewall_tx_hash': ..., 'trade_verifier_tx_hash': ..., 'mev_compliance_tx_hash': ...}), linking this commit to the specific checks that passed before it."),
    ] = None,
) -> CommitResult:
    """FINAL-STEP Leibniz Layer Crypto Commit ($0.01). Writes a trading/agent decision to the append-only Leibniz Layer audit chain and returns a Merkle-proof-style receipt: `tx_hash` (proof of this specific commit), `chain_hash` (the previous commit's hash, linking this one into the chain), and `chain_depth` (this commit's position in the chain). Unlike the evaluate_* tools, this call has no pass/fail verdict of its own — it always succeeds and simply seals the decision. Passing `prior_checks` is optional but recommended: it records which earlier pipeline steps (firewall/wallet/trade/MEV) this specific commit is downstream of, in one auditable record. Always run this LAST, after every other crypto-suite check has passed."""
    return _run_commit(decision, agent_id, prior_checks)


if __name__ == "__main__":
    # MCP_TRANSPORT=stdio lets platforms like Glama wrap this server via
    # mcp-proxy, which spawns the process and speaks stdio to it. Left
    # unset (default), this keeps running as an HTTP server for PM2/VPS.
    if os.environ.get("MCP_TRANSPORT") == "stdio":
        mcp.run(transport="stdio")
    else:
        port = int(os.environ.get("PORT", 8081))
        mcp.settings.host = "0.0.0.0"
        mcp.settings.port = port
        mcp.run(transport="streamable-http")
