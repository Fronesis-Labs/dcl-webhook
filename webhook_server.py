"""
DCL Webhook Server v2.2.0 — x402 Micropayments + Extended Metadata Audit

Deterministic AI audit layer. Tamper-evident. Metadata-only.

Chain/consensus protocol logic now comes from the published dcl-core
package (pip install dcl-core). Policy evaluation, drift detection, and
secret/PII scanning stay in audit_logic.py (closed). This file remains
transport/payment plumbing only.
"""

import json
import os
import time
import uuid
from typing import Any, Optional, List, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi_x402 import init_x402, pay
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from dcl_core import ChainState, sha256hex
from audit_logic import (
    BUILTIN_POLICIES, evaluate_policy, get_drift_mode,
    detect_secrets, detect_pii, format_seal,
)
from sentinel_db import SentinelDB
from sentinel_audit import audit_repo_release
from sentinel_x402 import require_x402_payment, scan_price, SCAN_PRICES
from sentinel_logic import (
    apply_webhook_scan,
    audit_to_dict,
    check_rate_limit,
    default_policy,
    new_webhook_secret,
    parse_github_release_version,
    verify_github_signature,
)

try:
    from telemetry import get_collector
except ImportError:
    class DummyCollector:
        def record_decision(self, **kwargs): pass
    def get_collector(): return DummyCollector()

# ════════════════════════════════════════════════════════════════════════════════
# App & x402 & Rate Limiting
# ════════════════════════════════════════════════════════════════════════════════

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="DCL Trust Oracle — Webhook API (x402)",
    description="Deterministic AI audit layer with micropayments. Tamper-evident. Metadata-only.",
    version="2.2.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

init_x402(
    app,
    pay_to=os.environ.get("X402_WALLET", "0x0000000000000000000000000000000000000000"),
    facilitator_url="https://x402.org/facilitator",
    network=["base", "avalanche", "iotex"],
)

_chain = ChainState(os.environ.get("DCL_DB_PATH", "dcl_chain.db"))
_sentinel_db = SentinelDB(os.environ.get("DCL_DB_PATH", "dcl_chain.db"))
_commit_rate: list[float] = []

# ════════════════════════════════════════════════════════════════════════════════
# Request / Response Models
# ════════════════════════════════════════════════════════════════════════════════

class EvaluateRequest(BaseModel):
    response: str
    policy: Optional[str] = "default"
    agent_id: Optional[str] = "unknown"
    model: Optional[str] = "unknown"
    model_provider: Optional[str] = "unknown"
    pipeline_id: Optional[str] = ""
    task_type: Optional[str] = "unknown"
    retry_count: Optional[int] = 0
    rag_source_count: Optional[int] = 0

class EvaluateResponse(BaseModel):
    verdict: str
    confidence: float
    reason: str
    tx_hash: str
    chain_index: int
    input_hash: str
    policy_version: str
    timestamp: float
    pipeline_id: str
    drift_mode: str
    drift_score: float
    seal_text: str
    verify_url: str

class ScanRequest(BaseModel):
    response: str
    agent_id: Optional[str] = "unknown"
    task_type: Optional[str] = "unknown"

class ScanFinding(BaseModel):
    type: str
    position: int
    redacted_sample: str
    severity: str
    category: str
    provider: Optional[str] = None

class ScanResponse(BaseModel):
    verdict: str
    risk_score: float
    findings: List[ScanFinding]
    detection_count: int
    categories_checked: List[str]
    categories_clear: List[str]
    tx_hash: str
    chain_index: int
    input_hash: str
    timestamp: float
    seal_text: str
    verify_url: str

class BatchItem(BaseModel):
    response: str
    policy: Optional[str] = "default"
    task_type: Optional[str] = "batch_item"

class BatchEvaluateRequest(BaseModel):
    items: List[BatchItem]
    agent_id: str
    max_items: int = 20

class PipelineStartRequest(BaseModel):
    agent_id: str
    scope: str = "default"
    ttl_seconds: int = 3600

class PipelineStartResponse(BaseModel):
    pipeline_id: str
    agent_id: str
    scope: str
    expires_at: float
    drift_mode: str

class SentinelRegisterRequest(BaseModel):
    repo_full_name: str
    owner_ref: str
    policy: Optional[dict[str, Any]] = None

class SentinelRegisterResponse(BaseModel):
    repo_full_name: str
    webhook_secret: str
    status: str
    plan_expires_at: str
    baseline: dict[str, Any]

class SentinelRenewRequest(BaseModel):
    repo_full_name: str
    owner_ref: str

class SentinelScanRequest(BaseModel):
    repo_full_name: str
    scan_type: Literal["update_rescan", "deep_scan", "forensic_audit"] = "update_rescan"
    payer_ref: str = "unknown"
    version: Optional[str] = None

# ════════════════════════════════════════════════════════════════════════════════
# Shared Evaluation Logic
# ════════════════════════════════════════════════════════════════════════════════

def _process_evaluation(req: EvaluateRequest, tier: str) -> EvaluateResponse:
    start = time.time()
    if not req.response or not req.response.strip():
        raise HTTPException(status_code=400, detail="response field is required")

    policy_yaml = BUILTIN_POLICIES.get(req.policy, req.policy or BUILTIN_POLICIES["default"])
    verdict, confidence, reason, policy_version = evaluate_policy(req.response, policy_yaml)
    input_hash = "0x" + sha256hex(req.response)[:16]
    policy_hash = sha256hex(policy_yaml)[:16]

    tx_hash, chain_idx = _chain.append(
        verdict=verdict, input_hash=input_hash, policy_hash=policy_hash,
        agent_id=req.agent_id, reason=reason, confidence=confidence, task_type=req.task_type,
        drift_context={"environment": "production-edge", "policy_version_hash": policy_hash},
    )

    _commit_rate.append(1.0 if verdict == "COMMIT" else 0.0)
    if len(_commit_rate) > 100:
        _commit_rate.pop(0)
    drift_mode, drift_score = get_drift_mode(_commit_rate)

    latency_ms = int((time.time() - start) * 1000)
    pipeline_id = req.pipeline_id or str(uuid.uuid4())[:8]

    error_type = None
    if verdict == "NO_COMMIT":
        if drift_mode != "NORMAL":
            error_type = "drift"
        elif confidence < 0.7:
            error_type = "low_confidence"
        else:
            error_type = "policy_violation"

    get_collector().record_decision(
        verdict=verdict, confidence=confidence, latency_ms=latency_ms,
        error_type=error_type, model_provider=req.model_provider, model_name=req.model,
        policy_path=req.policy, pipeline_id=pipeline_id, task_type=req.task_type,
        drift_score=drift_score, drift_mode=drift_mode, retry_count=req.retry_count,
        rag_source_count=req.rag_source_count, verification_steps=1,
        deterministic_trace=f"{verdict}:{policy_hash}", chain_length=chain_idx,
    )

    ts = time.time()
    seal = format_seal(tx_hash, input_hash, ts)

    return EvaluateResponse(
        verdict=verdict, confidence=confidence, reason=reason,
        tx_hash=tx_hash, chain_index=chain_idx, input_hash=input_hash,
        policy_version=policy_version, timestamp=ts,
        pipeline_id=pipeline_id, drift_mode=drift_mode, drift_score=drift_score,
        seal_text=seal["seal_text"], verify_url=seal["verify_url"],
    )

def _process_scan(req: ScanRequest, detector, policy_label: str) -> ScanResponse:
    if not req.response or not req.response.strip():
        raise HTTPException(status_code=400, detail="response field is required")

    result = detector(req.response)
    input_hash = "0x" + sha256hex(req.response)[:16]
    policy_hash = sha256hex(policy_label)[:16]
    reason = (
        "; ".join(f"{f['category']}.{f['type']}" for f in result["findings"])
        if result["findings"] else "No patterns matched"
    )

    tx_hash, chain_idx = _chain.append(
        verdict=result["verdict"], input_hash=input_hash, policy_hash=policy_hash,
        agent_id=req.agent_id, reason=reason, confidence=1.0 - result["risk_score"],
        task_type=req.task_type,
        drift_context={"environment": "production-edge", "policy_version_hash": policy_hash},
    )

    ts = time.time()
    seal = format_seal(tx_hash, input_hash, ts)

    return ScanResponse(
        verdict=result["verdict"], risk_score=result["risk_score"],
        findings=[ScanFinding(**f) for f in result["findings"]],
        detection_count=result["detection_count"],
        categories_checked=result["categories_checked"], categories_clear=result["categories_clear"],
        tx_hash=tx_hash, chain_index=chain_idx, input_hash=input_hash, timestamp=ts,
        seal_text=seal["seal_text"], verify_url=seal["verify_url"],
    )

# ════════════════════════════════════════════════════════════════════════════════
# PRE-ACTION & POST-ACTION ROUTES (with rate limits)
# ════════════════════════════════════════════════════════════════════════════════
@app.post("/evaluate/fast", response_model=EvaluateResponse)
@limiter.limit("100/minute")
@pay("$0.01")
async def evaluate_fast(request: Request, req: EvaluateRequest):
    return _process_evaluation(req, tier="fast")

@app.post("/evaluate/strict", response_model=EvaluateResponse)
@limiter.limit("30/minute")
@pay("$0.05")
async def evaluate_strict(request: Request, req: EvaluateRequest):
    return _process_evaluation(req, tier="strict")

@app.post("/evaluate/jailbreak", response_model=EvaluateResponse)
@limiter.limit("60/minute")
@pay("$0.02")
async def evaluate_jailbreak(request: Request, req: EvaluateRequest):
    req.policy = "anti_jailbreak"
    return _process_evaluation(req, tier="jailbreak")

@app.post("/evaluate/safety", response_model=EvaluateResponse)
@limiter.limit("100/minute")
@pay("$0.01")
async def evaluate_safety(request: Request, req: EvaluateRequest):
    req.policy = "safety"
    return _process_evaluation(req, tier="safety")

@app.post("/evaluate/quality", response_model=EvaluateResponse)
@limiter.limit("60/minute")
@pay("$0.03")
async def evaluate_quality(request: Request, req: EvaluateRequest):
    req.policy = "content_quality"
    return _process_evaluation(req, tier="quality")

@app.post("/evaluate/secrets", response_model=ScanResponse)
@limiter.limit("60/minute")
@pay("$0.02")
async def evaluate_secrets(request: Request, req: ScanRequest):
    return _process_scan(req, detect_secrets, "secret_leak_v1")

@app.post("/evaluate/pii", response_model=ScanResponse)
@limiter.limit("60/minute")
@pay("$0.02")
async def evaluate_pii(request: Request, req: ScanRequest):
    return _process_scan(req, detect_pii, "pii_v1")

@app.post("/evaluate/batch", response_model=dict)
@limiter.limit("30/minute")
@pay("$0.10")
async def evaluate_batch(request: Request, req: BatchEvaluateRequest):
    if len(req.items) > req.max_items:
        raise HTTPException(400, f"Batch limited to {req.max_items} items")
    results = []
    for item in req.items:
        temp_req = EvaluateRequest(
            response=item.response, policy=item.policy,
            agent_id=req.agent_id, task_type=item.task_type
        )
        results.append(_process_evaluation(temp_req, tier="batch"))
    return {"batch_id": str(uuid.uuid4())[:8], "agent_id": req.agent_id,
            "count": len(results), "results": results}

@app.post("/pipeline/start", response_model=PipelineStartResponse)
@limiter.limit("10/minute")
@pay("$0.05")
async def pipeline_start(request: Request, req: PipelineStartRequest):
    pipeline_id = f"pl_{uuid.uuid4().hex[:12]}"
    return PipelineStartResponse(
        pipeline_id=pipeline_id, agent_id=req.agent_id,
        scope=req.scope, expires_at=time.time() + req.ttl_seconds,
        drift_mode="NORMAL"
    )

@app.get("/audit/{tx_hash}")
@limiter.limit("20/minute")
@pay("$0.10")
async def audit_decode(request: Request, tx_hash: str):
    entry = _chain.get_by_tx(tx_hash)
    if not entry:
        raise HTTPException(404, "tx_hash not found in chain")
    intact, _, _ = _chain.verify()
    seal = format_seal(entry["tx_hash"], entry["input_hash"], entry["timestamp"])
    return {
        "tx_hash": entry["tx_hash"], "agent_id": entry["agent_id"],
        "verdict": entry["verdict"], "reason": entry["reason"],
        "confidence": entry["confidence"], "task_type": entry["task_type"],
        "timestamp": entry["timestamp"], "chain_index": entry["index"],
        "prev_hash": entry["prev_hash"], "chain_integrity": intact,
        "seal_text": seal["seal_text"], "verify_url": seal["verify_url"],
    }

@app.get("/audit/{tx_hash}/deep")
@limiter.limit("5/minute")
@pay("$0.50")
async def audit_decode_deep(request: Request, tx_hash: str):
    entry = _chain.get_by_tx(tx_hash)
    if not entry:
        raise HTTPException(404, "tx_hash not found in chain")
    intact, tampered_at, tamper_reason = _chain.verify()
    seal = format_seal(entry["tx_hash"], entry["input_hash"], entry["timestamp"])
    return {
        "tx_hash": entry["tx_hash"], "agent_id": entry["agent_id"],
        "verdict": entry["verdict"], "reason": entry["reason"],
        "confidence": entry["confidence"], "task_type": entry["task_type"],
        "timestamp": entry["timestamp"], "chain_index": entry["index"],
        "prev_hash": entry["prev_hash"], "chain_integrity": intact,
        "tampered_at_index": tampered_at, "tamper_reason": tamper_reason,
        "drift_context": entry["drift_context"],
        "seal_text": seal["seal_text"], "verify_url": seal["verify_url"],
    }

# ════════════════════════════════════════════════════════════════════════════════
# DCL Update Sentinel (subscription + pay-per-call monitoring)

async def _require_sentinel_subscription_payment(request: Request):
    """Verify Sentinel's $49 subscription payment and expose verified payer."""
    payment_error = await require_x402_payment(request, 49.0)
    if payment_error is not None:
        return payment_error
    payer = getattr(request.state, "payment_payer", None)
    if not payer:
        raise HTTPException(402, "Verified payment payer is unavailable")
    return None


@app.post("/sentinel/register", response_model=SentinelRegisterResponse)
@limiter.limit("10/minute")
async def sentinel_register(request: Request, req: SentinelRegisterRequest):
    if "/" not in req.repo_full_name or req.repo_full_name.count("/") != 1:
        raise HTTPException(400, "repo_full_name must be owner/repo")

    payment_error = await _require_sentinel_subscription_payment(request)
    if payment_error is not None:
        return payment_error
    payer_ref = request.state.payment_payer

    if _sentinel_db.get_skill_by_repo(req.repo_full_name):
        raise HTTPException(
            409,
            "Skill already registered — use /sentinel/renew to extend subscription",
        )

    policy = default_policy(req.policy)
    outcome = await audit_repo_release(
        req.repo_full_name,
        _chain,
        scan_type="update_rescan",
    )

    threshold = policy.get("score_threshold", 0.80)
    if outcome.verdict == "FAIL" or outcome.score < threshold:
        raise HTTPException(
            422,
            {
                "error": "initial_audit_failed",
                "message": "Initial audit does not satisfy Sentinel policy; registration refused",
                "audit": audit_to_dict(outcome),
                "score_threshold": threshold,
            },
        )

    skill = _sentinel_db.create_skill(
        repo_full_name=req.repo_full_name,
        owner_ref=req.owner_ref,
        owner_payer_ref=payer_ref,
        webhook_secret=new_webhook_secret(),
        policy=policy,
        version=outcome.version,
        verdict=outcome.verdict,
        score=outcome.score,
        audited_at=outcome.audited_at,
        payer_ref=payer_ref,
        amount_paid=49.0,
    )

    return SentinelRegisterResponse(
        repo_full_name=req.repo_full_name,
        webhook_secret=skill["webhook_secret"],
        status=skill["status"],
        plan_expires_at=skill["plan_expires_at"],
        baseline=audit_to_dict(outcome),
    )


@app.post("/sentinel/webhook/{webhook_secret}")
@limiter.limit("60/minute")
async def sentinel_webhook(request: Request, webhook_secret: str):
    try:
        check_rate_limit(get_remote_address(request))
    except ValueError as exc:
        raise HTTPException(429, str(exc)) from exc

    skill = _sentinel_db.get_skill_by_webhook_secret(webhook_secret)
    if not skill:
        raise HTTPException(404, "Unknown webhook secret")

    body = await request.body()
    if not verify_github_signature(
        body,
        webhook_secret,
        request.headers.get("X-Hub-Signature-256"),
    ):
        raise HTTPException(401, "Invalid GitHub webhook signature")

    if not _sentinel_db.subscription_active(skill):
        await _notify_subscription_lapsed(skill)
        return {"status": "ignored", "reason": "subscription_inactive"}

    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "Invalid JSON payload") from exc

    payload_repo = (payload.get("repository") or {}).get("full_name")
    if not payload_repo:
        raise HTTPException(400, "Webhook payload missing repository.full_name")
    if payload_repo != skill["repo_full_name"]:
        raise HTTPException(403, "Webhook repository does not match registered skill")

    delivery_id = request.headers.get("X-GitHub-Delivery")
    if delivery_id and _sentinel_db.event_exists(delivery_id):
        return {
            "status": "ignored",
            "reason": "duplicate_delivery",
            "delivery_id": delivery_id,
        }

    version = parse_github_release_version(payload)
    if not version:
        return {"status": "ignored", "reason": "not_a_release_event"}

    result = await apply_webhook_scan(
        _sentinel_db,
        skill,
        _chain,
        version=version,
        delivery_id=delivery_id,
    )
    return {"status": "processed", **result}


async def _notify_subscription_lapsed(skill: dict) -> None:
    from sentinel_logic import notify_owner
    await notify_owner(
        skill["owner_ref"],
        {
            "event": "subscription_lapsed",
            "repo": skill["repo_full_name"],
            "plan_expires_at": skill.get("plan_expires_at"),
            "action_required": "POST /sentinel/renew",
        },
    )


@app.post("/sentinel/scan")
@limiter.limit("30/minute")
async def sentinel_scan(request: Request):
    try:
        check_rate_limit(get_remote_address(request))
    except ValueError as exc:
        raise HTTPException(429, str(exc)) from exc

    try:
        raw = await request.json()
        req = SentinelScanRequest(**raw)
    except Exception as exc:
        raise HTTPException(400, f"Invalid request body: {exc}") from exc

    try:
        price = scan_price(req.scan_type)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    payment_error = await require_x402_payment(request, price)
    if payment_error is not None:
        return payment_error

    payer_ref = getattr(request.state, "payment_payer", None)
    if not payer_ref:
        raise HTTPException(402, "Verified payment payer is unavailable")

    outcome = await audit_repo_release(
        req.repo_full_name,
        _chain,
        scan_type=req.scan_type,
        version=req.version,
    )

    skill = _sentinel_db.get_skill_by_repo(req.repo_full_name)
    if skill:
        _sentinel_db.update_current(
            skill["id"],
            version=outcome.version,
            verdict=outcome.verdict,
            score=outcome.score,
            audited_at=outcome.audited_at,
        )
        skill_id = skill["id"]
    else:
        _sentinel_db.upsert_current_for_unregistered(
            req.repo_full_name,
            version=outcome.version,
            verdict=outcome.verdict,
            score=outcome.score,
            audited_at=outcome.audited_at,
        )
        skill_id = _sentinel_db.get_skill_by_repo(req.repo_full_name)["id"]

    amount = float(price.lstrip("$"))
    _sentinel_db.insert_event(
        skill_id=skill_id,
        event_type="rescan_paid",
        scan_type=req.scan_type,
        version=outcome.version,
        verdict=outcome.verdict,
        score=outcome.score,
        amount_paid=amount,
        payer_ref=payer_ref,
    )
    return {
        "repo_full_name": req.repo_full_name,
        "scan_type": req.scan_type,
        "amount_paid": amount,
        "audit": audit_to_dict(outcome),
    }


@app.get("/sentinel/status/{repo_full_name:path}")
@limiter.limit("120/minute")
async def sentinel_status(request: Request, repo_full_name: str):
    return _sentinel_db.status_payload(repo_full_name)


@app.get("/sentinel/prices")
def sentinel_prices():
    return {"scan_types": SCAN_PRICES, "subscription_30d": "$49"}


@app.post("/sentinel/renew")
@limiter.limit("10/minute")
async def sentinel_renew(request: Request, req: SentinelRenewRequest):
    skill = _sentinel_db.get_skill_by_repo(req.repo_full_name)
    if not skill:
        raise HTTPException(
            404, "Skill not registered — use /sentinel/register first"
        )

    payment_error = await _require_sentinel_subscription_payment(request)
    if payment_error is not None:
        return payment_error
    payer_ref = request.state.payment_payer

    registered_payer = skill.get("owner_payer_ref")
    if not registered_payer:
        raise HTTPException(
            409,
            "Legacy Sentinel registration has no verified owner payer; renewal is unavailable",
        )

    if registered_payer.lower() != payer_ref.lower():
        raise HTTPException(403, "Payment payer is not the registered skill owner")

    new_expires = _sentinel_db.renew_subscription(
        skill["id"],
        payer_ref,
        amount_paid=49.0,
    )
    refreshed = _sentinel_db.get_skill_by_repo(req.repo_full_name)

    return {
        "repo_full_name": req.repo_full_name,
        "status": refreshed["status"],
        "plan_expires_at": new_expires.isoformat(timespec="seconds"),
    }


# Utility Routes (no rate limits)
# ════════════════════════════════════════════════════════════════════════════════
@app.get("/")
def root():
    return {
        "service": "DCL Trust Oracle Webhook API (x402)",
        "version": "2.2.0",
        "by": "Fronesis Labs",
        "sentinel": "/sentinel/register",
    }

@app.get("/health")
def health():
    return {"status": "ok", "chain_length": len(_chain), "ts": time.time()}

@app.get("/policies")
def list_policies():
    return {"policies": list(BUILTIN_POLICIES.keys())}

@app.get("/chain/status")
def chain_status():
    intact, tampered_at, tamper_reason = _chain.verify()
    drift_mode, drift_score = get_drift_mode(_commit_rate)
    return {
        "chain_length": len(_chain), "integrity": intact,
        "tampered_at": tampered_at, "tamper_reason": tamper_reason,
        "drift_mode": drift_mode, "drift_score": drift_score
    }

@app.get("/chain/export")
def chain_export():
    intact, _, _ = _chain.verify()
    return {"chain": _chain.export(), "integrity": intact, "exported_at": time.time()}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    print("╔══════════════════════════════════════════════════════╗")
    print("║  DCL Trust Oracle — Webhook Server v2.2.0            ║")
    print("║  Fronesis Labs · fronesislabs.com                    ║")
    print("║  x402 Micropayments + Rate Limiting ENABLED          ║")
    print("╚══════════════════════════════════════════════════════╝")
    uvicorn.run("webhook_server:app", host="0.0.0.0", port=port, reload=False)
