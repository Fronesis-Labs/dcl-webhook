"""
DCL Webhook Server v2.2.0 — x402 Micropayments + Extended Metadata Audit

Deterministic AI audit layer. Tamper-evident. Metadata-only.

Chain/consensus protocol logic now comes from the published dcl-core
package (pip install dcl-core). Policy evaluation, drift detection, and
secret/PII scanning stay in audit_logic.py (closed). This file remains
transport/payment plumbing only.
"""

import os
import time
import uuid
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
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
    title="DCL Evaluator — Webhook API (x402)",
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
# Utility Routes (no rate limits)
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"service": "DCL Evaluator Webhook API (x402)", "version": "2.2.0", "by": "Fronesis Labs"}

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
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  DCL Evaluator — Webhook Server v2.2.0                ║")
    print("║  Fronesis Labs · fronesislabs.io                       ║")
    print("║  x402 Micropayments + Rate Limiting ENABLED             ║")
    print("╚══════════════════════════════════════════════════════╝\n")
    uvicorn.run("webhook_server:app", host="0.0.0.0", port=port, reload=False)
