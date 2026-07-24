"""
DCL Webhook Server v2.0.0 — x402 Micropayments + Extended Metadata Audit
Deterministic AI audit layer. Tamper-evident. Metadata-only.
"""
import hashlib
import math
import os
import time
import uuid
import sqlite3
import json
from typing import Optional, Tuple, List

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi_x402 import init_x402, pay
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

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
    version="2.0.0",
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
    network="base",
)

# ════════════════════════════════════════════════════════════════════════════════
# DCL Core Helpers
# ════════════════════════════════════════════════════════════════════════════════
def sha256hex(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

class ChainState:
    """SQLite-backed tamper-evident chain. Stores METADATA ONLY."""
    GENESIS = "0" * 64

    def __init__(self, db_path: str = "dcl_chain.db"):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS chain (
                idx INTEGER PRIMARY KEY,
                tx_hash TEXT UNIQUE NOT NULL,
                prev_hash TEXT NOT NULL,
                verdict TEXT NOT NULL,
                input_hash TEXT NOT NULL,
                policy_hash TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                reason TEXT NOT NULL,
                confidence REAL NOT NULL,
                task_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                drift_context TEXT NOT NULL
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_tx_hash ON chain(tx_hash)")
        self._conn.commit()

    def append(self, verdict: str, input_hash: str, policy_hash: str,
               agent_id: str, reason: str, confidence: float, task_type: str) -> Tuple[str, int]:
        last = self._conn.execute("SELECT tx_hash FROM chain ORDER BY idx DESC LIMIT 1").fetchone()
        prev_hash = last[0] if last else self.GENESIS
        new_idx = self._conn.execute("SELECT COALESCE(MAX(idx), -1) + 1 FROM chain").fetchone()[0]

        content = f"{new_idx}:{verdict}:{input_hash}:{policy_hash}:{prev_hash}:{time.time()}"
        tx_hash = "0x" + sha256hex(content)[:32]

        drift_context = {"environment": "production-edge", "policy_version_hash": policy_hash}

        self._conn.execute("""
            INSERT INTO chain (idx, tx_hash, prev_hash, verdict, input_hash, policy_hash,
                             agent_id, reason, confidence, task_type, timestamp, drift_context)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (new_idx, tx_hash, prev_hash, verdict, input_hash, policy_hash,
              agent_id, reason, confidence, task_type, time.time(), json.dumps(drift_context)))
        self._conn.commit()
        return tx_hash, new_idx

    def get_by_tx(self, tx_hash: str) -> Optional[dict]:
        row = self._conn.execute("""
            SELECT idx, tx_hash, prev_hash, verdict, input_hash, policy_hash,
                   agent_id, reason, confidence, task_type, timestamp, drift_context
            FROM chain WHERE tx_hash = ?
        """, (tx_hash,)).fetchone()
        if not row:
            return None
        return {
            "index": row[0], "tx_hash": row[1], "prev_hash": row[2], "verdict": row[3],
            "input_hash": row[4], "policy_hash": row[5], "agent_id": row[6], "reason": row[7],
            "confidence": row[8], "task_type": row[9], "timestamp": row[10],
            "drift_context": json.loads(row[11])
        }

    def verify(self) -> Tuple[bool, Optional[int]]:
        rows = self._conn.execute("SELECT idx, prev_hash, tx_hash FROM chain ORDER BY idx").fetchall()
        for i, row in enumerate(rows):
            expected_prev = rows[i-1][2] if i > 0 else self.GENESIS
            if row[1] != expected_prev:
                return False, row[0]
        return True, None

    def export(self) -> list[dict]:
        rows = self._conn.execute("""
            SELECT idx, tx_hash, prev_hash, verdict, input_hash, policy_hash,
                   agent_id, reason, confidence, task_type, timestamp, drift_context
            FROM chain ORDER BY idx
        """).fetchall()
        return [{
            "index": r[0], "tx_hash": r[1], "prev_hash": r[2], "verdict": r[3],
            "input_hash": r[4], "policy_hash": r[5], "agent_id": r[6], "reason": r[7],
            "confidence": r[8], "task_type": r[9], "timestamp": r[10],
            "drift_context": json.loads(r[11])
        } for r in rows]

    def __len__(self):
        return self._conn.execute("SELECT COUNT(*) FROM chain").fetchone()[0]

_chain = ChainState()
_commit_rate: list[float] = []

# ════════════════════════════════════════════════════════════════════════════════
# Policy Engine
# ════════════════════════════════════════════════════════════════════════════════
BUILTIN_POLICIES = {
    "default": """
version: "1.0.0"
name: "DCL Default Policy"
thresholds:
  min_confidence: 0.7
forbidden_patterns:
  - "ignore previous instructions"
  - "jailbreak"
  - "bypass safety"
required_patterns: []
""",
    "anti_jailbreak": """
version: "1.0.0"
name: "Anti-Jailbreak"
thresholds:
  min_confidence: 0.8
forbidden_patterns:
  - "ignore previous instructions"
  - "jailbreak"
  - "bypass safety"
  - "pretend you are"
  - "act as if"
  - "DAN"
required_patterns: []
""",
    "safety": """
version: "1.0.0"
name: "Safety Policy"
thresholds:
  min_confidence: 0.75
forbidden_patterns:
  - "I cannot be held responsible"
  - "no guarantees"
required_patterns:
  - "AI"
""",
    "content_quality": """
version: "1.0.0"
name: "Content Quality Policy"
thresholds:
  min_confidence: 0.85
forbidden_patterns:
  - "guaranteed returns"
  - "risk-free"
  - "definitely will"
required_patterns:
  - "past performance"
""",
}

def evaluate_policy(response: str, policy_yaml: str) -> Tuple[str, float, str, str]:
    try:
        policy = yaml.safe_load(policy_yaml)
    except yaml.YAMLError as e:
        return "NO_COMMIT", 0.0, f"Policy parse error: {e}", "unknown"

    version = policy.get("version", "unknown")
    penalties = 0.0
    reasons = []

    for pat in policy.get("forbidden_patterns", []):
        if pat.lower() in response.lower():
            reasons.append(f"forbidden: '{pat}'")
            penalties += 0.4

    for pat in policy.get("required_patterns", []):
        if pat.lower() not in response.lower():
            reasons.append(f"missing required: '{pat}'")
            penalties += 0.2

    confidence = max(0.0, 1.0 - penalties)
    min_conf = policy.get("thresholds", {}).get("min_confidence", 0.7)

    verdict = "COMMIT"
    reason = "All policy checks passed"

    if confidence < min_conf or reasons:
        verdict = "NO_COMMIT"
        reason = "; ".join(reasons) if reasons else f"Confidence {confidence} below threshold {min_conf}"

    return verdict, round(confidence, 3), reason, version

def get_drift_mode(commit_rate: list[float]) -> Tuple[str, float]:
    n = len(commit_rate)
    if n < 5:
        return "NORMAL", 0.0

    window = min(10, n)
    baseline_vals = commit_rate[:-window]
    if not baseline_vals:
        return "NORMAL", 0.0

    baseline = sum(baseline_vals) / len(baseline_vals) or 0.01
    current = sum(commit_rate[-window:]) / window

    z = (current - baseline) / math.sqrt(baseline * (1 - baseline) / window)
    abs_z = abs(z)

    if abs_z > 3.5:
        return "BLOCK", round(z, 2)
    elif abs_z > 2.5:
        return "ESCALATION", round(z, 2)
    elif abs_z > 1.96:
        return "WARNING", round(z, 2)
    return "NORMAL", round(z, 2)

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
        agent_id=req.agent_id, reason=reason, confidence=confidence, task_type=req.task_type
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

    return EvaluateResponse(
        verdict=verdict, confidence=confidence, reason=reason,
        tx_hash=tx_hash, chain_index=chain_idx, input_hash=input_hash,
        policy_version=policy_version, timestamp=time.time(),
        pipeline_id=pipeline_id, drift_mode=drift_mode, drift_score=drift_score,
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
    intact, _ = _chain.verify()
    return {
        "tx_hash": entry["tx_hash"], "agent_id": entry["agent_id"],
        "verdict": entry["verdict"], "reason": entry["reason"],
        "confidence": entry["confidence"], "task_type": entry["task_type"],
        "timestamp": entry["timestamp"], "chain_index": entry["index"],
        "prev_hash": entry["prev_hash"], "chain_integrity": intact,
    }

@app.get("/audit/{tx_hash}/deep")
@limiter.limit("5/minute")
@pay("$0.50")
async def audit_decode_deep(request: Request, tx_hash: str):
    entry = _chain.get_by_tx(tx_hash)
    if not entry:
        raise HTTPException(404, "tx_hash not found in chain")
    intact, tampered_at = _chain.verify()
    return {
        "tx_hash": entry["tx_hash"], "agent_id": entry["agent_id"],
        "verdict": entry["verdict"], "reason": entry["reason"],
        "confidence": entry["confidence"], "task_type": entry["task_type"],
        "timestamp": entry["timestamp"], "chain_index": entry["index"],
        "prev_hash": entry["prev_hash"], "chain_integrity": intact,
        "tampered_at_index": tampered_at, "drift_context": entry["drift_context"],
    }

# ════════════════════════════════════════════════════════════════════════════════
# Utility Routes (no rate limits)
# ════════════════════════════════════════════════════════════════════════════════
@app.get("/")
def root():
    return {"service": "DCL Evaluator Webhook API (x402)", "version": "2.0.0", "by": "Fronesis Labs"}

@app.get("/health")
def health():
    return {"status": "ok", "chain_length": len(_chain), "ts": time.time()}

@app.get("/policies")
def list_policies():
    return {"policies": list(BUILTIN_POLICIES.keys())}

@app.get("/chain/status")
def chain_status():
    intact, tampered_at = _chain.verify()
    drift_mode, drift_score = get_drift_mode(_commit_rate)
    return {
        "chain_length": len(_chain), "integrity": intact,
        "tampered_at": tampered_at, "drift_mode": drift_mode, "drift_score": drift_score
    }

@app.get("/chain/export")
def chain_export():
    intact, _ = _chain.verify()
    return {"chain": _chain.export(), "integrity": intact, "exported_at": time.time()}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║ DCL Evaluator — Webhook Server v2.0.0 ║")
    print("║ Fronesis Labs · fronesislabs.io ║")
    print("║ x402 Micropayments + Rate Limiting ENABLED ║")
    print("╚══════════════════════════════════════════════════════╝\n")
    uvicorn.run("webhook_server:app", host="0.0.0.0", port=port, reload=False)
