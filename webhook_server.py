"""
DCL Webhook Server v2.0.0 — x402 Micropayments + Extended Metadata Audit
Deterministic AI audit layer. Tamper-evident. Privacy-first.

Deploy: Railway / Render / VPS
pip install fastapi uvicorn pyyaml fastapi-x402
python webhook_server.py
"""
import hashlib
import math
import os
import time
import uuid
from typing import Optional, Tuple
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi_x402 import init_x402, pay

# Импортируем ваш коллектор телеметрии (должен быть в той же папке)
from telemetry import get_collector

# ════════════════════════════════════════════════════════════════════════════════
# App & x402 Initialization
# ════════════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="DCL Evaluator — Webhook API (x402)",
    description="Deterministic AI audit layer with micropayments. Tamper-evident. Privacy-first.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Инициализация x402 шлюза
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
    """In-memory tamper-evident chain. Stores METADATA ONLY (Privacy-First)."""
    GENESIS = "0" * 64

    def __init__(self):
        self._entries: list[dict] = []

    def append(self, verdict: str, input_hash: str, policy_hash: str, 
               agent_id: str, reason: str, confidence: float, task_type: str) -> Tuple[str, int]:
        prev_hash = self._entries[-1]["tx_hash"] if self._entries else self.GENESIS
        idx = len(self._entries)
        
        # Хэшируем только метаданные, сырой текст НЕ попадает в цепочку
        content = f"{idx}:{verdict}:{input_hash}:{policy_hash}:{prev_hash}:{time.time()}"
        tx_hash = "0x" + sha256hex(content)[:32]
        
        self._entries.append({
            "index": idx,
            "tx_hash": tx_hash,
            "prev_hash": prev_hash,
            "verdict": verdict,
            "input_hash": input_hash,
            "policy_hash": policy_hash,
            "agent_id": agent_id,
            "reason": reason,
            "confidence": confidence,
            "task_type": task_type,
            "timestamp": time.time(),
            # Расширенный контекст для Deep-аудита ($0.50)
            "drift_context": {
                "environment": "production-edge",
                "policy_version_hash": policy_hash
            }
        })
        return tx_hash, idx

    def get_by_tx(self, tx_hash: str) -> Optional[dict]:
        return next((e for e in self._entries if e["tx_hash"] == tx_hash), None)

    def verify(self) -> Tuple[bool, Optional[int]]:
        for i, entry in enumerate(self._entries):
            expected_prev = self._entries[i-1]["tx_hash"] if i > 0 else self.GENESIS
            if entry["prev_hash"] != expected_prev:
                return False, i
        return True, None

    def export(self) -> list[dict]:
        return list(self._entries)

    def __len__(self):
        return len(self._entries)

_chain = ChainState()
_commit_rate: list[float] = []

# ════════════════════════════════════════════════════════════════════════════════
# Policy Engine (из вашего исходного webhook_server.py)
# ════════════════════════════════════════════════════════════════════════════════
DEFAULT_POLICY = """
version: "1.0.0"
name: "DCL Default Policy"
thresholds:
  min_confidence: 0.7
forbidden_patterns:
  - "ignore previous instructions"
  - "jailbreak"
  - "bypass safety"
required_patterns: []
"""

BUILTIN_POLICIES = {
    "default": DEFAULT_POLICY,
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
    "eu_ai_act": """
version: "1.0.0"
name: "EU AI Act Compliance"
thresholds:
  min_confidence: 0.75
forbidden_patterns:
  - "I cannot be held responsible"
  - "no guarantees"
required_patterns:
  - "AI"
""",
    "finance": """
version: "1.0.0"
name: "Finance Policy"
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

class ChainStatusResponse(BaseModel):
    chain_length: int
    integrity: bool
    tampered_at: Optional[int]
    drift_mode: str
    drift_score: float

# ════════════════════════════════════════════════════════════════════════════════
# Shared Evaluation Logic (чтобы не дублировать код между fast и strict)
# ════════════════════════════════════════════════════════════════════════════════
def _process_evaluation(req: EvaluateRequest, tier: str) -> EvaluateResponse:
    start = time.time()
    if not req.response or not req.response.strip():
        raise HTTPException(status_code=400, detail="response field is required")
    
    policy_yaml = BUILTIN_POLICIES.get(req.policy, req.policy or DEFAULT_POLICY)
    
    verdict, confidence, reason, policy_version = evaluate_policy(req.response, policy_yaml)
    
    input_hash = "0x" + sha256hex(req.response)[:16]
    policy_hash = sha256hex(policy_yaml)[:16]
    
    # Расширенное добавление в цепочку с метаданными (без сырого текста!)
    tx_hash, chain_idx = _chain.append(
        verdict=verdict,
        input_hash=input_hash,
        policy_hash=policy_hash,
        agent_id=req.agent_id,
        reason=reason,
        confidence=confidence,
        task_type=req.task_type
    )
    
    # Drift & Telemetry
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
# PRE-ACTION ROUTES (Fixed Pricing)
# ════════════════════════════════════════════════════════════════════════════════
@app.post("/evaluate/fast", response_model=EvaluateResponse)
@pay("$0.01")
async def evaluate_fast(req: EvaluateRequest):
    return _process_evaluation(req, tier="fast")

@app.post("/evaluate/strict", response_model=EvaluateResponse)
@pay("$0.05")
async def evaluate_strict(req: EvaluateRequest):
    return _process_evaluation(req, tier="strict")

# ════════════════════════════════════════════════════════════════════════════════
# POST-ACTION ROUTES (Fixed Pricing)
# ════════════════════════════════════════════════════════════════════════════════
@app.get("/audit/{tx_hash}")
@pay("$0.10")
async def audit_decode(tx_hash: str):
    entry = _chain.get_by_tx(tx_hash)
    if not entry:
        raise HTTPException(404, "tx_hash not found in chain")
    
    intact, _ = _chain.verify()
    return {
        "tx_hash": entry["tx_hash"],
        "agent_id": entry["agent_id"],
        "verdict": entry["verdict"],
        "reason": entry["reason"],
        "confidence": entry["confidence"],
        "task_type": entry["task_type"],
        "timestamp": entry["timestamp"],
        "chain_index": entry["index"],
        "prev_hash": entry["prev_hash"],
        "chain_integrity": intact,
    }

@app.get("/audit/{tx_hash}/deep")
@pay("$0.50")
async def audit_decode_deep(tx_hash: str):
    entry = _chain.get_by_tx(tx_hash)
    if not entry:
        raise HTTPException(404, "tx_hash not found in chain")
    
    intact, tampered_at = _chain.verify()
    return {
        "tx_hash": entry["tx_hash"],
        "agent_id": entry["agent_id"],
        "verdict": entry["verdict"],
        "reason": entry["reason"],
        "confidence": entry["confidence"],
        "task_type": entry["task_type"],
        "timestamp": entry["timestamp"],
        "chain_index": entry["index"],
        "prev_hash": entry["prev_hash"],
        "chain_integrity": intact,
        "tampered_at_index": tampered_at,
        "drift_context": entry["drift_context"],  # Платная ценность за $0.50
    }

# ════════════════════════════════════════════════════════════════════════════════
# Utility Routes (сохранены из оригинала)
# ════════════════════════════════════════════════════════════════════════════════
@app.get("/")
def root():
    return {
        "service": "DCL Evaluator Webhook API (x402)",
        "version": "2.0.0",
        "by": "Fronesis Labs — fronesislabs.io",
        "endpoints": {
            "POST /evaluate/fast": "Evaluate LLM output (Fast tier, $0.01)",
            "POST /evaluate/strict": "Evaluate LLM output (Strict tier, $0.05)",
            "GET /audit/{tx_hash}": "Basic audit decode ($0.10)",
            "GET /audit/{tx_hash}/deep": "Deep forensic audit ($0.50)",
            "GET /chain/status": "Chain integrity + drift status",
            "GET /chain/export": "Export full audit trail",
            "GET /policies": "List available builtin policies",
            "GET /health": "Health check",
        },
        "demo": "POST /evaluate/fast with {response: '...', policy: 'default', agent_id: 'bot_1'}",
    }

@app.get("/health")
def health():
    return {"status": "ok", "chain_length": len(_chain), "ts": time.time()}

@app.get("/policies")
def list_policies():
    return {"policies": list(BUILTIN_POLICIES.keys())}

@app.get("/chain/status", response_model=ChainStatusResponse)
def chain_status():
    intact, tampered_at = _chain.verify()
    drift_mode, drift_score = get_drift_mode(_commit_rate)
    return ChainStatusResponse(
        chain_length=len(_chain),
        integrity=intact,
        tampered_at=tampered_at,
        drift_mode=drift_mode,
        drift_score=drift_score,
    )

@app.get("/chain/export")
def chain_export():
    intact, _ = _chain.verify()
    return {
        "chain": _chain.export(),
        "integrity": intact,
        "exported_at": time.time(),
        "by": "DCL Evaluator — Fronesis Labs",
    }

# ════════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    print(f"""
╔══════════════════════════════════════════════════════╗
║       DCL Evaluator — Webhook Server v2.0.0          ║
║       Fronesis Labs · fronesislabs.io                ║
║       x402 Micropayments ENABLED                     ║
╠══════════════════════════════════════════════════════╣
║  POST http://localhost:{port}/evaluate/fast        ║
║  POST http://localhost:{port}/evaluate/strict      ║
║  GET  http://localhost:{port}/audit/{{tx_hash}}      ║
║  GET  http://localhost:{port}/docs  (Swagger UI)     ║
╚══════════════════════════════════════════════════════╝
    """)
    uvicorn.run("webhook_server:app", host="0.0.0.0", port=port, reload=False)