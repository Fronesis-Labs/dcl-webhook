"""
DCL Webhook Server — plug-and-play audit endpoint for any AI pipeline.

Любой pipeline отправляет POST /evaluate → получает COMMIT/NO_COMMIT + tx_hash.
Все решения записываются в tamper-evident chain + anonymized telemetry.

Интеграция в 3 строки:
    import httpx
    result = httpx.post("https://your-dcl/evaluate", json={
        "response": llm_output, "policy": "default"
    }).json()
    if result["verdict"] == "NO_COMMIT":
        raise PolicyViolationError(result)

Deploy: Railway / Render / VPS
    pip install fastapi uvicorn pyyaml
    python webhook_server.py
"""

import hashlib
import math
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Optional

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from telemetry import get_collector

# ════════════════════════════════════════════════════════════════════════════════
# App
# ════════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="DCL Evaluator — Webhook API",
    description="Deterministic AI audit layer. Tamper-evident. Privacy-first.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ════════════════════════════════════════════════════════════════════════════════
# DCL Core (inline — не зависит от Go десктопа)
# ════════════════════════════════════════════════════════════════════════════════

def sha256hex(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


@dataclass
class EvalResult:
    verdict: str          # COMMIT | NO_COMMIT
    confidence: float
    reason: str
    tx_hash: str          # tamper-evident proof
    chain_index: int
    input_hash: str       # SHA-256 входных данных (не сам контент)
    policy_version: str
    timestamp: float
    pipeline_id: str


class ChainState:
    """In-memory tamper-evident chain. Persist to DB для production."""

    GENESIS = "0" * 64

    def __init__(self):
        self._entries: list[dict] = []

    def append(self, verdict: str, input_hash: str, policy_hash: str) -> tuple[str, int]:
        prev_hash = self._entries[-1]["tx_hash"] if self._entries else self.GENESIS
        idx = len(self._entries)
        content = f"{idx}:{verdict}:{input_hash}:{policy_hash}:{prev_hash}:{time.time()}"
        tx_hash = "0x" + sha256hex(content)[:32]
        self._entries.append({
            "index": idx,
            "tx_hash": tx_hash,
            "prev_hash": prev_hash,
            "verdict": verdict,
            "input_hash": input_hash,
            "timestamp": time.time(),
        })
        return tx_hash, idx

    def verify(self) -> tuple[bool, Optional[int]]:
        for i, entry in enumerate(self._entries):
            expected_prev = self._entries[i-1]["tx_hash"] if i > 0 else self.GENESIS
            if entry["prev_hash"] != expected_prev:
                return False, i
        return True, None

    def export(self) -> list[dict]:
        return list(self._entries)

    def __len__(self):
        return len(self._entries)


# Global state (для production — заменить на Redis/Postgres)
_chain = ChainState()
_commit_rate: list[float] = []


# ════════════════════════════════════════════════════════════════════════════════
# Policy engine
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
    "default":      DEFAULT_POLICY,
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


def evaluate_policy(response: str, policy_yaml: str) -> tuple[str, float, str, str]:
    """
    Детерминированный evaluation.
    Возвращает: (verdict, confidence, reason, policy_version)
    """
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

    if confidence < min_conf:
        verdict = "NO_COMMIT"

    if reasons:
        reason = "; ".join(reasons)
        verdict = "NO_COMMIT"

    return verdict, round(confidence, 3), reason, version


def get_drift_mode(commit_rate: list[float]) -> tuple[str, float]:
    """Z-test drift detection — та же логика что в app.go."""
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
# Request / Response models
# ════════════════════════════════════════════════════════════════════════════════

class EvaluateRequest(BaseModel):
    response: str                        # LLM output для evaluation
    policy: Optional[str] = "default"    # имя builtin policy ИЛИ YAML строка
    model: Optional[str] = "unknown"     # название модели (для telemetry)
    model_provider: Optional[str] = "unknown"
    pipeline_id: Optional[str] = ""      # ID вашего pipeline (для группировки)
    task_type: Optional[str] = "unknown" # classification|reasoning|summarization|etc
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
# Routes
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "service": "DCL Evaluator Webhook API",
        "version": "1.1.0",
        "by": "Fronesis Labs — fronesislabs.io",
        "endpoints": {
            "POST /evaluate":      "Evaluate LLM output against policy",
            "GET  /chain/status":  "Chain integrity + drift status",
            "GET  /chain/export":  "Export full audit trail",
            "GET  /policies":      "List available builtin policies",
            "GET  /health":        "Health check",
        },
        "demo": "POST /evaluate with {response: '...', policy: 'default'}",
    }


@app.get("/health")
def health():
    return {"status": "ok", "chain_length": len(_chain), "ts": time.time()}


@app.get("/policies")
def list_policies():
    return {"policies": list(BUILTIN_POLICIES.keys())}


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(req: EvaluateRequest):
    """
    Evaluate LLM output against a DCL policy.

    Returns COMMIT or NO_COMMIT with tamper-evident tx_hash.
    Every decision is recorded in the cryptographic chain.
    """
    start = time.time()

    if not req.response or not req.response.strip():
        raise HTTPException(status_code=400, detail="response field is required")

    # Resolve policy
    if req.policy in BUILTIN_POLICIES:
        policy_yaml = BUILTIN_POLICIES[req.policy]
    else:
        # Попытка распарсить как inline YAML
        policy_yaml = req.policy or BUILTIN_POLICIES["default"]

    # Evaluate
    verdict, confidence, reason, policy_version = evaluate_policy(
        req.response, policy_yaml
    )

    # Chain entry — хэшируем input, не храним контент
    input_hash = "0x" + sha256hex(req.response)[:16]
    policy_hash = sha256hex(policy_yaml)[:16]
    tx_hash, chain_idx = _chain.append(verdict, input_hash, policy_hash)

    # Drift
    _commit_rate.append(1.0 if verdict == "COMMIT" else 0.0)
    if len(_commit_rate) > 100:
        _commit_rate.pop(0)
    drift_mode, drift_score = get_drift_mode(_commit_rate)

    latency_ms = int((time.time() - start) * 1000)
    pipeline_id = req.pipeline_id or str(uuid.uuid4())[:8]

    # Telemetry — content-agnostic
    error_type = None
    if verdict == "NO_COMMIT":
        if drift_mode != "NORMAL":
            error_type = "drift"
        elif confidence < 0.7:
            error_type = "low_confidence"
        else:
            error_type = "policy_violation"

    get_collector().record_decision(
        verdict=verdict,
        confidence=confidence,
        latency_ms=latency_ms,
        error_type=error_type,
        model_provider=req.model_provider,
        model_name=req.model,
        policy_path=req.policy,
        pipeline_id=pipeline_id,
        task_type=req.task_type,
        drift_score=drift_score,
        drift_mode=drift_mode,
        retry_count=req.retry_count,
        rag_source_count=req.rag_source_count,
        verification_steps=1,
        deterministic_trace=f"{verdict}:{policy_hash}",
        chain_length=chain_idx,
    )

    return EvaluateResponse(
        verdict=verdict,
        confidence=confidence,
        reason=reason,
        tx_hash=tx_hash,
        chain_index=chain_idx,
        input_hash=input_hash,
        policy_version=policy_version,
        timestamp=time.time(),
        pipeline_id=pipeline_id,
        drift_mode=drift_mode,
        drift_score=drift_score,
    )


@app.get("/chain/status", response_model=ChainStatusResponse)
def chain_status():
    """Chain integrity check + current drift status."""
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
    """
    Export full audit trail.
    Содержит только хэши — никакого контента.
    """
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
║       DCL Evaluator — Webhook Server v1.1.0          ║
║       Fronesis Labs · fronesislabs.io                ║
╠══════════════════════════════════════════════════════╣
║  POST http://localhost:{port}/evaluate               ║
║  GET  http://localhost:{port}/chain/status           ║
║  GET  http://localhost:{port}/docs  (Swagger UI)     ║
╚══════════════════════════════════════════════════════╝
    """)
    uvicorn.run("webhook_server:app", host="0.0.0.0", port=port, reload=False)
