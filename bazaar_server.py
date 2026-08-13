"""
DCL Evaluator — Bazaar/x402 v2 Server
Parallel service alongside webhook_server.py (v1). Does NOT replace it.

Uses the official `x402` package (v2 protocol + Bazaar discovery extension)
instead of `fastapi_x402` (v1-only). Reuses the shared DCL logic in
dcl_core.py — same policies, same tamper-evident chain implementation —
so verdicts are consistent with the rest of the stack.

Install first:
    venv/bin/pip install "x402[fastapi,extensions]"

Required env vars (put in .env, loaded via python-dotenv):
    X402_WALLET — your payout wallet (same one used elsewhere)
    CDP credentials (required for CDP Bazaar indexing), any of:
      - CDP_API_KEY_JSON=/path/to/cdp_api_key.json  (recommended; downloaded from CDP Portal)
      - CDP_API_KEY_ID + CDP_API_KEY_SECRET         (Python cdp-sdk names)
      - CDP_KEY_ID + CDP_KEY_SECRET                  (alias for other tooling)

Uses the CDP facilitator (https://api.cdp.coinbase.com/platform/v2/x402) so verify+settle
transactions are cataloged in the CDP Bazaar. Falls back to X402_FACILITATOR_URL if CDP keys
are absent (e.g. local dev).

Run (does not touch dcl-evaluator/dcl-webhook — separate port, separate service):
    PORT=5000 python3 bazaar_server.py
"""
import os
import json
import time
import uuid
from typing import Optional

from dotenv import load_dotenv
load_dotenv(override=True)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from x402 import x402ResourceServer
from x402.http import FacilitatorConfig, HTTPFacilitatorClient
from x402.http.middleware.fastapi import PaymentMiddlewareASGI
from x402.http.types import PaymentOption, RouteConfig
from x402.mechanisms.evm.exact import register_exact_evm_server
from x402.extensions.bazaar import declare_discovery_extension, bazaar_resource_server_extension, OutputConfig

from dcl_core import ChainState, sha256hex
from audit_logic import BUILTIN_POLICIES, evaluate_policy, get_drift_mode

# ════════════════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════════════════
X402_WALLET = os.environ.get("X402_WALLET", "0x0000000000000000000000000000000000000000")
X402_NETWORK = os.environ.get("X402_NETWORK", "eip155:8453")  # Base mainnet, CAIP-2 format
CDP_FACILITATOR_URL = "https://api.cdp.coinbase.com/platform/v2/x402"
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "https://bazaar.fronesislabs.com")
USDC_BASE = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
PAYAI_FACILITATOR_URL = "https://facilitator.payai.network"

app = FastAPI(
    title="DCL Evaluator — Bazaar API (x402 v2)",
    description="Deterministic AI audit layer with x402 v2 micropayments, Bazaar-discoverable.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

_chain = ChainState(os.environ.get("DCL_DB_PATH", "dcl_chain_bazaar.db"))
_commit_rate: list = []

# ════════════════════════════════════════════════════════════════════════════════
# x402 v2 resource server setup
# ════════════════════════════════════════════════════════════════════════════════
def _load_cdp_credentials() -> tuple[str | None, str | None, str]:
    """Load CDP API key id/secret from JSON file or env (supports common alias names)."""
    json_path = os.environ.get("CDP_API_KEY_JSON")
    if not json_path:
        for candidate in ("cdp_api_key.json", "CDP_API_KEY.json"):
            if os.path.isfile(candidate):
                json_path = candidate
                break
    if json_path and os.path.isfile(json_path):
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)
        key_id = data.get("id") or data.get("name") or data.get("apiKeyId")
        secret = data.get("privateKey") or data.get("private_key") or data.get("secret")
        if key_id and secret:
            return str(key_id), str(secret), f"json:{json_path}"

    key_id = (
        os.environ.get("CDP_API_KEY_ID")
        or os.environ.get("CDP_KEY_ID")
        or os.environ.get("CDP_API_KEY_NAME")
    )
    secret = os.environ.get("CDP_API_KEY_SECRET") or os.environ.get("CDP_KEY_SECRET")
    if key_id and secret:
        return key_id, secret, "env"
    return None, None, "none"


def _describe_cdp_secret(secret: str) -> str:
    trimmed = secret.strip()
    if trimmed.startswith("-----BEGIN"):
        return "PEM private key"
    if len(trimmed) <= 120:
        return f"short secret ({len(trimmed)} chars; Ed25519/base64 or truncated PEM)"
    return f"secret ({len(trimmed)} chars)"


def _build_facilitator() -> HTTPFacilitatorClient:
    cdp_key_id, cdp_key_secret, source = _load_cdp_credentials()
    if cdp_key_id and cdp_key_secret:
        from cdp.x402 import create_facilitator_config

        print(
            f"CDP credentials loaded from {source}; "
            f"key_id={cdp_key_id[:8]}…; {_describe_cdp_secret(cdp_key_secret)}"
        )
        client = HTTPFacilitatorClient(create_facilitator_config(cdp_key_id, cdp_key_secret))
        try:
            client.get_supported()
            print(f"Using CDP facilitator at {CDP_FACILITATOR_URL}")
            return client
        except Exception as exc:
            print(
                f"WARNING: CDP facilitator auth failed ({exc}). "
                "Regenerate the key in CDP Portal (download fresh JSON) and restart. "
                "Falling back to PayAI — transactions will NOT appear in CDP Bazaar."
            )
    override_url = os.environ.get("X402_FACILITATOR_URL")
    fallback_url = override_url or PAYAI_FACILITATOR_URL
    if not (cdp_key_id and cdp_key_secret):
        print(
            f"WARNING: CDP_API_KEY_ID/SECRET not set — using {fallback_url}. "
            "Set CDP keys for CDP Bazaar indexing."
        )
    else:
        print(f"Using fallback facilitator at {fallback_url}")
    return HTTPFacilitatorClient(FacilitatorConfig(url=fallback_url))


facilitator = _build_facilitator()
server = x402ResourceServer(facilitator)
register_exact_evm_server(server, networks=X402_NETWORK)
server.register_extension(bazaar_resource_server_extension)

# Shared example schema for all /evaluate/* routes (they all take the same body shape)
_EVALUATE_INPUT_SCHEMA = {
    "properties": {
        "response": {"type": "string", "description": "Agent/LLM response text to audit"},
        "agent_id": {"type": "string", "description": "Identifier of the agent"},
    },
    "required": ["response", "agent_id"],
}
_EVALUATE_OUTPUT_EXAMPLE = {
    "verdict": "COMMIT",
    "confidence": 0.95,
    "reason": "All policy checks passed",
    "tx_hash": "0xabc123...",
    "chain_index": 42,
}
_EVALUATE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
        "tx_hash": {"type": "string"},
        "chain_index": {"type": "integer"},
        "input_hash": {"type": "string"},
        "policy_version": {"type": "string"},
        "timestamp": {"type": "number"},
        "drift_mode": {"type": "string"},
        "drift_score": {"type": "number"},
    },
    "required": ["verdict", "confidence", "reason", "tx_hash", "chain_index"],
}


def _route_config(path: str, price: str, description: str) -> RouteConfig:
    extension = declare_discovery_extension(
        input={"response": "example agent output", "agent_id": "agent-123"},
        input_schema=_EVALUATE_INPUT_SCHEMA,
        body_type="json",
        output=OutputConfig(example=_EVALUATE_OUTPUT_EXAMPLE, schema=_EVALUATE_OUTPUT_SCHEMA),
    )
    # Satisfy startup schema validation; bazaar_resource_server_extension also sets
    # method from the route key / request at runtime.
    extension["bazaar"]["info"]["input"]["method"] = "POST"
    return RouteConfig(
        accepts=PaymentOption(
            scheme="exact",
            pay_to=X402_WALLET,
            price=price,
            network=X402_NETWORK,
            max_timeout_seconds=300,
        ),
        resource=f"{PUBLIC_BASE_URL}{path}",
        description=description,
        mime_type="application/json",
        service_name="DCL Evaluator",
        tags=["ai-safety", "audit", "compliance"],
        extensions=extension,
    )


routes = {
    "POST /evaluate/fast": _route_config(
        "/evaluate/fast", "$0.01", "Fast pre-action policy audit of an agent response."
    ),
    "POST /evaluate/strict": _route_config(
        "/evaluate/strict", "$0.05", "Strict pre-action audit with a higher confidence bar."
    ),
    "POST /evaluate/jailbreak": _route_config(
        "/evaluate/jailbreak", "$0.02", "Jailbreak / instruction-adherence detection."
    ),
    "POST /evaluate/safety": _route_config(
        "/evaluate/safety", "$0.01", "Baseline safety policy check."
    ),
    "POST /evaluate/quality": _route_config(
        "/evaluate/quality", "$0.03", "Content quality and drift check."
    ),
}

# Payment middleware must run before route handlers (and before any auth middleware).
app.add_middleware(PaymentMiddlewareASGI, routes=routes, server=server)


@app.get("/.well-known/402index-verify.txt")
def index_402_verify():
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(os.environ.get("INDEX_402_VERIFY_HASH", ""))


@app.get("/.well-known/x402.json")
@app.get("/.well-known/x402")
def x402_manifest():
    """Facilitator-agnostic discovery manifest — crawled by x402scan and similar
    ecosystem-wide explorers, independent of any one facilitator/CDP account."""
    return {
        "x402Version": 2,
        "provider": {
            "name": "Fronesis Labs",
            "url": "https://fronesislabs.com",
        },
        "resources": [
            {
                "resource": {
                    "url": f"{PUBLIC_BASE_URL}{path}",
                    "description": cfg.description,
                    "mimeType": "application/json",
                    "method": "POST",
                },
                "accepts": [{
                    "scheme": "exact",
                    "network": X402_NETWORK,
                    "asset": USDC_BASE,
                    "amount": str(int(float(cfg.accepts.price.replace("$", "")) * 1_000_000)),
                    "payTo": X402_WALLET,
                    "maxTimeoutSeconds": 300,
                }],
            }
            for path_key, cfg in routes.items()
            for path in [path_key.split(" ", 1)[-1]]
        ],
    }


# ════════════════════════════════════════════════════════════════════════════════
# Request / Response models (mirrors webhook_server.py)
# ════════════════════════════════════════════════════════════════════════════════
class EvaluateRequest(BaseModel):
    response: str
    policy: Optional[str] = "default"
    agent_id: Optional[str] = "unknown"
    task_type: Optional[str] = "unknown"


class EvaluateResponse(BaseModel):
    verdict: str
    confidence: float
    reason: str
    tx_hash: str
    chain_index: int
    input_hash: str
    policy_version: str
    timestamp: float
    drift_mode: str
    drift_score: float


def _process_evaluation(req: EvaluateRequest, policy_name: str, task_type: str) -> EvaluateResponse:
    if not req.response or not req.response.strip():
        raise HTTPException(status_code=400, detail="response field is required")

    policy_yaml = BUILTIN_POLICIES.get(policy_name, BUILTIN_POLICIES["default"])
    verdict, confidence, reason, policy_version = evaluate_policy(req.response, policy_yaml)

    input_hash = "0x" + sha256hex(req.response)[:16]
    policy_hash = sha256hex(policy_yaml)[:16]

    tx_hash, chain_idx = _chain.append(
        verdict=verdict, input_hash=input_hash, policy_hash=policy_hash,
        agent_id=req.agent_id, reason=reason, confidence=confidence, task_type=task_type,
    )

    _commit_rate.append(1.0 if verdict == "COMMIT" else 0.0)
    if len(_commit_rate) > 100:
        _commit_rate.pop(0)
    drift_mode, drift_score = get_drift_mode(_commit_rate)

    return EvaluateResponse(
        verdict=verdict, confidence=confidence, reason=reason,
        tx_hash=tx_hash, chain_index=chain_idx, input_hash=input_hash,
        policy_version=policy_version, timestamp=time.time(),
        drift_mode=drift_mode, drift_score=drift_score,
    )


# ════════════════════════════════════════════════════════════════════════════════
# Routes (payment enforced by the middleware above, based on `routes` config)
# ════════════════════════════════════════════════════════════════════════════════
@app.post("/evaluate/fast", response_model=EvaluateResponse)
async def evaluate_fast(req: EvaluateRequest):
    return _process_evaluation(req, "default", "fast")


@app.post("/evaluate/strict", response_model=EvaluateResponse)
async def evaluate_strict(req: EvaluateRequest):
    return _process_evaluation(req, "default", "strict")


@app.post("/evaluate/jailbreak", response_model=EvaluateResponse)
async def evaluate_jailbreak(req: EvaluateRequest):
    return _process_evaluation(req, "anti_jailbreak", "jailbreak")


@app.post("/evaluate/safety", response_model=EvaluateResponse)
async def evaluate_safety(req: EvaluateRequest):
    return _process_evaluation(req, "safety", "safety")


@app.post("/evaluate/quality", response_model=EvaluateResponse)
async def evaluate_quality(req: EvaluateRequest):
    return _process_evaluation(req, "content_quality", "quality")


@app.get("/health")
def health():
    return {"status": "ok", "service": "DCL Evaluator Bazaar (x402 v2)", "chain_length": len(_chain)}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8083))
    print("\n=== DCL Evaluator — Bazaar Server (x402 v2) ===")
    print("=== Fronesis Labs · parallel to webhook_server.py (v1) ===\n")
    uvicorn.run("bazaar_server:app", host="0.0.0.0", port=port, reload=False)
