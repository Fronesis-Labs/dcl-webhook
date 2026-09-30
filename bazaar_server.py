"""
DCL Trust Oracle — Bazaar/x402 v2 Server
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
import threading
import time
import uuid
from pathlib import Path
from typing import List, Literal, Optional

from dotenv import load_dotenv
load_dotenv(override=True)

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field

from x402 import x402ResourceServer
from x402.http import FacilitatorConfig, HTTPFacilitatorClient
from x402.http.middleware.fastapi import PaymentMiddlewareASGI
from x402.http.types import PaymentOption, RouteConfig
from x402.mechanisms.evm.exact import register_exact_evm_server
from x402.extensions.bazaar import declare_discovery_extension, bazaar_resource_server_extension, OutputConfig

from dcl_core import ChainState, sha256hex
from audit_logic import (
    BUILTIN_POLICIES, evaluate_policy, get_drift_mode,
    detect_secrets, detect_pii, format_seal,
)
from dcl_audit_event import create_audit_event
from dcl_audit_store import ensure_schema, persist_audit_event
from payment_logger import BAZAAR_ROUTE_PRICES, log_payment

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
    title="DCL Trust Oracle — AI Agent Safety Evaluator",
    description=(
        "Pre-action safety gate for AI agents. Evaluates a proposed agent response or "
        "action before execution and returns a structured verdict (COMMIT / NO_COMMIT) "
        "with confidence, reasoning, and tamper-evident audit metadata. Use it before tool "
        "calls, code execution, financial workflows, or other high-impact actions to detect "
        "jailbreaks, prompt injection, unsafe instructions, policy violations, and output "
        "drift. Choose from fast, safety, jailbreak, quality, and strict evaluation modes. "
        "Pay per request via x402 — no API key or subscription required."
    ),
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

_chain = ChainState(os.environ.get("DCL_DB_PATH", "dcl_chain_bazaar.db"))
# Production dcl-core exposes ChainState._lock and append() holds it.
# The package installed for local tests does not, so fall back to a lock
# that only covers the audit/payment writes beside that connection.
_fallback_lock = threading.RLock()


def _chain_lock():
    return getattr(_chain, "_lock", None) or _fallback_lock


ensure_schema(_chain._conn, _chain_lock())
_commit_rate: list = []

# ════════════════════════════════════════════════════════════════════════════════
# Human-facing landing page (GET / only — does not touch /evaluate/* payment routes)
# ════════════════════════════════════════════════════════════════════════════════
_STATIC_DIR = Path(__file__).parent / "static"

@app.get("/", include_in_schema=False)
async def landing():
    return HTMLResponse((_STATIC_DIR / "index.html").read_text(encoding="utf-8"))

@app.get("/robots.txt", include_in_schema=False)
async def robots():
    return PlainTextResponse((_STATIC_DIR / "robots.txt").read_text(encoding="utf-8"))

@app.get("/sitemap.xml", include_in_schema=False)
async def sitemap():
    return Response((_STATIC_DIR / "sitemap.xml").read_text(encoding="utf-8"), media_type="application/xml")

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
    "type": "object",
    "properties": {
        "response": {
            "type": "string",
            "minLength": 1,
            "description": "Proposed tool call or agent text to check before execution.",
        },
        "agent_id": {
            "type": "string",
            "description": "Caller id stored on the audit record. Optional; omitted values are stored as unknown.",
        },
    },
    "required": ["response"],
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
        "verdict": {
            "type": "string",
            "enum": ["COMMIT", "NO_COMMIT"],
            "description": "COMMIT: the checked action may proceed. NO_COMMIT: do not execute the checked action.",
        },
        "confidence": {
            "type": "number",
            "description": "Phrase-policy score from 0 to 1. Not a probability that the action is safe.",
        },
        "reason": {
            "type": "string",
            "description": "Checks passed, or the phrase rule that produced NO_COMMIT.",
        },
        "tx_hash": {
            "type": "string",
            "description": "Hash of this audit record. Not the payment transaction.",
        },
        "chain_index": {
            "type": "integer",
            "description": "Position of this record in the append-only audit chain.",
        },
        "input_hash": {"type": "string"},
        "policy_version": {"type": "string"},
        "timestamp": {"type": "number"},
        "drift_mode": {"type": "string"},
        "drift_score": {"type": "number"},
    },
    "required": ["verdict", "confidence", "reason", "tx_hash", "chain_index"],
}


def _route_config(path: str, price: str, description: str, tags: list, invoke_method: str = "POST") -> RouteConfig:
    extension = declare_discovery_extension(
        input={"response": "example agent output", "agent_id": "agent-123"},
        input_schema=_EVALUATE_INPUT_SCHEMA,
        body_type="json",
        output=OutputConfig(example=_EVALUATE_OUTPUT_EXAMPLE, schema=_EVALUATE_OUTPUT_SCHEMA),
    )
    extension["bazaar"]["info"]["input"]["method"] = invoke_method
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
        service_name="DCL Trust Oracle",
        tags=tags,
        extensions=extension,
    )


_EVALUATE_PATHS = [
    ("/evaluate/fast", "$0.01",
     "DCL Trust Oracle guardrail before a tool call. POST JSON {\"response\",\"agent_id\"}. "
     "Policy evaluation flags only: \"ignore previous instructions\", \"jailbreak\", \"bypass safety\". "
     "COMMIT means the action may proceed. NO_COMMIT means do not execute the checked action. "
     "Not a general prompt-injection or agent-safety classifier.",
     ["ai-agent-safety", "tool-call-guardrail", "llm-evaluation"]),
    ("/evaluate/strict", "$0.05",
     "Higher-confidence evaluation for high-impact, sensitive, or difficult-to-reverse agent actions.",
     ["ai-agent-safety", "policy-enforcement", "audit-trail", "llm-evaluation"]),
    ("/evaluate/jailbreak", "$0.02",
     "Detect jailbreaks, prompt injection, and instruction conflicts before the agent follows them.",
     ["jailbreak-detection", "prompt-injection", "ai-agent-safety"]),
    ("/evaluate/safety", "$0.01",
     "DCL Trust Oracle. Call only when the text must include the substring AI and must not include "
     "\"I cannot be held responsible\" or \"no guarantees\". POST JSON {\"response\",\"agent_id\"}. "
     "COMMIT means those checks passed and the action may proceed. "
     "NO_COMMIT means do not execute the checked action. "
     "Not a tool-call guardrail, prompt-injection detector, or jailbreak screen; "
     "use /evaluate/fast for the phrase check.",
     ["ai-agent-safety", "policy-enforcement", "audit-trail"]),
    ("/evaluate/quality", "$0.03",
     "Check agent outputs for quality issues, inconsistency, and behavioral drift.",
     ["output-quality", "llm-evaluation", "ai-agent-safety"]),
]

def _protected_route(
    method: str,
    path: str,
    price: str,
    description: str,
    tags: list,
    extension: dict,
) -> RouteConfig:
    """Payment + discovery for a resource that is not one of the original five.

    Same wallet, network, scheme, and timeout as `_route_config`. Does not call
    `_route_config`, so the shared evaluate schema stays on those five only.
    `path` uses x402 route syntax (`:param`), which is what the payment matcher
    compiles. FastAPI still declares `{param}` on the handler.
    """
    extension["bazaar"]["info"]["input"]["method"] = method
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
        service_name="DCL Trust Oracle",
        tags=tags,
        extensions=extension,
    )


_SCAN_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "response": {
            "type": "string",
            "minLength": 1,
            "description": "Text to scan. Raw content is not stored.",
        },
        "agent_id": {
            "type": "string",
            "description": "Caller id stored on the audit record. Omitted values are stored as unknown.",
        },
        "task_type": {
            "type": "string",
            "description": "Optional task tag stored on the audit record. Defaults to unknown.",
        },
    },
    "required": ["response"],
}
_SCAN_FINDING_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string"},
        "position": {"type": "integer"},
        "redacted_sample": {"type": "string"},
        "severity": {"type": "string"},
        "category": {"type": "string"},
        "provider": {"type": ["string", "null"]},
    },
    "required": ["type", "position", "redacted_sample", "severity", "category"],
}
_SCAN_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["COMMIT", "NO_COMMIT"]},
        "risk_score": {"type": "number"},
        "findings": {"type": "array", "items": _SCAN_FINDING_SCHEMA},
        "detection_count": {"type": "integer"},
        "categories_checked": {"type": "array", "items": {"type": "string"}},
        "categories_clear": {"type": "array", "items": {"type": "string"}},
        "tx_hash": {"type": "string"},
        "chain_index": {"type": "integer"},
        "input_hash": {"type": "string"},
        "timestamp": {"type": "number"},
        "seal_text": {"type": "string"},
        "verify_url": {"type": "string"},
    },
    "required": [
        "verdict", "risk_score", "findings", "detection_count",
        "categories_checked", "categories_clear", "tx_hash", "chain_index",
        "input_hash", "timestamp", "seal_text", "verify_url",
    ],
}
_SECRETS_OUTPUT_EXAMPLE = {
    "verdict": "COMMIT",
    "risk_score": 0.0,
    "findings": [],
    "detection_count": 0,
    "categories_checked": ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"],
    "categories_clear": ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"],
    "tx_hash": "0xabc123...",
    "chain_index": 42,
    "input_hash": "0xdef456...",
    "timestamp": 1721635200.0,
    "seal_text": "Verified by Leibniz Layer",
    "verify_url": "https://x402.fronesislabs.com/verify/abc123",
}
_PII_OUTPUT_EXAMPLE = {
    **_SECRETS_OUTPUT_EXAMPLE,
    "categories_checked": ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"],
    "categories_clear": ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"],
}
_BATCH_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "description": "Texts to evaluate. Each item is capped by max_items.",
            "items": {
                "type": "object",
                "properties": {
                    "response": {"type": "string", "minLength": 1},
                    "policy": {
                        "type": "string",
                        "description": "Built-in policy name. Defaults to default.",
                    },
                    "task_type": {
                        "type": "string",
                        "description": "Task tag stored on the audit record. Defaults to batch_item.",
                    },
                },
                "required": ["response"],
            },
        },
        "agent_id": {"type": "string"},
        "max_items": {
            "type": "integer",
            "description": "Reject the call when len(items) exceeds this value. Default 20.",
            "default": 20,
        },
    },
    "required": ["items", "agent_id"],
}
_BATCH_ITEM_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["COMMIT", "NO_COMMIT"]},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
        "tx_hash": {"type": "string"},
        "chain_index": {"type": "integer"},
        "input_hash": {"type": "string"},
        "policy_version": {"type": "string"},
        "timestamp": {"type": "number"},
        "pipeline_id": {"type": "string"},
        "drift_mode": {"type": "string"},
        "drift_score": {"type": "number"},
        "seal_text": {"type": "string"},
        "verify_url": {"type": "string"},
    },
    "required": [
        "verdict", "confidence", "reason", "tx_hash", "chain_index", "input_hash",
        "policy_version", "timestamp", "pipeline_id", "drift_mode", "drift_score",
        "seal_text", "verify_url",
    ],
}
_BATCH_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "batch_id": {"type": "string"},
        "agent_id": {"type": "string"},
        "count": {"type": "integer"},
        "results": {"type": "array", "items": _BATCH_ITEM_OUTPUT_SCHEMA},
    },
    "required": ["batch_id", "agent_id", "count", "results"],
}
_BATCH_OUTPUT_EXAMPLE = {
    "batch_id": "ab12cd34",
    "agent_id": "agent-123",
    "count": 1,
    "results": [{
        "verdict": "COMMIT",
        "confidence": 0.95,
        "reason": "All policy checks passed",
        "tx_hash": "0xabc123...",
        "chain_index": 42,
        "input_hash": "0xdef456...",
        "policy_version": "1.0.0",
        "timestamp": 1721635200.0,
        "pipeline_id": "abc12345",
        "drift_mode": "NORMAL",
        "drift_score": 0.0,
        "seal_text": "Verified by Leibniz Layer",
        "verify_url": "https://x402.fronesislabs.com/verify/abc123",
    }],
}
_TX_HASH_PATH_SCHEMA = {
    "properties": {
        "tx_hash": {
            "type": "string",
            "description": "Transaction hash returned by a previous evaluate or scan call.",
        }
    },
    "required": ["tx_hash"],
}
_AUDIT_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "tx_hash": {"type": "string"},
        "agent_id": {"type": "string"},
        "verdict": {"type": "string"},
        "reason": {"type": "string"},
        "confidence": {"type": "number"},
        "task_type": {"type": "string"},
        "timestamp": {"type": "number"},
        "chain_index": {"type": "integer"},
        "prev_hash": {"type": "string"},
        "chain_integrity": {"type": "boolean"},
        "seal_text": {"type": "string"},
        "verify_url": {"type": "string"},
    },
    "required": [
        "tx_hash", "agent_id", "verdict", "reason", "confidence", "task_type",
        "timestamp", "chain_index", "prev_hash", "chain_integrity",
        "seal_text", "verify_url",
    ],
}
_AUDIT_OUTPUT_EXAMPLE = {
    "tx_hash": "0xabc123...",
    "agent_id": "agent-123",
    "verdict": "COMMIT",
    "reason": "All policy checks passed",
    "confidence": 0.95,
    "task_type": "fast",
    "timestamp": 1721635200.0,
    "chain_index": 42,
    "prev_hash": "0xprev...",
    "chain_integrity": True,
    "seal_text": "Verified by Leibniz Layer",
    "verify_url": "https://x402.fronesislabs.com/verify/abc123",
}
_AUDIT_DEEP_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        **_AUDIT_OUTPUT_SCHEMA["properties"],
        "tampered_at_index": {"type": ["integer", "null"]},
        "tamper_reason": {"type": ["string", "null"]},
        "drift_context": {"type": ["object", "null"]},
    },
    "required": _AUDIT_OUTPUT_SCHEMA["required"] + [
        "tampered_at_index", "tamper_reason", "drift_context",
    ],
}
_AUDIT_DEEP_OUTPUT_EXAMPLE = {
    **_AUDIT_OUTPUT_EXAMPLE,
    "tampered_at_index": None,
    "tamper_reason": None,
    "drift_context": {"environment": "production-edge"},
}


def _body_extension(example_input: dict, input_schema: dict, output_example: dict, output_schema: dict) -> dict:
    return declare_discovery_extension(
        input=example_input,
        input_schema=input_schema,
        body_type="json",
        output=OutputConfig(example=output_example, schema=output_schema),
    )


def _get_path_extension(output_example: dict, output_schema: dict) -> dict:
    """GET discovery: path param tx_hash, no JSON body."""
    return declare_discovery_extension(
        input_schema={},
        path_params_schema=_TX_HASH_PATH_SCHEMA,
        output=OutputConfig(example=output_example, schema=output_schema),
    )


# Original five stay in `_EVALUATE_PATHS`. These five are separate RouteConfig
# entries so their schemas are not the shared evaluate schema.
_EXTRA_ROUTES = [
    ("POST", "/evaluate/secrets", "$0.02",
     "Scan text for exposed credentials, API keys, private keys, database URLs, "
     "and webhook secrets. COMMIT when nothing matches. NO_COMMIT on any finding. "
     "POST JSON {\"response\",\"agent_id\"}.",
     ["secret-detection", "credential-leak", "ai-agent-safety"],
     _body_extension(
         {"response": "example agent output", "agent_id": "agent-123"},
         _SCAN_INPUT_SCHEMA, _SECRETS_OUTPUT_EXAMPLE, _SCAN_OUTPUT_SCHEMA,
     )),
    ("POST", "/evaluate/pii", "$0.02",
     "Scan text for personal data: email, phone, national id, bank card, IBAN, "
     "crypto address, IP address, passport. COMMIT when nothing matches. "
     "NO_COMMIT on any finding. POST JSON {\"response\",\"agent_id\"}.",
     ["pii-detection", "privacy", "ai-agent-safety"],
     _body_extension(
         {"response": "example agent output", "agent_id": "agent-123"},
         _SCAN_INPUT_SCHEMA, _PII_OUTPUT_EXAMPLE, _SCAN_OUTPUT_SCHEMA,
     )),
    ("POST", "/evaluate/batch", "$0.10",
     "Evaluate a list of texts in one paid call. Each item may name a built-in policy. "
     "The call is rejected when the list is longer than max_items (default 20). "
     "POST JSON {\"items\",\"agent_id\",\"max_items\"}.",
     ["batch-evaluation", "llm-evaluation", "ai-agent-safety"],
     _body_extension(
         {
             "items": [{"response": "example agent output", "policy": "default"}],
             "agent_id": "agent-123",
             "max_items": 20,
         },
         _BATCH_INPUT_SCHEMA, _BATCH_OUTPUT_EXAMPLE, _BATCH_OUTPUT_SCHEMA,
     )),
    ("GET", "/audit/:tx_hash", "$0.10",
     "Read one audit-chain record by tx_hash. GET path parameter, no JSON body. "
     "Returns verdict, confidence, agent_id, and reason. Does not return raw content.",
     ["audit-trail", "forensics"],
     _get_path_extension(_AUDIT_OUTPUT_EXAMPLE, _AUDIT_OUTPUT_SCHEMA)),
    ("GET", "/audit/:tx_hash/deep", "$0.50",
     "Read one audit-chain record by tx_hash, plus drift context and tamper location. "
     "GET path parameter, no JSON body.",
     ["audit-trail", "forensics"],
     _get_path_extension(_AUDIT_DEEP_OUTPUT_EXAMPLE, _AUDIT_DEEP_OUTPUT_SCHEMA)),
]

routes = {}
for _path, _price, _desc, _tags in _EVALUATE_PATHS:
    routes[f"POST {_path}"] = _route_config(_path, _price, _desc, _tags, invoke_method="POST")
for _method, _path, _price, _desc, _tags, _extension in _EXTRA_ROUTES:
    routes[f"{_method} {_path}"] = _protected_route(
        _method, _path, _price, _desc, _tags, _extension,
    )

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
                    "method": method,
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
            for method, path in [path_key.split(" ", 1)]
        ],
    }


# ════════════════════════════════════════════════════════════════════════════════
# Request / Response models (mirrors webhook_server.py)
# ════════════════════════════════════════════════════════════════════════════════
class EvaluateRequest(BaseModel):
    response: str = Field(
        min_length=1,
        description="Proposed tool call or agent text to check before execution.",
    )
    agent_id: Optional[str] = Field(
        default="unknown",
        description="Caller id stored on the audit record. Omitted values are stored as unknown.",
    )


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


class BatchItemResult(BaseModel):
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


class BatchEvaluateResponse(BaseModel):
    batch_id: str
    agent_id: str
    count: int
    results: List[BatchItemResult]


class EvaluateResponse(BaseModel):
    verdict: Literal["COMMIT", "NO_COMMIT"] = Field(
        description="COMMIT: the checked action may proceed. NO_COMMIT: do not execute the checked action.",
    )
    confidence: float = Field(
        description="Phrase-policy score from 0 to 1. Not a probability that the action is safe.",
    )
    reason: str = Field(
        description="Checks passed, or the phrase rule that produced NO_COMMIT.",
    )
    tx_hash: str = Field(
        description="Hash of this audit record. Not the payment transaction.",
    )
    chain_index: int = Field(
        description="Position of this record in the append-only audit chain.",
    )
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

    if task_type == "fast":
        audit_event = create_audit_event(
            service="dcl-trust-oracle",
            producer="bazaar",
            route="/evaluate/fast",
            agent_id=req.agent_id,
            identity_source="caller_supplied",
            identity_confidence="unverified",
            task_type=task_type,
            policy_id=policy_name,
            policy_version=policy_version,
            verdict=verdict,
            proof={"tx_hash": tx_hash},
        )
        persist_audit_event(_chain._conn, _chain_lock(), audit_event)

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


def _process_scan(req: ScanRequest, detector, policy_label: str) -> ScanResponse:
    """Same scan contract as webhook_server._process_scan, on this process's chain."""
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


def _process_batch_item(item: BatchItem, agent_id: str) -> BatchItemResult:
    """One batch row. Policy lookup matches webhook_server, not the fixed-policy evaluate routes."""
    if not item.response or not item.response.strip():
        raise HTTPException(status_code=400, detail="response field is required")

    policy_name = item.policy or "default"
    policy_yaml = BUILTIN_POLICIES.get(policy_name, policy_name or BUILTIN_POLICIES["default"])
    verdict, confidence, reason, policy_version = evaluate_policy(item.response, policy_yaml)
    input_hash = "0x" + sha256hex(item.response)[:16]
    policy_hash = sha256hex(policy_yaml)[:16]

    tx_hash, chain_idx = _chain.append(
        verdict=verdict, input_hash=input_hash, policy_hash=policy_hash,
        agent_id=agent_id, reason=reason, confidence=confidence, task_type=item.task_type,
        drift_context={"environment": "production-edge", "policy_version_hash": policy_hash},
    )

    _commit_rate.append(1.0 if verdict == "COMMIT" else 0.0)
    if len(_commit_rate) > 100:
        _commit_rate.pop(0)
    drift_mode, drift_score = get_drift_mode(_commit_rate)

    ts = time.time()
    seal = format_seal(tx_hash, input_hash, ts)
    return BatchItemResult(
        verdict=verdict, confidence=confidence, reason=reason,
        tx_hash=tx_hash, chain_index=chain_idx, input_hash=input_hash,
        policy_version=policy_version, timestamp=ts,
        pipeline_id=str(uuid.uuid4())[:8],
        drift_mode=drift_mode, drift_score=drift_score,
        seal_text=seal["seal_text"], verify_url=seal["verify_url"],
    )


# ════════════════════════════════════════════════════════════════════════════════
# Routes (payment enforced by the middleware above, based on `routes` config)
# ════════════════════════════════════════════════════════════════════════════════
def _log_chain_payment(tx_hash: str, route: str) -> None:
    log_payment(
        _chain._conn, _chain_lock(), tx_hash, route,
        BAZAAR_ROUTE_PRICES[route], network=X402_NETWORK,
    )


@app.post("/evaluate/fast", response_model=EvaluateResponse)
async def evaluate_fast(req: EvaluateRequest):
    result = _process_evaluation(req, "default", "fast")
    _log_chain_payment(result.tx_hash, "/evaluate/fast")
    return result


@app.post("/evaluate/strict", response_model=EvaluateResponse)
async def evaluate_strict(req: EvaluateRequest):
    result = _process_evaluation(req, "default", "strict")
    _log_chain_payment(result.tx_hash, "/evaluate/strict")
    return result


@app.post("/evaluate/jailbreak", response_model=EvaluateResponse)
async def evaluate_jailbreak(req: EvaluateRequest):
    result = _process_evaluation(req, "anti_jailbreak", "jailbreak")
    _log_chain_payment(result.tx_hash, "/evaluate/jailbreak")
    return result


@app.post("/evaluate/safety", response_model=EvaluateResponse)
async def evaluate_safety(req: EvaluateRequest):
    result = _process_evaluation(req, "safety", "safety")
    _log_chain_payment(result.tx_hash, "/evaluate/safety")
    return result


@app.post("/evaluate/quality", response_model=EvaluateResponse)
async def evaluate_quality(req: EvaluateRequest):
    result = _process_evaluation(req, "content_quality", "quality")
    _log_chain_payment(result.tx_hash, "/evaluate/quality")
    return result


@app.post("/evaluate/secrets", response_model=ScanResponse)
async def evaluate_secrets(req: ScanRequest):
    result = _process_scan(req, detect_secrets, "secret_leak_v1")
    _log_chain_payment(result.tx_hash, "/evaluate/secrets")
    return result


@app.post("/evaluate/pii", response_model=ScanResponse)
async def evaluate_pii(req: ScanRequest):
    result = _process_scan(req, detect_pii, "pii_v1")
    _log_chain_payment(result.tx_hash, "/evaluate/pii")
    return result


@app.post("/evaluate/batch", response_model=BatchEvaluateResponse)
async def evaluate_batch(req: BatchEvaluateRequest):
    if len(req.items) > req.max_items:
        raise HTTPException(400, f"Batch limited to {req.max_items} items")
    results = [_process_batch_item(item, req.agent_id) for item in req.items]
    if results:
        _log_chain_payment(results[0].tx_hash, "/evaluate/batch")
    return BatchEvaluateResponse(
        batch_id=str(uuid.uuid4())[:8],
        agent_id=req.agent_id,
        count=len(results),
        results=results,
    )


@app.get("/audit/{tx_hash}/deep")
async def audit_decode_deep(tx_hash: str):
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


@app.get("/audit/{tx_hash}")
async def audit_decode(tx_hash: str):
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


async def _post_payment_required_header(path: str) -> str:
    """402 header for a GET probe, advertising the POST JSON declaration."""
    import copy

    from x402.http.utils import encode_payment_required_header
    from x402.schemas import ResourceInfo

    cfg = routes[f"POST {path}"]
    if not server._initialized:
        server.initialize()
    requirements = server.build_payment_requirements(cfg.accepts)
    extensions = copy.deepcopy(cfg.extensions) if cfg.extensions else None
    if extensions and "bazaar" in extensions:
        extensions["bazaar"]["info"]["input"]["method"] = "POST"
    resource = ResourceInfo(
        url=cfg.resource or f"{PUBLIC_BASE_URL}{path}",
        description=cfg.description or "",
        mime_type=cfg.mime_type or "application/json",
        service_name=cfg.service_name,
        tags=cfg.tags,
    )
    payment_required = await server.create_payment_required_response(
        requirements,
        resource,
        "Payment required",
        extensions,
    )
    return encode_payment_required_header(payment_required)


@app.get("/evaluate/fast")
@app.get("/evaluate/strict")
@app.get("/evaluate/jailbreak")
@app.get("/evaluate/safety")
@app.get("/evaluate/quality")
async def evaluate_get_method_not_allowed(request: Request):
    return JSONResponse(
        status_code=402,
        content={},
        headers={
            "PAYMENT-REQUIRED": await _post_payment_required_header(request.url.path),
            "Cache-Control": "no-store",
        },
    )


@app.get("/health")
def health():
    return {"status": "ok", "service": "DCL Trust Oracle Bazaar (x402 v2)", "chain_length": len(_chain)}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8083))
    print("\n=== DCL Trust Oracle — Bazaar Server (x402 v2) ===")
    print("=== Fronesis Labs · parallel to webhook_server.py (v1) ===\n")
    uvicorn.run("bazaar_server:app", host="0.0.0.0", port=port, reload=False)
