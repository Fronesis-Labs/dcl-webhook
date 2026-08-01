# DCL Trust Oracle — x402 MCP Server

[![dcl-webhook MCP server](https://glama.ai/mcp/servers/Fronesis-Labs/dcl-webhook/badges/card.svg)](https://glama.ai/mcp/servers/Fronesis-Labs/dcl-webhook)
[![Score](https://glama.ai/mcp/servers/Fronesis-Labs/dcl-webhook/badges/score.svg)](https://glama.ai/mcp/servers/Fronesis-Labs/dcl-webhook/score)

Deterministic AI audit layer with cryptographic micropayments via x402 protocol.

## What It Does

DCL Trust Oracle provides deterministic policy evaluation for LLM outputs with a tamper-evident audit chain. The system stores only cryptographic hashes and decision metadata — never raw content — enabling verifiable, post-action forensic analysis across distributed AI agents.

Available two ways, both metered per call via x402:

- **REST API** (`webhook_server.py`) — direct HTTP integration, x402-gated with `fastapi-x402`.
- **MCP Server** (`mcp_server.py`) — native Model Context Protocol integration for AI agents, hosted on Smithery, x402-gated with `paymcp`.

Both servers share the same evaluation logic and tamper-evident chain (`dcl_core.py`), and are priced identically.

## Quick Start

### REST API
```
pip install -r requirements.txt
python webhook_server.py
```
Server runs on `http://localhost:8080`

### MCP Server
```
pip install -r requirements.txt
python mcp_server.py
```
Server runs on `http://localhost:8081` (streamable-http transport)

## Tools & Endpoints

### Pre-Action Evaluation

| REST Endpoint | MCP Tool | Price | Description |
| --- | --- | --- | --- |
| `POST /evaluate/fast` | `dcl_evaluate_fast` | $0.01 | Fast policy check for low-risk outputs. Returns tamper-evident `tx_hash`. |
| `POST /evaluate/strict` | `dcl_evaluate_strict` | $0.05 | Deep analysis for high-stakes outputs with higher confidence thresholds. |
| `POST /evaluate/jailbreak` | `dcl_evaluate_jailbreak` | $0.02 | Instruction adherence check — detects prompt injection patterns and role-hijacking attempts. |
| `POST /evaluate/safety` | `dcl_evaluate_safety` | $0.01 | Baseline screening for known harmful text patterns. Optimized for high throughput. |
| `POST /evaluate/quality` | `dcl_evaluate_quality` | $0.03 | Content quality & drift check — evaluates format adherence and contextual drift. |
| `POST /evaluate/batch` | `dcl_evaluate_batch` | $0.10 | Bulk processing — up to 20 items per transaction. Cost-effective for multi-turn history. |

### Session Management

| REST Endpoint | MCP Tool | Price | Description |
| --- | --- | --- | --- |
| `POST /pipeline/start` | `dcl_pipeline_start` | $0.05 | Initializes a long-running audit session for continuous drift tracking. Returns `pipeline_id`. |

### Post-Action Forensics

| REST Endpoint | MCP Tool | Price | Description |
| --- | --- | --- | --- |
| `GET /audit/{tx_hash}` | `dcl_audit_decode` | $0.10 | Basic post-action audit — returns verdict, confidence, agent_id, reason by `tx_hash`. |
| `GET /audit/{tx_hash}/deep` | `dcl_audit_decode_deep` | $0.50 | Deep forensic audit — includes drift context, tamper-evidence indices, environmental metadata. |

### Utility (free, REST only)

| Endpoint | Description |
| --- | --- |
| `GET /health` | Service status and chain length |
| `GET /policies` | List of built-in policy names |
| `GET /chain/status` | Chain integrity, drift mode, drift score |
| `GET /chain/export` | Full chain export with integrity verification |

## Example Response

```json
{
  "verdict": "COMMIT",
  "confidence": 0.95,
  "reason": "All policy checks passed",
  "tx_hash": "0x7a8f3b2c...",
  "chain_index": 42,
  "input_hash": "0x9d4e1f...",
  "policy_version": "1.0.0",
  "timestamp": 1721635200.123,
  "pipeline_id": "abc123",
  "drift_mode": "NORMAL",
  "drift_score": 0.15
}
```

## x402 Integration

All paid endpoints and tools require payment via the x402 protocol before returning a result — no free tier, no bypass. Both the REST API and MCP server settle to the same wallet.

- **Networks:** Base, Avalanche, IoTeX
- **Asset:** USDC
- **Facilitator (REST):** `https://x402.org/facilitator` (via `fastapi-x402`)
- **Facilitator (MCP):** `paymcp` in `Mode.AUTO` — automatic on-chain payment for x402-aware MCP clients, guided payment link (ELICITATION/RESUBMIT) for clients without a wallet. No path skips payment.

## Metadata-Only Architecture

DCL Trust Oracle is designed around a **hash-based audit trail**:

- Raw content never stored — only SHA-256 hashes and decision metadata
- Tamper-evident cryptographic chain (SQLite-backed, WAL mode)
- Chain survives server restarts — audit records persist indefinitely
- Verifiable integrity via `GET /chain/status`
- Full export via `GET /chain/export`
- Shared chain across REST and MCP — a `tx_hash` returned by one interface can be audited through the other

Each audit record contains: `tx_hash`, `prev_hash`, `verdict`, `input_hash`, `policy_hash`, `agent_id`, `reason`, `confidence`, `task_type`, `timestamp`, and `drift_context`.

## Built-in Policies

| Policy | Min Confidence | Purpose |
| --- | --- | --- |
| `default` | 0.70 | General-purpose evaluation |
| `anti_jailbreak` | 0.80 | Detects prompt injection, role-hijacking, DAN-style attacks |
| `safety` | 0.75 | Baseline harmful-pattern screening |
| `content_quality` | 0.85 | Quality assurance, format adherence, drift detection |

Custom policies can be passed inline as YAML via the `policy` field.

## Links

- **Live API:** https://fronesislabs.com/docs
- **MCP Server:** https://mcp.fronesislabs.com
- **MCP Manifest:** https://fronesislabs.com/.well-known/agent.json
- **GitHub:** https://github.com/Fronesis-Labs/dcl-webhook

## Contact

partnership@fronesislabs.com

