# DCL Trust Oracle

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-compatible-green.svg)](https://modelcontextprotocol.io)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![dcl-webhook MCP server](https://glama.ai/mcp/servers/Fronesis-Labs/dcl-webhook/badges/card.svg)](https://glama.ai/mcp/servers/Fronesis-Labs/dcl-webhook)
[![Smithery](https://img.shields.io/badge/Smithery-listed-orange.svg)](https://smithery.ai/servers/fronesislabs/dcl-trust-oracle)
[![dcl-webhook MCP server](https://glama.ai/mcp/servers/Fronesis-Labs/dcl-webhook/badges/card.svg)](https://glama.ai/mcp/servers/Fronesis-Labs/dcl-webhook)

**Don't trust the agent. Trust the proof.**

Autonomous AI agents now take actions with real consequences — financial,
legal, reputational. Most of them are black boxes: no record of what was
decided, why, or whether that decision was tampered with afterward.

DCL Trust Oracle closes that gap. Every agent output is evaluated against
policy in real time and sealed into a tamper-evident hash chain — a
deterministic, cryptographically verifiable record of what happened and
when. Edit any past entry and the entire chain invalidates. No one — not
even Fronesis Labs — has to be trusted for the record to hold up.

## What It Does

DCL Trust Oracle provides deterministic policy evaluation for LLM outputs
with a tamper-evident audit chain. The system stores only cryptographic
hashes and decision metadata — **never raw content** — enabling verifiable,
post-action forensic analysis across distributed AI agents.

Available two ways:

- **REST API** (`webhook_server.py`) — direct HTTP integration.
- **MCP Server** (`mcp_server.py`) — native Model Context Protocol
  integration for AI agents, hosted on Smithery.

Both servers share the same evaluation logic and tamper-evident chain
(`dcl_core.py`), and are priced identically.

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

Catch a bad output *before* it reaches a user, a wallet, or downstream
system.

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

When something *did* go wrong, reconstruct exactly what happened.

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

## Verifying the Chain Yourself

You don't have to take the server's word for it. `tx_hash` is recomputed
from the record's own fields, not just linked to the previous row — so
anyone can independently confirm a record wasn't edited after the fact,
without calling back into this server. See
[`@fronesis-labs/dcl-sdk`](https://github.com/Fronesis-Labs/dcl-sdk) (TS/JS)
or [`dcl-core`](https://github.com/Fronesis-Labs/dcl-core) (Python) for the
free, offline verification libraries.

## Metering & Settlement

Every paid call above is metered and settled automatically per request, via
the [x402 protocol](https://x402.org) (USDC on Base, Avalanche, or IoTeX) —
no subscription, no API-key provisioning, no invoicing overhead. This is
what makes per-call pricing practical at agent scale (an autonomous system
can make thousands of evaluation calls a day). The REST API is x402-gated
via `fastapi-x402`; the MCP server via `paymcp` in `Mode.AUTO`, which pays
automatically for x402-aware clients and falls back to a guided payment
link for clients without a wallet configured. Both settle to the same
wallet, and neither has a bypass path — an unpaid call simply gets no
verdict.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
