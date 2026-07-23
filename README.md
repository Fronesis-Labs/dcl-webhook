# DCL Trust Oracle — x402 MCP Server

**Deterministic AI audit layer with cryptographic micropayments via x402 protocol.**

## What It Does

DCL Trust Oracle provides deterministic policy evaluation for LLM outputs with a tamper-evident audit chain. The system stores only cryptographic hashes and decision metadata — never raw content — enabling verifiable, post-action forensic analysis across distributed AI agents.

Built for MCP-ecosystem integration. All endpoints are payable via x402 micropayments on Base (USDC).

## Quick Start

```bash
pip install -r requirements.txt
python webhook_server.py
```

Server runs on `http://localhost:8080`

## API Endpoints

### Pre-Action Evaluation (9 tools)

| Endpoint | Price | Description |
|---|---|---|
| `POST /evaluate/fast` | $0.01 | Fast policy check for low-risk outputs. Returns tamper-evident `tx_hash`. |
| `POST /evaluate/strict` | $0.05 | Deep analysis for high-stakes outputs with higher confidence thresholds. |
| `POST /evaluate/jailbreak` | $0.02 | Instruction adherence check — detects prompt injection patterns and role-hijacking attempts. |
| `POST /evaluate/safety` | $0.01 | Baseline screening for known harmful text patterns. Optimized for high throughput. |
| `POST /evaluate/quality` | $0.03 | Content quality & drift check — evaluates format adherence and contextual drift. |
| `POST /evaluate/batch` | $0.10 | Bulk processing — up to 20 items per x402 transaction. Cost-effective for multi-turn history. |

### Session Management

| Endpoint | Price | Description |
|---|---|---|
| `POST /pipeline/start` | $0.05 | Initializes a long-running audit session for continuous drift tracking. Returns `pipeline_id`. |

### Post-Action Forensics

| Endpoint | Price | Description |
|---|---|---|
| `GET /audit/{tx_hash}` | $0.10 | Basic post-action audit — returns verdict, confidence, agent_id, reason by `tx_hash`. |
| `GET /audit/{tx_hash}/deep` | $0.50 | Deep forensic audit — includes drift context, tamper-evidence indices, environmental metadata. |

### Utility (free)

| Endpoint | Description |
|---|---|
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

All paid endpoints return `402 Payment Required` with x402 payment instructions when called without a valid payment header.

- **Network:** Base
- **Asset:** USDC
- **Facilitator:** `https://x402.org/facilitator`

## Metadata-Only Architecture

DCL Trust Oracle is designed around a **hash-based audit trail**:

✅ Raw content never stored — only SHA-256 hashes and decision metadata  
✅ Tamper-evident cryptographic chain (SQLite-backed, WAL mode)  
✅ Chain survives server restarts — audit records persist indefinitely  
✅ Verifiable integrity via `GET /chain/status`  
✅ Full export via `GET /chain/export`

Each audit record contains: `tx_hash`, `prev_hash`, `verdict`, `input_hash`, `policy_hash`, `agent_id`, `reason`, `confidence`, `task_type`, `timestamp`, and `drift_context`.

## Built-in Policies

| Policy | Min Confidence | Purpose |
|---|---|---|
| `default` | 0.70 | General-purpose evaluation |
| `anti_jailbreak` | 0.80 | Detects prompt injection, role-hijacking, DAN-style attacks |
| `safety` | 0.75 | Baseline harmful-pattern screening |
| `content_quality` | 0.85 | Quality assurance, format adherence, drift detection |

Custom policies can be passed inline as YAML via the `policy` field.

## Links

- **Live API:** https://fronesislabs.com/docs
- **MCP Manifest:** https://fronesislabs.com/.well-known/agent.json
- **GitHub:** https://github.com/Fronesis-Labs/dcl-webhook

## Contact

partnership@fronesislabs.com
