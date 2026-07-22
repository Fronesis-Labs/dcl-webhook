markdown
12345678910111213
Server runs on http://localhost:8080
API Endpoints
Endpoint
Price
Description
POST /evaluate/fast
$0.01
Fast policy check for low-risk outputs
POST /evaluate/strict
$0.05
Deep analysis for high-stakes content
GET /audit/{tx_hash}
$0.10
Basic post-action audit
GET /audit/{tx_hash}/deep
$0.50
Deep forensic audit with drift context
Example Response
json
12345678910111213
x402 Integration
All paid endpoints return 402 Payment Required with x402 payment instructions.
Network: Base
Asset: USDC
Privacy-First
✅ Content never stored (only SHA-256 hashes)
✅ GDPR Art.25 compliant (privacy by design)
✅ Tamper-evident cryptographic chain
Built-in Policies
default — General purpose
anti_jailbreak — Prevents prompt injection
safety — Content safety checks
content_quality — Quality assurance
Links
Live API: https://fronesislabs.com/docs
MCP Manifest: https://fronesislabs.com/.well-known/agent.json
GitHub: https://github.com/Fronesis-Labs/dcl-webhook
Contact
partnership@fronesislabs.com
