"""
Call log_payment() right after `tx_hash, chain_idx = _chain.append(...)`
succeeds in each route handler — never before, since a failed audit call
should not create an orphaned payment row.

Reuses the SAME sqlite3 connection + threading.RLock that ChainState already
holds (see chain.py — the BEGIN IMMEDIATE pattern). Do NOT open a second
sqlite3.connect() on this db file from here; that reintroduces the exact
race condition class dcl-core v0.1.1 fixed. If ChainState doesn't currently
expose its connection/lock, add a small accessor (e.g. `chain_state._conn`,
`chain_state._lock`) rather than instantiating a new connection.

Static prices, copied from the @pay(...) decorators / _EVALUATE_PATHS list —
keep these in sync by hand if a price ever changes on a route.

NAMING TRAP — two unrelated things are both called "tx_hash" in this stack:
  - chain.py's tx_hash  = sha256 of the audit CONTENT (protocol-level, no chain involved)
  - request.state.tx_hash (fastapi_x402, dependencies.py) = the on-chain PAYMENT
    transaction hash from the facilitator's settle response
Always pass chain.py's tx_hash to log_payment() (it's the join key against
`chain`/`chain_payments`) — never the payment tx hash from request.state.
"""

# webhook_server.py (fastapi_x402, networks: base / avalanche / iotex — payer
# chooses at settlement time). Confirmed source of the actually-used network:
# request.state.payment_requirements.network (set in fastapi_x402's
# middleware.py during verify, read the same way Sentinel already reads
# request.state.payment_payer). May be absent on a non-standard path, so
# always read it defensively — see get_settled_network() below.
WEBHOOK_ROUTE_PRICES = {
    "/evaluate/fast": 0.01,
    "/evaluate/strict": 0.05,
    "/evaluate/jailbreak": 0.02,
    "/evaluate/safety": 0.01,
    "/evaluate/quality": 0.03,
    "/evaluate/secrets": 0.02,
    "/evaluate/pii": 0.02,
    "/evaluate/batch": 0.10,
    "/pipeline/start": 0.05,
    "/audit/{tx_hash}": 0.10,
    "/audit/{tx_hash}/deep": 0.50,
}

# bazaar_server.py (x402 v2, single configured network via X402_NETWORK env,
# CAIP-2 format e.g. "eip155:8453" — always known, pass it in directly)
BAZAAR_ROUTE_PRICES = {
    "/evaluate/fast": 0.01,
    "/evaluate/strict": 0.05,
    "/evaluate/jailbreak": 0.02,
    "/evaluate/safety": 0.01,
    "/evaluate/quality": 0.03,
    "/evaluate/secrets": 0.02,
    "/evaluate/pii": 0.02,
    "/evaluate/batch": 0.10,
    "/audit/:tx_hash": 0.10,
    "/audit/:tx_hash/deep": 0.50,
}


def get_settled_network(request) -> str | None:
    """webhook_server.py only. Defensive read — payment_requirements may be
    absent if verification took a non-standard path; never raise for that."""
    req = getattr(request.state, "payment_requirements", None)
    return getattr(req, "network", None) if req is not None else None


def log_payment(conn, lock, tx_hash: str, route: str, amount_usdc: float, network: str | None = None) -> None:
    """Best-effort. Never let a logging failure surface as a failed audit response."""
    try:
        with lock:
            conn.execute(
                """
                INSERT INTO chain_payments (tx_hash, route, amount_usdc, network)
                VALUES (?, ?, ?, ?)
                """,
                (tx_hash, route, amount_usdc, network),
            )
            conn.commit()
    except Exception as e:
        print(f"[payment_log] failed to log payment for {tx_hash}: {e}")


# ── Wiring example for webhook_server.py ─────────────────────────────────────
#
#   tx_hash, chain_idx = _chain.append(...)   # chain.py's tx_hash — content hash
#   log_payment(_chain._conn, _chain._lock, tx_hash, "/evaluate/fast",
#               WEBHOOK_ROUTE_PRICES["/evaluate/fast"],
#               network=get_settled_network(request))
#
# ── Wiring example for bazaar_server.py ──────────────────────────────────────
#
#   tx_hash, chain_idx = _chain.append(...)
#   log_payment(_chain._conn, _chain._lock, tx_hash, "/evaluate/fast",
#               BAZAAR_ROUTE_PRICES["/evaluate/fast"], network=X402_NETWORK)

