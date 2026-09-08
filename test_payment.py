#!/usr/bin/env python3
"""One-shot x402 v2 payment against the Bazaar evaluate endpoints.

Reads payer credentials from env (never commit this file with secrets):
  PAYER_PRIVATE_KEY  — hex private key, OR
  PAYER_MNEMONIC     — BIP39 mnemonic (must derive to EXPECTED_PAYER)

Usage:
  python test_payment.py
  python test_payment.py --endpoint /evaluate/fast
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import requests
from dotenv import load_dotenv
from eth_account import Account

from x402 import x402ClientSync
from x402.http.clients.requests import x402_requests
from x402.mechanisms.evm.exact import register_exact_evm_client

load_dotenv()

BASE_URL = os.environ.get("BAZAAR_BASE_URL", "https://bazaar.fronesislabs.com")
SENTINEL_BASE_URL = "https://webhook.fronesislabs.com"
EXPECTED_PAYER = os.environ.get(
    "PAYER_ADDRESS", "0x21d8844377ecf82218a5fa9f42290dd3a23577c3"
).lower()
EXPECTED_PAYEE = os.environ.get(
    "PAYEE_ADDRESS", "0xb790ed3796194E5511C44411CF045F67E069cdC0"
).lower()

DEFAULT_BODY = {
    "response": "Payment probe: agent output to audit for Bazaar verify+settle.",
    "agent_id": "bazaar-payment-probe",
    "policy": "default",
    "task_type": "fast",
}

ENDPOINTS = [
    "/evaluate/fast",
    "/evaluate/strict",
    "/evaluate/jailbreak",
    "/evaluate/safety",
    "/evaluate/quality",
]


def _load_payer_account() -> Account:
    pk = os.environ.get("PAYER_PRIVATE_KEY", "").strip()
    if pk:
        if not pk.startswith("0x"):
            pk = "0x" + pk
        return Account.from_key(pk)

    mnemonic = os.environ.get("PAYER_MNEMONIC", "").strip()
    if mnemonic:
        Account.enable_unaudited_hdwallet_features()
        return Account.from_mnemonic(mnemonic)

    raise SystemExit(
        "Set PAYER_PRIVATE_KEY or PAYER_MNEMONIC in the environment."
    )


def _build_session() -> requests.Session:
    account = _load_payer_account()
    if account.address.lower() != EXPECTED_PAYER:
        print(
            f"WARNING: payer {account.address} != expected {EXPECTED_PAYER}",
            file=sys.stderr,
        )
    client = x402ClientSync()
    register_exact_evm_client(client, account, networks="eip155:8453")
    return x402_requests(client)


def pay_endpoint(session: requests.Session, path: str, body: dict) -> None:
    url = f"{BASE_URL}{path}"
    print(f"\n=== POST {url} ===")
    r = session.post(url, json=body, timeout=120)
    print(f"HTTP {r.status_code}")
    for hdr in ("payment-response", "x-payment-response"):
        if hdr in r.headers:
            print(f"{hdr}: {r.headers[hdr][:120]}...")
    try:
        data = r.json()
        print(json.dumps(data, indent=2)[:2000])
    except Exception:
        print(r.text[:500])
    if r.status_code != 200:
        raise SystemExit(f"Payment call failed for {path}: HTTP {r.status_code}")
    pay_to = data.get("payTo") if isinstance(data, dict) else None
    if pay_to:
        pass
    print(f"OK — paid response from {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="x402 Bazaar payment probe")
    parser.add_argument(
        "--endpoint",
        action="append",
        dest="endpoints",
        help="Endpoint path (repeatable). Default: all /evaluate/*",
    )
    args = parser.parse_args()
    targets = args.endpoints or ENDPOINTS

    print(f"Base URL: {BASE_URL}")
    print(f"Expected payee (merchant): {EXPECTED_PAYEE}")
    session = _build_session()
    print(f"Payer wallet: {_load_payer_account().address}")

    pay_sentinel_scan(session, "torvalds/linux", "update_rescan")
    return
    
    for path in targets:
        body = dict(DEFAULT_BODY)
        body["task_type"] = path.rsplit("/", 1)[-1]
        pay_endpoint(session, path, body)

    print("\nAll payments completed.")

def pay_sentinel_scan(session, repo_full_name, scan_type="update_rescan"):
    url = f"{SENTINEL_BASE_URL}/sentinel/scan"
    body = {
        "repo_full_name": repo_full_name,
        "scan_type": scan_type,
        "payer_ref": _load_payer_account().address,
    }
    r = session.post(url, json=body, timeout=120)
    print(f"HTTP {r.status_code}")
    print(r.text[:1000])
    
if __name__ == "__main__":
    main()
