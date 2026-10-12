"""/sentinel/renew payer binding.

The facilitator is replaced by a recording fake. Each test asserts the order of
facilitator calls (``verify`` then, only for an accepted payer, ``settle``) and
the database state, so a rejection is proven to happen before settlement and
without extending the subscription.

Scope: this binds a renewal to the payer recorded at Sentinel registration
(``owner_payer_ref``). It is not TRC-8004 identity verification.
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("DCL_DB_PATH", os.path.join(tempfile.mkdtemp(), "dcl_test.db"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient  # noqa: E402
from fastapi_x402.core import get_config  # noqa: E402
from fastapi_x402.networks import get_default_asset_config  # noqa: E402

import sentinel_x402  # noqa: E402
import webhook_server  # noqa: E402
from sentinel_db import SentinelDB  # noqa: E402

REPO = "acme/skill"
OWNER = "0x" + "ab" * 20
OTHER = "0x" + "cd" * 20


class FakeFacilitator:
    """Records the order of facilitator calls; no network access."""

    def __init__(self, *, payer=OWNER, valid=True, payment_id="pay-1", settle_ok=True):
        self.payer = payer
        self.valid = valid
        self.payment_id = payment_id
        self.settle_ok = settle_ok
        self.calls: list[str] = []

    async def verify_payment(self, *, payment_header, payment_requirements):
        self.calls.append("verify")
        return SimpleNamespace(
            isValid=self.valid,
            payment_id=self.payment_id,
            payer=self.payer,
            error=None if self.valid else "invalid signature",
        )

    async def settle_payment(self, *, payment_header, payment_requirements):
        self.calls.append("settle")
        return SimpleNamespace(
            success=self.settle_ok,
            errorReason=None if self.settle_ok else "settlement failed",
        )


@pytest.fixture(autouse=True)
def _fresh_rate_limits():
    webhook_server.limiter.reset()
    yield
    webhook_server.limiter.reset()


@pytest.fixture
def db(tmp_path, monkeypatch):
    database = SentinelDB(str(tmp_path / "sentinel.db"))
    monkeypatch.setattr(webhook_server, "_sentinel_db", database)
    yield database
    database.close()


@pytest.fixture
def client():
    return TestClient(webhook_server.app)


def register_skill(database, *, owner_payer_ref):
    return database.create_skill(
        repo_full_name=REPO,
        owner_ref="owner",
        owner_payer_ref=owner_payer_ref,
        webhook_secret="whsec_test",
        policy={},
        version="1.0.0",
        verdict="PASS",
        score=0.9,
        audited_at=datetime.now(timezone.utc),
        payer_ref=owner_payer_ref,
    )


def use_facilitator(monkeypatch, facilitator):
    monkeypatch.setattr(sentinel_x402, "get_facilitator_client", lambda: facilitator)


def snapshot(database):
    skill = database.get_skill_by_repo(REPO)
    renewals = database._conn.execute(
        "SELECT COUNT(*) FROM sentinel_events WHERE event_type = 'renewal'"
    ).fetchone()[0]
    return {
        "plan_expires_at": skill["plan_expires_at"],
        "owner_payer_ref": skill["owner_payer_ref"],
        "renewals": renewals,
    }


def renew(client, *, with_payment=True):
    return client.post(
        "/sentinel/renew",
        json={"repo_full_name": REPO, "owner_ref": "owner"},
        headers={"X-PAYMENT": "dummy"} if with_payment else {},
    )


def expected_atomic_49():
    config = get_config()
    network = config.network[0] if isinstance(config.network, list) else config.network
    return str(49 * 10 ** get_default_asset_config(network).decimals)


# --- matching payer ---------------------------------------------------------

def test_matching_payer_settles_and_renews(db, client, monkeypatch):
    register_skill(db, owner_payer_ref=OWNER)
    facilitator = FakeFacilitator(payer=OWNER)
    use_facilitator(monkeypatch, facilitator)
    before = snapshot(db)

    response = renew(client)

    assert response.status_code == 200
    assert facilitator.calls == ["verify", "settle"]
    after = snapshot(db)
    assert after["plan_expires_at"] > before["plan_expires_at"]
    assert after["renewals"] == before["renewals"] + 1
    assert after["owner_payer_ref"] == OWNER  # never rewritten by renewal


def test_matching_payer_with_different_address_case_is_accepted(db, client, monkeypatch):
    register_skill(db, owner_payer_ref=OWNER)
    facilitator = FakeFacilitator(payer="0x" + "AB" * 20)
    use_facilitator(monkeypatch, facilitator)

    response = renew(client)

    assert response.status_code == 200
    assert facilitator.calls == ["verify", "settle"]
    assert snapshot(db)["owner_payer_ref"] == OWNER


# --- rejected before settlement --------------------------------------------

def test_mismatched_payer_is_rejected_before_settlement(db, client, monkeypatch):
    register_skill(db, owner_payer_ref=OWNER)
    facilitator = FakeFacilitator(payer=OTHER)
    use_facilitator(monkeypatch, facilitator)
    before = snapshot(db)

    response = renew(client)

    assert response.status_code == 403
    assert facilitator.calls == ["verify"]  # settle never called
    assert snapshot(db) == before  # no extension, no renewal event, owner unchanged


def test_missing_stored_owner_payer_fails_closed(db, client, monkeypatch):
    register_skill(db, owner_payer_ref=None)  # legacy registration
    facilitator = FakeFacilitator(payer=OWNER)
    use_facilitator(monkeypatch, facilitator)
    before = snapshot(db)
    assert before["owner_payer_ref"] is None

    response = renew(client)

    assert response.status_code == 409
    assert facilitator.calls == ["verify"]
    assert snapshot(db) == before  # owner_payer_ref is not back-filled either


@pytest.mark.parametrize("missing_payer", [None, ""])
def test_missing_verified_payer_is_rejected_before_settlement(
    db, client, monkeypatch, missing_payer
):
    register_skill(db, owner_payer_ref=OWNER)
    facilitator = FakeFacilitator(payer=missing_payer)
    use_facilitator(monkeypatch, facilitator)
    before = snapshot(db)

    response = renew(client)

    assert response.status_code == 402
    assert facilitator.calls == ["verify"]
    assert snapshot(db) == before


# --- existing x402 behavior stays intact -----------------------------------

def test_settlement_failure_for_matching_payer_does_not_renew(db, client, monkeypatch):
    register_skill(db, owner_payer_ref=OWNER)
    facilitator = FakeFacilitator(payer=OWNER, settle_ok=False)
    use_facilitator(monkeypatch, facilitator)
    before = snapshot(db)

    response = renew(client)

    assert response.status_code == 402
    assert facilitator.calls == ["verify", "settle"]
    assert snapshot(db) == before


def test_invalid_payment_is_402_with_facilitator_error(db, client, monkeypatch):
    register_skill(db, owner_payer_ref=OWNER)
    facilitator = FakeFacilitator(payer=OWNER, valid=False)
    use_facilitator(monkeypatch, facilitator)
    before = snapshot(db)

    response = renew(client)

    assert response.status_code == 402
    assert response.json()["error"] == "invalid signature"
    assert facilitator.calls == ["verify"]
    assert snapshot(db) == before


def test_missing_payment_header_returns_402_challenge(db, client, monkeypatch):
    register_skill(db, owner_payer_ref=OWNER)
    facilitator = FakeFacilitator()
    use_facilitator(monkeypatch, facilitator)
    before = snapshot(db)

    response = renew(client, with_payment=False)

    assert response.status_code == 402
    body = response.json()
    assert body["error"] == "X-PAYMENT header is required"
    assert body["accepts"][0]["maxAmountRequired"] == expected_atomic_49()
    assert facilitator.calls == []
    assert snapshot(db) == before


def test_unregistered_repo_is_404_before_any_payment_call(db, client, monkeypatch):
    facilitator = FakeFacilitator()
    use_facilitator(monkeypatch, facilitator)

    response = renew(client)

    assert response.status_code == 404
    assert facilitator.calls == []


def test_register_route_still_issues_the_402_challenge(db, client, monkeypatch):
    # /sentinel/register shares the subscription price helper; it must keep
    # answering with a 402 challenge for the same amount (not a 500).
    facilitator = FakeFacilitator()
    use_facilitator(monkeypatch, facilitator)

    response = client.post(
        "/sentinel/register",
        json={"repo_full_name": "acme/new-skill", "owner_ref": "owner"},
    )

    assert response.status_code == 402
    assert response.json()["accepts"][0]["maxAmountRequired"] == expected_atomic_49()
    assert facilitator.calls == []
