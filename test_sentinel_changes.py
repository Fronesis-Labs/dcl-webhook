import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from sentinel_db import SentinelDB
from sentinel_logic import apply_webhook_scan
from sentinel_audit import AuditOutcome


@pytest.fixture
def db(tmp_path):
    db_file = str(tmp_path / "test_sentinel.db")
    s_db = SentinelDB(db_file)
    yield s_db
    s_db.close()


@pytest.fixture
def mock_chain():
    return MagicMock()


@pytest.fixture
def sample_skill(db):
    return db.create_skill(
        repo_full_name="owner/test-repo",
        owner_ref="https://example.com/webhook-owner",
        owner_payer_ref="0xPayer123",
        webhook_secret="secret_123",
        policy={"score_threshold": 0.80},
        version="v1.0.0",
        verdict="PASS",
        score=0.95,
        audited_at=datetime.now(timezone.utc),
    )


# ------------------------------------------------------------------
# 1. Delivery ID idempotency & conditional notify_owner
# ------------------------------------------------------------------
@pytest.mark.asyncio
async def test_regression_idempotency_and_notify_owner(db, sample_skill, mock_chain):
    clean_outcome = AuditOutcome(
        version="v1.1.0",
        verdict="FAIL",
        score=0.40,
        reason="Security regression detected",
        tx_hash="0x123",
        chain_index=1,
        audited_at=datetime.now(timezone.utc),
        findings_summary=[],
    )

    with patch("sentinel_logic.audit_repo_release", new_callable=AsyncMock) as mock_audit, \
         patch("sentinel_logic.notify_owner", new_callable=AsyncMock) as mock_notify:
        
        mock_audit.return_value = clean_outcome

        # First delivery attempt (event inserted)
        res1 = await apply_webhook_scan(
            db, sample_skill, mock_chain, version="v1.1.0", delivery_id="delivery-001"
        )
        assert res1["action"] == "blocked"
        assert mock_notify.call_count == 1

        # Duplicate delivery retry (should ignore notification)
        res2 = await apply_webhook_scan(
            db, sample_skill, mock_chain, version="v1.1.0", delivery_id="delivery-001"
        )
        assert res2["action"] == "blocked"
        assert mock_notify.call_count == 1


# ------------------------------------------------------------------
# 2. Auto-recovery logic (blocked -> active + rescan_recovered event)
# ------------------------------------------------------------------
@pytest.mark.asyncio
async def test_auto_recovery_from_blocked(db, sample_skill, mock_chain):
    # Transition skill into blocked state
    db.block_skill(sample_skill["id"], reason="security_regression")
    blocked_skill = db.get_skill_by_repo(sample_skill["repo_full_name"])
    assert blocked_skill["status"] == "blocked"

    # Simulate subsequent clean scan
    recovered_outcome = AuditOutcome(
        version="v1.2.0",
        verdict="PASS",
        score=0.90,
        reason="Clean scan",
        tx_hash="0x456",
        chain_index=2,
        audited_at=datetime.now(timezone.utc),
        findings_summary=[],
    )

    with patch("sentinel_logic.audit_repo_release", new_callable=AsyncMock) as mock_audit:
        mock_audit.return_value = recovered_outcome

        res = await apply_webhook_scan(
            db, blocked_skill, mock_chain, version="v1.2.0", delivery_id="delivery-002"
        )

        assert res["action"] == "recovered"
        
        # Verify status reset to active
        updated_skill = db.get_skill_by_repo(sample_skill["repo_full_name"])
        assert updated_skill["status"] == "active"
        assert updated_skill["block_reason"] is None

        # Verify correct event type logged
        events = db._conn.execute(
            "SELECT event_type FROM sentinel_events WHERE delivery_id = 'delivery-002'"
        ).fetchall()
        assert len(events) == 1
        assert events[0]["event_type"] == "rescan_recovered"


# ------------------------------------------------------------------
# 3. Verify request.state payment_payer propagation in /sentinel/scan
# ------------------------------------------------------------------
def test_sentinel_scan_payer_ref_integration(db):
    from webhook_server import app
    
    client = TestClient(app)

    async def mock_require_x402(request: Request, price: float):
        request.state.payment_payer = "0xVerifiedPayerAddress"
        return None

    clean_outcome = AuditOutcome(
        version="v1.0.0",
        verdict="PASS",
        score=0.99,
        reason="OK",
        tx_hash="0x789",
        chain_index=3,
        audited_at=datetime.now(timezone.utc),
        findings_summary=[],
    )

    with patch("webhook_server._sentinel_db", db), \
         patch("webhook_server.require_x402_payment", side_effect=mock_require_x402), \
         patch("webhook_server.audit_repo_release", new_callable=AsyncMock) as mock_audit:

        mock_audit.return_value = clean_outcome

        payload = {
            "repo_full_name": "owner/scan-test-repo",
            "scan_type": "update_rescan",
        }
        response = client.post("/sentinel/scan", json=payload)

        assert response.status_code == 200
        
        # Verify verified payer recorded in events table
        skill = db.get_skill_by_repo("owner/scan-test-repo")
        event = db._conn.execute(
            "SELECT payer_ref FROM sentinel_events WHERE skill_id = ?", (skill["id"],)
        ).fetchone()
        
        assert event["payer_ref"] == "0xVerifiedPayerAddress"