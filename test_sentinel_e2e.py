#!/usr/bin/env python3
"""E2E tests for DCL Update Sentinel (offline — mocked GitHub + audit)."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

os.environ.setdefault("X402_WALLET", "0xb790ed3796194E5511C44411CF045F67E069cdC0")

import webhook_server  # noqa: E402
from sentinel_audit import AuditOutcome
from sentinel_db import SentinelDB
from sentinel_logic import is_regression


def _make_outcome(version: str, verdict: str, score: float) -> AuditOutcome:
    return AuditOutcome(
        version=version,
        verdict=verdict,
        score=score,
        reason="test",
        tx_hash="0xtest",
        chain_index=1,
        audited_at=datetime.now(timezone.utc),
        findings_summary={},
    )


class SentinelRegressionTests(unittest.TestCase):
    def test_fail_verdict_is_regression(self):
        self.assertTrue(is_regression("FAIL", 0.95, {"score_threshold": 0.80}))

    def test_low_score_is_regression(self):
        self.assertTrue(is_regression("PASS", 0.71, {"score_threshold": 0.80}))

    def test_pass_above_threshold_not_regression(self):
        self.assertFalse(is_regression("PASS", 0.94, {"score_threshold": 0.80}))


class SentinelDBTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.db = SentinelDB(self._tmp.name)

    def tearDown(self):
        self.db.close()
        os.unlink(self._tmp.name)

    def test_three_version_states_on_register(self):
        now = datetime.now(timezone.utc)
        skill = self.db.create_skill(
            repo_full_name="acme/skill",
            owner_ref="https://example.com/hook",
            webhook_secret="sec123",
            policy={"score_threshold": 0.80},
            version="2.4.0",
            verdict="PASS",
            score=0.94,
            audited_at=now,
        )
        self.assertEqual(skill["immutable_baseline_version"], "2.4.0")
        self.assertEqual(skill["last_known_good_version"], "2.4.0")
        self.assertEqual(skill["current_version"], "2.4.0")
        self.assertEqual(skill["status"], "active")

    def test_regression_blocks_without_touching_baseline(self):
        now = datetime.now(timezone.utc)
        skill = self.db.create_skill(
            repo_full_name="acme/skill",
            owner_ref="https://example.com/hook",
            webhook_secret="sec123",
            policy={"score_threshold": 0.80},
            version="2.4.0",
            verdict="PASS",
            score=0.94,
            audited_at=now,
        )
        self.db.update_current(
            skill["id"],
            version="2.4.1",
            verdict="PASS",
            score=0.71,
            audited_at=now,
        )
        self.db.block_skill(skill["id"])
        updated = self.db.get_skill_by_repo("acme/skill")
        assert updated is not None
        self.assertEqual(updated["immutable_baseline_version"], "2.4.0")
        self.assertEqual(updated["last_known_good_version"], "2.4.0")
        self.assertEqual(updated["current_version"], "2.4.1")
        self.assertEqual(updated["status"], "blocked")
        self.assertEqual(updated["block_reason"], "security_regression")

    def test_promote_updates_last_known_good_only(self):
        now = datetime.now(timezone.utc)
        skill = self.db.create_skill(
            repo_full_name="acme/skill",
            owner_ref="https://example.com/hook",
            webhook_secret="sec123",
            policy={"score_threshold": 0.80},
            version="2.4.0",
            verdict="PASS",
            score=0.94,
            audited_at=now,
        )
        later = now + timedelta(hours=1)
        self.db.update_current(
            skill["id"],
            version="2.4.1",
            verdict="PASS",
            score=0.91,
            audited_at=later,
        )
        self.db.promote_last_known_good(
            skill["id"],
            version="2.4.1",
            verdict="PASS",
            score=0.91,
            audited_at=later,
        )
        updated = self.db.get_skill_by_repo("acme/skill")
        assert updated is not None
        self.assertEqual(updated["immutable_baseline_version"], "2.4.0")
        self.assertEqual(updated["last_known_good_version"], "2.4.1")
        self.assertEqual(updated["current_version"], "2.4.1")


class SentinelWebhookFlowTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        webhook_server._sentinel_db = SentinelDB(self._tmp.name)
        self.client = TestClient(webhook_server.app)
        now = datetime.now(timezone.utc)
        self.skill = webhook_server._sentinel_db.create_skill(
            repo_full_name="acme/skill",
            owner_ref="https://example.com/hook",
            webhook_secret="whsec_test_secret",
            policy={"score_threshold": 0.80},
            version="2.4.0",
            verdict="PASS",
            score=0.94,
            audited_at=now,
        )

    def tearDown(self):
        webhook_server._sentinel_db.close()
        os.unlink(self._tmp.name)

    def _signed_post(self, payload: dict) -> TestClient:
        body = json.dumps(payload).encode()
        sig = hmac.new(
            b"whsec_test_secret",
            body,
            hashlib.sha256,
        ).hexdigest()
        return self.client.post(
            "/sentinel/webhook/whsec_test_secret",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": f"sha256={sig}",
            },
        )

    @patch("sentinel_logic.audit_repo_release", new_callable=AsyncMock)
    def test_non_regression_promotes_last_known_good(self, mock_audit):
        mock_audit.return_value = _make_outcome("2.4.1", "PASS", 0.91)
        resp = self._signed_post(
            {"action": "published", "release": {"tag_name": "2.4.1"}},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data["regression"])
        skill = webhook_server._sentinel_db.get_skill_by_repo("acme/skill")
        assert skill is not None
        self.assertEqual(skill["last_known_good_version"], "2.4.1")
        self.assertEqual(skill["immutable_baseline_version"], "2.4.0")
        self.assertEqual(skill["status"], "active")

    @patch("sentinel_logic.audit_repo_release", new_callable=AsyncMock)
    def test_regression_blocks_without_changing_baseline(self, mock_audit):
        mock_audit.return_value = _make_outcome("2.4.1", "PASS", 0.71)
        resp = self._signed_post(
            {"action": "published", "release": {"tag_name": "2.4.1"}},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["regression"])
        skill = webhook_server._sentinel_db.get_skill_by_repo("acme/skill")
        assert skill is not None
        self.assertEqual(skill["status"], "blocked")
        self.assertEqual(skill["last_known_good_version"], "2.4.0")
        self.assertEqual(skill["immutable_baseline_version"], "2.4.0")
        self.assertEqual(skill["current_version"], "2.4.1")


if __name__ == "__main__":
    unittest.main()
