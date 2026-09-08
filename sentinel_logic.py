"""Business logic helpers for DCL Update Sentinel."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from collections import defaultdict
from typing import Any, Optional

import httpx

from sentinel_audit import AuditOutcome, audit_repo_release
from sentinel_db import SentinelDB

RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_CALLS = 30
_rate_limit_hits: dict[str, list[float]] = defaultdict(list)


def default_policy(policy: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    merged = {"score_threshold": 0.80}
    if policy:
        merged.update(policy)
    if "score_threshold" not in merged:
        merged["score_threshold"] = 0.80
    return merged


def is_regression(verdict: str, score: float, policy: dict[str, Any]) -> bool:
    threshold = float(policy.get("score_threshold", 0.80))
    return verdict == "FAIL" or score < threshold


def check_rate_limit(key: str) -> None:
    now = time.time()
    hits = _rate_limit_hits[key]
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    while hits and hits[0] < cutoff:
        hits.pop(0)
    if len(hits) >= RATE_LIMIT_MAX_CALLS:
        raise ValueError(
            f"Rate limit exceeded: max {RATE_LIMIT_MAX_CALLS} calls per "
            f"{RATE_LIMIT_WINDOW_SECONDS}s. Try again shortly."
        )
    hits.append(now)


def verify_github_signature(body: bytes, secret: str, signature_header: Optional[str]) -> bool:
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    provided = signature_header.split("=", 1)[1]
    return hmac.compare_digest(expected, provided)


async def notify_owner(owner_ref: str, payload: dict[str, Any]) -> None:
    if not owner_ref.startswith(("http://", "https://")):
        return
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(owner_ref, json=payload)
    except Exception:
        pass


def new_webhook_secret() -> str:
    return secrets.token_urlsafe(32)


def audit_to_dict(outcome: AuditOutcome) -> dict[str, Any]:
    return {
        "version": outcome.version,
        "verdict": outcome.verdict,
        "score": outcome.score,
        "reason": outcome.reason,
        "tx_hash": outcome.tx_hash,
        "chain_index": outcome.chain_index,
        "audited_at": outcome.audited_at.isoformat(),
        "findings_summary": outcome.findings_summary,
    }


async def run_baseline_audit(repo_full_name: str, chain, scan_type: str = "update_rescan") -> AuditOutcome:
    return await audit_repo_release(repo_full_name, chain, scan_type=scan_type)


async def apply_webhook_scan(
    db: SentinelDB,
    skill: dict[str, Any],
    chain,
    *,
    version: Optional[str] = None,
    delivery_id: Optional[str] = None,
) -> dict[str, Any]:
    repo = skill["repo_full_name"]
    outcome = await audit_repo_release(repo, chain, scan_type="update_rescan", version=version)
    policy = db.get_policy(skill)
    now = outcome.audited_at

    db.update_current(
        skill["id"],
        version=outcome.version,
        verdict=outcome.verdict,
        score=outcome.score,
        audited_at=now,
    )

    if is_regression(outcome.verdict, outcome.score, policy):
        db.block_skill(skill["id"])
        inserted = db.insert_event(
            skill_id=skill["id"],
            event_type="regression_blocked",
            scan_type="update_rescan",
            version=outcome.version,
            verdict=outcome.verdict,
            score=outcome.score,
            delivery_id=delivery_id,
        )
        if inserted:
            await notify_owner(
                skill["owner_ref"],
                {
                    "event": "regression_blocked",
                    "repo": repo,
                    "version": outcome.version,
                    "verdict": outcome.verdict,
                    "score": outcome.score,
                    "baseline_version": skill.get("immutable_baseline_version"),
                    "last_known_good_version": skill.get("last_known_good_version"),
                },
            )
        return {
            "action": "blocked",
            "regression": True,
            "audit": audit_to_dict(outcome),
        }

    was_blocked = skill.get("status") == "blocked"

    db.promote_last_known_good(
        skill["id"],
        version=outcome.version,
        verdict=outcome.verdict,
        score=outcome.score,
        audited_at=now,
    )

    if was_blocked:
        # recover_skill() is the only normal path that clears a security
        # block — a clean rescan after regression must explicitly recover
        # the skill, or it would stay blocked forever.
        db.recover_skill(skill["id"])

    db.insert_event(
        skill_id=skill["id"],
        event_type="rescan_recovered" if was_blocked else "rescan_auto",
        scan_type="update_rescan",
        version=outcome.version,
        verdict=outcome.verdict,
        score=outcome.score,
        delivery_id=delivery_id,
    )
    return {
        "action": "recovered" if was_blocked else "promoted",
        "regression": False,
        "audit": audit_to_dict(outcome),
    }


def parse_github_release_version(payload: dict[str, Any]) -> Optional[str]:
    if payload.get("action") not in (None, "published", "released", "created"):
        return None
    release = payload.get("release") or {}
    return release.get("tag_name")
