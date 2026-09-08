"""SQLite persistence for DCL Update Sentinel (skills + sentinel_events)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

DEFAULT_SCORE_THRESHOLD = 0.80
SUBSCRIPTION_DAYS = 30


_SCHEMA = """
CREATE TABLE IF NOT EXISTS skills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_full_name TEXT NOT NULL UNIQUE,
    owner_ref TEXT NOT NULL,

    immutable_baseline_version TEXT,
    immutable_baseline_verdict TEXT,
    immutable_baseline_score REAL,
    immutable_baseline_audited_at TIMESTAMP,

    last_known_good_version TEXT,
    last_known_good_verdict TEXT,
    last_known_good_score REAL,
    last_known_good_audited_at TIMESTAMP,

    current_version TEXT,
    current_verdict TEXT,
    current_score REAL,
    current_audited_at TIMESTAMP,

    status TEXT DEFAULT 'unregistered',
    block_reason TEXT,

    policy_json TEXT,
    plan_type TEXT,
    plan_expires_at TIMESTAMP,

    webhook_secret TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sentinel_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_id INTEGER REFERENCES skills(id),
    event_type TEXT,
    scan_type TEXT,
    version TEXT,
    verdict TEXT,
    score REAL,
    amount_paid REAL,
    payer_ref TEXT,
    delivery_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_skills_webhook_secret
    ON skills(webhook_secret);

CREATE INDEX IF NOT EXISTS idx_sentinel_events_skill_id
    ON sentinel_events(skill_id);
"""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _ts(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).replace(tzinfo=None).isoformat(
        timespec="seconds"
    )


def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


class SentinelDB:
    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")

        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Schema migration
    # ------------------------------------------------------------------

    def _migrate(self) -> None:
        """Apply additive migrations to existing Sentinel databases."""

        skill_columns = {
            row["name"]
            for row in self._conn.execute(
                "PRAGMA table_info(skills)"
            ).fetchall()
        }

        event_columns = {
            row["name"]
            for row in self._conn.execute(
                "PRAGMA table_info(sentinel_events)"
            ).fetchall()
        }

        if "owner_payer_ref" not in skill_columns:
            self._conn.execute(
                "ALTER TABLE skills ADD COLUMN owner_payer_ref TEXT"
            )

        if "delivery_id" not in event_columns:
            self._conn.execute(
                "ALTER TABLE sentinel_events ADD COLUMN delivery_id TEXT"
            )

        self._conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_sentinel_events_delivery_id
            ON sentinel_events(delivery_id)
            WHERE delivery_id IS NOT NULL
            """
        )

    # ------------------------------------------------------------------
    # Skill lookup
    # ------------------------------------------------------------------

    def get_skill_by_repo(
        self,
        repo_full_name: str,
    ) -> Optional[dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM skills WHERE repo_full_name = ?",
            (repo_full_name,),
        ).fetchone()

        return dict(row) if row else None

    def get_skill_by_webhook_secret(
        self,
        webhook_secret: str,
    ) -> Optional[dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM skills WHERE webhook_secret = ?",
            (webhook_secret,),
        ).fetchone()

        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def create_skill(
        self,
        *,
        repo_full_name: str,
        owner_ref: str,
        owner_payer_ref: Optional[str] = None,
        webhook_secret: str,
        policy: Optional[dict[str, Any]],
        version: str,
        verdict: str,
        score: float,
        audited_at: datetime,
        amount_paid: float = 49.0,
        payer_ref: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Create a monitored skill.

        IMPORTANT:
        The caller is responsible for ensuring that the initial audit
        satisfies the configured policy before calling this method.
        """

        policy_json = json.dumps(
            policy or {"score_threshold": DEFAULT_SCORE_THRESHOLD}
        )

        audited_ts = _ts(audited_at)
        plan_expires = _ts(
            _utcnow() + timedelta(days=SUBSCRIPTION_DAYS)
        )

        cur = self._conn.execute(
            """
            INSERT INTO skills (
                repo_full_name,
                owner_ref,
                owner_payer_ref,

                immutable_baseline_version,
                immutable_baseline_verdict,
                immutable_baseline_score,
                immutable_baseline_audited_at,

                last_known_good_version,
                last_known_good_verdict,
                last_known_good_score,
                last_known_good_audited_at,

                current_version,
                current_verdict,
                current_score,
                current_audited_at,

                status,
                block_reason,
                policy_json,
                plan_type,
                plan_expires_at,
                webhook_secret
            )
            VALUES (
                ?, ?,
                ?,

                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,

                'active',
                NULL,
                ?,
                'subscription',
                ?,
                ?
            )
            """,
            (
                repo_full_name,
                owner_ref,
                owner_payer_ref,

                version,
                verdict,
                score,
                audited_ts,

                version,
                verdict,
                score,
                audited_ts,

                version,
                verdict,
                score,
                audited_ts,

                policy_json,
                plan_expires,
                webhook_secret,
            ),
        )

        self._conn.commit()

        skill_id = cur.lastrowid

        self.insert_event(
            skill_id=skill_id,
            event_type="registration",
            scan_type=None,
            version=version,
            verdict=verdict,
            score=score,
            amount_paid=amount_paid,
            payer_ref=payer_ref or owner_payer_ref,
        )

        return self.get_skill_by_repo(repo_full_name)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Subscription / entitlement
    # ------------------------------------------------------------------

    def renew_subscription(
        self,
        skill_id: int,
        payer_ref: Optional[str],
        *,
        amount_paid: float = 49.0,
    ) -> datetime:
        """
        Extend monitoring entitlement.

        SECURITY RULE:
        Renewal NEVER changes security status and NEVER clears block_reason.

        A blocked skill therefore remains blocked after renewal until a
        clean rescan explicitly recovers it.
        """

        skill = self._get_skill(skill_id)

        now = _utcnow()

        current_expires = _parse_ts(
            skill.get("plan_expires_at")
        )

        base = (
            current_expires
            if current_expires and current_expires > now
            else now
        )

        new_expires = base + timedelta(days=SUBSCRIPTION_DAYS)

        self._conn.execute(
            """
            UPDATE skills
            SET plan_type = 'subscription',
                plan_expires_at = ?
            WHERE id = ?
            """,
            (
                _ts(new_expires),
                skill_id,
            ),
        )

        self._conn.commit()

        self.insert_event(
            skill_id=skill_id,
            event_type="renewal",
            amount_paid=amount_paid,
            payer_ref=payer_ref,
        )

        return new_expires

    def monitoring_active(
        self,
        skill: dict[str, Any],
    ) -> bool:
        """
        Whether the paid monitoring entitlement is active.

        Security status is intentionally NOT considered here.

        A paid skill can therefore be:
            monitoring_active=True
            status='blocked'

        This is required so Sentinel can continue monitoring a blocked
        skill and automatically recover it after a clean scan.
        """

        if skill.get("plan_type") != "subscription":
            return False

        expires = _parse_ts(
            skill.get("plan_expires_at")
        )

        if not expires:
            return False

        return expires > _utcnow()

    def subscription_active(
        self,
        skill: dict[str, Any],
    ) -> bool:
        """
        Backwards-compatible alias.

        Historically this method also checked status == active.
        That behavior is intentionally removed: subscription/monitoring
        entitlement is independent from security status.
        """

        return self.monitoring_active(skill)

    # ------------------------------------------------------------------
    # Current / baseline / LKG
    # ------------------------------------------------------------------

    def update_current(
        self,
        skill_id: int,
        *,
        version: str,
        verdict: str,
        score: float,
        audited_at: datetime,
    ) -> None:
        self._conn.execute(
            """
            UPDATE skills
            SET current_version = ?,
                current_verdict = ?,
                current_score = ?,
                current_audited_at = ?
            WHERE id = ?
            """,
            (
                version,
                verdict,
                score,
                _ts(audited_at),
                skill_id,
            ),
        )

        self._conn.commit()

    def promote_last_known_good(
        self,
        skill_id: int,
        *,
        version: str,
        verdict: str,
        score: float,
        audited_at: datetime,
    ) -> None:
        self._conn.execute(
            """
            UPDATE skills
            SET last_known_good_version = ?,
                last_known_good_verdict = ?,
                last_known_good_score = ?,
                last_known_good_audited_at = ?
            WHERE id = ?
            """,
            (
                version,
                verdict,
                score,
                _ts(audited_at),
                skill_id,
            ),
        )

        self._conn.commit()

    # ------------------------------------------------------------------
    # Security state
    # ------------------------------------------------------------------

    def block_skill(
        self,
        skill_id: int,
        *,
        reason: str = "security_regression",
    ) -> None:
        self._conn.execute(
            """
            UPDATE skills
            SET status = 'blocked',
                block_reason = ?
            WHERE id = ?
            """,
            (
                reason,
                skill_id,
            ),
        )

        self._conn.commit()

    def recover_skill(
        self,
        skill_id: int,
    ) -> None:
        """
        Recover a blocked skill after a clean policy-compliant rescan.

        This is the ONLY normal path that clears a security block.
        """

        self._conn.execute(
            """
            UPDATE skills
            SET status = 'active',
                block_reason = NULL
            WHERE id = ?
            """,
            (skill_id,),
        )

        self._conn.commit()

    # ------------------------------------------------------------------
    # Pay-per-call / unregistered scans
    # ------------------------------------------------------------------

    def upsert_current_for_unregistered(
        self,
        repo_full_name: str,
        *,
        version: str,
        verdict: str,
        score: float,
        audited_at: datetime,
    ) -> None:
        """Ensure a skill row exists so pay-per-call scans can update current_*."""

        existing = self.get_skill_by_repo(repo_full_name)

        if existing:
            self.update_current(
                existing["id"],
                version=version,
                verdict=verdict,
                score=score,
                audited_at=audited_at,
            )
            return

        audited_ts = _ts(audited_at)

        self._conn.execute(
            """
            INSERT INTO skills (
                repo_full_name,
                owner_ref,
                current_version,
                current_verdict,
                current_score,
                current_audited_at,
                status,
                policy_json
            )
            VALUES (
                ?,
                'pay-per-call',
                ?,
                ?,
                ?,
                ?,
                'unregistered',
                ?
            )
            """,
            (
                repo_full_name,
                version,
                verdict,
                score,
                audited_ts,
                json.dumps(
                    {"score_threshold": DEFAULT_SCORE_THRESHOLD}
                ),
            ),
        )

        self._conn.commit()

    # ------------------------------------------------------------------
    # Events / idempotency
    # ------------------------------------------------------------------

    def insert_event(
        self,
        *,
        skill_id: Optional[int],
        event_type: str,
        scan_type: Optional[str] = None,
        version: Optional[str] = None,
        verdict: Optional[str] = None,
        score: Optional[float] = None,
        amount_paid: Optional[float] = None,
        payer_ref: Optional[str] = None,
        delivery_id: Optional[str] = None,
    ) -> bool:
        """
        Insert an event.

        Returns:
            True  -> event inserted
            False -> delivery_id already existed

        delivery_id is intended for GitHub webhook idempotency.
        """

        try:
            self._conn.execute(
                """
                INSERT INTO sentinel_events (
                    skill_id,
                    event_type,
                    scan_type,
                    version,
                    verdict,
                    score,
                    amount_paid,
                    payer_ref,
                    delivery_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    skill_id,
                    event_type,
                    scan_type,
                    version,
                    verdict,
                    score,
                    amount_paid,
                    payer_ref,
                    delivery_id,
                ),
            )

            self._conn.commit()
            return True

        except sqlite3.IntegrityError:
            if delivery_id is not None:
                self._conn.rollback()
                return False

            raise

    def event_exists(
        self,
        delivery_id: str,
    ) -> bool:
        row = self._conn.execute(
            """
            SELECT 1
            FROM sentinel_events
            WHERE delivery_id = ?
            LIMIT 1
            """,
            (delivery_id,),
        ).fetchone()

        return row is not None

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    def get_policy(
        self,
        skill: dict[str, Any],
    ) -> dict[str, Any]:
        raw = skill.get("policy_json")

        if not raw:
            return {
                "score_threshold": DEFAULT_SCORE_THRESHOLD
            }

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {
                "score_threshold": DEFAULT_SCORE_THRESHOLD
            }

        if "score_threshold" not in data:
            data["score_threshold"] = DEFAULT_SCORE_THRESHOLD

        return data

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status_payload(
        self,
        repo_full_name: str,
    ) -> dict[str, Any]:
        skill = self.get_skill_by_repo(repo_full_name)

        if not skill:
            return {
                "registered": False,
                "monitoring_active": False,
                "status": "unregistered",
                "reason": None,
                "immutable_baseline_version": None,
                "last_known_good_version": None,
                "current_version": None,
                "last_scanned_at": None,
                "plan_type": None,
                "plan_expires_at": None,
            }

        monitoring = self.monitoring_active(skill)

        return {
            "registered": skill.get("plan_type") == "subscription",
            "monitoring_active": monitoring,
            "status": skill.get("status"),
            "reason": skill.get("block_reason"),

            "immutable_baseline_version": skill.get(
                "immutable_baseline_version"
            ),

            "last_known_good_version": skill.get(
                "last_known_good_version"
            ),

            "current_version": skill.get(
                "current_version"
            ),

            "last_scanned_at": skill.get(
                "current_audited_at"
            ),

            "plan_type": skill.get("plan_type"),
            "plan_expires_at": skill.get(
                "plan_expires_at"
            ),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_skill(
        self,
        skill_id: int,
    ) -> dict[str, Any]:
        row = self._conn.execute(
            "SELECT * FROM skills WHERE id = ?",
            (skill_id,),
        ).fetchone()

        if not row:
            raise KeyError(
                f"skill {skill_id} not found"
            )

        return dict(row)