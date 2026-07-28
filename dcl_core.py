"""
DCL Core — clean policy evaluation and tamper-evident chain logic.
Used by both webhook_server.py (REST+payments) and mcp_server.py (MCP).
"""
import hashlib
import math
import time
import uuid
import sqlite3
import json
from typing import Optional, Tuple


def sha256hex(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


class ChainState:
    """SQLite-backed tamper-evident chain. Stores METADATA ONLY."""
    GENESIS = "0" * 64

    def __init__(self, db_path: str = "dcl_chain.db"):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS chain (
                idx INTEGER PRIMARY KEY,
                tx_hash TEXT UNIQUE NOT NULL,
                prev_hash TEXT NOT NULL,
                verdict TEXT NOT NULL,
                input_hash TEXT NOT NULL,
                policy_hash TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                reason TEXT NOT NULL,
                confidence REAL NOT NULL,
                task_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                drift_context TEXT NOT NULL
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_tx_hash ON chain(tx_hash)")
        self._conn.commit()

    def append(self, verdict: str, input_hash: str, policy_hash: str,
               agent_id: str, reason: str, confidence: float, task_type: str) -> Tuple[str, int]:
        last = self._conn.execute("SELECT tx_hash FROM chain ORDER BY idx DESC LIMIT 1").fetchone()
        prev_hash = last[0] if last else self.GENESIS
        new_idx = self._conn.execute("SELECT COALESCE(MAX(idx), -1) + 1 FROM chain").fetchone()[0]

        content = f"{new_idx}:{verdict}:{input_hash}:{policy_hash}:{prev_hash}:{time.time()}"
        tx_hash = "0x" + sha256hex(content)[:32]
        drift_context = {"environment": "production-edge", "policy_version_hash": policy_hash}

        self._conn.execute("""
            INSERT INTO chain (idx, tx_hash, prev_hash, verdict, input_hash, policy_hash,
                             agent_id, reason, confidence, task_type, timestamp, drift_context)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (new_idx, tx_hash, prev_hash, verdict, input_hash, policy_hash,
              agent_id, reason, confidence, task_type, time.time(), json.dumps(drift_context)))
        self._conn.commit()
        return tx_hash, new_idx

    def get_by_tx(self, tx_hash: str) -> Optional[dict]:
        row = self._conn.execute("""
            SELECT idx, tx_hash, prev_hash, verdict, input_hash, policy_hash,
                   agent_id, reason, confidence, task_type, timestamp, drift_context
            FROM chain WHERE tx_hash = ?
        """, (tx_hash,)).fetchone()
        if not row:
            return None
        return {
            "index": row[0], "tx_hash": row[1], "prev_hash": row[2], "verdict": row[3],
            "input_hash": row[4], "policy_hash": row[5], "agent_id": row[6], "reason": row[7],
            "confidence": row[8], "task_type": row[9], "timestamp": row[10],
            "drift_context": json.loads(row[11])
        }

    def verify(self) -> Tuple[bool, Optional[int]]:
        rows = self._conn.execute("SELECT idx, prev_hash, tx_hash FROM chain ORDER BY idx").fetchall()
        for i, row in enumerate(rows):
            expected_prev = rows[i - 1][2] if i > 0 else self.GENESIS
            if row[1] != expected_prev:
                return False, row[0]
        return True, None

    def __len__(self):
        return self._conn.execute("SELECT COUNT(*) FROM chain").fetchone()[0]


BUILTIN_POLICIES = {
    "default": """
version: "1.0.0"
name: "DCL Default Policy"
thresholds:
  min_confidence: 0.7
forbidden_patterns:
  - "ignore previous instructions"
  - "jailbreak"
  - "bypass safety"
required_patterns: []
""",
    "anti_jailbreak": """
version: "1.0.0"
name: "Anti-Jailbreak"
thresholds:
  min_confidence: 0.8
forbidden_patterns:
  - "ignore previous instructions"
  - "jailbreak"
  - "bypass safety"
  - "pretend you are"
  - "act as if"
  - "DAN"
required_patterns: []
""",
    "safety": """
version: "1.0.0"
name: "Safety Policy"
thresholds:
  min_confidence: 0.75
forbidden_patterns:
  - "I cannot be held responsible"
  - "no guarantees"
required_patterns:
  - "AI"
""",
    "content_quality": """
version: "2.0.0"
name: "Content Quality Policy"
thresholds:
  min_confidence: 0.85
forbidden_patterns:
  - "guaranteed returns"
  - "risk-free"
  - "definitely will"
  - "100% accurate"
  - "there is no doubt"
  - "it is certain that"
  - "studies show"
  - "research proves"
  - "everyone knows"
  - "as an ai, i can confirm"
  - "i am certain"
  - "without a doubt"
required_patterns: []
""",
}


def evaluate_policy(response: str, policy_yaml: str) -> Tuple[str, float, str, str]:
    import yaml
    try:
        policy = yaml.safe_load(policy_yaml)
    except yaml.YAMLError as e:
        return "NO_COMMIT", 0.0, f"Policy parse error: {e}", "unknown"

    version = policy.get("version", "unknown")
    penalties = 0.0
    reasons = []

    for pat in policy.get("forbidden_patterns", []):
        if pat.lower() in response.lower():
            reasons.append(f"forbidden: '{pat}'")
            penalties += 0.4

    for pat in policy.get("required_patterns", []):
        if pat.lower() not in response.lower():
            reasons.append(f"missing required: '{pat}'")
            penalties += 0.2

    confidence = max(0.0, 1.0 - penalties)
    min_conf = policy.get("thresholds", {}).get("min_confidence", 0.7)

    verdict = "COMMIT"
    reason = "All policy checks passed"
    if confidence < min_conf or reasons:
        verdict = "NO_COMMIT"
        reason = "; ".join(reasons) if reasons else f"Confidence {confidence} below threshold {min_conf}"

    return verdict, round(confidence, 3), reason, version


def get_drift_mode(commit_rate: list) -> Tuple[str, float]:
    n = len(commit_rate)
    if n < 5:
        return "NORMAL", 0.0
    window = min(10, n)
    baseline_vals = commit_rate[:-window]
    if not baseline_vals:
        return "NORMAL", 0.0
    baseline = sum(baseline_vals) / len(baseline_vals)
    baseline = min(max(baseline, 0.01), 0.99)  # avoid baseline*(1-baseline)==0 -> ZeroDivisionError
    current = sum(commit_rate[-window:]) / window
    z = (current - baseline) / math.sqrt(baseline * (1 - baseline) / window)
    abs_z = abs(z)
    if abs_z > 3.5:
        return "BLOCK", round(z, 2)
    elif abs_z > 2.5:
        return "ESCALATION", round(z, 2)
    elif abs_z > 1.96:
        return "WARNING", round(z, 2)
    return "NORMAL", round(z, 2)