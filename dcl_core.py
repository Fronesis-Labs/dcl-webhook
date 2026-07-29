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


from datetime import datetime, timezone


def format_seal(tx_hash: str, input_hash: str, timestamp: float) -> dict:
    """Produces the Leibniz Layer branded seal — one source of truth for both servers."""
    hash_display = tx_hash[2:] if tx_hash.startswith("0x") else tx_hash
    intent_display = input_hash[2:] if input_hash.startswith("0x") else input_hash
    sealed = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    verify_url = f"https://x402.fronesislabs.com/verify/{hash_display}"
    seal_text = (
        "🔒 Verified by Leibniz Layer | Fronesis Labs\n"
        f"Hash: {hash_display}\n"
        f"Intent: {intent_display}\n"
        f"Sealed: {sealed} — Base Mainnet\n"
        f"Verify: {verify_url}"
    )
    return {"seal_text": seal_text, "verify_url": verify_url}


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


import re


def _redact(s: str, keep_start: int = 2, keep_end: int = 4) -> str:
    """Mask a matched secret/PII string, keeping only a few edge chars."""
    if len(s) <= keep_start + keep_end:
        return "*" * len(s)
    middle = "*" * max(4, len(s) - keep_start - keep_end)
    return f"{s[:keep_start]}{middle}{s[-keep_end:]}"


def _luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(digits[::-1]):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


# ─── Secret Leak Detector patterns (S1-S8) ─────────────────────────────────
# (category, type, provider, compiled regex, severity)
SECRET_PATTERNS = [
    ("S1", "api_key", "openai",       re.compile(r"\bsk-(?:proj-|org-)?[A-Za-z0-9_-]{20,}\b")),
    ("S1", "api_key", "anthropic",    re.compile(r"\bsk-ant-[A-Za-z0-9\-_]{20,}\b")),
    ("S1", "api_key", "stripe_live",  re.compile(r"\bsk_live_[A-Za-z0-9]{24,}\b")),
    ("S1", "api_key", "github_pat",   re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("S1", "api_key", "slack",        re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("S1", "api_key", "sendgrid",     re.compile(r"\bSG\.[A-Za-z0-9_\-]{22}\.[A-Za-z0-9_\-]{43}\b")),
    ("S1", "api_key", "twilio_sid",   re.compile(r"\bAC[a-f0-9]{32}\b")),
    ("S2", "cloud_credential", "aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("S2", "cloud_credential", "aws_secret",      re.compile(r"(?i)aws_secret_access_key[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9/+=]{40}")),
    ("S2", "cloud_credential", "gcp_service_account", re.compile(r"\"private_key\"\s*:\s*\"-----BEGIN")),
    ("S3", "token", "jwt",    re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")),
    ("S3", "token", "bearer", re.compile(r"(?i)bearer\s+[A-Za-z0-9\-_.=]{20,}")),
    ("S4", "private_key_pem", None, re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----")),
    ("S5", "database_url", None, re.compile(r"(?i)\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^:\s]+:[^@\s]+@[^\s/]+")),
    ("S6", "connection_string", None, re.compile(r"(?i)(?:User ID|Uid)\s*=\s*[^;]+;\s*(?:Password|Pwd)\s*=\s*[^;]+")),
    ("S6", "env_assignment", None, re.compile(r"(?im)^[A-Za-z0-9_]*(?:KEY|SECRET|TOKEN|PASS|PWD|CREDENTIAL|AUTH)[A-Za-z0-9_]*\s*=\s*\S+")),
    ("S7", "webhook_secret", "stripe", re.compile(r"\bwhsec_[A-Za-z0-9]{32,}\b")),
    ("S8", "internal_endpoint", None, re.compile(r"(?i)[?&](?:api_key|apikey|token|secret|access_token)=[A-Za-z0-9\-_.]{8,}")),
]

# Severity per category (matches the S1-S8 checklist docs)
_SECRET_SEVERITY = {
    "S1": "critical", "S2": "critical", "S3": "critical", "S4": "critical",
    "S5": "critical", "S6": "major", "S7": "major", "S8": "major",
}


def detect_secrets(text: str) -> dict:
    """Real regex-based scan for credentials/secrets. Categories S1-S8."""
    findings = []
    categories_hit = set()
    for cat, typ, provider, pattern in SECRET_PATTERNS:
        for m in pattern.finditer(text):
            findings.append({
                "type": typ,
                "provider": provider,
                "position": m.start(),
                "redacted_sample": _redact(m.group(0)),
                "severity": _SECRET_SEVERITY[cat],
                "category": cat,
            })
            categories_hit.add(cat)
    all_cats = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]
    verdict = "NO_COMMIT" if findings else "COMMIT"
    return {
        "verdict": verdict,
        "risk_score": round(min(1.0, 0.5 + 0.1 * len(findings)), 3) if findings else 0.0,
        "findings": findings,
        "detection_count": len(findings),
        "categories_checked": all_cats,
        "categories_clear": [c for c in all_cats if c not in categories_hit],
    }


# ─── PII Detector patterns (T1-T8) ─────────────────────────────────────────
PII_PATTERNS = [
    ("T1", "email",          re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "major"),
    ("T2", "phone",          re.compile(r"(?<!\w)\+\d{1,3}[\s\-]?\(?\d{2,4}\)?[\s\-]?\d{2,4}[\s\-]?\d{2,4}(?!\w)"), "major"),
    ("T3", "national_id",    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "critical"),
    ("T4", "bank_card",      re.compile(r"\b(?:\d[ -]?){13,19}\b"), "critical"),
    ("T5", "iban",           re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"), "critical"),
    ("T6", "crypto_address", re.compile(r"\b(?:0x[a-fA-F0-9]{40}|bc1[a-z0-9]{25,39}|[13][a-km-zA-HJ-NP-Z1-9]{25,34})\b"), "major"),
    ("T7", "ip_address",     re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b|\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b"), "minor"),
    ("T8", "passport",       re.compile(r"(?i)passport\s*(?:no\.?|number)?\s*[:#]?\s*[A-Z0-9]{6,9}\b"), "critical"),
]


def detect_pii(text: str) -> dict:
    """Real regex-based scan for personal data. Categories T1-T8.

    Note: T4 (bank_card) applies a Luhn checksum to cut false positives from
    generic long digit runs. T2 (phone) requires a leading '+' to avoid
    flagging arbitrary numeric sequences — local-format phone numbers without
    a country code are intentionally not matched to keep false positives low.
    """
    findings = []
    categories_hit = set()
    for cat, typ, pattern, severity in PII_PATTERNS:
        for m in pattern.finditer(text):
            match_str = m.group(0)
            if typ == "bank_card":
                digits = re.sub(r"[ -]", "", match_str)
                if not (13 <= len(digits) <= 19) or not _luhn_valid(digits):
                    continue
            findings.append({
                "type": typ,
                "position": m.start(),
                "redacted_sample": _redact(match_str),
                "severity": severity,
                "category": cat,
            })
            categories_hit.add(cat)
    all_cats = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]
    verdict = "NO_COMMIT" if findings else "COMMIT"
    return {
        "verdict": verdict,
        "risk_score": round(min(1.0, 0.4 + 0.1 * len(findings)), 3) if findings else 0.0,
        "findings": findings,
        "detection_count": len(findings),
        "categories_checked": all_cats,
        "categories_clear": [c for c in all_cats if c not in categories_hit],
    }


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