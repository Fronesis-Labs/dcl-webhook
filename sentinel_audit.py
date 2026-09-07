"""GitHub release fetch + DCL audit pipeline for Update Sentinel."""

from __future__ import annotations

import io
import os
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx

from dcl_core import ChainState, sha256hex
from audit_logic import BUILTIN_POLICIES, detect_pii, detect_secrets, evaluate_policy

TEXT_EXTENSIONS = {
    ".py", ".md", ".txt", ".js", ".ts", ".tsx", ".jsx", ".json", ".yaml", ".yml",
    ".sh", ".bash", ".toml", ".ini", ".cfg", ".env.example", ".sql", ".html", ".css",
    ".rs", ".go", ".java", ".kt", ".rb", ".php", ".swift", ".cs", ".vue", ".mjs", ".cjs",
}
MAX_FILE_BYTES = 512_000
MAX_TOTAL_BYTES = 5_000_000

SCAN_POLICIES = {
    "update_rescan": ["default"],
    "deep_scan": ["default", "strict", "anti_jailbreak"],
    "forensic_audit": ["default", "strict", "anti_jailbreak", "content_quality", "safety"],
}


@dataclass
class AuditOutcome:
    version: str
    verdict: str
    score: float
    reason: str
    tx_hash: str
    chain_index: int
    audited_at: datetime
    findings_summary: dict


def _github_headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "dcl-update-sentinel"}
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _extract_text_files(tar_bytes: bytes) -> dict[str, str]:
    files: dict[str, str] = {}
    total = 0
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            name = member.name.split("/", 1)[-1] if "/" in member.name else member.name
            ext = os.path.splitext(name)[1].lower()
            if ext not in TEXT_EXTENSIONS and name not in ("SKILL.md", "LICENSE", "Dockerfile"):
                continue
            if member.size > MAX_FILE_BYTES:
                continue
            extracted = archive.extractfile(member)
            if not extracted:
                continue
            raw = extracted.read()
            if total + len(raw) > MAX_TOTAL_BYTES:
                break
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue
            files[name] = text
            total += len(raw)
    return files


async def fetch_release_tarball(repo_full_name: str, version: Optional[str] = None) -> tuple[str, bytes]:
    owner, repo = repo_full_name.split("/", 1)
    async with httpx.AsyncClient(timeout=120.0, headers=_github_headers()) as client:
        if version:
            tag = version.lstrip("v")
            release_resp = await client.get(
                f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"
            )
            if release_resp.status_code == 404:
                release_resp = await client.get(
                    f"https://api.github.com/repos/{owner}/{repo}/releases/tags/v{tag}"
                )
            if release_resp.status_code != 200:
                raise ValueError(f"Release {version} not found for {repo_full_name}")
            release = release_resp.json()
            tarball_url = release["tarball_url"]
            resolved_version = release.get("tag_name") or version
        else:
            release_resp = await client.get(
                f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
            )
            if release_resp.status_code == 200:
                release = release_resp.json()
                tarball_url = release["tarball_url"]
                resolved_version = release.get("tag_name") or "latest"
            else:
                tarball_url = f"https://api.github.com/repos/{owner}/{repo}/tarball/HEAD"
                resolved_version = "HEAD"

        tar_resp = await client.get(tarball_url, follow_redirects=True)
        tar_resp.raise_for_status()
        return resolved_version, tar_resp.content


def run_repo_audit(
    repo_full_name: str,
    files: dict[str, str],
    *,
    scan_type: str,
    version: str,
    chain: ChainState,
) -> AuditOutcome:
    if not files:
        raise ValueError(f"No auditable text files found for {repo_full_name}@{version}")

    combined = "\n\n".join(f"# file: {path}\n{content}" for path, content in sorted(files.items()))
    policies = SCAN_POLICIES.get(scan_type, SCAN_POLICIES["update_rescan"])

    secrets = detect_secrets(combined)
    pii = detect_pii(combined)

    min_confidence = 1.0
    reasons: list[str] = []
    hard_fail = False

    for finding in secrets["findings"]:
        if finding["severity"] == "critical":
            hard_fail = True
            reasons.append(f"secret:{finding['type']}")
    for finding in pii["findings"]:
        if finding["severity"] == "critical":
            hard_fail = True
            reasons.append(f"pii:{finding['type']}")

    if secrets["verdict"] == "NO_COMMIT":
        min_confidence = min(min_confidence, 1.0 - secrets["risk_score"])
    if pii["verdict"] == "NO_COMMIT":
        min_confidence = min(min_confidence, 1.0 - pii["risk_score"])

    for policy_name in policies:
        policy_yaml = BUILTIN_POLICIES[policy_name]
        verdict, confidence, reason, _ = evaluate_policy(combined, policy_yaml)
        min_confidence = min(min_confidence, confidence)
        if verdict == "NO_COMMIT":
            hard_fail = True
            reasons.append(f"policy:{policy_name}:{reason}")

    risk = max(secrets["risk_score"], pii["risk_score"])
    score = round(max(0.0, min_confidence * (1.0 - risk * 0.35)), 3)
    verdict = "FAIL" if hard_fail else "PASS"
    reason = "; ".join(reasons) if reasons else "All checks passed"

    input_hash = "0x" + sha256hex(f"{repo_full_name}@{version}:{scan_type}")[:16]
    policy_hash = sha256hex(scan_type)[:16]
    tx_hash, chain_idx = chain.append(
        verdict=verdict,
        input_hash=input_hash,
        policy_hash=policy_hash,
        agent_id=repo_full_name,
        reason=reason[:500],
        confidence=score,
        task_type=f"sentinel_{scan_type}",
        drift_context={"scan_type": scan_type, "version": version, "file_count": len(files)},
    )

    return AuditOutcome(
        version=version,
        verdict=verdict,
        score=score,
        reason=reason,
        tx_hash=tx_hash,
        chain_index=chain_idx,
        audited_at=datetime.now(timezone.utc),
        findings_summary={
            "secrets": secrets["detection_count"],
            "pii": pii["detection_count"],
            "policies_checked": policies,
        },
    )


async def audit_repo_release(
    repo_full_name: str,
    chain: ChainState,
    *,
    scan_type: str = "update_rescan",
    version: Optional[str] = None,
) -> AuditOutcome:
    resolved_version, tarball = await fetch_release_tarball(repo_full_name, version)
    files = _extract_text_files(tarball)
    return run_repo_audit(
        repo_full_name,
        files,
        scan_type=scan_type,
        version=resolved_version,
        chain=chain,
    )
