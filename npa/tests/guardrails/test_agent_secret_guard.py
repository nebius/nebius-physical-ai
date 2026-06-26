from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
GITIGNORE = REPO_ROOT / ".gitignore"
AGENT_PATH = REPO_ROOT / "npa" / "src" / "npa" / "cli" / "agent.py"

FORBIDDEN_TRACKED_SUFFIXES = (
    "auth.env",
    ".npa/credentials.yaml",
    ".npa/config.yaml",
)
FORBIDDEN_TRACKED_PREFIXES = (
    ".cursor/",
    ".npa/agents/",
)
# RFC 5737 documentation ranges + common test placeholders only.
ALLOWED_IPV4 = {
    "0.0.0.0",
    "127.0.0.1",
    "8.8.8.8",
    "203.0.113.50",
}
LITERAL_SECRET_PATTERNS = (
    re.compile(r"AGENT_PASSWORD\s*=\s*(?!\{)[^\s\n\"']{8,}"),
    re.compile(r"NEBIUS_TOKEN_FACTORY_KEY\s*[:=]\s*v1\.[A-Za-z0-9]{8,}"),
    re.compile(r"aws_secret_access_key\s*[:=]\s*(?!\{)[^\s\n\"']{8,}", re.IGNORECASE),
    re.compile(r"nvapi_[A-Za-z0-9]{8,}"),
)


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _gitignore_covers(fragment: str) -> bool:
    text = GITIGNORE.read_text(encoding="utf-8")
    return fragment in text


def test_gitignore_blocks_agent_and_cursor_secrets() -> None:
    assert _gitignore_covers("**/auth.env")
    assert _gitignore_covers(".npa/agents/")
    assert _gitignore_covers(".cursor/")


def test_tracked_files_exclude_agent_secret_paths() -> None:
    tracked = _tracked_files()
    violations: list[str] = []
    for path in tracked:
        if any(path.endswith(suffix) for suffix in FORBIDDEN_TRACKED_SUFFIXES):
            violations.append(path)
        if any(path.startswith(prefix) for prefix in FORBIDDEN_TRACKED_PREFIXES):
            violations.append(path)
    assert not violations, "Tracked secret/cursor paths:\n" + "\n".join(violations)


def test_agent_bootstrap_does_not_commit_generated_passwords() -> None:
    source = AGENT_PATH.read_text(encoding="utf-8")
    assert "secrets.token_urlsafe" in source
    assert "redact_value(auth_password)" in source
    assert "_write_auth_secret" in source
    assert "auth_secret_path" in source
    for pattern in LITERAL_SECRET_PATTERNS:
        assert not pattern.search(source), f"literal secret pattern in agent.py: {pattern.pattern}"


def test_agent_tracked_files_have_no_literal_secrets_or_live_ips() -> None:
    tracked = [
        path
        for path in _tracked_files()
        if path.startswith("npa/") and ("/agent" in path or path.endswith("agent.py"))
    ]
    violations: list[str] = []
    for rel in tracked:
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        for pattern in LITERAL_SECRET_PATTERNS:
            if pattern.search(text):
                violations.append(f"{rel}: {pattern.pattern}")
        for match in re.finditer(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text):
            ip = match.group(0)
            if ip not in ALLOWED_IPV4:
                violations.append(f"{rel}: unexpected IPv4 {ip}")
    assert not violations, "\n".join(violations)
