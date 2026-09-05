"""Live CLI authentication coverage; run with NPA_INTEGRATION_E2E=1.

Uses the operator-selected NPA_NEBIUS_PROFILE / NEBIUS_PROFILE (or the CLI's
active profile). Provider identity and token output must never reach pytest.
These tests do not change profiles or provision resources.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid

import pytest

pytestmark = pytest.mark.e2e


def _preflight(*options: str, env: dict[str, str]) -> tuple[int, dict]:
    result = subprocess.run(
        [
            sys.executable, "-m", "npa", "workbench", "health", "preflight",
            "--checks", "nebius", "--json", *options,
        ],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        payload = json.loads(result.stdout)
    except ValueError:
        pytest.fail("Authentication preflight did not return one JSON report")
    return result.returncode, payload


def test_authenticated_profile_ignores_stale_ambient_tokens() -> None:
    env = dict(os.environ)
    env["NEBIUS_IAM_TOKEN"] = "synthetic-expired-token"
    env["NEBIUS_IAM_TOKEN_FILE"] = "/nonexistent/synthetic-token-file"
    code, payload = _preflight(env=env)
    assert code == 0
    assert payload["ok"] is True
    # Exact public shape rules out leaked identity, token or profile metadata.
    assert payload["checks"] in [
        [{"name": "nebius", "status": "PASS", "details": [], "remedy": "",
          "summary": f"{source} Nebius CLI profile is authenticated."}]
        for source in ("Configured", "Default")
    ]


def test_missing_profile_fails_without_disclosing_its_name() -> None:
    env = dict(os.environ)
    missing = "npa-missing-auth-" + uuid.uuid4().hex
    env["NPA_NEBIUS_PROFILE"] = missing
    # NPA's explicit selector must win over a working NEBIUS_PROFILE.
    code, payload = _preflight(env=env)
    assert code == 1
    assert payload["ok"] is False
    assert payload["checks"][0]["status"] == "FAIL"
    assert missing not in json.dumps(payload)
    assert payload["checks"][0]["details"] == []


def test_offline_missing_profile_is_skipped() -> None:
    env = dict(os.environ)
    env["NPA_NEBIUS_PROFILE"] = "npa-missing-auth-" + uuid.uuid4().hex
    code, payload = _preflight("--offline", env=env)
    assert code == 0
    assert payload["ok"] is True
    assert payload["checks"][0]["status"] == "SKIP"
