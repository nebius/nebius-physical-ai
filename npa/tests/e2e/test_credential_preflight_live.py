"""Live evidence that concurrent credential preflight checks work end to end.

Run with ``NPA_INTEGRATION_E2E=1`` and ``NPA_E2E_PROJECT`` set to an
explicitly configured project alias whose S3 storage and Nebius CLI profile
are both real (see ``skills/atomic/health-preflight/SKILL.md``). This proves
the concurrent orchestration in ``run_credential_preflight`` still reaches
every real provider and returns a coherent report; the wall-clock speedup
itself is covered by a separate, non-committed diagnostic comparison
(base vs candidate orchestration against the same real checks), since
comparing two source trees in one pytest run is a benchmark, not a
regression test.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.e2e


@pytest.fixture
def live_project() -> str:
    project = os.environ.get("NPA_E2E_PROJECT", "").strip()
    if not project:
        pytest.skip("Set NPA_E2E_PROJECT to an explicitly configured test project")
    return project


def _run_preflight(project: str, checks: str) -> tuple[int, dict, float]:
    started = time.perf_counter()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "npa",
            "workbench",
            "health",
            "preflight",
            "--project",
            project,
            "--checks",
            checks,
            "--json",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    elapsed = time.perf_counter() - started
    try:
        payload = json.loads(result.stdout)
    except ValueError:
        pytest.fail(
            f"Preflight did not return one JSON document: {result.stdout!r} {result.stderr!r}"
        )
    return result.returncode, payload, elapsed


def test_concurrent_preflight_reaches_every_real_check(live_project):
    """All 5 checks must independently reach their real provider and PASS.

    Requires actual HF/NGC/Token Factory credentials, not just presence: a
    WARN (missing token) would otherwise silently pass this "every real
    provider" proof without ever making the network call it claims to
    verify. Skip explicitly rather than downgrade the assertion when those
    optional prerequisites are absent; ``test_s3_and_nebius_checks_pass_...``
    below stays required regardless, since the dedicated project always
    provisions S3 and a Nebius CLI profile.
    """

    from npa.clients.credentials import load_credentials

    creds = load_credentials()
    if not (creds.hf_token and creds.ngc_api_key and creds.token_factory_api_key):
        pytest.skip("HF/NGC/Token Factory credentials not configured for this project")

    returncode, payload, elapsed = _run_preflight(live_project, "all")

    checks = {check["name"]: check["status"] for check in payload["checks"]}
    assert checks == {
        "hf": "PASS",
        "ngc": "PASS",
        "s3": "PASS",
        "token_factory": "PASS",
        "nebius": "PASS",
    }
    assert returncode == 0
    print(
        json.dumps(
            {
                "scenario": "credential_preflight_live_all",
                "elapsed_seconds": elapsed,
                "checks": checks,
            }
        )
    )


def test_s3_and_nebius_checks_pass_against_the_dedicated_project(live_project):
    """The two checks the dedicated project specifically provisions must pass."""

    returncode, payload, elapsed = _run_preflight(live_project, "s3,nebius")

    checks = {check["name"]: check["status"] for check in payload["checks"]}
    assert checks == {"s3": "PASS", "nebius": "PASS"}
    assert returncode == 0
    print(
        json.dumps(
            {
                "scenario": "credential_preflight_live_s3_nebius",
                "elapsed_seconds": elapsed,
            }
        )
    )
