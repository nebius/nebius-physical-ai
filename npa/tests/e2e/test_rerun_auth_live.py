"""Verify the deployed viewer's real nginx authentication boundary."""

import hashlib
import os

import httpx
import pytest


pytestmark = pytest.mark.e2e


def test_deployed_viewer_authentication_and_artifact_bytes():
    names = ("NPA_E2E_RERUN_ARTIFACT_URL", "NPA_E2E_RERUN_AUTH_USER",
             "NPA_E2E_RERUN_AUTH_PASSWORD", "NPA_E2E_RERUN_ARTIFACT_SHA256")
    if os.environ.get("NPA_INTEGRATION_E2E") != "1" or not all(
        os.environ.get(name) for name in names
    ):
        pytest.skip("requires an explicitly configured live viewer and artifact")
    url, user, password, expected_sha256 = (os.environ[name] for name in names)
    # Never follow redirects with credentials; the configured URL must name the
    # actual deployment. Logs and assertions contain no credentials or URLs.
    with httpx.Client(follow_redirects=False) as client:
        anonymous = client.get(url)
        assert anonymous.status_code == 401
        wrong = client.get(url, auth=(user, "invalid-" + password))
        assert wrong.status_code == 401
        correct = client.get(url, auth=(user, password))
        assert correct.status_code == 200
        assert correct.content
        assert hashlib.sha256(correct.content).hexdigest() == expected_sha256
