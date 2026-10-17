"""Preserve fail-closed endpoint, UI, and test gates after CLI extraction."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import NoReturn

import httpx
import pytest

from npa.cli import agent_verification as verification
from npa.cli.agent import AGENT_UI_VERSION
from npa.cli.agent_contracts import rendered_agent_ui_html
from npa.cli.agent_verification_ui import verify_agent_ui


class _VerificationFailed(Exception):
    pass


def _fail(message: str) -> NoReturn:
    raise _VerificationFailed(message)


@pytest.fixture
def context() -> verification._VerificationContext:
    return verification._VerificationContext(
        record={"agent_url": "https://agent.invalid/", "public_https": False},
        auth=("test-user", "test-password"),
        ui_version="test-version",
        health_check=lambda *_args, **_kwargs: (True, 200),
        fail=_fail,
    )


@pytest.mark.parametrize("status", [200, 302, 403, 500])
def test_public_access_requires_unauthenticated_401(monkeypatch, context, status):
    calls = []

    def respond(url, **kwargs):
        calls.append((url, kwargs))
        code = status if url == context.agent_url else 200
        return httpx.Response(code, request=httpx.Request("GET", url))

    monkeypatch.setattr(verification.httpx, "get", respond)
    with pytest.raises(
        _VerificationFailed, match="did not enforce basic authentication"
    ):
        verification._verify_public_access(context)
    assert [url for url, _ in calls] == [
        "https://agent.invalid/welcome",
        "https://agent.invalid/healthz",
        "https://agent.invalid/",
    ]
    assert "auth" not in calls[-1][1]
    assert calls[-1][1]["follow_redirects"] is False


def test_public_access_probes_authenticated_api_after_401(monkeypatch, context):
    calls = []

    def respond(url, **kwargs):
        calls.append((url, kwargs))
        code = 401 if url == context.agent_url else 200
        return httpx.Response(code, request=httpx.Request("GET", url))

    monkeypatch.setattr(verification.httpx, "get", respond)
    verification._verify_public_access(context)
    assert len(calls) == 4
    assert calls[-1] == (
        "https://agent.invalid/api/health",
        {"auth": context.auth, "timeout": 5.0, "verify": True},
    )


@pytest.mark.parametrize(
    ("check", "payload", "message"),
    [
        (verification._verify_sim_viz_status, [], "did not return JSON object"),
        (verification._verify_sim_assets, {}, "missing scene_spec/robot_spec"),
        (verification._verify_cameras, {"cameras": []}, "returned no cameras"),
        (verification._verify_backend_health, {"ok": False}, "did not return ok=true"),
        (verification._verify_leisaac, {"available": "yes"}, "availability object"),
        (verification._verify_workflow_status, [], "did not return JSON object"),
        (verification._verify_session, [], "did not return JSON object"),
        (verification._verify_infra, {"ok": True}, "NPA runtime is not ready"),
    ],
)
def test_incomplete_endpoint_payloads_fail(
    monkeypatch, context, check, payload, message
):
    def respond(url, **_kwargs):
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

    monkeypatch.setattr(verification.httpx, "get", respond)
    with pytest.raises(_VerificationFailed, match=message):
        check(context)


def test_unreachable_endpoint_retains_diagnostic(monkeypatch, context):
    def unreachable(*_args, **_kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(verification.httpx, "get", unreachable)
    with pytest.raises(
        _VerificationFailed,
        match="sim viz status endpoint unhealthy: connection refused",
    ):
        verification._verify_sim_viz_status(context)


def test_run_scoped_recording_retains_identity_and_timeout(monkeypatch, context):
    calls = []

    def respond(url, **kwargs):
        calls.append((url, kwargs))
        return httpx.Response(200, content=b"short", request=httpx.Request("GET", url))

    monkeypatch.setattr(verification.httpx, "get", respond)
    with pytest.raises(_VerificationFailed, match="unexpectedly small payload"):
        verification._verify_submitted_recording(context, "selected-run")
    assert calls == [
        (
            "https://agent.invalid/api/sim-viz/rrd-blob",
            {
                "auth": context.auth,
                "params": {"run_id": "selected-run"},
                "timeout": 15.0,
                "verify": True,
            },
        )
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {"ok": False},
        {"ok": True, "submit_mode": "agent-live-infra-dry-run"},
        {"ok": True, "scheduler_plan": {}, "submit_mode": "live"},
    ],
)
def test_workflow_submission_requires_plan_and_dry_run_mode(
    monkeypatch, context, payload
):
    def respond(url, **kwargs):
        assert kwargs["json"] == {
            "yaml": "workflow-yaml",
            "run_id": "verify-live-agent-infra",
            "dry_run": True,
            "allow_provision": True,
            "validate_infra": False,
        }
        assert kwargs["timeout"] == 120.0
        return httpx.Response(200, json=payload, request=httpx.Request("POST", url))

    monkeypatch.setattr(verification.httpx, "post", respond)
    with pytest.raises(_VerificationFailed, match="workflow submit dry-run"):
        verification._verify_workflow_submission(context, "workflow-yaml")


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('id="chatForm"', "", "missing wiring marker"),
        ('name="npa-ui-version"', 'name="old-version"', "missing wiring marker"),
        ("</body>", '<video controls src="${previewUrl}"></body>', "<video src>"),
        (
            "</body>",
            '`<img alt="artifact image" src="${previewUrl}"</body>',
            "<img src>",
        ),
        ("</body>", 'loading="lazy"</body>', "must not use lazy-loading"),
        (
            "</body>",
            "await waitUntilRerunPastBundleSplash(iframe, 45000)</body>",
            "must not block mount",
        ),
    ],
)
def test_ui_gate_rejects_broken_wiring(old, new, message):
    html = rendered_agent_ui_html()
    verify_agent_ui(html, ui_version=AGENT_UI_VERSION, fail=_fail)
    with pytest.raises(_VerificationFailed, match=message):
        verify_agent_ui(html.replace(old, new), ui_version=AGENT_UI_VERSION, fail=_fail)


@pytest.mark.parametrize("failure_index", [0, 1, 2])
def test_pytest_gate_stops_at_first_failure(monkeypatch, tmp_path, failure_index):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, int(len(calls) == failure_index + 1))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NPA_AGENT_CHAT_LIVE", "1")
    monkeypatch.setattr(verification.subprocess, "run", run)
    with pytest.raises(_VerificationFailed, match="pytest npa/tests/"):
        verification.run_agent_verification_tests(
            "test-project", "test-agent", fail=_fail
        )
    assert len(calls) == failure_index + 1
    for argv, kwargs in calls:
        assert argv[:3] == [sys.executable, "-m", "pytest"]
        assert kwargs["cwd"] == str(Path(__file__).resolve().parents[3])
        assert kwargs["env"]["NPA_AGENT_PROJECT"] == "test-project"
        assert kwargs["env"]["NPA_AGENT_NAME"] == "test-agent"
        assert kwargs["env"]["NPA_AGENT_CHAT_LIVE"] == "1"
        assert kwargs["env"]["NPA_INTEGRATION_E2E"] == "1"
        assert kwargs["env"]["NPA_AGENT_LIVE"] == "1"
