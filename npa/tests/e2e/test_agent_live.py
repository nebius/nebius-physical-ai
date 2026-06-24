from __future__ import annotations

import os

import httpx
import pytest

from npa.cli.agent import DEFAULT_AGENT_NAME, DEFAULT_PROJECT_ALIAS, _agent_record, _load_auth_secret

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.environ.get("NPA_AGENT_LIVE") != "1" or os.environ.get("NPA_INTEGRATION_E2E") != "1",
        reason="Set NPA_AGENT_LIVE=1 and NPA_INTEGRATION_E2E=1 for live agent checks.",
    ),
]


def test_agent_live_endpoints_and_catalog() -> None:
    project = os.environ.get("NPA_AGENT_PROJECT", DEFAULT_PROJECT_ALIAS)
    name = os.environ.get("NPA_AGENT_NAME", DEFAULT_AGENT_NAME)
    record = _agent_record(project, name)
    assert record, f"missing agent config for {project}/{name}"

    auth_user, auth_password = _load_auth_secret(str(record.get("auth_secret_path", "")))
    ui_url = str(record.get("agent_url", ""))
    rerun_url = str(record.get("rerun_url", ""))
    tools_url = f"{ui_url.rstrip('/')}/api/tools"

    ui = httpx.get(ui_url, auth=(auth_user, auth_password), timeout=10.0)
    assert ui.status_code == 200

    rerun = httpx.get(rerun_url, auth=(auth_user, auth_password), timeout=10.0)
    assert rerun.status_code == 200

    tools = httpx.get(tools_url, auth=(auth_user, auth_password), timeout=10.0)
    tools.raise_for_status()
    payload = tools.json()
    refs = payload.get("tool_refs", [])
    assert isinstance(refs, list)
    assert len(refs) >= 19
