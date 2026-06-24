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
    sim_viz_url = str(record.get("sim_viz_url", rerun_url))
    sim_assets_url = str(record.get("sim_assets_url", ui_url))
    tools_url = f"{ui_url.rstrip('/')}/api/tools"

    ui = httpx.get(ui_url, auth=(auth_user, auth_password), timeout=10.0)
    assert ui.status_code == 200

    rerun = httpx.get(sim_viz_url, auth=(auth_user, auth_password), timeout=10.0)
    assert rerun.status_code == 200

    sim_assets = httpx.get(
        f"{sim_assets_url.rstrip('/')}/api/sim-assets",
        auth=(auth_user, auth_password),
        timeout=10.0,
    )
    sim_assets.raise_for_status()
    sim_assets_payload = sim_assets.json()
    assert "scene_spec" in sim_assets_payload
    assert "robot_spec" in sim_assets_payload

    cameras = httpx.get(
        f"{sim_assets_url.rstrip('/')}/api/sim-assets/cameras",
        auth=(auth_user, auth_password),
        timeout=10.0,
    )
    cameras.raise_for_status()
    cameras_payload = cameras.json()
    assert isinstance(cameras_payload.get("cameras"), list)
    assert len(cameras_payload["cameras"]) >= 1

    selection_payload = {
        "scene_spec_uri": "stock://scene/default",
        "robot_spec_uri": "stock://robot/franka",
        "cameras_uri": "stock://cameras/default",
        "robot_preset": "franka",
        "sim_backend": "isaac",
        "props": ["cube"],
    }
    selection_set = httpx.post(
        f"{sim_assets_url.rstrip('/')}/api/sim-assets/selection",
        auth=(auth_user, auth_password),
        json=selection_payload,
        timeout=10.0,
    )
    selection_set.raise_for_status()
    selection_get = httpx.get(
        f"{sim_assets_url.rstrip('/')}/api/sim-assets/selection",
        auth=(auth_user, auth_password),
        timeout=10.0,
    )
    selection_get.raise_for_status()
    selection = selection_get.json()
    assert selection.get("scene_spec_uri") == selection_payload["scene_spec_uri"]

    submit = httpx.post(
        f"{ui_url.rstrip('/')}/api/workflows/sim2real/submit",
        auth=(auth_user, auth_password),
        json={},
        timeout=10.0,
    )
    submit.raise_for_status()
    submit_payload = submit.json()
    assert submit_payload.get("run_id")

    tools = httpx.get(tools_url, auth=(auth_user, auth_password), timeout=10.0)
    tools.raise_for_status()
    payload = tools.json()
    refs = payload.get("tool_refs", [])
    assert isinstance(refs, list)
    assert len(refs) >= 19
    resolve = httpx.get(
        f"{ui_url.rstrip('/')}/api/tools/{refs[0]}",
        auth=(auth_user, auth_password),
        timeout=10.0,
    )
    resolve.raise_for_status()
    resolved = resolve.json()
    assert resolved.get("ok") is True
    assert isinstance(resolved.get("argv_template"), list)
