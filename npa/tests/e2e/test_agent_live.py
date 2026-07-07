from __future__ import annotations

import os

import httpx
import pytest

from .agent_live_helpers import (
    RERUN_STATIC_CANDIDATES,
    STOCK_FRANKA_SELECTION,
    UI_BUTTON_IDS,
    UI_WIRING_MARKERS,
    AgentLiveContext,
    load_agent_live_context,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.agent_live,
    pytest.mark.skipif(
        os.environ.get("NPA_AGENT_LIVE") != "1" or os.environ.get("NPA_INTEGRATION_E2E") != "1",
        reason="Set NPA_AGENT_LIVE=1 and NPA_INTEGRATION_E2E=1 for live agent checks.",
    ),
]


@pytest.fixture(scope="module")
def ctx() -> AgentLiveContext:
    return load_agent_live_context()


def test_agent_ui_html_smoke(ctx: AgentLiveContext) -> None:
    resp = ctx.get(ctx.agent_url)
    assert resp.status_code == 200
    html = resp.text
    for marker in UI_WIRING_MARKERS:
        assert marker in html, f"missing UI marker: {marker}"
    for control_id in UI_BUTTON_IDS:
        assert f'id="{control_id}"' in html
        assert f'bindClick("{control_id}"' in html
    assert 'id="chatSend"' in html
    assert 'id="chatForm"' in html
    assert 'id="chatSessionSelect"' in html
    assert 'chatForm.addEventListener("submit"' in html
    assert "/api/chat/sessions" in html


def test_agent_health_and_session(ctx: AgentLiveContext) -> None:
    health = ctx.get("/api/health")
    health.raise_for_status()
    assert health.json().get("ok") is True

    session = ctx.get("/api/session")
    session.raise_for_status()
    payload = session.json()
    assert isinstance(payload, dict)
    assert isinstance(payload.get("chat_history", []), list)


def test_agent_sim_assets_and_catalog(ctx: AgentLiveContext) -> None:
    assets = ctx.get("/api/sim-assets")
    assets.raise_for_status()
    payload = assets.json()
    assert "scene_spec" in payload
    assert "robot_spec" in payload
    assert "selection" in payload

    catalog = ctx.get("/api/sim-assets/catalog")
    catalog.raise_for_status()
    catalog_payload = catalog.json()
    assert isinstance(catalog_payload, dict)


def test_agent_cameras_and_selection_roundtrip(ctx: AgentLiveContext) -> None:
    cameras = ctx.get("/api/sim-assets/cameras")
    cameras.raise_for_status()
    cameras_payload = cameras.json()
    camera_list = cameras_payload.get("cameras", [])
    assert isinstance(camera_list, list) and camera_list

    selection_set = ctx.post("/api/sim-assets/selection", json=STOCK_FRANKA_SELECTION)
    selection_set.raise_for_status()

    selection_get = ctx.get("/api/sim-assets/selection")
    selection_get.raise_for_status()
    selected = selection_get.json()
    assert selected.get("scene_spec_uri") == STOCK_FRANKA_SELECTION["scene_spec_uri"]

    camera_name = str(camera_list[0]["name"])
    camera_put = ctx.put(
        "/api/sim-assets/cameras/selection",
        json={"selected": [camera_name]},
    )
    camera_put.raise_for_status()


def test_agent_workflow_submit_and_status(ctx: AgentLiveContext) -> None:
    submit = ctx.post("/api/workflows/sim2real/submit", json={})
    submit.raise_for_status()
    submit_payload = submit.json()
    run_id = str(submit_payload.get("run_id") or "").strip()
    assert run_id
    submit_viz = submit_payload.get("sim_viz", {})
    assert isinstance(submit_viz, dict)
    assert submit_viz.get("run_id") == run_id
    assert submit_viz.get("rrd_uri"), "submitted Sim2Real run did not get a visualization .rrd"

    status = ctx.get("/api/workflows/sim2real/status")
    status.raise_for_status()
    status_payload = status.json()
    assert isinstance(status_payload, dict)
    latest_submit = status_payload.get("latest_submit", {})
    sim_viz = status_payload.get("sim_viz", {})
    assert isinstance(latest_submit, dict)
    assert latest_submit.get("run_id") == run_id
    assert isinstance(sim_viz, dict)
    assert sim_viz.get("run_id") == run_id
    assert sim_viz.get("rrd_uri")

    run_status = ctx.get(f"/api/sim-viz/status?run_id={run_id}")
    run_status.raise_for_status()
    run_status_payload = run_status.json()
    assert run_status_payload.get("run_id") == run_id
    assert run_status_payload.get("rrd_uri")
    assert run_status_payload.get("rerun_ready") or run_status_payload.get("rrd_uri")

    load_run = ctx.post("/api/sim-viz/load-run", json={"run_id": run_id})
    load_run.raise_for_status()
    load_run_payload = load_run.json()
    loaded_viz = load_run_payload.get("sim_viz", {})
    assert isinstance(loaded_viz, dict)
    assert loaded_viz.get("run_id") == run_id
    assert loaded_viz.get("rrd_uri")

    rrd_blob = ctx.get(f"/api/sim-viz/rrd-blob?run_id={run_id}")
    rrd_blob.raise_for_status()
    assert len(rrd_blob.content) > 64, "submitted Sim2Real run .rrd payload was empty"


def test_agent_tools_catalog(ctx: AgentLiveContext) -> None:
    tools = ctx.get("/api/tools")
    tools.raise_for_status()
    refs = tools.json().get("tool_refs", [])
    assert isinstance(refs, list)
    assert len(refs) >= 19

    resolve = ctx.get(f"/api/tools/{refs[0]}")
    resolve.raise_for_status()
    resolved = resolve.json()
    assert resolved.get("ok") is True
    assert isinstance(resolved.get("argv_template"), list)


def test_agent_workbench_actions(ctx: AgentLiveContext) -> None:
    actions = ctx.get("/api/workbench/actions")
    actions.raise_for_status()
    payload = actions.json()
    assert isinstance(payload, dict)


def test_agent_rerun_iframe_reachable(ctx: AgentLiveContext) -> None:
    base = ctx.agent_url.rstrip("/")
    rerun = httpx.get(
        f"{base}/rerun/",
        auth=ctx.auth(),
        timeout=15.0,
        verify=ctx.tls_verify,
    )
    assert rerun.status_code == 200
    assert rerun.text.strip()

    legacy = httpx.get(
        ctx.sim_viz_url,
        auth=ctx.auth(),
        timeout=15.0,
        verify=ctx.tls_verify,
    )
    assert legacy.status_code == 200
    assert legacy.text.strip()


def test_agent_rerun_static_assets(ctx: AgentLiveContext) -> None:
    base = ctx.agent_url.rstrip("/")
    ok_paths: list[str] = []
    for path in RERUN_STATIC_CANDIDATES:
        resp = httpx.get(
            f"{base}{path}",
            auth=ctx.auth(),
            timeout=15.0,
            verify=ctx.tls_verify,
        )
        if resp.status_code == 200 and resp.content:
            ok_paths.append(path)
    assert ok_paths, f"no rerun static asset responded 200 among {RERUN_STATIC_CANDIDATES}"


def test_agent_load_franka_demo_and_rrd(ctx: AgentLiveContext) -> None:
    load_demo = ctx.post("/api/sim-viz/load-franka-demo", json={"camera": "workspace"})
    load_demo.raise_for_status()
    demo_payload = load_demo.json()
    assert demo_payload.get("ok") is True
    sim_viz = demo_payload.get("sim_viz", {})
    assert isinstance(sim_viz, dict)
    assert sim_viz.get("rerun_ready") or sim_viz.get("rrd_uri")

    status = ctx.get("/api/sim-viz/status")
    status.raise_for_status()
    status_payload = status.json()
    assert status_payload.get("rerun_ready") or status_payload.get("rrd_uri")

    rrd = ctx.get("/api/sim-viz/rrd")
    rrd.raise_for_status()
    content_type = rrd.headers.get("content-type", "")
    if "application/json" in content_type:
        assert rrd.json().get("ok") is True
    else:
        assert len(rrd.content) > 64, "expected non-trivial .rrd payload"


def test_agent_camera_preview(ctx: AgentLiveContext) -> None:
    preview = ctx.post("/api/sim-viz/camera-preview", json={"camera": "workspace"})
    preview.raise_for_status()
    payload = preview.json()
    assert payload.get("ok") is True
    assert payload.get("entity_path")


def _assert_grounded_status_reply(payload: dict[str, object]) -> str:
    assert payload.get("ok") is True
    reply = str(payload.get("reply") or "")
    assert reply
    assert "run_id" in reply or "stage" in reply, "reply missing run_id/stage fields"
    assert not reply.strip().startswith("GET /api"), "raw GET path instead of unpacked status"
    assert reply.strip() != "GET /api/sim-viz/status"
    return reply


def test_agent_chat_grounded_sim2real_status(ctx: AgentLiveContext) -> None:
    chat = ctx.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "what is the current sim2real status"}]},
        timeout=30.0,
    )
    chat.raise_for_status()
    _assert_grounded_status_reply(chat.json())


def test_agent_chat_grounded_field(ctx: AgentLiveContext) -> None:
    chat = ctx.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "what is the current sim2real status"}]},
        timeout=30.0,
    )
    chat.raise_for_status()
    payload = chat.json()
    _assert_grounded_status_reply(payload)
    assert payload.get("grounded") is True
    apis_used = payload.get("apis_used")
    assert isinstance(apis_used, list) and apis_used


def test_agent_chat_sim_assets_intent(ctx: AgentLiveContext) -> None:
    chat = ctx.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "what sim assets are selected"}]},
        timeout=30.0,
    )
    chat.raise_for_status()
    payload = chat.json()
    assert payload.get("ok") is True
    assert payload.get("grounded") is True
    reply = str(payload.get("reply") or "").lower()
    assert any(token in reply for token in ("franka", "isaac", "selection", "robot_preset"))


def test_agent_models_endpoint(ctx: AgentLiveContext) -> None:
    models = ctx.get("/api/models")
    models.raise_for_status()
    payload = models.json()
    assert payload.get("ok") is True
    model_list = payload.get("models")
    assert isinstance(model_list, list) and model_list
    default_model = str(payload.get("default_model") or payload.get("default") or "").strip()
    assert default_model
    assert default_model in [str(item) for item in model_list]


def test_agent_chat_onboard_solution_intent(ctx: AgentLiveContext) -> None:
    from .agent_live_helpers import ONBOARD_SOLUTION_PROMPT, assert_grounded_onboard_solution_reply

    chat = ctx.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": ONBOARD_SOLUTION_PROMPT}]},
        timeout=30.0,
    )
    chat.raise_for_status()
    assert_grounded_onboard_solution_reply(chat.json())


@pytest.mark.skipif(
    os.environ.get("NPA_AGENT_CHAT_LIVE") != "1",
    reason="Set NPA_AGENT_CHAT_LIVE=1 to smoke-test Token Factory chat on the live agent.",
)
def test_agent_chat_live(ctx: AgentLiveContext) -> None:
    chat = ctx.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "Reply with the word ok."}]},
        timeout=60.0,
    )
    chat.raise_for_status()
    payload = chat.json()
    assert payload.get("ok") is True
    assert payload.get("reply")


@pytest.mark.skipif(
    os.environ.get("NPA_AGENT_CHAT_LIVE") != "1",
    reason="Set NPA_AGENT_CHAT_LIVE=1 to live-test explicit model switching.",
)
def test_agent_chat_live_model_switch(ctx: AgentLiveContext) -> None:
    model_resp = ctx.get("/api/models")
    model_resp.raise_for_status()
    model_payload = model_resp.json()
    default_model = str(model_payload.get("default_model") or model_payload.get("default") or "").strip()
    models = [str(item) for item in (model_payload.get("models") or []) if str(item).strip()]
    assert models
    alternate = next((item for item in models if item != default_model), models[0])
    chat = ctx.post(
        "/api/chat",
        json={
            "messages": [{"role": "user", "content": "Reply with the word ok."}],
            "model": alternate,
        },
        timeout=60.0,
    )
    chat.raise_for_status()
    payload = chat.json()
    assert payload.get("ok") is True
    assert payload.get("reply")
    assert str(payload.get("model") or "") == alternate
