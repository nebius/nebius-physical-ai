from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
import pytest

from npa.agent_backend.foxglove_cloud import (
    DEFAULT_FOXGLOVE_CLOUD_IMPORT_TIMEOUT_SECONDS,
    FOXGLOVE_CLOUD_IMPORT_TIMEOUT_ENV,
    FOXGLOVE_LAYOUT_ID,
    FOXGLOVE_LAYOUT_NAME,
    FoxgloveCloudClient,
    FoxgloveCloudError,
    FoxgloveCloudTimeoutError,
    data_aware_layout_data,
    resolve_cloud_import_timeout_seconds,
)


def _rich_provenance() -> dict:
    return {
        "visualization_contract": "npa.foxglove.robot-motion.v3",
        "schemas": {
            "/camera": "foxglove.CompressedImage",
            "/camera/side": "foxglove.CompressedImage",
            "/camera/workspace": "foxglove.CompressedImage",
            "/robot/diagnostic_scene": "foxglove.SceneUpdate",
            "/robot/diagnostic_pose": "foxglove.PoseInFrame",
            "/robot/diagnostic_trajectory": "foxglove.PosesInFrame",
            "/robot/diagnostic_joint_states": "foxglove.JointStates",
            "/actuators/commands": "npa.ActuatorCommands",
            "/run/state": "npa.RunState",
            "/tf": "foxglove.FrameTransform",
            "/metrics/execution": "npa.RunMetrics.execution",
            "/log": "foxglove.Log",
        },
        "numeric_paths": {
            "/metrics/execution": [
                "object_goal_distance_m",
                "object_lift_m",
                "reward",
            ],
            "/run/state": ["progress", "sim_step", "step"],
        },
        "visualization_fixed_frame": "npa_action_space",
    }


def _panel_nodes(node: dict) -> list[dict]:
    if node.get("type") == "panel":
        return [node]
    split_panels = [
        panel
        for item in node.get("items", [])
        for panel in _panel_nodes(item.get("content", {}))
    ]
    tab_panels = [
        panel
        for item in node.get("tabs", [])
        for panel in _panel_nodes(item.get("content", {}))
    ]
    return split_panels + tab_panels


def _tab_nodes(node: dict) -> list[dict]:
    found = [node] if node.get("type") == "tabs" else []
    for item in node.get("items", []):
        found.extend(_tab_nodes(item.get("content", {})))
    for item in node.get("tabs", []):
        found.extend(_tab_nodes(item.get("content", {})))
    return found


def test_data_aware_layout_binds_only_real_rich_topics() -> None:
    layout = data_aware_layout_data(_rich_provenance())
    panels = _panel_nodes(layout["content"])

    assert [panel["panelType"] for panel in panels] == [
        "ThreeDee",
        "Image",
        "Image",
        "Image",
        "Plot",
        "StateTransitions",
        "Log",
    ]
    assert [
        panel["config"]["imageMode"]["imageTopic"]
        for panel in panels
        if panel["panelType"] == "Image"
    ] == ["/camera", "/camera/side", "/camera/workspace"]
    camera_tabs = _tab_nodes(layout["content"])[0]
    assert camera_tabs["selectedTabIndex"] == 0
    assert [tab["title"] for tab in camera_tabs["tabs"]] == [
        "Primary (/camera)",
        "Side (/camera/side)",
        "Workspace (/camera/workspace)",
    ]
    assert [
        tab["content"]["config"]["npaCamera"]["sourceFidelity"]
        for tab in camera_tabs["tabs"]
    ] == ["source-rgb-only"] * 3
    three_dee = panels[0]
    assert three_dee["config"]["fixedFrame"] == "npa_action_space"
    assert set(three_dee["config"]["topics"]) == {
        "/robot/diagnostic_scene",
        "/robot/diagnostic_pose",
        "/robot/diagnostic_trajectory",
    }
    plot = next(panel for panel in panels if panel["panelType"] == "Plot")
    assert [path["value"] for path in plot["config"]["paths"]] == [
        "/metrics/execution.reward",
        "/metrics/execution.object_lift_m",
        "/metrics/execution.object_goal_distance_m",
        "/run/state.progress",
    ]
    transitions = next(
        panel for panel in panels if panel["panelType"] == "StateTransitions"
    )
    assert transitions["config"]["paths"][0]["value"] == "/run/state.phase"
    assert layout["content"]["direction"] == "column"
    assert [item["proportion"] for item in layout["content"]["items"]] == [0.72, 0.28]
    assert "Settings" not in {panel["panelType"] for panel in panels}
    assert "UserScript" not in {panel["panelType"] for panel in panels}


def test_data_aware_layout_omits_unsupported_empty_3d_panel() -> None:
    layout = data_aware_layout_data(
        {
            "schemas": {
                "/camera": "foxglove.CompressedImage",
                "/tf": "foxglove.FrameTransform",
            }
        }
    )
    assert [panel["panelType"] for panel in _panel_nodes(layout["content"])] == [
        "Image"
    ]
    assert _tab_nodes(layout["content"])[0]["tabs"][0]["title"] == ("Primary (/camera)")


def test_cloud_layout_is_created_then_reused_without_quota_churn() -> None:
    layouts: list[dict] = []
    writes: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/layouts" and request.method == "GET":
            return httpx.Response(200, json=layouts)
        if request.url.path == "/v1/layouts" and request.method == "POST":
            body = __import__("json").loads(request.content)
            writes.append(request.method)
            assert body["id"] == FOXGLOVE_LAYOUT_ID
            assert len(body["id"]) == 20
            assert body["id"].startswith("lay_")
            assert body["name"] == FOXGLOVE_LAYOUT_NAME
            assert body["permission"] == "ORG_WRITE"
            layouts.append(
                {"id": "lay_rich_v1", "name": body["name"], "data": body["data"]}
            )
            return httpx.Response(201, json=layouts[-1])
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    cloud = FoxgloveCloudClient(
        "secret-token",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    first = cloud.ensure_layout(_rich_provenance())
    second = cloud.ensure_layout(_rich_provenance())

    assert first.layout_id == second.layout_id == "lay_rich_v1"
    assert first.created is True
    assert second.reused is True
    assert writes == ["POST"]


def test_cloud_layout_preserves_user_modified_versioned_layout() -> None:
    layouts = [
        {
            "id": "lay_user_arranged",
            "name": FOXGLOVE_LAYOUT_NAME,
            "data": {"version": 1, "content": {"type": "panel", "panelType": "Image"}},
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/v1/layouts"
        return httpx.Response(200, json=layouts)

    cloud = FoxgloveCloudClient(
        "secret-token",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = cloud.ensure_layout(_rich_provenance())

    assert result.layout_id == "lay_user_arranged"
    assert result.reused is True
    assert result.updated is False


def test_cloud_layout_reuses_concurrent_deterministic_create() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            assert request.method == "GET"
            return httpx.Response(200, json=[])
        if calls == 2:
            assert request.method == "POST"
            return httpx.Response(409, json={"error": "already exists"})
        assert request.method == "GET"
        return httpx.Response(
            200,
            json=[{"id": "layout-winner", "name": FOXGLOVE_LAYOUT_NAME}],
        )

    cloud = FoxgloveCloudClient(
        "secret-token",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = cloud.ensure_layout(_rich_provenance())

    assert result.layout_id == "layout-winner"
    assert result.reused is True
    assert result.created is False
    assert calls == 3


def test_cloud_layout_refuses_non_rich_recording() -> None:
    cloud = FoxgloveCloudClient(
        "secret-token",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: (_ for _ in ()).throw(
                    AssertionError("non-rich recording must not call the layout API")
                )
            )
        ),
    )

    result = cloud.ensure_layout({"schemas": {"/camera": "foxglove.CompressedImage"}})

    assert result.available is False
    assert result.layout_id == ""
    assert "robot-motion v3" in result.reason


def test_cloud_layout_plan_failure_has_explicit_token_safe_fallback() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=[])
        return httpx.Response(403, json={"error": "secret-token"})

    cloud = FoxgloveCloudClient(
        "secret-token",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    layout = cloud.ensure_layout(_rich_provenance())

    assert layout.available is False
    assert layout.layout_id == ""
    assert "secret-token" not in layout.reason


def test_cloud_layout_list_denial_does_not_block_indexed_recording_fallback() -> None:
    cloud = FoxgloveCloudClient(
        "secret-token",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(403, json={"error": "secret-token"})
            )
        ),
    )

    layout = cloud.ensure_layout(_rich_provenance())

    assert layout.available is False
    assert layout.layout_id == ""
    assert "project access" in layout.reason
    assert "secret-token" not in layout.reason


def test_cloud_upload_is_content_idempotent_and_token_safe(tmp_path: Path) -> None:
    recording = tmp_path / "run.mcap"
    recording.write_bytes(b"\x89MCAP0\r\ncloud-test")
    digest = hashlib.sha256(recording.read_bytes()).hexdigest()
    key = f"npa-{digest}"
    uploaded = False
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal uploaded
        requests.append(request)
        if request.url.host == "signed-upload.example":
            assert "authorization" not in request.headers
            assert request.headers["content-type"] == "application/octet-stream"
            uploaded = True
            return httpx.Response(200)
        assert request.headers["authorization"] == "Bearer secret-token"
        if request.url.path == "/v1/projects":
            return httpx.Response(200, json=[{"id": "prj_one"}])
        if request.url.path == f"/v1/recordings/{key}":
            if not uploaded:
                return httpx.Response(404, json={"error": "missing"})
            return httpx.Response(
                200,
                json={
                    "id": "rec_one",
                    "key": key,
                    "size": recording.stat().st_size,
                    "importStatus": "complete",
                },
            )
        if request.url.path == "/v1/data/pending-imports":
            return httpx.Response(200, json=[])
        if request.url.path == "/v1/data/upload":
            body = __import__("json").loads(request.content)
            assert body["key"] == key
            assert body["projectId"] == "prj_one"
            return httpx.Response(
                200,
                json={
                    "link": "https://signed-upload.example/object",
                    "requestId": "req_one",
                },
            )
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    cloud = FoxgloveCloudClient("secret-token", client=client, sleep=lambda _: None)

    first = cloud.ensure_recording(recording, run_id="run-one")
    second = cloud.ensure_recording(recording, run_id="run-one")

    assert first.uploaded is True and first.reused is False
    assert second.uploaded is False and second.reused is True
    assert first.recording_id == second.recording_id == "rec_one"
    assert sum(request.url.path == "/v1/data/upload" for request in requests) == 1
    assert all("secret-token" not in str(request.url) for request in requests)
    assert "secret-token" not in repr(first.to_dict())


def test_cloud_reuses_in_progress_import_without_upload(tmp_path: Path) -> None:
    recording = tmp_path / "run.mcap"
    recording.write_bytes(b"\x89MCAP0\r\npending")
    digest = hashlib.sha256(recording.read_bytes()).hexdigest()
    key = f"npa-{digest}"
    polls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal polls
        if request.url.path == f"/v1/recordings/{key}":
            polls += 1
            if polls < 3:
                return httpx.Response(404, json={})
            return httpx.Response(
                200,
                json={
                    "id": "rec_pending",
                    "key": key,
                    "size": recording.stat().st_size,
                    "importStatus": "complete",
                },
            )
        if request.url.path == "/v1/data/pending-imports":
            return httpx.Response(
                200,
                json=[
                    {
                        "requestId": "req_pending",
                        "status": "processing",
                        "updatedAt": "now",
                    }
                ],
            )
        if request.url.path == "/v1/data/upload":
            raise AssertionError("unchanged pending recording must not upload again")
        raise AssertionError(str(request.url))

    now = [10.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    cloud = FoxgloveCloudClient(
        "token",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=sleep,
        clock=lambda: now[0],
        wait_timeout_seconds=10,
    )

    result = cloud.ensure_recording(recording, run_id="run")

    assert result.recording_id == "rec_pending"
    assert result.reused is True
    assert sleeps == [2.0]


def test_cloud_wait_surfaces_terminal_import_failure(tmp_path: Path) -> None:
    recording = tmp_path / "run.mcap"
    recording.write_bytes(b"\x89MCAP0\r\nfailed")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/v1/recordings/"):
            return httpx.Response(404, json={})
        if request.url.path == "/v1/data/pending-imports":
            return httpx.Response(
                200,
                json=[{"requestId": "req", "status": "failed", "updatedAt": "now"}],
            )
        raise AssertionError(str(request.url))

    cloud = FoxgloveCloudClient(
        "token",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda _seconds: None,
        wait_timeout_seconds=10,
    )
    with pytest.raises(FoxgloveCloudError, match="could not import") as exc_info:
        cloud.ensure_recording(recording, run_id="run")
    assert exc_info.value.status_code == 422


def test_cloud_wait_has_monotonic_deadline_and_safe_context(tmp_path: Path) -> None:
    recording = tmp_path / "run.mcap"
    recording.write_bytes(b"\x89MCAP0\r\nforever-pending")
    digest = hashlib.sha256(recording.read_bytes()).hexdigest()
    key = f"npa-{digest}"
    now = [100.0]
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == f"/v1/recordings/{key}":
            return httpx.Response(404, json={})
        if request.url.path == "/v1/data/pending-imports":
            return httpx.Response(
                200,
                json=[{"requestId": "req", "status": "processing", "updatedAt": "now"}],
            )
        raise AssertionError(str(request.url))

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    cloud = FoxgloveCloudClient(
        "token-not-in-error",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=sleep,
        clock=lambda: now[0],
        wait_timeout_seconds=5,
    )

    with pytest.raises(FoxgloveCloudTimeoutError) as exc_info:
        cloud.ensure_recording(recording, run_id="run")

    error = exc_info.value
    assert error.status_code == 504
    assert error.context == {
        "recording_key": key,
        "recording_status": "missing",
        "import_status": "processing",
        "elapsed_seconds": 5.0,
    }
    assert sleeps == [2.0, 2.0, 1.0]
    assert "token-not-in-error" not in str(error)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "3600.1", "invalid"])
def test_cloud_import_timeout_rejects_non_positive_or_non_finite_values(
    value: str,
) -> None:
    with pytest.raises(ValueError, match=FOXGLOVE_CLOUD_IMPORT_TIMEOUT_ENV):
        resolve_cloud_import_timeout_seconds(value, environ={})
    with pytest.raises(FoxgloveCloudError) as exc_info:
        FoxgloveCloudClient("token", wait_timeout_seconds=value)
    assert exc_info.value.status_code == 503
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert "no greater than 3600" in str(exc_info.value)


def test_cloud_import_timeout_default_and_environment_override() -> None:
    assert (
        resolve_cloud_import_timeout_seconds(environ={})
        == DEFAULT_FOXGLOVE_CLOUD_IMPORT_TIMEOUT_SECONDS
    )
    assert (
        resolve_cloud_import_timeout_seconds(
            environ={FOXGLOVE_CLOUD_IMPORT_TIMEOUT_ENV: "12.5"}
        )
        == 12.5
    )


def test_cloud_errors_are_actionable_and_do_not_echo_token(tmp_path: Path) -> None:
    with pytest.raises(FoxgloveCloudError, match="requires FOXGLOVE_API_TOKEN"):
        FoxgloveCloudClient("")

    recording = tmp_path / "run.mcap"
    recording.write_bytes(b"\x89MCAP0\r\nquota")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/v1/recordings/"):
            return httpx.Response(404, json={})
        if request.url.path == "/v1/data/pending-imports":
            return httpx.Response(200, json=[])
        if request.url.path == "/v1/projects":
            return httpx.Response(200, json=[{"id": "prj_one"}])
        if request.url.path == "/v1/data/upload":
            return httpx.Response(429, json={"error": "token-value-must-not-escape"})
        raise AssertionError(str(request.url))

    cloud = FoxgloveCloudClient(
        "token-value-must-not-escape",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda _: None,
    )
    with pytest.raises(FoxgloveCloudError, match="quota") as exc_info:
        cloud.ensure_recording(recording, run_id="run")
    assert "token-value-must-not-escape" not in str(exc_info.value)
