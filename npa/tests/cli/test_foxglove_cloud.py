from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
import pytest

from npa.agent_backend.foxglove_cloud import (
    FOXGLOVE_LAYOUT_NAME,
    FoxgloveCloudClient,
    FoxgloveCloudError,
    data_aware_layout_data,
)


def _rich_provenance() -> dict:
    return {
        "schemas": {
            "/camera/overview": "foxglove.CompressedImage",
            "/camera/workspace": "foxglove.CompressedImage",
            "/trajectory": "foxglove.PointCloud",
            "/tf": "foxglove.FrameTransform",
            "/metrics/execution": "npa.metrics.execution",
            "/log": "foxglove.Log",
        },
        "numeric_paths": {"/metrics/execution": ["reward", "progress", "state_norm"]},
    }


def _panel_nodes(node: dict) -> list[dict]:
    if node.get("type") == "panel":
        return [node]
    return [
        panel
        for item in node.get("items", [])
        for panel in _panel_nodes(item.get("content", {}))
    ]


def test_data_aware_layout_binds_only_real_rich_topics() -> None:
    layout = data_aware_layout_data(_rich_provenance())
    panels = _panel_nodes(layout["content"])

    assert [panel["panelType"] for panel in panels] == [
        "Image",
        "Image",
        "ThreeDee",
        "Plot",
        "Log",
    ]
    assert [
        panel["config"]["imageMode"]["imageTopic"]
        for panel in panels
        if panel["panelType"] == "Image"
    ] == ["/camera/overview", "/camera/workspace"]
    plot = next(panel for panel in panels if panel["panelType"] == "Plot")
    assert [path["value"] for path in plot["config"]["paths"]] == [
        "/metrics/execution.reward",
        "/metrics/execution.progress",
        "/metrics/execution.state_norm",
    ]


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


def test_cloud_layout_is_created_then_reused_without_quota_churn() -> None:
    layouts: list[dict] = []
    writes: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/layouts" and request.method == "GET":
            return httpx.Response(200, json=layouts)
        if request.url.path == "/v1/layouts" and request.method == "POST":
            body = __import__("json").loads(request.content)
            writes.append(request.method)
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

    cloud = FoxgloveCloudClient(
        "token",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda _: None,
    )

    result = cloud.ensure_recording(recording, run_id="run")

    assert result.recording_id == "rec_pending"
    assert result.reused is True


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
