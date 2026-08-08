"""Capability, attestation, and security tests for the agent LeIsaac tab."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import get_type_hints

import pytest
import httpx
from fastapi import FastAPI, Response
from fastapi.testclient import TestClient
from starlette.websockets import WebSocket, WebSocketDisconnect

from npa.agent_backend.leisaac import (
    LEISAAC_MEDIA_PORT,
    LEISAAC_SIGNAL_PORT,
    LEISAAC_TASK,
    LEISAAC_TURN_PORT,
    LEISAAC_TURN_RELAY_PORT,
    LEISAAC_TURN_RELAY_MAX_PORT,
    load_manifest_artifact,
    normalize_manifest,
    selected_run_id,
    status_payload,
    validate_health,
)
from npa.agent_backend.leisaac_routes import (
    LeIsaacDeps,
    _health,
    _resolve,
    _same_https_origin,
    register_leisaac_routes,
)
from npa.agent_backend.leisaac_registry import REGISTRY_FINGERPRINT
from npa.agent_backend.leisaac_transport import (
    VIDEO_SUBPROTOCOL,
    FrameEnvelope,
    pack_frame,
)


@pytest.fixture
def anyio_backend() -> str:
    """The agent backend is deployed on asyncio/uvicorn."""
    return "asyncio"


def _manifest(**overrides):
    now = datetime.now(timezone.utc)
    data = {
        "schema": "npa.leisaac.session.v1",
        "run_id": "leisaac-live-1",
        "provider": "nebius-kubernetes",
        "task": LEISAAC_TASK,
        "teleop_device": "keyboard",
        "signal_host": "8.8.8.8",
        "signal_port": LEISAAC_SIGNAL_PORT,
        "media_host": "1.1.1.1",
        "media_server": "1.1.1.1",
        "media_port": LEISAAC_MEDIA_PORT,
        "turn_port": LEISAAC_TURN_PORT,
        "turn_relay_port": LEISAAC_TURN_RELAY_PORT,
        "turn_relay_max_port": LEISAAC_TURN_RELAY_MAX_PORT,
        "service_url": "http://8.8.8.8:8080",
        "session_nonce": "a" * 64,
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(hours=1)).isoformat(),
        "source_version": "0.4.0",
        "source_commit": "1" * 40,
        "isaac_sim_version": "5.1.0.0",
        "isaac_lab_version": "2.3.2.post1",
        "image": "registry.example/npa-leisaac@sha256:" + "2" * 64,
        "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    }
    data.update(overrides)
    return data


def _normalized(**overrides):
    manifest, reason = normalize_manifest(
        _manifest(**overrides), expected_run_id="leisaac-live-1"
    )
    assert reason == ""
    assert manifest is not None
    return manifest


def _manifest_v2(**overrides):
    nonce = "a" * 64
    data = _manifest(
        schema="npa.leisaac.session.v2",
        session_attestation=hashlib.sha256(
            f"npa-leisaac-session:{nonce}".encode()
        ).hexdigest(),
        task_registry_fingerprint=REGISTRY_FINGERPRINT,
        environment={
            "id": "kitchen-a",
            "index": 3,
            "seed": 47,
            "num_envs": 1,
            "model": "named-sequential",
        },
        dataset={
            "output_path": "s3://bucket/datasets/leisaac",
            "format": "LeRobotDataset",
            "lerobot_version": "0.5.1",
            "codebase_version": "v3.0",
        },
    )
    data.update(overrides)
    return data


def test_selected_run_requires_safe_exact_identifier() -> None:
    assert (
        selected_run_id({"sim_viz": {"active_run_id": "leisaac-live-1"}})
        == "leisaac-live-1"
    )
    assert selected_run_id({}, "../../etc/passwd") == ""
    assert selected_run_id({"sim_viz": {"run_id": "other"}}, "explicit") == "explicit"
    assert (
        selected_run_id(
            {
                "leisaac": {"run_id": "leisaac-live-1"},
                "sim_viz": {"run_id": "unrelated-artifact-run"},
            }
        )
        == "leisaac-live-1"
    )


def test_manifest_artifact_loader_requires_one_bounded_canonical_object() -> None:
    payload = b'{"schema":"npa.leisaac.session.v1"}'

    class S3:
        def get_object(self, **_kwargs):
            return {"Body": io.BytesIO(payload)}

    artifact = SimpleNamespace(key="runs/live/reports/leisaac-session.json")
    loaded = load_manifest_artifact(
        "live",
        validate_run_id=lambda value: value,
        s3_client=lambda: (S3(), {"prefix": "runs"}),
        s3_buckets=lambda _s3, _settings: ["bucket"],
        find_artifacts=lambda *_args, **_kwargs: ("bucket", [artifact]),
    )
    assert loaded == {"schema": "npa.leisaac.session.v1"}

    historical = SimpleNamespace(
        key="runs/live/reports/leisaac-session.json/live/reports/leisaac-session.json"
    )
    historical_loaded = load_manifest_artifact(
        "live",
        validate_run_id=lambda value: value,
        s3_client=lambda: (S3(), {"prefix": "runs"}),
        s3_buckets=lambda _s3, _settings: ["bucket"],
        find_artifacts=lambda *_args, **_kwargs: ("bucket", [historical]),
    )
    assert historical_loaded == {"schema": "npa.leisaac.session.v1"}
    preferred_loaded = load_manifest_artifact(
        "live",
        validate_run_id=lambda value: value,
        s3_client=lambda: (S3(), {"prefix": "runs"}),
        s3_buckets=lambda _s3, _settings: ["bucket"],
        find_artifacts=lambda *_args, **_kwargs: ("bucket", [historical, artifact]),
    )
    assert preferred_loaded == {"schema": "npa.leisaac.session.v1"}
    nested_historical = SimpleNamespace(
        key=("runs/live/reports/live/reports/leisaac-session.json")
    )
    exact_canonical_loaded = load_manifest_artifact(
        "live",
        validate_run_id=lambda value: value,
        s3_client=lambda: (S3(), {"prefix": "runs"}),
        s3_buckets=lambda _s3, _settings: ["bucket"],
        find_artifacts=lambda *_args, **_kwargs: (
            "bucket",
            [nested_historical, artifact],
        ),
    )
    assert exact_canonical_loaded == {"schema": "npa.leisaac.session.v1"}
    shortest_canonical_loaded = load_manifest_artifact(
        "live",
        validate_run_id=lambda value: value,
        s3_client=lambda: (S3(), {}),
        s3_buckets=lambda _s3, _settings: ["bucket"],
        find_artifacts=lambda *_args, **_kwargs: (
            "bucket",
            [nested_historical, artifact],
        ),
    )
    assert shortest_canonical_loaded == {"schema": "npa.leisaac.session.v1"}

    duplicated = load_manifest_artifact(
        "live",
        validate_run_id=lambda value: value,
        s3_client=lambda: (S3(), {}),
        s3_buckets=lambda _s3, _settings: ["bucket"],
        find_artifacts=lambda *_args, **_kwargs: ("bucket", [artifact, artifact]),
    )
    assert duplicated is None


@pytest.mark.parametrize(
    "override,reason_fragment",
    [
        ({"task": "Isaac-Cartpole-v0"}, "supported task"),
        ({"teleop_device": "so101leader"}, "keyboard"),
        ({"signal_host": "127.0.0.1"}, "network contract"),
        ({"service_url": "http://1.1.1.1:8080"}, "service endpoint"),
        ({"signal_port": "not-an-int"}, "signaling port"),
        ({"media_port": 80}, "media port"),
        ({"image": "registry.example/npa-leisaac:latest"}, "digest"),
        ({"source_commit": "main"}, "source commit"),
    ],
)
def test_manifest_failures_suppress_capability(override, reason_fragment: str) -> None:
    manifest, reason = normalize_manifest(
        _manifest(**override), expected_run_id="leisaac-live-1"
    )
    assert manifest is None
    assert reason_fragment in reason


def test_expired_or_cross_run_manifest_is_rejected() -> None:
    now = datetime.now(timezone.utc)
    assert normalize_manifest(_manifest(), expected_run_id="different")[0] is None
    manifest, reason = normalize_manifest(
        _manifest(expires_at=(now - timedelta(seconds=1)).isoformat()),
        expected_run_id="leisaac-live-1",
        now=now,
    )
    assert manifest is None
    assert "expired" in reason


def test_manifest_without_expiry_remains_lifecycle_gated() -> None:
    now = datetime.now(timezone.utc)
    normalized, reason = normalize_manifest(
        _manifest(expires_at=None),
        expected_run_id="leisaac-live-1",
        now=now,
    )

    assert reason == ""
    assert normalized is not None
    assert normalized["expires_at"] == ""


def test_legacy_v1_manifest_preserves_a_legitimate_zero_seed() -> None:
    normalized, reason = normalize_manifest(
        _manifest(seed=0), expected_run_id="leisaac-live-1"
    )
    assert reason == ""
    assert normalized is not None
    assert normalized["seed"] == 0


def test_agent_relay_manifest_accepts_only_fixed_loopback_tcp_contract() -> None:
    normalized, reason = normalize_manifest(
        _manifest(
            transport="agent-relay",
            signal_host="127.0.0.1",
            media_server="10.96.0.5",
            service_url="http://127.0.0.1:48080",
        ),
        expected_run_id="leisaac-live-1",
    )

    assert reason == ""
    assert normalized is not None
    assert normalized["transport"] == "agent-relay"
    assert normalized["signal_host"] == "127.0.0.1"

    for override in (
        {"signal_host": "127.0.0.2"},
        {"service_url": "http://127.0.0.1:8080"},
        {"service_url": "http://127.0.0.2:48080"},
        {"service_url": "http://169.254.169.254:48080"},
        {"turn_port": 80},
        {"turn_relay_port": 65535},
        {"turn_relay_max_port": 65535},
    ):
        relay_values = {
            "transport": "agent-relay",
            "signal_host": "127.0.0.1",
            "media_server": "10.96.0.5",
            "service_url": "http://127.0.0.1:48080",
        }
        relay_values.update(override)
        rejected, rejected_reason = normalize_manifest(
            _manifest(**relay_values),
            expected_run_id="leisaac-live-1",
        )
        assert rejected is None
        assert rejected_reason


def test_live_health_attestation_gates_secret_free_status() -> None:
    manifest = _normalized()
    health, reason = validate_health(
        manifest,
        {
            "schema": "npa.leisaac.health.v1",
            "state": "ready",
            "webrtc_ready": True,
            "run_id": manifest["run_id"],
            "task": manifest["task"],
            "source_commit": manifest["source_commit"],
            "session_nonce": manifest["session_nonce"],
            "signal_port": LEISAAC_SIGNAL_PORT,
            "pid": 42,
            "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        },
    )
    assert reason == ""
    payload = status_payload(manifest, health)
    assert payload["available"] is True
    assert payload["signaling_server"] == "same-origin"
    assert payload["signaling_path"] == "/api/leisaac/signal"
    assert payload["video_datachannel_url"] == "/api/leisaac/transport/video-webrtc"
    assert payload["control_datachannel_url"] == "/api/leisaac/transport/control-webrtc"
    assert payload["preferred_control_transport"] == "websocket-v1"
    assert payload["media_server"] == "1.1.1.1"
    serialized = repr(payload)
    assert manifest["session_nonce"] not in serialized
    assert manifest["service_url"] not in serialized


def test_v2_manifest_and_health_bind_task_environment_dataset_and_recorder() -> None:
    manifest, reason = normalize_manifest(
        _manifest_v2(), expected_run_id="leisaac-live-1"
    )
    assert reason == "" and manifest is not None
    recorder = {
        "state": "recording",
        "dataset_uri": "s3://bucket/datasets/leisaac",
        "dataset_version_uri": "",
        "task": LEISAAC_TASK,
        "environment_id": "kitchen-a",
        "environment_index": 3,
        "seed": 47,
        "active_episode": "episode-uuid",
        "last_episode_index": None,
        "frame_count": 12,
        "completed_episode_count": 2,
        "pending_outcome": "",
        "last_outcome": "success",
        "last_upload_status": "recording",
        "last_error": "",
    }
    health, reason = validate_health(
        manifest,
        {
            "schema": "npa.leisaac.health.v2",
            "state": "ready",
            "stream_ready": True,
            "stream_transport": "jpeg-poll",
            "run_id": manifest["run_id"],
            "task": manifest["task"],
            "source_commit": manifest["source_commit"],
            "session_attestation": manifest["session_attestation"],
            "task_registry_fingerprint": REGISTRY_FINGERPRINT,
            "environment_id": "kitchen-a",
            "environment_index": 3,
            "seed": 47,
            "signal_port": LEISAAC_SIGNAL_PORT,
            "selected_bundles": {
                "robot": {
                    "bundle_sha256": "b" * 64,
                    "name": "custom-so101",
                    "entrypoint": "robot.usda",
                }
            },
            "recorder": recorder,
        },
    )
    assert reason == "" and health is not None
    payload = status_payload(manifest, health)
    assert payload["environment_id"] == "kitchen-a"
    assert payload["dataset_uri"] == "s3://bucket/datasets/leisaac"
    assert payload["recorder"]["frame_count"] == 12
    assert payload["selected_bundles"]["robot"]["name"] == "custom-so101"
    assert payload["robot"] == "custom-so101"
    assert payload["scene"] == "table_with_cube"
    assert payload["device"] == "browser_keyboard_so101"
    assert payload["task"] == "LeIsaac-SO101-LiftCube-v0"
    assert payload["configuration"]["robot"]["source"] == "uploaded-bundle"
    assert payload["configuration"]["scene"]["source"] == "built-in-runtime"
    assert "session_nonce" not in repr(payload)

    stale = dict(_manifest_v2())
    stale["task_registry_fingerprint"] = "0" * 64
    assert normalize_manifest(stale, expected_run_id="leisaac-live-1")[0] is None


def test_agent_relay_status_returns_only_derived_session_turn_credential() -> None:
    manifest = _normalized(
        transport="agent-relay",
        signal_host="127.0.0.1",
        media_server="10.96.0.5",
        service_url="http://127.0.0.1:48080",
    )
    health = {
        "state": "ready",
        "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "started_at": "2026-08-04T00:00:00Z",
    }

    payload = status_payload(manifest, health)

    assert payload["transport"] == "agent-relay"
    assert payload["media_server"] == "10.96.0.5"
    assert payload["ice_transport_policy"] == "relay"
    assert payload["ice_servers"] == [
        {
            "urls": ["turn:1.1.1.1:3478?transport=udp"],
            "username": "leisaac-live-1",
            "credential": hashlib.sha256(
                ("npa-leisaac-turn:" + "a" * 64).encode()
            ).hexdigest(),
        }
    ]
    assert manifest["session_nonce"] not in repr(payload)


def test_health_nonce_or_readiness_mismatch_suppresses_tab() -> None:
    manifest = _normalized()
    health, reason = validate_health(
        manifest,
        {
            "schema": "npa.leisaac.health.v1",
            "state": "ready",
            "webrtc_ready": True,
            "run_id": manifest["run_id"],
            "task": manifest["task"],
            "source_commit": manifest["source_commit"],
            "session_nonce": "b" * 64,
            "signal_port": LEISAAC_SIGNAL_PORT,
        },
    )
    assert health is None
    assert "session_nonce" in reason
    assert status_payload(manifest, health, reason=reason)["available"] is False


def test_authenticated_backend_routes_gate_status_and_proxy_client(monkeypatch) -> None:
    raw_manifest = _manifest()
    client_content = b"window.OVWebStreamingLibrary={};"
    monkeypatch.setattr(
        "npa.agent_backend.leisaac_routes.LEISAAC_CLIENT_JS_SHA256",
        hashlib.sha256(client_content).hexdigest(),
    )

    class FakeResponse:
        def __init__(self, payload=None, *, content=b"", status_code=200):
            self._payload = payload
            self.content = content
            self.status_code = status_code

        def json(self):
            return self._payload

    def http_get(url: str, **_kwargs):
        if url.endswith("/status"):
            return FakeResponse(
                {
                    "schema": "npa.leisaac.health.v1",
                    "state": "ready",
                    "webrtc_ready": True,
                    "run_id": raw_manifest["run_id"],
                    "task": raw_manifest["task"],
                    "source_commit": raw_manifest["source_commit"],
                    "session_nonce": raw_manifest["session_nonce"],
                    "signal_port": LEISAAC_SIGNAL_PORT,
                    "stream_transport": "websocket-v1",
                    "pid": 42,
                    "input_events": 17,
                }
            )
        if url.endswith("/frame.jpg"):
            return FakeResponse(content=b"\xff\xd8" + b"frame" * 3000 + b"\xff\xd9")
        return FakeResponse(content=client_content)

    posted = []

    def http_post(url: str, **kwargs):
        posted.append((url, kwargs))
        if url.endswith("/transport/video-webrtc"):
            return FakeResponse(
                {
                    "v": 1,
                    "type": "answer",
                    "sdp": "v=0\r\nm=application 9 UDP/DTLS/SCTP webrtc-datachannel\r\n",
                },
                status_code=200,
            )
        return FakeResponse(status_code=202)

    existing_bundle_selection = {
        "robot": {
            "bundle_sha256": "a" * 64,
            "name": "custom-so101",
            "entrypoint": "robot.usda",
        }
    }
    state = {
        "sim_viz": {"active_run_id": raw_manifest["run_id"]},
        "leisaac": {"bundle_selection": existing_bundle_selection},
    }
    saved_states = []
    api = FastAPI()
    register_leisaac_routes(
        api,
        LeIsaacDeps(
            load_state=lambda: state,
            save_state=lambda value: saved_states.append(dict(value)),
            resolve_manifest=lambda run_id: (
                raw_manifest if run_id == raw_manifest["run_id"] else None
            ),
            http_get=http_get,
            http_post=http_post,
            response=Response,
            websocket_connect=lambda *_args, **_kwargs: None,
        ),
    )
    client = TestClient(api)
    websocket_routes = {
        route.path: route for route in api.routes if hasattr(route, "endpoint")
    }
    assert (
        get_type_hints(websocket_routes["/leisaac/signal"].endpoint)["websocket"]
        is WebSocket
    )
    assert (
        get_type_hints(websocket_routes["/leisaac/signal/{signal_path:path}"].endpoint)[
            "websocket"
        ]
        is WebSocket
    )
    assert (
        get_type_hints(websocket_routes["/leisaac/backhaul"].endpoint)["websocket"]
        is WebSocket
    )
    missing = client.get("/leisaac/status", params={"run_id": "other"})
    assert missing.status_code == 200
    assert missing.json()["available"] is False
    insecure = client.get("/leisaac/status", params={"run_id": raw_manifest["run_id"]})
    assert insecure.json()["available"] is False
    assert "HTTPS" in insecure.json()["reason"]
    status = client.get(
        "/leisaac/status",
        params={"run_id": raw_manifest["run_id"]},
        headers={"x-forwarded-proto": "https"},
    )
    assert status.status_code == 200
    assert status.json()["available"] is True
    assert status.json()["input_events"] == 17
    assert status.json()["video_datachannel_url"] == "/api/leisaac/transport/video-webrtc"
    assert status.headers["cache-control"] == "private, no-store"
    forbidden_selection = client.post(
        "/leisaac/select",
        headers={"x-forwarded-proto": "https"},
        json={"run_id": raw_manifest["run_id"]},
    )
    assert forbidden_selection.status_code == 403
    selection = client.post(
        "/leisaac/select",
        headers={
            "x-forwarded-proto": "https",
            "x-npa-leisaac-control": "1",
        },
        json={"run_id": raw_manifest["run_id"]},
    )
    assert selection.status_code == 200
    assert selection.json() == {
        "selected": True,
        "run_id": raw_manifest["run_id"],
        "available": True,
    }
    assert saved_states[-1]["leisaac"] == {
        "run_id": raw_manifest["run_id"],
        "bundle_selection": existing_bundle_selection,
    }
    ws_session_headers = {
        "host": "testserver",
        "origin": "https://testserver",
        "x-forwarded-proto": "https",
        "x-npa-leisaac-control": "1",
        "x-real-ip": "8.8.8.8",
    }
    ws_session = client.post(
        "/leisaac/ws-session",
        params={"run_id": raw_manifest["run_id"]},
        headers=ws_session_headers,
    )
    assert ws_session.status_code == 204
    assert ws_session.content == b""
    cookies = ws_session.headers.get_list("set-cookie")
    assert len(cookies) == 2
    assert any(
        "npa_leisaac_control_ws=" in cookie
        and "Path=/api/leisaac/transport/control" in cookie
        for cookie in cookies
    )
    assert any(
        "npa_leisaac_video_ws=" in cookie
        and "Path=/api/leisaac/transport/video" in cookie
        for cookie in cookies
    )
    assert all("HttpOnly" in cookie for cookie in cookies)
    assert all("Max-Age=120" in cookie for cookie in cookies)
    assert all("SameSite=strict" in cookie for cookie in cookies)
    assert all("Secure" in cookie for cookie in cookies)
    forbidden_ws_session = client.post(
        "/leisaac/ws-session",
        params={"run_id": raw_manifest["run_id"]},
        headers={**ws_session_headers, "origin": "https://evil.example"},
    )
    assert forbidden_ws_session.status_code == 403
    fetch_metadata_session = client.post(
        "/leisaac/ws-session",
        params={"run_id": raw_manifest["run_id"]},
        headers={
            **ws_session_headers,
            "origin": "",
            "referer": "https://testserver/",
            "sec-fetch-site": "same-origin",
        },
    )
    assert fetch_metadata_session.status_code == 204
    cross_site_metadata_session = client.post(
        "/leisaac/ws-session",
        params={"run_id": raw_manifest["run_id"]},
        headers={
            **ws_session_headers,
            "origin": "",
            "referer": "https://testserver/",
            "sec-fetch-site": "cross-site",
        },
    )
    assert cross_site_metadata_session.status_code == 403
    cross_site_video_offer = client.post(
        "/leisaac/transport/video-webrtc",
        headers={
            **ws_session_headers,
            "origin": "https://evil.example",
            "content-type": "application/json",
        },
        json={"v": 1, "run_id": raw_manifest["run_id"], "type": "offer", "sdp": "v=0"},
    )
    assert cross_site_video_offer.status_code == 403
    video_offer = client.post(
        "/leisaac/transport/video-webrtc",
        headers={
            **ws_session_headers,
            "origin": "https://testserver",
            "content-type": "application/json",
        },
        json={
            "v": 1,
            "run_id": raw_manifest["run_id"],
            "type": "offer",
            "sdp": "v=0\r\nm=application 9 UDP/DTLS/SCTP webrtc-datachannel\r\n",
        },
    )
    assert video_offer.status_code == 200
    assert video_offer.json()["type"] == "answer"
    video_post = next(item for item in posted if item[0].endswith("/transport/video-webrtc"))
    assert video_post[1]["headers"] == {
        "X-NPA-LeIsaac-Nonce": raw_manifest["session_nonce"],
        "X-NPA-LeIsaac-Run-ID": raw_manifest["run_id"],
    }
    assert video_post[1]["follow_redirects"] is False
    state["sim_viz"] = {"active_run_id": "unrelated-artifact-run"}
    remembered = client.get("/leisaac/status", headers={"x-forwarded-proto": "https"})
    assert remembered.status_code == 200
    assert remembered.json()["available"] is True
    assert remembered.json()["run_id"] == raw_manifest["run_id"]
    module = client.get(
        "/leisaac/client/index.js", params={"run_id": raw_manifest["run_id"]}
    )
    assert module.status_code == 200
    assert module.headers["cache-control"] == "private, no-store"
    assert module.content == client_content
    frame = client.get(
        "/leisaac/frame.jpg",
        params={"run_id": raw_manifest["run_id"]},
        headers={"x-forwarded-proto": "https"},
    )
    assert frame.status_code == 200
    assert frame.headers["content-type"] == "image/jpeg"
    assert frame.content.startswith(b"\xff\xd8")
    control = client.post(
        "/leisaac/input",
        params={"run_id": raw_manifest["run_id"]},
        headers={
            "x-forwarded-proto": "https",
            "x-npa-leisaac-control": "1",
        },
        json={"key": "W", "event": "press"},
    )
    assert control.status_code == 202
    input_post = next(item for item in posted if item[0].endswith("/input"))
    assert input_post[1]["json"] == {"key": "W", "event": "press"}
    assert input_post[1]["headers"] == {
        "X-NPA-LeIsaac-Nonce": raw_manifest["session_nonce"]
    }
    recorder_control = client.post(
        "/leisaac/recorder",
        params={"run_id": raw_manifest["run_id"]},
        headers={
            "x-forwarded-proto": "https",
            "x-npa-leisaac-control": "1",
        },
        json={"command": "mark-success", "request_id": "route-test-command"},
    )
    assert recorder_control.status_code == 202
    recorder_post = next(item for item in posted if item[0].endswith("/recorder/control"))
    assert recorder_post[1]["json"] == {
        "command": "mark-success",
        "request_id": "route-test-command",
    }
    rejected_control = client.post(
        "/leisaac/input",
        params={"run_id": raw_manifest["run_id"]},
        headers={"x-forwarded-proto": "https"},
        json={"key": "W", "event": "press"},
    )
    assert rejected_control.status_code == 403
    monkeypatch.setattr(
        "npa.agent_backend.leisaac_routes.LEISAAC_CLIENT_JS_SHA256", "0" * 64
    )
    rejected = client.get(
        "/leisaac/client/index.js", params={"run_id": raw_manifest["run_id"]}
    )
    assert rejected.status_code == 502
    assert "integrity" in rejected.json()["detail"]


def test_successful_immutable_manifest_is_cached_across_request_paths(
    monkeypatch,
) -> None:
    raw_manifest = _manifest(expires_at=None)
    calls: list[str] = []
    state = {"leisaac": {"run_id": raw_manifest["run_id"]}}
    clock = [100.0]
    monkeypatch.setattr(
        "npa.agent_backend.leisaac_routes.time.monotonic", lambda: clock[0]
    )

    class Healthy:
        status_code = 200

        @staticmethod
        def json():
            return {
                "schema": "npa.leisaac.health.v1",
                "state": "ready",
                "webrtc_ready": True,
                "run_id": raw_manifest["run_id"],
                "task": raw_manifest["task"],
                "source_commit": raw_manifest["source_commit"],
                "session_nonce": raw_manifest["session_nonce"],
                "signal_port": LEISAAC_SIGNAL_PORT,
            }

    def resolve(run_id: str):
        calls.append(run_id)
        return raw_manifest if run_id == raw_manifest["run_id"] else None

    api = FastAPI()
    register_leisaac_routes(
        api,
        LeIsaacDeps(
            load_state=lambda: state,
            save_state=lambda value: state.update(value),
            resolve_manifest=resolve,
            http_get=lambda *_args, **_kwargs: Healthy(),
            response=Response,
            websocket_connect=lambda *_args, **_kwargs: None,
        ),
    )
    client = TestClient(api)
    headers = {"x-forwarded-proto": "https"}
    assert client.get("/leisaac/status", headers=headers).json()["available"]
    clock[0] += 6.0
    assert client.get("/leisaac/status", headers=headers).json()["available"]
    selection = client.post(
        "/leisaac/select",
        headers={**headers, "x-npa-leisaac-control": "1"},
        json={"run_id": raw_manifest["run_id"]},
    )
    assert selection.status_code == 200
    assert calls == [raw_manifest["run_id"]]

    for _ in range(2):
        missing = client.get(
            "/leisaac/status", params={"run_id": "missing-run"}, headers=headers
        )
        assert missing.json()["available"] is False
    assert calls == [raw_manifest["run_id"], "missing-run"]


def test_signaling_proxy_preserves_only_upstream_sign_in_path() -> None:
    raw_manifest = _manifest()
    health_checks_off_event_loop: list[bool] = []

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "schema": "npa.leisaac.health.v1",
                "state": "ready",
                "webrtc_ready": True,
                "run_id": raw_manifest["run_id"],
                "task": raw_manifest["task"],
                "source_commit": raw_manifest["source_commit"],
                "session_nonce": raw_manifest["session_nonce"],
                "signal_port": LEISAAC_SIGNAL_PORT,
            }

    class FakeUpstream:
        subprotocol = None

        def __init__(self):
            self.sent_initial = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def send(self, _message):
            return None

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.sent_initial:
                self.sent_initial = True
                return '{"ackid":1}'
            raise StopAsyncIteration

    connected = []

    def connect(uri, **_kwargs):
        connected.append(uri)
        return FakeUpstream()

    def http_get(*_args, **_kwargs):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            health_checks_off_event_loop.append(True)
        else:
            health_checks_off_event_loop.append(False)
        return FakeResponse()

    api = FastAPI()
    register_leisaac_routes(
        api,
        LeIsaacDeps(
            load_state=lambda: {},
            resolve_manifest=lambda _run_id: raw_manifest,
            http_get=http_get,
            response=Response,
            websocket_connect=connect,
        ),
    )
    client = TestClient(api)
    headers = {
        "x-forwarded-proto": "https",
        "origin": "https://testserver",
        "host": "testserver",
    }
    query = f"run_id={raw_manifest['run_id']}&peer_id=browser-1&version=2"
    with client.websocket_connect(
        f"/leisaac/signal/sign_in?{query}", headers=headers
    ) as websocket:
        assert websocket.receive_text() == '{"ackid":1}'
    assert connected == [f"ws://8.8.8.8:{LEISAAC_SIGNAL_PORT}/sign_in?{query}"]
    assert health_checks_off_event_loop == [True]

    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect(
            f"/leisaac/signal/arbitrary?run_id={raw_manifest['run_id']}",
            headers=headers,
        ):
            pass
    assert exc_info.value.code == 1008

    for origin in (None, "https://foreign.example", "null", "not a url"):
        rejected_headers = {"x-forwarded-proto": "https", "host": "testserver"}
        if origin is not None:
            rejected_headers["origin"] = origin
        with pytest.raises(WebSocketDisconnect) as rejected:
            with client.websocket_connect(
                f"/leisaac/signal?run_id={raw_manifest['run_id']}",
                headers=rejected_headers,
            ):
                pass
        assert rejected.value.code == 1008


@pytest.mark.parametrize(
    "headers,allowed",
    [
        (
            {
                "x-forwarded-proto": "https",
                "host": "agent.example:8443",
                "origin": "https://agent.example:8443",
            },
            True,
        ),
        (
            {
                "x-forwarded-proto": "https",
                "host": "agent.example:8443",
                "origin": "https://agent.example",
            },
            False,
        ),
        (
            {"x-forwarded-proto": "https", "host": "agent.example", "origin": "null"},
            False,
        ),
        (
            {
                "x-forwarded-proto": "https",
                "host": "agent.example",
                "origin": "https://agent.example/path",
            },
            False,
        ),
        (
            {
                "x-forwarded-proto": "http",
                "host": "agent.example",
                "origin": "https://agent.example",
            },
            False,
        ),
    ],
)
def test_signaling_origin_validation_honors_forwarded_proto_host_and_port(
    headers, allowed
) -> None:
    assert _same_https_origin(headers) is allowed


@pytest.mark.anyio
async def test_selection_storage_work_does_not_block_the_shared_event_loop(
    anyio_backend,
) -> None:
    assert anyio_backend == "asyncio"
    raw_manifest = _manifest()
    entered = threading.Event()
    release = threading.Event()

    def load_state():
        entered.set()
        assert release.wait(timeout=5)
        return {}

    class Healthy:
        status_code = 200

        @staticmethod
        def json():
            return {
                "schema": "npa.leisaac.health.v1",
                "state": "ready",
                "webrtc_ready": True,
                "run_id": raw_manifest["run_id"],
                "task": raw_manifest["task"],
                "source_commit": raw_manifest["source_commit"],
                "session_nonce": raw_manifest["session_nonce"],
                "signal_port": LEISAAC_SIGNAL_PORT,
                "pid": 42,
            }

    api = FastAPI()

    @api.get("/probe")
    async def probe():
        return {"responsive": True}

    register_leisaac_routes(
        api,
        LeIsaacDeps(
            load_state=load_state,
            save_state=lambda _state: None,
            resolve_manifest=lambda _run_id: raw_manifest,
            http_get=lambda *_args, **_kwargs: Healthy(),
            response=Response,
            websocket_connect=lambda *_args, **_kwargs: None,
        ),
    )
    transport = httpx.ASGITransport(app=api)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://agent"
    ) as client:
        selection = asyncio.create_task(
            client.post(
                "/leisaac/select",
                headers={"x-forwarded-proto": "https", "x-npa-leisaac-control": "1"},
                json={"run_id": raw_manifest["run_id"]},
            )
        )
        await asyncio.to_thread(entered.wait, 5)
        assert (await client.get("/probe")).json() == {"responsive": True}
        release.set()
        assert (await selection).status_code == 200


def test_exception_logs_keep_tracebacks_but_redact_untrusted_details(caplog) -> None:
    secret = "offer=private-ice&token=credential&s3=https://private.example"
    deps = LeIsaacDeps(
        load_state=lambda: {"leisaac": {"run_id": "leisaac-live-1"}},
        resolve_manifest=lambda _run_id: (_ for _ in ()).throw(RuntimeError(secret)),
        http_get=lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError(secret)),
        response=Response,
        websocket_connect=lambda *_args, **_kwargs: None,
    )
    with caplog.at_level(logging.WARNING, logger="npa.agent_backend.leisaac_routes"):
        assert _resolve(deps, "leisaac-live-1")[0] is None
        assert _health(deps, _normalized())[0] is None
    assert secret not in caplog.text
    assert "LeIsaac capability resolution failed" in caplog.text
    assert "LeIsaac health request failed" in caplog.text
    assert "_RedactedException" in caplog.text


def test_backhaul_rejects_browser_shape_before_accept_and_accepts_scoped_pod(
    monkeypatch,
) -> None:
    api = FastAPI()
    register_leisaac_routes(
        api,
        LeIsaacDeps(
            load_state=lambda: {},
            resolve_manifest=lambda _run_id: None,
            http_get=lambda *_args, **_kwargs: None,
            response=Response,
            websocket_connect=lambda *_args, **_kwargs: None,
        ),
    )
    client = TestClient(api)
    for headers, subprotocols, suffix in (
        (
            {
                "x-forwarded-proto": "https",
                "x-real-ip": "8.8.8.8",
                "origin": "https://testserver",
            },
            ["npa.leisaac.backhaul.v1"],
            "",
        ),
        ({"x-forwarded-proto": "https", "x-real-ip": "8.8.8.8"}, [], ""),
        (
            {"x-forwarded-proto": "https", "x-real-ip": "127.0.0.1"},
            ["npa.leisaac.backhaul.v1"],
            "",
        ),
        (
            {"x-forwarded-proto": "https", "x-real-ip": "8.8.8.8"},
            ["npa.leisaac.backhaul.v1"],
            "?unexpected=1",
        ),
    ):
        with pytest.raises(WebSocketDisconnect) as rejected:
            with client.websocket_connect(
                "/leisaac/backhaul" + suffix,
                headers=headers,
                subprotocols=subprotocols,
            ):
                pass
        assert rejected.value.code == 1008

    class Reader:
        async def readexactly(self, _size):
            await asyncio.Future()

    writers = []

    class Writer:
        def __init__(self):
            self.closed = False
            self.waited = False
            self.settled = threading.Event()

        def close(self):
            self.closed = True

        async def wait_closed(self):
            self.waited = True
            self.settled.set()

    async def open_connection(*_args, **_kwargs):
        writer = Writer()
        writers.append(writer)
        return Reader(), writer

    monkeypatch.setattr(
        "npa.agent_backend.leisaac_routes.asyncio.open_connection", open_connection
    )
    for _ in range(10):
        with client.websocket_connect(
            "/leisaac/backhaul",
            headers={"x-forwarded-proto": "https", "x-real-ip": "8.8.8.8"},
            subprotocols=["npa.leisaac.backhaul.v1"],
        ) as websocket:
            assert websocket.accepted_subprotocol == "npa.leisaac.backhaul.v1"
            websocket.close()
            assert writers[-1].settled.wait(5), "backhaul cleanup did not complete"
    assert len(writers) == 10
    assert all(writer.closed and writer.waited for writer in writers)


def test_video_relay_credits_runtime_before_browser_ack() -> None:
    """The one-slot relay, not a browser round trip, owns runtime flow credit."""

    raw_manifest = _manifest()
    jpeg = b"\xff\xd8" + b"relay-frame" * 30 + b"\xff\xd9"
    frame = pack_frame(
        FrameEnvelope(
            sequence=7,
            capture_wall_ns=1,
            capture_monotonic_ns=2,
            encoded_wall_ns=3,
            encoded_monotonic_ns=4,
            runtime_send_monotonic_ns=5,
        ),
        jpeg,
    )

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "schema": "npa.leisaac.health.v1",
                "state": "ready",
                "stream_ready": True,
                "stream_transport": "websocket-v1",
                "run_id": raw_manifest["run_id"],
                "task": raw_manifest["task"],
                "source_commit": raw_manifest["source_commit"],
                "session_nonce": raw_manifest["session_nonce"],
                "signal_port": LEISAAC_SIGNAL_PORT,
            }

    upstream_closed = threading.Event()

    class FakeUpstream:
        subprotocol = VIDEO_SUBPROTOCOL

        def __init__(self):
            self.sent: list[str] = []
            self.first = True

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            upstream_closed.set()
            return None

        async def send(self, message):
            self.sent.append(message)

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.first:
                self.first = False
                return frame
            await asyncio.Event().wait()
            raise StopAsyncIteration

    upstream = FakeUpstream()
    api = FastAPI()
    register_leisaac_routes(
        api,
        LeIsaacDeps(
            load_state=lambda: {"leisaac": {"run_id": raw_manifest["run_id"]}},
            resolve_manifest=lambda _run_id: raw_manifest,
            http_get=lambda *_args, **_kwargs: FakeResponse(),
            response=Response,
            websocket_connect=lambda *_args, **_kwargs: upstream,
        ),
    )
    client = TestClient(api)
    session_headers = {
        "host": "testserver",
        "origin": "https://testserver",
        "x-forwarded-proto": "https",
        "x-npa-leisaac-control": "1",
        "x-real-ip": "8.8.8.8",
    }
    session = client.post(
        "/leisaac/ws-session",
        params={"run_id": raw_manifest["run_id"]},
        headers=session_headers,
    )
    video_cookie = next(
        cookie
        for cookie in session.headers.get_list("set-cookie")
        if cookie.startswith("npa_leisaac_video_ws=")
    )
    token = video_cookie.split("npa_leisaac_video_ws=", 1)[1].split(";", 1)[0]
    websocket_headers = {
        **session_headers,
        "cookie": f"npa_leisaac_video_ws={token}",
    }
    websocket_headers.pop("x-npa-leisaac-control")
    with client.websocket_connect(
        f"/leisaac/transport/video?run_id={raw_manifest['run_id']}",
        headers=websocket_headers,
        subprotocols=[VIDEO_SUBPROTOCOL],
    ) as websocket:
        assert websocket.receive_bytes().endswith(jpeg)
        assert len(upstream.sent) == 1
        assert json.loads(upstream.sent[0]) == {
            "v": 1,
            "type": "frame-ack",
            "run_id": raw_manifest["run_id"],
            "sequence": 7,
        }
        websocket.send_json(
            {
                "v": 1,
                "type": "frame-ack",
                "run_id": raw_manifest["run_id"],
                "sequence": 7,
            }
        )
        websocket.close()
        assert upstream_closed.wait(5), "video relay cleanup did not complete"

    status = client.get(
        "/leisaac/status",
        params={"run_id": raw_manifest["run_id"]},
        headers={"x-forwarded-proto": "https"},
    ).json()
    assert status["agent_transport_metrics"]["frames_relay_acked"] == 1
    assert status["agent_transport_metrics"]["frames_browser_acked"] == 1
    assert len(upstream.sent) == 1
