"""LeIsaac runtime, Kubernetes, assets, EULA, and GPU guardrails."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
import threading
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

try:  # tomllib is stdlib from 3.11; the repo still supports 3.10 via tomli.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

from npa.agent_backend.leisaac import LEISAAC_CLIENT_JS_SHA256
from npa.agent_backend.leisaac_transport import AsyncLatestByKey, unpack_frame
from npa.workbench.leisaac import (
    GPU_PRODUCT,
    GPU_PRODUCT_LABEL,
    GPU_PROVIDER_LABEL,
    GPU_PROVIDER_VALUE,
    MEDIA_PORT,
    TURN_PORT,
    TURN_RELAY_PORT,
    TURN_RELAY_MAX_PORT,
    TRANSPORT_AGENT_RELAY,
    LeIsaacConfigError,
    deployment_manifest,
    relay_service_manifest,
    relay_client_secret_manifest,
    session_manifest,
)

ROOT = Path(__file__).resolve().parents[3]
IMAGE = "registry.example/npa-leisaac@sha256:" + "1" * 64
NONCE = "a" * 64
OUTPUT_PATH = "s3://bucket/datasets/leisaac"
RECORDER_SECRET = "leisaac-live-recorder"


def _session_server_module():
    path = ROOT / "npa/docker/workbench/leisaac/session_server.py"
    spec = importlib.util.spec_from_file_location("npa_leisaac_session_server", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _client_archive(path: Path, client_js: bytes) -> None:
    with tarfile.open(path, mode="w:gz") as bundle:
        for name, content in (
            ("package/dist/omniverse-webrtc-streaming-library.umd.cjs", client_js),
            ("package/LICENSE.txt", b"NVIDIA test license"),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(content)
            bundle.addfile(member, io.BytesIO(content))


def test_runtime_datachannel_offer_is_private_authenticated_and_run_bound(
    monkeypatch,
) -> None:
    module = _session_server_module()
    run_id = "live-datachannel"
    nonce = "d" * 64
    monkeypatch.setenv("NPA_LEISAAC_RUN_ID", run_id)
    monkeypatch.setenv("NPA_LEISAAC_SESSION_NONCE", nonce)
    observed = {}

    async def create_answer(**kwargs):
        observed.update(kwargs)
        return {
            "v": 1,
            "type": "answer",
            "sdp": "v=0\r\nm=application 9 UDP/DTLS/SCTP webrtc-datachannel\r\n",
        }

    monkeypatch.setattr(module.VIDEO_DATACHANNEL_PEERS, "create_answer", create_answer)
    client = TestClient(module.build_app())
    payload = {
        "v": 1,
        "run_id": run_id,
        "type": "offer",
        "sdp": "v=0\r\nm=application 9 UDP/DTLS/SCTP webrtc-datachannel\r\n",
    }
    forbidden = client.post("/transport/video-webrtc", json=payload)
    assert forbidden.status_code == 403
    response = client.post(
        "/transport/video-webrtc",
        headers={
            "X-NPA-LeIsaac-Nonce": nonce,
            "X-NPA-LeIsaac-Run-ID": run_id,
        },
        json=payload,
    )
    assert response.status_code == 200
    assert response.json()["type"] == "answer"
    assert observed["ice_server"] is None
    assert observed["metrics"] is module.TRANSPORT_METRICS
    assert hasattr(observed["frame_source"](), "__aiter__")


def test_runtime_control_datachannel_offer_uses_shared_control_handler(
    monkeypatch,
) -> None:
    module = _session_server_module()
    run_id = "live-control-datachannel"
    nonce = "c" * 64
    monkeypatch.setenv("NPA_LEISAAC_RUN_ID", run_id)
    monkeypatch.setenv("NPA_LEISAAC_SESSION_NONCE", nonce)
    observed = {}

    async def create_answer(**kwargs):
        observed.update(kwargs)
        return {
            "v": 1,
            "type": "answer",
            "sdp": "v=0\r\nm=application 9 UDP/DTLS/SCTP webrtc-datachannel\r\n",
        }

    monkeypatch.setattr(
        module.CONTROL_DATACHANNEL_PEERS, "create_answer", create_answer
    )
    client = TestClient(module.build_app())
    payload = {
        "v": 1,
        "run_id": run_id,
        "type": "offer",
        "sdp": "v=0\r\nm=application 9 UDP/DTLS/SCTP webrtc-datachannel\r\n",
    }
    assert client.post("/transport/control-webrtc", json=payload).status_code == 403
    response = client.post(
        "/transport/control-webrtc",
        headers={
            "X-NPA-LeIsaac-Nonce": nonce,
            "X-NPA-LeIsaac-Run-ID": run_id,
        },
        json=payload,
    )
    assert response.status_code == 200
    assert observed["channel_handler"] is module._serve_control_datachannel
    assert observed["metrics"] is module.TRANSPORT_METRICS


def test_runtime_public_surfaces_require_nonce_while_health_endpoints_are_minimal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _session_server_module()
    nonce = "e" * 64
    monkeypatch.setenv("NPA_LEISAAC_SESSION_NONCE", nonce)
    client_root = tmp_path / "client"
    client_root.mkdir()
    (client_root / "index.js").write_text(
        "window.testClient = true;\n", encoding="utf-8"
    )
    provenance = tmp_path / "provenance.json"
    provenance.write_text('{"schema":"test-provenance"}\n', encoding="utf-8")
    monkeypatch.setattr(module, "CLIENT_ROOT", client_root)
    monkeypatch.setattr(module, "PROVENANCE_PATH", provenance)
    client = TestClient(module.build_app())

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json() == {"ok": True}
    assert "run_id" not in health.text and "dataset" not in health.text
    monkeypatch.setitem(module.STATE, "state", "starting")
    readiness = client.get("/readyz")
    assert readiness.status_code == 503
    assert readiness.json() == {"ready": False}
    monkeypatch.setitem(module.STATE, "state", "ready")
    readiness = client.get("/readyz")
    assert readiness.status_code == 200
    assert readiness.json() == {"ready": True}
    assert "run_id" not in readiness.text and "dataset" not in readiness.text
    for path in ("/status", "/provenance", "/client/index.js"):
        assert client.get(path).status_code == 403
        assert client.get(path, headers={"X-Real-IP": "8.8.8.8"}).status_code == 403
    headers = {"X-NPA-LeIsaac-Nonce": nonce, "X-Real-IP": "8.8.8.8"}
    assert client.get("/status", headers=headers).status_code in {200, 503}
    assert client.get("/provenance", headers=headers).status_code == 200
    assert client.get("/client/index.js", headers=headers).status_code == 200


def test_runtime_configuration_and_recorder_require_active_controller_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _session_server_module()
    nonce = "f" * 64
    lease_id = "a" * 64
    monkeypatch.setenv("NPA_LEISAAC_SESSION_NONCE", nonce)
    module.CONTROL_OWNER.update(
        token="owner-token",
        client_id="owner-browser",
        lease_id=lease_id,
        lease_generation=1,
    )
    applied: list[dict] = []
    recorded: list[tuple[str, str]] = []

    def apply(selection):
        applied.append(selection)
        return selection

    def record(command, request_id):
        recorded.append((command, request_id))
        return 202, {"accepted": True, "request_id": request_id}

    monkeypatch.setattr(module, "apply_bundle_selection", apply)
    monkeypatch.setattr(module, "enqueue_recorder_command", record)
    client = TestClient(module.build_app())
    nonce_headers = {"X-NPA-LeIsaac-Nonce": nonce}

    for path, payload in (
        ("/bundles/apply", {"selection": {}}),
        ("/recorder/control", {"command": "start", "request_id": "lease-test"}),
    ):
        missing = client.post(path, headers=nonce_headers, json=payload)
        assert missing.status_code == 409
        assert missing.json()["code"] == "controller_busy"
        second_client = client.post(
            path,
            headers={
                **nonce_headers,
                "X-NPA-LeIsaac-Client-ID": "other-browser",
                "X-NPA-LeIsaac-Lease-ID": "b" * 64,
            },
            json=payload,
        )
        assert second_client.status_code == 409
        assert second_client.json()["code"] == "controller_busy"

    owner_headers = {
        **nonce_headers,
        "X-NPA-LeIsaac-Client-ID": "owner-browser",
        "X-NPA-LeIsaac-Lease-ID": lease_id,
    }
    assert (
        client.post(
            "/bundles/apply", headers=owner_headers, json={"selection": {}}
        ).status_code
        == 202
    )
    assert (
        client.post(
            "/recorder/control",
            headers=owner_headers,
            json={"command": "start", "request_id": "lease-test"},
        ).status_code
        == 202
    )
    assert applied == [{}]
    assert recorded == [("start", "lease-test")]

    # A nonce-authenticated agent request still cannot mutate runtime state
    # without the active browser controller lease.
    restore_headers = {
        **nonce_headers,
        "X-NPA-LeIsaac-System-Restore": "1",
    }
    assert (
        client.post(
            "/bundles/apply", headers=restore_headers, json={"selection": {}}
        ).status_code
        == 409
    )
    restore_recorder = client.post(
        "/recorder/control",
        headers=restore_headers,
        json={"command": "start", "request_id": "restore-cannot-record"},
    )
    assert restore_recorder.status_code == 409


def test_runtime_datachannel_source_coalesces_stale_causal_frames() -> None:
    module = _session_server_module()
    module.FRAME_LATEST = AsyncLatestByKey(("workspace", "overview"))

    def item(sequence: int):
        jpeg = b"\xff\xd8" + bytes([sequence]) * 8 + b"\xff\xd9"
        metadata = {
            "sequence": sequence,
            "capture_wall_ns": 100 + sequence,
            "capture_monotonic_ns": 200 + sequence,
            "encoded_wall_ns": 300 + sequence,
            "encoded_monotonic_ns": 400 + sequence,
            "causal_action_sequence": 40 + sequence,
            "causal_applied_monotonic_ns": 500 + sequence,
            "sha256": hashlib.sha256(jpeg).hexdigest(),
        }
        return "workspace", metadata, jpeg

    async def verify() -> None:
        await module.FRAME_LATEST.publish("workspace", item(1))
        source = module._video_datachannel_frames()
        first, _content = unpack_frame(await anext(source))
        assert first.sequence == 1
        await module.FRAME_LATEST.publish("workspace", item(2))
        await module.FRAME_LATEST.publish("workspace", item(3))
        newest, content = unpack_frame(await anext(source))
        assert newest.sequence == 3
        assert newest.causal_action_sequence == 43
        assert newest.causal_applied_monotonic_ns == 503
        assert newest.dropped_before == 1
        assert content == item(3)[2]
        await source.aclose()

    asyncio.run(verify())


def test_deployment_is_real_rt_core_leisaac_and_operator_eula_runtime_config() -> None:
    deployment = deployment_manifest(
        run_id="live-1",
        namespace="default",
        image=IMAGE,
        media_host="1.1.1.1",
        session_nonce=NONCE,
        recorder_secret=RECORDER_SECRET,
    )
    pod = deployment["spec"]["template"]["spec"]
    assert pod["runtimeClassName"] == "nvidia"
    assert "nodeSelector" not in pod
    terms = pod["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]["nodeSelectorTerms"]
    assert terms == [
        {
            "matchExpressions": [
                {
                    "key": GPU_PRODUCT_LABEL,
                    "operator": "In",
                    "values": [GPU_PRODUCT],
                }
            ]
        },
        {
            "matchExpressions": [
                {
                    "key": GPU_PROVIDER_LABEL,
                    "operator": "In",
                    "values": [GPU_PROVIDER_VALUE],
                }
            ]
        },
    ]
    container = pod["containers"][0]
    assert container["resources"]["requests"]["cpu"] == "16"
    assert container["resources"]["limits"]["cpu"] == "32"
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert container["securityContext"]["runAsNonRoot"] is True
    env = {item["name"]: item["value"] for item in container["env"] if "value" in item}
    assert env["ACCEPT_EULA"] == "Y"
    assert "OMNI_KIT_ACCEPT_EULA" not in env
    assert "ISAACSIM_ACCEPT_EULA" not in env
    assert env["NPA_LEISAAC_MEDIA_HOST"] == "1.1.1.1"
    assert "/readyz" == container["readinessProbe"]["httpGet"]["path"]
    assert container["livenessProbe"]["failureThreshold"] == 30
    assert "hostPort" not in next(
        port for port in container["ports"] if port["name"] == "media"
    )


def test_agent_relay_service_is_private_clusterip_with_cleanup_metadata() -> None:
    service = relay_service_manifest(
        run_id="live-1",
        namespace="leisaac",
        agent_project="rtxpro",
        agent_name="agent",
        source_ranges=["8.8.8.8/32"],
        turn_peer_source="9.9.8.0/22",
    )

    assert service["spec"]["type"] == "ClusterIP"
    assert "loadBalancerSourceRanges" not in service["spec"]
    assert {item["name"] for item in service["spec"]["ports"]} == {
        "status",
        "signal",
        "media",
    }
    annotations = service["metadata"]["annotations"]
    assert annotations == {
        "npa.nebius.com/agent-project": "rtxpro",
        "npa.nebius.com/agent-name": "agent",
        "npa.nebius.com/source-ranges": "8.8.8.8/32",
        "npa.nebius.com/turn-peer-source": "9.9.8.0/22",
    }


def test_agent_relay_client_is_secret_mounted_as_non_gpu_sidecar() -> None:
    secret = relay_client_secret_manifest(
        run_id="live-relay",
        namespace="leisaac",
        agent_host="8.8.8.8",
        session_nonce=NONCE,
        certificate_sha256="b" * 64,
        auth_user="npa",
        auth_password="secret",
        client_source="print('client')\n",
    )
    assert secret["kind"] == "Secret"
    assert secret["stringData"]["config.json"]
    assert secret["stringData"]["NPA_LEISAAC_SESSION_NONCE"] == NONCE
    assert "listening-port=3478" in secret["stringData"]["turnserver.conf"]
    assert "min-port=47999" in secret["stringData"]["turnserver.conf"]
    assert "max-port=48015" in secret["stringData"]["turnserver.conf"]
    assert "total-quota=16" in secret["stringData"]["turnserver.conf"]
    assert "user-quota=16" in secret["stringData"]["turnserver.conf"]
    assert NONCE not in secret["stringData"]["turnserver.conf"]
    deployment = deployment_manifest(
        run_id="live-relay",
        namespace="leisaac",
        image=IMAGE,
        media_host="8.8.8.8",
        session_nonce=NONCE,
        media_server="10.96.0.5",
        relay_client_secret=secret["metadata"]["name"],
        recorder_secret=RECORDER_SECRET,
    )
    pod = deployment["spec"]["template"]["spec"]
    assert deployment["spec"]["strategy"] == {"type": "Recreate"}
    sidecar = pod["containers"][1]
    assert sidecar["name"] == "agent-relay-client"
    assert "nvidia.com/gpu" not in sidecar["resources"]["requests"]
    assert pod["volumes"][-1]["secret"]["secretName"] == secret["metadata"]["name"]
    media = next(
        port for port in pod["containers"][0]["ports"] if port["name"] == "media"
    )
    assert media["containerPort"] == MEDIA_PORT
    assert "hostPort" not in media
    env = {item["name"]: item for item in pod["containers"][0]["env"]}
    assert env["NPA_LEISAAC_MEDIA_HOST"]["valueFrom"] == {
        "fieldRef": {"fieldPath": "status.podIP"}
    }
    turn = pod["containers"][2]
    assert turn["name"] == "turn"
    assert turn["command"] == ["sh", "-c"]
    assert "--listening-ip=${NPA_LEISAAC_POD_IP}" in turn["args"][0]
    assert "--listening-ip=127.0.0.1" in turn["args"][0]
    assert "--relay-ip=${NPA_LEISAAC_POD_IP}" in turn["args"][0]
    assert "--allowed-peer-ip=${NPA_LEISAAC_POD_IP}" in turn["args"][0]
    assert "NPA_LEISAAC_MEDIA_SERVER" not in {item["name"] for item in turn["env"]}
    assert turn["env"] == [
        {
            "name": "NPA_LEISAAC_POD_IP",
            "valueFrom": {
                "fieldRef": {
                    "apiVersion": "v1",
                    "fieldPath": "status.podIP",
                }
            },
        },
    ]
    assert "@sha256:" in turn["image"]
    assert {item["containerPort"] for item in turn["ports"]} == {
        TURN_PORT,
        TURN_RELAY_PORT,
    }
    assert {item["name"] for item in turn["ports"]} == {
        "turn-control",
        "turn-ctrl-tcp",
        "turn-media",
    }
    assert all(len(item["name"]) <= 15 for item in turn["ports"])


def test_agent_relay_deployment_requires_resolved_private_service_ip() -> None:
    with pytest.raises(
        LeIsaacConfigError, match="agent relay media server must be an IP"
    ):
        deployment_manifest(
            run_id="live-relay",
            namespace="leisaac",
            image=IMAGE,
            media_host="8.8.8.8",
            session_nonce=NONCE,
            relay_client_secret="live-relay-relay-client",
            recorder_secret=RECORDER_SECRET,
        )

    with pytest.raises(LeIsaacConfigError, match="private IPv4"):
        deployment_manifest(
            run_id="live-relay",
            namespace="leisaac",
            image=IMAGE,
            media_host="8.8.8.8",
            media_server="8.8.4.4",
            session_nonce=NONCE,
            relay_client_secret="live-relay-relay-client",
            recorder_secret=RECORDER_SECRET,
        )


def test_agent_relay_manifest_keeps_tcp_private_and_media_on_agent_public_ip() -> None:
    manifest = session_manifest(
        run_id="live-relay",
        image=IMAGE,
        signal_host="127.0.0.1",
        media_host="8.8.8.8",
        session_nonce=NONCE,
        media_server="10.96.0.5",
        transport=TRANSPORT_AGENT_RELAY,
        output_path=OUTPUT_PATH,
    )

    assert manifest["transport"] == TRANSPORT_AGENT_RELAY
    assert manifest["signal_host"] == "127.0.0.1"
    assert manifest["service_url"] == "http://127.0.0.1:48080"
    assert manifest["media_host"] == "8.8.8.8"
    assert manifest["media_server"] == "10.96.0.5"
    assert manifest["turn_port"] == TURN_PORT
    assert manifest["turn_relay_port"] == TURN_RELAY_PORT
    assert manifest["turn_relay_max_port"] == TURN_RELAY_MAX_PORT


def test_agent_relay_rejects_non_loopback_signal_or_missing_agent_identity() -> None:
    with pytest.raises(LeIsaacConfigError, match="127.0.0.1"):
        session_manifest(
            run_id="live-relay",
            image=IMAGE,
            signal_host="8.8.8.8",
            media_host="8.8.8.8",
            session_nonce=NONCE,
            media_server="10.96.0.5",
            transport=TRANSPORT_AGENT_RELAY,
            output_path=OUTPUT_PATH,
        )
    with pytest.raises(LeIsaacConfigError, match="private IPv4"):
        session_manifest(
            run_id="live-relay",
            image=IMAGE,
            signal_host="127.0.0.1",
            media_host="8.8.8.8",
            session_nonce=NONCE,
            media_server="8.8.8.8",
            transport=TRANSPORT_AGENT_RELAY,
            output_path=OUTPUT_PATH,
        )
    with pytest.raises(LeIsaacConfigError, match="agent project and name"):
        relay_service_manifest(
            run_id="live-relay",
            namespace="default",
            source_ranges=["8.8.8.8/32"],
        )
    with pytest.raises(LeIsaacConfigError, match="public IPv4 CIDR"):
        relay_service_manifest(
            run_id="live-relay",
            namespace="default",
            agent_project="rtxpro",
            agent_name="agent",
            source_ranges=["8.8.8.8/32"],
            turn_peer_source="2001:4860:4860::8888/32",
        )


def test_manifest_records_exact_real_component_and_provenance() -> None:
    manifest = session_manifest(
        run_id="live-1",
        image=IMAGE,
        signal_host="127.0.0.1",
        media_host="1.1.1.1",
        media_server="10.96.0.5",
        session_nonce=NONCE,
        expires_at=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
        output_path=OUTPUT_PATH,
    )
    assert manifest["task"] == "LeIsaac-SO101-LiftCube-v0"
    assert manifest["teleop_device"] == "keyboard"
    assert manifest["configuration"]["robot"]["id"] == "so101_follower"
    assert manifest["configuration"]["scene"]["id"] == "table_with_cube"
    assert manifest["configuration"]["device"]["id"] == "browser_keyboard_so101"
    assert manifest["configuration"]["task"]["id"] == manifest["task"]
    assert manifest["source_commit"] == "1651c321e9b0c1bb54233211fc7b3cd70d8373d5"
    assert manifest["isaac_sim_version"] == "5.1.0.0"
    assert manifest["isaac_lab_version"] == "2.3.2.post1"
    assert manifest["image"] == IMAGE


def test_manifest_has_no_implicit_session_time_limit() -> None:
    manifest = session_manifest(
        run_id="live-unbounded",
        image=IMAGE,
        signal_host="127.0.0.1",
        media_host="1.1.1.1",
        media_server="10.96.0.5",
        session_nonce=NONCE,
        output_path=OUTPUT_PATH,
    )

    assert "expires_at" not in manifest


@pytest.mark.parametrize("value", ["", "latest", "x:tag", "x@sha256:bad"])
def test_image_must_be_digest_pinned(value: str) -> None:
    with pytest.raises(LeIsaacConfigError, match="digest"):
        deployment_manifest(
            run_id="live-1",
            namespace="default",
            image=value,
            media_host="1.1.1.1",
            session_nonce=NONCE,
            recorder_secret=RECORDER_SECRET,
        )


def test_private_media_endpoint_is_rejected() -> None:
    with pytest.raises(LeIsaacConfigError, match="public"):
        deployment_manifest(
            run_id="live-1",
            namespace="default",
            image=IMAGE,
            media_host="127.0.0.1",
            session_nonce=NONCE,
            recorder_secret=RECORDER_SECRET,
        )


def test_container_never_bakes_eula_client_or_assets() -> None:
    dockerfile = (ROOT / "npa/docker/workbench/leisaac/Dockerfile").read_text(
        encoding="utf-8"
    )
    server = (ROOT / "npa/docker/workbench/leisaac/session_server.py").read_text(
        encoding="utf-8"
    )
    instructions = "\n".join(
        line for line in dockerfile.splitlines() if not line.lstrip().startswith("#")
    )
    assert "ENV OMNI_KIT_ACCEPT_EULA" not in instructions
    assert "ENV ISAACSIM_ACCEPT_EULA" not in instructions
    copy_lines = [
        line
        for line in instructions.splitlines()
        if line.lstrip().startswith(("COPY ", "ADD "))
    ]
    assert not any(
        "so101_follower.usd" in line or "kitchen_with_orange" in line
        for line in copy_lines
    )
    assert "CLIENT_SHA512" in server and "CLIENT_JS_SHA256" in server
    assert "CLIENT_SOURCE_JS_SHA256" in server
    assert "5.6.0" in server
    assert LEISAAC_CLIENT_JS_SHA256 in server
    assert "CLIENT_WSS_PATCH_OLD" not in server
    assert "CLIENT_WSS_PATCH_NEW" not in server
    assert "vendor bytes remain pristine" in server
    assert 'f"--/app/livestream/publicEndpointPort={MEDIA_PORT}"' in server
    assert '"--/app/livestream/webrtc/logQosStatus=true"' in server
    assert 'f"--/app/livestream/fixedHostPort={MEDIA_PORT}"' in server
    assert 'f"--/app/livestream/minHostPort={MEDIA_PORT}"' in server
    assert 'f"--/app/livestream/maxHostPort={MEDIA_PORT}"' in server
    assert 'f"--/app/livestream/port={SIGNAL_PORT}"' in server
    assert "ROBOT_SHA256" in server and "KITCHEN_SHA256" in server
    assert "safe_extract_zip" in server and "safe_extract_client" in server
    assert '"--device=cuda:0"' in server
    assert 'f"--seed={TELEOP_SEED}"' in server
    assert 'module_root = "/opt/npa/leisaac"' in server
    assert 'environment["PYTHONPATH"]' in server
    assert 'environment["NPA_LEISAAC_BROWSER_TELEOP"] = "1"' in server
    assert "stdin=subprocess.DEVNULL" in server
    assert "start_new_session=True" in server
    assert "READY_PATH.is_file()" in server and "HEARTBEAT_PATH.is_file()" in server
    assert 'update_state(detail="warming RTX renderer")' in server
    assert 'requested_video_transport="webrtc-kit-h264"' in server
    assert (
        'active_video_transport=("webrtc-kit-h264"ifhardwareelse"jpeg-websocket")'
        in "".join(server.split())
    )
    assert '"--/renderer/multiGpu/enabled=False"' in server
    assert "NPA_LEISAAC_INPUT_COUNTER" in server
    assert "NPA_LEISAAC_APPLIED_COUNTER" in server
    assert "NPA_LEISAAC_INPUT_QUEUE" in server
    assert "NPA_LEISAAC_FRAME_PATH" in server
    assert "pandas==2.3.3" in dockerfile
    assert "aiortc==1.15.0" in dockerfile
    assert (
        "leisaac_datachannel.py /opt/npa/leisaac/leisaac_datachannel.py" in dockerfile
    )
    assert "EXPOSE 8080/tcp 49100/tcp 47998/udp" in dockerfile
    assert "feetech-servo-sdk" in dockerfile and "-m pip check" in dockerfile
    assert "sed -i" not in dockerfile
    assert "upstream-packaging.patch" in dockerfile
    assert "THIRD_PARTY_NOTICES.md" in dockerfile
    assert "git -C /opt/leisaac apply --recount --check --unidiff-zero" in dockerfile
    assert (
        "/opt/leisaac/source/leisaac/leisaac/devices/keyboard/so101_keyboard.py"
        in dockerfile
    )
    assert (
        "/opt/leisaac/scripts/environments/teleoperation/teleop_se3_agent.py"
        in dockerfile
    )
    assert '"self._remote_pulses.clear()" in m["reset"]' in dockerfile
    assert '"_remote_pulses" not in m["_add_device_control_description"]' in dockerfile
    assert os.access(ROOT / "npa/docker/workbench/leisaac/build.sh", os.X_OK)

    notices = (ROOT / "npa/docker/workbench/leisaac/THIRD_PARTY_NOTICES.md").read_text(
        encoding="utf-8"
    )
    pyproject = (ROOT / "npa/pyproject.toml").read_text(encoding="utf-8")
    assert "imageio-ffmpeg 0.6.0" in notices and "FFmpeg" in notices
    assert "pygame 2.6.1" in notices and "LGPL-2.1" in notices
    assert "leisaac = [" in pyproject and '"imageio-ffmpeg==0.6.0"' in pyproject
    extras = tomllib.loads(pyproject)["project"]["optional-dependencies"]
    assert extras["full"] == []
    assert extras["leisaac"] == ["imageio-ffmpeg==0.6.0"]
    assert "imageio-ffmpeg==0.6.0" in extras["dev"]


def test_observability_patch_is_exact_and_records_real_upstream_input() -> None:
    patch = ROOT / "npa/docker/workbench/leisaac/upstream-observability.patch"
    packaging_patch = ROOT / "npa/docker/workbench/leisaac/upstream-packaging.patch"
    server = _session_server_module()

    assert hashlib.sha256(patch.read_bytes()).hexdigest() == (
        server.UPSTREAM_OBSERVABILITY_PATCH_SHA256
    )
    assert hashlib.sha256(packaging_patch.read_bytes()).hexdigest() == (
        server.UPSTREAM_PACKAGING_PATCH_SHA256
    )
    assert '-    "feetech-servo-sdk",' in packaging_patch.read_text(encoding="utf-8")
    source = patch.read_text(encoding="utf-8")
    assert "source/leisaac/leisaac/devices/keyboard/so101_keyboard.py" in source
    assert "def get_device_state(self):" in source
    assert "self._delta_action + remote_action" in source
    assert "NPA_LEISAAC_INPUT_COUNTER" in source
    assert "NPA_LEISAAC_IPC_EVENT_PATH" in source
    assert "def notify_runtime_frame(metadata, frame_bytes):" in source
    assert "+    single_primary_capture_fps = 10.0 if native_video else 16.0" in source
    assert "+        if primary_due or secondary_due:" in source
    assert "(not native_video or recording_active)" not in source
    assert (
        "self._advance_counter(self._applied_counter, len(acknowledgements))" in source
    )
    assert "target.writelines(" in source
    assert "os.fsync(target.fileno())" in source
    assert "NPA_LEISAAC_READY_PATH" in source
    assert "NPA_LEISAAC_BROWSER_TELEOP" in source
    assert "self._browser_teleop =" in source
    assert (
        "def _on_keyboard_event(self, event, *args, **kwargs):\n"
        "+        # Native livestream viewers can emit unauthenticated Kit input."
        in source
    )
    assert "+        if self._browser_teleop:\n+            return" in source
    assert '"task": args_cli.task' in source
    assert (
        " env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)\n"
        '+    if os.environ.get("NPA_LEISAAC_BROWSER_TELEOP") == "1":' in source
    )
    assert (
        " env: ManagerBasedRLEnv | DirectRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped\n"
        '+    if os.environ.get("NPA_LEISAAC_BROWSER_TELEOP") == "1":\n'
        "+        env.cfg.sim.render_interval = 1_000_000_000" in source
    )
    assert "capture_viewport_to_buffer" in source
    assert 'workspace_camera_path = "/OmniverseKit_Persp"' in source
    assert '"viewport_camera_id": "workspace"' in source
    assert "UsdGeom.Camera.Define" not in source
    assert 'image.save(encoded, format="JPEG", quality=82, optimize=True)' in source
    assert "optimize=False" not in source
    assert "image=image" in source
    assert "await asyncio.to_thread(" not in source
    assert "ThreadPoolExecutor(max_workers=1" in source
    assert (
        "capture_encoder.submit(\n+                encode_and_publish_capture" in source
    )
    assert "def poll_encoded_capture():" in source
    assert "encode_and_publish_capture" in source
    assert "recorder.shutdown(wait=True)" in source
    assert " env.reset()\n+    recorder = EpisodeRecorder(" in source
    assert 'str(applied.get("event") or "") == "release"' in source
    assert "source_queue.pop(0)" in source
    assert 'if capture_state["active"]:' in source
    assert (
        "Submit only after Kit reports the prior GPU capture fully complete" in source
    )
    assert "await capture_helper.wait_for_result(completion_frames=0)" in source
    assert source.count("viewport.camera_path = workspace_camera_path") == 2
    assert source.count("schedule_browser_capture()\n+        env.render()") == 1
    assert (
        source.count(
            "apply_view_command()\n+                schedule_browser_capture()"
        )
        == 1
    )
    assert "def apply_mode_command():" in source
    assert "RecordingCameraMode.PRIMARY_AND_SECONDARY" in source
    assert "def browser_capture_needs_render():" in source
    assert (
        "def browser_capture_needs_render():\n"
        '+        """Keep physics/control cadence independent from background RTX work."""\n'
        "+        if native_video:\n"
        "+            now = time.monotonic()\n"
        '+            if now < capture_state["next_native_render_at"]:\n'
        "+                return False\n"
        '+            capture_state["next_native_render_at"] = now + 1.0 / 30.0\n'
        "+            return True\n"
        '+        if capture_state["encode_future"] is not None:\n'
        "+            return False\n"
        "+        return bool(\n"
        '+            capture_state["active"]' in source
    )
    assert (
        "capture_encoder.shutdown(wait=True, cancel_futures=True)\n"
        "+        poll_encoded_capture()\n"
        "+        recorder.shutdown(wait=True)\n"
        "+        ipc_event_socket.close()\n"
        "         signal.signal(signal.SIGINT, original_sigint_handler)" in source
    )
    assert "and browser_capture_needs_render()" in source
    assert 'capture_state["queue"].clear()' in source
    assert 'capture_state["priority_queue"]' in source
    assert (
        source.count("single_primary_capture_fps = 10.0 if native_video else 16.0") == 1
    )
    assert (
        source.count("dual_primary_capture_fps = 10.0 if native_video else 20.0") == 1
    )
    assert source.count("secondary_capture_fps = 4.0") == 1
    assert 'capture_state["last_causal_at"] = causal_at' in source
    assert 'capture_state["next_at"]["workspace"] = causal_at' in source
    assert (
        'if str(applied.get("event") or "") == "release":\n'
        "+            # Releases remain causal primary work; they never wait for the\n"
        "+            # slower secondary cadence or a stale background request."
        in source
    )
    assert 'time.monotonic() >= capture_state["next_at"]["workspace"]' in source
    assert 'mode_state["applied_view_mode"] == ViewMode.DUAL_SLOW.value' in source
    assert (
        '"causal_action_sequence": capture_result["causal_action_sequence"]' in source
    )
    assert "mark_remote_step_applied(sim_step)" in source
    assert "asyncio.ensure_future" in source
    assert "create_viewport_window" not in source
    assert 'overview_camera_path = "/World/NPAOverviewCamera"' in source
    assert "resolution=(640, 360)" in source
    assert "camera = Camera(" in source
    assert "camera.initialize()" in source
    assert "camera.get_rgba()" in source
    assert "camera.destroy()" in source
    assert "overview_camera_path = workspace_camera_path" not in source
    assert "next_viewport_frame_async" not in source
    assert 'if camera_id == "overview":' in source
    assert "NPA_LEISAAC_INPUT_QUEUE" in source
    assert "NPA_LEISAAC_APPLIED_COUNTER" in source
    assert "NPA_LEISAAC_FRAME_PATH" in source
    assert "self._remote_pulses[key] = 8" in source
    assert "env_cfg.sim.use_fabric = True" in source
    assert "env.cfg.sim.render_interval = 1_000_000_000" in source
    assert "if env is not None:" in source
    assert "rate_limiter.sleep(None if os.environ.get" in source
    assert "env_cfg.observations.policy.wrist = None" in source
    assert "env_cfg.observations.policy.front = None" in source
    assert "env_cfg.scene.wrist = None" in source
    assert "env_cfg.scene.front = None" in source
    assert "env_cfg.events.domain_randomize_4 = None" in source
    assert 'args_cli.task == "LeIsaac-SO101-LiftCube-v0"' in source
    assert "env_cfg.events.domain_randomize_1 = None" in source
    assert "env_cfg.wait_for_textures = False" in source
    assert "NPA_LEISAAC_CUSTOM_ROBOT_USD" in source
    assert "NPA_LEISAAC_CUSTOM_SCENE_USD" in source
    assert "custom USD is outside the verified bundle cache" in source
    assert 'if os.environ.get("NPA_LEISAAC_BROWSER_TELEOP") == "1":' in source
    assert "env.render()" in source


def test_health_reads_upstream_keyboard_counter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _session_server_module()
    counter = tmp_path / "input-events"
    counter.write_text("13\n", encoding="utf-8")
    applied = tmp_path / "applied-inputs"
    applied.write_text("12\n", encoding="utf-8")
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"\xff\xd8frame\xff\xd9")
    server.INPUT_COUNTER_PATH = counter
    server.APPLIED_COUNTER_PATH = applied
    server.FRAME_PATH = frame
    server.STATE.update(
        stream_ready=True,
        stream_transport="webrtc",
        requested_video_transport="webrtc-kit-h264",
        active_video_transport="webrtc-kit-h264",
        video_codec="H264",
        hardware_acceleration="runtime-nvenc",
    )
    mode_status = tmp_path / "view-mode-status.json"
    mode_status.write_text(
        json.dumps(
            {
                **server._default_mode_state(),
                "schema": "npa.leisaac.view-mode.v1",
                "applied_view_mode": "dual_slow",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(server, "MODE_STATUS_PATH", mode_status)

    health = server.health_document()

    assert health["schema"] == "npa.leisaac.health.v2"
    assert health["mode_schema"] == "npa.leisaac.view-mode.v1"
    assert health["input_events"] == 13
    assert health["applied_inputs"] == 12
    assert health["stream_ready"] is True
    assert health["stream_transport"] == "webrtc"
    assert health["active_video_transport"] == "webrtc-kit-h264"
    assert health["video_codec"] == "H264"
    assert health["hardware_acceleration"] == "runtime-nvenc"
    assert health["physics_device"] == "cuda:0"
    assert health["render_device"] == "cuda"
    assert health["seed"] == 42


def test_health_mode_status_cannot_override_authoritative_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _session_server_module()
    mode_status = tmp_path / "view-mode-status.json"
    mode_status.write_text(
        json.dumps(
            {
                "schema": "npa.leisaac.view-mode.v1",
                "applied_view_mode": "dual_slow",
                "run_id": "corrupt-child-run",
                "session_attestation": "corrupt-child-attestation",
                "task": "corrupt-child-task",
                "physics_device": "cpu",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(server, "MODE_STATUS_PATH", mode_status)
    monkeypatch.setenv("NPA_LEISAAC_RUN_ID", "authoritative-run")
    monkeypatch.setenv("NPA_LEISAAC_SESSION_NONCE", NONCE)

    health = server.health_document()

    assert health["run_id"] == "authoritative-run"
    assert health["session_attestation"] != "corrupt-child-attestation"
    assert health["task"] != "corrupt-child-task"
    assert health["physics_device"] == "cuda:0"
    assert health["applied_view_mode"] == "dual_slow"


def test_recorder_control_reservation_prevents_duplicate_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _session_server_module()
    status_path = tmp_path / "status.json"
    control_path = tmp_path / "control.jsonl"
    pending_path = tmp_path / "pending-command.json"
    status_path.write_text(
        json.dumps(
            {
                "state": "idle",
                "last_command_id": "",
                "last_command": "",
            }
        ),
        encoding="utf-8",
    )
    control_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(server, "RECORDER_STATUS_PATH", status_path)
    monkeypatch.setattr(server, "RECORDER_CONTROL_PATH", control_path)
    monkeypatch.setattr(server, "RECORDER_PENDING_PATH", pending_path)
    monkeypatch.setattr(server, "RECORDER_COMMAND_LOCK", threading.Lock())

    first_status, first = server.enqueue_recorder_command("start", "request-1")
    duplicate_status, duplicate = server.enqueue_recorder_command("start", "request-1")
    competing_status, competing = server.enqueue_recorder_command("start", "request-2")

    assert first_status == duplicate_status == 202
    assert first == {
        "accepted": True,
        "duplicate": False,
        "processed": False,
        "request_id": "request-1",
    }
    assert duplicate["duplicate"] is True
    assert competing_status == 409
    assert "in progress" in competing["detail"]
    queued = [json.loads(line) for line in control_path.read_text().splitlines()]
    assert queued == [{"command": "start", "request_id": "request-1"}]

    status_path.write_text(
        json.dumps(
            {
                "state": "recording",
                "last_command_id": "request-1",
                "last_command": "start",
                "processed_commands": {"request-1": "start"},
            }
        ),
        encoding="utf-8",
    )
    acknowledged_status, acknowledged = server.enqueue_recorder_command(
        "start", "request-1"
    )
    assert acknowledged_status == 202
    assert acknowledged["processed"] is True
    assert not pending_path.exists()

    mark_status, _mark = server.enqueue_recorder_command("mark-failure", "request-3")
    assert mark_status == 202
    reused_status, reused = server.enqueue_recorder_command("mark-success", "request-3")
    assert reused_status == 409
    assert "in progress" in reused["detail"]

    pending_path.unlink()
    status_path.write_text(
        json.dumps(
            {
                "state": "outcome-pending",
                "last_command_id": "request-3",
                "last_command": "mark-failure",
                "processed_commands": {
                    "request-1": "start",
                    "request-3": "mark-failure",
                },
            }
        ),
        encoding="utf-8",
    )
    delayed_status, delayed = server.enqueue_recorder_command("start", "request-1")
    assert delayed_status == 202
    assert delayed["duplicate"] is True
    assert delayed["processed"] is True
    reused_status, reused = server.enqueue_recorder_command("mark-success", "request-1")
    assert reused_status == 409
    assert "reused" in reused["detail"]


def _resolve_live_browser_mode(**values: str) -> subprocess.CompletedProcess[str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "NPA_AGENT_RUN_ID",
            "NPA_AGENT_CYPRESS_RUN_ID",
            "NPA_LEISAAC_RUN_ID",
            "NPA_AGENT_CYPRESS_ARTIFACT_KEY",
            "NPA_AGENT_TASK",
            "NPA_AGENT_ENVIRONMENT_ID",
            "NPA_AGENT_COMPLETED_EPISODES",
        }
    }
    environment.update(values)
    return subprocess.run(
        ["bash", str(ROOT / "npa/scripts/run_agent_cypress.sh"), "--resolve-mode"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_live_browser_runner_preserves_legacy_exact_rrd_selector() -> None:
    result = _resolve_live_browser_mode(
        NPA_AGENT_RUN_ID="legacy-rrd-run",
        NPA_AGENT_CYPRESS_ARTIFACT_KEY="reports/sim2real.rrd",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "cy:live-rrd"


@pytest.mark.parametrize("selector", ["explicit", "legacy"])
def test_live_browser_runner_resolves_complete_leisaac_context(selector: str) -> None:
    run_key = "NPA_LEISAAC_RUN_ID" if selector == "explicit" else "NPA_AGENT_RUN_ID"
    result = _resolve_live_browser_mode(
        **{
            run_key: "leisaac-live-run",
            "NPA_AGENT_TASK": "LeIsaac-SO101-LiftCube-v0",
            "NPA_AGENT_ENVIRONMENT_ID": "lift-cube-a",
            "NPA_AGENT_COMPLETED_EPISODES": "0",
        }
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "cy:live-leisaac"


def test_live_browser_runner_rejects_incomplete_leisaac_context() -> None:
    result = _resolve_live_browser_mode(
        NPA_LEISAAC_RUN_ID="leisaac-live-run",
        NPA_AGENT_TASK="LeIsaac-SO101-LiftCube-v0",
    )
    assert result.returncode == 2
    assert "requires NPA_LEISAAC_RUN_ID" in result.stderr


def test_live_browser_runner_rejects_ambiguous_legacy_and_explicit_selectors() -> None:
    result = _resolve_live_browser_mode(
        NPA_AGENT_RUN_ID="legacy-run",
        NPA_LEISAAC_RUN_ID="explicit-run",
        NPA_AGENT_TASK="LeIsaac-SO101-LiftCube-v0",
        NPA_AGENT_ENVIRONMENT_ID="lift-cube-a",
        NPA_AGENT_COMPLETED_EPISODES="0",
    )
    assert result.returncode == 2
    assert "compatibility selector" in result.stderr


def test_mock_browser_runner_ignores_ambient_live_context(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "npm"
    executable.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\"\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    environment = dict(os.environ)
    environment.update(
        PATH=f"{tmp_path}:{environment.get('PATH', '')}",
        NPA_AGENT_TASK="LeIsaac-SO101-LiftCube-v0",
    )
    result = subprocess.run(
        ["bash", str(ROOT / "npa/scripts/run_agent_cypress.sh"), "--mock"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    assert "run cy:mock" in result.stdout


def test_live_browser_runner_includes_leisaac_journey_and_environment_bridge() -> None:
    package = json.loads(
        (ROOT / "npa/tests/browser/package.json").read_text(encoding="utf-8")
    )
    runner = (ROOT / "npa/scripts/run_agent_cypress.sh").read_text(encoding="utf-8")

    assert "cypress/e2e/agent_leisaac_live.cy.js" in package["scripts"]["cy:live"]
    assert 'LIVE_RUN_ID="${NPA_AGENT_CYPRESS_RUN_ID:-}"' in runner
    assert 'LIVE_LEISAAC_RUN_ID="${NPA_LEISAAC_RUN_ID:-}"' in runner
    assert 'CYPRESS_NPA_AGENT_CYPRESS_RUN_ID="${LIVE_RUN_ID}"' in runner
    assert 'CYPRESS_NPA_AGENT_RUN_ID="${LIVE_LEISAAC_RUN_ID}"' in runner
    assert 'CYPRESS_NPA_AGENT_TASK="${LIVE_TASK}"' in runner
    assert 'CYPRESS_NPA_AGENT_ENVIRONMENT_ID="${LIVE_ENVIRONMENT_ID}"' in runner
    assert 'CYPRESS_NPA_AGENT_COMPLETED_EPISODES="${LIVE_COMPLETED_EPISODES}"' in runner


def test_liveness_preserves_live_initial_reset_and_restarts_dead_child() -> None:
    server = _session_server_module()

    server.STATE.update(state="starting", pid=0)
    assert server.liveness_status() == 200
    child = type("Child", (), {"poll": lambda self: None})()
    server.CHILD = child
    server.STATE.update(state="starting", pid=42)
    assert server.liveness_status() == 200
    server.STATE.update(state="ready", pid=42)
    assert server.liveness_status() == 200
    child.poll = lambda: 1
    assert server.liveness_status() == 503
    server.STATE.update(state="restarting", pid=42)
    assert server.liveness_status() == 200
    server.STATE.update(state="failed", pid=0)
    assert server.liveness_status() == 503


def test_frame_stall_revokes_controls_and_forces_safe_runtime_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _session_server_module()
    for name in (
        "READY_PATH",
        "INPUT_COUNTER_PATH",
        "APPLIED_COUNTER_PATH",
        "INPUT_QUEUE_PATH",
        "APPLIED_ACK_PATH",
        "FRAME_PATH",
        "FRAME_META_PATH",
        "SECONDARY_FRAME_PATH",
        "SECONDARY_FRAME_META_PATH",
        "VIEW_COMMAND_PATH",
        "MODE_COMMAND_PATH",
        "MODE_STATUS_PATH",
        "HEARTBEAT_PATH",
    ):
        monkeypatch.setattr(server, name, tmp_path / name.lower())
    monkeypatch.setattr(server, "RECORDER_ROOT", tmp_path / "recorder")
    server.CONTROL_LEDGER = server.ControlLedger()
    server.CONTROL_LEDGER.accept(
        {
            "v": 1,
            "type": "control",
            "run_id": "stall-test",
            "client_id": "browser-test",
            "seq": 1,
            "key": "W",
            "event": "press",
            "client_mono_ns": 1,
            "client_wall_ns": 2,
        }
    )
    server.CONTROL_OWNER.update(
        token="active", client_id="browser-test", lease_id="a" * 64
    )
    server.MODE_COMMAND_PATH.write_text(
        json.dumps(
            {
                "requested_view_mode": "dual_slow",
                "requested_recording_camera_mode": "primary_and_secondary",
                "view_revision": 7,
                "recording_revision": 9,
            }
        ),
        encoding="utf-8",
    )

    assert server._prepare_stall_recovery() == 1
    assert server.CONTROL_OWNER["token"] == ""
    assert server.CONTROL_OWNER["lease_id"] == ""
    assert server.FORCE_SAFE_RESTART.is_set()
    assert server.CONTROL_LEDGER.resume("browser-test")["next_seq"] == 1

    server._reset_runtime_files()
    restored = json.loads(server.MODE_COMMAND_PATH.read_text(encoding="utf-8"))
    assert restored["requested_view_mode"] == "single_fast"
    assert restored["requested_recording_camera_mode"] == "primary_only"
    assert server.FORCE_SAFE_RESTART.is_set() is False


def test_native_video_idle_does_not_trigger_jpeg_frame_watchdog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _session_server_module()
    heartbeat = tmp_path / "heartbeat"
    heartbeat.write_text("1\n", encoding="utf-8")
    monkeypatch.setattr(server, "HEARTBEAT_PATH", heartbeat)
    monkeypatch.setattr(server, "FRAME_PATH", tmp_path / "frame.jpg")

    server.VIDEO_PATH.update(hardware=False, fallback_reason="test")
    assert server._runtime_stream_stalled(now=heartbeat.stat().st_mtime + 31)

    server.VIDEO_PATH.update(hardware=True, fallback_reason="")
    assert not server._runtime_stream_stalled(now=heartbeat.stat().st_mtime + 31)


def test_custom_bundle_apply_is_mocked_at_s3_call_site_and_restart_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _session_server_module()
    recorder = tmp_path / "status.json"
    recorder.write_text('{"state":"idle"}\n', encoding="utf-8")
    monkeypatch.setattr(server, "RECORDER_STATUS_PATH", recorder)
    monkeypatch.setattr(server, "CUSTOM_BUNDLE_ROOT", tmp_path / "custom")
    monkeypatch.setenv("NPA_LEISAAC_OUTPUT_PATH", "s3://bucket/datasets/leisaac")
    calls: list[tuple[str, Path]] = []

    class FakeStore:
        def __init__(self, _client, _uri, *, allowed_buckets):
            assert allowed_buckets == ["bucket"]

        def materialize(self, digest, destination):
            calls.append((digest, destination))
            return {
                "bundle_sha256": digest,
                "kind": "robot",
                "name": "custom-so101",
                "entrypoint": "robot.usda",
                "entrypoint_path": str(destination / "robot.usda"),
            }

    monkeypatch.setattr(server, "BundleStore", FakeStore)
    monkeypatch.setitem(
        sys.modules, "boto3", SimpleNamespace(client=lambda *_args, **_kwargs: object())
    )
    server.BUNDLE_SELECTION.clear()
    server.BUNDLE_RESTART.clear()
    digest = "a" * 64

    selected = server.apply_bundle_selection({"robot": digest})

    assert selected["robot"]["bundle_sha256"] == digest
    assert calls == [(digest, tmp_path / "custom" / digest)]
    assert server.BUNDLE_RESTART.is_set()
    assert server.STATE["state"] == "restarting"
    assert server._mark_runtime_ready() is False
    assert server.STATE["state"] == "restarting"
    server.BUNDLE_RESTART.clear()
    assert server._mark_runtime_ready() is True
    assert server.STATE["state"] == "ready"
    recorder.write_text('{"state":"idle"}\n', encoding="utf-8")
    reset = server.apply_bundle_selection({})
    assert reset == {}
    assert server.BUNDLE_SELECTION == {}
    assert server.STATE["selected_bundles"] == {}
    assert server.STATE["detail"] == "restoring built-in defaults"
    recorder.write_text('{"state":"recording"}\n', encoding="utf-8")
    with pytest.raises(server.BundleError, match="finish or discard"):
        server.apply_bundle_selection({"robot": digest})


def test_agent_bootstrap_installs_turn_without_baking_session_configuration() -> None:
    agent = (ROOT / "npa/src/npa/cli/agent.py").read_text(encoding="utf-8")
    ui = (ROOT / "npa/src/npa/cli/agent_ui.html").read_text(encoding="utf-8")

    assert "ca-certificates coturn" in agent
    assert "leisaac-turn.conf" not in agent
    assert 'iceTransportPolicy: "relay"' in ui
    assert "installLeIsaacPeerConnection(status)" in ui
    assert "installLeIsaacVideoPlayGate(nativeVideo)" in ui
    assert 'track.kind === "video" && track.readyState === "live" && !track.muted' in ui
    assert "now - stableSince >= 250" in ui
    assert 'new DOMException("WebRTC media track did not arrive", "TimeoutError")' in ui


def test_client_is_served_pristine_after_exact_hash_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _session_server_module()
    source = b"pristine-vendor-client"
    archive = tmp_path / "client.tgz"
    destination = tmp_path / "client"
    _client_archive(archive, source)
    monkeypatch.setattr(
        server, "CLIENT_SOURCE_JS_SHA256", hashlib.sha256(source).hexdigest()
    )

    server.safe_extract_client(archive, destination)

    assert (destination / "index.js").read_bytes() == source

    _client_archive(archive, source + b"modified")
    with pytest.raises(RuntimeError, match="source hash mismatch"):
        server.safe_extract_client(archive, destination)


def test_build_script_supports_repository_python_310() -> None:
    script = (ROOT / "npa/docker/workbench/leisaac/build.sh").read_text(
        encoding="utf-8"
    )
    assert "except ModuleNotFoundError:" in script
    assert "import tomli as tomllib" in script


def test_asset_cache_moves_to_the_durable_claim_when_one_exists(monkeypatch) -> None:
    """The USD scenes and streaming client arrive at run time, into an emptyDir.

    A Recreate rollout, a node drain or a restart therefore re-downloads them onto a
    GPU that is already running. Each fetch is hash-verified and skipped when the
    file is present, so a warm cache turns the download into a checksum.
    """

    monkeypatch.setenv("NPA_MODEL_CACHE_PVC", "npa-model-cache")

    deployment = deployment_manifest(
        run_id="live-1",
        namespace="default",
        image=IMAGE,
        media_host="1.1.1.1",
        session_nonce=NONCE,
        recorder_secret=RECORDER_SECRET,
    )

    pod = deployment["spec"]["template"]["spec"]
    assert {
        "name": "npa-model-cache",
        "persistentVolumeClaim": {"claimName": "npa-model-cache"},
    } in pod["volumes"]
    sim = next(c for c in pod["containers"] if c["name"] == "leisaac")
    assert {
        "name": "npa-model-cache",
        "mountPath": "/opt/npa-model-cache",
    } in sim["volumeMounts"]
    env = {item["name"]: item["value"] for item in sim["env"] if "value" in item}
    assert env["NPA_LEISAAC_CACHE_DIR"] == "/opt/npa-model-cache/leisaac"
    assert env["LEISAAC_ASSETS_ROOT"] == "/opt/npa-model-cache/leisaac/assets/runtime"


def test_asset_cache_keeps_its_pod_local_volume_without_a_claim(monkeypatch) -> None:
    for name in ("NPA_MODEL_CACHE_PVC", "NPA_MODEL_CACHE_HOST_PATH", "NPA_MODEL_CACHE_DIR"):
        monkeypatch.delenv(name, raising=False)

    deployment = deployment_manifest(
        run_id="live-1",
        namespace="default",
        image=IMAGE,
        media_host="1.1.1.1",
        session_nonce=NONCE,
        recorder_secret=RECORDER_SECRET,
    )

    pod = deployment["spec"]["template"]["spec"]
    assert "npa-model-cache" not in [volume["name"] for volume in pod["volumes"]]
    sim = next(c for c in pod["containers"] if c["name"] == "leisaac")
    env = {item["name"]: item["value"] for item in sim["env"] if "value" in item}
    assert "NPA_LEISAAC_CACHE_DIR" not in env
