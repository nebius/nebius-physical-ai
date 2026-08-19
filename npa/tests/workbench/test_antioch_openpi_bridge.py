from __future__ import annotations

import json
import importlib.util
import re
import socket
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
import urllib.error

import numpy as np
import pytest
from typer.testing import CliRunner
import yaml

from npa.cli.main import app
from npa.workbench.antioch.openpi_bridge import (
    ACTION_SHAPE,
    IMAGE_SHAPE,
    OpenPIBridgeError,
    OpenPIWebsocketClient,
    RetryPolicy,
    contract_smoke,
    pack_message,
    render_stack,
    safe_position_targets,
    unpack_message,
    validate_actions,
    validate_observation,
)
from npa.workbench.antioch.openpi_health import wait_for_health
from npa.workbench.antioch.openpi_isaac import (
    _camera_frame,
    _close_runtime_resource,
    _capture_viewport_rgb,
    _compatible_franka_asset_url,
    _ensure_franka_asset_root,
    _position_target_tensor,
    _verify_vulkan_runtime,
    _wait_for_camera_observation,
)


def test_position_target_uses_torch_dtype_not_hosted_backend_dtype() -> None:
    sentinel_dtype = object()

    class Tensor:
        value: np.ndarray

        def unsqueeze(self, dimension: int) -> tuple[int, np.ndarray]:
            assert dimension == 0
            return dimension, self.value

    class FakeTorch:
        float32 = sentinel_dtype

        @staticmethod
        def as_tensor(value: np.ndarray, *, device: object, dtype: object) -> Tensor:
            assert device == "cuda:0"
            assert dtype is sentinel_dtype
            tensor = Tensor()
            tensor.value = value
            return tensor

    result = _position_target_tensor(
        FakeTorch,
        np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.5]),
        device="cuda:0",
    )
    assert result[0] == 0
    np.testing.assert_allclose(
        result[1], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.02, 0.02]
    )


def test_camera_observation_warmup_advances_until_complete() -> None:
    attempts = 0
    advances = 0
    now = 0.0

    def advance() -> None:
        nonlocal advances, now
        advances += 1
        now += 0.02

    def capture() -> dict[str, object]:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise OpenPIBridgeError("camera returned invalid RGB shape")
        return {"sequence": attempts}

    result = _wait_for_camera_observation(
        capture,
        advance,
        timeout_seconds=1.0,
        monotonic=lambda: now,
        sleep=lambda _seconds: None,
    )

    assert result == {"sequence": 3}
    assert advances == 3


def test_camera_observation_warmup_fails_closed_at_deadline() -> None:
    now = 0.0

    def advance() -> None:
        nonlocal now
        now += 0.1

    def capture() -> dict[str, object]:
        raise OpenPIBridgeError("empty camera")

    with pytest.raises(OpenPIBridgeError, match="warmup deadline"):
        _wait_for_camera_observation(
            capture,
            advance,
            timeout_seconds=0.25,
            monotonic=lambda: now,
            sleep=lambda _seconds: None,
        )


def test_failed_runtime_does_not_call_kit_owned_close() -> None:
    resource = SimpleNamespace(close=lambda: pytest.fail("close masked failure exit"))

    _close_runtime_resource(resource, failed=True)


def test_successful_runtime_closes_kit_owned_resource() -> None:
    closed: list[bool] = []
    resource = SimpleNamespace(close=lambda: closed.append(True))

    _close_runtime_resource(resource, failed=False)

    assert closed == [True]


def test_hosted_viewport_capture_advances_kit_application_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback: dict[str, object] = {}
    updates = 0
    renders = 0

    def schedule(_viewport: object, on_capture: object) -> object:
        callback["value"] = on_capture
        return object()

    class _App:
        def update(self) -> None:
            nonlocal updates
            updates += 1
            value = callback.pop("value", None)
            if value is not None:
                value(bytes(range(8)), 8, 2, 1, object())

    class _Sim:
        def render(self) -> None:
            nonlocal renders
            renders += 1

    app_module = SimpleNamespace(get_app=lambda: _App())
    renderer_module = SimpleNamespace(
        convert_raw_bytes_to_list=lambda buffer, *_args: list(buffer)
    )
    kit_module = SimpleNamespace(app=app_module, renderer_capture=renderer_module)
    monkeypatch.setitem(sys.modules, "omni", SimpleNamespace(kit=kit_module))
    monkeypatch.setitem(sys.modules, "omni.kit", kit_module)
    monkeypatch.setitem(sys.modules, "omni.kit.app", app_module)
    monkeypatch.setitem(
        sys.modules,
        "omni.kit.renderer_capture",
        renderer_module,
    )
    monkeypatch.setitem(sys.modules, "omni.kit.viewport", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "omni.kit.viewport.utility",
        SimpleNamespace(
            capture_viewport_to_buffer=schedule,
            get_active_viewport=lambda: object(),
        ),
    )
    monkeypatch.setitem(sys.modules, "isaacsim", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "isaacsim.core", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "isaacsim.core.utils", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "isaacsim.core.utils.viewports",
        SimpleNamespace(set_camera_view=lambda **_kwargs: None),
    )

    image = _capture_viewport_rgb(_Sim(), eye=np.zeros(3), target=np.ones(3))

    assert image.shape == (1, 2, 3)
    assert image.tolist() == [[[0, 1, 2], [4, 5, 6]]]
    assert updates == 1
    assert renders == 1


def test_health_module_import_does_not_load_offline_dataset_stack() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import npa.workbench.antioch.openpi_health; "
                "assert 'npa.workbench.antioch.manager' not in sys.modules; "
                "assert 'npa.workbench.antioch.dataset' not in sys.modules; "
                "assert 'npa.workbench.antioch.schemas' not in sys.modules; "
                "assert 'npa.workbench.antioch.openpi_bridge' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "antioch-openpi-franka"


def test_vulkan_preflight_rejects_missing_host_graphics_driver(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("VK_ICD_FILENAMES", str(tmp_path / "missing.json"))
    with pytest.raises(OpenPIBridgeError, match="NVIDIA Vulkan ICD is unavailable"):
        _verify_vulkan_runtime()


def test_vulkan_preflight_requires_an_nvidia_renderer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    icd = tmp_path / "nvidia.json"
    icd.write_text("{}")
    monkeypatch.setenv("VK_ICD_FILENAMES", str(icd))
    monkeypatch.setattr("shutil.which", lambda _command: "/usr/bin/vulkaninfo")
    monkeypatch.setattr(
        "subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["vulkaninfo"], returncode=0, stdout="GPU: llvmpipe", stderr=""
        ),
    )
    with pytest.raises(OpenPIBridgeError, match="did not find an NVIDIA renderer"):
        _verify_vulkan_runtime()


def test_vulkan_preflight_accepts_nvidia_renderer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    icd = tmp_path / "nvidia.json"
    icd.write_text("{}")
    monkeypatch.setenv("VK_ICD_FILENAMES", str(icd))
    monkeypatch.setattr("shutil.which", lambda _command: "/usr/bin/vulkaninfo")
    monkeypatch.setattr(
        "subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["vulkaninfo"], returncode=0, stdout="GPU: NVIDIA RTX", stderr=""
        ),
    )
    _verify_vulkan_runtime()


def test_franka_asset_root_uses_published_nvidia_compatibility_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assets = SimpleNamespace(
        NUCLEUS_ASSET_ROOT_DIR="https://assets.example/Assets/Isaac/6.0",
        ISAAC_NUCLEUS_DIR="https://assets.example/Assets/Isaac/6.0/Isaac",
        ISAACLAB_NUCLEUS_DIR=("https://assets.example/Assets/Isaac/6.0/Isaac/IsaacLab"),
        NVIDIA_NUCLEUS_DIR="https://assets.example/Assets/Isaac/6.0/NVIDIA",
    )

    class Response:
        status = 200

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    def urlopen(request: object, *, timeout: int) -> Response:
        assert timeout == 15
        if "/6.0/" in str(getattr(request, "full_url")):
            raise urllib.error.HTTPError("", 404, "missing", {}, None)
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    assert _ensure_franka_asset_root(assets) == "nvidia-5.1-compatibility"
    assert assets.NUCLEUS_ASSET_ROOT_DIR.endswith("/5.1")
    assert "/5.1/Isaac/IsaacLab" in assets.ISAACLAB_NUCLEUS_DIR


def test_franka_asset_root_fails_closed_when_compatibility_asset_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assets = SimpleNamespace(
        NUCLEUS_ASSET_ROOT_DIR="https://assets.example/Assets/Isaac/6.0"
    )

    def missing(*_args: object, **_kwargs: object) -> None:
        raise urllib.error.HTTPError("", 404, "missing", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", missing)
    with pytest.raises(OpenPIBridgeError, match="reviewed compatibility roots"):
        _ensure_franka_asset_root(assets)


def test_franka_compatibility_rewrites_an_already_imported_task_config() -> None:
    stale = "https://assets.example/Assets/Isaac/6.0/Isaac/Franka/panda.usd"
    assert _compatible_franka_asset_url(stale, "nvidia-5.1-compatibility") == (
        "https://assets.example/Assets/Isaac/5.1/Isaac/Franka/panda.usd"
    )
    assert _compatible_franka_asset_url(stale, "native") == stale


def test_camera_adapter_accepts_regular_and_tiled_rgb_frames() -> None:
    regular = _camera_frame(np.zeros((32, 48, 4), dtype=np.float32))
    tiled = _camera_frame(np.ones((1, 32, 48, 4), dtype=np.float32))
    assert regular.shape == (32, 48, 3)
    assert tiled.shape == (32, 48, 3)


def test_camera_adapter_rejects_malformed_frame_rank() -> None:
    with pytest.raises(OpenPIBridgeError, match="invalid RGB shape"):
        _camera_frame(np.zeros((32, 48), dtype=np.float32))


def test_hosted_example_pins_reviewed_npa_source_revision() -> None:
    manifest = yaml.safe_load((EXAMPLE_DIR / "antioch.yaml").read_text())
    source_ref = manifest["services"]["sim"]["build"]["args"]["NPA_SOURCE_REF"]

    assert re.fullmatch(r"[0-9a-f]{40}", source_ref)
    subprocess.run(
        ["git", "cat-file", "-e", f"{source_ref}^{{commit}}"],
        cwd=EXAMPLE_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    dockerfile = (EXAMPLE_DIR / "Dockerfile").read_text()
    assert "ARG NPA_SOURCE_REF" in dockerfile
    assert "@${NPA_SOURCE_REF}#subdirectory=npa" in dockerfile
    assert "COPY scenarios.py reverse_policy_relay.py /workspace/project/" in dockerfile
    assert dockerfile.rstrip().endswith("USER 1000:1000")
    dockerignore = (EXAMPLE_DIR / ".dockerignore").read_text().splitlines()
    assert ".antioch/" in dockerignore
    service = manifest["services"]["sim"]
    assert service["environment"] == {
        "OPENPI_POLICY_HOST": "127.0.0.1",
        "OPENPI_POLICY_PORT": "8000",
        "OPENPI_CONTROL_MODE": "continuous",
        "OPENPI_STREAM_DURATION_SECONDS": "30",
        "OPENPI_OBSERVATION_HZ": "5",
        "OPENPI_POLICY_REQUEST_HZ": "1",
        "OPENPI_CONTROL_HZ": "10",
        "OPENPI_EXECUTED_TARGETS_PER_CHUNK": "5",
        "OPENPI_MAXIMUM_OBSERVATION_AGE_SECONDS": "2",
        "OPENPI_MAXIMUM_RESPONSE_AGE_SECONDS": "10",
        "OPENPI_CAMERA_WARMUP_SECONDS": "10",
        "OPENPI_INFERENCE_DEADLINE_SECONDS": "15",
        "OPENPI_SAFE_HOLD_BEHAVIOR": "hold-current",
        "OPENPI_MINIMUM_READY_CYCLES": "3",
        "OPENPI_MINIMUM_READY_SECONDS": "10",
    }
    assert service["ports"] == ["18123:18123"]
    assert "secrets" not in service


def test_hosted_reverse_policy_relay_is_bidirectional() -> None:
    spec = importlib.util.spec_from_file_location(
        "npa_antioch_reverse_policy_relay",
        EXAMPLE_DIR / "reverse_policy_relay.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.ReversePolicyRelay.BACKEND_BIND_HOST == "0.0.0.0"
    assert module.ReversePolicyRelay.FRONTEND_BIND_HOST == "127.0.0.1"

    def unused_port() -> int:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            return int(probe.getsockname()[1])

    backend_port = unused_port()
    frontend_port = unused_port()
    with module.ReversePolicyRelay(
        backend_port=backend_port,
        frontend_port=frontend_port,
    ):
        backend = socket.create_connection(("127.0.0.1", backend_port), timeout=2)
        frontend = socket.create_connection(("127.0.0.1", frontend_port), timeout=2)
        frontend.sendall(b"request")
        assert backend.recv(7) == b"request"
        backend.sendall(b"response")
        assert frontend.recv(8) == b"response"
        frontend.close()
        backend.close()


def test_hosted_policy_relay_reports_transferred_bytes() -> None:
    spec = importlib.util.spec_from_file_location(
        "npa_antioch_reverse_policy_relay_bytes",
        EXAMPLE_DIR / "reverse_policy_relay.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    left, left_peer = socket.socketpair()
    right, right_peer = socket.socketpair()
    result: list[int] = []
    worker = threading.Thread(
        target=lambda: result.append(module.ReversePolicyRelay._pipe_pair(left, right))
    )
    worker.start()
    left_peer.sendall(b"request")
    assert right_peer.recv(7) == b"request"
    right_peer.sendall(b"response")
    assert left_peer.recv(8) == b"response"
    left_peer.close()
    right_peer.close()
    worker.join(2)
    assert result == [15]


def test_policy_tunnel_connector_retries_with_bounded_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    spec = importlib.util.spec_from_file_location(
        "npa_antioch_policy_tunnel_connector",
        EXAMPLE_DIR / "policy_tunnel_connector.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = 0
    delays: list[float] = []

    def unavailable(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise ConnectionRefusedError

    monkeypatch.setattr(module.socket, "create_connection", unavailable)
    monkeypatch.setattr(module.time, "sleep", delays.append)

    with pytest.raises(ConnectionError, match="exhausted connection attempts"):
        module._connect(
            "127.0.0.1",
            18123,
            attempts=3,
            initial_backoff_seconds=0.25,
            maximum_backoff_seconds=0.5,
        )
    assert calls == 3
    assert delays == [0.25, 0.5]


def _observation() -> dict[str, object]:
    return {
        "observation/exterior_image_1_left": np.zeros(IMAGE_SHAPE, dtype=np.uint8),
        "observation/wrist_image_left": np.ones(IMAGE_SHAPE, dtype=np.uint8),
        "observation/joint_position": np.asarray(
            [0, -0.5, 0, -1.5, 0, 1.0, 0], dtype=np.float32
        ),
        "observation/gripper_position": np.asarray([0.5], dtype=np.float32),
        "prompt": "pick up the fork",
    }


def _actions() -> np.ndarray:
    return np.tile(np.asarray([0, -0.5, 0, -1.5, 0, 1.0, 0, 0.5]), (15, 1))


def test_messagepack_round_trip_matches_openpi_numpy_contract() -> None:
    value = validate_observation(_observation())
    decoded = unpack_message(pack_message(value))
    assert isinstance(decoded, dict)
    np.testing.assert_array_equal(
        decoded["observation/wrist_image_left"],
        value["observation/wrist_image_left"],
    )
    assert decoded["observation/joint_position"].dtype == np.float32


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.pop("prompt"), "keys"),
        (
            lambda value: value.__setitem__(
                "observation/exterior_image_1_left",
                np.zeros((10, 10, 3), dtype=np.uint8),
            ),
            "uint8",
        ),
        (lambda value: value.__setitem__("prompt", ""), "prompt"),
    ],
)
def test_observation_validation_fails_closed(mutate, message: str) -> None:
    value = _observation()
    mutate(value)
    with pytest.raises(OpenPIBridgeError, match=message):
        validate_observation(value)


def test_response_requires_exact_finite_in_range_chunk() -> None:
    assert validate_actions({"actions": _actions()}).shape == ACTION_SHAPE
    malformed = _actions()
    malformed[2, 3] = np.nan
    with pytest.raises(OpenPIBridgeError, match="non-finite"):
        validate_actions({"actions": malformed})
    wrong = _actions()[:14]
    with pytest.raises(OpenPIBridgeError, match="shape"):
        validate_actions({"actions": wrong})
    unsafe = _actions()
    unsafe[0, 0] = 4.0
    with pytest.raises(OpenPIBridgeError, match="position limits"):
        validate_actions({"actions": unsafe})


def test_safe_targets_rate_limit_without_clipping_invalid_policy_output() -> None:
    actions = _actions()
    actions[:, 0] = 1.0
    result = safe_position_targets(
        actions,
        np.asarray([0, -0.5, 0, -1.5, 0, 1.0, 0]),
        max_joint_delta_rad=0.1,
        execute_steps=5,
    )
    np.testing.assert_allclose(result[:, 0], [0.1, 0.2, 0.3, 0.4, 0.5])


class _FakeConnection:
    def __init__(self, frames: list[bytes | str]) -> None:
        self.frames = frames
        self.sent: list[bytes] = []
        self.closed = False
        self.timeout = 0.0
        self.pings = 0

    def recv(self) -> bytes | str:
        return self.frames.pop(0)

    def send_binary(self, payload: bytes) -> None:
        self.sent.append(payload)

    def settimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def ping(self) -> None:
        self.pings += 1

    def close(self) -> None:
        self.closed = True


def test_client_reconnects_then_returns_exact_chunk() -> None:
    connection = _FakeConnection(
        [pack_message({"model": "pi0.5"}), pack_message({"actions": _actions()})]
    )
    calls = 0

    def factory(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError("not ready")
        return connection

    sleeps: list[float] = []
    client = OpenPIWebsocketClient(
        "policy.default.svc",
        retry=RetryPolicy(attempts=2, initial_backoff_seconds=0.25),
        connection_factory=factory,
        sleep=sleeps.append,
    )
    result = client.infer(_observation())
    assert result.shape == ACTION_SHAPE
    assert calls == 2
    assert sleeps == [0.25]
    assert len(connection.sent) == 1
    assert client.generation == 1
    assert client.reconnect_count == 0


def test_client_ping_reuses_connection_and_tracks_reconnect_generation() -> None:
    first = _FakeConnection([pack_message({"model": "pi0.5"})])
    second = _FakeConnection([pack_message({"model": "pi0.5"})])
    connections = iter([first, second])
    client = OpenPIWebsocketClient(
        "policy.default.svc",
        retry=RetryPolicy(attempts=1),
        connection_factory=lambda *_args, **_kwargs: next(connections),
    )
    client.ping()
    client.ping()
    assert first.pings == 2
    assert client.generation == 1
    client.close()
    client.ping()
    assert second.pings == 1
    assert client.generation == 2
    assert client.reconnect_count == 1


def test_client_exhaustion_is_no_action_and_hides_transport_detail() -> None:
    client = OpenPIWebsocketClient(
        "policy.default.svc",
        retry=RetryPolicy(attempts=2, initial_backoff_seconds=0),
        connection_factory=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ConnectionError("sensitive endpoint detail")
        ),
        sleep=lambda _seconds: None,
    )
    with pytest.raises(OpenPIBridgeError, match="failed after 2 attempts") as caught:
        client.infer(_observation())
    assert "sensitive endpoint" not in str(caught.value)


def _stack(**overrides: object) -> dict[str, object]:
    values = {
        "run_id": "test-run",
        "namespace": "default",
        "policy_image": "registry.example.invalid/openpi@sha256:" + "a" * 64,
        "bridge_image": "registry.example.invalid/isaac@sha256:" + "b" * 64,
        "policy_terms_secret": "openpi-terms",
        "isaac_acceptance_secret": "isaac-acceptance",
        "policy_gpu_selector_key": "example.invalid/gpu",
        "policy_gpu_selector_value": "B200",
        "bridge_gpu_selector_key": "example.invalid/gpu",
        "bridge_gpu_selector_value": "RTX-PRO-6000",
        "image_pull_secret": "registry-pull",
        "antioch_config_secret": "antioch-config",
        "s3_credentials_secret": "s3-runtime",
        "output_uri": "s3://example-bucket/run/report.json",
    }
    values.update(overrides)
    return render_stack(**values)


def test_stack_uses_separate_gpu_placement_and_private_policy_service() -> None:
    items = _stack()["items"]
    policy, service, bridge, network = items
    assert policy["spec"]["template"]["spec"]["nodeSelector"] == {
        "example.invalid/gpu": "B200"
    }
    assert bridge["spec"]["template"]["spec"]["nodeSelector"] == {
        "example.invalid/gpu": "RTX-PRO-6000"
    }
    for workload in (policy, bridge):
        security = workload["spec"]["template"]["spec"]["securityContext"]
        assert security["runAsNonRoot"] is True
        assert security["runAsUser"] == 1000
        assert security["runAsGroup"] == 1000
        assert security["fsGroup"] == 1000
        assert security["seccompProfile"] == {"type": "RuntimeDefault"}
    init = bridge["spec"]["template"]["spec"]["initContainers"][0]
    assert init["name"] == "wait-for-policy"
    assert init["args"][-1] == "1800"
    assert any(env["name"] == "NO_PROXY" for env in init["env"])
    graphics_init = bridge["spec"]["template"]["spec"]["initContainers"][1]
    assert graphics_init["name"] == "fetch-nvidia-graphics-runtime"
    assert graphics_init["command"] == [
        "/opt/npa/docker/workbench/common/nvidia_graphics_runtime.sh"
    ]
    assert graphics_init["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert graphics_init["env"][1]["valueFrom"]["secretKeyRef"] == {
        "name": "isaac-acceptance",
        "key": "ACCEPT_EULA",
    }
    bridge_env = bridge["spec"]["template"]["spec"]["containers"][0]["env"]
    assert any(env["name"] == "NO_PROXY" for env in bridge_env)
    assert service["spec"]["type"] == "ClusterIP"
    assert service["spec"]["ports"] == [
        {"name": "policy", "port": 8000, "targetPort": 8000}
    ]
    assert network["kind"] == "NetworkPolicy"
    assert network["spec"]["ingress"][0]["ports"][0]["port"] == 8000


def test_stack_mounts_runtime_fetched_graphics_readonly_in_bridge() -> None:
    bridge = _stack()["items"][2]
    pod = bridge["spec"]["template"]["spec"]
    container = pod["containers"][0]
    graphics_mount = next(
        mount
        for mount in container["volumeMounts"]
        if mount["name"] == "nvidia-graphics-runtime"
    )
    assert graphics_mount == {
        "name": "nvidia-graphics-runtime",
        "mountPath": "/opt/nvidia-graphics",
        "readOnly": True,
    }
    graphics_volume = next(
        volume
        for volume in pod["volumes"]
        if volume["name"] == "nvidia-graphics-runtime"
    )
    assert graphics_volume["emptyDir"]["sizeLimit"] == "4Gi"
    assert "source /opt/nvidia-graphics/runtime.env" in container["args"][0]
    serialized_server = json.dumps(_stack()["items"][0], sort_keys=True)
    assert "nvidia-graphics-runtime" not in serialized_server
    assert "isaac-acceptance" not in serialized_server


def test_stack_defaults_to_long_lived_continuous_control_with_readiness() -> None:
    bridge = _stack()["items"][2]
    assert bridge["kind"] == "Deployment"
    pod = bridge["spec"]["template"]["spec"]
    assert pod["restartPolicy"] == "Always"
    container = pod["containers"][0]
    env = {item["name"]: item.get("value") for item in container["env"]}
    assert env["OPENPI_CONTROL_MODE"] == "continuous"
    assert env["OPENPI_STREAM_DURATION_SECONDS"] == "0.0"
    assert env["OPENPI_OBSERVATION_HZ"] == "10.0"
    assert env["OPENPI_POLICY_REQUEST_HZ"] == "2.0"
    assert env["OPENPI_CONTROL_HZ"] == "10.0"
    assert env["OPENPI_EXECUTED_TARGETS_PER_CHUNK"] == "5"
    assert env["OPENPI_MAXIMUM_RESPONSE_AGE_SECONDS"] == "1.5"
    assert env["OPENPI_INFERENCE_DEADLINE_SECONDS"] == "10.0"
    assert env["OPENPI_SAFE_HOLD_BEHAVIOR"] == "hold-current"
    assert container["readinessProbe"]["exec"]["command"] == [
        "test",
        "-f",
        "/tmp/npa-openpi-stream-ready",
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {"control_mode": "finite-smoke"},
        {"control_mode": "continuous", "stream_duration_seconds": 15.0},
    ],
)
def test_finite_smoke_and_sustained_validation_are_explicit_jobs(
    overrides: dict[str, object],
) -> None:
    bridge = _stack(**overrides)["items"][2]
    assert bridge["kind"] == "Job"
    assert bridge["spec"]["template"]["spec"]["restartPolicy"] == "Never"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"control_mode": "single"}, "control mode"),
        ({"observation_hz": 0.0}, "positive and finite"),
        ({"executed_targets_per_chunk": 16}, "between 1 and 15"),
        ({"safe_hold_behavior": "random"}, "safe hold behavior"),
    ],
)
def test_stack_rejects_unsafe_streaming_configuration(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(OpenPIBridgeError, match=message):
        _stack(**overrides)


def test_stack_scopes_antioch_and_isaac_secrets_to_bridge_only() -> None:
    policy, _service, bridge, _network = _stack()["items"]
    serialized_policy = json.dumps(policy, sort_keys=True)
    serialized_bridge = json.dumps(bridge, sort_keys=True)
    assert "antioch-config" not in serialized_policy
    assert "isaac-acceptance" not in serialized_policy
    assert "s3-runtime" not in serialized_policy
    assert "openpi-terms" in serialized_policy
    assert "antioch-config" in serialized_bridge
    assert "isaac-acceptance" in serialized_bridge
    assert "s3-runtime" in serialized_bridge
    assert "NPA_OPENPI_ACCEPT_GEMMA_TERMS" not in serialized_bridge
    assert "token" not in serialized_bridge.lower()


def test_stack_warms_checkpoint_in_init_and_serves_from_readonly_cache() -> None:
    policy = _stack(policy_cache_pvc="openpi-model-cache")["items"][0]
    pod = policy["spec"]["template"]["spec"]
    warmer = pod["initContainers"][0]
    server = pod["containers"][0]
    assert warmer["name"] == "warm-openpi-checkpoint"
    assert warmer["command"][-3:] == [
        "warm",
        "--cache-root",
        "/opt/npa-model-cache/openpi",
    ]
    assert warmer["env"][0]["valueFrom"]["secretKeyRef"]["name"] == "openpi-terms"
    assert "NPA_OPENPI_ACCEPT_GEMMA_TERMS" not in json.dumps(server)
    assert {item["name"]: item.get("value") for item in server["env"]}[
        "OPENPI_DATA_HOME"
    ] == "/opt/npa-model-cache/openpi/openpi-data"
    assert server["volumeMounts"] == [
        {
            "name": "policy-cache",
            "mountPath": "/opt/npa-model-cache/openpi",
            "readOnly": True,
        }
    ]
    assert pod["volumes"] == [
        {
            "name": "policy-cache",
            "persistentVolumeClaim": {"claimName": "openpi-model-cache"},
        }
    ]


def test_stack_defaults_to_named_node_local_ephemeral_cache() -> None:
    policy = _stack()["items"][0]
    assert policy["spec"]["template"]["spec"]["volumes"] == [
        {"name": "policy-cache", "emptyDir": {"sizeLimit": "40Gi"}}
    ]


def test_stack_rejects_invalid_policy_cache_pvc() -> None:
    with pytest.raises(OpenPIBridgeError, match="Kubernetes DNS label"):
        _stack(policy_cache_pvc="Not/A/Claim")


def test_stack_rejects_mutable_images() -> None:
    with pytest.raises(OpenPIBridgeError, match="policy image must be digest-pinned"):
        _stack(policy_image="registry.example.invalid/openpi:latest")


def test_stack_rejects_invalid_readiness_timeout() -> None:
    with pytest.raises(OpenPIBridgeError, match="readiness timeout"):
        _stack(policy_ready_timeout_seconds=0)


def test_health_wait_rejects_non_private_style_host() -> None:
    with pytest.raises(ValueError, match="policy host"):
        wait_for_health("https://public.example.invalid")


def test_contract_smoke_is_real_serialization_and_rate_limit() -> None:
    assert contract_smoke() == {
        "schema": "npa.antioch.openpi-bridge.contract-smoke.v1",
        "status": "passed",
        "observation_keys": [
            "observation/exterior_image_1_left",
            "observation/gripper_position",
            "observation/joint_position",
            "observation/wrist_image_left",
            "prompt",
        ],
        "action_shape": [15, 8],
        "executed_target_shape": [5, 8],
        "fail_closed": True,
    }


def test_cli_renders_stack_without_secret_values() -> None:
    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "antioch",
            "openpi-stack",
            "--run-id",
            "cli-test",
            "--policy-image",
            "registry.example.invalid/openpi@sha256:" + "a" * 64,
            "--bridge-image",
            "registry.example.invalid/isaac@sha256:" + "b" * 64,
            "--policy-terms-secret",
            "terms",
            "--isaac-acceptance-secret",
            "isaac",
            "--observation-hz",
            "8",
            "--policy-request-hz",
            "1.5",
            "--control-hz",
            "20",
            "--maximum-response-age-seconds",
            "2",
            "--safe-hold-behavior",
            "no-action",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "rendered"
    assert payload["manifest"]["kind"] == "List"
    bridge_env = payload["manifest"]["items"][2]["spec"]["template"]["spec"][
        "containers"
    ][0]["env"]
    env = {item["name"]: item.get("value") for item in bridge_env}
    assert env["OPENPI_OBSERVATION_HZ"] == "8.0"
    assert env["OPENPI_POLICY_REQUEST_HZ"] == "1.5"
    assert env["OPENPI_CONTROL_HZ"] == "20.0"
    assert env["OPENPI_MAXIMUM_RESPONSE_AGE_SECONDS"] == "2.0"
    assert env["OPENPI_SAFE_HOLD_BEHAVIOR"] == "no-action"


def test_cli_contract_smoke() -> None:
    result = CliRunner().invoke(
        app, ["workbench", "antioch", "openpi-contract-smoke", "--output", "json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["action_shape"] == [15, 8]
