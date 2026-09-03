from __future__ import annotations

import importlib.util
import hashlib
import json
import re
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from npa.workbench.antioch import live
from npa.workbench.antioch import relay as live_relay
from npa.workbench.antioch import live_reconcile
from npa.workbench.antioch.vendor_cli import AntiochCliError
from npa.workflows.byof.openpi_live import _certificate

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "npa/examples/antioch-openpi-live"


def _load_live_scenario(monkeypatch: pytest.MonkeyPatch, module_name: str):
    class Logger:
        def __init__(self, root: str) -> None:
            self.root = root

    fake_antioch = types.ModuleType("antioch")
    fake_antioch.Logger = Logger
    fake_antioch.param = lambda default, **_kwargs: default
    fake_antioch.scenario = lambda **_kwargs: lambda function: function
    monkeypatch.setitem(sys.modules, "antioch", fake_antioch)
    path = EXAMPLE / "src/scenario_v2.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    scenario = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = scenario
    spec.loader.exec_module(scenario)
    return scenario


def _daemon_status(
    *, state: str = "idle", scenario_run_id: str = "", session: bool = True
) -> dict[str, object]:
    observed = int(time.time() * 1_000_000)
    stream = {"state": state, "scenario_run_id": scenario_run_id}
    leases: list[dict[str, str]] = []
    if scenario_run_id:
        leases = [
            {"kind": "process", "label": "scenario"},
            {"kind": "stream", "label": "stream"},
        ]
        if session:
            leases.append(
                {"kind": "session", "label": "antioch scenario run"}
            )
    observation = {"observed_at": observed, "leases": leases, "stream": stream}
    return {
        "daemon_error": None,
        "runtime": observation,
        "runtime_status": {
            "guest_state": "healthy",
            "guest_observed_at": observed,
            "guest_failure_started_at": None,
            "observation": observation,
        },
        # Deliberately include the cached compatibility field. Production
        # liveness must ignore it and use the two structured observations.
        "stream": stream,
    }


def test_live_example_uses_only_runtime_project_identity() -> None:
    manifest = yaml.safe_load((EXAMPLE / "antioch.yaml").read_text(encoding="utf-8"))
    assert manifest["id"] == "replace-at-runtime"
    sim = manifest["services"]["sim"]
    assert "image" not in sim
    assert sim["build"] == {"context": ".", "dockerfile": "Dockerfile"}
    assert sim["ports"] == [
        {"name": "policy-relay", "target": 8444, "published": 18444}
    ]
    assert sim["watch"] == [
        {"action": "rebuild", "path": "Dockerfile"},
        {"action": "rebuild", "path": "src/relay_bridge.py"},
    ]
    rendered = (EXAMPLE / "antioch.yaml").read_text(encoding="utf-8")
    assert not re.search(r"(?:project|tenant|cluster)-[a-z0-9]+", rendered)


def test_live_scenario_is_real_bounded_and_fail_closed() -> None:
    source = (EXAMPLE / "src/scenario_v2.py").read_text(encoding="utf-8")
    compile(source, "antioch-openpi-live-scenario", "exec")
    for contract in (
        "ACTION_SHAPE = (15, 8)",
        "MAX_RESPONSE_AGE_SECONDS",
        "MAX_JOINT_STEP",
        "GRIPPER_JOINT_MAX = 0.04",
        "DROID_RESET_JOINTS",
        '"pick up the red cube"',
        'TASK_LABEL = "red_cube_pickup"',
        "def openpi_franka_mk8s_live_v2(",
        "raw_gripper_range_mismatches",
        "raw_joint_limit_mismatches",
        "joint_limit_projections",
        "joint_step_projections",
        "_droid_gripper_observation",
        "_isaac_finger_target",
        "isinstance(exc, ActionValidationError)",
        "next_attempt = now + 1.0 / CONTROL_HZ",
        "ssl.create_default_context",
        'CLIENT_ROOT = Path("/tmp/npa-live-client-current")',
        'return "wss://127.0.0.1:8444", token, context',
        '"X-NPA-Relay-Role": "simulation"',
        'TELEMETRY_ROOT = "openpi-live"',
        "logger = antioch.Logger(TELEMETRY_ROOT)",
        "def _resolved_telemetry_entity(",
        "def _log_camera_pair_for_rerun(",
        "logger.image(entity, frame.rgb)",
        "LiveTelemetryPublisher(logger)",
        "LatestOnlyWorker(",
        "TELEMETRY_DISPLAY_HZ = 5.0",
        '"observation_sequence", observation_sequence',
        '"policy_requests", requests',
        '"policy_in_flight", int(pending is not None)',
        '"round_trips", round_trips',
        '"inference_latency_ms", last_latency * 1000.0',
        '"safe_hold", int(safe_hold)',
        '"reconnects", client.reconnects',
        '"safe_targets_applied", applied',
        '"raw_gripper_range_mismatches",',
        '"raw_joint_limit_mismatches",',
        '"joint_limit_projections", joint_limit_projections',
        '"joint_step_projections", joint_step_projections',
        '"end_effector_cube_distance_m", ee_distance',
        '"gripper_contact_force_n", contact_force',
        '"cube_lift_m", cube_lift',
        '"pickup_success", int(pickup_success)',
        'f"{FRANKA_SCENE_ENTITY}/base"',
        'f"{FRANKA_SCENE_ENTITY}/links"',
        'f"{FRANKA_SCENE_ENTITY}/joints"',
        'f"{FRANKA_SCENE_ENTITY}/gripper"',
        "rr.Boxes3D(**proxy",
        "rr.Ellipsoids3D(**proxy",
        "_franka_proxy_geometry(link_points)",
        "ArticulationAction",
        "track_contact_forces=True",
        "contact_filter_prim_paths_expr",
        'WRIST_CAMERA_PATH = "/World/PolicyWrist"',
        "_wrist_camera_pose_from_points",
        '_configure_camera_optics(world.stage, WRIST_CAMERA_PATH, "wrist")',
        "rr.send_blueprint(_camera_blueprint(rrb))",
        "_configure_lighting(world.stage)",
        'enable_extension("isaacsim.robot.manipulators.examples")',
        "NPA_OPENPI_ROUND_TRIP",
        "NPA_OPENPI_SAFE_HOLD",
        "NPA_OPENPI_LOOP_READY",
        "NPA_OPENPI_FIRST_FRAME",
        "NPA_OPENPI_REQUEST",
        "NPA_OPENPI_APPLIED",
        "NPA_OPENPI_CAMERA_STATUS",
        "NPA_OPENPI_IMAGE_LOG_BEGIN",
        "NPA_OPENPI_IMAGE_LOG_OK",
        "NPA_OPENPI_DISPLAY_LOG_BEGIN",
        "NPA_OPENPI_DISPLAY_LOG_OK",
        "NPA_OPENPI_LOOP_HEARTBEAT",
        "NPA_OPENPI_POLICY_STALL",
        'ThreadPoolExecutor(max_workers=1, thread_name_prefix="openpi-policy")',
    ):
        assert contract in source
    assert "WebsocketClientPolicy(" not in source
    assert "verify_mode = ssl.CERT_NONE" not in source
    assert "rr.LineStrips3D" not in source
    assert "while True:" in source

    relay = (ROOT / "npa/src/npa/workbench/antioch/relay.py").read_text(
        encoding="utf-8"
    )
    assert 'additional_headers={"Authorization": f"Api-Key {policy_token}"}' in relay
    assert '"X-NPA-Relay-Role": "operator"' in relay
    assert "proxy=None" in relay
    assert "port != 443" in relay
    assert "ssl.create_default_context" in relay
    assert "CERT_NONE" not in relay

    bridge = (EXAMPLE / "src/relay_bridge.py").read_text(encoding="utf-8")
    compile(bridge, "antioch-openpi-live-relay-bridge", "exec")
    assert "ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)" in bridge
    assert '"0.0.0.0",\n        8444,' in bridge
    assert "hmac.compare_digest" in bridge
    assert 'ROLES = frozenset({"operator", "simulation"})' in bridge


def test_live_franka_proxy_is_volumetric_oriented_and_asset_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _load_live_scenario(monkeypatch, "antioch_live_scenario_test")

    geometry = scenario._franka_proxy_geometry(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.3],
            [0.0, 0.0, 0.3],  # zero-length USD offset is ignored
            [0.2, 0.0, 0.5],
            [0.35, 0.1, 0.65],
        ]
    )

    assert geometry["base"]["sizes"] == [[0.20, 0.20, 0.11]]
    assert len(geometry["links"]["centers"]) == 3
    assert geometry["links"]["sizes"][0] == pytest.approx([0.105, 0.105, 0.3])
    assert geometry["links"]["quaternions"][0] == pytest.approx(
        [0.0, 0.0, 0.0, 1.0]
    )
    assert all(
        sum(component * component for component in quaternion)
        == pytest.approx(1.0)
        for quaternion in geometry["links"]["quaternions"]
    )
    assert geometry["joints"]["centers"][-1] == [0.35, 0.1, 0.65]
    assert len(geometry["gripper"]["centers"]) == 3
    assert geometry["gripper"]["sizes"][1:] == [
        [0.024, 0.024, 0.15],
        [0.024, 0.024, 0.15],
    ]
    assert all(color[-1] == 255 for color in geometry["links"]["colors"])
    rr = pytest.importorskip("rerun")
    rr.Boxes3D(**geometry["base"])
    rr.Boxes3D(**geometry["links"])
    rr.Ellipsoids3D(**geometry["joints"])
    rr.Boxes3D(**geometry["gripper"])
    assert "Mesh3D" not in (EXAMPLE / "src/scenario_v2.py").read_text(encoding="utf-8")


def test_live_camera_rejects_black_or_flat_annotator_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _load_live_scenario(monkeypatch, "antioch_live_camera_test")

    class Camera:
        def __init__(self, frame: np.ndarray | None) -> None:
            self.frame = frame

        def get_data(self, annotator: str) -> tuple[np.ndarray | None, dict]:
            assert annotator == "rgb"
            return self.frame, {"renderingFrame": 1}

    assert scenario._camera_frame(Camera(None), view="wrist").reason == "missing"
    assert (
        scenario._camera_frame(
            Camera(np.zeros((8, 8, 4), dtype=np.uint8)), view="wrist"
        ).reason
        == "wrong_shape"
    )
    assert scenario._camera_frame(
        Camera(np.full((224, 224, 4), 64, dtype=np.uint8)), view="wrist"
    ).reason == "flat"
    black = scenario._camera_frame(
        Camera(np.zeros((224, 224, 4), dtype=np.uint8)), view="wrist"
    )
    assert black.reason == "blank"
    assert black.raw_min == 0.0
    assert black.raw_max == 0.0
    assert black.raw_nonzero == 0
    assert black.raw_channels == 4
    flat = scenario._camera_frame(
        Camera(np.full((224, 224, 4), 64, dtype=np.uint8)), view="wrist"
    )
    assert flat.rgb is not None
    rendered = np.zeros((224, 224, 4), dtype=np.uint8)
    rendered[:, 112:, :3] = 255
    rendered[:, :, 3] = 255
    useful = scenario._camera_frame(Camera(rendered), view="wrist")
    assert useful.reason == ""
    assert useful.raw_max == 255.0
    assert useful.raw_nonzero > 0
    assert useful.raw_channels == 4
    np.testing.assert_array_equal(useful.rgb, rendered[:, :, :3])
    normalized = rendered.astype(np.float32) / 255.0
    np.testing.assert_array_equal(
        scenario._camera_frame(Camera(normalized), view="wrist").rgb,
        rendered[:, :, :3],
    )

    low_dynamic = np.full((224, 224, 4), 90, dtype=np.uint8)
    low_dynamic[:, 112:, :3] = 110
    low_dynamic[100:106, 100:106, :3] = [240, 5, 4]
    low_dynamic[..., 3] = 255
    classified = scenario._camera_frame(Camera(low_dynamic), view="exterior")
    assert classified.dynamic_range < 32.0
    assert classified.luminance_variance > scenario.MIN_CAMERA_LUMINANCE_VARIANCE
    assert classified.red_cube_pixels >= scenario.MIN_EXTERIOR_RED_CUBE_PIXELS
    assert classified.reason == ""


def test_live_rejected_camera_pixels_are_logged_and_metrics_are_not_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _load_live_scenario(monkeypatch, "antioch_live_rejected_visual_test")

    class Camera:
        def __init__(self, frame: np.ndarray) -> None:
            self.frame = frame

        def get_data(self, annotator: str) -> tuple[np.ndarray, dict]:
            assert annotator == "rgb"
            return self.frame, {"renderingFrame": 1}

    exterior = np.full((224, 224, 4), 64, dtype=np.uint8)
    exterior[..., 3] = 255
    rows, columns = np.indices((224, 224))
    wrist = np.zeros((224, 224, 4), dtype=np.uint8)
    wrist[..., :3] = ((rows * 2 + columns * 3) % 255)[..., None]
    wrist[..., 3] = 255
    pair = scenario._validate_camera_pair(
        Camera(exterior),
        Camera(wrist),
        render_sequence=17,
        last_accepted_render_sequence=0,
        exterior_cube_in_frame=True,
        wrist_cube_in_frame=True,
    )
    assert pair.accepted is False
    assert pair.reason == "flat"
    assert pair.exterior.rgb is not None
    assert pair.wrist.rgb is not None

    class Logger:
        images: list[tuple[str, np.ndarray]] = []
        scalars: list[tuple[str, float]] = []

        def image(self, path: str, value: np.ndarray) -> None:
            self.images.append((path, value))

        def scalar(self, path: str, value: float) -> None:
            self.scalars.append((path, value))

    logger = Logger()
    assert scenario._log_camera_pair_for_rerun(logger, pair) == (
        "exterior",
        "wrist",
    )
    assert [path for path, _value in logger.images] == [
        "camera/exterior",
        "camera/wrist",
    ]
    metrics = scenario._camera_rejection_metrics_line(
        elapsed_seconds=2.5,
        frames=0,
        requests=0,
        round_trips=0,
        applied=0,
        reconnects=1,
        camera_rejected_pairs=3,
        camera_validated_requests=0,
        camera_pair_id=0,
        request_camera_pair_id=0,
        round_trip_camera_pair_id=0,
        render_sequence=17,
        request_render_sequence=0,
        round_trip_render_sequence=0,
        exterior_cube_in_frame=True,
        wrist_cube_in_frame=True,
        pair=pair,
    )
    assert metrics.startswith("NPA_OPENPI_METRICS ")
    assert "frames=0 requests=0 round_trips=0" in metrics
    assert "camera_rejected_pairs=3" in metrics
    assert "camera_render_sequence=17" in metrics
    assert "camera_exterior_dynamic_range_current=" in metrics
    assert "camera_exterior_raw_channels_current=4" in metrics
    assert "camera_wrist_raw_nonzero_current=" in metrics


def test_live_telemetry_is_latest_only_and_control_does_not_wait_for_slow_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _load_live_scenario(monkeypatch, "antioch_live_backpressure_test")
    image_started = threading.Event()
    release_images = threading.Event()
    lock = threading.Lock()

    class Logger:
        def __init__(self) -> None:
            self.images: list[tuple[str, int]] = []
            self.values: list[tuple[str, object]] = []

        def image(self, path: str, value: np.ndarray) -> None:
            image_started.set()
            assert release_images.wait(timeout=3.0)
            with lock:
                self.images.append((path, int(value[0, 0, 0])))

        def scalar(self, path: str, value: float) -> None:
            with lock:
                self.values.append((path, value))

        def value(self, path: str, value: object) -> None:
            with lock:
                self.values.append((path, value))

    logger = Logger()
    publisher = scenario.LiveTelemetryPublisher(logger)

    def pair(sequence: int):  # noqa: ANN202
        rgb = np.full((224, 224, 3), sequence, dtype=np.uint8)
        frame = scenario.CameraFrame(rgb, "", 50.0, 100.0, 80.0, 25)
        return scenario.CameraPair(True, frame, frame, mean_difference=12.0)

    assert publisher.publish_camera_pair(pair(1), 1) == ("exterior", "wrist")
    assert image_started.wait(timeout=1.0)
    started = time.monotonic()
    for sequence in range(2, 64):
        publisher.publish_camera_pair(pair(sequence), sequence)
        publisher.publish_display(
            scenario.DisplayPublication(
                render_sequence=sequence,
                numeric_groups=(("decision", (("render_sequence", sequence),)),),
                values=(),
            )
        )
    assert time.monotonic() - started < 0.5
    snapshot = publisher.snapshot()
    assert snapshot["exterior"] is snapshot["wrist"] is snapshot["display"]
    assert snapshot["logger"]["max_pending"] <= 2
    assert snapshot["exterior"]["dropped"] > 0
    assert snapshot["wrist"]["dropped"] > 0
    assert snapshot["display"]["max_pending"] <= 2
    with lock:
        assert not any(path == "decision" for path, _ in logger.values)

    release_images.set()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        with lock:
            latest = dict(logger.images)
        display_latest = any(
            path == "decision"
            and isinstance(value, dict)
            and value.get("render_sequence") == 63
            for path, value in logger.values
        )
        if (
            latest.get("camera/exterior") == 63
            and latest.get("camera/wrist") == 63
            and display_latest
        ):
            break
        time.sleep(0.01)
    else:
        pytest.fail("camera entities did not advance to the newest queued frame")

    assert publisher.close() == {"exterior": True, "wrist": True, "display": True}
    assert all(
        not state["alive"] for state in publisher.snapshot().values()
    )


def test_rtx_camera_construction_wires_public_rgb_render_product(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _load_live_scenario(monkeypatch, "antioch_rtx_construction_test")
    calls: list[tuple] = []

    class Prim:
        def IsValid(self) -> bool:
            return True

        def GetPath(self) -> str:
            return "/Render/exterior"

        def __bool__(self) -> bool:
            return True

    class Product:
        def GetPrim(self) -> Prim:
            return Prim()

    class Authoring:
        def __init__(self, path: str, **kwargs) -> None:
            calls.append(("authoring", path, kwargs))

        def __getattr__(self, name: str):
            calls.append(("unsupported_authoring_api", name))
            raise AttributeError(name)

    class Sensor:
        def __init__(self, authoring, **kwargs) -> None:
            calls.append(("sensor", authoring, kwargs))
            self.render_product = Product()
            self.detached = False

        def detach_annotators(self, annotator: str) -> None:
            assert annotator == "rgb"
            self.detached = True

    class Clock:
        detached = False

        def get_data(self) -> dict[str, int]:
            return {"referenceTimeNumerator": 1, "referenceTimeDenominator": 60}

        def detach(self, products: list[str]) -> None:
            assert products == ["/Render/exterior"]
            self.detached = True

    clock = Clock()
    monkeypatch.setattr(scenario, "_new_reference_time_annotator", lambda _path: clock)
    camera = scenario._build_rtx_rgb_camera(
        Authoring,
        Sensor,
        path=scenario.EXTERIOR_CAMERA_PATH,
        position=(1.0, 2.0, 3.0),
        output_buffer=object(),
    )

    assert calls[0] == (
        "authoring",
        scenario.EXTERIOR_CAMERA_PATH,
        {
            "tick_rate": scenario.CAMERA_SENSOR_TICK_RATE_HZ,
            "positions": [(1.0, 2.0, 3.0)],
        },
    )
    assert calls[1][2] == {"resolution": (224, 224), "annotators": ["rgb"]}
    assert camera.render_product_path == "/Render/exterior"
    camera.close()
    assert camera.sensor.detached and clock.detached
    assert not [call for call in calls if call[0] == "unsupported_authoring_api"]


def test_rtx_camera_uses_documented_cpu_rgb_output_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _load_live_scenario(monkeypatch, "antioch_rtx_native_buffer_test")

    class Prim:
        def IsValid(self) -> bool:
            return True

        def GetPath(self) -> str:
            return "/Render/exterior"

    class Sensor:
        def __init__(self, _authoring, **_kwargs) -> None:
            self.render_product = SimpleNamespace(GetPrim=lambda: Prim())

    monkeypatch.setattr(
        scenario,
        "_new_reference_time_annotator",
        lambda _path: SimpleNamespace(),
    )
    camera = scenario._build_rtx_rgb_camera(
        lambda _path, **_kwargs: object(),
        Sensor,
        path=scenario.EXTERIOR_CAMERA_PATH,
        output_buffer=(224, 224, 3),
    )
    assert camera.output_buffer == (224, 224, 3)


def test_camera_capture_uses_completed_world_render_without_double_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _load_live_scenario(monkeypatch, "antioch_render_scheduler_test")
    producer_sequence = 1
    frame = scenario.CameraFrame(np.ones((224, 224, 3), dtype=np.uint8), "")

    class Camera:
        def sample(self, *, view: str):
            assert view in {"exterior", "wrist"}
            return scenario.CameraSample(frame, (producer_sequence, 60))

    cameras = (("exterior", Camera()), ("wrist", Camera()))
    stale = {view: camera.sample(view=view) for view, camera in cameras}
    advanced, _, reason = scenario._camera_markers_advanced(
        stale, {"exterior": (1, 60), "wrist": (1, 60)}
    )
    assert not advanced and reason == "exterior_producer_stale"

    producer_sequence += 1  # world.step(render=True)
    first = scenario._capture_camera_samples(cameras)
    producer_sequence += 1  # world.step(render=True)
    second = scenario._capture_camera_samples(cameras)
    assert first["exterior"].producer_marker == (2, 60)
    assert second["exterior"].producer_marker == (3, 60)


def test_scheduler_tick_does_not_fake_camera_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _load_live_scenario(monkeypatch, "antioch_scheduler_safe_hold_test")
    ticks = 0

    class MissingCamera:
        def sample(self, *, view: str):
            return scenario.CameraSample(
                scenario.CameraFrame(None, "missing"), (ticks, 60)
            )

    cameras = (("exterior", MissingCamera()), ("wrist", MissingCamera()))
    monitor = scenario.CameraReadinessMonitor()
    for render_sequence in (1, 2, 3):
        ticks += 1  # world.step(render=True)
        samples = scenario._capture_camera_samples(cameras)
        pair = scenario._validate_camera_pair(
            samples["exterior"].frame,
            samples["wrist"].frame,
            render_sequence=render_sequence,
            last_accepted_render_sequence=0,
            exterior_cube_in_frame=True,
            wrist_cube_in_frame=True,
        )
        decision = monitor.observe(pair)
        assert not pair.accepted
        assert not decision.policy_eligible
        assert decision.status == "waiting_for_camera"
    assert ticks == 3


def test_camera_timeline_is_committed_before_rendered_readback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _load_live_scenario(monkeypatch, "antioch_timeline_start_test")
    calls: list[dict[str, object]] = []
    scenario._start_camera_timeline(
        SimpleNamespace(play=lambda **kwargs: calls.append(kwargs))
    )
    assert calls == [{"commit": True}]


def test_rtx_camera_samples_delayed_numpy_and_warp_without_aliasing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _load_live_scenario(monkeypatch, "antioch_rtx_sampling_test")
    source = np.zeros((224, 224, 3), dtype=np.uint8)
    source[:, 112:] = 255

    class Warp:
        def numpy(self) -> np.ndarray:
            return source

    class Sensor:
        render_product = SimpleNamespace(
            GetPrim=lambda: SimpleNamespace(
                IsValid=lambda: True, GetPath=lambda: "/Render/wrist"
            )
        )
        samples = [
            (None, {}),
            (Warp(), {"referenceTimeNumerator": 2, "referenceTimeDenominator": 60}),
        ]

        def get_data(self, annotator: str, *, out):
            assert annotator == "rgb"
            assert out == "cpu-rgb"
            return self.samples.pop(0)

    clock = SimpleNamespace(
        get_data=lambda: {},
        detach=lambda _products: None,
    )
    camera = scenario.RtxRgbCamera(
        SimpleNamespace(destroy=lambda: None), Sensor(), clock, "cpu-rgb"
    )
    assert camera.sample(view="wrist").frame.reason == "missing"
    sample = camera.sample(view="wrist")
    assert sample.producer_marker == (2, 60)
    assert sample.frame.rgb is not None
    source[:] = 0
    assert int(sample.frame.rgb.max()) == 255


def test_live_capture_fails_closed_and_tears_down_invalid_rtx_product(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _load_live_scenario(monkeypatch, "antioch_rtx_cleanup_test")

    class Logger:
        def scalar(self, _path: str, _value: float) -> None:
            pass

    class Camera:
        render_product_path = ""
        closed = False

        def close(self) -> None:
            self.closed = True

    exterior, wrist = Camera(), Camera()
    world = SimpleNamespace(stage=SimpleNamespace(GetPrimAtPath=lambda _path: None))
    with pytest.raises(scenario.CameraReadinessError, match="render_product_invalid"):
        scenario._initialize_live_capture(
            world, (("exterior", exterior), ("wrist", wrist)), Logger()
        )
    assert exterior.closed and wrist.closed
def test_camera_readiness_waits_hundreds_of_frames_before_policy_eligibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _load_live_scenario(monkeypatch, "antioch_camera_none_test")
    missing = scenario.CameraFrame(None, "missing")
    missing_pair = scenario.CameraPair(
        False, missing, missing, "exterior", "missing"
    )
    monitor = scenario.CameraReadinessMonitor(status_interval_frames=100)
    decisions = [monitor.observe(missing_pair) for _ in range(400)]

    assert all(not item.policy_eligible for item in decisions)
    assert all(item.status == "waiting_for_camera" for item in decisions)
    assert [index for index, item in enumerate(decisions, 1) if item.emit_status] == [
        1,
        101,
        201,
        301,
    ]


def test_delayed_valid_pair_requires_advancement_then_enables_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _load_live_scenario(monkeypatch, "antioch_camera_delayed_test")
    missing = scenario.CameraFrame(None, "missing")
    missing_pair = scenario.CameraPair(
        False, missing, missing, "exterior", "missing"
    )
    exterior = scenario.CameraFrame(object(), "", 50.0, 100.0, 80.0, 25)
    wrist = scenario.CameraFrame(object(), "", 55.0, 110.0, 85.0, 0)
    valid_pair = scenario.CameraPair(True, exterior, wrist, mean_difference=12.0)
    monitor = scenario.CameraReadinessMonitor(status_interval_frames=50)
    policy_requests = 0

    for _ in range(350):
        policy_requests += int(monitor.observe(missing_pair).policy_eligible)
    first = monitor.observe(valid_pair)
    policy_requests += int(first.policy_eligible)
    second = monitor.observe(valid_pair)
    policy_requests += int(second.policy_eligible)

    assert not first.policy_eligible
    assert first.reason == "confirming_advancement"
    assert second.policy_eligible and second.status == "ready"
    assert policy_requests == 1

    outage = monitor.observe(missing_pair)
    recovering = monitor.observe(valid_pair)
    recovered = monitor.observe(valid_pair)
    assert outage.status == "runtime_outage" and not outage.policy_eligible
    assert recovering.status == "runtime_outage" and not recovering.policy_eligible
    assert recovered.policy_eligible


def test_camera_frame_advancement_uses_producer_marker_not_loop_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _load_live_scenario(monkeypatch, "antioch_camera_advancement_test")

    frame = scenario.CameraFrame(object(), "")
    samples = {
        "exterior": scenario.CameraSample(frame, (9, 60)),
        "wrist": scenario.CameraSample(frame, (9, 60)),
    }
    advanced, _, reason = scenario._camera_markers_advanced(
        samples, {"exterior": (9, 60), "wrist": (9, 60)}
    )
    assert not advanced and reason == "exterior_producer_stale"
    samples = {
        "exterior": scenario.CameraSample(frame, (10, 60)),
        "wrist": scenario.CameraSample(frame, (10, 60)),
    }
    advanced, current, reason = scenario._camera_markers_advanced(
        samples, {"exterior": (9, 60), "wrist": (9, 60)}
    )
    assert advanced and not reason and current["wrist"] == (10, 60)


def test_logger_first_use_is_on_owner_then_all_channels_share_one_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _load_live_scenario(monkeypatch, "antioch_logger_owner_test")
    owner = threading.get_ident()
    calls: list[tuple[str, int, object]] = []
    complete = threading.Event()

    class Logger:
        def scalar(self, path: str, value: float) -> None:
            calls.append((path, threading.get_ident(), value))

        def image(self, path: str, value: np.ndarray) -> None:
            calls.append((path, threading.get_ident(), int(value[0, 0, 0])))

        def value(self, path: str, value: object) -> None:
            calls.append((path, threading.get_ident(), value))
            complete.set()

    publisher = scenario.LiveTelemetryPublisher(Logger())
    assert calls == [("telemetry/initialized", owner, 1.0)]
    rgb = np.full((224, 224, 3), 42, dtype=np.uint8)
    frame = scenario.CameraFrame(rgb, "", 50.0, 100.0, 80.0, 25)
    publisher.publish_camera_pair(
        scenario.CameraPair(True, frame, frame, mean_difference=12.0), 42
    )
    publisher.publish_display(
        scenario.DisplayPublication(
            render_sequence=42,
            numeric_groups=(("decision", (("render_sequence", 42),)),),
            values=(),
        )
    )
    assert complete.wait(timeout=2.0)
    assert publisher.close() == {"exterior": True, "wrist": True, "display": True}
    worker_threads = {thread_id for _, thread_id, _ in calls[1:]}
    assert len(worker_threads) == 1 and owner not in worker_threads
    decision = next(value for path, _, value in calls if path == "decision")
    assert decision == {"render_sequence": 42}


def test_blocked_logger_teardown_is_detected_before_result_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _load_live_scenario(monkeypatch, "antioch_logger_teardown_test")
    started = threading.Event()
    release = threading.Event()

    class Logger:
        def scalar(self, _path: str, _value: float) -> None:
            pass

        def image(self, _path: str, _value: np.ndarray) -> None:
            started.set()
            assert release.wait(timeout=3.0)

        def value(self, _path: str, _value: object) -> None:
            pass

    publisher = scenario.LiveTelemetryPublisher(Logger())
    rgb = np.full((224, 224, 3), 7, dtype=np.uint8)
    frame = scenario.CameraFrame(rgb, "", 50.0, 100.0, 80.0, 25)
    publisher.publish_camera_pair(
        scenario.CameraPair(True, frame, frame, mean_difference=12.0), 7
    )
    assert started.wait(timeout=1.0)
    assert publisher.close() == {"exterior": False, "wrist": False, "display": False}
    assert publisher.snapshot()["logger"]["alive"]
    release.set()
    assert publisher.close() == {"exterior": True, "wrist": True, "display": True}


def test_live_display_rate_is_bounded_under_a_faster_physics_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _load_live_scenario(monkeypatch, "antioch_live_display_rate_test")
    limiter = scenario.DisplayRateLimiter()
    admitted = [step / 100.0 for step in range(1000) if limiter.due(step / 100.0)]

    assert 49 <= len(admitted) <= 50
    assert all(
        right - left >= (1.0 / scenario.TELEMETRY_DISPLAY_HZ) - 1e-9
        for left, right in zip(admitted, admitted[1:])
    )


def test_live_display_backpressure_is_bounded_and_does_not_block_cameras(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _load_live_scenario(monkeypatch, "antioch_live_display_stall_test")
    display_started = threading.Event()
    release_display = threading.Event()
    lock = threading.Lock()

    class Logger:
        def __init__(self) -> None:
            self.images: list[str] = []
            self.display_sequences: list[int] = []

        def image(self, path: str, _value: np.ndarray) -> None:
            with lock:
                self.images.append(path)

        def scalar(self, _path: str, _value: float) -> None:
            pass

        def value(self, path: str, value: object) -> None:
            if path != "decision" or not isinstance(value, dict):
                return
            display_started.set()
            assert release_display.wait(timeout=3.0)
            with lock:
                self.display_sequences.append(int(value["render_sequence"]))

    def publication(sequence: int):  # noqa: ANN202
        return scenario.DisplayPublication(
            render_sequence=sequence,
            numeric_groups=(("decision", (("render_sequence", sequence),)),),
            values=(),
        )

    rgb = np.full((224, 224, 3), 17, dtype=np.uint8)
    frame = scenario.CameraFrame(rgb, "", 50.0, 100.0, 80.0, 25)
    pair = scenario.CameraPair(True, frame, frame, mean_difference=12.0)
    logger = Logger()
    publisher = scenario.LiveTelemetryPublisher(logger)
    publisher.publish_display(publication(1))
    assert display_started.wait(timeout=1.0)

    started = time.monotonic()
    for sequence in range(2, 64):
        publisher.publish_display(publication(sequence))
    assert publisher.publish_camera_pair(pair, 64) == ("exterior", "wrist")
    assert time.monotonic() - started < 0.5
    snapshot = publisher.snapshot()
    assert snapshot["display"]["max_pending"] <= 2
    assert snapshot["display"]["dropped"] > 0

    with lock:
        assert logger.images == []

    release_display.set()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        with lock:
            if logger.display_sequences and logger.display_sequences[-1] == 63:
                break
        time.sleep(0.01)
    else:
        pytest.fail("display telemetry did not recover to its newest pending state")
    assert publisher.close() == {"exterior": True, "wrist": True, "display": True}


def test_live_telemetry_recovers_from_image_failure_with_supported_rgb_payload(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scenario = _load_live_scenario(monkeypatch, "antioch_live_image_failure_test")
    lock = threading.Lock()

    class Logger:
        def __init__(self) -> None:
            self.fail_once = True
            self.images: list[tuple[str, tuple[int, ...], str, bool]] = []

        def image(self, path: str, value: np.ndarray) -> None:
            if path == "camera/exterior" and self.fail_once:
                self.fail_once = False
                raise RuntimeError("injected image transport failure")
            with lock:
                self.images.append(
                    (path, value.shape, str(value.dtype), value.flags.c_contiguous)
                )

        def scalar(self, _path: str, _value: float) -> None:
            pass

        def value(self, _path: str, _value: object) -> None:
            pass

    noncontiguous_rgba = np.zeros((224, 224, 4), dtype=np.uint8)
    noncontiguous_rgba[..., :3] = [140, 30, 10]
    noncontiguous_rgba[..., 3] = 255

    class Camera:
        def get_data(self, annotator: str) -> tuple[np.ndarray, dict]:
            assert annotator == "rgb"
            return noncontiguous_rgba, {"renderingFrame": 1}

    frame = scenario._camera_frame(Camera(), view="wrist")
    assert frame.rgb is not None and frame.rgb.flags.c_contiguous
    pair = scenario.CameraPair(True, frame, frame, mean_difference=10.0)
    logger = Logger()
    publisher = scenario.LiveTelemetryPublisher(logger)
    publisher.publish_camera_pair(pair, 1)

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if publisher.snapshot()["exterior"]["failures"] == 1:
            break
        time.sleep(0.01)
    publisher.publish_camera_pair(pair, 2)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        with lock:
            paths = [item[0] for item in logger.images]
        if "camera/exterior" in paths and "camera/wrist" in paths:
            break
        time.sleep(0.01)
    else:
        pytest.fail("camera publication did not recover after a typed logger failure")

    with lock:
        assert all(item[1:] == ((224, 224, 3), "uint8", True) for item in logger.images)
    output = capsys.readouterr().out
    assert (
        "NPA_OPENPI_IMAGE_LOG_ERROR view=exterior render_sequence=1 "
        "phase=logger_image_encode_or_transport error_type=RuntimeError"
    ) in output
    assert publisher.close() == {"exterior": True, "wrist": True, "display": True}


def test_policy_future_blockage_does_not_suppress_viewer_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _load_live_scenario(monkeypatch, "antioch_live_policy_stall_test")
    release_policy = threading.Event()
    values: list[int] = []

    class Logger:
        def image(self, _path: str, _value: object) -> None:
            pass

        def scalar(self, _path: str, _value: float) -> None:
            pass

        def value(self, path: str, value: object) -> None:
            if path == "decision" and isinstance(value, dict):
                values.append(int(value["render_sequence"]))

    executor = scenario.ThreadPoolExecutor(max_workers=1)
    pending = executor.submit(release_policy.wait, 3.0)
    publisher = scenario.LiveTelemetryPublisher(Logger())
    for sequence in range(1, 21):
        assert not pending.done()
        publisher.publish_display(
            scenario.DisplayPublication(
                render_sequence=sequence,
                numeric_groups=(("decision", (("render_sequence", sequence),)),),
                values=(),
            )
        )
        time.sleep(0.005)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and (not values or values[-1] != 20):
        time.sleep(0.01)
    assert values and values[-1] == 20
    assert not pending.done()
    assert publisher.snapshot()["display"]["max_pending"] == 1
    release_policy.set()
    assert pending.result(timeout=1.0) is True
    executor.shutdown()
    assert publisher.close() == {"exterior": True, "wrist": True, "display": True}


def test_live_camera_optics_and_stock_franka_mount_are_explicit_and_rigid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _load_live_scenario(monkeypatch, "antioch_live_camera_geometry_test")
    exterior = scenario._camera_optical_config("exterior")
    wrist = scenario._camera_optical_config("wrist")
    assert exterior == {
        "focal_length": 18.0,
        "horizontal_aperture": 36.0,
        "vertical_aperture": 36.0,
        "clipping_range": (0.01, 100.0),
        "focus_distance": 1.0,
        "f_stop": 0.0,
    }
    assert wrist["focal_length"] == 12.0
    assert wrist["horizontal_aperture"] == 36.0

    hand = np.asarray([0.0, 0.0, 0.0])
    left = np.asarray([0.2, -0.04, 0.0])
    right = np.asarray([0.2, 0.04, 0.0])
    eye, target, up = scenario._wrist_camera_pose_from_points(hand, left, right)
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.asarray([0.5, -0.2, 0.3])
    moved = [rotation @ point + translation for point in (hand, left, right)]
    moved_eye, moved_target, moved_up = scenario._wrist_camera_pose_from_points(*moved)
    np.testing.assert_allclose(moved_eye, rotation @ eye + translation)
    np.testing.assert_allclose(moved_target, rotation @ target + translation)
    np.testing.assert_allclose(moved_up, rotation @ up)
    assert np.linalg.norm(eye - 0.5 * (left + right)) > 0.1
    assert scenario._point_in_camera_frame(
        target,
        (eye, target, up),
        wrist,
    )


def test_live_camera_optics_are_applied_to_usd_camera(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _load_live_scenario(monkeypatch, "antioch_live_camera_optics_test")
    values: dict[str, object] = {}

    class Attribute:
        def __init__(self, name: str) -> None:
            self.name = name

        def Set(self, value: object) -> None:  # noqa: N802
            values[self.name] = value

    class Camera:
        def __init__(self, _prim: object) -> None:
            pass

        def __getattr__(self, name: str):
            assert name.startswith("Create") and name.endswith("Attr")
            key = name.removeprefix("Create").removesuffix("Attr")
            key = re.sub(r"(?<!^)(?=[A-Z])", "_", key).lower()
            return lambda: Attribute(key)

    pxr = types.ModuleType("pxr")
    pxr.Gf = SimpleNamespace(Vec2f=lambda *items: tuple(items))
    pxr.UsdGeom = SimpleNamespace(Camera=Camera)
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    stage = SimpleNamespace(GetPrimAtPath=lambda _path: object())
    scenario._configure_camera_optics(stage, "/camera", "wrist")
    assert values == scenario._camera_optical_config("wrist")


def test_live_camera_pair_classifies_each_view_freshness_semantics_and_distinctness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _load_live_scenario(monkeypatch, "antioch_live_camera_pair_test")

    class Camera:
        def __init__(self, frame: np.ndarray | None) -> None:
            self.frame = frame

        def get_data(self, annotator: str) -> tuple[np.ndarray | None, dict]:
            assert annotator == "rgb"
            return self.frame, {"renderingFrame": 1}

    rows, columns = np.indices((224, 224))
    exterior = np.zeros((224, 224, 4), dtype=np.uint8)
    texture = (rows * 2 + columns * 3) % 255
    exterior[..., :3] = texture[..., None]
    exterior[95:115, 100:120, :3] = [240, 5, 4]
    exterior[..., 3] = 255
    wrist = np.roll(exterior, 37, axis=1)
    no_cube = exterior.copy()
    no_cube[95:115, 100:120, :3] = [20, 120, 40]

    accepted = scenario._validate_camera_pair(
        Camera(exterior),
        Camera(wrist),
        render_sequence=5,
        last_accepted_render_sequence=4,
        exterior_cube_in_frame=True,
        wrist_cube_in_frame=True,
    )
    assert accepted.accepted is True
    assert accepted.exterior.red_cube_pixels >= 20
    assert accepted.mean_difference >= scenario.MIN_CAMERA_PAIR_DIFFERENCE

    cases = (
        (Camera(None), Camera(wrist), 5, 4, True, True, "exterior", "missing"),
        (Camera(no_cube), Camera(wrist), 5, 4, True, True, "exterior", "cube_not_visible"),
        (Camera(exterior), Camera(np.zeros_like(wrist)), 5, 4, True, True, "wrist", "blank"),
        (Camera(exterior), Camera(wrist), 4, 4, True, True, "pair", "stale"),
        (Camera(exterior), Camera(wrist), 5, 4, False, True, "exterior", "cube_out_of_frame"),
        (Camera(exterior), Camera(wrist), 5, 4, True, False, "wrist", "cube_out_of_frame"),
        (Camera(exterior), Camera(exterior.copy()), 5, 4, True, True, "pair", "not_distinct"),
    )
    for ext, wr, sequence, last, ext_cube, wrist_cube, view, reason in cases:
        rejected = scenario._validate_camera_pair(
            ext,
            wr,
            render_sequence=sequence,
            last_accepted_render_sequence=last,
            exterior_cube_in_frame=ext_cube,
            wrist_cube_in_frame=wrist_cube,
        )
        assert rejected.accepted is False
        assert (rejected.rejected_view, rejected.reason) == (view, reason)


def test_live_observation_mapping_and_telemetry_blueprint_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _load_live_scenario(monkeypatch, "antioch_live_mapping_blueprint_test")
    assert scenario.logger.root == scenario.TELEMETRY_ROOT == "openpi-live"
    exterior = np.zeros((224, 224, 3), dtype=np.uint8)
    wrist = np.ones((224, 224, 3), dtype=np.uint8)
    joints = np.asarray([*scenario.DROID_RESET_JOINTS, 0.04, 0.04], dtype=np.float32)
    observation = scenario._build_policy_observation(exterior, wrist, joints, "prompt")
    assert tuple(observation) == (
        "observation/exterior_image_1_left",
        "observation/wrist_image_left",
        "observation/joint_position",
        "observation/gripper_position",
        "prompt",
    )
    assert observation["observation/exterior_image_1_left"] is exterior
    assert observation["observation/wrist_image_left"] is wrist
    assert abs(float(observation["observation/gripper_position"][0])) < 1e-6

    class Blueprint:
        def Spatial2DView(self, **kwargs):  # noqa: N802, ANN202
            return {"kind": "2d", **kwargs}

        def Spatial3DView(self, **kwargs):  # noqa: N802, ANN202
            return {"kind": "3d", **kwargs}

        def TimeSeriesView(self, **kwargs):  # noqa: N802, ANN202
            return {"kind": "series", **kwargs}

        def TextLogView(self, **kwargs):  # noqa: N802, ANN202
            return {"kind": "text", **kwargs}

        def Horizontal(self, *views, **kwargs):  # noqa: N802, ANN202
            return {"kind": "horizontal", "views": views, **kwargs}

        def Vertical(self, *views, **kwargs):  # noqa: N802, ANN202
            return {"kind": "vertical", "views": views, **kwargs}

        def Tabs(self, *views, **kwargs):  # noqa: N802, ANN202
            return {"kind": "tabs", "views": views, **kwargs}

        def Blueprint(self, *children, **kwargs):  # noqa: N802, ANN202
            return {"kind": "blueprint", "children": children, **kwargs}

    blueprint = scenario._camera_blueprint(Blueprint())
    vertical = blueprint["children"][0]
    cameras, lower = vertical["views"]
    scene, telemetry = lower["views"]
    views = [*cameras["views"], scene, *telemetry["views"]]
    assert [view["origin"] for view in views] == [
        scenario._resolved_telemetry_entity(scenario.CAMERA_EXTERIOR_ENTITY),
        scenario._resolved_telemetry_entity(scenario.CAMERA_WRIST_ENTITY),
        scenario._resolved_telemetry_entity(scenario.SCENE_ENTITY),
        scenario._resolved_telemetry_entity(scenario.DECISION_METRICS_ENTITY),
        scenario._resolved_telemetry_entity(scenario.GRASP_METRICS_ENTITY),
        scenario._resolved_telemetry_entity(scenario.FRANKA_ACTION_ENTITY),
        scenario._resolved_telemetry_entity(scenario.CAMERA_METRICS_ENTITY),
        scenario._resolved_telemetry_entity(scenario.POLICY_ERROR_ENTITY),
    ]
    assert all(origin.startswith(f"{scenario.TELEMETRY_ROOT}/") for origin in (
        view["origin"] for view in views
    ))
    assert cameras["column_shares"] == [1.0, 1.0]
    assert lower["column_shares"] == [1.0, 1.0]
    assert vertical["row_shares"] == [1.0, 1.0]
    assert blueprint["auto_layout"] is False

    rrb = pytest.importorskip("rerun.blueprint")
    assert scenario._camera_blueprint(rrb) is not None


def test_live_policy_request_round_trip_preserves_exact_camera_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _load_live_scenario(monkeypatch, "antioch_live_request_identity_test")

    class Packer:
        def pack(self, observation: dict) -> bytes:
            assert observation == {"prompt": "pick up the red cube"}
            return b"packed-request"

    protocol = types.ModuleType("openpi_protocol")
    protocol.Packer = Packer
    protocol.unpackb = lambda payload: {"actions": payload}
    monkeypatch.setitem(sys.modules, "openpi_protocol", protocol)

    class Connection:
        sent: list[bytes] = []

        def send(self, payload: bytes) -> None:
            self.sent.append(payload)

        def recv(self, *, timeout: float) -> bytes:
            assert timeout == scenario.MAX_RESPONSE_AGE_SECONDS
            return b"policy-response"

    client = scenario.SafePolicyClient()
    client._connection = Connection()
    response, latency, pair_id, render_sequence = client.infer(
        scenario.PolicyRequest(
            observation={"prompt": "pick up the red cube"},
            camera_pair_id=17,
            render_sequence=913,
        )
    )
    assert response == {"actions": b"policy-response"}
    assert latency >= 0.0
    assert (pair_id, render_sequence) == (17, 913)
    assert client._connection.sent == [b"packed-request"]


def test_live_droid_jointpos_and_gripper_mapping_matches_pinned_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _load_live_scenario(monkeypatch, "antioch_live_action_test")

    open_state = np.asarray([*scenario.DROID_RESET_JOINTS, 0.04, 0.04])
    closed_state = np.asarray([*scenario.DROID_RESET_JOINTS, 0.0, 0.0])
    assert scenario._droid_gripper_observation(open_state) == pytest.approx(0.0)
    assert scenario._droid_gripper_observation(closed_state) == pytest.approx(1.0)
    assert scenario._isaac_finger_target(0.0) == pytest.approx(0.04)
    assert scenario._isaac_finger_target(1.0) == pytest.approx(0.0)

    actions = np.tile(open_state[:8], (15, 1)).astype(np.float64)
    actions[:, 7] = np.asarray(
        [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    )
    targets, evidence = scenario._validated_actions(
        {"actions": actions}, open_state
    )
    np.testing.assert_allclose(targets[:, :7], actions[:, :7])
    assert set(np.unique(targets[:, 7])) <= {0.0, 1.0}
    assert evidence["raw_gripper_range_mismatches"] == 4
    assert evidence["raw_joint_limit_mismatches"] == 0
    assert evidence["joint_limit_projections"] == 0
    assert evidence["joint_step_projections"] == 0


def test_live_scene_is_tabletop_lit_and_droid_reset_aligned() -> None:
    source = (EXAMPLE / "src/scenario_v2.py").read_text(encoding="utf-8")
    assert 'world.scene.add_ground_plane(z_position=-0.75)' in source
    assert 'prim_path="/World/Tabletop"' in source
    assert 'prim_path="/World/Cube"' in source
    assert "position=np.array(CUBE_INITIAL_POSITION)" in source
    assert "robot.set_joint_positions(" in source
    assert 'WRIST_CAMERA_PATH = "/World/PolicyWrist"' in source
    assert "wrist_mount = _calibrate_wrist_camera_mount(" in source
    assert "wrist_pose = _aim_wrist_camera(world.stage, wrist_mount)" in source
    assert 'UsdLux.DomeLight.Define(stage, "/World/PolicyFillLight")' in source
    assert 'UsdLux.DistantLight.Define(stage, "/World/PolicyKeyLight")' in source


def test_live_protocol_codec_round_trips_arrays_and_rejects_objects() -> None:
    pytest.importorskip("msgpack")
    path = EXAMPLE / "src/openpi_protocol.py"
    spec = importlib.util.spec_from_file_location("openpi_protocol_test", path)
    assert spec is not None and spec.loader is not None
    codec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codec)

    value = np.arange(24, dtype=np.float32).reshape(3, 8)
    decoded = codec.unpackb(codec.Packer().pack({"actions": value}))
    np.testing.assert_array_equal(decoded["actions"], value)
    with pytest.raises(ValueError, match="unsupported array dtype"):
        codec.Packer().pack(np.asarray([object()], dtype=object))


def test_live_sim_image_contains_only_protocol_dependencies() -> None:
    dockerfile = (EXAMPLE / "Dockerfile").read_text(encoding="utf-8")
    assert dockerfile.startswith("FROM antioch-engine/isaac-sim-6.0.1:0.3.63\n")
    assert 'npa.antioch.live-transport="declared-port-double-wss-v1"' in dockerfile
    assert '"msgpack==1.1.1"' in dockerfile
    assert '"websockets==15.0.1"' in dockerfile
    assert "/workspace/project" in dockerfile
    assert "/tmp/npa-home/.cache \\" in dockerfile
    assert "/tmp/npa-home/.cache/ov" in dockerfile
    assert (
        "/usr/local/lib/python3.12/dist-packages/isaacsim/kit/cache/DerivedDataCache"
        in dockerfile
    )
    assert (
        "/usr/local/lib/python3.12/dist-packages/isaacsim/kit/data/documents/Kit/"
        "apps/Isaac-Sim Python/scripts"
        in dockerfile
    )
    assert (
        "/usr/local/lib/python3.12/dist-packages/isaacsim/kit/data/documents/Kit/"
        "shared"
        in dockerfile
    )
    assert "ENV HOME=/tmp/npa-home" in dockerfile
    assert "COPY --chown=1000:1000 src/ /workspace/project/src/" in dockerfile
    assert "USER 1000:1000" in dockerfile
    assert (
        'ENTRYPOINT ["/usr/local/bin/python", "/workspace/project/src/relay_bridge.py"'
        in dockerfile
    )
    assert '"--wait-for-bundle"]' in dockerfile
    assert 'args.service_command not in ([], ["sleep", "infinity"])' in (
        EXAMPLE / "src/relay_bridge.py"
    ).read_text(encoding="utf-8")
    assert "git clone" not in dockerfile
    assert "checkpoint" not in dockerfile.lower()


def test_live_example_documents_supported_renewal_boundary() -> None:
    readme = (EXAMPLE / "README.md").read_text(encoding="utf-8")
    assert "antioch services cp" in readme
    assert "finite supported timeout" in readme
    assert "resets the simulated episode" in readme
    assert "not one infinitely lived simulator process" in readme


def test_runtime_staging_keeps_private_project_id_out_of_source(tmp_path: Path) -> None:
    destination = tmp_path / "runtime"
    live._stage_project(EXAMPLE, destination, "assigned-project-for-test")
    staged = yaml.safe_load((destination / "antioch.yaml").read_text(encoding="utf-8"))
    source = yaml.safe_load((EXAMPLE / "antioch.yaml").read_text(encoding="utf-8"))
    assert staged["id"] == "assigned-project-for-test"
    assert source["id"] == "replace-at-runtime"
    assert destination.stat().st_mode & 0o777 == 0o700
    assert (destination / "antioch.yaml").stat().st_mode & 0o777 == 0o600
    for name in ("scenario_v2.py", "openpi_protocol.py", "relay_bridge.py"):
        assert (destination / "src" / name).stat().st_mode & 0o777 == 0o644


def test_supervisor_has_finite_run_boundary_but_no_total_limit(tmp_path: Path) -> None:
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    for name in ("scenario_v2.py", "openpi_protocol.py", "relay_bridge.py"):
        (source_dir / name).write_text("# reviewed source\n", encoding="utf-8")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name in live.REQUIRED_BUNDLE_FILES:
        (bundle / name).write_text(f"private-{name}\n", encoding="utf-8")
    script = tmp_path / "supervise.sh"
    live._write_supervisor(
        script,
        cli_path=Path("/opt/antioch/bin/antioch"),
        python_path=Path("/opt/npa/bin/python"),
        client_bundle=bundle,
        stop_file=tmp_path / ".stop",
        active_state_path=tmp_path / "active-run.json",
        scenario_timeout_seconds=14_400,
        owner_identity="owned-adapter",
        session_id="owned-session",
    )
    source = script.read_text(encoding="utf-8")
    assert "while [ ! -f" in source
    assert "scenario run --scenario openpi_droid_live" in source
    assert "--timeout 14400 --stream --verbose" in source
    assert "NPA_ANTIOCH_RENEWAL" in source
    assert "npa.workbench.antioch.live_reconcile" in source
    assert "PYTHONPATH=" in source
    assert "NPA_ANTIOCH_RECONCILED_TERMINAL" in source
    assert "--owner-identity owned-adapter" in source
    assert "--session-id owned-session" in source
    assert source.index("npa.workbench.antioch.live_reconcile") < source.index(
        "scenario run --scenario openpi_droid_live"
    )
    assert "services cp" in source
    assert "services exec sim /bin/sh -lc" in source
    assert "npa-live-supervisor-source-" in source
    assert "sha256sum /workspace/project/src/scenario_v2.py" in source
    assert "sha256sum /workspace/project/src/openpi_protocol.py" in source
    assert "install -m 0644" in source
    assert "services up --json" in source
    assert "services build --service sim --json" in source
    assert "services exec sim /bin/true" in source
    assert "NPA_ANTIOCH_SERVICE_NOT_READY" in source
    assert "npa-live-client-generation-" in source
    assert "npa-live-client-upload-" in source
    assert "install -m 0600" in source
    upload_dirs = list(tmp_path.glob(".bundle-upload-*"))
    assert len(upload_dirs) == 1
    assert upload_dirs[0].stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == 0o644 for path in upload_dirs[0].iterdir())
    assert "mv -Tf" in source
    assert "sleep 15" in source
    assert "timeout 14400s" not in source
    assert script.stat().st_mode & 0o777 == 0o700
    subprocess.run(["sh", "-n", str(script)], check=True)


def test_relay_supervisor_has_no_credential_values_in_arguments(tmp_path: Path) -> None:
    script = tmp_path / "relay-supervise.sh"
    live._write_relay_supervisor(
        script,
        python_path=Path("/opt/npa/bin/python"),
        client_bundle=tmp_path / "private-bundle",
        stop_file=tmp_path / ".stop",
        state_path=tmp_path / "relay-state.json",
    )
    source = script.read_text(encoding="utf-8")
    assert "npa.workbench.antioch.relay" in source
    assert "--local-port 18444" in source
    assert "api-key" not in source
    assert "Authorization" not in source
    assert "while [ ! -f" in source
    subprocess.run(["sh", "-n", str(script)], check=True)


def test_bridge_supervisor_uses_short_health_exec_calls(tmp_path: Path) -> None:
    script = tmp_path / "bridge-supervise.sh"
    live._write_bridge_supervisor(
        script,
        cli_path=Path("/opt/antioch/bin/antioch"),
        stop_file=tmp_path / ".stop",
    )
    source = script.read_text(encoding="utf-8")
    assert "services exec sim /usr/local/bin/python -c" in source
    assert "socket.create_connection" in source
    assert "NPA_ANTIOCH_BRIDGE_HEALTHY" in source
    assert "NPA_ANTIOCH_BRIDGE_NOT_READY" in source
    assert "relay_bridge.py" not in source
    assert "nohup" not in source
    assert "api-key" not in source
    assert "while [ ! -f" in source
    subprocess.run(["sh", "-n", str(script)], check=True)


def test_relay_certificate_covers_verified_localhost_endpoint() -> None:
    from cryptography import x509

    _ca, certificate, _key = live._relay_certificate()
    names = (
        x509.load_pem_x509_certificate(certificate)
        .extensions.get_extension_for_class(x509.SubjectAlternativeName)
        .value
    )

    assert names.get_values_for_type(x509.IPAddress)[0].compressed == "127.0.0.1"
    assert names.get_values_for_type(x509.DNSName) == []


def test_client_bundle_requires_private_files(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name, content in {
        "ca.crt": "certificate",
        "api-key": "x" * 48,
        "endpoint.json": '{"scheme":"wss","host":"example.invalid","port":443}',
    }.items():
        path = bundle / name
        path.write_text(content, encoding="utf-8")
        path.chmod(0o600)
    live._validate_upstream_bundle(bundle)
    (bundle / "api-key").chmod(0o644)
    try:
        live._validate_upstream_bundle(bundle)
    except live.AntiochLiveError as exc:
        assert "group/world" in str(exc)
    else:
        raise AssertionError("a public client credential was accepted")


def test_runtime_bundle_adds_private_relay_identity(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    for name, content in {
        "ca.crt": "certificate",
        "api-key": "x" * 48,
        "endpoint.json": '{"scheme":"wss","host":"example.invalid","port":443}',
    }.items():
        path = upstream / name
        path.write_text(content, encoding="utf-8")
        path.chmod(0o600)

    destination = tmp_path / "runtime-bundle"
    live._prepare_runtime_bundle(upstream, destination)

    assert set(path.name for path in destination.iterdir()) == set(
        live.REQUIRED_BUNDLE_FILES
    )
    assert destination.stat().st_mode & 0o777 == 0o700
    assert all(
        (destination / name).stat().st_mode & 0o777 == 0o600
        for name in live.REQUIRED_BUNDLE_FILES
    )
    assert "BEGIN PRIVATE KEY" in (destination / "relay-server.key").read_text(
        encoding="utf-8"
    )
    assert "example.invalid" not in (destination / "relay-server.crt").read_text(
        encoding="utf-8"
    )


def test_initial_bundle_staging_recovers_from_service_recreation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name in live.REQUIRED_BUNDLE_FILES:
        (bundle / name).write_text("private", encoding="utf-8")
    calls: list[str] = []

    class Cli:
        directory_attempts = 0

        def services_exec(self, _runtime, _service, command):  # noqa: ANN001, ANN202
            calls.append("exec:" + str(command[0]))
            if command[:2] == ["install", "-d"]:
                self.directory_attempts += 1

        def services_copy(self, _runtime, source, _destination):  # noqa: ANN001, ANN202
            calls.append("copy:" + source.name)
            assert source.stat().st_mode & 0o777 == 0o644
            assert source.parent.stat().st_mode & 0o777 == 0o700
            if self.directory_attempts == 1:
                raise AntiochCliError("container recreated")

    monkeypatch.setattr(live.time, "sleep", lambda _seconds: None)
    cli = Cli()
    live._stage_private_bundle(
        cli,  # type: ignore[arg-type]
        runtime=runtime,
        client_bundle=bundle,
        attempts=2,
    )

    assert cli.directory_attempts == 2
    assert calls.count("copy:ca.crt") == 2
    assert calls[-1] == "exec:/bin/sh"
    assert not list(runtime.glob(".bundle-upload-*"))


def test_runtime_source_is_staged_through_supported_service_copy(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src"
    source.mkdir()
    for name in ("scenario_v2.py", "openpi_protocol.py", "relay_bridge.py"):
        (source / name).write_text("# reviewed public source\n", encoding="utf-8")
    calls: list[tuple[str, object]] = []

    class Cli:
        def services_exec(self, _runtime, _service, command):  # noqa: ANN001, ANN202
            calls.append(("exec", command))
            if command[0] == "sha256sum":
                return hashlib.sha256(b"# reviewed public source\n").hexdigest()
            return ""

        def services_copy(self, _runtime, path, destination):  # noqa: ANN001, ANN202
            calls.append(("copy", (path.name, destination)))

    live._stage_runtime_source(Cli(), runtime=tmp_path)  # type: ignore[arg-type]

    assert calls[0][0] == "exec"
    copies = [call for call in calls if call[0] == "copy"]
    assert copies[0][1][0] == "scenario_v2.py"
    assert copies[0][1][1].startswith("sim:/tmp/npa-live-source-")
    assert copies[0][1][1].endswith("/scenario_v2.py")
    assert copies[1][1][0] == "openpi_protocol.py"
    assert copies[1][1][1].startswith("sim:/tmp/npa-live-source-")
    assert copies[1][1][1].endswith("/openpi_protocol.py")
    assert copies[2][1][0] == "relay_bridge.py"
    assert copies[2][1][1].endswith("/relay_bridge.py")
    assert calls[-1][0] == "exec"


def test_runtime_source_staging_recovers_from_service_recreation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "src"
    source.mkdir()
    for name in ("scenario_v2.py", "openpi_protocol.py", "relay_bridge.py"):
        (source / name).write_text("# reviewed public source\n", encoding="utf-8")
    copies: list[str] = []

    class Cli:
        attempts = 0

        def services_exec(self, _runtime, _service, command):  # noqa: ANN001, ANN202
            if command[:2] == ["install", "-d"]:
                self.attempts += 1
            if command[0] == "sha256sum":
                return hashlib.sha256(b"# reviewed public source\n").hexdigest()
            return ""

        def services_copy(self, _runtime, path, _destination):  # noqa: ANN001, ANN202
            copies.append(path.name)
            if self.attempts == 1 and path.name == "openpi_protocol.py":
                raise AntiochCliError("container recreated")

    monkeypatch.setattr(live.time, "sleep", lambda _seconds: None)
    cli = Cli()
    live._stage_runtime_source(cli, runtime=tmp_path, attempts=2)  # type: ignore[arg-type]

    assert cli.attempts == 2
    assert copies.count("scenario_v2.py") == 2
    assert copies.count("openpi_protocol.py") == 2
    assert copies.count("relay_bridge.py") == 1


def test_live_cleanup_cancels_only_exact_active_scenario(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = iter(
        (
            [
                {
                    "scenario": "openpi_droid_live",
                    "phase": "running",
                    "scenario_run_id": "exact-live-run",
                },
                {
                    "scenario": "other_scenario",
                    "phase": "running",
                    "scenario_run_id": "unrelated-run",
                },
                {
                    "scenario": "openpi_droid_live",
                    "phase": "completed",
                    "scenario_run_id": "terminal-run",
                },
            ],
            [],
            [],
            [],
        )
    )
    cancelled: list[str] = []

    class Cli:
        def list_for_project(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            return next(pages)

        def machine_status(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            return _daemon_status()

        def show(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            return {
                "scenario": "openpi_droid_live",
                "project_id": "assigned-project-for-test",
                "phase": "running",
            }

        def cancel(self, _runtime, *, kind, remote_id):  # noqa: ANN001, ANN202
            assert kind == "scenario"
            cancelled.append(remote_id)
            return {}

    monkeypatch.setattr(live.time, "sleep", lambda _seconds: None)
    count = live._cancel_remote_live_runs(
        Cli(),  # type: ignore[arg-type]
        runtime=tmp_path,
        project_id="assigned-project-for-test",
    )

    assert count == 1
    assert cancelled == ["exact-live-run"]


def test_live_cleanup_accepts_exact_list_cancel_terminalization_race(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = iter(
        (
            [
                {
                    "scenario": "openpi_droid_live",
                    "phase": "booting",
                    "scenario_run_id": "just-terminalized",
                }
            ],
            [],
            [],
            [],
        )
    )

    class Cli:
        def list_for_project(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            return next(pages)

        def machine_status(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            return _daemon_status()

        def cancel(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            raise AntiochCliError("scenario run 'just-terminalized' was not found")

    monkeypatch.setattr(live.time, "sleep", lambda _seconds: None)
    count = live._cancel_remote_live_runs(
        Cli(),  # type: ignore[arg-type]
        runtime=tmp_path,
        project_id="assigned-project-for-test",
        attempts=4,
    )

    assert count == 0


def test_live_cleanup_accepts_terminal_failed_stream_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Cli:
        def list_for_project(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            return []

        def machine_status(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            return _daemon_status(
                state="failed", scenario_run_id="terminal", session=False
            )

        def show(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            return {
                "scenario": "openpi_droid_live",
                "project_id": "assigned-project-for-test",
                "phase": "completed",
                "outcome": "cancelled",
            }

    monkeypatch.setattr(live.time, "sleep", lambda _seconds: None)
    assert (
        live._cancel_remote_live_runs(
            Cli(),  # type: ignore[arg-type]
            runtime=tmp_path,
            project_id="assigned-project-for-test",
        )
        == 0
    )


def test_live_cleanup_tolerates_typed_missing_run_during_cancel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = iter(
        (
            [
                {
                    "scenario": "openpi_droid_live",
                    "phase": "running",
                    "scenario_run_id": "exact-live-run",
                }
            ],
            [],
            [],
            [],
        )
    )

    class Cli:
        def list_for_project(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            return next(pages)

        def machine_status(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            return _daemon_status()

        def cancel(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            raise AntiochCliError(
                "remote run is gone",
                error_type="scenario_not_found",
                http_status=404,
            )

    monkeypatch.setattr(live.time, "sleep", lambda _seconds: None)
    assert (
        live._cancel_remote_live_runs(
            Cli(),  # type: ignore[arg-type]
            runtime=tmp_path,
            project_id="assigned-project-for-test",
        )
        == 0
    )


def test_live_reconcile_adopts_only_machine_stream_owner(tmp_path: Path) -> None:
    class Cli:
        def list_for_project(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            return [
                {
                    "scenario": "openpi_droid_live",
                    "phase": "running",
                    "scenario_run_id": "exact-active",
                }
            ]

        def machine_status(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            return _daemon_status(state="ready", scenario_run_id="exact-active")

    active = live_reconcile._active_run(
        Cli(),  # type: ignore[arg-type]
        runtime=tmp_path,
        project_id="assigned-project-for-test",
    )
    assert active is not None
    assert active["scenario_run_id"] == "exact-active"


def test_live_reconcile_rejects_unlisted_stream_owner(tmp_path: Path) -> None:
    class Cli:
        def list_for_project(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            return []

        def machine_status(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            return _daemon_status(state="ready", scenario_run_id="other")

    with pytest.raises(
        live_reconcile.AntiochLiveReconcileError,
        match="absent from the exact project",
    ):
        live_reconcile._active_run(
            Cli(),  # type: ignore[arg-type]
            runtime=tmp_path,
            project_id="assigned-project-for-test",
        )


def test_daemon_liveness_requires_exact_machine_stream_owner(tmp_path: Path) -> None:
    class Cli:
        def list_for_project(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            return [
                {
                    "scenario": "openpi_droid_live",
                    "phase": "running",
                    "scenario_run_id": "listed-but-unowned",
                }
            ]

        def machine_status(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            return _daemon_status()

    cli = Cli()
    assert live_reconcile._active_run(  # type: ignore[arg-type]
        cli,
        runtime=tmp_path,
        project_id="assigned-project-for-test",
    ) is not None
    assert (
        live_reconcile._active_run(  # type: ignore[arg-type]
            cli,
            runtime=tmp_path,
            project_id="assigned-project-for-test",
            require_stream_owner=True,
        )
        is None
    )


def test_stdout_compatible_cached_stream_cannot_mask_dead_rome_heartbeat() -> None:
    machine = _daemon_status(state="ready", scenario_run_id="exact-active")
    runtime_status = machine["runtime_status"]
    assert isinstance(runtime_status, dict)
    runtime_status["guest_state"] = "unreachable"
    runtime_status["guest_failure_started_at"] = int(time.time() * 1_000_000)

    with pytest.raises(
        live_reconcile.AntiochLiveReconcileError,
        match="Rome daemon liveness is unhealthy",
    ):
        live_reconcile._daemon_runtime_snapshot(machine)


def test_stale_exact_stream_without_vendor_session_fails_closed() -> None:
    machine = _daemon_status(
        state="ready", scenario_run_id="exact-active", session=False
    )
    with pytest.raises(
        live_reconcile.AntiochLiveReconcileError,
        match="process/session/stream lease ownership",
    ):
        live_reconcile._daemon_runtime_snapshot(machine)


def test_malformed_or_stale_daemon_observation_fails_closed() -> None:
    malformed = _daemon_status(state="ready", scenario_run_id="exact-active")
    runtime = malformed["runtime"]
    assert isinstance(runtime, dict)
    runtime["observed_at"] = "not-a-timestamp"
    with pytest.raises(
        live_reconcile.AntiochLiveReconcileError, match="timestamp is malformed"
    ):
        live_reconcile._daemon_runtime_snapshot(malformed)

    stale = _daemon_status(state="ready", scenario_run_id="exact-active")
    stale_runtime = stale["runtime"]
    assert isinstance(stale_runtime, dict)
    stale_runtime["observed_at"] = int((time.time() - 31) * 1_000_000)
    with pytest.raises(
        live_reconcile.AntiochLiveReconcileError,
        match="direct daemon observation is stale",
    ):
        live_reconcile._daemon_runtime_snapshot(stale)


def test_double_wss_relay_forwards_bounded_request_reply(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    ca, _certificate_bytes, _key = _certificate("127.0.0.1")
    for name, content in {
        "ca.crt": ca,
        "api-key": b"p" * 48,
        "endpoint.json": b'{"scheme":"wss","host":"127.0.0.1","port":443}',
    }.items():
        path = upstream / name
        path.write_bytes(content)
        path.chmod(0o600)
    bundle = tmp_path / "bundle"
    live._prepare_runtime_bundle(upstream, bundle)
    stop_file = tmp_path / ".stop"

    class Connection:
        def __init__(self, kind: str) -> None:
            self.kind = kind
            self.received = 0
            self.sent: list[bytes] = []

        def __enter__(self):  # noqa: ANN204
            return self

        def __exit__(self, *_args):  # noqa: ANN002, ANN204
            return None

        def recv(self, **_kwargs):  # noqa: ANN202
            self.received += 1
            if self.kind == "policy":
                return b"greeting" if self.received == 1 else b"response"
            if self.received == 1:
                return b"request"
            raise RuntimeError("test stream complete")

        def send(self, payload: bytes) -> None:
            self.sent.append(payload)
            if self.kind == "simulation" and payload == b"response":
                stop_file.touch()

    policy = Connection("policy")
    simulation = Connection("simulation")
    connection_order: list[str] = []

    def connect(uri: str, **kwargs):  # noqa: ANN003, ANN202
        assert kwargs["proxy"] is None
        assert kwargs["additional_headers"]["Authorization"].startswith("Api-Key ")
        kind = "policy" if uri.endswith(":443") else "simulation"
        connection_order.append(kind)
        return policy if kind == "policy" else simulation

    monkeypatch.setattr(live_relay, "connect", connect)
    state = live_relay.run_relay(
        bundle=bundle,
        local_port=18_444,
        stop_file=stop_file,
        state_path=tmp_path / "relay-state.json",
        owner_identity="test-owner",
    )

    assert policy.sent == [b"request"]
    assert simulation.sent == [b"greeting", b"response"]
    assert connection_order == ["simulation", "policy"]
    assert state["forwarded_requests"] == 1
    assert state["status"] == "stopped"


def test_cluster_relay_holds_stopped_state_until_controller_resumes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stop_file = tmp_path / "stop"
    stop_file.touch()
    state_path = tmp_path / "relay.json"
    sleeps = 0

    def resume(_seconds: float) -> None:
        nonlocal sleeps
        sleeps += 1
        stop_file.unlink()

    monkeypatch.setattr(live_relay.time, "sleep", resume)
    live_relay._publish_stopped_until_resumable(
        stop_file=stop_file,
        state_path=state_path,
        state={
            "schema_version": 2,
            "owner_identity": "owner",
            "status": "stopped",
        },
    )

    assert sleeps == 1
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "stopped"
    assert state["heartbeat_unix"] > 0


def test_live_metrics_bind_current_valid_camera_pair_to_policy_request() -> None:
    source = (EXAMPLE / "src/scenario_v2.py").read_text(encoding="utf-8")
    assert "camera_quality_schema=3" in source
    assert "action_horizon={ACTION_SHAPE[0]}" in source
    assert "action_dimension={ACTION_SHAPE[1]} action_finite=1" in source
    assert "camera_luminance_mean_current_min" in source
    assert "camera_luminance_variance_current_min" in source
    assert "camera_validated_requests += 1" in source
    assert "camera_rejected_pairs += 1" in source
    assert source.index("observation = _build_policy_observation(") < source.index(
        "camera_validated_requests += 1"
    )
    readiness = source.index("camera_policy_eligible = readiness.policy_eligible")
    request_gate = source.index(
        "elif chunk is None and pending is None and now >= next_attempt:"
    )
    request = source.index("requests += 1", request_gate)
    assert readiness < request_gate < request
    assert "chunk is not None\n                and camera_policy_eligible" in source


def test_live_stop_cancels_scenario_before_service(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    windows = {
        "scenario": iter((True, False, False)),
        "relay": iter((False,)),
        "bridge": iter((False,)),
    }
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setattr(
        live,
        "_read_state",
        lambda _project_id: {
            "project_id": "assigned-project-for-test",
            "session": "exact-session",
            "runtime": str(runtime),
            "cli": "/opt/antioch/bin/antioch",
        },
    )
    monkeypatch.setattr(live, "_session_running", lambda _session: False)
    monkeypatch.setattr(
        live,
        "_window_running",
        lambda _session, window: next(windows[window]),
    )
    monkeypatch.setattr(
        live,
        "_tmux",
        lambda *args, **_kwargs: calls.append("tmux:" + " ".join(args)),
    )

    class FakeCli:
        def __init__(self, _path: Path) -> None:
            pass

        def list_for_project(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            calls.append("list-live-runs")
            return []

        def machine_status(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            calls.append("machine-status")
            return _daemon_status()

        def services_down(self, _runtime: Path) -> None:
            calls.append("services-down")

    monkeypatch.setattr(live, "AntiochCli", FakeCli)
    result = live.stop_live(project_id="assigned-project-for-test")

    assert calls == [
        "tmux:send-keys -t exact-session:scenario.0 C-c",
        "list-live-runs",
        "machine-status",
        "list-live-runs",
        "machine-status",
        "list-live-runs",
        "machine-status",
        "services-down",
    ]
    assert result["service_stopped_after_scenario"] is True
    assert result["cancelled_remote_runs"] == 0
    assert (runtime / ".stop").stat().st_mode & 0o777 == 0o600


def test_start_live_failure_cleans_owned_session_and_service(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    source = tmp_path / "source"
    source.mkdir()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    monkeypatch.setattr(live, "_validate_upstream_bundle", lambda _path: None)
    monkeypatch.setattr(live, "ensure_runtime", lambda: Path("/opt/antioch"))
    monkeypatch.setattr(live, "live_state_root", lambda: tmp_path / "state")
    running = iter((False, True))
    monkeypatch.setattr(live, "_session_running", lambda _session: next(running))
    for name in (
        "_stage_project",
        "_prepare_runtime_bundle",
        "_write_supervisor",
        "_write_relay_supervisor",
        "_write_bridge_supervisor",
        "_stage_runtime_source",
        "_stage_private_bundle",
    ):
        monkeypatch.setattr(live, name, lambda *_args, **_kwargs: None)

    def tmux(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        calls.append("tmux:" + " ".join(args))
        if args[0] == "new-session":
            raise live.AntiochLiveError("injected startup failure")

    monkeypatch.setattr(live, "_tmux", tmux)

    class Cli:
        def __init__(self, _path: Path) -> None:
            pass

        def services_build(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            calls.append("services-build")

        def services_up(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            calls.append("services-up")

        def services_down(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            calls.append("services-down")

    monkeypatch.setattr(live, "AntiochCli", Cli)
    with pytest.raises(live.AntiochLiveError, match="injected startup failure"):
        live.start_live(
            source=source,
            project_id="assigned-project-for-test",
            client_bundle=bundle,
        )
    assert calls[-2:] == [
        "tmux:kill-session -t " + live._session_name("assigned-project-for-test"),
        "services-down",
    ]
