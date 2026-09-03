"""Live Franka camera -> OpenPI -> safe target loop for Antioch."""

from __future__ import annotations

import math
import operator
import ssl
import time
import contextlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Thread
from typing import Callable

import antioch

TELEMETRY_ROOT = "openpi-live"
CAMERA_EXTERIOR_ENTITY = "camera/exterior"
CAMERA_WRIST_ENTITY = "camera/wrist"
CAMERA_METRICS_ENTITY = "camera"
SCENE_ENTITY = "scene"
FRANKA_SCENE_ENTITY = "scene/franka"
DECISION_METRICS_ENTITY = "decision"
GRASP_METRICS_ENTITY = "grasp"
FRANKA_ACTION_ENTITY = "robot/franka"
POLICY_ERROR_ENTITY = "policy/error"
TASK_ENTITY = "task/label"


def _resolved_telemetry_entity(relative_entity: str) -> str:
    """Resolve an Antioch Logger-relative entity to its Rerun catalog path."""

    if not relative_entity or relative_entity.startswith("/") or relative_entity.endswith("/"):
        raise ValueError(f"invalid telemetry entity: {relative_entity!r}")
    return f"{TELEMETRY_ROOT}/{relative_entity}"


logger = antioch.Logger(TELEMETRY_ROOT)

CLIENT_ROOT = Path("/tmp/npa-live-client-current")
ACTION_SHAPE = (15, 8)
CONTROL_HZ = 15.0
CAMERA_SENSOR_TICK_RATE_HZ = CONTROL_HZ
PINNED_ANTIOCH_SDK_VERSION = "0.3.63"
PINNED_ANTIOCH_ENGINE = "isaac-sim-6.0.1"
TARGETS_PER_QUERY = 5
TELEMETRY_DISPLAY_HZ = 5.0
TELEMETRY_WORKER_JOIN_SECONDS = 0.5
CAMERA_READY_CONSECUTIVE_FRAMES = 2
CAMERA_STATUS_INTERVAL_FRAMES = 120
# A cold B200 model request can take tens of seconds even though warmed requests
# are normally tens of milliseconds. This is a stale-response safety deadline,
# not a real-time claim or a total run limit.
MAX_RESPONSE_AGE_SECONDS = 90.0
JOINT_LOW = (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973)
JOINT_HIGH = (2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973)
MAX_JOINT_STEP = 0.35
GRIPPER_JOINT_MAX = 0.04
GRIPPER_TOTAL_WIDTH_MAX = 2.0 * GRIPPER_JOINT_MAX
DROID_RESET_JOINTS = (
    0.0,
    -math.pi / 5.0,
    0.0,
    -4.0 * math.pi / 5.0,
    0.0,
    3.0 * math.pi / 5.0,
    0.0,
)
TASK_LABEL = "red_cube_pickup"
CUBE_SIZE_METERS = 0.07
CUBE_INITIAL_POSITION = (0.48, 0.0, CUBE_SIZE_METERS / 2.0)
PICKUP_LIFT_METERS = 0.05
PICKUP_HOLD_SECONDS = 1.0
GRIPPER_CONTACT_FORCE_NEWTONS = 0.1
MIN_CAMERA_LUMINANCE_MEAN = 5.0
MIN_CAMERA_LUMINANCE_VARIANCE = 25.0
MIN_CAMERA_PAIR_DIFFERENCE = 8.0
MIN_EXTERIOR_RED_CUBE_PIXELS = 20
EXTERIOR_CAMERA_PATH = "/World/PolicyExterior"
WRIST_CAMERA_PATH = "/World/PolicyWrist"
EXTERIOR_CAMERA_EYE = (1.45, -1.25, 0.95)
EXTERIOR_CAMERA_TARGET = (0.45, 0.0, 0.08)
WRIST_EYE_OFFSET_TOOL = (-0.12, 0.0, 0.10)
WRIST_TARGET_OFFSET_TOOL = (0.24, 0.0, -0.03)
STOCK_FRANKA_HAND_PATH = "/World/Franka/panda_hand"
STOCK_FRANKA_LEFT_FINGER_PATH = "/World/Franka/panda_leftfinger"
STOCK_FRANKA_RIGHT_FINGER_PATH = "/World/Franka/panda_rightfinger"


@dataclass(frozen=True)
class CameraFrame:
    rgb: object | None
    reason: str
    luminance_mean: float = 0.0
    luminance_variance: float = 0.0
    dynamic_range: float = 0.0
    red_cube_pixels: int = 0
    raw_min: float = 0.0
    raw_max: float = 0.0
    raw_nonzero: int = 0
    raw_channels: int = 0


@dataclass(frozen=True)
class CameraSample:
    frame: CameraFrame
    producer_marker: tuple[int, int] | None


class RtxRgbCamera:
    """Own one Isaac Sim 6 RTX camera authoring/runtime pair."""

    def __init__(self, authoring, sensor, producer_clock, output_buffer) -> None:
        self.authoring = authoring
        self.sensor = sensor
        self.producer_clock = producer_clock
        self.output_buffer = output_buffer

    @property
    def render_product_path(self) -> str:
        render_product = self.sensor.render_product
        prim = render_product.GetPrim()
        return str(prim.GetPath()) if prim and prim.IsValid() else ""

    def sample(self, *, view: str) -> CameraSample:
        # Isaac Sim 6 documents RGB as uint8 ``(height, width, 3)``.  Ask the
        # public CameraSensor API to copy directly into CPU memory so the live
        # controller never aliases the renderer's CUDA external-memory view.
        data, info = self.sensor.get_data("rgb", out=self.output_buffer)
        marker = _producer_marker_from_info(info)
        if marker is None:
            marker = _reference_time_marker(self.producer_clock)
        return CameraSample(_camera_frame_from_buffer(data, view=view), marker)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.sensor.detach_annotators("rgb")
        with contextlib.suppress(Exception):
            self.producer_clock.detach([self.render_product_path])


@dataclass(frozen=True)
class CameraPair:
    accepted: bool
    exterior: CameraFrame
    wrist: CameraFrame
    rejected_view: str = ""
    reason: str = ""
    mean_difference: float = 0.0


@dataclass(frozen=True)
class CameraReadinessDecision:
    policy_eligible: bool
    status: str
    reason: str
    consecutive_ready: int
    unavailable_frames: int
    emit_status: bool


@dataclass(frozen=True)
class PolicyRequest:
    observation: dict
    camera_pair_id: int
    render_sequence: int


@dataclass(frozen=True)
class WristCameraMount:
    eye_offset_tool: object
    look_direction_tool: object
    up_direction_tool: object


@dataclass(frozen=True)
class ImagePublication:
    view: str
    entity: str
    rgb: object
    render_sequence: int
    luminance_mean: float
    luminance_variance: float
    dynamic_range: float
    red_cube_pixels: int


@dataclass(frozen=True)
class DisplayPublication:
    render_sequence: int
    numeric_groups: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]
    values: tuple[tuple[str, object], ...]


class CameraReadinessError(RuntimeError):
    """Typed failure for structurally invalid camera producer setup."""

    def __init__(self, view: str, reason: str) -> None:
        super().__init__(f"{view} camera readiness failed: {reason}")
        self.view = view
        self.reason = reason


class TelemetryShutdownError(RuntimeError):
    """Raised when the only logger writer cannot be joined safely."""


class CameraReadinessMonitor:
    """Gate policy control while a camera pair is unavailable or recovering."""

    def __init__(
        self,
        *,
        ready_frames: int = CAMERA_READY_CONSECUTIVE_FRAMES,
        status_interval_frames: int = CAMERA_STATUS_INTERVAL_FRAMES,
    ) -> None:
        if ready_frames <= 0 or status_interval_frames <= 0:
            raise ValueError("camera readiness intervals must be positive")
        self._ready_frames = ready_frames
        self._status_interval_frames = status_interval_frames
        self._consecutive_ready = 0
        self._unavailable_frames = 0
        self._ever_ready = False
        self._status = ""
        self._last_status_frame = 0
        self._observed_frames = 0

    def observe(
        self, pair: CameraPair, producer_reason: str = ""
    ) -> CameraReadinessDecision:
        self._observed_frames += 1
        accepted = pair.accepted and not producer_reason
        if accepted:
            self._consecutive_ready += 1
            self._unavailable_frames = 0
            policy_eligible = self._consecutive_ready >= self._ready_frames
            if policy_eligible:
                self._ever_ready = True
                status = "ready"
                reason = ""
            else:
                status = (
                    "runtime_outage" if self._ever_ready else "waiting_for_camera"
                )
                reason = "confirming_advancement"
        else:
            self._consecutive_ready = 0
            self._unavailable_frames += 1
            policy_eligible = False
            status = "runtime_outage" if self._ever_ready else "waiting_for_camera"
            reason = producer_reason or pair.reason or "missing"

        transitioned = status != self._status
        periodic = (
            not policy_eligible
            and self._observed_frames - self._last_status_frame
            >= self._status_interval_frames
        )
        emit_status = transitioned or periodic
        if emit_status:
            self._last_status_frame = self._observed_frames
        self._status = status
        return CameraReadinessDecision(
            policy_eligible=policy_eligible,
            status=status,
            reason=reason,
            consecutive_ready=self._consecutive_ready,
            unavailable_frames=self._unavailable_frames,
            emit_status=emit_status,
        )


class LatestOnlyWorker:
    """Serialize keyed publications with one replaceable item per channel."""

    def __init__(self, name: str, publish: Callable[[object], None]) -> None:
        self.name = name
        self._publish = publish
        self._condition = Condition()
        self._pending: dict[str, object] = {}
        self._closed = False
        self._publishing = False
        self.submitted = 0
        self.published = 0
        self.dropped = 0
        self.failures = 0
        self.max_pending = 0
        self._thread = Thread(
            target=self._run,
            name=f"openpi-{name}-telemetry",
            daemon=True,
        )
        self._thread.start()

    def submit(self, channel: str, item: object) -> bool:
        """Replace an older pending sample without ever waiting for the sink."""

        with self._condition:
            if self._closed:
                return False
            self.submitted += 1
            if channel in self._pending:
                self.dropped += 1
            self._pending[channel] = item
            self.max_pending = max(self.max_pending, len(self._pending))
            self._condition.notify()
        return True

    def snapshot(self) -> dict[str, int | bool]:
        with self._condition:
            return {
                "submitted": self.submitted,
                "published": self.published,
                "dropped": self.dropped,
                "failures": self.failures,
                "pending": len(self._pending),
                "publishing": self._publishing,
                "max_pending": self.max_pending,
                "alive": self._thread.is_alive(),
            }

    def close(self) -> bool:
        with self._condition:
            self._closed = True
            self._condition.notify()
        self._thread.join(timeout=TELEMETRY_WORKER_JOIN_SECONDS)
        return not self._thread.is_alive()

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._closed:
                    self._condition.wait()
                if not self._pending and self._closed:
                    return
                pending = self._pending
                self._pending = {}
                self._publishing = True
            for channel in ("camera_pair", "display"):
                if channel not in pending:
                    continue
                try:
                    self._publish(pending[channel])
                except Exception as exc:  # noqa: BLE001 - telemetry cannot fail control
                    with self._condition:
                        self.failures += 1
                    print(
                        "NPA_OPENPI_TELEMETRY_ERROR "
                        f"lane={self.name} channel={channel} "
                        f"error_type={type(exc).__name__}",
                        flush=True,
                    )
                else:
                    with self._condition:
                        self.published += 1
            with self._condition:
                self._publishing = False


class DisplayRateLimiter:
    """Admit viewer snapshots at a bounded display cadence."""

    def __init__(self, rate_hz: float = TELEMETRY_DISPLAY_HZ) -> None:
        if rate_hz <= 0.0:
            raise ValueError("display rate must be positive")
        self._interval = 1.0 / rate_hz
        self._next_at = 0.0

    def due(self, now: float) -> bool:
        if now < self._next_at:
            return False
        self._next_at = now + self._interval
        return True


class LiveTelemetryPublisher:
    """Use one bounded writer for every call into the Antioch logger."""

    def __init__(self, logger) -> None:
        self._logger = logger
        # Keep initial telemetry and every later logger call serialized; this is
        # status evidence, not a camera or viewer lifecycle activation signal.
        self._logger.scalar("telemetry/initialized", 1.0)
        self._worker = LatestOnlyWorker("logger", self._publish)

    def publish_camera_pair(self, pair: CameraPair, render_sequence: int) -> tuple[str, ...]:
        """Copy and enqueue only the newest valid RGB payload for each camera."""

        import numpy as np

        publications: list[ImagePublication] = []
        for view, entity, frame in (
            ("exterior", CAMERA_EXTERIOR_ENTITY, pair.exterior),
            ("wrist", CAMERA_WRIST_ENTITY, pair.wrist),
        ):
            if frame.rgb is None:
                continue
            rgb = np.asarray(frame.rgb)
            if rgb.shape != (224, 224, 3) or rgb.dtype != np.uint8:
                print(
                    "NPA_OPENPI_IMAGE_ENQUEUE_ERROR "
                    f"view={view} render_sequence={render_sequence} reason=invalid_rgb",
                    flush=True,
                )
                continue
            publication = ImagePublication(
                view=view,
                entity=entity,
                rgb=np.ascontiguousarray(rgb).copy(),
                render_sequence=render_sequence,
                luminance_mean=frame.luminance_mean,
                luminance_variance=frame.luminance_variance,
                dynamic_range=frame.dynamic_range,
                red_cube_pixels=frame.red_cube_pixels,
            )
            publications.append(publication)
        if publications and self._worker.submit("camera_pair", tuple(publications)):
            return tuple(item.view for item in publications)
        return ()

    def publish_display(self, publication: DisplayPublication) -> bool:
        return self._worker.submit("display", publication)

    def snapshot(self) -> dict[str, dict[str, int | bool]]:
        state = self._worker.snapshot()
        # Preserve the established per-channel diagnostics while proving that
        # every channel is backed by the same serialized writer.
        return {
            "logger": state,
            "exterior": state,
            "wrist": state,
            "display": state,
        }

    def close(self) -> dict[str, bool]:
        closed = self._worker.close()
        return {"exterior": closed, "wrist": closed, "display": closed}

    def _publish(self, item: object) -> None:
        if isinstance(item, tuple):
            failure: Exception | None = None
            for publication in item:
                try:
                    self._publish_image(publication)
                except Exception as exc:  # noqa: BLE001 - publish the paired view too
                    failure = failure or exc
            if failure is not None:
                raise failure
            return
        self._publish_display(item)

    def _publish_image(self, item: object) -> None:
        publication = item
        if not isinstance(publication, ImagePublication):
            raise TypeError("invalid image publication")
        rgb = publication.rgb
        print(
            "NPA_OPENPI_IMAGE_LOG_BEGIN "
            f"view={publication.view} render_sequence={publication.render_sequence} "
            f"shape={getattr(rgb, 'shape', None)} dtype={getattr(rgb, 'dtype', None)} "
            f"contiguous={int(bool(getattr(getattr(rgb, 'flags', None), 'c_contiguous', False)))}",
            flush=True,
        )
        try:
            self._logger.image(publication.entity, rgb)
            self._logger.scalar(
                f"{publication.entity}/render_sequence", publication.render_sequence
            )
            self._logger.scalar(
                f"{publication.entity}/luminance_mean", publication.luminance_mean
            )
            self._logger.scalar(
                f"{publication.entity}/luminance_variance",
                publication.luminance_variance,
            )
            self._logger.scalar(
                f"{publication.entity}/dynamic_range", publication.dynamic_range
            )
            if publication.view == "exterior":
                self._logger.scalar(
                    f"{publication.entity}/red_cube_pixels",
                    publication.red_cube_pixels,
                )
        except Exception as exc:
            print(
                "NPA_OPENPI_IMAGE_LOG_ERROR "
                f"view={publication.view} "
                f"render_sequence={publication.render_sequence} "
                "phase=logger_image_encode_or_transport "
                f"error_type={type(exc).__name__}",
                flush=True,
            )
            raise
        print(
            "NPA_OPENPI_IMAGE_LOG_OK "
            f"view={publication.view} render_sequence={publication.render_sequence}",
            flush=True,
        )

    def _publish_display(self, item: object) -> None:
        publication = item
        if not isinstance(publication, DisplayPublication):
            raise TypeError("invalid display publication")
        print(
            "NPA_OPENPI_DISPLAY_LOG_BEGIN "
            f"render_sequence={publication.render_sequence} "
            f"numeric_groups={len(publication.numeric_groups)} "
            f"values={len(publication.values)}",
            flush=True,
        )
        for path, values in publication.numeric_groups:
            self._logger.value(path, dict(values))
        for path, value in publication.values:
            self._logger.value(path, value)
        print(
            "NPA_OPENPI_DISPLAY_LOG_OK "
            f"render_sequence={publication.render_sequence}",
            flush=True,
        )


class ActionValidationError(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class SafePolicyClient:
    """Authenticated WSS client for the persistent service-side bridge."""

    def __init__(self) -> None:
        self._connection = None
        self.reconnects = 0
        self._backoff = 1.0

    def _settings(self) -> tuple[str, str, ssl.SSLContext]:
        token = (CLIENT_ROOT / "relay-api-key").read_text().strip()
        if len(token) < 32:
            raise RuntimeError("policy relay API key is missing or malformed")
        context = ssl.create_default_context(cafile=str(CLIENT_ROOT / "relay-ca.crt"))
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        return "wss://127.0.0.1:8444", token, context

    def connect(self) -> None:
        import openpi_protocol
        from websockets.sync.client import connect

        self.close()
        uri, token, context = self._settings()
        self._connection = connect(
            uri,
            ssl=context,
            compression=None,
            max_size=32 * 1024 * 1024,
            max_queue=2,
            open_timeout=10,
            close_timeout=5,
            additional_headers={
                "Authorization": f"Api-Key {token}",
                "X-NPA-Relay-Role": "simulation",
            },
            proxy=None,
        )
        greeting = self._connection.recv(timeout=30)
        metadata = openpi_protocol.unpackb(greeting)
        if not isinstance(metadata, dict):
            raise RuntimeError("policy server greeting is malformed")
        self._backoff = 1.0

    def infer(self, request: PolicyRequest) -> tuple[dict, float, int, int]:
        import openpi_protocol

        if self._connection is None:
            self.connect()
        started = time.monotonic()
        try:
            self._connection.send(openpi_protocol.Packer().pack(request.observation))
            payload = self._connection.recv(timeout=MAX_RESPONSE_AGE_SECONDS)
            result = openpi_protocol.unpackb(payload)
        except Exception:
            self.close()
            raise
        latency = time.monotonic() - started
        if not isinstance(result, dict):
            self.close()
            raise RuntimeError("policy response is not an object")
        return (
            result,
            latency,
            request.camera_pair_id,
            request.render_sequence,
        )

    def reconnect_delay(self) -> float:
        self.reconnects += 1
        delay = self._backoff
        self._backoff = min(self._backoff * 2.0, 30.0)
        return delay

    def close(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass

    def shutdown(self) -> None:
        self.close()


def _look_at(stage, path: str, eye, target, up_hint=(0.0, 0.0, 1.0)) -> None:
    import numpy as np
    from pxr import Gf, UsdGeom

    forward = np.asarray(target, dtype=float) - np.asarray(eye, dtype=float)
    forward /= max(float(np.linalg.norm(forward)), 1e-9)
    right = np.cross(forward, np.asarray(up_hint, dtype=float))
    right /= max(float(np.linalg.norm(right)), 1e-9)
    up = np.cross(right, forward)
    matrix = Gf.Matrix4d(1.0)
    matrix.SetRow3(0, Gf.Vec3d(*right))
    matrix.SetRow3(1, Gf.Vec3d(*up))
    matrix.SetRow3(2, Gf.Vec3d(*(-forward)))
    matrix.SetTranslateOnly(Gf.Vec3d(*eye))
    transform = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    transform.ClearXformOpOrder()
    transform.AddTransformOp().Set(matrix)


def _camera_optical_config(view: str) -> dict[str, object]:
    """Return the explicit square-sensor optical contract for one policy view."""

    if view == "exterior":
        focal_length = 18.0
    elif view == "wrist":
        focal_length = 12.0
    else:
        raise ValueError(f"unknown camera view: {view}")
    return {
        "focal_length": focal_length,
        "horizontal_aperture": 36.0,
        "vertical_aperture": 36.0,
        "clipping_range": (0.01, 100.0),
        "focus_distance": 1.0,
        "f_stop": 0.0,
    }


def _configure_camera_optics(stage, path: str, view: str) -> None:
    """Apply focal length, sensor aperture, clipping, and focus explicitly."""

    from pxr import Gf, UsdGeom

    config = _camera_optical_config(view)
    camera = UsdGeom.Camera(stage.GetPrimAtPath(path))
    camera.CreateFocalLengthAttr().Set(config["focal_length"])
    camera.CreateHorizontalApertureAttr().Set(config["horizontal_aperture"])
    camera.CreateVerticalApertureAttr().Set(config["vertical_aperture"])
    camera.CreateClippingRangeAttr().Set(Gf.Vec2f(*config["clipping_range"]))
    camera.CreateFocusDistanceAttr().Set(config["focus_distance"])
    camera.CreateFStopAttr().Set(config["f_stop"])


def _world_position(stage, path: str):
    import numpy as np
    from pxr import Usd, UsdGeom

    transform = UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    return np.asarray(transform.ExtractTranslation(), dtype=np.float64)


def _stock_franka_gripper_frame(hand, left_finger, right_finger):
    """Return the measured stock-Franka grasp origin and orthonormal basis."""

    import numpy as np

    hand = np.asarray(hand, dtype=np.float64)
    left = np.asarray(left_finger, dtype=np.float64)
    right = np.asarray(right_finger, dtype=np.float64)
    grasp = 0.5 * (left + right)
    forward = grasp - hand
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm <= 1e-9:
        raise ValueError("stock Franka hand-to-fingertip axis is degenerate")
    forward /= forward_norm
    side = right - left
    side -= float(np.dot(side, forward)) * forward
    side_norm = float(np.linalg.norm(side))
    if side_norm <= 1e-9:
        raise ValueError("stock Franka fingertip axis is degenerate")
    side /= side_norm
    up = np.cross(forward, side)
    up /= max(float(np.linalg.norm(up)), 1e-9)
    side = np.cross(up, forward)
    basis = np.column_stack((forward, side, up))
    return grasp, basis


def _calibrate_wrist_camera_mount(hand, left_finger, right_finger, look_at):
    """Freeze a cube-framing camera pose in the measured gripper coordinates."""

    import numpy as np

    grasp, basis = _stock_franka_gripper_frame(hand, left_finger, right_finger)
    eye_offset = np.asarray(WRIST_EYE_OFFSET_TOOL, dtype=np.float64)
    eye = grasp + basis @ eye_offset
    look = np.asarray(look_at, dtype=np.float64) - eye
    look /= max(float(np.linalg.norm(look)), 1e-9)
    return WristCameraMount(
        eye_offset_tool=eye_offset,
        look_direction_tool=basis.T @ look,
        up_direction_tool=basis.T @ np.asarray([0.0, 0.0, 1.0]),
    )


def _wrist_camera_pose_from_points(hand, left_finger, right_finger, mount=None):
    """Resolve fixed camera extrinsics in the measured stock-Franka tool frame."""

    import numpy as np

    grasp, basis = _stock_franka_gripper_frame(hand, left_finger, right_finger)
    if mount is None:
        mount = WristCameraMount(
            eye_offset_tool=np.asarray(WRIST_EYE_OFFSET_TOOL, dtype=np.float64),
            look_direction_tool=np.asarray(WRIST_TARGET_OFFSET_TOOL, dtype=np.float64),
            up_direction_tool=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        )
    eye = grasp + basis @ np.asarray(mount.eye_offset_tool, dtype=np.float64)
    target = eye + basis @ np.asarray(mount.look_direction_tool, dtype=np.float64)
    up = basis @ np.asarray(mount.up_direction_tool, dtype=np.float64)
    return eye, target, up


def _stock_franka_camera_points(stage):
    return (
        _world_position(stage, STOCK_FRANKA_HAND_PATH),
        _world_position(stage, STOCK_FRANKA_LEFT_FINGER_PATH),
        _world_position(stage, STOCK_FRANKA_RIGHT_FINGER_PATH),
    )


def _aim_wrist_camera(stage, mount):
    pose = _wrist_camera_pose_from_points(
        *_stock_franka_camera_points(stage),
        mount,
    )
    _look_at(stage, WRIST_CAMERA_PATH, *pose)
    return pose


def _point_in_camera_frame(point, pose, optical_config, *, margin: float = 0.92) -> bool:
    """Geometrically prove a known scene point lies inside a camera frustum."""

    import numpy as np

    eye, target, up_hint = (np.asarray(value, dtype=np.float64) for value in pose)
    forward = target - eye
    forward /= max(float(np.linalg.norm(forward)), 1e-9)
    right = np.cross(forward, up_hint)
    right /= max(float(np.linalg.norm(right)), 1e-9)
    up = np.cross(right, forward)
    relative = np.asarray(point, dtype=np.float64) - eye
    depth = float(np.dot(relative, forward))
    if depth <= 0.0:
        return False
    focal = float(optical_config["focal_length"])
    half_horizontal = math.atan(float(optical_config["horizontal_aperture"]) / (2 * focal))
    half_vertical = math.atan(float(optical_config["vertical_aperture"]) / (2 * focal))
    horizontal = abs(float(np.dot(relative, right)) / depth)
    vertical = abs(float(np.dot(relative, up)) / depth)
    return bool(
        horizontal <= math.tan(half_horizontal) * margin
        and vertical <= math.tan(half_vertical) * margin
    )


def _configure_lighting(stage) -> None:
    """Create deterministic fill and key lights for both policy cameras."""

    from pxr import Gf, UsdGeom, UsdLux

    dome = UsdLux.DomeLight.Define(stage, "/World/PolicyFillLight")
    dome.CreateIntensityAttr(900.0)
    dome.CreateExposureAttr(1.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.95))

    key = UsdLux.DistantLight.Define(stage, "/World/PolicyKeyLight")
    key.CreateIntensityAttr(2_500.0)
    key.CreateExposureAttr(1.0)
    key.CreateAngleAttr(4.0)
    key.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9))
    transform = UsdGeom.Xformable(key.GetPrim())
    transform.AddRotateXYZOp().Set(Gf.Vec3f(-35.0, -25.0, -35.0))


def _contact_force_magnitude(contact_view, physics_dt: float) -> float:
    """Read gripper/cube contact from Isaac's tracked-contact view."""

    import numpy as np

    forces = contact_view.get_contact_force_matrix(dt=physics_dt)
    if forces is None:
        return 0.0
    if hasattr(forces, "detach"):
        forces = forces.detach().cpu().numpy()
    elif hasattr(forces, "numpy"):
        forces = forces.numpy()
    array = np.asarray(forces, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        return 0.0
    return float(np.linalg.norm(array, axis=-1).max(initial=0.0))


def _droid_gripper_observation(joint_positions) -> float:
    """Map Isaac finger opening to DROID's 0=open, 1=closed convention."""

    import numpy as np

    fingers = np.asarray(joint_positions, dtype=np.float64)[7:9]
    if fingers.shape != (2,) or not np.isfinite(fingers).all():
        raise ActionValidationError("non_finite")
    opening_width = float(np.clip(fingers.sum(), 0.0, GRIPPER_TOTAL_WIDTH_MAX))
    return 1.0 - opening_width / GRIPPER_TOTAL_WIDTH_MAX


def _isaac_finger_target(droid_gripper_target: float) -> float:
    """Map DROID gripper position to one Isaac finger joint target."""

    # DROID and the pinned OpenPI policy use 0=open and 1=closed.  The stock
    # Franka articulation is the inverse: each finger is 0.04 m open and 0 m
    # closed.  Upstream's deployment example binarizes the model target.
    closed = float(droid_gripper_target) > 0.5
    return 0.0 if closed else GRIPPER_JOINT_MAX


def _validated_actions(response: dict, current) -> tuple[object, dict[str, int]]:
    import numpy as np

    actions = np.asarray(response.get("actions"))
    if actions.shape != ACTION_SHAPE:
        raise ActionValidationError("wrong_shape")
    if not np.issubdtype(actions.dtype, np.number) or not np.isfinite(actions).all():
        raise ActionValidationError("non_finite")
    targets = actions.astype(np.float64, copy=True)
    low, high = np.asarray(JOINT_LOW), np.asarray(JOINT_HIGH)
    raw_joint_limit_mismatches = int(
        np.count_nonzero((targets[:, :7] < low) | (targets[:, :7] > high))
    )
    targets[:, :7] = np.clip(targets[:, :7], low, high)
    # The pinned Polaris data configuration trains absolute joint positions and
    # DROID gripper position. DroidOutputs intentionally returns both unchanged.
    # Preserve raw distribution evidence, but use upstream's documented binary
    # gripper mapping instead of treating an unbounded model value as finger
    # opening in metres.
    raw_gripper_range_mismatches = int(
        np.count_nonzero((targets[:, 7] < 0.0) | (targets[:, 7] > 1.0))
    )
    targets[:, 7] = (targets[:, 7] > 0.5).astype(np.float64)
    prior = np.asarray(current[:7], dtype=np.float64)
    joint_step_projections = 0
    for index in range(ACTION_SHAPE[0]):
        bounded = np.clip(
            targets[index, :7], prior - MAX_JOINT_STEP, prior + MAX_JOINT_STEP
        )
        joint_step_projections += int(np.count_nonzero(bounded != targets[index, :7]))
        targets[index, :7] = bounded
        prior = bounded
    return targets, {
        "raw_gripper_range_mismatches": raw_gripper_range_mismatches,
        "raw_joint_limit_mismatches": raw_joint_limit_mismatches,
        "joint_limit_projections": raw_joint_limit_mismatches,
        "joint_step_projections": joint_step_projections,
    }


def _install_overlay():
    import omni.ui as ui

    window = ui.Window("NPA OpenPI live decisions", width=520, height=250)
    with window.frame:
        with ui.VStack(spacing=4):
            title = ui.Label("LIVE camera → pi0.5 DROID → Franka", height=28)
            state = ui.Label("initializing", word_wrap=True)
            counters = ui.Label("", word_wrap=True)
            latency = ui.Label("", word_wrap=True)
    return window, title, state, counters, latency


def _camera_frame_from_buffer(buffer, *, view: str) -> CameraFrame:
    """Copy and classify one numpy/Warp RGB buffer without unsafe aliasing."""

    import numpy as np

    if buffer is None:
        return CameraFrame(None, "missing")
    try:
        source = buffer.numpy() if callable(getattr(buffer, "numpy", None)) else buffer
        frame = np.array(source, copy=True)
    except Exception:
        return CameraFrame(None, "unreadable")
    if frame.ndim != 3 or frame.shape not in ((224, 224, 3), (224, 224, 4)):
        return CameraFrame(None, "wrong_shape")
    if not np.issubdtype(frame.dtype, np.number) or not np.isfinite(frame).all():
        return CameraFrame(None, "non_finite")
    raw_min = float(frame.min())
    raw_max = float(frame.max())
    raw_nonzero = int(np.count_nonzero(frame))
    raw_channels = int(frame.shape[2])
    rgb_source = frame[:, :, :3]
    if np.issubdtype(rgb_source.dtype, np.floating):
        upper = float(rgb_source.max())
        if upper <= 1.0:
            rgb_source = rgb_source * 255.0
    rgb = np.ascontiguousarray(
        np.clip(rgb_source, 0, 255).astype(np.uint8, copy=False)
    )
    luminance = np.mean(rgb, axis=2)
    luminance_mean = float(luminance.mean())
    luminance_variance = float(luminance.var())
    dynamic_range = float(np.percentile(luminance, 95) - np.percentile(luminance, 5))
    red_mask = (
        (rgb[..., 0] > 80)
        & (rgb[..., 0].astype(np.float32) > rgb[..., 1] * 1.35)
        & (rgb[..., 0].astype(np.float32) > rgb[..., 2] * 1.35)
    )
    result = CameraFrame(
        rgb,
        "",
        luminance_mean,
        luminance_variance,
        dynamic_range,
        int(red_mask.sum()),
        raw_min,
        raw_max,
        raw_nonzero,
        raw_channels,
    )
    if luminance_mean <= MIN_CAMERA_LUMINANCE_MEAN:
        return CameraFrame(
            rgb,
            "blank",
            luminance_mean,
            luminance_variance,
            dynamic_range,
            result.red_cube_pixels,
            raw_min,
            raw_max,
            raw_nonzero,
            raw_channels,
        )
    if luminance_variance <= MIN_CAMERA_LUMINANCE_VARIANCE:
        return CameraFrame(
            rgb,
            "flat",
            luminance_mean,
            luminance_variance,
            dynamic_range,
            result.red_cube_pixels,
            raw_min,
            raw_max,
            raw_nonzero,
            raw_channels,
        )
    if view == "exterior" and result.red_cube_pixels < MIN_EXTERIOR_RED_CUBE_PIXELS:
        return CameraFrame(
            rgb,
            "cube_not_visible",
            luminance_mean,
            luminance_variance,
            dynamic_range,
            result.red_cube_pixels,
            raw_min,
            raw_max,
            raw_nonzero,
            raw_channels,
        )
    return result


def _camera_frame(camera, *, view: str) -> CameraFrame:
    """Read RGB through the supported CameraSensor API for focused helpers/tests."""

    data, _info = camera.get_data("rgb")
    return _camera_frame_from_buffer(data, view=view)


def _producer_marker_from_info(info) -> tuple[int, int] | None:
    """Extract an exact producer timestamp from public annotator metadata."""

    if not isinstance(info, dict):
        return None
    candidates = [info]
    candidates.extend(value for value in info.values() if isinstance(value, dict))
    for candidate in candidates:
        numerator = candidate.get("referenceTimeNumerator")
        denominator = candidate.get("referenceTimeDenominator")
        if denominator is None:
            for key in ("renderingFrame", "frameNumber", "frame_id"):
                if key in candidate:
                    numerator, denominator = candidate[key], 1
                    break
        if isinstance(numerator, bool) or isinstance(denominator, bool):
            continue
        try:
            exact_numerator = operator.index(numerator)
            exact_denominator = operator.index(denominator)
        except TypeError:
            continue
        if exact_denominator > 0:
            return exact_numerator, exact_denominator
    return None


def _new_reference_time_annotator(render_product: str):
    """Attach Isaac's public producer clock directly to one render product."""

    import omni.replicator.core as rep

    annotator = rep.AnnotatorRegistry.get_annotator("ReferenceTime")
    annotator.attach([render_product])
    return annotator


def _reference_time_marker(annotator) -> tuple[int, int] | None:
    """Return the render product's exact producer-owned reference time."""

    current = annotator.get_data()
    if not isinstance(current, dict):
        return None
    raw_numerator = current.get("referenceTimeNumerator")
    raw_denominator = current.get("referenceTimeDenominator")
    if isinstance(raw_numerator, bool) or isinstance(raw_denominator, bool):
        return None
    try:
        numerator = operator.index(raw_numerator)
        denominator = operator.index(raw_denominator)
    except TypeError:
        return None
    if denominator <= 0:
        return None
    return numerator, denominator


def _reference_time_advanced(
    current: tuple[int, int], prior: tuple[int, int]
) -> bool:
    """Compare exact reference-time fractions without float precision loss."""

    return current[0] * prior[1] > prior[0] * current[1]


def _build_rtx_rgb_camera(
    RtxCamera, CameraSensor, *, path: str, position=None, output_buffer=None
):
    """Construct one supported Isaac Sim 6 RTX camera and explicit RGB sensor."""

    kwargs = {"tick_rate": CAMERA_SENSOR_TICK_RATE_HZ}
    if position is not None:
        kwargs["positions"] = [position]
    authoring = RtxCamera(path, **kwargs)
    sensor = CameraSensor(authoring, resolution=(224, 224), annotators=["rgb"])
    if output_buffer is None:
        import warp as wp

        output_buffer = wp.empty((224, 224, 3), dtype=wp.uint8, device="cpu")
    render_product = sensor.render_product
    prim = render_product.GetPrim()
    render_product_path = str(prim.GetPath()) if prim and prim.IsValid() else ""
    if not render_product_path:
        with contextlib.suppress(Exception):
            sensor.detach_annotators("rgb")
        raise CameraReadinessError(path, "render_product_invalid")
    return RtxRgbCamera(
        authoring,
        sensor,
        _new_reference_time_annotator(render_product_path),
        output_buffer,
    )


def _capture_camera_samples(cameras) -> dict[str, CameraSample]:
    """Read both products after the caller's completed rendered world step."""

    return {view: camera.sample(view=view) for view, camera in cameras}


def _start_camera_timeline(app_utils) -> None:
    """Commit timeline play before the first rendered sensor update."""

    app_utils.play(commit=True)


def _initialize_live_capture(world, cameras, logger):
    """Create serialized telemetry and structurally valid camera producers."""
    telemetry = LiveTelemetryPublisher(logger)
    try:
        render_products = {}
        for view, camera in cameras:
            render_product = camera.render_product_path
            prim = world.stage.GetPrimAtPath(render_product)
            if not render_product or not prim or not prim.IsValid():
                raise CameraReadinessError(view, "render_product_invalid")
            render_products[view] = render_product
    except BaseException:
        telemetry.close()
        for _view, camera in cameras:
            with contextlib.suppress(Exception):
                camera.close()
        raise
    return telemetry, render_products, {view: None for view, _ in cameras}


def _camera_markers_advanced(
    samples: dict[str, CameraSample],
    prior: dict[str, tuple[int, int] | None],
) -> tuple[bool, dict[str, tuple[int, int] | None], str]:
    current = {view: sample.producer_marker for view, sample in samples.items()}
    for view, marker in current.items():
        if marker is None:
            return False, current, f"{view}_producer_marker_missing"
        if prior.get(view) is not None and not _reference_time_advanced(
            marker, prior[view]
        ):
            return False, current, f"{view}_producer_stale"
    return True, current, ""


def _validate_camera_pair(
    exterior_frame: CameraFrame,
    wrist_frame: CameraFrame,
    *,
    render_sequence: int,
    last_accepted_render_sequence: int,
    exterior_cube_in_frame: bool,
    wrist_cube_in_frame: bool,
) -> CameraPair:
    """Fail closed unless both current views are useful, fresh, and distinct."""

    import numpy as np

    if not isinstance(exterior_frame, CameraFrame):
        exterior_frame = _camera_frame(exterior_frame, view="exterior")
    if not isinstance(wrist_frame, CameraFrame):
        wrist_frame = _camera_frame(wrist_frame, view="wrist")

    difference = 0.0
    if exterior_frame.rgb is not None and wrist_frame.rgb is not None:
        difference = float(
            np.abs(
                np.asarray(exterior_frame.rgb, dtype=np.float32)
                - np.asarray(wrist_frame.rgb, dtype=np.float32)
            ).mean()
        )
    if exterior_frame.reason:
        return CameraPair(
            False,
            exterior_frame,
            wrist_frame,
            "exterior",
            exterior_frame.reason,
            difference,
        )
    if wrist_frame.reason:
        return CameraPair(
            False,
            exterior_frame,
            wrist_frame,
            "wrist",
            wrist_frame.reason,
            difference,
        )
    if render_sequence <= last_accepted_render_sequence:
        return CameraPair(False, exterior_frame, wrist_frame, "pair", "stale")
    if not exterior_cube_in_frame:
        return CameraPair(False, exterior_frame, wrist_frame, "exterior", "cube_out_of_frame")
    if not wrist_cube_in_frame:
        return CameraPair(False, exterior_frame, wrist_frame, "wrist", "cube_out_of_frame")
    if difference < MIN_CAMERA_PAIR_DIFFERENCE:
        return CameraPair(
            False,
            exterior_frame,
            wrist_frame,
            "pair",
            "not_distinct",
            difference,
        )
    return CameraPair(True, exterior_frame, wrist_frame, mean_difference=difference)


def _build_policy_observation(exterior_rgb, wrist_rgb, joint_positions, prompt: str):
    """Map the accepted camera pair to the exact pinned DROID observation keys."""

    import numpy as np

    return {
        "observation/exterior_image_1_left": exterior_rgb,
        "observation/wrist_image_left": wrist_rgb,
        "observation/joint_position": joint_positions[:7],
        "observation/gripper_position": np.asarray(
            [_droid_gripper_observation(joint_positions)], dtype=np.float32
        ),
        "prompt": prompt,
    }


def _camera_blueprint(rrb):
    """Bind every authored view to the Logger-resolved Rerun entity catalog."""

    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial2DView(
                    origin=_resolved_telemetry_entity(CAMERA_EXTERIOR_ENTITY),
                    name="Exterior policy input",
                ),
                rrb.Spatial2DView(
                    origin=_resolved_telemetry_entity(CAMERA_WRIST_ENTITY),
                    name="Wrist policy input",
                ),
                column_shares=[1.0, 1.0],
            ),
            rrb.Horizontal(
                rrb.Spatial3DView(
                    origin=_resolved_telemetry_entity(SCENE_ENTITY),
                    name="Franka scene",
                ),
                rrb.Tabs(
                    rrb.TimeSeriesView(
                        origin=_resolved_telemetry_entity(DECISION_METRICS_ENTITY),
                        name="Policy decisions",
                    ),
                    rrb.TimeSeriesView(
                        origin=_resolved_telemetry_entity(GRASP_METRICS_ENTITY),
                        name="Physical pickup evidence",
                    ),
                    rrb.TimeSeriesView(
                        origin=_resolved_telemetry_entity(FRANKA_ACTION_ENTITY),
                        name="Franka joint targets",
                    ),
                    rrb.TimeSeriesView(
                        origin=_resolved_telemetry_entity(CAMERA_METRICS_ENTITY),
                        name="Camera quality",
                    ),
                    rrb.TextLogView(
                        origin=_resolved_telemetry_entity(POLICY_ERROR_ENTITY),
                        name="Policy errors",
                    ),
                    name="Telemetry",
                ),
                column_shares=[1.0, 1.0],
            ),
            row_shares=[1.0, 1.0],
        ),
        auto_layout=False,
    )


def _log_camera_pair_for_rerun(logger, pair: CameraPair) -> tuple[str, ...]:
    """Publish structurally valid pixels even when policy quality rejects them."""

    logged: list[str] = []
    for view, entity, frame in (
        ("exterior", CAMERA_EXTERIOR_ENTITY, pair.exterior),
        ("wrist", CAMERA_WRIST_ENTITY, pair.wrist),
    ):
        if frame.rgb is None:
            continue
        logger.image(entity, frame.rgb)
        logger.scalar(f"{entity}/luminance_mean", frame.luminance_mean)
        logger.scalar(f"{entity}/luminance_variance", frame.luminance_variance)
        logger.scalar(f"{entity}/dynamic_range", frame.dynamic_range)
        if view == "exterior":
            logger.scalar(
                f"{CAMERA_EXTERIOR_ENTITY}/red_cube_pixels",
                frame.red_cube_pixels,
            )
        logged.append(view)
    if logged:
        logger.scalar(f"{CAMERA_METRICS_ENTITY}/pair_mean_difference", pair.mean_difference)
        logger.scalar(
            f"{CAMERA_METRICS_ENTITY}/policy_quality_accepted", int(pair.accepted)
        )
    return tuple(logged)


def _camera_rejection_metrics_line(
    *,
    elapsed_seconds: float,
    frames: int,
    requests: int,
    round_trips: int,
    applied: int,
    reconnects: int,
    camera_rejected_pairs: int,
    camera_validated_requests: int,
    camera_pair_id: int,
    request_camera_pair_id: int,
    round_trip_camera_pair_id: int,
    render_sequence: int,
    request_render_sequence: int,
    round_trip_render_sequence: int,
    exterior_cube_in_frame: bool,
    wrist_cube_in_frame: bool,
    pair: CameraPair,
) -> str:
    """Expose current rejected-frame evidence before any policy round trip."""

    return (
        "NPA_OPENPI_METRICS "
        f"elapsed_seconds={elapsed_seconds:.3f} "
        f"frames={frames} requests={requests} round_trips={round_trips} "
        f"applied={applied} reconnects={reconnects} "
        "camera_quality_schema=3 "
        f"camera_rejected_pairs={camera_rejected_pairs} "
        f"camera_validated_requests={camera_validated_requests} "
        f"camera_pair_id={camera_pair_id} "
        f"request_camera_pair_id={request_camera_pair_id} "
        f"round_trip_camera_pair_id={round_trip_camera_pair_id} "
        f"camera_render_sequence={render_sequence} "
        f"request_render_sequence={request_render_sequence} "
        f"round_trip_render_sequence={round_trip_render_sequence} "
        f"camera_pair_difference_current={pair.mean_difference:.3f} "
        f"camera_exterior_red_cube_pixels_current={pair.exterior.red_cube_pixels} "
        f"camera_exterior_cube_in_frame_current={int(exterior_cube_in_frame)} "
        f"camera_wrist_cube_in_frame_current={int(wrist_cube_in_frame)} "
        f"camera_exterior_luminance_mean_current={pair.exterior.luminance_mean:.3f} "
        f"camera_exterior_luminance_variance_current={pair.exterior.luminance_variance:.3f} "
        f"camera_exterior_dynamic_range_current={pair.exterior.dynamic_range:.3f} "
        f"camera_exterior_raw_min_current={pair.exterior.raw_min:.3f} "
        f"camera_exterior_raw_max_current={pair.exterior.raw_max:.3f} "
        f"camera_exterior_raw_nonzero_current={pair.exterior.raw_nonzero} "
        f"camera_exterior_raw_channels_current={pair.exterior.raw_channels} "
        f"camera_wrist_luminance_mean_current={pair.wrist.luminance_mean:.3f} "
        f"camera_wrist_luminance_variance_current={pair.wrist.luminance_variance:.3f} "
        f"camera_wrist_dynamic_range_current={pair.wrist.dynamic_range:.3f} "
        f"camera_wrist_raw_min_current={pair.wrist.raw_min:.3f} "
        f"camera_wrist_raw_max_current={pair.wrist.raw_max:.3f} "
        f"camera_wrist_raw_nonzero_current={pair.wrist.raw_nonzero} "
        f"camera_wrist_raw_channels_current={pair.wrist.raw_channels}"
    )


def _franka_link_points(stage):
    """Read the rendered Franka link transforms for live Rerun geometry."""

    from pxr import Usd, UsdGeom

    points = []
    for name in [*(f"panda_link{index}" for index in range(8)), "panda_hand"]:
        prim = stage.GetPrimAtPath(f"/World/Franka/{name}")
        if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Xformable):
            continue
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        points.append(list(transform.ExtractTranslation()))
    return points


def _unit_vector(vector):
    length = math.sqrt(sum(component * component for component in vector))
    if length <= 1e-9:
        return None
    return [component / length for component in vector]


def _z_axis_quaternion(direction):
    """Return an xyzw quaternion that rotates local +Z onto direction."""

    unit = _unit_vector(direction)
    if unit is None:
        raise ValueError("cannot orient geometry along a zero-length vector")
    dot = max(-1.0, min(1.0, unit[2]))
    if dot < -1.0 + 1e-8:
        return [1.0, 0.0, 0.0, 0.0]
    quaternion = [-unit[1], unit[0], 0.0, 1.0 + dot]
    norm = math.sqrt(sum(component * component for component in quaternion))
    return [component / norm for component in quaternion]


def _franka_proxy_geometry(link_points):
    """Build a volumetric Franka proxy from live USD link translations.

    The proxy intentionally contains only generated primitives. The Isaac/Franka
    mesh remains inside the operator-accepted runtime and is visible through the
    two rendered camera streams, while Rerun gets recognizable moving geometry
    without copying any simulator asset bytes.
    """

    points = [[float(component) for component in point] for point in link_points]
    links = {"centers": [], "sizes": [], "quaternions": [], "colors": []}
    last_direction = None
    for index, (start, end) in enumerate(zip(points, points[1:])):
        direction = [right - left for left, right in zip(start, end)]
        unit = _unit_vector(direction)
        if unit is None:
            continue
        length = math.sqrt(sum(component * component for component in direction))
        width = 0.105 if index < 2 else 0.085 if index < 5 else 0.065
        links["centers"].append(
            [(left + right) / 2.0 for left, right in zip(start, end)]
        )
        links["sizes"].append([width, width, length])
        links["quaternions"].append(_z_axis_quaternion(direction))
        links["colors"].append(
            [230, 232, 235, 255] if index % 2 == 0 else [195, 200, 206, 255]
        )
        last_direction = unit

    if not points:
        return {"base": None, "links": links, "joints": None, "gripper": None}

    base = {
        "centers": [[points[0][0], points[0][1], points[0][2] - 0.055]],
        "sizes": [[0.20, 0.20, 0.11]],
        "colors": [[58, 63, 70, 255]],
    }
    joints = {
        "centers": points,
        "radii": [0.057 if index < 2 else 0.046 for index in range(len(points))],
        "colors": [[40, 44, 52, 255]] * len(points),
    }
    if last_direction is None:
        return {"base": base, "links": links, "joints": joints, "gripper": None}

    up = [0.0, 0.0, 1.0]
    lateral = _unit_vector(
        [
            last_direction[1] * up[2] - last_direction[2] * up[1],
            last_direction[2] * up[0] - last_direction[0] * up[2],
            last_direction[0] * up[1] - last_direction[1] * up[0],
        ]
    )
    if lateral is None:
        lateral = [1.0, 0.0, 0.0]
    hand = points[-1]
    orientation = _z_axis_quaternion(last_direction)
    palm_center = [value + 0.025 * axis for value, axis in zip(hand, last_direction)]
    finger_centers = []
    for sign in (-1.0, 1.0):
        finger_centers.append(
            [
                value + 0.10 * axis + sign * 0.047 * side
                for value, axis, side in zip(hand, last_direction, lateral)
            ]
        )
    gripper = {
        "centers": [palm_center, *finger_centers],
        "sizes": [[0.14, 0.09, 0.055], [0.024, 0.024, 0.15], [0.024, 0.024, 0.15]],
        "quaternions": [orientation, orientation, orientation],
        "colors": [[58, 63, 70, 255], [34, 39, 48, 255], [34, 39, 48, 255]],
    }
    return {"base": base, "links": links, "joints": joints, "gripper": gripper}


@antioch.scenario(tags=["openpi-live", "mk8s-native"])
def openpi_franka_mk8s_live_v2(
    run: antioch.ScenarioRun,
    prompt: str = antioch.param(
        "pick up the red cube", description="DROID task prompt"
    ),
) -> None:
    """Continuously render, infer, validate and apply pi0.5 action chunks."""

    import numpy as np
    import rerun as rr
    import rerun.blueprint as rrb
    import isaacsim.core.experimental.utils.app as app_utils
    from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
    from isaacsim.core.prims import RigidPrim
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.core.utils.viewports import set_camera_view
    from isaacsim.core.utils.extensions import enable_extension

    # Isaac Sim 6 keeps the legacy Franka helper as an opt-in extension.
    enable_extension("isaacsim.robot.manipulators.examples")
    from isaacsim.robot.manipulators.examples.franka import Franka
    from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera

    world = antioch.world()
    world.scene.add_ground_plane(z_position=-0.75)
    tabletop = world.scene.add(
        FixedCuboid(
            prim_path="/World/Tabletop",
            name="tabletop",
            position=np.array([0.48, 0.0, -0.04]),
            scale=np.array([0.9, 0.7, 0.08]),
            color=np.array([0.55, 0.36, 0.2]),
        )
    )
    for index, (x, y) in enumerate(
        ((0.13, -0.25), (0.83, -0.25), (0.13, 0.25), (0.83, 0.25))
    ):
        world.scene.add(
            FixedCuboid(
                prim_path=f"/World/TableLeg{index}",
                name=f"table_leg_{index}",
                position=np.array([x, y, -0.39]),
                scale=np.array([0.07, 0.07, 0.7]),
                color=np.array([0.32, 0.22, 0.14]),
            )
        )
    robot = world.scene.add(Franka(prim_path="/World/Franka", name="franka"))
    cube = world.scene.add(
        DynamicCuboid(
            prim_path="/World/Cube",
            name="cube",
            position=np.array(CUBE_INITIAL_POSITION),
            size=CUBE_SIZE_METERS,
            color=np.array([0.95, 0.03, 0.02]),
        )
    )
    cube_gripper_contacts = world.scene.add(
        RigidPrim(
            prim_paths_expr="/World/Cube",
            name="cube_gripper_contacts",
            track_contact_forces=True,
            contact_filter_prim_paths_expr=[
                "/World/Franka/panda_leftfinger",
                "/World/Franka/panda_rightfinger",
            ],
            max_contact_count=16,
        )
    )
    exterior = _build_rtx_rgb_camera(
        RtxCamera,
        CameraSensor,
        path=EXTERIOR_CAMERA_PATH,
        position=EXTERIOR_CAMERA_EYE,
    )
    wrist = _build_rtx_rgb_camera(
        RtxCamera,
        CameraSensor,
        path=WRIST_CAMERA_PATH,
    )
    _configure_lighting(world.stage)
    world.reset()
    robot.set_joint_positions(
        np.asarray([*DROID_RESET_JOINTS, GRIPPER_JOINT_MAX, GRIPPER_JOINT_MAX])
    )
    _look_at(
        world.stage,
        EXTERIOR_CAMERA_PATH,
        EXTERIOR_CAMERA_EYE,
        EXTERIOR_CAMERA_TARGET,
    )
    wrist_mount = _calibrate_wrist_camera_mount(
        *_stock_franka_camera_points(world.stage),
        cube.get_world_pose()[0],
    )
    wrist_pose = _aim_wrist_camera(world.stage, wrist_mount)
    _configure_camera_optics(world.stage, EXTERIOR_CAMERA_PATH, "exterior")
    _configure_camera_optics(world.stage, WRIST_CAMERA_PATH, "wrist")
    _start_camera_timeline(app_utils)
    cameras = (("exterior", exterior), ("wrist", wrist))
    telemetry, render_products, camera_markers = _initialize_live_capture(
        world, cameras, logger
    )
    print(
        "NPA_OPENPI_CAMERAS_ATTACHED "
        f"views={','.join(sorted(render_products))} safe_hold=1",
        flush=True,
    )
    set_camera_view(
        eye=[1.35, -1.15, 0.9],
        target=[0.44, 0.02, 0.1],
        camera_prim_path="/OmniverseKit_Persp",
    )
    overlay = _install_overlay()
    rr.send_blueprint(_camera_blueprint(rrb))
    client = SafePolicyClient()
    observation_sequence = requests = round_trips = applied = safe_holds = 0
    camera_rejected_pairs = camera_validated_requests = 0
    rejected_actions: Counter[str] = Counter()
    transport_failures: Counter[str] = Counter()
    latencies_ms: list[float] = []
    luminance_means: list[float] = []
    luminance_variances: list[float] = []
    camera_rejections: Counter[str] = Counter()
    raw_gripper_range_mismatches = 0
    raw_joint_limit_mismatches = 0
    joint_limit_projections = 0
    joint_step_projections = 0
    last_latency = 0.0
    started = time.monotonic()
    next_attempt = 0.0
    chunk = None
    chunk_index = 0
    last_apply = time.monotonic()
    first_frame = True
    current_luminance_mean_min = 0.0
    current_luminance_variance_min = 0.0
    current_exterior_luminance_mean = 0.0
    current_exterior_luminance_variance = 0.0
    current_wrist_luminance_mean = 0.0
    current_wrist_luminance_variance = 0.0
    current_exterior_dynamic_range = 0.0
    current_wrist_dynamic_range = 0.0
    camera_pair_id = request_camera_pair_id = round_trip_camera_pair_id = 0
    render_sequence = request_render_sequence = round_trip_render_sequence = 0
    last_accepted_render_sequence = 0
    current_camera_pair_difference = 0.0
    current_exterior_red_cube_pixels = 0
    current_exterior_cube_in_frame = 0
    current_wrist_cube_in_frame = 0
    current_camera_quality_accepted = 0
    pending = None
    pending_observation = 0
    pending_camera_pair_id = 0
    pending_joint_positions = None
    pending_started_at = 0.0
    pending_stall_reported = False
    current_policy_error = ""
    cube_initial_height = float(cube.get_world_pose()[0][2])
    physics_dt = float(world.get_physics_dt())
    initial_ee_distance = None
    minimum_ee_distance = float("inf")
    maximum_cube_lift = 0.0
    gripper_contact_samples = 0
    maximum_gripper_contact_force = 0.0
    pickup_hold_started = None
    pickup_hold_seconds = 0.0
    pickup_success = False
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="openpi-policy")
    camera_readiness = CameraReadinessMonitor()
    camera_policy_eligible = False
    display_rate = DisplayRateLimiter()
    next_loop_heartbeat_at = 0.0
    next_camera_attempt = 0.0

    print("NPA_OPENPI_LOOP_READY", flush=True)
    try:
        while True:
            wrist_pose = _aim_wrist_camera(world.stage, wrist_mount)
            world.step(render=True)
            render_sequence += 1
            now = time.monotonic()

            if pending is not None and not pending.done():
                pending_age = now - pending_started_at
                if (
                    pending_age > MAX_RESPONSE_AGE_SECONDS
                    and not pending_stall_reported
                ):
                    pending_stall_reported = True
                    safe_holds += 1
                    transport_failures["policy_future_timeout"] += 1
                    client.close()
                    pending.cancel()
                    current_policy_error = "policy_future_timeout"
                    print(
                        "NPA_OPENPI_POLICY_STALL "
                        f"observation={pending_observation} "
                        f"age_seconds={pending_age:.3f} safe_hold=1",
                        flush=True,
                    )

            if pending is not None and pending.done():
                try:
                    (
                        response,
                        last_latency,
                        response_camera_pair_id,
                        response_render_sequence,
                    ) = pending.result()
                    if (
                        response_camera_pair_id != pending_camera_pair_id
                        or response_render_sequence != request_render_sequence
                    ):
                        raise RuntimeError("policy response camera-pair identity mismatch")
                    if last_latency > MAX_RESPONSE_AGE_SECONDS:
                        raise TimeoutError("policy response was stale")
                    chunk, action_evidence = _validated_actions(
                        response, pending_joint_positions
                    )
                    raw_gripper_range_mismatches += action_evidence[
                        "raw_gripper_range_mismatches"
                    ]
                    raw_joint_limit_mismatches += action_evidence[
                        "raw_joint_limit_mismatches"
                    ]
                    joint_limit_projections += action_evidence[
                        "joint_limit_projections"
                    ]
                    joint_step_projections += action_evidence["joint_step_projections"]
                    current_policy_error = ""
                    chunk_index = 0
                    round_trips += 1
                    round_trip_camera_pair_id = pending_camera_pair_id
                    round_trip_render_sequence = response_render_sequence
                    latencies_ms.append(last_latency * 1000.0)
                    percentiles = np.percentile(latencies_ms, [50, 95, 99])
                    print(
                        "NPA_OPENPI_ROUND_TRIP "
                        f"observation={pending_observation} "
                        f"round_trips={round_trips} "
                        f"latency_ms={last_latency * 1000.0:.3f} "
                        "action_shape=[15,8] finite=true safety_validated=true "
                        f"camera_pair_id={round_trip_camera_pair_id} "
                        f"render_sequence={round_trip_render_sequence} "
                        f"raw_gripper_range_mismatches={raw_gripper_range_mismatches} "
                        f"raw_joint_limit_mismatches={raw_joint_limit_mismatches} "
                        f"joint_limit_projections={joint_limit_projections} "
                        f"joint_step_projections={joint_step_projections}",
                        flush=True,
                    )
                    print(
                        "NPA_OPENPI_METRICS "
                        f"elapsed_seconds={now - started:.3f} "
                        f"frames={observation_sequence} requests={requests} "
                        f"round_trips={round_trips} "
                        f"applied={applied} rejected_actions={sum(rejected_actions.values())} "
                        f"action_horizon={ACTION_SHAPE[0]} "
                        f"action_dimension={ACTION_SHAPE[1]} action_finite=1 "
                        f"rejected_wrong_shape={rejected_actions['wrong_shape']} "
                        f"rejected_non_finite={rejected_actions['non_finite']} "
                        f"rejected_joint_limit={rejected_actions['joint_limit']} "
                        f"rejected_gripper_range={rejected_actions['gripper_range']} "
                        f"rejected_joint_step={rejected_actions['joint_step']} "
                        f"raw_gripper_range_mismatches={raw_gripper_range_mismatches} "
                        f"raw_joint_limit_mismatches={raw_joint_limit_mismatches} "
                        f"joint_limit_projections={joint_limit_projections} "
                        f"joint_step_projections={joint_step_projections} "
                        f"transport_failures={sum(transport_failures.values())} "
                        f"reconnects={client.reconnects} "
                        "camera_quality_schema=3 "
                        f"camera_rejected_pairs={camera_rejected_pairs} "
                        f"camera_validated_requests={camera_validated_requests} "
                        f"camera_pair_id={camera_pair_id} "
                        f"request_camera_pair_id={request_camera_pair_id} "
                        f"round_trip_camera_pair_id={round_trip_camera_pair_id} "
                        f"camera_render_sequence={render_sequence} "
                        f"request_render_sequence={request_render_sequence} "
                        f"round_trip_render_sequence={round_trip_render_sequence} "
                        f"camera_pair_difference_current={current_camera_pair_difference:.3f} "
                        f"camera_exterior_red_cube_pixels_current={current_exterior_red_cube_pixels} "
                        f"camera_exterior_cube_in_frame_current={current_exterior_cube_in_frame} "
                        f"camera_wrist_cube_in_frame_current={current_wrist_cube_in_frame} "
                        f"camera_luminance_mean_current_min={current_luminance_mean_min:.3f} "
                        f"camera_luminance_variance_current_min={current_luminance_variance_min:.3f} "
                        f"camera_exterior_luminance_mean_current={current_exterior_luminance_mean:.3f} "
                        f"camera_exterior_luminance_variance_current={current_exterior_luminance_variance:.3f} "
                        f"camera_wrist_luminance_mean_current={current_wrist_luminance_mean:.3f} "
                        f"camera_wrist_luminance_variance_current={current_wrist_luminance_variance:.3f} "
                        f"luminance_mean_min={min(luminance_means):.3f} "
                        f"luminance_variance_min={min(luminance_variances):.3f} "
                        f"end_effector_cube_distance_m={minimum_ee_distance:.6f} "
                        f"end_effector_cube_approach_m={max(0.0, (initial_ee_distance or minimum_ee_distance) - minimum_ee_distance):.6f} "
                        f"gripper_contact_samples={gripper_contact_samples} "
                        f"gripper_contact_force_max_n={maximum_gripper_contact_force:.6f} "
                        f"cube_lift_max_m={maximum_cube_lift:.6f} "
                        f"pickup_hold_seconds={pickup_hold_seconds:.3f} "
                        f"pickup_success={int(pickup_success)} "
                        f"latency_p50_ms={percentiles[0]:.3f} "
                        f"latency_p95_ms={percentiles[1]:.3f} "
                        f"latency_p99_ms={percentiles[2]:.3f} "
                        f"latency_max_ms={max(latencies_ms):.3f}",
                        flush=True,
                    )
                except Exception as exc:
                    safe_holds += 1
                    reason = (
                        exc.reason
                        if isinstance(exc, ActionValidationError)
                        else type(exc).__name__
                    )
                    if isinstance(exc, ActionValidationError):
                        rejected_actions[reason] += 1
                        next_attempt = now + 1.0 / CONTROL_HZ
                    else:
                        transport_failures[reason] += 1
                        client.close()
                        next_attempt = now + client.reconnect_delay()
                    current_policy_error = reason
                    print(
                        "NPA_OPENPI_SAFE_HOLD "
                        f"observation={pending_observation} "
                        f"reason={reason} "
                        f"reconnects={client.reconnects}",
                        flush=True,
                    )
                finally:
                    pending = None
                    pending_joint_positions = None
                    pending_started_at = 0.0
                    pending_stall_reported = False

            if now >= next_camera_attempt:
                next_camera_attempt = now + 1.0 / CONTROL_HZ
                joint_positions = np.asarray(
                    robot.get_joint_positions(), dtype=np.float32
                )
                cube_position_for_camera = np.asarray(
                    cube.get_world_pose()[0], dtype=np.float64
                )
                exterior_cube_in_frame = _point_in_camera_frame(
                    cube_position_for_camera,
                    (
                        EXTERIOR_CAMERA_EYE,
                        EXTERIOR_CAMERA_TARGET,
                        (0.0, 0.0, 1.0),
                    ),
                    _camera_optical_config("exterior"),
                )
                wrist_cube_in_frame = _point_in_camera_frame(
                    cube_position_for_camera,
                    wrist_pose,
                    _camera_optical_config("wrist"),
                )
                samples = _capture_camera_samples(cameras)
                pair = _validate_camera_pair(
                    samples["exterior"].frame,
                    samples["wrist"].frame,
                    render_sequence=render_sequence,
                    last_accepted_render_sequence=last_accepted_render_sequence,
                    exterior_cube_in_frame=exterior_cube_in_frame,
                    wrist_cube_in_frame=wrist_cube_in_frame,
                )
                producers_advanced, current_markers, producer_reason = (
                    _camera_markers_advanced(samples, camera_markers)
                )
                if producers_advanced:
                    camera_markers = current_markers
                else:
                    pair = CameraPair(
                        False,
                        pair.exterior,
                        pair.wrist,
                        "pair",
                        producer_reason,
                        pair.mean_difference,
                    )
                readiness = camera_readiness.observe(pair, producer_reason)
                camera_policy_eligible = readiness.policy_eligible
                current_exterior_cube_in_frame = int(exterior_cube_in_frame)
                current_wrist_cube_in_frame = int(wrist_cube_in_frame)
                current_exterior_red_cube_pixels = pair.exterior.red_cube_pixels
                current_camera_pair_difference = pair.mean_difference
                current_exterior_luminance_mean = pair.exterior.luminance_mean
                current_exterior_luminance_variance = pair.exterior.luminance_variance
                current_wrist_luminance_mean = pair.wrist.luminance_mean
                current_wrist_luminance_variance = pair.wrist.luminance_variance
                current_exterior_dynamic_range = pair.exterior.dynamic_range
                current_wrist_dynamic_range = pair.wrist.dynamic_range
                current_camera_quality_accepted = int(pair.accepted)
                published_views = telemetry.publish_camera_pair(pair, render_sequence)
                if published_views:
                    print(
                        "NPA_OPENPI_IMAGE_ENQUEUED "
                        f"render_sequence={render_sequence} "
                        f"views={','.join(published_views)}",
                        flush=True,
                    )
                if readiness.emit_status:
                    print(
                        "NPA_OPENPI_CAMERA_STATUS "
                        f"status={readiness.status} "
                        f"reason={readiness.reason or 'none'} "
                        f"render_sequence={render_sequence} "
                        f"consecutive_ready={readiness.consecutive_ready} "
                        f"unavailable_frames={readiness.unavailable_frames} "
                        f"safe_hold={int(not camera_policy_eligible)} "
                        f"policy_eligible={int(camera_policy_eligible)}",
                        flush=True,
                    )
                if not camera_policy_eligible:
                    camera_rejected_pairs += 1
                    rejection_view = pair.rejected_view or "pair"
                    rejection_reason = pair.reason or readiness.reason
                    camera_rejections[f"{rejection_view}_{rejection_reason}"] += 1
                    safe_holds += 1
                    overlay[2].text = (
                        f"SAFE HOLD / {rejection_view} camera {rejection_reason}"
                    )
                    if chunk is not None:
                        chunk = None
                        chunk_index = 0
                    if readiness.emit_status:
                        print(
                            _camera_rejection_metrics_line(
                                elapsed_seconds=now - started,
                                frames=observation_sequence,
                                requests=requests,
                                round_trips=round_trips,
                                applied=applied,
                                reconnects=client.reconnects,
                                camera_rejected_pairs=camera_rejected_pairs,
                                camera_validated_requests=camera_validated_requests,
                                camera_pair_id=camera_pair_id,
                                request_camera_pair_id=request_camera_pair_id,
                                round_trip_camera_pair_id=round_trip_camera_pair_id,
                                render_sequence=render_sequence,
                                request_render_sequence=request_render_sequence,
                                round_trip_render_sequence=round_trip_render_sequence,
                                exterior_cube_in_frame=exterior_cube_in_frame,
                                wrist_cube_in_frame=wrist_cube_in_frame,
                                pair=pair,
                            ),
                            flush=True,
                        )
                elif chunk is None and pending is None and now >= next_attempt:
                    exterior_rgb = pair.exterior.rgb
                    wrist_rgb = pair.wrist.rgb
                    if first_frame:
                        print("NPA_OPENPI_FIRST_FRAME", flush=True)
                        first_frame = False
                    observation_sequence += 1
                    luminance_means.extend(
                        [
                            pair.exterior.luminance_mean,
                            pair.wrist.luminance_mean,
                        ]
                    )
                    luminance_variances.extend(
                        [
                            pair.exterior.luminance_variance,
                            pair.wrist.luminance_variance,
                        ]
                    )
                    current_exterior_luminance_mean = luminance_means[-2]
                    current_wrist_luminance_mean = luminance_means[-1]
                    current_exterior_luminance_variance = luminance_variances[-2]
                    current_wrist_luminance_variance = luminance_variances[-1]
                    current_luminance_mean_min = min(luminance_means[-2:])
                    current_luminance_variance_min = min(luminance_variances[-2:])
                    camera_pair_id += 1
                    last_accepted_render_sequence = render_sequence
                    observation = _build_policy_observation(
                        exterior_rgb, wrist_rgb, joint_positions, prompt
                    )
                    requests += 1
                    camera_validated_requests += 1
                    request_camera_pair_id = camera_pair_id
                    request_render_sequence = render_sequence
                    print(
                        "NPA_OPENPI_REQUEST "
                        f"observation={observation_sequence} requests={requests} "
                        f"camera_pair_id={request_camera_pair_id} "
                        f"render_sequence={request_render_sequence} "
                        f"task_label={TASK_LABEL}",
                        flush=True,
                    )
                    pending_observation = observation_sequence
                    pending_camera_pair_id = request_camera_pair_id
                    pending_joint_positions = joint_positions.copy()
                    pending_started_at = now
                    pending_stall_reported = False
                    pending = executor.submit(
                        client.infer,
                        PolicyRequest(
                            observation=observation,
                            camera_pair_id=request_camera_pair_id,
                            render_sequence=request_render_sequence,
                        ),
                    )

            if (
                chunk is not None
                and camera_policy_eligible
                and now - last_apply >= 1.0 / CONTROL_HZ
            ):
                target = chunk[chunk_index]
                robot.apply_action(
                    ArticulationAction(
                        joint_positions=np.concatenate(
                            [
                                target[:7],
                                np.repeat(_isaac_finger_target(target[7]), 2),
                            ]
                        )
                    )
                )
                applied += 1
                chunk_index += 1
                last_apply = now
                print(
                    f"NPA_OPENPI_APPLIED applied={applied} chunk_index={chunk_index}",
                    flush=True,
                )
                if chunk_index >= TARGETS_PER_QUERY:
                    chunk = None

            cube_position = np.asarray(cube.get_world_pose()[0], dtype=np.float64)
            ee_position = np.asarray(
                robot.end_effector.get_world_pose()[0], dtype=np.float64
            )
            ee_distance = float(np.linalg.norm(ee_position - cube_position))
            if initial_ee_distance is None:
                initial_ee_distance = ee_distance
            minimum_ee_distance = min(minimum_ee_distance, ee_distance)
            cube_lift = max(0.0, float(cube_position[2]) - cube_initial_height)
            maximum_cube_lift = max(maximum_cube_lift, cube_lift)
            contact_force = _contact_force_magnitude(cube_gripper_contacts, physics_dt)
            maximum_gripper_contact_force = max(
                maximum_gripper_contact_force, contact_force
            )
            in_gripper_contact = contact_force >= GRIPPER_CONTACT_FORCE_NEWTONS
            if in_gripper_contact:
                gripper_contact_samples += 1
            current_joint_positions = np.asarray(
                robot.get_joint_positions(), dtype=float
            )
            gripper_closed = _droid_gripper_observation(current_joint_positions) > 0.5
            pickup_candidate = bool(
                cube_lift >= PICKUP_LIFT_METERS
                and in_gripper_contact
                and gripper_closed
            )
            if pickup_candidate:
                if pickup_hold_started is None:
                    pickup_hold_started = now
                pickup_hold_seconds = now - pickup_hold_started
                pickup_success = pickup_hold_seconds >= PICKUP_HOLD_SECONDS
            else:
                pickup_hold_started = None
                if not pickup_success:
                    pickup_hold_seconds = 0.0

            safe_hold = chunk is None
            if display_rate.due(now):
                link_points = _franka_link_points(world.stage)
                proxy = _franka_proxy_geometry(link_points)
                values: list[tuple[str, object]] = [
                    (TASK_ENTITY, rr.TextLog(TASK_LABEL)),
                    (
                        f"{SCENE_ENTITY}/cube",
                        rr.Boxes3D(
                            centers=[cube_position.tolist()],
                            sizes=[[CUBE_SIZE_METERS] * 3],
                            colors=[[242, 8, 5, 255]],
                        ),
                    ),
                    (
                        f"{SCENE_ENTITY}/table",
                        rr.Boxes3D(
                            centers=[tabletop.get_world_pose()[0].tolist()],
                            sizes=[[0.9, 0.7, 0.08]],
                            colors=[[140, 92, 51, 255]],
                        ),
                    ),
                ]
                if current_policy_error:
                    values.append(
                        (POLICY_ERROR_ENTITY, rr.TextLog(current_policy_error))
                    )
                if proxy["base"] is not None:
                    values.append(
                        (
                            f"{FRANKA_SCENE_ENTITY}/base",
                            rr.Boxes3D(**proxy["base"]),
                        )
                    )
                if proxy["links"]["centers"]:
                    values.append(
                        (
                            f"{FRANKA_SCENE_ENTITY}/links",
                            rr.Boxes3D(**proxy["links"]),
                        )
                    )
                if proxy["joints"] is not None:
                    values.append(
                        (
                            f"{FRANKA_SCENE_ENTITY}/joints",
                            rr.Ellipsoids3D(**proxy["joints"]),
                        )
                    )
                if proxy["gripper"] is not None:
                    values.append(
                        (
                            f"{FRANKA_SCENE_ENTITY}/gripper",
                            rr.Boxes3D(**proxy["gripper"]),
                        )
                    )
                telemetry.publish_display(
                    DisplayPublication(
                        render_sequence=render_sequence,
                        numeric_groups=(
                            (
                                DECISION_METRICS_ENTITY,
                                (
                                    ("observation_sequence", observation_sequence),
                                    ("observation_time_seconds", now - started),
                                    ("policy_requests", requests),
                                    ("policy_in_flight", int(pending is not None)),
                                    ("round_trips", round_trips),
                                    ("inference_latency_ms", last_latency * 1000.0),
                                    ("action_horizon", ACTION_SHAPE[0]),
                                    ("action_dimension", ACTION_SHAPE[1]),
                                    ("chunk_index", chunk_index),
                                    ("safe_hold", int(safe_hold)),
                                    ("reconnects", client.reconnects),
                                    ("safe_targets_applied", applied),
                                    (
                                        "raw_gripper_range_mismatches",
                                        raw_gripper_range_mismatches,
                                    ),
                                    (
                                        "rejected_actions",
                                        sum(rejected_actions.values()),
                                    ),
                                    (
                                        "raw_joint_limit_mismatches",
                                        raw_joint_limit_mismatches,
                                    ),
                                    ("joint_limit_projections", joint_limit_projections),
                                    ("joint_step_projections", joint_step_projections),
                                    (
                                        "applied_target_rate_hz",
                                        applied / max(now - started, 1e-6),
                                    ),
                                ),
                            ),
                            (
                                GRASP_METRICS_ENTITY,
                                (
                                    ("end_effector_cube_distance_m", ee_distance),
                                    (
                                        "end_effector_cube_approach_m",
                                        max(
                                            0.0,
                                            initial_ee_distance
                                            - minimum_ee_distance,
                                        ),
                                    ),
                                    ("gripper_contact_force_n", contact_force),
                                    ("gripper_contact", int(in_gripper_contact)),
                                    ("gripper_closed", int(gripper_closed)),
                                    ("cube_lift_m", cube_lift),
                                    ("pickup_hold_seconds", pickup_hold_seconds),
                                    ("pickup_success", int(pickup_success)),
                                ),
                            ),
                            (
                                CAMERA_METRICS_ENTITY,
                                (
                                    ("pair_id", request_camera_pair_id),
                                    ("render_sequence", render_sequence),
                                    (
                                        "pair_mean_difference",
                                        current_camera_pair_difference,
                                    ),
                                    (
                                        "policy_quality_accepted",
                                        current_camera_quality_accepted,
                                    ),
                                    (
                                        "exterior_cube_in_frame",
                                        current_exterior_cube_in_frame,
                                    ),
                                    (
                                        "wrist_cube_in_frame",
                                        current_wrist_cube_in_frame,
                                    ),
                                    (
                                        "exterior_luminance_mean",
                                        current_exterior_luminance_mean,
                                    ),
                                    (
                                        "exterior_luminance_variance",
                                        current_exterior_luminance_variance,
                                    ),
                                    (
                                        "exterior_dynamic_range",
                                        current_exterior_dynamic_range,
                                    ),
                                    (
                                        "wrist_luminance_mean",
                                        current_wrist_luminance_mean,
                                    ),
                                    (
                                        "wrist_luminance_variance",
                                        current_wrist_luminance_variance,
                                    ),
                                    (
                                        "wrist_dynamic_range",
                                        current_wrist_dynamic_range,
                                    ),
                                ),
                            ),
                            (
                                FRANKA_ACTION_ENTITY,
                                tuple(
                                    (f"joint_{index}", float(value))
                                    for index, value in enumerate(
                                        current_joint_positions[:9]
                                    )
                                ),
                            ),
                        ),
                        values=tuple(values),
                    )
                )

            if now >= next_loop_heartbeat_at:
                next_loop_heartbeat_at = now + 1.0
                telemetry_state = telemetry.snapshot()
                print(
                    "NPA_OPENPI_LOOP_HEARTBEAT "
                    f"elapsed_seconds={now - started:.3f} "
                    f"render_sequence={render_sequence} "
                    f"policy_in_flight={int(pending is not None)} "
                    f"logger_dropped={telemetry_state['logger']['dropped']}",
                    flush=True,
                )
            overlay[2].text = (
                "SAFE HOLD / reconnecting"
                if safe_hold
                else "VALIDATED ACTIONS APPLYING"
            )
            overlay[3].text = (
                f"obs {observation_sequence} | requests {requests} | round trips {round_trips}\n"
                f"action [15,8], row {chunk_index} | applied {applied} | reconnects {client.reconnects}"
            )
            overlay[
                4
            ].text = f"last inference {last_latency * 1000.0:.1f} ms | safe holds {safe_holds}"
    finally:
        client.shutdown()
        executor.shutdown(wait=False, cancel_futures=True)
        telemetry_closed = telemetry.close()
        telemetry_state = telemetry.snapshot()
        print(
            "NPA_OPENPI_TELEMETRY_STOP "
            f"closed={telemetry_closed} state={telemetry_state}",
            flush=True,
        )
        with contextlib.suppress(Exception):
            exterior.close()
        with contextlib.suppress(Exception):
            wrist.close()
        if not all(telemetry_closed.values()):
            raise TelemetryShutdownError(
                "logger worker remained alive; refusing result finalization"
            )
        run.add_result("observation_sequence", observation_sequence)
        run.add_result("policy_requests", requests)
        run.add_result("policy_round_trips", round_trips)
        run.add_result("safe_targets_applied", applied)
        run.add_result("raw_gripper_range_mismatches", raw_gripper_range_mismatches)
        run.add_result("raw_joint_limit_mismatches", raw_joint_limit_mismatches)
        run.add_result("joint_limit_projections", joint_limit_projections)
        run.add_result("joint_step_projections", joint_step_projections)
        run.add_result("reconnects", client.reconnects)
        run.add_result("rejected_actions", dict(sorted(rejected_actions.items())))
        run.add_result("transport_failures", dict(sorted(transport_failures.items())))
        run.add_result("camera_quality_schema", 3)
        run.add_result("camera_rejected_pairs", camera_rejected_pairs)
        run.add_result("camera_rejections", dict(sorted(camera_rejections.items())))
        run.add_result("camera_validated_requests", camera_validated_requests)
        run.add_result("camera_pair_id", camera_pair_id)
        run.add_result("request_camera_pair_id", request_camera_pair_id)
        run.add_result("round_trip_camera_pair_id", round_trip_camera_pair_id)
        run.add_result("camera_render_sequence", render_sequence)
        run.add_result("request_render_sequence", request_render_sequence)
        run.add_result("round_trip_render_sequence", round_trip_render_sequence)
        run.add_result("camera_pair_difference_current", current_camera_pair_difference)
        run.add_result(
            "camera_exterior_red_cube_pixels_current",
            current_exterior_red_cube_pixels,
        )
        run.add_result("task_label", TASK_LABEL)
        run.add_result("minimum_end_effector_cube_distance_m", minimum_ee_distance)
        run.add_result("maximum_cube_lift_m", maximum_cube_lift)
        run.add_result("gripper_contact_samples", gripper_contact_samples)
        run.add_result("maximum_gripper_contact_force_n", maximum_gripper_contact_force)
        run.add_result("pickup_hold_seconds", pickup_hold_seconds)
        run.add_result("pickup_success", pickup_success)
        if latencies_ms:
            percentiles = np.percentile(latencies_ms, [50, 95, 99])
            run.add_result("latency_p50_ms", float(percentiles[0]))
            run.add_result("latency_p95_ms", float(percentiles[1]))
            run.add_result("latency_p99_ms", float(percentiles[2]))
            run.add_result("latency_max_ms", float(max(latencies_ms)))
        if luminance_means:
            run.add_result("camera_luminance_mean_min", float(min(luminance_means)))
            run.add_result(
                "camera_luminance_variance_min", float(min(luminance_variances))
            )
