"""Isaac Lab Franka/camera runtime for the OpenPI bridge.

This module is started only through ``/isaac-sim/python.sh``.  It may also be
called from an Antioch-authored scenario after ``antioch.boot()``; the robot,
cameras, and position-control path remain native Isaac Lab in both cases.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import threading
import time
from typing import Any, Callable
import urllib.error
import urllib.request
from urllib.parse import urlparse

import numpy as np

from .openpi_bridge import (
    ACTION_SHAPE,
    OpenPIBridgeError,
    OpenPIWebsocketClient,
    RetryPolicy,
    safe_position_targets,
)
from .openpi_streaming import StreamingConfig, StreamingPolicyLoop


def _verify_vulkan_runtime() -> None:
    """Fail before Kit startup when the host exposed only CUDA driver userspace."""

    configured_icds = os.environ.get("VK_ICD_FILENAMES", "").split(":")
    icd_paths = [Path(value) for value in configured_icds if value]
    if not icd_paths:
        icd_paths = [
            Path("/etc/vulkan/icd.d/nvidia_icd.json"),
            Path("/usr/share/vulkan/icd.d/nvidia_icd.json"),
        ]
    if not any(path.is_file() for path in icd_paths):
        raise OpenPIBridgeError(
            "NVIDIA Vulkan ICD is unavailable; refusing to start Isaac without "
            "the host graphics driver capability"
        )
    vulkaninfo = shutil.which("vulkaninfo")
    if not vulkaninfo:
        raise OpenPIBridgeError(
            "vulkaninfo is unavailable; cannot prove Isaac rendering readiness"
        )
    try:
        probe = subprocess.run(
            [vulkaninfo, "--summary"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise OpenPIBridgeError("Vulkan readiness probe timed out") from exc
    output = probe.stdout + probe.stderr
    if probe.returncode != 0 or "NVIDIA" not in output.upper():
        raise OpenPIBridgeError(
            "Vulkan readiness probe did not find an NVIDIA renderer; refusing "
            "non-render fallback"
        )


def _ensure_franka_asset_root(assets: Any | None = None) -> str:
    """Use the latest published NVIDIA asset root when a newer SDK points at 404s."""

    if assets is None:
        import isaaclab.utils.assets as assets

    root = str(assets.NUCLEUS_ASSET_ROOT_DIR).rstrip("/")
    match = re.search(r"/Assets/Isaac/([^/]+)$", root)
    if not root.startswith("https://") or match is None:
        return "native"
    sentinel = f"{root}/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd"

    def available(url: str) -> bool:
        request = urllib.request.Request(url, method="HEAD")
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                return int(response.status) == 200
        except (OSError, urllib.error.HTTPError):
            return False

    if available(sentinel):
        return "native"
    # Antioch's Isaac Sim 6.0 engine currently advertises an asset prefix before
    # that prefix is published. NVIDIA's immutable 5.1 Franka USD is compatible
    # with the articulation/camera API used here; prove it exists before changing
    # any module constants so a network outage still fails closed.
    fallback_root = root[: match.start(1)] + "5.1"
    fallback_sentinel = sentinel.replace(root, fallback_root, 1)
    if not available(fallback_sentinel):
        raise OpenPIBridgeError(
            "NVIDIA Franka asset is unavailable at both the native and reviewed "
            "compatibility roots"
        )
    for name, value in vars(assets).items():
        if name.endswith("_DIR") and isinstance(value, str) and value.startswith(root):
            setattr(assets, name, fallback_root + value[len(root) :])
    return "nvidia-5.1-compatibility"


def _compatible_franka_asset_url(url: str, asset_compatibility: str) -> str:
    if asset_compatibility == "nvidia-5.1-compatibility":
        return url.replace("/Assets/Isaac/6.0/", "/Assets/Isaac/5.1/", 1)
    return url


def _write_report(uri: str, report: dict[str, object]) -> None:
    if not uri:
        print(
            "NPA_OPENPI_BRIDGE_RESULT=" + json.dumps(report, sort_keys=True), flush=True
        )
        return
    payload = json.dumps(report, indent=2, sort_keys=True).encode() + b"\n"
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        import boto3

        boto3.client(
            "s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or None
        ).put_object(
            Bucket=parsed.netloc,
            Key=parsed.path.lstrip("/"),
            Body=payload,
            ContentType="application/json",
            IfNoneMatch="*",
        )
        return
    path = Path(parsed.path if parsed.scheme == "file" else uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _camera_frame(image: Any) -> Any:
    value = image[0] if image.ndim == 4 else image
    if value.ndim != 3 or value.shape[-1] < 3:
        raise OpenPIBridgeError(
            f"camera returned invalid RGB shape {tuple(image.shape)}"
        )
    return value[:, :, :3]


def _position_target_tensor(
    torch_module: Any, target: np.ndarray, *, device: Any
) -> Any:
    """Create one Isaac target without trusting hosted backend dtype wrappers."""

    fingers = np.repeat(float(target[7]) * 0.04, 2)
    return torch_module.as_tensor(
        np.concatenate([target[:7], fingers]),
        device=device,
        dtype=torch_module.float32,
    ).unsqueeze(0)


def _resize_rgb(image: Any) -> np.ndarray:
    import torch

    value = _camera_frame(image).permute(2, 0, 1).unsqueeze(0).float()
    resized = torch.nn.functional.interpolate(
        value, size=(224, 224), mode="bilinear", align_corners=False
    )
    return np.clip(resized[0].permute(1, 2, 0).cpu().numpy(), 0, 255).astype(np.uint8)


def _wait_for_camera_observation(
    capture: Callable[[], dict[str, object]],
    advance: Callable[[], None],
    *,
    timeout_seconds: float,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Render boundedly until both cameras produce one complete observation."""

    if not np.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise OpenPIBridgeError(
            "OPENPI_CAMERA_WARMUP_SECONDS must be finite and positive"
        )
    deadline = monotonic() + timeout_seconds
    last_error: OpenPIBridgeError | None = None
    while monotonic() < deadline:
        advance()
        try:
            return capture()
        except OpenPIBridgeError as exc:
            last_error = exc
            sleep(0.01)
    raise OpenPIBridgeError(
        "camera did not produce a complete RGB observation before the warmup deadline"
    ) from last_error


def _close_runtime_resource(resource: Any | None, *, failed: bool) -> None:
    """Close Kit-owned resources only after success.

    Isaac environment/application close can terminate the interpreter with a
    zero status.  On failure, normal process unwinding owns cleanup so the
    already-emitted failed-safe report and nonzero exit cannot be masked.
    """

    if resource is not None and not failed:
        resource.close()


def _capture_viewport_rgb(
    sim: Any, *, eye: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Capture one real RTX viewport frame from Antioch's Kit-owned renderer."""

    import omni.kit.app
    import omni.kit.renderer_capture
    from omni.kit.viewport.utility import (
        capture_viewport_to_buffer,
        get_active_viewport,
    )
    from isaacsim.core.utils.viewports import set_camera_view

    viewport = get_active_viewport()
    if viewport is None:
        raise OpenPIBridgeError(
            "Antioch viewport is unavailable; run the camera suite with its "
            "authenticated stream enabled"
        )
    set_camera_view(eye=eye, target=target, viewport_api=viewport)
    captured: dict[str, object] = {}
    ready = threading.Event()

    def on_capture(
        buffer: object,
        buffer_size: int,
        width: int,
        height: int,
        pixel_format: object,
    ) -> None:
        captured["pixels"] = omni.kit.renderer_capture.convert_raw_bytes_to_list(
            buffer, buffer_size, width, height, pixel_format
        )
        captured["width"] = width
        captured["height"] = height
        ready.set()

    capture = capture_viewport_to_buffer(viewport, on_capture)
    app = omni.kit.app.get_app()
    for _ in range(120):
        sim.render()
        # Antioch owns Kit's outer application loop.  A simulation render alone
        # does not dispatch the viewport capture future when a scenario is
        # executing synchronously, so explicitly advance the supported Kit app
        # update before checking the callback.  The standalone launcher already
        # gets the same update through SimulationContext.render().
        app.update()
        if ready.wait(0.05):
            break
    _ = capture
    if not ready.is_set():
        raise OpenPIBridgeError("RTX viewport camera capture timed out")
    width = int(captured["width"])
    height = int(captured["height"])
    pixels = np.asarray(captured["pixels"], dtype=np.uint8)
    expected = width * height * 4
    if width < 1 or height < 1 or pixels.size != expected:
        raise OpenPIBridgeError(
            f"RTX viewport returned malformed RGBA buffer {pixels.size}, expected {expected}"
        )
    return pixels.reshape(height, width, 4)[:, :, :3]


def run(*, launch_application: bool = True) -> dict[str, object]:
    """Run continuous soft-real-time control, or an explicit finite smoke.

    Antioch's scenario runner already owns Kit startup, so an authored scenario
    passes ``launch_application=False``. The default ``continuous`` mode is
    long-lived when no duration is configured. ``finite-smoke`` preserves the
    one-observation/one-chunk diagnostic and is never the production default.
    """

    env = None
    client = None
    stream = None
    simulation_app = None
    failed = False
    report: dict[str, object]
    try:
        if launch_application:
            _verify_vulkan_runtime()
            from isaaclab.app import AppLauncher

            simulation_app = AppLauncher(headless=True, enable_cameras=True).app
        import gymnasium as gym
        import isaaclab.sim as sim_utils
        from isaaclab.utils import assets as asset_utils

        asset_compatibility = _ensure_franka_asset_root(asset_utils)
        import isaaclab_tasks  # noqa: F401
        import torch
        from isaaclab.sensors import CameraCfg
        from isaaclab_tasks.utils import parse_env_cfg

        from npa.workflows.isaac_capture import look_at_quaternion

        control_mode = os.environ.get("OPENPI_CONTROL_MODE", "continuous")
        if control_mode not in {"continuous", "finite-smoke"}:
            raise OpenPIBridgeError(
                "OPENPI_CONTROL_MODE must be continuous or finite-smoke"
            )
        client = OpenPIWebsocketClient(
            os.environ.get("OPENPI_POLICY_HOST", ""),
            port=int(os.environ.get("OPENPI_POLICY_PORT", "8000")),
            connect_timeout_seconds=float(
                os.environ.get("OPENPI_CONNECT_TIMEOUT_SECONDS", "10")
            ),
            inference_timeout_seconds=float(
                os.environ.get(
                    "OPENPI_INFERENCE_TIMEOUT_SECONDS",
                    os.environ.get("OPENPI_INFERENCE_DEADLINE_SECONDS", "10"),
                )
            ),
            retry=RetryPolicy(attempts=1 if control_mode == "continuous" else 4),
        )
        if not torch.cuda.is_available():
            raise OpenPIBridgeError("CUDA is unavailable; refusing non-render fallback")
        properties = torch.cuda.get_device_properties(0)
        capability = f"{properties.major}.{properties.minor}"
        if capability != "12.0":
            raise OpenPIBridgeError(
                f"Isaac bridge requires RTX PRO 6000 sm_120, received compute capability {capability}"
            )
        cfg = parse_env_cfg("Isaac-Lift-Cube-Franka-v0", device="cuda:0", num_envs=1)
        cfg.scene.robot.spawn.usd_path = _compatible_franka_asset_url(
            cfg.scene.robot.spawn.usd_path, asset_compatibility
        )
        if launch_application:
            cfg.scene.npa_exterior_camera = CameraCfg(
                prim_path="{ENV_REGEX_NS}/NpaExteriorCamera",
                offset=CameraCfg.OffsetCfg(
                    pos=(1.4, 1.4, 1.2),
                    rot=look_at_quaternion((1.4, 1.4, 1.2), (0.5, 0.0, 0.6)),
                    convention="world",
                ),
                data_types=["rgb"],
                width=320,
                height=320,
                spawn=sim_utils.PinholeCameraCfg(focal_length=24.0),
            )
            cfg.scene.npa_wrist_camera = CameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand/NpaWristCamera",
                offset=CameraCfg.OffsetCfg(
                    pos=(0.08, 0.0, 0.02),
                    rot=(0.7071068, 0.0, 0.7071068, 0.0),
                    convention="ros",
                ),
                data_types=["rgb"],
                width=320,
                height=320,
                spawn=sim_utils.PinholeCameraCfg(focal_length=18.0),
            )
        env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=cfg)
        env.reset()
        uenv = env.unwrapped
        robot = uenv.scene["robot"]
        for _ in range(4):
            uenv.scene.write_data_to_sim()
            uenv.sim.step(render=True)
            uenv.scene.update(uenv.sim.get_physics_dt())
        camera_backend = (
            "isaac-lab-camera-sensors"
            if launch_application
            else "antioch-authenticated-rtx-viewport"
        )

        def current_state() -> tuple[np.ndarray, float]:
            joints = robot.data.joint_pos[0, :7].detach().cpu().numpy()
            finger = robot.data.joint_pos[0, 7:9].mean().item()
            return joints, float(np.clip(finger / 0.04, 0.0, 1.0))

        def capture_observation() -> dict[str, object]:
            joint_position, gripper_position = current_state()
            if launch_application:
                exterior_rgb = _resize_rgb(
                    uenv.scene["npa_exterior_camera"].data.output["rgb"]
                )
                wrist_rgb = _resize_rgb(
                    uenv.scene["npa_wrist_camera"].data.output["rgb"]
                )
            else:
                from isaaclab.utils.math import quat_apply

                hand_ids, _ = robot.find_bodies("panda_hand")
                hand_position = robot.data.body_pos_w[0, hand_ids[0]]
                hand_rotation = robot.data.body_quat_w[0, hand_ids[0]]
                wrist_eye = hand_position + quat_apply(
                    hand_rotation,
                    torch.tensor([0.08, 0.0, 0.02], device=robot.device),
                )
                wrist_target = wrist_eye + quat_apply(
                    hand_rotation,
                    torch.tensor([0.3, 0.0, 0.0], device=robot.device),
                )
                exterior_rgb = _resize_rgb(
                    torch.as_tensor(
                        _capture_viewport_rgb(
                            uenv.sim,
                            eye=np.asarray([1.4, 1.4, 1.2]),
                            target=np.asarray([0.5, 0.0, 0.6]),
                        )
                    )
                )
                wrist_rgb = _resize_rgb(
                    torch.as_tensor(
                        _capture_viewport_rgb(
                            uenv.sim,
                            eye=wrist_eye.detach().cpu().numpy(),
                            target=wrist_target.detach().cpu().numpy(),
                        )
                    )
                )
            return {
                "observation/exterior_image_1_left": exterior_rgb,
                "observation/wrist_image_left": wrist_rgb,
                "observation/joint_position": joint_position.astype(np.float32),
                "observation/gripper_position": np.asarray(
                    [gripper_position], dtype=np.float32
                ),
                "prompt": os.environ.get("OPENPI_PROMPT", "pick up the fork"),
            }

        def set_position_target(target: np.ndarray) -> None:
            full_target = _position_target_tensor(
                torch, target, device=robot.device
            )
            robot.set_joint_position_target(full_target)

        def advance_simulation() -> None:
            uenv.scene.write_data_to_sim()
            uenv.sim.step(render=True)
            uenv.scene.update(uenv.sim.get_physics_dt())

        if control_mode == "finite-smoke":
            observation = capture_observation()
            actions = client.infer(observation)
            targets = safe_position_targets(
                actions,
                np.asarray(observation["observation/joint_position"]),
                max_joint_delta_rad=float(
                    os.environ.get("OPENPI_MAX_JOINT_DELTA_RAD", "0.08")
                ),
                execute_steps=int(os.environ.get("OPENPI_EXECUTE_STEPS", "5")),
            )
            for target in targets:
                set_position_target(target)
                advance_simulation()
            report = {
                "schema": "npa.antioch.openpi-franka-bridge.v2",
                "status": "passed",
                "control_mode": "finite-smoke",
                "simulator": "isaac-lab",
                "antioch_compatible": True,
                "gpu_compute_capability": capability,
                "asset_root": asset_compatibility,
                "camera_shapes": [[224, 224, 3], [224, 224, 3]],
                "camera_backend": camera_backend,
                "policy_action_shape": list(ACTION_SHAPE),
                "targets_executed": len(targets),
                "position_control": "absolute-rate-limited",
                "policy_transport": "private-openpi-websocket",
                "fail_closed": True,
            }
        else:
            config = StreamingConfig(
                observation_hz=float(os.environ.get("OPENPI_OBSERVATION_HZ", "10")),
                policy_request_hz=float(
                    os.environ.get("OPENPI_POLICY_REQUEST_HZ", "2")
                ),
                control_hz=float(os.environ.get("OPENPI_CONTROL_HZ", "10")),
                executed_targets_per_chunk=int(
                    os.environ.get("OPENPI_EXECUTED_TARGETS_PER_CHUNK", "5")
                ),
                maximum_observation_age_seconds=float(
                    os.environ.get("OPENPI_MAXIMUM_OBSERVATION_AGE_SECONDS", "0.75")
                ),
                maximum_response_age_seconds=float(
                    os.environ.get("OPENPI_MAXIMUM_RESPONSE_AGE_SECONDS", "1.5")
                ),
                inference_deadline_seconds=float(
                    os.environ.get("OPENPI_INFERENCE_DEADLINE_SECONDS", "10")
                ),
                ping_interval_seconds=float(
                    os.environ.get("OPENPI_PING_INTERVAL_SECONDS", "5")
                ),
                safe_hold_behavior=os.environ.get(
                    "OPENPI_SAFE_HOLD_BEHAVIOR", "hold-current"
                ),
                minimum_ready_cycles=int(
                    os.environ.get("OPENPI_MINIMUM_READY_CYCLES", "3")
                ),
                minimum_ready_seconds=float(
                    os.environ.get("OPENPI_MINIMUM_READY_SECONDS", "5")
                ),
                maximum_joint_delta_rad=float(
                    os.environ.get("OPENPI_MAX_JOINT_DELTA_RAD", "0.08")
                ),
            )
            duration_seconds = float(
                os.environ.get("OPENPI_STREAM_DURATION_SECONDS", "0")
            )
            if not np.isfinite(duration_seconds) or duration_seconds < 0:
                raise OpenPIBridgeError(
                    "OPENPI_STREAM_DURATION_SECONDS must be finite and non-negative"
                )
            initial_observation = _wait_for_camera_observation(
                capture_observation,
                advance_simulation,
                timeout_seconds=float(
                    os.environ.get("OPENPI_CAMERA_WARMUP_SECONDS", "10")
                ),
            )
            stream = StreamingPolicyLoop(client, config=config)
            shutdown = threading.Event()
            ready_path = Path(
                os.environ.get("OPENPI_READY_FILE", "/tmp/npa-openpi-stream-ready")
            )
            ready_path.unlink(missing_ok=True)
            old_handlers: dict[int, object] = {}

            def request_shutdown(_signum: int, _frame: object) -> None:
                shutdown.set()

            if threading.current_thread() is threading.main_thread():
                for signum in (signal.SIGINT, signal.SIGTERM):
                    old_handlers[signum] = signal.getsignal(signum)
                    signal.signal(signum, request_shutdown)
            started = time.monotonic()
            next_observation = started
            next_control = started
            next_metrics = started + 10.0
            stream.start()
            try:
                stream.record_render_tick()
                stream.publish_observation(
                    initial_observation, monotonic_seconds=time.monotonic()
                )
                while not shutdown.is_set() and (
                    duration_seconds == 0
                    or time.monotonic() - started < duration_seconds
                ):
                    advance_simulation()
                    stream.record_render_tick()
                    now = time.monotonic()
                    if now >= next_observation:
                        observation = capture_observation()
                        stream.publish_observation(
                            observation, monotonic_seconds=time.monotonic()
                        )
                        next_observation = now + 1.0 / config.observation_hz
                    if now >= next_control:
                        joints, gripper_position = current_state()
                        decision = stream.next_control_decision(
                            joints, gripper_position, monotonic_seconds=now
                        )
                        if decision.source == "policy":
                            stream.apply_if_current(decision, set_position_target)
                        elif decision.target is not None:
                            set_position_target(decision.target)
                        next_control = now + 1.0 / config.control_hz
                    if now >= next_metrics:
                        print(
                            "NPA_OPENPI_STREAM_METRICS="
                            + json.dumps(stream.metrics_snapshot(), sort_keys=True),
                            flush=True,
                        )
                        next_metrics = now + 10.0
                    if not ready_path.exists() and stream.metrics_snapshot()["ready"]:
                        temporary_ready = ready_path.with_suffix(".tmp")
                        temporary_ready.write_text("ready\n", encoding="utf-8")
                        temporary_ready.replace(ready_path)
                    next_due = min(next_observation, next_control, next_metrics)
                    time.sleep(min(0.01, max(0.001, next_due - time.monotonic())))
            finally:
                stream.stop()
                for signum, handler in old_handlers.items():
                    signal.signal(signum, handler)
            metrics = stream.metrics_snapshot()
            if not metrics["ready"]:
                raise OpenPIBridgeError(
                    "continuous bridge stopped before sustained streaming readiness"
                )
            report = {
                "schema": "npa.antioch.openpi-franka-bridge.v2",
                "status": "passed",
                "control_mode": "continuous",
                "soft_real_time": True,
                "hard_real_time": False,
                "simulator": "isaac-lab",
                "antioch_compatible": True,
                "gpu_compute_capability": capability,
                "asset_root": asset_compatibility,
                "camera_shapes": [[224, 224, 3], [224, 224, 3]],
                "camera_backend": camera_backend,
                "policy_action_shape": list(ACTION_SHAPE),
                "targets_executed": metrics["safely_applied_targets"],
                "position_control": "receding-horizon-absolute-rate-limited",
                "policy_transport": "persistent-private-msgpack-websocket",
                "streaming_metrics": metrics,
                "fail_closed": True,
            }
        # Kit shutdown may terminate the interpreter rather than return to
        # Python, so persist success evidence before closing the application.
        _write_report(os.environ.get("NPA_OPENPI_BRIDGE_OUTPUT_URI", ""), report)
    except Exception as exc:
        failed = True
        streaming_metrics = stream.metrics_snapshot() if stream is not None else None
        report = {
            "schema": "npa.antioch.openpi-franka-bridge.v2",
            "status": (
                "failed-safe-hold" if streaming_metrics is not None else "failed-no-action"
            ),
            "error_type": type(exc).__name__,
            "targets_executed": (
                streaming_metrics["safely_applied_targets"]
                if streaming_metrics is not None
                else 0
            ),
            **(
                {"streaming_metrics": streaming_metrics}
                if streaming_metrics is not None
                else {}
            ),
            "fail_closed": True,
        }
        _write_report(os.environ.get("NPA_OPENPI_BRIDGE_OUTPUT_URI", ""), report)
        raise
    finally:
        if stream is not None and stream.is_running():
            stream.stop()
        if client is not None:
            client.close()
        _close_runtime_resource(env, failed=failed)
        # On failure, normal interpreter unwinding tears Kit down. Calling
        # SimulationApp.close() can terminate with status zero and mask the
        # exception from Kubernetes; the failed-safe report is already durable.
        if launch_application:
            _close_runtime_resource(simulation_app, failed=failed)
    return report


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
