"""Retired controller Isaac and Genesis held-out simulation adapters."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any


from npa.clients.storage import StorageClient
from npa.workflows.sim2real.artifact_upload import (
    upload_run_artifacts,  # noqa: F401 - public engine import surface
)
from npa.workbench.cosmos.reason import (
    CosmosReasonError,
    resolve_cosmos_reason_model_id,
    run_cosmos_reason_vlm,
    task_description_from_manifest,
)
from npa.workflows.sim2real.constants import (
    DEFAULT_ISAAC_TASK,
    DEFAULT_REFERENCE_VLM_MODEL,
    SIM_BACKEND_ISAAC,
)
from npa.workflows.sim2real.models import (
    Sim2RealLoopError,
)
from npa.workflows.sim2real.k8s_components import (
    _component_job_script as _component_job_script,
    _kubernetes_component_env as _kubernetes_component_env,
)
from npa.workflows.sim2real.policy_actions_stage import (
    run_policy_actions_component_from_s3 as run_policy_actions_component_from_s3,
)
from npa.workflows.sim2real.reference_helpers import (
    _heldout_env_score,
    _signal_diversity_report as _signal_diversity_report,
    _signal_mean_reward as _signal_mean_reward,
    _write_env_manifest as _write_env_manifest,
    _write_train_heldout_split as _write_train_heldout_split,
)
from npa.workflows.sim2real.utils import (
    _bool_value,
)
from npa.workflows.sim2real.workflow_state_io import (
    _workflow_state_path,  # noqa: F401 - legacy engine import surface
    emit_active_progress_rerun,  # noqa: F401 - imported by runner from engine
    sync_workflow_state_to_s3,  # noqa: F401 - imported by runner from engine
)

# Isaac Sim app handle — closed only after held-out report upload.
_ISAAC_SIMULATION_APP: Any = None
HELDOUT_VIZ_CAMERA_NAME = "heldout_viz_camera"
DEFAULT_HELDOUT_RENDER_FRAMES = 8
SCHEMA_HELDOUT_RENDERS = "npa.sim2real.heldout_renders.v1"
if TYPE_CHECKING:
    pass


_HELDOUT_DISTANCE_THRESHOLDS = (0.05, 0.10, 0.15, 0.20)


def _rollout_image_paths(rollout_root: Path, observations: list[Any]) -> list[Path]:
    paths: list[Path] = []
    for observation in observations:
        path = rollout_root / str(observation)
        if path.is_file():
            paths.append(path)
    if paths:
        return paths
    return sorted(
        path
        for path in rollout_root.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".ppm", ".webp"}
    )


def _task_description_from_manifest(manifest: dict[str, Any]) -> str:
    return task_description_from_manifest(manifest)


def _resolve_cosmos_reason_model_id(model: str) -> str:
    return resolve_cosmos_reason_model_id(model, default=DEFAULT_REFERENCE_VLM_MODEL)


def _run_cosmos_reason_vlm(
    *,
    model_id: str,
    image_paths: list[Path],
    actions: list[dict[str, Any]],
    task_description: str,
    rollout_id: str,
    threshold: float,
) -> dict[str, Any]:
    try:
        return run_cosmos_reason_vlm(
            model_id=model_id,
            image_paths=image_paths,
            actions=actions,
            task_description=task_description,
            rollout_id=rollout_id,
            threshold=threshold,
        )
    except CosmosReasonError as exc:
        raise Sim2RealLoopError(str(exc)) from exc


def _run_genesis_heldout_rollouts(
    envs: list[dict[str, Any]],
    *,
    inner_evidence: dict[str, Any],
    threshold: float,
    scene: Any = None,
    robot: Any = None,
) -> list[dict[str, Any]]:
    """Run the trained adapter policy through real Genesis held-out episodes.

    When ``scene`` (a parsed ``npa.genesis.scene_assets.SceneSpec`` with
    resolved local asset paths) is provided, the manipulated object(s) are
    built from it (mesh / primitive) instead of the default red Box. The
    SceneSpec objects' ``loaded`` provenance flags are set as a side effect of
    building the env, so the caller can prove the requested mesh loaded.

    When ``robot`` (a resolved ``npa.genesis.robot_assets.RobotSpec``) is
    provided, the env loads that embodiment (URDF/MJCF/preset) instead of the
    hardcoded Franka Panda; its ``loaded`` flag is set when the env builds it.
    """

    try:
        import torch
        from npa.genesis.env_pick_place import EnvConfig, FrankaPickPlaceEnv
    except Exception as exc:
        raise Sim2RealLoopError(
            f"Genesis rollout eval requires torch and genesis-world in the image: {exc}"
        ) from exc
    if not torch.cuda.is_available():
        raise Sim2RealLoopError("Genesis rollout eval requires a CUDA GPU")

    if scene is not None:
        manip = scene.manipuland()
        print(
            json.dumps(
                {
                    "component": "heldout_eval",
                    "event": "byo_scene_loading",
                    "asset_source": manip.asset_source,
                    "manipuland": manip.name,
                    "local_path": manip.local_path,
                    "sha256": manip.sha256,
                    "object_count": len(scene.objects),
                },
                sort_keys=True,
            )
        )
    if robot is not None:
        print(
            json.dumps(
                {
                    "component": "heldout_eval",
                    "event": "byo_robot_loading",
                    "robot_source": robot.robot_source,
                    "robot_name": robot.name,
                    "ee_link": robot.ee_link,
                    "dof_count": robot.dof_count,
                    "local_path": robot.local_path,
                    "sha256": robot.sha256,
                },
                sort_keys=True,
            )
        )

    adapter = _policy_adapter_from_inner_evidence(inner_evidence)
    batch_size = max(1, int(os.environ.get("NPA_SIM2REAL_GENESIS_BATCH_SIZE", "16")))
    max_steps = max(1, int(os.environ.get("NPA_SIM2REAL_GENESIS_MAX_STEPS", "240")))
    per_env: list[dict[str, Any]] = []
    for start in range(0, len(envs), batch_size):
        batch = envs[start : start + batch_size]
        seed = int(batch[0].get("seed") or (42 + start))
        torch.manual_seed(seed)
        cfg = EnvConfig(
            n_envs=len(batch),
            enable_cameras=False,
            domain_randomize=True,
            max_episode_steps=max_steps,
            action_space="cartesian",
            action_scale=float(
                os.environ.get("NPA_SIM2REAL_GENESIS_ACTION_SCALE", "0.045")
            ),
            scene_spec=scene,
            robot_spec=robot,
        )
        env = FrankaPickPlaceEnv(cfg)
        if scene is not None and start == 0:
            print(
                json.dumps(
                    {
                        "component": "heldout_eval",
                        "event": "byo_scene_loaded",
                        "asset_fallback_used": scene.asset_fallback_used,
                        "loaded_objects": [
                            obj.name for obj in scene.objects if obj.loaded
                        ],
                    },
                    sort_keys=True,
                )
            )
        if robot is not None and start == 0:
            print(
                json.dumps(
                    {
                        "component": "heldout_eval",
                        "event": "byo_robot_loaded",
                        "robot_source": robot.robot_source,
                        "robot_name": robot.name,
                        "loaded": bool(robot.loaded),
                        "robot_fallback_used": False,
                    },
                    sort_keys=True,
                )
            )
        obs = env.reset()
        active = torch.ones(len(batch), device="cuda", dtype=torch.bool)
        success = torch.zeros(len(batch), device="cuda", dtype=torch.bool)
        steps_done = torch.zeros(len(batch), device="cuda", dtype=torch.long)
        max_reward = torch.full((len(batch),), -1.0e9, device="cuda")
        final_distance = torch.full((len(batch),), 1.0e9, device="cuda")
        for step in range(max_steps):
            actions = _adapter_policy_actions(obs, adapter, step=step)
            obs, reward, done, info = env.step(actions)
            distance = torch.norm(
                obs["object_pose"][:, :3] - obs["goal_position"], dim=-1
            )
            final_distance = torch.where(active, distance, final_distance)
            max_reward = torch.where(
                active, torch.maximum(max_reward, reward), max_reward
            )
            just_done = active & done
            if bool(just_done.any()):
                success = torch.where(just_done, info["success"].bool(), success)
                steps_done = torch.where(
                    just_done, torch.full_like(steps_done, step + 1), steps_done
                )
                active = active & ~just_done
            if not bool(active.any()):
                break
        steps_done = torch.where(
            steps_done == 0,
            torch.full_like(steps_done, max_steps),
            steps_done,
        )
        batch_successes = int(success.sum().item())
        print(
            json.dumps(
                {
                    "component": "heldout_eval",
                    "event": "genesis_rollout_batch_complete",
                    "batch_start": start,
                    "env_count": len(batch),
                    "successes": batch_successes,
                    "max_steps": max_steps,
                },
                sort_keys=True,
            )
        )
        for index, env_record in enumerate(batch):
            dist = float(final_distance[index].detach().item())
            reward_value = float(max_reward[index].detach().item())
            env_success = bool(success[index].detach().item())
            distance_score = max(0.0, min(1.0, 1.0 - dist / 0.5))
            reward_score = max(0.0, min(1.0, reward_value / 10.0))
            score = _heldout_env_score(
                distance_score, reward_score, env_success=env_success
            )
            per_env.append(
                {
                    "env_id": str(
                        env_record.get("env_id") or f"heldout-{start + index:04d}"
                    ),
                    "score": score,
                    "success": env_success,
                    "details": {
                        "source": "genesis_env_native_success",
                        "seed": env_record.get("seed"),
                        "target_threshold": cfg.target_threshold,
                        "final_target_distance": round(dist, 6),
                        "max_reward": round(reward_value, 6),
                        "steps": int(steps_done[index].detach().item()),
                        "policy_adapter": adapter,
                        "threshold": threshold,
                    },
                }
            )
    return per_env


def _isaac_import_mesh_to_usd(local_path: str, *, work_dir: Path) -> str:
    """Convert a BYO mesh/URDF to USD using Isaac Lab's offline converters.

    Returns the resolved USD path. Raises ``Sim2RealLoopError`` if conversion
    does not produce a USD file (no silent fallback to the stock asset).
    """

    src = Path(local_path)
    if not src.is_file() or src.stat().st_size == 0:
        raise Sim2RealLoopError(f"BYO asset missing/empty for Isaac import: {src}")
    work_dir.mkdir(parents=True, exist_ok=True)
    suffix = src.suffix.lower()
    try:
        if suffix == ".urdf":
            from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

            cfg = UrdfConverterCfg(
                asset_path=str(src),
                usd_dir=str(work_dir),
                usd_file_name=f"{src.stem}.usd",
                force_usd_conversion=True,
            )
            converter = UrdfConverter(cfg)
        else:
            import isaaclab.sim as sim_utils
            from isaaclab.sim.converters import MeshConverter, MeshConverterCfg

            # Bake RigidBody/Collision/Mass APIs into the converted USD so the
            # mesh spawns as a physics rigid body (Isaac Lab's RigidObject
            # requires 'USD RigidBodyAPI' on the prim).
            cfg = MeshConverterCfg(
                asset_path=str(src),
                usd_dir=str(work_dir),
                usd_file_name=f"{src.stem}.usd",
                force_usd_conversion=True,
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True
                ),
            )
            converter = MeshConverter(cfg)
    except Exception as exc:  # noqa: BLE001 - surface converter import/runtime errors
        raise Sim2RealLoopError(
            f"Isaac mesh->USD conversion failed for {src.name}: {exc}"
        ) from exc
    usd_path = getattr(converter, "usd_path", "")
    if not usd_path or not Path(usd_path).is_file():
        raise Sim2RealLoopError(
            f"Isaac mesh->USD conversion produced no USD for {src.name}"
        )
    return usd_path


def _set_isaac_object_usd(env_cfg: Any, usd_path: str, *, scale: Any) -> None:
    """Point the lift task's manipuland spawn at a converted BYO USD asset."""

    import isaaclab.sim as sim_utils

    usd_scale: tuple[float, float, float]
    if isinstance(scale, (int, float)):
        usd_scale = (float(scale), float(scale), float(scale))
    elif isinstance(scale, (list, tuple)) and len(scale) == 3:
        usd_scale = (float(scale[0]), float(scale[1]), float(scale[2]))
    else:
        usd_scale = (1.0, 1.0, 1.0)
    obj_cfg = env_cfg.scene.object
    obj_cfg.spawn = sim_utils.UsdFileCfg(
        usd_path=usd_path,
        scale=usd_scale,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    )


def _isaac_robot_usd_override(robot: Any) -> str:
    """Resolve a BYO robot to a USD path for the Isaac lift task, or "".

    Default / ``stock_franka`` robots keep the task's built-in Franka (returns
    ""). A BYO URDF (or genesis_builtin URDF) is imported to USD via Isaac's
    URDF converter; an explicit USD is used as-is. Marks the robot ``loaded``
    on success. A robot that cannot be imported raises ``Sim2RealLoopError``
    (no silent fallback to Franka). Isaac cannot import MJCF, so that raises.
    """

    if robot is None:
        return ""
    from npa.genesis import robot_assets

    if robot.robot_source == robot_assets.ROBOT_SOURCE_STOCK_FRANKA:
        robot.loaded = True
        return ""
    if robot.robot_source == robot_assets.ROBOT_SOURCE_BYO_MJCF:
        raise Sim2RealLoopError(
            "robot_source=byo_mjcf is not importable by the Isaac backend; "
            "supply a URDF/USD robot, or run the Genesis backend (no fallback)."
        )
    if not robot.local_path:
        raise Sim2RealLoopError(
            f"BYO robot {robot.name!r} has no resolved local_path for Isaac import"
        )
    if robot.robot_source == robot_assets.ROBOT_SOURCE_BYO_USD:
        usd = robot.local_path
        if not Path(usd).is_file():
            raise Sim2RealLoopError(f"BYO robot USD missing: {usd}")
        robot.loaded = True
        return usd
    import tempfile as _tempfile

    convert_dir = Path(_tempfile.mkdtemp(prefix="isaac-robot-usd-"))
    usd = _isaac_import_mesh_to_usd(robot.local_path, work_dir=convert_dir)
    robot.loaded = True
    return usd


def _set_isaac_robot_usd(env_cfg: Any, usd_path: str, robot: Any) -> None:
    """Point the lift task's articulation spawn at a converted BYO robot USD.

    Overrides the robot articulation's spawn USD and best-effort widens the
    actuator joint-name expressions so a non-Franka arm's joints are actuated.
    Full joint/actuator remapping for an arbitrary arm is a follow-up; this
    establishes the BYO-robot import seam and proves the asset loads.
    """

    import isaaclab.sim as sim_utils

    robot_cfg = env_cfg.scene.robot
    spawn = getattr(robot_cfg, "spawn", None)
    new_spawn = sim_utils.UsdFileCfg(usd_path=usd_path)
    # Preserve articulation/rigid props from the task's spawn when available.
    for attr in ("articulation_props", "rigid_props", "activate_contact_sensors"):
        if hasattr(spawn, attr) and hasattr(new_spawn, attr):
            setattr(new_spawn, attr, getattr(spawn, attr))
    robot_cfg.spawn = new_spawn
    actuators = getattr(robot_cfg, "actuators", None)
    if isinstance(actuators, dict):
        for actuator in actuators.values():
            if hasattr(actuator, "joint_names_expr"):
                actuator.joint_names_expr = [".*"]


def _isaac_goal_distance(env_unwrapped: Any) -> Any:
    """Return per-env object->goal world distance for the lift task.

    Uses the command manager's desired object pose (robot-base frame) combined
    with the robot root pose to get the world goal, then the object's world
    position. Returns a 1-D CUDA tensor.
    """

    import torch

    scene = env_unwrapped.scene
    object_pos_w = scene["object"].data.root_pos_w[:, :3]
    command = env_unwrapped.command_manager.get_command("object_pose")
    robot = scene["robot"]
    root_pos_w = robot.data.root_state_w[:, :3]
    root_quat_w = robot.data.root_state_w[:, 3:7]
    try:
        from isaaclab.utils.math import combine_frame_transforms

        des_pos_w, _ = combine_frame_transforms(
            root_pos_w, root_quat_w, command[:, :3], command[:, 3:7]
        )
    except Exception:  # noqa: BLE001 - fall back to base-frame offset
        des_pos_w = root_pos_w + command[:, :3]
    return torch.norm(object_pos_w - des_pos_w, dim=-1)


def _isaac_adapter_actions(
    action_dim: int, adapter: dict[str, Any], *, n_envs: int, step: int, device: str
):
    """Deterministic adapter-biased actions for the Isaac manipulation rollout.

    The inner-loop adapter bias steers the arm action; a small seeded,
    decaying exploration term keeps the rollout non-degenerate. The gripper
    channel closes progressively, mirroring the Genesis adapter contract.
    """

    import torch

    bias_values = adapter.get("action_bias") or [0.0, 0.0, 0.0]
    bias = torch.zeros(action_dim, device=device, dtype=torch.float32)
    for i in range(min(action_dim, len(bias_values))):
        bias[i] = float(bias_values[i])
    actions = bias.unsqueeze(0).repeat(n_envs, 1)
    decay = 1.0 / (1.0 + 0.05 * step)
    explore = (
        0.15
        * decay
        * torch.sin(
            torch.arange(action_dim, device=device, dtype=torch.float32)
            * (step + 1)
            * 0.37
        )
    )
    actions = actions + explore.unsqueeze(0)
    if action_dim >= 1:
        # Last channel = gripper: open early, close as the episode progresses.
        actions[:, -1] = 1.0 if step < 30 else -1.0
    return torch.clamp(actions, -1.0, 1.0)


def _heldout_render_frames_enabled() -> bool:
    return _bool_value(os.environ.get("NPA_SIM2REAL_HELDOUT_RENDER_FRAMES", "1"))


def _heldout_render_step_indices(
    max_steps: int,
    *,
    max_frames: int = DEFAULT_HELDOUT_RENDER_FRAMES,
) -> set[int]:
    if max_steps <= 0 or max_frames <= 0:
        return set()
    if max_steps <= max_frames:
        return set(range(max_steps))
    stride = max(1, max_steps // max_frames)
    indices = list(range(0, max_steps, stride))
    if indices[-1] != max_steps - 1:
        indices.append(max_steps - 1)
    if len(indices) > max_frames:
        keep = {0, max_steps - 1}
        middle = indices[1:-1]
        pick_stride = max(1, len(middle) // max(1, max_frames - len(keep)))
        keep.update(middle[::pick_stride])
        indices = sorted(keep)
        while len(indices) > max_frames:
            indices.pop(len(indices) // 2)
    return set(indices)


def _attach_isaac_viz_camera(env_cfg: Any) -> None:
    import isaaclab.sim as sim_utils

    try:
        from isaaclab.sensors import TiledCameraCfg as _CameraCfg
    except ImportError:  # pragma: no cover
        from isaaclab.sensors import CameraCfg as _CameraCfg

    camera_cfg = _CameraCfg(
        prim_path="{ENV_REGEX_NS}/HeldoutVizCamera",
        offset=_CameraCfg.OffsetCfg(
            pos=(1.35, 1.05, 0.95),
            rot=(0.8829, 0.0, 0.4695, 0.0),
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        width=128,
        height=128,
    )
    setattr(env_cfg.scene, HELDOUT_VIZ_CAMERA_NAME, camera_cfg)


def _isaac_extract_rgb_frame(env: Any, *, env_index: int = 0) -> Any:
    import numpy as np

    unwrapped = getattr(env, "unwrapped", env)
    scene = getattr(unwrapped, "scene", None)
    if scene is None:
        return None
    camera = None
    for name in (HELDOUT_VIZ_CAMERA_NAME, "tiled_camera"):
        try:
            camera = scene[name]
            break
        except (KeyError, TypeError, AttributeError):
            continue
    if camera is None:
        sensors = getattr(scene, "sensors", None)
        if sensors is not None:
            for name in (HELDOUT_VIZ_CAMERA_NAME, "tiled_camera"):
                try:
                    camera = sensors[name]
                    break
                except (KeyError, TypeError, AttributeError):
                    continue
    if camera is None:
        return None
    output = getattr(getattr(camera, "data", None), "output", None)
    if not output or "rgb" not in output:
        return None
    rgb = output["rgb"]
    if hasattr(rgb, "detach"):
        rgb = rgb.detach()
    if hasattr(rgb, "cpu"):
        rgb = rgb.cpu()
    array = np.asarray(rgb)
    if array.ndim == 4:
        array = array[env_index]
    if array.ndim != 3 or array.shape[-1] < 3:
        return None
    frame = array[..., :3]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _write_render_png(path: Path, frame: Any) -> None:
    import struct
    import zlib

    import numpy as np

    array = np.asarray(frame, dtype=np.uint8)
    height, width = int(array.shape[0]), int(array.shape[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = bytearray()
    for row in range(height):
        raw.append(0)
        raw.extend(array[row].tobytes())

    def _chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack("!I", len(payload))
            + tag
            + payload
            + struct.pack("!I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    header = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", header)
    png += _chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    png += _chunk(b"IEND", b"")
    path.write_bytes(png)


def _build_heldout_render_manifest(
    renders_dir: Path,
    *,
    sim_backend: str,
    isaac_task: str = "",
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    if renders_dir.is_dir():
        for env_dir in sorted(path for path in renders_dir.iterdir() if path.is_dir()):
            frames = sorted(env_dir.glob("camera-*.png"))
            if not frames:
                continue
            views: dict[str, list[str]] = {}
            for frame in frames:
                parts = frame.stem.split("-")
                view = (
                    parts[1]
                    if len(parts) >= 3 and not parts[1].isdigit()
                    else "primary"
                )
                views.setdefault(view, []).append(frame.name)
            episodes.append(
                {
                    "env_id": env_dir.name,
                    "frames": views.get("primary", []),
                    "camera_views": views,
                }
            )
    return {
        "schema": SCHEMA_HELDOUT_RENDERS,
        "sim_backend": sim_backend,
        "isaac_task": isaac_task,
        "episodes": episodes,
    }


def _run_isaac_heldout_rollouts(
    envs: list[dict[str, Any]],
    *,
    inner_evidence: dict[str, Any],
    threshold: float,
    scene: Any = None,
    robot: Any = None,
    isaac_task: str = DEFAULT_ISAAC_TASK,
    renders_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Run the adapter policy through headless Isaac Lab held-out episodes.

    Mirrors ``_run_genesis_heldout_rollouts``: it returns the identical
    per-env metric schema (``env_id``/``score``/``success``/``details``) so
    ``report.json`` stays backend-agnostic. Stock runs use the built-in Isaac
    lift-cube manipuland (``asset_source=isaac_stock``); BYO runs import the
    customer mesh/URDF to USD and load it into the task (``asset_source=
    byo_mesh``). A BYO mesh that fails to import raises (no silent fallback).
    """

    from npa.genesis.scene_assets import ASSET_SOURCE_ISAAC_STOCK

    try:
        from isaaclab.app import AppLauncher
    except Exception as exc:  # noqa: BLE001
        raise Sim2RealLoopError(
            f"Isaac rollout eval requires isaaclab/Isaac Sim in the image: {exc}"
        ) from exc

    capture_renders = renders_dir is not None and _heldout_render_frames_enabled()
    if capture_renders:
        assert renders_dir is not None
        renders_dir.mkdir(parents=True, exist_ok=True)
    kit_args = os.environ.get(
        "NPA_ISAAC_KIT_ARGS", "--portable-root /tmp/npa-isaac-kit"
    )
    try:
        launcher = AppLauncher(
            headless=True,
            enable_cameras=capture_renders,
            kit_args=kit_args,
        )
    except TypeError:  # pragma: no cover
        launcher = AppLauncher(headless=True, kit_args=kit_args)
    simulation_app = launcher.app
    # Isaac Sim's SimulationApp.close() hard-terminates the process, so it must
    # NOT be called here (the held-out report has to be uploaded first). The
    # handle is stashed and closed by the component entrypoint after upload.
    global _ISAAC_SIMULATION_APP
    _ISAAC_SIMULATION_APP = simulation_app
    try:
        import torch
        import gymnasium as gym  # noqa: PLC0415
        import isaaclab_tasks  # noqa: F401, PLC0415
        from isaaclab_tasks.utils import parse_env_cfg
    except Exception as exc:  # noqa: BLE001
        raise Sim2RealLoopError(
            f"Isaac rollout eval requires gymnasium and isaaclab_tasks: {exc}"
        ) from exc
    if not torch.cuda.is_available():
        raise Sim2RealLoopError("Isaac rollout eval requires a CUDA GPU")
    device = "cuda:0"

    usd_override = ""
    manip_scale: Any = 1.0
    if scene is not None:
        manip = scene.manipuland()
        manip_scale = manip.scale
        if manip.asset_source == ASSET_SOURCE_ISAAC_STOCK:
            manip.loaded = True
            print(
                json.dumps(
                    {
                        "component": "heldout_eval",
                        "event": "isaac_scene_loading",
                        "asset_source": manip.asset_source,
                        "isaac_task": isaac_task,
                        "stock_asset": manip.builtin_path,
                    },
                    sort_keys=True,
                )
            )
        elif manip.is_mesh():
            import tempfile as _tempfile

            convert_dir = Path(_tempfile.mkdtemp(prefix="isaac-usd-"))
            usd_override = _isaac_import_mesh_to_usd(
                manip.local_path, work_dir=convert_dir
            )
            manip.loaded = True
            print(
                json.dumps(
                    {
                        "component": "heldout_eval",
                        "event": "isaac_byo_mesh_imported",
                        "asset_source": manip.asset_source,
                        "manipuland": manip.name,
                        "local_path": manip.local_path,
                        "sha256": manip.sha256,
                        "usd_path": usd_override,
                    },
                    sort_keys=True,
                )
            )

    robot_usd_override = _isaac_robot_usd_override(robot)
    if robot_usd_override:
        print(
            json.dumps(
                {
                    "component": "heldout_eval",
                    "event": "isaac_byo_robot_imported",
                    "robot_source": robot.robot_source,
                    "robot_name": robot.name,
                    "ee_link": robot.ee_link,
                    "dof_count": robot.dof_count,
                    "local_path": robot.local_path,
                    "sha256": robot.sha256,
                    "usd_path": robot_usd_override,
                },
                sort_keys=True,
            )
        )

    adapter = _policy_adapter_from_inner_evidence(inner_evidence)
    batch_size = max(1, int(os.environ.get("NPA_SIM2REAL_ISAAC_BATCH_SIZE", "8")))
    max_steps = max(1, int(os.environ.get("NPA_SIM2REAL_ISAAC_MAX_STEPS", "120")))
    reward_norm = float(os.environ.get("NPA_SIM2REAL_ISAAC_REWARD_NORM", "20.0"))
    success_dist = float(os.environ.get("NPA_SIM2REAL_ISAAC_SUCCESS_DIST", "0.05"))
    render_steps = _heldout_render_step_indices(max_steps) if capture_renders else set()
    per_env: list[dict[str, Any]] = []
    for start in range(0, len(envs), batch_size):
        batch = envs[start : start + batch_size]
        seed = int(batch[0].get("seed") or (42 + start))
        torch.manual_seed(seed)
        env_cfg = parse_env_cfg(isaac_task, device=device, num_envs=len(batch))
        if capture_renders and start == 0:
            _attach_isaac_viz_camera(env_cfg)
        if usd_override:
            _set_isaac_object_usd(env_cfg, usd_override, scale=manip_scale)
        if robot_usd_override:
            _set_isaac_robot_usd(env_cfg, robot_usd_override, robot)
        env = gym.make(isaac_task, cfg=env_cfg)
        action_dim = int(env.action_space.shape[-1])
        obs, _ = env.reset()
        n = len(batch)
        max_reward = torch.full((n,), -1.0e9, device=device)
        final_distance = torch.full((n,), 1.0e9, device=device)
        if capture_renders and start == 0 and 0 in render_steps:
            assert renders_dir is not None
            frame = _isaac_extract_rgb_frame(env, env_index=0)
            if frame is not None:
                env_id = str(batch[0].get("env_id") or "heldout-0000")
                _write_render_png(
                    renders_dir / env_id / "camera-000.png",
                    frame,
                )
        for step in range(max_steps):
            actions = _isaac_adapter_actions(
                action_dim, adapter, n_envs=n, step=step, device=device
            )
            obs, reward, terminated, truncated, _ = env.step(actions)
            if capture_renders and start == 0 and (step + 1) in render_steps:
                assert renders_dir is not None
                frame = _isaac_extract_rgb_frame(env, env_index=0)
                if frame is not None:
                    env_id = str(batch[0].get("env_id") or "heldout-0000")
                    _write_render_png(
                        renders_dir / env_id / f"camera-{step + 1:03d}.png",
                        frame,
                    )
            reward_t = torch.as_tensor(
                reward, device=device, dtype=torch.float32
            ).reshape(-1)
            max_reward = torch.maximum(max_reward, reward_t)
            final_distance = _isaac_goal_distance(env.unwrapped).reshape(-1).detach()
            done = torch.as_tensor(terminated, device=device).reshape(
                -1
            ) | torch.as_tensor(truncated, device=device).reshape(-1)
            if bool(done.all()):
                break
        success = final_distance < success_dist
        batch_successes = int(success.sum().item())
        print(
            json.dumps(
                {
                    "component": "heldout_eval",
                    "event": "isaac_rollout_batch_complete",
                    "batch_start": start,
                    "env_count": n,
                    "successes": batch_successes,
                    "max_steps": max_steps,
                    "isaac_task": isaac_task,
                },
                sort_keys=True,
            )
        )
        for index, env_record in enumerate(batch):
            dist = float(final_distance[index].detach().item())
            reward_value = float(max_reward[index].detach().item())
            env_success = bool(success[index].detach().item())
            distance_score = max(0.0, min(1.0, 1.0 - dist / 0.5))
            reward_score = max(0.0, min(1.0, reward_value / reward_norm))
            score = _heldout_env_score(
                distance_score, reward_score, env_success=env_success
            )
            per_env.append(
                {
                    "env_id": str(
                        env_record.get("env_id") or f"heldout-{start + index:04d}"
                    ),
                    "score": score,
                    "success": env_success,
                    "details": {
                        "source": "isaac_lift_env_goal_distance",
                        "sim_backend": SIM_BACKEND_ISAAC,
                        "isaac_task": isaac_task,
                        "seed": env_record.get("seed"),
                        "target_threshold": success_dist,
                        "final_target_distance": round(dist, 6),
                        "max_reward": round(reward_value, 6),
                        "steps": max_steps,
                        "policy_adapter": adapter,
                        "threshold": threshold,
                    },
                }
            )
        env.close()
    return per_env


def _close_isaac_app() -> None:
    """Close the stashed Isaac Sim app, if any (hard-terminates the process).

    Called by the component entrypoint only after the held-out report has been
    written and uploaded. No-op for the Genesis backend.
    """

    global _ISAAC_SIMULATION_APP
    app = _ISAAC_SIMULATION_APP
    _ISAAC_SIMULATION_APP = None
    if app is not None:
        try:
            app.close()
        except Exception:  # noqa: BLE001
            logging.getLogger(__name__).debug("suppressed exception", exc_info=True)


def _policy_adapter_from_inner_evidence(
    inner_evidence: dict[str, Any],
) -> dict[str, Any]:
    iterations = inner_evidence.get("iterations") or []
    update: dict[str, Any] = {}
    if iterations and isinstance(iterations[-1], dict):
        update = iterations[-1].get("update") or {}
    action = update.get("policy_output_after") or [0.0, 0.0, 0.0]
    reward_head = float(update.get("reward_head_after") or 0.0)
    reward_trend = [float(item) for item in (inner_evidence.get("reward_trend") or [])]
    return {
        "action_bias": [float(value) for value in action[:3]],
        "reward_head_after": round(reward_head, 6),
        "reward_trend": [round(value, 6) for value in reward_trend],
        "source": "inner_evidence.update.policy_output_after",
    }


def _adapter_policy_actions(obs: dict[str, Any], adapter: dict[str, Any], *, step: int):
    import torch

    ee_pos = obs["ee_pos"]
    cube_pos = obs["object_pose"][:, :3]
    target_pos = obs["goal_position"]
    contacts = obs["contact_flags"].sum(dim=-1, keepdim=True) > 0.5
    to_cube = cube_pos - ee_pos
    to_target = target_pos - cube_pos
    bias_values = adapter.get("action_bias") or [0.0, 0.0, 0.0]
    bias = torch.tensor(
        bias_values[:3], device=ee_pos.device, dtype=ee_pos.dtype
    ).unsqueeze(0)
    approach_delta = to_cube * 0.45 + bias * 0.02
    place_delta = (to_target + (cube_pos - ee_pos) * 0.25) * 0.35 + bias * 0.02
    delta_xyz = torch.where(contacts, place_delta, approach_delta)
    dist_to_cube = torch.norm(to_cube, dim=-1, keepdim=True)
    should_close = contacts | (dist_to_cube < 0.065) | (step > 40)
    gripper = torch.where(
        should_close,
        torch.full_like(dist_to_cube, -1.0),
        torch.full_like(dist_to_cube, 1.0),
    )
    return torch.cat([delta_xyz, gripper], dim=-1)


def _resolve_env_records_s3_uri(uri: str) -> str:
    """Normalize train/heldout env URIs to the envs.jsonl object key."""

    uri = str(uri or "").strip()
    if not uri.startswith("s3://"):
        return uri
    if uri.endswith(".jsonl"):
        return uri
    base = uri.rstrip("/")
    leaf = base.rsplit("/", 1)[-1]
    if leaf in {
        "heldout",
        "gold-heldout",
        "validation",
        "train",
        "raw",
    } or uri.endswith("/"):
        return f"{base}/envs.jsonl"
    return uri


def _download_s3_env_records(
    client: StorageClient,
    uri: str,
    dest_path: Path,
    *,
    attempts: int | None = None,
) -> None:
    """Download sibling env records with retries and a stable local filename."""

    resolved = _resolve_env_records_s3_uri(uri)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    max_attempts = max(
        1,
        int(
            attempts
            if attempts is not None
            else os.environ.get("NPA_SIM2REAL_COMPONENT_DOWNLOAD_RETRIES", "12")
        ),
    )
    for attempt in range(max_attempts):
        if dest_path.exists():
            dest_path.unlink()
        client.download_path(resolved, str(dest_path))
        if dest_path.exists() and dest_path.stat().st_size > 0:
            return
        if attempt + 1 < max_attempts:
            time.sleep(min(2**attempt, 8))
    raise Sim2RealLoopError(
        f"env records not available at {resolved} after {max_attempts} download attempts"
    )


def _find_component_input_file(root: Path, filename: str) -> Path:
    if root.is_file() and root.name == filename:
        return root
    candidates = sorted(root.rglob(filename))
    if not candidates:
        raise Sim2RealLoopError(f"component input did not include {filename}")
    return candidates[0]


def _read_component_env_records(root: Path) -> list[dict[str, Any]]:
    if root.is_file():
        if root.suffix == ".jsonl":
            return _read_jsonl(root)
        payload = json.loads(root.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("envs"), list):
            return [dict(item) for item in payload["envs"]]
        if isinstance(payload, list):
            return [dict(item) for item in payload]
        return []
    jsonl_files = sorted(root.rglob("*.jsonl"))
    if jsonl_files:
        records: list[dict[str, Any]] = []
        for path in jsonl_files:
            records.extend(_read_jsonl(path))
        return records
    json_files = sorted(root.rglob("*.json"))
    for path in json_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("envs"), list):
            return [dict(item) for item in payload["envs"]]
        if isinstance(payload, list):
            return [dict(item) for item in payload]
    return []


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


__all__ = [
    "_adapter_policy_actions",
    "_attach_isaac_viz_camera",
    "_build_heldout_render_manifest",
    "_close_isaac_app",
    "_download_s3_env_records",
    "_find_component_input_file",
    "_heldout_render_frames_enabled",
    "_heldout_render_step_indices",
    "_isaac_adapter_actions",
    "_isaac_extract_rgb_frame",
    "_isaac_goal_distance",
    "_isaac_import_mesh_to_usd",
    "_isaac_robot_usd_override",
    "_policy_adapter_from_inner_evidence",
    "_read_component_env_records",
    "_read_jsonl",
    "_resolve_cosmos_reason_model_id",
    "_resolve_env_records_s3_uri",
    "_rollout_image_paths",
    "_run_cosmos_reason_vlm",
    "_run_genesis_heldout_rollouts",
    "_run_isaac_heldout_rollouts",
    "_set_isaac_object_usd",
    "_set_isaac_robot_usd",
    "_task_description_from_manifest",
    "_write_render_png",
]
