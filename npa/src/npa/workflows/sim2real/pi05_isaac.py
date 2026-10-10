"""Real Isaac collector for the pi0.5 released-surface pick-and-place task.

The expert commands the Franka end effector through Isaac's differential-IK
action term.  It never writes object pose or velocity.  Every admitted training
row contains synchronous exterior/wrist RTX pixels and the arm controller's
actual absolute joint target, rather than the expert's Cartesian command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Mapping

from npa.workflows.sim2real.pi05_contract import (
    CONTROL_HZ,
    EPISODE_SCHEMA,
    JOINT_NAMES,
    MAX_GRIPPER_WIDTH_M,
    OBJECT_INITIAL_X_RANGE_M,
    OBJECT_INITIAL_Y_RANGE_M,
    PROXIMAL_FINGER_CLEARANCE_M,
    WRIST_MOUNT_QUATERNION_XYZW,
    WRIST_MOUNT_TARGET_LOCAL_M,
    WRIST_MOUNT_TRANSLATION_M,
    build_contract,
    release_width_threshold,
)


EXPERT_BASE_TASK_ID = "Isaac-Lift-Cube-Franka-IK-Rel-v0"
EXPERT_SURFACE_TASK_ID = "NPA-Pi05-SurfacePlace-Franka-IK-Rel-v0"
POLICY_BASE_TASK_ID = "Isaac-Lift-Cube-Franka-v0"
POLICY_SURFACE_TASK_ID = "NPA-Pi05-SurfacePlace-Franka-v0"
DESTINATION_RADIUS_M = 0.05
DESTINATION_LABEL = "pi05_destination_marker"
SUPPORT_SURFACE_SIZE_M = (1.0, 0.8, 0.05)
SUPPORT_SURFACE_CENTER_M = (0.5, 0.0, -0.025)
EXTERIOR_CAMERA_POSITION_M = (0.65, 0.65, 0.85)
EXTERIOR_CAMERA_TARGET_M = (0.50, 0.0, 0.03)
DOWNWARD_HAND_QUATERNION_XYZW = (1.0, 0.0, 0.0, 0.0)
RESET_SETTLE_CONTROL_STEPS = 2
EXPECTED_SEMANTIC_LABELS = frozenset(
    {
        "pi05_target_object",
        DESTINATION_LABEL,
        "pi05_left_finger",
        "pi05_right_finger",
    }
)


class CollectionFailure(RuntimeError):
    """Collector failure with a stable evidence phase."""

    def __init__(self, message: str, *, phase: str) -> None:
        super().__init__(message)
        self.phase = phase
        self.runtime_app: Any | None = None


class _EnvironmentSetupError(RuntimeError):
    """Carry the live app so failure evidence can be written before shutdown."""

    def __init__(self, error: Exception, app: Any) -> None:
        super().__init__(str(error))
        self.original, self.app = error, app


def support_surface_contract() -> dict[str, Any]:
    """Return the exact one-body support geometry used by expert and policy."""

    return {
        "source": "isaaclab_task_owned_kinematic_cuboid",
        "prim_path": "{ENV_REGEX_NS}/Table",
        "size_m": list(SUPPORT_SURFACE_SIZE_M),
        "center_m": list(SUPPORT_SURFACE_CENTER_M),
        "top_z_m": SUPPORT_SURFACE_CENTER_M[2] + SUPPORT_SURFACE_SIZE_M[2] / 2,
        "rigid_body_count": 1,
        "kinematic": True,
        "contact_filter": "{ENV_REGEX_NS}/Table",
    }


def register_surface_task(gym: Any, *, base_id: str, task_id: str) -> None:
    """Register an owned surface-task ID over an explicitly configured base."""

    try:
        gym.spec(task_id)
        return
    except gym.error.Error:
        base = gym.spec(base_id)
    gym.register(id=task_id, entry_point=base.entry_point, disable_env_checker=True)


def configure_surface_goal(cfg: Any, target: tuple[float, float, float]) -> None:
    """Replace the stock airborne goal with one object center on the support."""

    command = cfg.commands.object_pose
    # The task renders its own stable destination cue. Gen3's inherited
    # command-frame PointInstancer is unnecessary and failed on first reset.
    command.debug_vis = False
    command.resampling_time_range = (1.0e9, 1.0e9)
    command.ranges.pos_x = (target[0], target[0])
    command.ranges.pos_y = (target[1], target[1])
    command.ranges.pos_z = (target[2], target[2])


def configure_object_reset(cfg: Any) -> None:
    """Bind the inherited relative reset event to the declared world XY range."""

    event = cfg.events.reset_object_position
    params = getattr(event, "params", None)
    if not isinstance(params, dict) or "pose_range" not in params:
        raise RuntimeError("reset_object_position lacks a mutable pose range")
    default = tuple(float(value) for value in cfg.scene.object.init_state.pos[:2])
    if default != (0.5, 0.0):
        raise RuntimeError("pi0.5 reset calibration requires the pinned object origin")
    params["pose_range"] = {
        "x": tuple(value - default[0] for value in OBJECT_INITIAL_X_RANGE_M),
        "y": tuple(value - default[1] for value in OBJECT_INITIAL_Y_RANGE_M),
        "z": (0.0, 0.0),
    }


def configure_destination_marker(
    cfg: Any, target: tuple[float, float, float], *, support_z: float = 0.0
) -> None:
    """Add the shared rendered, non-colliding cyan destination disc."""

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    cfg.scene.pi05_destination = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Pi05DestinationMarker",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(target[0], target[1], support_z + 0.0005)
        ),
        spawn=sim_utils.CylinderCfg(
            radius=DESTINATION_RADIUS_M,
            height=0.001,
            axis="Z",
            collision_props=None,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.85, 0.85),
                emissive_color=(0.0, 0.08, 0.08),
            ),
            semantic_tags=[("class", DESTINATION_LABEL)],
        ),
    )


def configure_support_surface(cfg: Any) -> None:
    """Replace the multi-body stock table with one filtered support body."""

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    cfg.scene.table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=SUPPORT_SURFACE_CENTER_M),
        spawn=sim_utils.CuboidCfg(
            size=SUPPORT_SURFACE_SIZE_M,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.24, 0.28, 0.32)
            ),
            activate_contact_sensors=True,
        ),
    )


def configure_hand_origin_frames(cfg: Any, *, require_ik: bool) -> None:
    """Make the observed and IK-controlled end-effector frame panda_hand."""

    targets = list(getattr(cfg.scene.ee_frame, "target_frames", ()))
    if len(targets) != 1 or not str(targets[0].prim_path).endswith("/panda_hand"):
        raise RuntimeError("ee_frame must expose exactly the panda_hand target")
    _zero_offset_cfg(targets[0].offset, "ee_frame panda_hand")
    if require_ik:
        action = cfg.actions.arm_action
        if getattr(action, "body_name", None) != "panda_hand":
            raise RuntimeError("expert IK action must control panda_hand")
        _zero_offset_cfg(action.body_offset, "expert IK panda_hand")


def configure_robot_plant(cfg: Any) -> None:
    """Use one pinned Franka rigid-body/actuator plant for expert and policy."""

    from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

    prim_path = str(cfg.scene.robot.prim_path)
    cfg.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path=prim_path)


def _zero_offset_cfg(offset: Any, label: str) -> None:
    if offset is None:
        raise RuntimeError(f"{label} does not expose an offset configuration")
    offset.pos = (0.0, 0.0, 0.0)
    offset.rot = (0.0, 0.0, 0.0, 1.0)


def _tag_spawned_finger_prims(root: Any) -> dict[str, str]:
    """Author Gen3 LabelsAPI finger tags before physics/Fabric initialization."""

    from isaaclab.sim.utils.semantics import add_labels
    from pxr import Usd

    tagged = {}
    for side in ("left", "right"):
        name = f"panda_{side}finger"
        matches = [prim for prim in Usd.PrimRange(root) if prim.GetName() == name]
        if len(matches) != 1:
            raise RuntimeError(f"spawned Franka must contain exactly one {name} prim")
        add_labels(matches[0], [f"pi05_{side}_finger"], instance_name="class")
        tagged[side] = str(matches[0].GetPath())
    return tagged


def configure_finger_semantics(cfg: Any) -> None:
    """Wrap robot spawning so finger schemas exist before physics views."""

    original = cfg.scene.robot.spawn.func

    def spawn_with_finger_tags(
        prim_path: str,
        spawn_cfg: Any,
        translation: Any = None,
        orientation: Any = None,
    ) -> Any:
        root = original(
            prim_path, spawn_cfg, translation=translation, orientation=orientation
        )
        _tag_spawned_finger_prims(root)
        return root

    cfg.scene.robot.spawn.func = spawn_with_finger_tags


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _matrix_to_xyzw(matrix: Any) -> tuple[float, float, float, float]:
    """Convert one proper 3x3 rotation matrix into a normalized XYZW quaternion."""

    import numpy as np

    value = np.asarray(matrix, dtype=float)
    trace = float(np.trace(value))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.asarray(
            [
                (value[2, 1] - value[1, 2]) / scale,
                (value[0, 2] - value[2, 0]) / scale,
                (value[1, 0] - value[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(value)))
        j, k = (index + 1) % 3, (index + 2) % 3
        scale = 2.0 * np.sqrt(1.0 + value[index, index] - value[j, j] - value[k, k])
        quaternion = np.zeros(4)
        quaternion[index] = 0.25 * scale
        quaternion[j] = (value[j, index] + value[index, j]) / scale
        quaternion[k] = (value[k, index] + value[index, k]) / scale
        quaternion[3] = (value[k, j] - value[j, k]) / scale
    quaternion /= np.linalg.norm(quaternion)
    quaternion *= 1.0 if quaternion[3] >= 0 else -1.0
    return tuple(float(item) for item in quaternion)


def _ros_look_at_xyzw(position: Any, target: Any) -> tuple[float, float, float, float]:
    """Aim ROS camera +Z at target while keeping image up near world +Z."""

    import numpy as np

    forward = np.asarray(target, dtype=float) - np.asarray(position, dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return _matrix_to_xyzw(np.column_stack((right, down, forward)))


def _camera_common() -> dict[str, Any]:
    import isaaclab.sim as sim_utils
    from isaaclab_physx.renderers import IsaacRtxRendererCfg

    return {
        "data_types": ["rgb", "semantic_segmentation"],
        "width": 224,
        "height": 224,
        "update_period": 0.0,
        "renderer_cfg": IsaacRtxRendererCfg(
            semantic_filter=["class"],
            colorize_semantic_segmentation=False,
        ),
        "update_latest_camera_pose": True,
        "spawn": sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            horizontal_aperture=20.955,
            clipping_range=(0.03, 10.0),
        ),
    }


def _camera_config(env_cfg: Any) -> None:
    """Install exterior and live-hand-driven RTX camera sensors."""

    from isaaclab.sensors import TiledCameraCfg

    exterior_position = EXTERIOR_CAMERA_POSITION_M
    env_cfg.scene.pi05_exterior = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Pi05ExteriorCamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=exterior_position,
            rot=_ros_look_at_xyzw(exterior_position, EXTERIOR_CAMERA_TARGET_M),
            convention="ros",
        ),
        **_camera_common(),
    )
    # The selected Franka finger-link reference lies near hand-local +Z.  A
    # side bracket keeps the closest 48/60/72 mm cube faces beyond the 30 mm
    # near clip and aims ROS +Z at that measured reference so object and both
    # contact surfaces can coexist in the close-grasp view.
    # Gen3's Isaac RTX renderer reads camera transforms from USD while Fabric
    # owns the moving articulation pose.  A world-pose-driven camera avoids a
    # stale authored hand transform; every capture republishes the calibrated
    # live hand mount to both USD/RTX and CameraData.
    env_cfg.scene.pi05_wrist = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Pi05WristCamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 1.0),
            rot=WRIST_MOUNT_QUATERNION_XYZW,
            convention="ros",
        ),
        **_camera_common(),
    )


def _contact_config(env_cfg: Any) -> None:
    """Enable filtered bilateral finger/object and object/table reporting."""

    from isaaclab.sensors import ContactSensorCfg

    env_cfg.scene.robot.spawn.activate_contact_sensors = True
    env_cfg.scene.object.spawn.activate_contact_sensors = True
    env_cfg.scene.table.spawn.activate_contact_sensors = True
    env_cfg.scene.left_object_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        history_length=3,
    )
    env_cfg.scene.right_object_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        history_length=3,
    )
    env_cfg.scene.object_support_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"],
        history_length=3,
    )


def _physics_randomization(env_cfg: Any) -> None:
    from isaaclab.envs import mdp
    from isaaclab.managers import EventTermCfg, SceneEntityCfg

    env_cfg.events.pi05_object_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.85, 1.15),
            "operation": "scale",
        },
    )
    env_cfg.events.pi05_object_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.45, 1.25),
            "dynamic_friction_range": (0.45, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 32,
        },
    )


def _rgb(native: Any, sensor_name: str) -> Any:
    import numpy as np

    value = native.scene[sensor_name].data.output["rgb"]
    if hasattr(value, "torch"):
        value = value.torch
    frame = value[0, ..., :3].detach().cpu().numpy()
    if frame.shape != (224, 224, 3) or frame.dtype != np.uint8:
        raise RuntimeError(
            f"{sensor_name} did not produce uint8[224,224,3]: {frame.shape} {frame.dtype}"
        )
    if int(frame.max()) - int(frame.min()) < 8:
        raise RuntimeError(f"{sensor_name} produced a blank RTX observation")
    return frame.copy()


def _proxy_torch(value: Any) -> Any:
    """Resolve Gen3 CameraData ProxyArray before tensor indexing."""

    return value.torch if hasattr(value, "torch") else value


def _rgb_receipt(frame: Any, capture_time: float, observation_time: float) -> dict:
    import numpy as np

    value = np.ascontiguousarray(frame)
    return {
        "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        "capture_time_s": capture_time,
        "observation_time_s": observation_time,
        "age_s": observation_time - capture_time,
        "minimum": int(value.min()),
        "maximum": int(value.max()),
        "standard_deviation": float(value.std()),
        "channel_order": "RGB",
    }


def _semantic_pixel_count(native: Any, sensor_name: str, label: str) -> int:
    import numpy as np

    sensor = native.scene[sensor_name]
    segmentation, labels = _semantic_encoding(sensor, sensor_name)
    verified = getattr(native, "_npa_pi05_verified_semantic_paths", {})
    if label not in EXPECTED_SEMANTIC_LABELS or label not in verified:
        raise RuntimeError(f"{sensor_name} requested an unverified semantic label")
    target_ids = [code for code, value in labels.items() if value == label]
    if not target_ids:
        return 0
    if len(target_ids) != 1:
        raise RuntimeError(
            f"{sensor_name} semantic output has ambiguous {label} labels"
        )
    return int(np.isin(segmentation.reshape(224, 224), target_ids).sum())


def _packed_rgba_code(value: Any, sensor_name: str) -> int:
    """Decode one renderer-declared RGBA tuple into its int32 pixel code."""

    import ast

    try:
        rgba = ast.literal_eval(str(value))
    except (SyntaxError, ValueError) as error:
        raise RuntimeError(f"{sensor_name} semantic metadata key is invalid") from error
    if not isinstance(rgba, tuple) or len(rgba) != 4:
        raise RuntimeError(f"{sensor_name} semantic metadata must use RGBA tuples")
    if any(type(channel) is not int or not 0 <= channel <= 255 for channel in rgba):
        raise RuntimeError(f"{sensor_name} semantic RGBA channels are invalid")
    return int.from_bytes(bytes(rgba), byteorder="little", signed=True)


def _semantic_metadata(sensor: Any, sensor_name: str) -> dict[int, str]:
    """Validate the pinned RTX packed-RGBA metadata as a strict bijection."""

    from collections.abc import Mapping

    raw = sensor.data.info.get("semantic_segmentation", {}).get("idToLabels", {})
    if not isinstance(raw, Mapping) or not raw:
        raise RuntimeError(f"{sensor_name} semantic metadata is missing")
    labels: dict[int, str] = {}
    for key, metadata in raw.items():
        if not isinstance(metadata, Mapping) or set(metadata) != {"class"}:
            raise RuntimeError(f"{sensor_name} semantic class metadata is invalid")
        label = metadata["class"]
        if not isinstance(label, str) or not label:
            raise RuntimeError(f"{sensor_name} semantic class label is invalid")
        code = _packed_rgba_code(key, sensor_name)
        if code in labels:
            raise RuntimeError(f"{sensor_name} semantic metadata is ambiguous")
        if label in labels.values():
            raise RuntimeError(f"{sensor_name} semantic class mapping is ambiguous")
        labels[code] = label
    return labels


def _semantic_encoding(sensor: Any, sensor_name: str) -> tuple[Any, dict[int, str]]:
    """Return pixels only after exact metadata-to-packed-code validation."""

    import numpy as np

    value = _proxy_torch(sensor.data.output["semantic_segmentation"])
    segmentation = np.asarray(value[0].detach().cpu().numpy())
    if segmentation.dtype != np.int32 or segmentation.shape != (224, 224, 1):
        raise RuntimeError(
            f"{sensor_name} semantic pixels must be int32[224,224,1], "
            f"got {segmentation.shape} {segmentation.dtype}"
        )
    labels = _semantic_metadata(sensor, sensor_name)
    observed = {int(code) for code in np.unique(segmentation)}
    if not observed.issubset(labels):
        raise RuntimeError(f"{sensor_name} semantic pixels contain unmapped codes")
    return segmentation, labels


def _semantic_encoding_receipt(native: Any, sensor_name: str) -> dict[str, Any]:
    """Describe the exact non-color-heuristic renderer encoding in one frame."""

    import numpy as np

    sensor = native.scene[sensor_name]
    segmentation, labels = _semantic_encoding(sensor, sensor_name)
    return {
        "encoding": "int32_little_endian_packed_rgba_metadata_bijection",
        "shape": list(segmentation.shape),
        "dtype": str(segmentation.dtype),
        "metadata_sha256": _sha(sorted(labels.items())),
        "mapped_code_count": len(labels),
        "observed_code_count": int(np.unique(segmentation).size),
    }


def _target_pixel_count(native: Any, sensor_name: str) -> int:
    return _semantic_pixel_count(native, sensor_name, "pi05_target_object")


def _destination_pixel_count(native: Any, sensor_name: str) -> int:
    return _semantic_pixel_count(native, sensor_name, DESTINATION_LABEL)


def _finger_pixel_counts(native: Any, sensor_name: str) -> dict[str, int]:
    return {
        side: _semantic_pixel_count(native, sensor_name, f"pi05_{side}_finger")
        for side in ("left", "right")
    }


def _camera_pose_sample(native: Any) -> dict[str, list[float]]:
    robot = native.scene["robot"]
    hand_index = robot.body_names.index("panda_hand")
    renderer_position, renderer_quaternion = _renderer_usd_pose(
        native.scene["pi05_wrist"]
    )

    def values(tensor: Any) -> list[float]:
        tensor = _proxy_torch(tensor)
        return tensor.detach().cpu().numpy().astype(float).tolist()

    return {
        "hand_position_m": values(_proxy_torch(robot.data.body_pos_w)[0, hand_index]),
        "hand_quaternion_xyzw": values(
            _proxy_torch(robot.data.body_quat_w)[0, hand_index]
        ),
        "wrist_camera_position_m": values(native.scene["pi05_wrist"].data.pos_w)[0],
        "wrist_camera_quaternion_xyzw": values(
            native.scene["pi05_wrist"].data.quat_w_ros
        )[0],
        "wrist_renderer_usd_position_m": renderer_position.tolist(),
        "wrist_renderer_usd_quaternion_xyzw": renderer_quaternion.tolist(),
        "exterior_camera_position_m": values(native.scene["pi05_exterior"].data.pos_w)[
            0
        ],
        "exterior_camera_quaternion_xyzw": values(
            native.scene["pi05_exterior"].data.quat_w_world
        )[0],
    }


def _xyzw_to_wxyz(quaternion: Any) -> Any:
    import numpy as np

    value = np.asarray(quaternion, dtype=float)
    if value.shape != (4,):
        raise RuntimeError(f"camera quaternion must be xyzw[4], got {value.shape}")
    return np.asarray([value[3], value[0], value[1], value[2]])


def _relative_camera_quaternion(camera_xyzw: Any, hand_xyzw: Any) -> Any:
    import numpy as np

    hand = _xyzw_to_wxyz(hand_xyzw)
    hand_conjugate = np.asarray([hand[0], -hand[1], -hand[2], -hand[3]])
    camera = _xyzw_to_wxyz(camera_xyzw)
    w1, x1, y1, z1 = hand_conjugate
    w2, x2, y2, z2 = camera
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _multiply_xyzw(left: Any, right: Any) -> Any:
    """Compose two finite unit XYZW quaternions."""

    import numpy as np

    values = []
    for name, quaternion in (("left", left), ("right", right)):
        value = np.asarray(quaternion, dtype=float)
        norm = float(np.linalg.norm(value))
        if value.shape != (4,) or not np.isfinite(value).all() or norm < 1.0e-8:
            raise RuntimeError(f"{name} camera quaternion is invalid")
        values.append(value / norm)
    lx, ly, lz, lw = values[0]
    rx, ry, rz, rw = values[1]
    return np.asarray(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ]
    )


def _ros_to_opengl_xyzw(quaternion: Any) -> Any:
    """Apply the pinned Gen3 ROS-to-OpenGL camera-frame conversion."""

    return _multiply_xyzw(quaternion, (1.0, 0.0, 0.0, 0.0))


def _rotate_vector(quaternion: Any, vector: Any) -> Any:
    import numpy as np

    q = np.asarray(quaternion, dtype=float)
    v = np.asarray(vector, dtype=float)
    q_vector = q[1:]
    return v + 2.0 * (
        q[0] * np.cross(q_vector, v) + np.cross(q_vector, np.cross(q_vector, v))
    )


def _finger_link_paths(native: Any) -> dict[str, str]:
    paths = getattr(native.scene["robot"].root_physx_view, "link_paths", None)
    if not paths:
        raise RuntimeError("Franka articulation did not expose runtime link paths")
    paths = list(paths[0] if isinstance(paths[0], (list, tuple)) else paths)
    result = {}
    for side in ("left", "right"):
        suffix = f"/panda_{side}finger"
        matches = [str(path) for path in paths if str(path).endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(f"expected one runtime link path ending {suffix}")
        result[side] = matches[0]
    return result


def _runtime_stage() -> Any:
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Isaac runtime did not expose the active USD stage")
    return stage


def _scene_cfg_prim_path(native: Any, name: str) -> str:
    """Resolve an authored prim expression from the environment configuration."""

    scene_cfg = getattr(getattr(native, "cfg", None), "scene", None)
    asset_cfg = getattr(scene_cfg, name, None)
    path = str(getattr(asset_cfg, "prim_path", "") or "").strip()
    if not path:
        raise RuntimeError(f"scene config lacks a prim path for {name}")
    return path


def _install_finger_semantics(native: Any) -> dict[str, str]:
    """Verify pre-physics finger labels without mutating the live USD schema."""

    from isaaclab.sim.utils.semantics import get_labels

    stage = _runtime_stage()
    paths = _finger_link_paths(native)
    for side, path in paths.items():
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"missing finger prim for semantic tag: {path}")
        labels = get_labels(prim).get("class", [])
        if f"pi05_{side}_finger" not in labels:
            raise RuntimeError(
                f"finger semantic tag was not authored before physics: {path}"
            )
    return paths


def _verify_scene_semantics(native: Any) -> dict[str, str]:
    """Verify expected tags independently of per-view renderer metadata."""

    from isaaclab.sim import find_first_matching_prim
    from isaaclab.sim.utils.semantics import get_labels

    paths = _install_finger_semantics(native)
    expected = {
        "object": "pi05_target_object",
        "pi05_destination": DESTINATION_LABEL,
    }
    for name, label in expected.items():
        prim = find_first_matching_prim(_scene_cfg_prim_path(native, name))
        if not prim or not prim.IsValid():
            raise RuntimeError(f"missing prim for semantic tag: {name}")
        if label not in get_labels(prim).get("class", []):
            raise RuntimeError(f"missing expected semantic tag: {label}")
        paths[name] = str(prim.GetPath())
    verified = {
        "pi05_left_finger": paths["left"],
        "pi05_right_finger": paths["right"],
        "pi05_target_object": paths["object"],
        DESTINATION_LABEL: paths["pi05_destination"],
    }
    native._npa_pi05_verified_semantic_paths = verified
    return verified


def _finger_local_bounds(native: Any) -> dict[str, Any]:
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    result, stage = {}, _runtime_stage()
    for side, path in _finger_link_paths(native).items():
        result[side] = _untransformed_aligned_bounds(cache, stage.GetPrimAtPath(path))
    return result


def _untransformed_aligned_bounds(cache: Any, prim: Any) -> tuple[Any, Any]:
    """Return geometry in the rigid-link frame, excluding its authored pose."""

    aligned = cache.ComputeUntransformedBound(prim).ComputeAlignedBox()
    return aligned.GetMin(), aligned.GetMax()


def _aligned_extent(bound: Any) -> list[float]:
    import numpy as np

    aligned = bound.ComputeAlignedBox()
    extent = np.asarray(aligned.GetMax(), dtype=float) - np.asarray(
        aligned.GetMin(), dtype=float
    )
    if extent.shape != (3,) or not np.isfinite(extent).all() or (extent <= 0).any():
        raise RuntimeError("object USD geometry has invalid measured extent")
    return extent.astype(float).tolist()


def _object_asset_geometry(native: Any, scale: tuple[float, ...]) -> dict[str, Any]:
    """Measure source and effective object extents from the composed runtime USD."""

    from isaaclab.sim import find_first_matching_prim
    from pxr import Usd, UsdGeom

    prim = find_first_matching_prim(_scene_cfg_prim_path(native, "object"))
    if not prim or not prim.IsValid():
        raise RuntimeError("object prim is missing from the composed runtime stage")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    source_bound = cache.ComputeUntransformedBound(prim).ComputeAlignedBox()
    source = _aligned_extent(cache.ComputeUntransformedBound(prim))
    effective = _aligned_extent(cache.ComputeLocalBound(prim))
    source_center = list(map(float, source_bound.GetMidpoint()))
    if any(abs(value) > 1.0e-4 for value in source_center):
        raise RuntimeError("object USD geometry must be centered on its rigid root")
    if effective[1] > MAX_GRIPPER_WIDTH_M:
        raise RuntimeError("measured object grasp width exceeds the Franka gripper")
    return {
        "source_untransformed_extent_m": source,
        "effective_local_extent_m": effective,
        "declared_spawn_scale": list(map(float, scale)),
        "grasp_width_axis": "object_local_y",
        "grasp_width_m": effective[1],
        "support_height_m": effective[2],
        "source_untransformed_center_m": source_center,
        "measurement": "UsdGeom.BBoxCache composed runtime prim",
    }


def _bounds_corners(bounds: Any) -> Any:
    import numpy as np

    minimum, maximum = (np.asarray(value, dtype=float) for value in bounds)
    return np.asarray(
        [
            [x, y, z]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ]
    )


def _finger_bounds(native: Any) -> dict[str, Any]:
    import numpy as np

    robot = native.scene["robot"]
    positions = _proxy_torch(robot.data.body_pos_w)[0].detach().cpu().numpy()
    quaternions = _proxy_torch(robot.data.body_quat_w)[0].detach().cpu().numpy()
    result = {}
    for side, local_bounds in _finger_local_bounds(native).items():
        index = robot.body_names.index(f"panda_{side}finger")
        quaternion = _xyzw_to_wxyz(quaternions[index])
        world = np.asarray(
            [
                _rotate_vector(quaternion, point) + positions[index]
                for point in _bounds_corners(local_bounds)
            ]
        )
        result[side] = {
            "local_bounds_m": [list(map(float, value)) for value in local_bounds],
            "world_corners_m": world,
            "geometry_source": "usd_compute_untransformed_bound_link_frame",
            "pose_source": "live_articulation_body_pos_w_body_quat_w_xyzw",
        }
    return result


def _finger_points_in_hand_frame(native: Any) -> tuple[Any, Any]:
    import numpy as np

    robot = native.scene["robot"]
    hand_index = list(robot.body_names).index("panda_hand")
    hand_position = _proxy_torch(robot.data.body_pos_w)[0, hand_index]
    hand_position = hand_position.detach().cpu().numpy()
    hand_xyzw = _proxy_torch(robot.data.body_quat_w)[0, hand_index]
    hand_wxyz = _xyzw_to_wxyz(hand_xyzw.detach().cpu().numpy())
    world = np.concatenate(
        [value["world_corners_m"] for value in _finger_bounds(native).values()]
    )
    inverse = hand_wxyz * np.asarray([1.0, -1.0, -1.0, -1.0])
    local = np.asarray(
        [_rotate_vector(inverse, point - hand_position) for point in world]
    )
    return world, local


def _finger_grasp_reference_geometry(native: Any) -> dict[str, Any]:
    """Measure the complete finger-link interval in the live hand frame."""

    import numpy as np

    world, local_points = _finger_points_in_hand_frame(native)
    local = np.mean(local_points, axis=0)
    interval = [float(local_points[:, 2].min()), float(local_points[:, 2].max())]
    if (
        np.linalg.norm(local[:2]) > 0.03
        or not 0.04 <= interval[0] < local[2] < interval[1] <= 0.16
    ):
        raise RuntimeError("measured Franka grasp reference is outside hand geometry")
    return {
        "hand_to_finger_link_centroid_local_m": local.astype(float).tolist(),
        "finger_link_hand_axis_interval_m": interval,
        "measured_finger_link_centroid_world_m": np.mean(world, axis=0)
        .astype(float)
        .tolist(),
        "hand_pose_source": "ArticulationData.body_pos_w/body_quat_w XYZW",
        "finger_geometry_source": "live link poses plus untransformed USD bounds",
        "contact_geometry_claim": (
            "complete-link interval, not pad geometry; bilateral contact required"
        ),
    }


def _object_grasp_geometry(geometry: dict[str, Any], height: float) -> dict[str, Any]:
    import math

    result = dict(geometry)
    centroid = float(result["hand_to_finger_link_centroid_local_m"][2])
    proximal = float(result["finger_link_hand_axis_interval_m"][0])
    shift = height / 2.0 + proximal + PROXIMAL_FINGER_CLEARANCE_M - centroid
    if not all(math.isfinite(value) for value in (height, centroid, proximal, shift)):
        raise RuntimeError("size-aware grasp geometry is not finite")
    if height <= 0.0 or shift <= 0.0:
        raise RuntimeError("size-aware grasp geometry is outside the object")
    result.update(
        {
            "object_height_m": height,
            "proximal_finger_clearance_m": PROXIMAL_FINGER_CLEARANCE_M,
            "contact_reference_above_object_center_m": shift,
            "contact_hand_origin_above_object_center_m": shift + centroid,
        }
    )
    return result


def _finger_in_frustum(native: Any) -> dict[str, Any]:
    import numpy as np

    sensor = native.scene["pi05_wrist"].data
    camera_position = _proxy_torch(sensor.pos_w)[0].detach().cpu().numpy()
    camera_quaternion = _xyzw_to_wxyz(
        _proxy_torch(sensor.quat_w_ros)[0].detach().cpu().numpy()
    )
    inverse = np.asarray([camera_quaternion[0], *(-camera_quaternion[1:])], dtype=float)
    intrinsics = _proxy_torch(sensor.intrinsic_matrices)[0].detach().cpu().numpy()
    result = {}
    for side, geometry in _finger_bounds(native).items():
        world_corners = geometry["world_corners_m"]
        camera = np.asarray(
            [
                _rotate_vector(inverse, point - camera_position)
                for point in world_corners
            ]
        )
        pixels = (intrinsics @ camera.T).T
        pixels[:, :2] /= np.maximum(pixels[:, 2:], 1.0e-12)
        inside = (camera[:, 2] > 0.03) & (pixels[:, 0] >= 0) & (pixels[:, 0] < 224)
        inside &= (pixels[:, 1] >= 0) & (pixels[:, 1] < 224)
        result[side] = {
            "in_frustum": bool(np.any(inside)),
            "asset_local_bounds_m": geometry["local_bounds_m"],
            "live_world_bounds_m": [
                np.min(world_corners, axis=0).tolist(),
                np.max(world_corners, axis=0).tolist(),
            ],
            "pose_source": geometry["pose_source"],
            "projected_corner_count": int(np.sum(inside)),
            "evidence_scope": "asset_bounds_in_frustum_not_rendered_visibility",
        }
    return result


def _camera_angular_drift(traces: list[dict[str, Any]]) -> float:
    import numpy as np

    relative_quaternions = np.asarray(
        [
            _relative_camera_quaternion(
                row["camera_pose"]["wrist_camera_quaternion_xyzw"],
                row["camera_pose"]["hand_quaternion_xyzw"],
            )
            for row in traces
        ]
    )
    relative_quaternions *= np.sign(relative_quaternions[:, :1] + 1.0e-12)
    return float(
        np.max(
            2.0
            * np.arccos(
                np.clip(
                    np.abs(relative_quaternions @ relative_quaternions[0]), 0.0, 1.0
                )
            )
        )
    )


def _camera_motion_metrics(traces: list[dict[str, Any]]) -> dict[str, float]:
    import numpy as np

    hand = np.asarray([row["camera_pose"]["hand_position_m"] for row in traces])
    wrist = np.asarray(
        [row["camera_pose"]["wrist_camera_position_m"] for row in traces]
    )
    exterior = np.asarray(
        [row["camera_pose"]["exterior_camera_position_m"] for row in traces]
    )
    wrist_world_motion = float(np.linalg.norm(wrist - wrist[0], axis=1).max())
    hand_world_motion = float(np.linalg.norm(hand - hand[0], axis=1).max())
    relative_drift = float(np.ptp(np.linalg.norm(wrist - hand, axis=1)))
    angular_drift = _camera_angular_drift(traces)
    exterior_drift = float(np.linalg.norm(exterior - exterior[0], axis=1).max())
    renderer = _renderer_motion_metrics(traces, wrist)
    metrics = {
        "wrist_world_motion_m": wrist_world_motion,
        "hand_world_motion_m": hand_world_motion,
        "wrist_mount_distance_drift_m": relative_drift,
        "wrist_mount_angular_drift_rad": angular_drift,
        "exterior_world_drift_m": exterior_drift,
        **renderer,
    }
    _validate_camera_motion_metrics(metrics)
    return metrics


def _validate_camera_motion_metrics(metrics: Mapping[str, float]) -> None:
    """Fail closed when physical, CameraData, USD, or static-view motion drifts."""

    moving = (
        "wrist_world_motion_m",
        "hand_world_motion_m",
        "wrist_renderer_world_motion_m",
    )
    if any(metrics[name] < 0.05 for name in moving):
        raise RuntimeError("wrist camera/hand did not move through the manipulation")
    limits = {
        "wrist_mount_distance_drift_m": 0.002,
        "wrist_mount_angular_drift_rad": 0.01,
        "exterior_world_drift_m": 1.0e-4,
        "wrist_renderer_position_error_m": 0.002,
        "wrist_renderer_orientation_error_rad": 0.01,
    }
    if any(metrics[name] > limit for name, limit in limits.items()):
        raise RuntimeError("camera extrinsics violate moving/static-view contract")


def _renderer_motion_metrics(
    traces: list[dict[str, Any]], wrist: Any
) -> dict[str, float]:
    """Compare the USD pose consumed by RTX with CameraData on every row."""

    import numpy as np

    renderer = np.asarray(
        [row["camera_pose"]["wrist_renderer_usd_position_m"] for row in traces]
    )
    orientation = max(
        _quaternion_angle_xyzw(
            row["camera_pose"]["wrist_renderer_usd_quaternion_xyzw"],
            row["camera_pose"]["wrist_camera_quaternion_xyzw"],
        )
        for row in traces
    )
    return {
        "wrist_renderer_world_motion_m": float(
            np.linalg.norm(renderer - renderer[0], axis=1).max()
        ),
        "wrist_renderer_position_error_m": float(
            np.linalg.norm(renderer - wrist, axis=1).max()
        ),
        "wrist_renderer_orientation_error_rad": float(orientation),
    }


def _phase_visibility(traces: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    phase_visibility = {}
    for phase in sorted({str(row.get("phase", "policy_query")) for row in traces}):
        rows = [row for row in traces if row.get("phase", "policy_query") == phase]
        phase_visibility[phase] = {
            "samples": len(rows),
            "wrist_visible_samples": sum(
                row["wrist_target_pixels"] >= 4 for row in rows
            ),
            "exterior_visible_samples": sum(
                row["exterior_target_pixels"] >= 4 for row in rows
            ),
            "destination_exterior_visible_samples": sum(
                row["exterior_destination_pixels"] >= 16 for row in rows
            ),
            "both_fingers_rendered_samples": sum(
                all(count >= 4 for count in row["wrist_finger_pixels"].values())
                for row in rows
            ),
            "occlusion_permitted": phase == "retreat",
        }
    return phase_visibility


def _require_expert_phase_visibility(
    visibility: Mapping[str, Mapping[str, Any]],
) -> None:
    required = {"approach", "grasp", "lift", "transport", "lower", "release"}
    missing = [
        phase
        for phase in required
        if visibility.get(phase, {}).get("wrist_visible_samples", 0) == 0
        or visibility.get(phase, {}).get("exterior_visible_samples", 0) == 0
        or visibility.get(phase, {}).get("destination_exterior_visible_samples", 0) == 0
    ]
    if missing:
        raise RuntimeError(f"target visibility missing for phases: {missing}")
    finger_phases = {"grasp", "lift", "transport", "lower", "release"}
    missing = [
        phase
        for phase in finger_phases
        if visibility.get(phase, {}).get("both_fingers_rendered_samples", 0) == 0
    ]
    if missing:
        raise RuntimeError(
            f"both rendered finger masks missing from wrist view: {missing}"
        )


def _verify_camera_motion(
    traces: list[dict[str, Any]], *, require_expert_phases: bool = True
) -> dict[str, Any]:
    metrics = _camera_motion_metrics(traces)
    visibility = _phase_visibility(traces)
    if require_expert_phases:
        _require_expert_phase_visibility(visibility)
    return {**metrics, "phase_visibility": visibility}


def _action_layout(native: Any) -> tuple[int, int, int]:
    names = list(native.action_manager.active_terms)
    dimensions = [int(v) for v in native.action_manager.action_term_dim]
    if "arm_action" not in names or "gripper_action" not in names:
        raise RuntimeError("Isaac task lacks named arm_action/gripper_action terms")
    arm_index, grip_index = names.index("arm_action"), names.index("gripper_action")
    arm_start = sum(dimensions[:arm_index])
    grip_start = sum(dimensions[:grip_index])
    if dimensions[arm_index] != 6 or dimensions[grip_index] != 1:
        raise RuntimeError("expert requires 6D differential IK plus binary gripper")
    return sum(dimensions), arm_start, grip_start


def _expert_action_scale(native: Any) -> Any:
    """Read the actual six-axis scaling applied by the IK action term."""

    import numpy as np

    term = native.action_manager.get_term("arm_action")
    value = _proxy_torch(getattr(term, "_scale", None))
    if value is None or not hasattr(value, "detach"):
        raise RuntimeError("expert IK term does not expose its action scale")
    scale = value[0].detach().cpu().numpy().astype(float)
    if scale.shape != (6,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise RuntimeError("expert IK action scale must be six finite positive values")
    return scale


def _offset_receipt(offset: Any, label: str) -> dict[str, Any]:
    import numpy as np

    position = np.asarray(getattr(offset, "pos", ()), dtype=float)
    quaternion = np.asarray(getattr(offset, "rot", ()), dtype=float)
    if position.shape != (3,) or not np.allclose(position, 0.0, atol=1.0e-9):
        raise RuntimeError(f"{label} is not configured at panda_hand origin")
    if quaternion.shape != (4,) or not np.allclose(
        np.abs(np.dot(quaternion, [0.0, 0.0, 0.0, 1.0])), 1.0, atol=1.0e-9
    ):
        raise RuntimeError(f"{label} rotation is not identity XYZW")
    return {"position_m": position.tolist(), "quaternion_xyzw": quaternion.tolist()}


def _hand_origin_control_receipt(native: Any, *, require_ik: bool) -> dict[str, Any]:
    """Verify live sensor and configured control points coincide with panda_hand."""

    import numpy as np

    robot = native.scene["robot"]
    hand_index = list(robot.body_names).index("panda_hand")
    hand = _proxy_torch(robot.data.body_pos_w)[0, hand_index].detach().cpu().numpy()
    hand_q = _proxy_torch(robot.data.body_quat_w)[0, hand_index].detach().cpu().numpy()
    sensor = native.scene["ee_frame"].data
    sensor_pos = _proxy_torch(sensor.target_pos_w)[0, 0, :3].detach().cpu().numpy()
    sensor_q = _proxy_torch(sensor.target_quat_w)[0, 0, :4].detach().cpu().numpy()
    position_error = float(np.linalg.norm(sensor_pos - hand))
    orientation_error = _quaternion_angle_xyzw(sensor_q, hand_q)
    if position_error > 1.0e-4 or orientation_error > 1.0e-3:
        raise RuntimeError("ee_frame sensor does not coincide with live panda_hand")
    return _control_frame_receipt(
        native,
        hand,
        hand_q,
        sensor_pos,
        sensor_q,
        position_error,
        orientation_error,
        require_ik,
    )


def _control_frame_receipt(
    native: Any,
    hand: Any,
    hand_q: Any,
    sensor: Any,
    sensor_q: Any,
    position_error: float,
    orientation_error: float,
    require_ik: bool,
) -> dict[str, Any]:
    targets = list(native.cfg.scene.ee_frame.target_frames)
    receipt = {
        "target_prim_path": str(targets[0].prim_path),
        "sensor_offset": _offset_receipt(targets[0].offset, "runtime ee_frame"),
        "live_hand_position_m": hand.tolist(),
        "live_hand_quaternion_xyzw": hand_q.tolist(),
        "sensor_position_m": sensor.tolist(),
        "sensor_quaternion_xyzw": sensor_q.tolist(),
        "sensor_hand_position_error_m": position_error,
        "sensor_hand_orientation_error_rad": orientation_error,
        "controlled_point_world_m": hand.tolist(),
        "controlled_frame": "panda_hand_origin",
    }
    if require_ik:
        term = native.action_manager.get_term("arm_action")
        receipt["ik_body_name"] = str(term.cfg.body_name)
        receipt["ik_body_offset"] = _offset_receipt(
            term.cfg.body_offset, "runtime expert IK"
        )
    return receipt


def _actuator_values(value: Any, label: str) -> list[float]:
    """Read finite unique values from the live actuator tensor."""

    import numpy as np

    tensor = _proxy_torch(value)
    if tensor is None or not hasattr(tensor, "detach"):
        raise RuntimeError(f"runtime actuator lacks {label}")
    values = tensor.detach().cpu().numpy().astype(float).reshape(-1)
    if not values.size or not np.isfinite(values).all():
        raise RuntimeError(f"runtime actuator {label} is invalid")
    return sorted(float(item) for item in np.unique(values))


def _robot_plant_receipt(native: Any) -> dict[str, Any]:
    """Record the actual pi0.5 Franka rigid-body and actuator plant."""

    robot = native.scene["robot"]
    rigid = robot.cfg.spawn.rigid_props
    actuators = {
        name: {
            "joint_names": list(actuator.joint_names),
            "stiffness": _actuator_values(actuator.stiffness, f"{name} stiffness"),
            "damping": _actuator_values(actuator.damping, f"{name} damping"),
        }
        for name, actuator in sorted(robot.actuators.items())
    }
    if not actuators:
        raise RuntimeError("runtime Franka exposes no actuators")
    gravity = _live_gravity_receipt(robot)
    configured = bool(rigid.disable_gravity)
    if not configured or not gravity["all_bodies_disabled"]:
        raise RuntimeError("pi0.5 Franka gravity disabling is not active")
    return {
        "schema": "npa.sim2real.pi05.robot_plant.v1",
        "source": "isaaclab_assets.FRANKA_PANDA_HIGH_PD_CFG",
        "configured_disable_robot_gravity": configured,
        "observed_disable_gravity": gravity,
        "actuators": actuators,
    }


def _live_gravity_receipt(robot: Any) -> dict[str, Any]:
    """Read per-body disable-gravity flags from the pinned PhysX view."""

    import numpy as np

    view = getattr(robot, "root_view", None)
    if view is None:
        raise RuntimeError("runtime Franka lacks its PhysX articulation root view")
    getter = getattr(view, "get_disable_gravities", None)
    if not callable(getter):
        raise RuntimeError("PhysX articulation view lacks get_disable_gravities")
    raw = getter()
    values, converter = _native_tensor_to_torch(raw)
    array = values.detach().cpu().numpy()
    expected_shape = (int(robot.num_instances), int(robot.num_bodies))
    if array.shape != expected_shape:
        raise RuntimeError("runtime Franka disable-gravity shape is invalid")
    if not np.isfinite(array.astype(float)).all() or not np.isin(array, [0, 1]).all():
        raise RuntimeError("runtime Franka disable-gravity flags are invalid")
    flags = array.astype(bool)
    return {
        "shape": list(flags.shape),
        "unique": sorted(int(item) for item in np.unique(flags)),
        "all_bodies_disabled": bool(flags.all()),
        "view_type": _qualified_type(view),
        "return_type": _qualified_type(raw),
        "converter": converter,
    }


def _native_tensor_to_torch(value: Any) -> tuple[Any, str]:
    """Resolve documented Torch, ProxyArray, or raw Warp tensor output."""

    if hasattr(value, "torch"):
        return value.torch, "proxyarray.torch"
    if hasattr(value, "detach"):
        return value, "torch"
    try:
        import warp as wp

        converted = wp.to_torch(value)
    except (ImportError, TypeError, AttributeError) as error:
        raise RuntimeError("unsupported PhysX native tensor return type") from error
    if not hasattr(converted, "detach"):
        raise RuntimeError("Warp conversion did not produce a Torch tensor")
    return converted, "warp.to_torch"


def _qualified_type(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _axis_angle_error_xyzw(current: Any, target: Any) -> Any:
    """Match pinned Isaac compute_pose_error for world-frame XYZW rotations."""

    import numpy as np

    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    inverse = np.asarray([-current[0], -current[1], -current[2], current[3]])
    error = _multiply_xyzw(target, inverse)
    error *= 1.0 if error[3] >= 0.0 else -1.0
    magnitude = float(np.linalg.norm(error[:3]))
    if magnitude < 1.0e-8:
        return np.zeros(3, dtype=float)
    return error[:3] * (2.0 * np.arctan2(magnitude, error[3]) / magnitude)


def _controller_target(native: Any) -> Any:
    import numpy as np

    robot = native.scene["robot"]
    names = list(robot.joint_names)
    indices = [names.index(name) for name in JOINT_NAMES]
    value = (
        _proxy_torch(robot.data.joint_pos_target)[0, indices]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=True)
    )
    if value.shape != (7,) or not np.isfinite(value).all():
        raise RuntimeError(f"invalid absolute controller target: {value.shape}")
    return value


def _gripper_width(native: Any) -> float:
    import numpy as np

    robot = native.scene["robot"]
    names = list(robot.joint_names)
    indices = [
        names.index(name) for name in ("panda_finger_joint1", "panda_finger_joint2")
    ]
    q = _proxy_torch(robot.data.joint_pos)[0, indices].detach().cpu().numpy()
    width = float(np.sum(q))
    if not 0.0 <= width <= MAX_GRIPPER_WIDTH_M + 0.005:
        raise RuntimeError(f"measured physical finger width is invalid: {width}")
    return width


def _finger_controller_target(native: Any) -> Any:
    import numpy as np

    robot = native.scene["robot"]
    names = list(robot.joint_names)
    indices = [
        names.index(name) for name in ("panda_finger_joint1", "panda_finger_joint2")
    ]
    target = (
        _proxy_torch(robot.data.joint_pos_target)[0, indices].detach().cpu().numpy()
    )
    if target.shape != (2,) or not np.isfinite(target).all():
        raise RuntimeError("Isaac did not expose finite two-finger controller targets")
    return target.astype(np.float32, copy=True)


def _first_array(value: Any) -> Any:
    """Return the first row of a Gen3 ProxyArray as a host array."""

    return _proxy_torch(value)[0].detach().cpu().numpy().astype(float, copy=True)


def _wrist_expected_pose(native: Any) -> tuple[Any, Any]:
    """Compose the declared local wrist mount with the live hand pose."""

    import numpy as np

    robot = native.scene["robot"]
    hand_index = robot.body_names.index("panda_hand")
    hand_position = _first_array(robot.data.body_pos_w)[hand_index]
    hand_xyzw = _first_array(robot.data.body_quat_w)[hand_index]
    offset = _rotate_vector(
        _xyzw_to_wxyz(hand_xyzw), np.asarray(WRIST_MOUNT_TRANSLATION_M)
    )
    world_xyzw = _multiply_xyzw(hand_xyzw, WRIST_MOUNT_QUATERNION_XYZW)
    return hand_position + offset, world_xyzw


def _wrist_sensor_state(sensor: Any) -> dict[str, Any]:
    """Record the pinned SensorBase and renderer state without raw pixels."""

    view = sensor._view
    return {
        "sensor_initialized": bool(getattr(sensor, "_is_initialized", False)),
        "sensor_timestamp_s": _sensor_clock_scalar(sensor._timestamp),
        "sensor_last_update_s": _sensor_clock_scalar(sensor._timestamp_last_update),
        "sensor_is_outdated": bool(_sensor_clock_scalar(sensor._is_outdated)),
        "frame": int(_proxy_torch(sensor.frame)[0].detach().cpu().item()),
        "renderer_class": type(getattr(sensor, "_renderer", None)).__name__,
        "render_data_initialized": getattr(sensor, "_render_data", None) is not None,
        "frame_view_class": type(view).__name__,
        "fabric_enabled": bool(getattr(view, "_use_fabric", False)),
        "fabric_initialized": bool(getattr(view, "_fabric_initialized", False)),
        "fabric_usd_sync_done": bool(getattr(view, "_fabric_usd_sync_done", False)),
    }


def _wrist_raw_pose_snapshot(native: Any) -> dict[str, Any]:
    """Read cached, Fabric, RTX/USD, and declared live-mount poses."""

    sensor = native.scene["pi05_wrist"]
    data = sensor.data
    world_position, world_quaternion = sensor._view.get_world_poses()
    local_position, local_quaternion = sensor._view.get_local_poses()
    expected_position, expected_quaternion = _wrist_expected_pose(native)
    renderer_position, renderer_quaternion = _renderer_usd_pose(sensor)
    frame_view_raw = _first_array(world_quaternion)
    return {
        "camera_data_position_m": _first_array(data.pos_w).tolist(),
        "camera_data_quaternion_xyzw": _first_array(data.quat_w_ros).tolist(),
        "frame_view_world_position_m": _first_array(world_position).tolist(),
        "frame_view_world_quaternion_raw": frame_view_raw.tolist(),
        "frame_view_world_quaternion_xyzw": _ros_to_opengl_xyzw(
            frame_view_raw
        ).tolist(),
        "authored_local_translation_m": _first_array(local_position).tolist(),
        "authored_local_quaternion_raw": _first_array(local_quaternion).tolist(),
        "renderer_usd_position_m": renderer_position.tolist(),
        "renderer_usd_quaternion_xyzw": renderer_quaternion.tolist(),
        "expected_live_position_m": expected_position.tolist(),
        "expected_live_quaternion_xyzw": expected_quaternion.tolist(),
        "sensor_state": _wrist_sensor_state(sensor),
    }


def _wrist_prim_identity(sensor: Any) -> dict[str, str]:
    """Return the one runtime camera prim and its immediate parent."""

    prim_paths = list(getattr(sensor._view, "prim_paths", ()) or ())
    if len(prim_paths) != 1:
        raise RuntimeError("wrist camera FrameView must resolve exactly one prim")
    path = str(prim_paths[0])
    parent = path.rsplit("/", 1)[0]
    return {"runtime_prim_path": path, "runtime_parent_prim": parent}


def _validate_wrist_parent(receipt: Mapping[str, Any]) -> None:
    """Require the explicit world-pose RTX bridge, never a stale hand child."""

    path = str(receipt["runtime_prim_path"])
    parent = str(receipt["runtime_parent_prim"])
    if not path.endswith("/Pi05WristCamera") or "/Robot/" in path:
        raise RuntimeError("wrist renderer camera must use the world-pose bridge")
    if not parent.rsplit("/", 1)[-1].startswith("env_"):
        raise RuntimeError("wrist renderer camera must be rooted in one environment")


def _renderer_usd_pose(sensor: Any) -> tuple[Any, Any]:
    """Read the USD pose Isaac RTX actually consumes, in ROS XYZW form."""

    import numpy as np

    hook = getattr(sensor, "_npa_renderer_usd_pose", None)
    if hook is not None:
        position, quaternion = hook()
        return np.asarray(position, dtype=float), np.asarray(quaternion, dtype=float)
    import omni.usd
    from pxr import Usd, UsdGeom

    path = str(sensor._view.prim_paths[0])
    stage = omni.usd.get_context().get_stage()
    prim = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    matrix = np.asarray(prim.ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    position = matrix[3, :3]
    opengl = _matrix_to_xyzw(matrix[:3, :3].T)
    return position, np.asarray(_ros_to_opengl_xyzw(opengl))


def _world_pose_matrix(position: Any, ros_quaternion: Any) -> Any:
    """Build the row-vector USD camera matrix for one ROS world pose."""

    import numpy as np

    opengl = _xyzw_to_wxyz(_ros_to_opengl_xyzw(ros_quaternion))
    rotation = np.column_stack(
        [_rotate_vector(opengl, axis) for axis in np.eye(3, dtype=float)]
    )
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3], matrix[:3, 3] = rotation, np.asarray(position, dtype=float)
    return matrix.T


def _publish_renderer_usd_pose(sensor: Any, position: Any, quaternion: Any) -> None:
    """Publish the calibrated live-hand pose to the USD-backed RTX camera."""

    hook = getattr(sensor, "_npa_publish_renderer_pose", None)
    if hook is not None:
        hook(position, quaternion)
        return
    import omni.usd
    from pxr import Gf, UsdGeom

    op = getattr(sensor, "_npa_pi05_renderer_transform", None)
    if op is None:
        path = str(sensor._view.prim_paths[0])
        stage = omni.usd.get_context().get_stage()
        xform = UsdGeom.Xformable(stage.GetPrimAtPath(path))
        xform.ClearXformOpOrder()
        xform.SetResetXformStack(True)
        op = xform.AddTransformOp(opSuffix="npaLiveWrist")
        sensor._npa_pi05_renderer_transform = op
    op.Set(Gf.Matrix4d(_world_pose_matrix(position, quaternion).tolist()))


def _pose_distance(snapshot: Mapping[str, Any], key: str) -> float:
    import numpy as np

    actual = np.asarray(snapshot[key], dtype=float)
    expected = np.asarray(snapshot["expected_live_position_m"], dtype=float)
    return float(np.linalg.norm(actual - expected))


def _quaternion_angle_xyzw(actual: Any, expected: Any) -> float:
    """Return a sign-invariant angle after strict quaternion validation."""

    import numpy as np

    normalized = []
    for name, quaternion in (("actual", actual), ("expected", expected)):
        value = np.asarray(quaternion, dtype=float)
        norm = float(np.linalg.norm(value))
        if value.shape != (4,) or not np.isfinite(value).all() or norm < 1.0e-8:
            raise RuntimeError(f"{name} camera quaternion is invalid")
        normalized.append(value / norm)
    dot = float(np.clip(abs(np.dot(*normalized)), 0.0, 1.0))
    angle = float(2.0 * np.arccos(dot))
    if not np.isfinite(angle):
        raise RuntimeError("camera orientation error is not finite")
    return angle


def _camera_orientation_error(snapshot: Mapping[str, Any]) -> float:
    """Return the shortest cached-camera error from the expected ROS pose."""

    return _quaternion_angle_xyzw(
        snapshot["camera_data_quaternion_xyzw"],
        snapshot["expected_live_quaternion_xyzw"],
    )


def _renderer_orientation_error(snapshot: Mapping[str, Any]) -> float:
    """Return the RTX/USD camera error from the expected ROS pose."""

    return _quaternion_angle_xyzw(
        snapshot["renderer_usd_quaternion_xyzw"],
        snapshot["expected_live_quaternion_xyzw"],
    )


def _apply_wrist_world_pose(native: Any, position: Any, quaternion: Any) -> None:
    """Publish one pose to RTX/USD and Fabric, then force a fresh frame."""

    import torch

    sensor = native.scene["pi05_wrist"]
    frame_before = int(_proxy_torch(sensor.frame)[0].detach().cpu().item())
    _publish_renderer_usd_pose(sensor, position, quaternion)
    sensor.set_world_poses(
        torch.as_tensor(position, dtype=torch.float32).reshape(1, 3),
        torch.as_tensor(quaternion, dtype=torch.float32).reshape(1, 4),
        convention="ros",
    )
    _pump_rtx_after_pose_write(sensor)
    _force_wrist_render(sensor, frame_before)


def _pump_rtx_after_pose_write(sensor: Any) -> None:
    """Make the USD-backed RTX view consume the new pose without physics."""

    hook = getattr(sensor, "_npa_render_after_pose", None)
    if hook is not None:
        hook()
        return
    from isaaclab.sim import SimulationContext

    simulation = SimulationContext.instance()
    if simulation is None:
        raise RuntimeError("wrist RTX pose publication requires SimulationContext")
    simulation.render()


def _force_wrist_render(sensor: Any, frame_before: int | None = None) -> None:
    """Force and verify the pinned public Camera render-refresh boundary."""

    if frame_before is None:
        frame_before = int(_proxy_torch(sensor.frame)[0].detach().cpu().item())
    sensor.update(0.0, force_recompute=True)
    frame_after = int(_proxy_torch(sensor.frame)[0].detach().cpu().item())
    if frame_after <= frame_before:
        raise RuntimeError("wrist camera renderer frame did not advance after refresh")


def _refresh_wrist_camera_pose(native: Any, snapshot: Mapping[str, Any]) -> str:
    """Synchronize the live mount across the pinned USD and Fabric views."""

    sensor = native.scene["pi05_wrist"]
    if (
        _pose_distance(snapshot, "camera_data_position_m") <= 0.002
        and _pose_distance(snapshot, "frame_view_world_position_m") <= 0.002
        and _pose_distance(snapshot, "renderer_usd_position_m") <= 0.002
        and _camera_orientation_error(snapshot) <= 0.01
        and _renderer_orientation_error(snapshot) <= 0.01
    ):
        _force_wrist_render(sensor)
        return "usd_fabric_pose_already_current_and_rendered"
    _apply_wrist_world_pose(
        native,
        snapshot["expected_live_position_m"],
        snapshot["expected_live_quaternion_xyzw"],
    )
    return "usd_and_fabric_world_pose_synchronized_from_live_hand_mount"


def _validate_refreshed_wrist_pose(snapshot: Mapping[str, Any]) -> None:
    """Require cached, Fabric and rendered USD poses to match the live mount."""

    for key in (
        "camera_data_position_m",
        "frame_view_world_position_m",
        "renderer_usd_position_m",
    ):
        if _pose_distance(snapshot, key) > 0.002:
            raise RuntimeError("wrist camera live pose did not match the hand mount")
    if _camera_orientation_error(snapshot) > 0.01:
        raise RuntimeError("wrist camera live orientation did not match the hand mount")
    if (
        _quaternion_angle_xyzw(
            snapshot["frame_view_world_quaternion_xyzw"],
            snapshot["expected_live_quaternion_xyzw"],
        )
        > 0.01
    ):
        raise RuntimeError("wrist Fabric orientation did not match the hand mount")
    if _renderer_orientation_error(snapshot) > 0.01:
        raise RuntimeError("wrist RTX/USD orientation did not match the hand mount")


def _wrist_render_receipt(native: Any) -> dict[str, Any]:
    """Bind the post-publication renderer frame to its RGB bytes."""

    import numpy as np

    sensor = native.scene["pi05_wrist"]
    frame = np.ascontiguousarray(_rgb(native, "pi05_wrist"))
    position, quaternion = _renderer_usd_pose(sensor)
    return {
        "renderer_frame": int(_proxy_torch(sensor.frame)[0].detach().cpu().item()),
        "rgb_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
        "rgb_shape": list(frame.shape),
        "rgb_dtype": str(frame.dtype),
        "sensor_timestamp_s": _sensor_clock_scalar(sensor._timestamp_last_update),
        "renderer_usd_position_m": position.tolist(),
        "renderer_usd_quaternion_xyzw": quaternion.tolist(),
    }


def _synchronize_wrist_before_capture(native: Any) -> None:
    """Keep the verified Fabric camera pose causal with the live articulation."""

    if not getattr(native, "_npa_pi05_wrist_mount_verified", False):
        raise RuntimeError("wrist mount provenance was not verified before capture")
    position, quaternion = _wrist_expected_pose(native)
    _apply_wrist_world_pose(native, position, quaternion)


def _wrist_mount_receipt(native: Any, buffers: dict[str, list[Any]]) -> dict[str, Any]:
    """Refresh and retain pre/post evidence for the Gen3 moving camera mount."""

    sensor = native.scene["pi05_wrist"]
    receipt = {
        "schema": "npa.sim2real.pi05.wrist-mount.v1",
        **_wrist_prim_identity(sensor),
        "pose_sources": {
            "hand": "ArticulationData.body_pos_w/body_quat_w",
            "camera_cached": "CameraData.pos_w/quat_w_ros",
            "camera_live": "FabricFrameView.get_world_poses",
            "renderer": "UsdGeom.ComputeLocalToWorldTransform",
            "mount": "declared hand-frame calibration composed with live articulation",
        },
        "before": _wrist_raw_pose_snapshot(native),
    }
    buffers["wrist_mount_receipts"].append(receipt)
    try:
        _validate_wrist_parent(receipt)
        receipt["refresh_action"] = _refresh_wrist_camera_pose(
            native, receipt["before"]
        )
        receipt["after"] = _wrist_raw_pose_snapshot(native)
        _validate_refreshed_wrist_pose(receipt["after"])
        receipt["post_refresh_render"] = _wrist_render_receipt(native)
        native._npa_pi05_wrist_mount_verified = True
    except Exception as error:
        receipt["failure"] = f"{type(error).__name__}: {error}"
        raise
    return receipt


def _wrist_geometry(native: Any, wrist: Any, camera: Any) -> dict[str, Any]:
    import numpy as np

    bounds = _finger_bounds(native)
    centers = [
        np.mean(np.asarray(value["world_corners_m"], dtype=float), axis=0)
        for value in bounds.values()
    ]
    finger_midpoint = np.mean(centers, axis=0)
    optical_quaternion = _xyzw_to_wxyz(
        _proxy_torch(wrist.quat_w_ros)[0].detach().cpu().numpy()
    )
    optical_axis = _rotate_vector(optical_quaternion, [0.0, 0.0, 1.0])
    finger_direction = finger_midpoint - camera
    finger_direction /= np.linalg.norm(finger_direction)
    alignment = float(np.dot(optical_axis, finger_direction))
    if alignment < 0.5:
        raise RuntimeError("wrist optical axis does not aim toward the actual fingers")
    return {
        "measured_optical_axis_world": optical_axis.tolist(),
        "measured_finger_midpoint_world_m": finger_midpoint.tolist(),
        "finger_geometry_source": (
            "usd_compute_untransformed_bound_link_frame_transformed_by_"
            "live_articulation_body_pose_xyzw"
        ),
        "measured_optical_finger_alignment_cosine": alignment,
    }


def _wrist_mount_evidence(native: Any, buffers: dict[str, list[Any]]) -> dict[str, Any]:
    import numpy as np

    sensor = native.scene["pi05_wrist"]
    receipt = _wrist_mount_receipt(native, buffers)
    wrist = sensor.data
    robot = native.scene["robot"]
    hand_index = robot.body_names.index("panda_hand")
    hand = _proxy_torch(robot.data.body_pos_w)[0, hand_index].detach().cpu().numpy()
    camera = _proxy_torch(wrist.pos_w)[0].detach().cpu().numpy()
    distance = float(np.linalg.norm(camera - hand))
    if not 0.02 <= distance <= 0.12:
        raise RuntimeError("wrist camera world pose is not attached near panda_hand")
    horizontal_fov = 2.0 * np.arctan(20.955 / (2.0 * 18.0))
    return {
        "runtime_prim_path": receipt["runtime_prim_path"],
        "runtime_parent_prim": receipt["runtime_parent_prim"],
        "mount_translation_m": list(WRIST_MOUNT_TRANSLATION_M),
        "mount_target_local_m": list(WRIST_MOUNT_TARGET_LOCAL_M),
        "mount_quaternion_xyzw": list(WRIST_MOUNT_QUATERNION_XYZW),
        "quaternion_order": "xyzw",
        "offset_convention": "ros",
        "optical_convention": "ros_forward_positive_z_right_positive_x_down_positive_y",
        "hand_asset_axis": "side_bracket_aimed_at_measured_finger_link_reference",
        **_wrist_geometry(native, wrist, camera),
        "latest_pose_updates_enabled": True,
        "renderer_pose_bridge": "live_hand_mount_to_world_usd_and_fabric",
        "semantic_segmentation": (
            "int32_little_endian_packed_rgba_with_exact_metadata_bijection"
        ),
        "resolution_hw": [224, 224],
        "focal_length_mm": 18.0,
        "horizontal_aperture_mm": 20.955,
        "horizontal_fov_rad": float(horizontal_fov),
        "vertical_fov_rad": float(horizontal_fov),
        "measured_camera_to_hand_distance_m": distance,
        "moving_with_end_effector": True,
        "runtime_pose_receipt": receipt,
    }


def _contact_forces(native: Any) -> tuple[float, float, float]:
    import torch

    values = []
    for name in (
        "left_object_contact",
        "right_object_contact",
        "object_support_contact",
    ):
        sensor = native.scene[name]
        force = getattr(sensor.data, "force_matrix_w", None)
        if force is None:
            raise RuntimeError(f"{name} did not publish filtered contact forces")
        force = _proxy_torch(force)
        values.append(float(torch.linalg.norm(force).item()))
    return values[0], values[1], values[2]


def _sim_time(native: Any) -> float:
    """Return the environment's episode-local control clock."""

    value = _proxy_torch(native.episode_length_buf)
    if hasattr(value, "detach"):
        value = value[0].detach().cpu().item()
    else:
        value = value[0]
    return float(value) * float(native.step_dt)


def _sensor_clock_scalar(value: Any) -> float:
    """Read one Gen3 Warp, ProxyArray, Torch, or NumPy sensor clock."""

    value = _proxy_torch(value)
    if not hasattr(value, "detach"):
        try:
            import warp as wp

            value = wp.to_torch(value)
        except (ImportError, TypeError, AttributeError) as error:
            raise RuntimeError("unsupported Isaac sensor clock buffer") from error
    return float(value[0].detach().cpu().item())


def _camera_time(native: Any, name: str) -> float:
    sensor = native.scene[name]
    # Camera.data performs the documented lazy renderer refresh. Its base-class
    # last-update clock then names the capture that produced those exact bytes.
    _ = sensor.data
    value = getattr(sensor, "_timestamp_last_update", None)
    if value is None:
        raise RuntimeError(f"{name} exposes no capture-update clock")
    return _sensor_clock_scalar(value)


def _expert_waypoints(
    object_position: Any, target: Any, geometry: dict[str, Any]
) -> list[tuple[str, Any, float]]:
    import numpy as np

    grasp_reference_local = geometry["hand_to_finger_link_centroid_local_m"]
    reference_shift = geometry["contact_reference_above_object_center_m"]
    world_offset = _rotate_vector(
        _xyzw_to_wxyz(DOWNWARD_HAND_QUATERNION_XYZW), grasp_reference_local
    )

    def hand_goal(grasp_reference_goal: Any) -> Any:
        return np.asarray(grasp_reference_goal, dtype=float) - world_offset

    contact = object_position + [0.0, 0.0, reference_shift]
    destination = target + [0.0, 0.0, reference_shift]
    return [
        ("approach", hand_goal(contact + [0.0, 0.0, 0.14]), 0.0),
        ("contact", hand_goal(contact), 0.0),
        ("grasp", hand_goal(contact), 1.0),
        ("lift", hand_goal(contact + [0.0, 0.0, 0.16]), 1.0),
        ("transport", hand_goal(destination + [0.0, 0.0, 0.16]), 1.0),
        ("lower", hand_goal(destination), 1.0),
        ("release", hand_goal(destination), 0.0),
        ("retreat", hand_goal(destination + [0.0, 0.0, 0.16]), 0.0),
    ]


def _sensor_clock_reference(native: Any) -> dict[str, Any]:
    """Reproduce SensorBase's reset-local float32 update arithmetic."""

    import numpy as np

    control_steps = int(_proxy_torch(native.episode_length_buf)[0].item())
    updates = control_steps * int(native.cfg.decimation)
    physics_dt = np.float32(native.physics_dt)
    cache = getattr(native, "_npa_pi05_sensor_clock", None)
    if cache is None or updates < cache["updates"]:
        epoch = 0 if cache is None else cache["reset_epoch"] + 1
        cache = {"updates": 0, "value": np.float32(0.0), "reset_epoch": epoch}
    for _ in range(cache["updates"], updates):
        cache["value"] = np.float32(cache["value"] + physics_dt)
    cache["updates"] = updates
    native._npa_pi05_sensor_clock = cache
    return {
        "expected_sensor_time_s": float(cache["value"]),
        "physics_update_count": updates,
        "physics_period_s": float(physics_dt),
        "reset_epoch": cache["reset_epoch"],
    }


def _validate_capture_ages(
    captures: tuple[float, float], reference: dict[str, Any], control_dt: float
) -> None:
    """Reject future or stale raw clocks against reproduced float32 time."""

    import numpy as np

    expected = reference["expected_sensor_time_s"]
    ages = tuple(expected - capture for capture in captures)
    tolerance = max(1.0e-7, 4.0 * abs(float(np.spacing(np.float32(expected)))))
    if min(ages) < -tolerance:
        raise RuntimeError("camera capture timestamp is in the simulator future")
    if max(ages) > control_dt + tolerance:
        raise RuntimeError(
            "camera captures are stale relative to the causal observation"
        )


def _raw_camera_observation(native: Any) -> dict[str, Any]:
    """Capture synchronized RGB and pose before semantic validation can fail."""

    _synchronize_wrist_before_capture(native)
    now = _sim_time(native)
    exterior_time = _camera_time(native, "pi05_exterior")
    wrist_time = _camera_time(native, "pi05_wrist")
    clock = _sensor_clock_reference(native)
    _validate_capture_ages((exterior_time, wrist_time), clock, native.step_dt)
    exterior, wrist = _rgb(native, "pi05_exterior"), _rgb(native, "pi05_wrist")
    exterior_receipt = _rgb_receipt(exterior, exterior_time, now)
    wrist_receipt = _rgb_receipt(wrist, wrist_time, now)
    expected = clock["expected_sensor_time_s"]
    exterior_receipt["sensor_causal_age_s"] = expected - exterior_time
    wrist_receipt["sensor_causal_age_s"] = expected - wrist_time
    return {
        "exterior": exterior,
        "wrist": wrist,
        "exterior_time": exterior_time,
        "wrist_time": wrist_time,
        "time": now,
        "sensor_clock_reference": {
            **clock,
            "representation_offset_s": clock["expected_sensor_time_s"] - now,
        },
        "camera_pose": _camera_pose_sample(native),
        "exterior_receipt": exterior_receipt,
        "wrist_receipt": wrist_receipt,
    }


def _stage_diagnostic_capture(
    buffers: dict[str, list[Any]], raw: dict[str, Any]
) -> None:
    """Preserve a non-training camera row for failures after RGB capture."""

    buffers["diagnostic_exterior"].append(raw["exterior"].copy())
    buffers["diagnostic_wrist"].append(raw["wrist"].copy())
    buffers["diagnostic_receipts"].append(
        {
            "observation_time_s": raw["time"],
            "exterior_capture_time_s": raw["exterior_time"],
            "wrist_capture_time_s": raw["wrist_time"],
            "sensor_clock_reference": raw["sensor_clock_reference"],
            "camera_pose": raw["camera_pose"],
            "exterior_rgb": raw["exterior_receipt"],
            "wrist_rgb": raw["wrist_receipt"],
        }
    )


def _causal_observation(
    native: Any, joint_indices: list[int], buffers: dict[str, list[Any]]
) -> dict[str, Any]:
    import numpy as np

    raw = _raw_camera_observation(native)
    _stage_diagnostic_capture(buffers, raw)
    semantic_receipts = {
        name: _semantic_encoding_receipt(native, name)
        for name in ("pi05_exterior", "pi05_wrist")
    }
    q = _proxy_torch(native.scene["robot"].data.joint_pos)[0, joint_indices]
    return {
        **raw,
        "semantic_receipts": semantic_receipts,
        "exterior_pixels": _target_pixel_count(native, "pi05_exterior"),
        "exterior_destination_pixels": _destination_pixel_count(
            native, "pi05_exterior"
        ),
        "wrist_pixels": _target_pixel_count(native, "pi05_wrist"),
        "wrist_finger_in_frustum": _finger_in_frustum(native),
        "wrist_finger_pixels": _finger_pixel_counts(native, "pi05_wrist"),
        "q": q.detach().cpu().numpy().astype(np.float32, copy=True),
        "width": _gripper_width(native),
    }


def _expert_command(
    context: dict[str, Any], goal: Any, gripper: float
) -> tuple[Any, float, float]:
    import numpy as np
    import torch

    native = context["native"]
    ee = _proxy_torch(native.scene["ee_frame"].data.target_pos_w)[0, 0, :3]
    ee_quaternion = _proxy_torch(native.scene["ee_frame"].data.target_quat_w)[0, 0, :4]
    error = torch.as_tensor(goal, device=ee.device) - ee
    rotation = _axis_angle_error_xyzw(
        ee_quaternion.detach().cpu().numpy(), DOWNWARD_HAND_QUATERNION_XYZW
    )
    processed = np.r_[
        np.clip(error.detach().cpu().numpy(), -0.07, 0.07),
        np.clip(rotation, -0.25, 0.25),
    ]
    raw = np.clip(processed / context["expert_action_scale"], -1.0, 1.0)
    command = torch.zeros((1, context["total_dim"]), device=ee.device)
    start = context["arm_start"]
    command[0, start : start + 6] = torch.as_tensor(raw, device=ee.device)
    command[0, context["grip_start"]] = -1.0 if gripper > 0.5 else 1.0
    context["orientation_control"] = {
        "current_quaternion_xyzw": ee_quaternion.detach().cpu().numpy().tolist(),
        "target_quaternion_xyzw": list(DOWNWARD_HAND_QUATERNION_XYZW),
        "axis_angle_error_rad": rotation.tolist(),
        "processed_delta": processed.tolist(),
        "raw_scaled_action": raw.tolist(),
    }
    return (
        command,
        float(torch.linalg.norm(error).item()),
        float(np.linalg.norm(rotation)),
    )


def _step_command(
    context: dict[str, Any], command: Any, phase: str
) -> tuple[float, float]:
    native, env = context["native"], context["env"]
    command_time = _sim_time(native)
    _, _, terminated, truncated, _ = env.step(command)
    resulting_time = _sim_time(native)
    measured_hold = resulting_time - command_time
    if abs(measured_hold - 1.0 / CONTROL_HZ) > 1.0e-6:
        raise RuntimeError(f"Isaac control cadence mismatch: {measured_hold} seconds")
    if bool(terminated[0]) or bool(truncated[0]):
        raise RuntimeError(f"Isaac reset during expert phase {phase}")
    return command_time, resulting_time


def _hand_object_distance(native: Any, object_position: Any) -> float:
    import numpy as np

    robot = native.scene["robot"]
    hand_index = list(robot.body_names).index("panda_hand")
    hand = _proxy_torch(robot.data.body_pos_w)[0, hand_index].detach().cpu().numpy()
    return float(np.linalg.norm(hand - object_position))


def _object_kinematics(native: Any) -> tuple[Any, float]:
    import numpy as np

    position = (
        _proxy_torch(native.scene["object"].data.root_pos_w)[0, :3]
        .detach()
        .cpu()
        .numpy()
    )
    velocity = (
        _proxy_torch(native.scene["object"].data.root_lin_vel_w)[0]
        .detach()
        .cpu()
        .numpy()
    )
    return position, float(np.linalg.norm(velocity))


def _physics_sample(
    context: dict[str, Any], phase: str, intent: float
) -> dict[str, Any]:
    import numpy as np

    native = context["native"]
    obj_np, speed = _object_kinematics(native)
    left, right, support = _contact_forces(native)
    width = _gripper_width(native)
    bilateral = left > 1.0e-3 and right > 1.0e-3
    lifted = float(obj_np[2] - context["initial_z"]) >= 0.05
    xy_error = float(np.linalg.norm(obj_np[:2] - context["target"][:2]))
    release_width = release_width_threshold(context["object_width_m"])
    released = width >= release_width and phase in {"release", "retreat"}
    retreat_distance = _hand_object_distance(native, obj_np)
    retreated = phase == "retreat" and retreat_distance >= 0.10
    supported = support > 1.0e-3 and not bilateral and released and retreated
    return {
        "object_position_m": obj_np.tolist(),
        "object_speed_m_s": speed,
        "left_finger_object_force_n": left,
        "right_finger_object_force_n": right,
        "object_support_force_n": support,
        "bilateral_finger_contact": bilateral,
        "support_contact": support > 1.0e-3,
        "gripper_width_m": width,
        "hand_object_distance_m": retreat_distance,
        "finger_controller_target_m": _finger_controller_target(native).tolist(),
        "target_xy_distance_m": xy_error,
        "lifted": lifted,
        "transported": xy_error < 0.08 and lifted,
        "released": released,
        "grasped": bilateral
        and intent > 0.5
        and abs(width - context["object_width_m"]) <= 0.015
        and (lifted or speed > 0.005),
        "supported": supported,
    }


def _update_events(context: dict[str, Any], sample: dict[str, Any]) -> None:
    stable = sample["supported"] and sample["target_xy_distance_m"] < 0.05
    stable = stable and sample["object_speed_m_s"] < 0.03
    context["stable_steps"] = context["stable_steps"] + 1 if stable else 0
    conditions = (
        ("contact", sample["bilateral_finger_contact"]),
        ("grasp", sample["grasped"]),
        ("lift", sample["lifted"]),
        ("transport", sample["transported"]),
        ("release", sample["released"]),
        ("support_contact", sample["supported"]),
        ("stable", context["stable_steps"] >= 3),
    )
    for name, condition in conditions:
        if condition and name not in context["buffers"]["events"]:
            context["buffers"]["events"].append(name)


def _record_step(
    context: dict[str, Any],
    observation: dict[str, Any],
    intent: float,
    phase: str,
    command_time: float,
    resulting_time: float,
) -> None:
    import numpy as np

    buffers, native = context["buffers"], context["native"]
    target_q = _controller_target(native)
    actual_q = _proxy_torch(native.scene["robot"].data.joint_pos)[
        0, context["joint_indices"]
    ]
    actual_q = actual_q.detach().cpu().numpy()
    buffers["controller_errors"].append(float(np.max(np.abs(target_q - actual_q))))
    state = np.r_[observation["q"], 1.0 - observation["width"] / MAX_GRIPPER_WIDTH_M]
    buffers["frames_exterior"].append(observation["exterior"])
    buffers["frames_wrist"].append(observation["wrist"])
    buffers["states"].append(state.astype(np.float32))
    buffers["actions"].append(np.r_[target_q, intent].astype(np.float32))
    buffers["timestamps"].append(observation["time"])
    sample = _physics_sample(context, phase, intent)
    _update_events(context, sample)
    sample.update(
        _trace_receipts(context, observation, phase, command_time, resulting_time)
    )
    buffers["traces"].append(sample)
    # Once the row is admitted, its ordinary episode arrays and trace own the
    # bytes. Keep diagnostic buffers only for a capture that fails validation.
    for name in ("diagnostic_exterior", "diagnostic_wrist", "diagnostic_receipts"):
        buffers[name].pop()


def _trace_receipts(
    context: dict,
    observation: dict,
    phase: str,
    command_time: float,
    result_time: float,
) -> dict[str, Any]:
    return {
        "step": len(context["buffers"]["actions"]) - 1,
        "phase": phase,
        "observation_time_s": observation["time"],
        "exterior_capture_time_s": observation["exterior_time"],
        "wrist_capture_time_s": observation["wrist_time"],
        "command_application_time_s": command_time,
        "resulting_state_time_s": result_time,
        "time_scope": "episode_local_since_reset",
        "sensor_clock_reference": observation["sensor_clock_reference"],
        "camera_pose": observation["camera_pose"],
        "wrist_finger_in_frustum": observation["wrist_finger_in_frustum"],
        "wrist_finger_pixels": observation["wrist_finger_pixels"],
        "exterior_rgb": observation["exterior_receipt"],
        "wrist_rgb": observation["wrist_receipt"],
        "semantic_encoding": observation["semantic_receipts"],
        "exterior_target_pixels": observation["exterior_pixels"],
        "exterior_destination_pixels": observation["exterior_destination_pixels"],
        "wrist_target_pixels": observation["wrist_pixels"],
        "stable_steps": context["stable_steps"],
        "orientation_control": dict(context["orientation_control"]),
    }


def _run_phase(context: dict[str, Any], phase: str, goal: Any, intent: float) -> None:
    reached = 0
    for _ in range(90):
        command, distance, orientation_error = _expert_command(context, goal, intent)
        observation = _causal_observation(
            context["native"], context["joint_indices"], context["buffers"]
        )
        command_time, resulting_time = _step_command(context, command, phase)
        _record_step(context, observation, intent, phase, command_time, resulting_time)
        reached = reached + 1 if distance < 0.012 and orientation_error < 0.08 else 0
        hold = 12 if phase in {"grasp", "release"} else 3
        if reached >= hold or (phase == "retreat" and context["stable_steps"] >= 3):
            return
    raise RuntimeError(f"expert failed to reach {phase} waypoint")


def _verify_saved_camera_rows(episode: Path, traces: list[dict[str, Any]]) -> None:
    import numpy as np

    for filename, receipt_name in (
        ("obs_workspace.npy", "exterior_rgb"),
        ("obs_wrist.npy", "wrist_rgb"),
    ):
        frames = np.load(episode / filename, mmap_mode="r", allow_pickle=False)
        if len(frames) != len(traces):
            raise RuntimeError(f"{filename} row count changed on readback")
        for index, frame in enumerate(frames):
            digest = hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()
            if digest != traces[index][receipt_name]["sha256"]:
                raise RuntimeError(f"{filename} row {index} changed on readback")


def _write_episode_arrays(episode: Path, buffers: dict[str, list[Any]]) -> None:
    import numpy as np

    np.save(episode / "obs_workspace.npy", np.stack(buffers["frames_exterior"]))
    np.save(episode / "obs_wrist.npy", np.stack(buffers["frames_wrist"]))
    np.save(episode / "state.npy", np.stack(buffers["states"]).astype(np.float32))
    np.save(episode / "actions.npy", np.stack(buffers["actions"]).astype(np.float32))
    np.save(
        episode / "timestamps.npy", np.asarray(buffers["timestamps"], dtype=np.float64)
    )
    trace = buffers["traces"]
    keys = (
        "object_position_m",
        "object_speed_m_s",
        "left_finger_object_force_n",
        "right_finger_object_force_n",
        "object_support_force_n",
        "gripper_width_m",
        "hand_object_distance_m",
        "finger_controller_target_m",
    )
    np.savez(
        episode / "physics_trace.npz",
        **{
            key: np.asarray([row[key] for row in trace], dtype=np.float32)
            for key in keys
        },
    )
    _verify_saved_camera_rows(episode, trace)


def _image_record(filename: str, index: int, receipt: dict) -> dict[str, Any]:
    return {
        "path": f"{filename}#{index}",
        "shape": [224, 224, 3],
        "dtype": "uint8",
        **receipt,
    }


def _visibility_record(trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_exterior_pixels": trace["exterior_target_pixels"],
        "target_wrist_pixels": trace["wrist_target_pixels"],
        "bilateral_finger_object_contact": trace["bilateral_finger_contact"],
        "left_finger_object_force_n": trace["left_finger_object_force_n"],
        "right_finger_object_force_n": trace["right_finger_object_force_n"],
        "wrist_finger_geometry": trace["wrist_finger_in_frustum"],
        "wrist_finger_rendered_pixels": trace["wrist_finger_pixels"],
        "semantic_encoding": trace["semantic_encoding"],
    }


def _step_record(buffers: dict[str, list[Any]], index: int) -> dict[str, Any]:
    trace = buffers["traces"][index]
    timestamp = buffers["timestamps"][index]
    record = {
        "step": index,
        "phase": trace["phase"],
        "timestamp_s": timestamp,
        "exterior_timestamp_s": trace["exterior_capture_time_s"],
        "wrist_timestamp_s": trace["wrist_capture_time_s"],
        "command_application_timestamp_s": trace["command_application_time_s"],
        "resulting_state_timestamp_s": trace["resulting_state_time_s"],
        "joint_position": buffers["states"][index][:7].tolist(),
        "gripper_position": [float(buffers["states"][index][7])],
        "action": buffers["actions"][index].tolist(),
        "exterior_image_1_left": _image_record(
            "obs_workspace.npy", index, trace["exterior_rgb"]
        ),
        "wrist_image_left": _image_record("obs_wrist.npy", index, trace["wrist_rgb"]),
        "visibility": _visibility_record(trace),
        "camera_pose": trace["camera_pose"],
    }
    record["row_identity_sha256"] = _sha(
        {
            "step": index,
            "phase": record["phase"],
            "timestamp_s": timestamp,
            "exterior_sha256": trace["exterior_rgb"]["sha256"],
            "wrist_sha256": trace["wrist_rgb"]["sha256"],
            "action": record["action"],
        }
    )
    return record


def _step_records(buffers: dict[str, list[Any]]) -> list[dict[str, Any]]:
    return [_step_record(buffers, index) for index in range(len(buffers["timestamps"]))]


def _controller_success_evidence(context: dict[str, Any]) -> dict[str, Any]:
    buffers = context["buffers"]
    events = buffers["events"]
    expected = [
        "contact",
        "grasp",
        "lift",
        "transport",
        "release",
        "support_contact",
        "stable",
    ]
    return {
        "controller": {
            "labels_are_actual_absolute_controller_targets": True,
            "maximum_tracking_error_rad": max(buffers["controller_errors"]),
            "hold_seconds": 1.0 / CONTROL_HZ,
            "gripper_convention": "DROID 0=open, 1=closed; predictions >0.5 close",
            "two_finger_targets_recorded_in_physics_trace": True,
        },
        "success_evidence": {
            "ordered_events": events,
            "released_supported_surface_placement": events == expected
            and context["stable_steps"] >= 3,
            "stable_steps": context["stable_steps"],
            "target_xy_distance_m_lt": 0.05,
            "object_speed_m_s_lt": 0.03,
            "physics_trace": "physics_trace.npz",
        },
    }


def _episode_evidence(
    context: dict[str, Any], mounted: dict[str, Any]
) -> dict[str, Any]:
    buffers = context["buffers"]
    count = len(buffers["actions"])
    ordered_receipts = [
        {
            "step": row["step"],
            "phase": row["phase"],
            "exterior_sha256": row["exterior_rgb"]["sha256"],
            "wrist_sha256": row["wrist_rgb"]["sha256"],
            "observation_time_s": row["observation_time_s"],
        }
        for row in buffers["traces"]
    ]
    return {
        "camera": {
            "exterior": {"mount": "world"},
            "wrist": mounted,
            "motion_and_visibility": _verify_camera_motion(buffers["traces"]),
            "capture_receipts_sha256": _sha(ordered_receipts),
            "saved_array_hash_readback": True,
            "visibility_semantics": "phase-specific; retreat occlusion is permitted",
        },
        "dense_alignment": {
            name: count
            for name in (
                "exterior_frames",
                "wrist_frames",
                "states",
                "actions",
                "timestamps",
            )
        },
        **_controller_success_evidence(context),
    }


def _episode_metadata(
    context: dict[str, Any],
    identity: str,
    scene_id: str,
    attempt: int,
    mounted: dict[str, Any],
) -> dict[str, Any]:
    buffers = context["buffers"]
    return {
        "schema": EPISODE_SCHEMA,
        "episode_id": f"episode-{attempt:06d}",
        "source_backend": "isaac",
        "object_motion": "physics_only",
        "direct_object_pose_writes": 0,
        "action_semantics": "absolute_joint_position_plus_normalized_gripper",
        "joint_order": list(JOINT_NAMES),
        "control_hz": CONTROL_HZ,
        "step_count": len(buffers["actions"]),
        "object_identity": identity,
        "object_width_m": context["object_width_m"],
        "object_height_m": context["object_height_m"],
        "finger_grasp_reference_geometry": context["finger_grasp_reference_geometry"],
        "hand_origin_control": context["hand_origin_control"],
        "robot_plant": context["robot_plant"],
        "initial_object_position_m": context["initial_object_position"].tolist(),
        "target_position_m": context["target"].tolist(),
        "active_surface_command_m": context["active_surface_command_m"],
        "reset_settle": context["reset_settle"],
        "scene_semantic_tags": dict(
            context["native"]._npa_pi05_verified_semantic_paths
        ),
        "support_surface": support_surface_contract(),
        "scene_configuration_digest": _scene_digest(context, scene_id, attempt),
        "scene_seed": context["scene_seed"],
        "destination_marker": _destination_marker_evidence(context, buffers),
        "instruction": build_contract()["task"]["instruction"],
        **_episode_evidence(context, mounted),
        "steps": _step_records(buffers),
    }


def _scene_digest(context: dict[str, Any], scene_id: str, attempt: int) -> str:
    return _sha(
        {
            "scene_id": scene_id,
            "target": context["target"].tolist(),
            "destination_marker": {
                "label": DESTINATION_LABEL,
                "radius_m": DESTINATION_RADIUS_M,
                "physics_collision": False,
            },
            "support_surface": support_surface_contract(),
            "attempt": attempt,
            "initial_z": context["initial_z"],
            "object_width_m": context["object_width_m"],
            "object_height_m": context["object_height_m"],
            "finger_grasp_reference_geometry": context[
                "finger_grasp_reference_geometry"
            ],
            "hand_origin_control": context["hand_origin_control"],
            "robot_plant": context["robot_plant"],
            "active_surface_command_m": context["active_surface_command_m"],
            "reset_settle": context["reset_settle"],
            "scene_semantic_tags": dict(
                context["native"]._npa_pi05_verified_semantic_paths
            ),
            "scene_seed": context["scene_seed"],
        }
    )


def _destination_marker_evidence(
    context: dict[str, Any], buffers: dict[str, list[Any]]
) -> dict[str, Any]:
    return {
        "source": "isaaclab_noncolliding_cylinder_primitive",
        "center_xy_m": context["target"][:2].tolist(),
        "support_z_m": 0.0,
        "radius_m": DESTINATION_RADIUS_M,
        "height_m": 0.001,
        "color_rgb": [0.0, 0.85, 0.85],
        "semantic_label": DESTINATION_LABEL,
        "minimum_exterior_semantic_pixels": min(
            row["exterior_destination_pixels"] for row in buffers["traces"]
        ),
        "physics_collision": False,
    }


def _attempt_context(
    native: Any,
    env: Any,
    buffers: dict[str, list[Any]],
    object_position: Any,
    target: Any,
    object_width_m: float,
    object_height_m: float,
) -> dict[str, Any]:
    total_dim, arm_start, grip_start = _action_layout(native)
    robot = native.scene["robot"]
    grasp_geometry = _object_grasp_geometry(
        _finger_grasp_reference_geometry(native), object_height_m
    )
    return {
        "native": native,
        "env": env,
        "buffers": buffers,
        "target": target,
        "initial_object_position": object_position.copy(),
        "initial_z": float(object_position[2]),
        "object_width_m": object_width_m,
        "object_height_m": object_height_m,
        "finger_grasp_reference_geometry": grasp_geometry,
        "hand_origin_control": _hand_origin_control_receipt(native, require_ik=True),
        "robot_plant": _robot_plant_receipt(native),
        "total_dim": total_dim,
        "arm_start": arm_start,
        "grip_start": grip_start,
        "expert_action_scale": _expert_action_scale(native),
        "joint_indices": [list(robot.joint_names).index(name) for name in JOINT_NAMES],
        "stable_steps": 0,
        "orientation_control": {},
    }


def _reset_settle_sample(native: Any, command_time: float) -> dict[str, Any]:
    position, speed = _object_kinematics(native)
    _left, _right, support = _contact_forces(native)
    return {
        "command_time_s": command_time,
        "result_time_s": _sim_time(native),
        "object_position_m": position.astype(float).tolist(),
        "object_speed_m_s": speed,
        "object_support_force_n": support,
    }


def _validate_reset_settle(samples: list[dict[str, Any]], steps: int) -> None:
    final = samples[-1]
    if steps != RESET_SETTLE_CONTROL_STEPS:
        raise RuntimeError("reset settle did not advance its declared control steps")
    if final["object_speed_m_s"] >= 0.03:
        raise RuntimeError("reset settle object remained in motion")
    if final["object_support_force_n"] <= 1.0e-3:
        raise RuntimeError("reset settle object lacked support contact")
    prior_result = None
    for sample in samples:
        command = float(sample["command_time_s"])
        result = float(sample["result_time_s"])
        if not math.isfinite(command) or not math.isfinite(result):
            raise RuntimeError("reset settle clocks were not finite")
        if abs((result - command) - 1.0 / CONTROL_HZ) > 1.0e-6:
            raise RuntimeError("reset settle did not hold the 15 Hz cadence")
        if prior_result is not None and abs(command - prior_result) > 1.0e-6:
            raise RuntimeError("reset settle clocks were not contiguous")
        prior_result = result


def _settle_expert_reset(native: Any, env: Any) -> dict[str, Any]:
    """Advance two measured control intervals before any training capture."""

    import torch

    total_dim, _arm_start, grip_start = _action_layout(native)
    action = torch.zeros((1, total_dim), device=native.device)
    action[0, grip_start] = 1.0
    before = int(_proxy_torch(native.episode_length_buf)[0].item())
    samples = []
    for _ in range(RESET_SETTLE_CONTROL_STEPS):
        command_time = _sim_time(native)
        _, _, terminated, truncated, _ = env.step(action)
        if bool(terminated[0]) or bool(truncated[0]):
            raise RuntimeError("expert environment reset during declared settle")
        samples.append(_reset_settle_sample(native, command_time))
    after = int(_proxy_torch(native.episode_length_buf)[0].item())
    _validate_reset_settle(samples, after - before)
    return {
        "control_steps": RESET_SETTLE_CONTROL_STEPS,
        "physics_steps": RESET_SETTLE_CONTROL_STEPS * int(native.cfg.decimation),
        "start_control_step": before,
        "result_control_step": after,
        "object_speed_m_s": samples[-1]["object_speed_m_s"],
        "object_support_force_n": samples[-1]["object_support_force_n"],
        "samples": samples,
        "training_row": False,
    }


def _bind_surface_command(native: Any, target: Any) -> list[float]:
    """Bind the inherited task command to the measured surface object center."""

    import torch

    command = native.command_manager.get_command("object_pose")
    target_tensor = torch.as_tensor(target, device=command.device, dtype=command.dtype)
    if command.ndim != 2 or command.shape[0] != 1 or command.shape[1] < 3:
        raise RuntimeError("object_pose command does not expose one Cartesian target")
    command[0, :3] = target_tensor
    actual = command[0, :3].detach().cpu().numpy().astype(float).tolist()
    if any(abs(left - right) > 1.0e-6 for left, right in zip(actual, target)):
        raise RuntimeError("object_pose command rejected measured surface target")
    return actual


def _run_expert_waypoints(context: dict[str, Any]) -> None:
    reference = context["finger_grasp_reference_geometry"]
    for phase, goal, intent in _expert_waypoints(
        context["initial_object_position"],
        context["target"],
        reference,
    ):
        _run_phase(context, phase, goal, intent)


def _capture_attempt(
    native: Any,
    env: Any,
    output: Path,
    attempt: int,
    *,
    object_identity: str,
    scene_id: str,
    buffers: dict[str, list[Any]],
    object_width_m: float,
    object_height_m: float,
    scene_seed: int,
) -> dict[str, Any]:
    import numpy as np

    settle = _settle_expert_reset(native, env)
    object_position, _speed = _object_kinematics(native)
    target = np.asarray([0.50, 0.18, float(object_position[2])], dtype=np.float32)
    context = _attempt_context(
        native, env, buffers, object_position, target, object_width_m, object_height_m
    )
    context["scene_seed"] = scene_seed
    context["reset_settle"] = settle
    context["active_surface_command_m"] = _bind_surface_command(native, target)
    mounted = _wrist_mount_evidence(native, buffers)
    _run_expert_waypoints(context)
    metadata = _episode_metadata(context, object_identity, scene_id, attempt, mounted)
    _publish_episode(output, attempt, buffers, metadata)
    return metadata


def _publish_episode(
    output: Path,
    attempt: int,
    buffers: dict[str, list[Any]],
    metadata: dict[str, Any],
) -> None:
    """Publish one complete episode atomically or leave no success directory."""

    final = output / f"episode-{attempt:06d}"
    if final.exists():
        raise RuntimeError(f"episode output already exists: {final.name}")
    staging = Path(tempfile.mkdtemp(prefix=f".{final.name}-", dir=output))
    try:
        _write_episode_arrays(staging, buffers)
        (staging / "episode.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        staging.replace(final)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _new_attempt_buffers() -> dict[str, list[Any]]:
    return {
        name: []
        for name in (
            "frames_exterior",
            "frames_wrist",
            "states",
            "actions",
            "timestamps",
            "controller_errors",
            "events",
            "traces",
            "diagnostic_exterior",
            "diagnostic_wrist",
            "diagnostic_receipts",
            "wrist_mount_receipts",
        )
    }


def _save_partial_attempt(
    root: Path, attempt: int, buffers: dict[str, list[Any]], error: Exception
) -> dict[str, Any]:
    import numpy as np

    episode = root / f"episode-{attempt:06d}-failed"
    episode.mkdir(parents=True, exist_ok=True)
    array_names = {
        "frames_exterior": "obs_workspace.npy",
        "frames_wrist": "obs_wrist.npy",
        "states": "state.npy",
        "actions": "actions.npy",
        "timestamps": "timestamps.npy",
    }
    for source, filename in array_names.items():
        if buffers[source]:
            np.save(episode / filename, np.asarray(buffers[source]))
    for source, filename in (
        ("diagnostic_exterior", "diagnostic_workspace.npy"),
        ("diagnostic_wrist", "diagnostic_wrist.npy"),
    ):
        if buffers[source]:
            np.save(episode / filename, np.asarray(buffers[source]))
    failure = {
        "schema": "npa.sim2real.pi05.failed_attempt.v1",
        "episode_id": f"episode-{attempt:06d}",
        "failure": f"{type(error).__name__}: {error}",
        "captured_steps": len(buffers["actions"]),
        "partial_frames_preserved": bool(buffers["frames_exterior"]),
        "diagnostic_camera_rows": len(buffers["diagnostic_receipts"]),
        "diagnostic_receipts": buffers["diagnostic_receipts"],
        "wrist_mount_receipts": buffers["wrist_mount_receipts"],
        "training_admitted": False,
        "trace": buffers["traces"],
    }
    (episode / "failure.json").write_text(
        json.dumps(failure, indent=2, sort_keys=True) + "\n"
    )
    return failure


def _make_environment(
    seed: int, object_usd: str, object_scale: tuple[float, float, float]
) -> tuple[Any, Any]:
    _require_isaac_camera_runtime()
    app = _launch_camera_app()
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    try:
        cfg = parse_env_cfg(EXPERT_BASE_TASK_ID, device="cuda:0", num_envs=1)
        _configure_environment(cfg, seed, object_usd, object_scale)
        register_surface_task(
            gym, base_id=EXPERT_BASE_TASK_ID, task_id=EXPERT_SURFACE_TASK_ID
        )
        env = gym.make(EXPERT_SURFACE_TASK_ID, cfg=cfg)
        _verify_scene_semantics(env.unwrapped)
        env.reset(seed=seed)
        return app, env
    except Exception as error:
        raise _EnvironmentSetupError(error, app) from error


def _validated_kit_args() -> str:
    """Bind Kit mutable state to the launcher-created writable job directory."""

    state_text = os.environ.get("NPA_ISAAC_STATE_DIR", "").strip()
    state = Path(state_text)
    if not state_text or not state.is_absolute() or not state.is_dir():
        raise RuntimeError("NPA_ISAAC_STATE_DIR must be an existing absolute directory")
    if not os.access(state, os.W_OK):
        raise RuntimeError("NPA_ISAAC_STATE_DIR is not writable")
    raw = os.environ.get("NPA_ISAAC_KIT_ARGS", "").strip()
    try:
        argv = shlex.split(raw)
    except ValueError as error:
        raise RuntimeError("NPA_ISAAC_KIT_ARGS is malformed") from error
    roots = _portable_roots(argv)
    if len(roots) != 1:
        raise RuntimeError("NPA_ISAAC_KIT_ARGS must declare one --portable-root")
    portable = Path(roots[0])
    if not portable.is_absolute():
        raise RuntimeError("Kit portable root must be absolute")
    state, portable = state.resolve(strict=True), portable.resolve(strict=False)
    if portable == state or not portable.is_relative_to(state):
        raise RuntimeError("Kit portable root must be inside NPA_ISAAC_STATE_DIR")
    return raw


def _portable_roots(argv: list[str]) -> list[str]:
    roots = []
    for index, value in enumerate(argv):
        if value == "--portable-root":
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                raise RuntimeError("--portable-root requires a path")
            roots.append(argv[index + 1])
        elif value.startswith("--portable-root="):
            roots.append(value.partition("=")[2])
    return roots


def _launch_camera_app() -> Any:
    """Launch camera-enabled Kit with the validated mutable-state boundary."""

    kit_args = _validated_kit_args()
    from isaaclab.app import AppLauncher

    return AppLauncher(headless=True, enable_cameras=True, kit_args=kit_args).app


def _graphics_preflight() -> dict[str, Any]:
    """Fail before Kit unless GLX/EGL and an NVIDIA Vulkan device are usable."""

    probe = (
        "import ctypes,os; ctypes.CDLL('libGLX_nvidia.so.0'); "
        "ctypes.CDLL('libEGL_nvidia.so.0'); os._exit(0)"
    )
    try:
        subprocess.run([sys.executable, "-c", probe], check=True, capture_output=True)
        result = subprocess.run(
            ["vulkaninfo", "--summary"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "pi0.5 RTX capture requires injected NVIDIA GLX/EGL and working Vulkan"
        ) from error
    output = result.stdout
    if not re.search(r"vendorID\s*=\s*0x10de\b", output, re.IGNORECASE):
        raise RuntimeError("Vulkan did not enumerate an NVIDIA physical device")
    return {
        "schema": "npa.sim2real.pi05.graphics-preflight.v1",
        "glx": "loaded",
        "egl": "loaded",
        "vulkan_nvidia_device": True,
        "vulkan_summary_sha256": hashlib.sha256(output.encode()).hexdigest(),
    }


def _configure_environment(
    cfg: Any,
    seed: int,
    object_usd: str,
    object_scale: tuple[float, float, float],
) -> None:
    from npa.workflows.sim2real.isaac_assets_compat import remap_moved_franka_usd

    cfg.seed, cfg.sim.dt, cfg.decimation = seed, 1.0 / 60.0, 4
    cfg.episode_length_s = 240.0
    configure_robot_plant(cfg)
    remap_moved_franka_usd(cfg)
    if object_usd:
        cfg.scene.object.spawn.usd_path = object_usd
    cfg.scene.object.spawn.scale = object_scale
    cfg.scene.object.spawn.semantic_tags = [("class", "pi05_target_object")]
    configure_object_reset(cfg)
    target = (0.50, 0.18, 0.0)
    configure_finger_semantics(cfg)
    configure_hand_origin_frames(cfg, require_ik=True)
    configure_support_surface(cfg)
    configure_surface_goal(cfg, target)
    configure_destination_marker(cfg, target)
    _camera_config(cfg)
    _contact_config(cfg)
    _physics_randomization(cfg)


def _require_isaac_camera_runtime() -> None:
    import importlib.metadata

    version = importlib.metadata.version("isaaclab")
    if version != "3.0.0b2.post1":
        raise RuntimeError(
            f"pi0.5 camera contract requires Isaac Lab 3.0.0b2.post1; found {version}"
        )


def _collect_attempts(
    env: Any,
    work: Path,
    episodes: int,
    seed: int,
    identity: str,
    scene_id: str,
    scale: tuple[float, float, float],
    geometry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    attempts = []
    for attempt in range(episodes):
        if attempt:
            env.reset(seed=seed + attempt)
        buffers = _new_attempt_buffers()
        try:
            result = _capture_attempt(
                env.unwrapped,
                env,
                work,
                attempt,
                object_identity=identity,
                scene_id=scene_id,
                buffers=buffers,
                object_width_m=float(geometry["grasp_width_m"]),
                object_height_m=float(geometry["support_height_m"]),
                scene_seed=seed + attempt,
            )
        except Exception as exc:
            failure = _save_partial_attempt(work, attempt, buffers, exc)
            result = {
                **failure,
                "success_evidence": {"released_supported_surface_placement": False},
            }
        attempts.append(result)
    return attempts


def _successful_scenarios(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "scene_seed": row["scene_seed"],
            "scene_configuration_digest": row["scene_configuration_digest"],
            "initial_object_position_m": row["initial_object_position_m"],
            "target_position_m": row["target_position_m"],
            "destination_marker": row["destination_marker"],
            "support_surface": row["support_surface"],
        }
        for row in attempts
        if row.get("success_evidence", {}).get("released_supported_surface_placement")
    ]


def _timestamp_contract() -> dict[str, str]:
    return {
        "scope": "episode_local_since_reset",
        "camera_source": "SensorBase._timestamp_last_update_after_data_refresh",
        "control_source": "episode_length_buf_times_step_dt",
        "representation": (
            "raw Warp float32 is retained; freshness uses the recorded update count "
            "and period to reproduce SensorBase float32 accumulation"
        ),
    }


def _destination_contract() -> dict[str, Any]:
    return {
        "source": "isaaclab_noncolliding_cylinder_primitive",
        "radius_m": DESTINATION_RADIUS_M,
        "color": "cyan",
        "semantic_label": DESTINATION_LABEL,
        "exterior_raw_semantic_pixels_required": True,
    }


def _collection_report(
    attempts: list[dict[str, Any]],
    identity: str,
    object_usd: str,
    scale: tuple[float, float, float],
    scene_id: str,
    geometry: Mapping[str, Any],
    robot_plant: Mapping[str, Any],
    graphics_preflight: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    scenarios = _successful_scenarios(attempts)
    return {
        "schema": "npa.sim2real.pi05.expert_collection.v1",
        "task_contract": build_contract(),
        "attempt_count": len(attempts),
        "successful_episode_count": len(scenarios),
        "attempts": attempts,
        "training_admission": "successful episodes only; failures retained",
        "timestamp_contract": _timestamp_contract(),
        "object_asset": {
            "family": "isaac-multicolor-cube",
            "identity": identity,
            "usd": object_usd or "task_default",
            "scale": list(scale),
            "geometry": dict(geometry),
        },
        "scene_id": scene_id,
        "scene_layout_holdout": False,
        "destination_marker_contract": _destination_contract(),
        "support_surface_contract": support_surface_contract(),
        "robot_plant": dict(robot_plant),
        "graphics_preflight": dict(graphics_preflight or {}),
        "successful_scenarios": scenarios,
    }


def _failure_report(error: Exception, scene_id: str, phase: str) -> dict[str, Any]:
    raw_traceback = "".join(traceback.format_exception(error))
    safe_traceback = re.sub(
        r"\b(?:s3|gs|https?)://[^\s'\"\\]+", "<redacted-uri>", raw_traceback
    )
    return {
        "schema": "npa.sim2real.pi05.collection_failure.v1",
        "phase": phase,
        "failure": re.sub(
            r"\b(?:s3|gs|https?)://[^\s'\"\\]+",
            "<redacted-uri>",
            f"{type(error).__name__}: {error}",
        ),
        "scene_id": scene_id,
        "valid_collection": False,
        "training_admitted": False,
        "traceback": safe_traceback,
    }


def _persist_failure(
    work: Path,
    storage: Any,
    output_uri: str,
    scene_id: str,
    error: Exception,
    phase: str,
) -> None:
    report = _failure_report(error, scene_id, phase)
    path = work / "collection-failure.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    try:
        storage.upload_directory(str(work), output_uri)
    except Exception as upload_error:
        raise RuntimeError(
            "collector failed and failure evidence upload also failed: "
            f"{type(upload_error).__name__}: {upload_error}"
        ) from error


def _close_runtime(env: Any | None, app: Any | None) -> None:
    try:
        if env is not None:
            env.close()
    finally:
        if app is not None:
            app.close()


def collect(
    *,
    output_uri: str,
    episodes: int,
    seed: int,
    object_identity: str,
    object_usd: str,
    object_scale: tuple[float, float, float],
    scene_id: str,
) -> dict[str, Any]:
    """Collect physics attempts. Args: collection settings. Returns: Report. Raises: RuntimeError."""

    from npa.clients.storage import StorageClient

    storage = StorageClient.from_environment()
    with tempfile.TemporaryDirectory(prefix="npa-pi05-isaac-") as directory:
        return _collect_in_directory(
            Path(directory),
            storage,
            output_uri,
            {
                "episodes": episodes,
                "seed": seed,
                "identity": object_identity,
                "object_usd": object_usd,
                "scale": object_scale,
                "scene_id": scene_id,
            },
        )


def _collect_in_directory(
    work: Path, storage: Any, output_uri: str, settings: Mapping[str, Any]
) -> dict[str, Any]:
    app, env = None, None
    try:
        app, env, graphics = _prepare_environment(work, settings)
        return _run_collection(
            env=env,
            work=work,
            storage=storage,
            output_uri=output_uri,
            episodes=int(settings["episodes"]),
            seed=int(settings["seed"]),
            identity=str(settings["identity"]),
            object_usd=str(settings["object_usd"]),
            scale=tuple(settings["scale"]),
            scene_id=str(settings["scene_id"]),
            graphics=graphics,
        )
    except Exception as error:
        app = app or getattr(error, "runtime_app", None)
        phase = getattr(
            error,
            "phase",
            "environment_startup" if env is None else "collection_runtime",
        )
        _persist_failure(
            work, storage, output_uri, str(settings["scene_id"]), error, phase
        )
        raise
    finally:
        _close_runtime(env, app)


def _prepare_environment(
    work: Path, settings: Mapping[str, Any]
) -> tuple[Any, Any, dict[str, Any]]:
    graphics = _graphics_preflight()
    (work / "graphics-preflight.json").write_text(
        json.dumps(graphics, indent=2, sort_keys=True) + "\n"
    )
    try:
        app, env = _make_environment(
            int(settings["seed"]),
            str(settings["object_usd"]),
            tuple(settings["scale"]),
        )
    except _EnvironmentSetupError as error:
        failure = CollectionFailure(str(error.original), phase="environment_startup")
        failure.runtime_app = error.app
        raise failure from error.original
    return app, env, graphics


def _run_collection(
    *,
    env: Any,
    work: Path,
    storage: Any,
    output_uri: str,
    episodes: int,
    seed: int,
    identity: str,
    object_usd: str,
    scale: tuple[float, float, float],
    scene_id: str,
    graphics: Mapping[str, Any],
) -> dict[str, Any]:
    geometry = _object_asset_geometry(env.unwrapped, scale)
    robot_plant = _robot_plant_receipt(env.unwrapped)
    attempts = _collect_attempts(
        env, work, episodes, seed, identity, scene_id, scale, geometry
    )
    report = _collection_report(
        attempts,
        identity,
        object_usd,
        scale,
        scene_id,
        geometry,
        robot_plant,
        graphics,
    )
    (work / "collection.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    storage.upload_directory(str(work), output_uri)
    if report["successful_episode_count"] < 1:
        raise CollectionFailure(
            "expert produced no physics-verified released placements",
            phase="physics_qualification",
        )
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the collector CLI. Args: argv. Returns: Exit status. Raises: RuntimeError."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--object-identity", required=True)
    parser.add_argument("--object-usd", default="")
    parser.add_argument("--object-scale", default="1,1,1")
    parser.add_argument("--scene-id", required=True)
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    try:
        scale = tuple(float(value) for value in args.object_scale.split(","))
    except ValueError as exc:
        parser.error(f"--object-scale must be three comma-separated numbers: {exc}")
    if len(scale) != 3 or any(value <= 0 for value in scale):
        parser.error("--object-scale must contain three positive values")
    try:
        report = collect(
            output_uri=args.output_uri,
            episodes=args.episodes,
            seed=args.seed,
            object_identity=args.object_identity,
            object_usd=args.object_usd,
            object_scale=scale,
            scene_id=args.scene_id,
        )
    except Exception as error:
        phase = getattr(error, "phase", "collection_runtime")
        failure = _failure_report(error, args.scene_id, phase)
        print(json.dumps(failure, sort_keys=True), file=sys.stderr, flush=True)
        return 1
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
