"""Native DROID embodiment and camera calibration for joint-position policies.

The joint and image contracts follow NVIDIA RoboLab's Apache-2.0 DROID
reference at ad45d4f974725d020f82c2b0d77d78533aeba2b3. Fixed camera mounts
and lenses are calibrated against native renders of recorded approach poses.
Robot assets remain in the operator's Isaac runtime; this module contains no
redistributed meshes or simulator payload.
"""

from __future__ import annotations

import math

ARM_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))
MODEL_JOINT_NAMES = (*ARM_JOINT_NAMES, "finger_joint")
GRIPPER_CLOSED_ANGLE = math.pi / 4
# A fixed, collision-free reset above the cube, verified with native physics.
# It defines a controlled pickup initial condition, not a controller or trajectory.
# The open fingers remain approximately 18 cm above the cube center.
PREGRASP_RESET_JOINTS = (
    -0.05077540868198145, -0.05783148862296681, 0.05077240484677532,
    -2.4232019767783046, 0.004187295087195224, 2.3654402068531453,
    0.7823222921618791,
)
GRIPPER_ROOT = "/World/Franka/Robotiq_2F_85_edit/Robotiq_2F_85"
GRIPPER_BASE = f"{GRIPPER_ROOT}/base_link"
CONTACT_BODIES = tuple(f"{GRIPPER_ROOT}/{side}_inner_finger" for side in ("left", "right"))
CAMERA_PATHS = {"exterior": "/World/ExteriorDroid", "wrist": f"{GRIPPER_BASE}/wrist_cam"}
NATIVE_POLICY_RESOLUTION = (180, 320)
CAMERA_CALIBRATION = {
    "exterior": {
        # Low side view keeps the target clear of the forearm across recorded
        # failure poses, while retaining the approaching fingers in frame.
        "position": (0.35, -0.65, 0.38),
        "quaternion_wxyz": (0.7760544786384438, 0.6276901716946259,
                             -0.03848190493529904, -0.04757769998365795),
        "focal_length": 2.8,
    },
    "wrist": {
        # A fixed bracket and wider lens retain the target during close approach.
        # The optical frame is calibrated in the native Robotiq body basis;
        # neither this transform nor the exterior camera tracks the object.
        "position": (0.12, -0.031, -0.025),
        "quaternion_wxyz": (0.2514280790974722, 0.6726227449022252,
                             0.6804223192506174, 0.14624647533244078),
        "focal_length": 2.1,
    },
}
DROID_REFERENCE_CALIBRATION = {
    "exterior": {
        "position": (0.05, 0.57, 0.66),
        "quaternion_wxyz": (-0.393, -0.195, 0.399, 0.805),
        "focal_length": 2.1,
    },
    "wrist": {
        # Convert the reference mount into the native accessory's +Z tool basis.
        # This is the previously rendered reference mount, not object tracking.
        "position": (0.074, -0.031, 0.011),
        "quaternion_wxyz": (0.110288973, 0.692134004, 0.704152674, 0.113823876),
        "focal_length": 2.8,
    },
}
TASK_VIEW_CALIBRATION = {
    "exterior": {
        # Looking toward the robot from in front of the table keeps the forearm
        # behind the target during approach. The lens resolves the cube without
        # a moving camera or a crop applied to the policy input.
        "position": (0.95, -0.8, 0.68),
        "quaternion_wxyz": (0.8097311153426455, 0.5133783487801894,
                             0.15218591261237785, 0.24003674687851262),
        "focal_length": 3.584,
    },
    "wrist": CAMERA_CALIBRATION["wrist"],
}


def camera_calibration(mounts="native_wide"):
    """Resolve one fixed camera rig without changing process-global calibration.

    Args:
        mounts: Named fixed policy camera mount pair.
    Returns:
        Independent per-view calibration dictionaries.
    Raises:
        ValueError: The requested mount pair is unsupported.
    """
    choices = {"native_wide": CAMERA_CALIBRATION,
               "droid_reference": DROID_REFERENCE_CALIBRATION,
               "task_view": TASK_VIEW_CALIBRATION}
    if mounts not in choices:
        raise ValueError("camera_mounts must be native_wide, droid_reference or task_view")
    return {view: values.copy() for view, values in choices[mounts].items()}


def joint_indices(names):
    """Bind each policy channel to its exact articulation joint name."""
    names = tuple(names)
    if any(names.count(name) != 1 for name in MODEL_JOINT_NAMES):
        raise ValueError("DROID requires seven unique Panda joints and finger_joint")
    return tuple(names.index(name) for name in MODEL_JOINT_NAMES)


class DroidRobot:
    """Expose measured policy joints while PhysX owns the passive finger joints."""

    def __init__(self, articulation, end_effector):
        self.articulation = articulation
        self.end_effector = end_effector

    def get_joint_positions(self):
        import numpy as np

        values = np.asarray(self.articulation.get_joint_positions(), dtype=np.float64)
        selected = values[np.asarray(joint_indices(self.articulation.dof_names))]
        if not np.isfinite(selected).all():
            raise ValueError("DROID measured joint positions must be finite")
        return selected

    def set_joint_positions(self, values):
        import numpy as np

        values = np.asarray(values, dtype=np.float64)
        if values.shape != (8,) or not np.isfinite(values).all():
            raise ValueError("DROID reset requires eight finite joint positions")
        self.articulation.set_joint_positions(
            values, joint_indices=np.asarray(joint_indices(self.articulation.dof_names)))

    def apply_policy_target(self, target):
        import numpy as np
        from isaacsim.core.utils.types import ArticulationAction

        values = np.asarray(target, dtype=np.float64).copy()
        if values.shape != (8,) or not np.isfinite(values).all():
            raise ValueError("DROID target requires eight finite action channels")
        values[7] = GRIPPER_CLOSED_ANGLE if values[7] > 0.5 else 0.0
        self.articulation.apply_action(ArticulationAction(
            joint_positions=values,
            joint_indices=np.asarray(joint_indices(self.articulation.dof_names))))


def create_robot(world):
    """Select the supported Robotiq variant before initializing physics views."""
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.robot.manipulators.examples")
    from isaacsim.robot.manipulators.examples.franka import Franka

    # The helper authors the installed asset reference; its Panda gripper helper
    # is never initialized or used after selecting the DROID accessory.
    Franka(prim_path="/World/Franka", name="droid_asset_reference")
    variants = world.stage.GetPrimAtPath("/World/Franka").GetVariantSet("Gripper")
    if "Robotiq_2F_85" not in variants.GetVariantNames():
        raise RuntimeError("Installed Franka asset lacks the DROID gripper variant")
    if not variants.SetVariantSelection("Robotiq_2F_85"):
        raise RuntimeError("Could not select the DROID gripper variant")
    articulation = world.scene.add(SingleArticulation(prim_path="/World/Franka", name="franka"))
    end_effector = world.scene.add(SingleRigidPrim(prim_path=GRIPPER_BASE, name="droid_end_effector"))
    return DroidRobot(articulation, end_effector)


def optical_config(view, mounts="native_wide"):
    """Resolve optical settings from the same rig used to author the camera.

    Args:
        view: Exterior or wrist camera name.
        mounts: Named fixed calibration pair.
    Returns:
        Lens, aperture, clipping and exposure settings in Isaac units.
    Raises:
        ValueError: The mount pair is unsupported.
        KeyError: The camera view is unsupported.
    """
    return {
        "focal_length": camera_calibration(mounts)[view]["focal_length"],
        "horizontal_aperture": 5.376, "vertical_aperture": 3.024,
        "clipping_range": (0.01, 100.0), "focus_distance": 28.0, "f_stop": 0.0,
    }


def configure_camera(stage, view, mounts="native_wide"):
    """Author a fixed mount; the wrist camera inherits the rigid body.

    Args:
        stage: Active USD stage containing the camera prim.
        view: Exterior or wrist camera name.
        mounts: Named fixed calibration pair.
    Returns:
        None.
    Raises:
        ValueError: The mount pair is unsupported.
        KeyError: The camera view is unsupported.
    """
    import numpy as np
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(CAMERA_PATHS[view])
    values = camera_calibration(mounts)[view]
    transform = UsdGeom.Xformable(prim)
    transform.ClearXformOpOrder()
    quaternion = np.asarray(values["quaternion_wxyz"], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    # One fixed local transform lets the rigid-body parent supply wrist motion.
    matrix = Gf.Matrix4d().SetRotate(
        Gf.Quatd(float(quaternion[0]), Gf.Vec3d(*map(float, quaternion[1:]))))
    matrix.SetTranslateOnly(Gf.Vec3d(*values["position"]))
    transform.AddTransformOp().Set(matrix)
    camera = UsdGeom.Camera(prim)
    config = optical_config(view, mounts)
    camera.CreateFocalLengthAttr().Set(config["focal_length"])
    camera.CreateHorizontalApertureAttr().Set(config["horizontal_aperture"])
    camera.CreateVerticalApertureAttr().Set(config["vertical_aperture"])
    camera.CreateClippingRangeAttr().Set(Gf.Vec2f(*config["clipping_range"]))
    camera.CreateFocusDistanceAttr().Set(config["focus_distance"])
    camera.CreateFStopAttr().Set(config["f_stop"])


def camera_pose(stage, view):
    """Read the actual rendered camera frame, without tracking the task object."""
    import numpy as np
    from pxr import Gf, Usd, UsdGeom

    transform = UsdGeom.Xformable(stage.GetPrimAtPath(CAMERA_PATHS[view])).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default())
    eye = np.asarray(transform.ExtractTranslation(), dtype=np.float64)
    forward = np.asarray(transform.TransformDir(Gf.Vec3d(0, 0, -1)), dtype=np.float64)
    up = np.asarray(transform.TransformDir(Gf.Vec3d(0, 1, 0)), dtype=np.float64)
    return eye, eye + forward, up


def grasp_region(stage):
    """Locate the rendered fingers for framing, not for the physical verdict.

    Native CAD body origins coincide with the flange when open. Their rendered
    geometry bounds locate the grasp region that the wrist camera must see.
    Contact and lift checks still use physics-backed readback independently.
    """
    import numpy as np
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
    centers = []
    for path in CONTACT_BODIES:
        bounds = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
        if bounds.IsEmpty():
            raise RuntimeError("Native DROID finger geometry has no usable bounds")
        centers.append(np.asarray((bounds.GetMin() + bounds.GetMax()) / 2, dtype=np.float64))
    point = np.mean(centers, axis=0)
    if not np.isfinite(point).all():
        raise RuntimeError("Native DROID grasp region must be finite")
    return point


def policy_image(buffer):
    """Match OpenPI's aspect-preserving resize and centered black padding."""
    import numpy as np
    from PIL import Image

    source = buffer.numpy() if callable(getattr(buffer, "numpy", None)) else buffer
    rgb = np.asarray(source)
    if rgb.shape != (180, 320, 3) or rgb.dtype != np.uint8:
        raise ValueError("DROID camera requires native 320x180 uint8 RGB")
    image = Image.fromarray(rgb).resize((224, 126), resample=Image.Resampling.BILINEAR)
    return np.pad(np.asarray(image), ((49, 49), (0, 0), (0, 0)))
