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
    -0.05095593945063183,
    -0.0895889309568556,
    0.05077240484677532,
    -2.4059685385594194,
    0.006180800038750362,
    2.3164848021835893,
    -0.004580619403992225,
)
GRIPPER_ROOT = "/World/Franka/Robotiq_2F_85_edit/Robotiq_2F_85"
GRIPPER_BASE = f"{GRIPPER_ROOT}/base_link"
GRIPPER_FLANGE = "/World/Franka/panda_link7"
GRIPPER_MOUNT_OFFSET_METERS = 0.018174
GRIPPER_BASE_IN_FLANGE_METERS = (0.0, 0.0, 0.125174)
CONTACT_BODIES = tuple(
    f"{GRIPPER_ROOT}/{side}_inner_finger" for side in ("left", "right")
)
CAMERA_PATHS = {
    "exterior": "/World/ExteriorDroid",
    "wrist": f"{GRIPPER_BASE}/wrist_cam",
}
NATIVE_POLICY_RESOLUTION = (180, 320)
ARM_STIFFNESS_NM_PER_RAD = 400.0
ARM_DAMPING_NM_S_PER_RAD = 80.0
GRIPPER_STIFFNESS_NM_PER_RAD = math.degrees(100.0)
GRIPPER_DAMPING_NM_S_PER_RAD = math.degrees(0.0002)
GRIPPER_EFFORT_LIMIT_NM = 16.5
ARM_EFFORT_LIMITS_NM = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)
JOINT_VELOCITY_LIMITS_RAD_S = (2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61, 5.0)
CAMERA_CALIBRATION = {
    "exterior": {
        # Low side view keeps the target clear of the forearm across recorded
        # failure poses, while retaining the approaching fingers in frame.
        "position": (0.35, -0.65, 0.38),
        "quaternion_wxyz": (
            0.7760544786384438,
            0.6276901716946259,
            -0.03848190493529904,
            -0.04757769998365795,
        ),
        "focal_length": 2.8,
    },
    "wrist": {
        # A fixed bracket and wider lens retain the target during close approach.
        # The optical frame is calibrated in the native Robotiq body basis;
        # neither this transform nor the exterior camera tracks the object.
        "position": (0.12, -0.031, -0.025),
        "quaternion_wxyz": (
            0.2514280790974722,
            0.6726227449022252,
            0.6804223192506174,
            0.14624647533244078,
        ),
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
        # Reference +X is native +Z, reference +Y is native -Y, and reference
        # +Z is native +X. Preserve the camera's physical side of the fingers.
        "position": (-0.074, 0.031, 0.011),
        "quaternion_wxyz": (
            -0.113823876022183,
            -0.7041526740254301,
            0.6921340038864418,
            0.11028897304012761,
        ),
        "focal_length": 2.8,
    },
}
DROID_DETAIL_CALIBRATION = {
    "exterior": {**DROID_REFERENCE_CALIBRATION["exterior"], "focal_length": 2.8},
    "wrist": DROID_REFERENCE_CALIBRATION["wrist"].copy(),
}
TASK_VIEW_CALIBRATION = {
    "exterior": {
        # Looking toward the robot from in front of the table keeps the forearm
        # behind the target during approach. The lens resolves the cube without
        # a moving camera or a crop applied to the policy input.
        "position": (0.95, -0.8, 0.68),
        "quaternion_wxyz": (
            0.8097311153426455,
            0.5133783487801894,
            0.15218591261237785,
            0.24003674687851262,
        ),
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
    choices = {
        "native_wide": CAMERA_CALIBRATION,
        "droid_reference": DROID_REFERENCE_CALIBRATION,
        "droid_detail": DROID_DETAIL_CALIBRATION,
        "task_view": TASK_VIEW_CALIBRATION,
    }
    if mounts not in choices:
        raise ValueError("Unsupported camera_mounts")
    return {view: values.copy() for view, values in choices[mounts].items()}


def joint_indices(names):
    """Bind each policy channel to its exact articulation joint name."""
    names = tuple(names)
    if any(names.count(name) != 1 for name in MODEL_JOINT_NAMES):
        raise ValueError("DROID requires seven unique Panda joints and finger_joint")
    return tuple(names.index(name) for name in MODEL_JOINT_NAMES)


class DroidRobot:
    """Expose measured policy joints while PhysX owns the passive finger joints."""

    def __init__(self, articulation, end_effector, dynamics=None, flange=None):
        self.articulation = articulation
        self.end_effector = end_effector
        self.dynamics = dynamics
        self.flange = flange

    def verify_dynamics(self):
        """Read effective PhysX units after reset and reject incompatible drives."""
        import numpy as np

        if self.dynamics is None:
            raise RuntimeError("DROID physics profile was not configured")
        indices = np.asarray(joint_indices(self.articulation.dof_names))
        properties = self.articulation.dof_properties[indices]
        expected = {
            "stiffness": np.asarray(
                [ARM_STIFFNESS_NM_PER_RAD] * 7 + [GRIPPER_STIFFNESS_NM_PER_RAD]
            ),
            "damping": np.asarray(
                [ARM_DAMPING_NM_S_PER_RAD] * 7 + [GRIPPER_DAMPING_NM_S_PER_RAD]
            ),
            "maxEffort": np.asarray([*ARM_EFFORT_LIMITS_NM, GRIPPER_EFFORT_LIMIT_NM]),
            "maxVelocity": np.asarray(JOINT_VELOCITY_LIMITS_RAD_S),
        }
        actual = {}
        for key, wanted in expected.items():
            values = np.asarray(properties[key][: len(wanted)], dtype=float)
            if not np.isfinite(values).all() or not np.allclose(
                values, wanted, rtol=1e-5, atol=1e-5
            ):
                raise RuntimeError(
                    f"DROID effective {key} differs from reference physics"
                )
            actual[key] = values.tolist()
        if (
            self.articulation.get_solver_position_iteration_count() != 64
            or self.articulation.get_solver_velocity_iteration_count() != 0
            or self.articulation.get_enabled_self_collisions()
        ):
            raise RuntimeError(
                "DROID effective solver settings differ from reference physics"
            )
        mount = {}
        if "gripper_mount" in self.dynamics:
            if self.flange is None:
                raise RuntimeError(
                    "DROID gripper mount requires physical flange readback"
                )
            mount = {
                "gripper_mount": _verify_mount_poses(
                    self.flange.get_world_pose(), self.end_effector.get_world_pose()
                )
            }
        return {
            **self.dynamics,
            **mount,
            "effective_joint_properties": actual,
            "verified": True,
        }

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
            values, joint_indices=np.asarray(joint_indices(self.articulation.dof_names))
        )

    def apply_policy_target(self, target):
        import numpy as np
        from isaacsim.core.utils.types import ArticulationAction

        values = np.asarray(target, dtype=np.float64).copy()
        if values.shape != (8,) or not np.isfinite(values).all():
            raise ValueError("DROID target requires eight finite action channels")
        values[7] = GRIPPER_CLOSED_ANGLE if values[7] > 0.5 else 0.0
        self.articulation.apply_action(
            ArticulationAction(
                joint_positions=values,
                joint_indices=np.asarray(joint_indices(self.articulation.dof_names)),
            )
        )


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
    mount = _configure_native_mount(world.stage)
    dynamics = _configure_native_dynamics(world.stage)
    dynamics["gripper_mount"] = mount
    articulation = world.scene.add(
        SingleArticulation(prim_path="/World/Franka", name="franka")
    )
    end_effector = world.scene.add(
        SingleRigidPrim(prim_path=GRIPPER_BASE, name="droid_end_effector")
    )
    flange = world.scene.add(
        SingleRigidPrim(prim_path=GRIPPER_FLANGE, name="droid_flange")
    )
    return DroidRobot(articulation, end_effector, dynamics, flange)


def _configure_native_mount(stage):
    """Match reference finger geometry without redistributing its robot asset.

    The installed accessory inherits the Panda hand's -45 degree yaw and has
    no spacer. DROID's equivalent native gripper basis aligns with link7 and
    includes an 18.174 mm spacer. Correct the fixed joint and initial assembly
    transform together so the physics reset starts from a consistent pose.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    joint = UsdPhysics.FixedJoint(
        stage.GetPrimAtPath(f"{GRIPPER_BASE}/AssemblerFixedJoint")
    )
    if (
        not joint
        or list(map(str, joint.GetBody0Rel().GetTargets()))
        != ["/World/Franka/panda_hand"]
        or list(map(str, joint.GetBody1Rel().GetTargets())) != [GRIPPER_BASE]
    ):
        raise RuntimeError("DROID requires the native hand-to-Robotiq fixed joint")

    def transform(path):
        return UsdGeom.Xformable(
            stage.GetPrimAtPath(path)
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    hand = transform("/World/Franka/panda_hand")
    base = transform(GRIPPER_BASE)
    assembly = stage.GetPrimAtPath(GRIPPER_ROOT)
    rotation = Gf.Quatd(math.cos(math.pi / 8), Gf.Vec3d(0, 0, math.sin(math.pi / 8)))
    delta = Gf.Matrix4d().SetRotate(rotation)
    delta.SetTranslateOnly(Gf.Vec3d(0, 0, GRIPPER_MOUNT_OFFSET_METERS))
    wanted = delta * hand
    local = (
        transform(GRIPPER_ROOT)
        * base.GetInverse()
        * wanted
        * transform(str(assembly.GetParent().GetPath())).GetInverse()
    )
    xform = UsdGeom.Xformable(assembly)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(local)
    joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, GRIPPER_MOUNT_OFFSET_METERS))
    joint.CreateLocalRot0Attr(Gf.Quatf(rotation))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    return {
        "profile": "droid_native_mount_v1",
        "assembly_yaw_degrees": 45.0,
        "assembly_spacer_m": GRIPPER_MOUNT_OFFSET_METERS,
    }


def _verify_mount_poses(flange_pose, gripper_pose):
    """Reject a model embodiment mismatch using the two physical body poses."""
    import numpy as np

    def rotation(q):
        q = np.asarray(q, dtype=float)
        if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-9:
            raise RuntimeError("DROID mount readback has an invalid quaternion")
        w, x, y, z = q / np.linalg.norm(q)
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )

    flange_position, flange_q = flange_pose
    gripper_position, gripper_q = gripper_pose
    frame = rotation(flange_q)
    offset = frame.T @ (np.asarray(gripper_position) - np.asarray(flange_position))
    orientation = frame.T @ rotation(gripper_q)
    if (
        offset.shape != (3,)
        or not np.isfinite(offset).all()
        or not np.allclose(offset, GRIPPER_BASE_IN_FLANGE_METERS, atol=1e-4, rtol=0)
        or not np.allclose(orientation, np.eye(3), atol=1e-4, rtol=0)
    ):
        raise RuntimeError(
            "DROID physical gripper mount differs from reference embodiment"
        )
    return {
        "profile": "droid_native_mount_v1",
        "verified": True,
        "base_in_flange_m": offset.tolist(),
        "axes_in_flange": orientation.tolist(),
    }


def _configure_native_dynamics(stage):
    """Match the reference DROID controller while preserving object gravity.

    USD angular drives use torque per degree; PhysX and the reference actuator
    expose torque per radian. Writing 400/80 directly to USD yields effective
    gains of approximately 22918/4584, so convert before authoring the drives.
    """
    from pxr import PhysxSchema, UsdPhysics

    prims = [
        prim
        for prim in stage.Traverse()
        if str(prim.GetPath()) == "/World/Franka"
        or str(prim.GetPath()).startswith("/World/Franka/")
    ]
    bodies = [prim for prim in prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    roots = [prim for prim in prims if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if len(bodies) < 9 or len(roots) != 1:
        raise RuntimeError(
            "DROID physics requires one articulation and its rigid links"
        )
    for prim in bodies:
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)
    root = PhysxSchema.PhysxArticulationAPI.Apply(roots[0])
    root.CreateSolverPositionIterationCountAttr(64)
    root.CreateSolverVelocityIterationCountAttr(0)
    root.CreateEnabledSelfCollisionsAttr(False)
    for index, name in enumerate(MODEL_JOINT_NAMES):
        matches = [
            prim
            for prim in prims
            if prim.GetName() == name and prim.IsA(UsdPhysics.RevoluteJoint)
        ]
        if len(matches) != 1:
            raise RuntimeError(f"DROID physics requires one revolute {name}")
        prim = matches[0]
        PhysxSchema.PhysxJointAPI.Apply(prim).CreateMaxJointVelocityAttr(
            math.degrees(JOINT_VELOCITY_LIMITS_RAD_S[index])
        )
        arm = index < len(ARM_JOINT_NAMES)
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateStiffnessAttr(
            math.radians(
                ARM_STIFFNESS_NM_PER_RAD if arm else GRIPPER_STIFFNESS_NM_PER_RAD
            )
        )
        drive.CreateDampingAttr(
            math.radians(
                ARM_DAMPING_NM_S_PER_RAD if arm else GRIPPER_DAMPING_NM_S_PER_RAD
            )
        )
        drive.CreateMaxForceAttr(
            ARM_EFFORT_LIMITS_NM[index] if arm else GRIPPER_EFFORT_LIMIT_NM
        )
        drive.CreateTypeAttr("force")
    return {
        "profile": "robolab_droid_jointpos_v1",
        "robot_gravity_compensation": True,
        "gravity_compensated_robot_links": len(bodies),
        "solver_position_iterations": 64,
        "solver_velocity_iterations": 0,
        "self_collisions": False,
        "angular_drive_authoring_units": "degrees",
        "effective_joint_units": "radians",
        "gripper_drive_gains": "reference_asset",
    }


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
        "horizontal_aperture": 5.376,
        "vertical_aperture": 3.024,
        "clipping_range": (0.01, 100.0),
        "focus_distance": 28.0,
        "f_stop": 0.0,
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
        Gf.Quatd(float(quaternion[0]), Gf.Vec3d(*map(float, quaternion[1:])))
    )
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

    transform = UsdGeom.Xformable(
        stage.GetPrimAtPath(CAMERA_PATHS[view])
    ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
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
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )
    centers = []
    for path in CONTACT_BODIES:
        bounds = cache.ComputeWorldBound(
            stage.GetPrimAtPath(path)
        ).ComputeAlignedRange()
        if bounds.IsEmpty():
            raise RuntimeError("Native DROID finger geometry has no usable bounds")
        centers.append(
            np.asarray((bounds.GetMin() + bounds.GetMax()) / 2, dtype=np.float64)
        )
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
