"""Drive the real Go1 articulation using a pinned policy against the Marble mesh."""

import hashlib
import json

import numpy as np

from .api import MarbleError
from .quadruped_assets import policy_path
from .quadruped_control import DEFAULT_ANGLES, apply_torques, observe, route_command
from .quadruped_state import cameras, record, validate_motion
from .rover_physics import _route, _warehouse


def _spawn(bullet, root, route):
    robot = bullet.loadURDF(
        str(root / "go1.urdf"),
        route["start"],
        bullet.getQuaternionFromEuler([0, 0, route["heading"]]),
        flags=bullet.URDF_USE_INERTIA_FROM_FILE,
    )
    joints = [
        i
        for i in range(bullet.getNumJoints(robot))
        if bullet.getJointInfo(robot, i)[2] == bullet.JOINT_REVOLUTE
    ]
    expected = [
        f"{leg}_{joint}_joint"
        for leg in ("FR", "FL", "RR", "RL")
        for joint in ("hip", "thigh", "calf")
    ]
    if [bullet.getJointInfo(robot, i)[1].decode() for i in joints] != expected:
        raise MarbleError("Go1 joint order differs from the pinned policy contract")
    for joint, angle in zip(joints, DEFAULT_ANGLES):
        bullet.resetJointState(robot, joint, float(angle))
    for index in range(-1, bullet.getNumJoints(robot)):
        bullet.changeDynamics(
            robot, index, lateralFriction=0.8, linearDamping=0, angularDamping=0
        )
    bullet.setJointMotorControlArray(
        robot, joints, bullet.VELOCITY_CONTROL, forces=[0] * 12
    )
    return robot, joints


def _policy_action(policy, observation):
    action = policy.run(["continuous_actions"], {"obs": observation[None]})[0][0]
    if action.shape != (12,) or not np.isfinite(action).all():
        raise MarbleError("Upstream Go1 policy returned invalid motor actions")
    return action


def _rollout(bullet, root, warehouse, route, robot, joints, request, policy):
    action, target = np.zeros(12, dtype=np.float32), DEFAULT_ANGLES.copy()
    command = np.zeros(3)
    center, records = np.array(route["start"]), []
    interval = 250 // request.sensor_hz
    settle = 500
    for step in range(settle + request.frames * interval):
        if step % 5 == 0:
            command = route_command(
                bullet, robot, route, request.speed_mps if step >= settle else 0
            )
            observation, _, _ = observe(bullet, robot, joints, action, command)
            action = _policy_action(policy, observation)
            target = DEFAULT_ANGLES + 0.5 * action
        torques = apply_torques(bullet, robot, joints, target)
        bullet.stepSimulation()
        validate_motion(bullet, robot, route)
        if step >= settle and (step - settle + 1) % interval == 0:
            position, orientation = bullet.getBasePositionAndOrientation(robot)
            center = center * 0.65 + np.asarray(position) * 0.35
            poses = cameras(
                bullet, np.asarray(position), orientation, center, route["heading"]
            )
            records.append(
                record(
                    bullet,
                    robot,
                    warehouse,
                    joints,
                    command,
                    action,
                    torques,
                    poses,
                    (len(records) + 1) / request.sensor_hz,
                )
            )
    return records


def simulate_quadruped(root, world, request):
    """Record real torque-driven articulated locomotion without teleporting the body.

    Args: Verified world/robot directory, world manifest, and collection request.
    Returns: Recorded states and measured physics/policy provenance.
    Raises: MarbleError when policy, physical validity, or travel requirements fail.
    """
    import onnxruntime as ort
    import pybullet
    from pybullet_utils.bullet_client import BulletClient

    policy = ort.InferenceSession(
        str(policy_path(root)), providers=["CPUExecutionProvider"]
    )
    bullet = BulletClient(connection_mode=pybullet.DIRECT)
    try:
        bullet.setGravity(0, 0, -9.81)
        bullet.setTimeStep(0.004)
        bullet.setPhysicsEngineParameter(numSolverIterations=100)
        warehouse = _warehouse(bullet, root, world)
        route = _route(bullet, warehouse)
        route["start"][2] += 0.09
        robot, joints = _spawn(bullet, root, route)
        records = _rollout(
            bullet, root, warehouse, route, robot, joints, request, policy
        )
    finally:
        bullet.disconnect()
    (root / "trajectory.json").write_text(json.dumps(records, allow_nan=False))
    summary = _summary(root, records, route, request)
    return records, summary


def _summary(root, records, route, request):
    positions = np.array([r["position_bullet"] for r in records])
    distance = float(np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1).sum())
    contact_frames = sum(sum(r["foot_normal_forces"].values()) > 1 for r in records)
    if distance < 1 or contact_frames < len(records) * 0.8:
        raise MarbleError(
            f"Go1 failed supported walking: {distance:.3f} meters, {contact_frames}/{len(records)} foot-contact frames"
        )
    return {
        "engine": "pybullet",
        "device": "CPU",
        "step_seconds": 0.004,
        "sensor_hz": request.sensor_hz,
        "distance_m": distance,
        "route": route,
        "embodiment": "Unitree Go1",
        "actuated_joints": 12,
        "foot_contact_frames": contact_frames,
        "controller": "MuJoCo Playground pretrained ONNX walking policy; Bullet PD torque adapter",
        "policy_device": "CPU",
        "policy_sha256": hashlib.sha256(policy_path(root).read_bytes()).hexdigest(),
        "trained_in_this_run": False,
        "collision_geometry": "original Marble concave collision triangles",
    }
