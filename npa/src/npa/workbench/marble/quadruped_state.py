"""Record articulated Go1 link poses, contacts, and calibrated collection cameras."""

import numpy as np

from .api import MarbleError
from .rover_physics import _WORLD_FROM_BULLET, _look_at


def cameras(bullet, position, orientation, center, heading):
    """Place the physical front camera and a stabilized inspection camera.

    Args: Physics client, measured body pose, filtered observer target, route heading.
    Returns: Onboard and observer camera-to-world matrices in the Marble frame.
    Raises: None.
    """
    rotation = np.asarray(bullet.getMatrixFromQuaternion(orientation)).reshape(3, 3)
    onboard = np.eye(4)
    onboard[:3, :3] = (
        _WORLD_FROM_BULLET @ rotation @ np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    )
    onboard[:3, 3] = _WORLD_FROM_BULLET @ (position + rotation @ [0.3, 0, 0.05])
    tracking = np.array(
        [
            [np.cos(heading), -np.sin(heading), 0],
            [np.sin(heading), np.cos(heading), 0],
            [0, 0, 1],
        ]
    )
    observer = center + tracking @ [0.4, -1.55, 0.7]
    return onboard, _look_at(observer, center + [0, 0, 0.02])


def link_poses(bullet, robot):
    """Record actual link frames, correcting the base inertial-frame offset.

    Args: Physics client and robot ID.
    Returns: Mapping from URDF link name to position and XYZW orientation.
    Raises: None.
    """
    position, orientation = bullet.getBasePositionAndOrientation(robot)
    dynamics = bullet.getDynamicsInfo(robot, -1)
    inverse = bullet.invertTransform(dynamics[3], dynamics[4])
    pose = bullet.multiplyTransforms(position, orientation, *inverse)
    result = {bullet.getBodyInfo(robot)[0].decode(): [list(v) for v in pose]}
    for index in range(bullet.getNumJoints(robot)):
        state = bullet.getLinkState(robot, index, computeForwardKinematics=True)
        name = bullet.getJointInfo(robot, index)[12].decode()
        result[name] = [list(state[4]), list(state[5])]
    return result


def record(bullet, robot, warehouse, joints, command, action, torques, cameras, time):
    """Capture one synchronized physical state and its sensor poses.

    Args: Physics objects, motor state, calibrated cameras, and simulation time.
    Returns: Serializable trajectory sample with per-foot normal forces.
    Raises: None.
    """
    position, orientation = bullet.getBasePositionAndOrientation(robot)
    contacts = bullet.getContactPoints(robot, warehouse)
    feet = dict.fromkeys(("FR", "FL", "RR", "RL"), 0.0)
    for contact in contacts:
        name = (
            bullet.getJointInfo(robot, contact[3])[12].decode()
            if contact[3] >= 0
            else "base"
        )
        if "foot" in name.lower() and name[:2] in feet:
            feet[name[:2]] += contact[9]
    states = bullet.getJointStates(robot, joints)
    velocity, angular = bullet.getBaseVelocity(robot)
    return {
        "time_seconds": time,
        "position_bullet": list(position),
        "orientation_xyzw_bullet": list(orientation),
        "linear_velocity_bullet": list(velocity),
        "angular_velocity_bullet": list(angular),
        "heading_radians": bullet.getEulerFromQuaternion(orientation)[2],
        "joint_positions": [s[0] for s in states],
        "joint_velocities": [s[1] for s in states],
        "joint_torques": torques.tolist(),
        "policy_action": action.tolist(),
        "velocity_command": command.tolist(),
        "contacts": len(contacts),
        "foot_normal_forces": feet,
        "contact_normal_force": sum(c[9] for c in contacts),
        "links": link_poses(bullet, robot),
        "camera_to_world": cameras[0].tolist(),
        "observer_camera_to_world": cameras[1].tolist(),
    }


def validate_motion(bullet, robot, route):
    """Reject falling or unsupported motion during collection.

    Args: Physics client, robot ID, and measured support route.
    Returns: None.
    Raises: MarbleError when the body falls or leaves the qualified height range.
    """
    position, orientation = bullet.getBasePositionAndOrientation(robot)
    rotation = np.asarray(bullet.getMatrixFromQuaternion(orientation)).reshape(3, 3)
    height = position[2] - route["start"][2]
    if (
        not np.isfinite(position).all()
        or rotation[2, 2] < 0.8
        or not -0.18 < height < 0.3
    ):
        raise MarbleError("Go1 lost upright supported locomotion; collection rejected")
