"""Run the published Go1 observation/action protocol against measured body state.

Observation ordering follows Google MuJoCo Playground's Apache-2.0
experimental/sim2sim/play_go1_joystick.py. This adapter supplies Bullet states
and route commands; the downloaded upstream network is unchanged.
"""

import math

import numpy as np

DEFAULT_ANGLES = np.array([0.1, 0.9, -1.8, -0.1, 0.9, -1.8] * 2)
TORQUE_LIMITS = np.array([23.7, 23.7, 35.55] * 4)


def observe(bullet, robot, joints, previous_action, command):
    """Build the upstream network's 48 measured observations.

    Args: Physics client, robot ID, ordered joints, prior network output, command.
    Returns: Float32 observation, measured joint angles, and velocities.
    Raises: ValueError on non-finite observations.
    """
    _, orientation = bullet.getBasePositionAndOrientation(robot)
    velocity, angular = bullet.getBaseVelocity(robot)
    rotation = np.asarray(bullet.getMatrixFromQuaternion(orientation)).reshape(3, 3)
    states = bullet.getJointStates(robot, joints)
    angles = np.array([state[0] for state in states])
    velocities = np.array([state[1] for state in states])
    observation = np.concatenate(
        [
            rotation.T @ velocity,
            rotation.T @ angular,
            rotation.T @ [0, 0, -1],
            angles - DEFAULT_ANGLES,
            velocities,
            previous_action,
            command,
        ]
    ).astype(np.float32)
    if not np.isfinite(observation).all():
        raise ValueError("Quadruped produced non-finite policy observations")
    return observation, angles, velocities


def route_command(bullet, robot, route, speed):
    """Track the supported aisle with bounded velocity inputs to the policy.

    Args: Physics client, robot ID, measured route, and desired forward speed.
    Returns: Forward, lateral, and yaw velocity commands.
    Raises: None.
    """
    position, orientation = bullet.getBasePositionAndOrientation(robot)
    yaw = bullet.getEulerFromQuaternion(orientation)[2]
    direction = np.array([math.cos(route["heading"]), math.sin(route["heading"])])
    delta = np.asarray(position[:2]) - route["start"][:2]
    cross_track = float(delta @ np.array([-direction[1], direction[0]]))
    heading = route["heading"] - np.clip(cross_track, -0.4, 0.4)
    error = math.atan2(math.sin(heading - yaw), math.cos(heading - yaw))
    remaining = route["distance"] - float(delta @ direction) - 0.7
    return np.array(
        [np.clip(remaining * 0.7, 0, speed), 0, np.clip(error * 1.5, -0.5, 0.5)]
    )


def apply_torques(bullet, robot, joints, target):
    """Apply measured PD torques with the upstream joint effort limits.

    Args: Physics client, body, ordered joints, and policy joint targets.
    Returns: The twelve commanded joint torques.
    Raises: None.
    """
    states = bullet.getJointStates(robot, joints)
    angles = np.array([state[0] for state in states])
    velocities = np.array([state[1] for state in states])
    torques = np.clip(
        35 * (target - angles) - 0.5 * velocities, -TORQUE_LIMITS, TORQUE_LIMITS
    )
    bullet.setJointMotorControlArray(
        robot, joints, bullet.TORQUE_CONTROL, forces=torques.tolist()
    )
    return torques
