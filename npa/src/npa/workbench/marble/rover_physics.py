"""Simulate a wheel-driven inspection rover against Marble's actual triangle mesh."""

import json
import math
from pathlib import Path

import numpy as np

from .api import MarbleError

_WORLD_FROM_BULLET = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
_WHEEL_RADIUS = 0.14
_TRACK_WIDTH = 0.48
_STEP = 1 / 240


def _warehouse(bullet, root, world):
    import trimesh

    mesh = trimesh.load(root / "collider.glb", force="scene").to_geometry()
    transform = world["mesh_transform"]
    vertices = np.asarray(mesh.vertices) * transform["scale"] + transform["translation"]
    vertices = vertices @ _WORLD_FROM_BULLET
    if not np.isfinite(vertices).all() or len(mesh.faces) == 0:
        raise MarbleError("Rover requires finite collision triangles")
    collision = bullet.createCollisionShape(
        bullet.GEOM_MESH,
        vertices=vertices.tolist(),
        indices=np.asarray(mesh.faces).flatten().tolist(),
        flags=bullet.GEOM_FORCE_CONCAVE_TRIMESH,
    )
    body = bullet.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision)
    bullet.changeDynamics(body, -1, lateralFriction=0.9)
    return body


def _floor(bullet, warehouse, x, y):
    hit = bullet.rayTest([x, y, 0.9], [x, y, -0.8])[0]
    if hit[0] != warehouse or hit[4][2] < 0.85:
        return None
    return float(hit[3][2])


def _route(bullet, warehouse):
    candidates = []
    for x, y in [(0, 0), (0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5)]:
        ground = _floor(bullet, warehouse, x, y)
        if ground is None:
            continue
        for heading in [math.pi / 2, -math.pi / 2, 0, math.pi]:
            direction = np.array([math.cos(heading), math.sin(heading)])
            clear = _route_clearance(
                bullet, warehouse, np.array([x, y]), direction, ground
            )
            candidates.append((clear, x, y, ground, heading))
    if not candidates or max(candidates)[0] < 2:
        raise MarbleError(
            "Generated collider has no supported two-meter inspection route"
        )
    distance, x, y, ground, heading = max(candidates)
    return {"start": [x, y, ground + 0.26], "heading": heading, "distance": distance}


def _route_clearance(bullet, warehouse, origin, direction, ground):
    distance = 0
    side = np.array([-direction[1], direction[0]])
    for step in np.arange(0.0, 8.25, 0.25):
        position = origin + direction * step
        floors = [
            _floor(bullet, warehouse, *(position + side * offset))
            for offset in [-0.35, 0, 0.35]
        ]
        if any(value is None or abs(value - ground) > 0.12 for value in floors):
            break
        start = [*position, ground + 0.35]
        end = [*(position + direction * 0.65), ground + 0.35]
        if bullet.rayTest(start, end)[0][0] == warehouse:
            break
        distance = float(step)
    return distance


def _rover_urdf():
    body = """<robot name="inspection-rover"><link name="chassis">
      <inertial><mass value="12"/><inertia ixx="0.15" iyy="0.39" izz="0.48" ixy="0" ixz="0" iyz="0"/></inertial>
      <visual><geometry><box size="0.60 0.36 0.18"/></geometry><material name="orange"><color rgba="1 0.42 0.05 1"/></material></visual>
      <visual><origin xyz="0.18 0 0.30"/><geometry><box size="0.09 0.12 0.42"/></geometry><material name="sensor"><color rgba="0.08 0.12 0.16 1"/></material></visual>
      <collision><geometry><box size="0.60 0.36 0.18"/></geometry></collision></link>"""
    wheels = []
    for index, (x, y) in enumerate(
        [(0.21, 0.24), (-0.21, 0.24), (0.21, -0.24), (-0.21, -0.24)]
    ):
        wheels.append(f'''<link name="wheel-{index}">
          <inertial><mass value="0.5"/><inertia ixx="0.003" iyy="0.005" izz="0.003" ixy="0" ixz="0" iyz="0"/></inertial>
          <visual><origin rpy="1.57079632679 0 0"/><geometry><cylinder radius="0.14" length="0.08"/></geometry><material name="tire"><color rgba="0.06 0.07 0.08 1"/></material></visual>
          <collision><origin rpy="1.57079632679 0 0"/><geometry><cylinder radius="0.14" length="0.08"/></geometry></collision></link>
          <joint name="drive-{index}" type="continuous"><parent link="chassis"/><child link="wheel-{index}"/><origin xyz="{x} {y} -0.10"/><axis xyz="0 1 0"/><limit effort="8" velocity="12"/><dynamics damping="0.01" friction="0"/></joint>''')
    return body + "".join(wheels) + "</robot>"


def _spawn(bullet, root, route):
    path = root / "rover.urdf"
    path.write_text(_rover_urdf())
    rover = bullet.loadURDF(
        str(path),
        route["start"],
        bullet.getQuaternionFromEuler([0, 0, route["heading"]]),
    )
    for wheel in range(4):
        bullet.changeDynamics(
            rover,
            wheel,
            lateralFriction=0.8,
            spinningFriction=0.002,
            rollingFriction=0.001,
        )
        bullet.setJointMotorControl2(
            rover, wheel, bullet.VELOCITY_CONTROL, targetVelocity=0, force=8
        )
    for _ in range(480):
        bullet.stepSimulation()
    return rover


def _action(bullet, rover, route):
    position, orientation = bullet.getBasePositionAndOrientation(rover)
    yaw = bullet.getEulerFromQuaternion(orientation)[2]
    direction = np.array([math.cos(route["heading"]), math.sin(route["heading"])])
    delta = np.asarray(position[:2]) - route["start"][:2]
    remaining = route["distance"] - float(delta @ direction)
    cross_track = float(delta @ np.array([-direction[1], direction[0]]))
    desired = route["heading"] - np.clip(cross_track * 0.8, -0.5, 0.5)
    error = math.atan2(math.sin(desired - yaw), math.cos(desired - yaw))
    speed = float(np.clip(remaining * 0.8, 0, 0.55))
    turn = float(np.clip(2 * error, -0.7, 0.7)) if speed > 0.02 else 0.0
    left = (speed - turn * _TRACK_WIDTH / 2) / _WHEEL_RADIUS
    right = (speed + turn * _TRACK_WIDTH / 2) / _WHEEL_RADIUS
    commands = [left, left, right, right]
    bullet.setJointMotorControlArray(
        rover,
        range(4),
        bullet.VELOCITY_CONTROL,
        targetVelocities=commands,
        forces=[8] * 4,
    )
    return commands


def _look_at(position, target):
    forward = np.asarray(target) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = _WORLD_FROM_BULLET @ np.stack([right, down, forward], axis=1)
    pose[:3, 3] = _WORLD_FROM_BULLET @ position
    return pose


def _cameras(bullet, position, orientation):
    rotation = np.asarray(bullet.getMatrixFromQuaternion(orientation)).reshape(3, 3)
    sensor = np.asarray(position) + rotation @ [0.24, 0, 0.56]
    onboard = np.eye(4, dtype=np.float32)
    onboard[:3, :3] = (
        _WORLD_FROM_BULLET @ rotation @ np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    )
    onboard[:3, 3] = _WORLD_FROM_BULLET @ sensor
    observer = np.asarray(position) + rotation @ [-2.1, -1.1, 1.35]
    return onboard, _look_at(observer, np.asarray(position) + [0, 0, 0.2])


def _actor(bullet, rover, pose, request, root, index):
    position = _WORLD_FROM_BULLET.T @ pose[:3, 3]
    rotation = _WORLD_FROM_BULLET.T @ pose[:3, :3]
    view = bullet.computeViewMatrix(
        position, position + rotation[:, 2], -rotation[:, 1]
    )
    aspect = request.width / request.height
    vertical_fov = math.degrees(2 * math.atan(math.tan(math.radians(75) / 2) / aspect))
    projection = bullet.computeProjectionMatrixFOV(vertical_fov, aspect, 0.05, 1000)
    _, _, rgba, depth, segmentation = bullet.getCameraImage(
        request.width,
        request.height,
        view,
        projection,
        renderer=bullet.ER_TINY_RENDERER,
    )
    mask = (np.asarray(segmentation, dtype=np.int64) & ((1 << 24) - 1)) == rover
    z = 1000 * 0.05 / (1000 - (1000 - 0.05) * np.asarray(depth))
    np.savez_compressed(
        root / "actors" / f"{index:04d}.npz",
        rgb=np.asarray(rgba)[..., :3],
        z=z,
        mask=mask,
    )


def _record(bullet, rover, warehouse, commands, index, request, root):
    position, orientation = bullet.getBasePositionAndOrientation(rover)
    onboard, observer = _cameras(bullet, position, orientation)
    contacts = bullet.getContactPoints(rover, warehouse)
    _actor(bullet, rover, observer, request, root, index)
    return {
        "time_seconds": (index + 1) / request.sensor_hz,
        "position_bullet": list(position),
        "orientation_xyzw_bullet": list(orientation),
        "wheel_velocity_command": list(commands),
        "wheel_states": [
            list(state[:2]) for state in bullet.getJointStates(rover, range(4))
        ],
        "contacts": len(contacts),
        "contact_normal_force": sum(contact[9] for contact in contacts),
        "camera_to_world": onboard.tolist(),
        "observer_camera_to_world": observer.tolist(),
    }


def _validate_motion(records):
    positions = np.array([row["position_bullet"] for row in records])
    traveled = float(np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1).sum())
    if not np.isfinite(positions).all() or np.abs(positions[:, 2]).max() > 1.5:
        raise MarbleError("Rover left the supported floor or produced invalid state")
    if traveled < 1 or sum(row["contacts"] > 0 for row in records) < len(records) * 0.8:
        raise MarbleError(
            "Rover did not demonstrate one meter of supported wheel-driven travel"
        )
    return traveled


def _episode(bullet, root, world, request):
    bullet.setGravity(0, 0, -9.81)
    bullet.setTimeStep(_STEP)
    bullet.setPhysicsEngineParameter(
        numSolverIterations=80, deterministicOverlappingPairs=1
    )
    warehouse = _warehouse(bullet, root, world)
    route = _route(bullet, warehouse)
    rover = _spawn(bullet, root, route)
    records = []
    for index in range(request.frames):
        commands = _action(bullet, rover, route)
        for _ in range(240 // request.sensor_hz):
            bullet.stepSimulation()
        records.append(
            _record(bullet, rover, warehouse, commands, index, request, root)
        )
    return records, route


def simulate_rover(root: Path, world, request):
    """Collect rigid-body states and sensor poses from actual driven wheel joints.

    Args: Verified world directory, aligned manifest, and RoverRequest.
    Returns: Trajectory records and measured physics summary.
    Raises: MarbleError on unsupported terrain or insufficient physical motion.
    """
    from pybullet_utils.bullet_client import BulletClient
    import pybullet

    bullet = BulletClient(connection_mode=pybullet.DIRECT)
    (root / "actors").mkdir(exist_ok=True)
    try:
        records, route = _episode(bullet, root, world, request)
        distance = _validate_motion(records)
    finally:
        bullet.disconnect()
    (root / "trajectory.json").write_text(json.dumps(records, allow_nan=False))
    return records, {
        "engine": "pybullet",
        "device": "CPU",
        "step_seconds": _STEP,
        "sensor_hz": request.sensor_hz,
        "distance_m": distance,
        "route": route,
        "embodiment": "four-wheel inspection rover",
        "controller": "heading feedback with wheel velocity motors",
        "collision_geometry": "generated static concave triangle mesh; no substitute floor",
    }
