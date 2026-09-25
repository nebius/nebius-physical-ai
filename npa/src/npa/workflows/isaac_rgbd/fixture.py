"""Author a repository-owned room and four calibrated cameras for renderer qualification."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .contract import REQUEST_SCHEMA, _json_bytes, _sha256, validate_request


def _room():
    objects = [
        ("Floor", (0, 0, -0.1), (5, 5, 0.1), (0.4, 0.4, 0.4)),
        ("East", (5, 0, 2), (0.1, 5, 2), (0.8, 0.2, 0.2)),
        ("West", (-5, 0, 2), (0.1, 5, 2), (0.2, 0.7, 0.2)),
        ("North", (0, 5, 2), (5, 0.1, 2), (0.2, 0.2, 0.8)),
        ("South", (0, -5, 2), (5, 0.1, 2), (0.7, 0.6, 0.2)),
        ("Crate", (2, 1, 0.5), (0.5, 0.5, 0.5), (0.2, 0.6, 0.7)),
    ]
    text = '#usda 1.0\n(defaultPrim = "World"\nmetersPerUnit = 1\nupAxis = "Z")\ndef Xform "World" {\n'
    for name, position, scale, color in objects:
        text += f'''def Cube "{name}" {{
            double size = 2
            double3 xformOp:translate = {position}
            double3 xformOp:scale = {scale}
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            color3f[] primvars:displayColor = [{color}]
        }}\n'''
    return text + 'def DomeLight "Light" {\n float inputs:intensity = 1500\n }\n}\n'


def _fixture_camera(name, forward):
    forward = np.asarray(forward, dtype=float)
    right = np.cross(forward, [0, 0, 1])
    transform = np.eye(4)
    transform[:3, :3] = np.stack([right, [0, 0, -1], forward], axis=1)
    transform[:3, 3] = forward * 0.15
    return {
        "id": name,
        "width": 320,
        "height": 240,
        "intrinsics": [[240, 0, 159.5], [0, 240, 119.5], [0, 0, 1]],
        "T_rig_camera": transform.tolist(),
        "depth_range_m": [0.1, 20.0],
    }


def _fixture_trajectory():
    samples = []
    for index in range(3):
        pose = np.eye(4)
        pose[:3, 3] = [index * 0.1, 0, 1.5]
        samples.append(
            {"timestamp_ns": index * 100_000_000, "T_world_rig": pose.tolist()}
        )
    return samples


def write_fixture(output_dir):
    """Write a static USD room and a short four-camera trajectory, without rendering.

    Args:
        output_dir: New or empty local input-bundle directory.

    Returns:
        Validated request dictionary, also written as request.json.

    Raises:
        ValueError: The destination is not empty.
        OSError: The bundle cannot be written.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("fixture destination must be empty")
    (root / "scene.usda").write_text(_room())
    request = {
        "schema": REQUEST_SCHEMA,
        "scene": "scene.usda",
        "files": {"scene.usda": _sha256(root / "scene.usda")},
        "pointcloud": True,
        "cameras": [
            _fixture_camera(name, forward)
            for name, forward in (
                ("front", [1, 0, 0]),
                ("left", [0, 1, 0]),
                ("rear", [-1, 0, 0]),
                ("right", [0, -1, 0]),
            )
        ],
        "trajectory": _fixture_trajectory(),
    }
    validate_request(request)
    (root / "request.json").write_bytes(_json_bytes(request))
    return request
