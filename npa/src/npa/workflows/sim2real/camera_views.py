"""Shared multi-camera configuration for real Isaac Sim2Real jobs.

The Isaac scripts are embedded into GPU Jobs, so this module serializes the
validated view poses into JSON that those scripts can consume without importing
the orchestrator environment.  Quaternions use Isaac Lab's ``world`` camera
convention and ``(w, x, y, z)`` ordering; the optical axis points along +X.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

DEFAULT_CAMERA_VIEWS = ("primary", "side", "overhead")
_VIEW_ALIASES = {"front": "primary", "left": "side", "top": "overhead"}


@dataclass(frozen=True)
class CameraViewSpec:
    name: str
    position: tuple[float, float, float]
    rotation: tuple[float, float, float, float]
    focal_length_mm: float = 24.0
    horizontal_aperture_mm: float = 20.955


def _camera_quaternion(
    *, yaw_degrees: float, pitch_degrees: float
) -> tuple[float, float, float, float]:
    """Return ``q_yaw * q_pitch`` as a normalized Isaac ``wxyz`` quaternion."""

    yaw = math.radians(yaw_degrees) / 2.0
    pitch = math.radians(pitch_degrees) / 2.0
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    # q_yaw=(cy,0,0,sy), q_pitch=(cp,0,sp,0)
    return (cy * cp, -sy * sp, cy * sp, sy * cp)


CAMERA_VIEW_SPECS = {
    # Existing proven oblique workspace view, retained as the compatibility stream.
    "primary": CameraViewSpec(
        "primary",
        (-2.0, 0.0, 1.0),
        _camera_quaternion(yaw_degrees=0.0, pitch_degrees=12.0),
    ),
    # Orthogonal table-side view, looking from -Y toward the workspace origin.
    "side": CameraViewSpec(
        "side",
        (0.0, -2.0, 1.0),
        _camera_quaternion(yaw_degrees=90.0, pitch_degrees=12.0),
    ),
    # Top-down view. A +90 degree pitch turns the +X optical axis toward -Z.
    "overhead": CameraViewSpec(
        "overhead",
        (0.0, 0.0, 3.0),
        _camera_quaternion(yaw_degrees=0.0, pitch_degrees=90.0),
    ),
}


def camera_view_names(value: str = "") -> tuple[str, ...]:
    """Parse a comma-separated view list, preserving order and rejecting typos."""

    requested = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not requested:
        return DEFAULT_CAMERA_VIEWS
    names: list[str] = []
    for requested_name in requested:
        name = _VIEW_ALIASES.get(requested_name, requested_name)
        if name not in CAMERA_VIEW_SPECS:
            available = ", ".join(CAMERA_VIEW_SPECS)
            raise ValueError(
                f"unknown Sim2Real camera view {requested_name!r}; choose from {available}"
            )
        if name not in names:
            names.append(name)
    if "primary" not in names:
        names.insert(0, "primary")
    return tuple(names)


def camera_views_json(value: str = "") -> str:
    """Serialize selected view specs for the embedded Isaac scripts."""

    return json.dumps(
        [asdict(CAMERA_VIEW_SPECS[name]) for name in camera_view_names(value)],
        separators=(",", ":"),
    )


def camera_metadata(value: str, *, width: int, height: int) -> list[dict[str, object]]:
    """Return pose and pinhole intrinsics for the selected Isaac cameras."""

    metadata: list[dict[str, object]] = []
    for name in camera_view_names(value):
        spec = CAMERA_VIEW_SPECS[name]
        fx = width * spec.focal_length_mm / spec.horizontal_aperture_mm
        vertical_aperture = spec.horizontal_aperture_mm * height / width
        fy = height * spec.focal_length_mm / vertical_aperture
        metadata.append(
            {
                **asdict(spec),
                "pose_frame": "isaac_world",
                "quaternion_order": "wxyz",
                "optical_axis": "+X",
                "width": int(width),
                "height": int(height),
                "intrinsics_px": {
                    "fx": round(fx, 6),
                    "fy": round(fy, 6),
                    "cx": width / 2.0,
                    "cy": height / 2.0,
                },
            }
        )
    return metadata
