from __future__ import annotations

import ast
import copy
import json
import math
from types import SimpleNamespace

import pytest

from npa.workflows.sim2real import camera_views
from npa.workflows.sim2real.byo_isaac_eval import ISAAC_EVAL_SCRIPT
from npa.workflows.sim2real.byo_isaac_policy_rollout import ISAAC_ROLLOUT_SCRIPT
from npa.workflows.sim2real.camera_views import (
    CAMERA_VIEW_SPECS,
    camera_metadata,
    camera_rotation_for_isaac_lab,
    camera_view_names,
    camera_views_json,
)


def test_default_camera_views_cover_primary_side_and_overhead() -> None:
    assert camera_view_names() == ("primary", "side", "overhead")
    payload = json.loads(camera_views_json())
    assert [view["name"] for view in payload] == ["primary", "side", "overhead"]


def test_camera_view_aliases_are_ordered_deduplicated_and_keep_primary() -> None:
    assert camera_view_names("left,top,left") == ("primary", "side", "overhead")
    assert camera_view_names("front,side") == ("primary", "side")


def test_camera_view_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="unknown Sim2Real camera view"):
        camera_view_names("primary,diagonal")


def test_camera_view_quaternions_are_normalized_and_poses_are_distinct() -> None:
    positions = {spec.position for spec in CAMERA_VIEW_SPECS.values()}
    assert len(positions) == 3
    for spec in CAMERA_VIEW_SPECS.values():
        assert math.isclose(
            sum(value * value for value in spec.rotation), 1.0, abs_tol=1e-6
        )


@pytest.mark.parametrize("version", ["2.3.2.post1", "3.0.0b2.post1"])
@pytest.mark.parametrize("name", ["primary", "side", "overhead"])
def test_camera_rotation_preserves_world_optical_axis(version: str, name: str) -> None:
    serialized = CAMERA_VIEW_SPECS[name].rotation
    converted = camera_rotation_for_isaac_lab(serialized, isaac_lab_version=version)
    # Interpret the actual sensor argument using that generation's convention.
    if version.startswith("3."):
        x, y, z, w = converted
    else:
        w, x, y, z = converted
    forward = (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y))
    expected = {
        "primary": (math.cos(math.radians(12)), 0, -math.sin(math.radians(12))),
        "side": (0, math.cos(math.radians(12)), -math.sin(math.radians(12))),
        "overhead": (0, 0, -1),
    }
    assert forward == pytest.approx(expected[name], abs=1e-12)
    assert serialized == CAMERA_VIEW_SPECS[name].rotation


def test_camera_rotation_resolves_installed_distribution_not_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = []

    def installed_version(package: str) -> str:
        requested.append(package)
        return "3.0.0b2.post1"

    monkeypatch.setenv("ISAAC_LAB_VERSION", "2.3.2.post1")
    monkeypatch.setattr(camera_views.metadata, "version", installed_version)
    assert camera_rotation_for_isaac_lab((1, 0, 0, 0)) == (0, 0, 0, 1)
    assert requested == ["isaaclab"]


@pytest.mark.parametrize("version", ["", "unknown", "1.4.1", "4.0.0"])
def test_camera_rotation_rejects_unknown_generation(version: str) -> None:
    with pytest.raises(ValueError, match="unsupported Isaac Lab camera quaternion"):
        camera_rotation_for_isaac_lab((1, 0, 0, 0), isaac_lab_version=version)


@pytest.mark.parametrize("rotation", [(1, 0, 0), (1, 0, float("nan"), 0)])
def test_camera_rotation_rejects_invalid_pose(rotation: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="four finite WXYZ values"):
        camera_rotation_for_isaac_lab(rotation, isaac_lab_version="3.0.0b2.post1")


@pytest.mark.parametrize("version", ["2.3.2.post1", "3.0.0b2.post1"])
@pytest.mark.parametrize(
    "script", [ISAAC_ROLLOUT_SCRIPT, ISAAC_EVAL_SCRIPT], ids=["rollout", "eval"]
)
def test_actual_isaac_sensor_boundary_converts_without_changing_artifact_poses(
    version: str, script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execute the real embedded sensor-construction block with a fake sensor API."""
    monkeypatch.setattr(camera_views.metadata, "version", lambda _: version)
    tree = ast.parse(script)
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "npa.workflows.sim2real.camera_views"
    ]
    camera_key = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_camera_key"
    )
    sensor_loop = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Name)
        and node.iter.id == "CAMERA_VIEWS"
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "setattr"
            for child in ast.walk(node)
        )
    )

    class SensorConfig(SimpleNamespace):
        OffsetCfg = SimpleNamespace

    poses = camera_metadata("", width=640, height=480)
    original = copy.deepcopy(poses)
    scene = SimpleNamespace()
    namespace = {
        "CAMERA_VIEWS": poses,
        "CAPTURE_WIDTH": 640,
        "CAPTURE_HEIGHT": 480,
        "CameraType": SensorConfig,
        "TiledCameraCfg": SensorConfig,
        "env_cfg": SimpleNamespace(scene=scene),
        "sim_utils": SimpleNamespace(PinholeCameraCfg=SimpleNamespace),
    }
    module = ast.Module(body=[*imports, camera_key, sensor_loop], type_ignores=[])
    exec(
        compile(ast.fix_missing_locations(module), "<actual-camera-boundary>", "exec"),
        namespace,
    )
    assert len(vars(scene)) == 3
    for pose, sensor in zip(poses, vars(scene).values(), strict=True):
        w, x, y, z = pose["rotation"]
        expected = (x, y, z, w) if version.startswith("3.") else (w, x, y, z)
        assert sensor.offset.rot == expected
        assert sensor.offset.pos == tuple(pose["position"])
        assert sensor.offset.convention == "world"
        assert (sensor.width, sensor.height) == (640, 480)
    assert poses == original
    assert all(pose["quaternion_order"] == "wxyz" for pose in poses)
