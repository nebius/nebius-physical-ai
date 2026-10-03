"""Exercise real USD visibility edits and CPU image-evidence failure contracts."""

import numpy as np
import pytest

from npa.workflows.navigation.cameras import compare_frames, validate_frame
from npa.workflows.navigation.visibility import hide_robot_geometry


def frames(config):
    rgb = np.full((config.height, config.width, 3), 64, dtype=np.uint8)
    depth = np.ones((config.height, config.width), dtype=np.float32)
    baseline = validate_frame(rgb, depth, config)
    moved = validate_frame(rgb + 64, depth + 0.25, config)
    return baseline, moved


def test_camera_positive_and_invariance_controls(camera_config):
    baseline, moved = frames(camera_config)
    evidence = compare_frames(baseline, baseline, moved, camera_config)
    assert evidence["rgb"]["changed_fraction"] == 1
    assert evidence["depth"]["maximum_peer_delta"] == 0


@pytest.mark.parametrize(
    "mode", ["frozen-rgb", "frozen-depth", "peer-rgb", "peer-depth"]
)
def test_camera_rejects_frozen_and_visible_peers(camera_config, mode):
    baseline, moved = frames(camera_config)
    hidden = {key: array.copy() for key, array in baseline.items()}
    if mode.startswith("frozen"):
        moved[mode.split("-")[1]] = baseline[mode.split("-")[1]]
    else:
        hidden[mode.split("-")[1]] += 0.2
    with pytest.raises(ValueError, match="peer visibility or frozen"):
        compare_frames(baseline, hidden, moved, camera_config)


@pytest.mark.parametrize(
    "mode", ["nan", "inf", "zero", "empty-rgb", "wrong-shape", "wrong-dtype"]
)
def test_camera_buffers_reject_invalid_evidence(camera_config, mode):
    rgb = np.full((4, 6, 3), 64, dtype=np.uint8)
    depth = np.ones((4, 6), dtype=np.float32)
    if mode in {"nan", "inf", "zero"}:
        depth[0, 0] = {"nan": np.nan, "inf": np.inf, "zero": 0}[mode]
    elif mode == "empty-rgb":
        rgb[:] = 0
    elif mode == "wrong-shape":
        rgb = rgb[:2]
    else:
        rgb = rgb.astype(float)
    with pytest.raises(ValueError):
        validate_frame(rgb, depth, camera_config)


def test_usd_hides_collision_meshes_without_touching_physics(usd_environment, recipe):
    from pxr import UsdGeom, UsdPhysics

    env, adapter, inventory = usd_environment
    evidence = hide_robot_geometry(adapter, env, recipe)
    assert len(evidence["hidden_gprims"]) == 2
    stage = env.unwrapped.sim.stage
    for path in evidence["hidden_gprims"]:
        prim = stage.GetPrimAtPath(path)
        assert UsdGeom.Imageable(prim).ComputeVisibility() == "invisible"
        assert UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is True
    for root in inventory["robot_roots"]:
        assert stage.GetPrimAtPath(root).IsActive()
        assert (
            UsdGeom.Imageable(stage.GetPrimAtPath(root)).ComputeVisibility()
            == "inherited"
        )
    hide_robot_geometry(adapter, env, recipe, author=False)


@pytest.mark.parametrize(
    "mode",
    [
        "attachment",
        "light",
        "instance",
        "sensor-child",
        "link-gprim",
        "missing-camera",
        "reset-visible",
    ],
)
def test_usd_coverage_and_unsafe_visibility_fail_closed(usd_environment, recipe, mode):
    from pxr import UsdGeom, UsdLux, UsdPhysics

    env, adapter, inventory = usd_environment
    stage = env.unwrapped.sim.stage
    if mode == "attachment":
        UsdGeom.Cube.Define(stage, "/World/UnlistedAttachment")
    elif mode == "light":
        UsdLux.SphereLight.Define(stage, "/World/Robot0/Headlight")
    elif mode == "instance":
        stage.GetPrimAtPath("/World/Robot0").SetInstanceable(True)
        stage.GetPrimAtPath("/World/Robot0").GetReferences().AddInternalReference(
            "/World/Robot1"
        )
    elif mode == "sensor-child":
        camera = "/World/Robot0/Link/Visual/Camera"
        UsdGeom.Camera.Define(stage, camera)
        inventory["camera_prims"][0] = camera
    elif mode == "link-gprim":
        UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath("/World/Robot0/Link/Visual"))
    elif mode == "missing-camera":
        inventory["camera_prims"][0] = "/World/Missing"
    else:
        hide_robot_geometry(adapter, env, recipe)
        UsdGeom.Imageable(
            stage.GetPrimAtPath("/World/Robot0/Link/Visual")
        ).MakeVisible()
    with pytest.raises(ValueError):
        hide_robot_geometry(adapter, env, recipe, author=mode != "reset-visible")


def test_nonfinite_comparison_cannot_pass_invariance(camera_config):
    baseline, moved = frames(camera_config)
    baseline["rgb"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        compare_frames(baseline, baseline, moved, camera_config)
