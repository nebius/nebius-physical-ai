"""Build synthetic contract-only navigation bundles without native simulation."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from npa.workflows.navigation.contract import Recipe


def case(index):
    return {
        "id": f"case-{index}",
        "seed": index,
        "position_m": [float(index), 0.0, 0.5],
        "heading_rad": 0.0,
        "goal_m": [float(index) + 3.0, 4.0],
    }


@pytest.fixture
def recipe_dict():
    return {
        "schema_version": "npa.navigation.recipe.v1",
        "task": "Fixture-Navigation-v0",
        "adapter_module": "fixture_navigation_adapter",
        "adapter_sha256": "a" * 64,
        "image": "registry.example.invalid/navigation@sha256:" + "b" * 64,
        "scene_file": "warehouse.usdz",
        "scene_sha256": hashlib.sha256(b"fixture-only").hexdigest(),
        "scene_prim": "/World/Warehouse",
        "sensor_mode": "state",
        "num_envs": 2,
        "iterations": 3,
        "episode_steps": 5,
        "goal_tolerance_m": 0.2,
        "minimum_success_rate": 0.5,
        "train_cases": [case(1), case(2)],
        "eval_cases": [case(3), case(4)],
        "probe": {
            "free": case(5),
            "obstacle": case(6),
            "parked": case(7),
            "actions": [[1.0, 0.0], [1.0, 0.0]],
            "tolerance": 1e-5,
        },
    }


@pytest.fixture
def recipe(recipe_dict):
    return Recipe.model_validate(recipe_dict)


@pytest.fixture
def camera_config():
    from npa.workflows.navigation.contract import Camera

    return Camera.model_validate(
        {
            "camera_visibility": "all_robot_geometry_hidden",
            "height": 4,
            "width": 6,
            "rgb_tolerance": 0.005,
            "depth_tolerance_m": 0.001,
            "minimum_changed_fraction": 0.1,
            "translation_m": [0.2, 0.0, 0.0],
            "peer_in_view": case(8),
        }
    )


@pytest.fixture
def raw_bundle(tmp_path, recipe_dict):
    root = tmp_path / "raw"
    root.mkdir()
    (root / "recipe.json").write_text(json.dumps(recipe_dict))
    (root / "warehouse.usdz").write_bytes(b"fixture-only")
    return root


@pytest.fixture
def usd_environment(recipe):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, recipe.scene_prim)
    UsdGeom.Cube.Define(stage, recipe.scene_prim + "/Wall")
    roots, cameras = [], []
    for index in range(recipe.num_envs):
        root = f"/World/Robot{index}"
        prim = UsdGeom.Xform.Define(stage, root).GetPrim()
        UsdPhysics.ArticulationRootAPI.Apply(prim)
        geom = UsdGeom.Cube.Define(stage, root + "/Link/Visual").GetPrim()
        UsdPhysics.CollisionAPI.Apply(geom).CreateCollisionEnabledAttr(True)
        camera = root + "/Link/Camera"
        UsdGeom.Camera.Define(stage, camera)
        roots.append(root)
        cameras.append(camera)
    camera_type = type("FixtureCamera", (), {})
    camera_sensor = camera_type()
    camera_sensor.cfg = SimpleNamespace(prim_path="/World/Robot.*/Link/Camera")
    native = SimpleNamespace(
        sim=SimpleNamespace(stage=stage),
        scene=SimpleNamespace(sensors={"camera": camera_sensor}),
    )
    env = SimpleNamespace(unwrapped=native)
    inventory = {"robot_roots": roots, "attachment_roots": [], "camera_prims": cameras}
    adapter = SimpleNamespace(visibility_paths=lambda _: inventory)
    return env, adapter, inventory
