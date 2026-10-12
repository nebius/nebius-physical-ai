"""Check Marble-to-Isaac geometry, goal separation, and native workflow wiring."""

import json
from pathlib import Path

import numpy as np
import pytest

from npa.workbench.marble.api import MarbleError
from npa.workbench.marble.navigation_cases import _connected, _split
from npa.workbench.marble.navigation_geometry import collision_geometry, write_scene
from npa.workbench.marble.schemas import NavigationRequest

ROOT = Path(__file__).resolve().parents[3]
IMAGE = "registry.example.invalid/npa-isaac-lab@sha256:" + "a" * 64


def _world(tmp_path):
    import trimesh

    mesh = trimesh.creation.box(extents=[2, 4, 6])
    mesh.export(tmp_path / "collider.glb")
    transform = {"scale": [2, -2, -2], "translation": [0, 0.25, 0]}
    return mesh, {
        "source_kind": "world-api",
        "mesh_transform": transform,
        "splat_transform": transform,
    }


def test_collider_conversion_preserves_original_triangles_and_metric_frame(tmp_path):
    source, world = _world(tmp_path)
    points, faces = collision_geometry(tmp_path, world)
    np.testing.assert_allclose(points.min(axis=0), [-2, -6, -3.75])
    np.testing.assert_allclose(points.max(axis=0), [2, 6, 4.25])
    assert len(faces) == len(source.faces)


@pytest.mark.parametrize("change", ["example", "unaligned", "zero-scale", "nan"])
def test_navigation_rejects_unqualified_source_geometry(tmp_path, change):
    _, world = _world(tmp_path)
    if change == "example":
        world["source_kind"] = "upstream-example"
    elif change == "unaligned":
        world["splat_transform"] = {"scale": [1, 1, 1]}
    elif change == "zero-scale":
        world["mesh_transform"]["scale"][0] = 0
    else:
        world["mesh_transform"]["translation"][0] = float("nan")
    with pytest.raises(MarbleError):
        collision_geometry(tmp_path, world)


def test_packaged_scene_has_only_original_static_colliders_and_source_hashes(tmp_path):
    from pxr import Usd, UsdGeom, UsdPhysics

    _, world = _world(tmp_path)
    points, faces = collision_geometry(tmp_path, world)
    source = {"world_manifest_sha256": "b" * 64, "collider_sha256": "c" * 64}
    destination = tmp_path / "scene.usdz"
    write_scene(destination, points, faces, source)
    stage = Usd.Stage.Open(str(destination))
    assert UsdGeom.GetStageUpAxis(stage) == "Z"
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1
    assert (
        json.loads(stage.GetRootLayer().customLayerData["npa_marble_source"]) == source
    )
    colliders = [
        prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    assert len(colliders) == 1
    mesh = UsdGeom.Mesh(colliders[0])
    np.testing.assert_allclose(mesh.GetPointsAttr().Get(), points)
    np.testing.assert_array_equal(
        np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3), faces
    )
    assert not any(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in stage.Traverse())


def test_training_and_evaluation_goal_locations_are_disjoint_and_repeatable():
    xx, yy = np.meshgrid(np.arange(8), np.arange(8))
    grid = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    train, evaluation = _split(grid, 128, 42)
    assert [train, evaluation] == _split(grid, 128, 42)
    assert {tuple(row["goal_m"]) for row in train}.isdisjoint(
        tuple(row["goal_m"]) for row in evaluation
    )
    assert {row["seed"] for row in train}.isdisjoint(row["seed"] for row in evaluation)
    resets = {
        (tuple(row["position_m"]), tuple(row["goal_m"])) for row in train + evaluation
    }
    assert len(resets) == 256


def test_unsupported_regions_and_insufficient_distinct_cases_fail():
    with pytest.raises(MarbleError, match="no supported"):
        _connected(np.zeros((3, 3), dtype=bool))
    mask = np.array([[True, True, False], [True, False, False], [False, False, True]])
    assert _connected(mask) == [(0, 0), (0, 1), (1, 0)]
    with pytest.raises(MarbleError, match="separate train"):
        _split(np.array([[0, 0, 0], [2, 0, 0]]), 2, 42)


def test_native_workflow_executes_training_and_checkpoint_evaluation():
    from npa.orchestration.npa_workflow.spec import load_spec

    spec = load_spec(ROOT / "workflows/testing/marble-navigation-rl.yaml")
    assert spec.states["prepare"].tool_ref == "workbench.marble.navigation_prepare"
    for name in ("train", "evaluate"):
        argv = spec.states[name].run.argv
        assert "npa.workflows.navigation.stages import run_stage" in argv[2]
        assert argv[3] == name
        assert spec.states[name].resources == "isaac"


def test_preparation_requires_immutable_native_image_identity():
    request = {
        "input_path": "s3://example-bucket/world",
        "output_path": "s3://example-bucket/prepared",
        "run_id": "navigation-test",
        "image": IMAGE,
    }
    assert NavigationRequest(**request).image == IMAGE
    with pytest.raises(ValueError):
        NavigationRequest(
            **{**request, "image": "registry.example.invalid/isaac:latest"}
        )
