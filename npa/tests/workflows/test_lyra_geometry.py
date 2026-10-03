"""Protect metric calibration and imported-scene identity before native manipulation."""

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from npa.workflows.lyra_geometry import align_camera_centers
from npa.workflows.lyra_scene_assembly import _task_transform, _author, _recipe
from npa.workflows.physical_augmentation_contract import read_recipe


def test_camera_calibration_recovers_metric_similarity():
    predicted = np.random.default_rng(17).normal(size=(24, 3))
    rotation = Rotation.from_euler("xyz", [0.2, -0.4, 0.8]).as_matrix()
    measured = 2.7 * predicted @ rotation.T + [0.8, -1.2, 0.4]
    scale, recovered, translation, errors = align_camera_centers(predicted, measured)
    assert scale == pytest.approx(2.7)
    np.testing.assert_allclose(recovered, rotation, atol=1e-12)
    np.testing.assert_allclose(translation, [0.8, -1.2, 0.4], atol=1e-12)
    assert errors.max() < 1e-12


def test_camera_calibration_rejects_static_and_linear_motion():
    centers = np.zeros((8, 3))
    centers[:, 0] = np.arange(8)
    with pytest.raises(ValueError, match="degenerate"):
        align_camera_centers(centers, centers)


@pytest.mark.parametrize("mutation", ["scale", "reflection", "homogeneous", "nan"])
def test_mounting_transform_cannot_change_scale_or_handedness(mutation):
    matrix = np.eye(4)
    if mutation == "scale":
        matrix[0, 0] = 2
    elif mutation == "reflection":
        matrix[0, 0] = -1
    elif mutation == "homogeneous":
        matrix[3, 2] = 1
    else:
        matrix[0, 3] = np.nan
    with pytest.raises(ValueError):
        _task_transform({"world_to_task": matrix.tolist()})


def test_imported_scene_recipe_rejects_changed_geometry(tmp_path):
    pytest.importorskip("pxr")
    from types import SimpleNamespace

    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    _author(points, np.array([[0, 1, 2]]), np.ones((3, 3)), tmp_path / "scene.usdc")
    _recipe(
        SimpleNamespace(run_id="geometry-test"), tmp_path, {"surface_sha256": "a" * 64}
    )
    recipe = read_recipe(tmp_path / "recipe.json")
    assert recipe["presentation"]["scene"] == "lyra-reconstructed-surface-v1"
    with (tmp_path / "scene.usdc").open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="sealed task asset"):
        read_recipe(tmp_path / "recipe.json")


def test_scene_binding_cannot_change_reference_success_gate(tmp_path):
    pytest.importorskip("pxr")
    from types import SimpleNamespace

    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    _author(points, np.array([[0, 1, 2]]), np.ones((3, 3)), tmp_path / "scene.usdc")
    _recipe(
        SimpleNamespace(run_id="geometry-test"), tmp_path, {"surface_sha256": "a" * 64}
    )
    path = tmp_path / "recipe.json"
    recipe = json.loads(path.read_text())
    recipe["success"]["lift_m"] = 0.001
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match="supported contract"):
        read_recipe(path)


@pytest.mark.parametrize(
    "observations,errors,passed",
    [
        ([(100, 95, 90)], [0.01, 0.02], True),
        ([(100, 40, 35)], [0.01, 0.02], False),
        ([(100, 100, 0)], [0.2, 0.3], False),
        ([(100, 0, 0)], [], False),
    ],
)
def test_measured_depth_does_not_accept_holes_or_wrong_surfaces(
    observations, errors, passed
):
    from npa.workflows.lyra_depth_validation import _report

    thresholds = {
        "min_coverage": 0.85,
        "min_inlier_fraction": 0.8,
        "max_mean_error_m": 0.08,
    }
    result = _report(observations, errors, thresholds)
    assert result["passed"] is passed
    assert result["native_physics_validated"] is False
    assert result["thresholds"] == thresholds


def test_html_refuses_actions_from_a_different_scene(tmp_path):
    from npa.workflows.lyra_demo import _trials

    data = {"scene_binding": {"geometry_sha256": "a" * 64}, "trials": []}
    (tmp_path / "demo.html").write_text(
        '<script id="demo-data" type="application/json">'
        + json.dumps(data)
        + "</script>"
    )
    with pytest.raises(ValueError, match="not bound"):
        _trials(tmp_path, {"surface_sha256": "b" * 64})
