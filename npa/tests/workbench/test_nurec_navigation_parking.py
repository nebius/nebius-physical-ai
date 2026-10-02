"""Reject unsupported stationary controls without changing navigation physics or cohorts."""

import math

import numpy as np
import pytest

from npa.workbench.nurec import navigation_sample_cases as cases
from npa.workbench.nurec import navigation_sample_parking as parking


def _mesh(path, *, hole=False, wall_height=None, duplicate=False):
    axis = np.linspace(-4, 4, 81)
    xx, yy = np.meshgrid(axis, axis)
    points = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    faces = []
    for y in range(80):
        for x in range(80):
            if hole and (x, y) == (40, 40):
                continue
            corner = y * 81 + x
            faces.extend(
                [[corner, corner + 1, corner + 82], [corner, corner + 82, corner + 81]]
            )
    if wall_height is not None:
        first = len(points)
        points = np.vstack(
            (
                points,
                [
                    [0, 0, wall_height],
                    [0, 0.1, wall_height],
                    [0, 0.1, wall_height + 0.1],
                ],
            )
        )
        faces.append([first, first + 1, first + 2])
    if duplicate:
        faces.append(faces[2 * (40 * 80 + 40)])
    np.savez(path, points=points, triangles=np.array(faces, dtype=np.int32))
    return path


def _point(x=0, y=0):
    return {"xy_m": [x, y], "floor_z_m": 0.0, "obstacle_clearance_m": 1.0}


def _focal():
    return {"position_m": [0, 0, 0.6], "heading_rad": math.pi / 2}


def test_dense_stance_covers_rotated_rear_foot_reach():
    offsets = parking._stance_offsets()
    assert offsets.shape == (405, 2)
    np.testing.assert_allclose(offsets.min(axis=0), [-0.35, -0.65])
    np.testing.assert_allclose(offsets.max(axis=0), [0.35, 0.65])
    assert np.any(np.all(np.isclose(offsets, [0, 0]), axis=1))


@pytest.mark.parametrize("invalid", ["hole", "height", "span", "steep", "nan-normal"])
def test_every_dense_ray_requires_finite_level_upward_support(invalid):
    distances = np.full(405, 0.3)
    normals = np.tile([0.0, 0.0, 1.0], (405, 1))
    if invalid == "hole":
        distances[19] = np.inf
    elif invalid == "height":
        distances[19] = 0.45
    elif invalid == "span":
        distances[19] = 0.25
    elif invalid == "steep":
        normals[19, 2] = 0.69
    else:
        normals[19, 0] = np.nan
    assert parking._floor_evidence(distances, normals) is None


def test_supported_ray_measurements_are_retained():
    distances = np.linspace(0.29, 0.31, 405)
    normals = np.tile([0, 0, 0.99], (405, 1))
    evidence = parking._floor_evidence(distances, normals)
    assert evidence["supported_rays"] == 405
    assert evidence["height_range_m"] == pytest.approx(0.02)
    assert evidence["minimum_normal_z"] == 0.99
    assert len(evidence["support_rays"]["floor_z_m"]) == 405
    np.testing.assert_allclose(evidence["support_rays"]["normals"], normals)


@pytest.mark.parametrize("mutation", ["boundary", "hole", "wall", "nonmanifold"])
def test_source_topology_rejects_faces_near_initial_feet(tmp_path, mutation):
    mesh = _mesh(
        tmp_path / "surface.npz",
        hole=mutation == "hole",
        wall_height=-0.01 if mutation == "wall" else None,
        duplicate=mutation == "nonmanifold",
    )
    point = _point(3.7 if mutation == "boundary" else 0, 0)
    assert parking._topology_evidence(parking._surface_index(mesh), point) is None


def test_continuous_upward_source_support_passes(tmp_path):
    mesh = _mesh(tmp_path / "surface.npz")
    evidence = parking._topology_evidence(parking._surface_index(mesh), _point())
    assert evidence["nearby_source_faces"] > 0
    assert evidence["unsafe_source_faces"] == 0


def test_closed_non_upward_geometry_is_rejected_without_boundary_edges(tmp_path):
    from npa.workflows.navigation.contact_surface_geometry import _topology

    points = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.03]])
    triangles = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    neighbors, _normals = _topology(points, triangles)
    assert np.all(neighbors >= 0)
    surface = tmp_path / "closed-obstacle.npz"
    np.savez(surface, points=points, triangles=triangles)
    assert parking._topology_evidence(parking._surface_index(surface), _point()) is None


def test_unrelated_geometry_above_foot_band_is_not_ground_topology(tmp_path):
    mesh = _mesh(tmp_path / "surface.npz", wall_height=0.2)
    assert parking._topology_evidence(parking._surface_index(mesh), _point())


def test_initial_stances_need_separation_including_margin():
    assert not parking._footprints_separated([0.15, 0], _focal())
    assert not parking._footprints_separated([0, 1.301], _focal())
    assert parking._footprints_separated([0, 1.303], _focal())


def test_selection_is_geometry_based_and_independent_of_candidate_order(
    tmp_path, monkeypatch
):
    mesh = _mesh(tmp_path / "surface.npz")
    evidence = parking._floor_evidence(np.full(405, 0.3), np.tile([0, 0, 1], (405, 1)))
    monkeypatch.setattr(parking, "_support", lambda _scene, _xy: evidence)
    grid = [_point(2, 0), _point(-2, 0), _point(0, 0)]
    selected, report = parking.select_parked_pose(mesh, None, grid, _focal())
    reversed_result = parking.select_parked_pose(
        mesh, None, list(reversed(grid)), _focal()
    )
    assert reversed_result == (selected, report)
    assert selected == [-2, 0, 0.6]
    assert report["candidate_counts"] == {
        "grid": 3,
        "dense_support": 3,
        "topology": 3,
        "separated": 2,
    }
    assert report["measured"]["focal_origin_distance_m"] == 2
    assert report["measured"]["supported_rays"] == 405


def test_no_safe_candidate_fails_without_fallback(tmp_path, monkeypatch):
    mesh = _mesh(tmp_path / "surface.npz")
    monkeypatch.setattr(parking, "_support", lambda _scene, _xy: None)
    with pytest.raises(ValueError, match="no separated parked stance"):
        parking.select_parked_pose(mesh, None, [_point(2, 0)], _focal())


def test_actual_raycast_support_uses_rotated_stance(tmp_path):
    pytest.importorskip("open3d")
    mesh = _mesh(tmp_path / "surface.npz")
    scene = cases._scene(mesh)
    evidence = parking._support(scene, [0, 0])
    assert evidence["supported_rays"] == 405
    assert evidence["height_range_m"] < 1e-6
    assert evidence["minimum_normal_z"] == pytest.approx(1)
    assert len(evidence["support_rays"]["source_face_indices"]) == 405
    assert parking._support(scene, [3.8, 0]) is None


def test_probe_selection_preserves_free_obstacle_and_action_contract(monkeypatch):
    marker = {"measured": True}
    monkeypatch.setattr(
        cases, "select_parked_pose", lambda *_args: ([-2, 0, 0.6], marker)
    )
    monkeypatch.setattr(cases, "_probe_support", lambda *_args: None)
    monkeypatch.setattr(cases, "_rays", lambda *_args: np.array([1.0]))
    probes, evidence = cases._probes(None, None, [])
    assert probes["free"] == cases._case(
        "free", 42, [-3.05, -2.2, 0.56624765], [-3.05, 1.4], math.pi / 2
    )
    assert probes["obstacle"] == cases._case(
        "obstacle", 43, [-3.05, -0.85, 0.59664259], [-0.2, 1.5], 0.687223393
    )
    assert probes["actions"] == [[0.6, 0, 0]] * 30
    assert probes["tolerance"] == 0.001
    assert probes["parked"]["position_m"] == [-2, 0, 0.6]
    assert evidence is marker
