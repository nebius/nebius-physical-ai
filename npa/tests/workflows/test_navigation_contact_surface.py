"""Exercise real Warp CPU geometry queries and fail-closed support boundaries."""

import numpy as np
import pytest

from npa.workflows.navigation.contact_surface import _query, _resolved_margins
from npa.workflows.navigation.contact_surface_geometry import SurfaceMesh

torch = pytest.importorskip("torch")
wp = pytest.importorskip("warp")
kernels = pytest.importorskip("npa.workflows.navigation.contact_surface_kernels")
_connected, _Faces = kernels._connected, kernels._Faces


def _plane(height=0):
    return np.array(
        [[-2, -2, height], [2, -2, height], [2, 2, height], [-2, 2, height]],
        dtype=np.float32,
    ), np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)


def _surface(points, faces):
    wp.init()
    mesh = wp.Mesh(
        points=wp.array(points, dtype=wp.vec3, device="cpu"),
        indices=wp.array(faces.reshape(-1), dtype=wp.int32, device="cpu"),
    )
    return SurfaceMesh(mesh, points, faces)


def _reason(
    points, faces, point=(0, 0, -0.01), radius=0.02, separation=-0.015, root=(0, 0, 0.5)
):
    values = _query(
        _surface(points, faces),
        torch.tensor([point], dtype=torch.float32),
        torch.tensor([radius]),
        torch.tensor([separation]),
        torch.tensor([root], dtype=torch.float32),
    )
    return int(values["support_reason"][0]), values


def test_actual_warp_cpu_recognizes_interior_floor_across_shared_triangle_edge():
    reason, fields = _reason(*_plane())
    assert reason == 1
    assert float(fields["source_distance_m"][0]) == pytest.approx(0.01)
    assert fields["source_point_world_m"][0].tolist() == pytest.approx([0, 0, 0])


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({"point": (0, 0, -0.03)}, 5),
        ({"root": (0, 0, -0.5)}, 7),
        ({"point": (1.995, 0.5, -0.01)}, 8),
        ({"radius": 0.0}, 3),
    ],
)
def test_deep_missing_below_and_boundary_contacts_keep_obstacle_classification(
    changes, expected
):
    assert _reason(*_plane(), **changes)[0] == expected


def test_native_penetration_does_not_bound_valid_terrain_witness():
    assert _reason(*_plane(), point=(0, 0, 0), separation=-0.03, radius=0.0012)[0] == 1


@pytest.mark.parametrize("force,normal", [(2.0, (0, 0, 1)), (-2.0, (0, 0, -1))])
def test_signed_sphere_witness_reaches_real_warp_floor_query(force, normal):
    from npa.workflows.navigation.contact_witness import _terrain_witness

    witness, valid, _ = _terrain_witness(
        torch.tensor([[0, 0, -0.016]], dtype=torch.float32),
        torch.tensor([normal], dtype=torch.float32),
        torch.tensor([-0.016]),
        torch.tensor([force]),
        torch.tensor([True]),
    )
    assert valid.tolist() == [True]
    assert _reason(*_plane(), point=witness[0].tolist(), radius=0.0012)[0] == 1


def test_terrain_witness_still_rejects_wall_and_open_edge():
    points, faces = _plane()
    wall = points[:, [2, 1, 0]].copy()
    assert _reason(wall, faces, point=(0, 0, 0), radius=0.0012)[0] == 6
    assert _reason(points, faces, point=(2, 0, 0), radius=0.0012)[0] == 8


def test_underside_and_nearby_wall_are_not_floor():
    points, faces = _plane()
    assert _reason(points, faces[:, ::-1].copy())[0] == 6
    wall = np.array(
        [[0.01, -1, -1], [0.01, 1, -1], [0.01, 1, 1], [0.01, -1, 1]], dtype=np.float32
    )
    assert (
        _reason(
            np.concatenate([points, wall]),
            np.concatenate([faces, np.array([[4, 5, 6], [4, 6, 7]])]),
        )[0]
        == 6
    )


def test_nearby_disconnected_parallel_support_is_ambiguous():
    points, faces = _plane()
    above, top = _plane(0.01)
    assert (
        _reason(
            np.concatenate([points, above]),
            np.concatenate([faces, top + 4]),
            point=(0, 0, 0.005),
            separation=0.005,
        )[0]
        == 11
    )


def test_nonmanifold_and_degenerate_faces_are_rejected():
    points, faces = _plane()
    assert _reason(points, np.concatenate([faces, faces[:1]]))[0] == 8
    degenerate = np.array([[-0.1, 0, 0], [0, 0, 0], [0.1, 0, 0]], dtype=np.float32)
    assert (
        _reason(
            np.concatenate([points, degenerate]), np.concatenate([faces, [[4, 5, 6]]])
        )[0]
        == 6
    )


def test_query_capacity_excess_keeps_obstacle_classification():
    points = np.array(
        [[x, y, 0] for y in np.linspace(-1, 1, 21) for x in np.linspace(-1, 1, 21)],
        dtype=np.float32,
    )
    faces = []
    for y in range(20):
        for x in range(20):
            first = y * 21 + x
            faces.extend(
                [[first, first + 1, first + 22], [first, first + 22, first + 21]]
            )
    assert _reason(points, np.array(faces, dtype=np.int32), radius=0.5)[0] == 10


@wp.kernel
def _empty_connected(
    output: wp.array(dtype=wp.int32), neighbors: wp.array2d(dtype=wp.int32)
):
    output[0] = int(
        _connected(wp.uint64(0), _Faces(-1), 0, 0, neighbors, wp.vec3(0.0), 0.02)
    )
    output[1] = int(
        _connected(wp.uint64(0), _Faces(-1), 1, 0, neighbors, wp.vec3(0.0), 0.02)
    )


def test_empty_or_missing_seed_neighborhood_cannot_be_support():
    output = wp.zeros(2, dtype=wp.int32, device="cpu")
    wp.launch(
        _empty_connected,
        dim=1,
        inputs=[output, wp.zeros((1, 3), dtype=wp.int32, device="cpu")],
        device="cpu",
    )
    assert output.numpy().tolist() == [0, 0]


@pytest.mark.parametrize("rest", [0.0, -0.003, 0.001])
def test_resolved_native_rest_offset_can_be_zero_or_negative(rest):
    class View:
        max_shapes = 1
        count = 1

        def get_contact_offsets(self):
            return torch.tensor([[0.02]])

        def get_rest_offsets(self):
            return torch.tensor([[rest]])

    offset, actual = _resolved_margins(View())
    assert offset.tolist() == pytest.approx([0.02])
    assert actual.tolist() == pytest.approx([rest])


def test_individually_upward_faces_with_conflicting_normals_are_rejected():
    points = np.array(
        [[-2, -2, -1.5], [0, -2, 0], [0, 2, 0], [2, -2, -1.5], [2, 2, -1.5]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [1, 3, 4], [1, 4, 2]], dtype=np.int32)
    assert _reason(points, faces, point=(0, 0, -0.005))[0] == 9


def test_surfaces_connected_only_outside_radius_are_still_ambiguous():
    points, faces = _plane()
    upper, top = _plane(0.01)
    bridge = np.array([[0, 4, 5], [0, 5, 1]], dtype=np.int32)
    assert (
        _reason(
            np.concatenate([points, upper]),
            np.concatenate([faces, top + 4, bridge]),
            point=(0, 0, 0.005),
        )[0]
        == 11
    )


@pytest.mark.parametrize(
    "points,faces",
    [
        (np.zeros((0, 3)), np.array([[0, 1, 2]])),
        (np.zeros((3, 2)), np.array([[0, 1, 2]])),
        (np.zeros((3, 3)), np.array([[0, -1, 2]])),
        (np.zeros((3, 3)), np.array([[0, 1, 3]])),
        (np.zeros((3, 3)), np.array([[0.0, 1.0, 2.0]])),
        (np.zeros((3, 3)), np.empty((0, 3), dtype=int)),
        (np.full((3, 3), np.inf), np.array([[0, 1, 2]])),
    ],
)
def test_malformed_mesh_is_rejected_before_native_upload(points, faces):
    with pytest.raises(ValueError):
        SurfaceMesh(None, points, faces)


def test_multishape_native_margin_is_not_guessed():
    from types import SimpleNamespace

    offset, rest = _resolved_margins(SimpleNamespace(max_shapes=2, count=4))
    assert offset.tolist() == rest.tolist() == [0.0] * 4


def test_native_path_order_maps_offsets_and_rejects_inferred_order_disagreement():
    from types import SimpleNamespace
    from pxr import Usd, UsdPhysics
    from npa.workflows.navigation.contact_surface import _sensor_tables

    prefix = "/World/envs/env_0/Robot"
    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.ArticulationRootAPI.Apply(stage.DefinePrim(prefix, "Xform"))
    names = ["base", "LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT"]
    sensors = [f"{prefix}/{name}" for name in names]
    view = SimpleNamespace(
        prim_paths=sensors[1:][::-1],
        max_shapes=1,
        count=4,
        get_contact_offsets=lambda: torch.tensor([[0.01], [0.02], [0.03], [0.04]]),
        get_rest_offsets=lambda: torch.zeros((4, 1)),
    )
    env = SimpleNamespace(
        num_envs=1,
        sim=SimpleNamespace(stage=stage),
        scene={
            "robot": SimpleNamespace(root_view=SimpleNamespace(prim_paths=[prefix]))
        },
    )
    measurements = SimpleNamespace(
        body_count=5, body_names=names, view=SimpleNamespace(sensor_paths=sensors)
    )
    margins, _, roots, _ = _sensor_tables(env, measurements, view)
    assert margins == pytest.approx([0, 0.04, 0.03, 0.02, 0.01]) and roots == [0] * 5
    sensors[1], sensors[2] = sensors[2], sensors[1]
    with pytest.raises(ValueError, match="ordering"):
        _sensor_tables(env, measurements, view)


def test_only_actual_feet_are_queried_and_unavailable_margin_retains_reason_three():
    from types import SimpleNamespace
    from npa.workflows.navigation.contact_surface import FootSupport

    support = FootSupport.__new__(FootSupport)
    support.measurements = SimpleNamespace(view=SimpleNamespace(filter_count=1))
    support.feet = torch.tensor([False, False, True, True])
    support.spheres = support.feet.clone()
    support.radii = torch.tensor([0.0, 0.0, 0.02, 0.0])
    support.rests = torch.zeros(4)
    support.surface = _surface(*_plane())
    support._inputs = lambda data, indices, sensors: (
        torch.tensor([[0, 0, -0.01]] * len(indices)),
        support.radii[sensors],
        torch.full((len(indices),), -0.01),
        torch.tensor([[0, 0, 0.5]] * len(indices)),
    )
    mask, fields = support.recognize(
        (
            torch.ones((4, 1)),
            torch.zeros((4, 3)),
            torch.tensor([[0, 0, 1]] * 4, dtype=torch.float32),
        ),
        torch.arange(4),
        torch.arange(4),
        torch.ones(4, dtype=torch.bool),
        retain=True,
    )
    assert mask.tolist() == [False, False, True, False]
    assert fields["support_reason"].tolist() == [2, 2, 1, 3]
    assert fields["support_witness_valid"][:2].tolist() == [False, False]
    assert fields["terrain_witness_world_m"][:2].tolist() == [[-1] * 3] * 2


def test_source_binding_rejects_multiple_paths_quads_and_stale_world_cache():
    from types import SimpleNamespace
    from pxr import Usd, UsdGeom
    from npa.workflows.navigation.contact_surface_geometry import _world_mesh

    points, faces = _plane()
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/Scene")
    mesh.CreatePointsAttr(points.tolist())
    mesh.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
    mesh.CreateFaceVertexCountsAttr([3, 3])
    native = _surface(points, faces).mesh
    sensor = SimpleNamespace(
        cfg=SimpleNamespace(mesh_prim_paths=["/Scene"]),
        device="cpu",
        meshes={("/Scene", "cpu"): native},
    )
    assert _world_mesh(sensor, stage)[0] is native
    sensor.cfg.mesh_prim_paths.append("/Other")
    with pytest.raises(ValueError, match="exactly one"):
        _world_mesh(sensor, stage)
    sensor.cfg.mesh_prim_paths.pop()
    mesh.GetFaceVertexCountsAttr().Set([4, 2])
    with pytest.raises(ValueError, match="triangular"):
        _world_mesh(sensor, stage)
    mesh.GetFaceVertexCountsAttr().Set([3, 3])
    UsdGeom.Xformable(mesh).AddTranslateOp().Set((1, 0, 0))
    with pytest.raises(ValueError, match="differs"):
        _world_mesh(sensor, stage)


def test_query_selects_current_torch_cuda_stream_without_touching_physics(monkeypatch):
    from types import SimpleNamespace
    from npa.workflows.navigation.contact_surface import _launch

    observed = {}
    device = SimpleNamespace(type="cuda")

    class Points:
        def __len__(self):
            return 2

        def __init__(self):
            self.device = device

    points = Points()
    stream = object()
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda actual: stream if actual is device else None,
    )
    monkeypatch.setattr(wp, "stream_from_torch", lambda actual: ("warp", actual))
    monkeypatch.setattr(wp, "from_torch", lambda value, **kwargs: value)
    monkeypatch.setattr(wp, "launch", lambda kernel, **kwargs: observed.update(kwargs))
    _launch(
        SimpleNamespace(
            mesh=SimpleNamespace(id=3, device=device), normals=4, neighbors=5
        ),
        points,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
    )
    assert observed["stream"] == ("warp", stream) and observed["dim"] == 2
    assert observed["inputs"] == [3, 4, 5, points, 6, 7, 8, 9, 10, 11, 12]
