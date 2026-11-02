"""Real Open3D registration, reconstruction and RRD qualification, CPU-only.

The input scene is sampled from Open3D's own mesh primitives rather than from the
upstream `open3d.data` download, for one reason that matters more than realism:
the transform between the clouds is then *known*, so this check can assert that
RANSAC and ICP recovered the true pose instead of only asserting that they
returned a plausible-looking matrix. Every pipeline stage below is the real
upstream call; nothing is stubbed. The upstream captured scans are exercised by
`npa workbench open3d stage-demo` in the shipped reference workflow, which needs
network access this offline check deliberately does not.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from npa.workbench.open3d.artifacts import (
    read_journal,
    validate_mesh,
    validate_support,
    validate_pose_graph,
    validate_up_axis_inference,
    verify_rerun_recording,
)
from npa.workbench.open3d.runner import (
    run_multiway,
    run_reconstruct,
    run_register,
    run_visualize,
)
from npa.workbench.open3d.schemas import RegistrationManifest

RUN_ID = "open3d-functional"
VOXEL_SIZE = 0.05
#: Sampled points per cloud. Enough that a 0.05 voxel grid keeps real structure.
SAMPLE_POINTS = 60_000
#: Ground-truth tolerances. ICP at this voxel size should land far inside both;
#: they are loose enough to survive RANSAC's randomness, tight enough that a
#: wrong basin of attraction fails.
MAX_ROTATION_DEGREES = 1.0
MAX_TRANSLATION = VOXEL_SIZE
#: Surface the scan never supported. The crop targets zero; the allowance covers
#: the boundary triangles whose worst corner sits a hair past a voxel.
MAX_UNSUPPORTED_AREA_FRACTION = 0.02
#: The demo scans leave 0.3848 of all surface area more than three voxels from any
#: sample before cropping. Sample spacing cannot produce that — a watertight mesh
#: sampled at roughly voxel spacing leaves none at all there — so this floor is what
#: says the scene still contains the invented shell the crop is being measured on.
MIN_SHELL_AREA_BEYOND_3_VOXELS = 0.30
#: The other direction, and the reason the crop is defensible: the observations
#: have to still lie on what survives it.
MIN_COVERAGE_WITHIN_VOXEL = 0.97
MAX_SAMPLE_TO_SURFACE_RMSE = VOXEL_SIZE / 2.0


def _scene(o3d):
    """An asymmetric scene, so FPFH matching is not left rotationally ambiguous."""

    mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=0.6, depth=0.2)
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
    sphere.translate((0.9, 0.2, 0.4))
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.15, height=0.8)
    cylinder.translate((-0.7, -0.3, 0.4))
    return mesh + sphere + cylinder


def _pose(o3d, degrees: float, translation: tuple[float, float, float]):
    import numpy as np

    pose = np.eye(4)
    pose[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle(
        np.array([0.0, 0.0, np.deg2rad(degrees)])
    )
    pose[:3, 3] = translation
    return pose


def _write_clouds(o3d, root: Path) -> tuple[dict[str, str], dict[str, Any]]:
    """Publish three views of one scene at known, distinct poses."""

    import numpy as np

    # `sample_points_uniformly` draws from Open3D's global generator, which is the
    # only way to make the sampled scene repeatable across runs.
    o3d.utility.random.seed(0)
    sampled = _scene(o3d).sample_points_uniformly(number_of_points=SAMPLE_POINTS)
    poses = {
        "view_0": np.eye(4),
        "view_1": _pose(o3d, 8.0, (0.10, -0.05, 0.03)),
        "view_2": _pose(o3d, -6.0, (-0.08, 0.07, -0.04)),
    }
    paths: dict[str, str] = {}
    for name, pose in poses.items():
        cloud = o3d.geometry.PointCloud(sampled)
        cloud.transform(pose)
        path = root / f"{name}.ply"
        if not o3d.io.write_point_cloud(str(path), cloud):
            raise AssertionError(f"could not write {path}")
        paths[name] = str(path)
    return paths, poses


def _assert_support_crop_kept_the_observations(report) -> None:
    """The crop has to buy honesty without costing accuracy.

    Discarding surface is only an improvement if the observations still lie on
    what is left. Both directions are asserted here so a future change that
    trims a prettier mesh by throwing away measured structure fails this eval
    instead of shipping.

    Tolerances rather than equalities, deliberately: Open3D reduces over its
    correspondences in thread-scheduling order, so `registration_icp`,
    `get_information_matrix_from_point_clouds`, `compute_fpfh_feature` and
    `global_optimization` return results that depend on how many threads the
    container was given. Integer counts repeat bit-for-bit; RANSAC correspondence
    choices do not, despite the seeded generator -- sixteen seeded repeats from
    fixed clouds and fixed features gave 2 distinct results on an ambiguous
    construction and 4 on the upstream demo scan
    (`docs/evidence/open3d/register-seed-determinism.json`), and the shipped
    point-to-plane ICP gave sixteen. Floating-point sums do not either. On
    the upstream demo scans two builds of this image agreed to 4e-06 relative on
    surface area and about 1e-08 on pose, which is the contract a consumer can
    actually rely on.

    Do not tighten these to byte equality: pinning the thread count will not get
    you there. Measured on the Armadillo 300k-sample construction rather than on
    this scene, at a fixed count inside one process, surface area takes four
    distinct values in four repeats, and `set_max_threads(1)` does not settle it
    -- it came out worse in some runs and better in others. There the points and
    the estimated normals entering reconstruction are byte-identical across the
    repeats, both hashed, which leaves the Poisson reconstruction and the
    density crop after it: neither of the two knobs named above reaches them.
    See `docs/evidence/open3d/declared-reproducibility-causes.json`. And do not
    loosen them to hide a real regression.
    """

    before = report["support_before_crop"]["unsupported_area_fraction"]
    after = report["support"]["unsupported_area_fraction"]
    coverage = report["coverage"]
    assert after <= MAX_UNSUPPORTED_AREA_FRACTION, (
        f"{after:.4f} of the surface area still has no sample within a voxel"
    )
    assert before > after, (
        "this scene is meant to exercise the crop, but Poisson returned nothing "
        f"unsupported to remove (before {before:.4f}, after {after:.4f})"
    )
    assert coverage["fraction_within_voxel"] >= MIN_COVERAGE_WITHIN_VOXEL, (
        f"only {coverage['fraction_within_voxel']:.4f} of samples lie within a "
        "voxel of the cropped surface; the crop took observed structure"
    )
    # The crop removing distant surface on this scene is the reason the smoke can assert
    # `before > after` at all. On a complete capture the same arithmetic holds while
    # the crop destroys correct geometry, so the assertion above is only meaningful
    # alongside this one.
    #
    # This asserts where the removed area sat, not that it was invented -- uneven coverage of
    # correct geometry reads the same way, which is measured in the unit tests. It is safe to
    # demand here because these scans are a known partial capture reading 0.985 of their
    # unsupported area past three voxels, the far pole rather than a marginal call.
    justification = report["crop_justification"]
    assert justification["removed_surface_reads_as"] == "far from any observation", (
        "these demo scans are a partial capture, so the unsupported area should sit "
        "well past three voxels; a near-threshold reading here means the scene or the "
        f"voxel changed ({justification['unsupported_area_share_beyond_3_voxels']:.4f} "
        "of unsupported area past three voxels)"
    )
    assert (
        report["support_before_crop"]["unsupported_area_beyond_3_voxels"]
        >= MIN_SHELL_AREA_BEYOND_3_VOXELS
    ), (
        "the shell this crop exists to remove is not present at the distance that "
        "distinguishes it from sample spacing"
    )
    assert coverage["sample_to_surface_rmse"] <= MAX_SAMPLE_TO_SURFACE_RMSE, (
        f"sample-to-surface RMSE {coverage['sample_to_surface_rmse']:.5f} exceeds "
        f"{MAX_SAMPLE_TO_SURFACE_RMSE}"
    )


def _assert_recovered(o3d, measured, source_pose, target_pose) -> None:
    """The registration must invert the pose difference we actually applied."""

    import numpy as np

    truth = target_pose @ np.linalg.inv(source_pose)
    residual = np.linalg.inv(truth) @ np.asarray(measured, dtype=float)
    angle = np.degrees(
        np.arccos(np.clip((np.trace(residual[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
    )
    offset = float(np.linalg.norm(residual[:3, 3]))
    assert angle <= MAX_ROTATION_DEGREES, f"rotation error {angle:.3f} deg"
    assert offset <= MAX_TRANSLATION, f"translation error {offset:.4f}"


def _unit_cube(o3d, per_face: int = 40):
    """A closed cube whose every vertex lies exactly on the analytic cube it samples."""

    import numpy as np

    verts: list = []
    tris: list = []

    def face(origin, u, v) -> None:
        base = len(verts)
        for i in range(per_face + 1):
            for j in range(per_face + 1):
                verts.append(
                    np.asarray(origin, dtype=float)
                    + u * (i / per_face)
                    + v * (j / per_face)
                )
        for i in range(per_face):
            for j in range(per_face):
                a = base + i * (per_face + 1) + j
                across = a + per_face + 1
                tris.extend([[a, across, a + 1], [a + 1, across, across + 1]])

    x, y, z = np.eye(3)
    for origin, u, v in _CUBE_FACES:
        face(origin, [x, y, z][u], [x, y, z][v])
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(verts)),
        o3d.utility.Vector3iVector(np.asarray(tris)),
    )
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    return mesh


#: (origin, u-axis, v-axis) for the six faces of the unit cube.
_CUBE_FACES = (
    ([0, 0, 0], 0, 1),
    ([0, 0, 1], 0, 1),
    ([0, 0, 0], 0, 2),
    ([0, 1, 0], 0, 2),
    ([0, 0, 0], 1, 2),
    ([1, 0, 0], 1, 2),
)


def _assert_the_up_axis_is_decided_by_the_scan(o3d) -> None:
    """Two clouds where the answer is known, against the real `segment_plane`.

    The unit tests cover the aggregation and the validator with no Open3D present.
    What they cannot cover is whether repeating a real RANSAC fit actually separates
    a scan that determines an up axis from one that does not, so both directions run
    here on constructed geometry whose answer is not in question.
    """

    import numpy as np

    from npa.workbench.open3d.artifacts import validate_up_axis_inference
    from npa.workbench.open3d.runner import _up_axis

    def room(floor_points: int, wall_points: int):
        rng = np.random.default_rng(7)
        floor = np.column_stack(
            [
                rng.uniform(0, 4.0, floor_points),
                rng.uniform(0, 4.0, floor_points),
                np.zeros(floor_points),
            ]
        )
        wall = np.column_stack(
            [
                np.zeros(wall_points),
                rng.uniform(0, 4.0, wall_points),
                rng.uniform(0, 4.0, wall_points),
            ]
        )
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.vstack([floor, wall]))
        return cloud

    determined = _up_axis(o3d, room(8100, 810), VOXEL_SIZE)
    validate_up_axis_inference({"up_axis_inference": determined})
    assert determined["source"] == "largest-plane-normal-snapped-to-axis", determined
    assert determined["up"] == [0.0, 0.0, 1.0], determined
    assert determined["agreement"]["distinct_axes"] == 1, determined["agreement"]
    runner = determined["agreement"]["runner_up_plane"]
    assert runner["measured"] and runner["inliers_relative_to_the_winner"] < 0.5, runner

    def two_walls(first_points: int, second_points: int):
        """Planes at x=0 and y=0, so the candidate axes are x and y -- never z.

        The floor-and-wall room above cannot test the fallback properly: its
        majority branch *is* the declared default, so publishing the default and
        publishing the winning branch look identical. Here the default is neither
        candidate, which makes the difference observable.
        """

        rng = np.random.default_rng(7)
        first = np.column_stack(
            [
                np.zeros(first_points),
                rng.uniform(0, 4.0, first_points),
                rng.uniform(0, 4.0, first_points),
            ]
        )
        second = np.column_stack(
            [
                rng.uniform(0, 4.0, second_points),
                np.zeros(second_points),
                rng.uniform(0, 4.0, second_points),
            ]
        )
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.vstack([first, second]))
        return cloud

    # Equal point counts on two perpendicular planes. The check is allowed to miss --
    # its measured rate is 25 to 27 of 30 runs on the floor-and-wall room -- so this asserts
    # the two outcomes it may legitimately produce, and that neither is a named axis
    # contradicted by its own repeats.
    ambiguous = _up_axis(o3d, two_walls(8100, 8100), VOXEL_SIZE)
    validate_up_axis_inference({"up_axis_inference": ambiguous})
    assert ambiguous["source"] in {
        "default-ambiguous-plane",
        "largest-plane-normal-snapped-to-axis",
    }, ambiguous
    candidates = [one["up"] for one in ambiguous["agreement"]["candidates"]]
    assert [0.0, 0.0, 1.0] not in candidates, (
        "this construction is meant to make the declared default unreachable as a "
        f"measured branch, and a fit returned it: {candidates}"
    )
    if ambiguous["source"] == "default-ambiguous-plane":
        assert ambiguous["agreement"]["distinct_axes"] >= 2, ambiguous["agreement"]
        # The assertion the whole fallback exists for: when the repeats disagree the
        # published vector is the declared default, not whichever branch won.
        assert ambiguous["up"] == [0.0, 0.0, 1.0], ambiguous
    competition = ambiguous["agreement"]["runner_up_plane"]
    assert competition["inliers_relative_to_the_winner"] > 0.9, competition
    assert (
        competition["inliers_relative_to_the_winner"]
        > runner["inliers_relative_to_the_winner"]
    ), "the ambiguous room's runner-up is not closer than the determined room's"


def _assert_uneven_coverage_is_not_called_fabrication(o3d) -> None:
    """A correct surface, unevenly observed, must not be reported as invented.

    The cube here *is* its own ground truth: every vertex lies on the analytic cube, and
    every observation is drawn on that same cube, so there is nothing anywhere for the
    reading to be right about if it says something was invented. Sampling five faces densely
    and the sixth sparsely still pushes a large share of the unsupported area past three
    voxels, because that distance is measured from the nearest observation and the sixth face
    has few. The reading may report where the area sat; it may not call it fabrication.

    This is the control for the over-call direction, which a sweep of even sampling cannot
    reach. It runs with real Open3D rather than a nearest-neighbour stand-in.
    """

    import numpy as np

    from npa.workbench.open3d.runner import _crop_justification, _support

    rng = np.random.default_rng(7)
    mesh = _unit_cube(o3d)
    axes = np.eye(3)

    def observe(dense: int, sparse: int):
        points = []
        for index, (origin, u, v) in enumerate(_CUBE_FACES):
            count = sparse if index == len(_CUBE_FACES) - 1 else dense
            st = rng.random((count, 2))
            points.append(
                np.asarray(origin, dtype=float)
                + np.outer(st[:, 0], axes[u])
                + np.outer(st[:, 1], axes[v])
            )
        return np.vstack(points)

    def read(points):
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        tree = o3d.geometry.KDTreeFlann(cloud)
        spacing = float(
            np.median([np.sqrt(tree.search_knn_vector_3d(p, 2)[2][1]) for p in points])
        )
        # The voxel rule the module recommends: twice the median sample spacing.
        return _crop_justification(_support(o3d, mesh, cloud, 2.0 * spacing), 0, mesh)

    # Nothing is invented in either case, so neither note may assert that anything was.
    # Both notes do mention invented surface while describing what they cannot see, which
    # is a caveat rather than a claim; what is forbidden is the claim.
    asserts_fabrication = ("removed invented surface", "is invented", "was invented")
    for label, points in (
        ("even", observe(4000, 4000)),
        ("uneven", observe(4000, 40)),
    ):
        note = read(points)["note"]
        for phrase in asserts_fabrication:
            assert phrase not in note, (
                f"{label} coverage of a cube that is exactly its own ground truth was "
                f"reported as invented surface ({phrase!r}): {note!r}"
            )

    # And the far reading really is reachable with zero error, which is why it cannot assert.
    uneven = read(observe(4000, 40))
    assert uneven["removed_surface_reads_as"] == "far from any observation", (
        "uneven coverage of a correct cube no longer reaches the far band "
        f"({uneven['unsupported_area_share_beyond_3_voxels']:.4f} share); the "
        "over-call control has stopped exercising the case it exists for"
    )
    assert "separate question this cannot answer" in uneven["note"]


def main() -> None:
    import open3d as o3d

    with tempfile.TemporaryDirectory(prefix="npa-open3d-functional-") as directory:
        root = Path(directory)
        paths, poses = _write_clouds(o3d, root)
        manifest = RegistrationManifest(
            fragments=[
                {
                    "id": name,
                    "uri": paths[name],
                    "sha256": "0" * 64,
                    "bytes": Path(paths[name]).stat().st_size,
                }
                for name in sorted(paths)
            ],
            voxel_size=VOXEL_SIZE,
        ).model_dump(mode="json")
        payload = {"manifest": manifest, "fragments": paths}

        pairs = root / "register"
        pairs.mkdir()
        run_register(payload, pairs, RUN_ID)
        rows = read_journal(pairs / "pairs.jsonl")
        assert len(rows) == 2, rows
        for row in rows:
            assert row["icp"]["fitness"] > 0.9, row["icp"]
            assert row["icp"]["inlier_rmse"] < row["global_registration"]["inlier_rmse"]
            _assert_recovered(
                o3d,
                row["transformation"],
                poses[row["source_id"]],
                poses[row["target_id"]],
            )

        graph_dir = root / "multiway"
        graph_dir.mkdir()
        report = run_multiway(payload, graph_dir, RUN_ID)
        validate_pose_graph(report["pose_graph"], fragment_count=3)

        surface = root / "reconstruct"
        surface.mkdir()
        mesh_report = run_reconstruct(
            {
                "request": {
                    "input_path": str(graph_dir),
                    "output_path": str(surface),
                    "run_id": RUN_ID,
                    "poisson_depth": 8,
                    "density_quantile": 0.02,
                    "support_distance_factor": 1.0,
                },
                "fused_path": str(graph_dir / "fused.ply"),
                "voxel_size": VOXEL_SIZE,
            },
            surface,
            RUN_ID,
        )
        validate_mesh(mesh_report["mesh"])
        validate_mesh(mesh_report["mesh_uncropped"])
        validate_support(mesh_report, voxel_size=VOXEL_SIZE)
        _assert_support_crop_kept_the_observations(mesh_report)

        recording = root / "visualize"
        recording.mkdir()
        view = run_visualize(
            {
                "pose_graph": report["pose_graph"],
                "fragments": paths,
                "fused_path": str(graph_dir / "fused.ply"),
                "mesh_path": str(surface / "mesh.ply"),
                "uncropped_mesh_path": str(surface / "mesh_uncropped.ply"),
                "voxel_size": VOXEL_SIZE,
            },
            recording,
            RUN_ID,
        )
        verify_rerun_recording(recording / "point_cloud.rrd")

        # The up-axis block the real pipeline wrote for its own fused cloud, held to
        # the same validator the runtime applies. The constructed rooms below test
        # both directions; this tests the scene the rest of this smoke built.
        validate_up_axis_inference(view)
        inference = view["up_axis_inference"]
        assert inference["source"], inference
        if "agreement" in inference:
            assert inference["agreement"]["repeats"] >= 2, inference["agreement"]

        _assert_uneven_coverage_is_not_called_fabrication(o3d)
        _assert_the_up_axis_is_decided_by_the_scan(o3d)

        print(
            "Open3D registration recovered the ground-truth pose "
            f"(ICP fitness {min(row['icp']['fitness'] for row in rows):.4f}+), "
            f"optimized a {len(report['pose_graph']['nodes'])}-node pose graph, "
            f"reconstructed {mesh_report['mesh']['triangle_count']} triangles, "
            "cropped surface the scan did not support "
            f"({mesh_report['support_before_crop']['unsupported_area_fraction']:.4f} "
            f"-> {mesh_report['support']['unsupported_area_fraction']:.4f} of area, "
            f"{mesh_report['support_before_crop']['unsupported_area_beyond_3_voxels']:.4f} "
            f"of it beyond three voxels so it reads as "
            f"{mesh_report['crop_justification']['removed_surface_reads_as']}) "
            "and wrote a verified RRD"
        )


if __name__ == "__main__":
    main()
