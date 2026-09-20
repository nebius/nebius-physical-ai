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
    container was given. Integer counts and RANSAC correspondence choices repeat
    bit-for-bit because the generator is seeded; floating-point sums do not. On
    the upstream demo scans two builds of this image agreed to 4e-06 relative on
    surface area and about 1e-08 on pose, which is the contract a consumer can
    actually rely on. Do not tighten these to byte equality without pinning the
    thread count, and do not loosen them to hide a real regression.
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
    # The crop being justified on this scene is the reason the smoke can assert
    # `before > after` at all. On a complete capture the same arithmetic holds while
    # the crop destroys correct geometry, so the assertion above is only meaningful
    # alongside this one: these scans really do carry an extrapolated shell.
    # Demanding a verdict is only legitimate because this scene is nowhere near the
    # undecided band: it reads 0.985 of unsupported area past three voxels against a band
    # topping out at 0.2. A scene that landed in the band would correctly refuse to answer,
    # and this assertion would be wrong to make of it.
    justification = report["crop_justification"]
    assert justification["removed_surface_reads_as"] == "extrapolated shell", (
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
        run_visualize(
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
