"""Measure where the support metric and the coverage guard disagree, across scenes.

The complete-capture control established that coverage cannot see correct surface being
discarded, and left that as prose. This measures it: three scenes, the tool's own
`_support` and `_crop_justification`, and the coverage movement the crop produced in
each. If coverage separated the cases there would be nothing to add. It does not.

The scenes are chosen to span the thing being tested, not to be three of the same:

- `demo_scans`, the fused Redwood living-room fragments, a genuinely partial capture
  whose Poisson shell is invented surface.
- `eagle`, a real scan of a solid object. Dense relative to its own features, but with
  real holes, so it sits between the other two rather than acting as a clean control.
- `armadillo_complete`, a watertight mesh sampled uniformly at roughly voxel spacing.
  There is no missing observation anywhere, so every square metre of unsupported area it
  reports is discretization by construction, and any crop is removing correct geometry.

Run inside the Open3D image:
    docker run --rm -v "$EVIDENCE_DIR:/data" --entrypoint python "$IMAGE" \
        /data/harness/band_profile_across_scenes.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d
from npa.workbench.open3d.runner import (
    _coverage,
    _crop_justification,
    _support,
)

BUNDLE = Path("/data/final-bundle")
OUT = Path("/data/band-profile-across-scenes.json")

#: Quantile of Poisson density discarded before anything else, matching the tool.
DENSITY_QUANTILE = 0.02


class _Mesh:
    """Vertex count is all `_crop_justification` reads off the kept mesh."""

    def __init__(self, count: int) -> None:
        self.vertices = range(count)


def spacing(cloud) -> float:
    """The voxel the tool would be run at: twice the median nearest-neighbour gap."""

    return float(np.median(np.asarray(cloud.compute_nearest_neighbor_distance())) * 2.0)


def poisson(cloud, voxel: float):
    down = cloud.voxel_down_sample(voxel)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30)
    )
    down.orient_normals_consistent_tangent_plane(30)
    mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        down, depth=9
    )
    density = np.asarray(density)
    mesh.remove_vertices_by_mask(density < np.quantile(density, DENSITY_QUANTILE))
    mesh.compute_vertex_normals()
    return mesh, down


def crop(mesh, cloud, voxel: float, factor: float):
    """Remove vertices further than `factor` voxels from any sample, as the tool does."""

    kept = o3d.geometry.TriangleMesh(mesh)
    tree = o3d.geometry.KDTreeFlann(cloud)
    limit = voxel * factor
    distances = np.array(
        [
            np.sqrt(tree.search_knn_vector_3d(point, 1)[2][0])
            for point in np.asarray(kept.vertices)
        ]
    )
    removed = int((distances > limit).sum())
    kept.remove_vertices_by_mask(distances > limit)
    kept.compute_vertex_normals()
    return kept, removed


def truth_distance(mesh, reference) -> dict[str, float]:
    """How far the reconstructed vertices sit from a known-correct surface.

    Only meaningful where a ground-truth surface exists, which is why it is reported for
    the synthetic scene and omitted for the two real scans: there is nothing to compare
    a real scan against except the samples, and that is the measurement under question.
    """

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(reference))
    distance = scene.compute_distance(
        o3d.core.Tensor(np.asarray(mesh.vertices).astype(np.float32))
    ).numpy()
    return {
        "median": float(np.median(distance)),
        "p99": float(np.percentile(distance, 99)),
        "max": float(distance.max()),
    }


def scene_demo():
    cloud = o3d.io.read_point_cloud(str(BUNDLE / "multiway/fused.ply"))
    mesh = o3d.io.read_triangle_mesh(str(BUNDLE / "mesh-baseline/mesh.ply"))
    return cloud, mesh, cloud, 0.05, None


def scene_eagle():
    cloud = o3d.io.read_point_cloud(o3d.data.EaglePointCloud().path)
    voxel = spacing(cloud)
    mesh, down = poisson(cloud, voxel)
    return cloud, mesh, down, voxel, None


def scene_armadillo():
    reference = o3d.io.read_triangle_mesh(o3d.data.ArmadilloMesh().path)
    reference.compute_vertex_normals()
    cloud = reference.sample_points_uniformly(number_of_points=300000)
    voxel = spacing(cloud)
    mesh, down = poisson(cloud, voxel)
    return cloud, mesh, down, voxel, reference


SCENES = {
    "demo_scans": scene_demo,
    "eagle": scene_eagle,
    "armadillo_complete": scene_armadillo,
}


def main() -> int:
    report: dict = {"scenes": {}}
    for name, build in SCENES.items():
        _cloud, mesh, samples, voxel, reference = build()
        before = _support(o3d, mesh, samples, voxel)
        coverage_before = _coverage(o3d, mesh, samples, voxel)
        kept, removed = crop(mesh, samples, voxel, 1.0)
        after = _support(o3d, kept, samples, voxel)
        coverage_after = _coverage(o3d, kept, samples, voxel)
        justification = _crop_justification(before, removed, _Mesh(len(kept.vertices)))

        row = {
            "voxel_size": voxel,
            "samples": len(samples.points),
            "unsupported_area_fraction": before["unsupported_area_fraction"],
            "unsupported_area_beyond_1_5_voxels": before[
                "unsupported_area_beyond_1_5_voxels"
            ],
            "unsupported_area_beyond_3_voxels": before[
                "unsupported_area_beyond_3_voxels"
            ],
            "max_vertex_distance_in_voxels": before["max_vertex_distance_to_sample"]
            / voxel,
            "crop_justification": justification,
            "after_crop_unsupported_area_fraction": after["unsupported_area_fraction"],
            "coverage_before": coverage_before["fraction_within_voxel"],
            "coverage_after": coverage_after["fraction_within_voxel"],
            "coverage_movement": abs(
                coverage_after["fraction_within_voxel"]
                - coverage_before["fraction_within_voxel"]
            ),
        }
        if reference is not None:
            row["vertex_distance_to_ground_truth_in_voxels"] = {
                key: value / voxel
                for key, value in truth_distance(mesh, reference).items()
            }
        report["scenes"][name] = row

    demo = report["scenes"]["demo_scans"]
    complete = report["scenes"]["armadillo_complete"]
    report["finding"] = {
        "coverage_cannot_separate_the_cases": (
            "Cropping moved coverage by "
            f"{demo['coverage_movement']:.2e} on the invented shell and "
            f"{complete['coverage_movement']:.2e} on the complete capture. Both are "
            "negligible and the smaller belongs to the case where correct geometry was "
            "destroyed, so coverage is not evidence either way."
        ),
        "headline_fraction_cannot_separate_the_cases": (
            f"{demo['unsupported_area_fraction']:.4f} against "
            f"{complete['unsupported_area_fraction']:.4f}: the same order of magnitude, "
            "so no threshold on this number alone distinguishes an extrapolated shell "
            "from discretization."
        ),
        "share_beyond_three_voxels_does": (
            f"{demo['crop_justification']['unsupported_area_share_beyond_3_voxels']:.4f} "
            "against "
            f"{complete['crop_justification']['unsupported_area_share_beyond_3_voxels']:.4f}."
        ),
        "readings": {
            name: row["crop_justification"]["removed_surface_reads_as"]
            for name, row in report["scenes"].items()
        },
    }

    OUT.write_text(json.dumps(report, indent=2, sort_keys=True))
    header = f"{'scene':22} {'unsup':>8} {'>1.5v':>8} {'>3v':>8} {'share':>7} {'maxv':>6} {'cov move':>9}  reading"
    print(header)
    for name, row in report["scenes"].items():
        print(
            f"{name:22} {row['unsupported_area_fraction']:8.4f} "
            f"{row['unsupported_area_beyond_1_5_voxels']:8.4f} "
            f"{row['unsupported_area_beyond_3_voxels']:8.4f} "
            f"{row['crop_justification']['unsupported_area_share_beyond_3_voxels']:7.4f} "
            f"{row['max_vertex_distance_in_voxels']:6.1f} "
            f"{row['coverage_movement']:9.2e}  "
            f"{row['crop_justification']['removed_surface_reads_as']}"
        )
    if "vertex_distance_to_ground_truth_in_voxels" in complete:
        truth = complete["vertex_distance_to_ground_truth_in_voxels"]
        print(
            f"\narmadillo vertices vs ground truth: median {truth['median']:.3f} voxels, "
            f"p99 {truth['p99']:.3f}, max {truth['max']:.3f}"
        )
    print(f"\n{report['finding']['coverage_cannot_separate_the_cases']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
