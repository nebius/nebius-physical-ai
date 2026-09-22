"""Find where the distance bands stop telling the two cases apart.

`crop_justification` reads "extrapolated shell" when enough unsupported area sits past
three voxels. That boundary was chosen because a correct surface sampled at roughly half
the voxel had nothing there. It is a reporting choice with no derivation, so its domain of
validity is unknown, and an advisory field whose failure mode is undocumented is worse
than no field.

This measures the domain directly. Sample spacing is swept against a *fixed* voxel, so the
ratio between them varies — the tool's own default ties the voxel to the spacing, which
holds the ratio constant and hides exactly this effect.

The sweep runs on a watertight mesh, so at every density there is a ground-truth surface
to compare against. That is what makes the result a measurement rather than a guess: for
each density it reports both what the reading says and how much surface is *actually*
fabricated, defined as reconstructed area further than one voxel from the true surface.
Where those two disagree is the false-positive zone, and its edge is the answer.

Run inside the Open3D image:
    docker run --rm -v "$EVIDENCE_DIR:/data" --entrypoint python "$IMAGE" \
        /data/harness/band_validity_sweep.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d
from npa.workbench.open3d.runner import (
    FABRICATION_AREA_SHARE,
    _crop_justification,
    _support,
)

OUT = Path("/data/band-validity-sweep.json")

#: Sample counts, densest first. The densest fixes the voxel; the rest are the same
#: surface observed more sparsely, so spacing rises against a voxel that does not move.
COUNTS = (300000, 150000, 80000, 40000, 20000, 10000, 5000, 2500)
DENSITY_QUANTILE = 0.02


class _Mesh:
    def __init__(self, count: int) -> None:
        self.vertices = range(count)


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


def fabricated_area_fraction(mesh, reference, voxel: float) -> dict[str, float]:
    """Area further than one voxel from the true surface: real fabrication, by area.

    Area-weighted rather than a vertex count, to match how the support metric weighs its
    own unsupported figure. Otherwise the two numbers being compared would not be
    measuring over the same thing.
    """

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(reference))
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    distance = scene.compute_distance(
        o3d.core.Tensor(vertices.astype(np.float32))
    ).numpy()
    a, b, c = (
        vertices[triangles[:, 0]],
        vertices[triangles[:, 1]],
        vertices[triangles[:, 2]],
    )
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    per_triangle = distance[triangles].max(axis=1)
    total = float(areas.sum())
    return {
        "fabricated_area_fraction": float(areas[per_triangle > voxel].sum() / total)
        if total > 0
        else 0.0,
        "max_vertex_distance_to_truth_in_voxels": float(distance.max() / voxel),
    }


def main() -> int:
    reference = o3d.io.read_triangle_mesh(o3d.data.ArmadilloMesh().path)
    reference.compute_vertex_normals()

    densest = reference.sample_points_uniformly(number_of_points=COUNTS[0])
    voxel = float(
        np.median(np.asarray(densest.compute_nearest_neighbor_distance())) * 2.0
    )

    rows = []
    for count in COUNTS:
        cloud = (
            densest
            if count == COUNTS[0]
            else reference.sample_points_uniformly(number_of_points=count)
        )
        spacing = float(
            np.median(np.asarray(cloud.compute_nearest_neighbor_distance()))
        )
        mesh, samples = poisson(cloud, voxel)
        support = _support(o3d, mesh, samples, voxel)
        justification = _crop_justification(support, 0, _Mesh(len(mesh.vertices)))
        truth = fabricated_area_fraction(mesh, reference, voxel)
        share = justification["unsupported_area_share_beyond_3_voxels"]
        reads_as_shell = (
            justification["removed_surface_reads_as"] == "extrapolated shell"
        )
        rows.append(
            {
                "samples": count,
                "spacing_over_voxel": spacing / voxel,
                "unsupported_area_fraction": support["unsupported_area_fraction"],
                "unsupported_area_beyond_1_5_voxels": support[
                    "unsupported_area_beyond_1_5_voxels"
                ],
                "unsupported_area_beyond_3_voxels": support[
                    "unsupported_area_beyond_3_voxels"
                ],
                "share_beyond_3_voxels": share,
                "reads_as_extrapolated_shell": reads_as_shell,
                **truth,
                # The reading claims a shell while the ground truth says there is barely
                # any fabricated area. That is the failure this sweep exists to locate.
                "false_positive": reads_as_shell
                and truth["fabricated_area_fraction"] < 0.05,
            }
        )

    honest = [r for r in rows if not r["false_positive"]]
    first_false = next((r for r in rows if r["false_positive"]), None)
    report = {
        "voxel_size": voxel,
        "held_fixed": (
            "The voxel is derived once from the densest sampling and then held fixed, so "
            "spacing rises against it. The tool's default ties the voxel to the spacing, "
            "which pins this ratio near 0.5 and hides the effect entirely."
        ),
        "reading_boundary": FABRICATION_AREA_SHARE,
        "fabrication_definition": (
            "Reconstructed area whose vertices lie further than one voxel from the "
            "watertight reference surface, area-weighted to match how the support metric "
            "weighs its own figure."
        ),
        "rows": rows,
        "valid_up_to_spacing_over_voxel": (
            max(r["spacing_over_voxel"] for r in honest) if honest else None
        ),
        "first_false_positive_at_spacing_over_voxel": (
            first_false["spacing_over_voxel"] if first_false else None
        ),
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True))

    print(f"voxel {voxel:.5f} held fixed; reading boundary {FABRICATION_AREA_SHARE}")
    print(
        f"{'samples':>8} {'space/vox':>10} {'unsup':>8} {'>3v':>8} {'share':>8} "
        f"{'fabricated':>11} {'maxtruth':>9}  reading"
    )
    for row in rows:
        flag = "  <== FALSE POSITIVE" if row["false_positive"] else ""
        print(
            f"{row['samples']:8d} {row['spacing_over_voxel']:10.3f} "
            f"{row['unsupported_area_fraction']:8.4f} "
            f"{row['unsupported_area_beyond_3_voxels']:8.4f} "
            f"{row['share_beyond_3_voxels']:8.4f} "
            f"{row['fabricated_area_fraction']:11.4f} "
            f"{row['max_vertex_distance_to_truth_in_voxels']:9.2f}  "
            f"{'shell' if row['reads_as_extrapolated_shell'] else 'near-threshold'}{flag}"
        )
    if first_false:
        print(
            f"\nThe reading first calls a correct surface a shell at spacing "
            f"{first_false['spacing_over_voxel']:.3f} of a voxel, where only "
            f"{first_false['fabricated_area_fraction']:.4f} of area is actually "
            "fabricated."
        )
    else:
        print("\nNo false positive anywhere in this sweep.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
