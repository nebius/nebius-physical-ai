"""Find the smallest occlusion whose invented surface the bands still catch.

The validity sweep covered one direction only: a correct surface misread as a shell. It
found none. The reverse error is the one that matters more — a genuine shell reading as
near-threshold, which would tell an operator to loosen a crop that was doing its job.

Uniform thinning cannot test that. It degrades the whole surface at once, so there is no
occluded region to invent across. This removes samples inside a sphere instead, which is
what an occlusion actually looks like, and sweeps the sphere's radius.

Ground truth needs care here, and the definition is deliberately strict against the tool.
Surface Poisson interpolates across a hole is *not* automatically fabricated: on a smooth
convex region the closure can follow the true surface closely, and calling that invented
would inflate the result in the tool's favour. So fabrication is measured the same way as
in the validity sweep — reconstructed area further than one voxel from the watertight
reference — and only surface that genuinely departs from truth counts.

Run inside the Open3D image:
    docker run --rm -v "$EVIDENCE_DIR:/data" --entrypoint python "$IMAGE" \
        /data/harness/band_false_negative_sweep.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d

from npa.workbench.open3d.runner import _crop_justification, _support

OUT = Path("/data/band-false-negative-sweep.json")

SAMPLES = 300000
DENSITY_QUANTILE = 0.02
#: Occlusion radii as a fraction of the object's bounding-box diagonal. A first pass used
#: small multiples of the voxel and was vacuous: nine voxels removed 208 of 300000 samples,
#: roughly a thousandth of the surface, and Poisson reproduced it to within 0.78 voxels of
#: truth. Nothing was fabricated, so nothing could be missed, and "no false negative" would
#: have been a statement about the sweep rather than about the bands. Occlusion has to be
#: sized against the object.
RADII_AS_DIAGONAL_FRACTION = (0.05, 0.10, 0.20, 0.30, 0.45)


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


def fabricated(mesh, reference, voxel: float) -> dict[str, float]:
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
    full = reference.sample_points_uniformly(number_of_points=SAMPLES)
    points = np.asarray(full.points)
    voxel = float(np.median(np.asarray(full.compute_nearest_neighbor_distance())) * 2.0)

    # Occlude around a point on the surface itself, so the removed region is a real patch
    # of the object rather than empty space. The most protruding vertex along the axis of
    # greatest extent is a deterministic choice that lands on a feature rather than a
    # flat panel, which is where a bridge deviates from truth most.
    extent = points.max(axis=0) - points.min(axis=0)
    axis = int(np.argmax(extent))
    centre = points[int(np.argmax(points[:, axis]))]

    diagonal = float(np.linalg.norm(extent))
    rows = []
    for fraction in RADII_AS_DIAGONAL_FRACTION:
        radius = diagonal * fraction
        keep = np.linalg.norm(points - centre, axis=1) > radius
        occluded = o3d.geometry.PointCloud()
        occluded.points = o3d.utility.Vector3dVector(points[keep])
        mesh, samples = poisson(occluded, voxel)
        support = _support(o3d, mesh, samples, voxel)
        reading = _crop_justification(support, 0, _Mesh(len(mesh.vertices)))
        truth = fabricated(mesh, reference, voxel)
        share = reading["unsupported_area_share_beyond_3_voxels"]
        reads_shell = reading["removed_surface_reads_as"] == "extrapolated shell"
        rows.append(
            {
                "occlusion_radius_as_diagonal_fraction": fraction,
                "occlusion_radius_in_voxels": radius / voxel,
                "samples_removed": int((~keep).sum()),
                "unsupported_area_fraction": support["unsupported_area_fraction"],
                "unsupported_area_beyond_3_voxels": support[
                    "unsupported_area_beyond_3_voxels"
                ],
                "share_beyond_3_voxels": share,
                "reads_as_extrapolated_shell": reads_shell,
                **truth,
                # A shell that reads near-threshold while real surface was invented. The
                # error the validity sweep could not reach.
                "false_negative": (not reads_shell)
                and truth["fabricated_area_fraction"] > 0.01,
            }
        )

    informative = [r for r in rows if r["fabricated_area_fraction"] > 0.01]
    missed = [r for r in rows if r["false_negative"]]
    caught = [r for r in rows if r["reads_as_extrapolated_shell"]]
    report = {
        "voxel_size": voxel,
        "occlusion_centre_rule": (
            "Most protruding sample along the axis of greatest extent: deterministic, and "
            "on a feature rather than a flat panel, where a bridge departs from truth most."
        ),
        "fabrication_definition": (
            "Reconstructed area further than one voxel from the watertight reference, "
            "area-weighted. Surface Poisson interpolates correctly across a hole is not "
            "counted, which is the strict choice against the tool."
        ),
        "rows": rows,
        "smallest_occlusion_caught_as_diagonal_fraction": (
            min(r["occlusion_radius_as_diagonal_fraction"] for r in caught)
            if caught
            else None
        ),
        "worst_missed_fabrication": (
            max(r["fabricated_area_fraction"] for r in missed) if missed else 0.0
        ),
        "false_negatives": len(missed),
        "informative_rows": len(informative),
        "conclusive": bool(informative),
        "why_conclusiveness_matters": (
            "A row that fabricates nothing cannot expose a false negative, so a sweep with "
            "no informative row says nothing about the bands. Reported rather than scored "
            "as a pass."
        ),
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True))

    print(f"voxel {voxel:.5f}; occlusion centred on the most protruding sample")
    print(
        f"{'r/diag':>7} {'r(vox)':>7} {'removed':>8} {'unsup':>8} {'>3v':>8} "
        f"{'share':>8} {'fabricated':>11} {'maxtruth':>9}  reading"
    )
    for row in rows:
        flag = "  <== FALSE NEGATIVE" if row["false_negative"] else ""
        print(
            f"{row['occlusion_radius_as_diagonal_fraction']:7.2f} "
            f"{row['occlusion_radius_in_voxels']:7.1f} {row['samples_removed']:8d} "
            f"{row['unsupported_area_fraction']:8.4f} "
            f"{row['unsupported_area_beyond_3_voxels']:8.4f} "
            f"{row['share_beyond_3_voxels']:8.4f} "
            f"{row['fabricated_area_fraction']:11.4f} "
            f"{row['max_vertex_distance_to_truth_in_voxels']:9.2f}  "
            f"{'shell' if row['reads_as_extrapolated_shell'] else 'near-threshold'}{flag}"
        )
    if not informative:
        print(
            "\nINCONCLUSIVE: no occlusion fabricated more than 0.01 of area, so this "
            "sweep cannot say whether a real shell would be missed. Enlarge the occlusion."
        )
    elif missed:
        worst = max(missed, key=lambda r: r["fabricated_area_fraction"])
        print(
            f"\nWorst miss: radius {worst['occlusion_radius_as_diagonal_fraction']:.2f} of "
            f"the diagonal invented {worst['fabricated_area_fraction']:.4f} of area and "
            "still read as near-threshold."
        )
    else:
        print(
            f"\nNo false negative across {len(informative)} informative row(s): every "
            "occlusion that fabricated more than 0.01 of area read as a shell."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
