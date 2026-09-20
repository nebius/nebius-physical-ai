"""Does the sensitivity floor hold on a second geometry and a second sampling scheme?

The review lane established that `crop_justification` is a monotone detector with a floor
near 9 percent invented area, and both of us flagged the same weakness: one synthetic
geometry, one sampling scheme, one resolution ratio. A floor quoted from a single cell of
that grid cannot be relied on, and the docstring currently says so rather than claiming
otherwise. This fills in the grid.

The construction is the review lane's, reproduced rather than reinvented so the numbers are
comparable: score a mesh against samples drawn from itself, then delete the observations
inside a polar cap while leaving the mesh covering it. The surface over the cap is then
genuinely invented relative to what was observed, and the invented fraction is known by
area rather than estimated.

Two axes are varied:

- **Geometry.** An icosphere, which is convex and smooth and reproduces the reviewer's
  baseline, against the Armadillo, which is concave with fine features. If the floor is a
  property of sphere-like geometry it should move here.
- **Sampling.** Uniform random, which leaves gaps well above the median nearest-neighbour
  spacing, against Poisson-disk, which is far more even. The reviewer showed the headline
  unsupported fraction is largely an artefact of this choice, so the question is whether
  the *bands* inherit that sensitivity.

Run inside the Open3D image:
    docker run --rm -v "$EVIDENCE_DIR:/data" --entrypoint python "$IMAGE" \
        /data/harness/floor_across_geometry_and_sampling.py
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

OUT = Path("/data/floor-across-geometry-and-sampling.json")

SAMPLES = 60000
#: Cap half-angles in degrees, spanning the reviewer's range so the curves line up.
CAP_DEGREES = (0.0, 20.0, 30.0, 35.0, 40.0, 50.0, 60.0, 90.0)


class _Mesh:
    def __init__(self, count: int) -> None:
        self.vertices = range(count)


def geometries() -> dict:
    sphere = o3d.geometry.TriangleMesh.create_icosahedron(radius=1.0)
    sphere = sphere.subdivide_loop(number_of_iterations=4)
    sphere.compute_vertex_normals()

    armadillo = o3d.io.read_triangle_mesh(o3d.data.ArmadilloMesh().path)
    armadillo.compute_vertex_normals()
    # Normalize scale so a voxel derived from spacing means the same thing on both.
    box = armadillo.get_axis_aligned_bounding_box()
    armadillo.translate(-box.get_center())
    armadillo.scale(2.0 / float(np.max(box.get_extent())), center=(0, 0, 0))
    armadillo.compute_vertex_normals()
    return {"icosphere": sphere, "armadillo": armadillo}


def sample(mesh, scheme: str):
    if scheme == "uniform_random":
        return mesh.sample_points_uniformly(number_of_points=SAMPLES)
    if scheme == "poisson_disk":
        return mesh.sample_points_poisson_disk(number_of_points=SAMPLES, init_factor=3)
    raise ValueError(scheme)


def triangle_areas(mesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    a, b, c = (
        vertices[triangles[:, 0]],
        vertices[triangles[:, 1]],
        vertices[triangles[:, 2]],
    )
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def centroids(mesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices)
    return vertices[np.asarray(mesh.triangles)].mean(axis=1)


def main() -> int:
    report: dict = {
        "construction": (
            "The review lane's: score a mesh against samples drawn from itself, then delete "
            "the observations inside a polar cap while leaving the mesh covering it. "
            "Reproduced rather than reinvented so the curves are comparable."
        ),
        "samples": SAMPLES,
        "reading_boundary": FABRICATION_AREA_SHARE,
        "cases": {},
    }

    for gname, mesh in geometries().items():
        areas = triangle_areas(mesh)
        total_area = float(areas.sum())
        centre = centroids(mesh)
        # Cap measured from the +z pole of the mesh's own centred frame.
        direction = centre - np.asarray(
            mesh.get_axis_aligned_bounding_box().get_center()
        )
        norms = np.linalg.norm(direction, axis=1)
        cosine = np.divide(
            direction[:, 2], norms, out=np.zeros_like(norms), where=norms > 0
        )

        for scheme in ("uniform_random", "poisson_disk"):
            cloud = sample(mesh, scheme)
            spacing = float(
                np.median(np.asarray(cloud.compute_nearest_neighbor_distance()))
            )
            voxel = spacing * 2.0
            points = np.asarray(cloud.points)
            pdir = points - np.asarray(
                mesh.get_axis_aligned_bounding_box().get_center()
            )
            pnorm = np.linalg.norm(pdir, axis=1)
            pcos = np.divide(
                pdir[:, 2], pnorm, out=np.zeros_like(pnorm), where=pnorm > 0
            )

            rows = []
            for degrees in CAP_DEGREES:
                limit = np.cos(np.radians(degrees))
                keep = pcos < limit if degrees > 0 else np.ones(len(points), bool)
                if keep.sum() < 100:
                    continue
                observed = o3d.geometry.PointCloud()
                observed.points = o3d.utility.Vector3dVector(points[keep])
                # Invented area: mesh area inside the cap, where no observation survives.
                invented = (
                    float(areas[cosine >= limit].sum() / total_area)
                    if degrees > 0
                    else 0.0
                )
                support = _support(o3d, mesh, observed, voxel)
                reading = _crop_justification(support, 0, _Mesh(len(mesh.vertices)))
                share = reading["unsupported_area_share_beyond_3_voxels"]
                rows.append(
                    {
                        "cap_degrees": degrees,
                        "invented_area_fraction": invented,
                        "unsupported_area_fraction": support[
                            "unsupported_area_fraction"
                        ],
                        "unsupported_area_beyond_3_voxels": support[
                            "unsupported_area_beyond_3_voxels"
                        ],
                        "share_beyond_3_voxels": share,
                        "reads_as_extrapolated_shell": reading[
                            "removed_surface_reads_as"
                        ]
                        == "extrapolated shell",
                    }
                )

            fired = [r for r in rows if r["reads_as_extrapolated_shell"]]
            shares = [r["share_beyond_3_voxels"] for r in rows]
            invented = [r["invented_area_fraction"] for r in rows]
            report["cases"][f"{gname}__{scheme}"] = {
                "spacing": spacing,
                "voxel": voxel,
                "spacing_over_voxel": spacing / voxel,
                "rows": rows,
                "floor_invented_area_fraction": (
                    min(r["invented_area_fraction"] for r in fired) if fired else None
                ),
                "monotone_in_invented_fraction": bool(
                    np.all(np.diff(np.array(shares)[np.argsort(invented)]) >= -1e-9)
                ),
                "zero_error_unsupported_fraction": rows[0]["unsupported_area_fraction"],
                "zero_error_reads_as_shell": rows[0]["reads_as_extrapolated_shell"],
            }

    floors = {
        k: v["floor_invented_area_fraction"]
        for k, v in report["cases"].items()
        if v["floor_invented_area_fraction"] is not None
    }
    report["summary"] = {
        "floors": floors,
        "floor_spread": (max(floors.values()) - min(floors.values()))
        if floors
        else None,
        "monotone_everywhere": all(
            v["monotone_in_invented_fraction"] for v in report["cases"].values()
        ),
        "no_false_positive_on_the_perfect_case": all(
            not v["zero_error_reads_as_shell"] for v in report["cases"].values()
        ),
        "zero_error_unsupported_fraction_spread": {
            k: v["zero_error_unsupported_fraction"] for k, v in report["cases"].items()
        },
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True))

    for case, block in report["cases"].items():
        print(f"\n=== {case}  (spacing/voxel {block['spacing_over_voxel']:.3f})")
        print(
            f"{'cap':>5} {'invented':>9} {'unsup':>8} {'>3v':>8} {'share':>8}  reading"
        )
        for row in block["rows"]:
            print(
                f"{row['cap_degrees']:5.0f} {row['invented_area_fraction']:9.4f} "
                f"{row['unsupported_area_fraction']:8.4f} "
                f"{row['unsupported_area_beyond_3_voxels']:8.4f} "
                f"{row['share_beyond_3_voxels']:8.4f}  "
                f"{'shell' if row['reads_as_extrapolated_shell'] else 'near-threshold'}"
            )
        print(
            f"floor {block['floor_invented_area_fraction']}, monotone "
            f"{block['monotone_in_invented_fraction']}"
        )
    print(f"\nsummary: {json.dumps(report['summary'], indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
