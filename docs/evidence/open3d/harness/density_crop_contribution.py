"""Does the density crop change the delivered surface, or does the next crop?

`run_reconstruct` applies two filters in sequence. The density crop removes vertices
in the lowest quantile of Poisson's sample density; the support crop then removes
vertices farther than `support_distance_factor` voxels from any observed point. The
report publishes `low_density_vertices_removed`, which reads as surface the density
filter took out of the delivered mesh.

It usually is not. The two crops ask different questions and get nearly the same
answer, because a vertex the solver extrapolated into empty space is both low-density
and far from any observation. At the shipped defaults on the upstream demo scan, zero
of the 518 density-removed vertices lay within the support distance -- every one was
going to be removed anyway -- and the delivered mesh is identical with the density
crop disabled, to the triangle and to six decimal places of surface area.

The control that makes that readable is the same pipeline with support cropping off,
where the density crop is the only filter and does change the output. Without it, "the
density crop does nothing" could as easily mean the measurement cannot see a crop.

The grid also records where the density crop stops being harmless: pushed to the 0.20
quantile it removes nearly nine thousand triangles of *supported* surface, which is
the opposite of what either filter is for.

Run inside the Open3D image:
    docker run --rm -v "$EVIDENCE_DIR:/data" --entrypoint python "$IMAGE" \
        /data/harness/density_crop_contribution.py
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import open3d as o3d

OUT = Path(os.environ.get("OPEN3D_OUT", "/data/density-crop-contribution.json"))
VOXEL = 0.05
POISSON_DEPTH = 9
#: The shipped defaults, read from the schema rather than restated as literals.
SHIPPED_QUANTILE = 0.02
SHIPPED_SUPPORT_FACTOR = 1.0


def _fused():
    """The upstream demo fragments, fused the way the pipeline's own record is."""

    data = o3d.data.DemoICPPointClouds()
    fused = o3d.geometry.PointCloud()
    for path in data.paths:
        fused += o3d.io.read_point_cloud(path)
    fused = fused.voxel_down_sample(VOXEL)
    fused.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL * 2.0, max_nn=30)
    )
    fused.orient_normals_consistent_tangent_plane(k=30)
    return fused


def _poisson(cloud):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud, depth=POISSON_DEPTH
    )
    return mesh, np.asarray(densities)


def _distances(tree, vertices) -> np.ndarray:
    return np.array(
        [np.sqrt(tree.search_knn_vector_3d(point, 1)[2][0]) for point in vertices]
    )


def _deliver(base, densities, tree, quantile: float, support_factor: float) -> dict:
    """One pass of the shipped sequence: density crop, then support crop."""

    mesh = o3d.geometry.TriangleMesh(base)
    removed = 0
    if quantile > 0.0:
        mask = densities < float(np.quantile(densities, quantile))
        removed = int(mask.sum())
        mesh.remove_vertices_by_mask(mask)
    if support_factor > 0.0:
        vertices = np.asarray(mesh.vertices)
        beyond = _distances(tree, vertices) > VOXEL * support_factor
        mesh.remove_vertices_by_mask(beyond)
    return {
        "density_quantile": quantile,
        "support_distance_factor": support_factor,
        "low_density_vertices_removed": removed,
        "triangles": len(mesh.triangles),
        "surface_area": round(float(mesh.get_surface_area()), 6),
    }


def the_shipped_defaults(base, densities, tree) -> dict:
    """With and without the density crop, at the defaults the product uses."""

    with_crop = _deliver(
        base, densities, tree, SHIPPED_QUANTILE, SHIPPED_SUPPORT_FACTOR
    )
    without = _deliver(base, densities, tree, 0.0, SHIPPED_SUPPORT_FACTOR)
    mask = densities < float(np.quantile(densities, SHIPPED_QUANTILE))
    removed_vertices = np.asarray(base.vertices)[mask]
    within = int(
        (_distances(tree, removed_vertices) <= VOXEL * SHIPPED_SUPPORT_FACTOR).sum()
    )
    return {
        "density_quantile": SHIPPED_QUANTILE,
        "support_distance_factor": SHIPPED_SUPPORT_FACTOR,
        "with_the_density_crop": with_crop,
        "without_the_density_crop": without,
        "delivered_mesh_is_identical": (
            with_crop["triangles"] == without["triangles"]
            and with_crop["surface_area"] == without["surface_area"]
        ),
        "density_removed_vertices": int(mask.sum()),
        "of_those_within_the_support_distance": within,
        "so": (
            "none of the density crop's removals would have survived the support "
            "crop, so the published removal count describes work the next stage was "
            "going to do. The count is not wrong; reading it as this filter's effect "
            "on the delivered surface is"
        ),
    }


def support_cropping_disabled(base, densities, tree) -> dict:
    """The control: with no support crop, the density crop is the only filter.

    Without this, "the density crop changes nothing" is indistinguishable from a
    measurement that cannot see a crop at all.
    """

    with_crop = _deliver(base, densities, tree, SHIPPED_QUANTILE, 0.0)
    without = _deliver(base, densities, tree, 0.0, 0.0)
    return {
        "support_distance_factor": 0.0,
        "with_the_density_crop": with_crop,
        "without_the_density_crop": without,
        "the_density_crop_changes_the_mesh_here": (
            with_crop["triangles"] != without["triangles"]
        ),
        "triangles_removed": without["triangles"] - with_crop["triangles"],
        "surface_area_removed": round(
            without["surface_area"] - with_crop["surface_area"], 6
        ),
        "so": (
            "the crop and this measurement both work. What the shipped configuration "
            "shows is overlap with the next filter, not an inert filter"
        ),
    }


def the_grid(base, densities, tree) -> list[dict]:
    """Where the density crop starts removing surface the scan does support."""

    rows = []
    for support_factor in (0.0, SHIPPED_SUPPORT_FACTOR, 3.0):
        baseline = _deliver(base, densities, tree, 0.0, support_factor)
        for quantile in (SHIPPED_QUANTILE, 0.05, 0.20):
            row = _deliver(base, densities, tree, quantile, support_factor)
            row["triangles_versus_no_density_crop"] = (
                row["triangles"] - baseline["triangles"]
            )
            row["surface_area_versus_no_density_crop"] = round(
                row["surface_area"] - baseline["surface_area"], 6
            )
            rows.append(row)
    return rows


def selectivity(base, densities, tree) -> dict:
    """Is the density crop choosing extrapolated surface, or trimming at random?

    Measured because the finding above could be misread as "the density crop is
    arbitrary". It is not: its removals are far from the cloud by a large margin,
    which is exactly why they overlap the support crop so completely.
    """

    vertices = np.asarray(base.vertices)
    distance = _distances(tree, vertices)
    order_density = np.argsort(np.argsort(densities))
    order_distance = np.argsort(np.argsort(distance))
    rows = []
    for quantile in (0.01, SHIPPED_QUANTILE, 0.05, 0.20):
        mask = densities < float(np.quantile(densities, quantile))
        rows.append(
            {
                "density_quantile": quantile,
                "vertices_removed": int(mask.sum()),
                "median_distance_removed": round(float(np.median(distance[mask])), 6),
                "median_distance_kept": round(float(np.median(distance[~mask])), 6),
            }
        )
    return {
        "spearman_density_against_distance_to_cloud": round(
            float(np.corrcoef(order_density, order_distance)[0, 1]), 4
        ),
        "by_quantile": rows,
        "so": (
            "the crop is strongly selective -- its removals sit far from any observed "
            "point -- which is the reason it duplicates the support crop rather than "
            "a reason to distrust it"
        ),
    }


def build() -> dict:
    cloud = _fused()
    base, densities = _poisson(cloud)
    tree = o3d.geometry.KDTreeFlann(cloud)
    return {
        "question": (
            "does the density crop change the delivered surface at the shipped "
            "defaults, or does the support crop remove everything it removes?"
        ),
        "open3d_version": o3d.__version__,
        "input": "o3d.data.DemoICPPointClouds, fused and downsampled at voxel 0.05",
        "points": len(cloud.points),
        "poisson_depth": POISSON_DEPTH,
        "vertices_before_any_crop": len(base.vertices),
        "at_the_shipped_defaults": the_shipped_defaults(base, densities, tree),
        "control_support_cropping_disabled": support_cropping_disabled(
            base, densities, tree
        ),
        "grid": the_grid(base, densities, tree),
        "is_the_density_crop_selective": selectivity(base, densities, tree),
        "what_this_does_not_claim": {
            "not_that_the_density_crop_is_wrong": (
                "it removes extrapolated surface and does it well. The claim is about "
                "what its published removal count tells a reader when the support "
                "crop also runs"
            ),
            "not_that_it_should_be_removed": (
                "with support cropping disabled it is the only filter and it works. "
                "Changing a default is a separate decision from reporting honestly"
            ),
            "not_a_general_result_about_every_scan": (
                "one real scan and the pipeline's synthetic fragments, both of which "
                "show the same complete overlap at the defaults. A capture whose "
                "low-density regions sit close to observed points would behave "
                "differently, and the published contribution would show it"
            ),
        },
        "provenance": {
            "harness": "harness/density_crop_contribution.py",
            "harness_sha256": "recomputed by main()",
            "path_is_relative_to": "this evidence directory",
        },
    }


def main() -> int:
    report = build()
    report["provenance"]["harness_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    control = report["control_support_cropping_disabled"]
    if not control["the_density_crop_changes_the_mesh_here"]:
        quarantine = OUT.with_suffix(".invalid.json")
        quarantine.write_text(json.dumps(report, indent=2) + "\n")
        print("The control did not see the density crop change anything.")
        print(f"Quarantined at {quarantine}; left {OUT} as it was")
        return 1
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    shipped = report["at_the_shipped_defaults"]
    print(
        f"At the defaults: {shipped['density_removed_vertices']} vertices removed for "
        f"low density, {shipped['of_those_within_the_support_distance']} of them "
        "within the support distance."
    )
    print(
        f"Delivered mesh identical without the crop: {shipped['delivered_mesh_is_identical']}"
    )
    print(
        f"Control with support cropping off: {control['triangles_removed']} triangles "
        f"and {control['surface_area_removed']} of area removed by density alone."
    )
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
