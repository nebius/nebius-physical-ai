"""Test whether anything reference-free separates invented surface from sparse coverage.

`_crop_justification` reports which side of `FAR_BAND_AREA_SHARE` the removed area fell on and
says plainly that this is its limit: *"thin coverage there and surface that does not belong
there both read this way [...] Deciding fabrication needs a reference surface."* Commit
`a828e58fd3` withdrew the stronger reading for that reason.

The last clause is the one worth testing. If it is false the runner can say more than it does;
if it is true it should stay withdrawn with a measurement behind it rather than an argument.

The hypothesis: a reconstructed point far from every sample is invented if its own normal
disagrees with the surface the nearby samples define, and is sparse-but-correct coverage if it
agrees. Poisson bridging a hole produces a membrane roughly perpendicular to the true surface
at the rim; a correctly reconstructed sparse region produces surface locally parallel to the
few samples that are there. Both lie far from samples. Only one is misoriented with respect to
them, and establishing that reads no reference surface -- only the cloud and the mesh.

The report says acceptance was frozen in a private program claim, but that file is not retained
in this repository, so its pre-measurement timing is not independently verifiable here. The
first implemented gate is the product's own counterexample: the cube sampled densely on five
faces and sparsely on the sixth.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import open3d as o3d
from npa.smoke.test_open3d_functional import _CUBE_FACES, _unit_cube
from npa.workbench.open3d.runner import _crop_justification, _support

#: Same as #1006, and for the same reason: normal orientation out of PCA is sign-ambiguous, so
#: the comparison is on |cos| and a right angle is the worst disagreement available.
NORMAL_AGREEMENT_DEGREES = 60.0

#: How many samples define the local surface a mesh normal is compared against. Ten is enough
#: for a stable plane fit and small enough that a sparse face still has neighbourhoods.
LOCAL_SAMPLES = 10

SEED = 7


def out_path() -> Path:
    raw = os.environ.get("OPEN3D_OUT_DIR")
    if not raw:
        raise SystemExit(
            "OPEN3D_OUT_DIR is required and must name an existing scratch directory; "
            "the harness emits raw measurements, not the annotated committed report."
        )
    directory = Path(raw)
    if not directory.is_dir():
        raise SystemExit(
            f"No directory at {directory}; set OPEN3D_OUT_DIR to an existing one."
        )
    return directory / "reference-free-fabrication.json"


def cube_observations(rng, dense: int, sparse: int) -> np.ndarray:
    """The product's own construction: `dense` samples on five faces, `sparse` on the sixth."""
    axes = np.eye(3)
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


def voxel_for(points: np.ndarray) -> tuple[float, float]:
    """The module's recommended voxel: twice the global median nearest-neighbour spacing."""
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    tree = o3d.geometry.KDTreeFlann(cloud)
    spacing = [np.sqrt(tree.search_knn_vector_3d(p, 2)[2][1]) for p in points]
    median = float(np.median(spacing))
    return 2.0 * median, median


def local_sample_normals(points: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """The surface the samples define near each query, by PCA over its nearest samples.

    Reference-free by construction: this reads the cloud and nothing else. The smallest
    eigenvector of the local covariance is the plane normal, with a sign that PCA cannot fix,
    which is why every comparison downstream takes an absolute cosine.
    """
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    tree = o3d.geometry.KDTreeFlann(cloud)
    normals = np.zeros_like(queries)
    for i, query in enumerate(queries):
        found, index, _ = tree.search_knn_vector_3d(query, LOCAL_SAMPLES)
        neighbourhood = points[np.asarray(index[:found])]
        if found < 3:
            normals[i] = (0.0, 0.0, 0.0)
            continue
        centred = neighbourhood - neighbourhood.mean(axis=0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        normals[i] = vt[-1]
    return normals


def sample_own_normals(points: np.ndarray) -> np.ndarray:
    """Each sample's own local surface normal, from the cloud alone."""
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=LOCAL_SAMPLES)
    )
    return np.asarray(cloud.normals)


#: The radius inside which an agreeing sample counts as support, in voxels. Six times the far
#: band, chosen once and held fixed across every case here. Also the censoring point: the
#: question asked is a boolean, so there is no reason to search further than the answer.
ORIENTATION_SUPPORT_VOXELS = 12.0


def orientation_consistent_support(
    points: np.ndarray,
    sample_normals: np.ndarray,
    centroids: np.ndarray,
    own_normal: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Is there a sample near this triangle that could be a sample *of* this triangle?

    The first candidate compares a triangle against whichever samples are nearest, and that is
    what makes it fire on the cube: a triangle on the sparse face near an edge has all ten of
    its nearest samples on the adjacent *dense* face, perpendicular to it. The samples are
    right and the mesh is right; they are simply not samples of each other.

    So this asks a different question. Within `radius`, is there any sample whose own
    orientation agrees with this triangle? Sparse-but-correct coverage has agreeing samples on
    its own patch, merely few and far apart. A membrane bridging a hole has none, because the
    surface it would belong to does not exist there at all.

    Returns the boolean and the distance to the nearest agreeing sample, the latter censored at
    `radius` because the boolean is the question and searching past the answer costs time for
    nothing. An earlier form of this built one KD-tree per distinct triangle normal, which for
    a sphere is one per triangle; it ran for fifteen minutes without finishing a case.

    Still reference-free: the cloud's normals are estimated from the cloud.
    """
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    tree = o3d.geometry.KDTreeFlann(cloud)
    cutoff = np.cos(np.radians(NORMAL_AGREEMENT_DEGREES))

    supported = np.zeros(len(centroids), dtype=bool)
    nearest = np.full(len(centroids), radius)
    for index, centroid in enumerate(centroids):
        found, neighbours, squared = tree.search_radius_vector_3d(centroid, radius)
        if found == 0:
            continue
        candidates = np.asarray(neighbours[:found])
        agreeing = np.abs(sample_normals[candidates] @ own_normal[index]) >= cutoff
        if not agreeing.any():
            continue
        supported[index] = True
        nearest[index] = np.sqrt(np.asarray(squared[:found])[agreeing].min())
    return ~supported, nearest


def measure(mesh, points: np.ndarray, voxel: float) -> dict[str, float]:
    """The product's far-band share, and the same share restricted to misoriented area.

    Same denominator as the product's reading -- unsupported area -- so the two are directly
    comparable, and the new one can only ever be the smaller of the pair.
    """
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    before = _support(o3d, mesh, cloud, voxel)
    justification = _crop_justification(before, 0, mesh)

    a, b, c = (
        vertices[triangles[:, 0]],
        vertices[triangles[:, 1]],
        vertices[triangles[:, 2]],
    )
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total = float(areas.sum())

    tree = o3d.geometry.KDTreeFlann(cloud)
    distances = np.array(
        [np.sqrt(tree.search_knn_vector_3d(v, 1)[2][0]) for v in vertices]
    )
    per_triangle_distance = distances[triangles].max(axis=1)

    centroids = vertices[triangles].mean(axis=1)
    sample_normal = local_sample_normals(points, centroids)
    own_normal = np.asarray(mesh.triangle_normals)
    cosine = np.clip(np.abs(np.sum(sample_normal * own_normal, axis=1)), 0.0, 1.0)
    disagreement = np.degrees(np.arccos(cosine))

    far = per_triangle_distance > voxel * 3.0
    unsupported = per_triangle_distance > voxel
    misoriented = disagreement > NORMAL_AGREEMENT_DEGREES

    unsupported_area = float(areas[unsupported].sum())

    def share(mask: np.ndarray) -> float:
        return (
            float(areas[mask].sum() / unsupported_area) if unsupported_area > 0 else 0.0
        )

    unsupported_in_orientation, consistent = orientation_consistent_support(
        points,
        sample_own_normals(points),
        centroids,
        own_normal,
        voxel * ORIENTATION_SUPPORT_VOXELS,
    )

    return {
        "unsupported_area_fraction": round(before["unsupported_area_fraction"], 6),
        "unsupported_area_beyond_3_voxels": round(
            before["unsupported_area_beyond_3_voxels"], 6
        ),
        "far_band_share_of_unsupported_area": round(
            justification["unsupported_area_share_beyond_3_voxels"], 6
        ),
        "far_share_recomputed_here": round(share(far), 6),
        "misoriented_far_share_of_unsupported_area": round(share(far & misoriented), 6),
        "orientation_unsupported_far_share_of_unsupported_area": round(
            share(far & unsupported_in_orientation), 6
        ),
        "median_orientation_consistent_support_in_voxels_on_far_area": round(
            float(np.median(consistent[far]) / voxel) if far.any() else 0.0, 3
        ),
        "misoriented_share_of_all_area": round(
            float(areas[misoriented].sum() / total), 6
        ),
        "median_disagreement_degrees_on_far_area": round(
            float(np.median(disagreement[far])) if far.any() else 0.0, 4
        ),
        "reads_as": justification["removed_surface_reads_as"],
    }


def cube_case(label: str, dense: int, sparse: int) -> dict[str, object]:
    rng = np.random.default_rng(SEED)
    mesh = _unit_cube(o3d)
    points = cube_observations(rng, dense, sparse)
    voxel, spacing = voxel_for(points)
    result = {
        "case": label,
        "per_face_counts": {"five_faces_each": dense, "sixth_face": sparse},
        "observations": len(points),
        "median_sample_spacing": round(spacing, 6),
        "voxel_size": round(voxel, 6),
        "invented_area_fraction": 0.0,
        "why_nothing_is_invented": (
            "Every mesh vertex lies on a face of the analytic unit cube and every observation "
            "is drawn on that same cube, so there is nothing for a fabrication reading to be "
            "right about."
        ),
    }
    result.update(measure(mesh, points, voxel))
    return result


def two_boxes(gap: float) -> object:
    """Two cubes with air between them. Poisson bridges the gap, and the bridge is in air."""
    mesh = o3d.geometry.TriangleMesh()
    for x in (0.0, 0.5 + gap):
        box = o3d.geometry.TriangleMesh.create_box(0.5, 0.5, 0.5)
        box.translate((x, 0.0, 0.0))
        mesh += box
    mesh.compute_vertex_normals()
    return mesh


def two_spheres(gap: float) -> object:
    """The same bridge over curved surface rather than flat, so the result is not about planes."""
    mesh = o3d.geometry.TriangleMesh()
    for x in (0.0, 0.5 + gap):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.25, resolution=40)
        sphere.translate((x + 0.25, 0.25, 0.25))
        mesh += sphere
    mesh.compute_vertex_normals()
    return mesh


def slotted_box(depth: float, width: float) -> object:
    """One connected solid with a slot cut into its top.

    Built by extruding a U profile, not by adding three boxes together. Adding them was the
    first attempt and it does not produce a solid: `+=` concatenates triangles without a
    boolean union, so the base's top face and the teeth's bottom faces survive *inside* the
    shape, `sample_points_poisson_disk` draws samples on them, and distance-to-reference then
    measures against a surface soup.

    That was not a small error in a number, it was the whole case. The area it called invented
    turned out to be a band just above the internal face, spanning the full width and only 2
    percent inside the slot -- the opposite of a slot roof. The fraction came out at a
    plausible 0.1041 either way, so nothing in the reading itself showed it. Checking the
    mechanism I had asserted is what showed it.
    """
    top = 0.5
    floor = top - depth
    left, right = (1.0 - width) / 2, (1.0 + width) / 2
    profile = [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, top),
        (right, top),
        (right, floor),
        (left, floor),
        (left, top),
        (0.0, top),
    ]
    #: Five rectangles that tile the U, so the end caps are triangulated without a general
    #: polygon triangulator and with no triangle crossing the slot. Three was the obvious
    #: tiling and it leaves a T-junction at (left, floor): the tooth tile's vertical edge runs
    #: the full height while the floor tile's stops at the slot floor, so the two edges only
    #: partially coincide and the mesh cannot close. `is_watertight()` said so.
    tiles = [
        (0.0, 0.0, left, floor),
        (0.0, floor, left, top),
        (right, 0.0, 1.0, floor),
        (right, floor, 1.0, top),
        (left, 0.0, right, floor),
    ]

    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []

    def add(point: tuple[float, float, float]) -> int:
        vertices.append(point)
        return len(vertices) - 1

    for y, flip in ((0.0, False), (0.5, True)):
        for x0, z0, x1, z1 in tiles:
            a = add((x0, y, z0))
            b = add((x1, y, z0))
            c = add((x1, y, z1))
            d = add((x0, y, z1))
            triangles += [(a, c, b), (a, d, c)] if flip else [(a, b, c), (a, c, d)]

    for (x0, z0), (x1, z1) in zip(profile, profile[1:] + profile[:1]):
        p0 = add((x0, 0.0, z0))
        p1 = add((x1, 0.0, z1))
        p2 = add((x1, 0.5, z1))
        p3 = add((x0, 0.5, z0))
        triangles += [(p0, p1, p2), (p0, p2, p3)]

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(vertices, dtype=float)),
        o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32)),
    )
    mesh = mesh.merge_close_vertices(1e-9)
    mesh.remove_duplicated_triangles()
    mesh.compute_vertex_normals()
    return mesh


def internal_face_share(mesh, samples: int = 20000, eps: float = 1e-4) -> float:
    """Share of surface samples that have solid on *both* sides, i.e. lie on an internal face.

    This is the check that catches the construction error above, and it is not an assertion
    about it: on the three-boxes-added version it reads 0.2203, on an extruded U it reads
    0.0000, and on a plain box it reads 0.0000. A reference with internal faces is sampled on
    them, so distance-to-reference then measures against a surface soup and "invented" means
    nothing.

    `is_watertight()` is not the check. It returns False for the extruded U and also for
    Open3D's own `boolean_difference` output, on winding rather than on holes, while occupancy
    through the slot is exactly right for both. The boolean output does have internal faces --
    0.0125 of samples, not falling away as eps shrinks, sitting exactly on the cut planes --
    which is why the U is built by hand here.
    """
    cloud = mesh.sample_points_uniformly(
        number_of_points=samples, use_triangle_normal=True
    )
    points, normals = np.asarray(cloud.points), np.asarray(cloud.normals)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    outward = scene.compute_occupancy(
        o3d.core.Tensor((points + eps * normals).astype(np.float32))
    ).numpy()
    inward = scene.compute_occupancy(
        o3d.core.Tensor((points - eps * normals).astype(np.float32))
    ).numpy()
    return float(((outward > 0.5) & (inward > 0.5)).mean())


def reference_geometries() -> list[tuple[str, object]]:
    """Three distinct geometries, as the claim's third criterion requires.

    All three invent surface when Poisson closes them, which the positive cases need and which
    is not automatic. A sphere with a cap of samples removed invents 0.0004 of its area --
    Poisson closing a cap of a sphere reproduces the sphere -- and a thin torus invents exactly
    nothing. Neither is a test of a fabrication detector. Both were measured and discarded
    before these three were chosen, and are recorded because a construction that fails to
    exercise an effect is not evidence the effect is absent.

    The 0.03-wide slot rows report unlocalized invented area, but this harness does not compute
    what share lies inside the slot footprint. It therefore cannot call that area a roof or
    attribute the zero candidate readings to a sampled floor. The retained label describes the
    20000-sample voxel ratio only; the 8000-sample ratio differs.
    """
    return [
        ("two boxes, gap 0.20", two_boxes(0.20)),
        ("two spheres, gap 0.20", two_spheres(0.20)),
        ("narrow slot, 1.5 voxels wide", slotted_box(0.40, 0.03)),
    ]


def bridged_case(label: str, reference, samples: int) -> dict[str, object]:
    """Sample a concave or disconnected surface and let Poisson invent the closure.

    The reference appears twice and only twice: to produce the samples, and afterwards to
    confirm the closure really is invented. Neither measure reads it.
    """
    cloud = reference.sample_points_poisson_disk(number_of_points=samples)
    kept = np.asarray(cloud.points)

    observed = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(kept))
    observed.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    observed.orient_normals_consistent_tangent_plane(k=30)
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        observed, depth=7
    )
    mesh.compute_vertex_normals()

    voxel, spacing = voxel_for(kept)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(reference))
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    centroids = vertices[triangles].mean(axis=1)
    to_reference = scene.compute_distance(
        o3d.core.Tensor(centroids.astype(np.float32))
    ).numpy()
    a, b, c = (
        vertices[triangles[:, 0]],
        vertices[triangles[:, 1]],
        vertices[triangles[:, 2]],
    )
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    invented = to_reference > voxel

    result = {
        "case": label,
        "observations": len(kept),
        "median_sample_spacing": round(spacing, 6),
        "voxel_size": round(voxel, 6),
        "invented_area_fraction": round(float(areas[invented].sum() / areas.sum()), 6),
        "internal_face_sample_share_of_the_reference": round(
            internal_face_share(reference), 6
        ),
        "how_invented_was_established": (
            "Distance from each reconstructed triangle's centroid to the reference surface, "
            "past one voxel. The reference produced the samples and confirms this afterwards; "
            "neither measure below reads it."
        ),
    }
    result.update(measure(mesh, kept, voxel))
    return result


def main() -> None:
    o3d.utility.random.seed(SEED)

    control = cube_case("dense on five faces, sparse on the sixth", 4000, 40)
    even = cube_case("even coverage of all six faces", 4000, 4000)

    recorded = {"unsupported": 0.3157, "beyond_3": 0.1316, "share": 0.4169}
    reproduced = {
        "what_the_product_records": recorded,
        "what_this_run_measures": {
            "unsupported": control["unsupported_area_fraction"],
            "beyond_3": control["unsupported_area_beyond_3_voxels"],
            "share": control["far_band_share_of_unsupported_area"],
        },
        "largest_absolute_difference": round(
            max(
                abs(recorded["unsupported"] - control["unsupported_area_fraction"]),
                abs(recorded["beyond_3"] - control["unsupported_area_beyond_3_voxels"]),
                abs(recorded["share"] - control["far_band_share_of_unsupported_area"]),
            ),
            6,
        ),
        "gate": "the claim allows 0.02; nothing after this counts if it is not met",
    }

    sparse_sweep = [
        cube_case(f"sparse sixth face, {count} samples", 4000, count)
        for count in (160, 80, 40, 20, 10)
    ]

    bridged = []
    for name, reference in reference_geometries():
        for samples in (20000, 8000):
            bridged.append(
                bridged_case(f"{name}, {samples} samples", reference, samples)
            )

    report = {
        "construction_validity_instrument": {
            "what_it_reads": (
                "Share of reference-surface samples with solid on both sides, so on an "
                "internal face rather than the boundary."
            ),
            "retained_literal_controls_not_recomputed_by_this_harness": {
                "three_boxes_added_together_my_first_slot": 0.2203,
                "open3d_boolean_difference_output": 0.0125,
                "extruded_u_profile_used_here": 0.0,
                "plain_box_control": 0.0,
            },
            "why_is_watertight_is_not_the_check": (
                "It returns False for the extruded U and for Open3D's own boolean_difference "
                "output, on winding rather than on holes, while occupancy through the slot is "
                "exactly right for both."
            ),
        },
        "what_this_tests": (
            "Whether the runner's own statement -- that deciding fabrication needs a reference "
            "surface -- is true. The candidate reads only the cloud and the mesh."
        ),
        "gate_1_the_products_counterexample_reproduces": reproduced,
        "cases": [control, even],
        "sparse_density_sweep_zero_invented_everywhere": sparse_sweep,
        "bridged_cases_real_invented_surface": bridged,
        "open3d_version": o3d.__version__,
        "numpy_seed": SEED,
        "normal_agreement_cutoff_degrees": NORMAL_AGREEMENT_DEGREES,
        "local_samples_per_plane_fit": LOCAL_SAMPLES,
        "report_scope": (
            "This raw harness output ends at the measurement fields above. The committed "
            "report's verdict, decisive-case explanation, product interpretation, and visual "
            "evaluation are hand-written annotations and are not regenerated here."
        ),
    }
    out_path().write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["gate_1_the_products_counterexample_reproduces"], indent=2))
    header = f"{'case':<46}{'invented':>9}{'far':>9}{'misor':>9}{'orient':>9}"
    print(f"\n{header}\n{'-' * len(header)}")
    for group in (
        "cases",
        "sparse_density_sweep_zero_invented_everywhere",
        "bridged_cases_real_invented_surface",
    ):
        for case in report[group]:
            print(
                f"{case['case'][:45]:<46}"
                f"{case['invented_area_fraction']:>9.4f}"
                f"{case['far_band_share_of_unsupported_area']:>9.4f}"
                f"{case['orientation_unsupported_far_share_of_unsupported_area']:>9.4f}"
            )


if __name__ == "__main__":
    main()
