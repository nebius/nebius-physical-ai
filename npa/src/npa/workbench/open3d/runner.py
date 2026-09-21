"""Subprocess that invokes real Open3D pipelines, never a surrogate alignment.

Isolated for two reasons. Open3D is a native extension that aborts the whole
interpreter on a bad input rather than raising, and keeping it out of the parent
means `runtime`, `schemas` and `artifacts` stay importable — and testable —
without the 400 MB wheel present.

Every registration number this writes comes back from Open3D's own
`RegistrationResult`; nothing is recomputed, smoothed, or defaulted here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, NamedTuple

from .artifacts import (
    PAIRS_JOURNAL,
    RESULT_SCHEMA,
    Open3dError,
    canonical,
    summarize,
)
from .schemas import (
    DEMO_DATA_CLASS,
    DEMO_DATA_RELEASE,
    SOURCE_VERSION,
    ReconstructRequest,
    RegistrationManifest,
    fragment_id_for,
)

#: Cap on points logged per Rerun entity. Recorded in the recording's own
#: provenance so a decimated view is never mistaken for the full cloud.
RRD_POINT_BUDGET = 200_000


def _open3d():
    import open3d as o3d

    if o3d.__version__ != SOURCE_VERSION:
        raise Open3dError(
            f"installed open3d {o3d.__version__} does not match the reviewed "
            f"contract {SOURCE_VERSION}"
        )
    return o3d


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()


def _read_cloud(o3d, path: Path):
    cloud = o3d.io.read_point_cloud(str(path))
    if not cloud.has_points():
        raise Open3dError(f"{path.name} decoded to an empty point cloud")
    return cloud


def _prepared(o3d, cloud, manifest: RegistrationManifest):
    """`voxel_down_sample` + `estimate_normals` + `compute_fpfh_feature`."""

    voxel = manifest.voxel_size
    down = cloud.voxel_down_sample(voxel)
    if not down.has_points():
        raise Open3dError(
            f"voxel_size {voxel} removed every point; use a smaller voxel size"
        )
    normals = o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel * manifest.normal_radius_factor, max_nn=30
    )
    down.estimate_normals(normals)
    cloud.estimate_normals(normals)
    feature = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel * manifest.feature_radius_factor, max_nn=100
        ),
    )
    return down, feature


def _result_summary(result) -> dict[str, Any]:
    return {
        "fitness": float(result.fitness),
        "inlier_rmse": float(result.inlier_rmse),
        "correspondence_count": int(len(result.correspondence_set)),
    }


def _register_pair(o3d, source, target, manifest: RegistrationManifest):
    """Upstream global registration followed by upstream ICP refinement."""

    voxel = manifest.voxel_size
    source_down, source_feature = _prepared(o3d, source, manifest)
    target_down, target_feature = _prepared(o3d, target, manifest)
    ransac_distance = voxel * manifest.distance_factor
    registration = o3d.pipelines.registration
    coarse = registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_feature,
        target_feature,
        True,
        ransac_distance,
        registration.TransformationEstimationPointToPoint(False),
        3,
        [
            registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            registration.CorrespondenceCheckerBasedOnDistance(ransac_distance),
        ],
        registration.RANSACConvergenceCriteria(
            manifest.ransac_max_iteration, manifest.ransac_confidence
        ),
    )
    if len(coarse.correspondence_set) == 0:
        raise Open3dError(
            "RANSAC feature matching found no correspondences; the fragments may "
            "not overlap at this voxel size"
        )
    estimation = (
        registration.TransformationEstimationPointToPlane()
        if manifest.icp_estimation == "point_to_plane"
        else registration.TransformationEstimationPointToPoint()
    )
    icp_distance = voxel * manifest.icp_distance_factor
    fine = registration.registration_icp(
        source, target, icp_distance, coarse.transformation, estimation
    )
    if len(fine.correspondence_set) == 0:
        raise Open3dError("ICP refinement found no correspondences")
    information = registration.get_information_matrix_from_point_clouds(
        source, target, icp_distance, fine.transformation
    )
    return {
        "source_points": len(source.points),
        "target_points": len(target.points),
        "source_downsampled_points": len(source_down.points),
        "target_downsampled_points": len(target_down.points),
        "fpfh_dimension": int(source_feature.data.shape[0]),
        "global_registration": _result_summary(coarse),
        "icp": _result_summary(fine),
        "transformation": [
            [float(cell) for cell in row] for row in fine.transformation
        ],
        "information_matrix": [[float(cell) for cell in row] for row in information],
    }


def _report(kind: str, rows: list[dict[str, Any]], run_id: str) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA,
        "engine": "open3d",
        "source_version": SOURCE_VERSION,
        "run_id": run_id,
        "kind": kind,
        "summary": summarize(rows),
    }


def _load_manifest(payload: dict[str, Any]) -> RegistrationManifest:
    return RegistrationManifest.model_validate(payload["manifest"])


def _fragment_paths(payload: dict[str, Any]) -> dict[str, Path]:
    return {key: Path(value) for key, value in payload["fragments"].items()}


def _seeded(o3d, manifest: RegistrationManifest) -> None:
    """Make RANSAC repeatable; Open3D seeds a global generator, not a call."""

    o3d.utility.random.seed(manifest.random_seed)


def run_stage_demo(
    payload: dict[str, Any], output: Path, run_id: str
) -> dict[str, Any]:
    """Stage the upstream `open3d.data` demo fragments as real input scans."""

    o3d = _open3d()
    dataset = getattr(o3d.data, DEMO_DATA_CLASS)()
    staged: list[dict[str, Any]] = []
    fragments = output / "fragments"
    fragments.mkdir(parents=True, exist_ok=True)
    for path in map(Path, dataset.paths):
        # Decode before publishing: a staged file that Open3D cannot read is not
        # an input scan, and the download itself does not prove readability.
        cloud = _read_cloud(o3d, path)
        target = fragments / path.name
        target.write_bytes(path.read_bytes())
        staged.append(
            {
                "id": fragment_id_for(path.name),
                "filename": path.name,
                "sha256": _digest(target),
                "bytes": target.stat().st_size,
                "points": len(cloud.points),
            }
        )
    if len(staged) < 2:
        raise Open3dError("demo dataset did not provide at least two fragments")
    return {
        "run_id": run_id,
        "demo_data_class": DEMO_DATA_CLASS,
        "demo_data_release": DEMO_DATA_RELEASE,
        "fragments": staged,
    }


def run_register(payload: dict[str, Any], output: Path, run_id: str) -> dict[str, Any]:
    """Register consecutive fragment pairs and publish each aligned cloud."""

    o3d = _open3d()
    manifest = _load_manifest(payload)
    paths = _fragment_paths(payload)
    _seeded(o3d, manifest)
    aligned_dir = output / "aligned"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    ordered = [fragment.id for fragment in manifest.fragments]
    for source_id, target_id in zip(ordered, ordered[1:]):
        started = time.perf_counter()
        source = _read_cloud(o3d, paths[source_id])
        target = _read_cloud(o3d, paths[target_id])
        measured = _register_pair(o3d, source, target, manifest)
        pair_id = f"{source_id}__{target_id}"
        aligned = aligned_dir / f"{pair_id}.ply"
        # Write the source in the target's frame: that is the thing the
        # transformation claims, so it is the thing a reviewer should be able to
        # open. `transform` mutates, hence the explicit copy.
        moved = o3d.geometry.PointCloud(source)
        moved.transform(measured["transformation"])
        if not o3d.io.write_point_cloud(str(aligned), moved):
            raise Open3dError(f"Open3D could not write the aligned cloud {pair_id}")
        rows.append(
            {
                "pair_id": pair_id,
                "source_id": source_id,
                "target_id": target_id,
                **measured,
                "aligned_points": len(moved.points),
                "aligned_sha256": _digest(aligned),
                "aligned_bytes": aligned.stat().st_size,
                "wall_seconds": time.perf_counter() - started,
            }
        )
    (output / PAIRS_JOURNAL).write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    report = _report("register", rows, run_id)
    (output / "result.json").write_bytes(canonical(report))
    return report


def run_multiway(payload: dict[str, Any], output: Path, run_id: str) -> dict[str, Any]:
    """Upstream multiway registration: full pairwise graph, then optimization."""

    o3d = _open3d()
    registration = o3d.pipelines.registration
    manifest = _load_manifest(payload)
    paths = _fragment_paths(payload)
    _seeded(o3d, manifest)
    import numpy as np

    ordered = [fragment.id for fragment in manifest.fragments]
    clouds = {key: _read_cloud(o3d, paths[key]) for key in ordered}
    graph = registration.PoseGraph()
    odometry = np.identity(4)
    graph.nodes.append(registration.PoseGraphNode(odometry))
    rows: list[dict[str, Any]] = []
    for source_index in range(len(ordered)):
        for target_index in range(source_index + 1, len(ordered)):
            source_id, target_id = ordered[source_index], ordered[target_index]
            started = time.perf_counter()
            measured = _register_pair(
                o3d, clouds[source_id], clouds[target_id], manifest
            )
            transformation = np.asarray(measured["transformation"], dtype=float)
            information = np.asarray(measured["information_matrix"], dtype=float)
            # Consecutive pairs are odometry (certain); everything else is a loop
            # closure the optimizer is allowed to prune.
            consecutive = target_index == source_index + 1
            if consecutive:
                odometry = transformation @ odometry
                graph.nodes.append(registration.PoseGraphNode(np.linalg.inv(odometry)))
            graph.edges.append(
                registration.PoseGraphEdge(
                    source_index,
                    target_index,
                    transformation,
                    information,
                    uncertain=not consecutive,
                )
            )
            rows.append(
                {
                    "pair_id": f"{source_id}__{target_id}",
                    "source_id": source_id,
                    "target_id": target_id,
                    **measured,
                    # Multiway publishes one fused cloud rather than per-pair
                    # files, so the per-pair aligned identity is the fused input.
                    "aligned_points": measured["source_points"],
                    "aligned_sha256": hashlib.sha256(
                        canonical(measured["transformation"])
                    ).hexdigest(),
                    "wall_seconds": time.perf_counter() - started,
                }
            )
    option = registration.GlobalOptimizationOption(
        max_correspondence_distance=manifest.voxel_size * manifest.icp_distance_factor,
        edge_prune_threshold=0.25,
        reference_node=0,
    )
    registration.global_optimization(
        graph,
        registration.GlobalOptimizationLevenbergMarquardt(),
        registration.GlobalOptimizationConvergenceCriteria(),
        option,
    )
    # Serialize the graph Open3D actually optimized. `global_optimization` may
    # prune edges below the threshold, so reusing the pre-optimization list would
    # publish a graph that no longer matches the poses below it.
    edges = [
        {
            "source_node_id": int(edge.source_node_id),
            "target_node_id": int(edge.target_node_id),
            "uncertain": bool(edge.uncertain),
            "confidence": float(edge.confidence),
            "transformation": [
                [float(cell) for cell in row] for row in edge.transformation
            ],
        }
        for edge in graph.edges
    ]
    fused = o3d.geometry.PointCloud()
    nodes: list[dict[str, Any]] = []
    for index, key in enumerate(ordered):
        pose = np.asarray(graph.nodes[index].pose, dtype=float)
        moved = o3d.geometry.PointCloud(clouds[key])
        moved.transform(pose)
        fused += moved
        nodes.append(
            {
                "fragment_id": key,
                "pose": [[float(cell) for cell in row] for row in pose],
                "points": len(moved.points),
            }
        )
    fused_down = fused.voxel_down_sample(manifest.voxel_size)
    if not fused_down.has_points():
        raise Open3dError("fused cloud is empty after voxel downsampling")
    fused_path = output / "fused.ply"
    if not o3d.io.write_point_cloud(str(fused_path), fused_down):
        raise Open3dError("Open3D could not write the fused cloud")
    pose_graph = {
        "nodes": nodes,
        "edges": edges,
        "reference_node": 0,
        "edge_prune_threshold": 0.25,
        "fused_points": len(fused_down.points),
        "fused_points_before_downsample": len(fused.points),
        "fused_sha256": _digest(fused_path),
    }
    (output / "pose_graph.json").write_bytes(canonical(pose_graph))
    (output / PAIRS_JOURNAL).write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    report = _report("multiway", rows, run_id)
    report["pose_graph"] = pose_graph
    (output / "result.json").write_bytes(canonical(report))
    return report


def _sample_distances(o3d, cloud, vertices):
    """Distance from each mesh vertex to the nearest observed sample."""

    import numpy as np

    tree = o3d.geometry.KDTreeFlann(cloud)
    return np.array(
        [np.sqrt(tree.search_knn_vector_3d(vertex, 1)[2][0]) for vertex in vertices]
    )


#: Below this share of the pre-crop unsupported area lying past three voxels, what the crop
#: removed sat near the support threshold rather than far from any observation.
#:
#: This is a detector with a sensitivity floor, not a threshold between two clean
#: populations, and **it errs in both directions**.
#:
#: **It over-calls fabrication when the observations are spatially uneven.** This measures
#: distance from the nearest *observation*, which conflates two different things: surface
#: far from the true object, and surface the scanner simply covered sparsely. A closed
#: triangulated unit cube that is exactly its own ground truth -- every vertex on the true
#: cube, nothing invented anywhere -- sampled densely on five faces and sparsely on the
#: sixth, at the recommended voxel of twice the global median sample spacing, measured an
#: unsupported area fraction of 0.3157 with 0.1316 past three voxels: a far share of 0.4169,
#: which reads as `far from any observation`. The same cube sampled evenly reads 0.0 past three
#: voxels. Evenness is the whole difference. This was found by an independent audit and
#: reproduced here with these functions and real Open3D; see
#: `evidence/open3d/irregular-density-counterexample.json`.
#:
#: So a *high* reading does not establish that anything was invented. It says the removed
#: area sat far from any observation, and sparse coverage of correct geometry produces that
#: too. Distinguishing the two needs a reference surface, which a scan does not have.
#:
#: **It also under-calls small fabrications, and below the floor it is confidently wrong
#: rather than undecided.** Measured by scoring a mesh against samples drawn from itself
#: and deleting the observations inside a polar cap, so the invented fraction is known by
#: area: the share is monotone in that fraction in every case tried, but there is a floor
#: below which genuine fabrication reads as `near-threshold surface`.
#:
#: So a *low* reading does not mean the surface is clean either. Neither branch is a verdict
#: about ground truth. Both are statements about distance from observations.
#:
#: **Superseded conclusion, kept so it is not re-derived:** earlier revisions of this
#: constant claimed the error was one-sided and that an area called invented here is
#: invented, on the strength of twelve zero-error reconstructions that read near-threshold.
#: Those measurements stand and are unaltered; the conclusion drawn from them does not. Every
#: one of them sampled convex geometry evenly -- uniform or Poisson-disk over the whole
#: surface -- so the sweep varied sample *spacing* and never sample *evenness*, which is the
#: variable that breaks it.
#:
#: **Do not quote the floor as a number.** It moves with geometry, sampling and resolution
#: — 0.011 to 0.037 invented area across two geometries and two sampling schemes here,
#: against 0.09 in the review lane's independent setup at a different resolution ratio.
#: That is nearly an order of magnitude. What held in every case is monotonicity in the
#: invented fraction at fixed sampling, and that is all it is safe to rely on: the ordering
#: is informative, the absolute level is not comparable across scans.
#:
#: **`unsupported_area_fraction` fails for a reason that is fixable at capture time.** On a
#: reconstruction with *zero* error, scored against the samples it was drawn from, it
#: measured 0.1393 to 0.7513 depending only on how those samples were drawn — and exactly
#: 0.0 under Poisson-disk sampling, in both geometries. Its failure is not inherent: it is
#: what uneven sampling does to a nearest-sample distance, because uniform random sampling
#: leaves gaps well above the median nearest-neighbour spacing and the per-triangle maximum
#: lands in them. Evenly sampled, the headline fraction is meaningful again. The 0.2055 figure
#: recorded elsewhere in this module is specific to its sampling and does not transfer.
#:
#: The bands are *less* sensitive to this than the headline fraction, which is why they were
#: added, but the cube counterexample above shows they are not immune to it: they survive
#: uneven spacing at a roughly even *density*, and fail when the density itself is uneven
#: across the surface. Read either number as a property of the capture as much as of the
#: reconstruction.
#:
#: The real scan measured at 0.1012 is therefore not an anomaly sitting awkwardly on a
#: boundary: it is a scan of a solid object with modest real holes sitting near where the
#: floor falls, which is where it should sit.
#:
#: Nothing gates on this reading, and it stays that way while the floor moves this much.
#:
#: What moves it is now measured, and it is the one thing the caller chooses: the voxel
#: relative to the sample spacing. Sweeping that ratio on a fixed scene moved the floor from
#: 0.0467 invented area at a voxel equal to the median nearest-neighbour spacing down to
#: 0.0075 at three times it, and moved the zero-error `unsupported_area_fraction` from 0.8780
#: to 0.0055 with it. A tight voxel costs both ways at once: it inflates the headline fraction
#: on a correct reconstruction *and* blinds the reading to smaller fabrications. Prefer a
#: voxel at or above twice the median sample spacing. Mesh resolution, by contrast, does not
#: matter here: the floor showed no trend above sampling noise across a 32-fold range of
#: triangle-edge-to-voxel ratio.
#:
#: Measurements: `evidence/open3d/band-validity-sweep.json`,
#: `evidence/open3d/floor-across-geometry-and-sampling.json`,
#: `evidence/open3d/floor-vs-voxel-multiple.json`,
#: `evidence/open3d/floor-vs-resolution.json`, and the review lane's
#: `program/review/evidence/verify_602_boundary.py`.
FAR_BAND_AREA_SHARE = 0.1

#: One known leniency in how the supporting measurements were validated, recorded here so
#: it is not rediscovered: fabrication was checked as unsigned distance to the whole
#: reference surface, so a bridge cutting through an object's interior can register as
#: faithful when it passes near any other part of that surface. The bias is in this tool's
#: favour, and it is largest on concave geometry. A signed distance or a per-region
#: correspondence would be stricter; neither was implemented.


def _crop_justification(before: dict[str, Any], removed: int, mesh) -> dict[str, Any]:
    """Say how far past the support threshold the removed surface actually lay.

    This does not decide whether the removed surface was invented, and an earlier version
    of this function said it did. It reports which side of `FAR_BAND_AREA_SHARE` the removed
    area fell on, in distance from the nearest observation. Sparse coverage of correct
    geometry lands on the far side too; see that constant for the reproduced counterexample.

    The coverage pair cannot answer even this much. Coverage asks whether the observations
    are still explained, and surface removed from a *correct* reconstruction leaves coverage
    almost untouched because the remaining surface still passes under every sample. That
    is measured, not hypothetical: on a watertight mesh sampled at roughly one voxel
    spacing, cropping at factor 1.0 discarded 8.45 percent of vertices that were within
    one voxel of ground truth, and coverage moved by 6.6e-4.

    So a run can crop correct geometry with every published number looking healthy. What
    separates the cases is how far past the threshold the removed area lay, which is why
    this reads the distance bands rather than the headline fraction.
    """

    unsupported = float(before.get("unsupported_area_fraction") or 0.0)
    far = float(before.get("unsupported_area_beyond_3_voxels") or 0.0)
    share = (far / unsupported) if unsupported > 0 else 0.0
    if share >= FAR_BAND_AREA_SHARE:
        reads_as = "far from any observation"
        note = (
            f"{share:.1%} of the unsupported area lay more than three voxels from any "
            "sample, above the 10% mark that separates these two readings, so the crop "
            "removed surface that no observation lies near. Why it lies far out is not "
            "something this separates: thin coverage there and surface that does not "
            "belong there both read this way. Whether the surface is wrong is a separate "
            "question this cannot answer: a correct surface over a sparsely covered "
            "region reads the same way. Measured, on a cube that was exactly its own "
            "ground truth, sampled densely on five faces and sparsely on the sixth: "
            "0.4169 of its unsupported area sat past three voxels with nothing invented "
            "anywhere. Deciding fabrication needs a reference surface. Check whether "
            "coverage of the removed region was thin before concluding anything."
        )
    else:
        reads_as = "near-threshold surface"
        note = (
            "The unsupported area was concentrated within three voxels of a sample, "
            "which is where a correct surface reconstructed from samples of this "
            "spacing also falls. The crop may have removed correct geometry, and "
            "coverage cannot rule that out. Raise --support-distance-factor to 1.5 "
            "or 2.0, or pass 0, unless the tighter crop is wanted deliberately. "
            "This reading is not a clean bill of health either: it cannot see "
            "fabrication below the sensitivity floor, which is setup-dependent, so "
            "a small invented region reads exactly like none at all."
        )
    return {
        "unsupported_area_share_beyond_3_voxels": share,
        "removed_surface_reads_as": reads_as,
        "vertices_removed": removed,
        "vertex_fraction_removed": (
            removed / (removed + len(mesh.vertices))
            if removed + len(mesh.vertices) > 0
            else 0.0
        ),
        "note": note,
    }


def _support(o3d, mesh, cloud, voxel: float) -> dict[str, Any]:
    """Measure how much of this surface any observation actually supports.

    Poisson reconstruction closes a surface over an open scan, so a partial
    capture comes back wrapped in an extrapolated shell. That shell renders as
    smooth opaque geometry indistinguishable from observed structure, which is
    exactly how invented detail gets reviewed as sensor truth. Reporting it as a
    fraction of *area* rather than of vertices matters: the shell is dense and
    fine-grained, so a vertex count understates it.
    """

    import numpy as np

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    distances = _sample_distances(o3d, cloud, vertices)
    a, b, c = (
        vertices[triangles[:, 0]],
        vertices[triangles[:, 1]],
        vertices[triangles[:, 2]],
    )
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total = float(areas.sum())
    per_triangle = distances[triangles].max(axis=1)
    unsupported = float(areas[per_triangle > voxel].sum())

    def beyond(multiple: float) -> float:
        """Unsupported area past `multiple` voxels, as a fraction of all area."""

        if total <= 0:
            return 0.0
        return float(areas[per_triangle > voxel * multiple].sum() / total)

    return {
        "voxel_size": voxel,
        "unsupported_area_fraction": (unsupported / total) if total > 0 else 0.0,
        "unsupported_area": unsupported,
        # How far past the threshold the unsupported area actually lies. Without this
        # the headline fraction cannot distinguish invented surface from discretization,
        # and the two need opposite responses.
        #
        # A correct surface reconstructed from samples roughly one voxel apart puts a
        # fifth of its area past one voxel purely because a vertex interpolating between
        # two samples can sit slightly further than one voxel from the nearer of them.
        # Measured on a watertight mesh sampled uniformly: 0.2055 past one voxel, 0.0075
        # past 1.5, 0.000084 past 2, and nothing at all past 3 — while no vertex sat
        # further than one voxel from the ground-truth surface.
        #
        # An extrapolated Poisson shell does not fall off like that. On the demo scans:
        # 0.5273 past one voxel, still 0.3848 past three, reaching 18.1 voxels.
        #
        # So these two numbers, not the headline fraction, are what says which case a
        # run is in.
        "unsupported_area_beyond_1_5_voxels": beyond(1.5),
        "unsupported_area_beyond_3_voxels": beyond(3.0),
        "max_vertex_distance_to_sample": float(distances.max()),
        "median_vertex_distance_to_sample": float(np.median(distances)),
        "p95_vertex_distance_to_sample": float(np.percentile(distances, 95)),
    }


def _coverage(o3d, mesh, cloud, voxel: float) -> dict[str, Any]:
    """The other direction: do the observed samples still lie on this surface?

    Cropping unsupported area is only honest if it does not also remove surface
    that explains real observations, so the crop is measured against this.
    """

    import numpy as np

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    points = np.asarray(cloud.points).astype(np.float32)
    distances = scene.compute_distance(o3d.core.Tensor(points)).numpy()
    return {
        "samples": int(len(distances)),
        "fraction_within_voxel": float((distances < voxel).mean()),
        "fraction_within_two_voxels": float((distances < 2.0 * voxel).mean()),
        "sample_to_surface_rmse": float(np.sqrt((distances**2).mean())),
        "sample_to_surface_p95": float(np.percentile(distances, 95)),
    }


def _mesh_facts(o3d, mesh, path: Path) -> dict[str, Any]:
    extent = mesh.get_axis_aligned_bounding_box().get_extent()
    return {
        "vertex_count": len(mesh.vertices),
        "triangle_count": len(mesh.triangles),
        "surface_area": float(mesh.get_surface_area()),
        "is_edge_manifold": bool(mesh.is_edge_manifold()),
        "is_vertex_manifold": bool(mesh.is_vertex_manifold()),
        "is_watertight": bool(mesh.is_watertight()),
        "extent": [float(value) for value in extent],
        "sha256": _digest(path),
        "bytes": path.stat().st_size,
    }


def run_reconstruct(
    payload: dict[str, Any], output: Path, run_id: str
) -> dict[str, Any]:
    """Poisson surface reconstruction over the fused multiway cloud."""

    o3d = _open3d()
    import numpy as np

    request = ReconstructRequest.model_validate(payload["request"])
    fused_path = Path(payload["fused_path"])
    cloud = _read_cloud(o3d, fused_path)
    voxel = float(payload["voxel_size"])
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30)
    )
    cloud.orient_normals_consistent_tangent_plane(k=30)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud, depth=request.poisson_depth
    )
    if not mesh.has_triangles():
        raise Open3dError("Poisson reconstruction produced no triangles")
    density = np.asarray(densities)
    removed = 0
    if request.density_quantile > 0.0:
        threshold = float(np.quantile(density, request.density_quantile))
        mask = density < threshold
        removed = int(mask.sum())
        mesh.remove_vertices_by_mask(mask)
        if not mesh.has_triangles():
            raise Open3dError(
                "density cropping removed the entire surface; lower --density-quantile"
            )
    mesh.compute_vertex_normals()

    # Publish the pre-crop surface too. The crop below is a claim about which
    # geometry is observed, and a reviewer must be able to check that claim
    # against what Poisson actually returned.
    full_path = output / "mesh_uncropped.ply"
    if not o3d.io.write_triangle_mesh(str(full_path), mesh):
        raise Open3dError("Open3D could not write the uncropped mesh")
    before = _support(o3d, mesh, cloud, voxel)
    unsupported_removed = 0
    if request.support_distance_factor > 0.0:
        limit = voxel * request.support_distance_factor
        distances = _sample_distances(o3d, cloud, np.asarray(mesh.vertices))
        crop = distances > limit
        unsupported_removed = int(crop.sum())
        mesh.remove_vertices_by_mask(crop)
        if not mesh.has_triangles():
            raise Open3dError(
                "support cropping removed the entire surface; raise "
                "--support-distance-factor or pass 0 to disable it"
            )
        mesh.compute_vertex_normals()
    mesh_path = output / "mesh.ply"
    if not o3d.io.write_triangle_mesh(str(mesh_path), mesh):
        raise Open3dError("Open3D could not write the reconstructed mesh")
    return {
        "run_id": run_id,
        "poisson_depth": request.poisson_depth,
        "density_quantile": request.density_quantile,
        "support_distance_factor": request.support_distance_factor,
        "input_points": len(cloud.points),
        "low_density_vertices_removed": removed,
        "unsupported_vertices_removed": unsupported_removed,
        "mesh": _mesh_facts(o3d, mesh, mesh_path),
        "mesh_uncropped": _mesh_facts(o3d, mesh, full_path)
        if request.support_distance_factor <= 0.0
        else _uncropped_facts(o3d, full_path),
        # Both directions, before and after. The pair is the evidence: support
        # should rise sharply while coverage stays put, and a coverage drop means
        # the crop took surface that explained real observations.
        "support_before_crop": before,
        "support": _support(o3d, mesh, cloud, voxel),
        "coverage": _coverage(o3d, mesh, cloud, voxel),
        "crop_justification": _crop_justification(before, unsupported_removed, mesh),
        "cloud_extent": [
            float(value) for value in cloud.get_axis_aligned_bounding_box().get_extent()
        ],
    }


def _uncropped_facts(o3d, path: Path) -> dict[str, Any]:
    """Re-read the published uncropped mesh so its facts describe those bytes."""

    mesh = o3d.io.read_triangle_mesh(str(path))
    if not mesh.has_triangles():
        raise Open3dError("uncropped mesh artifact decoded without triangles")
    return _mesh_facts(o3d, mesh, path)


def _up_axis(o3d, cloud, voxel: float) -> dict[str, Any]:
    """Infer which way is up from the scan's largest plane, and say so.

    A 3D view whose camera rolls the room onto its side is unreadable, and these
    scans carry no axis convention we are entitled to assume. The largest planar
    segment in an indoor capture is the floor or a wall, so its normal is a
    measured basis for the camera's up vector. The inlier count and normal are
    returned and published, because this is an inference and a reviewer should be
    able to see it was one — and see when it was weak.
    """

    import numpy as np

    try:
        model, inliers = cloud.segment_plane(
            distance_threshold=voxel, ransac_n=3, num_iterations=1000
        )
    except RuntimeError:
        return {"source": "default", "up": [0.0, 0.0, 1.0], "plane_inliers": 0}
    normal = np.asarray(model[:3], dtype=float)
    norm = float(np.linalg.norm(normal))
    total = len(cloud.points)
    # A plane holding under a tenth of the scan is not a floor; do not steer the
    # camera by it.
    if norm == 0.0 or total == 0 or len(inliers) < 0.1 * total:
        return {
            "source": "default-weak-plane",
            "up": [0.0, 0.0, 1.0],
            "plane_inliers": int(len(inliers)),
            "plane_inlier_fraction": (len(inliers) / total) if total else 0.0,
        }
    normal = normal / norm
    points = np.asarray(cloud.points)
    plane_centre = points[np.asarray(inliers)].mean(axis=0)
    # Point away from the plane, towards where the scene actually is.
    if float(np.dot(points.mean(axis=0) - plane_centre, normal)) < 0.0:
        normal = -normal
    # Snap to the nearest world axis. The largest plane in a room can be a wall
    # rather than the floor, and an oblique up vector rolls the horizon, which is
    # the reviewability problem this is meant to fix. Snapping keeps the view
    # level and stops the camera implying a gravity direction the scan never
    # declared; the raw normal is published beside it so the inference is visible.
    axis = int(np.argmax(np.abs(normal)))
    snapped = np.zeros(3)
    snapped[axis] = 1.0 if normal[axis] > 0 else -1.0
    return {
        "source": "largest-plane-normal-snapped-to-axis",
        "up": [float(value) for value in snapped],
        "measured_plane_normal": [float(value) for value in normal],
        "angle_to_snapped_axis_degrees": float(
            np.degrees(np.arccos(np.clip(abs(normal[axis]), -1.0, 1.0)))
        ),
        "plane_inliers": int(len(inliers)),
        "plane_inlier_fraction": len(inliers) / total,
    }


CAMERA_FOV_DEGREES = 55.0
CAMERA_ELEVATION_DEGREES = 28.0
CAMERA_FRAME_MARGIN = 1.08
#: Rerun's 3D views are wider than tall; the horizontal half-angle scales by this.
#: Fallback pane shape for a camera fitted outside the shipped layout. Views in the
#: blueprint carry their own pane aspect; this covers a direct _camera call.
CAMERA_ASPECT = 4.0 / 3.0


def _camera(points, up: list[float], aspect: float = CAMERA_ASPECT) -> dict[str, Any]:
    """Place an elevated three-quarter eye far enough back to fit ``points``.

    The distance is solved rather than guessed: a hand-picked multiple of the
    scene span either clips the scan or strands it in the middle of an empty
    frame, and both make the result harder to review than it needs to be.

    Takes the points to frame rather than a cloud, because the caller decides
    what belongs in frame and that is not always one cloud. Fitting this against
    the fused cloud while a view drew the uncropped surface on top is what left
    the audit view clipped 23% past its own edge.

    ``aspect`` is the shape of the pane the result will be displayed in. Getting it
    wrong clips the sides while leaving vertical headroom, which is exactly what a
    captured viewer frame showed: fitted for 4:3, the scene filled the full width of
    a 1.13 pane and a 0.57 tab and used 44% of the tab's height.
    """

    import numpy as np

    points = np.asarray(points, dtype=float)
    centre = (points.min(axis=0) + points.max(axis=0)) / 2.0
    extent = points.max(axis=0) - points.min(axis=0)
    up_vector = np.asarray(up, dtype=float)
    up_vector = up_vector / float(np.linalg.norm(up_vector))

    horizontal = np.array([1.0, 1.0, 1.0])
    horizontal = horizontal - float(np.dot(horizontal, up_vector)) * up_vector
    if float(np.linalg.norm(horizontal)) < 1e-6:
        horizontal = np.array([1.0, 0.0, 0.0])
        horizontal = horizontal - float(np.dot(horizontal, up_vector)) * up_vector
    horizontal = horizontal / float(np.linalg.norm(horizontal))
    elevation = np.radians(CAMERA_ELEVATION_DEGREES)
    offset = np.cos(elevation) * horizontal + np.sin(elevation) * up_vector
    offset = offset / float(np.linalg.norm(offset))

    # Fit against the points themselves rather than the bounding box. A room scan
    # fills very little of its own box, so fitting the box strands the scan in the
    # middle of an empty frame -- the same reviewability problem as clipping it,
    # from the other direction. Solve for the distance at which the widest actual
    # sample sits just inside the frame, then iterate once because perspective
    # makes that distance depend on itself.
    forward = -offset
    right = np.cross(forward, up_vector)
    right = right / float(np.linalg.norm(right))
    true_up = np.cross(right, forward)
    relative = points - centre
    half_angle = np.tan(np.radians(CAMERA_FOV_DEGREES) / 2.0)
    lateral = relative @ right
    vertical = relative @ true_up
    along = relative @ forward
    distance = float(np.linalg.norm(extent))
    for _ in range(8):
        depth = distance + along
        depth = np.where(depth < 1e-6, 1e-6, depth)
        reach = np.abs(
            np.stack([lateral / half_angle / aspect, vertical / half_angle]) / depth
        ).max()
        if reach <= 1e-9:
            break
        distance = float(distance * reach * CAMERA_FRAME_MARGIN)
    return {
        "eye": [float(value) for value in centre + offset * distance],
        "look_target": [float(value) for value in centre],
        "up": [float(value) for value in up_vector],
        "distance": distance,
        "pane_aspect": float(aspect),
        "fov_degrees": CAMERA_FOV_DEGREES,
        "elevation_degrees": CAMERA_ELEVATION_DEGREES,
        "scene_span": float(np.linalg.norm(extent)),
        "scene_extent": [float(value) for value in extent],
    }


SCENE_VIEW = "Scene: observed scan and supported surface"
SCAN_VIEW = "Observed scan only"
SURFACE_VIEW = "Supported surface only"
REMOVED_VIEW = "Removed: unsupported surface"


#: Nominal viewer window shape, used only to turn the layout's column shares into
#: a per-pane aspect. Deliberately narrower than a wide monitor: a pane fitted for
#: a shape narrower than the real one keeps a margin, and one fitted for a wider
#: shape clips the sides. Only the first mistake is reviewable.
#:
#: 4:3 rather than 16:10 because horizontal is the binding direction in every narrow window
#: measured, so the frame extent scales as the inverse of the window aspect and the safe set
#: is everything at or above a single boundary. At 16:10 that boundary sat at 1.4815, which
#: put an ordinary 4:3 window 11.1% outside its own tabs. At 4:3 it sits at 1.2346 and a 4:3
#: window fits with 7.4% to spare. The price is emptier frames in wide windows: in 16:9 the
#: geometry reaches 76% of the scene pane and 69% of a tab rather than 93% of both. Vertical
#: takes over as the binding direction in the scene pane somewhere below 16:9, which is why
#: its extent stops falling at 0.7603 however wide the window gets.
#:
#: This is not window-independence and no static camera can offer it: a window narrower than
#: 1.2346, including one snapped to half of a wide monitor, still clips. Measured boundaries
#: are in `evidence/open3d/pane-fit-window-aspect-sweep.json`.
CAMERA_WINDOW_ASPECT = 4.0 / 3.0

#: Horizontal share of the window each pane of the blueprint receives, matching the
#: column_shares the layout below actually sets.
CAMERA_SCENE_SHARE = 2.0 / 3.0
CAMERA_TAB_SHARE = 1.0 / 3.0


class _View(NamedTuple):
    """What a view draws, and which geometry its camera therefore has to frame.

    ``aspect`` is the shape of the pane this view occupies, not a property of the
    geometry. A camera has to be fitted against the pane it will be displayed in:
    the same fit that frames a scene comfortably in a wide pane clips it in a
    narrow tab, which is what the layout's 2:1 column split produces.
    """

    geometry: tuple[str, ...]
    contents: tuple[str, ...]
    aspect: float


#: One table for every view: the geometry its camera frames and the entities it
#: draws. Both come from here so they cannot drift apart, and they did drift --
#: every view shared a camera fitted to the fused cloud, which frames three of
#: these four. The one it clipped was the audit view, whose entire purpose is
#: showing the surface the crop removed, which lies outside the cloud by
#: construction. A clipped audit view is worse than a missing one, because it
#: looks like the audit happened.
#:
#: The first geometry key defines the view, and a view whose defining geometry is
#: absent is left out of the blueprint entirely rather than rendered as an empty
#: tab: no removed surface means no audit view, and a cloud-only run has no
#: surface view. Later keys are context, framed when present so the defining
#: geometry is never shown floating free of the scan it came from.
VIEW_GEOMETRY: dict[str, _View] = {
    SCENE_VIEW: _View(
        ("fused", "mesh"),
        ("/world/fused", "/world/mesh", "/world/fragments/**"),
        CAMERA_WINDOW_ASPECT * CAMERA_SCENE_SHARE,
    ),
    SCAN_VIEW: _View(
        ("fused",),
        ("/world/fused", "/world/fragments/**"),
        CAMERA_WINDOW_ASPECT * CAMERA_TAB_SHARE,
    ),
    SURFACE_VIEW: _View(
        ("mesh",),
        ("/world/mesh",),
        CAMERA_WINDOW_ASPECT * CAMERA_TAB_SHARE,
    ),
    REMOVED_VIEW: _View(
        ("unsupported", "mesh", "fused"),
        ("/world/mesh", "/world/mesh_unsupported", "/world/fused"),
        CAMERA_WINDOW_ASPECT * CAMERA_TAB_SHARE,
    ),
}


def _view_cameras(geometry: dict[str, Any], up: list[float]) -> dict[str, Any]:
    """Fit one camera per view against that view's own geometry."""

    import numpy as np

    def present(key: str) -> bool:
        return geometry.get(key) is not None and len(geometry[key]) > 0

    cameras: dict[str, Any] = {}
    for view_name, spec in VIEW_GEOMETRY.items():
        if not present(spec.geometry[0]):
            continue
        cameras[view_name] = _camera(
            np.vstack(
                [
                    np.asarray(geometry[key], float)
                    for key in spec.geometry
                    if present(key)
                ]
            ),
            up,
            spec.aspect,
        )
    return cameras


def _mesh_colors(np, mesh, vertex_count: int):
    """Carry the mesh's own per-vertex colour through, or None to leave shading alone.

    Reconstruction inherits colour from the scans, and the offscreen renders of
    these same PLY bytes show it. The viewer was logging positions, triangles and
    normals only, so the surface arrived untinted there while the audit overlay
    that sits beside it is deliberately red -- which reads as a colour that means
    something rather than an absence of one.

    Nothing is invented. Without usable per-vertex colour this returns None and
    Rerun shades the surface as before, because a made-up tint on a reconstruction
    would imply measurement that did not happen.
    """

    if not mesh.has_vertex_colors():
        return None
    colors = np.asarray(mesh.vertex_colors)
    if colors.shape != (vertex_count, 3):
        return None
    # Open3D holds colour as float in 0..1; Rerun wants 8-bit channels.
    return (np.clip(colors, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def _blueprint(rr, rrb, cameras: dict[str, Any]):
    """A first view that shows the result, with the evidence a click away.

    The default Rerun layout gave every entity one auto view, so the closed
    surface covered the scan it was built from and the provenance document took a
    third of the window. This puts the scene first, keeps the scan and the surface
    on independent toggles, gives the removed surface its own tab instead of
    deleting it from the record, and moves provenance into a tab.

    Each view carries its own fitted camera. Sharing one costs nothing on the
    three views whose geometry sits inside the fused cloud, and clips the fourth.

    A view is built only when `cameras` has an entry for it, which happens only
    when its defining geometry was logged. That is what keeps a cloud-only run
    from offering an empty "supported surface" tab.
    """

    scene_camera = cameras[SCENE_VIEW]
    # A metre grid on the floor plane, so the view carries a readable scale.
    grid = rrb.LineGrid3D(visible=True, plane=rr.components.Plane3D(scene_camera["up"]))

    def view(name: str, **kwargs):
        camera = cameras[name]
        return rrb.Spatial3DView(
            name=name,
            origin="/world",
            contents=list(VIEW_GEOMETRY[name].contents),
            eye_controls=rrb.EyeControls3D(
                kind=rrb.Eye3DKind.Orbital,
                position=camera["eye"],
                look_target=camera["look_target"],
                eye_up=camera["up"],
            ),
            line_grid=grid,
            **kwargs,
        )

    tabs = [
        view(name)
        for name in (SCAN_VIEW, SURFACE_VIEW, REMOVED_VIEW)
        if name in cameras
    ]
    tabs.append(rrb.TextDocumentView(name="Provenance", origin="/provenance"))
    return rrb.Blueprint(
        rrb.Horizontal(
            view(
                SCENE_VIEW,
                overrides={
                    # Per-fragment clouds duplicate the fused cloud in space. Keep
                    # them in the recording and one click away, not stacked on top
                    # of it by default.
                    "/world/fragments": rrb.EntityBehavior(visible=False),
                },
            ),
            rrb.Tabs(*tabs),
            column_shares=[2, 1],
        ),
        collapse_panels=True,
    )


def run_visualize(payload: dict[str, Any], output: Path, run_id: str) -> dict[str, Any]:
    """Log the optimized fragments, fused cloud and surfaces into one recording."""

    o3d = _open3d()
    import numpy as np
    import rerun as rr
    import rerun.blueprint as rrb

    pose_graph = payload["pose_graph"]
    fragments = _fragment_paths(payload)
    fused_path = Path(payload["fused_path"])
    mesh_path = Path(payload["mesh_path"]) if payload.get("mesh_path") else None
    uncropped_path = (
        Path(payload["uncropped_mesh_path"])
        if payload.get("uncropped_mesh_path")
        else None
    )
    # The overlay has to redraw the crop that ran. A path being present says only
    # that an uncropped mesh was published, which reconstruct does unconditionally,
    # so it cannot stand in for "a crop happened at this threshold".
    support_factor = float(payload.get("support_distance_factor") or 0.0)
    vertices_cropped = int(payload.get("unsupported_vertices_removed") or 0)
    voxel = float(payload.get("voxel_size") or 0.05)
    recording_path = output / "point_cloud.rrd"
    recording = rr.RecordingStream("npa.open3d", recording_id=run_id)
    recording.save(str(recording_path))
    logged: list[dict[str, Any]] = []
    removed_triangles = 0
    try:
        for index, node in enumerate(pose_graph["nodes"]):
            key = node["fragment_id"]
            cloud = _read_cloud(o3d, fragments[key])
            cloud.transform(np.asarray(node["pose"], dtype=float))
            points = np.asarray(cloud.points)
            colors = np.asarray(cloud.colors) if cloud.has_colors() else None
            step = max(1, (len(points) + RRD_POINT_BUDGET - 1) // RRD_POINT_BUDGET)
            recording.set_time("fragment_index", sequence=index)
            recording.log(
                f"world/fragments/{key}",
                rr.Points3D(
                    positions=points[::step],
                    colors=None if colors is None else colors[::step],
                ),
            )
            logged.append(
                {
                    "fragment_id": key,
                    "source_points": int(len(points)),
                    "logged_points": int(len(points[::step])),
                    "decimation_step": step,
                }
            )
        fused = _read_cloud(o3d, fused_path)
        fused_points = np.asarray(fused.points)
        recording.log(
            "world/fused",
            rr.Points3D(
                positions=fused_points,
                colors=np.asarray(fused.colors) if fused.has_colors() else None,
                # Sized against the sampling scale. Default-sized points vanish
                # into an opaque surface drawn at the same depth, which is how a
                # combined view stops showing which geometry was measured.
                radii=voxel * 0.25,
            ),
            static=True,
        )
        mesh_summary: dict[str, Any] | None = None
        mesh_vertices = None
        removed_vertices = None
        if mesh_path is not None:
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            if not mesh.has_triangles():
                raise Open3dError("mesh artifact decoded without triangles")
            mesh.compute_vertex_normals()
            mesh_vertices = np.asarray(mesh.vertices)
            recording.log(
                "world/mesh",
                rr.Mesh3D(
                    vertex_positions=mesh_vertices,
                    triangle_indices=np.asarray(mesh.triangles),
                    vertex_normals=np.asarray(mesh.vertex_normals),
                    vertex_colors=_mesh_colors(np, mesh, len(mesh_vertices)),
                ),
                static=True,
            )
            mesh_summary = {
                "vertex_count": int(len(mesh.vertices)),
                "triangle_count": int(len(mesh.triangles)),
            }
            if (
                uncropped_path is not None
                and support_factor > 0.0
                and vertices_cropped > 0
            ):
                removed_triangles, removed_vertices = _log_removed_surface(
                    o3d,
                    rr,
                    recording,
                    uncropped_path,
                    fused,
                    voxel * support_factor,
                )
        up = _up_axis(o3d, fused, voxel)
        cameras = _view_cameras(
            {
                "fused": fused_points,
                "mesh": mesh_vertices,
                "unsupported": removed_vertices,
            },
            up["up"],
        )
        camera = cameras[SCENE_VIEW]
        recording.log(
            "provenance",
            rr.TextDocument(
                json.dumps(
                    {
                        "producer": "npa.workbench.open3d",
                        "engine": "open3d",
                        "source_version": SOURCE_VERSION,
                        "run_id": run_id,
                        "fused_sha256": pose_graph["fused_sha256"],
                        "logged_fragments": logged,
                        "camera": camera,
                        "view_cameras": cameras,
                        "up_axis_inference": up,
                        "unsupported_triangles_shown": removed_triangles,
                        "limitations": (
                            "Fragments are shown in their optimized pose-graph "
                            "poses; per-entity decimation is recorded above. The "
                            "up axis is inferred from the largest planar segment, "
                            "not from a declared convention. No ground-truth pose, "
                            "scale or semantic claim; a surface that passed the "
                            "support check is still not certified for collision."
                        ),
                    },
                    sort_keys=True,
                )
            ),
            static=True,
        )
        recording.send_blueprint(_blueprint(rr, rrb, cameras))
    finally:
        recording.flush()
        del recording
    if not recording_path.is_file() or recording_path.stat().st_size == 0:
        raise Open3dError("RRD writer produced no bytes")
    return {
        "run_id": run_id,
        "logged_fragments": logged,
        "fused_points": int(len(fused_points)),
        "mesh": mesh_summary,
        "unsupported_triangles_shown": removed_triangles,
        "camera": camera,
        "view_cameras": cameras,
        "up_axis_inference": up,
        "sha256": _digest(recording_path),
        "bytes": recording_path.stat().st_size,
    }


def _log_removed_surface(o3d, rr, recording, uncropped_path: Path, cloud, limit: float):
    """Log the surface the support crop removed, so the crop is auditable.

    Showing what was taken out is the difference between a defensible cleanup and
    a flattering camera angle. It is off by default and has its own tab.

    ``limit`` is the distance reconstruct actually cropped at, ``voxel`` times the
    support factor, and not the bare voxel. Thresholding here at anything else
    paints an overlay that contradicts the mesh beside it: at factor 2.0 the
    surface between one and two voxels of a sample is kept in ``mesh.ply`` yet
    would be drawn as removed. Every published number would still be right, which
    is exactly what makes that failure hard to catch from evidence alone.
    """

    import numpy as np

    full = o3d.io.read_triangle_mesh(str(uncropped_path))
    if not full.has_triangles():
        return 0, None
    vertices = np.asarray(full.vertices)
    distances = _sample_distances(o3d, cloud, vertices)
    keep = np.asarray(full.triangles)[
        distances[np.asarray(full.triangles)].max(axis=1) > limit
    ]
    if len(keep) == 0:
        return 0, None
    removed = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(keep)
    )
    removed.remove_unreferenced_vertices()
    removed.compute_vertex_normals()
    removed_vertices = np.asarray(removed.vertices)
    recording.log(
        "world/mesh_unsupported",
        rr.Mesh3D(
            vertex_positions=removed_vertices,
            triangle_indices=np.asarray(removed.triangles),
            vertex_normals=np.asarray(removed.vertex_normals),
            # Red, so an unsupported surface never reads as observed structure.
            vertex_colors=np.tile([220, 60, 60], (len(removed_vertices), 1)),
        ),
        static=True,
    )
    return int(len(removed.triangles)), removed_vertices


KINDS = {
    "stage-demo": run_stage_demo,
    "register": run_register,
    "multiway": run_multiway,
    "reconstruct": run_reconstruct,
    "visualize": run_visualize,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=sorted(KINDS))
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    payload = json.loads(Path(args.input).read_text())
    result = KINDS[args.kind](payload, output, args.run_id)
    (output / "runner.json").write_bytes(canonical(result))
    return 0


if __name__ == "__main__":  # pragma: no cover - subprocess entrypoint
    raise SystemExit(main())
