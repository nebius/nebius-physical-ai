"""Did the multiway optimization do anything, and could the record tell you?

`run_multiway` builds a full pairwise pose graph and calls `global_optimization`,
which is the step that distinguishes multiway registration from chaining pairwise
transforms. Two things were unmeasured.

First, whether it corrects anything on the data the pipeline actually runs on. On the
upstream demo fragments it moves the worst pose by 4.7e-06 across a scene some metres
across, which is nothing a consumer could act on; through the product's own
`run_multiway` on the synthetic fragments its smoke uses, the correction is exactly
0.0. The pairwise estimates are already mutually consistent, so the odometry chain is
the optimum it starts from. That is correct behaviour and it means the capability was
only ever exercised by cases in which it cannot fail -- the same trap the up-axis
control fell into, where the control's answer coincides with the trivial answer.

The correction is reported as a fraction of the scene's bounding-box diagonal, because
4.7e-06 is only small relative to something.

Second, whether a reader could tell. `global_optimization` prunes loop-closure edges
below `edge_prune_threshold` in place, and the record serializes the surviving list,
so a pruned edge leaves no trace. There is no figure for the pose correction either.
A run whose optimization did real work and a run whose optimization did nothing
publish the same shape.

The control that makes the zero readable is a two-node graph whose node pose is
deliberately 0.5 off with an edge asserting identity: the optimizer moves it back to
0.0, so a zero elsewhere is a measurement rather than a broken probe.

Run inside the Open3D image:
    docker run --rm -v "$EVIDENCE_DIR:/data" --entrypoint python "$IMAGE" \
        /data/harness/pose_graph_optimization.py
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d

OUT = Path(os.environ.get("OPEN3D_OUT", "/data/pose-graph-optimization.json"))
VOXEL = 0.05
REPEATS = 5

reg = o3d.pipelines.registration


def _edge_counts(submitted: int, survived: int) -> dict:
    """All three edge counts, in every case, from one place.

    Written as a helper after review found the one case that omitted `edges_pruned`
    was also the one case whose figure the PR body got wrong -- it said "3 of 3
    survived" where 3 were submitted and 2 survived, which is precisely the mistake
    this record is about. A case that computes only two of the three numbers leaves
    the third to prose.
    """

    return {
        "edges_submitted": submitted,
        "edges_that_survived_pruning": survived,
        "edges_pruned": submitted - survived,
    }


def _optimize(graph, voxel: float = VOXEL) -> dict:
    """Run the product's optimization call and report what it changed."""

    before = [np.asarray(node.pose).copy() for node in graph.nodes]
    submitted = len(graph.edges)
    uncertain = sum(1 for edge in graph.edges if edge.uncertain)
    reg.global_optimization(
        graph,
        reg.GlobalOptimizationLevenbergMarquardt(),
        reg.GlobalOptimizationConvergenceCriteria(),
        reg.GlobalOptimizationOption(
            max_correspondence_distance=voxel * 1.5,
            edge_prune_threshold=0.25,
            reference_node=0,
        ),
    )
    after = [np.asarray(node.pose).copy() for node in graph.nodes]
    deltas = [float(np.abs(a - b).max()) for a, b in zip(after, before, strict=True)]
    return {
        **_edge_counts(submitted, len(graph.edges)),
        "edges_submitted_as_loop_closures": uncertain,
        "max_pose_correction_per_node": [round(value, 8) for value in deltas],
        "max_pose_correction": round(max(deltas), 8),
    }


def probe_can_detect_a_correction() -> dict:
    """The control: a graph that must move, so a zero elsewhere means something."""

    graph = reg.PoseGraph()
    graph.nodes.append(reg.PoseGraphNode(np.identity(4)))
    wrong = np.identity(4)
    wrong[0, 3] = 0.5
    graph.nodes.append(reg.PoseGraphNode(wrong))
    graph.edges.append(
        reg.PoseGraphEdge(0, 1, np.identity(4), np.identity(6), uncertain=False)
    )
    measured = _optimize(graph)
    corrected = float(np.asarray(graph.nodes[1].pose)[0, 3])
    return {
        "construction": (
            "two nodes, the second deliberately 0.5 off in x, with an edge asserting "
            "the relative transform is identity"
        ),
        "node_offset_before": 0.5,
        "node_offset_after": round(corrected, 8),
        **measured,
        "so": (
            "the optimizer moves a pose when the graph disagrees with its edges, and "
            "this probe sees it, so a zero correction measured below is the "
            "optimizer's answer rather than a probe that cannot read poses"
        ),
    }


def _pair(source, target, voxel: float = VOXEL):
    prepared = []
    for cloud in (source, target):
        down = cloud.voxel_down_sample(voxel)
        down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30)
        )
        prepared.append(down)
    icp = reg.registration_icp(
        prepared[0],
        prepared[1],
        voxel * 1.5,
        np.identity(4),
        reg.TransformationEstimationPointToPlane(),
    )
    information = reg.get_information_matrix_from_point_clouds(
        prepared[0], prepared[1], voxel * 1.5, icp.transformation
    )
    return icp.transformation, information, float(icp.fitness)


def _graph_over(clouds) -> object:
    """The same graph `run_multiway` builds: full pairwise, consecutive = odometry."""

    graph = reg.PoseGraph()
    odometry = np.identity(4)
    graph.nodes.append(reg.PoseGraphNode(odometry))
    for source in range(len(clouds)):
        for target in range(source + 1, len(clouds)):
            transformation, information, _fitness = _pair(
                clouds[source], clouds[target]
            )
            consecutive = target == source + 1
            if consecutive:
                odometry = transformation @ odometry
                graph.nodes.append(reg.PoseGraphNode(np.linalg.inv(odometry)))
            graph.edges.append(
                reg.PoseGraphEdge(
                    source,
                    target,
                    transformation,
                    information,
                    uncertain=not consecutive,
                )
            )
    return graph


def _demo_clouds():
    """The upstream DemoICPPointClouds fragments the pipeline's demo stage fetches."""

    data = o3d.data.DemoICPPointClouds()
    return [o3d.io.read_point_cloud(path) for path in data.paths]


def _extent(clouds) -> float:
    """Bounding-box diagonal of the whole scene, so a correction has a scale."""

    merged = o3d.geometry.PointCloud()
    for cloud in clouds:
        merged += cloud
    box = merged.get_axis_aligned_bounding_box()
    return float(np.linalg.norm(np.asarray(box.get_extent())))


def on_the_demo_fragments() -> dict:
    """What the optimization does on the data every accepted record is built from."""

    clouds = _demo_clouds()
    runs = [_optimize(_graph_over(clouds)) for _ in range(REPEATS)]
    corrections = [run["max_pose_correction"] for run in runs]
    extent = _extent(clouds)
    return {
        "source": "o3d.data.DemoICPPointClouds, the pipeline's own demo fragments",
        "nodes": len(clouds),
        "repeats": len(runs),
        "scene_extent": round(extent, 4),
        "max_pose_correction_over_repeats": [
            round(min(corrections), 10),
            round(max(corrections), 10),
        ],
        "largest_correction_as_a_share_of_scene_extent": round(
            max(corrections) / extent, 12
        ),
        **_edge_counts(
            runs[0]["edges_submitted"], runs[0]["edges_that_survived_pruning"]
        ),
        "why": (
            "the pairwise estimates are mutually consistent, so the odometry chain "
            "already is the optimum and there is nothing for the optimizer to "
            "distribute. This is correct behaviour and it is also why the capability "
            "was never exercised by a case where it could fail"
        ),
    }


def _run_multiway_over(paths: dict[str, str], output: Path) -> tuple[dict, dict]:
    """Call the shipped `run_multiway`, snapshotting what the optimizer was handed."""

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "npa" / "src"))
    from npa.smoke.test_open3d_functional import VOXEL_SIZE
    from npa.workbench.open3d.runner import run_multiway
    from npa.workbench.open3d.schemas import RegistrationManifest

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
    snapshot: dict = {}
    real = reg.global_optimization

    def spy(graph, method, criteria, option):
        before = [np.asarray(node.pose).copy() for node in graph.nodes]
        snapshot["edges_submitted"] = len(graph.edges)
        real(graph, method, criteria, option)
        after = [np.asarray(node.pose).copy() for node in graph.nodes]
        snapshot["correction"] = max(
            float(np.abs(a - b).max()) for a, b in zip(after, before, strict=True)
        )
        snapshot["edges_that_survived_pruning"] = len(graph.edges)

    reg.global_optimization = spy
    try:
        report = run_multiway(
            {"manifest": manifest, "fragments": paths}, output, "evidence"
        )
    finally:
        reg.global_optimization = real
    return report, snapshot


def through_the_product_on_a_loop() -> dict:
    """The loop again, through `run_multiway` at the manifest's own parameters.

    Separate from the standalone loop above because it does not agree with it: the
    product's voxel size and ICP distance factor prune three of the six edges where
    the standalone measurement at this file's own constants prunes one. The figure a
    test may assert on is this one, so both are recorded rather than averaged.
    """

    with tempfile.TemporaryDirectory(prefix="npa-posegraph-loop-") as directory:
        root = Path(directory)
        paths = {}
        for index, start in enumerate((0.0, 80.0, 160.0, 240.0)):
            cloud = _arc(start, seed=0)
            path = root / f"loop_{index}.ply"
            o3d.io.write_point_cloud(str(path), cloud)
            paths[f"view_{index}"] = str(path)
        output = root / "multiway"
        output.mkdir()
        report, snapshot = _run_multiway_over(paths, output)

    optimization = report["pose_graph"]["optimization"]
    return {
        "entry_point": "npa.workbench.open3d.runner.run_multiway",
        "construction": "the same four 140-degree views, written as PLY fragments",
        **_edge_counts(
            snapshot["edges_submitted"], snapshot["edges_that_survived_pruning"]
        ),
        "max_pose_correction": snapshot["correction"],
        "correction_is_nonzero": snapshot["correction"] > 0.0,
        "published_block_matches_the_snapshot": (
            optimization["edges_submitted"] == snapshot["edges_submitted"]
            and optimization["edges_pruned"]
            == snapshot["edges_submitted"] - snapshot["edges_that_survived_pruning"]
            and optimization["max_pose_correction"] == snapshot["correction"]
        ),
        "why_this_case_exists": (
            "on consistent fragments the correction is 0.0 and nothing is prunable, so "
            "reading the poses after optimization rather than before, and publishing "
            "the surviving edge count as the submitted one, both produce identical "
            "output. This is the case where those two mutations are observable"
        ),
    }


def through_the_product_pipeline() -> dict:
    """The same question asked of `run_multiway`, on the fragments its smoke uses.

    Measured separately from the upstream fragments above because it is a different
    input and a different voxel size, and it gives a different answer: exactly zero
    rather than 4.7e-06. Conflating the two would have published one figure for two
    measurements.
    """

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "npa" / "src"))
    from npa.smoke.test_open3d_functional import VOXEL_SIZE, _write_clouds
    from npa.workbench.open3d.runner import run_multiway
    from npa.workbench.open3d.schemas import RegistrationManifest

    snapshot: dict = {}
    real = reg.global_optimization

    def spy(graph, method, criteria, option):
        before = [np.asarray(node.pose).copy() for node in graph.nodes]
        submitted = len(graph.edges)
        real(graph, method, criteria, option)
        after = [np.asarray(node.pose).copy() for node in graph.nodes]
        snapshot["correction"] = max(
            float(np.abs(a - b).max()) for a, b in zip(after, before, strict=True)
        )
        snapshot["edges_submitted"] = submitted
        snapshot["edges_that_survived_pruning"] = len(graph.edges)

    with tempfile.TemporaryDirectory(prefix="npa-posegraph-") as directory:
        root = Path(directory)
        paths, _poses = _write_clouds(o3d, root)
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
        graph_dir = root / "multiway"
        graph_dir.mkdir()
        reg.global_optimization = spy
        try:
            report = run_multiway(
                {"manifest": manifest, "fragments": paths}, graph_dir, "evidence"
            )
        finally:
            reg.global_optimization = real

    published = report["pose_graph"]
    return {
        "entry_point": "npa.workbench.open3d.runner.run_multiway",
        "fragments": "the three synthetic clouds the functional smoke writes",
        "max_pose_correction": snapshot["correction"],
        "correction_is_exactly_zero": snapshot["correction"] == 0.0,
        **_edge_counts(
            snapshot["edges_submitted"], snapshot["edges_that_survived_pruning"]
        ),
        "edges_the_record_serialized": len(published["edges"]),
        "so": (
            "the optimizer provably moved nothing, the smoke passed, and its success "
            "line still read that it had optimized a three-node pose graph"
        ),
    }


def _arc(start_degrees: float, seed: int, span: float = 140.0, points: int = 9000):
    """A partial view of a cylindrical room, as one sensor position would see it."""

    rng = np.random.default_rng(seed)
    angles = np.radians(rng.uniform(start_degrees, start_degrees + span, points))
    radius = 2.0
    cloud = np.column_stack(
        [
            radius * np.cos(angles),
            radius * np.sin(angles),
            rng.uniform(-1.0, 1.0, points),
        ]
    )
    cloud = cloud + rng.normal(0.0, 0.004, cloud.shape)
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud)).voxel_down_sample(
        VOXEL
    )


def on_a_loop_with_overlap() -> dict:
    """Four overlapping views closing a loop: does pruning happen, and is it visible?"""

    runs = []
    for seed in range(REPEATS):
        clouds = [_arc(start, seed) for start in (0.0, 80.0, 160.0, 240.0)]
        runs.append(_optimize(_graph_over(clouds)))
    pruned = [run["edges_pruned"] for run in runs]
    corrections = [run["max_pose_correction"] for run in runs]
    return {
        "construction": (
            "four 140-degree views of a cylindrical room at 80-degree spacing, so "
            "consecutive views overlap and the last closes back toward the first"
        ),
        "nodes": 4,
        "repeats": len(runs),
        **_edge_counts(
            runs[0]["edges_submitted"], runs[0]["edges_that_survived_pruning"]
        ),
        "edges_pruned_over_repeats": [min(pruned), max(pruned)],
        "at_least_one_edge_was_pruned": min(pruned) > 0,
        "max_pose_correction_over_repeats": [
            round(min(corrections), 8),
            round(max(corrections), 8),
        ],
        "so": (
            "an edge the optimizer rejected leaves no trace in a record that "
            "serializes only the survivors, and the correction is still zero, so "
            "neither of the two things the optimization does is readable afterwards"
        ),
    }


def what_the_record_omitted() -> dict:
    """Which of these measurements `run_multiway` published before this change."""

    return {
        "pose_graph_keys_before": [
            "nodes",
            "edges",
            "reference_node",
            "edge_prune_threshold",
            "fused_points",
            "fused_points_before_downsample",
            "fused_sha256",
        ],
        "edges_submitted_was_published": False,
        "edges_pruned_was_published": False,
        "pose_correction_was_published": False,
        "a_keyword_scan_said_otherwise_and_was_wrong": (
            "scanning the keys for 'before' matched fused_points_before_downsample, "
            "which is about voxel downsampling and not about edges. Recorded because "
            "it is the kind of false positive that makes an audit look complete"
        ),
        "consequence": (
            "a run whose optimization corrected real drift and a run whose "
            "optimization did nothing serialize the same shape, and the smoke's "
            "success line says 'optimized a 3-node pose graph' either way"
        ),
    }


def build() -> dict:
    return {
        "question": (
            "does global_optimization change anything on the pipeline's own data, "
            "and could a reader of the record tell either way?"
        ),
        "open3d_version": o3d.__version__,
        "probe_control": probe_can_detect_a_correction(),
        "demo_fragments": on_the_demo_fragments(),
        "through_the_product_pipeline": through_the_product_pipeline(),
        "loop_with_overlap": on_a_loop_with_overlap(),
        "through_the_product_on_a_loop": through_the_product_on_a_loop(),
        "what_the_record_omitted": what_the_record_omitted(),
        "provenance": {
            "harness": "harness/pose_graph_optimization.py",
            "harness_sha256": "recomputed by main() after this dict is built",
            "path_is_relative_to": "this evidence directory",
            "open3d_version": o3d.__version__,
        },
        "what_this_does_not_claim": {
            "not_that_optimization_is_broken": (
                "zero correction on consistent input is the right answer. The defect "
                "is that the record cannot distinguish it from a failure, and that no "
                "test exercised a case where the optimizer has work to do"
            ),
            "not_that_pruning_is_wrong": (
                "rejecting a loop closure that disagrees is the mechanism working. "
                "Dropping it silently is the part a reviewer cannot audit"
            ),
            "not_a_drift_correction_benchmark": (
                "these constructions establish that the measurements are missing and "
                "that the optimizer reads as a no-op on the shipped data. How much "
                "drift multiway recovers on real captures is a separate question"
            ),
        },
    }


def main() -> int:
    report = build()
    # Self-hash last, so the digest covers the file that produced the report rather
    # than a placeholder. A digest nothing recomputes is decoration.
    report["provenance"]["harness_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    control = report["probe_control"]
    if control["max_pose_correction"] <= 0.0:
        quarantine = OUT.with_suffix(".invalid.json")
        quarantine.write_text(json.dumps(report, indent=2) + "\n")
        print("The probe could not detect a pose correction it induced itself.")
        print(f"Quarantined at {quarantine}; left {OUT} as it was")
        return 1
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    demo = report["demo_fragments"]
    loop = report["loop_with_overlap"]
    product = report["through_the_product_pipeline"]
    print(
        f"Control corrected {control['max_pose_correction']} of an induced 0.5, so the "
        "probe reads poses."
    )
    print(
        f"Demo fragments: {demo['nodes']} nodes, correction "
        f"{demo['max_pose_correction_over_repeats']} over {demo['repeats']} repeats."
    )
    print(
        f"Loop: {loop['edges_submitted']} edges submitted, "
        f"{loop['edges_pruned_over_repeats']} pruned, correction "
        f"{loop['max_pose_correction_over_repeats']}."
    )
    print(
        f"Through run_multiway: correction {product['max_pose_correction']}, "
        f"{product['edges_submitted']} edges submitted and "
        f"{product['edges_that_survived_pruning']} serialized."
    )
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
