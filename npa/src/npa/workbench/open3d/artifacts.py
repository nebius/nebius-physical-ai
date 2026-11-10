"""Validate factual Open3D journals and derive reviewable Rerun recordings.

Nothing here imports `open3d`, so `validate` runs wherever the CLI runs — the
published journal is self-describing, and a reviewer should not need the native
library to check it. The real geometry work lives in `runner.py`.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .schemas import SOURCE_VERSION

#: Registration result schema this module reads and `runner.py` writes.
RESULT_SCHEMA = "npa.open3d.result.v1"
#: Journal filename holding one row per registered fragment pair.
PAIRS_JOURNAL = "pairs.jsonl"
#: Absolute tolerance for "this 3x3 block is a rotation".
ROTATION_TOLERANCE = 1e-6
#: Prefix Rerun reserves for itself. Not a guess: `rerun/_log.py` documents that "paths
#: prefixed with `__` are considered reserved for use by the Rerun SDK", and the SDK
#: exports `RECORDING_PROPERTIES_PATH = "/__properties"`, which it writes into every
#: recording whether or not the caller logged anything. A recording holding only these
#: decodes fine and carries nothing anyone can look at.
RERUN_INTERNAL = "/__"


class Open3dError(RuntimeError):
    """An Open3D operation failed without a synthetic fallback."""


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


#: Distinguishes "this record predates the field" from "this record published the
#: field as nothing". Only the first is exempt from a check; keying a
#: backward-compatibility exemption on `None` makes it a switch a writer can throw.
_ABSENT = object()


def validate_rigid_transform(value: Any, *, what: str) -> None:
    """Reject anything that is not a real 4x4 rigid body transform.

    A registration stage that silently returns a scaled, sheared or non-finite
    matrix still writes a plausible-looking 4x4 array, and every downstream
    consumer (fusion, pose graph, RRD) would carry the damage forward.
    """

    import numpy as np

    matrix = np.asarray(value, dtype=object)
    if matrix.shape != (4, 4):
        raise Open3dError(f"{what} must be a 4x4 matrix")
    if not all(_finite_number(cell) for row in value for cell in row):
        raise Open3dError(f"{what} must contain only finite numbers")
    array = np.asarray(value, dtype=float)
    if not np.allclose(array[3], [0.0, 0.0, 0.0, 1.0], atol=ROTATION_TOLERANCE):
        raise Open3dError(f"{what} bottom row must be [0, 0, 0, 1]")
    rotation = array[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5
    ):
        raise Open3dError(f"{what} rotation block is not a proper rotation")


def validate_registration_result(value: Any, *, what: str) -> None:
    """Check one Open3D `RegistrationResult` summary as numbers, not as JSON."""

    if not isinstance(value, dict) or set(value) != {
        "fitness",
        "inlier_rmse",
        "correspondence_count",
    }:
        raise Open3dError(
            f"{what} must report fitness, inlier_rmse and correspondences"
        )
    fitness = value["fitness"]
    rmse = value["inlier_rmse"]
    count = value["correspondence_count"]
    if not _finite_number(fitness) or not 0.0 <= float(fitness) <= 1.0:
        raise Open3dError(f"{what} fitness must be a fraction in [0, 1]")
    if not _finite_number(rmse) or float(rmse) < 0.0:
        raise Open3dError(f"{what} inlier_rmse must be a nonnegative distance")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise Open3dError(f"{what} must have found at least one correspondence")
    # Open3D defines fitness as inlier correspondences over source points, so a
    # positive count and a zero fitness cannot both be true.
    if float(fitness) == 0.0:
        raise Open3dError(f"{what} reports correspondences with zero fitness")


def validate_pair(row: Any) -> None:
    """Validate one row of the registration journal."""

    required = {
        "pair_id",
        "source_id",
        "target_id",
        "source_points",
        "target_points",
        "source_downsampled_points",
        "target_downsampled_points",
        "fpfh_dimension",
        "global_registration",
        "icp",
        "transformation",
        "aligned_points",
        "aligned_sha256",
        "wall_seconds",
    }
    if not isinstance(row, dict) or not required <= set(row):
        raise Open3dError("registration journal row is missing required fields")
    for key in ("source_points", "target_points"):
        if isinstance(row[key], bool) or not isinstance(row[key], int) or row[key] <= 0:
            raise Open3dError(f"{key} must be a positive point count")
    for key in ("source_downsampled_points", "target_downsampled_points"):
        full = row[key.replace("_downsampled", "")]
        if (
            isinstance(row[key], bool)
            or not isinstance(row[key], int)
            or not 0 < row[key] <= full
        ):
            raise Open3dError(f"{key} must be positive and no larger than {full}")
    # FPFH is a 33-bin histogram in Open3D; a different width means the feature
    # is not the one the RANSAC matcher was validated against.
    if row["fpfh_dimension"] != 33:
        raise Open3dError("FPFH features must be 33-dimensional")
    validate_registration_result(
        row["global_registration"], what="global registration result"
    )
    validate_registration_result(row["icp"], what="ICP result")
    validate_rigid_transform(row["transformation"], what="pair transformation")
    if (
        isinstance(row["aligned_points"], bool)
        or not isinstance(row["aligned_points"], int)
        or row["aligned_points"] != row["source_points"]
    ):
        raise Open3dError("aligned cloud must retain every source point")
    if not isinstance(row["aligned_sha256"], str) or len(row["aligned_sha256"]) != 64:
        raise Open3dError("aligned cloud must carry a sha256 digest")
    if not _finite_number(row["wall_seconds"]) or row["wall_seconds"] < 0:
        raise Open3dError("wall_seconds must be a nonnegative duration")
    _validate_agreement(row)


_AGREEMENT_FIELDS = (
    "seeds",
    "worst_pairwise_displacement",
    "worst_pairwise_displacement_voxels",
    "worst_disagreeing_seeds",
    "fitness_spread",
    "inlier_rmse_spread",
    "fitness_by_seed",
    "inlier_rmse_by_seed",
    "agreement_is_not_correctness",
)


def _is_seed(value: Any) -> bool:
    """An Open3D seed is an int. `bool` is an int in Python and is not a seed."""

    return isinstance(value, int) and not isinstance(value, bool)


def _validate_failed_seeds(failed: Any) -> None:
    """Seeds whose diagnostic re-run raised, recorded rather than dropped.

    Optional, because a row from before this existed does not carry it and a run
    where nothing failed has nothing to record.
    """

    if failed is None:
        return
    if not isinstance(failed, list) or not failed:
        raise Open3dError("failed_seeds must be a non-empty list when present")
    for entry in failed:
        if not isinstance(entry, dict) or not _is_seed(entry.get("seed")):
            raise Open3dError("each failed seed must name the integer seed it was")
        if not isinstance(entry.get("error"), str) or not entry["error"]:
            raise Open3dError("each failed seed must carry why it failed")


def _validate_agreement(row: Any) -> None:
    """Check the multi-seed agreement block against the attempts it summarizes.

    Opt-in, so a row without it is a row from a manifest that did not ask for it and
    is left alone. A row that carries the key with nothing in it is a different thing
    and is an error: `None` would otherwise be indistinguishable from not having asked.

    What this can check is internal consistency. It cannot check that the alignment is
    correct -- nothing in the report can, which is the finding this block exists to
    publish rather than a gap in it.
    """

    if "agreement" not in row:
        return
    block = row["agreement"]
    if not isinstance(block, dict):
        raise Open3dError("registration agreement block is missing measurements")

    # A requested measurement that could not be taken. Fewer than two seeds
    # registered successfully, so there is no legal agreement to summarize -- and
    # omitting the key would be indistinguishable from never having asked for it.
    if block.get("measured") is False:
        if not isinstance(block.get("why_not"), str) or not block["why_not"]:
            raise Open3dError(
                "an agreement that was not measured has to say why it was not"
            )
        _validate_failed_seeds(block.get("failed_seeds"))
        return

    if not set(_AGREEMENT_FIELDS) <= set(block):
        raise Open3dError("registration agreement block is missing measurements")
    seeds = block["seeds"]
    if not isinstance(seeds, list) or len(seeds) < 2:
        raise Open3dError("agreement needs at least two seeds to compare")
    # F7: `set()` on list contents without a type guard. A published journal row
    # carrying `"seeds": [[1], [2]]` raised `TypeError: unhashable type` out of
    # `read_journal`, which the CLI does not catch -- the same hole the `float(None)`
    # guard below already closes for the counterexample path.
    if not all(_is_seed(one) for one in seeds):
        raise Open3dError("agreement seeds must be integers")
    if len(set(seeds)) != len(seeds):
        raise Open3dError("agreement seeds must be distinct, or they repeat one run")
    _validate_failed_seeds(block.get("failed_seeds"))
    for key in ("fitness_by_seed", "inlier_rmse_by_seed"):
        values = block[key]
        if not isinstance(values, list) or len(values) != len(seeds):
            raise Open3dError(f"{key} must carry one value per seed")
        if not all(_finite_number(value) for value in values):
            raise Open3dError(f"{key} must be finite numbers")
    for key, series in (
        ("fitness_spread", "fitness_by_seed"),
        ("inlier_rmse_spread", "inlier_rmse_by_seed"),
    ):
        values = [float(value) for value in block[series]]
        if not _finite_number(block[key]) or float(block[key]) < 0:
            raise Open3dError(f"{key} must be a nonnegative spread")
        if abs(float(block[key]) - (max(values) - min(values))) > 1e-12:
            raise Open3dError(f"{key} does not match the per-seed values beside it")
    worst = block["worst_pairwise_displacement"]
    if not _finite_number(worst) or float(worst) < 0:
        raise Open3dError("worst_pairwise_displacement must be a nonnegative distance")
    pair = block["worst_disagreeing_seeds"]
    if float(worst) > 0:
        if not isinstance(pair, list) or len(pair) != 2:
            raise Open3dError("a nonzero disagreement must name the two seeds it is")
        if not all(_is_seed(one) for one in pair):
            raise Open3dError("worst_disagreeing_seeds must name integer seeds")
        if not set(pair) <= set(seeds):
            raise Open3dError("worst_disagreeing_seeds names a seed that was not run")
    voxels = block["worst_pairwise_displacement_voxels"]
    if voxels is not None and not _finite_number(voxels):
        raise Open3dError("worst_pairwise_displacement_voxels must be finite or null")
    # F8: the voxel-normalised figure is the one the README, the evidence record and
    # the counterexample all quote, and it was the only field in the block with no
    # internal consistency check -- a stale or wrong normalisation passed `validate`
    # silently. The block now carries the voxel size it divided by, so the quotient
    # is checkable from the row alone.
    if voxels is not None:
        voxel_size = block.get("voxel_size")
        if not _finite_number(voxel_size) or float(voxel_size) <= 0:
            raise Open3dError(
                "a voxel-normalised disagreement must carry the positive voxel size "
                "it was divided by, or the figure cannot be checked from the row"
            )
        expected = float(worst) / float(voxel_size)
        # Relative, because the quotient spans many orders of magnitude across
        # scenes and an absolute epsilon would be vacuous at one end and wrong at
        # the other.
        if abs(float(voxels) - expected) > max(1e-9, abs(expected) * 1e-9):
            raise Open3dError(
                "worst_pairwise_displacement_voxels is not "
                "worst_pairwise_displacement divided by voxel_size"
            )
    _validate_counterexample(block["agreement_is_not_correctness"])


def _validate_counterexample(limitation: Any) -> None:
    """The limitation is a relation between two measured numbers, so check the relation.

    A consumer who reads agreement as correctness is the failure this measurement
    invites, so the block has to carry the case where agreement was high and every
    seed was wrong. Carried as a sentence it could be inverted by one word and still
    satisfy any phrase a guard looks for. Carried as figures, the claim is that the
    error dwarfs the agreement, and that either holds or it does not.
    """

    if not isinstance(limitation, dict) or "measured_counterexample" not in limitation:
        raise Open3dError(
            "the agreement block must carry the measured case where every seed agreed "
            "and every seed was wrong"
        )
    case = limitation["measured_counterexample"]
    required = ("seeds_agreed_within_voxels", "every_seed_was_wrong_by_voxels")
    if not isinstance(case, dict) or not set(required) <= set(case):
        raise Open3dError("the counterexample must carry both of its measurements")
    # Before the float() below, not after. Unpacking forced the conversion first, so
    # `float(None)` raised TypeError -- which the CLI's `(Open3dError, ValueError)` does
    # not catch -- before the guard written to reject exactly that was consulted. JSON
    # `null` is the likeliest real shape for a measurement that was not taken.
    if not all(_finite_number(case[key]) for key in required):
        raise Open3dError("the counterexample's measurements must be finite numbers")
    agreed, wrong = (float(case[key]) for key in required)
    if agreed <= 0 or wrong <= 0:
        raise Open3dError("the counterexample's measurements must be positive")
    # Strictly more than ten times, so a case sitting exactly on the boundary is
    # rejected. This guard exists to make the claim hard to satisfy, and the measured
    # figures clear it by a factor of twenty-five.
    #
    # Note the coupling this creates, because it is deliberate and has a cost: the
    # ratio is enforced on every agreement block the runtime *reads*, and the figures
    # it is checked against are the editorial constant `AGREEMENT_COUNTEREXAMPLE` in
    # `runner.py`. A future re-measurement that moved those figures below 10x would
    # therefore make the runner emit journals its own validator rejects -- a runtime
    # failure, not merely a red test. The trade is accepted: shipping the limitation
    # as a checkable relation is what stops it from being invertible prose, and if the
    # counterexample ever stops being one, the measurement's published claim has
    # changed and the constant is the right place to be forced to notice.
    if wrong <= 10.0 * agreed:
        raise Open3dError(
            f"the counterexample is not one: the seeds agreed within {agreed} voxels "
            f"and were wrong by {wrong}, which does not show that agreement and "
            "correctness are different things"
        )


def read_journal(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    except ValueError as exc:
        raise Open3dError("invalid registration journal") from exc
    if not rows:
        raise Open3dError("registration journal contains no fragment pairs")
    for row in rows:
        validate_pair(row)
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise Open3dError("duplicate pair identity in registration journal")
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the journal without hiding any pair from the denominator."""

    import numpy as np

    if not rows:
        raise Open3dError("registration journal contains no fragment pairs")
    icp_fitness = [float(row["icp"]["fitness"]) for row in rows]
    icp_rmse = [float(row["icp"]["inlier_rmse"]) for row in rows]
    global_rmse = [float(row["global_registration"]["inlier_rmse"]) for row in rows]
    return {
        "pair_count": len(rows),
        "input_points": sum(int(row["source_points"]) for row in rows),
        "icp_fitness": {
            "min": float(np.min(icp_fitness)),
            "mean": float(np.mean(icp_fitness)),
            "max": float(np.max(icp_fitness)),
        },
        "icp_inlier_rmse": {
            "min": float(np.min(icp_rmse)),
            "mean": float(np.mean(icp_rmse)),
            "max": float(np.max(icp_rmse)),
        },
        # The point of running ICP after RANSAC is that it lowers the residual.
        # Reporting both makes a no-op refinement visible instead of implied.
        "pairs_refined_by_icp": sum(
            1 for before, after in zip(global_rmse, icp_rmse) if after < before
        ),
    }


def validate_result(
    report: Any, rows: list[dict[str, Any]], *, run_id: str, kind: str
) -> None:
    """Validate a published report against the journal it claims to describe."""

    try:
        summary = summarize(rows)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise Open3dError("invalid registration journal fields") from exc
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != RESULT_SCHEMA
        or report.get("engine") != "open3d"
        or report.get("source_version") != SOURCE_VERSION
        or report.get("run_id") != run_id
        or report.get("kind") != kind
        or report.get("summary") != summary
    ):
        raise Open3dError(
            "result schema, identity or summary does not match the journal"
        )


def validate_pose_graph(graph: Any, *, fragment_count: int) -> None:
    """Validate an `open3d.pipelines.registration.PoseGraph` summary.

    Open3D's multiway registration anchors node 0 at the identity and optimizes
    the rest, so a graph whose first node has drifted is not the graph
    `global_optimization` produced.
    """

    import numpy as np

    if not isinstance(graph, dict) or not {"nodes", "edges"} <= set(graph):
        raise Open3dError("pose graph must report nodes and edges")
    nodes = graph["nodes"]
    edges = graph["edges"]
    if not isinstance(nodes, list) or len(nodes) != fragment_count:
        raise Open3dError("pose graph must hold exactly one node per fragment")
    if not isinstance(edges, list) or len(edges) < fragment_count - 1:
        raise Open3dError("pose graph must connect every fragment")
    for index, node in enumerate(nodes):
        if not isinstance(node, dict) or "pose" not in node:
            raise Open3dError("pose graph node must carry a pose")
        validate_rigid_transform(node["pose"], what=f"pose graph node {index}")
        if index == 0 and not np.allclose(
            np.asarray(node["pose"], dtype=float), np.eye(4), atol=ROTATION_TOLERANCE
        ):
            raise Open3dError("pose graph node 0 must remain the identity anchor")
    for edge in edges:
        if not isinstance(edge, dict) or not {
            "source_node_id",
            "target_node_id",
            "transformation",
            "uncertain",
        } <= set(edge):
            raise Open3dError("pose graph edge is missing required fields")
        validate_rigid_transform(edge["transformation"], what="pose graph edge")
        if not isinstance(edge["uncertain"], bool):
            raise Open3dError("pose graph edge uncertainty must be a boolean")
        for key in ("source_node_id", "target_node_id"):
            value = edge[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < fragment_count
            ):
                raise Open3dError(f"pose graph edge {key} is out of range")
    if "optimization" in graph:
        _validate_optimization_report(
            graph["optimization"], nodes=len(nodes), edges=edges
        )


def _validate_optimization_report(report: Any, *, nodes: int, edges: list) -> None:
    """Hold the optimization summary to the graph it is a summary of.

    Optional, because reports written before this block existed are still valid
    pose graphs. What it must not be is present and hollow: an empty mapping would
    otherwise satisfy `"optimization" in graph` while asserting nothing.
    """

    if not isinstance(report, dict) or not report:
        raise Open3dError("pose graph optimization summary must be a non-empty mapping")
    required = (
        "edges_submitted",
        "loop_closures_submitted",
        "edges_that_survived_pruning",
        "edges_pruned",
        "max_pose_correction",
        "max_pose_correction_per_node",
    )
    missing = sorted(key for key in required if key not in report)
    if missing:
        raise Open3dError(
            f"pose graph optimization summary is missing {', '.join(missing)}"
        )
    counts = {}
    for key in required[:4]:
        value = report[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise Open3dError(f"pose graph optimization {key} must be a count")
        counts[key] = value
    if counts["edges_that_survived_pruning"] != len(edges):
        raise Open3dError(
            "pose graph optimization reports a different surviving edge count than "
            "the graph serializes"
        )
    if (
        counts["edges_submitted"] - counts["edges_pruned"]
        != counts["edges_that_survived_pruning"]
    ):
        raise Open3dError(
            "pose graph optimization edge counts do not add up: submitted minus "
            "pruned must equal the edges that survived"
        )
    # Only loop closures are prunable -- consecutive pairs are submitted as certain
    # odometry -- so more pruned than submitted as closures means the summary is
    # describing a different graph.
    if counts["loop_closures_submitted"] > counts["edges_submitted"]:
        raise Open3dError(
            "pose graph optimization reports more loop closures than edges"
        )
    if counts["edges_pruned"] > counts["loop_closures_submitted"]:
        raise Open3dError(
            "pose graph optimization reports more pruned edges than it submitted as "
            "prunable loop closures"
        )
    per_node = report["max_pose_correction_per_node"]
    if not isinstance(per_node, list) or len(per_node) != nodes:
        raise Open3dError(
            "pose graph optimization must report one pose correction per node"
        )
    if not all(_finite_number(value) and value >= 0.0 for value in per_node):
        raise Open3dError("every pose correction must be a finite, non-negative number")
    largest = report["max_pose_correction"]
    if not _finite_number(largest) or largest < 0.0:
        raise Open3dError("the largest pose correction must be finite and non-negative")
    if per_node and abs(float(largest) - max(float(v) for v in per_node)) > 1e-12:
        raise Open3dError(
            "the largest pose correction is not the largest of the per-node "
            "corrections it summarizes"
        )
    _validate_optimization_displacement(report, nodes=nodes)


def _validate_optimization_displacement(report: dict, *, nodes: int) -> None:
    """Hold the displacement figures to each other and to the extent they divide by.

    These are the fields a reader is meant to act on -- a bare correction is only
    small compared with something -- and they were published outside this validator,
    so a record could carry a real displacement under a zero share and pass. That is
    the same failure this block exists to prevent, one field over.

    Optional as a group, for the same reason the block itself is: records written
    before these fields existed are still valid pose graphs. Present but internally
    inconsistent is what is rejected.
    """

    if "max_point_displacement" not in report:
        return
    per_node = report.get("max_point_displacement_per_node")
    if not isinstance(per_node, list) or len(per_node) != nodes:
        raise Open3dError(
            "pose graph optimization must report one point displacement per node"
        )
    if not all(_finite_number(value) and value >= 0.0 for value in per_node):
        raise Open3dError(
            "every point displacement must be a finite, non-negative number"
        )
    largest = report["max_point_displacement"]
    if not _finite_number(largest) or largest < 0.0:
        raise Open3dError(
            "the largest point displacement must be finite and non-negative"
        )
    if per_node and abs(float(largest) - max(float(v) for v in per_node)) > 1e-12:
        raise Open3dError(
            "the largest point displacement is not the largest of the per-node "
            "displacements it summarizes"
        )
    if "scene_extent" not in report:
        return
    extent = report["scene_extent"]
    # A zero or negative extent is a degenerate fused cloud, and the share below is
    # derived from it. Rejecting it here is cheaper than a division the reader has to
    # reconstruct to notice.
    if not _finite_number(extent) or extent <= 0.0:
        raise Open3dError("the scene extent must be a finite, positive length")
    share = report.get("max_point_displacement_as_share_of_extent")
    if share is None:
        raise Open3dError(
            "pose graph optimization publishes a scene extent and a displacement but "
            "no share of the extent, which is the figure a reader acts on"
        )
    if not _finite_number(share) or share < 0.0:
        raise Open3dError(
            "the displacement share of the scene extent must be finite and non-negative"
        )
    expected = float(largest) / float(extent)
    if abs(float(share) - expected) > max(1e-12, 1e-9 * expected):
        raise Open3dError(
            f"the published share {share} is not the displacement {largest} over the "
            f"scene extent {extent}, which is {expected}"
        )


def validate_mesh(mesh: Any) -> None:
    """Validate a reconstructed surface as geometry, not as a file that exists."""

    required = {
        "vertex_count",
        "triangle_count",
        "surface_area",
        "is_edge_manifold",
        "is_vertex_manifold",
        "is_watertight",
        "extent",
        "sha256",
    }
    if not isinstance(mesh, dict) or not required <= set(mesh):
        raise Open3dError("mesh summary is missing required fields")
    for key in ("vertex_count", "triangle_count"):
        if (
            isinstance(mesh[key], bool)
            or not isinstance(mesh[key], int)
            or mesh[key] <= 0
        ):
            raise Open3dError(f"mesh {key} must be positive")
    # Poisson reconstruction on a degenerate cloud can return a mesh with
    # vertices and no faces; a surface with no area is not a reconstruction.
    if not _finite_number(mesh["surface_area"]) or float(mesh["surface_area"]) <= 0.0:
        raise Open3dError("mesh surface area must be positive")
    for key in ("is_edge_manifold", "is_vertex_manifold", "is_watertight"):
        if not isinstance(mesh[key], bool):
            raise Open3dError(f"mesh {key} must be a boolean")
    extent = mesh["extent"]
    if (
        not isinstance(extent, list)
        or len(extent) != 3
        or not all(_finite_number(value) and value > 0 for value in extent)
    ):
        raise Open3dError("mesh bounding-box extent must be three positive lengths")
    if not isinstance(mesh["sha256"], str) or len(mesh["sha256"]) != 64:
        raise Open3dError("mesh must carry a sha256 digest")


def validate_distance_bands(report: Any) -> None:
    """Check the unsupported-area bands against each other.

    The bands are nested by construction — area past three voxels is also past 1.5, which
    is also past one — so they must decrease. If they do not, the distances and the areas
    were not measured over the same triangles, and the band that says whether a crop
    removed invented surface or correct geometry cannot be trusted.

    Absent bands are accepted. `validate` is also how an already-published report gets
    re-verified, and a report written before the bands existed is not wrong, only older.
    Demanding them would reject artifacts this tool itself produced.
    """

    if not isinstance(report, dict):
        raise Open3dError("support report must be an object")
    keys = tuple(
        key
        for key in (
            "unsupported_area_fraction",
            "unsupported_area_beyond_1_5_voxels",
            "unsupported_area_beyond_3_voxels",
        )
        if key in report
    )
    if len(keys) < 2:
        return
    values = []
    for key in keys:
        value = report.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not 0.0 <= float(value) <= 1.0
        ):
            raise Open3dError(
                f"support report {key} must be a fraction between 0 and 1"
            )
        values.append(float(value))
    for (near_key, near), (far_key, far) in zip(
        list(zip(keys, values, strict=True))[:-1],
        list(zip(keys, values, strict=True))[1:],
        strict=True,
    ):
        if far > near + 1e-9:
            raise Open3dError(
                f"support report is inconsistent: {far_key} ({far}) exceeds {near_key} "
                f"({near}), but the further band is a subset of the nearer one"
            )


def validate_overlay_matches_crop(recording: Any, reconstruction: Any) -> None:
    """Hold the "removed surface" overlay to the crop it claims to show.

    The overlay redraws the crop from the uncropped mesh, so it can silently use a
    different threshold than the reconstruction did and still leave every published
    number correct — the view is wrong and nothing else is. Tying the triangle count
    it logged to the triangles the crop actually removed turns that into an
    arithmetic disagreement a reviewer can see, instead of something only found by
    reading both functions side by side.
    """

    if not isinstance(recording, dict) or not isinstance(reconstruction, dict):
        raise Open3dError("overlay check needs both the recording and mesh reports")
    shown = recording.get("unsupported_triangles_shown")
    if not isinstance(shown, int) or shown < 0:
        raise Open3dError(
            "recording does not report how many unsupported triangles it drew"
        )
    factor = reconstruction.get("support_distance_factor")
    if not isinstance(factor, int | float):
        raise Open3dError("reconstruction does not record its support_distance_factor")
    if factor <= 0 or not reconstruction.get("unsupported_vertices_removed"):
        if shown:
            raise Open3dError(
                "recording drew removed surface for a run that cropped nothing; the "
                "overlay and the published mesh disagree"
            )
        return
    kept = (reconstruction.get("mesh") or {}).get("triangle_count")
    full = (reconstruction.get("mesh_uncropped") or {}).get("triangle_count")
    if not isinstance(kept, int) or not isinstance(full, int):
        raise Open3dError("reconstruction does not record both triangle counts")
    if shown != full - kept:
        raise Open3dError(
            f"recording drew {shown} unsupported triangles but the crop removed "
            f"{full - kept}; the overlay is not showing the crop that ran"
        )


#: Sources that mean "the scan did not determine an up axis, so this is the declared
#: default". Every one of them must publish the default rather than a measured axis.
_DECLARED_UP_SOURCES = frozenset(
    {"default", "default-weak-plane", "default-ambiguous-plane"}
)
#: The one source that names a measured axis, plus the defaults above. Kept as a
#: closed set because the semantic checks are keyed to these literals: a source
#: outside it would reach none of them and be accepted unexamined.
_MEASURED_UP_SOURCE = "largest-plane-normal-snapped-to-axis"
_KNOWN_UP_SOURCES = _DECLARED_UP_SOURCES | {_MEASURED_UP_SOURCE}
#: What the counterexample must say to be one: repeats that agreed, and a tilt large
#: enough that agreeing on it is visibly not the same as the plane meaning something.
_COUNTEREXAMPLE_TILT_DEGREES = 10.0


def validate_up_axis_inference(recording: Any) -> None:
    """Stop a coin-flipped camera axis from being published as a measured one.

    The up vector steers the camera, and `segment_plane` is RANSAC: on a scan with
    two comparable planes the axis differs run to run while every published field
    stays the same, so a reviewer reading one run cannot tell the next run will roll
    the view ninety degrees. The runner answers that by repeating the fit and
    refusing to name an axis when the repeats disagree. This holds the record to
    that: a report that says it measured an axis must show the repeats agreed, and a
    report that says the repeats disagreed must publish the declared default rather
    than one of the branches.
    """

    if not isinstance(recording, dict):
        raise Open3dError("up-axis check needs the recording report")
    inference = recording.get("up_axis_inference", _ABSENT)
    if inference is _ABSENT:
        return  # Predates the inference being reported at all.
    if not isinstance(inference, dict):
        raise Open3dError("up_axis_inference is not a report")
    source = inference.get("source")
    up = inference.get("up")
    if not isinstance(source, str) or not source:
        raise Open3dError("up_axis_inference does not say where its axis came from")
    # Both semantic claims below are keyed to literal source strings, so a source
    # this function does not recognise reaches neither of them and is accepted with
    # any number of disagreeing axes. Nothing emits an unrecognised source today, so
    # this was safe by circumstance rather than by construction -- and the dangerous
    # shape is not a wild string but a near miss like `largest-plane-normal`, a
    # truncation of the real one, which would turn both checks off silently instead
    # of failing loudly.
    if source not in _KNOWN_UP_SOURCES:
        raise Open3dError(
            f"up_axis_inference reports an unrecognised source {source!r}; the "
            "checks below are keyed to the known sources and would not apply"
        )
    if not isinstance(up, list) or len(up) != 3:
        raise Open3dError("up_axis_inference does not publish a three-vector")
    if not all(
        isinstance(value, int | float) and _finite_number(value) for value in up
    ):
        raise Open3dError("the published up vector is not made of finite numbers")
    if source in _DECLARED_UP_SOURCES and tuple(float(v) for v in up) != (
        0.0,
        0.0,
        1.0,
    ):
        raise Open3dError(
            f"source {source!r} means no axis was determined, but the report "
            f"publishes {up} instead of the declared default"
        )

    agreement = inference.get("agreement", _ABSENT)
    if agreement is _ABSENT:
        if source == "default-ambiguous-plane":
            raise Open3dError(
                "the report claims the repeats disagreed but does not say what they "
                "disagreed about"
            )
        return
    if not isinstance(agreement, dict):
        raise Open3dError("up_axis_inference.agreement is not a report")
    distinct = agreement.get("distinct_axes")
    repeats = agreement.get("repeats")
    # `isinstance(True, int)` is True, so without the bool exclusions these two
    # accepted `distinct: True` where `attempted` below deliberately rejects it. One
    # rule for all three counts.
    if not isinstance(distinct, int) or isinstance(distinct, bool) or distinct < 1:
        raise Open3dError("the agreement report does not count the axes it found")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise Open3dError("the agreement report does not count its repeats")
    if distinct > repeats:
        raise Open3dError(
            f"the agreement report found {distinct} axes in {repeats} repeats"
        )
    # Required, not optional. The check below is the only thing standing between a
    # one-usable-fit sample and the word "measured", and an optional field is a check
    # the producer can turn off by omission -- which is the dependency on the producer
    # staying correct that this validator exists to remove.
    attempted = agreement.get("attempted_fits")
    if not isinstance(attempted, int) or isinstance(attempted, bool):
        raise Open3dError(
            "the agreement report does not say how many fits were attempted, so the "
            "repeats it counts cannot be read as a majority of anything"
        )
    if attempted < repeats:
        raise Open3dError(
            f"the agreement report counts {repeats} repeats out of "
            f"{attempted} attempted fits"
        )
    # A majority of attempts was not enough on its own: it is a ratio, and a ratio is
    # satisfied by 1 of 1. `_up_axis(..., repeats=0)` produces exactly that, and the
    # resulting report -- a single fit, which is the coin flip this whole change exists
    # to replace -- passed every guard here and was published as a measured axis. The
    # product call site uses the default, so there was no live exposure; the point of
    # this validator is not to depend on that.
    #
    # Two, not more: the claim being checked is that repeats agreed, and two repeats
    # are the fewest that can agree. A higher floor would be this validator deciding
    # how many fits the producer must perform, which is the producer's parameter.
    if source == _MEASURED_UP_SOURCE and attempted < 2:
        raise Open3dError(
            f"the report names a measured axis from {attempted} attempted fit, and "
            "one fit cannot agree with anything"
        )
    # A majority, so "the repeats agreed" cannot be satisfied by a sample of one out
    # of nine. Without this, a scan whose fits were weak eight times in nine and
    # strong on the draw that happened to be first was published as measured.
    if source == "largest-plane-normal-snapped-to-axis" and 2 * repeats <= attempted:
        raise Open3dError(
            f"the report names a measured axis from {repeats} usable fits out of "
            f"{attempted} attempted, which is not a majority of them"
        )
    # The two claims that must not come apart: naming an axis requires the repeats to
    # have agreed, and reporting disagreement requires not naming one.
    if source == "largest-plane-normal-snapped-to-axis" and distinct != 1:
        raise Open3dError(
            f"the report names a measured axis while its {repeats} repeats found "
            f"{distinct} different ones"
        )
    if source == "default-ambiguous-plane" and distinct < 2:
        raise Open3dError(
            "the report falls back to the default for ambiguity while its repeats "
            "all agreed"
        )
    candidates = agreement.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != distinct:
        raise Open3dError(
            f"the agreement report counts {distinct} axes but lists "
            f"{len(candidates) if isinstance(candidates, list) else 'no'} candidates"
        )
    # Checked before `.get` below: a non-dict element raised `AttributeError`, which
    # is outside the CLI's `(Open3dError, ValueError)` and so broke the
    # machine-readable failure contract for that one shape.
    if not all(isinstance(one, dict) for one in candidates):
        raise Open3dError("the agreement report lists a candidate that is not a report")
    if sum(one.get("chosen_by_repeats", 0) for one in candidates) != repeats:
        raise Open3dError(
            "the candidates' repeat counts do not add up to the repeats performed"
        )
    # The direction, not only the counts. Every check above constrains how many axes
    # were found and how often; none constrained *which* axis got published. A report
    # naming a measured axis of (0,0,1) whose single agreeing candidate is (1,0,0)
    # satisfied all of them, so the vector the view cameras consume could differ from
    # the axis the agreement block attests to and the record still validated. The
    # ambiguous-source branch is already bound to the declared default; this binds the
    # measured branch to its own evidence.
    if source == _MEASURED_UP_SOURCE:
        agreed = candidates[0].get("up")
        if not isinstance(agreed, list) or len(agreed) != 3:
            raise Open3dError(
                "the agreement report's candidate does not name the axis it agreed on"
            )
        try:
            published = [float(value) for value in up]
            attested = [float(value) for value in agreed]
        except (TypeError, ValueError):
            raise Open3dError(
                "the agreement report's axis is not three numbers"
            ) from None
        if published != attested:
            raise Open3dError(
                f"the report publishes {published} as a measured axis while its "
                f"repeats agreed on {attested}"
            )
    _validate_up_counterexample(agreement.get("agreement_is_not_meaningfulness"))


def _validate_up_counterexample(limitation: Any) -> None:
    """Require the published limitation to be a measured case, not a sentence.

    The claim this guards is that repeat agreement does not establish the plane
    means anything. Carried as prose, that survives being negated; carried as a
    case where the repeats agreed unanimously on a plane tilted well off the axis
    it was snapped to, it does not.
    """

    if not isinstance(limitation, dict):
        raise Open3dError("the agreement report does not publish its limitation")
    case = limitation.get("measured_counterexample")
    if not isinstance(case, dict):
        raise Open3dError("the limitation is not backed by a measured counterexample")
    required = (
        "repeats_that_agreed",
        "of_repeats",
        "plane_tilt_to_the_snapped_axis_degrees",
    )
    missing = [key for key in required if key not in case]
    if missing:
        raise Open3dError(f"the counterexample does not publish {missing}")
    if not all(_finite_number(case[key]) for key in required):
        raise Open3dError("the counterexample's measurements must be finite numbers")
    agreed = float(case["repeats_that_agreed"])
    total = float(case["of_repeats"])
    tilt = float(case["plane_tilt_to_the_snapped_axis_degrees"])
    if total <= 0 or agreed != total:
        raise Open3dError(
            f"the counterexample is not one: {agreed} of {total} repeats agreed, so it "
            "does not show agreement failing to mean anything"
        )
    if tilt < _COUNTEREXAMPLE_TILT_DEGREES:
        raise Open3dError(
            f"the counterexample's plane is {tilt} degrees off the axis it was snapped "
            f"to, under the {_COUNTEREXAMPLE_TILT_DEGREES} this claim needs to be "
            "visible rather than arguable"
        )


def validate_density_crop_contribution(report: Any) -> None:
    """Hold the density crop's reported contribution to its own removal count.

    `low_density_vertices_removed` reads as surface this filter took out of the
    delivered mesh. Whenever support cropping is on, it usually is not: the two
    crops overlap almost entirely, because a vertex the solver extrapolated into
    empty space is both low-density and far from any observation. The contribution
    block says how much of the removal was this crop's own, and this keeps that
    block honest about the count beside it.

    Optional: reports written before the block existed are still valid.
    """

    if not isinstance(report, dict):
        raise Open3dError("density-crop check needs the reconstruction report")
    contribution = report.get("density_crop_contribution", _ABSENT)
    if contribution is _ABSENT:
        return
    # `None` is refused and absence is not, which is the whole reason for the
    # sentinel above. Collapsing them here would have let a report publish the
    # measurement as nothing while the less alarming empty mapping was refused.
    if not isinstance(contribution, dict) or not contribution:
        raise Open3dError(
            "density_crop_contribution must be a non-empty mapping when present"
        )
    for key in ("vertices_removed", "removed_only_by_this_crop"):
        value = contribution.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise Open3dError(f"density_crop_contribution {key} must be a count")
    removed = contribution["vertices_removed"]
    exclusive = contribution["removed_only_by_this_crop"]
    if exclusive > removed:
        raise Open3dError(
            "density_crop_contribution claims more exclusive removals than removals"
        )
    disabled = contribution.get("support_crop_was_disabled")
    if not isinstance(disabled, bool):
        raise Open3dError(
            "density_crop_contribution must say whether support cropping was on"
        )
    # With no support crop to overlap with, every density removal is this crop's own.
    # A report claiming otherwise is describing a pipeline that did not run.
    if disabled and exclusive != removed:
        raise Open3dError(
            "support cropping was disabled, so every density removal is this crop's "
            f"own, but the report credits it with {exclusive} of {removed}"
        )
    # And the flag against the field it is derived from, which sits in the same
    # report. The check above was one-directional: it held the counts to the flag and
    # never held the flag to the configuration, so a report could declare
    # `support_crop_was_disabled: false` beside `support_distance_factor: 0.0` and
    # credit the crop with 0 of 518 -- impossible by the same construction the True
    # branch enforces -- and validate cleanly. `validate_overlay_matches_crop` already
    # ties the overlay to this field; this ties the flag to it.
    factor = report.get("support_distance_factor")
    if _finite_number(factor):
        if disabled != (float(factor) <= 0.0):
            raise Open3dError(
                f"density_crop_contribution says support cropping was "
                f"{'disabled' if disabled else 'enabled'} while the report ran at a "
                f"support distance factor of {factor}"
            )
        # The distance it published is that factor times the voxel size, and a
        # mismatch means the exclusive count was measured against a different
        # threshold than the one the report names.
        distance = contribution.get("support_distance")
        voxel = report.get("voxel_size")
        if not disabled and _finite_number(distance) and _finite_number(voxel):
            expected = float(voxel) * float(factor)
            if abs(float(distance) - expected) > max(1e-12, 1e-9 * abs(expected)):
                raise Open3dError(
                    f"density_crop_contribution measured against a support distance "
                    f"of {distance} where the report's voxel size and factor give "
                    f"{expected}"
                )
    published = report.get("low_density_vertices_removed")
    if isinstance(published, int) and not isinstance(published, bool):
        if published != removed:
            raise Open3dError(
                "density_crop_contribution counts a different number of removals "
                f"({removed}) than the report publishes ({published})"
            )


def validate_support(report: Any, *, voxel_size: float) -> None:
    """Check the support/coverage pair a reconstruction publishes about itself.

    The point of these numbers is that they are checkable against each other. A
    crop that claims to have removed unsupported surface must leave the observed
    samples where they were: if coverage fell with the area, the crop took real
    surface, and the reconstruction is worse rather than more honest.
    """

    if not isinstance(report, dict):
        raise Open3dError("reconstruction report must be an object")
    before = report.get("support_before_crop")
    after = report.get("support")
    coverage = report.get("coverage")
    for name, block, keys in (
        ("support_before_crop", before, ("unsupported_area_fraction",)),
        ("support", after, ("unsupported_area_fraction",)),
        (
            "coverage",
            coverage,
            ("fraction_within_voxel", "sample_to_surface_rmse", "samples"),
        ),
    ):
        if not isinstance(block, dict) or not set(keys) <= set(block):
            raise Open3dError(f"reconstruction report is missing {name} measurements")
    for block, key in ((before, "unsupported"), (after, "unsupported")):
        fraction = block["unsupported_area_fraction"]
        if not _finite_number(fraction) or not 0.0 <= float(fraction) <= 1.0:
            raise Open3dError(f"{key} area fraction must be a fraction in [0, 1]")
        validate_distance_bands(block)
    within = coverage["fraction_within_voxel"]
    if not _finite_number(within) or not 0.0 <= float(within) <= 1.0:
        raise Open3dError("coverage fraction must be a fraction in [0, 1]")
    rmse = coverage["sample_to_surface_rmse"]
    if not _finite_number(rmse) or float(rmse) < 0.0:
        raise Open3dError("sample-to-surface RMSE must be a nonnegative distance")
    if isinstance(coverage["samples"], bool) or not isinstance(
        coverage["samples"], int
    ):
        raise Open3dError("coverage must report how many samples it measured")
    if coverage["samples"] <= 0:
        raise Open3dError("coverage must measure at least one sample")
    factor = report.get("support_distance_factor")
    if not _finite_number(factor) or float(factor) < 0.0:
        raise Open3dError("support_distance_factor must be a nonnegative multiple")
    # Cropping can only remove area, so it cannot raise the unsupported fraction.
    if (
        float(factor) > 0.0
        and float(after["unsupported_area_fraction"])
        > float(before["unsupported_area_fraction"]) + 1e-9
    ):
        raise Open3dError(
            "support cropping reported more unsupported area than before it ran"
        )
    if float(factor) > 0.0 and float(after["max_vertex_distance_to_sample"]) > (
        voxel_size * float(factor)
    ) * (1.0 + 1e-6):
        raise Open3dError(
            "surface remains farther from the scan than the support limit allows"
        )
    _validate_coverage_cost(report, coverage, float(factor))


#: Distinguishes "this record predates the coverage pair" from "this record published
#: the pair as nothing". Only the first is exempt.
_ABSENT = object()


def _validate_coverage_cost(report: Any, coverage: dict, factor: float) -> None:
    """Check what the crop cost coverage against the coverage pair it sits beside.

    Reports written before this pair existed do not carry it, and are left alone. A
    report that carries the keys with nothing in them is not one of those: it is a new
    report declining to measure, and it is an error. What
    this does check is the one direction the crop can move coverage: removing surface
    cannot bring a sample closer to it, so coverage can only fall, and a sample covered
    afterwards must have been covered before. That makes the abandonment count equal to
    the drop, and a report where those disagree is describing something the crop cannot
    do.
    """

    # A sentinel rather than `None`, because `report.get(k)` gives `None` both for an
    # old record that predates this pair and for a new one that published the keys
    # carrying nothing, and those deserve opposite answers. Keyed on `None` the
    # exemption was a switch: emitting both as `None` kept the keys present, satisfied
    # the guard that reads them, and skipped every check below.
    cost = report.get("coverage_cost_of_crop", _ABSENT)
    before = report.get("coverage_before_crop", _ABSENT)
    if cost is _ABSENT and before is _ABSENT:
        return
    cost = None if cost is _ABSENT else cost
    before = None if before is _ABSENT else before
    for name, block, keys in (
        ("coverage_before_crop", before, ("fraction_within_voxel", "samples")),
        (
            "coverage_cost_of_crop",
            cost,
            (
                "samples",
                "samples_covered_before_crop",
                "samples_abandoned_by_crop",
                "coverage_is_bit_identical",
            ),
        ),
    ):
        if not isinstance(block, dict) or not set(keys) <= set(block):
            raise Open3dError(f"reconstruction report is missing {name} measurements")
    if not (coverage["samples"] == before["samples"] == cost["samples"]):
        raise Open3dError(
            "the coverage pair and its cost must measure the same samples"
        )
    samples = int(coverage["samples"])
    fraction_before = before["fraction_within_voxel"]
    if not _finite_number(fraction_before) or not 0.0 <= float(fraction_before) <= 1.0:
        raise Open3dError("coverage fraction must be a fraction in [0, 1]")
    if float(coverage["fraction_within_voxel"]) > float(fraction_before) + 1e-9:
        raise Open3dError(
            "coverage rose across the crop, which removing surface cannot do"
        )
    covered_before = round(float(fraction_before) * samples)
    covered_after = round(float(coverage["fraction_within_voxel"]) * samples)
    for key in ("samples_covered_before_crop", "samples_abandoned_by_crop"):
        value = cost[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise Open3dError(f"{key} must be a nonnegative count")
    if cost["samples_covered_before_crop"] != covered_before:
        raise Open3dError(
            "the cost block disagrees with coverage_before_crop about how many "
            "samples the surface explained"
        )
    if cost["samples_abandoned_by_crop"] != covered_before - covered_after:
        raise Open3dError(
            "abandoned samples must account for the whole drop in coverage"
        )
    if cost["coverage_is_bit_identical"] is not (covered_before == covered_after):
        raise Open3dError(
            "the cost block disagrees with the coverage pair about whether coverage "
            "moved"
        )
    worst = cost.get("worst_abandoned_sample_voxels_from_surface")
    if cost["samples_abandoned_by_crop"]:
        if not _finite_number(worst) or float(worst) < 1.0:
            raise Open3dError(
                "an abandoned sample is by definition more than one voxel from the "
                "surface, so the worst of them cannot be nearer than that"
            )
    elif worst is not None:
        raise Open3dError(
            "no sample was abandoned, so there is no worst abandoned sample"
        )
    if factor <= 0.0 and cost["samples_abandoned_by_crop"]:
        raise Open3dError(
            "the crop was disabled, so it cannot have abandoned any sample"
        )


def verify_rerun_recording(path: Path) -> None:
    """Decode the recording with Rerun's own verifier, not by checking a suffix."""

    import shutil
    import subprocess
    import sys

    sibling = Path(sys.executable).with_name("rerun")
    rerun = str(sibling) if sibling.is_file() else shutil.which("rerun")
    if not rerun:
        raise Open3dError("Rerun CLI is unavailable")
    checked = subprocess.run(
        [rerun, "rrd", "verify", str(path)], capture_output=True, check=False
    )
    if checked.returncode:
        raise Open3dError("Rerun rejected the generated recording")
    # `rrd verify` alone does not keep the promise in the line above. Measured against
    # rerun-sdk 0.38.1, it exits 0 for a zero-byte file, for one truncated to its
    # 40-byte header, and for a recording opened and closed without logging anything.
    # Decoding rejects the first two. The third decodes cleanly, because Rerun writes a
    # `/__properties` chunk whether or not the caller logged data, so the only thing
    # that separates it from a real recording is an entity path the caller put there.
    #
    # None of the three subsumes the others, and each rejection below is the only one
    # that fires on some shape -- a recording truncated to between 75 and 99 percent of
    # its length still decodes to `/points` through the reader's whole-file fallback,
    # and only the CLI rejects it. That band widens with recording size rather than
    # being an artifact of a small one: 50 to 95 percent at 18 MB. The measurements are in
    # tests/workbench/test_open3d_rrd_verifier.py, which fails if any layer stops being
    # the sole catcher of its shape, rather than letting one be dropped as redundant.
    try:
        # Imported here, after the CLI check, so the usual "no Rerun at all" case still
        # reports the unavailable CLI above. A machine with the binary but not the SDK
        # would otherwise raise ImportError, which no caller catches.
        from npa.viz.recordings import load_recordings

        recordings = load_recordings(path)
        logged = [
            str(chunk.entity_path)
            for recording in recordings
            for chunk in recording.chunks()
        ]
    except ImportError as exc:
        raise Open3dError(
            f"the Rerun SDK is unavailable to decode the recording: {exc}"
        )
    except (OSError, RuntimeError, ValueError) as exc:
        # OSError because the decode is a second read of the file, minutes of
        # subprocess later: the path can be gone, unreadable, or a directory by the
        # time `RrdReader` opens it, and pyo3 surfaces `std::io::Error` as `OSError`.
        # Without it the function raised something `_call` does not catch
        # ((Open3dError, ValueError)), so the CLI died with a traceback instead of the
        # contracted message and exit 1 -- the one way this change could have made the
        # failure contract worse rather than better.
        #
        # Deliberately not `BaseException`, which is what it would take to also catch
        # a pyo3 `PanicException`. That would swallow `KeyboardInterrupt` and
        # `SystemExit`, and a panic is a bug in the decoder rather than a condition
        # this function can report on; it should keep its traceback. The 26-cut
        # truncation sweep in the tests is the evidence that real corruption arrives
        # as `RuntimeError` and not as a panic.
        raise Open3dError(f"Rerun could not decode the generated recording: {exc}")
    substantive = [name for name in logged if not name.startswith(RERUN_INTERNAL)]
    if not substantive:
        raise Open3dError(
            "the generated recording carries no logged data: "
            f"{len(recordings)} recordings, entity paths {sorted(set(logged))}"
        )
