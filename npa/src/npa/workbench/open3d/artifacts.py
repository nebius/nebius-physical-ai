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
