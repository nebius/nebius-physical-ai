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
