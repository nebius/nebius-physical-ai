"""Derive the publishable report for a workflow run from the run's own artifacts.

A run's artifacts are written for the infrastructure that produced them, so they name
their own object-store locations, their run, and the resolved execution profile they
were scheduled onto. None of that belongs in this repository, and redacting it after
the fact does not work: the first attempt committed the artifacts raw, and the second
replaced the bucket name and left the run identifier and every object key under it.

So this selects rather than redacts. Every field is dropped unless a pattern in
`ALLOWED` names it, and a field that matches neither `ALLOWED` nor `OMITTED` stops the
script instead of being quietly dropped or quietly kept -- a new field in a future run
has to be classified by someone. Nothing is substituted for what is dropped: an invented
run identifier or a `<bucket>` placeholder reads as though a location is being disclosed,
and the categories that were removed are reported instead.

What survives is what the evidence is for: the measured numbers, the parameters that
produced them, the hashes that identify the bytes, and the shape of the stage graph.

Run from the repository root, against a directory holding the run's artifacts:
    python3 docs/evidence/open3d/harness/derive_public_run_report.py \
        --artifacts <dir> --output docs/evidence/open3d/replay-b18d-public-report.json

Exits non-zero on an unclassified field, so it cannot half-succeed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

#: Artifacts to read, in stage order.
ARTIFACTS = (
    "npa-workflow/status.json",
    "npa-workflow/manifest.json",
    "prepared/manifest.json",
    "pairs/result.json",
    "validation/validation.json",
    "graph/result.json",
    "graph/pose_graph.json",
    "surface/result.json",
    "reports/rrd-manifest.json",
)

#: What an allowed leaf is permitted to hold. The name alone is not enough: an earlier
#: version of this table allowed `mesh.*`, and a review's counterexample showed a string
#: field nobody had classified surviving under it untouched. A field has to be named
#: *and* hold the kind of value it is named for, so a private string cannot arrive in a
#: slot the schema says is a measurement.
NUM = "number"  # int or float, never bool
TEXT = "text"  # a short schema-defined string: an engine, a kind, a fragment id
HASH = "sha256"  # 64 lowercase hex
FLAG = "bool"
NOTE = "prose"  # the two semantic strings this evidence exists to carry
COUNT_OR_NULL = "number-or-null"

_SUMMARY = {
    "summary.icp_fitness.max": NUM,
    "summary.icp_fitness.mean": NUM,
    "summary.icp_fitness.min": NUM,
    "summary.icp_inlier_rmse.max": NUM,
    "summary.icp_inlier_rmse.mean": NUM,
    "summary.icp_inlier_rmse.min": NUM,
    "summary.input_points": NUM,
    "summary.pair_count": NUM,
    "summary.pairs_refined_by_icp": NUM,
}


def _pose_graph(prefix: str = "") -> dict[str, str]:
    return {
        f"{prefix}edge_prune_threshold": NUM,
        f"{prefix}edges[].confidence": NUM,
        f"{prefix}edges[].source_node_id": NUM,
        f"{prefix}edges[].target_node_id": NUM,
        f"{prefix}edges[].transformation[][]": NUM,
        f"{prefix}edges[].uncertain": FLAG,
        f"{prefix}fused_points": NUM,
        f"{prefix}fused_points_before_downsample": NUM,
        f"{prefix}fused_sha256": HASH,
        f"{prefix}nodes[].fragment_id": TEXT,
        f"{prefix}nodes[].points": NUM,
        f"{prefix}nodes[].pose[][]": NUM,
        f"{prefix}reference_node": NUM,
    }


def _mesh(prefix: str) -> dict[str, str]:
    return {
        f"{prefix}.bytes": NUM,
        f"{prefix}.extent[]": NUM,
        f"{prefix}.is_edge_manifold": FLAG,
        f"{prefix}.is_vertex_manifold": FLAG,
        f"{prefix}.is_watertight": FLAG,
        f"{prefix}.sha256": HASH,
        f"{prefix}.surface_area": NUM,
        f"{prefix}.triangle_count": NUM,
        f"{prefix}.vertex_count": NUM,
    }


def _support(prefix: str) -> dict[str, str]:
    return {
        f"{prefix}.max_vertex_distance_to_sample": NUM,
        f"{prefix}.median_vertex_distance_to_sample": NUM,
        f"{prefix}.p95_vertex_distance_to_sample": NUM,
        f"{prefix}.unsupported_area": NUM,
        f"{prefix}.unsupported_area_beyond_1_5_voxels": NUM,
        f"{prefix}.unsupported_area_beyond_3_voxels": NUM,
        f"{prefix}.unsupported_area_fraction": NUM,
        f"{prefix}.voxel_size": NUM,
    }


def _camera(prefix: str) -> dict[str, str]:
    return {
        f"{prefix}.distance": NUM,
        f"{prefix}.elevation_degrees": NUM,
        f"{prefix}.eye[]": NUM,
        f"{prefix}.fov_degrees": NUM,
        f"{prefix}.look_target[]": NUM,
        f"{prefix}.pane_aspect": NUM,
        f"{prefix}.scene_extent[]": NUM,
        f"{prefix}.scene_span": NUM,
        f"{prefix}.up[]": NUM,
    }


#: Every leaf carried into the public report, per artifact, with the value kind it must
#: hold. `[]` marks a list element. The only `*` is the view-name key of `view_cameras`,
#: whose keys are labels rather than schema; its leaves are still named one by one.
#: There is no subtree wildcard: a nested key nobody has classified must reach `OMITTED`
#: or stop the run, never fall through.
ALLOWED: dict[str, dict[str, str]] = {
    "npa-workflow/status.json": {
        "schema_version": TEXT,
        "status": TEXT,
        "step_count": NUM,
        "workflow": TEXT,
    },
    "npa-workflow/manifest.json": {
        "api_version": TEXT,
        "schema_version": TEXT,
        "status": TEXT,
        "workflow": TEXT,
        "steps[].inputs[].schema": TEXT,
        "steps[].iteration": COUNT_OR_NULL,
        "steps[].outputs[].schema": TEXT,
        "steps[].resources": TEXT,
        "steps[].state": TEXT,
        "steps[].status": TEXT,
        "steps[].tool_ref": TEXT,
    },
    "prepared/manifest.json": {
        "demo_data_release": TEXT,
        "distance_factor": NUM,
        "feature_radius_factor": NUM,
        "fragments[].bytes": NUM,
        "fragments[].id": TEXT,
        "fragments[].sha256": HASH,
        "icp_distance_factor": NUM,
        "icp_estimation": TEXT,
        "normal_radius_factor": NUM,
        "random_seed": NUM,
        "ransac_confidence": NUM,
        "ransac_max_iteration": NUM,
        "schema_version": TEXT,
        "voxel_size": NUM,
    },
    "pairs/result.json": {
        "engine": TEXT,
        "fragment_ids[]": TEXT,
        "icp_estimation": TEXT,
        "journal_sha256": HASH,
        "kind": TEXT,
        "manifest_sha256": HASH,
        "schema_version": TEXT,
        "source_version": TEXT,
        "subprocess_wall_seconds": NUM,
        "voxel_size": NUM,
        **_SUMMARY,
    },
    "validation/validation.json": {
        "journal_sha256": HASH,
        "kind": TEXT,
        "pair_count": NUM,
        "schema_version": TEXT,
        "valid": FLAG,
        **_SUMMARY,
    },
    "graph/result.json": {
        "engine": TEXT,
        "fragment_ids[]": TEXT,
        "icp_estimation": TEXT,
        "journal_sha256": HASH,
        "kind": TEXT,
        "manifest_sha256": HASH,
        "schema_version": TEXT,
        "source_version": TEXT,
        "subprocess_wall_seconds": NUM,
        "voxel_size": NUM,
        **_SUMMARY,
        **_pose_graph("pose_graph."),
    },
    "graph/pose_graph.json": _pose_graph(),
    "surface/result.json": {
        "cloud_extent[]": NUM,
        "coverage.fraction_within_two_voxels": NUM,
        "coverage.fraction_within_voxel": NUM,
        "coverage.sample_to_surface_p95": NUM,
        "coverage.sample_to_surface_rmse": NUM,
        "coverage.samples": NUM,
        "crop_justification.note": NOTE,
        "crop_justification.removed_surface_reads_as": NOTE,
        "crop_justification.unsupported_area_share_beyond_3_voxels": NUM,
        "crop_justification.vertex_fraction_removed": NUM,
        "crop_justification.vertices_removed": NUM,
        "density_quantile": NUM,
        "input_points": NUM,
        "low_density_vertices_removed": NUM,
        "poisson_depth": NUM,
        "schema_version": TEXT,
        "support_distance_factor": NUM,
        "unsupported_vertices_removed": NUM,
        **_mesh("mesh"),
        **_mesh("mesh_uncropped"),
        **_support("support"),
        **_support("support_before_crop"),
    },
    "reports/rrd-manifest.json": {
        "bytes": NUM,
        "fused_points": NUM,
        "logged_fragments[].decimation_step": NUM,
        "logged_fragments[].fragment_id": TEXT,
        "logged_fragments[].logged_points": NUM,
        "logged_fragments[].source_points": NUM,
        "mesh.triangle_count": NUM,
        "mesh.vertex_count": NUM,
        "schema_version": TEXT,
        "sha256": HASH,
        "unsupported_triangles_shown": NUM,
        "up_axis_inference.angle_to_snapped_axis_degrees": NUM,
        "up_axis_inference.measured_plane_normal[]": NUM,
        "up_axis_inference.plane_inlier_fraction": NUM,
        "up_axis_inference.plane_inliers": NUM,
        "up_axis_inference.source": TEXT,
        "up_axis_inference.up[]": NUM,
        **_camera("camera"),
        **_camera("view_cameras.*"),
    },
}

#: Longest string a `TEXT` leaf may hold. Schema strings here are short names; a long
#: one in that slot is something else, and something else has not been classified.
TEXT_LIMIT = 64
HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")

#: Every field deliberately left out, and the category it is left out as. Patterns are
#: matched against the same normalized paths as `ALLOWED`, across all artifacts.
OMITTED: tuple[tuple[str, str], ...] = (
    ("run_id", "run identity"),
    ("validated_run_id", "run identity"),
    ("run_prefix_uri", "object-store location"),
    ("input_path", "object-store location"),
    ("registration_path", "object-store location"),
    ("aligned_uris.*", "object-store location"),
    ("fragments[].uri", "object-store location"),
    ("steps[].inputs[].uri", "object-store location"),
    ("steps[].outputs[].uri", "object-store location"),
    ("sky_job_id", "scheduler job identity"),
    ("steps[].resources_profile.**", "resolved execution profile"),
    ("updated_at", "run timestamp"),
)


def _matches(pattern: str, path: str) -> bool:
    expected = pattern.split(".")
    actual = path.split(".")
    if expected[-1] == "**":
        expected = expected[:-1]
        if len(actual) <= len(expected):
            return False
        actual = actual[: len(expected)]
    elif len(expected) != len(actual):
        return False
    return all(e in ("*", a) for e, a in zip(expected, actual))


def _holds(kind: str, value: object) -> bool:
    if kind == NUM:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == COUNT_OR_NULL:
        return value is None or (isinstance(value, int) and not isinstance(value, bool))
    if kind == FLAG:
        return isinstance(value, bool)
    if kind == HASH:
        return isinstance(value, str) and bool(HEX64.match(value))
    if kind == TEXT:
        return isinstance(value, str) and len(value) <= TEXT_LIMIT
    if kind == NOTE:
        return isinstance(value, str)
    raise AssertionError(f"unknown value kind {kind}")


def _classify(path: str, value: object, allowed: dict[str, str]) -> str | None:
    """Return None to keep the field, or the category it is omitted as.

    Raises rather than guessing in both directions that matter: a leaf nobody has
    classified, and a leaf whose value is not the kind its name was allowed for.
    """

    for pattern, kind in allowed.items():
        if not _matches(pattern, path):
            continue
        if _holds(kind, value):
            return None
        raise SystemExit(
            f"{path} is allowed as {kind} but holds {type(value).__name__} "
            f"{value!r:.60}. Classify what this now is before publishing this run."
        )
    for pattern, category in OMITTED:
        if _matches(pattern, path):
            return category
    raise SystemExit(
        f"{path} is in neither the allowlist nor the omitted categories. "
        "Classify it before publishing this run."
    )


def select(node: object, allowed: dict[str, str], removed: Counter, where: str = ""):
    """Copy `node`, keeping only allowlisted leaves and tallying what was dropped."""

    if isinstance(node, dict):
        kept = {}
        for key, value in node.items():
            here = f"{where}.{key}" if where else key
            child = select(value, allowed, removed, here)
            if child is not _DROPPED:
                kept[key] = child
        return kept or _DROPPED
    if isinstance(node, list):
        kept_items = [
            item
            for item in (select(v, allowed, removed, f"{where}[]") for v in node)
            if item is not _DROPPED
        ]
        return kept_items or _DROPPED
    category = _classify(where, node, allowed)
    if category is None:
        return node
    removed[category] += 1
    return _DROPPED


class _Dropped:
    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return "<dropped>"


_DROPPED = _Dropped()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(artifacts: Path) -> dict:
    removed: Counter = Counter()
    omitted_keys: dict[str, set[str]] = {}
    derived: dict[str, object] = {}
    originals: list[dict[str, object]] = []

    for name in ARTIFACTS:
        source = artifacts / name
        if not source.is_file():
            raise SystemExit(f"missing run artifact: {source}")
        data = json.loads(source.read_text())
        before = Counter(removed)
        kept = select(data, ALLOWED[name], removed)
        derived[name] = {} if kept is _DROPPED else kept
        originals.append({"artifact": name, "original_sha256": digest(source)})
        for category in removed - before:
            omitted_keys.setdefault(category, set()).add(name)

    return {
        "what_this_is": (
            "A derived, publishable view of one Open3D workflow run's JSON artifacts. "
            "It is not a copy of them: fields were selected by allowlist, not redacted."
        ),
        "generator": "harness/derive_public_run_report.py",
        "selection_rule": (
            "Deny by default. A field appears here only because a pattern in the "
            "generator's ALLOWED table names it. An unclassified field stops the "
            "generator rather than defaulting either way."
        ),
        "omitted_field_categories": {
            category: {
                "fields_dropped": removed[category],
                "artifacts_affected": sorted(omitted_keys[category]),
            }
            for category in sorted(removed)
        },
        "no_substitutes": (
            "Nothing was put in place of an omitted field. There is no placeholder "
            "bucket, path or run identifier here to be mistaken for a real one."
        ),
        "retained_originals": {
            "where": (
                "The unmodified artifacts are retained in access-controlled lane "
                "evidence, not in this repository."
            ),
            "hashes_are_of_the_originals": (
                "Each sha256 below is of the artifact as the run wrote it, so it "
                "identifies those bytes and deliberately matches no file here. The "
                "hash of this derived file is recorded separately, in "
                "capability-record-workflow-replay.json."
            ),
            "artifacts": originals,
        },
        "derived_fields": derived,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts", required=True, type=Path, help="directory holding the run's artifacts"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    report = build(args.artifacts)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n")
    dropped = sum(
        entry["fields_dropped"] for entry in report["omitted_field_categories"].values()
    )
    print(
        f"{args.output}: {len(ARTIFACTS)} artifact(s) selected from, "
        f"{dropped} field(s) omitted across "
        f"{len(report['omitted_field_categories'])} categor(ies)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
