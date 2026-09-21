"""Validate factual planner journals and derive reviewable Rerun recordings."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np

from .benchmark_inventory import benchmark_identities
from .schemas import DATASET_REVISION, SOURCE_REVISION


class CuroboError(RuntimeError):
    """A cuRobo operation failed without a synthetic fallback."""


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep every input in the full denominator; eligible rate is separate."""
    if not rows:
        raise CuroboError("planner journal contains no problems")
    identities = [(r["mode"], r["dataset"], r["problem_id"]) for r in rows]
    if len(set(identities)) != len(rows):
        raise CuroboError("duplicate problem identity")
    groups: dict[str, Any] = {}
    for mode in sorted({r["mode"] for r in rows}):
        subset = [r for r in rows if r["mode"] == mode]
        counts = {
            status: sum(r["status"] == status for r in subset)
            for status in ("success", "failed", "invalid")
        }
        if sum(counts.values()) != len(subset):
            raise CuroboError("unknown problem status")
        eligible = counts["success"] + counts["failed"]
        metrics: dict[str, Any] = {}
        for row in subset:
            if row["status"] == "success":
                validate_trajectory(row["trajectory"])
            elif "trajectory" in row:
                raise CuroboError("unsolved problem cannot carry a solution trajectory")
            for key, value in row.get("metrics", {}).items():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (float, int))
                    or not math.isfinite(value)
                ):
                    raise CuroboError("metrics must contain finite numbers")
                metrics.setdefault(key, []).append(value)
        groups[mode] = {
            "input_count": len(subset),
            "eligible_count": eligible,
            **counts,
            "success_fraction_all": counts["success"] / len(subset),
            "success_fraction_eligible": counts["success"] / eligible
            if eligible
            else None,
            "metrics": {
                k: {
                    "count": len(v),
                    "mean": float(np.mean(v)),
                    "p50": float(np.percentile(v, 50)),
                    "p95": float(np.percentile(v, 95)),
                    "p98": float(np.percentile(v, 98)),
                }
                for k, v in metrics.items()
            },
        }
    return groups


def validate_report(
    report: dict[str, Any], rows: list[dict[str, Any]], *, run_id: str
) -> None:
    """Validate published facts against source inventory and runner output contracts."""
    try:
        summary = summarize(rows)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise CuroboError("invalid planner journal fields") from exc
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != "npa.curobo.result.v1"
        or report.get("engine") != "nvidia-curobo-v2"
        or report.get("source_revision") != SOURCE_REVISION
        or report.get("run_id") != run_id
        or report.get("kind") not in {"benchmark", "plan"}
        or report.get("summary") != summary
    ):
        raise CuroboError(
            "result schema, identity or summary does not match the journal"
        )
    if report["kind"] == "benchmark":
        modes = report.get("requested_modes", [])
        if (
            not isinstance(modes, list)
            or not modes
            or not all(isinstance(mode, str) for mode in modes)
            or len(modes) != len(set(modes))
            or not set(modes) <= {"kinematic", "dynamics"}
            or set(report["summary"]) != set(modes)
            or report.get("dataset_revision") != DATASET_REVISION
        ):
            raise CuroboError("benchmark recipe or dataset identity mismatch")
        expected = benchmark_identities(modes)
        observed = {(r["mode"], r["dataset"], r["problem_id"]) for r in rows}
        if observed != set(expected):
            raise CuroboError("incomplete benchmark or unexpected problem identities")
        for row in rows:
            identity = (row["mode"], row["dataset"], row["problem_id"])
            if (row["status"] == "invalid") != expected[identity]:
                raise CuroboError("benchmark exclusions disagree with pinned inputs")
    else:
        if (
            report.get("requested_modes") != ["kinematic"]
            or report.get("dataset_revision") is not None
            or any(r["mode"] != "kinematic" or r["dataset"] != "operator" for r in rows)
        ):
            raise CuroboError("plan report must use operator/kinematic identities")
        if any(r["status"] == "invalid" for r in rows):
            raise CuroboError("operator plan cannot exclude a validated input problem")
    for row in rows:
        _validate_planner_metrics(row, kind=report["kind"])


def _validate_query(row: dict[str, Any]) -> None:
    if row["status"] == "invalid":
        if "query" in row:
            raise CuroboError("excluded benchmark input cannot carry an executed query")
        return
    query = row.get("query")
    if not isinstance(query, dict) or set(query) != {"start", "goal_pose"}:
        raise CuroboError("planner row lacks the executed start and goal query")
    start = np.asarray(query["start"], dtype=float)
    goal = query["goal_pose"]
    if (
        start.ndim != 1
        or not len(start)
        or not np.isfinite(start).all()
        or not isinstance(goal, dict)
        or set(goal) != {"position_xyz", "quaternion_wxyz"}
    ):
        raise CuroboError("planner query shape or values are invalid")
    position = np.asarray(goal["position_xyz"], dtype=float)
    quaternion = np.asarray(goal["quaternion_wxyz"], dtype=float)
    if (
        position.shape != (3,)
        or quaternion.shape != (4,)
        or not np.isfinite(position).all()
        or not np.isfinite(quaternion).all()
        or not math.isclose(
            float(np.linalg.norm(quaternion)), 1.0, rel_tol=1e-6, abs_tol=1e-6
        )
    ):
        raise CuroboError("planner goal pose is invalid")


def _validate_planner_metrics(row: dict[str, Any], *, kind: str) -> None:
    _validate_query(row)
    metrics = row.get("metrics", {})
    expected = set()
    if row["status"] in {"success", "failed"}:
        expected.add("wall_plan_seconds")
    if row["status"] == "success":
        expected.update(
            {
                "planner_total_seconds",
                "solver_seconds",
                "position_error_m",
                "rotation_error_rad",
                "joint_path_length_rad",
                "tool_path_length_m",
                "trajectory_duration_seconds",
                "max_abs_jerk_rad_s3",
            }
        )
        if kind == "benchmark":
            expected.update({"energy_proxy_j", "max_torque_nm", "torque_violation"})
    if not isinstance(metrics, dict) or set(metrics) != expected:
        raise CuroboError("planner metrics do not match the known problem status")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in metrics.values()
    ):
        raise CuroboError("planner metrics must be nonnegative finite numbers")
    if "torque_violation" in metrics and metrics["torque_violation"] not in (0, 1):
        raise CuroboError("torque violation must be a zero/one indicator")
    if row["status"] == "success":
        trajectory = row["trajectory"]
        duration = (len(trajectory["position"]) - 1) * trajectory["dt"]
        if metrics["trajectory_duration_seconds"] <= 0 or not math.isclose(
            metrics["trajectory_duration_seconds"], duration, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise CuroboError("trajectory duration does not match its sample timeline")
    if kind == "benchmark" and row["status"] == "success":
        _validate_dynamics_evidence(row)
    elif "dynamics_evidence" in row:
        raise CuroboError("only solved benchmark rows can carry dynamics evidence")


def validate_joint_series(value: dict[str, Any]) -> np.ndarray:
    names = value["joint_names"]
    position = np.asarray(value["position"], dtype=float)
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(names) != len(set(names))
    ):
        raise CuroboError("joint names must be unique nonempty strings")
    if position.ndim != 2 or len(position) < 2 or position.shape[1] != len(names):
        raise CuroboError("trajectory requires at least two aligned joint samples")
    if (
        isinstance(value["dt"], bool)
        or not isinstance(value["dt"], (int, float))
        or not math.isfinite(value["dt"])
        or value["dt"] <= 0
    ):
        raise CuroboError("trajectory dt must be finite and positive")
    for field in ("position", "velocity", "acceleration", "jerk"):
        array = np.asarray(value[field], dtype=float)
        if array.shape != position.shape or not np.isfinite(array).all():
            raise CuroboError(f"invalid {field} trajectory shape or nonfinite samples")
    return position


def validate_trajectory(value: dict[str, Any]) -> None:
    position = validate_joint_series(value)
    tool = np.asarray(value["tool_position"], dtype=float)
    quaternion = np.asarray(value["tool_quaternion"], dtype=float)
    if (
        tool.shape != (len(position), 3)
        or quaternion.shape != (len(position), 4)
        or not np.isfinite(tool).all()
        or not np.isfinite(quaternion).all()
        or not np.allclose(np.linalg.norm(quaternion, axis=1), 1.0, atol=1e-5)
    ):
        raise CuroboError(
            "tool poses must align with actual normalized FK joint samples"
        )


def _validate_dynamics_evidence(row: dict[str, Any]) -> None:
    evidence = row.get("dynamics_evidence")
    if not isinstance(evidence, dict) or set(evidence) != {
        "attached_mass_kg",
        "trajectory",
        "torques_nm",
        "torque_limits_nm",
    }:
        raise CuroboError("solved benchmark row lacks complete dynamics evidence")
    expected_mass = 3.0 if row["mode"] == "dynamics" else 0.0
    if evidence["attached_mass_kg"] != expected_mass:
        raise CuroboError(
            "dynamics evidence payload mass disagrees with benchmark mode"
        )
    trajectory = evidence["trajectory"]
    positions = validate_joint_series(trajectory)
    velocities = np.asarray(trajectory["velocity"], dtype=float)
    torques = np.asarray(evidence["torques_nm"], dtype=float)
    limits = np.asarray(evidence["torque_limits_nm"], dtype=float)
    dynamics_names = trajectory["joint_names"]
    retained_names = row["trajectory"]["joint_names"]
    expected_limits = np.asarray([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
    if (
        positions.shape[1] != 7
        or not set(dynamics_names).issubset(retained_names)
        or torques.shape != velocities.shape
        or limits.shape != (positions.shape[1],)
        or not np.isfinite(torques).all()
        or not np.isfinite(limits).all()
        or not np.array_equal(limits, expected_limits)
    ):
        raise CuroboError(
            "inverse-dynamics evidence shape or torque limits are invalid"
        )
    energy = float(np.abs(torques * velocities).sum() * trajectory["dt"])
    max_torque = float(np.abs(torques).max())
    violation = int(np.any(np.abs(torques).max(axis=0) > limits))
    metrics = row["metrics"]
    if (
        not math.isclose(metrics["energy_proxy_j"], energy, rel_tol=1e-9, abs_tol=1e-9)
        or not math.isclose(
            metrics["max_torque_nm"], max_torque, rel_tol=1e-9, abs_tol=1e-9
        )
        or metrics["torque_violation"] != violation
    ):
        raise CuroboError("reported dynamics metrics do not match retained evidence")


def read_journal(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        summarize(rows)
        return rows
    except (ValueError, KeyError, TypeError) as exc:
        raise CuroboError("invalid planner journal") from exc


_RRD_LAYOUT = "npa.curobo.problem-index.v1"
_RRD_PROBLEM_ROOT = "problems"
_RRD_TRAJECTORY_ROOT = "trajectory"


def log_trajectory_columns(
    recording, root: str, trajectory: dict[str, Any], *, problem_index: int
) -> None:
    """Send one genuine Rerun column batch per entity, preserving every sample."""
    import rerun as rr

    frames = len(trajectory["position"])
    indexes = [
        rr.TimeColumn("trajectory_time", duration=np.arange(frames) * trajectory["dt"]),
        rr.TimeColumn(
            "problem_index", sequence=np.full(frames, problem_index, dtype=np.int64)
        ),
    ]
    recording.send_columns(
        root + "/tool",
        indexes=indexes,
        columns=rr.Points3D.columns(positions=trajectory["tool_position"]),
        strict=True,
    )
    quaternions = np.asarray(trajectory["tool_quaternion"], dtype=float)
    for component, name in enumerate(("w", "x", "y", "z")):
        recording.send_columns(
            f"{root}/tool_quaternion/{name}",
            indexes=indexes,
            columns=rr.Scalars.columns(scalars=quaternions[:, component]),
            strict=True,
        )
    for field in ("position", "velocity", "acceleration", "jerk"):
        values = np.asarray(trajectory[field], dtype=float)
        for joint in range(values.shape[1]):
            recording.send_columns(
                f"{root}/joints/{joint}/{field}",
                indexes=indexes,
                columns=rr.Scalars.columns(scalars=values[:, joint]),
                strict=True,
            )


def _rrd_provenance(
    rows: list[dict[str, Any]], journal: Path, *, run_id: str
) -> dict[str, Any]:
    return {
        "producer": "npa.workbench.curobo",
        "source_revision": SOURCE_REVISION,
        "dataset_revision": DATASET_REVISION
        if any(row["dataset"] != "operator" for row in rows)
        else None,
        "run_id": run_id,
        "journal_sha256": hashlib.sha256(journal.read_bytes()).hexdigest(),
        "rrd_layout": _RRD_LAYOUT,
        "limitations": "FK tool paths and joint traces; no rendered robot meshes or independent collision certification.",
    }


def _log_rrd_problem(recording, row: dict[str, Any], *, problem_index: int) -> None:
    import rerun as rr

    recording.set_time("problem_index", sequence=problem_index)
    status = {key: row[key] for key in ("problem_id", "mode", "dataset", "status")}
    recording.log(
        f"{_RRD_PROBLEM_ROOT}/status", rr.TextDocument(json.dumps(status))
    )
    if row["status"] != "invalid":
        recording.log(
            f"{_RRD_PROBLEM_ROOT}/goal",
            rr.Points3D(
                [row["query"]["goal_pose"]["position_xyz"]],
                radii=0.015,
                colors=[0, 255, 0],
            ),
        )
    for name, value in row.get("metrics", {}).items():
        recording.log(f"metrics/{name}", rr.Scalars(value))
    if row["status"] != "success":
        return
    trajectory = row["trajectory"]
    recording.log(
        f"{_RRD_TRAJECTORY_ROOT}/joint_names",
        rr.TextDocument(json.dumps(trajectory["joint_names"])),
    )
    recording.log(
        f"{_RRD_TRAJECTORY_ROOT}/tool_path",
        rr.LineStrips3D([trajectory["tool_position"]]),
    )
    log_trajectory_columns(
        recording, _RRD_TRAJECTORY_ROOT, trajectory, problem_index=problem_index
    )


def _rrd_manifest(
    rows: list[dict[str, Any]], journal: Path, output: Path, *, run_id: str
) -> dict[str, Any]:
    status_counts = {
        status: sum(row["status"] == status for row in rows)
        for status in ("success", "failed", "invalid")
    }
    journal_sha256 = hashlib.sha256(journal.read_bytes()).hexdigest()
    rrd_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    expected_chunks = _expected_rrd_chunks(rows)
    return {
        "schema_version": "npa.curobo.rrd-manifest.v1",
        "run_id": run_id,
        "source_revision": SOURCE_REVISION,
        "dataset_revision": DATASET_REVISION
        if any(row["dataset"] != "operator" for row in rows)
        else None,
        "journal_sha256": journal_sha256,
        "rrd_layout": _RRD_LAYOUT,
        "entity_path_count": len(expected_chunks),
        "problem_count": len(rows),
        "status_counts": status_counts,
        "successful_trajectories": status_counts["success"],
        "status_entities": len(rows),
        "goal_markers": sum(row["status"] != "invalid" for row in rows),
        "metric_samples": sum(len(row.get("metrics", {})) for row in rows),
        "trajectory_samples": sum(
            len(row["trajectory"]["position"])
            for row in rows
            if row["status"] == "success"
        ),
        "sha256": rrd_sha256,
        "rrd_sha256": rrd_sha256,
    }


def build_rrd(journal: Path, output: Path, *, run_id: str) -> dict[str, Any]:
    """Log actual joint, FK and problem-status facts, without robot-mesh claims."""
    import rerun as rr

    rows = read_journal(journal)
    recording = rr.RecordingStream("npa.curobo", recording_id=run_id)
    recording.save(str(output))
    try:
        recording.log(
            "provenance",
            rr.TextDocument(json.dumps(_rrd_provenance(rows, journal, run_id=run_id))),
            static=True,
        )
        for index, row in enumerate(rows):
            _log_rrd_problem(recording, row, problem_index=index)
    finally:
        recording.flush()
        del recording
    if not output.is_file() or output.stat().st_size == 0:
        raise CuroboError("RRD writer produced no bytes")
    return _rrd_manifest(rows, journal, output, run_id=run_id)


_RRD_CHUNK_HEADER = re.compile(
    rb"Chunk\([^)]*\) with ([0-9]+) rows \([^)]*\) - /([^ ]+) -"
)


def _add_expected_rows(
    expected: dict[str, int], entity_path: str, row_count: int = 1
) -> None:
    expected[entity_path] = expected.get(entity_path, 0) + row_count


def _expected_rrd_chunks(rows: list[dict[str, Any]]) -> dict[str, int]:
    expected: dict[str, int] = {"provenance": 1}
    for row in rows:
        _add_expected_rows(expected, f"{_RRD_PROBLEM_ROOT}/status")
        if row["status"] != "invalid":
            _add_expected_rows(expected, f"{_RRD_PROBLEM_ROOT}/goal")
        for name in row.get("metrics", {}):
            _add_expected_rows(expected, f"metrics/{name}")
        if row["status"] != "success":
            continue
        trajectory = row["trajectory"]
        samples = len(trajectory["position"])
        _add_expected_rows(expected, f"{_RRD_TRAJECTORY_ROOT}/joint_names")
        _add_expected_rows(expected, f"{_RRD_TRAJECTORY_ROOT}/tool_path")
        _add_expected_rows(expected, f"{_RRD_TRAJECTORY_ROOT}/tool", samples)
        for component in ("w", "x", "y", "z"):
            _add_expected_rows(
                expected,
                f"{_RRD_TRAJECTORY_ROOT}/tool_quaternion/{component}",
                samples,
            )
        for joint in range(len(trajectory["joint_names"])):
            for field in ("position", "velocity", "acceleration", "jerk"):
                _add_expected_rows(
                    expected,
                    f"{_RRD_TRAJECTORY_ROOT}/joints/{joint}/{field}",
                    samples,
                )
    return expected


def _scan_decoded_chunks(
    path: Path, expected: dict[str, int], *, run_id: str
) -> tuple[str, int, list[str]]:
    digest = hashlib.sha256()
    observed = {entity: 0 for entity in expected}
    identities = {b"npa.curobo": False, run_id.encode(): False}
    size = 0
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            size += len(line)
            for marker in identities:
                if marker in line:
                    identities[marker] = True
            match = _RRD_CHUNK_HEADER.search(line)
            if match:
                entity = match.group(2).decode()
                if entity in observed:
                    observed[entity] += int(match.group(1))
    mismatches = [
        f"{entity}: expected {count}, decoded {observed[entity]}"
        for entity, count in expected.items()
        if observed[entity] != count
    ]
    mismatches.extend(
        f"recording identity {marker.decode()} is absent"
        for marker, seen in identities.items()
        if not seen
    )
    return digest.hexdigest(), size, sorted(mismatches)


def _require_rrd_command(command: list[str]) -> None:
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode:
        raise CuroboError("Rerun could not normalize factual recording")


def _normalize_rrd_for_compare(
    source: Path, output: Path, *, rerun: str
) -> None:
    filtered_output = output.with_name(output.stem + "-filtered.rrd")
    _require_rrd_command(
        [
            rerun,
            "rrd",
            "filter",
            "--drop-timeline",
            "log_tick",
            "--drop-timeline",
            "log_time",
            "--output",
            str(filtered_output),
            str(source),
        ]
    )
    _require_rrd_command(
        [
            rerun,
            "rrd",
            "compact",
            "--max-rows",
            "1000000",
            "--max-rows-if-unsorted",
            "1000000",
            "--max-bytes",
            "134217728",
            "--output",
            str(output),
            str(filtered_output),
        ]
    )
    filtered_output.unlink()


def _semantic_compare_rrd(
    path: Path, *, rows: list[dict[str, Any]], run_id: str, rerun: str
) -> str:
    """Regenerate journal-derived facts and compare every decoded value/timeline."""
    with tempfile.TemporaryDirectory(prefix="npa-curobo-rrd-compare-") as directory:
        root = Path(directory)
        journal = root / "expected.jsonl"
        journal.write_bytes(b"".join(canonical(row) + b"\n" for row in rows))
        expected = root / "expected.rrd"
        build_rrd(journal, expected, run_id=run_id)
        normalized = []
        for name, source in (("observed", path), ("expected", expected)):
            output = root / f"{name}-normalized.rrd"
            _normalize_rrd_for_compare(source, output, rerun=rerun)
            normalized.append(output)
        compared = subprocess.run(
            [rerun, "rrd", "compare", "--unordered", *(str(p) for p in normalized)],
            capture_output=True,
            check=False,
        )
        if compared.returncode:
            raise CuroboError(
                "decoded RRD values or factual timelines differ from journal"
            )
        return hashlib.sha256(expected.read_bytes()).hexdigest()


def _decode_rrd_to(
    path: Path,
    decoded_output: Path,
    *,
    rows: list[dict[str, Any]],
    run_id: str,
) -> dict[str, Any]:
    sibling = Path(sys.executable).with_name("rerun")
    rerun = str(sibling) if sibling.is_file() else shutil.which("rerun")
    if not rerun:
        raise CuroboError("Rerun CLI is unavailable")
    verified = subprocess.run(
        [rerun, "rrd", "verify", str(path)], capture_output=True, check=False
    )
    if verified.returncode:
        raise CuroboError("Rerun rejected the generated recording")
    with decoded_output.open("wb") as stream:
        printed = subprocess.run(
            [rerun, "rrd", "print", "-vv", str(path)],
            stdout=stream,
            stderr=subprocess.PIPE,
            check=False,
        )
    if printed.returncode:
        raise CuroboError("Rerun could not decode the generated recording")
    expected = _expected_rrd_chunks(rows)
    digest, size, mismatches = _scan_decoded_chunks(
        decoded_output, expected, run_id=run_id
    )
    if mismatches:
        raise CuroboError("decoded RRD omits required factual coverage")
    reference_sha256 = _semantic_compare_rrd(
        path, rows=rows, run_id=run_id, rerun=rerun
    )
    return {
        "verify": "passed",
        "print": "passed",
        "semantic_compare": "passed",
        "semantic_reference_sha256": reference_sha256,
        "rrd_layout": _RRD_LAYOUT,
        "print_sha256": digest,
        "print_bytes": size,
        "required_chunk_count": len(expected),
        "decoded_chunk_rows": sum(expected.values()),
        "status_entities": len(rows),
        "goal_entities": sum(row["status"] != "invalid" for row in rows),
        "metric_samples": sum(len(row.get("metrics", {})) for row in rows),
        "trajectory_entities": sum(row["status"] == "success" for row in rows),
        "trajectory_samples": sum(
            len(row["trajectory"]["position"])
            for row in rows
            if row["status"] == "success"
        ),
        "chunk_mismatches": [],
    }


def decode_rrd(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    run_id: str,
    decoded_output: Path | None = None,
) -> dict[str, Any]:
    """Verify and fully decode an RRD while bounding in-memory evidence."""
    if decoded_output is not None:
        return _decode_rrd_to(path, decoded_output, rows=rows, run_id=run_id)
    with tempfile.TemporaryDirectory(prefix="npa-curobo-rrd-decode-") as directory:
        return _decode_rrd_to(
            path, Path(directory) / "rrd-print.txt", rows=rows, run_id=run_id
        )
