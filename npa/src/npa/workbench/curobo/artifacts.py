"""Validate factual planner journals and derive reviewable Rerun recordings."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
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


def _validate_planner_metrics(row: dict[str, Any], *, kind: str) -> None:
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


def validate_trajectory(value: dict[str, Any]) -> None:
    names = value["joint_names"]
    position = np.asarray(value["position"], dtype=float)
    if position.ndim != 2 or len(position) < 2 or position.shape[1] != len(names):
        raise CuroboError("trajectory requires at least two aligned joint samples")
    if len(set(names)) != len(names) or not names:
        raise CuroboError("joint names must be unique")
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
    tool = np.asarray(value["tool_position"], dtype=float)
    if tool.shape != (len(position), 3) or not np.isfinite(tool).all():
        raise CuroboError("tool positions must align with actual FK joint samples")


def read_journal(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        summarize(rows)
        return rows
    except (ValueError, KeyError, TypeError) as exc:
        raise CuroboError("invalid planner journal") from exc


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
    for field in ("position", "velocity", "acceleration", "jerk"):
        values = np.asarray(trajectory[field], dtype=float)
        for joint in range(values.shape[1]):
            recording.send_columns(
                f"{root}/joints/{joint}/{field}",
                indexes=indexes,
                columns=rr.Scalars.columns(scalars=values[:, joint]),
                strict=True,
            )


def build_rrd(journal: Path, output: Path, *, run_id: str) -> dict[str, Any]:
    """Log actual joint, FK and problem-status facts, without robot-mesh claims."""
    import rerun as rr

    rows = read_journal(journal)
    recording = rr.RecordingStream("npa.curobo", recording_id=run_id)
    recording.save(str(output))
    try:
        recording.log(
            "provenance",
            rr.TextDocument(
                json.dumps(
                    {
                        "producer": "npa.workbench.curobo",
                        "source_revision": SOURCE_REVISION,
                        "dataset_revision": DATASET_REVISION
                        if any(r["dataset"] != "operator" for r in rows)
                        else None,
                        "run_id": run_id,
                        "journal_sha256": hashlib.sha256(
                            journal.read_bytes()
                        ).hexdigest(),
                        "limitations": "FK tool paths and joint traces; no rendered robot meshes or independent collision certification.",
                    }
                )
            ),
            static=True,
        )
        for index, row in enumerate(rows):
            root = f"problems/{index:06d}"
            recording.set_time("problem_index", sequence=index)
            recording.log(
                root + "/status",
                rr.TextDocument(
                    json.dumps(
                        {k: row[k] for k in ("problem_id", "mode", "dataset", "status")}
                    )
                ),
            )
            for name, value in row.get("metrics", {}).items():
                recording.log(f"metrics/{name}", rr.Scalars(value))
            if row["status"] != "success":
                continue
            trajectory = row["trajectory"]
            recording.log(
                root + "/joint_names",
                rr.TextDocument(json.dumps(trajectory["joint_names"])),
            )
            recording.log(
                root + "/tool_path", rr.LineStrips3D([trajectory["tool_position"]])
            )
            log_trajectory_columns(recording, root, trajectory, problem_index=index)
    finally:
        recording.flush()
        del recording
    if not output.is_file() or output.stat().st_size == 0:
        raise CuroboError("RRD writer produced no bytes")
    return {
        "problem_count": len(rows),
        "successful_trajectories": sum(r["status"] == "success" for r in rows),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
