"""Independent success verification for the Sim2Real model-agent benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class VerificationError(RuntimeError):
    """Raised when live artifacts do not prove the benchmark predicate."""


@dataclass(frozen=True)
class LiftEvidence:
    manifest: str
    rollout_id: str
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    minimum_lift_m: float
    samples: int
    checkpoint_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: Any, field: str) -> float:
    if type(value) not in (int, float):
        raise VerificationError(f"{field} must be a finite JSON number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise VerificationError(f"{field} must be a finite JSON number") from exc
    if not math.isfinite(number):
        raise VerificationError(f"{field} must be a finite JSON number")
    return number


def _simulation_step_seconds(manifest: dict[str, Any]) -> float:
    value = manifest.get("simulation_step_seconds")
    if value is None:
        capture = manifest.get("capture", {})
        if not isinstance(capture, dict):
            raise VerificationError("rollout capture must be an object")
        value = capture.get("simulation_step_seconds")
    seconds = _finite_number(value, "simulation_step_seconds")
    if seconds <= 0:
        raise VerificationError("simulation_step_seconds must be positive")
    return seconds


def _sample_step(row: dict[str, Any]) -> int | None:
    value = row.get("sim_step")
    if value is not None and (type(value) is not int or value < 0):
        raise VerificationError("sim_step must be a non-negative JSON integer")
    return value


def _sample_time(row: dict[str, Any], step_seconds: float) -> float:
    for key in ("sim_time_seconds", "timestamp_seconds", "time_seconds"):
        if row.get(key) is not None:
            return _finite_number(row[key], key)
    step = _sample_step(row)
    if step is not None:
        timestamp = _finite_number(step, "sim_step") * step_seconds
        return _finite_number(timestamp, "derived simulation timestamp")
    raise VerificationError(
        "rollout ground truth has no physical timestamp; record sim_time_seconds "
        "or simulation_step_seconds instead of inferring duration from sample count"
    )


def _qualifies(row: dict[str, Any], lift_m: float) -> bool:
    truth = row["simulator_ground_truth"]
    if truth["stable_grasp"] is not True:
        return False
    measured_lift = _finite_number(truth.get("object_lift_m"), "object_lift_m")
    return measured_lift >= lift_m


def _trained_checkpoint(manifest: dict[str, Any]) -> str | None:
    if manifest.get("schema") != "npa.sim2real.action_rollout.v1":
        return None
    if manifest.get("source") != "byo_isaac_policy_rollout":
        return None
    if (
        manifest.get("sim_backend") != "isaac"
        or manifest.get("policy_trained") is not True
    ):
        return None
    checkpoint_sha = str(manifest.get("policy_checkpoint_sha256") or "")
    if len(checkpoint_sha) != 64:
        return None
    size = manifest.get("policy_checkpoint_size_bytes")
    if type(size) is not int or size <= 0:
        return None
    return checkpoint_sha


def _hold_interrupted(
    previous: tuple[float, dict[str, Any]],
    timestamp: float,
    row: dict[str, Any],
    step_seconds: float,
) -> bool:
    delta = timestamp - previous[0]
    if delta <= 0:
        raise VerificationError("rollout timestamps are not strictly increasing")
    previous_step = _sample_step(previous[1])
    current_step = _sample_step(row)
    return (
        delta > step_seconds * 1.5
        or previous_step is None
        or current_step is None
        or current_step != previous_step + 1
    )


def _longest_lift_hold(
    manifest: dict[str, Any], minimum_lift_m: float, step_seconds: float
) -> list[tuple[float, dict[str, Any]]]:
    actions = manifest.get("actions")
    if not isinstance(actions, list):
        raise VerificationError("rollout actions must be an array")

    current: list[tuple[float, dict[str, Any]]] = []
    best: list[tuple[float, dict[str, Any]]] = []
    for row in actions:
        if not isinstance(row, dict):
            return []
        truth = row.get("simulator_ground_truth")
        if not isinstance(truth, dict):
            return []
        stable_grasp = truth.get("stable_grasp")
        if stable_grasp is not True and stable_grasp is not False:
            return []
        _sample_step(row)
        timestamp = _sample_time(row, step_seconds)
        if _qualifies(row, minimum_lift_m):
            if current and _hold_interrupted(current[-1], timestamp, row, step_seconds):
                current = []
            current.append((timestamp, row))
            if not best or current[-1][0] - current[0][0] > best[-1][0] - best[0][0]:
                best = list(current)
        else:
            current = []
    return best


def _lift_evidence(
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    minimum_lift_m: float,
    minimum_hold_seconds: float,
) -> LiftEvidence | None:
    checkpoint_sha = _trained_checkpoint(manifest)
    if checkpoint_sha is None:
        return None
    best = _longest_lift_hold(
        manifest, minimum_lift_m, _simulation_step_seconds(manifest)
    )
    if len(best) < 2 or best[-1][0] - best[0][0] < minimum_hold_seconds:
        return None
    lifts = [
        _finite_number(row["simulator_ground_truth"]["object_lift_m"], "object_lift_m")
        for _, row in best
    ]
    return LiftEvidence(
        manifest=str(manifest_path),
        rollout_id=str(manifest.get("rollout_id") or ""),
        start_seconds=best[0][0],
        end_seconds=best[-1][0],
        duration_seconds=best[-1][0] - best[0][0],
        minimum_lift_m=min(lifts),
        samples=len(best),
        checkpoint_sha256=checkpoint_sha,
    )


def _verify_mcap(path: Path) -> dict[str, Any]:
    from mcap.reader import make_reader

    with path.open("rb") as handle:
        summary = make_reader(handle).get_summary()
    if summary is None or not summary.channels or not summary.statistics:
        raise VerificationError(f"MCAP has no decodable channels/statistics: {path}")
    if int(summary.statistics.message_count or 0) <= 0:
        raise VerificationError(f"MCAP contains no messages: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "channel_count": len(summary.channels),
        "message_count": int(summary.statistics.message_count),
    }


def _verify_rrd(path: Path, *, rerun_bin: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise VerificationError(f"Rerun recording is empty or missing: {path}")
    completed = subprocess.run(
        [rerun_bin, "rrd", "print", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise VerificationError(
            f"Rerun could not decode {path}: {completed.stderr.strip()[:500]}"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def verify_artifact_tree(
    artifact_root: Path,
    *,
    minimum_lift_m: float = 0.05,
    minimum_hold_seconds: float = 2.0,
    rerun_bin: str = "rerun",
) -> dict[str, Any]:
    """Verify real lift/hold evidence and independently decode final recordings."""

    root = artifact_root.resolve()
    if not root.is_dir():
        raise VerificationError(f"artifact root is not a directory: {root}")
    evidence: list[LiftEvidence] = []
    parse_errors: list[str] = []
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != "npa.sim2real.action_rollout.v1"
        ):
            continue
        try:
            found = _lift_evidence(
                path,
                payload,
                minimum_lift_m=minimum_lift_m,
                minimum_hold_seconds=minimum_hold_seconds,
            )
        except VerificationError as exc:
            parse_errors.append(f"{path}: {exc}")
            continue
        if found:
            evidence.append(found)
    if not evidence:
        detail = (
            f" Invalid rollout evidence: {'; '.join(parse_errors)}"
            if parse_errors
            else ""
        )
        raise VerificationError(
            f"no trained real-Isaac rollout proves stable grasp, >= {minimum_lift_m:.3f} m "
            f"lift, and >= {minimum_hold_seconds:.3f} s continuous hold.{detail}"
        )

    mcaps = sorted(root.rglob("sim2real.mcap"))
    rrds = sorted(root.rglob("sim2real.rrd"))
    if not mcaps or not rrds:
        raise VerificationError(
            "final sim2real.mcap and sim2real.rrd are both required"
        )
    mcap = _verify_mcap(mcaps[-1])
    rrd = _verify_rrd(rrds[-1], rerun_bin=rerun_bin)
    strongest = max(
        evidence, key=lambda item: (item.duration_seconds, item.minimum_lift_m)
    )
    return {
        "schema": "npa.sim2real.model_agent_benchmark.success.v1",
        "passed": True,
        "predicate": {
            "minimum_lift_m": minimum_lift_m,
            "minimum_hold_seconds": minimum_hold_seconds,
            "stable_grasp_required": True,
            "trained_policy_required": True,
            "sim_backend": "isaac",
        },
        "strongest_lift_evidence": asdict(strongest),
        "qualifying_rollouts": len(evidence),
        "mcap": mcap,
        "rrd": rrd,
    }
