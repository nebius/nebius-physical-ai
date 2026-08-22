"""Independent success verification for the Sim2Real model-agent benchmark."""

from __future__ import annotations

import hashlib
import json
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


def _sample_time(row: dict[str, Any], manifest: dict[str, Any]) -> float:
    for key in ("sim_time_seconds", "timestamp_seconds", "time_seconds"):
        if row.get(key) is not None:
            return float(row[key])
    step_seconds = manifest.get("simulation_step_seconds")
    if step_seconds is None:
        step_seconds = (manifest.get("capture") or {}).get("simulation_step_seconds")
    if step_seconds is not None and row.get("sim_step") is not None:
        return float(row["sim_step"]) * float(step_seconds)
    raise VerificationError(
        "rollout ground truth has no physical timestamp; record sim_time_seconds "
        "or simulation_step_seconds instead of inferring duration from sample count"
    )


def _qualifies(row: dict[str, Any], lift_m: float) -> bool:
    truth = row.get("simulator_ground_truth") or {}
    return (
        bool(truth.get("stable_grasp"))
        and float(truth.get("object_lift_m") or 0) >= lift_m
    )


def _lift_evidence(
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    minimum_lift_m: float,
    minimum_hold_seconds: float,
) -> LiftEvidence | None:
    if manifest.get("schema") != "npa.sim2real.action_rollout.v1":
        return None
    if manifest.get("source") != "byo_isaac_policy_rollout":
        return None
    if manifest.get("sim_backend") != "isaac" or not manifest.get("policy_trained"):
        return None
    checkpoint_sha = str(manifest.get("policy_checkpoint_sha256") or "")
    if (
        len(checkpoint_sha) != 64
        or int(manifest.get("policy_checkpoint_size_bytes") or 0) <= 0
    ):
        return None
    step_seconds_raw = manifest.get("simulation_step_seconds")
    if step_seconds_raw is None:
        step_seconds_raw = (manifest.get("capture") or {}).get(
            "simulation_step_seconds"
        )
    if step_seconds_raw is None or float(step_seconds_raw) <= 0:
        raise VerificationError(
            "rollout has no positive simulation_step_seconds needed to prove "
            "continuous temporal coverage"
        )
    step_seconds = float(step_seconds_raw)

    current: list[tuple[float, dict[str, Any]]] = []
    best: list[tuple[float, dict[str, Any]]] = []
    for row in manifest.get("actions") or []:
        if not isinstance(row, dict):
            current = []
            continue
        timestamp = _sample_time(row, manifest)
        if _qualifies(row, minimum_lift_m):
            if current:
                delta = timestamp - current[-1][0]
                if delta <= 0:
                    raise VerificationError(
                        "rollout timestamps are not strictly increasing"
                    )
                previous_step = current[-1][1].get("sim_step")
                current_step = row.get("sim_step")
                if (
                    delta > step_seconds * 1.5
                    or previous_step is None
                    or current_step is None
                    or int(current_step) != int(previous_step) + 1
                ):
                    current = []
            current.append((timestamp, row))
            if not best or current[-1][0] - current[0][0] > best[-1][0] - best[0][0]:
                best = list(current)
        else:
            current = []
    if len(best) < 2 or best[-1][0] - best[0][0] < minimum_hold_seconds:
        return None
    lifts = [
        float((row.get("simulator_ground_truth") or {}).get("object_lift_m") or 0)
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
        detail = f" Timestamp errors: {'; '.join(parse_errors)}" if parse_errors else ""
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
