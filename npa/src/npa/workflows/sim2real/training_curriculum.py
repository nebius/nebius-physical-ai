"""Choose resumed PPO exploration from exact simulator validation evidence."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from npa.workflows.sim2real.checkpoint_selection import (
    require_complete_checkpoint_candidate_pool,
    select_best_checkpoint,
)


def _goal_approach_count(rows: list[dict[str, Any]]) -> int:
    """Require paired grasp/lift and measured progress into the goal basin."""
    count = 0
    for row in rows:
        details = row["details"]
        distance = details.get("closest_object_goal_distance_m")
        if distance is None:
            distance = details.get("object_goal_distance_m")
        if distance is None:
            continue
        if (
            type(distance) not in (float, int)
            or not math.isfinite(distance)
            or distance < 0
        ):
            raise ValueError(
                "resume goal approach requires a finite nonnegative distance"
            )
        count += bool(details["stable_grasp"] and details["lift"] and distance < 0.08)
    return count


def _skill_phase(reliable_lifts: int, goal_approaches: int, episodes: int) -> str:
    """Keep policy noise trainable until validation demonstrates goal approach."""
    if reliable_lifts * 2 < episodes:
        return "exploration"
    if goal_approaches * 2 < episodes:
        return "transport"
    return "convergence"


def _validated_skill_progress(candidate: dict[str, Any]) -> dict[str, Any]:
    report = candidate["validation_report"]
    checkpoint = candidate["checkpoint_uri"]
    checksum = str(candidate.get("checkpoint_sha256") or "")
    checkpoint_size = candidate.get("checkpoint_size_bytes")
    if not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise ValueError("resume candidate requires an exact checkpoint SHA-256")
    if (
        type(checkpoint_size) is not int
        or isinstance(checkpoint_size, bool)
        or checkpoint_size <= 0
    ):
        raise ValueError("resume candidate requires a positive checkpoint byte size")
    if report.get("evaluation_split", candidate["evaluation_split"]) != "validation":
        raise ValueError("resume curriculum may consume only validation evidence")
    provenance = report.get("policy_inference_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("resume validation lacks checkpoint inference provenance")
    if (
        report.get("policy_checkpoint"),
        report.get("policy_checkpoint_sha256"),
        report.get("policy_checkpoint_size_bytes"),
        provenance.get("checkpoint_uri"),
        provenance.get("checkpoint_sha256"),
        provenance.get("checkpoint_size_bytes"),
        provenance.get("generator_policy_sha256"),
    ) != (
        checkpoint,
        checksum,
        checkpoint_size,
        checkpoint,
        checksum,
        checkpoint_size,
        checksum,
    ):
        raise ValueError("resume validation checkpoint identity differs")
    rows = report["per_env"]
    pairs = [
        (row["details"].get("stable_grasp"), row["details"].get("lift")) for row in rows
    ]
    if any(type(grasp) is not bool or type(lift) is not bool for grasp, lift in pairs):
        raise ValueError(
            "resume curriculum requires literal simulator grasp/lift verdicts"
        )
    reliable_lifts = sum(grasp and lift for grasp, lift in pairs)
    goal_approaches = _goal_approach_count(rows)
    return {
        "checkpoint_uri": checkpoint,
        "checkpoint_sha256": checksum,
        "validation_report_uri": candidate["validation_report_uri"],
        "validation_episodes": len(rows),
        "stable_grasp_and_lift_episodes": reliable_lifts,
        "stable_grasp_and_lift_rate": reliable_lifts / len(rows),
        "goal_approach_episodes": goal_approaches,
        "goal_approach_distance_m": 0.08,
        "phase": _skill_phase(reliable_lifts, goal_approaches, len(rows)),
        "decision_source": "simulator_validation_only",
    }


def resume_environment(evidence: dict[str, Any]) -> dict[str, str]:
    """Bind a resumed checkpoint and PPO phase to the best validation candidate.

    Args:
        evidence: Durable inner-loop evidence from the current or preceding pass.
    Returns:
        Explicit checkpoint identity, schedule phase, and its validation audit.
        Empty evidence disables unvalidated automatic training resume.
    Raises:
        ValueError: When selection or simulator checkpoint/skill evidence is invalid.
        KeyError: When a required validation field is missing.
    """
    environment = {"NPA_BYO_ISAAC_AUTO_RESUME": "0"}
    if not evidence:
        return environment
    candidates = list(evidence.get("checkpoint_candidates") or [])
    require_complete_checkpoint_candidate_pool(candidates)
    candidate = select_best_checkpoint(candidates)
    progress = _validated_skill_progress(candidate)
    environment.update(
        {
            "NPA_SIM2REAL_RESUME_CHECKPOINT_URI": progress["checkpoint_uri"],
            "NPA_SIM2REAL_RESUME_CHECKPOINT_SHA256": progress["checkpoint_sha256"],
            "NPA_SIM2REAL_POLICY_CHECKPOINT_URI": progress["checkpoint_uri"],
            "NPA_BYO_ISAAC_RESUME_PHASE": progress["phase"],
            "NPA_SIM2REAL_RESUME_CURRICULUM_JSON": json.dumps(progress, sort_keys=True),
        }
    )
    return environment


def ppo_environment(
    args: Any,
    base_environment: dict[str, Any],
    prior_evidence: dict[str, Any],
    signal_batch: Path,
    training_output: Path,
) -> dict[str, Any]:
    """Build the native Stage 9 PPO environment from validated resume evidence.

    Args:
        args: Stage arguments with loop and PPO settings.
        base_environment: Current canonical Isaac runtime environment.
        prior_evidence: Durable evidence from the preceding completed pass.
        signal_batch: Local temporal-signal batch consumed by the trainer.
        training_output: Local path where the trainer writes its result.
    Returns:
        Fully bound native PPO environment.
    Raises:
        ValueError: If prior checkpoint selection is not validation-eligible.
    """
    environment = dict(base_environment)
    environment.update(resume_environment(prior_evidence))
    environment.update(
        {
            "NPA_SIM2REAL_SIGNAL_JSON": signal_batch,
            "NPA_SIM2REAL_OUTPUT_JSON": training_output,
            "NPA_SIM2REAL_TRAINER_TAG": (
                f"outer-{args.outer_iteration:02d}-iter-{args.inner_iteration:02d}"
            ),
            "NPA_BYO_ISAAC_NUM_ENVS": args.ppo_num_envs,
            "NPA_BYO_ISAAC_ITERATIONS": args.ppo_iterations,
            "NPA_BYO_ISAAC_STEPS_PER_ENV": args.ppo_steps_per_env,
            "NPA_BYO_ISAAC_VALIDATION_INTERVAL": min(100, args.ppo_iterations),
        }
    )
    return environment
