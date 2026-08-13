"""Simulator-grounded temporal credit with bounded Cosmos-Reason shaping."""

from __future__ import annotations

import math
from statistics import pvariance
from typing import Any

from npa.workflows.sim2real.constants import CORRECTIVE_TARGETS, ERROR_SEVERITY


class TemporalCreditError(ValueError):
    """Raised when an evaluation cannot produce a trustworthy reward signal."""


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _tags(raw: dict[str, Any]) -> list[str]:
    values = raw.get("error_tags") or ["ok"]
    if isinstance(values, str):
        values = [values]
    return [str(value) for value in values]


def _target(tags: list[str]) -> dict[str, Any]:
    corrections = [CORRECTIVE_TARGETS[tag] for tag in tags if tag in CORRECTIVE_TARGETS]
    if not corrections:
        return {"nl_correction": "maintain stable task progress", "action_delta": []}
    width = max(
        (len(item.get("action_delta") or []) for item in corrections), default=0
    )
    delta = [0.0] * width
    for item in corrections:
        for index, value in enumerate(item.get("action_delta") or []):
            if isinstance(value, (int, float, str)):
                delta[index] += float(value) / len(corrections)
    return {
        "nl_correction": " ".join(
            str(item.get("nl_correction") or "") for item in corrections
        ).strip(),
        "action_delta": [round(value, 6) for value in delta],
    }


def _vlm_agrees(tags: list[str], truth: dict[str, Any]) -> bool:
    """Conservative semantic calibration against simulator events."""

    if not truth:
        return True
    tag_set = set(tags)
    if tag_set == {"ok"}:
        return (
            bool(truth.get("placement_stable"))
            or float(truth.get("object_goal_distance_m", 1.0)) <= 0.05
        )
    if "missed_target" in tag_set or "minor_alignment" in tag_set:
        return float(truth.get("object_goal_distance_m", 1.0)) > 0.05
    if "late_grasp" in tag_set:
        return not bool(truth.get("stable_grasp"))
    if "unstable" in tag_set:
        return not bool(truth.get("placement_stable")) or bool(truth.get("dropped"))
    return True


def _grounded_components(
    truth: dict[str, Any], previous: dict[str, Any] | None
) -> dict[str, float]:
    previous = previous or {}
    goal_change = float(
        truth.get(
            "object_goal_distance_change_m",
            float(
                previous.get(
                    "object_goal_distance_m", truth.get("object_goal_distance_m", 0.0)
                )
            )
            - float(truth.get("object_goal_distance_m", 0.0)),
        )
    )
    ee_change = float(
        truth.get(
            "end_effector_distance_change_m",
            float(
                previous.get(
                    "end_effector_object_distance_m",
                    truth.get("end_effector_object_distance_m", 0.0),
                )
            )
            - float(truth.get("end_effector_object_distance_m", 0.0)),
        )
    )
    distance = max(0.0, float(truth.get("object_goal_distance_m", 0.5)))
    lift_m = max(0.0, float(truth.get("object_lift_m", 0.0)))
    return {
        "goal_progress": 0.35 * _clip(goal_change / 0.03),
        "reach_progress": 0.20 * _clip(ee_change / 0.03),
        "contact": 0.08 if truth.get("contact") else 0.0,
        "stable_grasp": 0.14 if truth.get("stable_grasp") else 0.0,
        "lift": 0.12 * _clip(lift_m / 0.10, 0.0, 1.0),
        "placement": 0.30 if truth.get("placement_stable") else 0.0,
        "distance_penalty": -0.08 * _clip(distance / 0.50, 0.0, 1.0),
        "drop_penalty": -0.15 if truth.get("dropped") else 0.0,
        "termination_penalty": (
            -0.10
            if truth.get("terminated")
            and str(truth.get("termination_reason") or "")
            not in {"success", "goal_reached"}
            else 0.0
        ),
    }


def _fallback_grounded_rewards(steps: list[dict[str, Any]]) -> list[float]:
    """Second grounded signal when event rewards happen to be trajectory-constant."""

    rewards: list[float] = []
    for step in steps:
        truth = dict(step.get("simulator_ground_truth") or {})
        distance = float(truth.get("object_goal_distance_m", 0.5))
        ee_distance = float(truth.get("end_effector_object_distance_m", 0.5))
        action = step.get("action") or []
        action_l2 = math.sqrt(
            sum(float(value) ** 2 for value in action if isinstance(value, int | float))
        )
        rewards.append(_clip(-0.7 * distance - 0.25 * ee_distance - 0.05 * action_l2))
    return rewards


def convert_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    """Convert one VLM evaluation into calibrated dense temporal rewards.

    Simulator measurements are authoritative. Cosmos-Reason contributes at most
    0.12 reward magnitude and is down-weighted for broadcast critiques, model
    disagreement, low confidence, or contradiction with simulator state.
    """

    raw_steps = evaluation.get("per_step")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise TemporalCreditError("evaluation must include a non-empty per_step list")

    items: list[dict[str, Any]] = []
    rewards: list[float] = []
    previous_truth: dict[str, Any] | None = None
    calibrated = 0
    rejected = 0
    disagreements = 0
    missing_or_malformed = 0
    low_confidence = 0
    contradictory = 0
    summary_broadcast = 0
    for raw in raw_steps:
        if not isinstance(raw, dict) or "step" not in raw:
            raise TemporalCreditError("per_step entries must be objects with step")
        tags = _tags(raw)
        truth = dict(raw.get("simulator_ground_truth") or {})
        confidence = _clip(float(raw.get("confidence", 0.65)), 0.0, 1.0)
        source = str(raw.get("critique_source") or "model_per_step")
        reasons: set[str] = set()
        if source in {"model_missing", "model_malformed"}:
            confidence = 0.0
            missing_or_malformed += 1
            reasons.add("missing_or_malformed")
        if confidence < 0.5:
            low_confidence += 1
            reasons.add("low_confidence")
        if source == "summary_broadcast":
            confidence = min(confidence, 0.10)
            summary_broadcast += 1
            reasons.add("summary_broadcast")
        disagreement = bool(raw.get("model_disagreement"))
        if disagreement:
            confidence *= 0.25
            disagreements += 1
            reasons.add("model_disagreement")
        agrees = _vlm_agrees(tags, truth)
        if not agrees:
            confidence *= 0.25
            contradictory += 1
            reasons.add("simulator_contradiction")
        if reasons:
            rejected += 1
        else:
            calibrated += 1

        components = _grounded_components(truth, previous_truth) if truth else {}
        grounded = sum(components.values()) if components else 0.0
        severity = max(ERROR_SEVERITY.get(tag, 0.5) for tag in tags)
        vlm_shape = 0.12 * confidence * _clip(1.0 - 2.0 * severity)
        reward = _clip(grounded + vlm_shape)
        source_action = raw.get("action") or []
        items.append(
            {
                "step": int(raw["step"]),
                "reward": round(reward, 6),
                "target": _target(tags),
                "critique_text": str(raw.get("critique_text") or ""),
                "error_tags": tags,
                "confidence": round(confidence, 6),
                "model_disagreement": disagreement,
                "simulator_ground_truth": truth,
                "scenario_config_digest": str(
                    truth.get("scenario_config_digest")
                    or raw.get("scenario_config_digest")
                    or ""
                ),
                "reward_components": {
                    **{key: round(value, 6) for key, value in components.items()},
                    "vlm_auxiliary": round(vlm_shape, 6),
                },
                "action_credit": {
                    "source_action": source_action,
                    "credit": [
                        round(abs(float(value)) * reward, 6)
                        for value in source_action
                        if isinstance(value, int | float)
                    ],
                },
            }
        )
        rewards.append(reward)
        previous_truth = truth or previous_truth

    grounded_present = any(item["simulator_ground_truth"] for item in items)
    variance = pvariance(rewards) if len(rewards) > 1 else 0.0
    fallback_used = False
    if grounded_present and variance <= 1.0e-12:
        fallback = _fallback_grounded_rewards(items)
        if len(fallback) > 1 and pvariance(fallback) > 1.0e-12:
            rewards = fallback
            fallback_used = True
            for item, reward in zip(items, rewards, strict=True):
                item["reward"] = round(reward, 6)
                item["reward_components"]["degenerate_fallback"] = round(reward, 6)
            variance = pvariance(rewards)

    baseline = sum(rewards) / len(rewards)
    nonzero = 0
    for item in items:
        advantage = float(item["reward"]) - baseline
        item["advantage"] = round(advantage, 6)
        nonzero += int(abs(advantage) > 1.0e-8)
    calibration = {
        "step_count": len(items),
        "simulator_grounded_steps": sum(
            bool(item["simulator_ground_truth"]) for item in items
        ),
        "vlm_calibrated_steps": calibrated,
        "vlm_accepted_steps": calibrated,
        "vlm_rejected_or_downweighted_steps": rejected,
        "vlm_missing_or_malformed_steps": missing_or_malformed,
        "vlm_low_confidence_steps": low_confidence,
        "vlm_contradictory_steps": contradictory,
        "vlm_summary_broadcast_steps": summary_broadcast,
        "vlm_disagreement_downweighted_steps": disagreements,
        "model_disagreement_steps": disagreements,
        "reward_variance": round(variance, 10),
        "nonzero_advantage_count": nonzero,
        "degenerate_simulator_fallback_used": fallback_used,
        "degenerate": nonzero == 0,
    }
    return {
        "schema": "npa.sim2real.rl_signal.v1",
        "rollout_id": str(evaluation.get("rollout_id") or ""),
        "source": "simulator_ground_truth_with_bounded_vlm_auxiliary",
        "success": bool(evaluation.get("success")),
        "score": evaluation.get("score"),
        "per_step": items,
        "calibration": calibration,
        "mapping_rules": {
            "authority": "Isaac simulator ground truth",
            "vlm_role": "bounded auxiliary shaping (absolute contribution <= 0.12)",
            "advantage": "per-step calibrated reward minus trajectory mean",
            "reward_bounds": [-1.0, 1.0],
        },
    }
