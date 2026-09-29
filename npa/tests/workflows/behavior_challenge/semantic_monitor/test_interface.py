"""Test the non-mutating monitor interface and observation leakage guard."""

from __future__ import annotations

import numpy as np
import pytest

from npa.workflows.behavior_challenge.semantic_monitor import (
    PromotionDecision,
    PromotionRequest,
    validate_evaluation_observation,
)


def _observation(*, depth: bool = True) -> dict[str, np.ndarray]:
    result = {"robot_r1::proprio": np.zeros(61, dtype=np.float32)}
    for camera, size in (
        ("zed_link", 720),
        ("left_realsense_link", 480),
        ("right_realsense_link", 480),
    ):
        stem = f"robot_r1::robot_r1:{camera}:Camera:0"
        result[f"{stem}::rgb"] = np.zeros((size, size, 3), dtype=np.uint8)
        if depth:
            result[f"{stem}::depth_linear"] = np.zeros(
                (size, size, 1), dtype=np.float32
            )
    return result


@pytest.mark.parametrize(
    "privileged_key",
    (
        "object_state",
        "target_object_pose",
        "robot_global_pose",
        "segmentation",
        "task_progress",
        "inside_target",
    ),
)
def test_evaluation_observation_rejects_privileged_fields(privileged_key: str) -> None:
    observation = _observation()
    observation[privileged_key] = np.zeros(1)

    with pytest.raises(ValueError, match="privileged or unknown"):
        validate_evaluation_observation(observation)


def test_evaluation_observation_accepts_only_onboard_modalities() -> None:
    validate_evaluation_observation(_observation())
    validate_evaluation_observation(_observation(depth=False))


def test_evaluation_observation_rejects_bad_shape_dtype_and_nonfinite() -> None:
    bad_rgb = _observation()
    bad_rgb["robot_r1::robot_r1:zed_link:Camera:0::rgb"] = np.zeros(
        (720, 720, 3), dtype=np.float32
    )
    with pytest.raises(ValueError, match="RGB must be uint8"):
        validate_evaluation_observation(bad_rgb)

    bad_depth = _observation()
    bad_depth["robot_r1::robot_r1:zed_link:Camera:0::depth_linear"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite floating-point"):
        validate_evaluation_observation(bad_depth)

    bad_state = _observation()
    bad_state["robot_r1::proprio"] = np.zeros(60, dtype=np.float32)
    with pytest.raises(ValueError, match="61-element"):
        validate_evaluation_observation(bad_state)


def test_evaluation_observation_rejects_wire_batch_dimension() -> None:
    batched = _observation()
    batched["robot_r1::proprio"] = batched["robot_r1::proprio"][None]
    with pytest.raises(ValueError, match="61-element"):
        validate_evaluation_observation(batched)


def test_promotion_request_allows_only_hold_or_next_stage() -> None:
    request = PromotionRequest((_observation(),), current_stage=2, proposed_stage=3)
    assert request.validated() is request

    with pytest.raises(ValueError, match="hold or next-stage"):
        PromotionRequest(
            (_observation(),), current_stage=2, proposed_stage=4
        ).validated()
    with pytest.raises(ValueError, match="history length"):
        PromotionRequest((), current_stage=2, proposed_stage=2).validated()


@pytest.mark.parametrize("proposal", (True, 1.0))
def test_promotion_request_rejects_non_integer_proposal(proposal: object) -> None:
    with pytest.raises(ValueError, match="hold or next-stage"):
        PromotionRequest(
            (_observation(),), current_stage=0, proposed_stage=proposal
        ).validated()


def test_promotion_decision_requires_calibrated_bound_provenance() -> None:
    decision = PromotionDecision("hold", 0.25, 0.75, "a" * 64)
    assert decision.validated() is decision

    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        PromotionDecision("accept", np.nan, 0.5, "a" * 64).validated()
    with pytest.raises(ValueError, match="artifact SHA-256"):
        PromotionDecision("accept", 0.5, 0.5, "not-a-digest").validated()
