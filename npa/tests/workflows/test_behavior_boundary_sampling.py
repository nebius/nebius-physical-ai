"""Verify deterministic matched sampling around expert gripper transitions."""

from __future__ import annotations

import numpy as np
import pytest

from npa.workflows.behavior_challenge.boundary_sampling import (
    BoundarySamplingPlan,
    CLOSED_COMMAND,
    EpisodeActions,
    OPEN_COMMAND,
    SampleKey,
    construct_matched_samples,
    persistent_gripper_transitions,
)


def _actions(length: int, *, left: float = 1.0, right: float = 1.0) -> np.ndarray:
    result = np.zeros((length, 23), dtype=np.float32)
    result[:, 14] = left
    result[:, 22] = right
    return result


def _plan(**changes: object) -> BoundarySamplingPlan:
    values = {
        "task_quotas": {1: 2},
        "fit_membership": {1: (0,)},
        "calibration_membership": {1: (99,)},
        "focal_task_id": 1,
        "focal_uniform_quota": 0,
        "close_quota": 0,
        "open_quota": 2,
        "seed": 2,
        "action_horizon": 4,
    }
    values.update(changes)
    return BoundarySamplingPlan(**values)  # type: ignore[arg-type]


def test_persistent_transitions_use_exact_expert_gripper_coordinates() -> None:
    actions = _actions(26)
    actions[4:, 14] = -1.0
    actions[10:, 14] = 1.0
    actions[16:, 22] = -1.0
    actions[22:, 22] = 1.0

    transitions = persistent_gripper_transitions(actions)

    assert [(item.side, item.command, item.frame_index) for item in transitions] == [
        ("left", "close", 4),
        ("left", "open", 10),
        ("right", "close", 16),
        ("right", "open", 22),
    ]


def test_transition_requires_persistence() -> None:
    unstable = _actions(10)
    unstable[4, 14] = CLOSED_COMMAND
    assert persistent_gripper_transitions(unstable) == ()


@pytest.mark.parametrize("coordinate", [14, 22])
@pytest.mark.parametrize("command", [0.0, 0.5])
def test_transition_rejects_nonbinary_commands(coordinate: int, command: float) -> None:
    nonbinary = _actions(10)
    nonbinary[4, coordinate] = command
    with pytest.raises(ValueError, match="signed binary"):
        persistent_gripper_transitions(nonbinary)


def test_simultaneous_two_arm_transition_is_ambiguous() -> None:
    actions = _actions(10)
    actions[4:, (14, 22)] = CLOSED_COMMAND

    assert persistent_gripper_transitions(actions) == ()


def test_matched_sampling_preserves_anchor_draws_and_exact_task_quotas() -> None:
    anchor_zero = EpisodeActions(0, 5, _actions(12))
    focal = _actions(16)
    focal[5:, 14] = CLOSED_COMMAND
    focal[12:, 14] = OPEN_COMMAND
    anchor_twenty_two = EpisodeActions(22, 8, _actions(12))
    episodes = [EpisodeActions(1, 7, focal), anchor_twenty_two, anchor_zero]

    result = construct_matched_samples(
        episodes,
        plan=_plan(
            task_quotas={0: 4, 1: 6, 22: 4},
            fit_membership={0: (5,), 1: (7,), 22: (8,)},
            calibration_membership={0: (105,), 1: (107,), 22: (108,)},
            focal_uniform_quota=2,
            close_quota=2,
            open_quota=2,
            seed=9,
        ),
    )

    control_by_task = {
        task: [row for row in result.control if row.task_id == task]
        for task in (0, 1, 22)
    }
    candidate_by_task = {
        task: [row for row in result.candidate if row.task_id == task]
        for task in (0, 1, 22)
    }
    assert {task: len(rows) for task, rows in control_by_task.items()} == {
        0: 4,
        1: 6,
        22: 4,
    }
    assert {task: len(rows) for task, rows in candidate_by_task.items()} == {
        0: 4,
        1: 6,
        22: 4,
    }
    assert candidate_by_task[0] == control_by_task[0]
    assert candidate_by_task[22] == control_by_task[22]
    assert [row.task_id for row in result.control] == [
        row.task_id for row in result.candidate
    ]
    for control, candidate, stratum in zip(
        result.control, result.candidate, result.candidate_strata, strict=True
    ):
        if control.task_id != 1 or stratum == "uniform":
            assert candidate == control
    assert [row.task_id for row in result.control] != sorted(
        row.task_id for row in result.control
    )
    assert result.candidate_strata.count("close") == 2
    assert result.candidate_strata.count("open") == 2
    repeated = construct_matched_samples(
        episodes,
        plan=_plan(
            task_quotas={22: 4, 1: 6, 0: 4},
            fit_membership={22: (8,), 1: (7,), 0: (5,)},
            calibration_membership={22: (108,), 1: (107,), 0: (105,)},
            focal_uniform_quota=2,
            close_quota=2,
            open_quota=2,
            seed=9,
        ),
    )
    assert result == repeated


def test_release_near_chunk_edge_stays_inside_episode() -> None:
    actions = _actions(12, left=CLOSED_COMMAND)
    actions[9:, 14] = OPEN_COMMAND
    episodes = [EpisodeActions(1, 3, actions)]

    result = construct_matched_samples(
        episodes,
        plan=_plan(fit_membership={1: (3,)}),
    )

    assert all(row.episode_id == 3 for row in result.candidate)
    assert all(row.frame_index + 4 <= len(actions) for row in result.candidate)
    assert all(row.frame_index <= 9 < row.frame_index + 4 for row in result.candidate)


def test_chunks_with_overlapping_transitions_are_not_boundary_samples() -> None:
    actions = _actions(15)
    actions[5:, 14] = CLOSED_COMMAND
    actions[9:, 22] = CLOSED_COMMAND
    episodes = [EpisodeActions(1, 4, actions)]

    result = construct_matched_samples(
        episodes,
        plan=_plan(
            close_quota=2,
            open_quota=0,
            seed=1,
            action_horizon=6,
            fit_membership={1: (4,)},
        ),
    )

    assert all(
        not (row.frame_index <= 5 and 9 < row.frame_index + 6)
        for row in result.candidate
    )


def test_anchor_task_boundaries_cannot_fill_focal_task_quota() -> None:
    anchor = _actions(12)
    anchor[5:, 14] = CLOSED_COMMAND

    with pytest.raises(ValueError, match="close boundary sample pool is empty"):
        construct_matched_samples(
            [EpisodeActions(0, 0, anchor), EpisodeActions(1, 0, _actions(12))],
            plan=_plan(
                task_quotas={0: 1, 1: 1},
                fit_membership={0: (0,), 1: (0,)},
                calibration_membership={0: (99,), 1: (99,)},
                close_quota=1,
                open_quota=0,
                seed=0,
            ),
        )


def test_valid_release_and_paired_hand_change_in_one_chunk_is_excluded() -> None:
    actions = _actions(12, left=CLOSED_COMMAND)
    actions[4:, 14] = OPEN_COMMAND
    actions[8:, (14, 22)] = CLOSED_COMMAND

    with pytest.raises(ValueError, match="open boundary sample pool is empty"):
        construct_matched_samples(
            [EpisodeActions(1, 0, actions)],
            plan=_plan(action_horizon=12),
        )


def test_calibration_episode_injection_is_rejected() -> None:
    with pytest.raises(ValueError, match="differ from frozen fit membership"):
        construct_matched_samples(
            [EpisodeActions(1, 99, _actions(12))],
            plan=_plan(),
        )


def test_fit_and_calibration_memberships_must_be_disjoint() -> None:
    with pytest.raises(ValueError, match="must not overlap"):
        construct_matched_samples(
            [EpisodeActions(1, 0, _actions(12))],
            plan=_plan(calibration_membership={1: (0,)}),
        )


def test_action_horizon_is_required() -> None:
    values = {
        "task_quotas": {1: 1},
        "fit_membership": {1: (0,)},
        "calibration_membership": {1: (99,)},
        "focal_task_id": 1,
        "focal_uniform_quota": 1,
        "close_quota": 0,
        "open_quota": 0,
        "seed": 0,
    }
    with pytest.raises(TypeError, match="action_horizon"):
        BoundarySamplingPlan(**values)  # type: ignore[call-arg]


def test_schema_and_quota_mismatches_fail_closed() -> None:
    bad_shape = EpisodeActions(1, 0, np.zeros((8, 22), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        construct_matched_samples(
            [bad_shape],
            plan=_plan(
                task_quotas={1: 1},
                focal_uniform_quota=1,
                open_quota=0,
                seed=0,
            ),
        )

    with pytest.raises(ValueError, match="strata"):
        construct_matched_samples(
            [EpisodeActions(1, 0, _actions(40))],
            plan=_plan(focal_uniform_quota=1, open_quota=0, seed=0),
        )


def test_sample_key_keeps_original_episode_and_frame_identity() -> None:
    assert SampleKey(1, 7, 9) != SampleKey(1, 8, 9)
