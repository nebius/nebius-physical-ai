"""Test TRAIN-only task-1 semantic event collection."""

from __future__ import annotations

import pytest

from npa.workflows.behavior_challenge.semantic_monitor import (
    BoundaryRequirement,
    collect_episode_labels,
    deterministic_episode_split,
)


def _record(
    frame: int,
    *,
    source: str = "train-rollout-7",
    split: str = "train",
    grasped: bool = False,
    inside: bool = False,
    near: bool = True,
    grasp_attempted: bool = False,
    release_attempted: bool = False,
    before: int = 0,
    proposal: int = 0,
) -> dict:
    states = []
    for slot in range(3):
        active = slot == 0
        states.append(
            {
                "slot": slot,
                "near": near if active else False,
                "grasped": grasped if active else False,
                "inside_target": inside if active else False,
                "grasp_attempted": grasp_attempted if active else False,
                "release_attempted": release_attempted if active else False,
            }
        )
    return {
        "schema": "npa.behavior.task1-semantic-trace.v1",
        "split": split,
        "source": source,
        "task_id": 1,
        "task_name": "picking_up_trash",
        "episode_id": 7,
        "frame_index": frame,
        "observation_sha256": f"{frame:064x}",
        "native_stage_before": before,
        "native_stage_proposal": proposal,
        "target_can_slot": 0,
        "can_states": states,
    }


def _requirements() -> tuple[BoundaryRequirement, ...]:
    return (
        BoundaryRequirement(0, 0, "grasp_established"),
        BoundaryRequirement(1, 0, "grasp_retained"),
        BoundaryRequirement(2, 0, "placement_confirmed"),
    )


def test_split_is_order_independent_disjoint_and_episode_level() -> None:
    first = deterministic_episode_split(tuple(range(180)), calibration_count=20)
    second = deterministic_episode_split(
        tuple(reversed(range(180))), calibration_count=20
    )

    assert first == second
    assert len(first["fit"]) == 160
    assert len(first["calibration"]) == 20
    assert set(first["fit"]).isdisjoint(first["calibration"])


@pytest.mark.parametrize(
    ("source", "split"),
    (
        ("task1-dev", "train"),
        ("task1-report", "train"),
        ("task1-holdout", "train"),
        ("task1-validation", "train"),
        ("train-rollout", "dev"),
        ("train-rollout", "report"),
    ),
)
def test_collector_rejects_non_train_provenance(source: str, split: str) -> None:
    with pytest.raises(ValueError, match="TRAIN|forbidden"):
        collect_episode_labels(
            [_record(0, source=source, split=split)],
            authorized_episodes=frozenset({7}),
            boundary_requirements=_requirements(),
        )


def test_collector_rejects_unauthorized_or_mixed_episodes() -> None:
    with pytest.raises(ValueError, match="authorized TRAIN"):
        collect_episode_labels(
            [_record(0)],
            authorized_episodes=frozenset({8}),
            boundary_requirements=_requirements(),
        )


def test_collector_rejects_nonconsecutive_frames() -> None:
    with pytest.raises(ValueError, match="consecutive TRAIN frames"):
        collect_episode_labels(
            [_record(0), _record(100)],
            authorized_episodes=frozenset({7}),
            boundary_requirements=_requirements(),
        )
    other = _record(1)
    other["episode_id"] = 8
    with pytest.raises(ValueError, match="exactly one"):
        collect_episode_labels(
            [_record(0), other],
            authorized_episodes=frozenset({7, 8}),
            boundary_requirements=_requirements(),
        )


def test_grasp_requires_persistence_before_promotion_acceptance() -> None:
    rows = [
        _record(0, grasped=True, proposal=1),
        _record(1, grasped=True, proposal=1),
        _record(2, grasped=True, before=1, proposal=2),
    ]
    labels = collect_episode_labels(
        rows,
        authorized_episodes=frozenset({7}),
        boundary_requirements=_requirements(),
    )

    assert [row.event for row in labels] == [
        "none",
        "grasp_established",
        "grasp_retained",
    ]
    assert [row.promotion_label for row in labels] == ["hold", "accept", "accept"]
    assert labels[-1].phase == "transport"


def test_delayed_proposal_uses_persistent_grasp_state() -> None:
    rows = [
        _record(0, grasped=True),
        _record(1, grasped=True),
        _record(2, grasped=True, proposal=1),
    ]
    labels = collect_episode_labels(
        rows,
        authorized_episodes=frozenset({7}),
        boundary_requirements=_requirements(),
    )

    assert labels[1].event == "grasp_established"
    assert labels[2].event == "grasp_retained"
    assert labels[2].promotion_label == "accept"


def test_drop_and_missed_attempts_are_distinct_boundaries() -> None:
    rows = [
        _record(0, grasped=True),
        _record(1, grasped=True),
        _record(2, grasp_attempted=True),
        _record(3, release_attempted=True),
    ]
    labels = collect_episode_labels(
        rows,
        authorized_episodes=frozenset({7}),
        boundary_requirements=_requirements(),
    )

    assert labels[2].failure_boundary == "dropped"
    assert labels[3].failure_boundary == "missed_placement"


def test_regrasp_requires_new_persistence_after_drop() -> None:
    rows = [
        _record(0, grasped=True),
        _record(1, grasped=True),
        _record(2),
        _record(3, grasped=True, proposal=1),
        _record(4, grasped=True, proposal=1),
    ]
    labels = collect_episode_labels(
        rows,
        authorized_episodes=frozenset({7}),
        boundary_requirements=_requirements(),
    )

    assert labels[2].failure_boundary == "dropped"
    assert labels[3].promotion_label == "hold"
    assert labels[4].event == "grasp_established"
    assert labels[4].promotion_label == "accept"


def test_placement_requires_persistence_and_reverses_when_can_exits() -> None:
    rows = [
        _record(0, grasped=True),
        _record(1, grasped=True),
        _record(2, inside=True, before=2, proposal=3),
        _record(3, inside=True, before=2, proposal=3),
        _record(4, inside=False, before=3, proposal=3),
    ]
    labels = collect_episode_labels(
        rows,
        authorized_episodes=frozenset({7}),
        boundary_requirements=_requirements(),
    )

    assert labels[2].promotion_label == "hold"
    assert labels[3].event == "placement_confirmed"
    assert labels[3].promotion_label == "accept"
    assert labels[3].completed_can_count == 1
    assert labels[4].completed_can_count == 0


def test_delayed_placement_proposal_holds_after_reversal() -> None:
    rows = [
        _record(0, inside=True, before=2, proposal=2),
        _record(1, inside=True, before=2, proposal=2),
        _record(2, inside=True, before=2, proposal=3),
        _record(3, inside=False, before=2, proposal=3),
    ]
    labels = collect_episode_labels(
        rows,
        authorized_episodes=frozenset({7}),
        boundary_requirements=_requirements(),
    )

    assert labels[1].event == "placement_confirmed"
    assert labels[2].event == "none"
    assert labels[2].promotion_label == "accept"
    assert labels[3].promotion_label == "hold"


def test_successful_placement_then_regrasp_requires_new_persistence() -> None:
    rows = [
        _record(0, grasped=True),
        _record(1, grasped=True),
        _record(2, inside=True),
        _record(3, inside=True),
        _record(4, inside=True, grasped=True, proposal=1),
        _record(5, inside=True, grasped=True, proposal=1),
    ]
    labels = collect_episode_labels(
        rows,
        authorized_episodes=frozenset({7}),
        boundary_requirements=_requirements(),
    )

    assert labels[3].phase == "placed"
    assert labels[3].grasp_established is False
    assert labels[4].completed_can_count == 0
    assert labels[4].promotion_label == "hold"
    assert labels[5].event == "grasp_established"
    assert labels[5].promotion_label == "accept"


@pytest.mark.parametrize("task_id", (True, 1.0))
def test_schema_rejects_non_integer_task_identity(task_id: object) -> None:
    row = _record(0)
    row["task_id"] = task_id
    with pytest.raises(ValueError, match="task 1 only"):
        collect_episode_labels(
            [row],
            authorized_episodes=frozenset({7}),
            boundary_requirements=_requirements(),
        )


@pytest.mark.parametrize("slot", (True, 1.0))
def test_boundary_requirement_rejects_non_integer_can_slot(slot: object) -> None:
    with pytest.raises(ValueError, match="can slot"):
        collect_episode_labels(
            [_record(0)],
            authorized_episodes=frozenset({7}),
            boundary_requirements=(BoundaryRequirement(0, slot, "grasp_established"),),
        )


def test_completed_count_tracks_nontarget_cans() -> None:
    rows = [_record(0), _record(1)]
    for row in rows:
        row["can_states"][1]["inside_target"] = True

    labels = collect_episode_labels(
        rows,
        authorized_episodes=frozenset({7}),
        boundary_requirements=_requirements(),
    )

    assert labels[-1].can_slot == 0
    assert labels[-1].completed_can_count == 1


def test_duplicate_frames_and_malformed_can_rows_fail_closed() -> None:
    with pytest.raises(ValueError, match="unique and chronological"):
        collect_episode_labels(
            [_record(0), _record(0)],
            authorized_episodes=frozenset({7}),
            boundary_requirements=_requirements(),
        )
    malformed = _record(0)
    malformed["can_states"] = malformed["can_states"][:2]
    with pytest.raises(ValueError, match="exactly three"):
        collect_episode_labels(
            [malformed],
            authorized_episodes=frozenset({7}),
            boundary_requirements=_requirements(),
        )
