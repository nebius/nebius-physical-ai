"""Reset intervals must remain visible without acquiring policy training credit."""

from copy import deepcopy
import json

import pytest

from npa.workflows.sim2real.episode_boundaries import (
    EpisodeBoundaries, validate_episode_sequence,
)
from npa.workflows.sim2real.temporal_credit import convert_evaluation
from npa.workflows.sim2real.byo_isaac_trainer import read_signal_stats
from npa.workbench.cosmos.visual_grounding import (
    bind_action_frames, validate_stored_visual_grounding,
)
from npa.workbench.cosmos.reason import (
    _parse_hosted_rollout_output, _cosmos_reason_prompt, _parse_cosmos_reason_output,
    merge_dual_reason_evaluations, CosmosReasonError,
)


def _boundary(generation=0, resets=(), current=False):
    return {
        "schema": "npa.sim2real.episode_boundary.v1",
        "simulator_episode_id": generation,
        "action_episode_id": generation - int(current),
        "reset_events": [
            {"sim_step": step, "terminated_episode_id": generation - len(resets) + index,
             "next_episode_id": generation - len(resets) + index + 1}
            for index, step in enumerate(resets)
        ],
        "reset_on_current_step": current, "action_outcome_valid": not current,
        "temporal_credit_valid": not resets,
    }


def _actions():
    return [
        {"step": index, "sim_step": step, "action": [action],
         "episode_boundary": boundary,
         "simulator_ground_truth": {
             "object_goal_distance_m": 0.3, "end_effector_object_distance_m": 0.2,
             "object_goal_distance_change_m": 0.0, "end_effector_distance_change_m": 0.0,
         }, "confidence": 0, "error_tags": ["ok"]}
        for index, (step, action, boundary) in enumerate([
            (0, 0.0, _boundary()), (9, 0.2, _boundary()),
            (19, 500, _boundary(1, [12])), (29, 0.5, _boundary(1)),
        ])
    ]


def _frames(actions):
    return [{"path": f"camera-{row['step']:03d}.png", "sim_step": row["sim_step"],
             "view_name": "primary", "episode_id": "rollout-test",
             "simulator_episode_id": (None if row["episode_boundary"]["reset_on_current_step"]
                                      else row["episode_boundary"]["simulator_episode_id"])}
            for row in actions]


def test_tracker_retains_multiple_unsampled_resets_independently():
    tracker = EpisodeBoundaries(2)
    for step in range(5):
        tracker.advance([step in {1, 3}, step == 4], step)
    assert tracker.sample(0) == _boundary(2, [1, 3])
    assert tracker.sample(1) == _boundary(1, [4], current=True)
    assert tracker.frame_episode(0) == 2 and tracker.frame_episode(1) is None
    tracker.sampled()
    tracker.advance([False, False], 5)
    assert tracker.sample(0) == _boundary(2)
    assert tracker.sample(1) == _boundary(1)
    assert tracker.frame_episode(1) == 1


@pytest.mark.parametrize("corruption", [
    "missing", "missing_one", "boolean_generation", "false_validity", "missing_reset",
    "wrong_action_episode", "sampled_flag", "repeated_reset", "future_reset",
    "noncontiguous_generation", "outcome_in_metadata",
])
def test_episode_sequence_rejects_missing_or_contradictory_evidence(corruption):
    rows = _actions()
    boundary = rows[2]["episode_boundary"]
    if corruption == "missing":
        for row in rows:
            del row["episode_boundary"]
    elif corruption == "missing_one":
        del rows[2]["episode_boundary"]
    elif corruption == "boolean_generation":
        boundary["simulator_episode_id"] = True
    elif corruption == "false_validity":
        boundary["temporal_credit_valid"] = True
    elif corruption == "missing_reset":
        rows[2]["episode_boundary"] = _boundary(1)
    elif corruption == "wrong_action_episode":
        boundary["action_episode_id"] = 0
    elif corruption == "sampled_flag":
        boundary["reset_on_current_step"] = True
    elif corruption == "repeated_reset":
        boundary["reset_events"][0]["sim_step"] = 9
    elif corruption == "future_reset":
        boundary["reset_events"][0]["sim_step"] = 20
    elif corruption == "noncontiguous_generation":
        boundary["reset_events"][0]["terminated_episode_id"] = 3
    else:
        boundary["placement_stable"] = True
    with pytest.raises(ValueError, match="episode boundary"):
        validate_episode_sequence(rows)
    with pytest.raises(ValueError, match="episode boundary"):
        convert_evaluation({"schema": "npa.sim2real.vlm_eval.v5", "per_step": rows})


@pytest.mark.parametrize("current", [False, True])
def test_reset_jump_cannot_earn_credit_or_influence_training_statistics(tmp_path, current):
    rows = _actions()
    if current:
        rows[2]["episode_boundary"] = _boundary(1, [19], current=True)
    rows[2]["simulator_ground_truth"].update(
        object_goal_distance_m=0.01, object_goal_distance_change_m=0.5,
        end_effector_distance_change_m=0.5, object_lift_m=0.2,
        placement_stable=True, stable_grasp=True, contact=True,
    )
    signal = convert_evaluation({"schema": "npa.sim2real.vlm_eval.v5", "per_step": rows})
    invalid = signal["per_step"][2]
    assert invalid["reward"] == invalid["advantage"] == invalid["confidence"] == 0
    assert invalid["action_credit"]["credit"] == [0]
    assert not any(invalid["target"]["action_delta"])
    assert signal["calibration"]["degenerate_simulator_fallback_used"] is True
    eligible = [signal["per_step"][index] for index in (0, 1, 3)]
    assert sum(row["advantage"] for row in eligible) == pytest.approx(0, abs=2e-6)
    assert all(row["reward"] < 0 for row in eligible)
    path = tmp_path / "signals.json"
    path.write_text(json.dumps({"signals": [signal]}))
    stats = read_signal_stats(str(path))
    assert stats["step_count"] == 3
    assert stats["mean_reward"] == pytest.approx(sum(row["reward"] for row in eligible) / 3)
    assert stats["error_tags"] == {}
    assert signal["calibration"]["episode_boundary_excluded_steps"] == 1


def test_all_reset_intervals_remain_zero_and_degenerate():
    rows = _actions()[:3]
    for index, row in enumerate(rows):
        row["episode_boundary"] = _boundary(index + 1, [row["sim_step"]], current=True)
    signal = convert_evaluation({"schema": "npa.sim2real.vlm_eval.v5", "per_step": rows})
    assert signal["calibration"]["degenerate"] is True
    assert signal["calibration"]["credit_eligible_steps"] == 0
    assert all(row["reward"] == row["advantage"] == 0 for row in signal["per_step"])


def test_hosted_boundary_binding_survives_prompt_parse_and_archived_validation():
    actions = _actions()
    frames = _frames(actions)
    names = [frame["path"] for frame in frames]
    bindings = bind_action_frames(actions=actions, frame_metadata=frames,
                                  frame_names=names, rollout_id="rollout-test")
    assert bindings[2]["supported"] is False
    assert bindings[3]["supported"] is True
    payload = {"success": False, "score": 0.1, "summary": "Cube stays on table.",
               "per_step": [{"step": step, "camera_observation": binding["camera_observation"],
                             "confidence": 0, "error_tags": ["ok"],
                             "critique_text": f"Insufficient visual evidence for step {step}."}
                            for step, binding in bindings.items()]}
    result = _parse_hosted_rollout_output(
        json.dumps(payload), actions=actions, rollout_id="rollout-test", threshold=0.5,
        family="minimax_m3", frame_names=names, visual_bindings=bindings,
    )
    result.update(frame_count=len(names), selected_frames=names, selected_frame_metadata=frames)
    validate_stored_visual_grounding(result)
    assert [row["episode_boundary"] for row in result["per_step"]] == [
        row["episode_boundary"] for row in actions]
    prompt = _cosmos_reason_prompt(
        family="minimax_m3", task_description="cube task", actions=actions,
        frame_names=names, visual_bindings=bindings, selected_frame_metadata=frames,
    )
    assert '"simulator_episode_id": 1' in prompt
    assert "Never infer motion" in prompt and "object_goal_distance_m" not in prompt
    corrupted = deepcopy(result)
    corrupted["per_step"][2]["episode_boundary"]["temporal_credit_valid"] = True
    with pytest.raises(ValueError, match="episode boundary"):
        validate_stored_visual_grounding(corrupted)


def test_same_time_frame_from_wrong_simulator_episode_is_rejected():
    actions = _actions()
    frames = _frames(actions)
    frames[3]["simulator_episode_id"] = 0
    with pytest.raises(ValueError, match="different simulator episode"):
        bind_action_frames(actions=actions, frame_metadata=frames,
                           frame_names=[frame["path"] for frame in frames], rollout_id="rollout-test")


def test_current_reset_frame_and_final_context_preserve_unknown_rendered_episode():
    actions = _actions()
    actions[-1]["episode_boundary"] = _boundary(2, [29], current=True)
    frames = _frames(actions)
    frames.append({**frames[-1], "path": "camera-final.png", "sim_step": 30})
    bindings = bind_action_frames(actions=actions, frame_metadata=frames,
                                  frame_names=[frame["path"] for frame in frames], rollout_id="rollout-test")
    assert bindings[3]["supported"] is False
    frames[-1]["simulator_episode_id"] = 2
    with pytest.raises(ValueError, match="different simulator episode"):
        bind_action_frames(actions=actions, frame_metadata=frames,
                           frame_names=[frame["path"] for frame in frames], rollout_id="rollout-test")


@pytest.mark.parametrize("family", ["reason1", "reason2", "cosmos3"])
def test_common_reason_parser_preserves_native_time_for_episode_conversion(family):
    actions = _actions()
    payload = {"success": False, "score": 0.1, "summary": "Cube stays on table.",
               "per_step": [{"step": row["step"], "error_tags": ["ok"],
                             "critique_text": "Insufficient evidence.", "confidence": 0}
                            for row in actions]}
    result = _parse_cosmos_reason_output(
        json.dumps(payload), actions=actions, rollout_id="rollout-test", threshold=0.5, family=family,
    )
    assert [row["sim_step"] for row in result["per_step"]] == [0, 9, 19, 29]
    signal = convert_evaluation(result)
    assert signal["per_step"][2]["reward"] == signal["per_step"][2]["advantage"] == 0
    merged = merge_dual_reason_evaluations(result, deepcopy(result), threshold=0.5)
    assert merged["per_step"][2]["episode_boundary"] == actions[2]["episode_boundary"]
    assert convert_evaluation(merged)["per_step"][2]["reward"] == 0
    malformed = deepcopy(result)
    del malformed["per_step"][2]["episode_boundary"]
    with pytest.raises(CosmosReasonError, match="episode boundaries"):
        merge_dual_reason_evaluations(result, malformed, threshold=0.5)
