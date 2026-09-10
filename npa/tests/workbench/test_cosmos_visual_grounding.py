"""Temporal visual evidence must belong to the action it is allowed to shape."""
import json
from copy import deepcopy

import pytest

from npa.workbench.cosmos.reason import CosmosReasonError, _parse_hosted_rollout_output
from npa.workflows.sim2real.byo_isaac_trainer import read_signal_stats
from npa.workbench.cosmos.visual_grounding import (
    bind_action_frames, validate_stored_visual_grounding,
)
from npa.workflows.sim2real.temporal_credit import convert_evaluation
from npa.workbench.lerobot.policy_container import (
    parse_vlm_signal_batch, run_vlm_signal_training_step,
)


def test_hosted_events_without_time_bindings_are_rejected():
    payload = {
        "score": 0.1, "success": False, "summary": "The cube stays on the table.",
        "per_step": [{"step": 1, "camera_observation": "camera-004.png",
                      "confidence": 0.9, "error_tags": ["missed_target"],
                      "critique_text": "The gripper misses the cube."}],
    }
    with pytest.raises(CosmosReasonError, match="visual|frame|grounding"):
        _parse_hosted_rollout_output(
            json.dumps(payload), actions=[{"step": 1, "sim_step": 9, "action": [0.1]}],
            rollout_id="rollout-0000", threshold=0.5, family="minimax_m3",
            frame_names=["camera-004.png"],
        )


def test_zero_confidence_visual_tags_are_excluded_from_trainer_statistics(tmp_path):
    path = tmp_path / "signals.json"
    path.write_text(json.dumps({"signals": [{"per_step": [{
        "step": 1, "reward": -0.2, "advantage": 0.1, "confidence": 0,
        "error_tags": ["missed_target"], "visual_grounding": {"supported": False},
    }]}]}))
    stats = read_signal_stats(str(path))
    assert stats["error_tags"] == {}
    assert stats["step_count"] == 1
    assert stats["mean_reward"] == -0.2


def _capture():
    times = [index * 299 // 31 for index in range(32)] + [300]
    metadata = [{"path": f"camera-{index:03d}.png", "sim_step": sim_step,
                 "view_name": "primary", "episode_id": "rollout-0000"}
                for index, sim_step in enumerate(times)]
    actions = [{"step": index, "sim_step": sim_step, "action": [index / 100],
                "simulator_ground_truth": {"object_goal_distance_m": 0.4 - index / 100,
                                           "scenario_config_digest": "original-config"}}
               for index, sim_step in enumerate(times[:-1])]
    selected = [metadata[index]["path"] for index in [0, 4, 9, 13, 18, 22, 27, 32]]
    return actions, metadata, selected


def _bound_payload(bindings):
    return {"score": 0.1, "success": False, "summary": "The cube remains on the table.",
            "per_step": [{"step": step, "camera_observation": binding["camera_observation"],
                          "confidence": 0.8 if binding["supported"] else 0,
                          "error_tags": ["missed_target"] if binding["supported"] else ["ok"],
                          "critique_text": "Cube stays on table." if binding["supported"]
                          else f"Insufficient visual evidence for step {step}."}
                         for step, binding in bindings.items()]}


def test_sampled_frames_bind_by_simulation_time_including_final_context_frame():
    actions, metadata, selected = _capture()
    bindings = bind_action_frames(actions=actions, frame_metadata=metadata,
                                  frame_names=selected, rollout_id="rollout-0000")
    assert {step for step, binding in bindings.items() if binding["supported"]} == {0, 4, 9, 13, 18, 22, 27}
    assert bindings[4]["camera_observation"] == "camera-004.png"
    assert bindings[4]["action_sim_step"] == bindings[4]["frame_sim_step"] == 38
    assert bindings[31]["action_sim_step"] == 299
    assert bindings[31]["camera_observation"] is None
    assert bind_action_frames(actions=list(reversed(actions)), frame_metadata=list(reversed(metadata)),
                              frame_names=list(reversed(selected)), rollout_id="rollout-0000") == bindings


@pytest.mark.parametrize("corruption", ["duplicate_name", "duplicate_time", "wrong_episode", "wrong_view", "boolean_time", "missing_metadata"])
def test_ambiguous_or_foreign_frame_metadata_is_rejected(corruption):
    actions, metadata, selected = _capture()
    if corruption == "duplicate_name":
        metadata[1]["path"] = metadata[0]["path"]
    elif corruption == "duplicate_time":
        metadata[1]["sim_step"] = metadata[0]["sim_step"]
    elif corruption == "wrong_episode":
        metadata[0]["episode_id"] = "another-rollout"
    elif corruption == "wrong_view":
        metadata[0]["view_name"] = "side"
    elif corruption == "boolean_time":
        metadata[0]["sim_step"] = False
    else:
        metadata.pop(0)
    with pytest.raises(ValueError, match="visual"):
        bind_action_frames(actions=actions, frame_metadata=metadata, frame_names=selected,
                           rollout_id="rollout-0000")


@pytest.mark.parametrize("corruption", ["future_frame", "past_frame", "missing_null_camera", "unsupported_confidence", "unsupported_tag", "unsupported_critique"])
def test_wrong_temporal_evidence_is_rejected_without_relabeling(corruption):
    actions, metadata, selected = _capture()
    bindings = bind_action_frames(actions=actions, frame_metadata=metadata,
                                  frame_names=selected, rollout_id="rollout-0000")
    payload = _bound_payload(bindings)
    if corruption == "future_frame":
        payload["per_step"][1].update(camera_observation="camera-004.png", confidence=0.9)
    elif corruption == "past_frame":
        payload["per_step"][4]["camera_observation"] = "camera-000.png"
    elif corruption == "missing_null_camera":
        payload["per_step"][1].pop("camera_observation")
    elif corruption == "unsupported_confidence":
        payload["per_step"][1]["confidence"] = 0.1
    elif corruption == "unsupported_tag":
        payload["per_step"][1]["error_tags"] = ["collision"]
    else:
        payload["per_step"][1]["critique_text"] = "An unseen collision occurred."
    before = deepcopy(payload)
    with pytest.raises(CosmosReasonError, match="visual|unobserved"):
        _parse_hosted_rollout_output(json.dumps(payload), actions=actions, rollout_id="rollout-0000",
                                    threshold=0.5, family="minimax_m3", frame_names=selected,
                                    visual_bindings=bindings)
    assert payload == before


def test_unobserved_events_keep_ground_truth_without_visual_training_effects(tmp_path):
    actions, metadata, selected = _capture()
    bindings = bind_action_frames(actions=actions, frame_metadata=metadata,
                                  frame_names=selected, rollout_id="rollout-0000")
    result = _parse_hosted_rollout_output(json.dumps(_bound_payload(bindings)), actions=actions,
                                         rollout_id="rollout-0000", threshold=0.5, family="minimax_m3",
                                         frame_names=selected, visual_bindings=bindings)
    result.update(frame_count=8, selected_frames=selected,
                  selected_frame_metadata=[row for row in metadata if row["path"] in selected])
    validate_stored_visual_grounding(result)
    assert result["schema"] == "npa.sim2real.vlm_eval.v4"
    signal = convert_evaluation(result)
    for original, event, step in zip(actions, result["per_step"], signal["per_step"], strict=True):
        assert event["action"] == original["action"]
        assert event["sim_step"] == original["sim_step"]
        assert step["simulator_ground_truth"] == original["simulator_ground_truth"]
        if not bindings[event["step"]]["supported"]:
            assert event["camera_observation"] is None
            assert step["confidence"] == step["reward_components"]["vlm_auxiliary"] == 0
            assert not any(step["target"]["action_delta"])
    assert signal["calibration"]["vlm_unobserved_visual_steps"] == 25
    path = tmp_path / "signal.json"
    path.write_text(json.dumps({"signals": [signal]}))
    stats = read_signal_stats(str(path))
    assert stats["step_count"] == 32 and stats["visual_step_count"] == 7
    assert stats["error_tags"] == {"missed_target": 7}
    # Valid observed corrections still reach the real adapter optimizer.
    parsed = parse_vlm_signal_batch(signal)
    update = run_vlm_signal_training_step(parsed, output_dir=tmp_path / "update")
    control = run_vlm_signal_training_step(parsed, output_dir=tmp_path / "control", control=True)
    assert update.policy_delta_l2 > control.policy_delta_l2
    result["per_step"][1]["camera_observation"] = "camera-004.png"
    with pytest.raises(ValueError, match="binding"):
        validate_stored_visual_grounding(result)
