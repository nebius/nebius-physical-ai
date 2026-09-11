"""Live hosted evaluator contract on synthetic visual event sequences."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from npa.clients.token_factory import DEFAULT_REASONER_MODEL, TokenFactoryClient
from npa.workbench.cosmos.reason import (
    hosted_rollout_model_family,
    run_token_factory_rollout_vlm,
)

from .test_token_factory_e2e import _require_key, _shape_frame

pytestmark = pytest.mark.token_factory_e2e


@pytest.mark.parametrize("completed", [True, False])
def test_live_hosted_rollout_scores_visual_events(tmp_path: Path, completed: bool) -> None:
    """Verify model-local event labels and distinguish successful final geometry.

    These diagrams exercise the same API/parser used by Sim2Real Stage 8. They
    are synthetic inputs, not an Isaac simulation or a robot policy benchmark.
    """
    _require_key()
    frames = [
        _shape_frame(tmp_path / f"camera-{index:03d}.png", red_inside=inside)
        for index, inside in enumerate((False, False, completed))
    ]
    result = run_token_factory_rollout_vlm(
        model_id=DEFAULT_REASONER_MODEL,
        image_paths=frames,
        actions=[{"step": index, "sim_step": index, "action": [0.0], "episode_boundary": _no_reset_boundary()} for index in range(len(frames))],
        frame_metadata=[{"path": frame.name, "sim_step": index, "view_name": "primary",
                         "episode_id": "synthetic-visual-events", "simulator_episode_id": 0} for index, frame in enumerate(frames)],
        task_description=(
            "These are synthetic geometric diagrams. The task is to move the red "
            "square fully inside the green rectangular outline by the final frame. "
            "Judge this visible geometric condition only; no robot is depicted. "
            "Score success only if the final red square is inside the green outline."
        ),
        rollout_id="synthetic-visual-events",
        threshold=0.5,
        client=TokenFactoryClient(),
    )
    artifact = tmp_path / "rollout-evaluation.json"
    artifact.write_text(json.dumps(result, indent=2) + "\n")
    saved = json.loads(artifact.read_text())
    assert saved["schema"] == "npa.sim2real.vlm_eval.v5"
    assert saved["model"] == DEFAULT_REASONER_MODEL
    assert saved["reason_family"] == hosted_rollout_model_family(DEFAULT_REASONER_MODEL)
    assert saved["backend"] == "token_factory"
    assert saved["provider"] == "nebius"
    assert saved["success"] is completed
    assert (saved["score"] >= 0.5) is completed
    assert saved["summary"].strip()
    assert saved["frame_count"] == saved["action_count"] == len(frames)
    assert saved["selected_frames"] == [frame.name for frame in frames]
    assert [event["step"] for event in saved["per_step"]] == list(range(len(frames)))
    for event in saved["per_step"]:
        assert event["critique_source"] == "model_per_step"
        assert event["critique_text"].strip()
        assert event["confidence"] > 0
        assert event["camera_observation"] == saved["selected_frames"][event["step"]]
        assert event["visual_grounding"]["supported"] is True
    assert saved["request"]["request_id"]
    assert saved["request"]["input_tokens"] > 0
    assert saved["request"]["output_tokens"] > 0
    assert saved["request"]["latency_seconds"] > 0


def _no_reset_boundary():
    return {
        "schema": "npa.sim2real.episode_boundary.v1",
        "simulator_episode_id": 0, "action_episode_id": 0,
        "reset_events": [], "reset_on_current_step": False,
        "action_outcome_valid": True, "temporal_credit_valid": True,
    }


def test_live_hosted_autoreset_cannot_support_terminal_action(tmp_path: Path) -> None:
    """Exercise neutral reset bindings through the real hosted response schema."""
    _require_key()
    frames = [_shape_frame(tmp_path / f"camera-{index:03d}.png", red_inside=bool(index))
              for index in range(2)]
    reset = {
        "schema": "npa.sim2real.episode_boundary.v1",
        "simulator_episode_id": 1, "action_episode_id": 0,
        "reset_events": [{"sim_step": 1, "terminated_episode_id": 0, "next_episode_id": 1}],
        "reset_on_current_step": True, "action_outcome_valid": False,
        "temporal_credit_valid": False,
    }
    result = run_token_factory_rollout_vlm(
        model_id=DEFAULT_REASONER_MODEL, image_paths=frames,
        actions=[{"step": index, "sim_step": index, "action": [0.0], "episode_boundary": boundary}
                 for index, boundary in enumerate([_no_reset_boundary(), reset])],
        frame_metadata=[{"path": frame.name, "sim_step": index, "view_name": "primary",
                         "episode_id": "synthetic-reset", "simulator_episode_id": None if index else 0}
                        for index, frame in enumerate(frames)],
        task_description=("Synthetic diagrams show a red square and green outline. The second "
                          "image belongs to a reset episode; it cannot establish action progress."),
        rollout_id="synthetic-reset", threshold=0.5, client=TokenFactoryClient(),
    )
    (tmp_path / "reset-evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    event = result["per_step"][1]
    assert event["episode_boundary"] == reset
    assert event["camera_observation"] is None and event["confidence"] == 0
    assert event["error_tags"] == ["ok"]
    assert event["critique_text"] == "Insufficient visual evidence for step 1."
    assert event["visual_grounding"]["supported"] is False
