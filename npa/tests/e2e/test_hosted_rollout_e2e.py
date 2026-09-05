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
        actions=[{"step": index, "action": [0.0]} for index in range(len(frames))],
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
    assert saved["schema"] == "npa.sim2real.vlm_eval.v3"
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
        assert event["camera_observation"] in saved["selected_frames"]
    assert saved["request"]["request_id"]
    assert saved["request"]["input_tokens"] > 0
    assert saved["request"]["output_tokens"] > 0
    assert saved["request"]["latency_seconds"] > 0
