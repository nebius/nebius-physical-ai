"""Truth and statistics gates for closed-loop GR00T PushT evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pytest

from npa.workbench.foxglove.inspect import summarize_mcap
from npa.workbench.foxglove.mcap_writer import FrameInput, MetricsInput, write_run_mcap
from npa.workflows.groot_task_performance import (
    _paired_replay_timebase,
    _policy_observation,
    _verify_checkpoint_identity,
    deterministic_seeds,
    paired_bootstrap,
)
from npa.workflows.groot_visualization import GrootVisualizationError


@dataclass
class _Modality:
    modality_keys: list[str]


class _Policy:
    def get_modality_config(self):
        return {
            "video": _Modality(["front"]),
            "state": _Modality(["single_arm", "gripper"]),
            "language": _Modality(["annotation.human.task_description"]),
        }


def test_final_seeds_are_stable_unique_and_require_twenty() -> None:
    first = deterministic_seeds("groot-final", 24)
    assert first == deterministic_seeds("groot-final", 24)
    assert len(first) == len(set(first)) == 24


def test_policy_observation_is_current_pusht_shape() -> None:
    observations = [
        {
            "pixels": np.full((96, 96, 3), index, dtype=np.uint8),
            "agent_pos": np.asarray([100 + index, 200 + index], dtype=np.float32),
        }
        for index in range(3)
    ]
    parsed = _policy_observation(_Policy(), observations)
    assert parsed["video"]["front"].shape == (3, 1, 96, 96, 3)
    assert parsed["state"]["single_arm"].shape == (3, 1, 1)
    assert parsed["state"]["gripper"].shape == (3, 1, 1)
    assert parsed["language"]["annotation.human.task_description"] == [
        ["Push the T-shaped block onto the T-shaped target."],
        ["Push the T-shaped block onto the T-shaped target."],
        ["Push the T-shaped block onto the T-shaped target."],
    ]


def test_checkpoint_identity_uses_canonical_directory_sha256_and_fails_closed() -> None:
    _verify_checkpoint_identity({"sha256": "immutable"}, "immutable")
    with pytest.raises(GrootVisualizationError, match="missing != immutable"):
        _verify_checkpoint_identity({"identity_sha256": "immutable"}, "immutable")


def test_paired_evidence_excludes_zero_for_consistent_improvement() -> None:
    evidence = paired_bootstrap([0.05 + index / 10_000 for index in range(24)], samples=10_000)
    assert evidence["mean_delta"] > 0
    assert evidence["ci_low"] > 0
    assert evidence["p_value"] < 0.05


def test_mcap_and_rrd_replay_timebase_has_explicit_episode_boundary() -> None:
    pair = {"seed": 17}
    baseline = [{"step": index} for index in range(3)]
    trained = [{"step": index} for index in range(5)]
    first = _paired_replay_timebase(pair, baseline, trained)
    second = _paired_replay_timebase(pair, baseline, trained)

    assert first == second
    assert first["sample_count"] == 5
    assert first["baseline_step_count"] == 3
    assert first["trained_step_count"] == 5
    assert first["episode_boundaries"] == [
        {"seed": 17, "start_sample": 0, "end_sample_exclusive": 5}
    ]
    assert first["id"].startswith("sha256:")


def test_mcap_inputs_can_request_exact_rollout_topics(tmp_path) -> None:
    from PIL import Image

    frame = tmp_path / "frame.png"
    Image.new("RGB", (16, 16), "green").save(frame)
    metric = tmp_path / "action.json"
    metric.write_text(json.dumps([{"x": 1.0, "y": 2.0}]))
    output = tmp_path / "task.mcap"
    summary = write_run_mcap(
        output=output,
        frames=[FrameInput(frame, topic="/rollout/baseline/camera")],
        metrics=[MetricsInput(metric, topic="/rollout/baseline/action")],
        run_id="task",
    )
    inspected = summarize_mcap(output)
    assert summary.channels["/rollout/baseline/camera"] == 1
    assert summary.channels["/rollout/baseline/action"] == 1
    assert "/rollout/baseline/camera" in inspected.channels
    assert "/rollout/baseline/action" in inspected.channels
