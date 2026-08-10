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
    _canonical_initial_state,
    _verify_checkpoint_identity,
    deterministic_seeds,
    paired_bootstrap,
)
from npa.workflows.groot_visualization import GrootVisualizationError
from npa.workflows import groot_task_performance as task_performance


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


def test_initial_state_canonicalization_excludes_only_derived_coverage() -> None:
    step = {
        "pusher_xy": [77.0, 212.0],
        "object_xy": [195.0, 336.0],
        "object_angle_rad": -1.9,
        "goal_xy": [256.0, 256.0],
        "goal_angle_rad": 0.785,
        "coverage": 0.25958774978670296,
    }
    changed_coverage = {**step, "coverage": 0.2595877497867031}
    assert _canonical_initial_state(step) == _canonical_initial_state(changed_coverage)
    changed_physics = {**step, "object_xy": [196.0, 336.0]}
    assert _canonical_initial_state(step) != _canonical_initial_state(changed_physics)


def test_paired_evidence_excludes_zero_for_consistent_improvement() -> None:
    evidence = paired_bootstrap(
        [0.05 + index / 10_000 for index in range(24)], samples=10_000
    )
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


def test_trained_checkpoint_resolution_requires_completed_gpu_step_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = {
        "s3://bucket/training.json": {
            "schema": "npa.groot.finetune.v1",
            "status": "completed",
            "run_id": "run",
            "checkpoint_uri": "s3://bucket/checkpoint/",
            "optimizer_step_ok": True,
            "collective_ok": True,
            "loss_steps_real": True,
            "loss_decreased": True,
            "model_config_contract": task_performance.GROOT_MODEL_CONFIG_CONTRACT,
            "num_gpus": 7,
            "world_size": 7,
            "distinct_gpu_count": 7,
            "max_steps": 10000,
            "training_step": 10000,
            "checkpoint_steps": [2500, 5000, 7500, 10000],
            "global_batch_size": 224,
            "per_device_batch_size": 8,
            "gradient_accumulation_steps": 4,
            "training_examples": 2_240_000,
        },
        "s3://bucket/split.json": {
            "schema": "npa.groot.episode_split.v1",
            "status": "prepared",
            "run_id": "run",
            "training_plan": {
                "configured_max_steps": 10000,
                "effective_max_steps": 10000,
                "global_batch_size": 224,
                "per_device_batch_size": 8,
                "gradient_accumulation_steps": 4,
            },
        },
    }
    written = {}
    monkeypatch.setattr(
        task_performance, "_read_s3_json", lambda _client, uri: documents[uri]
    )
    monkeypatch.setattr(task_performance, "_download_prefix", lambda *_args: [])
    monkeypatch.setattr(
        task_performance,
        "_resolve_highest_checkpoint_directory",
        lambda path: (path, 10000),
    )
    identities = iter(
        [
            {
                "sha256": "a" * 64,
                "weights_sha256": "b" * 64,
                "objects": 38,
                "bytes": 13_000,
            },
            {
                "sha256": "c" * 64,
                "weights_sha256": "d" * 64,
                "objects": 38,
                "bytes": 13_000,
            },
        ]
    )
    monkeypatch.setattr(
        task_performance,
        "_checkpoint_identity",
        lambda _path: next(identities),
    )
    monkeypatch.setattr(
        task_performance, "checkpoint_model_config_contract", lambda _path: {}
    )
    monkeypatch.setattr(
        task_performance,
        "_put_json",
        lambda _client, uri, payload: written.update({uri: payload}),
    )

    result = task_performance.resolve_trained_checkpoint(
        "s3://bucket/training.json",
        "s3://bucket/split.json",
        "s3://bucket/checkpoint/",
        "s3://bucket/reference.json",
        "run",
        baseline_checkpoint_uri="s3://bucket/baseline/",
        expected_gpu_count=7,
        expected_max_steps=10000,
        s3_client=object(),
    )
    assert result["checkpoint"]["sha256"] == "a" * 64
    assert written["s3://bucket/reference.json"]["training"]["optimizer_steps"] == 10000

    documents["s3://bucket/training.json"]["training_step"] = 9999
    with pytest.raises(GrootVisualizationError, match="did not complete"):
        task_performance.resolve_trained_checkpoint(
            "s3://bucket/training.json",
            "s3://bucket/split.json",
            "s3://bucket/checkpoint/",
            "s3://bucket/reference.json",
            "run",
            baseline_checkpoint_uri="s3://bucket/baseline/",
            expected_gpu_count=7,
            expected_max_steps=10000,
            s3_client=object(),
        )


def test_selection_requires_passed_closed_loop_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = {
        "schema": task_performance.REPORT_SCHEMA,
        "status": "passed",
        "run_id": "run",
        "performance": {"improvement_gate_passed": True},
        "paired_evaluation": {"seed_set_sha256": "validation-seeds"},
    }
    reference = {
        "schema": task_performance.CHECKPOINT_REF_SCHEMA,
        "status": "resolved",
        "run_id": "run",
        "checkpoint": {"uri": "s3://bucket/checkpoint/", "sha256": "b" * 64},
    }
    docs = {"s3://bucket/validation.json": report, "s3://bucket/ref.json": reference}
    written = {}
    monkeypatch.setattr(
        task_performance, "_read_s3_json", lambda _client, uri: docs[uri]
    )
    monkeypatch.setattr(
        task_performance,
        "_put_json",
        lambda _client, uri, payload: written.update({uri: payload}),
    )
    selected = task_performance.select_checkpoint(
        "s3://bucket/validation.json",
        "s3://bucket/ref.json",
        "s3://bucket/selected.json",
        "run",
        s3_client=object(),
    )
    assert selected["status"] == "selected"
    assert selected["validation_seed_set_sha256"] == "validation-seeds"

    report["performance"]["improvement_gate_passed"] = False
    with pytest.raises(GrootVisualizationError, match="passed validation"):
        task_performance.select_checkpoint(
            "s3://bucket/validation.json",
            "s3://bucket/ref.json",
            "s3://bucket/selected.json",
            "run",
            s3_client=object(),
        )
