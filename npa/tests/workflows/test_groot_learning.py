"""Truth and integrity gates for the GR00T learning workflow."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from npa.cli.groot.training_evidence import (
    parse_training_loss_evidence,
    render_training_manifest_script,
)
from npa.workbench.foxglove.mcap_writer import (
    FrameInput,
    LogInput,
    MetricsInput,
    write_run_mcap,
)
from npa.workflows import groot_learning as learning


def _evaluation(*, real: bool = True) -> dict:
    return {
        "schema": learning.EVAL_SCHEMA,
        "status": "completed",
        "run_id": "run",
        "phase": "baseline",
        "real_model_forward": real,
        "model_forward_calls": 2,
        "episodes": 1,
        "samples": 3,
        "action_dimensions": 2,
        "action_horizon": 2,
        "sample_alignment": [{"sample_index": index} for index in range(3)],
        "metrics": {
            "mse": 2.0,
            "mae": 1.0,
            "per_dimension_mse": [1.0, 3.0],
            "per_dimension_mae": [0.5, 1.5],
            "per_horizon_mse": [1.0, 3.0],
            "per_horizon_counts": [2, 1],
        },
        "repeat_evaluation": {
            "same_seed_deterministic": True,
            "independent_seed_count": 3,
        },
        "model_config_contract": learning.GROOT_MODEL_CONFIG_CONTRACT,
    }


def test_seeded_cuda_evaluation_configures_deterministic_cublas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    cudnn = SimpleNamespace(benchmark=True, deterministic=False)
    fake_torch = SimpleNamespace(
        manual_seed=lambda seed: calls.__setitem__("torch_seed", seed),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            manual_seed_all=lambda seed: calls.__setitem__("cuda_seed", seed),
        ),
        use_deterministic_algorithms=lambda enabled, warn_only=False: calls.update(
            deterministic=enabled, warn_only=warn_only
        ),
        backends=SimpleNamespace(cudnn=cudnn),
    )
    fake_transformers = SimpleNamespace(
        set_seed=lambda seed, deterministic=False: calls.update(
            transformers_seed=seed,
            transformers_deterministic=deterministic,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    learning._seed_stochastic_sources(1701)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert calls == {
        "torch_seed": 1701,
        "cuda_seed": 1701,
        "deterministic": True,
        "warn_only": True,
        "transformers_seed": 1701,
        "transformers_deterministic": True,
    }
    assert cudnn.benchmark is False
    assert cudnn.deterministic is True


def test_configurable_evaluation_repeats_reuse_one_policy_with_seed_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split = {
        "schema": learning.SPLIT_SCHEMA,
        "run_id": "run",
        "split_hash": "sha256:split",
        "integrity": {"leakage_free": True},
        "source": {"embodiment": "NEW_EMBODIMENT"},
        "train": {"uri": "s3://bucket/train"},
        "heldout": {"uri": "s3://bucket/heldout"},
    }
    monkeypatch.setattr(learning, "_read_s3_json", lambda *_args: split)

    def fake_download(_client: object, uri: str, path: Path) -> None:
        if uri.endswith("train"):
            (path / "meta").mkdir(parents=True)
            (path / "meta" / "stats.json").write_text(
                json.dumps({"action": {"mean": [1.0]}}), encoding="utf-8"
            )

    monkeypatch.setattr(learning, "_download_prefix", fake_download)
    monkeypatch.setattr(
        learning,
        "_initialize_baseline_checkpoint",
        lambda **kwargs: (kwargs["output_path"].mkdir(parents=True), (kwargs["output_path"] / "model.safetensors").write_bytes(b"x")),
    )
    monkeypatch.setattr(
        learning,
        "_upload_directory",
        lambda *_args: {"uri": "s3://bucket/baseline", "sha256": "a" * 64},
    )
    monkeypatch.setattr(
        learning, "checkpoint_model_config_contract", lambda *_args: learning.GROOT_MODEL_CONFIG_CONTRACT
    )
    monkeypatch.setattr(
        learning,
        "_put_bytes",
        lambda _client, uri, payload: {"uri": uri, "bytes": len(payload)},
    )
    monkeypatch.setattr(learning, "_put_json", lambda *_args: None)
    calls: list[tuple[int, object]] = []
    runtime = {"policy": object(), "loader": object(), "tag": object()}

    def fake_evaluate(**kwargs: object) -> dict:
        seed = int(kwargs["seed"])
        supplied_runtime = kwargs.get("runtime")
        calls.append((seed, supplied_runtime))
        prediction = np.asarray([[float(seed)]], dtype=np.float32)
        return {
            "expert": np.asarray([[0.0]], dtype=np.float32),
            "predicted": prediction,
            "states": np.asarray([[0.0]], dtype=np.float32),
            "horizon_indices": np.asarray([0], dtype=np.int64),
            "samples": [{"sample_index": 0}],
            "metrics": {
                "mse": float(seed * seed),
                "mae": float(seed),
                "per_dimension_mse": [float(seed * seed)],
                "per_dimension_mae": [float(seed)],
                "per_dimension_max_abs_error": [float(seed)],
                "per_horizon_mse": [float(seed * seed)],
                "per_horizon_counts": [1],
            },
            "episode_metrics": [{"episode_index": 0}],
            "episode_count": 1,
            "sample_count": 1,
            "action_dimensions": 1,
            "forward_calls": 1,
            "fps": 10.0,
            "action_horizon": 1,
            "evaluation_seed": seed,
            "prediction_sha256": f"seed-{seed}",
            "gpu_name": "test-gpu",
            "_runtime": runtime,
        }

    monkeypatch.setattr(learning, "_evaluate_checkpoint", fake_evaluate)
    result = learning.evaluate(
        "s3://bucket/split.json",
        "",
        "s3://bucket/eval.json",
        "s3://bucket/actions.npz",
        "run",
        "baseline",
        base_model="nvidia/GR00T-N1.7-3B",
        baseline_checkpoint_uri="s3://bucket/baseline",
        action_horizon=1,
        evaluation_seeds=(11, 11, 22, 33),
        evaluation_repeats=4,
        s3_client=object(),
    )

    assert [seed for seed, _runtime in calls] == [11, 11, 22, 33]
    assert calls[0][1] is None
    assert all(supplied is runtime for _seed, supplied in calls[1:])
    assert result["repeat_evaluation"]["configured_repeats"] == 4
    assert result["repeat_evaluation"]["policy_constructions"] == 1
    assert "scales linearly" in result["repeat_evaluation"]["cost_note"]


def test_deterministic_split_is_episode_disjoint_and_stable() -> None:
    first = learning.deterministic_episode_split(
        206, train_episodes=24, heldout_episodes=6, seed="groot17-learning-v1"
    )
    second = learning.deterministic_episode_split(
        206, train_episodes=24, heldout_episodes=6, seed="groot17-learning-v1"
    )
    assert first == second
    assert first["heldout"] == [60, 75, 54, 101, 15, 26]
    assert len(first["train"]) == 24
    assert not (set(first["train"]) & set(first["heldout"]))
    assert len(set(first["train"] + first["heldout"] + first["excluded"])) == 206


def test_custom_modality_metadata_is_required_in_each_materialized_split() -> None:
    assert learning.CUSTOM_DATASET_METADATA == (
        "npa_groot_adapter.json",
        "npa_groot_modality_config.py",
    )


def test_camera_names_are_derived_from_dataset_modality_metadata() -> None:
    cameras = learning._camera_contract(
        {
            "features": {
                "observation.images.overhead": {"dtype": "video"},
                "observation.images.wrist": {"dtype": "video"},
            }
        },
        {
            "video": {
                "overhead": {"original_key": "observation.images.overhead"},
                "wrist_rgb": {"original_key": "observation.images.wrist"},
            }
        },
    )

    assert cameras == [
        {"name": "overhead", "original_key": "observation.images.overhead"},
        {"name": "wrist_rgb", "original_key": "observation.images.wrist"},
    ]


@pytest.mark.parametrize("steps", [2, 4, 1000, 10000])
def test_preflight_couples_final_checkpoint_to_configured_optimizer_steps(
    monkeypatch: pytest.MonkeyPatch, steps: int
) -> None:
    stored: dict[str, object] = {}
    monkeypatch.setattr(
        learning,
        "_put_json",
        lambda _client, _uri, payload: stored.update(payload),
    )
    result = learning.preflight_rigor_contract(
        "s3://bucket/preflight.json",
        "run",
        gpu_type="RTXPRO6000",
        gpu_count=2,
        global_batch_size=2,
        per_device_batch_size=1,
        gradient_accumulation_steps=1,
        train_episodes=2,
        validation_episodes=1,
        final_episodes=0,
        max_steps=steps,
        save_steps=steps,
        save_total_limit=1,
        minimum_epochs=0.001,
        s3_client=object(),
    )
    assert result["max_steps"] == steps
    assert result["checkpoint_schedule"]["save_steps"] == steps


def test_preflight_rejects_checkpoint_schedule_before_gpu_work() -> None:
    with pytest.raises(
        learning.GrootVisualizationError, match="final optimizer step must be saved"
    ):
        learning.preflight_rigor_contract(
            "s3://bucket/preflight.json",
            "run",
            gpu_type="RTXPRO6000",
            gpu_count=2,
            global_batch_size=2,
            per_device_batch_size=1,
            gradient_accumulation_steps=1,
            train_episodes=2,
            validation_episodes=1,
            final_episodes=0,
            max_steps=8,
            save_steps=4,
            save_total_limit=1,
            minimum_epochs=0.001,
            s3_client=object(),
        )


def test_posttrain_evaluation_consumes_resolved_checkpoint_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = {
        "schema": learning.CHECKPOINT_REF_SCHEMA,
        "status": "resolved",
        "run_id": "run",
        "checkpoint": {
            "uri": "s3://bucket/candidate/checkpoint-17/",
            "sha256": "a" * 64,
            "resolved_checkpoint_step": 17,
        },
        "training": {"optimizer_steps": 17},
    }
    monkeypatch.setattr(learning, "_read_s3_json", lambda *_args: reference)
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        learning,
        "evaluate",
        lambda *args, **kwargs: captured.update(args=args, kwargs=kwargs) or {"ok": True},
    )
    result = learning.posttrain_eval(
        "s3://bucket/split.json",
        "s3://bucket/ref.json",
        "s3://bucket/eval.json",
        "s3://bucket/actions.npz",
        "run",
        s3_client=object(),
    )
    assert result == {"ok": True}
    assert captured["args"][1] == "s3://bucket/candidate/checkpoint-17/"
    assert captured["kwargs"]["expected_checkpoint_step"] == 17
    assert captured["kwargs"]["expected_checkpoint_sha256"] == "a" * 64


def test_episode_timebase_covers_every_episode_and_is_contiguous() -> None:
    alignment = [
        {"sample_index": 0, "episode_index": 0, "frame_index": 0},
        {"sample_index": 1, "episode_index": 0, "frame_index": 1},
        {"sample_index": 2, "episode_index": 1, "frame_index": 0},
    ]
    timebase = learning._episode_timebase(
        alignment, fps=10.0, camera_names=["overhead"]
    )

    assert timebase["sample_count"] == 3
    assert timebase["entries"][-1]["time_seconds"] == pytest.approx(0.2)
    assert timebase["episode_boundaries"] == [
        {
            "episode_index": 0,
            "start_sample": 0,
            "end_sample_exclusive": 2,
            "sample_count": 2,
        },
        {
            "episode_index": 1,
            "start_sample": 2,
            "end_sample_exclusive": 3,
            "sample_count": 1,
        },
    ]
    assert timebase["id"].startswith("sha256:")


def test_video_inventory_requires_every_camera_for_every_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objects = [
        {
            "bucket": "bucket",
            "key": f"heldout/videos/chunk-000/observation.image/episode_{episode:06d}.mp4",
            "size": 10,
        }
        for episode in range(2)
    ]
    monkeypatch.setattr(learning, "_list_objects", lambda *_args: objects)
    inventory = learning._heldout_video_inventory(
        object(),
        "s3://bucket/heldout/",
        cameras=[{"name": "front_rgb", "original_key": "observation.image"}],
        episode_count=2,
    )
    assert [item["episode_index"] for item in inventory] == [0, 1]
    assert {item["camera_name"] for item in inventory} == {"front_rgb"}

    with pytest.raises(learning.GrootVisualizationError, match="inventory mismatch"):
        learning._heldout_video_inventory(
            object(),
            "s3://bucket/heldout/",
            cameras=[
                {"name": "front_rgb", "original_key": "observation.image"},
                {"name": "wrist", "original_key": "observation.wrist"},
            ],
            episode_count=2,
        )


def test_custom_modality_loader_fails_closed_when_registration_is_absent(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        learning.GrootVisualizationError, match="modality config is absent"
    ):
        learning._load_custom_modality_config(tmp_path)


def test_real_evaluation_validation_rejects_json_only_or_misaligned_outputs() -> None:
    payload = _evaluation(real=False)
    with pytest.raises(learning.GrootVisualizationError, match="real model forward"):
        learning.validate_evaluation(payload, phase="baseline", run_id="run")
    payload = _evaluation()
    payload["sample_alignment"] = payload["sample_alignment"][:-1]
    with pytest.raises(learning.GrootVisualizationError, match="alignment"):
        learning.validate_evaluation(payload, phase="baseline", run_id="run")


def test_predicted_expert_alignment_rejects_shape_and_nonfinite_values() -> None:
    expert = np.zeros((3, 2), dtype=np.float32)
    learning.validate_action_alignment(expert, expert.copy(), label="test")
    with pytest.raises(learning.GrootVisualizationError, match="alignment"):
        learning.validate_action_alignment(expert, np.zeros((2, 2)), label="test")
    bad = expert.copy()
    bad[0, 0] = np.nan
    with pytest.raises(learning.GrootVisualizationError, match="non-finite"):
        learning.validate_action_alignment(expert, bad, label="test")


def test_rrd_squared_error_is_mean_of_squared_residuals_not_squared_mae() -> None:
    mae, mse = learning._per_sample_action_errors(
        np.asarray([0.0, 0.0]), np.asarray([1.0, 3.0])
    )
    assert mae == pytest.approx(2.0)
    assert mse == pytest.approx(5.0)
    assert mse != pytest.approx(mae**2)


def test_comparison_discloses_per_dimension_regression_and_gates_primary_metric() -> (
    None
):
    improved = learning.compare_metrics(
        {"mse": 2.0, "per_dimension_mse": [1.0, 3.0]},
        {"mse": 1.5, "per_dimension_mse": [1.1, 1.9]},
    )
    assert improved["improved"] is True
    assert improved["absolute_improvement"] == pytest.approx(0.5)
    assert [item["dimension"] for item in improved["regressions"]] == [0]
    learning.require_learning_improvement(improved)
    failed = learning.compare_metrics(
        {"mse": 1.0, "per_dimension_mse": [1.0]},
        {"mse": 1.1, "per_dimension_mse": [1.1]},
    )
    with pytest.raises(learning.GrootVisualizationError, match="did not improve"):
        learning.require_learning_improvement(failed)


def test_training_coverage_uses_factual_batch_for_maximum_free_gpu_allocation() -> None:
    coverage = learning.calculate_training_coverage(
        optimizer_steps=480, global_batch_size=7, train_samples=3354
    )
    assert coverage == {
        "training_examples": 3360,
        "epoch_equivalent": pytest.approx(1.0017889087656529),
    }
    short = learning.calculate_training_coverage(
        optimizer_steps=4, global_batch_size=2, train_samples=201
    )
    assert short == {
        "training_examples": 8,
        "epoch_equivalent": pytest.approx(8 / 201),
    }


def test_training_step_contract_derives_prior_480_step_case() -> None:
    contract = learning.derive_training_step_contract(
        train_samples=3354, global_batch_size=7
    )
    assert contract["required_optimizer_steps"] == 480
    assert contract["effective_max_steps"] == 480
    assert contract["configured_max_steps"] is None


def test_training_step_contract_changes_with_dataset_size() -> None:
    contract = learning.derive_training_step_contract(
        train_samples=101, global_batch_size=8
    )
    assert contract["required_optimizer_steps"] == 13
    assert contract["effective_max_steps"] == 13


def test_training_step_contract_rejects_insufficient_explicit_budget() -> None:
    with pytest.raises(learning.GrootVisualizationError, match="at least 13 steps"):
        learning.derive_training_step_contract(
            train_samples=101, global_batch_size=8, configured_max_steps=12
        )


class _Blueprint:
    def __init__(self) -> None:
        self.views: list[dict] = []
        self.PanelState = SimpleNamespace(Hidden="hidden", Expanded="expanded")

    def __getattr__(self, name: str):
        def make(*args, **kwargs):
            value = {"kind": name, "args": list(args), **kwargs}
            if name.endswith("View"):
                self.views.append(value)
            return value

        return make


def test_rrd_blueprint_has_only_panels_that_render_on_the_default_timeline() -> (
    None
):
    rrb = _Blueprint()
    learning._learning_blueprint(rrb)
    origins = {str(view.get("origin")) for view in rrb.views}
    assert {
        "heldout/camera/front",
        "actions",
        "error",
        "metrics",
        "provenance",
    } <= origins
    assert "train" not in origins
    assert "validation" not in origins
    metrics = next(view for view in rrb.views if view.get("origin") == "metrics")
    assert metrics["contents"] == [
        "metrics/heldout_before/**",
        "metrics/heldout_after/**",
    ]
    assert set(learning.REQUIRED_RRD_ENTITIES) >= {
        "heldout/camera/front",
        "actions/expert/dim_0",
        "actions/predicted_after/dim_0",
        "error/after/absolute",
    }


def test_learning_rrd_is_closed_with_a_parseable_footer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("rerun")
    image_module = pytest.importorskip("PIL.Image")
    report = {
        "run_id": "run",
        "dataset": {
            "split_hash": "split",
            "train_episodes": 1,
            "heldout_episodes": 1,
            "heldout_samples": 2,
            "source_resolution": "8x6",
            "camera_names": ["overhead"],
        },
        "training": {
            "checkpoint_uri": "s3://bucket/checkpoint",
            "loss_history": [{"optimizer_step": 1, "loss": 0.5}],
            "resolved_checkpoint_step": 1,
        },
        "evaluation": {
            "baseline_value": 2.0,
            "posttrain_value": 1.0,
            "candidate_skill_score": 0.1,
            "per_horizon_mse": {
                "baseline": [2.0],
                "posttrain": [1.0],
                "counts": [2],
                "action_horizon": 1,
            },
        },
        "provenance": {
            "primary_camera": "overhead",
            "heldout_source_videos": [
                {
                    "episode_index": 0,
                    "camera_name": "overhead",
                    "uri": "s3://bucket/video.mp4",
                }
            ],
        },
        "visualizations": {
            "timebase": learning._episode_timebase(
                [
                    {"sample_index": 0, "episode_index": 0, "frame_index": 0},
                    {"sample_index": 1, "episode_index": 0, "frame_index": 1},
                ],
                fps=10.0,
                camera_names=["overhead"],
            )
        },
    }
    arrays = {
        "expert": np.zeros((2, 1), dtype=np.float32),
        "predicted": np.ones((2, 1), dtype=np.float32),
    }
    post_arrays = {
        "expert": arrays["expert"],
        "predicted": np.full((2, 1), 0.5, dtype=np.float32),
    }
    monkeypatch.setattr(
        learning,
        "_evaluation_bundle",
        lambda *_args: (
            report,
            {
                "fps": 10.0,
                "sample_alignment": [
                    {"sample_index": 0, "episode_index": 0, "frame_index": 0},
                    {"sample_index": 1, "episode_index": 0, "frame_index": 1},
                ],
            },
            {},
            arrays,
            post_arrays,
        ),
    )
    monkeypatch.setattr(learning, "_download", lambda *_args: None)
    monkeypatch.setattr(
        learning,
        "_decode_synchronized_camera",
        lambda *_args, **_kwargs: ([image_module.new("RGB", (8, 6))] * 2, 10.0),
    )
    uploaded: dict[str, bytes] = {}

    def capture(_client, uri: str, payload: bytes) -> dict:
        uploaded[uri] = payload
        return {"uri": uri, "bytes": len(payload), "sha256": "test"}

    monkeypatch.setattr(learning, "_put_bytes", capture)
    result = learning.emit_learning_rrd(
        "s3://bucket/report.json",
        "s3://bucket/learning.rrd",
        "run",
        s3_client=object(),
    )
    assert result["inspect"]["parseable"] is True
    assert result["inspect"]["bytes"] == len(uploaded["s3://bucket/learning.rrd"])


def test_operational_smoke_uses_final_evaluation_as_checkpoint_curve() -> None:
    curve = learning._evaluated_checkpoint_curve(
        {
            "training": {"resolved_checkpoint_step": 4},
            "evaluation": {
                "posttrain_value": 12.5,
                "candidate_skill_score": -0.25,
            },
        },
    )

    assert curve == [
        {
            "optimizer_step": 4,
            "mse": 12.5,
            "skill_score": -0.25,
            "source": "final_evaluated_smoke_checkpoint",
        }
    ]


def test_learning_mcap_topics_use_real_camera_log_and_metric_schemas(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mcap")
    image_module = pytest.importorskip("PIL.Image")
    frame = tmp_path / "000000.png"
    image_module.new("RGB", (8, 6), (20, 40, 60)).save(frame, "PNG")
    inputs = []
    for name in (
        "policy/predicted_action",
        "expert/action",
        "metrics/action_error",
        "metrics/heldout_before",
        "metrics/heldout_after",
        "metrics/train_loss",
        "metrics/per_horizon_error",
        "metrics/checkpoint_curve",
    ):
        path = tmp_path / f"{name.replace('/', '_')}.json"
        path.write_text(json.dumps([{"value": 1.0}]), encoding="utf-8")
        inputs.append(MetricsInput(path=path, name=name))
    log = tmp_path / "offline.log"
    log.write_text(
        "Offline held-out policy evaluation; not rollout.\n", encoding="utf-8"
    )
    output = tmp_path / "learning.mcap"
    write_run_mcap(
        output=output,
        frames=[FrameInput(frame, camera="front", timestamp_ns=1)],
        metrics=inputs,
        logs=[LogInput(log)],
        fps=10,
        start_time_ns=1,
        run_id="run",
        camera_topic_prefix="/camera",
        metrics_topic_prefix="",
        metadata={
            "evaluation_kind": learning.EVALUATION_KIND,
            "timestamps": learning.TIMESTAMP_SEMANTICS,
            "timeline_origin": "relative-zero-plus-1ns",
            "training_loss_clock": "optimizer_step-as-seconds",
            "is_robot_capture_time": "false",
            "primary_camera": "front",
            "timebase_id": "sha256:test-timebase",
            "dataset_sample_count": "1",
            "dataset_end_time_ns": "1",
            "declared_end_time_ns": str(1 + (len(inputs) - 1) * 100_000_000),
        },
    )
    inspected = learning._validate_learning_mcap(output, run_id="run")
    assert set(learning.REQUIRED_MCAP_TOPICS) <= set(inspected["channels"])
    assert inspected["start_time_ns"] == 1
    assert inspected["timestamps_in_int64_domain"] is True
    assert inspected["channels_monotonic"] is True


def test_comparison_video_preserves_native_camera_pixels_and_truthful_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("av")
    image_module = pytest.importorskip("PIL.Image")
    images = [image_module.new("RGB", (8, 6), (200, 10, 10)) for _ in range(2)]
    monkeypatch.setattr(
        learning, "_decode_video", lambda *_args, **_kwargs: (images, 10.0)
    )
    actions = np.zeros((2, 2), dtype=np.float32)
    meta = learning._comparison_video(
        tmp_path / "ignored.mp4",
        tmp_path / "comparison.mp4",
        expert=actions,
        baseline=actions + 1,
        posttrain=actions + 0.5,
        baseline_mse=1.0,
        posttrain_mse=0.25,
    )
    assert meta["resolution"] == "640x360"
    assert meta["source_resolution"] == "8x6"
    assert meta["native_camera_scale"] == "1:1"
    assert meta["label"] == learning.EVALUATION_KIND


@pytest.mark.parametrize("size", [(1280, 720), (720, 1280), (641, 361)])
def test_comparison_video_canvas_never_crops_large_or_portrait_native_frames(
    tmp_path: Path, size: tuple[int, int]
) -> None:
    pytest.importorskip("av")
    image_module = pytest.importorskip("PIL.Image")
    image = image_module.new("RGB", size, (10, 20, 30))
    actions = np.zeros((1, 1), dtype=np.float32)
    meta = learning._comparison_video_frames(
        [image],
        10.0,
        tmp_path / f"comparison-{size[0]}x{size[1]}.mp4",
        expert=actions,
        baseline=actions,
        posttrain=actions,
        baseline_mse=0.0,
        posttrain_mse=0.0,
    )
    canvas_width, canvas_height = (int(value) for value in meta["resolution"].split("x"))
    region = meta["camera_region"]
    assert canvas_width >= region["x"] + size[0]
    assert canvas_height >= region["y"] + size[1]
    assert region["width"] == size[0]
    assert region["height"] == size[1]
    assert meta["native_resolution_preserved"] is True
    assert meta["native_camera_scale"] == "1:1"


@pytest.mark.parametrize("camera", ["front", "overhead_rgb"])
def test_publish_learning_uses_report_primary_camera_everywhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, camera: str
) -> None:
    report_uri = "s3://bucket/run/reports/two-gpu-pipeline-report.json"
    output_uri = "s3://bucket/run/reports/publish-manifest.json"
    report = {
        "schema": learning.REPORT_SCHEMA,
        "run_id": "run",
        "learning_outcome": "not_improved",
        "candidate_promoted": False,
        "dataset": {"camera_names": [camera]},
        "provenance": {
            "primary_camera": camera,
            "heldout_source_videos": [{"camera_name": camera}],
        },
        "visualizations": {
            "timebase": {"camera_names": [camera]},
            "comparison_video": {"resolution": "640x360"},
        },
    }
    stored: dict[str, dict] = {}
    calls: dict[str, object] = {}
    monkeypatch.setattr(learning, "_read_s3_json", lambda *_args: report)
    monkeypatch.setattr(
        learning,
        "_read_s3_bytes",
        lambda *_args: b"apiVersion: npa.workflow/v0.0.1\nkind: Workflow\n",
    )
    monkeypatch.setattr(
        learning, "_download", lambda _client, _uri, path: path.write_bytes(b"artifact")
    )
    monkeypatch.setattr(
        learning,
        "_validate_learning_mcap",
        lambda _path, *, run_id, camera_name: calls.update(
            mcap_camera=camera_name, mcap_run=run_id
        )
        or {"size_bytes": 8, "channels": {f"/camera/{camera_name}": 1}},
    )
    monkeypatch.setattr(
        learning,
        "inspect_rrd",
        lambda _path, **kwargs: calls.update(rrd_entities=kwargs["expected_entities"])
        or {"bytes": 8},
    )
    monkeypatch.setattr(
        learning,
        "_decode_video",
        lambda *_args, **_kwargs: ([SimpleNamespace(size=(640, 360))], 10.0),
    )
    monkeypatch.setattr(
        learning,
        "_head_artifact",
        lambda _client, uri: {"uri": uri, "bytes": 8},
    )
    monkeypatch.setattr(
        learning,
        "_put_json",
        lambda _client, uri, payload: stored.__setitem__(uri, payload),
    )

    result = learning.publish_learning(
        report_uri,
        "s3://bucket/run/reports/groot-offline-evaluation.mcap",
        "s3://bucket/run/reports/groot-offline-evaluation.rrd",
        "s3://bucket/run/reports/offline-heldout-comparison.mp4",
        "s3://bucket/run/workflow.yaml",
        output_uri,
        "run",
        s3_client=object(),
    )

    assert calls["mcap_camera"] == camera
    assert f"heldout/camera/{camera}" in calls["rrd_entities"]
    assert stored[report_uri]["visualizations"]["primary_camera"] == camera
    assert f"/camera/{camera}" in result["mcap_inspection"]["channels"]


def test_publish_learning_rejects_missing_or_mismatched_primary_camera(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = {
        "schema": learning.REPORT_SCHEMA,
        "run_id": "run",
        "dataset": {"camera_names": ["front"]},
        "provenance": {
            "primary_camera": "overhead",
            "heldout_source_videos": [{"camera_name": "front"}],
        },
        "visualizations": {
            "timebase": {"camera_names": ["front"]},
            "comparison_video": {"resolution": "640x360"},
        },
    }
    monkeypatch.setattr(learning, "_read_s3_json", lambda *_args: report)
    monkeypatch.setattr(
        learning,
        "_read_s3_bytes",
        lambda *_args: b"apiVersion: npa.workflow/v0.0.1\nkind: Workflow\n",
    )
    with pytest.raises(learning.GrootVisualizationError, match="primary camera"):
        learning.publish_learning(
            "s3://bucket/report.json",
            "s3://bucket/report.mcap",
            "s3://bucket/report.rrd",
            "s3://bucket/report.mp4",
            "s3://bucket/workflow.yaml",
            "s3://bucket/publish.json",
            "run",
            s3_client=object(),
        )


def test_training_manifest_records_real_loss_trajectory(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "checkpoint-420").mkdir()
    (output / "checkpoint-420" / "model.bin").write_bytes(b"real")
    (output / "training.log").write_text(
        "{'loss': 1.2, 'step': 10, 'epoch': 0.02}\n"
        "{'loss': 0.8, 'step': 20, 'epoch': 1.0}\n"
        "{'train_loss': 0.91, 'epoch': 1.0}\n",
        encoding="utf-8",
    )
    (output / "npa_groot_distributed_evidence.json").write_text(
        json.dumps(
            {
                "world_size": 2,
                "distinct_gpu_count": 2,
                "gpu_uuids": ["a", "b"],
                "ranks": [
                    {"rank": 0, "cuda_device_name": "GPU"},
                    {"rank": 1, "cuda_device_name": "GPU"},
                ],
                "collective_sum": 3,
                "collective_expected": 3,
                "collective_ok": True,
            }
        ),
        encoding="utf-8",
    )
    training_ranks = output / "training-ranks"
    training_ranks.mkdir()
    for rank in (0, 1):
        (training_ranks / f"rank-{rank:04d}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "local_rank": rank,
                    "world_size": 2,
                    "status": "completed_vendor_training",
                }
            ),
            encoding="utf-8",
        )
    manifest = tmp_path / "manifest.json"
    script = render_training_manifest_script(
        output_dir=str(output),
        manifest_path=str(manifest),
        manifest_fields={"global_batch_size": 8, "logging_steps": 10},
    )
    assert "from npa." not in script
    assert "def parse_training_loss_evidence(" in script
    exec(compile(script, "training-manifest", "exec"), {})
    payload = json.loads(manifest.read_text())
    assert payload["training_step"] == 420
    assert payload["training_examples"] == 3360
    assert payload["loss_history"] == [
        {"optimizer_step": 10, "loss": 1.2},
        {"optimizer_step": 20, "loss": 0.8},
    ]
    assert payload["aggregate_train_loss"] == pytest.approx(0.91)
    assert payload["final_step_loss"] == pytest.approx(0.8)
    assert payload["aggregate_train_loss"] != payload["final_step_loss"]
    assert payload["loss_step_inference"] is None
    assert payload["observed_ranks"] == [0, 1]
    assert payload["training_observed_ranks"] == [0, 1]
    assert payload["both_ranks_trained"] is True
    assert payload["rank_zero_checkpoint_only"] is True
    assert payload["checkpoint_upload_invocations"] == 1


def test_text_loss_never_synthesizes_optimizer_steps_from_logging_interval() -> None:
    with pytest.raises(ValueError, match="refusing to synthesize steps"):
        parse_training_loss_evidence(
            "{'loss': 1.2}\n{'loss': 0.8}\n{'train_loss': 0.91}\n",
            training_step=420,
            logging_steps=10,
        )


def test_text_loss_without_steps_or_trainer_config_fails_closed() -> None:
    with pytest.raises(ValueError, match="actual optimizer/global steps"):
        parse_training_loss_evidence(
            "{'loss': 1.2}\n", training_step=420, logging_steps=None
        )


def test_loss_step_provenance_is_truthful_for_aggregate_only_and_empty_logs() -> None:
    aggregate = parse_training_loss_evidence(
        "{'train_loss': 0.91}\n", training_step=4, logging_steps=1
    )
    assert aggregate["loss_step_source"] == "aggregate_train_loss_only"
    assert aggregate["loss_step_inference"] is None
    assert aggregate["loss_history"] == []
    empty = parse_training_loss_evidence("", training_step=4, logging_steps=1)
    assert empty["loss_step_source"] == "no_loss_records"
    assert empty["loss_logging_cadence_matches"] is None


def test_learning_ui_is_replay_first_groups_frames_and_never_upscales() -> None:
    source = (
        Path(__file__).resolve().parents[2] / "src" / "npa" / "cli" / "agent_ui.html"
    ).read_text(encoding="utf-8")
    assert "Policy learning summary" in source
    assert "Offline held-out policy evaluation" in source
    assert "Open GR00T offline RRD" in source
    assert "Open GR00T offline MCAP" in source
    assert "groupArtifactSequences" in source
    assert "native; preview is not enlarged" in source
    assert "width: auto" in source and "object-fit: contain" in source
    path_contract = (
        Path(__file__).resolve().parents[2] / "src" / "npa" / "workflows" / "artifacts.py"
    ).read_text(encoding="utf-8")
    assert "Prepare leakage-free split" in path_contract
    assert "Synchronized diagnostics" in path_contract


def test_trivial_predictor_floor_and_positive_skill_score() -> None:
    expert = np.asarray([[1.0], [3.0]], dtype=np.float32)
    weak = learning.trivial_predictor_metrics(expert, np.zeros_like(expert), [2.0])
    skilled = learning.trivial_predictor_metrics(
        expert, np.asarray([[1.0], [2.0]]), [2.0]
    )
    assert weak["zero_predictor_mse"] == pytest.approx(5.0)
    assert weak["train_mean_predictor_mse"] == pytest.approx(1.0)
    assert weak["skill_score"] == pytest.approx(-4.0)
    assert skilled["skill_score"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("metric_gate", "loss_decreased", "outcome"),
    [
        (True, True, "improved"),
        (True, False, "not_improved"),
        (False, True, "not_improved"),
    ],
)
def test_operational_learning_status_is_separate_and_never_promotes(
    metric_gate: bool, loss_decreased: bool, outcome: str
) -> None:
    decision = learning.operational_learning_decision(
        {"gate_passed": metric_gate}, {"loss_decreased": loss_decreased}
    )
    assert decision == {
        "pipeline_status": "succeeded",
        "learning_outcome": outcome,
        "candidate_promoted": False,
    }


def test_not_improved_report_remains_valid_artifact_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = {
        "schema": learning.REPORT_SCHEMA,
        "status": "completed",
        "pipeline_status": "succeeded",
        "learning_outcome": "not_improved",
        "candidate_promoted": False,
        "run_id": "run",
        "evaluation": {
            "baseline_uri": "s3://bucket/baseline.json",
            "posttrain_uri": "s3://bucket/trained.json",
        },
    }
    baseline = {"arrays": {"uri": "s3://bucket/baseline.npz"}}
    trained = {"arrays": {"uri": "s3://bucket/trained.npz"}}
    documents = {
        "s3://bucket/report.json": report,
        "s3://bucket/baseline.json": baseline,
        "s3://bucket/trained.json": trained,
    }
    arrays = {
        "s3://bucket/baseline.npz": {"predicted": np.zeros((2, 1))},
        "s3://bucket/trained.npz": {"predicted": np.ones((2, 1))},
    }
    monkeypatch.setattr(
        learning, "_read_s3_json", lambda _client, uri: documents[uri]
    )
    monkeypatch.setattr(
        learning, "validate_evaluation", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(learning, "_read_npz", lambda _client, uri: arrays[uri])

    bundle = learning._evaluation_bundle(object(), "s3://bucket/report.json")
    assert bundle[0]["learning_outcome"] == "not_improved"
    assert bundle[0]["pipeline_status"] == "succeeded"


@pytest.mark.parametrize(
    ("after", "kwargs", "failure"),
    [
        (
            {"mse": 0.95, "per_dimension_mse": [0.95], "skill_score": 0.2},
            {"minimum_relative_improvement": 0.10},
            "relative improvement",
        ),
        (
            {"mse": 0.80, "per_dimension_mse": [0.80], "skill_score": 0.2},
            {"repeat_noise_spread": 0.1, "repeat_noise_multiple": 3.0},
            "repeat-noise",
        ),
        (
            {"mse": 0.80, "per_dimension_mse": [1.1], "skill_score": 0.2},
            {"max_dimension_regression": 0.0},
            "per-dimension regression",
        ),
    ],
)
def test_meaningful_gate_blocks_epsilon_noise_and_dimension_regression(
    after: dict, kwargs: dict, failure: str
) -> None:
    result = learning.compare_metrics(
        {"mse": 1.0, "per_dimension_mse": [1.0]}, after, **kwargs
    )
    assert any(failure in item for item in result["gate_failures"])
    with pytest.raises(learning.GrootVisualizationError, match="learning gate failed"):
        learning.require_learning_improvement(result)


def test_weight_only_digest_is_separate_and_checkpoint_resolution_is_exact(
    tmp_path: Path,
) -> None:
    root = tmp_path / "output"
    for step, weight in ((2500, b"first"), (10000, b"final")):
        checkpoint = root / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "model.safetensors").write_bytes(weight)
        (checkpoint / "trainer_state.json").write_text("{}")
    resolved, step = learning._resolve_highest_checkpoint_directory(root)
    identity = learning._checkpoint_identity(resolved)
    assert step == 10000
    assert identity["sha256"] != identity["weights_sha256"]
    assert identity["weight_objects"] == 1
    with pytest.raises(learning.GrootVisualizationError, match="equal baseline"):
        learning.require_distinct_trained_weights(identity, identity)


def test_robust_loss_gate_preserves_flat_regions_and_requires_decrease() -> None:
    flat = [{"optimizer_step": i, "loss": 1.0} for i in range(1, 11)]
    assert learning.robust_loss_decrease(flat)["loss_decreased"] is False
    decreasing = [
        {"optimizer_step": i, "loss": value}
        for i, value in enumerate([1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.6, 0.5, 0.5], 1)
    ]
    evidence = learning.robust_loss_decrease(decreasing)
    assert evidence["loss_decreased"] is True
    assert len(decreasing) == 10
    smoke = [
        {"optimizer_step": i, "loss": value}
        for i, value in enumerate([1.4, 1.3, 1.2, 1.1], 1)
    ]
    smoke_evidence = learning.robust_loss_decrease(smoke)
    assert smoke_evidence["window"] == 2
    assert smoke_evidence["loss_decreased"] is True


def test_absolute_action_configuration_contract_is_identical(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "experiment_cfg").mkdir(parents=True)
    (checkpoint / "experiment_cfg" / "final_model_config.json").write_text(
        json.dumps(
            {
                key: value
                for key, value in learning.GROOT_MODEL_CONFIG_CONTRACT.items()
                if key != "action_representation"
            }
        )
    )
    (checkpoint / "processor_config.json").write_text(
        json.dumps(
            {
                "processor_kwargs": {
                    "use_relative_action": False,
                    "modality_configs": {
                        "new_embodiment": {
                            "action": {
                                "action_configs": [
                                    {"rep": "ABSOLUTE"},
                                    {"rep": "ABSOLUTE"},
                                ]
                            }
                        }
                    },
                }
            }
        )
    )
    assert learning.checkpoint_model_config_contract(checkpoint) == (
        learning.GROOT_MODEL_CONFIG_CONTRACT
    )
    assert learning.GROOT_MODEL_CONFIG_CONTRACT["use_relative_action"] is False


def test_expanded_split_and_real_batch_budget_contract() -> None:
    split = learning.deterministic_experiment_split(
        206,
        train_episodes=154,
        validation_episodes=26,
        final_episodes=26,
        seed="rigorous",
    )
    assert {name: len(values) for name, values in split.items()} == {
        "validation": 26,
        "final": 26,
        "train": 154,
        "excluded": 0,
    }
    contract = learning.derive_training_step_contract(
        train_samples=19_000,
        global_batch_size=224,
        configured_max_steps=10_000,
        minimum_epochs=2.0,
        minimum_effective_global_batch=128,
        gpu_count=7,
        per_device_batch_size=8,
        gradient_accumulation_steps=4,
    )
    assert contract["epoch_equivalent"] > 2
    with pytest.raises(learning.GrootVisualizationError, match="below required"):
        learning.derive_training_step_contract(
            train_samples=19_000,
            global_batch_size=7,
            minimum_epochs=2,
            minimum_effective_global_batch=128,
        )


def test_stats_emit_canonical_quantiles(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "episode.parquet"
    pq.write_table(
        pa.table(
            {
                "action": pa.array(
                    [[0.0], [1.0], [2.0]], type=pa.list_(pa.float32(), 1)
                ),
                "observation.state": pa.array(
                    [[2.0], [3.0], [4.0]], type=pa.list_(pa.float32(), 1)
                ),
            }
        ),
        path,
    )
    stats = learning._write_dataset_stats([path])
    assert {"q10", "q50", "q90"} <= set(stats["action"])


def test_denoising_knob_is_not_exposed_when_upstream_ignores_it() -> None:
    parser = learning.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "posttrain-eval",
                "--split-manifest-uri",
                "s3://bucket/split",
                "--checkpoint-uri",
                "s3://bucket/checkpoint",
                "--output-uri",
                "s3://bucket/eval",
                "--arrays-uri",
                "s3://bucket/arrays",
                "--run-id",
                "run",
                "--denoising-steps",
                "4",
            ]
        )
