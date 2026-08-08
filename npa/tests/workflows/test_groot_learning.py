"""Truth and integrity gates for the GR00T learning workflow."""

from __future__ import annotations

import json
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
        "sample_alignment": [{"sample_index": index} for index in range(3)],
        "metrics": {
            "mse": 2.0,
            "mae": 1.0,
            "per_dimension_mse": [1.0, 3.0],
            "per_dimension_mae": [0.5, 1.5],
        },
    }


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
    with pytest.raises(learning.GrootVisualizationError, match="modality config is absent"):
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


def test_comparison_discloses_per_dimension_regression_and_gates_primary_metric() -> None:
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
    with pytest.raises(learning.GrootVisualizationError, match="complete train-set pass"):
        learning.calculate_training_coverage(
            optimizer_steps=479, global_batch_size=7, train_samples=3354
        )


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


def test_rrd_blueprint_has_camera_action_error_metric_loss_and_provenance_panels() -> None:
    rrb = _Blueprint()
    learning._learning_blueprint(rrb)
    origins = {str(view.get("origin")) for view in rrb.views}
    assert {
        "heldout/camera/front",
        "actions",
        "error",
        "metrics",
        "train",
        "provenance",
    } <= origins
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
        },
        "evaluation": {"baseline_value": 2.0, "posttrain_value": 1.0},
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


def test_learning_mcap_topics_use_real_camera_log_and_metric_schemas(tmp_path: Path) -> None:
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
    ):
        path = tmp_path / f"{name.replace('/', '_')}.json"
        path.write_text(json.dumps([{"value": 1.0}]), encoding="utf-8")
        inputs.append(MetricsInput(path=path, name=name))
    log = tmp_path / "offline.log"
    log.write_text("Offline held-out policy evaluation; not rollout.\n", encoding="utf-8")
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
        },
    )
    inspected = learning._validate_learning_mcap(output, run_id="run")
    assert set(learning.REQUIRED_MCAP_TOPICS) <= set(inspected["channels"])
    assert inspected["start_time_ns"] == 1
    assert inspected["end_time_ns"] < 1_000_000_000_000


def test_comparison_video_preserves_native_camera_pixels_and_truthful_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("av")
    image_module = pytest.importorskip("PIL.Image")
    images = [image_module.new("RGB", (8, 6), (200, 10, 10)) for _ in range(2)]
    monkeypatch.setattr(learning, "_decode_video", lambda *_args, **_kwargs: (images, 10.0))
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
                "ranks": [{"cuda_device_name": "GPU"}],
                "collective_sum": 3,
                "collective_expected": 3,
                "collective_ok": True,
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


def test_text_loss_step_inference_is_explicit_and_never_forces_final_checkpoint() -> None:
    evidence = parse_training_loss_evidence(
        "{'loss': 1.2}\n{'loss': 0.8}\n{'train_loss': 0.91}\n",
        training_step=420,
        logging_steps=10,
    )

    assert evidence["loss_history"] == [
        {"optimizer_step": 10, "loss": 1.2},
        {"optimizer_step": 20, "loss": 0.8},
    ]
    assert evidence["aggregate_train_loss"] == pytest.approx(0.91)
    assert evidence["loss_step_source"] == "explicit_inference"
    assert evidence["loss_step_inference"]["logging_steps"] == 10


def test_text_loss_without_steps_or_trainer_config_fails_closed() -> None:
    with pytest.raises(ValueError, match="no validated trainer logging_steps"):
        parse_training_loss_evidence(
            "{'loss': 1.2}\n", training_step=420, logging_steps=None
        )


def test_learning_ui_is_replay_first_groups_frames_and_never_upscales() -> None:
    source = (
        Path(__file__).resolve().parents[2] / "src" / "npa" / "cli" / "agent_ui.html"
    ).read_text(encoding="utf-8")
    assert "Policy learning summary" in source
    assert "Offline held-out policy evaluation" in source
    assert "Open in Rerun" in source and "Open MCAP" in source
    assert "groupArtifactSequences" in source
    assert "native; preview is not enlarged" in source
    assert "width: auto" in source and "object-fit: contain" in source
    assert "Prepare leakage-free split" in source
    assert "Synchronized learning replay" in source
