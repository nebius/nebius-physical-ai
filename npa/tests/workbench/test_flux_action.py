"""Exercise robot contracts, native command boundaries, and artifact completion rules."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from npa.workbench.flux_action.artifacts import sha256, verify_checkpoint, verify_export
from npa.workbench.flux_action.configuration import (
    index_arguments,
    training_config,
    validate_dataset,
)
from npa.workbench.flux_action.schemas import FinetuneRequest, Recipe, SOURCE_REVISION
from npa.workbench.flux_action import runner


def recipe_data(width=7):
    return {
        "robot": {
            "embodiment": "custom_arm",
            "fps": 20,
            "camera_layout": "single",
            "cameras": {"front": "observation.images.front"},
            "actions": [{"name": f"joint_{i}", "unit": "rad"} for i in range(width)],
            "states": [{"name": f"joint_{i}", "unit": "rad"} for i in range(width)],
        },
        "val_episodes": 0,
        "training": {
            "steps": 4,
            "checkpoint_every": 4,
            "frozen_steps": 0,
            "trunk_warmup_steps": 1,
            "heads_warmup_steps": 1,
        },
    }


def dataset_metadata(root, recipe):
    robot = recipe.robot
    info = {
        "codebase_version": "v3.0",
        "fps": robot.fps,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            robot.action_key: {
                "shape": [len(robot.actions)],
                "names": [v.name for v in robot.actions],
            },
            robot.state_key: {
                "shape": [len(robot.states)],
                "names": [v.name for v in robot.states],
            },
            "observation.images.front": {"dtype": "video", "shape": [64, 64, 3]},
        },
    }
    path = root / "meta/info.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info))
    return path


@pytest.mark.parametrize("width", [3, 7, 14])
def test_custom_robot_has_explicit_controls_and_new_head(tmp_path, width):
    recipe = Recipe.model_validate(recipe_data(width))
    dataset_metadata(tmp_path, recipe)
    validate_dataset(tmp_path, recipe)
    config = training_config(recipe, tmp_path, tmp_path / "out", 8)
    assert config["policy"]["action_dim"] == width
    assert config["policy"]["action_modality"] == "robot_custom_arm"
    assert config["policy"]["gripper_flip_dims"] == []
    assert config["policy"]["action_parameterization"] == "absolute"
    assert config["policy"]["camera_keys"] == ["images.front"]
    assert config["dataset"] == "index"
    assert config["resume"] == "none"
    assert config["keep_checkpoints"] is None
    assert (
        "@eb267865d35e49e4066bde4936237f8d9f15a68c"
        in config["policy"]["text_encoder_id"]
    )


@pytest.mark.parametrize(
    "change,match",
    [
        ({"states": [{"name": "x", "unit": "m"}]}, "equal state"),
        ({"fps": 29.97}, "integer"),
        ({"canvas_hw": [500, 512]}, "multiples"),
        ({"camera_layout": "side_by_side"}, "camera count"),
        ({"cameras": {"front": "../escape"}}, "real video"),
        ({"gripper_flip_dims": [7]}, "indices"),
        ({"absolute_action_dims": [6]}, "only to joint_delta"),
    ],
)
def test_incompatible_robot_is_rejected(change, match):
    data = recipe_data()
    data["robot"].update(change)
    with pytest.raises(ValueError, match=match):
        Recipe.model_validate(data)


def test_joint_delta_indices_are_explicit(tmp_path):
    data = recipe_data()
    data["robot"].update(
        action_parameterization="joint_delta", absolute_action_dims=[6]
    )
    recipe = Recipe.model_validate(data)
    args = index_arguments(recipe, tmp_path, tmp_path / "index")
    assert args[args.index("--absolute-action-dims") + 1] == "6"
    assert args[args.index("--val-episodes") + 1] == "0"


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("fps", 30, "fps"),
        ("data_path", "../../outside.parquet", "unsafe"),
        ("video_path", "/absolute/{video_key}.mp4", "unsafe"),
    ],
)
def test_metadata_must_match_robot(tmp_path, field, value, match):
    recipe = Recipe.model_validate(recipe_data())
    path = dataset_metadata(tmp_path, recipe)
    info = json.loads(path.read_text())
    info[field] = value
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match=match):
        validate_dataset(tmp_path, recipe)


def test_swapped_channel_order_is_rejected(tmp_path):
    recipe = Recipe.model_validate(recipe_data())
    path = dataset_metadata(tmp_path, recipe)
    info = json.loads(path.read_text())
    info["features"]["action"]["names"].reverse()
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="channel order"):
        validate_dataset(tmp_path, recipe)


def completed_artifacts(output, recipe):
    checkpoint = output / "train" / f"step-{recipe.training.steps}"
    for name in ("COMPLETE", "model/.metadata", "optimizer/.metadata"):
        path = checkpoint / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    (checkpoint / "state.json").write_text(json.dumps({"step": recipe.training.steps}))
    (checkpoint / "config.json").write_text("{}")
    (output / "train/metrics.jsonl").write_text(
        json.dumps({"step": 4, "loss": 0.1, "lr_trunk": 0.00001, "lr_heads": 0.00005})
        + "\n"
    )
    return checkpoint


def export_artifacts(output):
    export = output / "export"
    export.mkdir()
    (export / "config.json").write_text(
        json.dumps(
            {
                "torch_dtype": "bfloat16",
                "video_vae_id": "pinned-vae",
                "text_encoder_id": "pinned-text",
            }
        )
    )
    (export / "model.safetensors").write_bytes(b"not-real-weights-test-fixture")
    (export / "manifest.json").write_text(
        json.dumps(
            {
                "kind": "policy_export",
                "format_version": 1,
                "weight_profile": "ema_0p10",
                "sha256": {
                    name: sha256(export / name)
                    for name in ("config.json", "model.safetensors")
                },
            }
        )
    )


def test_missing_final_checkpoint_and_nonfinite_metrics_fail(tmp_path):
    recipe = Recipe.model_validate(recipe_data())
    with pytest.raises(ValueError, match="checkpoint"):
        verify_checkpoint(tmp_path, recipe)
    completed_artifacts(tmp_path, recipe)
    verify_checkpoint(tmp_path, recipe)
    (tmp_path / "train/metrics.jsonl").write_text(
        '{"step":4,"loss":NaN,"lr_trunk":1,"lr_heads":1}\n'
    )
    with pytest.raises(ValueError, match="nonfinite"):
        verify_checkpoint(tmp_path, recipe)


def test_export_hashes_detect_corrupted_weights(tmp_path):
    export_artifacts(tmp_path)
    verify_export(tmp_path / "export")
    (tmp_path / "export/model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_export(tmp_path / "export")


def test_request_rejects_local_and_overlapping_handoffs():
    base = {
        "input_path": "s3://test-bucket/data",
        "recipe_uri": "s3://test-bucket/recipe.json",
        "output_path": "s3://test-bucket/run",
    }
    for bad in (
        {"input_path": "/tmp/data"},
        {"output_path": "s3://test-bucket/data/run"},
        {"output_path": "s3://test-bucket/run?token=secret"},
    ):
        with pytest.raises(ValueError):
            FinetuneRequest(**(base | bad))


class MemoryStorage:
    def __init__(self, root):
        self.root = root
        self.published = []
        self.claimed = False
        self.occupied = False

    def download_file(self, uri, destination):
        src = self.root / (
            "recipe.json" if uri.endswith("recipe.json") else "dataset/meta/info.json"
        )
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, destination)

    def download_directory(self, uri, destination):
        shutil.copytree(self.root / "dataset", destination, dirs_exist_ok=True)

    def require_empty_prefix(self, uri):
        if self.occupied:
            raise ValueError("output prefix must be empty")

    def put_bytes_conditional(self, *args, **kwargs):
        if self.claimed:
            raise ValueError("output is already claimed")
        self.claimed = True

    def upload_directory(self, directory, uri):
        self.published.extend(p.name for p in Path(directory).rglob("*") if p.is_file())

    def upload_file(self, path, uri):
        self.published.append(Path(path).name)


def execution_setup(tmp_path, monkeypatch, fail_stage=None):
    recipe = Recipe.model_validate(recipe_data())
    (tmp_path / "recipe.json").write_text(json.dumps(recipe_data()))
    dataset_metadata(tmp_path / "dataset", recipe)
    storage = MemoryStorage(tmp_path)
    monkeypatch.setattr(
        runner.StorageClient, "from_environment", lambda **kwargs: storage
    )
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "SOURCE_REVISION").write_text(SOURCE_REVISION)
    monkeypatch.setenv("NPA_FLUX_ACTION_ROOT", str(runtime))
    monkeypatch.setattr(runner, "verify_remote_export", lambda *args: None)
    calls = []

    def native_command(argv, **kwargs):
        calls.append(argv)
        stage = Path(kwargs["stdout"].name).stem
        output = Path(kwargs["stdout"].name).parent
        if stage == fail_stage:
            raise subprocess.CalledProcessError(1, argv)
        if stage == "index":
            (output / "index").mkdir()
            for name in ("manifest.json", "statistics.json"):
                (output / "index" / name).write_text("{}")
        if stage == "train":
            completed_artifacts(output, recipe)
        if stage == "export":
            export_artifacts(output)
        if stage == "reload":
            (output / "reload.json").write_text(
                json.dumps(
                    {
                        "n_windows": 1,
                        "split": "train",
                        "action_mse_raw_per_channel": [0.1] * 7,
                        "action_mse_normalized": 0.1,
                        "action_mse_raw": 0.1,
                    }
                )
            )

    monkeypatch.setattr(runner.subprocess, "run", native_command)
    request = FinetuneRequest(
        input_path="s3://test-bucket/data",
        recipe_uri="s3://test-bucket/recipe.json",
        output_path="s3://test-bucket/run",
        processes=2,
    )
    return storage, calls, request


def test_native_training_receipt_publishes_last(tmp_path, monkeypatch):
    storage, calls, request = execution_setup(tmp_path, monkeypatch)
    result = runner.finetune(request)
    assert result["status"] == "completed"
    assert result["closed_loop_evaluated"] is False
    assert storage.published[-1] == "result.json"
    train = next(call for call in calls if "torch.distributed.run" in call)
    assert train[train.index("--nproc_per_node") + 1] == "2"
    assert "--standalone" in train
    with pytest.raises(runner.FluxActionError, match="already claimed"):
        runner.finetune(request)


@pytest.mark.parametrize("stage", ["index", "weights", "train", "export", "reload"])
def test_failed_native_stage_preserves_logs_without_success(
    tmp_path, monkeypatch, stage
):
    storage, calls, request = execution_setup(tmp_path, monkeypatch, fail_stage=stage)
    with pytest.raises(runner.FluxActionError, match="retained"):
        runner.finetune(request)
    assert "failure.json" in storage.published
    assert "result.json" not in storage.published
    assert f"{stage}.log" in storage.published


def test_invalid_reload_prevents_success_receipt(tmp_path, monkeypatch):
    storage, calls, request = execution_setup(tmp_path, monkeypatch)

    def invalid_reload(path, recipe):
        raise ValueError("nonfinite reload metrics")

    monkeypatch.setattr(runner, "verify_reload", invalid_reload)
    with pytest.raises(runner.FluxActionError, match="nonfinite"):
        runner.finetune(request)
    assert "failure.json" in storage.published
    assert "result.json" not in storage.published


def test_dry_run_does_not_claim_or_launch(tmp_path, monkeypatch):
    storage, calls, request = execution_setup(tmp_path, monkeypatch)
    result = runner.finetune(request, dry_run=True)
    assert result["status"] == "planned"
    assert not storage.claimed
    assert not calls


def test_preexisting_output_without_claim_is_rejected(tmp_path, monkeypatch):
    storage, calls, request = execution_setup(tmp_path, monkeypatch)
    storage.occupied = True
    with pytest.raises(runner.FluxActionError, match="must be empty"):
        runner.finetune(request)
    assert not storage.claimed
    assert not storage.published
    assert not calls


def test_full_training_requires_positive_trunk_and_head_learning_rates(tmp_path):
    recipe = Recipe.model_validate(recipe_data())
    completed_artifacts(tmp_path, recipe)
    metrics = tmp_path / "train/metrics.jsonl"
    metrics.write_text('{"step":4,"loss":0.1,"lr_trunk":0,"lr_heads":0.001}\n')
    with pytest.raises(ValueError, match="trunk and head"):
        verify_checkpoint(tmp_path, recipe)


@pytest.mark.parametrize("width,value", [(6, 0.1), (7, float("nan"))])
def test_reload_rejects_wrong_embodiment_or_nonfinite_predictions(
    tmp_path, width, value
):
    from npa.workbench.flux_action.validation import verify_reload

    report = {
        "n_windows": 1,
        "split": "val",
        "action_mse_raw_per_channel": [value] * width,
        "action_mse_normalized": 0.1,
        "action_mse_raw": 0.1,
    }
    p = tmp_path / "reload.json"
    p.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        verify_reload(p, Recipe.model_validate(recipe_data()))


def test_remote_export_hash_detects_upload_corruption():
    import hashlib
    from types import SimpleNamespace
    from npa.workbench.flux_action.validation import verify_remote_export

    class Stream:
        def iter_chunks(self, **kwargs):
            return iter([b"changed bytes"])

    storage = SimpleNamespace(
        s3=SimpleNamespace(get_object=lambda **kwargs: {"Body": Stream()})
    )
    with pytest.raises(ValueError, match="published export SHA-256 mismatch"):
        verify_remote_export(
            storage,
            "s3://test-bucket/run",
            {"model.safetensors": hashlib.sha256(b"original bytes").hexdigest()},
        )


def test_single_gpu_recipe_preserves_full_training_and_native_precision(tmp_path):
    path = (
        Path(__file__).parents[2]
        / "workflows/workbench/configs/flux-action-aloha-single-gpu-smoke.json"
    )
    recipe = Recipe.model_validate_json(path.read_text())
    config = training_config(recipe, tmp_path, tmp_path / "out", 1)

    assert config["param_dtype"] == "bfloat16"
    assert config["compute_dtype"] == "bfloat16"
    assert config["ema_sigma_rels"] == ()
    assert config["frozen_steps"] == 0
    assert config["steps"] == 4
    assert config["policy"]["optimizer_lr"] > 0
    assert config["policy"]["optimizer_lr_heads_multiplier"] > 0
    assert recipe.export_profile == "model"


def test_default_training_keeps_fp32_and_both_emas(tmp_path):
    recipe = Recipe.model_validate(recipe_data())
    config = training_config(recipe, tmp_path, tmp_path / "out", 8)
    assert config["param_dtype"] == "float32"
    assert config["ema_sigma_rels"] == (0.1, 0.05)


@pytest.mark.parametrize(
    "training, profile, message",
    (
        ({"param_dtype": "float16"}, "model", "param_dtype"),
        ({"ema_sigma_rels": []}, "ema_0p10", "requires its EMA"),
        ({"ema_sigma_rels": [0.1]}, "ema_0p05", "requires its EMA"),
        ({"ema_sigma_rels": [0.1, 0.1]}, "model", "duplicates"),
        ({"ema_sigma_rels": [0.3]}, "model", "ema_sigma_rels"),
    ),
)
def test_recipe_rejects_unsupported_precision_or_unavailable_export(
    training, profile, message
):
    data = recipe_data()
    data["training"].update(training)
    data["export_profile"] = profile
    with pytest.raises(ValueError, match=message):
        Recipe.model_validate(data)


@pytest.mark.parametrize(
    "inference",
    [
        {"sampler": "invalid"},
        {"num_inference_steps": 0},
        {"sampler_shift": 0},
        {"guidance_scale": float("nan")},
    ],
)
def test_unusable_inference_settings_are_rejected(inference):
    data = recipe_data()
    data["inference"] = inference
    with pytest.raises(ValueError):
        Recipe.model_validate(data)


def test_export_owns_operator_inference_settings(tmp_path):
    data = recipe_data()
    data["inference"] = {
        "sampler": "cosmos_unipc",
        "num_inference_steps": 8,
        "sampler_shift": 5,
        "guidance_scale": 4,
        "guidance_scale_action": 1,
        "inference_seed": 17,
    }
    recipe = Recipe.model_validate(data)
    policy = training_config(recipe, tmp_path, tmp_path / "output", 1)["policy"]
    for key, value in data["inference"].items():
        assert policy[key] == value
