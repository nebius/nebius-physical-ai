"""Hermetic contracts for the compositional GR00T N1.5 workflow stages."""

from __future__ import annotations

import builtins
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from npa.workbench.groot.custom_inputs import verify_minmax_roundtrip
from npa.workflows import groot_n15_offline as workflow


def _episodes(count: int) -> list[dict[str, int]]:
    return [{"episode_index": index, "length": index + 2} for index in range(count)]


def test_uniform_feedback_is_complete_episode_grouped_and_unlabeled() -> None:
    feedback = workflow._split_feedback(
        _episodes(5),
        mode="uniform",
        grouping="episode",
        split_strategy="contiguous-tail",
        train_episode_count=3,
        eval_episode_count=2,
    )

    assert [item["split"] for item in feedback] == [
        "train",
        "train",
        "train",
        "eval",
        "eval",
    ]
    assert [item["group"] for item in feedback] == [
        "episode-000000",
        "episode-000001",
        "episode-000002",
        "episode-000003",
        "episode-000004",
    ]
    assert all(set(item) == {"episode_index", "split", "group"} for item in feedback)


def test_uniform_feedback_refuses_incomplete_partition() -> None:
    with pytest.raises(ValueError, match="cover"):
        workflow._split_feedback(
            _episodes(5),
            mode="uniform",
            grouping="episode",
            split_strategy="contiguous-tail",
            train_episode_count=3,
            eval_episode_count=1,
        )


@pytest.mark.parametrize(
    "uri", ["", "https://example.invalid/a", "s3:///missing", "s3://bucket/a/../b"]
)
def test_s3_uri_refuses_noncanonical_or_unsafe_values(uri: str) -> None:
    with pytest.raises(ValueError):
        workflow.S3Uri.parse(uri)


def test_s3_uri_child_cannot_escape_prefix() -> None:
    root = workflow.S3Uri.parse("s3://bucket/run/output")
    assert root.child("receipt.json").as_uri() == "s3://bucket/run/output/receipt.json"
    with pytest.raises(ValueError, match="unsafe"):
        root.child("../receipt.json")


def test_custom_arguments_require_an_explicit_source_identity(tmp_path: Path) -> None:
    args = workflow._parser().parse_args(
        [
            "prepare",
            "--workspace",
            str(tmp_path),
            "--prepared-dataset-uri",
            "s3://bucket/prepared/dataset",
            "--manifest-uri",
            "s3://bucket/prepared/manifest.json",
            "--prepare-receipt-uri",
            "s3://bucket/prepared/provenance.json",
            "--prepare-commit-uri",
            "s3://bucket/prepared/commit.json",
            "--feedback-output-uri",
            "s3://bucket/prepared/feedback.json",
            "--checkpoint-root-uri",
            "s3://bucket/checkpoints",
            "--candidate-uri",
            "s3://bucket/candidate",
            "--training-receipt-uri",
            "s3://bucket/training.json",
            "--training-commit-uri",
            "s3://bucket/training-commit.json",
            "--actions-uri",
            "s3://bucket/actions.npz",
            "--video-uri",
            "s3://bucket/video.mp4",
            "--html-uri",
            "s3://bucket/proof.html",
            "--evaluation-uri",
            "s3://bucket/evaluation.json",
            "--media-validation-uri",
            "s3://bucket/media.json",
            "--gallery-uri",
            "s3://bucket/gallery",
            "--evidence-commit-uri",
            "s3://bucket/evidence-commit.json",
            "--verify-receipt-uri",
            "s3://bucket/readback.json",
            "--dataset-repository",
            "",
            "--dataset-revision",
            "",
            "--dataset-archive-uri",
            "s3://bucket/input.tar",
            "--input-provenance-uri",
            "s3://bucket/provenance.json",
            "--action-contract-uri",
            "s3://bucket/action.json",
            "--external-code-uri",
            "s3://bucket/code.tar",
            "--external-code-sha256",
            "a" * 64,
            "--external-code-package",
            "operator_extension",
            "--adapter-callable",
            "operator_extension.adapter:adapt",
            "--feedback-callable",
            "operator_extension.adapter:feedback",
            "--data-config",
            "operator_extension.config:Profile",
            "--gallery-reference-uri",
            "s3://bucket/gallery-reference",
        ]
    )

    with pytest.raises(ValueError, match="explicit dataset_repository"):
        workflow._validate_arguments(args)


def test_train_only_statistics_do_not_read_eval_parquet(tmp_path: Path) -> None:
    data = tmp_path / "data" / "chunk-000"
    data.mkdir(parents=True)
    for episode, value in ((0, 1.0), (1, 100.0)):
        pd.DataFrame(
            {
                "action": [np.asarray([value], dtype=np.float32)],
                "observation.state": [np.asarray([value + 1], dtype=np.float32)],
            }
        ).to_parquet(data / f"episode_{episode:06d}.parquet")

    stats = workflow._statistics_from_train_parquet(tmp_path, {0})

    assert stats["action"]["mean"] == [1.0]
    assert stats["observation.state"]["max"] == [2.0]


def test_custom_native_layout_requires_declared_raw_columns(tmp_path: Path) -> None:
    data = tmp_path / "data" / "chunk-000"
    data.mkdir(parents=True)
    pd.DataFrame(
        {
            "observation.state": [np.asarray([1.0, 2.0], dtype=np.float32)],
            "action": [np.asarray([3.0, 4.0], dtype=np.float32)],
        }
    ).to_parquet(data / "episode_000000.parquet")
    modality = {
        "state": {"arm": {"start": 0, "end": 2, "original_key": "observation.state"}},
        "action": {"arm": {"start": 0, "end": 2, "original_key": "action"}},
    }
    workflow._validate_custom_native_layout(
        tmp_path, modality, [{"episode_index": 0, "length": 1}]
    )

    pd.DataFrame({"action": [np.asarray([3.0, 4.0], dtype=np.float32)]}).to_parquet(
        data / "episode_000000.parquet"
    )
    with pytest.raises(workflow.WorkflowArtifactError, match="observation.state"):
        workflow._validate_custom_native_layout(
            tmp_path, modality, [{"episode_index": 0, "length": 1}]
        )


def test_custom_native_layout_rejects_string_language_annotations(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data" / "chunk-000"
    data.mkdir(parents=True)
    pd.DataFrame(
        {
            "observation.state": [np.asarray([1.0], dtype=np.float32)],
            "action": [np.asarray([2.0], dtype=np.float32)],
            "annotation.human.task_description": ["push cube"],
        }
    ).to_parquet(data / "episode_000000.parquet")
    modality = {
        "state": {"arm": {"start": 0, "end": 1, "original_key": "observation.state"}},
        "action": {"arm": {"start": 0, "end": 1, "original_key": "action"}},
        "annotation": {
            "human.task_description": {
                "original_key": "annotation.human.task_description"
            }
        },
    }

    with pytest.raises(workflow.WorkflowArtifactError, match="integer task indices"):
        workflow._validate_custom_native_layout(
            tmp_path, modality, [{"episode_index": 0, "length": 1}]
        )


def test_custom_action_roundtrip_records_terminal_mask_and_declared_keys(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data" / "chunk-000"
    data.mkdir(parents=True)
    pd.DataFrame(
        {
            "action.left_arm": [np.asarray([1.0], dtype=np.float32)] * 2,
            "action.right_arm": [np.asarray([-2.0], dtype=np.float32)] * 2,
        }
    ).to_parquet(data / "episode_000000.parquet")
    evidence = workflow._custom_action_roundtrip_evidence(
        tmp_path,
        [{"episode_index": 0, "length": 2}],
        {0},
        ("action.left_arm", "action.right_arm"),
        4,
        {
            "action.left_arm": {"min": [1.0], "max": [1.0]},
            "action.right_arm": {"min": [-3.0], "max": [3.0]},
        },
        verify_minmax_roundtrip,
    )
    assert evidence["action_keys"] == ["action.left_arm", "action.right_arm"]
    assert evidence["valid_action_horizon_steps"] == 1
    assert evidence["masked_action_horizon_steps"] == 3
    assert evidence["constant_training_dimensions"] == 1


def test_custom_action_roundtrip_resolves_declared_raw_action_slices(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data" / "chunk-000"
    data.mkdir(parents=True)
    pd.DataFrame(
        {"action": [np.asarray([1.0, -2.0], dtype=np.float32)] * 2}
    ).to_parquet(data / "episode_000000.parquet")
    evidence = workflow._custom_action_roundtrip_evidence(
        tmp_path,
        [{"episode_index": 0, "length": 2}],
        {0},
        ("action.left_arm", "action.right_arm"),
        4,
        {"action": {"min": [1.0, -3.0], "max": [1.0, 3.0]}},
        verify_minmax_roundtrip,
        {
            "left_arm": {"start": 0, "end": 1, "original_key": "action"},
            "right_arm": {"start": 1, "end": 2, "original_key": "action"},
        },
    )
    assert evidence["action_keys"] == ["action.left_arm", "action.right_arm"]
    assert evidence["valid_action_horizon_steps"] == 1


def test_metrics_apply_terminal_mask_without_claiming_outcomes() -> None:
    prediction = np.asarray([[[1.0], [5.0]], [[3.0], [100.0]]])
    target = np.asarray([[[0.0], [0.0]], [[1.0], [0.0]]])
    mask = np.asarray([[True, True], [True, False]])

    result = workflow._metrics(prediction, target, mask)

    assert result["valid_action_values"] == 3
    assert result["mae"] == pytest.approx((1.0 + 5.0 + 2.0) / 3)


def test_metrics_reject_nonfinite_native_action_predictions() -> None:
    with pytest.raises(workflow.WorkflowArtifactError, match="finite"):
        workflow._metrics(
            np.asarray([[[float("nan")]]]),
            np.asarray([[[0.0]]]),
            np.asarray([[True]]),
        )


def test_matched_ordinary_bc_manifest_changes_only_training_weights(
    tmp_path: Path,
) -> None:
    source = tmp_path / "feedback-manifest.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": "npa.groot-offline/v2",
                "dataset_sha256": "d" * 64,
                "feedback_sha256": "f" * 64,
                "mode": "hybrid",
                "action_horizon": 2,
                "temperature": 1.0,
                "episodes": [
                    {
                        "episode_index": 1,
                        "split": "train",
                        "length": 3,
                        "weights": [0.25, 1.0, 2.75],
                        "success": False,
                        "collection_kind": "policy_rollout",
                    },
                    {
                        "episode_index": 2,
                        "split": "eval",
                        "length": 2,
                        "weights": [1.0, 1.0],
                        "success": True,
                        "collection_kind": "operator_recovery",
                    },
                ],
            }
        )
    )

    control_path = tmp_path / "ordinary-bc-manifest.json"
    digest = workflow._ordinary_bc_manifest(source, control_path)
    control = json.loads(control_path.read_text())

    assert digest == workflow.sha256_file(control_path)
    assert control["mode"] == "uniform"
    assert control["dataset_sha256"] == "d" * 64
    assert [item["episode_index"] for item in control["episodes"]] == [1, 2]
    assert control["episodes"][0]["weights"] == [1.0, 1.0, 1.0]
    assert control["episodes"][1]["weights"] == [1.0, 1.0]


def test_inverse_transformed_actions_use_recorded_target_width_not_model_padding() -> (
    None
):
    """GR00T's padded action head must decode back to the SO-100's six axes."""

    decoded = workflow._decoded_action_array(
        {
            "action.single_arm": np.zeros((1, 16, 5), dtype=np.float32),
            "action.gripper": np.zeros((1, 16, 1), dtype=np.float32),
        },
        action_horizon=16,
        expected_action_dim=6,
    )

    assert decoded.shape == (16, 6)
    with pytest.raises(workflow.WorkflowArtifactError, match=r"expected \(1, 16, 32\)"):
        workflow._decoded_action_array(
            {
                "action.single_arm": np.zeros((1, 16, 5), dtype=np.float32),
                "action.gripper": np.zeros((1, 16, 1), dtype=np.float32),
            },
            action_horizon=16,
            expected_action_dim=32,
        )


def test_standalone_html_embeds_video_and_no_remote_dependency(tmp_path: Path) -> None:
    video = tmp_path / "proof.mp4"
    video.write_bytes(b"not-a-real-mp4-but-an-embedded-fixture")
    evaluation = {
        "source": {
            "dataset_repository": "public/dataset",
            "dataset_revision": "a" * 40,
            "model_repository": "nvidia/model",
            "model_revision": "b" * 40,
        },
        "held_out": {"frames": 3},
        "metrics": {"baseline": {"mae": 2.0}, "candidate": {"mae": 1.0}},
    }

    document = workflow._standalone_html(evaluation, video)

    assert "data:video/mp4;base64," in document
    assert "https://" not in document
    assert "not a policy rollout" in document


def test_gallery_reference_binds_media_hash_and_recorded_frame_range(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    media = tmp_path / "recovery.mp4"
    media.write_bytes(b"recorded-reference")
    payload = {
        "schema_version": "npa.groot-recorded-gallery-reference/v1",
        "items": [
            {
                "id": "recovery-0",
                "media_path": "recovery.mp4",
                "media_sha256": workflow.sha256_file(media),
                "source": {"revision": "public-immutable-revision"},
                "license": "CC-BY-4.0",
                "privacy_classification": "public",
                "source_episode_index": 0,
                "camera_key": "observation.images.wrist",
                "recorded_fps": 15,
                "recorded_frame_count": 10,
                "frame_start": 3,
                "frame_end": 7,
                "outcome": "unknown",
                "collection_kind": "operator_recovery",
                "tags": ["human_intervention"],
                "label_source": "upstream recorded intervention field",
                "action_predictions_available": False,
                "limitation": "recorded reference only",
            }
        ],
    }
    (tmp_path / "gallery.json").write_text(json.dumps(payload))
    monkeypatch.setattr(
        workflow,
        "_validate_video",
        lambda _path, count: {"ffprobe": {"avg_frame_rate": "15/1"}, "count": count},
    )

    result = workflow._validate_gallery_reference(tmp_path)

    assert result["item_count"] == 1
    assert result["items"][0]["collection_kind"] == "operator_recovery"
    payload["items"][0]["media_path"] = "../escape.mp4"
    (tmp_path / "gallery.json").write_text(json.dumps(payload))
    with pytest.raises(workflow.WorkflowArtifactError, match="unsafe"):
        workflow._validate_gallery_reference(tmp_path)


def test_workflow_argument_validation_requires_progress_input_and_safe_uris(
    tmp_path: Path,
) -> None:
    parser = workflow._parser()
    args = parser.parse_args(
        [
            "prepare",
            "--workspace",
            str(tmp_path),
            "--prepared-dataset-uri",
            "s3://bucket/prepared",
            "--manifest-uri",
            "s3://bucket/manifest.json",
            "--prepare-receipt-uri",
            "s3://bucket/prepare.json",
            "--prepare-commit-uri",
            "s3://bucket/prepare-commit.json",
            "--feedback-output-uri",
            "s3://bucket/feedback.json",
            "--groot-modality-path",
            str(tmp_path / "modality.json"),
            "--checkpoint-root-uri",
            "s3://bucket/checkpoints",
            "--candidate-uri",
            "s3://bucket/candidate",
            "--training-receipt-uri",
            "s3://bucket/training.json",
            "--training-commit-uri",
            "s3://bucket/training-commit.json",
            "--actions-uri",
            "s3://bucket/actions.npz",
            "--video-uri",
            "s3://bucket/video.mp4",
            "--html-uri",
            "s3://bucket/proof.html",
            "--evaluation-uri",
            "s3://bucket/evaluation.json",
            "--media-validation-uri",
            "s3://bucket/media.json",
            "--gallery-uri",
            "s3://bucket/gallery",
            "--evidence-commit-uri",
            "s3://bucket/evidence-commit.json",
            "--verify-receipt-uri",
            "s3://bucket/verify.json",
            "--feedback-mode",
            "progress",
        ]
    )

    with pytest.raises(ValueError, match="feedback_uri"):
        workflow._validate_arguments(args)


def test_custom_workflow_requires_a_profile_from_the_declared_extension(
    tmp_path: Path,
) -> None:
    parser = workflow._parser()
    common = [
        "prepare",
        "--workspace",
        str(tmp_path),
        "--prepared-dataset-uri",
        "s3://bucket/prepared",
        "--manifest-uri",
        "s3://bucket/manifest.json",
        "--prepare-receipt-uri",
        "s3://bucket/prepare.json",
        "--prepare-commit-uri",
        "s3://bucket/prepare-commit.json",
        "--feedback-output-uri",
        "s3://bucket/feedback.json",
        "--gallery-reference-uri",
        "s3://bucket/prepared/gallery-reference",
        "--checkpoint-root-uri",
        "s3://bucket/checkpoints",
        "--candidate-uri",
        "s3://bucket/candidate",
        "--training-receipt-uri",
        "s3://bucket/training.json",
        "--training-commit-uri",
        "s3://bucket/training-commit.json",
        "--actions-uri",
        "s3://bucket/actions.npz",
        "--video-uri",
        "s3://bucket/video.mp4",
        "--html-uri",
        "s3://bucket/proof.html",
        "--evaluation-uri",
        "s3://bucket/evaluation.json",
        "--media-validation-uri",
        "s3://bucket/media.json",
        "--gallery-uri",
        "s3://bucket/gallery",
        "--evidence-commit-uri",
        "s3://bucket/evidence-commit.json",
        "--verify-receipt-uri",
        "s3://bucket/verify.json",
        "--dataset-archive-uri",
        "s3://bucket/source.tar.gz",
        "--input-provenance-uri",
        "s3://bucket/provenance.json",
        "--action-contract-uri",
        "s3://bucket/action.json",
        "--external-code-uri",
        "s3://bucket/extension.tar.gz",
        "--external-code-sha256",
        "a" * 64,
        "--external-code-package",
        "operator_extension",
        "--adapter-callable",
        "operator_extension.adapter:adapt",
        "--feedback-callable",
        "operator_extension.adapter:feedback",
    ]
    mismatched = parser.parse_args(
        [*common, "--data-config", "other_extension.profile:Config"]
    )
    with pytest.raises(ValueError, match="data_config"):
        workflow._validate_arguments(mismatched)
    matching = parser.parse_args(
        [*common, "--data-config", "operator_extension.profile:Config"]
    )
    workflow._validate_arguments(matching)


def test_resume_identity_binds_all_model_and_training_inputs(tmp_path: Path) -> None:
    parser = workflow._parser()
    common = [
        "train",
        "--workspace",
        str(tmp_path),
        "--prepared-dataset-uri",
        "s3://bucket/prepared",
        "--manifest-uri",
        "s3://bucket/manifest.json",
        "--prepare-receipt-uri",
        "s3://bucket/prepare.json",
        "--prepare-commit-uri",
        "s3://bucket/prepare-commit.json",
        "--feedback-output-uri",
        "s3://bucket/feedback.json",
        "--groot-modality-path",
        str(tmp_path / "modality.json"),
        "--checkpoint-root-uri",
        "s3://bucket/checkpoints",
        "--candidate-uri",
        "s3://bucket/candidate",
        "--training-receipt-uri",
        "s3://bucket/training.json",
        "--training-commit-uri",
        "s3://bucket/training-commit.json",
        "--actions-uri",
        "s3://bucket/actions.npz",
        "--video-uri",
        "s3://bucket/video.mp4",
        "--html-uri",
        "s3://bucket/proof.html",
        "--evaluation-uri",
        "s3://bucket/evaluation.json",
        "--media-validation-uri",
        "s3://bucket/media.json",
        "--gallery-uri",
        "s3://bucket/gallery",
        "--evidence-commit-uri",
        "s3://bucket/evidence-commit.json",
        "--verify-receipt-uri",
        "s3://bucket/verify.json",
    ]
    args = parser.parse_args(common)
    baseline = workflow._resume_identity(
        args, manifest_sha="a" * 64, base_checkpoint_sha="b" * 64
    )
    changed = parser.parse_args([*common, "--batch-size", "2"])
    assert baseline != workflow._resume_identity(
        changed, manifest_sha="a" * 64, base_checkpoint_sha="b" * 64
    )


def test_native_child_uses_stable_module_name_when_parent_is_main(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A staged runner must not turn the native child into ``-m __main__``."""

    command: list[str] = []

    def fake_run(argv: list[str], **_kwargs: object) -> SimpleNamespace:
        command.extend(argv)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(workflow, "__name__", "__main__")
    monkeypatch.setattr(
        workflow, "_bootstrap_native_runtime", lambda _args: tmp_path / "python"
    )
    monkeypatch.setattr(workflow.subprocess, "run", fake_run)

    workflow._run_native(["prepare", "--workspace", str(tmp_path)], SimpleNamespace())

    assert command[:3] == [str(tmp_path / "python"), "-m", workflow.RUNTIME_MODULE]
    assert "__main__" not in command


def test_native_runtime_uses_the_image_owned_python_not_the_path_shim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def captured(command: list[str]) -> str:
        commands.append(command)
        return "3.10"

    monkeypatch.setattr(workflow, "_captured", captured)

    assert workflow._native_python() == "/usr/bin/python3"
    assert commands == [
        [
            "/usr/bin/python3",
            "-c",
            "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')",
        ]
    ]


@pytest.mark.parametrize("version", ["3.9", "3.12", "invalid"])
def test_native_runtime_rejects_an_incompatible_image_python(
    monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    monkeypatch.setattr(workflow, "_captured", lambda _command: version)

    with pytest.raises(workflow.WorkflowArtifactError, match="Python|invalid"):
        workflow._native_python()


def test_complete_prepare_commit_is_read_back_before_reuse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    modality = tmp_path / "modality.json"
    modality.write_text('{"modalities": []}\n')
    manifest_payload = {
        "mode": "uniform",
        "action_horizon": 16,
        "temperature": 1.0,
        "dataset_sha256": "dataset-digest",
        "feedback_sha256": "feedback-digest",
    }
    provenance_payload = {
        "source": {
            "repository": workflow.DATASET_REPOSITORY_DEFAULT,
            "revision": workflow.DATASET_REVISION_DEFAULT,
            "license": workflow.PUBLIC_DATA_LICENSE,
        },
        "dataset": {
            "grouping": "episode",
            "split": "contiguous-tail",
            "train_episodes": 64,
            "eval_episodes": 16,
            "train_frames": 12,
            "eval_frames": 3,
            "modality_sha256": workflow.sha256_file(modality),
            "derivative_sha256": "dataset-digest",
        },
        "feedback": {"mode": "uniform", "feedback_sha256": "feedback-digest"},
        "video_derivative": {
            "sample_derivative_probe": {"codec_name": "h264"},
            "source_video_count": 2,
            "derivative_video_count": 2,
        },
    }
    commit = {
        "schema_version": workflow.SCHEMA_VERSION,
        "kind": "prepare-commit",
        "dataset_tree_sha256": "d" * 64,
        "manifest_sha256": "m" * 64,
        "outputs": {
            "prepared_dataset": {"tree_sha256": "d" * 64},
            "feedback": {"sha256": "f" * 64},
            "manifest": {"sha256": "m" * 64},
            "provenance": {"sha256": "p" * 64},
        },
    }

    class Store:
        def exists(self, _uri: str) -> bool:
            return True

        def get_json(self, _uri: str) -> dict[str, object]:
            return commit

    payloads = {
        "s3://bucket/feedback.json": json.dumps({}),
        "s3://bucket/manifest.json": json.dumps(manifest_payload),
        "s3://bucket/provenance.json": json.dumps(provenance_payload),
    }

    def verify_object(
        _store: object, uri: str, digest: str, destination: Path
    ) -> dict[str, object]:
        destination.write_text(payloads[uri])
        return {"sha256": digest, "bytes": destination.stat().st_size}

    monkeypatch.setattr(workflow, "_verify_object", verify_object)
    monkeypatch.setattr(
        workflow,
        "_verify_directory",
        lambda *_args: {"tree_sha256": "d" * 64, "files": 7, "bytes": 99},
    )
    args = SimpleNamespace(
        prepare_commit_uri="s3://bucket/commit.json",
        prepared_dataset_uri="s3://bucket/prepared",
        feedback_output_uri="s3://bucket/feedback.json",
        manifest_uri="s3://bucket/manifest.json",
        prepare_receipt_uri="s3://bucket/provenance.json",
        dataset_repository=workflow.DATASET_REPOSITORY_DEFAULT,
        dataset_revision=workflow.DATASET_REVISION_DEFAULT,
        grouping="episode",
        split_strategy="contiguous-tail",
        train_episode_count=64,
        eval_episode_count=16,
        feedback_mode="uniform",
        action_horizon=16,
        feedback_temperature=1.0,
        derivative_codec="h264",
        groot_modality_path=modality,
    )

    stage = tmp_path / "stage"
    stage.mkdir()
    reused = workflow._reuse_prepared_inputs(args, Store(), stage)

    assert reused == {
        "status": "prepared",
        "reused_prepared_inputs": True,
        "manifest_sha256": "m" * 64,
        "dataset_tree_sha256": "d" * 64,
        "train_frames": 12,
        "eval_frames": 3,
        "readback_files": 7,
        "readback_bytes": (
            99
            + len(payloads["s3://bucket/feedback.json"])
            + len(payloads["s3://bucket/manifest.json"])
            + len(payloads["s3://bucket/provenance.json"])
        ),
    }


def test_reused_prepare_commit_rejects_a_different_dataset_revision(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        dataset_repository=workflow.DATASET_REPOSITORY_DEFAULT,
        dataset_revision=workflow.DATASET_REVISION_DEFAULT,
        grouping="episode",
        split_strategy="contiguous-tail",
        train_episode_count=64,
        eval_episode_count=16,
        feedback_mode="uniform",
        action_horizon=16,
        feedback_temperature=1.0,
        derivative_codec="h264",
        groot_modality_path=tmp_path / "missing-modality.json",
    )

    with pytest.raises(workflow.WorkflowArtifactError, match="dataset revision"):
        workflow._validate_reused_prepare_inputs(
            args,
            {
                "source": {
                    "repository": workflow.DATASET_REPOSITORY_DEFAULT,
                    "revision": "other",
                },
                "dataset": {},
                "feedback": {},
                "video_derivative": {},
            },
            {},
        )


def test_missing_external_prepared_inputs_fail_before_public_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Store:
        def exists(self, _uri: str) -> bool:
            return False

    monkeypatch.setattr(workflow, "S3Store", Store)
    original_import = builtins.__import__

    def reject_huggingface_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "huggingface_hub":
            raise AssertionError(
                "external prepared inputs must be rejected before download"
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_huggingface_import)
    args = SimpleNamespace(
        workspace=tmp_path / "workspace",
        prepare_commit_uri="s3://bucket/old/prepare-commit.json",
        prepared_dataset_uri="s3://bucket/old/prepared/dataset",
        candidate_uri="s3://bucket/new/candidate",
        prepared_input_mode="build-or-reuse",
    )

    with pytest.raises(
        workflow.WorkflowArtifactError, match="required prepared inputs"
    ):
        workflow._prepare(args)


def test_fresh_train_keeps_candidate_directory_new_for_native_trainer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A fresh native Trainer owns creation of its otherwise-new output path."""

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "npa.groot-offline/v1",
                "dataset_sha256": "d" * 64,
                "feedback_sha256": "f" * 64,
                "mode": "uniform",
                "action_horizon": 16,
                "temperature": 1.0,
                "episodes": [
                    {
                        "episode_index": 0,
                        "split": "train",
                        "length": 4,
                        "weights": [1.0] * 4,
                    },
                    {
                        "episode_index": 1,
                        "split": "eval",
                        "length": 2,
                        "weights": [1.0] * 2,
                    },
                ],
            }
        )
    )
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    captured: list[Path] = []

    class Store:
        def put_directory(self, _path: Path, _uri: str) -> dict[str, object]:
            return {"tree_sha256": "t" * 64}

        def put_file(self, path: Path, _uri: str) -> dict[str, object]:
            return {"sha256": workflow.sha256_file(path)}

        def put_json(self, _value: object, _uri: str) -> dict[str, object]:
            return {}

    def fake_download_base(_args: object, destination: Path) -> None:
        destination.mkdir()

    def fake_restore(
        _store: object,
        _root: str,
        output: Path,
        _manifest_sha: str,
        _identity: str,
    ) -> None:
        assert not output.exists()
        return None

    def fake_train_offline(**kwargs: object) -> dict[str, object]:
        output = kwargs["output_dir"]
        assert isinstance(output, Path)
        assert not output.exists()
        captured.append(output)
        output.mkdir()
        (output / "weights.safetensors").write_bytes(b"candidate")
        return {
            "dataset_sha256": "d" * 64,
            "base_checkpoint_sha256": "b" * 64,
            "candidate_checkpoint_sha256": "c" * 64,
            "optimizer_steps": 1,
            "held_out_eval_loss": 1.0,
            "optimizer_loss_trace": [{"step": 1, "loss": 1.0}],
            "training_config": {
                "batch_size": 4,
                "learning_rate": 1e-4,
                "seed": 42,
            },
        }

    monkeypatch.setattr(workflow, "S3Store", Store)
    monkeypatch.setattr(
        workflow,
        "_download_prepared_inputs",
        lambda _args, _stage: (
            dataset,
            manifest_path,
            {"dataset": {"derivative_sha256": "d" * 64}},
        ),
    )
    monkeypatch.setattr(workflow, "_download_base_model", fake_download_base)
    monkeypatch.setattr(workflow, "_restore_checkpoint", fake_restore)
    monkeypatch.setattr(
        workflow, "_resume_identity", lambda *_args, **_kwargs: "r" * 64
    )
    monkeypatch.setattr(workflow, "_gpu_runtime_evidence", lambda: {"gpu": "test"})
    monkeypatch.setattr(
        workflow, "_checkpoint_sync", lambda *_args: lambda *_args: None
    )
    monkeypatch.setattr(
        workflow,
        "_action_decoder_hashes",
        lambda path: {"tensor": "base" if path.name == "base-model" else "candidate"},
    )
    monkeypatch.setattr(
        "npa.workbench.groot.offline_feedback.file_digest", lambda _path: "m" * 64
    )
    monkeypatch.setattr(
        "npa.workbench.groot.offline_training.checkpoint_digest", lambda _path: "b" * 64
    )
    monkeypatch.setattr(
        "npa.workbench.groot.offline_training.train_offline", fake_train_offline
    )

    args = SimpleNamespace(
        workspace=tmp_path / "workspace",
        batch_size=4,
        complete_train_passes=1,
        checkpoint_root_uri="s3://bucket/checkpoints",
        candidate_uri="s3://bucket/candidate",
        control_uri="s3://bucket/ordinary-bc-control",
        training_receipt_uri="s3://bucket/training.json",
        training_commit_uri="s3://bucket/training-commit.json",
        groot_revision=workflow.GROOT_REVISION_DEFAULT,
        groot_repository=workflow.GROOT_REPOSITORY_DEFAULT,
        model_repository=workflow.MODEL_REPOSITORY_DEFAULT,
        model_revision=workflow.MODEL_REVISION_DEFAULT,
        data_config="so100_dualcam",
        embodiment_tag="new_embodiment",
        training_seed=42,
        video_backend="decord",
        checkpoint_interval_steps=1,
        action_horizon=16,
        learning_rate=1e-4,
    )

    result = workflow._train(args)

    assert result["status"] == "trained"
    assert {path.name for path in captured} == {"candidate", "ordinary-bc-control"}
    assert all(path.is_dir() for path in captured)


def test_checked_in_workflow_dispatches_every_compositional_stage() -> None:
    from npa.orchestration.npa_workflow.interpreter import build_plan
    from npa.orchestration.npa_workflow.spec import load_spec
    from npa.orchestration.npa_workflow.submit import merge_config_overrides

    root = Path(__file__).resolve().parents[3]
    spec = merge_config_overrides(
        load_spec(root / "workflows/testing/groot-n15-offline-rl.yaml"),
        {"bucket": "example-bucket"},
    )
    plan = build_plan(spec, run_id="groot-n15-test-plan")

    assert [step.state for step in plan.steps] == [
        "prepare-custom-lerobot",
        "train-native-n15",
        "evaluate-recorded-observations",
        "verify-durable-publication",
    ]
    assert [step.argv[3] for step in plan.steps] == [
        "prepare",
        "train",
        "evaluate",
        "verify",
    ]
    for step in plan.steps:
        assert "--standard-input-config-uri" in step.argv
        assert "--dataset-archive-uri" in step.argv
        assert "--external-code-uri" in step.argv
        assert "--feedback-callable" in step.argv
        assert "--action-contract-uri" in step.argv
        assert "--control-uri" in step.argv
        assert "--evidence-video-key" in step.argv


def test_standard_reuse_rejects_a_changed_declaration() -> None:
    args = SimpleNamespace(
        input_mode=workflow.STANDARD_INPUT_MODE,
        _standard_input_config_sha256="a" * 64,
        dataset_repository="operator/recorded-rollouts",
        dataset_revision="2026-11-10",
        feedback_mode="hybrid",
        action_horizon=16,
    )

    with pytest.raises(
        workflow.WorkflowArtifactError, match="standard input declaration"
    ):
        workflow._validate_reused_prepare_inputs(
            args,
            {
                "kind": "standard-lerobot-prepare",
                "standard_input_config_sha256": "b" * 64,
                "source": {},
                "feedback": {},
            },
            {},
        )


def test_standard_declaration_resolves_model_before_native_bootstrap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = Path(__file__).resolve().parents[3]
    declaration = root / "workflows/testing/groot-n15-standard-input.example.json"

    class Store:
        def get_file(self, uri: str, destination: Path) -> None:
            assert uri == "s3://operator-inputs/standard.json"
            shutil.copyfile(declaration, destination)

    monkeypatch.setattr(workflow, "S3Store", Store)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    args = SimpleNamespace(
        workspace=workspace,
        native=False,
        input_mode=workflow.STANDARD_INPUT_MODE,
        standard_input_config_uri="s3://operator-inputs/standard.json",
    )

    workflow._resolve_standard_runtime_args(args)

    assert args.model_repository == workflow.MODEL_REPOSITORY_DEFAULT
    assert args.groot_revision == workflow.GROOT_REVISION_DEFAULT
    assert (
        args.data_config
        == "npa.workbench.groot.standard_inputs:StandardSingleWristConfig"
    )
    assert args.action_target_keys == "action.robot"


def test_standard_prepare_materializes_declared_tree_without_external_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The no-code adapter binds source/feedback bytes and native mappings."""

    source = tmp_path / "source"
    (source / "meta").mkdir(parents=True)
    (source / "data/chunk-000").mkdir(parents=True)
    video = source / "videos/chunk-000/observation.images.wrist"
    video.mkdir(parents=True)
    episodes = []
    feedback_episodes = []
    for index in range(4):
        length = 2
        pd.DataFrame(
            {
                "observation.state": [np.asarray([index, index + 1], dtype=np.float32)]
                * length,
                "action": [np.asarray([index + 2, index + 3], dtype=np.float32)]
                * length,
                "task_index": [0] * length,
                "episode_index": [index] * length,
                "frame_index": list(range(length)),
            }
        ).to_parquet(source / "data/chunk-000" / f"episode_{index:06d}.parquet")
        (video / f"episode_{index:06d}.mp4").write_bytes(b"recorded-video")
        episodes.append({"episode_index": index, "length": length})
        feedback_episodes.append(
            {
                "source_id": index,
                "partition": "eval" if index == 3 else "train",
                "observed_outcome": bool(index % 2),
                "session": f"collection-{index}",
                "collection": "policy_rollout",
                "source_tags": ["recorded"],
                "measured_completion": [0.0, 1.0],
            }
        )
    (source / "meta/info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.1",
                "chunks_size": 10,
                "fps": 15,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "features": {
                    "observation.images.wrist": {"dtype": "video"},
                    "observation.state": {"dtype": "float32", "shape": [2]},
                    "action": {"dtype": "float32", "shape": [2]},
                },
            }
        )
    )
    (source / "meta/episodes.jsonl").write_text(
        "\n".join(json.dumps(record) for record in episodes) + "\n"
    )
    (source / "meta/tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "recorded task"}) + "\n"
    )
    feedback = {
        "schema_version": "operator-label-sidecar/v1",
        "episodes": feedback_episodes,
    }
    feedback_path = tmp_path / "feedback.json"
    feedback_path.write_text(json.dumps(feedback))

    from npa.workbench.groot.standard_inputs import StandardLeRobotConfig

    config = StandardLeRobotConfig.model_validate(
        {
            "dataset": {
                "uri": "s3://input/dataset",
                "tree_sha256": workflow.tree_digest(
                    workflow.directory_inventory(source)
                ),
                "source_kind": "private-object-store",
                "source_identifier": "operator/recorded-rollouts",
                "source_revision": "rev-1",
                "license": "operator-declared-private",
                "privacy_classification": "private",
                "redistributable_media": False,
                "label_provenance": "operator-observed outcomes",
            },
            "feedback": {
                "uri": "s3://input/feedback.json",
                "sha256": workflow.sha256_file(feedback_path),
                "mode": "hybrid",
                "label_protocol": "operator observed outcome",
                "episode_index_field": "source_id",
                "split_field": "partition",
                "group_field": "session",
                "outcome_field": "observed_outcome",
                "collection_kind_field": "collection",
                "tags_field": "source_tags",
                "progress_field": "measured_completion",
                "progress_source": "measured",
                "progress_contract": {
                    "units": "normalized measured progress",
                    "direction": "higher is further along the recorded task",
                    "alignment": "one value per recorded frame",
                },
            },
            "mapping": {
                "camera_key": "observation.images.wrist",
                "state_key": "observation.state",
                "action_key": "action",
                "instruction_key": "task_index",
                "state_dimension": 2,
                "action_dimension": 2,
            },
            "action": {
                "kind": "cartesian_relative",
                "frame": "tool",
                "translation_unit": "millimeter",
                "rotation_representation": "axis_angle",
                "rotation_unit": "radian",
                "gripper_fields": ["gripper"],
                "transform_identity": "operator-calibration",
            },
            "split": {
                "train_episode_count": 3,
                "eval_episode_count": 1,
                "grouping": "collection",
                "strategy": "adapter",
            },
            "evidence": {
                "episode_index": 3,
                "start_frame": 0,
                "frame_count": 2,
                "fps": 15,
            },
            "model": {
                "model_repository": workflow.MODEL_REPOSITORY_DEFAULT,
                "model_revision": workflow.MODEL_REVISION_DEFAULT,
                "groot_repository": workflow.GROOT_REPOSITORY_DEFAULT,
                "groot_revision": workflow.GROOT_REVISION_DEFAULT,
            },
        }
    )

    class Store:
        def get_directory(self, uri: str, destination: Path) -> list[Path]:
            assert uri == "s3://input/dataset"
            shutil.copytree(source, destination)
            return [path for path in destination.rglob("*") if path.is_file()]

        def get_file(self, uri: str, destination: Path) -> None:
            assert uri == "s3://input/feedback.json"
            shutil.copyfile(feedback_path, destination)

        def put_directory(self, root: Path, _uri: str) -> dict[str, object]:
            return {
                "tree_sha256": workflow.tree_digest(workflow.directory_inventory(root)),
                "files": [],
            }

        def put_file(self, path: Path, _uri: str) -> dict[str, object]:
            return {"sha256": workflow.sha256_file(path), "bytes": path.stat().st_size}

        def put_json(self, value: object, _uri: str) -> dict[str, object]:
            payload = json.dumps(value, sort_keys=True).encode()
            return {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }

    def transcode(
        source_root: Path, derivative: Path, *_args: object
    ) -> dict[str, object]:
        shutil.copytree(source_root, derivative)
        return {
            "source_video_count": 4,
            "derivative_video_count": 4,
            "sample_derivative_probe": {"codec_name": "h264"},
        }

    def gallery(
        _dataset: Path, _config: object, _feedback: object, root: Path
    ) -> dict[str, object]:
        root.mkdir()
        (root / "gallery.json").write_text(
            json.dumps(
                {
                    "schema_version": "npa.groot-recorded-gallery-reference/v1",
                    "items": [],
                }
            )
        )
        return {
            "schema_version": "npa.groot-recorded-gallery-reference/v1",
            "item_count": 0,
            "items": [],
        }

    monkeypatch.setattr(workflow, "_transcode_h264", transcode)
    monkeypatch.setattr(workflow, "_standard_gallery_reference", gallery)
    args = SimpleNamespace(
        _standard_input_config=config,
        _standard_input_config_sha256="c" * 64,
        codec_workers=1,
        derivative_codec="h264",
        action_horizon=2,
        feedback_mode="hybrid",
        feedback_temperature=1.0,
        train_episode_count=3,
        eval_episode_count=1,
        prepared_dataset_uri="s3://output/prepared/dataset",
        feedback_output_uri="s3://output/prepared/feedback.json",
        manifest_uri="s3://output/prepared/manifest.json",
        gallery_reference_uri="s3://output/prepared/gallery",
        prepare_receipt_uri="s3://output/prepared/provenance.json",
        prepare_commit_uri="s3://output/prepared/commit.json",
    )

    result = workflow._prepare_standard_inputs(args, Store(), tmp_path / "stage")

    assert result["status"] == "prepared"
    assert result["feedback_label_counts"] == {
        "success": 2,
        "failure": 2,
        "unknown": 0,
        "operator_recovery": 0,
    }
    assert result["standard_input_config_sha256"] == "c" * 64


def test_readiness_record_is_bound_to_the_checked_in_workflow() -> None:
    root = Path(__file__).resolve().parents[3]
    workflow_bytes = (root / "workflows/testing/groot-n15-offline-rl.yaml").read_bytes()
    readiness = json.loads(
        (root / "workflows/testing/groot-n15-offline-rl.readiness.json").read_text()
    )

    assert readiness["schema_version"] == "workflow-readiness/v1"
    assert readiness["workflow_sha256"] == hashlib.sha256(workflow_bytes).hexdigest()
