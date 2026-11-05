"""Hermetic contracts for the compositional GR00T N1.5 workflow stages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

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

    assert [item["split"] for item in feedback] == ["train", "train", "train", "eval", "eval"]
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


@pytest.mark.parametrize("uri", ["", "https://example.invalid/a", "s3:///missing", "s3://bucket/a/../b"])
def test_s3_uri_refuses_noncanonical_or_unsafe_values(uri: str) -> None:
    with pytest.raises(ValueError):
        workflow.S3Uri.parse(uri)


def test_s3_uri_child_cannot_escape_prefix() -> None:
    root = workflow.S3Uri.parse("s3://bucket/run/output")
    assert root.child("receipt.json").as_uri() == "s3://bucket/run/output/receipt.json"
    with pytest.raises(ValueError, match="unsafe"):
        root.child("../receipt.json")


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


def test_workflow_argument_validation_requires_progress_input_and_safe_uris(tmp_path: Path) -> None:
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
    monkeypatch.setattr(workflow, "_bootstrap_native_runtime", lambda _args: tmp_path / "python")
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

    def verify_object(_store: object, uri: str, digest: str, destination: Path) -> dict[str, object]:
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
                "source": {"repository": workflow.DATASET_REPOSITORY_DEFAULT, "revision": "other"},
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
    args = SimpleNamespace(
        workspace=tmp_path / "workspace",
        prepare_commit_uri="s3://bucket/old/prepare-commit.json",
        prepared_dataset_uri="s3://bucket/old/prepared/dataset",
        candidate_uri="s3://bucket/new/candidate",
        prepared_input_mode="build-or-reuse",
    )

    with pytest.raises(workflow.WorkflowArtifactError, match="required prepared inputs"):
        workflow._prepare(args)


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
        "prepare-public-lerobot",
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


def test_readiness_record_is_bound_to_the_checked_in_workflow() -> None:
    root = Path(__file__).resolve().parents[3]
    workflow_bytes = (root / "workflows/testing/groot-n15-offline-rl.yaml").read_bytes()
    readiness = json.loads(
        (root / "workflows/testing/groot-n15-offline-rl.readiness.json").read_text()
    )

    assert readiness["schema_version"] == "workflow-readiness/v1"
    assert readiness["workflow_sha256"] == hashlib.sha256(workflow_bytes).hexdigest()
