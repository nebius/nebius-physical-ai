"""Exercise content binding, feedback alignment, and offline split isolation."""

import hashlib
import json
import math

import pyarrow as pa
import pyarrow.parquet as parquet
import pytest

from npa.workbench.groot.offline_feedback import (
    OfflineManifest,
    dataset_digest,
    prepare_feedback,
)


@pytest.fixture
def feedback_inputs(tmp_path):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "videos").mkdir()
    (root / "videos/fingerprint-only.bin").write_bytes(b"synthetic-observation")
    info = {
        "codebase_version": "v2.1",
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    episodes = [{"episode_index": i, "length": 3} for i in range(4)]
    (root / "meta/episodes.jsonl").write_text("\n".join(map(json.dumps, episodes)))
    for episode in episodes:
        identifier = episode["episode_index"]
        parquet.write_table(
            pa.table({"episode_index": [identifier] * 3, "frame_index": [0, 1, 2]}),
            root / f"data/chunk-000/episode_{identifier:06d}.parquet",
        )
    feedback = [
        {
            "episode_index": i,
            "split": "train" if i < 2 else "eval",
            "success": i % 2 == 0,
            "group": f"collection-{i}",
        }
        for i in range(4)
    ]
    return root, feedback, tmp_path


def _prepare(inputs, *, feedback=None, **options):
    root, original, directory = inputs
    path = directory / "feedback.json"
    path.write_text(json.dumps(original if feedback is None else feedback))
    return prepare_feedback(
        root,
        path,
        directory / "weights.json",
        action_horizon=options.pop("action_horizon", 1),
        mode=options.pop("mode", "outcome"),
        **options,
    )


def test_binary_returns_are_train_normalized_and_source_is_immutable(feedback_inputs):
    root, _, _ = feedback_inputs
    before = dataset_digest(root)
    manifest = _prepare(feedback_inputs)
    success, failure, *held_out = manifest.episodes
    assert success.weights[0] / failure.weights[0] == pytest.approx(math.e)
    assert sum(success.weights + failure.weights) / 6 == pytest.approx(1)
    assert all(e.weights == [1.0] * 3 for e in held_out)
    assert dataset_digest(root) == before == manifest.dataset_sha256
    assert (
        OfflineManifest.model_validate_json(
            (feedback_inputs[2] / "weights.json").read_text()
        )
        == manifest
    )


def test_dense_feedback_keeps_regression_and_recovery(feedback_inputs):
    _, feedback, _ = feedback_inputs
    for episode in feedback:
        episode.update(progress=[0.0, 0.9, 0.4], progress_source="measured")
    manifest = _prepare(feedback_inputs, mode="progress")
    for episode in manifest.episodes[:2]:
        assert episode.weights[0] > episode.weights[2] > episode.weights[1]
    assert manifest.episodes[1].weights[0] > 0


def test_uniform_mode_accepts_unlabeled_demonstrations(feedback_inputs):
    _, feedback, directory = feedback_inputs
    for episode in feedback:
        episode.pop("success")
    manifest = _prepare(feedback_inputs, mode="uniform")
    assert manifest.mode == "uniform"
    assert all(
        weight == 1 for episode in manifest.episodes for weight in episode.weights
    )
    assert (
        manifest.feedback_sha256
        == hashlib.sha256((directory / "feedback.json").read_bytes()).hexdigest()
    )


def test_eval_feedback_does_not_fit_training_weights(feedback_inputs):
    _, feedback, directory = feedback_inputs
    first = _prepare(feedback_inputs)
    (directory / "weights.json").unlink()
    feedback[2]["success"] = False
    feedback[3]["success"] = True
    second = _prepare(feedback_inputs)
    assert first.episodes[:2] == second.episodes[:2]
    assert first.feedback_sha256 != second.feedback_sha256


@pytest.mark.parametrize(
    "change",
    [
        lambda rows: rows.append(rows[0]),
        lambda rows: rows.pop(),
        lambda rows: rows[0].update(success="true"),
        lambda rows: rows[0].update(episode_index=True),
        lambda rows: rows[2].update(group="collection-0"),
        lambda rows: rows[0].update(success=False),
        lambda rows: rows[0].update(progress=[0.0, 1.0], progress_source="measured"),
        lambda rows: rows[0].update(
            progress=[0.0, float("nan"), 1.0], progress_source="measured"
        ),
        lambda rows: rows[0].update(progress=[0.0, 0.5, 1.0], progress_source="arm"),
        lambda rows: rows[0].update(progress_source="measured"),
    ],
)
def test_invalid_feedback_fails_before_output(feedback_inputs, change):
    change(feedback_inputs[1])
    with pytest.raises(ValueError):
        _prepare(feedback_inputs)
    assert not (feedback_inputs[2] / "weights.json").exists()


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf"), True])
def test_invalid_temperature(feedback_inputs, temperature):
    with pytest.raises(ValueError, match="temperature"):
        _prepare(feedback_inputs, temperature=temperature)


def test_tiny_temperature_stays_finite(feedback_inputs):
    manifest = _prepare(feedback_inputs, temperature=1e-300)
    assert manifest.episodes[0].weights == [2.0] * 3
    assert manifest.episodes[1].weights == [0.0] * 3


def test_progress_requires_explicit_per_frame_labels(feedback_inputs):
    with pytest.raises(ValueError, match="explicit progress"):
        _prepare(feedback_inputs, mode="progress")


def test_outcome_requires_observed_labels_and_uniform_rejects_them(feedback_inputs):
    _, feedback, directory = feedback_inputs
    feedback[0].pop("success")
    with pytest.raises(ValueError, match="observed success"):
        _prepare(feedback_inputs)
    assert not (directory / "weights.json").exists()
    feedback_inputs[1][:] = [
        {**episode, "success": True} for episode in feedback_inputs[1]
    ]
    with pytest.raises(ValueError, match="unlabeled"):
        _prepare(feedback_inputs, mode="uniform")


@pytest.mark.parametrize("frames", [[0, 2, 1], [0, 1], [0.0, 1.0, 2.0]])
def test_parquet_alignment_and_integer_types(feedback_inputs, frames):
    root, _, _ = feedback_inputs
    parquet.write_table(
        pa.table({"episode_index": [0] * len(frames), "frame_index": frames}),
        root / "data/chunk-000/episode_000000.parquet",
    )
    with pytest.raises(ValueError):
        _prepare(feedback_inputs)


def test_rejects_dataset_symlink(feedback_inputs):
    root, _, directory = feedback_inputs
    (directory / "outside.bin").write_bytes(b"outside")
    (root / "videos/symlink.bin").symlink_to(directory / "outside.bin")
    with pytest.raises(ValueError, match="symlink"):
        _prepare(feedback_inputs)


def test_rejects_output_inside_dataset_or_overwrite(feedback_inputs):
    root, _, directory = feedback_inputs
    manifest = _prepare(feedback_inputs)
    assert manifest.episodes
    with pytest.raises(FileExistsError):
        _prepare(feedback_inputs)
    with pytest.raises(ValueError, match="outside"):
        prepare_feedback(
            root,
            directory / "feedback.json",
            root / "weights.json",
            action_horizon=1,
            mode="outcome",
        )


def test_manifest_rejects_duplicate_splits_and_nonfinite_weights(feedback_inputs):
    payload = _prepare(feedback_inputs).model_dump()
    payload["episodes"][0]["weights"][0] = float("inf")
    with pytest.raises(ValueError):
        OfflineManifest.model_validate(payload)


def test_runtime_local_prepare_emits_json(feedback_inputs, capsys):
    from npa.workbench.groot.offline import main

    root, feedback, directory = feedback_inputs
    path = directory / "feedback.json"
    path.write_text(json.dumps(feedback))
    main(
        [
            "prepare",
            "--dataset-root",
            str(root),
            "--feedback-path",
            str(path),
            "--output-path",
            str(directory / "weights.json"),
            "--action-horizon",
            "1",
            "--mode",
            "outcome",
        ]
    )
    assert (
        json.loads(capsys.readouterr().out)["schema_version"] == "npa.groot-offline/v1"
    )


def test_runtime_local_train_parser_exposes_native_video_backend():
    from npa.workbench.groot.offline import _parser

    args = _parser().parse_args(
        [
            "train",
            "--dataset-root",
            "dataset",
            "--manifest-path",
            "manifest.json",
            "--groot-source",
            "native",
            "--base-model",
            "checkpoint",
            "--output-dir",
            "output",
            "--data-config",
            "so100_dualcam",
            "--embodiment-tag",
            "new_embodiment",
            "--max-steps",
            "1",
            "--video-backend",
            "decord",
        ]
    )
    assert args.video_backend == "decord"


@pytest.mark.parametrize("payload", [None, 3, {}, {"episodes": []}])
def test_feedback_root_must_be_json_array(feedback_inputs, payload):
    root, _, directory = feedback_inputs
    path = directory / "feedback.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        prepare_feedback(
            root, path, directory / "weights.json", action_horizon=1, mode="outcome"
        )


@pytest.mark.parametrize(
    "field,directory", [("data_path", "meta"), ("video_path", "meta")]
)
def test_native_input_paths_cannot_escape_hashed_directories(
    feedback_inputs, field, directory
):
    root, _, _ = feedback_inputs
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info[field] = f"{directory}/unhashed-episode-{{episode_index:06d}}.bin"
    info_path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="hashed"):
        _prepare(feedback_inputs)
