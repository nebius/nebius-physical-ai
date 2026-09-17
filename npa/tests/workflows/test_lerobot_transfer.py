"""Verify transfer data isolation, paired statistics, and native-stage failure boundaries."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from npa.workflows import lerobot_transfer_data as data
from npa.workflows import lerobot_transfer_training as training
from npa.workflows.lerobot_transfer import _recipe, build_parser
from npa.workflows.lerobot_transfer_eval import make_shifted_env, shift_pixels, summarize_rollout
from npa.workflows.lerobot_transfer_report import _record_trials, compare_trials


@pytest.fixture
def recipe():
    return _recipe(build_parser().parse_args([
        "prepare", "--output-path", "unused", "--validation-episodes", "4", "--test-episodes", "8",
    ]))


@pytest.fixture
def evaluated(recipe):
    recipe["conditions"] = ["clean", "dim", "warm", "delay"]
    trials = []
    for arm in ("baseline", "robust"):
        for split in ("validation", "test"):
            for condition in recipe["conditions"]:
                for seed in range(recipe[f"{split}_seed"], recipe[f"{split}_seed"]
                                  + recipe[f"{split}_episodes"]):
                    success = arm == "robust"
                    trials.append({"arm": arm, "split": split, "condition": condition,
                                   "seed": seed, "success": success,
                                   "sum_reward": 1.0, "max_reward": 1.0 if success else 0.2})
    return {"recipe": recipe, "trials": trials, "checkpoint_hashes": {"baseline": {}, "robust": {}},
            "recipe_sha256": "a" * 64}


def _dataset(root):
    target = root / "snapshot"
    data.write_json(target / "meta/info.json", {
        "codebase_version": "v3.0", "fps": 10, "total_frames": 30, "total_episodes": 10,
        "features": {"action": {"shape": [2]}, "observation.state": {"shape": [2]}},
    })
    rows = [{"episode_index": episode, "frame_index": frame, "timestamp": frame / 10,
             "action": [float(episode * 20), float(episode * 20 + frame)],
             "observation.state": [float(episode), float(frame)]}
            for episode in range(10) for frame in range(3)]
    (target / "data").mkdir()
    pq.write_table(pa.Table.from_pylist(rows), target / "data/frames.parquet")
    return target


def test_preparation_fits_statistics_only_on_training_episodes(tmp_path, monkeypatch, recipe):
    monkeypatch.setattr(data, "download_public_lerobot_dataset", _dataset)
    data.prepare_dataset(tmp_path / "prepared", recipe)
    root = tmp_path / "prepared"
    sealed = json.loads((root / "recipe.json").read_text())
    assert set(sealed["train_episodes"]).isdisjoint(sealed["reserved_episodes"])
    assert sorted(sealed["train_episodes"] + sealed["reserved_episodes"]) == list(range(10))
    stats = json.loads((root / "dataset/meta/stats.json").read_text())
    expected = np.mean(sealed["train_episodes"]) * 20
    assert stats["action"]["mean"][0] == expected
    assert stats["action"]["count"] == [24]
    assert sealed["train_frames"] == 24
    assert sealed["dataset_revision"] == data.DEFAULT_PUBLIC_LEROBOT_REVISION
    assert sealed["video_backend"] == "torchcodec"


@pytest.mark.parametrize("mutation", ["duplicate", "gap", "timestamp", "nan", "workspace"])
def test_dataset_rejects_invalid_action_and_timing_contract(tmp_path, mutation):
    root = _dataset(tmp_path)
    path = root / "data/frames.parquet"
    rows = pq.read_table(path).to_pylist()
    if mutation == "duplicate":
        rows[1]["frame_index"] = 0
    elif mutation == "gap":
        rows[1]["frame_index"] = 7
    elif mutation == "timestamp":
        rows[1]["timestamp"] = 5
    else:
        rows[1]["action"][0] = float("nan") if mutation == "nan" else 600
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError):
        data._read_rows(root, json.loads((root / "meta/info.json").read_text()))


def test_stage_exchange_rejects_tampered_and_unlisted_files(tmp_path):
    source = tmp_path / "source"
    data.write_json(source / "recipe.json", {"seed": 7})
    destination = tmp_path / "published"
    data.publish(source, str(destination))
    assert data.materialize(str(destination), tmp_path / "unused") == destination
    (destination / "unexpected.txt").write_text("stale artifact")
    with pytest.raises(ValueError, match="checksum"):
        data.materialize(str(destination), tmp_path / "unused")
    (destination / "unexpected.txt").unlink()
    data.write_json(destination / "recipe.json", {"seed": 8})
    with pytest.raises(ValueError, match="checksum"):
        data.materialize(str(destination), tmp_path / "unused")


def test_training_uses_sealed_split_and_geometry_preserving_augmentation(tmp_path, recipe, monkeypatch):
    monkeypatch.setenv("NPA_LEROBOT_VERSION", "0.5.1")
    recipe.update(dataset_revision="b" * 40, dataset_repo="lerobot/pusht", train_episodes=[1, 3, 6],
                  lerobot_version="0.6.0", video_backend="torchcodec")
    data.write_json(tmp_path / "recipe.json", recipe)
    baseline = training.training_command(tmp_path, tmp_path / "train", "baseline")
    robust = training.training_command(tmp_path, tmp_path / "train", "robust")
    assert "--dataset.episodes=[1, 3, 6]" in baseline
    assert "--dataset.video_backend=torchcodec" in baseline
    transforms = json.loads(next(arg.split("=", 1)[1] for arg in robust
                                 if arg.startswith("--dataset.image_transforms.tfs=")))
    assert transforms["affine"]["weight"] == 0
    assert transforms["brightness"]["kwargs"]["brightness"] == [0.4, 1.4]
    assert set(robust) - set(baseline) == {"--dataset.image_transforms.enable=true"}
    assert set(baseline) - set(robust) == {"--dataset.image_transforms.enable=false"}
    assert "--env_eval_freq=0" in baseline
    assert "--eval_freq=0" not in baseline


def test_training_failure_cannot_publish_a_checkpoint(tmp_path, monkeypatch, recipe):
    recipe.update(dataset_revision="b" * 40, dataset_repo="lerobot/pusht", train_episodes=[0],
                  lerobot_version="0.6.0", video_backend="torchcodec")
    data.write_json(tmp_path / "recipe.json", recipe)
    monkeypatch.setattr(training, "runtime_versions", lambda: {"lerobot": "0.6.0"})
    reached = []

    def fail(command, log_path):
        reached.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(training, "_run_training", fail)
    with pytest.raises(subprocess.CalledProcessError):
        training.train_policy(tmp_path, tmp_path / "output", "baseline")
    assert reached and not (tmp_path / "output/training.json").exists()


def test_native_training_failure_remains_visible_in_worker_logs(tmp_path, capsys):
    command = [sys.executable, "-c", "print('native failure detail'); raise SystemExit(3)"]
    with pytest.raises(subprocess.CalledProcessError) as failure:
        training._run_training(command, tmp_path / "train.log")
    assert failure.value.returncode == 3
    assert "native failure detail" in capsys.readouterr().out
    assert "native failure detail" in (tmp_path / "train.log").read_text()


def test_improvement_requires_validation_selection_and_paired_test_evidence(evaluated):
    report = compare_trials(evaluated)
    assert report["selected_arm"] == "robust"
    assert report["improvement_demonstrated"] is True
    assert report["test_paired_delta_95ci"] == [1.0, 1.0]
    assert report["ready_for_robot_deployment"] is False
    assert report["physical_robot_tested"] is False


def test_test_results_never_select_checkpoint_or_drive_demonstration_queue(evaluated):
    for trial in evaluated["trials"]:
        if trial["split"] == "validation":
            trial["success"] = False
    report = compare_trials(evaluated)
    assert report["selected_arm"] == "baseline"
    assert report["improvement_demonstrated"] is False
    assert len(report["next_demonstrations"]) == 16
    assert {request["source_split"] for request in report["next_demonstrations"]} == {"validation"}
    assert max(request["reset_seed"] for request in report["next_demonstrations"]) < evaluated["recipe"]["test_seed"]


def test_clean_regression_blocks_an_average_shift_improvement(evaluated):
    for trial in evaluated["trials"]:
        if trial["split"] == "test" and trial["condition"] == "clean":
            trial["success"] = trial["arm"] == "baseline"
    report = compare_trials(evaluated)
    assert report["test_paired_delta_95ci"][0] > 0
    assert report["improvement_demonstrated"] is False


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unexpected", "nan", "overlap"])
def test_incomplete_or_unpaired_evaluation_fails_closed(evaluated, mutation):
    if mutation == "missing":
        evaluated["trials"].pop()
    elif mutation == "duplicate":
        evaluated["trials"].append(copy.deepcopy(evaluated["trials"][0]))
    elif mutation == "unexpected":
        evaluated["trials"][0]["seed"] = -1
    elif mutation == "nan":
        evaluated["trials"][0]["max_reward"] = float("nan")
    else:
        evaluated["recipe"]["test_seed"] = evaluated["recipe"]["validation_seed"]
    with pytest.raises(ValueError):
        compare_trials(evaluated)


def test_photometric_shifts_preserve_shape_and_object_coordinates():
    pixels = np.zeros((96, 96, 3), dtype=np.uint8)
    pixels[10:20, 30:40] = 180
    for condition in ("clean", "dim", "warm", "delay"):
        shifted = shift_pixels(pixels, condition)
        assert shifted.shape == pixels.shape and shifted.dtype == pixels.dtype
        assert np.array_equal(np.any(shifted, axis=2), np.any(pixels, axis=2))
    assert shift_pixels(pixels, "dim").max() < pixels.max()


def test_post_reset_success_and_reward_never_count_toward_the_original_episode():
    rollout = {"reward": np.array([[0.1, 0.2, 1.0], [0.1, 0.2, 1.0]]),
               "success": np.array([[False, False, True], [False, False, True]]),
               "done": np.array([[False, True, True], [False, False, True]])}
    trials = summarize_rollout(rollout, [101, 102])
    assert trials[0]["success"] is False
    assert trials[0]["max_reward"] == 0.2
    assert trials[0]["steps"] == 2
    assert trials[1]["success"] is True
    assert trials[1]["steps"] == 3


def test_native_delay_holds_one_step_and_clears_between_resets(monkeypatch):
    pytest.importorskip("gym_pusht")
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    delayed, reference = make_shifted_env("delay"), make_shifted_env("clean")
    try:
        for seed in (100_001, 100_002):
            observation, _ = delayed.reset(seed=seed)
            reference.reset(seed=seed)
            first = np.array([111, 222], dtype=np.float32)
            second = np.array([333, 444], dtype=np.float32)
            delayed_first = delayed.step(first)[0]
            expected_first = reference.step(observation["agent_pos"])[0]
            assert np.array_equal(delayed_first["pixels"], expected_first["pixels"])
            delayed_second = delayed.step(second)[0]
            expected_second = reference.step(first)[0]
            assert np.array_equal(delayed_second["pixels"], expected_second["pixels"])
    finally:
        delayed.close()
        reference.close()


def test_rerun_records_actual_trial_entities_and_seed_timeline(tmp_path, evaluated):
    output = tmp_path / "transfer.rrd"
    _record_trials(evaluated, output, "fixture-transfer")
    binary = Path(__import__("sys").executable).parent / "rerun"
    result = subprocess.run([str(binary), "rrd", "verify", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    decoded = subprocess.run([str(binary), "rrd", "print", "-vv", str(output)],
                             capture_output=True, text=True, check=True).stdout
    assert "reset_seed" in decoded
    assert "test/clean/robust/success" in decoded
    assert "npa_lerobot_transfer" in decoded
    assert "fixture-transfer" in decoded
