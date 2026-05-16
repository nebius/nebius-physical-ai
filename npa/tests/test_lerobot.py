from __future__ import annotations

from types import SimpleNamespace

import pytest

from npa.lerobot.train_student import (
    StudentTrainingError,
    _estimate_steps,
    build_train_command,
    train_student,
)


def test_estimate_steps_reads_dataset_frame_count(tmp_path):
    dataset = tmp_path / "dataset"
    meta = dataset / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text('{"total_frames": 96}')

    assert _estimate_steps(dataset, num_epochs=3, batch_size=24) == 12


def test_estimate_steps_uses_default_when_info_missing(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    assert _estimate_steps(dataset, num_epochs=2, batch_size=50_000) == 2


def test_build_train_command_uses_local_dataset_root_and_extra_args(tmp_path):
    dataset = tmp_path / "robot-demos"
    output = tmp_path / "student"

    cmd = build_train_command(
        str(dataset),
        str(output),
        policy_type="diffusion",
        steps=123,
        batch_size=16,
        device="cpu",
        num_workers=0,
        extra_args={"policy.n_action_steps": "8"},
    )

    assert cmd[0] == "lerobot-train"
    assert "--policy.type=diffusion" in cmd
    assert f"--dataset.repo_id={dataset.name}" in cmd
    assert f"--dataset.root={dataset}" in cmd
    assert "--save_freq=123" in cmd
    assert "--policy.n_action_steps=8" in cmd


def test_train_student_runs_lerobot_subprocess_offline(tmp_path, mocker):
    dataset = tmp_path / "dataset"
    meta = dataset / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text('{"total_frames": 128}')
    output = tmp_path / "student"
    run = mocker.patch(
        "npa.lerobot.train_student.subprocess.run",
        return_value=SimpleNamespace(returncode=0),
    )

    result = train_student(
        dataset,
        output,
        num_epochs=2,
        batch_size=32,
        device="cpu",
        stream=False,
    )

    command = run.call_args.args[0]
    assert "--steps=8" in command
    assert run.call_args.kwargs["env"]["HF_HUB_OFFLINE"] == "1"
    assert result["status"] == "success"
    assert result["checkpoint_path"].endswith(
        "student/checkpoints/last/pretrained_model"
    )


def test_train_student_validates_dataset_and_output_paths(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    with pytest.raises(StudentTrainingError, match="missing meta/info.json"):
        train_student(dataset, tmp_path / "out", stream=False)

    meta = dataset / "meta"
    meta.mkdir()
    (meta / "info.json").write_text("{}")
    output = tmp_path / "out"
    output.mkdir()

    with pytest.raises(StudentTrainingError, match="Output directory already exists"):
        train_student(dataset, output, stream=False)


def test_train_student_maps_nonzero_exit_to_error(tmp_path, mocker):
    dataset = tmp_path / "dataset"
    meta = dataset / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text("{}")
    mocker.patch(
        "npa.lerobot.train_student.subprocess.run",
        return_value=SimpleNamespace(returncode=7),
    )

    with pytest.raises(StudentTrainingError, match="exit 7"):
        train_student(dataset, tmp_path / "out", stream=False)
