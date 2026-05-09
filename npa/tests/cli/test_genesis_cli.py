from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.clients.config import SSHConfig, StorageConfig, WorkbenchConfig
from npa.clients.ssh import SSHError


runner = CliRunner()


class FakeTrainingError(Exception):
    pass


class FakeDemoGenerationError(Exception):
    pass


class FakeEvalError(Exception):
    pass


class FakeDiagnoseError(Exception):
    pass


class FakeTuneError(Exception):
    pass


def _ssh_cfg() -> WorkbenchConfig:
    return WorkbenchConfig(
        endpoint="",
        ssh=SSHConfig(host="sim", user="ubuntu", key_path="~/.ssh/id"),
        storage=StorageConfig(checkpoint_bucket="", endpoint_url=""),
    )


def _install_fake_diagnose(monkeypatch: pytest.MonkeyPatch, mocker):
    diagnose_teacher = mocker.MagicMock(
        return_value={
            "n_episodes": 1,
            "success_count": 1,
            "success_rate": 1.0,
            "phase_counts": {"success": 1},
            "bottleneck": "none",
        }
    )
    save_diagnosis = mocker.MagicMock()
    module = SimpleNamespace(
        DiagnoseError=FakeDiagnoseError,
        EpisodeTrace=None,
        _THRESHOLD_KEYS={
            "approach_threshold": "approach_dist",
            "lift_threshold": "lift_height",
            "place_threshold": "place_dist",
        },
        diagnose_teacher=diagnose_teacher,
        save_diagnosis=save_diagnosis,
    )
    monkeypatch.setitem(sys.modules, "npa.genesis.diagnose", module)
    return diagnose_teacher, save_diagnosis


@pytest.mark.parametrize(
    "command",
    [
        "train-teacher",
        "generate-demos",
        "simulate",
        "eval-teacher",
        "eval-student",
        "diagnose",
        "tune",
        "list",
        "deploy",
        "status",
        "system-info",
    ],
)
def test_genesis_command_help(command: str) -> None:
    result = runner.invoke(app, ["workbench", "genesis", command, "--help"])

    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_parse_env_overrides_and_range_expansion(
    monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    _install_fake_diagnose(monkeypatch, mocker)
    from npa.cli.genesis import _expand_env_overrides, _parse_env_overrides

    parsed = _parse_env_overrides(
        ["domain_randomize=true", "n_envs=2", "gain=1.5", "name=test"]
    )
    expanded = _expand_env_overrides(
        {**parsed, "friction_min": 0.6, "friction_max": 1.5}
    )

    assert parsed["domain_randomize"] is True
    assert parsed["n_envs"] == 2
    assert parsed["gain"] == 1.5
    assert parsed["name"] == "test"
    assert expanded["friction_range"] == (0.6, 1.5)


def test_train_teacher_dispatches(monkeypatch: pytest.MonkeyPatch, mocker) -> None:
    _install_fake_diagnose(monkeypatch, mocker)
    train_teacher = mocker.MagicMock(
        return_value={"status": "success", "checkpoint_path": "/tmp/model.pt"}
    )
    monkeypatch.setitem(
        sys.modules,
        "npa.genesis.train_teacher",
        SimpleNamespace(TrainingError=FakeTrainingError, train_teacher=train_teacher),
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "genesis",
            "train-teacher",
            "--n-envs",
            "1",
            "--max-iterations",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "Teacher training complete" in result.output
    train_teacher.assert_called_once()


def test_train_teacher_rejects_bad_n_envs() -> None:
    result = runner.invoke(
        app,
        ["workbench", "genesis", "train-teacher", "--n-envs", "0"],
    )

    assert result.exit_code == 1
    assert "n-envs must be positive" in result.output


def test_generate_demos_dispatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("checkpoint")
    generate_demos = mocker.MagicMock(
        return_value={"status": "success", "total_episodes": 1}
    )
    monkeypatch.setitem(
        sys.modules,
        "npa.genesis.generate_demos",
        SimpleNamespace(
            DemoGenerationError=FakeDemoGenerationError,
            generate_demos=generate_demos,
        ),
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "genesis",
            "generate-demos",
            "--checkpoint",
            str(checkpoint),
            "--n-envs",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "Demo generation complete" in result.output
    generate_demos.assert_called_once()


def test_generate_demos_uses_multiprocess_when_gpu_count_env_is_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("checkpoint")
    generate_demos = mocker.MagicMock(
        return_value={"status": "success", "total_episodes": 1}
    )
    monkeypatch.setitem(
        sys.modules,
        "npa.genesis.generate_demos",
        SimpleNamespace(
            DemoGenerationError=FakeDemoGenerationError,
            generate_demos=generate_demos,
        ),
    )
    monkeypatch.setenv("NPA_GPU_COUNT", "2")

    from npa.cli import genesis as genesis_cli

    multi = mocker.patch(
        "npa.cli.genesis._run_multi_gpu_generate_demos",
        return_value={
            "status": "success",
            "gpu_count": 2,
            "total_episodes": 2,
            "failed_shards": [],
        },
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "genesis",
            "generate-demos",
            "--checkpoint",
            str(checkpoint),
            "--n-envs",
            "8",
        ],
    )

    assert result.exit_code == 0
    assert "mode=multiprocess" in result.output
    generate_demos.assert_not_called()
    multi.assert_called_once()
    assert multi.call_args.kwargs["gpu_count"] == 2
    assert multi.call_args.kwargs["n_envs"] == 8
    assert genesis_cli._configured_generate_gpu_count(0) == 2


def test_generate_demos_reports_partial_multiprocess_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("checkpoint")
    monkeypatch.setitem(
        sys.modules,
        "npa.genesis.generate_demos",
        SimpleNamespace(
            DemoGenerationError=FakeDemoGenerationError,
            generate_demos=mocker.MagicMock(),
        ),
    )
    monkeypatch.setenv("NPA_GPU_COUNT", "2")
    mocker.patch(
        "npa.cli.genesis._run_multi_gpu_generate_demos",
        return_value={
            "status": "partial_failure",
            "gpu_count": 2,
            "total_episodes": 1,
            "failed_shards": [{"rank": 1, "gpu_id": "1", "exit_code": 134}],
        },
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "genesis",
            "simulate",
            "--checkpoint",
            str(checkpoint),
            "--n-envs",
            "8",
        ],
    )

    assert result.exit_code == 1
    assert "partially failed" in result.output
    assert "Failed Genesis GPU shard(s): 1" in result.output


def test_genesis_multi_gpu_split_total() -> None:
    from npa.cli.genesis import _split_total

    assert _split_total(10, 3) == [4, 3, 3]
    assert _split_total(0, 3) == [0, 0, 0]


def test_genesis_generate_shard_pins_single_gpu_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen_env: dict[str, str] = {}

    def fake_generate_demos(**_kwargs):
        seen_env.update({
            "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
            "QD_VISIBLE_DEVICE": os.environ["QD_VISIBLE_DEVICE"],
            "EGL_DEVICE_ID": os.environ["EGL_DEVICE_ID"],
        })
        return {"status": "success", "total_episodes": 1}

    monkeypatch.setitem(
        sys.modules,
        "npa.genesis.generate_demos",
        SimpleNamespace(generate_demos=fake_generate_demos),
    )

    from npa.cli.genesis import _generate_demos_shard, _GenesisGenerateShard

    messages: list[dict[str, object]] = []
    queue = SimpleNamespace(put=messages.append)
    shard = _GenesisGenerateShard(
        rank=0,
        gpu_id="3",
        n_envs=4,
        n_episodes=0,
        output_dir=str(tmp_path / "shard"),
        checkpoint_path=str(tmp_path / "model.pt"),
        domain_randomize=True,
        fps=20,
        seed=42,
        allow_failure_demos=True,
        action_space="cartesian",
    )

    _generate_demos_shard(queue, shard)

    assert seen_env == {
        "CUDA_VISIBLE_DEVICES": "3",
        "QD_VISIBLE_DEVICE": "3",
        "EGL_DEVICE_ID": "3",
    }
    assert messages == [
        {
            "rank": 0,
            "gpu_id": "3",
            "ok": True,
            "output_dir": str(tmp_path / "shard"),
            "result": {"status": "success", "total_episodes": 1},
        }
    ]


def test_genesis_multi_gpu_merge_renumbers_episode_dirs(tmp_path: Path) -> None:
    from npa.cli.genesis import _copy_shard_episodes

    shard_a = tmp_path / "shard-a"
    shard_b = tmp_path / "shard-b"
    final = tmp_path / "final"
    for root, names in (
        (shard_a, ("episode_0000", "episode_0001")),
        (shard_b, ("episode_0000",)),
    ):
        for name in names:
            episode = root / name
            episode.mkdir(parents=True)
            (episode / "actions.npy").write_text(name)
    final.mkdir()

    next_idx = _copy_shard_episodes(shard_a, final, 0)
    next_idx = _copy_shard_episodes(shard_b, final, next_idx)

    assert next_idx == 3
    assert sorted(path.name for path in final.iterdir()) == [
        "episode_0000",
        "episode_0001",
        "episode_0002",
    ]
    assert (final / "episode_0002" / "actions.npy").read_text() == "episode_0000"


def test_generate_demos_uses_output_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("checkpoint")
    output_path = tmp_path / "demos-out"
    generate_demos = mocker.MagicMock(
        return_value={"status": "success", "total_episodes": 1}
    )
    monkeypatch.setitem(
        sys.modules,
        "npa.genesis.generate_demos",
        SimpleNamespace(
            DemoGenerationError=FakeDemoGenerationError,
            generate_demos=generate_demos,
        ),
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "genesis",
            "generate-demos",
            "--checkpoint",
            str(checkpoint),
            "--n-envs",
            "1",
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert generate_demos.call_args.kwargs["output_dir"] == output_path
    assert "output_path:" in result.output


def test_generate_demos_accepts_deprecated_output_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("checkpoint")
    output_path = tmp_path / "old-demos-out"
    generate_demos = mocker.MagicMock(
        return_value={"status": "success", "total_episodes": 1}
    )
    monkeypatch.setitem(
        sys.modules,
        "npa.genesis.generate_demos",
        SimpleNamespace(
            DemoGenerationError=FakeDemoGenerationError,
            generate_demos=generate_demos,
        ),
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "genesis",
            "generate-demos",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert generate_demos.call_args.kwargs["output_dir"] == output_path


def test_genesis_simulate_alias_dispatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("checkpoint")
    output_path = tmp_path / "sim-out"
    generate_demos = mocker.MagicMock(
        return_value={"status": "success", "total_episodes": 1}
    )
    monkeypatch.setitem(
        sys.modules,
        "npa.genesis.generate_demos",
        SimpleNamespace(
            DemoGenerationError=FakeDemoGenerationError,
            generate_demos=generate_demos,
        ),
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "genesis",
            "simulate",
            "--checkpoint",
            str(checkpoint),
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert generate_demos.call_args.kwargs["output_dir"] == output_path


def test_generate_demos_missing_checkpoint_errors(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "genesis",
            "generate-demos",
            "--checkpoint",
            str(tmp_path / "missing.pt"),
        ],
    )

    assert result.exit_code == 1
    assert "Checkpoint not found" in result.output


def test_eval_teacher_dispatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("checkpoint")
    eval_teacher = mocker.MagicMock(return_value=0.5)
    monkeypatch.setitem(
        sys.modules,
        "npa.genesis.generate_demos",
        SimpleNamespace(
            DemoGenerationError=FakeDemoGenerationError,
            eval_teacher=eval_teacher,
        ),
    )

    result = runner.invoke(
        app,
        ["workbench", "genesis", "eval-teacher", "--checkpoint", str(checkpoint)],
    )

    assert result.exit_code == 0
    assert "Teacher eval complete" in result.output
    eval_teacher.assert_called_once()


def test_eval_teacher_missing_checkpoint_errors(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "genesis",
            "eval-teacher",
            "--checkpoint",
            str(tmp_path / "missing.pt"),
        ],
    )

    assert result.exit_code == 1
    assert "Checkpoint not found" in result.output


def test_eval_student_dispatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    checkpoint = tmp_path / "student"
    checkpoint.mkdir()
    eval_student = mocker.MagicMock(
        return_value={"success_rate": 0.75, "n_episodes": 4}
    )
    monkeypatch.setitem(
        sys.modules,
        "npa.genesis.eval_student",
        SimpleNamespace(EvalError=FakeEvalError, eval_student=eval_student),
    )

    result = runner.invoke(
        app,
        ["workbench", "genesis", "eval-student", "--checkpoint", str(checkpoint)],
    )

    assert result.exit_code == 0
    assert "Evaluation complete" in result.output
    eval_student.assert_called_once()


def test_eval_student_uses_input_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    checkpoint = tmp_path / "student"
    checkpoint.mkdir()
    eval_student = mocker.MagicMock(
        return_value={"success_rate": 0.75, "n_episodes": 4}
    )
    monkeypatch.setitem(
        sys.modules,
        "npa.genesis.eval_student",
        SimpleNamespace(EvalError=FakeEvalError, eval_student=eval_student),
    )

    result = runner.invoke(
        app,
        ["workbench", "genesis", "eval-student", "--input-path", str(checkpoint)],
    )

    assert result.exit_code == 0
    assert eval_student.call_args.kwargs["checkpoint_path"] == checkpoint


def test_eval_student_missing_checkpoint_errors(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "genesis",
            "eval-student",
            "--checkpoint",
            str(tmp_path / "missing"),
        ],
    )

    assert result.exit_code == 1
    assert "Checkpoint not found" in result.output


def test_diagnose_dispatches_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("checkpoint")
    diagnose_teacher, _save = _install_fake_diagnose(monkeypatch, mocker)

    result = runner.invoke(
        app,
        [
            "workbench",
            "genesis",
            "diagnose",
            "--checkpoint",
            str(checkpoint),
            "--n-envs",
            "1",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 0
    assert '"success_rate": 1.0' in result.output
    diagnose_teacher.assert_called_once()


def test_diagnose_rejects_bad_n_envs(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("checkpoint")

    result = runner.invoke(
        app,
        [
            "workbench",
            "genesis",
            "diagnose",
            "--checkpoint",
            str(checkpoint),
            "--n-envs",
            "0",
        ],
    )

    assert result.exit_code == 1
    assert "n-envs must be positive" in result.output


def test_tune_dispatches_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("checkpoint")
    _install_fake_diagnose(monkeypatch, mocker)
    tune_teacher = mocker.MagicMock(
        return_value={
            "status": "success",
            "rounds_completed": 1,
            "final_success_rate": 1.0,
            "final_checkpoint": "/tmp/model.pt",
            "rounds": [],
        }
    )
    monkeypatch.setitem(
        sys.modules,
        "npa.genesis.tune",
        SimpleNamespace(TuneError=FakeTuneError, tune_teacher=tune_teacher),
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "genesis",
            "tune",
            "--checkpoint",
            str(checkpoint),
            "--max-rounds",
            "1",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 0
    assert '"status": "success"' in result.output
    tune_teacher.assert_called_once()


def test_tune_rejects_bad_max_rounds(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("checkpoint")

    result = runner.invoke(
        app,
        [
            "workbench",
            "genesis",
            "tune",
            "--checkpoint",
            str(checkpoint),
            "--max-rounds",
            "0",
        ],
    )

    assert result.exit_code == 1
    assert "max-rounds must be positive" in result.output


def test_genesis_list_filters_to_genesis_workbenches(mocker) -> None:
    mocker.patch("npa.clients.config.default_project_name", return_value="proj")
    mocker.patch("npa.clients.config.default_workbench_name", return_value="sim")
    mocker.patch(
        "npa.clients.config.list_projects",
        return_value={
            "proj": {
                "region": "eu-north1",
                "workbenches": {
                    "sim": {
                        "workbench_type": "genesis",
                        "gpu_platform": "gpu-l40s",
                        "ssh": {"host": "sim"},
                    },
                    "train": {
                        "workbench_type": "lerobot",
                        "endpoint": "http://train:8080",
                    },
                },
            }
        },
    )

    result = runner.invoke(app, ["workbench", "genesis", "list"])

    assert result.exit_code == 0
    assert "sim" in result.output
    assert "train" not in result.output


def test_genesis_list_no_projects_message(mocker) -> None:
    mocker.patch("npa.clients.config.default_project_name", return_value="default")
    mocker.patch("npa.clients.config.default_workbench_name", return_value="default")
    mocker.patch("npa.clients.config.list_projects", return_value={})

    result = runner.invoke(app, ["workbench", "genesis", "list"])

    assert result.exit_code == 0
    assert "No projects configured" in result.output


def test_genesis_deploy_dry_run_avoids_infra(mocker) -> None:
    mocker.patch("npa.clients.config.resolve_environment", return_value=None)
    mocker.patch("npa.clients.config.list_projects", return_value={})

    result = runner.invoke(
        app,
        [
            "workbench",
            "genesis",
            "-p",
            "proj",
            "-n",
            "sim",
            "deploy",
            "--project-id",
            "project",
            "--tenant-id",
            "tenant",
            "--region",
            "eu-north1",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "Deploy complete" in result.output
    assert "ssh -i ~/.ssh/id_ed25519 ubuntu@<pending>" in result.output


def test_genesis_deploy_runtime_container_starts_image(tmp_path: Path, mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "connected", "")

    mocker.patch("npa.deploy.provisioner.init")
    apply = mocker.patch(
        "npa.deploy.provisioner.apply",
        return_value={
            "vm_ip": "10.0.0.30",
            "ssh_user": "ubuntu",
            "ssh_key_path": "~/.ssh/id",
            "storage_bucket": "bucket",
            "storage_endpoint": "https://storage.example",
        },
    )
    mocker.patch("npa.clients.config.resolve_environment", return_value=None)
    mocker.patch("npa.clients.config.list_projects", return_value={})
    write_config = mocker.patch("npa.clients.config.write_config")
    update_status = mocker.patch("npa.clients.config.update_workbench_app_status")
    mocker.patch("npa.clients.ssh.SSHClient", return_value=ssh)
    deploy_container = mocker.patch("npa.deploy.configurator.deploy_workbench_container")
    mocker.patch("npa.deploy.configurator.write_remote_env_file")
    mocker.patch("npa.deploy.configurator.write_manifest")

    result = runner.invoke(
        app,
        [
            "workbench",
            "genesis",
            "-p",
            "proj",
            "-n",
            "sim-container",
            "deploy",
            "--project-id",
            "project",
            "--tenant-id",
            "tenant",
            "--region",
            "eu-north1",
            "--tf-dir",
            str(tmp_path),
            "--runtime",
            "container",
        ],
    )

    assert result.exit_code == 0
    tf_vars = apply.call_args.kwargs["tf_vars"]
    assert tf_vars["workbench_type"] == "lerobot-container"
    assert tf_vars["boot_disk_size_gb"] == "250"
    deploy_container.assert_called_once()
    assert deploy_container.call_args.kwargs["container_name"] == "npa-genesis"
    assert deploy_container.call_args.kwargs["image_ref"].endswith("/npa-genesis:0.4.6")
    wb_cfg = write_config.call_args.args[0]["projects"]["proj"]["workbenches"]["sim-container"]
    assert wb_cfg["runtime"] == "container"
    assert update_status.call_args_list[0].args == ("proj", "sim-container", "installing")
    assert update_status.call_args_list[-1].args == ("proj", "sim-container", "healthy")


def test_genesis_deploy_vm_keeps_terraform_boot_disk_default(tmp_path: Path, mocker) -> None:
    mocker.patch("npa.deploy.provisioner.init")
    apply = mocker.patch(
        "npa.deploy.provisioner.apply",
        return_value={
            "vm_ip": "10.0.0.31",
            "ssh_user": "ubuntu",
            "ssh_key_path": "~/.ssh/id",
            "storage_bucket": "bucket",
            "storage_endpoint": "https://storage.example",
        },
    )
    mocker.patch("npa.clients.config.resolve_environment", return_value=None)
    mocker.patch("npa.clients.config.list_projects", return_value={})
    mocker.patch("npa.clients.config.write_config")
    mocker.patch("npa.clients.config.update_workbench_app_status")

    result = runner.invoke(
        app,
        [
            "workbench",
            "genesis",
            "-p",
            "proj",
            "-n",
            "sim",
            "deploy",
            "--project-id",
            "project",
            "--tenant-id",
            "tenant",
            "--region",
            "eu-north1",
            "--tf-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "boot_disk_size_gb" not in apply.call_args.kwargs["tf_vars"]


def test_genesis_deploy_rejects_invalid_tf_var() -> None:
    result = runner.invoke(
        app,
        ["workbench", "genesis", "deploy", "--tf-var", "bad"],
    )

    assert result.exit_code == 1
    assert "Invalid --tf-var format" in result.output


def test_genesis_status_prints_ssh_output(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run_or_raise.return_value = (0, "status text", "")
    mocker.patch("npa.clients.config.resolve_ssh_config", return_value=_ssh_cfg())
    mocker.patch("npa.clients.ssh.SSHClient", return_value=ssh)

    result = runner.invoke(app, ["workbench", "genesis", "status"])

    assert result.exit_code == 0
    assert "status text" in result.output


def test_genesis_status_maps_ssh_error(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run_or_raise.side_effect = SSHError("ssh failed")
    mocker.patch("npa.clients.config.resolve_ssh_config", return_value=_ssh_cfg())
    mocker.patch("npa.clients.ssh.SSHClient", return_value=ssh)

    result = runner.invoke(app, ["workbench", "genesis", "status"])

    assert result.exit_code == 1
    assert "ssh failed" in result.output


def test_genesis_system_info_prints_ssh_output(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run_or_raise.return_value = (0, "gpu info", "")
    mocker.patch("npa.clients.config.resolve_ssh_config", return_value=_ssh_cfg())
    mocker.patch("npa.clients.ssh.SSHClient", return_value=ssh)

    result = runner.invoke(app, ["workbench", "genesis", "system-info"])

    assert result.exit_code == 0
    assert "gpu info" in result.output


def test_genesis_system_info_maps_ssh_error(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run_or_raise.side_effect = SSHError("ssh failed")
    mocker.patch("npa.clients.config.resolve_ssh_config", return_value=_ssh_cfg())
    mocker.patch("npa.clients.ssh.SSHClient", return_value=ssh)

    result = runner.invoke(app, ["workbench", "genesis", "system-info"])

    assert result.exit_code == 1
    assert "ssh failed" in result.output
