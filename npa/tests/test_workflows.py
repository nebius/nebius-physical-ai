from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from npa.clients.config import SSHConfig, StorageConfig, WorkbenchConfig
from npa.clients.ssh import SSHError
from npa.workflows import distill
from npa.workflows import distill_two_vm
from npa.workflows.distill import DistillationError, RunConfig
from npa.workflows.distill_two_vm import TwoVMDistillError


class FakeSSH:
    def __init__(self, host: str = "host") -> None:
        self._config = SSHConfig(host=host, user="ubuntu", key_path="key")
        self.commands: list[str] = []
        self.responses: list[tuple[int, str, str]] = []

    def run(self, command: str, **_kwargs):
        self.commands.append(command)
        if self.responses:
            return self.responses.pop(0)
        return 0, '{"ok": true}\n', ""


def _workbench(host: str) -> WorkbenchConfig:
    return WorkbenchConfig(
        endpoint=f"http://{host}:8080",
        ssh=SSHConfig(host=host, user="ubuntu", key_path="key"),
        storage=StorageConfig(checkpoint_bucket="s3://bucket/checkpoints/", endpoint_url="url"),
    )


def test_generate_run_id_contains_timestamp_and_hash(mocker) -> None:
    mocker.patch("time.strftime", return_value="20260101-000000")
    mocker.patch("time.time_ns", return_value=123)

    run_id = distill.generate_run_id()

    assert run_id.startswith("20260101-000000-")
    assert len(run_id.rsplit("-", 1)[1]) == 8


def test_task_description_maps_known_and_unknown_tasks() -> None:
    assert distill._task_description("pick_place") == "Pick and place cube to target"
    assert distill._task_description("custom") == "custom"


def test_run_status_and_stage_logs_read_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "runs" / "run-1"
    (run_dir / "logs" / "convert").mkdir(parents=True)
    (run_dir / "logs" / "convert" / "log.txt").write_text("convert logs")
    (run_dir / "result.json").write_text(json.dumps({"status": "success"}))

    assert distill.get_run_status("run-1") == {"status": "success"}
    assert distill.get_stage_logs("run-1", "convert") == "convert logs"


def test_run_status_and_logs_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(DistillationError, match="Run not found"):
        distill.get_run_status("missing")
    with pytest.raises(DistillationError, match="Unknown stage"):
        distill.get_stage_logs("run-1", "bad")


def test_run_distillation_requires_s3_for_remote() -> None:
    with pytest.raises(DistillationError, match="Remote mode requires"):
        distill.run_distillation(remote=True, s3_bucket="")


def test_run_distillation_local_sequences_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(distill, "generate_run_id", lambda: "run-1")

    train_teacher = mocker.MagicMock(
        return_value={"checkpoint_path": str(tmp_path / "teacher" / "model.pt")}
    )
    generate_demos = mocker.MagicMock(
        return_value={"output_dir": str(tmp_path / "demos"), "includes_failures": False}
    )
    eval_teacher = mocker.MagicMock(return_value=0.8)
    train_student = mocker.MagicMock(
        return_value={"checkpoint_path": str(tmp_path / "student" / "pretrained_model")}
    )
    eval_student = mocker.MagicMock(return_value={"success_rate": 0.5})

    monkeypatch.setitem(
        sys.modules,
        "npa.genesis.train_teacher",
        SimpleNamespace(train_teacher=train_teacher),
    )
    monkeypatch.setitem(
        sys.modules,
        "npa.genesis.generate_demos",
        SimpleNamespace(generate_demos=generate_demos, eval_teacher=eval_teacher),
    )
    monkeypatch.setitem(
        sys.modules,
        "npa.lerobot.train_student",
        SimpleNamespace(train_student=train_student),
    )
    monkeypatch.setitem(
        sys.modules,
        "npa.genesis.eval_student",
        SimpleNamespace(eval_student=eval_student),
    )
    convert = mocker.patch("npa.adapter.sim_to_lerobot.convert")

    result = distill.run_distillation(n_envs=2, action_space="joint")

    assert result["status"] == "success"
    assert list(result["stages"]) == [
        "train_teacher",
        "generate_demos",
        "convert",
        "train_student",
        "eval_teacher",
        "eval_student",
    ]
    assert train_teacher.call_args.kwargs["action_space"] == "joint"
    convert.assert_called_once()


def test_run_distillation_local_saves_failed_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(distill, "generate_run_id", lambda: "run-1")
    monkeypatch.setitem(
        sys.modules,
        "npa.genesis.train_teacher",
        SimpleNamespace(train_teacher=mocker.MagicMock(side_effect=RuntimeError("boom"))),
    )

    with pytest.raises(DistillationError, match="train_teacher"):
        distill.run_distillation(n_envs=1)

    saved = json.loads((tmp_path / "runs" / "run-1" / "result.json").read_text())
    assert saved["stages"]["train_teacher"]["status"] == "failed"


def test_run_remote_sequences_stages_and_cross_vm_s3(
    tmp_path: Path, mocker
) -> None:
    cfg = RunConfig(
        run_id="run-1",
        project="proj",
        robot="franka",
        task="pick_place",
        n_envs=2,
        s3_bucket="s3://bucket",
        s3_prefix="distill/run-1",
        sim_workbench="sim",
        train_workbench="train",
    )
    ssh_instances: list[FakeSSH] = []

    def fake_ssh(config: SSHConfig) -> FakeSSH:
        ssh = FakeSSH(config.host)
        ssh_instances.append(ssh)
        return ssh

    mocker.patch(
        "npa.clients.config.resolve_config",
        side_effect=[_workbench("sim-host"), _workbench("train-host")],
    )
    mocker.patch("npa.clients.ssh.SSHClient", side_effect=fake_ssh)
    sync_calls: list[tuple[str, str, str]] = []

    def fake_sync(_ssh, _conda, *, direction, s3_uri, local_path):
        sync_calls.append((direction, s3_uri, local_path))

    mocker.patch("npa.workflows.distill._s3_sync_dir", side_effect=fake_sync)

    result = distill._run_remote(cfg, tmp_path, {"run_id": "run-1", "stages": {}})

    assert result["status"] == "success"
    assert list(result["stages"]) == distill.STAGES
    assert [call[0] for call in sync_calls] == ["upload", "download", "upload", "download"]
    assert any("train-teacher" in cmd for ssh in ssh_instances for cmd in ssh.commands)
    assert any("train-student" in cmd for ssh in ssh_instances for cmd in ssh.commands)


def test_s3_sync_dir_builds_upload_and_download_commands() -> None:
    ssh = FakeSSH()

    distill._s3_sync_dir(
        ssh,
        "activate && ",
        direction="upload",
        s3_uri="s3://bucket/prefix/",
        local_path="/tmp/local/",
    )
    distill._s3_sync_dir(
        ssh,
        "activate && ",
        direction="download",
        s3_uri="s3://bucket/prefix/",
        local_path="/tmp/local/",
    )

    assert "upload_file" in ssh.commands[0]
    assert "download_file" in ssh.commands[1]


def test_s3_sync_dir_maps_ssh_and_exit_errors() -> None:
    ssh = FakeSSH()
    ssh.run = lambda *_args, **_kwargs: (_ for _ in ()).throw(SSHError("down"))  # type: ignore[method-assign]

    with pytest.raises(DistillationError, match="S3 sync"):
        distill._s3_sync_dir(
            ssh,
            "",
            direction="upload",
            s3_uri="s3://bucket/prefix/",
            local_path="/tmp/local",
        )

    ssh2 = FakeSSH()
    ssh2.responses.append((1, "", "bad"))
    with pytest.raises(DistillationError, match="exit 1"):
        distill._s3_sync_dir(
            ssh2,
            "",
            direction="download",
            s3_uri="s3://bucket/prefix/",
            local_path="/tmp/local",
        )


def test_two_vm_run_stage_parses_trailing_json() -> None:
    ssh = FakeSSH()
    ssh.responses.append((0, 'logs\n{"answer": {"nested": true}}\n', ""))

    result = distill_two_vm._run_stage(ssh, "genesis", "stage", "cmd", "/remote")

    assert result == {
        "status": "success",
        "exit_code": 0,
        "output": {"answer": {"nested": True}},
    }
    assert "conda activate genesis" in ssh.commands[0]


def test_two_vm_run_stage_returns_failure_on_nonzero_and_ssh_error() -> None:
    ssh = FakeSSH()
    ssh.responses.append((2, "", "stderr text"))

    result = distill_two_vm._run_stage(ssh, "genesis", "stage", "cmd", "/remote")

    assert result["status"] == "failed"
    assert result["exit_code"] == 2

    ssh_error = FakeSSH()
    ssh_error.run = lambda *_args, **_kwargs: (_ for _ in ()).throw(SSHError("down"))  # type: ignore[method-assign]
    assert distill_two_vm._run_stage(
        ssh_error, "genesis", "stage", "cmd", "/remote"
    )["status"] == "failed"


def test_two_vm_pipeline_orders_stages_and_s3_handoffs(
    tmp_path: Path, mocker
) -> None:
    sim = FakeSSH("sim")
    train = FakeSSH("train")
    stage_calls: list[str] = []
    upload_calls: list[str] = []
    download_calls: list[str] = []

    def fake_run_stage(_ssh, _env, stage_name, _command, _remote_base):
        stage_calls.append(stage_name)
        output = {"teacher_success_rate": 0.9} if stage_name == "eval_teacher" else {}
        if stage_name == "generate_demos":
            output = {"includes_failures": False}
        return {"status": "success", "exit_code": 0, "output": output}

    mocker.patch("npa.workflows.distill_two_vm._write_s3_env")
    mocker.patch("npa.workflows.distill_two_vm._deploy_http_server")
    mocker.patch("npa.workflows.distill_two_vm._run_stage", side_effect=fake_run_stage)
    mocker.patch(
        "npa.workflows.distill_two_vm._s3_upload",
        side_effect=lambda *_args: upload_calls.append(_args[-1]),
    )
    mocker.patch(
        "npa.workflows.distill_two_vm._s3_download",
        side_effect=lambda *_args: download_calls.append(_args[-2]),
    )

    result = {"stages": {}}
    distill_two_vm._run_pipeline(
        sim_ssh=sim,
        train_ssh=train,
        nebius_creds={"s3_bucket": "bucket", "s3_endpoint": "url"},
        skip_infra=True,
        skip_setup=True,
        run_id="run-1",
        s3_bucket="bucket",
        s3_prefix="distill/run-1/",
        n_envs=8,
        teacher_max_iterations=1,
        demo_domain_randomize=True,
        demo_fps=20,
        demo_seed=42,
        allow_failure_demos=False,
        student_policy="act",
        student_epochs=1,
        student_batch_size=2,
        eval_n_episodes=4,
        eval_seed=7777,
        action_space="cartesian",
        result=result,
        base_dir=tmp_path,
    )

    assert stage_calls == [
        "train_teacher",
        "generate_demos",
        "eval_teacher",
        "convert",
        "train_student",
        "eval_student",
    ]
    assert upload_calls == [
        "distill/run-1/teacher/",
        "distill/run-1/dataset/",
        "distill/run-1/student/",
        "distill/run-1/eval/",
    ]
    assert download_calls == ["distill/run-1/dataset/", "distill/run-1/student/"]


def test_two_vm_s3_upload_and_download_validate_markers() -> None:
    ssh = FakeSSH()
    ssh.responses.append((0, "s3_upload_count=1\ns3_upload_done\n", ""))
    distill_two_vm._s3_upload(ssh, "genesis", "/local", "bucket", "prefix/")
    assert "upload_file" in ssh.commands[-1]

    ssh.responses.append((0, "s3_download_count=1\ns3_download_done\n", ""))
    distill_two_vm._s3_download(ssh, "genesis", "bucket", "prefix/", "/local")
    assert "download_file" in ssh.commands[-1]

    bad = FakeSSH()
    bad.responses.append((0, "s3_upload_count=0\ns3_upload_done\n", ""))
    with pytest.raises(TwoVMDistillError, match="0 files"):
        distill_two_vm._s3_upload(bad, "genesis", "/local", "bucket", "prefix/")
