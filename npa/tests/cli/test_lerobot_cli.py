"""Compatibility entrypoint for the LeRobot CLI tests.

The LeRobot workbench tests historically lived in test_workbench_cli.py. Keep
this file so targeted runs match the per-tool naming convention used by the
other workbench tools.
"""

import json
from types import SimpleNamespace
from urllib.parse import urlparse

from npa.cli.workbench import lerobot
from npa.clients.config import StorageConfig
from npa.clients.serverless import EndpointNotFoundError, JobInfo

from .test_workbench_cli import _cfg, runner
from .test_workbench_cli import *  # noqa: F401,F403


def _mock_serverless_train(mocker, *, existing: JobInfo | None = None, poll_status: str = "succeeded"):
    client = mocker.Mock()
    if existing is None:
        client.get_job.side_effect = EndpointNotFoundError("missing")
    else:
        client.get_job.return_value = existing
    client.create_job.return_value = JobInfo(
        id="job-1",
        name="train-1",
        project_id="project-1",
        status="queued",
    )
    client.poll_job.return_value = JobInfo(
        id="job-1",
        name="train-1",
        project_id="project-1",
        status=poll_status,
    )
    mocker.patch("npa.cli.workbench.lerobot.ServerlessClient", return_value=client)
    mocker.patch("npa.cli.workbench.lerobot.resolve_environment", return_value=SimpleNamespace(project_id="project-1"))
    mocker.patch(
        "npa.cli.workbench.lerobot.resolve_project_storage",
        return_value=StorageConfig(
            checkpoint_bucket="s3://bucket/checkpoints/",
            endpoint_url="https://storage.example",
            aws_access_key_id="key",
            aws_secret_access_key="secret",
        ),
    )
    mocker.patch(
        "npa.cli.workbench.lerobot.resolve_credentials",
        return_value=SimpleNamespace(
            hf_token="hf-token",
            s3_access_key_id="",
            s3_secret_access_key="",
            s3_endpoint="",
        ),
    )
    mocker.patch("npa.cli.workbench.lerobot.resolve_container_registry", return_value="registry.example/npa")
    mocker.patch("npa.cli.workbench.lerobot._lerobot_serverless_train_subnet_id", return_value="vpcsubnet-1")
    update = mocker.patch("npa.cli.workbench.lerobot.update_workbench_serverless_job")
    return client, update


def _serverless_train_args(*extra: str) -> list[str]:
    return [
        "workbench",
        "lerobot",
        "-p",
        "proj",
        "-n",
        "lerobot",
        "train",
        "--runtime",
        "serverless",
        "--policy-type",
        "act",
        "--dataset",
        "lerobot/pusht",
        "--job-name",
        "train-1",
        "--steps",
        "7",
        "--batch-size",
        "2",
        "--num-workers",
        "0",
        "--gpu-count",
        "1",
        "--output-path",
        "s3://bucket/out/",
        "--output",
        "json",
        *extra,
    ]


def test_lerobot_serverless_runtime_helper_accepts_enum_and_string() -> None:
    assert lerobot.is_serverless_runtime(lerobot.WorkbenchRuntime.serverless)
    assert lerobot.is_serverless_runtime("serverless")
    assert not lerobot.is_serverless_runtime(lerobot.WorkbenchRuntime.vm)


def test_lerobot_serverless_output_path_adds_s3_scheme() -> None:
    result = lerobot._lerobot_serverless_train_output_path("bucket/checkpoints", "wb", "job")
    parsed = urlparse(result)

    assert result == "s3://bucket/checkpoints/lerobot/wb/job/"
    assert parsed.scheme == "s3"
    assert parsed.netloc == "bucket"


def test_lerobot_serverless_output_path_preserves_s3_prefix() -> None:
    assert (
        lerobot._lerobot_serverless_train_output_path("s3://bucket/base/", "wb", "job")
        == "s3://bucket/base/lerobot/wb/job/"
    )


def test_lerobot_serverless_job_name_sanitizes_and_uses_suffix() -> None:
    assert lerobot._lerobot_serverless_job_name("Le Robot!*", "abc") == "npa-lerobot-le-robot-abc"


def test_lerobot_serverless_job_env_includes_expected_keys() -> None:
    env = lerobot._lerobot_serverless_job_env(
        "hf",
        "key",
        "secret",
        "s3://bucket/out/",
        s3_endpoint="https://storage.example",
    )

    assert env["HF_TOKEN"] == "hf"
    assert env["AWS_ACCESS_KEY_ID"] == "key"
    assert env["AWS_SECRET_ACCESS_KEY"] == "secret"
    assert env["NPA_OUTPUT_PATH"] == "s3://bucket/out/"
    assert env["LEROBOT_HF_HOME"] == "/tmp/hf_home"


def test_lerobot_serverless_env_split_keeps_secrets_extra() -> None:
    safe, extra = lerobot._split_serverless_env({"HF_TOKEN": "hf", "NPA_JOB_NAME": "job"})

    assert safe == {"NPA_JOB_NAME": "job"}
    assert extra == {"HF_TOKEN": "hf"}


def test_lerobot_gpu_platform_aliases() -> None:
    assert lerobot._lerobot_gpu_platform("h200") == "gpu-h200-sxm"
    assert lerobot._lerobot_gpu_platform("b300") == "gpu-b300-sxm"
    assert lerobot._lerobot_gpu_platform("gpu-h100-sxm") == "gpu-h100-sxm"


def test_lerobot_train_container_command_uses_smoke_settings() -> None:
    command = lerobot._lerobot_train_container_command(
        "act",
        "lerobot/pusht",
        "",
        5000,
        16,
        2,
        smoke=True,
    )

    assert "lerobot-train" in command
    assert "--steps=50" in command
    assert "--batch_size=4" in command
    assert "NPA_TRAIN_COMPLETE" in command


def test_lerobot_train_container_command_supports_s3_input() -> None:
    command = lerobot._lerobot_train_container_command(
        "diffusion",
        "",
        "s3://bucket/datasets/pusht/",
        10,
        2,
        0,
    )

    assert "download_file" in command
    assert "--dataset.repo_id=pusht" in command
    assert "--dataset.root=/tmp/lerobot_dataset/pusht" in command


def test_lerobot_serverless_subnet_selection_prefers_config(mocker) -> None:
    mocker.patch(
        "npa.cli.workbench.lerobot.list_projects",
        return_value={
            "proj": {
                "project_id": "project-1",
                "workbenches": {"lerobot": {"serverless_job": {"subnet_id": "vpcsubnet-config"}}},
            }
        },
    )
    run = mocker.patch("npa.cli.workbench.lerobot.subprocess.run")

    assert lerobot._lerobot_serverless_train_subnet_id("project-1", "proj", "lerobot") == "vpcsubnet-config"
    run.assert_not_called()


def test_lerobot_serverless_subnet_selection_prefers_lerobot_name(mocker) -> None:
    mocker.patch("npa.cli.workbench.lerobot.list_projects", return_value={})
    run = mocker.patch("npa.cli.workbench.lerobot.subprocess.run")
    run.return_value = SimpleNamespace(
        returncode=0,
        stdout=json.dumps(
            {
                "items": [
                    {"metadata": {"id": "vpcsubnet-default", "name": "default"}, "status": {"state": "READY"}},
                    {"metadata": {"id": "vpcsubnet-lerobot", "name": "lerobot-train"}, "status": {"state": "READY"}},
                ]
            }
        ),
        stderr="",
    )

    assert lerobot._lerobot_serverless_train_subnet_id("project-1") == "vpcsubnet-lerobot"


def test_lerobot_train_serverless_submit_only_creates_job(mocker) -> None:
    client, update = _mock_serverless_train(mocker)

    result = runner.invoke(app, _serverless_train_args("--submit-only"))

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "submitted"
    kwargs = client.create_job.call_args.kwargs
    assert kwargs["project_id"] == "project-1"
    assert kwargs["name"] == "train-1"
    assert kwargs["image"] == "registry.example/npa/npa-lerobot:0.5.1"
    assert kwargs["gpu_type"] == "gpu-h200-sxm"
    assert kwargs["subnet_id"] == "vpcsubnet-1"
    assert kwargs["output_path"] == "s3://bucket/out/"
    assert kwargs["env"]["NPA_JOB_NAME"] == "train-1"
    assert kwargs["extra_env"]["HF_TOKEN"] == "hf-token"
    client.poll_job.assert_not_called()
    update.assert_called_once()


def test_lerobot_train_serverless_sync_polls(mocker) -> None:
    client, _update = _mock_serverless_train(mocker, poll_status="succeeded")

    result = runner.invoke(app, _serverless_train_args())

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "succeeded"
    client.poll_job.assert_called_once_with("job-1", "project-1", interval_s=30.0, ceiling_s=3600)


def test_lerobot_train_serverless_existing_submit_is_idempotent(mocker) -> None:
    existing = JobInfo(id="job-1", name="train-1", project_id="project-1", status="succeeded")
    client, _update = _mock_serverless_train(mocker, existing=existing)

    result = runner.invoke(app, _serverless_train_args("--submit-only"))

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "existing"
    client.create_job.assert_not_called()
    client.poll_job.assert_not_called()


def test_lerobot_train_serverless_existing_running_polls(mocker) -> None:
    existing = JobInfo(id="job-1", name="train-1", project_id="project-1", status="running")
    client, _update = _mock_serverless_train(mocker, existing=existing, poll_status="succeeded")

    result = runner.invoke(app, _serverless_train_args())

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["job_status"] == "succeeded"
    client.create_job.assert_not_called()
    client.poll_job.assert_called_once_with("job-1", "project-1", interval_s=30.0, ceiling_s=3600)


def test_lerobot_train_serverless_b300_diffusion_warning(mocker) -> None:
    client, _update = _mock_serverless_train(mocker)
    args = _serverless_train_args("--submit-only", "--gpu-type", "b300")
    args[args.index("--policy-type") + 1] = "diffusion"

    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    assert "B300 is ~2.5x slower than H200 on Diffusion Policy" in result.output
    assert client.create_job.call_args.kwargs["gpu_type"] == "gpu-b300-sxm"


def test_lerobot_train_default_runtime_still_uses_ssh(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "", "")
    mocker.patch("npa.cli.workbench.lerobot.resolve_config", return_value=_cfg())
    mocker.patch("npa.clients.ssh.SSHClient", return_value=ssh)
    client_cls = mocker.patch("npa.cli.workbench.lerobot.ServerlessClient")

    result = runner.invoke(
        app,
        [
            "workbench",
            "lerobot",
            "train",
            "--policy-type",
            "act",
            "--dataset",
            "user/ds",
            "--job-name",
            "job",
            "--steps",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "lerobot-train" in ssh.run.call_args.args[0]
    client_cls.assert_not_called()
