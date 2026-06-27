"""Compatibility entrypoint for the LeRobot CLI tests.

The LeRobot workbench tests historically lived in test_workbench_cli.py. Keep
this file so targeted runs match the per-tool naming convention used by the
other workbench tools.
"""

import base64
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest

from npa.cli.workbench import lerobot
from npa.clients.config import ServerlessJobConfig, StorageConfig
from npa.clients.serverless import EndpointNotFoundError, JobInfo, NotEnoughResourcesError

from .test_workbench_cli import _cfg, app, runner
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
            hf_token="PLACEHOLDER_HF_TOKEN",
            s3_access_key_id="",
            s3_secret_access_key="",
            s3_endpoint="",
        ),
    )
    mocker.patch("npa.cli.workbench.lerobot.resolve_container_registry", return_value="registry.example/npa")
    client.subnet_resolver = mocker.patch(
        "npa.cli.workbench.lerobot.resolve_subnet",
        return_value="vpcsubnet-1",
    )
    update = mocker.patch("npa.cli.workbench.lerobot.update_workbench_serverless_job")
    return client, update


def _mock_serverless_profile(mocker, *, existing: JobInfo | None = None, poll_status: str = "succeeded"):
    client = mocker.Mock()
    if existing is None:
        client.get_job.side_effect = EndpointNotFoundError("missing")
    else:
        client.get_job.return_value = existing
    client.create_job.return_value = JobInfo(
        id="job-1",
        name="profile-1",
        project_id="project-1",
        status="queued",
    )
    client.poll_job.return_value = JobInfo(
        id="job-1",
        name="profile-1",
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
            hf_token="PLACEHOLDER_HF_TOKEN",
            s3_access_key_id="",
            s3_secret_access_key="",
            s3_endpoint="",
        ),
    )
    mocker.patch("npa.cli.workbench.lerobot.resolve_container_registry", return_value="registry.example/npa")
    client.subnet_resolver = mocker.patch(
        "npa.cli.workbench.lerobot.resolve_subnet",
        return_value="vpcsubnet-1",
    )
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


def _serverless_profile_args(script: Path, *extra: str) -> list[str]:
    return [
        "workbench",
        "lerobot",
        "-p",
        "proj",
        "-n",
        "lerobot",
        "profile-train",
        "--runtime",
        "serverless",
        "--project-id",
        "project-1",
        "--script",
        str(script),
        "--mode",
        "wallclock",
        "--policy-type",
        "act",
        "--dataset-repo-id",
        "lerobot/pusht",
        "--steps",
        "7",
        "--batch-size",
        "2",
        "--num-workers",
        "0",
        "--warmup-steps",
        "1",
        "--gpu-type",
        "h200",
        "--gpu-count",
        "1",
        "--job-name",
        "profile-1",
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


def test_lerobot_serverless_storage_env_prefers_credentials_for_cross_bucket() -> None:
    storage = StorageConfig(
        checkpoint_bucket="s3://project-bucket/checkpoints/",
        endpoint_url="https://storage.eu-north1.nebius.cloud",
        aws_access_key_id="project-key",
        aws_secret_access_key="project-secret",
    )
    credentials = SimpleNamespace(
        s3_access_key_id="shared-key",
        s3_secret_access_key="shared-secret",
        s3_endpoint="https://storage.uk-south1.nebius.cloud",
    )

    assert lerobot._serverless_storage_env_values(
        storage,
        credentials,
        "s3://your-bucket-name/w7-fresh/run/",
    ) == (
        "shared-key",
        "shared-secret",
        "https://storage.uk-south1.nebius.cloud",
    )


def test_lerobot_serverless_storage_env_prefers_credentials_for_matching_bucket_endpoint() -> None:
    storage = StorageConfig(
        checkpoint_bucket="s3://lerobot-ccc9d3c7/checkpoints/",
        endpoint_url="https://storage.eu-north1.nebius.cloud",
        aws_access_key_id="project-key",
        aws_secret_access_key="project-secret",
    )
    credentials = SimpleNamespace(
        s3_access_key_id="shared-key",
        s3_secret_access_key="shared-secret",
        s3_endpoint="https://storage.us-central1.nebius.cloud",
        s3_bucket="s3://lerobot-ccc9d3c7/checkpoints/",
    )

    assert lerobot._serverless_storage_env_values(
        storage,
        credentials,
        "s3://lerobot-ccc9d3c7/checkpoints/lerobot/default/run-1/",
    ) == (
        "shared-key",
        "shared-secret",
        "https://storage.us-central1.nebius.cloud",
    )


def test_lerobot_serverless_env_split_keeps_secrets_extra() -> None:
    safe, extra = lerobot._split_serverless_env({"HF_TOKEN": "hf", "NPA_JOB_NAME": "job"})

    assert safe == {"NPA_JOB_NAME": "job"}
    assert extra == {"HF_TOKEN": "hf"}


def test_lerobot_status_shows_waiting_for_capacity_with_hint(mocker) -> None:
    cfg = _cfg()
    cfg.runtime = "serverless"
    cfg.serverless_job = ServerlessJobConfig(
        job_id="job-1",
        job_name="train-1",
        project_id="project-1",
        gpu_type="gpu-h200-sxm",
        gpu_count=8,
    )
    client = mocker.MagicMock()
    client.get_job.return_value = JobInfo(
        id="job-1",
        name="train-1",
        project_id="project-1",
        status="queued",
        queued_for_seconds=492,
    )
    client.classify_queue_state.return_value = "waiting_for_capacity"
    mocker.patch("npa.cli.workbench.lerobot.resolve_config", return_value=cfg)
    mocker.patch("npa.cli.workbench.lerobot.ServerlessClient", return_value=client)

    result = runner.invoke(app, ["workbench", "lerobot", "status"])

    assert result.exit_code == 0
    assert "status: waiting_for_capacity" in result.output
    assert "queue_state_classification: capacity" in result.output
    assert "Platform may be at capacity" in result.output


def test_lerobot_status_json_includes_queue_state_classification(mocker) -> None:
    cfg = _cfg()
    cfg.runtime = "serverless"
    cfg.serverless_job = ServerlessJobConfig(
        job_id="job-1",
        job_name="train-1",
        project_id="project-1",
        gpu_type="gpu-h200-sxm",
        gpu_count=8,
    )
    client = mocker.MagicMock()
    client.get_job.return_value = JobInfo(
        id="job-1",
        name="train-1",
        project_id="project-1",
        status="queued",
        queued_for_seconds=492,
    )
    client.classify_queue_state.return_value = "waiting_for_capacity"
    mocker.patch("npa.cli.workbench.lerobot.resolve_config", return_value=cfg)
    mocker.patch("npa.cli.workbench.lerobot.ServerlessClient", return_value=client)

    result = runner.invoke(app, ["workbench", "lerobot", "status", "--output", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "waiting_for_capacity"
    assert payload["queue_state_classification"] == "capacity"
    assert payload["queued_for_seconds"] == 492
    assert payload["platform"] == "gpu-h200-sxm"


def test_lerobot_gpu_platform_aliases() -> None:
    assert lerobot._lerobot_gpu_platform("h200") == "gpu-h200-sxm"
    assert lerobot._lerobot_gpu_platform("b300") == "gpu-b300-sxm"
    assert lerobot._lerobot_gpu_platform("gpu-rtx-pro-6000") == "gpu-rtx6000"
    assert lerobot._lerobot_gpu_platform("gpu-h100-sxm") == "gpu-h100-sxm"
    assert lerobot._lerobot_serverless_gpu_preset("gpu-b300-sxm", 1) == "1gpu-24vcpu-346gb"
    assert lerobot._lerobot_serverless_gpu_preset("gpu-rtx6000", 1) == "1gpu-24vcpu-218gb"


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


def test_lerobot_train_container_command_does_not_pre_create_output_dir() -> None:
    command = lerobot._lerobot_train_container_command(
        "act",
        "lerobot/pusht",
        "",
        50,
        4,
        2,
        smoke=True,
    )

    assert "mkdir -p /tmp/lerobot_output" not in command
    assert "mkdir -p /tmp/hf_home" in command


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
    assert kwargs["extra_env"]["HF_TOKEN"] == "PLACEHOLDER_HF_TOKEN"
    client.subnet_resolver.assert_called_once_with(project_id="project-1", explicit_subnet_id="")
    client.poll_job.assert_not_called()
    update.assert_called_once()


def test_lerobot_train_serverless_sync_polls(mocker) -> None:
    client, _update = _mock_serverless_train(mocker, poll_status="succeeded")

    result = runner.invoke(app, _serverless_train_args())

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "succeeded"
    client.poll_job.assert_called_once_with("job-1", "project-1", interval_s=30.0, ceiling_s=3600)


def test_lerobot_train_serverless_ner_error_formatted_for_user(mocker) -> None:
    client, _update = _mock_serverless_train(mocker)
    client.create_job.side_effect = NotEnoughResourcesError(
        "capacity blocked",
        project_id="project-1",
        platform="gpu-h200-sxm",
        suggested_alternatives=["Retry in a few minutes"],
    )
    args = _serverless_train_args("--submit-only")
    args[args.index("--output") + 1] = "text"

    result = runner.invoke(app, args)

    assert result.exit_code == 1
    assert "Not enough resources" in result.output
    assert "Retry in a few minutes" in result.output
    assert "Traceback" not in result.output


def test_lerobot_train_serverless_ner_error_json_mode(mocker) -> None:
    client, _update = _mock_serverless_train(mocker)
    client.create_job.side_effect = NotEnoughResourcesError(
        "capacity blocked",
        project_id="project-1",
        platform="gpu-h200-sxm",
        suggested_alternatives=["Retry in a few minutes"],
    )

    result = runner.invoke(app, _serverless_train_args("--submit-only"))

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"] == "NotEnoughResources"
    assert payload["platform"] == "gpu-h200-sxm"


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
    assert client.create_job.call_args.kwargs["preset"] == "1gpu-24vcpu-346gb"


def test_profile_train_serverless_requires_output_path(tmp_path: Path, mocker) -> None:
    script = tmp_path / "profile_train.py"
    script.write_text("print('profile')\n")
    _mock_serverless_profile(mocker)
    args = _serverless_profile_args(script)
    del args[args.index("--output-path"): args.index("--output-path") + 2]

    result = runner.invoke(app, args)

    assert result.exit_code == 1
    assert "requires --output-path" in result.output


def test_profile_train_serverless_default_script_path(tmp_path: Path, mocker) -> None:
    script = tmp_path / "profile_train.py"
    script.write_text("print('profile')\n")
    client, _update = _mock_serverless_profile(mocker)
    mocker.patch("npa.cli.workbench.lerobot._default_lerobot_profile_script_path", return_value=script)
    args = _serverless_profile_args(script, "--submit-only")
    del args[args.index("--script"): args.index("--script") + 2]

    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "submitted"
    assert "/tmp/profile_train.py" in client.create_job.call_args.kwargs["command"]


def test_profile_train_serverless_inline_embeds_script(tmp_path: Path) -> None:
    script = tmp_path / "profile_train.py"
    script.write_text("print('embedded profile')\n")

    command = lerobot._lerobot_profile_train_container_command(
        script_path=script,
        mode="wallclock",
        policy_type="act",
        dataset_repo_id="lerobot/pusht",
        steps=7,
        batch_size=2,
        num_workers=1,
        warmup_steps=1,
        output_path="s3://bucket/out/",
    )

    assert base64.b64encode(gzip.compress(script.read_bytes())).decode("ascii") in command
    assert "base64 -d | gzip -dc > /tmp/profile_train.py" in command
    assert "NPA_PROFILE_COMPLETE" in command


def test_profile_train_serverless_num_workers_zero_resolves_to_nproc(tmp_path: Path) -> None:
    script = tmp_path / "profile_train.py"
    script.write_text("print('profile')\n")

    command = lerobot._lerobot_profile_train_container_command(
        script_path=script,
        mode="wallclock",
        policy_type="act",
        dataset_repo_id="lerobot/pusht",
        steps=7,
        batch_size=2,
        num_workers=0,
        warmup_steps=1,
        output_path="s3://bucket/out/",
    )

    assert "NUM_WORKERS=0" in command
    assert "NUM_WORKERS=$(nproc)" in command
    assert '--num_workers="$NUM_WORKERS"' in command


def test_profile_train_serverless_validates_s3_scheme(tmp_path: Path) -> None:
    script = tmp_path / "profile_train.py"
    script.write_text("print('profile')\n")

    with pytest.raises(ValueError, match="must start with s3://"):
        lerobot._lerobot_profile_train_container_command(
            script_path=script,
            mode="wallclock",
            policy_type="act",
            dataset_repo_id="lerobot/pusht",
            steps=7,
            batch_size=2,
            num_workers=1,
            warmup_steps=1,
            output_path="/tmp/out",
        )


def test_profile_train_serverless_warns_b300_diffusion(tmp_path: Path, mocker) -> None:
    script = tmp_path / "profile_train.py"
    script.write_text("print('profile')\n")
    client, _update = _mock_serverless_profile(mocker)
    args = _serverless_profile_args(script, "--submit-only", "--gpu-type", "b300")
    args[args.index("--policy-type") + 1] = "diffusion"
    args[args.index("--output") + 1] = "text"

    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    assert "B300 is ~2.5x slower than H200 on Diffusion Policy" in result.output
    assert client.create_job.call_args.kwargs["gpu_type"] == "gpu-b300-sxm"
    assert client.create_job.call_args.kwargs["preset"] == "1gpu-24vcpu-346gb"


def test_profile_train_serverless_script_size_limit(tmp_path: Path) -> None:
    script = tmp_path / "profile_train.py"
    script.write_bytes(b"x" * 100_001)

    with pytest.raises(ValueError, match="inline embed supports up to 100KB"):
        lerobot._lerobot_profile_train_container_command(
            script_path=script,
            mode="wallclock",
            policy_type="act",
            dataset_repo_id="lerobot/pusht",
            steps=7,
            batch_size=2,
            num_workers=1,
            warmup_steps=1,
            output_path="s3://bucket/out/",
        )


def test_profile_train_serverless_submit_only(tmp_path: Path, mocker) -> None:
    script = tmp_path / "profile_train.py"
    script.write_text("print('profile')\n")
    client, update = _mock_serverless_profile(mocker)

    result = runner.invoke(app, _serverless_profile_args(script, "--submit-only"))

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "submitted"
    client.poll_job.assert_not_called()
    update.assert_called_once()


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
