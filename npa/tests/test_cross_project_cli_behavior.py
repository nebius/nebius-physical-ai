from __future__ import annotations

from pathlib import Path
from unittest.mock import ANY, call

from typer.testing import CliRunner

from npa.cli.main import app
from npa.clients.config import SSHConfig, StorageConfig, WorkbenchConfig
from npa.errors import ScopedCredentialError


runner = CliRunner()


def _cosmos_cfg() -> WorkbenchConfig:
    return WorkbenchConfig(
        endpoint="http://cosmos:8080",
        ssh=SSHConfig(host="cosmos", user="ubuntu", key_path="~/.ssh/id"),
        storage=StorageConfig(checkpoint_bucket="", endpoint_url=""),
    )


def _groot_cfg() -> WorkbenchConfig:
    return WorkbenchConfig(
        endpoint="http://groot:8080",
        ssh=SSHConfig(host="groot", user="ubuntu", key_path="~/.ssh/id"),
        storage=StorageConfig(
            checkpoint_bucket="s3://bucket/checkpoints/",
            endpoint_url="https://storage.example",
            aws_access_key_id="base-key",
            aws_secret_access_key="base-secret",
        ),
    )


def _isaac_cfg() -> WorkbenchConfig:
    return WorkbenchConfig(
        endpoint="",
        ssh=SSHConfig(host="isaac", user="ubuntu", key_path="~/.ssh/id"),
        storage=StorageConfig(checkpoint_bucket="", endpoint_url=""),
        runtime="container",
    )


def test_cosmos_infer_routes_source_and_target_project_credentials(
    tmp_path: Path, mocker
) -> None:
    downloaded = tmp_path / "downloaded.jpg"
    downloaded.write_bytes(b"image-bytes")
    source_store = mocker.MagicMock()
    source_store.download_path.return_value = str(downloaded)
    target_store = mocker.MagicMock()
    target_store.upload_file.return_value = "s3://target/results/out.mp4"
    project_clients = {
        "project-source": source_store,
        "project-target": target_store,
    }
    storage_client_for_project = mocker.patch(
        "npa.cli.cosmos.storage_client_for_project",
        side_effect=lambda project, **_: project_clients[project],
    )
    http = mocker.MagicMock()
    http.infer.return_value = {"job_id": "job-1", "status": "running"}
    http.job_status.return_value = {
        "job_id": "job-1",
        "status": "completed",
        "output_path": "/opt/cosmos/outputs/out.mp4",
    }
    ssh = mocker.MagicMock()
    mocker.patch("npa.cli.cosmos.resolve_config", return_value=_cosmos_cfg())
    mocker.patch("npa.cli.cosmos.HTTPClient", return_value=http)
    mocker.patch("npa.cli.cosmos.SSHClient", return_value=ssh)

    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos",
            "infer",
            "--prompt",
            "robot arm moving a cube",
            "--input-path",
            "s3://source/inputs/input.jpg",
            "--output-path",
            "s3://target/results/out.mp4",
            "--source-project",
            "project-source",
            "--target-project",
            "project-target",
        ],
    )

    assert result.exit_code == 0
    assert storage_client_for_project.call_args_list == [
        call("project-source", allow_host_creds=False),
        call("project-target", allow_host_creds=False),
    ]
    source_store.download_path.assert_called_once_with(
        "s3://source/inputs/input.jpg", ANY
    )
    target_store.upload_file.assert_called_once()


def test_cosmos_infer_download_failure_stays_on_source_project(mocker) -> None:
    source_store = mocker.MagicMock()
    source_store.download_path.side_effect = ScopedCredentialError(
        "source",
        "download input",
        failed_project="project-source",
    )
    target_store = mocker.MagicMock()
    project_clients = {
        "project-source": source_store,
        "project-target": target_store,
    }
    storage_client_for_project = mocker.patch(
        "npa.cli.cosmos.storage_client_for_project",
        side_effect=lambda project, **_: project_clients[project],
    )
    mocker.patch("npa.cli.cosmos.resolve_config", return_value=_cosmos_cfg())

    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos",
            "infer",
            "--prompt",
            "robot arm moving a cube",
            "--input-path",
            "s3://source/inputs/input.jpg",
            "--output-path",
            "s3://target/results/out.mp4",
            "--source-project",
            "project-source",
            "--target-project",
            "project-target",
        ],
    )

    assert result.exit_code == 1
    assert storage_client_for_project.call_args_list == [
        call("project-source", allow_host_creds=False)
    ]
    target_store.upload_file.assert_not_called()


def test_groot_infer_routes_single_project_to_ssh_env(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "NPA_GROOT_INFER_COMPLETE\n", "")
    mocker.patch("npa.cli.groot.resolve_ssh_config", return_value=_groot_cfg())
    ssh_cls = mocker.patch("npa.cli.groot.SSHClient", return_value=ssh)
    storage_env_for_project = mocker.patch(
        "npa.cli.groot.storage_env_for_project",
        return_value={"AWS_ACCESS_KEY_ID": "scoped-key"},
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "groot",
            "infer",
            "--input-path",
            "s3://bucket/checkpoint/",
            "--dataset-path",
            "s3://bucket/dataset/",
            "--output-path",
            "s3://bucket/infer/",
            "--source-project",
            "project-shared",
            "--target-project",
            "project-shared",
        ],
    )

    assert result.exit_code == 0
    storage_env_for_project.assert_called_once_with("project-shared")
    ssh_config = ssh_cls.call_args.args[0]
    assert ssh_config.tokens["AWS_ACCESS_KEY_ID"] == "scoped-key"


def test_groot_infer_rejects_conflicting_projects_before_env_resolution(mocker) -> None:
    storage_env_for_project = mocker.patch("npa.cli.groot.storage_env_for_project")

    result = runner.invoke(
        app,
        [
            "workbench",
            "groot",
            "infer",
            "--input-path",
            "s3://bucket/checkpoint/",
            "--dataset-path",
            "s3://bucket/dataset/",
            "--output-path",
            "s3://bucket/infer/",
            "--source-project",
            "project-source",
            "--target-project",
            "project-target",
        ],
    )

    assert result.exit_code == 1
    assert "NOVEL_ISSUE_E6_AUTH_SCOPE" in result.output
    storage_env_for_project.assert_not_called()


def test_isaac_lab_export_routes_target_project_credentials(
    tmp_path: Path, mocker
) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "ISAAC_LAB_EXPORT_LEROBOT_COMPLETE", "")
    cfg = _isaac_cfg()
    mocker.patch("npa.cli.isaac_lab.resolve_ssh_config", return_value=cfg)
    mocker.patch("npa.cli.isaac_lab.SSHClient", return_value=ssh)
    mocker.patch(
        "npa.cli.isaac_lab._download_remote_directory", return_value=tmp_path / "raw"
    )
    converted = tmp_path / "converted"
    converted.mkdir()
    mocker.patch("npa.adapter.isaac_lab_lerobot.convert", return_value=converted)
    storage = mocker.MagicMock()
    storage.upload_directory.return_value = "s3://target/isaac-lab/g1/"
    storage_client = mocker.patch(
        "npa.cli.isaac_lab._storage_client", return_value=storage
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "isaac-lab",
            "export-lerobot",
            "--task",
            "Isaac-Velocity-Flat-G1-v0",
            "--output-path",
            "s3://target/isaac-lab/g1/",
            "--target-project",
            "project-target",
            "--allow-host-creds",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 0
    storage_client.assert_called_once()
    assert storage_client.call_args.args == (cfg,)
    assert storage_client.call_args.kwargs == {
        "project": "project-target",
        "allow_host_creds": True,
    }
    storage.upload_directory.assert_called_once_with(
        str(converted), "s3://target/isaac-lab/g1/"
    )
