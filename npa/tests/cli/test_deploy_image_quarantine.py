"""Reject withdrawn container defaults before creating or modifying a workbench."""

import pytest
from typer.testing import CliRunner

from npa.cli.main import app


@pytest.mark.parametrize("tool", ["genesis", "isaac-lab", "lerobot"])
@pytest.mark.parametrize("dry_run", [False, True])
def test_quarantined_deploy_fails_before_infrastructure(
    tool, dry_run, tmp_path, monkeypatch, mocker
) -> None:
    monkeypatch.setenv("ACCEPT_EULA", "Y")
    bootstrap = mocker.patch("npa.clients.nebius.bootstrap_environment")
    provision = mocker.patch("npa.deploy.provisioner.apply")
    initialize = mocker.patch("npa.deploy.provisioner.init")
    mocker.patch("npa.deploy.provisioner.plan", return_value="No changes.")
    mocker.patch("npa.clients.ssh.SSHClient")
    write_config = mocker.patch("npa.clients.config.write_config")
    write_env = mocker.patch("npa.deploy.configurator.write_remote_docker_env_file")
    args = [
        "workbench",
        tool,
        "deploy",
        "--runtime",
        "container",
        "--project-id",
        "project",
        "--tenant-id",
        "tenant",
        "--region",
        "eu-north1",
        "--tf-dir",
        str(tmp_path),
        "--gpu-type",
        "gpu-l40s-a",
        "--gpu-preset",
        "1gpu-40vcpu-160gb",
    ]
    if dry_run:
        args.append("--dry-run")

    result = CliRunner().invoke(app, args)

    assert result.exit_code == 1
    assert "quarantined" in result.output
    assert not isinstance(result.exception, ValueError)
    for mutation in (bootstrap, initialize, provision, write_config, write_env):
        mutation.assert_not_called()
