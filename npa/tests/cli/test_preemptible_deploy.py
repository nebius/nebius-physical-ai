"""CLI regressions for Workbench preemptible VM deploy flags."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.cli.fiftyone import FIFTYONE_VERSION

runner = CliRunner()
TERRAFORM_PLAN_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "terraform_plans"


@pytest.fixture(autouse=True)
def _terraform_plan_allows_apply(mocker):
    mocker.patch(
        "npa.cli.fiftyone.provisioner.plan",
        return_value=(TERRAFORM_PLAN_FIXTURES / "fresh_create.txt").read_text(),
    )


def _mock_fiftyone_deploy(mocker, tmp_path: Path):
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "ok", "")

    mocker.patch("npa.cli.fiftyone.provisioner.init")
    apply = mocker.patch(
        "npa.cli.fiftyone.provisioner.apply",
        return_value={
            "vm_ip": "10.0.0.5",
            "ssh_user": "ubuntu",
            "ssh_key_path": "~/.ssh/id",
            "storage_bucket": "bucket",
            "storage_endpoint": "https://storage.example",
        },
    )
    mocker.patch("npa.cli.fiftyone.SSHClient", return_value=ssh)
    mocker.patch("npa.cli.fiftyone.resolve_environment", return_value=None)
    mocker.patch("npa.cli.fiftyone.list_projects", return_value={})
    mocker.patch("npa.cli.fiftyone.write_config")
    mocker.patch("npa.cli.fiftyone.update_workbench_app_status")
    mocker.patch("npa.cli.fiftyone.write_manifest")
    mocker.patch("npa.cli.fiftyone._app_health_check", return_value=True)
    return apply


def test_fiftyone_gpu_deploy_defaults_to_preemptible(tmp_path: Path, mocker) -> None:
    apply = _mock_fiftyone_deploy(mocker, tmp_path)

    result = runner.invoke(
        app,
        [
            "workbench",
            "fiftyone",
            "-p",
            "proj",
            "-n",
            "gpu-preempt",
            "deploy",
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
        ],
    )

    assert result.exit_code == 0, result.output
    assert apply.call_args.kwargs["tf_vars"]["enable_preemptible"] == "true"


def test_fiftyone_gpu_deploy_honors_no_preemptible(tmp_path: Path, mocker) -> None:
    apply = _mock_fiftyone_deploy(mocker, tmp_path)

    result = runner.invoke(
        app,
        [
            "workbench",
            "fiftyone",
            "-p",
            "proj",
            "-n",
            "stable-gpu",
            "deploy",
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
            "--no-preemptible",
        ],
    )

    assert result.exit_code == 0, result.output
    tf_vars = apply.call_args.kwargs["tf_vars"]
    assert tf_vars["enable_preemptible"] == "false"
    assert tf_vars["fiftyone_version"] == FIFTYONE_VERSION
