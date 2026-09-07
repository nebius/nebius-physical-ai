"""CLI tests for `npa workbench robocasa`."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from npa.cli.main import app as main_app
from npa.cli.workbench.robocasa import app as robocasa_app
from npa.cli.workbench.robocasa import deploy as deploy_module


runner = CliRunner()


def test_registered_under_workbench() -> None:
    result = runner.invoke(main_app, ["workbench", "robocasa", "--help"])
    assert result.exit_code == 0
    assert "RoboCasa kitchen-task simulation workbench" in result.stdout


def test_help_lists_commands() -> None:
    result = runner.invoke(robocasa_app, ["--help"])
    assert result.exit_code == 0
    for command in ("deploy", "run", "status", "system-info", "list"):
        assert command in result.stdout


def test_run_help() -> None:
    result = runner.invoke(robocasa_app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--capability" in result.stdout
    assert "--output-path" in result.stdout
    assert "--output-uri" in result.stdout


def test_deploy_help() -> None:
    result = runner.invoke(robocasa_app, ["deploy", "--help"])
    assert result.exit_code == 0
    assert "--gpu-type" in result.stdout
    assert "--auth-mode" in result.stdout


def test_deploy_service_env_prefers_project_scoped_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(deploy_module, "load_credentials", object)
    monkeypatch.setattr(
        deploy_module,
        "apply_shared_credential_env",
        lambda env, _creds: env.update(
            {
                "AWS_ACCESS_KEY_ID": "host-ak",
                "AWS_SECRET_ACCESS_KEY": "host-sk",
                "AWS_ENDPOINT_URL": "https://host.invalid",
            }
        ),
    )
    monkeypatch.setattr(
        deploy_module,
        "storage_env_for_project",
        lambda project: {
            "AWS_ACCESS_KEY_ID": f"{project}-ak",
            "AWS_SECRET_ACCESS_KEY": f"{project}-sk",
            "AWS_ENDPOINT_URL": "https://project.invalid",
            "NEBIUS_S3_ENDPOINT": "https://project.invalid",
        },
    )

    env = deploy_module._service_env(
        project="fleet-test",
        output_path="s3://example/output",
        auth_mode="none",
        token_env="ROBOCASA_TOKEN",
        port=8791,
    )

    assert env["AWS_ACCESS_KEY_ID"] == "fleet-test-ak"
    assert env["AWS_SECRET_ACCESS_KEY"] == "fleet-test-sk"
    assert env["AWS_ENDPOINT_URL"] == "https://project.invalid"
    assert env["AWS_ENDPOINT_URL_S3"] == "https://project.invalid"


def test_deploy_manifest_rolls_when_service_env_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def manifest() -> dict:
        return deploy_module._kubernetes_manifest(
            project="fleet-test",
            image="example.invalid/npa-robocasa@sha256:" + "1" * 64,
            name="npa-robocasa",
            namespace="default",
            port=8791,
            output_path="s3://example/output",
            node_selector_key="node.kubernetes.io/instance-type",
            node_selector_value="gpu-test",
            image_pull_secret="pull-secret",
            auth_mode="none",
            token_env="ROBOCASA_TOKEN",
        )

    monkeypatch.setattr(
        deploy_module,
        "_service_env",
        lambda **_kwargs: {"AWS_ACCESS_KEY_ID": "first-ak"},
    )
    first = manifest()
    monkeypatch.setattr(
        deploy_module,
        "_service_env",
        lambda **_kwargs: {"AWS_ACCESS_KEY_ID": "second-ak"},
    )
    second = manifest()

    first_annotation = first["items"][1]["spec"]["template"]["metadata"]["annotations"]
    second_annotation = second["items"][1]["spec"]["template"]["metadata"]["annotations"]
    assert first_annotation != second_annotation
    assert len(first_annotation["npa.nebius.ai/env-checksum"]) == 64


def test_status_help() -> None:
    result = runner.invoke(robocasa_app, ["status", "--help"])
    assert result.exit_code == 0
    assert "--run-id" in result.stdout


def test_system_info_help() -> None:
    result = runner.invoke(robocasa_app, ["system-info", "--help"])
    assert result.exit_code == 0


def test_list_help() -> None:
    result = runner.invoke(robocasa_app, ["list", "--help"])
    assert result.exit_code == 0


def test_run_requires_capability() -> None:
    result = runner.invoke(robocasa_app, ["run"])
    assert result.exit_code != 0


def test_run_invalid_capability_local() -> None:
    result = runner.invoke(
        robocasa_app,
        ["run", "--capability", "bogus", "--output-uri", "s3://bucket/out"],
    )
    assert result.exit_code != 0


@pytest.mark.parametrize("value", ["/tmp/output", "file:///tmp/output", "https://example.invalid/out"])
def test_run_rejects_non_s3_output_at_cli_boundary(value: str) -> None:
    result = runner.invoke(
        robocasa_app,
        ["run", "--capability", "kitchen_random_rollout", "--output-path", value],
    )
    assert result.exit_code == 1
    assert "expects an S3 URI" in result.stderr


def test_run_accepts_legacy_output_uri_alias_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, str] = {}

    class Response:
        def model_dump(self, *, mode: str) -> dict[str, str]:
            return {"run_id": "local", "status": "completed"}

    def fake_run(**kwargs):
        observed.update(kwargs)
        return Response()

    monkeypatch.setattr("npa.sdk.workbench.robocasa.run", fake_run)
    result = runner.invoke(
        robocasa_app,
        [
            "run",
            "--capability",
            "kitchen_task_registration",
            "--output-uri",
            "s3://example/output",
        ],
    )
    assert result.exit_code == 0
    assert observed["output_path"] == "s3://example/output"


def test_system_info_local() -> None:
    result = runner.invoke(robocasa_app, ["system-info", "--output", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"


def test_run_trajectory_export_help() -> None:
    result = runner.invoke(robocasa_app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--capability" in result.stdout


def test_run_trajectory_export_capability_accepted() -> None:
    # The schema accepts the new trajectory export capability.
    from npa.workbench.robocasa.schemas import RoboCasaRunRequest

    req = RoboCasaRunRequest(
        capability="kitchen_trajectory_export",
        output_uri="s3://bucket/out",
        iterations=5,
        num_envs=2,
    )
    assert req.capability == "kitchen_trajectory_export"
    assert req.iterations == 5
    assert req.num_envs == 2
