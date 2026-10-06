"""Verify quarantined image denial at serverless CLI launch boundaries."""

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.clients.serverless import EndpointNotFoundError


runner = CliRunner()
_LAUNCHES = [
    ("genesis", "train-teacher", "npa.cli.genesis"),
    ("lerobot", "train", "npa.cli.workbench.lerobot"),
    ("lerobot", "profile-train", "npa.cli.workbench.lerobot"),
]
_OPERATOR_IMAGE = "registry.example.invalid/workbench@sha256:" + "a" * 64
_SERVERLESS_ARGS = [
    "--runtime",
    "serverless",
    "--project-id",
    "unit-project-id",
    "--output-path",
    "s3://unit-bucket/output/",
    "--gpu-type",
    "h200",
    "--job-name",
    "unit-job",
    "--submit-only",
]


def _launch_args(tool: str, command: str, script: Path) -> list[str]:
    args = ["workbench", tool, command, *_SERVERLESS_ARGS]
    if tool == "isaac-lab":
        args.extend(["--task", "Isaac-Cartpole-v0"])
    elif tool == "genesis":
        args.extend(["--n-envs", "1", "--max-iterations", "1"])
    elif command == "train":
        args.extend(["--dataset", "lerobot/pusht", "--policy-type", "act"])
    else:
        args.extend(
            [
                "--script",
                str(script),
                "--policy-type",
                "act",
                "--dataset-repo-id",
                "lerobot/pusht",
                "--steps",
                "2",
                "--warmup-steps",
                "1",
            ]
        )
    return args


def _launch_boundaries(mocker, module_name: str):
    module = import_module(module_name)
    client = mocker.Mock()
    client.get_job.side_effect = EndpointNotFoundError("missing")
    client.create_job.return_value = SimpleNamespace(
        id="unit-job-id", name="unit-job", status="queued", output_uris=()
    )
    boundaries = {
        "client": mocker.patch.object(module, "ServerlessClient", return_value=client),
        "subnet": mocker.patch.object(
            module, "resolve_subnet", return_value="unit-subnet"
        ),
    }
    mocker.patch.object(module, "resolve_environment", return_value=None)
    mocker.patch.object(module, "default_project_name", return_value="unit-project")
    mocker.patch.object(module, "default_workbench_name", return_value="unit-workbench")
    if module_name.endswith("genesis"):
        boundaries["image_pin"] = mocker.patch.object(
            module, "_pin_serverless_image", side_effect=lambda image: image
        )
        boundaries["execution_preflight"] = mocker.patch(
            "npa.execution_preflight.verify_serverless_execution"
        )
    if module_name.endswith("lerobot"):
        boundaries.update(_lerobot_boundaries(mocker, module))
    else:
        boundaries["environment"] = mocker.patch.object(
            module, "_serverless_job_env", return_value=({}, {})
        )
    return module, client, boundaries


def _lerobot_boundaries(mocker, module):
    return {
        "storage": mocker.patch.object(
            module,
            "resolve_project_storage",
            return_value=SimpleNamespace(
                checkpoint_bucket="s3://unit-bucket/checkpoints/",
                endpoint_url="https://storage.example.invalid",
                aws_access_key_id="unit-access-key",
                aws_secret_access_key="unit-secret-key",
            ),
        ),
        "credentials": mocker.patch.object(
            module,
            "resolve_credentials",
            return_value=SimpleNamespace(
                hf_token="",
                s3_access_key_id="",
                s3_secret_access_key="",
                s3_endpoint="",
            ),
        ),
        "state": mocker.patch.object(module, "update_workbench_serverless_job"),
    }


def _credential_writes(mocker):
    return [
        mocker.patch("npa.clients.credentials.write_credentials_file"),
        mocker.patch("npa.clients.credentials.persist_supported_env_credentials"),
    ]


@pytest.mark.parametrize(("tool", "command", "module_name"), _LAUNCHES)
def test_quarantined_launch_is_clean_and_precedes_provider_calls(
    mocker, tmp_path, tool, command, module_name
) -> None:
    script = tmp_path / "profile.py"
    script.write_text("print('profile')\n", encoding="utf-8")
    _module, client, boundaries = _launch_boundaries(mocker, module_name)
    writes = _credential_writes(mocker)

    result = runner.invoke(app, _launch_args(tool, command, script))

    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    assert "Error:" in result.output
    assert "quarantined" in result.output
    assert "operator-controlled" in result.output
    assert "Traceback" not in result.output
    for boundary in [*boundaries.values(), *writes]:
        boundary.assert_not_called()
    client.get_job.assert_not_called()
    client.create_job.assert_not_called()


@pytest.mark.parametrize(
    ("tool", "command", "module_name"),
    [*_LAUNCHES, ("isaac-lab", "train", "npa.cli.isaac_lab")],
)
def test_explicit_operator_image_preserves_serverless_launch(
    mocker, tmp_path, tool, command, module_name
) -> None:
    script = tmp_path / "profile.py"
    script.write_text("print('profile')\n", encoding="utf-8")
    module, client, _boundaries = _launch_boundaries(mocker, module_name)
    resolver = mocker.patch.object(module, "container_image_for_tool")
    resolver.side_effect = AssertionError("Explicit images must skip public defaults")
    writes = _credential_writes(mocker)
    args = [*_launch_args(tool, command, script), "--image", _OPERATOR_IMAGE]

    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    resolver.assert_not_called()
    client.create_job.assert_called_once()
    assert client.create_job.call_args.kwargs["image"] == _OPERATOR_IMAGE
    for write in writes:
        write.assert_not_called()


@pytest.mark.parametrize(
    "selection", ["explicit-060", "environment-060", "unsupported"]
)
def test_lerobot_version_denial_precedes_existing_job_lookup(
    mocker, tmp_path, monkeypatch, selection
) -> None:
    _module, client, boundaries = _launch_boundaries(
        mocker, "npa.cli.workbench.lerobot"
    )
    client.get_job.side_effect = None
    client.get_job.return_value = SimpleNamespace(
        id="unit-existing-id", status="running"
    )
    args = _launch_args("lerobot", "train", tmp_path / "unused")
    if selection == "environment-060":
        monkeypatch.setenv("NPA_LEROBOT_VERSION", "0.6.0")
    else:
        args.extend(
            ["--lerobot-version", "9.9.9" if selection == "unsupported" else "0.6.0"]
        )

    result = runner.invoke(app, args)

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    expected = (
        "Unsupported LeRobot version" if selection == "unsupported" else "quarantined"
    )
    assert expected in result.output
    for boundary in boundaries.values():
        boundary.assert_not_called()
    client.get_job.assert_not_called()
    client.create_job.assert_not_called()


def test_isaac_fresh_launch_quarantine_is_clean_without_writes(mocker, tmp_path):
    _module, client, boundaries = _launch_boundaries(mocker, "npa.cli.isaac_lab")
    writes = _credential_writes(mocker)

    result = runner.invoke(app, _launch_args("isaac-lab", "train", tmp_path / "unused"))

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "Error:" in result.output and "quarantined" in result.output
    assert "Traceback" not in result.output
    client.get_job.assert_called_once_with("unit-job", "unit-project-id")
    client.create_job.assert_not_called()
    for write in writes:
        write.assert_not_called()
    boundaries["client"].assert_called_once()


def test_isaac_existing_job_reconnect_does_not_select_new_image(mocker, tmp_path):
    module, client, _boundaries = _launch_boundaries(mocker, "npa.cli.isaac_lab")
    client.get_job.side_effect = None
    client.get_job.return_value = SimpleNamespace(
        id="unit-existing-id", name="unit-job", status="running"
    )
    resolver = mocker.patch.object(module, "container_image_for_tool")
    writes = _credential_writes(mocker)

    result = runner.invoke(app, _launch_args("isaac-lab", "train", tmp_path / "unused"))

    assert result.exit_code == 0, result.output
    assert "existing" in result.output
    assert "unit-existing-id" in result.output
    resolver.assert_not_called()
    client.create_job.assert_not_called()
    client.poll_job.assert_not_called()
    for write in writes:
        write.assert_not_called()


def test_sonic_serverless_default_remains_clean_denial_before_launch(mocker):
    module = import_module("npa.cli.workbench.sonic.train")
    mocker.patch.object(module, "resolve_project_id", return_value="unit-project-id")
    boundaries = [
        mocker.patch.object(module, "ServerlessClient"),
        mocker.patch.object(module, "resolve_subnet"),
        mocker.patch.object(module, "serverless_job_env"),
        mocker.patch.object(module, "sonic_image"),
        *_credential_writes(mocker),
    ]
    result = runner.invoke(
        app,
        [
            "workbench",
            "sonic",
            "train",
            "--runtime",
            "serverless",
            "--project-id",
            "unit-project-id",
            "--output-path",
            "s3://unit-bucket/output/",
            "--submit-only",
        ],
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "quarantined" in result.output and "--image" in result.output
    for boundary in boundaries:
        boundary.assert_not_called()
