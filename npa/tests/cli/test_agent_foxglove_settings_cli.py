"""Real CLI-boundary coverage for Foxglove deployment settings."""

from __future__ import annotations

import typer
from typer.testing import CliRunner

import pytest

from npa.cli import agent as agent_module
from npa.cli import agent_foxglove_config
from npa.cli.main import app


runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated_operation_journal(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(tmp_path / "operations"))


@pytest.mark.parametrize(
    ("command", "extra_args", "blocked_symbols"),
    [
        (
            "deploy",
            ["--project-id", "project-test", "--tenant-id", "tenant-test"],
            (
                "_agent_check_whole_path_capacity",
                "_agent_storage_result",
                "_apply_agent_terraform",
            ),
        ),
        (
            "bootstrap",
            [],
            (
                "_resolve_agent_storage_credentials",
                "converge_remote_agent_setup",
                "ensure_ingress",
            ),
        ),
    ],
)
def test_invalid_viewer_backend_is_a_clean_cli_error_before_mutation(
    monkeypatch,
    mocker,
    command: str,
    extra_args: list[str],
    blocked_symbols: tuple[str, ...],
) -> None:
    monkeypatch.setattr(
        agent_module,
        "_agent_record",
        lambda *_args, **_kwargs: {"foxglove": {}} if command == "bootstrap" else {},
    )
    blocked = [mocker.patch.object(agent_module, name) for name in blocked_symbols]

    result = runner.invoke(
        app,
        [
            "agent",
            command,
            "--project",
            "settings-test",
            *extra_args,
            "--foxglove-viewer-backend",
            "invalid-backend",
        ],
    )

    assert result.exit_code == 1, result.output
    assert "--foxglove-viewer-backend" in result.output
    assert "'invalid-backend'" in result.output
    assert "foxglove-sdk, self-hosted" in result.output
    assert "Traceback" not in result.output
    assert "Unexpected error" not in result.output
    assert all(not call.called for call in blocked)


@pytest.mark.parametrize("command", ["deploy", "bootstrap"])
@pytest.mark.parametrize("backend", ["foxglove-sdk", "self-hosted"])
def test_valid_viewer_backends_reach_the_real_cli_settings_boundary(
    monkeypatch, command: str, backend: str
) -> None:
    monkeypatch.setattr(
        agent_module,
        "_agent_record",
        lambda *_args, **_kwargs: {"foxglove": {}} if command == "bootstrap" else {},
    )
    seen: list[dict[str, str]] = []
    real_resolve = agent_foxglove_config.resolve_settings

    def resolve_then_stop(**kwargs):
        seen.append(dict(kwargs))
        real_resolve(**kwargs)
        raise typer.Exit(code=0)

    monkeypatch.setattr(agent_foxglove_config, "resolve_settings", resolve_then_stop)
    args = ["agent", command, "--project", "settings-test"]
    if command == "deploy":
        args.extend(["--project-id", "project-test", "--tenant-id", "tenant-test"])
    args.extend(["--foxglove-viewer-backend", backend])
    if backend == "foxglove-sdk":
        args.extend(["--foxglove-embed-src", "https://embed.foxglove.dev/"])

    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    assert seen[-1]["viewer_backend"] == backend


@pytest.mark.parametrize(
    ("command", "extra_args", "blocked_symbol"),
    [
        (
            "deploy",
            ["--project-id", "project-test", "--tenant-id", "tenant-test"],
            "_agent_check_whole_path_capacity",
        ),
        ("bootstrap", [], "_resolve_agent_storage_credentials"),
    ],
)
def test_invalid_cloud_import_timeout_fails_before_mutation(
    monkeypatch,
    mocker,
    command: str,
    extra_args: list[str],
    blocked_symbol: str,
) -> None:
    monkeypatch.setenv(
        agent_foxglove_config.FOXGLOVE_CLOUD_IMPORT_TIMEOUT_ENV,
        "not-finite",
    )
    monkeypatch.setattr(
        agent_module,
        "_agent_record",
        lambda *_args, **_kwargs: {"foxglove": {}} if command == "bootstrap" else {},
    )
    blocked = mocker.patch.object(agent_module, blocked_symbol)

    result = runner.invoke(
        app,
        ["agent", command, "--project", "settings-test", *extra_args],
    )

    assert result.exit_code == 1, result.output
    assert agent_foxglove_config.FOXGLOVE_CLOUD_IMPORT_TIMEOUT_ENV in result.output
    assert "positive finite number" in result.output
    assert "not-finite" not in result.output
    assert "Traceback" not in result.output
    assert "Unexpected error" not in result.output
    assert not blocked.called
