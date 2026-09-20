"""SeedVR2 command-tree, request, error, and machine-output contracts."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.workbench.seedvr2 import artifacts, runtime


@pytest.mark.parametrize(
    ("verb", "module"),
    [
        ("probe", artifacts),
        ("restore", runtime),
        ("verify", artifacts),
        ("review", artifacts),
    ],
)
def test_commands_invoke_shared_operations(verb, module, monkeypatch):
    runner = CliRunner()
    result = runner.invoke(app, ["workbench", "seedvr2", verb, "--help"])
    assert result.exit_code == 0
    assert "--input-path" in result.stdout
    assert "--output-path" in result.stdout
    seen = []
    monkeypatch.setattr(
        module,
        verb,
        lambda request: (
            seen.append(request)
            or {
                "status": "ok",
                "run_id": request.run_id,
            }
        ),
    )
    result = runner.invoke(
        app,
        [
            "workbench",
            "seedvr2",
            verb,
            "--input-path",
            "s3://example-bucket/input.mp4",
            "--output-path",
            "s3://example-bucket/output/",
            "--run-id",
            "cli-test",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"run_id": "cli-test", "status": "ok"}
    assert len(seen) == 1


def test_restore_dry_run_emits_real_upstream_argv() -> None:
    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "seedvr2",
            "restore",
            "--input-path",
            "s3://example-bucket/input.mp4",
            "--output-path",
            "s3://example-bucket/output/",
            "--run-id",
            "dry-run",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "dry_run"
    assert payload["argv"][0].endswith("/torchrun")
    assert "inference_seedvr2_3b.py" in payload["argv"][3]


def test_restore_exposes_optional_probe_binding() -> None:
    result = CliRunner().invoke(
        app,
        ["workbench", "seedvr2", "restore", "--help"],
    )
    assert result.exit_code == 0
    assert "--probe-path" in result.stdout


def test_local_path_is_rejected_without_calling_runtime(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime,
        "restore",
        lambda request: pytest.fail("runtime must not receive an invalid path"),
    )
    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "seedvr2",
            "restore",
            "--input-path",
            "local-input.mp4",
            "--output-path",
            "s3://example-bucket/output/",
            "--run-id",
            "invalid",
        ],
    )
    assert result.exit_code == 1
    assert "expects an S3 URI" in result.stderr


def test_system_info_declares_runtime_only_exact_model() -> None:
    result = CliRunner().invoke(app, ["workbench", "seedvr2", "system-info"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["model"]["runtime_fetch"] is True
    assert payload["model"]["revision"] == "37255ff8cccfb01071b87f635a5948ca8d53117c"
    assert payload["source"]["license"] == "Apache-2.0"
