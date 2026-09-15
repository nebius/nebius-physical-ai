"""Explicit unlimited evaluations must never inherit an execution deadline."""

from __future__ import annotations

from dataclasses import replace
import argparse
import importlib.util
import json
import math
from pathlib import Path
import subprocess
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner
import yaml

from npa.cli.main import app
from npa.cli.workbench import golden_eval as cli
from npa.smoke import batch, manifest, serverless_runner


def _spec(monkeypatch: pytest.MonkeyPatch, timeout: object) -> manifest.ContainerSpec:
    payload = {
        "format": manifest.MANIFEST_FORMAT,
        "containers": {
            "fixture-eval": {
                "image": "npa-fixture-eval",
                "dockerfile": "Dockerfile",
                "physical_ai": {"useful": True, "role": "Fixture capability"},
                "safety": {
                    "runs_as": "worker",
                    "base_image": "fixture",
                    "network": "none",
                    "notes": "Hermetic test fixture",
                },
                "golden_eval": {
                    "kind": "container-smoke",
                    "command": "fixture-capability --verify",
                    "gpu": "none",
                    "timeout_seconds": timeout,
                    "status": "ready",
                },
            }
        },
    }
    monkeypatch.setattr(manifest, "_manifest_text", lambda: yaml.safe_dump(payload))
    # Do not replace the package-wide cached catalog used by other test modules.
    return manifest.load_manifest.__wrapped__()["fixture-eval"]


def test_explicit_unlimited_parses_and_validates(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _spec(monkeypatch, "unlimited")
    assert spec.golden_eval.timeout_seconds == math.inf
    assert spec.golden_eval.timeout_seconds > 0
    assert spec.golden_eval.execution_timeout is None
    monkeypatch.setattr(manifest, "load_manifest", lambda: {spec.name: spec})
    assert manifest.validate_manifest(check_paths=False, check_modules=False).ok


@pytest.mark.parametrize("value", [45, 45.0, 45.8, "45"])
def test_finite_timeout_keeps_existing_integer_conversion(
    monkeypatch: pytest.MonkeyPatch, value: object
) -> None:
    spec = _spec(monkeypatch, value)
    assert spec.golden_eval.timeout_seconds == 45
    assert spec.golden_eval.execution_timeout == 45


@pytest.mark.parametrize(
    "value",
    [math.nan, math.inf, -math.inf, "NaN", "Infinity", "inf", "Unlimited", "unlimited ", "none", None, True, False, [], {}],
)
def test_implicit_or_invalid_unlimited_values_are_rejected(
    monkeypatch: pytest.MonkeyPatch, value: object
) -> None:
    with pytest.raises(ValueError, match="golden_eval.timeout_seconds"):
        _spec(monkeypatch, value)


@pytest.mark.parametrize("value", [0, -1])
def test_nonpositive_numeric_timeout_still_fails_validation(
    monkeypatch: pytest.MonkeyPatch, value: int
) -> None:
    spec = _spec(monkeypatch, value)
    monkeypatch.setattr(manifest, "load_manifest", lambda: {spec.name: spec})
    report = manifest.validate_manifest(check_paths=False, check_modules=False)
    assert not report.ok
    assert any("timeout_seconds must be > 0" in issue.message for issue in report.issues)


def test_validator_rejects_nan_in_programmatically_supplied_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(monkeypatch, 45)
    spec = replace(spec, golden_eval=replace(spec.golden_eval, timeout_seconds=math.nan))
    monkeypatch.setattr(manifest, "load_manifest", lambda: {spec.name: spec})
    assert not manifest.validate_manifest(check_paths=False, check_modules=False).ok


@pytest.mark.parametrize("value, expected", [("unlimited", "unlimited"), (45, 45)])
def test_cli_show_emits_standard_json_timeout(
    monkeypatch: pytest.MonkeyPatch, value: object, expected: object
) -> None:
    spec = _spec(monkeypatch, value)
    monkeypatch.setattr(cli, "container", lambda _name: spec)
    result = CliRunner().invoke(app, ["workbench", "golden-eval", "show", spec.name])
    assert result.exit_code == 0, result.output

    def reject_nonstandard_constant(value: str) -> None:
        raise AssertionError(f"Nonstandard JSON constant: {value}")

    payload = json.loads(result.output, parse_constant=reject_nonstandard_constant)
    assert payload["golden_eval"]["timeout_seconds"] == expected


@pytest.mark.parametrize("value, expected", [("unlimited", None), (45, 45)])
def test_cli_local_execution_passes_exact_deadline(
    monkeypatch: pytest.MonkeyPatch, value: object, expected: int | None
) -> None:
    spec = _spec(monkeypatch, value)
    monkeypatch.setattr(cli, "container", lambda _name: spec)
    run = Mock(return_value=subprocess.CompletedProcess(["fixture-capability"], 0))
    monkeypatch.setattr(cli.subprocess, "run", run)
    result = CliRunner().invoke(app, ["workbench", "golden-eval", "run", spec.name, "--execute"])
    assert result.exit_code == 0, result.output
    run.assert_called_once_with(["fixture-capability", "--verify"], timeout=expected, check=False)


@pytest.mark.parametrize("value, expected", [("unlimited", None), (45, 45)])
def test_batch_local_execution_passes_exact_deadline(
    monkeypatch: pytest.MonkeyPatch, value: object, expected: int | None
) -> None:
    spec = _spec(monkeypatch, value)
    monkeypatch.setattr(batch, "container", lambda _name: spec)
    run = Mock(return_value=subprocess.CompletedProcess(["fixture-capability"], 0))
    monkeypatch.setattr(batch.subprocess, "run", run)
    result = batch.run_container_eval(spec.name, execute=True)
    assert result.ok
    run.assert_called_once_with(["fixture-capability", "--verify"], timeout=expected, check=False)


def test_unlimited_cli_serverless_fails_before_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(monkeypatch, "unlimited")
    monkeypatch.setattr(cli, "container", lambda _name: spec)
    submit = Mock(side_effect=AssertionError("No cloud submission is authorized by this test"))
    monkeypatch.setattr(serverless_runner, "submit_golden_eval", submit)
    result = CliRunner().invoke(app, ["workbench", "golden-eval", "run", spec.name, "--serverless"])
    assert result.exit_code == 1
    assert "local --execute" in result.output
    assert "mk8s" in result.output
    submit.assert_not_called()


def test_unlimited_batch_serverless_fails_before_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(monkeypatch, "unlimited")
    monkeypatch.setattr(batch, "container", lambda _name: spec)
    submit = Mock(side_effect=AssertionError("No cloud submission is authorized by this test"))
    monkeypatch.setattr(serverless_runner, "submit_golden_eval", submit)
    result = batch.run_container_eval(spec.name, serverless=True)
    assert not result.ok and result.exit_code == 1
    assert result.detail["error"] == "UnlimitedServerlessUnsupported"
    assert "local --execute" in result.detail["message"]
    submit.assert_not_called()


def test_finite_serverless_evaluation_keeps_existing_explicit_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(monkeypatch, 45)
    monkeypatch.setattr(batch, "container", lambda _name: spec)
    submit = Mock(return_value={"ok": True})
    monkeypatch.setattr(serverless_runner, "submit_golden_eval", submit)
    assert batch.run_container_eval(spec.name, serverless=True, timeout="9m").ok
    assert submit.call_args.kwargs["timeout"] == "9m"


def test_direct_unlimited_serverless_call_refuses_before_config_or_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(monkeypatch, "unlimited")
    monkeypatch.setattr(serverless_runner, "container", lambda _name: spec)
    forbidden_calls = []
    for name in ("_project_id", "resolve_golden_image", "load_credentials", "ServerlessClient"):
        forbidden = Mock(side_effect=AssertionError(f"Unexpected access through {name}"))
        monkeypatch.setattr(serverless_runner, name, forbidden)
        forbidden_calls.append(forbidden)
    with pytest.raises(RuntimeError, match="local --execute"):
        serverless_runner.submit_golden_eval(spec.name)
    for forbidden in forbidden_calls:
        forbidden.assert_not_called()


@pytest.mark.parametrize("value, expected", [("unlimited", None), (45, 45)])
def test_script_local_execution_passes_exact_deadline(
    monkeypatch: pytest.MonkeyPatch, value: object, expected: int | None
) -> None:
    spec = _spec(monkeypatch, value)
    script_path = Path(__file__).resolve().parents[2] / "scripts/run_golden_evals.py"
    module_spec = importlib.util.spec_from_file_location("golden_eval_timeout_script", script_path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    monkeypatch.setattr(module, "container", lambda _name: spec)
    run = Mock(return_value=subprocess.CompletedProcess(["fixture-capability"], 0))
    monkeypatch.setattr(module.subprocess, "run", run)
    args = argparse.Namespace(container=spec.name, execute=True, serverless=False)
    assert module._cmd_run(args) == 0
    run.assert_called_once_with(["fixture-capability", "--verify"], timeout=expected, check=False)
