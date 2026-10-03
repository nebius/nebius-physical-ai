"""Explicit unlimited evaluations must never inherit an execution deadline."""

from __future__ import annotations

from dataclasses import replace
import argparse
import importlib.util
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace
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


def test_explicit_unlimited_parses_and_validates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    [
        math.nan,
        math.inf,
        -math.inf,
        "NaN",
        "Infinity",
        "inf",
        "Unlimited",
        "unlimited ",
        "none",
        None,
        True,
        False,
        [],
        {},
    ],
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
    assert any(
        "timeout_seconds must be > 0" in issue.message for issue in report.issues
    )


def test_validator_rejects_nan_in_programmatically_supplied_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(monkeypatch, 45)
    spec = replace(
        spec, golden_eval=replace(spec.golden_eval, timeout_seconds=math.nan)
    )
    monkeypatch.setattr(manifest, "load_manifest", lambda: {spec.name: spec})
    assert not manifest.validate_manifest(check_paths=False, check_modules=False).ok


@pytest.mark.parametrize("value", [True, False, 0, -1, "8", 8.0, None])
def test_validator_rejects_invalid_serverless_gpu_counts(
    monkeypatch: pytest.MonkeyPatch, value: object
) -> None:
    """Counts reach a cloud API, so bools/coercible strings must fail closed."""

    spec = _spec(monkeypatch, 45)
    spec = replace(
        spec, golden_eval=replace(spec.golden_eval, serverless_gpu_count=value)
    )
    monkeypatch.setattr(manifest, "load_manifest", lambda: {spec.name: spec})
    report = manifest.validate_manifest(check_paths=False, check_modules=False)
    assert not report.ok
    assert any("serverless_gpu_count" in issue.message for issue in report.issues)


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
    result = CliRunner().invoke(
        app, ["workbench", "golden-eval", "run", spec.name, "--execute"]
    )
    assert result.exit_code == 0, result.output
    run.assert_called_once_with(
        ["fixture-capability", "--verify"], timeout=expected, check=False
    )


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
    run.assert_called_once_with(
        ["fixture-capability", "--verify"], timeout=expected, check=False
    )


def test_unlimited_cli_serverless_fails_before_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(monkeypatch, "unlimited")
    monkeypatch.setattr(cli, "container", lambda _name: spec)
    submit = Mock(
        side_effect=AssertionError("No cloud submission is authorized by this test")
    )
    monkeypatch.setattr(serverless_runner, "submit_golden_eval", submit)
    result = CliRunner().invoke(
        app, ["workbench", "golden-eval", "run", spec.name, "--serverless"]
    )
    assert result.exit_code == 1
    assert "local --execute" in result.output
    assert "mk8s" in result.output
    submit.assert_not_called()


def test_cli_serverless_forwards_candidate_image_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(monkeypatch, 45)
    monkeypatch.setattr(cli, "container", lambda _name: spec)
    submit = Mock(return_value={"ok": True, "status": "COMPLETED"})
    monkeypatch.setattr(serverless_runner, "submit_golden_eval", submit)

    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "golden-eval",
            "run",
            spec.name,
            "--serverless",
            "--registry",
            "registry.example.invalid/team",
            "--tag",
            "candidate-123",
        ],
    )

    assert result.exit_code == 0, result.output
    assert submit.call_args.kwargs["registry"] == "registry.example.invalid/team"
    assert submit.call_args.kwargs["tag"] == "candidate-123"


def test_cli_rejects_candidate_override_without_serverless() -> None:
    result = CliRunner().invoke(
        app,
        ["workbench", "golden-eval", "run", "lerobot", "--tag", "candidate-123"],
    )
    assert result.exit_code == 2
    assert "require --serverless" in result.output


def test_unlimited_batch_serverless_fails_before_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(monkeypatch, "unlimited")
    monkeypatch.setattr(batch, "container", lambda _name: spec)
    submit = Mock(
        side_effect=AssertionError("No cloud submission is authorized by this test")
    )
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
    for name in (
        "_project_id",
        "resolve_golden_image",
        "load_credentials",
        "ServerlessClient",
    ):
        forbidden = Mock(
            side_effect=AssertionError(f"Unexpected access through {name}")
        )
        monkeypatch.setattr(serverless_runner, name, forbidden)
        forbidden_calls.append(forbidden)
    with pytest.raises(RuntimeError, match="local --execute"):
        serverless_runner.submit_golden_eval(spec.name)
    for forbidden in forbidden_calls:
        forbidden.assert_not_called()


@pytest.mark.parametrize("gpu_override", [None, "h200"])
def test_serverless_submission_honors_manifest_gpu_count(
    monkeypatch: pytest.MonkeyPatch,
    gpu_override: str | None,
) -> None:
    spec = _spec(monkeypatch, 45)
    spec = replace(
        spec,
        golden_eval=replace(
            spec.golden_eval, serverless_gpu="b200", serverless_gpu_count=8
        ),
    )
    monkeypatch.setattr(serverless_runner, "container", lambda _name: spec)
    monkeypatch.setattr(serverless_runner, "_project_id", lambda _value: "project-test")
    monkeypatch.setattr(
        serverless_runner, "resolve_golden_image", lambda *_a, **_k: "example/image:tag"
    )
    monkeypatch.setattr(
        serverless_runner,
        "load_credentials",
        lambda **_kwargs: SimpleNamespace(
            s3_bucket="s3://bucket",
            s3_access_key_id="access",
            s3_secret_access_key="secret",
            s3_endpoint="https://storage.invalid",
            hf_token="",
        ),
    )
    seen: dict[str, object] = {}

    def resolve(gpu: str, count: int) -> tuple[str, str, int]:
        seen.update(gpu=gpu, count=count)
        raise RuntimeError("stop after platform resolution")

    monkeypatch.setattr(serverless_runner, "resolve_gpu_platform", resolve)
    with pytest.raises(RuntimeError, match="stop after platform resolution"):
        serverless_runner.submit_golden_eval(spec.name, gpu_type=gpu_override)
    assert seen == {"gpu": gpu_override or "b200", "count": 8}


@pytest.mark.parametrize("value, expected", [("unlimited", None), (45, 45)])
def test_script_local_execution_passes_exact_deadline(
    monkeypatch: pytest.MonkeyPatch, value: object, expected: int | None
) -> None:
    spec = _spec(monkeypatch, value)
    script_path = Path(__file__).resolve().parents[2] / "scripts/run_golden_evals.py"
    module_spec = importlib.util.spec_from_file_location(
        "golden_eval_timeout_script", script_path
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    monkeypatch.setattr(module, "container", lambda _name: spec)
    run = Mock(return_value=subprocess.CompletedProcess(["fixture-capability"], 0))
    monkeypatch.setattr(module.subprocess, "run", run)
    args = argparse.Namespace(container=spec.name, execute=True, serverless=False)
    assert module._cmd_run(args) == 0
    run.assert_called_once_with(
        ["fixture-capability", "--verify"], timeout=expected, check=False
    )


def test_script_rejects_candidate_override_without_serverless(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script_path = Path(__file__).resolve().parents[2] / "scripts/run_golden_evals.py"
    module_spec = importlib.util.spec_from_file_location(
        "golden_eval_candidate_script", script_path
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    run = Mock(side_effect=AssertionError("candidate override must not run locally"))
    monkeypatch.setattr(module.subprocess, "run", run)
    args = argparse.Namespace(
        container="lerobot",
        execute=True,
        serverless=False,
        registry=None,
        tag="candidate-123",
    )

    assert module._cmd_run(args) == 2
    assert "require --serverless" in capsys.readouterr().err
    run.assert_not_called()
