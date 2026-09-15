"""Exercise the installed Fleet graphics CLI and shared SDK implementation."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner
import yaml

from npa.cli.fleet.graphics import verify_graphics_cmd
from npa.cli.main import app
from npa.fleet import graphics_verification
from npa.lifecycle_intent import OperationIntent, current_intent
from npa.sdk import fleet as fleet_sdk

runner = CliRunner()


@pytest.fixture
def spec_path(tmp_path):
    path = tmp_path / "fleet.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "npa.fleet/v0.0.1",
                "name": "graphics-cli-test",
                "projects": [
                    {
                        "name": "team",
                        "clusters": [
                            {
                                "name": "render",
                                "gpu_workload_profile": "rtx-rendering",
                                "gpu_nodes": {
                                    "count": 2,
                                    "platform": "gpu-rtx6000-a",
                                    "preset": "8gpu-192vcpu-1744gb",
                                },
                            }
                        ],
                    }
                ],
            }
        )
    )
    return path


def _report(passed=True):
    return {
        "passed": passed,
        "selected_clusters": 1,
        "verified_clusters": int(passed),
        "gpu_workers": 2 if passed else 0,
        "gpus": 16 if passed else 0,
        "cuda_workers": 2 if passed else 0,
        "glx_workers": 2 if passed else 0,
        "egl_workers": 2 if passed else 0,
        "vulkan_workers": 2 if passed else 0,
        "clusters": [],
        "failures": [],
        "evidence_sha256": "a" * 64,
    }


def test_json_has_one_document_and_mutation_boundary(spec_path, monkeypatch):
    observed = []

    def verify(spec, **kwargs):
        observed.append(current_intent())
        return _report()

    monkeypatch.setattr(graphics_verification, "verify_graphics", verify)
    result = runner.invoke(
        app,
        [
            "fleet",
            "verify-graphics",
            "--spec",
            str(spec_path),
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0
    assert json.loads(result.stdout) == _report()
    assert observed == [OperationIntent.MUTATE]


def test_cli_passes_scope_and_runtime_options(spec_path, monkeypatch, tmp_path):
    observed = []
    monkeypatch.setattr(
        graphics_verification,
        "verify_graphics",
        lambda spec, **kwargs: observed.append(kwargs) or _report(),
    )
    result = runner.invoke(
        app,
        [
            "fleet",
            "verify-graphics",
            "--spec",
            str(spec_path),
            "--only-projects",
            "team,other",
            "--only-clusters",
            "render,inference",
            "--project-prefix",
            "custom-",
            "--profile",
            "operator-test",
            "--evidence-dir",
            str(tmp_path),
            "--concurrency",
            "3",
            "--stabilization-seconds",
            "9",
            "--timeout-minutes",
            "12",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0
    assert observed == [
        {
            "only_projects": ["team", "other"],
            "only_clusters": ["render", "inference"],
            "project_prefix": "custom-",
            "profile": "operator-test",
            "evidence_dir": tmp_path,
            "concurrency": 3,
            "stabilization_seconds": 9,
            "timeout_minutes": 12,
        }
    ]


def test_failed_or_unavailable_verification_exits_safely(spec_path, monkeypatch):
    monkeypatch.setattr(
        graphics_verification, "verify_graphics", lambda *args, **kwargs: _report(False)
    )
    failed = runner.invoke(
        app,
        [
            "fleet",
            "verify-graphics",
            "--spec",
            str(spec_path),
            "--output",
            "json",
        ],
    )
    assert failed.exit_code == 1
    assert json.loads(failed.stdout)["passed"] is False

    def unavailable(*args, **kwargs):
        raise RuntimeError("private-provider-detail")

    monkeypatch.setattr(graphics_verification, "verify_graphics", unavailable)
    unavailable_result = runner.invoke(
        app,
        [
            "fleet",
            "verify-graphics",
            "--spec",
            str(spec_path),
            "--output",
            "json",
        ],
    )
    assert unavailable_result.exit_code == 1
    assert "private-provider-detail" not in unavailable_result.output


def test_direct_call_help_and_sdk_use_shared_implementation(
    spec_path, monkeypatch, capsys
):
    observed = []
    monkeypatch.setattr(
        graphics_verification,
        "verify_graphics",
        lambda spec, **kwargs: observed.append(kwargs) or _report(),
    )
    verify_graphics_cmd(spec_path=spec_path)
    assert "verification passed" in capsys.readouterr().out
    assert observed[0]["concurrency"] == 1
    help_result = runner.invoke(app, ["fleet", "verify-graphics", "--help"])
    assert help_result.exit_code == 0
    assert "--concurrency" in help_result.stdout
    assert fleet_sdk.verify_graphics("spec", concurrency=2) == _report()
    assert observed[-1]["concurrency"] == 2
    assert "verify_graphics" in fleet_sdk.__all__
