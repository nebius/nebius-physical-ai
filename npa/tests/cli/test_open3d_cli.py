"""Real command-tree Open3D registration and thin-client invocation."""

import json

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.sdk.workbench import open3d as sdk
from npa.workbench.open3d import runtime
from npa.workbench.open3d.artifacts import Open3dError

RUN_VERBS = ("register", "multiway", "validate", "reconstruct", "visualize")
PREFIX_VERBS = ("stage-demo", "prepare")


def _args(verb: str) -> list[str]:
    args = ["workbench", "open3d", verb, "--output-path", "s3://example-bucket/output"]
    if verb == "prepare":
        args += ["--input-path", "s3://example-bucket/scans"]
    elif verb != "stage-demo":
        args += ["--input-path", "s3://example-bucket/input", "--run-id", "test"]
    return args


@pytest.mark.parametrize("verb", RUN_VERBS + PREFIX_VERBS)
def test_help_and_shared_operation(verb, monkeypatch):
    runner = CliRunner()
    result = runner.invoke(app, ["workbench", "open3d", verb, "--help"])
    assert result.exit_code == 0
    assert "--output-path" in result.stdout

    seen = []
    monkeypatch.setattr(
        runtime,
        verb.replace("-", "_"),
        lambda request: seen.append(request) or {"ok": True},
    )
    result = runner.invoke(app, _args(verb))
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"ok": True}
    assert len(seen) == 1


@pytest.mark.parametrize("verb", RUN_VERBS + PREFIX_VERBS)
def test_failures_exit_nonzero_without_a_traceback(verb, monkeypatch):
    """A stage failure must be a diagnosable message, not a Python traceback."""

    def explode(request):
        raise Open3dError("fragments do not overlap at this voxel size")

    monkeypatch.setattr(runtime, verb.replace("-", "_"), explode)
    result = CliRunner().invoke(app, _args(verb))
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "fragments do not overlap" in result.output


def test_out_of_contract_options_are_rejected_before_any_work(monkeypatch):
    called = []
    monkeypatch.setattr(
        runtime, "reconstruct", lambda request: called.append(request) or {}
    )
    result = CliRunner().invoke(
        app,
        _args("reconstruct") + ["--poisson-depth", "99"],
    )
    assert result.exit_code == 1
    assert not called


def test_sdk_and_cli_drive_the_same_operations(monkeypatch):
    """The SDK is not a second implementation; it builds the same requests."""

    seen = {}
    for name in (
        "stage_demo",
        "prepare",
        "register",
        "multiway",
        "validate",
        "visualize",
    ):
        monkeypatch.setattr(
            runtime, name, lambda request, _n=name: seen.setdefault(_n, request) or {}
        )
    monkeypatch.setattr(
        runtime,
        "reconstruct",
        lambda request: seen.setdefault("reconstruct", request) or {},
    )

    sdk.stage_demo(output_path="s3://example-bucket/prepared", voxel_size=0.02)
    sdk.prepare(
        input_path="s3://example-bucket/scans",
        output_path="s3://example-bucket/prepared",
    )
    sdk.register(
        input_path="s3://example-bucket/prepared",
        output_path="s3://example-bucket/pairs",
        run_id="r",
    )
    sdk.multiway(
        input_path="s3://example-bucket/prepared",
        output_path="s3://example-bucket/graph",
        run_id="r",
    )
    sdk.validate(
        input_path="s3://example-bucket/pairs",
        output_path="s3://example-bucket/validation",
        run_id="r",
    )
    sdk.reconstruct(
        input_path="s3://example-bucket/graph",
        output_path="s3://example-bucket/surface",
        run_id="r",
        poisson_depth=8,
    )
    sdk.visualize(
        input_path="s3://example-bucket/surface",
        output_path="s3://example-bucket/reports",
        run_id="r",
    )

    assert seen["stage_demo"].voxel_size == 0.02
    assert seen["prepare"].input_path == "s3://example-bucket/scans"
    assert seen["reconstruct"].poisson_depth == 8
    assert seen["visualize"].run_id == "r"
