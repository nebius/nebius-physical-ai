"""Real command-tree cuRobo registration and thin-client invocation."""

import json

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.workbench.curobo import runtime


@pytest.mark.parametrize(
    "verb", ["prepare", "benchmark", "plan", "validate", "visualize"]
)
def test_help_and_shared_operation(verb, monkeypatch):
    runner = CliRunner()
    result = runner.invoke(app, ["workbench", "curobo", verb, "--help"])
    assert result.exit_code == 0
    assert "--output-path" in result.stdout
    seen = []
    monkeypatch.setattr(
        runtime, verb, lambda request: seen.append(request) or {"ok": True}
    )
    args = ["workbench", "curobo", verb, "--output-path", "s3://example-bucket/output"]
    if verb != "prepare":
        args += ["--input-path", "s3://example-bucket/input", "--run-id", "test"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"ok": True}
    assert len(seen) == 1
