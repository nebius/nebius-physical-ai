"""Exercise exact image requirements through the production planning CLI."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from npa.cli.workbench.workflow import app

ROOT = Path(__file__).resolve().parents[3]
ISAAC = "registry.example.invalid/npa-isaac-lab@sha256:" + "a" * 64
CPU = "registry.example.invalid/npa-sonic@sha256:" + "b" * 64


@pytest.mark.parametrize(
    "name",
    [
        "field-failure-reference-demo",
        "rgbd-scan-to-policy-demo",
        "rgbd-scan-to-isaac",
        "scan-to-isaac-navigation",
    ],
)
def test_plan_cli_reports_missing_exact_images_as_clean_error(name):
    path = ROOT / f"workflows/testing/{name}.yaml"
    result = CliRunner().invoke(app, ["plan-spec", str(path), "--check-render"])
    assert result.exit_code == 1
    assert "Error:" in result.output and "requires an explicit" in result.output
    assert "--var" in result.output and "Traceback" not in result.output


def test_plan_cli_uses_operator_vars_before_real_render(monkeypatch):
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/source-fixture")
    path = ROOT / "workflows/testing/rgbd-scan-to-policy-demo.yaml"
    result = CliRunner().invoke(
        app,
        [
            "plan-spec",
            str(path),
            "--check-render",
            "--json",
            "--var",
            "bucket=example-output",
            "--var",
            f"isaac_image={ISAAC}",
            "--var",
            f"assembly_image={CPU}",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["render_check"]["tasks"] == 8
    assert ISAAC in result.stdout and CPU in result.stdout
