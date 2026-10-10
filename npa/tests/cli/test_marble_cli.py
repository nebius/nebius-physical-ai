"""Check native Marble registration and forwarding through the shared implementation."""

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.workbench.marble import runtime


@pytest.mark.parametrize(
    "verb",
    [
        "acquire",
        "capture",
        "scan",
        "report",
        "pallet-preflight",
        "pallet-benchmark",
        "pallet-report",
        "navigation-prepare",
        "quadruped-collect",
    ],
)
def test_marble_command_registered(verb):
    result = CliRunner().invoke(app, ["workbench", "marble", verb, "--help"])
    assert result.exit_code == 0, result.output
    assert "--output-path" in result.output


def test_capture_forwards_request(monkeypatch):
    observed = []
    monkeypatch.setattr(
        runtime,
        "capture",
        lambda request: observed.append(request) or {"frames": request.frames},
    )
    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "marble",
            "capture",
            "--input-path",
            "s3://example-bucket/world/",
            "--output-path",
            "s3://example-bucket/result/",
            "--run-id",
            "demo",
            "--frames",
            "17",
        ],
    )
    assert result.exit_code == 0, result.output
    assert observed[0].frames == 17


def test_manufacturing_submit_requires_token_without_secret_flag(monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    observed = []

    def credentials(**kwargs):
        observed.extend(kwargs["requested"])
        return SimpleNamespace(
            endpoint_url="https://storage.example",
            secret_values={},
            missing=("WLT_API_KEY",),
        )

    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.submit_credentials.resolve_submit_credentials",
        credentials,
    )
    monkeypatch.setattr(
        "npa.orchestration.skypilot.workflow.submit_workflow",
        lambda *a, **k: pytest.fail("submitted without key"),
    )
    path = (
        Path(__file__).resolve().parents[3]
        / "workflows/testing/marble-manufacturing-pallet-detection.yaml"
    )
    result = CliRunner().invoke(
        app, ["workbench", "workflow", "submit", str(path), "--project", "unit-project"]
    )
    assert result.exit_code == 1, result.output
    assert "WLT_API_KEY" in observed
    assert "Required secret values" in result.output
