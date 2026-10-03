"""Require source staging when an imaged workflow explicitly selects an overlay."""

import json

import pytest
from typer.testing import CliRunner
import yaml

from npa.cli.main import app


def _write_spec(tmp_path, overlay):
    resource = {
        "cloud": "kubernetes",
        "cpus": 1,
        "memory": "1Gi",
        "image": "registry.example.invalid/tool@sha256:" + "a" * 64,
    }
    state = {
        "resources": "cpu",
        "run": {"shell": "python3 -m npa.workflows.behavior_challenge --help"},
        "terminal": True,
    }
    spec = {
        "apiVersion": "npa.workflow/v0.0.1",
        "kind": "Workflow",
        "metadata": {"name": "overlay-test"},
        "config": {"bucket": "unit-output", "source_overlay": overlay},
        "resources": {"cpu": resource},
        "initial": "execute",
        "states": {"execute": state},
    }
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(spec))
    return path


@pytest.mark.parametrize("selection", ["spec", "override", "environment"])
def test_pinned_image_overlay_plans_source_before_rendering(
    tmp_path, monkeypatch, mocker, selection
):
    monkeypatch.delenv("NPA_SRC_S3_URI", raising=False)
    monkeypatch.delenv("NPA_E2E_NPA_SRC_S3_URI", raising=False)
    monkeypatch.delenv("NPA_SRC_OVERLAY", raising=False)
    stage = mocker.patch("npa.orchestration.npa_workflow.src_staging.stage_npa_source")
    path = _write_spec(tmp_path, selection == "spec")
    args = [
        "workbench",
        "workflow",
        "submit",
        str(path),
        "--plan-only",
        "--output-format",
        "json",
    ]
    if selection == "override":
        args.extend(["--var", "source_overlay=true"])
    if selection == "environment":
        monkeypatch.setenv("NPA_SRC_OVERLAY", "1")

    result = CliRunner().invoke(app, args)

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["source"]["status"] == "planned"
    task = next(
        d
        for d in yaml.safe_load_all(payload["skypilot_yaml"])
        if d and "resources" in d
    )
    assert task["envs"]["NPA_SRC_OVERLAY"] == "1"
    assert task["envs"]["NPA_SRC_S3_URI"].startswith("s3://unit-output/npa-src/npa/")
    stage.assert_not_called()


def test_overlay_without_source_refuses_disabled_staging(tmp_path, monkeypatch):
    monkeypatch.delenv("NPA_SRC_S3_URI", raising=False)
    monkeypatch.delenv("NPA_E2E_NPA_SRC_S3_URI", raising=False)
    path = _write_spec(tmp_path, True)
    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(path),
            "--plan-only",
            "--no-stage-src",
        ],
    )
    assert result.exit_code == 1
    assert "NPA_SRC_S3_URI is unset" in result.output
