from __future__ import annotations

import json
from pathlib import Path
import subprocess

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from npa.cli.main import app
from npa.sdk.workbench.alpamayo2_super import infer
from npa.workbench.alpamayo2_super.runtime import (
    DEFAULT_DATASET_REVISION,
    DEFAULT_MODEL_REVISION,
    ARTIFACT_SCHEMA,
    Alpamayo2SuperRequest,
    run_inference,
)
from npa.workbench.alpamayo2_super.service import app as service_app, create_app


def test_dry_run_is_revision_pinned_and_real_upstream_argv(tmp_path: Path) -> None:
    result = run_inference(
        Alpamayo2SuperRequest(output_path=str(tmp_path), dry_run=True)
    )
    assert result["status"] == "dry_run"
    assert result["schema"] == ARTIFACT_SCHEMA
    assert result["model"]["revision"] == DEFAULT_MODEL_REVISION
    assert result["dataset"]["revision"] == DEFAULT_DATASET_REVISION
    assert result["argv"][1:3] == ["-m", "alpamayo2_super.inference_smoke"]
    assert "--require-camera-projection" in result["argv"]


def test_execution_requires_real_json_and_png_then_publishes(tmp_path: Path) -> None:
    def fake_runner(argv, **kwargs):  # type: ignore[no-untyped-def]
        Path(argv[argv.index("--save-viz") + 1]).write_bytes(b"png")
        Path(argv[argv.index("--save-json") + 1]).write_text(
            json.dumps({"minADE": 1.0}), encoding="utf-8"
        )
        return subprocess.CompletedProcess(argv, 0, stdout="minADE: 1.0")

    result = run_inference(
        Alpamayo2SuperRequest(output_path=str(tmp_path)),
        runner=fake_runner,
        model_resolver=lambda request: "/runtime/model/snapshot",
    )
    assert result["status"] == "ok"
    assert set(result["artifacts"]) == {
        "result.json",
        "trajectory.json",
        "trajectory.png",
    }
    assert (tmp_path / "trajectory.png").read_bytes() == b"png"
    provenance = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert provenance["runtime"]["weights_baked"] is False
    assert provenance["runtime"]["dataset_baked"] is False


def test_cli_and_api_share_dry_run_contract(tmp_path: Path) -> None:
    cli = CliRunner().invoke(
        app,
        [
            "workbench",
            "alpamayo2-super",
            "infer",
            "--output-path",
            str(tmp_path),
            "--dry-run",
        ],
    )
    assert cli.exit_code == 0, cli.output
    assert json.loads(cli.output)["schema"] == ARTIFACT_SCHEMA

    response = TestClient(service_app).post(
        "/run", json={"output_path": str(tmp_path), "dry_run": True}
    )
    assert response.status_code == 200
    assert response.json()["schema"] == ARTIFACT_SCHEMA

    sdk = infer(output_path=str(tmp_path), dry_run=True)
    assert sdk["schema"] == ARTIFACT_SCHEMA


def test_service_factory_exposes_standard_endpoints() -> None:
    paths = {route.path for route in create_app().routes}
    assert {"/health", "/run", "/status", "/system-info", "/list"} <= paths


def test_terms_keep_model_and_dataset_licenses_separate() -> None:
    result = CliRunner().invoke(app, ["workbench", "alpamayo2-super", "terms"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["model"]["license"] == "OpenMDW-1.1"
    assert payload["dataset"]["redistribution"] is False
