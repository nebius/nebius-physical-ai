from __future__ import annotations

import json
from pathlib import Path
import subprocess
import pytest
from PIL import Image

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
    Alpamayo2SuperError,
    _resolve_model_snapshot,
)
from npa.workbench.alpamayo2_super.service import create_app


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


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"samples": [{"clip_id": "fixture-clip", "t0_us": 1}]}))
    return path


def _artifact_runner(argv, **kwargs):
    Image.new("RGB", (8, 8), "blue").save(argv[argv.index("--save-viz") + 1])
    Path(argv[argv.index("--save-json") + 1]).write_text(json.dumps({
        "clip_id": "fixture-clip", "t0_us": 1, "seed": 42,
        "projection_available": True, "min_ade_m": 1.0, "min_fde_m": 2.0,
        "fde_at_min_ade_m": 2.0,
    }))
    return subprocess.CompletedProcess(argv, 0, stdout="minADE: 1.0")


def test_execution_requires_real_json_and_png_then_publishes(tmp_path: Path, manifest) -> None:
    def fake_runner(argv, **kwargs):  # type: ignore[no-untyped-def]
        return _artifact_runner(argv, **kwargs)

    result = run_inference(
        Alpamayo2SuperRequest(output_path=str(tmp_path / "outputs"), manifest=str(manifest)),
        runner=fake_runner,
        model_resolver=lambda request: "/runtime/model/snapshot",
    )
    assert result["status"] == "ok"
    assert set(result["artifacts"]) == {
        "result.json",
        "trajectory.json",
        "trajectory.png",
    }
    with Image.open(tmp_path / "outputs/trajectory.png") as picture:
        assert picture.size == (8, 8)
    provenance = json.loads((tmp_path / "outputs/result.json").read_text(encoding="utf-8"))
    assert provenance["runtime"]["weights_baked"] is False
    assert provenance["runtime"]["dataset_baked"] is False
    assert provenance["sample"]["clip_id"] == "fixture-clip"
    assert provenance["metrics"]["min_ade_m"] == 1.0


@pytest.mark.parametrize("document,index", [
    (None, 0), ("not json", 0), ({"samples": []}, 0),
    ({"samples": [{"clip_id": "fixture", "t0_us": 1}]}, 1),
    ({"samples": [{"clip_id": "fixture", "t0_us": True}]}, 0),
])
def test_invalid_manifest_fails_before_model_fetch(tmp_path, document, index):
    path = tmp_path / "missing.json"
    if document is not None:
        path.write_text(document if isinstance(document, str) else json.dumps(document))
    def must_not_fetch(request):
        pytest.fail("invalid sample triggered model download")
    with pytest.raises(Alpamayo2SuperError, match="manifest/sample"):
        run_inference(Alpamayo2SuperRequest(
            output_path=str(tmp_path / "outputs"), manifest=str(path), sample_index=index,
        ), model_resolver=must_not_fetch)


def test_manifest_is_snapshotted_before_model_fetch(tmp_path, manifest):
    def resolver(request):
        manifest.unlink()
        return "/runtime/model/snapshot"
    def runner(argv, **kwargs):
        snapshot = Path(argv[argv.index("--manifest") + 1])
        assert json.loads(snapshot.read_text())["samples"][0]["clip_id"] == "fixture-clip"
        return _artifact_runner(argv, **kwargs)
    result = run_inference(Alpamayo2SuperRequest(
        output_path=str(tmp_path / "outputs"), manifest=str(manifest),
    ), runner=runner, model_resolver=resolver)
    assert set(result["artifacts"]) == {"result.json", "trajectory.json", "trajectory.png"}


@pytest.mark.parametrize("damage", ["png", "json", "sample", "projection", "missing_metric", "nan_metric", "negative_metric"])
def test_invalid_artifacts_are_not_published(tmp_path, manifest, damage):
    def runner(argv, **kwargs):
        result = _artifact_runner(argv, **kwargs)
        picture = Path(argv[argv.index("--save-viz") + 1])
        metadata = Path(argv[argv.index("--save-json") + 1])
        if damage == "png":
            picture.write_bytes(b"nonempty invalid png")
        elif damage == "json":
            metadata.write_text("[]")
        else:
            data = json.loads(metadata.read_text())
            if damage.endswith("_metric"):
                data["min_ade_m"] = {"missing_metric": None, "nan_metric": float("nan"), "negative_metric": -1}[damage]
            else:
                data["clip_id" if damage == "sample" else "projection_available"] = False
            metadata.write_text(json.dumps(data))
        return result
    output = tmp_path / "outputs"
    with pytest.raises(Alpamayo2SuperError, match="invalid required inference artifacts"):
        run_inference(Alpamayo2SuperRequest(output_path=str(output), manifest=str(manifest)),
                      runner=runner, model_resolver=lambda request: "/runtime/model/snapshot")
    assert not output.exists()


def test_empty_snapshot_resolver_output_is_a_domain_error(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout=""))
    with pytest.raises(Alpamayo2SuperError, match="local model snapshot"):
        _resolve_model_snapshot(Alpamayo2SuperRequest(output_path=str(tmp_path)))


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

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"samples": [{"clip_id": "sample", "t0_us": 1}]}))
    service = create_app(
        token="test-inference-credential", output_root=str(tmp_path / "service-outputs"),
        manifest=str(manifest),
    )
    with TestClient(service) as client:
        response = client.post(
            "/run", json={"output_path": "sample", "dry_run": True},
            headers={"Authorization": "Bearer test-inference-credential"},
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
