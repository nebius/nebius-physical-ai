from __future__ import annotations

import json
from pathlib import Path
import subprocess

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from npa.cli.main import app
from npa.sdk.workbench.flex_pi import infer
from npa.workbench.flex_pi.runtime import (
    ARTIFACT_SCHEMA,
    DEFAULT_CHECKPOINT_REVISION,
    FlexPiError,
    FlexPiRequest,
    run_inference,
)
from npa.workbench.flex_pi.service import create_app


def _manifest(path: Path) -> Path:
    path.write_text(json.dumps({
        "schema": "npa.flex_pi.public_observation.v1",
        "dataset": {"id": "flex-pi/robotwin_3d", "revision": "bee164afe94041d8c3d7dd1203725c41163fc3f4"},
        "observation": {"episode": 0, "rgb": {"a": {}, "b": {}, "c": {}}},
    }), encoding="utf-8")
    return path


def _runner(argv, **kwargs):  # type: ignore[no-untyped-def]
    output = Path(argv[argv.index("--output-json") + 1])
    output.write_text(json.dumps({
        "schema": "npa.flex_pi.actions.v1", "regime": "action-only",
        "actions": [[float(index) for index in range(14)] for _ in range(32)],
        "metrics": {"inference_seconds": 0.25, "action_l2_mean": 1.0, "peak_gpu_memory_bytes": 1},
        "runtime": {"cuda": True, "gpu_name": "NVIDIA RTX PRO 6000 Blackwell"},
    }), encoding="utf-8")
    return subprocess.CompletedProcess(argv, 0, stdout="passed")


def test_real_contract_executes_and_publishes(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "input.json")
    output = tmp_path / "output"
    result = run_inference(FlexPiRequest(
        input_path=str(manifest), output_path=str(output), expected_gpu="RTXPRO6000",
    ), runner=_runner)
    assert result["status"] == "ok"
    assert result["schema"] == ARTIFACT_SCHEMA
    assert result["checkpoint"]["revision"] == DEFAULT_CHECKPOINT_REVISION
    assert set(result["artifacts"]) == {"actions.json", "input.json", "result.json"}
    assert len(json.loads((output / "actions.json").read_text())["actions"]) == 32


def test_invalid_action_shape_is_not_published(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "input.json")
    def broken(argv, **kwargs):  # type: ignore[no-untyped-def]
        Path(argv[argv.index("--output-json") + 1]).write_text(json.dumps({
            "regime": "action-only", "actions": [[0.0]],
            "metrics": {"inference_seconds": 1.0},
            "runtime": {"cuda": True, "gpu_name": "RTXPRO6000"},
        }))
        return subprocess.CompletedProcess(argv, 0)
    output = tmp_path / "output"
    try:
        run_inference(FlexPiRequest(input_path=str(manifest), output_path=str(output)), runner=broken)
    except FlexPiError as exc:
        assert "32x14" in str(exc)
    else:
        raise AssertionError("invalid action tensor was accepted")
    assert not output.exists()


def test_failed_upstream_output_is_bounded_and_redacted(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "input.json")

    def broken(argv, **kwargs):  # type: ignore[no-untyped-def]
        return subprocess.CompletedProcess(
            argv,
            1,
            stdout=(
                "prefix " * 2_000
                + "Bearer private-value https://example.invalid/object?signature=private "
                + "hf_abcdefghijklmnopqrstuvwxyz"
            ),
        )

    try:
        run_inference(
            FlexPiRequest(input_path=str(manifest), output_path=str(tmp_path / "output")),
            runner=broken,
        )
    except FlexPiError as exc:
        message = str(exc)
        assert "<uri-ref>" in message
        assert "<redacted>" in message
        assert "private-value" not in message
        assert "example.invalid" not in message
        assert "hf_abcdefghijklmnopqrstuvwxyz" not in message
        assert len(message) < 8_400
    else:
        raise AssertionError("failed upstream execution was accepted")


def test_cli_sdk_and_service_share_dry_run(tmp_path: Path, monkeypatch) -> None:
    manifest = _manifest(tmp_path / "input.json")
    monkeypatch.setattr(
        "npa.cli.workbench.flex_pi.run_inference",
        lambda request: {"schema": ARTIFACT_SCHEMA, "status": "dry_run"},
    )
    cli = CliRunner().invoke(app, [
        "workbench", "flex-pi", "infer", "--input-path", "s3://fixture/input.json",
        "--output-path", "s3://fixture/output/", "--dry-run",
    ])
    assert cli.exit_code == 0, cli.output
    assert json.loads(cli.output)["schema"] == ARTIFACT_SCHEMA
    assert infer(input_path=str(manifest), output_path=str(tmp_path), dry_run=True)["status"] == "dry_run"
    service = create_app(token="test-token", output_root=str(tmp_path / "service"), input_manifest=str(manifest))
    with TestClient(service) as client:
        response = client.post("/run", json={"output_path": "sample", "dry_run": True}, headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 200
    assert response.json()["schema"] == ARTIFACT_SCHEMA


def test_service_requires_authentication(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "input.json")
    service = create_app(token="test-token", output_root=str(tmp_path / "service"), input_manifest=str(manifest))
    with TestClient(service) as client:
        assert client.get("/health").status_code == 401
        assert client.get("/health", headers={"Authorization": "Bearer test-token"}).status_code == 200
    paths = {route.path for route in service.routes}
    assert {"/health", "/run", "/status", "/system-info", "/list"} <= paths
