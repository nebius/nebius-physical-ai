"""Guard the failure boundaries identified by independent SeedVR2 review."""

from __future__ import annotations

import json

from botocore.exceptions import ClientError
from fastapi.testclient import TestClient
import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.smoke.capabilities import GOLDEN_EVAL_CAPABILITIES
from npa.workbench.seedvr2 import artifacts, hardware, runtime
from npa.workbench.seedvr2.service import create_app


def test_golden_capability_keeps_target_and_capture_limits():
    claims = " ".join(GOLDEN_EVAL_CAPABILITIES["seedvr2"])
    assert "targets one B200" in claims
    assert "execution remains gated" in claims
    assert "physical-capture provenance is unverified" in claims
    assert "real Aloha" not in claims


def test_probe_failure_retains_owned_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "_validate_artifact_request", lambda *a, **k: None)
    monkeypatch.setattr(artifacts, "_create_work_directory", lambda _: tmp_path)

    class FailedDownload:
        def download_file(self, uri, destination):
            from pathlib import Path

            Path(destination).write_bytes(b"partial input")
            raise OSError("interrupted input")

    request = artifacts.VideoArtifactRequest(
        input_path="s3://example-bucket/input.mp4",
        output_path="s3://example-bucket/probe.json",
        run_id="retained",
    )
    with pytest.raises(OSError, match="interrupted input"):
        artifacts.probe(request, storage_factory=FailedDownload)
    assert (tmp_path / "input.mp4").read_bytes() == b"partial input"


@pytest.mark.parametrize(
    "flag", ["missing_exact_sass", "unexpected_sass", "unexpected_ptx"]
)
def test_b200_rejects_each_producer_architecture_failure(tmp_path, monkeypatch, flag):
    monkeypatch.setattr(hardware, "ARCHES_ROOT", tmp_path)
    entry = {"sass": ["sm_90", "sm_100"], "ptx": [], flag: ["sm_120"]}
    for name in ("flash-attn", "apex"):
        (tmp_path / f"{name}.json").write_text(json.dumps({"extension.so": entry}))
    with pytest.raises(ValueError, match="B200"):
        hardware.require_b200_build_inventory()


@pytest.mark.parametrize("verb", ["probe", "restore", "verify", "review"])
def test_invalid_cli_request_is_reported_before_operation(monkeypatch, verb):
    module = runtime if verb == "restore" else artifacts
    monkeypatch.setattr(module, verb, lambda _: pytest.fail("operation must not run"))
    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "seedvr2",
            verb,
            "--input-path",
            "s3://example-bucket/input.mp4",
            "--output-path",
            "s3://example-bucket/output/",
            "--run-id",
            "",
        ],
    )
    assert result.exit_code == 1
    assert "SeedVR2 failed" in result.stderr
    assert json.loads(result.stdout)["result"] == "error"


def test_invalid_restore_dimensions_use_cli_error_contract(monkeypatch):
    monkeypatch.setattr(runtime, "restore", lambda _: pytest.fail("must not run"))
    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "seedvr2",
            "restore",
            "--input-path",
            "s3://example-bucket/input.mp4",
            "--output-path",
            "s3://example-bucket/output/",
            "--run-id",
            "invalid",
            "--output-height",
            "100",
        ],
    )
    assert result.exit_code == 1
    assert "SeedVR2 failed" in result.stderr
    assert json.loads(result.stdout)["result"] == "error"


class _DeniedOutputStorage:
    def read_bytes_with_etag(self, uri):
        raise ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "denied"}}, "GetObject"
        )


def test_output_existence_denial_keeps_cause_and_fails_closed():
    with pytest.raises(runtime.SeedVR2Error, match="output read authority") as error:
        runtime._ensure_artifacts_absent(
            _DeniedOutputStorage(), ["s3://example-bucket/out"]
        )
    assert isinstance(error.value.__cause__, ClientError)


@pytest.mark.parametrize("verb", ["probe", "restore", "verify", "review"])
def test_output_denial_maps_to_service_error_and_releases_lock(monkeypatch, verb):
    def denied(request):
        runtime._ensure_artifacts_absent(_DeniedOutputStorage(), [request.output_path])

    module = runtime if verb == "restore" else artifacts
    monkeypatch.setattr(module, verb, denied)
    client = TestClient(
        create_app(token="test-token", allowed_s3_roots=["s3://example-bucket/"])
    )
    headers = {"Authorization": "Bearer test-token"}
    response = client.post(
        f"/{verb}",
        headers=headers,
        json={
            "input_path": "s3://example-bucket/input.mp4",
            "output_path": "s3://example-bucket/output/",
            "run_id": "denied",
        },
    )
    assert response.status_code == 400
    assert "output read authority" in response.json()["detail"]
    assert client.get("/status", headers=headers).json() == {"busy": False}
