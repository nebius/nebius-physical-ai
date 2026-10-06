"""Keep existing Sim2Real artifact operations independent of image acceptance."""

from __future__ import annotations

from pathlib import Path
import os
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.workflows.rerun_serve import RerunServeConfig
from npa.workflows.sim2real.artifact_config import (
    Sim2RealArtifactConfig,
    build_artifact_config_from_env,
)
from npa.workflows.sim2real.config import build_config_from_env
from npa.workflows.sim2real_rerun_regen import RegenResult, download_rrd_from_s3


def _reject_execution_images(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject(*_args: object, **_kwargs: object) -> str:
        raise ValueError("public execution image remains quarantined")

    monkeypatch.setattr(
        "npa.workflows.sim2real.models.container_image_for_tool", reject
    )


def _invoke_rerun(command: str, *arguments: str):
    return CliRunner().invoke(
        app,
        [
            "workbench",
            "sim2real",
            "rerun",
            command,
            "--run-id",
            "archived-run",
            *arguments,
        ],
    )


def test_regen_reaches_artifacts_while_execution_images_are_quarantined(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reject_execution_images(monkeypatch)
    seen: list[Sim2RealArtifactConfig] = []

    def regenerate(config: Sim2RealArtifactConfig, **kwargs: object) -> RegenResult:
        seen.append(config)
        assert kwargs["sync_inputs"] is False
        assert kwargs["upload"] is False
        return RegenResult(config.run_id, str(tmp_path), "review.rrd", "", 4, 1, 4)

    monkeypatch.setattr("npa.cli.workbench.sim2real.regen_sim2real_rrd", regenerate)
    result = _invoke_rerun(
        "regen",
        "--s3-bucket",
        "example-bucket",
        "--local-dir",
        str(tmp_path),
        "--no-sync",
        "--no-upload",
        "--output",
        "json",
    )
    assert result.exit_code == 0, result.output
    assert seen[0].s3_bucket == "example-bucket"
    assert isinstance(seen[0], Sim2RealArtifactConfig)
    assert "image" not in " ".join(vars(seen[0]))
    with pytest.raises(ValueError, match="remains quarantined"):
        build_config_from_env(run_id="execution-run")


def test_public_execution_defaults_remain_quarantined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in tuple(os.environ):
        if name.endswith("_IMAGE") or name == "NPA_SIM2REAL_REGISTRY":
            monkeypatch.delenv(name)
    with pytest.raises(
        ValueError, match="'reference-policy' has no consumable public release"
    ):
        build_config_from_env(run_id="execution-run")


def test_serve_local_record_uses_resolved_viewer_storage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reject_execution_images(monkeypatch)
    config = RerunServeConfig(
        run_id="archived-run",
        s3_bucket="resolved-bucket",
        s3_prefix="completed",
        s3_endpoint="https://storage.example.invalid",
        rrd_s3_uri_override="s3://resolved-bucket/completed/custom.rrd",
    )
    _mock_viewer_deploy(monkeypatch, config)
    downloads: list[tuple[Sim2RealArtifactConfig, dict[str, object]]] = []

    def download(coordinates: Sim2RealArtifactConfig, **kwargs: object) -> Path:
        downloads.append((coordinates, kwargs))
        return tmp_path / "local.rrd"

    monkeypatch.setattr("npa.cli.workbench.sim2real.download_rrd_from_s3", download)
    result = _invoke_rerun(
        "serve",
        "--project",
        "operator-project",
        "--local-record",
        "--local-rrd-path",
        str(tmp_path / "local.rrd"),
        "--output",
        "json",
    )
    assert result.exit_code == 0, result.output
    assert downloads[0][0].s3_bucket == config.s3_bucket
    assert downloads[0][0].s3_endpoint == config.s3_endpoint
    assert downloads[0][1]["rrd_uri"] == config.rrd_s3_uri


def _mock_viewer_deploy(
    monkeypatch: pytest.MonkeyPatch,
    config: RerunServeConfig,
) -> None:
    monkeypatch.setattr(
        "npa.cli.workbench.sim2real._rerun_serve_credentials", lambda: ("ak", "sk")
    )
    monkeypatch.setattr(
        "npa.cli.workbench.sim2real.build_rerun_serve_config", lambda **_: config
    )
    monkeypatch.setattr(
        "npa.cli.workbench.sim2real.require_kubeconfig", lambda **_: "fake-kubeconfig"
    )
    monkeypatch.setattr(
        "npa.cli.workbench.sim2real.apply_rerun_serve",
        lambda *_args, **_kwargs: SimpleNamespace(
            to_dict=lambda: {"status": "deployed", "run_id": config.run_id}
        ),
    )


@pytest.mark.parametrize("explicit", [False, True])
def test_artifact_config_preserves_environment_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
    explicit: bool,
) -> None:
    _reject_execution_images(monkeypatch)
    monkeypatch.setenv("NPA_SIM2REAL_BUCKET", "environment-bucket")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://environment.example.invalid")
    monkeypatch.setenv("OUTER_ITERATIONS", "7")
    monkeypatch.setenv("NPA_SIM2REAL_K8S_GPU_CANDIDATES", "RTXPRO6000; L40S")
    config = build_artifact_config_from_env(
        run_id="archived-run",
        s3_prefix="completed",
        s3_bucket="explicit-bucket" if explicit else "",
        s3_endpoint="https://explicit.example.invalid" if explicit else "",
    )
    assert config.s3_bucket == ("explicit-bucket" if explicit else "environment-bucket")
    assert config.s3_endpoint == (
        "https://explicit.example.invalid"
        if explicit
        else "https://environment.example.invalid"
    )
    assert config.outer_iterations == 7
    assert config.k8s_gpu_candidates == ("RTXPRO6000", "L40S")


def test_recording_download_uses_exact_selected_uri(tmp_path: Path) -> None:
    selected = "s3://example-bucket/archived/custom.rrd"
    downloads: list[str] = []

    def download(uri: str, destination: str) -> None:
        downloads.append(uri)
        Path(destination).write_bytes(b"existing-recording")

    config = Sim2RealArtifactConfig(run_id="archived-run", s3_bucket="example-bucket")
    path = download_rrd_from_s3(
        config,
        dest_path=tmp_path / "review.rrd",
        rrd_uri=selected,
        client=SimpleNamespace(download_path=download),
    )
    assert downloads == [selected]
    assert path.read_bytes() == b"existing-recording"
