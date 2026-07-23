"""CLI + module tests for the Lichtblick (Foxglove-compatible OSS) workbench viewer.

Infra-free: no Docker, S3, or network calls. Image resolution is pinned via
NPA_REGISTRY so the tests never touch the real registry.
"""

from __future__ import annotations

import json

import pytest
import yaml
from pathlib import Path

from typer.testing import CliRunner

from npa.cli.main import app
from npa.deploy.images import CONTAINER_IMAGE_NAMES, SUPPORTED_TOOL_VERSIONS
from npa.workbench.lichtblick import (
    DEFAULT_PORT,
    LichtblickError,
    LichtblickLaunchPlan,
    build_launch_plan,
    launch_viewer,
)

runner = CliRunner()

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE_PATH = REPO_ROOT / "npa" / "docker" / "workbench" / "lichtblick" / "Dockerfile"
PACKAGING_CONTRACT = REPO_ROOT / "npa" / "docker" / "workbench" / "packaging-contract.yaml"


@pytest.fixture(autouse=True)
def _pin_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPA_REGISTRY", "cr.example/reg")


def test_lichtblick_is_registered_everywhere() -> None:
    assert CONTAINER_IMAGE_NAMES["lichtblick"] == "npa-lichtblick"
    assert SUPPORTED_TOOL_VERSIONS["lichtblick"] == "1.26.0"
    # The old, incorrectly-named key must not exist.
    assert "foxglove" not in CONTAINER_IMAGE_NAMES


def test_dockerfile_exists_and_is_non_root_service() -> None:
    text = DOCKERFILE_PATH.read_text(encoding="utf-8")
    assert 'LABEL npa.tool="lichtblick"' in text
    assert "EXPOSE 8080" in text
    assert "USER nobody" in text
    assert "HEALTHCHECK" in text
    # Digest-pinned bases (fiftyone header convention).
    assert "node:22-bookworm@sha256:" in text
    assert "caddy:2.11.4-alpine@sha256:" in text
    # Caddy's XDG dirs must be writable by the non-root runtime user.
    assert "chown -R 65534:65534" in text


def test_packaging_contract_entry() -> None:
    contract = yaml.safe_load(PACKAGING_CONTRACT.read_text(encoding="utf-8"))
    assert "foxglove" not in contract["images"]
    entry = contract["images"]["lichtblick"]
    assert entry["tier"] == "service"
    assert entry["ports"] == [8080]
    assert entry["final_user"] == "nobody"


def test_lichtblick_registered_under_workbench() -> None:
    result = runner.invoke(app, ["workbench", "--help"])
    assert result.exit_code == 0
    assert "lichtblick" in result.output


def test_lichtblick_command_help() -> None:
    for command in ("serve", "launch", "status", "list"):
        result = runner.invoke(app, ["workbench", "lichtblick", command, "--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.output


def test_serve_plans_viewer_for_s3_mcap() -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "lichtblick",
            "serve",
            "--input-path",
            "s3://bucket/run42/recording.mcap",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "planned"
    assert payload["artifact_name"] == "recording.mcap"
    assert payload["image"] == "cr.example/reg/npa-lichtblick:1.26.0"
    assert payload["port"] == DEFAULT_PORT
    assert payload["served_artifact_path"] == "/srv/data/recording.mcap"
    assert "ds=remote-file" in payload["viewer_url"]


def test_serve_rejects_unsupported_artifact() -> None:
    result = runner.invoke(
        app,
        ["workbench", "lichtblick", "serve", "--input-path", "s3://bucket/run/notes.txt"],
    )
    assert result.exit_code == 1
    assert "unsupported artifact" in result.output.lower()


def test_serve_rejects_non_s3_scheme() -> None:
    result = runner.invoke(
        app,
        ["workbench", "lichtblick", "serve", "--input-path", "gs://bucket/x.mcap"],
    )
    assert result.exit_code == 1


def test_build_launch_plan_requires_input() -> None:
    with pytest.raises(LichtblickError):
        build_launch_plan(input_path="")


def test_build_launch_plan_local_path() -> None:
    plan = build_launch_plan(input_path="/data/local.mcap", image="npa-lichtblick:test", port=9099)
    assert isinstance(plan, LichtblickLaunchPlan)
    assert plan.artifact_name == "local.mcap"
    assert plan.image == "npa-lichtblick:test"
    assert plan.port == 9099


def test_launch_viewer_uses_injected_runner() -> None:
    plan = build_launch_plan(input_path="s3://b/k/x.mcap", image="npa-lichtblick:test")
    captured: list[list[str]] = []
    result = launch_viewer(plan, local_artifact="/tmp/x.mcap", runner=captured.append)
    assert result.status == "launched"
    assert captured, "runner was not invoked"
    argv = captured[0]
    assert argv[0] == "docker"
    assert "/tmp/x.mcap:/srv/data/x.mcap:ro" in argv
    assert "npa-lichtblick:test" in argv


def test_launch_viewer_requires_local_artifact() -> None:
    plan = build_launch_plan(input_path="s3://b/k/x.mcap", image="npa-lichtblick:test")
    with pytest.raises(LichtblickError):
        launch_viewer(plan, local_artifact="", runner=lambda argv: None)
