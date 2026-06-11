"""Completeness and consistency tests for the container golden-eval manifest.

These run in the standard (infra-free) unit suite and act as the nightly CI gate:
they guarantee every Workbench container has a valid golden-eval / hello-world
definition and that each definition points at real Dockerfiles and entrypoints.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.utils import strip_ansi
from typer.testing import CliRunner

from npa.cli.main import app
from npa.deploy.images import CONTAINER_IMAGE_NAMES
from npa.smoke.manifest import (
    VALID_GPU,
    VALID_KINDS,
    VALID_STATUS,
    load_manifest,
    validate_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
runner = CliRunner()


def test_manifest_loads_and_is_valid() -> None:
    report = validate_manifest(expected_tools=set(CONTAINER_IMAGE_NAMES))
    assert report.ok, "\n".join(str(issue) for issue in report.issues)


def test_every_container_image_has_an_entry() -> None:
    specs = load_manifest()
    missing = set(CONTAINER_IMAGE_NAMES) - set(specs)
    assert not missing, f"containers missing golden-eval entries: {sorted(missing)}"


def test_entries_map_to_known_images_or_foundation() -> None:
    specs = load_manifest()
    known_images = set(CONTAINER_IMAGE_NAMES.values())
    for name, spec in specs.items():
        if spec.foundation:
            continue
        assert name in CONTAINER_IMAGE_NAMES, f"{name} is not a known tool"
        assert spec.image in known_images, f"{name} image {spec.image} is unknown"


@pytest.mark.parametrize("name", sorted(load_manifest()))
def test_dockerfile_exists(name: str) -> None:
    spec = load_manifest()[name]
    if spec.external_build:
        pytest.skip(f"{name} is built outside this repo")
    assert (REPO_ROOT / spec.dockerfile).is_file(), spec.dockerfile


@pytest.mark.parametrize("name", sorted(load_manifest()))
def test_golden_eval_fields_are_well_formed(name: str) -> None:
    ge = load_manifest()[name].golden_eval
    assert ge.kind in VALID_KINDS
    assert ge.gpu in VALID_GPU
    assert ge.status in VALID_STATUS
    assert ge.command.strip()
    assert ge.timeout_seconds > 0


@pytest.mark.parametrize("name", sorted(load_manifest()))
def test_safety_and_physical_ai_documented(name: str) -> None:
    spec = load_manifest()[name]
    assert spec.physical_ai.get("role"), f"{name} missing physical_ai.role"
    assert "useful" in spec.physical_ai, f"{name} missing physical_ai.useful"
    for field in ("runs_as", "base_image", "network", "notes"):
        assert spec.safety.get(field), f"{name} missing safety.{field}"


def test_serverless_gpu_values_are_known() -> None:
    known = {"h200", "h100", "l40s", "b300", "rtx6000", "b200"}
    for name, spec in load_manifest().items():
        gpu = spec.golden_eval.serverless_gpu
        if gpu is not None:
            assert gpu in known, f"{name}: unknown serverless_gpu {gpu!r}"


def test_serverless_runner_imports() -> None:
    # Import-safe: pulls in no GPU/framework deps.
    from npa.smoke import serverless_runner

    assert hasattr(serverless_runner, "submit_golden_eval")


def test_referenced_smoke_modules_import() -> None:
    """Every module-backed golden eval points at an importable module."""

    from importlib.util import find_spec

    for name, spec in load_manifest().items():
        for module in (spec.golden_eval.module, spec.golden_eval.env_module):
            if module:
                assert find_spec(module) is not None, f"{name}: {module} not importable"


def test_cli_validate_succeeds() -> None:
    result = runner.invoke(app, ["workbench", "golden-eval", "validate"])
    assert result.exit_code == 0, strip_ansi(result.output)


def test_cli_list_shows_all_containers() -> None:
    result = runner.invoke(app, ["workbench", "golden-eval", "list"])
    assert result.exit_code == 0, strip_ansi(result.output)
    output = strip_ansi(result.output)
    for name in load_manifest():
        assert name in output


def test_cli_show_emits_record() -> None:
    result = runner.invoke(app, ["workbench", "golden-eval", "show", "lerobot"])
    assert result.exit_code == 0, strip_ansi(result.output)
    assert "npa-lerobot" in strip_ansi(result.output)


def test_cli_run_dry_run_prints_command() -> None:
    result = runner.invoke(app, ["workbench", "golden-eval", "run", "fiftyone"])
    assert result.exit_code == 0, strip_ansi(result.output)
    assert "test_fiftyone_functional" in strip_ansi(result.output)


def test_cli_run_rejects_unknown_container() -> None:
    result = runner.invoke(app, ["workbench", "golden-eval", "run", "nope"])
    assert result.exit_code != 0
