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


def _copy_directives(dockerfile_text: str) -> tuple[list[str], list[str]]:
    """Return (sources, destinations) across all COPY lines in a Dockerfile."""

    sources: list[str] = []
    dests: list[str] = []
    for line in dockerfile_text.splitlines():
        stripped = line.strip()
        if not stripped.upper().startswith("COPY "):
            continue
        tokens = [t for t in stripped.split()[1:] if not t.startswith("--")]
        if len(tokens) < 2:
            continue
        sources.extend(tokens[:-1])
        dests.append(tokens[-1])
    return sources, dests


# Golden evals that execute an in-image smoke (module or script). Build-import,
# workflow-smoke (entrypoint comes from a base image), and entrypoint-smoke kinds
# are provisioned differently and are not covered by this static contract.
_IN_IMAGE_SMOKE_KINDS = {"container-smoke", "server-smoke"}


@pytest.mark.parametrize(
    "name",
    sorted(
        n
        for n, s in load_manifest().items()
        if s.golden_eval.kind in _IN_IMAGE_SMOKE_KINDS and not s.external_build
    ),
)
def test_dockerfile_provides_golden_eval_entrypoint(name: str) -> None:
    """Each in-image smoke must actually be built into its image.

    This is the regression guard for the packaging bugs the live serverless run
    surfaced (npa.smoke not bundled in lancedb/detection-training; isaac-lab/
    fiftyone using a module command for an image that only ships a script).
    """

    spec = load_manifest()[name]
    text = (REPO_ROOT / spec.dockerfile).read_text(encoding="utf-8")
    sources, dests = _copy_directives(text)
    command = spec.golden_eval.command

    if command.startswith("python -m npa.smoke."):
        module = command.split("python -m ", 1)[1].split()[0]
        module_file = "src/" + module.replace(".", "/") + ".py"
        provides = any(
            src == "src/npa" or src.startswith("src/npa/smoke") or src == module_file
            for src in sources
        )
        assert provides, (
            f"{name}: {spec.dockerfile} runs `{command}` but does not COPY the "
            f"npa.smoke package (need src/npa, src/npa/smoke, or {module_file})"
        )
    elif command.startswith("python /"):
        script_path = command.split("python ", 1)[1].split()[0]
        assert script_path in dests, (
            f"{name}: {spec.dockerfile} runs `{command}` but no COPY writes "
            f"{script_path} into the image"
        )
    else:  # pragma: no cover - guards against an unhandled command shape.
        raise AssertionError(f"{name}: unexpected in-image smoke command: {command!r}")


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


def test_versions_helper_works_without_toml_library(tmp_path: Path) -> None:
    """genesis runs Python 3.10 without tomllib/tomli; the helper must still work.

    Regression for the genesis golden-eval failure where npa.smoke._versions
    raised ModuleNotFoundError importing tomllib/tomli.
    """

    import importlib

    from npa.smoke import _versions

    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[tool.npa.supported-tools]\n"
        'genesis = "0.4.6"\n'
        'lerobot = "0.5.1"\n',
        encoding="utf-8",
    )
    start = tmp_path / "pkg" / "smoke.py"
    start.parent.mkdir()
    start.write_text("", encoding="utf-8")

    # Force the stdlib-only path (as on a py3.10 image without tomli).
    saved = _versions._tomllib
    try:
        _versions._tomllib = None
        assert _versions.supported_tool_version("genesis", str(start)) == "0.4.6"
        assert _versions.supported_tool_version("lerobot", str(start)) == "0.5.1"
    finally:
        _versions._tomllib = saved
        importlib.reload(_versions)


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
    result = runner.invoke(app, ["workbench", "golden-eval", "run", "lerobot"])
    assert result.exit_code == 0, strip_ansi(result.output)
    assert "test_lerobot_functional" in strip_ansi(result.output)


def test_cli_run_rejects_unknown_container() -> None:
    result = runner.invoke(app, ["workbench", "golden-eval", "run", "nope"])
    assert result.exit_code != 0
