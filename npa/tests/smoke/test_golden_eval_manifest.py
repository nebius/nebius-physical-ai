"""Completeness and consistency tests for the container golden-eval manifest.

These run in the standard (infra-free) unit suite and act as the nightly CI gate:
they guarantee every Workbench container has a valid golden-eval / hello-world
definition and that each definition points at real Dockerfiles and entrypoints.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import json
import re
import shlex
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
        if spec.foundation or spec.internal:
            continue
        if spec.variant_of is not None:
            # A build variant of another tool (e.g. sonic-mujoco is built FROM npa-sonic
            # and resolves through sonic's image manifest). It is separately built and
            # published with its own capability, so it carries its own golden eval, but
            # it is deliberately not a CONTAINER_IMAGE_NAMES key.
            assert spec.variant_of in CONTAINER_IMAGE_NAMES, (
                f"{name}: variant_of={spec.variant_of!r} is not a known tool"
            )
            assert spec.image not in known_images, (
                f"{name}: {spec.image} is a canonical tool image, so it should be a tool "
                f"entry rather than a variant"
            )
            continue
        assert name in CONTAINER_IMAGE_NAMES, f"{name} is not a known tool"
        assert spec.image in known_images, f"{name} image {spec.image} is unknown"


def test_every_separately_built_image_has_a_golden_eval() -> None:
    """The packaging contract lists every image we build and publish; each needs a
    tested rerun, whether it is a canonical tool or a build variant of one.

    sonic-mujoco is why this exists: it is built and published separately, but because
    it is a sonic variant it slipped through the CONTAINER_IMAGE_NAMES-based
    completeness check and had no golden eval at all.
    """
    import yaml

    contract_path = (
        REPO_ROOT / "npa" / "docker" / "workbench" / "packaging-contract.yaml"
    )
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    # Compare by Dockerfile path, not by key: the two files legitimately use different
    # names for the same image (contract `sim2real-envgen` is tool `envgen`, contract
    # `sim2real-eval` is tool `loop-eval`), so a key-set diff reports false positives.
    covered = {
        str(Path(spec.dockerfile).relative_to("npa/docker/workbench"))
        for spec in load_manifest().values()
        if spec.dockerfile
    }
    missing = sorted(
        f"{name} ({entry['dockerfile']})"
        for name, entry in contract["images"].items()
        if entry["dockerfile"] not in covered
        and entry.get("redistribution") == "public"
    )
    assert not missing, (
        f"packaging-contract images with no golden-eval entry: {missing}"
    )


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
    elif command.startswith("/isaac-sim/python.sh /"):
        # An image whose ENTRYPOINT is bash cannot use `python <file>`: docker turns it
        # into `bash python <file>`, and bash then tries to execute the python BINARY as
        # a shell script ("cannot execute binary file"). Passing an interpreter SCRIPT
        # works, which is what /isaac-sim/python.sh is - the Isaac bootstrap shim.
        #
        # Must be tested BEFORE the "sh /" branch below: this path does not start with
        # "sh /" so the two cannot collide today, but ordering it first keeps that true if
        # the shim ever moves.
        script_path = command.split(None, 1)[1].split()[0]
        assert script_path in dests, (
            f"{name}: {spec.dockerfile} runs `{command}` but no COPY writes "
            f"{script_path} into the image"
        )
    elif command.startswith("sh /"):
        # POSIX-sh smokes (images without bash, e.g. the alpine-based static hosts).
        script_path = command.split("sh ", 1)[1].split()[0]
        assert script_path in dests or any(
            src.endswith(Path(script_path).name) for src in sources
        ), (
            f"{name}: {spec.dockerfile} runs `{command}` but no COPY writes "
            f"{script_path} into the image"
        )
    elif command.startswith("bash "):
        script_path = command.split("bash ", 1)[1].split()[0]
        assert (
            any(
                src.endswith(Path(script_path).name) or script_path in dest
                for src in sources
                for dest in dests
            )
            or script_path in dests
        ), (
            f"{name}: {spec.dockerfile} runs `{command}` but no COPY writes "
            f"{script_path} into the image"
        )
    else:  # pragma: no cover - guards against an unhandled command shape.
        if name != "leisaac":
            raise AssertionError(
                f"{name}: unexpected in-image smoke command: {command!r}"
            )
        assert command == "golden-smoke"


@pytest.mark.parametrize(
    "name",
    sorted(
        n
        for n, s in load_manifest().items()
        if s.golden_eval.kind in _IN_IMAGE_SMOKE_KINDS and not s.external_build
    ),
)
def test_golden_eval_composes_with_dockerfile_entrypoint(name: str) -> None:
    """Model Kubernetes args semantics instead of testing the command alone."""

    spec = load_manifest()[name]
    dockerfile = (REPO_ROOT / spec.dockerfile).read_text(encoding="utf-8")
    matches = re.findall(r"^ENTRYPOINT\s+(\[[^\n]+\])", dockerfile, re.MULTILINE)
    # No declared entrypoint composes as an empty argv prefix; the supplied
    # container command is then the executable. Images with an entrypoint must
    # prove the appended argument vector remains meaningful.
    entrypoint = json.loads(matches[-1]) if matches else []
    composed = [*entrypoint, *shlex.split(spec.golden_eval.command)]
    assert composed[: len(entrypoint)] == entrypoint
    if name == "leisaac":
        assert composed == [
            "/opt/npa/sim/venv/bin/python",
            "/opt/npa/leisaac/session_server.py",
            "golden-smoke",
        ]
        assert 'sys.argv[1:] == ["golden-smoke"]' in (
            REPO_ROOT / "npa/docker/workbench/leisaac/session_server.py"
        ).read_text(encoding="utf-8")


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
        '[tool.npa.supported-tools]\ngenesis = "0.4.6"\nlerobot = "0.5.1"\n',
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


# Isaac Lab and SONIC render through RTX ray-tracing cores. H100 and H200 have none: they
# are throughput parts. A render path scheduled there does not fail cleanly -- imports pass
# and the job burns its budget before producing wrong or empty output, which is the worst
# kind of failure to debug. See skills/atomic/gpu-selection/SKILL.md.
RT_CORE_GPUS = {"l40s", "rtx6000"}
RT_CORE_REQUIRED = {"isaac-lab", "sonic"}


@pytest.mark.parametrize("name", sorted(RT_CORE_REQUIRED))
def test_rt_core_tools_are_not_scheduled_on_throughput_gpus(name: str) -> None:
    spec = load_manifest()[name]
    gpu = spec.golden_eval.serverless_gpu
    assert gpu in RT_CORE_GPUS, (
        f"{name} renders through RT cores, but its golden eval requests {gpu!r}, which has "
        f"none. Use one of {sorted(RT_CORE_GPUS)}. (A live submission also showed the "
        f"serverless project offers gpu-rtx6000/gpu-b200-sxm/cpu-d3 and no h100 at all, so "
        f"an h100 request fails outright with NotFound.)"
    )


def test_every_manifest_entry_resolves_to_a_real_image() -> None:
    """A golden eval you cannot resolve an image for is not a golden eval.

    This exists because adding sonic-mujoco's entry passed every other check and then died
    at submission time with `KeyError: 'sonic-mujoco'` — a variant is deliberately not a
    CONTAINER_IMAGE_NAMES key, so it has to resolve through its parent tool's image
    manifest instead. Nothing checked that the entry pointed at something resolvable.
    """
    from npa.deploy.images import container_image_for_tool
    from npa.smoke.serverless_runner import resolve_golden_image

    for name, spec in load_manifest().items():
        if spec.foundation:
            continue  # foundation images are not resolved through CONTAINER_IMAGE_NAMES
        if spec.internal:
            ref = resolve_golden_image(name, registry="registry.example/test")
        elif spec.variant_of:
            ref = container_image_for_tool(
                spec.variant_of,
                registry="registry.example/test",
                image_variant=spec.image_variant,
            )
        else:
            ref = container_image_for_tool(name, registry="registry.example/test")
        assert ref.rsplit("/", 1)[-1].startswith(spec.image + ":"), (
            f"{name}: manifest declares image {spec.image!r} but resolves to {ref!r}"
        )
