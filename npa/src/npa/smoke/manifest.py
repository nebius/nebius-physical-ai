"""Loader and validator for the container golden-eval manifest.

The manifest (`golden_evals.yaml`, packaged alongside this module) is the single
source of truth for each Workbench container's safety posture, Physical AI
usefulness, and its "golden eval" / "hello world" tested rerun.

This module is import-safe: it pulls in no GPU or infrastructure dependencies, so
it is usable from unit tests, the CLI, and the nightly CI driver alike.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from importlib.util import find_spec
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - PyYAML is a hard dep.
    raise RuntimeError("PyYAML is required to read the golden-eval manifest") from exc

MANIFEST_RESOURCE = "golden_evals.yaml"
MANIFEST_FORMAT = "npa_golden_evals_v1"

VALID_KINDS = {
    "container-smoke",
    "server-smoke",
    "entrypoint-smoke",
    "workflow-smoke",
    "build-import",
}
VALID_GPU = {"required", "optional", "none"}
VALID_STATUS = {"ready", "gpu-gated", "blocked-on-upstream"}


@dataclass(frozen=True)
class GoldenEval:
    """The minimal tested rerun that proves a container works."""

    kind: str
    command: str
    gpu: str
    timeout_seconds: int
    status: str
    module: str | None = None
    env_module: str | None = None
    artifact: str | None = None
    serverless_gpu: str | None = None

    @property
    def runnable_in_ci(self) -> bool:
        """True when the eval needs no GPU and is not blocked upstream."""

        return self.gpu == "none" and self.status == "ready"


@dataclass(frozen=True)
class ContainerSpec:
    """A single container entry in the golden-eval manifest."""

    name: str
    image: str
    dockerfile: str
    physical_ai: dict[str, Any]
    safety: dict[str, Any]
    golden_eval: GoldenEval
    foundation: bool = False
    external_build: bool = False


@dataclass
class ValidationIssue:
    container: str
    message: str

    def __str__(self) -> str:
        return f"{self.container}: {self.message}"


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def add(self, container: str, message: str) -> None:
        self.issues.append(ValidationIssue(container, message))


def _manifest_text() -> str:
    return (
        resources.files(__package__)
        .joinpath(MANIFEST_RESOURCE)
        .read_text(encoding="utf-8")
    )


@lru_cache(maxsize=1)
def load_manifest() -> dict[str, ContainerSpec]:
    """Parse the packaged manifest into ``ContainerSpec`` objects."""

    payload = yaml.safe_load(_manifest_text()) or {}
    if payload.get("format") != MANIFEST_FORMAT:
        raise RuntimeError(
            f"Unsupported golden-eval manifest format: {payload.get('format')!r}"
        )
    containers = payload.get("containers")
    if not isinstance(containers, dict):
        raise RuntimeError("Manifest must define a 'containers' mapping")

    specs: dict[str, ContainerSpec] = {}
    for name, raw in containers.items():
        if not isinstance(raw, dict):
            raise RuntimeError(f"Container {name!r} entry must be a mapping")
        eval_raw = raw.get("golden_eval")
        if not isinstance(eval_raw, dict):
            raise RuntimeError(f"Container {name!r} is missing a golden_eval block")
        golden_eval = GoldenEval(
            kind=str(eval_raw.get("kind", "")),
            command=str(eval_raw.get("command", "")),
            gpu=str(eval_raw.get("gpu", "")),
            timeout_seconds=int(eval_raw.get("timeout_seconds", 0)),
            status=str(eval_raw.get("status", "")),
            module=eval_raw.get("module"),
            env_module=eval_raw.get("env_module"),
            artifact=eval_raw.get("artifact"),
            serverless_gpu=eval_raw.get("serverless_gpu"),
        )
        specs[name] = ContainerSpec(
            name=name,
            image=str(raw.get("image", "")),
            dockerfile=str(raw.get("dockerfile", "")),
            physical_ai=dict(raw.get("physical_ai") or {}),
            safety=dict(raw.get("safety") or {}),
            golden_eval=golden_eval,
            foundation=bool(raw.get("foundation", False)),
            external_build=bool(raw.get("external_build", False)),
        )
    return specs


def _repo_root() -> Path:
    # src/npa/smoke/manifest.py -> repo root is four parents up.
    return Path(__file__).resolve().parents[4]


def _module_exists(module: str | None) -> bool:
    if not module:
        return True
    try:
        return find_spec(module) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def validate_manifest(
    *,
    expected_tools: set[str] | None = None,
    check_paths: bool = True,
    check_modules: bool = True,
) -> ValidationReport:
    """Validate manifest completeness and internal consistency.

    ``expected_tools`` is the set of container tool keys that must be present
    (typically ``CONTAINER_IMAGE_NAMES`` keys). The manifest may carry extra
    foundation entries (e.g. ``base-cuda13-b300``) that are not tools.
    """

    report = ValidationReport()
    specs = load_manifest()
    repo_root = _repo_root()

    for name, spec in specs.items():
        ge = spec.golden_eval
        if not spec.image:
            report.add(name, "missing image name")
        if spec.external_build:
            # Image is built outside this repo; no in-tree Dockerfile to check.
            pass
        elif not spec.dockerfile:
            report.add(name, "missing dockerfile path")
        elif check_paths and not (repo_root / spec.dockerfile).is_file():
            report.add(name, f"dockerfile not found: {spec.dockerfile}")

        if not spec.physical_ai.get("role"):
            report.add(name, "physical_ai.role is empty")
        if "useful" not in spec.physical_ai:
            report.add(name, "physical_ai.useful is missing")
        for safety_field in ("runs_as", "base_image", "network", "notes"):
            if not spec.safety.get(safety_field):
                report.add(name, f"safety.{safety_field} is empty")

        if ge.kind not in VALID_KINDS:
            report.add(name, f"invalid golden_eval.kind: {ge.kind!r}")
        if ge.gpu not in VALID_GPU:
            report.add(name, f"invalid golden_eval.gpu: {ge.gpu!r}")
        if ge.status not in VALID_STATUS:
            report.add(name, f"invalid golden_eval.status: {ge.status!r}")
        if not ge.command:
            report.add(name, "golden_eval.command is empty")
        if ge.timeout_seconds <= 0:
            report.add(name, "golden_eval.timeout_seconds must be > 0")

        if check_modules:
            if not _module_exists(ge.module):
                report.add(name, f"golden_eval.module not importable: {ge.module}")
            if not _module_exists(ge.env_module):
                report.add(
                    name, f"golden_eval.env_module not importable: {ge.env_module}"
                )

    if expected_tools is not None:
        missing = expected_tools - set(specs)
        for tool in sorted(missing):
            report.add(tool, "container has no golden-eval manifest entry")

    return report


def container(name: str) -> ContainerSpec:
    try:
        return load_manifest()[name]
    except KeyError as exc:
        choices = ", ".join(sorted(load_manifest()))
        raise KeyError(f"Unknown container {name!r}; choose one of: {choices}") from exc
