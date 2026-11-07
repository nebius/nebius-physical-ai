"""Fail-closed contracts for operator-supplied GR00T offline inputs.

The Workbench workflow deliberately treats custom code as a trusted, explicit
operator input.  A dataset must not be able to select Python code by putting a
module name in metadata.  This module verifies and extracts one
content-addressed source archive, installs the declared package without
resolving dependencies, and invokes one declared adapter callable with a JSON
request.  The adapter produces a native LeRobot v2 dataset plus a feedback
sidecar; it cannot silently replace the workflow's provenance contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
from shutil import copyfileobj
import subprocess
import sys
import tarfile
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InputProvenance(BaseModel):
    """Operator-declared identity and redistribution classification of bytes."""

    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["npa.groot-input-provenance/v1"] = (
        "npa.groot-input-provenance/v1"
    )
    source_kind: Literal["huggingface", "archive", "private-object-store"]
    source_identifier: str = Field(min_length=1)
    source_revision: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    license: str = Field(min_length=1)
    privacy_classification: Literal["public", "private", "restricted"]
    redistributable_media: bool
    label_provenance: str = Field(min_length=1)


class ActionContract(BaseModel):
    """The raw action coordinates an adapter promises to preserve."""

    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["npa.groot-action-contract/v1"] = (
        "npa.groot-action-contract/v1"
    )
    kind: Literal["joint_position", "cartesian_relative"]
    target_keys: list[str] = Field(min_length=1)
    frame: str = Field(min_length=1)
    translation_unit: str = Field(min_length=1)
    rotation_representation: str = Field(min_length=1)
    rotation_unit: str = Field(min_length=1)
    gripper_fields: list[str]
    transform_identity: str = Field(min_length=1)

    @model_validator(mode="after")
    def _distinct_target_keys(self) -> "ActionContract":
        if len(self.target_keys) != len(set(self.target_keys)):
            raise ValueError("action target_keys must be distinct")
        return self


class ExternalCodeSpec(BaseModel):
    """Verified archive and the exact callable the operator authorizes."""

    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["npa.groot-external-code/v1"] = "npa.groot-external-code/v1"
    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    package: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.]*$")
    adapter_callable: str = Field(
        pattern=r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$"
    )
    feedback_callable: str = Field(
        pattern=r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$"
    )


def verify_provenance_selection(
    provenance: InputProvenance, source_identifier: str, source_revision: str
) -> None:
    """Reject a workflow override that differs from its provenance document.

    Args:
        provenance: Parsed content-bound input provenance.
        source_identifier: Explicit workflow source identifier.
        source_revision: Explicit workflow source revision.
    Returns:
        None.
    Raises:
        ValueError: The workflow selection is absent or differs from provenance.
    """

    if not source_identifier or not source_revision:
        raise ValueError(
            "custom adapter requires explicit source identifier and revision"
        )
    if (
        provenance.source_identifier != source_identifier
        or provenance.source_revision != source_revision
    ):
        raise ValueError(
            "custom source identifier/revision differs from content-bound provenance"
        )


@dataclass(frozen=True)
class InstalledExternalCode:
    """A package made importable in a run-local target directory."""

    source_root: Path
    install_root: Path
    receipt: dict[str, Any]


def sha256_file(path: Path) -> str:
    """Return a bounded-memory SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_minmax_roundtrip(
    raw: Any,
    minimum: Any,
    maximum: Any,
    action_mask: Any,
) -> dict[str, Any]:
    """Verify the pinned N1.5 min/max action-coordinate convention.

    This is deliberately a numerical contract check, not a kinematic claim.
    It retains the raw coordinate system declared by :class:`ActionContract`:
    each nonconstant dimension maps to ``[-1, 1]`` and back, while a constant
    training dimension maps to zero and can only round-trip its constant value.
    ``action_mask`` identifies valid recorded horizon entries, including the
    one valid action at an episode terminal frame.
    """

    import numpy as np

    values = np.asarray(raw, dtype=np.float32)
    lower = np.asarray(minimum, dtype=np.float32)
    upper = np.asarray(maximum, dtype=np.float32)
    mask = np.asarray(action_mask, dtype=bool)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("raw action values must be [horizon, action_dimension]")
    if lower.shape != (values.shape[1],) or upper.shape != lower.shape:
        raise ValueError("action min/max dimensions differ from raw action values")
    if mask.shape != (values.shape[0],) or not mask.any():
        raise ValueError("action_mask must select at least one horizon entry")
    if not (
        np.isfinite(values).all()
        and np.isfinite(lower).all()
        and np.isfinite(upper).all()
        and np.all(lower <= upper)
    ):
        raise ValueError("action roundtrip inputs must be finite ordered coordinates")
    varying = lower != upper
    normalized = np.zeros_like(values)
    normalized[:, varying] = (
        2 * (values[:, varying] - lower[varying]) / (upper[varying] - lower[varying])
        - 1
    )
    restored = (normalized + 1) / 2 * (upper - lower) + lower
    difference = np.abs(restored[mask] - values[mask])
    maximum_error = float(difference.max())
    if not np.allclose(restored[mask], values[mask], rtol=1e-5, atol=1e-5):
        raise ValueError(
            "native min/max action roundtrip changed valid raw coordinates"
        )
    return {
        "normalization": "native-min-max-to-[-1,1]",
        "horizon": int(values.shape[0]),
        "action_dimensions": int(values.shape[1]),
        "valid_action_horizon_steps": int(mask.sum()),
        "masked_action_horizon_steps": int((~mask).sum()),
        "constant_training_dimensions": int((~varying).sum()),
        "maximum_absolute_roundtrip_error": maximum_error,
    }


def _safe_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    if not members:
        raise ValueError("external-code archive is empty")
    for member in members:
        candidate = Path(member.name)
        if (
            candidate.is_absolute()
            or ".." in candidate.parts
            or member.issym()
            or member.islnk()
            or member.isdev()
            or not (member.isdir() or member.isfile())
        ):
            raise ValueError("external-code archive has unsafe member")
    return members


def _extract_safe_members(
    archive: tarfile.TarFile, members: list[tarfile.TarInfo], destination: Path
) -> None:
    """Copy validated regular files without delegating extraction to ``tarfile``.

    ``TarFile.extractall`` is deliberately avoided even after validation: this
    keeps the archive boundary auditable and prevents a later tarfile behavior
    change from bypassing the member contract above.
    """

    for member in members:
        target = destination / Path(member.name)
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if target.exists():
            raise ValueError("external-code archive has duplicate file member")
        target.parent.mkdir(parents=True, exist_ok=True)
        source = archive.extractfile(member)
        if source is None:
            raise ValueError("external-code archive regular member is unreadable")
        with source, target.open("xb") as output:
            copyfileobj(source, output)


def extract_content_addressed_archive(
    archive: Path, expected_sha256: str, destination: Path
) -> Path:
    """Safely extract a checked archive into a new run-local directory."""

    if not archive.is_file():
        raise ValueError("declared input archive is absent")
    if sha256_file(archive) != expected_sha256:
        raise ValueError(
            "declared input archive SHA-256 differs from source provenance"
        )
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    try:
        with tarfile.open(archive, "r:*") as opened:
            members = _safe_members(opened)
            _extract_safe_members(opened, members, destination)
    except (tarfile.TarError, OSError) as error:
        raise ValueError(
            "declared input archive could not be safely extracted"
        ) from error
    return destination


def install_external_code(
    archive: Path,
    spec: ExternalCodeSpec,
    destination: Path,
) -> InstalledExternalCode:
    """Verify, safely extract, and install an explicit source archive.

    Dependencies are intentionally not resolved here: runtime dependencies are
    selected by the declarative workflow image, not mutable package metadata.
    """

    if destination.exists():
        raise FileExistsError(destination)
    source = extract_content_addressed_archive(
        archive, spec.archive_sha256, destination / "source"
    )
    target = destination / "site"
    candidates = [path for path in source.iterdir() if path.is_dir()]
    project = candidates[0] if len(candidates) == 1 else source
    pyproject = project / "pyproject.toml"
    if not pyproject.is_file():
        raise ValueError("external-code archive needs a pyproject.toml project root")
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--no-compile",
        "--target",
        str(target),
        str(project),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    if not (target / spec.package.split(".")[0]).exists():
        raise ValueError("declared external package was not installed")
    receipt = {
        "schema_version": "npa.groot-external-code-receipt/v1",
        "archive_sha256": spec.archive_sha256,
        "package": spec.package,
        "adapter_callable": spec.adapter_callable,
        "feedback_callable": spec.feedback_callable,
        "install": "pip install --no-deps --no-compile --target <run-local>",
    }
    return InstalledExternalCode(project, target, receipt)


def invoke_external_callable(
    installed: InstalledExternalCode,
    dotted: str,
    request: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Invoke an explicit trusted adapter using a serializable request/result."""

    module_name, separator, name = dotted.partition(":")
    if not separator:
        raise ValueError("external callable must be module:attribute")
    if not isinstance(request, Mapping):
        raise ValueError("external adapter request must be a mapping")
    # Round-trip first so extension code never receives non-serializable process
    # objects or accidental credentials from the workflow runtime.
    clean_request = json.loads(json.dumps(dict(request), sort_keys=True))
    sys.path.insert(0, str(installed.install_root))
    try:
        target = getattr(importlib.import_module(module_name), name)
        result = target(clean_request)
    finally:
        sys.path.pop(0)
    if not isinstance(result, Mapping):
        raise ValueError("external adapter must return a JSON-object mapping")
    return json.loads(json.dumps(dict(result), sort_keys=True))
