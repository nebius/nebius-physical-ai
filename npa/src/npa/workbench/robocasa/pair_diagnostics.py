"""Allowlisted evidence for refused RoboCasa policy/random comparisons.

Only the failed pair's two initial workspace PNGs, two videos and typed report
may survive worker cleanup. Checkpoints, credentials and arbitrary scratch files
are never part of this diagnostic publication.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import stat
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field
import numpy as np
from PIL import Image

from npa.workbench.robocasa.asset_paths import hold_asset_directory

DIRECTORY = "failure-diagnostics"
REPORT = "failure.json"
ROLES = ("policy", "random_baseline")
MEDIA = tuple(f"{role}.{suffix}" for role in ROLES for suffix in ("png", "mp4"))
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ArmEvidence(_StrictModel):
    """Observed initial robot state and workspace, not a matched outcome."""

    initial_state_sha256: Digest
    initial_workspace_sha256: Digest
    initial_simulator_state_sha256: Digest | None
    raw_action_sha256: Digest
    action_sha256: Digest
    steps: Annotated[int, Field(ge=0)]


class FileEvidence(_StrictModel):
    """Exact bytes retained for one fixed diagnostic filename."""

    sha256: Digest
    bytes: Annotated[int, Field(gt=0)]


class PairFailure(_StrictModel):
    """Failure-only record, deliberately excluding free-form logs and paths."""

    schema_version: Literal["npa.robocasa.pair-failure.v1"]
    status: Literal["failed"]
    reason: Literal["initial_workspace_mismatch", "initial_robot_state_mismatch"]
    episode_index: Annotated[int, Field(ge=0)]
    env_id: Annotated[str, Field(pattern=r"^robocasa/[A-Za-z0-9_-]+$")]
    seed: Annotated[int, Field(ge=0)]
    camera: Literal["video.robot0_agentview_left"]
    initial_capture: Literal["after_reset_before_action_0"]
    state_scope: Literal["robot_observation; simulator hash unavailable when null"]
    policy: ArmEvidence
    random_baseline: ArmEvidence
    checkpoint_sha256: Digest
    artifact_tree_sha256: Digest
    request_manifest_sha256: Digest
    image_source_sha: Annotated[str, Field(pattern=r"^(?:[0-9a-f]{40})?$")]
    image_manifest_digest: Annotated[str, Field(pattern=r"^(?:sha256:[0-9a-f]{64})?$")]
    files: dict[str, FileEvidence]


def _regular_file(info: os.stat_result) -> None:
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("diagnostic evidence must be a non-linked regular file")
    if info.st_uid != os.geteuid():
        raise ValueError("diagnostic evidence has a foreign owner")


@contextlib.contextmanager
def _open_regular(path: Path):
    with hold_asset_directory(path.parent) as parent:
        descriptor = os.open(
            path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent.descriptor
        )
        with os.fdopen(descriptor, "rb") as stream:
            _regular_file(os.fstat(stream.fileno()))
            yield stream
            parent.verify()


def file_evidence(path: Path) -> dict[str, str | int]:
    """Hash one owned regular file without following a symbolic link.

    Args:
        path: Exact diagnostic file.
    Returns:
        Observed byte count and SHA-256.
    Raises:
        OSError: A directory or file cannot be opened without following links.
        ValueError: File ownership or type is unsafe.
    """
    with _open_regular(path) as stream:
        size = os.fstat(stream.fileno()).st_size
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "bytes": size}


def copy_evidence(source: Path, destination: Path) -> None:
    """Copy one no-follow source into a fresh owned diagnostic file.

    Args:
        source: Owned regular evidence file.
        destination: Fresh file inside the caller-owned diagnostic directory.
    Returns:
        None.
    Raises:
        OSError: The source is unsafe or destination already exists.
        ValueError: Source identity changed while copying.
    """
    expected = file_evidence(source)
    with _open_regular(source) as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer)
    if file_evidence(destination) != expected:
        raise ValueError("diagnostic source changed while copying")


def validate_directory(directory: Path) -> dict:
    """Validate the complete failure-only population before retaining/uploading it.

    Args:
        directory: Diagnostic root, not the entire worker workspace.
    Returns:
        Validated failure report.
    Raises:
        OSError: A file or ancestor is unavailable or unsafe.
        ValueError: Population, schema, media bytes or observed pixels differ.
    """
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("diagnostic root must be a real directory")
    if {path.name for path in directory.iterdir()} != {REPORT, *MEDIA}:
        raise ValueError("unexpected or missing diagnostic files")
    report = directory / REPORT
    with _open_regular(report) as stream:
        raw = stream.read(64 * 1024 + 1)
    if len(raw) > 64 * 1024:
        raise ValueError("diagnostic report exceeds its schema envelope")
    parsed = PairFailure.model_validate_json(raw)
    if set(parsed.files) != set(MEDIA):
        raise ValueError("diagnostic report must name exactly the fixed media")
    for name, expected in parsed.files.items():
        if file_evidence(directory / name) != expected.model_dump():
            raise ValueError("diagnostic media identity changed")
    for role in ROLES:
        _verify_initial_pixels(directory / f"{role}.png", getattr(parsed, role))
    return parsed.model_dump(mode="json")


def _verify_initial_pixels(path: Path, arm: ArmEvidence) -> None:
    with _open_regular(path) as stream, Image.open(stream) as image:
        if image.format != "PNG" or image.mode != "RGB":
            raise ValueError("diagnostic initial frame must be an RGB PNG")
        pixels = np.asarray(image)
        digest = hashlib.sha256()
        digest.update(str(pixels.dtype).encode())
        digest.update(json.dumps(pixels.shape).encode())
        digest.update(pixels.tobytes())
        if digest.hexdigest() != arm.initial_workspace_sha256:
            raise ValueError("diagnostic initial pixels differ from the observed hash")


def snapshot_directory(directory: Path, destination: Path) -> dict:
    """Copy only validated failure evidence after all worker descendants stop.

    Args:
        directory: Failed worker's diagnostic root.
        destination: Fresh parent-owned directory.
    Returns:
        Validated report bound to the copied bytes.
    Raises:
        OSError: A file cannot be safely copied.
        ValueError: Validation or copy integrity fails.
    """
    report = validate_directory(directory)
    for name in (REPORT, *MEDIA):
        copy_evidence(directory / name, destination / name)
    if validate_directory(destination) != report:
        raise ValueError("diagnostic snapshot identity changed")
    return report


def write_report(directory: Path, payload: dict) -> None:
    """Seal a typed failure report; no arbitrary worker metadata is accepted.

    Args:
        directory: Owned diagnostic directory containing the four media files.
        payload: Failure-only fields with exact media/source/request bindings.
    Returns:
        None.
    Raises:
        OSError: Evidence cannot be written or safely reopened.
        ValueError: Schema or file validation fails.
    """
    report = PairFailure.model_validate(payload)
    (directory / REPORT).write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )
    validate_directory(directory)
