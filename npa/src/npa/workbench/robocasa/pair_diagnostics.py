"""Allowlisted evidence for refused RoboCasa policy/random comparisons.

The failed pair and already completed pairs retain exactly two initial workspace
PNGs and two videos each, with typed outcomes under an overall failed report.
Checkpoints, credentials and arbitrary scratch never enter this publication.
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

from pydantic import BaseModel, ConfigDict, Field, model_validator
import numpy as np

from npa.workbench.robocasa.asset_paths import hold_asset_directory

DIRECTORY = "failure-diagnostics"
FAILED_DIRECTORY = "failed-pair-diagnostics"
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


class CompletedArm(ArmEvidence):
    """Observed completed arm, not an aggregate or a claim of task success."""

    reward_sum: Annotated[float, Field(allow_inf_nan=False)]
    max_reward: Annotated[float, Field(allow_inf_nan=False)]
    success: bool
    success_sources: list[
        Literal[
            "info.success",
            "info.is_success",
            "info.goal_reached",
            "environment._check_success",
            "binary_reward",
        ]
    ]
    terminated: bool
    truncated: bool
    state_keys: list[
        Literal[
            "state.base_position",
            "state.base_rotation",
            "state.end_effector_position_relative",
            "state.end_effector_rotation_relative",
            "state.gripper_qpos",
        ]
    ]
    state_dim: Annotated[int, Field(gt=0)]
    terminal_state_sha256: Digest
    terminal_workspace_sha256: Digest
    action_count: Annotated[int, Field(ge=0)]
    action_clipping_applied: bool
    action_out_of_bounds_steps: Annotated[int, Field(ge=0)]
    max_action_bound_violation: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    video_sha256: Digest
    video_bytes: Annotated[int, Field(gt=0)]
    video_frames: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def _consistent_counts(self):
        if (
            self.action_count != self.steps
            or self.video_frames != self.steps + 1
            or self.action_out_of_bounds_steps > self.steps
            or self.action_clipping_applied != (self.action_out_of_bounds_steps > 0)
        ):
            raise ValueError("completed arm counts differ")
        return self


class EvidenceBinding(_StrictModel):
    """Immutable run, checkpoint bytes and runtime identity shared by every pair."""

    checkpoint_sha256: Digest
    artifact_tree_sha256: Digest
    request_manifest_sha256: Digest
    image_source_sha: Annotated[str, Field(pattern=r"^(?:[0-9a-f]{40})?$")]
    image_manifest_digest: Annotated[str, Field(pattern=r"^(?:sha256:[0-9a-f]{64})?$")]


class CompletedPair(EvidenceBinding):
    """A contiguous completed pair sharing its enclosing run/source bindings."""

    episode_index: Annotated[int, Field(ge=0)]
    env_id: Annotated[str, Field(pattern=r"^robocasa/[A-Za-z0-9_-]+$")]
    seed: Annotated[int, Field(ge=0)]
    policy: CompletedArm
    random_baseline: CompletedArm

    @model_validator(mode="after")
    def _matched(self):
        for field in ("initial_state_sha256", "initial_workspace_sha256"):
            if getattr(self.policy, field) != getattr(self.random_baseline, field):
                raise ValueError("completed pair does not satisfy the matching guard")
        return self


class PairFailure(EvidenceBinding):
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
    files: dict[str, FileEvidence]
    completed_prefix: Literal["unavailable; failed-pair evidence only"] | None = None


class PairFailureWithPrefix(PairFailure):
    """Failed run plus completed prefix; never a final population success rate.

    Checkpoint, artifact tree, request and image bindings on the enclosing record
    apply to every arm. Every completed pair repeats the same bindings; differing
    bindings and free-form metadata are forbidden.
    """

    schema_version: Literal["npa.robocasa.pair-failure.v2"]
    requested_pairs: Annotated[int, Field(gt=0)]
    unattempted_pairs: Annotated[int, Field(ge=0)]
    completed_pairs: list[CompletedPair]

    @model_validator(mode="after")
    def _contiguous_prefix(self):
        if self.completed_prefix is not None:
            raise ValueError("completed prefix cannot also be unavailable")
        if (
            len(self.completed_pairs) != self.episode_index
            or any(
                pair.episode_index != index
                for index, pair in enumerate(self.completed_pairs)
            )
            or self.unattempted_pairs != self.requested_pairs - self.episode_index - 1
        ):
            raise ValueError("completed prefix or requested population differs")
        for pair in self.completed_pairs:
            if any(
                getattr(pair, key) != getattr(self, key)
                for key in EvidenceBinding.model_fields
            ):
                raise ValueError(
                    "completed pair source/checkpoint/request binding differs"
                )
        return self


def media_names(report: dict) -> tuple[str, ...]:
    """Enumerate the only publishable media, never paths supplied by a worker.

    Args:
        report: Validated failure report.
    Returns:
        Four fixed filenames per completed pair followed by the failed media.
    Raises:
        KeyError: A completed pair lacks its index.
    """
    return (
        tuple(
            f"completed-{pair['episode_index']:04d}-{name}"
            for pair in report.get("completed_pairs", [])
            for name in MEDIA
        )
        + MEDIA
    )


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


def validate_directory(directory: Path, *, requested_pairs: int | None = None) -> dict:
    """Validate the complete failure-only population before retaining/uploading it.

    Args:
        directory: Diagnostic root, not the entire worker workspace.
        requested_pairs: Admitted request population, required for v2 records.
    Returns:
        Validated failure report.
    Raises:
        OSError: A file or ancestor is unavailable or unsafe.
        ValueError: Population, schema, media bytes or observed pixels differ.
    """
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("diagnostic root must be a real directory")
    parsed = _read_report(directory / REPORT, requested_pairs)
    report = _report_payload(parsed)
    names = media_names(report)
    if {path.name for path in directory.iterdir()} != {REPORT, *names}:
        raise ValueError("unexpected or missing diagnostic files")
    if set(parsed.files) != set(names):
        raise ValueError("diagnostic report must name exactly the fixed media")
    for name, expected in parsed.files.items():
        if file_evidence(directory / name) != expected.model_dump():
            raise ValueError("diagnostic media identity changed")
    for role in ROLES:
        _verify_initial_pixels(directory / f"{role}.png", getattr(parsed, role))
    for pair in getattr(parsed, "completed_pairs", []):
        _verify_completed_media(directory, pair, parsed.files)
    return report


def _read_report(path: Path, requested_pairs: int | None):
    if requested_pairs is not None and (
        type(requested_pairs) is not int or requested_pairs < 1
    ):
        raise ValueError("invalid admitted diagnostic population")
    # Only the admitted caller can enlarge the old single-pair envelope. Never
    # use child metadata or a file's advertised size as allocation authority.
    limit = 64 * 1024 * (requested_pairs or 1)
    with _open_regular(path) as stream:
        raw = bytearray()
        while len(raw) <= limit:
            chunk = stream.read(min(64 * 1024, limit + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
    if len(raw) > limit:
        raise ValueError("diagnostic report exceeds its schema envelope")
    payload = json.loads(raw)
    if payload.get("schema_version") == "npa.robocasa.pair-failure.v2":
        parsed = PairFailureWithPrefix.model_validate(payload)
        if requested_pairs is None or parsed.requested_pairs != requested_pairs:
            raise ValueError("diagnostic population differs from admitted request")
        return parsed
    return PairFailure.model_validate(payload)


def _report_payload(report: PairFailure) -> dict:
    payload = report.model_dump(mode="json")
    if payload["completed_prefix"] is None:
        payload.pop("completed_prefix")
    return payload


def _verify_completed_media(directory: Path, pair: CompletedPair, files: dict) -> None:
    for role in ROLES:
        arm = getattr(pair, role)
        stem = f"completed-{pair.episode_index:04d}-{role}"
        _verify_initial_pixels(directory / f"{stem}.png", arm)
        video = files[f"{stem}.mp4"]
        if video.sha256 != arm.video_sha256 or video.bytes != arm.video_bytes:
            raise ValueError("completed outcome video identity differs")


def _verify_initial_pixels(path: Path, arm: ArmEvidence) -> None:
    from PIL import Image

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


def snapshot_directory(
    directory: Path, destination: Path, *, requested_pairs: int | None = None
) -> dict:
    """Copy only validated failure evidence after all worker descendants stop.

    Args:
        directory: Failed worker's diagnostic root.
        destination: Fresh parent-owned directory.
        requested_pairs: Admitted parent request population, not child metadata.
    Returns:
        Validated report bound to the copied bytes.
    Raises:
        OSError: A file cannot be safely copied.
        ValueError: Validation or copy integrity fails.
    """
    report = validate_directory(directory, requested_pairs=requested_pairs)
    for name in (REPORT, *media_names(report)):
        copy_evidence(directory / name, destination / name)
    if validate_directory(destination, requested_pairs=requested_pairs) != report:
        raise ValueError("diagnostic snapshot identity changed")
    return report


def write_report(directory: Path, payload: dict, *, requested_pairs: int) -> None:
    """Seal a typed failure report; no arbitrary worker metadata is accepted.

    Args:
        directory: Owned diagnostic directory containing enumerated media files.
        payload: Failure-only fields with exact media/source/request bindings.
        requested_pairs: Population from the admitted operation request.
    Returns:
        None.
    Raises:
        OSError: Evidence cannot be written or safely reopened.
        ValueError: Schema or file validation fails.
    """
    model = (
        PairFailureWithPrefix
        if payload.get("schema_version") == "npa.robocasa.pair-failure.v2"
        else PairFailure
    )
    report = model.model_validate(payload)
    (directory / REPORT).write_text(
        json.dumps(_report_payload(report), indent=2, sort_keys=True) + "\n"
    )
    validate_directory(directory, requested_pairs=requested_pairs)
