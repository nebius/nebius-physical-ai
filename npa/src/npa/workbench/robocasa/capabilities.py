"""Real RoboCasa capability operations.

This module is the single source of truth for RoboCasa capability behavior. The
FastAPI service, the CLI, and the SDK all call into it. It exercises the real
upstream RoboCasa surface: Gymnasium task registration, kitchen asset
availability, headless EGL environment reset, and a random rollout with a video
artifact.

GPU-heavy imports (robocasa, robosuite, mujoco, gymnasium) are deferred to call
time so that importing this module on a client without the simulation stack
never fails.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import stat
import struct
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

import numpy as np
from typing import Any, BinaryIO, Callable

from npa.clients.storage import safe_s3_download_target
from npa.workbench.robocasa.schemas import (
    DEFAULT_ENV_ID,
    RoboCasaRunRequest,
    RoboCasaSystemInfo,
)

LOGGER = logging.getLogger(__name__)

#: Capabilities this tool can exercise, keyed by the upstream capability id.
SUPPORTED_CAPABILITIES = {
    "kitchen_task_registration",
    "kitchen_asset_availability",
    "kitchen_egl_env_reset",
    "kitchen_random_rollout",
    "kitchen_trajectory_export",
    "kitchen_policy_eval",
}

ROBOCASA_EMBODIMENT = "PandaOmron"
ROBOCASA_OBJECT_REGISTRIES = ("objaverse",)
ROBOCASA_STATE_LAYOUT = (
    ("state.base_position", 3),
    ("state.base_rotation", 4),
    ("state.end_effector_position_relative", 3),
    ("state.end_effector_rotation_relative", 4),
    ("state.gripper_qpos", 2),
)
ROBOCASA_STATE_KEYS = tuple(key for key, _width in ROBOCASA_STATE_LAYOUT)
ROBOCASA_STATE_DIM = sum(width for _key, width in ROBOCASA_STATE_LAYOUT)
_SOURCE_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_MANIFEST_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_TREE_HASH_DOMAIN = b"npa.canonical-tree-sha256.v1\0"

ROBOCASA_ASSET_REPOSITORY = "robocasa/robocasa-assets"
ROBOCASA_ASSET_REVISION = "1b92c3d02ca4354984fec961357db0bff7b32166"
NVIDIA_KITCHEN_ASSET_REPOSITORY = (
    "nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF"
)
NVIDIA_KITCHEN_ASSET_REVISION = "420a04af939c34873e6839a586b70844baf28aab"
DEPLOYED_SOURCE_SHA_ENV = "ROBOCASA_DEPLOYED_IMAGE_SOURCE_SHA"
DEPLOYED_MANIFEST_DIGEST_ENV = "ROBOCASA_DEPLOYED_IMAGE_MANIFEST_DIGEST"
TRAINING_PROVENANCE_FILENAME = "training_dataset_provenance.json"
WORKER_TEMP_ROOT_ENV = "ROBOCASA_WORKER_TEMP_ROOT"
WORKER_ASSET_TEMP_ROOT_ENV = "ROBOCASA_WORKER_ASSET_TEMP_ROOT"
_ASSET_ARCHIVE_MEMBER_LIMIT = 100_000
_ASSET_ARCHIVE_MEMBER_SIZE_LIMIT = 8 * 1024**3
_ASSET_ARCHIVE_UNCOMPRESSED_LIMIT = 64 * 1024**3
_ASSET_ARCHIVE_COMPRESSED_LIMIT = 66 * 1024**3
_ASSET_ARCHIVE_CENTRAL_DIRECTORY_LIMIT = 128 * 1024**2
_ASSET_ARCHIVE_DIRECTORY_LIMIT = 100_000
_ASSET_ARCHIVE_PATH_LIMIT = 4096
_ASSET_ARCHIVE_DEPTH_LIMIT = 64
_ASSET_EXTRACT_CHUNK = 1024 * 1024
_ASSET_RECEIPT_SIZE_LIMIT = 64 * 1024
_SAFE_ZIP_COMPRESSION = frozenset({zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED})


class RoboCasaError(RuntimeError):
    """Raised when a RoboCasa capability operation fails."""


@dataclass(frozen=True)
class _RuntimeIdentity:
    """Immutable source and image identity visible inside the service."""

    source_identity: str
    image_source_sha: str
    image_manifest_digest: str


@dataclass(frozen=True)
class _AssetArchive:
    """One immutable archive and its validated publication location."""

    repo_id: str
    revision: str
    filename: str
    extract_to: str
    publish_path: str
    required_path: str


@dataclass(frozen=True)
class _PreparedAction:
    """Keep the raw policy action distinct from the action applied to the env."""

    value: Any
    raw_flat: np.ndarray
    applied_flat: np.ndarray
    max_bound_violation: float


@dataclass
class _ActionTrace:
    """Hash raw/applied actions and count action-bound corrections."""

    raw_digest: Any = field(default_factory=hashlib.sha256)
    applied_digest: Any = field(default_factory=hashlib.sha256)
    out_of_bounds_steps: int = 0
    max_bound_violation: float = 0.0

    def update(self, prepared: _PreparedAction) -> None:
        self.raw_digest.update(prepared.raw_flat.tobytes())
        self.applied_digest.update(prepared.applied_flat.tobytes())
        if prepared.max_bound_violation > 0:
            self.out_of_bounds_steps += 1
        self.max_bound_violation = max(
            self.max_bound_violation, prepared.max_bound_violation
        )


def make_run_id(capability: str, manifest: str) -> str:
    """Build a deterministic run id from a capability and request manifest."""
    digest = hashlib.sha256(f"{capability}:{manifest}".encode("utf-8")).hexdigest()[:12]
    return f"robocasa-{capability}-{digest}"


def compute_manifest_sha256(capability: str, payload: dict[str, Any]) -> str:
    """Compute a content hash over a request payload."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{capability}:{canonical}".encode("utf-8")).hexdigest()


def _import_robocasa() -> Any:
    """Import the real robocasa package, raising a clear error if absent."""
    try:
        import robocasa  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on the container.
        raise RoboCasaError(
            "robocasa is not installed in this environment; run inside the "
            "npa-robocasa image"
        ) from exc
    return robocasa


def _import_gymnasium() -> Any:
    try:
        import gymnasium as gym
    except Exception as exc:  # pragma: no cover - depends on the container.
        raise RoboCasaError("gymnasium is not installed in this environment") from exc
    return gym


def _assets_root() -> Path:
    """Locate RoboCasa assets without importing its eager object catalog.

    Importing :mod:`robocasa` before the runtime-fetched object assets exist
    permanently caches empty ``mjcf_paths`` in the service process.  Keep
    read-only system-info and asset checks from changing later simulation
    behavior.
    """
    import importlib.util

    loaded = sys.modules.get("robocasa")
    loaded_path = getattr(loaded, "__file__", None)
    if loaded_path:
        return Path(loaded_path).resolve().parent / "models" / "assets"
    try:
        robocasa_spec = importlib.util.find_spec("robocasa")
    except ValueError:
        robocasa_spec = None
    if robocasa_spec is None or robocasa_spec.origin is None:
        raise RoboCasaError("robocasa package not found")
    return Path(robocasa_spec.origin).resolve().parent / "models" / "assets"


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # pragma: no cover - best effort.
        return ""


def _runtime_identity(*, require_deployment: bool = False) -> _RuntimeIdentity:
    """Return and validate the immutable identity carried by this runtime."""
    source_sha = os.environ.get("NPA_IMAGE_SOURCE_SHA", "").strip().lower()
    required = os.environ.get("ROBOCASA_REQUIRE_IMAGE_SOURCE_SHA", "").strip().lower()
    deployed_source_sha = os.environ.get(DEPLOYED_SOURCE_SHA_ENV, "").strip().lower()
    manifest_digest = os.environ.get(DEPLOYED_MANIFEST_DIGEST_ENV, "").strip().lower()
    if required not in {"", "0", "false", "1", "true"}:
        raise RoboCasaError("ROBOCASA_REQUIRE_IMAGE_SOURCE_SHA must be boolean")
    if source_sha and not _SOURCE_SHA_PATTERN.fullmatch(source_sha):
        raise RoboCasaError("NPA_IMAGE_SOURCE_SHA must be a 40-character git SHA")
    if required in {"1", "true"} and not source_sha:
        raise RoboCasaError("RoboCasa image is missing required NPA_IMAGE_SOURCE_SHA")
    if deployed_source_sha and not _SOURCE_SHA_PATTERN.fullmatch(deployed_source_sha):
        raise RoboCasaError(f"{DEPLOYED_SOURCE_SHA_ENV} must be a 40-character git SHA")
    if manifest_digest and not _MANIFEST_DIGEST_PATTERN.fullmatch(manifest_digest):
        raise RoboCasaError(
            f"{DEPLOYED_MANIFEST_DIGEST_ENV} must be an exact sha256 manifest digest"
        )
    if require_deployment or deployed_source_sha or manifest_digest:
        _require_deployed_identity(source_sha, deployed_source_sha, manifest_digest)
    return _RuntimeIdentity(
        source_identity="container_image" if source_sha else "local_unbound",
        image_source_sha=source_sha,
        image_manifest_digest=manifest_digest,
    )


def _require_deployed_identity(
    source_sha: str, deployed_source_sha: str, manifest_digest: str
) -> None:
    if not deployed_source_sha or not manifest_digest:
        raise RoboCasaError("RoboCasa deployment runtime identity is incomplete")
    if not source_sha:
        raise RoboCasaError("RoboCasa deployed image has no baked source identity")
    if source_sha != deployed_source_sha:
        raise RoboCasaError(
            "RoboCasa baked source identity does not match its deployment identity"
        )


def verify_runtime_identity(
    expected_source_sha: str, expected_manifest_digest: str
) -> _RuntimeIdentity:
    """Fail unless a request names the exact deployed runtime identity."""
    identity = _runtime_identity(require_deployment=True)
    if not _SOURCE_SHA_PATTERN.fullmatch(expected_source_sha):
        raise RoboCasaError("request expected image source SHA is missing or malformed")
    if not _MANIFEST_DIGEST_PATTERN.fullmatch(expected_manifest_digest):
        raise RoboCasaError(
            "request expected image manifest digest is missing or malformed"
        )
    if identity.image_source_sha != expected_source_sha:
        raise RoboCasaError("request expected image source SHA does not match runtime")
    if identity.image_manifest_digest != expected_manifest_digest:
        raise RoboCasaError(
            "request expected image manifest digest does not match runtime"
        )
    return identity


def system_info() -> RoboCasaSystemInfo:
    """Collect system and RoboCasa stack information."""
    runtime_identity = _runtime_identity()
    info = RoboCasaSystemInfo(
        status="ok",
        python=platform.python_version(),
        platform=platform.platform(),
        robocasa_version=_package_version("robocasa"),
        robosuite_version=_package_version("robosuite"),
        mujoco_version=_package_version("mujoco"),
        gymnasium_version=_package_version("gymnasium"),
        lerobot_version=_package_version("lerobot"),
        torch_version=_package_version("torch"),
        torchvision_version=_package_version("torchvision"),
        source_identity=runtime_identity.source_identity,
        image_source_sha=runtime_identity.image_source_sha,
        image_manifest_digest=runtime_identity.image_manifest_digest,
    )
    try:
        import torch

        info.cuda_available = bool(torch.cuda.is_available())
        info.cuda_device_count = (
            int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        )
        if torch.cuda.is_available():
            info.cuda_device_name = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover - torch optional on client.
        LOGGER.debug("torch unavailable: %s", exc)
    try:
        gym = _import_gymnasium()
        robocasa_envs = sorted(
            env for env in gym.envs.registry.keys() if env.startswith("robocasa/")
        )
        info.registered_env_count = len(robocasa_envs)
    except Exception as exc:  # pragma: no cover - depends on the container.
        LOGGER.debug("gymnasium unavailable: %s", exc)
    try:
        info.assets_root_exists = _assets_root().exists()
    except Exception as exc:  # pragma: no cover - depends on the container.
        LOGGER.debug("assets root unavailable: %s", exc)
    return info


_LIGHTWHEEL_FIXTURES = (
    "blenders",
    "cabinets",
    "coffee_machines",
    "dishwashers",
    "electric_kettles",
    "fridges",
    "handles",
    "hoods",
    "microwaves",
    "ovens",
    "sinks",
    "stand_mixers",
    "stoves",
    "stovetops",
    "toaster_ovens",
    "toasters",
    "windows",
)
_LIGHTWHEEL_OBJECTS = (
    "aluminum_foil",
    "basket",
    "blender_jug",
    "cheese_grater",
    "chicken_drumstick",
    "cinnamon",
    "colander",
    "cookie_dough_ball",
    "cream_cheese_stick",
    "digital_scale",
    "dish_brush",
    "dish_rack",
    "flour_bag",
    "flower_vase",
    "fruit_bowl",
    "glass_cup",
    "honey_bottle",
    "hotdog_bun",
    "ice_cube",
    "ice_cube_tray",
    "jar",
    "juice",
    "kebab_skewer",
    "kettle",
    "knife_block",
    "lemon_wedge",
    "lettuce",
    "marshmallow",
    "mayonnaise",
    "measuring_cup",
    "mug_tree",
    "mustard",
    "oil_and_vinegar_bottle",
    "oven_tray",
    "pancake",
    "paper_towel_holder",
    "paprika",
    "peeler",
    "pickle_slice",
    "pitcher",
    "pizza",
    "pizza_cutter",
    "placemat",
    "plant",
    "pot",
    "reamer",
    "salt_and_pepper_shaker",
    "sandwich_bread",
    "saucepan",
    "shrimp",
    "soap_dispenser",
    "spray",
    "stool",
    "strainer",
    "straw",
    "sugar_cube",
    "syrup_bottle",
    "tiered_basket",
    "tiered_shelf",
    "tomato_slice",
    "tongs",
    "tray",
    "tupperware",
    "turkey_slice",
    "turmeric",
    "utensil_rack",
    "utensil_set",
    "whisk",
    "wooden_spoon",
)


def _asset_archives() -> tuple[_AssetArchive, ...]:
    standard = (
        _robocasa_asset("textures.zip", ".", "textures", "textures"),
        _robocasa_asset(
            "generative_textures.zip",
            ".",
            "generative_textures",
            "generative_textures",
        ),
        _robocasa_asset("fixtures.zip", ".", "fixtures", "fixtures/accessories"),
        _robocasa_asset(
            "objaverse.zip", "objects", "objects/objaverse", "objects/objaverse"
        ),
        _robocasa_asset(
            "aigen_objs.zip", "objects", "objects/aigen_objs", "objects/aigen_objs"
        ),
    )
    fixtures = tuple(
        _nvidia_asset(
            f"fixtures_lightwheel/{name}.zip",
            "fixtures",
            f"fixtures/{name}",
        )
        for name in _LIGHTWHEEL_FIXTURES
    )
    objects = tuple(
        _nvidia_asset(
            f"objects_lightwheel/{name}.zip",
            "objects/lightwheel",
            f"objects/lightwheel/{name}",
        )
        for name in _LIGHTWHEEL_OBJECTS
    )
    return standard + fixtures + objects


def _robocasa_asset(
    filename: str, extract_to: str, publish_path: str, required_path: str
) -> _AssetArchive:
    return _AssetArchive(
        ROBOCASA_ASSET_REPOSITORY,
        ROBOCASA_ASSET_REVISION,
        filename,
        extract_to,
        publish_path,
        required_path,
    )


def _nvidia_asset(filename: str, extract_to: str, publish_path: str) -> _AssetArchive:
    return _AssetArchive(
        NVIDIA_KITCHEN_ASSET_REPOSITORY,
        NVIDIA_KITCHEN_ASSET_REVISION,
        filename,
        extract_to,
        publish_path,
        publish_path,
    )


@contextlib.contextmanager
def _asset_fetch_lock(assets_root: Path) -> Any:
    state_root = assets_root / ".npa_asset_fetch"
    state_root.mkdir(parents=True, exist_ok=True)
    with (state_root / "fetch.lock").open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield state_root
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _download_assets() -> None:
    """Fetch every asset archive under one lock and fail on any partial fetch."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - container dependency.
        raise RoboCasaError(
            "huggingface_hub is required to fetch RoboCasa assets"
        ) from exc

    assets_root = _assets_root()
    assets_root.mkdir(parents=True, exist_ok=True)
    with _asset_fetch_lock(assets_root) as state_root:
        for archive in _asset_archives():
            _fetch_asset_archive(
                archive,
                assets_root=assets_root,
                state_root=state_root,
                downloader=hf_hub_download,
            )


def _fetch_asset_archive(
    archive: _AssetArchive,
    *,
    assets_root: Path,
    state_root: Path,
    downloader: Callable[..., str],
) -> None:
    receipt_path = _asset_receipt_path(state_root, archive)
    if _asset_receipt_is_valid(receipt_path, archive, assets_root):
        return
    try:
        zip_path = Path(
            downloader(
                repo_id=archive.repo_id,
                repo_type="dataset",
                filename=archive.filename,
                revision=archive.revision,
            )
        )
        _stage_publish_and_receipt(archive, zip_path, assets_root, receipt_path)
    except RoboCasaError:
        raise
    except Exception as exc:
        raise RoboCasaError(
            f"failed to fetch RoboCasa asset {archive.repo_id}@{archive.revision}:"
            f"{archive.filename}: {exc}"
        ) from exc
    LOGGER.info(
        "downloaded RoboCasa asset %s from %s@%s",
        archive.filename,
        archive.repo_id,
        archive.revision,
    )


def _stage_publish_and_receipt(
    archive: _AssetArchive,
    zip_path: Path,
    assets_root: Path,
    receipt_path: Path,
) -> None:
    temporary_parent = _asset_temporary_parent(receipt_path)
    with tempfile.TemporaryDirectory(
        prefix="asset-", dir=temporary_parent
    ) as temporary:
        staging_root = Path(temporary)
        extract_root = (
            staging_root
            if archive.extract_to == "."
            else staging_root / archive.extract_to
        )
        extract_root.mkdir(parents=True, exist_ok=True)
        with _open_asset_archive(zip_path) as archive_file:
            archive_sha256 = _sha256_open_file(archive_file)
            _extract_validated_zip_file(archive_file, zip_path, extract_root)
            if _sha256_open_file(archive_file) != archive_sha256:
                raise RoboCasaError(
                    f"RoboCasa asset archive changed while reading: {zip_path}"
                )
        _validate_asset_tree(staging_root / archive.required_path, archive)
        staged_publish = staging_root / archive.publish_path
        staged_digest, file_count = _published_asset_identity(staged_publish, archive)
        _replace_asset_tree(staged_publish, assets_root / archive.publish_path)
    receipt = _asset_receipt(archive, archive_sha256, staged_digest, file_count)
    _write_json_atomic(receipt_path, receipt)


def _asset_temporary_parent(receipt_path: Path) -> Path:
    configured = os.environ.get(WORKER_ASSET_TEMP_ROOT_ENV, "").strip()
    if not configured:
        return receipt_path.parent
    state_root = receipt_path.parent.parent.resolve()
    workers_root = (state_root / "workers").resolve()
    candidate = Path(configured).resolve()
    try:
        candidate.relative_to(workers_root)
    except ValueError as exc:
        raise RoboCasaError(
            f"{WORKER_ASSET_TEMP_ROOT_ENV} must stay under {workers_root}"
        ) from exc
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def _extract_validated_zip(zip_path: Path, destination: Path) -> None:
    with _open_asset_archive(zip_path) as archive_file:
        _extract_validated_zip_file(archive_file, zip_path, destination)


@contextlib.contextmanager
def _open_asset_archive(zip_path: Path) -> Any:
    try:
        resolved = zip_path.resolve(strict=True)
        descriptor = os.open(
            resolved,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise RoboCasaError(
            f"failed to open RoboCasa asset archive {zip_path}: {exc}"
        ) from exc
    try:
        with os.fdopen(descriptor, "rb") as archive_file:
            yield archive_file
    except Exception:
        raise


def _extract_validated_zip_file(
    archive_file: BinaryIO, zip_path: Path, destination: Path
) -> None:
    try:
        _preflight_asset_zip(archive_file, zip_path)
        with ZipFile(archive_file) as archive:
            members = archive.infolist()
            if not members:
                raise RoboCasaError(f"RoboCasa asset archive is empty: {zip_path}")
            if len(members) > _ASSET_ARCHIVE_MEMBER_LIMIT:
                raise RoboCasaError(
                    "RoboCasa asset archive exceeds the member limit: "
                    f"{len(members)} > {_ASSET_ARCHIVE_MEMBER_LIMIT}"
                )
            total_declared = 0
            paths: set[str] = set()
            file_paths: set[tuple[str, ...]] = set()
            directory_paths: set[tuple[str, ...]] = set()
            for member in members:
                relative = _validate_zip_member(member.filename, member.external_attr)
                normalized = relative.as_posix().rstrip("/")
                if normalized in paths:
                    raise RoboCasaError(
                        f"RoboCasa asset archive has duplicate path: {member.filename}"
                    )
                paths.add(normalized)
                parts = relative.parts
                parents = tuple(parts[:depth] for depth in range(1, len(parts)))
                if any(parent in file_paths for parent in parents):
                    raise RoboCasaError(
                        "RoboCasa asset archive has a file/directory prefix "
                        f"collision: {member.filename}"
                    )
                member_path = tuple(parts)
                if not member.is_dir() and member_path in directory_paths:
                    raise RoboCasaError(
                        "RoboCasa asset archive has a file/directory collision: "
                        f"{member.filename}"
                    )
                directories = parents + ((member_path,) if member.is_dir() else ())
                for directory in directories:
                    directory_paths.add(directory)
                    if len(directory_paths) > _ASSET_ARCHIVE_DIRECTORY_LIMIT:
                        raise RoboCasaError(
                            "RoboCasa asset archive exceeds the directory limit"
                        )
                if not member.is_dir():
                    file_paths.add(member_path)
                if member.flag_bits & 0x1:
                    raise RoboCasaError(
                        f"RoboCasa asset archive has encrypted member: {member.filename}"
                    )
                if member.compress_type not in _SAFE_ZIP_COMPRESSION:
                    raise RoboCasaError(
                        "RoboCasa asset archive uses unsupported compression: "
                        f"{member.filename}"
                    )
                if member.file_size > _ASSET_ARCHIVE_MEMBER_SIZE_LIMIT:
                    raise RoboCasaError(
                        "RoboCasa asset archive member exceeds the size limit: "
                        f"{member.filename}"
                    )
                total_declared += member.file_size
                if total_declared > _ASSET_ARCHIVE_UNCOMPRESSED_LIMIT:
                    raise RoboCasaError(
                        "RoboCasa asset archive exceeds the uncompressed size limit"
                    )

            total_extracted = 0
            for member in members:
                relative = _validate_zip_member(member.filename, member.external_attr)
                target = destination.joinpath(*relative.parts)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                extracted = 0
                with archive.open(member, "r") as source, target.open("xb") as output:
                    while chunk := source.read(_ASSET_EXTRACT_CHUNK):
                        extracted += len(chunk)
                        total_extracted += len(chunk)
                        if (
                            extracted > member.file_size
                            or extracted > _ASSET_ARCHIVE_MEMBER_SIZE_LIMIT
                            or total_extracted > _ASSET_ARCHIVE_UNCOMPRESSED_LIMIT
                        ):
                            raise RoboCasaError(
                                "RoboCasa asset archive exceeded declared extraction "
                                f"bounds: {member.filename}"
                            )
                        output.write(chunk)
                if extracted != member.file_size:
                    raise RoboCasaError(
                        "RoboCasa asset archive member size disagrees with metadata: "
                        f"{member.filename}"
                    )
    except BadZipFile as exc:
        raise RoboCasaError(
            f"RoboCasa asset archive is not a valid zip: {zip_path}"
        ) from exc
    except OSError as exc:
        raise RoboCasaError(
            f"failed to extract RoboCasa asset archive {zip_path}: {exc}"
        ) from exc


def _preflight_asset_zip(archive_file: BinaryIO, zip_path: Path) -> None:
    """Bound ZIP metadata before ``ZipFile`` allocates the central directory."""
    archive_stat = os.fstat(archive_file.fileno())
    if not stat.S_ISREG(archive_stat.st_mode):
        raise RoboCasaError(f"RoboCasa asset archive is not a regular file: {zip_path}")
    if archive_stat.st_size > _ASSET_ARCHIVE_COMPRESSED_LIMIT:
        raise RoboCasaError("RoboCasa asset archive exceeds the compressed size limit")
    # CPython's bounded EOCD reader also resolves ZIP64 records without loading
    # central-directory entries. It reads at most the ZIP comment window plus
    # fixed-size EOCD/ZIP64 records.
    end_record = zipfile._EndRecData(archive_file)  # noqa: SLF001
    if end_record is None:
        raise BadZipFile("end-of-central-directory record is missing")
    if (
        int(end_record[zipfile._ECD_DISK_NUMBER]) != 0  # noqa: SLF001
        or int(end_record[zipfile._ECD_DISK_START]) != 0  # noqa: SLF001
    ):
        raise BadZipFile("multi-disk RoboCasa asset archives are not supported")
    declared_member_count = int(
        end_record[zipfile._ECD_ENTRIES_TOTAL]  # noqa: SLF001
    )
    entries_this_disk = int(
        end_record[zipfile._ECD_ENTRIES_THIS_DISK]  # noqa: SLF001
    )
    if entries_this_disk != declared_member_count:
        raise BadZipFile("inconsistent central-directory entry counts")
    central_directory_size = int(end_record[zipfile._ECD_SIZE])  # noqa: SLF001
    central_directory_offset = int(end_record[zipfile._ECD_OFFSET])  # noqa: SLF001
    if declared_member_count > _ASSET_ARCHIVE_MEMBER_LIMIT:
        raise RoboCasaError(
            "RoboCasa asset archive exceeds the member limit: "
            f"{declared_member_count} > {_ASSET_ARCHIVE_MEMBER_LIMIT}"
        )
    if central_directory_size > _ASSET_ARCHIVE_CENTRAL_DIRECTORY_LIMIT:
        raise RoboCasaError(
            "RoboCasa asset archive exceeds the central-directory size limit"
        )
    concatenated_prefix = (
        int(end_record[zipfile._ECD_LOCATION])  # noqa: SLF001
        - central_directory_size
        - central_directory_offset
    )
    if end_record[zipfile._ECD_SIGNATURE] == zipfile.stringEndArchive64:  # noqa: SLF001
        concatenated_prefix -= (
            zipfile.sizeEndCentDir64 + zipfile.sizeEndCentDir64Locator
        )
    start = central_directory_offset + concatenated_prefix
    if (
        start < 0
        or central_directory_size < 0
        or start + central_directory_size > archive_stat.st_size
    ):
        raise BadZipFile("invalid central-directory bounds")

    archive_file.seek(start)
    remaining = central_directory_size
    actual_member_count = 0
    while remaining:
        if remaining < zipfile.sizeCentralDir:
            raise BadZipFile("truncated central-directory record")
        header = archive_file.read(zipfile.sizeCentralDir)
        if len(header) != zipfile.sizeCentralDir:
            raise BadZipFile("truncated central directory")
        fields = struct.unpack(zipfile.structCentralDir, header)
        if fields[zipfile._CD_SIGNATURE] != zipfile.stringCentralDir:  # noqa: SLF001
            raise BadZipFile("invalid central-directory record signature")
        variable_size = (
            int(fields[zipfile._CD_FILENAME_LENGTH])  # noqa: SLF001
            + int(fields[zipfile._CD_EXTRA_FIELD_LENGTH])  # noqa: SLF001
            + int(fields[zipfile._CD_COMMENT_LENGTH])  # noqa: SLF001
        )
        record_size = zipfile.sizeCentralDir + variable_size
        if record_size > remaining:
            raise BadZipFile("central-directory record exceeds declared bounds")
        actual_member_count += 1
        if actual_member_count > _ASSET_ARCHIVE_MEMBER_LIMIT:
            raise RoboCasaError(
                "RoboCasa asset archive exceeds the actual member limit"
            )
        archive_file.seek(variable_size, os.SEEK_CUR)
        remaining -= record_size
    if actual_member_count != declared_member_count:
        raise BadZipFile("declared and actual central-directory entry counts disagree")
    archive_file.seek(0)


def _validate_zip_member(name: str, external_attr: int) -> PurePosixPath:
    path = PurePosixPath(name)
    mode = external_attr >> 16
    if (
        not name
        or "\x00" in name
        or "\\" in name
        or len(name.encode("utf-8")) > _ASSET_ARCHIVE_PATH_LIMIT
        or len(path.parts) > _ASSET_ARCHIVE_DEPTH_LIMIT
        or path.is_absolute()
        or path == PurePosixPath(".")
        or ".." in path.parts
    ):
        raise RoboCasaError(f"RoboCasa asset archive has unsafe path: {name}")
    if stat.S_ISLNK(mode):
        raise RoboCasaError(f"RoboCasa asset archive has unsafe symlink: {name}")
    file_type = stat.S_IFMT(mode)
    if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise RoboCasaError(
            f"RoboCasa asset archive has unsupported member type: {name}"
        )
    return path


def _validate_asset_tree(path: Path, archive: _AssetArchive) -> None:
    try:
        _digest, file_count = _asset_tree_identity(path)
    except (OSError, RoboCasaError) as exc:
        raise RoboCasaError(
            f"RoboCasa asset {archive.filename} has an unsafe tree: {exc}"
        ) from exc
    if file_count == 0:
        raise RoboCasaError(
            f"RoboCasa asset {archive.filename} lacks {archive.required_path}"
        )


def _replace_asset_tree(staged: Path, target: Path) -> None:
    if not staged.is_dir():
        raise RoboCasaError(f"staged RoboCasa asset tree does not exist: {staged}")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target_mode = target.lstat().st_mode
    except FileNotFoundError:
        pass
    else:
        if stat.S_ISDIR(target_mode):
            shutil.rmtree(target)
        else:
            target.unlink()
    os.replace(staged, target)


def _asset_receipt_path(state_root: Path, archive: _AssetArchive) -> Path:
    identity = "\0".join(
        (archive.repo_id, archive.revision, archive.filename, archive.publish_path)
    )
    name = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    receipts = state_root / "receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    return receipts / f"{name}.json"


def _asset_receipt(
    archive: _AssetArchive,
    archive_sha256: str,
    tree_sha256: str,
    file_count: int,
) -> dict[str, Any]:
    return {
        "schema": "npa.robocasa.asset_receipt.v1",
        "repo_id": archive.repo_id,
        "revision": archive.revision,
        "filename": archive.filename,
        "publish_path": archive.publish_path,
        "required_path": archive.required_path,
        "archive_sha256": archive_sha256,
        "tree_sha256": tree_sha256,
        "file_count": file_count,
    }


def _published_asset_identity(
    published: Path, archive: _AssetArchive
) -> tuple[str, int]:
    """Hash bytes owned by one archive while ignoring nested archive mounts."""
    publish_path = PurePosixPath(archive.publish_path)
    exclusions: list[PurePosixPath] = []
    for candidate in _asset_archives():
        if candidate == archive:
            continue
        try:
            relative = PurePosixPath(candidate.publish_path).relative_to(publish_path)
        except ValueError:
            continue
        if relative.parts:
            exclusions.append(relative)
    return _tree_identity(
        published,
        max_files=_ASSET_ARCHIVE_MEMBER_LIMIT,
        max_bytes=_ASSET_ARCHIVE_UNCOMPRESSED_LIMIT,
        max_directories=_ASSET_ARCHIVE_DIRECTORY_LIMIT,
        excluded_paths=tuple(sorted(set(exclusions), key=lambda item: item.as_posix())),
    )


def _asset_receipt_is_valid(
    receipt_path: Path, archive: _AssetArchive, assets_root: Path
) -> bool:
    try:
        receipt_stat = receipt_path.lstat()
        if (
            not stat.S_ISREG(receipt_stat.st_mode)
            or receipt_stat.st_size > _ASSET_RECEIPT_SIZE_LIMIT
        ):
            return False
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return False
    if not isinstance(receipt, dict):
        return False
    expected = {
        "schema": "npa.robocasa.asset_receipt.v1",
        "repo_id": archive.repo_id,
        "revision": archive.revision,
        "filename": archive.filename,
        "publish_path": archive.publish_path,
        "required_path": archive.required_path,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        return False
    if not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("archive_sha256", ""))):
        return False
    if not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("tree_sha256", ""))):
        return False
    receipt_file_count = receipt.get("file_count")
    if type(receipt_file_count) is not int or receipt_file_count <= 0:
        return False
    required = assets_root / archive.required_path
    published = assets_root / archive.publish_path
    try:
        if not required.is_dir() or required.is_symlink():
            return False
        tree_sha256, file_count = _published_asset_identity(published, archive)
    except (OSError, RoboCasaError):
        return False
    return tree_sha256 == receipt["tree_sha256"] and file_count == receipt_file_count


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _make_env(env_id: str, *, download_assets: bool = True) -> Any:
    """Create a headless EGL RoboCasa env, downloading assets when requested.

    The upstream gym wrapper defaults ``split="test"``, which the pinned
    ``create_env`` rejects; pass ``split="all"`` so real rollouts can run.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    if download_assets:
        _download_assets()
    # Import robocasa AFTER assets are downloaded so OBJ_CATEGORIES are
    # populated with real object paths. This also registers the gymnasium
    # environments.
    _import_robocasa()
    gym = _import_gymnasium()
    try:
        return gym.make(
            env_id,
            split="all",
            obj_registries=ROBOCASA_OBJECT_REGISTRIES,
        )
    except Exception as exc:  # pragma: no cover - depends on the container.
        raise RoboCasaError(f"failed to create RoboCasa env {env_id}: {exc}") from exc


def kitchen_task_registration(
    *, env_id: str = DEFAULT_ENV_ID, download_assets: bool = False
) -> dict[str, Any]:
    """Verify Gymnasium task registration for a RoboCasa env id."""
    if download_assets:
        _download_assets()
    _import_robocasa()
    gym = _import_gymnasium()
    if env_id not in gym.envs.registry:
        raise RoboCasaError(f"RoboCasa env id not registered: {env_id}")
    spec = gym.envs.registry[env_id]
    robocasa_envs = sorted(
        env for env in gym.envs.registry.keys() if env.startswith("robocasa/")
    )
    return {
        "env_id": env_id,
        "entry_point": str(spec.entry_point),
        "registered_env_count": len(robocasa_envs),
        "sample_registered_envs": robocasa_envs[:10],
    }


def kitchen_asset_availability() -> dict[str, Any]:
    """Verify the kitchen assets root exists and is populated."""
    assets_root = _assets_root()
    if not assets_root.exists():
        raise RoboCasaError(f"RoboCasa assets root does not exist: {assets_root}")
    subdirs = sorted(p.name for p in assets_root.iterdir() if p.is_dir())
    return {
        "assets_root": str(assets_root),
        "assets_root_exists": True,
        "subdirs": subdirs,
    }


def kitchen_egl_env_reset(
    *,
    env_id: str = DEFAULT_ENV_ID,
    seed: int | None = None,
    download_assets: bool = True,
) -> dict[str, Any]:
    """Create a headless EGL RoboCasa env and reset it."""
    env = _make_env(env_id, download_assets=download_assets)
    try:
        obs, info = env.reset(seed=seed)
        return {
            "env_id": env_id,
            "reset_ok": True,
            "observation_keys": sorted(obs.keys()) if isinstance(obs, dict) else [],
            "info_keys": sorted(info.keys()) if isinstance(info, dict) else [],
            "mujoco_gl": os.environ.get("MUJOCO_GL", ""),
        }
    except Exception as exc:  # pragma: no cover - depends on the container.
        raise RoboCasaError(f"failed to reset RoboCasa env {env_id}: {exc}") from exc
    finally:
        try:
            env.close()
        except Exception as exc:  # pragma: no cover - best effort.
            LOGGER.debug("env close failed: %s", exc)


def kitchen_random_rollout(
    *,
    env_id: str = DEFAULT_ENV_ID,
    iterations: int = 1,
    seed: int | None = None,
    output_dir: Path | None = None,
    download_assets: bool = True,
) -> dict[str, Any]:
    """Run a real random rollout and write a video artifact."""
    if output_dir is None:
        raise RoboCasaError("random rollout requires an output directory for its MP4")
    env = _make_env(env_id, download_assets=download_assets)
    try:
        obs, _ = env.reset(seed=seed)
        frames: list[Any] = []
        for _ in range(iterations):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            try:
                frames.append(env.render())
            except Exception as exc:  # pragma: no cover - render may be unavailable.
                LOGGER.debug("render unavailable: %s", exc)
            if terminated or truncated:
                break
        result: dict[str, Any] = {
            "env_id": env_id,
            "rollout_ok": True,
            "iterations": iterations,
            "final_reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "observation_keys": sorted(obs.keys()) if isinstance(obs, dict) else [],
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = _write_required_video(frames, output_dir / "rollout.mp4")
        result["video_exists"] = True
        result["video_bytes"] = video_path.stat().st_size
        result["video_sha256"] = _sha256_file(video_path)
        return result
    except Exception as exc:  # pragma: no cover - depends on the container.
        raise RoboCasaError(f"failed to run RoboCasa rollout {env_id}: {exc}") from exc
    finally:
        try:
            env.close()
        except Exception as exc:  # pragma: no cover - best effort.
            LOGGER.debug("env close failed: %s", exc)


def kitchen_trajectory_export(
    *,
    env_id: str = DEFAULT_ENV_ID,
    iterations: int = 1,
    num_envs: int = 1,
    seed: int | None = None,
    output_dir: Path | None = None,
    download_assets: bool = True,
) -> dict[str, Any]:
    """Run real RoboCasa rollouts and export trajectories for LeRobotDataset.

    Writes one ``episode_NNNN/`` directory per rollout, each containing the
    numpy arrays the ``npa adapter convert`` adapter consumes:

      obs_workspace.npy  (T, H, W, 3) uint8   workspace camera
      obs_wrist.npy      (T, H, W, 3) uint8   wrist camera
      state.npy          (T, n_joints) float32
      actions.npy        (T, n_actions) float32

    plus a per-episode ``rollout.mp4`` and run-level ``metadata.json`` /
    ``metrics.json``. This is the real trajectory export seam between RoboCasa
    simulation and LeRobotDataset policy training.
    """
    try:
        env_ids = _parse_env_ids(env_id)
        episodes = _collect_trajectory_episodes(
            env_ids=env_ids,
            iterations=iterations,
            num_envs=num_envs,
            seed=seed,
            output_dir=output_dir,
            download_assets=download_assets,
        )
        result = _trajectory_export_result(env_id, env_ids, iterations, episodes)
        if output_dir is not None:
            _write_run_metadata(output_dir, env_id, episodes)
            result["output_dir"] = str(output_dir)
        return result
    except Exception as exc:  # pragma: no cover - depends on the container.
        raise RoboCasaError(
            f"failed to run RoboCasa trajectory export {env_id}: {exc}"
        ) from exc


def _collect_trajectory_episodes(
    *,
    env_ids: list[str],
    iterations: int,
    num_envs: int,
    seed: int | None,
    output_dir: Path | None,
    download_assets: bool,
) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    env = None
    try:
        for episode_index in range(num_envs):
            if env is not None:
                env.close()
            episode_env_id = env_ids[episode_index % len(env_ids)]
            env = _make_env(
                episode_env_id,
                download_assets=download_assets and episode_index == 0,
            )
            episode_seed = seed + episode_index if seed is not None else episode_index
            arrays, outcome, video_frames = _collect_trajectory_episode(
                env, iterations=iterations, seed=episode_seed
            )
            if output_dir is not None:
                episode_dir = output_dir / f"episode_{episode_index:04d}"
                episode_dir.mkdir(parents=True, exist_ok=True)
                _write_trajectory_artifacts(episode_dir, arrays, video_frames)
            episodes.append(
                _trajectory_episode_identity(episode_index, episode_env_id, outcome)
            )
        return episodes
    finally:
        try:
            if env is not None:
                env.close()
        except Exception as exc:  # pragma: no cover - best effort.
            LOGGER.debug("env close failed: %s", exc)


def _trajectory_episode_identity(
    episode_index: int, env_id: str, outcome: dict[str, Any]
) -> dict[str, Any]:
    return {
        "episode_index": episode_index,
        "env_id": env_id,
        "task": env_id.removeprefix("robocasa/"),
        "embodiment": ROBOCASA_EMBODIMENT,
        **outcome,
    }


def _trajectory_export_result(
    env_id: str,
    env_ids: list[str],
    iterations: int,
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "npa.robocasa.trajectory_export.v1",
        "env_id": env_id,
        "env_ids": env_ids,
        "embodiment": ROBOCASA_EMBODIMENT,
        "trajectory_export_ok": True,
        "temporal_alignment": "observation_before_action",
        "policy": "random_action_baseline",
        "num_episodes": len(episodes),
        "iterations": iterations,
        "successful_episodes": sum(int(episode["success"]) for episode in episodes),
        "episodes": episodes,
    }


def _collect_trajectory_episode(
    env: Any, *, iterations: int, seed: int
) -> tuple[dict[str, list[Any]], dict[str, Any], list[np.ndarray]]:
    """Collect causally aligned observation/action rows from one random rollout."""
    _seed_action_space(env.action_space, seed)
    observation, _ = env.reset(seed=seed)
    arrays: dict[str, list[Any]] = {
        "workspace": [],
        "wrist": [],
        "state": [],
        "actions": [],
    }
    outcome = _empty_episode_outcome()
    for _ in range(iterations):
        action = env.action_space.sample()
        _append_observation_action(arrays, observation, action)
        observation, reward, terminated, truncated, info = env.step(action)
        _update_episode_outcome(outcome, env, reward, terminated, truncated, info)
        if terminated or truncated:
            break
    terminal_frame = _obs_image(observation, "video.robot0_agentview_left")
    _validate_trajectory_arrays(arrays)
    video_frames = [*arrays["workspace"], terminal_frame]
    return (
        arrays,
        _trajectory_episode_record(arrays, outcome, terminal_frame, seed=seed),
        video_frames,
    )


def _append_observation_action(
    arrays: dict[str, list[Any]], observation: dict[str, Any], action: Any
) -> None:
    arrays["workspace"].append(_obs_image(observation, "video.robot0_agentview_left"))
    arrays["wrist"].append(_obs_image(observation, "video.robot0_eye_in_hand"))
    arrays["state"].append(_validated_finite("robot state", _obs_state(observation)))
    arrays["actions"].append(_validated_action(action))


def _empty_episode_outcome() -> dict[str, Any]:
    return {
        "reward_sum": 0.0,
        "final_reward": 0.0,
        "max_reward": float("-inf"),
        "success": False,
        "success_sources": set(),
        "terminated": False,
        "truncated": False,
    }


def _update_episode_outcome(
    outcome: dict[str, Any],
    env: Any,
    reward: Any,
    terminated: Any,
    truncated: Any,
    info: Any,
) -> None:
    reward_value = float(reward)
    if not np.isfinite(reward_value):
        raise RoboCasaError("RoboCasa reward contains a non-finite value")
    success, sources = _native_task_success(env, info, reward_value)
    outcome["reward_sum"] += reward_value
    outcome["final_reward"] = reward_value
    outcome["max_reward"] = max(outcome["max_reward"], reward_value)
    outcome["success"] = bool(outcome["success"] or success)
    outcome["success_sources"].update(sources)
    outcome["terminated"] = bool(terminated)
    outcome["truncated"] = bool(truncated)


def _trajectory_episode_record(
    arrays: dict[str, list[Any]],
    outcome: dict[str, Any],
    terminal_frame: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    workspace = arrays["workspace"]
    return {
        "seed": seed,
        "length": len(arrays["actions"]),
        "reward_sum": outcome["reward_sum"],
        "final_reward": outcome["final_reward"],
        "max_reward": outcome["max_reward"],
        "success": outcome["success"],
        "success_sources": sorted(outcome["success_sources"]),
        "terminated": outcome["terminated"],
        "truncated": outcome["truncated"],
        "state_keys": list(ROBOCASA_STATE_KEYS),
        "state_dim": int(np.asarray(arrays["state"][0]).size),
        "initial_workspace_sha256": _sha256_array(workspace[0]),
        "terminal_workspace_sha256": _sha256_array(terminal_frame),
        "video_frames": len(workspace) + 1,
    }


def _write_trajectory_artifacts(
    episode_dir: Path,
    arrays: dict[str, list[Any]],
    video_frames: list[np.ndarray],
) -> None:
    np.save(episode_dir / "obs_workspace.npy", np.stack(arrays["workspace"]))
    np.save(episode_dir / "obs_wrist.npy", np.stack(arrays["wrist"]))
    np.save(episode_dir / "state.npy", np.stack(arrays["state"]))
    np.save(episode_dir / "actions.npy", np.stack(arrays["actions"]))
    _write_required_video(video_frames, episode_dir / "rollout.mp4")


def _validate_trajectory_arrays(arrays: dict[str, list[Any]]) -> None:
    lengths = {name: len(values) for name, values in arrays.items()}
    if len(set(lengths.values())) != 1 or not next(iter(lengths.values()), 0):
        raise RoboCasaError(f"trajectory arrays are not aligned: {lengths}")


def _flatten_action(action: Any) -> np.ndarray:
    """Flatten a RoboCasa action (OrderedDict or array) into a float32 vector."""
    if isinstance(action, dict):
        parts = []
        for key in sorted(action.keys()):
            value = action[key]
            if isinstance(value, dict):
                for sub_key in sorted(value.keys()):
                    parts.append(
                        np.asarray(value[sub_key], dtype=np.float32).reshape(-1)
                    )
            else:
                parts.append(np.asarray(value, dtype=np.float32).reshape(-1))
        return np.concatenate(parts)
    return np.asarray(action, dtype=np.float32).reshape(-1)


def _obs_image(obs: dict[str, Any], key: str) -> Any:
    """Return a uint8 (H, W, 3) image frame for a RoboCasa observation key."""
    frame = obs.get(key)
    if frame is None:
        raise RoboCasaError(f"RoboCasa observation missing image key: {key}")
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr.astype(np.uint8)
    if arr.ndim == 4 and arr.shape[0] == 1:
        return arr[0].astype(np.uint8)
    raise RoboCasaError(f"RoboCasa image key {key!r} has unexpected shape {arr.shape}")


def _obs_state(obs: dict[str, Any]) -> np.ndarray:
    """Build a float32 robot-state vector from a RoboCasa observation."""
    parts: list[np.ndarray] = []
    missing: list[str] = []
    for key, expected_width in ROBOCASA_STATE_LAYOUT:
        value = obs.get(key)
        if value is None:
            missing.append(key)
            continue
        part = np.asarray(value, dtype=np.float32).reshape(-1)
        if part.size != expected_width:
            raise RoboCasaError(
                f"RoboCasa robot state key {key!r} has width {part.size}; "
                f"expected {expected_width} for {ROBOCASA_EMBODIMENT}"
            )
        parts.append(part)
    if missing:
        raise RoboCasaError(f"RoboCasa observation missing robot state keys: {missing}")
    state = np.concatenate(parts)
    if state.size != ROBOCASA_STATE_DIM:
        raise RoboCasaError(
            f"RoboCasa robot state has width {state.size}; "
            f"expected {ROBOCASA_STATE_DIM} for {ROBOCASA_EMBODIMENT}"
        )
    return state


def _write_run_metadata(
    output_dir: Path, env_id: str, episodes: list[dict[str, Any]]
) -> None:
    """Write run-level metadata.json and metrics.json for the trajectory export."""
    metadata = {
        "env_id": env_id,
        "num_episodes": len(episodes),
        "episodes": episodes,
        "format": "lerobot-adapter-input",
        "schema": "npa.robocasa.trajectory_export.v1",
        "temporal_alignment": "observation_before_action",
        "policy": "random_action_baseline",
        "embodiment": ROBOCASA_EMBODIMENT,
        "robot_type": "panda_omron",
        "state_keys": list(ROBOCASA_STATE_KEYS),
        "state_dim": int(episodes[0]["state_dim"]) if episodes else 0,
        "task_env_ids": sorted({str(ep["env_id"]) for ep in episodes}),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )
    metrics = {
        "num_episodes": len(episodes),
        "total_steps": sum(int(ep["length"]) for ep in episodes),
        "mean_episode_length": (
            sum(int(ep["length"]) for ep in episodes) / len(episodes)
            if episodes
            else 0.0
        ),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True)
    )


def _parse_env_ids(value: str) -> list[str]:
    env_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not env_ids:
        raise RoboCasaError("at least one RoboCasa env id is required")
    if len(set(env_ids)) != len(env_ids):
        raise RoboCasaError("RoboCasa env ids must be unique")
    if any(not item.startswith("robocasa/") for item in env_ids):
        raise RoboCasaError("all RoboCasa env ids must start with 'robocasa/'")
    return env_ids


def _sha256_tree(root: Path) -> str:
    """Hash a tree with canonical path/content length framing."""
    return _tree_identity(root)[0]


def _asset_tree_identity(root: Path) -> tuple[str, int]:
    return _tree_identity(
        root,
        max_files=_ASSET_ARCHIVE_MEMBER_LIMIT,
        max_bytes=_ASSET_ARCHIVE_UNCOMPRESSED_LIMIT,
        max_directories=_ASSET_ARCHIVE_DIRECTORY_LIMIT,
    )


def _tree_identity(
    root: Path,
    *,
    max_files: int | None = None,
    max_bytes: int | None = None,
    max_directories: int | None = None,
    excluded_paths: tuple[PurePosixPath, ...] = (),
) -> tuple[str, int]:
    """Return a canonical digest and count without following non-regular nodes."""
    try:
        root_stat = root.lstat()
    except OSError as exc:
        raise RoboCasaError(f"cannot inspect tree root {root}: {exc}") from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise RoboCasaError(f"tree root must be a real directory: {root}")

    def raise_walk_error(exc: OSError) -> None:
        raise RoboCasaError(f"cannot walk tree {root}: {exc}") from exc

    files: list[tuple[str, Path]] = []
    declared_bytes = 0
    directory_count = 0

    def excluded(path: Path) -> bool:
        relative = PurePosixPath(path.relative_to(root).as_posix())
        return any(
            relative == prefix or prefix in relative.parents
            for prefix in excluded_paths
        )

    for directory, dirnames, filenames in os.walk(
        root, followlinks=False, onerror=raise_walk_error
    ):
        directory_path = Path(directory)
        dirnames.sort()
        filenames.sort()
        dirnames[:] = [name for name in dirnames if not excluded(directory_path / name)]
        directory_count += len(dirnames)
        if max_directories is not None and directory_count > max_directories:
            raise RoboCasaError(
                f"tree exceeds directory-count limit: {max_directories}"
            )
        for name in dirnames:
            child = directory_path / name
            child_stat = child.lstat()
            if stat.S_ISLNK(child_stat.st_mode) or not stat.S_ISDIR(child_stat.st_mode):
                raise RoboCasaError(f"tree contains unsafe directory node: {child}")
        for name in filenames:
            child = directory_path / name
            if excluded(child):
                continue
            child_stat = child.lstat()
            if stat.S_ISLNK(child_stat.st_mode) or not stat.S_ISREG(child_stat.st_mode):
                raise RoboCasaError(f"tree contains unsafe file node: {child}")
            if max_files is not None and len(files) >= max_files:
                raise RoboCasaError(f"tree exceeds file-count limit: {max_files}")
            declared_bytes += child_stat.st_size
            if max_bytes is not None and declared_bytes > max_bytes:
                raise RoboCasaError(f"tree exceeds byte limit: {max_bytes}")
            files.append((child.relative_to(root).as_posix(), child))

    files.sort()
    digest = hashlib.sha256()
    digest.update(_TREE_HASH_DOMAIN)
    digest.update(len(files).to_bytes(8, "big"))
    for relative_path, path in files:
        _update_length_frame(digest, relative_path.encode("utf-8"))
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        with os.fdopen(descriptor, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise RoboCasaError(f"tree file changed type while hashing: {path}")
            digest.update(before.st_size.to_bytes(8, "big"))
            bytes_read = 0
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                bytes_read += len(chunk)
                digest.update(chunk)
            after = os.fstat(handle.fileno())
            if (
                bytes_read != before.st_size
                or before.st_dev != after.st_dev
                or before.st_ino != after.st_ino
                or before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
            ):
                raise RoboCasaError(f"tree file changed while hashing: {path}")
    return digest.hexdigest(), len(files)


def _update_length_frame(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _download_s3_tree(uri: str, destination: Path) -> Path:
    if not uri.startswith("s3://"):
        raise RoboCasaError(
            "policy evaluation requires an exact s3:// checkpoint prefix"
        )
    import boto3

    bucket, prefix = uri[5:].split("/", 1)
    client = boto3.client(
        "s3",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL")
        or os.environ.get("NEBIUS_S3_ENDPOINT")
        or None,
    )
    paginator = client.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for item in page.get("Contents", []):
            key = str(item["Key"])
            if key.endswith("/"):
                continue
            target = safe_s3_download_target(destination, key, prefix.rstrip("/") + "/")
            target.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(target))
            count += 1
    if count == 0:
        raise RoboCasaError("checkpoint prefix contains no objects")
    return destination


def _resolve_pretrained_dir(root: Path) -> Path:
    def is_loadable(candidate: Path) -> bool:
        return (candidate / "config.json").is_file() and any(
            (candidate / name).is_file()
            for name in ("model.safetensors", "pytorch_model.bin")
        )

    preferred = root / "checkpoints" / "last" / "pretrained_model"
    if is_loadable(preferred):
        return preferred

    nested = sorted(
        candidate
        for candidate in root.rglob("pretrained_model")
        if candidate != preferred and is_loadable(candidate)
    )
    if len(nested) == 1:
        return nested[0]
    if len(nested) > 1:
        choices = [candidate.relative_to(root).as_posix() for candidate in nested]
        raise RoboCasaError(
            "exact checkpoint prefix contains multiple loadable pretrained_model "
            f"directories without checkpoints/last: {choices}"
        )
    if is_loadable(root):
        return root
    raise RoboCasaError("exact checkpoint contains no loadable pretrained_model")


def _checkpoint_identity(checkpoint_root: Path) -> tuple[Path, str, str]:
    """Resolve and hash the exact loadable policy separately from run artifacts."""
    pretrained = _resolve_pretrained_dir(checkpoint_root)
    return pretrained, _sha256_tree(pretrained), _sha256_tree(checkpoint_root)


def _verify_training_provenance(
    checkpoint_root: Path, declared_train_ids: list[str]
) -> dict[str, Any]:
    matches = sorted(checkpoint_root.rglob(TRAINING_PROVENANCE_FILENAME))
    if len(matches) != 1:
        raise RoboCasaError(
            "training artifact must contain exactly one "
            f"{TRAINING_PROVENANCE_FILENAME}; found {len(matches)}"
        )
    path = matches[0]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RoboCasaError("training dataset provenance is unreadable") from exc
    _require_training_provenance_fields(payload)
    artifact_ids = payload.get("declared_training_env_ids")
    if artifact_ids != declared_train_ids:
        raise RoboCasaError(
            "checkpoint training tasks do not exactly match declared training tasks"
        )
    if payload.get("normalized_dataset_training_env_ids") != declared_train_ids:
        raise RoboCasaError(
            "checkpoint dataset tasks do not exactly match declared training tasks"
        )
    expected_task_digest = _task_set_sha256(declared_train_ids)
    if payload.get("declared_task_set_sha256") != expected_task_digest:
        raise RoboCasaError("training task provenance digest does not match its tasks")
    dataset_tasks = payload.get("dataset_tasks")
    if not isinstance(dataset_tasks, list) or not all(
        isinstance(item, str) and item for item in dataset_tasks
    ):
        raise RoboCasaError("training dataset task metadata is missing")
    if payload.get("dataset_task_metadata_sha256") != _task_set_sha256(dataset_tasks):
        raise RoboCasaError("training dataset task metadata digest does not match")
    return {
        **payload,
        "artifact_path": path.relative_to(checkpoint_root).as_posix(),
        "artifact_sha256": _sha256_file(path),
    }


def _require_training_provenance_fields(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise RoboCasaError("training dataset provenance must be a JSON object")
    if payload.get("schema") != "npa.lerobot.training_dataset_provenance.v1":
        raise RoboCasaError("training dataset provenance schema is unsupported")
    if payload.get("declared_training_tasks_verified") is not True:
        raise RoboCasaError("training artifact did not verify its declared tasks")
    digest = str(payload.get("dataset_tree_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise RoboCasaError("training dataset provenance lacks a content digest")


def _task_set_sha256(task_ids: list[str]) -> str:
    encoded = json.dumps(task_ids, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _policy_observation(obs: dict[str, Any], device: Any) -> dict[str, Any]:
    import torch

    def image_tensor(key: str) -> Any:
        array = _obs_image(obs, key)
        return (
            torch.from_numpy(array.copy())
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(device)
        )

    return {
        "observation.images.workspace": image_tensor("video.robot0_agentview_left"),
        "observation.images.wrist": image_tensor("video.robot0_eye_in_hand"),
        "observation.state": torch.from_numpy(_obs_state(obs).copy())
        .float()
        .unsqueeze(0)
        .to(device),
    }


def _unflatten_action(space: Any, values: np.ndarray) -> Any:
    if hasattr(space, "spaces"):
        offset = 0
        result: dict[str, Any] = {}
        for key in sorted(space.spaces):
            child = space.spaces[key]
            size = int(np.prod(child.shape))
            result[key] = values[offset : offset + size].reshape(child.shape)
            offset += size
        if offset != len(values):
            raise RoboCasaError(
                f"policy action dimension {len(values)} does not match RoboCasa action space {offset}"
            )
        return result
    expected = int(np.prod(space.shape))
    if expected != len(values):
        raise RoboCasaError(
            f"policy action dimension {len(values)} does not match RoboCasa action space {expected}"
        )
    return values.reshape(space.shape)


def _seed_action_space(space: Any, seed: int) -> None:
    seed_method = getattr(space, "seed", None)
    if callable(seed_method):
        seed_method(seed)


def _validated_finite(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if not np.isfinite(array).all():
        raise RoboCasaError(f"RoboCasa {name} contains non-finite values")
    return array


def _validated_action(action: Any, space: Any | None = None) -> np.ndarray:
    flat = _validated_finite("action", _flatten_action(action))
    contains = getattr(space, "contains", None)
    if callable(contains) and not bool(contains(action)):
        raise RoboCasaError("policy action is outside the RoboCasa action space")
    return flat


def _action_bounds(space: Any) -> tuple[np.ndarray, np.ndarray] | None:
    children = getattr(space, "spaces", None)
    if children is not None:
        child_bounds = [_action_bounds(children[key]) for key in sorted(children)]
        if any(bounds is None for bounds in child_bounds):
            return None
        lows = [bounds[0] for bounds in child_bounds if bounds is not None]
        highs = [bounds[1] for bounds in child_bounds if bounds is not None]
        return np.concatenate(lows), np.concatenate(highs)
    low = getattr(space, "low", None)
    high = getattr(space, "high", None)
    if low is None or high is None:
        return None
    return np.asarray(low).reshape(-1), np.asarray(high).reshape(-1)


def _prepare_eval_action(action: Any, space: Any) -> _PreparedAction:
    raw_flat = _validated_action(action)
    contains = getattr(space, "contains", None)
    if not callable(contains) or bool(contains(action)):
        return _PreparedAction(action, raw_flat, raw_flat, 0.0)
    bounds = _action_bounds(space)
    if bounds is None or any(len(bound) != len(raw_flat) for bound in bounds):
        _validated_action(action, space)
        raise AssertionError("unreachable")
    applied_flat = np.clip(raw_flat, bounds[0], bounds[1]).astype(np.float32)
    violation = float(np.max(np.abs(raw_flat - applied_flat)))
    prepared = _unflatten_action(space, applied_flat)
    if not bool(contains(prepared)):
        raise RoboCasaError("policy action is outside the RoboCasa action space")
    return _PreparedAction(prepared, raw_flat, applied_flat, violation)


def _native_task_success(env: Any, info: Any, reward: float) -> tuple[bool, list[str]]:
    signals: dict[str, bool] = {}
    if isinstance(info, dict):
        for key in ("success", "is_success", "goal_reached"):
            if key in info:
                signals[f"info.{key}"] = bool(info[key])
    unwrapped = getattr(env, "unwrapped", env)
    checker = getattr(unwrapped, "_check_success", None)
    if callable(checker):
        signals["environment._check_success"] = bool(checker())
    if reward in {0.0, 1.0}:
        signals["binary_reward"] = reward == 1.0
    if not signals:
        raise RoboCasaError("RoboCasa native task-success signal is unavailable")
    if len(set(signals.values())) != 1:
        raise RoboCasaError(f"RoboCasa native task-success signals disagree: {signals}")
    return next(iter(signals.values())), sorted(signals)


def _sha256_array(value: Any) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _load_act_policy(pretrained: Path) -> tuple[Any, Any, Any, Any, Any]:
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = ACTPolicy.from_pretrained(str(pretrained))
    policy.eval()
    device = next(policy.parameters()).device
    config = PreTrainedConfig.from_pretrained(str(pretrained))
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config, pretrained_path=str(pretrained)
    )
    return policy, device, preprocessor, postprocessor, torch


def _act_action_selector(
    runtime: tuple[Any, Any, Any, Any, Any],
) -> Callable[[Any, dict[str, Any]], Any]:
    policy, device, preprocessor, postprocessor, torch = runtime

    def select(_env: Any, observation: dict[str, Any]) -> Any:
        model_observation = preprocessor(_policy_observation(observation, device))
        with torch.inference_mode():
            action = postprocessor(policy.select_action(model_observation))
        flat = np.asarray(action.squeeze(0).detach().cpu(), dtype=np.float32)
        return _unflatten_action(_env.action_space, flat)

    return select


def _random_action_selector(env: Any, _observation: dict[str, Any]) -> Any:
    return env.action_space.sample()


def _rollout_eval_episode(
    env: Any,
    *,
    iterations: int,
    selector: Callable[[Any, dict[str, Any]], Any],
    observation: dict[str, Any],
) -> tuple[list[np.ndarray], dict[str, Any]]:
    frames = [_obs_image(observation, "video.robot0_agentview_left")]
    initial_state = _validated_finite("robot state", _obs_state(observation))
    terminal_state = initial_state
    outcome = _empty_episode_outcome()
    action_trace = _ActionTrace()
    steps = 0
    for _ in range(iterations):
        prepared = _prepare_eval_action(selector(env, observation), env.action_space)
        action_trace.update(prepared)
        observation, reward, terminated, truncated, info = env.step(prepared.value)
        frames.append(_obs_image(observation, "video.robot0_agentview_left"))
        terminal_state = _validated_finite("robot state", _obs_state(observation))
        _update_episode_outcome(outcome, env, reward, terminated, truncated, info)
        steps += 1
        if terminated or truncated:
            break
    return frames, _eval_episode_record(
        frames,
        outcome,
        steps,
        action_trace,
        initial_state=initial_state,
        terminal_state=terminal_state,
    )


def _eval_episode_record(
    frames: list[np.ndarray],
    outcome: dict[str, Any],
    steps: int,
    action_trace: _ActionTrace,
    *,
    initial_state: np.ndarray,
    terminal_state: np.ndarray,
) -> dict[str, Any]:
    return {
        "steps": steps,
        "reward_sum": outcome["reward_sum"],
        "max_reward": outcome["max_reward"],
        "success": outcome["success"],
        "success_sources": sorted(outcome["success_sources"]),
        "terminated": outcome["terminated"],
        "truncated": outcome["truncated"],
        "state_keys": list(ROBOCASA_STATE_KEYS),
        "state_dim": int(initial_state.size),
        "initial_state_sha256": _sha256_array(initial_state),
        "terminal_state_sha256": _sha256_array(terminal_state),
        "initial_workspace_sha256": _sha256_array(frames[0]),
        "terminal_workspace_sha256": _sha256_array(frames[-1]),
        "action_sha256": action_trace.applied_digest.hexdigest(),
        "raw_action_sha256": action_trace.raw_digest.hexdigest(),
        "action_count": steps,
        "action_clipping_applied": action_trace.out_of_bounds_steps > 0,
        "action_out_of_bounds_steps": action_trace.out_of_bounds_steps,
        "max_action_bound_violation": action_trace.max_bound_violation,
    }


def _run_eval_episode(
    task_id: str,
    *,
    seed: int,
    iterations: int,
    selector: Callable[[Any, dict[str, Any]], Any],
    video_path: Path,
    download_assets: bool,
) -> dict[str, Any]:
    env = _make_env(task_id, download_assets=download_assets)
    try:
        _seed_action_space(env.action_space, seed)
        observation, _ = env.reset(seed=seed)
        frames, result = _rollout_eval_episode(
            env, iterations=iterations, selector=selector, observation=observation
        )
        video = _write_required_video(frames, video_path)
        result.update(
            {
                "video_sha256": _sha256_file(video),
                "video_bytes": video.stat().st_size,
                "video_frames": len(frames),
            }
        )
        return result
    finally:
        try:
            env.close()
        except Exception as exc:  # pragma: no cover - best effort.
            LOGGER.debug("env close failed: %s", exc)


def _write_eval_manifest(
    output_dir: Path,
    *,
    train_ids: list[str],
    heldout_ids: list[str],
    num_envs: int,
    seed: int,
) -> tuple[list[dict[str, Any]], str]:
    episodes = [
        {
            "episode_index": index,
            "env_id": heldout_ids[index % len(heldout_ids)],
            "seed": seed + index,
        }
        for index in range(num_envs)
    ]
    manifest = {
        "schema": "npa.robocasa.eval_manifest.v1",
        "train_env_ids": train_ids,
        "heldout_env_ids": heldout_ids,
        "episodes": episodes,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "eval_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return episodes, _sha256_file(path)


def _paired_outcome_counts(pairs: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"policy_wins": 0, "baseline_wins": 0, "ties": 0}
    for pair in pairs:
        policy_success = bool(pair["policy"]["success"])
        baseline_success = bool(pair["random_baseline"]["success"])
        if policy_success and not baseline_success:
            counts["policy_wins"] += 1
        elif baseline_success and not policy_success:
            counts["baseline_wins"] += 1
        else:
            counts["ties"] += 1
    return counts


def kitchen_policy_eval(
    *,
    checkpoint_uri: str,
    train_env_ids: str,
    heldout_env_ids: str,
    iterations: int,
    num_envs: int,
    seed: int | None,
    output_dir: Path,
    download_assets: bool = True,
) -> dict[str, Any]:
    """Evaluate ACT and random actions on matched, disjoint RoboCasa tasks."""
    train_ids = _parse_env_ids(train_env_ids)
    heldout_ids = _parse_env_ids(heldout_env_ids)
    overlap = sorted(set(train_ids) & set(heldout_ids))
    if overlap:
        raise RoboCasaError(f"train/held-out RoboCasa task overlap: {overlap}")

    base_seed = seed if seed is not None else 42
    episode_manifest, manifest_sha256 = _write_eval_manifest(
        output_dir,
        train_ids=train_ids,
        heldout_ids=heldout_ids,
        num_envs=num_envs,
        seed=base_seed,
    )
    execution = _evaluate_policy_checkpoint(
        checkpoint_uri,
        declared_train_ids=train_ids,
        episode_manifest=episode_manifest,
        iterations=iterations,
        output_dir=output_dir,
        download_assets=download_assets,
    )
    result = _policy_eval_result(
        checkpoint_uri=checkpoint_uri,
        base_seed=base_seed,
        train_ids=train_ids,
        heldout_ids=heldout_ids,
        manifest_sha256=manifest_sha256,
        execution=execution,
    )
    _write_policy_eval_outputs(output_dir, result)
    return result


def _evaluate_policy_checkpoint(
    checkpoint_uri: str,
    *,
    declared_train_ids: list[str],
    episode_manifest: list[dict[str, Any]],
    iterations: int,
    output_dir: Path,
    download_assets: bool,
) -> dict[str, Any]:
    configured_temp_root = os.environ.get(WORKER_TEMP_ROOT_ENV, "").strip()
    temporary_parent = Path(configured_temp_root) if configured_temp_root else None
    if temporary_parent is not None:
        temporary_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="robocasa-checkpoint-",
        dir=temporary_parent,
    ) as tmp:
        checkpoint_root = _download_s3_tree(checkpoint_uri, Path(tmp))
        pretrained, checkpoint_sha256, artifact_tree_sha256 = _checkpoint_identity(
            checkpoint_root
        )
        training_provenance = _verify_training_provenance(
            checkpoint_root, declared_train_ids
        )
        checkpoint_selection = pretrained.relative_to(checkpoint_root).as_posix()
        runtime = _load_act_policy(pretrained)
        policy, *_ = runtime
        selector = _act_action_selector(runtime)
        pairs = _run_matched_eval_pairs(
            episode_manifest,
            policy=policy,
            selector=selector,
            iterations=iterations,
            output_dir=output_dir,
            download_assets=download_assets,
        )
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_selection": checkpoint_selection,
        "artifact_tree_sha256": artifact_tree_sha256,
        "training_provenance": training_provenance,
        "pairs": pairs,
    }


def _policy_eval_split_proof(
    train_ids: list[str],
    heldout_ids: list[str],
    manifest_sha256: str,
    training_provenance: dict[str, Any],
) -> dict[str, Any]:
    task_sets_disjoint = set(train_ids).isdisjoint(heldout_ids)
    return {
        "train_env_ids": train_ids,
        "heldout_env_ids": heldout_ids,
        "configured_task_sets_disjoint": task_sets_disjoint,
        "basis": (
            "content-bound training dataset provenance plus held-out "
            "evaluation manifest"
        ),
        "checkpoint_training_tasks_verified": True,
        "training_dataset_tree_sha256": training_provenance["dataset_tree_sha256"],
        "training_provenance_artifact_sha256": training_provenance["artifact_sha256"],
        "training_provenance_artifact_path": training_provenance["artifact_path"],
        "declared_train_task_set_sha256": _task_set_sha256(train_ids),
        "heldout_task_set_sha256": _task_set_sha256(heldout_ids),
        "heldout_episode_manifest_sha256": manifest_sha256,
    }


def _policy_eval_result(
    *,
    checkpoint_uri: str,
    base_seed: int,
    train_ids: list[str],
    heldout_ids: list[str],
    manifest_sha256: str,
    execution: dict[str, Any],
) -> dict[str, Any]:
    pairs = execution["pairs"]
    episodes = [pair["policy"] for pair in pairs]
    baselines = [pair["random_baseline"] for pair in pairs]
    policy_success_rate = _success_rate(episodes)
    baseline_success_rate = _success_rate(baselines)
    return {
        "schema": "npa.robocasa.policy_eval.v1",
        "embodiment": ROBOCASA_EMBODIMENT,
        "checkpoint_uri": checkpoint_uri,
        "checkpoint_sha256": execution["checkpoint_sha256"],
        "checkpoint_selection": execution["checkpoint_selection"],
        "training_artifact_tree_sha256": execution["artifact_tree_sha256"],
        "checkpoint_loadable": True,
        "split_proof": _policy_eval_split_proof(
            train_ids,
            heldout_ids,
            manifest_sha256,
            execution["training_provenance"],
        ),
        "num_episodes": len(episodes),
        "base_seed": base_seed,
        "success_rate": policy_success_rate,
        "baseline_success_rate": baseline_success_rate,
        "success_rate_delta": policy_success_rate - baseline_success_rate,
        "paired_outcomes": _paired_outcome_counts(pairs),
        "mean_reward": _mean_reward(episodes),
        "baseline_mean_reward": _mean_reward(baselines),
        "episodes": episodes,
        "baseline_episodes": baselines,
        "paired_episodes": pairs,
    }


def _write_policy_eval_outputs(output_dir: Path, result: dict[str, Any]) -> None:
    (output_dir / "eval.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "success_rate": result["success_rate"],
                "baseline_success_rate": result["baseline_success_rate"],
                "success_rate_delta": result["success_rate_delta"],
                "mean_reward": result["mean_reward"],
                "baseline_mean_reward": result["baseline_mean_reward"],
                **result["paired_outcomes"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def _run_matched_eval_pairs(
    manifest: list[dict[str, Any]],
    *,
    policy: Any,
    selector: Callable[[Any, dict[str, Any]], Any],
    iterations: int,
    output_dir: Path,
    download_assets: bool,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for item in manifest:
        policy.reset()
        episode_dir = output_dir / f"episode_{item['episode_index']:04d}"
        policy_result = _run_eval_episode(
            item["env_id"],
            seed=item["seed"],
            iterations=iterations,
            selector=selector,
            video_path=episode_dir / "rollout.mp4",
            download_assets=download_assets and not pairs,
        )
        baseline = _run_eval_episode(
            item["env_id"],
            seed=item["seed"],
            iterations=iterations,
            selector=_random_action_selector,
            video_path=episode_dir / "random_baseline.mp4",
            download_assets=False,
        )
        _require_matched_initial_state(policy_result, baseline)
        identity = {key: item[key] for key in ("episode_index", "env_id", "seed")}
        policy_result.update(identity)
        baseline.update(identity)
        pairs.append({**identity, "policy": policy_result, "random_baseline": baseline})
    return pairs


def _require_matched_initial_state(
    policy_result: dict[str, Any], baseline_result: dict[str, Any]
) -> None:
    if (
        policy_result["initial_workspace_sha256"]
        != baseline_result["initial_workspace_sha256"]
    ):
        raise RoboCasaError(
            "policy and random baseline initial workspace frames do not match"
        )
    if policy_result["initial_state_sha256"] != baseline_result["initial_state_sha256"]:
        raise RoboCasaError(
            "policy and random baseline initial robot states do not match"
        )


def _success_rate(episodes: list[dict[str, Any]]) -> float:
    return sum(int(episode["success"]) for episode in episodes) / len(episodes)


def _mean_reward(episodes: list[dict[str, Any]]) -> float:
    return sum(float(episode["reward_sum"]) for episode in episodes) / len(episodes)


def _write_required_video(frames: list[Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = _write_video(frames, path)
    if written is None or not written.is_file() or written.stat().st_size <= 0:
        raise RoboCasaError(f"RoboCasa video was not written: {path}")
    return written


def _write_video(frames: list[Any], path: Path) -> Path | None:
    """Write frames to an MP4 using imageio's ffmpeg backend when available."""
    if not frames:
        return None
    try:
        import imageio

        imageio.mimsave(path, frames, fps=20)
        return path
    except Exception:  # pragma: no cover - ffmpeg backend may be absent.
        return None


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return _sha256_open_file(handle)


def _sha256_open_file(handle: BinaryIO) -> str:
    handle.seek(0)
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(65536), b""):
        digest.update(chunk)
    handle.seek(0)
    return digest.hexdigest()


def run_capability(
    request: RoboCasaRunRequest,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Dispatch a RoboCasa capability request to the real implementation."""
    if request.capability == "kitchen_task_registration":
        return kitchen_task_registration(
            env_id=request.env_id,
            download_assets=request.download_assets,
        )
    if request.capability == "kitchen_asset_availability":
        return kitchen_asset_availability()
    if request.capability == "kitchen_egl_env_reset":
        return kitchen_egl_env_reset(
            env_id=request.env_id,
            seed=request.seed,
            download_assets=request.download_assets,
        )
    if request.capability == "kitchen_random_rollout":
        return kitchen_random_rollout(
            env_id=request.env_id,
            iterations=request.iterations,
            seed=request.seed,
            output_dir=output_dir,
            download_assets=request.download_assets,
        )
    if request.capability == "kitchen_trajectory_export":
        return kitchen_trajectory_export(
            env_id=request.env_id,
            iterations=request.iterations,
            num_envs=request.num_envs,
            seed=request.seed,
            output_dir=output_dir,
            download_assets=request.download_assets,
        )
    if request.capability == "kitchen_policy_eval":
        if output_dir is None:
            raise RoboCasaError("policy evaluation requires an output directory")
        return kitchen_policy_eval(
            checkpoint_uri=request.checkpoint_uri,
            train_env_ids=request.train_env_ids,
            heldout_env_ids=request.heldout_env_ids,
            iterations=request.iterations,
            num_envs=request.num_envs,
            seed=request.seed,
            output_dir=output_dir,
            download_assets=request.download_assets,
        )
    raise RoboCasaError(f"unsupported robocasa capability: {request.capability}")


def run_capability_with_output(
    request: RoboCasaRunRequest,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run a capability and persist/upload its output truthfully.

    Creates a temporary output directory when none is provided, runs the
    capability, and uploads any produced artifacts to ``request.output_uri``
    when set. Returns the capability result dict.

    This is the single entrypoint used by both the FastAPI service and the SDK
    local path so that local execution persists and uploads output exactly like
    a service run, instead of silently dropping artifacts.
    """
    if output_dir is None:
        with tempfile.TemporaryDirectory(prefix="robocasa_") as tmp:
            return run_capability_with_output(request, output_dir=Path(tmp))
    result = run_capability(request, output_dir=output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = _execution_provenance(request, output_dir, result)
    result["execution_provenance"] = provenance
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    if request.output_uri:
        upload_output(output_dir, request.output_uri, result)
    return result


def _provenance_environment_ids(
    request: RoboCasaRunRequest, result: dict[str, Any]
) -> list[str]:
    env_ids = result.get("env_ids")
    if not isinstance(env_ids, list):
        heldout = result.get("split_proof", {}).get("heldout_env_ids", [])
        env_ids = heldout if isinstance(heldout, list) and heldout else [request.env_id]
    return [str(item) for item in env_ids]


def _provenance_mp4_artifacts(output_dir: Path) -> list[dict[str, str]]:
    return [
        {
            "path": path.relative_to(output_dir).as_posix(),
            "sha256": _sha256_file(path),
        }
        for path in sorted(output_dir.rglob("*.mp4"))
    ]


def _execution_provenance(
    request: RoboCasaRunRequest,
    output_dir: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Describe how run artifacts were produced, without overstating validation."""
    rollout_capabilities = {
        "kitchen_random_rollout",
        "kitchen_trajectory_export",
        "kitchen_policy_eval",
    }
    runtime_identity = _runtime_identity()
    mp4_artifacts = _provenance_mp4_artifacts(output_dir)
    return {
        "schema": "npa.robocasa.execution_provenance.v2",
        "generator": "robocasa",
        "simulator": "mujoco",
        "source_identity": runtime_identity.source_identity,
        "image_source_sha": runtime_identity.image_source_sha,
        "image_manifest_digest": runtime_identity.image_manifest_digest,
        "capability": request.capability,
        "environment_ids": _provenance_environment_ids(request, result),
        "execution_path": (
            "gymnasium.make(robocasa/*)->RoboCasa->MuJoCo->step/render"
            if request.capability in rollout_capabilities
            else "RoboCasa runtime capability probe"
        ),
        "capture_source": (
            "runtime_environment_observation_or_render"
            if request.capability in rollout_capabilities
            else "none"
        ),
        "stock_or_copied_fixture": False,
        "runtime_result_sha256": hashlib.sha256(
            json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "recording_formats": {
            "mp4": bool(mp4_artifacts),
            "rrd": False,
            "mcap": False,
        },
        "mp4_artifacts": mp4_artifacts,
        "validation_scope": "runtime artifact provenance; no GPU validation claim",
    }


def upload_output(local_dir: Path, output_uri: str, result: dict[str, Any]) -> None:
    """Upload a capability's local output tree to S3 when the run produced one.

    Capabilities that write artifacts (rollouts, trajectory exports, policy
    evaluation) publish their output directory to ``output_uri`` so downstream
    workflow stages can read it from S3. Capabilities that only return a result
    dict (task registration, asset availability) have nothing to upload.
    """
    if not output_uri:
        return
    root = Path(local_dir)
    if not root.exists() or not any(root.iterdir()):
        return
    import boto3

    endpoint = os.environ.get("AWS_ENDPOINT_URL") or os.environ.get(
        "NEBIUS_S3_ENDPOINT", ""
    )
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint or None,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID") or None,
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY") or None,
    )
    bucket, prefix = parse_s3_uri(output_uri)
    for file_path in sorted(root.rglob("*")):
        if file_path.is_file():
            rel = file_path.relative_to(root)
            s3.upload_file(str(file_path), bucket, f"{prefix}/{rel}")
    result["output_uri"] = output_uri


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse an s3:// URI into (bucket, prefix)."""
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3:// URI: {uri}")
    rest = uri[len("s3://") :]
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix.rstrip("/")


__all__ = [
    "SUPPORTED_CAPABILITIES",
    "RoboCasaError",
    "compute_manifest_sha256",
    "kitchen_asset_availability",
    "kitchen_egl_env_reset",
    "kitchen_random_rollout",
    "kitchen_task_registration",
    "kitchen_trajectory_export",
    "kitchen_policy_eval",
    "make_run_id",
    "parse_s3_uri",
    "run_capability",
    "run_capability_with_output",
    "system_info",
    "upload_output",
    "verify_runtime_identity",
]
