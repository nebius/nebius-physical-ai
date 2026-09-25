"""Fail-closed identity checks for the baked Isaac Arena runtime."""

from __future__ import annotations

from importlib import metadata
import json
import os
from pathlib import Path
from typing import Mapping

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the Python 3.10 CI lane
    import tomli as tomllib

from .errors import IsaacArenaError
from .identity import (
    ISAAC_ARENA_ARCHIVE_SHA256,
    ISAAC_ARENA_REVISION,
    ISAAC_ARENA_ROOT,
    ISAAC_ARENA_VERSION,
    LIGHTWHEEL_SDK_VERSION,
    SOURCE_IDENTITY_SCHEMA,
)


def _source_identity(root: Path) -> dict[str, str]:
    path = root / ".npa-source-identity.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IsaacArenaError(
            "Arena source identity metadata is missing or invalid"
        ) from exc
    if not isinstance(payload, dict):
        raise IsaacArenaError("Arena source identity metadata must be an object")
    return {str(key): str(value) for key, value in payload.items()}


def _pyproject_version(root: Path) -> str:
    try:
        payload = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        return str(payload["project"]["version"])
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as exc:
        raise IsaacArenaError(
            "Arena installed source has no readable project version"
        ) from exc


def assert_runtime_identity(
    *, root: Path | None = None, environment: Mapping[str, str] | None = None
) -> dict[str, object]:
    """Assert baked source and installed SDK identities when Arena is present.

    Args:
        root: Arena source root; defaults to the image's immutable location.
        environment: Process environment used for image identity labels.
    Returns:
        Sanitized identity evidence, or an explicit local-development skip.
    Raises:
        IsaacArenaError: A baked runtime identity is absent or disagrees with its pin.
    """

    env = environment if environment is not None else os.environ
    source_root = root or Path(env.get("ISAAC_ARENA_ROOT", ISAAC_ARENA_ROOT))
    if not source_root.is_dir():
        if env.get("NPA_LIGHT_WORKBENCH_TOOL") == "isaac-arena":
            raise IsaacArenaError(
                "Arena image marker is set but its source root is missing"
            )
        return {"asserted": False, "reason": "baked_source_absent"}
    source = _source_identity(source_root)
    expected = {
        "schema": SOURCE_IDENTITY_SCHEMA,
        "version": ISAAC_ARENA_VERSION,
        "revision": ISAAC_ARENA_REVISION,
        "archive_sha256": ISAAC_ARENA_ARCHIVE_SHA256,
    }
    if source != expected or _pyproject_version(source_root) != ISAAC_ARENA_VERSION:
        raise IsaacArenaError(
            "Arena installed source identity does not match the pinned release"
        )
    if env.get("ISAAC_ARENA_REVISION") != ISAAC_ARENA_REVISION:
        raise IsaacArenaError(
            "ISAAC_ARENA_REVISION does not match the installed source"
        )
    if env.get("ISAAC_ARENA_VERSION") != ISAAC_ARENA_VERSION:
        raise IsaacArenaError("ISAAC_ARENA_VERSION does not match the installed source")
    try:
        lightwheel_version = metadata.version("lightwheel-sdk")
    except metadata.PackageNotFoundError as exc:
        raise IsaacArenaError(
            "the pinned Lightwheel SDK distribution is not installed"
        ) from exc
    if lightwheel_version != LIGHTWHEEL_SDK_VERSION:
        raise IsaacArenaError("installed Lightwheel SDK version does not match the pin")
    return {
        "asserted": True,
        "arena_version": ISAAC_ARENA_VERSION,
        "arena_revision": ISAAC_ARENA_REVISION,
        "arena_archive_sha256": ISAAC_ARENA_ARCHIVE_SHA256,
        "lightwheel_sdk_version": lightwheel_version,
    }
