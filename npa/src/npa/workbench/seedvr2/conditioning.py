"""Materialize the pinned 3B configuration without modifying upstream source."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
from typing import Any

import yaml

SOURCE_MAIN_SHA256 = "2606dcfde7544c3b69686b633411fcf6f37a4c159182596c4ce8347537d22f35"


def _configuration_bytes(source_root: Path, mode: str) -> tuple[bytes, bytes]:
    if mode not in {"sample", "posterior-mode"}:
        raise ValueError("unsupported SeedVR2 conditioning mode")
    original = (source_root / "configs_3b/main.yaml").read_bytes()
    if hashlib.sha256(original).hexdigest() != SOURCE_MAIN_SHA256:
        raise ValueError("SeedVR2 upstream 3B configuration hash mismatch")
    document = yaml.safe_load(original)
    if not isinstance(document, dict) or not isinstance(document.get("vae"), dict):
        raise ValueError("SeedVR2 upstream VAE configuration is invalid")
    if "use_sample" in document["vae"]:
        raise ValueError("SeedVR2 upstream conditioning default changed")
    if mode == "sample":
        return original, original
    document["vae"]["use_sample"] = False
    return original, yaml.safe_dump(document, sort_keys=False).encode()


def _config_files(root: Path) -> dict[str, bytes]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("SeedVR2 configuration root must be a real directory")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("SeedVR2 configuration must contain only regular files")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path.read_bytes()
    return files


def configuration_identity(source_root: Path, mode: str) -> dict[str, Any]:
    """Return exact expected config identities, not an execution attestation.

    Args:
        source_root: Immutable pinned upstream tree.
        mode: Explicit sample or posterior-mode selection.
    Returns:
        Expected source/effective configuration hashes.
    Raises:
        ValueError: Source or selection violates the pinned contract.
        OSError: Configuration bytes cannot be read.
    """
    original, effective = _configuration_bytes(source_root, mode)
    files = _config_files(source_root / "configs_3b")
    files["main.yaml"] = effective
    return {
        "conditioning_mode": mode,
        "vae_use_sample": mode == "sample",
        "source_main_sha256": hashlib.sha256(original).hexdigest(),
        "effective_main_sha256": hashlib.sha256(effective).hexdigest(),
        "effective_files": {
            name: hashlib.sha256(data).hexdigest() for name, data in files.items()
        },
        "scope": "configuration_bytes_not_execution_attestation",
    }


def prepare_configuration(source_root: Path, workspace: Path, mode: str) -> None:
    """Copy regular config files and change only the explicit VAE sampling flag.

    Args:
        source_root: Immutable pinned upstream tree.
        workspace: Owned directory whose config copy must not exist.
        mode: Explicit sample or posterior-mode selection.
    Returns:
        None.
    Raises:
        ValueError: Configuration identity is invalid.
        OSError: The owned copy cannot be created.
    """
    _, effective = _configuration_bytes(source_root, mode)
    _config_files(source_root / "configs_3b")
    target = workspace / "configs_3b"
    shutil.copytree(source_root / "configs_3b", target)
    (target / "main.yaml").write_bytes(effective)
    verify_configuration(source_root, workspace, mode)


def verify_configuration(source_root: Path, workspace: Path, mode: str) -> None:
    """Reject changed configuration bytes before or after inference.

    Args:
        source_root: Immutable pinned upstream tree.
        workspace: Owned inference working directory.
        mode: Expected conditioning selection.
    Returns:
        None.
    Raises:
        ValueError: Configuration bytes differ from the expected selection.
        OSError: Configuration bytes cannot be read.
    """
    expected = configuration_identity(source_root, mode)["effective_files"]
    actual = {
        name: hashlib.sha256(data).hexdigest()
        for name, data in _config_files(workspace / "configs_3b").items()
    }
    if actual != expected:
        raise ValueError("SeedVR2 workspace configuration changed")
