"""Inventory and validate complete native OpenPI training milestones."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "npa.behavior.comet-native-training-milestone.v1"
STATUS = "complete_full_state_and_serving_params_provider_readback"


def file_identity(path: Path) -> dict[str, int | str]:
    """Return a file identity.

    Args: path: Regular file to stream.
    Returns: Byte count and SHA-256.
    Raises: ValueError for a missing, linked, or non-file path.
    """
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required regular file differs: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {"bytes": size, "sha256": digest.hexdigest()}


def checkpoint_inventory(root: Path) -> dict[str, Any]:
    """Hash every checkpoint member.

    Args: root: Checkpoint manager root.
    Returns: Canonical complete member inventory.
    Raises: ValueError for unsafe or empty trees.
    """
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("checkpoint contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("checkpoint contains a special member")
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                **file_identity(path),
                "mode": oct(path.stat().st_mode & 0o777),
            }
        )
    if not rows:
        raise ValueError("checkpoint inventory is empty")
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {
        "files": rows,
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "max_member_bytes": max(row["bytes"] for row in rows),
        "content_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def serving_params(checkpoint: dict[str, Any], manager_step: int) -> dict[str, Any]:
    """Select dense serving parameters.

    Args: checkpoint: Full inventory. manager_step: Native manager index.
    Returns: FP32 parameter subset inventory.
    Raises: ValueError when parameters are absent.
    """
    root = f"{manager_step}/params"
    rows = [row for row in checkpoint["files"] if row["path"].startswith(root + "/")]
    if not rows:
        raise ValueError("checkpoint has no native serving parameters")
    return {
        "root": root,
        "dtype": "float32_native_training_state",
        "bf16_serving_derivative_created": False,
        "files": rows,
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
    }


def validate_complete_checkpoint(checkpoint: dict[str, Any], manager_step: int) -> None:
    """Require parameter and optimizer items.

    Args: checkpoint: Full inventory. manager_step: Native manager index.
    Returns: None.
    Raises: ValueError when parameters or TrainState are absent.
    """
    paths = [row["path"] for row in checkpoint.get("files", [])]
    root = f"{manager_step}/"
    if not any(path.startswith(root + "params/") for path in paths):
        raise ValueError("native checkpoint parameters are absent")
    if not any(path.startswith(root + "train_state/") for path in paths):
        raise ValueError("native checkpoint optimizer/TrainState is absent")


def _validate_array_inventory(value: Any, label: str) -> dict[str, Any]:
    leaves = value.get("leaves") if isinstance(value, dict) else None
    if not isinstance(leaves, dict) or not leaves:
        raise ValueError(f"{label} leaf inventory differs")
    for path, row in leaves.items():
        if not isinstance(path, str) or not path or not isinstance(row, dict):
            raise ValueError(f"{label} leaf path differs")
        if (
            not isinstance(row.get("shape"), list)
            or not isinstance(row.get("dtype"), str)
            or isinstance(row.get("elements"), bool)
            or not isinstance(row.get("elements"), int)
            or row["elements"] <= 0
            or isinstance(row.get("bytes"), bool)
            or not isinstance(row.get("bytes"), int)
            or row["bytes"] <= 0
            or not isinstance(row.get("sha256"), str)
            or len(row["sha256"]) != 64
        ):
            raise ValueError(f"{label} leaf identity differs")
    encoded = json.dumps(leaves, separators=(",", ":"), sort_keys=True).encode()
    expected = {
        "row_count": len(leaves),
        "element_count": sum(row["elements"] for row in leaves.values()),
        "total_bytes": sum(row["bytes"] for row in leaves.values()),
        "content_sha256": hashlib.sha256(encoded).hexdigest(),
    }
    if any(value.get(name) != observed for name, observed in expected.items()):
        raise ValueError(f"{label} inventory summary differs")
    return leaves


def _validate_moment_paths(
    params: dict[str, Any], mu: dict[str, Any], nu: dict[str, Any]
) -> None:
    if set(mu) != set(params) or set(nu) != set(params):
        raise ValueError("AdamW moment paths do not match parameter paths")
    for path, row in params.items():
        signature = (row["shape"], row["elements"], row["bytes"])
        observed_mu = (mu[path]["shape"], mu[path]["elements"], mu[path]["bytes"])
        observed_nu = (nu[path]["shape"], nu[path]["elements"], nu[path]["bytes"])
        if observed_mu != signature or observed_nu != signature:
            raise ValueError("AdamW moment shape does not match parameter path")


def validate_full_state_contract(value: dict[str, Any], step: int) -> None:
    """Validate a value-hashed complete FP32/AdamW state contract.

    Args: value: Recorded TrainState contract. step: Expected logical step.
    Returns: None.
    Raises: ValueError when values, moments, or progress are incomplete.
    """
    if value.get("schema") != "npa.behavior.comet-native-full-train-state.v1":
        raise ValueError("full native state schema differs")
    if value.get("step") != step or value.get("ema_params_present") is not False:
        raise ValueError("full native state chronology/EMA differs")
    params = _validate_array_inventory(value.get("params"), "parameter")
    mu = _validate_array_inventory(value.get("adamw_mu"), "AdamW mu")
    nu = _validate_array_inventory(value.get("adamw_nu"), "AdamW nu")
    _validate_array_inventory(value.get("optimizer_state"), "optimizer state")
    _validate_moment_paths(params, mu, nu)
    if any(row["dtype"] != "float32" for row in params.values()):
        raise ValueError("full native TrainState parameters must remain FP32")
    counts = value.get("optimizer_scalar_progress")
    if (
        not isinstance(counts, dict)
        or not counts
        or any(count != step for count in counts.values())
    ):
        raise ValueError("optimizer scalar progress differs")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Create immutable JSON.

    Args: path: Destination. value: JSON object.
    Returns: None.
    Raises: ValueError when existing bytes differ.
    """
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"immutable JSON conflict: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def canonical_member(name: str) -> str:
    """Validate one provider member name.

    Args: name: Relative POSIX member name.
    Returns: The unchanged canonical name.
    Raises: ValueError for traversal or noncanonical syntax.
    """
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or path.as_posix() != name
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("milestone member name is not canonical")
    return name


def _validate_provider_file(name: str, row: dict[str, Any], base: str) -> None:
    canonical_member(name)
    size = row.get("bytes")
    digest = row.get("sha256")
    if (
        row.get("uri") != base + name
        or row.get("provider_readback") is not True
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
    ):
        raise ValueError("milestone provider destination differs")


def validate_manifest(value: dict[str, Any], *, step: int, prefix: str) -> None:
    """Validate a provider-read milestone.

    Args: value: Manifest object. step: Logical update. prefix: Run prefix.
    Returns: None.
    Raises: ValueError for schema, chronology, inventory, or URI drift.
    """
    if value.get("schema") != SCHEMA or value.get("status") != STATUS:
        raise ValueError("milestone schema or status differs")
    if (
        value.get("logical_update_count") != step
        or value.get("manager_step") != step - 1
    ):
        raise ValueError("milestone chronology differs")
    if value.get("full_state_resume_ready") is not True:
        raise ValueError("milestone is not resume-ready")
    rows = value.get("checkpoint", {}).get("files")
    files = value.get("files")
    if not isinstance(rows, list) or not rows or not isinstance(files, dict):
        raise ValueError("milestone inventory is absent")
    expected = {
        "milestone-receipt.json",
        *(f"checkpoint/{row['path']}" for row in rows),
    }
    if set(files) != expected:
        raise ValueError("milestone provider/file inventory differs")
    base = f"{prefix.rstrip('/')}/milestones/step-{step:05d}/originals/"
    for name, row in files.items():
        _validate_provider_file(name, row, base)
