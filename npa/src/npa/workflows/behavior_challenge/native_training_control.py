"""Dispatch full milestone publication through one storage-capable process."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from npa.workflows.behavior_challenge.native_training_checkpoint import (
    atomic_json,
    checkpoint_inventory,
    file_identity,
)

REQUEST_SCHEMA = "npa.behavior.comet-native-storage-request.v1"
RESULT_SCHEMA = "npa.behavior.comet-native-storage-result.v1"


def _control_paths() -> tuple[Path, Path]:
    python = Path(os.environ.get("NPA_STORAGE_CONTROL_PYTHON", ""))
    worker = Path(os.environ.get("NPA_STORAGE_CONTROL_WORKER", ""))
    if not python.is_absolute() or not python.is_file():
        raise ValueError("storage control Python differs")
    if not worker.is_absolute() or worker.is_symlink() or not worker.is_file():
        raise ValueError("storage control worker differs")
    return python, worker


def _environment() -> dict[str, str]:
    value = dict(os.environ)
    value.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "JAX_PLATFORM_NAME": "cpu",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.environ["NPA_STORAGE_CONTROL_PYTHONPATH"],
        }
    )
    return value


def _request(
    checkpoint: Path, receipt: Path, prefix: str, scratch: Path
) -> dict[str, Any]:
    return {
        "schema": REQUEST_SCHEMA,
        "checkpoint_root": str(checkpoint.resolve()),
        "receipt_path": str(receipt.resolve()),
        "receipt": file_identity(receipt),
        "checkpoint": checkpoint_inventory(checkpoint),
        "prefix": prefix,
        "scratch": str(scratch.resolve()),
        "logical_update_count": int(
            json.loads(receipt.read_text())["logical_update_count"]
        ),
    }


def _invoke(python: Path, worker: Path, request: Path, result: Path) -> None:
    done = subprocess.run(
        [
            str(python),
            "-B",
            str(worker),
            "publish",
            "--request",
            str(request),
            "--output",
            str(result),
        ],
        check=False,
        env=_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if done.returncode:
        raise RuntimeError(
            f"storage control publication failed: {done.returncode}\n{done.stdout}"
        )


def _verify_result(
    result: dict[str, Any], request: dict[str, Any], identity: dict[str, Any]
) -> None:
    expected = {
        "request": identity,
        "receipt": request["receipt"],
        "checkpoint": request["checkpoint"],
        "prefix": request["prefix"],
        "logical_update_count": request["logical_update_count"],
    }
    if result.get("schema") != RESULT_SCHEMA or result.get("binding") != expected:
        raise ValueError("storage control result binding differs")
    payload = (
        json.dumps(result.get("manifest"), indent=2, sort_keys=True) + "\n"
    ).encode()
    provider = result.get("provider_manifest", {})
    target = f"{request['prefix'].rstrip('/')}/output-manifest.json"
    if provider != {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "uri": target,
        "provider_readback": True,
    }:
        raise ValueError("storage control manifest readback differs")


def publish_with_control(
    checkpoint_root: Path, receipt_path: Path, prefix: str, scratch: Path
) -> dict[str, Any]:
    """Publish one milestone in a control subprocess.

    Args: checkpoint_root: Complete Orbax root. receipt_path: Immutable receipt.
        prefix: Canonical provider prefix. scratch: Transaction directory.
    Returns: Bound child result after provider readback verification.
    Raises: ValueError for unsafe evidence; RuntimeError for child failure.
    """
    python, worker = _control_paths()
    request = _request(checkpoint_root, receipt_path, prefix, scratch)
    step = request["logical_update_count"]
    request_path = scratch / f"publish-step-{step:05d}-request.json"
    result_path = scratch / f"publish-step-{step:05d}-result.json"
    if request_path.is_symlink() or result_path.is_symlink():
        raise ValueError("storage control transaction is unsafe")
    atomic_json(request_path, request)
    identity = file_identity(request_path)
    if not result_path.exists():
        _invoke(python, worker, request_path, result_path)
    if result_path.is_symlink() or not result_path.is_file():
        raise ValueError("storage control result is absent")
    result = json.loads(result_path.read_text())
    _verify_result(result, request, identity)
    if checkpoint_inventory(checkpoint_root) != request["checkpoint"]:
        raise ValueError("checkpoint changed during storage control operation")
    return result
