"""Publish a complete native milestone from the storage-capable control runtime."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from npa.clients.storage import StorageClient
from npa.workflows.behavior_challenge.checkpoint_publication import (
    publish_immutable_checkpoint_files,
)
from npa.workflows.behavior_challenge.native_training_checkpoint import (
    SCHEMA,
    STATUS,
    atomic_json,
    checkpoint_inventory,
    file_identity,
)
from npa.workflows.behavior_challenge.native_training_control import (
    REQUEST_SCHEMA,
    RESULT_SCHEMA,
)


def _contained(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _publish_manifest(
    storage: Any, manifest: dict[str, Any], uri: str, scratch: Path
) -> dict[str, Any]:
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    try:
        storage.put_bytes_conditional(payload, uri, if_none_match=True)
    except Exception as error:
        if type(error).__name__ != "StoragePreconditionFailed":
            raise
    target = scratch / "manifest-readback.json"
    storage.download_file(uri, str(target))
    if target.read_bytes() != payload:
        raise ValueError("provider manifest readback differs")
    target.unlink()
    return {**file_identity_bytes(payload), "uri": uri, "provider_readback": True}


def file_identity_bytes(payload: bytes) -> dict[str, Any]:
    """Return an in-memory identity.

    Args: payload: Exact bytes. Returns: Byte count and SHA-256.
    Raises: None.
    """
    import hashlib

    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _validated_request(request_path: Path) -> tuple[dict[str, Any], Path, Path, Path]:
    request_identity = file_identity(request_path)
    request = json.loads(request_path.read_text())
    checkpoint = Path(request["checkpoint_root"])
    receipt = Path(request["receipt_path"])
    scratch = Path(request["scratch"])
    allowed = Path(os.environ["NPA_STORAGE_ALLOWED_CHECKPOINT_ROOT"])
    allowed_prefix = os.environ["NPA_STORAGE_ALLOWED_OUTPUT_PREFIX"].rstrip("/")
    if (
        request.get("schema") != REQUEST_SCHEMA
        or not _contained(checkpoint, allowed)
        or checkpoint.parent != allowed
        or receipt.parent != allowed
        or not _contained(scratch, allowed)
    ):
        raise ValueError("storage control scope differs")
    if (
        request["prefix"]
        != f"{allowed_prefix}/milestones/step-{json.loads(receipt.read_text())['logical_update_count']:05d}"
    ):
        raise ValueError("storage control prefix differs")
    inventory = checkpoint_inventory(checkpoint)
    if file_identity(receipt) != request.get("receipt"):
        raise ValueError("milestone receipt identity differs")
    if inventory != request.get("checkpoint"):
        raise ValueError("checkpoint inventory differs")
    request["_identity"] = request_identity
    return request, checkpoint, receipt, scratch


def _published_files(
    storage: Any,
    request: dict[str, Any],
    checkpoint: Path,
    receipt: Path,
    scratch: Path,
) -> dict[str, Any]:
    inventory = request["checkpoint"]
    files = {
        f"checkpoint/{row['path']}": checkpoint / row["path"]
        for row in inventory["files"]
    }
    files["milestone-receipt.json"] = receipt
    expected = {name: file_identity(path) for name, path in files.items()}
    published = publish_immutable_checkpoint_files(
        files,
        f"{request['prefix']}/originals",
        scratch,
        storage_factory=StorageClient.from_environment,
        expected=expected,
    )
    return published


def _milestone_manifest(
    request: dict[str, Any], receipt: Path, published: dict[str, Any]
) -> dict[str, Any]:
    record = json.loads(receipt.read_text())
    step = record["logical_update_count"]
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "outcome": "success",
        "logical_update_count": step,
        "manager_step": step - 1,
        "checkpoint": request["checkpoint"],
        "serving_params": record["serving_params"],
        "cursor": record["cursor"],
        "files": published,
        "full_state_resume_ready": True,
        "serving_export_qualified": False,
        "scoring_executed": False,
        "admission": record["admission"],
        "workflow_inputs": record["workflow_inputs"],
        "static_reconstruction": record["static_reconstruction"],
    }


def publish(request_path: Path, output: Path) -> None:
    """Publish one validated milestone.

    Args: request_path: Immutable control request. output: New result path.
    Returns: None.
    Raises: ValueError when scope, bytes, or provider readback differ.
    """
    request, checkpoint, receipt, scratch = _validated_request(request_path)
    storage = StorageClient.from_environment()
    published = _published_files(storage, request, checkpoint, receipt, scratch)
    manifest = _milestone_manifest(request, receipt, published)
    step = request["logical_update_count"]
    provider = _publish_manifest(
        storage, manifest, f"{request['prefix']}/output-manifest.json", scratch
    )
    binding = {
        "request": request.pop("_identity"),
        "receipt": request["receipt"],
        "checkpoint": request["checkpoint"],
        "prefix": request["prefix"],
        "logical_update_count": step,
    }
    atomic_json(
        output,
        {
            "schema": RESULT_SCHEMA,
            "binding": binding,
            "manifest": manifest,
            "provider_manifest": provider,
        },
    )


def main() -> None:
    """Run one whole publication operation.

    Args: None. Returns: None.
    Raises: Propagates argument, scope, storage, and readback errors.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("publish",))
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    publish(args.request, args.output)


if __name__ == "__main__":
    main()
