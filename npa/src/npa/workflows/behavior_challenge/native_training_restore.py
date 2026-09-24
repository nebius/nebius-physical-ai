"""Restore the newest complete provider-read native training milestone."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from npa.workflows.behavior_challenge.native_training_checkpoint import (
    atomic_json,
    checkpoint_inventory,
    file_identity,
    validate_complete_checkpoint,
    validate_full_state_contract,
    validate_manifest,
)


def _download_manifest(storage: Any, uri: str) -> tuple[dict[str, Any], bytes] | None:
    observed = storage.read_bytes_with_etag(uri)
    if observed is None:
        return None
    payload, _etag = observed
    return json.loads(payload), payload


def _required_bytes(manifest: dict[str, Any]) -> int:
    return sum(int(row["bytes"]) for row in manifest["files"].values())


def _transaction_binding(manifest: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema": "npa.behavior.comet-native-restore-transaction.v1",
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _safe_transaction(transaction: Path, manifest: dict[str, Any]) -> None:
    if transaction.is_symlink() or (transaction.exists() and not transaction.is_dir()):
        raise ValueError("restore transaction is unsafe")
    transaction.mkdir(exist_ok=True)
    binding = transaction / "restore-binding.json"
    atomic_json(binding, _transaction_binding(manifest))
    expected = set(manifest["files"]) | {"restore-binding.json"}
    expected |= {name + ".download" for name in manifest["files"]}
    for path in transaction.rglob("*"):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise ValueError("restore transaction contains an unsafe member")
        if path.is_file() and path.relative_to(transaction).as_posix() not in expected:
            raise ValueError("restore transaction contains an unexpected member")


def _checkpoint_modes(manifest: dict[str, Any]) -> dict[str, int]:
    modes: dict[str, int] = {}
    for row in manifest["checkpoint"]["files"]:
        value = row.get("mode")
        if not isinstance(value, str) or re.fullmatch(r"0o[0-7]{3}", value) is None:
            raise ValueError("checkpoint member mode differs")
        name = "checkpoint/" + row["path"]
        if name in modes:
            raise ValueError("checkpoint member mode is duplicated")
        modes[name] = int(value, 8)
    return modes


def _download_files(storage: Any, manifest: dict[str, Any], transaction: Path) -> None:
    _safe_transaction(transaction, manifest)
    checkpoint_modes = _checkpoint_modes(manifest)
    for name, row in manifest["files"].items():
        destination = transaction / name
        expected = {key: row[key] for key in ("bytes", "sha256")}
        if destination.exists():
            if file_identity(destination) != expected:
                raise ValueError("completed restore member differs")
            if name in checkpoint_modes:
                destination.chmod(checkpoint_modes[name])
            continue
        partial = transaction / (name + ".download")
        partial.parent.mkdir(parents=True, exist_ok=True)
        if partial.exists():
            if partial.is_symlink() or not partial.is_file():
                raise ValueError("partial restore member is unsafe")
            partial.unlink()
        storage.download_file(row["uri"], str(partial))
        if file_identity(partial) != expected:
            raise ValueError("provider milestone member differs")
        if name in checkpoint_modes:
            partial.chmod(checkpoint_modes[name])
        partial.replace(destination)


def _validate_receipt(
    record: dict[str, Any],
    manifest: dict[str, Any],
    milestones: tuple[int, ...],
    step: int,
    admission: dict[str, Any],
    static: dict[str, Any],
    workflow_inputs: dict[str, Any],
) -> dict[str, Any]:
    expected_prior = [value for value in milestones if value < step]
    prior = record.get("durable_milestones_before")
    validate_full_state_contract(record.get("train_state", {}), step)
    if (
        record.get("checkpoint") != manifest["checkpoint"]
        or record.get("cursor") != manifest["cursor"]
        or record.get("admission") != admission
        or record.get("static_reconstruction") != static
        or record.get("workflow_inputs") != workflow_inputs
        or record.get("full_state_resume_ready") is not True
        or record.get("train_state", {}).get("step") != step
        or not isinstance(prior, dict)
        or sorted(int(value) for value in prior) != expected_prior
    ):
        raise ValueError("milestone receipt state/admission/history differs")
    return prior


def _member_matches(path: Path, row: dict[str, Any]) -> bool:
    return (
        path.is_file()
        and not path.is_symlink()
        and file_identity(path) == {key: row[key] for key in ("bytes", "sha256")}
    )


def _finish_restore_moves(
    transaction: Path, target: Path, receipt: Path, manifest: dict[str, Any]
) -> tuple[Path, Path] | None:
    staged_receipt = transaction / "milestone-receipt.json"
    receipt_row = manifest["files"]["milestone-receipt.json"]
    target_ready = (
        target.is_dir()
        and not target.is_symlink()
        and checkpoint_inventory(target) == manifest["checkpoint"]
    )
    receipt_ready = _member_matches(receipt, receipt_row)
    staged_ready = _member_matches(staged_receipt, receipt_row)
    if target_ready and not receipt.exists() and staged_ready:
        staged_receipt.replace(receipt)
        receipt_ready = True
    if target_ready and receipt_ready:
        binding = transaction / "restore-binding.json"
        if binding.exists():
            binding.unlink()
        if transaction.exists():
            transaction.rmdir()
        return target, receipt
    if target.exists() or receipt.exists():
        raise ValueError("local durable milestone conflicts with provider state")
    return None


def _local_or_restore(
    storage: Any, manifest: dict[str, Any], checkpoint_base: Path, step: int
) -> tuple[Path, Path]:
    target = checkpoint_base / f"step-{step:05d}"
    receipt = checkpoint_base / f"step-{step:05d}.json"
    transaction = checkpoint_base / f".restore-step-{step:05d}.partial"
    if transaction.exists():
        _safe_transaction(transaction, manifest)
    if target.is_dir() and not target.is_symlink() and not receipt.exists():
        if checkpoint_inventory(target) != manifest["checkpoint"]:
            raise ValueError("local durable checkpoint differs")
        _safe_transaction(transaction, manifest)
        row = manifest["files"]["milestone-receipt.json"]
        partial = transaction / "milestone-receipt.json.download"
        if partial.exists():
            if partial.is_symlink() or not partial.is_file():
                raise ValueError("partial receipt restore is unsafe")
            partial.unlink()
        storage.download_file(row["uri"], str(partial))
        if not _member_matches(partial, row):
            raise ValueError("provider milestone receipt differs")
        partial.replace(transaction / "milestone-receipt.json")
    recovered = _finish_restore_moves(transaction, target, receipt, manifest)
    if recovered is not None:
        return recovered
    if shutil.disk_usage(checkpoint_base).free < _required_bytes(manifest):
        raise ValueError("checkpoint restore free-space gate failed")
    _download_files(storage, manifest, transaction)
    restored = transaction / "checkpoint"
    if checkpoint_inventory(restored) != manifest["checkpoint"]:
        raise ValueError("restored checkpoint inventory differs")
    restored.replace(target)
    recovered = _finish_restore_moves(transaction, target, receipt, manifest)
    if recovered is None:
        raise ValueError("restore transaction did not produce a complete milestone")
    return recovered


def _restore_manifest(
    storage: Any, manifest: dict[str, Any], payload: bytes, expected: dict[str, Any]
) -> dict[str, Any]:
    step = expected["step"]
    prefix = expected["prefix"]
    validate_manifest(manifest, step=step, prefix=prefix)
    validate_complete_checkpoint(manifest["checkpoint"], step - 1)
    if (
        manifest.get("admission") != expected["admission"]
        or manifest.get("static_reconstruction") != expected["static"]
        or manifest.get("workflow_inputs") != expected["inputs"]
    ):
        raise ValueError("provider milestone admission/static/workflow inputs differ")
    target, receipt = _local_or_restore(storage, manifest, expected["base"], step)
    record = json.loads(receipt.read_text())
    prior = _validate_receipt(
        record,
        manifest,
        expected["milestones"],
        step,
        expected["admission"],
        expected["static"],
        expected["inputs"],
    )
    provider = _provider_identity(payload, expected["uri"])
    durable = dict(prior)
    durable[str(step)] = {**record, "provider_manifest": provider}
    return {
        "schema": "npa.behavior.comet-native-resume-selection.v1",
        "logical_update": step,
        "resume_checkpoint": str(target),
        "resume_receipt": str(receipt),
        "cursor": manifest["cursor"],
        "durable_milestones": durable,
        "provider_manifest": provider,
    }


def _provider_identity(payload: bytes, uri: str) -> dict[str, Any]:
    return {
        "uri": uri,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "provider_readback": True,
    }


def _find_latest(storage: Any, expected: dict[str, Any]) -> dict[str, Any]:
    for step in reversed(expected["milestones"][1:]):
        uri = f"{expected['prefix'].rstrip('/')}/milestones/step-{step:05d}/output-manifest.json"
        observed = _download_manifest(storage, uri)
        if observed is None:
            continue
        manifest, payload = observed
        values = {**expected, "uri": uri, "step": step}
        return _restore_manifest(storage, manifest, payload, values)
    return {
        "schema": "npa.behavior.comet-native-resume-selection.v1",
        "logical_update": 0,
        "resume_checkpoint": None,
        "resume_receipt": None,
        "cursor": {"epoch": 0, "batch_offset": 0, "committed_global_batch": 0},
        "durable_milestones": {},
    }


def restore_latest(
    storage: Any,
    *,
    prefix: str,
    milestones: tuple[int, ...],
    checkpoint_base: Path,
    selection: Path,
    expected_admission: dict[str, Any],
    expected_static_reconstruction: dict[str, Any],
    expected_workflow_inputs: dict[str, Any],
) -> dict[str, Any]:
    """Restore the newest exact milestone.

    Args: storage: Read-only provider client. prefix: Run prefix. milestones:
        Ordered schedule. checkpoint_base: Local durable root. selection: Output.
        expected_admission: Admission identity. expected_static_reconstruction:
        Exact recipe. expected_workflow_inputs: Complete input contract.
    Returns: A resume selection, or released-parent cursor zero.
    Raises: ValueError when any local/provider/input contract differs.
    """
    checkpoint_base.mkdir(parents=True, exist_ok=True)
    expected = {
        "prefix": prefix,
        "milestones": milestones,
        "base": checkpoint_base,
        "admission": expected_admission,
        "static": expected_static_reconstruction,
        "inputs": expected_workflow_inputs,
    }
    result = _find_latest(storage, expected)
    atomic_json(selection, result)
    return result


def main() -> None:
    """Run restore selection under the control interpreter.

    Args: None. Returns: None.
    Raises: Propagates argument, provider, inventory, and restore errors.
    """
    import argparse

    from npa.clients.storage import StorageClient

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--milestones", required=True)
    parser.add_argument("--checkpoint-base", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--workflow-input-contract", type=Path, required=True)
    args = parser.parse_args()
    admission = json.loads(args.admission.read_text())
    result = restore_latest(
        StorageClient.from_environment(),
        prefix=args.prefix,
        milestones=tuple(int(value) for value in args.milestones.split(",")),
        checkpoint_base=args.checkpoint_base,
        selection=args.selection,
        expected_admission={
            "bytes": args.admission.stat().st_size,
            "sha256": hashlib.sha256(args.admission.read_bytes()).hexdigest(),
        },
        expected_static_reconstruction=admission["static_reconstruction"],
        expected_workflow_inputs=json.loads(args.workflow_input_contract.read_text()),
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
