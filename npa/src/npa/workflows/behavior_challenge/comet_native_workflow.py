"""Materialize declared native-training inputs and invoke control/science runtimes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

from npa.clients.storage import StorageClient
from npa.workflows.behavior_challenge.native_training_checkpoint import (
    checkpoint_inventory,
    file_identity,
    validate_complete_checkpoint,
    validate_full_state_contract,
    validate_manifest,
)
from npa.workflows.behavior_challenge.native_training_control import (
    publish_with_control,
)
from npa.workflows.behavior_challenge.runtime_cache import prepare_runtime

STRING_ARGUMENTS = (
    "prefix",
    "openpi-uri",
    "openpi-sha256",
    "worker-uri",
    "worker-sha256",
    "dataset-uri",
    "dataset-sha256",
    "parent-uri",
    "parent-sha256",
    "admission-uri",
    "admission-sha256",
    "runtime-manifest-uri",
    "runtime-manifest-sha256",
    "config-name",
    "split-sha256",
    "milestones",
    "dataset-relative",
    "split-relative",
    "preflight-uri",
    "scientific-python-relative",
    "scientific-python-link-target",
    "scientific-python-base-relative",
    "scientific-python-base-sha256",
    "scientific-runtime-receipt-relative",
    "scientific-runtime-receipt-sha256",
    "attempt-id",
)
INTEGER_ARGUMENTS = (
    "openpi-bytes",
    "worker-bytes",
    "dataset-bytes",
    "parent-bytes",
    "admission-bytes",
    "final-step",
    "scientific-python-base-bytes",
    "scientific-runtime-receipt-bytes",
)


def _expected(path: Path, sha256: str, size: int) -> None:
    if file_identity(path) != {"bytes": size, "sha256": sha256}:
        raise ValueError(f"downloaded input identity differs: {path.name}")


def _download(storage: Any, uri: str, target: Path, sha256: str, size: int) -> None:
    if target.exists() or target.is_symlink():
        raise ValueError("input target must be new")
    storage.download_file(uri, str(target))
    _expected(target, sha256, size)


def _tar_name(name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError("archive member is unsafe")
    return Path(*relative.parts)


def _extract(archive: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        raise ValueError("archive target must be new")
    target.mkdir()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            if not member.isfile() and not member.isdir():
                raise ValueError("archive has a non-file member")
            destination = target / _tar_name(member.name)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            stream = bundle.extractfile(member)
            if stream is None:
                raise ValueError("archive member payload is absent")
            with destination.open("xb") as output:
                while chunk := stream.read(8 * 1024 * 1024):
                    output.write(chunk)
            destination.chmod(member.mode & 0o777)


def _root(directory: Path) -> Path:
    children = list(directory.iterdir())
    return children[0] if len(children) == 1 and children[0].is_dir() else directory


def _runtime(args: argparse.Namespace, storage: Any) -> tuple[Path, dict[str, Any]]:
    receipt = prepare_runtime(
        storage,
        storage,
        args.runtime_manifest_uri,
        args.runtime_manifest_sha256,
        args.home,
        args.runtime_workspace,
    )
    allowed = tuple(PurePosixPath(value) for value in receipt["allowed_directories"])
    python = _scientific_python(args, allowed)
    scientific = _scientific_receipt(args, allowed, receipt)
    return python, {**receipt, "scientific_runtime": scientific}


def _scientific_python(
    args: argparse.Namespace, allowed: tuple[PurePosixPath, ...]
) -> Path:
    relative = _safe_runtime_relative(args.scientific_python_relative, allowed)
    python = args.home / Path(*relative.parts)
    if python.is_symlink():
        if os.readlink(python) != args.scientific_python_link_target:
            raise ValueError("locked scientific venv Python link differs")
    elif not python.is_file():
        raise ValueError("locked scientific venv Python is absent")
    base_relative = _safe_runtime_relative(
        args.scientific_python_base_relative, allowed
    )
    base = args.home / Path(*base_relative.parts)
    if not base.is_file() or base.is_symlink():
        raise ValueError("restored scientific Python base differs")
    if python.is_symlink() and python.resolve(strict=True) != base.resolve(strict=True):
        raise ValueError("scientific Python link resolves to a different base")
    _expected(
        base, args.scientific_python_base_sha256, args.scientific_python_base_bytes
    )
    _expected(
        python.resolve(strict=True) if python.is_symlink() else python,
        args.scientific_python_base_sha256,
        args.scientific_python_base_bytes,
    )
    return python


def _verify_scientific_contract(
    contract: dict[str, Any], args: argparse.Namespace, runtime_ready: dict[str, Any]
) -> None:
    expected_python = {
        "relative": args.scientific_python_relative,
        "link_target": args.scientific_python_link_target,
        "base_relative": args.scientific_python_base_relative,
        "base_sha256": args.scientific_python_base_sha256,
        "base_bytes": args.scientific_python_base_bytes,
    }
    expected_base = {
        name: runtime_ready[name] for name in ("schema", "python_directory")
    }
    distributions = contract.get("installed_distributions")
    digests = ("sha256", "canonical_sha256")
    valid_distributions = (
        isinstance(distributions, dict)
        and isinstance(distributions.get("count"), int)
        and not isinstance(distributions.get("count"), bool)
        and distributions["count"] > 0
        and isinstance(distributions.get("bytes"), int)
        and not isinstance(distributions.get("bytes"), bool)
        and distributions["bytes"] > 0
        and all(
            isinstance(distributions.get(name), str) and len(distributions[name]) == 64
            for name in digests
        )
    )
    if (
        contract.get("schema") != "npa.behavior.locked-scientific-runtime.v1"
        or contract.get("status") != "complete"
        or contract.get("python") != expected_python
        or contract.get("runtime_base") != expected_base
        or not valid_distributions
    ):
        raise ValueError("locked scientific runtime semantic binding differs")


def _scientific_receipt(
    args: argparse.Namespace,
    allowed: tuple[PurePosixPath, ...],
    runtime_ready: dict[str, Any],
) -> dict[str, Any]:
    runtime_relative = _safe_runtime_relative(
        args.scientific_runtime_receipt_relative, allowed
    )
    runtime_path = args.home / Path(*runtime_relative.parts)
    _expected(
        runtime_path,
        args.scientific_runtime_receipt_sha256,
        args.scientific_runtime_receipt_bytes,
    )
    contract = json.loads(runtime_path.read_text())
    _verify_scientific_contract(contract, args, runtime_ready)
    return {
        **file_identity(runtime_path),
        "relative": str(runtime_relative),
        "contract": contract,
    }


def _safe_runtime_relative(
    value: str, allowed: tuple[PurePosixPath, ...]
) -> PurePosixPath:
    """Validate one home-relative path against runtime-ready roots."""
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError("scientific runtime relative path is unsafe")
    if not any(relative == root or root in relative.parents for root in allowed):
        raise ValueError("scientific runtime path is outside allowed directories")
    return relative


def _input_contract(
    args: argparse.Namespace, runtime: dict[str, Any], python: Path
) -> dict[str, Any]:
    names = ("openpi", "worker", "dataset", "parent", "admission")
    return {
        "schema": "npa.behavior.comet-native-workflow-inputs.v1",
        "archives": {
            name: {
                "uri": getattr(args, f"{name}_uri"),
                "sha256": getattr(args, f"{name}_sha256"),
                "bytes": getattr(args, f"{name}_bytes"),
            }
            for name in names
        },
        "runtime_manifest": {
            "uri": args.runtime_manifest_uri,
            "sha256": args.runtime_manifest_sha256,
        },
        "runtime_ready": runtime,
        "scientific_python": {
            **file_identity(python.resolve(strict=True)),
            "relative": args.scientific_python_relative,
            "link_target": args.scientific_python_link_target,
            "base_relative": args.scientific_python_base_relative,
        },
        "dataset_relative": args.dataset_relative,
        "split_relative": args.split_relative,
        "split_sha256": args.split_sha256,
        "config_name": args.config_name,
        "final_step": args.final_step,
        "milestones": args.milestones,
    }


def _stage_archives(args: argparse.Namespace, storage: Any) -> dict[str, Path]:
    staged = {}
    admission = args.workspace / "admission.json"
    _download(
        storage,
        args.admission_uri,
        admission,
        args.admission_sha256,
        args.admission_bytes,
    )
    admission_value = json.loads(admission.read_text())
    required = admission_value.get("minimum_materialization_free_bytes")
    archive_bytes = sum(
        getattr(args, f"{name}_bytes")
        for name in ("openpi", "worker", "dataset", "parent")
    )
    if (
        isinstance(required, bool)
        or not isinstance(required, int)
        or required < archive_bytes
    ):
        raise ValueError("materialization capacity contract differs")
    if shutil.disk_usage(args.workspace).free < required:
        raise ValueError("workspace free space is below materialization requirement")
    for name in ("openpi", "worker", "dataset", "parent"):
        archive = args.workspace / f"{name}.tar.gz"
        _download(
            storage,
            getattr(args, f"{name}_uri"),
            archive,
            getattr(args, f"{name}_sha256"),
            getattr(args, f"{name}_bytes"),
        )
        destination = args.workspace / name
        _extract(archive, destination)
        staged[name] = _root(destination)
    staged["admission"] = admission
    return staged


def _science_command(
    args: argparse.Namespace,
    staged: dict[str, Path],
    operation: str,
    output: Path,
    selection: Path,
) -> list[str]:
    implementation = staged["worker"] / "workflows/implementations/behavior-comet12"
    command = ["-B", str(implementation / "train_comet_native.py"), operation]
    paths = _science_paths(args, staged, operation, output, implementation)
    for name, value in paths.items():
        command.extend((f"--{name}", str(value)))
    command.extend(
        (
            "--admission-sha256",
            args.admission_sha256,
            "--split-sha256",
            args.split_sha256,
            "--config-name",
            args.config_name,
            "--output-prefix",
            args.prefix,
            "--final-step",
            str(args.final_step),
            "--milestones",
            args.milestones,
        )
    )
    if operation == "train":
        command.extend(("--resume-selection", str(selection)))
    return command


def _science_paths(
    args: argparse.Namespace,
    staged: dict[str, Path],
    operation: str,
    output: Path,
    implementation: Path,
) -> dict[str, Path]:
    return {
        "source-root": staged["openpi"],
        "dataset-root": staged["dataset"] / args.dataset_relative,
        "split": staged["dataset"] / args.split_relative,
        "parent-checkpoint": staged["parent"],
        "workspace": args.workspace / f"science-{operation}",
        "admission": staged["admission"],
        "runtime-implementation": implementation / "comet_openpi_runtime.py",
        "workflow-input-contract": args.workspace / "input-contract.json",
        "checkpoint-root": args.checkpoint_root,
        "output": output,
    }


def _run(
    command: list[str], python: Path, environment: dict[str, str], log: Path
) -> None:
    with log.open("xb") as stream:
        completed = subprocess.run(
            [str(python), *command],
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
        stream.flush()
        os.fsync(stream.fileno())
    if completed.returncode != 0:
        raise RuntimeError(f"native-training child failed: {completed.returncode}")


def _publish_small(storage: Any, path: Path, uri: str) -> dict[str, Any]:
    payload = path.read_bytes()
    try:
        storage.put_bytes_conditional(payload, uri, if_none_match=True)
    except Exception as error:
        if type(error).__name__ != "StoragePreconditionFailed":
            raise
    readback = path.with_name(path.name + ".readback")
    storage.download_file(uri, str(readback))
    if readback.read_bytes() != payload:
        raise ValueError("small artifact provider readback differs")
    readback.unlink()
    return {**file_identity(path), "uri": uri, "provider_readback": True}


def _reuse_completed_operation(
    storage: Any, args: argparse.Namespace, contract: dict[str, Any]
) -> dict[str, Any] | None:
    """Return one exact canonical receipt without rerunning the operation."""
    uri = f"{args.prefix.rstrip('/')}/{args.operation}/receipt.json"
    observed = storage.read_bytes_with_etag(uri)
    if observed is None:
        return None
    payload, _etag = observed
    value = json.loads(payload)
    validators = {
        "preflight": _validate_completed_preflight,
        "train": _validate_completed_train,
    }
    validators[args.operation](value, args, contract)
    return {
        "schema": "npa.behavior.comet-native-workflow-operation.v1",
        "operation": args.operation,
        "receipt": {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "uri": uri,
            "provider_readback": True,
        },
        "runtime": contract["runtime_ready"],
        "reused_completed_canonical_receipt": True,
    }


def _validate_completed_preflight(
    value: dict[str, Any], args: argparse.Namespace, contract: dict[str, Any]
) -> None:
    expected = {
        "schema": "npa.behavior.comet-native-input-preflight.v1",
        "status": "real_overlay_config_dataset_verified_before_policy_initialization",
        "config_name": args.config_name,
        "source_overlay": "ephemeral_verified_source_overlay",
        "workflow_inputs": contract,
        "full_policy_initialized": False,
        "optimizer_updates": 0,
        "jax_backend": "cpu",
    }
    if any(value.get(name) != observed for name, observed in expected.items()):
        raise ValueError("canonical completed preflight receipt differs")
    if not isinstance(value.get("dataset_size"), int) or value["dataset_size"] <= 0:
        raise ValueError("canonical completed preflight dataset differs")
    if not isinstance(value.get("admission"), dict) or not value["admission"]:
        raise ValueError("canonical completed preflight admission differs")


def _validate_completed_metrics(value: dict[str, Any], start: int, final: int) -> None:
    metrics = value.get("hot_path_metrics")
    names = {
        "step",
        "loss",
        "grad_norm",
        "param_norm",
        "learning_rate",
        "loader_consumer_wait_seconds",
        "synchronized_native_update_seconds",
    }
    if not isinstance(metrics, list) or len(metrics) != final - start:
        raise ValueError("canonical completed training metrics differ")
    for step, row in zip(range(start + 1, final + 1), metrics, strict=True):
        if not isinstance(row, dict) or set(row) != names or row.get("step") != step:
            raise ValueError("canonical completed training metric chronology differs")
        numbers = [number for name, number in row.items() if name != "step"]
        if not numbers or any(
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
            for number in numbers
        ):
            raise ValueError("canonical completed training metric values differ")


def _validate_completed_milestones(
    value: dict[str, Any], args: argparse.Namespace, milestones: list[int]
) -> None:
    checkpoints = value.get("checkpoints")
    expected_names = {str(value) for value in milestones}
    if not isinstance(checkpoints, dict) or set(checkpoints) != expected_names:
        raise ValueError("canonical completed training checkpoint schedule differs")
    if not isinstance(checkpoints["0"], dict) or not checkpoints["0"]:
        raise ValueError("canonical completed training parent milestone differs")
    runtime = value["runtime"]
    for step in milestones[1:]:
        row = checkpoints[str(step)]
        durable = row.get("durable") if isinstance(row, dict) else None
        manifest = durable.get("manifest") if isinstance(durable, dict) else None
        if not isinstance(manifest, dict):
            raise TypeError("canonical completed training milestone differs")
        validate_manifest(manifest, step=step, prefix=args.prefix)
        validate_complete_checkpoint(manifest["checkpoint"], step - 1)
        expected = {
            "outcome": "success",
            "admission": runtime["admission"],
            "workflow_inputs": runtime["workflow_inputs"],
            "serving_export_qualified": False,
            "scoring_executed": False,
        }
        if any(manifest.get(name) != observed for name, observed in expected.items()):
            raise ValueError("canonical completed training milestone binding differs")
        _validate_completed_provider(durable, manifest, args.prefix, step)


def _validate_completed_provider(
    durable: dict[str, Any], manifest: dict[str, Any], prefix: str, step: int
) -> None:
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    root = f"{prefix.rstrip('/')}/milestones/step-{step:05d}"
    expected = {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "uri": f"{root}/output-manifest.json",
        "provider_readback": True,
    }
    binding = durable.get("binding")
    if durable.get("schema") != "npa.behavior.comet-native-storage-result.v1":
        raise ValueError("canonical completed training storage result differs")
    if durable.get("provider_manifest") != expected or not isinstance(binding, dict):
        raise ValueError("canonical completed training provider identity differs")
    if (
        binding.get("checkpoint") != manifest["checkpoint"]
        or binding.get("prefix") != root
        or binding.get("logical_update_count") != step
    ):
        raise ValueError("canonical completed training provider binding differs")
    for name in ("request", "receipt"):
        identity = binding.get(name)
        if not isinstance(identity, dict) or set(identity) != {"bytes", "sha256"}:
            raise ValueError("canonical completed training provider evidence differs")


def _validate_completed_train(
    value: dict[str, Any], args: argparse.Namespace, contract: dict[str, Any]
) -> None:
    milestones = [int(part) for part in args.milestones.split(",")]
    final = args.final_step
    expected = {
        "schema": "npa.behavior.comet-native-full-training.v1",
        "status": "native_updates_and_declared_milestones_durable",
        "logical_updates": final,
        "milestones": milestones,
        "manager_index_semantics": "manager_step_equals_logical_update_minus_one",
        "selection_or_scoring_executed": False,
        "serving_export_qualified": False,
    }
    if any(value.get(name) != observed for name, observed in expected.items()):
        raise ValueError("canonical completed training receipt differs")
    runtime = value.get("runtime")
    start = value.get("start_logical_update")
    if not isinstance(runtime, dict) or runtime.get("workflow_inputs") != contract:
        raise ValueError("canonical completed training runtime differs")
    if isinstance(start, bool) or not isinstance(start, int) or start not in milestones:
        raise ValueError("canonical completed training start differs")
    _validate_completed_metrics(value, start, final)
    _validate_completed_milestones(value, args, milestones)


def _publish_failure(
    storage: Any, args: argparse.Namespace, error: BaseException
) -> dict[str, Any]:
    failure = args.workspace / "failure.json"
    _write_json(
        failure,
        {
            "schema": "npa.behavior.comet-native-workflow-failure.v1",
            "status": "failed",
            "operation": args.operation,
            "attempt_id": args.attempt_id,
            "error_type": type(error).__name__,
            "error": str(error),
        },
    )
    root = f"{args.prefix.rstrip('/')}/attempts/{args.attempt_id}"
    originals = {}
    evidence = sorted(args.workspace.glob("*.log"))
    contract = args.workspace / "input-contract.json"
    if contract.is_file() and not contract.is_symlink():
        evidence.append(contract)
    for path in evidence + [failure]:
        originals[path.name] = _publish_small(
            storage, path, f"{root}/originals/{path.name}"
        )
    manifest = args.workspace / "failure-manifest.json"
    _write_json(
        manifest,
        {
            "schema": "npa.behavior.comet-native-workflow-failure-manifest.v1",
            "status": "failed_originals_provider_readback",
            "operation": args.operation,
            "attempt_id": args.attempt_id,
            "originals": originals,
        },
    )
    return _publish_small(storage, manifest, f"{root}/failure-manifest.json")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _environment(
    staged: dict[str, Path], args: argparse.Namespace, control_python: Path
) -> dict[str, str]:
    package = staged["worker"] / "npa/src"
    implementation = staged["worker"] / "workflows/implementations/behavior-comet12"
    client = staged["openpi"] / "packages/openpi-client/src"
    if not (client / "openpi_client").is_dir():
        raise ValueError("OpenPI client source is unavailable")
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = f"{package}:{implementation}:{client}"
    environment["NPA_STORAGE_CONTROL_PYTHON"] = str(control_python)
    environment["NPA_STORAGE_CONTROL_PYTHONPATH"] = str(package)
    environment["NPA_STORAGE_CONTROL_WORKER"] = str(
        implementation / "storage_control_worker.py"
    )
    environment["NPA_STORAGE_ALLOWED_CHECKPOINT_ROOT"] = str(
        args.checkpoint_root.resolve()
    )
    environment["NPA_STORAGE_ALLOWED_OUTPUT_PREFIX"] = args.prefix.rstrip("/")
    return environment


def _recover_saved_milestones(
    args: argparse.Namespace,
    staged: dict[str, Path],
    environment: dict[str, str],
    contract: dict[str, Any],
    storage: Any,
) -> None:
    admission = json.loads(staged["admission"].read_text())
    admission_identity = file_identity(staged["admission"])
    previous = dict(os.environ)
    os.environ.update(environment)
    try:
        for step in (
            int(value) for value in args.milestones.split(",") if int(value) > 0
        ):
            _recover_one(args, step, storage, admission, admission_identity, contract)
    finally:
        os.environ.clear()
        os.environ.update(previous)


def _recover_one(args, step, storage, admission, identity, contract) -> None:
    checkpoint = args.checkpoint_root / f"step-{step:05d}"
    receipt = args.checkpoint_root / f"step-{step:05d}.json"
    if not checkpoint.exists() and not receipt.exists():
        return
    uri = f"{args.prefix.rstrip('/')}/milestones/step-{step:05d}/output-manifest.json"
    provider = storage.read_bytes_with_etag(uri)
    if provider is not None and _accept_durable_local(
        provider[0],
        args,
        step,
        checkpoint,
        receipt,
        admission,
        identity,
        contract,
    ):
        return
    if (
        checkpoint.is_symlink()
        or not checkpoint.is_dir()
        or receipt.is_symlink()
        or not receipt.is_file()
    ):
        raise ValueError("saved unpublished milestone is incomplete or unsafe")
    observed = checkpoint_inventory(checkpoint)
    record = json.loads(receipt.read_text())
    _validate_saved_record(record, observed, admission, identity, contract, step)
    publish_with_control(
        checkpoint,
        receipt,
        uri.removesuffix("/output-manifest.json"),
        args.checkpoint_root,
    )


def _validate_saved_record(
    record: dict[str, Any],
    observed: dict[str, Any],
    admission: dict[str, Any],
    identity: dict[str, Any],
    contract: dict[str, Any],
    step: int,
) -> None:
    expected = {
        "logical_update_count": step,
        "manager_step": step - 1,
        "checkpoint": observed,
        "admission": identity,
        "workflow_inputs": contract,
        "static_reconstruction": admission["static_reconstruction"],
        "full_state_resume_ready": True,
    }
    if any(record.get(name) != value for name, value in expected.items()):
        raise ValueError("saved unpublished milestone contract differs")
    validate_full_state_contract(record.get("train_state", {}), step)
    validate_complete_checkpoint(observed, step - 1)


def _accept_durable_local(
    payload: bytes,
    args: argparse.Namespace,
    step: int,
    checkpoint: Path,
    receipt: Path,
    admission: dict[str, Any],
    identity: dict[str, Any],
    contract: dict[str, Any],
) -> bool:
    manifest = json.loads(payload)
    validate_manifest(manifest, step=step, prefix=args.prefix)
    expected = {
        "admission": identity,
        "static_reconstruction": admission["static_reconstruction"],
        "workflow_inputs": contract,
    }
    if any(manifest.get(name) != value for name, value in expected.items()):
        raise ValueError("provider-durable milestone input binding differs")
    row = manifest["files"]["milestone-receipt.json"]
    if not receipt.exists() or receipt.is_symlink() or not receipt.is_file():
        raise ValueError("provider-durable local receipt is absent")
    if file_identity(receipt) != {key: row[key] for key in ("bytes", "sha256")}:
        raise ValueError("provider-durable local receipt differs")
    record = json.loads(receipt.read_text())
    _validate_saved_record(
        record, manifest["checkpoint"], admission, identity, contract, step
    )
    if (
        checkpoint.exists()
        and checkpoint_inventory(checkpoint) != manifest["checkpoint"]
    ):
        raise ValueError("provider-durable local checkpoint differs")
    if not checkpoint.exists():
        receipt.unlink()
    return True


def _require_preflight(
    storage: Any, args: argparse.Namespace, contract: dict[str, Any], workspace: Path
) -> None:
    path = workspace / "preflight-receipt.json"
    storage.download_file(args.preflight_uri, str(path))
    value = json.loads(path.read_text())
    if value.get("schema") != "npa.behavior.comet-native-input-preflight.v1":
        raise ValueError("provider preflight receipt differs")
    if (
        value.get("status")
        != "real_overlay_config_dataset_verified_before_policy_initialization"
    ):
        raise ValueError("provider preflight receipt differs")
    if value.get("workflow_inputs") != contract:
        raise ValueError("preflight complete input contract differs")


def _restore_command(
    args: argparse.Namespace,
    staged: dict[str, Path],
    contract_path: Path,
    selection: Path,
) -> list[str]:
    return [
        "-B",
        "-m",
        "npa.workflows.behavior_challenge.native_training_restore",
        "--prefix",
        args.prefix,
        "--milestones",
        args.milestones,
        "--checkpoint-base",
        str(args.checkpoint_root),
        "--selection",
        str(selection),
        "--admission",
        str(staged["admission"]),
        "--workflow-input-contract",
        str(contract_path),
    ]


def _prepare_operation(
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Path], Path, dict[str, Any], Path, dict[str, Any]]:
    if args.workspace.exists() or args.workspace.is_symlink():
        raise ValueError("workflow workspace must be new")
    args.workspace.mkdir(parents=True)
    args._workspace_owned = True
    args.checkpoint_root.mkdir(parents=True, exist_ok=True)
    storage = StorageClient.from_environment()
    staged = _stage_archives(args, storage)
    science_python, runtime = _runtime(args, storage)
    contract = _input_contract(args, runtime, science_python)
    contract_path = args.workspace / "input-contract.json"
    contract_path.write_text(json.dumps(contract, sort_keys=True) + "\n")
    return storage, staged, science_python, contract, contract_path, runtime


def _prepare_train(
    args: argparse.Namespace,
    storage: Any,
    staged: dict[str, Path],
    environment: dict[str, str],
    contract: dict[str, Any],
    contract_path: Path,
    selection: Path,
) -> None:
    _require_preflight(storage, args, contract, args.workspace)
    _recover_saved_milestones(args, staged, environment, contract, storage)
    command = _restore_command(args, staged, contract_path, selection)
    _run(command, Path(os.sys.executable), environment, args.workspace / "restore.log")


def execute(args: argparse.Namespace) -> dict[str, Any]:
    """Run one real operation.

    Args: args: Fully parsed immutable input and path bindings.
    Returns: Provider-read canonical operation receipt identity.
    Raises: ValueError for contract/path differences; RuntimeError for child failure.
    """
    storage, staged, science_python, contract, contract_path, runtime = (
        _prepare_operation(args)
    )
    completed = _reuse_completed_operation(storage, args, contract)
    if completed is not None:
        return completed
    environment = _environment(staged, args, Path(os.sys.executable))
    output = args.workspace / f"{args.operation}.json"
    selection = args.workspace / "resume-selection.json"
    if args.operation == "train":
        _prepare_train(
            args, storage, staged, environment, contract, contract_path, selection
        )
    _run(
        _science_command(args, staged, args.operation, output, selection),
        science_python,
        environment,
        args.workspace / f"{args.operation}.log",
    )
    provider = _publish_small(
        storage, output, f"{args.prefix.rstrip('/')}/{args.operation}/receipt.json"
    )
    return {
        "schema": "npa.behavior.comet-native-workflow-operation.v1",
        "operation": args.operation,
        "receipt": provider,
        "runtime": runtime,
    }


def parser() -> argparse.ArgumentParser:
    """Build the worker parser.

    Args: None.
    Returns: Parser for immutable inputs and operation paths.
    Raises: None.
    """
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("operation", choices=("preflight", "train"))
    for name in STRING_ARGUMENTS:
        value.add_argument("--" + name, required=True)
    for name in INTEGER_ARGUMENTS:
        value.add_argument("--" + name, type=int, required=True)
    for name in ("home", "runtime-workspace", "workspace", "checkpoint-root"):
        value.add_argument("--" + name, type=Path, required=True)
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute one operation and preserve owned failures.

    Args: args: Parsed operation inputs.
    Returns: Successful canonical operation receipt identity.
    Raises: The original execution error after best-effort failure publication.
    """
    if re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", args.attempt_id) is None:
        raise ValueError("attempt ID differs")
    try:
        return execute(args)
    except Exception as error:
        if getattr(args, "_workspace_owned", False):
            try:
                _publish_failure(StorageClient.from_environment(), args, error)
            except Exception as publication_error:  # noqa: BLE001
                error.add_note(
                    f"failure evidence publication also failed: {publication_error}"
                )
        raise


def main() -> None:
    """Execute the requested portable workflow operation.

    Args: None. Returns: None.
    Raises: Propagates parsing, preparation, child, and publication errors.
    """
    result = run(parser().parse_args())
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
