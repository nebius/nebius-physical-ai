"""Prepare a verified native Comet TRAIN checkpoint for one evaluator case."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from .comet_policy import CONFIG_NAME, SOURCE_COMMIT, verify_source
from .campaign import canonical_digest
from .native_comet_checkpoint import validate_native_train_admission
from .native_comet_server import (
    validate_load_qualification,
    validate_process_receipt,
    validate_trace_row,
)
from .native_training_checkpoint import atomic_json, file_identity
from .serving_identity import serving_artifact


def case_seed(rng_identity: dict, case: dict) -> int:
    """Derive a checkpoint-independent seed from the case and RNG contract.

    Args:
        rng_identity: Frozen RNG-contract byte identity.
        case: Prescribed TRAIN case.
    Returns:
        Unsigned 32-bit seed independent of checkpoint identity.
    Raises:
        KeyError: A required identity or case field is absent.
    """
    payload = {
        "schema": "npa.behavior.comet-native-case-seed.v1",
        "rng_contract": rng_identity,
        "task": case["task"],
        "instance_id": case["instance_id"],
        "rollout_id": case["rollout_id"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "big")


def _stage_adapters(output: Path) -> dict[str, dict]:
    names = (
        "native_comet_server.py",
        "native_comet_policy.py",
        "native_comet_checkpoint.py",
        "comet_policy.py",
        "evaluator_versions.py",
        "evaluator_wire.py",
    )
    rows = {}
    for name in names:
        source = Path(__file__).with_name(name)
        target = output / name
        shutil.copyfile(source, target)
        rows[name] = file_identity(target)
    return rows


def _server_command(args, plan: dict, output: Path, seed: int) -> list[str]:
    binding = json.loads(Path(args.policy_native_binding).read_text())
    case = plan["cases"][0]
    verified = json.loads(Path(args.policy_archive).read_text())
    command = [
        str(args.policy_python),
        str(output / "native_comet_server.py"),
        "--source-root",
        str(args.policy_root),
        "--checkpoint",
        str(args.policy_checkpoint),
        "--manager-step",
        str(verified["manager_step"]),
        "--asset-id",
        verified["serving_assets"]["asset_id"],
        "--task-id",
        str(binding["task_id"]),
        "--task-name",
        binding["task"],
        "--port",
        str(args.port),
        "--upstream-commit",
        plan["upstream_commit"],
        "--case-id",
        case["case_id"],
        "--instance-id",
        str(case["instance_id"]),
        "--rollout-id",
        str(case["rollout_id"]),
        "--case-seed",
        str(seed),
        "--checkpoint-sha256",
        verified["checkpoint"]["content_sha256"],
        "--rng-contract-sha256",
        binding["artifacts"]["rng_contract"]["identity"]["sha256"],
        "--trace-configuration-sha256",
        canonical_digest(binding["trace"]),
        "--process-receipt",
        str(output / "native-process.json"),
    ]
    if binding["trace"]["enabled"]:
        command.extend(("--action-trace", str(output / "native-actions.jsonl")))
    return command


def _validate_case_plan(args, plan: dict, binding: dict) -> dict:
    if plan.get("recipe", {}).get("split") != "train":
        raise ValueError("Native Comet policy requires TRAIN split")
    cases = plan.get("cases")
    if not isinstance(cases, list) or len(cases) != 1:
        raise ValueError("Native Comet policy requires one case")
    case = cases[0]
    if (
        plan["recipe"].get("tasks") != [binding["task"]]
        or case.get("task") != binding["task"]
        or case.get("rollout_id") != 0
        or getattr(args, "policy_task_name", None) != binding["task"]
    ):
        raise ValueError("Native Comet task/case binding differs")
    return case


def _verify_task_mapping(root: Path, task_id: int, task: str) -> None:
    mapping = json.loads((root / "scripts/task_mapping.json").read_text())
    row = mapping.get(task)
    if (
        not isinstance(row, dict)
        or row.get("task_index") != task_id
        or not isinstance(row.get("task"), str)
        or not row["task"].strip()
    ):
        raise ValueError("Native Comet source task mapping differs")


def prepare_policy(args, plan: dict, output: Path) -> list[str]:
    """Revalidate admission, qualify a discarded load, and stage serving.

    Args:
        args: Managed-policy runtime paths and settings.
        plan: One-case frozen TRAIN execution plan.
        output: Fresh case evidence directory.
    Returns:
        Serving-process argv for the managed supervisor.
    Raises:
        ValueError: Admission, source, task, or qualification differs.
        subprocess.CalledProcessError: The discarded load process fails.
    """
    verify_source(args.policy_root)
    binding = json.loads(Path(args.policy_native_binding).read_text())
    case = _validate_case_plan(args, plan, binding)
    _verify_task_mapping(args.policy_root, binding["task_id"], binding["task"])
    admission = validate_native_train_admission(
        Path(args.policy_native_binding),
        Path(args.policy_native_input_root),
        Path(args.policy_checkpoint),
        args.policy_native_panel,
        expected_verified_checkpoint=Path(args.policy_archive),
        expected_serving_identity=serving_artifact(args),
    )
    if admission != args.policy_native_admission:
        raise ValueError("Native Comet admission changed after worker preflight")
    output.mkdir(parents=True, exist_ok=True)
    adapters = _stage_adapters(output)
    seed = case_seed(
        binding["artifacts"]["rng_contract"]["identity"],
        case,
    )
    command = _server_command(args, plan, output, seed)
    qualification = output / "native-load-qualification.json"
    qualifier = [
        *command,
        "--qualify-only",
        "--qualification-output",
        str(qualification),
    ]
    subprocess.run(qualifier, cwd=args.policy_root, check=True)
    receipt = json.loads(qualification.read_text())
    verified = json.loads(Path(args.policy_archive).read_text())
    validate_load_qualification(
        receipt,
        {
            "schema": "npa.behavior.comet-native-serving-load-qualification.v1",
            "status": "checkpoint_loaded_with_explicit_case_rng",
            "case_id": case["case_id"],
            "task": binding["task"],
            "instance_id": case["instance_id"],
            "rollout_id": case["rollout_id"],
            "case_seed": seed,
            "checkpoint_content_sha256": verified["checkpoint"]["content_sha256"],
            "manager_step": verified["manager_step"],
            "asset_id": verified["serving_assets"]["asset_id"],
            "source_commit": SOURCE_COMMIT,
            "config_name": CONFIG_NAME,
            "rng_contract_sha256": binding["artifacts"]["rng_contract"]["identity"][
                "sha256"
            ],
            "trace_configuration_sha256": canonical_digest(binding["trace"]),
            "inference_count": 0,
        },
    )
    evidence = {
        "schema": "npa.behavior.comet-native-policy.v1",
        "status": "discarded_load_qualification_complete_serving_not_started",
        "admission": admission,
        "case": case,
        "case_seed": seed,
        "qualification": file_identity(qualification),
        "adapters": adapters,
        "command": command,
    }
    atomic_json(output / "policy-provenance.json", evidence)
    return command


def _progress_receipts(output: Path, ready: dict) -> tuple[list[dict], dict | None]:
    path = output / "native-process-progress.jsonl"
    if not path.exists():
        return [], None
    if path.is_symlink() or not path.is_file():
        raise ValueError("Native Comet process progress differs")
    progress = [
        validate_process_receipt(json.loads(line), ready)
        for line in path.read_text().splitlines()
    ]
    previous = ready
    for action_count, receipt in enumerate(progress, start=1):
        if receipt["action_count"] != action_count:
            raise ValueError("Native Comet process action count differs")
        difference = receipt["inference_count"] - previous["inference_count"]
        same_rng = receipt["current_rng_sha256"] == previous["current_rng_sha256"]
        if difference not in {0, 1} or (difference == 1) == same_rng:
            raise ValueError("Native Comet process RNG chronology differs")
        previous = receipt
    return progress, file_identity(path)


def _trace_receipt(output: Path, ready: dict, progress: list[dict]) -> dict | None:
    trace_path = output / "native-actions.jsonl"
    if ready["trace_enabled"] is not True:
        if trace_path.exists() or trace_path.is_symlink():
            raise ValueError("Disabled Native Comet trace exists")
        return None
    if trace_path.is_symlink() or not trace_path.is_file():
        raise ValueError("Enabled Native Comet trace is absent")
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    if len(rows) != len(progress):
        raise ValueError("Native Comet trace/process chronology differs")
    previous_rng = ready["current_rng_sha256"]
    previous_ordinal = -1
    for index, (row, receipt) in enumerate(zip(rows, progress, strict=True)):
        validate_trace_row(row, ready, index)
        if row["rng_before_sha256"] != previous_rng:
            raise ValueError("Native Comet trace RNG prefix differs")
        new_inference = row["inference_ordinal"] == previous_ordinal + 1
        queued_action = row["inference_ordinal"] == previous_ordinal
        if not (new_inference or queued_action):
            raise ValueError("Native Comet trace inference ordinal differs")
        if new_inference == (row["rng_before_sha256"] == row["rng_after_sha256"]):
            raise ValueError("Native Comet trace RNG transition differs")
        if (
            receipt["current_rng_sha256"] != row["rng_after_sha256"]
            or receipt["inference_count"] != row["inference_ordinal"] + 1
        ):
            raise ValueError("Native Comet trace progress receipt differs")
        previous_rng = row["rng_after_sha256"]
        previous_ordinal = row["inference_ordinal"]
    return {"rows": len(rows), **file_identity(trace_path)}


def finalize_process(output: Path, returncode: int | None) -> dict:
    """Validate process and trace evidence after supervisor shutdown.

    Args:
        output: Case evidence directory.
        returncode: Managed-process return code after shutdown.
    Returns:
        Immutable final process receipt.
    Raises:
        ValueError: Receipt or trace chronology and identities differ.
    """
    ready_path = output / "native-process.json"
    final_path = output / "native-process-final.json"
    if not ready_path.exists():
        result = {
            "schema": "npa.behavior.comet-native-serving-final.v1",
            "status": "process_exited_before_ready",
            "inference_count": 0,
            "action_count": 0,
            "trace": None,
            "supervisor_returncode": returncode,
            "ready_receipt": None,
            "progress_journal": None,
        }
        atomic_json(final_path, result)
        return result
    ready = validate_process_receipt(json.loads(ready_path.read_text()))
    if (
        ready["status"] != "ready"
        or ready["current_rng_sha256"] != ready["initial_rng_sha256"]
    ):
        raise ValueError("Native Comet ready receipt differs")
    progress, progress_identity = _progress_receipts(output, ready)
    if progress:
        latest = progress[-1]
        if latest["inference_count"] <= 0:
            raise ValueError("Native Comet process progress differs")
    else:
        latest = ready
    trace = _trace_receipt(output, ready, progress)
    result = {
        "schema": "npa.behavior.comet-native-serving-final.v1",
        "status": "supervisor_stopped_fresh_case_process",
        "case_id": ready["case_id"],
        "checkpoint_content_sha256": ready["checkpoint_content_sha256"],
        "initial_rng_sha256": ready["initial_rng_sha256"],
        "final_rng_sha256": latest["current_rng_sha256"],
        "inference_count": latest["inference_count"],
        "action_count": len(progress),
        "trace": trace,
        "supervisor_returncode": returncode,
        "ready_receipt": file_identity(ready_path),
        "progress_journal": progress_identity,
    }
    atomic_json(final_path, result)
    return result
