"""Admit one parity-qualified trained Comet BF16 export for panel serving."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

import numpy as np

from .campaign import canonical_digest, validate_panel
from .native_comet_checkpoint import canonical_asset_id, canonical_provider_row
from .native_training_checkpoint import file_identity
from . import trained_comet_producer as producer

SCORER_STATIC_RECONSTRUCTION = producer.SCORER_STATIC_RECONSTRUCTION

ADMISSION_SCHEMA = "npa.behavior.comet-trained-serving-admission.v1"
PARITY_SCHEMA = "npa.behavior.comet12-trained-selected-milestone-parity-output.v1"
PARITY_STATUS = "selected_milestone_bf16_export_and_two_cold_processes_qualified"
SOURCE_COMMIT = "4bb2aa7bb2da32614cac128ebb4b2f96eb66e5b5"
FIXED_NOISE = {
    "schema": "npa.private.comet12-selected-parity-fixed-noise.v1",
    "algorithm": "jax.random.normal",
    "key_constructor": "jax.random.PRNGKey",
    "seed": 0,
    "shape": [32, 32],
    "dtype": "float32",
    "expected_c_sha256": "e24953cc067a191dd12ee8e1f439409875ac67f0b7fbc67bcfa9b846047f059a",
    "pinned_source": {
        "commit": SOURCE_COMMIT,
        "path": "src/openpi/models/pi0.py",
        "sha256": "95892ad57b72a42f38041256843f3a95c37a83e01ef754f38d4e39b835467fb4",
    },
}
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is absent or unsafe")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _exact_identity(path: Path, row: dict, label: str) -> None:
    expected = {name: row.get(name) for name in ("bytes", "sha256")}
    if file_identity(path) != expected:
        raise ValueError(f"{label} local bytes differ from provider readback")


def _safe_path(root: Path, relative: PurePosixPath, label: str) -> Path:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{label} root is absent or unsafe")
    current = root
    for part in relative.parts[:-1]:
        current /= part
        if current.is_symlink() or not current.is_dir():
            raise ValueError(f"{label} parent is absent or unsafe")
    return current / relative.name


def _parity_files(value: dict) -> dict[str, dict]:
    files = value.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Selected parity file rows differ")
    for name, row in files.items():
        if not isinstance(name, str) or not name or not isinstance(row, dict):
            raise ValueError("Selected parity file rows differ")
        canonical_provider_row(row, f"selected parity {name}", allow_empty=True)
    return files


def _validate_cold_processes(evidence: dict) -> None:
    rows = evidence.get("processes")
    if (
        not isinstance(rows, list)
        or len(rows) != 2
        or any(not isinstance(row, dict) for row in rows)
        or {row.get("process_label") for row in rows if isinstance(row, dict)}
        != {"cold-a", "cold-b"}
        or rows[0].get("pid") == rows[1].get("pid")
        or rows[0].get("actions") != rows[1].get("actions")
    ):
        raise ValueError("Selected parity cold processes differ")
    for row in rows:
        actions = row.get("actions")
        schema = row.get("input_schema")
        if (
            row.get("schema")
            != "npa.behavior.comet12-trained-milestone-serving-parity-process.v1"
            or row.get("status")
            != "exact_bf16_policy_action_parity_from_selected_milestone"
            or row.get("trained_milestone") is not True
            or row.get("rollout_quality") is not False
            or row.get("fp32_full_state_inventory_sha256")
            != evidence.get("fp32_full_state_inventory_sha256")
            or row.get("device_count") != 1
            or row.get("device_platform") != "gpu"
            or row.get("parameter_precision") != "bfloat16_for_native_and_exported"
            or not isinstance(actions, dict)
            or set(actions) != {"native", "exported"}
            or actions["native"] != actions["exported"]
            or actions["native"].get("shape") != [32, 32]
            or actions["native"].get("dtype") != "float32"
            or _SHA256.fullmatch(str(actions["native"].get("sha256"))) is None
            or not isinstance(schema, dict)
            or schema.get("synthetic") is not False
        ):
            raise ValueError("Selected parity cold process differs")


def _action_identity(value: np.ndarray) -> dict[str, Any]:
    if not np.isfinite(value).all():
        raise ValueError("Selected parity raw action contains non-finite values")
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def _validate_raw_actions(input_root: Path, files: dict, evidence: dict) -> None:
    by_label = {row["process_label"]: row for row in evidence["processes"]}
    for label in ("cold-a", "cold-b"):
        process = _original(input_root, files, f"work/result-{label}.json")
        receipt = _original(input_root, files, f"work/raw-actions-{label}.json")
        npz_name = f"work/raw-actions-{label}.npz"
        npz_row = files.get(npz_name)
        if not isinstance(npz_row, dict):
            raise ValueError("Selected parity raw action archive is absent")
        npz_path = _safe_path(
            input_root, PurePosixPath(npz_name), "selected parity raw actions"
        )
        _exact_identity(npz_path, npz_row, "selected parity raw actions")
        with np.load(npz_path, allow_pickle=False) as archive:
            if set(archive.files) != {"native", "exported"}:
                raise ValueError("Selected parity raw action arrays differ")
            native = np.asarray(archive["native"])
            exported = np.asarray(archive["exported"])
        if (
            process != by_label[label]
            or receipt.get("schema")
            != "npa.behavior.comet12-trained-milestone-parity-raw-actions.v1"
            or receipt.get("status")
            != "raw_native_and_exported_actions_recorded_before_equality_gate"
            or receipt.get("process_label") != label
            or receipt.get("pid") != process.get("pid")
            or receipt.get("actions") != process.get("actions")
            or receipt.get("npz") != file_identity(npz_path)
            or receipt.get("admission_granted") is not False
            or _action_identity(native) != process["actions"]["native"]
            or _action_identity(exported) != process["actions"]["exported"]
            or native.dtype != exported.dtype
            or native.shape != exported.shape
            or native.tobytes(order="C") != exported.tobytes(order="C")
        ):
            raise ValueError("Selected parity raw action proof differs")


def _original(input_root: Path, files: dict, name: str) -> dict:
    row = files.get(name)
    if not isinstance(row, dict):
        raise ValueError(f"Selected parity original is absent: {name}")
    path = _safe_path(input_root, PurePosixPath(name), f"selected parity {name}")
    _exact_identity(path, row, f"selected parity {name}")
    return _json(path, f"selected parity {name}")


def _validate_selected_reader(root, binding, chosen, score, score_row) -> None:
    path, provider = _admitted_file(root, binding, "selected reader")
    reader = _json(path, "selected reader")
    score_fields = {
        "checkpoint_inventory_sha256",
        "ordered_input_sha256",
        "process_identity",
        "rng_schedule_sha256",
        "sample_inventory_sha256",
        "scorer_contract_sha256",
        "serving_precision_receipt_sha256",
        "step",
        "tasks",
    }
    reader_score = reader.get("score")
    valid = (
        {name: provider[name] for name in ("bytes", "sha256")} == chosen["reader"]
        and reader.get("schema")
        == "npa.behavior.comet12-dp4-score-independent-readback.v2"
        and reader.get("status")
        == "one_exact_dp4_train_score_provider_originals_verified"
        and reader.get("candidate_arm") == chosen["arm"]
        and reader.get("logical_update") == chosen["logical_update"]
        and reader.get("selection_executed") is False
        and reader.get("development_or_report_read") is False
        and isinstance(reader_score, dict)
        and set(reader_score) == score_fields
        and reader_score
        == {key: score.get("score", {}).get(key) for key in score_fields}
        and reader.get("originals", {}).get("score-process.json")
        == {name: score_row[name] for name in ("bytes", "sha256")}
    )
    if not valid:
        raise ValueError("Selected parity reader differs")


def _runtime_bootstrap_lineage(
    input_root, bootstrap, locked, installed, runtime
) -> None:
    locked_path = _safe_path(
        input_root,
        PurePosixPath("runtime-inputs/locked-runtime.json"),
        "selected parity locked runtime",
    )
    installed_path = _safe_path(
        input_root,
        PurePosixPath("runtime-inputs/installed-distributions.json"),
        "selected parity installed distributions",
    )
    bootstrap_path = _safe_path(
        input_root,
        PurePosixPath("runtime-bootstrap.json"),
        "selected parity runtime bootstrap",
    )
    valid = (
        bootstrap.get("schema")
        == "npa.behavior.comet12-trained-parity-runtime-bootstrap.v1"
        and bootstrap.get("status")
        == "runtime_base_locked_venv_and_nested_npa_source_ready"
        and bootstrap.get("locked_receipt") == file_identity(locked_path)
        and bootstrap.get("locked_inventory") == file_identity(installed_path)
        and locked.get("schema") == "npa.behavior.comet-locked-runtime.v2"
        and locked.get("status") == "exact_lock_environment_with_upstream_import_ready"
        and locked.get("source_commit") == SOURCE_COMMIT
        and isinstance(installed, dict)
        and runtime.get("runtime_bootstrap") == file_identity(bootstrap_path)
    )
    if not valid:
        raise ValueError("Selected parity runtime bootstrap differs")


def _runtime_model_lineage(runtime, evidence, selection, score_row, milestone) -> None:
    valid = (
        runtime.get("schema")
        == "npa.behavior.comet12-trained-selected-milestone-parity-input.v1"
        and runtime.get("model_kind") == "qualified_comet12"
        and runtime.get("trained_milestone") is True
        and runtime.get("milestone_parity_satisfied") is False
        and runtime.get("official_24gb_hardware_qualified") is False
        and runtime.get("source_commit") == SOURCE_COMMIT
        and runtime.get("exported_inventory_sha256")
        == evidence.get("exported_inventory_sha256")
        and runtime.get("bf16_cast_inventory") == evidence.get("bf16_cast_inventory")
        and runtime.get("selected_v13_precision_receipt")
        == evidence.get("selected_v13_precision_receipt")
        and runtime.get("selection_receipt") == selection
        and {
            name: runtime.get("score_receipt_identity", {}).get(name)
            for name in ("bytes", "sha256")
        }
        == {name: score_row[name] for name in ("bytes", "sha256")}
        and runtime.get("milestone_receipt") == milestone
        and runtime.get("static_reconstruction") == SCORER_STATIC_RECONSTRUCTION
    )
    if not valid:
        raise ValueError("Selected parity runtime lineage differs")


def _runtime_inventory_lineage(runtime, evidence, admitted, checkpoint_sha256) -> None:
    full = runtime.get("fp32_full_state_inventory")
    valid = (
        full == admitted["checkpoint"]
        and runtime.get("fp32_full_state_inventory_sha256")
        == evidence.get("fp32_full_state_inventory_sha256")
        and _canonical_sha256(full) == runtime.get("fp32_full_state_inventory_sha256")
        and runtime.get("native_inventory") == admitted["serving"]
        and _canonical_sha256(runtime.get("native_inventory"))
        == runtime.get("native_inventory_sha256")
        and runtime.get("native_inventory_sha256")
        == evidence.get("native_inventory_sha256")
        and runtime.get("native_inventory_sha256") == checkpoint_sha256
    )
    if not valid:
        raise ValueError("Selected parity runtime inventory differs")


def _validate_fixed_observation(
    schema, fixed_identity, admission, files, processes
) -> None:
    provenance = schema.get("provenance") if isinstance(schema, dict) else None
    observation = (
        provenance.get("observation_contract") if isinstance(provenance, dict) else None
    )
    valid = (
        schema.get("schema") == "npa.behavior.comet12-real-native-fixed-input.v1"
        and schema.get("status")
        == "real_r6_cpu_transformed_observation_and_fixed_noise"
        and schema.get("noise", {}).get("shape") == [32, 32]
        and schema.get("noise", {}).get("dtype") == "float32"
        and schema.get("noise", {}).get("sha256") == FIXED_NOISE["expected_c_sha256"]
        and schema.get("npz") == fixed_identity
        and schema.get("synthetic") is False
        and provenance.get("noise_recipe") == FIXED_NOISE
        and isinstance(observation, dict)
        and observation.get("split") == "TRAIN"
        and observation.get("task_id") == admission["task_id"]
        and type(observation.get("episode_index")) is int
        and observation["episode_index"] >= 0
        and type(observation.get("ordinal")) is int
        and observation["ordinal"] >= 0
    )
    if not valid:
        raise ValueError("Selected parity fixed observation differs")
    _validate_process_inputs(processes, schema, fixed_identity, files)


def _validate_process_inputs(processes, schema, fixed_identity, files) -> None:
    schema_sha = files["prepared-run/inputs/fixed_input_schema.json"]["sha256"]
    expected_sample = schema["provenance"].get("sample")
    for row in processes:
        value = row.get("input_schema")
        valid = (
            isinstance(value, dict)
            and value.get("sha256") == schema_sha
            and value.get("npz") == fixed_identity
            and value.get("sample") == expected_sample
            and value.get("synthetic") is False
        )
        if not valid:
            raise ValueError("Selected parity process input differs")


def _validate_parity_result(result, evidence) -> None:
    claims = {
        "full_model_weights_loaded": True,
        "gpu_execution_verified": True,
        "official_24gb_hardware_qualified": False,
        "rollout_quality": False,
        "trained_milestone": True,
        "milestone_parity_satisfied": True,
        "selected_milestone_serving_parity_admission": True,
    }
    valid = (
        result.get("schema")
        == "npa.behavior.comet12-trained-selected-milestone-serving-parity.v1"
        and result.get("status")
        == "two_cold_processes_exact_selected_milestone_action_parity"
        and result.get("claims") == claims
        and result.get("processes") == evidence.get("processes")
    )
    if not valid:
        raise ValueError("Selected parity result differs")


def _parity_original_documents(input_root, files) -> dict:
    names = {
        "bootstrap": "runtime-bootstrap.json",
        "locked": "runtime-inputs/locked-runtime.json",
        "installed": "runtime-inputs/installed-distributions.json",
        "runtime": "runtime-contract.json",
        "result": "parity-result.json",
        "manifest": "prepared-run/inputs/milestone_manifest.json",
        "milestone": "prepared-run/inputs/milestone_receipt.json",
        "selection": "prepared-run/inputs/selection_receipt.json",
        "score": "prepared-run/inputs/score_receipt.json",
        "schema": "prepared-run/inputs/fixed_input_schema.json",
    }
    return {key: _original(input_root, files, name) for key, name in names.items()}


def _fixed_input_identity(input_root, files) -> dict:
    name = "prepared-run/inputs/fixed_input_npz.npz"
    row = files.get(name)
    if not isinstance(row, dict):
        raise ValueError("Selected parity fixed input is absent")
    path = _safe_path(input_root, PurePosixPath(name), "selected parity fixed input")
    _exact_identity(path, row, "selected parity fixed input")
    return file_identity(path)


def _validate_parity_originals(
    input_root: Path, files: dict, evidence: dict, admission: dict
) -> dict:
    documents = _parity_original_documents(input_root, files)
    fixed_identity = _fixed_input_identity(input_root, files)
    profile, admitted = _validate_producer_lineage(
        input_root, files, evidence, admission, documents
    )
    _validate_runtime_lineage(
        input_root, files, evidence, admission, documents, profile, admitted
    )
    _validate_parity_result(documents["result"], evidence)
    _validate_fixed_observation(
        documents["schema"], fixed_identity, admission, files, evidence["processes"]
    )
    _validate_raw_actions(input_root, files, evidence)
    return admitted


def _validate_producer_lineage(input_root, files, evidence, admission, documents):
    step = admission["selected_step"]
    checkpoint_sha = admission["selected_checkpoint_inventory_sha256"]
    precision_sha = admission["selected_precision_receipt_sha256"]
    profile = producer._validate_selection_and_score(
        documents["selection"],
        documents["score"],
        evidence["selected_v13_precision_receipt"],
        step,
        checkpoint_sha,
        precision_sha,
    )
    chosen = producer._validate_selected_candidate(
        documents["selection"], step, checkpoint_sha, precision_sha
    )
    _validate_selected_reader(
        input_root,
        admission["selected_reader"],
        chosen,
        documents["score"],
        files["prepared-run/inputs/score_receipt.json"],
    )
    admitted = producer._validate_milestone(
        documents["manifest"],
        documents["milestone"],
        files["prepared-run/inputs/milestone_receipt.json"],
        step,
        profile,
    )
    return profile, admitted


def _validate_runtime_lineage(
    input_root, files, evidence, admission, documents, profile, admitted
) -> None:
    step = admission["selected_step"]
    checkpoint_sha = admission["selected_checkpoint_inventory_sha256"]
    _runtime_bootstrap_lineage(
        input_root,
        documents["bootstrap"],
        documents["locked"],
        documents["installed"],
        documents["runtime"],
    )
    _runtime_model_lineage(
        documents["runtime"],
        evidence,
        documents["selection"],
        files["prepared-run/inputs/score_receipt.json"],
        documents["milestone"],
    )
    _runtime_inventory_lineage(documents["runtime"], evidence, admitted, checkpoint_sha)
    if (
        documents["runtime"].get("selected_step") != step
        or profile != admission["profile"]
    ):
        raise ValueError("Selected parity runtime candidate differs")
    if (
        documents["runtime"].get("input_schema_sha256")
        != files["prepared-run/inputs/fixed_input_schema.json"]["sha256"]
    ):
        raise ValueError("Selected parity runtime input differs")


def _validate_parity_terminal(value: object, selected_step: int) -> dict:
    if not isinstance(value, dict):
        raise ValueError("Selected parity manifest differs")
    evidence = value.get("success_evidence")
    valid = (
        value.get("schema") == PARITY_SCHEMA
        and value.get("status") == PARITY_STATUS
        and value.get("outcome") == "success"
        and value.get("exit_code") == 0
        and value.get("selected_milestone_serving_parity_admission") is True
        and value.get("official_24gb_hardware_qualified") is False
        and isinstance(value.get("run_id"), str)
        and bool(value["run_id"])
        and isinstance(evidence, dict)
        and evidence.get("selected_step") == selected_step
    )
    if not valid:
        raise ValueError("Selected parity manifest differs")
    return evidence


def validate_parity_manifest(
    value: object,
    *,
    selected_step: int,
    checkpoint_sha256: str,
    precision_sha256: str,
) -> tuple[dict, dict[str, dict]]:
    """Validate the selected trained-Comet parity terminal needed by panel serving.

    Args:
        value: Parsed selected-parity output manifest.
    Returns:
        Validated success evidence and exact provider file rows.
    Raises:
        ValueError: Outcome, selection, precision, or files differ.
    """
    evidence = _validate_parity_terminal(value, selected_step)
    producer._validate_precision(
        evidence, selected_step, checkpoint_sha256, precision_sha256
    )
    _validate_cold_processes(evidence)
    files = _parity_files(value)
    required = {"runtime-contract.json", "parity-result.json"}
    if not required.issubset(files):
        raise ValueError("Selected parity runtime originals are incomplete")
    return evidence, files


def _export_rows(files: dict[str, dict], step: int) -> tuple[list[dict], dict]:
    prefix = "exported-checkpoint/"
    selected = {name: row for name, row in files.items() if name.startswith(prefix)}
    rows = []
    providers = {}
    for name, provider in sorted(selected.items()):
        relative = name.removeprefix(prefix)
        path = PurePosixPath(relative)
        if (
            not relative.startswith("params/")
            or path.is_absolute()
            or path.as_posix() != relative
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("Selected parity export member differs")
        rows.append(
            {
                "path": relative,
                "bytes": provider["bytes"],
                "sha256": provider["sha256"],
            }
        )
        providers[f"{step}/{relative}"] = provider
    if not rows:
        raise ValueError("Selected parity export is absent")
    return rows, providers


def _trace(value: object) -> dict:
    expected = {
        "schema",
        "enabled",
        "action_width",
        "left_command_index",
        "right_command_index",
        "left_proprio_indices",
        "right_proprio_indices",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or value.get("schema") != "npa.behavior.comet-native-action-trace.v1"
        or not isinstance(value.get("enabled"), bool)
        or value.get("action_width") != 23
        or value.get("left_command_index") != 14
        or value.get("right_command_index") != 22
        or value.get("left_proprio_indices") != [24, 25]
        or value.get("right_proprio_indices") != [49, 50]
    ):
        raise ValueError("Trained Comet trace configuration differs")
    return value


def _admission(path: Path) -> dict:
    value = _json(path, "trained Comet admission")
    required = {
        "schema",
        "status",
        "profile",
        "selected_step",
        "manager_step",
        "selected_checkpoint_inventory_sha256",
        "selected_precision_receipt_sha256",
        "task",
        "task_id",
        "source_commit",
        "parity_output",
        "selected_reader",
        "normalization",
        "rng_contract",
        "trace",
        "serving_tree_sha256",
        "claims",
    }
    _admission_fields(value, required)
    canonical_provider_row(value["parity_output"], "selected parity output")
    if not isinstance(value["selected_reader"], dict):
        raise ValueError("Selected reader binding differs")
    _trace(value["trace"])
    return value


def _admission_fields(value: dict, required: set[str]) -> None:
    claims = {
        "optimizer_or_train_state_claimed": False,
        "development_or_report_read": False,
        "rollout_quality_claimed": False,
    }
    if (
        set(value) != required
        or value.get("schema") != ADMISSION_SCHEMA
        or value.get("status")
        != "actual_selected_parity_export_and_normalization_admitted"
        or not isinstance(value.get("profile"), str)
        or not value["profile"]
        or type(value.get("selected_step")) is not int
        or value["selected_step"] <= 0
        or value.get("manager_step") != value["selected_step"]
        or _SHA256.fullmatch(str(value.get("selected_checkpoint_inventory_sha256")))
        is None
        or _SHA256.fullmatch(str(value.get("selected_precision_receipt_sha256")))
        is None
        or value.get("source_commit") != SOURCE_COMMIT
        or not isinstance(value.get("task"), str)
        or not value["task"]
        or type(value.get("task_id")) is not int
        or value["task_id"] < 0
        or _SHA256.fullmatch(str(value.get("serving_tree_sha256"))) is None
        or value.get("claims") != claims
    ):
        raise ValueError("Trained Comet admission differs")


def _admitted_file(
    root: Path, value: object, label: str, *, extra_keys: set[str] | None = None
) -> tuple[Path, dict]:
    expected = {"path", "provider"} | (extra_keys or set())
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} binding differs")
    relative = PurePosixPath(str(value.get("path", "")))
    if (
        relative.is_absolute()
        or relative.as_posix() != value.get("path")
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"{label} path differs")
    provider = canonical_provider_row(value["provider"], label)
    path = _safe_path(root, relative, label)
    _exact_identity(path, provider, label)
    return path, provider


def _panel_binding(panel: dict, admission_path: Path, serving: dict) -> None:
    valid = validate_panel(panel)
    if (
        valid.get("split") not in {"development", "report"}
        or valid.get("selected_tasks") is None
        or len(valid["selected_tasks"]) != 1
    ):
        raise ValueError("Trained Comet requires one DEV or REPORT task panel")
    policy = valid["policy"]
    if (
        policy["artifacts"]["checkpoint"] != file_identity(admission_path)
        or policy["artifacts"]["serving"] != serving
    ):
        raise ValueError("Trained Comet panel policy identity differs")


def _verify_checkpoint_tree(
    root: Path, providers: dict[str, dict], serving_tree_sha256: str
) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Trained Comet serving root is absent or unsafe")
    actual = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise ValueError("Trained Comet serving tree contains an unsafe member")
        if path.is_file():
            actual[path.relative_to(root).as_posix()] = file_identity(path)
    expected = {
        path: {name: row[name] for name in ("bytes", "sha256")}
        for path, row in providers.items()
    }
    if actual != expected or canonical_digest(actual) != serving_tree_sha256:
        raise ValueError("Trained Comet serving tree differs")


def _parity_admission(admission: dict, input_root: Path) -> tuple[dict, dict]:
    parity_path = input_root / "parity-output-manifest.json"
    _exact_identity(parity_path, admission["parity_output"], "selected parity output")
    parity = _json(parity_path, "selected parity output")
    evidence, files = validate_parity_manifest(
        parity,
        selected_step=admission["selected_step"],
        checkpoint_sha256=admission["selected_checkpoint_inventory_sha256"],
        precision_sha256=admission["selected_precision_receipt_sha256"],
    )
    milestone = _validate_parity_originals(input_root, files, evidence, admission)
    rows, providers = _export_rows(files, admission["manager_step"])
    inventory = {
        "schema": "npa.behavior.orbax-params-inventory.v1",
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "files": rows,
    }
    if _canonical_sha256(inventory) != evidence.get("exported_inventory_sha256"):
        raise ValueError("Selected parity export inventory differs")
    return {"manifest": parity, "evidence": evidence, "milestone": milestone}, providers


def _normalization_source(admission: dict, parity: dict) -> dict:
    asset_id = canonical_asset_id(admission["normalization"]["asset_id"])
    manager_step = admission["selected_step"] - 1
    path = f"{manager_step}/assets/{asset_id}/norm_stats.json"
    rows = {row["path"]: row for row in parity["milestone"]["checkpoint"]["files"]}
    source = rows.get(path)
    if not isinstance(source, dict):
        raise ValueError("Selected milestone normalization is absent")
    return source


def _serving_tree(
    admission: dict, input_root: Path, checkpoint_root: Path
) -> tuple[dict, dict]:
    parity, providers = _parity_admission(admission, input_root)
    normalization, norm_provider = _admitted_file(
        checkpoint_root,
        admission["normalization"],
        "normalization",
        extra_keys={"asset_id"},
    )
    source_norm = _normalization_source(admission, parity)
    if any(
        norm_provider.get(key) != source_norm.get(key) for key in ("bytes", "sha256")
    ):
        raise ValueError("Trained Comet normalization differs from selected milestone")
    _rng_path, rng_provider = _admitted_file(
        input_root, admission["rng_contract"], "RNG contract"
    )
    relative_norm = normalization.relative_to(checkpoint_root).as_posix()
    providers[relative_norm] = norm_provider
    _verify_checkpoint_tree(
        checkpoint_root, providers, admission["serving_tree_sha256"]
    )
    expected = (
        f"{admission['manager_step']}/assets/"
        f"{canonical_asset_id(admission['normalization']['asset_id'])}/norm_stats.json"
    )
    if relative_norm != expected:
        raise ValueError("Trained Comet normalization location differs")
    return parity, rng_provider


def validate_trained_comet_admission(
    admission_path: Path,
    input_root: Path,
    checkpoint_root: Path,
    panel: dict,
    *,
    expected_serving_identity: dict,
) -> dict:
    """Validate actual parity, export bytes, normalization, and panel identity.

    Args:
        admission_path: Frozen trained-serving admission JSON.
        input_root: Root containing the provider-read parity manifest and RNG record.
        checkpoint_root: Pre-materialized BF16 params plus normalization tree.
        panel: Frozen candidate DEV or REPORT panel.
        expected_serving_identity: Identity of the executing adapter and settings.
    Returns:
        Canonical execution admission projected from validated inputs.
    Raises:
        ValueError: Lineage, provider rows, local bytes, or panel identity differ.
    """
    admission = _admission(admission_path)
    _panel_binding(panel, admission_path, expected_serving_identity)
    parity, rng_provider = _serving_tree(admission, input_root, checkpoint_root)
    evidence = parity["evidence"]
    return _execution_admission(
        admission, parity["manifest"], evidence, rng_provider, expected_serving_identity
    )


def _execution_admission(admission, parity, evidence, rng_provider, serving) -> dict:
    return {
        "schema": "npa.behavior.comet-trained-execution-admission.v1",
        "status": "selected_bf16_export_ready_for_fresh_panel_process",
        "task": admission["task"],
        "task_id": admission["task_id"],
        "selected_step": admission["selected_step"],
        "manager_step": admission["manager_step"],
        "selected_checkpoint_inventory_sha256": admission[
            "selected_checkpoint_inventory_sha256"
        ],
        "selected_precision_receipt_sha256": admission[
            "selected_precision_receipt_sha256"
        ],
        "parity_run_id": parity["run_id"],
        "parity_output": admission["parity_output"],
        "exported_inventory_sha256": evidence["exported_inventory_sha256"],
        "serving_tree_sha256": admission["serving_tree_sha256"],
        "normalization": admission["normalization"],
        "rng_contract": rng_provider,
        "trace": admission["trace"],
        "serving_identity": serving,
    }
