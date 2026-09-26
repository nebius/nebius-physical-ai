"""Freeze and summarize prescribed TRAIN rollouts without reporting claims."""

from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any

from .campaign import canonical_digest, validate_policy_identity
from .protocol import EVAL_DIRECTORY, UPSTREAM_COMMITS, require_supported_upstream

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_TASK_NAME = re.compile(r"[a-z][a-z0-9_]*")
_IMPORT_PATH = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+")
_PROTOCOL_SCHEMA = "npa.behavior.nonreporting-train-protocol.v1"
_PANEL_SCHEMA = "npa.behavior.nonreporting-train-panel.v1"
_PARTITION_SCHEMA = "npa.behavior.nonreporting-train-partition.v1"
_CASE_RECEIPT_SCHEMA = "npa.behavior.nonreporting-train-case-receipt.v1"
_AGGREGATE_SCHEMA = "npa.behavior.nonreporting-train-panel-aggregate.v1"
_STUDY_SCHEMA = "npa.behavior.nonreporting-train-study.v1"
_TASK_MAPPING_SCHEMA = "npa.behavior.nonreporting-train-task-mapping.v1"
_INSPECTION_SHAPE = "npa.behavior.caller-supplied-rollout-record.v1"
_DIRECT_INSPECTION = "npa.behavior.inspect-rollout.v1"

TRAIN_PANEL_SCHEMA = _PANEL_SCHEMA
TRAIN_PARTITION_SCHEMA = _PARTITION_SCHEMA

_RANK_AGGREGATE_KEYS = {
    "aggregate_sha256",
    "artifact_bytes_verified_by_aggregator",
    "artifact_validation_contract",
    "case_count",
    "case_receipts",
    "complete",
    "development_or_report_allowed",
    "mean_q",
    "mean_steps",
    "panel_id",
    "policy_binding_sha256",
    "protocol_sha256",
    "reporting_claim_allowed",
    "schema",
    "split",
    "success_count",
    "success_rate",
    "sum_q",
    "task",
}

_SCIENCE_ARTIFACTS = {
    "action_semantics",
    "dataset",
    "dataset_view",
    "normalization",
    "task_registry",
    "tokenizer",
}
_EVALUATOR_ARTIFACTS = {
    "argv_contract",
    "controller_source",
    "evaluator_source",
    "rng_contract",
    "robot_config",
}


def _exact_keys(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} requires exactly {sorted(expected)}")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} requires a lowercase SHA-256")
    return value


def _artifact(value: object, label: str) -> dict[str, Any]:
    artifact = _exact_keys(value, {"bytes", "sha256"}, label)
    digest = _sha256(artifact["sha256"], label)
    if type(artifact["bytes"]) is not int or artifact["bytes"] <= 0:
        raise ValueError(f"{label} requires a positive byte count")
    return {"bytes": artifact["bytes"], "sha256": digest}


def _task_name(value: object) -> str:
    if not isinstance(value, str) or _TASK_NAME.fullmatch(value) is None:
        raise ValueError("TRAIN task must be a CaseStore-compatible task name")
    return value


def _task_mapping(task: str, value: object) -> dict[str, Any]:
    mapping = _exact_keys(
        value,
        {
            "data_namespace",
            "data_task_id",
            "mapping_artifact",
            "schema",
            "source_manifest",
            "split",
            "split_manifest",
            "task",
        },
        "task mapping",
    )
    if (
        mapping["schema"] != _TASK_MAPPING_SCHEMA
        or mapping["split"] != "train"
        or mapping["task"] != task
    ):
        raise ValueError("Task mapping must bind the prescribed TRAIN source split")
    namespace = mapping["data_namespace"]
    if not isinstance(namespace, str) or _IDENTIFIER.fullmatch(namespace) is None:
        raise ValueError("Task mapping namespace must be a stable identifier")
    task_id = mapping["data_task_id"]
    valid_id = type(task_id) is int and task_id >= 0
    valid_id |= (
        isinstance(task_id, str) and bool(task_id) and task_id.strip() == task_id
    )
    if not valid_id:
        raise ValueError(
            "Task mapping data_task_id must be a nonnegative int or string"
        )
    return {
        "schema": _TASK_MAPPING_SCHEMA,
        "split": "train",
        "task": task,
        "data_namespace": namespace,
        "data_task_id": task_id,
        "source_manifest": _artifact(mapping["source_manifest"], "source manifest"),
        "split_manifest": _artifact(mapping["split_manifest"], "TRAIN split manifest"),
        "mapping_artifact": _artifact(mapping["mapping_artifact"], "mapping artifact"),
    }


def _prescribed_cases(value: object) -> list[dict[str, int]]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("TRAIN protocol requires one or more prescribed cases")
    cases = [_prescribed_case(case) for case in value]
    pairs = [(case["instance_id"], case["rollout_id"]) for case in cases]
    if len(set(pairs)) != len(pairs):
        raise ValueError("Prescribed TRAIN cases must be distinct")
    return cases


def _prescribed_case(value: object) -> dict[str, int]:
    case = _exact_keys(value, {"instance_id", "rollout_id"}, "prescribed case")
    for field in ("instance_id", "rollout_id"):
        if type(case[field]) is not int or case[field] < 0:
            raise ValueError(f"Prescribed case {field} must be a nonnegative integer")
    if case["rollout_id"] != 0:
        raise ValueError("One-rollout TRAIN execution currently requires rollout_id 0")
    return {"instance_id": case["instance_id"], "rollout_id": case["rollout_id"]}


def _science_lineage(value: object) -> dict[str, Any]:
    keys = {"behavior_upstream_commit", *_SCIENCE_ARTIFACTS}
    lineage = _exact_keys(value, keys, "science lineage")
    revision = lineage["behavior_upstream_commit"]
    if not isinstance(revision, str) or _COMMIT.fullmatch(revision) is None:
        raise ValueError("Science lineage requires an exact upstream commit")
    return {
        "behavior_upstream_commit": revision,
        **{name: _artifact(lineage[name], name) for name in sorted(_SCIENCE_ARTIFACTS)},
    }


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _evaluator_contract(value: object) -> dict[str, Any]:
    keys = {
        *_EVALUATOR_ARTIFACTS,
        "executed_prefix",
        "fresh_policy_process_per_case",
        "max_steps_argument",
        "model_prediction_horizon",
        "mode",
        "num_envs",
        "num_rollouts",
        "qualification_process_discarded",
        "wrapper",
        "write_video",
    }
    contract = _exact_keys(value, keys, "evaluator contract")
    _validate_evaluator_settings(contract)
    return {
        **{
            name: _artifact(contract[name], name)
            for name in sorted(_EVALUATOR_ARTIFACTS)
        },
        "wrapper": contract["wrapper"],
        "mode": "train",
        "num_envs": 1,
        "num_rollouts": 1,
        "write_video": True,
        "max_steps_argument": None,
        "model_prediction_horizon": contract["model_prediction_horizon"],
        "executed_prefix": contract["executed_prefix"],
        "fresh_policy_process_per_case": True,
        "qualification_process_discarded": True,
    }


def _validate_evaluator_settings(contract: dict[str, Any]) -> None:
    wrapper = contract["wrapper"]
    if not isinstance(wrapper, str) or _IMPORT_PATH.fullmatch(wrapper) is None:
        raise ValueError("Evaluator wrapper must be a dotted import path")
    fixed = (
        contract["mode"],
        contract["num_envs"],
        contract["num_rollouts"],
        contract["write_video"],
    )
    if fixed != ("train", 1, 1, True) or contract["max_steps_argument"] is not None:
        raise ValueError("TRAIN cases require N=1, one rollout, video, and no max step")
    if contract["fresh_policy_process_per_case"] is not True:
        raise ValueError("TRAIN protocol requires a fresh policy process per case")
    if contract["qualification_process_discarded"] is not True:
        raise ValueError("TRAIN restore qualification must use a discarded process")
    horizon = _positive_int(contract["model_prediction_horizon"], "model horizon")
    prefix = _positive_int(contract["executed_prefix"], "executed prefix")
    if prefix > horizon:
        raise ValueError("Executed prefix cannot exceed the model horizon")


def _protocol_payload(
    task: str,
    mapping: dict[str, Any],
    cases: list[dict[str, int]],
    lineage: dict[str, Any],
    evaluator: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": _PROTOCOL_SCHEMA,
        "split": "train",
        "mode": "train",
        "task": task,
        "task_mapping": mapping,
        "prescribed_cases": cases,
        "case_count": len(cases),
        "science_lineage": lineage,
        "evaluator_contract": evaluator,
        "development_or_report_allowed": False,
        "reporting_claim_allowed": False,
    }


def declare_train_protocol(
    task: str,
    task_mapping: dict[str, Any],
    prescribed_cases: list[dict[str, int]] | tuple[dict[str, int], ...],
    science_lineage: dict[str, Any],
    evaluator_contract: dict[str, Any],
) -> dict[str, Any]:
    """Freeze a prescribed non-reporting TRAIN protocol.

    Args:
        task: Exact evaluator task name.
        task_mapping: Typed TRAIN mapping with exact source, split, and mapping
            artifact identities. This declaration does not read those artifacts.
        prescribed_cases: Ordered, distinct instances using rollout ID 0.
        science_lineage: Exact upstream, dataset, and preprocessing identities.
        evaluator_contract: Exact evaluator, controller, RNG, and action settings.
    Returns:
        Canonical protocol with a content-derived identity.
    Raises:
        ValueError: Any field is missing, ambiguous, or reporting-capable.
    """
    name = _task_name(task)
    payload = _protocol_payload(
        name,
        _task_mapping(name, task_mapping),
        _prescribed_cases(prescribed_cases),
        _science_lineage(science_lineage),
        _evaluator_contract(evaluator_contract),
    )
    return {**payload, "protocol_sha256": canonical_digest(payload)}


def validate_train_protocol(protocol: object) -> dict[str, Any]:
    """Validate and canonically reconstruct a non-reporting TRAIN protocol.

    Args:
        protocol: Protocol produced by :func:`declare_train_protocol`.
    Returns:
        Newly reconstructed canonical protocol.
    Raises:
        ValueError: Schema, content, flags, or identity differs.
    """
    keys = set(_protocol_payload("x", {}, [], {}, {})) | {"protocol_sha256"}
    value = _exact_keys(protocol, keys, "TRAIN protocol")
    if value["schema"] != _PROTOCOL_SCHEMA:
        raise ValueError("Unsupported non-reporting TRAIN protocol schema")
    expected = declare_train_protocol(
        value["task"],
        value["task_mapping"],
        value["prescribed_cases"],
        value["science_lineage"],
        value["evaluator_contract"],
    )
    if value != expected:
        raise ValueError("TRAIN protocol differs from its canonical declaration")
    return expected


def _panel_case(
    protocol: dict[str, Any], policy_sha256: str, prescribed: dict[str, int]
) -> dict[str, Any]:
    core = {"split": "train", "task": protocol["task"], **prescribed}
    identity = {
        "schema": _PANEL_SCHEMA,
        "protocol_sha256": protocol["protocol_sha256"],
        "policy_identity_sha256": policy_sha256,
        **core,
    }
    return {**core, "case_id": canonical_digest(identity)}


def _panel_payload(protocol: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    digest = policy["identity_sha256"]
    cases = [
        _panel_case(protocol, digest, case) for case in protocol["prescribed_cases"]
    ]
    return {
        "schema": _PANEL_SCHEMA,
        "protocol": protocol,
        "protocol_sha256": protocol["protocol_sha256"],
        "policy_binding": policy,
        "policy_binding_sha256": digest,
        "case_count": len(cases),
        "cases": cases,
        "development_or_report_allowed": False,
        "reporting_claim_allowed": False,
    }


def declare_train_panel(
    protocol: dict[str, Any], policy_binding: dict[str, Any]
) -> dict[str, Any]:
    """Bind one immutable policy to every case in a TRAIN protocol.

    Args:
        protocol: Valid non-reporting TRAIN protocol.
        policy_binding: Existing validated policy identity with exact checkpoint
            and serving artifact identities.
    Returns:
        Canonical panel with stable case and panel identities.
    Raises:
        ValueError: Protocol or policy binding is invalid.
    """
    payload = _panel_payload(
        validate_train_protocol(protocol), validate_policy_identity(policy_binding)
    )
    return {**payload, "panel_id": canonical_digest(payload)}


def validate_train_panel(panel: object) -> dict[str, Any]:
    """Validate and canonically reconstruct a non-reporting TRAIN panel.

    Args:
        panel: Panel produced by :func:`declare_train_panel`.
    Returns:
        Newly reconstructed canonical panel.
    Raises:
        ValueError: Protocol, policy, cases, flags, or identity differs.
    """
    keys = {
        "case_count",
        "cases",
        "development_or_report_allowed",
        "panel_id",
        "policy_binding",
        "policy_binding_sha256",
        "protocol",
        "protocol_sha256",
        "reporting_claim_allowed",
        "schema",
    }
    value = _exact_keys(panel, keys, "TRAIN panel")
    if value["schema"] != _PANEL_SCHEMA:
        raise ValueError("Unsupported non-reporting TRAIN panel schema")
    expected = declare_train_panel(value["protocol"], value["policy_binding"])
    if value != expected:
        raise ValueError("TRAIN panel differs from its canonical declaration")
    return expected


def _partition_payload(panel: dict[str, Any], worker_count: int) -> dict[str, Any]:
    workers = []
    for worker_index in range(worker_count):
        case_ids = [
            case["case_id"]
            for index, case in enumerate(panel["cases"])
            if index % worker_count == worker_index
        ]
        workers.append({"worker_index": worker_index, "case_ids": case_ids})
    return {
        "schema": _PARTITION_SCHEMA,
        "panel_id": panel["panel_id"],
        "worker_count": worker_count,
        "workers": workers,
    }


def partition_train_panel(panel: dict[str, Any], worker_count: int) -> dict[str, Any]:
    """Assign every prescribed TRAIN case once by deterministic round robin.

    Args:
        panel: Valid non-reporting TRAIN panel.
        worker_count: Positive count no larger than the number of cases.
    Returns:
        Canonical worker partition bound to the panel.
    Raises:
        ValueError: Panel or worker count is invalid.
    """
    valid = validate_train_panel(panel)
    if type(worker_count) is not int or not 1 <= worker_count <= valid["case_count"]:
        raise ValueError("worker_count must fit inside the TRAIN panel")
    payload = _partition_payload(valid, worker_count)
    return {**payload, "partition_sha256": canonical_digest(payload)}


def validate_train_partition(
    partition: object, panel: dict[str, Any]
) -> dict[str, Any]:
    """Validate exact deterministic ownership for a TRAIN panel.

    Args:
        partition: Partition produced by :func:`partition_train_panel`.
        panel: Referenced non-reporting TRAIN panel.
    Returns:
        Newly reconstructed canonical partition.
    Raises:
        ValueError: Ownership, panel binding, or identity differs.
    """
    value = _exact_keys(
        partition,
        {"schema", "panel_id", "worker_count", "workers", "partition_sha256"},
        "TRAIN partition",
    )
    if value["schema"] != _PARTITION_SCHEMA:
        raise ValueError("Unsupported non-reporting TRAIN partition schema")
    expected = partition_train_panel(panel, value["worker_count"])
    if value != expected:
        raise ValueError("TRAIN partition differs from deterministic ownership")
    return expected


def train_evaluator_argv(
    protocol: dict[str, Any],
    case: dict[str, Any],
    *,
    root: Path,
    python: str,
    host: str,
    port: int,
    output: Path,
) -> list[str]:
    """Construct one exact official evaluator invocation for a TRAIN case.

    Args:
        protocol: Valid non-reporting TRAIN protocol.
        case: One exact case derived from that protocol.
        root: Verified upstream checkout.
        python: Interpreter with the upstream evaluator installed.
        host: Policy WebSocket host.
        port: Policy WebSocket port.
        output: New case output directory.
    Returns:
        Argument vector for one TRAIN instance and rollout zero.
    Raises:
        ValueError: Protocol, case, revision, or endpoint differs.
    """
    valid = validate_train_protocol(protocol)
    expected_cases = {
        (item["instance_id"], item["rollout_id"]) for item in valid["prescribed_cases"]
    }
    prescribed = _prescribed_case(
        {"instance_id": case.get("instance_id"), "rollout_id": case.get("rollout_id")}
    )
    if (
        case.get("task") != valid["task"]
        or case.get("split") != "train"
        or (prescribed["instance_id"], prescribed["rollout_id"]) not in expected_cases
    ):
        raise ValueError("Evaluator case is not prescribed by the TRAIN protocol")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("Policy port must be between 1 and 65535")
    revision = require_supported_upstream(
        valid["science_lineage"]["behavior_upstream_commit"]
    )
    contract = valid["evaluator_contract"]
    argv = [
        python,
        "-m",
        "omnigibson.eval.eval",
        "--task-name",
        valid["task"],
        "--mode",
        "train",
        "--instance-indices",
        str(prescribed["instance_id"]),
        "--num-rollouts",
        str(contract["num_rollouts"]),
        "--env-wrapper",
        contract["wrapper"],
        "--robot-config",
        str(root / EVAL_DIRECTORY / "r1pro.yaml"),
        "--host",
        host,
        "--port",
        str(port),
        "--output-dir",
        str(output),
    ]
    if revision == UPSTREAM_COMMITS["3.9.3"]:
        argv.extend(("--num-envs", "1", "--replay-action-chunk-size", "0"))
    return argv + ["--write-video", "--headless"]


def _rollout_files(record: dict[str, Any], case: dict[str, Any]) -> dict[str, str]:
    stem = f"{case['task']}_{case['instance_id']}_{case['rollout_id']}"
    expected = {f"json/{stem}.json", f"videos/{stem}.mp4"}
    files = record["files"]
    if not isinstance(files, dict) or set(files) != expected:
        raise ValueError("Inspected TRAIN rollout requires exact JSON and video files")
    return {name: _sha256(files[name], name) for name in sorted(files)}


def _rollout_measurements(record: dict[str, Any]) -> dict[str, Any]:
    score = record["q_score"]
    if (
        type(score) not in (int, float)
        or not math.isfinite(score)
        or not 0 <= score <= 1
    ):
        raise ValueError("Inspected TRAIN q_score must be finite and in [0, 1]")
    if type(record["success"]) is not bool:
        raise ValueError("Inspected TRAIN success must be boolean")
    if type(record["steps"]) is not int or record["steps"] <= 0:
        raise ValueError("Inspected TRAIN steps must be a positive integer")
    if type(record["video_frames"]) is not int or record["video_frames"] <= 0:
        raise ValueError("Inspected TRAIN video must contain decoded frames")
    if record["video_frames"] != record["steps"]:
        raise ValueError("TRAIN decoded video frame count must equal evaluator steps")
    return {
        "q_score": float(score),
        "success": record["success"],
        "steps": record["steps"],
        "video_frames": record["video_frames"],
    }


def _bind_inspected_rollout(
    panel: dict[str, Any],
    inspected: object,
    *,
    artifact_validation_contract: str,
) -> dict[str, Any]:
    fields = {
        "case_id",
        "files",
        "instance_id",
        "q_score",
        "rollout_id",
        "split",
        "steps",
        "success",
        "task",
        "video_frames",
    }
    record = _exact_keys(inspected, fields, "inspected TRAIN rollout")
    case_fields = {"split", "task", "instance_id", "rollout_id", "case_id"}
    case = {name: record[name] for name in case_fields}
    expected = {item["case_id"]: item for item in panel["cases"]}
    if case.get("case_id") not in expected or case != expected[case["case_id"]]:
        raise ValueError("Inspected rollout is not a prescribed TRAIN case")
    payload = {
        "schema": _CASE_RECEIPT_SCHEMA,
        "panel_id": panel["panel_id"],
        "protocol_sha256": panel["protocol_sha256"],
        "policy_binding_sha256": panel["policy_binding_sha256"],
        "case": case,
        **_rollout_measurements(record),
        "files": _rollout_files(record, case),
        "artifact_validation_contract": artifact_validation_contract,
    }
    return {**payload, "case_receipt_sha256": canonical_digest(payload)}


def _aggregate_payload(
    panel: dict[str, Any],
    receipts: list[dict[str, Any]],
    *,
    artifact_validation_contract: str,
    artifact_bytes_verified_by_aggregator: bool,
) -> dict[str, Any]:
    scores = [receipt["q_score"] for receipt in receipts]
    successes = sum(receipt["success"] for receipt in receipts)
    return {
        "schema": _AGGREGATE_SCHEMA,
        "panel_id": panel["panel_id"],
        "protocol_sha256": panel["protocol_sha256"],
        "policy_binding_sha256": panel["policy_binding_sha256"],
        "split": "train",
        "task": panel["protocol"]["task"],
        "case_count": len(receipts),
        "complete": True,
        "sum_q": math.fsum(scores),
        "mean_q": math.fsum(scores) / len(scores),
        "success_count": successes,
        "success_rate": successes / len(receipts),
        "mean_steps": math.fsum(item["steps"] for item in receipts) / len(receipts),
        "case_receipts": receipts,
        "artifact_validation_contract": artifact_validation_contract,
        "artifact_bytes_verified_by_aggregator": (
            artifact_bytes_verified_by_aggregator
        ),
        "development_or_report_allowed": False,
        "reporting_claim_allowed": False,
    }


def aggregate_train_panel(
    panel: dict[str, Any], inspected_cases: list[dict[str, Any]]
) -> dict[str, Any]:
    """Aggregate complete caller-supplied rollout records for one TRAIN panel.

    Args:
        panel: Valid non-reporting TRAIN panel.
        inspected_cases: Inspection-shaped records for every prescribed case. This
            pure function does not open or decode the referenced artifacts.
    Returns:
        Complete canonical descriptive aggregate with bound case receipts.
    Raises:
        ValueError: A case is malformed, missing, duplicated, or extra.
    """
    valid = validate_train_panel(panel)
    if not isinstance(inspected_cases, list):
        raise ValueError("Inspected TRAIN cases must be a list")
    return _aggregate_train_records(
        valid,
        inspected_cases,
        artifact_validation_contract=_INSPECTION_SHAPE,
        artifact_bytes_verified_by_aggregator=False,
    )


def bind_train_rollout_record(
    panel: dict[str, Any], inspected: object
) -> dict[str, Any]:
    """Validate one inspection-shaped record against a TRAIN panel.

    This pure validator does not open the referenced files. The file-backed
    aggregate entrypoint performs that stronger operation.
    """
    return _bind_inspected_rollout(
        validate_train_panel(panel),
        inspected,
        artifact_validation_contract=_INSPECTION_SHAPE,
    )


def _aggregate_train_records(
    panel: dict[str, Any],
    inspected_cases: list[dict[str, Any]],
    *,
    artifact_validation_contract: str,
    artifact_bytes_verified_by_aggregator: bool,
) -> dict[str, Any]:
    if not isinstance(inspected_cases, list):
        raise ValueError("Inspected TRAIN cases must be a list")
    receipts = [
        _bind_inspected_rollout(
            panel,
            item,
            artifact_validation_contract=artifact_validation_contract,
        )
        for item in inspected_cases
    ]
    by_case = {item["case"]["case_id"]: item for item in receipts}
    expected = [case["case_id"] for case in panel["cases"]]
    if len(by_case) != len(receipts) or set(by_case) != set(expected):
        raise ValueError("TRAIN aggregation requires every prescribed case once")
    payload = _aggregate_payload(
        panel,
        [by_case[case_id] for case_id in expected],
        artifact_validation_contract=artifact_validation_contract,
        artifact_bytes_verified_by_aggregator=artifact_bytes_verified_by_aggregator,
    )
    return {**payload, "aggregate_sha256": canonical_digest(payload)}


def aggregate_train_outputs(
    panel: dict[str, Any], case_outputs: dict[str, Path]
) -> dict[str, Any]:
    """Inspect and aggregate every prescribed TRAIN case from local originals.

    Args:
        panel: Valid non-reporting TRAIN panel.
        case_outputs: Exact case-ID to output-directory mapping.
    Returns:
        Complete aggregate whose verification flag reflects direct file reads and
        full video decoding performed by this call.
    Raises:
        ValueError: Outputs are missing, duplicated, extra, or malformed.
        OSError: An original artifact cannot be read.
    """
    from .artifacts import inspect_rollout

    valid = validate_train_panel(panel)
    if not isinstance(case_outputs, dict) or any(
        not isinstance(key, str) or not isinstance(value, Path)
        for key, value in case_outputs.items()
    ):
        raise ValueError("TRAIN output mapping requires case IDs and Path values")
    expected = {case["case_id"]: case for case in valid["cases"]}
    if set(case_outputs) != set(expected):
        raise ValueError("TRAIN output mapping requires every prescribed case once")
    for case_id, output in case_outputs.items():
        case = expected[case_id]
        stem = f"{case['task']}_{case['instance_id']}_{case['rollout_id']}"
        required = (
            output / "json" / f"{stem}.json",
            output / "videos" / f"{stem}.mp4",
        )
        if output.is_symlink() or not output.is_dir():
            raise ValueError("TRAIN case output must be a real directory")
        for directory in (output / "json", output / "videos"):
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError("TRAIN artifact directory must be real")
        if any(path.is_symlink() or not path.is_file() for path in required):
            raise ValueError("TRAIN artifacts must be regular non-symlink files")
    inspected = [
        inspect_rollout(case_outputs[case["case_id"]], case) for case in valid["cases"]
    ]
    return _aggregate_train_records(
        valid,
        inspected,
        artifact_validation_contract=_DIRECT_INSPECTION,
        artifact_bytes_verified_by_aggregator=True,
    )


def _receipt_to_inspected(receipt: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(receipt, dict) or not isinstance(receipt.get("case"), dict):
        raise ValueError("TRAIN aggregate receipt must contain one case")
    required = {"q_score", "success", "steps", "video_frames", "files"}
    if not required.issubset(receipt):
        raise ValueError("TRAIN aggregate receipt is missing inspected evidence")
    return {
        **receipt["case"],
        **{
            name: receipt[name]
            for name in ("q_score", "success", "steps", "video_frames")
        },
        "files": receipt["files"],
    }


def validate_train_panel_aggregate(
    aggregate: object, panel: dict[str, Any]
) -> dict[str, Any]:
    """Validate a complete TRAIN aggregate against its immutable panel.

    Args:
        aggregate: Aggregate produced by :func:`aggregate_train_panel`.
        panel: Referenced non-reporting TRAIN panel.
    Returns:
        Newly reconstructed canonical aggregate.
    Raises:
        ValueError: Evidence is incomplete, altered, or belongs to another panel.
    """
    if not isinstance(aggregate, dict) or not isinstance(
        aggregate.get("case_receipts"), list
    ):
        raise ValueError("TRAIN aggregate requires case receipts")
    inspected = [_receipt_to_inspected(item) for item in aggregate["case_receipts"]]
    contract = aggregate.get("artifact_validation_contract")
    verified = aggregate.get("artifact_bytes_verified_by_aggregator")
    if (contract, verified) not in {
        (_INSPECTION_SHAPE, False),
        (_DIRECT_INSPECTION, True),
    }:
        raise ValueError("TRAIN aggregate has an unsupported evidence boundary")
    expected = _aggregate_train_records(
        validate_train_panel(panel),
        inspected,
        artifact_validation_contract=contract,
        artifact_bytes_verified_by_aggregator=verified,
    )
    if aggregate != expected:
        raise ValueError("TRAIN aggregate differs from canonical inspected evidence")
    return expected


def _study_row(value: object) -> dict[str, Any]:
    keys = {
        "aggregate_sha256",
        "case_count",
        "panel_id",
        "policy_binding_sha256",
        "policy_key",
        "primary_loss",
        "protocol_sha256",
        "secondary_loss",
    }
    row = _exact_keys(value, keys, "TRAIN study row")
    if (
        not isinstance(row["policy_key"], str)
        or _IDENTIFIER.fullmatch(row["policy_key"]) is None
    ):
        raise ValueError("Study policy_key must be a stable identifier")
    for field in (
        "aggregate_sha256",
        "panel_id",
        "policy_binding_sha256",
        "protocol_sha256",
    ):
        _sha256(row[field], field)
    _positive_int(row["case_count"], "study case_count")
    for field in ("primary_loss", "secondary_loss"):
        if type(row[field]) not in (int, float) or not math.isfinite(row[field]):
            raise ValueError(f"Study {field} must be finite")
    return {
        **{
            field: row[field]
            for field in sorted(keys - {"primary_loss", "secondary_loss"})
        },
        "primary_loss": float(row["primary_loss"]),
        "secondary_loss": float(row["secondary_loss"]),
    }


def _distinct_study_rows(rows: list[dict[str, Any]]) -> None:
    for field in (
        "aggregate_sha256",
        "panel_id",
        "policy_binding_sha256",
        "policy_key",
    ):
        values = [row[field] for row in rows]
        if len(set(values)) != len(values):
            raise ValueError(f"TRAIN study {field} values must be distinct")


def _study_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": _STUDY_SCHEMA,
        "policy_count": len(rows),
        "case_count": rows[0]["case_count"] if rows else 0,
        "rows": rows,
        "calibration_order": "ascending_primary_loss_then_secondary_loss",
        "closed_loop_order": "descending_mean_q_then_success_count",
        "rank_orientation": "rank_1_is_best",
        "ties": "complete_key_uses_shared_midrank",
        "claim": "descriptive_only_no_selection_or_significance",
    }


def declare_train_study(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Freeze calibration and aggregate identities for descriptive rank agreement.

    Args:
        rows: Ordered policy, panel, aggregate, and two-loss bindings.
    Returns:
        Canonical cardinality-independent study declaration.
    Raises:
        ValueError: Rows are missing, duplicated, non-finite, or unbound.
    """
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("TRAIN study requires at least two policy rows")
    normalized = [_study_row(row) for row in rows]
    _distinct_study_rows(normalized)
    if len({row["protocol_sha256"] for row in normalized}) != 1:
        raise ValueError("TRAIN study policies must share one protocol")
    if len({row["case_count"] for row in normalized}) != 1:
        raise ValueError("TRAIN study policies must share one case cardinality")
    payload = _study_payload(normalized)
    return {**payload, "study_sha256": canonical_digest(payload)}


def validate_train_study(study: object) -> dict[str, Any]:
    """Validate and canonically reconstruct a descriptive TRAIN study.

    Args:
        study: Study produced by :func:`declare_train_study`.
    Returns:
        Newly reconstructed canonical study.
    Raises:
        ValueError: Fields, cardinality, rows, or digest differ.
    """
    keys = set(_study_payload([])) | {"study_sha256"}
    value = _exact_keys(study, keys, "TRAIN study")
    if value["schema"] != _STUDY_SCHEMA:
        raise ValueError("Unsupported non-reporting TRAIN study schema")
    expected = declare_train_study(value["rows"])
    if value != expected:
        raise ValueError("TRAIN study differs from its canonical declaration")
    return expected


def _rank_aggregate(value: object) -> dict[str, Any]:
    aggregate = _exact_keys(value, _RANK_AGGREGATE_KEYS, "rank aggregate")
    if aggregate["schema"] != _AGGREGATE_SCHEMA:
        raise ValueError("Rank input must be a non-reporting TRAIN aggregate")
    digest = aggregate["aggregate_sha256"]
    _sha256(digest, "aggregate identity")
    payload = {
        key: item for key, item in aggregate.items() if key != "aggregate_sha256"
    }
    if canonical_digest(payload) != digest:
        raise ValueError("Rank aggregate identity differs from its content")
    if aggregate["complete"] is not True or aggregate["split"] != "train":
        raise ValueError("Rank input requires a complete TRAIN aggregate")
    fixed = (
        aggregate["artifact_validation_contract"],
        aggregate["artifact_bytes_verified_by_aggregator"],
        aggregate["development_or_report_allowed"],
        aggregate["reporting_claim_allowed"],
    )
    if fixed != (_DIRECT_INSPECTION, True, False, False):
        raise ValueError("Rank aggregate crosses its non-reporting evidence boundary")
    for field in ("panel_id", "policy_binding_sha256", "protocol_sha256"):
        _sha256(aggregate[field], field)
    _task_name(aggregate["task"])
    _validate_rank_receipts(aggregate)
    return aggregate


def _validate_rank_receipts(aggregate: dict[str, Any]) -> None:
    receipts = aggregate["case_receipts"]
    if not isinstance(receipts, list) or not receipts:
        raise ValueError("Rank aggregate requires nonempty case receipts")
    case_ids = [_validate_standalone_receipt(item, aggregate) for item in receipts]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Rank aggregate case receipts must be distinct")
    scores = [item["q_score"] for item in receipts]
    successes = sum(item["success"] for item in receipts)
    mean_steps = math.fsum(item["steps"] for item in receipts) / len(receipts)
    expected = (len(receipts), math.fsum(scores), successes, mean_steps)
    types = (
        type(aggregate["case_count"]) is int,
        type(aggregate["sum_q"]) is float,
        type(aggregate["success_count"]) is int,
        type(aggregate["mean_steps"]) is float,
        type(aggregate["mean_q"]) is float,
        type(aggregate["success_rate"]) is float,
    )
    if not all(types):
        raise ValueError("Rank aggregate summaries require canonical numeric types")
    observed = (
        aggregate["case_count"],
        aggregate["sum_q"],
        aggregate["success_count"],
        aggregate["mean_steps"],
    )
    if observed != expected:
        raise ValueError("Rank aggregate summaries differ from case receipts")
    if aggregate["mean_q"] != expected[1] / expected[0]:
        raise ValueError("Rank aggregate mean_q differs from case receipts")
    if aggregate["success_rate"] != expected[2] / expected[0]:
        raise ValueError("Rank aggregate success rate differs from case receipts")


def _validate_standalone_receipt(value: object, aggregate: dict[str, Any]) -> str:
    keys = {
        "artifact_validation_contract",
        "case",
        "case_receipt_sha256",
        "files",
        "panel_id",
        "policy_binding_sha256",
        "protocol_sha256",
        "q_score",
        "schema",
        "steps",
        "success",
        "video_frames",
    }
    receipt = _exact_keys(value, keys, "rank case receipt")
    if receipt["schema"] != _CASE_RECEIPT_SCHEMA:
        raise ValueError("Rank case receipt schema differs")
    if receipt["artifact_validation_contract"] != _DIRECT_INSPECTION:
        raise ValueError("Rank case receipt differs from its declared evidence shape")
    for field in ("panel_id", "policy_binding_sha256", "protocol_sha256"):
        if receipt[field] != aggregate[field]:
            raise ValueError(f"Rank case receipt {field} differs")
    _rollout_measurements(receipt)
    case = _rank_case(receipt["case"], aggregate)
    _rollout_files(receipt, case)
    payload = {
        key: item for key, item in receipt.items() if key != "case_receipt_sha256"
    }
    if canonical_digest(payload) != receipt["case_receipt_sha256"]:
        raise ValueError("Rank case receipt identity differs from content")
    return case["case_id"]


def _rank_case(value: object, aggregate: dict[str, Any]) -> dict[str, Any]:
    case = _exact_keys(
        value,
        {"case_id", "instance_id", "rollout_id", "split", "task"},
        "rank case",
    )
    if case["split"] != "train" or case["task"] != aggregate["task"]:
        raise ValueError("Rank case differs from aggregate TRAIN task")
    _prescribed_case(
        {"instance_id": case["instance_id"], "rollout_id": case["rollout_id"]}
    )
    identity = {
        "schema": _PANEL_SCHEMA,
        "protocol_sha256": aggregate["protocol_sha256"],
        "policy_identity_sha256": aggregate["policy_binding_sha256"],
        **{key: case[key] for key in ("split", "task", "instance_id", "rollout_id")},
    }
    if case["case_id"] != canonical_digest(identity):
        raise ValueError("Rank case identity differs from its panel binding")
    return case


def pareto_dominates(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """Return whether the first closed-loop result Pareto-dominates the second.

    Args:
        first: Mapping with finite ``mean_q`` and integer ``success_count``.
        second: Mapping with finite ``mean_q`` and integer ``success_count``.
    Returns:
        True when first is no worse on both metrics and better on at least one.
    Raises:
        ValueError: Either result lacks valid closed-loop metrics.
    """
    left = _closed_loop_pair(first)
    right = _closed_loop_pair(second)
    no_worse = left[0] >= right[0] and left[1] >= right[1]
    return no_worse and left != right


def _closed_loop_pair(value: object) -> tuple[float, int]:
    if not isinstance(value, dict):
        raise ValueError("Closed-loop result must be an object")
    mean_q = value.get("mean_q")
    successes = value.get("success_count")
    if type(mean_q) not in (int, float) or not math.isfinite(mean_q):
        raise ValueError("Closed-loop mean_q must be finite")
    if not 0 <= mean_q <= 1:
        raise ValueError("Closed-loop mean_q must be in [0, 1]")
    if type(successes) is not int or successes < 0:
        raise ValueError("Closed-loop success_count must be a nonnegative integer")
    return float(mean_q), successes


def _midranks(keys: list[tuple[float, ...]]) -> list[float]:
    order = sorted(range(len(keys)), key=keys.__getitem__)
    ranks = [0.0] * len(keys)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and keys[order[end]] == keys[order[start]]:
            end += 1
        rank = ((start + 1) + end) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _kendall_counts(first: list[float], second: list[float]) -> dict[str, int]:
    counts = {
        "P_concordant": 0,
        "Q_discordant": 0,
        "T_c_calibration_only_ties": 0,
        "T_r_rollout_only_ties": 0,
        "T_joint_ties": 0,
    }
    for left in range(len(first)):
        for right in range(left + 1, len(first)):
            first_delta = first[left] - first[right]
            second_delta = second[left] - second[right]
            if first_delta == second_delta == 0:
                counts["T_joint_ties"] += 1
            elif first_delta == 0:
                counts["T_c_calibration_only_ties"] += 1
            elif second_delta == 0:
                counts["T_r_rollout_only_ties"] += 1
            elif first_delta * second_delta > 0:
                counts["P_concordant"] += 1
            else:
                counts["Q_discordant"] += 1
    return counts


def _kendall_tau(first: list[float], second: list[float]) -> dict[str, Any]:
    counts = _kendall_counts(first, second)
    common = counts["P_concordant"] + counts["Q_discordant"]
    denominator = math.sqrt(
        (common + counts["T_c_calibration_only_ties"])
        * (common + counts["T_r_rollout_only_ties"])
    )
    value = (
        None
        if denominator == 0
        else (counts["P_concordant"] - counts["Q_discordant"]) / denominator
    )
    return {
        **counts,
        "pair_count": len(first) * (len(first) - 1) // 2,
        "value": value,
        "undefined_reason": "zero_denominator" if value is None else None,
    }


def _spearman(first: list[float], second: list[float]) -> dict[str, Any]:
    first_mean = math.fsum(first) / len(first)
    second_mean = math.fsum(second) / len(second)
    first_centered = [value - first_mean for value in first]
    second_centered = [value - second_mean for value in second]
    numerator = math.fsum(
        a * b for a, b in zip(first_centered, second_centered, strict=True)
    )
    denominator = math.sqrt(
        math.fsum(value * value for value in first_centered)
        * math.fsum(value * value for value in second_centered)
    )
    value = None if denominator == 0 else numerator / denominator
    return {
        "value": value,
        "undefined_reason": "zero_variance" if value is None else None,
    }


def _rank_rows(
    study: dict[str, Any], aggregates: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for declared in study["rows"]:
        aggregate = aggregates[declared["aggregate_sha256"]]
        joins = (
            aggregate["panel_id"] == declared["panel_id"],
            aggregate["policy_binding_sha256"] == declared["policy_binding_sha256"],
            aggregate["protocol_sha256"] == declared["protocol_sha256"],
            aggregate["case_count"] == declared["case_count"],
        )
        if not all(joins):
            raise ValueError("TRAIN aggregate differs from its study row bindings")
        rows.append(
            {
                **declared,
                "mean_q": aggregate["mean_q"],
                "success_count": aggregate["success_count"],
            }
        )
    return rows


def _case_keys(aggregate: dict[str, Any]) -> list[tuple[str, int, int]]:
    return [
        (
            receipt["case"]["task"],
            receipt["case"]["instance_id"],
            receipt["case"]["rollout_id"],
        )
        for receipt in aggregate["case_receipts"]
    ]


def _validate_matched_aggregates(values: list[dict[str, Any]]) -> None:
    if len({value["protocol_sha256"] for value in values}) != 1:
        raise ValueError("TRAIN rank aggregates must share one protocol")
    if any(_case_keys(value) != _case_keys(values[0]) for value in values[1:]):
        raise ValueError("TRAIN rank aggregates must contain the same prescribed cases")


def _pareto_relations(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        others = [other for other in rows if other is not row]
        row["dominates"] = [
            other["policy_key"] for other in others if pareto_dominates(row, other)
        ]
        row["dominated_by"] = [
            other["policy_key"] for other in others if pareto_dominates(other, row)
        ]


def _rank_payload(
    study: dict[str, Any],
    rows: list[dict[str, Any]],
    calibration: list[float],
    closed_loop: list[float],
) -> dict[str, Any]:
    return {
        "schema": "npa.behavior.nonreporting-train-rank-agreement.v1",
        "study_sha256": study["study_sha256"],
        "policy_count": len(rows),
        "rows": rows,
        "calibration_midrank_vector": calibration,
        "closed_loop_midrank_vector": closed_loop,
        "kendall_tau_b": _kendall_tau(calibration, closed_loop),
        "spearman_midrank": _spearman(calibration, closed_loop),
        "claim": "descriptive_only_no_selection_or_significance",
    }


def rank_train_study(
    study: dict[str, Any], complete_aggregates: list[dict[str, Any]]
) -> dict[str, Any]:
    """Measure descriptive rank agreement for complete TRAIN panel aggregates.

    Args:
        study: Frozen cardinality-independent study declaration.
        complete_aggregates: One complete aggregate for every declared policy.
    Returns:
        Tie-aware ranks, correlations, pair counts, and Pareto relations.
    Raises:
        ValueError: Study or aggregate coverage and identities are invalid.
    """
    declared = validate_train_study(study)
    if not isinstance(complete_aggregates, list):
        raise ValueError("TRAIN rank aggregates must be a list")
    values = [_rank_aggregate(item) for item in complete_aggregates]
    _validate_matched_aggregates(values)
    aggregates = {item["aggregate_sha256"]: item for item in values}
    expected = {row["aggregate_sha256"] for row in declared["rows"]}
    if len(aggregates) != len(values) or set(aggregates) != expected:
        raise ValueError("TRAIN ranking requires every declared aggregate exactly once")
    rows = _rank_rows(declared, aggregates)
    calibration = _midranks(
        [(row["primary_loss"], row["secondary_loss"]) for row in rows]
    )
    closed_loop = _midranks([(-row["mean_q"], -row["success_count"]) for row in rows])
    for row, calibration_rank, rollout_rank in zip(
        rows, calibration, closed_loop, strict=True
    ):
        row["calibration_rank"] = calibration_rank
        row["closed_loop_rank"] = rollout_rank
    _pareto_relations(rows)
    payload = _rank_payload(declared, rows, calibration, closed_loop)
    return {**payload, "rank_agreement_sha256": canonical_digest(payload)}
