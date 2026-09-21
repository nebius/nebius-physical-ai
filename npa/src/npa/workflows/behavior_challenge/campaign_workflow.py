"""Generate a validated parallel workflow for one frozen BEHAVIOR campaign panel."""

from __future__ import annotations

from copy import deepcopy
from pathlib import PurePosixPath
import re
from typing import Any
from urllib.parse import urlparse

from npa.orchestration.npa_workflow.schema_validation import validate_document

from .campaign import validate_panel, validate_partition

_JsonObject = dict[str, Any]
_WorkerSlots = list[_JsonObject]
_IMAGE = re.compile(r"[^\s]+@sha256:[0-9a-f]{64}")
_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,62}")
_BASE_RUNTIME_FIELDS = (
    "upstream_root",
    "evaluator_python",
    "data_root",
    "host",
    "port",
    "policy_kind",
    "policy_root",
    "policy_python",
    "policy_checkpoint",
    "policy_archive",
    "policy_execution_variant",
)
_OPTIONAL_POLICY_FLAGS = {
    "policy_task_name": "--policy-task-name",
    "policy_selected_export_receipt": "--policy-selected-export-receipt",
    "policy_correlation_manifest": "--policy-correlation-manifest",
    "policy_validation_receipt": "--policy-validation-receipt",
    "policy_stock_correlation_asset": "--policy-stock-correlation-asset",
    "policy_stock_correlation_sha256": "--policy-stock-correlation-sha256",
}


def build_campaign_workflow(
    *,
    name: str,
    panel: _JsonObject,
    partition: _JsonObject,
    panel_uri: str,
    partition_uri: str,
    state_prefix: str,
    worker_receipts_prefix: str,
    runtime_image: str,
    runtime: _JsonObject,
    worker_slots: _WorkerSlots,
    aggregate: _JsonObject | None = None,
) -> _JsonObject:
    """Build one runtime-mode fan-out and optional aggregation barrier.

    Args:
        name: Stable workflow name.
        panel, partition: Frozen panel and deterministic worker ownership.
        panel_uri, partition_uri: Worker-readable URIs for those exact bytes.
        state_prefix: Stable case-state prefix shared by every resume.
        worker_receipts_prefix: Invocation-specific receipt prefix. It may contain
            the standard ``{{run.id}}`` token so a restarted invocation cannot
            collide with immutable worker receipts or policy provenance.
        runtime_image: Operator image pinned by a full SHA-256 digest.
        runtime: Preinstalled evaluator and managed-policy path parameters.
        worker_slots: Resource, workspace, and optional PVC for each worker.
        aggregate: Optional barrier resource, workspace, and receipt URI.
    Returns:
        JSON-compatible ``npa.workflow/v0.0.1`` document.
    Raises:
        ValueError: Frozen identities, paths, image, slots, or resources differ.

    Example:
        ``build_campaign_workflow(..., panel_uri="s3://private/panel.json",``
        ``partition_uri="s3://private/partition.json",``
        ``state_prefix="s3://private/campaign-state",``
        ``worker_receipts_prefix="s3://private/runs/{{run.id}}/workers", ...)``
    """
    valid_panel = validate_panel(panel)
    valid_partition = validate_partition(partition, valid_panel)
    return _build_workflow(
        name,
        panel_uri,
        partition_uri,
        state_prefix,
        worker_receipts_prefix,
        runtime_image,
        runtime,
        worker_slots,
        aggregate,
        valid_partition,
    )


def _build_workflow(
    name: str,
    panel_uri: str,
    partition_uri: str,
    state_prefix: str,
    worker_receipts_prefix: str,
    runtime_image: str,
    runtime: dict[str, Any],
    worker_slots: list[dict[str, Any]],
    aggregate: dict[str, Any] | None,
    partition: dict[str, Any],
) -> dict[str, Any]:
    _validate_common(name, runtime_image, panel_uri, partition_uri)
    _validate_s3_location(state_prefix, "state_prefix")
    _validate_s3_location(
        worker_receipts_prefix,
        "worker_receipts_prefix",
        allow_run_id=True,
    )
    common = _runtime_config(runtime)
    slots = _validate_slots(worker_slots, partition["worker_count"])
    resources = _resources(slots, aggregate)
    config = _config(
        runtime_image,
        panel_uri,
        partition_uri,
        state_prefix,
        worker_receipts_prefix,
        partition,
        common,
    )
    states = _states(slots, aggregate, common)
    document = {
        "apiVersion": "npa.workflow/v0.0.1",
        "kind": "Workflow",
        "metadata": {"name": name, "executionMode": "runtime"},
        "config": config,
        "resources": resources,
        "initial": "campaign-workers",
        "states": states,
    }
    validate_document(document)
    return document


def _validate_common(name: str, image: str, panel_uri: str, partition_uri: str) -> None:
    if not isinstance(name, str) or _NAME.fullmatch(name) is None:
        raise ValueError("Workflow name must be a lowercase DNS-style identifier")
    if not isinstance(image, str) or _IMAGE.fullmatch(image) is None:
        raise ValueError("Runtime image must use an immutable full SHA-256 digest")
    _validate_s3_location(panel_uri, "panel_uri")
    _validate_s3_location(partition_uri, "partition_uri")


def _validate_stable_location(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty location")
    if "{{" in value or "}}" in value:
        raise ValueError(f"{label} must remain stable across workflow resumes")
    return value.rstrip("/")


def _validate_s3_location(
    value: object,
    label: str,
    *,
    allow_run_id: bool = False,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a scoped S3 location")
    if not allow_run_id:
        candidate = _validate_stable_location(value, label)
    else:
        candidate = value.replace("{{run.id}}", "invocation")
    if "{{" in candidate or "}}" in candidate:
        raise ValueError(f"{label} contains an unsupported workflow token")
    candidate = _validate_stable_location(candidate, label)
    parsed = urlparse(candidate)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not parsed.path.strip("/")
        or parsed.query
        or parsed.fragment
        or parsed.username
        or ".." in parsed.path.split("/")
    ):
        raise ValueError(f"{label} must be a scoped S3 location")
    return value.rstrip("/")


def _runtime_config(runtime: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(runtime, dict):
        raise ValueError("runtime must be an object")
    allowed = set(_BASE_RUNTIME_FIELDS) | set(_OPTIONAL_POLICY_FLAGS)
    if set(runtime) - allowed or any(
        field not in runtime for field in _BASE_RUNTIME_FIELDS
    ):
        raise ValueError("runtime fields differ from the campaign worker contract")
    values = {field: runtime[field] for field in _BASE_RUNTIME_FIELDS}
    if any(value is None or str(value).strip() == "" for value in values.values()):
        raise ValueError("runtime base fields must be nonempty")
    paths = {field: value for field, value in values.items() if field != "port"}
    if not all(isinstance(value, str) for value in paths.values()):
        raise ValueError("runtime paths and policy selections must be strings")
    port = values["port"]
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("runtime port must be between 1 and 65535")
    optional = {key: runtime[key] for key in _OPTIONAL_POLICY_FLAGS if key in runtime}
    if any(
        not isinstance(value, str) or not value.strip() for value in optional.values()
    ):
        raise ValueError("optional policy fields must be nonempty when supplied")
    stock = {"policy_stock_correlation_asset", "policy_stock_correlation_sha256"}
    if bool(stock & set(optional)) != stock.issubset(optional):
        raise ValueError("stock correlation artifact and SHA-256 must appear together")
    return {**values, **optional}


def _validate_slots(slots: object, worker_count: int) -> list[dict[str, Any]]:
    if not isinstance(slots, list) or len(slots) != worker_count:
        raise ValueError("worker_slots must match the frozen worker count")
    normalized = []
    for slot in slots:
        if not isinstance(slot, dict) or set(slot) - {
            "worker_index",
            "resource",
            "workspace",
            "pvc",
        }:
            raise ValueError("worker slot fields differ")
        if not {"worker_index", "resource", "workspace"}.issubset(slot):
            raise ValueError("worker slot requires index, resource, and workspace")
        if type(slot["worker_index"]) is not int:
            raise ValueError("worker index must be an integer")
        normalized.append(deepcopy(slot))
    normalized.sort(key=lambda value: value["worker_index"])
    if [slot["worker_index"] for slot in normalized] != list(range(worker_count)):
        raise ValueError("Each frozen worker index must have exactly one slot")
    for slot in normalized:
        _validate_slot(slot)
    return normalized


def _validate_slot(slot: dict[str, Any]) -> None:
    if not isinstance(slot["resource"], dict) or not slot["resource"]:
        raise ValueError("worker resource must be a nonempty profile")
    if "image" in slot["resource"]:
        raise ValueError("worker resource image is controlled by runtime_image")
    workspace = slot["workspace"]
    if not isinstance(workspace, str) or not PurePosixPath(workspace).is_absolute():
        raise ValueError("worker workspace must be an absolute container path")
    pvc = slot.get("pvc")
    if pvc is None:
        return
    if not isinstance(pvc, dict) or set(pvc) != {"claim_name", "mount_path"}:
        raise ValueError("worker PVC requires claim_name and mount_path")
    if not all(isinstance(value, str) and value for value in pvc.values()):
        raise ValueError("worker PVC values must be nonempty strings")
    mount = PurePosixPath(pvc["mount_path"])
    if not mount.is_absolute() or not PurePosixPath(workspace).is_relative_to(mount):
        raise ValueError("worker workspace must be inside its writable PVC mount")


def _resources(
    slots: list[dict[str, Any]], aggregate: dict[str, Any] | None
) -> dict[str, Any]:
    resources = {}
    for slot in slots:
        name = f"campaign-worker-{slot['worker_index']}"
        resources[name] = _resource(slot)
    if aggregate is not None:
        _validate_aggregate(aggregate)
        resources["campaign-aggregate"] = _resource(aggregate)
    return resources


def _resource(slot: dict[str, Any]) -> dict[str, Any]:
    resource = deepcopy(slot["resource"])
    resource["image"] = "{{config.runtime_image}}"
    pvc = slot.get("pvc")
    if pvc is None:
        return resource
    kubernetes = resource.setdefault("kubernetes", {})
    pod = kubernetes.setdefault("pod_config", {}).setdefault("spec", {})
    volumes = pod.setdefault("volumes", [])
    containers = pod.setdefault("containers", [])
    if not isinstance(volumes, list) or not all(
        isinstance(item, dict) for item in volumes
    ):
        raise ValueError("resource pod volumes must be a list of objects")
    if not isinstance(containers, list) or not all(
        isinstance(item, dict) for item in containers
    ):
        raise ValueError("resource pod containers must be a list of objects")
    if any(item.get("name") == "campaign-workspace" for item in volumes):
        raise ValueError("resource already defines campaign-workspace volume")
    volumes.append(_workspace_volume(pvc))
    container = _ray_container(containers)
    mounts = container.setdefault("volumeMounts", [])
    if any(item.get("name") == "campaign-workspace" for item in mounts):
        raise ValueError("resource already mounts campaign-workspace")
    mounts.append({"name": "campaign-workspace", "mountPath": pvc["mount_path"]})
    return resource


def _workspace_volume(pvc: dict[str, str]) -> dict[str, Any]:
    return {
        "name": "campaign-workspace",
        "persistentVolumeClaim": {"claimName": pvc["claim_name"]},
    }


def _ray_container(containers: list[dict[str, Any]]) -> dict[str, Any]:
    matches = [item for item in containers if item.get("name") == "ray-node"]
    if len(matches) > 1:
        raise ValueError("resource defines duplicate ray-node containers")
    if matches:
        return matches[0]
    container: dict[str, Any] = {"name": "ray-node"}
    containers.append(container)
    return container


def _validate_aggregate(aggregate: object) -> None:
    if not isinstance(aggregate, dict) or set(aggregate) - {
        "resource",
        "workspace",
        "receipt_uri",
        "pvc",
    }:
        raise ValueError("aggregate fields differ")
    if not {"resource", "workspace", "receipt_uri"}.issubset(aggregate):
        raise ValueError("aggregate requires resource, workspace, and receipt_uri")
    slot = {
        key: aggregate[key]
        for key in ("resource", "workspace", "pvc")
        if key in aggregate
    }
    _validate_slot({"worker_index": 0, **slot})
    _validate_s3_location(
        aggregate["receipt_uri"],
        "aggregate receipt_uri",
        allow_run_id=True,
    )


def _config(
    image: str,
    panel_uri: str,
    partition_uri: str,
    state_prefix: str,
    receipts_prefix: str,
    partition: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source_overlay": True,
        "runtime_image": image,
        "panel_uri": panel_uri,
        "partition_uri": partition_uri,
        "state_prefix": state_prefix.rstrip("/"),
        "worker_receipts_prefix": receipts_prefix.rstrip("/"),
        "worker_count": partition["worker_count"],
        "partition_sha256": partition["partition_sha256"],
        **runtime,
    }


def _states(
    slots: list[dict[str, Any]],
    aggregate: dict[str, Any] | None,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    members = [f"campaign-worker-{slot['worker_index']}" for slot in slots]
    group: dict[str, Any] = {
        "description": "Run every frozen partition worker exactly once.",
        "parallel": members,
        "parallelCount": len(slots),
        "maxConcurrency": len(slots),
    }
    if aggregate is None:
        group["terminal"] = True
    else:
        group["next"] = "campaign-aggregate"
    states = {"campaign-workers": group}
    for slot in slots:
        index = slot["worker_index"]
        states[f"campaign-worker-{index}"] = _worker_state(slot, runtime)
    if aggregate is not None:
        states["campaign-aggregate"] = _aggregate_state(aggregate)
    return states


def _worker_state(slot: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    index = slot["worker_index"]
    receipt = f"{{{{config.worker_receipts_prefix}}}}/worker-{index}.json"
    return {
        "description": f"Run immutable campaign partition worker {index}.",
        "resources": f"campaign-worker-{index}",
        "run": {"argv": _worker_argv(index, slot["workspace"], receipt, runtime)},
        "inputs": [
            {"uri": "{{config.panel_uri}}", "schema": "npa.behavior.campaign-panel.v1"},
            {
                "uri": "{{config.partition_uri}}",
                "schema": "npa.behavior.campaign-partition.v1",
            },
        ],
        "outputs": [{"uri": receipt}],
    }


def _worker_argv(
    index: int, workspace: str, receipt: str, runtime: dict[str, Any]
) -> list[str]:
    argv = [
        "python3",
        "-m",
        "npa.workflows.behavior_challenge",
        "campaign-worker",
        "--panel-uri",
        "{{config.panel_uri}}",
        "--partition-uri",
        "{{config.partition_uri}}",
        "--worker-index",
        str(index),
        "--workspace",
        workspace,
        "--output-path",
        "{{config.state_prefix}}",
        "--worker-receipt-uri",
        receipt,
    ]
    for field in _BASE_RUNTIME_FIELDS:
        argv.extend((f"--{field.replace('_', '-')}", f"{{{{config.{field}}}}}"))
    for field, flag in _OPTIONAL_POLICY_FLAGS.items():
        if field in runtime:
            argv.extend((flag, f"{{{{config.{field}}}}}"))
    return argv


def _aggregate_state(aggregate: dict[str, Any]) -> dict[str, Any]:
    receipt = aggregate["receipt_uri"]
    argv = [
        "python3",
        "-m",
        "npa.workflows.behavior_challenge",
        "campaign-aggregate",
        "--panel-uri",
        "{{config.panel_uri}}",
        "--output-path",
        "{{config.state_prefix}}",
        "--workspace",
        aggregate["workspace"],
        "--receipt-uri",
        receipt,
    ]
    return {
        "description": "Aggregate only after every campaign worker succeeds.",
        "needs": ["campaign-workers"],
        "resources": "campaign-aggregate",
        "run": {"argv": argv},
        "inputs": [
            {"uri": "{{config.panel_uri}}", "schema": "npa.behavior.campaign-panel.v1"},
            {
                "uri": "{{config.state_prefix}}/",
                "kind": "directory",
            },
        ],
        "outputs": [{"uri": receipt, "schema": "npa.behavior.verified-panel.v1"}],
        "terminal": True,
    }
