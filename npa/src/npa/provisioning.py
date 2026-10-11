"""Additive-only runtime provisioning hooks."""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import logging
import os
import re
import shlex
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping
from urllib.parse import urlparse

from npa.clients import config as config_module
from npa.clients.config import EnvironmentConfig, StorageConfig
from npa.cluster.gpu_driver import DEFAULT_MANAGED_DRIVER_PRESET
from npa.cluster.gpu_health import (
    DEFAULT_CUDA_SMOKE_IMAGE,
    DEFAULT_GRAPHICS_SMOKE_IMAGE,
    DEFAULT_STABILIZATION_SECONDS,
    resolve_graphics_smoke_image,
)
from npa.cluster.gpu_workload_profile import resolve_gpu_workload_profile
from npa.cluster.state import cluster_dir, kubeconfig_file, load_cluster_state
from npa.provisioning_journal import (
    OperationJournalError,
    OperationIdentityError,
    ProvisioningOperation,
    current_operation,
    deterministic_operation_id,
    emit_recovery_summary,
    operation_context,
)


@dataclass
class ProvisionIfAbsentResult:
    status: str
    project: str
    cluster_name: str
    kubeconfig_path: str = ""
    context_name: str = ""
    storage_bucket: str = ""
    storage_endpoint: str = ""
    registry: str = ""
    gpu_readiness: str = "not-requested"
    operation_id: str = ""
    operation_journal: str = ""
    recovery_command: str = ""
    actions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    preflight: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_RECOVERY_VALUE_FLAGS = frozenset(
    {
        "--project",
        "--cluster-name",
        "--context",
        "--kubeconfig",
        "--terraform-dir",
        "--gpu-nodes",
        "--gpu-platform",
        "--gpu-preset",
        "--accelerator",
        "--output-format",
    }
)
_RECOVERY_BOOLEAN_FLAGS = frozenset({"--skip-s3", "--on-demand"})
_STANDARD_TFVARS = frozenset(
    {
        "cluster_name",
        "ssh_public_key",
        "mk8s_cluster_public_endpoint",
        "enable_k8s_node_group_sa",
        "enable_egress_gateway",
        "cpu_nodes_public_ips",
        "gpu_nodes_public_ips",
        "cpu_nodes_preemptible",
        "gpu_nodes_preemptible",
        "mig_strategy",
        "gpu_nodes_driverfull_image",
        "gpu_nodes_driver_preset",
        "custom_driver",
        "cpu_nodes_fixed_count",
        "cpu_nodes_platform",
        "cpu_nodes_preset",
        "cpu_disk_size",
        "gpu_nodes_fixed_count_per_group",
        "gpu_node_groups",
        "gpu_nodes_platform",
        "gpu_nodes_preset",
        "gpu_disk_size",
        "enable_gpu_cluster",
        "enable_filestore",
        "existing_filestore",
        "filestore_disk_size_gibibytes",
        "filestore_mount_path",
        "filestore_mount_tag",
        "filesystem_csi",
        "enable_nebius_o11y_agent",
        "enable_grafana",
        "enable_prometheus",
        "collectK8sClusterMetrics",
        "enable_kuberay_cluster",
        "enable_kuberay_service",
        "enable_opa_gatekeeper",
        "loki",
    }
)
_STATE_RESOURCE_TYPES = frozenset(
    {
        ("data", "nebius_compute_v1_filesystem"),
        ("data", "nebius_iam_v1_group"),
        ("managed", "nebius_compute_v1_filesystem"),
        ("managed", "nebius_compute_v1_gpu_cluster"),
        ("managed", "nebius_iam_v1_group_membership"),
        ("managed", "nebius_iam_v1_service_account"),
        ("managed", "nebius_iam_v2_access_key"),
        ("managed", "random_password"),
        ("managed", "random_string"),
        ("managed", "terraform_data"),
        ("managed", "time_static"),
    }
)


@dataclass(frozen=True)
class EmptyFailureRecoveryProfile:
    """Evidence-backed, supported producer shape for one empty recovery."""

    retained_hashes: dict[str, str]
    topology: dict[str, object]
    expected_names: dict[str, str]


@dataclass(frozen=True)
class RetainedBackendIdentity:
    """The recorded recipe identity required to validate retained backend bytes."""

    hashes: dict[str, str]
    source_hashes: dict[str, str]
    provider_defaults: dict[str, str]


def _journal_sha256(operation: ProvisioningOperation) -> str:
    """Return a journal byte digest without following a replacement symlink."""

    if operation.path.is_symlink() or not operation.path.is_file():
        raise OperationJournalError("reconciliation journal is unavailable")
    return hashlib.sha256(operation.path.read_bytes()).hexdigest()


def _sha256(data: bytes) -> str:
    """Return a stable hash for retained, non-secret recovery evidence."""

    return hashlib.sha256(data).hexdigest()


def _require_regular_file(path: Path, label: str, *, private: bool = False) -> bytes:
    """Read one regular retained file without following links."""

    try:
        status = path.lstat()
        valid_mode = not private or not status.st_mode & 0o077
        _require_reconciliation_value(
            path.is_file() and not path.is_symlink() and valid_mode,
            f"{label} is unavailable or insufficiently private",
        )
        return path.read_bytes()
    except OSError as exc:
        raise OperationJournalError(f"{label} is unavailable") from exc


def _json_object(data: bytes, label: str) -> dict[str, Any]:
    """Parse a retained JSON object or reject it before reconciliation."""

    try:
        value = json.loads(data)
    except json.JSONDecodeError as exc:
        raise OperationJournalError(f"{label} is malformed") from exc
    _require_reconciliation_value(isinstance(value, dict), f"{label} is not an object")
    return value


def _owner_is_alive(owner_pid: object) -> bool:
    """Return whether a journal owner can still mutate its operation."""

    try:
        pid = int(owner_pid)
    except (TypeError, ValueError):
        return True
    if pid <= 0:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _require_reconciliation_value(condition: bool, message: str) -> None:
    """Fail closed when a failed provision cannot be safely reconciled."""

    if not condition:
        raise OperationJournalError(message)


def _require_empty_failed_provision(
    payload: Mapping[str, Any],
    *,
    alias: str,
    environment: EnvironmentConfig,
    cluster_name: str,
) -> None:
    """Require the exact failed, unowned provisioning shape before provider reads."""

    expected = _expected_reconciliation_scope(alias, environment, cluster_name)
    for key, value in expected.items():
        _require_reconciliation_value(
            str(payload.get(key) or "") == str(value or ""),
            f"reconciliation journal has mismatched {key}",
        )
    _require_reconciliation_value(
        not _owner_is_alive(payload.get("owner_pid")),
        "reconciliation journal still has a live owner",
    )
    _require_no_owned_reconciliation_state(payload)
    _require_original_preflight_scope(payload, expected, cluster_name)
    _require_original_failure_provenance(payload)


def _expected_reconciliation_scope(
    alias: str, environment: EnvironmentConfig, cluster_name: str
) -> dict[str, str]:
    """Return the immutable journal fields that select one failed provision."""

    return {
        "command": "npa provision-if-absent",
        "project_alias": alias,
        "project_id": environment.project_id,
        "tenant_id": environment.tenant_id,
        "region": environment.region,
        "resource_type": "cluster",
        "requested_name": cluster_name,
        "phase": "recovery-required",
        "lifecycle": "failed",
    }


def _require_no_owned_reconciliation_state(payload: Mapping[str, Any]) -> None:
    """Reject a recovery when the original operation recorded any mutation."""

    _require_reconciliation_value(
        not any(
            payload.get(key)
            for key in ("resources", "local_state_copies", "config_mutations")
        ),
        "reconciliation journal records owned or mutated state",
    )
    rollback = payload.get("rollback")
    _require_reconciliation_value(
        isinstance(rollback, Mapping)
        and rollback.get("attempted") is False
        and rollback.get("completed") is False,
        "reconciliation journal has an incomplete rollback record",
    )


def _require_original_preflight_scope(
    payload: Mapping[str, Any], expected: Mapping[str, str], cluster_name: str
) -> None:
    """Bind recovery to the original plan instead of a caller-selected topology."""

    plan = payload.get("preflight_plan")
    topology = plan.get("topology") if isinstance(plan, Mapping) else None
    scope_keys = {"project_alias", "project_id", "tenant_id", "region"}
    scope_matches = isinstance(plan, Mapping) and all(
        str(plan.get(key) or "") == str(value or "")
        for key, value in expected.items()
        if key in scope_keys
    )
    _require_reconciliation_value(
        isinstance(topology, Mapping)
        and str(topology.get("cluster_name") or "") == cluster_name
        and scope_matches,
        "reconciliation journal lacks the original exact preflight scope",
    )


def _require_original_failure_provenance(payload: Mapping[str, Any]) -> None:
    """Require the source failure evidence retained by the original producer."""

    error = str(payload.get("last_error") or "").lower()
    _require_reconciliation_value(
        payload.get("last_error_type") == "RuntimeError"
        and "terraform apply" in error
        and "timeout" not in error
        and "interrupt" not in error
        and isinstance(payload.get("events"), list)
        and bool(payload.get("events")),
        "reconciliation journal does not record a completed apply rejection",
    )


def _original_topology(payload: Mapping[str, Any]) -> dict[str, object]:
    """Return the one single-GPU topology supported by empty recovery."""

    plan = payload.get("preflight_plan")
    topology = plan.get("topology") if isinstance(plan, Mapping) else None
    _require_reconciliation_value(
        isinstance(topology, Mapping), "reconciliation journal lacks topology"
    )
    keys = ("gpu_nodes", "cpu_nodes", "gpu_platform", "gpu_preset", "accelerator")
    values = {key: topology.get(key) for key in keys}
    _require_reconciliation_value(
        isinstance(values["gpu_nodes"], int)
        and values["gpu_nodes"] == 1
        and isinstance(values["cpu_nodes"], int)
        and values["cpu_nodes"] == 1
        and all(str(values[key] or "").strip() for key in keys[2:])
        and topology.get("gpu_preemptible") is False
        and not str(topology.get("capacity_block_group") or "").strip(),
        "reconciliation journal is outside the supported single-GPU profile",
    )
    return values


def _recorded_provision_options(record: Mapping[str, Any]) -> dict[str, object]:
    """Parse one saved Bash command without evaluating any of its shell text."""

    command = record.get("command")
    exit_code = record.get("exit_code")
    _require_reconciliation_value(
        isinstance(command, str)
        and record.get("status") == "failed"
        and isinstance(exit_code, int)
        and exit_code != 0,
        "retained command event is not a failed provision",
    )
    outer = shlex.split(command)
    _require_reconciliation_value(
        len(outer) == 3 and Path(outer[0]).name == "bash" and outer[1] == "-lc",
        "retained command event is not a Bash command",
    )
    lines = [line for line in outer[2].splitlines() if "provision-if-absent" in line]
    _require_reconciliation_value(len(lines) == 1, "retained command is ambiguous")
    tokens = shlex.split(lines[0])
    index = tokens.index("provision-if-absent")
    return _parse_recorded_options(tokens[index + 1 :])


def _parse_recorded_options(tokens: list[str]) -> dict[str, object]:
    """Read the bounded option vocabulary of the standard failed producer."""

    parsed: dict[str, object] = {}
    position = 0
    while position < len(tokens):
        token = tokens[position]
        if token in {">", ">>", "2>", "2>>"}:
            break
        _require_reconciliation_value(
            token.startswith("--"), "retained command has a positional argument"
        )
        _require_reconciliation_value(
            token not in parsed, "retained command repeats an option"
        )
        if token in _RECOVERY_BOOLEAN_FLAGS:
            parsed[token] = True
            position += 1
            continue
        _require_reconciliation_value(
            token in _RECOVERY_VALUE_FLAGS and position + 1 < len(tokens),
            "retained command has an unsupported option",
        )
        value = tokens[position + 1]
        _require_reconciliation_value(
            not value.startswith("--"), "retained command lacks an option value"
        )
        parsed[token] = value
        position += 2
    return parsed


def _matching_retained_event(
    evidence_path: Path,
    *,
    alias: str,
    cluster_name: str,
    topology: Mapping[str, object],
) -> dict[str, str]:
    """Bind recovery to exactly one collected canonical failed producer event."""

    raw = _require_regular_file(evidence_path, "recovery evidence", private=True)
    evidence = _json_object(raw, "recovery evidence")
    records = evidence.get("records")
    _require_reconciliation_value(
        set(evidence) == {"scope", "records"}
        and isinstance(evidence.get("scope"), str)
        and isinstance(records, list)
        and all(isinstance(record, Mapping) for record in records),
        "recovery evidence has an unsupported schema",
    )
    matches = [
        record
        for record in records
        if _retained_event_matches(record, alias, cluster_name, topology)
    ]
    _require_reconciliation_value(
        len(matches) == 1, "recovery evidence does not select one supported producer"
    )
    return _retained_failure_log_hash(matches[0], evidence_path)


def _retained_event_matches(
    record: Mapping[str, Any],
    alias: str,
    cluster_name: str,
    topology: Mapping[str, object],
) -> bool:
    """Return whether a saved event is the exact standard no-S3 producer shape."""

    try:
        options = _recorded_provision_options(record)
        from npa.cli.cluster.terraform_lifecycle import _resolve_terraform_dir

        expected = {
            "--project": alias,
            "--cluster-name": cluster_name,
            "--context": cluster_name,
            "--gpu-nodes": str(topology["gpu_nodes"]),
            "--gpu-platform": str(topology["gpu_platform"]),
            "--gpu-preset": str(topology["gpu_preset"]),
            "--accelerator": str(topology["accelerator"]),
            "--output-format": "json",
        }
        allowed = (
            set(expected)
            | {"--kubeconfig", "--terraform-dir"}
            | set(_RECOVERY_BOOLEAN_FLAGS)
        )
        return (
            set(options) == allowed
            and all(options.get(key) == value for key, value in expected.items())
            and options.get("--skip-s3") is True
            and options.get("--on-demand") is True
            and bool(str(options.get("--kubeconfig") or "").strip())
            and Path(str(options.get("--terraform-dir") or "")).resolve()
            == _resolve_terraform_dir(None)
        )
    except (OperationJournalError, OSError, ValueError):
        return False


def _retained_failure_log_hash(
    record: Mapping[str, Any], evidence_path: Path
) -> dict[str, str]:
    """Bind the selected event to its task-local completed provider rejection."""

    command = str(record["command"])
    script = shlex.split(command)[2]
    service_dir = _recorded_service_dir(script)
    line = next(line for line in script.splitlines() if "provision-if-absent" in line)
    tokens = shlex.split(line)
    _require_reconciliation_value(
        "2>" in tokens, "retained command has no provider error log"
    )
    target = tokens[tokens.index("2>") + 1]
    log_path = _recorded_log_path(service_dir, target, evidence_path.parent)
    log = _require_regular_file(log_path, "retained provider failure log", private=True)
    _require_completed_allocation_rejection(log)
    return {
        "command_event_sha256": _sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
        "provider_failure_log_sha256": _sha256(log),
    }


def _require_completed_allocation_rejection(log: bytes) -> None:
    """Reject timeout-like failures that might leave a create running provider-side."""

    lines = log.decode("utf-8", errors="replace").splitlines()
    preconditions = [
        index
        for index, line in enumerate(lines)
        if "failedprecondition" in line.lower()
    ]
    allocations = [
        index for index, line in enumerate(lines) if "allocation" in line.lower()
    ]
    related = [lines[index].lower() for index in [*preconditions, *allocations]]
    _require_reconciliation_value(
        len(preconditions) == 1
        and len(allocations) == 1
        and abs(preconditions[0] - allocations[0]) <= 2
        and not any(
            token in line
            for line in related
            for token in ("timeout", "interrupted", "deadline", "cancelled")
        ),
        "retained provider failure is not a completed allocation rejection",
    )


def _recorded_service_dir(script: str) -> Path:
    """Read the literal task directory assignment from the collected script."""

    lines = [
        line.strip()
        for line in script.splitlines()
        if line.strip().startswith("service_dir=")
    ]
    _require_reconciliation_value(
        len(lines) == 1, "retained command has no unique service directory"
    )
    value = shlex.split(lines[0].split("=", 1)[1])
    _require_reconciliation_value(
        len(value) == 1 and Path(value[0]).is_absolute(),
        "retained service directory is invalid",
    )
    return Path(value[0]).resolve()


def _recorded_log_path(service_dir: Path, target: str, evidence_root: Path) -> Path:
    """Resolve only the saved service-dir stderr target, never a shell expansion."""

    prefix = "$service_dir/"
    _require_reconciliation_value(
        target.startswith(prefix), "retained provider log is not task-local"
    )
    path = (service_dir / target.removeprefix(prefix)).resolve()
    _require_reconciliation_value(
        path.parent == service_dir and path.is_relative_to(evidence_root.resolve()),
        "retained provider log escapes task evidence",
    )
    return path


def _backend_recipe_dir(
    environment: EnvironmentConfig, cluster_name: str, context_name: str
) -> Path:
    """Return the task-local retained standard backend material directory."""

    return (
        cluster_dir(context_name)
        / "backend-state"
        / environment.project_id
        / cluster_name
        / "k8s-training"
    )


def _valid_hash(value: object) -> bool:
    """Return whether a retained source identity is a full SHA-256 digest."""

    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _validate_backend_sidecar(
    sidecar: Mapping[str, Any], environment: EnvironmentConfig, cluster_name: str
) -> RetainedBackendIdentity:
    """Validate retained standard-recipe provenance without disclosing its values."""

    _validate_retained_backend_scope(sidecar, environment, cluster_name)
    return _provider_deadline_identity(sidecar)


def _validate_retained_backend_scope(
    sidecar: Mapping[str, Any], environment: EnvironmentConfig, cluster_name: str
) -> None:
    """Require the retained backend's exact scope and reused-subnet input."""

    required = {
        "backend": "mk8s",
        "cluster_name": cluster_name,
        "project_id": environment.project_id,
        "tenant_id": environment.tenant_id,
        "region": environment.region,
        "gpu_driver_mode": "managed-image",
        "status": "provisioning",
    }
    _require_reconciliation_value(
        all(sidecar.get(key) == value for key, value in required.items())
        and bool(str(sidecar.get("context") or "").strip())
        and bool(str(sidecar.get("subnet_id") or "").strip()),
        "retained backend scope is not the supported producer",
    )


def _provider_deadline_identity(sidecar: Mapping[str, Any]) -> RetainedBackendIdentity:
    """Extract the validated source and materialized-provider identity."""

    deadlines = sidecar.get("provider_rpc_deadlines")
    _require_reconciliation_value(
        isinstance(deadlines, Mapping), "retained provider deadlines are missing"
    )
    sources = deadlines.get("source_sha256")
    defaults = deadlines.get("inserted_defaults")
    timeout = deadlines.get("apply_timeout_minutes")
    materialized = deadlines.get("materialized_provider_sha256")
    _require_provider_source_identity(sources, materialized)
    _require_provider_deadline_defaults(timeout, defaults, deadlines)
    return RetainedBackendIdentity(
        hashes={
            "provider_source_set_sha256": _sha256(
                json.dumps(sources, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ),
            "materialized_provider_sha256": str(materialized),
        },
        source_hashes={str(key): str(value) for key, value in sources.items()},
        provider_defaults={str(key): str(value) for key, value in defaults.items()},
    )


def _require_provider_source_identity(sources: object, materialized: object) -> None:
    """Require full hashes for every original source file and materialized provider."""

    _require_reconciliation_value(
        isinstance(sources, Mapping)
        and bool(sources)
        and all(
            isinstance(key, str) and _valid_hash(value)
            for key, value in sources.items()
        )
        and _valid_hash(materialized),
        "retained provider source identity is malformed",
    )


def _require_provider_deadline_defaults(
    timeout: object, defaults: object, deadlines: Mapping[str, Any]
) -> None:
    """Require the standard complete set of injected provider RPC defaults."""

    _require_reconciliation_value(
        isinstance(timeout, int)
        and timeout > 0
        and isinstance(defaults, Mapping)
        and set(defaults) == {"auth_timeout", "per_retry_timeout", "timeout"}
        and all(value == f"{timeout}m" for value in defaults.values())
        and deadlines.get("preserved_operator_fields") == [],
        "retained provider defaults are outside the supported recipe",
    )


def _supported_recipe_root() -> Path:
    """Return the checked-in source graph for the one accepted producer shape."""

    root = (
        Path(__file__).resolve().parents[3]
        / "deploy"
        / "cluster"
        / "vendor"
        / "nebius-solutions-library"
    )
    _require_reconciliation_value(
        root.is_dir() and not root.is_symlink(),
        "supported Terraform recipe is unavailable",
    )
    return root


def _expected_recipe_sources(source_root: Path, region: str) -> dict[str, bytes]:
    """Read reviewed top-level Terraform bytes after their deterministic patch."""

    from npa.cluster_backends.mk8s_render import (
        patch_explicit_region_defaults,
        patch_provider_domain,
    )

    sources: dict[str, bytes] = {}
    for path in sorted((source_root / "k8s-training").glob("*.tf")):
        raw = _require_regular_file(path, "supported Terraform source")
        text = raw.decode("utf-8")
        if path.name == "locals.tf":
            raw = patch_explicit_region_defaults(text).encode("utf-8")
        if path.name == "provider.tf":
            raw = patch_provider_domain(text, region).encode("utf-8")
        sources[path.name] = raw
    _require_reconciliation_value(bool(sources), "supported Terraform source is empty")
    return sources


def _materialized_provider_source(source: bytes, defaults: Mapping[str, str]) -> bytes:
    """Apply the existing provider-default materializer to reviewed source bytes."""

    from npa.cluster_backends.provider_rpc import _hcl_providers, _materialized_text

    text = source.decode("utf-8")
    providers = _hcl_providers(Path("provider.tf"), text)
    _require_reconciliation_value(
        len(providers) == 1, "supported provider source is not materializable"
    )
    return _materialized_text(providers[0], text, dict(defaults), {}).encode("utf-8")


def _tree_sha256(root: Path, label: str) -> str:
    """Hash a regular-file recipe tree with names so extra modules cannot hide."""

    _require_reconciliation_value(
        root.is_dir() and not root.is_symlink(), f"{label} is unavailable"
    )
    entries: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        _require_reconciliation_value(
            not path.is_symlink(), f"{label} contains a symlink"
        )
        if path.is_file():
            entries.append((str(path.relative_to(root)), _sha256(path.read_bytes())))
        else:
            _require_reconciliation_value(
                path.is_dir(), f"{label} contains a special file"
            )
    _require_reconciliation_value(bool(entries), f"{label} is empty")
    return _sha256(json.dumps(entries, separators=(",", ":")).encode("utf-8"))


def _validate_recipe_graph(
    recipe: Path, identity: RetainedBackendIdentity, region: str
) -> dict[str, str]:
    """Bind the retained materialized graph to the checked-in standard recipe."""

    source_root = _supported_recipe_root()
    expected = _expected_recipe_sources(source_root, region)
    expected_hashes = {name: _sha256(raw) for name, raw in expected.items()}
    _require_reconciliation_value(
        identity.source_hashes == expected_hashes,
        "retained producer source does not match the supported recipe",
    )
    _require_materialized_recipe_match(recipe, expected, identity)
    source_modules = _tree_sha256(source_root / "modules", "supported module graph")
    retained_modules = _tree_sha256(recipe.parent / "modules", "retained module graph")
    _require_reconciliation_value(
        source_modules == retained_modules,
        "retained modules differ from the supported graph",
    )
    return _recipe_graph_hashes(expected_hashes, source_modules)


def _require_materialized_recipe_match(
    recipe: Path, expected: Mapping[str, bytes], identity: RetainedBackendIdentity
) -> None:
    """Require the retained files to equal the reviewed source materialization."""

    retained = {
        path.name: _require_regular_file(path, "retained Terraform source")
        for path in recipe.glob("*.tf")
    }
    _require_reconciliation_value(
        set(retained) == set(expected), "retained Terraform source graph is incomplete"
    )
    materialized = dict(expected)
    materialized["provider.tf"] = _materialized_provider_source(
        expected["provider.tf"], identity.provider_defaults
    )
    _require_reconciliation_value(
        retained == materialized
        and _sha256(retained["provider.tf"])
        == identity.hashes["materialized_provider_sha256"],
        "retained Terraform bytes do not match the supported materialization",
    )


def _recipe_graph_hashes(
    expected_hashes: Mapping[str, str], module_graph: str
) -> dict[str, str]:
    """Return redacted recipe graph identities for the reconciliation receipt."""

    return {
        "supported_recipe_graph_sha256": _sha256(
            json.dumps(expected_hashes, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ),
        "module_graph_sha256": module_graph,
    }


def _validate_standard_tfvars(
    values: Mapping[str, Any], topology: Mapping[str, object], cluster_name: str
) -> None:
    """Reject every Terraform input shape outside the retained standard producer."""

    _require_standard_tfvars_core(values, topology, cluster_name)
    _require_standard_tfvars_options(values, topology)
    _require_standard_tfvars_observability(values)


def _require_standard_tfvars_core(
    values: Mapping[str, Any], topology: Mapping[str, object], cluster_name: str
) -> None:
    """Require the topology and disabled optional features of the retained profile."""

    disabled = {
        "enable_k8s_node_group_sa",
        "enable_egress_gateway",
        "enable_gpu_cluster",
        "enable_filestore",
        "enable_nebius_o11y_agent",
        "enable_grafana",
        "enable_prometheus",
        "collectK8sClusterMetrics",
        "enable_kuberay_cluster",
        "enable_kuberay_service",
        "enable_opa_gatekeeper",
    }
    _require_reconciliation_value(
        set(values) == _STANDARD_TFVARS
        and all(values.get(key) is False for key in disabled)
        and values.get("cluster_name") == cluster_name
        and values.get("gpu_nodes_fixed_count_per_group") == topology["gpu_nodes"]
        and values.get("gpu_nodes_platform") == topology["gpu_platform"]
        and values.get("gpu_nodes_preset") == topology["gpu_preset"],
        "retained Terraform inputs are outside the supported producer",
    )


def _require_standard_tfvars_options(
    values: Mapping[str, Any], topology: Mapping[str, object]
) -> None:
    """Reject retained optional resources beyond the standard one-GPU shape."""

    _require_reconciliation_value(
        values.get("cpu_nodes_fixed_count") == topology["cpu_nodes"]
        and values.get("gpu_node_groups") == 1
        and values.get("cpu_nodes_preemptible") is False
        and values.get("gpu_nodes_preemptible") is False
        and values.get("gpu_nodes_driverfull_image") is True
        and values.get("custom_driver") is False
        and values.get("existing_filestore") == ""
        and values.get("mig_strategy") == "none",
        "retained Terraform inputs have unsupported optional resources",
    )


def _require_standard_tfvars_observability(values: Mapping[str, Any]) -> None:
    """Require the retained profile's non-resource observability settings."""

    _require_reconciliation_value(
        bool(str(values.get("ssh_public_key") or "").strip())
        and bool(str(values.get("filesystem_csi") or "").strip())
        and bool(
            re.fullmatch(
                r"\{\s*enabled\s*=\s*false\s*,\s*replication_factor\s*=\s*1\s*\}",
                str(values.get("loki") or ""),
            )
        ),
        "retained Terraform inputs have an unsupported observability shape",
    )


def _validate_empty_terraform_state(state: Mapping[str, Any]) -> None:
    """Use retained state as a contradiction detector, never as absence proof."""

    _require_reconciliation_value(
        set(state)
        == {
            "version",
            "terraform_version",
            "serial",
            "lineage",
            "outputs",
            "resources",
            "check_results",
        }
        and state.get("outputs") == {}
        and isinstance(state.get("resources"), list),
        "retained Terraform state is unsupported",
    )
    for resource in state["resources"]:
        _require_reconciliation_value(
            isinstance(resource, Mapping), "retained state has invalid resource"
        )
        kind = (str(resource.get("mode") or ""), str(resource.get("type") or ""))
        instances = resource.get("instances")
        _require_reconciliation_value(
            kind in _STATE_RESOURCE_TYPES and isinstance(instances, list),
            "retained state has an unsupported resource type",
        )
        _require_reconciliation_value(
            not instances
            or kind in {("managed", "random_string"), ("managed", "time_static")},
            "retained Terraform state records a possible provider resource",
        )


def _require_reused_subnet(root: Path) -> None:
    """Accept only the retained shape that reused, rather than created, VPC state."""

    network_state = root / ".npa-fleet-network.json"
    _require_reconciliation_value(
        not network_state.exists() and not network_state.is_symlink(),
        "retained producer may have created network or subnet state",
    )


def _retained_recovery_profile(
    *,
    evidence_path: Path,
    payload: Mapping[str, Any],
    alias: str,
    environment: EnvironmentConfig,
    cluster_name: str,
    context_name: str,
) -> EmptyFailureRecoveryProfile:
    """Load the exact retained inputs that prove the producer's supported shape."""

    topology = _original_topology(payload)
    event_hashes = _matching_retained_event(
        evidence_path, alias=alias, cluster_name=cluster_name, topology=topology
    )
    recipe = _backend_recipe_dir(environment, cluster_name, context_name)
    root = recipe.parent
    sidecar_raw = _require_regular_file(root / ".npa-fleet-env.json", "backend sidecar")
    tfvars_raw = _require_regular_file(recipe / "terraform.tfvars", "Terraform inputs")
    state_raw = _require_regular_file(recipe / "terraform.tfstate", "Terraform state")
    lock_raw = _require_regular_file(recipe / ".terraform.lock.hcl", "Terraform lock")
    readme_raw = _require_regular_file(recipe / "README.md", "Terraform recipe")
    sidecar = _json_object(sidecar_raw, "backend sidecar")
    identity = _validate_backend_sidecar(sidecar, environment, cluster_name)
    _require_reused_subnet(root.parent)
    return _complete_retained_profile(
        recipe,
        tfvars_raw,
        state_raw,
        event_hashes,
        identity,
        cluster_name,
        topology,
        lock_raw,
        readme_raw,
        context_name,
        environment.region,
    )


def _complete_retained_profile(
    recipe: Path,
    tfvars_raw: bytes,
    state_raw: bytes,
    event_hashes: Mapping[str, str],
    identity: RetainedBackendIdentity,
    cluster_name: str,
    topology: dict[str, object],
    lock_raw: bytes,
    readme_raw: bytes,
    context_name: str,
    region: str,
) -> EmptyFailureRecoveryProfile:
    """Validate retained Terraform data and construct its redacted recovery profile."""

    from npa.cli.cluster.terraform_lifecycle import _read_tfvars

    try:
        tfvars = _read_tfvars(recipe)
    except (OSError, ValueError) as exc:
        raise OperationJournalError("retained Terraform inputs are malformed") from exc
    _validate_standard_tfvars(tfvars, topology, cluster_name)
    _validate_empty_terraform_state(_json_object(state_raw, "Terraform state"))
    hashes = {
        **event_hashes,
        **identity.hashes,
        **_validate_recipe_graph(recipe, identity, region),
        "terraform_tfvars_sha256": _sha256(tfvars_raw),
        "terraform_state_sha256": _sha256(state_raw),
        "terraform_lock_sha256": _sha256(lock_raw),
        "terraform_recipe_readme_sha256": _sha256(readme_raw),
    }
    return EmptyFailureRecoveryProfile(
        retained_hashes=hashes,
        topology=topology,
        expected_names={
            "cluster": cluster_name,
        },
    )


def _verified_empty_provision_footprint(
    environment: EnvironmentConfig, profile: EmptyFailureRecoveryProfile
) -> dict[str, object]:
    """Prove the original cluster footprint is absent under the exact project scope."""

    _verify_reconciliation_project(environment)
    return _verify_absent_footprint_inventory(environment, profile.expected_names)


def _verify_reconciliation_project(environment: EnvironmentConfig) -> None:
    """Confirm the provider project still exactly matches the original scope."""

    from npa.clients import nebius

    identity = nebius.get_project_identity(
        environment.project_id, tenant_id=environment.tenant_id
    )
    _require_reconciliation_value(identity is not None, "provider project is absent")
    _require_reconciliation_value(
        identity.region == environment.region,
        "provider project region does not match the failed provision",
    )


def _verify_absent_footprint_inventory(
    environment: EnvironmentConfig, expected_names: Mapping[str, str]
) -> dict[str, object]:
    """Reject every surviving resource in the original provision's named footprint."""

    from npa.cluster.api import MK8sClient

    clusters = MK8sClient().list_clusters_complete(environment.project_id)
    if any(cluster.name == expected_names["cluster"] for cluster in clusters.clusters):
        raise OperationJournalError(
            "original managed Kubernetes cluster is still present"
        )
    evidence: dict[str, object] = {
        "cluster": {
            "count": len(clusters.clusters),
            "response_sha256": clusters.response_sha256,
            "matched": False,
        },
        "node_group": {
            "matched": False,
            "absence_basis": "managed node groups require the absent cluster parent",
        },
    }
    return evidence


def _failed_provision_operation(
    *, alias: str, environment: EnvironmentConfig, cluster_name: str
) -> ProvisioningOperation:
    """Load the deterministic original generation without creating a replacement."""

    operation = ProvisioningOperation(
        deterministic_operation_id(
            command="npa provision-if-absent",
            project_id=environment.project_id,
            project_alias=alias,
            resource_type="cluster",
            requested_name=cluster_name,
        )
    )
    if not operation.read():
        raise OperationJournalError("no exact failed provisioning journal was found")
    return operation


def reconcile_empty_failed_provision(
    *,
    project: str | None,
    cluster_name: str,
    context_name: str,
    recovery_evidence: Path,
) -> ProvisionIfAbsentResult:
    """Terminalize one failed empty provision after fresh exact provider absence proof.

    Args:
        project: Configured project alias selecting the original provider scope.
        cluster_name: Original managed Kubernetes cluster name.
        context_name: Original Kubernetes context name.
    Returns:
        A reconciliation-only result; no resources are created or destroyed.
    Raises:
        OperationJournalError: Ownership, journal, or provider absence cannot be proven.
    """

    alias, environment, storage, registry = _resolve_project_runtime(project)
    operation = _failed_provision_operation(
        alias=alias, environment=environment, cluster_name=cluster_name
    )
    with operation_context(operation):
        _reconcile_locked_empty_failed_provision(
            operation,
            alias,
            environment,
            cluster_name,
            context_name,
            recovery_evidence,
        )
    return _reconciled_provision_result(
        alias, storage, registry, cluster_name, context_name, operation
    )


def _reconcile_locked_empty_failed_provision(
    operation: ProvisioningOperation,
    alias: str,
    environment: EnvironmentConfig,
    cluster_name: str,
    context_name: str,
    recovery_evidence: Path,
) -> None:
    """Verify absence while locks are held and then terminalize one journal."""

    original = operation.read()
    _require_empty_failed_provision(
        original, alias=alias, environment=environment, cluster_name=cluster_name
    )
    original_sha256 = _journal_sha256(operation)
    profile = _retained_recovery_profile(
        evidence_path=recovery_evidence,
        payload=original,
        alias=alias,
        environment=environment,
        cluster_name=cluster_name,
        context_name=context_name,
    )
    footprint = _verified_empty_provision_footprint(environment, profile)
    _require_reconciliation_value(
        _journal_sha256(operation) == original_sha256,
        "reconciliation journal changed during provider verification",
    )
    _terminalize_empty_failed_provision(
        operation=operation,
        environment=environment,
        cluster_name=cluster_name,
        context_name=context_name,
        original_sha256=original_sha256,
        footprint=footprint,
        profile=profile,
    )


def _reconciled_provision_result(
    alias: str,
    storage: StorageConfig,
    registry: str,
    cluster_name: str,
    context_name: str,
    operation: ProvisioningOperation,
) -> ProvisionIfAbsentResult:
    """Build the non-mutating result for a successfully reconciled journal."""

    return ProvisionIfAbsentResult(
        status="reconciled",
        project=alias,
        cluster_name=cluster_name,
        context_name=context_name,
        storage_bucket=storage.checkpoint_bucket,
        storage_endpoint=storage.endpoint_url,
        registry=registry,
        operation_id=operation.operation_id,
        operation_journal=str(operation.path),
        actions=[
            "recovery:terminalized failed empty provision after provider absence proof"
        ],
    )


def _terminalize_empty_failed_provision(
    *,
    operation: ProvisioningOperation,
    environment: EnvironmentConfig,
    cluster_name: str,
    context_name: str,
    original_sha256: str,
    footprint: Mapping[str, object],
    profile: EmptyFailureRecoveryProfile,
) -> None:
    """Record fresh absence evidence before terminalizing the original journal."""

    operation.checkpoint(
        "empty-failure-provider-absence",
        {
            "original_journal_sha256": original_sha256,
            "scope": {
                "project_id": environment.project_id,
                "tenant_id": environment.tenant_id,
                "region": environment.region,
                "requested_name": cluster_name,
                "context_name": context_name,
            },
            "retained_producer": {
                "evidence_kind": "collected-command-events-and-backend-material",
                "supported_profile": "standard-no-s3-mk8s-single-gpu",
                "hashes": profile.retained_hashes,
                "topology": profile.topology,
            },
            "provider_inventory": dict(footprint),
        },
    )
    operation.record_rollback(
        attempted=False,
        completed=True,
        removed=[],
        preserved=[],
        error="fresh provider inventory proved the original provision footprint absent",
    )
    operation.transition("rolled-back", details={"reconciliation": "empty-failure"})


def _provision_recovery_argv(
    arguments: dict[str, Any],
    *,
    alias: str,
    cluster_name: str,
    context: str,
    kubeconfig: Path,
) -> list[str]:
    """Serialize every effective, non-secret topology option for exact resume."""

    argv = [
        "npa",
        "provision-if-absent",
        "--cluster-name",
        cluster_name,
        "--context",
        context,
        "--kubeconfig",
        str(kubeconfig),
        "--timeout",
        str(int(arguments.get("timeout") or 120)),
        "--gpu-nodes",
        str(int(arguments.get("gpu_nodes", -1))),
        "--cpu-nodes",
        str(int(arguments.get("cpu_nodes", -1))),
        "--gpu-readiness-timeout",
        str(float(arguments.get("gpu_readiness_timeout") or 600.0)),
        "--gpu-readiness-poll-interval",
        str(float(arguments.get("gpu_readiness_poll_interval") or 10.0)),
        "--gpu-health-stabilization-seconds",
        str(
            int(
                arguments.get("gpu_health_stabilization_seconds")
                if arguments.get("gpu_health_stabilization_seconds") is not None
                else DEFAULT_STABILIZATION_SECONDS
            )
        ),
        "--output-format",
        str(arguments.get("output_format") or "text"),
    ]
    if alias:
        argv.extend(["--project", alias])
    terraform_dir = arguments.get("terraform_dir")
    if terraform_dir:
        argv.extend(["--terraform-dir", str(terraform_dir)])
    for key, flag in (
        ("cpu_platform", "--cpu-platform"),
        ("cpu_preset", "--cpu-preset"),
        ("gpu_platform", "--gpu-platform"),
        ("gpu_preset", "--gpu-preset"),
        ("gpu_driver_mode", "--gpu-driver-mode"),
        ("gpu_workload_profile", "--gpu-workload-profile"),
        ("managed_driver_preset", "--managed-driver-preset"),
        ("gpu_cuda_smoke_image", "--gpu-cuda-smoke-image"),
        ("gpu_graphics_smoke_image", "--gpu-graphics-smoke-image"),
        ("capacity_block_group", "--capacity-block-group"),
        ("accelerator", "--accelerator"),
        ("sky_bin", "--sky-bin"),
    ):
        value = str(arguments.get(key) or "")
        if value:
            argv.extend([flag, value])
    for key, enabled, disabled in (
        ("validate", "--validate", "--skip-validate"),
        ("sky_smoke", "--sky-smoke", "--skip-sky-smoke"),
        ("gpu_cuda_smoke", "--gpu-cuda-smoke", "--skip-gpu-cuda-smoke"),
    ):
        argv.append(enabled if bool(arguments.get(key)) else disabled)
    if bool(arguments.get("skip_k8s")):
        argv.append("--skip-k8s")
    if bool(arguments.get("skip_s3")):
        argv.append("--skip-s3")
    if bool(arguments.get("dry_run")):
        argv.append("--dry-run")
    preemptible = arguments.get("preemptible")
    if preemptible is not None:
        argv.append("--preemptible" if bool(preemptible) else "--on-demand")
    unsafe_operator = arguments.get("allow_unsafe_nvswitch_operator")
    if unsafe_operator is not None:
        argv.append(
            "--allow-unsafe-nvswitch-operator"
            if bool(unsafe_operator)
            else "--deny-unsafe-nvswitch-operator"
        )
    return argv


def _transactional_provision(function):
    signature = inspect.signature(function)

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        from npa.lifecycle_intent import forbid_destructive_provisioning

        forbid_destructive_provisioning("provision_if_absent")
        if current_operation() is not None:
            return function(*args, **kwargs)
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        requested_project = bound.arguments.get("project")
        alias, environment, storage, _registry = _resolve_project_runtime(
            requested_project
        )
        cluster_name = str(bound.arguments.get("cluster_name") or "npa-cluster")
        skip_k8s = bool(bound.arguments.get("skip_k8s"))
        context = str(bound.arguments.get("context_name") or "").strip() or cluster_name
        if bool(bound.arguments.get("reconcile_empty_failure")):
            if bool(bound.arguments.get("dry_run")):
                raise ValueError(
                    "--reconcile-empty-failure requires a live provider absence check"
                )
            recovery_evidence = bound.arguments.get("recovery_evidence")
            if skip_k8s or bool(bound.arguments.get("skip_s3")):
                raise ValueError(
                    "--reconcile-empty-failure cannot be combined with skip options"
                )
            if not isinstance(recovery_evidence, Path):
                raise ValueError(
                    "--reconcile-empty-failure requires --recovery-evidence"
                )
            if bound.arguments.get("terraform_dir") or bound.arguments.get(
                "kubeconfig"
            ):
                raise ValueError(
                    "--reconcile-empty-failure uses retained producer inputs only"
                )
            return reconcile_empty_failed_provision(
                project=requested_project,
                cluster_name=cluster_name,
                context_name=context,
                recovery_evidence=recovery_evidence,
            )
        kubeconfig = bound.arguments.get("kubeconfig") or kubeconfig_file(context)
        plan = bound.arguments.get("_resolved_plan") or _build_provision_plan(
            alias=alias,
            environment=environment,
            cluster_name=cluster_name,
            kubeconfig=Path(kubeconfig),
            context=context,
            skip_k8s=skip_k8s,
            dry_run=bool(bound.arguments.get("dry_run")),
            accelerator=str(bound.arguments.get("accelerator") or ""),
            gpu_nodes=int(bound.arguments.get("gpu_nodes", -1)),
            cpu_nodes=int(bound.arguments.get("cpu_nodes", -1)),
            cpu_platform=str(bound.arguments.get("cpu_platform") or ""),
            cpu_preset=str(bound.arguments.get("cpu_preset") or ""),
            gpu_platform=str(bound.arguments.get("gpu_platform") or ""),
            gpu_preset=str(bound.arguments.get("gpu_preset") or ""),
            capacity_block_group=str(bound.arguments.get("capacity_block_group") or ""),
            preemptible=bound.arguments.get("preemptible"),
        )
        kwargs["_resolved_plan"] = plan
        if bool(bound.arguments.get("dry_run")):
            return function(*args, **kwargs)
        plan.assert_mutation_ready()
        resource_type = "storage" if skip_k8s else "cluster"
        requested_name = (
            _bucket_name(storage.checkpoint_bucket) if skip_k8s else cluster_name
        )
        recovery_arguments = dict(bound.arguments)
        recovery_arguments["capacity_block_group"] = plan.topology.capacity_block_group
        recovery_arguments["preemptible"] = plan.topology.gpu_preemptible
        resume_argv = _provision_recovery_argv(
            recovery_arguments,
            alias=alias,
            cluster_name=cluster_name,
            context=context,
            kubeconfig=Path(kubeconfig),
        )
        destroy_argv = (
            [
                "npa",
                "cluster",
                "down",
                "--project",
                alias,
                "--context",
                context,
                "--kubeconfig",
                str(kubeconfig),
                "--timeout",
                str(int(bound.arguments.get("timeout") or 120)),
                "--force",
            ]
            if not skip_k8s
            else []
        )
        operation_kwargs = dict(
            command="npa provision-if-absent",
            project_alias=alias,
            project_id=str(getattr(environment, "project_id", "") or ""),
            tenant_id=str(getattr(environment, "tenant_id", "") or ""),
            region=str(getattr(environment, "region", "") or ""),
            backend={
                "bucket": _bucket_name(storage.checkpoint_bucket),
                "endpoint": storage.endpoint_url,
                "region": str(getattr(environment, "region", "") or ""),
            },
            resource_type=resource_type,
            requested_name=requested_name or resource_type,
            ownership_source="provision-if-absent",
            resume_command="",
            destroy_command="",
            resume_argv=resume_argv,
            destroy_argv=destroy_argv,
        )
        operation = ProvisioningOperation.prepare(**operation_kwargs)
        try:
            operation.record_preflight_plan(plan.to_dict())
        except OperationIdentityError:
            # A retry with a different topology must never overwrite an
            # incomplete operation's immutable plan. When no cloud resource,
            # local Terraform state, or config mutation was recorded, however,
            # the old operation is safely empty: terminalize that evidence and
            # start a new deterministic generation for the new requested shape.
            prior = operation.read()
            if any(
                prior.get(key)
                for key in ("resources", "local_state_copies", "config_mutations")
            ):
                raise
            operation.record_rollback(
                attempted=False,
                completed=True,
                removed=[],
                preserved=[],
                error="superseded empty operation after requested topology changed",
            )
            operation.transition(
                "rolled-back",
                error="superseded empty operation after requested topology changed",
            )
            operation = ProvisioningOperation.prepare(**operation_kwargs)
            operation.record_preflight_plan(plan.to_dict())
        with operation_context(operation):
            sys.stderr.write(
                f"Provisioning operation {operation.operation_id}: preflight complete; beginning mutation\n"
            )
            sys.stderr.flush()
            operation.transition("mutating")
            try:
                result = function(*args, **kwargs)
            except BaseException as exc:
                operation.record_failure(exc)
                if str(operation.read().get("phase") or "") not in {
                    "rolled-back",
                    "destroyed",
                }:
                    operation.transition("recovery-required")
                    _rollback_owned_cluster(
                        operation,
                        project_alias=alias,
                        context=context,
                        terraform_dir=bound.arguments.get("terraform_dir"),
                        kubeconfig=bound.arguments.get("kubeconfig"),
                        timeout=int(bound.arguments.get("timeout") or 120),
                    )
                sys.stderr.write(emit_recovery_summary(operation) + "\n")
                raise
            if result.status == "partial":
                operation.transition(
                    "recovery-required",
                    error="; ".join(result.warnings) or "provisioning returned partial",
                    details={"error_type": "PartialProvisioningError"},
                )
                rolled_back = _rollback_owned_cluster(
                    operation,
                    project_alias=alias,
                    context=context,
                    terraform_dir=bound.arguments.get("terraform_dir"),
                    kubeconfig=bound.arguments.get("kubeconfig"),
                    timeout=int(bound.arguments.get("timeout") or 120),
                )
                if rolled_back:
                    result.actions.append(
                        "rollback:removed only cluster resources created by this operation"
                    )
            else:
                phase = str(operation.read().get("phase") or "")
                if phase in {"mutating", "resource-created"}:
                    operation.transition("state-durable")
                if phase not in {"rolled-back", "rollback-incomplete"}:
                    operation.commit()
            summary = operation.recovery_summary()
            result.operation_id = operation.operation_id
            result.operation_journal = str(operation.path)
            result.recovery_command = str(summary.get("resume_command") or "")
            return result

    return wrapped


@_transactional_provision
def provision_if_absent(
    *,
    project: str | None = None,
    cluster_name: str = "npa-cluster",
    terraform_dir: Path | None = None,
    kubeconfig: Path | None = None,
    context_name: str = "",
    skip_k8s: bool = False,
    skip_s3: bool = False,
    validate: bool = True,
    sky_smoke: bool = False,
    dry_run: bool = False,
    timeout: int = 120,
    gpu_nodes: int = -1,
    cpu_nodes: int = -1,
    cpu_platform: str = "",
    cpu_preset: str = "",
    gpu_platform: str = "",
    gpu_preset: str = "",
    gpu_driver_mode: str = "",
    gpu_workload_profile: str = "",
    managed_driver_preset: str = "",
    allow_unsafe_nvswitch_operator: bool | None = None,
    gpu_health_stabilization_seconds: int = DEFAULT_STABILIZATION_SECONDS,
    gpu_health_timeout_minutes: int = 60,
    gpu_cuda_smoke: bool = True,
    gpu_cuda_smoke_image: str = DEFAULT_CUDA_SMOKE_IMAGE,
    gpu_graphics_smoke_image: str = DEFAULT_GRAPHICS_SMOKE_IMAGE,
    mig_enabled: bool = False,
    mig_strategy: str = "mixed",
    mig_config: str = "all-balanced",
    capacity_block_group: str = "",
    infiniband_fabric: str = "",
    preemptible: bool | None = None,
    accelerator: str = "",
    gpu_readiness_timeout: float = 600.0,
    gpu_readiness_poll_interval: float = 10.0,
    sky_bin: str = "",
    agent_exists: bool = False,
    reconcile_empty_failure: bool = False,
    recovery_evidence: Path | None = None,
    output_format: str = "text",
    _resolved_plan=None,
) -> ProvisionIfAbsentResult:
    """Ensure configured S3 and Kubernetes exist without deleting resources."""
    from npa.lifecycle_intent import forbid_destructive_provisioning

    forbid_destructive_provisioning("provision_if_absent")
    alias, environment, storage, registry = _resolve_project_runtime(project)
    workload = resolve_gpu_workload_profile(
        profile=gpu_workload_profile,
        gpu_nodes=gpu_nodes,
        gpu_platform=gpu_platform,
        gpu_preset=gpu_preset,
        gpu_driver_mode=gpu_driver_mode,
    )
    gpu_nodes = workload.gpu_nodes
    gpu_platform = workload.gpu_platform
    gpu_preset = workload.gpu_preset
    gpu_driver_mode = workload.gpu_driver_mode
    if validate and not skip_k8s and not dry_run and workload.graphics_smoke:
        gpu_graphics_smoke_image = resolve_graphics_smoke_image(
            gpu_graphics_smoke_image
        )
    context = context_name.strip() or cluster_name
    kubeconfig_path = kubeconfig or kubeconfig_file(context)
    actions: list[str] = []
    warnings: list[str] = []
    k8s_ready = False
    gpu_readiness = "not-requested"
    plan = _resolved_plan or _build_provision_plan(
        alias=alias,
        environment=environment,
        cluster_name=cluster_name,
        kubeconfig=Path(kubeconfig_path),
        context=context,
        skip_k8s=skip_k8s,
        dry_run=dry_run,
        accelerator=accelerator,
        gpu_nodes=gpu_nodes,
        cpu_nodes=cpu_nodes,
        cpu_platform=cpu_platform,
        cpu_preset=cpu_preset,
        gpu_platform=gpu_platform,
        gpu_preset=gpu_preset,
        capacity_block_group=capacity_block_group,
        preemptible=preemptible,
        agent_exists=agent_exists,
    )
    topology = plan.topology
    gpu_nodes = topology.gpu_nodes
    cpu_nodes = topology.cpu_nodes
    cpu_platform = topology.cpu_platform
    cpu_preset = topology.cpu_preset
    gpu_platform = topology.gpu_platform
    gpu_preset = topology.gpu_preset
    capacity_block_group = topology.capacity_block_group
    preemptible = topology.gpu_preemptible
    actions.extend(_preflight_actions(plan))

    storage_ready = skip_s3
    storage_bucket = storage.checkpoint_bucket
    storage_endpoint = storage.endpoint_url
    if skip_s3:
        actions.append("s3:skipped")
    else:
        from npa.clients.nebius import bucket_name_for, redact_nebius_output
        from npa.clients.storage_setup import provision_storage, storage_setup_record
        from npa.clients.storage_validation import probe_storage_write

        bucket_name = _bucket_name(storage.checkpoint_bucket)
        partial = (
            storage_setup_record(environment.project_id)
            if environment.project_id
            else {}
        )
        bucket_name = bucket_name or str(partial.get("bucket_name", "") or "").strip()
        if not bucket_name and environment.project_id and environment.tenant_id:
            bucket_name = bucket_name_for(environment.tenant_id, environment.project_id)
        if not environment.project_id:
            warnings.append("project_id is required to ensure S3")
        elif not environment.tenant_id:
            warnings.append("tenant_id is required to ensure S3")
        elif dry_run:
            action = (
                "reconcile"
                if partial and partial.get("status") != "complete"
                else "ensure"
            )
            actions.append(f"s3:dry-run {action} writable bucket {bucket_name}")
            storage_ready = True
        else:
            probe = probe_storage_write(
                bucket=bucket_name,
                endpoint_url=storage.endpoint_url,
                access_key_id=storage.aws_access_key_id,
                secret_access_key=storage.aws_secret_access_key,
                region=environment.region,
            )
            if probe.ok:
                storage_ready = True
                actions.append(f"s3:verified writable bucket {bucket_name}")
                if probe.retained_object:
                    warnings.append(probe.summary)
            else:
                try:
                    credentials, reconciled_probe = provision_storage(
                        project_id=environment.project_id,
                        tenant_id=environment.tenant_id,
                        region=environment.region,
                        bucket_name=bucket_name,
                        project_alias=alias,
                    )
                except Exception as exc:  # noqa: BLE001 - return an actionable partial result
                    warnings.append(
                        "writable S3 reconciliation failed: "
                        + redact_nebius_output(str(exc))
                    )
                    actions.append(
                        "s3:partial; resume with `npa provision-if-absent"
                        + (f" --project {alias}" if alias else "")
                        + " --skip-k8s`"
                    )
                else:
                    storage_ready = reconciled_probe.ok
                    storage_bucket = f"s3://{credentials['s3_bucket'].strip().removeprefix('s3://').strip('/')}/"
                    storage_endpoint = credentials["s3_endpoint"]
                    # The transaction committed the newly validated credentials;
                    # re-resolve so a following Terraform apply receives them.
                    storage = config_module.resolve_project_storage(alias or None)
                    actions.append(f"s3:reconciled writable bucket {bucket_name}")

    if not storage_ready and not skip_s3:
        actions.append("k8s:blocked until writable S3 is reconciled")
    elif skip_k8s:
        actions.append("k8s:skipped")
    elif not dry_run and _has_cached_kubeconfig(context, kubeconfig_path):
        actions.append(f"k8s:reused kubeconfig {kubeconfig_path}")
        if validate and gpu_nodes > 0:
            if mig_enabled:
                from npa.cluster_backends import get_backend
                from npa.cluster_backends.mk8s import MK8sStatusRequest
                from npa.cluster_backends.mig import MigSpec
                from npa.fleet.spec import ClusterSpec, NodePoolSpec

                desired = ClusterSpec(
                    name=context,
                    cpu_nodes=(
                        NodePoolSpec(
                            count=cpu_nodes,
                            platform=cpu_platform,
                            preset=cpu_preset,
                        )
                        if cpu_nodes
                        else None
                    ),
                    gpu_nodes=NodePoolSpec(
                        count=gpu_nodes,
                        platform=gpu_platform,
                        preset=gpu_preset,
                        disk_size_gib=128,
                        capacity_block_group=capacity_block_group,
                        preemptible=bool(preemptible),
                    ),
                    gpu_driver_mode="operator",
                    gpu_health_stabilization_seconds=gpu_health_stabilization_seconds,
                    gpu_health_timeout_minutes=gpu_health_timeout_minutes,
                    gpu_cuda_smoke=gpu_cuda_smoke,
                    gpu_cuda_smoke_image=gpu_cuda_smoke_image,
                    gpu_graphics_smoke_image=gpu_graphics_smoke_image,
                    mig=MigSpec(enabled=True, strategy=mig_strategy, config=mig_config),
                )
                kubectl_bin = os.environ.get("NPA_KUBECTL_BIN") or "kubectl"
                get_backend("mk8s").verify(
                    desired,
                    MK8sStatusRequest(
                        kubeconfig=Path(kubeconfig_path), kubectl_bin=kubectl_bin
                    ),
                )
                actions.append("k8s:verified exact MIG topology and CUDA allocation")
            else:
                from npa.cli.cluster.terraform_lifecycle import (
                    _require_bin,
                    _validate_cluster,
                )

                kubectl_bin = _require_bin(
                    os.environ.get("NPA_KUBECTL_BIN") or "kubectl"
                )
                _validate_cluster(
                    kubectl_bin,
                    Path(kubeconfig_path),
                    {
                        "cpu_nodes_count": cpu_nodes,
                        "gpu_nodes_count": gpu_nodes,
                        "gpu_nodes_platform": gpu_platform,
                        "gpu_nodes_preset": gpu_preset,
                        "gpu_driver_mode": gpu_driver_mode or "auto",
                        "managed_driver_preset": (
                            managed_driver_preset or DEFAULT_MANAGED_DRIVER_PRESET
                        ),
                        "allow_unsafe_nvswitch_operator": bool(
                            allow_unsafe_nvswitch_operator
                        ),
                    },
                    gpu_health_timeout_minutes,
                    gpu_health_stabilization_seconds=(gpu_health_stabilization_seconds),
                    gpu_cuda_smoke=gpu_cuda_smoke,
                    gpu_cuda_smoke_image=gpu_cuda_smoke_image,
                    gpu_graphics_smoke=workload.graphics_smoke,
                    gpu_graphics_smoke_image=gpu_graphics_smoke_image,
                )
                actions.append("k8s:validated stable GPU health and CUDA vectorAdd")
                if workload.graphics_smoke:
                    actions.append("k8s:validated RTX GLX/EGL/Vulkan readiness")
        k8s_ready = True
    elif not dry_run and (not environment.project_id or not environment.tenant_id):
        warnings.append("project_id and tenant_id are required to ensure Kubernetes")
    elif dry_run:
        from npa.cluster_backends import get_backend
        from npa.cluster_backends.mk8s import MK8sApplyRequest
        from npa.cluster_backends.mig import MigSpec
        from npa.fleet.spec import ClusterSpec, NodePoolSpec

        # The immutable preflight topology above already resolved all defaults.
        # A dry-run must remain package-only: do not discover deploy/cluster,
        # read tfvars, clone a recipe, or materialize anything on disk.
        desired_gpu_count = gpu_nodes
        desired_cpu_count = cpu_nodes
        backend_desired = ClusterSpec(
            name=context,
            cpu_nodes=(
                NodePoolSpec(
                    count=desired_cpu_count,
                    platform=cpu_platform,
                    preset=cpu_preset,
                    disk_size_gib=topology.cpu_disk_gib,
                )
                if desired_cpu_count
                else None
            ),
            gpu_nodes=(
                NodePoolSpec(
                    count=desired_gpu_count,
                    platform=gpu_platform,
                    preset=gpu_preset,
                    disk_size_gib=(128 if mig_enabled else topology.gpu_disk_gib),
                    capacity_block_group=capacity_block_group,
                    preemptible=bool(preemptible),
                )
                if desired_gpu_count
                else None
            ),
            gpu_driver_mode="operator" if mig_enabled else (gpu_driver_mode or "auto"),
            managed_driver_preset=(
                managed_driver_preset or DEFAULT_MANAGED_DRIVER_PRESET
            ),
            allow_unsafe_nvswitch_operator=bool(allow_unsafe_nvswitch_operator),
            gpu_health_stabilization_seconds=gpu_health_stabilization_seconds,
            gpu_health_timeout_minutes=gpu_health_timeout_minutes,
            gpu_cuda_smoke=gpu_cuda_smoke,
            gpu_cuda_smoke_image=gpu_cuda_smoke_image,
            gpu_graphics_smoke_image=gpu_graphics_smoke_image,
            gpu_workload_profile=workload.profile,
            infiniband_fabric=infiniband_fabric,
            mig=(
                MigSpec(enabled=True, strategy=mig_strategy, config=mig_config)
                if mig_enabled
                else None
            ),
            allow_control_plane_only=True,
        )
        backend = get_backend("mk8s")
        backend_plan = backend.plan(backend_desired)
        backend.preflight(backend_desired, MK8sApplyRequest())
        actions.append(
            "k8s:shared-backend plan "
            f"backend={backend_plan['backend']} target={backend_plan['name']} "
            f"cpu_nodes={backend_plan['cpu_nodes']} "
            f"gpu_nodes={backend_plan['gpu_nodes']} provider_mutation=false"
        )
        shape = ", ".join(
            [
                f"{name}={count}"
                for name, count in (("gpu_nodes", gpu_nodes), ("cpu_nodes", cpu_nodes))
                if count >= 0
            ]
            + (
                [f"preemptible={str(bool(preemptible)).lower()}"]
                if preemptible is not None
                else []
            )
            + [
                f"{name}={value.strip()}"
                for name, value in (
                    ("cpu_platform", cpu_platform),
                    ("cpu_preset", cpu_preset),
                    ("gpu_platform", gpu_platform),
                    ("gpu_preset", gpu_preset),
                    ("gpu_driver_mode", gpu_driver_mode),
                    ("gpu_workload_profile", workload.profile),
                    ("managed_driver_preset", managed_driver_preset),
                )
                if value.strip()
            ]
        )
        actions.append(
            f"k8s:dry-run terraform apply {terraform_dir or 'deploy/cluster'}"
            + (f" ({shape})" if shape else "")
        )
    else:
        from npa.provisioning_preflight import resolved_plan_context

        with (
            _runtime_env(alias, environment, storage, registry),
            resolved_plan_context(plan),
        ):
            from npa.cli.cluster.terraform_lifecycle import up_cmd

            # Every parameter is passed explicitly: `up_cmd` is a Typer command,
            # so an omitted one arrives as an OptionInfo sentinel rather than its
            # default (the §23 bug), and `gpu_nodes`/`cpu_nodes` would reach the
            # Terraform override as objects.
            up_cmd(
                terraform_dir=terraform_dir,
                kubeconfig=kubeconfig_path,
                context_name=context,
                project=alias or "",
                validate=validate,
                # Provisioning owns one shared cached/fresh GPU-readiness and
                # smoke boundary below. Running smoke inside up_cmd would put
                # the fresh path ahead of label repair and readiness.
                sky_smoke=False,
                sky_gpus="",
                sky_bin=sky_bin,
                capacity_block_group=capacity_block_group,
                infiniband_fabric=infiniband_fabric,
                gpu_nodes=gpu_nodes,
                cpu_nodes=cpu_nodes,
                cpu_platform=cpu_platform,
                cpu_preset=cpu_preset,
                gpu_platform=gpu_platform,
                gpu_preset=gpu_preset,
                gpu_driver_mode=gpu_driver_mode,
                gpu_workload_profile=workload.profile,
                managed_driver_preset=managed_driver_preset,
                allow_unsafe_nvswitch_operator=allow_unsafe_nvswitch_operator,
                gpu_health_stabilization_seconds=(gpu_health_stabilization_seconds),
                gpu_cuda_smoke=gpu_cuda_smoke,
                gpu_cuda_smoke_image=gpu_cuda_smoke_image,
                gpu_graphics_smoke_image=gpu_graphics_smoke_image,
                mig_enabled=mig_enabled,
                mig_strategy=mig_strategy,
                mig_config=mig_config,
                preemptible=preemptible,
                validation_timeout=gpu_health_timeout_minutes,
                timeout=timeout,
            )
        actions.append(f"k8s:ensured terraform cluster {context}")
        k8s_ready = True

    requested_accelerator = str(accelerator or "").strip()
    needs_gpu_setup = bool(requested_accelerator or sky_smoke)
    if needs_gpu_setup and k8s_ready and not skip_k8s and not dry_run:
        from npa.controller_ownership import ensure_controller_owner
        from npa.cli.cluster.terraform_lifecycle import (
            _check_skypilot_kubernetes,
            _recover_skypilot_smoke,
            _run_skypilot_smoke,
        )
        from npa.orchestration.skypilot.k8s_gpu_catalog import (
            wait_for_kubernetes_accelerators,
        )

        try:
            owner = ensure_controller_owner(alias, context)
            actions.append(
                f"controller:bound {owner.project_alias}/{owner.context}/{owner.cluster_id}"
            )

            from npa.orchestration.skypilot.cluster_validation import (
                cluster_validation_session,
            )

            with cluster_validation_session(Path(kubeconfig_path), context):
                _check_skypilot_kubernetes(
                    Path(kubeconfig_path),
                    context,
                    sky_bin=sky_bin,
                )
                actions.append("skypilot:kubernetes-enabled")
                if sky_smoke:
                    _recover_skypilot_smoke(
                        Path(kubeconfig_path), context, cluster_name, sky_bin=sky_bin
                    )

                def report_gpu_status(message: str) -> None:
                    actions.append(f"gpu:{message}")
                    sys.stderr.write(message.rstrip() + "\n")
                    sys.stderr.flush()
                    operation = current_operation()
                    if operation is not None:
                        operation.heartbeat(details={"gpu_readiness": message})

                resolutions = wait_for_kubernetes_accelerators(
                    [requested_accelerator] if requested_accelerator else [],
                    context=context,
                    kubeconfig=kubeconfig_path,
                    sky_bin=sky_bin or None,
                    label_known_gpus=True,
                    timeout=gpu_readiness_timeout,
                    poll_interval=gpu_readiness_poll_interval,
                    on_status=report_gpu_status,
                )
                if sky_smoke:
                    smoke_accelerator = requested_accelerator
                    if requested_accelerator and requested_accelerator in resolutions:
                        smoke_accelerator = resolutions[requested_accelerator].resolved
                    _run_skypilot_smoke(
                        Path(kubeconfig_path),
                        context,
                        cluster_name,
                        smoke_accelerator,
                        sky_bin=sky_bin,
                        credentials_checked=True,
                    )
                    actions.append("sky-smoke:passed")
        except Exception as exc:  # noqa: BLE001 - return a resumable partial result
            gpu_readiness = (
                "timeout" if str(exc).startswith("Timed out after ") else "failed"
            )
            warnings.append(str(exc))
            operation = current_operation()
            created_cluster = bool(
                operation
                and any(
                    isinstance(item, dict)
                    and item.get("resource_type") == "managed_kubernetes_cluster"
                    and item.get("ownership") == "created_by_this_operation"
                    for item in operation.read().get("resources", [])
                )
            )
            actions.append(
                "gpu:new cluster will be transactionally rolled back"
                if created_cluster
                else "gpu:pre-existing capacity preserved; resume readiness without reprovisioning"
            )
        else:
            gpu_readiness = "ready"
            actions.append(
                "gpu:SkyPilot ready for "
                + (requested_accelerator or "auto-detected smoke accelerator")
            )
    elif needs_gpu_setup and dry_run:
        gpu_readiness = "dry-run"
        actions.append(
            "gpu:dry-run wait for SkyPilot "
            + (requested_accelerator or "auto-detected smoke accelerator")
        )

    status = "ok" if not warnings and (skip_s3 or storage_ready) else "partial"
    if dry_run:
        status = plan.decision
    return ProvisionIfAbsentResult(
        status=status,
        project=alias,
        cluster_name=cluster_name,
        kubeconfig_path=str(kubeconfig_path),
        context_name=context,
        storage_bucket=storage_bucket,
        storage_endpoint=storage_endpoint,
        registry=registry,
        gpu_readiness=gpu_readiness,
        actions=actions,
        warnings=warnings,
        preflight=plan.to_dict(),
    )


def _build_provision_plan(
    *,
    alias: str,
    environment: EnvironmentConfig,
    cluster_name: str,
    kubeconfig: Path,
    context: str,
    skip_k8s: bool,
    dry_run: bool,
    accelerator: str,
    gpu_nodes: int,
    cpu_nodes: int,
    cpu_platform: str,
    cpu_preset: str,
    gpu_platform: str,
    gpu_preset: str,
    capacity_block_group: str,
    preemptible: bool | None,
    agent_exists: bool = False,
):
    from npa.provisioning_preflight import (
        build_whole_path_plan,
        discover_existing_capacity,
        resolve_topology,
    )

    cluster_exists = skip_k8s or _has_cached_kubeconfig(context, kubeconfig)
    cpu_disk_gib = _terraform_disk_size_gib("TF_VAR_cpu_disk_size", 128)
    gpu_disk_gib = _terraform_disk_size_gib("TF_VAR_gpu_disk_size", 1023)
    requested = resolve_topology(
        cluster_name=cluster_name,
        accelerator=accelerator,
        agent_exists=agent_exists,
        cpu_nodes=0 if skip_k8s else cpu_nodes,
        gpu_nodes=0 if skip_k8s else gpu_nodes,
        cpu_platform=cpu_platform,
        cpu_preset=cpu_preset,
        gpu_platform=gpu_platform,
        gpu_preset=gpu_preset,
        capacity_block_group=capacity_block_group,
        preemptible=preemptible,
        cpu_disk_gib=cpu_disk_gib,
        gpu_disk_gib=gpu_disk_gib,
    )
    checks = []
    if skip_k8s:
        existing_cpu_nodes = existing_gpu_nodes = 0
    elif cluster_exists:
        existing_cpu_nodes = requested.cpu_nodes
        existing_gpu_nodes = requested.gpu_nodes
    else:
        existing = discover_existing_capacity(
            project_id=str(getattr(environment, "project_id", "") or ""),
            cluster_name=cluster_name,
            cpu_platform=requested.cpu_platform,
            cpu_preset=requested.cpu_preset,
            gpu_platform=requested.gpu_platform,
            gpu_preset=requested.gpu_preset,
        )
        existing_cpu_nodes = min(requested.cpu_nodes, existing.cpu_nodes)
        existing_gpu_nodes = min(requested.gpu_nodes, existing.gpu_nodes)
        checks.append(existing.check)
    if accelerator and not skip_k8s:
        from npa.controller_ownership import controller_preflight

        owner_status, owner_reason = controller_preflight(alias, context)
        from npa.provisioning_preflight import PreflightCheck

        checks.append(
            PreflightCheck(
                name="controller_ownership",
                status=owner_status,
                reason=owner_reason,
            )
        )
    topology = resolve_topology(
        cluster_name=cluster_name,
        accelerator=accelerator,
        agent_exists=agent_exists,
        cpu_nodes=requested.cpu_nodes,
        gpu_nodes=requested.gpu_nodes,
        existing_cpu_nodes=existing_cpu_nodes,
        existing_gpu_nodes=existing_gpu_nodes,
        cpu_platform=requested.cpu_platform,
        cpu_preset=requested.cpu_preset,
        gpu_platform=requested.gpu_platform,
        gpu_preset=requested.gpu_preset,
        capacity_block_group=requested.capacity_block_group,
        preemptible=requested.gpu_preemptible,
        cpu_disk_gib=requested.cpu_disk_gib,
        gpu_disk_gib=requested.gpu_disk_gib,
    )
    quota_reader = None
    quota_names = tuple(topology.quota_requirements())

    def fixed_quota_reader(observations):
        def read_quota_snapshot(_tenant, _region, _names):
            return observations

        return read_quota_snapshot

    try:
        from npa.provisioning_preflight import read_provider_quotas

        tenant_observations = read_provider_quotas(
            str(getattr(environment, "tenant_id", "") or ""),
            str(getattr(environment, "region", "") or ""),
            quota_names,
        )
    except Exception as tenant_exc:  # noqa: BLE001 - a project fallback is narrowly typed
        from npa.clients.nebius import is_permission_denied

        project_id = str(getattr(environment, "project_id", "") or "")
        if project_id and is_permission_denied(str(tenant_exc)):
            try:
                from npa.provisioning_preflight import (
                    PROJECT_QUOTA_RBAC_FALLBACK_REASON,
                    PreflightCheck,
                    read_project_quota_observations,
                )

                project_observations = read_project_quota_observations(
                    project_id,
                    str(getattr(environment, "region", "") or ""),
                    quota_names,
                )
            except Exception as project_exc:  # noqa: BLE001 - preserve fail-closed quota evidence
                # Keep the ordinary planner's fail-closed unknown evidence when
                # the exact-project view is unavailable or malformed too.
                logging.getLogger(__name__).debug(
                    "project-scoped quota fallback was unavailable",
                    exc_info=project_exc,
                )
            else:
                quota_reader = fixed_quota_reader(project_observations)
                evidence_complete = all(
                    observation.state in {"known", "unbounded"}
                    for observation in project_observations.values()
                )
                checks.append(
                    PreflightCheck(
                        name="quota_evidence_scope",
                        status="ready" if evidence_complete else "unknown",
                        reason=(
                            PROJECT_QUOTA_RBAC_FALLBACK_REASON
                            if evidence_complete
                            else (
                                "tenant-wide quota query unavailable due to RBAC; "
                                "project-scoped quota response did not verify every "
                                "requested allowance"
                            )
                        ),
                    )
                )
    else:
        quota_reader = fixed_quota_reader(tenant_observations)
    return build_whole_path_plan(
        project_alias=alias,
        project_id=str(getattr(environment, "project_id", "") or ""),
        tenant_id=str(getattr(environment, "tenant_id", "") or ""),
        region=str(getattr(environment, "region", "") or ""),
        topology=topology,
        quota_reader=quota_reader,
        checks=checks,
        mutation=not dry_run,
    )


def _terraform_disk_size_gib(name: str, default: int) -> int:
    """Resolve the Terraform worker-disk override used by ``cluster up``.

    ``provision-if-absent`` delegates the real apply to ``cluster up``, which
    already honors ``TF_VAR_cpu_disk_size`` and ``TF_VAR_gpu_disk_size``. The
    immutable outer preflight must account for those same values or it can
    reject an otherwise valid topology before Terraform sees it.
    """

    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer GiB value") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer GiB value")
    return value


def resolve_provision_plan(
    *,
    project: str | None = None,
    cluster_name: str = "npa-cluster",
    kubeconfig: Path | None = None,
    context_name: str = "",
    skip_k8s: bool = False,
    accelerator: str = "",
    gpu_nodes: int = -1,
    cpu_nodes: int = -1,
    cpu_platform: str = "",
    cpu_preset: str = "",
    gpu_platform: str = "",
    gpu_preset: str = "",
    capacity_block_group: str = "",
    preemptible: bool | None = None,
    mutation: bool = False,
):
    """Resolve the immutable plan without creating journals or resources."""

    alias, environment, _storage, _registry = _resolve_project_runtime(project)
    context = context_name.strip() or cluster_name
    kubeconfig_path = kubeconfig or kubeconfig_file(context)
    return _build_provision_plan(
        alias=alias,
        environment=environment,
        cluster_name=cluster_name,
        kubeconfig=Path(kubeconfig_path),
        context=context,
        skip_k8s=skip_k8s,
        dry_run=not mutation,
        accelerator=accelerator,
        gpu_nodes=gpu_nodes,
        cpu_nodes=cpu_nodes,
        cpu_platform=cpu_platform,
        cpu_preset=cpu_preset,
        gpu_platform=gpu_platform,
        gpu_preset=gpu_preset,
        capacity_block_group=capacity_block_group,
        preemptible=preemptible,
    )


def _preflight_actions(plan) -> list[str]:
    topology = plan.topology
    actions = [
        "preflight:"
        f"{plan.decision} cpu={topology.cpu_nodes}x{topology.cpu_platform}/{topology.cpu_preset} "
        f"gpu={topology.gpu_nodes}x{topology.gpu_platform}/{topology.gpu_preset} "
        f"preemptible={str(topology.gpu_preemptible).lower()} disks={topology.required_disks} "
        f"network_ssd_bytes={topology.required_network_ssd_bytes}"
    ]
    for quota in plan.quotas:
        rendered = (
            "quota:"
            f"{quota.name}:{quota.status} required={quota.required} used={quota.used} "
            f"limit={quota.limit} available={quota.available} shortfall={quota.shortfall}"
        )
        structured = quota.to_dict()
        if structured.get("unit") == "bytes":
            rendered += (
                f" bytes required_gib={structured.get('required_gib')} "
                f"available_gib={structured.get('available_gib')} "
                f"shortfall_gib={structured.get('shortfall_gib')}"
            )
        actions.append(rendered)
    return actions


def _rollback_owned_cluster(
    operation: ProvisioningOperation,
    *,
    project_alias: str,
    context: str,
    terraform_dir: Path | None,
    kubeconfig: Path | None,
    timeout: int,
) -> bool:
    """Roll back only a cluster explicitly receipted as created by this operation."""

    payload = operation.read()
    if str(payload.get("phase") or "") == "rolled-back":
        return True
    created = [
        item
        for item in payload.get("resources", [])
        if isinstance(item, dict)
        and item.get("ownership") == "created_by_this_operation"
    ]
    owned = [
        item
        for item in created
        if item.get("resource_type") == "managed_kubernetes_cluster"
    ]
    if not created:
        operation.record_rollback(
            attempted=False,
            completed=False,
            removed=[],
            preserved=list(payload.get("resources") or []),
        )
        return False
    if not owned or not str(owned[0].get("provider_id") or ""):
        error = (
            "rollback requires the exact created managed Kubernetes cluster ID; "
            "owned resources were preserved for exact retry"
        )
        operation.record_rollback(
            attempted=True,
            completed=False,
            removed=[],
            preserved=list(payload.get("resources") or []),
            outcomes=[
                {**item, "outcome": "preserved_missing_cluster_identity"}
                for item in created
            ],
            error=error,
        )
        operation.transition("rollback-incomplete", error=error)
        return False
    terraform_owned = [
        item
        for item in payload.get("resources", [])
        if isinstance(item, dict)
        and item.get("ownership") == "created_by_this_operation"
        and str(item.get("ownership_source") or "").startswith("terraform")
    ]
    preserved = [
        item for item in payload.get("resources", []) if item not in terraform_owned
    ]
    operation.record_rollback(
        attempted=True,
        completed=False,
        removed=[],
        preserved=list(payload.get("resources") or []),
        outcomes=[{**item, "outcome": "rollback_started"} for item in terraform_owned],
    )
    operation.transition("rolling-back")
    try:
        from npa.cli.cluster.terraform_lifecycle import down_cmd

        down_cmd(
            terraform_dir=terraform_dir,
            project=project_alias,
            receipt="",
            project_id="",
            tenant_id="",
            region="",
            cluster_id=str(owned[0].get("provider_id") or ""),
            operation_id=operation.operation_id,
            context_name=context,
            keep_local_state=False,
            force=True,
            timeout=timeout,
            kubeconfig=kubeconfig,
            output_json=False,
        )
    except BaseException as exc:
        operation.record_rollback(
            attempted=True,
            completed=False,
            removed=[],
            preserved=list(payload.get("resources") or []),
            outcomes=[
                {**item, "outcome": "rollback_failed", "error": str(exc)}
                for item in terraform_owned
            ],
            error=str(exc),
        )
        operation.transition("rollback-incomplete", error=str(exc))
        return False
    try:
        from npa.controller_ownership import clear_controller_owner

        clear_controller_owner(
            project_id=str(payload.get("project_id") or ""),
            cluster_id=str(owned[0].get("provider_id") or ""),
            context=context,
        )
    except (OSError, RuntimeError, ValueError):
        # cluster down succeeded; a non-matching/unavailable owner is preserved
        # for explicit reconciliation and is never cleared by alias alone.
        pass
    operation.record_rollback(
        attempted=True,
        completed=True,
        removed=terraform_owned,
        preserved=preserved,
        outcomes=[
            *({**item, "outcome": "removed"} for item in terraform_owned),
            *(
                {**item, "outcome": "preserved_not_owned_by_terraform"}
                for item in preserved
            ),
        ],
    )
    operation.transition("rolled-back")
    return True


def _resolve_project_runtime(
    project: str | None,
) -> tuple[str, EnvironmentConfig, StorageConfig, str]:
    yml = config_module._load_yaml()
    alias = config_module._resolved_project_name(yml, project)
    environment = config_module.resolve_environment(project) or EnvironmentConfig(
        "", "", ""
    )
    storage = config_module.resolve_project_storage(project)
    registry = config_module.resolve_container_registry(project)
    return alias, environment, storage, registry


def _bucket_name(uri_or_name: str) -> str:
    value = uri_or_name.strip()
    if not value:
        return ""
    if value.startswith("s3://"):
        return urlparse(value).netloc
    return value.split("/", 1)[0]


def _has_cached_kubeconfig(context: str, kubeconfig_path: Path) -> bool:
    if kubeconfig_path.exists():
        return True
    state = load_cluster_state(context)
    return bool(
        state and state.kubeconfig_path and Path(state.kubeconfig_path).exists()
    )


@contextmanager
def _runtime_env(
    alias: str,
    environment: EnvironmentConfig,
    storage: StorageConfig,
    registry: str,
) -> Iterator[None]:
    values = {
        "NPA_PROJECT_ID": environment.project_id,
        "NPA_TENANT_ID": environment.tenant_id,
        "NPA_REGION": environment.region,
        "NPA_REGISTRY": registry,
        # Consumers of NPA_S3_BUCKET pass it as the provider Bucket argument;
        # keep URI/prefix forms in checkpoint_bucket only.
        "NPA_S3_BUCKET": _bucket_name(storage.checkpoint_bucket),
        "NPA_STORAGE_ENDPOINT": storage.endpoint_url,
        "AWS_ENDPOINT_URL": storage.endpoint_url,
        "NEBIUS_S3_ENDPOINT": storage.endpoint_url,
        "AWS_ACCESS_KEY_ID": storage.aws_access_key_id,
        "AWS_SECRET_ACCESS_KEY": storage.aws_secret_access_key,
        "TF_VAR_parent_id": environment.project_id,
        "TF_VAR_tenant_id": environment.tenant_id,
        "TF_VAR_region": environment.region,
    }
    previous = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value:
                os.environ[key] = value
        yield
    finally:
        for key, previous_value in previous.items():
            if previous_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous_value
