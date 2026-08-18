"""Launch a real LeIsaac browser-teleoperation session on Kubernetes."""

from __future__ import annotations

import base64
import hashlib
import http.client
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import secrets
import shlex
import socket
import ssl
import threading
import time
from enum import Enum
from typing import Any, Callable, NoReturn

import typer

from npa.agent_backend.leisaac_registry import (
    DEFAULT_ENVIRONMENT_ID,
    DEFAULT_TASK,
    REGISTRY_FINGERPRINT,
    registry_payload,
    validate_environment_id,
    validate_environment_index,
    validate_num_envs,
    validate_seed,
    validate_task,
)
from npa.clients.config import SSHConfig, list_projects
from npa.clients.kube import KubectlResult, run_kubectl
from npa.clients.network import (
    ensure_ingress,
    remove_exact_npa_ingress_for_instance,
    resolve_instance_network_context,
)
from npa.clients.ssh import SSHClient, SSHTimeoutError
from npa.serverless_common.env import (
    MissingIsaacEulaAcceptanceError,
    require_isaac_eula_acceptance,
)
from npa.workbench.leisaac import (
    GPU_PRODUCT,
    GPU_PRODUCT_LABEL,
    GPU_PROVIDER_LABEL,
    GPU_PROVIDER_VALUE,
    SESSION_SCHEMA,
    SOURCE_COMMIT,
    LeIsaacConfigError,
    MEDIA_PORT,
    RELAY_SERVICE_PORT,
    TURN_PORT,
    TURN_RELAY_PORT,
    TRANSPORT_AGENT_RELAY,
    TRANSPORT_LOAD_BALANCER,
    deployment_manifest,
    relay_client_secret_manifest,
    recorder_secret_manifest,
    relay_service_manifest,
    resource_name,
    session_manifest,
    session_attestation,
    split_s3_uri,
    validate_expiry,
    validate_image,
    validate_private_ip,
    validate_public_ip,
    validate_run_id,
    validate_source_ranges,
)
from npa.workbench.leisaac.paidf import (
    export_episode_to_paidf,
    materialize_paidf_dataset,
)
from npa.workflows.sim2real.registry_auth import ensure_registry_pull_secret_for_images

app = typer.Typer(
    name="leisaac",
    help="LeIsaac SO101 browser teleoperation on the RTX PRO 6000 Kubernetes pool.",
    no_args_is_help=True,
)


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


class Transport(str, Enum):
    load_balancer = TRANSPORT_LOAD_BALANCER
    agent_relay = TRANSPORT_AGENT_RELAY


_RELAY_TOOL = "leisaac-relay"
_RELAY_CONFIG = "/etc/npa/leisaac-relay.json"
_RELAY_SCRIPT = "/opt/npa-agent/leisaac-agent-relay.py"
_RELAY_UNIT = "npa-leisaac-relay.service"
_RELAY_COTURN_RESTORE_MARKER = "/etc/npa/leisaac-relay.restore-coturn"
_TURN_CONTROL_TOOL = "leisaac-turn-control"
_TURN_TCP_TOOL = "leisaac-turn-control-tcp"
_TURN_MEDIA_TOOL = "leisaac-turn-media"
_TURN_CONFIG = "/etc/npa/leisaac-turn.conf"
_TURN_UNIT = "npa-leisaac-turn.service"
_EXTERNAL_IP_TIMEOUT_ENV = "NPA_LEISAAC_EXTERNAL_IP_TIMEOUT_SECONDS"
_READY_TIMEOUT_ENV = "NPA_LEISAAC_READY_TIMEOUT_SECONDS"
# The release operator explicitly requires bounded launch waits. These bound
# only CLI readiness observation (with rollback), never workload lifetime.
_DEFAULT_EXTERNAL_IP_TIMEOUT_SECONDS = 600.0
_DEFAULT_READY_TIMEOUT_SECONDS = 14_400.0
_LIFECYCLE_LOCK_STALE_SECONDS = 10 * 60
_LIFECYCLE_LOCK_RENEW_SECONDS = 30.0
_LIFECYCLE_LOCK_IO_TIMEOUT_SECONDS = 30.0
_LEISAAC_EULA_ENV = {"ACCEPT_EULA": "Y"}


def _fail(message: str) -> NoReturn:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(1)


def _require_isaac_consent() -> str:
    """Resolve the shared Isaac default or fail before any launch mutation."""

    try:
        return require_isaac_eula_acceptance(
            context="a LeIsaac GPU session",
            resume_command="npa workbench leisaac launch ...",
        )
    except MissingIsaacEulaAcceptanceError as exc:
        _fail(str(exc))


def _wait_timeout(environment_name: str, default: float) -> float:
    raw = str(os.environ.get(environment_name) or "").strip()
    try:
        value = float(raw) if raw else default
    except ValueError as exc:
        raise ValueError(
            f"{environment_name} must be a positive number of seconds"
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{environment_name} must be a finite number greater than 0")
    return value


def _kubectl_wait_diagnostic(result: KubectlResult) -> str:
    detail = " ".join((result.stderr or result.stdout or "no provider detail").split())
    return f"kubectl exit {result.returncode}: {detail[:400]}"


def _kubectl(
    context: str,
    namespace: str,
    args: list[str],
    stdin: str | None = None,
    *,
    timeout: float | None = None,
) -> KubectlResult:
    return run_kubectl(
        ["--namespace", namespace, *args],
        context=context,
        stdin=stdin,
        # Mutation calls were historically unbounded. Readiness observation is
        # bounded separately by _external_ip/_wait_ready with useful state.
        timeout=timeout,
    )


def _apply(context: str, namespace: str, documents: list[dict[str, Any]]) -> None:
    payload = json.dumps({"apiVersion": "v1", "kind": "List", "items": documents})
    result = _kubectl(context, namespace, ["apply", "-f", "-"], stdin=payload)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())


def _lifecycle_lock_name(deployment: str) -> str:
    """Return the shared lock name used by every same-run mutator."""

    digest = hashlib.sha256(deployment.encode("utf-8")).hexdigest()[:12]
    prefix = deployment[:34].rstrip("-") or "leisaac"
    return f"{prefix}-{digest}-lifecycle-lock"


def _lifecycle_lock_document(
    *,
    namespace: str,
    name: str,
    holder: str,
    resource_version: str,
    acquired_epoch: float,
    renewed_epoch: float,
    uid: str = "",
) -> dict[str, Any]:
    annotations = {
        "npa.nebius.com/lifecycle-holder": holder,
        "npa.nebius.com/lifecycle-acquired-epoch": f"{acquired_epoch:.6f}",
        "npa.nebius.com/lifecycle-renewed-epoch": f"{renewed_epoch:.6f}",
    }
    document: dict[str, Any] = {
        "apiVersion": "v1",
        # LeIsaac already creates, reads, updates, and deletes per-run Secrets.
        # Reuse that established RBAC surface instead of silently requiring
        # ConfigMap permissions from scoped operator roles.
        "kind": "Secret",
        "type": "Opaque",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "annotations": annotations,
        },
    }
    if resource_version:
        document["metadata"]["resourceVersion"] = resource_version
    if uid:
        document["metadata"]["uid"] = uid
    return document


def _lifecycle_lock_values(document: dict[str, Any]) -> dict[str, str]:
    annotations = (document.get("metadata") or {}).get("annotations") or {}
    return {
        "holder": str(annotations.get("npa.nebius.com/lifecycle-holder") or ""),
        "acquired_epoch": str(
            annotations.get("npa.nebius.com/lifecycle-acquired-epoch") or ""
        ),
        "renewed_epoch": str(
            annotations.get("npa.nebius.com/lifecycle-renewed-epoch") or ""
        ),
    }


def _lifecycle_lock_json(result: KubectlResult, label: str) -> dict[str, Any]:
    if result.returncode:
        detail = " ".join((result.stderr or result.stdout or "").split())
        raise RuntimeError(f"{label} failed: {detail[:500] or 'no provider detail'}")
    try:
        payload = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} returned a non-object document")
    return payload


def _acquire_lifecycle_lock(
    context: str,
    namespace: str,
    deployment: str,
    holder: str,
) -> str:
    """Atomically exclude every launch, destroy, and live proof for one run."""

    name = _lifecycle_lock_name(deployment)
    acquired_epoch = time.time()
    document = _lifecycle_lock_document(
        namespace=namespace,
        name=name,
        holder=holder,
        resource_version="",
        acquired_epoch=acquired_epoch,
        renewed_epoch=acquired_epoch,
    )
    result = _kubectl(
        context,
        namespace,
        ["create", "-f", "-"],
        stdin=json.dumps(document, sort_keys=True),
        timeout=_LIFECYCLE_LOCK_IO_TIMEOUT_SECONDS,
    )
    if not result.returncode:
        return name
    detail = " ".join((result.stderr or result.stdout or "").split()).lower()
    if "alreadyexists" not in detail and "already exists" not in detail:
        raise RuntimeError("could not acquire the selected run lifecycle lock")

    current = _lifecycle_lock_json(
        _kubectl(
            context,
            namespace,
            ["get", "secret", name, "-o", "json"],
            timeout=_LIFECYCLE_LOCK_IO_TIMEOUT_SECONDS,
        ),
        "existing lifecycle lock lookup",
    )
    metadata = current.get("metadata") or {}
    data = _lifecycle_lock_values(current)
    resource_version = str(metadata.get("resourceVersion") or "")
    try:
        prior_epoch = float(str(data.get("renewed_epoch") or ""))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("existing lifecycle lock has no valid renewal time") from exc
    age = acquired_epoch - prior_epoch
    if not resource_version or age < 0 or age <= _LIFECYCLE_LOCK_STALE_SECONDS:
        raise RuntimeError(
            "another LeIsaac lifecycle operation already holds the selected run lock"
        )

    replacement = _lifecycle_lock_document(
        namespace=namespace,
        name=name,
        holder=holder,
        resource_version=resource_version,
        acquired_epoch=acquired_epoch,
        renewed_epoch=acquired_epoch,
    )
    reclaimed = _kubectl(
        context,
        namespace,
        ["replace", "-f", "-"],
        stdin=json.dumps(replacement, sort_keys=True),
        timeout=_LIFECYCLE_LOCK_IO_TIMEOUT_SECONDS,
    )
    if reclaimed.returncode:
        raise RuntimeError("stale lifecycle lock changed before it could be reclaimed")
    return name


def _renew_lifecycle_lock(
    context: str,
    namespace: str,
    name: str,
    holder: str,
) -> None:
    current = _lifecycle_lock_json(
        _kubectl(
            context,
            namespace,
            ["get", "secret", name, "-o", "json"],
            timeout=_LIFECYCLE_LOCK_IO_TIMEOUT_SECONDS,
        ),
        "lifecycle lock renewal lookup",
    )
    metadata = current.get("metadata") or {}
    data = _lifecycle_lock_values(current)
    if str(data.get("holder") or "") != holder:
        raise RuntimeError("selected run lifecycle lock ownership changed")
    resource_version = str(metadata.get("resourceVersion") or "")
    try:
        acquired_epoch = float(str(data.get("acquired_epoch") or ""))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "selected run lifecycle lock has no valid acquisition time"
        ) from exc
    if not resource_version:
        raise RuntimeError("selected run lifecycle lock has no resource version")
    renewed = _kubectl(
        context,
        namespace,
        ["replace", "-f", "-"],
        stdin=json.dumps(
            _lifecycle_lock_document(
                namespace=namespace,
                name=name,
                holder=holder,
                resource_version=resource_version,
                acquired_epoch=acquired_epoch,
                renewed_epoch=time.time(),
            ),
            sort_keys=True,
        ),
        timeout=_LIFECYCLE_LOCK_IO_TIMEOUT_SECONDS,
    )
    if renewed.returncode:
        raise RuntimeError("could not renew the selected run lifecycle lock")


def _release_lifecycle_lock(
    context: str,
    namespace: str,
    name: str,
    holder: str,
) -> None:
    current = _lifecycle_lock_json(
        _kubectl(
            context,
            namespace,
            ["get", "secret", name, "-o", "json"],
            timeout=_LIFECYCLE_LOCK_IO_TIMEOUT_SECONDS,
        ),
        "lifecycle lock lookup",
    )
    if _lifecycle_lock_values(current)["holder"] != holder:
        raise RuntimeError("selected run lifecycle lock ownership changed")
    metadata = current.get("metadata") or {}
    resource_version = str(metadata.get("resourceVersion") or "")
    uid = str(metadata.get("uid") or "")
    if not resource_version or not uid:
        raise RuntimeError("selected run lifecycle lock has no release preconditions")
    # kubectl's ``delete --raw`` does not attach stdin as a DELETE body, so an
    # apparent DeleteOptions precondition would actually be ignored. First use
    # an atomic PUT to replace the owned Secret with a fresh quarantine marker.
    # A contender treats that marker as held, so the following ordinary delete
    # cannot race with a new owner. If deletion fails, the marker becomes
    # reclaimable after the normal stale interval rather than blocking forever.
    released_epoch = time.time()
    released = _kubectl(
        context,
        namespace,
        ["replace", "-f", "-"],
        stdin=json.dumps(
            _lifecycle_lock_document(
                namespace=namespace,
                name=name,
                holder="",
                resource_version=resource_version,
                acquired_epoch=released_epoch,
                renewed_epoch=released_epoch,
                uid=uid,
            ),
            sort_keys=True,
        ),
        timeout=_LIFECYCLE_LOCK_IO_TIMEOUT_SECONDS,
    )
    if released.returncode:
        raise RuntimeError("could not release the selected run lifecycle lock")
    deleted = _kubectl(
        context,
        namespace,
        ["delete", "secret", name, "--ignore-not-found=true"],
        timeout=_LIFECYCLE_LOCK_IO_TIMEOUT_SECONDS,
    )
    if deleted.returncode:
        raise RuntimeError("could not remove the released run lifecycle lock")


class _RunLifecycleLease:
    """Renewed Kubernetes lease shared by all same-run mutating operations."""

    def __init__(
        self,
        context: str,
        namespace: str,
        name: str,
        holder: str,
    ) -> None:
        self.context = context
        self.namespace = namespace
        self.name = name
        self.holder = holder
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._failure: Exception | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.wait(_LIFECYCLE_LOCK_RENEW_SECONDS):
            try:
                _renew_lifecycle_lock(
                    self.context, self.namespace, self.name, self.holder
                )
            except Exception as exc:  # noqa: BLE001 - latched for foreground
                with self._state_lock:
                    self._failure = exc
                return

    def assert_healthy(self) -> None:
        with self._state_lock:
            failure = self._failure
        if failure is not None:
            raise RuntimeError(
                "selected run lifecycle lock renewal failed"
            ) from failure

    def close(self) -> None:
        failures: list[str] = []
        self._stop_event.set()
        self._thread.join(_LIFECYCLE_LOCK_IO_TIMEOUT_SECONDS + 5.0)
        if self._thread.is_alive():
            failures.append("renewal did not stop")
        else:
            try:
                self.assert_healthy()
            except Exception as exc:  # noqa: BLE001 - preserve cleanup sequence
                failures.append(str(exc))
        try:
            _release_lifecycle_lock(
                self.context, self.namespace, self.name, self.holder
            )
        except Exception as exc:  # noqa: BLE001 - preserve cleanup sequence
            failures.append(str(exc))
        if failures:
            raise RuntimeError("lifecycle lock cleanup failed: " + "; ".join(failures))


def _require_lifecycle_lock_permissions(context: str, namespace: str) -> None:
    """Fail before mutation when the operator cannot maintain the shared lock."""

    missing: list[str] = []
    for verb in ("get", "create", "update", "delete"):
        result = _kubectl(
            context,
            namespace,
            ["auth", "can-i", verb, "secrets"],
            timeout=_LIFECYCLE_LOCK_IO_TIMEOUT_SECONDS,
        )
        if result.returncode or result.stdout.strip().lower() != "yes":
            missing.append(verb)
    if missing:
        raise RuntimeError(
            "LeIsaac lifecycle exclusion requires Kubernetes Secret permissions "
            f"get/create/update/delete in namespace {namespace!r}; missing or "
            f"unverified: {', '.join(missing)}. No LeIsaac resource was changed."
        )


def _acquire_run_lifecycle_lease(
    context: str,
    namespace: str,
    deployment: str,
) -> _RunLifecycleLease:
    _require_lifecycle_lock_permissions(context, namespace)
    holder = secrets.token_hex(16)
    name = _acquire_lifecycle_lock(context, namespace, deployment, holder)
    lease = _RunLifecycleLease(context, namespace, name, holder)
    lease.start()
    return lease


def _external_ip(
    context: str,
    namespace: str,
    service: str,
    *,
    timeout_seconds: float | None = None,
    poll_interval_seconds: float = 3.0,
) -> str:
    timeout = (
        _wait_timeout(_EXTERNAL_IP_TIMEOUT_ENV, _DEFAULT_EXTERNAL_IP_TIMEOUT_SECONDS)
        if timeout_seconds is None
        else timeout_seconds
    )
    started = time.monotonic()
    deadline = started + timeout
    last_diagnostic = "service has no assigned address"
    next_progress = started
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Timed out after {timeout:g}s waiting for Service "
                f"{namespace}/{service} external IPv4; last observation: "
                f"{last_diagnostic}"
            )
        result = _kubectl(
            context,
            namespace,
            [
                "get",
                "service",
                service,
                "-o",
                "jsonpath={.status.loadBalancer.ingress[0].ip}",
            ],
            timeout=min(30.0, remaining),
        )
        value = result.stdout.strip()
        assigned = result.returncode == 0 and bool(value)
        if assigned:
            last_diagnostic = "external IPv4 was assigned"
        else:
            last_diagnostic = (
                "service has no assigned address"
                if result.returncode == 0
                else _kubectl_wait_diagnostic(result)
            )
        now = time.monotonic()
        if now >= deadline:
            raise TimeoutError(
                f"Timed out after {timeout:g}s waiting for Service "
                f"{namespace}/{service} external IPv4; last observation: "
                f"{last_diagnostic}"
            )
        if assigned:
            return value
        if now >= next_progress:
            typer.echo(
                f"Waiting for Service {namespace}/{service} external IPv4; "
                f"{last_diagnostic}.",
                err=True,
            )
            next_progress = now + 60.0
        time.sleep(min(poll_interval_seconds, max(0.0, deadline - now)))


def _wait_ready(
    context: str,
    namespace: str,
    deployment: str,
    *,
    timeout_seconds: float | None = None,
    poll_interval_seconds: float = 5.0,
    progress_check: Callable[[], None] | None = None,
) -> None:
    timeout = (
        _wait_timeout(_READY_TIMEOUT_ENV, _DEFAULT_READY_TIMEOUT_SECONDS)
        if timeout_seconds is None
        else timeout_seconds
    )
    started = time.monotonic()
    deadline = started + timeout
    last_diagnostic = "deployment status has not been observed"
    next_progress = started
    while True:
        if progress_check is not None:
            progress_check()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Timed out after {timeout:g}s waiting for Deployment "
                f"{namespace}/{deployment} readiness; last observation: "
                f"{last_diagnostic}"
            )
        result = _kubectl(
            context,
            namespace,
            ["get", "deployment", deployment, "-o", "json"],
            timeout=min(30.0, remaining),
        )
        if result.returncode == 0:
            try:
                data = json.loads(result.stdout)
            except json.JSONDecodeError as exc:
                last_diagnostic = f"invalid deployment JSON: {exc}"
                data = {}
            metadata = data.get("metadata", {}) or {}
            spec = data.get("spec", {}) or {}
            status = data.get("status", {}) or {}
            generation = int(metadata.get("generation") or 0)
            ready = (
                generation > 0
                and int(status.get("observedGeneration") or 0) == generation
                and int(spec.get("replicas") or 0) == 1
                and int(status.get("updatedReplicas") or 0) == 1
                and int(status.get("readyReplicas") or 0) == 1
                and int(status.get("availableReplicas") or 0) == 1
                and int(status.get("unavailableReplicas") or 0) == 0
            )
            last_diagnostic = (
                "ready deployment observed"
                if ready
                else (
                    f"generation={generation}, "
                    f"observed={int(status.get('observedGeneration') or 0)}, "
                    f"desired={int(spec.get('replicas') or 0)}, "
                    f"updated={int(status.get('updatedReplicas') or 0)}, "
                    f"ready={int(status.get('readyReplicas') or 0)}, "
                    f"available={int(status.get('availableReplicas') or 0)}, "
                    f"unavailable={int(status.get('unavailableReplicas') or 0)}"
                )
            )
        else:
            ready = False
            last_diagnostic = _kubectl_wait_diagnostic(result)
        if progress_check is not None:
            progress_check()
        now = time.monotonic()
        if now >= deadline:
            raise TimeoutError(
                f"Timed out after {timeout:g}s waiting for Deployment "
                f"{namespace}/{deployment} readiness; last observation: "
                f"{last_diagnostic}"
            )
        if ready:
            return
        if now >= next_progress:
            typer.echo(
                f"Waiting for Deployment {namespace}/{deployment}; {last_diagnostic}.",
                err=True,
            )
            next_progress = now + 60.0
        time.sleep(min(poll_interval_seconds, max(0.0, deadline - now)))


def _delete_resources(context: str, namespace: str, name: str) -> None:
    result = _kubectl(
        context,
        namespace,
        [
            "delete",
            f"deployment/{name}",
            f"service/{name}-tcp",
            f"service/{name}-media",
            f"service/{name}-relay",
            f"secret/{name}-relay-client",
            f"secret/{name}-recorder",
            "--ignore-not-found=true",
        ],
    )
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())


def _relay_media_server(context: str, namespace: str, deployment: str) -> str:
    """Return the stable private Service IP for the simulator media peer."""

    result = _kubectl(
        context,
        namespace,
        ["get", "service", f"{deployment}-relay", "-o", "json"],
    )
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())
    service = json.loads(result.stdout)
    return validate_private_ip(
        str((service.get("spec") or {}).get("clusterIP") or ""),
        "LeIsaac private media Service address",
    )


def _node_internal_ip(context: str, namespace: str) -> str:
    result = _kubectl(context, namespace, ["get", "nodes", "-o", "json"])
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())
    candidates: list[str] = []
    for node in json.loads(result.stdout).get("items", []):
        labels = node.get("metadata", {}).get("labels", {}) or {}
        ready = any(
            condition.get("type") == "Ready" and condition.get("status") == "True"
            for condition in node.get("status", {}).get("conditions", []) or []
        )
        is_rtx6000 = (
            labels.get(GPU_PRODUCT_LABEL) == GPU_PRODUCT
            or labels.get(GPU_PROVIDER_LABEL) == GPU_PROVIDER_VALUE
        )
        if not ready or not is_rtx6000:
            continue
        for address in node.get("status", {}).get("addresses", []) or []:
            if address.get("type") == "InternalIP" and address.get("address"):
                candidates.append(str(address["address"]))
    if not candidates:
        raise RuntimeError(f"no Ready {GPU_PRODUCT} node with an internal IP was found")
    return sorted(set(candidates))[0]


def _relay_nodeports(context: str, namespace: str, service: str) -> dict[str, int]:
    result = _kubectl(context, namespace, ["get", "service", service, "-o", "json"])
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())
    ports = {
        str(item.get("name") or ""): int(item.get("nodePort") or 0)
        for item in json.loads(result.stdout).get("spec", {}).get("ports", [])
    }
    if set(ports) != {"status", "signal", "media"} or any(
        value < 30000 or value > 32767 for value in ports.values()
    ):
        raise RuntimeError("LeIsaac relay service did not receive valid NodePorts")
    return ports


def _agent_record(project: str, name: str) -> dict[str, Any]:
    project_record = list_projects().get(project, {})
    agents = (
        project_record.get("agents", {}) if isinstance(project_record, dict) else {}
    )
    record = agents.get(name, {}) if isinstance(agents, dict) else {}
    if not isinstance(record, dict) or not record:
        raise LeIsaacConfigError(f"agent config not found for {project}/{name}")
    return record


def _agent_artifact_storage(project: str, name: str) -> dict[str, str]:
    """Return the selected agent's S3 scope for capability publication.

    Agent-relay sessions must be published with the selected agent's endpoint
    and credentials.  Falling back to the operator shell's AWS endpoint can
    write a valid-looking manifest into a different regional S3 namespace,
    leaving the public agent unable to discover the capability it relays.
    """

    record = _agent_record(project, name)
    # Bootstrap keeps storage keys in the owner-only project credential store
    # instead of duplicating them into the config-backed agent record. Reuse
    # that resolver so current records and legacy embedded records both work.
    from npa.cli.agent import _resolve_agent_storage_credentials

    bucket, prefix, endpoint, access_key, secret_key, _service_account_id = (
        _resolve_agent_storage_credentials(project, record)
    )
    storage = {
        "bucket": str(bucket or "").strip(),
        "prefix": str(prefix or "").strip().strip("/"),
        "endpoint": str(endpoint or "").strip(),
        "access_key": str(access_key or "").strip(),
        "secret_key": str(secret_key or "").strip(),
        "region": str(record.get("region") or "").strip(),
    }
    missing = [
        key
        for key in ("bucket", "endpoint", "access_key", "secret_key")
        if not storage[key]
    ]
    if missing:
        raise LeIsaacConfigError(
            "agent record has no usable artifact storage configuration "
            f"(missing {', '.join(missing)})"
        )
    return storage


def _agent_relay_context(
    project: str, name: str
) -> tuple[str, str, SSHClient, str, str]:
    record = _agent_record(project, name)
    instance_id = str(record.get("instance_id") or "").strip()
    key_path = str(record.get("ssh_key_path") or "").strip()
    if not instance_id:
        raise LeIsaacConfigError("agent record has no provider instance id")
    if not key_path or not Path(key_path).expanduser().is_file():
        raise LeIsaacConfigError("agent record has no usable SSH private key")
    network = resolve_instance_network_context(instance_id)
    public_ip = validate_public_ip(
        str(network.public_ip).split("/", 1)[0], "agent public IP"
    )
    saved_ip = str(record.get("public_ip") or "").split("/", 1)[0]
    if saved_ip and saved_ip != public_ip:
        raise LeIsaacConfigError(
            "agent public IP differs from provider state; bootstrap the agent to refresh it"
        )
    auth_path = Path(str(record.get("auth_secret_path") or "")).expanduser()
    values: dict[str, str] = {}
    if auth_path.is_file():
        for line in auth_path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip()
    auth_user = values.get("AGENT_USER", "")
    auth_password = values.get("AGENT_PASSWORD", "")
    if not auth_user or not auth_password:
        raise LeIsaacConfigError("agent record has no usable basic-auth secret")
    ssh = SSHClient(
        SSHConfig(
            host=public_ip,
            user=str(record.get("ssh_user") or "ubuntu"),
            key_path=key_path,
        )
    )
    return instance_id, public_ip, ssh, auth_user, auth_password


def _agent_certificate_sha256(public_ip: str) -> str:
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    with socket.create_connection((public_ip, 443), timeout=10) as raw:
        with context.wrap_socket(raw, server_hostname=public_ip) as connection:
            certificate = connection.getpeercert(binary_form=True)
    if not certificate:
        raise RuntimeError("public agent HTTPS endpoint returned no certificate")
    return hashlib.sha256(certificate).hexdigest()


def _select_agent_leisaac_run(
    public_ip: str,
    *,
    auth_user: str,
    auth_password: str,
    run_id: str,
    certificate_sha256: str,
    timeout_seconds: float | None = None,
    progress_check: Callable[[], None] | None = None,
) -> None:
    """Register the live run through the selected agent's pinned TLS endpoint."""

    host = validate_public_ip(public_ip, "agent public IP")
    selected_run = validate_run_id(run_id)
    expected_certificate = str(certificate_sha256 or "").strip().lower()
    if len(expected_certificate) != 64:
        raise LeIsaacConfigError("agent certificate fingerprint is invalid")
    try:
        bytes.fromhex(expected_certificate)
    except ValueError as exc:
        raise LeIsaacConfigError("agent certificate fingerprint is invalid") from exc

    payload = json.dumps({"run_id": selected_run}, separators=(",", ":"))
    credential = base64.b64encode(
        f"{auth_user}:{auth_password}".encode("utf-8")
    ).decode("ascii")
    timeout = (
        _wait_timeout(_READY_TIMEOUT_ENV, _DEFAULT_READY_TIMEOUT_SECONDS)
        if timeout_seconds is None
        else timeout_seconds
    )
    deadline = time.monotonic() + timeout
    while True:
        if progress_check is not None:
            progress_check()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Timed out after {timeout:g}s waiting for agent run selection"
            )
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        raw = socket.create_connection((host, 443), timeout=min(10.0, remaining))
        tls = context.wrap_socket(raw, server_hostname=host)
        connection: http.client.HTTPConnection | None = None
        try:
            certificate = tls.getpeercert(binary_form=True)
            actual_certificate = (
                hashlib.sha256(certificate).hexdigest() if certificate else ""
            )
            if not secrets.compare_digest(actual_certificate, expected_certificate):
                raise RuntimeError("public agent TLS certificate fingerprint changed")
            # The TCP/TLS connection must fail fast, but selection performs capability
            # resolution plus a conditional state write against object storage. Preserve
            # the pinned socket while giving those off-loop operations their own response
            # budget instead of inheriting the 10-second connect timeout.
            tls.settimeout(min(60.0, remaining))
            connection = http.client.HTTPConnection(
                host, 443, timeout=min(10.0, remaining)
            )
            connection.sock = tls
            connection.request(
                "POST",
                "/api/leisaac/select",
                body=payload,
                headers={
                    "Authorization": f"Basic {credential}",
                    "Content-Type": "application/json",
                    "X-NPA-LeIsaac-Control": "1",
                },
            )
            response = connection.getresponse()
            body = response.read(131073)
            status = int(response.status)
        finally:
            if connection is not None:
                connection.close()
            else:
                tls.close()
        # Kubernetes readiness covers the simulator and relay-client processes,
        # while the reverse backhaul can still be completing its first connection.
        # A 503 is the agent's explicit transient-unavailable response. Keep the
        # live pod and retry with a freshly certificate-pinned HTTPS connection;
        # authentication and every other response class remain fail closed.
        if status == 503:
            if progress_check is not None:
                progress_check()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Timed out after {timeout:g}s waiting for agent run selection"
                )
            time.sleep(min(2.0, remaining))
            continue
        if status != 200 or len(body) > 131072:
            raise RuntimeError(
                f"public agent rejected LeIsaac run selection (HTTP {status})"
            )
        try:
            result = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise RuntimeError(
                "public agent returned an invalid LeIsaac selection response"
            ) from exc
        if (
            not isinstance(result, dict)
            or result.get("selected") is not True
            or result.get("run_id") != selected_run
        ):
            raise RuntimeError("public agent did not persist the LeIsaac run selection")
        return


def _relay_source(path: str) -> bytes:
    source = Path(__file__).resolve().parents[2] / "workbench" / "leisaac" / path
    return source.read_bytes()


def _install_agent_relay(
    ssh: SSHClient,
    *,
    run_id: str,
    session_nonce: str,
    expires_at: str,
    manifest_uri: str,
    media_target_host: str = "",
    media_target_port: int = 0,
) -> None:
    config: dict[str, Any] = {
        "run_id": run_id,
        "session_nonce": session_nonce,
        "expires_at": expires_at,
        "manifest_uri": manifest_uri,
    }
    if media_target_host or media_target_port:
        config["media_target_host"] = media_target_host
        config["media_target_port"] = media_target_port
    unit = """[Unit]
Description=NPA LeIsaac private-cluster relay
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
DynamicUser=yes
LoadCredential=leisaac.json:/etc/npa/leisaac-relay.json
ExecStart=/usr/bin/python3 /opt/npa-agent/leisaac-agent-relay.py --config ${CREDENTIALS_DIRECTORY}/leisaac.json
Restart=on-failure
RestartSec=2
NoNewPrivileges=yes
PrivateTmp=yes
ProtectHome=yes
ProtectSystem=strict
ProtectKernelTunables=yes
ProtectControlGroups=yes
RestrictSUIDSGID=yes

[Install]
WantedBy=multi-user.target
"""
    script_b64 = base64.b64encode(_relay_source("agent_relay.py")).decode("ascii")
    config_b64 = base64.b64encode(
        (json.dumps(config, sort_keys=True) + "\n").encode("utf-8")
    ).decode("ascii")
    unit_b64 = base64.b64encode(unit.encode("utf-8")).decode("ascii")
    run_q = shlex.quote(run_id)
    command = f"""set -eu
existing=''
if sudo test -f {_RELAY_CONFIG}; then
  existing=$(sudo /usr/bin/python3 -c 'import json; print(json.load(open("{_RELAY_CONFIG}"))["run_id"])')
fi
if sudo systemctl is-active --quiet {_RELAY_UNIT} && [ "$existing" != {run_q} ]; then
  echo 'another LeIsaac relay session is active' >&2
  exit 42
fi
marker_owner=''
if sudo test -f {_RELAY_COTURN_RESTORE_MARKER}; then
  marker_owner=$(sudo cat {_RELAY_COTURN_RESTORE_MARKER})
fi
if [ -n "$marker_owner" ] && [ "$marker_owner" != {run_q} ]; then
  echo 'another LeIsaac relay owns the baseline coturn handoff' >&2
  exit 43
fi
sudo install -d -m 0755 /etc/npa /opt/npa-agent
sudo tee /etc/sysctl.d/90-npa-leisaac-relay.conf >/dev/null <<'EOF'
net.core.rmem_max=8388608
net.core.wmem_max=8388608
net.core.netdev_max_backlog=5000
EOF
sudo sysctl --system >/dev/null
echo {shlex.quote(script_b64)} | base64 -d | sudo tee {_RELAY_SCRIPT} >/dev/null
echo {shlex.quote(config_b64)} | base64 -d | sudo tee {_RELAY_CONFIG} >/dev/null
echo {shlex.quote(unit_b64)} | base64 -d | sudo tee /etc/systemd/system/{_RELAY_UNIT} >/dev/null
sudo chmod 0644 {_RELAY_SCRIPT} /etc/systemd/system/{_RELAY_UNIT}
sudo chmod 0600 {_RELAY_CONFIG}
if sudo systemctl is-active --quiet coturn.service; then
  echo {run_q} | sudo tee {_RELAY_COTURN_RESTORE_MARKER} >/dev/null
  sudo chmod 0600 {_RELAY_COTURN_RESTORE_MARKER}
  sudo systemctl stop coturn.service
fi
sudo systemctl daemon-reload
sudo systemctl enable --now {_RELAY_UNIT} >/dev/null
sudo systemctl restart {_RELAY_UNIT}
"""
    ssh.run_or_raise(command, label="install LeIsaac agent relay")


def _remove_agent_relay(ssh: SSHClient, *, run_id: str) -> None:
    run_q = shlex.quote(run_id)
    command = f"""set -eu
existing=''
if sudo test -f {_RELAY_CONFIG}; then
  existing=$(sudo /usr/bin/python3 -c 'import json; print(json.load(open("{_RELAY_CONFIG}"))["run_id"])')
elif sudo test -f {_RELAY_COTURN_RESTORE_MARKER}; then
  existing=$(sudo cat {_RELAY_COTURN_RESTORE_MARKER})
fi
if [ "$existing" != {run_q} ]; then exit 0; fi
sudo systemctl disable --now {_RELAY_UNIT} >/dev/null 2>&1 || true
sudo rm -f /etc/systemd/system/{_RELAY_UNIT} {_RELAY_CONFIG} {_RELAY_SCRIPT}
sudo systemctl daemon-reload
if sudo test -f {_RELAY_COTURN_RESTORE_MARKER} && [ "$(sudo cat {_RELAY_COTURN_RESTORE_MARKER})" = {run_q} ]; then
  sudo systemctl start coturn.service
  sudo rm -f {_RELAY_COTURN_RESTORE_MARKER}
fi
"""
    ssh.run_or_raise(command, label="remove LeIsaac agent relay")


class _TransientRelayStatusError(RuntimeError):
    """A bounded relay probe failure that can resolve during normal startup."""


def _relay_status(
    ssh: SSHClient,
    *,
    session_nonce: str,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    if timeout_seconds <= 0:
        raise TimeoutError("LeIsaac relay status probe deadline expired")
    connect_timeout = min(5.0, timeout_seconds)
    try:
        code, stdout, _stderr = ssh.run(
            # A newly attached backhaul can accept the loopback stream before the
            # runtime has completed its own status response. Bound every probe so
            # _wait_relay_status retains control of its overall deadline and cleanup
            # can never wedge behind one half-open stream.
            "curl --silent --show-error "
            f"--connect-timeout {connect_timeout:g} --max-time {timeout_seconds:g} "
            "--write-out '\nNPA_HTTP_STATUS:%{http_code}\n' "
            f"-H {shlex.quote('X-NPA-LeIsaac-Nonce: ' + session_nonce)} "
            f"http://127.0.0.1:{RELAY_SERVICE_PORT}/status",
            timeout=timeout_seconds,
        )
    except SSHTimeoutError as exc:
        # The aggregate SSH watchdog starts before connection setup, so it can
        # expire before remote curl has a chance to report exit 28. This is the
        # same bounded transient observation and must remain inside the caller's
        # readiness retry loop.
        raise _TransientRelayStatusError(
            "LeIsaac relay status transport timed out"
        ) from exc
    if code:
        if code in {5, 6, 7, 18, 28, 52, 55, 56}:
            raise _TransientRelayStatusError(
                f"LeIsaac relay status transport failed with curl exit {code}"
            )
        raise RuntimeError(f"LeIsaac relay status probe failed with curl exit {code}")
    body, marker, raw_status = stdout.rpartition("\nNPA_HTTP_STATUS:")
    if not marker or not raw_status.strip().isdigit():
        raise RuntimeError("LeIsaac relay status probe returned no HTTP status")
    http_status = int(raw_status.strip())
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        if http_status in {429, 500, 502, 503, 504}:
            raise _TransientRelayStatusError(
                f"LeIsaac relay status returned transient HTTP {http_status}"
            ) from exc
        raise RuntimeError("LeIsaac relay returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("LeIsaac relay returned a non-object health document")
    if http_status != 200:
        if payload.get("state") == "failed":
            raise RuntimeError("LeIsaac runtime reported terminal state failed")
        if http_status in {429, 500, 502, 503, 504}:
            raise _TransientRelayStatusError(
                f"LeIsaac relay status returned transient HTTP {http_status}"
            )
        raise RuntimeError(f"LeIsaac relay status returned HTTP {http_status}")
    return payload


def _wait_relay_status(
    ssh: SSHClient,
    *,
    session_nonce: str,
    timeout_seconds: float,
    poll_interval_seconds: float = 2.0,
    progress_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Wait for a ready relay attestation within the launch deadline."""

    started = time.monotonic()
    deadline = started + timeout_seconds
    next_progress = started
    last_diagnostic = "relay status has not been observed"
    while True:
        if progress_check is not None:
            progress_check()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Timed out after {timeout_seconds:g}s waiting for LeIsaac relay "
                f"readiness; last observation: {last_diagnostic}"
            )
        try:
            status = _relay_status(
                ssh,
                session_nonce=session_nonce,
                timeout_seconds=min(10.0, remaining),
            )
        except _TransientRelayStatusError as exc:
            last_diagnostic = f"{type(exc).__name__}: {exc}"
        else:
            state = str(status.get("state") or "")
            now = time.monotonic()
            last_diagnostic = f"state={state or 'missing'}"
            if now >= deadline:
                raise TimeoutError(
                    f"Timed out after {timeout_seconds:g}s waiting for LeIsaac relay "
                    f"readiness; last observation: {last_diagnostic}"
                )
            if state == "ready":
                return status
            if state not in {"starting", "restarting"}:
                raise RuntimeError(
                    f"LeIsaac relay returned terminal or invalid state "
                    f"{state or 'missing'}"
                )
        now = time.monotonic()
        if now >= deadline:
            raise TimeoutError(
                f"Timed out after {timeout_seconds:g}s waiting for LeIsaac relay "
                f"readiness; last observation: {last_diagnostic}"
            )
        if now >= next_progress:
            typer.echo(
                f"Waiting for LeIsaac relay readiness; {last_diagnostic}.",
                err=True,
            )
            next_progress = now + 60.0
        time.sleep(min(poll_interval_seconds, max(0.0, deadline - now)))


def _turn_peer_source(value: str) -> str:
    ranges = validate_source_ranges([value])
    network = ipaddress.ip_network(ranges[0])
    if len(ranges) != 1 or network.version != 4 or network.prefixlen < 22:
        raise LeIsaacConfigError(
            "TURN peer source must be one public IPv4 CIDR between /22 and /32"
        )
    return ranges[0]


def _existing_turn_peer_source(context: str, namespace: str, service: str) -> str:
    result = _kubectl(context, namespace, ["get", "service", service, "-o", "json"])
    if result.returncode:
        return ""
    annotations = json.loads(result.stdout).get("metadata", {}).get("annotations", {})
    value = str((annotations or {}).get("npa.nebius.com/turn-peer-source") or "")
    return _turn_peer_source(value) if value else ""


def _remove_agent_turn(ssh: SSHClient, *, run_id: str) -> None:
    run_q = shlex.quote(validate_run_id(run_id))
    command = f"""set -eu
if ! sudo test -f {_TURN_CONFIG}; then exit 0; fi
existing=$(sudo sed -n 's/^user=\\([^:]*\\):.*$/\\1/p' {_TURN_CONFIG})
if [ "$existing" != {run_q} ]; then exit 0; fi
sudo systemctl disable --now {_TURN_UNIT} >/dev/null 2>&1 || true
sudo rm -f /etc/systemd/system/{_TURN_UNIT} {_TURN_CONFIG}
sudo systemctl daemon-reload
"""
    ssh.run_or_raise(command, label="remove LeIsaac TURN relay")


def _put_manifest(
    uri: str,
    manifest: dict[str, Any],
    *,
    storage: dict[str, str] | None = None,
) -> str:
    import boto3

    bucket, prefix = split_s3_uri(uri)
    client_kwargs: dict[str, Any] = {}
    if storage is not None:
        configured_bucket = str(storage.get("bucket") or "").strip()
        configured_prefix = str(storage.get("prefix") or "").strip().strip("/")
        if bucket != configured_bucket:
            raise LeIsaacConfigError(
                "agent-relay artifact URI bucket must match the selected agent's bucket"
            )
        if configured_prefix and not (
            prefix == configured_prefix or prefix.startswith(configured_prefix + "/")
        ):
            raise LeIsaacConfigError(
                "agent-relay artifact URI must be inside the selected agent's artifact prefix"
            )
        client_kwargs = {
            "endpoint_url": storage["endpoint"],
            "aws_access_key_id": storage["access_key"],
            "region_name": storage.get("region") or None,
        }
        client_kwargs["aws" + "_secret_access_key"] = storage["secret_key"]
    manifest_uri = _manifest_object_uri(uri, str(manifest["run_id"]))
    _manifest_bucket, key = split_s3_uri(manifest_uri)
    if storage is None:
        client_kwargs["endpoint_url"] = (
            os.environ.get("NEBIUS_S3_ENDPOINT")
            or os.environ.get("AWS_ENDPOINT_URL")
            or None
        )
    client = boto3.client("s3", **client_kwargs)
    public_manifest = dict(manifest)
    public_manifest.pop("session_nonce", None)
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=(json.dumps(public_manifest, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
        ContentType="application/json",
    )
    return manifest_uri


def _load_manifest(
    uri: str,
    *,
    run_id: str,
    storage: dict[str, str],
) -> dict[str, Any]:
    """Load one exact agent-owned session manifest without broad discovery."""

    import boto3

    bucket, key = split_s3_uri(uri)
    configured_bucket = str(storage.get("bucket") or "").strip()
    configured_prefix = str(storage.get("prefix") or "").strip().strip("/")
    if bucket != configured_bucket:
        raise LeIsaacConfigError(
            "agent-relay manifest URI bucket must match the selected agent's bucket"
        )
    if configured_prefix and not (
        key == configured_prefix or key.startswith(configured_prefix + "/")
    ):
        raise LeIsaacConfigError(
            "agent-relay manifest URI must be inside the selected agent's artifact prefix"
        )
    if _manifest_object_uri(uri, run_id) != uri:
        raise LeIsaacConfigError(
            "agent-relay reconnect requires the exact reports/leisaac-session.json URI"
        )
    client_kwargs: dict[str, Any] = {
        "endpoint_url": storage["endpoint"],
        "aws_access_key_id": storage["access_key"],
        "region_name": storage.get("region") or None,
    }
    client_kwargs["aws" + "_secret_access_key"] = storage["secret_key"]
    response = boto3.client("s3", **client_kwargs).get_object(Bucket=bucket, Key=key)
    if int(response.get("ContentLength") or 0) > 1_048_576:
        raise LeIsaacConfigError("agent-relay session manifest exceeds 1 MiB")
    body = response["Body"].read(1_048_577)
    if len(body) > 1_048_576:
        raise LeIsaacConfigError("agent-relay session manifest exceeds 1 MiB")
    try:
        manifest = json.loads(body)
    except (TypeError, UnicodeDecodeError, ValueError) as exc:
        raise LeIsaacConfigError(
            "agent-relay session manifest is invalid JSON"
        ) from exc
    if not isinstance(manifest, dict):
        raise LeIsaacConfigError("agent-relay session manifest is not an object")
    if manifest.get("schema") != SESSION_SCHEMA or manifest.get("run_id") != run_id:
        raise LeIsaacConfigError(
            "agent-relay session manifest does not match the selected live run"
        )
    if (
        manifest.get("transport") != TRANSPORT_AGENT_RELAY
        or manifest.get("signal_host") != "127.0.0.1"
    ):
        raise LeIsaacConfigError(
            "agent-relay session manifest does not use the secure private transport"
        )
    return manifest


def _manifest_object_uri(prefix_uri: str, run_id: str) -> str:
    """Return the exact immutable manifest object for one validated run."""

    bucket, prefix = split_s3_uri(prefix_uri)
    run_id = validate_run_id(run_id)
    leaf = "reports/leisaac-session.json"
    # A deprecated launch accepted a leaf URI but treated it as a prefix,
    # producing .../leisaac-session.json/<run>/reports/leisaac-session.json.
    # Honor leaf semantics for new writes while discovery continues to find
    # historical objects by their canonical basename.
    key = (
        prefix.rstrip("/")
        if prefix.rstrip("/").endswith(leaf)
        else f"{prefix.rstrip('/')}/{run_id}/{leaf}"
    )
    return f"s3://{bucket}/{key}"


def _emit(payload: dict[str, Any], output: OutputFormat) -> None:
    if output == OutputFormat.json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    for key, value in payload.items():
        typer.echo(f"{key}: {value}")


def _existing_relay_contract(
    context: str,
    namespace: str,
    *,
    run_id: str,
    agent_project: str,
    agent_name: str,
) -> tuple[str, list[str]]:
    """Validate one live, NPA-owned relay workload before credential rotation."""

    name = resource_name(validate_run_id(run_id))

    def resource(kind: str, resource_name: str) -> dict[str, Any]:
        result = _kubectl(
            context,
            namespace,
            ["get", kind, resource_name, "-o", "json"],
        )
        if result.returncode:
            raise LeIsaacConfigError(
                f"existing LeIsaac {kind}/{resource_name} is unavailable"
            )
        try:
            payload = json.loads(result.stdout)
        except ValueError as exc:
            raise LeIsaacConfigError(
                f"existing LeIsaac {kind}/{resource_name} returned invalid JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise LeIsaacConfigError(
                f"existing LeIsaac {kind}/{resource_name} is invalid"
            )
        metadata = payload.get("metadata") or {}
        labels = metadata.get("labels") or {}
        if (
            metadata.get("name") != resource_name
            or metadata.get("namespace") != namespace
            or labels.get("app.kubernetes.io/name") != "leisaac"
            or labels.get("app.kubernetes.io/instance") != name
            or labels.get("app.kubernetes.io/managed-by") != "npa"
        ):
            raise LeIsaacConfigError(
                f"existing {kind}/{resource_name} is not owned by the selected LeIsaac run"
            )
        return payload

    service = resource("service", f"{name}-relay")
    deployment = resource("deployment", name)
    annotations = (service.get("metadata") or {}).get("annotations") or {}
    if (
        annotations.get("npa.nebius.com/agent-project") != agent_project
        or annotations.get("npa.nebius.com/agent-name") != agent_name
    ):
        raise LeIsaacConfigError(
            "existing LeIsaac relay belongs to a different agent identity"
        )
    source_ranges = validate_source_ranges(
        str(annotations.get("npa.nebius.com/source-ranges") or "").split(",")
    )
    if not source_ranges:
        raise LeIsaacConfigError("existing LeIsaac relay has no operator source range")
    service_spec = service.get("spec") or {}
    if service_spec.get("type") != "ClusterIP":
        raise LeIsaacConfigError(
            "existing LeIsaac relay is not private; refusing credential rotation"
        )
    media_server = validate_private_ip(
        str(service_spec.get("clusterIP") or ""),
        "LeIsaac private media Service address",
    )

    deployment_spec = deployment.get("spec") or {}
    pod_spec = (deployment_spec.get("template") or {}).get("spec") or {}
    containers = pod_spec.get("containers") or []
    container_names = {
        str(item.get("name") or "") for item in containers if isinstance(item, dict)
    }
    if int(deployment_spec.get("replicas") or 0) != 1 or not {
        "leisaac",
        "agent-relay-client",
        "turn",
    }.issubset(container_names):
        raise LeIsaacConfigError(
            "existing LeIsaac Deployment does not match the secure single-run topology"
        )
    main_container = next(
        item
        for item in containers
        if isinstance(item, dict) and item.get("name") == "leisaac"
    )
    environment_items = [
        item for item in (main_container.get("env") or []) if isinstance(item, dict)
    ]
    environment = {
        str(item.get("name") or ""): str(item.get("value") or "")
        for item in environment_items
        if "value" in item
    }
    if environment.get("NPA_LEISAAC_RUN_ID") != run_id:
        raise LeIsaacConfigError("existing LeIsaac Deployment run identity changed")
    if any(environment.get(key) != value for key, value in _LEISAAC_EULA_ENV.items()):
        raise LeIsaacConfigError(
            "existing LeIsaac Deployment does not retain its exact run-scoped EULA values"
        )
    nonce_items = [
        item
        for item in environment_items
        if item.get("name") == "NPA_LEISAAC_SESSION_NONCE"
    ]
    if len(nonce_items) != 1:
        raise LeIsaacConfigError(
            "existing LeIsaac Deployment has no valid relay credential"
        )
    nonce_item = nonce_items[0]
    nonce = str(nonce_item.get("value") or "")
    value_from = nonce_item.get("valueFrom") or {}
    nonce_reference = (
        value_from.get("secretKeyRef") or {} if isinstance(value_from, dict) else {}
    )
    valid_literal = bool(re.fullmatch(r"[a-f0-9]{64}", nonce))
    valid_reference = nonce_reference == {
        "name": f"{name}-relay-client",
        "key": "NPA_LEISAAC_SESSION_NONCE",
    }
    if not (valid_literal or valid_reference):
        raise LeIsaacConfigError(
            "existing LeIsaac Deployment has no valid relay credential"
        )
    relay_volume = next(
        (
            item
            for item in (pod_spec.get("volumes") or [])
            if isinstance(item, dict) and item.get("name") == "relay-client"
        ),
        {},
    )
    if ((relay_volume.get("secret") or {}).get("secretName")) != f"{name}-relay-client":
        raise LeIsaacConfigError(
            "existing LeIsaac Deployment does not mount its run-scoped relay Secret"
        )
    return media_server, source_ranges


@app.command("list-tasks")
def list_tasks_cmd(
    output: OutputFormat = typer.Option(OutputFormat.text, "--output"),
) -> None:
    """List the pinned SO101 tasks that support browser keyboard control."""

    payload = registry_payload()
    if output == OutputFormat.json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    for task in payload["tasks"]:
        typer.echo(f"{task['task']}: {task['description']}")
    typer.echo(
        "environment_model: named-sequential (one active environment per launch)"
    )


@app.command("export-paidf")
def export_paidf_cmd(
    dataset_uri: str = typer.Option(..., "--dataset-uri"),
    episode: int = typer.Option(..., "--episode", min=0),
    run_id: str = typer.Option(..., "--run-id"),
    output_path: str = typer.Option(..., "--output-path"),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output"),
) -> None:
    """Export one finalized episode directly from S3 into a PAIDF run input."""

    try:
        run_id = validate_run_id(run_id)
        result = export_episode_to_paidf(
            dataset_uri=dataset_uri,
            episode_index=episode,
            paidf_run_id=run_id,
            paidf_output_path=output_path,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        _fail(str(exc))
        return
    _emit(result, output)


@app.command("materialize-paidf")
def materialize_paidf_cmd(
    dataset_uri: str = typer.Option(..., "--dataset-uri"),
    episode: int = typer.Option(..., "--episode", min=0),
    paidf_run_uri: str = typer.Option(..., "--paidf-run-uri"),
    output_path: str = typer.Option(..., "--output-path"),
    variant: int = typer.Option(0, "--variant", min=0),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output"),
) -> None:
    """Create an immutable derived dataset after strict PAIDF video alignment."""

    try:
        result = materialize_paidf_dataset(
            dataset_uri=dataset_uri,
            episode_index=episode,
            paidf_run_uri=paidf_run_uri,
            output_path=output_path,
            variant_index=variant,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        _fail(str(exc))
        return
    _emit(result, output)


@app.command("launch")
def launch_cmd(
    run_id: str = typer.Option(
        ..., "--run-id", help="Run id used for artifact discovery and UI selection."
    ),
    image: str = typer.Option(
        ..., "--image", help="npa-leisaac image pinned as repository@sha256:digest."
    ),
    context: str = typer.Option(
        "", "--context", help="kubectl context for the RT-core GPU cluster."
    ),
    namespace: str = typer.Option(
        "default", "--namespace", help="Kubernetes namespace."
    ),
    source_range: list[str] = typer.Option(
        ...,
        "--source-range",
        help="Public operator CIDR allowed to reach the session; repeat when needed.",
    ),
    output_path: str = typer.Option(
        ...,
        "--output-path",
        help="Operator-owned S3 prefix for immutable LeRobot demonstration datasets.",
    ),
    manifest_prefix: str = typer.Option(
        "",
        "--manifest-prefix",
        help="S3 prefix (or exact .../reports/leisaac-session.json leaf) for capability publication.",
    ),
    artifact_uri: str = typer.Option(
        "",
        "--artifact-uri",
        help="Deprecated alias for --manifest-prefix; exact leaf URIs retain leaf semantics.",
    ),
    task: str = typer.Option(
        DEFAULT_TASK, "--task", help="Pinned task returned by list-tasks."
    ),
    environment_id: str = typer.Option(
        DEFAULT_ENVIRONMENT_ID,
        "--environment-id",
        help="Stable sequential environment identity.",
    ),
    environment_index: int = typer.Option(
        0,
        "--environment-index",
        min=0,
        max=2**31 - 1,
        help="Stable non-negative environment index.",
    ),
    seed: int = typer.Option(42, "--seed", min=0, max=2**32 - 1),
    num_envs: int = typer.Option(
        1,
        "--num-envs",
        help="Must be 1; browser control and episode boundaries are not parallel-routed.",
    ),
    expires_at: str = typer.Option(
        "",
        "--expires-at",
        help="Optional operator-chosen ISO-8601 expiry for UI discovery; omitted sessions remain live until destroyed.",
    ),
    image_pull_secret: str = typer.Option(
        "npa-registry", "--image-pull-secret", help="Existing registry pull secret."
    ),
    transport: Transport = typer.Option(
        Transport.agent_relay,
        "--transport",
        help="Secure agent relay. The historical public-load-balancer value is rejected.",
    ),
    agent_project: str = typer.Option(
        "", "--agent-project", help="Saved NPA agent project alias for agent-relay."
    ),
    agent_name: str = typer.Option(
        "", "--agent-name", help="Saved NPA agent deployment name for agent-relay."
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", help="Output format."
    ),
) -> None:
    """Launch a supported SO101 task and publish its secure collector capability."""

    if transport == Transport.load_balancer:
        _fail(
            "public-load-balancer is unsupported because its S3 discovery "
            "manifest cannot securely provision browser credentials; use agent-relay"
        )
    accept_eula = _require_isaac_consent()
    name = ""
    instance_id = ""
    ssh: SSHClient | None = None
    artifact_storage: dict[str, str] | None = None
    auth_user = ""
    auth_password = ""
    certificate_sha256 = ""
    relay_installed = False
    turn_cleanup_required = False
    prior_turn_peer_source = ""
    created_ingress_specs: list[tuple[int, str, str, str]] = []
    resources_mutated = False
    lifecycle_lease: _RunLifecycleLease | None = None
    failure_message = ""
    lifecycle_warning = ""
    try:
        run_id = validate_run_id(run_id)
        image = validate_image(image)
        task = validate_task(task)
        environment_id = validate_environment_id(environment_id)
        environment_index = validate_environment_index(environment_index)
        seed = validate_seed(seed)
        num_envs = validate_num_envs(num_envs)
        expires_at = validate_expiry(expires_at)
        source_ranges = validate_source_ranges(source_range)
        launch_timeout = _wait_timeout(
            _READY_TIMEOUT_ENV, _DEFAULT_READY_TIMEOUT_SECONDS
        )
        split_s3_uri(output_path)
        if manifest_prefix and artifact_uri:
            raise LeIsaacConfigError(
                "use --manifest-prefix or deprecated --artifact-uri, not both"
            )
        resolved_manifest_prefix = manifest_prefix or artifact_uri
        if not resolved_manifest_prefix:
            raise LeIsaacConfigError("--manifest-prefix is required")
        split_s3_uri(resolved_manifest_prefix)
        exact_manifest_uri = _manifest_object_uri(resolved_manifest_prefix, run_id)
        name = resource_name(run_id)
        lifecycle_lease = _acquire_run_lifecycle_lease(context, namespace, name)
        nonce = secrets.token_hex(32)
        if image_pull_secret:
            # Nebius IAM-backed registry credentials are intentionally short lived. Refresh
            # the named secret for every launch so a pod scheduled onto a cold GPU node does
            # not depend on an old node cache or somebody else's credential-refresh cycle.
            ensure_registry_pull_secret_for_images(
                image,
                secret_name=image_pull_secret,
                namespace=namespace,
                k8s_context=context,
            )
            secret = _kubectl(
                context, namespace, ["get", "secret", image_pull_secret, "-o", "name"]
            )
            if secret.returncode:
                raise LeIsaacConfigError(
                    f"image pull secret {image_pull_secret!r} is missing in namespace {namespace!r}"
                )
        if not agent_project or not agent_name:
            raise LeIsaacConfigError(
                "agent-relay requires --agent-project and --agent-name"
            )
        instance_id, media_host, ssh, auth_user, auth_password = _agent_relay_context(
            agent_project, agent_name
        )
        turn_cleanup_required = True
        artifact_storage = _agent_artifact_storage(agent_project, agent_name)
        dataset_bucket, dataset_prefix = split_s3_uri(output_path)
        if dataset_bucket != artifact_storage["bucket"]:
            raise LeIsaacConfigError(
                "agent-relay output path bucket must match the selected agent's bucket"
            )
        agent_prefix = artifact_storage["prefix"]
        if agent_prefix and not (
            dataset_prefix == agent_prefix
            or dataset_prefix.startswith(agent_prefix + "/")
        ):
            raise LeIsaacConfigError(
                "agent-relay output path must be inside the selected agent's artifact prefix"
            )
        prior_turn_peer_source = _existing_turn_peer_source(
            context, namespace, f"{name}-relay"
        )
        service = relay_service_manifest(
            run_id=run_id,
            namespace=namespace,
            agent_project=agent_project,
            agent_name=agent_name,
            source_ranges=source_ranges,
        )
        lifecycle_lease.assert_healthy()
        resources_mutated = True
        _apply(context, namespace, [service])
        relay_installed = True
        # The ClusterIP is allocated when the relay Service is created and
        # remains stable for this launch transaction. Resolve it before
        # rendering the Deployment so the relay topology is validated
        # before any GPU workload is scheduled.
        media_server = _relay_media_server(context, namespace, name)
        # Sessions from the previous topology ran coturn on the public agent
        # itself. Remove only this run's matching unit before the backhaul
        # relay takes ownership of public UDP 3478.
        _remove_agent_turn(ssh, run_id=run_id)
        _install_agent_relay(
            ssh,
            run_id=run_id,
            session_nonce=nonce,
            expires_at=expires_at,
            manifest_uri=exact_manifest_uri,
        )
        certificate_sha256 = _agent_certificate_sha256(media_host)
        relay_secret = relay_client_secret_manifest(
            run_id=run_id,
            namespace=namespace,
            agent_host=media_host,
            session_nonce=nonce,
            certificate_sha256=certificate_sha256,
            auth_user=auth_user,
            auth_password=auth_password,
            client_source=_relay_source("reverse_client.py").decode("utf-8"),
        )
        _apply(context, namespace, [relay_secret])
        signal_host = "127.0.0.1"
        if artifact_storage is None:
            raise LeIsaacConfigError("recorder storage configuration is unavailable")
        recorder_secret = recorder_secret_manifest(
            run_id=run_id,
            namespace=namespace,
            output_path=output_path,
            endpoint=artifact_storage["endpoint"],
            access_key=artifact_storage["access_key"],
            secret_key=artifact_storage["secret_key"],
            region=artifact_storage["region"],
        )
        _apply(context, namespace, [recorder_secret])
        deployment = deployment_manifest(
            run_id=run_id,
            namespace=namespace,
            image=image,
            media_host=media_host,
            session_nonce=nonce,
            media_server=media_server,
            image_pull_secret=image_pull_secret,
            relay_client_secret=f"{name}-relay-client",
            recorder_secret=f"{name}-recorder",
            task=task,
            environment_id=environment_id,
            environment_index=environment_index,
            seed=seed,
            num_envs=num_envs,
            accept_eula=accept_eula,
        )
        _apply(context, namespace, [deployment])
        launch_deadline = time.monotonic() + launch_timeout
        _wait_ready(
            context,
            namespace,
            name,
            timeout_seconds=launch_timeout,
            progress_check=lifecycle_lease.assert_healthy,
        )
        if ssh is None:
            raise RuntimeError("LeIsaac agent relay has no SSH transport")
        for source in source_ranges:
            for protocol in ("UDP", "TCP"):
                ingress_tool = (
                    _TURN_CONTROL_TOOL if protocol == "UDP" else _TURN_TCP_TOOL
                )
                ingress = ensure_ingress(
                    vm_id=instance_id,
                    ports=(TURN_PORT,),
                    source=source,
                    tool=ingress_tool,
                    protocol=protocol,
                )
                if ingress.changed:
                    created_ingress_specs.append(
                        (TURN_PORT, source, ingress_tool, protocol)
                    )
        if prior_turn_peer_source:
            remove_exact_npa_ingress_for_instance(
                instance_id,
                ports=(TURN_RELAY_PORT,),
                source=prior_turn_peer_source,
                tool=_TURN_MEDIA_TOOL,
                protocol="UDP",
            )
        health = _wait_relay_status(
            ssh,
            session_nonce=nonce,
            timeout_seconds=max(0.0, launch_deadline - time.monotonic()),
            progress_check=lifecycle_lease.assert_healthy,
        )
        if (
            health.get("state") != "ready"
            or health.get("task") != task
            or health.get("source_commit") != SOURCE_COMMIT
            or health.get("task_registry_fingerprint") != REGISTRY_FINGERPRINT
            or health.get("session_attestation") != session_attestation(nonce)
            or health.get("environment_id") != environment_id
            or int(health.get("environment_index", -1)) != environment_index
            or int(health.get("seed", -1)) != seed
        ):
            raise RuntimeError(f"LeIsaac live attestation failed: {health}")
        manifest = session_manifest(
            run_id=run_id,
            image=image,
            signal_host=signal_host,
            media_host=media_host,
            media_server=media_server,
            session_nonce=nonce,
            expires_at=expires_at,
            gpu=str(health.get("gpu") or GPU_PRODUCT),
            created_at=str(health.get("started_at") or "") or None,
            transport=transport.value,
            task=task,
            environment_id=environment_id,
            environment_index=environment_index,
            seed=seed,
            num_envs=num_envs,
            output_path=output_path,
        )
        manifest_uri = _put_manifest(
            resolved_manifest_prefix,
            manifest,
            storage=artifact_storage,
        )
        _select_agent_leisaac_run(
            media_host,
            auth_user=auth_user,
            auth_password=auth_password,
            run_id=run_id,
            certificate_sha256=certificate_sha256,
            timeout_seconds=max(0.0, launch_deadline - time.monotonic()),
            progress_check=lifecycle_lease.assert_healthy,
        )
        lifecycle_lease.assert_healthy()
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - CLI boundary
        cleanup_errors: list[str] = []
        if turn_cleanup_required and ssh is not None:
            try:
                _remove_agent_turn(ssh, run_id=run_id)
            except Exception as cleanup_exc:  # noqa: BLE001 - preserve primary failure
                cleanup_errors.append(f"TURN cleanup: {cleanup_exc}")
        if relay_installed and ssh is not None:
            try:
                _remove_agent_relay(ssh, run_id=run_id)
            except Exception as cleanup_exc:  # noqa: BLE001 - preserve primary failure
                cleanup_errors.append(f"relay cleanup: {cleanup_exc}")
        for port, source, tool, protocol in created_ingress_specs:
            try:
                remove_exact_npa_ingress_for_instance(
                    instance_id,
                    ports=(port,),
                    source=source,
                    tool=tool,
                    protocol=protocol,
                )
            except Exception as cleanup_exc:  # noqa: BLE001 - preserve primary failure
                cleanup_errors.append(f"ingress cleanup: {cleanup_exc}")
        if resources_mutated:
            try:
                _delete_resources(context, namespace, name)
            except Exception as cleanup_exc:  # noqa: BLE001 - preserve primary failure
                cleanup_errors.append(f"Kubernetes cleanup: {cleanup_exc}")
        primary_error = str(exc) or "LeIsaac launch interrupted"
        failure_message = primary_error
        if cleanup_errors:
            failure_message = (
                f"{primary_error}; cleanup also failed: {'; '.join(cleanup_errors)}"
            )
    finally:
        # A normal error, Ctrl-C, or embedded caller cancellation must never
        # leave the renewal thread running or the selected run locked.
        if lifecycle_lease is not None:
            try:
                lifecycle_lease.close()
                lifecycle_lease = None
            except Exception as cleanup_exc:  # noqa: BLE001 - preserve primary failure
                lock_error = f"lifecycle lock cleanup failed: {cleanup_exc}"
                if failure_message:
                    failure_message = f"{failure_message}; {lock_error}"
                else:
                    # The deployment, manifest publication, and authenticated
                    # agent selection have already succeeded. A best-effort
                    # lock-release failure must be visible, but reporting the
                    # live workload as a failed launch invites a destructive
                    # retry against resources that are serving traffic.
                    lifecycle_warning = lock_error
    if failure_message:
        _fail(failure_message)
    result = {
        "status": "ready",
        "run_id": run_id,
        "task": task,
        "environment_id": environment_id,
        "environment_index": environment_index,
        "seed": seed,
        "dataset": output_path.rstrip("/"),
        "gpu": health.get("gpu"),
        "image": image,
        "transport": transport.value,
        "transport_security": "secure-agent-relay",
        "deployment": name,
        "signal_host": signal_host,
        "media_host": media_host,
        "artifact": manifest_uri,
        "public_agent_url": f"https://{media_host}/",
        "expires_at": expires_at or "none (service lifecycle)",
    }
    if lifecycle_warning:
        result["warning"] = lifecycle_warning
    _emit(result, output)


@app.command("reconnect-agent")
def reconnect_agent_cmd(
    run_id: str = typer.Option(..., "--run-id"),
    manifest_uri: str = typer.Option(
        ...,
        "--manifest-uri",
        help="Exact existing .../reports/leisaac-session.json object for this run.",
    ),
    agent_project: str = typer.Option(
        ..., "--agent-project", help="Saved replacement NPA agent project alias."
    ),
    agent_name: str = typer.Option(
        ..., "--agent-name", help="Saved replacement NPA agent deployment name."
    ),
    context: str = typer.Option("", "--context"),
    namespace: str = typer.Option("default", "--namespace"),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output"),
) -> None:
    """Reconnect one existing private LeIsaac run to a replacement agent.

    This rotates only the run-scoped relay credential and Deployment nonce. It
    never creates a new LeIsaac Deployment, changes the task/dataset/image, or
    records a new EULA acceptance.
    """

    lifecycle_lease: _RunLifecycleLease | None = None
    failure_message = ""
    lifecycle_warning = ""
    try:
        run_id = validate_run_id(run_id)
        split_s3_uri(manifest_uri)
        reconnect_timeout = _wait_timeout(
            _READY_TIMEOUT_ENV, _DEFAULT_READY_TIMEOUT_SECONDS
        )
        name = resource_name(run_id)
        lifecycle_lease = _acquire_run_lifecycle_lease(context, namespace, name)
        media_server, source_ranges = _existing_relay_contract(
            context,
            namespace,
            run_id=run_id,
            agent_project=agent_project,
            agent_name=agent_name,
        )
        lifecycle_lease.assert_healthy()
        storage = _agent_artifact_storage(agent_project, agent_name)
        manifest = _load_manifest(
            manifest_uri,
            run_id=run_id,
            storage=storage,
        )
        if manifest.get("media_server") != media_server:
            raise LeIsaacConfigError(
                "session manifest does not match the existing private relay Service"
            )
        instance_id, public_ip, ssh, auth_user, auth_password = _agent_relay_context(
            agent_project, agent_name
        )
        certificate_sha256 = _agent_certificate_sha256(public_ip)
        nonce = secrets.token_hex(32)
        for source in source_ranges:
            for protocol in ("UDP", "TCP"):
                ensure_ingress(
                    vm_id=instance_id,
                    ports=(TURN_PORT,),
                    source=source,
                    tool=(_TURN_CONTROL_TOOL if protocol == "UDP" else _TURN_TCP_TOOL),
                    protocol=protocol,
                )
        lifecycle_lease.assert_healthy()
        _install_agent_relay(
            ssh,
            run_id=run_id,
            session_nonce=nonce,
            expires_at=str(manifest.get("expires_at") or ""),
            manifest_uri=manifest_uri,
        )
        relay_secret = relay_client_secret_manifest(
            run_id=run_id,
            namespace=namespace,
            agent_host=public_ip,
            session_nonce=nonce,
            certificate_sha256=certificate_sha256,
            auth_user=auth_user,
            auth_password=auth_password,
            client_source=_relay_source("reverse_client.py").decode("utf-8"),
        )
        _apply(context, namespace, [relay_secret])
        lifecycle_lease.assert_healthy()
        # Reference the rotated Secret instead of putting the nonce in argv,
        # shell history, or the operator host's process table. The non-secret
        # annotation always changes so a second reconnect restarts the pod even
        # when the valueFrom contract is already installed.
        rotation_patch = {
            "spec": {
                # Preserved runs may predate the explicit Recreate strategy.
                # Change the strategy in the same patch as the pod template so
                # a one-GPU Deployment never deadlocks waiting for surge capacity.
                "strategy": {"type": "Recreate", "rollingUpdate": None},
                "template": {
                    "metadata": {
                        "annotations": {
                            "npa.nebius.com/relay-rotation": str(time.time_ns())
                        }
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": "leisaac",
                                "env": [
                                    {
                                        "name": "NPA_LEISAAC_SESSION_NONCE",
                                        "value": None,
                                        "valueFrom": {
                                            "secretKeyRef": {
                                                "name": f"{name}-relay-client",
                                                "key": "NPA_LEISAAC_SESSION_NONCE",
                                            }
                                        },
                                    }
                                ],
                            }
                        ]
                    },
                },
            }
        }
        rotated = _kubectl(
            context,
            namespace,
            [
                "patch",
                f"deployment/{name}",
                "--type=strategic",
                "--patch",
                json.dumps(rotation_patch, separators=(",", ":")),
            ],
        )
        if rotated.returncode:
            raise RuntimeError("failed to rotate the existing LeIsaac relay credential")
        reconnect_deadline = time.monotonic() + reconnect_timeout
        _wait_ready(
            context,
            namespace,
            name,
            timeout_seconds=reconnect_timeout,
            progress_check=lifecycle_lease.assert_healthy,
        )
        health = _wait_relay_status(
            ssh,
            session_nonce=nonce,
            timeout_seconds=max(0.0, reconnect_deadline - time.monotonic()),
            progress_check=lifecycle_lease.assert_healthy,
        )
        if (
            health.get("state") != "ready"
            or health.get("task") != manifest.get("task")
            or health.get("source_commit") != manifest.get("source_commit")
            or health.get("task_registry_fingerprint")
            != manifest.get("task_registry_fingerprint")
            or health.get("session_attestation") != session_attestation(nonce)
        ):
            raise RuntimeError(
                "reconnected LeIsaac relay attestation does not match the run"
            )
        updated_manifest = dict(manifest)
        updated_manifest["media_host"] = public_ip
        updated_manifest["session_nonce"] = nonce
        updated_manifest["session_attestation"] = session_attestation(nonce)
        _put_manifest(manifest_uri, updated_manifest, storage=storage)
        _select_agent_leisaac_run(
            public_ip,
            auth_user=auth_user,
            auth_password=auth_password,
            run_id=run_id,
            certificate_sha256=certificate_sha256,
            timeout_seconds=max(0.0, reconnect_deadline - time.monotonic()),
            progress_check=lifecycle_lease.assert_healthy,
        )
        lifecycle_lease.assert_healthy()
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - CLI boundary
        failure_message = str(exc) or "LeIsaac agent reconnect interrupted"
    finally:
        if lifecycle_lease is not None:
            try:
                lifecycle_lease.close()
            except Exception as cleanup_exc:  # noqa: BLE001 - preserve primary failure
                lock_error = f"lifecycle lock cleanup failed: {cleanup_exc}"
                if failure_message:
                    failure_message = f"{failure_message}; {lock_error}"
                else:
                    lifecycle_warning = lock_error
    if failure_message:
        _fail(failure_message)
    result = {
        "status": "reconnected",
        "run_id": run_id,
        "transport": TRANSPORT_AGENT_RELAY,
        "deployment": resource_name(run_id),
        "public_agent_url": f"https://{public_ip}/",
    }
    if lifecycle_warning:
        result["warning"] = lifecycle_warning
    _emit(result, output)


@app.command("status")
def status_cmd(
    run_id: str = typer.Option(..., "--run-id"),
    context: str = typer.Option("", "--context"),
    namespace: str = typer.Option("default", "--namespace"),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output"),
) -> None:
    """Report the live Kubernetes objects for a LeIsaac run."""

    try:
        name = resource_name(validate_run_id(run_id))
    except LeIsaacConfigError as exc:
        _fail(str(exc))
        return
    result = _kubectl(
        context,
        namespace,
        [
            "get",
            "deployment,service,pod",
            "-l",
            f"app.kubernetes.io/instance={name}",
            "-o",
            "json",
        ],
    )
    if result.returncode:
        _fail((result.stderr or result.stdout).strip())
    data = json.loads(result.stdout)
    _emit({"run_id": run_id, "resources": data.get("items", [])}, output)


@app.command("destroy")
def destroy_cmd(
    run_id: str = typer.Option(..., "--run-id"),
    context: str = typer.Option("", "--context"),
    namespace: str = typer.Option("default", "--namespace"),
) -> None:
    """Delete this run's transient GPU deployment and LBs, preserving S3 evidence."""

    try:
        name = resource_name(validate_run_id(run_id))
    except LeIsaacConfigError as exc:
        _fail(str(exc))
        return
    lifecycle_lease: _RunLifecycleLease | None = None
    failure: Exception | None = None
    failure_stage = "lifecycle lock acquisition"
    try:
        lifecycle_lease = _acquire_run_lifecycle_lease(context, namespace, name)
        failure_stage = "agent relay cleanup"
        relay = _kubectl(
            context, namespace, ["get", "service", f"{name}-relay", "-o", "json"]
        )
        if relay.returncode == 0:
            annotations = (
                json.loads(relay.stdout).get("metadata", {}).get("annotations", {})
                or {}
            )
            project = str(annotations.get("npa.nebius.com/agent-project") or "")
            agent_name = str(annotations.get("npa.nebius.com/agent-name") or "")
            sources = validate_source_ranges(
                str(annotations.get("npa.nebius.com/source-ranges") or "").split(",")
            )
            peer_source = str(
                annotations.get("npa.nebius.com/turn-peer-source") or ""
            ).strip()
            instance_id, _public_ip, ssh, _auth_user, _auth_password = (
                _agent_relay_context(project, agent_name)
            )
            _remove_agent_turn(ssh, run_id=run_id)
            _remove_agent_relay(ssh, run_id=run_id)
            ingress_specs = [
                (
                    TURN_PORT,
                    source,
                    _TURN_CONTROL_TOOL if protocol == "UDP" else _TURN_TCP_TOOL,
                    protocol,
                )
                for source in sources
                for protocol in ("UDP", "TCP")
            ]
            if peer_source:
                validated_peer = _turn_peer_source(peer_source)
                ingress_specs.append(
                    (TURN_RELAY_PORT, validated_peer, _TURN_MEDIA_TOOL, "UDP")
                )
            else:
                # Compatibility cleanup for sessions launched before TURN support.
                ingress_specs.extend(
                    (MEDIA_PORT, source, _RELAY_TOOL, "UDP") for source in sources
                )
            for port, source, tool, protocol in ingress_specs:
                lifecycle_lease.assert_healthy()
                remove_exact_npa_ingress_for_instance(
                    instance_id,
                    ports=(port,),
                    source=source,
                    tool=tool,
                    protocol=protocol,
                )
        lifecycle_lease.assert_healthy()
        failure_stage = "Kubernetes cleanup"
        _delete_resources(context, namespace, name)
    except Exception as exc:  # noqa: BLE001 - CLI cleanup boundary
        if failure_stage == "agent relay cleanup":
            failure = RuntimeError(
                f"agent relay cleanup failed; Kubernetes resources retained: {exc}"
            )
        else:
            failure = exc
    finally:
        if lifecycle_lease is not None:
            try:
                lifecycle_lease.close()
            except Exception as lock_exc:  # noqa: BLE001 - preserve primary failure
                if failure is None:
                    failure = lock_exc
                else:
                    failure = RuntimeError(
                        f"{failure}; lifecycle lock cleanup also failed: {lock_exc}"
                    )
    if failure is not None:
        _fail(str(failure))
    typer.echo("transient LeIsaac Kubernetes and relay resources removed")
