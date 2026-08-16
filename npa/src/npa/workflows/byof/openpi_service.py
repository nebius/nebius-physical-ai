"""Cross-pod Kubernetes serving gate for the pinned upstream OpenPI server."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shlex
import time
from typing import Any, Mapping, Sequence
import urllib.request

from npa.workflows.byof.openpi import (
    OPENPI_TERMS_ACCEPTED_VALUE,
    OPENPI_TERMS_ENV,
)
from npa.workflows.byof.openpi_pipeline import (
    ACTION_DIM,
    DEFAULT_CHECKPOINT_URI,
    DEFAULT_CONFIG_NAME,
    RUNTIME_IMAGE_RE,
    SOURCE_LICENSE,
    SOURCE_REF,
    _checkpoint_provenance,
    _redistribution_evidence,
    _write_json_uri,
)

MANAGED_BY = "npa-openpi-four-mode"
SERVER_PORT = 8000
LABELS = {
    "app.kubernetes.io/managed-by": "npa",
    "app.kubernetes.io/part-of": "openpi-four-mode",
}
CONTROLLER_MANAGED_BY = "npa-openpi-service-controller"


class OpenPIServiceError(RuntimeError):
    """Raised when the cross-pod OpenPI service contract fails."""


def _safe_name(run_id: str) -> str:
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:12]
    return f"npa-openpi-{digest}"


def controller_service_account_name(run_id: str) -> str:
    """Return the run-scoped ServiceAccount used by the Sky control task."""

    return f"{_safe_name(run_id)}-ctl"


def build_controller_rbac_manifests(
    *, run_id: str, namespace: str, service_account: str
) -> dict[str, dict[str, Any]]:
    """Build least-privilege, exact-identity RBAC for the service control task."""

    if not re.fullmatch(r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?", service_account):
        raise OpenPIServiceError(
            f"service account must be a Kubernetes DNS label: {service_account!r}"
        )
    if len(service_account) > 63:
        raise OpenPIServiceError(
            "service account exceeds the 63-character DNS-label limit"
        )
    owner_hash = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:16]
    metadata: dict[str, Any] = {
        "name": service_account,
        "namespace": namespace,
        "labels": {
            "app.kubernetes.io/managed-by": CONTROLLER_MANAGED_BY,
            "npa.nebius.ai/run-hash": owner_hash,
        },
        "annotations": {"npa.nebius.ai/cleanup-owner": run_id},
    }

    def fresh_metadata() -> dict[str, Any]:
        return {
            **metadata,
            "labels": dict(metadata["labels"]),
            "annotations": dict(metadata["annotations"]),
        }

    role_rules = [
        {
            "apiGroups": [""],
            "resources": ["pods"],
            "verbs": ["get", "list", "delete"],
        },
        {"apiGroups": [""], "resources": ["pods/log"], "verbs": ["get"]},
        {
            "apiGroups": [""],
            "resources": ["services", "secrets"],
            "verbs": ["create", "get", "delete"],
        },
        {
            "apiGroups": ["apps"],
            "resources": ["deployments"],
            "verbs": ["create", "get", "delete"],
        },
        {
            "apiGroups": ["batch"],
            "resources": ["jobs"],
            "verbs": ["create", "get", "delete"],
        },
    ]
    return {
        "service_account": {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": fresh_metadata(),
        },
        "role": {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "Role",
            "metadata": fresh_metadata(),
            "rules": role_rules,
        },
        "role_binding": {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "RoleBinding",
            "metadata": fresh_metadata(),
            "subjects": [
                {
                    "kind": "ServiceAccount",
                    "name": service_account,
                    "namespace": namespace,
                }
            ],
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "Role",
                "name": service_account,
            },
        },
    }


def _run_labels(run_id: str, name: str) -> dict[str, str]:
    return {
        **LABELS,
        "app.kubernetes.io/name": "openpi-policy",
        "npa.nebius.ai/run-hash": hashlib.sha256(run_id.encode("utf-8")).hexdigest()[
            :16
        ],
        "npa.nebius.ai/cleanup-owner": name,
    }


def _client_program() -> str:
    """Program executed in a separate CPU-only client pod."""

    return r"""
import json
import os
import time
import urllib.request

import numpy as np
from openpi_client import websocket_client_policy

host = os.environ["OPENPI_SERVICE_HOST"]
port = int(os.environ.get("OPENPI_SERVICE_PORT", "8000"))
frame = np.arange(224 * 224 * 3, dtype=np.uint32).reshape(224, 224, 3)
observation = {
    "observation/exterior_image_1_left": np.asarray(frame % 251, dtype=np.uint8),
    "observation/wrist_image_left": np.asarray(np.flip(frame, axis=1) % 253, dtype=np.uint8),
    "observation/joint_position": np.asarray(
        [0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398],
        dtype=np.float32,
    ),
    "observation/gripper_position": np.asarray([0.04], dtype=np.float32),
    "prompt": "pick up the fork",
}
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(f"http://{host}:{port}/healthz") as response:
    health = response.read().decode("utf-8").strip()
client = websocket_client_policy.WebsocketClientPolicy(host, port)
requests = []
for index in range(2):
    started = time.perf_counter()
    response = client.infer(observation)
    actions = np.asarray(response["actions"])
    if actions.ndim != 2 or actions.shape[0] < 5 or actions.shape[1] != 8:
        raise RuntimeError(f"served trajectory {index} has invalid shape {actions.shape}")
    if actions.dtype != np.dtype("float64") or not np.isfinite(actions).all():
        raise RuntimeError(f"served trajectory {index} violates dtype/finiteness")
    requests.append({
        "request_index": index,
        "shape": list(actions.shape),
        "dtype": str(actions.dtype),
        "finite": True,
        "round_trip_ms": round((time.perf_counter() - started) * 1000, 3),
        "server_infer_ms": round(float(response["server_timing"]["infer_ms"]), 3),
        "first_five_targets": actions[:5].tolist(),
    })
print("NPA_OPENPI_CLIENT_RESULT=" + json.dumps({
    "schema": "npa.workbench.openpi.cross-pod-client.v1",
    "status": "passed",
    "healthz": health,
    "request_count": len(requests),
    "requests": requests,
}, sort_keys=True), flush=True)
""".strip()


def _server_hardware_program() -> str:
    """Self-contained GPU proof for a server image that does not embed NPA."""

    return r"""
import importlib.metadata
import json
import os
import re
import subprocess

import jax
from jax.extend import backend as jax_backend

expected_type = os.environ["OPENPI_EXPECTED_GPU_TYPE"]
expected_count = int(os.environ["OPENPI_EXPECTED_GPU_COUNT"])
expected_capability = os.environ["OPENPI_EXPECTED_COMPUTE_CAPABILITY"]
devices = jax.devices()
if len(devices) != expected_count:
    raise RuntimeError(f"expected {expected_count} visible GPU(s), got {len(devices)}")
kinds = [str(getattr(device, "device_kind", "")) for device in devices]
if expected_type and any(expected_type.upper() not in item.upper() for item in kinds):
    raise RuntimeError(f"expected GPU type {expected_type!r}, got {kinds!r}")
capabilities = []
for device in devices:
    raw = getattr(device, "compute_capability", "")
    raw = raw() if callable(raw) else raw
    major = getattr(raw, "major", None)
    minor = getattr(raw, "minor", None)
    if major is not None and minor is not None:
        value = f"{int(major)}.{int(minor)}"
    else:
        match = re.search(r"(\d{1,2})[.,](\d)", str(raw))
        if match:
            value = f"{int(match.group(1))}.{int(match.group(2))}"
        else:
            compact = re.search(r"(?:sm_)?(\d{2,3})", str(raw), re.IGNORECASE)
            if not compact:
                raise RuntimeError(f"cannot parse compute capability {raw!r}")
            number = int(compact.group(1))
            value = f"{number // 10}.{number % 10}"
    capabilities.append(value)
if expected_capability and any(value != expected_capability for value in capabilities):
    raise RuntimeError(
        f"expected compute capability {expected_capability}, got {capabilities}"
    )
sm100_probe = {"required": expected_capability == "10.0"}
if sm100_probe["required"]:
    probe = "/usr/local/bin/npa-openpi-sm100-probe"
    output = subprocess.run(
        [probe], check=True, capture_output=True, text=True
    ).stdout.strip()
    elf = subprocess.run(
        ["cuobjdump", "--list-elf", probe],
        check=True,
        capture_output=True,
        text=True,
    )
    if "sm_100" not in (elf.stdout + elf.stderr).lower():
        raise RuntimeError("CUDA probe has no sm_100 ELF")
    sm100_probe.update({"passed": True, "output": output, "elf_contains_sm100": True})
nvidia_smi = subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total",
        "--format=csv,noheader",
    ],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip().splitlines()
if len(nvidia_smi) != expected_count:
    raise RuntimeError("nvidia-smi visible GPU count disagrees with JAX")
result = {
    "gpu_count_allocated": len(devices),
    "device_kinds": kinds,
    "compute_capabilities": capabilities,
    "nvidia_smi": nvidia_smi,
    "jax": jax.__version__,
    "jaxlib": importlib.metadata.version("jaxlib"),
    "xla_platform_version": str(jax_backend.get_backend().platform_version),
    "sm100_probe": sm100_probe,
}
print("NPA_OPENPI_SERVER_HARDWARE=" + json.dumps(result, sort_keys=True), flush=True)
""".strip()


def build_manifests(
    *,
    run_id: str,
    namespace: str,
    runtime_image: str,
    checkpoint_uri: str,
    config_name: str,
    gpu_count: int,
    expected_gpu_type: str,
    expected_compute_capability: str,
    server_cpu: str,
    server_memory: str,
    client_cpu: str,
    client_memory: str,
    pull_secret: str,
    liveness_initial_delay_seconds: int,
    gpu_node_selector_key: str,
    gpu_node_selector_value: str,
    cache_size: str,
) -> dict[str, dict[str, Any]]:
    """Build the exact Secret/Deployment/ClusterIP/Job objects as testable data."""

    if not RUNTIME_IMAGE_RE.fullmatch(runtime_image):
        raise OpenPIServiceError("service runtime image must be digest-pinned")
    if gpu_count < 1:
        raise OpenPIServiceError("OpenPI policy server requires at least one GPU")
    name = _safe_name(run_id)
    labels = _run_labels(run_id, name)
    selector = {"npa.nebius.ai/cleanup-owner": name, "app": name}
    pod_labels = {**labels, **selector}
    secret_name = f"{name}-terms"
    service_name = f"{name}-policy"
    client_name = f"{name}-client"
    pull_secrets = [{"name": pull_secret}] if pull_secret else []
    hardware_probe = "/opt/venv/bin/python -c " + shlex.quote(
        _server_hardware_program()
    )
    server_shell = (
        'set -euo pipefail; test "$NPA_OPENPI_ACCEPT_GEMMA_TERMS" = "YES" || exit 64; '
        f"{hardware_probe}; "
        "checkpoint_dir=$(/opt/venv/bin/python -c 'import os; "
        "from openpi.shared import download; "
        'print(download.maybe_download(os.environ["OPENPI_CHECKPOINT_URI"], token="anon"))\'); '
        "exec /opt/venv/bin/python /opt/byof/scripts/serve_policy.py "
        f"--port={SERVER_PORT} policy:checkpoint "
        f'--policy.config={config_name} --policy.dir="$checkpoint_dir"'
    )
    secret = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": secret_name, "namespace": namespace, "labels": labels},
        "type": "Opaque",
        "data": {
            OPENPI_TERMS_ENV: base64.b64encode(
                OPENPI_TERMS_ACCEPTED_VALUE.encode("utf-8")
            ).decode("ascii")
        },
    }
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": name, "namespace": namespace, "labels": labels},
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": selector},
            "template": {
                "metadata": {"labels": pod_labels},
                "spec": {
                    "restartPolicy": "Always",
                    "imagePullSecrets": pull_secrets,
                    "nodeSelector": (
                        {gpu_node_selector_key: gpu_node_selector_value}
                        if gpu_node_selector_key and gpu_node_selector_value
                        else {}
                    ),
                    "containers": [
                        {
                            "name": "openpi-policy",
                            "image": runtime_image,
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["/bin/bash", "-lc", server_shell],
                            "ports": [
                                {"name": "websocket", "containerPort": SERVER_PORT}
                            ],
                            "env": [
                                {
                                    "name": OPENPI_TERMS_ENV,
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": secret_name,
                                            "key": OPENPI_TERMS_ENV,
                                        }
                                    },
                                },
                                {
                                    "name": "OPENPI_DATA_HOME",
                                    "value": "/workspace/openpi-server-cache",
                                },
                                {
                                    "name": "OPENPI_CHECKPOINT_URI",
                                    "value": checkpoint_uri,
                                },
                                {
                                    "name": "OPENPI_EXPECTED_GPU_TYPE",
                                    "value": expected_gpu_type,
                                },
                                {
                                    "name": "OPENPI_EXPECTED_GPU_COUNT",
                                    "value": str(gpu_count),
                                },
                                {
                                    "name": "OPENPI_EXPECTED_COMPUTE_CAPABILITY",
                                    "value": expected_compute_capability,
                                },
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": server_cpu,
                                    "memory": server_memory,
                                    "nvidia.com/gpu": str(gpu_count),
                                },
                                "limits": {"nvidia.com/gpu": str(gpu_count)},
                            },
                            "volumeMounts": [
                                {
                                    "name": "openpi-cache",
                                    "mountPath": "/workspace/openpi-server-cache",
                                }
                            ],
                            "readinessProbe": {
                                "httpGet": {"path": "/healthz", "port": SERVER_PORT},
                                "periodSeconds": 5,
                                "failureThreshold": 240,
                            },
                            "livenessProbe": {
                                "httpGet": {"path": "/healthz", "port": SERVER_PORT},
                                "initialDelaySeconds": liveness_initial_delay_seconds,
                                "periodSeconds": 30,
                                "failureThreshold": 6,
                            },
                        }
                    ],
                    "volumes": [
                        {
                            "name": "openpi-cache",
                            "emptyDir": {"sizeLimit": cache_size},
                        }
                    ],
                },
            },
        },
    }
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": service_name, "namespace": namespace, "labels": labels},
        "spec": {
            "type": "ClusterIP",
            "selector": selector,
            "ports": [
                {"name": "websocket", "port": SERVER_PORT, "targetPort": SERVER_PORT}
            ],
        },
    }
    client_job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": client_name, "namespace": namespace, "labels": labels},
        "spec": {
            "backoffLimit": 0,
            "template": {
                "metadata": {"labels": {**labels, "app": client_name}},
                "spec": {
                    "restartPolicy": "Never",
                    "imagePullSecrets": pull_secrets,
                    "containers": [
                        {
                            "name": "openpi-client",
                            "image": runtime_image,
                            "imagePullPolicy": "IfNotPresent",
                            "command": [
                                "/opt/venv/bin/python",
                                "-c",
                                _client_program(),
                            ],
                            "env": [
                                {"name": "OPENPI_SERVICE_HOST", "value": service_name},
                                {
                                    "name": "OPENPI_SERVICE_PORT",
                                    "value": str(SERVER_PORT),
                                },
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": client_cpu,
                                    "memory": client_memory,
                                },
                            },
                        }
                    ],
                },
            },
        },
    }
    return {
        "secret": secret,
        "deployment": deployment,
        "service": service,
        "client_job": client_job,
    }


def _api_exception_status(exc: Exception) -> int | None:
    return getattr(exc, "status", None)


def _create_server_objects(
    api: Any,
    apps: Any,
    manifests: Mapping[str, dict[str, Any]],
    *,
    created: set[str] | None = None,
) -> None:
    tracker = created if created is not None else set()
    namespace = manifests["secret"]["metadata"]["namespace"]
    api.create_namespaced_secret(namespace, manifests["secret"])
    tracker.add("secret")
    apps.create_namespaced_deployment(namespace, manifests["deployment"])
    tracker.add("deployment")
    api.create_namespaced_service(namespace, manifests["service"])
    tracker.add("service")


def _create_client_job(
    batch: Any,
    manifests: Mapping[str, dict[str, Any]],
    *,
    created: set[str] | None = None,
) -> None:
    tracker = created if created is not None else set()
    namespace = manifests["client_job"]["metadata"]["namespace"]
    batch.create_namespaced_job(namespace, manifests["client_job"])
    tracker.add("client_job")


def _assert_targets_absent(
    api: Any,
    apps: Any,
    batch: Any,
    *,
    namespace: str,
    names: Mapping[str, str],
) -> None:
    """Fail closed before create if any deterministic target name already exists."""

    readers = (
        ("client_job", batch.read_namespaced_job),
        ("deployment", apps.read_namespaced_deployment),
        ("service", api.read_namespaced_service),
        ("secret", api.read_namespaced_secret),
    )
    for key, reader in readers:
        try:
            reader(names[key], namespace)
        except Exception as exc:
            if _api_exception_status(exc) == 404:
                continue
            raise
        raise OpenPIServiceError(
            f"refusing to create or clean pre-existing {key} {names[key]!r}; "
            "run ownership is not proven"
        )
    pod_selector = f"npa.nebius.ai/cleanup-owner={names['deployment']}"
    pods = api.list_namespaced_pod(namespace, label_selector=pod_selector).items
    if pods:
        raise OpenPIServiceError(
            "refusing to create or clean pre-existing pods with the deterministic "
            f"run selector: {[pod.metadata.name for pod in pods]}"
        )


def _pod_failure_message(api: Any, namespace: str, pod: Any) -> str:
    name = pod.metadata.name
    try:
        logs = api.read_namespaced_pod_log(name, namespace, tail_lines=200)
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        logs = f"logs unavailable: {exc}"
    statuses = []
    for status in pod.status.container_statuses or []:
        statuses.append(str(status.state))
    return f"pod {name} failed: statuses={statuses}; logs={logs[-8000:]}"


def _wait_server_ready(api: Any, apps: Any, namespace: str, name: str) -> Any:
    selector = f"npa.nebius.ai/cleanup-owner={name},app={name}"
    while True:
        deployment = apps.read_namespaced_deployment(name, namespace)
        if int(deployment.status.available_replicas or 0) >= 1:
            pods = api.list_namespaced_pod(namespace, label_selector=selector).items
            ready = [
                pod
                for pod in pods
                if pod.status.phase == "Running"
                and any(
                    condition.type == "Ready" and condition.status == "True"
                    for condition in pod.status.conditions or []
                )
            ]
            if len(ready) == 1:
                return ready[0]
        pods = api.list_namespaced_pod(namespace, label_selector=selector).items
        for pod in pods:
            if pod.status.phase == "Failed":
                raise OpenPIServiceError(_pod_failure_message(api, namespace, pod))
        time.sleep(5)


def _wait_client(api: Any, batch: Any, namespace: str, name: str) -> Any:
    while True:
        job = batch.read_namespaced_job(name, namespace)
        pods = api.list_namespaced_pod(
            namespace, label_selector=f"job-name={name}"
        ).items
        if int(job.status.succeeded or 0) == 1 and len(pods) == 1:
            return pods[0]
        if int(job.status.failed or 0) > 0:
            if pods:
                raise OpenPIServiceError(_pod_failure_message(api, namespace, pods[0]))
            raise OpenPIServiceError(f"client job {name} failed without a pod")
        time.sleep(3)


def _prefixed_json(text: str, prefix: str) -> dict[str, object]:
    for line in text.splitlines():
        if prefix not in line:
            continue
        payload = line.split(prefix, 1)[1].strip()
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise OpenPIServiceError(
                f"invalid {prefix.rstrip('=')} evidence: {payload[-2000:]}"
            ) from exc
        if isinstance(value, dict):
            return value
    raise OpenPIServiceError(f"server logs contain no {prefix.rstrip('=')} evidence")


def _delete_and_verify(
    api: Any,
    apps: Any,
    batch: Any,
    *,
    namespace: str,
    names: Mapping[str, str],
    created: set[str] | None = None,
) -> dict[str, bool]:
    created_keys = set(names) if created is None else set(created)
    pod_selector = f"npa.nebius.ai/cleanup-owner={names['deployment']}"
    owned_pod_names: list[str] = []
    for pod in api.list_namespaced_pod(namespace, label_selector=pod_selector).items:
        labels = pod.metadata.labels or {}
        if (
            labels.get("app.kubernetes.io/managed-by") != "npa"
            or labels.get("app.kubernetes.io/part-of") != "openpi-four-mode"
            or labels.get("npa.nebius.ai/cleanup-owner") != names["deployment"]
        ):
            raise OpenPIServiceError(
                f"refusing cleanup because pod {pod.metadata.name!r} does not "
                "prove exact OpenPI service ownership"
            )
        pod_app = labels.get("app")
        if pod_app == names["deployment"] and "deployment" not in created_keys:
            raise OpenPIServiceError(
                f"refusing cleanup because server pod {pod.metadata.name!r} has no "
                "controller-created Deployment"
            )
        if pod_app == names["client_job"] and "client_job" not in created_keys:
            raise OpenPIServiceError(
                f"refusing cleanup because client pod {pod.metadata.name!r} has no "
                "controller-created Job"
            )
        if pod_app not in {names["deployment"], names["client_job"]}:
            raise OpenPIServiceError(
                f"refusing cleanup because pod {pod.metadata.name!r} has an "
                f"unexpected app identity {pod_app!r}"
            )
        owned_pod_names.append(str(pod.metadata.name))
    delete_options = {
        "apiVersion": "v1",
        "kind": "DeleteOptions",
        "propagationPolicy": "Foreground",
    }
    operations = (
        ("client_job", batch.delete_namespaced_job, names["client_job"]),
        ("deployment", apps.delete_namespaced_deployment, names["deployment"]),
        ("service", api.delete_namespaced_service, names["service"]),
        ("secret", api.delete_namespaced_secret, names["secret"]),
    )
    for key, function, name in operations:
        if key not in created_keys:
            continue
        try:
            function(name, namespace, body=delete_options)
        except TypeError:
            function(name, namespace, propagation_policy="Foreground")
        except Exception as exc:
            if _api_exception_status(exc) != 404:
                raise
    readers = (
        ("client_job", batch.read_namespaced_job),
        ("deployment", apps.read_namespaced_deployment),
        ("service", api.read_namespaced_service),
        ("secret", api.read_namespaced_secret),
    )
    verified: dict[str, bool] = {}
    for key, reader in readers:
        if key not in created_keys:
            try:
                reader(names[key], namespace)
            except Exception as exc:
                if _api_exception_status(exc) == 404:
                    verified[key] = True
                    continue
                raise
            raise OpenPIServiceError(
                f"unowned {key} {names[key]!r} appeared during cleanup"
            )
        while True:
            try:
                reader(names[key], namespace)
            except Exception as exc:
                if _api_exception_status(exc) == 404:
                    verified[key] = True
                    break
                raise
            time.sleep(2)
    for pod_name in owned_pod_names:
        try:
            api.delete_namespaced_pod(pod_name, namespace)
        except Exception as exc:
            if _api_exception_status(exc) != 404:
                raise
    while True:
        pods = api.list_namespaced_pod(namespace, label_selector=pod_selector).items
        if not pods:
            verified["pods"] = True
            break
        # Parents are absent and every pre-existing exact pod was explicitly
        # deleted. A new matching pod here would mean the cleanup identity changed
        # concurrently, so fail closed instead of broadening the deletion set.
        raise OpenPIServiceError(
            f"new pods appeared during exact cleanup: "
            f"{[pod.metadata.name for pod in pods]}"
        )
    return verified


def _run(args: argparse.Namespace) -> int:
    if os.environ.get(OPENPI_TERMS_ENV) != OPENPI_TERMS_ACCEPTED_VALUE:
        refusal = {
            "schema": "npa.workbench.openpi.terms-gate.v1",
            "status": "refused",
            "exit_code": 64,
            "checkpoint_fetch_started": False,
            "model_import_started": False,
        }
        _write_json_uri(args.output_uri, refusal)
        print(json.dumps(refusal, sort_keys=True), flush=True)
        return 64
    if not RUNTIME_IMAGE_RE.fullmatch(args.runtime_image):
        raise OpenPIServiceError("service runtime image must be digest-pinned")

    from kubernetes import client, config

    config.load_incluster_config()
    api = client.CoreV1Api()
    apps = client.AppsV1Api()
    batch = client.BatchV1Api()
    manifests = build_manifests(
        run_id=args.run_id,
        namespace=args.namespace,
        runtime_image=args.runtime_image,
        checkpoint_uri=args.checkpoint_uri,
        config_name=args.config_name,
        gpu_count=args.gpu_count,
        expected_gpu_type=args.expected_gpu_type,
        expected_compute_capability=args.expected_compute_capability,
        server_cpu=args.server_cpu,
        server_memory=args.server_memory,
        client_cpu=args.client_cpu,
        client_memory=args.client_memory,
        pull_secret=args.pull_secret,
        liveness_initial_delay_seconds=args.liveness_initial_delay_seconds,
        gpu_node_selector_key=args.gpu_node_selector_key,
        gpu_node_selector_value=args.gpu_node_selector_value,
        cache_size=args.service_cache_size,
    )
    checkpoint_provenance = _checkpoint_provenance(args.checkpoint_uri)
    names = {key: str(value["metadata"]["name"]) for key, value in manifests.items()}
    cleanup_verified: dict[str, bool] = {}
    created: set[str] = set()
    preflight_passed = False
    result: dict[str, object] | None = None
    started = time.perf_counter()
    try:
        _assert_targets_absent(
            api,
            apps,
            batch,
            namespace=args.namespace,
            names=names,
        )
        preflight_passed = True
        _create_server_objects(api, apps, manifests, created=created)
        server_pod = _wait_server_ready(api, apps, args.namespace, names["deployment"])
        service_dns = f"{names['service']}.{args.namespace}.svc.cluster.local"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://{service_dns}:{SERVER_PORT}/healthz") as response:
            health = response.read().decode("utf-8").strip()
        if health != "OK":
            raise OpenPIServiceError(f"unexpected service health response {health!r}")
        server_logs = api.read_namespaced_pod_log(
            server_pod.metadata.name, args.namespace
        )
        server_hardware = _prefixed_json(server_logs, "NPA_OPENPI_SERVER_HARDWARE=")
        # A Kubernetes Job starts as soon as it is created. Create the independent
        # client only after readiness and an in-cluster ClusterIP health request,
        # otherwise a clean cold-start can consume backoffLimit=0 before the
        # 12 GB checkpoint has loaded.
        _create_client_job(batch, manifests, created=created)
        client_pod = _wait_client(api, batch, args.namespace, names["client_job"])
        client_logs = api.read_namespaced_pod_log(
            client_pod.metadata.name, args.namespace
        )
        # Prefixing prevents a Kubernetes client/content-type layer from
        # interpreting a log body that consists solely of JSON and returning
        # its Python mapping representation instead of the original bytes.
        client_result = _prefixed_json(client_logs, "NPA_OPENPI_CLIENT_RESULT=")
        if (
            client_result.get("status") != "passed"
            or client_result.get("request_count") != 2
        ):
            raise OpenPIServiceError(
                f"cross-pod client evidence is invalid: {client_result}"
            )
        requests = client_result.get("requests")
        if not isinstance(requests, list) or len(requests) != 2:
            raise OpenPIServiceError(
                "cross-pod client did not return exactly two requests"
            )
        for request in requests:
            if (
                request.get("dtype") != "float64"
                or request.get("finite") is not True
                or list(request.get("shape", []))[1:] != [ACTION_DIM]
                or int(list(request.get("shape", [0]))[0]) < 5
            ):
                raise OpenPIServiceError(f"invalid served request evidence: {request}")
        server_uid = str(server_pod.metadata.uid)
        client_uid = str(client_pod.metadata.uid)
        if not server_uid or not client_uid or server_uid == client_uid:
            raise OpenPIServiceError(
                "server and client were not distinct Kubernetes pods"
            )
        result = {
            "schema": "npa.workbench.openpi.pi05-cross-pod-service.v1",
            "status": "passed",
            "mode": "serve",
            "source": {
                "repository": "https://github.com/Physical-Intelligence/openpi",
                "ref": SOURCE_REF,
                "license": SOURCE_LICENSE,
                "server_entrypoint": "/opt/byof/scripts/serve_policy.py",
                "transport": "upstream_openpi_websocket",
            },
            "redistribution": _redistribution_evidence(),
            "runtime_image": args.runtime_image,
            "hardware": server_hardware,
            "checkpoint": {
                "uri": args.checkpoint_uri,
                "provenance": checkpoint_provenance,
                "weights_baked": False,
            },
            "topology": {
                "deployment": names["deployment"],
                "service": names["service"],
                "service_type": "ClusterIP",
                "public_ingress": False,
                "server_pod_uid": server_uid,
                "client_pod_uid": client_uid,
                "separate_pods": True,
                "server_gpu_request": args.gpu_count,
                "client_gpu_request": 0,
                "controller_service_account": args.controller_service_account,
            },
            "probes": {
                "readiness": {"path": "/healthz", "passed": True},
                "liveness": {"path": "/healthz", "configured": True},
                "clusterip_health": health,
            },
            "client": client_result,
            "request_count": 2,
            "all_trajectories_float64_finite_t_ge_5_x_8": True,
            "service_seconds_before_cleanup": round(time.perf_counter() - started, 3),
            "terms": {
                "forwarded_via_ephemeral_secret": True,
                "persisted_after_cleanup": False,
            },
            "cleanup_identity": {
                "managed_by": MANAGED_BY,
                "exact_names": names,
                "pod_selector": (f"npa.nebius.ai/cleanup-owner={names['deployment']}"),
                "controller_service_account": args.controller_service_account,
                "controller_rbac_lifecycle": "external_exact_owner_apply_delete",
            },
            "limitations": [
                "clusterip_only_no_external_ingress_tested",
                "no_physical_franka_task_success_claim",
            ],
        }
    finally:
        if preflight_passed:
            cleanup_verified = _delete_and_verify(
                api,
                apps,
                batch,
                namespace=args.namespace,
                names=names,
                created=created,
            )
    if result is None:
        raise OpenPIServiceError("service run did not produce evidence")
    if not cleanup_verified or not all(cleanup_verified.values()):
        raise OpenPIServiceError(
            "service resources were not independently verified absent"
        )
    result["cleanup"] = {
        "all_exact_resources_absent": True,
        "verified": cleanup_verified,
    }
    # Replace the pre-cleanup artifact atomically by writing a separate final URI is
    # undesirable for lineage. S3 does not support If-Match through put_object in all
    # compatible providers, so publish cleanup as its own sibling proof.
    cleanup_uri = args.output_uri.removesuffix(".json") + ".cleanup.json"
    _write_json_uri(
        cleanup_uri,
        {
            "schema": "npa.workbench.openpi.service-cleanup.v1",
            "service_artifact_uri": args.output_uri,
            "all_exact_resources_absent": True,
            "verified": cleanup_verified,
        },
    )
    result["cleanup_artifact_uri"] = cleanup_uri
    _write_json_uri(args.output_uri, result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--runtime-image", required=True)
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--checkpoint-uri", default=DEFAULT_CHECKPOINT_URI)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--expected-gpu-type", required=True)
    parser.add_argument("--expected-compute-capability", required=True)
    parser.add_argument("--server-cpu", default="16")
    parser.add_argument("--server-memory", default="96Gi")
    parser.add_argument("--client-cpu", default="2")
    parser.add_argument("--client-memory", default="8Gi")
    parser.add_argument("--pull-secret", default="npa-nebius-registry")
    parser.add_argument("--liveness-initial-delay-seconds", type=int, default=600)
    parser.add_argument("--gpu-node-selector-key", default="nebius.com/gpu-name")
    parser.add_argument("--gpu-node-selector-value", default="B200")
    parser.add_argument("--service-cache-size", default="40Gi")
    parser.add_argument("--controller-service-account", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return _run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
