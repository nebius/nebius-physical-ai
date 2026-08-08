"""Kubernetes contract for a browser-teleoperated LeIsaac session."""

from __future__ import annotations

import ipaddress
import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from npa.agent_backend.leisaac_registry import (
    DEFAULT_ENVIRONMENT_ID,
    DEFAULT_TASK,
    REGISTRY_FINGERPRINT,
    TELEOP_DEVICE,
    resolve_configuration,
    validate_environment_id,
    validate_environment_index,
    validate_num_envs,
    validate_seed,
    validate_task,
)

SESSION_SCHEMA = "npa.leisaac.session.v2"
TASK = DEFAULT_TASK  # compatibility alias for callers that predate the registry
SOURCE_VERSION = "0.4.0"
SOURCE_COMMIT = "1651c321e9b0c1bb54233211fc7b3cd70d8373d5"
ISAAC_SIM_VERSION = "5.1.0.0"
ISAAC_LAB_VERSION = "2.3.2.post1"
SIGNAL_PORT = 49100
MEDIA_PORT = 47998
TURN_PORT = 3478
TURN_RELAY_PORT = 47999
TURN_RELAY_MAX_PORT = 48015
TURN_ALLOCATION_QUOTA = 16
TURN_IMAGE = (
    "docker.io/coturn/coturn@"
    "sha256:747ffd6c11fffad8c9c344a116d45f1365ee69a3e3af6475ce5c49e1024848f5"
)
SERVICE_PORT = 8080
RELAY_SERVICE_PORT = 48080
GPU_PRODUCT = "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition"
TRANSPORT_LOAD_BALANCER = "public-load-balancer"
TRANSPORT_AGENT_RELAY = "agent-relay"

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class LeIsaacConfigError(ValueError):
    """Raised when a teleoperation session would be unsafe or unusable."""


def resource_name(run_id: str) -> str:
    normalized = re.sub(r"[^a-z0-9-]+", "-", run_id.lower()).strip("-")
    if not normalized:
        raise LeIsaacConfigError("run id does not produce a Kubernetes resource name")
    return f"leisaac-{normalized[:45]}"


def validate_run_id(run_id: str) -> str:
    value = str(run_id or "").strip()
    if not _RUN_ID.fullmatch(value):
        raise LeIsaacConfigError(
            "run id must contain only letters, numbers, '.', '_' and '-'"
        )
    return value


def turn_credential(session_nonce: str) -> str:
    """Derive the ephemeral TURN password without publishing the session nonce."""

    nonce = str(session_nonce or "").strip().lower()
    if not re.fullmatch(r"[a-f0-9]{64}", nonce):
        raise LeIsaacConfigError("session nonce is invalid")
    return hashlib.sha256(f"npa-leisaac-turn:{nonce}".encode()).hexdigest()


def session_attestation(session_nonce: str) -> str:
    """Derive a public health attestation without returning the bearer nonce."""

    nonce = str(session_nonce or "").strip().lower()
    if not re.fullmatch(r"[a-f0-9]{64}", nonce):
        raise LeIsaacConfigError("session nonce is invalid")
    return hashlib.sha256(f"npa-leisaac-session:{nonce}".encode()).hexdigest()


def validate_image(image: str) -> str:
    value = str(image or "").strip()
    if not re.fullmatch(r"[^\s@]+@sha256:[a-f0-9]{64}", value):
        raise LeIsaacConfigError("LeIsaac image must be pinned by sha256 digest")
    return value


def validate_public_ip(value: str, label: str) -> str:
    try:
        address = ipaddress.ip_address(str(value or "").strip())
    except ValueError as exc:
        raise LeIsaacConfigError(f"{label} must be an IP address") from exc
    if not address.is_global:
        raise LeIsaacConfigError(f"{label} must be a public IP address")
    return address.compressed


def validate_private_ip(value: str, label: str) -> str:
    try:
        address = ipaddress.ip_address(str(value or "").strip())
    except ValueError as exc:
        raise LeIsaacConfigError(f"{label} must be an IP address") from exc
    if (
        address.version != 4
        or not address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_unspecified
    ):
        raise LeIsaacConfigError(f"{label} must be a private IPv4 address")
    return address.compressed


def validate_source_ranges(values: list[str] | tuple[str, ...]) -> list[str]:
    result: list[str] = []
    for raw in values:
        try:
            network = ipaddress.ip_network(str(raw or "").strip(), strict=False)
        except ValueError as exc:
            raise LeIsaacConfigError(f"invalid LeIsaac source range: {raw}") from exc
        # ``ipaddress`` changed its classification of the all-addresses
        # network across supported Python releases.  Reject an unrestricted
        # route explicitly instead of depending on that version-specific
        # classification.
        if network.prefixlen == 0 or not network.is_global:
            raise LeIsaacConfigError(f"LeIsaac source range must be public: {network}")
        result.append(network.with_prefixlen)
    if not result:
        raise LeIsaacConfigError("at least one agent/operator source range is required")
    return sorted(set(result))


def validate_expiry(value: str, *, now: datetime | None = None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LeIsaacConfigError("expires-at must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise LeIsaacConfigError("expires-at must include a timezone")
    parsed = parsed.astimezone(timezone.utc)
    if parsed <= (now or datetime.now(timezone.utc)):
        raise LeIsaacConfigError("expires-at must be in the future")
    return parsed.isoformat().replace("+00:00", "Z")


def split_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(str(uri or ""))
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise LeIsaacConfigError("S3 path must be s3://BUCKET/PREFIX")
    return parsed.netloc, parsed.path.strip("/")


def recorder_secret_manifest(
    *,
    run_id: str,
    namespace: str,
    output_path: str = "",
    endpoint: str,
    access_key: str,
    secret_key: str,
    region: str,
) -> dict[str, Any]:
    """Create the run-scoped Secret used only by the recorder process."""

    name = resource_name(validate_run_id(run_id))
    split_s3_uri(output_path)
    if not endpoint or not access_key or not secret_key:
        raise LeIsaacConfigError("recorder storage credentials are incomplete")
    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": f"{name}-recorder",
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/name": "leisaac",
                "app.kubernetes.io/instance": name,
                "app.kubernetes.io/managed-by": "npa",
                "npa.nebius.com/transient": "true",
            },
        },
        "type": "Opaque",
        "stringData": {
            "AWS_ENDPOINT_URL_S3": endpoint,
            "AWS_ACCESS_KEY_ID": access_key,
            "AWS_SECRET_ACCESS_KEY": secret_key,
            "AWS_REGION": region or "eu-north1",
            "NPA_LEISAAC_OUTPUT_PATH": output_path.rstrip("/"),
        },
    }


def service_manifests(
    *,
    run_id: str,
    namespace: str,
    source_ranges: list[str] | tuple[str, ...],
) -> list[dict[str, Any]]:
    """Build the two LBs before the GPU pod so its public media IP is known."""

    run_id = validate_run_id(run_id)
    ranges = validate_source_ranges(source_ranges)
    name = resource_name(run_id)
    labels = {
        "app": name,
        "app.kubernetes.io/name": "leisaac",
        "app.kubernetes.io/instance": name,
        "app.kubernetes.io/managed-by": "npa",
    }
    return [
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{name}-tcp",
                "namespace": namespace,
                "labels": labels,
            },
            "spec": {
                "type": "LoadBalancer",
                "loadBalancerSourceRanges": ranges,
                "selector": {"app": name},
                "ports": [
                    {
                        "name": "status",
                        "protocol": "TCP",
                        "port": SERVICE_PORT,
                        "targetPort": SERVICE_PORT,
                    },
                    {
                        "name": "signal",
                        "protocol": "TCP",
                        "port": SIGNAL_PORT,
                        "targetPort": SIGNAL_PORT,
                    },
                ],
            },
        },
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{name}-media",
                "namespace": namespace,
                "labels": labels,
            },
            "spec": {
                "type": "LoadBalancer",
                "loadBalancerSourceRanges": ranges,
                "selector": {"app": name},
                "ports": [
                    {
                        "name": "media",
                        "protocol": "UDP",
                        "port": MEDIA_PORT,
                        "targetPort": MEDIA_PORT,
                    }
                ],
            },
        },
    ]


def relay_service_manifest(
    *,
    run_id: str,
    namespace: str,
    agent_project: str = "",
    agent_name: str = "",
    source_ranges: list[str] | tuple[str, ...] = (),
    turn_peer_source: str = "",
) -> dict[str, Any]:
    """Build one private ClusterIP service for an agent-relayed session.

    The service has no cloud load balancer or public address. The authenticated
    reverse sidecar carries status, signaling, and TURN control datagrams to the
    agent's fixed listeners. The TURN allocation itself stays beside the
    simulator, so browser media does not require cross-VPC GPU ingress.
    """

    run_id = validate_run_id(run_id)
    name = resource_name(run_id)
    labels = {
        "app": name,
        "app.kubernetes.io/name": "leisaac",
        "app.kubernetes.io/instance": name,
        "app.kubernetes.io/managed-by": "npa",
    }
    annotations = {
        "npa.nebius.com/agent-project": str(agent_project),
        "npa.nebius.com/agent-name": str(agent_name),
        "npa.nebius.com/source-ranges": ",".join(validate_source_ranges(source_ranges)),
    }
    if turn_peer_source:
        peer = validate_source_ranges([turn_peer_source])
        network = ipaddress.ip_network(peer[0])
        if len(peer) != 1 or network.version != 4 or network.prefixlen < 22:
            raise LeIsaacConfigError(
                "TURN peer source must be one public IPv4 CIDR between /22 and /32"
            )
        annotations["npa.nebius.com/turn-peer-source"] = peer[0]
    if not agent_project or not agent_name:
        raise LeIsaacConfigError("agent relay requires an agent project and name")
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": f"{name}-relay",
            "namespace": namespace,
            "labels": labels,
            "annotations": annotations,
        },
        "spec": {
            "type": "ClusterIP",
            "selector": {"app": name},
            "ports": [
                {
                    "name": "status",
                    "protocol": "TCP",
                    "port": SERVICE_PORT,
                    "targetPort": SERVICE_PORT,
                },
                {
                    "name": "signal",
                    "protocol": "TCP",
                    "port": SIGNAL_PORT,
                    "targetPort": SIGNAL_PORT,
                },
                {
                    "name": "media",
                    "protocol": "UDP",
                    "port": MEDIA_PORT,
                    "targetPort": MEDIA_PORT,
                },
            ],
        },
    }


def relay_client_secret_manifest(
    *,
    run_id: str,
    namespace: str,
    agent_host: str,
    session_nonce: str,
    certificate_sha256: str,
    auth_user: str,
    auth_password: str,
    client_source: str,
) -> dict[str, Any]:
    """Mount the authenticated TLS backhaul client into the GPU pod."""

    name = resource_name(validate_run_id(run_id))
    agent_host = validate_public_ip(agent_host, "agent host")
    if not re.fullmatch(r"[a-f0-9]{64}", session_nonce):
        raise LeIsaacConfigError("session nonce is invalid")
    if not re.fullmatch(r"[a-f0-9]{64}", certificate_sha256):
        raise LeIsaacConfigError("relay certificate fingerprint is invalid")
    if not auth_user or not auth_password or "\n" in auth_user + auth_password:
        raise LeIsaacConfigError("agent basic-auth credential is invalid")
    turn_config = f"""listening-port={TURN_PORT}
min-port={TURN_RELAY_PORT}
max-port={TURN_RELAY_MAX_PORT}
realm=npa-leisaac
user={run_id}:{turn_credential(session_nonce)}
fingerprint
lt-cred-mech
stale-nonce=600
total-quota={TURN_ALLOCATION_QUOTA}
user-quota={TURN_ALLOCATION_QUOTA}
no-tcp
no-tls
no-dtls
no-cli
no-multicast-peers
pidfile=/tmp/npa-leisaac-turn.pid
simple-log
log-file=stdout
"""
    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": f"{name}-relay-client",
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/name": "leisaac",
                "app.kubernetes.io/instance": name,
                "app.kubernetes.io/managed-by": "npa",
            },
        },
        "type": "Opaque",
        "stringData": {
            "reverse_client.py": client_source,
            "config.json": json.dumps(
                {
                    "agent_host": agent_host,
                    "session_nonce": session_nonce,
                    "certificate_sha256": certificate_sha256,
                    "auth_user": auth_user,
                    "auth_password": auth_password,
                },
                sort_keys=True,
            ),
            "turnserver.conf": turn_config,
        },
    }


def deployment_manifest(
    *,
    run_id: str,
    namespace: str,
    image: str,
    media_host: str,
    session_nonce: str,
    media_server: str = "",
    image_pull_secret: str = "npa-registry",
    relay_client_secret: str = "",
    recorder_secret: str = "",
    task: str = DEFAULT_TASK,
    environment_id: str = DEFAULT_ENVIRONMENT_ID,
    environment_index: int = 0,
    seed: int = 42,
    num_envs: int = 1,
) -> dict[str, Any]:
    run_id = validate_run_id(run_id)
    image = validate_image(image)
    try:
        task = validate_task(task)
        environment_id = validate_environment_id(environment_id)
        environment_index = validate_environment_index(environment_index)
        seed = validate_seed(seed)
        num_envs = validate_num_envs(num_envs)
    except ValueError as exc:
        raise LeIsaacConfigError(str(exc)) from exc
    if not recorder_secret:
        raise LeIsaacConfigError(
            "LeIsaac demonstration collection requires a recorder Secret"
        )
    media_host = validate_public_ip(media_host, "media host")
    if not relay_client_secret and media_server:
        media_server = validate_public_ip(media_server, "media server")
        if media_server != media_host:
            raise LeIsaacConfigError(
                "public load-balancer media server must match the media host"
            )
    if not re.fullmatch(r"[a-f0-9]{64}", session_nonce):
        raise LeIsaacConfigError(
            "session nonce must be 64 lowercase hexadecimal characters"
        )
    name = resource_name(run_id)
    labels = {
        "app": name,
        "app.kubernetes.io/name": "leisaac",
        "app.kubernetes.io/instance": name,
        "app.kubernetes.io/managed-by": "npa",
        "npa.nebius.com/leisaac-task": task,
        "npa.nebius.com/environment-id": environment_id,
    }
    configuration = resolve_configuration(task)
    environment = {
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "ISAACSIM_ACCEPT_EULA": "YES",
        "NPA_LEISAAC_RUN_ID": run_id,
        "NPA_LEISAAC_SESSION_NONCE": session_nonce,
        "NPA_LEISAAC_TASK": task,
        "NPA_LEISAAC_ENVIRONMENT_ID": environment_id,
        "NPA_LEISAAC_ENVIRONMENT_INDEX": str(environment_index),
        "NPA_LEISAAC_SEED": str(seed),
        "NPA_LEISAAC_NUM_ENVS": str(num_envs),
        "NPA_LEISAAC_REGISTRY_FINGERPRINT": REGISTRY_FINGERPRINT,
        "NPA_LEISAAC_SOURCE_COMMIT": SOURCE_COMMIT,
        "NPA_LEISAAC_SOURCE_VERSION": SOURCE_VERSION,
        "NPA_LEISAAC_ISAAC_SIM_VERSION": ISAAC_SIM_VERSION,
        "NPA_LEISAAC_ISAAC_LAB_VERSION": ISAAC_LAB_VERSION,
        "NPA_LEISAAC_IMAGE": image,
        "NPA_LEISAAC_ROBOT": str(configuration["robot"]["id"]),
        "NPA_LEISAAC_SCENE": str(configuration["scene"]["id"]),
        "NPA_LEISAAC_DEVICE": str(configuration["device"]["id"]),
        "NPA_LEISAAC_BUNDLE": "built-in",
        "NVIDIA_DRIVER_CAPABILITIES": "all",
    }
    environment_items: list[dict[str, Any]] = [
        {"name": key, "value": value} for key, value in sorted(environment.items())
    ]
    for key in (
        "AWS_ENDPOINT_URL_S3",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION",
        "NPA_LEISAAC_OUTPUT_PATH",
    ):
        environment_items.append(
            {
                "name": key,
                "valueFrom": {"secretKeyRef": {"name": recorder_secret, "key": key}},
            }
        )
    if relay_client_secret:
        # TURN runs in this pod. The browser reaches its control port through the
        # authenticated agent backhaul, while its relay allocation and simulator
        # exchange media directly inside the shared pod network namespace.
        environment_items.append(
            {
                "name": "NPA_LEISAAC_MEDIA_HOST",
                "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}},
            }
        )
    else:
        environment_items.append(
            {"name": "NPA_LEISAAC_MEDIA_HOST", "value": media_host}
        )
    media_port: dict[str, Any] = {
        "name": "media",
        "containerPort": MEDIA_PORT,
        "protocol": "UDP",
    }
    pod_spec: dict[str, Any] = {
        "nodeSelector": {"nvidia.com/gpu.product": GPU_PRODUCT},
        "containers": [
            {
                "name": "leisaac",
                "image": image,
                "imagePullPolicy": "IfNotPresent",
                "ports": [
                    {
                        "name": "status",
                        "containerPort": SERVICE_PORT,
                        "protocol": "TCP",
                    },
                    {"name": "signal", "containerPort": SIGNAL_PORT, "protocol": "TCP"},
                    media_port,
                ],
                "env": environment_items,
                "resources": {
                    "requests": {
                        # LeIsaac uses CPU PhysX on sm_120 while the RTX GPU
                        # renders and encodes the interactive viewport.  The
                        # first scene reset saturates the old eight-core quota.
                        "cpu": "16",
                        "memory": "24Gi",
                        "ephemeral-storage": "70Gi",
                        "nvidia.com/gpu": "1",
                    },
                    "limits": {
                        "cpu": "32",
                        "memory": "48Gi",
                        "ephemeral-storage": "90Gi",
                        "nvidia.com/gpu": "1",
                    },
                },
                "readinessProbe": {
                    "httpGet": {"path": "/status", "port": "status"},
                    "periodSeconds": 5,
                    "failureThreshold": 720,
                },
                "livenessProbe": {
                    "httpGet": {"path": "/healthz", "port": "status"},
                    "initialDelaySeconds": 30,
                    "periodSeconds": 30,
                    "failureThreshold": 30,
                },
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "capabilities": {"drop": ["ALL"]},
                    "runAsNonRoot": True,
                    "runAsUser": 1000,
                    "seccompProfile": {"type": "RuntimeDefault"},
                },
                "volumeMounts": [
                    {"name": "isaac-cache", "mountPath": "/opt/isaac-cache"},
                    {"name": "leisaac-cache", "mountPath": "/opt/leisaac-cache"},
                    {"name": "tmp", "mountPath": "/tmp"},
                    {"name": "shm", "mountPath": "/dev/shm"},
                ],
            }
        ],
        "volumes": [
            {"name": "isaac-cache", "emptyDir": {"sizeLimit": "30Gi"}},
            {"name": "leisaac-cache", "emptyDir": {"sizeLimit": "2Gi"}},
            {"name": "tmp", "emptyDir": {"sizeLimit": "20Gi"}},
            {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "8Gi"}},
        ],
        "restartPolicy": "Always",
    }
    if relay_client_secret:
        pod_spec["containers"].append(
            {
                "name": "agent-relay-client",
                "image": image,
                "imagePullPolicy": "IfNotPresent",
                "command": [
                    "/opt/npa/sim/venv/bin/python",
                    "/opt/npa-relay/reverse_client.py",
                    "--config",
                    "/opt/npa-relay/config.json",
                ],
                "resources": {
                    "requests": {"cpu": "50m", "memory": "64Mi"},
                    "limits": {"cpu": "500m", "memory": "256Mi"},
                },
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "capabilities": {"drop": ["ALL"]},
                    "runAsNonRoot": True,
                    "runAsUser": 1000,
                    "seccompProfile": {"type": "RuntimeDefault"},
                },
                "volumeMounts": [
                    {
                        "name": "relay-client",
                        "mountPath": "/opt/npa-relay",
                        "readOnly": True,
                    }
                ],
            }
        )
        pod_spec["containers"].append(
            {
                "name": "turn",
                "image": TURN_IMAGE,
                "imagePullPolicy": "IfNotPresent",
                "command": ["turnserver"],
                "args": ["-c", "/opt/npa-relay/turnserver.conf"],
                "ports": [
                    {
                        "name": "turn-control",
                        "containerPort": TURN_PORT,
                        "protocol": "UDP",
                    },
                    {
                        "name": "turn-media",
                        "containerPort": TURN_RELAY_PORT,
                        "protocol": "UDP",
                    },
                ],
                "resources": {
                    "requests": {"cpu": "50m", "memory": "64Mi"},
                    "limits": {"cpu": "500m", "memory": "256Mi"},
                },
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    # The upstream image marks turnserver with
                    # cap_net_bind_service. Keep only that file capability;
                    # dropping the full bounding set makes Linux reject exec.
                    "capabilities": {
                        "drop": ["ALL"],
                        "add": ["NET_BIND_SERVICE"],
                    },
                    "runAsNonRoot": True,
                    "runAsUser": 65534,
                    "runAsGroup": 65533,
                    "seccompProfile": {"type": "RuntimeDefault"},
                },
                "volumeMounts": [
                    {
                        "name": "relay-client",
                        "mountPath": "/opt/npa-relay",
                        "readOnly": True,
                    },
                ],
            }
        )
        pod_spec["volumes"].append(
            {
                "name": "relay-client",
                "secret": {
                    "secretName": relay_client_secret,
                    "defaultMode": 0o555,
                },
            }
        )
    if image_pull_secret:
        pod_spec["imagePullSecrets"] = [{"name": image_pull_secret}]
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": name, "namespace": namespace, "labels": labels},
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": {"app": name}},
            "template": {"metadata": {"labels": labels}, "spec": pod_spec},
        },
    }


def session_manifest(
    *,
    run_id: str,
    image: str,
    signal_host: str,
    media_host: str,
    session_nonce: str,
    media_server: str = "",
    expires_at: str = "",
    gpu: str = GPU_PRODUCT,
    created_at: str | None = None,
    transport: str = TRANSPORT_LOAD_BALANCER,
    task: str = DEFAULT_TASK,
    environment_id: str = DEFAULT_ENVIRONMENT_ID,
    environment_index: int = 0,
    seed: int = 42,
    num_envs: int = 1,
    output_path: str,
) -> dict[str, Any]:
    run_id = validate_run_id(run_id)
    image = validate_image(image)
    try:
        task = validate_task(task)
        environment_id = validate_environment_id(environment_id)
        environment_index = validate_environment_index(environment_index)
        seed = validate_seed(seed)
        num_envs = validate_num_envs(num_envs)
    except ValueError as exc:
        raise LeIsaacConfigError(str(exc)) from exc
    split_s3_uri(output_path)
    if transport not in (TRANSPORT_LOAD_BALANCER, TRANSPORT_AGENT_RELAY):
        raise LeIsaacConfigError(f"unsupported LeIsaac transport: {transport}")
    if transport == TRANSPORT_AGENT_RELAY:
        if signal_host != "127.0.0.1":
            raise LeIsaacConfigError("agent-relay signaling must use 127.0.0.1")
        media_server = validate_private_ip(media_server, "agent relay media server")
    else:
        signal_host = validate_public_ip(signal_host, "signal host")
    media_host = validate_public_ip(media_host, "media host")
    if transport != TRANSPORT_AGENT_RELAY:
        media_server = validate_public_ip(media_server or media_host, "media server")
        if media_server != media_host:
            raise LeIsaacConfigError(
                "public load-balancer media server must match the media host"
            )
    expires_at = validate_expiry(expires_at)
    if not re.fullmatch(r"[a-f0-9]{64}", session_nonce):
        raise LeIsaacConfigError(
            "session nonce must be 64 lowercase hexadecimal characters"
        )
    manifest = {
        "schema": SESSION_SCHEMA,
        "run_id": run_id,
        "provider": "nebius-kubernetes",
        "transport": transport,
        "task": task,
        "task_registry_fingerprint": REGISTRY_FINGERPRINT,
        "teleop_device": TELEOP_DEVICE,
        "configuration": resolve_configuration(task),
        "environment": {
            "id": environment_id,
            "index": environment_index,
            "seed": seed,
            "num_envs": num_envs,
            "model": "named-sequential",
        },
        "dataset": {
            "output_path": output_path.rstrip("/"),
            "format": "LeRobotDataset",
            "lerobot_version": "0.5.1",
            "codebase_version": "v3.0",
        },
        "signal_host": signal_host,
        "signal_port": SIGNAL_PORT,
        "media_host": media_host,
        "media_server": media_server,
        "media_port": MEDIA_PORT,
        "turn_port": TURN_PORT,
        "turn_relay_port": TURN_RELAY_PORT,
        "turn_relay_max_port": TURN_RELAY_MAX_PORT,
        "service_url": (
            f"http://127.0.0.1:{RELAY_SERVICE_PORT}"
            if transport == TRANSPORT_AGENT_RELAY
            else f"http://{signal_host}:{SERVICE_PORT}"
        ),
        "session_nonce": session_nonce,
        "session_attestation": session_attestation(session_nonce),
        "created_at": created_at
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_version": SOURCE_VERSION,
        "source_commit": SOURCE_COMMIT,
        "isaac_sim_version": ISAAC_SIM_VERSION,
        "isaac_lab_version": ISAAC_LAB_VERSION,
        "image": image,
        "gpu": gpu,
    }
    if expires_at:
        manifest["expires_at"] = expires_at
    return manifest
