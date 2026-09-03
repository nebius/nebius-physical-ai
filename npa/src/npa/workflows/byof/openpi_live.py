"""Persistent, authenticated OpenPI serving manifests for a live Antioch client.

The finite cross-pod gate remains in :mod:`openpi_service`.  This module derives a
long-lived Deployment from the same reviewed server contract, mounts the checkpoint
cache from a PVC, and exposes only an authenticated TLS WebSocket gateway.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import ipaddress
import json
import os
import secrets
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from npa.workflows.byof.openpi_service import OpenPIServiceError, build_manifests

GATEWAY_PORT = 8443
GATEWAY_HEALTH_PORT = 8002
LIVE_MANAGED_BY = "npa-openpi-antioch-live"


def gateway_program() -> str:
    """Return the dependency-free wrapper program run in the OpenPI image.

    ``websockets`` is already part of the pinned upstream server/client closure.
    One upstream connection is created per authenticated client connection. Requests
    are sequential and bounded because the OpenPI protocol itself is request/reply.
    """

    return r"""
import collections
import hmac
import http.server
import json
import os
import ssl
import threading
import time

from websockets.sync.client import connect
from websockets.sync.server import serve
from websockets.exceptions import ConnectionClosedOK

TOKEN = open("/run/npa-openpi-auth/api-key", encoding="utf-8").read().strip()
if len(TOKEN) < 32:
    raise RuntimeError("OpenPI gateway API key is missing or too short")
MAX_MESSAGE = int(os.environ.get("NPA_OPENPI_MAX_MESSAGE_BYTES", "33554432"))
REQUEST_TIMEOUT = float(os.environ.get("NPA_OPENPI_REQUEST_TIMEOUT_SECONDS", "120"))
MAX_REQUESTS = int(os.environ.get("NPA_OPENPI_MAX_REQUESTS_PER_CONNECTION", "1000000"))
MAX_CONNECTIONS = int(os.environ.get("NPA_OPENPI_MAX_CONNECTIONS", "4"))
if MAX_MESSAGE < 1 or REQUEST_TIMEOUT <= 0 or MAX_REQUESTS < 1 or MAX_CONNECTIONS < 1:
    raise RuntimeError("OpenPI gateway bounds must be positive")
slots = threading.BoundedSemaphore(MAX_CONNECTIONS)

lock = threading.Lock()
state = {
    "total_connections": 0,
    "rejected_connections": 0,
    "requests": 0,
    "failures": 0,
    "last_latency_ms": None,
    "latencies_ms": collections.deque(maxlen=512),
}

class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/healthz":
            payload, status = b"OK\n", 200
        elif self.path == "/metrics.json":
            with lock:
                snapshot = dict(state)
                values = list(snapshot.pop("latencies_ms"))
            snapshot["mean_latency_ms"] = (
                round(sum(values) / len(values), 3) if values else None
            )
            snapshot["sample_count"] = len(values)
            payload, status = json.dumps(snapshot, sort_keys=True).encode(), 200
        else:
            payload, status = b"not found\n", 404
        self.send_response(status)
        self.send_header("Content-Type", "application/json" if self.path.endswith(".json") else "text/plain")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
    def log_message(self, *_args):
        return

threading.Thread(
    target=lambda: http.server.ThreadingHTTPServer(("0.0.0.0", 8002), Health).serve_forever(),
    daemon=True,
).start()

def handle(client):
    authorization = client.request.headers.get("Authorization", "")
    if not hmac.compare_digest(authorization, "Api-Key " + TOKEN):
        with lock:
            state["rejected_connections"] += 1
        client.close(code=1008, reason="authentication required")
        return
    if not slots.acquire(blocking=False):
        with lock:
            state["rejected_connections"] += 1
        client.close(code=1013, reason="gateway at bounded connection capacity")
        return
    with lock:
        state["total_connections"] += 1
    try:
        with connect(
            "ws://127.0.0.1:8000",
            open_timeout=10,
            close_timeout=5,
            max_size=MAX_MESSAGE,
        ) as upstream:
            # Forward the server metadata greeting expected by openpi clients.
            client.send(upstream.recv(timeout=REQUEST_TIMEOUT))
            for _ in range(MAX_REQUESTS):
                request = client.recv(timeout=REQUEST_TIMEOUT)
                started = time.perf_counter()
                upstream.send(request)
                response = upstream.recv(timeout=REQUEST_TIMEOUT)
                client.send(response)
                latency = (time.perf_counter() - started) * 1000.0
                with lock:
                    state["requests"] += 1
                    state["last_latency_ms"] = round(latency, 3)
                    state["latencies_ms"].append(latency)
    except ConnectionClosedOK:
        pass
    except Exception:
        with lock:
            state["failures"] += 1
        raise
    finally:
        slots.release()

context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.minimum_version = ssl.TLSVersion.TLSv1_2
context.load_cert_chain("/run/npa-openpi-tls/tls.crt", "/run/npa-openpi-tls/tls.key")
with serve(
    handle,
    "0.0.0.0",
    8443,
    ssl=context,
    max_size=MAX_MESSAGE,
    max_queue=4,
    open_timeout=10,
    close_timeout=5,
) as server:
    server.serve_forever()
""".strip()


def build_live_manifests(
    *,
    run_id: str,
    namespace: str,
    runtime_image: str,
    checkpoint_uri: str,
    config_name: str,
    expected_gpu_type: str,
    expected_compute_capability: str,
    cache_pvc: str,
    auth_secret: str,
    tls_secret: str,
    kubelet_source_cidrs: Sequence[str],
    pull_secret: str = "",
    gpu_node_selector_key: str = "nebius.com/gpu-name",
    gpu_node_selector_value: str = "B200",
    source_ranges: Sequence[str] = (),
    server_cpu: str = "16",
    server_memory: str = "96Gi",
) -> dict[str, dict[str, Any]]:
    """Build a persistent single-replica Deployment and restricted LB Service."""

    for label, value in {
        "cache PVC": cache_pvc,
        "auth Secret": auth_secret,
        "TLS Secret": tls_secret,
    }.items():
        if not value.strip():
            raise OpenPIServiceError(f"{label} must not be empty")
    probe_sources: list[str] = []
    for value in kubelet_source_cidrs:
        try:
            network = ipaddress.ip_network(value, strict=True)
        except ValueError as exc:
            raise OpenPIServiceError(
                "kubelet probe sources must be exact node-address CIDRs"
            ) from exc
        if network.prefixlen != network.max_prefixlen:
            raise OpenPIServiceError(
                "kubelet probe sources must be exact node-address CIDRs"
            )
        probe_sources.append(str(network))
    probe_sources = sorted(set(probe_sources))
    if not probe_sources:
        raise OpenPIServiceError("at least one kubelet probe source is required")

    finite = build_manifests(
        run_id=run_id,
        namespace=namespace,
        runtime_image=runtime_image,
        checkpoint_uri=checkpoint_uri,
        config_name=config_name,
        gpu_count=1,
        expected_gpu_type=expected_gpu_type,
        expected_compute_capability=expected_compute_capability,
        server_cpu=server_cpu,
        server_memory=server_memory,
        client_cpu="1",
        client_memory="1Gi",
        pull_secret=pull_secret,
        liveness_initial_delay_seconds=600,
        gpu_node_selector_key=gpu_node_selector_key,
        gpu_node_selector_value=gpu_node_selector_value,
        cache_size="40Gi",
    )
    deployment = finite["deployment"]
    pod = deployment["spec"]["template"]["spec"]
    selector = deployment["spec"]["selector"]["matchLabels"]
    labels = deployment["metadata"]["labels"]
    live_labels = _live_labels(run_id)
    labels.update(live_labels)
    deployment["spec"]["template"]["metadata"].setdefault("labels", {}).update(
        live_labels
    )

    deployment["spec"].update(
        {"strategy": {"type": "Recreate"}, "revisionHistoryLimit": 2}
    )
    pod["terminationGracePeriodSeconds"] = 30
    pod["securityContext"] = {
        "fsGroup": 1000,
        "fsGroupChangePolicy": "OnRootMismatch",
    }
    pod["initContainers"] = [
        {
            "name": "cache-permissions",
            "image": runtime_image,
            "imagePullPolicy": "IfNotPresent",
            "command": [
                "/bin/sh",
                "-ceu",
                "install -d -o 1000 -g 1000 /cache && chmod 0700 /cache",
            ],
            "securityContext": {
                "runAsUser": 0,
                "runAsNonRoot": False,
                "allowPrivilegeEscalation": False,
            },
            "resources": {
                "requests": {"cpu": "50m", "memory": "64Mi"},
                "limits": {"cpu": "500m", "memory": "256Mi"},
            },
            "volumeMounts": [{"name": "openpi-cache", "mountPath": "/cache"}],
        }
    ]
    policy = pod["containers"][0]
    policy["securityContext"] = {
        "runAsUser": 1000,
        "runAsGroup": 1000,
        "runAsNonRoot": True,
        "allowPrivilegeEscalation": False,
    }
    policy["volumeMounts"] = [
        {
            "name": "openpi-cache",
            "mountPath": "/workspace/openpi-server-cache",
        }
    ]
    pod["volumes"] = [
        {
            "name": "openpi-cache",
            "persistentVolumeClaim": {"claimName": cache_pvc},
        },
        {"name": "gateway-auth", "secret": {"secretName": auth_secret}},
        {"name": "gateway-tls", "secret": {"secretName": tls_secret}},
    ]
    gateway = {
        "name": "authenticated-gateway",
        "image": runtime_image,
        "imagePullPolicy": "IfNotPresent",
        "command": ["/opt/venv/bin/python", "-c", gateway_program()],
        "ports": [
            {"name": "gateway", "containerPort": GATEWAY_PORT},
            {"name": "gateway-health", "containerPort": GATEWAY_HEALTH_PORT},
        ],
        "env": [
            {"name": "NPA_OPENPI_MAX_MESSAGE_BYTES", "value": "33554432"},
            {"name": "NPA_OPENPI_REQUEST_TIMEOUT_SECONDS", "value": "120"},
            {"name": "NPA_OPENPI_MAX_REQUESTS_PER_CONNECTION", "value": "1000000"},
            {"name": "NPA_OPENPI_MAX_CONNECTIONS", "value": "4"},
        ],
        "resources": {
            "requests": {"cpu": "250m", "memory": "256Mi"},
            "limits": {"cpu": "2", "memory": "2Gi"},
        },
        "securityContext": {
            "runAsUser": 1000,
            "runAsGroup": 1000,
            "runAsNonRoot": True,
            "allowPrivilegeEscalation": False,
        },
        "volumeMounts": [
            {
                "name": "gateway-auth",
                "mountPath": "/run/npa-openpi-auth",
                "readOnly": True,
            },
            {
                "name": "gateway-tls",
                "mountPath": "/run/npa-openpi-tls",
                "readOnly": True,
            },
        ],
        "readinessProbe": {
            "httpGet": {"path": "/healthz", "port": GATEWAY_HEALTH_PORT},
            "periodSeconds": 5,
            "failureThreshold": 12,
        },
        "livenessProbe": {
            "httpGet": {"path": "/healthz", "port": GATEWAY_HEALTH_PORT},
            "initialDelaySeconds": 15,
            "periodSeconds": 30,
            "failureThreshold": 3,
        },
    }
    pod["containers"].append(gateway)

    service = finite["service"]
    service["spec"] = {
        "type": "LoadBalancer",
        # Nebius Managed Kubernetes currently rejects Local for managed load
        # balancers. Cluster is therefore part of the tested provider contract.
        "externalTrafficPolicy": "Cluster",
        "selector": selector,
        "ports": [
            {
                "name": "wss",
                "protocol": "TCP",
                "port": 443,
                "targetPort": GATEWAY_PORT,
            }
        ],
    }
    if source_ranges:
        service["spec"]["loadBalancerSourceRanges"] = list(source_ranges)

    probe_ports = sorted(
        {
            int(container[probe]["httpGet"]["port"])
            for container in pod["containers"]
            for probe in ("readinessProbe", "livenessProbe")
        }
    )

    network_policy = {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "NetworkPolicy",
        "metadata": {
            "name": deployment["metadata"]["name"],
            "namespace": namespace,
            "labels": labels,
            "annotations": {
                # Standard NetworkPolicy has no kubelet identity selector. On the
                # target Cilium CNI, exact node-IP ipBlocks preserve probe access
                # without admitting ordinary pod sources. A host-network process
                # on an enumerated node can still reach these health-only ports;
                # that is the irreducible standard-NetworkPolicy tradeoff.
                "npa.nebius.ai/kubelet-probe-source-contract": "cilium-node-ip-ipblock",
                "npa.nebius.ai/kubelet-probe-host-network-tradeoff": "documented",
            },
        },
        "spec": {
            "podSelector": {"matchLabels": selector},
            "policyTypes": ["Ingress"],
            "ingress": [
                {
                    "ports": [
                        {"protocol": "TCP", "port": GATEWAY_PORT},
                    ]
                },
                {
                    "from": [{"ipBlock": {"cidr": cidr}} for cidr in probe_sources],
                    "ports": [
                        {"protocol": "TCP", "port": port} for port in probe_ports
                    ],
                },
            ],
        },
    }
    return {
        "terms_secret": finite["secret"],
        "deployment": deployment,
        "service": service,
        "network_policy": network_policy,
    }


def public_contract(manifests: dict[str, dict[str, Any]]) -> str:
    """Render only non-secret desired state for review and tests."""

    return json.dumps(manifests, sort_keys=True, separators=(",", ":"))


def _live_labels(run_id: str) -> dict[str, str]:
    from npa.workflows.byof.openpi_service import _safe_name

    return {
        "app.kubernetes.io/managed-by": LIVE_MANAGED_BY,
        "app.kubernetes.io/part-of": "openpi-antioch-live",
        "npa.nebius.ai/live-name": _safe_name(run_id),
    }


def _owned(metadata: Any, run_id: str) -> bool:
    labels = getattr(metadata, "labels", None) or {}
    expected = _live_labels(run_id)
    return all(labels.get(key) == value for key, value in expected.items())


def _api_status(exc: Exception) -> int:
    try:
        return int(getattr(exc, "status", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _kubelet_source_cidrs(nodes: Sequence[Any]) -> tuple[str, ...]:
    """Return exact InternalIP host routes used by kubelet probe traffic."""

    sources: set[str] = set()
    for node in nodes:
        for address in getattr(getattr(node, "status", None), "addresses", None) or []:
            if getattr(address, "type", "") != "InternalIP":
                continue
            try:
                parsed = ipaddress.ip_address(str(address.address))
            except ValueError as exc:
                raise OpenPIServiceError(
                    "node InternalIP inventory is malformed"
                ) from exc
            sources.add(f"{parsed}/{parsed.max_prefixlen}")
    if not sources:
        raise OpenPIServiceError(
            "node InternalIP inventory is empty; refusing to open probe ports"
        )
    return tuple(sorted(sources))


def _cache_pvc_manifest(
    *, name: str, namespace: str, size: str, storage_class: str, labels: dict[str, str]
) -> dict[str, Any]:
    """Build the single-replica cache claim supported by block CSI classes."""

    return {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {"name": name, "namespace": namespace, "labels": labels},
        "spec": {
            # The Deployment is one replica with Recreate strategy, so RWX adds
            # no availability and fails on standard Nebius block CSI classes.
            "accessModes": ["ReadWriteOnce"],
            "resources": {"requests": {"storage": size}},
            **({"storageClassName": storage_class} if storage_class else {}),
        },
    }


def _apply_owned(
    *,
    read: Callable[..., Any],
    create: Callable[..., Any],
    patch: Callable[..., Any],
    name: str,
    namespace: str,
    body: dict[str, Any],
    run_id: str,
) -> str:
    try:
        current = read(name=name, namespace=namespace)
    except Exception as exc:
        if _api_status(exc) != 404:
            raise
        create(namespace=namespace, body=body)
        return "created"
    if not _owned(current.metadata, run_id):
        raise OpenPIServiceError(
            f"refusing to replace unowned Kubernetes object {name!r}"
        )
    patch(name=name, namespace=namespace, body=body)
    return "reconciled"


def _certificate(endpoint: str) -> tuple[bytes, bytes, bytes]:
    """Create a short-lived run-local CA and server certificate for one LB address."""

    now = dt.datetime.now(dt.timezone.utc)
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "NPA OpenPI live CA")])
    ca = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=30))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .sign(ca_key, hashes.SHA256())
    )
    server_key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    server_name = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "NPA OpenPI live gateway")]
    )
    try:
        alternate_name: x509.GeneralName = x509.IPAddress(
            ipaddress.ip_address(endpoint)
        )
    except ValueError:
        alternate_name = x509.DNSName(endpoint)
    server = (
        x509.CertificateBuilder()
        .subject_name(server_name)
        .issuer_name(ca.subject)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=30))
        .add_extension(x509.SubjectAlternativeName([alternate_name]), critical=False)
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    private_key = server_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    return (
        ca.public_bytes(serialization.Encoding.PEM),
        server.public_bytes(serialization.Encoding.PEM),
        private_key,
    )


def _tls_rollout_digest(ca: bytes, certificate: bytes, private_key: bytes) -> str:
    """Bind a gateway pod generation to the exact in-memory TLS material."""

    return hashlib.sha256(ca + b"\0" + certificate + b"\0" + private_key).hexdigest()


def _wait_load_balancer(
    core: Any,
    *,
    name: str,
    namespace: str,
    timeout_seconds: float,
) -> str:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        service = core.read_namespaced_service(name=name, namespace=namespace)
        ingress = getattr(getattr(service, "status", None), "load_balancer", None)
        entries = getattr(ingress, "ingress", None) or []
        if len(entries) == 1:
            endpoint = str(entries[0].ip or entries[0].hostname or "").strip()
            if endpoint:
                return endpoint
        time.sleep(5)
    raise OpenPIServiceError("timed out waiting for the live gateway address")


def _wait_deployment_ready(
    apps: Any, *, name: str, namespace: str, timeout_seconds: float
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        deployment = apps.read_namespaced_deployment(name=name, namespace=namespace)
        status = deployment.status
        generation = int(deployment.metadata.generation or 0)
        observed = int(status.observed_generation or 0)
        if (
            observed >= generation
            and int(status.updated_replicas or 0) == 1
            and int(status.available_replicas or 0) == 1
            and int(status.unavailable_replicas or 0) == 0
        ):
            return
        for condition in status.conditions or []:
            if (
                condition.type == "Progressing"
                and condition.status == "False"
                and condition.reason == "ProgressDeadlineExceeded"
            ):
                raise OpenPIServiceError(
                    "live OpenPI Deployment exceeded its progress deadline"
                )
        time.sleep(5)
    raise OpenPIServiceError("timed out waiting for the live OpenPI Deployment")


def _write_client_bundle(
    directory: Path, *, endpoint: str, ca: bytes, token: str
) -> None:
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    files = {
        "ca.crt": ca,
        "api-key": (token + "\n").encode(),
        "endpoint.json": json.dumps(
            {"scheme": "wss", "host": endpoint, "port": 443},
            sort_keys=True,
        ).encode()
        + b"\n",
    }
    for name, content in files.items():
        path = directory / name
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(descriptor, content)
        finally:
            os.close(descriptor)
        os.chmod(path, 0o600)


def deploy_live(args: argparse.Namespace) -> dict[str, Any]:
    """Reconcile one live service and emit no credential or network endpoint."""

    from kubernetes import client, config

    if os.environ.get("NPA_OPENPI_ACCEPT_GEMMA_TERMS") != "YES":
        raise OpenPIServiceError(
            "NPA_OPENPI_ACCEPT_GEMMA_TERMS=YES is required before live deployment"
        )
    config.load_kube_config(config_file=args.kubeconfig or None)
    core = client.CoreV1Api()
    apps = client.AppsV1Api()
    networking = client.NetworkingV1Api()
    labels = _live_labels(args.run_id)
    kubelet_source_cidrs = _kubelet_source_cidrs(core.list_node().items)

    try:
        namespace_object = core.read_namespace(name=args.namespace)
    except Exception as exc:
        if _api_status(exc) != 404:
            raise
        core.create_namespace(
            body={
                "apiVersion": "v1",
                "kind": "Namespace",
                "metadata": {"name": args.namespace, "labels": labels},
            }
        )
    else:
        existing_labels = namespace_object.metadata.labels or {}
        if existing_labels.get("app.kubernetes.io/managed-by") not in (
            None,
            LIVE_MANAGED_BY,
        ):
            raise OpenPIServiceError(
                "refusing to use a namespace managed by another tool"
            )

    pvc = _cache_pvc_manifest(
        name=args.cache_pvc,
        namespace=args.namespace,
        size=args.cache_size,
        storage_class=args.storage_class,
        labels=labels,
    )
    _apply_owned(
        read=core.read_namespaced_persistent_volume_claim,
        create=core.create_namespaced_persistent_volume_claim,
        patch=core.patch_namespaced_persistent_volume_claim,
        name=args.cache_pvc,
        namespace=args.namespace,
        body=pvc,
        run_id=args.run_id,
    )
    auth_name = f"{args.cache_pvc}-auth"
    tls_name = f"{args.cache_pvc}-tls"
    manifests = build_live_manifests(
        run_id=args.run_id,
        namespace=args.namespace,
        runtime_image=args.runtime_image,
        checkpoint_uri=args.checkpoint_uri,
        config_name=args.config_name,
        expected_gpu_type=args.expected_gpu_type,
        expected_compute_capability=args.expected_compute_capability,
        cache_pvc=args.cache_pvc,
        auth_secret=auth_name,
        tls_secret=tls_name,
        kubelet_source_cidrs=kubelet_source_cidrs,
        pull_secret=args.pull_secret,
        source_ranges=tuple(args.source_range),
    )
    for manifest in manifests.values():
        manifest["metadata"].setdefault("labels", {}).update(labels)
    manifests["deployment"]["spec"]["template"]["metadata"].setdefault(
        "labels", {}
    ).update(labels)
    service = manifests["service"]
    _apply_owned(
        read=core.read_namespaced_service,
        create=core.create_namespaced_service,
        patch=core.patch_namespaced_service,
        name=service["metadata"]["name"],
        namespace=args.namespace,
        body=service,
        run_id=args.run_id,
    )
    endpoint = _wait_load_balancer(
        core,
        name=service["metadata"]["name"],
        namespace=args.namespace,
        timeout_seconds=args.load_balancer_timeout_seconds,
    )

    try:
        existing_auth = core.read_namespaced_secret(
            name=auth_name, namespace=args.namespace
        )
    except Exception as exc:
        if _api_status(exc) != 404:
            raise
        existing_auth = None
        api_key = secrets.token_urlsafe(48)
    else:
        if not _owned(existing_auth.metadata, args.run_id):
            raise OpenPIServiceError(
                "refusing to reuse an unowned authentication Secret"
            )
        encoded = (existing_auth.data or {}).get("api-key", "")
        api_key = base64.b64decode(encoded).decode("utf-8").strip()
        if len(api_key) < 32:
            raise OpenPIServiceError("existing authentication Secret is malformed")
    if existing_auth is None:
        ca, certificate, private_key = _certificate(endpoint)
    else:
        existing_tls = core.read_namespaced_secret(
            name=tls_name, namespace=args.namespace
        )
        if not _owned(existing_tls.metadata, args.run_id):
            raise OpenPIServiceError("refusing to reuse an unowned TLS Secret")
        encoded_tls = existing_tls.data or {}
        try:
            ca = base64.b64decode(encoded_tls["ca.crt"], validate=True)
            certificate = base64.b64decode(encoded_tls["tls.crt"], validate=True)
            private_key = base64.b64decode(encoded_tls["tls.key"], validate=True)
        except (KeyError, ValueError) as exc:
            raise OpenPIServiceError(
                "existing TLS Secret lacks reusable CA/certificate material"
            ) from exc
    secret_bodies = {
        auth_name: {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {
                "name": auth_name,
                "namespace": args.namespace,
                "labels": labels,
            },
            "type": "Opaque",
            "data": {"api-key": base64.b64encode(api_key.encode()).decode()},
        },
        tls_name: {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {
                "name": tls_name,
                "namespace": args.namespace,
                "labels": labels,
            },
            "type": "kubernetes.io/tls",
            "data": {
                "ca.crt": base64.b64encode(ca).decode(),
                "tls.crt": base64.b64encode(certificate).decode(),
                "tls.key": base64.b64encode(private_key).decode(),
            },
        },
    }
    for name, body in secret_bodies.items():
        _apply_owned(
            read=core.read_namespaced_secret,
            create=core.create_namespaced_secret,
            patch=core.patch_namespaced_secret,
            name=name,
            namespace=args.namespace,
            body=body,
            run_id=args.run_id,
        )
    terms = manifests["terms_secret"]
    _apply_owned(
        read=core.read_namespaced_secret,
        create=core.create_namespaced_secret,
        patch=core.patch_namespaced_secret,
        name=terms["metadata"]["name"],
        namespace=args.namespace,
        body=terms,
        run_id=args.run_id,
    )
    deployment = manifests["deployment"]
    deployment["spec"]["template"]["metadata"].setdefault("annotations", {})[
        "npa.nebius.ai/tls-material-sha256"
    ] = _tls_rollout_digest(ca, certificate, private_key)
    _apply_owned(
        read=apps.read_namespaced_deployment,
        create=apps.create_namespaced_deployment,
        patch=apps.patch_namespaced_deployment,
        name=deployment["metadata"]["name"],
        namespace=args.namespace,
        body=deployment,
        run_id=args.run_id,
    )
    policy = manifests["network_policy"]
    _apply_owned(
        read=networking.read_namespaced_network_policy,
        create=networking.create_namespaced_network_policy,
        patch=networking.patch_namespaced_network_policy,
        name=policy["metadata"]["name"],
        namespace=args.namespace,
        body=policy,
        run_id=args.run_id,
    )
    _wait_deployment_ready(
        apps,
        name=deployment["metadata"]["name"],
        namespace=args.namespace,
        timeout_seconds=args.deployment_timeout_seconds,
    )
    _write_client_bundle(
        Path(args.client_bundle_dir), endpoint=endpoint, ca=ca, token=api_key
    )
    return {
        "status": "reconciled",
        "deployment": deployment["metadata"]["name"],
        "namespace": args.namespace,
        "client_bundle_written": True,
        "endpoint_redacted": True,
        "cache_pvc": args.cache_pvc,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--runtime-image", required=True)
    parser.add_argument("--checkpoint-uri", required=True)
    parser.add_argument("--config-name", default="pi05_droid_jointpos_polaris")
    parser.add_argument("--expected-gpu-type", default="B200")
    parser.add_argument("--expected-compute-capability", default="10.0")
    parser.add_argument("--cache-pvc", required=True)
    parser.add_argument("--cache-size", default="64Gi")
    parser.add_argument("--storage-class", default="")
    parser.add_argument("--pull-secret", default="")
    parser.add_argument("--source-range", action="append", default=[])
    parser.add_argument("--client-bundle-dir", required=True)
    parser.add_argument("--kubeconfig", default="")
    parser.add_argument("--load-balancer-timeout-seconds", type=float, default=600)
    parser.add_argument("--deployment-timeout-seconds", type=float, default=1_800)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = deploy_live(build_parser().parse_args(argv))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
