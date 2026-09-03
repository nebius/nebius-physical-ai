"""Idempotent MK8s deployment for the cluster-native Antioch live path."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import stat
import tarfile
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from npa.workflows.byof.openpi_live import LIVE_MANAGED_BY, _certificate

from .live import _relay_certificate

CONFIG_SCHEMA = "npa.antioch.mk8s-live-config.v1"
ANTIOCH_TLS_EGRESS_PORTS = (22, 443, 8443)
UNRESTRICTED_VENDOR_EGRESS_CIDR = "0.0.0.0/0"
MANAGED_BY = "npa-antioch-mk8s-live"
SCENARIO = "openpi_franka_mk8s_live_v2"
_DNS_LABEL = re.compile(r"^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$")
_SECRET_KEY = re.compile(r"^[A-Za-z0-9._-]+$")
_METRIC_KEY = re.compile(r"^[a-z][a-z0-9_]*$")
_ANTIOCH_TERMS_ENV = "NPA_ANTIOCH_ACCEPT_TERMS"


class ClusterLiveError(RuntimeError):
    """The cluster-native desired state could not be reconciled safely."""


class ClusterLiveConfig(BaseModel):
    """Private, owner-readable runtime coordinates; never emitted by the CLI."""

    model_config = ConfigDict(extra="forbid")

    schema_name: Literal[CONFIG_SCHEMA] = CONFIG_SCHEMA
    workflow_run: str = Field(min_length=1, max_length=128)
    state_id: str = Field(min_length=1, max_length=128)
    kubeconfig: str
    context: str = ""
    namespace: str = "workbench"
    adapter_image: str
    policy_selector: dict[str, str]
    policy_gateway_port: int = Field(default=8443, ge=1, le=65535)
    policy_probe_ports: list[int] = Field(default_factory=lambda: [8001, 8002])
    policy_network_policy_name: str
    policy_auth_secret_name: str
    policy_tls_secret_name: str
    policy_cache_pvc_name: str
    public_rollback_service_name: str = ""
    image_pull_secret: str = ""
    antioch_config_dir: str
    antioch_project_id_file: str
    adapter_replicas: int = Field(default=1, ge=0, le=1)
    scenario_timeout_seconds: int = Field(default=14_400, ge=60)
    kubelet_source_cidrs: list[str] = Field(min_length=1)

    @field_validator(
        "namespace",
        "policy_network_policy_name",
        "policy_auth_secret_name",
        "policy_tls_secret_name",
        "policy_cache_pvc_name",
        "public_rollback_service_name",
        "image_pull_secret",
    )
    @classmethod
    def _dns_label(cls, value: str) -> str:
        resolved = value.strip()
        if resolved and (len(resolved) > 63 or not _DNS_LABEL.fullmatch(resolved)):
            raise ValueError("Kubernetes names must be DNS labels")
        return resolved

    @field_validator("adapter_image")
    @classmethod
    def _immutable_image(cls, value: str) -> str:
        resolved = value.strip()
        if "@sha256:" not in resolved:
            raise ValueError("adapter_image must be pinned by sha256 digest")
        return resolved

    @model_validator(mode="after")
    def _identity_and_paths(self) -> "ClusterLiveConfig":
        if not self.policy_selector:
            raise ValueError("policy_selector must not be empty")
        for key, value in self.policy_selector.items():
            if not key.strip() or not value.strip():
                raise ValueError("policy_selector entries must not be empty")
        if len(set(self.policy_probe_ports)) != len(self.policy_probe_ports):
            raise ValueError("policy_probe_ports must be unique")
        return self

    @property
    def identity(self) -> str:
        return hashlib.sha256(
            f"{self.workflow_run}\n{self.state_id}".encode()
        ).hexdigest()[:12]

    @property
    def adapter_name(self) -> str:
        return f"npa-antioch-live-{self.identity}"

    @property
    def policy_service_name(self) -> str:
        return f"npa-openpi-internal-{self.identity}"

    @property
    def live_bundle_secret_name(self) -> str:
        return f"{self.adapter_name}-bundle"

    @property
    def config_secret_name(self) -> str:
        return f"{self.adapter_name}-config"

    @property
    def terms_secret_name(self) -> str:
        return f"{self.adapter_name}-terms"

    @property
    def project_secret_name(self) -> str:
        return f"{self.adapter_name}-project"


def load_private_config(path: Path) -> ClusterLiveConfig:
    if not path.is_file() or path.is_symlink():
        raise ClusterLiveError("private runtime config is not a regular file")
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ClusterLiveError("private runtime config must be mode 0600")
    try:
        return ClusterLiveConfig.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ClusterLiveError("private runtime config is malformed") from exc


def _labels(config: ClusterLiveConfig) -> dict[str, str]:
    return {
        "app.kubernetes.io/name": "npa-antioch-live",
        "app.kubernetes.io/managed-by": MANAGED_BY,
        "app.kubernetes.io/part-of": "antioch-openpi-mk8s-live",
        "npa.nebius.ai/live-identity": config.identity,
    }


def _container_security() -> dict[str, Any]:
    return {
        "runAsUser": 10001,
        "runAsGroup": 10001,
        "runAsNonRoot": True,
        "allowPrivilegeEscalation": False,
        "readOnlyRootFilesystem": True,
        "capabilities": {"drop": ["ALL"]},
    }


def build_public_manifests(config: ClusterLiveConfig) -> dict[str, dict[str, Any]]:
    """Build secret-free desired state suitable for review and apply."""

    labels = _labels(config)
    private_root = "/run/npa-antioch-private"
    state_root = "/var/run/npa-antioch"
    copy_private = (
        "install -d -m 0700 /private/antioch-config /private/live-bundle; "
        # Kubernetes Secret volumes are symlink farms backed by read-only
        # projections. Dereference their regular files into the writable tmpfs;
        # preserving those symlinks makes the later ownership change fail.
        "tar --extract --file /sources/config/config.tar "
        "--directory /private/antioch-config --no-same-owner --no-same-permissions; "
        "cp -L /sources/bundle/* /private/live-bundle/; "
        "install -m 0600 /sources/terms/accepted /private/antioch-terms; "
        "install -m 0600 /sources/project/project-id /private/project-id; "
        "find /private -type d -exec chmod 0700 {} +; "
        "find /private -type f -exec chmod 0600 {} +; "
        "chmod 0700 /state /runtime /runtime-cache; "
        "chown -R 10001:10001 /private /state /runtime /runtime-cache"
    )
    volumes: list[dict[str, Any]] = [
        {"name": "private", "emptyDir": {"medium": "Memory"}},
        {"name": "state", "emptyDir": {}},
        {"name": "runtime", "emptyDir": {}},
        {"name": "runtime-cache", "emptyDir": {}},
        {"name": "tmp", "emptyDir": {}},
        {
            "name": "source-config",
            "secret": {"secretName": config.config_secret_name, "defaultMode": 256},
        },
        {
            "name": "source-bundle",
            "secret": {
                "secretName": config.live_bundle_secret_name,
                "defaultMode": 256,
            },
        },
        {
            "name": "source-terms",
            "secret": {"secretName": config.terms_secret_name, "defaultMode": 256},
        },
        {
            "name": "source-project",
            "secret": {
                "secretName": config.project_secret_name,
                "defaultMode": 256,
            },
        },
    ]
    controller_private_mount = {
        "name": "private",
        "mountPath": private_root,
        "readOnly": False,
    }
    relay_private_mount = {
        "name": "private",
        "mountPath": private_root,
        "readOnly": True,
    }
    controller = {
        "name": "antioch-controller",
        "image": config.adapter_image,
        "imagePullPolicy": "IfNotPresent",
        "command": [
            "python",
            "-m",
            "npa.workbench.antioch.cluster_runtime",
            "run",
            "--scenario",
            SCENARIO,
            "--scenario-timeout-seconds",
            str(config.scenario_timeout_seconds),
            "--owner-identity",
            config.identity,
            "--health-port",
            "18080",
            "--daemon-max-age-seconds",
            "120",
            "--stop-file",
            f"{state_root}/stop",
        ],
        "env": [
            {"name": "ANTIOCH_CONFIG_DIR", "value": f"{private_root}/antioch-config"},
            {
                "name": "NPA_ANTIOCH_RUNTIME_CACHE",
                "value": "/workspace/.cache/npa/antioch",
            },
        ],
        "securityContext": _container_security(),
        "resources": {
            "requests": {"cpu": "500m", "memory": "768Mi"},
            "limits": {"cpu": "2", "memory": "2Gi"},
        },
        "volumeMounts": [
            controller_private_mount,
            {"name": "state", "mountPath": state_root},
            {"name": "runtime", "mountPath": "/var/lib/npa-antioch-live"},
            {"name": "runtime-cache", "mountPath": "/workspace/.cache/npa/antioch"},
            {"name": "tmp", "mountPath": "/tmp"},
        ],
        "ports": [{"name": "ctrl-health", "containerPort": 18080}],
        "readinessProbe": {
            # Null is the strategic-merge tombstone for the former exec probe.
            # Without it, a reconcile leaves both handlers and the API rejects
            # the Deployment before rollout.
            "exec": None,
            "httpGet": {"path": "/ready", "port": "ctrl-health"},
            "periodSeconds": 5,
            "failureThreshold": 3,
        },
        "livenessProbe": {
            "exec": None,
            "httpGet": {"path": "/live", "port": "ctrl-health"},
            "periodSeconds": 10,
            "failureThreshold": 3,
            "initialDelaySeconds": 600,
        },
    }
    relay = {
        "name": "policy-relay",
        "image": config.adapter_image,
        "imagePullPolicy": "IfNotPresent",
        "command": [
            "python",
            "-m",
            "npa.workbench.antioch.relay",
            "--bundle",
            f"{private_root}/live-bundle",
            "--local-port",
            "18444",
            "--stop-file",
            f"{state_root}/stop",
            "--state-path",
            f"{state_root}/relay.json",
            "--owner-identity",
            config.identity,
            "--health-port",
            "18081",
            "--resume-after-stop",
        ],
        "securityContext": _container_security(),
        "resources": {
            "requests": {"cpu": "100m", "memory": "128Mi"},
            "limits": {"cpu": "1", "memory": "1Gi"},
        },
        "volumeMounts": [
            relay_private_mount,
            {"name": "state", "mountPath": state_root},
            {"name": "tmp", "mountPath": "/tmp"},
        ],
        "ports": [{"name": "relay-health", "containerPort": 18081}],
        "readinessProbe": {
            "exec": None,
            "httpGet": {"path": "/ready", "port": "relay-health"},
            "periodSeconds": 5,
            "failureThreshold": 3,
        },
        "livenessProbe": {
            "exec": None,
            "httpGet": {"path": "/live", "port": "relay-health"},
            "periodSeconds": 15,
            "failureThreshold": 6,
        },
    }
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": config.adapter_name,
            "namespace": config.namespace,
            "labels": labels,
        },
        "spec": {
            "replicas": config.adapter_replicas,
            "strategy": {"type": "Recreate"},
            "selector": {"matchLabels": labels},
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "automountServiceAccountToken": False,
                    # Five supported cancellation rounds can each make bounded
                    # list/machine/cancel calls (3 * 60s), followed by the bounded
                    # supervisor and service teardown. Keep SIGKILL outside that path.
                    "terminationGracePeriodSeconds": 1_100,
                    "securityContext": {
                        "runAsNonRoot": True,
                        "runAsUser": 10001,
                        "runAsGroup": 10001,
                        "fsGroup": 10001,
                        "seccompProfile": {"type": "RuntimeDefault"},
                    },
                    "initContainers": [
                        {
                            "name": "stage-private-runtime",
                            "image": config.adapter_image,
                            "command": ["/bin/sh", "-ceu", copy_private],
                            "securityContext": {
                                "runAsUser": 0,
                                "runAsNonRoot": False,
                                "allowPrivilegeEscalation": False,
                                "readOnlyRootFilesystem": True,
                                # The init container copies root-owned Secret
                                # projections into tmpfs, then hands that copy
                                # to the non-root runtime containers. Retain
                                # only the capability needed for that handoff.
                                "capabilities": {
                                    "drop": ["ALL"],
                                    "add": ["CHOWN"],
                                },
                            },
                            "resources": {
                                "requests": {"cpu": "25m", "memory": "32Mi"},
                                "limits": {"cpu": "250m", "memory": "128Mi"},
                            },
                            "volumeMounts": [
                                {"name": "private", "mountPath": "/private"},
                                {"name": "state", "mountPath": "/state"},
                                {"name": "runtime", "mountPath": "/runtime"},
                                {
                                    "name": "runtime-cache",
                                    "mountPath": "/runtime-cache",
                                },
                                {
                                    "name": "source-config",
                                    "mountPath": "/sources/config",
                                    "readOnly": True,
                                },
                                {
                                    "name": "source-bundle",
                                    "mountPath": "/sources/bundle",
                                    "readOnly": True,
                                },
                                {
                                    "name": "source-terms",
                                    "mountPath": "/sources/terms",
                                    "readOnly": True,
                                },
                                {
                                    "name": "source-project",
                                    "mountPath": "/sources/project",
                                    "readOnly": True,
                                },
                                {"name": "tmp", "mountPath": "/tmp"},
                            ],
                        }
                    ],
                    "containers": [controller, relay],
                    "volumes": volumes,
                    **(
                        {"imagePullSecrets": [{"name": config.image_pull_secret}]}
                        if config.image_pull_secret
                        else {}
                    ),
                },
            },
        },
    }
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": config.policy_service_name,
            "namespace": config.namespace,
            "labels": labels,
        },
        "spec": {
            "type": "ClusterIP",
            "selector": config.policy_selector,
            "ports": [
                {
                    "name": "wss",
                    "protocol": "TCP",
                    "port": 443,
                    "targetPort": config.policy_gateway_port,
                }
            ],
        },
    }
    ingress: list[dict[str, Any]] = [
        {
            "from": [{"podSelector": {"matchLabels": labels}}],
            "ports": [{"protocol": "TCP", "port": config.policy_gateway_port}],
        },
        {
            "from": [
                {"ipBlock": {"cidr": cidr}}
                for cidr in sorted(set(config.kubelet_source_cidrs))
            ],
            "ports": [
                {"protocol": "TCP", "port": port}
                for port in sorted(set(config.policy_probe_ports))
            ],
        },
    ]
    policy_network_policy = {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "NetworkPolicy",
        "metadata": {
            "name": config.policy_network_policy_name,
            "namespace": config.namespace,
            "labels": labels,
        },
        "spec": {
            "podSelector": {"matchLabels": config.policy_selector},
            "policyTypes": ["Ingress"],
            "ingress": ingress,
        },
    }
    adapter_network_policy = {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "NetworkPolicy",
        "metadata": {
            "name": config.adapter_name,
            "namespace": config.namespace,
            "labels": labels,
        },
        "spec": {
            "podSelector": {"matchLabels": labels},
            "policyTypes": ["Ingress", "Egress"],
            "ingress": [
                {
                    "from": [
                        {"ipBlock": {"cidr": cidr}}
                        for cidr in sorted(set(config.kubelet_source_cidrs))
                    ],
                    "ports": [
                        {"protocol": "TCP", "port": 18080},
                        {"protocol": "TCP", "port": 18081},
                    ],
                }
            ],
            "egress": [
                {
                    "to": [{"podSelector": {"matchLabels": config.policy_selector}}],
                    "ports": [{"protocol": "TCP", "port": config.policy_gateway_port}],
                },
                {
                    "to": [
                        {
                            "namespaceSelector": {
                                "matchLabels": {
                                    "kubernetes.io/metadata.name": "kube-system"
                                }
                            }
                        }
                    ],
                    "ports": [
                        {"protocol": "UDP", "port": 53},
                        {"protocol": "TCP", "port": 53},
                    ],
                },
                {
                    "to": [{"ipBlock": {"cidr": UNRESTRICTED_VENDOR_EGRESS_CIDR}}],
                    "ports": [
                        {"protocol": "TCP", "port": port}
                        for port in ANTIOCH_TLS_EGRESS_PORTS
                    ],
                },
            ],
        },
    }
    return {
        "policy_service": service,
        "adapter_deployment": deployment,
        "policy_network_policy": policy_network_policy,
        "adapter_network_policy": adapter_network_policy,
    }


def _private_file(path: Path, *, label: str) -> bytes:
    if (
        not path.is_file()
        or path.is_symlink()
        or stat.S_IMODE(path.stat().st_mode) & 0o077
    ):
        raise ClusterLiveError(f"private {label} must be a mode-0600 regular file")
    value = path.read_bytes()
    if not value:
        raise ClusterLiveError(f"private {label} must not be empty")
    return value


def _terms_acceptance() -> bytes:
    """Return an explicit process-scoped attestation; never read it from disk."""
    accepted = os.environ.get(_ANTIOCH_TERMS_ENV, "").encode()
    if accepted != b"YES":
        raise ClusterLiveError(
            "Antioch terms acceptance is not the exact required value"
        )
    return accepted


def _config_archive(directory: Path) -> dict[str, bytes]:
    if (
        not directory.is_dir()
        or directory.is_symlink()
        or stat.S_IMODE(directory.stat().st_mode) & 0o077
    ):
        raise ClusterLiveError("private Antioch config directory must be mode 0700")
    members: list[tuple[Path, Path]] = []
    nonempty_files = 0
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if path.is_symlink() or any(
            not _SECRET_KEY.fullmatch(component) for component in relative.parts
        ):
            raise ClusterLiveError("Antioch config contains an unsafe path")
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:
            raise ClusterLiveError("Antioch config entries must be owner-only")
        if path.is_dir():
            members.append((path, relative))
            continue
        if not path.is_file():
            raise ClusterLiveError(
                "Antioch config must contain only regular files and directories"
            )
        nonempty_files += int(path.stat().st_size > 0)
        members.append((path, relative))
    if not nonempty_files:
        raise ClusterLiveError("private Antioch config directory is empty")
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as output:
        for path, relative in members:
            info = tarfile.TarInfo(relative.as_posix())
            info.mtime = 0
            info.uid = 10001
            info.gid = 10001
            info.uname = ""
            info.gname = ""
            if path.is_dir():
                info.type = tarfile.DIRTYPE
                info.mode = 0o700
                output.addfile(info)
            else:
                value = path.read_bytes()
                info.size = len(value)
                info.mode = 0o600
                output.addfile(info, io.BytesIO(value))
    return {"config.tar": archive.getvalue()}


def _secret(
    name: str, namespace: str, labels: dict[str, str], data: dict[str, bytes]
) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": name, "namespace": namespace, "labels": labels},
        "type": "Opaque",
        "data": {key: base64.b64encode(value).decode() for key, value in data.items()},
    }


def _api_status(exc: Exception) -> int:
    try:
        return int(getattr(exc, "status", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _parse_live_metrics(logs: str) -> dict[str, int | float]:
    latest: dict[str, int | float] = {}
    for line in logs.splitlines():
        marker = "NPA_OPENPI_METRICS "
        if marker not in line:
            continue
        candidate: dict[str, int | float] = {}
        try:
            for item in line.split(marker, 1)[1].split():
                key, separator, value = item.partition("=")
                if separator and _METRIC_KEY.fullmatch(key):
                    candidate[key] = float(value) if "." in value else int(value)
        except ValueError:
            continue
        latest = candidate
    return latest


def qualify_live_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return the fixed, physical live-acceptance gate without identifiers."""

    def number(name: str, default: float = 0.0) -> float:
        try:
            value = float(metrics.get(name, default))
        except (TypeError, ValueError):
            return default
        return value if value == value else default

    requests = int(number("requests"))
    round_trips = int(number("round_trips"))
    checks = {
        "duration": number("elapsed_seconds") >= 930.0,
        "valid_camera_pairs": int(number("frames")) >= 120,
        "policy_round_trips": round_trips >= 100,
        "applied_targets": int(number("applied")) >= 500,
        "finite_action_shape": (
            int(number("action_horizon")) == 15
            and int(number("action_dimension")) == 8
            and int(number("action_finite")) == 1
        ),
        "policy_success_rate": round_trips / max(requests, 1) >= 0.90,
        "no_rejected_actions": all(
            int(number(name)) == 0
            for name in (
                "rejected_wrong_shape",
                "rejected_non_finite",
                "rejected_joint_limit",
                "rejected_gripper_range",
                "rejected_joint_step",
            )
        ),
        "camera_pair_identity": (
            int(number("camera_quality_schema")) == 3
            and int(number("camera_validated_requests")) == requests
            and int(number("camera_pair_id")) == requests
            and int(number("request_camera_pair_id"))
            == int(number("camera_pair_id", -1.0))
            and int(number("round_trip_camera_pair_id"))
            == int(number("request_camera_pair_id", -1.0))
            and int(number("request_render_sequence")) > 0
            and int(number("round_trip_render_sequence"))
            == int(number("request_render_sequence", -1.0))
            and int(number("camera_render_sequence"))
            >= int(number("request_render_sequence", -1.0))
        ),
        "current_camera_quality": all(
            number(name) > threshold
            for name, threshold in (
                ("camera_exterior_luminance_mean_current", 5.0),
                ("camera_exterior_luminance_variance_current", 25.0),
                ("camera_wrist_luminance_mean_current", 5.0),
                ("camera_wrist_luminance_variance_current", 25.0),
            )
        ),
        "accepted_camera_quality": (
            number("luminance_mean_min") > 5.0
            and number("luminance_variance_min") > 25.0
            and number("camera_pair_difference_current") >= 8.0
            and int(number("camera_exterior_red_cube_pixels_current")) >= 20
            and int(number("camera_exterior_cube_in_frame_current")) == 1
            and int(number("camera_wrist_cube_in_frame_current")) == 1
        ),
        "no_safety_projection": (
            int(number("joint_limit_projections", -1.0)) == 0
            and int(number("joint_step_projections", -1.0)) == 0
        ),
        "physical_approach": (
            number("end_effector_cube_approach_m") > 0.0
            and number("end_effector_cube_distance_m", float("inf")) < 0.12
        ),
        "physical_gripper_contact": (
            int(number("gripper_contact_samples")) > 0
            and number("gripper_contact_force_max_n") > 0.1
        ),
        "sustained_pickup": (
            number("cube_lift_max_m") >= 0.05
            and number("pickup_hold_seconds") >= 1.0
            and int(number("pickup_success")) == 1
        ),
        "latency": (
            number("latency_p95_ms", float("inf")) <= 2_000.0
            and number("latency_p99_ms", float("inf")) <= 90_000.0
            and number("latency_max_ms", float("inf")) <= 90_000.0
        ),
        "reconnects": int(number("reconnects", 1_000_000_000.0)) <= 5,
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    return {
        "schema_name": "npa.antioch.live-acceptance.v1",
        "accepted": not failures,
        "checks": checks,
        "failures": failures,
    }


def _read_remote_state(
    stream_fn: Any,
    core: Any,
    *,
    pod_name: str,
    namespace: str,
    path: str,
    container: str = "policy-relay",
    attempts: int = 3,
) -> dict[str, Any]:
    """Converge across transient Kubernetes exec frames without JSON coercion.

    Some Kubernetes exec transports coerce stdout that is exactly one JSON
    object into a language-native mapping representation.  Frame the bytes as
    base64 in the container, then strictly decode and parse them locally so the
    versioned JSON contract survives that transport unchanged.
    """

    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            raw = stream_fn(
                core.connect_get_namespaced_pod_exec,
                pod_name,
                namespace,
                container=container,
                command=["/usr/bin/base64", "-w", "0", path],
                stderr=False,
                stdin=False,
                stdout=True,
                tty=False,
            )
            decoded = base64.b64decode(str(raw).strip(), validate=True)
            parsed = json.loads(decoded)
            if not isinstance(parsed, dict):
                raise TypeError("remote state is not an object")
            return parsed
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(0.05)
    assert last_error is not None
    raise last_error


def _owned(
    metadata: Any,
    identity: str,
    *,
    allow_openpi: bool = False,
    openpi_cleanup_owner: str = "",
) -> bool:
    labels = getattr(metadata, "labels", None) or {}
    if labels.get("npa.nebius.ai/live-identity") == identity:
        return labels.get("app.kubernetes.io/managed-by") == MANAGED_BY
    if not allow_openpi:
        return False
    if labels.get("app.kubernetes.io/managed-by") == LIVE_MANAGED_BY:
        return True
    return bool(
        openpi_cleanup_owner
        and labels.get("npa.nebius.ai/cleanup-owner") == openpi_cleanup_owner
    )


def _apply_owned(
    read: Any,
    create: Any,
    patch: Any,
    *,
    name: str,
    namespace: str,
    body: dict[str, Any],
    identity: str,
    allow_openpi: bool = False,
    openpi_cleanup_owner: str = "",
) -> str:
    try:
        current = read(name=name, namespace=namespace)
    except Exception as exc:
        if _api_status(exc) != 404:
            raise
        create(namespace=namespace, body=body)
        return "created"
    if not _owned(
        current.metadata,
        identity,
        allow_openpi=allow_openpi,
        openpi_cleanup_owner=openpi_cleanup_owner,
    ):
        raise ClusterLiveError("refusing to replace an unowned Kubernetes object")
    patch(name=name, namespace=namespace, body=body)
    return "reconciled"


def _matching_policy_deployments(
    deployments: list[Any], selector: dict[str, str]
) -> list[Any]:
    """Match the pod selector contract, not unrelated Deployment metadata."""
    return [
        deployment
        for deployment in deployments
        if dict(deployment.spec.selector.match_labels or {}) == selector
    ]


def _recover_unready_adapter(
    apps: Any, core: Any, config: ClusterLiveConfig
) -> str:
    """Roll only an exact owned, reconciled adapter whose replica is unhealthy."""

    deployment = apps.read_namespaced_deployment(
        name=config.adapter_name, namespace=config.namespace
    )
    if not _owned(deployment.metadata, config.identity):
        raise ClusterLiveError("adapter Deployment ownership is not proven")
    pods = core.list_namespaced_pod(
        config.namespace,
        label_selector=f"npa.nebius.ai/live-identity={config.identity}",
    ).items
    restart_count = sum(
        int(status.restart_count or 0)
        for pod in pods
        for status in (getattr(pod.status, "container_statuses", None) or [])
    )
    expected_containers = {"antioch-controller", "policy-relay"}
    selected_statuses = [
        status
        for pod in pods
        for status in (getattr(pod.status, "container_statuses", None) or [])
        if str(status.name) in expected_containers
    ]
    selected_containers = [
        container
        for pod in pods
        for container in (getattr(pod.spec, "containers", None) or [])
        if str(container.name) in expected_containers
    ]
    container_ready = {
        str(status.name) for status in selected_statuses
    } == expected_containers and all(bool(status.ready) for status in selected_statuses)
    matching_images = {
        str(container.name) for container in selected_containers
    } == expected_containers and all(
        str(container.image) == config.adapter_image
        for container in selected_containers
    )
    healthy = (
        int(deployment.status.ready_replicas or 0) == 1
        and len(pods) == 1
        and restart_count == 0
        and container_ready
        and matching_images
    )
    if config.adapter_replicas != 1 or healthy:
        return "not_needed"
    generation = hashlib.sha256(
        f"{config.identity}\n{time.time_ns()}".encode()
    ).hexdigest()[:16]
    apps.patch_namespaced_deployment(
        name=config.adapter_name,
        namespace=config.namespace,
        body={
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": {
                            "npa.nebius.ai/owned-recovery-generation": generation
                        }
                    }
                }
            }
        },
    )
    return "rolled_out"


def _policy_placement_status(
    core: Any, policy_deployment: Any, config: ClusterLiveConfig, stream_fn: Any
) -> dict[str, Any]:
    """Return sanitized live policy request, placement, image, and CUDA evidence."""

    selector = dict(policy_deployment.spec.selector.match_labels or {})
    label_selector = ",".join(f"{key}={selector[key]}" for key in sorted(selector))
    pods = core.list_namespaced_pod(
        config.namespace, label_selector=label_selector
    ).items
    container = policy_deployment.spec.template.spec.containers[0]
    requests = dict(getattr(container.resources, "requests", None) or {})
    gpu_request = int(requests.get("nvidia.com/gpu", 0) or 0)
    image = str(container.image or "")
    image_digest = image.rsplit("@", 1)[-1] if "@sha256:" in image else ""
    scheduled = [pod for pod in pods if str(pod.spec.node_name or "")]
    products: set[str] = set()
    capabilities: list[str] = []
    for pod in scheduled:
        node = core.read_node(name=pod.spec.node_name)
        product = str(
            (getattr(node.metadata, "labels", None) or {}).get(
                "nvidia.com/gpu.product", ""
            )
        ).strip()
        if product:
            products.add(product)
        raw = stream_fn(
            core.connect_get_namespaced_pod_exec,
            pod.metadata.name,
            config.namespace,
            container=container.name,
            command=[
                "nvidia-smi",
                "--query-gpu=compute_cap",
                "--format=csv,noheader",
            ],
            stderr=False,
            stdin=False,
            stdout=True,
            tty=False,
        )
        capabilities.extend(
            line.strip() for line in str(raw).splitlines() if line.strip()
        )
    unique_capabilities = sorted(set(capabilities))
    cuda_capability = (
        unique_capabilities[0] if len(unique_capabilities) == 1 else ""
    )
    cuda_sm = ""
    match = re.fullmatch(r"(\d+)\.(\d+)", cuda_capability)
    if match:
        cuda_sm = f"sm_{int(match.group(1))}{int(match.group(2))}"
    return {
        "deployment_replicas": int(policy_deployment.spec.replicas or 0),
        "pod_count": len(pods),
        "scheduled_pod_count": len(scheduled),
        "gpu_request_per_pod": gpu_request,
        "gpu_request_total": gpu_request * len(pods),
        "visible_gpu_count": len(capabilities),
        "gpu_products": sorted(products),
        "cuda_capability": cuda_capability,
        "cuda_sm": cuda_sm,
        "image_digest": image_digest,
    }


def apply_cluster(config: ClusterLiveConfig) -> dict[str, Any]:
    """Stage owner-scoped Secrets, rotate cluster-DNS TLS, and reconcile workloads."""

    from kubernetes import client, config as kube_config

    kube_config.load_kube_config(
        config_file=config.kubeconfig, context=config.context or None
    )
    core = client.CoreV1Api()
    apps = client.AppsV1Api()
    networking = client.NetworkingV1Api()
    core.read_namespace(name=config.namespace)
    deployments = _matching_policy_deployments(
        apps.list_namespaced_deployment(namespace=config.namespace).items,
        config.policy_selector,
    )
    if len(deployments) != 1:
        raise ClusterLiveError("policy selector must resolve exactly one Deployment")
    policy_deployment = deployments[0]
    if not _owned(policy_deployment.metadata, config.identity, allow_openpi=True):
        raise ClusterLiveError("policy Deployment ownership is not proven")
    pvc = core.read_namespaced_persistent_volume_claim(
        name=config.policy_cache_pvc_name, namespace=config.namespace
    )
    if str(getattr(getattr(pvc, "status", None), "phase", "")) != "Bound":
        raise ClusterLiveError("policy checkpoint PVC is not Bound")
    auth = core.read_namespaced_secret(
        name=config.policy_auth_secret_name, namespace=config.namespace
    )
    if not _owned(auth.metadata, config.identity, allow_openpi=True):
        raise ClusterLiveError("policy authentication Secret ownership is not proven")
    encoded_api_key = (auth.data or {}).get("api-key", "")
    try:
        api_key = base64.b64decode(encoded_api_key, validate=True)
    except ValueError as exc:
        raise ClusterLiveError("policy authentication Secret is malformed") from exc
    if len(api_key.strip()) < 32:
        raise ClusterLiveError("policy authentication Secret is malformed")

    labels = _labels(config)
    host = f"{config.policy_service_name}.{config.namespace}.svc"
    bundle_keys = {
        "ca.crt",
        "api-key",
        "endpoint.json",
        "relay-ca.crt",
        "relay-server.crt",
        "relay-server.key",
        "relay-api-key",
    }
    try:
        existing_bundle = core.read_namespaced_secret(
            name=config.live_bundle_secret_name, namespace=config.namespace
        )
    except Exception as exc:
        if _api_status(exc) != 404:
            raise
        existing_bundle = None
    if existing_bundle is not None:
        if not _owned(existing_bundle.metadata, config.identity):
            raise ClusterLiveError("refusing to reuse an unowned live bundle Secret")
        encoded_bundle = existing_bundle.data or {}
        if set(encoded_bundle) != bundle_keys:
            raise ClusterLiveError("existing live bundle Secret has an invalid schema")
        try:
            bundle = {
                key: base64.b64decode(encoded_bundle[key], validate=True)
                for key in bundle_keys
            }
            endpoint = json.loads(bundle["endpoint.json"])
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ClusterLiveError("existing live bundle Secret is malformed") from exc
        if endpoint != {"host": host, "port": 443, "scheme": "wss"}:
            raise ClusterLiveError(
                "existing live bundle targets a different policy Service"
            )
        if bundle["api-key"].strip() != api_key.strip():
            raise ClusterLiveError(
                "existing live bundle does not match policy authentication"
            )
        existing_tls = core.read_namespaced_secret(
            name=config.policy_tls_secret_name, namespace=config.namespace
        )
        if not _owned(existing_tls.metadata, config.identity, allow_openpi=True):
            raise ClusterLiveError("policy TLS Secret ownership is not proven")
        encoded_tls = existing_tls.data or {}
        try:
            certificate = base64.b64decode(encoded_tls["tls.crt"], validate=True)
            private_key = base64.b64decode(encoded_tls["tls.key"], validate=True)
        except (KeyError, ValueError) as exc:
            raise ClusterLiveError("policy TLS Secret is malformed") from exc
        ca = bundle["ca.crt"]
    else:
        ca, certificate, private_key = _certificate(host)
        relay_ca, relay_certificate, relay_key = _relay_certificate()
        relay_token = base64.urlsafe_b64encode(os.urandom(48)).rstrip(b"=")
        bundle = {
            "ca.crt": ca,
            "api-key": api_key.strip() + b"\n",
            "endpoint.json": json.dumps(
                {"scheme": "wss", "host": host, "port": 443}, sort_keys=True
            ).encode()
            + b"\n",
            "relay-ca.crt": relay_ca,
            "relay-server.crt": relay_certificate,
            "relay-server.key": relay_key,
            "relay-api-key": relay_token + b"\n",
        }
    terms = _terms_acceptance()
    project_id = _private_file(
        Path(config.antioch_project_id_file), label="Antioch project identity"
    )
    secrets = [
        _secret(
            config.config_secret_name,
            config.namespace,
            labels,
            _config_archive(Path(config.antioch_config_dir)),
        ),
        _secret(
            config.terms_secret_name,
            config.namespace,
            labels,
            {"accepted": terms.strip() + b"\n"},
        ),
        _secret(
            config.project_secret_name,
            config.namespace,
            labels,
            {"project-id": project_id.strip() + b"\n"},
        ),
        _secret(config.live_bundle_secret_name, config.namespace, labels, bundle),
    ]
    actions: dict[str, str] = {}
    for body in secrets:
        name = body["metadata"]["name"]
        actions[f"secret:{name.rsplit('-', 1)[-1]}"] = _apply_owned(
            core.read_namespaced_secret,
            core.create_namespaced_secret,
            core.patch_namespaced_secret,
            name=name,
            namespace=config.namespace,
            body=body,
            identity=config.identity,
        )

    tls_body = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": config.policy_tls_secret_name,
            "namespace": config.namespace,
            "labels": dict(getattr(auth.metadata, "labels", None) or {}),
        },
        "type": "kubernetes.io/tls",
        "data": {
            "tls.crt": base64.b64encode(certificate).decode(),
            "tls.key": base64.b64encode(private_key).decode(),
        },
    }
    actions["policy_tls"] = _apply_owned(
        core.read_namespaced_secret,
        core.create_namespaced_secret,
        core.patch_namespaced_secret,
        name=config.policy_tls_secret_name,
        namespace=config.namespace,
        body=tls_body,
        identity=config.identity,
        allow_openpi=True,
    )
    manifests = build_public_manifests(config)
    service = manifests["policy_service"]
    actions["policy_service"] = _apply_owned(
        core.read_namespaced_service,
        core.create_namespaced_service,
        core.patch_namespaced_service,
        name=service["metadata"]["name"],
        namespace=config.namespace,
        body=service,
        identity=config.identity,
    )
    policy_np = manifests["policy_network_policy"]
    actions["policy_network_policy"] = _apply_owned(
        networking.read_namespaced_network_policy,
        networking.create_namespaced_network_policy,
        networking.patch_namespaced_network_policy,
        name=policy_np["metadata"]["name"],
        namespace=config.namespace,
        body=policy_np,
        identity=config.identity,
        allow_openpi=True,
        openpi_cleanup_owner=config.policy_selector.get(
            "npa.nebius.ai/cleanup-owner", ""
        ),
    )
    annotations = {
        "npa.nebius.ai/cluster-live-tls-sha256": hashlib.sha256(
            ca + b"\0" + certificate + b"\0" + private_key
        ).hexdigest()
    }
    apps.patch_namespaced_deployment(
        name=policy_deployment.metadata.name,
        namespace=config.namespace,
        body={"spec": {"template": {"metadata": {"annotations": annotations}}}},
    )
    adapter = manifests["adapter_deployment"]
    actions["adapter_deployment"] = _apply_owned(
        apps.read_namespaced_deployment,
        apps.create_namespaced_deployment,
        apps.patch_namespaced_deployment,
        name=adapter["metadata"]["name"],
        namespace=config.namespace,
        body=adapter,
        identity=config.identity,
    )
    if actions["adapter_deployment"] == "reconciled":
        actions["adapter_recovery"] = _recover_unready_adapter(apps, core, config)
    adapter_np = manifests["adapter_network_policy"]
    actions["adapter_network_policy"] = _apply_owned(
        networking.read_namespaced_network_policy,
        networking.create_namespaced_network_policy,
        networking.patch_namespaced_network_policy,
        name=adapter_np["metadata"]["name"],
        namespace=config.namespace,
        body=adapter_np,
        identity=config.identity,
    )
    return {
        "status": "reconciled",
        "identity": config.identity,
        "actions": actions,
        "policy_service_type": "ClusterIP",
        "transport": "same-pod-antioch-tunnel-to-cluster-local-policy",
        "dev_vm_in_data_path": False,
        "credentials_emitted": False,
    }


def cluster_status(config: ClusterLiveConfig) -> dict[str, Any]:
    from kubernetes import client, config as kube_config
    from kubernetes.client.exceptions import ApiException
    from kubernetes.stream import stream

    from .cluster_runtime import _state_ready

    kube_config.load_kube_config(
        config_file=config.kubeconfig, context=config.context or None
    )
    apps = client.AppsV1Api()
    core = client.CoreV1Api()
    deployment = apps.read_namespaced_deployment(config.adapter_name, config.namespace)
    if not _owned(deployment.metadata, config.identity):
        raise ClusterLiveError("adapter Deployment ownership is not proven")
    pods = core.list_namespaced_pod(
        config.namespace,
        label_selector=f"npa.nebius.ai/live-identity={config.identity}",
    ).items
    restart_count = sum(
        int(status.restart_count or 0)
        for pod in pods
        for status in (getattr(pod.status, "container_statuses", None) or [])
    )
    container_states: dict[str, dict[str, Any]] = {}
    for pod in pods:
        for container_status in getattr(pod.status, "container_statuses", None) or []:
            state = getattr(container_status, "state", None)
            current_reason = ""
            for phase in ("waiting", "terminated", "running"):
                detail = getattr(state, phase, None)
                if detail is not None:
                    current_reason = str(getattr(detail, "reason", "") or phase)
                    break
            last_terminated = getattr(
                getattr(container_status, "last_state", None), "terminated", None
            )
            container_states[str(container_status.name)] = {
                "ready": bool(container_status.ready),
                "restart_count": int(container_status.restart_count or 0),
                "state": current_reason,
                "last_exit_code": (
                    int(last_terminated.exit_code)
                    if last_terminated is not None
                    else None
                ),
                "last_reason": (
                    str(last_terminated.reason or "")
                    if last_terminated is not None
                    else ""
                ),
            }
    kubernetes_ready = (
        int(getattr(deployment.status, "ready_replicas", 0) or 0) == 1
        and len(pods) == 1
    )
    policy = _matching_policy_deployments(
        apps.list_namespaced_deployment(config.namespace).items,
        config.policy_selector,
    )
    if len(policy) != 1 or not _owned(
        policy[0].metadata, config.identity, allow_openpi=True
    ):
        raise ClusterLiveError("policy Deployment ownership is not proven")
    policy_ready = int(policy[0].status.ready_replicas or 0) == 1
    policy_placement: dict[str, Any] = {}
    relay_state: dict[str, Any] = {}
    controller_state: dict[str, Any] = {}
    probe_diagnostics: dict[str, dict[str, str]] = {}
    live_metrics: dict[str, int | float] = {}
    cluster_local_policy_resolved = False
    checkpoint_cache_bound = False
    try:
        policy_placement = _policy_placement_status(core, policy[0], config, stream)
    except Exception as exc:
        probe_diagnostics["policy_placement"] = {
            "status": "failed",
            "exception_class": type(exc).__name__,
        }
    try:
        checkpoint_cache_bound = (
            str(
                core.read_namespaced_persistent_volume_claim(
                    name=config.policy_cache_pvc_name, namespace=config.namespace
                ).status.phase
            )
            == "Bound"
        )
    except Exception as exc:
        probe_diagnostics["checkpoint_cache"] = {
            "status": "failed",
            "exception_class": type(exc).__name__,
        }
    if len(pods) == 1:
        pod_name = pods[0].metadata.name
        try:
            parsed_controller = _read_remote_state(
                stream,
                core,
                pod_name=pod_name,
                namespace=config.namespace,
                path="/var/run/npa-antioch/controller.json",
                container="policy-relay",
            )
            controller_allowed = {
                "schema",
                "schema_version",
                "status",
                "daemon_status",
                "owner_identity",
                "session_id",
                "scenario",
                "scenario_run_id",
                "run_phase",
                "stream_state",
                "heartbeat_unix",
                "recoveries",
                "error_type",
                "recovery_reason",
                "vendor_exit_class",
                "vendor_exit_code",
                "vendor_process_status",
                "vendor_output_bytes",
                "vendor_output_age_seconds",
                "controller_pid",
                "vendor_pid",
                "vendor_parent_pid",
                "vendor_process_group_isolated",
                "daemon_guest_state",
                "daemon_observed_at",
                "rome_guest_observed_at",
                "scenario_session_leases",
                "process_leases",
                "stream_leases",
                "transport",
                "dev_vm_in_data_path",
            }
            controller_state = {
                key: parsed_controller.get(key) for key in sorted(controller_allowed)
            }
            heartbeat = parsed_controller.get("heartbeat_unix")
            controller_state["heartbeat_age_seconds"] = (
                round(max(0.0, time.time() - float(heartbeat)), 3)
                if isinstance(heartbeat, (int, float))
                and not isinstance(heartbeat, bool)
                else None
            )
        except Exception as exc:
            controller_state = {"status": "unavailable"}
            probe_diagnostics["controller_state"] = {
                "status": "failed",
                "exception_class": type(exc).__name__,
            }
        try:
            parsed = _read_remote_state(
                stream,
                core,
                pod_name=pod_name,
                namespace=config.namespace,
                path="/var/run/npa-antioch/relay.json",
                container="antioch-controller",
            )
            allowed = {
                "schema",
                "schema_version",
                "owner_identity",
                "heartbeat_unix",
                "status",
                "connections",
                "reconnects",
                "forwarded_requests",
                "failures",
                "last_round_trip_ms",
                "last_error_type",
                "last_failed_phase",
            }
            relay_state = {key: parsed.get(key) for key in sorted(allowed)}
            heartbeat = parsed.get("heartbeat_unix")
            relay_state["heartbeat_age_seconds"] = (
                round(max(0.0, time.time() - float(heartbeat)), 3)
                if isinstance(heartbeat, (int, float))
                and not isinstance(heartbeat, bool)
                else None
            )
        except Exception as exc:
            relay_state = {"status": "unavailable"}
            probe_diagnostics["relay_state"] = {
                "status": "failed",
                "exception_class": type(exc).__name__,
            }
        try:
            logs = core.read_namespaced_pod_log(
                pod_name,
                config.namespace,
                container="antioch-controller",
                tail_lines=2_000,
                timestamps=False,
            )
            live_metrics = _parse_live_metrics(str(logs))
        except ApiException:
            live_metrics = {}
        try:
            resolution = stream(
                core.connect_get_namespaced_pod_exec,
                pod_name,
                config.namespace,
                container="policy-relay",
                command=[
                    "python",
                    "-c",
                    (
                        "import json,socket;"
                        "p=json.load(open('/run/npa-antioch-private/live-bundle/endpoint.json'));"
                        "h=str(p['host']);"
                        "assert h.endswith('.svc') and socket.getaddrinfo(h,443);"
                        "print('ok')"
                    ),
                ],
                stderr=False,
                stdin=False,
                stdout=True,
                tty=False,
            )
            cluster_local_policy_resolved = str(resolution).strip() == "ok"
        except Exception as exc:
            cluster_local_policy_resolved = False
            probe_diagnostics["policy_dns"] = {
                "status": "failed",
                "exception_class": type(exc).__name__,
            }
    controller_ready = _state_ready(
        controller_state,
        component="controller",
        expected_owner_identity=config.identity,
        max_age_seconds=30.0,
    )
    relay_ready = _state_ready(
        relay_state,
        component="relay",
        expected_owner_identity=config.identity,
        max_age_seconds=150.0,
    )
    ready = kubernetes_ready and controller_ready and relay_ready and policy_ready
    return {
        "status": "ready" if ready else "not_ready",
        "identity": config.identity,
        "adapter_ready": ready,
        "kubernetes_ready": kubernetes_ready,
        "daemon_liveness_ready": controller_ready,
        "relay_liveness_ready": relay_ready,
        "policy_ready": policy_ready,
        "policy_placement": policy_placement,
        "checkpoint_cache_bound": checkpoint_cache_bound,
        "adapter_image_digest": config.adapter_image.rsplit("@", 1)[-1],
        "adapter_pods": len(pods),
        "adapter_restarts": restart_count,
        "adapter_container_states": container_states,
        "controller": controller_state,
        "policy_service_type": "ClusterIP",
        "cluster_local_policy_resolved": cluster_local_policy_resolved,
        "relay": relay_state,
        "live_metrics": live_metrics,
        "live_acceptance": qualify_live_metrics(live_metrics),
        "probe_diagnostics": probe_diagnostics,
        "dev_vm_in_data_path": False,
    }


def stop_cluster(
    config: ClusterLiveConfig, *, timeout_seconds: float = 1_200.0
) -> dict[str, Any]:
    from kubernetes import client, config as kube_config
    from kubernetes.stream import stream

    kube_config.load_kube_config(
        config_file=config.kubeconfig, context=config.context or None
    )
    apps = client.AppsV1Api()
    core = client.CoreV1Api()
    deployment = apps.read_namespaced_deployment(config.adapter_name, config.namespace)
    if not _owned(deployment.metadata, config.identity):
        raise ClusterLiveError("refusing to stop an unowned adapter Deployment")
    pods = core.list_namespaced_pod(
        config.namespace,
        label_selector=f"npa.nebius.ai/live-identity={config.identity}",
    ).items
    if len(pods) != 1:
        raise ClusterLiveError(
            "exact adapter pod is absent or ambiguous; remote cleanup is unproven"
        )
    pod_name = pods[0].metadata.name
    stream(
        core.connect_get_namespaced_pod_exec,
        pod_name,
        config.namespace,
        container="antioch-controller",
        command=["/usr/bin/touch", "/var/run/npa-antioch/stop"],
        stderr=False,
        stdin=False,
        stdout=True,
        tty=False,
    )
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            cleanup_state = _read_remote_state(
                stream,
                core,
                pod_name=pod_name,
                namespace=config.namespace,
                container="antioch-controller",
                path="/var/run/npa-antioch/controller.json",
            )
            cleanup_status = str(cleanup_state.get("status") or "")
        except Exception:
            # Kubernetes exec can yield an empty or partial stdout frame while
            # the controller atomically replaces its state file.  A single
            # malformed read is not affirmative cleanup evidence, so keep
            # polling within the caller's existing timeout and still fail
            # closed if a complete supported state never arrives.
            time.sleep(2)
            continue
        if cleanup_status == "cleanup_failed":
            raise ClusterLiveError(
                "supported remote scenario/service cleanup failed; Deployment retained"
            )
        if cleanup_status == "stopped":
            apps.patch_namespaced_deployment_scale(
                config.adapter_name, config.namespace, {"spec": {"replicas": 0}}
            )
            return {
                "status": "stopped",
                "identity": config.identity,
                "cleanup_order": "scenario_then_service",
                "remote_terminal_evidence": "supported-controller-cleanup",
            }
        time.sleep(2)
    raise ClusterLiveError("adapter did not finish supported scenario/service cleanup")


def disable_public_rollback_service(config: ClusterLiveConfig) -> dict[str, Any]:
    from kubernetes import client, config as kube_config

    if not config.public_rollback_service_name:
        return {"status": "not_configured"}
    kube_config.load_kube_config(
        config_file=config.kubeconfig, context=config.context or None
    )
    core = client.CoreV1Api()
    service = core.read_namespaced_service(
        config.public_rollback_service_name, config.namespace
    )
    if not _owned(service.metadata, config.identity, allow_openpi=True):
        raise ClusterLiveError("refusing to alter an unowned rollback Service")
    if str(service.spec.type) != "LoadBalancer":
        return {"status": "already_private"}
    core.delete_namespaced_service(
        config.public_rollback_service_name, config.namespace
    )
    return {"status": "disabled", "former_type": "LoadBalancer"}
