"""Own the Kubernetes prerequisites for a clean-start Sim2Real policy run."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import re
import secrets
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from npa.workflows.sim2real.smolvla_service_lifecycle import resource_names
from npa.workflows.sim2real.workflow_io import write_json

MANAGED_BY = "npa-sim2real-runtime-prerequisites"
SCHEMA = "npa.sim2real.runtime_prerequisites.v1"
_CONTAINER_TMP_MOUNT_PATH = PurePosixPath("/").joinpath("tmp").as_posix()


class RuntimePrerequisiteError(RuntimeError):
    """Report an unsafe or incomplete runtime-prerequisite operation."""


def _slug(run_id: str) -> str:
    value = re.sub(r"[^a-z0-9-]+", "-", run_id.lower()).strip("-")
    if not value:
        raise RuntimePrerequisiteError("run id must contain a DNS-safe character")
    if value != run_id or len(value) > 180:
        raise RuntimePrerequisiteError(
            "run id must already be a lowercase DNS label of at most 180 characters"
        )
    # The longest prerequisite suffix is ``-policy-cache`` (13 characters),
    # while ``npa-smolvla-`` consumes another 12.  Keep the variable stem at
    # most 38 characters so every generated name is also safe when copied into
    # the warm Job's 63-character Kubernetes label value.  The hash prevents
    # distinct long run IDs with the same readable prefix from colliding.
    if len(value) > 38:
        digest = hashlib.sha256(run_id.encode()).hexdigest()[:8]
        value = f"{value[:29].rstrip('-')}-{digest}"
    return value


def resource_identities(run_id: str) -> dict[str, str]:
    """Return deterministic names for one workflow-owned prerequisite set."""

    stem = f"npa-smolvla-{_slug(run_id)}"
    return {
        "isaac_cache_pvc": f"{stem}-isaac-cache",
        "policy_cache_pvc": f"{stem}-policy-cache",
        "auth_secret": f"{stem}-auth",
        "base_service_account": f"{stem}-base-ctl",
        "adapted_service_account": f"{stem}-adapted-ctl",
        "isaac_warm_job": f"{stem}-isaac-warm",
    }


def _metadata(name: str, namespace: str, run_id: str) -> dict[str, Any]:
    return {
        "name": name,
        "namespace": namespace,
        "labels": {
            "app.kubernetes.io/managed-by": MANAGED_BY,
            "npa.nebius.ai/run-hash": hashlib.sha256(run_id.encode()).hexdigest()[:16],
        },
        "annotations": {"npa.nebius.ai/cleanup-owner": run_id},
    }


def _controller_rules(run_id: str, role: str) -> list[dict[str, Any]]:
    names = resource_names(run_id, role)
    return [
        {"apiGroups": [""], "resources": ["pods"], "verbs": ["list"]},
        {"apiGroups": [""], "resources": ["services", "secrets"], "verbs": ["create"]},
        {
            "apiGroups": [""],
            "resources": ["services"],
            "resourceNames": [names["service"]],
            "verbs": ["get", "delete"],
        },
        {
            "apiGroups": [""],
            "resources": ["secrets"],
            "resourceNames": [names["credentials"]],
            "verbs": ["get", "delete"],
        },
        {"apiGroups": ["apps"], "resources": ["deployments"], "verbs": ["create"]},
        {
            "apiGroups": ["apps"],
            "resources": ["deployments"],
            "resourceNames": [names["deployment"]],
            "verbs": ["get", "delete"],
        },
    ]


def build_manifests(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    """Build the exact run-owned PVC, secret, RBAC, and cache-warm manifests."""

    expected = resource_identities(args.run_id)
    for key, value in expected.items():
        if getattr(args, key) != value:
            raise RuntimePrerequisiteError(
                f"{key} must be the deterministic run-owned name {value!r}"
            )
    if not re.fullmatch(r"[^@\s]+@sha256:[0-9a-f]{64}", args.isaac_image):
        raise RuntimePrerequisiteError("Isaac image must be registry digest-pinned")
    if args.storage_access_mode not in {"ReadWriteOnce", "ReadWriteMany"}:
        raise RuntimePrerequisiteError(
            "storage access mode must be ReadWriteOnce or ReadWriteMany"
        )
    manifests: dict[str, dict[str, Any]] = {}
    for purpose, size in (
        ("isaac_cache", args.isaac_cache_size),
        ("policy_cache", args.policy_cache_size),
    ):
        name = getattr(args, f"{purpose}_pvc")
        manifests[f"{purpose}_pvc"] = {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": _metadata(name, args.namespace, args.run_id),
            "spec": {
                "accessModes": [args.storage_access_mode],
                "storageClassName": args.storage_class,
                "resources": {"requests": {"storage": size}},
            },
        }
    manifests["auth_secret"] = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": _metadata(args.auth_secret, args.namespace, args.run_id),
        "type": "Opaque",
        "data": {"auth_token": base64.b64encode(secrets.token_bytes(32)).decode()},
    }
    for role in ("base", "adapted"):
        account = getattr(args, f"{role}_service_account")
        metadata = _metadata(account, args.namespace, args.run_id)
        manifests[f"{role}_service_account"] = {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": metadata,
            "automountServiceAccountToken": True,
        }
        manifests[f"{role}_role"] = {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "Role",
            "metadata": metadata,
            "rules": _controller_rules(args.run_id, role),
        }
        manifests[f"{role}_role_binding"] = {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "RoleBinding",
            "metadata": metadata,
            "subjects": [
                {"kind": "ServiceAccount", "name": account, "namespace": args.namespace}
            ],
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "Role",
                "name": account,
            },
        }
    manifests["isaac_warm_job"] = _warm_job(args)
    return manifests


def _warm_job(args: argparse.Namespace) -> dict[str, Any]:
    metadata = _metadata(args.isaac_warm_job, args.namespace, args.run_id)
    metadata["labels"]["app"] = args.isaac_warm_job
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": metadata,
        "spec": {
            "backoffLimit": 2,
            "template": {
                "metadata": {"labels": dict(metadata["labels"])},
                "spec": {
                    "restartPolicy": "Never",
                    "securityContext": {
                        "runAsNonRoot": True,
                        "runAsUser": 1000,
                        "runAsGroup": 1000,
                        "fsGroup": 1000,
                        "fsGroupChangePolicy": "OnRootMismatch",
                        "seccompProfile": {"type": "RuntimeDefault"},
                    },
                    "containers": [
                        {
                            "name": "warm",
                            "image": args.isaac_image,
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["/bin/bash", "-ec"],
                            "args": [
                                "/opt/npa/bin/isaac-bootstrap warm\nexec /opt/npa/bin/isaac-bootstrap status"
                            ],
                            "env": [
                                {"name": "ACCEPT_EULA", "value": "Y"},
                                {"name": "OMNI_KIT_ACCEPT_EULA", "value": "YES"},
                                {"name": "ISAACSIM_ACCEPT_EULA", "value": "YES"},
                                {
                                    "name": "NPA_ISAAC_CACHE_DIR",
                                    "value": "/opt/isaac-cache",
                                },
                            ],
                            "securityContext": {
                                "readOnlyRootFilesystem": True,
                                "allowPrivilegeEscalation": False,
                                "runAsNonRoot": True,
                                "capabilities": {"drop": ["ALL"]},
                            },
                            "resources": {
                                "requests": {
                                    "cpu": "2",
                                    "memory": "8Gi",
                                    "ephemeral-storage": "8Gi",
                                },
                                "limits": {
                                    "cpu": "8",
                                    "memory": "16Gi",
                                    "ephemeral-storage": "16Gi",
                                },
                            },
                            "volumeMounts": [
                                {
                                    "name": "isaac-cache",
                                    "mountPath": "/opt/isaac-cache",
                                },
                                {
                                    "name": "tmp",
                                    "mountPath": _CONTAINER_TMP_MOUNT_PATH,
                                },
                            ],
                        }
                    ],
                    "volumes": [
                        {
                            "name": "isaac-cache",
                            "persistentVolumeClaim": {
                                "claimName": args.isaac_cache_pvc
                            },
                        },
                        {"name": "tmp", "emptyDir": {"sizeLimit": "12Gi"}},
                    ],
                },
            },
        },
    }


def _clients() -> tuple[Any, Any, Any, Any]:
    from kubernetes import client, config

    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()
    return (
        client.CoreV1Api(),
        client.RbacAuthorizationV1Api(),
        client.BatchV1Api(),
        client.AppsV1Api(),
    )


def _status(error: Exception) -> int | None:
    return getattr(error, "status", None)


def _read(reader: Callable[..., Any], name: str, namespace: str) -> Any | None:
    try:
        return reader(name, namespace, _request_timeout=30)
    except Exception as error:
        if _status(error) == 404:
            return None
        raise


def _assert_owned(actual: Any, manifest: dict[str, Any], run_id: str) -> None:
    if isinstance(actual, dict):
        metadata = actual.get("metadata") or {}
        labels = metadata.get("labels") or {}
        annotations = metadata.get("annotations") or {}
    else:
        metadata = getattr(actual, "metadata", None)
        labels = getattr(metadata, "labels", None) or {}
        annotations = getattr(metadata, "annotations", None) or {}
    if (
        labels.get("app.kubernetes.io/managed-by") != MANAGED_BY
        or annotations.get("npa.nebius.ai/cleanup-owner") != run_id
    ):
        raise RuntimePrerequisiteError(
            f"refusing foreign Kubernetes resource {manifest['metadata']['name']!r}"
        )


def _serialized(actual: Any) -> dict[str, Any]:
    if isinstance(actual, dict):
        return actual
    from kubernetes import client

    value = client.ApiClient().sanitize_for_serialization(actual)
    if not isinstance(value, dict):
        raise RuntimePrerequisiteError("Kubernetes API returned a non-object resource")
    return value


def _normalized_rules(rules: list[dict[str, Any]]) -> list[str]:
    return sorted(json.dumps(rule, sort_keys=True) for rule in rules)


def _assert_contract(key: str, actual: Any, desired: dict[str, Any]) -> None:
    value = _serialized(actual)
    if key.endswith("_pvc"):
        actual_spec = value.get("spec") or {}
        desired_spec = desired["spec"]
        actual_request = (
            (actual_spec.get("resources") or {}).get("requests") or {}
        ).get("storage")
        desired_request = desired_spec["resources"]["requests"]["storage"]
        matches = (
            actual_spec.get("accessModes") == desired_spec["accessModes"]
            and actual_spec.get("storageClassName") == desired_spec["storageClassName"]
            and actual_request == desired_request
        )
    elif key == "auth_secret":
        encoded = (value.get("data") or {}).get("auth_token", "")
        try:
            token = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError, TypeError):
            token = b""
        matches = value.get("type") == "Opaque" and len(token) == 32
    elif key.endswith("_service_account"):
        matches = value.get("automountServiceAccountToken") is True
    elif key.endswith("_role"):
        matches = _normalized_rules(value.get("rules") or []) == _normalized_rules(
            desired["rules"]
        )
    elif key.endswith("_role_binding"):
        matches = (
            value.get("subjects") == desired["subjects"]
            and value.get("roleRef") == desired["roleRef"]
        )
    elif key == "isaac_warm_job":
        actual_spec = value.get("spec") or {}
        desired_spec = desired["spec"]
        actual_pod = (actual_spec.get("template") or {}).get("spec") or {}
        desired_pod = desired_spec["template"]["spec"]
        actual_container = (actual_pod.get("containers") or [{}])[0]
        desired_container = desired_pod["containers"][0]
        matches = (
            all(
                actual_container.get(field) == desired_container[field]
                for field in (
                    "image",
                    "command",
                    "args",
                    "env",
                    "securityContext",
                    "resources",
                    "volumeMounts",
                )
            )
            and actual_pod.get("volumes") == desired_pod["volumes"]
            and actual_pod.get("securityContext") == desired_pod["securityContext"]
            and actual_pod.get("restartPolicy") == "Never"
            and actual_spec.get("backoffLimit") == desired_spec["backoffLimit"]
        )
    else:
        matches = True
    if not matches:
        raise RuntimePrerequisiteError(f"owned Kubernetes resource {key!r} has drifted")


def _objects(core: Any, rbac: Any, batch: Any) -> dict[str, tuple[Any, Any, Any]]:
    return {
        "isaac_cache_pvc": (
            core.read_namespaced_persistent_volume_claim,
            core.create_namespaced_persistent_volume_claim,
            core.delete_namespaced_persistent_volume_claim,
        ),
        "policy_cache_pvc": (
            core.read_namespaced_persistent_volume_claim,
            core.create_namespaced_persistent_volume_claim,
            core.delete_namespaced_persistent_volume_claim,
        ),
        "auth_secret": (
            core.read_namespaced_secret,
            core.create_namespaced_secret,
            core.delete_namespaced_secret,
        ),
        "base_service_account": (
            core.read_namespaced_service_account,
            core.create_namespaced_service_account,
            core.delete_namespaced_service_account,
        ),
        "adapted_service_account": (
            core.read_namespaced_service_account,
            core.create_namespaced_service_account,
            core.delete_namespaced_service_account,
        ),
        "base_role": (
            rbac.read_namespaced_role,
            rbac.create_namespaced_role,
            rbac.delete_namespaced_role,
        ),
        "adapted_role": (
            rbac.read_namespaced_role,
            rbac.create_namespaced_role,
            rbac.delete_namespaced_role,
        ),
        "base_role_binding": (
            rbac.read_namespaced_role_binding,
            rbac.create_namespaced_role_binding,
            rbac.delete_namespaced_role_binding,
        ),
        "adapted_role_binding": (
            rbac.read_namespaced_role_binding,
            rbac.create_namespaced_role_binding,
            rbac.delete_namespaced_role_binding,
        ),
        "isaac_warm_job": (
            batch.read_namespaced_job,
            batch.create_namespaced_job,
            batch.delete_namespaced_job,
        ),
    }


def _wait_for_warm(batch: Any, args: argparse.Namespace) -> Any:
    deadline = time.monotonic() + args.warm_timeout_seconds
    while time.monotonic() < deadline:
        job = batch.read_namespaced_job(
            args.isaac_warm_job, args.namespace, _request_timeout=30
        )
        if int(getattr(job.status, "succeeded", 0) or 0) >= 1:
            return job
        if int(getattr(job.status, "failed", 0) or 0) > 2:
            raise RuntimePrerequisiteError("Isaac cache warm Job exhausted its retries")
        time.sleep(args.poll_interval_seconds)
    raise RuntimePrerequisiteError("Isaac cache warm Job timed out")


def _apply(
    args: argparse.Namespace, manifests: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    core, rbac, batch, _apps = _clients()
    created: list[str] = []
    reused: list[str] = []
    for key, (reader, creator, _deleter) in _objects(core, rbac, batch).items():
        manifest = manifests[key]
        name = manifest["metadata"]["name"]
        actual = _read(reader, name, args.namespace)
        if actual is not None:
            _assert_owned(actual, manifest, args.run_id)
            _assert_contract(key, actual, manifest)
            reused.append(key)
            continue
        creator(args.namespace, manifest, _request_timeout=30)
        created.append(key)
    job = _wait_for_warm(batch, args)
    return _receipt(args, "apply", created, reused, job)


def _assert_service_owned(actual: Any, run_id: str, role: str) -> None:
    names = resource_names(run_id, role)
    metadata = getattr(actual, "metadata", None)
    labels = getattr(metadata, "labels", None) or {}
    expected = {
        "app.kubernetes.io/managed-by": "npa",
        "npa.nebius.ai/cleanup-owner": names["deployment"],
        "npa.nebius.ai/run-hash": hashlib.sha256(run_id.encode()).hexdigest()[:16],
        "npa.nebius.ai/checkpoint-role": role,
    }
    if any(labels.get(key) != value for key, value in expected.items()):
        raise RuntimePrerequisiteError(
            f"refusing foreign or drifted {role} SmolVLA service resource"
        )


def _wait_for_service_pods_absent(
    core: Any, args: argparse.Namespace, role: str
) -> None:
    deployment = resource_names(args.run_id, role)["deployment"]
    deadline = time.monotonic() + min(args.warm_timeout_seconds, 600.0)
    while time.monotonic() < deadline:
        pods = core.list_namespaced_pod(
            args.namespace,
            label_selector=f"app={deployment}",
            _request_timeout=30,
        )
        if not list(getattr(pods, "items", None) or []):
            return
        time.sleep(args.poll_interval_seconds)
    raise RuntimePrerequisiteError(f"{role} SmolVLA pods did not terminate")


def _cleanup_smolvla_services(core: Any, apps: Any, args: argparse.Namespace) -> None:
    for role in ("base", "adapted"):
        names = resource_names(args.run_id, role)
        found = False
        targets = (
            (
                apps.read_namespaced_deployment,
                apps.delete_namespaced_deployment,
                names["deployment"],
            ),
            (
                core.read_namespaced_service,
                core.delete_namespaced_service,
                names["service"],
            ),
            (
                core.read_namespaced_secret,
                core.delete_namespaced_secret,
                names["credentials"],
            ),
        )
        for reader, deleter, name in targets:
            actual = _read(reader, name, args.namespace)
            if actual is None:
                continue
            found = True
            _assert_service_owned(actual, args.run_id, role)
            deleter(name, args.namespace, _request_timeout=30)
        if found:
            _wait_for_service_pods_absent(core, args, role)


def _delete(
    args: argparse.Namespace, manifests: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    core, rbac, batch, apps = _clients()
    _cleanup_smolvla_services(core, apps, args)
    deleted: list[str] = []
    absent: list[str] = []
    objects = _objects(core, rbac, batch)
    order = [
        "isaac_warm_job",
        "base_role_binding",
        "adapted_role_binding",
        "base_role",
        "adapted_role",
        "base_service_account",
        "adapted_service_account",
        "auth_secret",
        "policy_cache_pvc",
        "isaac_cache_pvc",
    ]
    for key in order:
        reader, _creator, deleter = objects[key]
        manifest = manifests[key]
        name = manifest["metadata"]["name"]
        actual = _read(reader, name, args.namespace)
        if actual is None:
            absent.append(key)
            continue
        _assert_owned(actual, manifest, args.run_id)
        if key == "isaac_warm_job":
            # A background/default Job deletion can orphan its completed pod.
            # Cleanup then waits forever for a pod whose controller has already
            # disappeared. Foreground propagation keeps the Job as the owner
            # until Kubernetes has removed its exact dependent pod.
            deleter(
                name,
                args.namespace,
                body={
                    "apiVersion": "v1",
                    "kind": "DeleteOptions",
                    "propagationPolicy": "Foreground",
                },
                _request_timeout=30,
            )
        else:
            deleter(name, args.namespace, _request_timeout=30)
        deleted.append(key)
        if key == "isaac_warm_job":
            deadline = time.monotonic() + min(args.warm_timeout_seconds, 600.0)
            while time.monotonic() < deadline:
                pods = core.list_namespaced_pod(
                    args.namespace,
                    label_selector=f"app={args.isaac_warm_job}",
                    _request_timeout=30,
                )
                if not list(getattr(pods, "items", None) or []):
                    break
                time.sleep(args.poll_interval_seconds)
            else:
                raise RuntimePrerequisiteError(
                    "Isaac cache warm pods did not terminate"
                )
    return _receipt(args, "cleanup", deleted, absent, None)


def _receipt(
    args: argparse.Namespace,
    action: str,
    changed: list[str],
    retained: list[str],
    job: Any | None,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "passed",
        "action": action,
        "run_id": args.run_id,
        "namespace": args.namespace,
        "identities": resource_identities(args.run_id),
        "changed": changed,
        "retained_or_absent": retained,
        "isaac_image": args.isaac_image,
        "warm_job_uid": getattr(getattr(job, "metadata", None), "uid", None),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the clean-start runtime prerequisite command parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("apply", "cleanup"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--isaac-image", required=True)
    for name in resource_identities("placeholder"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    parser.add_argument("--storage-class", default="csi-mounted-fs-path-sc")
    parser.add_argument("--storage-access-mode", default="ReadWriteOnce")
    parser.add_argument("--isaac-cache-size", default="40Gi")
    parser.add_argument("--policy-cache-size", default="100Gi")
    parser.add_argument("--warm-timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=5.0)
    parser.add_argument("--output-uri", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Apply or remove one exact workflow-owned prerequisite set."""

    args = build_parser().parse_args(argv)
    manifests = build_manifests(args)
    result = (
        _apply(args, manifests) if args.action == "apply" else _delete(args, manifests)
    )
    with tempfile.TemporaryDirectory(prefix="npa-sim2real-runtime-") as directory:
        write_json(args.output_uri, result, directory=Path(directory))
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
