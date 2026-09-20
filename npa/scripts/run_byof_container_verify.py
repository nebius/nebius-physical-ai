#!/usr/bin/env python3
"""Submit BYOF container-verify SkyPilot workloads (CPU smoke for /opt/byof clone)."""

import argparse
import base64
import binascii
from dataclasses import dataclass, replace
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import Any, Callable
from urllib.parse import urlparse

import yaml

from npa.deploy.images import (
    libero_image_manifest,
    libero_publication_lineage_values,
    validate_libero_customer_runtime_authorization,
    validate_libero_authenticated_caller_assertion,
    validate_libero_output_storage_authorization,
    validate_libero_qualified_image_manifest,
)
from npa.execution_preflight import (
    SKYPILOT_ENGINE_SERVICE_ACCOUNT,
    libero_executable_profile_bytes,
    is_libero_official_image_reference,
    skypilot_task_documents,
    verify_solution_payload_service_accounts,
)
from npa.workflows.byof.live import resolve_byof_profile_path
from npa.clients.project_credentials import (
    s3_client_for_project,
    storage_env_for_project,
)
from npa.orchestration.skypilot import (
    SkyPilotSubmitError,
    submit_workflow,
    workflow_status,
)
from npa.orchestration.skypilot._bin import (
    SkyPilotConfigError,
    SkyPilotNotInstalledError,
    SkyPilotVersionError,
    resolve_isolated_config_dir,
    resolve_sky_bin,
)
from npa.orchestration.skypilot.cleanup import (
    CleanupResult,
    cluster_name_patterns_for_run,
    sky_environment,
)
from npa.orchestration.skypilot.signal_teardown import (
    SignalTeardown,
    install_teardown_signal_handlers,
    restore_signal_handlers,
)
from npa.orchestration.skypilot.workflow_state import cancel_workflow_job

DEFAULT_YAML = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "npa"
    / "workflows"
    / "byof"
    / "profiles"
    / "byof-container-smoke-rtxpro.yaml"
)
DEFAULT_IMAGE_PULL_SECRETS = ("agent-sa",)
LIBERO_SOLUTION_NAME = "libero"
LIBERO_PAYLOAD_SERVICE_ACCOUNT = "npa-byof-libero-payload"
LIBERO_PAYLOAD_ROLE = "npa-byof-libero-pod-reader"
LIBERO_PAYLOAD_ROLE_BINDING = "npa-byof-libero-payload-pod-reader"
LIBERO_PAYLOAD_UNBOUND_RESOURCE_NAME = "npa-libero-unbound-pod"
LIBERO_PAYLOAD_RELEASE_ANNOTATION = "npa.nebius.com/libero-release"
LIBERO_CONTROLLER_ROLE = f"{SKYPILOT_ENGINE_SERVICE_ACCOUNT}-role"
LIBERO_CONTROLLER_ROLE_BINDING = f"{SKYPILOT_ENGINE_SERVICE_ACCOUNT}-role-binding"
LIBERO_STORAGE_VERIFICATION_CONFIGMAP = "npa-byof-libero-storage-verification"
LIBERO_CONTROLLER_RULES = [
    {
        "apiGroups": [""],
        "resources": ["pods"],
        "verbs": ["create", "delete", "get", "list", "patch", "watch"],
    },
    {
        "apiGroups": [""],
        "resources": ["pods/exec"],
        "verbs": ["create", "get"],
    },
    {"apiGroups": [""], "resources": ["pods/log"], "verbs": ["get"]},
    {
        "apiGroups": [""],
        "resources": ["services"],
        "verbs": ["create", "delete", "get", "list", "patch", "watch"],
    },
    {
        "apiGroups": [""],
        "resources": ["configmaps", "secrets"],
        "verbs": ["create", "delete", "get"],
    },
]
LIBERO_PROFILE_FILENAME = "byof-solution-smoke-libero-b200-gpu.yaml"
LIBERO_PROFILE_TASK_NAME = "byof-solution-smoke-libero-b200-gpu"
LIBERO_RUNTIME_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "docker"
    / "workbench"
    / "libero"
    / "runtime-manifest.json"
)
#: Credentials every BYOF resource profile needs, because each one uploads its summary
#: and artifacts to S3. Forwarded as SkyPilot task secrets (never written into the
#: rendered YAML). Without this a run provisions, pulls the image, executes the profile
#: and then dies at the upload with
#: ``botocore.exceptions.NoCredentialsError: Unable to locate credentials``.
#: Operator-held runtime state that a vendor gate reads inside the pod: vendor
#: terms acceptances and the operator's own gated-repository token. These are
#: things a person holds or did, not workflow configuration, so they travel
#: through SkyPilot's redacted secret channel and never appear in a rendered
#: YAML. Unset names are dropped, so a run that holds nothing forwards nothing
#: and the container's own gate refuses.
#:
#: Keyed by solution, because these are per-vendor answers and a single shared
#: tuple quietly widens every other image's environment: a variable added for
#: one solution is forwarded into every other BYOF run whenever it happens to be
#: set in the operator's shell. A solution that is not listed forwards none.
OPERATOR_RUNTIME_ENVS_BY_SOLUTION: dict[str, tuple[str, ...]] = {
    "openpi": ("NPA_OPENPI_ACCEPT_GEMMA_TERMS",),
    "ltx2.5": (
        "NPA_LTX_ACCEPT_NVIDIA_RUNTIME_TERMS",
        # The gated-repository entitlement, which the container requires for the
        # LTX source as well as the weights.
        "HF_TOKEN",
    ),
    "libero": (
        "NPA_LIBERO_CUSTOMER_AUTHORIZATION_B64",
        "NPA_LIBERO_CUSTOMER_AUTHORIZATION_SHA256",
        "NPA_LIBERO_CUSTOMER_IDENTITY_SHA256",
        "NPA_LIBERO_AUTHENTICATED_CALLER_B64",
        "NPA_LIBERO_AUTHENTICATED_CALLER_SHA256",
        "NPA_LIBERO_OUTPUT_STORAGE_AUTHORIZATION_B64",
    ),
}
DEFAULT_SECRET_ENVS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
)
LIBERO_PUBLIC_GROUP_NON_RESOURCE_URLS = frozenset(
    {
        "/api",
        "/api/*",
        "/apis",
        "/apis/*",
        "/healthz",
        "/livez",
        "/openapi",
        "/openapi/*",
        "/readyz",
        "/version",
        "/version/",
    }
)


def resolve_secret_envs(
    explicit: list[str] | None, *, solution_name: str = ""
) -> list[str]:
    """Return the secret env names to forward to SkyPilot.

    An explicit ``--secret-env`` list replaces the default storage names. The
    operator-runtime gates *this solution* reads are appended in either case, so
    acceptance cannot fall back to rendered YAML — and a solution never receives
    another vendor's answers. Names with no value are dropped, since SkyPilot
    rejects a secret it cannot resolve.
    """

    names = list(explicit if explicit is not None else DEFAULT_SECRET_ENVS)
    # Customer authorization is runtime state, not workflow configuration. Always
    # carry an explicitly set gate through SkyPilot's redacted secret channel,
    # even when a caller supplies an otherwise explicit secret allowlist.
    names.extend(OPERATOR_RUNTIME_ENVS_BY_SOLUTION.get(solution_name.strip(), ()))
    return [name for name in dict.fromkeys(names) if os.environ.get(name)]


def _normalize_s3_bucket(value: str) -> str:
    """Return a bare bucket name from ``bucket`` or ``s3://bucket[/prefix]``."""

    text = (value or "").strip()
    if not text:
        return ""
    if text.startswith("s3://"):
        remainder = text[len("s3://") :]
        return remainder.split("/", 1)[0].strip()
    return text.split("/", 1)[0].strip()


def _normalize_output_root(value: str, *, default_prefix: str = "byof") -> str:
    """Normalize output roots that may already include ``s3://`` or a path prefix."""

    text = (value or "").strip()
    if not text:
        bucket = _normalize_s3_bucket(os.environ.get("NPA_S3_BUCKET", ""))
        if not bucket:
            bucket = "your-bucket-name"
        return f"s3://{bucket}/{default_prefix}"
    if text.startswith("s3://"):
        # Collapse accidental ``s3://s3://bucket/...`` forms.
        while text.startswith("s3://s3://"):
            text = "s3://" + text[len("s3://s3://") :]
        return text.rstrip("/")
    bucket = _normalize_s3_bucket(text)
    remainder = text.split("/", 1)[1].strip("/") if "/" in text else default_prefix
    return f"s3://{bucket}/{remainder or default_prefix}"


DEFAULT_BUCKET = (
    _normalize_s3_bucket(os.environ.get("NPA_S3_BUCKET", "")) or "your-bucket-name"
)
DEFAULT_OUTPUT_ROOT = _normalize_output_root(
    os.environ.get("NPA_BYOF_OUTPUT_ROOT", ""), default_prefix="byof"
)
TERMINAL_STATUSES = {
    "ABSENT",
    "SUCCEEDED",
    "CANCELLED",
    "FAILED",
    "FAILED_SETUP",
    "FAILED_PRECHECKS",
    "FAILED_NO_RESOURCE",
    "FAILED_CONTROLLER",
}
VERIFIED_DRAIN_STATUSES = TERMINAL_STATUSES - {"FAILED_CONTROLLER"}


def _is_libero_invocation(
    args: argparse.Namespace, documents: list[dict[str, Any]]
) -> bool:
    if args.solution_name.strip().lower() == LIBERO_SOLUTION_NAME:
        return True
    if Path(args.yaml_path).name == LIBERO_PROFILE_FILENAME:
        return True
    for document in skypilot_task_documents(documents):
        resources = document.get("resources") or {}
        envs = document.get("envs") or {}
        if not isinstance(resources, dict) or not isinstance(envs, dict):
            continue
        if any(
            is_libero_official_image_reference(image)
            for image in (resources.get("image_id"), envs.get("BYOF_IMAGE"))
        ):
            return True
    return False


def _uses_libero_payload_service_account(documents: list[dict[str, Any]]) -> bool:
    """Detect the reserved identity without treating its name as authorization."""

    for document in documents:
        config = document.get("config")
        if not isinstance(config, dict):
            continue
        kubernetes = config.get("kubernetes")
        if not isinstance(kubernetes, dict):
            continue
        pod_config = kubernetes.get("pod_config")
        if not isinstance(pod_config, dict):
            continue
        pod_spec = pod_config.get("spec")
        if (
            isinstance(pod_spec, dict)
            and pod_spec.get("serviceAccountName") == LIBERO_PAYLOAD_SERVICE_ACCOUNT
        ):
            return True
    return False


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _libero_allowed_node(global_config: dict[str, Any]) -> str:
    kubernetes = global_config.get("kubernetes") or {}
    if not isinstance(kubernetes, dict) or not set(kubernetes) <= {
        "allowed_nodes",
        "allowed_contexts",
        "pod_config",
    }:
        raise ValueError("LIBERO Kubernetes configuration is not closed")
    allowed = kubernetes.get("allowed_nodes") if isinstance(kubernetes, dict) else None
    if not isinstance(allowed, dict) or set(allowed) != {"names"}:
        raise ValueError(
            "LIBERO requires owner-supplied kubernetes.allowed_nodes.names"
        )
    names = allowed.get("names")
    if (
        not isinstance(names, list)
        or len(names) != 1
        or not isinstance(names[0], str)
        or not names[0].strip()
        or names[0] != names[0].strip()
    ):
        raise ValueError(
            "LIBERO requires exactly one exact non-empty allowed node name"
        )
    return names[0]


def _kubectl_json(
    arguments: list[str], *, purpose: str, kubeconfig: Path
) -> dict[str, Any]:
    result = subprocess.run(
        ["kubectl", "--kubeconfig", str(kubeconfig), *arguments, "-o", "json"],
        env=sky_environment(None),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Kubernetes {purpose} observation failed")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Kubernetes {purpose} returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Kubernetes {purpose} did not return an object")
    return payload


def _mode_private_regular_file(value: str, *, label: str) -> Path:
    if not value:
        raise ValueError(f"LIBERO requires an owner-supplied {label}")
    path = Path(value).expanduser()
    try:
        metadata = path.lstat()
    except OSError:
        metadata = None
    if (
        metadata is None
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_mode & 0o077
    ):
        raise ValueError(f"LIBERO {label} must be an owner-private regular file")
    return path.resolve()


def _stable_owner_private_bytes(
    path: Path, *, label: str, limit: int = 8 << 20
) -> bytes:
    """Snapshot one owner-private file through a stable no-follow descriptor."""

    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as exc:
        raise ValueError(f"LIBERO {label} is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        payload = os.pread(descriptor, limit + 1, 0)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.getuid()
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) & 0o077
        or len(payload) > limit
        or before.st_size != len(payload)
        or identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise ValueError(f"LIBERO {label} is not a stable owner-private file")
    return payload


def _immutable_exact_run_copy(source: Path, target: Path, *, label: str) -> Path:
    """Create one no-overwrite mode-0400 copy used for all later operations."""

    payload = _stable_owner_private_bytes(source, label=label)
    descriptor = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o400,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if _stable_owner_private_bytes(target, label=f"immutable {label}") != payload:
        raise ValueError(f"LIBERO immutable {label} readback differs")
    return target


def _stop_sky_api(
    *,
    sky_bin: str,
    isolated_config_dir: Path | None,
    config_path: Path | str | None,
) -> None:
    """Stop the selected Sky API or fail without claiming successful cleanup."""

    environment = sky_environment(isolated_config_dir)
    if config_path:
        environment["SKYPILOT_GLOBAL_CONFIG"] = str(config_path)
    try:
        result = subprocess.run(
            [sky_bin, "api", "stop"],
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001 - shutdown ambiguity is fatal
        raise SkyPilotConfigError(
            "SkyPilot API shutdown raised; isolated API absence is unverified"
        ) from exc
    if result.returncode != 0:
        raise SkyPilotConfigError(
            "SkyPilot API shutdown failed; isolated API absence is unverified"
        )


def _libero_payload_kubeconfig() -> Path:
    return _mode_private_regular_file(
        os.environ.get("NPA_LIBERO_PAYLOAD_KUBECONFIG", "").strip(),
        label="payload kubeconfig",
    )


def _libero_isolated_state_root(path: Path | None, run_id: str) -> Path:
    if path is None:
        raise ValueError("LIBERO requires an isolated SkyPilot state root")
    if path.is_symlink() or not path.is_dir():
        raise ValueError("LIBERO isolated SkyPilot state must be a directory")
    metadata = path.stat()
    if metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
        raise ValueError("LIBERO isolated SkyPilot state must be owner-private")
    resolved = path.resolve()
    if resolved.name != run_id or resolved.parent == resolved:
        raise ValueError("LIBERO isolated SkyPilot state must be exact-run scoped")
    return resolved


def _libero_global_config_path(args: argparse.Namespace) -> str:
    configured = (
        args.config_path or os.environ.get("NPA_LIBERO_SKYPILOT_CONFIG", "").strip()
    )
    return str(_mode_private_regular_file(configured, label="SkyPilot global config"))


def _libero_context_contract(
    kubeconfig: Path, *, expected_context: str = "", require_namespace: bool
) -> tuple[str, str, str]:
    config = _kubectl_json(
        ["config", "view", "--minify", "--flatten", "--raw"],
        purpose="selected-context",
        kubeconfig=kubeconfig,
    )
    context = str(config.get("current-context") or "").strip()
    if not context:
        raise RuntimeError("LIBERO kubeconfig has no current context")
    if expected_context and context != expected_context:
        raise RuntimeError("LIBERO kubeconfig differs from its expected context")
    contexts = config.get("contexts") or []
    matches = [item for item in contexts if item.get("name") == context]
    if len(matches) != 1:
        raise RuntimeError("LIBERO selected context is absent or ambiguous")
    context_record = matches[0].get("context") or {}
    namespace = str(context_record.get("namespace") or "").strip()
    if require_namespace and not namespace:
        raise RuntimeError("LIBERO selected context has no namespace")
    selected_cluster = str(context_record.get("cluster") or "").strip()
    clusters = [
        item
        for item in config.get("clusters") or []
        if item.get("name") == selected_cluster
    ]
    if len(clusters) != 1:
        raise RuntimeError("LIBERO selected cluster is absent or ambiguous")
    cluster = clusters[0].get("cluster") or {}
    server = str(cluster.get("server") or "").strip()
    certificate_data = str(cluster.get("certificate-authority-data") or "").strip()
    if not server or not certificate_data or cluster.get("insecure-skip-tls-verify"):
        raise RuntimeError("LIBERO selected cluster lacks strict TLS identity")
    cluster_identity_sha256 = _sha256_json(
        {"server": server, "certificate_authority_data": certificate_data}
    )
    return context, namespace, cluster_identity_sha256


def _libero_resource(
    kubeconfig: Path, context: str, namespace: str, kind: str, name: str
) -> dict[str, Any]:
    payload = _kubectl_json(
        ["--context", context, "--namespace", namespace, "get", kind, name],
        purpose=f"LIBERO {kind}",
        kubeconfig=kubeconfig,
    )
    metadata = payload.get("metadata") or {}
    if metadata.get("name") != name or metadata.get("namespace") != namespace:
        raise RuntimeError(f"LIBERO {kind} identity differs from the expected object")
    if not metadata.get("uid") or not metadata.get("creationTimestamp"):
        raise RuntimeError(f"LIBERO {kind} has incomplete ownership metadata")
    return payload


@dataclass(frozen=True)
class LiberoAccessState:
    """Raw run-owned access identities retained only for guarded cleanup."""

    kubeconfig: Path
    context: str
    namespace: str
    namespace_uid: str
    service_account_uid: str
    role_uid: str
    role_binding_uid: str
    controller_service_account_uid: str = ""
    controller_role_uid: str = ""
    controller_role_binding_uid: str = ""
    execution_kubeconfig: Path | None = None
    execution_context: str = ""
    run_id: str = ""
    payload_pod_name: str = ""
    payload_pod_uid: str = ""
    skypilot_cluster_name: str = ""


@dataclass(frozen=True)
class LiberoRuntimeBinding:
    evidence: dict[str, str]
    access_state: LiberoAccessState
    customer_authorization_b64: str = ""
    candidate_image: str = ""
    task_name: str = ""


def _libero_payload_rules(resource_name: str) -> list[dict[str, Any]]:
    """Return the one-object payload grant, including its no-match initial state."""

    if re.fullmatch(r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?", resource_name) is None:
        raise RuntimeError("LIBERO payload Pod resourceName is invalid")
    return [
        {
            "apiGroups": [""],
            "resources": ["pods"],
            "resourceNames": [resource_name],
            "verbs": ["get"],
        }
    ]


def _libero_managed_job_display_name(task_name: str, scheduler_job_id: str) -> str:
    """Mirror the pinned SkyPilot managed-job name before its per-user suffix."""

    if re.fullmatch(r"[1-9][0-9]*", scheduler_job_id) is None:
        raise RuntimeError("LIBERO scheduler job ID is invalid")
    normalized = re.sub(r"[._]", "-", task_name).lower()
    if re.fullmatch(r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?", normalized) is None:
        raise RuntimeError("LIBERO SkyPilot task name is invalid")
    if len(normalized) > 25:
        alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
        number = int(
            hashlib.md5(task_name.encode(), usedforsecurity=False).hexdigest(), 16
        )
        encoded = ""
        while number:
            number, remainder = divmod(number, 36)
            encoded = alphabet[remainder] + encoded
        normalized = f"{normalized[:22].rstrip('-')}-{encoded[:2]}"
    return f"{normalized}-{scheduler_job_id}"


def _reject_libero_cluster_role_bindings(
    kubeconfig: Path, context: str, namespace: str
) -> list[dict[str, Any]]:
    cluster_bindings = _kubectl_json(
        ["--context", context, "get", "clusterrolebindings"],
        purpose="LIBERO cluster RoleBinding inventory",
        kubeconfig=kubeconfig,
    )
    items = cluster_bindings.get("items")
    if not isinstance(items, list):
        raise RuntimeError("LIBERO cluster RoleBinding inventory is invalid")
    namespace_service_account_user = f"system:serviceaccount:{namespace}:"
    namespace_service_account_group = f"system:serviceaccounts:{namespace}"
    broad_authenticated_groups = {
        "system:authenticated",
        "system:unauthenticated",
    }
    safe_bindings: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            raise RuntimeError("LIBERO cluster RoleBinding inventory is invalid")
        subjects = item.get("subjects") or []
        if not isinstance(subjects, list):
            raise RuntimeError("LIBERO cluster RoleBinding subjects are invalid")
        for subject in subjects:
            if not isinstance(subject, dict):
                raise RuntimeError("LIBERO cluster RoleBinding subject is invalid")
            kind = subject.get("kind")
            name = subject.get("name")
            direct_namespace_grant = (
                kind == "ServiceAccount" and subject.get("namespace") == namespace
            )
            service_account_user_grant = (
                kind == "User"
                and isinstance(name, str)
                and name.startswith(namespace_service_account_user)
            )
            service_account_group_grant = kind == "Group" and name in {
                "system:serviceaccounts",
                namespace_service_account_group,
            }
            if (
                direct_namespace_grant
                or service_account_user_grant
                or service_account_group_grant
            ):
                raise RuntimeError(
                    "LIBERO namespace service accounts may not receive "
                    "ClusterRoleBindings"
                )
            if kind == "Group" and name in broad_authenticated_groups:
                role_ref = item.get("roleRef")
                if (
                    not isinstance(role_ref, dict)
                    or role_ref.get("apiGroup") != "rbac.authorization.k8s.io"
                    or role_ref.get("kind") != "ClusterRole"
                    or not isinstance(role_ref.get("name"), str)
                    or not role_ref["name"]
                ):
                    raise RuntimeError(
                        "LIBERO broad-group ClusterRoleBinding is invalid"
                    )
                cluster_role = _kubectl_json(
                    [
                        "--context",
                        context,
                        "get",
                        "clusterrole",
                        role_ref["name"],
                    ],
                    purpose="LIBERO broad-group ClusterRole",
                    kubeconfig=kubeconfig,
                )
                rules = cluster_role.get("rules")
                metadata = item.get("metadata") or {}
                cluster_role_metadata = cluster_role.get("metadata") or {}
                if (
                    not isinstance(metadata.get("name"), str)
                    or not metadata["name"]
                    or not isinstance(metadata.get("uid"), str)
                    or not metadata["uid"]
                    or not isinstance(cluster_role_metadata.get("uid"), str)
                    or not cluster_role_metadata["uid"]
                    or not isinstance(rules, list)
                    or not _libero_safe_public_group_rules(rules)
                ):
                    raise RuntimeError(
                        "LIBERO broad authenticated groups may not receive "
                        "workload resource access"
                    )
                safe_bindings.append(
                    {
                        "kind": "ClusterRoleBinding",
                        "name": metadata.get("name"),
                        "uid": metadata.get("uid"),
                        "role_ref": role_ref,
                        "role_uid": cluster_role_metadata["uid"],
                        "rules": rules,
                        "subject": subject,
                    }
                )
    return safe_bindings


def _libero_safe_public_group_rules(rules: list[Any]) -> bool:
    """Allow only discovery and self-access review for Kubernetes public groups."""

    self_review_resources = {
        "selfsubjectaccessreviews",
        "selfsubjectrulesreviews",
        "selfsubjectreviews",
    }
    for rule in rules:
        if not isinstance(rule, dict):
            return False
        resources = rule.get("resources") or []
        non_resource_urls = rule.get("nonResourceURLs") or []
        if resources:
            if (
                not isinstance(resources, list)
                or not set(resources) <= self_review_resources
                or set(rule.get("apiGroups") or []) != {"authorization.k8s.io"}
                or set(rule.get("verbs") or []) != {"create"}
                or non_resource_urls
                or rule.get("resourceNames")
            ):
                return False
        elif (
            not isinstance(non_resource_urls, list)
            or not non_resource_urls
            or any(
                not isinstance(url, str) or not url.startswith("/")
                for url in non_resource_urls
            )
            or not set(non_resource_urls) <= LIBERO_PUBLIC_GROUP_NON_RESOURCE_URLS
            or set(rule.get("verbs") or []) != {"get"}
        ):
            return False
    return True


def _libero_role_binding_rules(
    item: dict[str, Any],
    *,
    kubeconfig: Path,
    context: str,
    binding_namespace: str,
) -> tuple[list[Any], str]:
    role_ref = item.get("roleRef")
    if (
        not isinstance(role_ref, dict)
        or role_ref.get("apiGroup") != "rbac.authorization.k8s.io"
        or role_ref.get("kind") not in {"Role", "ClusterRole"}
        or not isinstance(role_ref.get("name"), str)
        or not role_ref["name"]
    ):
        raise RuntimeError("LIBERO broad-group RoleBinding is invalid")
    arguments = ["--context", context]
    if role_ref["kind"] == "Role":
        arguments.extend(["--namespace", binding_namespace, "get", "role"])
    else:
        arguments.extend(["get", "clusterrole"])
    arguments.append(role_ref["name"])
    role = _kubectl_json(
        arguments,
        purpose="LIBERO broad-group RoleBinding role",
        kubeconfig=kubeconfig,
    )
    rules = role.get("rules")
    role_uid = str((role.get("metadata") or {}).get("uid") or "")
    if not isinstance(rules, list) or not role_uid:
        raise RuntimeError("LIBERO broad-group RoleBinding rules are invalid")
    return rules, role_uid


def _reject_libero_cross_namespace_role_bindings(
    kubeconfig: Path, context: str, namespace: str
) -> list[dict[str, Any]]:
    bindings = _kubectl_json(
        ["--context", context, "--all-namespaces", "get", "rolebindings"],
        purpose="LIBERO all-namespace RoleBinding inventory",
        kubeconfig=kubeconfig,
    )
    items = bindings.get("items")
    if not isinstance(items, list):
        raise RuntimeError("LIBERO all-namespace RoleBinding inventory is invalid")
    namespace_user = f"system:serviceaccount:{namespace}:"
    namespace_group = f"system:serviceaccounts:{namespace}"
    safe_bindings: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            raise RuntimeError("LIBERO all-namespace RoleBinding is invalid")
        metadata = item.get("metadata") or {}
        binding_namespace = metadata.get("namespace")
        binding_name = metadata.get("name")
        subjects = item.get("subjects") or []
        if (
            not isinstance(binding_namespace, str)
            or not binding_namespace
            or not isinstance(binding_name, str)
            or not binding_name
            or not isinstance(subjects, list)
        ):
            raise RuntimeError("LIBERO all-namespace RoleBinding is invalid")
        for subject in subjects:
            if not isinstance(subject, dict):
                raise RuntimeError(
                    "LIBERO all-namespace RoleBinding subject is invalid"
                )
            kind = subject.get("kind")
            name = subject.get("name")
            reaches_run_account = (
                kind == "ServiceAccount" and subject.get("namespace") == namespace
            ) or (
                kind == "User"
                and isinstance(name, str)
                and name.startswith(namespace_user)
            )
            reaches_run_group = kind == "Group" and name in {
                "system:serviceaccounts",
                namespace_group,
            }
            intended_local_binding = (
                binding_namespace == namespace
                and binding_name
                in {LIBERO_PAYLOAD_ROLE_BINDING, LIBERO_CONTROLLER_ROLE_BINDING}
            )
            if (
                reaches_run_account or reaches_run_group
            ) and not intended_local_binding:
                raise RuntimeError(
                    "LIBERO service accounts may not receive cross-namespace "
                    "RoleBindings"
                )
            if kind == "Group" and name in {
                "system:authenticated",
                "system:unauthenticated",
            }:
                rules, role_uid = _libero_role_binding_rules(
                    item,
                    kubeconfig=kubeconfig,
                    context=context,
                    binding_namespace=binding_namespace,
                )
                if not _libero_safe_public_group_rules(rules):
                    raise RuntimeError(
                        "LIBERO broad authenticated groups may not receive "
                        "namespaced workload resource access"
                    )
                safe_bindings.append(
                    {
                        "kind": "RoleBinding",
                        "name": binding_name,
                        "namespace": binding_namespace,
                        "uid": metadata.get("uid"),
                        "role_ref": item.get("roleRef"),
                        "role_uid": role_uid,
                        "rules": rules,
                        "subject": subject,
                    }
                )
    return safe_bindings


def _libero_external_rbac_inventory_sha256(
    kubeconfig: Path, context: str, namespace: str
) -> str:
    records = [
        *_reject_libero_cluster_role_bindings(kubeconfig, context, namespace),
        *_reject_libero_cross_namespace_role_bindings(kubeconfig, context, namespace),
    ]
    records.sort(
        key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"))
    )
    return _sha256_json(records)


def _verify_libero_storage_configmap(item: dict[str, Any]) -> None:
    """Require the immutable mounted key to match the qualified control plane."""
    data = item.get("data")
    key_name = "output-storage-authorization-public-key.b64"
    if (
        item.get("immutable") is not True
        or not isinstance(data, dict)
        or set(data) != {key_name}
        or item.get("binaryData")
    ):
        raise RuntimeError("LIBERO storage verification ConfigMap is not immutable or closed")
    try:
        key = base64.b64decode(data[key_name], validate=True)
    except (ValueError, TypeError) as exc:
        raise RuntimeError("LIBERO storage verification ConfigMap key is invalid") from exc
    qualification = validate_libero_qualified_image_manifest(libero_image_manifest())
    if len(key) != 32 or hashlib.sha256(key).hexdigest() != qualification[
        "output_storage_authorization_public_key_sha256"
    ]:
        raise RuntimeError("LIBERO storage verification ConfigMap key differs from qualification")


def _libero_namespaced_inventory(
    kubeconfig: Path, context: str, namespace: str
) -> dict[str, list[str]]:
    inventory: dict[str, list[str]] = {}
    expected = {
        "pods": set(),
        "serviceaccounts": {
            "default",
            LIBERO_PAYLOAD_SERVICE_ACCOUNT,
            SKYPILOT_ENGINE_SERVICE_ACCOUNT,
        },
        "roles": {LIBERO_PAYLOAD_ROLE, LIBERO_CONTROLLER_ROLE},
        "rolebindings": {
            LIBERO_PAYLOAD_ROLE_BINDING,
            LIBERO_CONTROLLER_ROLE_BINDING,
        },
        "secrets": set(),
        "configmaps": {"kube-root-ca.crt", LIBERO_STORAGE_VERIFICATION_CONFIGMAP},
        "services": set(),
    }
    for kind, expected_names in expected.items():
        payload = _kubectl_json(
            ["--context", context, "--namespace", namespace, "get", kind],
            purpose=f"LIBERO namespace {kind} inventory",
            kubeconfig=kubeconfig,
        )
        items = payload.get("items")
        if not isinstance(items, list):
            raise RuntimeError(f"LIBERO namespace {kind} inventory is invalid")
        names = sorted(
            str((item.get("metadata") or {}).get("name") or "")
            for item in items
            if isinstance(item, dict)
        )
        if (
            any(not name for name in names)
            or set(names) != expected_names
            or len(names) != len(expected_names)
        ):
            raise RuntimeError(
                f"LIBERO namespace is not isolated to its reviewed {kind} inventory"
            )
        inventory[kind] = names
        if kind == "configmaps":
            records = []
            for item in items:
                metadata = item.get("metadata") or {}
                if metadata["name"] == LIBERO_STORAGE_VERIFICATION_CONFIGMAP:
                    _verify_libero_storage_configmap(item)
                else:
                    data = item.get("data")
                    if (
                        metadata.get("ownerReferences")
                        or not isinstance(data, dict)
                        or set(data) != {"ca.crt"}
                        or not isinstance(data["ca.crt"], str)
                        or not data["ca.crt"].strip()
                        or item.get("binaryData")
                    ):
                        raise RuntimeError("LIBERO root-CA ConfigMap has an unexpected shape")
                if not metadata.get("uid") or metadata.get("namespace") != namespace:
                    raise RuntimeError("LIBERO ConfigMap has no namespace-bound UID")
                records.append(_sha256_json({"metadata": metadata, "data": item.get("data"), "immutable": item.get("immutable")}))
            inventory["configmap_records"] = sorted(records)
    return inventory


def _libero_controller_rbac_evidence(
    kubeconfig: Path, context: str, namespace: str
) -> tuple[dict[str, str], dict[str, str]]:
    """Verify the manager-precreated engine identity before managed launch.

    The role enumerates only the namespaced primitives SkyPilot uses for this
    container job. The fetched payload never receives this account: it keeps
    the independent exact-resourceName pods/get identity verified above.
    """

    account = _libero_resource(
        kubeconfig,
        context,
        namespace,
        "serviceaccount",
        SKYPILOT_ENGINE_SERVICE_ACCOUNT,
    )
    role = _libero_resource(
        kubeconfig, context, namespace, "role", LIBERO_CONTROLLER_ROLE
    )
    binding = _libero_resource(
        kubeconfig,
        context,
        namespace,
        "rolebinding",
        LIBERO_CONTROLLER_ROLE_BINDING,
    )
    rules = LIBERO_CONTROLLER_RULES
    subjects = [
        {
            "kind": "ServiceAccount",
            "name": SKYPILOT_ENGINE_SERVICE_ACCOUNT,
            "namespace": namespace,
        }
    ]
    role_ref = {
        "apiGroup": "rbac.authorization.k8s.io",
        "kind": "Role",
        "name": LIBERO_CONTROLLER_ROLE,
    }
    if role.get("rules") != rules:
        raise RuntimeError(
            "LIBERO SkyPilot controller Role differs from the reviewed namespace-only contract"
        )
    if binding.get("subjects") != subjects or binding.get("roleRef") != role_ref:
        raise RuntimeError(
            "LIBERO SkyPilot controller RoleBinding differs from the reviewed contract"
        )
    external_rbac_inventory_sha256 = _libero_external_rbac_inventory_sha256(
        kubeconfig, context, namespace
    )
    evidence = {
        "controller_service_account_uid_sha256": hashlib.sha256(
            account["metadata"]["uid"].encode()
        ).hexdigest(),
        "controller_role_uid_sha256": hashlib.sha256(
            role["metadata"]["uid"].encode()
        ).hexdigest(),
        "controller_role_binding_uid_sha256": hashlib.sha256(
            binding["metadata"]["uid"].encode()
        ).hexdigest(),
        "controller_rbac_spec_sha256": _sha256_json(
            {"rules": rules, "subjects": subjects, "roleRef": role_ref}
        ),
        "external_rbac_inventory_sha256": external_rbac_inventory_sha256,
    }
    identities = {
        "controller_service_account_uid": account["metadata"]["uid"],
        "controller_role_uid": role["metadata"]["uid"],
        "controller_role_binding_uid": binding["metadata"]["uid"],
    }
    return evidence, identities


def _libero_rbac_evidence(
    kubeconfig: Path,
    context: str,
    namespace: str,
    run_id: str,
    *,
    require_empty_inventory: bool = True,
    expected_pod_name: str = LIBERO_PAYLOAD_UNBOUND_RESOURCE_NAME,
) -> tuple[dict[str, str], LiberoAccessState]:
    namespace_record = _kubectl_json(
        ["--context", context, "get", "namespace", namespace],
        purpose="LIBERO namespace",
        kubeconfig=kubeconfig,
    )
    namespace_metadata = namespace_record.get("metadata") or {}
    run_id_sha256 = hashlib.sha256(run_id.encode()).hexdigest()
    labels = namespace_metadata.get("labels") or {}
    if (
        namespace_metadata.get("name") != namespace
        or not namespace_metadata.get("uid")
        or labels.get("npa.nebius.ai/solution") != "libero"
        or labels.get("npa.nebius.ai/run-id-sha256") != run_id_sha256
    ):
        raise RuntimeError("LIBERO namespace is not bound to this exact run")
    account = _libero_resource(
        kubeconfig,
        context,
        namespace,
        "serviceaccount",
        LIBERO_PAYLOAD_SERVICE_ACCOUNT,
    )
    role = _libero_resource(kubeconfig, context, namespace, "role", LIBERO_PAYLOAD_ROLE)
    binding = _libero_resource(
        kubeconfig,
        context,
        namespace,
        "rolebinding",
        LIBERO_PAYLOAD_ROLE_BINDING,
    )
    rules = _libero_payload_rules(expected_pod_name)
    subjects = [
        {
            "kind": "ServiceAccount",
            "name": LIBERO_PAYLOAD_SERVICE_ACCOUNT,
            "namespace": namespace,
        }
    ]
    role_ref = {
        "apiGroup": "rbac.authorization.k8s.io",
        "kind": "Role",
        "name": LIBERO_PAYLOAD_ROLE,
    }
    if role.get("rules") != rules:
        raise RuntimeError(
            "LIBERO payload Role is not scoped to the exact Pod resourceName"
        )
    if binding.get("subjects") != subjects or binding.get("roleRef") != role_ref:
        raise RuntimeError(
            "LIBERO payload RoleBinding differs from the reviewed contract"
        )
    external_rbac_inventory_sha256 = _libero_external_rbac_inventory_sha256(
        kubeconfig, context, namespace
    )
    inventory = None
    if require_empty_inventory:
        inventory = _libero_namespaced_inventory(kubeconfig, context, namespace)
    evidence = {
        "service_account_uid_sha256": hashlib.sha256(
            account["metadata"]["uid"].encode()
        ).hexdigest(),
        "role_uid_sha256": hashlib.sha256(role["metadata"]["uid"].encode()).hexdigest(),
        "role_binding_uid_sha256": hashlib.sha256(
            binding["metadata"]["uid"].encode()
        ).hexdigest(),
        "rbac_spec_sha256": _sha256_json(
            {
                "rules": rules,
                "subjects": subjects,
                "roleRef": role_ref,
                "namespace_run_id_sha256": run_id_sha256,
            }
        ),
        "namespace_sha256": hashlib.sha256(namespace.encode()).hexdigest(),
        "namespace_uid_sha256": hashlib.sha256(
            namespace_metadata["uid"].encode()
        ).hexdigest(),
        "external_rbac_inventory_sha256": external_rbac_inventory_sha256,
    }
    if inventory is not None:
        evidence["namespace_inventory_sha256"] = _sha256_json(inventory)
    else:
        services = _kubectl_json(
            ["--context", context, "--namespace", namespace, "get", "services"],
            purpose="LIBERO namespace services inventory",
            kubeconfig=kubeconfig,
        ).get("items")
        if not isinstance(services, list) or services:
            raise RuntimeError(
                "LIBERO namespace must remain free of unreviewed Services"
            )
    access_state = LiberoAccessState(
        kubeconfig=kubeconfig,
        context=context,
        namespace=namespace,
        namespace_uid=namespace_metadata["uid"],
        service_account_uid=account["metadata"]["uid"],
        role_uid=role["metadata"]["uid"],
        role_binding_uid=binding["metadata"]["uid"],
        run_id=run_id,
        payload_pod_name=(
            ""
            if expected_pod_name == LIBERO_PAYLOAD_UNBOUND_RESOURCE_NAME
            else expected_pod_name
        ),
    )
    return evidence, access_state


def _bind_libero_runtime_contract(
    args: argparse.Namespace,
    documents: list[dict[str, Any]],
    *,
    global_config: dict[str, Any],
    infra: str,
    run_id: str,
) -> LiberoRuntimeBinding | None:
    if not _is_libero_invocation(args, documents):
        return None
    from npa.execution_preflight import libero_executable_profile_sha256

    executable_profile_sha256 = libero_executable_profile_sha256(documents)
    executable_profile_b64 = base64.b64encode(
        libero_executable_profile_bytes(documents)
    ).decode("ascii")
    if args.solution_name.strip().lower() != LIBERO_SOLUTION_NAME:
        raise ValueError("the LIBERO profile requires --solution-name libero")
    if args.direct_launch:
        raise ValueError("LIBERO requires managed scheduler submission")
    if not getattr(args, "cleanup", True):
        raise ValueError("LIBERO requires verified managed cleanup")
    if os.environ.get("NPA_BYOF_REFRESH_SKY_API", "1") == "0":
        raise ValueError("LIBERO requires verified isolated Sky API shutdown")
    try:
        image_manifest = libero_image_manifest()
        qualification = validate_libero_qualified_image_manifest(image_manifest)
        libero_publication_lineage_values(
            qualification,
            Path(__file__).resolve().parents[2],
            development_sha=qualification["development_sha"],
        )
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    image = str(args.image or "").strip().removeprefix("docker:")
    if image != qualification["candidate_image"]:
        raise ValueError("LIBERO image differs from checked-in qualified lineage")
    encoded_caller = os.environ.get("NPA_LIBERO_AUTHENTICATED_CALLER_B64", "").strip()
    try:
        caller_bytes = base64.b64decode(encoded_caller, validate=True)
        caller, caller_sha256 = validate_libero_authenticated_caller_assertion(
            caller_bytes, run_id=run_id
        )
    except (ValueError, binascii.Error, RuntimeError) as exc:
        raise ValueError("LIBERO authenticated caller identity is invalid") from exc
    authenticated_customer_identity = str(caller["customer_identity_sha256"])
    if (
        os.environ.get("NPA_LIBERO_AUTHENTICATED_CALLER_SHA256", "").strip()
        != caller_sha256
        or os.environ.get("NPA_LIBERO_CUSTOMER_IDENTITY_SHA256", "").strip()
        != authenticated_customer_identity
    ):
        raise ValueError("LIBERO authenticated caller identity differs")
    encoded_authorization = os.environ.get(
        "NPA_LIBERO_CUSTOMER_AUTHORIZATION_B64", ""
    ).strip()
    try:
        authorization_bytes = base64.b64decode(encoded_authorization, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("LIBERO customer authorization secret is invalid") from exc
    try:
        customer_authorization, authorization_sha256 = (
            validate_libero_customer_runtime_authorization(
                authorization_bytes,
                image_manifest=image_manifest,
                run_id=run_id,
                customer_identity_sha256=authenticated_customer_identity,
                customer_signer_public_key_sha256=str(
                    caller["customer_signer_public_key_sha256"]
                ),
                executable_profile_sha256=executable_profile_sha256,
            )
        )
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    if (
        os.environ.get("NPA_LIBERO_CUSTOMER_AUTHORIZATION_SHA256", "").strip()
        != authorization_sha256
        or os.environ.get("NPA_LIBERO_CUSTOMER_IDENTITY_SHA256", "").strip()
        != customer_authorization["customer_identity_sha256"]
    ):
        raise ValueError("LIBERO customer authorization secret pair differs")
    runtime_manifest_sha256 = hashlib.sha256(
        LIBERO_RUNTIME_MANIFEST.read_bytes()
    ).hexdigest()
    if (
        runtime_manifest_sha256 != qualification["runtime_manifest_sha256"]
        or runtime_manifest_sha256 != customer_authorization["runtime_manifest_sha256"]
    ):
        raise ValueError("LIBERO runtime manifest differs from authorized lineage")
    build_metadata_sha256 = qualification["canonical_build_metadata_sha256"]
    storage_authorization = _validate_libero_output_storage_authorization(
        documents=documents,
        customer_authorization=customer_authorization,
        run_id=run_id,
    )
    if not infra.startswith("k8s/") or not infra.removeprefix("k8s/").strip():
        raise ValueError("LIBERO requires one explicit Kubernetes context")
    allowed_node = _libero_allowed_node(global_config)
    execution_context = infra.removeprefix("k8s/").strip()
    payload_kubeconfig = _libero_payload_kubeconfig()
    payload_kubeconfig_sha256 = hashlib.sha256(
        payload_kubeconfig.read_bytes()
    ).hexdigest()
    payload_context, namespace, payload_cluster_sha256 = _libero_context_contract(
        payload_kubeconfig, require_namespace=True
    )
    if payload_context == execution_context:
        raise ValueError(
            "LIBERO payload and execution contexts must remain explicitly separated"
        )
    execution_kubeconfig_value = os.environ.get("KUBECONFIG", "").strip()
    execution_kubeconfig = _mode_private_regular_file(
        execution_kubeconfig_value, label="NPA execution kubeconfig"
    )
    execution_kubeconfig_sha256 = hashlib.sha256(
        execution_kubeconfig.read_bytes()
    ).hexdigest()
    _, execution_namespace, execution_cluster_sha256 = _libero_context_contract(
        execution_kubeconfig,
        expected_context=execution_context,
        require_namespace=True,
    )
    if payload_cluster_sha256 != execution_cluster_sha256:
        raise ValueError(
            "LIBERO payload and execution kubeconfigs select different clusters"
        )
    if execution_namespace != namespace:
        raise ValueError(
            "LIBERO payload and execution contexts must select the exact run namespace"
        )
    evidence, access_state = _libero_rbac_evidence(
        payload_kubeconfig, payload_context, namespace, run_id
    )
    controller_evidence, controller_identities = _libero_controller_rbac_evidence(
        execution_kubeconfig, execution_context, namespace
    )
    if (
        controller_evidence["external_rbac_inventory_sha256"]
        != evidence["external_rbac_inventory_sha256"]
    ):
        raise ValueError(
            "LIBERO payload and execution contexts observe different external RBAC"
        )
    evidence.update(controller_evidence)
    access_state = replace(
        access_state,
        **controller_identities,
        execution_kubeconfig=execution_kubeconfig,
        execution_context=execution_context,
    )
    evidence["cluster_identity_sha256"] = payload_cluster_sha256
    evidence["allowed_node_sha256"] = hashlib.sha256(allowed_node.encode()).hexdigest()
    evidence["payload_kubeconfig_sha256"] = payload_kubeconfig_sha256
    evidence["execution_kubeconfig_sha256"] = execution_kubeconfig_sha256
    evidence["skypilot_config_sha256"] = _sha256_json(global_config)
    output_authorization_sha256 = storage_authorization["authorization_sha256"]
    output_prefix = str(storage_authorization["output_prefix"])
    output_policy_sha256 = str(storage_authorization["policy_sha256"])
    infrastructure_bundle_sha256 = _sha256_json(
        {
            "schema": "npa.libero.infrastructure-bundle.v2",
            "run_id": run_id,
            **evidence,
            "output_storage_authorization_sha256": output_authorization_sha256,
            "output_storage_prefix_sha256": hashlib.sha256(
                output_prefix.encode()
            ).hexdigest(),
            "output_storage_policy_sha256": output_policy_sha256,
        }
    )
    for document in documents[1:]:
        resources = document.get("resources")
        if (
            not isinstance(resources, dict)
            or resources.get("cloud") != "kubernetes"
            or resources.get("region") not in (None, "", execution_context)
        ):
            raise ValueError("LIBERO requires its exact Kubernetes execution profile")
        resources["region"] = execution_context
        envs = document.setdefault("envs", {})
        for name, value in evidence.items():
            envs[f"NPA_LIBERO_EXPECTED_{name.upper()}"] = value
        envs["NPA_LIBERO_EXPECTED_CANONICAL_BUILD_METADATA_SHA256"] = (
            build_metadata_sha256
        )
        envs["NPA_LIBERO_EXPECTED_CUSTOMER_AUTHORIZATION_EXPIRES_AT"] = (
            customer_authorization["expires_at"]
        )
        envs["NPA_LIBERO_EXPECTED_PUBLICATION_BUNDLE_SHA256"] = qualification[
            "publication_bundle_sha256"
        ]
        envs["NPA_LIBERO_EXPECTED_INFRASTRUCTURE_BUNDLE_SHA256"] = (
            infrastructure_bundle_sha256
        )
        envs["NPA_LIBERO_EXPECTED_OUTPUT_STORAGE_AUTHORIZATION_SHA256"] = (
            output_authorization_sha256
        )
        envs["NPA_LIBERO_EXPECTED_OUTPUT_STORAGE_PREFIX_SHA256"] = hashlib.sha256(
            output_prefix.encode()
        ).hexdigest()
        envs["NPA_LIBERO_EXPECTED_OUTPUT_STORAGE_POLICY_SHA256"] = (
            storage_authorization["policy_sha256"]
        )
        envs["NPA_LIBERO_EXPECTED_EXECUTABLE_PROFILE_SHA256"] = (
            executable_profile_sha256
        )
        envs["NPA_LIBERO_EXECUTABLE_PROFILE_B64"] = executable_profile_b64
        envs["NPA_LIBERO_EXPECTED_CUSTOMER_SIGNER_PUBLIC_KEY_SHA256"] = str(
            caller["customer_signer_public_key_sha256"]
        )
    task_names = {str(document.get("name") or "").strip() for document in documents[1:]}
    if task_names != {LIBERO_PROFILE_TASK_NAME}:
        raise ValueError("LIBERO requires one exact named SkyPilot task")
    return LiberoRuntimeBinding(
        evidence=evidence,
        access_state=access_state,
        customer_authorization_b64=encoded_authorization,
        candidate_image=qualification["candidate_image"],
        task_name=LIBERO_PROFILE_TASK_NAME,
    )


def _libero_payload_pod_record(
    binding: LiberoRuntimeBinding, *, scheduler_job_id: str = ""
) -> tuple[dict[str, Any], dict[str, str]]:
    """Return the sole run Pod after verifying its scheduler and image identity."""

    state = binding.access_state
    payload = _kubectl_json(
        [
            "--context",
            state.context,
            "--namespace",
            state.namespace,
            "get",
            "pods",
        ],
        purpose="LIBERO exact payload Pod inventory",
        kubeconfig=state.kubeconfig,
    )
    items = payload.get("items")
    if not isinstance(items, list):
        raise RuntimeError("LIBERO payload Pod inventory is invalid")
    if not items:
        raise LookupError("LIBERO payload Pod has not been created")
    if len(items) != 1 or not isinstance(items[0], dict):
        raise RuntimeError("LIBERO namespace does not contain exactly one payload Pod")
    pod = items[0]
    metadata = pod.get("metadata") or {}
    spec = pod.get("spec") or {}
    labels = metadata.get("labels") or {}
    annotations = metadata.get("annotations") or {}
    pod_name = str(metadata.get("name") or "")
    pod_uid = str(metadata.get("uid") or "")
    namespace = str(metadata.get("namespace") or "")
    cluster_name = str(labels.get("skypilot-cluster-name") or "")
    display_name = str(annotations.get("skypilot-cluster-name") or "")
    if (
        re.fullmatch(r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?", pod_name) is None
        or not pod_uid
        or not metadata.get("creationTimestamp")
        or namespace != state.namespace
        or not cluster_name
        or pod_name != f"{cluster_name}-head"
        or labels.get("ray-node-type") != "head"
        or labels.get("component") != pod_name
        or not display_name
    ):
        raise RuntimeError("LIBERO payload Pod lacks its exact SkyPilot head identity")
    if scheduler_job_id and display_name != _libero_managed_job_display_name(
        binding.task_name, scheduler_job_id
    ):
        raise RuntimeError(
            "LIBERO payload Pod differs from the submitted scheduler job"
        )
    if state.payload_pod_name and (
        pod_name != state.payload_pod_name
        or pod_uid != state.payload_pod_uid
        or cluster_name != state.skypilot_cluster_name
    ):
        raise RuntimeError("LIBERO payload Pod identity changed after exact binding")
    if spec.get("serviceAccountName") != LIBERO_PAYLOAD_SERVICE_ACCOUNT:
        raise RuntimeError("LIBERO payload Pod used a different service account")
    if spec.get("automountServiceAccountToken") is not True:
        raise RuntimeError("LIBERO payload Pod release token contract is absent")
    containers = spec.get("containers")
    if (
        not isinstance(containers, list)
        or len(containers) != 1
        or not isinstance(containers[0], dict)
        or containers[0].get("image") != binding.candidate_image
        or spec.get("initContainers")
        or spec.get("ephemeralContainers")
    ):
        raise RuntimeError(
            "LIBERO payload Pod differs from the accepted candidate image"
        )
    if any(
        spec.get(name) not in (None, False)
        for name in ("hostNetwork", "hostPID", "hostIPC", "shareProcessNamespace")
    ) or spec.get("runtimeClassName") not in (None, ""):
        raise RuntimeError("LIBERO payload Pod uses an unsafe host or runtime boundary")
    pod_security = spec.get("securityContext") or {}
    if (
        pod_security.get("runAsNonRoot") is not True
        or (pod_security.get("seccompProfile") or {}).get("type") != "RuntimeDefault"
    ):
        raise RuntimeError("LIBERO payload Pod security context is not confined")
    container = containers[0]
    security = container.get("securityContext") or {}
    capabilities = security.get("capabilities") or {}
    if (
        security.get("privileged") not in (None, False)
        or security.get("allowPrivilegeEscalation") is not False
        or security.get("readOnlyRootFilesystem") is not True
        or set(capabilities.get("drop") or ()) != {"ALL"}
        or capabilities.get("add")
        or any(port.get("hostPort") for port in (container.get("ports") or ()))
    ):
        raise RuntimeError("LIBERO payload container security context is not confined")
    volumes = spec.get("volumes") or []
    volume_types: dict[str, str] = {}
    for volume in volumes:
        if not isinstance(volume, dict) or not volume.get("name"):
            raise RuntimeError("LIBERO payload volume identity is invalid")
        kinds = set(volume) - {"name"}
        if len(kinds) != 1 or not kinds <= {"emptyDir", "projected"}:
            raise RuntimeError("LIBERO payload Pod contains a forbidden volume")
        volume_types[str(volume["name"])] = next(iter(kinds))
    safe_writable_roots = (
        "/workspace",
        str(PurePosixPath("/") / "tmp"),
        str(PurePosixPath("/dev") / "shm"),
    )
    for mount in container.get("volumeMounts") or []:
        mount_path = str(mount.get("mountPath") or "")
        volume_type = volume_types.get(str(mount.get("name") or ""))
        if volume_type is None or (
            mount.get("readOnly") is not True
            and not any(
                mount_path == root or mount_path.startswith(root + "/")
                for root in safe_writable_roots
            )
        ):
            raise RuntimeError("LIBERO payload Pod contains a broad writable mount")
    evidence = {
        "payload_pod_name_sha256": hashlib.sha256(pod_name.encode()).hexdigest(),
        "payload_pod_uid_sha256": hashlib.sha256(pod_uid.encode()).hexdigest(),
        "skypilot_cluster_name_sha256": hashlib.sha256(
            cluster_name.encode()
        ).hexdigest(),
    }
    return pod, evidence


def _libero_manager_live_evidence(
    binding: LiberoRuntimeBinding, scheduler_job_id: str
) -> dict[str, Any]:
    """Collect manager-side Pod, image, node, and accelerator evidence."""

    pod, pod_evidence = _libero_payload_pod_record(
        binding, scheduler_job_id=scheduler_job_id
    )
    spec = pod["spec"]
    node_name = str(spec.get("nodeName") or "")
    if hashlib.sha256(node_name.encode()).hexdigest() != binding.evidence.get(
        "allowed_node_sha256"
    ):
        raise RuntimeError("LIBERO payload Pod used a different accepted node")
    containers = spec["containers"]
    resources = containers[0].get("resources") or {}
    requests = resources.get("requests") or {}
    limits = resources.get("limits") or {}
    if (
        str(limits.get("nvidia.com/gpu") or "") != "1"
        or str(requests.get("nvidia.com/gpu") or limits.get("nvidia.com/gpu") or "")
        != "1"
    ):
        raise RuntimeError("LIBERO payload Pod did not request exactly one GPU")
    statuses = (pod.get("status") or {}).get("containerStatuses") or []
    if (
        not isinstance(statuses, list)
        or len(statuses) != 1
        or not isinstance(statuses[0], dict)
        or statuses[0].get("name") != containers[0].get("name")
    ):
        raise RuntimeError("LIBERO payload container status is not exact")
    image_ids = re.findall(
        r"sha256:[0-9a-f]{64}", str(statuses[0].get("imageID") or "")
    )
    expected_digest = binding.candidate_image.rsplit("@", 1)[-1]
    if image_ids != [expected_digest]:
        raise RuntimeError(
            "LIBERO payload status differs from the accepted image digest"
        )

    execution_kubeconfig = binding.access_state.execution_kubeconfig
    if execution_kubeconfig is None:
        raise RuntimeError("LIBERO execution kubeconfig is unavailable")
    node = _kubectl_json(
        [
            "--context",
            binding.access_state.execution_context,
            "get",
            "node",
            node_name,
        ],
        purpose="LIBERO manager-owned live node evidence",
        kubeconfig=execution_kubeconfig,
    )
    node_metadata = node.get("metadata") or {}
    node_status = node.get("status") or {}
    if (
        node_metadata.get("name") != node_name
        or not node_metadata.get("uid")
        or not node_metadata.get("creationTimestamp")
    ):
        raise RuntimeError("LIBERO live node identity is incomplete")
    provider_id = str((node.get("spec") or {}).get("providerID") or "")
    node_info = node_status.get("nodeInfo") or {}
    machine_id = str(node_info.get("machineID") or "")
    system_uuid = str(node_info.get("systemUUID") or "")
    if not provider_id or not machine_id or not system_uuid:
        raise RuntimeError("LIBERO live node provider identity is incomplete")
    try:
        allocatable_gpus = int(
            (node_status.get("allocatable") or {}).get("nvidia.com/gpu", 0)
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("LIBERO live node GPU capacity is invalid") from exc
    if allocatable_gpus < 1:
        raise RuntimeError("LIBERO live node has no allocatable GPU")
    exact_resource = "nvidia.com/b200"
    try:
        b200_count = int((node_status.get("allocatable") or {}).get(exact_resource, 0))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("LIBERO live node B200 inventory is invalid") from exc
    allocated = statuses[0].get("allocatedResourcesStatus") or []
    if (
        b200_count < 1
        or len(allocated) != 1
        or allocated[0].get("name") != exact_resource
        or not allocated[0].get("resources")
        or len(allocated[0]["resources"]) != 1
        or not str(allocated[0]["resources"][0].get("resourceID") or "").startswith(
            "GPU-"
        )
    ):
        raise RuntimeError(
            "LIBERO live Pod lacks authoritative B200 allocation evidence"
        )
    return {
        "schema": "npa.libero.manager-live-evidence.v1",
        "scheduler_job_id_sha256": hashlib.sha256(
            scheduler_job_id.encode()
        ).hexdigest(),
        **pod_evidence,
        "namespace_sha256": hashlib.sha256(
            binding.access_state.namespace.encode()
        ).hexdigest(),
        "node_name_sha256": hashlib.sha256(node_name.encode()).hexdigest(),
        "node_uid_sha256": hashlib.sha256(
            str(node_metadata["uid"]).encode()
        ).hexdigest(),
        "provider_identity_sha256": hashlib.sha256(provider_id.encode()).hexdigest(),
        "machine_identity_sha256": hashlib.sha256(
            f"{machine_id}:{system_uuid}".encode()
        ).hexdigest(),
        "service_account_uid_sha256": hashlib.sha256(
            binding.access_state.service_account_uid.encode()
        ).hexdigest(),
        "pod_observed_image_digest": expected_digest,
        "gpu_family": "B200",
        "pod_gpu_count": 1,
        "node_allocatable_gpu_count": allocatable_gpus,
        "node_allocatable_b200_count": b200_count,
        "device_resource_id_sha256": hashlib.sha256(
            str(allocated[0]["resources"][0]["resourceID"]).encode()
        ).hexdigest(),
        "observation_method": "manager_kubernetes_provider_and_device_plugin",
    }


def _revoke_libero_payload_access(binding: LiberoRuntimeBinding) -> None:
    """Return the payload Role to its inert exact-name rule."""

    _kubectl_json(
        [
            "--context",
            binding.access_state.context,
            "--namespace",
            binding.access_state.namespace,
            "patch",
            "role",
            LIBERO_PAYLOAD_ROLE,
            "--type=json",
            "--patch",
            json.dumps(
                [
                    {
                        "op": "test",
                        "path": "/metadata/uid",
                        "value": binding.access_state.role_uid,
                    },
                    {
                        "op": "replace",
                        "path": "/rules",
                        "value": _libero_payload_rules(
                            LIBERO_PAYLOAD_UNBOUND_RESOURCE_NAME
                        ),
                    },
                ],
                sort_keys=True,
                separators=(",", ":"),
            ),
        ],
        purpose="LIBERO payload Role revocation",
        kubeconfig=binding.access_state.kubeconfig,
    )


def _bind_libero_payload_pod_access(
    binding: LiberoRuntimeBinding,
    scheduler_job_id: str,
    *,
    timeout: int,
    poll_interval: int,
) -> LiberoRuntimeBinding:
    """Bind the initially inert Role to the one observed managed-job Pod."""

    deadline = time.time() + max(timeout, 1)
    while True:
        try:
            pod, pod_evidence = _libero_payload_pod_record(
                binding, scheduler_job_id=scheduler_job_id
            )
            break
        except LookupError:
            if time.time() >= deadline:
                raise RuntimeError(
                    "LIBERO payload Pod was not created before the binding deadline"
                ) from None
            time.sleep(max(poll_interval, 1))
    metadata = pod["metadata"]
    pod_name = str(metadata["name"])
    pod_uid = str(metadata["uid"])
    cluster_name = str(metadata["labels"]["skypilot-cluster-name"])
    initial_rules = _libero_payload_rules(LIBERO_PAYLOAD_UNBOUND_RESOURCE_NAME)
    bound_rules = _libero_payload_rules(pod_name)
    patch = [
        {"op": "test", "path": "/metadata/uid", "value": binding.access_state.role_uid},
        {"op": "test", "path": "/rules", "value": initial_rules},
        {"op": "replace", "path": "/rules", "value": bound_rules},
    ]
    role = _kubectl_json(
        [
            "--context",
            binding.access_state.context,
            "--namespace",
            binding.access_state.namespace,
            "patch",
            "role",
            LIBERO_PAYLOAD_ROLE,
            "--type=json",
            "--patch",
            json.dumps(patch, sort_keys=True, separators=(",", ":")),
        ],
        purpose="LIBERO exact payload Role binding",
        kubeconfig=binding.access_state.kubeconfig,
    )
    if (role.get("metadata") or {}).get(
        "uid"
    ) != binding.access_state.role_uid or role.get("rules") != bound_rules:
        raise RuntimeError("LIBERO payload Role exact binding was not observed")
    release_value = hashlib.sha256(
        f"{pod_uid}:{pod_name}:npa-libero-release-v1".encode()
    ).hexdigest()
    try:
        execution_kubeconfig = binding.access_state.execution_kubeconfig
        if execution_kubeconfig is None:
            raise RuntimeError("LIBERO execution kubeconfig is unavailable")
        released = _kubectl_json(
            [
                "--context",
                binding.access_state.execution_context,
                "--namespace",
                binding.access_state.namespace,
                "patch",
                "pod",
                pod_name,
                "--type=json",
                "--patch",
                json.dumps(
                    [
                        {"op": "test", "path": "/metadata/uid", "value": pod_uid},
                        {
                            "op": "add",
                            "path": "/metadata/annotations/npa.nebius.com~1libero-release",
                            "value": release_value,
                        },
                    ],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ],
            purpose="LIBERO exact payload release",
            kubeconfig=execution_kubeconfig,
        )
        released_metadata = released.get("metadata") or {}
        if (
            released_metadata.get("name") != pod_name
            or released_metadata.get("uid") != pod_uid
            or (released_metadata.get("annotations") or {}).get(
                LIBERO_PAYLOAD_RELEASE_ANNOTATION
            )
            != release_value
        ):
            raise RuntimeError("LIBERO payload release identity differs")
        reread, _ = _libero_payload_pod_record(
            binding, scheduler_job_id=scheduler_job_id
        )
        reread_metadata = reread.get("metadata") or {}
        if (
            reread_metadata.get("name") != pod_name
            or reread_metadata.get("uid") != pod_uid
            or (reread_metadata.get("annotations") or {}).get(
                LIBERO_PAYLOAD_RELEASE_ANNOTATION
            )
            != release_value
        ):
            raise RuntimeError("LIBERO payload Pod changed during exact release")
    except Exception:
        _revoke_libero_payload_access(binding)
        raise
    observed, observed_state = _libero_rbac_evidence(
        binding.access_state.kubeconfig,
        binding.access_state.context,
        binding.access_state.namespace,
        binding.access_state.run_id,
        require_empty_inventory=False,
        expected_pod_name=pod_name,
    )
    for name, value in observed.items():
        if name == "rbac_spec_sha256":
            continue
        if binding.evidence.get(name) != value:
            raise RuntimeError("LIBERO payload identity changed during exact binding")
    state = replace(
        observed_state,
        controller_service_account_uid=binding.access_state.controller_service_account_uid,
        controller_role_uid=binding.access_state.controller_role_uid,
        controller_role_binding_uid=binding.access_state.controller_role_binding_uid,
        execution_kubeconfig=binding.access_state.execution_kubeconfig,
        execution_context=binding.access_state.execution_context,
        payload_pod_name=pod_name,
        payload_pod_uid=pod_uid,
        skypilot_cluster_name=cluster_name,
    )
    evidence = {
        **binding.evidence,
        **pod_evidence,
        "bound_rbac_spec_sha256": observed["rbac_spec_sha256"],
        "scheduler_job_id_sha256": hashlib.sha256(
            scheduler_job_id.encode()
        ).hexdigest(),
        "payload_release_sha256": hashlib.sha256(release_value.encode()).hexdigest(),
    }
    bound = replace(binding, evidence=evidence, access_state=state)
    _libero_payload_pod_record(bound, scheduler_job_id=scheduler_job_id)
    return bound


def _verify_libero_controller_unchanged(binding: LiberoRuntimeBinding) -> None:
    execution_kubeconfig = binding.access_state.execution_kubeconfig
    if execution_kubeconfig is None:
        raise RuntimeError("LIBERO execution kubeconfig is unavailable")
    observed, _ = _libero_controller_rbac_evidence(
        execution_kubeconfig,
        binding.access_state.execution_context,
        binding.access_state.namespace,
    )
    for name, value in observed.items():
        if binding.evidence.get(name) != value:
            raise RuntimeError("LIBERO controller RBAC changed after binding")


def _verify_libero_payload_unchanged(
    binding: LiberoRuntimeBinding,
    *,
    require_empty_inventory: bool,
    allow_bound_pod_absent: bool = False,
) -> None:
    state = binding.access_state

    def refuse(message: str) -> None:
        if state.payload_pod_name:
            _revoke_libero_payload_access(binding)
        raise RuntimeError(message)

    observed, observed_state = _libero_rbac_evidence(
        state.kubeconfig,
        state.context,
        state.namespace,
        state.run_id,
        require_empty_inventory=require_empty_inventory,
        expected_pod_name=(
            state.payload_pod_name or LIBERO_PAYLOAD_UNBOUND_RESOURCE_NAME
        ),
    )
    for name, value in observed.items():
        expected_name = (
            "bound_rbac_spec_sha256"
            if name == "rbac_spec_sha256" and state.payload_pod_name
            else name
        )
        if binding.evidence.get(expected_name) != value:
            refuse("LIBERO payload RBAC changed after binding")
    identities = (
        "namespace_uid",
        "service_account_uid",
        "role_uid",
        "role_binding_uid",
    )
    if any(
        getattr(state, name) != getattr(observed_state, name) for name in identities
    ):
        refuse("LIBERO payload RBAC identity changed after binding")
    if state.payload_pod_name:
        try:
            _, pod_evidence = _libero_payload_pod_record(binding)
        except LookupError:
            if allow_bound_pod_absent:
                _revoke_libero_payload_access(binding)
                return
            refuse("LIBERO payload Pod disappeared before terminal status")
        except RuntimeError:
            _revoke_libero_payload_access(binding)
            raise
        if any(
            binding.evidence.get(name) != value for name, value in pod_evidence.items()
        ):
            refuse("LIBERO payload Pod changed after exact binding")


def _validate_libero_output_storage_authorization(
    *,
    documents: list[dict[str, Any]],
    customer_authorization: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    """Verify short-lived, run-prefix-only upload credentials independently."""

    prefixes = {
        str((document.get("envs") or {}).get("S3_OUTPUT_PREFIX") or "").strip()
        for document in documents[1:]
    }
    if len(prefixes) != 1:
        raise ValueError("LIBERO requires one exact output storage prefix")
    output_prefix = prefixes.pop().rstrip("/") + "/"
    endpoints = {
        str((document.get("envs") or {}).get("AWS_ENDPOINT_URL") or "").rstrip("/")
        for document in documents[1:]
    }
    if len(endpoints) != 1 or not next(iter(endpoints)):
        raise ValueError("LIBERO requires one exact output storage endpoint")
    endpoint = endpoints.pop()
    parsed_endpoint = urlparse(endpoint)
    if (
        parsed_endpoint.scheme != "https"
        or not parsed_endpoint.hostname
        or parsed_endpoint.username is not None
        or parsed_endpoint.password is not None
        or parsed_endpoint.path not in {"", "/"}
        or parsed_endpoint.params
        or parsed_endpoint.query
        or parsed_endpoint.fragment
    ):
        raise ValueError(
            "LIBERO output storage endpoint must be an origin-only HTTPS URL"
        )
    encoded = os.environ.get("NPA_LIBERO_OUTPUT_STORAGE_AUTHORIZATION_B64", "").strip()
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(
            "LIBERO output storage authorization secret is invalid"
        ) from exc
    try:
        authorization, authorization_sha256 = (
            validate_libero_output_storage_authorization(
                payload,
                image_manifest=libero_image_manifest(),
                customer_authorization=customer_authorization,
                run_id=run_id,
                output_prefix=output_prefix,
                endpoint_url=endpoint,
                access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", ""),
                secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
                session_token=os.environ.get("AWS_SESSION_TOKEN", ""),
            )
        )
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    return {**authorization, "authorization_sha256": authorization_sha256}


def _materialize_task_kubernetes_config(document: dict[str, Any]) -> None:
    """Move the NPA resource-profile Kubernetes contract to SkyPilot config."""

    resources = document.get("resources") or {}
    if not isinstance(resources, dict):
        raise ValueError("BYOF task resources must be a mapping")
    kubernetes = resources.pop("kubernetes", None)
    if kubernetes in (None, {}):
        return
    if not isinstance(kubernetes, dict):
        raise ValueError("BYOF task resources.kubernetes must be a mapping")
    config = document.setdefault("config", {})
    if not isinstance(config, dict):
        raise ValueError("BYOF task config must be a mapping")
    existing = config.get("kubernetes")
    if existing not in (None, {}, kubernetes):
        raise ValueError("BYOF task Kubernetes resource and config contracts disagree")
    config["kubernetes"] = kubernetes


def render_workflow(
    yaml_path: Path,
    *,
    run_id: str,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    image: str = "",
    repo_root: str = "/opt/byof",
    smoke_command: str = "",
    solution_name: str = "",
    capability_name: str = "",
    smoke_artifact_name: str = "",
) -> list[dict[str, Any]]:
    docs = _load_yaml_documents(yaml_path)
    for doc in docs[1:]:
        envs = doc.get("envs")
        if not isinstance(envs, dict):
            continue
        envs["NPA_BYOF_RUN_ID"] = run_id
        envs["BYOF_REPO_ROOT"] = repo_root
        envs["BYOF_SMOKE_COMMAND"] = smoke_command
        envs["BYOF_SOLUTION_NAME"] = solution_name
        envs["BYOF_CAPABILITY_NAME"] = capability_name
        envs["BYOF_SMOKE_ARTIFACT_NAME"] = smoke_artifact_name
        normalized_root = _normalize_output_root(output_root)
        output_prefix = normalized_root.rstrip("/") + f"/{run_id}/"
        envs["S3_OUTPUT_PREFIX"] = output_prefix
        envs["NPA_EXECUTION_OUTPUTS"] = json.dumps(
            [{"uri": output_prefix, "kind": "directory"}], separators=(",", ":")
        )
        bucket = _normalize_s3_bucket(normalized_root) or _normalize_s3_bucket(
            os.environ.get("NPA_S3_BUCKET", "")
        )
        if bucket:
            envs["NPA_S3_BUCKET"] = bucket
        storage_env = _resolved_storage_env()
        explicit_endpoint = os.environ.get("NPA_BYOF_S3_ENDPOINT", "").strip()
        for key in (
            "AWS_ENDPOINT_URL",
            "NEBIUS_S3_ENDPOINT",
            "NPA_S3_BUCKET",
        ):
            value = ""
            candidates = (
                (
                    explicit_endpoint,
                    storage_env.get(key, "").strip(),
                    os.environ.get(key, "").strip(),
                )
                if explicit_endpoint
                and key in {"AWS_ENDPOINT_URL", "NEBIUS_S3_ENDPOINT"}
                else (os.environ.get(key, "").strip(), storage_env.get(key, "").strip())
            )
            for candidate in candidates:
                if candidate and not (
                    candidate.startswith("${") and candidate.endswith("}")
                ):
                    value = candidate
                    break
            if not value:
                continue
            # Prefer the bucket derived from --output-root when already set.
            # Compare via `key` only — avoid a second "NPA_S3_BUCKET" literal that
            # trips gitleaks generic-api-key on `key == "…"`.
            if key.endswith("_S3_BUCKET") and envs.get(key):
                continue
            envs[key] = value
        if image:
            image_ref = image.removeprefix("docker:")
            envs["BYOF_IMAGE"] = image_ref
            resources = doc.setdefault("resources", {})
            if isinstance(resources, dict):
                resources["image_id"] = f"docker:{image_ref}"
        _materialize_task_kubernetes_config(doc)
    return docs


def _load_yaml_documents(path: Path) -> list[dict[str, Any]]:
    docs = [
        doc
        for doc in yaml.safe_load_all(path.read_text(encoding="utf-8"))
        if doc is not None
    ]
    if not docs:
        raise ValueError(f"empty SkyPilot YAML: {path}")
    return docs


def _task_docs(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if (
        len(docs) > 1
        and isinstance(docs[0], dict)
        and "execution" in docs[0]
        and "run" not in docs[0]
    ):
        return docs[1:]
    return docs


def _serialized_task_documents(docs: list[dict[str, Any]]) -> bytes:
    """Return the exact executable SkyPilot profile bytes."""

    return yaml.safe_dump_all(_task_docs(docs), sort_keys=False).encode()


def _write_yaml_documents(path: Path, docs: list[dict[str, Any]]) -> None:
    path.write_bytes(_serialized_task_documents(docs))


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("byof-container-%Y%m%dT%H%M%SZ")


def _resolved_storage_env() -> dict[str, str]:
    """Resolve project/host S3 env so rendered BYOF YAMLs are not left with ${...}."""

    project = (
        os.environ.get("NPA_E2E_PROJECT", "").strip()
        or os.environ.get("NPA_PROJECT", "").strip()
        or os.environ.get("NPA_BYOF_PROJECT", "").strip()
    )
    try:
        return dict(
            storage_env_for_project(
                project or None,
                allow_host_creds=True,
                endpoint_url=os.environ.get("NPA_BYOF_S3_ENDPOINT", ""),
            )
        )
    except Exception as exc:  # noqa: BLE001 - best-effort for render/launch paths
        print(f"WARN: skipped BYOF storage env resolution: {exc}", file=sys.stderr)
        return {}


def _libero_output_storage_client() -> Any:
    """Use only the complete control-plane-authorized storage principal."""

    import boto3
    from botocore.config import Config

    access = str(os.environ.get("AWS_ACCESS_KEY_ID") or "")
    secret = str(os.environ.get("AWS_SECRET_ACCESS_KEY") or "")
    session = str(os.environ.get("AWS_SESSION_TOKEN") or "")
    if not all((access, secret, session)):
        raise ValueError(
            "LIBERO output storage requires one complete authorized credential triplet"
        )
    endpoints = {
        str(os.environ.get(name) or "").strip().rstrip("/")
        for name in (
            "AWS_ENDPOINT_URL_S3",
            "AWS_ENDPOINT_URL",
            "NEBIUS_S3_ENDPOINT",
            "NPA_STORAGE_ENDPOINT",
            "S3_ENDPOINT_URL",
        )
        if os.environ.get(name)
    }
    if len(endpoints) != 1:
        raise ValueError("LIBERO output storage requires one exact authorized endpoint")
    return boto3.client(
        "s3",
        endpoint_url=endpoints.pop(),
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        aws_session_token=session,
        config=Config(s3={"addressing_style": "path"}),
    )


@dataclass(frozen=True)
class OutputPrefixLease:
    bucket: str
    key: str
    etag: str
    version_id: str


def _release_output_prefix_lease(lease: OutputPrefixLease) -> None:
    """Delete only the exact version created by this manager transaction."""

    client = _libero_output_storage_client()
    try:
        head = client.head_object(
            Bucket=lease.bucket, Key=lease.key, VersionId=lease.version_id
        )
    except Exception as exc:  # noqa: BLE001 - 404 is provider-specific
        response = getattr(exc, "response", {})
        status = (response.get("ResponseMetadata") or {}).get("HTTPStatusCode")
        if status == 404:
            return
        raise
    if head.get("ETag") != lease.etag or head.get("VersionId") != lease.version_id:
        raise RuntimeError("LIBERO output-prefix lease identity changed")
    client.delete_object(Bucket=lease.bucket, Key=lease.key, VersionId=lease.version_id)
    try:
        client.head_object(
            Bucket=lease.bucket, Key=lease.key, VersionId=lease.version_id
        )
    except Exception as exc:  # noqa: BLE001 - 404 is provider-specific
        response = getattr(exc, "response", {})
        status = (response.get("ResponseMetadata") or {}).get("HTTPStatusCode")
        if status == 404:
            return
        raise
    raise RuntimeError("LIBERO output-prefix lease cleanup is incomplete")


def preflight_output_storage(
    *, output_root: str, run_id: str, libero: bool = False
) -> OutputPrefixLease | None:
    """Reserve a new S3 run prefix and prove it is writable before compute."""

    parsed = urlparse(_normalize_output_root(output_root))
    bucket = parsed.netloc.strip()
    if parsed.scheme != "s3" or not bucket:
        raise ValueError("BYOF output root must be a valid s3:// URI")
    prefix = parsed.path.strip("/")
    run_prefix = "/".join(part for part in (prefix, run_id) if part).rstrip("/") + "/"
    key = run_prefix + (".npa-output-lease" if libero else ".npa-write-preflight")
    if libero:
        client = _libero_output_storage_client()
    else:
        project = (
            os.environ.get("NPA_E2E_PROJECT", "").strip()
            or os.environ.get("NPA_PROJECT", "").strip()
            or os.environ.get("NPA_BYOF_PROJECT", "").strip()
        )
        client = s3_client_for_project(
            project or None,
            allow_host_creds=True,
            endpoint_url=os.environ.get("NPA_BYOF_S3_ENDPOINT", ""),
        )
    created = False
    try:
        existing = client.list_objects_v2(Bucket=bucket, Prefix=run_prefix, MaxKeys=1)
        if existing.get("Contents"):
            raise RuntimeError(
                "refusing to reuse a non-empty BYOF run prefix; choose a new run ID"
            )
        created_response = client.put_object(
            Bucket=bucket,
            Key=key,
            Body=b"npa BYOF write preflight\n",
            ContentType="text/plain",
            IfNoneMatch="*",
        )
        created = True
        version_id = str(created_response.get("VersionId") or "")
        etag = str(created_response.get("ETag") or "")
        if libero and (not version_id or re.fullmatch(r'"[^\"]+"', etag) is None):
            raise RuntimeError("output lease lacks immutable provider identity")
        head_kwargs = {"Bucket": bucket, "Key": key}
        if version_id:
            head_kwargs["VersionId"] = version_id
        head = client.head_object(**head_kwargs)
        if int(head.get("ContentLength", -1)) <= 0:
            raise RuntimeError("S3 write preflight object is unexpectedly empty")
        if libero:
            if head.get("ETag") != etag or head.get("VersionId") != version_id:
                raise RuntimeError("output lease readback identity differs")
            return OutputPrefixLease(bucket, key, etag, version_id)
        client.delete_object(Bucket=bucket, Key=key)
        created = False
        return None
    except Exception as exc:  # noqa: BLE001 - preserve provider error as launch blocker
        if created:
            try:
                delete_kwargs = {"Bucket": bucket, "Key": key}
                if "version_id" in locals() and version_id:
                    delete_kwargs["VersionId"] = version_id
                client.delete_object(**delete_kwargs)
            except Exception:  # noqa: BLE001 - retain the original preflight failure
                pass
        raise RuntimeError(
            f"BYOF output storage preflight failed for s3://{bucket}/{prefix}: {exc}"
        ) from exc


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # `resolve_byof_profile_path` also accepts a bare packaged profile NAME, which is what
    # an npa.workflow spec must pass: a stage runs in a pod with no repo checkout.
    parser.add_argument(
        "--yaml", dest="yaml_path", type=resolve_byof_profile_path, default=DEFAULT_YAML
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument("--image", default="")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default="/opt/byof")
    parser.add_argument("--smoke-command", default="")
    parser.add_argument("--solution-name", default="")
    parser.add_argument("--capability-name", default="")
    parser.add_argument("--smoke-artifact-name", default="")
    parser.add_argument("--config-path", default="")
    parser.add_argument("--infra", default=os.environ.get("NPA_BYOF_INFRA", ""))
    parser.add_argument("--sky-bin", default="")
    parser.add_argument(
        "--secret-env",
        action="append",
        default=None,
        help=(
            "Env var name to forward as a SkyPilot task secret (repeatable). "
            "Defaults to the S3 credentials the profile needs for its uploads."
        ),
    )
    parser.add_argument("--submit-timeout", type=int, default=600)
    parser.add_argument(
        "--wait-timeout",
        type=int,
        default=3600,
        help="0 checks once, positive values bound the wait, and -1 waits until terminal.",
    )
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--isolated-config-dir", default="")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument(
        "--direct-launch",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("NPA_BYOF_DIRECT_LAUNCH", "1") != "0",
    )
    parser.add_argument(
        "--cleanup", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return _submit_and_wait(args)
    except (
        SkyPilotNotInstalledError,
        SkyPilotConfigError,
        SkyPilotVersionError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


def _wait_for_terminal(
    scheduler_job_id: str,
    *,
    sky_bin: str,
    isolated_config_dir: Path | None = None,
    config_path: Path | None = None,
    wait_timeout: int,
    poll_interval: int,
    observation_guard: Callable[[str], None] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Poll with explicit immediate, bounded, or indefinite semantics."""

    if wait_timeout < -1:
        raise ValueError("--wait-timeout must be -1, zero, or a positive number")
    mode = (
        "indefinite"
        if wait_timeout == -1
        else ("immediate" if wait_timeout == 0 else "bounded")
    )
    deadline = None if wait_timeout == -1 else time.time() + wait_timeout
    statuses: list[str] = []
    status_kwargs = {
        "isolated_config_dir": isolated_config_dir,
        "config_path": config_path,
        "sky_bin": sky_bin,
    }
    final = workflow_status(scheduler_job_id, **status_kwargs)
    statuses.append(final.status)
    if observation_guard is not None:
        observation_guard(final.status)
    polls = 1
    while (
        final.status not in TERMINAL_STATUSES
        and wait_timeout != 0
        and (deadline is None or time.time() < deadline)
    ):
        time.sleep(max(poll_interval, 1))
        final = workflow_status(scheduler_job_id, **status_kwargs)
        statuses.append(final.status)
        if observation_guard is not None:
            observation_guard(final.status)
        polls += 1
    diagnostics = {
        "mode": mode,
        "polls": polls,
        "statuses": statuses,
        "terminal": final.status in TERMINAL_STATUSES,
        "deadline_exhausted": bool(
            wait_timeout > 0
            and final.status not in TERMINAL_STATUSES
            and deadline is not None
            and time.time() >= deadline
        ),
    }
    if not diagnostics["terminal"]:
        diagnostics["stuck_state"] = final.status
        diagnostics["hint"] = (
            "workflow is not terminal; inspect SkyPilot controller/job and pod events"
        )
    return final, diagnostics


def _exact_scheduler_job_id(value: Any) -> str:
    """Return one positive numeric scheduler ID without normalizing text."""

    candidate = str(value or "")
    if candidate != candidate.strip() or not re.fullmatch(r"[1-9][0-9]*", candidate):
        raise ValueError(
            "workflow submission returned no exact numeric scheduler job ID"
        )
    return candidate


def _cancel_exact_managed_job(
    scheduler_job_id: str,
    *,
    teardown_guard: SignalTeardown,
    sky_bin: str,
    isolated_config_dir: Path | None,
    config_path: Path | None,
    poll_interval: int,
) -> CleanupResult:
    """Cancel one exact managed job in its submitted scheduler state."""

    cleanup = CleanupResult()
    cleanup.commands.append([sky_bin, "jobs", "cancel", "--yes", scheduler_job_id])
    try:
        cancel = cancel_workflow_job(
            sky_bin=sky_bin,
            job_id=scheduler_job_id,
            run_id=teardown_guard.run_id,
            isolated_config_dir=isolated_config_dir,
            config_path=config_path,
            timeout=max(int(teardown_guard.timeout), 1),
            poll_seconds=max(float(poll_interval), 0.1),
            also_down_cluster=False,
        )
    except Exception:  # noqa: BLE001 - preserve resources on cancellation ambiguity
        cleanup.errors.append("exact managed-job cancellation raised unexpectedly")
        return cleanup
    if cancel["cancel_returncode"] != 0:
        cleanup.errors.append("exact managed-job cancellation failed")
    else:
        cleanup.resources_removed.append(f"managed-job:{scheduler_job_id}")
    return cleanup


def _cancel_then_teardown_managed_job(
    scheduler_job_id: str,
    *,
    teardown_guard: SignalTeardown,
    sky_bin: str,
    isolated_config_dir: Path | None,
    config_path: Path | None,
    poll_interval: int,
) -> CleanupResult:
    """Cancel and drain one exact managed job before tearing down its clusters."""

    cleanup = CleanupResult()
    if not scheduler_job_id:
        cleanup.errors.append(
            "scheduler job ID is unavailable; preserving possible managed resources"
        )
        return cleanup
    status_kwargs = {
        "sky_bin": sky_bin,
        "isolated_config_dir": isolated_config_dir,
        "config_path": config_path,
        "poll_interval": poll_interval,
    }
    try:
        final, _ = _wait_for_terminal(scheduler_job_id, wait_timeout=0, **status_kwargs)
    except Exception:  # noqa: BLE001 - cleanup must still attempt exact cancellation
        final = None
    if final is None or final.status not in VERIFIED_DRAIN_STATUSES:
        cleanup.extend(
            _cancel_exact_managed_job(
                scheduler_job_id,
                teardown_guard=teardown_guard,
                **status_kwargs,
            )
        )
        if not cleanup.ok:
            return cleanup
        try:
            final, _ = _wait_for_terminal(
                scheduler_job_id,
                wait_timeout=max(int(teardown_guard.timeout), 1),
                **status_kwargs,
            )
        except Exception:  # noqa: BLE001 - ambiguous drain must preserve resources
            cleanup.errors.append(
                "managed job drain status could not be verified after exact "
                "cancellation; preserving its clusters"
            )
            return cleanup
    if final is None or final.status not in VERIFIED_DRAIN_STATUSES:
        cleanup.errors.append(
            "managed job did not reach a verified terminal or absent state; "
            "preserving its clusters"
        )
        return cleanup
    try:
        cleanup.extend(teardown_guard.teardown())
    except Exception:  # noqa: BLE001 - never claim teardown or absence on ambiguity
        cleanup.errors.append("run-cluster teardown raised unexpectedly")
        return cleanup
    if cleanup.ok:
        absence = _verify_managed_clusters_absent(
            run_id=teardown_guard.run_id,
            sky_bin=sky_bin,
            isolated_config_dir=isolated_config_dir,
            config_path=config_path,
            timeout=max(int(teardown_guard.timeout), 1),
        )
        cleanup.extend(absence)
        cleanup.verified = absence.verified
        cleanup.remote_absence_verified = absence.remote_absence_verified
    return cleanup


def _strict_cluster_names(output: str) -> list[str]:
    """Return names from one exact SkyPilot cluster inventory document."""

    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        raise ValueError("inventory was not exact JSON") from exc
    if isinstance(payload, list):
        clusters = payload
    elif (
        isinstance(payload, dict)
        and set(payload) == {"clusters"}
        and isinstance(payload["clusters"], list)
    ):
        clusters = payload["clusters"]
    else:
        raise ValueError("inventory had an invalid schema")
    names: list[str] = []
    for cluster in clusters:
        if not isinstance(cluster, dict):
            raise ValueError("inventory contained an invalid row")
        if any(key in cluster for key in ("error", "errors", "exception")):
            raise ValueError("inventory contained contradictory error metadata")
        name_fields = [key for key in ("name", "cluster") if key in cluster]
        if not name_fields:
            raise ValueError("inventory contained an invalid row")
        if len(name_fields) > 1:
            raise ValueError("inventory contained an ambiguous name row")
        name = cluster[name_fields[0]]
        if (
            not isinstance(name, str)
            or not name
            or name != name.strip()
            or not re.fullmatch(r"[A-Za-z0-9._-]+", name)
        ):
            raise ValueError("inventory contained an invalid row")
        names.append(name)
    return names


def _verify_managed_clusters_absent(
    *,
    run_id: str,
    sky_bin: str,
    isolated_config_dir: Path | None,
    config_path: Path | None,
    timeout: int,
) -> CleanupResult:
    """Fail closed unless a fresh structured inventory proves run absence."""

    cmd = [sky_bin, "status", "--refresh", "--output", "json"]
    if config_path is not None:
        cmd[2:2] = ["--config", str(config_path)]
    cleanup = CleanupResult(commands=[cmd])
    try:
        result = subprocess.run(
            cmd,
            env=sky_environment(isolated_config_dir),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except Exception:  # noqa: BLE001 - absence must remain unverified
        cleanup.errors.append("post-teardown SkyPilot cluster inventory raised")
        return cleanup
    if result.returncode != 0:
        cleanup.errors.append("post-teardown SkyPilot cluster inventory command failed")
        return cleanup
    try:
        names = _strict_cluster_names(result.stdout)
    except ValueError as exc:
        cleanup.errors.append(f"post-teardown SkyPilot cluster {exc}")
        return cleanup
    patterns = cluster_name_patterns_for_run(run_id)
    matches = [
        name
        for name in names
        if any(fnmatchcase(name, pattern) for pattern in patterns)
    ]
    if matches:
        cleanup.errors.append(
            "post-teardown inventory still contains run-owned SkyPilot clusters"
        )
        return cleanup
    cleanup.verified = True
    cleanup.remote_absence_verified = True
    return cleanup


def _cleanup_libero_access_objects(
    state: LiberoAccessState, *, timeout: int
) -> CleanupResult:
    """Delete exact UID-bound RBAC objects and their per-run namespace."""

    cleanup = CleanupResult()
    try:
        from kubernetes import client as kubernetes_client
        from kubernetes import config as kubernetes_config
        from kubernetes.client.exceptions import ApiException

        api_client = kubernetes_config.new_client_from_config(
            config_file=str(state.kubeconfig), context=state.context
        )
        core = kubernetes_client.CoreV1Api(api_client)
        rbac = kubernetes_client.RbacAuthorizationV1Api(api_client)
    except Exception:  # noqa: BLE001 - access cleanup ambiguity is fatal
        cleanup.errors.append("LIBERO Kubernetes cleanup client initialization failed")
        return cleanup

    delete_options = kubernetes_client.V1DeleteOptions
    preconditions = kubernetes_client.V1Preconditions
    operations = (
        (
            "controller-rolebinding",
            LIBERO_CONTROLLER_ROLE_BINDING,
            state.controller_role_binding_uid,
            rbac.delete_namespaced_role_binding,
            rbac.read_namespaced_role_binding,
        ),
        (
            "controller-role",
            LIBERO_CONTROLLER_ROLE,
            state.controller_role_uid,
            rbac.delete_namespaced_role,
            rbac.read_namespaced_role,
        ),
        (
            "controller-serviceaccount",
            SKYPILOT_ENGINE_SERVICE_ACCOUNT,
            state.controller_service_account_uid,
            core.delete_namespaced_service_account,
            core.read_namespaced_service_account,
        ),
        (
            "rolebinding",
            LIBERO_PAYLOAD_ROLE_BINDING,
            state.role_binding_uid,
            rbac.delete_namespaced_role_binding,
            rbac.read_namespaced_role_binding,
        ),
        (
            "role",
            LIBERO_PAYLOAD_ROLE,
            state.role_uid,
            rbac.delete_namespaced_role,
            rbac.read_namespaced_role,
        ),
        (
            "serviceaccount",
            LIBERO_PAYLOAD_SERVICE_ACCOUNT,
            state.service_account_uid,
            core.delete_namespaced_service_account,
            core.read_namespaced_service_account,
        ),
    )
    for kind, name, uid, delete, read in operations:
        if not uid:
            cleanup.errors.append(f"LIBERO {kind} cleanup identity is unavailable")
            continue
        try:
            delete(
                name,
                state.namespace,
                body=delete_options(preconditions=preconditions(uid=uid)),
            )
        except ApiException as exc:
            if exc.status != 404:
                cleanup.errors.append(f"LIBERO {kind} cleanup could not prove absence")
        except Exception:  # noqa: BLE001 - never claim an ambiguous delete
            cleanup.errors.append(f"LIBERO {kind} cleanup raised unexpectedly")
        deadline = time.time() + max(timeout, 1)
        absent = False
        while True:
            try:
                read(name, state.namespace)
            except ApiException as exc:
                if exc.status == 404:
                    absent = True
                    break
                cleanup.errors.append(f"LIBERO {kind} absence check failed")
                break
            except Exception:  # noqa: BLE001 - absence must be objective
                cleanup.errors.append(f"LIBERO {kind} absence check raised")
                break
            if time.time() >= deadline:
                cleanup.errors.append(
                    f"LIBERO {kind} still exists after exact deletion"
                )
                break
            time.sleep(1)
        if absent:
            cleanup.resources_removed.append(f"libero-{kind}")

    namespace_absent = False
    if not state.namespace_uid:
        cleanup.errors.append("LIBERO namespace cleanup identity is unavailable")
    else:
        try:
            core.delete_namespace(
                state.namespace,
                body=delete_options(
                    preconditions=preconditions(uid=state.namespace_uid)
                ),
            )
        except ApiException as exc:
            if exc.status != 404:
                cleanup.errors.append("LIBERO namespace exact deletion failed")
        except Exception:  # noqa: BLE001 - preserve ambiguity but keep verifying
            cleanup.errors.append("LIBERO namespace exact deletion failed")
        deadline = time.time() + max(timeout, 1)
        while True:
            try:
                core.read_namespace(state.namespace)
            except ApiException as exc:
                if exc.status == 404:
                    namespace_absent = True
                    break
                cleanup.errors.append("LIBERO namespace absence check failed")
                break
            except Exception:  # noqa: BLE001 - absence must be objective
                cleanup.errors.append("LIBERO namespace absence check raised")
                break
            if time.time() >= deadline:
                cleanup.errors.append(
                    "LIBERO namespace still exists after exact deletion"
                )
                break
            time.sleep(1)
    if namespace_absent:
        cleanup.resources_removed.append("libero-namespace")
    cleanup.verified = not cleanup.errors and namespace_absent
    cleanup.remote_absence_verified = cleanup.verified
    return cleanup


def _cleanup_libero_local_state(
    *, isolated_state_root: Path, payload_kubeconfig: Path, run_id: str
) -> CleanupResult:
    """Remove only exact run-scoped local state after remote absence and API stop."""

    cleanup = CleanupResult()
    try:
        state_root = _libero_isolated_state_root(isolated_state_root, run_id)
        kubeconfig = _mode_private_regular_file(
            str(payload_kubeconfig), label="payload kubeconfig"
        )
        if kubeconfig.parent != state_root:
            raise RuntimeError(
                "payload kubeconfig is not inside the exact-run isolated state"
            )
        kubeconfig.unlink()
        if kubeconfig.exists() or kubeconfig.is_symlink():
            raise RuntimeError("payload kubeconfig still exists")
        shutil.rmtree(state_root)
        if state_root.exists() or state_root.is_symlink():
            raise RuntimeError("isolated SkyPilot state still exists")
    except Exception:  # noqa: BLE001 - local absence must remain unverified
        cleanup.errors.append("LIBERO exact local-state cleanup failed")
        return cleanup
    cleanup.resources_removed.extend(
        ["libero-payload-kubeconfig", "libero-isolated-skypilot-state"]
    )
    cleanup.verified = True
    cleanup.remote_absence_verified = True
    return cleanup


def _complete_libero_cleanup(
    cleanup: CleanupResult,
    *,
    binding: LiberoRuntimeBinding,
    timeout: int,
    sky_bin: str,
    isolated_config_dir: Path | None,
    config_path: Path | str | None,
    run_id: str,
) -> tuple[CleanupResult, bool, bool]:
    """Finish the ordered LIBERO cleanup transaction after managed absence.

    Returns the combined result, whether API stop was attempted, and whether
    API stop succeeded.  Local recovery state is removed only after remote
    managed resources and exact access objects are objectively absent and the
    isolated SkyPilot API has stopped.
    """

    managed_absent = bool(
        cleanup.ok and cleanup.verified and cleanup.remote_absence_verified
    )
    if not managed_absent:
        return cleanup, False, False

    access_cleanup = _cleanup_libero_access_objects(
        binding.access_state, timeout=max(timeout, 1)
    )
    cleanup.extend(access_cleanup)
    cleanup.verified = cleanup.verified and access_cleanup.verified
    cleanup.remote_absence_verified = (
        cleanup.remote_absence_verified and access_cleanup.remote_absence_verified
    )
    access_absent = bool(
        cleanup.ok and cleanup.verified and cleanup.remote_absence_verified
    )
    if not access_absent:
        return cleanup, False, False
    if isolated_config_dir is None:
        cleanup.errors.append(
            "LIBERO isolated state is unavailable; local recovery state preserved"
        )
        cleanup.verified = False
        return cleanup, False, False

    try:
        _stop_sky_api(
            sky_bin=sky_bin,
            isolated_config_dir=isolated_config_dir,
            config_path=config_path,
        )
    except SkyPilotConfigError:
        cleanup.errors.append(
            "LIBERO SkyPilot API stop failed; local recovery state preserved"
        )
        cleanup.verified = False
        return cleanup, True, False
    cleanup.resources_removed.append("libero-skypilot-api")

    local_cleanup = _cleanup_libero_local_state(
        isolated_state_root=isolated_config_dir,
        payload_kubeconfig=binding.access_state.kubeconfig,
        run_id=run_id,
    )
    cleanup.extend(local_cleanup)
    cleanup.verified = cleanup.verified and local_cleanup.verified
    cleanup.remote_absence_verified = (
        cleanup.remote_absence_verified and local_cleanup.remote_absence_verified
    )
    return cleanup, True, True


def _submit_and_wait(args: argparse.Namespace) -> int:
    run_id = args.run_id or _default_run_id()
    output_root = _normalize_output_root(args.output_root)
    docs = render_workflow(
        args.yaml_path,
        run_id=run_id,
        output_root=output_root,
        image=args.image,
        repo_root=args.repo_root,
        smoke_command=args.smoke_command,
        solution_name=args.solution_name,
        capability_name=args.capability_name,
        smoke_artifact_name=args.smoke_artifact_name,
    )
    is_libero = _is_libero_invocation(args, docs)
    if _uses_libero_payload_service_account(docs) and not is_libero:
        raise ValueError(
            "the reserved LIBERO payload service account requires an explicit "
            "LIBERO solution or profile"
        )
    if is_libero and args.solution_name != LIBERO_SOLUTION_NAME:
        raise ValueError("the LIBERO profile requires --solution-name libero")
    if is_libero:
        cluster_name_patterns_for_run(run_id)
    outputs = {
        "root": output_root.rstrip("/") + f"/{run_id}/",
        "receipt": output_root.rstrip("/") + f"/{run_id}/npa_upload_receipt.json",
    }

    if args.render_only:
        render_dir = Path(tempfile.mkdtemp(prefix=f"npa-byof-container-{run_id}-"))
        rendered_yaml = render_dir / "byof-container.rendered.yaml"
        _write_yaml_documents(rendered_yaml, docs)
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "rendered_yaml": str(rendered_yaml),
                    "outputs": outputs,
                },
                indent=2,
            )
        )
        return 0

    with tempfile.TemporaryDirectory(prefix=f"npa-byof-container-{run_id}-") as tmp:
        tmp_path = Path(tmp)
        previous_kubeconfig = os.environ.get("KUBECONFIG")
        previous_customer_authorization = os.environ.get(
            "NPA_LIBERO_CUSTOMER_AUTHORIZATION_B64"
        )
        previous_payload_kubeconfig = os.environ.get("NPA_LIBERO_PAYLOAD_KUBECONFIG")
        sky_bin = str(
            resolve_sky_bin(args.sky_bin or os.environ.get("NPA_SKYPILOT_BIN"))
        )
        isolated_config_dir = resolve_isolated_config_dir(
            args.isolated_config_dir or None
        )
        if is_libero:
            isolated_config_dir = _libero_isolated_state_root(
                isolated_config_dir, run_id
            )
        api_stop_attempted = False
        api_stop_succeeded = False
        preserve_api = False
        api_config_path: Path | None = None
        libero_binding: LiberoRuntimeBinding | None = None
        signal_previous_handlers: dict[int, Any] | None = None
        active_signal_cleanup: Callable[[], CleanupResult] | None = None
        early_cleanup_result: CleanupResult | None = None
        submission_started = False
        submission_absence_proven = False
        output_lease: OutputPrefixLease | None = None
        try:
            rendered_yaml = Path(tmp) / "byof-container.rendered.yaml"
            infra = args.infra or _default_infra()
            if is_libero:
                payload_source = _mode_private_regular_file(
                    os.environ.get("NPA_LIBERO_PAYLOAD_KUBECONFIG", ""),
                    label="payload kubeconfig",
                )
                execution_source = _mode_private_regular_file(
                    os.environ.get("KUBECONFIG", ""),
                    label="NPA execution kubeconfig",
                )
                config_source = Path(_libero_global_config_path(args))
                config_path = str(config_source)
            else:
                _normalize_kubeconfig_current_context(tmp_path)
                config_path = args.config_path or _write_default_k8s_config(
                    tmp_path, infra
                )
            api_config_path = Path(config_path) if config_path else None
            global_config: dict[str, Any] = {}
            if config_path:
                loaded_config = yaml.safe_load(
                    Path(config_path).read_text(encoding="utf-8")
                )
                if loaded_config is not None and not isinstance(loaded_config, dict):
                    raise ValueError("SkyPilot global config must be a mapping")
                global_config = loaded_config or {}
            verify_solution_payload_service_accounts(docs, global_config=global_config)
            libero_binding = _bind_libero_runtime_contract(
                args,
                docs,
                global_config=global_config,
                infra=infra,
                run_id=run_id,
            )
            if libero_binding is not None:
                # Authentication, authorization, inline-secret rejection, and
                # exact profile validation above are entirely in memory and
                # use the owner-supplied inputs directly. Only now may the
                # submission transaction create its immutable input copies.
                payload_copy = _immutable_exact_run_copy(
                    payload_source,
                    isolated_config_dir / "libero-payload-kubeconfig",
                    label="payload kubeconfig",
                )
                execution_copy = _immutable_exact_run_copy(
                    execution_source,
                    isolated_config_dir / "libero-execution-kubeconfig",
                    label="execution kubeconfig",
                )
                config_copy = _immutable_exact_run_copy(
                    config_source,
                    isolated_config_dir / "libero-skypilot-config.yaml",
                    label="SkyPilot global config",
                )
                os.environ["NPA_LIBERO_PAYLOAD_KUBECONFIG"] = str(payload_copy)
                os.environ["KUBECONFIG"] = str(execution_copy)
                config_path = str(config_copy)
                api_config_path = config_copy
                libero_binding = replace(
                    libero_binding,
                    access_state=replace(
                        libero_binding.access_state,
                        kubeconfig=payload_copy,
                        execution_kubeconfig=execution_copy,
                    ),
                )
                os.environ["NPA_LIBERO_CUSTOMER_AUTHORIZATION_B64"] = (
                    libero_binding.customer_authorization_b64
                )

                def cleanup_before_submission() -> CleanupResult:
                    """Retract exact bound access when no scheduler job can exist."""

                    nonlocal early_cleanup_result
                    nonlocal api_stop_attempted, api_stop_succeeded, preserve_api
                    if early_cleanup_result is not None:
                        return early_cleanup_result
                    managed_absent = CleanupResult()
                    managed_absent.verified = True
                    managed_absent.remote_absence_verified = True
                    early_cleanup_result, attempted, stopped = _complete_libero_cleanup(
                        managed_absent,
                        binding=libero_binding,
                        timeout=max(int(args.submit_timeout), 1),
                        sky_bin=sky_bin,
                        isolated_config_dir=isolated_config_dir,
                        config_path=api_config_path,
                        run_id=run_id,
                    )
                    api_stop_attempted = api_stop_attempted or attempted
                    api_stop_succeeded = api_stop_succeeded or stopped
                    preserve_api = not stopped
                    return early_cleanup_result

                active_signal_cleanup = cleanup_before_submission

                def cleanup_active_transaction() -> CleanupResult:
                    assert active_signal_cleanup is not None
                    return active_signal_cleanup()

                signal_previous_handlers = install_teardown_signal_handlers(
                    cleanup_active_transaction
                )
            _ensure_infra_enabled(
                sky_bin=sky_bin,
                infra=infra,
                config_path=config_path,
                isolated_config_dir=isolated_config_dir,
            )
            # Recheck the accepted namespaced controller grant immediately
            # before submission. Any ClusterRoleBinding or wildcard drift stops
            # before the untrusted payload can be created.
            if libero_binding is not None:
                _verify_libero_payload_unchanged(
                    libero_binding, require_empty_inventory=True
                )
                _verify_libero_controller_unchanged(libero_binding)
            output_lease = preflight_output_storage(
                output_root=output_root, run_id=run_id, libero=is_libero
            )
            if libero_binding is not None:
                if output_lease is None:
                    raise RuntimeError("LIBERO output-prefix lease is unavailable")
                for document in docs[1:]:
                    envs = document.setdefault("envs", {})
                    envs["NPA_LIBERO_EXPECTED_OUTPUT_LEASE_KEY"] = output_lease.key
                    envs["NPA_LIBERO_EXPECTED_OUTPUT_LEASE_ETAG"] = output_lease.etag
                    envs["NPA_LIBERO_EXPECTED_OUTPUT_LEASE_VERSION_ID"] = (
                        output_lease.version_id
                    )
            _write_yaml_documents(rendered_yaml, docs)
            if args.direct_launch:
                return _direct_launch(
                    rendered_yaml=rendered_yaml,
                    run_id=run_id,
                    outputs=outputs,
                    sky_bin=sky_bin,
                    infra=infra,
                    config_path=config_path,
                    isolated_config_dir=isolated_config_dir,
                    cleanup=args.cleanup,
                    secret_envs=resolve_secret_envs(
                        args.secret_env, solution_name=args.solution_name
                    ),
                )
            teardown_guard = SignalTeardown(
                run_id=run_id,
                isolated_config_dir=isolated_config_dir,
                sky_bin=sky_bin,
                poll_interval=max(float(args.poll_interval), 0.0),
            )
            scheduler_job_id = ""
            submitted_config_path = Path(config_path) if config_path else None
            expected_submission_config_path = (
                isolated_config_dir / "submissions" / run_id / "skypilot-config.yaml"
                if isolated_config_dir is not None
                else None
            )
            cleanup_result: CleanupResult | None = None
            cleanup_started = False

            def cleanup_submission() -> CleanupResult:
                nonlocal cleanup_result, cleanup_started
                if cleanup_started:
                    if cleanup_result is not None:
                        return cleanup_result
                    pending = CleanupResult()
                    pending.outcome = "unsafe"
                    pending.errors.append(
                        "cleanup is already in progress; absence is not verified"
                    )
                    return pending
                cleanup_started = True
                try:
                    if not scheduler_job_id and submission_absence_proven:
                        cleanup_result = _verify_managed_clusters_absent(
                            run_id=run_id,
                            sky_bin=sky_bin,
                            isolated_config_dir=teardown_guard.isolated_config_dir,
                            config_path=submitted_config_path,
                            timeout=max(int(teardown_guard.timeout), 1),
                        )
                    else:
                        cleanup_result = _cancel_then_teardown_managed_job(
                            scheduler_job_id,
                            teardown_guard=teardown_guard,
                            sky_bin=sky_bin,
                            isolated_config_dir=teardown_guard.isolated_config_dir,
                            config_path=submitted_config_path,
                            poll_interval=args.poll_interval,
                        )
                except Exception:  # noqa: BLE001 - never claim ambiguous cleanup
                    cleanup_result = CleanupResult()
                    cleanup_result.outcome = "unsafe"
                    cleanup_result.errors.append(
                        "cleanup transaction failed; owned-resource absence is unverified"
                    )
                return cleanup_result

            def cleanup_after_signal() -> CleanupResult:
                nonlocal cleanup_result
                nonlocal api_stop_attempted, api_stop_succeeded, preserve_api
                cleanup_result = cleanup_submission()
                if libero_binding is not None:
                    cleanup_result, attempted, stopped = _complete_libero_cleanup(
                        cleanup_result,
                        binding=libero_binding,
                        timeout=max(int(teardown_guard.timeout), 1),
                        sky_bin=sky_bin,
                        isolated_config_dir=isolated_config_dir,
                        config_path=api_config_path,
                        run_id=run_id,
                    )
                    api_stop_attempted = api_stop_attempted or attempted
                    api_stop_succeeded = api_stop_succeeded or stopped
                    preserve_api = not stopped
                return cleanup_result

            if libero_binding is not None:
                previous_handlers = None
            else:
                previous_handlers = install_teardown_signal_handlers(
                    cleanup_after_signal
                )
            summary: dict[str, Any] | None = None
            return_code = 1
            libero_live_evidence: dict[str, Any] | None = None
            try:
                teardown_guard.mark_launched()
                submit_config_path = Path(config_path) if config_path else None
                if libero_binding is not None:
                    _verify_libero_payload_unchanged(
                        libero_binding, require_empty_inventory=True
                    )
                    _verify_libero_controller_unchanged(libero_binding)
                submission_started = True
                active_signal_cleanup = cleanup_after_signal
                result = submit_workflow(
                    rendered_yaml,
                    run_id,
                    isolated_config_dir=isolated_config_dir,
                    config_path=submit_config_path,
                    sky_bin=sky_bin,
                    infra=infra,
                    secret_envs=resolve_secret_envs(
                        args.secret_env, solution_name=args.solution_name
                    ),
                    timeout=args.submit_timeout,
                )
                scheduler_job_id = _exact_scheduler_job_id(result.job_id)
                if libero_binding is not None:
                    libero_binding = _bind_libero_payload_pod_access(
                        libero_binding,
                        scheduler_job_id,
                        timeout=args.submit_timeout,
                        poll_interval=args.poll_interval,
                    )
                    _verify_libero_payload_unchanged(
                        libero_binding, require_empty_inventory=False
                    )
                    _verify_libero_controller_unchanged(libero_binding)
                submitted_config_path = (
                    Path(result.log_paths["config"])
                    if result.log_paths.get("config")
                    else submit_config_path
                )
                api_config_path = submitted_config_path
                teardown_guard.mark_launched(config_path=submitted_config_path)
                summary = {
                    "run_id": run_id,
                    "submit": result.__dict__,
                    "outputs": outputs,
                }
                if libero_binding is not None:
                    summary["libero_runtime_binding"] = libero_binding.evidence
                observation_guard: Callable[[str], None] | None = None
                if libero_binding is not None:

                    def verify_libero_observation(status: str) -> None:
                        nonlocal libero_live_evidence
                        _verify_libero_payload_unchanged(
                            libero_binding,
                            require_empty_inventory=False,
                            allow_bound_pod_absent=status in TERMINAL_STATUSES,
                        )
                        _verify_libero_controller_unchanged(libero_binding)
                        if status == "RUNNING" and libero_live_evidence is None:
                            libero_live_evidence = _libero_manager_live_evidence(
                                libero_binding, scheduler_job_id
                            )

                    observation_guard = verify_libero_observation
                final, wait_diagnostics = _wait_for_terminal(
                    scheduler_job_id,
                    sky_bin=sky_bin,
                    isolated_config_dir=teardown_guard.isolated_config_dir,
                    config_path=submitted_config_path,
                    wait_timeout=args.wait_timeout,
                    poll_interval=args.poll_interval,
                    observation_guard=observation_guard,
                )
                summary["final"] = final.__dict__
                summary["wait"] = wait_diagnostics
                if libero_binding is not None:
                    if final.status == "SUCCEEDED":
                        if libero_live_evidence is None:
                            raise RuntimeError(
                                "LIBERO completed without retained live hardware evidence"
                            )
                        summary["libero_manager_live_evidence"] = libero_live_evidence
                    _verify_libero_payload_unchanged(
                        libero_binding,
                        require_empty_inventory=False,
                        allow_bound_pod_absent=True,
                    )
                    _verify_libero_controller_unchanged(libero_binding)
                return_code = 0 if final.status == "SUCCEEDED" else 1
                if (
                    not is_libero
                    and os.environ.get("NPA_ISAAC_LAB_ACCEPT_PRECHECK_FAILURE") == "1"
                    and final.status == "FAILED_PRECHECKS"
                ):
                    return_code = 0
            except SkyPilotSubmitError as exc:
                transaction = exc.transaction
                reconciled_scheduler_job_id = False
                generated_config_recovered = False
                if transaction is not None and transaction.job_id:
                    reconciled_scheduler_job_id = True
                    try:
                        scheduler_job_id = _exact_scheduler_job_id(transaction.job_id)
                    except ValueError:
                        scheduler_job_id = ""
                if (
                    transaction is not None
                    and not transaction.job_id
                    and transaction.existence == "absent"
                ):
                    submission_absence_proven = True
                if scheduler_job_id and expected_submission_config_path is not None:
                    try:
                        submitted_config_path = _mode_private_regular_file(
                            str(expected_submission_config_path),
                            label="generated SkyPilot submission config",
                        )
                    except ValueError:
                        scheduler_job_id = ""
                    else:
                        generated_config_recovered = True
                        api_config_path = submitted_config_path
                teardown_guard.mark_launched(config_path=submitted_config_path)
                summary = {
                    "run_id": run_id,
                    "submit": {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "scheduler_job_id_recovered": bool(scheduler_job_id),
                        "reconciled_scheduler_job_id": (reconciled_scheduler_job_id),
                        "generated_config_recovered": generated_config_recovered,
                    },
                    "outputs": outputs,
                }
                if libero_binding is not None:
                    summary["libero_runtime_binding"] = libero_binding.evidence
                return_code = 2
            except Exception as exc:  # noqa: BLE001 - cleanup must still be reported
                summary = {
                    "run_id": run_id,
                    "submit": {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "scheduler_job_id_recovered": bool(scheduler_job_id),
                    },
                    "outputs": outputs,
                }
                if libero_binding is not None:
                    summary["libero_runtime_binding"] = libero_binding.evidence
                return_code = 2
            finally:
                if previous_handlers is not None:
                    restore_signal_handlers(previous_handlers)
                if args.cleanup:
                    cleanup_result = cleanup_submission()
            cleanup_verified = bool(
                cleanup_result is not None
                and cleanup_result.ok
                and cleanup_result.verified
                and cleanup_result.remote_absence_verified
            )
            if cleanup_verified and libero_binding is not None:
                assert cleanup_result is not None
                cleanup_result, attempted, stopped = _complete_libero_cleanup(
                    cleanup_result,
                    binding=libero_binding,
                    timeout=max(int(teardown_guard.timeout), 1),
                    sky_bin=sky_bin,
                    isolated_config_dir=isolated_config_dir,
                    config_path=api_config_path,
                    run_id=run_id,
                )
                api_stop_attempted = api_stop_attempted or attempted
                api_stop_succeeded = api_stop_succeeded or stopped
                preserve_api = not stopped
                cleanup_verified = bool(
                    cleanup_result.ok
                    and cleanup_result.verified
                    and cleanup_result.remote_absence_verified
                )
                if not cleanup_verified:
                    return_code = 1
                    preserve_api = True
            if cleanup_result is not None and not cleanup_verified:
                return_code = 1
                if is_libero and not api_stop_succeeded:
                    preserve_api = True
            if preserve_api:
                if summary is not None:
                    summary["api_stop"] = {
                        "ok": False,
                        "preserved_for_cleanup_recovery": True,
                    }
            elif api_stop_attempted:
                if summary is not None:
                    summary["api_stop"] = {
                        "ok": api_stop_succeeded,
                        "preserved_for_cleanup_recovery": False,
                    }
            elif os.environ.get("NPA_BYOF_REFRESH_SKY_API", "1") != "0":
                api_stop_attempted = True
                try:
                    _stop_sky_api(
                        sky_bin=sky_bin,
                        isolated_config_dir=isolated_config_dir,
                        config_path=api_config_path,
                    )
                except SkyPilotConfigError:
                    if summary is None:
                        raise
                    summary["api_stop"] = {
                        "ok": False,
                        "preserved_for_cleanup_recovery": False,
                    }
                    return_code = 1
                else:
                    api_stop_succeeded = True
                    if summary is not None:
                        summary["api_stop"] = {
                            "ok": True,
                            "preserved_for_cleanup_recovery": False,
                        }
            if summary is not None and cleanup_result is not None:
                summary["cleanup"] = {
                    "ok": cleanup_verified,
                    "errors": cleanup_result.errors,
                    "resources_removed": cleanup_result.resources_removed,
                    "verified": cleanup_result.verified,
                    "remote_absence_verified": (cleanup_result.remote_absence_verified),
                }
            print(json.dumps(summary or {"run_id": run_id}, indent=2, sort_keys=True))
            return return_code
        finally:
            if (
                libero_binding is not None
                and not submission_started
                and early_cleanup_result is None
                and active_signal_cleanup is not None
            ):
                early_cleanup_result = active_signal_cleanup()
            if signal_previous_handlers is not None:
                restore_signal_handlers(signal_previous_handlers)
            if output_lease is not None:
                _release_output_prefix_lease(output_lease)
            # Restore KUBECONFIG and stop the Sky API so a temp kubeconfig path
            # written under TemporaryDirectory cannot poison later sky launches.
            if previous_kubeconfig is None:
                os.environ.pop("KUBECONFIG", None)
            else:
                os.environ["KUBECONFIG"] = previous_kubeconfig
            if previous_customer_authorization is None:
                os.environ.pop("NPA_LIBERO_CUSTOMER_AUTHORIZATION_B64", None)
            else:
                os.environ["NPA_LIBERO_CUSTOMER_AUTHORIZATION_B64"] = (
                    previous_customer_authorization
                )
            if previous_payload_kubeconfig is None:
                os.environ.pop("NPA_LIBERO_PAYLOAD_KUBECONFIG", None)
            else:
                os.environ["NPA_LIBERO_PAYLOAD_KUBECONFIG"] = (
                    previous_payload_kubeconfig
                )
            if (
                os.environ.get("NPA_BYOF_REFRESH_SKY_API", "1") != "0"
                and not api_stop_attempted
                and not preserve_api
            ):
                _stop_sky_api(
                    sky_bin=sky_bin,
                    isolated_config_dir=isolated_config_dir,
                    config_path=api_config_path,
                )


def _direct_launch(
    *,
    rendered_yaml: Path,
    run_id: str,
    outputs: dict[str, str],
    sky_bin: str,
    infra: str,
    config_path: str = "",
    isolated_config_dir: Path | None = None,
    cleanup: bool = True,
    secret_envs: list[str] | None = None,
) -> int:
    cmd = [
        sky_bin,
        "launch",
        "--yes",
        "--cluster",
        run_id,
        "--name",
        run_id,
    ]
    if cleanup:
        cmd.append("--down")
    if infra:
        # Prefer k8s/<context> form when we know the context; bare "kubernetes"
        # is fine once sky check has enabled it.
        cmd.extend(["--infra", infra])
    if config_path:
        cmd.extend(["--config", config_path])
    launch_env = sky_environment(isolated_config_dir)
    for secret_name in secret_envs or ():
        if launch_env.get(secret_name):
            cmd.extend(["--secret", secret_name])
    cmd.append(str(rendered_yaml))
    # Ensure kubeconfig is visible to sky even when only KUBECONTEXT was set.
    if not launch_env.get("KUBECONFIG"):
        default_kube = Path.home() / ".kube" / "config"
        if default_kube.is_file():
            launch_env["KUBECONFIG"] = str(default_kube)
    result = subprocess.run(
        cmd,
        env=launch_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    summary = {
        "run_id": run_id,
        "mode": "direct-launch",
        "outputs": outputs,
        "command": cmd,
        "final": {
            "status": "SUCCEEDED" if result.returncode == 0 else "FAILED",
            "returncode": result.returncode,
        },
    }
    if result.stdout:
        summary["stdout_tail"] = result.stdout[-8000:]
    if result.stderr:
        summary["stderr_tail"] = result.stderr[-8000:]
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result.returncode == 0 else 1


def _default_infra() -> str:
    configured = (
        os.environ.get("NPA_BYOF_INFRA", "").strip()
        or os.environ.get("NPA_SKYPILOT_INFRA", "").strip()
    )
    if configured:
        return configured
    context = (
        os.environ.get("NPA_BYOF_K8S_CONTEXT", "")
        or os.environ.get("NPA_K8S_CONTEXT", "")
        or os.environ.get("KUBECONTEXT", "")
    ).strip()
    if context:
        return f"k8s/{context}"
    return "kubernetes"


def _normalize_kubeconfig_current_context(
    tmp_path: Path, *, immutable: bool = False
) -> None:
    kubeconfig = os.environ.get("KUBECONFIG", "").strip()
    context = (
        os.environ.get("KUBECONTEXT", "")
        or os.environ.get("NPA_BYOF_K8S_CONTEXT", "")
        or os.environ.get("NPA_K8S_CONTEXT", "")
    ).strip()
    if not kubeconfig or not context:
        return
    if not immutable:
        path = Path(kubeconfig)
        if not path.is_file():
            return
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return
        data["current-context"] = context
        target = tmp_path / "kubeconfig"
        target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        os.environ["KUBECONFIG"] = str(target)
        return
    source = _immutable_exact_run_copy(
        Path(kubeconfig).expanduser(),
        tmp_path / "execution-kubeconfig.source",
        label="execution kubeconfig",
    )
    try:
        data = yaml.safe_load(
            _stable_owner_private_bytes(
                source, label="immutable execution kubeconfig"
            ).decode("utf-8")
        )
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError("LIBERO execution kubeconfig is invalid") from exc
    if not isinstance(data, dict):
        raise ValueError("LIBERO execution kubeconfig must be a mapping")
    data["current-context"] = context
    target = tmp_path / "kubeconfig"
    payload = yaml.safe_dump(data, sort_keys=False).encode()
    descriptor = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o400,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.environ["KUBECONFIG"] = str(target)


def _write_default_k8s_config(tmp_path: Path, infra: str) -> str:
    normalized = infra.strip().lower()
    if not (normalized.startswith("k8s") or normalized.startswith("kubernetes")):
        return ""
    context = infra.split("/", 1)[1].strip() if "/" in infra else ""
    kubernetes_config: dict[str, Any] = {
        "pod_config": {
            "spec": {
                "imagePullSecrets": [
                    {"name": name} for name in DEFAULT_IMAGE_PULL_SECRETS
                ],
                "serviceAccountName": SKYPILOT_ENGINE_SERVICE_ACCOUNT,
            }
        }
    }
    if context:
        # An inherited Sky config can carry an allowlist for a different
        # workload cluster. Pin the explicitly selected infra context so `sky
        # check` and `sky launch` cannot silently diverge.
        kubernetes_config["allowed_contexts"] = [context]
    path = tmp_path / "skypilot-byof-k8s-config.yaml"
    path.write_text(
        yaml.safe_dump(
            {"kubernetes": kubernetes_config},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return str(path)


def _json_values_from_mixed_output(text: str) -> list[Any]:
    """Decode every JSON value embedded in command prose, in source order."""

    decoder = json.JSONDecoder()
    values: list[Any] = []
    index = 0
    while index < len(text):
        starts = [
            position
            for token in ("{", "[")
            if (position := text.find(token, index)) >= 0
        ]
        if not starts:
            break
        start = min(starts)
        try:
            candidate, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            index = start + 1
            continue
        values.append(candidate)
        index = max(end, start + 1)
    return values


def _kubernetes_value_enables_compute(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() == "compute"
    if isinstance(value, list):
        return any(_kubernetes_value_enables_compute(item) for item in value)
    if not isinstance(value, dict):
        return False

    enabled = value.get("enabled")
    status = str(value.get("status") or value.get("state") or "").strip().lower()
    if enabled is False or status in {"disabled", "error", "failed", "unavailable"}:
        return False
    compute = value.get("compute")
    if compute is True or (
        isinstance(compute, str) and compute.strip().lower() == "enabled"
    ):
        return True
    for key in ("capabilities", "features", "enabled_capabilities"):
        if key in value and _kubernetes_value_enables_compute(value[key]):
            return True
    return False


def _has_enabled_kubernetes_compute(value: Any) -> bool:
    """Traverse the complete response; a disabled entry does not end the search."""

    if isinstance(value, list):
        return any(_has_enabled_kubernetes_compute(item) for item in value)
    if not isinstance(value, dict):
        return False
    if _is_error_payload(value):
        return False
    for key, item in value.items():
        if str(
            key
        ).strip().lower() == "kubernetes" and _kubernetes_value_enables_compute(item):
            return True
        if _has_enabled_kubernetes_compute(item):
            return True
    return False


def _is_error_payload(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if any(value.get(key) not in (None, "", [], {}) for key in ("error", "errors")):
        return True
    return str(value.get("status") or "").strip().lower() in {"error", "failed"}


def _contains_error_payload(value: Any) -> bool:
    """Return whether any structured branch reports an error."""

    if isinstance(value, list):
        return any(_contains_error_payload(item) for item in value)
    if not isinstance(value, dict):
        return False
    return _is_error_payload(value) or any(
        _contains_error_payload(item) for item in value.values()
    )


def _ensure_infra_enabled(
    *,
    sky_bin: str,
    infra: str,
    config_path: str = "",
    isolated_config_dir: Path | None = None,
) -> None:
    if os.environ.get("NPA_BYOF_SKIP_SKY_CHECK") == "1":
        return
    normalized = infra.strip().lower()
    if not (normalized.startswith("kubernetes") or normalized.startswith("k8s")):
        return
    if os.environ.get("NPA_BYOF_REFRESH_SKY_API", "1") != "0":
        _stop_sky_api(
            sky_bin=sky_bin,
            isolated_config_dir=isolated_config_dir,
            config_path=config_path,
        )
    cmd = [sky_bin, "check", "kubernetes", "-o", "json"]
    if config_path:
        cmd.extend(["--config", config_path])
    result = subprocess.run(
        cmd,
        env=sky_environment(isolated_config_dir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    enabled = False
    if result.returncode == 0:
        combined = "\n".join(part for part in (result.stdout, result.stderr) if part)
        payloads = _json_values_from_mixed_output(combined)
        enabled = (
            bool(payloads)
            and not any(_contains_error_payload(payload) for payload in payloads)
            and any(_has_enabled_kubernetes_compute(payload) for payload in payloads)
        )
    if result.returncode != 0 or not enabled:
        detail = (result.stderr or result.stdout or "").strip()
        raise SkyPilotConfigError(
            "SkyPilot Kubernetes check did not enable compute before BYOF smoke "
            f"submission: {detail or 'empty structured result'}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
