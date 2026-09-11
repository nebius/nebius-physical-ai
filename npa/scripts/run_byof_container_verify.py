#!/usr/bin/env python3
"""Submit BYOF container-verify SkyPilot workloads (CPU smoke for /opt/byof clone)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from npa.execution_preflight import (
    SKYPILOT_ENGINE_SERVICE_ACCOUNT,
    verify_solution_payload_service_accounts,
)
from npa.workflows.byof.live import resolve_byof_profile_path
from npa.clients.project_credentials import (
    s3_client_for_project,
    storage_env_for_project,
)
from npa.orchestration.skypilot import submit_workflow, workflow_status
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
LIBERO_PROFILE_FILENAME = "byof-solution-smoke-libero-b200-gpu.yaml"


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
}
DEFAULT_SECRET_ENVS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
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
    # Operator acceptance is runtime state, not workflow configuration. Always
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
        if not isinstance(pod_spec, dict):
            continue
        if pod_spec.get("serviceAccountName") == LIBERO_PAYLOAD_SERVICE_ACCOUNT:
            return True
    return False


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _libero_allowed_node(global_config: dict[str, Any]) -> str:
    kubernetes = global_config.get("kubernetes") or {}
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
    ):
        raise ValueError("LIBERO requires exactly one non-empty allowed node name")
    return names[0].strip()


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
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o077:
        raise ValueError(f"LIBERO {label} must be a mode-private regular file")
    return path.resolve()


def _libero_payload_kubeconfig() -> Path:
    return _mode_private_regular_file(
        os.environ.get("NPA_LIBERO_PAYLOAD_KUBECONFIG", "").strip(),
        label="payload kubeconfig",
    )


def _libero_isolated_state_root(path: Path | None) -> Path:
    if path is None:
        raise ValueError("LIBERO requires an isolated SkyPilot state root")
    if path.is_symlink() or not path.is_dir():
        raise ValueError("LIBERO isolated SkyPilot state must be a directory")
    metadata = path.stat()
    if metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
        raise ValueError("LIBERO isolated SkyPilot state must be owner-private")
    return path.resolve()


def _libero_global_config_path(args: argparse.Namespace) -> str:
    configured = args.config_path or os.environ.get(
        "NPA_LIBERO_SKYPILOT_CONFIG", ""
    ).strip()
    return str(
        _mode_private_regular_file(configured, label="SkyPilot global config")
    )


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
        item for item in config.get("clusters") or [] if item.get("name") == selected_cluster
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


def _libero_rbac_evidence(
    kubeconfig: Path, context: str, namespace: str
) -> dict[str, str]:
    account = _libero_resource(
        kubeconfig,
        context,
        namespace,
        "serviceaccount",
        LIBERO_PAYLOAD_SERVICE_ACCOUNT,
    )
    role = _libero_resource(
        kubeconfig, context, namespace, "role", LIBERO_PAYLOAD_ROLE
    )
    binding = _libero_resource(
        kubeconfig,
        context,
        namespace,
        "rolebinding",
        LIBERO_PAYLOAD_ROLE_BINDING,
    )
    rules = [{"apiGroups": [""], "resources": ["pods"], "verbs": ["get"]}]
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
        raise RuntimeError("LIBERO payload Role is broader than pods/get")
    if binding.get("subjects") != subjects or binding.get("roleRef") != role_ref:
        raise RuntimeError("LIBERO payload RoleBinding differs from the reviewed contract")
    return {
        "service_account_uid_sha256": hashlib.sha256(
            account["metadata"]["uid"].encode()
        ).hexdigest(),
        "role_uid_sha256": hashlib.sha256(
            role["metadata"]["uid"].encode()
        ).hexdigest(),
        "role_binding_uid_sha256": hashlib.sha256(
            binding["metadata"]["uid"].encode()
        ).hexdigest(),
        "rbac_spec_sha256": _sha256_json(
            {"rules": rules, "subjects": subjects, "roleRef": role_ref}
        ),
        "namespace_sha256": hashlib.sha256(namespace.encode()).hexdigest(),
    }


def _bind_libero_runtime_contract(
    args: argparse.Namespace,
    documents: list[dict[str, Any]],
    *,
    global_config: dict[str, Any],
    infra: str,
) -> dict[str, str] | None:
    if not _is_libero_invocation(args, documents):
        return None
    if args.solution_name.strip().lower() != LIBERO_SOLUTION_NAME:
        raise ValueError("the LIBERO profile requires --solution-name libero")
    if args.direct_launch:
        raise ValueError("LIBERO requires managed scheduler submission")
    if not infra.startswith("k8s/") or not infra.removeprefix("k8s/").strip():
        raise ValueError("LIBERO requires one explicit Kubernetes context")
    allowed_node = _libero_allowed_node(global_config)
    execution_context = infra.removeprefix("k8s/").strip()
    payload_kubeconfig = _libero_payload_kubeconfig()
    payload_context, namespace, payload_cluster_sha256 = _libero_context_contract(
        payload_kubeconfig, require_namespace=True
    )
    if payload_context == execution_context:
        raise ValueError(
            "LIBERO payload and execution contexts must remain explicitly separated"
        )
    execution_kubeconfig_value = os.environ.get("KUBECONFIG", "").strip()
    execution_kubeconfig = Path(execution_kubeconfig_value)
    if not execution_kubeconfig_value or not execution_kubeconfig.is_file():
        raise ValueError("LIBERO requires the selected NPA execution kubeconfig")
    _, _, execution_cluster_sha256 = _libero_context_contract(
        execution_kubeconfig,
        expected_context=execution_context,
        require_namespace=False,
    )
    if payload_cluster_sha256 != execution_cluster_sha256:
        raise ValueError("LIBERO payload and execution kubeconfigs select different clusters")
    evidence = _libero_rbac_evidence(payload_kubeconfig, payload_context, namespace)
    evidence["cluster_identity_sha256"] = payload_cluster_sha256
    evidence["allowed_node_sha256"] = hashlib.sha256(allowed_node.encode()).hexdigest()
    for name, observed in evidence.items():
        variable = f"NPA_LIBERO_EXPECTED_{name.upper()}"
        expected = os.environ.get(variable, "").strip()
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"LIBERO requires owner-receipted hash {variable}")
        if expected != observed:
            raise ValueError(f"LIBERO owner-receipted hash differs for {name}")
    for document in documents[1:]:
        envs = document.setdefault("envs", {})
        for name, value in evidence.items():
            envs[f"NPA_LIBERO_EXPECTED_{name.upper()}"] = value
    return evidence


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


def _write_yaml_documents(path: Path, docs: list[dict[str, Any]]) -> None:
    path.write_text(
        yaml.safe_dump_all(_task_docs(docs), sort_keys=False), encoding="utf-8"
    )


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


def preflight_output_storage(*, output_root: str, run_id: str) -> None:
    """Reserve a new S3 run prefix and prove it is writable before compute."""

    parsed = urlparse(_normalize_output_root(output_root))
    bucket = parsed.netloc.strip()
    if parsed.scheme != "s3" or not bucket:
        raise ValueError("BYOF output root must be a valid s3:// URI")
    prefix = parsed.path.strip("/")
    run_prefix = "/".join(part for part in (prefix, run_id) if part).rstrip("/") + "/"
    key = run_prefix + ".npa-write-preflight"
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
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=b"npa BYOF write preflight\n",
            ContentType="text/plain",
            IfNoneMatch="*",
        )
        created = True
        head = client.head_object(Bucket=bucket, Key=key)
        if int(head.get("ContentLength", -1)) <= 0:
            raise RuntimeError("S3 write preflight object is unexpectedly empty")
        client.delete_object(Bucket=bucket, Key=key)
        created = False
    except Exception as exc:  # noqa: BLE001 - preserve provider error as launch blocker
        if created:
            try:
                client.delete_object(Bucket=bucket, Key=key)
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
    polls = 1
    while (
        final.status not in TERMINAL_STATUSES
        and wait_timeout != 0
        and (deadline is None or time.time() < deadline)
    ):
        time.sleep(max(poll_interval, 1))
        final = workflow_status(scheduler_job_id, **status_kwargs)
        statuses.append(final.status)
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
    cleanup.commands.append([sky_bin, "jobs", "cancel", "--yes", scheduler_job_id])
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
    final, _ = _wait_for_terminal(scheduler_job_id, wait_timeout=0, **status_kwargs)
    if final.status not in VERIFIED_DRAIN_STATUSES:
        cleanup.extend(
            _cancel_exact_managed_job(
                scheduler_job_id,
                teardown_guard=teardown_guard,
                **status_kwargs,
            )
        )
        if not cleanup.ok:
            return cleanup
        final, _ = _wait_for_terminal(
            scheduler_job_id,
            wait_timeout=max(int(teardown_guard.timeout), 1),
            **status_kwargs,
        )
    if final.status not in VERIFIED_DRAIN_STATUSES:
        cleanup.errors.append(
            "managed job did not reach a verified terminal or absent state; "
            "preserving its clusters"
        )
        return cleanup
    cleanup.extend(teardown_guard.teardown())
    if cleanup.ok:
        cleanup.extend(
            _verify_managed_clusters_absent(
                run_id=teardown_guard.run_id,
                sky_bin=sky_bin,
                isolated_config_dir=isolated_config_dir,
                config_path=config_path,
                timeout=max(int(teardown_guard.timeout), 1),
            )
        )
    return cleanup


def _strict_cluster_names(output: str) -> list[str]:
    """Return names from one exact SkyPilot cluster inventory document."""

    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        raise ValueError("inventory was not exact JSON") from exc
    if isinstance(payload, list):
        clusters = payload
    elif isinstance(payload, dict) and isinstance(payload.get("clusters"), list):
        clusters = payload["clusters"]
    else:
        raise ValueError("inventory had an invalid schema")
    names: list[str] = []
    for cluster in clusters:
        name = (
            (cluster.get("name") or cluster.get("cluster"))
            if isinstance(cluster, dict)
            else None
        )
        if not isinstance(name, str) or not name.strip():
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
    result = subprocess.run(
        cmd,
        env=sky_environment(isolated_config_dir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    cleanup = CleanupResult(commands=[cmd])
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
    if is_libero and args.solution_name != LIBERO_SOLUTION_NAME:
        raise ValueError("the LIBERO profile requires --solution-name libero")
    outputs = {
        "root": output_root.rstrip("/") + f"/{run_id}/",
        "summary": output_root.rstrip("/") + f"/{run_id}/npa_byof_summary.json",
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
        sky_bin = str(
            resolve_sky_bin(args.sky_bin or os.environ.get("NPA_SKYPILOT_BIN"))
        )
        isolated_config_dir = resolve_isolated_config_dir(
            args.isolated_config_dir or None
        )
        if is_libero:
            isolated_config_dir = _libero_isolated_state_root(isolated_config_dir)
        try:
            _normalize_kubeconfig_current_context(tmp_path)
            rendered_yaml = Path(tmp) / "byof-container.rendered.yaml"
            infra = args.infra or _default_infra()
            config_path = (
                _libero_global_config_path(args)
                if is_libero
                else args.config_path or _write_default_k8s_config(tmp_path, infra)
            )
            global_config: dict[str, Any] = {}
            if config_path:
                loaded_config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
                if loaded_config is not None and not isinstance(loaded_config, dict):
                    raise ValueError("SkyPilot global config must be a mapping")
                global_config = loaded_config or {}
            verify_solution_payload_service_accounts(
                docs, global_config=global_config
            )
            libero_binding = _bind_libero_runtime_contract(
                args, docs, global_config=global_config, infra=infra
            )
            _write_yaml_documents(rendered_yaml, docs)
            preflight_output_storage(output_root=output_root, run_id=run_id)
            _ensure_infra_enabled(
                sky_bin=sky_bin,
                infra=infra,
                config_path=config_path,
                isolated_config_dir=isolated_config_dir,
            )
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
            cleanup_result: CleanupResult | None = None
            cleanup_started = False

            def cleanup_submission() -> CleanupResult:
                nonlocal cleanup_result, cleanup_started
                if cleanup_started:
                    return cleanup_result or CleanupResult()
                cleanup_started = True
                cleanup_result = _cancel_then_teardown_managed_job(
                    scheduler_job_id,
                    teardown_guard=teardown_guard,
                    sky_bin=sky_bin,
                    isolated_config_dir=teardown_guard.isolated_config_dir,
                    config_path=submitted_config_path,
                    poll_interval=args.poll_interval,
                )
                return cleanup_result

            previous_handlers = install_teardown_signal_handlers(cleanup_submission)
            summary: dict[str, Any] | None = None
            return_code = 1
            try:
                teardown_guard.mark_launched()
                submit_config_path = Path(config_path) if config_path else None
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
                scheduler_job_id = str(result.job_id or "").strip()
                if not scheduler_job_id:
                    raise RuntimeError(
                        "workflow submission returned no scheduler job ID"
                    )
                submitted_config_path = (
                    Path(result.log_paths["config"])
                    if result.log_paths.get("config")
                    else submit_config_path
                )
                teardown_guard.mark_launched(config_path=submitted_config_path)
                summary = {
                    "run_id": run_id,
                    "submit": result.__dict__,
                    "outputs": outputs,
                }
                if libero_binding is not None:
                    summary["libero_runtime_binding"] = libero_binding
                final, wait_diagnostics = _wait_for_terminal(
                    scheduler_job_id,
                    sky_bin=sky_bin,
                    isolated_config_dir=teardown_guard.isolated_config_dir,
                    config_path=submitted_config_path,
                    wait_timeout=args.wait_timeout,
                    poll_interval=args.poll_interval,
                )
                summary["final"] = final.__dict__
                summary["wait"] = wait_diagnostics
                return_code = 0 if final.status == "SUCCEEDED" else 1
                if (
                    os.environ.get("NPA_ISAAC_LAB_ACCEPT_PRECHECK_FAILURE") == "1"
                    and final.status == "FAILED_PRECHECKS"
                ):
                    return_code = 0
            finally:
                restore_signal_handlers(previous_handlers)
                if args.cleanup:
                    cleanup_result = cleanup_submission()
            if summary is not None and cleanup_result is not None:
                summary["cleanup"] = {
                    "ok": cleanup_result.ok,
                    "errors": cleanup_result.errors,
                    "resources_removed": cleanup_result.resources_removed,
                }
                if not cleanup_result.ok:
                    return_code = 1
            print(json.dumps(summary or {"run_id": run_id}, indent=2, sort_keys=True))
            return return_code
        finally:
            # Restore KUBECONFIG and stop the Sky API so a temp kubeconfig path
            # written under TemporaryDirectory cannot poison later sky launches.
            if previous_kubeconfig is None:
                os.environ.pop("KUBECONFIG", None)
            else:
                os.environ["KUBECONFIG"] = previous_kubeconfig
            if os.environ.get("NPA_BYOF_REFRESH_SKY_API", "1") != "0":
                subprocess.run(
                    [sky_bin, "api", "stop"],
                    env=sky_environment(isolated_config_dir),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
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


def _normalize_kubeconfig_current_context(tmp_path: Path) -> None:
    kubeconfig = os.environ.get("KUBECONFIG", "").strip()
    context = (
        os.environ.get("KUBECONTEXT", "")
        or os.environ.get("NPA_BYOF_K8S_CONTEXT", "")
        or os.environ.get("NPA_K8S_CONTEXT", "")
    ).strip()
    if not kubeconfig or not context:
        return
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
        subprocess.run(
            [sky_bin, "api", "stop"],
            env=sky_environment(isolated_config_dir),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
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
