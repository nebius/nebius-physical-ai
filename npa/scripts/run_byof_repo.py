#!/usr/bin/env python3
"""Build/push a BYOF image from an OSS repo and launch a live workload on Nebius."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import io
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from npa.clients.config import resolve_container_registry
from npa.clients.project_credentials import storage_env_for_project
from npa.deploy.images import (
    LIBERO_AUTHENTICATED_CALLER_PUBLIC_KEY_FILE_ENV,
    LIBERO_OUTPUT_STORAGE_AUTHORIZATION_PUBLIC_KEY_FILE_ENV,
    LiberoCustomerAuthorizationDenied,
    container_image_for_tool,
    libero_customer_acceptance_notification,
    libero_image_manifest,
    libero_publication_lineage_values,
    validate_libero_authenticated_caller_assertion,
    validate_libero_customer_runtime_authorization,
    validate_libero_qualified_image_manifest,
    wan_accepted_image_manifest,
)
from npa.orchestration.npa_workflow.robotwin_preflight import (
    BUILD_COMMAND_SHA256 as ROBOTWIN_BUILD_COMMAND_SHA256,
    CHILD_BUCKET_ENV as ROBOTWIN_CHILD_BUCKET_ENV,
    CHILD_CONFIG_PATH_ENV as ROBOTWIN_CHILD_CONFIG_PATH_ENV,
    CHILD_IMAGE_ENV as ROBOTWIN_CHILD_IMAGE_ENV,
    CHILD_OUTPUT_PREFIX_ENV as ROBOTWIN_CHILD_OUTPUT_PREFIX_ENV,
    CHILD_OUTPUT_ROOT_ENV as ROBOTWIN_CHILD_OUTPUT_ROOT_ENV,
    CHILD_RUNTIME_AUTH_ENV as ROBOTWIN_CHILD_RUNTIME_AUTH_ENV,
    CHILD_RUN_ID_ENV as ROBOTWIN_CHILD_RUN_ID_ENV,
    CONTEXT_ENV_NAMES as ROBOTWIN_CONTEXT_ENV_NAMES,
    CUSTOMER_ENTITLEMENT_ENV as ROBOTWIN_CUSTOMER_ENTITLEMENT_ENV,
    CUSTOMER_TERMS as ROBOTWIN_CUSTOMER_TERMS,
    INVOCATION as ROBOTWIN_INVOCATION,
    PUBLIC_CONTEXT_ENV as ROBOTWIN_RUNTIME_CONTEXT_ENV,
    SMOKE_COMMAND_SHA256 as ROBOTWIN_SMOKE_COMMAND_SHA256,
    RobotwinAuthorization as _RuntimeAuthorization,
    encode_runtime_authorization,
    is_robotwin_request,
    load_runtime_authorization,
    require_runtime_lock_complete,
    validate_invocation,
)
from npa.orchestration.skypilot.cleanup import cluster_name_patterns_for_run
from npa.workflows.byof.live import (
    resolve_byof_kubernetes_target,
    resolve_byof_profile_path,
)
from npa.workflows.byof.openpi import is_openpi_request, require_openpi_terms
from npa.workflows.byof.postprocess import (
    PostprocessContext,
    has_registered_postprocess,
    run_registered_postprocess,
)
from npa.workflows.byof.source_auth import (
    RepositoryAuthenticationError,
    RepositorySecretFiles,
    private_repository_secrets,
    validate_repository_url,
)
from npa.workflows.byof.worker import (
    WorkerSmoke,
    in_workflow_worker,
    run_prebuilt_smoke,
)

__all__ = [
    "ROBOTWIN_BUILD_COMMAND_SHA256",
    "ROBOTWIN_INVOCATION",
    "ROBOTWIN_CUSTOMER_ENTITLEMENT_ENV",
    "ROBOTWIN_CUSTOMER_TERMS",
    "ROBOTWIN_RUNTIME_CONTEXT_ENV",
    "ROBOTWIN_SMOKE_COMMAND_SHA256",
    "main",
]

SCRIPT_DIR = Path(__file__).resolve().parent
ISAAC_RUNNER = SCRIPT_DIR / "run_isaac_lab_rl.py"
DATAGEN_RUNNER = SCRIPT_DIR / "run_byof_datagen.py"
CONTAINER_VERIFY_RUNNER = SCRIPT_DIR / "run_byof_container_verify.py"
ROBOTWIN_IMAGE_SCANNER = SCRIPT_DIR / "scan_image_robotwin_payload.py"
BYOF_REPO_MOUNT = "/opt/byof"
LIBERO_SOLUTION_NAME = "libero"
LIBERO_PROFILE_NAME = "byof-solution-smoke-libero-b200-gpu"
LIBERO_REPOSITORY = "https://github.com/Lifelong-Robot-Learning/LIBERO"
LIBERO_REPOSITORY_REF = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
LIBERO_BASE_IMAGE = "tool://libero"
LIBERO_SOURCE_PRUNE_PATH = ""
LIBERO_BUILD_COMMAND_SHA256 = hashlib.sha256(b"").hexdigest()
LIBERO_SMOKE_COMMAND_SHA256 = (
    "a38bfcb66f40fd7d192b2793873aca4ba959a0136f422d3cfe46c9e5c3e9dc43"
)
LIBERO_CAPABILITY = "libero_spatial_bc_rnn_train_reload_heldout"
LIBERO_SMOKE_ARTIFACT = "libero-smoke.json"
LIBERO_TASK = (
    "libero_spatial/"
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"
)

# SkyPilot 0.12.2 checks these package capabilities synchronously while starting a
# Kubernetes worker.  Ubuntu's ``fuse3`` package provides the logical ``fuse``
# capability and both fusermount command names; installing the conflicting
# ``fuse`` and ``fuse3`` packages together is not valid on Ubuntu 22.04.
SKYPILOT_BOOTSTRAP_PACKAGE_CAPABILITIES = (
    "rsync",
    "curl",
    "wget",
    "netcat",
    "gcc",
    "patch",
    "pciutils",
    "fuse",
    "fuse3",
    "openssh-server",
)

# These are ordinary execution prerequisites, not authorization or routing
# inputs. Keep the allowlist explicit so authorized RoboTwin handoff does not
# inherit ambient credentials, controller knobs, or output destinations.
_ROBOTWIN_EXECUTION_BASELINE_ENV_NAMES = (
    "HOME",
    "LANG",
    "LC_ALL",
    "LD_LIBRARY_PATH",
    "LOGNAME",
    "PATH",
    "PYTHONPATH",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TMPDIR",
    "USER",
    "VIRTUAL_ENV",
)

DEFAULT_REPO_URL = "https://github.com/LightwheelAI/leisaac.git"
DEFAULT_REPO_REF = "main"
DEFAULT_UBUNTU_BASE_IMAGE = "ubuntu:22.04"
BASE_PROFILES = frozenset({"ubuntu", "isaac-lab", "prebuilt"})
PLACEHOLDER_VALUES = frozenset(
    {
        "",
        "<base-image>",
        "<repo-url>",
        "<repo-ref>",
        "<workload>",
        "<resource-profile.yaml>",
        "<task>",
        "<base-profile>",
    }
)
WAN_POSTPROCESS_CONTRACTS = {
    "wan2.2_ti2v_5b_text_to_video": (
        "wan2.2",
        "wan2_2_ti2v_5b_text_to_video.json",
    ),
    "wan2.2_ti2v_5b_image_to_video": (
        "wan2.2",
        "wan2_2_ti2v_5b_image_to_video.json",
    ),
    "wan2.2_ti2v_5b_text_to_video_multigpu_fsdp_ulysses": (
        "wan2.2-multigpu",
        "wan2_2_ti2v_5b_multigpu.json",
    ),
}


def _redact_text(value: str, redactions: tuple[str, ...]) -> str:
    result = value
    for secret in sorted((item for item in redactions if item), key=len, reverse=True):
        result = result.replace(secret, "<redacted>")
    return result


def _redact_payload(value: Any, redactions: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        return _redact_text(value, redactions)
    if isinstance(value, list):
        return [_redact_payload(item, redactions) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_payload(item, redactions) for item in value)
    if isinstance(value, dict):
        return {
            _redact_text(str(key), redactions): _redact_payload(item, redactions)
            for key, item in value.items()
        }
    return value


def _load_runtime_authorization(
    args: argparse.Namespace,
) -> _RuntimeAuthorization | None:
    reference = args.runtime_context_env.strip()
    if not _is_robotwin_request(args):
        if reference:
            raise ValueError("runtime authorization is not supported for this solution")
        return None
    if reference != ROBOTWIN_RUNTIME_CONTEXT_ENV:
        raise ValueError(
            f"RoboTwin requires --runtime-context-env {ROBOTWIN_RUNTIME_CONTEXT_ENV}"
        )
    return require_runtime_lock_complete(load_runtime_authorization())


def _apply_runtime_authorization(
    args: argparse.Namespace, authorization: _RuntimeAuthorization
) -> None:
    args.project = authorization.project
    args.base_image = authorization.bootstrap_image
    args.image = authorization.bootstrap_image
    args.registry = _registry_path(authorization.bootstrap_image)
    args.output_root = authorization.output_root
    args.run_id = authorization.run_id
    args.config_path = authorization.skypilot_config_path


def _validate_robotwin_invocation(args: argparse.Namespace) -> None:
    validate_invocation(args)


def _is_robotwin_request(args: argparse.Namespace) -> bool:
    return is_robotwin_request(
        solution_name=args.solution_name,
        repo_url=args.repo_url,
        base_image=args.base_image,
        image=args.image,
        smoke_command=args.smoke_command,
        capability_name=args.capability_name,
        yaml_path=args.yaml,
    )


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _normalize_optional(value: str) -> str:
    cleaned = str(value or "").strip()
    if cleaned in PLACEHOLDER_VALUES:
        return ""
    return cleaned


class LiberoCustomerAcceptanceRequired(ValueError):
    """Carry the structured, public customer acknowledgement prompt."""

    def __init__(self, notification: dict[str, Any]) -> None:
        self.notification = notification
        super().__init__("LIBERO customer authorization is required")


def _libero_qualified_candidate(value: str, qualification: dict[str, Any]) -> str:
    candidate = str(value or "").strip().removeprefix("docker:")
    if candidate != qualification.get("candidate_image"):
        raise ValueError(
            "LIBERO candidate must exactly match checked-in qualified image lineage"
        )
    return candidate


def _libero_customer_authorization(
    args: argparse.Namespace, image_manifest: dict[str, Any]
) -> tuple[Path, bytes, str, str, str, bytes, str]:
    caller_path_value = str(
        args.libero_authenticated_caller_identity_file or ""
    ).strip()
    if not caller_path_value:
        raise LiberoCustomerAcceptanceRequired(
            libero_customer_acceptance_notification(
                image_manifest, reason="authenticated_customer_identity_required"
            )
        )
    caller_bytes = _owner_private_file_bytes(
        Path(caller_path_value).expanduser(), label="authenticated caller assertion"
    )
    try:
        caller, caller_sha256 = validate_libero_authenticated_caller_assertion(
            caller_bytes, run_id=args.run_id
        )
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    customer_identity_sha256 = str(caller["customer_identity_sha256"])
    customer_signer_public_key_sha256 = str(caller["customer_signer_public_key_sha256"])
    path_value = str(args.libero_customer_runtime_authorization_file or "").strip()
    if not path_value:
        raise LiberoCustomerAcceptanceRequired(
            libero_customer_acceptance_notification(image_manifest)
        )
    path = Path(path_value).expanduser()
    authorization_bytes = _owner_private_file_bytes(
        path, label="customer authorization"
    )
    try:
        authorization, observed = validate_libero_customer_runtime_authorization(
            authorization_bytes,
            image_manifest=image_manifest,
            run_id=args.run_id,
            customer_identity_sha256=customer_identity_sha256,
            customer_signer_public_key_sha256=(customer_signer_public_key_sha256),
        )
    except LiberoCustomerAuthorizationDenied as exc:
        raise LiberoCustomerAcceptanceRequired(
            libero_customer_acceptance_notification(
                image_manifest, reason="authorization_denied"
            )
        ) from exc
    except RuntimeError as exc:
        message = str(exc)
        if any(
            infrastructure_failure in message
            for infrastructure_failure in (
                "trust-root file is unavailable",
                "trust-root file is mutable or invalid",
                "trust root differs",
            )
        ):
            raise ValueError(message) from exc
        reason = (
            "authorization_expired_or_replayable"
            if "expired or replayable" in message
            else "authorization_invalid"
        )
        raise LiberoCustomerAcceptanceRequired(
            libero_customer_acceptance_notification(image_manifest, reason=reason)
        ) from exc
    return (
        path,
        authorization_bytes,
        observed,
        customer_identity_sha256,
        customer_signer_public_key_sha256,
        caller_bytes,
        caller_sha256,
    )


def _owner_private_file_bytes(path: Path, *, label: str) -> bytes:
    """Read one stable owner-private control-plane artifact."""

    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except FileNotFoundError as exc:
        raise ValueError(f"LIBERO {label} file is unavailable") from exc
    except OSError as exc:
        raise ValueError(f"LIBERO {label} file is unavailable or invalid") from exc
    try:
        before = os.fstat(descriptor)
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            authorization_bytes = stream.read(1024 * 1024 + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.getuid()
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) & 0o077
        or len(authorization_bytes) > 1024 * 1024
        or before.st_size != len(authorization_bytes)
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise ValueError(f"LIBERO {label} must be a stable owner-private regular file")
    return authorization_bytes


def _validate_libero_identity(args: argparse.Namespace) -> None:
    """Reject ambiguous LIBERO identity before registry or build operations."""

    solution = args.solution_name.strip().lower()
    profile = Path(args.yaml).name.removesuffix(".yaml")
    repository = args.repo_url.strip().removesuffix(".git").rstrip("/")
    is_libero = (
        solution == LIBERO_SOLUTION_NAME
        or profile == LIBERO_PROFILE_NAME
        or repository == LIBERO_REPOSITORY
    )
    if not is_libero:
        return
    if args.solution_name != LIBERO_SOLUTION_NAME:
        raise ValueError("LIBERO requires the exact --solution-name libero identity")
    cluster_name_patterns_for_run(args.run_id)
    if args.workload != "solution-smoke":
        raise ValueError("LIBERO requires the solution-smoke workload")
    if profile != LIBERO_PROFILE_NAME:
        raise ValueError("LIBERO requires its exact B200 solution-smoke profile")
    selected_profile = resolve_byof_profile_path(args.yaml).resolve()
    packaged_profile = resolve_byof_profile_path(LIBERO_PROFILE_NAME).resolve()
    if selected_profile != packaged_profile:
        raise ValueError("LIBERO requires the packaged B200 solution-smoke profile")
    exact_values = {
        "repository": (args.repo_url, f"{LIBERO_REPOSITORY}.git"),
        "source revision": (args.repo_ref, LIBERO_REPOSITORY_REF),
        "repository authentication": (args.repo_auth, "none"),
        "base profile": (args.base_profile, "prebuilt"),
        "base image": (args.base_image, LIBERO_BASE_IMAGE),
        "source prune path": (args.source_prune_path, LIBERO_SOURCE_PRUNE_PATH),
        "capability": (args.capability_name, LIBERO_CAPABILITY),
        "smoke artifact": (args.smoke_artifact_name, LIBERO_SMOKE_ARTIFACT),
        "task": (args.task, LIBERO_TASK),
        "iteration count": (args.iterations, 1),
        "environment count": (args.num_envs, 1),
        "demonstration count": (args.num_demos, 1),
    }
    for label, (observed, expected) in exact_values.items():
        if observed != expected:
            raise ValueError(f"LIBERO requires its exact {label} contract")
    command_hashes = {
        "build command": (
            hashlib.sha256(args.build_command.encode()).hexdigest(),
            LIBERO_BUILD_COMMAND_SHA256,
        ),
        "smoke command": (
            hashlib.sha256(args.smoke_command.encode()).hexdigest(),
            LIBERO_SMOKE_COMMAND_SHA256,
        ),
    }
    for label, (observed, expected) in command_hashes.items():
        if observed != expected:
            raise ValueError(f"LIBERO requires its exact {label} contract")
    if args.skip_run:
        raise ValueError(
            "LIBERO cannot use --skip-run because live qualification is mandatory"
        )
    try:
        image_manifest = libero_image_manifest()
        if not str(args.libero_customer_runtime_authorization_file or "").strip():
            raise LiberoCustomerAcceptanceRequired(
                libero_customer_acceptance_notification(image_manifest)
            )
        qualification = validate_libero_qualified_image_manifest(image_manifest)
        libero_publication_lineage_values(
            qualification,
            SCRIPT_DIR.parents[1],
            development_sha=qualification["development_sha"],
        )
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    args._libero_image_manifest = image_manifest
    args._libero_qualification = qualification
    _libero_qualified_candidate(args.libero_qualified_candidate_image, qualification)
    (
        _,
        authorization_bytes,
        authorization_sha256,
        customer_identity_sha256,
        customer_signer_public_key_sha256,
        caller_bytes,
        caller_sha256,
    ) = _libero_customer_authorization(args, image_manifest)
    args._libero_customer_authorization_bytes = authorization_bytes
    args._libero_customer_authorization_sha256 = authorization_sha256
    args._libero_customer_identity_sha256 = customer_identity_sha256
    args._libero_customer_signer_public_key_sha256 = customer_signer_public_key_sha256
    args._libero_authenticated_caller_bytes = caller_bytes
    args._libero_authenticated_caller_sha256 = caller_sha256


def _image_repository_name(image_ref: str) -> str:
    ref = image_ref.removeprefix("docker:").split("@", 1)[0]
    return ref.rsplit("/", 1)[-1].split(":", 1)[0]


def _required_postprocess_key(
    args: argparse.Namespace, *, base_image: str, base_profile: str
) -> str | None:
    """Resolve mandatory postprocessing from the immutable workload contract.

    A caller-provided solution label may select an ordinary registered
    postprocessor, but it may not turn postprocessing off for a Wan image.  The
    accepted Wan path is bound independently to its prebuilt digest, capability,
    and artifact contract so an empty or misspelled label fails before launch.
    """

    if args.workload != "solution-smoke":
        return None
    requested = args.solution_name.strip().lower()
    capability = args.capability_name.strip()
    artifact = args.smoke_artifact_name.strip()
    base_is_wan = _image_repository_name(base_image) == "npa-wan2-2"
    wan_signaled = (
        base_is_wan
        or requested in {"wan2.2", "wan2.2-multigpu"}
        or capability.startswith("wan2.2_")
        or artifact.startswith("wan2_2_")
    )
    if not wan_signaled:
        return requested if has_registered_postprocess(requested) else None

    accepted_digest = str(wan_accepted_image_manifest()["oci_digest"])
    acceptance_candidate = str(
        getattr(args, "wan_acceptance_candidate_image", "") or ""
    ).strip()
    if acceptance_candidate:
        if (
            acceptance_candidate != base_image
            or not getattr(args, "skip_build", False)
            or re.fullmatch(
                r"ghcr\.io/nebius/nebius-physical-ai/"
                r"npa-wan2-2@sha256:[0-9a-f]{64}",
                acceptance_candidate,
            )
            is None
        ):
            raise ValueError(
                "--wan-acceptance-candidate-image requires --skip-build and must "
                "exactly match the official digest-pinned Wan --base-image"
            )
    if (
        not base_is_wan
        or base_profile != "prebuilt"
        or (
            not base_image.endswith(f"@{accepted_digest}")
            and base_image != acceptance_candidate
        )
    ):
        raise ValueError(
            "Wan solution-smoke requires the exact GPU-accepted prebuilt image "
            f"digest {accepted_digest}, or the explicitly digest-pinned candidate "
            "inside the gated live acceptance path"
        )
    contract = WAN_POSTPROCESS_CONTRACTS.get(capability)
    if contract is None:
        raise ValueError(
            f"Wan solution-smoke capability {capability!r} has no mandatory postprocess contract"
        )
    expected_key, expected_artifact = contract
    if artifact != expected_artifact:
        raise ValueError(
            f"Wan capability {capability!r} requires smoke artifact {expected_artifact!r}"
        )
    if requested != expected_key:
        raise ValueError(
            f"Wan capability {capability!r} requires solution name {expected_key!r}; "
            "the label cannot disable verified RRD postprocessing"
        )
    return expected_key


def _run(
    cmd: list[str],
    *,
    stdin: str | None = None,
    capture: bool = False,
    env: dict[str, str] | None = None,
    redactions: tuple[str, ...] = (),
    inherit_env: bool = True,
) -> subprocess.CompletedProcess[str]:
    runtime_env = (
        dict(os.environ)
        if inherit_env
        else {
            name: os.environ[name]
            for name in (
                "HOME",
                "LANG",
                "LC_ALL",
                "LD_LIBRARY_PATH",
                "LOGNAME",
                "PATH",
                "PYTHONPATH",
                "SSL_CERT_DIR",
                "SSL_CERT_FILE",
                "TMPDIR",
                "USER",
                "VIRTUAL_ENV",
            )
            if os.environ.get(name)
        }
    )
    # Avoid stale operator tokens overriding profile-based auth on shared VMs.
    runtime_env.pop("NEBIUS_IAM_TOKEN", None)
    runtime_env.pop("NEBIUS_IAM_TOKEN_FILE", None)
    for name in ROBOTWIN_CONTEXT_ENV_NAMES:
        runtime_env.pop(name, None)
    if env is not None:
        runtime_env.update(env)
    for name in ROBOTWIN_CONTEXT_ENV_NAMES:
        runtime_env.pop(name, None)
    kwargs: dict[str, Any] = {"text": True, "check": False}
    if stdin is not None:
        kwargs["input"] = stdin
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    kwargs["env"] = runtime_env
    print("+", _redact_text(" ".join(cmd), redactions))
    proc = subprocess.run(cmd, **kwargs)
    if proc.returncode != 0:
        if capture:
            raise RuntimeError(
                f"command failed ({proc.returncode}): {_redact_text(' '.join(cmd), redactions)}\n"
                f"stdout:\n{_redact_text(proc.stdout or '', redactions)}\n"
                f"stderr:\n{_redact_text(proc.stderr or '', redactions)}"
            )
        raise RuntimeError(
            f"command failed ({proc.returncode}): {_redact_text(' '.join(cmd), redactions)}"
        )
    return proc


def _registry_server(image_ref: str) -> str:
    ref = image_ref.removeprefix("docker:")
    return ref.split("/", 1)[0]


def _repository_without_tag(image_ref: str) -> str:
    ref = image_ref.removeprefix("docker:").split("@", 1)[0]
    slash = ref.rfind("/")
    colon = ref.rfind(":")
    return ref[:colon] if colon > slash else ref


def _source_prune_path(value: str) -> str:
    """Validate one repo-relative path removed in the source checkout layer."""

    path = str(value or "").strip()
    if not path:
        return ""
    parts = path.split("/")
    if (
        path.startswith("/")
        or re.fullmatch(r"[A-Za-z0-9._/-]+", path) is None
        or ".." in path
        or any(part in {"", ".", "..", ".git"} for part in parts)
        or path == "npa_source_metadata.json"
    ):
        raise argparse.ArgumentTypeError(
            "--source-prune-path must be a safe relative repository path"
        )
    return path


def _resolve_pushed_image_digest(
    image_ref: str,
    *,
    env: dict[str, str] | None = None,
    redactions: tuple[str, ...] = (),
) -> str:
    """Resolve a just-pushed tag and return the immutable pull reference."""

    run_kwargs: dict[str, Any] = {"env": env, "capture": True}
    if redactions:
        run_kwargs["redactions"] = redactions
    inspected = _run(
        ["docker", "buildx", "imagetools", "inspect", image_ref], **run_kwargs
    )
    combined = "\n".join((inspected.stdout or "", inspected.stderr or ""))
    match = re.search(r"(?im)^\s*Digest:\s*(sha256:[0-9a-f]{64})\s*$", combined)
    if match is None:
        raise RuntimeError(
            "pushed BYOF image did not resolve to an immutable sha256 digest"
        )
    return f"{_repository_without_tag(image_ref)}@{match.group(1)}"


def _bare_s3_bucket(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    if text.startswith("s3://"):
        text = text[len("s3://") :]
    return text.split("/", 1)[0].strip()


def _live_runner_env(
    project: str, *, redactions: tuple[str, ...] = (), libero: bool = False
) -> dict[str, str]:
    env: dict[str, str] = {}
    target = resolve_byof_kubernetes_target(project or None)
    if target.kubeconfig:
        env["KUBECONFIG"] = target.kubeconfig
    if target.context:
        env["KUBECONTEXT"] = target.context
        env["NPA_BYOF_K8S_CONTEXT"] = target.context
    if target.namespace:
        env["NPA_BYOF_K8S_NAMESPACE"] = target.namespace
    if libero:
        exact_names = (
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            LIBERO_AUTHENTICATED_CALLER_PUBLIC_KEY_FILE_ENV,
            "NPA_LIBERO_CUSTOMER_SIGNER_REGISTRATION_FILE",
            LIBERO_OUTPUT_STORAGE_AUTHORIZATION_PUBLIC_KEY_FILE_ENV,
        )
        exact_values = {name: str(os.environ.get(name) or "") for name in exact_names}
        if not all(exact_values.values()):
            raise ValueError(
                "LIBERO requires one complete control-plane-authorized storage credential triplet"
            )
        endpoint_names = (
            "AWS_ENDPOINT_URL_S3",
            "AWS_ENDPOINT_URL",
            "NEBIUS_S3_ENDPOINT",
            "NPA_STORAGE_ENDPOINT",
            "S3_ENDPOINT_URL",
        )
        endpoints = {
            str(os.environ.get(name) or "").strip().rstrip("/")
            for name in endpoint_names
            if os.environ.get(name)
        }
        if len(endpoints) != 1:
            raise ValueError(
                "LIBERO requires one exact control-plane-authorized storage endpoint"
            )
        env.update(exact_values)
        env.update(
            {
                name: str(os.environ[name])
                for name in endpoint_names
                if os.environ.get(name)
            }
        )
    else:
        try:
            env.update(
                storage_env_for_project(
                    project or None,
                    allow_host_creds=True,
                    endpoint_url=os.environ.get("NPA_BYOF_S3_ENDPOINT", ""),
                )
            )
        except Exception as exc:
            warning = _redact_text(
                f"WARN: skipped BYOF storage env resolution: {exc}", redactions
            )
            print(warning, file=sys.stderr)
    # Project configs often store checkpoint_bucket as s3://bucket/prefix. BYOF
    # SkyPilot templates expect a bare bucket name in NPA_S3_BUCKET.
    for key in ("NPA_S3_BUCKET", "S3_BUCKET"):
        bare = _bare_s3_bucket(os.environ.get(key, "") or env.get(key, ""))
        if bare:
            env["NPA_S3_BUCKET"] = bare
            break
    if "NPA_S3_BUCKET" not in env and not libero:
        try:
            from npa.clients.config import _load_yaml, _resolve_project_section

            yml = _load_yaml()
            section = _resolve_project_section(yml, project or None) if project else {}
            storage = section.get("storage") if isinstance(section, dict) else {}
            if not isinstance(storage, dict):
                root_storage = yml.get("storage") if isinstance(yml, dict) else None
                storage = root_storage if isinstance(root_storage, dict) else {}
            bare = _bare_s3_bucket(
                str(
                    storage.get("checkpoint_bucket")
                    or storage.get("bucket")
                    or storage.get("s3_bucket")
                    or ""
                )
            )
            if bare:
                env["NPA_S3_BUCKET"] = bare
        except Exception as exc:
            warning = _redact_text(
                f"WARN: skipped BYOF bucket resolution: {exc}", redactions
            )
            print(warning, file=sys.stderr)
    return env


def _registry_path(image_ref: str) -> str:
    ref = image_ref.removeprefix("docker:")
    without_digest = ref.split("@", 1)[0]
    last_slash = without_digest.rfind("/")
    if last_slash <= 0:
        return ""
    return without_digest[:last_slash]


def _private_image_redactions(image_ref: str) -> tuple[str, ...]:
    """Return every private coordinate derivable from an authorized image."""

    return tuple(
        dict.fromkeys(
            value
            for value in (
                image_ref,
                _repository_without_tag(image_ref),
                _registry_path(image_ref),
                _registry_server(image_ref),
            )
            if value
        )
    )


def _skypilot_bootstrap_guard_script() -> str:
    """Return the image-resident SkyPilot package and deadline guard."""

    return r"""#!/bin/sh
set -u

real_apt_get=/usr/bin/apt-get
real_timeout=/usr/bin/timeout
real_ssh_keygen=/usr/bin/ssh-keygen
guard_runtime_dir=/run/npa-skypilot-bootstrap
guard_owner_uid=0
guard_state=$guard_runtime_dir/apt.state
guard_failure=/tmp/npa-skypilot-bootstrap-contract.failed
sky_failure=/tmp/apt-ssh-setup.failed
trusted_bootstrap_marker=/tmp/apt_ssh_setup_complete
trusted_bootstrap_marker_value=skypilot-apt-v1

atomic_replace_text() {
    target=$1
    content=$2
    target_directory=${target%/*}
    target_name=${target##*/}
    temporary="$(umask 077; mktemp "$target_directory/.${target_name}.XXXXXX")" \
        || return 1
    temporary_identity="$(stat -c '%u:%a' -- "$temporary" 2>/dev/null)" \
        || { rm -f -- "$temporary"; return 1; }
    if [ -L "$temporary" ] || [ ! -f "$temporary" ] \
        || [ "$temporary_identity" != "$guard_owner_uid:600" ]; then
        rm -f -- "$temporary"
        return 1
    fi
    printf '%s\n' "$content" > "$temporary" \
        || { rm -f -- "$temporary"; return 1; }
    mv -fT -- "$temporary" "$target" \
        || { rm -f -- "$temporary"; return 1; }
}

private_state_failure() {
    contract_failure "unsafe-private-state:$1" 87
    return $?
}

trusted_bootstrap_mode() {
    if [ "$(id -u)" -ne "$guard_owner_uid" ]; then
        contract_failure apt-requires-root 87
        return $?
    fi
    if [ -L "$trusted_bootstrap_marker" ] || [ ! -f "$trusted_bootstrap_marker" ]; then
        contract_failure trusted-marker-missing 87
        return $?
    fi
    marker_identity="$(stat -c '%u:%a' -- "$trusted_bootstrap_marker" 2>/dev/null)" \
        || { contract_failure trusted-marker-identity 87; return $?; }
    if [ "$marker_identity" != "$guard_owner_uid:600" ]; then
        contract_failure trusted-marker-identity-or-mode 87
        return $?
    fi
    marker_value="$(cat -- "$trusted_bootstrap_marker" 2>/dev/null)" \
        || { contract_failure trusted-marker-read 87; return $?; }
    if [ "$marker_value" != "$trusted_bootstrap_marker_value" ]; then
        contract_failure trusted-marker-value 87
        return $?
    fi
}

prepare_private_state_directory() {
    if [ -L "$guard_runtime_dir" ]; then
        private_state_failure symlink
        return $?
    fi
    if [ ! -e "$guard_runtime_dir" ]; then
        (umask 077; mkdir -m 0700 -- "$guard_runtime_dir") \
            || { private_state_failure create; return $?; }
    fi
    if [ -L "$guard_runtime_dir" ] || [ ! -d "$guard_runtime_dir" ]; then
        private_state_failure wrong-type
        return $?
    fi
    private_state_identity="$(stat -c '%u:%a' -- "$guard_runtime_dir" 2>/dev/null)" \
        || { private_state_failure identity; return $?; }
    if [ "$private_state_identity" != "$guard_owner_uid:700" ]; then
        private_state_failure identity-or-mode
        return $?
    fi
}

read_guard_state() {
    state=""
    if [ -L "$guard_state" ]; then
        private_state_failure state-symlink
        return $?
    fi
    [ -e "$guard_state" ] || return 0
    if [ ! -f "$guard_state" ]; then
        private_state_failure state-wrong-type
        return $?
    fi
    state_identity="$(stat -c '%u:%a' -- "$guard_state" 2>/dev/null)" \
        || { private_state_failure state-identity; return $?; }
    if [ "$state_identity" != "$guard_owner_uid:600" ]; then
        private_state_failure state-identity-or-mode
        return $?
    fi
    state_read_failed=false
    state_extra=false
    state_extra_value=""
    {
        IFS= read -r state || state_read_failed=true
        IFS= read -r state_extra_value && state_extra=true
        [ -z "$state_extra_value" ] || state_extra=true
    } < "$guard_state"
    if [ "$state_read_failed" = true ] || [ "$state_extra" = true ]; then
        private_state_failure state-format
        return $?
    fi
    case "$state" in
        verified-update|complete) ;;
        *) private_state_failure state-value; return $? ;;
    esac
}

write_guard_state() {
    atomic_replace_text "$guard_state" "$1"
}

package_installed() {
    package_status="$(dpkg-query -W -f='${Status}' "$1" 2>/dev/null)" || return 1
    [ "$package_status" = "install ok installed" ]
}

contract_failure() {
    detail=$1
    status=${2:-86}
    sentinel="NPA_SKYPILOT_BOOTSTRAP_FAILED status=${status} detail=${detail}"
    printf '%s\n' "$sentinel" >&2
    sentinel_write_failed=false
    atomic_replace_text "$guard_failure" "$sentinel" \
        || sentinel_write_failed=true
    atomic_replace_text "$sky_failure" "$sentinel" \
        || sentinel_write_failed=true
    if [ "$sentinel_write_failed" = true ]; then
        printf '%s\n' 'NPA_SKYPILOT_BOOTSTRAP_SENTINEL_WRITE_FAILED' >&2
    fi
    return "$status"
}

verify_contract() {
    missing=""
    for package in rsync curl wget gcc patch pciutils fuse3 openssh-server coreutils; do
        package_installed "$package" || missing="$missing $package"
    done
    netcat_installed=false
    for package in netcat-openbsd netcat-traditional netcat; do
        if package_installed "$package"; then
            netcat_installed=true
            break
        fi
    done
    [ "$netcat_installed" = true ] || missing="$missing netcat"
    fuse_provides="$(dpkg-query -W -f='${Provides}' fuse3 2>/dev/null)" || fuse_provides=""
    case " $fuse_provides " in
        *" fuse "*|*" fuse ("*) ;;
        *) missing="$missing fuse" ;;
    esac
    for command_name in sh sudo sshd ssh-keygen rsync service curl wget nc gcc \
        patch lspci fusermount fusermount3 timeout; do
        command -v "$command_name" >/dev/null 2>&1 \
            || missing="$missing command:$command_name"
    done
    [ -x "$real_ssh_keygen" ] || missing="$missing command:real-ssh-keygen"
    if [ -n "$missing" ]; then
        contract_failure "missing:${missing# }" 86
        return $?
    fi
    printf '%s\n' 'NPA_SKYPILOT_BOOTSTRAP_VERIFIED status=0 apt_required=false'
}

bootstrap_apt_get() {
    trusted_bootstrap_mode || exit $?
    verify_contract >/dev/null || exit $?
    prepare_private_state_directory || exit $?
    exec 9>"$guard_runtime_dir/apt.lock" \
        || { contract_failure private-state-lock-open 87; exit $?; }
    flock -x 9 \
        || { contract_failure private-state-lock-acquire 87; exit $?; }
    read_guard_state || exit $?
    operation=${1:-}
    if [ "$operation" = update ]; then
        case "$state" in
        "")
            write_guard_state verified-update \
                || { contract_failure private-state-write 87; exit $?; }
            ;;
        verified-update|complete) ;;
        *) contract_failure "unexpected-state:$state" 87; exit $? ;;
        esac
        printf '%s\n' 'NPA_SKYPILOT_BOOTSTRAP_APT_BYPASSED status=0 operation=update'
        exit 0
    fi
    if [ "$operation" = install ] && { [ "$state" = verified-update ] || [ "$state" = complete ]; }; then
        shift
        requested=""
        while [ "$#" -gt 0 ]; do
            case "$1" in
                -o|--option)
                    shift
                    [ "$#" -gt 0 ] \
                        || { contract_failure malformed-apt-options 87; exit $?; }
                    ;;
                -*) ;;
                *) requested="$requested $1" ;;
            esac
            shift
        done
        if [ "$requested" != " fuse" ]; then
            contract_failure "unexpected-install:${requested# }" 87
            exit $?
        fi
        if [ "$state" = verified-update ]; then
            write_guard_state complete \
                || { contract_failure private-state-write 87; exit $?; }
        fi
        printf '%s\n' \
            'NPA_SKYPILOT_BOOTSTRAP_APT_BYPASSED status=0 operation=install package=fuse provider=fuse3'
        exit 0
    fi
    contract_failure "unexpected-operation:${operation:-missing}:state:${state:-empty}" 87
    exit $?
}

bootstrap_timeout() {
    trusted_bootstrap_mode || exit $?
    "$real_timeout" --signal=TERM --kill-after=5s "$@"
    status=$?
    case "$status" in
        124|137) contract_failure deadline-exceeded "$status" >/dev/null ;;
    esac
    exit "$status"
}

bootstrap_ssh_keygen() {
    if [ "$#" -ne 1 ] || [ "$1" != -A ]; then
        contract_failure unexpected-ssh-keygen-arguments 87
        exit $?
    fi
    if [ "$(id -u)" -ne 0 ]; then
        contract_failure ssh-keygen-requires-root 87
        exit $?
    fi
    umask 077
    exec "$real_ssh_keygen" -A
}

case "$(basename "$0")" in
    apt-get) bootstrap_apt_get "$@" ;;
    timeout) bootstrap_timeout "$@" ;;
    ssh-keygen) bootstrap_ssh_keygen "$@" ;;
    npa-skypilot-bootstrap-guard)
        [ "${1:-}" = verify ] \
            || { printf '%s\n' 'usage: npa-skypilot-bootstrap-guard verify' >&2; exit 64; }
        verify_contract
        ;;
    *) printf '%s\n' 'NPA SkyPilot bootstrap guard invoked under an unknown name' >&2; exit 64 ;;
esac
"""


def _immutable_image_digest(image: str) -> str:
    """Return the immutable digest from an exact image reference, if present."""

    match = re.search(r"@(sha256:[0-9a-f]{64})$", image.strip())
    return match.group(1) if match else ""


def _dockerfile_text() -> str:
    bootstrap_guard = _skypilot_bootstrap_guard_script()
    return (
        "# syntax=docker/dockerfile:1.7\n"
        "ARG BYOF_BASE_IMAGE\n"
        "FROM ${BYOF_BASE_IMAGE}\n"
        "ARG BYOF_BASE_IMAGE\n"
        'ARG BYOF_BASE_IMAGE_DIGEST=""\n'
        'ARG OSS_REPO_URL=""\n'
        'ARG OSS_REPO_REF=""\n'
        "ARG BYOF_SOURCE_VISIBILITY=public\n"
        "ARG BYOF_SOURCE_CACHE_KEY=public\n"
        "ARG BYOF_SOURCE_LABEL_REPO\n"
        "ARG BYOF_SOURCE_LABEL_REF\n"
        'ARG BYOF_SOURCE_PRUNE_PATH=""\n'
        "ARG BYOF_BUILD_COMMAND\n"
        "USER root\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends \\\n"
        "      git ca-certificates python3 python3-pip sudo coreutils rsync \\\n"
        "      curl wget gcc patch pciutils fuse3 \\\n"
        "      openssh-client openssh-server netcat-openbsd \\\n"
        "  && rm -rf /var/lib/apt/lists/*\n"
        "RUN <<'NPA_SKYPILOT_GUARD_INSTALL'\n"
        "set -eu\n"
        "cat > /usr/local/sbin/npa-skypilot-bootstrap-guard <<'NPA_SKYPILOT_GUARD'\n"
        f"{bootstrap_guard}"
        "NPA_SKYPILOT_GUARD\n"
        "chmod 0755 /usr/local/sbin/npa-skypilot-bootstrap-guard\n"
        "ln -sf npa-skypilot-bootstrap-guard /usr/local/sbin/apt-get\n"
        "ln -sf /usr/local/sbin/npa-skypilot-bootstrap-guard /usr/local/bin/timeout\n"
        "NPA_SKYPILOT_GUARD_INSTALL\n"
        "RUN /usr/local/sbin/npa-skypilot-bootstrap-guard verify\n"
        "RUN id -u ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash -u 1000 ubuntu\n"
        "RUN install -d -m 0755 /run/sshd \\\n"
        "  && (grep -q 'ssh-keygen -A' /etc/init.d/ssh \\\n"
        "    || sed -i '/^  start)$/a\\    ssh-keygen -A' /etc/init.d/ssh) \\\n"
        "  && grep -q 'ssh-keygen -A' /etc/init.d/ssh \\\n"
        "  && rm -f /etc/ssh/ssh_host_* \\\n"
        "  && printf '%s\\n' 'PasswordAuthentication no' 'PermitRootLogin no' \\\n"
        "    > /etc/ssh/sshd_config.d/99-npa-container.conf \\\n"
        "  && echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/ubuntu \\\n"
        "  && chmod 440 /etc/sudoers.d/ubuntu\n"
        "RUN mkdir -p /workspace && chown ubuntu:ubuntu /workspace\n"
        "WORKDIR /workspace\n"
        "RUN --mount=type=secret,id=npa_byof_repo_token \\\n"
        "    --mount=type=secret,id=npa_byof_repo_url \\\n"
        "    --mount=type=secret,id=npa_byof_repo_ref \\\n"
        "    --mount=type=secret,id=npa_byof_source_prune_path \\\n"
        "    set -eu; \\\n"
        '    test -n "${BYOF_SOURCE_CACHE_KEY}"; \\\n'
        '    repo_url="${OSS_REPO_URL}"; repo_ref="${OSS_REPO_REF}"; \\\n'
        '    if [ -s /run/secrets/npa_byof_repo_url ]; then repo_url="$(cat /run/secrets/npa_byof_repo_url)"; fi; \\\n'
        '    if [ -s /run/secrets/npa_byof_repo_ref ]; then repo_ref="$(cat /run/secrets/npa_byof_repo_ref)"; fi; \\\n'
        '    source_prune_path="${BYOF_SOURCE_PRUNE_PATH}"; \\\n'
        '    if [ -s /run/secrets/npa_byof_source_prune_path ]; then source_prune_path="$(cat /run/secrets/npa_byof_source_prune_path)"; fi; \\\n'
        "    export GIT_TERMINAL_PROMPT=0; \\\n"
        '    git_with_auth() { git "$@"; }; \\\n'
        "    if [ -s /run/secrets/npa_byof_repo_token ]; then \\\n"
        "      printf '%s\\n' '#!/bin/sh' \\\n"
        "        '[ \"$1\" = get ] || exit 0' \\\n"
        "        'printf \"username=x-access-token\\\\npassword=\"' \\\n"
        "        'cat /run/secrets/npa_byof_repo_token' \\\n"
        "        'printf \"\\\\n\\\\n\"' \\\n"
        "        > /tmp/npa-byof-git-credential; \\\n"
        "      chmod 700 /tmp/npa-byof-git-credential; \\\n"
        '      git_with_auth() { git -c credential.useHttpPath=true -c credential.helper=/tmp/npa-byof-git-credential "$@"; }; \\\n'
        "    fi; \\\n"
        f'    git_with_auth clone --depth 1 --branch "$repo_ref" "$repo_url" {BYOF_REPO_MOUNT} \\\n'
        f"    || (rm -rf {BYOF_REPO_MOUNT}; \\\n"
        f'      git_with_auth clone "$repo_url" {BYOF_REPO_MOUNT}; \\\n'
        f'      cd {BYOF_REPO_MOUNT}; git checkout "$repo_ref"); \\\n'
        f'    observed_commit="$(git -C {BYOF_REPO_MOUNT} rev-parse HEAD)"; \\\n'
        "    git_objects_removed=false; \\\n"
        "    source_pruned=false; \\\n"
        '    if [ -n "${source_prune_path}" ]; then \\\n'
        '      case "${source_prune_path}" in /*|*..*|.git|.git/*|*/.git|*/.git/*) echo "invalid source prune path" >&2; exit 2;; esac; \\\n'
        '      old_ifs="$IFS"; IFS="/"; set -f; set -- $source_prune_path; set +f; IFS="$old_ifs"; \\\n'
        f'      prune_target="{BYOF_REPO_MOUNT}"; \\\n'
        '      for component do prune_target="$prune_target/$component"; test ! -L "$prune_target" || { echo "symlinked source prune path" >&2; exit 2; }; done; \\\n'
        '      test -e "$prune_target"; \\\n'
        f'      rm -rf -- "$prune_target" {BYOF_REPO_MOUNT}/.git; \\\n'
        '      test ! -e "$prune_target"; \\\n'
        f"      test ! -e {BYOF_REPO_MOUNT}/.git; \\\n"
        "      git_objects_removed=true; \\\n"
        "      source_pruned=true; \\\n"
        "    fi; \\\n"
        f'    if [ "${{BYOF_SOURCE_VISIBILITY}}" = private ]; then \\\n'
        "      repo_sha=\"$(printf '%s' \"$repo_url\" | sha256sum | cut -d' ' -f1)\"; \\\n"
        "      ref_sha=\"$(printf '%s' \"$repo_ref\" | sha256sum | cut -d' ' -f1)\"; \\\n"
        "      commit_sha=\"$(printf '%s' \"$observed_commit\" | sha256sum | cut -d' ' -f1)\"; \\\n"
        "      prune_sha=\"$(printf '%s' \"$source_prune_path\" | sha256sum | cut -d' ' -f1)\"; \\\n"
        '      if [ -n "$source_prune_path" ]; then prune_label="<private-source-prune-path>"; else prune_label=""; fi; \\\n'
        f"      rm -rf {BYOF_REPO_MOUNT}/.git; \\\n"
        f"      test ! -e {BYOF_REPO_MOUNT}/.git; git_objects_removed=true; \\\n"
        f'      REPO_SHA="$repo_sha" REF_SHA="$ref_sha" COMMIT_SHA="$commit_sha" PRUNE_LABEL="$prune_label" PRUNE_SHA="$prune_sha" SOURCE_PRUNED="$source_pruned" GIT_OBJECTS_REMOVED="$git_objects_removed" python3 -c \'import json, os; print(json.dumps({{"source":"private-byof","repository_sha256":os.environ["REPO_SHA"],"ref_sha256":os.environ["REF_SHA"],"commit":"<private-commit>","commit_sha256":os.environ["COMMIT_SHA"],"source_prune_path":os.environ["PRUNE_LABEL"],"source_prune_path_sha256":os.environ["PRUNE_SHA"],"source_pruned":os.environ["SOURCE_PRUNED"]=="true","git_objects_removed":os.environ["GIT_OBJECTS_REMOVED"]=="true"}}, sort_keys=True))\' > {BYOF_REPO_MOUNT}/npa_source_metadata.json; \\\n'
        "    else \\\n"
        f'      REPO_URL="$repo_url" REPO_REF="$repo_ref" OBSERVED_COMMIT="$observed_commit" SOURCE_PRUNE_PATH="$source_prune_path" SOURCE_PRUNED="$source_pruned" GIT_OBJECTS_REMOVED="$git_objects_removed" python3 -c \'import json, os; print(json.dumps({{"source":"oss-byof","repo":os.environ["REPO_URL"],"ref":os.environ["REPO_REF"],"commit":os.environ["OBSERVED_COMMIT"],"source_prune_path":os.environ["SOURCE_PRUNE_PATH"],"source_pruned":os.environ["SOURCE_PRUNED"]=="true","git_objects_removed":os.environ["GIT_OBJECTS_REMOVED"]=="true"}}, sort_keys=True))\' > {BYOF_REPO_MOUNT}/npa_source_metadata.json; \\\n'
        "    fi; \\\n"
        "    rm -f /tmp/npa-byof-git-credential; \\\n"
        f"    chown -R ubuntu:ubuntu {BYOF_REPO_MOUNT}\n"
        f"WORKDIR {BYOF_REPO_MOUNT}\n"
        'RUN if [ -n "${BYOF_BUILD_COMMAND}" ]; then /bin/sh -lc "${BYOF_BUILD_COMMAND}"; fi\n'
        "RUN build_command_sha256=\"$(printf '%s' \"${BYOF_BUILD_COMMAND}\" | sha256sum | cut -d' ' -f1)\" \\\n"
        '  && if [ -n "${BYOF_BUILD_COMMAND}" ]; then build_command_executed=true; else build_command_executed=false; fi \\\n'
        '  && if [ -n "${BYOF_BASE_IMAGE_DIGEST}" ] && [ "${BYOF_BASE_IMAGE#*@}" = "${BYOF_BASE_IMAGE_DIGEST}" ]; then base_image_digest_pinned=true; else base_image_digest_pinned=false; fi \\\n'
        f'  && printf \'{{"schema":"npa.byof.build.v1","build_command_executed":%s,"build_command_sha256":"%s","base_image_reference":"%s","base_image_digest":"%s","base_image_digest_pinned":%s}}\\n\' \\\n'
        f'    "$build_command_executed" "$build_command_sha256" "$BYOF_BASE_IMAGE" "$BYOF_BASE_IMAGE_DIGEST" "$base_image_digest_pinned" > {BYOF_REPO_MOUNT}/npa_build_metadata.json\n'
        f"RUN chown ubuntu:ubuntu {BYOF_REPO_MOUNT}/npa_source_metadata.json {BYOF_REPO_MOUNT}/npa_build_metadata.json\n"
        'LABEL npa.byof.repo="${BYOF_SOURCE_LABEL_REPO}" npa.byof.ref="${BYOF_SOURCE_LABEL_REF}" '
        'npa.tool="byof" npa.source.repo="${BYOF_SOURCE_LABEL_REPO}" '
        'npa.source.ref="${BYOF_SOURCE_LABEL_REF}" '
        'org.opencontainers.image.source="${BYOF_SOURCE_LABEL_REPO}" '
        'org.opencontainers.image.revision="${BYOF_SOURCE_LABEL_REF}" '
        'org.opencontainers.image.base.name="${BYOF_BASE_IMAGE}" '
        'org.opencontainers.image.base.digest="${BYOF_BASE_IMAGE_DIGEST}" '
        'npa.packaging.tier="interactive" '
        'org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1" '
        'org.nebius.npa.byof-bootstrap-guard="skypilot-0.12.2-v1"\n'
        "USER ubuntu\n"
        "ENV HOME=/home/ubuntu\n"
        f"WORKDIR {BYOF_REPO_MOUNT}\n"
        'ENTRYPOINT ["/bin/sh", "-c", "if [ \\"$#\\" -gt 0 ]; then exec \\"$@\\"; fi; exec /bin/bash", "npa-byof-entrypoint"]\n'
        'CMD ["/bin/bash"]\n'
    )


def _parse_last_json(text: str) -> dict[str, Any] | None:
    """Decode the final complete object embedded in mixed command output.

    The verifier emits a pretty-printed receipt, so line-oriented parsing can
    never recover its launch identity or terminal result.  Decode complete
    JSON values instead and let the authorized RoboTwin path validate the
    required receipt fields before recording a result.
    """

    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
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
            payload, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(payload, dict):
            candidates.append(payload)
        index = max(end, start + 1)
    return candidates[-1] if candidates else None


def _ubuntu_base_image_candidates() -> list[str]:
    configured = os.environ.get(
        "NPA_BYOF_UBUNTU_BASE_IMAGE", DEFAULT_UBUNTU_BASE_IMAGE
    ).strip()
    return [configured or DEFAULT_UBUNTU_BASE_IMAGE]


def _isaac_lab_base_image_candidates(*, image: str, registry: str) -> list[str]:
    candidates: list[str] = []
    derived_registry = _registry_path(image) or registry
    try:
        canonical = str(container_image_for_tool("isaac-lab")).strip()
        if canonical:
            candidates.append(canonical)
    except TypeError:
        pass
    for candidate_registry in (registry, derived_registry):
        candidate = str(
            container_image_for_tool("isaac-lab", registry=candidate_registry)
        ).strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    for public_candidate in (
        "nvcr.io/nvidia/isaac-lab:2.3.2",
        "nvcr.io/nvidia/isaac-sim:4.5.0",
    ):
        if public_candidate not in candidates:
            candidates.append(public_candidate)
    return candidates


def _base_image_candidates(
    *,
    profile: str,
    image: str,
    registry: str,
    explicit_base: str,
) -> list[str]:
    if explicit_base.startswith("tool://"):
        tool = explicit_base.removeprefix("tool://").strip()
        if not tool:
            raise ValueError("tool:// base image must name a registered image tool")
        resolved = container_image_for_tool(tool, registry=registry)
        if tool == "wan2-2":
            slash = resolved.rfind("/")
            colon = resolved.rfind(":")
            repository = resolved[:colon] if colon > slash else resolved
            resolved = (
                repository + "@" + str(wan_accepted_image_manifest()["oci_digest"])
            )
        return [resolved]
    if explicit_base:
        return [explicit_base]
    normalized_profile = profile if profile in BASE_PROFILES else "ubuntu"
    if normalized_profile == "isaac-lab":
        return _isaac_lab_base_image_candidates(image=image, registry=registry)
    if normalized_profile == "prebuilt":
        raise ValueError(
            "prebuilt profile requires --base-image tool://<registered-tool>"
        )
    return _ubuntu_base_image_candidates()


def _scan_robotwin_image(image: str, *, redactions: tuple[str, ...]) -> dict[str, Any]:
    if re.fullmatch(r".+@sha256:[0-9a-f]{64}", image) is None:
        raise ValueError("RoboTwin requires an immutable image before byte scanning")
    with tempfile.TemporaryDirectory(prefix="npa-robotwin-image-scan-") as tmp:
        report_path = Path(tmp) / "report.json"
        scanner_env = {
            name: os.environ[name]
            for name in ("DOCKER_CONFIG",)
            if os.environ.get(name)
        }
        _run(
            [
                sys.executable,
                str(ROBOTWIN_IMAGE_SCANNER),
                "--image-stdin",
                "--output",
                str(report_path),
            ],
            stdin=image,
            capture=True,
            env=scanner_env,
            redactions=redactions,
            inherit_env=False,
        )
        report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    if report.get("status") != "pass" or report.get("findings") != []:
        raise RuntimeError("RoboTwin exact-image byte scan did not pass")
    if report.get("image") != image or int(report.get("archives_scanned", 0)) < 2:
        raise RuntimeError("RoboTwin exact-image byte scan evidence is incomplete")
    return {
        "format": "npa_robotwin_image_byte_scan_v1",
        "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "archives_scanned": int(report["archives_scanned"]),
        "status": "pass",
    }


def _authorized_live_env(
    authorization: _RuntimeAuthorization | None,
    scan_evidence: dict[str, Any],
    *,
    project: str,
    image: str,
) -> dict[str, str]:
    if authorization is None:
        return _live_runner_env(project)
    # The outer worker receives only values selected by the owner client. Do not
    # inherit generic BYOF/SkyPilot knobs that could redirect or weaken the
    # already-authorized inner launch.
    env = {
        name: os.environ[name]
        for name in _ROBOTWIN_EXECUTION_BASELINE_ENV_NAMES
        if os.environ.get(name)
    }
    env.update(
        {
            name: os.environ[name]
            for name in (
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "AWS_SESSION_TOKEN",
            )
            if os.environ.get(name)
        }
    )
    endpoint = next(
        (
            os.environ[name]
            for name in (
                "AWS_ENDPOINT_URL",
                "NEBIUS_S3_ENDPOINT",
                "AWS_ENDPOINT_URL_S3",
                "NPA_STORAGE_ENDPOINT",
                "S3_ENDPOINT_URL",
            )
            if os.environ.get(name)
        ),
        "",
    )
    if endpoint:
        env.update(
            {
                "AWS_ENDPOINT_URL": endpoint,
                "NEBIUS_S3_ENDPOINT": endpoint,
            }
        )
    output_prefix = f"{authorization.output_root.rstrip('/')}/{authorization.run_id}/"
    env.update(
        {
            "KUBECONFIG": authorization.kubeconfig,
            "KUBECONTEXT": authorization.kubernetes_context,
            "NPA_BYOF_K8S_CONTEXT": authorization.kubernetes_context,
            "NPA_NEBIUS_PROFILE": authorization.profile,
            "NEBIUS_PROFILE": authorization.profile,
            "NPA_BYOF_PROJECT": authorization.project,
            ROBOTWIN_CHILD_BUCKET_ENV: authorization.bucket,
            ROBOTWIN_CHILD_CONFIG_PATH_ENV: authorization.skypilot_config_path,
            ROBOTWIN_CHILD_IMAGE_ENV: image,
            ROBOTWIN_CHILD_OUTPUT_PREFIX_ENV: output_prefix,
            ROBOTWIN_CHILD_OUTPUT_ROOT_ENV: authorization.output_root,
            ROBOTWIN_CHILD_RUN_ID_ENV: authorization.run_id,
            ROBOTWIN_CHILD_RUNTIME_AUTH_ENV: encode_runtime_authorization(
                authorization
            ),
            "NPA_BYOF_ROBOTWIN_RESERVATION_EVIDENCE_SHA256": authorization.context_sha256,
            "NPA_BYOF_ROBOTWIN_IMAGE_SCAN_SHA256": str(scan_evidence["report_sha256"]),
            "NPA_BYOF_ROBOTWIN_IMAGE_SCAN_ARCHIVES": str(
                scan_evidence["archives_scanned"]
            ),
        }
    )
    return env


def _private_runtime_redactions(env: dict[str, str]) -> tuple[str, ...]:
    """Close over every private environment value and useful textual alias."""

    aliases: list[str] = []
    public_evidence_names = {"NPA_BYOF_ROBOTWIN_IMAGE_SCAN_ARCHIVES"}

    def scalar_values(value: Any) -> list[str]:
        if isinstance(value, dict):
            return [item for nested in value.values() for item in scalar_values(nested)]
        if isinstance(value, (list, tuple)):
            return [item for nested in value for item in scalar_values(nested)]
        return [str(value)] if isinstance(value, (str, int, float)) else []

    def add_aliases(raw: str) -> None:
        value = raw.strip()
        if not value:
            return
        decoded = unquote(value)
        candidates = {value, decoded, value.rstrip("/"), decoded.rstrip("/")}
        for candidate in (value, decoded):
            try:
                parsed = urlsplit(candidate)
            except ValueError:
                parsed = None
            if parsed is not None and parsed.scheme and parsed.netloc:
                candidates.update(
                    {
                        f"{parsed.netloc}{parsed.path}".rstrip("/"),
                        parsed.netloc,
                        parsed.path,
                        parsed.path.rstrip("/"),
                    }
                )
        if "@sha256:" in value:
            candidates.add(value.split("@sha256:", 1)[0])
        aliases.extend(
            candidate for candidate in candidates if candidate and len(candidate) >= 4
        )

    for name, raw_value in env.items():
        value = str(raw_value or "").strip()
        if not value or name in public_evidence_names:
            continue
        add_aliases(value)
        try:
            structured = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            structured = None
        for scalar in scalar_values(structured):
            add_aliases(scalar)
    return tuple(dict.fromkeys(aliases))


def _run_robotwin_container_verify(
    cmd: list[str],
    *,
    authorization: _RuntimeAuthorization,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    """Keep the validated authorization in-process until the inner Sky bridge."""

    spec = importlib.util.spec_from_file_location(
        "npa_robotwin_container_verify", CONTAINER_VERIFY_RUNNER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("RoboTwin container verifier could not be loaded")
    module = importlib.util.module_from_spec(spec)
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        spec.loader.exec_module(module)
        returncode = module.run_authorized_robotwin(
            cmd[2:],
            authorization=authorization,
            environment=environment,
        )
    return subprocess.CompletedProcess(
        cmd,
        int(returncode),
        stdout=stdout.getvalue(),
        stderr=stderr.getvalue(),
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--repo-ref", default=DEFAULT_REPO_REF)
    parser.add_argument(
        "--repo-auth",
        choices=("none", "github"),
        default="none",
        help="Source authentication mode; github uses a BuildKit secret mount.",
    )
    parser.add_argument(
        "--repo-token-env",
        default="",
        help=(
            "Environment variable holding a GitHub token. When omitted, github "
            "auth uses GH_TOKEN, GITHUB_TOKEN, then the existing gh login."
        ),
    )
    parser.add_argument(
        "--project",
        default="",
        help="Project alias used for container-registry resolution.",
    )
    parser.add_argument("--registry", default="", help="Override registry host/path.")
    parser.add_argument(
        "--image", default="", help="Fully-qualified image ref to build/push and run."
    )
    parser.add_argument(
        "--base-profile",
        choices=sorted(BASE_PROFILES),
        default=os.environ.get("NPA_BYOF_BASE_PROFILE", "ubuntu"),
        help="Base family: ubuntu, isaac-lab, or prebuilt with tool://<registered-tool>.",
    )
    parser.add_argument(
        "--base-image",
        default=os.environ.get("NPA_BYOF_BASE_IMAGE", ""),
        help="Explicit base image (overrides --base-profile), e.g. ubuntu:24.04.",
    )
    parser.add_argument("--run-id", default=f"byof-{_utc_stamp()}")
    parser.add_argument(
        "--workload",
        choices=("rl-train", "datagen", "container-verify", "solution-smoke"),
        default="rl-train",
        help="Live workload: RL training, scripted datagen, container-verify, or solution smoke.",
    )
    parser.add_argument(
        "--build-command",
        default=os.environ.get("NPA_BYOF_BUILD_COMMAND", ""),
        help="Optional shell command run during image build from /opt/byof.",
    )
    parser.add_argument(
        "--source-prune-path",
        type=_source_prune_path,
        default="",
        help=(
            "Optional safe repo-relative path removed with .git in the source clone "
            "layer, before later image layers can retain its bytes."
        ),
    )
    parser.add_argument(
        "--smoke-command",
        default=os.environ.get("NPA_BYOF_SMOKE_COMMAND", ""),
        help="Optional documented shell command run during solution-smoke from /opt/byof.",
    )
    parser.add_argument(
        "--solution-name", default=os.environ.get("NPA_BYOF_SOLUTION_NAME", "")
    )
    parser.add_argument(
        "--runtime-context-env",
        default="",
        help=(
            "Environment-variable name containing owner-only runtime authorization; "
            "the authorization value never enters argv."
        ),
    )
    parser.add_argument(
        "--capability-name", default=os.environ.get("NPA_BYOF_CAPABILITY_NAME", "")
    )
    parser.add_argument(
        "--smoke-artifact-name",
        default=os.environ.get("NPA_BYOF_SMOKE_ARTIFACT_NAME", ""),
    )
    parser.add_argument(
        "--wan-acceptance-candidate-image",
        default="",
        help=(
            "Explicit official digest-pinned Wan image permitted only for this "
            "skip-build acceptance run; never read from ambient environment."
        ),
    )
    parser.add_argument(
        "--libero-qualified-candidate-image",
        default="",
        help=(
            "Explicit digest-pinned npa-libero image matching the checked-in "
            "qualification; never read from ambient environment."
        ),
    )
    parser.add_argument("--libero-customer-runtime-authorization-file", default="")
    parser.add_argument("--libero-authenticated-caller-identity-file", default="")
    parser.add_argument(
        "--num-envs", type=int, default=4, help="Parallel sim envs (datagen workload)."
    )
    parser.add_argument(
        "--num-demos",
        type=int,
        default=4,
        help="Demonstrations to record (datagen workload).",
    )
    parser.add_argument("--task", default="Isaac-Cartpole-v0")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument(
        "--yaml",
        default="",
        help="Optional SkyPilot YAML override for the selected workload.",
    )
    parser.add_argument(
        "--output-root", default="", help="Override workload output root."
    )
    parser.add_argument("--wait-timeout", type=int, default=21600)
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--sky-bin", default="")
    parser.add_argument(
        "--config-path",
        default="",
        help="SkyPilot global config YAML for kubernetes pod_config.",
    )
    parser.add_argument(
        "--cleanup", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-push", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    return parser.parse_args(argv)


def _authorization_failure(
    args: argparse.Namespace, exc: Exception, redactions: tuple[str, ...]
) -> int:
    """Emit one sanitized authorization refusal."""

    payload = {
        "status": "failed",
        "solution_name": args.solution_name,
        "error": _redact_text(str(exc), redactions),
    }
    print(json.dumps(_redact_payload(payload, redactions), indent=2, sort_keys=True))
    return 1


def _run_authorized_robotwin(
    argv: list[str], *, authorization: _RuntimeAuthorization
) -> int:
    """Run RoboTwin only from the validated normal-submit worker bridge.

    Args:
        argv: Exact public toolRef arguments already recognized by preflight.
        authorization: Validated manager context materialized by the CPU worker.

    Returns:
        The standard BYOF process exit status.
    """

    args = _parse_args(argv)
    redactions = tuple(
        dict.fromkeys(
            (
                *authorization.redactions,
                *_private_image_redactions(authorization.bootstrap_image),
            )
        )
    )
    try:
        require_runtime_lock_complete(authorization)
        _validate_robotwin_invocation(args)
        _apply_runtime_authorization(args, authorization)
    except Exception as exc:
        return _authorization_failure(args, exc, redactions)
    return _run_parsed(args, authorization=authorization, redactions=redactions)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if _is_robotwin_request(args):
        return _authorization_failure(
            args,
            ValueError(
                "RoboTwin requires the normal npa workbench workflow submit "
                "CPU launcher"
            ),
            (),
        )
    return _run_parsed(args, authorization=None, redactions=())


def _run_parsed(
    args: argparse.Namespace,
    *,
    authorization: _RuntimeAuthorization | None,
    redactions: tuple[str, ...],
) -> int:
    try:
        _validate_libero_identity(args)
    except LiberoCustomerAcceptanceRequired as exc:
        print(json.dumps(exc.notification, indent=2, sort_keys=True))
        return 3
    except ValueError as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "solution_name": args.solution_name,
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    private_source = args.repo_auth == "github"
    effective_redactions = redactions
    try:
        validate_repository_url(args.repo_url, private=private_source)
    except RepositoryAuthenticationError as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "repo_auth": args.repo_auth,
                    "error": str(exc),
                },
                indent=2,
            )
        )
        return 1
    if is_openpi_request(
        solution_name=args.solution_name,
        repo_url=args.repo_url,
        smoke_command=args.smoke_command,
    ):
        try:
            require_openpi_terms()
        except ValueError as exc:
            print(
                json.dumps(
                    {
                        "status": "failed",
                        "solution_name": args.solution_name or "openpi",
                        "error": str(exc),
                    },
                    indent=2,
                )
            )
            return 1
    summary: dict[str, Any] = {
        "repo_url": "<private-repository>" if private_source else args.repo_url,
        "repo_ref": "<private-ref>" if private_source else args.repo_ref,
        "repo_auth": args.repo_auth,
        "run_id": args.run_id,
        "workload": args.workload,
        "build_command": args.build_command,
        "source_prune_path": (
            "<private-source-prune-path>"
            if private_source and args.source_prune_path
            else args.source_prune_path
        ),
        "smoke_command": args.smoke_command,
        "solution_name": args.solution_name,
        "capability_name": args.capability_name,
        "smoke_artifact_name": args.smoke_artifact_name,
    }

    # BYOF registries are standards-based operator interoperability. Authentication
    # must already exist in the caller's Docker config; NPA never mints provider IAM
    # registry tokens or creates a hidden registry-specific credential directory.
    docker_env: dict[str, str] = {}
    try:
        explicit_base = _normalize_optional(args.base_image)
        if args.solution_name.strip().lower() == LIBERO_SOLUTION_NAME:
            explicit_base = _libero_qualified_candidate(
                args.libero_qualified_candidate_image, args._libero_qualification
            )
        base_profile = _normalize_optional(args.base_profile) or "ubuntu"
        registry = args.registry.strip() or resolve_container_registry(
            args.project or None
        )
        image = args.image.strip() or f"{registry.rstrip('/')}/npa-byof:{args.run_id}"
        base_candidates = _base_image_candidates(
            profile=base_profile,
            image=image,
            registry=registry,
            explicit_base=explicit_base,
        )
        if not base_candidates:
            raise RuntimeError("unable to resolve a BYOF base image candidate")
        base_image = base_candidates[0]
        if base_profile == "prebuilt":
            if args.build_command.strip():
                raise ValueError(
                    "prebuilt profile forbids --build-command; image bytes are immutable"
                )
            image = base_image
        skip_build = args.skip_build or base_profile == "prebuilt"
        skip_push = args.skip_push or base_profile == "prebuilt"
        summary.update(
            {
                "registry": registry,
                "base_profile": base_profile,
                "base_registry": _registry_path(base_image)
                or (_registry_path(image) or registry),
                "image": image,
                "base_image": base_image,
                "base_image_candidates": base_candidates,
            }
        )
        with ExitStack() as secret_stack:
            source_secrets: RepositorySecretFiles | None = None
            if private_source:
                source_secrets = secret_stack.enter_context(
                    private_repository_secrets(
                        args.repo_url,
                        args.repo_ref,
                        source_prune_path=args.source_prune_path,
                        token_env=args.repo_token_env,
                    )
                )
                summary["source_identity"] = {
                    "repository_sha256": source_secrets.repository_sha256,
                    "ref_sha256": source_secrets.ref_sha256,
                    "source_prune_path_sha256": (
                        source_secrets.source_prune_path_sha256
                    ),
                }
            effective_redactions = tuple(
                dict.fromkeys(
                    (
                        *effective_redactions,
                        *(
                            source_secrets.redaction_values
                            if source_secrets is not None
                            else ()
                        ),
                    )
                )
            )
            return _run_byof(
                args,
                authorization=authorization,
                summary=summary,
                source_secrets=source_secrets,
                redactions=effective_redactions,
                docker_env=docker_env,
                base_candidates=base_candidates,
                base_image=base_image,
                base_profile=base_profile,
                image=image,
                registry=registry,
                skip_build=skip_build,
                skip_push=skip_push,
            )
    except Exception as exc:
        message = _redact_text(str(exc), effective_redactions)
        summary["status"] = "failed"
        summary["error"] = message
        if isinstance(exc, RepositoryAuthenticationError):
            summary["hint"] = (
                "Provide a fine-grained GitHub token with repository read access via "
                "--repo-token-env, or refresh the existing GitHub CLI login."
            )
        elif "403 Forbidden" in message and (
            "BYOF_BASE_IMAGE" in message or "ISAAC_BASE_IMAGE" in message
        ):
            summary["hint"] = (
                "Registry pull for the base image was denied. "
                "Pass --base-image from an accessible registry (e.g. ubuntu:22.04), "
                "or use --base-profile isaac-lab with registry access to the sim image."
            )
        elif "docker push" in message and "403 Forbidden" in message:
            summary["hint"] = (
                "Registry push was denied for the target image. "
                "Grant write access to the target repository, or use --skip-push "
                "with an already-published image."
            )
        print(
            json.dumps(
                _redact_payload(summary, effective_redactions),
                indent=2,
                sort_keys=True,
            )
        )
        hint = str(summary.get("hint") or "").strip()
        if hint:
            print(f"HINT: {hint}", file=sys.stderr)
        return 1


def _postprocess_solution(
    args: argparse.Namespace, key: str | None, summary: dict[str, Any]
) -> None:
    if key is None:
        return
    result = run_registered_postprocess(
        key,
        PostprocessContext(
            run_prefix_uri=f"{args.output_root.rstrip('/')}/{args.run_id}/",
            project=args.project or None,
            wan_acceptance_candidate_image=args.wan_acceptance_candidate_image.strip(),
        ),
    )
    if result is None:
        raise RuntimeError(
            f"registered solution {key!r} returned no verified postprocess result"
        )
    summary["postprocess"] = result


def _run_worker(
    args: argparse.Namespace,
    summary: dict[str, Any],
    *,
    image: str,
    base_profile: str,
    postprocess_key: str | None,
) -> int:
    if base_profile != "prebuilt" or args.workload != "solution-smoke" or args.skip_run:
        raise ValueError(
            "Allocated BYOF workflow workers support prebuilt solution-smoke only; "
            "build and launch other workloads from the operator host"
        )
    summary["build"] = {"ok": True, "skipped": True}
    summary["run"] = run_prebuilt_smoke(
        WorkerSmoke(
            run_id=args.run_id,
            image=image,
            repo_url=args.repo_url,
            repo_ref=args.repo_ref,
            output_root=args.output_root,
            command=args.smoke_command,
            solution=args.solution_name,
            capability=args.capability_name,
            artifact_name=args.smoke_artifact_name,
            repo_root=Path(BYOF_REPO_MOUNT),
        )
    )
    _postprocess_solution(args, postprocess_key, summary)
    summary["status"] = "ok"
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _run_byof(
    args: argparse.Namespace,
    *,
    authorization: _RuntimeAuthorization | None,
    summary: dict[str, Any],
    source_secrets: RepositorySecretFiles | None,
    redactions: tuple[str, ...],
    docker_env: dict[str, str],
    base_candidates: list[str],
    base_image: str,
    base_profile: str,
    image: str,
    registry: str,
    skip_build: bool,
    skip_push: bool,
) -> int:
    effective_redactions = redactions
    try:
        postprocess_key = _required_postprocess_key(
            args, base_image=base_image, base_profile=base_profile
        )
        if postprocess_key is not None and not args.output_root.strip():
            raise ValueError(
                f"registered solution {postprocess_key!r} requires --output-root "
                "so its verified postprocess cannot be skipped"
            )
        if postprocess_key is not None and args.skip_run:
            raise ValueError(
                f"registered solution {postprocess_key!r} cannot use --skip-run "
                "because verified postprocessing is mandatory"
            )
        if args.source_prune_path and skip_build:
            raise ValueError(
                "--source-prune-path requires building the source image in this invocation"
            )
        # RoboTwin's authenticated CPU outer task launches its one reserved RTX
        # workload; it is not the generic already-allocated capability worker.
        if authorization is None and in_workflow_worker():
            return _run_worker(
                args,
                summary,
                image=image,
                base_profile=base_profile,
                postprocess_key=postprocess_key,
            )
        if not skip_build:
            with tempfile.TemporaryDirectory(prefix="npa-byof-build-") as tmp:
                context = Path(tmp)
                (context / "Dockerfile").write_text(
                    _dockerfile_text(), encoding="utf-8"
                )
                last_build_error: Exception | None = None
                for idx, candidate_base in enumerate(base_candidates):
                    base_image = candidate_base
                    summary["base_image"] = base_image
                    summary["base_registry"] = _registry_path(base_image) or (
                        _registry_path(image) or registry
                    )
                    try:
                        build_cmd = [
                            "docker",
                            "build",
                            "--platform",
                            "linux/amd64",
                            "--build-arg",
                            f"BYOF_BASE_IMAGE={base_image}",
                            "--build-arg",
                            f"BYOF_BASE_IMAGE_DIGEST={_immutable_image_digest(base_image)}",
                            "--build-arg",
                            f"BYOF_SOURCE_VISIBILITY={'private' if source_secrets else 'public'}",
                            "--build-arg",
                            (
                                "BYOF_SOURCE_CACHE_KEY="
                                + (
                                    source_secrets.repository_sha256
                                    + source_secrets.ref_sha256
                                    + source_secrets.source_prune_path_sha256
                                    if source_secrets
                                    else "public"
                                )
                            ),
                            "--build-arg",
                            f"BYOF_SOURCE_LABEL_REPO={'<private-repository>' if source_secrets else args.repo_url}",
                            "--build-arg",
                            f"BYOF_SOURCE_LABEL_REF={'<private-ref>' if source_secrets else args.repo_ref}",
                            "--build-arg",
                            f"BYOF_BUILD_COMMAND={args.build_command}",
                            "-t",
                            image,
                            str(context),
                        ]
                        if source_secrets is None:
                            build_cmd[8:8] = [
                                "--build-arg",
                                f"OSS_REPO_URL={args.repo_url}",
                                "--build-arg",
                                f"OSS_REPO_REF={args.repo_ref}",
                            ]
                        else:
                            build_cmd[8:8] = [
                                "--secret",
                                f"id=npa_byof_repo_token,src={source_secrets.token}",
                                "--secret",
                                f"id=npa_byof_repo_url,src={source_secrets.repo_url}",
                                "--secret",
                                f"id=npa_byof_repo_ref,src={source_secrets.repo_ref}",
                                "--secret",
                                (
                                    "id=npa_byof_source_prune_path,src="
                                    f"{source_secrets.source_prune_path}"
                                ),
                            ]
                        if source_secrets is None:
                            build_cmd[8:8] = [
                                "--build-arg",
                                f"BYOF_SOURCE_PRUNE_PATH={args.source_prune_path}",
                            ]
                        run_kwargs: dict[str, Any] = {
                            "env": docker_env or None,
                            "capture": True,
                        }
                        if redactions:
                            run_kwargs["redactions"] = redactions
                        _run(build_cmd, **run_kwargs)
                        break
                    except Exception as exc:
                        last_build_error = exc
                        message = str(exc)
                        forbidden_pull = "403 Forbidden" in message and (
                            "BYOF_BASE_IMAGE" in message
                            or "ISAAC_BASE_IMAGE" in message
                            or "failed to resolve source metadata" in message
                            or "pull access denied" in message
                        )
                        if forbidden_pull and idx + 1 < len(base_candidates):
                            continue
                        raise
                else:
                    assert last_build_error is not None
                    raise last_build_error
            if not skip_push:
                push_kwargs: dict[str, Any] = {
                    "env": docker_env or None,
                    "capture": True,
                }
                if redactions:
                    push_kwargs["redactions"] = redactions
                push_proc = _run(["docker", "push", image], **push_kwargs)
                if push_proc.stdout:
                    sys.stdout.write(_redact_text(push_proc.stdout, redactions))
                if push_proc.stderr:
                    sys.stderr.write(_redact_text(push_proc.stderr, redactions))
                tagged_image = image
                image = _resolve_pushed_image_digest(
                    tagged_image,
                    env=docker_env or None,
                    redactions=redactions,
                )
                summary["image_tag"] = tagged_image
                summary["image"] = image
                summary["build"] = {
                    "ok": True,
                    "pushed": True,
                    "runtime_image": image,
                    "digest": image.rsplit("@", 1)[1],
                }
            else:
                summary["build"] = {"ok": True, "pushed": False}
        else:
            summary["build"] = {"ok": True, "skipped": True}

        scan_evidence: dict[str, Any] = {}
        if authorization is not None:
            scan_evidence = _scan_robotwin_image(image, redactions=effective_redactions)
            summary["robotwin_image_scan"] = scan_evidence

        if not args.skip_run:
            if args.workload == "datagen":
                cmd = [
                    sys.executable,
                    str(DATAGEN_RUNNER),
                    "--image",
                    image,
                    "--task",
                    args.task,
                    "--num-envs",
                    str(args.num_envs),
                    "--num-demos",
                    str(args.num_demos),
                    "--run-id",
                    args.run_id,
                    "--wait-timeout",
                    str(args.wait_timeout),
                    "--poll-interval",
                    str(args.poll_interval),
                    "--repo-root",
                    BYOF_REPO_MOUNT,
                ]
            elif args.workload in {"container-verify", "solution-smoke"}:
                cmd = [sys.executable, str(CONTAINER_VERIFY_RUNNER)]
                if authorization is None:
                    cmd.extend(["--image", image, "--run-id", args.run_id])
                cmd.extend(
                    [
                        "--wait-timeout",
                        str(args.wait_timeout),
                        "--poll-interval",
                        str(args.poll_interval),
                        "--repo-root",
                        BYOF_REPO_MOUNT,
                    ]
                )
                if args.smoke_command:
                    cmd.extend(["--smoke-command", args.smoke_command])
                if args.solution_name:
                    cmd.extend(["--solution-name", args.solution_name])
                if args.capability_name:
                    cmd.extend(["--capability-name", args.capability_name])
                if args.smoke_artifact_name:
                    cmd.extend(["--smoke-artifact-name", args.smoke_artifact_name])
                if args.solution_name.strip().lower() == LIBERO_SOLUTION_NAME:
                    cmd.append("--no-direct-launch")
            else:
                cmd = [
                    sys.executable,
                    str(ISAAC_RUNNER),
                    "--image",
                    image,
                    "--task",
                    args.task,
                    "--iterations",
                    str(args.iterations),
                    "--run-id",
                    args.run_id,
                    "--wait-timeout",
                    str(args.wait_timeout),
                    "--poll-interval",
                    str(args.poll_interval),
                ]
            if args.yaml:
                cmd.extend(["--yaml", args.yaml])
            if args.output_root and authorization is None:
                cmd.extend(["--output-root", args.output_root])
            if args.sky_bin:
                cmd.extend(["--sky-bin", args.sky_bin])
            if args.config_path and authorization is None:
                cmd.extend(["--config-path", args.config_path])
            if args.cleanup:
                cmd.append("--cleanup")
            if args.solution_name.strip().lower() == LIBERO_SOLUTION_NAME:
                live_env = _live_runner_env(args.project, libero=True)
            else:
                live_env = _authorized_live_env(
                    authorization, scan_evidence, project=args.project, image=image
                )
            effective_redactions = (
                tuple(
                    dict.fromkeys(
                        (
                            *effective_redactions,
                            *_private_runtime_redactions(live_env),
                        )
                    )
                )
                if authorization is not None
                else effective_redactions
            )
            if args.solution_name.strip().lower() == LIBERO_SOLUTION_NAME:
                authorization_bytes = args._libero_customer_authorization_bytes
                authorization_sha256 = args._libero_customer_authorization_sha256
                live_env.update(
                    {
                        "NPA_LIBERO_CUSTOMER_AUTHORIZATION_B64": base64.b64encode(
                            authorization_bytes
                        ).decode("ascii"),
                        "NPA_LIBERO_CUSTOMER_AUTHORIZATION_SHA256": (
                            authorization_sha256
                        ),
                        "NPA_LIBERO_CUSTOMER_IDENTITY_SHA256": (
                            args._libero_customer_identity_sha256
                        ),
                        "NPA_LIBERO_AUTHENTICATED_CALLER_B64": base64.b64encode(
                            args._libero_authenticated_caller_bytes
                        ).decode("ascii"),
                        "NPA_LIBERO_AUTHENTICATED_CALLER_SHA256": (
                            args._libero_authenticated_caller_sha256
                        ),
                        "NPA_LIBERO_EXPECTED_CANONICAL_BUILD_METADATA_SHA256": (
                            args._libero_qualification[
                                "canonical_build_metadata_sha256"
                            ]
                        ),
                    }
                )
            run_kwargs = {
                "capture": True,
                "env": live_env,
            }
            if authorization is not None:
                run_kwargs["inherit_env"] = False
            if effective_redactions:
                run_kwargs["redactions"] = effective_redactions
            run_proc = (
                _run_robotwin_container_verify(
                    cmd,
                    authorization=authorization,
                    environment=live_env,
                )
                if authorization is not None
                else _run(cmd, **run_kwargs)
            )
            # Validate the authorized receipt against its raw captured bytes before
            # redaction. The run-scoped launch identity is intentionally also present
            # in the private runtime authorization envelope, so redacting first would
            # turn a valid receipt into a deterministic identity mismatch. Raw bytes
            # remain transient and are never printed, stored, or included in errors.
            parsed_run = (
                _parse_last_json(run_proc.stdout) if authorization is not None else None
            )
            sanitized_stdout = _redact_text(run_proc.stdout, effective_redactions)
            sanitized_stderr = _redact_text(run_proc.stderr, effective_redactions)
            sys.stdout.write(sanitized_stdout)
            if sanitized_stderr:
                sys.stderr.write(sanitized_stderr)
            if authorization is not None and run_proc.returncode != 0:
                summary["run"] = {
                    "status": "failed",
                    "returncode": run_proc.returncode,
                    "stdout": sanitized_stdout,
                    "stderr": sanitized_stderr,
                }
                raise RuntimeError(
                    f"authorized RoboTwin verifier failed with exit {run_proc.returncode}"
                )
            if authorization is None:
                parsed_run = _parse_last_json(sanitized_stdout)
            if authorization is not None:
                if not isinstance(parsed_run, dict):
                    raise RuntimeError(
                        "authorized RoboTwin verifier returned no complete JSON receipt"
                    )
                if parsed_run.get("launch_id") != authorization.inner_launch_id:
                    raise RuntimeError(
                        "authorized RoboTwin verifier receipt launch identity mismatch"
                    )
                if (
                    str(parsed_run.get("status", "")).strip().lower() != "succeeded"
                    or type(parsed_run.get("returncode")) is not int
                    or parsed_run.get("returncode") != 0
                ):
                    raise RuntimeError(
                        "authorized RoboTwin verifier receipt was not successful"
                    )
            else:
                parsed_run = parsed_run or {"status": "submitted"}
            summary["run"] = _redact_payload(parsed_run, effective_redactions)
            _postprocess_solution(args, postprocess_key, summary)
        else:
            summary["run"] = {"skipped": True}
        summary["status"] = "ok"
        print(
            json.dumps(
                _redact_payload(summary, effective_redactions),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        _raise_sanitized_runtime_error(_redact_text(str(exc), effective_redactions))


def _raise_sanitized_runtime_error(message: str) -> None:
    """Raise without retaining any exception handled by the BYOF boundary."""

    try:
        raise RuntimeError(message) from None
    except RuntimeError as failure:
        failure.__cause__ = None
        failure.__context__ = None
        raise


if __name__ == "__main__":
    raise SystemExit(main())
