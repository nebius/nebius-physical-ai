#!/usr/bin/env python3
"""Build/push a BYOF image from an OSS repo and launch a live workload on Nebius."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from npa.clients.config import resolve_container_registry
from npa.clients.project_credentials import storage_env_for_project
from npa.deploy.images import (
    container_image_for_tool,
    is_public_registry,
    wan_accepted_image_manifest,
)
from npa.workflows.byof.live import resolve_byof_kubernetes_target
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

SCRIPT_DIR = Path(__file__).resolve().parent
ISAAC_RUNNER = SCRIPT_DIR / "run_isaac_lab_rl.py"
DATAGEN_RUNNER = SCRIPT_DIR / "run_byof_datagen.py"
CONTAINER_VERIFY_RUNNER = SCRIPT_DIR / "run_byof_container_verify.py"
ROBOTWIN_IMAGE_SCANNER = SCRIPT_DIR / "scan_image_robotwin_payload.py"
BYOF_REPO_MOUNT = "/opt/byof"

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
ROBOTWIN_RUNTIME_CONTEXT_ENV = "NPA_BYOF_ROBOTWIN_RUNTIME_CONTEXT"
ROBOTWIN_REQUIRED_DECISIONS = (
    "nvidia_cuda_eula",
    "nvidia_cudnn_sla",
    "curobo_noncommercial_research_or_evaluation",
    "robotwin2_aggregate_asset_and_output_terms",
)
ROBOTWIN_INVOCATION = {
    "repo_url": "https://github.com/RoboTwin-Platform/RoboTwin.git",
    "repo_ref": "96c1feab536306b50c26af200044fcdf126e8904",
    "base_profile": "ubuntu",
    "base_image": (
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04@"
        "sha256:61f6c08f2b59036cb935e56d1e31a6b64e3ae2c7ddb86d33fa0b044c7917b719"
    ),
    "workload": "solution-smoke",
    "capability_name": "beat_block_hammer_successful_seed_replay_collection",
    "smoke_artifact_name": "robotwin-smoke.json",
    "yaml": "byof-solution-smoke-robotwin-rtxpro-gpu",
    "task": "beat_block_hammer",
}
ROBOTWIN_BUILD_COMMAND_SHA256 = (
    "87cb636a259cbec5d59283e9e03cc076b139182fb01f9b4504c81040ae384931"
)
ROBOTWIN_SMOKE_COMMAND_SHA256 = (
    "bc26a3019d36863326cf4e55bd3536c303355c7b661ede5bc20ed9a4607deb2d"
)


@dataclass(frozen=True)
class _RuntimeAuthorization:
    project: str
    profile: str
    kubeconfig: str
    kubernetes_context: str
    skypilot_config_path: str
    registry: str
    bucket: str
    output_root: str
    run_id: str
    context_sha256: str
    redactions: tuple[str, ...]


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
        return {key: _redact_payload(item, redactions) for key, item in value.items()}
    return value


def _runtime_context_bytes(reference: str) -> bytes:
    secret = os.environ.get(reference, "").strip()
    if not secret:
        raise ValueError(f"{reference} is required and must not be empty")
    if secret.startswith("{"):
        return secret.encode()
    path = Path(secret).expanduser()
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("runtime authorization must be a regular file")
        if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
            raise ValueError("runtime authorization file must be owner-only")
        if metadata.st_size > 64 * 1024:
            raise ValueError("runtime authorization file is unexpectedly large")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            raw = stream.read(64 * 1024 + 1)
    except OSError as exc:
        raise ValueError("runtime authorization file is not readable") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if len(raw) > 64 * 1024:
        raise ValueError("runtime authorization file is unexpectedly large")
    return raw


def _context_text(payload: dict[str, Any], field: str) -> str:
    value = str(payload.get(field) or "").strip()
    if not value:
        raise ValueError(f"runtime authorization lacks {field}")
    return value


def _validate_robotwin_context(
    payload: dict[str, Any], raw: bytes
) -> _RuntimeAuthorization:
    if payload.get("solution") != "robotwin":
        raise ValueError("runtime authorization has the wrong solution")
    _context_text(payload, "ownership_provenance")
    reservation = payload.get("reservation")
    if not isinstance(reservation, dict):
        raise ValueError("runtime authorization lacks reservation evidence")
    if reservation.get("policy") != "STRICT":
        raise ValueError("RoboTwin reservation policy must be STRICT")
    if reservation.get("accelerator") != "RTXPRO-6000-BLACKWELL-SERVER-EDITION":
        raise ValueError("RoboTwin requires the RTX PRO 6000 Blackwell target")
    if reservation.get("count") != 1:
        raise ValueError("RoboTwin requires exactly one reserved GPU")
    decisions = payload.get("license_acceptance")
    if not isinstance(decisions, dict):
        raise ValueError("runtime authorization lacks operator use decisions")
    for decision in ROBOTWIN_REQUIRED_DECISIONS:
        if decisions.get(decision) is not True:
            raise ValueError(f"runtime authorization does not permit {decision}")
    return _robotwin_authorization(payload, raw)


def _robotwin_authorization(
    payload: dict[str, Any], raw: bytes
) -> _RuntimeAuthorization:
    values = {
        field: _context_text(payload, field)
        for field in (
            "project",
            "nebius_profile",
            "kubeconfig",
            "kubernetes_context",
            "skypilot_config_path",
            "registry",
            "bucket",
            "output_root",
            "run_id",
        )
    }
    if is_public_registry(values["registry"]):
        raise ValueError("RoboTwin requires an operator-private registry")
    parsed_output = urlparse(values["output_root"])
    if parsed_output.scheme != "s3" or parsed_output.netloc != values["bucket"]:
        raise ValueError("RoboTwin output storage does not match its authorized bucket")
    if not values["run_id"].startswith("robotwin-"):
        raise ValueError("RoboTwin requires a solution-scoped run ID")
    redactions = tuple(dict.fromkeys((*values.values(), raw.decode(errors="replace"))))
    return _RuntimeAuthorization(
        project=values["project"],
        profile=values["nebius_profile"],
        kubeconfig=values["kubeconfig"],
        kubernetes_context=values["kubernetes_context"],
        skypilot_config_path=values["skypilot_config_path"],
        registry=values["registry"],
        bucket=values["bucket"],
        output_root=values["output_root"],
        run_id=values["run_id"],
        context_sha256=hashlib.sha256(raw).hexdigest(),
        redactions=redactions,
    )


def _load_runtime_authorization(
    args: argparse.Namespace,
) -> _RuntimeAuthorization | None:
    solution = args.solution_name.strip().lower()
    reference = args.runtime_context_env.strip()
    if solution != "robotwin":
        if reference:
            raise ValueError("runtime authorization is not supported for this solution")
        return None
    if reference != ROBOTWIN_RUNTIME_CONTEXT_ENV:
        raise ValueError(
            f"RoboTwin requires --runtime-context-env {ROBOTWIN_RUNTIME_CONTEXT_ENV}"
        )
    raw = _runtime_context_bytes(reference)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("runtime authorization is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("runtime authorization must be a JSON object")
    return _validate_robotwin_context(payload, raw)


def _apply_runtime_authorization(
    args: argparse.Namespace, authorization: _RuntimeAuthorization
) -> None:
    args.project = authorization.project
    args.registry = authorization.registry
    args.output_root = authorization.output_root
    args.run_id = authorization.run_id
    args.config_path = authorization.skypilot_config_path


def _validate_robotwin_invocation(args: argparse.Namespace) -> None:
    for field, expected in ROBOTWIN_INVOCATION.items():
        if getattr(args, field) != expected:
            raise ValueError(f"RoboTwin invocation has unexpected {field}")
    command_hashes = {
        "build_command": ROBOTWIN_BUILD_COMMAND_SHA256,
        "smoke_command": ROBOTWIN_SMOKE_COMMAND_SHA256,
    }
    for field, expected in command_hashes.items():
        observed = hashlib.sha256(getattr(args, field).encode()).hexdigest()
        if observed != expected:
            raise ValueError(f"RoboTwin invocation has unexpected {field}")
    if (
        args.repo_auth != "none"
        or args.iterations != 1
        or args.num_envs != 1
        or args.num_demos != 1
        or args.wait_timeout != -1
        or args.poll_interval != 60
        or not args.cleanup
    ):
        raise ValueError("RoboTwin invocation does not match the accepted smoke contract")
    if any(
        getattr(args, field).strip()
        for field in ("project", "registry", "image", "config_path")
    ) or args.output_root.strip() not in {
        "",
        "s3://example-bucket/oss-solutions/robotwin",
    }:
        raise ValueError(
            "RoboTwin private runtime coordinates must come from authorization"
        )
    if args.skip_build or args.skip_push or args.skip_run:
        raise ValueError("RoboTwin requires the complete build, scan, and live smoke path")


def _activate_runtime_profile(authorization: _RuntimeAuthorization) -> None:
    _run(
        ["nebius", "profile", "activate", authorization.profile],
        capture=True,
        redactions=authorization.redactions,
    )


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _normalize_optional(value: str) -> str:
    cleaned = str(value or "").strip()
    if cleaned in PLACEHOLDER_VALUES:
        return ""
    return cleaned


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
) -> subprocess.CompletedProcess[str]:
    runtime_env = dict(os.environ)
    # Avoid stale operator tokens overriding profile-based auth on shared VMs.
    runtime_env.pop("NEBIUS_IAM_TOKEN", None)
    runtime_env.pop("NEBIUS_IAM_TOKEN_FILE", None)
    runtime_env.pop(ROBOTWIN_RUNTIME_CONTEXT_ENV, None)
    if env is not None:
        runtime_env.update(env)
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
    project: str, *, redactions: tuple[str, ...] = ()
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
    if "NPA_S3_BUCKET" not in env:
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


def _dockerfile_text() -> str:
    return (
        "# syntax=docker/dockerfile:1.7\n"
        "ARG BYOF_BASE_IMAGE\n"
        "FROM ${BYOF_BASE_IMAGE}\n"
        'ARG OSS_REPO_URL=""\n'
        'ARG OSS_REPO_REF=""\n'
        "ARG BYOF_SOURCE_VISIBILITY=public\n"
        "ARG BYOF_SOURCE_CACHE_KEY=public\n"
        "ARG BYOF_SOURCE_LABEL_REPO\n"
        "ARG BYOF_SOURCE_LABEL_REF\n"
        "ARG BYOF_BUILD_COMMAND\n"
        "USER root\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends \\\n"
        "      git ca-certificates python3 python3-pip sudo rsync \\\n"
        "      openssh-client openssh-server netcat-openbsd \\\n"
        "  && rm -rf /var/lib/apt/lists/*\n"
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
        "RUN --mount=type=secret,id=npa_byof_repo_token \\\n"
        "    --mount=type=secret,id=npa_byof_repo_url \\\n"
        "    --mount=type=secret,id=npa_byof_repo_ref \\\n"
        "    set -eu; \\\n"
        '    test -n "${BYOF_SOURCE_CACHE_KEY}"; \\\n'
        '    repo_url="${OSS_REPO_URL}"; repo_ref="${OSS_REPO_REF}"; \\\n'
        "    if [ -s /run/secrets/npa_byof_repo_url ]; then repo_url=\"$(cat /run/secrets/npa_byof_repo_url)\"; fi; \\\n"
        "    if [ -s /run/secrets/npa_byof_repo_ref ]; then repo_ref=\"$(cat /run/secrets/npa_byof_repo_ref)\"; fi; \\\n"
        "    export GIT_TERMINAL_PROMPT=0; \\\n"
        "    git_with_auth() { git \"$@\"; }; \\\n"
        "    if [ -s /run/secrets/npa_byof_repo_token ]; then \\\n"
        "      printf '%s\\n' '#!/bin/sh' \\\n"
        "        '[ \"$1\" = get ] || exit 0' \\\n"
        "        'printf \"username=x-access-token\\\\npassword=\"' \\\n"
        "        'cat /run/secrets/npa_byof_repo_token' \\\n"
        "        'printf \"\\\\n\\\\n\"' \\\n"
        "        > /tmp/npa-byof-git-credential; \\\n"
        "      chmod 700 /tmp/npa-byof-git-credential; \\\n"
        "      git_with_auth() { git -c credential.useHttpPath=true -c credential.helper=/tmp/npa-byof-git-credential \"$@\"; }; \\\n"
        "    fi; \\\n"
        f'    git_with_auth clone --depth 1 --branch "$repo_ref" "$repo_url" {BYOF_REPO_MOUNT} \\\n'
        f"    || (rm -rf {BYOF_REPO_MOUNT}; \\\n"
        f'      git_with_auth clone "$repo_url" {BYOF_REPO_MOUNT}; \\\n'
        f"      cd {BYOF_REPO_MOUNT}; git checkout \"$repo_ref\"); \\\n"
        f"    if [ \"${{BYOF_SOURCE_VISIBILITY}}\" = private ]; then \\\n"
        "      repo_sha=\"$(printf '%s' \"$repo_url\" | sha256sum | cut -d' ' -f1)\"; \\\n"
        "      ref_sha=\"$(printf '%s' \"$repo_ref\" | sha256sum | cut -d' ' -f1)\"; \\\n"
        f"      printf '{{\"source\":\"private-byof\",\"repository_sha256\":\"%s\",\"ref_sha256\":\"%s\"}}\\n' \"$repo_sha\" \"$ref_sha\" > {BYOF_REPO_MOUNT}/npa_source_metadata.json; \\\n"
        f"      rm -rf {BYOF_REPO_MOUNT}/.git; \\\n"
        "    else \\\n"
        f"      printf '{{\\n  \"source\": \"oss-byof\",\\n  \"repo\": \"%s\",\\n  \"ref\": \"%s\"\\n}}\\n' \"$repo_url\" \"$repo_ref\" > {BYOF_REPO_MOUNT}/npa_source_metadata.json; \\\n"
        "    fi; \\\n"
        "    rm -f /tmp/npa-byof-git-credential; \\\n"
        f"    chown -R ubuntu:ubuntu {BYOF_REPO_MOUNT}\n"
        f"WORKDIR {BYOF_REPO_MOUNT}\n"
        'RUN if [ -n "${BYOF_BUILD_COMMAND}" ]; then /bin/sh -lc "${BYOF_BUILD_COMMAND}"; fi\n'
        "RUN build_command_sha256=\"$(printf '%s' \"${BYOF_BUILD_COMMAND}\" | sha256sum | cut -d' ' -f1)\" \\\n"
        '  && if [ -n "${BYOF_BUILD_COMMAND}" ]; then build_command_executed=true; else build_command_executed=false; fi \\\n'
        f'  && printf \'{{"schema":"npa.byof.build.v1","build_command_executed":%s,"build_command_sha256":"%s"}}\\n\' \\\n'
        f'    "$build_command_executed" "$build_command_sha256" > {BYOF_REPO_MOUNT}/npa_build_metadata.json\n'
        f"RUN chown ubuntu:ubuntu {BYOF_REPO_MOUNT}/npa_source_metadata.json {BYOF_REPO_MOUNT}/npa_build_metadata.json\n"
        'LABEL npa.byof.repo="${BYOF_SOURCE_LABEL_REPO}" npa.byof.ref="${BYOF_SOURCE_LABEL_REF}" '
        'npa.packaging.tier="interactive" '
        'org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"\n'
        "USER ubuntu\n"
        "ENV HOME=/home/ubuntu\n"
        f"WORKDIR {BYOF_REPO_MOUNT}\n"
        'ENTRYPOINT ["/bin/sh", "-c", "if [ \\"$#\\" -gt 0 ]; then exec \\"$@\\"; fi; exec /bin/bash", "npa-byof-entrypoint"]\n'
        'CMD ["/bin/bash"]\n'
    )


def _parse_last_json(text: str) -> dict[str, Any] | None:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


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
        _run(
            [
                sys.executable,
                str(ROBOTWIN_IMAGE_SCANNER),
                image,
                "--output",
                str(report_path),
            ],
            capture=True,
            redactions=redactions,
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
) -> dict[str, str]:
    selected_project = authorization.project if authorization is not None else project
    context_redactions = authorization.redactions if authorization is not None else ()
    env = (
        _live_runner_env(selected_project, redactions=context_redactions)
        if context_redactions
        else _live_runner_env(selected_project)
    )
    if authorization is None:
        return env
    env.update(
        {
            "KUBECONFIG": authorization.kubeconfig,
            "KUBECONTEXT": authorization.kubernetes_context,
            "NPA_BYOF_K8S_CONTEXT": authorization.kubernetes_context,
            "NPA_NEBIUS_PROFILE": authorization.profile,
            "NPA_BYOF_PROJECT": authorization.project,
            "NPA_BYOF_ROBOTWIN_RESERVATION_EVIDENCE_SHA256": authorization.context_sha256,
            "NPA_BYOF_ROBOTWIN_IMAGE_SCAN_SHA256": str(scan_evidence["report_sha256"]),
            "NPA_BYOF_ROBOTWIN_IMAGE_SCAN_ARCHIVES": str(
                scan_evidence["archives_scanned"]
            ),
        }
    )
    return env


def _private_runtime_redactions(env: dict[str, str]) -> tuple[str, ...]:
    return tuple(
        value
        for key in (
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "AWS_SECURITY_TOKEN",
            "AWS_ENDPOINT_URL",
            "NEBIUS_S3_ENDPOINT",
            "NPA_S3_BUCKET",
            "S3_BUCKET",
        )
        if (value := env.get(key, "").strip())
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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    redactions: tuple[str, ...] = ()
    try:
        authorization = _load_runtime_authorization(args)
        if authorization is not None:
            redactions = authorization.redactions
            _validate_robotwin_invocation(args)
            _apply_runtime_authorization(args, authorization)
    except Exception as exc:
        payload = {
            "status": "failed",
            "solution_name": args.solution_name,
            "error": _redact_text(str(exc), redactions),
        }
        print(
            json.dumps(_redact_payload(payload, redactions), indent=2, sort_keys=True)
        )
        return 1
    private_source = args.repo_auth == "github"
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
        if authorization is not None:
            _activate_runtime_profile(authorization)
        with ExitStack() as secret_stack:
            source_secrets: RepositorySecretFiles | None = None
            if private_source:
                source_secrets = secret_stack.enter_context(
                    private_repository_secrets(
                        args.repo_url,
                        args.repo_ref,
                        token_env=args.repo_token_env,
                    )
                )
                summary["source_identity"] = {
                    "repository_sha256": source_secrets.repository_sha256,
                    "ref_sha256": source_secrets.ref_sha256,
                }
            redactions = tuple(
                dict.fromkeys(
                    (
                        *redactions,
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
                redactions=redactions,
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
        message = _redact_text(str(exc), redactions)
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
        print(json.dumps(_redact_payload(summary, redactions), indent=2, sort_keys=True))
        hint = str(summary.get("hint") or "").strip()
        if hint:
            print(f"HINT: {hint}", file=sys.stderr)
        return 1


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
                            f"BYOF_SOURCE_VISIBILITY={'private' if source_secrets else 'public'}",
                            "--build-arg",
                            (
                                "BYOF_SOURCE_CACHE_KEY="
                                + (
                                    source_secrets.repository_sha256
                                    + source_secrets.ref_sha256
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
            scan_evidence = _scan_robotwin_image(image, redactions=redactions)
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
                cmd = [
                    sys.executable,
                    str(CONTAINER_VERIFY_RUNNER),
                    "--image",
                    image,
                    "--run-id",
                    args.run_id,
                    "--wait-timeout",
                    str(args.wait_timeout),
                    "--poll-interval",
                    str(args.poll_interval),
                    "--repo-root",
                    BYOF_REPO_MOUNT,
                ]
                if args.smoke_command:
                    cmd.extend(["--smoke-command", args.smoke_command])
                if args.solution_name:
                    cmd.extend(["--solution-name", args.solution_name])
                if args.capability_name:
                    cmd.extend(["--capability-name", args.capability_name])
                if args.smoke_artifact_name:
                    cmd.extend(["--smoke-artifact-name", args.smoke_artifact_name])
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
            if args.output_root:
                cmd.extend(["--output-root", args.output_root])
            if args.sky_bin:
                cmd.extend(["--sky-bin", args.sky_bin])
            if args.config_path:
                cmd.extend(["--config-path", args.config_path])
            if args.cleanup:
                cmd.append("--cleanup")
            live_env = _authorized_live_env(
                authorization, scan_evidence, project=args.project
            )
            runtime_redactions = (
                tuple(
                    dict.fromkeys((*redactions, *_private_runtime_redactions(live_env)))
                )
                if authorization is not None
                else redactions
            )
            run_kwargs = {
                "capture": True,
                "env": live_env,
            }
            if runtime_redactions:
                run_kwargs["redactions"] = runtime_redactions
            run_proc = _run(cmd, **run_kwargs)
            sys.stdout.write(_redact_text(run_proc.stdout, runtime_redactions))
            if run_proc.stderr:
                sys.stderr.write(_redact_text(run_proc.stderr, runtime_redactions))
            parsed_run = _parse_last_json(run_proc.stdout) or {"status": "submitted"}
            summary["run"] = _redact_payload(parsed_run, runtime_redactions)
            if postprocess_key is not None:
                postprocess = run_registered_postprocess(
                    postprocess_key,
                    PostprocessContext(
                        run_prefix_uri=f"{args.output_root.rstrip('/')}/{args.run_id}/",
                        project=args.project or None,
                        wan_acceptance_candidate_image=(
                            args.wan_acceptance_candidate_image.strip()
                        ),
                    ),
                )
                if postprocess is None:
                    raise RuntimeError(
                        f"registered solution {postprocess_key!r} returned no "
                        "verified postprocess result"
                    )
                summary["postprocess"] = postprocess
        else:
            summary["run"] = {"skipped": True}
        summary["status"] = "ok"
        print(json.dumps(_redact_payload(summary, redactions), indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        # Do not retain an unsanitized exception as ``__cause__``: callers may
        # serialize the exception chain even though the top-level message is safe.
        raise RuntimeError(_redact_text(str(exc), redactions)) from None


if __name__ == "__main__":
    raise SystemExit(main())
