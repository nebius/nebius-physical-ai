"""SONIC SkyPilot workflow materialization and submission helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence

import yaml

from npa.cluster.config import DEFAULT_REGION, SUPPORTED_REGIONS
from npa.deploy.images import container_image_for_tool, sonic_image_entry
from npa.orchestration.skypilot.controller import DEFAULT_CONTROLLER_BACKEND, ControllerBackend
from npa.orchestration.skypilot.workflow import WorkflowResult
from npa.orchestration.skypilot.workflow import submit_workflow as _submit_skypilot_workflow


DEFAULT_S3_ENDPOINT = "https://storage.eu-north1.nebius.cloud"
DEFAULT_AWS_PROFILE = "nebius"
DEFAULT_GPU_TARGET = "l40s"
DEFAULT_SONIC_WORKFLOW_PREFIX = "sonic-locomotion"
UNRESOLVED_SUBMIT_TOKENS = (
    "<your-",
    "<sonic-image-tag>",
    "<npa-image-tag>",
    "${NPA_WORKBENCH_IMAGE}",
    "${NPA_RETARGETING_IMAGE}",
    "example.invalid",
)
SKYPILOT_DOCKER_USERNAME = "SKYPILOT_DOCKER_USERNAME"
SKYPILOT_DOCKER_PASSWORD = "SKYPILOT_DOCKER_PASSWORD"
SKYPILOT_DOCKER_SERVER = "SKYPILOT_DOCKER_SERVER"
DEFAULT_NEBIUS_REGISTRY_USERNAME = "iam"
NEBIUS_REGISTRY_SERVER_SUFFIX = ".nebius.cloud"
REGISTRY_AUTH_USERNAME_ENVS = ("NPA_REGISTRY_USERNAME", SKYPILOT_DOCKER_USERNAME)
REGISTRY_AUTH_PASSWORD_ENVS = ("NPA_REGISTRY_PASSWORD", SKYPILOT_DOCKER_PASSWORD)
REGISTRY_AUTH_SERVER_ENVS = ("NPA_REGISTRY_SERVER", SKYPILOT_DOCKER_SERVER)


@dataclass(frozen=True)
class SonicWorkflowPlan:
    """Resolved values used to submit a SONIC workflow YAML."""

    yaml_text: str = field(repr=False)
    run_id: str
    policy_image: str
    npa_image: str
    retargeting_image: str
    gpu_target: str
    image_variant: str
    aws_profile: str
    s3_endpoint: str
    s3_bucket: str
    s3_prefix: str
    accelerators: str
    cloud: str
    region: str
    use_spot: bool | None = None
    registry_auth_server: str = ""
    registry_auth_username: str = ""
    registry_auth_source: str = ""


@dataclass(frozen=True)
class _RegistryAuthConfig:
    username: str
    password: str
    server: str
    source: str


def materialize_sonic_workflow(
    yaml_path: Path,
    *,
    run_id: str,
    registry: str = "",
    image: str = "",
    npa_image: str = "",
    registry_auth: bool = True,
    registry_username: str = "",
    registry_password: str = "",
    registry_server: str = "",
    gpu_target: str = DEFAULT_GPU_TARGET,
    image_variant: str = "",
    aws_profile: str = "",
    s3_endpoint: str = "",
    s3_bucket: str = "",
    s3_prefix: str = "",
    accelerators: str = "",
    cloud: str = "",
    region: str = "",
    use_spot: bool | None = None,
    env_overrides: dict[str, str] | None = None,
) -> SonicWorkflowPlan:
    """Return a concrete SONIC workflow YAML with no submit-time env indirection."""

    docs = _load_yaml_documents(yaml_path)
    resolved_run_id = run_id or _default_run_id(yaml_path)
    resolved_gpu_target = (gpu_target or DEFAULT_GPU_TARGET).strip()
    entry = sonic_image_entry(
        gpu_target=resolved_gpu_target or None,
        image_variant=image_variant or None,
    )
    resolved_variant = str(entry["id"])
    resolved_policy_image = image or container_image_for_tool(
        "sonic",
        registry=registry or None,
        gpu_target=resolved_gpu_target or None,
        image_variant=resolved_variant,
    )
    resolved_retargeting_image = container_image_for_tool("retargeting", registry=registry or None)
    resolved_npa_image = npa_image
    resolved_aws_profile = aws_profile or os.environ.get("AWS_PROFILE", "") or DEFAULT_AWS_PROFILE
    resolved_endpoint = _resolve_s3_endpoint(s3_endpoint)
    resolved_bucket = s3_bucket or os.environ.get("NPA_S3_BUCKET", "")
    resolved_prefix = _resolve_s3_prefix(s3_prefix, resolved_run_id)
    resolved_accelerators = accelerators or _default_accelerators(resolved_gpu_target)
    resolved_cloud = cloud or _default_cloud(resolved_gpu_target)
    resolved_region = _resolve_region(region)
    resolved_registry_auth = _resolve_registry_auth(
        enabled=registry_auth and resolved_cloud.strip().lower() != "kubernetes",
        username=registry_username,
        password=registry_password,
        server=registry_server,
        policy_image=resolved_policy_image,
        npa_image=resolved_npa_image,
    )

    replacements = {
        "sonic-locomotion/<run-id>/": resolved_prefix,
        f"sonic-locomotion/{resolved_run_id}/": resolved_prefix,
        "<run-id>": resolved_run_id,
        "${NPA_PIPELINE_RUN_ID}": resolved_run_id,
        "${POLICY_IMAGE}": resolved_policy_image,
        "${S3_ENDPOINT_URL}": resolved_endpoint,
    }
    if resolved_bucket:
        replacements["<your-bucket-name>"] = resolved_bucket

    materialized = [_replace_strings(doc, replacements) for doc in docs]
    for doc in materialized:
        if isinstance(doc, dict):
            _materialize_task_doc(
                doc,
                run_id=resolved_run_id,
                policy_image=resolved_policy_image,
                npa_image=resolved_npa_image,
                retargeting_image=resolved_retargeting_image,
                gpu_target=resolved_gpu_target,
                image_variant=resolved_variant,
                aws_profile=resolved_aws_profile,
                s3_endpoint=resolved_endpoint,
                s3_bucket=resolved_bucket,
                s3_prefix=resolved_prefix,
                accelerators=resolved_accelerators,
                cloud=resolved_cloud,
                region=resolved_region,
                use_spot=use_spot,
                registry_auth=resolved_registry_auth,
                env_overrides=env_overrides or {},
            )

    yaml_text = yaml.safe_dump_all(materialized, sort_keys=False)
    return SonicWorkflowPlan(
        yaml_text=yaml_text,
        run_id=resolved_run_id,
        policy_image=resolved_policy_image,
        npa_image=resolved_npa_image,
        retargeting_image=resolved_retargeting_image,
        gpu_target=resolved_gpu_target,
        image_variant=resolved_variant,
        aws_profile=resolved_aws_profile,
        s3_endpoint=resolved_endpoint,
        s3_bucket=resolved_bucket,
        s3_prefix=resolved_prefix,
        accelerators=resolved_accelerators,
        cloud=resolved_cloud,
        region=resolved_region,
        use_spot=use_spot,
        registry_auth_server=resolved_registry_auth.server if resolved_registry_auth else "",
        registry_auth_username=resolved_registry_auth.username if resolved_registry_auth else "",
        registry_auth_source=resolved_registry_auth.source if resolved_registry_auth else "",
    )


def submit_sonic_workflow(
    yaml_path: Path,
    *,
    run_id: str = "",
    registry: str = "",
    image: str = "",
    npa_image: str = "",
    registry_auth: bool = True,
    registry_username: str = "",
    registry_password: str = "",
    registry_server: str = "",
    gpu_target: str = DEFAULT_GPU_TARGET,
    image_variant: str = "",
    aws_profile: str = "",
    s3_endpoint: str = "",
    s3_bucket: str = "",
    s3_prefix: str = "",
    accelerators: str = "",
    cloud: str = "",
    region: str = "",
    use_spot: bool | None = None,
    env_overrides: dict[str, str] | None = None,
    isolated_config_dir: Path | None = None,
    config_path: Path | None = None,
    sky_bin: str | None = None,
    controller_backend: ControllerBackend = DEFAULT_CONTROLLER_BACKEND,
    secret_envs: Sequence[str] | None = None,
    require_controller_up: bool = False,
    timeout: int = 1800,
) -> WorkflowResult:
    """Materialize and submit a SONIC SkyPilot workflow."""

    plan = materialize_sonic_workflow(
        yaml_path,
        run_id=run_id,
        registry=registry,
        image=image,
        npa_image=npa_image,
        registry_auth=registry_auth,
        registry_username=registry_username,
        registry_password=registry_password,
        registry_server=registry_server,
        gpu_target=gpu_target,
        image_variant=image_variant,
        aws_profile=aws_profile,
        s3_endpoint=s3_endpoint,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        accelerators=accelerators,
        cloud=cloud,
        region=region,
        use_spot=use_spot,
        env_overrides=env_overrides,
    )
    unresolved = unresolved_submit_placeholders(plan.yaml_text)
    if unresolved:
        raise ValueError(
            "SONIC workflow still has unresolved submit placeholders: "
            + ", ".join(unresolved)
        )
    with tempfile.TemporaryDirectory(prefix="npa-sonic-workflow-") as tmp:
        prepared = Path(tmp) / Path(yaml_path).name
        prepared.write_text(plan.yaml_text, encoding="utf-8")
        return _submit_skypilot_workflow(
            prepared,
            plan.run_id,
            isolated_config_dir=isolated_config_dir,
            config_path=config_path,
            sky_bin=sky_bin,
            controller_backend=controller_backend,
            secret_envs=secret_envs,
            require_controller_up=require_controller_up,
            timeout=timeout,
        )


def _load_yaml_documents(path: Path) -> list[Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [doc for doc in yaml.safe_load_all(handle) if doc is not None]


def unresolved_submit_placeholders(yaml_text: str) -> list[str]:
    """Return placeholder tokens that should never reach SkyPilot submit."""

    return [token for token in UNRESOLVED_SUBMIT_TOKENS if token in yaml_text]


def _default_run_id(yaml_path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", Path(yaml_path).stem).strip("-")
    return stem or "sonic-workflow"


def _resolve_s3_endpoint(explicit: str) -> str:
    return (
        explicit
        or os.environ.get("S3_ENDPOINT_URL")
        or os.environ.get("AWS_ENDPOINT_URL")
        or os.environ.get("NEBIUS_S3_ENDPOINT")
        or DEFAULT_S3_ENDPOINT
    )


def _resolve_s3_prefix(explicit: str, run_id: str) -> str:
    prefix = explicit.strip("/") if explicit else f"{DEFAULT_SONIC_WORKFLOW_PREFIX}/{run_id}"
    return prefix.rstrip("/") + "/"


def _default_accelerators(gpu_target: str) -> str:
    normalized = gpu_target.strip().lower().replace("_", "-")
    if "rtx" in normalized or "blackwell" in normalized or "sm-120" in normalized:
        return "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1"
    if "h200" in normalized:
        return "H200:1"
    if "h100" in normalized:
        return "H100:1"
    if "b200" in normalized:
        return "B200:1"
    return "L40S:1"


def _default_cloud(gpu_target: str) -> str:
    normalized = gpu_target.strip().lower().replace("_", "-")
    if "rtx" in normalized or "blackwell" in normalized or "sm-120" in normalized:
        return "kubernetes"
    return "nebius"


def _resolve_region(region: str) -> str:
    resolved = (region or os.environ.get("NPA_SKYPILOT_REGION", "") or DEFAULT_REGION).strip()
    if resolved == "me-west1":
        raise ValueError("SONIC H100/H200 workflows explicitly exclude me-west1")
    if resolved and resolved not in SUPPORTED_REGIONS:
        choices = ", ".join(sorted(SUPPORTED_REGIONS))
        raise ValueError(f"Unsupported SONIC SkyPilot region {resolved!r}; choose one of: {choices}")
    return resolved


def _default_cpus(gpu_target: str) -> int:
    normalized = gpu_target.strip().lower().replace("_", "-")
    if "b200" in normalized:
        return 20
    return 16


def _default_memory(gpu_target: str) -> int:
    normalized = gpu_target.strip().lower().replace("_", "-")
    if "h100" in normalized or "h200" in normalized:
        return 200
    if "b200" in normalized:
        return 224
    return 64


def _replace_strings(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [_replace_strings(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace_strings(item, replacements) for key, item in value.items()}
    return value


def _materialize_task_doc(
    doc: dict[str, Any],
    *,
    run_id: str,
    policy_image: str,
    npa_image: str,
    retargeting_image: str,
    gpu_target: str,
    image_variant: str,
    aws_profile: str,
    s3_endpoint: str,
    s3_bucket: str,
    s3_prefix: str,
    accelerators: str,
    cloud: str,
    region: str,
    use_spot: bool | None,
    registry_auth: _RegistryAuthConfig | None,
    env_overrides: dict[str, str],
) -> None:
    envs = doc.get("envs")
    resources = doc.get("resources")
    if not isinstance(envs, dict) or not isinstance(resources, dict):
        return

    if "NPA_RETARGETING_IMAGE" in envs:
        resources["image_id"] = f"docker:{retargeting_image}"
        envs["NPA_RETARGETING_IMAGE"] = retargeting_image

    payload_mode = str(
        env_overrides.get("SONIC_PAYLOAD_MODE", envs.get("SONIC_PAYLOAD_MODE", "direct"))
    ).strip().lower()
    has_sonic_env = _has_sonic_env(doc)
    uses_sonic_runtime_image = _uses_sonic_runtime_image(doc)
    if uses_sonic_runtime_image:
        resources["cloud"] = cloud
        if payload_mode == "docker":
            resources.pop("image_id", None)
        else:
            resources["image_id"] = f"docker:{policy_image}"
        if resources.get("accelerators"):
            resources["accelerators"] = accelerators
        if cloud.strip().lower() != "kubernetes":
            if region:
                resources["region"] = region
            resources["cpus"] = _default_cpus(gpu_target)
            resources["memory"] = _default_memory(gpu_target)
            if use_spot is not None:
                resources["use_spot"] = use_spot
    elif npa_image and _looks_like_npa_helper_image(doc):
        resources["image_id"] = f"docker:{npa_image}"
        if "NPA_WORKBENCH_IMAGE" in envs:
            envs["NPA_WORKBENCH_IMAGE"] = npa_image

    for key in ("POLICY_IMAGE", "CONTAINER_IMAGE", "SONIC_EVAL_CONTAINER_IMAGE"):
        if key in envs and has_sonic_env:
            envs[key] = policy_image
    for key in ("SONIC_GPU_TYPE", "SONIC_GPU_TARGET", "CONTAINER_GPU_TARGET", "SONIC_EVAL_CONTAINER_GPU_TARGET"):
        if key in envs and has_sonic_env:
            envs[key] = gpu_target
    for key in ("SONIC_IMAGE_VARIANT", "CONTAINER_IMAGE_VARIANT", "SONIC_EVAL_CONTAINER_IMAGE_VARIANT"):
        if key in envs and has_sonic_env:
            envs[key] = image_variant
    if "AWS_PROFILE" in envs:
        envs["AWS_PROFILE"] = aws_profile
    for key in ("S3_ENDPOINT_URL", "AWS_ENDPOINT_URL", "NEBIUS_S3_ENDPOINT"):
        if key in envs:
            envs[key] = s3_endpoint
    if "S3_BUCKET" in envs and s3_bucket:
        envs["S3_BUCKET"] = s3_bucket
    if "NPA_PIPELINE_RUN_ID" in envs:
        envs["NPA_PIPELINE_RUN_ID"] = run_id
    if "SONIC_OUTPUT_PREFIX" in envs:
        current_prefix = str(envs["SONIC_OUTPUT_PREFIX"]).strip()
        if not current_prefix or not current_prefix.strip("/").startswith(s3_prefix.strip("/")):
            envs["SONIC_OUTPUT_PREFIX"] = s3_prefix
    for key, value in env_overrides.items():
        if key in envs:
            envs[key] = value
    if registry_auth and _uses_registry_auth_target(doc, registry_auth.server, policy_image):
        envs[SKYPILOT_DOCKER_USERNAME] = registry_auth.username
        envs[SKYPILOT_DOCKER_PASSWORD] = registry_auth.password
        envs[SKYPILOT_DOCKER_SERVER] = registry_auth.server


def _has_sonic_env(doc: dict[str, Any]) -> bool:
    envs = doc.get("envs")
    return isinstance(envs, dict) and (
        "SONIC_GPU_TYPE" in envs
        or "SONIC_GPU_TARGET" in envs
        or "POLICY_IMAGE" in envs
        or "CONTAINER_IMAGE" in envs
        or "SONIC_EVAL_CONTAINER_IMAGE" in envs
    )


def _uses_sonic_runtime_image(doc: dict[str, Any]) -> bool:
    envs = doc.get("envs")
    resources = doc.get("resources")
    image_id = str(resources.get("image_id", "")) if isinstance(resources, dict) else ""
    return isinstance(envs, dict) and ("npa-sonic" in image_id or "POLICY_IMAGE" in envs)


def _looks_like_npa_helper_image(doc: dict[str, Any]) -> bool:
    envs = doc.get("envs")
    resources = doc.get("resources")
    image_id = str(resources.get("image_id", "")) if isinstance(resources, dict) else ""
    return (
        "/npa:" in image_id
        or image_id.endswith("/npa:<npa-image-tag>")
        or "${NPA_WORKBENCH_IMAGE}" in image_id
        or (isinstance(envs, dict) and "NPA_WORKBENCH_IMAGE" in envs)
    )


def _resolve_registry_auth(
    *,
    enabled: bool,
    username: str,
    password: str,
    server: str,
    policy_image: str,
    npa_image: str,
) -> _RegistryAuthConfig | None:
    if not enabled:
        return None

    explicit = any(value.strip() for value in (username, password, server))
    env_username = _first_env(REGISTRY_AUTH_USERNAME_ENVS)
    env_password = _first_env(REGISTRY_AUTH_PASSWORD_ENVS)
    env_server = _first_env(REGISTRY_AUTH_SERVER_ENVS)
    env_provided = any(value for value in (env_username, env_password, env_server))

    resolved_server = _normalize_registry_server(
        server or env_server or _registry_server_for_images(policy_image, npa_image)
    )
    resolved_username = username or env_username
    resolved_password = password or env_password
    source = "explicit" if explicit else "env" if env_provided else ""

    if resolved_server and _is_nebius_registry_server(resolved_server):
        if not resolved_username:
            resolved_username = DEFAULT_NEBIUS_REGISTRY_USERNAME
        if not resolved_password:
            resolved_password = _mint_nebius_registry_token()
            source = "nebius-iam-token"
    elif explicit or env_provided:
        source = source or "explicit"
    else:
        return None

    missing = [
        name
        for name, value in (
            ("username", resolved_username),
            ("password", resolved_password),
            ("server", resolved_server),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            "Registry auth requires username, password, and server; missing "
            + ", ".join(missing)
        )

    return _RegistryAuthConfig(
        username=resolved_username,
        password=resolved_password,
        server=resolved_server,
        source=source or "explicit",
    )


def _first_env(names: Sequence[str]) -> str:
    for name in names:
        value = os.environ.get(name, "")
        if value:
            return value
    return ""


def _mint_nebius_registry_token() -> str:
    cli = os.environ.get("NEBIUS_CLI", "nebius")
    try:
        result = subprocess.run(
            [cli, "iam", "get-access-token"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(
            "Could not mint Nebius registry token with `nebius iam get-access-token`"
        ) from exc

    token = result.stdout.strip()
    if result.returncode != 0 or not token:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise ValueError(
            "Could not mint Nebius registry token with `nebius iam get-access-token`: "
            + detail
        )
    return token


def _registry_server_for_images(*images: str) -> str:
    for image in images:
        server = _registry_server_from_image(image)
        if server:
            return server
    return ""


def _registry_server_from_image(image: str) -> str:
    image_ref = image.removeprefix("docker:").strip()
    if "/" not in image_ref:
        return ""
    candidate = image_ref.split("/", 1)[0]
    if "." in candidate or ":" in candidate or candidate == "localhost":
        return _normalize_registry_server(candidate)
    return ""


def _normalize_registry_server(server: str) -> str:
    normalized = server.strip()
    normalized = normalized.removeprefix("https://").removeprefix("http://")
    return normalized.rstrip("/")


def _is_nebius_registry_server(server: str) -> bool:
    normalized = _normalize_registry_server(server)
    return normalized.startswith("cr.") and normalized.endswith(NEBIUS_REGISTRY_SERVER_SUFFIX)


def _uses_registry_auth_target(doc: dict[str, Any], server: str, policy_image: str) -> bool:
    resources = doc.get("resources")
    envs = doc.get("envs")
    if not isinstance(resources, dict) or not isinstance(envs, dict):
        return False
    if str(resources.get("cloud", "")).strip().lower() == "kubernetes":
        return False
    normalized_server = _normalize_registry_server(server)
    image_server = _registry_server_from_image(str(resources.get("image_id", "")))
    policy_server = _registry_server_from_image(str(envs.get("POLICY_IMAGE", policy_image)))
    return normalized_server in {image_server, policy_server}
