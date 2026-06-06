"""SONIC SkyPilot workflow materialization and submission helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Sequence

import yaml

from npa.deploy.images import container_image_for_tool, sonic_image_entry
from npa.orchestration.skypilot.controller import DEFAULT_CONTROLLER_BACKEND, ControllerBackend
from npa.orchestration.skypilot.workflow import WorkflowResult
from npa.orchestration.skypilot.workflow import submit_workflow as _submit_skypilot_workflow


DEFAULT_S3_ENDPOINT = "https://storage.eu-north1.nebius.cloud"
DEFAULT_GPU_TARGET = "l40s"
DEFAULT_SONIC_WORKFLOW_PREFIX = "sonic-locomotion"
UNRESOLVED_SUBMIT_TOKENS = ("<your-", "<sonic-image-tag>", "<npa-image-tag>", "example.invalid")


@dataclass(frozen=True)
class SonicWorkflowPlan:
    """Resolved values used to submit a SONIC workflow YAML."""

    yaml_text: str
    run_id: str
    policy_image: str
    npa_image: str
    gpu_target: str
    image_variant: str
    s3_endpoint: str
    s3_bucket: str
    s3_prefix: str
    accelerators: str
    cloud: str


def materialize_sonic_workflow(
    yaml_path: Path,
    *,
    run_id: str,
    registry: str = "",
    image: str = "",
    npa_image: str = "",
    gpu_target: str = DEFAULT_GPU_TARGET,
    image_variant: str = "",
    s3_endpoint: str = "",
    s3_bucket: str = "",
    s3_prefix: str = "",
    accelerators: str = "",
    cloud: str = "",
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
    resolved_npa_image = npa_image
    resolved_endpoint = _resolve_s3_endpoint(s3_endpoint)
    resolved_bucket = s3_bucket or os.environ.get("NPA_S3_BUCKET", "")
    resolved_prefix = _resolve_s3_prefix(s3_prefix, resolved_run_id)
    resolved_accelerators = accelerators or _default_accelerators(resolved_gpu_target)
    resolved_cloud = cloud or _default_cloud(resolved_gpu_target)

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
                gpu_target=resolved_gpu_target,
                image_variant=resolved_variant,
                s3_endpoint=resolved_endpoint,
                s3_bucket=resolved_bucket,
                s3_prefix=resolved_prefix,
                accelerators=resolved_accelerators,
                cloud=resolved_cloud,
                env_overrides=env_overrides or {},
            )

    yaml_text = yaml.safe_dump_all(materialized, sort_keys=False)
    return SonicWorkflowPlan(
        yaml_text=yaml_text,
        run_id=resolved_run_id,
        policy_image=resolved_policy_image,
        npa_image=resolved_npa_image,
        gpu_target=resolved_gpu_target,
        image_variant=resolved_variant,
        s3_endpoint=resolved_endpoint,
        s3_bucket=resolved_bucket,
        s3_prefix=resolved_prefix,
        accelerators=resolved_accelerators,
        cloud=resolved_cloud,
    )


def submit_sonic_workflow(
    yaml_path: Path,
    *,
    run_id: str = "",
    registry: str = "",
    image: str = "",
    npa_image: str = "",
    gpu_target: str = DEFAULT_GPU_TARGET,
    image_variant: str = "",
    s3_endpoint: str = "",
    s3_bucket: str = "",
    s3_prefix: str = "",
    accelerators: str = "",
    cloud: str = "",
    env_overrides: dict[str, str] | None = None,
    isolated_config_dir: Path | None = None,
    config_path: Path | None = None,
    sky_bin: str | None = None,
    controller_backend: ControllerBackend = DEFAULT_CONTROLLER_BACKEND,
    secret_envs: Sequence[str] | None = None,
    timeout: int = 1800,
) -> WorkflowResult:
    """Materialize and submit a SONIC SkyPilot workflow."""

    plan = materialize_sonic_workflow(
        yaml_path,
        run_id=run_id,
        registry=registry,
        image=image,
        npa_image=npa_image,
        gpu_target=gpu_target,
        image_variant=image_variant,
        s3_endpoint=s3_endpoint,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        accelerators=accelerators,
        cloud=cloud,
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
    return "L40S:1"


def _default_cloud(gpu_target: str) -> str:
    normalized = gpu_target.strip().lower().replace("_", "-")
    if "rtx" in normalized or "blackwell" in normalized or "sm-120" in normalized:
        return "kubernetes"
    return "nebius"


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
    gpu_target: str,
    image_variant: str,
    s3_endpoint: str,
    s3_bucket: str,
    s3_prefix: str,
    accelerators: str,
    cloud: str,
    env_overrides: dict[str, str],
) -> None:
    envs = doc.get("envs")
    resources = doc.get("resources")
    if not isinstance(envs, dict) or not isinstance(resources, dict):
        return

    payload_mode = str(
        env_overrides.get("SONIC_PAYLOAD_MODE", envs.get("SONIC_PAYLOAD_MODE", "direct"))
    ).strip().lower()
    has_sonic_env = _has_sonic_env(doc)
    uses_sonic_runtime_image = _uses_sonic_runtime_image(doc)
    image_id = str(resources.get("image_id", ""))
    if uses_sonic_runtime_image:
        resources["cloud"] = cloud
        if payload_mode == "docker":
            resources.pop("image_id", None)
        else:
            resources["image_id"] = f"docker:{policy_image}"
        if resources.get("accelerators"):
            resources["accelerators"] = accelerators
    elif npa_image and _looks_like_npa_helper_image(image_id):
        resources["image_id"] = f"docker:{npa_image}"

    for key in ("POLICY_IMAGE", "CONTAINER_IMAGE", "SONIC_EVAL_CONTAINER_IMAGE"):
        if key in envs and has_sonic_env:
            envs[key] = policy_image
    for key in ("SONIC_GPU_TYPE", "SONIC_GPU_TARGET", "CONTAINER_GPU_TARGET", "SONIC_EVAL_CONTAINER_GPU_TARGET"):
        if key in envs and has_sonic_env:
            envs[key] = gpu_target
    for key in ("SONIC_IMAGE_VARIANT", "CONTAINER_IMAGE_VARIANT", "SONIC_EVAL_CONTAINER_IMAGE_VARIANT"):
        if key in envs and has_sonic_env:
            envs[key] = image_variant
    for key in ("S3_ENDPOINT_URL", "AWS_ENDPOINT_URL", "NEBIUS_S3_ENDPOINT"):
        if key in envs:
            envs[key] = s3_endpoint
    if "S3_BUCKET" in envs and s3_bucket:
        envs["S3_BUCKET"] = s3_bucket
    if "NPA_PIPELINE_RUN_ID" in envs:
        envs["NPA_PIPELINE_RUN_ID"] = run_id
    if "SONIC_OUTPUT_PREFIX" in envs:
        envs["SONIC_OUTPUT_PREFIX"] = s3_prefix
    for key, value in env_overrides.items():
        if key in envs:
            envs[key] = value


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


def _looks_like_npa_helper_image(image_id: str) -> bool:
    return "/npa:" in image_id or image_id.endswith("/npa:<npa-image-tag>")
