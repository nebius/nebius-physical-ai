"""Shared Workbench container image naming."""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
import json
import os
from pathlib import Path
from typing import Any

# The registry ID must never be hardcoded to a real Nebius identifier in the
# committed tree (see SECURITY.md and the repo rules). The committed default is
# an inert placeholder; operators supply the real registry via NPA_REGISTRY_ID /
# NPA_REGISTRY env vars or `container_registry` in ~/.npa/config.yaml, and
# resolve_workflow_image() substitutes the placeholder at submit time.
DEFAULT_CONTAINER_REGISTRY_ID = "<your-registry-id>"
DEFAULT_CONTAINER_REGISTRY = f"cr.eu-north1.nebius.cloud/{DEFAULT_CONTAINER_REGISTRY_ID}"
DEFAULT_VLM_IMAGE_ENV = "NPA_VLM_IMAGE"
DEFAULT_WORKBENCH_IMAGE_ENV = "NPA_WORKBENCH_IMAGE"
SONIC_IMAGE_MANIFEST_RESOURCE = "sonic_image_manifest.json"

CONTAINER_IMAGE_NAMES = {
    "lerobot": "npa-lerobot",
    "lerobot-policy": "npa-lerobot-policy",
    "genesis": "npa-genesis",
    "isaac-lab": "npa-isaac-lab",
    "cosmos": "npa-cosmos",
    "cosmos2-transfer": "npa-cosmos2-transfer",
    "cosmos3-reason": "npa-cosmos3-reason",
    "groot": "npa-groot",
    "fiftyone": "npa-fiftyone",
    "sonic": "npa-sonic",
    "retargeting": "npa-retargeting",
    "sim2real-envgen": "npa-sim2real-envgen",
    "sim2real-reference-policy": "npa-sim2real-reference-policy",
    "lerobot-vlm-rl": "npa-lerobot-vlm-rl",
    "sim2real-eval": "npa-sim2real-eval",
    "lancedb": "npa-lancedb",
    "detection-training": "npa-detection-training",
}

SUPPORTED_TOOL_VERSIONS = {
    "lerobot": "0.5.1",
    "lerobot-policy": "0.1.0",
    "genesis": "0.4.6",
    "isaac-lab": "2.3.2.post1",
    "cosmos": "1.0.9",
    "cosmos2-transfer": "2.5.0",
    "cosmos3-reason": "3.0.1-genuine-sm120",
    "groot": "0.1.0",
    "fiftyone": "1.15.0",
    "sonic": "0.1.2",
    "retargeting": "0.1.0",
    "sim2real-envgen": "0.1.1",
    "sim2real-reference-policy": "0.1.1",
    "lerobot-vlm-rl": "0.1.0",
    "sim2real-eval": "0.1.1-genuine-sm120",
    "lancedb": "0.30.2",
    "detection-training": "bdd100k-real-labelmap-eval-w9-registry-fix-20260519T214847Z",
    "nebius-cli": "0.12.192",
    "terraform": "~> 0.5.201",
    "terraform-cli": "1.13.3",
}


@lru_cache(maxsize=1)
def sonic_image_manifest() -> dict[str, Any]:
    """Return the packaged SONIC image compatibility manifest."""

    text = resources.files(__package__).joinpath(SONIC_IMAGE_MANIFEST_RESOURCE).read_text(
        encoding="utf-8"
    )
    payload = json.loads(text)
    if payload.get("format") != "npa_sonic_image_manifest_v1":
        raise RuntimeError("Unsupported SONIC image manifest format")
    return payload


def sonic_image_variants() -> dict[str, dict[str, Any]]:
    """Return SONIC image manifest entries by variant id."""

    variants: dict[str, dict[str, Any]] = {}
    for item in sonic_image_manifest().get("images", []):
        if not isinstance(item, dict):
            continue
        variant_id = str(item.get("id", ""))
        if variant_id:
            variants[variant_id] = item
    return variants


def supported_tool_version(tool: str) -> str:
    if tool == "sonic":
        return str(_default_sonic_image()["tag"])

    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    for directory in Path(__file__).resolve().parents:
        pyproject = directory / "pyproject.toml"
        if pyproject.is_file():
            with pyproject.open("rb") as handle:
                data = tomllib.load(handle)
            return str(data["tool"]["npa"]["supported-tools"][tool])
    try:
        return SUPPORTED_TOOL_VERSIONS[tool]
    except KeyError as exc:
        raise RuntimeError(f"Could not find supported version for tool: {tool}") from exc


def sonic_image_variant_for_gpu(gpu_target: str | None = None) -> str:
    """Return the SONIC image variant id for a GPU or provider target."""

    manifest = sonic_image_manifest()
    default = str(manifest.get("default_variant", "sonic-l40s-baked"))
    normalized = _normalize_gpu_target(gpu_target)
    if not normalized:
        return default
    for rule in manifest.get("gpu_selection", []):
        if not isinstance(rule, dict):
            continue
        variant = str(rule.get("variant", ""))
        for match in rule.get("matches", []):
            if str(match).lower() in normalized:
                return variant
    return default


def sonic_image_entry(
    *,
    gpu_target: str | None = None,
    image_variant: str | None = None,
) -> dict[str, Any]:
    """Return the SONIC manifest entry selected by variant or GPU target."""

    variants = sonic_image_variants()
    if image_variant:
        resolved = _normalize_sonic_variant(image_variant, variants)
    else:
        resolved = sonic_image_variant_for_gpu(gpu_target)
    try:
        return variants[resolved]
    except KeyError as exc:
        choices = ", ".join(sorted(variants))
        raise ValueError(f"Unknown SONIC image variant {resolved!r}; choose one of: {choices}") from exc


def container_image_for_tool(
    tool: str,
    *,
    registry: str | None = None,
    tag: str | None = None,
    gpu_target: str | None = None,
    image_variant: str | None = None,
) -> str:
    """Return the fully qualified image ref for a Workbench tool."""
    if tool == "sonic":
        entry = sonic_image_entry(gpu_target=gpu_target, image_variant=image_variant)
        image_name = str(entry["name"])
        resolved_tag = tag or str(entry["tag"])
    else:
        if image_variant:
            raise ValueError(f"Image variants are only defined for SONIC, got tool={tool!r}")
        image_name = CONTAINER_IMAGE_NAMES[tool]
        resolved_tag = tag or supported_tool_version(tool)
    resolved_registry = registry or _registry_from_env() or DEFAULT_CONTAINER_REGISTRY
    return f"{resolved_registry.rstrip('/')}/{image_name}:{resolved_tag}"


def _registry_from_env() -> str:
    """Resolve a container registry from environment without hardcoding an ID."""
    explicit = os.environ.get("NPA_REGISTRY", "").strip()
    if explicit:
        return explicit
    registry_id = os.environ.get("NPA_REGISTRY_ID", "").strip()
    if registry_id:
        return f"cr.eu-north1.nebius.cloud/{registry_id}"
    return ""


def default_vlm_image(*, registry: str | None = None) -> str:
    """Return the default self-hosted VLM workflow image, honoring BYO override."""

    override = os.environ.get(DEFAULT_VLM_IMAGE_ENV, "").strip()
    if override:
        return override
    return container_image_for_tool("cosmos", registry=registry)


def default_workbench_image(*, registry: str | None = None) -> str:
    """Return the default generic Workbench workflow image, honoring BYO override."""

    override = os.environ.get(DEFAULT_WORKBENCH_IMAGE_ENV, "").strip()
    if override:
        return override
    return container_image_for_tool("genesis", registry=registry)


def _default_sonic_image() -> dict[str, Any]:
    return sonic_image_entry(image_variant=str(sonic_image_manifest().get("default_variant", "")))


def _normalize_gpu_target(gpu_target: str | None) -> str:
    return (gpu_target or "").strip().lower().replace("_", "-")


def _normalize_sonic_variant(image_variant: str, variants: dict[str, dict[str, Any]]) -> str:
    normalized = image_variant.strip().lower().replace("_", "-")
    aliases = {
        "baked": "sonic-l40s-baked",
        "l40s": "sonic-l40s-baked",
        "l40s-baked": "sonic-l40s-baked",
        "host-mounted": "sonic-k8s-host-mounted",
        "host": "sonic-k8s-host-mounted",
        "k8s": "sonic-k8s-host-mounted",
        "rtx": "sonic-k8s-host-mounted",
        "rtxpro": "sonic-k8s-host-mounted",
        "rtx-pro": "sonic-k8s-host-mounted",
        "rtx6000": "sonic-k8s-host-mounted",
        "rtx-pro-6000": "sonic-k8s-host-mounted",
        "mujoco": "sonic-mujoco-h100-mvp",
        "h100": "sonic-mujoco-h100-mvp",
        "h200": "sonic-mujoco-h100-mvp",
        "sonic-mujoco": "sonic-mujoco-h100-mvp",
        "mvp": "sonic-mujoco-h100-mvp",
    }
    resolved = aliases.get(normalized, normalized)
    if resolved not in variants:
        choices = ", ".join(sorted(variants))
        raise ValueError(f"Unknown SONIC image variant {image_variant!r}; choose one of: {choices}")
    return resolved
