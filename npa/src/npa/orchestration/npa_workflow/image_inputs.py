"""Validate exact operator image inputs before planning or executing a workflow."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

from npa.orchestration.npa_workflow.errors import NpaWorkflowError


# Match distribution/reference path and domain components, including repeated
# dashes, double underscores and bracketed IPv6 authorities.
# https://github.com/distribution/reference/blob/main/regexp.go
_REPOSITORY = r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*"
_DOMAIN_COMPONENT = r"(?:[a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9])"
_DOMAIN = rf"{_DOMAIN_COMPONENT}(?:\.{_DOMAIN_COMPONENT})*"
_QUALIFIED_DOMAIN = rf"{_DOMAIN_COMPONENT}(?:\.{_DOMAIN_COMPONENT})+"
_IPV6 = r"\[[a-fA-F0-9]*:[a-fA-F0-9:]+\]"
_REGISTRY = (
    rf"(?:(?:localhost|{_QUALIFIED_DOMAIN}|{_IPV6})(?::[0-9]+)?|{_DOMAIN}:[0-9]+)"
)
# Shared by exact provenance consumers, including their serialized field schema.
IMMUTABLE_IMAGE_REFERENCE_PATTERN = (
    rf"{_REGISTRY}/(?:{_REPOSITORY}/)*{_REPOSITORY}@sha256:[0-9a-f]{{64}}"
)
_DIGEST_REFERENCE = re.compile(IMMUTABLE_IMAGE_REFERENCE_PATTERN)
_SOURCE_SHA = re.compile(r"[0-9a-f]{40}")
_CONFIG_KEY = re.compile(r"[a-zA-Z0-9_.-]+")
_CONFIG_TOKEN = re.compile(r"\{\{config\.([a-zA-Z0-9_.-]+)\}\}")


def _required_keys(value: Any) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(key, str) or not re.fullmatch(r"[a-zA-Z0-9_.-]+", key)
        for key in value
    ):
        raise NpaWorkflowError(
            "config.required_immutable_images must be a list of config keys"
        )
    if len(set(value)) != len(value):
        raise NpaWorkflowError(
            "config.required_immutable_images must contain unique config keys"
        )
    return value


def _is_exact_image(reference: Any) -> bool:
    return isinstance(reference, str) and bool(_DIGEST_REFERENCE.fullmatch(reference))


def resolved_baked_image_sources(config: Mapping[str, Any]) -> dict[str, str]:
    """Resolve exact baked-image source attestations from declarative config keys.

    The legacy direct map remains supported. The config-key map lets normal
    ``--var`` overrides supply image references and source SHAs independently,
    without requiring a nested mapping value on the command line.
    """

    direct = config.get("baked_image_source_sha_by_image") or {}
    if not isinstance(direct, Mapping):
        raise NpaWorkflowError(
            "config.baked_image_source_sha_by_image must map images to SHAs"
        )
    resolved: dict[str, str] = {}
    for image, source_sha in direct.items():
        if not _is_exact_image(image) or not isinstance(source_sha, str):
            raise NpaWorkflowError(
                "config.baked_image_source_sha_by_image must map exact images to SHAs"
            )
        normalized = source_sha.strip().lower()
        if not _SOURCE_SHA.fullmatch(normalized):
            raise NpaWorkflowError(
                "config.baked_image_source_sha_by_image contains an invalid source SHA"
            )
        resolved[image] = normalized

    keyed = config.get("baked_image_source_sha_by_config_key") or {}
    if not isinstance(keyed, Mapping):
        raise NpaWorkflowError(
            "config.baked_image_source_sha_by_config_key must map config keys"
        )
    for image_key, source_key in keyed.items():
        if (
            not isinstance(image_key, str)
            or not _CONFIG_KEY.fullmatch(image_key)
            or not isinstance(source_key, str)
            or not _CONFIG_KEY.fullmatch(source_key)
        ):
            raise NpaWorkflowError(
                "config.baked_image_source_sha_by_config_key contains an invalid key"
            )
        image = config.get(image_key)
        source_sha = config.get(source_key)
        if isinstance(source_sha, str) and (
            token := _CONFIG_TOKEN.fullmatch(source_sha.strip())
        ):
            source_sha = config.get(token.group(1))
        if image in (None, ""):
            continue
        if not _is_exact_image(image):
            raise NpaWorkflowError(
                f"config.{image_key} must be an exact immutable image"
            )
        if not isinstance(source_sha, str) or not _SOURCE_SHA.fullmatch(
            source_sha.strip().lower()
        ):
            raise NpaWorkflowError(
                f"config.{source_key} must be an exact 40-character source SHA"
            )
        normalized = source_sha.strip().lower()
        previous = resolved.get(image)
        if previous is not None and previous != normalized:
            raise NpaWorkflowError(
                f"immutable image {image} has conflicting baked source attestations"
            )
        resolved[image] = normalized
    return resolved


def required_immutable_image_inputs(config: Mapping[str, Any]) -> dict[str, str]:
    """Return declared provenance image inputs after exact-reference validation.

    Args:
        config: Workflow config after overrides and token resolution.
    Returns:
        Exact operator image references keyed by their config field.
    Raises:
        NpaWorkflowError: The declaration or a required image input is invalid.
    """
    if "required_immutable_images" not in config:
        return {}
    inputs: dict[str, str] = {}
    invalid_keys: list[str] = []
    for key in _required_keys(config["required_immutable_images"]):
        reference = config.get(key)
        if not _is_exact_image(reference):
            invalid_keys.append(key)
        else:
            inputs[key] = reference
    if invalid_keys:
        config_keys = ", ".join(f"config.{key}" for key in invalid_keys)
        overrides = " ".join(
            f"--var {key}=<registry>/<repository>@sha256:<64-hex-digest>"
            for key in invalid_keys
        )
        raise NpaWorkflowError(
            f"Each of {config_keys} requires an explicit registry-qualified immutable "
            "image; "
            f"set {overrides}. Use qualified images for this workload; tool:// and "
            "tag-only references cannot bind their provenance or child launches. Use "
            "digest-only references; remove any :tag segment before @sha256:."
        )
    return inputs


def validate_immutable_image_inputs(config: Mapping[str, Any]) -> None:
    """Require declared provenance image inputs to be exact digest references.

    Args:
        config: Workflow config after token resolution and operator overrides.

    Returns:
        None.

    Raises:
        NpaWorkflowError: A declared required image is not an exact digest.
    """
    required_immutable_image_inputs(config)
