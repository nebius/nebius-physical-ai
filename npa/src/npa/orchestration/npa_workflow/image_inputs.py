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


def validate_immutable_image_inputs(config: Mapping[str, Any]) -> None:
    """Require declared provenance image inputs to be exact digest references.

    Args:
        config: Workflow config after overrides and token resolution.
    Returns:
        None.
    Raises:
        NpaWorkflowError: The declaration or a required image input is invalid.
    """
    if "required_immutable_images" not in config:
        return
    for key in _required_keys(config["required_immutable_images"]):
        if _is_exact_image(config.get(key)):
            continue
        raise NpaWorkflowError(
            f"config.{key} requires an explicit registry-qualified immutable image; "
            f"set --var {key}=<registry>/<repository>@sha256:<64-hex-digest>. "
            "Use a qualified image for this workload; tool:// and tag-only "
            "references cannot bind its provenance or child launches."
        )
