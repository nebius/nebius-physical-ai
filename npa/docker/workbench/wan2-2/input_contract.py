"""Fail-closed input/capability contract for the Wan TI2V-5B image."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse


class WanInputContractError(ValueError):
    """The configured input mode and advertised artifact contract disagree."""


@dataclass(frozen=True)
class WanInputContract:
    task: str
    capability_name: str
    artifact_name: str


TEXT_TO_VIDEO = WanInputContract(
    task="text-to-video",
    capability_name="wan2.2_ti2v_5b_text_to_video",
    artifact_name="wan2_2_ti2v_5b_text_to_video.json",
)
IMAGE_TO_VIDEO = WanInputContract(
    task="image-to-video",
    capability_name="wan2.2_ti2v_5b_image_to_video",
    artifact_name="wan2_2_ti2v_5b_image_to_video.json",
)


def resolve_wan_input_contract(
    *,
    context_image_uri: str,
    declared_capability: str,
    declared_artifact: str,
) -> WanInputContract:
    """Resolve the input mode and reject malformed or mismatched declarations."""

    uri = context_image_uri.strip()
    if uri:
        parsed = urlparse(uri)
        if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
            raise WanInputContractError("context_image_uri must be an s3:// object URI")
        expected = IMAGE_TO_VIDEO
    else:
        expected = TEXT_TO_VIDEO

    actual = (declared_capability.strip(), declared_artifact.strip())
    wanted = (expected.capability_name, expected.artifact_name)
    if actual != wanted:
        raise WanInputContractError(
            "Wan input mode and declared BYOF contract disagree: "
            f"actual={wanted}, declared={actual}"
        )
    return expected
