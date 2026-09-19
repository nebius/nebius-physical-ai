"""Validate task-specific appearance profiles and compose source-grounded prompts."""

from __future__ import annotations

import json
from typing import Any


APPEARANCE_FIELDS = ("lighting", "background", "color_grade", "surface_finish")


def parse_appearance_profiles(value: str) -> list[dict[str, str]] | None:
    """Validate an optional JSON array of coherent appearance profiles.

    Args:
        value: JSON text; an empty string selects the existing default sampler.
    Returns:
        Normalized profiles, or None for the default sampler.
    Raises:
        ValueError: JSON is malformed, empty, duplicated, or has invalid fields.
    """
    if not isinstance(value, str):
        raise ValueError("appearance_profiles_json must be a JSON string")
    if not value.strip():
        return None
    try:
        profiles = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("appearance_profiles_json must contain valid JSON") from exc
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("appearance_profiles_json must contain a non-empty array")
    normalized = [_validate_profile(profile) for profile in profiles]
    identities = [
        tuple(profile[key] for key in APPEARANCE_FIELDS) for profile in normalized
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("appearance_profiles_json contains duplicate profiles")
    return normalized


def _validate_profile(profile: Any) -> dict[str, str]:
    if not isinstance(profile, dict) or set(profile) != set(APPEARANCE_FIELDS):
        raise ValueError(
            "each appearance profile requires exactly " + ", ".join(APPEARANCE_FIELDS)
        )
    if any(
        not isinstance(value, str) or not value.strip() for value in profile.values()
    ):
        raise ValueError("appearance profile values must be non-empty strings")
    return {key: profile[key].strip() for key in APPEARANCE_FIELDS}


def appearance_prompt(profile: dict[str, str], subject: str) -> str:
    """Describe a coherent appearance edit without replacing task geometry.

    Args:
        profile: Validated lighting, background, color grade and surface finish.
        subject: Operator description of the source scene or task.
    Returns:
        Appearance instructions for the model, also retained in the manifest.
    Raises:
        KeyError: A required profile field is missing.
    """
    requirements = "; ".join(
        f"{key.replace('_', ' ')}: {profile[key]}" for key in APPEARANCE_FIELDS
    )
    return (
        f"Photorealistic {subject}. Appearance requirements: {requirements}. "
        "Apply these consistently across the complete video and all generation windows. "
        "Change the appearance of existing surfaces only; retain their boundaries, "
        "positions, depth, occlusions and contact relationships. Preserve the exact "
        "foreground objects, robot parts, object counts, identity colors and markings. "
        "Preserve small task features, gripper openings, grasp points and insertion "
        "openings. Keep source camera motion, trajectories, timing and task outcome. "
        "Keep realistic exposure, contact shadows and material response; no added "
        "objects, scene replacement, geometry deformation, flicker or motion retiming."
    )


def generation_prompt(instructions: str, caption: str, appearance: str) -> str:
    """Separate fallible source observations from the requested appearance edit.

    Args:
        instructions: Operator's scene and task preservation instructions.
        caption: Sampled source observations spanning the complete episode.
        appearance: The selected appearance profile's model instructions.
    Returns:
        Effective prompt recorded with the generated video.
    Raises:
        None.
    """
    return (
        "The source video is authoritative for objects, geometry, camera, motion, "
        "contacts and task outcome. Source observations describe the original, "
        "not instructions to recreate its lighting or materials. Do not invent "
        "objects or actions from an uncertain caption. "
        f"Source observations: {caption}\n"
        f"Task preservation instructions: {instructions.strip()}\n"
        f"Requested appearance edit: {appearance.strip()}"
    )
