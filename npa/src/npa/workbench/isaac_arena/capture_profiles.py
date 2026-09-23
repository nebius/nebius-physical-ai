"""Define named native Arena capture settings and their verifiable quality contract."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from .errors import IsaacArenaError

VIDEO_PROFILE_ENV = "NPA_ISAAC_ARENA_VIDEO_PROFILE"
STARTUP_RENDER_SETTINGS = {
    "/persistent/rtx/modes/rt/enabled": True,
    "/persistent/rtx/modes/rt2/enabled": False,
    "/persistent/rtx/modes/pt/enabled": False,
}
CAPTURE_RENDER_SETTINGS = {
    "/rtx/rendermode": "RaytracedLighting",
    "/rtx/post/aa/op": 1,
    "/rtx/post/dlss/execMode": 2,
    "/rtx-transient/dldenoiser/enabled": True,
    "/rtx-transient/dlssg/enabled": False,
    "/rtx/ecoMode/enabled": False,
    "/rtx/directLighting/sampledLighting/samplesPerPixel": 32,
    "/rtx/indirectDiffuse/fetchSampleCount": 32,
    "/rtx/reflections/sampledLighting/samplesPerPixel": 16,
}


@dataclass(frozen=True)
class CaptureProfile:
    """Describe one native render profile without changing physics or camera pose.

    Args:
        name: Public profile selector.
        resolution: Required native dimensions, or the environment's existing size.
        settling_renders: Consecutive ready renders after the readiness baseline.
        direct_samples: Native direct-light samples per pixel.
        diffuse_samples: Native indirect-diffuse sample count.
        reflection_samples: Native reflection samples per pixel.
    Returns:
        An immutable profile description.
    Raises:
        None.
    """

    name: str
    resolution: tuple[int, int] | None
    settling_renders: int
    direct_samples: int
    diffuse_samples: int
    reflection_samples: int


_PROFILES = {
    "standard": CaptureProfile("standard", None, 8, 32, 32, 16),
    "film": CaptureProfile("film", (3840, 2160), 32, 128, 128, 64),
}


def capture_profile(name: str = "standard") -> CaptureProfile:
    """Resolve a supported profile and reject unknown quality requests.

    Args:
        name: ``standard`` retains existing capture; ``film`` requests native 4K.
    Returns:
        The named immutable capture profile.
    Raises:
        IsaacArenaError: The selector is unsupported.
    """
    if not isinstance(name, str) or name not in _PROFILES:
        raise IsaacArenaError("video_profile must be standard or film")
    return _PROFILES[name]


def simulator_capture_profile() -> CaptureProfile:
    """Read the validated request forwarded to the isolated simulator process.

    Args:
        None.
    Returns:
        The explicitly forwarded profile, or standard for older callers.
    Raises:
        IsaacArenaError: The internal selector is unsupported.
    """
    return capture_profile(os.environ.get(VIDEO_PROFILE_ENV, "standard"))


def capture_settings(profile: CaptureProfile) -> dict[str, Any]:
    """Return the exact live Carb settings required by a named capture profile.

    Args:
        profile: Trusted profile returned by ``capture_profile``.
    Returns:
        A new mapping, preserving renderer, anti-aliasing and frame-generation rules.
    Raises:
        None.
    """
    return {
        **CAPTURE_RENDER_SETTINGS,
        "/rtx/directLighting/sampledLighting/samplesPerPixel": profile.direct_samples,
        "/rtx/indirectDiffuse/fetchSampleCount": profile.diffuse_samples,
        "/rtx/reflections/sampledLighting/samplesPerPixel": profile.reflection_samples,
    }


def render_settings(profile: CaptureProfile) -> dict[str, Any]:
    """Return required startup and live renderer settings for readback validation.

    Args:
        profile: Trusted profile returned by ``capture_profile``.
    Returns:
        A new complete mapping of settings whose actual values must match.
    Raises:
        None.
    """
    return {**STARTUP_RENDER_SETTINGS, **capture_settings(profile)}
