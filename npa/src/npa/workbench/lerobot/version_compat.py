"""LeRobot version resolution and CLI flag compatibility helpers."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from importlib import resources
from typing import Any

LEROBOT_VERSION_MANIFEST_RESOURCE = "lerobot_version_manifest.json"
LEROBOT_VERSION_ENV = "NPA_LEROBOT_VERSION"


class LeRobotVersionError(ValueError):
    """Raised when a LeRobot version is unsupported or misconfigured."""


@lru_cache(maxsize=1)
def lerobot_version_manifest() -> dict[str, Any]:
    """Return the packaged LeRobot version compatibility manifest."""

    text = resources.files("npa.deploy").joinpath(LEROBOT_VERSION_MANIFEST_RESOURCE).read_text(
        encoding="utf-8"
    )
    payload = json.loads(text)
    if payload.get("format") != "npa_lerobot_version_manifest_v1":
        raise RuntimeError("Unsupported LeRobot version manifest format")
    return payload


def default_lerobot_version() -> str:
    """Return the default LeRobot version (pyproject / supported-tools pin)."""

    from npa.deploy.images import supported_tool_version

    return supported_tool_version("lerobot")


def supported_lerobot_versions() -> tuple[str, ...]:
    """Return supported LeRobot versions in declared order."""

    versions = lerobot_version_manifest().get("supported_versions") or []
    return tuple(str(item) for item in versions)


def resolve_lerobot_version(version: str | None = None) -> str:
    """Resolve a LeRobot version from explicit arg, env, or default."""

    candidates = (
        (version or "").strip(),
        os.environ.get(LEROBOT_VERSION_ENV, "").strip(),
        default_lerobot_version(),
    )
    resolved = next((item for item in candidates if item), "")
    if not resolved:
        raise LeRobotVersionError("Could not resolve a LeRobot version")
    supported = supported_lerobot_versions()
    if resolved not in supported:
        choices = ", ".join(supported)
        raise LeRobotVersionError(
            f"Unsupported LeRobot version {resolved!r}; choose one of: {choices}"
        )
    return resolved


def lerobot_version_entry(version: str | None = None) -> dict[str, Any]:
    """Return the manifest entry for a resolved LeRobot version."""

    resolved = resolve_lerobot_version(version)
    versions = lerobot_version_manifest().get("versions") or {}
    try:
        entry = versions[resolved]
    except KeyError as exc:
        raise LeRobotVersionError(f"Missing LeRobot version entry for {resolved!r}") from exc
    if not isinstance(entry, dict):
        raise LeRobotVersionError(f"Invalid LeRobot version entry for {resolved!r}")
    return {"version": resolved, **entry}


def lerobot_pip_spec(version: str | None = None) -> str:
    """Return the pinned pip install spec for a LeRobot version."""

    entry = lerobot_version_entry(version)
    extras = str(entry.get("pip_extras") or "").strip()
    resolved = str(entry["version"])
    if extras:
        return f"lerobot[{extras}]=={resolved}"
    return f"lerobot=={resolved}"


def train_env_eval_flag(version: str | None = None) -> str:
    """Return the train-time env-eval cadence flag name for this version."""

    return str(lerobot_version_entry(version)["train_env_eval_flag"])


def train_env_eval_arg(value: int = 1_000_000, *, version: str | None = None) -> str:
    """Return ``--eval_freq=...`` or ``--env_eval_freq=...`` for this version."""

    return f"--{train_env_eval_flag(version)}={int(value)}"


def eval_checkpoint_arg(
    checkpoint: str,
    *,
    version: str | None = None,
    style: str = "cli",
) -> str:
    """Return the eval checkpoint flag for workbench CLI-style invocations."""

    entry = lerobot_version_entry(version)
    if style == "policy":
        flag = str(entry["policy_eval_checkpoint_flag"])
    else:
        flag = str(entry["eval_checkpoint_flag"])
    return f"--{flag}={checkpoint}"


def torch_install_pins(version: str | None = None) -> list[str]:
    """Return optional torch/torchvision/diffusers pins after lerobot install."""

    entry = lerobot_version_entry(version)
    pins: list[str] = []
    for key in ("torch_pin", "torchvision_pin", "diffusers_pin"):
        value = entry.get(key)
        if value:
            pins.append(str(value))
    return pins
