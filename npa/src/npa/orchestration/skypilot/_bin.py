"""SkyPilot CLI executable resolution."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

SkyBin = str | os.PathLike[str] | None

_SETUP_DOC = "docs/orchestration/skypilot-setup.md"


class SkyPilotNotInstalledError(RuntimeError):
    """Raised when NPA cannot find an executable SkyPilot CLI."""


def resolve_sky_bin(sky_bin: SkyBin = None) -> Path:
    """Resolve the SkyPilot CLI executable for subprocess calls."""

    if sky_bin is not None:
        return _resolve_candidate(sky_bin, "explicit sky_bin")

    env_value = os.environ.get("NPA_SKYPILOT_BIN", "").strip()
    if env_value:
        return _resolve_candidate(env_value, "NPA_SKYPILOT_BIN")

    discovered = shutil.which("sky")
    if discovered:
        path = Path(discovered)
        if _is_executable_file(path):
            return path.resolve()

    raise SkyPilotNotInstalledError(
        "SkyPilot CLI executable was not found. Install SkyPilot in an isolated "
        "venv, set NPA_SKYPILOT_BIN to that venv's sky binary, or put sky on "
        f"PATH. See {_SETUP_DOC}."
    )


def _resolve_candidate(candidate: str | os.PathLike[str], source: str) -> Path:
    value = os.fspath(candidate).strip()
    if not value:
        raise SkyPilotNotInstalledError(
            f"SkyPilot CLI from {source} is empty. Set it to an executable sky "
            f"binary. See {_SETUP_DOC}."
        )

    path = Path(value).expanduser()
    if _is_executable_file(path):
        return path.resolve()

    if _looks_like_command_name(value):
        discovered = shutil.which(value)
        if discovered:
            discovered_path = Path(discovered)
            if _is_executable_file(discovered_path):
                return discovered_path.resolve()

    raise SkyPilotNotInstalledError(
        f"SkyPilot CLI from {source} does not resolve to an executable file: "
        f"{value}. Install SkyPilot in an isolated venv and point "
        f"NPA_SKYPILOT_BIN at its sky binary. See {_SETUP_DOC}."
    )


def _is_executable_file(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _looks_like_command_name(value: str) -> bool:
    if os.sep in value:
        return False
    return os.altsep is None or os.altsep not in value
