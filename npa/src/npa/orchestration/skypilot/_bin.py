"""SkyPilot CLI executable resolution."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from npa.config_schema import (
    SKYPILOT_CONFIG_KEYS,
    SKYPILOT_RUNTIME_CONFIG_KEYS,
    unknown_config_keys,
)

SkyBin = str | os.PathLike[str] | None

_SETUP_DOC = "docs/orchestration/skypilot-setup.md"
CONFIG_PATH = Path.home() / ".npa" / "config.yaml"
REQUIRED_SKYPILOT_VERSION = "0.12.2"
_VERSION_CHECK_CACHE: set[tuple[str, int, int]] = set()


@dataclass(frozen=True)
class SkyPilotConfig:
    """Resolved SkyPilot runtime config.

    Precedence for each field is explicit argument, then environment variable,
    then the ``skypilot`` section in ``~/.npa/config.yaml``.
    """

    sky_bin: Path
    global_config_path: Path | None = None
    isolated_config_dir: Path | None = None


class SkyPilotNotInstalledError(RuntimeError):
    """Raised when NPA cannot find an executable SkyPilot CLI."""


class SkyPilotConfigError(ValueError):
    """Raised when NPA SkyPilot config is invalid."""


class SkyPilotVersionError(RuntimeError):
    """Raised when the resolved SkyPilot CLI does not match NPA's pin."""


def resolve_config(
    *,
    sky_bin: SkyBin = None,
    global_config_path: str | os.PathLike[str] | None = None,
    isolated_config_dir: str | os.PathLike[str] | None = None,
    npa_config_path: str | os.PathLike[str] | None = None,
) -> SkyPilotConfig:
    """Resolve SkyPilot runtime config with explicit -> env -> config precedence."""

    config_path = Path(npa_config_path) if npa_config_path is not None else CONFIG_PATH
    file_config = _load_skypilot_file_config(config_path)
    sky_value, sky_source = _first_config_value(
        (sky_bin, "explicit sky_bin"),
        (os.environ.get("NPA_SKYPILOT_BIN", "").strip(), "NPA_SKYPILOT_BIN"),
        (file_config.get("sky_bin"), f"{config_path}: skypilot.sky_bin"),
    )
    if sky_value is None:
        raise SkyPilotNotInstalledError(
            "SkyPilot CLI executable is not configured. Pass sky_bin, set "
            f"NPA_SKYPILOT_BIN, or set skypilot.sky_bin in {config_path}. "
            f"See {_SETUP_DOC}."
        )

    global_value, _ = _first_config_value(
        (global_config_path, "explicit config_path"),
        (os.environ.get("SKYPILOT_GLOBAL_CONFIG", "").strip(), "SKYPILOT_GLOBAL_CONFIG"),
        (file_config.get("global_config_path"), f"{config_path}: skypilot.global_config_path"),
    )
    isolated_value, _ = _first_config_value(
        (isolated_config_dir, "explicit isolated_config_dir"),
        (
            os.environ.get("NPA_SKYPILOT_ISOLATED_CONFIG_DIR", "").strip(),
            "NPA_SKYPILOT_ISOLATED_CONFIG_DIR",
        ),
        (file_config.get("isolated_config_dir"), f"{config_path}: skypilot.isolated_config_dir"),
    )
    return SkyPilotConfig(
        sky_bin=_resolve_candidate(sky_value, sky_source),
        global_config_path=_optional_path(global_value),
        isolated_config_dir=_optional_path(isolated_value),
    )


def resolve_sky_bin(sky_bin: SkyBin = None) -> Path:
    """Resolve the SkyPilot CLI executable for subprocess calls."""

    return resolve_config(sky_bin=sky_bin).sky_bin


def ensure_skypilot_version(sky_bin: SkyBin = None) -> Path:
    """Assert the executable and isolated dependency matrix before mutation."""

    sky_path = resolve_sky_bin(sky_bin)
    stat_result = sky_path.stat()
    cache_key = (str(sky_path), stat_result.st_ino, stat_result.st_mtime_ns)
    if cache_key in _VERSION_CHECK_CACHE:
        return sky_path
    try:
        result = subprocess.run(
            [str(sky_path), "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SkyPilotVersionError(f"Unable to check SkyPilot version via {sky_path}") from exc
    output = f"{result.stdout}\n{result.stderr}"
    version = re.search(r"(\d+\.\d+\.\d+)", output)
    actual = version.group(1) if version else "unknown"
    if result.returncode != 0 or actual != REQUIRED_SKYPILOT_VERSION:
        raise SkyPilotVersionError(
            f"SkyPilot version mismatch: expected {REQUIRED_SKYPILOT_VERSION}, got {actual}"
        )
    python_path = sky_path.parent / "python"
    try:
        compatibility = subprocess.run(
            [
                str(python_path),
                "-c",
                (
                    "import sys,kubernetes;"
                    "print(f'{sys.version_info.major}.{sys.version_info.minor} '"
                    "f'{kubernetes.__version__}')"
                ),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SkyPilotVersionError(
            f"Unable to validate the isolated SkyPilot runtime via {python_path}. "
            "Run `npa skypilot bootstrap`."
        ) from exc
    fields = (compatibility.stdout or "").strip().split()
    py_version = fields[0] if fields else "unknown"
    kube_version = fields[1] if len(fields) > 1 else "unknown"
    try:
        py_tuple = tuple(int(part) for part in py_version.split(".")[:2])
        kube_major = int(kube_version.split(".", 1)[0])
    except ValueError as exc:
        raise SkyPilotVersionError(
            "SkyPilot runtime compatibility self-check returned malformed version "
            f"evidence (python={py_version}, kubernetes={kube_version}). Run "
            "`npa skypilot bootstrap`."
        ) from exc
    if (
        compatibility.returncode != 0
        or py_tuple < (3, 9)
        or py_tuple > (3, 12)
        or kube_major >= 36
        or kube_version == "32.0.0"
    ):
        raise SkyPilotVersionError(
            "Incompatible isolated SkyPilot runtime: requires Python 3.9-3.12 and "
            "kubernetes>=20,!=32.0.0,<36; found "
            f"Python {py_version}, kubernetes {kube_version}. Run "
            "`npa skypilot bootstrap --python python3.12` before retrying."
        )
    _VERSION_CHECK_CACHE.add(cache_key)
    return sky_path


def clear_skypilot_version_cache() -> None:
    """Clear the lazy SkyPilot version-check cache for tests."""

    _VERSION_CHECK_CACHE.clear()


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

    raise SkyPilotNotInstalledError(
        f"SkyPilot CLI from {source} does not resolve to an executable file: "
        f"{value}. Install SkyPilot in an isolated venv and point "
        f"NPA_SKYPILOT_BIN at its sky binary. See {_SETUP_DOC}."
    )


def _is_executable_file(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _load_skypilot_file_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SkyPilotConfigError(f"NPA config must be a mapping: {path}")
    section = data.get("skypilot", {})
    if section in (None, ""):
        return {}
    if not isinstance(section, dict):
        raise SkyPilotConfigError(f"NPA config skypilot section must be a mapping: {path}")
    unknown = unknown_config_keys("skypilot", section)
    if unknown:
        valid = ", ".join(sorted(SKYPILOT_CONFIG_KEYS))
        keys = ", ".join(unknown)
        raise SkyPilotConfigError(f"Unrecognized SkyPilot config key(s): {keys}. Valid keys: {valid}")
    # NPA-owned controller metadata shares the section for atomic persistence,
    # but never becomes a runtime setting or participates in runtime precedence.
    return {key: section[key] for key in SKYPILOT_RUNTIME_CONFIG_KEYS if key in section}


def _first_config_value(*candidates: tuple[Any, str]) -> tuple[Any | None, str]:
    for value, source in candidates:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value, source
    return None, ""


def _optional_path(value: Any | None) -> Path | None:
    if value is None:
        return None
    text = os.fspath(value).strip()
    if not text:
        return None
    return Path(text).expanduser()
