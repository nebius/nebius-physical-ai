"""Version helpers shared by standalone smoke checks.

These run inside container images whose Python may predate 3.11 and may not have
``tomli`` installed (e.g. the genesis py3.10 venv). To stay dependency-free,
the package-version lookup is parsed with the stdlib only; ``tomllib`` is used
when available purely as a fast path. Image release tags live in
``[tool.npa.supported-tools]`` while installed versions may be declared in
``[tool.npa.package-versions]``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

try:  # Fast path on Python >= 3.11; never required.
    import tomllib as _tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on py<3.11 images.
    _tomllib = None


def _find_pyproject(start_file: str) -> Path:
    start = Path(start_file).resolve()
    for directory in (start.parent, *start.parent.parents):
        candidate = directory / "pyproject.toml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find pyproject.toml above {start}")


def _parse_table(text: str, table: str) -> dict[str, str]:
    """Parse a simple ``[tool.npa.*]`` string table without a TOML library."""

    versions: dict[str, str] = {}
    in_section = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            in_section = line == f"[tool.npa.{table}]"
            continue
        if not in_section:
            continue
        match = re.match(r'^([A-Za-z0-9._-]+)\s*=\s*"([^"]*)"', line)
        if match:
            versions[match.group(1)] = match.group(2)
    return versions


def _parse_supported_tools(text: str) -> dict[str, str]:
    return _parse_table(text, "supported-tools")


def _supported_tools(pyproject: Path) -> dict[str, str]:
    text = pyproject.read_text(encoding="utf-8")
    if _tomllib is not None:
        data = _tomllib.loads(text)
        table = data.get("tool", {}).get("npa", {}).get("supported-tools", {})
        if isinstance(table, dict):
            return {str(key): str(value) for key, value in table.items()}
    return _parse_supported_tools(text)


def _package_versions(pyproject: Path) -> dict[str, str]:
    text = pyproject.read_text(encoding="utf-8")
    if _tomllib is not None:
        data = _tomllib.loads(text)
        table = data.get("tool", {}).get("npa", {}).get("package-versions", {})
        if isinstance(table, dict):
            return {str(key): str(value) for key, value in table.items()}
    return _parse_table(text, "package-versions")


def _secure_lerobot_integration_version() -> str | None:
    """Return the reviewed local integration version, if this image has one."""

    receipt_path = Path("/usr/share/doc/npa-lerobot/source-integration.json")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    version = receipt.get("integration_version")
    if version in {
        "0.5.1+npa.secure2",
        "0.6.0+npa.secure2",
        "0.6.0+npa.secure3",
    }:
        return str(version)
    return None


def _has_secure_lerobot_integration() -> bool:
    """Whether this image contains the reviewed 0.5.1 integration."""

    return _secure_lerobot_integration_version() == "0.5.1+npa.secure2"


def _integrated_lerobot_version(version: str) -> str | None:
    """Return a matching reviewed integration without crossing version lanes."""

    if version == "0.5.1" and _has_secure_lerobot_integration():
        return "0.5.1+npa.secure2"
    secure_version = _secure_lerobot_integration_version()
    if secure_version and secure_version.split("+", 1)[0] == version:
        return secure_version
    return None


def supported_tool_version(tool: str, start_file: str) -> str:
    pyproject = _find_pyproject(start_file)
    package_versions = _package_versions(pyproject)
    if tool in package_versions:
        return package_versions[tool]
    versions = _supported_tools(pyproject)
    try:
        return str(versions[tool])
    except KeyError as exc:
        raise KeyError(
            f"Missing [tool.npa.supported-tools].{tool} in {pyproject}"
        ) from exc


def expected_lerobot_version(start_file: str) -> str:
    """Return the LeRobot version this image/smoke should assert.

    Prefer ``NPA_LEROBOT_VERSION`` (set in multi-version container images) so
    ``npa-lerobot:0.6.0`` can pass env smoke while the pyproject default remains
    ``0.5.1``.
    """

    import os

    override = os.environ.get("NPA_LEROBOT_VERSION", "").strip()
    if override:
        return _integrated_lerobot_version(override) or override
    version = supported_tool_version("lerobot", start_file)
    return _integrated_lerobot_version(version) or version


def train_env_eval_arg_for_version(version: str, value: int = 1_000_000) -> str:
    """Return the train env-eval cadence CLI arg for a LeRobot version.

    Kept dependency-free for container smokes that do not ship the full
    ``npa.workbench`` package tree.
    """

    flag = "env_eval_freq" if str(version).startswith("0.6") else "eval_freq"
    return f"--{flag}={int(value)}"
