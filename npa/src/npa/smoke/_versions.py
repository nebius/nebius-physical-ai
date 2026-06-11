"""Version helpers shared by standalone smoke checks.

These run inside container images whose Python may predate 3.11 and may not have
``tomli`` installed (e.g. the genesis py3.10 venv). To stay dependency-free, the
``[tool.npa.supported-tools]`` lookup is parsed with the stdlib only; ``tomllib``
is used when available purely as a fast path.
"""

from __future__ import annotations

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


def _parse_supported_tools(text: str) -> dict[str, str]:
    """Parse the ``[tool.npa.supported-tools]`` table without a TOML library."""

    versions: dict[str, str] = {}
    in_section = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            in_section = line == "[tool.npa.supported-tools]"
            continue
        if not in_section:
            continue
        match = re.match(r'^([A-Za-z0-9._-]+)\s*=\s*"([^"]*)"', line)
        if match:
            versions[match.group(1)] = match.group(2)
    return versions


def _supported_tools(pyproject: Path) -> dict[str, str]:
    text = pyproject.read_text(encoding="utf-8")
    if _tomllib is not None:
        data = _tomllib.loads(text)
        table = data.get("tool", {}).get("npa", {}).get("supported-tools", {})
        if isinstance(table, dict):
            return {str(key): str(value) for key, value in table.items()}
    return _parse_supported_tools(text)


def supported_tool_version(tool: str, start_file: str) -> str:
    pyproject = _find_pyproject(start_file)
    versions = _supported_tools(pyproject)
    try:
        return str(versions[tool])
    except KeyError as exc:
        raise KeyError(
            f"Missing [tool.npa.supported-tools].{tool} in {pyproject}"
        ) from exc
