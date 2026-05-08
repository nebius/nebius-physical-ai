"""Version helpers shared by standalone smoke checks."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib


def _find_pyproject(start_file: str) -> Path:
    start = Path(start_file).resolve()
    for directory in (start.parent, *start.parent.parents):
        candidate = directory / "pyproject.toml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find pyproject.toml above {start}")


def supported_tool_version(tool: str, start_file: str) -> str:
    pyproject = _find_pyproject(start_file)
    with pyproject.open("rb") as handle:
        data = tomllib.load(handle)

    try:
        version = data["tool"]["npa"]["supported-tools"][tool]
    except KeyError as exc:
        raise KeyError(
            f"Missing [tool.npa.supported-tools].{tool} in {pyproject}"
        ) from exc
    return str(version)
