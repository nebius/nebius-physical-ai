"""Version helpers shared by standalone smoke checks.

The supported NPA runtime is Python 3.11 or newer, so metadata parsing uses its
stdlib ``tomllib`` implementation. Image release tags live in
``[tool.npa.supported-tools]`` while installed versions may be declared in
``[tool.npa.package-versions]``.
"""

from __future__ import annotations

from pathlib import Path
import tomllib


def _find_pyproject(start_file: str) -> Path:
    start = Path(start_file).resolve()
    for directory in (start.parent, *start.parent.parents):
        candidate = directory / "pyproject.toml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find pyproject.toml above {start}")


def _supported_tools(pyproject: Path) -> dict[str, str]:
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    table = data.get("tool", {}).get("npa", {}).get("supported-tools", {})
    return (
        {str(key): str(value) for key, value in table.items()}
        if isinstance(table, dict)
        else {}
    )


def _package_versions(pyproject: Path) -> dict[str, str]:
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    table = data.get("tool", {}).get("npa", {}).get("package-versions", {})
    return (
        {str(key): str(value) for key, value in table.items()}
        if isinstance(table, dict)
        else {}
    )


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
        return override
    return supported_tool_version("lerobot", start_file)


def train_env_eval_arg_for_version(version: str, value: int = 1_000_000) -> str:
    """Return the train env-eval cadence CLI arg for a LeRobot version.

    Kept dependency-free for container smokes that do not ship the full
    ``npa.workbench`` package tree.
    """

    flag = "env_eval_freq" if str(version).startswith("0.6") else "eval_freq"
    return f"--{flag}={int(value)}"
