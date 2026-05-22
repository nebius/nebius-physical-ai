"""Shared Workbench container image naming."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_CONTAINER_REGISTRY = "cr.eu-north1.nebius.cloud/your-registry-id"

CONTAINER_IMAGE_NAMES = {
    "lerobot": "npa-lerobot",
    "genesis": "npa-genesis",
    "isaac-lab": "npa-isaac-lab",
    "cosmos": "npa-cosmos",
    "groot": "npa-groot",
    "fiftyone": "npa-fiftyone",
    "sonic": "npa-sonic",
}


def supported_tool_version(tool: str) -> str:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    for directory in Path(__file__).resolve().parents:
        pyproject = directory / "pyproject.toml"
        if pyproject.is_file():
            with pyproject.open("rb") as handle:
                data = tomllib.load(handle)
            return str(data["tool"]["npa"]["supported-tools"][tool])
    raise RuntimeError(f"Could not find pyproject.toml for tool version lookup: {tool}")


def container_image_for_tool(
    tool: str,
    *,
    registry: str | None = None,
    tag: str | None = None,
) -> str:
    """Return the fully qualified image ref for a Workbench tool."""
    image_name = CONTAINER_IMAGE_NAMES[tool]
    resolved_tag = tag or supported_tool_version(tool)
    resolved_registry = registry or os.environ.get("NPA_REGISTRY") or DEFAULT_CONTAINER_REGISTRY
    return f"{resolved_registry.rstrip('/')}/{image_name}:{resolved_tag}"
