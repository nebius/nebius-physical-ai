"""Solution registry bootstrap."""

from __future__ import annotations

from importlib import resources
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback.
    import tomli as tomllib

from . import registry

__all__ = ["registry"]


def _load_solutions() -> None:
    data = _read_solutions_toml()
    existing = {solution.name for solution in registry.list_solutions()}
    for entry in data.get("solutions", []):
        solution = _solution_from_entry(entry)
        if solution.name not in existing:
            registry.register_solution(solution)
            existing.add(solution.name)


def _read_solutions_toml() -> dict[str, Any]:
    solutions_file = resources.files(__package__).joinpath("solutions.toml")
    with solutions_file.open("rb") as handle:
        return tomllib.load(handle)


def _solution_from_entry(entry: dict[str, Any]) -> registry.Solution:
    return registry.Solution(
        name=str(entry["name"]),
        description=str(entry["description"]),
        version=str(entry["version"]),
        cli_namespace=str(entry["cli_namespace"]),
        tools=[str(tool) for tool in entry.get("tools", [])],
    )


_load_solutions()
