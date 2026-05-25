"""Registry primitives for NPA solutions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Solution:
    name: str
    description: str
    version: str
    cli_namespace: str
    tools: list[str]


_registry: dict[str, Solution] = {}


def register_solution(solution: Solution) -> None:
    """Register a solution by unique name."""
    if solution.name in _registry:
        raise ValueError(f"solution already registered: {solution.name}")
    _registry[solution.name] = _copy_solution(solution)


def list_solutions() -> list[Solution]:
    """Return registered solutions in registration order."""
    return [_copy_solution(solution) for solution in _registry.values()]


def _reset() -> None:
    """Clear registry state for unit tests."""
    _registry.clear()


def _copy_solution(solution: Solution) -> Solution:
    return Solution(
        name=solution.name,
        description=solution.description,
        version=solution.version,
        cli_namespace=solution.cli_namespace,
        tools=list(solution.tools),
    )
