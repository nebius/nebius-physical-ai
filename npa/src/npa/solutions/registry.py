"""Lightweight registry for NPA solutions."""

from __future__ import annotations

from importlib import resources
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback.
    import tomli as tomllib


SolutionEntry = dict[str, str]

_configured_solutions: list[SolutionEntry] | None = None
_registered_solutions: list[SolutionEntry] = []


def register_solution(name: str, description: str, cli_command: str) -> None:
    """Register an in-memory solution entry."""
    entry = _solution_entry(
        name=name,
        description=description,
        cli_command=cli_command,
    )
    if any(solution["name"] == entry["name"] for solution in _registered_solutions):
        raise ValueError(f"solution already registered: {entry['name']}")
    _registered_solutions.append(entry)


def list_solutions() -> list[SolutionEntry]:
    """Return configured and in-memory solution entries."""
    return [
        dict(solution)
        for solution in [*_load_configured_solutions(), *_registered_solutions]
    ]


def _reset() -> None:
    """Clear registry state for unit tests."""
    global _configured_solutions

    _configured_solutions = None
    _registered_solutions.clear()


def _load_configured_solutions() -> list[SolutionEntry]:
    global _configured_solutions

    if _configured_solutions is None:
        _configured_solutions = _read_solutions_toml()
    return _configured_solutions


def _read_solutions_toml() -> list[SolutionEntry]:
    solutions_file = resources.files("npa.solutions").joinpath("solutions.toml")
    with solutions_file.open("rb") as handle:
        data = tomllib.load(handle)

    entries = data.get("solutions")
    if entries is None:
        entries = [data]
    if not isinstance(entries, list):
        raise ValueError("solutions.toml must define solution entries")

    return [_solution_entry_from_config(entry) for entry in entries]


def _solution_entry_from_config(entry: Any) -> SolutionEntry:
    if not isinstance(entry, dict):
        raise ValueError("solutions.toml entries must be tables")
    return _solution_entry(
        name=str(entry["name"]),
        description=str(entry["description"]),
        cli_command=str(entry["cli_command"]),
    )


def _solution_entry(name: str, description: str, cli_command: str) -> SolutionEntry:
    return {
        "name": name,
        "description": description,
        "cli_command": cli_command,
    }
