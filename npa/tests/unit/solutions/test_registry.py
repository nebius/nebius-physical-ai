from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import npa.solutions as solutions
from npa.solutions import registry


@pytest.fixture(autouse=True)
def reset_registry() -> None:
    registry._reset()
    yield
    registry._reset()
    importlib.reload(solutions)


@pytest.fixture
def solutions_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def write_solutions_toml(contents: str) -> None:
        (tmp_path / "solutions.toml").write_text(contents, encoding="utf-8")
        monkeypatch.setattr(registry.resources, "files", lambda _package: tmp_path)
        registry._reset()

    return write_solutions_toml


def test_register_solution_lists_registered_solution() -> None:
    registry.register_solution(
        "demo",
        "Demo solution",
        "npa demo",
    )

    assert registry.list_solutions() == [
        {
            "name": "workbench",
            "description": "First solution: physical AI robotics workflows built on Nebius infrastructure",
            "cli_command": "npa workbench",
        },
        {"name": "demo", "description": "Demo solution", "cli_command": "npa demo"},
    ]


def test_register_solution_rejects_duplicate_name() -> None:
    registry.register_solution("demo", "Demo solution", "npa demo")
    with pytest.raises(ValueError, match="duplicate solution name: demo"):
        registry.register_solution("demo", "Duplicate demo", "npa demo")


def test_list_solutions_returns_entry_copies() -> None:
    registry.register_solution("demo", "Demo solution", "npa demo")

    listed = registry.list_solutions()
    listed[1]["description"] = "mutated"

    assert registry.list_solutions()[1]["description"] == "Demo solution"


def test_solutions_package_import_does_not_load_toml(mocker) -> None:
    registry._reset()
    load_mock = mocker.patch(
        "npa.solutions.registry._read_solutions_toml",
        side_effect=AssertionError("solutions.toml loaded during import"),
    )

    importlib.reload(solutions)

    load_mock.assert_not_called()


def test_list_solutions_lazily_loads_workbench_solution(mocker) -> None:
    registry._reset()
    load_spy = mocker.spy(registry, "_read_solutions_toml")

    first = registry.list_solutions()
    second = registry.list_solutions()

    assert first == [
        {
            "name": "workbench",
            "description": "First solution: physical AI robotics workflows built on Nebius infrastructure",
            "cli_command": "npa workbench",
        }
    ]
    assert second == first
    assert load_spy.call_count == 1


def test_list_solutions_loads_multiple_configured_solutions(solutions_toml) -> None:
    solutions_toml(
        """
[[solutions]]
name = "workbench"
description = "Workbench solution"
cli_command = "npa workbench"

[[solutions]]
name = "datalake"
description = "Datalake solution"
cli_command = "npa datalake"
"""
    )

    assert registry.list_solutions() == [
        {
            "name": "workbench",
            "description": "Workbench solution",
            "cli_command": "npa workbench",
        },
        {
            "name": "datalake",
            "description": "Datalake solution",
            "cli_command": "npa datalake",
        },
    ]


def test_configured_solution_names_must_be_unique(solutions_toml) -> None:
    solutions_toml(
        """
[[solutions]]
name = "workbench"
description = "Workbench solution"
cli_command = "npa workbench"

[[solutions]]
name = "workbench"
description = "Duplicate workbench solution"
cli_command = "npa workbench-duplicate"
"""
    )

    with pytest.raises(ValueError, match="duplicate solution name: workbench"):
        registry.list_solutions()


def test_registered_solution_cannot_duplicate_configured_solution() -> None:
    with pytest.raises(ValueError, match="duplicate solution name: workbench"):
        registry.register_solution(
            "workbench",
            "Duplicate workbench solution",
            "npa workbench",
        )
