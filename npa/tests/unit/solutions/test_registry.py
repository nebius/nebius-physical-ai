from __future__ import annotations

import importlib

import pytest

import npa.solutions as solutions
from npa.solutions import registry
from npa.solutions.registry import Solution


@pytest.fixture(autouse=True)
def reset_registry() -> None:
    registry._reset()
    yield
    registry._reset()
    importlib.reload(solutions)


def test_register_solution_lists_registered_solution() -> None:
    solution = Solution(
        name="demo",
        description="Demo solution",
        version="1.0.0",
        cli_namespace="demo",
        tools=["tool-a"],
    )

    registry.register_solution(solution)

    assert registry.list_solutions() == [solution]


def test_register_solution_rejects_duplicate_name() -> None:
    solution = Solution(
        name="demo",
        description="Demo solution",
        version="1.0.0",
        cli_namespace="demo",
        tools=["tool-a"],
    )

    registry.register_solution(solution)
    with pytest.raises(ValueError, match="solution already registered: demo"):
        registry.register_solution(solution)


def test_list_solutions_returns_tool_list_copy() -> None:
    registry.register_solution(
        Solution(
            name="demo",
            description="Demo solution",
            version="1.0.0",
            cli_namespace="demo",
            tools=["tool-a"],
        )
    )

    listed = registry.list_solutions()
    listed[0].tools.append("mutated")

    assert registry.list_solutions()[0].tools == ["tool-a"]


def test_solutions_toml_bootstrap_registers_workbench_solution() -> None:
    importlib.reload(solutions)

    registered = {solution.name: solution for solution in registry.list_solutions()}

    assert "workbench" in registered
    assert registered["workbench"].cli_namespace == "workbench"
    assert registered["workbench"].tools == [
        "lerobot",
        "fiftyone",
        "genesis",
        "isaac-lab",
        "cosmos",
        "lancedb",
        "groot",
        "sonic",
    ]
