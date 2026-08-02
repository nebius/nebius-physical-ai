from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import npa.solutions as solutions
from npa.solutions import registry

CONFIGURED_SOLUTIONS = [
    {
        "name": "workbench",
        "description": "First solution: physical AI robotics workflows built on Nebius infrastructure",
        "cli_command": "npa workbench",
    },
    {
        "name": "sim-to-real",
        "description": "Generic sim-to-real pipeline tools within the Workbench solution",
        # The staged 14-stage engine is the maintained sim-to-real path, and
        # `workflow submit` detects the runbook and routes to direct K8s. The retiring
        # skypilot/sim-to-real-loop.yaml was only the VLM-eval loop stage of it; that
        # capability now lives in `npa workbench vlm-eval loop` and vlm-eval-loop.yaml.
        "cli_command": (
            "npa workbench workflow submit npa/workflows/workbench/sim2real/runbook.yaml"
        ),
    },
    {
        "name": "retargeting",
        "description": "Motion retargeting Workbench tool for SONIC locomotion pipelines",
        "cli_command": "npa workbench sonic retargeting",
    },
    {
        "name": "mjlab",
        "description": "MJLab locomotion evaluation Workbench tool",
        "cli_command": "npa workbench mjlab",
    },
    {
        "name": "sonic-locomotion-finetuning",
        "description": (
            "SONIC locomotion fine-tuning workflow (retarget -> train -> MJLab eval)"
        ),
        "cli_command": (
            "npa workbench workflow submit "
            "npa/workflows/workbench/npa-workflows/sonic-locomotion-finetuning.yaml"
        ),
    },
]


# npa/tests/unit/solutions/test_registry.py -> repo root is four levels up plus one
# for the `npa/` package directory.
REPO_ROOT = Path(__file__).resolve().parents[4]


def test_configured_solution_workflow_paths_exist() -> None:
    """A shipped solution must not advertise a workflow file that was deleted.

    `solutions.toml` embeds `npa workbench workflow submit <path>` strings, so a
    retired workflow YAML silently turns a listed solution into a broken command.
    """

    from npa.orchestration.npa_workflow.detect import detect_submit_format

    missing: list[str] = []
    formats: dict[str, str] = {}
    for entry in CONFIGURED_SOLUTIONS:
        command = entry["cli_command"]
        if "workflow submit " not in command:
            continue
        path = command.split("workflow submit ", 1)[1].strip()
        resolved = REPO_ROOT / path
        if not resolved.is_file():
            missing.append(f"{entry['name']}: {path}")
            continue
        formats[entry["name"]] = detect_submit_format(resolved)
    assert not missing, "solutions.toml points at missing workflow files: " + ", ".join(
        missing
    )
    # Existing is not enough: submit has to recognise the file. The retiring raw catalog
    # classified as "skypilot"; every shipped solution should now name either an
    # npa.workflow spec or the sim2real runbook.
    assert formats, "expected at least one solution to advertise a workflow file"
    assert set(formats.values()) <= {"npa.workflow", "sim2real_runbook"}, formats


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
        *CONFIGURED_SOLUTIONS,
        {"name": "demo", "description": "Demo solution", "cli_command": "npa demo"},
    ]


def test_register_solution_rejects_duplicate_name() -> None:
    registry.register_solution("demo", "Demo solution", "npa demo")
    with pytest.raises(ValueError, match="duplicate solution name: demo"):
        registry.register_solution("demo", "Duplicate demo", "npa demo")


def test_list_solutions_returns_entry_copies() -> None:
    registry.register_solution("demo", "Demo solution", "npa demo")

    listed = registry.list_solutions()
    listed[-1]["description"] = "mutated"

    assert registry.list_solutions()[-1]["description"] == "Demo solution"


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

    assert first == CONFIGURED_SOLUTIONS
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
