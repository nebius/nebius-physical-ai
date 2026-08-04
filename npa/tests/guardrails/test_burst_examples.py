"""Guardrail: burst examples stay single-task burst inputs, not workflows.

`npa/src/npa/burst/examples/` holds SkyPilot YAMLs that are inputs to
`npa burst submit-yaml`. They were relocated out of the retiring workflow catalog for the same
reason the BYOF resource profiles were (DESIGN.md §R10): they are not workflows. There is no
plan, no stage graph, no decision artifact and nothing for a `toolRef` to describe — the burst
API is deliberately scoped to *one* executable task.

Two properties keep that boundary honest:

* **one task per file** — the moment a second stage appears it is a workflow, and belongs in
  `npa/workflows/workbench/npa-workflows/` as an `npa.workflow/v0.0.1` spec;
* **`${VAR}` placeholders survive** — they are the burst substitution surface, so a concrete
  registry id, bucket name or run id must never be committed here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_DIR = REPO_ROOT / "npa" / "src" / "npa" / "burst" / "examples"

#: Pinned so a new file cannot appear without answering "why is this not a spec?".
PINNED_EXAMPLES = frozenset({"isaac-lab-cosmos-sdg-burst-smoke.yaml"})


def _documents(path: Path) -> list[dict]:
    return [doc for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")) if doc]


def test_example_set_is_pinned() -> None:
    found = {path.name for path in EXAMPLES_DIR.glob("*.yaml")}

    assert found == set(PINNED_EXAMPLES), (
        "burst examples changed. A multi-stage pipeline is a workflow: author an "
        "npa.workflow/v0.0.1 spec under npa/workflows/workbench/npa-workflows/ instead. "
        f"expected {sorted(PINNED_EXAMPLES)}, found {sorted(found)}"
    )


def test_the_directory_documents_the_boundary() -> None:
    # Normalised, because the prose is hard-wrapped and a phrase may span a line break.
    readme = " ".join((EXAMPLES_DIR / "README.md").read_text(encoding="utf-8").split())

    assert "not workflow templates" in readme
    assert "One task per file" in readme


@pytest.mark.parametrize("name", sorted(PINNED_EXAMPLES))
def test_example_is_a_single_task(name: str) -> None:
    docs = _documents(EXAMPLES_DIR / name)

    assert len(docs) == 1, f"{name} has {len(docs)} documents; burst submits exactly one task"
    task = docs[0]
    assert "run" in task, f"{name} must be an executable task"
    # A SkyPilot pipeline header (`execution:`) is the marker of a multi-stage document.
    assert "execution" not in task, f"{name} looks like a pipeline, not a burst task"


@pytest.mark.parametrize("name", sorted(PINNED_EXAMPLES))
def test_example_keeps_its_substitution_placeholders(name: str) -> None:
    """Concrete infra ids must never be committed; burst fills these in from `--var`."""

    from npa.burst.core import _unresolved_task_placeholders

    task = _documents(EXAMPLES_DIR / name)[0]
    unresolved = _unresolved_task_placeholders(task)

    assert unresolved, f"{name} has no ${{VAR}} placeholders left; did a real value get baked in?"


@pytest.mark.parametrize("name", sorted(PINNED_EXAMPLES))
def test_burst_accepts_the_example_once_vars_are_supplied(name: str) -> None:
    """`submit_yaml` validates before launching, so acceptance is checkable offline."""

    from npa.burst.core import (
        _load_burst_yaml_documents,
        _replace_task_placeholders,
        _single_task_from_documents,
        _unresolved_task_placeholders,
        _validate_burst_yaml_runtime,
    )

    source = EXAMPLES_DIR / name
    task = _single_task_from_documents(_load_burst_yaml_documents(source), source)
    overrides = {
        "NPA_RUN_ID": "burst-guardrail",
        "ISAAC_LAB_IMAGE": "example-registry/npa-isaac-lab:2.3.2.post1",
        "NPA_OUTPUT_URI": "s3://example-bucket/burst/burst-guardrail/",
    }
    task = _replace_task_placeholders(task, overrides)
    task.setdefault("envs", {}).update(overrides)

    assert not _unresolved_task_placeholders(task), task.get("envs")
    # Raises BurstConfigError if the task is not submittable.
    _validate_burst_yaml_runtime(task, source)


def test_examples_are_not_in_the_retiring_workflow_catalog() -> None:
    catalog = REPO_ROOT / "npa" / "src" / "npa" / "workflows" / "skypilot"

    for name in PINNED_EXAMPLES:
        assert not (catalog / name).exists(), f"{name} came back to the retiring catalog"
