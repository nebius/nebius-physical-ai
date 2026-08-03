"""Guardrail: NuRec's single-pod example stays an example, not a catalog.

#234 deliberately shipped both a single-pod SkyPilot task and a declarative
``npa.workflow`` twin. The twin is the workflow authoring surface; this file is
kept only because it exercises the NuRec tool with all stages sharing one pod's
``/tmp`` scratch space instead of using cross-pod S3 handoff.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = REPO_ROOT / "npa" / "src" / "npa" / "workbench" / "nurec" / "examples"
RETIRED_CATALOG = REPO_ROOT / "npa" / "src" / "npa" / "workflows" / "skypilot"

#: Pinned so a new raw SkyPilot task cannot appear here without answering why it
#: is not an ``npa.workflow/v0.0.1`` spec.
PINNED_EXAMPLES = frozenset({"nurec-reconstruct.yaml"})


def _documents(path: Path) -> list[dict]:
    return [doc for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")) if doc]


def test_example_set_is_pinned() -> None:
    found = {path.name for path in EXAMPLES.glob("*.yaml")}

    assert found == set(PINNED_EXAMPLES), (
        "NuRec examples changed. A multi-stage pipeline is a workflow: author an "
        "npa.workflow/v0.0.1 spec under npa/workflows/workbench/npa-workflows/ instead. "
        f"expected {sorted(PINNED_EXAMPLES)}, found {sorted(found)}"
    )


def test_the_directory_documents_the_boundary() -> None:
    readme = " ".join((EXAMPLES / "README.md").read_text(encoding="utf-8").split())

    for token in ("single-pod", "not a workflow authoring catalog", "#234", "npa.workflow"):
        assert token in readme, f"NuRec examples README should mention {token!r}"
    assert "One task per file" in readme


@pytest.mark.parametrize("name", sorted(PINNED_EXAMPLES))
def test_example_is_a_single_skypilot_task(name: str) -> None:
    docs = _documents(EXAMPLES / name)

    assert len(docs) == 1, f"{name} has {len(docs)} documents; this example submits one task"
    task = docs[0]
    assert "run" in task, f"{name} must be an executable SkyPilot task"
    assert "execution" not in task, f"{name} looks like a pipeline, not a single-pod task"


@pytest.mark.parametrize("name", sorted(PINNED_EXAMPLES))
def test_example_keeps_substitution_placeholders(name: str) -> None:
    text = (EXAMPLES / name).read_text(encoding="utf-8")

    for placeholder in (
        "${NPA_NUREC_RUN_ID}",
        "${NPA_NUREC_RUN_URI}",
        "${NPA_SRC_S3_URI}",
        "${AWS_ACCESS_KEY_ID}",
        "${AWS_SECRET_ACCESS_KEY}",
        "${HF_TOKEN}",
        "${NGC_API_KEY}",
    ):
        assert placeholder in text, f"{name} lost placeholder {placeholder}"


def test_example_is_not_in_the_retiring_workflow_catalog() -> None:
    for name in PINNED_EXAMPLES:
        assert not (RETIRED_CATALOG / name).exists(), f"{name} came back to the retiring catalog"
