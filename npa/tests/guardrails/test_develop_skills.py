"""Guardrail: the development skills must describe a repo that still exists.

The skills that tell an agent how to change this repo are dense with concrete
file paths and guardrail names. That is what makes them useful and also what
makes them rot: a rename lands in one PR, and the runbook keeps pointing at the
old path until someone follows it and wastes a cycle. Prose cannot be trusted to
stay true on its own, so the claims that *can* be checked mechanically are.

Path checking runs at two strictnesses, because a skill may legitimately
describe *another* repo's layout. Cosmos3 navigation cites the upstream
framework's ``docs/inference.md``; Open Dreamer cites its ``scripts/``. Those
directory names collide with ours, so only unambiguously first-party prefixes
are checked everywhere, while the development skills — which exist to point at
this repo and nothing else — are held to every path they name.

The failure index has its own invariant: it must stay complete, so that adding a
guardrail also means documenting how to satisfy it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SKILLS_ROOT = REPO_ROOT / "skills"
GUARDRAIL_DIR = REPO_ROOT / "npa/tests/guardrails"
FAILURE_INDEX = SKILLS_ROOT / "atomic/guardrail-failures/SKILL.md"

#: Skills whose whole purpose is to navigate this repository. They get the
#: strict check: every path they name, under any top-level directory.
DEVELOPMENT_SKILLS = (
    "skills/workflows/add-workbench-tool/SKILL.md",
    "skills/atomic/npa-cli-conventions/SKILL.md",
    "skills/atomic/toolref-argv-contract/SKILL.md",
    "skills/atomic/guardrail-failures/SKILL.md",
    "skills/atomic/pre-pr-validation/SKILL.md",
    "skills/tools/workbench-tool/SKILL.md",
)

#: Checked in every skill. ``npa/`` and ``.github/`` cannot belong to an upstream
#: project we document, and a ``skills/**/SKILL.md`` reference is always ours.
FIRST_PARTY_PATTERN = re.compile(
    r"^(?:npa/|\.github/|skills/[\w.-]+/[\w.-]+/SKILL\.md$)[\w./<>{}$*-]*$"
)
#: Checked only in DEVELOPMENT_SKILLS, where a bare ``docs/`` or ``scripts/``
#: path is unambiguous.
ANY_REPO_PATH_PATTERN = re.compile(
    r"^(?:npa|docs|skills|scripts|deploy|\.github)/[\w./<>{}$*-]+$"
)

#: Template syntax, not a reference to a file that should exist.
PLACEHOLDER_MARKERS = ("<", ">", "{{", "${", "*")

#: Paths that are correct to name and correct to be absent.
#:
#: ``npa/.venv`` is a developer's local virtualenv: present on a working machine,
#: never in a fresh checkout, and the documented interpreter for every command.
#: ``npa/src/npa/workflows/skypilot`` is the retired raw SkyPilot catalog, named
#: only so the failure index can explain the guard that keeps it deleted.
EXPECTED_ABSENT = ("npa/.venv", "npa/src/npa/workflows/skypilot")


def _skill_files() -> list[Path]:
    return sorted(SKILLS_ROOT.glob("*/*/SKILL.md"))


def _referenced_paths(text: str, pattern: re.Pattern[str]) -> list[str]:
    found = [
        token
        for token in re.findall(r"`([^`\n]+)`", text)
        if pattern.match(token)
        and not any(marker in token for marker in PLACEHOLDER_MARKERS)
        and not token.startswith(EXPECTED_ABSENT)
    ]
    return list(dict.fromkeys(found))


def _missing_paths(path: Path, pattern: re.Pattern[str]) -> list[str]:
    return [
        reference
        for reference in _referenced_paths(path.read_text(encoding="utf-8"), pattern)
        if not (REPO_ROOT / reference).exists()
    ]


@pytest.mark.parametrize(
    "skill", _skill_files(), ids=lambda path: str(path.relative_to(SKILLS_ROOT))
)
def test_every_skill_resolves_its_first_party_paths(skill: Path) -> None:
    missing = _missing_paths(skill, FIRST_PARTY_PATTERN)
    assert not missing, (
        f"{skill.relative_to(REPO_ROOT)} names repo paths that do not exist: "
        f"{missing}. Repoint the reference or remove it."
    )


@pytest.mark.parametrize("relative", DEVELOPMENT_SKILLS)
def test_development_skills_resolve_every_path_they_name(relative: str) -> None:
    skill = REPO_ROOT / relative
    assert skill.is_file(), relative

    missing = _missing_paths(skill, ANY_REPO_PATH_PATTERN)
    assert not missing, (
        f"{relative} is a repository navigation runbook, so every path it names "
        f"must resolve; these do not: {missing}."
    )


def test_the_path_guard_scans_a_real_corpus() -> None:
    """A guard that silently matches nothing would pass forever."""

    skills = _skill_files()
    references = [
        reference
        for skill in skills
        for reference in _referenced_paths(
            skill.read_text(encoding="utf-8"), FIRST_PARTY_PATTERN
        )
    ]

    assert len(skills) >= 25
    assert len(references) >= 100


def test_path_guard_detects_a_missing_reference(tmp_path: Path) -> None:
    victim = tmp_path / "SKILL.md"
    victim.write_text("see `npa/src/npa/definitely_missing.py`\n", encoding="utf-8")

    assert _missing_paths(victim, FIRST_PARTY_PATTERN) == [
        "npa/src/npa/definitely_missing.py"
    ]


def test_path_guard_ignores_placeholders_and_prose(tmp_path: Path) -> None:
    victim = tmp_path / "SKILL.md"
    victim.write_text(
        "`npa/src/npa/cli/workbench/<tool_snake>.py`\n"
        "`npa workbench lancedb create-mv`\n"
        "`~/.npa/credentials.yaml`\n"
        "`{{config.input_uri}}`\n",
        encoding="utf-8",
    )

    assert _referenced_paths(victim.read_text(encoding="utf-8"), ANY_REPO_PATH_PATTERN) == []


def test_first_party_pattern_leaves_upstream_layouts_alone(tmp_path: Path) -> None:
    """Cosmos3 and Open Dreamer skills document another repo's directories."""

    victim = tmp_path / "SKILL.md"
    victim.write_text(
        "`docs/inference.md`\n`scripts/train_tokenizer.py`\n", encoding="utf-8"
    )

    assert _referenced_paths(victim.read_text(encoding="utf-8"), FIRST_PARTY_PATTERN) == []
    assert len(
        _referenced_paths(victim.read_text(encoding="utf-8"), ANY_REPO_PATH_PATTERN)
    ) == 2


def test_first_party_pattern_still_checks_cross_skill_links(tmp_path: Path) -> None:
    victim = tmp_path / "SKILL.md"
    victim.write_text("`skills/atomic/no-such-skill/SKILL.md`\n", encoding="utf-8")

    assert _missing_paths(victim, FIRST_PARTY_PATTERN) == [
        "skills/atomic/no-such-skill/SKILL.md"
    ]


def test_path_guard_exempts_paths_that_are_meant_to_be_absent(tmp_path: Path) -> None:
    """The local venv and the retired catalog are named on purpose."""

    victim = tmp_path / "SKILL.md"
    victim.write_text(
        "`npa/.venv/bin/python`\n`npa/src/npa/workflows/skypilot/`\n",
        encoding="utf-8",
    )

    assert _referenced_paths(victim.read_text(encoding="utf-8"), ANY_REPO_PATH_PATTERN) == []
    assert not (REPO_ROOT / "npa/src/npa/workflows/skypilot").exists()


def _named_tests(text: str) -> set[str]:
    return set(re.findall(r"\btest_[a-z0-9_]+\b", text))


def test_failure_index_documents_every_guardrail() -> None:
    """Adding a guardrail means telling contributors how to satisfy it."""

    stems = {path.stem for path in GUARDRAIL_DIR.glob("test_*.py")}
    undocumented = sorted(
        stems - _named_tests(FAILURE_INDEX.read_text(encoding="utf-8"))
    )

    assert not undocumented, (
        "these guardrails are not in the failure index, so a contributor who "
        f"hits one has no documented fix: {undocumented}. Add a row to "
        f"{FAILURE_INDEX.relative_to(REPO_ROOT)}."
    )


def test_failure_index_names_no_test_that_disappeared() -> None:
    real = {path.stem for path in (REPO_ROOT / "npa/tests").rglob("test_*.py")}
    stale = sorted(_named_tests(FAILURE_INDEX.read_text(encoding="utf-8")) - real)

    assert not stale, (
        f"the failure index names tests that no longer exist: {stale}. "
        "Remove the row or repoint it."
    )
