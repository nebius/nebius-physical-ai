"""Shipped workflow/example YAML references must resolve to real files.

The raw SkyPilot catalog retirement proved that deleting a template is easier
than finding every skill, document, script, or code snippet that still names it.
Strict npa.workflow metadata already rejects the retired ``skypilotTwin`` fields;
this guard covers the broader file-reference class instead.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

# Historical records are allowed to name paths that no longer exist.
HISTORY_FILES = {"CHANGELOG.md", "DESIGN.md", "EVIDENCE.md", "PLAN.md"}

# Shipped prose, specs, scripts, and package code can all carry operator-facing
# paths. Tests are omitted because their frozen fixtures intentionally model old
# files and have their own guards. Top-level Markdown is added separately because
# repository entry points such as README.md and CONTRIBUTING.md are operator-facing.
SEARCH_ROOTS = (
    "skills",
    "docs",
    ".github",
    "npa/workflows",
    "npa/src/npa",
    "npa/scripts",
    "npa/docs",
    "scripts",
)
SEARCH_SUFFIXES = {".md", ".py", ".sh", ".toml", ".yaml", ".yml"}

# The declarative catalog plus the guarded raw-task/resource-profile locations
# that remain after retirement: burst, NuRec, and BYOF profiles.
FULL_WORKFLOW_PATH = re.compile(
    r"npa/(?:workflows|src/npa/(?:burst/examples|workbench/nurec/examples|"
    r"workflows/byof/profiles))/[A-Za-z0-9._/{}$<>*-]+\.ya?ml"
)
WORKFLOW_SHORTHAND = re.compile(
    r"(?<![A-Za-z0-9._/-])npa-workflows/[A-Za-z0-9._/{}$<>*-]+\.ya?ml"
)
CATALOG_ROOT = "npa/workflows/workbench/npa-workflows/"

# Only explicit template syntax is exempt. Literal names such as
# ``example-training.yaml`` or ``your-data.yaml`` may be real shipped files and
# must not bypass the existence check merely because of their prose-like names.
PLACEHOLDER_MARKERS = ("<", "{{", "${", "*")


def _candidate_files() -> list[Path]:
    files = [
        path
        for path in REPO_ROOT.glob("*.md")
        if path.is_file() and path.name not in HISTORY_FILES
    ]
    for root in SEARCH_ROOTS:
        base = REPO_ROOT / root
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix not in SEARCH_SUFFIXES:
                continue
            if path.name in HISTORY_FILES or "__pycache__" in path.parts:
                continue
            files.append(path)
    return sorted(set(files))


def _normalize_workflow_reference(reference: str) -> str:
    if reference.startswith("npa-workflows/"):
        return CATALOG_ROOT + reference.removeprefix("npa-workflows/")
    return reference


def _workflow_references_in(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):  # pragma: no cover - unreadable/binary
        return []
    matches = FULL_WORKFLOW_PATH.findall(text) + WORKFLOW_SHORTHAND.findall(text)
    references = (
        _normalize_workflow_reference(match)
        for match in matches
        if not any(marker in match for marker in PLACEHOLDER_MARKERS)
    )
    return list(dict.fromkeys(references))


def _dangling_in(path: Path) -> list[str]:
    return [
        reference
        for reference in _workflow_references_in(path)
        if not (REPO_ROOT / reference).is_file()
    ]


@pytest.mark.parametrize(
    "path", _candidate_files(), ids=lambda path: str(path.relative_to(REPO_ROOT))
)
def test_no_shipped_file_points_at_a_missing_workflow(path: Path) -> None:
    dangling = _dangling_in(path)
    assert not dangling, (
        f"{path.relative_to(REPO_ROOT)} names workflow YAML(s) that do not exist: "
        f"{dangling}. Repoint the reference at the current spec/example or remove it."
    )


@pytest.mark.parametrize(
    "missing",
    [
        "npa/workflows/workbench/npa-workflows/definitely-missing.yaml",
        "npa/src/npa/burst/examples/definitely-missing.yaml",
        "npa/src/npa/workbench/nurec/examples/definitely-missing.yaml",
        "npa/src/npa/workflows/byof/profiles/definitely-missing.yaml",
        "npa-workflows/definitely-missing.yaml",
    ],
)
def test_guard_detects_missing_paths_in_every_guarded_location(
    tmp_path: Path, missing: str
) -> None:
    victim = tmp_path / "doc.md"
    victim.write_text(f"see {missing}\n", encoding="utf-8")
    expected = _normalize_workflow_reference(missing)
    assert _dangling_in(victim) == [expected]


def test_guard_ignores_placeholder_paths(tmp_path: Path) -> None:
    victim = tmp_path / "doc.md"
    victim.write_text(
        "npa/workflows/workbench/npa-workflows/<your-spec>.yaml\n",
        encoding="utf-8",
    )
    assert _dangling_in(victim) == []


def test_guard_normalizes_catalog_shorthand_before_checking(tmp_path: Path) -> None:
    victim = tmp_path / "doc.md"
    victim.write_text(
        "npa-workflows/vlm-eval-single.yaml\n"
        "npa-workflows/definitely-missing.yaml\n",
        encoding="utf-8",
    )

    assert _workflow_references_in(victim) == [
        f"{CATALOG_ROOT}vlm-eval-single.yaml",
        f"{CATALOG_ROOT}definitely-missing.yaml",
    ]
    assert _dangling_in(victim) == [f"{CATALOG_ROOT}definitely-missing.yaml"]


@pytest.mark.parametrize(
    "reference",
    [
        "npa/workflows/workbench/npa-workflows/example-missing.yaml",
        "npa-workflows/example-missing.yaml",
        "npa-workflows/your-data-missing.yaml",
    ],
)
def test_guard_does_not_exempt_literal_example_names(
    tmp_path: Path, reference: str
) -> None:
    victim = tmp_path / "doc.md"
    victim.write_text(f"see {reference}\n", encoding="utf-8")

    assert _dangling_in(victim) == [_normalize_workflow_reference(reference)]


def test_guard_scans_top_level_operator_markdown() -> None:
    candidates = set(_candidate_files())

    assert REPO_ROOT / "README.md" in candidates
    assert REPO_ROOT / "CONTRIBUTING.md" in candidates


def test_guard_scans_npa_operator_roots_without_virtualenv() -> None:
    candidates = set(_candidate_files())

    for root in ("npa/scripts", "npa/docs"):
        base = REPO_ROOT / root
        shipped = {
            path
            for path in base.rglob("*")
            if path.is_file() and path.suffix in SEARCH_SUFFIXES
        }
        assert shipped, f"expected a non-empty shipped corpus under {root}"
        assert shipped <= candidates

    assert not any(".venv" in path.relative_to(REPO_ROOT).parts for path in candidates)


def test_guard_scans_a_non_empty_corpus_with_real_references() -> None:
    candidates = _candidate_files()
    references = [
        reference for path in candidates for reference in _workflow_references_in(path)
    ]

    assert candidates, "workflow reference guard found no shipped files to scan"
    assert references, "workflow reference guard found no workflow paths to check"
