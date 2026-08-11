"""Filesystem locations of the checked-in ``npa.workflow`` blueprint specs.

These YAMLs are discovered from a source checkout. The prominent Sim2Real
direct-runbook is also force-included in the wheel. The flagship Physical AI
Data Factory spec stays at the top of ``npa/workflows/`` in the repo; the rest
of the declarative shown catalog is under
``npa/workflows/workbench/npa-workflows/``.

This module is the single source of truth for both roots so spec discovery,
the smoke/guardrail tests, and the live-submit matrix stay in sync when a spec
is promoted out of the catalog directory. When installed as a wheel (no repo
tree) the directories simply do not exist and the helpers degrade to empty —
only source-checkout callers (tests, the operator runner) rely on them.
"""

from __future__ import annotations

from pathlib import Path

import yaml

# blueprints.py -> npa_workflow -> orchestration -> npa -> src -> npa -> <repo root>
_REPO_ROOT = Path(__file__).resolve().parents[5]

#: Ordered spec roots. The promoted top-level directory is searched first so a
#: name that exists in both wins from the prominent location.
NPA_WORKFLOW_SPEC_DIRS: tuple[Path, ...] = (
    _REPO_ROOT / "npa" / "workflows",
    _REPO_ROOT / "npa" / "workflows" / "workbench" / "npa-workflows",
)


def npa_workflow_spec_dirs() -> tuple[Path, ...]:
    """Return the existing spec roots (skips missing dirs, e.g. wheel installs)."""

    return tuple(
        directory for directory in NPA_WORKFLOW_SPEC_DIRS if directory.is_dir()
    )


def iter_npa_workflow_specs() -> list[Path]:
    """All ``*.yaml`` blueprint specs across the roots, de-duplicated by name.

    Non-recursive per root, so the ``workbench/`` subtree under the top-level
    ``npa/workflows`` directory is not double-counted.
    """

    seen: dict[str, Path] = {}
    for directory in npa_workflow_spec_dirs():
        for path in sorted(directory.glob("*.yaml")):
            if _is_npa_workflow_spec(path):
                seen.setdefault(path.name, path)
    return [seen[name] for name in sorted(seen)]


def resolve_npa_workflow_spec(name: str) -> Path | None:
    """Resolve a blueprint spec by file name across the roots, or ``None``."""

    for directory in npa_workflow_spec_dirs():
        candidate = directory / name
        if candidate.is_file() and _is_npa_workflow_spec(candidate):
            return candidate
    return None


def _is_npa_workflow_spec(path: Path) -> bool:
    """Keep special direct-runbook YAMLs out of the declarative spec catalog."""

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return False
    return isinstance(payload, dict) and str(payload.get("apiVersion", "")).startswith(
        "npa.workflow/"
    )
