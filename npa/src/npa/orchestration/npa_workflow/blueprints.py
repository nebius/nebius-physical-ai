"""Locations of the supported ``npa.workflow`` blueprint catalog.

The source catalog lives in repo-root ``workflows/main`` and
``workflows/testing``. Wheels include the same directories as package data so
discovery and canonical workflow consumers also work without a source checkout.
Guarded raw SkyPilot examples and resource profiles are separate from this catalog.
"""

from __future__ import annotations

from pathlib import Path

import yaml

# blueprints.py -> npa_workflow -> orchestration -> npa -> src -> npa -> <repo root>
_REPO_ROOT = Path(__file__).resolve().parents[5]
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]

#: The first pair is the source catalog; the second pair is its installed fallback.
NPA_WORKFLOW_SPEC_DIRS: tuple[Path, ...] = (
    _REPO_ROOT / "workflows" / "main",
    _REPO_ROOT / "workflows" / "testing",
    _PACKAGE_ROOT / "workflows" / "main",
    _PACKAGE_ROOT / "workflows" / "testing",
)


def npa_workflow_spec_dirs() -> tuple[Path, ...]:
    """Return the existing source or installed catalog directories."""

    source_dirs = tuple(
        directory for directory in NPA_WORKFLOW_SPEC_DIRS[:2] if directory.is_dir()
    )
    # Generated container copies can outlive a source rename or deletion. A
    # checkout's catalog remains authoritative until the next build stages it.
    return source_dirs or tuple(
        directory for directory in NPA_WORKFLOW_SPEC_DIRS[2:] if directory.is_dir()
    )


def iter_npa_workflow_specs() -> list[Path]:
    """All ``*.yaml`` blueprint specs across the roots, de-duplicated by name.

    Only the two catalog directories are scanned; nested examples and raw
    SkyPilot/profile YAML homes are outside the supported declarative catalog.
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
