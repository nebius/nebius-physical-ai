"""Locations of the supported ``npa.workflow`` blueprint catalog.

The source catalog lives in repo-root ``workflows/main``, ``workflows/testing``,
and ``workflows/partners/<partner>``. Wheels include the same directories so
discovery and canonical workflow consumers also work without a source checkout.
Guarded raw SkyPilot examples and resource profiles are separate from this catalog.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from npa.workflow_build import catalog_directories

# blueprints.py -> npa_workflow -> orchestration -> npa -> src -> npa -> <repo root>
_REPO_ROOT = Path(__file__).resolve().parents[5]
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def npa_workflow_spec_dirs() -> tuple[Path, ...]:
    """Return the existing source or installed catalog directories."""

    # Generated container copies can outlive a source rename or deletion. A
    # checkout's catalog remains authoritative until the next build stages it.
    for catalog in (_REPO_ROOT / "workflows", _PACKAGE_ROOT / "workflows"):
        directories = tuple(
            path for path in catalog_directories(catalog) if path.is_dir()
        )
        if directories:
            return directories
    return ()


def iter_npa_workflow_specs() -> list[Path]:
    """All ``*.yaml`` blueprint specs across the roots, de-duplicated by name.

    Only supported catalog directories are scanned; nested examples and raw
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
