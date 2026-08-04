"""Guardrail: the retired raw SkyPilot catalog README must not return."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SKYPILOT_DIR = REPO_ROOT / "npa" / "src" / "npa" / "workflows" / "skypilot"
README = SKYPILOT_DIR / "README.md"


def test_retired_skypilot_catalog_readme_is_gone() -> None:
    assert not README.exists(), "the retired raw SkyPilot catalog README came back"
