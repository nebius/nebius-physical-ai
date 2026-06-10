from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SKYPILOT_DIR = REPO_ROOT / "npa" / "workflows" / "workbench" / "skypilot"
README = SKYPILOT_DIR / "README.md"


def test_skypilot_readme_lists_every_yaml() -> None:
    text = README.read_text(encoding="utf-8")
    missing = [path.name for path in sorted(SKYPILOT_DIR.glob("*.yaml")) if f"`{path.name}`" not in text]

    assert not missing, "Missing SkyPilot README entries: " + ", ".join(missing)


def test_skypilot_readme_keeps_raw_run_caveats() -> None:
    text = README.read_text(encoding="utf-8")
    required_phrases = [
        "SkyPilot 0.12.2",
        "does not interpolate",
        "autodown",
        "sky down -y",
        "HF rights",
        "NGC entitlement",
    ]
    missing = [phrase for phrase in required_phrases if phrase not in text]

    assert not missing, "Missing SkyPilot README caveats: " + ", ".join(missing)


def test_skypilot_readme_uses_public_repo_hygiene() -> None:
    text = README.read_text(encoding="utf-8").lower()
    forbidden = [
        "89" ".169.",
        "dev" "-vm",
        "agent" "-vm",
        "nebius-ai" "-agents",
        "p" "m2",
        "n" "tfy",
    ]
    hits = [term for term in forbidden if term in text]

    assert not hits, "Private operational terms leaked into SkyPilot README: " + ", ".join(hits)
