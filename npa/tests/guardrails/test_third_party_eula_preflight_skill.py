"""Discovery and consent guardrails for the reusable third-party EULA preflight."""

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
SKILL = REPO_ROOT / "skills/atomic/third-party-eula-preflight/SKILL.md"
INDEX = REPO_ROOT / "skills/index.yaml"
LINKED_SKILLS = (
    REPO_ROOT / "skills/workflows/sim2real-operate/SKILL.md",
    REPO_ROOT / "skills/tools/isaac-lab/SKILL.md",
    REPO_ROOT / "skills/atomic/solution-licensing/SKILL.md",
)
EULA_VALUE = "ACCEPT_EULA=Y"


def test_eula_preflight_is_canonical_and_discoverable() -> None:
    index = yaml.safe_load(INDEX.read_text(encoding="utf-8"))
    entry = next(
        item for item in index["skills"] if item["name"] == "third-party-eula-preflight"
    )

    assert REPO_ROOT / entry["path"] == SKILL
    assert entry["category"] == "atomic"
    assert "before provisioning" in entry["when_to_use"].lower()


def test_eula_preflight_documents_scoped_default_and_explicit_opt_out() -> None:
    text = SKILL.read_text(encoding="utf-8")
    required = (
        "official terms links",
        "defaults vendor acceptance on",
        "explicit opt-out",
        "fail before provisioning",
        "Do not reuse this",
        "Never default optional privacy",
        "Do not store",
        "secret values or unnecessary personal data",
    )
    for phrase in required:
        assert phrase.lower() in text.lower(), phrase
    assert EULA_VALUE in text


def test_operational_skills_link_the_preflight() -> None:
    for path in LINKED_SKILLS:
        assert "skills/atomic/third-party-eula-preflight/SKILL.md" in path.read_text(
            encoding="utf-8"
        ), path
