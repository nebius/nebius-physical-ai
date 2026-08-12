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
EULA_VARS = ("OMNI_KIT_ACCEPT_EULA", "ISAACSIM_ACCEPT_EULA")


def test_eula_preflight_is_canonical_and_discoverable() -> None:
    index = yaml.safe_load(INDEX.read_text(encoding="utf-8"))
    entry = next(item for item in index["skills"] if item["name"] == "third-party-eula-preflight")

    assert REPO_ROOT / entry["path"] == SKILL
    assert entry["category"] == "atomic"
    assert "before provisioning" in entry["when_to_use"].lower()


def test_eula_preflight_requires_scoped_explicit_consent_and_early_refusal() -> None:
    text = SKILL.read_text(encoding="utf-8")
    required = (
        "explicit operator",
        "official terms links",
        "fail before provisioning",
        "exact resume command",
        "Never precheck a box",
        "Do not reuse it",
        "Do not store",
        "secret values or unnecessary personal data",
    )
    for phrase in required:
        assert phrase.lower() in text.lower(), phrase
    for variable in EULA_VARS:
        assert f"{variable}=YES" in text


def test_operational_skills_link_the_preflight() -> None:
    for path in LINKED_SKILLS:
        assert "skills/atomic/third-party-eula-preflight/SKILL.md" in path.read_text(
            encoding="utf-8"
        ), path
