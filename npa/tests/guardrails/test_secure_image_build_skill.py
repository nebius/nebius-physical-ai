from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SKILL = ROOT / "skills" / "atomic" / "secure-image-build" / "SKILL.md"


def test_secure_image_build_skill_locks_public_development_sequence() -> None:
    text = SKILL.read_text(encoding="utf-8")
    for required in (
        "dev-<full-git-sha>",
        "redistribution: public",
        "Hard-refuse `restricted`",
        "layers, history, and OCI config",
        "non-root runtime",
        "bootstrap contract",
        "SBOM",
        "vulnerability",
        "provenance",
        "anonymous pull",
        "physical GPU",
        "exact digest identity",
        "deletion cannot revoke prior downloads",
    ):
        assert required in text


def test_secure_image_build_skill_delegates_to_existing_procedures() -> None:
    text = SKILL.read_text(encoding="utf-8")
    for required in (
        "skills/atomic/solution-licensing/SKILL.md",
        "skills/atomic/build-and-push-image/SKILL.md",
        "skills/workflows/contribute-workbench-image/SKILL.md",
        "skills/atomic/third-party-eula-preflight/SKILL.md",
        "skills/atomic/testing-conventions/SKILL.md",
        "skills/atomic/gpu-selection/SKILL.md",
        "skills/atomic/submit-workflow/SKILL.md",
        "skills/atomic/protect-nebius-infra-details/SKILL.md",
        "docs/workbench/container-packaging.md",
    ):
        assert required in text
