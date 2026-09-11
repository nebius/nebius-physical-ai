"""Guard the reusable runtime-fetch onboarding decision and proof contract."""

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
SKILL = REPO_ROOT / "skills/workflows/runtime-fetch-onboard/SKILL.md"
CONTRACT = SKILL.parent / "references/onboarding-contract.md"
INDEX = REPO_ROOT / "skills/index.yaml"


def _normalized(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split()).lower()


def test_runtime_fetch_onboard_is_discoverable_for_legal_packaging_blocks() -> None:
    index = yaml.safe_load(INDEX.read_text(encoding="utf-8"))
    entry = next(
        item for item in index["skills"] if item["name"] == "runtime-fetch-onboard"
    )

    assert entry["category"] == "workflows"
    assert REPO_ROOT / entry["path"] == SKILL
    trigger = entry["when_to_use"].lower()
    for phrase in ("legal", "model weights", "runtime"):
        assert phrase in trigger


def test_skill_uses_runtime_fetch_as_remediation_not_a_license_bypass() -> None:
    text = _normalized(SKILL)
    required = (
        "runtime fetch is a delivery design, not a license bypass",
        "do not reject an otherwise viable solution",
        "weights/checkpoints only",
        "source as well as weights",
        "sdk or runtime",
        "base image",
        "outputs, service use, or field of use",
        "no verified right to fetch or use",
        "block and escalate",
        "defer that capability",
    )
    for phrase in required:
        assert phrase in text, phrase


def test_skill_keeps_restricted_bytes_and_secrets_out_of_images() -> None:
    text = _normalized(SKILL)
    required = (
        "digest-pinned redistributable base",
        "leave artifact/cache directories empty",
        "exact immutable artifact",
        "download to a unique temporary path",
        "atomically publish a ready marker",
        "runtime secret plumbing",
        "never put credentials",
        "a token normally proves access only",
        "demonstrably records acceptance",
        "do not require a credential for a genuinely public, anonymous artifact",
        "do not invent a generic `accept_terms=yes` variable",
        "operator-owned, access-restricted cache",
        "default to node-local ephemeral caching",
        "mount a completed durable cache read-only",
        "never copy it into another image",
    )
    for phrase in required:
        assert phrase in text, phrase


def test_skill_requires_negative_positive_and_built_byte_proof() -> None:
    text = _normalized(SKILL)
    required = (
        "**byte absence:**",
        "files, layers, history, oci configuration, sbom, and caches",
        "**negative gate:**",
        "leaves the cache empty",
        "mutation-test independent gates",
        "**positive fetch:**",
        "**real capability:**",
        "**cache behavior:**",
        "**no secret leakage:**",
        "operator-controlled private registry",
        "official public development push is already publication",
        "fail-closed gate",
    )
    for phrase in required:
        assert phrase in text, phrase


def test_copyable_contract_covers_all_artifact_boundaries_and_results() -> None:
    text = _normalized(CONTRACT)
    for phrase in (
        "packaging shape",
        "image redistribution",
        "source",
        "baked runtime",
        "weights",
        "dataset/assets",
        "runtime cache",
        "outputs",
        "authorization source",
        "acceptance mechanism",
        "cache owner/access policy",
        "cache reuse permission",
        "missing-access refusal before network",
        "private staging digest before public publication",
        "exact runtime fetch and checksum verification",
        "accepted:",
        "deferred:",
        "rejected:",
        "human/vendor decision required:",
    ):
        assert phrase in text, phrase


def test_primary_onboarding_skills_route_to_runtime_fetch() -> None:
    for relative in (
        "skills/atomic/solution-licensing/SKILL.md",
        "skills/workflows/byof-onboard/SKILL.md",
    ):
        assert "skills/workflows/runtime-fetch-onboard/SKILL.md" in (
            REPO_ROOT / relative
        ).read_text(encoding="utf-8")
