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


def _section(text: str, heading: str, next_heading: str | None = None) -> str:
    """Return one Markdown section so shape-specific rules cannot bleed together."""
    start = text.index(heading)
    end = text.index(next_heading, start) if next_heading else len(text)
    return text[start:end]


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
        "documented product policy for non-secret acceptance controls",
        "isaac default and explicit opt-out",
        "never put credentials",
        "secret acceptance material",
        "a token normally proves access only",
        "demonstrably records acceptance",
        "do not require a credential for a genuinely public, anonymous artifact",
        "genuinely anonymous artifacts need no secret reference",
        "do not invent a generic `accept_terms=yes` variable",
        "do not add an npa-side eula or terms-acceptance boolean",
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
        "**applicable negative gate:**",
        "missing entitlement for a gated artifact",
        "explicit opt-out from a documented default-on policy such as isaac",
        "missing exact opt-in for a product-specific policy such as openpi",
        "genuinely anonymous artifact has no missing-access gate",
        "anonymous artifacts may run without credentials",
        "separate documented product-specific acceptance gate",
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
        "applicable refusal before network",
        "not applicable only for anonymous access with no documented acceptance gate",
        "private staging digest before public publication",
        "accepted:",
        "deferred:",
        "rejected:",
        "human/vendor decision required:",
        "operator-owned build-time credential source",
        "credential phase",
        "exact delivery and checksum verification",
    ):
        assert phrase in text, phrase


def test_skill_separates_runtime_fetch_from_operator_build_credentials() -> None:
    text = _normalized(SKILL)
    for phrase in (
        "for either runtime-fetch shape",
        "for the build-your-own shape",
        "builds directly into an operator-controlled private registry",
        "do not claim first-run fetch",
        "trusted operator build's secret mechanism",
        "build-only credentials never appear in the workflow",
    ):
        assert phrase in text, phrase


def test_proof_and_worksheet_requirements_are_packaging_shape_specific() -> None:
    skill = _normalized(SKILL)
    runtime_proof = _section(
        skill, "### runtime-fetch shapes only", "### build-your-own shape only"
    )
    build_proof = _section(
        skill, "### build-your-own shape only", "## deliverables"
    )
    for phrase in (
        "byte absence",
        "positive fetch",
        "cache behavior",
        "leaves the cache empty",
    ):
        assert phrase in runtime_proof, phrase
        assert phrase not in build_proof, phrase
    for phrase in (
        "restricted-input provenance",
        "resulting-byte inventory",
        "build-secret absence",
        "private-registry containment",
    ):
        assert phrase in build_proof, phrase
        assert phrase not in runtime_proof, phrase

    contract = _normalized(CONTRACT)
    runtime_delivery = _section(
        contract, "## runtime-fetch delivery only", "## build-your-own delivery only"
    )
    build_delivery = _section(
        contract, "## build-your-own delivery only", "## shared validation ledger"
    )
    for phrase in ("cache reuse permission", "temporary-download path"):
        assert phrase in runtime_delivery, phrase
        assert phrase not in build_delivery, phrase
    for phrase in (
        "restricted build inputs",
        "resulting image inventory",
        "private-registry containment",
    ):
        assert phrase in build_delivery, phrase
        assert phrase not in runtime_delivery, phrase

    assert "not applicable — build-your-own" in runtime_delivery
    assert "not applicable — runtime-fetch shape" in build_delivery
    runtime_cache_row = next(
        line
        for line in CONTRACT.read_text(encoding="utf-8").splitlines()
        if line.startswith("| Runtime cache |")
    ).lower()
    assert "not applicable for build-your-own without runtime fetch" in runtime_cache_row


def test_primary_onboarding_skills_route_to_runtime_fetch() -> None:
    for relative in (
        "skills/atomic/solution-licensing/SKILL.md",
        "skills/workflows/byof-onboard/SKILL.md",
    ):
        assert "skills/workflows/runtime-fetch-onboard/SKILL.md" in (
            REPO_ROOT / relative
        ).read_text(encoding="utf-8")
