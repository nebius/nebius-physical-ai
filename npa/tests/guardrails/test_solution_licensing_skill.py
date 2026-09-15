"""Guard model-weight packaging and runtime-cache licensing guidance."""

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
SKILL = REPO_ROOT / "skills/atomic/solution-licensing/SKILL.md"
INDEX = REPO_ROOT / "skills/index.yaml"


def _skill_text() -> str:
    return SKILL.read_text(encoding="utf-8")


def _normalized_skill_text() -> str:
    return " ".join(_skill_text().split()).lower()


def test_solution_licensing_is_discoverable_for_weights_and_runtime_caches() -> None:
    index = yaml.safe_load(INDEX.read_text(encoding="utf-8"))
    entry = next(
        item for item in index["skills"] if item["name"] == "solution-licensing"
    )

    assert REPO_ROOT / entry["path"] == SKILL
    trigger = entry["when_to_use"].lower()
    assert "model weights" in trigger
    assert "runtime cache" in trigger


def test_skill_classifies_every_artifact_boundary_separately() -> None:
    text = _skill_text()

    for boundary in (
        "**Source**",
        "**Baked runtime**",
        "**Weights**",
        "**Datasets**",
        "**Runtime caches**",
    ):
        assert boundary in text


def test_skill_keeps_gated_weights_and_acceptance_out_of_image_access() -> None:
    text = _normalized_skill_text()
    required = (
        "never bake gated or redistribution-restricted weights",
        "exact immutable revision",
        "do not require a token for genuinely public",
        "token presence alone proves neither access nor acceptance",
        "operationally sufficient for npa to fetch that exact artifact",
        "not legal acceptance, proof of compliance, or permission to redistribute",
    )
    for phrase in required:
        assert phrase in text, phrase


def test_default_policy_reuses_scope_and_separates_access_from_publication() -> None:
    text = _normalized_skill_text()
    for phrase in (
        "### default operator-responsibility policy",
        "operator initiates runtime fetch with their own credential",
        "exact operator statement once",
        "one bounded manager task/run id",
        "record exactly `noncommercial`",
        "without another per-image question",
        "never global or permanent",
        "keep entitlement provider- and artifact-scoped",
        "classify the public image from its built bytes",
        "every baked byte has verified redistribution rights",
        "all `secure-image-build` publication gates pass",
        "ready` grants no redistribution rights",
        "do not invent output or service restrictions",
    ):
        assert phrase in text, phrase

    for unsafe in (
        "the token accepts terms",
        "token proves compliance",
        "noncommercial clears every license",
        "runtime fetch makes use legal",
        "ready permits redistribution",
        "one declaration applies globally",
    ):
        assert unsafe not in text, unsafe


def test_skill_distinguishes_cache_tiers_and_durable_population_contract() -> None:
    text = _normalized_skill_text()
    required = (
        "**image layer**",
        "**node-local ephemeral**",
        "**shared durable pvc/object storage**",
        "does not make a runtime weight cache durable",
        "wire the pvc or object-storage location explicitly",
        "exact immutable revision or digest",
        "populate safely under concurrency",
        "atomically publish",
        "never write tokens into cache metadata",
        "mount the immutable cache read-only",
        "never `copy` that cache",
    )
    for phrase in required:
        assert phrase in text, phrase
