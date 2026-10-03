"""Guard reusable operator-owned access decisions for gated runtime fetches."""

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
SKILL = REPO_ROOT / "skills/atomic/access-approval/SKILL.md"
INDEX = REPO_ROOT / "skills/index.yaml"
FIRST_RUN = REPO_ROOT / "skills/workflows/first-run-setup/SKILL.md"
CREDENTIAL_PREFLIGHT = REPO_ROOT / "npa/src/npa/workflows/credential_preflight.py"


def _normalized() -> str:
    return " ".join(SKILL.read_text(encoding="utf-8").split()).lower()


def test_access_approval_is_discoverable_for_reusable_operator_access() -> None:
    index = yaml.safe_load(INDEX.read_text(encoding="utf-8"))
    entry = next(item for item in index["skills"] if item["name"] == "access-approval")

    assert REPO_ROOT / entry["path"] == SKILL
    trigger = entry["when_to_use"].lower()
    for phrase in (
        "operator-owned",
        "payload probe",
        "operationally ready",
        "run-scoped use declaration",
        "duplicate npa acceptance prompts",
    ):
        assert phrase in trigger, phrase


def test_ready_is_sufficient_to_proceed_but_not_a_legal_conclusion() -> None:
    text = _normalized()
    for phrase in (
        "operationally sufficient for npa to proceed with that exact fetch",
        "never legal acceptance, proof of compliance, or redistribution rights",
        "token presence alone proves identity only",
        "tokens inherit account access",
        "do not accept terms or own licences",
        "operator who supplies the credential and invokes the fetch is responsible",
        "exact field-of-use statement once",
        "one bounded manager task/run id",
        "child solutions may reference the same scope record",
        "do not ask again per image",
        "expires with that task/run",
        "never a global or permanent declaration",
        "re-probe access when the provider",
        "reopen the use question only when the operator changes scope",
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


def test_setup_guidance_describes_access_without_token_acceptance_language() -> None:
    first_run = _normalized_path(FIRST_RUN)
    preflight = _normalized_path(CREDENTIAL_PREFLIGHT)

    assert "credential did not have exact artifact access" in first_run
    assert "optionally authenticated" in preflight
    assert "token was never accepted" not in first_run
    assert "optionally accepted" not in preflight


def _normalized_path(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split()).lower()
