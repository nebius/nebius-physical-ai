"""Guardrail: every readiness claim stays bound to its adjacent workflow bytes."""

from pathlib import Path

from npa.orchestration.npa_workflow.readiness import (
    validate_repository_readiness_records,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_all_workflow_readiness_records_are_valid_and_digest_bound() -> None:
    records = validate_repository_readiness_records(REPO_ROOT / "workflows")

    assert records
