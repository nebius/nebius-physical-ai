from __future__ import annotations

from pathlib import Path

from npa.deploy.safety import (
    PlanDecision,
    analyze_terraform_plan,
    format_replacement_required_error,
)


FIXTURES = Path(__file__).parent / "fixtures" / "terraform_plans"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


def test_fresh_create_plan_is_allowed_for_new_alias() -> None:
    analysis = analyze_terraform_plan(_fixture("fresh_create.txt"), existing_state=False)

    assert analysis.decision == PlanDecision.FRESH_CREATE
    assert analysis.add_count == 9
    assert analysis.destroy_count == 0


def test_env_only_change_plan_is_in_place_update() -> None:
    analysis = analyze_terraform_plan(_fixture("env_only_change.txt"), existing_state=True)

    assert analysis.decision == PlanDecision.IN_PLACE_UPDATE
    assert analysis.change_count == 1
    assert analysis.replacement_resources == ()


def test_gpu_type_change_requires_replacement() -> None:
    analysis = analyze_terraform_plan(
        _fixture("gpu_type_change_full_replace.txt"), existing_state=True
    )

    assert analysis.decision == PlanDecision.REPLACEMENT_REQUIRED
    addresses = {resource.address for resource in analysis.replacement_resources}
    assert "nebius_compute_v1_disk.boot" in addresses
    assert "nebius_compute_v1_instance.workbench" in addresses


def test_unclassified_destroy_defaults_to_replacement_required() -> None:
    analysis = analyze_terraform_plan(
        _fixture("unclassified_destroy.txt"), existing_state=True
    )

    assert analysis.decision == PlanDecision.REPLACEMENT_REQUIRED
    assert analysis.replacement_resources[0].address == "random_id.workbench_suffix"
    assert "unclassified" in analysis.replacement_resources[0].reason


def test_no_changes_plan_is_noop() -> None:
    analysis = analyze_terraform_plan(_fixture("no_changes.txt"), existing_state=True)

    assert analysis.decision == PlanDecision.NO_CHANGES


def test_replacement_error_names_resources() -> None:
    analysis = analyze_terraform_plan(
        _fixture("gpu_type_change_full_replace.txt"), existing_state=True
    )

    message = format_replacement_required_error(analysis)

    assert "nebius_compute_v1_instance.workbench" in message
    assert "--replace --yes" in message
