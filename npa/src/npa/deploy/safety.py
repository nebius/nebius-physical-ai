"""Safety checks for Terraform-managed workbench deploys."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re


class PlanDecision(str, Enum):
    NO_CHANGES = "no_changes"
    FRESH_CREATE = "fresh_create"
    IN_PLACE_UPDATE = "in_place_update"
    REPLACEMENT_REQUIRED = "replacement_required"


@dataclass(frozen=True)
class PlanResourceChange:
    address: str
    action: str
    reason: str = ""


@dataclass(frozen=True)
class PlanAnalysis:
    decision: PlanDecision
    resources: tuple[PlanResourceChange, ...] = ()
    add_count: int = 0
    change_count: int = 0
    destroy_count: int = 0

    @property
    def replacement_resources(self) -> tuple[PlanResourceChange, ...]:
        return self.resources


CRITICAL_RESOURCE_TYPES = {
    "nebius_compute_v1_disk",
    "nebius_compute_v1_instance",
    "nebius_vpc_v1_network",
    "nebius_vpc_v1_security_group",
    "nebius_vpc_v1_subnet",
}
RESTARTABLE_RESOURCE_TYPES = {
    "null_resource",
}

_PLAN_SUMMARY_RE = re.compile(
    r"Plan:\s+(?P<add>\d+)\s+to add,\s+(?P<change>\d+)\s+to change,\s+(?P<destroy>\d+)\s+to destroy"
)
_RESOURCE_HEADER_RE = re.compile(
    r"^\s*#\s+(?P<address>\S+)\s+(?P<action>will be destroyed|must be replaced|will be updated in-place|will be created)\b",
    re.MULTILINE,
)


def _resource_type(address: str) -> str:
    parts = [part for part in address.split(".") if part and part != "module"]
    if len(parts) < 2:
        return ""
    return parts[-2]


def _summary_counts(plan_output: str) -> tuple[int, int, int] | None:
    match = _PLAN_SUMMARY_RE.search(plan_output)
    if not match:
        return None
    return (
        int(match.group("add")),
        int(match.group("change")),
        int(match.group("destroy")),
    )


def _destroying_resources(plan_output: str) -> list[PlanResourceChange]:
    resources: list[PlanResourceChange] = []
    for match in _RESOURCE_HEADER_RE.finditer(plan_output):
        action = match.group("action")
        if action not in {"will be destroyed", "must be replaced"}:
            continue
        address = match.group("address")
        resource_type = _resource_type(address)
        if resource_type in RESTARTABLE_RESOURCE_TYPES:
            continue
        if resource_type in CRITICAL_RESOURCE_TYPES:
            reason = f"{resource_type} is critical infrastructure"
        else:
            reason = "destroy touches an unclassified Terraform resource"
        resources.append(
            PlanResourceChange(address=address, action=action, reason=reason)
        )
    return resources


def analyze_terraform_plan(
    plan_output: str,
    *,
    existing_state: bool = False,
) -> PlanAnalysis:
    """Classify a Terraform plan using resource addresses, not only counts."""
    if "No changes." in plan_output:
        return PlanAnalysis(decision=PlanDecision.NO_CHANGES)

    counts = _summary_counts(plan_output)
    if counts is None:
        return PlanAnalysis(
            decision=PlanDecision.REPLACEMENT_REQUIRED,
            resources=(
                PlanResourceChange(
                    address="<unknown>",
                    action="unknown",
                    reason="Terraform plan summary could not be parsed",
                ),
            ),
        )

    add_count, change_count, destroy_count = counts
    if add_count == 0 and change_count == 0 and destroy_count == 0:
        return PlanAnalysis(decision=PlanDecision.NO_CHANGES)

    if destroy_count <= 0:
        decision = PlanDecision.IN_PLACE_UPDATE if existing_state else PlanDecision.FRESH_CREATE
        return PlanAnalysis(
            decision=decision,
            add_count=add_count,
            change_count=change_count,
            destroy_count=destroy_count,
        )

    resources = _destroying_resources(plan_output)
    if not resources:
        resources = [
            PlanResourceChange(
                address="<unknown>",
                action="destroy",
                reason="Terraform plan includes destroys but no resource headers were parsed",
            )
        ]

    return PlanAnalysis(
        decision=PlanDecision.REPLACEMENT_REQUIRED,
        resources=tuple(resources),
        add_count=add_count,
        change_count=change_count,
        destroy_count=destroy_count,
    )


def format_replacement_required_error(analysis: PlanAnalysis) -> str:
    resources = ", ".join(
        f"{change.address} ({change.reason})" for change in analysis.replacement_resources
    )
    return (
        "Terraform plan would replace or destroy managed infrastructure: "
        f"{resources}. Re-run with --replace --yes only if replacing this alias is intentional."
    )
