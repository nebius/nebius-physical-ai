"""Map planned workflow steps to scheduler task documents (SkyPilot/K8s hints)."""

from __future__ import annotations

from typing import Any, Mapping

from npa.orchestration.npa_workflow.interpreter import PlanStep
from npa.orchestration.npa_workflow.spec import NpaWorkflowSpec


def resources_for_step(spec: NpaWorkflowSpec, step: PlanStep) -> dict[str, Any]:
    profile = step.resources or "default"
    raw = spec.resources.get(profile) or spec.resources.get("default") or {}
    return dict(raw) if isinstance(raw, dict) else {}


def build_scheduler_task(
    spec: NpaWorkflowSpec,
    step: PlanStep,
    *,
    run_id: str,
    image: str = "",
) -> dict[str, Any]:
    """Return a portable task document for one workflow step."""

    resources = resources_for_step(spec, step)
    command = step.argv or (["bash", "-lc", step.shell] if step.shell.strip() else [])
    name = step.state
    if step.iteration is not None:
        name = f"{name}-{step.iteration}"
    return {
        "name": name,
        "run_id": run_id,
        "workflow": spec.name,
        "tool_ref": step.tool_ref,
        "resources": resources,
        "command": command,
        "image": image or str(resources.get("image") or ""),
        "outputs": list(step.outputs),
    }


def build_scheduler_plan(
    spec: NpaWorkflowSpec,
    steps: list[PlanStep],
    *,
    run_id: str,
    image: str = "",
) -> dict[str, Any]:
    return {
        "workflow": spec.name,
        "run_id": run_id,
        "tasks": [
            build_scheduler_task(spec, step, run_id=run_id, image=image) for step in steps
        ],
    }
