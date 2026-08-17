"""Map planned workflow steps to scheduler task documents (SkyPilot/K8s hints)."""

from __future__ import annotations

from typing import Any

from npa.orchestration.npa_workflow.interpreter import PlanStep
from npa.orchestration.npa_workflow.spec import NpaWorkflowSpec


def resources_for_step(spec: NpaWorkflowSpec, step: PlanStep) -> dict[str, Any]:
    # The interpreter resolves config/run/loop tokens while materializing the
    # step.  Prefer that immutable snapshot so the scheduler executes exactly
    # what ``plan-spec`` displayed, including per-state image digests.
    if step.resources_profile:
        return dict(step.resources_profile)
    profile = step.resources or "default"
    raw = spec.resources.get(profile) or spec.resources.get("default") or {}
    return dict(raw) if isinstance(raw, dict) else {}


def num_nodes_for_step(spec: NpaWorkflowSpec, step: PlanStep) -> int:
    """Return the node count a step's resource profile asks for (default 1).

    SkyPilot places ``num_nodes`` at the **task** level, next to ``resources`` — not
    inside it (see ``sky/utils/schemas.py`` and ``npa.burst.core.build_task_spec``). It
    lives on the resource profile in a spec because that is where per-stage shape
    belongs, and the renderer lifts it back out to the task document.

    A ``{{config.*}}`` value is resolved here against the (already ``--var``-merged)
    config, which is how a shipped blueprint's block size can be chosen at submit
    time.
    """

    from npa.orchestration.npa_workflow.spec import profile_num_nodes

    profile = step.resources or "default"
    return profile_num_nodes(
        resources_for_step(spec, step), name=profile, config=spec.config
    )


def build_scheduler_task(
    spec: NpaWorkflowSpec,
    step: PlanStep,
    *,
    run_id: str,
    image: str = "",
) -> dict[str, Any]:
    """Return a portable task document for one workflow step."""

    resources = resources_for_step(spec, step)
    # `bash -c`, not `bash -lc`: a LOGIN shell re-runs the image's profile scripts and
    # can resolve a DIFFERENT python3/PATH than the task environment SkyPilot set up.
    # Two live GPU images broke exactly there (one login python3 had no npa and no
    # numpy, another had no pip at all), so stage commands inherit the task
    # environment instead. `render_task_run_script` sources /etc/profile.d/*.sh first,
    # which is what images actually rely on for activation.
    command = step.argv or (["bash", "-c", step.shell] if step.shell.strip() else [])
    name = step.state
    if step.iteration is not None:
        name = f"{name}-{step.iteration}"
    return {
        "name": name,
        "run_id": run_id,
        "workflow": spec.name,
        "tool_ref": step.tool_ref,
        "resources": resources,
        # Task-level in SkyPilot, so it is task-level in the portable seam too.
        "num_nodes": num_nodes_for_step(spec, step),
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
            build_scheduler_task(spec, step, run_id=run_id, image=image)
            for step in steps
        ],
    }
