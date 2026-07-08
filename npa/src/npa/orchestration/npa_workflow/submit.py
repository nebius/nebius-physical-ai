"""Prepare an ``npa.workflow`` spec for SkyPilot submission."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from npa.orchestration.npa_workflow.errors import NpaWorkflowError
from npa.orchestration.npa_workflow.interpreter import ExecutionPlan, build_plan
from npa.orchestration.npa_workflow.skypilot_render import (
    SkypilotRenderOptions,
    assert_no_unresolved_placeholders,
    render_skypilot_yaml,
    secret_env_hints_for_plan,
)
from npa.orchestration.npa_workflow.spec import NpaWorkflowSpec, load_spec


@dataclass(frozen=True)
class PreparedNpaWorkflowSubmit:
    """Result of rendering an npa.workflow spec into a SkyPilot YAML on disk."""

    skypilot_yaml_path: Path
    spec: NpaWorkflowSpec
    plan: ExecutionPlan
    secret_env_hints: tuple[str, ...]
    temp_dir: tempfile.TemporaryDirectory[str]


def _merge_config_overrides(
    spec: NpaWorkflowSpec,
    overrides: Mapping[str, str] | None,
) -> NpaWorkflowSpec:
    if not overrides:
        return spec
    merged = dict(spec.config)
    for key, value in overrides.items():
        merged[key] = value
    return NpaWorkflowSpec(
        api_version=spec.api_version,
        kind=spec.kind,
        metadata=dict(spec.metadata),
        config=merged,
        run_defaults=dict(spec.run_defaults),
        resources=dict(spec.resources),
        initial=spec.initial,
        states=dict(spec.states),
    )


def _resolve_assume_decision(spec: NpaWorkflowSpec, assume_decision: str) -> str:
    if assume_decision.strip():
        return assume_decision.strip()
    return str(spec.config.get("plan_assume_decision") or "").strip()


def _spec_needs_assume_decision(spec: NpaWorkflowSpec) -> bool:
    return any(state.transitions for state in spec.states.values())


def prepare_npa_workflow_for_submit(
    yaml_path: Path,
    *,
    run_id: str,
    assume_decision: str = "",
    config_overrides: Mapping[str, str] | None = None,
    render_options: SkypilotRenderOptions | None = None,
) -> PreparedNpaWorkflowSubmit:
    """Load, plan, and render an npa.workflow spec into a temporary SkyPilot YAML.

    The returned ``temp_dir`` must be kept alive until ``submit_workflow`` returns;
    callers own cleanup via ``temp_dir.cleanup()``.
    """

    if not run_id.strip():
        raise NpaWorkflowError("run_id is required when preparing an npa.workflow for submit")

    spec = load_spec(yaml_path)
    spec = _merge_config_overrides(spec, config_overrides)
    resolved_assume = _resolve_assume_decision(spec, assume_decision)
    if _spec_needs_assume_decision(spec) and not resolved_assume:
        raise NpaWorkflowError(
            f"workflow {spec.name!r} has dynamic transitions; pass --assume-decision "
            "promote_checkpoint|loop_back (or set config.plan_assume_decision)"
        )

    plan = build_plan(spec, run_id=run_id, assume_decision=resolved_assume or None)
    opts = render_options or SkypilotRenderOptions()
    yaml_text = render_skypilot_yaml(spec, plan, run_id=run_id, options=opts)
    assert_no_unresolved_placeholders(yaml_text)

    temp_dir = tempfile.TemporaryDirectory(prefix="npa-workflow-npa-")
    out_path = Path(temp_dir.name) / f"{spec.name}.skypilot.yaml"
    out_path.write_text(yaml_text, encoding="utf-8")
    return PreparedNpaWorkflowSubmit(
        skypilot_yaml_path=out_path,
        spec=spec,
        plan=plan,
        secret_env_hints=secret_env_hints_for_plan(plan.steps),
        temp_dir=temp_dir,
    )
