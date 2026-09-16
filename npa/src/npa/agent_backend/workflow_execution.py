"""Durable workflow-execution helpers shipped with the Agent backend.

The browser-facing Agent backend binds the state, confirmation, and subprocess
callbacks from its local runtime. Keeping the NPA workflow rendering and submit
logic here keeps that generated backend reviewable without changing its
authority: NPA remains the durable execution runtime.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any


def workflow_requires_staged_source(
    yaml_path: Path, *, project: str, run_id: str, assume_decision: str = ""
) -> bool:
    """Return whether the submitted workflow needs the staged NPA source tree.

    Args:
        yaml_path: Validated temporary workflow specification.
        project: NPA project alias selected by the confirmed action.
        run_id: Durable workflow execution identifier.
        assume_decision: Optional workflow decision supplied at planning time.

    Returns:
        Whether submit rendering requires an uploaded NPA source tree.
    """

    try:
        from npa.cli.workbench.workflow import (
            _plan_requires_npa_source,
            _resolve_submit_registry,
        )
        from npa.orchestration.npa_workflow.skypilot_render import (
            SkypilotRenderOptions,
        )

        return bool(
            _plan_requires_npa_source(
                yaml_path,
                run_id=run_id,
                assume_decision=assume_decision,
                options=SkypilotRenderOptions(
                    registry=_resolve_submit_registry("", project),
                    materialize_registry_secrets=False,
                ),
            )
        )
    except Exception:
        # The confirmation path already validates the spec. The authoritative
        # submit command still validates unusual user-authored configurations.
        return False


def workflow_secret_envs(
    yaml_path: Path, *, run_id: str, assume_decision: str = ""
) -> tuple[str, ...]:
    """Resolve catalog-declared runtime secret environment names for a plan.

    Args:
        yaml_path: Validated temporary workflow specification.
        run_id: Durable workflow execution identifier.
        assume_decision: Optional workflow decision supplied at planning time.

    Returns:
        De-duplicated secret environment variable names required by the plan.
    """

    from npa.orchestration.npa_workflow.interpreter import build_plan
    from npa.orchestration.npa_workflow.skypilot_render import secret_env_hints_for_plan
    from npa.orchestration.npa_workflow.spec import load_spec

    spec = load_spec(yaml_path)
    plan = build_plan(spec, run_id=run_id, assume_decision=assume_decision)
    return tuple(
        dict.fromkeys(
            str(name).strip()
            for name in secret_env_hints_for_plan(plan.steps)
            if str(name).strip()
        )
    )


def execute_workflow_yaml(
    yaml_text: str,
    *,
    run_id: str,
    project: str,
    kubernetes_context: str,
    assume_decision: str = "",
    progress: Callable[[str], None] | None = None,
    write_temp_yaml: Callable[[str], Path],
    run_npa_json: Callable[..., dict[str, Any]],
    requires_staged_source: Callable[..., bool] = workflow_requires_staged_source,
    secret_envs: Callable[..., tuple[str, ...]] = workflow_secret_envs,
    bind_shared_controller: bool = True,
) -> dict[str, Any]:
    """Submit a confirmed workflow through NPA's durable SkyPilot runtime.

    Args:
        yaml_text: Validated workflow YAML exactly matching the confirmation.
        run_id: Durable workflow execution identifier.
        project: Confirmed NPA project alias.
        kubernetes_context: Exact Kubernetes context selected for execution.
        assume_decision: Optional workflow decision supplied at planning time.
        progress: Optional browser-safe progress callback.
        write_temp_yaml: Runtime callback that writes a private temporary spec.
        run_npa_json: Runtime callback that invokes NPA with agent credentials.
        requires_staged_source: Plan-aware callback for source staging.
        secret_envs: Plan-aware callback for required secret environment names.
        bind_shared_controller: Whether this execution uses shared controller
            ownership. Isolated SkyPilot state derives a controller identity and
            must not bind a separate shared owner.

    Returns:
        The durable NPA workflow result.
    """

    yaml_path = write_temp_yaml(yaml_text)
    try:
        if progress:
            progress("resolving runtime credentials")
        resolved_secret_envs = secret_envs(
            yaml_path,
            run_id=run_id,
            assume_decision=assume_decision,
        )
        if requires_staged_source(
            yaml_path,
            project=project,
            run_id=run_id,
            assume_decision=assume_decision,
        ):
            if progress:
                progress("staging workflow source")
            run_npa_json(
                [
                    "workbench",
                    "workflow",
                    "stage-src",
                    "--project",
                    project,
                    "--run-id",
                    f"{run_id}-source",
                ],
                timeout_s=900,
                expect_json=False,
            )
        if progress:
            progress("preparing workflow controller")
        run_npa_json(
            ["skypilot", "bootstrap", "--save"], timeout_s=600, expect_json=False
        )
        if bind_shared_controller:
            if progress:
                progress("binding shared workflow controller")
            run_npa_json(
                [
                    "skypilot",
                    "bind-controller",
                    "--project",
                    project,
                    "--context",
                    kubernetes_context,
                    "--json",
                ],
                timeout_s=300,
                expect_json=False,
            )
        elif progress:
            progress("using isolated workflow controller")
        args = [
            "workbench",
            "workflow",
            "submit",
            str(yaml_path),
            "--project",
            project,
            "--run-id",
            run_id,
        ]
        for secret_env in resolved_secret_envs:
            args.extend(["--secret-env", secret_env])
        args.extend(
            [
                "--infra",
                f"k8s/{kubernetes_context}",
                "--runtime",
                "--output-format",
                "json",
            ]
        )
        if assume_decision:
            args.extend(["--assume-decision", assume_decision])
        if progress:
            progress("submitting durable workflow")
        return run_npa_json(args, timeout_s=None)
    finally:
        try:
            yaml_path.unlink(missing_ok=True)
        except OSError:
            pass


__all__ = [
    "execute_workflow_yaml",
    "workflow_requires_staged_source",
    "workflow_secret_envs",
]
