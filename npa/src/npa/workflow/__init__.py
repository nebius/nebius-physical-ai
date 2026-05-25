"""npa.workflow - multi-stage training workflow orchestration."""

from __future__ import annotations

from npa._sdk import call_cli_callback


def run(
    workflow: str = "distill",
    *,
    project: str | None = None,
    robot: str = "franka_panda",
    task: str = "pick_place",
    n_envs: int = 4096,
    remote: bool = False,
    s3_bucket: str = "",
    sim_workbench: str = "",
    train_workbench: str = "",
    action_space: str = "cartesian",
) -> dict:
    """Run a named workflow end-to-end."""
    if workflow != "distill":
        raise ValueError(f"Unknown workflow: {workflow!r}. Available: 'distill'")
    from npa.workflows.distill import run_distillation

    return run_distillation(
        project=project,
        robot=robot,
        task=task,
        n_envs=n_envs,
        remote=remote,
        s3_bucket=s3_bucket,
        sim_workbench=sim_workbench,
        train_workbench=train_workbench,
        action_space=action_space,
    )


def status(run_id: str, *, output_format: str = "text") -> None:
    """Check the status of a workflow run."""
    from npa.cli.workbench.workflow import status_cmd

    return call_cli_callback(status_cmd, run_id=run_id, output_format=output_format)


def logs(run_id: str, stage: str) -> None:
    """Show logs for a specific stage of a workflow run."""
    from npa.cli.workbench.workflow import logs_cmd

    return call_cli_callback(logs_cmd, run_id=run_id, stage=stage)


def teardown(*, output_format: str = "text") -> None:
    """Destroy both VMs from a distill workflow run."""
    from npa.cli.workbench.workflow import teardown_cmd

    return call_cli_callback(teardown_cmd, output_format=output_format)


def distill(**kwargs):
    """Run expert distillation with the existing two-VM workflow."""
    from npa.cli.workbench.workflow import distill_cmd

    return call_cli_callback(distill_cmd, **kwargs)


__all__ = ["run", "status", "logs", "teardown", "distill"]
