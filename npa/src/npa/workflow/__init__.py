"""npa.workflow - multi-stage training workflow orchestration."""

from __future__ import annotations

from pathlib import Path

from npa._sdk import call_cli_callback


def submit(
    yaml_path: str | Path,
    *,
    run_id: str = "",
    var: list[str] | None = None,
    secret_env: list[str] | None = None,
    controller_backend: str = "kubernetes",
    sky_bin: str = "",
    submit_timeout: int = 1800,
    require_controller_up: bool = False,
    image: str = "",
    skip_preflight: bool = False,
    resolve_accelerators: bool = True,
    output_format: str = "text",
) -> None:
    """Submit a SkyPilot workflow YAML (e.g. a Token Factory + GPU combo).

    Drives ``npa workbench workflow submit`` from Python. ``var`` entries are
    ``KEY=VALUE`` substitutions and ``secret_env`` are environment-variable names
    (e.g. ``NEBIUS_TOKEN_FACTORY_KEY``, ``AWS_ACCESS_KEY_ID``) forwarded to SkyPilot as
    secrets. Use this to launch the ``tokenfactory-*`` combos.

    ``skip_preflight`` mirrors the CLI's ``--skip-preflight``: it bypasses the
    prerequisite checks (SkyPilot CLI, npa source, placeholder bucket). Without it a
    Python caller had no way to submit a spec the CLI would let through.

    ``resolve_accelerators=False`` is the explicit SDK equivalent of
    ``--no-resolve-accelerators`` for hermetic rendering or an operator-supplied
    accelerator override. Normal submits keep the SkyPilot readiness gate.

    ``image`` mirrors the CLI's ``--image``: an image reference for every stage, or ``"none"``
    to clear the specs' workbench image pins and run on SkyPilot's default image. Without it a
    spec that pins workbench images could not be submitted from Python against anything else.
    """
    from npa.cli.workbench.workflow import (
        ControllerBackendOption,
        OutputFormat,
        submit_cmd,
    )

    return call_cli_callback(
        submit_cmd,
        yaml_path=Path(yaml_path),
        run_id=run_id,
        var=list(var or []),
        secret_env=list(secret_env or []),
        controller_backend=ControllerBackendOption(controller_backend),
        sky_bin=sky_bin,
        submit_timeout=submit_timeout,
        require_controller_up=require_controller_up,
        image=image,
        skip_preflight=skip_preflight,
        resolve_accelerators=resolve_accelerators,
        output_format=OutputFormat(output_format),
    )


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


__all__ = ["submit", "run", "status", "logs", "teardown", "distill"]
