"""npa workbench workflow — orchestrate multi-stage training workflows."""

from __future__ import annotations

import json
import re
import tempfile
from enum import Enum
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(
    name="workflow",
    help="Multi-stage training workflow orchestration.",
    no_args_is_help=True,
)

console = Console(stderr=True)
_PLACEHOLDER_RE = re.compile(r"\$\{([^}]+)\}")


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


class ActionSpace(str, Enum):
    cartesian = "cartesian"
    joint = "joint"


class ControllerBackendOption(str, Enum):
    kubernetes = "kubernetes"
    nebius = "nebius"


def _fail(msg: str, code: int = 1) -> None:
    console.print(f"[red]Error:[/red] {msg}")
    raise typer.Exit(code)


@app.command("submit")
def submit_cmd(
    yaml_path: Path = typer.Argument(help="SkyPilot workflow YAML path."),
    run_id: str = typer.Option(
        "",
        "--run-id",
        help="SkyPilot managed job name. Defaults to the YAML filename stem.",
    ),
    sky_bin: str = typer.Option(
        "",
        "--sky-bin",
        help="SkyPilot executable path. Defaults to NPA_SKYPILOT_BIN or PATH resolution.",
    ),
    isolated_config_dir: Path | None = typer.Option(
        None,
        "--isolated-config-dir",
        help="Directory for isolated SkyPilot state.",
    ),
    config_path: Path | None = typer.Option(
        None,
        "--config-path",
        help="SkyPilot global config path.",
    ),
    controller_backend: ControllerBackendOption = typer.Option(
        ControllerBackendOption.kubernetes,
        "--controller-backend",
        help="Managed-jobs controller backend.",
    ),
    submit_timeout: int = typer.Option(
        1800,
        "--submit-timeout",
        help="Submission timeout in seconds.",
    ),
    var: list[str] = typer.Option(
        [],
        "--var",
        help="Variable substitution as KEY=VALUE.",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text,
        "--output-format",
        help="Output format.",
    ),
) -> None:
    """Submit a SkyPilot workflow YAML through the NPA controller convention."""
    from npa.orchestration.skypilot.workflow import SkyPilotSubmitError, submit_workflow

    if submit_timeout <= 0:
        _fail(f"--submit-timeout must be positive, got {submit_timeout}")

    substitutions = _parse_submit_vars(var)
    submitted_yaml_path = yaml_path
    submitted_yaml_context: tempfile.TemporaryDirectory[str] | None = None
    if substitutions:
        submitted_yaml_context = tempfile.TemporaryDirectory(prefix="npa-workflow-")
        submitted_yaml_path = Path(submitted_yaml_context.name) / yaml_path.name

    try:
        if substitutions:
            substituted = _substitute_workflow_vars(yaml_path, substitutions)
            _warn_unresolved_placeholders(substituted)
            submitted_yaml_path.write_text(substituted, encoding="utf-8")
        else:
            _warn_unresolved_placeholders(yaml_path.read_text(encoding="utf-8"))

        result = submit_workflow(
            submitted_yaml_path,
            run_id or _default_submit_run_id(yaml_path),
            isolated_config_dir=isolated_config_dir,
            config_path=config_path,
            sky_bin=sky_bin or None,
            controller_backend=controller_backend.value,
            timeout=submit_timeout,
        )
    except OSError as exc:
        _fail(f"SkyPilot workflow submission failed: {exc}")
        return
    except SkyPilotSubmitError as exc:
        _fail(str(exc))
        return
    finally:
        if submitted_yaml_context is not None:
            submitted_yaml_context.cleanup()

    if output_format == OutputFormat.json:
        typer.echo(json.dumps(result.__dict__, indent=2, sort_keys=True))
        return

    typer.echo(f"status: {result.status}")
    if result.job_id:
        typer.echo(f"job_id: {result.job_id}")


def _default_submit_run_id(yaml_path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", Path(yaml_path).stem).strip("-")
    return stem or "workflow"


def _parse_submit_vars(var: list[str]) -> dict[str, str]:
    substitutions: dict[str, str] = {}
    for item in var:
        if "=" not in item:
            _fail("Invalid --var format. Use KEY=VALUE.")
        key, value = item.split("=", 1)
        if not key:
            _fail("Invalid --var format. Use KEY=VALUE.")
        substitutions[key] = value
    return substitutions


def _substitute_workflow_vars(yaml_path: Path, substitutions: dict[str, str]) -> str:
    content = yaml_path.read_text(encoding="utf-8")
    for key, value in substitutions.items():
        content = content.replace(f"${{{key}}}", value)
    return content


def _warn_unresolved_placeholders(content: str) -> None:
    unresolved = sorted({f"${{{match}}}" for match in _PLACEHOLDER_RE.findall(content)})
    if unresolved:
        typer.echo(
            f"Warning: unresolved placeholders remain: {', '.join(unresolved)}",
            err=True,
        )


@app.command("run")
def run_cmd(
    workflow: str = typer.Argument(help="Workflow name (e.g., 'distill')."),
    project: str = typer.Option(
        "", "--project", "-p", help="Project alias from ~/.npa/config.yaml."
    ),
    robot: str = typer.Option("franka_panda", "--robot", help="Robot type."),
    task: str = typer.Option("pick_place", "--task", help="Task name."),
    n_envs: int = typer.Option(4096, "--n-envs", help="Parallel environments for simulation."),
    remote: bool = typer.Option(
        False, "--remote/--local",
        help="Execute on remote VMs via SSH (requires --s3-bucket).",
    ),
    s3_bucket: str = typer.Option(
        "", "--s3-bucket", help="S3 bucket URI for artifact storage (required for --remote)."
    ),
    sim_workbench: str = typer.Option(
        "", "--sim-workbench", help="Workbench name for sim VM (Genesis stages)."
    ),
    train_workbench: str = typer.Option(
        "", "--train-workbench", help="Workbench name for training VM (LeRobot stages). Defaults to sim workbench."
    ),
    action_space: ActionSpace = typer.Option(
        ActionSpace.cartesian, "--action-space",
        help="Action space for Genesis env: 'cartesian' (4D: delta xyz + gripper) "
             "or 'joint' (8D: delta joint positions + gripper).",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--output-format", help="Output format."
    ),
) -> None:
    """Run a named workflow end-to-end."""
    if workflow != "distill":
        _fail(f"Unknown workflow: '{workflow}'. Available: distill")

    if n_envs <= 0:
        _fail(f"--n-envs must be positive, got {n_envs}")

    if remote and not s3_bucket:
        _fail("--remote requires --s3-bucket for artifact handoff between VMs.")

    mode = "remote" if remote else "local"
    console.print(f"[bold]Running workflow: {workflow} ({mode})[/bold]")
    console.print(f"  project={project or '(default)'}  robot={robot}  task={task}")
    console.print(f"  n_envs={n_envs}")
    if remote:
        console.print(f"  sim_workbench={sim_workbench or '(default)'}  train_workbench={train_workbench or '(same as sim)'}")

    from npa.workflows.distill import DistillationError, run_distillation

    try:
        result = run_distillation(
            project=project or None,
            robot=robot,
            task=task,
            n_envs=n_envs,
            remote=remote,
            s3_bucket=s3_bucket,
            sim_workbench=sim_workbench,
            train_workbench=train_workbench,
            action_space=action_space.value,
        )
    except DistillationError as exc:
        _fail(str(exc))
        return

    if output_format == OutputFormat.json:
        typer.echo(json.dumps(result, indent=2))
    else:
        console.print("[green]Workflow complete.[/green]")
        console.print(f"  run_id: {result.get('run_id')}")
        for stage, info in result.get("stages", {}).items():
            status = info.get("status", "unknown")
            tag = "[green]OK[/green]" if status == "success" else "[red]FAILED[/red]"
            console.print(f"  {stage}: {tag}")


@app.command("status")
def status_cmd(
    run_id: str = typer.Argument(help="Run ID to check status of."),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--output-format", help="Output format."
    ),
) -> None:
    """Check the status of a workflow run."""
    from npa.workflows.distill import DistillationError, get_run_status

    try:
        result = get_run_status(run_id)
    except DistillationError as exc:
        _fail(str(exc))
        return

    if output_format == OutputFormat.json:
        typer.echo(json.dumps(result, indent=2))
    else:
        console.print(f"  run_id: {result.get('run_id')}")
        console.print(f"  status: {result.get('status')}")
        for stage, info in result.get("stages", {}).items():
            console.print(f"  {stage}: {info.get('status', 'unknown')}")


@app.command("logs")
def logs_cmd(
    run_id: str = typer.Argument(help="Run ID."),
    stage: str = typer.Argument(help="Stage name (train_teacher, generate_demos, convert, train_student, eval_student)."),
) -> None:
    """Show logs for a specific stage of a workflow run."""
    from npa.workflows.distill import DistillationError, get_stage_logs

    try:
        logs = get_stage_logs(run_id, stage)
    except DistillationError as exc:
        _fail(str(exc))
        return

    typer.echo(logs)


@app.command("teardown")
def teardown_cmd(
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--output-format", help="Output format.",
    ),
) -> None:
    """Destroy both VMs from a distill workflow run.

    Reads the sim and train VM specs from the distill module, bootstraps
    Nebius credentials, and destroys each VM via Terraform.  Also removes
    the workbench entries from ~/.npa/config.yaml.
    """
    from npa.workflows.distill_two_vm import (
        PROJECT_ALIAS, PROJECT_ID, REGION, SIM_VM, TENANT_ID,
        TRAIN_VM, TwoVMDistillError, _destroy_vm,
    )
    from npa.clients.config import (
        ConfigError, resolve_ssh_config,
        remove_workbench_config,
    )

    # Verify at least one VM is still registered.
    found_any = False
    for spec in (SIM_VM, TRAIN_VM):
        try:
            resolve_ssh_config(project=PROJECT_ALIAS, name=spec.name)
            found_any = True
        except ConfigError:
            pass

    if not found_any:
        _fail(
            f"No distill VMs found in config. Expected "
            f"'{SIM_VM.name}' and/or '{TRAIN_VM.name}' in "
            f"project '{PROJECT_ALIAS}'."
        )
        return

    # Bootstrap Nebius credentials for Terraform.
    console.print("[bold]Tearing down distill infrastructure[/bold]")
    console.print(f"  sim:   {SIM_VM.name} ({SIM_VM.gpu_platform})")
    console.print(f"  train: {TRAIN_VM.name} ({TRAIN_VM.gpu_platform})")

    from npa.clients.nebius import NebiusError, bootstrap_environment

    console.print("  Bootstrapping Nebius credentials...")
    try:
        nebius_creds = bootstrap_environment(
            PROJECT_ID, TENANT_ID, REGION,
            on_status=lambda msg: console.print(f"    {msg}"),
        )
    except NebiusError as exc:
        _fail(f"Nebius bootstrap failed: {exc}")
        return

    failed: list[str] = []
    destroyed: list[str] = []

    for spec in (SIM_VM, TRAIN_VM):
        # Skip VMs that are already gone from config.
        try:
            resolve_ssh_config(project=PROJECT_ALIAS, name=spec.name)
        except ConfigError:
            console.print(f"  {spec.name}: not in config, skipping")
            continue

        console.print(f"  Destroying {spec.name}...")
        try:
            _destroy_vm(spec, nebius_creds)
            remove_workbench_config(PROJECT_ALIAS, spec.name)
            destroyed.append(spec.name)
            console.print(f"    {spec.name}: destroyed")
        except TwoVMDistillError as exc:
            failed.append(spec.name)
            console.print(f"    [red]{spec.name}: destroy failed: {exc}[/red]")

    result = {"destroyed": destroyed, "failed": failed}

    if output_format == OutputFormat.json:
        typer.echo(json.dumps(result, indent=2))
    else:
        if destroyed:
            console.print(f"\n[green]Destroyed: {', '.join(destroyed)}[/green]")
        if failed:
            console.print(f"\n[red]Failed: {', '.join(failed)}[/red]")

    if failed:
        raise typer.Exit(1)


@app.command("distill")
def distill_cmd(
    teardown: bool = typer.Option(
        False, "--teardown/--no-teardown",
        help="Destroy both VMs after the workflow completes (even on failure).",
    ),
    skip_infra: bool = typer.Option(
        False, "--skip-infra/--provision",
        help="Skip provisioning and Nebius bootstrap; resolve VMs and S3 "
             "credentials from ~/.npa/config.yaml.",
    ),
    skip_setup: bool = typer.Option(
        False, "--skip-setup/--setup",
        help="Skip runtime setup (conda env + npa install). Use when VMs "
             "already have the correct environment.",
    ),
    n_envs: int = typer.Option(4096, "--n-envs", help="Parallel environments for simulation."),
    teacher_max_iterations: int = typer.Option(
        500, "--teacher-max-iterations",
        help="PPO training iterations for teacher.",
    ),
    student_policy: str = typer.Option(
        "act", "--student-policy",
        help="Student policy type: act, diffusion, smolvla.",
    ),
    student_epochs: int = typer.Option(
        100, "--student-epochs", help="Training epochs for student.",
    ),
    student_batch_size: int = typer.Option(
        64, "--student-batch-size", help="Batch size for student training.",
    ),
    eval_n_episodes: int = typer.Option(
        1024, "--eval-n-episodes", help="Number of eval episodes for the student.",
    ),
    action_space: ActionSpace = typer.Option(
        ActionSpace.cartesian, "--action-space",
        help="Action space for Genesis env: 'cartesian' (4D: delta xyz + gripper) "
             "or 'joint' (8D: delta joint positions + gripper).",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--output-format", help="Output format.",
    ),
) -> None:
    """Run expert distillation: L40S (Genesis) + H100 (LeRobot).

    Provisions an L40S VM for Genesis simulation (stages 1, 2, 3, 5)
    and an H100 VM for LeRobot training (stage 4), with S3 artifact
    handoff between VMs.
    """
    if n_envs <= 0:
        _fail(f"--n-envs must be positive, got {n_envs}")
    if teacher_max_iterations <= 0:
        _fail(f"--teacher-max-iterations must be positive, got {teacher_max_iterations}")
    if student_epochs <= 0:
        _fail(f"--student-epochs must be positive, got {student_epochs}")
    if student_batch_size <= 0:
        _fail(f"--student-batch-size must be positive, got {student_batch_size}")
    if eval_n_episodes <= 0:
        _fail(f"--eval-n-episodes must be positive, got {eval_n_episodes}")
    if student_policy not in ("act", "diffusion", "smolvla"):
        _fail(f"--student-policy must be act, diffusion, or smolvla, got {student_policy}")

    mode = "skip-infra" if skip_infra else "provision"
    console.print(f"[bold]Expert distillation ({mode})[/bold]")
    console.print(f"  sim:   L40S  ({mode})")
    console.print(f"  train: H100  ({mode})")
    console.print(f"  policy={student_policy}  n_envs={n_envs}  epochs={student_epochs}")

    from npa.workflows.distill_two_vm import TwoVMDistillError, distill

    try:
        result = distill(
            teardown=teardown,
            skip_infra=skip_infra,
            skip_setup=skip_setup,
            n_envs=n_envs,
            teacher_max_iterations=teacher_max_iterations,
            student_policy=student_policy,
            student_epochs=student_epochs,
            student_batch_size=student_batch_size,
            eval_n_episodes=eval_n_episodes,
            action_space=action_space.value,
        )
    except TwoVMDistillError as exc:
        _fail(str(exc))
        return

    if output_format == OutputFormat.json:
        typer.echo(json.dumps(result, indent=2))
    else:
        console.print(f"\n[green]Workflow {result.get('status', 'unknown')}.[/green]")
        console.print(f"  run_id: {result.get('run_id')}")
        console.print(f"  s3:     {result.get('s3_base', '')}")
        for stage, info in result.get("stages", {}).items():
            status = info.get("status", "unknown")
            tag = "[green]OK[/green]" if status == "success" else "[red]FAILED[/red]"
            console.print(f"  {stage}: {tag}")
