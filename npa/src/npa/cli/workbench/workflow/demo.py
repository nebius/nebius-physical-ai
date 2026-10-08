"""Launch public workflow demos and open their measured HTML reports."""

from enum import Enum
import json
import shlex
import webbrowser

from botocore.exceptions import BotoCoreError, ClientError
import typer

from npa.clients.config import ConfigError
from npa.lifecycle_intent import OperationIntent, intent_boundary, json_stdout_contract
from npa.orchestration.npa_workflow.demos import (
    demo_storage_environment,
    download_demo_report,
    list_demos,
    prepare_demo,
)
from npa.orchestration.npa_workflow.errors import NpaWorkflowError
from npa.orchestration.npa_workflow.interpreter import build_plan
from npa.orchestration.npa_workflow.submit import load_spec_for_submit

app = typer.Typer(
    help="Run complete public sample workflows and view their results.",
    no_args_is_help=True,
)


class OutputFormat(str, Enum):
    """Supported machine and human output formats."""

    text = "text"
    json = "json"

    def __str__(self) -> str:
        return self.value


@app.command("list")
@json_stdout_contract
def list_cmd(
    output_format: OutputFormat = typer.Option(OutputFormat.text, "--output-format"),
) -> None:
    """Show the four public sample demos; no credentials or GPUs are needed.

    Args:
        output_format: Human-readable or JSON output.
    Returns:
        None.
    Raises:
        None.
    """
    demos = list_demos()
    if output_format == OutputFormat.json:
        typer.echo(json.dumps({"demos": demos}))
        return
    for demo in demos:
        typer.echo(
            f"{demo['name']}\n  Sample: {demo['sample']}\n  Result: {demo['result']}"
        )


@app.command("run")
@intent_boundary(OperationIntent.ENSURE_PRESENT)
@json_stdout_contract
def run_cmd(
    name: str = typer.Argument(help="Public demo name; see demo list."),
    project: str = typer.Option("", "--project", "-p"),
    infra: str = typer.Option("", "--infra", help="Optional RTX Kubernetes target."),
    run_id: str = typer.Option("", "--run-id"),
    resume_run: str = typer.Option(
        "", "--resume-run", help="Resume the exact previous run and its durable state."
    ),
    plan_only: bool = typer.Option(
        False,
        "--plan-only",
        help="Show the standard submission plan without launching.",
    ),
    var: list[str] = typer.Option(
        [],
        "--var",
        help="Workflow config override as KEY=VALUE; repeat for each input.",
    ),
    output_format: OutputFormat = typer.Option(OutputFormat.text, "--output-format"),
) -> None:
    """Fetch public samples and run a complete demo through the standard workflow engine.

    Args:
        name: Public reference demo.
        project: Configured project alias.
        infra: Optional exact RTX Kubernetes target.
        run_id: Fresh run identity; otherwise generated.
        resume_run: Previously printed identity, mutually exclusive with run_id.
        plan_only: Render without execution.
        var: Workflow config overrides, including exact operator image inputs.
        output_format: Human-readable or JSON output.
    Returns:
        None.
    Raises:
        typer.Exit: Selection or standard workflow submission failed.
    """
    _run_demo(name, project, infra, run_id, resume_run, plan_only, var, output_format)


def _run_demo(name, project, infra, run_id, resume_run, plan_only, var, output_format):
    if run_id and resume_run:
        _fail("Use either --run-id or --resume-run.", output_format)
    try:
        selection = prepare_demo(name, project=project, run_id=resume_run or run_id)
        _validate_demo_plan(selection, var)
    except (ValueError, ConfigError, NpaWorkflowError) as exc:
        _fail(str(exc), output_format)
    _submit(selection, infra, resume_run, plan_only, output_format)


def _validate_demo_plan(selection, variables):
    from npa.cli.workbench.workflow import _parse_submit_vars

    overrides = _parse_submit_vars(variables, error_type=NpaWorkflowError)
    if {"bucket", "prefix"} & overrides.keys():
        raise NpaWorkflowError(
            "Demo storage is selected by --project. Use workflow submit for "
            "custom --var bucket or prefix values."
        )
    selection["var"] = [*selection.get("var", []), *variables]
    spec = load_spec_for_submit(
        selection["yaml_path"],
        config_overrides=_parse_submit_vars(
            selection["var"], error_type=NpaWorkflowError
        ),
    )
    build_plan(spec, run_id=selection["run_id"])


def _submit(selection, infra, resume_run, plan_only, output_format):
    from npa.cli.workbench.workflow import (
        OutputFormat as WorkflowOutputFormat,
        _temporary_runtime_environment,
        submit_cmd,
    )

    options = {
        key: value
        for key, value in selection.items()
        if key not in {"name", "report_uri"}
    }
    options.update(
        infra=infra,
        resume_run=resume_run,
        stage_src=True,
        runtime=True,
        durable_s3=True,
        max_wait_seconds=0,
        plan_only=plan_only,
        output_format=WorkflowOutputFormat(output_format.value),
    )
    if resume_run:
        options["run_id"] = ""
    try:
        environment = demo_storage_environment(selection)
    except ConfigError as exc:
        _fail(str(exc), output_format)
    with _temporary_runtime_environment(environment):
        if not plan_only:
            _show_run_identity(selection)
        submit_cmd(**options)


def _show_run_identity(selection):
    command = shlex.join(
        [
            "npa",
            "workbench",
            "workflow",
            "demo",
            "view",
            selection["name"],
            selection["run_id"],
            "--project",
            selection["project"],
        ]
    )
    typer.echo(f"Run ID: {selection['run_id']}", err=True)
    typer.echo(f"View the measured results when available: {command}", err=True)


@app.command("view")
@json_stdout_contract
def view_cmd(
    name: str = typer.Argument(help="Demo name used for the run."),
    run_id: str = typer.Argument(help="Exact run ID printed by demo run."),
    project: str = typer.Option("", "--project", "-p"),
    open_browser: bool = typer.Option(True, "--open/--no-open"),
    output_format: OutputFormat = typer.Option(OutputFormat.text, "--output-format"),
) -> None:
    """Open a compact offline HTML report containing the actual run's evidence.

    Args:
        name: Public reference demo.
        run_id: Exact completed workflow identity.
        project: Configured project alias.
        open_browser: Open the downloaded report in the default browser.
        output_format: Human-readable or JSON output.
    Returns:
        None.
    Raises:
        typer.Exit: The report is unavailable or access failed.
    """
    try:
        path = download_demo_report(name, run_id, project=project)
    except (ValueError, ConfigError, OSError, BotoCoreError, ClientError):
        _fail(
            "Report unavailable. Check workflow status and the selected project's storage access.",
            output_format,
        )
    if open_browser:
        webbrowser.open(path.resolve().as_uri())
    typer.echo(
        json.dumps({"report_path": str(path)})
        if output_format == OutputFormat.json
        else str(path)
    )


def _fail(message: str, output_format: OutputFormat) -> None:
    typer.echo(
        json.dumps({"error": message})
        if output_format == OutputFormat.json
        else f"Error: {message}"
    )
    raise typer.Exit(1)
