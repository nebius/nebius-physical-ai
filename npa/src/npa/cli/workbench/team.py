"""Expose Workbench access administration and authenticated run commands."""

from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml

from npa.lifecycle_intent import OperationIntent, intent_boundary, json_stdout_contract
from npa.workbench.team.client import TeamClient
from npa.workbench.team.errors import TeamError
from npa.workbench.team.models import load_config
from .team_accounts import app as accounts_app
from .team_login import login_cmd, logout_cmd

app = typer.Typer(
    help="Compatibility commands for Workbench access and workflow execution.",
    no_args_is_help=True,
)
access_app = typer.Typer(
    help="Configure Workbench accounts, allocations, connections, and service access.",
    no_args_is_help=True,
)

app.add_typer(accounts_app, name="account")
access_app.add_typer(accounts_app, name="account")
app.command("login")(login_cmd)
app.command("logout")(logout_cmd)


@app.command("export-connections")
@access_app.command("export-connections")
@json_stdout_contract
def export_connections_cmd(
    config: Path = typer.Option(..., "--config"),
    output_dir: Path = typer.Option(..., "--output-dir"),
    server_directory: Path = typer.Option(Path("/etc/npa-team"), "--server-directory"),
    output_format: str = typer.Option("json", "--output-format"),
):
    """Prepare private connections to existing Kubernetes clusters on any cloud.

    Args:
        config: Administrator policy and locally usable Kubernetes connections.
        output_dir: New private directory for exported server configuration files.
        server_directory: Configuration mount in both server containers.
        output_format: Required JSON response format.
    Returns:
        None; emits file names and cluster aliases without credentials.
    Raises:
        TeamError, OSError: Selected connections or private export fail.
    """
    from npa.workbench.team.cluster_connections import export_connections

    _json_only(output_format)
    typer.echo(
        json.dumps(
            export_connections(
                load_config(config), output_dir, server_directory=server_directory
            )
        )
    )


@app.command("check-clusters")
@access_app.command("check-clusters")
@json_stdout_contract
def check_clusters_cmd(
    config: Path = typer.Option(..., "--config"),
    output_format: str = typer.Option("json", "--output-format"),
):
    """Verify existing cluster access and boundaries from the server environment.

    Args:
        config: Private server configuration with mounted cluster credentials.
        output_format: Required JSON response format.
    Returns:
        None; emits verified allocation counts without credentials.
    Raises:
        TeamError, OSError: Connectivity, authentication, enrollment, or file reads fail.
    """
    from npa.workbench.team.cluster_connections import check_clusters

    _json_only(output_format)
    typer.echo(json.dumps(check_clusters(load_config(config))))


@app.command("setup")
@access_app.command("setup")
@intent_boundary(OperationIntent.ENSURE_PRESENT)
@json_stdout_contract
def setup_cmd(
    input_path: Path = typer.Option(..., "--input-path"),
    output_path: Path = typer.Option(..., "--output-path"),
    output_format: str = typer.Option("json", "--output-format"),
):
    """Deploy the shared HTTPS control plane and retain its Nebius LB address.

    Args:
        input_path: Private operator YAML selecting one server installation.
        output_path: Private persistent receipt reused when retrying setup.
        output_format: Required JSON response format.
    Returns:
        None; prints status without personal credentials or infrastructure identities.
    Raises:
        TeamError, OSError: Setup selection, ownership, transport or I/O fails.
    """
    from npa.workbench.team.setup import setup_control_plane
    from npa.workbench.team.setup_models import read_setup_request

    _json_only(output_format)
    result = setup_control_plane(read_setup_request(input_path), output_path)
    typer.echo(json.dumps(result))
    if result["status"] != "ready":
        raise typer.Exit(2)


@app.command("stop-run")
@access_app.command("stop-run")
@intent_boundary(OperationIntent.DESTROY)
@json_stdout_contract
def stop_cmd(
    run_id: str = typer.Argument(...),
    config: Path = typer.Option(..., "--config"),
    output_format: str = typer.Option("json", "--output-format"),
):
    """Cancel an exact run as the local server operator, including after offboarding.

    Args:
        run_id: Exact server-issued run identity.
        config: Private administrator configuration.
        output_format: Required JSON response format.
    Returns:
        None; emits JSON to standard output.
    Raises:
        TeamError, OSError: Validation, authorization, operation, or local I/O fails.
    """
    from npa.workbench.team.administration import stop_run

    _json_only(output_format)
    typer.echo(json.dumps(stop_run(load_config(config), run_id)))


@app.command("render")
@access_app.command("render")
@json_stdout_contract
def render_cmd(
    config: Path = typer.Option(..., "--config"),
    output_dir: Path = typer.Option(..., "--output-dir"),
    output_format: str = typer.Option("json", "--output-format"),
):
    """Render cluster boundaries and private SkyPilot configuration for review.

    Args:
        config, output_dir: Administrator policy and new private destination.
        output_format: Required JSON response format.
    Returns:
        None; emits JSON to standard output.
    Raises:
        TeamError, OSError: Validation, authorization, operation, or local I/O fails.
    """
    from npa.workbench.team.deployment import render_installation

    _json_only(output_format)
    typer.echo(json.dumps(render_installation(load_config(config), output_dir)))


@app.command("enroll")
@access_app.command("enroll")
@intent_boundary(OperationIntent.ENSURE_PRESENT)
@json_stdout_contract
def enroll_cmd(
    config: Path = typer.Option(..., "--config"),
    workspace: str = typer.Option(..., "--workspace"),
    cluster: str = typer.Option(..., "--cluster"),
    subject: str = typer.Option(..., "--subject"),
    output_format: str = typer.Option("json", "--output-format"),
):
    """Apply and verify one administrator-selected personal cluster allocation.

    Args:
        config: Private administrator policy.
        workspace, cluster, subject: Exact allocation to enroll.
        output_format: Required JSON response format.
    Returns:
        None; emits JSON to standard output.
    Raises:
        TeamError, OSError: Validation, authorization, operation, or local I/O fails.
    """
    from npa.workbench.team.deployment import allocations
    from npa.workbench.team.enrollment import apply_enrollment

    _json_only(output_format)
    matches = [
        binding
        for binding in allocations(load_config(config))
        if (binding.workspace, binding.cluster, binding.allocation.subject)
        == (workspace, cluster, subject)
    ]
    if len(matches) != 1:
        raise TeamError("allocation does not exist in the administrator configuration")
    typer.echo(json.dumps(apply_enrollment(matches[0])))


@app.command("export-kubeconfig")
@access_app.command("export-kubeconfig")
@json_stdout_contract
def export_cmd(
    config: Path = typer.Option(..., "--config"),
    output_path: Path = typer.Option(..., "--output-path"),
    output_format: str = typer.Option("json", "--output-format"),
):
    """Export selected cluster credentials to a new private server-only file.

    Args:
        config, output_path: Administrator policy and new server credential file.
        output_format: Required JSON response format.
    Returns:
        None; emits JSON to standard output.
    Raises:
        TeamError, OSError: Validation, authorization, operation, or local I/O fails.
    """
    from npa.workbench.team.deployment import export_server_kubeconfig

    _json_only(output_format)
    typer.echo(json.dumps(export_server_kubeconfig(load_config(config), output_path)))


@app.command("serve")
@access_app.command("serve")
def serve_cmd(
    config: Path = typer.Option(..., "--config"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8443, "--port"),
):
    """Run one private Workbench supervisor behind the administrator's HTTPS ingress.

    Args:
        config: Private administrator policy.
        host, port: Gateway listener behind HTTPS ingress.
    Returns:
        Does not return after a successful service exec.
    Raises:
        TeamError, OSError: Validation, authorization, operation, or local I/O fails.
    """
    from npa.workbench.team.server import serve

    serve(config.resolve(), host=host, port=port)


@app.command("whoami")
@json_stdout_contract
def whoami_cmd(
    endpoint: str | None = typer.Option(None, "--endpoint", envvar="NPA_TEAM_ENDPOINT"),
    token_env: str = typer.Option("NPA_TEAM_TOKEN", "--token-env"),
    token_file: Path | None = typer.Option(None, "--token-file"),
    profile: str | None = typer.Option(
        None, "--profile", help="Saved Workbench connection."
    ),
    ca_file: Path | None = typer.Option(
        None, "--ca-file", help="Private service CA certificate."
    ),
    output_format: str = typer.Option("json", "--output-format"),
):
    """Show verified identity and workspace access without a personal cloud account.

    Args:
        endpoint, token_env: HTTPS gateway and bearer-token environment variable.
        token_file: Optional private mode-0600 file that overrides token_env.
        profile, ca_file: Saved connection and optional trusted private CA.
        output_format: Required JSON response format.
    Returns:
        None; emits JSON to standard output.
    Raises:
        TeamError: Authentication, authorization, or transport fails.
    """
    _json_only(output_format)
    client = _team_client(endpoint, token_env, token_file, profile, ca_file)
    try:
        typer.echo(json.dumps(client.whoami()))
    finally:
        client.close()


@app.command("submit")
@json_stdout_contract
def submit_cmd(
    spec: Path = typer.Option(..., "--spec"),
    var: list[str] = typer.Option(
        [], "--var", help="Repeatable workflow config override as KEY=VALUE."
    ),
    workspace: str | None = typer.Option(None, "--workspace"),
    cluster: str | None = typer.Option(None, "--cluster"),
    idempotency_key: str | None = typer.Option(None, "--idempotency-key"),
    new_run: bool = typer.Option(
        False,
        "--new-run",
        help="Start another run after a prior acknowledged submission.",
    ),
    endpoint: str | None = typer.Option(None, "--endpoint", envvar="NPA_TEAM_ENDPOINT"),
    token_env: str = typer.Option("NPA_TEAM_TOKEN", "--token-env"),
    token_file: Path | None = typer.Option(None, "--token-file"),
    profile: str | None = typer.Option(
        None, "--profile", help="Saved Workbench connection."
    ),
    ca_file: Path | None = typer.Option(
        None, "--ca-file", help="Private service CA certificate."
    ),
    output_format: str = typer.Option("json", "--output-format"),
):
    """Start a workflow and return its run ID; unchanged retries reuse that run.

    Args:
        spec: Canonical workflow YAML.
        var: Config values, such as an explicit LeRobot episode and camera.
        workspace, cluster: Requested enrolled placement.
        idempotency_key: Optional explicit retry identity; otherwise retained automatically.
        new_run: Start another run after acknowledging the previous submission.
        endpoint, token_env: HTTPS gateway and bearer-token environment variable.
        token_file: Optional private mode-0600 file that overrides token_env.
        profile, ca_file: Saved connection and optional trusted private CA.
        output_format: Required JSON response format.
    Returns:
        None; emits JSON to standard output.
    Raises:
        TeamError, OSError: Validation, authorization, operation, or local I/O fails.
    """
    _json_only(output_format)
    document = _submission_document(spec, var)
    client, session = _team_connection(
        endpoint, token_env, token_file, profile, ca_file
    )
    try:
        from npa.workbench.team.submission_receipts import submit_workflow

        result = submit_workflow(
            client,
            session,
            document,
            workspace=workspace,
            cluster=cluster,
            idempotency_key=idempotency_key,
            new_run=new_run,
        )
        typer.echo(json.dumps(result))
    finally:
        client.close()


def _submission_document(spec: Path, variables: list[str]) -> dict:
    from .workflow import _parse_submit_vars

    document = yaml.safe_load(spec.read_text())
    if not isinstance(document, dict):
        raise TeamError("workflow must be a mapping")
    overrides = _parse_submit_vars(variables)
    if not overrides:
        return document
    config = document.setdefault("config", {})
    if not isinstance(config, dict):
        raise TeamError("workflow config must be a mapping")
    config.update(overrides)
    return document


@app.command("run")
@json_stdout_contract
def run_cmd(
    run_id: str = typer.Argument(...),
    action: str = typer.Option(
        "status",
        "--action",
        help="Existing run operation: status (default), logs, artifacts, cancel, or resume.",
    ),
    endpoint: str | None = typer.Option(None, "--endpoint", envvar="NPA_TEAM_ENDPOINT"),
    token_env: str = typer.Option("NPA_TEAM_TOKEN", "--token-env"),
    token_file: Path | None = typer.Option(None, "--token-file"),
    profile: str | None = typer.Option(
        None, "--profile", help="Saved Workbench connection."
    ),
    ca_file: Path | None = typer.Option(
        None, "--ca-file", help="Private service CA certificate."
    ),
    output_format: str = typer.Option("json", "--output-format"),
):
    """Show an existing run's status, or manage that run with --action.

    Args:
        run_id, action: Owned run and supported operation.
        endpoint, token_env: HTTPS gateway and token environment variable.
        token_file: Optional private mode-0600 file that overrides token_env.
        profile, ca_file: Saved connection and optional trusted private CA.
        output_format: Required JSON response format.
    Returns:
        None; emits JSON to standard output.
    Raises:
        TeamError, OSError: Validation, authorization, operation, or local I/O fails.
    """
    _json_only(output_format)
    client = _team_client(endpoint, token_env, token_file, profile, ca_file)
    try:
        typer.echo(json.dumps(client.run(run_id, action)))
    finally:
        client.close()


@app.command("list")
@json_stdout_contract
def list_cmd(
    workspace: str | None = typer.Option(None, "--workspace"),
    endpoint: str | None = typer.Option(None, "--endpoint", envvar="NPA_TEAM_ENDPOINT"),
    token_env: str = typer.Option("NPA_TEAM_TOKEN", "--token-env"),
    token_file: Path | None = typer.Option(None, "--token-file"),
    profile: str | None = typer.Option(
        None, "--profile", help="Saved Workbench connection."
    ),
    ca_file: Path | None = typer.Option(
        None, "--ca-file", help="Private service CA certificate."
    ),
    output_format: str = typer.Option("json", "--output-format"),
):
    """List the authenticated person's runs in one workspace.

    Args:
        workspace: Authorized workspace to list.
        endpoint, token_env: HTTPS gateway and token environment variable.
        token_file: Optional private mode-0600 file that overrides token_env.
        profile, ca_file: Saved connection and optional trusted private CA.
        output_format: Required JSON response format.
    Returns:
        None; emits JSON to standard output.
    Raises:
        TeamError, OSError: Validation, authorization, operation, or local I/O fails.
    """
    _json_only(output_format)
    client, session = _team_connection(
        endpoint, token_env, token_file, profile, ca_file
    )
    try:
        typer.echo(
            json.dumps(
                client.list(_placement(workspace, session.workspace, "workspace"))
            )
        )
    finally:
        client.close()


def _json_only(value):
    if value != "json":
        raise TeamError("--output-format must be json")


def _team_client(endpoint, token_env, token_file, profile=None, ca_file=None):
    return _team_connection(endpoint, token_env, token_file, profile, ca_file)[0]


def _team_connection(endpoint, token_env, token_file, profile, ca_file):
    from npa.workbench.team.connection import open_connection

    return open_connection(
        client_factory=TeamClient,
        endpoint=endpoint,
        token_env=token_env,
        token_file=token_file,
        profile=profile,
        ca_file=ca_file,
    )


def _placement(explicit, saved, name):
    selected = explicit or saved
    if not selected:
        raise TeamError(
            f"Choose --{name}; use npa workbench whoami to see permitted choices."
        )
    return selected


@app.command("storage-create")
@access_app.command("storage-create")
@intent_boundary(OperationIntent.ENSURE_PRESENT)
@json_stdout_contract
def storage_cmd(
    project: str = typer.Option(..., "--project", "-p"),
    endpoint: str = typer.Option(..., "--endpoint"),
    output_dir: Path = typer.Option(..., "--output-dir"),
    output_format: str = typer.Option("json", "--output-format"),
):
    """Create a personal Nebius bucket and narrowly scoped storage principal.

    Args:
        project, endpoint: Operator project alias and HTTPS object endpoint.
        output_dir: New private credential and provisioning-receipt directory.
        output_format: Required JSON response format.
    Returns:
        None; emits JSON to standard output.
    Raises:
        TeamError, OSError: Validation, authorization, operation, or local I/O fails.
    """
    from npa.workbench.team.nebius_storage import create_storage

    _json_only(output_format)
    typer.echo(json.dumps(create_storage(project, endpoint, output_dir.resolve())))


@app.command("render-service")
@access_app.command("render-service")
@json_stdout_contract
def service_cmd(
    namespace: str = typer.Option(..., "--namespace"),
    image: str = typer.Option(..., "--image"),
    secret: str = typer.Option(..., "--secret"),
    claim: str = typer.Option(..., "--claim"),
    output_path: Path = typer.Option(..., "--output-path"),
    output_format: str = typer.Option("json", "--output-format"),
):
    """Render a CPU-only gateway and private SkyPilot sidecar into a new YAML file.

    Args:
        namespace, image: Dedicated management namespace and reviewed image.
        secret, claim: Existing configuration Secret and state PVC.
        output_path: New rendered deployment file.
        output_format: Required JSON response format.
    Returns:
        None; emits JSON to standard output.
    Raises:
        TeamError, OSError: Validation, authorization, operation, or local I/O fails.
    """
    from npa.workbench.team.pod_deployment import service_manifests

    _json_only(output_format)
    documents = service_manifests(
        namespace=namespace, image=image, secret=secret, claim=claim
    )
    with output_path.open("x") as stream:
        stream.write(yaml.safe_dump_all(documents, sort_keys=False))
    typer.echo(json.dumps({"manifest": str(output_path)}))
