"""Expose team enrollment, private service operation, and authenticated run commands."""

from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml

from npa.lifecycle_intent import OperationIntent, intent_boundary, json_stdout_contract
from npa.workbench.team.client import TeamClient, load_bearer_token
from npa.workbench.team.errors import TeamError
from npa.workbench.team.models import SubmitRequest, load_config
from .team_accounts import app as accounts_app

app = typer.Typer(
    help="Optional team access and authenticated workflow execution.",
    no_args_is_help=True,
)

app.add_typer(accounts_app, name="account")


@app.command("setup")
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
def serve_cmd(
    config: Path = typer.Option(..., "--config"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8443, "--port"),
):
    """Run one private team supervisor behind the administrator's HTTPS ingress.

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
    endpoint: str = typer.Option(..., "--endpoint", envvar="NPA_TEAM_ENDPOINT"),
    token_env: str = typer.Option("NPA_TEAM_TOKEN", "--token-env"),
    token_file: Path | None = typer.Option(None, "--token-file"),
    output_format: str = typer.Option("json", "--output-format"),
):
    """Show verified identity and workspace access without a personal cloud account.

    Args:
        endpoint, token_env: HTTPS gateway and bearer-token environment variable.
        token_file: Optional private mode-0600 file that overrides token_env.
        output_format: Required JSON response format.
    Returns:
        None; emits JSON to standard output.
    Raises:
        TeamError: Authentication, authorization, or transport fails.
    """
    _json_only(output_format)
    client = _team_client(endpoint, token_env, token_file)
    try:
        typer.echo(json.dumps(client.whoami()))
    finally:
        client.close()


@app.command("submit")
@json_stdout_contract
def submit_cmd(
    spec: Path = typer.Option(..., "--spec"),
    workspace: str = typer.Option(..., "--workspace"),
    cluster: str = typer.Option(..., "--cluster"),
    idempotency_key: str = typer.Option(..., "--idempotency-key"),
    endpoint: str = typer.Option(..., "--endpoint", envvar="NPA_TEAM_ENDPOINT"),
    token_env: str = typer.Option("NPA_TEAM_TOKEN", "--token-env"),
    token_file: Path | None = typer.Option(None, "--token-file"),
    output_format: str = typer.Option("json", "--output-format"),
):
    """Submit an NPA workflow through the authenticated team execution boundary.

    Args:
        spec: Canonical workflow YAML.
        workspace, cluster: Requested enrolled placement.
        idempotency_key: Caller-retained retry identity.
        endpoint, token_env: HTTPS gateway and bearer-token environment variable.
        token_file: Optional private mode-0600 file that overrides token_env.
        output_format: Required JSON response format.
    Returns:
        None; emits JSON to standard output.
    Raises:
        TeamError, OSError: Validation, authorization, operation, or local I/O fails.
    """
    _json_only(output_format)
    request = SubmitRequest(
        workspace=workspace,
        cluster=cluster,
        idempotency_key=idempotency_key,
        workflow=yaml.safe_load(spec.read_text()),
    )
    client = _team_client(endpoint, token_env, token_file)
    try:
        typer.echo(json.dumps(client.submit(request)))
    finally:
        client.close()


@app.command("run")
@json_stdout_contract
def run_cmd(
    run_id: str = typer.Argument(...),
    action: str = typer.Option(
        "status", "--action", help="status, cancel, resume, logs, or artifacts"
    ),
    endpoint: str = typer.Option(..., "--endpoint", envvar="NPA_TEAM_ENDPOINT"),
    token_env: str = typer.Option("NPA_TEAM_TOKEN", "--token-env"),
    token_file: Path | None = typer.Option(None, "--token-file"),
    output_format: str = typer.Option("json", "--output-format"),
):
    """Inspect, cancel, or resume an owned run using the authenticated team API.

    Args:
        run_id, action: Owned run and supported operation.
        endpoint, token_env: HTTPS gateway and token environment variable.
        token_file: Optional private mode-0600 file that overrides token_env.
        output_format: Required JSON response format.
    Returns:
        None; emits JSON to standard output.
    Raises:
        TeamError, OSError: Validation, authorization, operation, or local I/O fails.
    """
    _json_only(output_format)
    client = _team_client(endpoint, token_env, token_file)
    try:
        typer.echo(json.dumps(client.run(run_id, action)))
    finally:
        client.close()


@app.command("list")
@json_stdout_contract
def list_cmd(
    workspace: str = typer.Option(..., "--workspace"),
    endpoint: str = typer.Option(..., "--endpoint", envvar="NPA_TEAM_ENDPOINT"),
    token_env: str = typer.Option("NPA_TEAM_TOKEN", "--token-env"),
    token_file: Path | None = typer.Option(None, "--token-file"),
    output_format: str = typer.Option("json", "--output-format"),
):
    """List the authenticated person's runs in one workspace.

    Args:
        workspace: Authorized workspace to list.
        endpoint, token_env: HTTPS gateway and token environment variable.
        token_file: Optional private mode-0600 file that overrides token_env.
        output_format: Required JSON response format.
    Returns:
        None; emits JSON to standard output.
    Raises:
        TeamError, OSError: Validation, authorization, operation, or local I/O fails.
    """
    _json_only(output_format)
    client = _team_client(endpoint, token_env, token_file)
    try:
        typer.echo(json.dumps(client.list(workspace)))
    finally:
        client.close()


def _json_only(value):
    if value != "json":
        raise TeamError("--output-format must be json")


def _team_client(endpoint: str, token_env: str, token_file: Path | None) -> TeamClient:
    """Create a bearer-authenticated team client from one selected secret source.

    Args:
        endpoint: HTTPS team API endpoint.
        token_env: Environment variable used when token_file is absent.
        token_file: Optional mode-0600 file containing the personal bearer token.
    Returns:
        Authenticated client for one command invocation.
    Raises:
        AuthenticationError, TeamError: Secret source or endpoint is invalid.
    """
    return TeamClient(endpoint, load_bearer_token(token_env, token_file))


@app.command("storage-create")
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
