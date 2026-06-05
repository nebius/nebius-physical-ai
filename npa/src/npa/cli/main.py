"""npa CLI entry point."""

from __future__ import annotations

import os
import sys
import traceback
from importlib.metadata import version as package_version

import typer

from npa.cli._error_formatting import format_error_for_user
from npa.cli.workbench import app as workbench_app
from npa.cli.adapter import app as adapter_app
from npa.cli.cluster import app as cluster_app
from npa.cli.convert import app as convert_app
from npa.cli.demo import app as demo_app
from npa.cli.network import app as network_app
from npa.cli.provision import app as provision_app
from npa.cli.rerun import app as rerun_app
from npa.cli.skypilot import app as skypilot_app
from npa.cli.viz import app as viz_app
from npa.cli.workflow_shim import workflow_shim_app
from npa.clients.config import (
    RegistryConfig,
    RuntimeConfig,
    StorageConfig,
    resolve_runtime_config,
    write_runtime_config,
)
from npa.clients.serverless import ServerlessClientError

app = typer.Typer(
    name="npa",
    help=(
        "Nebius Physical AI workbench CLI. "
        "Start with `npa workbench --help` for Workbench tools and workflows."
    ),
    no_args_is_help=True,
)
app.add_typer(
    workbench_app,
    name="workbench",
    short_help="Primary Workbench solution: tools and workflows.",
    rich_help_panel="Primary solution",
)

# FIXME(solutions): These platform-level command groups predate the solution
# namespace model. They remain top-level for compatibility in this PR and should
# migrate to appropriate namespaces in a future change. New commands should be
# registered under a solution namespace, such as `npa workbench ...`, instead of
# adding more top-level registrations here.
app.add_typer(adapter_app, name="adapter", rich_help_panel="Platform utilities")
app.add_typer(cluster_app, name="cluster", rich_help_panel="Platform utilities")
app.add_typer(convert_app, name="convert", rich_help_panel="Platform utilities")
app.add_typer(demo_app, name="demo", rich_help_panel="Platform utilities")
app.add_typer(network_app, name="network", rich_help_panel="Platform utilities")
app.add_typer(provision_app, name="provision-if-absent", rich_help_panel="Setup")
app.add_typer(rerun_app, name="rerun", rich_help_panel="Platform utilities")
app.add_typer(skypilot_app, name="skypilot", rich_help_panel="Platform utilities")
app.add_typer(viz_app, name="viz", rich_help_panel="Platform utilities")
app.add_typer(workflow_shim_app, name="workflow", hidden=True)


_SETUP_GUIDANCE = """Credential setup

Create ~/.npa/credentials.yaml for user-level tokens and BYOVM SSH defaults:

tokens:
  HF_TOKEN: hf_REPLACE_ME
ngc:
  api_key: nvapi_REPLACE_ME
  # org: optional-ngc-org
  # team: optional-ngc-team
ssh:
  host: 203.0.113.10
  user: ubuntu
  key_path: ~/.ssh/id_ed25519

Then secure it:

chmod 600 ~/.npa/credentials.yaml

Deploy commands create and update ~/.npa/config.yaml for projects, workbenches,
SSH targets, container registry settings, and Terraform state.
"""


def _version_callback(value: bool) -> None:
    if not value:
        return
    typer.echo(f"npa {package_version('npa')}")
    raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show the installed npa version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Nebius Physical AI workbench CLI."""


@app.command(
    "configure",
    help="Interactive credential and config setup guidance.",
    rich_help_panel="Setup",
)
def configure(
    project: str = typer.Option("", "--project", help="Project alias in ~/.npa/config.yaml."),
    project_id: str = typer.Option("", "--project-id", help="Nebius project ID."),
    tenant_id: str = typer.Option("", "--tenant-id", help="Nebius tenant ID."),
    region: str = typer.Option("", "--region", help="Nebius region, for example eu-north1."),
    registry: str = typer.Option("", "--registry", help="Full container registry prefix."),
    registry_id: str = typer.Option("", "--registry-id", help="Nebius container registry ID."),
    s3_endpoint: str = typer.Option("", "--s3-endpoint", help="S3-compatible endpoint URL."),
    s3_bucket: str = typer.Option("", "--s3-bucket", help="S3 bucket URI or bucket name."),
    aws_access_key_id: str = typer.Option("", "--aws-access-key-id", help="S3 access key ID."),
    aws_secret_access_key: str = typer.Option("", "--aws-secret-access-key", help="S3 secret access key."),
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        help="Write a populated template from flags and placeholders without prompting.",
    ),
) -> None:
    """Capture project, registry, and storage runtime settings."""
    has_values = any(
        (
            project,
            project_id,
            tenant_id,
            region,
            registry,
            registry_id,
            s3_endpoint,
            s3_bucket,
            aws_access_key_id,
            aws_secret_access_key,
        )
    )
    if not non_interactive and not has_values and not sys.stdin.isatty():
        typer.echo(_SETUP_GUIDANCE)
        return

    resolved = resolve_runtime_config(
        project or None,
        project_id=project_id or None,
        tenant_id=tenant_id or None,
        region=region or None,
        registry=registry or None,
        registry_id=registry_id or None,
        s3_endpoint=s3_endpoint or None,
        s3_bucket=s3_bucket or None,
        aws_access_key_id=aws_access_key_id or None,
        aws_secret_access_key=aws_secret_access_key or None,
    )

    if non_interactive:
        runtime_config = _non_interactive_runtime_config(resolved)
    else:
        runtime_config = _interactive_runtime_config(resolved)

    config_path, credentials_path = write_runtime_config(runtime_config)
    typer.echo(f"Wrote config: {config_path}")
    if credentials_path is not None:
        typer.echo(f"Wrote credentials: {credentials_path}")
    typer.echo(f"Project: {runtime_config.project}")
    typer.echo(f"Registry: {runtime_config.registry.registry}")
    typer.echo(f"Storage: {runtime_config.storage.checkpoint_bucket}")


@app.command(
    "init",
    help="Show credential and config setup guidance.",
    rich_help_panel="Setup",
)
def init() -> None:
    """Show credential and config setup guidance."""
    typer.echo(_SETUP_GUIDANCE)


def _non_interactive_runtime_config(resolved: RuntimeConfig) -> RuntimeConfig:
    region = resolved.region or "eu-north1"
    registry_id = resolved.registry.registry_id or "<registry-id>"
    registry = (
        resolved.registry.registry
        if resolved.registry.registry and "<" not in resolved.registry.registry
        else f"cr.{region}.nebius.cloud/{registry_id}"
    )
    return RuntimeConfig(
        project=resolved.project or "default",
        project_id=resolved.project_id or "<project-id>",
        tenant_id=resolved.tenant_id or "<tenant-id>",
        region=region,
        registry=RegistryConfig(
            registry=registry,
            registry_id=registry_id,
        ),
        storage=StorageConfig(
            checkpoint_bucket=resolved.storage.checkpoint_bucket or "s3://<bucket>/checkpoints/",
            endpoint_url=resolved.storage.endpoint_url or f"https://storage.{region}.nebius.cloud",
            aws_access_key_id=resolved.storage.aws_access_key_id or "<aws-access-key-id>",
            aws_secret_access_key=resolved.storage.aws_secret_access_key or "<aws-secret-access-key>",
        ),
    )


def _interactive_runtime_config(resolved: RuntimeConfig) -> RuntimeConfig:
    project = _prompt("Project alias", resolved.project or "default")
    region = _prompt("Region", resolved.region or "eu-north1")
    project_id = _prompt("Project ID", resolved.project_id)
    tenant_id = _prompt("Tenant ID", resolved.tenant_id)
    registry_id_default = resolved.registry.registry_id or ""
    registry_id = _prompt("Registry ID", registry_id_default)
    registry_default = resolved.registry.registry or (
        f"cr.{region}.nebius.cloud/{registry_id}" if registry_id else ""
    )
    registry = _prompt("Registry prefix", registry_default)
    s3_endpoint = _prompt(
        "S3 endpoint",
        resolved.storage.endpoint_url or f"https://storage.{region}.nebius.cloud",
    )
    s3_bucket = _prompt("S3 bucket URI", resolved.storage.checkpoint_bucket)
    aws_access_key_id = _prompt("S3 access key ID", resolved.storage.aws_access_key_id)
    aws_secret_access_key = typer.prompt(
        "S3 secret access key",
        default=resolved.storage.aws_secret_access_key,
        hide_input=True,
        show_default=bool(resolved.storage.aws_secret_access_key),
    )
    return RuntimeConfig(
        project=project,
        project_id=project_id,
        tenant_id=tenant_id,
        region=region,
        registry=RegistryConfig(registry=registry, registry_id=registry_id),
        storage=StorageConfig(
            checkpoint_bucket=s3_bucket,
            endpoint_url=s3_endpoint,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        ),
    )


def _prompt(label: str, default: str) -> str:
    return typer.prompt(label, default=default, show_default=bool(default))


def app_entry() -> None:
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)
    except ServerlessClientError as exc:
        print(format_error_for_user(exc, output_format=_detect_error_format()), file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(format_error_for_user(exc, output_format=_detect_error_format()), file=sys.stderr)
        if os.environ.get("NPA_DEBUG"):
            traceback.print_exc()
        else:
            print("  Run with NPA_DEBUG=1 for full traceback.", file=sys.stderr)
        sys.exit(2)


def _detect_error_format() -> str:
    env_format = os.environ.get("NPA_ERROR_FORMAT", "").lower()
    if env_format in {"json", "text"}:
        return env_format
    args = sys.argv[1:]
    for index, value in enumerate(args):
        if value in {"--output", "--output-format", "--format"} and index + 1 < len(args):
            if args[index + 1].lower() == "json":
                return "json"
        if value in {"--output=json", "--output-format=json", "--format=json"}:
            return "json"
    return "text"
