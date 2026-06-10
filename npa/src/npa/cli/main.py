"""npa CLI entry point."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import traceback
from importlib.metadata import version as package_version
from typing import Callable, Optional

import typer

from npa.cli._error_formatting import format_error_for_user
from npa.cli.workbench import app as workbench_app
from npa.cli.adapter import app as adapter_app
from npa.cli.cluster import app as cluster_app
from npa.cli.convert import app as convert_app
from npa.cli.demo import app as demo_app
from npa.cli.network import app as network_app
from npa.cli.rerun import app as rerun_app
from npa.cli.skypilot import app as skypilot_app
from npa.cli.viz import app as viz_app
from npa.cli.workflow_shim import workflow_shim_app
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
app.add_typer(rerun_app, name="rerun", rich_help_panel="Platform utilities")
app.add_typer(skypilot_app, name="skypilot", rich_help_panel="Platform utilities")
app.add_typer(viz_app, name="viz", rich_help_panel="Platform utilities")
app.add_typer(workflow_shim_app, name="workflow", hidden=True)


DEFAULT_STORAGE_ENDPOINT = "https://storage.eu-north1.nebius.cloud"
DEFAULT_REGION = "eu-north1"

_SETUP_GUIDANCE = """Credential setup

Run `npa configure` in a terminal for an interactive setup, or create
~/.npa/credentials.yaml by hand for user-level tokens, object storage, and
BYOVM SSH defaults:

tokens:
  HF_TOKEN: hf_REPLACE_ME
ngc:
  api_key: nvapi_REPLACE_ME
  # org: optional-ngc-org
  # team: optional-ngc-team
storage:
  aws_access_key_id: <your-s3-access-key-id>
  aws_secret_access_key: <your-s3-secret-access-key>
  endpoint_url: https://storage.eu-north1.nebius.cloud
  bucket: s3://<your-bucket>/
ssh:
  host: <your-byovm-host>
  user: ubuntu
  key_path: ~/.ssh/id_ed25519

Then secure it:

chmod 600 ~/.npa/credentials.yaml

`npa configure` also writes ~/.npa/config.yaml with your Nebius project id,
tenant id, region, and container registry so commands no longer need those
values exported in the shell or read from the Nebius CLI. Deploy commands
extend the same file with workbench endpoints and Terraform state.
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


def _nebius_profile_ready(*, runner: Callable[..., object] = subprocess.run) -> bool:
    """Return True when the local Nebius CLI has a usable, authenticated profile."""

    if not shutil.which("nebius"):
        return False
    try:
        result = runner(
            ["nebius", "iam", "get-access-token"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return getattr(result, "returncode", 1) == 0


def _create_nebius_profile(*, runner: Callable[..., object] = subprocess.run) -> bool:
    """Run the interactive `nebius profile create` flow."""

    try:
        result = runner(["nebius", "profile", "create"], check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return getattr(result, "returncode", 1) == 0


def _ensure_nebius_profile() -> None:
    """Detect or interactively create a local Nebius CLI profile."""

    if _nebius_profile_ready():
        typer.echo("Nebius CLI profile detected (`nebius iam get-access-token` works).")
        return
    if not shutil.which("nebius"):
        typer.echo(
            "Nebius CLI not found. Install it from https://docs.nebius.com/cli/install, "
            "then re-run `npa configure` to create a local profile."
        )
        return
    if not typer.confirm(
        "No authenticated Nebius CLI profile found. Create one now?",
        default=True,
    ):
        typer.echo("Skipped Nebius profile creation. Run `nebius profile create` later.")
        return
    if _create_nebius_profile() and _nebius_profile_ready():
        typer.echo("Nebius CLI profile is ready.")
    else:
        typer.echo(
            "Could not verify a Nebius profile. Run `nebius profile create` manually, "
            "then `nebius iam get-access-token` to confirm."
        )


def _run_interactive_configure() -> None:
    """Prompt for credentials/config and write the NPA dotfiles."""

    from npa.clients.config import CONFIG_PATH, write_config
    from npa.clients.credentials import write_credentials_file
    from npa.deploy.images import DEFAULT_CONTAINER_REGISTRY

    typer.echo("Interactive npa setup. Press Enter to skip any optional field.\n")
    _ensure_nebius_profile()
    typer.echo("")

    def ask(label: str, *, default: str = "", secret: bool = False) -> str:
        return str(
            typer.prompt(
                label,
                default=default,
                hide_input=secret,
                show_default=bool(default) and not secret,
            )
        ).strip()

    hf_token = ask("Hugging Face token (HF_TOKEN)", secret=True)
    s3_access_key = ask("S3 access key id (AWS_ACCESS_KEY_ID)", secret=True)
    s3_secret_key = ask("S3 secret access key (AWS_SECRET_ACCESS_KEY)", secret=True)
    s3_endpoint = ask("S3 endpoint URL", default=DEFAULT_STORAGE_ENDPOINT)
    s3_bucket = ask("S3 bucket (e.g. s3://my-bucket/)")
    registry = ask("Container registry", default=DEFAULT_CONTAINER_REGISTRY)
    project_id = ask("Nebius project id")
    tenant_id = ask("Nebius tenant id")
    region = ask("Region", default=DEFAULT_REGION)

    credentials_path = write_credentials_file(
        {
            "tokens": {"HF_TOKEN": hf_token},
            "storage": {
                "aws_access_key_id": s3_access_key,
                "aws_secret_access_key": s3_secret_key,
                "endpoint_url": s3_endpoint,
                "bucket": s3_bucket,
            },
        }
    )

    project_stanza = {
        key: value
        for key, value in (
            ("project_id", project_id),
            ("tenant_id", tenant_id),
            ("region", region),
            ("container_registry", registry),
        )
        if value
    }
    wrote_config = False
    if project_id or tenant_id:
        alias = region or "default"
        write_config({"projects": {alias: project_stanza}, "default_project": alias})
        wrote_config = True

    typer.echo(f"\nWrote {credentials_path} (chmod 600).")
    if wrote_config:
        typer.echo(f"Wrote {CONFIG_PATH}.")
    else:
        typer.echo(
            "Skipped ~/.npa/config.yaml: provide a Nebius project id to write a "
            "project profile."
        )
    typer.echo("Setup complete. Run `npa configure --show` to see the file layout.")


def _configure_impl(*, show: bool, interactive: Optional[bool]) -> None:
    if show:
        typer.echo(_SETUP_GUIDANCE)
        return
    should_prompt = interactive if interactive is not None else sys.stdin.isatty()
    if not should_prompt:
        typer.echo(_SETUP_GUIDANCE)
        return
    try:
        _run_interactive_configure()
    except (EOFError, typer.Abort):
        typer.echo("\n")
        typer.echo(_SETUP_GUIDANCE)


@app.command(
    "configure",
    help="Interactive credential and config setup guidance.",
    rich_help_panel="Setup",
)
def configure(
    show: bool = typer.Option(
        False,
        "--show",
        help="Print the credential/config file layout instead of prompting.",
    ),
    interactive: Optional[bool] = typer.Option(
        None,
        "--interactive/--no-interactive",
        help="Force or disable interactive prompting (defaults to auto-detect TTY).",
    ),
) -> None:
    """Interactively write ~/.npa credentials and config, or show guidance."""
    _configure_impl(show=show, interactive=interactive)


@app.command(
    "init",
    help="Interactive credential and config setup guidance.",
    rich_help_panel="Setup",
)
def init(
    show: bool = typer.Option(
        False,
        "--show",
        help="Print the credential/config file layout instead of prompting.",
    ),
    interactive: Optional[bool] = typer.Option(
        None,
        "--interactive/--no-interactive",
        help="Force or disable interactive prompting (defaults to auto-detect TTY).",
    ),
) -> None:
    """Interactively write ~/.npa credentials and config, or show guidance."""
    _configure_impl(show=show, interactive=interactive)


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
