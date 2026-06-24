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
from npa.cli.provision import app as provision_app
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
app.add_typer(provision_app, name="provision-if-absent", rich_help_panel="Setup")
app.add_typer(rerun_app, name="rerun", rich_help_panel="Platform utilities")
app.add_typer(skypilot_app, name="skypilot", rich_help_panel="Platform utilities")
app.add_typer(viz_app, name="viz", rich_help_panel="Platform utilities")
app.add_typer(workflow_shim_app, name="workflow", hidden=True)


DEFAULT_REGION = "eu-north1"
# Recommended default cap for an auto-created object-storage bucket.
RECOMMENDED_BUCKET_SIZE_GB = 50
DEFAULT_BUCKET_STORAGE_CLASS = "standard"

_SETUP_GUIDANCE = """Credential setup

Run `npa configure` in a terminal for interactive setup (use
`npa configure --interactive` when stdin is not a TTY). The flow uses the
installed Nebius CLI binary internally (profile setup stays inside
`npa configure`; no separate Nebius CLI onboarding commands), bootstraps a profile
when needed, then with an authenticated profile
auto-creates an S3 bucket (and access key) when you press Enter at the bucket
prompt, so you supply your Nebius tenant id, project id, and region plus optional
bucket name, storage class (standard or enhanced), bucket size, Hugging Face,
Token Factory, and NGC tokens. Use `npa configure --no-provision` to enter
existing S3 credentials instead, or create ~/.npa/credentials.yaml by hand for
user-level tokens, object storage, and BYOVM SSH defaults:

tokens:
  HF_TOKEN: hf_REPLACE_ME
  # Optional: Nebius Token Factory API key (OpenAI-compatible hosted inference).
  # Get one at https://tokenfactory.nebius.com/ -> API keys. The key is a long
  # opaque token (it starts with "v1."); it is NOT your Nebius IAM/CLI token.
  NEBIUS_TOKEN_FACTORY_KEY: <paste-your-token-factory-api-key>  # e.g. v1.XXXXXXXX...
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


def _list_nebius_profiles(*, runner: Callable[..., object] = subprocess.run) -> list[str]:
    """Return local Nebius CLI profile names, or [] when listing is unavailable."""

    if not shutil.which("nebius"):
        return []
    try:
        result = runner(
            ["nebius", "profile", "list"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if getattr(result, "returncode", 1) != 0:
        return []
    stdout = getattr(result, "stdout", "") or ""
    profiles: list[str] = []
    for line in stdout.splitlines():
        name = line.strip().split(maxsplit=1)[0] if line.strip() else ""
        if name:
            profiles.append(name)
    return profiles


def _region_from_registry_host(registry: str) -> str:
    """Best-effort region from a container registry host such as cr.eu-north1.nebius.cloud."""

    host = (registry or "").split("/", 1)[0].strip()
    parts = host.split(".")
    if len(parts) >= 4 and parts[0] == "cr" and parts[2] == "nebius":
        return parts[1]
    return ""


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
            "Nebius CLI not found. Install the binary from "
            "https://docs.nebius.com/cli/install (onboarding stays in "
            "`npa configure`; no separate profile CLI steps), then re-run "
            "`npa configure`."
        )
        return
    existing_profiles = _list_nebius_profiles()
    if existing_profiles:
        typer.echo(
            "Nebius CLI profiles exist but `nebius iam get-access-token` failed. "
            "Try `nebius profile activate <profile>` or recreate the active profile."
        )
        create_prompt = "Create a new Nebius CLI profile now?"
        create_default = False
    else:
        create_prompt = "No authenticated Nebius CLI profile found. Create one now?"
        create_default = True
    if not typer.confirm(create_prompt, default=create_default):
        typer.echo(
            "Skipped Nebius profile creation. Re-run `npa configure` when ready "
            "to create or refresh a profile."
        )
        return
    if _create_nebius_profile() and _nebius_profile_ready():
        typer.echo("Nebius CLI profile is ready.")
    else:
        typer.echo(
            "Could not verify a Nebius profile. Re-run `npa configure` in a "
            "terminal to retry profile creation."
        )


def _endpoint_for_region(region: str) -> str:
    """Return the Nebius S3-compatible storage endpoint URL for *region*."""
    reg = (region or DEFAULT_REGION).strip() or DEFAULT_REGION
    return f"https://storage.{reg}.nebius.cloud"


def _gb_to_bytes(value: str) -> int:
    """Parse a GB amount into bytes; non-negative or invalid means unlimited (0)."""
    try:
        gb = float(str(value).strip())
    except (TypeError, ValueError):
        gb = float(RECOMMENDED_BUCKET_SIZE_GB)
    if gb <= 0:
        return 0
    return int(gb * 1024**3)


def _as_bucket_uri(name: str) -> str:
    """Normalize a bucket name to an ``s3://<name>/`` URI."""
    value = (name or "").strip()
    if not value:
        return ""
    if value.startswith("s3://"):
        return value
    return f"s3://{value.rstrip('/')}/"


def _prompt_new_bucket_settings(
    ask: Callable[..., str],
    *,
    bucket_name: str,
) -> tuple[str, int]:
    """Prompt for storage class and size when creating a new bucket."""

    from npa.clients import nebius as nebius_client

    storage_raw = ask(
        "New bucket storage class (standard/enhanced)",
        default=DEFAULT_BUCKET_STORAGE_CLASS,
    )
    storage_class = nebius_client.normalize_bucket_storage_class(storage_raw)
    if storage_class == DEFAULT_BUCKET_STORAGE_CLASS:
        typer.echo("  Using standard storage (default).")
    size_gb = ask(
        f"New bucket size limit in GB (recommended {RECOMMENDED_BUCKET_SIZE_GB})",
        default=str(RECOMMENDED_BUCKET_SIZE_GB),
    )
    max_size_bytes = _gb_to_bytes(size_gb)
    if max_size_bytes == 0:
        typer.echo("  Using no size limit (unlimited, up to quota).")
    else:
        typer.echo(
            f"  Will create '{bucket_name}' with {storage_class} storage "
            f"and a {size_gb or RECOMMENDED_BUCKET_SIZE_GB} GB cap."
        )
    return storage_class, max_size_bytes


def _provision_object_storage(
    nebius_client,
    ask: Callable[..., str],
    *,
    project_id: str,
    tenant_id: str,
    region: str,
) -> dict[str, str] | None:
    """Auto-create the S3 bucket + access key for the project."""
    if not (project_id and tenant_id):
        return None

    typer.echo(
        "\nObject storage: enter an existing bucket name to reuse it, "
        "or press Enter to have npa create a default npa-bucket for this project."
    )
    bucket_name = ask("Object-storage bucket name")
    if not bucket_name:
        bucket_name = nebius_client.bucket_name_for(tenant_id, project_id)
        typer.echo("  No bucket name provided; npa will create a default bucket.")

    try:
        already_exists = nebius_client.bucket_exists(project_id, bucket_name)
    except Exception:
        already_exists = False

    bucket_max_size_bytes = 0
    bucket_storage_class = DEFAULT_BUCKET_STORAGE_CLASS
    if already_exists:
        typer.echo(f"Reusing existing object-storage bucket '{bucket_name}'.")
    else:
        bucket_storage_class, bucket_max_size_bytes = _prompt_new_bucket_settings(
            ask,
            bucket_name=bucket_name,
        )

    try:
        typer.echo("Provisioning Nebius object storage (bucket + access key)...")
        creds = nebius_client.bootstrap_environment(
            project_id,
            tenant_id,
            region,
            bucket_name=bucket_name,
            bucket_max_size_bytes=bucket_max_size_bytes,
            bucket_storage_class=bucket_storage_class,
            on_status=lambda msg: typer.echo(f"  - {msg}"),
        )
    except nebius_client.NebiusError as exc:
        typer.echo(f"  Could not auto-provision object storage: {exc}")
        return None
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"  Could not auto-provision object storage: {exc}")
        return None

    access_key = creds.get("nebius_api_key", "")
    secret_key = creds.get("nebius_secret_key", "")
    if not (access_key and secret_key):
        typer.echo("  Provisioning did not return usable S3 credentials.")
        return None

    bucket = _as_bucket_uri(creds.get("s3_bucket", ""))
    typer.echo(f"  Provisioned bucket {bucket} and an S3 access key.")
    payload: dict[str, str] = {
        "aws_access_key_id": access_key,
        "aws_secret_access_key": secret_key,
        "endpoint_url": creds.get("s3_endpoint", "") or _endpoint_for_region(region),
        "bucket": bucket,
    }
    sa_id = creds.get("service_account_id", "").strip()
    if sa_id:
        payload["service_account_id"] = sa_id
    return payload


def _run_interactive_configure(*, provision: bool = True) -> None:
    """Prompt for credentials/config and write the NPA dotfiles."""

    from npa.clients.config import CONFIG_PATH, write_config
    from npa.clients.credentials import write_credentials_file
    from npa.clients import nebius as nebius_client
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

    project_id = ask("Nebius project id")
    tenant_id = ask("Nebius tenant id")
    registry_default = (
        nebius_client.discover_container_registry(project_id)
        or DEFAULT_CONTAINER_REGISTRY
    )
    region_default = _region_from_registry_host(registry_default) or DEFAULT_REGION
    region = ask("Region", default=region_default)
    registry = ask("Container registry", default=registry_default)

    storage: dict[str, str] | None = None
    if provision and project_id and tenant_id:
        storage = _provision_object_storage(
            nebius_client,
            ask,
            project_id=project_id,
            tenant_id=tenant_id,
            region=region,
        )
        if storage is None:
            typer.echo(
                "\nFalling back to manual object-storage entry. "
                "Provide existing S3 credentials or press Enter to skip."
            )
    if storage is None:
        storage = {
            "aws_access_key_id": ask(
                "S3 access key id (AWS_ACCESS_KEY_ID)", secret=True
            ),
            "aws_secret_access_key": ask(
                "S3 secret access key (AWS_SECRET_ACCESS_KEY)", secret=True
            ),
            "endpoint_url": ask("S3 endpoint URL", default=_endpoint_for_region(region)),
            "bucket": ask("S3 bucket URI (e.g. s3://<your-bucket>/)"),
        }

    hf_token = ask("Hugging Face token (HF_TOKEN)", secret=True)
    nebius_api_key = ask(
        "Nebius Token Factory API key (NEBIUS_TOKEN_FACTORY_KEY, optional)", secret=True
    )
    ngc_api_key = ask("NVIDIA NGC API key (NGC_API_KEY)", secret=True)

    credentials_payload: dict[str, object] = {
        "tokens": {
            "HF_TOKEN": hf_token,
            "NEBIUS_TOKEN_FACTORY_KEY": nebius_api_key,
        },
        "ngc": {"api_key": ngc_api_key},
        "storage": {
            key: value
            for key, value in storage.items()
            if key != "service_account_id" and value
        },
    }
    sa_id = str(storage.get("service_account_id", "") or "").strip()
    if sa_id:
        credentials_payload["nebius"] = {"service_account_id": sa_id}

    credentials_path = write_credentials_file(credentials_payload)

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


def _store_token_factory_key(api_key: str) -> None:
    from npa.clients.credentials import set_token_factory_api_key

    path = set_token_factory_api_key(api_key)
    typer.echo(
        f"Stored Nebius Token Factory API key in {path} under tokens.NEBIUS_TOKEN_FACTORY_KEY."
    )


def _configure_impl(
    *,
    show: bool,
    interactive: Optional[bool],
    provision: bool = True,
    token_factory_key: str = "",
) -> None:
    if token_factory_key.strip():
        _store_token_factory_key(token_factory_key.strip())
        return
    if show:
        typer.echo(_SETUP_GUIDANCE)
        return
    should_prompt = interactive if interactive is not None else sys.stdin.isatty()
    if not should_prompt:
        typer.echo(_SETUP_GUIDANCE)
        return
    try:
        _run_interactive_configure(provision=provision)
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
    provision: bool = typer.Option(
        True,
        "--provision/--no-provision",
        help=(
            "Auto-create a Nebius S3 bucket (when missing) and an access key "
            "(default). Reuse an existing bucket by name, or press Enter to "
            "create a default npa-bucket with standard storage and a size cap. "
            "Use --no-provision to enter existing S3 credentials."
        ),
    ),
    token_factory_key: str = typer.Option(
        "",
        "--token-factory-key",
        help=(
            "Store a Nebius Token Factory API key in ~/.npa/credentials.yaml "
            "under tokens.NEBIUS_TOKEN_FACTORY_KEY (skips interactive setup)."
        ),
    ),
) -> None:
    """Interactively write ~/.npa credentials and config, or show guidance."""
    _configure_impl(
        show=show,
        interactive=interactive,
        provision=provision,
        token_factory_key=token_factory_key,
    )


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
    provision: bool = typer.Option(
        True,
        "--provision/--no-provision",
        help=(
            "Auto-create a Nebius S3 bucket (when missing) and an access key "
            "(default). Reuse an existing bucket by name, or press Enter to "
            "create a default npa-bucket with standard storage and a size cap. "
            "Use --no-provision to enter existing S3 credentials."
        ),
    ),
    token_factory_key: str = typer.Option(
        "",
        "--token-factory-key",
        help=(
            "Store a Nebius Token Factory API key in ~/.npa/credentials.yaml "
            "under tokens.NEBIUS_TOKEN_FACTORY_KEY (skips interactive setup)."
        ),
    ),
) -> None:
    """Interactively write ~/.npa credentials and config, or show guidance."""
    _configure_impl(
        show=show,
        interactive=interactive,
        provision=provision,
        token_factory_key=token_factory_key,
    )


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
