"""npa CLI entry point."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import traceback
from importlib.metadata import version as package_version
from typing import Any, Callable, Iterable, Optional

import typer

from npa.cli._error_formatting import format_error_for_user
from npa.cli.agent import app as agent_app
from npa.cli.burst import app as burst_app
from npa.cli.workbench import app as workbench_app
from npa.cli.adapter import app as adapter_app
from npa.cli.cluster import app as cluster_app
from npa.cli.convert import app as convert_app
from npa.cli.demo import app as demo_app
from npa.cli.network import app as network_app
from npa.cli.provision import app as provision_app
from npa.cli.rerun import app as rerun_app
from npa.cli.skypilot import app as skypilot_app
from npa.cli.soperator import app as soperator_app
from npa.cli.viz import app as viz_app
from npa.cli.workflow_shim import workflow_shim_app
from npa.clients.serverless import ServerlessClientError

logger = logging.getLogger(__name__)

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
app.add_typer(agent_app, name="agent", rich_help_panel="Platform utilities")
app.add_typer(burst_app, name="burst", rich_help_panel="Platform utilities")
app.add_typer(cluster_app, name="cluster", rich_help_panel="Platform utilities")
app.add_typer(convert_app, name="convert", rich_help_panel="Platform utilities")
app.add_typer(demo_app, name="demo", rich_help_panel="Platform utilities")
app.add_typer(network_app, name="network", rich_help_panel="Platform utilities")
app.add_typer(provision_app, name="provision-if-absent", rich_help_panel="Setup")
app.add_typer(rerun_app, name="rerun", rich_help_panel="Platform utilities")
app.add_typer(skypilot_app, name="skypilot", rich_help_panel="Platform utilities")
app.add_typer(soperator_app, name="soperator", rich_help_panel="Platform utilities")
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
  # Hugging Face access token (for model + dataset downloads, incl. gated repos).
  # Get one at https://huggingface.co/settings/tokens -> "Create new token"
  # (a "Read" token is enough). For gated models (e.g. Llama, some GR00T assets),
  # also click "Agree and access repository" on each model page while signed in.
  # Step-by-step guide: docs/workbench/huggingface-token.md
  HF_TOKEN: hf_REPLACE_ME
  # Optional: Nebius Token Factory API key (OpenAI-compatible hosted inference).
  # Get one at https://tokenfactory.nebius.com/ -> API keys. The key is a long
  # opaque token (it starts with "v1."); it is NOT your Nebius IAM/CLI token.
  # Step-by-step guide: docs/workbench/token-factory-key.md
  NEBIUS_TOKEN_FACTORY_KEY: <paste-your-token-factory-api-key>  # e.g. v1.XXXXXXXX...
ngc:
  # NVIDIA NGC API key (for GR00T / Cosmos NVIDIA container + model pulls).
  # Get one at https://org.ngc.nvidia.com/setup/api-key -> "Generate API Key"
  # (sign in / create a free NGC account first). The key starts with "nvapi-".
  # Step-by-step guide: docs/workbench/ngc-api-key.md
  api_key: nvapi-REPLACE_ME
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
    from npa.clients.nebius import nebius_cli_env

    try:
        result = runner(
            ["nebius", "iam", "get-access-token"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
            # Sanitize a stale NEBIUS_IAM_TOKEN / NEBIUS_IAM_TOKEN_FILE so
            # readiness reflects the active profile — the same identity
            # provisioning actually uses. Otherwise a shadowing token lets the
            # CLI skip a real token exchange and falsely report "profile ready"
            # (or "not ready") regardless of whether the profile itself works.
            env=nebius_cli_env(),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return getattr(result, "returncode", 1) == 0


def _list_nebius_profiles(*, runner: Callable[..., object] = subprocess.run) -> list[str]:
    """Return local Nebius CLI profile names, or [] when listing is unavailable."""

    if not shutil.which("nebius"):
        return []
    from npa.clients.nebius import nebius_cli_env

    try:
        result = runner(
            ["nebius", "profile", "list"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=nebius_cli_env(),
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


def _ensure_nebius_profile() -> bool:
    """Detect or interactively create a local Nebius CLI profile.

    Returns ``True`` when a usable, authenticated Nebius CLI profile is available
    (either already present or created during this call), ``False`` otherwise.
    Callers use the result to decide whether object-storage auto-provisioning,
    which needs an authenticated profile, can proceed.
    """

    if _nebius_profile_ready():
        typer.echo("Nebius CLI profile detected (`nebius iam get-access-token` works).")
        return True
    if not shutil.which("nebius"):
        typer.echo(
            "Nebius CLI not found. Install the binary from "
            "https://docs.nebius.com/cli/install (onboarding stays in "
            "`npa configure`; no separate profile CLI steps), then re-run "
            "`npa configure`."
        )
        return False
    existing_profiles = _list_nebius_profiles()
    if existing_profiles:
        typer.echo(
            "Nebius CLI profiles exist but `nebius iam get-access-token` failed. "
            "Run `nebius iam get-access-token` in this shell to see the error. "
            "Common causes: no active profile (`nebius profile activate "
            "<profile>`), or a stale ambient token — `unset NEBIUS_IAM_TOKEN "
            "NEBIUS_IAM_TOKEN_FILE` and retry. Otherwise recreate the profile."
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
        return False
    if _create_nebius_profile() and _nebius_profile_ready():
        typer.echo("Nebius CLI profile is ready.")
        return True
    typer.echo(
        "Could not verify a Nebius profile. Re-run `npa configure` in a "
        "terminal to retry profile creation."
    )
    return False


def _endpoint_for_region(region: str) -> str:
    """Return the Nebius S3-compatible storage endpoint URL for *region*."""
    reg = (region or DEFAULT_REGION).strip() or DEFAULT_REGION
    return f"https://storage.{reg}.nebius.cloud"


def _normalize_pasted_secret(value: str) -> str:
    """Clean a pasted credential: drop wrapping quotes and auth prefixes.

    Users routinely paste a token copied from a curl example or a password
    manager with surrounding quotes or an ``Authorization: Bearer`` prefix. Those
    silently break auth (the stored value is not the bare token), so strip them.
    """
    text = (value or "").strip()
    # Unwrap matching surrounding quotes (may wrap a "Bearer ..." string).
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1].strip()
    # Drop a pasted "Authorization:" header label.
    if text.lower().startswith("authorization:"):
        text = text.split(":", 1)[1].strip()
    # Drop a leading auth scheme (Bearer/Token), case-insensitively.
    for scheme in ("bearer ", "token "):
        if text.lower().startswith(scheme):
            text = text[len(scheme):].strip()
            break
    # Unwrap again in case the scheme was inside the quotes.
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1].strip()
    return text


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


def _bucket_name_from_uri(bucket: str) -> str:
    """Extract the bare bucket name from an ``s3://bucket/prefix`` URI."""
    value = (bucket or "").strip()
    if not value:
        return ""
    if "://" in value:
        value = value.split("://", 1)[1]
    return value.strip("/").split("/", 1)[0]


def _provision_object_storage(
    nebius_client,
    ask: Callable[..., str],
    *,
    project_id: str,
    tenant_id: str,
    region: str,
    existing_bucket: str = "",
) -> dict[str, str] | None:
    """Auto-create the S3 bucket + access key for the project."""
    if not (project_id and tenant_id):
        return None

    typer.echo(
        "\nObject storage: enter a bucket name to reuse it (or create it if it "
        "does not exist yet), or press Enter to use npa's default bucket for this "
        "project. The default name is derived from your tenant + project, so it "
        "is stable across runs and reused rather than duplicated."
    )
    bucket_name = ask("Object-storage bucket name", default=existing_bucket).strip()
    if not bucket_name:
        bucket_name = nebius_client.bucket_name_for(tenant_id, project_id)
        typer.echo(
            f"  No bucket name provided; using npa's default bucket "
            f"'{bucket_name}' (reused if it already exists)."
        )

    # Whether the named bucket already exists: True (reuse), False (create), or
    # None when the search itself could not run. Only prompt for new-bucket
    # settings when we know the bucket is absent — provisioning below reuses an
    # existing bucket and creates one only if it does not exist yet.
    try:
        exists: bool | None = nebius_client.bucket_exists(project_id, bucket_name)
    except Exception as exc:  # noqa: BLE001 - search is best-effort
        exists = None
        typer.echo(
            f"  Could not verify whether '{bucket_name}' already exists ({exc}); "
            "npa will reuse it if present or create it otherwise."
        )

    bucket_max_size_bytes = 0
    bucket_storage_class = DEFAULT_BUCKET_STORAGE_CLASS
    if exists is True:
        typer.echo(f"Reusing existing object-storage bucket '{bucket_name}'.")
    elif exists is False:
        typer.echo(f"No existing bucket named '{bucket_name}' found; npa will create it.")
        bucket_storage_class, bucket_max_size_bytes = _prompt_new_bucket_settings(
            ask,
            bucket_name=bucket_name,
        )
    # exists is None: existence unknown, so skip the create-only prompts and let
    # provisioning get-or-create with defaults rather than risk creating a
    # duplicate of a bucket that may already exist.

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
        if nebius_client.is_permission_denied(str(exc)):
            typer.echo(
                "  Could not auto-provision object storage: the Nebius storage "
                "service denied the request (AccessDenied)."
            )
            typer.echo(
                "  npa runs the CLI as your active Nebius profile (any stale "
                "NEBIUS_IAM_TOKEN in the environment is ignored). Reproduce with:\n"
                f"    nebius storage bucket list --parent-id {project_id}"
            )
            typer.echo(
                "  If that also fails, grant your identity the project 'editors' "
                "role (or enable object storage for it) and re-run `npa configure` "
                "— a newly granted role can take ~a minute to propagate. If it "
                "succeeds but npa still fails, unset NPA_REUSE_IAM_TOKEN so npa "
                "does not reuse an injected token."
            )
            typer.echo(f"  Underlying error: {exc}")
        else:
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


def _prompt_setup_tokens(
    ask: Callable[..., str],
    existing_credentials: Any,
) -> tuple[str, str, str]:
    """Prompt for the optional HF / Token Factory / NGC keys. Returns the trio."""
    typer.echo(
        "\nHugging Face token: create a Read token at "
        "https://huggingface.co/settings/tokens (it starts with 'hf_'). "
        "For gated models, also click 'Agree and access repository' on each "
        "model page while signed in. Guide: docs/workbench/huggingface-token.md."
    )
    hf_token = _normalize_pasted_secret(
        ask(
            "Hugging Face token (HF_TOKEN)",
            default=existing_credentials.hf_token,
            secret=True,
        )
    )
    typer.echo(
        "\nNebius Token Factory API key (optional): OpenAI-compatible hosted "
        "inference, zero GPU. Create one at https://tokenfactory.nebius.com/ -> "
        "API keys. It starts with 'v1.' and is NOT your Nebius IAM/CLI token. "
        "Guide: docs/workbench/token-factory-key.md."
    )
    token_factory_api_key = _normalize_pasted_secret(
        ask(
            "Nebius Token Factory API key (NEBIUS_TOKEN_FACTORY_KEY, optional)",
            default=existing_credentials.token_factory_api_key,
            secret=True,
        )
    )
    if token_factory_api_key and not token_factory_api_key.startswith("v1."):
        typer.echo(
            "  Warning: that does not look like a Token Factory key (they start "
            "with 'v1.'). It is a separate credential from your Nebius IAM/CLI "
            "token — pasting an IAM token here returns 403. Verify with "
            "`npa workbench token-factory verify`; see "
            "docs/workbench/token-factory-key.md."
        )
    typer.echo(
        "\nNVIDIA NGC API key (for GR00T / Cosmos NVIDIA assets): create one at "
        "https://org.ngc.nvidia.com/setup/api-key (sign in or make a free NGC "
        "account first). The key starts with 'nvapi-'. "
        "Guide: docs/workbench/ngc-api-key.md."
    )
    ngc_api_key = _normalize_pasted_secret(
        ask(
            "NVIDIA NGC API key (NGC_API_KEY)",
            default=existing_credentials.ngc_api_key,
            secret=True,
        )
    )
    return hf_token, token_factory_api_key, ngc_api_key


def _slugify_alias(name: str, project_id: str) -> str:
    """Return a filesystem/flag-friendly local alias for a discovered project."""
    import re as _re

    slug = _re.sub(r"[^a-z0-9-]+", "-", str(name or "").strip().lower()).strip("-")
    if slug:
        return slug
    tail = str(project_id or "").split("-")[-1][:8]
    return f"project-{tail}" if tail else "default"


def _parse_selection(raw: str, count: int) -> list[int]:
    """Parse a 1-based index selection like ``1,3 4`` or ``all`` into 0-based ints."""
    text = raw.strip().lower()
    if text in ("", "all", "*"):
        return list(range(count))
    picked: list[int] = []
    for token in text.replace(",", " ").split():
        if not token.isdigit():
            continue
        idx = int(token) - 1
        if 0 <= idx < count and idx not in picked:
            picked.append(idx)
    return picked


def _select_discovered_projects(
    projects: list[dict[str, str]],
    ask: Callable[..., str],
    nebius_client: Any,
    *,
    current_project_id: str = "",
) -> tuple[list[tuple[str, dict[str, str]]], str]:
    """Present discovered projects and return ``([(alias, stanza)...], default_alias)``.

    Auto-derives tenant/project/region from each pick and best-effort discovers
    the container registry. npa is multi-project: the user may select several.
    """
    from npa.deploy.images import DEFAULT_CONTAINER_REGISTRY

    # Large Nebius accounts can expose hundreds/thousands of projects. Dumping
    # them all (and defaulting to 'all', which then discovers a registry per
    # project) is unusable, so offer a name/id filter and cap the printed list.
    display_cap = 40
    shown = list(projects)
    if len(shown) > display_cap:
        typer.echo(
            f"{len(shown)} projects are accessible. Type part of a project name or "
            "id to filter (blank = just the current/first project)."
        )
        needle = ask("Filter projects", default="").strip().lower()
        if needle:
            filtered = [
                proj
                for proj in projects
                if needle in (proj.get("name", "") or "").lower()
                or needle in proj["id"].lower()
            ]
            shown = filtered or list(projects)
        else:
            current = [proj for proj in projects if proj["id"] == current_project_id]
            shown = current or list(projects)
        if len(shown) > display_cap:
            typer.echo(
                f"Showing the first {display_cap} of {len(shown)} matches; refine the "
                "filter to narrow further."
            )
            shown = shown[:display_cap]

    typer.echo("Nebius projects accessible with your profile:\n")
    for i, proj in enumerate(shown, start=1):
        marker = "  *" if proj["id"] == current_project_id else "   "
        region = proj.get("region", "") or "?"
        typer.echo(
            f"{marker}{i:>2}. {proj.get('name', '') or proj['id']} "
            f"({region})  [{proj['id']}]"
        )
    # Default to the current project (never 'all', which would configure every
    # discovered project and run per-project registry discovery).
    default_pick = "1"
    for i, proj in enumerate(shown, start=1):
        if proj["id"] == current_project_id:
            default_pick = str(i)
            break
    raw = ask(
        "\nSelect project(s) to configure (comma-separated numbers, or 'all')",
        default=default_pick,
    )
    chosen = _parse_selection(raw, len(shown)) or _parse_selection(default_pick, len(shown))

    selected: list[tuple[str, dict[str, str]]] = []
    used_aliases: set[str] = set()
    for idx in chosen:
        proj = shown[idx]
        alias = _slugify_alias(proj.get("name", ""), proj["id"])
        base_alias = alias
        suffix = 2
        while alias in used_aliases:
            alias = f"{base_alias}-{suffix}"
            suffix += 1
        used_aliases.add(alias)
        registry = nebius_client.discover_container_registry(proj["id"]) or DEFAULT_CONTAINER_REGISTRY
        stanza = {
            "project_id": proj["id"],
            "tenant_id": proj["tenant_id"],
            "region": proj.get("region", "") or DEFAULT_REGION,
            "container_registry": registry,
        }
        selected.append((alias, stanza))

    if not selected:
        return [], ""
    if len(selected) == 1:
        return selected, selected[0][0]
    aliases = [alias for alias, _ in selected]
    default_alias = ask(
        f"Which project is the default for bare `-p`-less commands? ({', '.join(aliases)})",
        default=aliases[0],
    )
    if default_alias not in aliases:
        default_alias = aliases[0]
    return selected, default_alias


def _run_interactive_configure(*, provision: bool = True) -> None:
    """Prompt for credentials/config and write the NPA dotfiles."""

    from npa.clients.config import (
        CONFIG_PATH,
        default_project_name,
        list_projects,
        write_config,
    )
    from npa.clients.credentials import load_credentials, write_credentials_file
    from npa.clients import nebius as nebius_client
    from npa.deploy.images import DEFAULT_CONTAINER_REGISTRY

    typer.echo(
        "Interactive npa setup. Existing values are shown as defaults — press "
        "Enter to keep them, or type a new value to update.\n"
    )
    profile_ready = _ensure_nebius_profile()
    typer.echo("")

    # Object-storage auto-provisioning needs an authenticated Nebius CLI profile.
    # Without one, the flow used to fall through ~10 prompts (including a manual
    # S3 access-key entry a brand-new user cannot answer) and then abort writing
    # nothing. Stop up front with an actionable choice instead.
    if provision and not profile_ready:
        typer.echo(
            "Object-storage auto-provisioning needs an authenticated Nebius CLI "
            "profile, which is not available yet. Choose one of:\n"
            "  - Install and authenticate the Nebius CLI "
            "(https://docs.nebius.com/cli/install), then re-run `npa configure`.\n"
            "  - Re-run `npa configure --no-provision` to enter existing S3 "
            "credentials manually.\n"
            "Nothing was written under ~/.npa."
        )
        raise typer.Exit(code=1)

    def ask(label: str, *, default: str = "", secret: bool = False) -> str:
        return str(
            typer.prompt(
                label,
                default=default,
                hide_input=secret,
                show_default=bool(default) and not secret,
            )
        ).strip()

    existing_credentials = load_credentials(environ={})

    # Re-running configure should be idempotent: default every prompt to the
    # values already saved so pressing Enter keeps the current setup, while
    # typing a new value updates it. Config is deep-merged on write.
    existing_projects = list_projects()
    existing_default_alias = default_project_name()
    existing_stanza = existing_projects.get(existing_default_alias, {}) or {}

    # Prefer discovering accessible projects via the Nebius CLI so the user picks
    # from a list instead of hand-typing tenant + project ids (and region). npa
    # is not confined to one project: several may be selected and written. When
    # discovery is unavailable (no CLI / not authenticated / no results) we fall
    # back to the manual prompts below, which also covers the offline unit tests.
    discovered_selection: list[tuple[str, dict[str, str]]] = []
    discovered_default_alias = ""
    # Scope discovery to the active profile's tenant. Enumerating every tenant
    # (list_accessible_projects) is O(tenants) serial CLI calls — hundreds of
    # tenants take minutes and dump thousands of projects. The profile's own
    # tenant holds the projects the operator actually deploys into; other tenants
    # are reachable by switching the Nebius profile and re-running configure.
    current_tenant = nebius_client.current_tenant_id() if profile_ready else ""
    discovered_projects = (
        nebius_client.list_projects_in_tenant(current_tenant)
        if (profile_ready and current_tenant)
        else []
    )
    if discovered_projects:
        discovered_selection, discovered_default_alias = _select_discovered_projects(
            discovered_projects,
            ask,
            nebius_client,
            current_project_id=nebius_client.current_project_id(),
        )

    if discovered_selection:
        default_stanza = dict(
            next(
                stanza
                for alias, stanza in discovered_selection
                if alias == discovered_default_alias
            )
        )
        tenant_id = str(default_stanza.get("tenant_id", ""))
        project_id = str(default_stanza.get("project_id", ""))
        region = str(default_stanza.get("region", "") or DEFAULT_REGION)
        registry = str(default_stanza.get("container_registry", ""))
    else:
        # Tenant is the parent of the project, so ask for it first.
        tenant_id = ask("Nebius tenant id", default=str(existing_stanza.get("tenant_id", "")))
        project_id = ask("Nebius project id", default=str(existing_stanza.get("project_id", "")))
        existing_registry = str(existing_stanza.get("container_registry", ""))
        # The main NPA registry (workbench images) is in eu-north1, and registries
        # are readable cross-region, so default to eu-north1: keep a saved registry,
        # else use the project's own eu-north1 registry if it has one, else the
        # eu-north1 first-party default. A discovered non-eu-north1 registry is not
        # auto-selected as the default (the operator can still type it). Only hit
        # Nebius for discovery when nothing is saved (idempotent re-runs stay offline).
        if existing_registry:
            registry_default = existing_registry
        else:
            discovered = nebius_client.discover_container_registry(project_id)
            if discovered and _region_from_registry_host(discovered) == DEFAULT_REGION:
                registry_default = discovered
            else:
                registry_default = DEFAULT_CONTAINER_REGISTRY
        region_default = (
            str(existing_stanza.get("region", ""))
            or _region_from_registry_host(registry_default)
            or DEFAULT_REGION
        )
        region = ask("Region", default=region_default)
        # The registry host region is only used as a sensible default guess for the
        # region above; it is not a constraint. Container registries are readable
        # cross-region and a project can hold registries in several regions, so we do
        # not warn when the chosen region differs from the registry's region.
        registry = ask("Container registry", default=registry_default)

    storage: dict[str, str] | None = None

    # Object storage is opt-in: `npa configure` sets up the Nebius connection and
    # optional model/inference tokens. Storage (an S3 bucket + access key) is only
    # needed by workbench data workflows, so when projects were discovered we ask
    # before provisioning instead of doing it by default.
    if discovered_selection and provision:
        want_storage = ask(
            "Set up object storage (S3 bucket + access key) now? "
            "(needed for data/checkpoint workflows; you can add it later) [y/N]",
            default="N",
        )
        if want_storage.lower() not in ("y", "yes"):
            provision = False
            storage = {}
            typer.echo(
                "  Skipping object storage. Re-run `npa configure` later to add it."
            )

    existing_has_storage = bool(
        existing_credentials.s3_access_key_id
        and existing_credentials.s3_secret_access_key
        and existing_credentials.s3_bucket
    )
    provisioning_failed = False
    if provision and project_id and tenant_id:
        if existing_has_storage:
            # Reuse already-provisioned storage by default so a re-run does not
            # mint a fresh S3 access key each time.
            keep = ask(
                f"Keep existing object storage ({existing_credentials.s3_bucket})? [Y/n]",
                default="Y",
            )
            if keep.lower() in ("", "y", "yes"):
                storage = {
                    "aws_access_key_id": existing_credentials.s3_access_key_id,
                    "aws_secret_access_key": existing_credentials.s3_secret_access_key,
                    "endpoint_url": existing_credentials.s3_endpoint
                    or _endpoint_for_region(region),
                    "bucket": existing_credentials.s3_bucket,
                }
                typer.echo("  Keeping existing object-storage credentials.")
        if storage is None:
            storage = _provision_object_storage(
                nebius_client,
                ask,
                project_id=project_id,
                tenant_id=tenant_id,
                region=region,
                existing_bucket=_bucket_name_from_uri(existing_credentials.s3_bucket),
            )
            provisioning_failed = storage is None

    # When auto-provisioning was attempted and failed (e.g. storage AccessDenied),
    # don't dead-end at a manual S3 prompt a first-time user can't answer. Offer
    # to skip storage and still write tokens + project config, so non-storage
    # workflows (Token Factory, health checks) work immediately.
    if storage is None and provisioning_failed:
        skip = ask(
            "Skip object storage for now and finish setup? "
            "(you can add it later by re-running `npa configure`) [Y/n]",
            default="Y",
        )
        if skip.lower() in ("", "y", "yes"):
            storage = {}
            typer.echo(
                "  Skipping object storage. Tokens and project config will still "
                "be written; re-run `npa configure` once storage access is granted."
            )
        else:
            typer.echo(
                "Enter existing S3 credentials (or press Enter to leave blank)."
            )
    if storage is None:
        storage = {
            "aws_access_key_id": ask(
                "S3 access key id (AWS_ACCESS_KEY_ID)",
                default=existing_credentials.s3_access_key_id,
                secret=True,
            ),
            "aws_secret_access_key": ask(
                "S3 secret access key (AWS_SECRET_ACCESS_KEY)",
                default=existing_credentials.s3_secret_access_key,
                secret=True,
            ),
            "endpoint_url": ask(
                "S3 endpoint URL",
                default=existing_credentials.s3_endpoint or _endpoint_for_region(region),
            ),
            "bucket": ask(
                "S3 bucket URI (e.g. s3://<your-bucket>/)",
                default=existing_credentials.s3_bucket,
            ),
        }

    hf_token, token_factory_api_key, ngc_api_key = _prompt_setup_tokens(
        ask, existing_credentials
    )

    credentials_payload: dict[str, object] = {
        "tokens": {
            "HF_TOKEN": hf_token,
            "NEBIUS_TOKEN_FACTORY_KEY": token_factory_api_key,
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

    wrote_config = False
    alias = ""
    if discovered_selection:
        # Multi-project aware: write every selected project stanza and point
        # default_project at the chosen default. No alias prompt — the alias is
        # derived from the Nebius project name.
        projects_payload = {
            sel_alias: {k: v for k, v in stanza.items() if v}
            for sel_alias, stanza in discovered_selection
        }
        alias = discovered_default_alias
        write_config(
            {"projects": projects_payload, "default_project": alias}
        )
        wrote_config = True
        if len(discovered_selection) > 1:
            typer.echo(
                f"\nConfigured {len(discovered_selection)} projects: "
                f"{', '.join(a for a, _ in discovered_selection)} "
                f"(default: {alias})."
            )
    elif project_id or tenant_id:
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
        # Local name for later `-p <alias>` flags. Derived automatically (no
        # prompt): reuse the existing default alias so a re-run updates the same
        # stanza, otherwise use the region so a first-time configure still
        # resolves. Multi-project users rename via ~/.npa/config.yaml.
        alias = (
            existing_default_alias
            if existing_default_alias and existing_default_alias != "default"
            else (region or "default")
        )
        write_config({"projects": {alias: project_stanza}, "default_project": alias})
        wrote_config = True

    typer.echo(f"\nWrote {credentials_path} (chmod 600).")
    if wrote_config:
        typer.echo(
            f"Wrote {CONFIG_PATH} (project alias: {alias}). "
            f"Pass `-p {alias}` to workbench commands, or omit `-p` to use this "
            "default."
        )
    else:
        typer.echo(
            "Skipped ~/.npa/config.yaml: provide a Nebius project id to write a "
            "project profile."
        )

    typer.echo(_model_access_note(hf_token, ngc_api_key))
    typer.echo("Setup complete. Run `npa configure --show` to see the file layout.")


def _probe_hf_repos_parallel(
    validator: Callable[..., Any],
    token: str,
    repos: Iterable[str],
    *,
    per_probe_timeout: float = 2.0,
    total_budget: float = 5.0,
) -> dict[str, Any]:
    """Probe HF access for *repos* concurrently within a wall-clock budget.

    Returns ``{repo: result}``. Repos that do not finish inside ``total_budget``
    (or whose probe raises) are omitted, so the caller treats them as unverified
    rather than stalling the primary onboarding command.
    """

    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FuturesTimeout
    from concurrent.futures import as_completed

    repo_list = list(repos)
    results: dict[str, Any] = {}
    if not repo_list:
        return results
    pool = ThreadPoolExecutor(max_workers=min(8, len(repo_list)))
    try:
        futures = {
            pool.submit(validator, token, repo, timeout=per_probe_timeout): repo
            for repo in repo_list
        }
        try:
            for fut in as_completed(futures, timeout=total_budget):
                repo = futures[fut]
                try:
                    results[repo] = fut.result()
                except Exception:  # noqa: BLE001 - a failed probe -> unverified
                    logger.debug("HF access probe failed for %s", repo, exc_info=True)
        except FuturesTimeout:
            # Budget exceeded: keep whatever finished; the rest stay unverified.
            logger.debug("HF access probe budget of %.1fs exceeded", total_budget)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return results


def _model_access_note(hf_token: str, ngc_key: str) -> str:
    """Return a one-line ``[NOTE]`` on which gated workbench models the tokens can access.

    Runs a live Hugging Face access check for each license-gated model the
    workbench uses and a presence/format check for the NGC key, then summarizes
    the models without access on a single line. HF probes run in parallel under a
    total wall-clock budget, and any failure is swallowed, so a preflight note
    can never stall or break `npa configure`.
    """

    try:
        from npa.clients import huggingface
        from npa.clients.huggingface import HFAccessResult
        from npa.workbench.model_access import (
            access_note,
            check_workbench_access,
            gated_hf_repos,
        )

        cache: dict[str, Any] = {}
        if hf_token:
            cache = _probe_hf_repos_parallel(
                huggingface.validate_hf_access, hf_token, gated_hf_repos()
            )

        def _validator(token: str, repo: str):
            return cache.get(repo) or HFAccessResult(
                repo=repo, ok=False, error="not verified (timed out)"
            )

        results = check_workbench_access(
            hf_token=hf_token,
            ngc_key=ngc_key,
            hf_validator=_validator if hf_token else None,
            gated_only=True,
        )
        return access_note(results)
    except Exception:  # noqa: BLE001 - a preflight note must never break configure
        return "[NOTE] Skipped model-access check (verify later with `npa workbench health access`)."


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
        # Store the key, then continue with the rest of configure. Returning here
        # left users thinking configure finished when no project/S3/HF/NGC/config
        # had been written.
        _store_token_factory_key(token_factory_key.strip())
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
        # Cancelling mid-flow (Ctrl-C / Ctrl-D / no more input) previously exited
        # 0 having written nothing under ~/.npa, so the next cloud command failed
        # mysteriously. Fail loudly instead so the missing setup is obvious.
        typer.echo("")
        typer.echo(
            "Setup was cancelled before anything was written under ~/.npa. "
            "Re-run `npa configure` in a terminal to finish, or run "
            "`npa configure --show` for the file layout to create it by hand."
        )
        raise typer.Exit(code=1)


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
            "under tokens.NEBIUS_TOKEN_FACTORY_KEY, then continue the rest of setup."
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
            "under tokens.NEBIUS_TOKEN_FACTORY_KEY, then continue the rest of setup."
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


if __name__ == "__main__":
    app_entry()
