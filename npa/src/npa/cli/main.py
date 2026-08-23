"""npa CLI entry point."""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import traceback
from enum import Enum
from typing import Any, Callable, Iterable, Optional

import click
import typer

from npa.cli._error_formatting import format_error_for_user
from npa.cli.agent import app as agent_app
from npa.cli.burst import app as burst_app
from npa.cli.workbench import app as workbench_app
from npa.cli.adapter import app as adapter_app
from npa.cli.cluster import app as cluster_app
from npa.cli.convert import app as convert_app
from npa.cli.demo import app as demo_app
from npa.cli.fleet import app as fleet_app
from npa.cli.network import app as network_app
from npa.cli.provision import app as provision_app
from npa.cli.rerun import app as rerun_app
from npa.cli.skypilot import app as skypilot_app
from npa.cli.cleanup import cleanup_cmd as _cleanup_cmd
from npa.cli.uninstall import uninstall_cmd as _uninstall_cmd
from npa.cli.storage import app as storage_app
from npa.cli.soperator import app as soperator_app
from npa.cli.viz import app as viz_app
from npa.cli.workflow_shim import workflow_shim_app
from npa.clients.serverless import ServerlessClientError
from npa.provisioning_journal import (
    ProvisioningOperation,
    current_operation,
    emit_recovery_summary,
    operation_context,
)
from npa.lifecycle_intent import OperationIntent, intent_boundary, json_stdout_contract

logger = logging.getLogger(__name__)


def _restore_recorded_destroy_phases(value: object):  # noqa: ANN201
    """Rehydrate the immutable first-attempt target set for a destroy retry."""

    from npa.project_destroy import DestroyPhase

    if not isinstance(value, dict):
        return None
    topology = value.get("topology")
    phase_values = topology.get("phases") if isinstance(topology, dict) else None
    if not isinstance(phase_values, list):
        raise RuntimeError("destroy operation has an invalid recorded phase topology")
    phases = []
    for item in phase_values:
        if not isinstance(item, dict) or not isinstance(item.get("commands"), list):
            raise RuntimeError("destroy operation has an invalid recorded phase")
        commands = item["commands"]
        if any(
            not isinstance(command, list)
            or not command
            or any(not isinstance(argument, str) for argument in command)
            for command in commands
        ):
            raise RuntimeError("destroy operation has invalid recorded command argv")
        requires = item.get("requires", [])
        metadata = item.get("metadata", {})
        if not isinstance(requires, list) or not all(
            isinstance(requirement, str) for requirement in requires
        ):
            raise RuntimeError("destroy operation has invalid phase dependencies")
        if not isinstance(metadata, dict):
            raise RuntimeError("destroy operation has invalid phase metadata")
        phases.append(
            DestroyPhase(
                name=str(item.get("phase") or ""),
                commands=tuple(tuple(command) for command in commands),
                detail=str(item.get("detail") or ""),
                requires=tuple(requires),
                metadata=dict(metadata),
            )
        )
    if not phases or any(not phase.name for phase in phases):
        raise RuntimeError(
            "destroy operation has an incomplete recorded phase topology"
        )
    return phases


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
app.add_typer(fleet_app, name="fleet", rich_help_panel="Platform utilities")
app.add_typer(network_app, name="network", rich_help_panel="Platform utilities")
app.add_typer(provision_app, name="provision-if-absent", rich_help_panel="Setup")
app.add_typer(rerun_app, name="rerun", rich_help_panel="Platform utilities")
app.add_typer(skypilot_app, name="skypilot", rich_help_panel="Platform utilities")
app.add_typer(storage_app, name="storage", rich_help_panel="Platform utilities")
app.command("cleanup", rich_help_panel="Platform utilities")(_cleanup_cmd)
app.command("uninstall", rich_help_panel="Setup")(_uninstall_cmd)
app.add_typer(soperator_app, name="soperator", rich_help_panel="Platform utilities")
app.add_typer(viz_app, name="viz", rich_help_panel="Platform utilities")
app.add_typer(workflow_shim_app, name="workflow", hidden=True)


@app.command("destroy", rich_help_panel="Platform utilities")
@intent_boundary(OperationIntent.DESTROY)
@json_stdout_contract
def destroy_project_cmd(
    project: str = typer.Option(
        "", "--project", help="Exact configured project alias."
    ),
    receipt: str = typer.Option(
        "",
        "--receipt",
        help="Opaque durable receipt for post-forget exact project deletion.",
    ),
    all_resources: bool = typer.Option(
        False,
        "--all",
        help="Required acknowledgement to plan the full project lifecycle.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Execute the rendered plan."),
    delete_project: bool = typer.Option(
        False,
        "--delete-project",
        help="Also delete an exact, empty project with durable NPA creation proof.",
    ),
    output_json: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Plan or execute project-scoped teardown through guarded NPA commands."""

    if not all_resources:
        raise typer.BadParameter("Full project teardown requires --all.")
    from npa.project_destroy import (
        build_project_destroy_plan,
        build_receipt_project_delete_plan,
        execute_project_destroy,
    )

    recovery_identity: dict[str, str] | None = None
    receipt_id = receipt.strip()
    if receipt_id:
        if not delete_project:
            raise typer.BadParameter(
                "Receipt recovery on `npa destroy` is limited to explicit --delete-project."
            )
        from npa.teardown_receipts import TeardownReceiptError, load_teardown_receipt

        try:
            saved = load_teardown_receipt(receipt_id)
        except TeardownReceiptError as exc:
            raise typer.BadParameter(str(exc)) from exc
        identity = saved.get("identity")
        identity = identity if isinstance(identity, dict) else {}
        saved_alias = str(
            saved.get("project_alias") or identity.get("project_alias") or ""
        ).strip()
        saved_project_id = str(
            saved.get("project_id") or identity.get("project_id") or ""
        ).strip()
        saved_tenant_id = str(identity.get("tenant_id") or "").strip()
        saved_region = str(identity.get("region") or "").strip()
        saved_profile = str(identity.get("profile") or "").strip()
        if (
            not saved_alias
            or not saved_project_id
            or not saved_tenant_id
            or not saved_region
        ):
            raise typer.BadParameter(
                "Receipt lacks exact project alias, project ID, tenant ID, or region."
            )
        if project.strip() and project.strip() != saved_alias:
            raise typer.BadParameter(
                "--project conflicts with the durable receipt project alias."
            )
        project = saved_alias
        recovery_identity = {
            "project_id": saved_project_id,
            "tenant_id": saved_tenant_id,
            "region": saved_region,
            "profile": saved_profile,
        }
        phases = build_receipt_project_delete_plan(
            project=project,
            project_id=saved_project_id,
            tenant_id=saved_tenant_id,
            receipt_id=receipt_id,
        )
    else:
        if not project.strip():
            raise typer.BadParameter(
                "Full project teardown requires --project or --receipt."
            )
        project = project.strip()
        phases = build_project_destroy_plan(project, delete_project=delete_project)
    if not yes:
        payload = {
            "status": "plan_only",
            "changed": False,
            "project": project,
            "phases": [phase.to_dict() for phase in phases],
        }
        if output_json:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            typer.echo(f"Project teardown plan for {project} (no changes):")
            for phase in phases:
                typer.echo(f"  {phase.name}: {phase.detail}")
                for command in phase.commands:
                    typer.echo("    " + shlex.join(command))
        return

    if recovery_identity is None:
        from npa.clients.config import resolve_environment

        environment = resolve_environment(project)
        if environment is None or not environment.project_id:
            raise typer.BadParameter(
                f"Project {project!r} has no immutable project identity; refusing teardown."
            )
        operation_project_id = str(environment.project_id)
        operation_tenant_id = str(environment.tenant_id)
        operation_region = str(environment.region)
    else:
        operation_project_id = recovery_identity["project_id"]
        operation_tenant_id = recovery_identity["tenant_id"]
        operation_region = recovery_identity["region"]
    operation = ProvisioningOperation.prepare(
        command="npa destroy",
        project_alias=project,
        project_id=operation_project_id,
        tenant_id=operation_tenant_id,
        region=operation_region,
        resource_type="project-teardown",
        # Receipt recovery resumes the same exact-project lifecycle operation.
        # A distinct operation would conflict with the still-authoritative
        # project lease after an interrupted post-alias-removal final audit.
        requested_name=project,
        ownership_source="project-destroy-cli",
        resume_command="",
        resume_argv=(
            [
                "npa",
                "destroy",
                "--receipt",
                receipt_id,
                "--all",
                "--delete-project",
                "--yes",
                "--json",
            ]
            if receipt_id
            else [
                "npa",
                "destroy",
                "--project",
                project,
                "--all",
                *(["--delete-project"] if delete_project else []),
                "--yes",
                "--json",
            ]
        ),
    )
    recorded_phases = _restore_recorded_destroy_phases(
        operation.read().get("preflight_plan")
    )
    if recorded_phases is not None:
        # A partial retry must replay the original exact target set. Rebuilding
        # from converged local state makes completed commands disappear and was
        # incorrectly rejected as a topology change; admitting newly appeared
        # targets would be a more dangerous widening of authorization.
        phases = recorded_phases
    operation.record_preflight_plan(
        {
            "project_alias": project,
            "project_id": operation_project_id,
            "tenant_id": operation_tenant_id,
            "region": operation_region,
            "receipt_id": receipt_id,
            "topology": {"phases": [phase.to_dict() for phase in phases]},
            "decision": "execute",
        }
    )
    try:
        with operation_context(operation):
            operation.transition("mutating")
            result = execute_project_destroy(
                project,
                phases,
                on_phase=lambda phase: operation.heartbeat(details={"phase": phase}),
                exact_identity=recovery_identity,
            )
            if result["status"] == "success":
                operation.transition("state-durable")
                operation.commit()
            else:
                operation.transition(
                    "recovery-required",
                    error="one or more independent teardown phases remain",
                    details={"error_type": "PartialProjectTeardown"},
                )
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        journal_phase = str(operation.read().get("phase") or "")
        if journal_phase not in {"recovery-required", "rollback-incomplete"}:
            operation.transition(
                "recovery-required",
                error=str(exc),
                details={"error_type": type(exc).__name__},
            )
        raise
    if output_json:
        typer.echo(json.dumps(result, indent=2, sort_keys=True))
    else:
        typer.echo(f"status: {result['status']}")
        for phase in result["phases"]:
            typer.echo(f"{phase['phase']}: {phase['status']}")
    if result["status"] != "success":
        raise typer.Exit(code=2)


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
  # (a "Read" token is enough). Public defaults work anonymously; for gated models,
  # also click "Agree and access repository" on each model page while signed in.
  # Step-by-step guide: docs/workbench/huggingface-token.md
  HF_TOKEN: hf_REPLACE_ME
  # Optional: Nebius Token Factory API key (OpenAI-compatible hosted inference).
  # Get one at https://tokenfactory.nebius.com/ -> API keys. The key is a long
  # opaque token (it starts with "v1."); it is NOT your Nebius IAM/CLI token.
  # Step-by-step guide: docs/workbench/token-factory-key.md
  NEBIUS_TOKEN_FACTORY_KEY: <paste-your-token-factory-api-key>  # e.g. v1.XXXXXXXX...
ngc:
  # NVIDIA NGC API key (only for entitlement-controlled NGC artifact pulls).
  # Get one at https://org.ngc.nvidia.com/setup/api-key -> "Generate API Key"
  # (sign in / create a free NGC account first). The key starts with "nvapi-".
  # Step-by-step guide: docs/workbench/ngc-api-key.md
  api_key: nvapi-REPLACE_ME
  # org: optional-ngc-org
  # team: optional-ngc-team
storage:
  aws_access_key_id: <your-s3-access-key-id>
  aws_secret_access_key: <your-s3-secret-access-key>
  # Region-specific: use your project's region, which `npa configure` fills in
  # (the "Current configuration" block below shows what is actually saved).
  endpoint_url: https://storage.<your-region>.nebius.cloud
  bucket: s3://<your-bucket>/
ssh:
  host: <your-byovm-host>
  user: ubuntu
  key_path: ~/.ssh/id_ed25519

Then secure it:

chmod 600 ~/.npa/credentials.yaml

`npa configure` also writes ~/.npa/config.yaml with your Nebius project id,
tenant id, and region so commands no longer need those values exported in the
shell or read from the Nebius CLI. Workbench images use the anonymous GHCR
mirror by default; set NPA_REGISTRY or pass an explicit image when you need a
private or locally modified image. Deploy commands extend the same file with
workbench endpoints and Terraform state.

Treat ~/.npa/config.yaml as sensitive too: deploys persist the Terraform remote
state S3 access key and secret under projects.<alias>.terraform_state. npa keeps
it at mode 600 (and ~/.npa at 700); do not copy it between machines or into a
repo. If you ever see the permissions warning:

chmod 700 ~/.npa && chmod 600 ~/.npa/config.yaml
"""


def _version_callback(value: bool) -> None:
    if not value:
        return
    # Reuse the fast-path resolver so both --version paths emit an identical
    # string (and both fall back to 0.0.0.dev0 in an uninstalled source tree).
    from npa.cli.entry import _resolve_version

    typer.echo(f"npa {_resolve_version()}")
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


def _list_nebius_profiles(
    *, runner: Callable[..., object] = subprocess.run
) -> list[str]:
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


def _create_nebius_profile(
    *, project_id: str = "", runner: Callable[..., object] = subprocess.run
) -> bool:
    """Run the interactive ``nebius profile create`` flow.

    Supplying a project ID makes the provider CLI project-scoped from the start.
    Without it, the CLI tries to enumerate projects through the tenant resource,
    which correctly fails for operators whose custom group has access only to a
    specific project.
    """

    command = ["nebius", "profile", "create"]
    project = str(project_id or "").strip()
    if project:
        command.extend(["--parent-id", project])

    try:
        result = runner(command, check=False)
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
    typer.echo(
        "A Nebius CLI profile should point at one project. Supplying the exact "
        "project ID avoids tenant-wide project discovery, which project-scoped "
        "IAM groups cannot perform."
    )
    profile_project_id = str(
        typer.prompt(
            "Nebius project ID (project-...; leave blank to use CLI discovery)",
            default="",
            show_default=False,
        )
    ).strip()
    if not profile_project_id:
        typer.echo(
            "No project ID supplied. If the Nebius CLI reports PermissionDenied "
            "while fetching projects, enter the exact project ID at `Set Parent ID`."
        )
    if (
        _create_nebius_profile(project_id=profile_project_id)
        and _nebius_profile_ready()
    ):
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


def _normalize_pasted_secret(value: str, *, strip_auth_wrapper: bool = True) -> str:
    """Clean a pasted credential: drop wrapping quotes and auth prefixes.

    Users routinely paste a token copied from a curl example or a password
    manager with surrounding quotes or an ``Authorization: Bearer`` prefix. Those
    silently break auth (the stored value is not the bare token), so strip them.
    """
    text = (value or "").strip()
    # Unwrap matching surrounding quotes (may wrap a "Bearer ..." string).
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1].strip()
    if strip_auth_wrapper:
        # Drop a pasted "Authorization:" header label.
        if text.lower().startswith("authorization:"):
            text = text.split(":", 1)[1].strip()
        # Drop a leading auth scheme (Bearer/Token), case-insensitively.
        for scheme in ("bearer ", "token "):
            if text.lower().startswith(scheme):
                text = text[len(scheme) :].strip()
                break
    # Unwrap again in case the scheme was inside the quotes.
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1].strip()
    return text


def _gb_to_bytes(value: str) -> int:
    """Parse GiB into bytes; invalid uses the recommended cap, <=0 disables it."""
    from decimal import Decimal, InvalidOperation, ROUND_DOWN

    try:
        gb = Decimal(str(value).strip())
        if not gb.is_finite():
            raise InvalidOperation
    except (InvalidOperation, TypeError, ValueError):
        gb = Decimal(str(RECOMMENDED_BUCKET_SIZE_GB))
    if gb <= 0:
        return 0
    return int((gb * Decimal(1024**3)).to_integral_value(rounding=ROUND_DOWN))


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


def _collision_bucket_name(bucket_name: str, *, tenant_id: str, project_id: str) -> str:
    from npa.lifecycle_intent import forbid_destructive_provisioning

    forbid_destructive_provisioning("collision_bucket_name")
    suffix = hashlib.sha256(
        f"{tenant_id}\0{project_id}\0{bucket_name}\0collision".encode("utf-8")
    ).hexdigest()[:8]
    return f"{bucket_name[:54].rstrip('-')}-{suffix}"


def _storage_relationship_verified(credentials: Any, project_id: str) -> bool:
    """Return whether saved storage has durable provenance for this project."""

    return bool(
        str(project_id or "").strip()
        and str(getattr(credentials, "s3_project_id", "") or "").strip()
        == str(project_id or "").strip()
        and str(getattr(credentials, "s3_ownership", "") or "").strip()
    )


def _credentials_for_exact_project(
    project_id: str,
    *,
    allow_provable_legacy: bool = False,
) -> Any:
    """Load shared tokens plus storage from exactly one project record.

    Top-level storage in ``credentials.yaml`` is only a compatibility view. It
    is never retained for a selected project unless a legacy ownership record
    proves that exact project and the caller explicitly permits migration.
    """

    from npa.clients.credentials import load_credentials
    from npa.clients.project_credential_store import read_project_credential_record

    credentials = load_credentials(environ={})
    exact = str(project_id or "").strip()
    record = read_project_credential_record(exact) if exact else None
    storage = record.get("storage") if isinstance(record, dict) else None
    if isinstance(storage, dict) and record.get("storage_selected") is not False:
        credentials.s3_access_key_id = str(
            storage.get("aws_access_key_id")
            or storage.get("access_key")
            or storage.get("nebius_api_key")
            or ""
        )
        credentials.s3_secret_access_key = str(
            storage.get("aws_secret_access_key")
            or storage.get("secret_key")
            or storage.get("nebius_secret_key")
            or ""
        )
        credentials.s3_endpoint = str(
            storage.get("endpoint_url")
            or storage.get("endpoint")
            or storage.get("s3_endpoint")
            or ""
        )
        credentials.s3_bucket = str(
            storage.get("checkpoint_bucket")
            or storage.get("bucket")
            or storage.get("s3_bucket")
            or ""
        )
        credentials.s3_project_id = exact
        credentials.s3_ownership = "exact-project-record"
        return credentials
    if allow_provable_legacy and _storage_relationship_verified(credentials, exact):
        return credentials
    credentials.s3_access_key_id = ""
    credentials.s3_secret_access_key = ""
    credentials.s3_endpoint = ""
    credentials.s3_bucket = ""
    credentials.s3_project_id = exact
    credentials.s3_ownership = ""
    return credentials


def _provision_object_storage(
    nebius_client,
    ask: Callable[..., str],
    *,
    project_id: str,
    tenant_id: str,
    region: str,
    existing_bucket: str = "",
    interactive: bool = True,
) -> dict[str, str] | None:
    """Auto-create the S3 bucket + access key for the project."""
    if not (project_id and tenant_id):
        return None

    if interactive:
        typer.echo(
            "\nObject storage: enter a bucket name to reuse it (or create it if it "
            "does not exist yet), or press Enter to use npa's default bucket for this "
            "project. The default name is derived from your tenant + project, so it "
            "is stable across runs and reused rather than duplicated."
        )
        bucket_name = ask("Object-storage bucket name", default=existing_bucket).strip()
    else:
        bucket_name = str(existing_bucket or "").strip()
        selected = bucket_name or nebius_client.bucket_name_for(tenant_id, project_id)
        typer.echo(
            "Object storage (non-interactive): selected "
            f"'{selected}'; it will be reused if present or provisioned if absent."
        )
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
            "npa will not create or adopt it until project ownership and access "
            "can be verified."
        )
        return None

    bucket_max_size_bytes = 0
    bucket_storage_class = DEFAULT_BUCKET_STORAGE_CLASS
    if exists is True:
        typer.echo(f"Reusing existing object-storage bucket '{bucket_name}'.")
    elif exists is False:
        typer.echo(
            f"No existing bucket named '{bucket_name}' found; npa will create it."
        )
        bucket_storage_class, bucket_max_size_bytes = _prompt_new_bucket_settings(
            ask,
            bucket_name=bucket_name,
        )
    # exists is None: existence unknown, so skip the create-only prompts and let
    # provisioning get-or-create with defaults rather than risk creating a
    # duplicate of a bucket that may already exist.

    try:
        typer.echo("Provisioning Nebius object storage (bucket + access key)...")
        from npa.clients.storage_setup import provision_storage

        creds, probe = provision_storage(
            project_id=project_id,
            tenant_id=tenant_id,
            region=region,
            bucket_name=bucket_name,
            bucket_max_size_bytes=bucket_max_size_bytes,
            bucket_storage_class=bucket_storage_class,
            on_status=lambda msg: typer.echo(f"  - {msg}"),
        )
    except nebius_client.NebiusError as exc:
        if "already taken" in str(exc).lower():
            alternate = _collision_bucket_name(
                bucket_name,
                tenant_id=tenant_id,
                project_id=project_id,
            )
            typer.echo(
                f"  Bucket name collision: preserving the unrelated bucket and "
                f"proposing '{alternate}' instead."
            )
            return _provision_object_storage(
                nebius_client,
                ask,
                project_id=project_id,
                tenant_id=tenant_id,
                region=region,
                existing_bucket=alternate,
                interactive=interactive,
            )
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
                "  NPA first tries a bucket-scoped storage.object-editor binding "
                "covering GetObject, HeadObject, PutObject, DeleteObject, and "
                "ListObjectsV2. Existing editors memberships remain compatible. "
                "Only if this provider reports that narrow role unsupported may "
                "you explicitly opt into the broader compatibility fallback with "
                "NPA_ALLOW_EDITORS_STORAGE_FALLBACK=1. Newly changed IAM can take "
                "about a minute to propagate; then re-run `npa configure`."
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
    typer.echo(f"  Provisioned bucket {bucket} and an S3 access key; {probe.summary}")
    payload: dict[str, str] = {
        "aws_access_key_id": access_key,
        "aws_secret_access_key": secret_key,
        "endpoint_url": creds.get("s3_endpoint", "") or _endpoint_for_region(region),
        "bucket": bucket,
        "_validated": "true",
    }
    sa_id = creds.get("service_account_id", "").strip()
    if sa_id:
        payload["service_account_id"] = sa_id
    for key in (
        "service_account_name",
        "service_account_project_id",
        "service_account_managed_by",
    ):
        value = str(creds.get(key, "") or "").strip()
        if value:
            payload[key] = value
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
        "\nNVIDIA NGC API key (for entitlement-controlled NGC artifact pulls): create one at "
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


def _existing_alias_for_project(
    existing_projects: dict[str, dict[str, Any]],
    default_alias: str,
    project_id: str,
) -> str:
    """Return the alias already configured for *project_id*, preferring the default."""
    if not project_id:
        return ""
    matches = [
        alias
        for alias, stanza in (existing_projects or {}).items()
        if str((stanza or {}).get("project_id", "") or "") == project_id
    ]
    if not matches:
        return ""
    if default_alias in matches:
        return default_alias
    return sorted(matches)[0]


def _unclaimed_alias(
    candidate: str,
    project_id: str,
    existing_projects: dict[str, dict[str, Any]],
    used: set[str],
) -> str:
    """Return *candidate* (or a numbered variant) that no other project claims.

    ``write_config`` deep-merges, so writing a project into an alias that already
    describes a *different* project produces a mixed stanza — the new project id
    next to the old project's terraform_state, workbench endpoints and registry.
    """
    alias = candidate
    suffix = 2
    while alias in used or _alias_holds_other_project(
        existing_projects, alias, project_id
    ):
        alias = f"{candidate}-{suffix}"
        suffix += 1
    return alias


def _alias_holds_other_project(
    existing_projects: dict[str, dict[str, Any]],
    alias: str,
    project_id: str,
) -> bool:
    existing_id = str(
        ((existing_projects or {}).get(alias) or {}).get("project_id", "") or ""
    )
    return bool(existing_id) and existing_id != project_id


def _validate_alias_project_identity(
    alias: str, stanza: dict[str, Any], project_id: str
) -> None:
    """Refuse to repoint an existing alias at a different exact project."""

    if not stanza:
        return
    previous = str((stanza or {}).get("project_id", "") or "")
    if previous == project_id:
        return
    identity = previous or "an unreadable/legacy project"
    raise typer.BadParameter(
        f"Project alias '{alias}' already belongs to {identity}; refusing to repoint "
        f"it at {project_id} because project-scoped state must never be merged across "
        "projects. Choose a new --project-alias."
    )


def _select_discovered_projects(
    projects: list[dict[str, str]],
    ask: Callable[..., str],
    *,
    current_project_id: str = "",
    existing_projects: dict[str, dict[str, Any]] | None = None,
    existing_default_alias: str = "",
) -> tuple[list[tuple[str, dict[str, str]]], str]:
    """Present discovered projects and return ``([(alias, stanza)...], default_alias)``.

    Auto-derives tenant/project/region from each pick. npa is multi-project: the
    user may select several.
    Aliases reuse the stanza a project already has in ``~/.npa/config.yaml`` so a
    re-run updates it in place instead of stranding the workbench endpoints and
    Terraform state saved under the old alias.
    """
    # Large Nebius accounts can expose hundreds/thousands of projects. Dumping
    # and configuring them all by default is unusable, so offer a name/id filter
    # and cap the printed list.
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
    # discovered project).
    default_pick = "1"
    for i, proj in enumerate(shown, start=1):
        if proj["id"] == current_project_id:
            default_pick = str(i)
            break
    raw = ask(
        "\nSelect project(s) to configure (comma-separated numbers, or 'all')",
        default=default_pick,
    )
    chosen = _parse_selection(raw, len(shown)) or _parse_selection(
        default_pick, len(shown)
    )

    selected: list[tuple[str, dict[str, str]]] = []
    used_aliases: set[str] = set()
    configured = existing_projects or {}
    for idx in chosen:
        proj = shown[idx]
        alias = _existing_alias_for_project(
            configured, existing_default_alias, proj["id"]
        ) or _unclaimed_alias(
            _slugify_alias(proj.get("name", ""), proj["id"]),
            proj["id"],
            configured,
            used_aliases,
        )
        used_aliases.add(alias)
        stanza = {
            "project_id": proj["id"],
            "tenant_id": proj["tenant_id"],
            "region": proj.get("region", "") or DEFAULT_REGION,
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


def _resolve_discovery_tenant(
    nebius_client, ask: Callable[..., str]
) -> tuple[str, str]:
    """Return ``(tenant_id, skip_reason)`` for project discovery.

    ``nebius config get tenant-id`` is empty for plenty of real profiles —
    federation profiles and profiles created against a single project commonly
    set only ``parent-id``. Discovery used to bail out silently in that case and
    drop the operator into hand-typing tenant/project ids even though the
    account was perfectly reachable. Recover the tenant instead:

    1. the profile's own ``tenant-id``;
    2. the parent tenant of the profile's ``parent-id`` project;
    3. the tenants the profile can list (auto-select a single one, otherwise
       prompt).

    ``skip_reason`` explains the failure so the caller can say why discovery was
    skipped rather than staying quiet.
    """

    tenant = str(nebius_client.current_tenant_id() or "").strip()
    if tenant:
        return tenant, ""

    project_id = str(nebius_client.current_project_id() or "").strip()
    if project_id:
        derived = str(nebius_client.get_project_tenant_id(project_id) or "").strip()
        if derived:
            typer.echo(
                f"Nebius profile has no tenant-id; using {derived} "
                f"(the parent tenant of project {project_id})."
            )
            return derived, ""

    tenants = nebius_client.list_tenants()
    if len(tenants) == 1:
        only = str(tenants[0].get("id", "") or "").strip()
        if only:
            typer.echo(
                f"Nebius profile has no tenant-id; using the only tenant you can "
                f"see: {only}."
            )
            return only, ""
    if len(tenants) > 1:
        typer.echo("\nNebius profile has no tenant-id. Tenants you can see:\n")
        for index, tenant_entry in enumerate(tenants, start=1):
            label = str(tenant_entry.get("name", "") or "").strip()
            suffix = f"  ({label})" if label else ""
            typer.echo(f"   {index:>2}. {tenant_entry.get('id', '')}{suffix}")
        raw = ask("\nDiscover projects in which tenant? (number)", default="1")
        choice = raw.strip()
        index = int(choice) - 1 if choice.isdigit() else 0
        if not (0 <= index < len(tenants)):
            index = 0
        picked = str(tenants[index].get("id", "") or "").strip()
        if picked:
            return picked, ""

    if project_id:
        return "", (
            f"the Nebius profile has no tenant-id and the parent tenant of "
            f"{project_id} could not be read"
        )
    return "", "the Nebius profile has no tenant-id and no tenants are listable"


def _offer_profile_binding(
    nebius_client,
    ask: Callable[..., str],
    *,
    project_id: str,
    tenant_id: str,
) -> bool:
    """Offer to point the active Nebius CLI profile at the chosen project.

    Returns True when the profile was updated. Best-effort and confirm-gated:
    declining just prints what stays out of sync.
    """

    try:
        profile_project = str(nebius_client.current_project_id() or "").strip()
        profile_tenant = str(nebius_client.current_tenant_id() or "").strip()
    except Exception:  # noqa: BLE001 - never fail configure over a profile read
        return False
    if profile_project == project_id and (not tenant_id or profile_tenant == tenant_id):
        return False

    if profile_project or profile_tenant:
        detail = (
            f"currently parent-id={profile_project or '<unset>'}, "
            f"tenant-id={profile_tenant or '<unset>'}"
        )
    else:
        detail = "currently unset"
    typer.echo(
        f"\nYour Nebius CLI profile does not point at this project ({detail}). "
        "npa runs the Nebius CLI with that profile, so leaving them out of sync "
        "disables project discovery on the next run."
    )
    answer = ask(f"Point the active Nebius profile at {project_id}? [Y/n]", default="Y")
    if answer.lower() not in ("", "y", "yes"):
        typer.echo(
            "  Leaving the Nebius profile unchanged. Set it later with "
            f"`nebius config set parent-id {project_id}`"
            + (f" && `nebius config set tenant-id {tenant_id}`." if tenant_id else ".")
        )
        return False
    if nebius_client.set_profile_project(project_id, tenant_id):
        typer.echo(f"  Nebius profile now points at {project_id}.")
        return True
    typer.echo(
        "  Could not update the Nebius profile. Set it by hand with "
        f"`nebius config set parent-id {project_id}`."
    )
    return False


def _run_interactive_configure(
    *,
    provision: bool = True,
    allow_visible_secret_input: bool = False,
) -> None:
    """Prompt for credentials/config and write the NPA dotfiles."""

    from npa.clients.config import (
        CONFIG_PATH,
        config_permissions_warning,
        default_project_name,
        list_projects,
        write_config,
    )
    from npa.clients.credentials import load_credentials, write_credentials_file
    from npa.clients import nebius as nebius_client

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
            "credentials manually."
        )
        typer.echo("Nothing was written under ~/.npa.")
        raise typer.Exit(code=1)

    # Hidden input needs a controlling terminal.  The mode resolver rejects
    # forced interactive use on a pipe unless the operator explicitly accepts
    # visible secret entry.
    stdin_is_tty = sys.stdin.isatty()
    echo_notice_shown = False

    def ask(label: str, *, default: str = "", secret: bool = False) -> str:
        nonlocal echo_notice_shown
        hide_input = secret and stdin_is_tty
        if secret and not hide_input and not allow_visible_secret_input:
            raise typer.BadParameter(
                "Refusing visible secret input on non-TTY stdin. Use environment "
                "credential import for automation, or explicitly pass "
                "--allow-visible-secret-input."
            )
        if secret and not hide_input and not echo_notice_shown:
            echo_notice_shown = True
            typer.echo(
                "\nNote: stdin is not a terminal, so secret values will be "
                "visible as you enter them. For automation prefer environment "
                "variables or `npa configure --no-interactive --save-env-credentials`."
            )
        return str(
            typer.prompt(
                label,
                default=default,
                hide_input=hide_input,
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
    # Sample the file mode before anything is written: write_config() chmods
    # config.yaml to 0600, so a check after the write can never see the loose
    # mode this warning exists for.
    permissions_warning = config_permissions_warning()

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
    current_tenant = ""
    current_profile_project = ""
    tenant_skip_reason = "no authenticated Nebius CLI profile"
    if profile_ready:
        current_profile_project = str(nebius_client.current_project_id() or "").strip()
        current_tenant, tenant_skip_reason = _resolve_discovery_tenant(
            nebius_client, ask
        )
    discovered_projects = (
        nebius_client.list_projects_in_tenant(current_tenant)
        if (profile_ready and current_tenant)
        else []
    )
    if profile_ready and not current_tenant:
        # Silence here used to look like "you have no projects": discovery was
        # skipped and the operator was dropped into hand-typing ids.
        typer.echo(
            f"Skipping project discovery ({tenant_skip_reason}). Enter the ids "
            "manually below, or set them on the profile with `nebius config set "
            "tenant-id <id>` / `nebius config set parent-id <project-id>` and "
            "re-run `npa configure`.\n"
        )
    elif profile_ready and not discovered_projects and current_profile_project:
        typer.echo(
            "Tenant-wide project discovery returned no projects. This is expected "
            "for project-scoped IAM access; continuing with the active profile's "
            "parent-id as the project default.\n"
        )
    if discovered_projects:
        discovered_selection, discovered_default_alias = _select_discovered_projects(
            discovered_projects,
            ask,
            current_project_id=current_profile_project,
            existing_projects=existing_projects,
            existing_default_alias=existing_default_alias,
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
    else:
        # Tenant is the parent of the project, so ask for it first.
        tenant_id = ask(
            "Nebius tenant id",
            default=str(existing_stanza.get("tenant_id", "")) or current_tenant,
        )
        project_id = ask(
            "Nebius project id",
            default=str(existing_stanza.get("project_id", ""))
            or current_profile_project,
        )
        region_default = str(existing_stanza.get("region", "")) or DEFAULT_REGION
        region = ask("Region", default=region_default)

    if project_id:
        existing_credentials = _credentials_for_exact_project(
            project_id, allow_provable_legacy=True
        )

    operation = current_operation()
    if operation is not None:
        operation.update_identity(
            project_alias=discovered_default_alias,
            project_id=project_id,
            tenant_id=tenant_id,
            region=region,
        )

    # Keep the Nebius CLI profile and the npa project in sync. npa shells out to
    # the CLI with the operator's active profile, so a profile whose parent-id /
    # tenant-id are empty (or point at a different project) silently disables
    # discovery on the next run and sends later commands somewhere else.
    if profile_ready and project_id:
        _offer_profile_binding(
            nebius_client, ask, project_id=project_id, tenant_id=tenant_id
        )

    storage: dict[str, str] | None = None

    # Object storage is opt-in: `npa configure` sets up the Nebius connection and
    # optional model/inference tokens. Storage (an S3 bucket + access key) is only
    # needed by workbench data workflows, so when projects were discovered we ask
    # before provisioning instead of doing it by default.
    if discovered_selection and provision:
        want_storage = ask(
            "Set up object storage (S3 bucket + access key) now? "
            "The agent VM, workflow submits (`stage-src`) and the Physical AI "
            "Data Factory all need it; you can add it later. [Y/n]",
            default="Y",
        )
        if want_storage.lower() not in ("", "y", "yes"):
            provision = False
            storage = {}
            typer.echo(
                "  Skipping object storage. Note: `npa agent setup` and "
                "`npa workbench workflow submit` (Physical AI Data Factory) need "
                "an S3 bucket + access key — re-run `npa configure` to add it before "
                "using them."
            )

    existing_has_storage = bool(
        existing_credentials.s3_access_key_id
        and existing_credentials.s3_secret_access_key
        and existing_credentials.s3_bucket
    )
    existing_relationship_verified = _storage_relationship_verified(
        existing_credentials, project_id
    )
    declined_existing_bucket = False
    provisioning_failed = False
    if provision and project_id and tenant_id:
        if existing_has_storage and not existing_relationship_verified:
            typer.echo(
                "  Saved object storage was not offered as a default because no "
                "durable record proves it belongs to the selected project. A fresh "
                "project-scoped name will be proposed."
            )
        if existing_has_storage and existing_relationship_verified:
            candidate = {
                "aws_access_key_id": existing_credentials.s3_access_key_id,
                "aws_secret_access_key": existing_credentials.s3_secret_access_key,
                "endpoint_url": existing_credentials.s3_endpoint
                or _endpoint_for_region(region),
                "bucket": existing_credentials.s3_bucket,
            }
            from npa.clients.storage_validation import probe_storage_write

            probe = probe_storage_write(
                bucket=candidate["bucket"],
                endpoint_url=candidate["endpoint_url"],
                access_key_id=candidate["aws_access_key_id"],
                secret_access_key=candidate["aws_secret_access_key"],
                region=region,
            )
            if probe.ok:
                keep = ask(
                    "Keep access-verified object storage "
                    f"({existing_credentials.s3_bucket}, project {project_id}, "
                    f"provenance: {existing_credentials.s3_ownership})? [Y/n]",
                    default="Y",
                )
                if keep.lower() in ("", "y", "yes"):
                    candidate["_validated"] = "true"
                    storage = candidate
                    typer.echo(
                        "  Adopting the verified pre-existing bucket; rollback will "
                        "never delete it. " + probe.summary
                    )
                else:
                    declined_existing_bucket = True
                    typer.echo(
                        "  Existing bucket declined; enter a new/editable name below."
                    )
            else:
                typer.echo(
                    "  Project-matched object storage is not usable: "
                    f"{probe.summary} NPA will reconcile it now."
                )
        if storage is None:
            storage = _provision_object_storage(
                nebius_client,
                ask,
                project_id=project_id,
                tenant_id=tenant_id,
                region=region,
                existing_bucket=(
                    ""
                    if declined_existing_bucket or not existing_relationship_verified
                    else _bucket_name_from_uri(existing_credentials.s3_bucket)
                ),
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
            typer.echo("Enter existing S3 credentials (or press Enter to leave blank).")
    if storage is None:
        storage = {
            "aws_access_key_id": _normalize_pasted_secret(
                ask(
                    "S3 access key id (AWS_ACCESS_KEY_ID)",
                    default=existing_credentials.s3_access_key_id,
                    secret=True,
                ),
                strip_auth_wrapper=False,
            ),
            "aws_secret_access_key": _normalize_pasted_secret(
                ask(
                    "S3 secret access key (AWS_SECRET_ACCESS_KEY)",
                    default=existing_credentials.s3_secret_access_key,
                    secret=True,
                ),
                strip_auth_wrapper=False,
            ),
            "endpoint_url": ask(
                "S3 endpoint URL",
                default=existing_credentials.s3_endpoint
                or _endpoint_for_region(region),
            ),
            "bucket": ask(
                "S3 bucket URI (e.g. s3://<your-bucket>/)",
                default=existing_credentials.s3_bucket,
            ),
        }

    # Manual credentials are never committed merely because all four strings
    # were entered. A typed write/read probe is the boundary between partial input
    # and usable storage; optional cleanup is reported independently.
    if storage and storage.get("_validated") != "true":
        from npa.clients.storage_validation import probe_storage_write

        probe = probe_storage_write(
            bucket=storage.get("bucket", ""),
            endpoint_url=storage.get("endpoint_url", ""),
            access_key_id=storage.get("aws_access_key_id", ""),
            secret_access_key=storage.get("aws_secret_access_key", ""),
            region=region,
        )
        if probe.ok:
            storage["_validated"] = "true"
            typer.echo(f"  {probe.summary}")
        else:
            typer.echo(f"  Object storage remains incomplete: {probe.summary}")
            storage = {}
            provisioning_failed = True

    hf_token, token_factory_api_key, ngc_api_key = _prompt_setup_tokens(
        ask, existing_credentials
    )

    service_account_keys = {
        key: str(storage.get(key, "") or "").strip()
        for key in (
            "service_account_id",
            "service_account_name",
            "service_account_project_id",
            "service_account_managed_by",
        )
        if str(storage.get(key, "") or "").strip()
    }
    credentials_payload: dict[str, object] = {
        "tokens": {
            "HF_TOKEN": hf_token,
            "NEBIUS_TOKEN_FACTORY_KEY": token_factory_api_key,
        },
        "ngc": {"api_key": ngc_api_key},
        "storage": {
            key: value
            for key, value in storage.items()
            if not key.startswith("service_account_")
            and not key.startswith("_")
            and value
        },
    }
    if service_account_keys:
        # Keep the generic ID for existing credential resolution, but store the
        # ownership proof under its own lifecycle key. Agent bootstrap also uses
        # nebius.service_account_id and may replace it with the npa-agent ID; that
        # must never rewrite which identity NPA proved it created for storage.
        account_id = service_account_keys.get("service_account_id", "")
        if account_id:
            credentials_payload["nebius"] = {"service_account_id": account_id}
        if all(
            service_account_keys.get(key)
            for key in (
                "service_account_id",
                "service_account_name",
                "service_account_project_id",
                "service_account_managed_by",
            )
        ):
            credentials_payload["storage_iam"] = service_account_keys

    # Service tokens remain host-scoped. Storage/IAM identity is authoritative
    # only in the exact-project v2 store; the top-level storage keys are a
    # derived compatibility view of the selected project.
    shared_sections = {"tokens", "ngc"}
    if not project_id:
        shared_sections.update({"storage", "storage_iam", "nebius"})
    shared_payload = {
        key: value
        for key, value in credentials_payload.items()
        if key in shared_sections
    }
    credentials_path = write_credentials_file(shared_payload)

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
        write_config({"projects": projects_payload, "default_project": alias})
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
            )
            if value
        }
        # Local name for later `-p <alias>` flags. Derived automatically (no
        # prompt): reuse the alias this project already has so a re-run updates
        # the same stanza, else the Nebius project's own name (matching what the
        # discovery path derives), else the region. A region-shaped alias
        # (`us-central1`) reads like a region field sitting next to the real
        # `region:` key, so it is only the last resort. An alias that already
        # describes a different project is never reused — config is deep-merged,
        # so that would splice the new project id into the old project's
        # terraform_state and workbench endpoints.
        # Multi-project users rename via ~/.npa/config.yaml.
        alias = _existing_alias_for_project(
            existing_projects, existing_default_alias, project_id
        )
        if not alias:
            project_name = ""
            try:
                project_name = nebius_client.get_project_name(project_id)
            except Exception:  # noqa: BLE001 - alias derivation is best-effort
                project_name = ""
            alias = _unclaimed_alias(
                _slugify_alias(project_name, project_id)
                if project_name
                else (region or "default"),
                project_id,
                existing_projects,
                set(),
            )
        write_config({"projects": {alias: project_stanza}, "default_project": alias})
        wrote_config = True

    if project_id:
        from npa.clients.project_credential_store import (
            select_project_credentials,
            write_project_credentials,
        )

        if storage:
            write_project_credentials(
                project_id,
                {
                    key: value
                    for key, value in credentials_payload.items()
                    if key in {"storage", "storage_iam", "nebius"}
                },
                alias=alias,
            )
        else:
            select_project_credentials(project_id, alias=alias, select_storage=False)

    if permissions_warning:
        note = (
            "config.yaml was readable by other users and holds Terraform backend "
            "S3 keys under projects.<alias>.terraform_state; npa has tightened it "
            "to 0600. Rotate those keys if the file was exposed."
            if wrote_config
            else permissions_warning
        )
        typer.echo(f"\nWarning: {note}", err=True)

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
    storage_complete = storage.get("_validated") == "true"
    if storage_complete:
        typer.echo("Setup complete. Run `npa configure --show` to see the file layout.")
        return
    if alias:
        next_command = f"npa provision-if-absent --project {alias} --skip-k8s"
    else:
        next_command = "npa configure"
    typer.echo(
        "Setup incomplete: writable object storage is not configured. "
        f"Any NPA-created partial-resource provenance is saved in {credentials_path}."
    )
    typer.echo(f"Resume safely with: `{next_command}`")
    # The configuration that was entered is valid and deliberately retained for
    # read-only/non-S3 modes. Do not label it complete; the downstream
    # provision/deploy commands enforce writable storage and exit non-zero until
    # the resumable prerequisite succeeds.


def _probe_hf_assets_parallel(
    validator: Callable[..., Any],
    token: str,
    assets: Iterable[Any],
    *,
    per_probe_timeout: float = 2.0,
    total_budget: float = 5.0,
) -> dict[tuple[str, str, str, str], Any]:
    """Probe HF access for *assets* concurrently within a wall-clock budget.

    Cache keys include the repository type and representative artifact so a
    stale catalog entry cannot borrow another probe's success. Assets that do
    not finish inside ``total_budget`` (or
    whose probe raises) are omitted, so the caller treats them as unverified
    rather than stalling the primary onboarding command. Exception messages are
    deliberately not logged because an injected client may echo its credential.
    """

    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FuturesTimeout
    from concurrent.futures import as_completed

    asset_list = list(assets)
    results: dict[tuple[str, str, str, str], Any] = {}
    if not asset_list:
        return results
    pool = ThreadPoolExecutor(max_workers=min(8, len(asset_list)))
    try:
        futures = {
            pool.submit(
                validator,
                token,
                asset.repo,
                asset.repo_type,
                revision=asset.revision,
                filename=asset.filename,
                timeout=per_probe_timeout,
            ): (asset.repo, asset.repo_type, asset.revision, asset.filename)
            for asset in asset_list
        }
        try:
            for fut in as_completed(futures, timeout=total_budget):
                cache_key = futures[fut]
                try:
                    results[cache_key] = fut.result()
                except Exception as exc:  # noqa: BLE001 - failed probe -> unverified
                    logger.debug(
                        "HF access probe failed for %s (%s)",
                        cache_key[0],
                        type(exc).__name__,
                    )
        except FuturesTimeout:
            # Budget exceeded: keep whatever finished; the rest stay unverified.
            logger.debug("HF access probe budget of %.1fs exceeded", total_budget)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return results


def _model_access_note(hf_token: str, ngc_key: str) -> str:
    """Return a one-line ``[NOTE]`` on which gated workbench models the tokens can access.

    Runs the same artifact-aware checks as ``npa workbench health access``:
    each license-gated Hugging Face model or dataset probes one representative
    revision/file, and NGC performs a real token-exchange/tag-listing entitlement
    probe.
    Configure keeps these advisory: HF probes run in parallel under a wall-clock
    budget, NGC uses the same short per-probe timeout, and failures never break
    setup. The health command remains the authoritative enforcement gate.
    """

    try:
        from npa.clients import huggingface
        from npa.clients.huggingface import HFAccessResult
        from npa.workbench.model_access import (
            access_note,
            check_workbench_access,
            gated_hf_assets,
        )
        from npa.workbench.nurec.nurec import check_ngc_image_access

        cache: dict[tuple[str, str, str, str], Any] = {}
        if hf_token:
            cache = _probe_hf_assets_parallel(
                huggingface.validate_hf_access, hf_token, gated_hf_assets()
            )

        def _validator(
            token: str,
            repo: str,
            repo_type: str = "model",
            *,
            revision: str = "",
            filename: str = "",
        ):
            return cache.get((repo, repo_type, revision, filename)) or HFAccessResult(
                repo=repo, ok=False, error="not verified (timed out)"
            )

        def _ngc_validator(api_key: str) -> str:
            return check_ngc_image_access(api_key, timeout=2.0)

        results = check_workbench_access(
            hf_token=hf_token,
            ngc_key=ngc_key,
            hf_validator=_validator if hf_token else None,
            ngc_validator=_ngc_validator,
            gated_only=True,
        )
        return access_note(results)
    except Exception:  # noqa: BLE001 - a preflight note must never break configure
        return "[NOTE] Skipped model-access check (verify later with `npa workbench health access`)."


def _saved_model_access_note() -> str:
    """Return the advisory using only persisted, redacted credential state."""

    try:
        from npa.clients.credentials import load_credentials

        credentials = load_credentials(environ={})
        return _model_access_note(credentials.hf_token, credentials.ngc_api_key)
    except Exception:  # noqa: BLE001 - advisory state never gates configure
        return (
            "[NOTE] Skipped model-access check "
            "(verify later with `npa workbench health access`)."
        )


def _configured_env_lines() -> str:
    """Return the saved ~/.npa values as ``NPA_*=`` shell assignments.

    Lets a runbook stop asking the operator to hand-substitute the alias, bucket
    and kube context: ``eval "$(npa configure --show --env)"``. Secrets are not
    included; workflow submit resolves requested ``--secret-env`` names directly
    from credentials.yaml without printing them.
    """
    alias, stanza = _configured_display_project(required=True)
    project_id = str(stanza.get("project_id", "") or "")
    lines: list[str] = []
    lines.append(f"NPA_PROJECT_ALIAS={shlex.quote(alias)}")
    for name, key in (
        ("NPA_PROJECT_ID", "project_id"),
        ("NPA_TENANT_ID", "tenant_id"),
        ("NPA_REGION", "region"),
    ):
        value = str(stanza.get(key, "") or "")
        if value:
            lines.append(f"{name}={shlex.quote(value)}")
    try:
        credentials = _credentials_for_exact_project(project_id)
    except Exception:  # noqa: BLE001
        credentials = None
    if credentials is not None:
        bucket_uri = str(credentials.s3_bucket or "")
        if bucket_uri:
            lines.append(f"NPA_BUCKET_URI={shlex.quote(bucket_uri)}")
            lines.append(
                "NPA_BUCKET="
                + shlex.quote(
                    bucket_uri.removeprefix("s3://").strip("/").split("/", 1)[0]
                )
            )
        if credentials.s3_endpoint:
            lines.append(f"NPA_S3_ENDPOINT={shlex.quote(credentials.s3_endpoint)}")
    context = _saved_kube_context(project_id)
    if context:
        lines.append(f"NPA_KUBE_CONTEXT={shlex.quote(context)}")
    return "\n".join(lines)


def _configured_display_project(*, required: bool) -> tuple[str, dict[str, Any]]:
    """Return the exact selected project, failing closed for machine output."""

    from npa.clients.config import default_project_name, list_projects

    try:
        projects = list_projects()
        alias = str(default_project_name() or "")
    except Exception as exc:
        if required:
            raise typer.BadParameter(
                "Saved project configuration is unreadable; refusing machine output."
            ) from exc
        return "", {}
    if not projects:
        if required:
            raise typer.BadParameter(
                "No saved project identity; refusing machine-evaluated --env output."
            )
        return "", {}
    if alias in projects:
        resolved = alias
    elif len(projects) == 1:
        resolved = next(iter(projects))
    else:
        if required:
            raise typer.BadParameter(
                "default_project does not select an exact saved project; refusing "
                "machine output."
            )
        return "", {}
    stanza = projects.get(resolved)
    if not isinstance(stanza, dict) or not str(stanza.get("project_id") or "").strip():
        if required:
            raise typer.BadParameter(
                "Selected project has no readable exact project_id; refusing "
                "machine output."
            )
        return "", {}
    return resolved, stanza


def _saved_kube_context(project_id: str) -> str:
    """Return the most recently saved local cluster context, or "".

    Scoped to the configured exact project, so
    ``configure --show --env`` never emits an unrelated project's cluster context
    (which would point a runbook / PAIDF submit at the wrong infra).
    """
    try:
        from npa.cluster.state import list_local_clusters

        clusters = list(list_local_clusters() or [])
    except Exception:  # noqa: BLE001 - no cluster cache is normal before provisioning
        return ""
    scoped = [
        state
        for state in clusters
        if project_id and str(getattr(state, "project_id", "") or "") == project_id
    ]
    for state in reversed(scoped):
        # `cluster up` names the kubeconfig context after the cluster by default.
        context = str(getattr(state, "name", "") or "")
        if context:
            return context
    return ""


def _configured_summary() -> str:
    """Return the resolved ~/.npa values a first run needs to fill in placeholders.

    `configure --show` used to print only the empty file template, so an operator
    following the runbook had to open the YAML by hand to find the alias, bucket
    and ids the quickstart placeholders want. Secrets are reported as present or
    missing, never echoed.
    """
    from npa.clients.config import CONFIG_PATH, default_project_name, list_projects
    from npa.clients.credentials import CREDENTIALS_PATH, load_credentials

    lines: list[str] = ["Current configuration"]
    projects = {}
    alias = ""
    try:
        projects = list_projects()
        alias = default_project_name()
    except Exception:  # noqa: BLE001 - a broken config must still print the layout
        projects = {}
    stanza: dict[str, Any] = {}
    resolved_alias = ""
    if not projects:
        lines.append(f"  (no projects in {CONFIG_PATH} — run `npa configure`)")
    else:
        lines.append(f"  config file:        {CONFIG_PATH}")
        if alias in projects:
            resolved_alias = alias
        elif len(projects) == 1:
            resolved_alias = next(iter(projects))
        if resolved_alias:
            candidate = projects.get(resolved_alias)
            stanza = candidate if isinstance(candidate, dict) else {}
            lines.append(f"  project alias:      {resolved_alias}  (use with -p)")
        else:
            lines.append(
                "  project alias:      (no exact default selected; storage hidden)"
            )
        if len(projects) > 1 and resolved_alias:
            lines.append(
                f"  other aliases:      {', '.join(a for a in projects if a != resolved_alias)}"
            )
        elif not resolved_alias:
            lines.append(f"  available aliases:  {', '.join(projects)}")
        if resolved_alias:
            for label, key in (
                ("project id", "project_id"),
                ("tenant id", "tenant_id"),
                ("region", "region"),
            ):
                value = str(stanza.get(key, "") or "")
                lines.append(f"  {label + ':':<19} {value or '(unset)'}")

    try:
        project_id = str(stanza.get("project_id", "") or "")
        credentials = load_credentials(environ={})
        if project_id:
            credentials = _credentials_for_exact_project(project_id)
        else:
            # Tokens are host-scoped and still useful without a project. S3 is
            # project-scoped, so never display an unselected compatibility view.
            credentials.s3_access_key_id = ""
            credentials.s3_secret_access_key = ""
            credentials.s3_endpoint = ""
            credentials.s3_bucket = ""
    except Exception:  # noqa: BLE001 - report what is readable
        credentials = None
    if credentials is not None:
        lines.append(f"  credentials file:   {CREDENTIALS_PATH}")
        lines.append(f"  s3 bucket:          {credentials.s3_bucket or '(unset)'}")
        lines.append(f"  s3 endpoint:        {credentials.s3_endpoint or '(unset)'}")
        for label, value in (
            ("s3 access key", credentials.s3_access_key_id),
            ("HF token", credentials.hf_token),
            ("Token Factory key", credentials.token_factory_api_key),
            ("NGC API key", credentials.ngc_api_key),
        ):
            lines.append(f"  {label + ':':<19} {'set' if value else 'not set'}")
    return "\n".join(lines)


def _forget_project(alias: str) -> None:
    """Remove a project stanza from ~/.npa/config.yaml (the configure inverse)."""
    from npa.clients.config import ConfigError, forget_project, resolve_environment
    from npa.cleanup_identity import project_cleanup_identity_snapshot
    from npa.teardown_receipts import record_teardown_event

    def persist_receipt(
        *,
        terminal_state: str,
        configured_project_found: bool,
        full_identity: dict[str, Any],
    ) -> Path | None:
        event: dict[str, Any] = {
            "phase": "project_config",
            "resource": cleaned,
            "terminal_state": terminal_state,
            "project_alias": cleaned,
            "project_id": project_id,
            "precheck": {"configured_project_found": configured_project_found},
            "action": {"kind": "forget_project_configuration"},
            "verification": {"config_removed": terminal_state == "completed"},
        }
        try:
            return record_teardown_event(**event, identity=full_identity)
        except (OSError, RuntimeError, ValueError) as exc:
            typer.echo(
                "Warning: full cleanup receipt could not be written; trying a "
                f"minimal safe fallback: {type(exc).__name__}: {exc}",
                err=True,
            )
        fallback = {
            "project_alias": cleaned,
            "project_id": project_id,
            "parent_id": project_id,
        }
        try:
            path = record_teardown_event(**event, identity=fallback)
        except (OSError, RuntimeError, ValueError) as exc:
            typer.echo(
                "Warning: minimal cleanup receipt also failed; local cleanup will "
                "continue with degraded audit evidence: "
                f"{type(exc).__name__}: {exc}",
                err=True,
            )
            return None
        typer.echo(
            "Warning: wrote a minimal safe cleanup receipt; detailed recovery "
            "evidence is degraded.",
            err=True,
        )
        return path

    cleaned = alias.strip()
    environment = resolve_environment(cleaned)
    project_id = str(getattr(environment, "project_id", "") or "")
    identity = project_cleanup_identity_snapshot(cleaned)
    # The intent/evidence lands outside config before the destructive rewrite.
    receipt_path = persist_receipt(
        terminal_state="in_progress",
        configured_project_found=environment is not None,
        full_identity=identity,
    )
    if receipt_path is not None:
        typer.echo(
            f"Durable cleanup identity: {receipt_path.stem}. Resume after config removal "
            f"with `npa agent destroy --receipt {receipt_path.stem} --name <name> --yes` "
            "or the matching cluster/storage/controller command."
        )
    try:
        forgotten = forget_project(cleaned)
    except ConfigError as exc:
        typer.echo(f"Partial cleanup: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    if forgotten:
        typer.echo(
            f"Removed project '{cleaned}' (stanza + terraform_state) from "
            "~/.npa/config.yaml."
        )
    else:
        typer.echo(f"No project '{cleaned}' in ~/.npa/config.yaml; nothing to remove.")
    persist_receipt(
        terminal_state="completed",
        configured_project_found=bool(forgotten),
        full_identity=identity,
    )


def _store_src_s3_uri(uri: str) -> None:
    from npa.clients.config import default_project_name, write_config

    if not uri.startswith("s3://"):
        typer.echo(f"Error: --src-s3-uri must be an s3:// URI, got {uri!r}", err=True)
        raise typer.Exit(code=1)
    try:
        project = str(default_project_name() or "")
    except Exception:  # noqa: BLE001 - fall back to a top-level key when unconfigured
        project = ""
    if project:
        path = write_config({"projects": {project: {"src_s3_uri": uri}}})
        location = f"projects.{project}.src_s3_uri"
    else:
        path = write_config({"src_s3_uri": uri})
        location = "src_s3_uri"
    typer.echo(f"Stored staged npa source prefix in {path} under {location}.")
    typer.echo("Workflow submits now resolve NPA_SRC_S3_URI without re-exporting it.")


def _run_known_project_configure(
    *,
    tenant_id: str,
    project_id: str,
    region: str,
    project_alias: str,
    provision: bool,
) -> None:
    """Configure a known project without profile creation, discovery, or prompts."""

    import re

    from npa.clients import nebius as nebius_client
    from npa.clients.config import CONFIG_PATH, list_projects, write_config

    values = {
        "--tenant-id": str(tenant_id or "").strip(),
        "--project-id": str(project_id or "").strip(),
        "--region": str(region or "").strip(),
        "--project-alias": str(project_alias or "").strip(),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise typer.BadParameter(
            "Known-project non-interactive configure requires all of --tenant-id, "
            "--project-id, --region, and --project-alias; missing " + ", ".join(missing)
        )
    alias = values["--project-alias"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", alias):
        raise typer.BadParameter(
            "--project-alias must start with a letter or digit and contain only "
            "letters, digits, '.', '_', or '-' (maximum 64 characters)"
        )

    existing_projects = list_projects()
    _validate_alias_project_identity(
        alias, existing_projects.get(alias) or {}, values["--project-id"]
    )

    # This path consumes existing local profile/credential material only. It
    # never invokes profile creation or tenant/project discovery, so a valid
    # federation/service-account profile cannot fall into a browser flow or a
    # large tenant picker.
    if provision:
        try:
            nebius_client.get_iam_token()
        except nebius_client.NebiusError as exc:
            raise typer.BadParameter(
                "The active Nebius CLI profile cannot authenticate non-interactively. "
                "Activate or refresh a profile/service-account credential outside "
                "this command, then retry."
            ) from exc

    existing = _credentials_for_exact_project(
        values["--project-id"], allow_provable_legacy=True
    )
    storage: dict[str, str] = {}
    existing_complete = bool(
        existing.s3_access_key_id
        and existing.s3_secret_access_key
        and existing.s3_bucket
    )
    existing_relationship_verified = _storage_relationship_verified(
        existing, values["--project-id"]
    )
    if provision and existing_complete and existing_relationship_verified:
        from npa.clients.storage_validation import probe_storage_write

        probe = probe_storage_write(
            bucket=existing.s3_bucket,
            endpoint_url=existing.s3_endpoint
            or _endpoint_for_region(values["--region"]),
            access_key_id=existing.s3_access_key_id,
            secret_access_key=existing.s3_secret_access_key,
            region=values["--region"],
        )
        if probe.ok:
            storage = {
                "aws_access_key_id": existing.s3_access_key_id,
                "aws_secret_access_key": existing.s3_secret_access_key,
                "endpoint_url": existing.s3_endpoint
                or _endpoint_for_region(values["--region"]),
                "bucket": existing.s3_bucket,
                "_validated": "true",
            }
            typer.echo(
                "Reusing health-verified configured object-storage credentials; "
                "no access keys were listed, created, or rotated."
            )
    elif provision and existing_complete:
        typer.echo(
            "Ignoring saved object storage for this non-interactive configure: "
            "its durable ownership record does not match the selected project."
        )
    elif not provision and existing_complete:
        typer.echo(
            "Warning: saved object storage was not selected or probed because "
            "--no-provision performs project-only setup.",
            err=True,
        )

    if provision and not storage:
        # Supply documented defaults programmatically; no prompt or secret argv
        # is involved. The provisioner retains its safe parser, health probe,
        # ownership journal, and partial-resource recovery semantics.
        def _accept_default(_label: str, *, default: str = "", **_kwargs: Any) -> str:
            return str(default or "")

        provisioned = _provision_object_storage(
            nebius_client,
            _accept_default,
            project_id=values["--project-id"],
            tenant_id=values["--tenant-id"],
            region=values["--region"],
            existing_bucket=(
                _bucket_name_from_uri(existing.s3_bucket)
                if existing_relationship_verified
                else ""
            ),
            interactive=False,
        )
        if provisioned is None or provisioned.get("_validated") != "true":
            raise typer.BadParameter(
                "Non-interactive object-storage provisioning did not produce "
                "health-verified credentials. Fix the reported IAM/storage error "
                "and retry; no secret should be passed on the command line."
            )
        storage = provisioned

    from npa.clients.project_credential_store import (
        select_project_credentials,
        write_project_credentials,
    )

    if storage:
        service_account_keys = {
            key: str(storage.get(key, "") or "").strip()
            for key in (
                "service_account_id",
                "service_account_name",
                "service_account_project_id",
                "service_account_managed_by",
            )
            if str(storage.get(key, "") or "").strip()
        }
        credentials_payload: dict[str, object] = {
            "storage": {
                key: value
                for key, value in storage.items()
                if not key.startswith("service_account_")
                and not key.startswith("_")
                and value
            }
        }
        account_id = service_account_keys.get("service_account_id", "")
        if account_id:
            credentials_payload["nebius"] = {"service_account_id": account_id}
        if len(service_account_keys) == 4:
            credentials_payload["storage_iam"] = service_account_keys
        credentials_path = write_project_credentials(
            values["--project-id"],
            credentials_payload,
            alias=alias,
        )
        typer.echo(f"Wrote {credentials_path} (chmod 600).")
    else:
        select_project_credentials(
            values["--project-id"], alias=alias, select_storage=False
        )

    write_config(
        {
            "projects": {
                alias: {
                    "project_id": values["--project-id"],
                    "tenant_id": values["--tenant-id"],
                    "region": values["--region"],
                }
            },
            "default_project": alias,
        }
    )
    if provision and not nebius_client.set_profile_project(
        values["--project-id"], values["--tenant-id"]
    ):
        typer.echo(
            "Warning: the active Nebius CLI profile could not be rebound, but NPA "
            "saved the explicit project/tenant IDs. Keep the intended profile active "
            "for later provider commands.",
            err=True,
        )
    typer.echo(f"Wrote {CONFIG_PATH} (project alias: {alias}, non-interactive).")
    if storage:
        typer.echo(
            "Project and writable object storage configuration is health-verified."
        )
    else:
        typer.echo(
            "Project setup complete without object storage (--no-provision). "
            "Configure writable storage before agent or workflow submission."
        )
    typer.echo(_saved_model_access_note())


class ConfigureMode(str, Enum):
    DISPLAY = "display"
    GUIDANCE = "guidance"
    SOURCE_URI = "source-uri"
    FORGET_PROJECT = "forget-project"
    IMPORT_CREDENTIALS = "import-credentials"
    KNOWN_PROJECT = "known-project"
    INTERACTIVE = "interactive"


def _click_parameter_was_explicit(
    name: str, context: Any | None = None
) -> bool:
    """Return whether Click received *name* on argv (False for direct calls)."""

    context = context or click.get_current_context(silent=True)
    if context is None:
        return False
    source = context.get_parameter_source(name)
    # Typer 0.21 may provide its vendored Click context/enum here, so compare
    # the stable enum name rather than relying on cross-package identity.
    return getattr(source, "name", "") == "COMMANDLINE"


def _configure_mode(arguments: dict[str, Any]) -> ConfigureMode:
    """Resolve and validate one unambiguous configure mode before any write."""

    show = bool(arguments.get("show"))
    env_output = bool(arguments.get("env_output"))
    source_uri = str(arguments.get("src_s3_uri") or "").strip()
    forget = str(arguments.get("forget_project") or "").strip()
    save_env = bool(arguments.get("save_env_credentials"))
    known_names = ("tenant_id", "project_id", "region", "project_alias")
    known = {
        name: str(arguments.get(name) or "").strip()
        for name in known_names
        if str(arguments.get(name) or "").strip()
    }
    requested: list[ConfigureMode] = []
    if show or env_output:
        requested.append(ConfigureMode.DISPLAY)
    if source_uri:
        requested.append(ConfigureMode.SOURCE_URI)
    if forget:
        requested.append(ConfigureMode.FORGET_PROJECT)
    if save_env:
        requested.append(ConfigureMode.IMPORT_CREDENTIALS)
    if known:
        requested.append(ConfigureMode.KNOWN_PROJECT)
    if len(requested) > 1:
        raise typer.BadParameter(
            "Configure modes cannot be combined: choose exactly one of display "
            "(--show/--env), --src-s3-uri, --forget-project, "
            "--save-env-credentials, or known-project setup."
        )
    interactive = arguments.get("interactive")
    mode = (
        requested[0]
        if requested
        else (
            ConfigureMode.INTERACTIVE
            if interactive is True or (interactive is None and sys.stdin.isatty())
            else ConfigureMode.GUIDANCE
        )
    )
    allow_visible_flag = bool(arguments.get("allow_visible_secret_input"))
    allow_visible = (
        allow_visible_flag or os.environ.get("NPA_ALLOW_VISIBLE_SECRET_INPUT") == "1"
    )
    context = arguments.get("context")
    provision_explicit = _click_parameter_was_explicit(
        "provision",
        context if callable(getattr(context, "get_parameter_source", None)) else None,
    ) or (
        mode != ConfigureMode.INTERACTIVE and arguments.get("provision") is False
    )
    if mode in {
        ConfigureMode.DISPLAY,
        ConfigureMode.SOURCE_URI,
        ConfigureMode.FORGET_PROJECT,
    } and (interactive is not None or provision_explicit or allow_visible_flag):
        raise typer.BadParameter(
            f"{mode.value} mode is read-only or self-contained and cannot be "
            "combined with interactive/provisioning options."
        )
    if mode == ConfigureMode.IMPORT_CREDENTIALS:
        if interactive is True or provision_explicit or allow_visible_flag:
            raise typer.BadParameter(
                "--save-env-credentials is a prompt-free import mode and cannot "
                "be combined with interactive/provisioning options."
            )
    if mode == ConfigureMode.KNOWN_PROJECT:
        missing = [name for name in known_names if name not in known]
        if missing:
            flags = ", ".join("--" + name.replace("_", "-") for name in missing)
            raise typer.BadParameter(
                "Known-project non-interactive configure requires all of "
                "--tenant-id, --project-id, --region, and --project-alias; "
                f"missing {flags}"
            )
        if interactive is True or allow_visible_flag:
            raise typer.BadParameter(
                "Known-project flags select the prompt-free configure path; use "
                "--no-interactive (or omit --interactive)."
            )
        import re

        alias = known["project_alias"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", alias):
            raise typer.BadParameter(
                "--project-alias must start with a letter or digit and contain "
                "only letters, digits, '.', '_', or '-' (maximum 64 characters)"
            )
        from npa.clients.config import list_projects

        projects = list_projects()
        _validate_alias_project_identity(
            alias, projects.get(alias) or {}, known["project_id"]
        )
    if source_uri and not source_uri.startswith("s3://"):
        raise typer.BadParameter("--src-s3-uri must be an s3:// URI")
    if allow_visible_flag and interactive is not True:
        raise typer.BadParameter(
            "Visible-secret input opt-in is valid only with --interactive."
        )
    if interactive is True and not sys.stdin.isatty() and not allow_visible:
        raise typer.BadParameter(
            "Refusing forced interactive setup on non-TTY stdin because secret "
            "input would be visible. Prefer --save-env-credentials for automation, "
            "or explicitly pass --allow-visible-secret-input."
        )
    return mode


def _configure_impl(
    *,
    show: bool,
    interactive: Optional[bool],
    provision: bool = True,
    save_env_credentials: bool = False,
    env_output: bool = False,
    forget_project: str = "",
    src_s3_uri: str = "",
    tenant_id: str = "",
    project_id: str = "",
    region: str = "",
    project_alias: str = "",
    allow_visible_secret_input: bool = False,
) -> None:
    arguments = locals()
    mode = _configure_mode(arguments)
    if mode == ConfigureMode.SOURCE_URI:
        _store_src_s3_uri(src_s3_uri.strip())
        typer.echo(_saved_model_access_note())
        return
    if mode == ConfigureMode.FORGET_PROJECT:
        # Deconfigure a single project: the inverse of the stanza configure
        # writes. Storage credentials (host-scoped) and a deleted bucket's
        # remote-state keys are handled by `npa storage bucket delete`.
        _forget_project(forget_project.strip())
        return
    from npa.clients.credentials import (
        CREDENTIALS_PATH,
        SUPPORTED_ENV_CREDENTIALS,
        persist_supported_env_credentials,
    )

    if mode == ConfigureMode.DISPLAY:
        if env_output:
            typer.echo(_configured_env_lines())
        else:
            typer.echo(_configured_summary())
            typer.echo("")
            typer.echo(_SETUP_GUIDANCE)
        return
    if mode == ConfigureMode.IMPORT_CREDENTIALS:
        report = persist_supported_env_credentials()
        typer.echo(
            "Credential environment sources detected: "
            + (", ".join(report["detected"]) if report["detected"] else "none"),
        )
        typer.echo(
            "Credential fields persisted (values redacted): "
            + (", ".join(report["persisted"]) if report["persisted"] else "none")
            + f"; store={CREDENTIALS_PATH}",
        )
        for warning in report["warnings"]:
            typer.echo(f"Credential warning: {warning}", err=True)
        typer.echo(_saved_model_access_note())
        return
    if mode == ConfigureMode.KNOWN_PROJECT:
        _run_known_project_configure(
            tenant_id=tenant_id,
            project_id=project_id,
            region=region,
            project_alias=project_alias,
            provision=provision,
        )
        return

    detected = [name for name in SUPPORTED_ENV_CREDENTIALS if os.environ.get(name)]
    if detected and interactive is not True:
        typer.echo(
            f"Credential environment sources detected: {', '.join(detected)}",
        )
        typer.echo(
            "Credential persistence: skipped. These values remain process-only; "
            "use --save-env-credentials to make later agent/workflow commands durable.",
        )
    elif interactive is False:
        typer.echo(
            "Credential environment sources detected: none; persistence: none.",
        )

    should_prompt = interactive if interactive is not None else sys.stdin.isatty()
    if not should_prompt:
        typer.echo(_SETUP_GUIDANCE)
        return
    try:
        _run_interactive_configure(
            provision=provision,
            allow_visible_secret_input=(
                allow_visible_secret_input
                or os.environ.get("NPA_ALLOW_VISIBLE_SECRET_INPUT") == "1"
            ),
        )
    except (EOFError, typer.Abort):
        # Cancelling mid-flow (Ctrl-C / Ctrl-D / no more input) previously exited
        # 0 having written nothing under ~/.npa, so the next cloud command failed
        # mysteriously. Fail loudly instead so the missing setup is obvious.
        typer.echo("")
        written = "Setup was cancelled before anything was written under ~/.npa."
        typer.echo(
            f"{written} Re-run `npa configure` in a terminal to finish, or run "
            "`npa configure --show` for the file layout to create it by hand."
        )
        raise typer.Exit(code=1)


def _transactional_configure(function):
    """Journal configure mutations without ever copying credential values."""

    signature = inspect.signature(function)

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        mode = _configure_mode(bound.arguments)
        if current_operation() is not None:
            from npa.lifecycle_intent import operation_intent

            with operation_intent(
                OperationIntent.DESTROY
                if mode == ConfigureMode.FORGET_PROJECT
                else OperationIntent.ENSURE_PRESENT
            ):
                return function(*args, **kwargs)
        if mode in {ConfigureMode.DISPLAY, ConfigureMode.GUIDANCE}:
            return function(*args, **kwargs)
        forget = str(bound.arguments.get("forget_project") or "").strip()
        alias = str(bound.arguments.get("project_alias") or "").strip()
        project_id = str(bound.arguments.get("project_id") or "").strip()
        tenant_id = str(bound.arguments.get("tenant_id") or "").strip()
        region = str(bound.arguments.get("region") or "").strip()
        if mode == ConfigureMode.FORGET_PROJECT:
            from npa.clients.config import resolve_environment

            environment = resolve_environment(forget)
            alias = forget
            if environment is not None:
                project_id = str(environment.project_id or "")
                tenant_id = str(environment.tenant_id or "")
                region = str(environment.region or "")
        requested_name = alias or project_id or mode.value
        resume = (
            f"npa configure --forget-project {forget}" if forget else "npa configure"
        )
        if not forget and all((tenant_id, project_id, region, alias)):
            resume += (
                f" --no-interactive --tenant-id {tenant_id} --project-id {project_id}"
                f" --region {region} --project-alias {alias}"
            )
        operation = ProvisioningOperation.prepare(
            command=(
                "npa configure --forget-project"
                if mode == ConfigureMode.FORGET_PROJECT
                else "npa configure"
            ),
            project_alias=alias,
            project_id=project_id,
            tenant_id=tenant_id,
            region=region,
            resource_type=(
                "forget-project"
                if mode == ConfigureMode.FORGET_PROJECT
                # "configure" owns the bootstrap lease until an interactive
                # setup has discovered its exact project identity. The
                # selected mode remains available in requested_name.
                else "configure"
            ),
            requested_name=requested_name,
            ownership_source="configure-cli",
            resume_command=resume,
        )
        from npa.clients.credentials import SUPPORTED_ENV_CREDENTIALS

        detected = [name for name in SUPPORTED_ENV_CREDENTIALS if os.environ.get(name)]
        writes_credentials = mode in {
            ConfigureMode.IMPORT_CREDENTIALS,
            ConfigureMode.KNOWN_PROJECT,
            ConfigureMode.INTERACTIVE,
        }
        writes_config = mode in {
            ConfigureMode.SOURCE_URI,
            ConfigureMode.FORGET_PROJECT,
            ConfigureMode.KNOWN_PROJECT,
            ConfigureMode.INTERACTIVE,
        }
        if writes_credentials:
            operation.record_config_mutation(
                store="credentials.yaml",
                fields=(detected if mode == ConfigureMode.IMPORT_CREDENTIALS else []),
                secret_fields=(
                    detected if mode == ConfigureMode.IMPORT_CREDENTIALS else []
                ),
            )
        if writes_config:
            operation.record_config_mutation(
                store="config.yaml",
                fields=(
                    ["default_project", f"projects.{alias}"] if alias else [mode.value]
                ),
            )
        with operation_context(operation):
            from npa.clients.config import CONFIG_PATH
            from npa.clients.credentials import (
                CREDENTIALS_PATH,
                preflight_private_yaml_store,
            )

            # Configuration durability is a prerequisite, not a cleanup task:
            # prove both protected stores can be atomically replaced before any
            # profile, storage, or other provider mutation is attempted.
            if writes_config:
                preflight_private_yaml_store(CONFIG_PATH)
            if writes_credentials:
                preflight_private_yaml_store(CREDENTIALS_PATH)
            operation.transition("mutating")

            def finish_success() -> None:
                phase = str(operation.read().get("phase") or "")
                if phase in {"mutating", "resource-created"}:
                    operation.transition(
                        "state-durable", details={"private_stores": "fsynced"}
                    )
                operation.commit()

            try:
                intent = (
                    OperationIntent.DESTROY
                    if mode == ConfigureMode.FORGET_PROJECT
                    else OperationIntent.ENSURE_PRESENT
                )
                from npa.lifecycle_intent import operation_intent

                with operation_intent(intent):
                    result = function(*args, **kwargs)
            except typer.Exit as exc:
                if exc.exit_code == 0:
                    finish_success()
                    raise
                phase = str(operation.read().get("phase") or "")
                if phase not in {
                    "recovery-required",
                    "rolled-back",
                    "rollback-incomplete",
                }:
                    operation.transition("recovery-required", error=type(exc).__name__)
                typer.echo(emit_recovery_summary(operation), err=True)
                raise
            except BaseException as exc:
                phase = str(operation.read().get("phase") or "")
                if phase not in {
                    "recovery-required",
                    "rolled-back",
                    "rollback-incomplete",
                }:
                    operation.transition("recovery-required", error=type(exc).__name__)
                typer.echo(emit_recovery_summary(operation), err=True)
                raise
            finish_success()
            return result

    return wrapped


@app.command(
    "init",
    help="Deprecated alias for interactive credential and config setup guidance.",
    rich_help_panel="Setup",
)
@app.command(
    "configure",
    help="Interactive credential and config setup guidance.",
    rich_help_panel="Setup",
)
@_transactional_configure
def configure(
    context: typer.Context,
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
    save_env_credentials: bool = typer.Option(
        False,
        "--save-env-credentials",
        help=(
            "Persist supported credentials already present in environment variables "
            "to ~/.npa/credentials.yaml using an atomic 0600 write; values are never printed."
        ),
    ),
    env_output: bool = typer.Option(
        False,
        "--env",
        help=(
            "Print the saved project/bucket/kube-context values as NPA_* shell "
            "assignments (no secrets) instead of prompting: "
            'eval "$(npa configure --show --env)".'
        ),
    ),
    forget_project: str = typer.Option(
        "",
        "--forget-project",
        help=(
            "Remove a project stanza (and its terraform_state) from "
            "~/.npa/config.yaml, then exit — the inverse of writing it. Use "
            "`npa storage bucket delete` and `npa agent destroy` to clean up the "
            "cloud resources and their credentials first."
        ),
    ),
    src_s3_uri: str = typer.Option(
        "",
        "--src-s3-uri",
        help=(
            "Persist the staged npa source prefix (s3://bucket/prefix/npa) in "
            "~/.npa/config.yaml so workflow submits resolve NPA_SRC_S3_URI without "
            "re-exporting it in every shell (skips interactive setup)."
        ),
    ),
    tenant_id: str = typer.Option(
        "",
        "--tenant-id",
        help="Known Nebius tenant ID for prompt-free configure (requires the other known-project flags).",
    ),
    project_id: str = typer.Option(
        "",
        "--project-id",
        help="Known Nebius project ID for prompt-free configure (requires the other known-project flags).",
    ),
    region: str = typer.Option(
        "",
        "--region",
        help="Known Nebius project region for prompt-free configure.",
    ),
    project_alias: str = typer.Option(
        "",
        "--project-alias",
        help="Local NPA alias to write for the known project (prompt-free configure).",
    ),
    allow_visible_secret_input: bool = typer.Option(
        False,
        "--allow-visible-secret-input",
        help=(
            "Explicitly allow --interactive secret prompts on non-TTY stdin. "
            "Values may be visible; prefer --save-env-credentials for automation. "
            "Equivalent explicit opt-in: NPA_ALLOW_VISIBLE_SECRET_INPUT=1."
        ),
    ),
) -> None:
    """Interactively write ~/.npa credentials and config, or show guidance."""
    _configure_impl(
        show=show,
        interactive=interactive,
        provision=provision,
        save_env_credentials=save_env_credentials,
        env_output=env_output,
        forget_project=forget_project,
        src_s3_uri=src_s3_uri,
        tenant_id=tenant_id,
        project_id=project_id,
        region=region,
        project_alias=project_alias,
        allow_visible_secret_input=allow_visible_secret_input,
    )


def app_entry() -> None:
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)
    except ServerlessClientError as exc:
        print(
            format_error_for_user(exc, output_format=_detect_error_format()),
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as exc:
        print(
            format_error_for_user(exc, output_format=_detect_error_format()),
            file=sys.stderr,
        )
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
        if value in {"--output", "--output-format", "--format"} and index + 1 < len(
            args
        ):
            if args[index + 1].lower() == "json":
                return "json"
        if value in {"--output=json", "--output-format=json", "--format=json"}:
            return "json"
    return "text"


if __name__ == "__main__":
    app_entry()
