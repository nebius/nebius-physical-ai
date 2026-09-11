"""npa CLI entry point."""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import secrets
import shlex
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from enum import Enum
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
prompt. Enter an exact existing bucket name only when you intend to reuse it;
otherwise pressing Enter generates a fresh collision-safe name. Use
`npa configure --no-provision` for provider-free project and token configuration,
or create ~/.npa/credentials.yaml by hand for user-level tokens, object storage,
and BYOVM SSH defaults. Prompt-free known-project setup is also provider-free
unless `--provision` is explicit. Hugging Face and NGC checks performed during
provisioning or credential import are informational; use
`npa workbench health access` as the enforcing gate:

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
  # (sign in / create a free NGC account first). Personal keys commonly start
  # with "nvapi-"; online NGC authentication accepts other credential shapes.
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
mirror by default and ignore ambient/saved private-registry settings. Use a
complete image or an explicit workflow --registry when you need private or
locally modified bytes. Deploy commands extend the same file with
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
    return _generated_configure_bucket_name(tenant_id, project_id)


def _generated_configure_bucket_name(tenant_id: str, project_id: str) -> str:
    """Return a fresh, globally collision-resistant configure bucket name."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%Sz")
    identity = hashlib.sha256(
        f"{tenant_id}\0{project_id}".encode("utf-8")
    ).hexdigest()[:6]
    return f"npa-bucket-{timestamp}-{identity}-{secrets.token_hex(4)}"


def _storage_relationship_verified(credentials: Any, project_id: str) -> bool:
    """Return whether saved storage has durable provenance for this project."""

    return bool(
        str(project_id or "").strip()
        and str(getattr(credentials, "s3_project_id", "") or "").strip()
        == str(project_id or "").strip()
        and str(getattr(credentials, "s3_ownership", "") or "").strip()
    )


def _provision_object_storage(
    nebius_client,
    ask: Callable[..., str],
    *,
    project_id: str,
    tenant_id: str,
    region: str,
    existing_bucket: str = "",
    interactive: bool = True,
    bucket_storage_class: str = "",
    bucket_max_size_bytes: int | None = None,
    _generated_name: bool = False,
) -> dict[str, str] | None:
    """Auto-create the S3 bucket + access key for the project."""
    if not (project_id and tenant_id):
        return None

    if interactive:
        typer.echo(
            "\nObject storage: enter an exact existing bucket name to reuse it, or "
            "enter a new name to create it. Press Enter for a fresh collision-safe "
            "name containing a UTC timestamp and random suffix. Existing buckets "
            "are reused only after this explicit selection."
        )
        bucket_name = ask("Object-storage bucket name", default=existing_bucket).strip()
    else:
        bucket_name = str(existing_bucket or "").strip()
        selected = bucket_name or _generated_configure_bucket_name(
            tenant_id, project_id
        )
        _generated_name = _generated_name or not bool(bucket_name)
        bucket_name = selected
        typer.echo(
            "Object storage (non-interactive): "
            + (
                f"generated fresh name '{selected}'."
                if _generated_name
                else f"explicitly selected exact name '{selected}'."
            )
        )
    if not bucket_name:
        bucket_name = _generated_configure_bucket_name(tenant_id, project_id)
        _generated_name = True
        typer.echo(
            f"  No bucket name provided; generated fresh bucket name "
            f"'{bucket_name}'."
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

    requested_storage_class = str(bucket_storage_class or "").strip()
    requested_max_size_bytes = bucket_max_size_bytes
    create_max_size_bytes = 0
    create_storage_class = DEFAULT_BUCKET_STORAGE_CLASS
    if exists is True:
        if _generated_name:
            typer.echo(
                f"  Generated name collision: exact bucket '{bucket_name}' already "
                "exists. It will not be adopted or changed; re-run configure to "
                "generate another fresh name."
            )
            return None
        if requested_storage_class or requested_max_size_bytes is not None:
            item = nebius_client.get_bucket_by_name(project_id, bucket_name)
            spec = (item or {}).get("spec", {})
            actual_storage_class = nebius_client.normalize_bucket_storage_class(
                str(spec.get("default_storage_class", ""))
            )
            actual_max_size_bytes = int(spec.get("max_size_bytes", 0) or 0)
            expected_storage_class = (
                nebius_client.normalize_bucket_storage_class(requested_storage_class)
                if requested_storage_class
                else actual_storage_class
            )
            expected_max_size_bytes = (
                int(requested_max_size_bytes)
                if requested_max_size_bytes is not None
                else actual_max_size_bytes
            )
            if (
                actual_storage_class != expected_storage_class
                or actual_max_size_bytes != expected_max_size_bytes
            ):
                typer.echo(
                    "Existing object-storage bucket does not match the requested "
                    "create-only storage class and size; refusing to reuse it."
                )
                return None
        typer.echo(
            f"Reusing existing object-storage bucket '{bucket_name}'. Explicitly "
            f"reusing exact existing bucket '{bucket_name}'; no bucket was created."
        )
    elif exists is False:
        typer.echo(
            f"Exact lookup found no bucket named '{bucket_name}'; npa will create it."
        )
        if interactive:
            create_storage_class, create_max_size_bytes = _prompt_new_bucket_settings(
                ask,
                bucket_name=bucket_name,
            )
        else:
            create_storage_class = nebius_client.normalize_bucket_storage_class(
                requested_storage_class
            )
            create_max_size_bytes = int(requested_max_size_bytes or 0)
            typer.echo(
                f"  Will create '{bucket_name}' with {create_storage_class} storage "
                f"and "
                + (
                    f"a {create_max_size_bytes} byte cap."
                    if create_max_size_bytes
                    else "no size limit."
                )
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
            bucket_max_size_bytes=create_max_size_bytes,
            bucket_storage_class=create_storage_class,
            allow_existing_bucket=not _generated_name,
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
                "  Global bucket-name collision: the exact name is not in this "
                f"project and will not be adopted; retrying with '{alternate}'."
            )
            return _provision_object_storage(
                nebius_client,
                ask,
                project_id=project_id,
                tenant_id=tenant_id,
                region=region,
                existing_bucket=alternate,
                interactive=interactive,
                bucket_storage_class=requested_storage_class,
                bucket_max_size_bytes=requested_max_size_bytes,
                _generated_name=True,
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
    disposition = str(creds.get("bucket_disposition") or "").strip().lower()
    if disposition not in {"created", "reused"}:
        disposition = "reused" if exists is True else "created"
    if disposition == "reused":
        typer.echo(f"  Configured existing bucket {bucket}; {probe.summary}")
    else:
        typer.echo(f"  Created object-storage bucket {bucket}; {probe.summary}")
    payload: dict[str, str] = {
        "aws_access_key_id": access_key,
        "aws_secret_access_key": secret_key,
        "endpoint_url": creds.get("s3_endpoint", "") or _endpoint_for_region(region),
        "bucket": bucket,
        "_validated": "true",
        "_disposition": disposition,
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
    *,
    skip: set[str] = frozenset(),  # type: ignore[assignment]
) -> tuple[str, str, str]:
    """Prompt for the optional HF / Token Factory / NGC keys. Returns the trio.

    A token env-key in ``skip`` was already persisted from the environment; keep
    the existing value instead of re-prompting.
    """
    if "HF_TOKEN" in skip:
        typer.echo("\nHugging Face token: kept from the credential store.")
        hf_token = existing_credentials.hf_token
    else:
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
    if "NEBIUS_TOKEN_FACTORY_KEY" in skip:
        typer.echo("Nebius Token Factory API key: kept from the credential store.")
        token_factory_api_key = existing_credentials.token_factory_api_key
    else:
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
    if "NGC_API_KEY" in skip:
        typer.echo("NVIDIA NGC API key: kept from the credential store.")
        ngc_api_key = existing_credentials.ngc_api_key
    else:
        typer.echo(
            "\nNVIDIA NGC API key (for entitlement-controlled NGC artifact pulls): create one at "
            "https://org.ngc.nvidia.com/setup/api-key (sign in or make a free NGC "
            "account first). Personal keys commonly start with 'nvapi-'; NGC "
            "authentication is the authority for other credential shapes. "
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


def _warn_repointed_alias(alias: str, stanza: dict[str, Any], project_id: str) -> None:
    """Warn when an alias is repointed at a new project but keeps per-project state.

    Config is deep-merged, so ``terraform_state`` (the old project's remote-state
    bucket and access key) and ``workbenches`` (endpoints of VMs in the old
    project) survive under the alias and would be used for the new project.
    """
    previous = str((stanza or {}).get("project_id", "") or "")
    if not previous or previous == project_id:
        return
    stale = [
        key for key in ("terraform_state", "workbenches") if (stanza or {}).get(key)
    ]
    typer.echo(
        f"\nWarning: project alias '{alias}' pointed at {previous} and now points at "
        f"{project_id}."
        + (
            f" Its saved {' and '.join(stale)} still describe {previous}; remove those "
            f"keys from ~/.npa/config.yaml (or use a new alias) unless they apply to "
            f"{project_id} too."
            if stale
            else ""
        ),
        err=True,
    )


def _validate_alias_project_identity(
    alias: str, stanza: dict[str, Any], project_id: str
) -> None:
    """Refuse to merge a new project into an alias holding another project."""

    previous = str((stanza or {}).get("project_id", "") or "").strip()
    if not previous or previous == project_id:
        return
    raise typer.BadParameter(
        f"Project alias '{alias}' already belongs to {previous}; choose a new "
        "--project-alias instead of mixing project-scoped state."
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
    already_written: str = "",
    preset_tokens: set[str] | None = None,
) -> None:
    """Prompt for credentials/config and write the NPA dotfiles.

    ``already_written`` names what a caller persisted before this flow started
    (currently ``--save-env-credentials``), so the bail-out paths below never claim
    that nothing was saved. ``preset_tokens`` names token env-keys already stored
    via a flag (``HF_TOKEN`` / ``NEBIUS_TOKEN_FACTORY_KEY`` / ``NGC_API_KEY``) so
    the interactive flow keeps them instead of re-prompting (and risking a wipe).
    """

    from npa.clients.config import (
        CONFIG_PATH,
        config_permissions_warning,
        default_project_name,
        list_projects,
        write_config,
    )
    from npa.clients.credentials import load_credentials, write_credentials_file
    from npa.clients import nebius as nebius_client

    provider_free = not provision
    typer.echo(
        "Intent: "
        + (
            "interactive setup with explicit object-storage provisioning"
            if provision
            else "interactive project configuration only (no provider calls)"
        )
        + ".\n"
    )
    typer.echo(
        "Interactive npa setup. Existing values are shown as defaults — press "
        "Enter to keep them, or type a new value to update.\n"
    )
    profile_ready = _ensure_nebius_profile() if provision else False
    if not provision:
        typer.echo(
            "Provider discovery and object-storage checks are skipped by "
            "--no-provision."
        )
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
            "  - Re-run `npa configure --no-provision` for provider-free project "
            "and token configuration; add storage later with an explicit "
            "provisioning run."
        )
        if already_written:
            # The requested write succeeded; only the rest of setup is pending.
            typer.echo(
                f"{already_written} was saved; nothing else was written under ~/.npa."
            )
            raise typer.Exit(code=0)
        typer.echo("Nothing was written under ~/.npa.")
        raise typer.Exit(code=1)

    # Hidden input needs a controlling terminal. On piped stdin (scripted setup,
    # `printf ... | npa configure --interactive`) getpass cannot turn the echo
    # off and emits `GetPassWarning: Can not control echo on the terminal`,
    # after which the value it reads is easy to mis-bind. Fall back to a visible
    # prompt there and say so once, instead of warning per secret.
    stdin_is_tty = sys.stdin.isatty()
    echo_notice_shown = False

    def ask(label: str, *, default: str = "", secret: bool = False) -> str:
        nonlocal echo_notice_shown
        hide_input = secret and stdin_is_tty
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
        region_default = (
            str(existing_stanza.get("region", ""))
            or DEFAULT_REGION
        )
        region = ask("Region", default=region_default)

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
    if not provision:
        storage = {}
        typer.echo(
            "  Object storage not selected; no bucket, access key, or profile "
            "state will be created, probed, or adopted."
        )

    # Interactive configure offers storage by default because agent and workbench
    # data paths need it, but the user can decline before any storage mutation.
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
        ask, existing_credentials, skip=preset_tokens or set()
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
        if not alias and existing_default_alias not in ("", "default"):
            # Typing a new project id at the prompt (its default is the saved one)
            # is an explicit request to repoint this alias, so keep the alias — but
            # say which of its saved values still describe the previous project.
            alias = existing_default_alias
            _warn_repointed_alias(alias, existing_projects.get(alias) or {}, project_id)
        if not alias:
            project_name = ""
            try:
                if not profile_ready:
                    raise RuntimeError("provider discovery disabled")
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
            select_project_credentials(
                project_id,
                alias=alias,
                select_storage=False,
            )

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

    storage_disposition = (
        str(storage.get("_disposition") or "configured")
        if storage
        else "not selected"
    )
    typer.echo(
        f"Summary: mode={'provision' if provision else 'project-only'}; "
        f"project={alias or 'not set'}; storage={storage_disposition}; "
        "configuration=saved."
    )
    typer.echo(
        _skipped_model_access_note()
        if provider_free
        else _model_access_note(hf_token, ngc_api_key)
    )
    if sys.stdin.isatty():
        prepare_access = ask(
            "Prepare access for the full known Workbench HF/NGC catalog now? "
            "This is optional and does not affect unrelated capabilities. [y/N]",
            default="N",
        )
        if prepare_access.lower() in {"y", "yes"}:
            _prepare_full_catalog_access()
    if not provision:
        typer.echo(
            "Project configuration saved without object storage. Re-run with "
            "--provision when cloud storage should be created or explicitly reused."
        )
        return
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

    Cache keys include repository type, exact revision, and payload path so
    datasets and multiple pinned revisions cannot collapse into metadata-only
    or cross-revision evidence. Assets that do not finish inside ``total_budget``
    (or whose probe raises) are omitted, so the caller treats them as unverified.
    Exception messages are deliberately not logged because an injected client
    may echo its credential.
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
                asset.revision,
                asset.probe_path,
                timeout=per_probe_timeout,
            ): (asset.repo, asset.repo_type, asset.revision, asset.probe_path)
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
    """Return a redacted, non-gating HF/NGC access summary for configure."""

    try:
        return _build_model_access_note(hf_token, ngc_key)
    except Exception as exc:  # noqa: BLE001 - the whole advisory is non-gating
        logger.debug(
            "Configure model-access advisory unavailable (%s)",
            type(exc).__name__,
        )
        return (
            "[NOTE] Access checks are informational; HF/NGC access advisory "
            "unavailable. Configuration remains saved. Use `npa workbench health "
            "access` when model access must be enforced."
        )


def _build_model_access_note(hf_token: str, ngc_key: str) -> str:
    """Build the advisory inside the fail-safe configure boundary."""

    from npa.clients import huggingface
    from npa.clients.huggingface import HFAccessResult
    from npa.workbench.model_access import gated_hf_assets
    from npa.workbench.nurec.nurec import check_ngc_image_access

    if not hf_token:
        hf_summary = (
            "HF token missing; gated model(s) unverified"
        )
    else:
        try:
            identity = huggingface.validate_hf_identity(hf_token, timeout=2.0)
        except Exception:  # noqa: BLE001 - advisory probes never gate configure
            identity = HFAccessResult(repo="whoami-v2", ok=False)
        assets = gated_hf_assets()
        if identity.ok:
            cache = _probe_hf_assets_parallel(
                huggingface.validate_hf_access, hf_token, assets
            )
            denied = 0
            unverified = 0
            for asset in assets:
                result = cache.get(
                    (asset.repo, asset.repo_type, asset.revision, asset.probe_path)
                )
                if result is None or (
                    not result.ok and result.status_code not in {401, 403}
                ):
                    unverified += 1
                elif not result.ok:
                    denied += 1
            if denied:
                names = ", ".join(
                    asset.repo
                    for asset in assets
                    if (
                        (
                            result := cache.get(
                                (
                                    asset.repo,
                                    asset.repo_type,
                                    asset.revision,
                                    asset.probe_path,
                                )
                            )
                        )
                        is not None
                        and not result.ok
                        and result.status_code in {401, 403}
                    )
                )
                hf_summary = (
                    f"HF token valid; gated access missing for {denied} "
                    f"checked asset(s); HF has no access to: {names}. Accept gated "
                    "licenses at https://huggingface.co/<model>"
                )
            elif unverified:
                hf_summary = (
                    f"HF token valid; gated access unverified for {unverified} "
                    "checked asset(s)"
                )
            else:
                hf_summary = (
                    f"HF token valid; gated access confirmed for {len(assets)} "
                    "checked asset(s)"
                )
        else:
            # A successful exact repository probe still proves useful access if
            # the identity endpoint is independently unavailable. Conversely,
            # no completed repository probes means the provider/network path is
            # genuinely unverified. A definitive identity rejection wins.
            if identity.status_code in {401, 403}:
                hf_summary = "HF token rejected"
            else:
                cache = _probe_hf_assets_parallel(
                    huggingface.validate_hf_access, hf_token, assets
                )
                if cache and all(result.ok for result in cache.values()):
                    hf_summary = (
                        f"HF token valid; gated access confirmed for {len(cache)} "
                        "checked asset(s)"
                    )
                else:
                    hf_summary = (
                        "HF provider/network unavailable; token validity unverified"
                    )

    if not ngc_key:
        ngc_summary = "NGC key missing; NGC not configured (blocks nurec pulls)"
    else:
        try:
            ngc_outcome = check_ngc_image_access(ngc_key, timeout=2.0)
        except Exception:  # noqa: BLE001 - advisory probes never gate configure
            ngc_outcome = "unreachable"
        if ngc_outcome == "reachable":
            ngc_summary = "NGC key valid; repository access confirmed"
        elif ngc_outcome in {"auth-no-token", "auth-401", "auth-403"}:
            ngc_summary = "NGC key rejected"
        elif ngc_outcome in {
            "entitlement-required",
            "tags-401",
            "tags-403",
        }:
            ngc_summary = (
                "NGC entitlement denied; NGC repository entitlement denied for: "
                "nurec"
            )
        else:
            ngc_summary = "NGC provider/network unavailable; access unverified"

    return (
        "[NOTE] Access checks are informational and never block configuration: "
        f"{hf_summary}; {ngc_summary}. Use `npa workbench health access` when "
        "model access must be enforced."
    )


def _saved_model_access_note() -> str:
    """Return the advisory using persisted credentials without exposing values."""

    try:
        from npa.clients.credentials import load_credentials

        credentials = load_credentials(environ={})
        return _model_access_note(credentials.hf_token, credentials.ngc_api_key)
    except Exception:  # noqa: BLE001 - malformed/unreadable state stays non-gating
        return (
            "[NOTE] Access checks are informational; saved credentials could not "
            "be read. Use `npa workbench health access` after fixing the store."
        )


def _skipped_model_access_note() -> str:
    """Return the explicit provider-free advisory for ``--no-provision``."""

    return (
        "[NOTE] Access checks are informational; HF/NGC access probes skipped by "
        "--no-provision. Run `npa workbench health access` when model access must "
        "be enforced."
    )


def _prepare_full_catalog_access(*, open_pages: bool = False) -> dict[str, object]:
    """Audit the full HF/NGC catalog without making Workbench depend on it."""

    import webbrowser

    from npa.clients.credentials import load_credentials
    from npa.clients.huggingface import validate_hf_access
    from npa.workbench.access_approval import (
        DEFAULT_STATE_PATH,
        approval_plan,
        blocked,
        exact_requirements,
        probe_requirements,
    )
    from npa.workbench.nurec.nurec import check_ngc_image_access

    credentials = load_credentials()
    state_path = Path(
        os.environ.get("NPA_ACCESS_APPROVAL_STATE_PATH", str(DEFAULT_STATE_PATH))
    )
    evidence = probe_requirements(
        exact_requirements(),
        hf_token=credentials.hf_token,
        ngc_key=credentials.ngc_api_key,
        hf_validator=validate_hf_access,
        ngc_validator=check_ngc_image_access,
        state_path=state_path,
    )
    plan = approval_plan(
        evidence, resume_command="npa configure --prepare-catalog-access"
    )
    counts = plan["counts"]
    typer.echo(
        "Full Workbench catalog access audit: "
        f"{counts['hf']} Hugging Face resource(s), "
        f"{counts['ngc']} NVIDIA NGC artifact(s) need attention."
    )
    if blocked(plan) and not open_pages and sys.stdin.isatty():
        open_pages = typer.confirm(
            "Open the exact official HF/NGC approval pages now?",
            default=False,
        )
    if open_pages:
        for url in plan["official_urls"]:
            webbrowser.open_new_tab(str(url))
    for provider, rows in plan["providers"].items():
        typer.echo(f"{provider}:")
        for row in rows:
            typer.echo(
                f"  - {row['artifact']}@{row['revision'] or 'provider-current'}: "
                f"{row['status']} — {row['official_url']}"
            )
    if blocked(plan):
        typer.echo(
            "NPA did not accept any terms. Other Workbench capabilities remain "
            "usable. After completing any user-bound approval, resume with:"
        )
        typer.echo(f"  {plan['resume_command']}")
    else:
        typer.echo(
            "All exact gated HF/NGC artifacts have technical fetch access Ready; "
            "NPA performed no legal assent."
        )
    return plan


def _store_token_factory_key(api_key: str) -> None:
    from npa.clients.credentials import set_token_factory_api_key

    path = set_token_factory_api_key(api_key)
    typer.echo(
        f"Stored Nebius Token Factory API key in {path} under tokens.NEBIUS_TOKEN_FACTORY_KEY."
    )


def _store_tokens(hf_token: str, ngc_api_key: str) -> list[str]:
    """Persist non-interactive token flags; return what was written."""
    from npa.clients.credentials import write_credentials_file

    payload: dict[str, object] = {}
    written: list[str] = []
    if hf_token:
        payload["tokens"] = {"HF_TOKEN": hf_token}
        written.append("The Hugging Face token")
    if ngc_api_key:
        payload["ngc"] = {"api_key": ngc_api_key}
        written.append("The NGC API key")
    if not payload:
        return []
    path = write_credentials_file(payload)
    typer.echo(f"Stored {' and '.join(written).lower()} in {path}.")
    return written


def _configured_env_lines() -> str:
    """Return the saved ~/.npa values as ``NPA_*=`` shell assignments.

    Lets a runbook stop asking the operator to hand-substitute the alias, bucket
    and kube context: ``eval "$(npa configure --show --env)"``. Secrets are not
    included; workflow submit resolves requested ``--secret-env`` names directly
    from credentials.yaml without printing them.
    """
    from npa.clients.config import default_project_name, list_projects
    from npa.clients.credentials import load_credentials

    try:
        projects = list_projects()
        alias = default_project_name()
    except Exception:  # noqa: BLE001 - emit whatever is readable
        projects, alias = {}, ""
    lines: list[str] = []
    if projects:
        resolved = alias if alias in projects else next(iter(projects))
        stanza = projects.get(resolved) or {}
        lines.append(f"NPA_PROJECT_ALIAS={shlex.quote(resolved)}")
        for name, key in (
            ("NPA_PROJECT_ID", "project_id"),
            ("NPA_TENANT_ID", "tenant_id"),
            ("NPA_REGION", "region"),
        ):
            value = str((stanza or {}).get(key, "") or "")
            if value:
                lines.append(f"{name}={shlex.quote(value)}")
    try:
        credentials = load_credentials(environ={})
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
    context = _saved_kube_context()
    if context:
        lines.append(f"NPA_KUBE_CONTEXT={shlex.quote(context)}")
    return "\n".join(lines)


def _saved_kube_context() -> str:
    """Return the most recently saved local cluster context, or "".

    Scoped to the configured project when its ``project_id`` is resolvable, so
    ``configure --show --env`` never emits an unrelated project's cluster context
    (which would point a runbook / PAIDF submit at the wrong infra). Falls back to
    the most recent cluster of any project when no scoped match exists.
    """
    try:
        from npa.cluster.state import list_local_clusters

        clusters = list(list_local_clusters() or [])
    except Exception:  # noqa: BLE001 - no cluster cache is normal before provisioning
        return ""
    configured_pid = _configured_project_id()
    scoped = [
        state
        for state in clusters
        if configured_pid
        and str(getattr(state, "project_id", "") or "") == configured_pid
    ]
    for state in reversed(scoped or clusters):
        # `cluster up` names the kubeconfig context after the cluster by default.
        context = str(getattr(state, "name", "") or "")
        if context:
            return context
    return ""


def _configured_project_id() -> str:
    """Return the configured default project's Nebius project id, or ""."""
    try:
        from npa.clients.config import default_project_name, list_projects

        projects = list_projects() or {}
        if not projects:
            return ""
        alias = default_project_name()
        stanza = projects.get(alias) or next(iter(projects.values()))
        return str((stanza or {}).get("project_id", "") or "")
    except Exception:  # noqa: BLE001 - unreadable config scopes nothing
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
    if not projects:
        lines.append(f"  (no projects in {CONFIG_PATH} — run `npa configure`)")
        return "\n".join(lines)

    stanza = projects.get(alias) or next(iter(projects.values()))
    resolved_alias = alias if alias in projects else next(iter(projects))
    lines.append(f"  config file:        {CONFIG_PATH}")
    lines.append(f"  project alias:      {resolved_alias}  (use with -p)")
    if len(projects) > 1:
        lines.append(
            f"  other aliases:      {', '.join(a for a in projects if a != resolved_alias)}"
        )
    for label, key in (
        ("project id", "project_id"),
        ("tenant id", "tenant_id"),
        ("region", "region"),
    ):
        value = str((stanza or {}).get(key, "") or "")
        lines.append(f"  {label + ':':<19} {value or '(unset)'}")

    try:
        credentials = load_credentials(environ={})
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
    bucket_storage_class: str = "",
    bucket_size_gb: str = "",
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
    typer.echo(
        "Intent: "
        + (
            "provision object storage explicitly"
            if provision
            else "save project configuration only (no provider calls)"
        )
        + "."
    )
    alias = values["--project-alias"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", alias):
        raise typer.BadParameter(
            "--project-alias must start with a letter or digit and contain only "
            "letters, digits, '.', '_', or '-' (maximum 64 characters)"
        )
    requested_storage_class = str(bucket_storage_class or "").strip()
    requested_size_gb = str(bucket_size_gb or "").strip()
    if (requested_storage_class or requested_size_gb) and not provision:
        raise typer.BadParameter(
            "--bucket-storage-class and --bucket-size-gb require --provision"
        )
    if requested_storage_class:
        normalized_label = (
            requested_storage_class.lower().replace("-", "_").replace(" ", "_")
        )
        if normalized_label not in {
            "standard",
            "std",
            "storage_class_unspecified",
            "enhanced",
            "enhanced_throughput",
            "enhancedthroughput",
            "intelligent",
        }:
            raise typer.BadParameter(
                "--bucket-storage-class must be standard, enhanced, or intelligent"
            )
        requested_storage_class = nebius_client.normalize_bucket_storage_class(
            requested_storage_class
        )
    requested_max_size_bytes: int | None = None
    if requested_size_gb:
        from decimal import Decimal, InvalidOperation

        try:
            parsed_size = Decimal(requested_size_gb)
        except InvalidOperation as exc:
            raise typer.BadParameter(
                "--bucket-size-gb must be a finite non-negative number"
            ) from exc
        if not parsed_size.is_finite() or parsed_size < 0:
            raise typer.BadParameter(
                "--bucket-size-gb must be a finite non-negative number"
            )
        requested_max_size_bytes = _gb_to_bytes(requested_size_gb)
        if parsed_size > 0 and requested_max_size_bytes == 0:
            raise typer.BadParameter(
                "--bucket-size-gb must resolve to at least one byte, or be 0 for unlimited"
            )

    existing_projects = list_projects()
    _validate_alias_project_identity(
        alias, existing_projects.get(alias) or {}, values["--project-id"]
    )
    # Project-only setup is deliberately local. Provider authentication, bucket
    # probes, and profile rebinding occur only behind the explicit --provision
    # intent.
    from npa.clients.project_credential_store import (
        project_credential_record,
        select_project_credentials,
    )

    if provision:
        try:
            nebius_client.get_iam_token()
        except nebius_client.NebiusError as exc:
            raise typer.BadParameter(
                "The active Nebius CLI profile cannot authenticate "
                "non-interactively. Activate or refresh it, then retry."
            ) from exc
        existing_record = project_credential_record(values["--project-id"])
    else:
        existing_record = {}
        select_project_credentials(
            values["--project-id"],
            alias=alias,
            select_storage=False,
        )

    existing_storage = existing_record.get("storage")
    existing_storage = (
        dict(existing_storage) if isinstance(existing_storage, dict) else {}
    )
    storage: dict[str, str] = {}
    existing_complete = bool(
        existing_storage.get("aws_access_key_id")
        and existing_storage.get("aws_secret_access_key")
        and existing_storage.get("bucket")
    )
    existing_relationship_verified = bool(existing_complete and existing_record)
    if (
        provision
        and existing_complete
        and existing_relationship_verified
        and not (requested_storage_class or requested_size_gb)
    ):
        from npa.clients.storage_validation import probe_storage_write

        probe = probe_storage_write(
            bucket=str(existing_storage.get("bucket") or ""),
            endpoint_url=str(existing_storage.get("endpoint_url") or "")
            or _endpoint_for_region(values["--region"]),
            access_key_id=str(existing_storage.get("aws_access_key_id") or ""),
            secret_access_key=str(
                existing_storage.get("aws_secret_access_key") or ""
            ),
            region=values["--region"],
        )
        if probe.ok:
            storage = {
                "aws_access_key_id": str(
                    existing_storage.get("aws_access_key_id") or ""
                ),
                "aws_secret_access_key": str(
                    existing_storage.get("aws_secret_access_key") or ""
                ),
                "endpoint_url": str(existing_storage.get("endpoint_url") or "")
                or _endpoint_for_region(values["--region"]),
                "bucket": str(existing_storage.get("bucket") or ""),
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
                _bucket_name_from_uri(str(existing_storage.get("bucket") or ""))
                if existing_relationship_verified
                else ""
            ),
            interactive=False,
            bucket_storage_class=requested_storage_class,
            bucket_max_size_bytes=requested_max_size_bytes,
        )
        if provisioned is None or provisioned.get("_validated") != "true":
            raise typer.BadParameter(
                "Non-interactive object-storage provisioning did not produce "
                "health-verified credentials. Fix the reported IAM/storage error "
                "and retry; no secret should be passed on the command line."
            )
        storage = provisioned

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
        from npa.clients.project_credential_store import write_project_credentials

        credentials_path = write_project_credentials(
            values["--project-id"],
            credentials_payload,
            alias=alias,
        )
        typer.echo(f"Wrote {credentials_path} (chmod 600).")

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
    storage_disposition = str(storage.get("_disposition") or "configured") if storage else "not selected"
    typer.echo(
        f"Summary: mode={'provision' if provision else 'project-only'}; "
        f"project={alias}; storage={storage_disposition}; configuration=saved."
    )
    typer.echo(
        _saved_model_access_note() if provision else _skipped_model_access_note()
    )


class ConfigureMode(str, Enum):
    DISPLAY = "display"
    GUIDANCE = "guidance"
    SOURCE_URI = "source-uri"
    FORGET_PROJECT = "forget-project"
    IMPORT_CREDENTIALS = "import-credentials"
    KNOWN_PROJECT = "known-project"
    INTERACTIVE = "interactive"
    CATALOG_ACCESS = "catalog-access"


def _configure_mode(arguments: dict[str, Any]) -> ConfigureMode:
    """Resolve one configure intent and reject ambiguous combinations."""

    show = bool(arguments.get("show"))
    env_output = bool(arguments.get("env_output"))
    source_uri = str(arguments.get("src_s3_uri") or "").strip()
    forget = str(arguments.get("forget_project") or "").strip()
    save_env = bool(arguments.get("save_env_credentials"))
    prepare_catalog_access = bool(arguments.get("prepare_catalog_access"))
    open_approval_pages = bool(arguments.get("open_approval_pages"))
    interactive = arguments.get("interactive")
    provision = arguments.get("provision")
    known_names = ("tenant_id", "project_id", "region", "project_alias")
    bucket_names = ("bucket_storage_class", "bucket_size_gb")
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
    if prepare_catalog_access or open_approval_pages:
        requested.append(ConfigureMode.CATALOG_ACCESS)
    if known or any(str(arguments.get(name) or "").strip() for name in bucket_names):
        requested.append(ConfigureMode.KNOWN_PROJECT)
    if requested == [ConfigureMode.DISPLAY, ConfigureMode.IMPORT_CREDENTIALS]:
        requested = [ConfigureMode.IMPORT_CREDENTIALS]
    if requested == [ConfigureMode.IMPORT_CREDENTIALS, ConfigureMode.KNOWN_PROJECT]:
        # One deterministic prompt-free transaction: import supported environment
        # credentials, then save the explicit project (and provision only when the
        # caller also passed --provision).
        requested = [ConfigureMode.KNOWN_PROJECT]
    if len(requested) > 1:
        raise typer.BadParameter(
            "Configure modes cannot be combined: choose exactly one of display "
            "(--show/--env), --src-s3-uri, --forget-project, "
            "--save-env-credentials, --prepare-catalog-access/"
            "--open-approval-pages, or known-project setup."
        )

    mode = (
        requested[0]
        if requested
        else (
            ConfigureMode.INTERACTIVE
            if interactive is True or (interactive is None and sys.stdin.isatty())
            else ConfigureMode.GUIDANCE
        )
    )
    if mode == ConfigureMode.DISPLAY and (
        interactive is True or provision is not None
    ):
        raise typer.BadParameter(
            f"{mode.value} mode is read-only or self-contained and cannot be "
            "combined with interactive/provisioning options."
        )
    if mode in {
        ConfigureMode.SOURCE_URI,
        ConfigureMode.FORGET_PROJECT,
        ConfigureMode.CATALOG_ACCESS,
    } and (interactive is not None or provision is not None):
        raise typer.BadParameter(
            f"{mode.value} mode is read-only or self-contained and cannot be "
            "combined with interactive/provisioning options."
        )
    if mode == ConfigureMode.IMPORT_CREDENTIALS and (
        interactive is True or provision is not None
    ):
        raise typer.BadParameter(
            "--save-env-credentials is a prompt-free import mode and cannot be "
            "combined with interactive/provisioning options."
        )
    if mode == ConfigureMode.KNOWN_PROJECT:
        missing = [name for name in known_names if name not in known]
        if missing:
            flags = ", ".join("--" + name.replace("_", "-") for name in missing)
            raise typer.BadParameter(
                "Known-project configure requires --tenant-id, --project-id, "
                f"--region, and --project-alias; missing {flags}"
            )
        if interactive is True:
            raise typer.BadParameter(
                "Known-project flags select prompt-free setup; omit --interactive "
                "or use --no-interactive."
            )
        if any(str(arguments.get(name) or "").strip() for name in bucket_names) and provision is not True:
            raise typer.BadParameter(
                "--bucket-storage-class and --bucket-size-gb require explicit "
                "--provision"
            )
    if mode == ConfigureMode.GUIDANCE and provision is True:
        raise typer.BadParameter(
            "--provision requires either interactive setup or all known-project "
            "flags."
        )
    return mode


def _configure_impl(
    *,
    show: bool,
    interactive: Optional[bool],
    provision: Optional[bool] = None,
    save_env_credentials: bool = False,
    env_output: bool = False,
    forget_project: str = "",
    src_s3_uri: str = "",
    tenant_id: str = "",
    project_id: str = "",
    region: str = "",
    project_alias: str = "",
    bucket_storage_class: str = "",
    bucket_size_gb: str = "",
    prepare_catalog_access: bool = False,
    open_approval_pages: bool = False,
) -> None:
    mode = _configure_mode(locals())
    effective_provision = (
        bool(provision)
        if provision is not None
        else mode == ConfigureMode.INTERACTIVE
    )
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
    if mode == ConfigureMode.CATALOG_ACCESS:
        _prepare_full_catalog_access(open_pages=open_approval_pages)
        return
    from npa.clients.credentials import (
        CREDENTIALS_PATH,
        SUPPORTED_ENV_CREDENTIALS,
        persist_supported_env_credentials,
    )

    def persist_environment_credentials(*, diagnostics_to_stderr: bool = False) -> None:
        report = persist_supported_env_credentials()
        typer.echo(
            "Credential environment sources detected: "
            + (", ".join(report["detected"]) if report["detected"] else "none"),
            err=diagnostics_to_stderr,
        )
        typer.echo(
            "Credential fields persisted (values redacted): "
            + (", ".join(report["persisted"]) if report["persisted"] else "none")
            + f"; store={CREDENTIALS_PATH}",
            err=diagnostics_to_stderr,
        )
        typer.echo(
            f"Environment credentials (values redacted) saved to {CREDENTIALS_PATH}.",
            err=diagnostics_to_stderr,
        )
        for warning in report["warnings"]:
            typer.echo(f"Credential warning: {warning}", err=True)

    if mode == ConfigureMode.DISPLAY:
        detected = [
            name for name in SUPPORTED_ENV_CREDENTIALS if os.environ.get(name)
        ]
        if detected:
            typer.echo(
                "Credential environment sources detected: " + ", ".join(detected),
                err=env_output,
            )
            typer.echo(
                "Credential persistence: skipped in display mode; use "
                "--save-env-credentials separately to persist values.",
                err=env_output,
            )
        if env_output:
            typer.echo(_configured_env_lines())
        else:
            typer.echo(_configured_summary())
            typer.echo("")
            typer.echo(_SETUP_GUIDANCE)
        return
    if mode == ConfigureMode.IMPORT_CREDENTIALS:
        machine_output = env_output
        persist_environment_credentials(diagnostics_to_stderr=machine_output)
        typer.echo(_saved_model_access_note(), err=machine_output)
        if show or env_output:
            typer.echo(_configured_env_lines() if env_output else _configured_summary())
        return
    if mode == ConfigureMode.KNOWN_PROJECT:
        if save_env_credentials:
            persist_environment_credentials()
        _run_known_project_configure(
            tenant_id=tenant_id,
            project_id=project_id,
            region=region,
            project_alias=project_alias,
            provision=effective_provision,
            bucket_storage_class=bucket_storage_class,
            bucket_size_gb=bucket_size_gb,
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
    if mode == ConfigureMode.GUIDANCE:
        typer.echo(_SETUP_GUIDANCE)
        return
    try:
        _run_interactive_configure(
            provision=effective_provision,
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
        if mode in {
            ConfigureMode.DISPLAY,
            ConfigureMode.GUIDANCE,
            ConfigureMode.CATALOG_ACCESS,
        }:
            return function(*args, **kwargs)
        forget = str(bound.arguments.get("forget_project") or "").strip()
        alias = str(bound.arguments.get("project_alias") or "").strip()
        project_id = str(bound.arguments.get("project_id") or "").strip()
        tenant_id = str(bound.arguments.get("tenant_id") or "").strip()
        region = str(bound.arguments.get("region") or "").strip()
        if forget:
            from npa.clients.config import resolve_environment

            environment = resolve_environment(forget)
            alias = forget
            if environment is not None:
                project_id = str(environment.project_id or "")
                tenant_id = str(environment.tenant_id or "")
                region = str(environment.region or "")
        requested_name = alias or project_id or "interactive"
        resume = (
            f"npa configure --forget-project {forget}" if forget else "npa configure"
        )
        if not forget and all((tenant_id, project_id, region, alias)):
            resume += (
                f" --no-interactive --tenant-id {tenant_id} --project-id {project_id}"
                f" --region {region} --project-alias {alias}"
            )
            if bound.arguments.get("provision") is True:
                resume += " --provision"
            elif bound.arguments.get("provision") is False:
                resume += " --no-provision"
        operation = ProvisioningOperation.prepare(
            command=("npa configure --forget-project" if forget else "npa configure"),
            project_alias=alias,
            project_id=project_id,
            tenant_id=tenant_id,
            region=region,
            resource_type=("forget-project" if forget else "configure"),
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
                    ["default_project", f"projects.{alias}"]
                    if alias
                    else [mode.value]
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
            try:
                intent = (
                    OperationIntent.DESTROY
                    if forget
                    else OperationIntent.ENSURE_PRESENT
                )
                from npa.lifecycle_intent import operation_intent

                with operation_intent(intent):
                    result = function(*args, **kwargs)
            except BaseException as exc:
                phase = str(operation.read().get("phase") or "")
                if phase not in {
                    "recovery-required",
                    "rolled-back",
                    "rollback-incomplete",
                }:
                    operation.transition("recovery-required", error=str(exc))
                typer.echo(emit_recovery_summary(operation), err=True)
                raise
            phase = str(operation.read().get("phase") or "")
            if phase in {"mutating", "resource-created"}:
                operation.transition(
                    "state-durable", details={"private_stores": "fsynced"}
                )
            operation.commit()
            return result

    return wrapped


@app.command(
    "configure",
    help="Interactive credential and config setup guidance.",
    rich_help_panel="Setup",
)
@_transactional_configure
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
    provision: Optional[bool] = typer.Option(
        None,
        "--provision/--no-provision",
        help=(
            "Explicitly create or reuse Nebius object storage. Interactive setup "
            "offers provisioning by default; prompt-free known-project setup is "
            "project-only unless --provision is passed. --no-provision performs "
            "no provider calls or storage adoption."
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
    bucket_storage_class: str = typer.Option(
        "",
        "--bucket-storage-class",
        help="Storage class for a newly created known-project bucket: standard, enhanced, or intelligent.",
    ),
    bucket_size_gb: str = typer.Option(
        "",
        "--bucket-size-gb",
        help="GiB cap for a newly created known-project bucket; 0 means unlimited.",
    ),
    prepare_catalog_access: bool = typer.Option(
        False,
        "--prepare-catalog-access",
        help=(
            "Audit every exact gated HF/NGC artifact in the current Workbench "
            "catalog without changing project or storage configuration."
        ),
    ),
    open_approval_pages: bool = typer.Option(
        False,
        "--open-approval-pages",
        help=(
            "Affirmatively open missing official HF/NGC pages during the catalog "
            "audit; NPA never submits acceptance."
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
        bucket_storage_class=bucket_storage_class,
        bucket_size_gb=bucket_size_gb,
        prepare_catalog_access=prepare_catalog_access,
        open_approval_pages=open_approval_pages,
    )


@app.command(
    "init",
    help="Interactive credential and config setup guidance.",
    rich_help_panel="Setup",
)
@_transactional_configure
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
    provision: Optional[bool] = typer.Option(
        None,
        "--provision/--no-provision",
        help=(
            "Explicitly create or reuse Nebius object storage. Interactive setup "
            "offers provisioning by default; prompt-free known-project setup is "
            "project-only unless --provision is passed. --no-provision performs "
            "no provider calls or storage adoption."
        ),
    ),
    save_env_credentials: bool = typer.Option(
        False,
        "--save-env-credentials",
        help="Persist supported environment credentials atomically with mode 0600.",
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
        "", "--tenant-id", help="Known Nebius tenant ID for prompt-free configure."
    ),
    project_id: str = typer.Option(
        "", "--project-id", help="Known Nebius project ID for prompt-free configure."
    ),
    region: str = typer.Option(
        "", "--region", help="Known Nebius project region for prompt-free configure."
    ),
    project_alias: str = typer.Option(
        "", "--project-alias", help="Local NPA alias for prompt-free configure."
    ),
    bucket_storage_class: str = typer.Option(
        "",
        "--bucket-storage-class",
        help="Storage class for a newly created known-project bucket: standard, enhanced, or intelligent.",
    ),
    bucket_size_gb: str = typer.Option(
        "",
        "--bucket-size-gb",
        help="GiB cap for a newly created known-project bucket; 0 means unlimited.",
    ),
    prepare_catalog_access: bool = typer.Option(
        False, "--prepare-catalog-access", help="Audit full HF/NGC catalog access."
    ),
    open_approval_pages: bool = typer.Option(
        False,
        "--open-approval-pages",
        help="Affirmatively open missing official HF/NGC approval pages.",
    ),
) -> None:
    """Interactively write ~/.npa credentials and config, or show guidance."""
    _configure_impl(
        show=show,
        interactive=interactive,
        provision=provision,
        save_env_credentials=save_env_credentials,
        env_output=env_output,
        src_s3_uri=src_s3_uri,
        tenant_id=tenant_id,
        project_id=project_id,
        region=region,
        project_alias=project_alias,
        bucket_storage_class=bucket_storage_class,
        bucket_size_gb=bucket_size_gb,
        prepare_catalog_access=prepare_catalog_access,
        open_approval_pages=open_approval_pages,
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
