"""Cheap, side-effect-free prerequisite checks for ``npa agent``.

Extracted from the ``npa.cli.agent`` monolith (kept under a size ratchet). These
run before any cloud IAM side effect or Terraform apply, so a missing binary, key
or credential surfaces up front instead of mid-run — after which a failure would
auto-roll-back a freshly provisioned VM. Re-imported into ``npa.cli.agent``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

if TYPE_CHECKING:  # pragma: no cover - type-checker visibility only
    from npa.workflows.sim2real_health import CheckResult


def _terraform_binary() -> str:
    """Return the configured Terraform binary without importing the CLI monolith."""
    return (
        os.environ.get("NPA_TERRAFORM_BIN") or shutil.which("terraform") or ""
    ).strip()


def _agent_hard_prereq_results(ssh_public_key_path: str) -> list[CheckResult]:
    """Cheap, side-effect-free Route C prerequisites (terraform + SSH keys).

    These are checked before any cloud IAM side effects or Terraform apply so a
    missing binary or key surfaces up front instead of mid-run (after which a
    transient SSH failure would auto-roll-back a freshly provisioned VM).
    """
    from npa.workflows.sim2real_health import CheckResult, FAIL, PASS

    results: list[Any] = []

    terraform = _terraform_binary()
    if terraform:
        from npa.deploy import provisioner
        from npa.terraform_lock import TerraformLockError, validate_provider_lock

        terraform_dir = Path(provisioner.__file__).parent / "terraform"
        try:
            target_platform = validate_provider_lock(terraform_dir)
        except TerraformLockError as exc:
            results.append(
                CheckResult(
                    name="terraform",
                    status=FAIL,
                    summary=(
                        "Terraform is installed, but provider-lock compatibility "
                        "failed before provisioning."
                    ),
                    remedy=str(exc),
                )
            )
        else:
            results.append(
                CheckResult(
                    name="terraform",
                    status=PASS,
                    summary=(
                        f"terraform found ({terraform}); provider lock covers "
                        f"{target_platform}."
                    ),
                )
            )
    else:
        results.append(
            CheckResult(
                name="terraform",
                status=FAIL,
                summary="terraform binary not found on PATH.",
                remedy="Install it: https://developer.hashicorp.com/terraform/install",
            )
        )

    pub_path = Path(ssh_public_key_path).expanduser()
    if pub_path.is_file():
        results.append(
            CheckResult(
                name="ssh_public_key",
                status=PASS,
                summary=f"SSH public key present ({pub_path}).",
            )
        )
    else:
        priv_hint = (
            str(pub_path)[:-4] if str(pub_path).endswith(".pub") else str(pub_path)
        )
        results.append(
            CheckResult(
                name="ssh_public_key",
                status=FAIL,
                summary=f"SSH public key not found: {pub_path}",
                remedy=(
                    f"Generate a keypair (`ssh-keygen -t ed25519 -f {priv_hint}`) "
                    "or pass --ssh-public-key-path to an existing key."
                ),
            )
        )

    # The deploy flow uses the private key alongside the public key (pub path
    # minus the .pub suffix) to bootstrap the VM over SSH. If --ssh-public-key-path
    # is given without a .pub suffix, this resolves to the same path as the public
    # key check above, which at worst yields a slightly redundant message.
    priv_str = str(pub_path)[:-4] if str(pub_path).endswith(".pub") else str(pub_path)
    priv_path = Path(priv_str)
    if priv_path.is_file():
        results.append(
            CheckResult(
                name="ssh_private_key",
                status=PASS,
                summary=f"SSH private key present ({priv_path}).",
            )
        )
    else:
        results.append(
            CheckResult(
                name="ssh_private_key",
                status=FAIL,
                summary=f"SSH private key not found: {priv_path}",
                remedy="The private key next to the public key is required to bootstrap the VM over SSH.",
            )
        )

    return results


def _agent_nebius_auth_result() -> CheckResult:
    """Live Nebius auth check (FAIL): deploy needs an authenticated profile to provision."""
    from npa.workflows.sim2real_health import CheckResult, FAIL, PASS

    try:
        from npa.clients.nebius import get_iam_token

        token = get_iam_token()
    except Exception as exc:  # noqa: BLE001 - any auth/CLI error means "not ready"
        return CheckResult(
            name="nebius_profile",
            status=FAIL,
            summary="No authenticated Nebius CLI profile.",
            remedy="Install/authenticate the Nebius CLI and run `npa configure`.",
            details=(str(exc),),
        )
    if token:
        return CheckResult(
            name="nebius_profile",
            status=PASS,
            summary="Nebius CLI profile is authenticated.",
        )
    return CheckResult(
        name="nebius_profile",
        status=FAIL,
        summary="Nebius IAM token unavailable.",
        remedy="Run `npa configure` / `nebius profile create` to authenticate.",
    )


def _agent_token_factory_result(tf_key: str | None = None) -> CheckResult:
    """Token Factory key check (WARN): the headline chat feature needs it.

    Pass a pre-resolved ``tf_key`` to avoid re-reading credentials when the
    caller already has them.
    """
    from npa.cli.agent import _resolve_deploy_llm_credentials
    from npa.workflows.sim2real_health import CheckResult, PASS, WARN

    if tf_key is None:
        tf_key, _ = _resolve_deploy_llm_credentials()
    if tf_key:
        return CheckResult(
            name="token_factory",
            status=PASS,
            summary="Token Factory API key is configured.",
        )
    return CheckResult(
        name="token_factory",
        status=WARN,
        summary="Token Factory API key not found; agent chat will return 503 until it is set.",
        remedy=(
            "Get a key (starts with 'v1.') at https://tokenfactory.nebius.com/ and run "
            "export NEBIUS_TOKEN_FACTORY_KEY, run `npa configure --no-interactive "
            "--save-env-credentials`, then re-run `npa agent bootstrap`."
        ),
    )


def _agent_storage_result(
    project: str = "", region: str = "", name: str = "default"
) -> CheckResult:
    """Exercise the exact writable-storage decision used by agent deployment."""

    from npa.cli.agent import (
        AgentStorageCredentialError,
        _resolve_deploy_storage_credentials,
    )
    from npa.clients.config import resolve_environment
    from npa.workflows.sim2real_health import CheckResult, FAIL, PASS

    try:
        environment = resolve_environment(project or None)
        resolved_region = str(
            region or getattr(environment, "region", "") or ""
        ).strip()
        credentials = _resolve_deploy_storage_credentials(
            region=resolved_region,
            project_alias=project,
            emit_status=False,
        )
        from npa.clients.storage_validation import (
            probe_terraform_backend,
            terraform_state_key,
        )

        exact_values = (
            str(credentials.get("s3_bucket", "")),
            str(credentials.get("s3_endpoint", "")),
            str(credentials.get("nebius_api_key", "")),
            str(credentials.get("nebius_secret_key", "")),
        )
        # Production resolution is all-or-error. The conditional preserves the
        # small mapping contract used by injected callers while never treating a
        # partially resolved real configuration as deployable.
        if all(exact_values):
            probe = probe_terraform_backend(
                bucket=exact_values[0],
                state_key=terraform_state_key(project, name),
                endpoint_url=exact_values[1],
                access_key_id=exact_values[2],
                secret_access_key=exact_values[3],
                region=resolved_region,
            )
            if not probe.ok:
                raise AgentStorageCredentialError(probe.summary)
    except AgentStorageCredentialError as exc:
        return CheckResult(
            name="writable_s3",
            status=FAIL,
            summary="Writable S3 configuration cannot be resolved for agent deploy.",
            remedy=(
                "Run `npa provision-if-absent --project <alias> --skip-k8s` "
                "to reconcile storage before deploying the agent."
            ),
            details=(str(exc),),
        )
    except Exception as exc:  # noqa: BLE001 - any provider probe failure is not ready
        return CheckResult(
            name="writable_s3",
            status=FAIL,
            summary="Writable S3 configuration cannot be resolved for agent deploy.",
            remedy=(
                "Run `npa provision-if-absent --project <alias> --skip-k8s` "
                "to reconcile storage before deploying the agent."
            ),
            details=(str(exc),),
        )
    return CheckResult(
        name="writable_s3",
        status=PASS,
        summary=(
            "Deployment credential path selected and verified the exact Terraform backend object "
            f"contract ({credentials['s3_bucket']})."
        ),
    )


def _render_agent_checks(results: list[CheckResult], *, output_json: bool) -> bool:
    """Render agent preflight CheckResults; return True when any FAIL is present.

    Uses the shared report renderer so agent and workbench-health preflight
    output stay aligned.
    """
    from npa.workflows.sim2real_health import format_check_report, has_failure

    typer.echo(format_check_report(results, output_json=output_json))
    return has_failure(results)
