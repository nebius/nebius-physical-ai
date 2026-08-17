"""Prerequisite checks for ``npa agent`` lifecycle commands.

The checks run before any cloud mutation.  Public names support focused callers;
underscore aliases preserve the established ``npa.cli.agent`` compatibility API.
"""

from __future__ import annotations

import base64
import binascii
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

if TYPE_CHECKING:  # pragma: no cover - type-checker visibility only
    from npa.workflows.sim2real_health import CheckResult


_OPENSSH_PUBLIC_KEY_PREFIXES = (
    "ssh-",
    "ecdsa-sha2-",
    "sk-ssh-",
    "sk-ecdsa-",
)


def _is_openssh_public_key(path: Path) -> bool:
    """Return whether *path* is one bounded OpenSSH public-key line.

    Merely checking that the path exists lets an operator accidentally pass the
    adjacent private key. Terraform would then interpolate a multiline secret
    into cloud-init user-data before the provider rejects the malformed YAML.
    """

    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return False
    if len(raw.encode("utf-8")) > 16 * 1024:
        return False
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) != 1:
        return False
    fields = lines[0].split()
    if len(fields) < 2 or not fields[0].startswith(_OPENSSH_PUBLIC_KEY_PREFIXES):
        return False
    try:
        decoded = base64.b64decode(fields[1], validate=True)
    except (binascii.Error, ValueError):
        return False
    return bool(decoded)


def _terraform_binary() -> str:
    return (
        os.environ.get("NPA_TERRAFORM_BIN") or shutil.which("terraform") or ""
    ).strip()


def agent_hard_prereq_results(
    ssh_public_key_path: str, *, terraform_bin: str | None = None
) -> list[CheckResult]:
    """Check Terraform, its provider lock, and the SSH key pair."""
    from npa.workflows.sim2real_health import CheckResult, FAIL, PASS

    results: list[Any] = []
    terraform = (
        str(terraform_bin).strip()
        if terraform_bin is not None
        else _terraform_binary()
    )
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

    public_path = Path(ssh_public_key_path).expanduser()
    private_path = Path(
        str(public_path)[:-4]
        if str(public_path).endswith(".pub")
        else str(public_path)
    )
    if public_path.is_file() and _is_openssh_public_key(public_path):
        results.append(
            CheckResult(
                name="ssh_public_key",
                status=PASS,
                summary=f"SSH public key present ({public_path}).",
            )
        )
    else:
        present_but_invalid = public_path.is_file()
        results.append(
            CheckResult(
                name="ssh_public_key",
                status=FAIL,
                summary=(
                    f"SSH public key is not a single OpenSSH public-key line: {public_path}"
                    if present_but_invalid
                    else f"SSH public key not found: {public_path}"
                ),
                remedy=(
                    f"Generate a keypair (`ssh-keygen -t ed25519 -f {private_path}`) "
                    "or pass --ssh-public-key-path to the public `.pub` file; never "
                    "pass a private key."
                ),
            )
        )

    if private_path.is_file():
        results.append(
            CheckResult(
                name="ssh_private_key",
                status=PASS,
                summary=f"SSH private key present ({private_path}).",
            )
        )
    else:
        results.append(
            CheckResult(
                name="ssh_private_key",
                status=FAIL,
                summary=f"SSH private key not found: {private_path}",
                remedy=(
                    "The private key next to the public key is required to "
                    "bootstrap the VM over SSH."
                ),
            )
        )
    return results


def _agent_hard_prereq_results(ssh_public_key_path: str) -> list[CheckResult]:
    return agent_hard_prereq_results(ssh_public_key_path)


def agent_nebius_auth_result() -> CheckResult:
    """Require a live Nebius CLI identity before deployment."""
    from npa.workflows.sim2real_health import CheckResult, FAIL, PASS

    try:
        from npa.clients.nebius import get_iam_token

        token = get_iam_token()
    except Exception as exc:  # noqa: BLE001 - every auth/CLI error means not ready
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


_agent_nebius_auth_result = agent_nebius_auth_result


def agent_token_factory_result(tf_key: str | None = None) -> CheckResult:
    """Report Token Factory availability without making a network request."""
    from npa.clients.credentials import load_credentials
    from npa.workflows.sim2real_health import CheckResult, PASS, WARN

    resolved_key = tf_key
    if resolved_key is None:
        resolved_key = load_credentials().token_factory_api_key
    if resolved_key:
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
            "`npa configure --token-factory-key <key>`, then re-run `npa agent bootstrap`."
        ),
    )


_agent_token_factory_result = agent_token_factory_result


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

    detail: Exception
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
        detail = exc
    except Exception as exc:  # noqa: BLE001 - provider failure means not ready
        detail = exc
    else:
        return CheckResult(
            name="writable_s3",
            status=PASS,
            summary=(
                "Deployment credential path selected and verified the exact "
                f"Terraform backend object contract ({credentials['s3_bucket']})."
            ),
        )
    return CheckResult(
        name="writable_s3",
        status=FAIL,
        summary="Writable S3 configuration cannot be resolved for agent deploy.",
        remedy=(
            "Run `npa provision-if-absent --project <alias> --skip-k8s` "
            "to reconcile storage before deploying the agent."
        ),
        details=(str(detail),),
    )


def render_agent_checks(
    results: list[CheckResult], *, output_json: bool
) -> tuple[str, bool]:
    """Return the shared rendered report and whether it contains a failure."""
    from npa.workflows.sim2real_health import format_check_report, has_failure

    return format_check_report(results, output_json=output_json), has_failure(results)


def _render_agent_checks(results: list[CheckResult], *, output_json: bool) -> bool:
    report, failed = render_agent_checks(results, output_json=output_json)
    typer.echo(report)
    return failed
