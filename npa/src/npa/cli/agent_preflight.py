"""Prerequisite checks for ``npa agent`` lifecycle commands.

The checks run before any cloud mutation.  Public names support focused callers;
underscore aliases preserve the established ``npa.cli.agent`` compatibility API.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
import yaml

if TYPE_CHECKING:  # pragma: no cover - type-checker visibility only
    from npa.workflows.sim2real_health import CheckResult


_OPENSSH_PUBLIC_KEY_PREFIXES = (
    "ssh-",
    "ecdsa-sha2-",
    "sk-ssh-",
    "sk-ecdsa-",
)
_AGENT_CLOUD_INIT_BRANCH = '%{ if workbench_type != "agent" ~}'


def _openssh_public_key_line(path: Path) -> str:
    """Read and normalize exactly one bounded OpenSSH public-key record."""

    raw = path.read_text(encoding="utf-8")
    if len(raw.encode("utf-8")) > 16 * 1024:
        raise ValueError("SSH public key exceeds 16 KiB")
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("SSH public key must contain exactly one non-empty line")
    fields = lines[0].split()
    if len(fields) < 2 or not fields[0].startswith(_OPENSSH_PUBLIC_KEY_PREFIXES):
        raise ValueError("SSH public key does not use a supported OpenSSH key type")
    try:
        decoded = base64.b64decode(fields[1], validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("SSH public key payload is not valid base64") from exc
    if not decoded:
        raise ValueError("SSH public key payload is empty")
    ssh_keygen = shutil.which("ssh-keygen")
    if not ssh_keygen:
        raise ValueError("ssh-keygen is required to validate the OpenSSH public key")
    inspected = subprocess.run(
        [ssh_keygen, "-lf", str(path)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if inspected.returncode != 0:
        raise ValueError("SSH public key is not accepted by ssh-keygen")
    return lines[0]


def _render_agent_cloud_init(ssh_user: str, public_key: str) -> str:
    """Render and parse the exact Terraform-template branch used by agent VMs.

    Agent VMs intentionally use only the credential-free ``users`` prefix of
    ``cloud_init.yaml.tpl``. Rendering that prefix here makes malformed user data
    a local preflight failure, before Terraform can call the provider.
    """

    template = (
        Path(__file__).resolve().parents[1]
        / "deploy"
        / "terraform"
        / "cloud_init.yaml.tpl"
    ).read_text(encoding="utf-8")
    if template.count(_AGENT_CLOUD_INIT_BRANCH) != 1:
        raise ValueError("agent cloud-init template boundary is missing or ambiguous")
    rendered = template.split(_AGENT_CLOUD_INIT_BRANCH, 1)[0]
    rendered = rendered.replace("${jsonencode(ssh_user)}", json.dumps(ssh_user))
    rendered = rendered.replace("${jsonencode(ssh_public_key)}", json.dumps(public_key))
    if "${" in rendered or "%{" in rendered:
        raise ValueError("agent cloud-init template contains unresolved interpolation")
    parsed = yaml.safe_load(rendered)
    users = parsed.get("users") if isinstance(parsed, dict) else None
    if not isinstance(users, list) or len(users) != 1:
        raise ValueError("agent cloud-init must contain exactly one user")
    user = users[0]
    keys = user.get("ssh_authorized_keys") if isinstance(user, dict) else None
    if user.get("name") != ssh_user or keys != [public_key]:
        raise ValueError("agent cloud-init changed the requested SSH identity")
    return rendered


def _is_openssh_public_key(path: Path) -> bool:
    """Return whether *path* is one bounded OpenSSH public-key line.

    Merely checking that the path exists lets an operator accidentally pass the
    adjacent private key. Terraform would then interpolate a multiline secret
    into cloud-init user-data before the provider rejects the malformed YAML.
    """

    try:
        _openssh_public_key_line(path)
    except (OSError, UnicodeError, ValueError):
        return False
    return True


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
        str(terraform_bin).strip() if terraform_bin is not None else _terraform_binary()
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
        str(public_path)[:-4] if str(public_path).endswith(".pub") else str(public_path)
    )
    public_key = ""
    if public_path.is_file():
        try:
            public_key = _openssh_public_key_line(public_path)
        except (OSError, UnicodeError, ValueError):
            public_key = ""
    if public_key:
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

    if public_key:
        try:
            _render_agent_cloud_init("ubuntu", public_key)
        except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
            results.append(
                CheckResult(
                    name="cloud_init_yaml",
                    status=FAIL,
                    summary="Rendered agent cloud-init YAML failed local validation.",
                    remedy=str(exc),
                )
            )
        else:
            results.append(
                CheckResult(
                    name="cloud_init_yaml",
                    status=PASS,
                    summary=(
                        "Rendered agent cloud-init YAML is valid and contains exactly "
                        "one OpenSSH public-key record."
                    ),
                )
            )
    else:
        results.append(
            CheckResult(
                name="cloud_init_yaml",
                status=FAIL,
                summary="Agent cloud-init was not rendered with an invalid public key.",
                remedy="Pass exactly one valid OpenSSH public-key line in the `.pub` file.",
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
