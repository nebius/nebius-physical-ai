"""Executable prerequisite checks for the agent CLI."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from enum import Enum
from pathlib import Path

from npa.workflows.sim2real_health import CheckResult, FAIL, PASS


class PreflightOutput(str, Enum):
    """Machine-readable choices accepted by ``npa agent preflight --output``."""

    text = "text"
    json = "json"


def terraform_cli_result(
    executable: str,
    *,
    run: Callable[..., subprocess.CompletedProcess[str]],
) -> CheckResult:
    """Validate the documented Terraform CLI 1.x compatibility contract."""
    if not executable:
        return CheckResult(
            name="terraform",
            status=FAIL,
            summary="terraform binary not found on PATH.",
            remedy="Install it: https://developer.hashicorp.com/terraform/install",
        )

    try:
        result = run(
            [executable, "version", "-json"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return CheckResult(
            name="terraform",
            status=FAIL,
            summary=f"Could not execute terraform version ({type(exc).__name__}).",
            remedy="Install Terraform CLI 1.x and ensure it is executable on PATH.",
        )
    if result.returncode != 0:
        return CheckResult(
            name="terraform",
            status=FAIL,
            summary=f"terraform version failed (exit {result.returncode}).",
            remedy="Install Terraform CLI 1.x and ensure it is executable on PATH.",
        )
    try:
        actual = str(json.loads(result.stdout).get("terraform_version") or "")
    except (json.JSONDecodeError, AttributeError):
        actual = ""
    try:
        major = int(actual.split(".", 1)[0])
    except (TypeError, ValueError):
        major = -1
    if major != 1:
        found = actual or "unparseable version output"
        return CheckResult(
            name="terraform",
            status=FAIL,
            summary=f"Unsupported Terraform CLI version ({found}); supported range is 1.x.",
            remedy="Install Terraform CLI 1.x: https://developer.hashicorp.com/terraform/install",
        )
    return CheckResult(
        name="terraform",
        status=PASS,
        summary=f"Terraform CLI {actual} found ({executable}).",
    )


def agent_ssh_key_results(ssh_public_key_path: str) -> list[CheckResult]:
    """Return side-effect-free public/private SSH key prerequisites."""
    results: list[CheckResult] = []
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
                    f"Generate a keypair (`ssh-keygen -t ed25519 -f {priv_hint}`) or pass --ssh-public-key-path to an existing key."
                ),
            )
        )

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
