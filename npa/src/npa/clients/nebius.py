"""Nebius CLI wrapper for authentication and resource management.

Calls the ``nebius`` binary to obtain IAM tokens, manage service accounts,
access keys, and S3 buckets — replacing the need to manually source
``environment.sh`` before running ``npa workbench lerobot deploy``.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import warnings
from datetime import datetime, timezone
from collections.abc import Callable
from typing import Any

from npa.smoke._versions import supported_tool_version


class NebiusError(Exception):
    pass


# ── Low-level CLI runner ─────────────────────────────────────────────────

_NEBIUS_VERSION_CHECKED = False


def _parse_cli_version(output: str) -> str | None:
    match = re.search(r"\b(?:v)?(\d+\.\d+\.\d+)\b", output)
    if match is None:
        return None
    return match.group(1)


def _warn_if_nebius_version_mismatch(nebius_path: str) -> None:
    global _NEBIUS_VERSION_CHECKED

    if _NEBIUS_VERSION_CHECKED:
        return
    _NEBIUS_VERSION_CHECKED = True

    try:
        expected = supported_tool_version("nebius-cli", __file__)
        result = subprocess.run(
            [nebius_path, "version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        warnings.warn(
            f"Could not check Nebius CLI version: {type(exc).__name__}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    if result.returncode != 0:
        warnings.warn(
            f"Could not check Nebius CLI version (exit {result.returncode}): {output}",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    actual = _parse_cli_version(output)
    if actual is None:
        warnings.warn(
            f"Could not parse Nebius CLI version from output: {output}",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    if actual != expected:
        warnings.warn(
            f"Nebius CLI version mismatch: expected {expected}; found {actual}",
            RuntimeWarning,
            stacklevel=2,
        )


def _require_nebius() -> str:
    path = shutil.which("nebius")
    if path is None:
        raise NebiusError(
            "nebius CLI not found on PATH. "
            "Install it: https://docs.nebius.com/cli/install"
        )
    _warn_if_nebius_version_mismatch(path)
    return path


def _run(args: list[str], *, check: bool = True) -> str:
    """Run a nebius CLI command, return stdout."""
    nebius = _require_nebius()
    result = subprocess.run(
        [nebius] + args,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        stderr = result.stderr.strip()
        raise NebiusError(
            f"nebius {' '.join(args[:3])} failed (exit {result.returncode}):\n{stderr}"
        )
    return result.stdout.strip()


def _run_json(args: list[str], *, check: bool = True) -> dict[str, Any]:
    """Run a nebius CLI command with --format json, parse and return the result."""
    raw = _run(args + ["--format", "json"], check=check)
    if not raw:
        return {}
    return json.loads(raw)


# ── IAM token ────────────────────────────────────────────────────────────


def get_iam_token() -> str:
    """Obtain a short-lived IAM access token via ``nebius iam get-access-token``."""
    token = _run(["iam", "get-access-token"])
    if not token:
        raise NebiusError("nebius iam get-access-token returned an empty token")
    return token


# ── Service account ──────────────────────────────────────────────────────


def ensure_service_account(
    project_id: str,
    name: str = "lerobot-training",
) -> str:
    """Get or create a service account, return its ID."""
    # Try to find existing.
    try:
        data = _run_json([
            "iam", "service-account", "get-by-name",
            "--parent-id", project_id,
            "--name", name,
        ])
        sa_id = data.get("metadata", {}).get("id", "")
        if sa_id:
            return sa_id
    except NebiusError:
        pass  # Not found — create below.

    data = _run_json([
        "iam", "service-account", "create",
        "--parent-id", project_id,
        "--name", name,
        "--description", "Service account for LeRobot training on Nebius",
    ])
    sa_id = data.get("metadata", {}).get("id", "")
    if not sa_id:
        raise NebiusError("Service account creation did not return an ID")
    return sa_id


# ── Editors group membership ─────────────────────────────────────────────


def ensure_editors_membership(tenant_id: str, sa_id: str) -> None:
    """Add the service account to the tenant's *editors* group."""
    group_data = _run_json([
        "iam", "group", "get-by-name",
        "--parent-id", tenant_id,
        "--name", "editors",
    ])
    group_id = group_data.get("metadata", {}).get("id", "")
    if not group_id:
        raise NebiusError(f"Could not find editors group in tenant {tenant_id}")

    # Check membership.
    members_data = _run_json([
        "iam", "group-membership", "list-members",
        "--parent-id", group_id,
        "--page-size", "1000",
    ])
    memberships = members_data.get("memberships", [])
    for m in memberships:
        if m.get("spec", {}).get("member_id") == sa_id:
            return  # Already a member.

    _run([
        "iam", "group-membership", "create",
        "--parent-id", group_id,
        "--member-id", sa_id,
    ])


# ── Access keys ──────────────────────────────────────────────────────────


def _find_active_access_key(
    project_id: str,
    sa_id: str,
    *,
    key_name: str | None = None,
) -> dict[str, Any] | None:
    """Return the first ACTIVE access key for the given service account, or None."""
    data = _run_json([
        "iam", "v2", "access-key", "list",
        "--parent-id", project_id,
    ])
    for item in data.get("items", []):
        spec = item.get("spec", {})
        account = spec.get("account", {})
        # The SA ID can live under different JSON paths depending on API version.
        item_sa_id = (
            account.get("service_account", {}).get("id", "")
            or account.get("service_account_id", "")
        )
        if item_sa_id != sa_id:
            continue
        status = item.get("status", {})
        if status.get("state") != "ACTIVE":
            continue
        # Check expiry.
        expires_at = spec.get("expires_at", "")
        if expires_at:
            try:
                exp_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                # Nebius uses the Unix epoch as a "no expiration" sentinel for
                # access keys created without an expiry.
                if exp_dt.year > 1971 and exp_dt < datetime.now(timezone.utc):
                    continue  # Expired.
            except ValueError:
                pass
        if key_name and item.get("metadata", {}).get("name") != key_name:
            continue
        return item
    return None


def ensure_access_key(
    project_id: str,
    sa_id: str,
    *,
    key_name: str = "lerobot-access-key",
) -> tuple[str, str]:
    """Ensure an active access key exists, return (aws_access_key_id, aws_secret_access_key).

    Reuses an existing key when possible; creates a new one otherwise.
    """
    existing = (
        _find_active_access_key(project_id, sa_id, key_name=key_name)
        or _find_active_access_key(project_id, sa_id)
    )
    if existing:
        key_id = existing["metadata"]["id"]
        # Retrieve the AWS access key ID.
        get_data = _run_json(["iam", "v2", "access-key", "get", "--id", key_id])
        aws_access_key = get_data.get("status", {}).get("aws_access_key_id", "")
        # Try to retrieve the secret (works for keys where the secret is stored).
        try:
            secret_data = _run_json(["iam", "v2", "access-key", "get-secret", "--id", key_id])
            aws_secret_key = secret_data.get("secret", "")
        except NebiusError:
            aws_secret_key = ""

        if aws_access_key and aws_secret_key:
            return aws_access_key, aws_secret_key
        # Secret not retrievable — fall through to create a new key.

    # Create a fresh key without deleting existing keys. Existing keys may own
    # Terraform remote-state objects for workbenches that still need destroy.
    create_name = key_name
    if existing:
        create_name = f"{key_name}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    create_data = _run_json([
        "iam", "v2", "access-key", "create",
        "--parent-id", project_id,
        "--name", create_name,
        "--account-service-account-id", sa_id,
        "--description", "Access key for LeRobot S3 and API access",
    ])
    new_key_id = create_data.get("metadata", {}).get("id", "")
    if not new_key_id:
        raise NebiusError("Access key creation did not return an ID")

    # Fetch the AWS-compatible credentials.
    get_data = _run_json(["iam", "v2", "access-key", "get", "--id", new_key_id])
    aws_access_key = get_data.get("status", {}).get("aws_access_key_id", "")

    secret_data = _run_json(["iam", "v2", "access-key", "get-secret", "--id", new_key_id])
    aws_secret_key = secret_data.get("secret", "")

    if not aws_access_key or not aws_secret_key:
        raise NebiusError("Failed to retrieve AWS-compatible credentials from new access key")

    return aws_access_key, aws_secret_key


# ── S3 bucket ────────────────────────────────────────────────────────────


def bucket_name_for(tenant_id: str, project_id: str) -> str:
    """Derive a deterministic bucket name from tenant + project IDs.

    Matches the logic in ``environment.sh`` so existing buckets are reused.
    """
    raw = f"{tenant_id}-{project_id}"
    suffix = hashlib.md5(raw.encode()).hexdigest()[:8]
    return f"lerobot-{suffix}"


def ensure_bucket(project_id: str, bucket_name: str) -> str:
    """Get or create an S3 bucket, return its name."""
    data = _run_json([
        "storage", "bucket", "list",
        "--parent-id", project_id,
    ])
    for item in data.get("items", []):
        if item.get("metadata", {}).get("name") == bucket_name:
            return bucket_name

    _run([
        "storage", "bucket", "create",
        "--name", bucket_name,
        "--parent-id", project_id,
        "--versioning-policy", "enabled",
    ])
    return bucket_name


# ── Composite bootstrap ─────────────────────────────────────────────────


def bootstrap_environment(
    project_id: str,
    tenant_id: str,
    region: str,
    *,
    on_status: Callable[[str], None] | None = None,
) -> dict[str, str]:
    """Run the full environment bootstrap, return a dict of credentials.

    This is the Python equivalent of ``source environment.sh``.

    *on_status* is an optional callback ``(message: str) -> None`` for
    progress reporting.
    """

    def _status(msg: str) -> None:
        if on_status:
            on_status(msg)

    _status("Getting IAM access token...")
    iam_token = get_iam_token()

    _status("Setting up service account...")
    sa_id = ensure_service_account(project_id)

    _status("Configuring service account permissions...")
    ensure_editors_membership(tenant_id, sa_id)

    _status("Setting up S3 bucket...")
    bucket_name = bucket_name_for(tenant_id, project_id)
    ensure_bucket(project_id, bucket_name)

    _status("Setting up access key for S3...")
    aws_access_key, aws_secret_key = ensure_access_key(project_id, sa_id)

    s3_endpoint = f"https://storage.{region}.nebius.cloud"

    return {
        "iam_token": iam_token,
        "service_account_id": sa_id,
        "nebius_api_key": aws_access_key,
        "nebius_secret_key": aws_secret_key,
        "s3_bucket": bucket_name,
        "s3_endpoint": s3_endpoint,
        "nebius_project_id": project_id,
        "nebius_region": region,
    }
