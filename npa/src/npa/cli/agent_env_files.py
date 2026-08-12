"""Remote env/profile files the agent VM bootstrap writes.

Split out of ``npa.cli.agent`` to keep that module under its size ratchet; the
functions are re-exported from ``npa.cli.agent`` so callers and tests that patch
them there keep working.
"""

from __future__ import annotations

import json
import shlex
import uuid
from typing import Any

from npa.clients.ssh import SSHClient


def _stage_private_text(
    ssh: SSHClient,
    *,
    content: str,
    target: str,
    owner: str = "root:root",
) -> None:
    """Stage a secret over SFTP, then atomically install it owner-only."""

    nonce = uuid.uuid4().hex
    remote_source = f"/tmp/.npa-private-{nonce}"
    remote_target = f"{target}.npa-{nonce}"
    ssh.upload_private_text(content, remote_source)
    command = (
        "set -eu; "
        f"sudo install -m 600 -o {shlex.quote(owner.split(':', 1)[0])} "
        f"-g {shlex.quote(owner.split(':', 1)[1])} "
        f"{shlex.quote(remote_source)} {shlex.quote(remote_target)}; "
        f"sudo mv -f {shlex.quote(remote_target)} {shlex.quote(target)}; "
        f"rm -f {shlex.quote(remote_source)}"
    )
    try:
        ssh.run_or_raise(command, label=f"stage private {target}")
    finally:
        ssh.run(f"rm -f {shlex.quote(remote_source)} {shlex.quote(remote_target)}")


def _write_agent_s3_env(
    ssh: SSHClient,
    *,
    bucket: str,
    prefix: str = "",
    endpoint: str,
    access_key: str,
    secret_key: str,
    region: str,
) -> None:
    """Stage S3 discovery credentials on the VM (read-only operator scope preferred)."""
    if not (bucket.strip() and access_key.strip() and secret_key.strip()):
        return
    env_lines = [
        f"NPA_AGENT_S3_BUCKET={bucket.strip()}",
        f"NPA_AGENT_S3_PREFIX={prefix.strip().strip('/')}",
        f"NPA_AGENT_S3_ENDPOINT={endpoint.strip()}",
        f"AWS_ACCESS_KEY_ID={access_key.strip()}",
        f"AWS_SECRET_ACCESS_KEY={secret_key.strip()}",
        f"AWS_REGION={region.strip() or 'eu-north1'}",
        "",
    ]
    _stage_private_text(
        ssh,
        content="\n".join(env_lines),
        target="/opt/npa-agent/s3.env",
    )

def _write_agent_operator_profile(
    ssh: SSHClient,
    *,
    ssh_user: str,
    project_alias: str,
    project_id: str,
    tenant_id: str,
    region: str,
    tf_api_key: str,
    nebius_ai_key: str = "",
    s3_bucket: str,
    s3_prefix: str = "",
    s3_endpoint: str,
    s3_access_key: str,
    s3_secret_key: str,
    service_account_id: str = "",
) -> None:
    """Write ~/.npa/config.yaml + credentials.yaml on the agent VM for operator workflows."""
    if not (project_alias and project_id and tenant_id and region):
        return
    config_payload: dict[str, Any] = {
        "default_project": project_alias,
        "projects": {
            project_alias: {
                "project_id": project_id,
                "tenant_id": tenant_id,
                "region": region,
            }
        },
    }
    credentials_payload: dict[str, Any] = {"tokens": {}}
    tokens = credentials_payload["tokens"]
    if isinstance(tokens, dict):
        if nebius_ai_key.strip():
            tokens["NEBIUS_AI_CLOUD_KEY"] = nebius_ai_key.strip()
        if tf_api_key.strip():
            tokens["NEBIUS_TOKEN_FACTORY_KEY"] = tf_api_key.strip()
    storage_payload = {
        "access_key_id": s3_access_key.strip(),
        "secret_access_key": s3_secret_key.strip(),
        "endpoint": s3_endpoint.strip(),
        "bucket": "s3://" + s3_bucket.strip() + (("/" + s3_prefix.strip().strip("/") + "/") if s3_prefix.strip().strip("/") else ""),
    }
    if any(storage_payload.values()):
        credentials_payload["storage"] = storage_payload
    if service_account_id.strip():
        credentials_payload["nebius"] = {"service_account_id": service_account_id.strip()}
    user_home = f"/home/{ssh_user}"
    targets = [
        (f"{user_home}/.npa", f"{ssh_user}:{ssh_user}"),
        ("/root/.npa", "root:root"),
    ]
    for npa_dir, owner in targets:
        config_path = f"{npa_dir}/config.yaml"
        creds_path = f"{npa_dir}/credentials.yaml"
        ssh.run_or_raise(
            f"sudo install -d -m 700 -o {shlex.quote(owner.split(':', 1)[0])} "
            f"-g {shlex.quote(owner.split(':', 1)[1])} {shlex.quote(npa_dir)}",
            label=f"prepare private {npa_dir}",
        )
        _stage_private_text(
            ssh,
            content=json.dumps(config_payload, indent=2) + "\n",
            target=config_path,
            owner=owner,
        )
        _stage_private_text(
            ssh,
            content=json.dumps(credentials_payload, indent=2) + "\n",
            target=creds_path,
            owner=owner,
        )

def _write_agent_nebius_env(
    ssh: SSHClient,
    *,
    project_alias: str,
    agent_name: str,
    project_id: str,
    tenant_id: str,
    region: str,
    service_account_id: str,
    bucket: str,
    endpoint: str,
    access_key: str,
    secret_key: str,
    iam_token: str = "",
) -> None:
    """Stage long-lived Nebius project credentials on the agent VM.

    This stages the S3 access key (HMAC — required for Nebius object storage and
    not replaceable by an IAM bearer token) plus project identifiers. It does NOT
    stage any IAM/management token: the VM authenticates to Nebius IAM using its
    ATTACHED service account, which self-mints fresh tokens via the metadata /
    token-file sources ``get_iam_token()`` reads. Copying the operator's
    short-lived token here would go stale on the long-lived VM.
    """
    del iam_token
    if not (project_id.strip() and access_key.strip() and secret_key.strip()):
        return
    env_lines = [
        f"NPA_AGENT_PROJECT_ALIAS={project_alias.strip()}",
        f"NPA_AGENT_NAME={agent_name.strip()}",
        f"NEBIUS_PROJECT_ID={project_id.strip()}",
        f"NEBIUS_TENANT_ID={tenant_id.strip()}",
        f"NEBIUS_REGION={region.strip() or 'eu-north1'}",
        f"NEBIUS_SERVICE_ACCOUNT_ID={service_account_id.strip()}",
        "NEBIUS_PROFILE=cursor-sa",
        "NPA_NEBIUS_PROFILE=cursor-sa",
        "NPA_NEBIUS_CONFIG=/root/.nebius/config.yaml",
        "NPA_NEBIUS_CREDENTIAL_SOURCE=instance_metadata",
        f"NEBIUS_S3_BUCKET={bucket.strip()}",
        f"NEBIUS_S3_ENDPOINT={endpoint.strip()}",
        f"AWS_ACCESS_KEY_ID={access_key.strip()}",
        f"AWS_SECRET_ACCESS_KEY={secret_key.strip()}",
        f"AWS_REGION={region.strip() or 'eu-north1'}",
        "",
    ]
    _stage_private_text(
        ssh,
        content="\n".join(env_lines),
        target="/opt/npa-agent/nebius.env",
    )
