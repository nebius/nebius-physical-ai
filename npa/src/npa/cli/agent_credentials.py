"""Secure credential staging helpers for an NPA agent VM."""

from __future__ import annotations

import base64
import json
import secrets
import shlex
import tempfile
from pathlib import Path
from typing import Any, Protocol


class CredentialSSH(Protocol):
    def run_or_raise(self, command: str) -> Any: ...

    def run(self, command: str) -> Any: ...

    def upload_file(self, local_path: str, remote_path: str) -> Any: ...


def resolve_operator_credentials() -> tuple[str, str, str]:
    """Return scoped API credentials staged onto the remote agent VM."""
    from npa.clients.credentials import load_credentials

    creds = load_credentials()
    return creds.ai_cloud_api_key, creds.token_factory_api_key, creds.foxglove_api_token


def write_agent_operator_profile(
    ssh: CredentialSSH,
    *,
    ssh_user: str,
    project_alias: str,
    project_id: str,
    tenant_id: str,
    region: str,
    tf_api_key: str,
    nebius_ai_key: str,
    foxglove_api_token: str = "",
    s3_bucket: str,
    s3_prefix: str = "",
    s3_endpoint: str,
    s3_access_key: str,
    s3_secret_key: str,
    service_account_id: str = "",
) -> None:
    """Write mode-safe operator config and credentials on the agent VM."""
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
        if foxglove_api_token.strip():
            tokens["FOXGLOVE_API_TOKEN"] = foxglove_api_token.strip()
    storage_payload = {
        "access_key_id": s3_access_key.strip(),
        "secret_access_key": s3_secret_key.strip(),
        "endpoint": s3_endpoint.strip(),
        "bucket": "s3://"
        + s3_bucket.strip()
        + (
            ("/" + s3_prefix.strip().strip("/") + "/")
            if s3_prefix.strip().strip("/")
            else ""
        ),
    }
    if any(storage_payload.values()):
        credentials_payload["storage"] = storage_payload
    if service_account_id.strip():
        credentials_payload["nebius"] = {
            "service_account_id": service_account_id.strip()
        }
    targets = [
        (f"/home/{ssh_user}/.npa", f"{ssh_user}:{ssh_user}"),
        ("/root/.npa", "root:root"),
    ]
    local_paths: list[Path] = []
    remote_paths: list[str] = []
    remote_stage_dir = f"/tmp/npa-agent-{secrets.token_hex(8)}"
    try:
        ssh.run_or_raise(f"install -d -m 0700 {shlex.quote(remote_stage_dir)}")
        for label, payload in (
            ("config", config_payload),
            ("credentials", credentials_payload),
        ):
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", delete=False
            ) as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
                local_path = Path(handle.name)
            local_paths.append(local_path)
            local_path.chmod(0o600)
            remote_path = f"{remote_stage_dir}/{label}.json"
            remote_paths.append(remote_path)
            ssh.upload_file(str(local_path), remote_path)

        commands: list[str] = []
        for npa_dir, owner in targets:
            owner_user, owner_group = owner.split(":", 1)
            commands.extend(
                [
                    f"sudo mkdir -p {shlex.quote(npa_dir)}",
                    f"sudo install -o {shlex.quote(owner_user)} -g {shlex.quote(owner_group)} -m 0600 {shlex.quote(remote_paths[0])} {shlex.quote(npa_dir + '/config.yaml')}",
                    f"sudo install -o {shlex.quote(owner_user)} -g {shlex.quote(owner_group)} -m 0600 {shlex.quote(remote_paths[1])} {shlex.quote(npa_dir + '/credentials.yaml')}",
                    f"sudo chown {shlex.quote(owner)} {shlex.quote(npa_dir)}",
                    f"sudo chmod 700 {shlex.quote(npa_dir)}",
                ]
            )
        ssh.run_or_raise(" && ".join(commands))
    finally:
        for local_path in local_paths:
            local_path.unlink(missing_ok=True)
        for remote_path in remote_paths:
            ssh.run(f"rm -f {shlex.quote(remote_path)}")
        ssh.run(f"rmdir {shlex.quote(remote_stage_dir)} 2>/dev/null || true")


def write_agent_nebius_env(
    ssh: CredentialSSH,
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
    """Stage long-lived Nebius project credentials on the agent VM."""
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
    ]
    env_lines.append("")
    env_b64 = base64.b64encode("\n".join(env_lines).encode()).decode("ascii")
    ssh.run_or_raise(
        f"echo {shlex.quote(env_b64)} | base64 -d | sudo tee /opt/npa-agent/nebius.env >/dev/null && sudo chmod 600 /opt/npa-agent/nebius.env"
    )
