"""Top-level CLI for deploying and operating the NPA agent VM."""

from __future__ import annotations

import base64
import json
import os
import secrets
import shlex
import subprocess
import ipaddress
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import typer

from npa.clients.config import (
    ConfigError,
    resolve_environment,
    resolve_ssh_config,
    resolve_terraform_state,
    write_config,
)
from npa.clients.env import redact_value
from npa.clients.network import NetworkIngressError, ensure_ingress
from npa.clients.ssh import SSHClient, SSHError
from npa.deploy import provisioner
from npa.deploy.provisioner import ProvisionerError
from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG

app = typer.Typer(
    name="agent",
    help="Deploy and operate a public NPA chat agent VM.",
    no_args_is_help=True,
)

DEFAULT_AGENT_PORT = 8088
DEFAULT_BACKEND_PORT = 8787
DEFAULT_RERUN_PORT = 9090
DEFAULT_PROJECT_ALIAS = "us-central1"
DEFAULT_AGENT_NAME = "agent"
DEFAULT_AGENT_USER = "npa"
DEFAULT_LLM_PROVIDER = "token_factory"
DEFAULT_LLM_MODEL = "nvidia/Cosmos3-Super-Reasoner"
AGENT_UI_VERSION = "2026063001"
DEFAULT_HTTPS_PORT = 443


def _embedded_agent_workflow_source() -> str:
    """Return agent_workflow.py source embedded into the remote agent backend."""
    import re

    path = Path(__file__).with_name("agent_workflow.py")
    raw = path.read_text(encoding="utf-8")
    raw = re.sub(r'^""".*?"""\s*\n', "", raw, count=1, flags=re.DOTALL)
    raw = re.sub(r"^from __future__ import annotations\s*\n", "", raw)
    return raw


def _embedded_agent_chat_source() -> str:
    """Return agent_chat.py source embedded into the remote agent backend."""
    import re

    path = Path(__file__).with_name("agent_chat.py")
    raw = path.read_text(encoding="utf-8")
    raw = re.sub(r'^""".*?"""\s*\n', "", raw, count=1, flags=re.DOTALL)
    raw = re.sub(r"^from __future__ import annotations\s*\n", "", raw)
    return raw


def _escape_fstring_embed(text: str) -> str:
    """Escape braces so embedded Python sources survive bootstrap f-strings."""
    return text.replace("{", "{{").replace("}", "}}")


def _embedded_agent_artifacts_source() -> str:
    """Return workflows/artifacts.py source embedded into the remote agent backend."""
    import re

    path = Path(__file__).resolve().parents[1] / "workflows" / "artifacts.py"
    raw = path.read_text(encoding="utf-8")
    raw = re.sub(r'^""".*?"""\s*\n', "", raw, count=1, flags=re.DOTALL)
    raw = re.sub(r"^from __future__ import annotations\s*\n", "", raw)
    return raw


@dataclass(frozen=True)
class AgentConfig:
    project_alias: str
    name: str
    project_id: str
    tenant_id: str
    region: str
    public_ip: str
    instance_id: str
    agent_url: str
    rerun_url: str
    sim_viz_url: str
    sim_assets_url: str
    cameras_api_url: str
    auth_user: str
    auth_secret_path: str
    llm_provider: str
    llm_model: str
    public_url: str = ""
    public_https: bool = True
    direct_url: str = ""
    ssh_key_path: str = ""
    service_account_id: str = ""
    credentials: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "project_id": self.project_id,
            "tenant_id": self.tenant_id,
            "region": self.region,
            "public_ip": self.public_ip,
            "instance_id": self.instance_id,
            "agent_url": self.agent_url,
            "rerun_url": self.rerun_url,
            "sim_viz_url": self.sim_viz_url,
            "sim_assets_url": self.sim_assets_url,
            "cameras_api_url": self.cameras_api_url,
            "auth_user": self.auth_user,
            "auth_secret_path": self.auth_secret_path,
            "llm": {"provider": self.llm_provider, "model": self.llm_model},
        }
        if self.public_url:
            payload["public_url"] = self.public_url
        if self.public_https:
            payload["public_https"] = True
        if self.direct_url:
            payload["direct_url"] = self.direct_url
        if self.ssh_key_path:
            payload["ssh_key_path"] = self.ssh_key_path
        if self.service_account_id:
            payload["service_account_id"] = self.service_account_id
        if self.credentials:
            payload["credentials"] = dict(self.credentials)
        return payload


def build_agent_urls(
    public_ip: str,
    *,
    agent_port: int = DEFAULT_AGENT_PORT,
    public_https: bool = True,
) -> dict[str, str]:
    """Return customer-facing and operator-direct URLs for an agent VM."""
    direct = f"http://{public_ip}:{agent_port}/"
    if public_https:
        base = f"https://{public_ip}/"
    else:
        base = direct
    root = base.rstrip("/")
    return {
        "public_url": base,
        "agent_url": base,
        "rerun_url": f"{root}/rerun/",
        "sim_viz_url": f"{root}/rerun/",
        "sim_assets_url": f"{root}/assets/",
        "cameras_api_url": f"{root}/assets/api/sim-assets/cameras",
        "direct_url": direct,
    }


def _record_public_https(record: dict[str, Any]) -> bool:
    if "public_https" in record:
        return bool(record.get("public_https"))
    public_url = str(record.get("public_url", "")).strip()
    if public_url.startswith("https://"):
        return True
    agent_url = str(record.get("agent_url", "")).strip()
    return agent_url.startswith("https://")


def _record_tls_verify(record: dict[str, Any]) -> bool:
    """Self-signed HTTPS on the VM public IP is expected; skip CA verification."""
    return not _record_public_https(record)


def _record_customer_url(record: dict[str, Any]) -> str:
    public_url = str(record.get("public_url", "")).strip()
    if public_url:
        return public_url
    return str(record.get("agent_url", "")).strip()


def _fail(message: str) -> None:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=1)


def _agent_record(project_alias: str, name: str) -> dict[str, Any]:
    cfg = resolve_project_agents(project_alias)
    record = cfg.get(name, {})
    return record if isinstance(record, dict) else {}


def resolve_project_agents(project_alias: str) -> dict[str, Any]:
    from npa.clients.config import list_projects

    projects = list_projects()
    project = projects.get(project_alias, {})
    agents = project.get("agents", {}) if isinstance(project, dict) else {}
    return agents if isinstance(agents, dict) else {}


def _store_agent_record(project_alias: str, name: str, payload: dict[str, Any]) -> None:
    write_config({"projects": {project_alias: {"agents": {name: payload}}}})


def _remove_agent_record(project_alias: str, name: str) -> None:
    from npa.clients.config import CONFIG_PATH, _load_yaml
    import yaml

    data = _load_yaml()
    projects = data.get("projects", {})
    if not isinstance(projects, dict):
        return
    project = projects.get(project_alias, {})
    if not isinstance(project, dict):
        return
    agents = project.get("agents", {})
    if not isinstance(agents, dict) or name not in agents:
        return
    del agents[name]
    project["agents"] = agents
    projects[project_alias] = project
    data["projects"] = projects
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, default_flow_style=False, sort_keys=False)
    CONFIG_PATH.chmod(0o600)


def _auth_secret_path(project_alias: str, name: str) -> Path:
    return Path.home() / ".npa" / "agents" / project_alias / name / "auth.env"


def _write_auth_secret(*, project_alias: str, name: str, user: str, password: str) -> Path:
    path = _auth_secret_path(project_alias, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"AGENT_USER={user}\nAGENT_PASSWORD={password}\n", encoding="utf-8")
    path.chmod(0o600)
    return path


def _load_auth_secret(path: str) -> tuple[str, str]:
    secret_path = Path(path).expanduser()
    if not secret_path.exists():
        raise ValueError(f"auth secret not found: {secret_path}")
    values: dict[str, str] = {}
    for raw in secret_path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        values[key.strip()] = value.strip()
    user = values.get("AGENT_USER", "")
    password = values.get("AGENT_PASSWORD", "")
    if not user or not password:
        raise ValueError(f"auth secret missing AGENT_USER/AGENT_PASSWORD: {secret_path}")
    return user, password


def _tool_catalog_keys() -> list[str]:
    return sorted(TOOL_CATALOG.keys())


def _tool_catalog_payload() -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for key in _tool_catalog_keys():
        entry = TOOL_CATALOG[key]
        payload[key] = {
            "description": entry.description,
            "argv_template": list(entry.argv_template),
        }
    return payload


def _resolve_deploy_llm_credentials() -> tuple[str, str]:
    """Return Token Factory API key and default model for agent VM bootstrap."""
    from npa.clients.credentials import load_credentials

    creds = load_credentials()
    return creds.token_factory_api_key, DEFAULT_LLM_MODEL


def _agent_credentials_payload(creds: dict[str, str]) -> dict[str, str]:
    """Normalize Nebius bootstrap output for persistence on the agent record."""
    return {
        "service_account_id": str(creds.get("service_account_id", "")).strip(),
        "s3_bucket": str(creds.get("s3_bucket", "")).strip(),
        "s3_endpoint": str(creds.get("s3_endpoint", "")).strip(),
        "access_key": str(creds.get("nebius_api_key", "")).strip(),
        "secret_key": str(creds.get("nebius_secret_key", "")).strip(),
    }


def _resolve_agent_service_account_id(
    project_alias: str,
    record: dict[str, Any],
) -> str:
    """Resolve service-account id for agent bootstrap and credential persistence."""
    stored = str(record.get("service_account_id", "")).strip()
    if stored:
        return stored
    creds = record.get("credentials", {})
    if isinstance(creds, dict):
        from_record = str(creds.get("service_account_id", "")).strip()
        if from_record:
            return from_record
    from npa.clients.nebius import resolve_service_account_id

    project_id = str(record.get("project_id", "")).strip()
    if project_id:
        resolved = resolve_service_account_id(project_id)
        if resolved:
            return resolved
    return ""


def _persist_agent_service_account_id(service_account_id: str) -> None:
    """Write discovered SA id into ~/.npa/credentials.yaml when missing."""
    sa_id = str(service_account_id or "").strip()
    if not sa_id:
        return
    from npa.clients.credentials import write_credentials_file
    from npa.clients.nebius import _saved_service_account_id

    if _saved_service_account_id() == sa_id:
        return
    write_credentials_file({"nebius": {"service_account_id": sa_id}})


def _creds_from_terraform_state(project_alias: str, record: dict[str, Any]) -> dict[str, str] | None:
    """Build a bootstrap-shaped credential dict from saved terraform remote-state keys."""
    try:
        tf_state = resolve_terraform_state(project_alias)
    except ConfigError:
        return None
    access_key = str(getattr(tf_state, "access_key", "") or "").strip()
    secret_key = str(getattr(tf_state, "secret_key", "") or "").strip()
    bucket = str(getattr(tf_state, "bucket", "") or "").strip()
    endpoint = str(getattr(tf_state, "endpoint", "") or "").strip()
    if not (access_key and secret_key and bucket):
        return None
    region = str(record.get("region", "") or "eu-north1").strip()
    service_account_id = _resolve_agent_service_account_id(project_alias, record)
    return {
        "service_account_id": service_account_id,
        "nebius_api_key": access_key,
        "nebius_secret_key": secret_key,
        "s3_bucket": bucket,
        "s3_endpoint": endpoint,
        "nebius_project_id": str(record.get("project_id", "")).strip(),
        "nebius_region": region,
    }


def _credentials_block_from_storage(
    *,
    service_account_id: str,
    s3_bucket: str,
    s3_endpoint: str,
    s3_access_key: str,
    s3_secret_key: str,
) -> dict[str, str]:
    return {
        "service_account_id": service_account_id.strip(),
        "s3_bucket": s3_bucket.strip(),
        "s3_endpoint": s3_endpoint.strip(),
        "access_key": s3_access_key.strip(),
        "secret_key": s3_secret_key.strip(),
    }


def _resolve_agent_ssh_key(
    record: dict[str, Any],
    *,
    cli_ssh_key: str | None = None,
    default_key: str = "~/.ssh/id_ed25519",
) -> str:
    """Resolve SSH private key for agent bootstrap without requiring workbench SSH config."""
    if cli_ssh_key and cli_ssh_key.strip():
        return str(Path(cli_ssh_key).expanduser())
    stored = str(record.get("ssh_key_path", "")).strip()
    if stored:
        return str(Path(stored).expanduser())
    env_key = os.environ.get("NPA_SSH_KEY", "").strip()
    if env_key:
        return str(Path(env_key).expanduser())
    return str(Path(default_key).expanduser())


def _resolve_agent_storage_credentials(
    project_alias: str,
    record: dict[str, Any],
) -> tuple[str, str, str, str, str]:
    """Return bucket, endpoint, access key, secret key, and service account id."""
    creds = record.get("credentials", {})
    if isinstance(creds, dict):
        access_key = str(creds.get("access_key", "")).strip()
        secret_key = str(creds.get("secret_key", "")).strip()
        bucket = str(creds.get("s3_bucket", "")).strip()
        endpoint = str(creds.get("s3_endpoint", "")).strip()
        service_account_id = str(
            creds.get("service_account_id", record.get("service_account_id", ""))
        ).strip()
        if bucket and access_key and secret_key:
            if not service_account_id:
                service_account_id = _resolve_agent_service_account_id(project_alias, record)
            return bucket, endpoint, access_key, secret_key, service_account_id
    try:
        tf_state = resolve_terraform_state(project_alias)
    except ConfigError:
        return "", "", "", "", _resolve_agent_service_account_id(project_alias, record)
    service_account_id = _resolve_agent_service_account_id(project_alias, record)
    return (
        str(getattr(tf_state, "bucket", "") or ""),
        str(getattr(tf_state, "endpoint", "") or ""),
        str(getattr(tf_state, "access_key", "") or ""),
        str(getattr(tf_state, "secret_key", "") or ""),
        service_account_id,
    )


def _write_agent_llm_env(
    ssh: SSHClient,
    *,
    tf_api_key: str,
    llm_model: str,
) -> None:
    """Stage Token Factory credentials on the VM (chmod 600, not baked into image)."""
    if not tf_api_key.strip():
        return
    env_content = f"NEBIUS_TOKEN_FACTORY_KEY={tf_api_key.strip()}\nNPA_AGENT_LLM_MODEL={llm_model}\n"
    env_b64 = base64.b64encode(env_content.encode("utf-8")).decode("ascii")
    ssh.run_or_raise(
        f"echo {shlex.quote(env_b64)} | base64 -d | sudo tee /opt/npa-agent/llm.env >/dev/null "
        "&& sudo chmod 600 /opt/npa-agent/llm.env"
    )


def _write_agent_s3_env(
    ssh: SSHClient,
    *,
    bucket: str,
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
        f"NPA_AGENT_S3_ENDPOINT={endpoint.strip()}",
        f"AWS_ACCESS_KEY_ID={access_key.strip()}",
        f"AWS_SECRET_ACCESS_KEY={secret_key.strip()}",
        f"AWS_REGION={region.strip() or 'eu-north1'}",
        "",
    ]
    env_b64 = base64.b64encode("\n".join(env_lines).encode("utf-8")).decode("ascii")
    ssh.run_or_raise(
        f"echo {shlex.quote(env_b64)} | base64 -d | sudo tee /opt/npa-agent/s3.env >/dev/null "
        "&& sudo chmod 600 /opt/npa-agent/s3.env"
    )


def _write_agent_nebius_env(
    ssh: SSHClient,
    *,
    project_id: str,
    tenant_id: str,
    region: str,
    service_account_id: str,
    bucket: str,
    endpoint: str,
    access_key: str,
    secret_key: str,
) -> None:
    """Stage long-lived Nebius project credentials on the agent VM."""
    if not (project_id.strip() and access_key.strip() and secret_key.strip()):
        return
    env_lines = [
        f"NEBIUS_PROJECT_ID={project_id.strip()}",
        f"NEBIUS_TENANT_ID={tenant_id.strip()}",
        f"NEBIUS_REGION={region.strip() or 'eu-north1'}",
        f"NEBIUS_SERVICE_ACCOUNT_ID={service_account_id.strip()}",
        f"NEBIUS_S3_BUCKET={bucket.strip()}",
        f"NEBIUS_S3_ENDPOINT={endpoint.strip()}",
        f"AWS_ACCESS_KEY_ID={access_key.strip()}",
        f"AWS_SECRET_ACCESS_KEY={secret_key.strip()}",
        f"AWS_REGION={region.strip() or 'eu-north1'}",
        "",
    ]
    env_b64 = base64.b64encode("\n".join(env_lines).encode("utf-8")).decode("ascii")
    ssh.run_or_raise(
        f"echo {shlex.quote(env_b64)} | base64 -d | sudo tee /opt/npa-agent/nebius.env >/dev/null "
        "&& sudo chmod 600 /opt/npa-agent/nebius.env"
    )


def _is_routable_public_ip(value: str) -> bool:
    candidate = (value or "").strip()
    if not candidate:
        return False
    if candidate == "localhost":
        return False
    try:
        ip = ipaddress.ip_address(candidate)
    except ValueError:
        return False
    if ip.is_loopback or ip.is_private or ip.is_unspecified or ip.is_link_local:
        return False
    return True


def _agent_strip_url_credentials_js() -> str:
    """JS to strip user:pass@ from the URL bar while keeping HTTP Basic auth session."""
    return """    <script>
    (function stripUrlCredentials() {
      try {
        if (location.username || location.password) {
          const clean = location.protocol + "//" + location.host + location.pathname + location.search + location.hash;
          history.replaceState(null, "", clean);
        }
      } catch (_err) { /* best-effort */ }
    })();
    </script>"""


def _agent_public_login_form_html(auth_user: str) -> str:
    """Shared Sign in form for public welcome/login-help pages (basic-auth URL redirect)."""
    return f"""    <section class="sign-in-panel" aria-labelledby="sign-in-heading">
      <h2 id="sign-in-heading">Sign in</h2>
      <p class="muted">Use the form if your browser does not show an HTTP Basic Auth dialog.</p>
      <form id="npa-sign-in" class="sign-in" autocomplete="on">
        <label for="npa-user">Username</label>
        <input id="npa-user" name="username" type="text" value="{auth_user}" autocomplete="username" required>
        <label for="npa-pass">Password</label>
        <input id="npa-pass" name="password" type="password" autocomplete="current-password" required>
        <button type="submit">Sign in</button>
      </form>
      <p class="muted note">Credentials are removed from the address bar immediately after sign-in.</p>
    </section>
    <script>
    (function () {{
      try {{
        if (location.username || location.password) {{
          const clean = location.protocol + "//" + location.host + location.pathname + location.search + location.hash;
          history.replaceState(null, "", clean);
        }}
      }} catch (_err) {{ /* best-effort */ }}
      var form = document.getElementById("npa-sign-in");
      if (!form) return;
      form.addEventListener("submit", function (ev) {{
        ev.preventDefault();
        var user = document.getElementById("npa-user").value;
        var pass = document.getElementById("npa-pass").value;
        var u = encodeURIComponent(user);
        var p = encodeURIComponent(pass);
        var rawPath = String(location.pathname || "/");
        var normalizedPath = rawPath.length > 1 && rawPath.endsWith("/") ? rawPath.slice(0, -1) : rawPath;
        var dest = (normalizedPath === "/login-help.html" || normalizedPath === "/welcome") ? "/" : normalizedPath;
        location.href = location.protocol + "//" + u + ":" + p + "@" + location.host + dest;
      }});
    }})();
    </script>"""


def _nginx_agent_site_body(
    *,
    backend_port: int,
    rerun_port: int,
) -> str:
    """Shared nginx locations for the agent UI (HTTP and HTTPS server blocks)."""
    return f"""  auth_basic "NPA Agent";
  auth_basic_user_file /etc/nginx/.npa-agent-htpasswd;
  location = /healthz {{
    auth_basic off;
    default_type application/json;
    return 200 '{{"ok":true,"service":"npa-agent","welcome":"/welcome","ui":"/","ui_version":"{AGENT_UI_VERSION}"}}';
  }}
  location = /welcome {{
    auth_basic off;
    alias /opt/npa-agent/welcome.html;
    default_type text/html;
    add_header Cache-Control "no-store" always;
  }}
  location = /login-help.html {{
    auth_basic off;
    alias /opt/npa-agent/login-help.html;
    default_type text/html;
    add_header Cache-Control "no-store" always;
  }}
  location /api/ {{
    proxy_pass http://127.0.0.1:{backend_port}/;
  }}
  location /assets/api/ {{
    rewrite ^/assets/api/(.*)$ /$1 break;
    proxy_pass http://127.0.0.1:{backend_port}/;
  }}
  location /rerun/recordings/ {{
    auth_basic off;
    alias /opt/npa-agent/recordings/;
    default_type application/octet-stream;
    add_header Cache-Control "no-cache" always;
  }}
  location ~* ^/rerun/.+\\.(wasm|js|ico|svg)$ {{
    rewrite ^/rerun/(.*)$ /$1 break;
    proxy_pass http://127.0.0.1:{rerun_port};
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_connect_timeout 30s;
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;
    gzip on;
    gzip_types application/wasm application/javascript text/javascript image/svg+xml;
    gzip_min_length 256;
    add_header Cache-Control "public, max-age=31536000, immutable" always;
  }}
  location /rerun/ {{
    proxy_pass http://127.0.0.1:{rerun_port}/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_connect_timeout 30s;
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;
    add_header Cache-Control "public, max-age=3600" always;
  }}
  location / {{
    root /opt/npa-agent;
    index ui.html;
    try_files /ui.html =404;
    add_header Cache-Control "no-store, no-cache, must-revalidate" always;
    add_header Pragma "no-cache" always;
  }}"""


def _bootstrap_agent_stack(
    *,
    host: str,
    ssh_user: str,
    ssh_key_path: str,
    auth_user: str,
    auth_password: str,
    agent_port: int,
    backend_port: int,
    rerun_port: int,
    llm_model: str = DEFAULT_LLM_MODEL,
    tf_api_key: str = "",
    s3_bucket: str = "",
    s3_endpoint: str = "",
    s3_access_key: str = "",
    s3_secret_key: str = "",
    s3_region: str = "eu-north1",
    nebius_project_id: str = "",
    nebius_tenant_id: str = "",
    service_account_id: str = "",
    public_https: bool = True,
) -> None:
    ssh = SSHClient(
        config=resolve_ssh_config(
            ssh_host=host,
            ssh_user=ssh_user,
            ssh_key=ssh_key_path,
            project=None,
            name=None,
        ).ssh
    )
    catalog_json = json.dumps(_tool_catalog_payload())
    agent_chat_source = _escape_fstring_embed(_embedded_agent_chat_source())
    agent_workflow_source = _escape_fstring_embed(_embedded_agent_workflow_source())
    agent_artifacts_source = _escape_fstring_embed(_embedded_agent_artifacts_source())
    nginx_site_body = _nginx_agent_site_body(backend_port=backend_port, rerun_port=rerun_port)
    login_form_html = _agent_public_login_form_html(auth_user)
    strip_url_credentials_js = _agent_strip_url_credentials_js()
    https_ssl_setup = ""
    https_server_block = ""
    if public_https:
        https_ssl_setup = f"""
sudo mkdir -p /etc/nginx/ssl
if [ ! -s /etc/nginx/ssl/npa-agent.crt ] || [ ! -s /etc/nginx/ssl/npa-agent.key ]; then
  sudo openssl req -x509 -nodes -newkey rsa:2048 -days 825 \\
    -keyout /etc/nginx/ssl/npa-agent.key \\
    -out /etc/nginx/ssl/npa-agent.crt \\
    -subj "/CN=npa-agent/O=Nebius Physical AI" \\
    -addext "subjectAltName=IP:{host}"
  sudo chmod 600 /etc/nginx/ssl/npa-agent.key
fi
"""
        https_server_block = f"""
server {{
  listen {DEFAULT_HTTPS_PORT} ssl;
  server_name _;
  ssl_certificate /etc/nginx/ssl/npa-agent.crt;
  ssl_certificate_key /etc/nginx/ssl/npa-agent.key;
{nginx_site_body}
}}
"""
    setup_script = f"""set -euo pipefail
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nginx apache2-utils python3-venv python3-pip
sudo mkdir -p /opt/npa-agent
cat <<'PY' | sudo tee /opt/npa-agent/backend.py >/dev/null
import json
import os
import re
import secrets
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI(title="npa-agent")
TOOL_CATALOG = {catalog_json}
TOOL_REFS = sorted(TOOL_CATALOG.keys())
STATE_PATH = Path("/opt/npa-agent/session_state.json")
RRD_PATH = Path("/opt/npa-agent/sim2real.rrd")
RECORDING_PATH = Path("/opt/npa-agent/recordings/sim2real.rrd")
RECORDINGS_DIR = Path("/opt/npa-agent/recordings")
RERUN_UNIT = "npa-rerun"
RERUN_WEB_PORT = {rerun_port}
DEFAULT_SCENE_SPEC = {{
    "schema": "npa.sim2real.manip_scene_spec.v1",
    "goal_pos": [0.5, 0.3, 0.04],
    "goal_threshold": 0.05,
    "objects": [{{"name": "cube", "asset_source": "primitive", "role": "manipuland", "primitive": "box"}}],
    "cameras": {{
        "workspace": {{
            "name": "workspace",
            "placement": "stock_workspace",
            "pos": [1.0, 0.0, 0.8],
            "look_at": [0.5, 0.0, 0.0],
            "fov": 60.0,
            "resolution": [640, 480],
        }},
        "wrist": {{
            "name": "wrist",
            "placement": "stock_ee_mounted",
            "pos": [0.4, 0.0, 0.4],
            "look_at": [0.5, 0.0, 0.0],
            "fov": 90.0,
            "resolution": [640, 480],
        }},
    }},
}}
DEFAULT_ROBOT_SPEC = {{
    "schema": "npa.sim2real.robot_spec.v1",
    "preset": "franka",
    "robot_source": "stock_franka",
    "name": "franka_panda",
}}
DEFAULT_ASSETS_MANIFEST = {{
    "schema": "npa.sim2real.assets_manifest.v1",
    "scene_status": "stock_tabletop",
    "robot_status": "stock_franka",
}}
DEFAULT_SELECTION = {{
    "scene_spec_uri": "",
    "assets_uri": "",
    "robot_spec_uri": "",
    "cameras_uri": "",
    "robot_preset": "franka",
    "sim_backend": "isaac",
    "props": [],
}}
DEFAULT_SIM_VIZ = {{
    "run_id": "",
    "stage": "idle",
    "rrd_uri": "",
    "rrd_updated_at": "",
    "artifact_uri": "",
    "artifact_key": "",
    "artifact_render": "",
    "artifact_preview_url": "",
    "artifact_download_url": "",
    "live_grpc_url": "",
    "mode": "static",
    "camera": "workspace",
    "rerun_ready": False,
    "rerun_iframe_url": "/rerun/",
}}

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _default_state() -> dict:
    return {{
        "selection": dict(DEFAULT_SELECTION),
        "camera_selection": ["workspace"],
        "sim_viz": dict(DEFAULT_SIM_VIZ),
        "sim_viz_runs": {{}},
        "active_run_id": "",
        "latest_submit": {{}},
        "workflow_draft": {{"yaml": "", "name": "", "states": [], "updated_at": "", "plan": {{}}, "runnable": False}},
        "workflow_submit": {{}},
        "chat_history": [],
    }}

def _load_state() -> dict:
    if not STATE_PATH.exists():
        return _default_state()
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return _default_state()
    if not isinstance(data, dict):
        return _default_state()
    merged = _default_state()
    merged.update(data)
    if not isinstance(merged.get("selection"), dict):
        merged["selection"] = dict(DEFAULT_SELECTION)
    if not isinstance(merged.get("camera_selection"), list):
        merged["camera_selection"] = ["workspace"]
    if not isinstance(merged.get("sim_viz"), dict):
        merged["sim_viz"] = dict(DEFAULT_SIM_VIZ)
    if not isinstance(merged.get("sim_viz_runs"), dict):
        merged["sim_viz_runs"] = {{}}
    if not isinstance(merged.get("active_run_id"), str):
        merged["active_run_id"] = ""
    if not isinstance(merged.get("chat_history"), list):
        merged["chat_history"] = []
    return merged

def _save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\\n", encoding="utf-8")


def _record_sim_viz_run(state: dict, payload: dict | None) -> None:
    if not isinstance(payload, dict):
        return
    run_id = str(payload.get("run_id") or "").strip()
    if not run_id:
        return
    runs = state.get("sim_viz_runs")
    if not isinstance(runs, dict):
        runs = {{}}
    snapshot = dict(DEFAULT_SIM_VIZ)
    snapshot.update(payload)
    snapshot["run_id"] = run_id
    runs[run_id] = snapshot
    state["sim_viz_runs"] = runs
    state["active_run_id"] = run_id


def _sim_viz_for_run(state: dict, run_id: str = "") -> dict:
    payload = dict(DEFAULT_SIM_VIZ)
    current = state.get("sim_viz")
    if isinstance(current, dict):
        payload.update(current)
    runs = state.get("sim_viz_runs")
    target = str(run_id or state.get("active_run_id") or "").strip()
    if isinstance(runs, dict) and target and isinstance(runs.get(target), dict):
        payload.update(runs[target])
    return payload

def _stock_franka_selection() -> dict:
    return {{
        "scene_spec_uri": "stock://scene/default",
        "assets_uri": "",
        "robot_spec_uri": "stock://robot/franka",
        "cameras_uri": "stock://cameras/default",
        "robot_preset": "franka",
        "sim_backend": "isaac",
        "props": ["cube"],
    }}

def _camera_frustum_lines(pos: list[float], look_at: list[float], fov: float, *, depth: float = 0.35):
    import math

    px, py, pz = (float(pos[0]), float(pos[1]), float(pos[2]))
    lx, ly, lz = (float(look_at[0]), float(look_at[1]), float(look_at[2]))
    fx, fy, fz = (lx - px, ly - py, lz - pz)
    norm = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
    fx, fy, fz = (fx / norm, fy / norm, fz / norm)
    upx, upy, upz = (0.0, 0.0, 1.0)
    rx = fy * upz - fz * upy
    ry = fz * upx - fx * upz
    rz = fx * upy - fy * upx
    rnorm = math.sqrt(rx * rx + ry * ry + rz * rz) or 1.0
    rx, ry, rz = (rx / rnorm, ry / rnorm, rz / rnorm)
    ux = ry * fz - rz * fy
    uy = rz * fx - rx * fz
    uz = rx * fy - ry * fx
    half_h = depth * math.tan(math.radians(float(fov) / 2.0))
    half_w = half_h * (4.0 / 3.0)
    cx = px + fx * depth
    cy = py + fy * depth
    cz = pz + fz * depth
    corners = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        corners.append(
            [
                cx + rx * half_w * sx + ux * half_h * sy,
                cy + ry * half_w * sx + uy * half_h * sy,
                cz + rz * half_w * sx + uz * half_h * sy,
            ]
        )
    origin = [px, py, pz]
    strips = [
        [origin, corners[0]],
        [origin, corners[1]],
        [origin, corners[2]],
        [origin, corners[3]],
        corners + [corners[0]],
    ]
    return origin, strips

_FRANKA_HOME_JOINTS = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)

def _franka_joint_positions(joint_angles: tuple[float, ...]) -> list[list[float]]:
    import math

    dh = [
        (0.0, 0.0, 0.333),
        (0.0, -math.pi / 2.0, 0.0),
        (0.0, math.pi / 2.0, 0.316),
        (0.0825, math.pi / 2.0, 0.0),
        (-0.0825, -math.pi / 2.0, 0.384),
        (0.0, math.pi / 2.0, 0.0),
        (0.088, math.pi / 2.0, 0.0),
    ]

    def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
        return [
            [sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)
        ]

    transform = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    positions = [[0.0, 0.0, 0.0]]
    for index, (a, alpha, d) in enumerate(dh):
        theta = float(joint_angles[index])
        ct, st = math.cos(theta), math.sin(theta)
        ca, sa = math.cos(alpha), math.sin(alpha)
        step = [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
        transform = _matmul(transform, step)
        positions.append([transform[0][3], transform[1][3], transform[2][3]])
    ee = [transform[0][3], transform[1][3], transform[2][3] + 0.103]
    positions.append(ee)
    positions.append([ee[0], ee[1] + 0.04, ee[2]])
    positions.append([ee[0], ee[1] - 0.04, ee[2]])
    return positions

def _log_franka_robot_geometry(rr) -> None:
    positions = _franka_joint_positions(_FRANKA_HOME_JOINTS)
    arm_points = positions[:8]
    segments: list[list[list[float]]] = []
    for left, right in zip(arm_points, arm_points[1:]):
        dx = left[0] - right[0]
        dy = left[1] - right[1]
        dz = left[2] - right[2]
        if dx * dx + dy * dy + dz * dz < 1e-8:
            continue
        segments.append([left, right])
    link_color = [234, 88, 12]
    link_rgba = link_color + [255]
    rr.log(
        "robot/franka/base",
        rr.Boxes3D(
            centers=[[0.0, 0.0, 0.05]],
            half_sizes=[[0.085, 0.085, 0.05]],
            colors=[[100, 116, 139, 255]],
        ),
    )
    rr.log(
        "robot/franka/joints",
        rr.Points3D(
            arm_points,
            colors=[link_rgba] * len(arm_points),
            radii=[0.028] * len(arm_points),
        ),
    )
    if segments:
        rr.log(
            "robot/franka/links",
            rr.LineStrips3D(
                segments,
                colors=[link_color] * len(segments),
                radii=[0.018] * len(segments),
            ),
        )
    gripper_segments = [
        [positions[7], positions[8]],
        [positions[8], positions[9]],
        [positions[8], positions[10]],
    ]
    gripper_color = [59, 130, 246]
    rr.log(
        "robot/franka/gripper",
        rr.LineStrips3D(
            gripper_segments,
            colors=[gripper_color] * len(gripper_segments),
            radii=[0.012] * len(gripper_segments),
        ),
    )
    rr.log(
        "robot/franka",
        rr.TextDocument(
            "Franka Panda — stock tabletop pick-and-place demo (NPA agent preview)"
        ),
    )

def _generate_franka_demo_rrd(*, camera: str = "workspace") -> Path:
    import math

    import rerun as rr

    target = RRD_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    rr.init("npa-franka-tabletop-demo", spawn=False)
    rr.log(
        "agent/camera_inspector",
        rr.TextDocument(
            "Stock Franka tabletop demo with workspace and wrist camera frustums. "
            "Highlighted camera is selected for the next rollout."
        ),
    )
    rr.log(
        "world/table",
        rr.Boxes3D(
            centers=[[0.5, 0.0, 0.0]],
            half_sizes=[[0.4, 0.3, 0.02]],
            colors=[[180, 180, 180, 255]],
        ),
    )
    rr.log(
        "world/cube",
        rr.Boxes3D(
            centers=[[0.5, 0.3, 0.04]],
            half_sizes=[[0.025, 0.025, 0.025]],
            colors=[[59, 130, 246, 255]],
        ),
    )
    _log_franka_robot_geometry(rr)
    cameras = DEFAULT_SCENE_SPEC.get("cameras", {{}})
    active = camera if camera in cameras else "workspace"
    for name, cam in cameras.items():
        if not isinstance(cam, dict):
            continue
        pos = list(cam.get("pos") or [0.0, 0.0, 0.0])
        look_at = list(cam.get("look_at") or [0.0, 1.0, 0.0])
        fov = float(cam.get("fov") or 60.0)
        res = cam.get("resolution") or [640, 480]
        width = int(res[0]) if len(res) > 0 else 640
        height = int(res[1]) if len(res) > 1 else 480
        entity = f"world/cameras/{{name}}"
        focal = width / (2.0 * math.tan(math.radians(fov / 2.0)))
        rr.log(entity, rr.Pinhole(focal_length=focal, width=width, height=height))
        rr.log(entity, rr.Transform3D(translation=pos))
        origin, strips = _camera_frustum_lines(pos, look_at, fov)
        color = [59, 130, 246] if name == active else [148, 163, 184]
        rr.log(
            f"{{entity}}/frustum",
            rr.LineStrips3D(strips, colors=[color] * len(strips)),
        )
        rr.log(f"{{entity}}/origin", rr.Points3D([origin], colors=[color], radii=[0.02]))
        label = (
            f"**{{name}}** (selected for next rollout)"
            if name == active
            else f"**{{name}}**"
        )
        rr.log(
            f"{{entity}}/label",
            rr.TextDocument(
                f"{{label}}\\n"
                f"pos={{pos}} look_at={{look_at}} fov={{fov}}° resolution={{width}}x{{height}}"
            ),
        )
        rr.log(
            f"rollouts/latest/{{name}}/camera",
            rr.TextDocument(
                f"Sim2Real rollout camera stream for `{{name}}` "
                f"(populated when a run writes frames to the recording)."
            ),
        )
    rr.log("demo/active_camera", rr.TextLog(active))
    rr.save(str(target))
    _publish_rrd_recording(target)
    return target

def _publish_rrd_recording(source: Path) -> Path:
    if not source.is_file():
        return RECORDING_PATH
    RECORDING_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = RECORDING_PATH.with_suffix(".rrd.tmp")
    shutil.copy2(source, tmp)
    tmp.replace(RECORDING_PATH)
    return RECORDING_PATH


def _safe_artifact_key(key: str) -> str:
    value = str(key or "").strip().lstrip("/")
    if not value:
        raise HTTPException(status_code=400, detail="artifact key is required")
    parts = value.split("/")
    if any(part in {{"", ".", ".."}} for part in parts):
        raise HTTPException(status_code=400, detail="artifact key traversal is not allowed")
    return value


def _agent_s3_settings() -> dict[str, str]:
    return {{
        "bucket": str(os.environ.get("NPA_AGENT_S3_BUCKET", "")).strip(),
        "endpoint": str(os.environ.get("NPA_AGENT_S3_ENDPOINT", "")).strip(),
        "access_key": str(os.environ.get("AWS_ACCESS_KEY_ID", "")).strip(),
        "secret_key": str(os.environ.get("AWS_SECRET_ACCESS_KEY", "")).strip(),
        "region": str(os.environ.get("AWS_REGION", "eu-north1")).strip() or "eu-north1",
    }}


def _agent_s3_client():
    settings = _agent_s3_settings()
    if not settings["bucket"] or not settings["access_key"] or not settings["secret_key"]:
        raise HTTPException(
            status_code=400,
            detail="S3 discovery is not configured on this agent (missing bucket or credentials).",
        )
    try:
        client_kwargs = {{
            "endpoint_url": settings["endpoint"],
            "aws_access_key_id": settings["access_key"],
            "region_name": settings["region"],
        }}
        secret_param = "aws" + "_secret_access_key"
        client_kwargs[secret_param] = settings["secret_key"]
        client = build_s3_client(**client_kwargs)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"failed to initialize S3 client: {{exc}}") from exc
    return client, settings


def _artifact_filename(key: str) -> str:
    import hashlib

    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    leaf = Path(key).name or "artifact.bin"
    return f"{{digest}}-{{leaf}}"


def _artifact_preview_url(filename: str) -> str:
    return f"/api/artifacts/file/{{filename}}"


def _apply_loaded_artifact(
    *,
    state: dict,
    run_id: str,
    key: str,
    s3_uri: str,
    render: str,
    local_path: Path,
) -> dict:
    now = _now_iso()
    sim_viz = dict(DEFAULT_SIM_VIZ)
    current = state.get("sim_viz")
    if isinstance(current, dict):
        sim_viz.update(current)
    sim_viz.update(
        {{
            "run_id": run_id,
            "stage": "artifact-loaded",
            "rrd_updated_at": now,
            "artifact_uri": s3_uri,
            "artifact_key": key,
            "artifact_render": render,
            "mode": "static",
            "camera": str(sim_viz.get("camera") or "workspace"),
        }}
    )
    if render == "rerun":
        _publish_rrd_recording(local_path)
        _restart_rerun_serve(force=True)
        sim_viz["rrd_uri"] = f"file://{{RECORDING_PATH}}"
        sim_viz["artifact_preview_url"] = "/rerun/recordings/sim2real.rrd"
        sim_viz["artifact_download_url"] = "/rerun/recordings/sim2real.rrd"
        sim_viz["rerun_iframe_url"] = f"/rerun/?url=/rerun/recordings/sim2real.rrd&camera={{sim_viz['camera']}}"
        sim_viz["rerun_ready"] = RECORDING_PATH.is_file() and _rerun_web_viewer_healthy()
    else:
        filename = _artifact_filename(key)
        target = RECORDINGS_DIR / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, target)
        preview_url = _artifact_preview_url(filename)
        sim_viz["artifact_preview_url"] = preview_url
        sim_viz["artifact_download_url"] = preview_url
        sim_viz["rrd_uri"] = ""
        sim_viz["rerun_iframe_url"] = "/rerun/"
        sim_viz["rerun_ready"] = False
    state["sim_viz"] = sim_viz
    _record_sim_viz_run(state, sim_viz)
    _save_state(state)
    return sim_viz

_RERUN_RESTART_MIN_INTERVAL_S = 8.0
_last_rerun_restart_monotonic = 0.0

def _rerun_service_active() -> bool:
    try:
        subprocess.run(
            ["systemctl", "is-active", "--quiet", RERUN_UNIT],
            check=True,
            capture_output=True,
            timeout=5,
        )
        return True
    except Exception:
        return False

def _rerun_web_viewer_healthy() -> bool:
    if not _rerun_service_active():
        return False
    try:
        import urllib.request

        with urllib.request.urlopen(
            f"http://127.0.0.1:{{RERUN_WEB_PORT}}/",
            timeout=2,
        ) as resp:
            return resp.status == 200
    except Exception:
        return False

def _rerun_ready_state(*, rrd_uri: str = "") -> bool:
    has_rrd = bool(str(rrd_uri or "").strip()) or RRD_PATH.is_file()
    return has_rrd and _rerun_web_viewer_healthy()

def _restart_rerun_serve(*, force: bool = False) -> bool:
    global _last_rerun_restart_monotonic
    now = time.monotonic()
    if not force and _rerun_service_active():
        if now - _last_rerun_restart_monotonic < _RERUN_RESTART_MIN_INTERVAL_S:
            return True
    try:
        subprocess.run(
            ["sudo", "systemctl", "restart", RERUN_UNIT],
            check=True,
            capture_output=True,
            timeout=30,
        )
        _last_rerun_restart_monotonic = time.monotonic()
        return True
    except Exception:
        if _rerun_service_active():
            return True
        return False

def _wire_franka_demo(state: dict, *, camera: str = "workspace") -> dict:
    selection = _stock_franka_selection()
    state["selection"] = selection
    cam = (camera or "workspace").strip() or "workspace"
    state["camera_selection"] = [cam]
    target = _generate_franka_demo_rrd(camera=cam)
    restarted = _restart_rerun_serve()
    now = _now_iso()
    prior = state.get("sim_viz", {{}})
    run_id = str(prior.get("run_id") or "").strip() or "franka-demo"
    viz = {{
        "run_id": run_id,
        "stage": "demo",
        "rrd_uri": f"file://{{target}}",
        "rrd_updated_at": now,
        "live_grpc_url": "",
        "mode": "static",
        "camera": cam,
        "preview_camera": cam,
        "preview_entity": f"world/cameras/{{cam}}",
        "rerun_ready": target.is_file() and _rerun_web_viewer_healthy(),
        "rerun_iframe_url": f"/rerun/?url=/rerun/recordings/sim2real.rrd&camera={{cam}}",
    }}
    state["sim_viz"] = viz
    _record_sim_viz_run(state, viz)
    _save_state(state)
    return viz

LLM_MODEL = os.environ.get("NPA_AGENT_LLM_MODEL", "{DEFAULT_LLM_MODEL}")
TF_BASE_URL = os.environ.get(
    "NEBIUS_TOKEN_FACTORY_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"
).rstrip("/")
_THINK_RE = re.compile(
    r"\\A\\s*<think>(?P<reasoning>.*?)</think>\\s*", re.DOTALL
)

def _agent_system_prompt() -> str:
    lines = [
        "You are the NPA workbench assistant on a Nebius Physical AI agent VM.",
        "Help operators configure NPA: provision infrastructure, Cosmos3, S3 storage,",
        "workflows, sim assets, and Sim2Real runs. Be concise and actionable.",
        "",
        "Agent HTTP APIs on this VM (same-origin relative paths; nginx proxies /api/):",
        "- GET /api/sim-assets, /api/sim-assets/selection, /api/sim-assets/cameras",
        "- GET /api/sim-viz/status — active run + .rrd URI for the Rerun iframe at /rerun/",
        "- GET /api/sim-viz/recordings — list available .rrd recording files for quick viewer switching",
        "- GET /api/sim-viz/runs — list run-scoped history (run_id, stage, camera, rrd_uri)",
        "- POST /api/sim-viz/load-run — switch active run context by run_id",
        "- GET /api/artifacts/runs?prefix=&limit= — discover run prefixes from object storage (no workflow allowlist)",
        "- GET /api/artifacts/run/{{run_id}} — list every object for a run with render hints",
        "- POST /api/sim-viz/load-artifact — load explicit s3_uri (or run_id+key) into viewer/download",
        "- POST /api/sim-viz/load-franka-demo — load stock Franka tabletop demo into Rerun",
        "- POST /api/workflows/sim2real/submit — submit Sim2Real with current asset selection",
        "- GET/POST /api/workflows/draft — workflow YAML draft in session",
        "- POST /api/workflows/validate — validate npa.workflow/v0.0.1 or npa.workflow/v0.0.1-beta YAML",
        "- POST /api/workflows/plan — dry-run plan-spec for workflow YAML",
        "- POST /api/workflows/submit — validate + plan workflow YAML (validate-only on agent VM)",
        "- GET /api/tools — workbench toolRef catalog",
        "",
        "To view Franka immediately, tell users to click **Load Franka in Rerun** in the Sim Assets panel",
        "(or POST /api/sim-viz/load-franka-demo). Open the embedded viewer at /rerun/.",
        "Artifact-first browsing flow: call `/api/artifacts/runs`, inspect `/api/artifacts/run/{{id}}`,",
        "then `POST /api/sim-viz/load-artifact` with explicit `s3_uri` or `run_id` + `key`.",
        "The **Cameras** panel is the center column below chat: stock workspace and wrist cameras",
        "with 2D frustum schematics, selection, and **Preview in Rerun**.",
        "Never suggest localhost, 127.0.0.1, or port 8080 — use relative /api/... paths or /rerun/.",
        "When asked about Sim2Real, workflow, or Rerun status, summarize run_id, stage, camera,",
        "rerun_ready, and latest_submit from session state — never reply with only a raw GET path.",
        "When generating workflow YAML, always emit canonical keys: apiVersion, kind, metadata, config,",
        "initial, and states. Never emit api_version, stages, or previous.outputs placeholders.",
        "",
        "Workbench toolRefs (invoke via npa workbench / npa.workflow on operator machine):",
    ]
    for key in TOOL_REFS:
        entry = TOOL_CATALOG.get(key, {{}})
        desc = entry.get("description", "")
        lines.append(f"- {{key}}: {{desc}}")
    lines.extend(
        [
            "",
            "Before Sim2Real submit, confirm scene/robot/camera selection.",
            "Always use real registry-qualified images from your Nebius container registry",
            "(or `NPA_REGISTRY` / `container_registry` in ~/.npa/config.yaml); never keep",
            "`<your-registry-id>` placeholders in runnable workflows.",
            "For BYOF solution onboarding, use the",
            "`npa/scripts/run_byof_repo.py` flow to containerize an OSS repo,",
            "push to the configured Nebius registry, then launch a real Isaac-Lab run",
            "with `--image` override on RT-core GPUs (L40S / RTX PRO 6000).",
            "For live infra runs, verify GPU compatibility first (`sky check`, `sky gpus list`)",
            "and loop submit attempts in tmux until validation+plan+prechecks pass.",
            "After submit, point users to /rerun/ and poll /api/sim-viz/status until rrd_uri is set.",
        ]
    )
    return "\\n".join(lines)

def _split_reasoning(message: dict) -> tuple[str, str | None]:
    content = message.get("content")
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    if reasoning is not None and not isinstance(reasoning, str):
        reasoning = str(reasoning)
    if isinstance(content, str):
        match = _THINK_RE.match(content)
        if match:
            visible = content[match.end() :].strip()
            trace = (match.group("reasoning").strip() or reasoning)
            return visible, trace
        return content.strip(), reasoning
    return "", (reasoning.strip() if reasoning else None)

def _token_factory_chat(*, messages: list, model: str | None = None) -> dict:
    api_key = os.environ.get("NEBIUS_TOKEN_FACTORY_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=503, detail="Token Factory API key not configured on agent VM")
    url = f"{{TF_BASE_URL}}/chat/completions"
    payload = {{
        "model": model or LLM_MODEL,
        "messages": messages,
        "temperature": 0.2,
    }}
    try:
        response = httpx.post(
            url,
            headers={{
                "Authorization": f"Bearer {{api_key}}",
                "Content-Type": "application/json",
            }},
            json=payload,
            timeout=120.0,
        )
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Token Factory request failed: {{exc}}") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="Token Factory returned non-object response")
    return data

{agent_chat_source}

{agent_workflow_source}

{agent_artifacts_source}

def _workflow_draft_from_state(state: dict) -> dict:
    draft = state.get("workflow_draft", {{}})
    return draft if isinstance(draft, dict) else {{}}

def _save_workflow_draft(
    state: dict,
    yaml_text: str,
    validation: dict,
    *,
    plan: dict | None = None,
    runnable: bool | None = None,
) -> dict:
    resolved_plan = plan if isinstance(plan, dict) else {{}}
    resolved_runnable = bool(runnable) if runnable is not None else bool(validation.get("ok") and resolved_plan.get("ok"))
    draft = {{
        "yaml": yaml_text,
        "validation": validation if isinstance(validation, dict) else {{}},
        "plan": resolved_plan,
        "runnable": resolved_runnable,
        "updated_at": _now_iso(),
        "name": str((validation or {{}}).get("name") or ""),
        "status": str((validation or {{}}).get("status") or ""),
        "states": (validation or {{}}).get("states") or [],
    }}
    state["workflow_draft"] = draft
    _save_state(state)
    return draft

def _record_sim_viz_run(state: dict, record: dict) -> None:
    if not isinstance(record, dict):
        return
    run_id = str(record.get("run_id") or "").strip()
    if not run_id:
        return
    entries = state.get("sim_viz_runs")
    if not isinstance(entries, dict):
        entries = {{}}
    snapshot = dict(DEFAULT_SIM_VIZ)
    snapshot.update(record)
    snapshot["run_id"] = run_id
    entries[run_id] = snapshot
    state["sim_viz_runs"] = entries
    state["active_run_id"] = run_id

def _sim_viz_runs(state: dict) -> list[dict]:
    runs = state.get("sim_viz_runs")
    if not isinstance(runs, dict):
        return []
    snapshots: list[dict] = []
    for run_id, item in runs.items():
        if not isinstance(item, dict):
            continue
        snapshot = dict(DEFAULT_SIM_VIZ)
        snapshot.update(item)
        snapshot["run_id"] = str(item.get("run_id") or run_id or "").strip()
        if not snapshot["run_id"]:
            continue
        snapshots.append(snapshot)
    return sorted(
        snapshots,
        key=lambda item: (
            str(item.get("rrd_updated_at") or ""),
            str(item.get("run_id") or ""),
        ),
        reverse=True,
    )

def _resolve_workflow_yaml(payload: dict) -> str:
    yaml_text = str(payload.get("yaml") or "").strip()
    if yaml_text:
        return yaml_text
    draft = _workflow_draft_from_state(_load_state())
    return str(draft.get("yaml") or "").strip()

def _last_user_message(raw_messages: list) -> str:
    for item in reversed(raw_messages):
        if not isinstance(item, dict):
            continue
        if str(item.get("role", "")).strip() == "user":
            return str(item.get("content", "")).strip()
    return ""

def _maybe_toolground_chat_reply(user_text: str) -> tuple[str | None, list[str], str | None, dict | None]:
    intent = match_chat_intent(user_text)
    if not intent and re.search(r"\\bworkflow\\b.*\\b(?:yaml|spec)\\b", str(user_text or ""), re.IGNORECASE):
        intent = "create_workflow"
    if not intent:
        return None, [], None, None
    state = _load_state()
    loaded_now = False
    rerun_ready = None
    default_cameras = list(DEFAULT_SCENE_SPEC.get("cameras", {{}}).values())
    if intent == "load_franka":
        sim_viz = state.get("sim_viz", {{}})
        if not isinstance(sim_viz, dict):
            sim_viz = {{}}
        rerun_ready = _rerun_ready_state(rrd_uri=str(sim_viz.get("rrd_uri") or ""))
        if not rerun_ready:
            selected = state.get("camera_selection", ["workspace"])
            cam = str(selected[0] if isinstance(selected, list) and selected else "workspace")
            _wire_franka_demo(state, camera=cam)
            state = _load_state()
            loaded_now = True
            sim_viz = state.get("sim_viz", {{}})
            if not isinstance(sim_viz, dict):
                sim_viz = {{}}
            rerun_ready = _rerun_ready_state(rrd_uri=str(sim_viz.get("rrd_uri") or ""))
    elif intent in {"sim2real_status", "watch_sim"}:
        # Ground watch/status replies on the same payload exposed by
        # GET /api/sim-viz/status so chat mirrors the live iframe panel.
        try:
            live_status = sim_viz_status()
            if isinstance(live_status, dict):
                state["sim_viz"] = dict(live_status)
                _save_state(state)
        except Exception:
            live_status = None
        sim_viz = state.get("sim_viz", {{}})
        if not isinstance(sim_viz, dict) and isinstance(live_status, dict):
            sim_viz = dict(live_status)
        if not isinstance(sim_viz, dict):
            sim_viz = {{}}
        rerun_ready = _rerun_ready_state(rrd_uri=str(sim_viz.get("rrd_uri") or ""))
    elif intent in {{"create_workflow", "create_vlm_rl_workflow", "create_gate_workflow"}}:
        draft = generate_workflow_draft(
            user_text=user_text,
            intent=intent,
            tool_refs=frozenset(TOOL_REFS),
            capabilities={{"tool_refs": list(TOOL_REFS)}},
        )
        yaml_text = str(draft.get("yaml") or "").strip()
        validation = draft.get("validation") if isinstance(draft.get("validation"), dict) else {{}}
        plan = draft.get("plan") if isinstance(draft.get("plan"), dict) else {{}}
        runnable = bool(draft.get("runnable"))
        template = str(draft.get("template") or "two-step")
        _save_workflow_draft(state, yaml_text, validation, plan=plan, runnable=runnable)
        state["workflow_draft"]["template"] = template
        _save_state(state)
        if not runnable:
            fail_reason = str(validation.get("error") or plan.get("error") or "validate+plan gate did not pass")
            reply = (
                "**Could not generate runnable workflow YAML yet.**\\n"
                f"- **reason**: `{{fail_reason}}`\\n"
                "- Adjust your request or template details and retry;"
                " chat returns YAML only after both validation and planning succeed."
            )
            return reply, apis_for_intent(intent), None, {{"ok": False, "validation": validation, "plan": plan}}
        reply = format_workflow_chat_reply(yaml_text, validation, template=template, plan=plan, runnable=runnable)
        return reply, apis_for_intent(intent), yaml_text, validation
    reply = build_grounded_reply(
        intent,
        state,
        TOOL_REFS,
        rerun_ready=rerun_ready,
        loaded_franka_now=loaded_now,
        default_cameras=default_cameras,
    )
    return reply, apis_for_intent(intent), None, None

def _agent_chat_with_tools(*, raw_messages: list, model: str) -> dict | None:
    last_user = _last_user_message(raw_messages)
    if not last_user:
        return None
    tool_reply, apis_used, workflow_yaml, workflow_validation = _maybe_toolground_chat_reply(last_user)
    if not tool_reply:
        return None
    payload = {{
        "ok": True,
        "model": model,
        "reply": tool_reply,
        "reasoning": None,
        "grounded": True,
        "apis_used": apis_used,
    }}
    if workflow_yaml:
        payload["workflow_yaml"] = workflow_yaml
    if isinstance(workflow_validation, dict):
        payload["workflow_validation"] = workflow_validation
        draft = _workflow_draft_from_state(_load_state())
        if isinstance(draft, dict) and draft.get("yaml"):
            payload["workflow_draft"] = draft
    return payload

@app.post("/chat")
def chat(payload: dict):
    raw_messages = payload.get("messages", [])
    if not isinstance(raw_messages, list) or not raw_messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty list")
    model = str(payload.get("model") or LLM_MODEL).strip() or LLM_MODEL
    tool_result = _agent_chat_with_tools(raw_messages=raw_messages, model=model)
    if tool_result is not None:
        reply = str(tool_result.get("reply") or "").strip()
        state = _load_state()
        history: list[dict] = []
        for item in raw_messages:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "user")).strip() or "user"
            content = str(item.get("content", "")).strip()
            if role in {{"user", "assistant"}} and content:
                history.append({{"role": role, "content": content}})
        if reply:
            history.append({{"role": "assistant", "content": reply}})
        state["chat_history"] = history[-50:]
        _save_state(state)
        return tool_result
    live_ctx = format_live_context_block(_load_state())
    messages: list[dict] = [
        {{"role": "system", "content": _agent_system_prompt() + "\\n\\n" + live_ctx}}
    ]
    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "user")).strip() or "user"
        content = str(item.get("content", "")).strip()
        if content:
            messages.append({{"role": role, "content": content}})
    if len(messages) < 2:
        raise HTTPException(status_code=400, detail="at least one user message is required")
    data = _token_factory_chat(messages=messages, model=model)
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise HTTPException(status_code=502, detail="Token Factory response missing assistant message") from exc
    reply, reasoning = _split_reasoning(message)
    if not reply and reasoning:
        reply = reasoning
        reasoning = None
    state = _load_state()
    history: list[dict] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "user")).strip() or "user"
        content = str(item.get("content", "")).strip()
        if role in {{"user", "assistant"}} and content:
            history.append({{"role": role, "content": content}})
    if reply:
        history.append({{"role": "assistant", "content": reply}})
    state["chat_history"] = history[-50:]
    _save_state(state)
    return {{
        "ok": True,
        "model": model,
        "reply": reply,
        "reasoning": reasoning,
    }}

@app.get("/health")
def health():
    return {{"ok": True, "tool_refs": len(TOOL_REFS)}}

@app.get("/session")
def session_bootstrap():
    state = _load_state()
    sim_viz = dict(DEFAULT_SIM_VIZ)
    if isinstance(state.get("sim_viz"), dict):
        sim_viz.update(state["sim_viz"])
    selected = state.get("camera_selection", ["workspace"])
    camera = str(sim_viz.get("camera") or (selected[0] if isinstance(selected, list) and selected else "workspace"))
    sim_viz["camera"] = camera
    if not sim_viz.get("rrd_uri") and RRD_PATH.is_file():
        sim_viz["rrd_uri"] = f"file://{{RRD_PATH}}"
    sim_viz["rerun_ready"] = _rerun_ready_state(rrd_uri=str(sim_viz.get("rrd_uri") or ""))
    history = state.get("chat_history", [])
    if not isinstance(history, list):
        history = []
    return {{
        "selection": state.get("selection", dict(DEFAULT_SELECTION)),
        "sim_viz": sim_viz,
        "latest_submit": state.get("latest_submit", {{}}),
        "sim_viz_runs": _sim_viz_runs(state),
        "workflow_draft": _workflow_draft_from_state(state),
        "workflow_submit": state.get("workflow_submit", {{}}),
        "camera_selection": state.get("camera_selection", ["workspace"]),
        "chat_history": history,
    }}

@app.get("/tools")
def tools():
    return {{"tool_refs": TOOL_REFS}}

@app.get("/tools/{{tool_ref:path}}")
def tool(tool_ref: str):
    payload = TOOL_CATALOG.get(tool_ref)
    if payload is None:
        return {{"ok": False, "error": "unknown toolRef", "tool_ref": tool_ref}}
    return {{"ok": True, "tool_ref": tool_ref, **payload}}

@app.get("/sim-viz/status")
def sim_viz_status(run_id: str = ""):
    state = _load_state()
    payload = _sim_viz_for_run(state, run_id=run_id)
    selected = state.get("camera_selection", ["workspace"])
    camera = str(payload.get("camera") or (selected[0] if isinstance(selected, list) and selected else "workspace"))
    payload["camera"] = camera
    latest_submit = state.get("latest_submit", {{}})
    if not isinstance(latest_submit, dict):
        latest_submit = {{}}
    if not str(payload.get("run_id") or "").strip():
        payload["run_id"] = str(latest_submit.get("run_id") or "").strip()
    if str(payload.get("stage") or "idle").strip().lower() == "idle" and payload.get("run_id"):
        payload["stage"] = "submitted"
    _record_sim_viz_run(state, payload)
    if str(payload.get("artifact_render") or "").strip().lower() in {"", "rerun"}:
        payload["rerun_iframe_url"] = f"/rerun/?url=/rerun/recordings/sim2real.rrd&camera={{camera}}"
    if not payload.get("rrd_uri") and RRD_PATH.is_file():
        payload["rrd_uri"] = f"file://{{RRD_PATH}}"
    mode = str(payload.get("mode") or "static").strip().lower()
    payload["mode"] = "live" if mode == "live" else "static"
    payload["rerun_ready"] = _rerun_ready_state(rrd_uri=str(payload.get("rrd_uri") or ""))
    runs = state.get("sim_viz_runs")
    if isinstance(runs, dict):
        payload["available_run_ids"] = sorted(str(key) for key in runs.keys() if str(key).strip())
    else:
        payload["available_run_ids"] = []
    payload["active_run_id"] = str(state.get("active_run_id") or payload.get("run_id") or "").strip()
    _save_state(state)
    return payload

@app.get("/sim-viz/runs")
def sim_viz_runs():
    state = _load_state()
    active = sim_viz_status()
    runs = _sim_viz_runs(state)
    active_id = str(active.get("run_id") or "").strip()
    return {{
        "ok": True,
        "active_run_id": active_id,
        "runs": runs,
    }}

@app.post("/sim-viz/select-run")
def sim_viz_select_run(payload: dict | None = None):
    body = payload if isinstance(payload, dict) else {{}}
    requested_run = str(body.get("run_id") or "").strip()
    if not requested_run:
        raise HTTPException(status_code=400, detail="run_id is required")
    state = _load_state()
    runs = _sim_viz_runs(state)
    selected = next((item for item in runs if str(item.get("run_id") or "").strip() == requested_run), None)
    if not isinstance(selected, dict):
        raise HTTPException(status_code=404, detail=f"run_id not found: {{requested_run}}")
    sim_viz = dict(DEFAULT_SIM_VIZ)
    if isinstance(state.get("sim_viz"), dict):
        sim_viz.update(state["sim_viz"])
    sim_viz.update(
        {{
            "run_id": requested_run,
            "stage": str(selected.get("stage") or sim_viz.get("stage") or "submitted"),
            "rrd_uri": str(selected.get("rrd_uri") or sim_viz.get("rrd_uri") or ""),
            "rrd_updated_at": str(selected.get("rrd_updated_at") or sim_viz.get("rrd_updated_at") or ""),
            "camera": str(selected.get("camera") or sim_viz.get("camera") or "workspace"),
        }}
    )
    state["sim_viz"] = sim_viz
    state["latest_submit"] = {{
        "run_id": requested_run,
        "submitted_at": str(selected.get("submitted_at") or _now_iso()),
        "submit_mode": str(selected.get("submit_mode") or "history-select"),
    }}
    _save_state(state)
    return {{"ok": True, "sim_viz": sim_viz_status(), "selected": selected}}

@app.post("/sim-viz/load-run")
def sim_viz_load_run(payload: dict | None = None):
    body = payload if isinstance(payload, dict) else {{}}
    run_id = str(body.get("run_id") or "").strip()
    if not run_id:
        raise HTTPException(status_code=400, detail="run_id is required")
    state = _load_state()
    runs = state.get("sim_viz_runs")
    if not isinstance(runs, dict):
        runs = {{}}
    selected = runs.get(run_id)
    if not isinstance(selected, dict):
        selected = {{}}
    rrd_uri = str(body.get("rrd_uri") or "").strip()
    if rrd_uri:
        selected["rrd_uri"] = rrd_uri
    camera = str(body.get("camera") or "").strip()
    if camera:
        selected["camera"] = camera
    stage = str(body.get("stage") or "").strip()
    if stage:
        selected["stage"] = stage
    mode = str(body.get("mode") or "").strip().lower()
    if mode in {{"static", "live"}}:
        selected["mode"] = mode
    if not selected:
        raise HTTPException(status_code=404, detail=f"run_id not found: {{run_id}}")
    selected["run_id"] = run_id
    selected["rrd_updated_at"] = _now_iso()
    state["sim_viz"] = selected
    _record_sim_viz_run(state, selected)
    _save_state(state)
    return {{
        "ok": True,
        "sim_viz": sim_viz_status(run_id=run_id),
    }}

@app.get("/sim-viz/recordings")
def sim_viz_recordings():
    # List available .rrd recording files in /opt/npa-agent/recordings/ for quick viewer switching.
    recordings_dir = Path("/opt/npa-agent/recordings")
    result = []
    if recordings_dir.is_dir():
        for rrd_file in sorted(recordings_dir.glob("*.rrd"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                stat = rrd_file.stat()
                result.append({{
                    "name": rrd_file.name,
                    "path": f"/rerun/recordings/{{rrd_file.name}}",
                    "size_bytes": stat.st_size,
                    "updated_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                    "active": rrd_file.name == "sim2real.rrd",
                }})
            except OSError:
                continue
    return {{"recordings": result, "count": len(result)}}


@app.get("/artifacts/runs")
def artifacts_runs(prefix: str = "", limit: int = 50):
    try:
        s3, settings = _agent_s3_client()
        page = list_runs(
            settings["bucket"],
            prefix=prefix,
            limit=limit,
            s3=s3,
        )
        return {{"ok": True, "bucket": settings["bucket"], "prefix": prefix, **page.to_dict()}}
    except HTTPException:
        raise
    except Exception as exc:
        return JSONResponse(status_code=502, content={{"ok": False, "error": str(exc)}})


@app.get("/artifacts/run/{{run_id:path}}")
def artifacts_for_run(run_id: str, prefix: str = ""):
    try:
        normalized_run = validate_run_id(run_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        s3, settings = _agent_s3_client()
        artifacts = list_artifacts(
            settings["bucket"],
            normalized_run,
            prefix=prefix,
            s3=s3,
        )
        preferred = select_preferred_artifact(artifacts)
        return {{
            "ok": True,
            "bucket": settings["bucket"],
            "run_id": normalized_run,
            "count": len(artifacts),
            "artifacts": [item.to_dict() for item in artifacts],
            "preferred": preferred.to_dict() if preferred else None,
        }}
    except HTTPException:
        raise
    except Exception as exc:
        return JSONResponse(status_code=502, content={{"ok": False, "error": str(exc)}})


@app.get("/artifacts/file/{{filename}}")
def artifact_file(filename: str):
    safe_name = Path(str(filename)).name
    if safe_name != filename:
        raise HTTPException(status_code=400, detail="invalid artifact filename")
    target = RECORDINGS_DIR / safe_name
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"artifact file not found: {{filename}}")
    return FileResponse(str(target), media_type="application/octet-stream")


@app.post("/sim-viz/load-artifact")
def sim_viz_load_artifact(payload: dict | None = None):
    body = payload if isinstance(payload, dict) else {{}}
    requested_uri = str(body.get("s3_uri") or "").strip()
    requested_run = str(body.get("run_id") or "").strip()
    requested_key = str(body.get("key") or "").strip()
    if not requested_uri and not (requested_run and requested_key):
        raise HTTPException(status_code=400, detail="Provide either s3_uri or run_id + key")
    try:
        s3, settings = _agent_s3_client()
        if requested_uri:
            bucket, key = parse_s3_uri(requested_uri)
            run_guess = str(body.get("run_id") or _run_id_for_key(key, ""))
            run_id = validate_run_id(run_guess) if run_guess else "artifact"
            s3_uri = requested_uri
        else:
            run_id = validate_run_id(requested_run)
            key = _safe_artifact_key(requested_key)
            bucket = settings["bucket"]
            s3_uri = f"s3://{{bucket}}/{{key}}"
        local_name = _artifact_filename(key)
        local_path = RECORDINGS_DIR / local_name
        download_s3_uri(s3_uri, local_path, s3=s3)
        render = render_hint_for_object(key=key)
        state = _load_state()
        sim_viz = _apply_loaded_artifact(
            state=state,
            run_id=run_id,
            key=key,
            s3_uri=s3_uri,
            render=render,
            local_path=local_path,
        )
        return {{"ok": True, "sim_viz": sim_viz, "render": render, "artifact_uri": s3_uri}}
    except HTTPException:
        raise
    except Exception as exc:
        return JSONResponse(status_code=502, content={{"ok": False, "error": str(exc)}})


@app.post("/sim-viz/load-franka-demo")
def load_franka_demo(payload: dict | None = None):
    body = payload if isinstance(payload, dict) else {{}}
    camera = str(body.get("camera") or "").strip()
    if not camera:
        state = _load_state()
        selected = state.get("camera_selection", ["workspace"])
        if isinstance(selected, list) and selected:
            camera = str(selected[0])
        else:
            camera = "workspace"
    state = _load_state()
    viz = _wire_franka_demo(state, camera=camera)
    return {{"ok": True, "sim_viz": viz, "selection": state["selection"]}}

@app.post("/sim-viz/camera-preview")
def sim_viz_camera_preview(payload: dict | None = None):
    body = payload if isinstance(payload, dict) else {{}}
    camera = str(body.get("camera") or "").strip()
    if not camera:
        state = _load_state()
        selected = state.get("camera_selection", ["workspace"])
        if isinstance(selected, list) and selected:
            camera = str(selected[0])
        else:
            camera = "workspace"
    cameras = DEFAULT_SCENE_SPEC.get("cameras", {{}})
    if camera not in cameras:
        raise HTTPException(status_code=404, detail=f"unknown camera: {{camera}}")
    state = _load_state()
    viz = _wire_franka_demo(state, camera=camera)
    entity_path = f"world/cameras/{{camera}}"
    return {{
        "ok": True,
        "camera": camera,
        "entity_path": entity_path,
        "rollout_entity_guess": f"rollouts/latest/{{camera}}/camera",
        "sim_viz": viz,
        "hint": "Open the Rerun panel and expand world/cameras/<name>.",
    }}

def _sim_viz_rrd_file_response(run_id: str = ""):
    state = _load_state()
    sim_viz = _sim_viz_for_run(state, run_id=run_id)
    uri = str(sim_viz.get("rrd_uri") or "").strip()
    if uri.startswith("file://"):
        file_path = Path(uri[len("file://"):])
        if file_path.is_file():
            return FileResponse(str(file_path), media_type="application/octet-stream")
    if uri.startswith("http://") or uri.startswith("https://"):
        try:
            proxied = httpx.get(uri, timeout=20.0)
            proxied.raise_for_status()
            return Response(content=proxied.content, media_type="application/octet-stream")
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Unable to fetch remote sim2real.rrd: {{exc}}") from exc
    if RRD_PATH.is_file():
        return FileResponse(str(RRD_PATH), media_type="application/octet-stream")
    raise HTTPException(status_code=404, detail="No sim2real.rrd file on disk yet")

@app.get("/sim-viz/rrd")
def sim_viz_rrd(run_id: str = ""):
    return _sim_viz_rrd_file_response(run_id=run_id)

@app.get("/sim-viz/rrd-blob")
def sim_viz_rrd_blob(run_id: str = ""):
    # Authenticated .rrd bytes for parent-page blob URL (Rerun wasm cannot send basic auth).
    return _sim_viz_rrd_file_response(run_id=run_id)

@app.on_event("startup")
def _boot_preload_sim_viz() -> None:
    if not RRD_PATH.is_file():
        return
    _publish_rrd_recording(RRD_PATH)
    state = _load_state()
    sim_viz = state.get("sim_viz", {{}})
    if not isinstance(sim_viz, dict):
        sim_viz = {{}}
    if str(sim_viz.get("rrd_uri") or "").strip():
        return
    selected = state.get("camera_selection", ["workspace"])
    cam = str(selected[0] if isinstance(selected, list) and selected else "workspace")
    run_id = str(sim_viz.get("run_id") or "").strip() or "franka-demo"
    now = _now_iso()
    state["sim_viz"] = {{
        "run_id": run_id,
        "stage": "demo",
        "rrd_uri": f"file://{{RRD_PATH}}",
        "rrd_updated_at": now,
        "live_grpc_url": "",
        "mode": "static",
        "camera": cam,
        "preview_camera": cam,
        "preview_entity": f"world/cameras/{{cam}}",
        "rerun_ready": _rerun_ready_state(rrd_uri=f"file://{{RRD_PATH}}"),
        "rerun_iframe_url": f"/rerun/?url=/rerun/recordings/sim2real.rrd&camera={{cam}}",
    }}
    _record_sim_viz_run(state, state["sim_viz"])
    _save_state(state)

@app.get("/sim-assets")
def sim_assets():
    state = _load_state()
    selection = state.get("selection", {{}})
    if not isinstance(selection, dict):
        selection = dict(DEFAULT_SELECTION)
    return {{
        "scene_spec": DEFAULT_SCENE_SPEC,
        "robot_spec": DEFAULT_ROBOT_SPEC,
        "assets_manifest": DEFAULT_ASSETS_MANIFEST,
        "selection": selection,
        "resolved_uris": {{
            "scene_spec_uri": selection.get("scene_spec_uri", ""),
            "assets_uri": selection.get("assets_uri", ""),
            "robot_spec_uri": selection.get("robot_spec_uri", ""),
            "cameras_uri": selection.get("cameras_uri", ""),
        }},
    }}

@app.get("/sim-assets/catalog")
def sim_assets_catalog():
    return {{
        "entries": [
            {{"name": "stock_scene", "uri": "stock://scene/default"}},
            {{"name": "stock_robot_franka", "uri": "stock://robot/franka"}},
            {{"name": "customer_assets_root", "uri": "s3://customer-assets/"}},
        ]
    }}

@app.get("/sim-assets/cameras")
def sim_assets_cameras():
    state = _load_state()
    selected = state.get("camera_selection", ["workspace"])
    cameras = []
    for entry in list(DEFAULT_SCENE_SPEC["cameras"].values()):
        if not isinstance(entry, dict):
            continue
        camera_name = str(entry.get("name") or "").strip()
        camera_payload = dict(entry)
        if camera_name:
            camera_payload["preview_url"] = f"/api/sim-viz/camera-preview?camera={{camera_name}}"
        cameras.append(camera_payload)
    return {{"cameras": cameras, "selected": selected}}

@app.put("/sim-assets/cameras/selection")
def set_camera_selection(payload: dict):
    selected = payload.get("selected", [])
    if not isinstance(selected, list):
        raise HTTPException(status_code=400, detail="selected must be a list")
    state = _load_state()
    state["camera_selection"] = [str(item) for item in selected if str(item).strip()]
    cam = state["camera_selection"][0] if state["camera_selection"] else "workspace"
    preset = str((state.get("selection") or {{}}).get("robot_preset", "")).strip().lower()
    if preset == "franka":
        viz = _wire_franka_demo(state, camera=cam)
        return {{"selected": state["camera_selection"], "sim_viz": viz}}
    _save_state(state)
    return {{"selected": state["camera_selection"]}}

@app.post("/sim-assets/selection")
def set_sim_assets_selection(payload: dict):
    state = _load_state()
    selection = dict(DEFAULT_SELECTION)
    current = state.get("selection", {{}})
    if isinstance(current, dict):
        selection.update(current)
    for key in ("scene_spec_uri", "assets_uri", "robot_spec_uri", "cameras_uri", "robot_preset", "sim_backend"):
        if key in payload and payload[key] is not None:
            selection[key] = str(payload[key]).strip()
    if "props" in payload and isinstance(payload["props"], list):
        selection["props"] = [str(item) for item in payload["props"] if str(item).strip()]
    state["selection"] = selection
    preset = str(selection.get("robot_preset", "")).strip().lower()
    if preset == "franka":
        cam = str((state.get("camera_selection") or ["workspace"])[0])
        viz = _wire_franka_demo(state, camera=cam)
        return {{"ok": True, "selection": selection, "sim_viz": viz}}
    _save_state(state)
    return {{"ok": True, "selection": selection}}

@app.get("/sim-assets/selection")
def get_sim_assets_selection():
    state = _load_state()
    selection = state.get("selection", {{}})
    if not isinstance(selection, dict):
        selection = dict(DEFAULT_SELECTION)
    return selection

@app.get("/workflows/sim2real/status")
def sim2real_status():
    state = _load_state()
    latest = state.get("latest_submit", {{}})
    sim_viz = state.get("sim_viz", {{}})
    return {{
        "ok": True,
        "latest_submit": latest if isinstance(latest, dict) else {{}},
        "sim_viz": sim_viz if isinstance(sim_viz, dict) else dict(DEFAULT_SIM_VIZ),
    }}

@app.get("/workbench/actions")
def workbench_actions():
    return {{
        "actions": [
            {{
                "id": "configure_s3",
                "title": "Configure S3",
                "hint": "Run `npa configure` on operator machine to set storage credentials.",
            }},
            {{
                "id": "setup_cosmos",
                "title": "Setup Cosmos3",
                "hint": "Use `npa workbench cosmos check|fetch` before inference workflows.",
            }},
            {{
                "id": "submit_sim2real",
                "title": "Submit Sim2Real",
                "hint": "POST /api/workflows/sim2real/submit after confirming selection.",
            }},
            {{
                "id": "watch_sim",
                "title": "Watch sim",
                "hint": "GET /api/sim-viz/status and open /rerun/ iframe.",
            }},
        ]
    }}


@app.get("/workflows/draft")
@app.get("/workflows/npa/draft")
def get_workflow_draft():
    state = _load_state()
    draft = _workflow_draft_from_state(state)
    return {{"ok": True, "draft": draft}}

@app.post("/workflows/draft")
@app.put("/workflows/npa/draft")
def save_workflow_draft(payload: dict):
    body = payload if isinstance(payload, dict) else {{}}
    yaml_text = str(body.get("yaml") or "").strip()
    if not yaml_text:
        raise HTTPException(status_code=400, detail="yaml is required")
    validation = validate_workflow_yaml_text(yaml_text, tool_refs=frozenset(TOOL_REFS))
    plan = (
        plan_workflow_yaml_text(yaml_text, run_id="draft-save", tool_refs=frozenset(TOOL_REFS))
        if validation.get("ok")
        else {{"ok": False, "error": str(validation.get("error") or "validation failed")}}
    )
    runnable = bool(validation.get("ok") and plan.get("ok"))
    state = _load_state()
    draft = _save_workflow_draft(state, yaml_text, validation, plan=plan, runnable=runnable)
    return {{"ok": runnable, "draft": draft, "validation": validation, "plan": plan, "runnable": runnable}}

@app.post("/workflows/validate")
@app.post("/workflows/npa/validate")
def validate_workflow(payload: dict):
    body = payload if isinstance(payload, dict) else {{}}
    yaml_text = _resolve_workflow_yaml(body)
    if not yaml_text:
        raise HTTPException(status_code=400, detail="yaml is required")
    validation = validate_workflow_yaml_text(yaml_text, tool_refs=frozenset(TOOL_REFS))
    plan = (
        plan_workflow_yaml_text(yaml_text, run_id="validate-check", tool_refs=frozenset(TOOL_REFS))
        if validation.get("ok")
        else {{"ok": False, "error": str(validation.get("error") or "validation failed")}}
    )
    runnable = bool(validation.get("ok") and plan.get("ok"))
    state = _load_state()
    _save_workflow_draft(state, yaml_text, validation, plan=plan, runnable=runnable)
    return {{"ok": runnable, "validation": validation, "plan": plan, "runnable": runnable}}

@app.post("/workflows/plan")
@app.post("/workflows/npa/plan")
def plan_workflow(payload: dict):
    body = payload if isinstance(payload, dict) else {{}}
    yaml_text = _resolve_workflow_yaml(body)
    if not yaml_text:
        raise HTTPException(status_code=400, detail="yaml is required")
    run_id = str(body.get("run_id") or "").strip()
    assume_decision = str(body.get("assume_decision") or "").strip()
    plan = plan_workflow_yaml_text(
        yaml_text,
        run_id=run_id,
        assume_decision=assume_decision,
        tool_refs=frozenset(TOOL_REFS),
    )
    if not plan.get("ok"):
        raise HTTPException(status_code=400, detail=str(plan.get("error") or "plan failed"))
    return {{"ok": True, "plan": plan, "yaml": yaml_text}}

@app.post("/workflows/submit")
@app.post("/workflows/npa/submit")
def submit_npa_workflow(payload: dict):
    body = payload if isinstance(payload, dict) else {{}}
    yaml_text = _resolve_workflow_yaml(body)
    if not yaml_text:
        raise HTTPException(status_code=400, detail="yaml is required")
    validation = validate_workflow_yaml_text(yaml_text, tool_refs=frozenset(TOOL_REFS))
    if not validation.get("ok"):
        raise HTTPException(status_code=400, detail=str(validation.get("error") or "validation failed"))
    run_id = str(body.get("run_id") or f"agent-wf-{{secrets.token_hex(6)}}")
    assume_decision = str(body.get("assume_decision") or "").strip()
    plan = plan_workflow_yaml_text(
        yaml_text,
        run_id=run_id,
        assume_decision=assume_decision,
        tool_refs=frozenset(TOOL_REFS),
    )
    if not plan.get("ok"):
        raise HTTPException(status_code=400, detail=str(plan.get("error") or "plan failed"))
    state = _load_state()
    _save_workflow_draft(state, yaml_text, validation, plan=plan, runnable=True)
    submit_record = {{
        "run_id": run_id,
        "submitted_at": _now_iso(),
        "name": str(validation.get("name") or ""),
        "validation": validation,
        "plan": plan,
        "submit_mode": "validate-only",
        "note": "Agent VM records validate+plan; live SkyPilot submit runs on operator machine.",
    }}
    state["workflow_submit"] = submit_record
    state["latest_submit"] = {{
        "run_id": run_id,
        "submitted_at": submit_record["submitted_at"],
        "workflow_name": str(validation.get("name") or ""),
        "submit_mode": "validate-only",
    }}
    _record_sim_viz_run(
        state,
        {{
            "run_id": run_id,
            "submitted_at": submit_record["submitted_at"],
            "stage": "submitted",
            "camera": str((state.get("sim_viz", {{}}) or {{}}).get("camera") or "workspace"),
            "rrd_uri": str((state.get("sim_viz", {{}}) or {{}}).get("rrd_uri") or ""),
            "rrd_updated_at": str((state.get("sim_viz", {{}}) or {{}}).get("rrd_updated_at") or ""),
            "submit_mode": "validate-only",
            "workflow_name": str(validation.get("name") or ""),
        }},
    )
    _save_state(state)
    return {{"ok": True, **submit_record}}

@app.post("/workflows/sim2real/submit")
def submit_sim2real(payload: dict):
    state = _load_state()
    selection = state.get("selection", {{}})
    if not isinstance(selection, dict):
        selection = dict(DEFAULT_SELECTION)
    run_id = f"agent-run-{{secrets.token_hex(6)}}"
    env_block = {{
        "NPA_SIM2REAL_SCENE_SPEC_URI": selection.get("scene_spec_uri", ""),
        "NPA_SIM2REAL_ASSETS_URI": selection.get("assets_uri", ""),
        "NPA_SIM2REAL_CAMERAS_URI": selection.get("cameras_uri", ""),
        "NPA_SIM2REAL_ROBOT_SPEC_URI": selection.get("robot_spec_uri", ""),
        "NPA_SIM2REAL_ROBOT_PRESET": selection.get("robot_preset", "franka"),
        "NPA_SIM2REAL_SIM_BACKEND": selection.get("sim_backend", "isaac") or "isaac",
    }}
    state["latest_submit"] = {{
        "run_id": run_id,
        "submitted_at": _now_iso(),
        "selection": selection,
        "env": env_block,
    }}
    state["sim_viz"] = {{
        "run_id": run_id,
        "stage": "submitted",
        "rrd_uri": "",
        "rrd_updated_at": _now_iso(),
        "live_grpc_url": "",
        "mode": "static",
    }}
    _record_sim_viz_run(
        state,
        {{
            "run_id": run_id,
            "submitted_at": state["latest_submit"]["submitted_at"],
            "stage": "submitted",
            "camera": str((state.get("sim_viz", {{}}) or {{}}).get("camera") or "workspace"),
            "rrd_uri": "",
            "rrd_updated_at": str((state.get("sim_viz", {{}}) or {{}}).get("rrd_updated_at") or ""),
            "submit_mode": "sim2real",
            "workflow_name": "sim2real",
        }},
    )
    _save_state(state)
    return {{"ok": True, "run_id": run_id, "selection": selection, "env": env_block}}
PY
cat <<'PY' | sudo tee /opt/npa-agent/bootstrap_rrd.py >/dev/null
import math
from pathlib import Path

import rerun as rr

_FRANKA_HOME = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)

def _franka_joint_positions(joint_angles):
    dh = [
        (0.0, 0.0, 0.333),
        (0.0, -math.pi / 2.0, 0.0),
        (0.0, math.pi / 2.0, 0.316),
        (0.0825, math.pi / 2.0, 0.0),
        (-0.0825, -math.pi / 2.0, 0.384),
        (0.0, math.pi / 2.0, 0.0),
        (0.088, math.pi / 2.0, 0.0),
    ]

    def _matmul(a, b):
        return [
            [sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)
        ]

    transform = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    positions = [[0.0, 0.0, 0.0]]
    for index, (a, alpha, d) in enumerate(dh):
        theta = float(joint_angles[index])
        ct, st = math.cos(theta), math.sin(theta)
        ca, sa = math.cos(alpha), math.sin(alpha)
        step = [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
        transform = _matmul(transform, step)
        positions.append([transform[0][3], transform[1][3], transform[2][3]])
    ee = [transform[0][3], transform[1][3], transform[2][3] + 0.103]
    positions.append(ee)
    positions.append([ee[0], ee[1] + 0.04, ee[2]])
    positions.append([ee[0], ee[1] - 0.04, ee[2]])
    return positions

def _log_franka_robot_geometry():
    positions = _franka_joint_positions(_FRANKA_HOME)
    arm_points = positions[:8]
    segments = []
    for left, right in zip(arm_points, arm_points[1:]):
        dx = left[0] - right[0]
        dy = left[1] - right[1]
        dz = left[2] - right[2]
        if dx * dx + dy * dy + dz * dz < 1e-8:
            continue
        segments.append([left, right])
    link_color = [234, 88, 12]
    link_rgba = link_color + [255]
    rr.log(
        "robot/franka/base",
        rr.Boxes3D(
            centers=[[0.0, 0.0, 0.05]],
            half_sizes=[[0.085, 0.085, 0.05]],
            colors=[[100, 116, 139, 255]],
        ),
    )
    rr.log(
        "robot/franka/joints",
        rr.Points3D(
            arm_points,
            colors=[link_rgba] * len(arm_points),
            radii=[0.028] * len(arm_points),
        ),
    )
    if segments:
        rr.log(
            "robot/franka/links",
            rr.LineStrips3D(
                segments,
                colors=[link_color] * len(segments),
                radii=[0.018] * len(segments),
            ),
        )
    gripper_segments = [
        [positions[7], positions[8]],
        [positions[8], positions[9]],
        [positions[8], positions[10]],
    ]
    gripper_color = [59, 130, 246]
    rr.log(
        "robot/franka/gripper",
        rr.LineStrips3D(
            gripper_segments,
            colors=[gripper_color] * len(gripper_segments),
            radii=[0.012] * len(gripper_segments),
        ),
    )
    rr.log(
        "robot/franka",
        rr.TextDocument("Franka Panda — stock tabletop demo (bootstrap)"),
    )

target = Path("/opt/npa-agent/sim2real.rrd")
target.parent.mkdir(parents=True, exist_ok=True)
rr.init("npa-franka-tabletop-demo", spawn=False)
rr.log(
    "world/table",
    rr.Boxes3D(
        centers=[[0.5, 0.0, 0.0]],
        half_sizes=[[0.4, 0.3, 0.02]],
        colors=[[180, 180, 180, 255]],
    ),
)
rr.log(
    "world/cube",
    rr.Boxes3D(
        centers=[[0.5, 0.3, 0.04]],
        half_sizes=[[0.025, 0.025, 0.025]],
        colors=[[59, 130, 246, 255]],
    ),
)
_log_franka_robot_geometry()
rr.log("cameras/workspace", rr.Pinhole(fov_y=60.0))
rr.log("cameras/wrist", rr.Pinhole(fov_y=90.0))
rr.save(str(target))
from pathlib import Path as _Path
import shutil as _shutil
_rec = _Path("/opt/npa-agent/recordings/sim2real.rrd")
_rec.parent.mkdir(parents=True, exist_ok=True)
_shutil.copy2(target, _rec)
PY
sudo mkdir -p /opt/npa-agent/recordings
sudo cp -f /opt/npa-agent/sim2real.rrd /opt/npa-agent/recordings/sim2real.rrd || true
cat <<'WELCOME' | sudo tee /opt/npa-agent/welcome.html >/dev/null
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>NPA Agent — welcome</title>
    <style>
      body {{ font-family: system-ui, sans-serif; max-width: 640px; margin: 48px auto; padding: 0 16px; line-height: 1.5; color: #1f2430; }}
      h1 {{ font-size: 1.4rem; }}
      h2 {{ font-size: 1.1rem; margin-bottom: 0.5rem; }}
      code, pre {{ background: #f0f2f5; padding: 2px 6px; border-radius: 4px; }}
      .ok {{ color: #18794e; }}
      .muted {{ color: #5f6573; font-size: 0.95rem; }}
      .sign-in-panel {{ margin: 24px 0; padding: 16px; border: 1px solid #e0e0e0; border-radius: 8px; background: #fafbfc; }}
      .sign-in {{ display: grid; gap: 10px; max-width: 360px; }}
      .sign-in label {{ font-weight: 600; font-size: 0.9rem; }}
      .sign-in input {{ padding: 8px 10px; border: 1px solid #c8ccd4; border-radius: 6px; font: inherit; }}
      .sign-in button {{ justify-self: start; padding: 8px 16px; border: 0; border-radius: 6px; background: #5e43f3; color: #fff; font: inherit; font-weight: 600; cursor: pointer; }}
      a {{ color: #5e43f3; }}
    </style>
  </head>
  <body>
    <h1>NPA Agent is running</h1>
    <p class="ok">This page is public (no login). The workbench UI at <code>/</code> is protected by HTTP Basic Auth.</p>
{strip_url_credentials_js}
{login_form_html}
    <ol>
      <li>Enter your password above and click <strong>Sign in</strong>, or open <a href="/">the workbench UI</a> if your browser shows the auth dialog.</li>
      <li>Username: <code>{auth_user}</code></li>
      <li>Password: from your operator&apos;s deploy output (<code>auth_password</code>) or <code>auth.env</code> on the machine that ran <code>npa agent deploy</code>.</li>
      <li>Customer URL: use <code>https://</code> on port <strong>443</strong> (no VPN or SSH tunnel). Your browser may warn about a self-signed certificate — choose to proceed.</li>
      <li>More help: <a href="/login-help.html">login help</a></li>
    </ol>
    <p>Health check (no auth): <a href="/healthz">/healthz</a></p>
    <p>UI version after login: check <code>&lt;meta name="npa-ui-version"&gt;</code> — expect <code>{AGENT_UI_VERSION}</code>.</p>
  </body>
</html>
WELCOME
cat <<'LOGINHELP' | sudo tee /opt/npa-agent/login-help.html >/dev/null
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Login required — NPA Agent</title>
    <style>
      body {{ font-family: system-ui, sans-serif; max-width: 640px; margin: 48px auto; padding: 0 16px; line-height: 1.5; color: #1f2430; }}
      h1 {{ font-size: 1.4rem; }}
      h2 {{ font-size: 1.1rem; margin-bottom: 0.5rem; }}
      code {{ background: #f0f2f5; padding: 2px 6px; border-radius: 4px; }}
      .muted {{ color: #5f6573; font-size: 0.95rem; }}
      .sign-in-panel {{ margin: 24px 0; padding: 16px; border: 1px solid #e0e0e0; border-radius: 8px; background: #fafbfc; }}
      .sign-in {{ display: grid; gap: 10px; max-width: 360px; }}
      .sign-in label {{ font-weight: 600; font-size: 0.9rem; }}
      .sign-in input {{ padding: 8px 10px; border: 1px solid #c8ccd4; border-radius: 6px; font: inherit; }}
      .sign-in button {{ justify-self: start; padding: 8px 16px; border: 0; border-radius: 6px; background: #5e43f3; color: #fff; font: inherit; font-weight: 600; cursor: pointer; }}
      a {{ color: #5e43f3; }}
    </style>
  </head>
  <body>
    <h1>HTTP Basic Auth required</h1>
    <p>The NPA Agent workbench did not receive valid credentials. Sign in below or use your browser&apos;s Basic-auth dialog for <code>/</code> and <code>/api/*</code>.</p>
{strip_url_credentials_js}
{login_form_html}
    <ul>
      <li>Username: <code>{auth_user}</code></li>
      <li>Password: from your operator&apos;s <code>auth.env</code> file (<code>AGENT_PASSWORD</code>).</li>
      <li>Try the public <a href="/welcome">welcome page</a> for step-by-step instructions.</li>
      <li>Health (no auth): <a href="/healthz">/healthz</a></li>
    </ul>
  </body>
</html>
LOGINHELP
cat <<'HTML' | sudo tee /opt/npa-agent/ui.html >/dev/null
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <meta http-equiv="Cache-Control" content="no-store, no-cache, must-revalidate">
    <meta name="npa-ui-version" content="{AGENT_UI_VERSION}">
    <title>NPA Agent</title>
    <link rel="preload" href="/rerun/re_viewer.js" as="script" crossorigin>
    <link rel="preload" href="/rerun/re_viewer_bg.wasm" as="fetch" type="application/wasm" crossorigin>
    <link rel="prefetch" href="/rerun/recordings/sim2real.rrd" as="fetch">
    <style>
      :root {{
        --bg: #f5f6f8;
        --surface: #ffffff;
        --text: #1f2430;
        --muted: #5f6573;
        --border: #e0e0e0;
        --brand: #5e43f3;
        --brand-strong: #4d35d4;
        --sidebar: #1e1f22;
        --ok-bg: #e8f7ee;
        --ok-text: #18794e;
        --shadow: 0 8px 22px rgba(30, 31, 34, 0.08);
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        padding-bottom: 36px;
        font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        background: var(--bg);
        color: var(--text);
      }}
      .chrome {{
        max-width: 1640px;
        margin: 0 auto;
        min-height: 100vh;
      }}
      .topbar {{
        background: var(--sidebar);
        color: #eef0f3;
        padding: 14px 18px;
        border-bottom: 1px solid #2a2c31;
        display: flex;
        align-items: center;
        justify-content: space-between;
      }}
      .brand {{
        font-weight: 700;
        letter-spacing: 0.02em;
        font-size: 13px;
      }}
      .brand-sub {{
        color: #abb2bf;
        font-size: 12px;
      }}
      .page {{
        padding: 16px;
        display: grid;
        gap: 14px;
      }}
      .layout {{ display: grid; gap: 14px; }}
      .layout-3 {{ grid-template-columns: 1fr 0.95fr 1.35fr; }}
      .panel {{
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 14px;
        background: var(--surface);
        box-shadow: var(--shadow);
      }}
      .panel h3 {{ margin: 0 0 10px 0; font-size: 17px; }}
      .panel p {{ margin: 0; color: var(--muted); }}
      .subsection {{
        border: 1px solid #e7e8ee;
        border-radius: 10px;
        background: #fafbff;
        padding: 10px;
        margin-top: 10px;
      }}
      .subsection h4 {{ margin: 0 0 8px 0; font-size: 13px; color: #303649; }}
      .field-row {{ display: grid; gap: 8px; grid-template-columns: 1fr 1fr; }}
      .field label {{ display: block; font-size: 12px; color: #4f5668; margin-bottom: 4px; }}
      .field select, .field input {{
        width: 100%;
        border: 1px solid #d4d8e2;
        border-radius: 9px;
        padding: 8px;
        font-family: inherit;
        background: #fff;
      }}
      .pill-list {{ display: flex; gap: 6px; flex-wrap: wrap; }}
      .pill {{
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        border: 1px solid #d8dbeb;
        color: #394056;
        background: #fff;
        padding: 4px 9px;
        font-size: 12px;
      }}
      .cameras-panel {{ border-color: #d7dbf6; }}
      .camera-card {{
        border: 1px solid #e2e8f0; border-radius: 10px; padding: 10px; margin-bottom: 10px;
        background: #fff;
      }}
      .camera-card.selected {{ border: 2px solid #5e43f3; box-shadow: 0 0 0 1px rgba(94, 67, 243, 0.18); }}
      .camera-card h4 {{ margin: 0 0 6px 0; }}
      .camera-meta {{ font-size: 12px; color: #4b5568; margin-bottom: 6px; }}
      .camera-actions {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }}
      .camera-frustum {{ display: flex; justify-content: center; }}
      .rollout-hint {{ font-size: 13px; color: #39465c; margin: 0 0 10px 0; }}
      .chat-panel {{ margin-bottom: 12px; }}
      .chat-log {{
        height: 320px; overflow-y: auto; background: #f9fafc; border: 1px solid var(--border);
        border-radius: 10px; padding: 10px; margin-bottom: 10px;
        font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      }}
      .msg-row {{
        display: flex;
        margin: 8px 0;
      }}
      .msg-row.user {{ justify-content: flex-end; }}
      .msg-row.assistant, .msg-row.error, .msg-row.thinking {{ justify-content: flex-start; }}
      .msg-card {{
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        gap: 6px;
        max-width: 78%;
      }}
      .msg-row.user .msg-card {{ align-items: flex-end; }}
      .bubble {{
        max-width: 100%;
        border-radius: 12px;
        border: 1px solid var(--border);
        background: #fff;
        color: var(--text);
        padding: 10px 12px;
        font-size: 14px;
        line-height: 1.45;
      }}
      .bubble pre {{
        margin: 8px 0;
        background: #f4f6fc;
        border: 1px solid #dde2f0;
        border-radius: 8px;
        padding: 10px;
        overflow-x: auto;
      }}
      .bubble pre code {{
        white-space: pre;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
        font-size: 12px;
        line-height: 1.4;
        border: none;
        background: transparent;
        padding: 0;
      }}
      .bubble p {{ margin: 0 0 6px 0; color: inherit; }}
      .bubble p:last-child {{ margin-bottom: 0; }}
      .bubble ul {{ margin: 4px 0 6px 18px; padding: 0; }}
      .bubble li {{ margin: 2px 0; }}
      .bubble code {{
        background: #f0f1f6;
        border: 1px solid #e4e6f0;
        border-radius: 6px;
        padding: 1px 5px;
        font-size: 12px;
      }}
      .msg-row.user .bubble {{
        background: var(--brand);
        border-color: var(--brand);
        color: #fff;
      }}
      .msg-row.error .bubble {{
        border-color: #e9b8b8;
        background: #fff8f8;
      }}
      .msg-actions {{
        display: inline-flex;
        gap: 6px;
      }}
      .msg-copy-btn {{
        border: 1px solid #d5d9e6;
        border-radius: 999px;
        background: #fff;
        color: #3c4458;
        font-size: 11px;
        padding: 4px 10px;
        cursor: pointer;
      }}
      .msg-copy-btn:hover {{ background: #f5f7ff; }}
      .msg-row.thinking .bubble {{
        min-width: 90px;
        background: #f2f3f8;
      }}
      .thinking-dots {{
        display: inline-flex;
        gap: 4px;
        align-items: center;
      }}
      .thinking-dots span {{
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #6f7785;
        display: inline-block;
        animation: pulse 1s infinite ease-in-out;
      }}
      .thinking-dots span:nth-child(2) {{ animation-delay: 0.18s; }}
      .thinking-dots span:nth-child(3) {{ animation-delay: 0.36s; }}
      .sparkle {{
        display: inline-block;
        color: #5e43f3;
        margin-right: 6px;
        font-size: 13px;
      }}
      @keyframes pulse {{
        0%, 80%, 100% {{ opacity: 0.35; transform: translateY(0); }}
        40% {{ opacity: 1; transform: translateY(-1px); }}
      }}
      .chat-input {{ display: flex; gap: 10px; align-items: flex-end; }}
      .chat-input textarea {{
        flex: 1; min-height: 58px; resize: vertical; font-family: inherit; padding: 10px 12px;
        border: 1px solid var(--border);
        border-radius: 12px;
        outline: none;
      }}
      .chat-input textarea:focus {{ border-color: #c2bae7; box-shadow: 0 0 0 3px rgba(94, 67, 243, 0.12); }}
      iframe {{ width: 100%; height: 380px; border: 1px solid var(--border); border-radius: 10px; }}
      .rerun-placeholder {{
        width: 100%;
        min-height: 220px;
        border: 1px dashed var(--border);
        border-radius: 10px;
        padding: 24px 20px;
        text-align: center;
        color: var(--muted);
        background: #fafbff;
      }}
      .rerun-placeholder strong {{ color: var(--text); }}
      .status-row {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 8px; font-size: 14px; }}
      .btn-row {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 8px; }}
      .btn {{
        border: 1px solid #d6d9e4;
        background: #fff;
        border-radius: 999px;
        color: #2d3342;
        padding: 8px 12px;
        font-size: 13px;
        cursor: pointer;
      }}
      .btn:hover {{ background: #f5f6fb; }}
      .btn-primary {{
        background: var(--brand);
        border-color: var(--brand);
        color: #fff;
      }}
      .btn-primary:hover {{ background: var(--brand-strong); }}
      .btn[disabled], .btn:disabled {{
        opacity: 0.65;
        cursor: not-allowed;
      }}
      .cta {{ color: #92400e; background: #fffbeb; border: 1px solid #fcd34d; border-radius: 8px; padding: 8px 10px; }}
      .badge {{ display: inline-block; padding: 3px 9px; border-radius: 999px; background: #ece9ff; color: #33207d; font-size: 12px; }}
      .badge-ok {{ background: var(--ok-bg); color: var(--ok-text); }}
      .actions-inline {{ margin-top: 10px; display:flex; gap:8px; flex-wrap:wrap; }}
      .quick-pill {{
        border-radius: 999px;
        border: 1px solid #c8c0f5;
        background: #f6f4ff;
        color: #3d2f9c;
        font-size: 12px;
        padding: 7px 12px;
      }}
      .quick-pill:hover {{ background: #ede9ff; }}
      .hint {{ font-size: 13px; color: var(--muted); }}
      .status-bar {{
        position: fixed;
        left: 0;
        right: 0;
        bottom: 0;
        z-index: 900;
        padding: 8px 14px;
        font-size: 12px;
        color: #334155;
        background: rgba(255, 255, 255, 0.96);
        border-top: 1px solid var(--border);
        box-shadow: 0 -4px 16px rgba(30, 31, 34, 0.06);
      }}
      .toast-host {{
        position: fixed;
        top: 14px;
        right: 14px;
        z-index: 1000;
        display: flex;
        flex-direction: column;
        gap: 8px;
        max-width: min(420px, calc(100vw - 28px));
        pointer-events: none;
      }}
      .toast {{
        pointer-events: auto;
        padding: 10px 12px;
        border-radius: 10px;
        font-size: 13px;
        border: 1px solid #d6d9e4;
        background: #fff;
        color: #1f2430;
        box-shadow: var(--shadow);
        animation: toast-in 0.18s ease-out;
      }}
      .toast-info {{ border-color: #c8c0f5; background: #f6f4ff; color: #3d2f9c; }}
      .toast-success {{ border-color: #86efac; background: var(--ok-bg); color: var(--ok-text); }}
      .toast-error {{ border-color: #fca5a5; background: #fef2f2; color: #991b1b; }}
      @keyframes toast-in {{
        from {{ opacity: 0; transform: translateY(-6px); }}
        to {{ opacity: 1; transform: translateY(0); }}
      }}
      @media (max-width: 1280px) {{
        .layout-3 {{ grid-template-columns: 1fr; }}
      }}
      .workflow-panel textarea {{
        width: 100%;
        min-height: 220px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 12px;
        line-height: 1.45;
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 10px 12px;
        resize: vertical;
        background: #fafbff;
      }}
      .workflow-meta {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 8px 0 10px;
        font-size: 12px;
      }}
      .workflow-meta .pill {{
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 4px 8px;
        border-radius: 999px;
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
      }}
      .yaml-block {{
        margin: 8px 0;
        padding: 10px 12px;
        border-radius: 8px;
        background: #0f172a;
        color: #e2e8f0;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 12px;
        line-height: 1.45;
        overflow-x: auto;
        white-space: pre-wrap;
      }}
      #workflowYaml {{
        width: 100%;
        min-height: 220px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 12px;
        border: 1px solid #d4d8e2;
        border-radius: 10px;
        padding: 10px 12px;
        resize: vertical;
        background: #fafbff;
      }}
    </style>
  </head>
  <body>
    <div class="chrome">
      <header class="topbar">
        <div>
          <div class="brand">NEBIUS | NPA WORKBENCH AGENT</div>
          <div class="brand-sub">Sim2Real operations, assets, cameras, and Rerun visualization</div>
        </div>
        <span class="badge badge-ok">Secure basic-auth session</span>
      </header>
      <main class="page">
        <section class="panel chat-panel">
          <h3>Workbench Chat</h3>
          <p class="hint">Ask about configure, provision, Cosmos3, S3, workflows, sim assets, and Rerun visualization.</p>
          <div id="chatLog" class="chat-log"></div>
          <div class="chat-input">
            <textarea id="chatInput" placeholder="How do I configure S3 for Sim2Real?"></textarea>
            <button id="chatSend" class="btn btn-primary" type="button">Send</button>
          </div>
          <div class="actions-inline">
            <button id="chatActionS3" class="btn quick-pill" type="button">Configure S3</button>
            <button id="chatActionCosmos" class="btn quick-pill" type="button">Setup Cosmos3</button>
            <button id="chatActionWatch" class="btn quick-pill" type="button">Watch sim</button>
            <button id="chatActionWorkflow" class="btn quick-pill" type="button">2-step Sim2Real YAML</button>
          </div>
        </section>
        <section class="panel workflow-panel">
          <h3>Workflow YAML</h3>
          <p class="hint">npa.workflow/v0.0.1-beta specs — generate via chat, upload, validate, plan, or submit.</p>
          <div class="workflow-meta">
            <span class="pill">name: <strong id="workflowName">—</strong></span>
            <span class="pill">validation: <strong id="workflowValidation">pending</strong></span>
            <span class="pill">states: <strong id="workflowStates">—</strong></span>
          </div>
          <textarea id="workflowYaml" spellcheck="false" placeholder="Ask chat to create a 2-step Sim2Real workflow, or paste YAML here…"></textarea>
          <div class="btn-row" style="margin-top:8px;">
            <button id="workflowUpload" class="btn" type="button">Upload YAML</button>
            <button id="workflowValidate" class="btn" type="button">Validate</button>
            <button id="workflowPlan" class="btn" type="button">Plan</button>
            <button id="workflowSubmitYaml" class="btn btn-primary" type="button">Submit YAML</button>
          </div>
          <pre id="workflowPlanOutput" class="hint" style="margin-top:8px; white-space:pre-wrap;"></pre>
        </section>
        <div class="layout layout-3">
          <section class="panel">
            <h3>Sim Assets</h3>
            <div class="subsection">
              <h4>Selection</h4>
              <div class="field-row">
                <div class="field">
                  <label for="sceneMode">Scene mode</label>
                  <select id="sceneMode">
                    <option value="stock" selected>stock</option>
                    <option value="byo_mesh">byo_mesh</option>
                    <option value="scene_spec">scene_spec</option>
                  </select>
                </div>
                <div class="field">
                  <label for="robotPreset">Robot preset</label>
                  <select id="robotPreset">
                    <option value="franka" selected>stock_franka</option>
                    <option value="ur5e">preset:ur5e</option>
                  </select>
                </div>
              </div>
              <div class="field-row" style="margin-top:8px;">
                <div class="field">
                  <label for="cameraMode">Camera mode</label>
                  <select id="cameraMode">
                    <option value="stock" selected>stock</option>
                    <option value="custom">custom</option>
                  </select>
                </div>
                <div class="field">
                  <label for="simBackend">Sim backend</label>
                  <select id="simBackend">
                    <option value="isaac" selected>isaac</option>
                    <option value="genesis">genesis</option>
                  </select>
                </div>
              </div>
            </div>
            <div class="subsection">
              <h4>Props</h4>
              <div class="pill-list">
                <label class="pill"><input id="propCube" type="checkbox" checked> cube</label>
              </div>
            </div>
            <div class="subsection">
              <h4>Resolved assets</h4>
              <div id="assetsSummary" class="hint"></div>
            </div>
            <div class="btn-row">
              <button id="applySelection" class="btn" type="button">Apply stock selection</button>
              <button id="loadFrankaRerun" class="btn" type="button">Load Franka in Rerun (fallback)</button>
              <button id="submitWorkflow" class="btn btn-primary" type="button">Submit Sim2Real</button>
              <button id="workflowStatus" class="btn" type="button">Workflow status</button>
            </div>
            <div class="field-row" style="margin-top:8px;">
              <div class="field">
                <label for="runIdInput">Run ID</label>
                <input id="runIdInput" type="text" placeholder="agent-run-..." />
              </div>
              <div class="field">
                <label for="runIdSelect">Known runs</label>
                <select id="runIdSelect">
                  <option value="">(select run)</option>
                </select>
              </div>
            </div>
            <div class="btn-row" style="margin-top:8px;">
              <button id="loadRunData" class="btn" type="button">Load run data</button>
            </div>
            <div class="subsection" style="margin-top:10px;">
              <h4>Artifact browser</h4>
              <div class="field-row">
                <div class="field">
                  <label for="artifactPrefix">Prefix</label>
                  <input id="artifactPrefix" type="text" placeholder="optional/path/prefix" />
                </div>
                <div class="field">
                  <label for="artifactRunSelect">Discovered runs</label>
                  <select id="artifactRunSelect">
                    <option value="">(select discovered run)</option>
                  </select>
                </div>
              </div>
              <div class="btn-row" style="margin-top:8px;">
                <button id="artifactRefreshRuns" class="btn" type="button">Discover runs</button>
                <button id="artifactLoadRunArtifacts" class="btn" type="button">List artifacts</button>
              </div>
              <div id="artifactList" class="hint" style="margin-top:8px;"></div>
            </div>
          </section>
          <section class="panel cameras-panel">
            <h3>Cameras</h3>
            <p id="cameraRolloutHint" class="rollout-hint">Active for next rollout: <strong id="activeCameraLabel">workspace</strong></p>
            <p class="rollout-hint">Stock workspace and wrist cameras from the Sim2Real default scene spec.</p>
            <div id="cameraCards"></div>
            <select id="cameraSelect" hidden aria-hidden="true"></select>
            <div id="rerunEntityHint" class="rollout-hint"></div>
          </section>
          <section class="panel">
            <h3>Rerun (embedded)</h3>
            <div id="simviz">
              <div class="status-row">
                <span>Run: <strong id="simRunId">—</strong></span>
                <span>Stage: <span id="simStage" class="badge">idle</span></span>
                <span>Camera: <strong id="simCamera">workspace</strong></span>
              </div>
              <div class="btn-row">
                <button id="openRerun" class="btn" type="button">Open in Rerun</button>
              </div>
              <p id="simvizCta" class="cta">Discover artifacts first; use Franka demo fallback when no S3 artifacts are available.</p>
            </div>
            <div id="rerunPlaceholder" class="rerun-placeholder">
              <p>Loading Rerun viewer…</p>
              <p class="hint">Preloading WASM and stock Franka recording from bootstrap cache.</p>
              <button id="loadRerunViewer" class="btn btn-primary" type="button">Load Rerun viewer</button>
            </div>
            <iframe id="rerunFrame" title="rerun" hidden></iframe>
            <div id="artifactPreviewHost" class="hint" hidden style="margin-top:10px;"></div>
          </section>
        </div>
      </main>
    </div>
    <noscript><p style="padding:16px;background:#fff3cd;">JavaScript is required for the NPA Agent workbench. Enable JS and reload.</p></noscript>
    <div id="statusBar" class="status-bar" aria-live="polite">Ready</div>
    <div id="toastHost" class="toast-host" aria-live="polite"></div>
    <script>
      (function initNpaAgentUi() {{
      try {{
      "use strict";
      if (location.username || location.password) {{
        const clean = location.protocol + "//" + location.host + location.pathname + location.search + location.hash;
        history.replaceState(null, "", clean);
      }}
      const chatHistory = [];
      let thinkingNode = null;
      function setStatus(text) {{
        const bar = document.getElementById("statusBar");
        if (bar) bar.textContent = String(text || "");
      }}
      function showToast(message, kind) {{
        const host = document.getElementById("toastHost");
        if (!host) return;
        const toast = document.createElement("div");
        const tone = kind === "error" ? "toast-error" : kind === "success" ? "toast-success" : "toast-info";
        toast.className = "toast " + tone;
        toast.textContent = String(message || "");
        host.appendChild(toast);
        window.setTimeout(() => {{
          if (toast.parentNode) toast.parentNode.removeChild(toast);
        }}, 4200);
      }}
      function bindClick(id, fn, label) {{
        const el = document.getElementById(id);
        if (!el) {{
          console.error("Missing UI control:", id);
          showToast("Missing control: " + id, "error");
          return;
        }}
        el.addEventListener("click", async (event) => {{
          event.preventDefault();
          const actionLabel = String(label || id);
          setStatus(actionLabel + "...");
          showToast(actionLabel, "info");
        try {{
          const result = await fn(event);
          if (result === false) {{
            setStatus("Ready");
            return;
          }}
          setStatus(actionLabel + " done");
          showToast(actionLabel + " done", "success");
        }} catch (err) {{
            const msg = String(err && err.message ? err.message : err);
            setStatus(actionLabel + " failed");
            showToast(msg, "error");
            console.error(actionLabel, err);
          }}
        }});
      }}
      function wireUi() {{
        bindClick("chatSend", sendChat, "Send chat");
        bindClick("chatActionS3", () => {{
          setChatInput("Help me configure S3 credentials and bucket for NPA workflows.");
        }}, "Insert S3 prompt");
        bindClick("chatActionCosmos", () => {{
          setChatInput("How do I set up Cosmos3 in the NPA workbench?");
        }}, "Insert Cosmos3 prompt");
        bindClick("chatActionWatch", () => {{
          setChatInput("Watch the sim in Rerun and keep retrying blob+iframe mount until SUCCESS using /api/sim-viz/status.");
        }}, "Insert watch-sim prompt");
        bindClick("chatActionWorkflow", () => {{
          setChatInput("Create a 2-step sim2real workflow YAML with real toolRefs from the catalog.");
        }}, "Insert workflow YAML prompt");
        bindClick("workflowUpload", uploadWorkflowYaml, "Upload workflow YAML");
        bindClick("workflowValidate", validateWorkflowYaml, "Validate workflow YAML");
        bindClick("workflowPlan", planWorkflowYaml, "Plan workflow YAML");
        bindClick("workflowSubmitYaml", submitWorkflowYaml, "Submit workflow YAML");
        bindClick("loadFrankaRerun", loadFrankaDemo, "Load Franka in Rerun");
        bindClick("artifactRefreshRuns", refreshArtifactRuns, "Discover artifact runs");
        bindClick("artifactLoadRunArtifacts", loadArtifactsForSelectedRun, "List run artifacts");
        bindClick("loadRerunViewer", () => loadRerunViewer(), "Load Rerun viewer");
        bindClick("openRerun", openRerunTab, "Open Rerun");
        bindClick("applySelection", applySelection, "Apply stock selection");
        bindClick("submitWorkflow", submitWorkflow, "Submit Sim2Real");
        bindClick("workflowStatus", showWorkflowStatus, "Workflow status");
        bindClick("loadRunData", loadRunData, "Load run data");
        const chatInput = document.getElementById("chatInput");
        if (chatInput) {{
          chatInput.addEventListener("keydown", (e) => {{
            if (e.key === "Enter" && !e.shiftKey) {{
              e.preventDefault();
              sendChat().catch((err) => showToast(String(err), "error"));
            }}
          }});
        }}
        const cameraSelect = document.getElementById("cameraSelect");
        if (cameraSelect) {{
          cameraSelect.addEventListener("change", async (e) => {{
            try {{
              await selectCamera(String(e.target.value || ""));
              showToast("Camera selected", "success");
            }} catch (err) {{
              showToast(String(err), "error");
            }}
          }});
        }}
        const runIdSelect = document.getElementById("runIdSelect");
        if (runIdSelect) {{
          runIdSelect.addEventListener("change", () => {{
            const chosen = String(runIdSelect.value || "").trim();
            const input = document.getElementById("runIdInput");
            if (input && chosen) input.value = chosen;
          }});
        }}
        const artifactRunSelect = document.getElementById("artifactRunSelect");
        if (artifactRunSelect) {{
          artifactRunSelect.addEventListener("change", async () => {{
            const selectedRun = String(artifactRunSelect.value || "").trim();
            if (!selectedRun) return;
            await loadArtifactsForSelectedRun();
          }});
        }}
        const robotPreset = document.getElementById("robotPreset");
        if (robotPreset) {{
          robotPreset.addEventListener("change", async (e) => {{
            try {{
              const data = await apiJson("/api/sim-assets/selection", {{
                method: "POST",
                headers: {{ "content-type": "application/json" }},
                body: JSON.stringify(selectionPayloadFromUi()),
              }});
              if (String(e.target.value || "franka") === "franka" && data.sim_viz) {{
                reloadRerunIframe(data.sim_viz.camera || "workspace");
              }}
              await refresh();
              showToast("Robot preset updated", "success");
            }} catch (err) {{
              showToast(String(err), "error");
            }}
          }});
        }}
      }}
      function escapeHtml(text) {{
        return String(text || "")
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;");
      }}
      function renderInlineMarkdownLite(text) {{
        let value = escapeHtml(text);
        value = value.replace(/`([^`]+)`/g, "<code>$1</code>");
        value = value.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
        return value;
      }}
      function normalizeAssistantReply(text) {{
        const raw = String(text || "").trim();
        if (!raw) return raw;
        if (!/^[\[\{{]/.test(raw)) return raw;
        try {{
          const parsed = JSON.parse(raw);
          if (parsed && typeof parsed === "object") {{
            const lines = [];
            for (const [key, value] of Object.entries(parsed)) {{
              const rendered = (value !== null && typeof value === "object")
                ? "`" + JSON.stringify(value) + "`"
                : "`" + String(value) + "`";
              lines.push("- **" + key + "**: " + rendered);
            }}
            return lines.join("\\n");
          }}
        }} catch (_err) {{
          // Keep original non-JSON text.
        }}
        return raw;
      }}
      function markdownLiteHtml(text) {{
        const lines = String(text || "").split(/\\r?\\n/);
        let html = "";
        let inList = false;
        let listKind = "";
        let inCode = false;
        let codeLang = "";
        let codeLines = [];
        const closeList = () => {{
          if (!inList) return;
          html += listKind === "ol" ? "</ol>" : "</ul>";
          inList = false;
          listKind = "";
        }};
        const closeCode = () => {{
          if (!inCode) return;
          const langAttr = codeLang ? ' data-lang="' + escapeHtml(codeLang) + '"' : "";
          html += "<pre><code" + langAttr + ">" + escapeHtml(codeLines.join("\\n")) + "</code></pre>";
          inCode = false;
          codeLang = "";
          codeLines = [];
        }};
        for (const raw of lines) {{
          const line = String(raw || "");
          const fenceMatch = line.match(/^```([a-zA-Z0-9_-]+)?\s*$/);
          if (fenceMatch) {{
            if (inCode) {{
              closeCode();
            }} else {{
              closeList();
              inCode = true;
              codeLang = String(fenceMatch[1] || "").toLowerCase();
              codeLines = [];
            }}
            continue;
          }}
          if (inCode) {{
            codeLines.push(line);
            continue;
          }}
          if (/^\s*[-*]\s+/.test(line)) {{
            if (!inList || listKind !== "ul") {{
              closeList();
              html += "<ul>";
              inList = true;
              listKind = "ul";
            }}
            html += "<li>" + renderInlineMarkdownLite(line.replace(/^\s*[-*]\s+/, "")) + "</li>";
            continue;
          }}
          if (/^\s*\d+\.\s+/.test(line)) {{
            if (!inList || listKind !== "ol") {{
              closeList();
              html += "<ol>";
              inList = true;
              listKind = "ol";
            }}
            html += "<li>" + renderInlineMarkdownLite(line.replace(/^\s*\d+\.\s+/, "")) + "</li>";
            continue;
          }}
          closeList();
          if (!line.trim()) {{
            continue;
          }}
          html += "<p>" + renderInlineMarkdownLite(line) + "</p>";
        }}
        closeList();
        closeCode();
        return html || "<p></p>";
      }}
      function extractFencedCode(text, preferredLang) {{
        const raw = String(text || "");
        const blocks = [];
        const re = /```([a-zA-Z0-9_-]+)?\s*\n([\s\S]*?)```/g;
        let match;
        while ((match = re.exec(raw)) !== null) {{
          blocks.push({{
            lang: String(match[1] || "").toLowerCase(),
            body: String(match[2] || "").replace(/\s+$/, ""),
          }});
        }}
        if (!blocks.length) return "";
        if (preferredLang) {{
          const target = blocks.find((item) => item.lang === String(preferredLang).toLowerCase());
          if (target) return target.body;
        }}
        return blocks[0].body;
      }}
      async function copyTextToClipboard(text) {{
        const value = String(text || "");
        if (!value) return false;
        if (typeof navigator !== "undefined" && navigator.clipboard && navigator.clipboard.writeText) {{
          await navigator.clipboard.writeText(value);
          return true;
        }}
        const ta = document.createElement("textarea");
        ta.value = value;
        ta.setAttribute("readonly", "readonly");
        ta.style.position = "absolute";
        ta.style.left = "-9999px";
        document.body.appendChild(ta);
        ta.select();
        const ok = document.execCommand("copy");
        document.body.removeChild(ta);
        return Boolean(ok);
      }}
      function appendChat(role, text, opts) {{
        const options = opts || {{}};
        const rawText = String(text || "");
        const log = document.getElementById("chatLog");
        const row = document.createElement("div");
        row.className = "msg-row " + role;
        const card = document.createElement("div");
        card.className = "msg-card";
        const bubble = document.createElement("div");
        bubble.className = "bubble";
        if (options.thinking) {{
          bubble.innerHTML =
            '<span class="sparkle">✦</span><span class="thinking-dots"><span></span><span></span><span></span></span>';
        }} else {{
          bubble.innerHTML = markdownLiteHtml(rawText);
        }}
        card.appendChild(bubble);
        if (!options.thinking && role === "assistant") {{
          const actions = document.createElement("div");
          actions.className = "msg-actions";
          const copyBtn = document.createElement("button");
          copyBtn.type = "button";
          copyBtn.className = "msg-copy-btn";
          const yamlBlock = extractFencedCode(rawText, "yaml");
          const payload = yamlBlock || rawText;
          copyBtn.textContent = yamlBlock ? "Copy YAML" : "Copy";
          copyBtn.addEventListener("click", async () => {{
            try {{
              const ok = await copyTextToClipboard(payload);
              if (!ok) throw new Error("copy failed");
              showToast(yamlBlock ? "YAML copied" : "Message copied", "success");
            }} catch (err) {{
              showToast(String(err && err.message ? err.message : err), "error");
            }}
          }});
          actions.appendChild(copyBtn);
          card.appendChild(actions);
        }}
        row.appendChild(card);
        log.appendChild(row);
        log.scrollTop = log.scrollHeight;
        return row;
      }}
      function showThinkingBubble() {{
        if (thinkingNode) return;
        thinkingNode = appendChat("thinking", "", {{ thinking: true }});
      }}
      function clearThinkingBubble() {{
        if (thinkingNode && thinkingNode.parentNode) {{
          thinkingNode.parentNode.removeChild(thinkingNode);
        }}
        thinkingNode = null;
      }}
      function setChatBusy(isBusy) {{
        const btn = document.getElementById("chatSend");
        const input = document.getElementById("chatInput");
        btn.disabled = Boolean(isBusy);
        input.disabled = Boolean(isBusy);
      }}
      function setWorkflowYaml(text, validation) {{
        const area = document.getElementById("workflowYaml");
        if (area) area.value = String(text || "");
        updateWorkflowMeta(validation || {{}});
      }}
      function updateWorkflowMeta(validation) {{
        const nameEl = document.getElementById("workflowName");
        const valEl = document.getElementById("workflowValidation");
        const statesEl = document.getElementById("workflowStates");
        const name = String((validation && validation.name) || "—");
        const status = String((validation && validation.status) || (validation && validation.ok ? "valid" : "pending"));
        const states = Array.isArray(validation && validation.states)
          ? validation.states.join(", ")
          : String((validation && validation.states) || "—");
        if (nameEl) nameEl.textContent = name;
        if (valEl) valEl.textContent = status;
        if (statesEl) statesEl.textContent = states || "—";
      }}
      function currentWorkflowYaml() {{
        const area = document.getElementById("workflowYaml");
        return String((area && area.value) || "").trim();
      }}
      async function uploadWorkflowYaml() {{
        const yaml = currentWorkflowYaml();
        if (!yaml) throw new Error("Paste or generate workflow YAML first");
        const data = await apiJson("/api/workflows/draft", {{
          method: "POST",
          headers: {{ "content-type": "application/json" }},
          body: JSON.stringify({{ yaml }}),
        }});
        updateWorkflowMeta((data.validation) || {{}});
        appendChat("assistant", "Uploaded workflow YAML to the **Workflow YAML** panel (`" + String((data.validation && data.validation.name) || "draft") + "`).");
        return true;
      }}
      async function validateWorkflowYaml() {{
        const yaml = currentWorkflowYaml();
        if (!yaml) throw new Error("Paste or generate workflow YAML first");
        const data = await apiJson("/api/workflows/validate", {{
          method: "POST",
          headers: {{ "content-type": "application/json" }},
          body: JSON.stringify({{ yaml }}),
        }});
        updateWorkflowMeta((data.validation) || {{}});
        if (!data.ok) throw new Error(String((data.validation && data.validation.error) || "validation failed"));
        return true;
      }}
      async function planWorkflowYaml() {{
        const yaml = currentWorkflowYaml();
        if (!yaml) throw new Error("Paste or generate workflow YAML first");
        const data = await apiJson("/api/workflows/plan", {{
          method: "POST",
          headers: {{ "content-type": "application/json" }},
          body: JSON.stringify({{ yaml, run_id: "agent-plan" }}),
        }});
        const out = document.getElementById("workflowPlanOutput");
        const steps = Array.isArray(data.plan && data.plan.steps) ? data.plan.steps : [];
        const lines = steps.map((step, idx) =>
          String(idx + 1).padStart(2, "0") + ". " + String(step.state || "?") +
          " toolRef=" + String(step.tool_ref || step.toolRef || "")
        );
        if (out) out.textContent = lines.length ? lines.join("\\n") : JSON.stringify(data.plan || {{}}, null, 2);
        updateWorkflowMeta((data.plan && data.plan.workflow) ? {{ name: data.plan.workflow, status: "planned", states: steps.map((s) => s.state) }} : {{}});
        return true;
      }}
      async function submitWorkflowYaml() {{
        const yaml = currentWorkflowYaml();
        if (!yaml) throw new Error("Paste or generate workflow YAML first");
        const data = await apiJson("/api/workflows/submit", {{
          method: "POST",
          headers: {{ "content-type": "application/json" }},
          body: JSON.stringify({{ yaml }}),
        }});
        updateWorkflowMeta((data.validation) || {{}});
        appendChat("assistant", "Submitted npa.workflow YAML — **run_id**: `" + String(data.run_id || "") + "` (validate-only on agent VM).");
        return true;
      }}
      async function sendChat() {{
        const input = document.getElementById("chatInput");
        const text = String(input.value || "").trim();
        if (!text) {{
          showToast("Enter a message first", "info");
          return false;
        }}
        input.value = "";
        appendChat("user", text);
        chatHistory.push({{ role: "user", content: text }});
        setChatBusy(true);
        showThinkingBubble();
        try {{
          const data = await apiJson("/api/chat", {{
            method: "POST",
            headers: {{ "content-type": "application/json" }},
            body: JSON.stringify({{ messages: chatHistory }}),
          }});
          clearThinkingBubble();
          const reply = normalizeAssistantReply(data.reply || "");
          if (reply) {{
            appendChat("assistant", reply);
            chatHistory.push({{ role: "assistant", content: reply }});
          }} else {{
            appendChat("error", "empty reply from model");
          }}
          if (data.workflow_yaml) {{
            setWorkflowYaml(data.workflow_yaml, data.workflow_validation || {{}});
          }}
          const draft = data.workflow_draft;
          if (!data.workflow_yaml && draft && draft.yaml) {{
            setWorkflowYaml(draft.yaml, draft.validation || draft);
          }}
        }} catch (err) {{
          clearThinkingBubble();
          appendChat("error", String(err));
          throw err;
        }} finally {{
          setChatBusy(false);
          input.focus();
        }}
      }}
      function setChatInput(text) {{
        const input = document.getElementById("chatInput");
        input.value = text;
        input.focus();
      }}
      let lastRrdUpdatedAt = "";
      let rerunIframeLoaded = false;
      let rerunBootInProgress = false;
      let lastRrdBlobUrl = "";
      let activeRunId = "";
      let activeArtifactRender = "";
      let lastRerunBlobStatus = "pending";
      let lastRerunMountStatus = "pending";
      const RERUN_RECORDING_PATH = "/rerun/recordings/sim2real.rrd";
      const RERUN_BUNDLE_ASSETS = ["/rerun/re_viewer.js", "/rerun/re_viewer_bg.wasm"];
      let rerunBundleWarmPromise = null;
      const RERUN_BLOB_SUCCESS = "SUCCESS";
      const RERUN_MOUNT_SUCCESS = "SUCCESS";
      function setRerunBlobStatus(status, detail) {{
        const text = String(status || "").trim() || "pending";
        lastRerunBlobStatus = text;
        const extra = detail ? " (" + String(detail) + ")" : "";
        setStatus("Rerun blob: " + text + extra);
      }}
      function setRerunMountStatus(status, detail) {{
        const text = String(status || "").trim() || "pending";
        lastRerunMountStatus = text;
        const extra = detail ? " (" + String(detail) + ")" : "";
        setStatus("Rerun mount: " + text + extra);
      }}
      async function warmRerunBundle() {{
        if (rerunBundleWarmPromise) {{
          return rerunBundleWarmPromise;
        }}
        rerunBundleWarmPromise = Promise.all(
          RERUN_BUNDLE_ASSETS.map((path) =>
            fetchWithTimeout(
              path,
              {{ credentials: "include", cache: "force-cache" }},
              120000
            ).then((resp) => {{
              if (!resp.ok) {{
                throw new Error("Rerun bundle asset failed: " + path);
              }}
              return resp.blob();
            }})
          )
        ).catch((err) => {{
          rerunBundleWarmPromise = null;
          throw err;
        }});
        return rerunBundleWarmPromise;
      }}
      async function resolveRerunRecordingUrl() {{
        const cacheBust = "?t=" + String(Date.now());
        const recordingUrl = location.origin + RERUN_RECORDING_PATH + cacheBust;
        const resp = await fetchWithTimeout(RERUN_RECORDING_PATH, {{ credentials: "include", method: "HEAD" }}, 8000);
        if (!resp.ok) {{
          throw new Error("Rerun recording not published yet");
        }}
        setRerunBlobStatus("fallback", "recording=public");
        return recordingUrl;
      }}
      async function resolveRerunRrdUrl(maxAttempts, runId) {{
        const attempts = Math.max(1, Number(maxAttempts || 18));
        const targetRunId = String(runId || activeRunId || "").trim();
        const query = targetRunId ? ("?run_id=" + encodeURIComponent(targetRunId)) : "";
        let lastErr = null;
        for (let i = 0; i < attempts; i += 1) {{
          try {{
            const resp = await fetchWithTimeout("/api/sim-viz/rrd-blob" + query, {{ credentials: "include" }}, 12000);
            if (!resp.ok) {{
              if (resp.status === 401) {{
                window.location.href = "/login-help.html";
              }}
              throw new Error("Failed to fetch .rrd for Rerun viewer");
            }}
            const blob = await resp.blob();
            if (blob.size < 64) {{
              throw new Error("Rerun .rrd payload is too small");
            }}
            if (lastRrdBlobUrl) {{
              URL.revokeObjectURL(lastRrdBlobUrl);
            }}
            lastRrdBlobUrl = URL.createObjectURL(blob);
            setRerunBlobStatus(RERUN_BLOB_SUCCESS, "bytes=" + String(blob.size));
            return lastRrdBlobUrl;
          }} catch (err) {{
            lastErr = err;
            setRerunBlobStatus("retrying", "attempt " + String(i + 1) + "/" + String(attempts));
            if (i + 1 >= attempts) {{
              break;
            }}
            const backoffMs = Math.min(2500, 700 + i * 175);
            await new Promise((resolve) => window.setTimeout(resolve, backoffMs));
          }}
        }}
        setRerunBlobStatus("failed");
        throw lastErr || new Error("Failed to fetch .rrd for Rerun viewer");
      }}
      async function rerunIframeSrc(camera, runId) {{
        const cam = String(camera || "workspace");
        let rrdUrl = "";
        try {{
          rrdUrl = await resolveRerunRrdUrl(18, runId);
        }} catch (_blobErr) {{
          rrdUrl = await resolveRerunRecordingUrl();
        }}
        // Prefer authenticated blob fetch; public recording path is fallback.
        return (
          "/rerun/?url=" +
          encodeURIComponent(rrdUrl) +
          "&renderer=webgl&hide_welcome_screen=1&camera=" +
          encodeURIComponent(cam)
        );
      }}
      function showRerunPlaceholder(message) {{
        const placeholder = document.getElementById("rerunPlaceholder");
        const iframe = document.getElementById("rerunFrame");
        if (placeholder) {{
          placeholder.hidden = false;
          if (message) {{
            const hint = placeholder.querySelector(".hint");
            if (hint) hint.textContent = String(message);
          }}
        }}
        if (iframe) {{
          iframe.hidden = true;
          iframe.removeAttribute("src");
        }}
        rerunIframeLoaded = false;
      }}
      function hideRerunPlaceholder() {{
        const placeholder = document.getElementById("rerunPlaceholder");
        const iframe = document.getElementById("rerunFrame");
        if (placeholder) placeholder.hidden = true;
        if (iframe) iframe.hidden = false;
      }}
      function hideArtifactPreview() {{
        const host = document.getElementById("artifactPreviewHost");
        if (!host) return;
        host.hidden = true;
        host.innerHTML = "";
      }}
      async function showArtifactPreview(simViz, render) {{
        const host = document.getElementById("artifactPreviewHost");
        if (!host) return;
        const previewUrl = String((simViz && simViz.artifact_preview_url) || "");
        const downloadUrl = String((simViz && simViz.artifact_download_url) || previewUrl || "");
        const safeRender = String(render || "");
        if (!previewUrl) {{
          host.hidden = false;
          host.innerHTML = "<p>No preview URL available. Use download.</p>";
          return;
        }}
        if (safeRender === "image") {{
          host.hidden = false;
          host.innerHTML = `<img alt="artifact image" src="${{previewUrl}}" style="max-width:100%;border-radius:8px;border:1px solid #dbe2ef;" />`;
          return;
        }}
        if (safeRender === "video") {{
          host.hidden = false;
          host.innerHTML = `<video controls style="max-width:100%;border-radius:8px;border:1px solid #dbe2ef;" src="${{previewUrl}}"></video>`;
          return;
        }}
        if (safeRender === "json" || safeRender === "text") {{
          try {{
            const resp = await fetchWithTimeout(previewUrl, {{ credentials: "include" }}, 12000);
            if (!resp.ok) throw new Error("preview fetch failed");
            const text = await resp.text();
            host.hidden = false;
            host.innerHTML = `<pre style="white-space:pre-wrap;background:#0f172a;color:#e2e8f0;padding:10px;border-radius:8px;max-height:280px;overflow:auto;">${{escapeHtml(text.slice(0, 20000))}}</pre>`;
            return;
          }} catch (_err) {{
            // fall through to download link
          }}
        }}
        host.hidden = false;
        host.innerHTML = `<p>Artifact render: <strong>${{escapeHtml(safeRender || "download")}}</strong>. <a href="${{downloadUrl}}" target="_blank" rel="noopener">Download artifact</a></p>`;
      }}
      async function waitForRerunReady(maxAttempts) {{
        const attempts = Number(maxAttempts || 12);
        for (let i = 0; i < attempts; i += 1) {{
          const simViz = await loadJson("/api/sim-viz/status");
          if (simViz && simViz.rerun_ready) {{
            return simViz;
          }}
          await new Promise((resolve) => window.setTimeout(resolve, 500));
        }}
        throw new Error("Rerun viewer is not ready yet (service or .rrd missing)");
      }}
      async function waitForIframeLoad(iframe, timeoutMs) {{
        const timeout = Math.max(500, Number(timeoutMs || 8000));
        return await new Promise((resolve, reject) => {{
          let done = false;
          const timer = window.setTimeout(() => {{
            if (done) return;
            done = true;
            reject(new Error("Rerun iframe load timed out"));
          }}, timeout);
          function finish(ok, err) {{
            if (done) return;
            done = true;
            window.clearTimeout(timer);
            iframe.removeEventListener("load", onLoad);
            iframe.removeEventListener("error", onError);
            if (ok) {{
              resolve(true);
            }} else {{
              reject(err || new Error("Rerun iframe failed to load"));
            }}
          }}
          function onLoad() {{
            finish(true);
          }}
          function onError() {{
            finish(false, new Error("Rerun iframe error event"));
          }}
          iframe.addEventListener("load", onLoad, {{ once: true }});
          iframe.addEventListener("error", onError, {{ once: true }});
        }});
      }}
      async function mountRerunIframe(camera, runId) {{
        const iframe = document.getElementById("rerunFrame");
        if (!iframe) return true;
        const src = await rerunIframeSrc(camera, runId);
        setRerunMountStatus("retrying", "navigating");
        iframe.src = src;
        hideRerunPlaceholder();
        await waitForIframeLoad(iframe, 12000);
        rerunIframeLoaded = true;
        setRerunMountStatus(RERUN_MOUNT_SUCCESS, "loaded");
        return true;
      }}
      async function mountRerunIframeUntilSuccess(camera, maxAttempts, runId) {{
        const attempts = Math.max(1, Number(maxAttempts || 8));
        let lastErr = null;
        for (let i = 0; i < attempts; i += 1) {{
          try {{
            setRerunBlobStatus("retrying", "mount " + String(i + 1) + "/" + String(attempts));
            setRerunMountStatus("retrying", "mount " + String(i + 1) + "/" + String(attempts));
            await mountRerunIframe(camera, runId);
            if (lastRerunBlobStatus !== RERUN_BLOB_SUCCESS || lastRerunMountStatus !== RERUN_MOUNT_SUCCESS) {{
              throw new Error("Rerun iframe mount missing SUCCESS blob/mount state");
            }}
            return true;
          }} catch (err) {{
            lastErr = err;
            setRerunBlobStatus("retrying", "mount " + String(i + 1) + "/" + String(attempts));
            setRerunMountStatus("retrying", "mount " + String(i + 1) + "/" + String(attempts));
            if (i + 1 >= attempts) {{
              break;
            }}
            await new Promise((resolve) => window.setTimeout(resolve, 1000));
          }}
        }}
        throw lastErr || new Error("Rerun iframe mount did not reach SUCCESS");
      }}
      async function waitForRerunSuccess(camera, options) {{
        const opts = options || {{}};
        const deadlineMs = Math.max(5000, Number(opts.deadlineMs || 120000));
        const sleepMs = Math.max(500, Number(opts.sleepMs || 1200));
        const mountAttemptsPerLoop = Math.max(1, Number(opts.mountAttemptsPerLoop || 4));
        const successStreakTarget = Math.max(1, Number(opts.successStreakTarget || 2));
        const targetRunId = String(opts.runId || "").trim();
        const baselineUpdatedAt = String(opts.baselineRrdUpdatedAt || "").trim();
        const start = Date.now();
        let lastErr = null;
        let successStreak = 0;
        while (Date.now() - start < deadlineMs) {{
          try {{
            const statusPath = targetRunId
              ? "/api/sim-viz/status?run_id=" + encodeURIComponent(targetRunId)
              : "/api/sim-viz/status";
            const status = await loadJson(statusPath);
            const selectedCamera = String((status && status.camera) || camera || "workspace");
            const activeRunId = String((status && status.run_id) || "").trim();
            const activeStage = String((status && status.stage) || "idle").trim().toLowerCase();
            const activeUpdatedAt = String((status && status.rrd_updated_at) || "").trim();
            const runMatches = !targetRunId || (activeRunId && activeRunId === targetRunId);
            const stageAdvanced = !["", "idle", "submitted", "queued"].includes(activeStage);
            const rrdFresh = Boolean(activeUpdatedAt && (!baselineUpdatedAt || activeUpdatedAt !== baselineUpdatedAt));
            if (status && status.rrd_uri && runMatches && (stageAdvanced || rrdFresh)) {{
              await mountRerunIframeUntilSuccess(selectedCamera, mountAttemptsPerLoop, targetRunId);
              if (lastRerunBlobStatus === RERUN_BLOB_SUCCESS && lastRerunMountStatus === RERUN_MOUNT_SUCCESS) {{
                successStreak += 1;
                if (successStreak >= successStreakTarget) {{
                  return status;
                }}
              }} else {{
                successStreak = 0;
              }}
            }} else {{
              successStreak = 0;
            }}
          }} catch (err) {{
            lastErr = err;
            successStreak = 0;
          }}
          await new Promise((resolve) => window.setTimeout(resolve, sleepMs));
        }}
        if (lastErr) {{
          throw lastErr;
        }}
        throw new Error("Timed out waiting for rerun blob/iframe SUCCESS");
      }}
      function reloadRerunIframe(camera) {{
        if (!rerunIframeLoaded) return Promise.resolve();
        return mountRerunIframeUntilSuccess(camera, 6);
      }}
      async function loadRerunViewer(camera) {{
        const cam = String(camera || document.getElementById("cameraSelect").value || "workspace");
        const simViz = await waitForRerunReady();
        const runId = String((simViz && simViz.run_id) || activeRunId || "").trim();
        if (runId) activeRunId = runId;
        await waitForRerunSuccess(String(simViz.camera || cam), {{ deadlineMs: 90000, mountAttemptsPerLoop: 4, runId }});
        return true;
      }}
      async function pollSimVizUntilRrd(maxAttempts, delayMs, targetRunId) {{
        const attempts = Math.max(1, Number(maxAttempts || 36));
        const sleepMs = Math.max(250, Number(delayMs || 1500));
        const runId = String(targetRunId || "").trim();
        let last = null;
        for (let i = 0; i < attempts; i += 1) {{
          const statusPath = runId
            ? "/api/sim-viz/status?run_id=" + encodeURIComponent(runId)
            : "/api/sim-viz/status";
          const simViz = await loadJson(statusPath);
          last = simViz;
          const activeRunId = String((simViz && simViz.run_id) || "").trim();
          const runMatches = !runId || (activeRunId && activeRunId === runId);
          if (simViz && simViz.rrd_uri && runMatches) {{
            return simViz;
          }}
          await new Promise((resolve) => window.setTimeout(resolve, sleepMs));
        }}
        return last || {{}};
      }}
      async function loadFrankaDemo() {{
        const camera = String(document.getElementById("cameraSelect").value || "workspace");
        const result = await apiJson("/api/sim-viz/load-franka-demo", {{
          method: "POST",
          headers: {{ "content-type": "application/json" }},
          body: JSON.stringify({{ camera }}),
        }});
        await waitForRerunReady();
        await waitForRerunSuccess(camera, {{ deadlineMs: 90000, mountAttemptsPerLoop: 4 }});
        activeArtifactRender = "rerun";
        hideArtifactPreview();
        const simViz = await loadJson("/api/sim-viz/status");
        if (simViz && (simViz.rerun_ready || simViz.rrd_uri)) {{
          const stage = String(simViz.stage || "demo");
          appendChat(
            "assistant",
            "Loaded **stock Franka** demo — **stage**: `" + stage + "`, **camera**: `" + camera + "`, **rerun_ready**: `" +
              String(Boolean(simViz.rerun_ready)) + "`."
          );
        }} else if (result && result.ok === false) {{
          appendChat("error", "Franka demo load did not complete — check Rerun service status.");
        }}
        await refresh();
        return true;
      }}
      async function fetchWithTimeout(path, init, timeoutMs) {{
        const ms = Number(timeoutMs || 12000);
        const ctrl = new AbortController();
        const timer = window.setTimeout(() => ctrl.abort(), ms);
        try {{
          return await fetch(path, {{ ...(init || {{}}), signal: ctrl.signal }});
        }} finally {{
          window.clearTimeout(timer);
        }}
      }}
      async function apiJson(path, init) {{
        const req = init || {{}};
        const opts = {{
          ...req,
          credentials: "include",
          headers: {{
            ...(req.headers || {{}}),
          }},
        }};
        let resp;
        try {{
          resp = await fetchWithTimeout(path, opts, 12000);
        }} catch (err) {{
          if (err && err.name === "AbortError") {{
            throw new Error("Request timed out: " + String(path));
          }}
          throw err;
        }}
        let data = null;
        try {{
          data = await resp.json();
        }} catch (_err) {{
          data = null;
        }}
        if (!resp.ok) {{
          if (resp.status === 401) {{
            window.location.href = "/login-help.html";
            throw new Error("Authentication required. Open / and sign in with HTTP Basic Auth.");
          }}
          const detail =
            (data && (data.detail || data.error || data.message)) ||
            resp.statusText ||
            "request failed";
          throw new Error(String(detail));
        }}
        return data;
      }}
      async function loadJson(path) {{
        return await apiJson(path);
      }}
      function artifactPrefixValue() {{
        const input = document.getElementById("artifactPrefix");
        return String((input && input.value) || "").trim();
      }}
      async function refreshArtifactRuns() {{
        const prefix = artifactPrefixValue();
        const query = prefix ? ("?prefix=" + encodeURIComponent(prefix) + "&limit=100") : "?limit=100";
        const data = await apiJson("/api/artifacts/runs" + query);
        const select = document.getElementById("artifactRunSelect");
        if (select) {{
          select.innerHTML = '<option value="">(select discovered run)</option>';
          const runs = Array.isArray(data.runs) ? data.runs : [];
          for (const run of runs) {{
            const runId = String(run.run_id || "").trim();
            if (!runId) continue;
            const opt = document.createElement("option");
            opt.value = runId;
            opt.textContent = runId + (run.has_viewable ? " [viewable]" : " [download]");
            select.appendChild(opt);
          }}
        }}
        const list = document.getElementById("artifactList");
        if (list) {{
          const trunc = data.truncated ? " (truncated)" : "";
          list.textContent = "Runs discovered: " + String(data.total_runs || 0) + trunc;
        }}
        return true;
      }}
      async function loadArtifactsForSelectedRun() {{
        const select = document.getElementById("artifactRunSelect");
        const runId = String((select && select.value) || "").trim();
        if (!runId) throw new Error("Select a discovered run first");
        const prefix = artifactPrefixValue();
        const query = prefix ? ("?prefix=" + encodeURIComponent(prefix)) : "";
        const data = await apiJson("/api/artifacts/run/" + encodeURIComponent(runId) + query);
        const list = document.getElementById("artifactList");
        if (!list) return false;
        const artifacts = Array.isArray(data.artifacts) ? data.artifacts : [];
        if (!artifacts.length) {{
          list.innerHTML = "<p>No artifacts found for this run.</p>";
          return true;
        }}
        list.innerHTML = artifacts.map((item, idx) => {{
          const key = String(item.key || "");
          const render = String(item.render || "download");
          const s3uri = String(item.s3_uri || "");
          return (
            '<div style="padding:8px 0;border-top:1px solid #e2e8f0;">' +
            '<div><code>' + escapeHtml(key) + '</code></div>' +
            '<div style="font-size:12px;color:#64748b;">render=' + escapeHtml(render) + ' size=' + escapeHtml(String(item.size || 0)) + '</div>' +
            '<div style="margin-top:6px;"><button class="btn" type="button" data-action="load-artifact" data-run-id="' + escapeHtml(runId) + '" data-key="' + escapeHtml(key) + '" data-s3-uri="' + escapeHtml(s3uri) + '" data-render="' + escapeHtml(render) + '">Load</button></div>' +
            '</div>'
          );
        }}).join("");
        list.querySelectorAll("button[data-action='load-artifact']").forEach((btn) => {{
          btn.addEventListener("click", async (event) => {{
            event.preventDefault();
            const payload = {{
              run_id: String(btn.getAttribute("data-run-id") || "").trim(),
              key: String(btn.getAttribute("data-key") || "").trim(),
              s3_uri: String(btn.getAttribute("data-s3-uri") || "").trim(),
            }};
            await loadArtifact(payload);
          }});
        }});
        return true;
      }}
      async function loadArtifact(payload) {{
        const body = {{
          run_id: String((payload && payload.run_id) || "").trim(),
          key: String((payload && payload.key) || "").trim(),
          s3_uri: String((payload && payload.s3_uri) || "").trim(),
        }};
        const data = await apiJson("/api/sim-viz/load-artifact", {{
          method: "POST",
          headers: {{ "content-type": "application/json" }},
          body: JSON.stringify(body),
        }});
        const simViz = data.sim_viz || {{}};
        const render = String(data.render || simViz.artifact_render || "");
        activeArtifactRender = render;
        if (render === "rerun") {{
          hideArtifactPreview();
          await waitForRerunReady();
          await waitForRerunSuccess(String(simViz.camera || "workspace"), {{ deadlineMs: 90000, mountAttemptsPerLoop: 4 }});
        }} else {{
          showRerunPlaceholder("Artifact loaded. Use download/preview below.");
          await showArtifactPreview(simViz, render);
        }}
        appendChat(
          "assistant",
          "Loaded artifact `" + String(simViz.artifact_key || body.key || "") + "` with render `" + render + "`."
        );
        await refresh();
        return true;
      }}
      async function refresh() {{
        try {{
          const session = await loadJson("/api/session");
          if (session && session.workflow_draft && session.workflow_draft.yaml) {{
            setWorkflowYaml(session.workflow_draft.yaml, session.workflow_draft.validation || {{}});
          }}
          const assets = await loadJson("/api/sim-assets");
          const cameras = await loadJson("/api/sim-assets/cameras");
          const statusPath = activeRunId
            ? "/api/sim-viz/status?run_id=" + encodeURIComponent(activeRunId)
            : "/api/sim-viz/status";
          const simViz = await loadJson(statusPath);
          activeRunId = String((simViz && (simViz.active_run_id || simViz.run_id)) || activeRunId || "").trim();
          updateRunSelector(simViz);
          renderAssetsSummary(assets);
          document.getElementById("simRunId").textContent = String(simViz.run_id || "-");
          document.getElementById("simStage").textContent = String(simViz.stage || "idle");
          document.getElementById("simCamera").textContent = String(simViz.camera || "workspace");
          activeArtifactRender = String((simViz && simViz.artifact_render) || activeArtifactRender || "");
          const cta = document.getElementById("simvizCta");
          const ready = Boolean(simViz.rerun_ready || simViz.rrd_uri);
          if (cta) {{
            cta.hidden = ready && rerunIframeLoaded;
          }}
          if (activeArtifactRender && activeArtifactRender !== "rerun") {{
            showRerunPlaceholder("Non-RRD artifact loaded. Use preview/download below.");
            await showArtifactPreview(simViz, activeArtifactRender);
          }} else {{
            hideArtifactPreview();
          }}
          if (!ready && (!activeArtifactRender || activeArtifactRender === "rerun")) {{
            showRerunPlaceholder("Waiting for .rrd data. Click Load Franka in Rerun or submit Sim2Real.");
          }} else if (!rerunIframeLoaded && !rerunBootInProgress && (!activeArtifactRender || activeArtifactRender === "rerun")) {{
            showRerunPlaceholder("Rerun is ready. Click Load Rerun viewer or Load Franka in Rerun.");
          }}
          const robotPreset = document.getElementById("robotPreset");
          if (assets.selection && assets.selection.robot_preset) {{
            robotPreset.value = String(assets.selection.robot_preset);
          }}
          document.getElementById("simBackend").value = String((assets.selection && assets.selection.sim_backend) || "isaac");
          const props = Array.isArray(assets.selection && assets.selection.props) ? assets.selection.props.map(String) : [];
          document.getElementById("propCube").checked = props.includes("cube");
          const select = document.getElementById("cameraSelect");
          const selected = new Set((cameras.selected || []).map(String));
          const list = Array.isArray(cameras.cameras) ? cameras.cameras : [];
          const activeName = String(
            (selected.size ? [...selected][0] : null) || simViz.camera || "workspace"
          );
          document.getElementById("activeCameraLabel").textContent = activeName;
          select.innerHTML = "";
          for (const cam of list) {{
            const opt = document.createElement("option");
            opt.value = String(cam.name || "");
            opt.textContent = String(cam.name || "");
            if (selected.has(opt.value) || opt.value === activeName) opt.selected = true;
            select.appendChild(opt);
          }}
          renderCameraCards(list, activeName, simViz);
          const updatedAt = String(simViz.rrd_updated_at || "");
          if (rerunIframeLoaded && updatedAt && updatedAt !== lastRrdUpdatedAt) {{
            lastRrdUpdatedAt = updatedAt;
            reloadRerunIframe(simViz.camera || activeName);
          }} else if (updatedAt) {{
            lastRrdUpdatedAt = updatedAt;
          }}
        }} catch (err) {{
          document.getElementById("simvizCta").hidden = false;
          document.getElementById("simvizCta").textContent = "Failed to fetch sim viz status";
        }}
      }}
      function frustumSvg(camera, selected) {{
        const pos = Array.isArray(camera.pos) ? camera.pos : [0, 0, 0];
        const lookAt = Array.isArray(camera.look_at) ? camera.look_at : [0, 1, 0];
        const fov = Number(camera.fov || 60);
        const dx = lookAt[0] - pos[0];
        const dy = lookAt[1] - pos[1];
        const angle = Math.atan2(dy, dx);
        const spread = (fov * Math.PI / 180) / 2;
        const r = 50;
        const cx = 80;
        const cy = 80;
        const p1x = cx + r * Math.cos(angle - spread);
        const p1y = cy + r * Math.sin(angle - spread);
        const p2x = cx + r * Math.cos(angle + spread);
        const p2y = cy + r * Math.sin(angle + spread);
        const stroke = selected ? "#5e43f3" : "#94a3b8";
        const fill = selected ? "rgba(94,67,243,0.18)" : "rgba(148,163,184,0.15)";
        return `<svg width="160" height="160" viewBox="0 0 160 160" role="img" aria-label="camera frustum">
          <circle cx="${{cx}}" cy="${{cy}}" r="4" fill="${{stroke}}"/>
          <polygon points="${{cx}},${{cy}} ${{p1x}},${{p1y}} ${{p2x}},${{p2y}}" fill="${{fill}}" stroke="${{stroke}}"/>
          <text x="8" y="152" font-size="11" fill="#334155">${{String(camera.name || "")}}</text>
        </svg>`;
      }}
      function renderCameraCards(list, activeName, simViz) {{
        const holder = document.getElementById("cameraCards");
        holder.innerHTML = "";
        for (const cam of list) {{
          const name = String(cam.name || "");
          const selected = name === activeName;
          const card = document.createElement("div");
          card.className = "camera-card" + (selected ? " selected" : "");
          const pos = Array.isArray(cam.pos) ? cam.pos.map((v) => Number(v).toFixed(2)).join(", ") : "-";
          const look = Array.isArray(cam.look_at) ? cam.look_at.map((v) => Number(v).toFixed(2)).join(", ") : "-";
          const res = Array.isArray(cam.resolution) ? cam.resolution.join("x") : "640x480";
          card.innerHTML = `
            <h4>` + name + (selected ? ' <span class="badge">selected</span>' : '') + `</h4>
            <div class="camera-meta">placement: ${{String(cam.placement || "custom")}} - fov ${{Number(cam.fov || 60)}}deg - ${{res}}</div>
            <div class="camera-meta">pos [${{pos}}] - look_at [${{look}}]</div>
            <div class="camera-frustum">${{frustumSvg(cam, selected)}}</div>
            <div class="camera-actions">
              <button class="btn" type="button" data-action="select" data-camera="${{name}}">Select</button>
              <button class="btn" type="button" data-action="preview" data-camera="${{name}}">Preview in Rerun</button>
            </div>`;
          holder.appendChild(card);
        }}
        const entity = String(simViz.preview_entity || ("world/cameras/" + activeName));
        const rollout = "rollouts/latest/" + activeName + "/camera";
        document.getElementById("rerunEntityHint").textContent =
          (simViz.rerun_ready || simViz.rrd_uri)
            ? "Rerun entities: " + entity + " (frustum) - " + rollout + " (rollout frames when available)"
            : "Preview in Rerun to log camera frustums; rollout frames appear after Sim2Real runs.";
        holder.querySelectorAll("button[data-action]").forEach((btn) => {{
          btn.addEventListener("click", async (event) => {{
            event.preventDefault();
            const camera = String(btn.getAttribute("data-camera") || "");
            const action = String(btn.getAttribute("data-action") || "");
            const label = action === "select" ? "Select " + camera : "Preview " + camera;
            setStatus(label + "...");
            showToast(label, "info");
            try {{
              if (action === "select") {{
                await selectCamera(camera);
              }} else {{
                await previewCamera(camera);
              }}
              setStatus(label + " done");
              showToast(label + " done", "success");
            }} catch (err) {{
              setStatus(label + " failed");
              showToast(String(err), "error");
            }}
          }});
        }});
      }}
      function renderAssetsSummary(assets) {{
        const selection = (assets && assets.selection) || {{}};
        const resolved = (assets && assets.resolved_uris) || {{}};
        const scene = String(selection.scene_spec_uri || "stock://scene/default");
        const robot = String(selection.robot_spec_uri || "stock://robot/franka");
        const cameras = String(selection.cameras_uri || "stock://cameras/default");
        const backend = String(selection.sim_backend || "isaac");
        const props = Array.isArray(selection.props) ? selection.props : [];
        const propsText = props.length ? props.join(", ") : "none";
        document.getElementById("assetsSummary").innerHTML =
          "<div><span class='pill'>scene: " + escapeHtml(scene) + "</span></div>" +
          "<div style='margin-top:6px;'><span class='pill'>robot: " + escapeHtml(robot) + "</span></div>" +
          "<div style='margin-top:6px;'><span class='pill'>cameras: " + escapeHtml(cameras) + "</span></div>" +
          "<div style='margin-top:6px;'><span class='pill'>backend: " + escapeHtml(backend) + "</span></div>" +
          "<div style='margin-top:6px;'>props: " + escapeHtml(propsText) + "</div>" +
          "<div style='margin-top:6px;'>resolved scene URI: <code>" + escapeHtml(String(resolved.scene_spec_uri || scene)) + "</code></div>";
      }}
      function selectionPayloadFromUi() {{
        const robotPreset = String(document.getElementById("robotPreset").value || "franka");
        const sceneMode = String(document.getElementById("sceneMode").value || "stock");
        const cameraMode = String(document.getElementById("cameraMode").value || "stock");
        const backend = String(document.getElementById("simBackend").value || "isaac");
        return {{
          scene_spec_uri: sceneMode === "stock" ? "stock://scene/default" : "",
          robot_spec_uri: robotPreset === "franka" ? "stock://robot/franka" : "",
          cameras_uri: cameraMode === "stock" ? "stock://cameras/default" : "",
          robot_preset: robotPreset,
          sim_backend: backend,
          props: document.getElementById("propCube").checked ? ["cube"] : []
        }};
      }}
      function updateRunSelector(simViz) {{
        const select = document.getElementById("runIdSelect");
        if (!select) return;
        const current = String((simViz && simViz.run_id) || "").trim();
        const runs = Array.isArray(simViz && simViz.available_run_ids) ? simViz.available_run_ids.map(String) : [];
        select.innerHTML = '<option value="">(select run)</option>';
        for (const runId of runs) {{
          const opt = document.createElement("option");
          opt.value = runId;
          opt.textContent = runId;
          if (runId === current) opt.selected = true;
          select.appendChild(opt);
        }}
        const input = document.getElementById("runIdInput");
        if (input && current) input.value = current;
      }}
      async function loadRunData() {{
        const input = document.getElementById("runIdInput");
        const runId = String((input && input.value) || "").trim();
        if (!runId) {{
          throw new Error("Enter or select a run_id first");
        }}
        const data = await apiJson("/api/sim-viz/load-run", {{
          method: "POST",
          headers: {{ "content-type": "application/json" }},
          body: JSON.stringify({{ run_id: runId }}),
        }});
        activeRunId = runId;
        appendChat("assistant", "Loaded run context — **run_id**: `" + runId + "`.");
        if (data && data.sim_viz && (data.sim_viz.rrd_uri || data.sim_viz.rerun_ready)) {{
          await waitForRerunSuccess(String(data.sim_viz.camera || "workspace"), {{ runId }});
        }}
        await refresh();
      }}
      async function selectCamera(camera) {{
        const selected = String(camera || "");
        await apiJson("/api/sim-assets/cameras/selection", {{
          method: "PUT",
          headers: {{ "content-type": "application/json" }},
          body: JSON.stringify({{ selected: selected ? [selected] : [] }}),
        }});
        await apiJson("/api/sim-assets/selection", {{
          method: "POST",
          headers: {{ "content-type": "application/json" }},
          body: JSON.stringify(selectionPayloadFromUi()),
        }});
        await refresh();
      }}
      async function previewCamera(camera) {{
        const data = await apiJson("/api/sim-viz/camera-preview", {{
          method: "POST",
          headers: {{ "content-type": "application/json" }},
          body: JSON.stringify({{ camera }}),
        }});
        await waitForRerunReady();
        await waitForRerunSuccess(camera, {{ deadlineMs: 90000, mountAttemptsPerLoop: 4 }});
        const entity = String(data.entity_path || ("world/cameras/" + camera));
        appendChat("assistant", "Previewing `" + camera + "` in Rerun at `" + entity + "`.");
        await refresh();
      }}
      async function applySelection() {{
        const data = await apiJson("/api/sim-assets/selection", {{
          method: "POST",
          headers: {{ "content-type": "application/json" }},
          body: JSON.stringify(selectionPayloadFromUi()),
        }});
        if (data.sim_viz && rerunIframeLoaded) {{
          reloadRerunIframe(data.sim_viz.camera || "workspace");
        }}
        await refresh();
      }}
      async function submitWorkflow() {{
        const baseline = await loadJson("/api/sim-viz/status");
        const baselineUpdatedAt = String((baseline && baseline.rrd_updated_at) || "").trim();
        const data = await apiJson("/api/workflows/sim2real/submit", {{
          method: "POST",
          headers: {{ "content-type": "application/json" }},
          body: JSON.stringify({{}}),
        }});
        appendChat("assistant", `Submitted Sim2Real run: **${{data.run_id || "unknown"}}**`);
        appendChat("assistant", "Watching sim progress: polling `/api/sim-viz/status` until `.rrd` is available and iframe blob mount reaches `SUCCESS`.");
        const submittedRunId = String(data.run_id || "").trim();
        if (submittedRunId) activeRunId = submittedRunId;
        const simViz = await pollSimVizUntilRrd(60, 1500, submittedRunId);
        if (simViz && simViz.rrd_uri) {{
          await waitForRerunSuccess(
            simViz.camera || "workspace",
            {{
              deadlineMs: 180000,
              mountAttemptsPerLoop: 5,
              runId: submittedRunId,
              baselineRrdUpdatedAt: baselineUpdatedAt,
            }}
          );
          if (lastRerunBlobStatus !== RERUN_BLOB_SUCCESS || lastRerunMountStatus !== RERUN_MOUNT_SUCCESS) {{
            throw new Error("Rerun blob/iframe did not reach SUCCESS after workflow submit");
          }}
          appendChat(
            "assistant",
            "Rerun update: **run_id** `" +
              String(simViz.run_id || data.run_id || "unknown") +
              "`, **stage** `" +
              String(simViz.stage || "running") +
              "`, **iframe** `/rerun/`, **blob_mount** `" + RERUN_BLOB_SUCCESS + "`"
          );
        }} else {{
          throw new Error("Sim2Real run submitted, but no .rrd is available yet after polling");
        }}
        await refresh();
      }}
      async function showWorkflowStatus() {{
        const status = await loadJson("/api/workflows/sim2real/status");
        appendChat(
          "assistant",
          "Latest workflow status:\\n- run_id: `" +
            String((status.latest_submit || {{}}).run_id || "none") +
            "`\\n- stage: `" +
            String((status.sim_viz || {{}}).stage || "idle") +
            "`"
        );
      }}
      async function openRerunTab() {{
        const statusPath = activeRunId
          ? "/api/sim-viz/status?run_id=" + encodeURIComponent(activeRunId)
          : "/api/sim-viz/status";
        const simViz = await loadJson(statusPath);
        const camera = String(simViz.camera || document.getElementById("cameraSelect").value || "workspace");
        const src = await rerunIframeSrc(camera, String((simViz && simViz.run_id) || activeRunId || "").trim());
        window.open(src, "_blank", "noopener");
      }}
      async function restoreSession() {{
        try {{
          const session = await loadJson("/api/session");
          const hist = Array.isArray(session.chat_history) ? session.chat_history : [];
          for (const msg of hist) {{
            const role = String(msg.role || "");
            const content = String(msg.content || "").trim();
            if (!content || (role !== "user" && role !== "assistant")) continue;
            appendChat(role, content);
            chatHistory.push({{ role, content }});
          }}
          const draft = session.workflow_draft;
          if (draft && draft.yaml) {{
            setWorkflowYaml(draft.yaml, draft.validation || draft);
          }}
        }} catch (_err) {{
          // Session restore is best-effort on first load.
        }}
      }}
      async function ensureFrankaRerunLoaded() {{
        const simViz = await loadJson("/api/sim-viz/status");
        const artifactRender = String((simViz && simViz.artifact_render) || "");
        if (artifactRender && artifactRender !== "rerun") {{
          activeArtifactRender = artifactRender;
          await showArtifactPreview(simViz, artifactRender);
          return;
        }}
        const camera = String(simViz.camera || "workspace");
        if (!simViz.rerun_ready && !simViz.rrd_uri) {{
          setStatus("Loading Franka demo...");
          showToast("Loading stock Franka in Rerun", "info");
          await loadFrankaDemo();
          return;
        }}
        if (!rerunIframeLoaded) {{
          setStatus("Opening Rerun viewer...");
          await loadRerunViewer(camera);
        }}
      }}
      async function bootPage() {{
        rerunBootInProgress = true;
        showRerunPlaceholder("Loading stock Franka preview...");
        setStatus("Preloading Rerun…");
        try {{
          await warmRerunBundle();
        }} catch (warmErr) {{
          console.warn("rerun bundle warm failed", warmErr);
        }}
        try {{
          await restoreSession();
        }} catch (_err) {{
          // Session restore is best-effort on first paint.
        }}
        try {{
          await refresh();
          try {{
            await refreshArtifactRuns();
          }} catch (_artifactErr) {{
            // artifact discovery is optional when S3 credentials are missing.
          }}
          await ensureFrankaRerunLoaded();
          setStatus("Ready");
          showToast("Franka demo ready in Rerun", "success");
        }} catch (err) {{
          console.warn("franka auto-load failed", err);
          showRerunPlaceholder("Could not auto-load Franka. Click Load Franka in Rerun.");
          showToast(String(err && err.message ? err.message : err), "error");
          setStatus("Ready");
        }} finally {{
          rerunBootInProgress = false;
        }}
      }}
      let refreshTimer = null;
      function startPeriodicRefresh() {{
        if (refreshTimer !== null) return;
        refreshTimer = window.setInterval(() => {{
          refresh().catch(() => {{ /* periodic refresh is best-effort */ }});
        }}, 10000);
      }}
      function startApp() {{
        try {{
          wireUi();
          setStatus("UI wired");
        }} catch (err) {{
          showToast("UI wiring failed: " + String(err), "error");
          console.error(err);
        }}
        bootPage().catch((err) => {{
          showToast("Boot failed: " + String(err), "error");
          console.error(err);
        }});
        const armPeriodic = () => startPeriodicRefresh();
        document.addEventListener("click", armPeriodic, {{ once: true }});
        document.addEventListener("keydown", armPeriodic, {{ once: true }});
      }}
      if (document.readyState === "loading") {{
        document.addEventListener("DOMContentLoaded", startApp);
      }} else {{
        startApp();
      }}
      }} catch (bootErr) {{
        console.error("NPA Agent UI failed to boot", bootErr);
        const bar = document.getElementById("statusBar");
        if (bar) bar.textContent = "UI boot error — reload or check console";
      }}
      }})();
    </script>
  </body>
</html>
HTML
sudo python3 -m venv /opt/npa-agent/venv
sudo /opt/npa-agent/venv/bin/pip install --upgrade pip
sudo /opt/npa-agent/venv/bin/pip install fastapi uvicorn httpx pyyaml boto3 "rerun-sdk>=0.32"
sudo /opt/npa-agent/venv/bin/python /opt/npa-agent/bootstrap_rrd.py
sudo systemctl restart npa-rerun || true
cat <<'UNIT' | sudo tee /etc/systemd/system/npa-agent-backend.service >/dev/null
[Unit]
Description=NPA agent backend
After=network.target
[Service]
Type=simple
EnvironmentFile=-/opt/npa-agent/llm.env
EnvironmentFile=-/opt/npa-agent/s3.env
EnvironmentFile=-/opt/npa-agent/nebius.env
ExecStart=/opt/npa-agent/venv/bin/uvicorn backend:app --host 0.0.0.0 --port {backend_port}
WorkingDirectory=/opt/npa-agent
Restart=always
[Install]
WantedBy=multi-user.target
UNIT
cat <<'UNIT' | sudo tee /etc/systemd/system/npa-rerun.service >/dev/null
[Unit]
Description=NPA rerun service
After=network.target
[Service]
Type=simple
ExecStart=/opt/npa-agent/venv/bin/rerun /opt/npa-agent/sim2real.rrd --serve-web --web-viewer --bind 0.0.0.0 --web-viewer-port {rerun_port} --port 9876
WorkingDirectory=/opt/npa-agent
Restart=always
StartLimitIntervalSec=0
[Install]
WantedBy=multi-user.target
UNIT
sudo htpasswd -bc /etc/nginx/.npa-agent-htpasswd {shlex.quote(auth_user)} {shlex.quote(auth_password)}
{https_ssl_setup}
cat <<'NGINX' | sudo tee /etc/nginx/sites-available/npa-agent >/dev/null
server {{
  listen {agent_port};
  server_name _;
{nginx_site_body}
}}
{https_server_block}
NGINX
sudo ln -sf /etc/nginx/sites-available/npa-agent /etc/nginx/sites-enabled/npa-agent
sudo rm -f /etc/nginx/sites-enabled/default
sudo systemctl daemon-reload
sudo systemctl reset-failed npa-agent-backend npa-rerun nginx || true
sudo systemctl enable --now npa-agent-backend npa-rerun nginx
sudo systemctl restart npa-rerun nginx
sudo systemctl restart npa-agent-backend
"""
    local_setup_script = ""
    # Use a unique remote path so concurrent bootstrap runs cannot clobber each other.
    remote_setup_script = f"/tmp/npa-agent-bootstrap-{secrets.token_hex(6)}.sh"
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            handle.write(setup_script)
            local_setup_script = handle.name
        ssh.upload_file(local_setup_script, remote_setup_script)
        ssh.run_or_raise(f"chmod 700 {shlex.quote(remote_setup_script)} && {shlex.quote(remote_setup_script)}")
    finally:
        if local_setup_script:
            Path(local_setup_script).unlink(missing_ok=True)
        ssh.run(f"rm -f {shlex.quote(remote_setup_script)}")
    _write_agent_llm_env(ssh, tf_api_key=tf_api_key, llm_model=llm_model)
    _write_agent_s3_env(
        ssh,
        bucket=s3_bucket,
        endpoint=s3_endpoint,
        access_key=s3_access_key,
        secret_key=s3_secret_key,
        region=s3_region,
    )
    _write_agent_nebius_env(
        ssh,
        project_id=nebius_project_id,
        tenant_id=nebius_tenant_id,
        region=s3_region,
        service_account_id=service_account_id,
        bucket=s3_bucket,
        endpoint=s3_endpoint,
        access_key=s3_access_key,
        secret_key=s3_secret_key,
    )
    if (
        tf_api_key.strip()
        or (s3_bucket.strip() and s3_access_key.strip() and s3_secret_key.strip())
        or (nebius_project_id.strip() and s3_access_key.strip() and s3_secret_key.strip())
    ):
        ssh.run_or_raise(
            "sudo systemctl reset-failed npa-agent-backend || true; "
            "sudo systemctl restart npa-agent-backend"
        )


def _health(
    url: str,
    *,
    user: str,
    password: str,
    timeout: float = 5.0,
    verify: bool = True,
) -> tuple[bool, int]:
    try:
        response = httpx.get(url, auth=(user, password), timeout=timeout, verify=verify)
    except httpx.HTTPError:
        return False, 0
    return response.status_code == 200, response.status_code


@app.command("deploy")
def deploy_cmd(
    project: str = typer.Option(DEFAULT_PROJECT_ALIAS, "--project", help="NPA project alias to store config under."),
    name: str = typer.Option(DEFAULT_AGENT_NAME, "--name", help="Agent deployment name."),
    project_id: str = typer.Option("", "--project-id", help="Nebius project ID."),
    tenant_id: str = typer.Option("", "--tenant-id", help="Nebius tenant ID."),
    region: str = typer.Option("us-central1", "--region", help="Nebius region."),
    ssh_user: str = typer.Option("ubuntu", "--ssh-user", help="SSH username."),
    ssh_public_key_path: str = typer.Option("~/.ssh/id_ed25519.pub", "--ssh-public-key-path", help="SSH public key path for Terraform."),
    tf_var: list[str] = typer.Option([], "--tf-var", help="Additional Terraform var key=value."),
    agent_port: int = typer.Option(DEFAULT_AGENT_PORT, "--agent-port", help="Public agent UI port."),
    backend_port: int = typer.Option(DEFAULT_BACKEND_PORT, "--backend-port", help="Internal agent backend port."),
    rerun_port: int = typer.Option(DEFAULT_RERUN_PORT, "--rerun-port", help="Rerun service port."),
    no_public_https: bool = typer.Option(
        False,
        "--no-public-https",
        help="Disable HTTPS on port 443 (customer access uses http://IP:agent-port only).",
    ),
) -> None:
    """Provision VM + bootstrap the public NPA agent stack."""
    saved_env = resolve_environment(
        project,
        project_id=project_id or None,
        tenant_id=tenant_id or None,
        region=region or None,
    )
    env_project_id = project_id or (saved_env.project_id if saved_env else "")
    env_tenant_id = tenant_id or (saved_env.tenant_id if saved_env else "")
    env_region = region or (saved_env.region if saved_env else "")
    if not env_project_id or not env_tenant_id or not env_region:
        _fail("--project-id, --tenant-id, and --region are required")

    from npa.clients.nebius import NebiusError, bootstrap_agent_environment, get_iam_token

    try:
        creds = bootstrap_agent_environment(
            env_project_id,
            env_tenant_id,
            env_region,
            on_status=lambda msg: typer.echo(f"  {msg}"),
        )
        iam_token = get_iam_token()
    except NebiusError as exc:
        _fail(f"Nebius bootstrap failed: {exc}")

    merged_vars: dict[str, str] = {
        "nebius_project_id": env_project_id,
        "nebius_region": env_region,
        "service_account_id": str(creds.get("service_account_id", "")),
        "iam_token": iam_token,
        "nebius_api_key": str(creds.get("nebius_api_key", "")),
        "nebius_secret_key": str(creds.get("nebius_secret_key", "")),
        "s3_bucket": str(creds.get("s3_bucket", "")),
        "s3_endpoint": str(creds.get("s3_endpoint", "")),
        "instance_name": f"agent-{project}-{name}",
        "server_port": str(agent_port),
        "workbench_type": "lerobot",
        "gpu_platform": "cpu-d3",
        "gpu_preset": "8vcpu-32gb",
        "ssh_user": ssh_user,
        "ssh_public_key_path": ssh_public_key_path,
        "enable_preemptible": "false",
    }
    for item in tf_var:
        if "=" not in item:
            _fail(f"Invalid --tf-var value {item!r}; expected key=value")
        key, value = item.split("=", 1)
        merged_vars[key.strip()] = value.strip()

    try:
        tf_dir = provisioner.prepare_working_dir(
            project,
            name,
            bucket=merged_vars.get("s3_bucket", ""),
            region=env_region,
            endpoint=merged_vars.get("s3_endpoint", ""),
        )
        provisioner.init(
            tf_dir=tf_dir,
            backend_config={
                "access_key": merged_vars.get("nebius_api_key", ""),
                "secret_key": merged_vars.get("nebius_secret_key", ""),
            },
        )
        tf_outputs = provisioner.apply(tf_dir=tf_dir, tf_vars=merged_vars)
    except ProvisionerError as exc:
        _fail(f"Terraform deploy failed: {exc}")

    public_ip = str(tf_outputs.get("vm_ip", ""))
    instance_id = str(tf_outputs.get("instance_id", ""))
    ssh_key_path = str(tf_outputs.get("ssh_key_path", "") or ssh_public_key_path.removesuffix(".pub"))
    if not _is_routable_public_ip(public_ip):
        _fail("Terraform output did not include a routable public IP")

    auth_password = secrets.token_urlsafe(18)
    auth_path = _write_auth_secret(
        project_alias=project,
        name=name,
        user=DEFAULT_AGENT_USER,
        password=auth_password,
    )
    tf_api_key, llm_model = _resolve_deploy_llm_credentials()
    if not tf_api_key:
        typer.echo(
            "Warning: Token Factory API key not found in credentials; "
            "agent chat will return 503 until `npa agent bootstrap` with a configured key.",
            err=True,
        )
    public_https = not no_public_https
    try:
        _bootstrap_agent_stack(
            host=public_ip,
            ssh_user=ssh_user,
            ssh_key_path=ssh_key_path,
            auth_user=DEFAULT_AGENT_USER,
            auth_password=auth_password,
            agent_port=agent_port,
            backend_port=backend_port,
            rerun_port=rerun_port,
            llm_model=llm_model,
            tf_api_key=tf_api_key,
            s3_bucket=str(merged_vars.get("s3_bucket", "")),
            s3_endpoint=str(merged_vars.get("s3_endpoint", "")),
            s3_access_key=str(merged_vars.get("nebius_api_key", "")),
            s3_secret_key=str(merged_vars.get("nebius_secret_key", "")),
            s3_region=env_region,
            nebius_project_id=env_project_id,
            nebius_tenant_id=env_tenant_id,
            service_account_id=str(creds.get("service_account_id", "")),
            public_https=public_https,
        )
    except (ConfigError, SSHError, ValueError) as exc:
        _fail(f"VM bootstrap failed: {exc}")

    ingress_ports: list[int] = [agent_port, rerun_port]
    if public_https:
        ingress_ports.append(DEFAULT_HTTPS_PORT)
    try:
        ensure_ingress(vm_id=instance_id, ports=tuple(ingress_ports), tool="agent")
    except NetworkIngressError as exc:
        _fail(f"npa network ensure-ingress failed: {exc}")

    urls = build_agent_urls(public_ip, agent_port=agent_port, public_https=public_https)
    agent_credentials = _agent_credentials_payload(creds)
    record = AgentConfig(
        project_alias=project,
        name=name,
        project_id=env_project_id,
        tenant_id=env_tenant_id,
        region=env_region,
        public_ip=public_ip,
        instance_id=instance_id,
        agent_url=urls["agent_url"],
        rerun_url=urls["rerun_url"],
        sim_viz_url=urls["sim_viz_url"],
        sim_assets_url=urls["sim_assets_url"],
        cameras_api_url=urls["cameras_api_url"],
        auth_user=DEFAULT_AGENT_USER,
        auth_secret_path=str(auth_path),
        llm_provider=DEFAULT_LLM_PROVIDER,
        llm_model=DEFAULT_LLM_MODEL,
        public_url=urls["public_url"],
        public_https=public_https,
        direct_url=urls["direct_url"],
        ssh_key_path=ssh_key_path,
        service_account_id=str(creds.get("service_account_id", "")),
        credentials=agent_credentials,
    )
    _store_agent_record(project, name, record.to_dict())
    write_config(
        {
            "projects": {
                project: {
                    "project_id": env_project_id,
                    "tenant_id": env_tenant_id,
                    "region": env_region,
                    "terraform_state": {
                        "bucket": merged_vars.get("s3_bucket", ""),
                        "endpoint": merged_vars.get("s3_endpoint", ""),
                        "access_key": merged_vars.get("nebius_api_key", ""),
                        "secret_key": merged_vars.get("nebius_secret_key", ""),
                    },
                }
            }
        }
    )

    typer.echo(f"Customer URL: {urls['public_url']}")
    typer.echo(f"public_url: {urls['public_url']}")
    if public_https:
        typer.echo(
            "Note: HTTPS uses a self-signed certificate — browsers will warn once; "
            "choose to proceed or use curl with -k."
        )
        typer.echo(f"direct_url: {urls['direct_url']}")
    typer.echo(f"rerun_url: {urls['rerun_url']}")
    typer.echo(f"sim_viz_url: {urls['sim_viz_url']}")
    typer.echo(f"sim_assets_url: {urls['sim_assets_url']}")
    typer.echo(f"cameras_api_url: {urls['cameras_api_url']}")
    typer.echo(f"llm: {DEFAULT_LLM_PROVIDER}:{DEFAULT_LLM_MODEL}")
    typer.echo(f"auth_user: {DEFAULT_AGENT_USER}")
    typer.echo(f"auth_secret_path: {auth_path}")
    typer.echo(f"auth_password: {redact_value(auth_password)}")


@app.command("bootstrap")
def bootstrap_cmd(
    project: str = typer.Option(DEFAULT_PROJECT_ALIAS, "--project", help="NPA project alias."),
    name: str = typer.Option(DEFAULT_AGENT_NAME, "--name", help="Agent deployment name."),
    ssh_user: str = typer.Option("ubuntu", "--ssh-user", help="SSH username."),
    ssh_key: str = typer.Option("", "--ssh-key", help="SSH private key path (defaults to agent record or NPA_SSH_KEY)."),
    agent_port: int = typer.Option(DEFAULT_AGENT_PORT, "--agent-port", help="Public agent UI port."),
    backend_port: int = typer.Option(DEFAULT_BACKEND_PORT, "--backend-port", help="Internal agent backend port."),
    rerun_port: int = typer.Option(DEFAULT_RERUN_PORT, "--rerun-port", help="Rerun service port."),
    refresh_credentials: bool = typer.Option(
        False,
        "--refresh-credentials",
        help="Re-provision the long-lived npa-agent service account and restage VM credentials.",
    ),
    no_public_https: bool = typer.Option(
        False,
        "--no-public-https",
        help="Disable HTTPS on port 443 (customer access uses http://IP:agent-port only).",
    ),
) -> None:
    """Re-bootstrap agent UI/backend/nginx on an existing VM (refresh without Terraform)."""
    record = _agent_record(project, name)
    if not record:
        _fail(f"Agent config not found for {project}/{name}")
    public_ip = str(record.get("public_ip", "")).strip()
    if not _is_routable_public_ip(public_ip):
        _fail("agent VM does not have a routable public IP")
    public_https = not no_public_https
    ssh_key_path = _resolve_agent_ssh_key(record, cli_ssh_key=ssh_key or None)
    if not Path(ssh_key_path).expanduser().exists():
        _fail(
            f"SSH private key not found at {ssh_key_path!r}. "
            "Pass --ssh-key, set NPA_SSH_KEY, or redeploy to persist ssh_key_path on the agent record."
        )
    try:
        auth_user, auth_password = _load_auth_secret(str(record.get("auth_secret_path", "")))
    except ValueError as exc:
        _fail(str(exc))
    tf_api_key, llm_model = _resolve_deploy_llm_credentials()
    llm_block = record.get("llm", {}) if isinstance(record.get("llm"), dict) else {}
    if isinstance(llm_block.get("model"), str) and llm_block["model"].strip():
        llm_model = llm_block["model"].strip()
    if not tf_api_key:
        typer.echo(
            "Warning: Token Factory API key not found; chat endpoint will return 503.",
            err=True,
        )
    project_id = str(record.get("project_id", "")).strip()
    tenant_id = str(record.get("tenant_id", "")).strip()
    region = str(record.get("region", "") or "eu-north1")
    s3_bucket, s3_endpoint, s3_access_key, s3_secret_key, service_account_id = (
        _resolve_agent_storage_credentials(project, record)
    )
    if not service_account_id:
        service_account_id = _resolve_agent_service_account_id(project, record)
    if refresh_credentials:
        if not (project_id and tenant_id and region):
            _fail("agent record is missing project_id, tenant_id, or region for credential refresh")
        from npa.clients.nebius import NebiusError, bootstrap_agent_environment

        creds: dict[str, str] | None = None
        try:
            creds = bootstrap_agent_environment(
                project_id,
                tenant_id,
                region,
                on_status=lambda msg: typer.echo(f"  {msg}"),
            )
        except NebiusError as exc:
            typer.echo(
                f"Warning: npa-agent provisioning failed ({exc}); reusing existing credentials.",
                err=True,
            )
        if creds is None:
            creds = _creds_from_terraform_state(project, record)
        if creds is None:
            _fail("Nebius credential refresh failed and no terraform_state fallback is configured")
        agent_credentials = _agent_credentials_payload(creds)
        s3_bucket = agent_credentials["s3_bucket"]
        s3_endpoint = agent_credentials["s3_endpoint"]
        s3_access_key = agent_credentials["access_key"]
        s3_secret_key = agent_credentials["secret_key"]
        service_account_id = agent_credentials["service_account_id"]
        if not service_account_id:
            service_account_id = _resolve_agent_service_account_id(project, record)
            agent_credentials["service_account_id"] = service_account_id
        if s3_access_key and s3_secret_key:
            write_config(
                {
                    "projects": {
                        project: {
                            "terraform_state": {
                                "bucket": s3_bucket,
                                "endpoint": s3_endpoint,
                                "access_key": s3_access_key,
                                "secret_key": s3_secret_key,
                            },
                        }
                    }
                }
            )
    try:
        _bootstrap_agent_stack(
            host=public_ip,
            ssh_user=ssh_user,
            ssh_key_path=ssh_key_path,
            auth_user=auth_user,
            auth_password=auth_password,
            agent_port=agent_port,
            backend_port=backend_port,
            rerun_port=rerun_port,
            llm_model=llm_model,
            tf_api_key=tf_api_key,
            s3_bucket=s3_bucket,
            s3_endpoint=s3_endpoint,
            s3_access_key=s3_access_key,
            s3_secret_key=s3_secret_key,
            s3_region=region,
            nebius_project_id=project_id,
            nebius_tenant_id=tenant_id,
            service_account_id=service_account_id,
            public_https=public_https,
        )
    except (ConfigError, SSHError, ValueError) as exc:
        _fail(f"VM bootstrap failed: {exc}")
    instance_id = str(record.get("instance_id", "")).strip()
    if instance_id:
        ingress_ports: list[int] = [agent_port, rerun_port]
        if public_https:
            ingress_ports.append(DEFAULT_HTTPS_PORT)
        try:
            ensure_ingress(vm_id=instance_id, ports=tuple(ingress_ports), tool="agent")
        except NetworkIngressError as exc:
            typer.echo(
                f"Warning: npa network ensure-ingress failed ({exc}). "
                "Customer HTTPS on port 443 may be unreachable until ingress is opened.",
                err=True,
            )
    urls = build_agent_urls(public_ip, agent_port=agent_port, public_https=public_https)
    updated = dict(record)
    updated.update(urls)
    updated["public_https"] = public_https
    updated["ssh_key_path"] = ssh_key_path
    if service_account_id:
        updated["service_account_id"] = service_account_id
        _persist_agent_service_account_id(service_account_id)
    if s3_bucket and s3_access_key and s3_secret_key:
        updated["credentials"] = _credentials_block_from_storage(
            service_account_id=service_account_id,
            s3_bucket=s3_bucket,
            s3_endpoint=s3_endpoint,
            s3_access_key=s3_access_key,
            s3_secret_key=s3_secret_key,
        )
    elif refresh_credentials:
        updated["credentials"] = agent_credentials
    _store_agent_record(project, name, updated)
    typer.echo(f"Customer URL: {urls['public_url']}")
    typer.echo(f"bootstrapped: {project}/{name} at {urls['public_url']}")
    if public_https:
        typer.echo(f"direct_url: {urls['direct_url']}")


@app.command("status")
def status_cmd(
    project: str = typer.Option(DEFAULT_PROJECT_ALIAS, "--project", help="NPA project alias."),
    name: str = typer.Option(DEFAULT_AGENT_NAME, "--name", help="Agent deployment name."),
    output_json: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Show agent status, URLs, and health checks."""
    record = _agent_record(project, name)
    if not record:
        _fail(f"Agent config not found for {project}/{name}")
    try:
        auth_user, auth_password = _load_auth_secret(str(record.get("auth_secret_path", "")))
    except ValueError as exc:
        _fail(str(exc))
    agent_url = str(record.get("agent_url", ""))
    rerun_url = str(record.get("rerun_url", ""))
    sim_viz_url = str(record.get("sim_viz_url", rerun_url))
    sim_assets_url = str(record.get("sim_assets_url", agent_url))
    cameras_api_url = str(
        record.get("cameras_api_url", f"{agent_url.rstrip('/')}/assets/api/sim-assets/cameras")
    )
    public_url = _record_customer_url(record)
    tls_verify = _record_tls_verify(record)
    ui_ok, ui_code = _health(agent_url, user=auth_user, password=auth_password, verify=tls_verify)
    rerun_ok, rerun_code = _health(sim_viz_url, user=auth_user, password=auth_password, verify=tls_verify)
    payload = {
        "project": project,
        "name": name,
        "public_ip": record.get("public_ip", ""),
        "public_url": public_url,
        "public_https": _record_public_https(record),
        "direct_url": record.get("direct_url", ""),
        "ui_url": agent_url,
        "rerun_url": rerun_url,
        "sim_viz_url": sim_viz_url,
        "sim_assets_url": sim_assets_url,
        "cameras_api_url": cameras_api_url,
        "health": bool(ui_ok and rerun_ok),
        "ui_status_code": ui_code,
        "rerun_status_code": rerun_code,
        "llm": record.get("llm", {}),
    }
    if output_json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    for key, value in payload.items():
        typer.echo(f"{key}: {value}")


@app.command("destroy")
def destroy_cmd(
    project: str = typer.Option(DEFAULT_PROJECT_ALIAS, "--project", help="NPA project alias."),
    name: str = typer.Option(DEFAULT_AGENT_NAME, "--name", help="Agent deployment name."),
) -> None:
    """Destroy agent VM/resources and remove saved config entry."""
    record = _agent_record(project, name)
    if not record:
        _fail(f"Agent config not found for {project}/{name}")
    state = resolve_terraform_state(project)
    region = str(record.get("region", "")) or "us-central1"
    tf_vars = {
        "nebius_project_id": str(record.get("project_id", "")),
        "nebius_region": region,
        "instance_name": f"agent-{project}-{name}",
        "server_port": str(DEFAULT_AGENT_PORT),
        "workbench_type": "lerobot",
        "gpu_platform": "cpu-d3",
        "gpu_preset": "8vcpu-32gb",
        "enable_preemptible": "false",
        "nebius_api_key": state.access_key,
        "nebius_secret_key": state.secret_key,
        "s3_bucket": state.bucket,
        "s3_endpoint": state.endpoint,
    }
    tf_dir = provisioner.prepare_working_dir(
        project,
        name,
        bucket=state.bucket,
        region=region,
        endpoint=state.endpoint,
    )
    try:
        provisioner.init(
            tf_dir=tf_dir,
            backend_config={"access_key": state.access_key, "secret_key": state.secret_key},
        )
        provisioner.destroy(tf_dir=tf_dir, tf_vars=tf_vars)
    except ProvisionerError as exc:
        _fail(f"Terraform destroy failed: {exc}")
    _remove_agent_record(project, name)
    typer.echo(f"destroyed: {project}/{name}")


@app.command("verify-live")
def verify_live_cmd(
    project: str = typer.Option(DEFAULT_PROJECT_ALIAS, "--project", help="NPA project alias."),
    name: str = typer.Option(DEFAULT_AGENT_NAME, "--name", help="Agent deployment name."),
) -> None:
    """Exit 0 only when live infra checks and tests pass."""
    record = _agent_record(project, name)
    if not record:
        _fail(f"Agent config not found for {project}/{name}")
    public_ip = str(record.get("public_ip", "")).strip()
    region = str(record.get("region", "")).strip()
    if not public_ip or public_ip in {"localhost", "127.0.0.1"} or public_ip.startswith("127."):
        _fail("agent VM does not have a non-localhost public IP")
    if not _is_routable_public_ip(public_ip):
        _fail("agent VM does not have a non-localhost public IP")
    if region != "us-central1":
        _fail(f"agent region mismatch: expected us-central1, got {region!r}")
    try:
        auth_user, auth_password = _load_auth_secret(str(record.get("auth_secret_path", "")))
    except ValueError as exc:
        _fail(str(exc))

    customer_url = _record_customer_url(record)
    tls_verify = _record_tls_verify(record)
    if customer_url:
        try:
            welcome_resp = httpx.get(
                f"{customer_url.rstrip('/')}/welcome",
                timeout=5.0,
                verify=tls_verify,
            )
            if welcome_resp.status_code != 200:
                _fail(f"public welcome page unhealthy (status={welcome_resp.status_code})")
            healthz_resp = httpx.get(
                f"{customer_url.rstrip('/')}/healthz",
                timeout=5.0,
                verify=tls_verify,
            )
            if healthz_resp.status_code != 200:
                _fail(f"public healthz unhealthy (status={healthz_resp.status_code})")
        except httpx.HTTPError as exc:
            _fail(f"public customer URL unreachable: {exc}")

    ui_ok, ui_code = _health(
        str(record.get("agent_url", "")),
        user=auth_user,
        password=auth_password,
        verify=tls_verify,
    )
    if not ui_ok:
        _fail(f"UI health failed behind basic auth (status={ui_code})")
    sim_viz_url = str(record.get("sim_viz_url", record.get("rerun_url", "")))
    rerun_ok, rerun_code = _health(
        sim_viz_url,
        user=auth_user,
        password=auth_password,
        verify=tls_verify,
    )
    if not rerun_ok:
        _fail(f"embedded rerun iframe endpoint unhealthy (status={rerun_code})")
    try:
        sim_viz_status_resp = httpx.get(
            f"{str(record.get('agent_url', '')).rstrip('/')}/api/sim-viz/status",
            auth=(auth_user, auth_password),
            timeout=5.0,
            verify=tls_verify,
        )
        sim_viz_status_resp.raise_for_status()
        sim_viz_status = sim_viz_status_resp.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"sim viz status endpoint unhealthy: {exc}")
    if not isinstance(sim_viz_status, dict):
        _fail("sim viz status endpoint did not return JSON object")

    sim_assets_base = str(record.get("sim_assets_url", record.get("agent_url", ""))).rstrip("/")
    if not sim_assets_base:
        _fail("sim_assets_url missing from agent config")
    try:
        sim_assets_resp = httpx.get(
            f"{sim_assets_base}/api/sim-assets",
            auth=(auth_user, auth_password),
            timeout=5.0,
            verify=tls_verify,
        )
        sim_assets_resp.raise_for_status()
        sim_assets_payload = sim_assets_resp.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"sim assets endpoint unhealthy: {exc}")
    if not isinstance(sim_assets_payload, dict) or "scene_spec" not in sim_assets_payload or "robot_spec" not in sim_assets_payload:
        _fail("sim assets endpoint missing scene_spec/robot_spec payload")

    try:
        cameras_resp = httpx.get(
            f"{sim_assets_base}/api/sim-assets/cameras",
            auth=(auth_user, auth_password),
            timeout=5.0,
            verify=tls_verify,
        )
        cameras_resp.raise_for_status()
        cameras_payload = cameras_resp.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"cameras endpoint unhealthy: {exc}")
    cameras = cameras_payload.get("cameras", []) if isinstance(cameras_payload, dict) else []
    if not isinstance(cameras, list) or not cameras:
        _fail("cameras endpoint returned no cameras")

    selection_body = {
        "robot_preset": "franka",
        "sim_backend": "isaac",
        "scene_spec_uri": "stock://scene/default",
        "assets_uri": "",
        "robot_spec_uri": "stock://robot/franka",
        "cameras_uri": "stock://cameras/default",
        "props": ["cube"],
    }
    try:
        selection_set = httpx.post(
            f"{sim_assets_base}/api/sim-assets/selection",
            auth=(auth_user, auth_password),
            json=selection_body,
            timeout=5.0,
            verify=tls_verify,
        )
        selection_set.raise_for_status()
        selection_get = httpx.get(
            f"{sim_assets_base}/api/sim-assets/selection",
            auth=(auth_user, auth_password),
            timeout=5.0,
            verify=tls_verify,
        )
        selection_get.raise_for_status()
        selected_payload = selection_get.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"sim asset selection round-trip failed: {exc}")
    if not isinstance(selected_payload, dict):
        _fail("sim asset selection GET did not return JSON object")
    for key in (
        "scene_spec_uri",
        "assets_uri",
        "robot_spec_uri",
        "cameras_uri",
        "robot_preset",
        "sim_backend",
    ):
        if selected_payload.get(key) != selection_body[key]:
            _fail(f"sim asset selection round-trip did not persist {key}")

    try:
        submit_resp = httpx.post(
            f"{str(record.get('agent_url', '')).rstrip('/')}/api/workflows/sim2real/submit",
            auth=(auth_user, auth_password),
            json={},
            timeout=5.0,
            verify=tls_verify,
        )
        submit_resp.raise_for_status()
        submit_payload = submit_resp.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"workflow submit endpoint failed: {exc}")
    if not isinstance(submit_payload, dict) or not submit_payload.get("run_id"):
        _fail("workflow submit endpoint did not return run_id")

    try:
        load_demo_resp = httpx.post(
            f"{str(record.get('agent_url', '')).rstrip('/')}/api/sim-viz/load-franka-demo",
            auth=(auth_user, auth_password),
            json={"camera": "workspace"},
            timeout=30.0,
            verify=tls_verify,
        )
        load_demo_resp.raise_for_status()
        load_demo_payload = load_demo_resp.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"load-franka-demo endpoint failed: {exc}")
    if not isinstance(load_demo_payload, dict) or not load_demo_payload.get("ok"):
        _fail("load-franka-demo endpoint did not return ok=true")
    sim_viz_after_demo = load_demo_payload.get("sim_viz", {})
    if not isinstance(sim_viz_after_demo, dict) or not (
        sim_viz_after_demo.get("rerun_ready") or sim_viz_after_demo.get("rrd_uri")
    ):
        _fail("load-franka-demo did not mark rerun_ready/rrd_uri")

    try:
        preview_resp = httpx.post(
            f"{str(record.get('agent_url', '')).rstrip('/')}/api/sim-viz/camera-preview",
            auth=(auth_user, auth_password),
            json={"camera": "workspace"},
            timeout=15.0,
            verify=tls_verify,
        )
        preview_resp.raise_for_status()
        preview_payload = preview_resp.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"camera-preview endpoint failed: {exc}")
    if not isinstance(preview_payload, dict) or not preview_payload.get("ok"):
        _fail("camera-preview endpoint did not return ok=true")

    agent_base = str(record.get("agent_url", "")).rstrip("/")
    try:
        rrd_resp = httpx.get(
            f"{agent_base}/api/sim-viz/rrd",
            auth=(auth_user, auth_password),
            timeout=15.0,
            verify=tls_verify,
        )
        rrd_resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        _fail(f"sim-viz rrd endpoint failed after load-franka-demo: {exc}")
    rrd_ct = str(rrd_resp.headers.get("content-type", ""))
    if "application/json" in rrd_ct:
        if not isinstance(rrd_resp.json(), dict):
            _fail("sim-viz rrd JSON response was not an object")
    elif len(rrd_resp.content) < 64:
        _fail("sim-viz rrd endpoint returned unexpectedly small payload")
    try:
        rrd_blob_resp = httpx.get(
            f"{agent_base}/api/sim-viz/rrd-blob",
            auth=(auth_user, auth_password),
            timeout=15.0,
            verify=tls_verify,
        )
        rrd_blob_resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        _fail(f"sim-viz rrd-blob endpoint failed after load-franka-demo: {exc}")
    rrd_blob_ct = str(rrd_blob_resp.headers.get("content-type", ""))
    if "application/json" in rrd_blob_ct:
        if not isinstance(rrd_blob_resp.json(), dict):
            _fail("sim-viz rrd-blob JSON response was not an object")
    elif len(rrd_blob_resp.content) < 64:
        _fail("sim-viz rrd-blob endpoint returned unexpectedly small payload")

    rerun_static_ok = False
    for static_path in (
        "/rerun/index.js",
        "/rerun/re_viewer.js",
        "/rerun/favicon.ico",
        "/rerun/version",
    ):
        try:
            static_resp = httpx.get(
                f"{agent_base}{static_path}",
                auth=(auth_user, auth_password),
                timeout=15.0,
                verify=tls_verify,
            )
            if static_resp.status_code == 200 and static_resp.content:
                rerun_static_ok = True
                break
        except httpx.HTTPError:
            continue
    if not rerun_static_ok:
        _fail("rerun static asset probe failed (no /rerun/*.js|ico|version responded 200)")

    try:
        health_resp = httpx.get(
            f"{agent_base}/api/health",
            auth=(auth_user, auth_password),
            timeout=5.0,
            verify=tls_verify,
        )
        health_resp.raise_for_status()
        health_payload = health_resp.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"health endpoint failed: {exc}")
    if not isinstance(health_payload, dict) or not health_payload.get("ok"):
        _fail("health endpoint did not return ok=true")

    try:
        workflow_status_resp = httpx.get(
            f"{agent_base}/api/workflows/sim2real/status",
            auth=(auth_user, auth_password),
            timeout=5.0,
            verify=tls_verify,
        )
        workflow_status_resp.raise_for_status()
        workflow_status_payload = workflow_status_resp.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"workflow status endpoint failed: {exc}")
    if not isinstance(workflow_status_payload, dict):
        _fail("workflow status endpoint did not return JSON object")

    ui_resp = httpx.get(
        str(record.get("agent_url", "")),
        auth=(auth_user, auth_password),
        timeout=10.0,
        verify=tls_verify,
    )
    if ui_resp.status_code != 200:
        _fail(f"UI html fetch failed (status={ui_resp.status_code})")
    ui_html = ui_resp.text
    for marker in (
        'bindClick("chatSend"',
        "function wireUi(",
        "initNpaAgentUi",
        "history.replaceState",
        "location.username",
        f'name="npa-ui-version" content="{AGENT_UI_VERSION}"',
    ):
        if marker not in ui_html:
            _fail(f"UI html missing wiring marker: {marker}")

    try:
        session_resp = httpx.get(
            f"{str(record.get('agent_url', '')).rstrip('/')}/api/session",
            auth=(auth_user, auth_password),
            timeout=5.0,
            verify=tls_verify,
        )
        session_resp.raise_for_status()
        session_payload = session_resp.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"session endpoint failed: {exc}")
    if not isinstance(session_payload, dict):
        _fail("session endpoint did not return JSON object")

    try:
        tools_resp = httpx.get(
            f"{str(record.get('agent_url', '')).rstrip('/')}/api/tools",
            auth=(auth_user, auth_password),
            timeout=5.0,
            verify=tls_verify,
        )
        tools_resp.raise_for_status()
        tool_refs = tools_resp.json().get("tool_refs", [])
    except Exception as exc:  # noqa: BLE001
        _fail(f"agent toolRef catalog request failed: {exc}")
    if len(tool_refs) < 19:
        _fail(f"toolRef catalog too small: expected >=19, got {len(tool_refs)}")
    try:
        resolve_resp = httpx.get(
            f"{str(record.get('agent_url', '')).rstrip('/')}/api/tools/{tool_refs[0]}",
            auth=(auth_user, auth_password),
            timeout=5.0,
            verify=tls_verify,
        )
        resolve_resp.raise_for_status()
        resolved = resolve_resp.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"agent toolRef resolve failed: {exc}")
    if not resolved.get("ok"):
        _fail("agent failed to resolve toolRef catalog entry")
    if not isinstance(resolved.get("argv_template"), list):
        _fail("resolved toolRef entry missing argv_template list")
    try:
        chat_smoke = httpx.post(
            f"{str(record.get('agent_url', '')).rstrip('/')}/api/chat",
            auth=(auth_user, auth_password),
            json={"messages": [{"role": "user", "content": "what is the current sim2real status"}]},
            timeout=30.0,
            verify=tls_verify,
        )
        chat_smoke.raise_for_status()
        chat_payload = chat_smoke.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"chat endpoint smoke failed: {exc}")
    if not isinstance(chat_payload, dict) or not chat_payload.get("ok"):
        _fail("chat endpoint did not return ok=true")
    reply = str(chat_payload.get("reply") or "")
    if "run_id" not in reply and "stage" not in reply:
        _fail("chat status reply missing run_id/stage fields")
    if reply.strip().startswith("GET /api") or reply.strip() == "GET /api/sim-viz/status":
        _fail("chat status reply returned raw GET path instead of unpacked status")
    if not chat_payload.get("grounded"):
        _fail("chat status reply expected grounded=true from intent router")
    apis_used = chat_payload.get("apis_used")
    if not isinstance(apis_used, list) or not apis_used:
        _fail("chat status reply expected non-empty apis_used list")

    try:
        wf_chat = httpx.post(
            f"{str(record.get('agent_url', '')).rstrip('/')}/api/chat",
            auth=(auth_user, auth_password),
            json={"messages": [{"role": "user", "content": "create 2-step sim2real workflow"}]},
            timeout=30.0,
            verify=tls_verify,
        )
        wf_chat.raise_for_status()
        wf_payload = wf_chat.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"create-workflow chat smoke failed: {exc}")
    if not isinstance(wf_payload, dict) or not wf_payload.get("workflow_yaml"):
        _fail("create-workflow chat did not return workflow_yaml")
    wf_yaml = str(wf_payload.get("workflow_yaml") or "")
    if "augment" not in wf_yaml or "envgen" not in wf_yaml:
        _fail("create-workflow chat yaml missing sim2real stages")
    try:
        wf_validate = httpx.post(
            f"{agent_base}/api/workflows/validate",
            auth=(auth_user, auth_password),
            json={"yaml": wf_yaml},
            timeout=15.0,
            verify=tls_verify,
        )
        wf_validate.raise_for_status()
        wf_val_payload = wf_validate.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"workflow validate endpoint failed: {exc}")
    if not isinstance(wf_val_payload, dict) or not wf_val_payload.get("ok"):
        _fail("workflow validate endpoint did not return ok=true")

    try:
        onboard_chat = httpx.post(
            f"{str(record.get('agent_url', '')).rstrip('/')}/api/chat",
            auth=(auth_user, auth_password),
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "add an open source repo, containerize, push to registry, "
                            "and run LeIsaac on live infra"
                        ),
                    }
                ]
            },
            timeout=30.0,
            verify=tls_verify,
        )
        onboard_chat.raise_for_status()
        onboard_payload = onboard_chat.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"onboard_solution chat smoke failed: {exc}")
    if not isinstance(onboard_payload, dict) or not onboard_payload.get("ok"):
        _fail("onboard_solution chat did not return ok=true")
    onboard_reply = str(onboard_payload.get("reply") or "")
    if "run_byof_repo.py" not in onboard_reply:
        _fail("onboard_solution chat reply missing run_byof_repo.py command")
    if "byof-onboard" not in onboard_reply and "skills/workflows/byof-onboard" not in onboard_reply:
        _fail("onboard_solution chat reply missing byof-onboard skill reference")
    if "--base-profile" not in onboard_reply and "--base-image" not in onboard_reply:
        _fail("onboard_solution chat reply missing base image guidance")
    if "<repo-url>" not in onboard_reply:
        _fail("onboard_solution chat reply missing runnable placeholders")
    if onboard_reply.strip().startswith("GET /api"):
        _fail("onboard_solution chat returned raw GET path instead of guidance")
    if not onboard_payload.get("grounded"):
        _fail("onboard_solution chat expected grounded=true")
    onboard_apis = onboard_payload.get("apis_used")
    if not isinstance(onboard_apis, list) or "tools" not in onboard_apis:
        _fail("onboard_solution chat expected tools in apis_used")

    test_env = {
        **dict(os.environ),
        "NPA_INTEGRATION_E2E": "1",
        "NPA_AGENT_LIVE": "1",
        "NPA_AGENT_PROJECT": project,
        "NPA_AGENT_NAME": name,
    }
    if os.environ.get("NPA_AGENT_CHAT_LIVE") == "1":
        test_env["NPA_AGENT_CHAT_LIVE"] = "1"
    smoke = subprocess.run(
        [
            "npa/.venv/bin/python",
            "-m",
            "pytest",
            "npa/tests/smoke/test_agent_smoke.py",
            "npa/tests/smoke/test_agent_chat_smoke.py",
            "-q",
        ],
        check=False,
        env=test_env,
    )
    if smoke.returncode != 0:
        _fail("pytest npa/tests/smoke/test_agent_smoke.py test_agent_chat_smoke.py failed")
    unit = subprocess.run(
        [
            "npa/.venv/bin/python",
            "-m",
            "pytest",
            "npa/tests/cli/test_agent.py",
            "npa/tests/cli/test_agent_workflow.py",
            "-q",
        ],
        check=False,
        env=test_env,
    )
    if unit.returncode != 0:
        _fail("pytest npa/tests/cli/test_agent.py failed")
    e2e = subprocess.run(
        ["npa/.venv/bin/python", "-m", "pytest", "npa/tests/e2e/test_agent_live.py", "-q"],
        check=False,
        env=test_env,
    )
    if e2e.returncode != 0:
        _fail("pytest npa/tests/e2e/test_agent_live.py failed")
    typer.echo("verify-live: ok")
