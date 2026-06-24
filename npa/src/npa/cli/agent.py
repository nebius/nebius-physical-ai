"""Top-level CLI for deploying and operating the NPA agent VM."""

from __future__ import annotations

import json
import os
import secrets
import shlex
import subprocess
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
    auth_user: str
    auth_secret_path: str
    llm_provider: str
    llm_model: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "tenant_id": self.tenant_id,
            "region": self.region,
            "public_ip": self.public_ip,
            "instance_id": self.instance_id,
            "agent_url": self.agent_url,
            "rerun_url": self.rerun_url,
            "auth_user": self.auth_user,
            "auth_secret_path": self.auth_secret_path,
            "llm": {"provider": self.llm_provider, "model": self.llm_model},
        }


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
    catalog_json = json.dumps(_tool_catalog_keys())
    setup_script = f"""set -euo pipefail
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nginx apache2-utils python3-venv python3-pip
sudo mkdir -p /opt/npa-agent
cat <<'PY' | sudo tee /opt/npa-agent/backend.py >/dev/null
from fastapi import FastAPI

app = FastAPI(title="npa-agent")
TOOL_REFS = {catalog_json}

@app.get("/health")
def health():
    return {{"ok": True, "tool_refs": len(TOOL_REFS)}}

@app.get("/tools")
def tools():
    return {{"tool_refs": TOOL_REFS}}
PY
cat <<'PY' | sudo tee /opt/npa-agent/rerun_stub.py >/dev/null
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="npa-agent-rerun")

@app.get("/healthz")
def health():
    return {{"ok": True}}

@app.get("/", response_class=HTMLResponse)
def index():
    return "<html><body><h3>Rerun panel ready</h3></body></html>"
PY
cat <<'HTML' | sudo tee /opt/npa-agent/ui.html >/dev/null
<!doctype html>
<html>
  <head><meta charset="utf-8"><title>NPA Agent</title></head>
  <body>
    <h2>NPA Agent</h2>
    <p>Agent UI is live.</p>
    <iframe title="rerun" src="/rerun/" style="width:100%;height:75vh;border:1px solid #ddd;"></iframe>
  </body>
</html>
HTML
sudo python3 -m venv /opt/npa-agent/venv
sudo /opt/npa-agent/venv/bin/pip install --upgrade pip
sudo /opt/npa-agent/venv/bin/pip install fastapi uvicorn
cat <<'UNIT' | sudo tee /etc/systemd/system/npa-agent-backend.service >/dev/null
[Unit]
Description=NPA agent backend
After=network.target
[Service]
Type=simple
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
ExecStart=/opt/npa-agent/venv/bin/uvicorn rerun_stub:app --host 0.0.0.0 --port {rerun_port}
WorkingDirectory=/opt/npa-agent
Restart=always
[Install]
WantedBy=multi-user.target
UNIT
sudo htpasswd -bc /etc/nginx/.npa-agent-htpasswd {shlex.quote(auth_user)} {shlex.quote(auth_password)}
cat <<'NGINX' | sudo tee /etc/nginx/sites-available/npa-agent >/dev/null
server {{
  listen {agent_port};
  server_name _;
  auth_basic "NPA Agent";
  auth_basic_user_file /etc/nginx/.npa-agent-htpasswd;
  location = /healthz {{
    auth_basic off;
    return 200 "ok";
  }}
  location /api/ {{
    proxy_pass http://127.0.0.1:{backend_port}/;
  }}
  location /rerun/ {{
    proxy_pass http://127.0.0.1:{rerun_port}/;
  }}
  location / {{
    root /opt/npa-agent;
    index ui.html;
    try_files /ui.html =404;
  }}
}}
NGINX
sudo ln -sf /etc/nginx/sites-available/npa-agent /etc/nginx/sites-enabled/npa-agent
sudo rm -f /etc/nginx/sites-enabled/default
sudo systemctl daemon-reload
sudo systemctl enable --now npa-agent-backend npa-rerun nginx
"""
    ssh.run_or_raise(setup_script)


def _health(url: str, *, user: str, password: str, timeout: float = 5.0) -> tuple[bool, int]:
    try:
        response = httpx.get(url, auth=(user, password), timeout=timeout)
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

    from npa.clients.nebius import NebiusError, bootstrap_environment, get_iam_token

    try:
        creds = bootstrap_environment(
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
    if not public_ip or public_ip.startswith("127.") or public_ip == "localhost":
        _fail("Terraform output did not include a routable public IP")

    auth_password = secrets.token_urlsafe(18)
    auth_path = _write_auth_secret(
        project_alias=project,
        name=name,
        user=DEFAULT_AGENT_USER,
        password=auth_password,
    )
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
        )
    except (ConfigError, SSHError, ValueError) as exc:
        _fail(f"VM bootstrap failed: {exc}")

    try:
        ensure_ingress(vm_id=instance_id, ports=(agent_port, rerun_port), tool="agent")
    except NetworkIngressError as exc:
        _fail(f"npa network ensure-ingress failed: {exc}")

    agent_url = f"http://{public_ip}:{agent_port}/"
    rerun_url = f"http://{public_ip}:{agent_port}/rerun/"
    record = AgentConfig(
        project_alias=project,
        name=name,
        project_id=env_project_id,
        tenant_id=env_tenant_id,
        region=env_region,
        public_ip=public_ip,
        instance_id=instance_id,
        agent_url=agent_url,
        rerun_url=rerun_url,
        auth_user=DEFAULT_AGENT_USER,
        auth_secret_path=str(auth_path),
        llm_provider=DEFAULT_LLM_PROVIDER,
        llm_model=DEFAULT_LLM_MODEL,
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

    typer.echo(f"public_url: {agent_url}")
    typer.echo(f"rerun_url: {rerun_url}")
    typer.echo(f"llm: {DEFAULT_LLM_PROVIDER}:{DEFAULT_LLM_MODEL}")
    typer.echo(f"auth_user: {DEFAULT_AGENT_USER}")
    typer.echo(f"auth_secret_path: {auth_path}")
    typer.echo(f"auth_password: {redact_value(auth_password)}")


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
    ui_ok, ui_code = _health(agent_url, user=auth_user, password=auth_password)
    rerun_ok, rerun_code = _health(rerun_url, user=auth_user, password=auth_password)
    payload = {
        "project": project,
        "name": name,
        "public_ip": record.get("public_ip", ""),
        "ui_url": agent_url,
        "rerun_url": rerun_url,
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
    if region != "us-central1":
        _fail(f"agent region mismatch: expected us-central1, got {region!r}")
    try:
        auth_user, auth_password = _load_auth_secret(str(record.get("auth_secret_path", "")))
    except ValueError as exc:
        _fail(str(exc))

    ui_ok, ui_code = _health(str(record.get("agent_url", "")), user=auth_user, password=auth_password)
    if not ui_ok:
        _fail(f"UI health failed behind basic auth (status={ui_code})")
    rerun_ok, rerun_code = _health(
        str(record.get("rerun_url", "")),
        user=auth_user,
        password=auth_password,
    )
    if not rerun_ok:
        _fail(f"embedded rerun iframe endpoint unhealthy (status={rerun_code})")

    try:
        tools_resp = httpx.get(
            f"{str(record.get('agent_url', '')).rstrip('/')}/api/tools",
            auth=(auth_user, auth_password),
            timeout=5.0,
        )
        tools_resp.raise_for_status()
        tool_refs = tools_resp.json().get("tool_refs", [])
    except Exception as exc:  # noqa: BLE001
        _fail(f"agent toolRef catalog request failed: {exc}")
    if len(tool_refs) < 19:
        _fail(f"toolRef catalog too small: expected >=19, got {len(tool_refs)}")

    test_env = {
        **dict(os.environ),
        "NPA_INTEGRATION_E2E": "1",
        "NPA_AGENT_LIVE": "1",
    }
    unit = subprocess.run(
        ["npa/.venv/bin/python", "-m", "pytest", "npa/tests/cli/test_agent.py", "-q"],
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
