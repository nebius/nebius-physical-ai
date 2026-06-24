"""Top-level CLI for deploying and operating the NPA agent VM."""

from __future__ import annotations

import base64
import json
import os
import secrets
import shlex
import subprocess
import ipaddress
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
    sim_viz_url: str
    sim_assets_url: str
    cameras_api_url: str
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
            "sim_viz_url": self.sim_viz_url,
            "sim_assets_url": self.sim_assets_url,
            "cameras_api_url": self.cameras_api_url,
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
    setup_script = f"""set -euo pipefail
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nginx apache2-utils python3-venv python3-pip
sudo mkdir -p /opt/npa-agent
cat <<'PY' | sudo tee /opt/npa-agent/backend.py >/dev/null
import json
import os
import re
import secrets
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI(title="npa-agent")
TOOL_CATALOG = {catalog_json}
TOOL_REFS = sorted(TOOL_CATALOG.keys())
STATE_PATH = Path("/opt/npa-agent/session_state.json")
RRD_PATH = Path("/opt/npa-agent/sim2real.rrd")
RERUN_UNIT = "npa-rerun"
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
        "latest_submit": {{}},
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
    return merged

def _save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\\n", encoding="utf-8")

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
    rr.log(
        "robot/franka",
        rr.TextDocument(
            "Franka Panda — stock tabletop pick-and-place demo (NPA agent preview)"
        ),
    )
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
    return target

def _restart_rerun_serve() -> bool:
    try:
        subprocess.run(
            ["sudo", "systemctl", "restart", RERUN_UNIT],
            check=True,
            capture_output=True,
            timeout=30,
        )
        return True
    except Exception:
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
        "mode": "camera_preview",
        "camera": cam,
        "preview_camera": cam,
        "preview_entity": f"world/cameras/{{cam}}",
        "rerun_ready": restarted or target.is_file(),
        "rerun_iframe_url": f"/rerun/?url=/api/sim-viz/rrd&camera={{cam}}",
    }}
    state["sim_viz"] = viz
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
        "- POST /api/sim-viz/load-franka-demo — load stock Franka tabletop demo into Rerun",
        "- POST /api/workflows/sim2real/submit — submit Sim2Real with current asset selection",
        "- GET /api/tools — workbench toolRef catalog",
        "",
        "To view Franka immediately, tell users to click **Load Franka in Rerun** in the Sim Assets panel",
        "(or POST /api/sim-viz/load-franka-demo). Open the embedded viewer at /rerun/.",
        "The **Cameras** panel is the center column below chat: stock workspace and wrist cameras",
        "with 2D frustum schematics, selection, and **Preview in Rerun**.",
        "Never suggest localhost, 127.0.0.1, or port 8080 — use relative /api/... paths or /rerun/.",
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

@app.post("/chat")
def chat(payload: dict):
    raw_messages = payload.get("messages", [])
    if not isinstance(raw_messages, list) or not raw_messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty list")
    model = str(payload.get("model") or LLM_MODEL).strip() or LLM_MODEL
    messages: list[dict] = [{{"role": "system", "content": _agent_system_prompt()}}]
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
    return {{
        "ok": True,
        "model": model,
        "reply": reply,
        "reasoning": reasoning,
    }}

@app.get("/health")
def health():
    return {{"ok": True, "tool_refs": len(TOOL_REFS)}}

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
def sim_viz_status():
    state = _load_state()
    sim_viz = state.get("sim_viz", {{}})
    payload = dict(DEFAULT_SIM_VIZ)
    if isinstance(sim_viz, dict):
        payload.update(sim_viz)
    selected = state.get("camera_selection", ["workspace"])
    camera = str(payload.get("camera") or (selected[0] if isinstance(selected, list) and selected else "workspace"))
    payload["camera"] = camera
    payload["rerun_iframe_url"] = f"/rerun/?url=/api/sim-viz/rrd&camera={{camera}}"
    if not payload.get("rrd_uri") and RRD_PATH.is_file():
        payload["rrd_uri"] = f"file://{{RRD_PATH}}"
    payload["rerun_ready"] = bool(payload.get("rrd_uri")) or RRD_PATH.is_file()
    return payload

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

@app.get("/sim-viz/rrd")
def sim_viz_rrd():
    state = _load_state()
    sim_viz = state.get("sim_viz", {{}})
    if not isinstance(sim_viz, dict) or not sim_viz.get("rrd_uri"):
        raise HTTPException(status_code=404, detail="No sim2real.rrd available yet")
    uri = str(sim_viz.get("rrd_uri"))
    if uri.startswith("file://"):
        file_path = Path(uri[len("file://"):])
        if file_path.is_file():
            return FileResponse(str(file_path), media_type="application/octet-stream")
    return JSONResponse({{"ok": True, "rrd_uri": uri}}, status_code=200)

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
    cameras = list(DEFAULT_SCENE_SPEC["cameras"].values())
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
    _save_state(state)
    return {{"ok": True, "run_id": run_id, "selection": selection, "env": env_block}}
PY
cat <<'PY' | sudo tee /opt/npa-agent/bootstrap_rrd.py >/dev/null
from pathlib import Path

import rerun as rr

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
rr.log(
    "robot/franka",
    rr.TextDocument("Franka Panda — stock tabletop demo (bootstrap)"),
)
rr.log("cameras/workspace", rr.Pinhole(fov_y=60.0))
rr.log("cameras/wrist", rr.Pinhole(fov_y=90.0))
rr.save(str(target))
PY
cat <<'HTML' | sudo tee /opt/npa-agent/ui.html >/dev/null
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>NPA Agent</title>
    <style>
      body {{ font-family: sans-serif; margin: 16px; max-width: 1600px; }}
      .layout {{ display: grid; gap: 12px; }}
      .layout-3 {{ grid-template-columns: 1fr 1.15fr 1.25fr; }}
      .panel {{ border: 1px solid #ddd; border-radius: 6px; padding: 10px; }}
      .cameras-panel {{ border-color: #93c5fd; background: #f8fafc; }}
      .camera-card {{
        border: 1px solid #dbeafe; border-radius: 6px; padding: 8px; margin-bottom: 8px;
        background: #fff;
      }}
      .camera-card.selected {{ border: 2px solid #3b82f6; box-shadow: 0 0 0 1px #bfdbfe; }}
      .camera-card h4 {{ margin: 0 0 6px 0; }}
      .camera-meta {{ font-size: 12px; color: #475569; margin-bottom: 6px; }}
      .camera-actions {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }}
      .camera-frustum {{ display: flex; justify-content: center; }}
      .rollout-hint {{ font-size: 13px; color: #334155; margin: 0 0 10px 0; }}
      .chat-panel {{ margin-bottom: 12px; }}
      .chat-log {{
        height: 280px; overflow-y: auto; background: #fafafa; border: 1px solid #eee;
        border-radius: 4px; padding: 8px; margin-bottom: 8px;
      }}
      .chat-msg {{ margin: 6px 0; }}
      .chat-msg.user {{ color: #111; }}
      .chat-msg.assistant {{ color: #1e40af; }}
      .chat-msg.error {{ color: #b91c1c; }}
      .chat-input {{ display: flex; gap: 8px; }}
      .chat-input textarea {{
        flex: 1; min-height: 56px; resize: vertical; font-family: inherit; padding: 8px;
      }}
      iframe {{ width: 100%; height: 360px; border: 1px solid #ddd; }}
      pre {{ white-space: pre-wrap; word-break: break-word; background: #fafafa; padding: 8px; }}
      .status-row {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 8px; font-size: 14px; }}
      .btn-row {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 8px; }}
      .cta {{ color: #92400e; background: #fffbeb; border: 1px solid #fcd34d; border-radius: 4px; padding: 8px; }}
      .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; background: #e0e7ff; color: #1e3a8a; font-size: 12px; }}
    </style>
  </head>
  <body>
    <h2>NPA Agent</h2>
    <section class="panel chat-panel">
      <h3>Workbench chat</h3>
      <p>Ask about configure, provision, Cosmos3, S3, workflows, sim assets, and Rerun viz.</p>
      <div id="chatLog" class="chat-log"></div>
      <div class="chat-input">
        <textarea id="chatInput" placeholder="How do I configure S3 for Sim2Real?"></textarea>
        <button id="chatSend" type="button">Send</button>
      </div>
      <div style="margin-top:8px; display:flex; gap:8px; flex-wrap:wrap;">
        <button id="chatActionS3" type="button">Configure S3</button>
        <button id="chatActionCosmos" type="button">Setup Cosmos3</button>
        <button id="chatActionWatch" type="button">Watch sim</button>
      </div>
    </section>
    <div class="layout layout-3">
      <section class="panel">
        <h3>Sim Assets</h3>
        <label for="robotPreset">Robot preset</label>
        <select id="robotPreset">
          <option value="franka" selected>Franka (stock tabletop)</option>
          <option value="ur5e">UR5e</option>
        </select>
        <div id="assets"></div>
        <div class="btn-row">
          <button id="applySelection" type="button">Apply stock selection</button>
          <button id="loadFrankaRerun" type="button">Load Franka in Rerun</button>
          <button id="submitWorkflow" type="button">Submit Sim2Real</button>
          <button id="workflowStatus" type="button">Workflow status</button>
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
            <button id="openRerun" type="button">Open in Rerun</button>
          </div>
          <p id="simvizCta" class="cta" hidden>No .rrd yet — click <strong>Load Franka in Rerun</strong> or submit Sim2Real.</p>
        </div>
        <iframe id="rerunFrame" title="rerun" src="/rerun/?url=%2Fapi%2Fsim-viz%2Frrd"></iframe>
      </section>
    </div>
    <script>
      const chatHistory = [];
      function appendChat(role, text) {{
        const log = document.getElementById("chatLog");
        const div = document.createElement("div");
        div.className = "chat-msg " + role;
        div.textContent = (role === "user" ? "You: " : role === "error" ? "Error: " : "Agent: ") + text;
        log.appendChild(div);
        log.scrollTop = log.scrollHeight;
      }}
      async function sendChat() {{
        const input = document.getElementById("chatInput");
        const text = String(input.value || "").trim();
        if (!text) return;
        input.value = "";
        appendChat("user", text);
        chatHistory.push({{ role: "user", content: text }});
        const btn = document.getElementById("chatSend");
        btn.disabled = true;
        try {{
          const resp = await fetch("/api/chat", {{
            method: "POST",
            headers: {{ "content-type": "application/json" }},
            credentials: "include",
            body: JSON.stringify({{ messages: chatHistory }}),
          }});
          const data = await resp.json();
          if (!resp.ok) {{
            appendChat("error", data.detail || resp.statusText || "chat failed");
            return;
          }}
          const reply = String(data.reply || "").trim();
          if (reply) {{
            appendChat("assistant", reply);
            chatHistory.push({{ role: "assistant", content: reply }});
          }} else {{
            appendChat("error", "empty reply from model");
          }}
        }} catch (err) {{
          appendChat("error", String(err));
        }} finally {{
          btn.disabled = false;
          input.focus();
        }}
      }}
      document.getElementById("chatSend").addEventListener("click", sendChat);
      document.getElementById("chatInput").addEventListener("keydown", (e) => {{
        if (e.key === "Enter" && !e.shiftKey) {{
          e.preventDefault();
          sendChat();
        }}
      }});
      function setChatInput(text) {{
        const input = document.getElementById("chatInput");
        input.value = text;
        input.focus();
      }}
      document.getElementById("chatActionS3").addEventListener("click", () => setChatInput("Help me configure S3 credentials and bucket for NPA workflows."));
      document.getElementById("chatActionCosmos").addEventListener("click", () => setChatInput("How do I set up Cosmos3 in the NPA workbench?"));
      document.getElementById("chatActionWatch").addEventListener("click", () => setChatInput("Watch the sim in Rerun — use Load Franka in Rerun or check /api/sim-viz/status."));
      let lastRrdUpdatedAt = "";
      function rerunIframeSrc(camera) {{
        const cam = String(camera || "workspace");
        const source = "/api/sim-viz/rrd";
        return (
          "/rerun/?url=" +
          encodeURIComponent(source) +
          "&camera=" +
          encodeURIComponent(cam) +
          "&t=" +
          Date.now()
        );
      }}
      function reloadRerunIframe(camera) {{
        const iframe = document.getElementById("rerunFrame");
        iframe.src = rerunIframeSrc(camera);
      }}
      async function loadFrankaDemo() {{
        const camera = String(document.getElementById("cameraSelect").value || "workspace");
        const resp = await fetch("/api/sim-viz/load-franka-demo", {{
          method: "POST",
          headers: {{ "content-type": "application/json" }},
          credentials: "include",
          body: JSON.stringify({{ camera }}),
        }});
        const data = await resp.json();
        if (!resp.ok) {{
          appendChat("error", data.detail || "failed to load Franka demo");
          return false;
        }}
        reloadRerunIframe(camera);
        appendChat("assistant", "Loaded stock Franka tabletop demo in Rerun (" + camera + " camera).");
        await refresh();
        return true;
      }}
      async function loadJson(path) {{
        const resp = await fetch(path, {{ credentials: "include" }});
        return await resp.json();
      }}
      async function refresh() {{
        try {{
          const assets = await loadJson("/api/sim-assets");
          const cameras = await loadJson("/api/sim-assets/cameras");
          const simViz = await loadJson("/api/sim-viz/status");
          document.getElementById("assets").innerHTML = "<pre>" + JSON.stringify(assets.selection, null, 2) + "</pre>";
          document.getElementById("simRunId").textContent = String(simViz.run_id || "—");
          document.getElementById("simStage").textContent = String(simViz.stage || "idle");
          document.getElementById("simCamera").textContent = String(simViz.camera || "workspace");
          const cta = document.getElementById("simvizCta");
          const ready = Boolean(simViz.rerun_ready || simViz.rrd_uri);
          cta.hidden = ready;
          const robotPreset = document.getElementById("robotPreset");
          if (assets.selection && assets.selection.robot_preset) {{
            robotPreset.value = String(assets.selection.robot_preset);
          }}
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
          if (updatedAt && updatedAt !== lastRrdUpdatedAt) {{
            lastRrdUpdatedAt = updatedAt;
            reloadRerunIframe(simViz.camera || activeName);
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
        const stroke = selected ? "#2563eb" : "#94a3b8";
        const fill = selected ? "rgba(37,99,235,0.18)" : "rgba(148,163,184,0.15)";
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
          const pos = Array.isArray(cam.pos) ? cam.pos.map((v) => Number(v).toFixed(2)).join(", ") : "—";
          const look = Array.isArray(cam.look_at) ? cam.look_at.map((v) => Number(v).toFixed(2)).join(", ") : "—";
          const res = Array.isArray(cam.resolution) ? cam.resolution.join("×") : "640×480";
          card.innerHTML = `
            <h4>${{name}}${{selected ? " <span class=\\"badge\\">selected</span>" : ""}}</h4>
            <div class="camera-meta">placement: ${{String(cam.placement || "custom")}} · fov ${{Number(cam.fov || 60)}}° · ${{res}}</div>
            <div class="camera-meta">pos [${{pos}}] · look_at [${{look}}]</div>
            <div class="camera-frustum">${{frustumSvg(cam, selected)}}</div>
            <div class="camera-actions">
              <button type="button" data-action="select" data-camera="${{name}}">Select</button>
              <button type="button" data-action="preview" data-camera="${{name}}">Preview in Rerun</button>
            </div>`;
          holder.appendChild(card);
        }}
        const entity = String(simViz.preview_entity || ("world/cameras/" + activeName));
        const rollout = "rollouts/latest/" + activeName + "/camera";
        document.getElementById("rerunEntityHint").textContent =
          (simViz.rerun_ready || simViz.rrd_uri)
            ? "Rerun entities: " + entity + " (frustum) · " + rollout + " (rollout frames when available)"
            : "Preview in Rerun to log camera frustums; rollout frames appear after Sim2Real runs.";
        holder.querySelectorAll("button[data-action]").forEach((btn) => {{
          btn.addEventListener("click", async () => {{
            const camera = String(btn.getAttribute("data-camera") || "");
            if (btn.getAttribute("data-action") === "select") {{
              await selectCamera(camera);
            }} else {{
              await previewCamera(camera);
            }}
          }});
        }});
      }}
      async function selectCamera(camera) {{
        const selected = String(camera || "");
        await fetch("/api/sim-assets/cameras/selection", {{
          method: "PUT",
          headers: {{ "content-type": "application/json" }},
          credentials: "include",
          body: JSON.stringify({{ selected: selected ? [selected] : [] }}),
        }});
        await fetch("/api/sim-assets/selection", {{
          method: "POST",
          headers: {{ "content-type": "application/json" }},
          credentials: "include",
          body: JSON.stringify({{
            scene_spec_uri: "stock://scene/default",
            robot_spec_uri: "stock://robot/franka",
            cameras_uri: "stock://cameras/default",
            robot_preset: String(document.getElementById("robotPreset").value || "franka"),
            sim_backend: "isaac",
            props: ["cube"],
          }}),
        }});
        await refresh();
      }}
      async function previewCamera(camera) {{
        const resp = await fetch("/api/sim-viz/camera-preview", {{
          method: "POST",
          headers: {{ "content-type": "application/json" }},
          credentials: "include",
          body: JSON.stringify({{ camera }}),
        }});
        const data = await resp.json();
        if (!resp.ok) {{
          appendChat("error", data.detail || "camera preview failed");
          return;
        }}
        reloadRerunIframe(camera);
        const entity = String(data.entity_path || ("world/cameras/" + camera));
        appendChat("assistant", "Previewing " + camera + " in Rerun at " + entity + ".");
        await refresh();
      }}
      document.getElementById("cameraSelect").addEventListener("change", async (e) => {{
        await selectCamera(String(e.target.value || ""));
      }});
      document.getElementById("robotPreset").addEventListener("change", async (e) => {{
        const preset = String(e.target.value || "franka");
        const resp = await fetch("/api/sim-assets/selection", {{
          method: "POST",
          headers: {{ "content-type": "application/json" }},
          credentials: "include",
          body: JSON.stringify({{
            scene_spec_uri: "stock://scene/default",
            robot_spec_uri: preset === "franka" ? "stock://robot/franka" : "",
            cameras_uri: "stock://cameras/default",
            robot_preset: preset,
            sim_backend: "isaac",
            props: ["cube"]
          }}),
        }});
        const data = await resp.json();
        if (preset === "franka" && data.sim_viz) {{
          reloadRerunIframe(data.sim_viz.camera || "workspace");
        }}
        await refresh();
      }});
      document.getElementById("loadFrankaRerun").addEventListener("click", loadFrankaDemo);
      document.getElementById("openRerun").addEventListener("click", async () => {{
        const simViz = await loadJson("/api/sim-viz/status");
        const camera = String(simViz.camera || document.getElementById("cameraSelect").value || "workspace");
        window.open(rerunIframeSrc(camera), "_blank", "noopener");
      }});
      document.getElementById("applySelection").addEventListener("click", async () => {{
        const resp = await fetch("/api/sim-assets/selection", {{
          method: "POST",
          headers: {{ "content-type": "application/json" }},
          credentials: "include",
          body: JSON.stringify({{
            scene_spec_uri: "stock://scene/default",
            robot_spec_uri: "stock://robot/franka",
            cameras_uri: "stock://cameras/default",
            robot_preset: "franka",
            sim_backend: "isaac",
            props: ["cube"]
          }}),
        }});
        const data = await resp.json();
        if (data.sim_viz) {{
          reloadRerunIframe(data.sim_viz.camera || "workspace");
        }}
        await refresh();
      }});
      document.getElementById("submitWorkflow").addEventListener("click", async () => {{
        const resp = await fetch("/api/workflows/sim2real/submit", {{
          method: "POST",
          headers: {{ "content-type": "application/json" }},
          credentials: "include",
          body: JSON.stringify({{}}),
        }});
        const data = await resp.json();
        if (!resp.ok) {{
          appendChat("error", data.detail || "failed to submit sim2real");
          return;
        }}
        appendChat("assistant", `Submitted sim2real run: ${{data.run_id || "unknown"}}`);
        await refresh();
      }});
      document.getElementById("workflowStatus").addEventListener("click", async () => {{
        try {{
          const status = await loadJson("/api/workflows/sim2real/status");
          appendChat("assistant", "Latest workflow status: " + JSON.stringify(status));
        }} catch (err) {{
          appendChat("error", String(err));
        }}
      }});
      refresh().then(async () => {{
        const simViz = await loadJson("/api/sim-viz/status");
        if (!simViz.rerun_ready && !simViz.rrd_uri) {{
          await loadFrankaDemo();
        }}
      }});
      setInterval(refresh, 10000);
    </script>
  </body>
</html>
HTML
sudo python3 -m venv /opt/npa-agent/venv
sudo /opt/npa-agent/venv/bin/pip install --upgrade pip
sudo /opt/npa-agent/venv/bin/pip install fastapi uvicorn httpx "rerun-sdk>=0.32"
sudo /opt/npa-agent/venv/bin/python /opt/npa-agent/bootstrap_rrd.py
sudo systemctl restart npa-rerun || true
cat <<'UNIT' | sudo tee /etc/systemd/system/npa-agent-backend.service >/dev/null
[Unit]
Description=NPA agent backend
After=network.target
[Service]
Type=simple
EnvironmentFile=-/opt/npa-agent/llm.env
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
  location ~* ^/rerun/.+\\.(wasm|js|ico|svg)$ {{
    rewrite ^/rerun/(.*)$ /$1 break;
    proxy_pass http://127.0.0.1:{rerun_port};
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    gzip on;
    gzip_types application/wasm application/javascript text/javascript image/svg+xml;
    gzip_min_length 256;
    add_header Cache-Control "public, max-age=604800, immutable" always;
  }}
  location /rerun/ {{
    proxy_pass http://127.0.0.1:{rerun_port}/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    add_header Cache-Control "no-cache" always;
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
sudo systemctl restart npa-agent-backend nginx
"""
    ssh.run_or_raise(setup_script)
    _write_agent_llm_env(ssh, tf_api_key=tf_api_key, llm_model=llm_model)
    if tf_api_key.strip():
        ssh.run_or_raise("sudo systemctl restart npa-agent-backend")


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
        )
    except (ConfigError, SSHError, ValueError) as exc:
        _fail(f"VM bootstrap failed: {exc}")

    try:
        ensure_ingress(vm_id=instance_id, ports=(agent_port, rerun_port), tool="agent")
    except NetworkIngressError as exc:
        _fail(f"npa network ensure-ingress failed: {exc}")

    agent_url = f"http://{public_ip}:{agent_port}/"
    rerun_url = f"http://{public_ip}:{agent_port}/rerun/"
    sim_viz_url = f"http://{public_ip}:{agent_port}/rerun/"
    sim_assets_url = f"http://{public_ip}:{agent_port}/"
    cameras_api_url = f"http://{public_ip}:{agent_port}/api/sim-assets/cameras"
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
        sim_viz_url=sim_viz_url,
        sim_assets_url=sim_assets_url,
        cameras_api_url=cameras_api_url,
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
    typer.echo(f"sim_viz_url: {sim_viz_url}")
    typer.echo(f"sim_assets_url: {sim_assets_url}")
    typer.echo(f"cameras_api_url: {cameras_api_url}")
    typer.echo(f"llm: {DEFAULT_LLM_PROVIDER}:{DEFAULT_LLM_MODEL}")
    typer.echo(f"auth_user: {DEFAULT_AGENT_USER}")
    typer.echo(f"auth_secret_path: {auth_path}")
    typer.echo(f"auth_password: {redact_value(auth_password)}")


@app.command("bootstrap")
def bootstrap_cmd(
    project: str = typer.Option(DEFAULT_PROJECT_ALIAS, "--project", help="NPA project alias."),
    name: str = typer.Option(DEFAULT_AGENT_NAME, "--name", help="Agent deployment name."),
    ssh_user: str = typer.Option("ubuntu", "--ssh-user", help="SSH username."),
    agent_port: int = typer.Option(DEFAULT_AGENT_PORT, "--agent-port", help="Public agent UI port."),
    backend_port: int = typer.Option(DEFAULT_BACKEND_PORT, "--backend-port", help="Internal agent backend port."),
    rerun_port: int = typer.Option(DEFAULT_RERUN_PORT, "--rerun-port", help="Rerun service port."),
) -> None:
    """Re-bootstrap agent UI/backend/nginx on an existing VM (refresh without Terraform)."""
    record = _agent_record(project, name)
    if not record:
        _fail(f"Agent config not found for {project}/{name}")
    public_ip = str(record.get("public_ip", "")).strip()
    if not _is_routable_public_ip(public_ip):
        _fail("agent VM does not have a routable public IP")
    ssh_key_path = resolve_ssh_config(
        ssh_host=public_ip,
        ssh_user=ssh_user,
        ssh_key=None,
        project=None,
        name=None,
    ).ssh.key_path or str(Path.home() / ".ssh" / "id_ed25519")
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
        )
    except (ConfigError, SSHError, ValueError) as exc:
        _fail(f"VM bootstrap failed: {exc}")
    typer.echo(f"bootstrapped: {project}/{name} at http://{public_ip}:{agent_port}/")


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
        record.get("cameras_api_url", f"{agent_url.rstrip('/')}/api/sim-assets/cameras")
    )
    ui_ok, ui_code = _health(agent_url, user=auth_user, password=auth_password)
    rerun_ok, rerun_code = _health(sim_viz_url, user=auth_user, password=auth_password)
    payload = {
        "project": project,
        "name": name,
        "public_ip": record.get("public_ip", ""),
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

    ui_ok, ui_code = _health(str(record.get("agent_url", "")), user=auth_user, password=auth_password)
    if not ui_ok:
        _fail(f"UI health failed behind basic auth (status={ui_code})")
    sim_viz_url = str(record.get("sim_viz_url", record.get("rerun_url", "")))
    rerun_ok, rerun_code = _health(
        sim_viz_url,
        user=auth_user,
        password=auth_password,
    )
    if not rerun_ok:
        _fail(f"embedded rerun iframe endpoint unhealthy (status={rerun_code})")
    try:
        sim_viz_status_resp = httpx.get(
            f"{str(record.get('agent_url', '')).rstrip('/')}/api/sim-viz/status",
            auth=(auth_user, auth_password),
            timeout=5.0,
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
        )
        selection_set.raise_for_status()
        selection_get = httpx.get(
            f"{sim_assets_base}/api/sim-assets/selection",
            auth=(auth_user, auth_password),
            timeout=5.0,
        )
        selection_get.raise_for_status()
        selected_payload = selection_get.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"sim asset selection round-trip failed: {exc}")
    if not isinstance(selected_payload, dict):
        _fail("sim asset selection GET did not return JSON object")
    if selected_payload.get("scene_spec_uri") != selection_body["scene_spec_uri"]:
        _fail("sim asset selection round-trip did not persist scene_spec_uri")

    try:
        submit_resp = httpx.post(
            f"{str(record.get('agent_url', '')).rstrip('/')}/api/workflows/sim2real/submit",
            auth=(auth_user, auth_password),
            json={},
            timeout=5.0,
        )
        submit_resp.raise_for_status()
        submit_payload = submit_resp.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"workflow submit endpoint failed: {exc}")
    if not isinstance(submit_payload, dict) or not submit_payload.get("run_id"):
        _fail("workflow submit endpoint did not return run_id")

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
    try:
        resolve_resp = httpx.get(
            f"{str(record.get('agent_url', '')).rstrip('/')}/api/tools/{tool_refs[0]}",
            auth=(auth_user, auth_password),
            timeout=5.0,
        )
        resolve_resp.raise_for_status()
        resolved = resolve_resp.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"agent toolRef resolve failed: {exc}")
    if not resolved.get("ok"):
        _fail("agent failed to resolve toolRef catalog entry")
    if not isinstance(resolved.get("argv_template"), list):
        _fail("resolved toolRef entry missing argv_template list")
    if os.environ.get("NPA_AGENT_CHAT_LIVE") == "1":
        try:
            chat_smoke = httpx.post(
                f"{str(record.get('agent_url', '')).rstrip('/')}/api/chat",
                auth=(auth_user, auth_password),
                json={"messages": [{"role": "user", "content": "status"}]},
                timeout=30.0,
            )
            chat_smoke.raise_for_status()
            chat_payload = chat_smoke.json()
        except Exception as exc:  # noqa: BLE001
            _fail(f"chat endpoint smoke failed: {exc}")
        if not isinstance(chat_payload, dict) or not chat_payload.get("ok"):
            _fail("chat endpoint did not return ok=true")

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
