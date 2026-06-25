---
name: npa-agent
description: Use when operating the NPA agent VM, chat UX, API grounding, bootstrap deployment, or verify-live checks.
---

# NPA Agent VM

The NPA agent is a public HTTPS workbench VM with basic-auth UI, grounded chat,
Sim Assets + Cameras panels, embedded Rerun viewer, and Sim2Real submit hooks.

## When To Use

- Deploy, bootstrap, or verify an agent VM (`npa agent …`)
- Debug chat hallucinations (raw `GET /api/…` replies) or false “Loaded Franka” messages
- Fix Rerun iframe black screen (basic auth + wasm fetch)
- Operate customer HTTPS access and sign-in UX

## Bootstrap And Verify

```bash
npa/.venv/bin/npa agent bootstrap --project rtxpro --name agent
NPA_AGENT_CHAT_LIVE=1 npa/.venv/bin/npa agent verify-live --project rtxpro --name agent
bash npa/scripts/verify_agent_franka.sh
```

Auth secrets live at `~/.npa/agents/<project>/<name>/auth.env` (`AGENT_USER`, `AGENT_PASSWORD`).

## Customer HTTPS Access

- Public URL: `https://<public_ip>/` (self-signed cert on VM IP)
- Sign-in form at `/login-help.html` embeds credentials via URL then **replaceState** strips them
- All `fetch` calls use `credentials: "include"` for session basic auth
- Never suggest `localhost`, `127.0.0.1`, or port `8080` — use same-origin `/api/…` paths

## Chat Maturity Patterns

Intent router in `npa/src/npa/cli/agent_chat.py` (embedded in remote `backend.py` at bootstrap):

| Intent | Example triggers | APIs |
|--------|------------------|------|
| `sim2real_status` | "current status", "workflow status" | sim-viz/status, workflows/sim2real/status |
| `sim_assets` | "sim assets", "selection" | sim-assets, sim-assets/selection |
| `cameras` | "cameras", "workspace camera" | sim-assets/cameras |
| `tools_catalog` | "tools", "toolRef" | tools |
| `configure_s3` | "configure S3", "bucket" | tools (nebius-infra) |
| `cosmos3` | "cosmos3", "setup cosmos" | skill steps (operator machine) |
| `load_franka` | "load franka", "show demo" | sim-viz/load-franka-demo |

Rules:

- **Never** return only `GET /api/…` as the chat reply
- Always unpack **run_id**, **stage**, **rerun_ready**, **camera** in markdown (`**key**: \`value\``)
- Grounded replies set `"grounded": true` and `"apis_used": ["sim-viz/status", …]`
- LLM fallback injects `format_live_context_block(state)` JSON snapshot into the system prompt

## Rerun Iframe Fix

Rerun wasm inside `/rerun/?url=…` cannot send HTTP basic auth. Parent page:

1. `fetch("/api/sim-viz/rrd-blob", { credentials: "include" })` with auth
2. `URL.createObjectURL(blob)` → pass blob URL to Rerun iframe `url=` param

Do not point the iframe directly at `/api/sim-viz/rrd` (black screen).

## HTTP API Reference

All paths are under `/api/` (nginx proxies to FastAPI backend on `:8787`).

### `GET /api/health`

```json
{"ok": true, "tool_refs": 19}
```

### `GET /api/session`

```json
{
  "selection": {"robot_preset": "franka", "sim_backend": "isaac", "scene_spec_uri": "stock://scene/default"},
  "sim_viz": {"run_id": "franka-demo", "stage": "demo", "camera": "workspace", "rerun_ready": true},
  "latest_submit": {},
  "camera_selection": ["workspace"],
  "chat_history": []
}
```

### `POST /api/chat`

Request: `{"messages": [{"role": "user", "content": "what is the current sim2real status"}]}`

Grounded response:

```json
{
  "ok": true,
  "model": "nvidia/Cosmos3-Super-Reasoner",
  "reply": "**Sim2Real status** … **run_id**: `franka-demo` …",
  "grounded": true,
  "apis_used": ["sim-viz/status", "workflows/sim2real/status"]
}
```

### `GET /api/sim-viz/status`

```json
{
  "run_id": "franka-demo",
  "stage": "demo",
  "camera": "workspace",
  "rrd_uri": "file:///opt/npa-agent/sim2real.rrd",
  "rerun_ready": true,
  "rerun_iframe_url": "/rerun/?url=…"
}
```

### `POST /api/sim-viz/load-franka-demo`

Body: `{"camera": "workspace"}` → generates `.rrd`, restarts Rerun service, returns `sim_viz`.

### `GET /api/sim-viz/rrd-blob`

Authenticated octet-stream of `.rrd` bytes (for parent blob URL).

### `GET /api/sim-assets`

Scene/robot specs + current selection and `resolved_uris`.

### `GET /api/sim-assets/cameras`

```json
{"cameras": [{"name": "workspace", "placement": "stock_workspace", "fov": 60.0}], "selected": ["workspace"]}
```

### `POST /api/workflows/sim2real/submit`

Submits workflow with current selection; updates `latest_submit` and `sim_viz.run_id`.

### `GET /api/tools`

```json
{"tool_refs": ["workbench.genesis.train", "..."]}
```

## Source Layout

- CLI + bootstrap: `npa/src/npa/cli/agent.py`
- Chat router (testable): `npa/src/npa/cli/agent_chat.py`
- Franka verify script: `npa/scripts/verify_agent_franka.sh`
- Mature deploy loop: `npa/scripts/agent_mature_verify_loop.sh`
