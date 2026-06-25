"""Grounded chat intent routing for the NPA agent VM backend."""

from __future__ import annotations

import json
import re
from typing import Any

STATUS_QUERY_RE = re.compile(
    r"(?:\b(?:what(?:'s| is)|show|tell me|check|get)\b.*\b(?:current\s+)?"
    r"(?:sim\s*[- ]?2\s*[- ]?real|sim2real|workflow|rerun|sim\s+viz))"
    r"|\b(?:sim\s*[- ]?2\s*[- ]?real|workflow|rerun)\b.*\bstatus\b"
    r"|\bstatus\b.*\b(?:sim\s*[- ]?2\s*[- ]?real|sim2real|workflow|rerun|stage|run)\b"
    r"|\b(?:watch|monitor|follow|observe)\b.*\b(?:sim|simulation|sim2real|rerun|timeline|rollout|run)\b",
    re.IGNORECASE,
)

_INTENT_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "load_franka",
        re.compile(
            r"\b(load|show|open)\b.*\b(franka|demo|rerun)\b"
            r"|\b(franka|demo)\b.*\b(load|rerun|show|view)\b"
            r"|\bload franka\b",
            re.IGNORECASE,
        ),
    ),
    (
        "watch_sim",
        re.compile(
            r"\b(?:watch|monitor|follow|observe)\b.*\b(?:sim|simulation|sim2real|rerun|timeline|rollout|run)\b"
            r"|\b(?:track|tail)\b.*\b(?:sim|simulation|sim2real|rerun|timeline|rollout|run)\b"
            r"|\b(?:open|show)\b.*\b(?:rerun|timeline|sim\s+viz)\b",
            re.IGNORECASE,
        ),
    ),
    ("sim2real_status", STATUS_QUERY_RE),
    (
        "sim_assets",
        re.compile(
            r"\b(sim\s*assets?|selection|resolved_uris?|robot_preset|scene_spec)\b"
            r"|\bfranka\b.*\b(preset|selection|assets?)\b"
            r"|\bwhat(?:'s| is)\b.*\bselected\b",
            re.IGNORECASE,
        ),
    ),
    (
        "cameras",
        re.compile(
            r"\b(cameras?|workspace camera|wrist camera|camera selection|frustum)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "tools_catalog",
        re.compile(
            r"\b(tools?|toolref|tool refs?|workbench catalog|what can workbench do)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "configure_s3",
        re.compile(
            r"\b(configure\s+s3|s3\s+bucket|storage\s+bucket|bucket\s+name)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "cosmos3",
        re.compile(
            r"\b(cosmos3?|setup cosmos|cosmos check|cosmos fetch)\b",
            re.IGNORECASE,
        ),
    ),
]

INTENT_APIS: dict[str, list[str]] = {
    "watch_sim": ["sim-viz/status", "workflows/sim2real/status"],
    "sim2real_status": ["sim-viz/status", "workflows/sim2real/status"],
    "sim_assets": ["sim-assets", "sim-assets/selection"],
    "cameras": ["sim-assets/cameras"],
    "tools_catalog": ["tools"],
    "configure_s3": ["tools"],
    "cosmos3": [],
    "load_franka": ["sim-viz/load-franka-demo", "sim-viz/status"],
}


def match_chat_intent(user_text: str) -> str | None:
    text = str(user_text or "").strip()
    if not text:
        return None
    for intent, pattern in _INTENT_RULES:
        if pattern.search(text):
            return intent
    return None


def _sim_viz(state: dict[str, Any]) -> dict[str, Any]:
    sim_viz = state.get("sim_viz", {})
    return sim_viz if isinstance(sim_viz, dict) else {}


def _selection(state: dict[str, Any]) -> dict[str, Any]:
    selection = state.get("selection", {})
    return selection if isinstance(selection, dict) else {}


def _latest_submit(state: dict[str, Any]) -> dict[str, Any]:
    latest = state.get("latest_submit", {})
    return latest if isinstance(latest, dict) else {}


def format_live_context_block(state: dict[str, Any]) -> str:
    """Compact JSON snapshot for LLM system prompt injection (no secrets)."""
    sim_viz = _sim_viz(state)
    selection = _selection(state)
    latest = _latest_submit(state)
    snapshot = {
        "sim_viz": {
            "run_id": sim_viz.get("run_id", ""),
            "stage": sim_viz.get("stage", "idle"),
            "camera": sim_viz.get("camera", "workspace"),
            "rerun_ready": bool(sim_viz.get("rerun_ready")),
            "rrd_updated_at": sim_viz.get("rrd_updated_at", ""),
        },
        "selection": {
            "robot_preset": selection.get("robot_preset", ""),
            "sim_backend": selection.get("sim_backend", ""),
            "scene_spec_uri": selection.get("scene_spec_uri", ""),
            "robot_spec_uri": selection.get("robot_spec_uri", ""),
        },
        "latest_submit": {
            "run_id": latest.get("run_id", ""),
            "submitted_at": latest.get("submitted_at", ""),
        },
        "camera_selection": state.get("camera_selection", ["workspace"]),
    }
    return "Live session snapshot (authoritative — prefer over guessing):\n```json\n" + json.dumps(
        snapshot, indent=2, sort_keys=True
    ) + "\n```"


def format_sim2real_status(state: dict[str, Any], *, rerun_ready: bool | None = None) -> str:
    sim_viz = _sim_viz(state)
    latest = _latest_submit(state)
    selection = _selection(state)
    ready = rerun_ready if rerun_ready is not None else bool(sim_viz.get("rerun_ready"))
    run_id = str(sim_viz.get("run_id") or latest.get("run_id") or "").strip() or "none"
    stage = str(sim_viz.get("stage") or "idle").strip() or "idle"
    camera = str(sim_viz.get("camera") or "workspace")
    rrd_updated = str(sim_viz.get("rrd_updated_at") or "").strip() or "n/a"
    rerun_iframe_url = str(sim_viz.get("rerun_iframe_url") or "/rerun/").strip() or "/rerun/"
    submitted_at = str(latest.get("submitted_at") or "").strip() or "n/a"
    robot = str(selection.get("robot_preset") or "franka")
    backend = str(selection.get("sim_backend") or "isaac")
    lines = [
        "**Sim2Real status** (from live session state):",
        f"- **run_id**: `{run_id}`",
        f"- **stage**: `{stage}`",
        f"- **camera**: `{camera}`",
        f"- **rerun_ready**: `{str(ready).lower()}`",
        f"- **rrd_updated_at**: `{rrd_updated}`",
        f"- **rerun_iframe_url**: `{rerun_iframe_url}`",
        f"- **latest_submit_at**: `{submitted_at}`",
        f"- **robot_preset**: `{robot}`",
        f"- **sim_backend**: `{backend}`",
    ]
    if ready:
        lines.append("- Open the **Rerun** panel or use **Load Franka in Rerun** to view the scene.")
    else:
        lines.append("- No `.rrd` yet — click **Load Franka in Rerun** or submit a Sim2Real workflow.")
    return "\n".join(lines)


def format_sim_assets(state: dict[str, Any]) -> str:
    selection = _selection(state)
    resolved = {
        "scene_spec_uri": selection.get("scene_spec_uri", ""),
        "assets_uri": selection.get("assets_uri", ""),
        "robot_spec_uri": selection.get("robot_spec_uri", ""),
        "cameras_uri": selection.get("cameras_uri", ""),
    }
    props = selection.get("props", [])
    if not isinstance(props, list):
        props = []
    lines = [
        "**Sim assets selection**:",
        f"- **robot_preset**: `{selection.get('robot_preset', 'franka')}`",
        f"- **sim_backend**: `{selection.get('sim_backend', 'isaac')}`",
        f"- **scene_spec_uri**: `{resolved['scene_spec_uri'] or 'unset'}`",
        f"- **robot_spec_uri**: `{resolved['robot_spec_uri'] or 'unset'}`",
        f"- **cameras_uri**: `{resolved['cameras_uri'] or 'unset'}`",
        f"- **assets_uri**: `{resolved['assets_uri'] or 'unset'}`",
        f"- **props**: `{', '.join(str(p) for p in props) or 'none'}`",
        "- Use the **Sim Assets** panel or POST selection to change presets before submit.",
    ]
    return "\n".join(lines)


def format_cameras(state: dict[str, Any], *, default_cameras: list[dict[str, Any]] | None = None) -> str:
    selected = state.get("camera_selection", ["workspace"])
    if not isinstance(selected, list):
        selected = ["workspace"]
    cameras = default_cameras or []
    lines = [
        "**Cameras**:",
        f"- **selected**: `{', '.join(str(s) for s in selected) or 'workspace'}`",
    ]
    if cameras:
        for cam in cameras:
            if not isinstance(cam, dict):
                continue
            name = str(cam.get("name", ""))
            placement = str(cam.get("placement", ""))
            fov = cam.get("fov", "")
            lines.append(f"- **{name}**: placement `{placement}`, fov `{fov}`")
    else:
        lines.extend(
            [
                "- **workspace**: stock workspace overview camera",
                "- **wrist**: end-effector mounted camera",
            ]
        )
    lines.append("- Use **Preview in Rerun** in the Cameras panel to highlight a frustum.")
    return "\n".join(lines)


def format_tools_catalog(tool_refs: list[str], *, sample_size: int = 8) -> str:
    count = len(tool_refs)
    sample = tool_refs[:sample_size]
    lines = [
        f"**Workbench tool catalog** ({count} toolRefs):",
    ]
    for ref in sample:
        lines.append(f"- `{ref}`")
    if count > sample_size:
        lines.append(f"- … and **{count - sample_size}** more (GET full list from tools API)")
    lines.append("- Invoke tools via `npa workbench <tool> …` or npa.workflow specs on your operator machine.")
    return "\n".join(lines)


def format_configure_s3() -> str:
    return "\n".join(
        [
            "**Configure S3 storage** for NPA workflows:",
            "1. Ensure credentials in `~/.npa/credentials.yaml` on your operator machine.",
            "2. Run `npa configure provision` (or `npa workbench` deploy with `--storage-endpoint`).",
            "3. Use **storage.eu-north1.nebius.cloud** for the primary cluster (not uk-south1 default).",
            "4. Pass `--input-path` / `--output-path` as `s3://…` URIs between pipeline stages.",
            "- **workbench toolRef**: `workbench.nebius-infra` for cluster/storage provisioning.",
        ]
    )


def format_cosmos3_setup() -> str:
    return "\n".join(
        [
            "**Cosmos3 setup** on your operator machine:",
            "1. Check access (no download): `npa workbench cosmos check --output json`",
            "2. Stage source + checkpoint: `npa workbench cosmos fetch --output json`",
            "3. GPU inference smoke: SkyPilot `cosmos3-text-to-image-inference.yaml`",
            "4. Keep guardrails on unless explicitly disabled via workflow env.",
            "- Credentials: Hugging Face token in `~/.npa/credentials.yaml`; optional NGC for some assets.",
        ]
    )


def format_load_franka_status(state: dict[str, Any], *, rerun_ready: bool, loaded_now: bool) -> str:
    sim_viz = _sim_viz(state)
    camera = str(sim_viz.get("camera") or "workspace")
    stage = str(sim_viz.get("stage") or "demo")
    run_id = str(sim_viz.get("run_id") or "franka-demo")
    action = "Loaded" if loaded_now else "Already loaded"
    lines = [
        f"**{action} stock Franka demo** in Rerun:",
        f"- **run_id**: `{run_id}`",
        f"- **stage**: `{stage}`",
        f"- **camera**: `{camera}`",
        f"- **rerun_ready**: `{str(rerun_ready).lower()}`",
    ]
    if rerun_ready:
        lines.append("- The **Rerun** iframe uses an authenticated blob fetch — open the center panel to view.")
    else:
        lines.append("- Rerun service still starting — wait a few seconds and refresh.")
    return "\n".join(lines)


def build_grounded_reply(
    intent: str,
    state: dict[str, Any],
    tool_refs: list[str],
    *,
    rerun_ready: bool | None = None,
    loaded_franka_now: bool = False,
    default_cameras: list[dict[str, Any]] | None = None,
) -> str:
    if intent == "watch_sim":
        status = format_sim2real_status(state, rerun_ready=rerun_ready)
        return (
            status
            + "\n- Keep the **Rerun** panel open; poll `/api/sim-viz/status` until `rrd_uri` becomes non-empty."
            + "\n- Workflow watchers should continue until iframe blob mount reports **SUCCESS**."
        )
    if intent == "sim2real_status":
        return format_sim2real_status(state, rerun_ready=rerun_ready)
    if intent == "sim_assets":
        return format_sim_assets(state)
    if intent == "cameras":
        return format_cameras(state, default_cameras=default_cameras)
    if intent == "tools_catalog":
        return format_tools_catalog(tool_refs)
    if intent == "configure_s3":
        return format_configure_s3()
    if intent == "cosmos3":
        return format_cosmos3_setup()
    if intent == "load_franka":
        ready = rerun_ready if rerun_ready is not None else bool(_sim_viz(state).get("rerun_ready"))
        return format_load_franka_status(state, rerun_ready=ready, loaded_now=loaded_franka_now)
    return format_sim2real_status(state, rerun_ready=rerun_ready)


def apis_for_intent(intent: str) -> list[str]:
    return list(INTENT_APIS.get(intent, []))
