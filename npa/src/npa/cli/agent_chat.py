"""Grounded chat intent routing for the NPA agent VM backend."""

from __future__ import annotations

import json
import re
from typing import Any

STATUS_QUERY_RE = re.compile(
    r"(?:\b(?:what(?:'s| is)|show|tell me|check|get)\b.*\b(?:current\s+)?"
    r"(?:sim\s*[- ]?2\s*[- ]?real|sim2real|workflow|rerun|sim(?:\s*[-_ ]?viz)))"
    r"|\b(?:sim\s*[- ]?2\s*[- ]?real|workflow|rerun)\b.*\bstatus\b"
    r"|\bstatus\b.*\b(?:sim\s*[- ]?2\s*[- ]?real|sim2real|workflow|rerun|sim(?:\s*[-_ ]?viz)|stage|run)\b"
    r"|\b(?:watch|monitor|follow|observe)\b.*\b(?:sim|simulation|sim2real|rerun|timeline|rollout|run)\b",
    re.IGNORECASE,
)

_WATCH_SUCCESS_GATE_RE = re.compile(
    r"\b(?:rerun|blob|iframe|rrd(?:-blob)?)\b[\s:;,_\-/+]*(?:blob|iframe|mount|rrd)?"
    r".{0,240}\b(?:until|till|when|once|retry|wait)\b.{0,240}\b(?:success|successful|ready|succeeded|passed|green)\b"
    r"|\b(?:rerun[_ -]?blob[_ -]?success|rerun[_ -]?mount[_ -]?success)\b"
    r"|\bRERUN_(?:BLOB|MOUNT)_SUCCESS\b"
    r"|\brrd[_ -]?uri\b.{0,240}\b(?:non[- ]?empty|set|available|present|populated|not\s+empty)\b"
    r"|\brun[_ -]?id\b.{0,240}\b(?:scoped|scope|match|matching)\b.{0,240}\b(?:success|ready)\b"
    r"|\b(?:run[_ -]?id|stage|stage[_ -]?id)\b.{0,240}\b(?:scoped|scope|match|matching)\b.{0,240}\b(?:success|ready)\b",
    re.IGNORECASE,
)

_RERUN_SUCCESS_PHRASE_RE = re.compile(
    r"\b(?:rerun|rrd|blob|iframe|mount)\b.{0,300}\b(?:until|till|when|once|retry|wait|keep trying)\b.{0,300}\b(?:success|successful|ready|succeeded|passed|green)\b"
    r"|\b(?:until|till|when|once)\b.{0,300}\b(?:success|successful|ready|succeeded|passed|green)\b.{0,300}\b(?:rerun|rrd|blob|iframe|mount)\b"
    r"|\b(?:blob|iframe|mount)\b.{0,300}\b(?:both|all)\b.{0,120}\b(?:success|successful|ready)\b"
    r"|\b(?:both|all)\b.{0,300}\b(?:blob|iframe|mount)\b.{0,120}\b(?:success|successful|ready)\b",
    re.IGNORECASE,
)

_INTENT_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "watch_sim",
        re.compile(
            r"\b(?:watch|monitor|follow|observe)\b.*\b(?:sim|simulation|sim2real|rerun|timeline|rollout|run)\b"
            r"|\b(?:track|tail)\b.*\b(?:sim|simulation|sim2real|rerun|timeline|rollout|run)\b"
            r"|\b(?:open|show|view)\b.*\b(?:rerun|timeline|sim\s+viz|iframe)\b"
            r"|\b(?:live|latest)\b.*\b(?:sim|simulation|rerun|timeline)\b"
            r"|\b(?:stage|run)\b.*\b(?:badge|overlay)\b"
            r"|\b(?:keep|continue)\b.*\b(?:watching|monitoring|tracking)\b.*\b(?:sim|rerun|timeline|run)\b"
            r"|\b(?:keep me posted|live updates?)\b.*\b(?:sim|simulation|rerun|timeline|run)\b"
            r"|\b(?:poll|refresh)\b.*\b(?:sim-viz/status|rrd|iframe|rerun)\b"
            r"|\b(?:blob|iframe)\b.*\b(?:success|ready)\b"
            r"|\bblob\s*(?:\+|and)\s*iframe\b.*\b(?:success|ready)\b"
            r"|\bblob\s*/\s*iframe\b.*\b(?:success|ready)\b"
            r"|\brerun\s+blob\s*/\s*iframe\b.*\b(?:success|ready)\b"
            r"|\bboth\b.*\b(?:blob|iframe)\b.*\b(?:success|ready)\b"
            r"|\buntil\b.*\b(?:success|ready)\b.*\b(?:blob|iframe|rerun)\b"
            r"|\b(?:wait|retry|rerun)\b.*\b(?:blob|iframe)\b.*\b(?:success|ready)\b"
            r"|\b(?:retry|rerun)\b.*\b(?:blob|iframe)\b.*\buntil\b.*\b(?:success|ready)\b"
            r"|\brerun\b.*\b(?:blob|iframe)\b.*\buntil\b.*\bsuccess\b"
            r"|\b(?:blob|iframe)\b.*\buntil\b.*\bsuccess\b"
            r"|\buntil\b.*\b(?:blob|iframe)\b.*\bsuccess\b"
            r"|\b(?:rerun_blob_success|rerun_mount_success)\b"
            r"|\brerun[_ -]?blob[_ -]?success\b|\brerun[_ -]?mount[_ -]?success\b"
            r"|\brerun[_ -]?iframe[_ -]?until[_ -]?success\b"
            r"|\brerun[_ -]?blob[_ -]?iframe[_ -]?until[_ -]?success\b"
            r"|\bblob\s*\+\s*iframe\s*until\s*success\b"
            r"|\bblob\b.*\biframe\b.*\buntil\b.*\b(?:success|ready)\b"
            r"|\biframe\b.*\bblob\b.*\buntil\b.*\b(?:success|ready)\b"
            r"|\brerun\b.*\bblob\b.*\biframe\b.*\buntil\b.*\bsuccess\b"
            r"|\brerun\b.*\bblob\s*/\s*iframe\b.*\buntil\b.*\bsuccess\b"
            r"|\brrd-blob\b.*\b(?:success|ready)\b"
            r"|\b(?:watch|monitor)\b.*\brrd\b.*\b(?:success|ready)\b"
            r"|\bRERUN_BLOB_SUCCESS\b|\bRERUN_MOUNT_SUCCESS\b"
            r"|\bblob[_ -]?mount\b.*\bsuccess\b"
            r"|\biframe[_ -]?mount\b.*\bsuccess\b"
            r"|\bboth\b.*\bsuccess\b.*\b(?:blob|iframe|mount)\b"
            r"|\b(?:when|once)\b.*\brrd\b.*\b(?:arrives|lands|updates?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "load_franka",
        re.compile(
            r"\b(load|show|open)\b.*\b(franka|demo)\b"
            r"|\b(franka|demo)\b.*\b(load|rerun|show|view)\b"
            r"|\bload franka\b",
            re.IGNORECASE,
        ),
    ),
    ("sim2real_status", STATUS_QUERY_RE),
    (
        "sim_assets",
        re.compile(
            r"\b(sim\s*assets?|selection|resolved_uris?|robot_preset|scene_spec)\b"
            r"|\bfranka\b.*\b(preset|selection|assets?)\b"
            r"|\bwhat(?:'s| is)\b.*\bselected\b"
            r"|\b(?:select|specify|set)\b.*\b(?:scene|robot|props?|cameras?)\b"
            r"|\b(?:scene|robot|props?|cameras?)\b.*\b(?:selection|selector|mode)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "cameras",
        re.compile(
            r"\b(cameras?|workspace camera|wrist camera|camera selection|frustum)\b"
            r"|\bcamera\s+angle\s+inspector\b"
            r"|\b(?:preview|thumbnail|top[- ]down)\b.*\b(?:camera|frustum)\b",
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
    "watch_sim": ["sim-viz/status", "sim-viz/rrd", "sim-viz/rrd-blob", "workflows/sim2real/status"],
    "sim2real_status": ["sim-viz/status", "workflows/sim2real/status"],
    "sim_assets": ["sim-assets", "sim-assets/selection"],
    "cameras": ["sim-assets/cameras"],
    "tools_catalog": ["tools"],
    "configure_s3": ["tools"],
    "cosmos3": [],
    "load_franka": ["sim-viz/load-franka-demo", "sim-viz/status"],
}


def _normalize_intent_text(text: str) -> str:
    """Normalize user text so intent routing survives punctuation/newlines."""
    lowered = str(text or "").lower()
    lowered = lowered.replace("\n", " ")
    lowered = lowered.replace("rerunblobiframeuntilsuccess", "rerun blob iframe until success")
    lowered = lowered.replace("blobiframeuntilsuccess", "blob iframe until success")
    lowered = lowered.replace("rerunblobuntilsuccess", "rerun blob until success")
    lowered = lowered.replace("reruniframeuntilsuccess", "rerun iframe until success")
    lowered = lowered.replace("rerunblobiframetilsuccess", "rerun blob iframe till success")
    lowered = lowered.replace("rerunblobiframeuntilsuccessful", "rerun blob iframe until successful")
    lowered = lowered.replace("rrduripopulated", "rrd uri populated")
    lowered = lowered.replace("rrduri", "rrd uri")
    lowered = lowered.replace("runid", "run id")
    lowered = lowered.replace("rrduriuntilsuccess", "rrd uri until success")
    lowered = lowered.replace("runidrrduriuntilsuccess", "run id rrd uri until success")
    lowered = lowered.replace("runid/rrduri", "run id rrd uri")
    lowered = lowered.replace("runidrrdurisuccess", "run id rrd uri success")
    lowered = lowered.replace("rrdurinonempty", "rrd uri non-empty")
    lowered = lowered.replace("rrdurinotempty", "rrd uri not empty")
    lowered = lowered.replace("rrduriset", "rrd uri set")
    lowered = lowered.replace("runidscoped", "run id scoped")
    lowered = lowered.replace("runidscope", "run id scope")
    lowered = lowered.replace("stageid", "stage id")
    lowered = lowered.replace("stagescoped", "stage scoped")
    lowered = lowered.replace("runidstagescoped", "run id stage scoped")
    lowered = lowered.replace("stagematch", "stage match")
    lowered = lowered.replace("stagematching", "stage matching")
    lowered = lowered.replace("watchthesim", "watch the sim")
    lowered = lowered.replace("watchsim", "watch sim")
    lowered = lowered.replace("watchsimuntilsuccess", "watch sim until success")
    # Normalize common alias/camelcase variants before regex matching.
    lowered = re.sub(r"\bsim[_\s-]?viz\b", "sim viz", lowered)
    lowered = re.sub(r"\brrd[_\s-]?blob\b", "rrd blob", lowered)
    lowered = re.sub(r"\brerun[_\s-]?blob\b", "rerun blob", lowered)
    lowered = re.sub(r"\brerun[_\s-]?mount\b", "rerun mount", lowered)
    lowered = re.sub(r"\brrd[_\s-]?uri\b", "rrd uri", lowered)
    lowered = re.sub(r"[^\w\s/+.-]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _success_gated_watch_request(lowered: str) -> bool:
    """Detect explicit blob/iframe SUCCESS gating language for watch intent."""
    if _WATCH_SUCCESS_GATE_RE.search(lowered) or _RERUN_SUCCESS_PHRASE_RE.search(lowered):
        return True
    # Keep explicit "rerun blob iframe until SUCCESS" intent sticky even when
    # operators append branch/bootstrap notes in the same sentence.
    if (
        "rerun" in lowered
        and "blob" in lowered
        and "iframe" in lowered
        and any(token in lowered for token in ("until", "till", "when", "once", "retry", "wait"))
        and any(
            token in lowered
            for token in ("success", "successful", "succeeded", "passed", "green", "ready")
        )
    ):
        return True
    compact = re.sub(r"[^a-z0-9]+", "", lowered)
    if any(
        token in compact
        for token in (
            "rerunblobiframeuntilsuccess",
            "blobiframeuntilsuccess",
            "rerunblobuntilsuccess",
            "reruniframeuntilsuccess",
            "rerunblobiframetilsuccess",
            "rerunblobiframeuntilsuccessful",
            "rerunmountsuccess",
            "rerunblobsuccess",
            "runidscopedrerunblobiframeuntilsuccess",
            "watchrrduriuntilsuccess",
            "rrduriuntilsuccess",
            "rrdurinonemptyuntilsuccess",
            "rrdurinotemptyuntilsuccess",
            "rrdurisetuntilsuccess",
            "rrduripopulateduntilsuccess",
            "runidrrduriuntilsuccess",
            "runidrrdurisuccess",
            "runidrrduri",
            "runidstagescopedrerunblobiframeuntilsuccess",
            "runidstageuntilsuccess",
        )
    ):
        return True
    has_rerun_surface = any(
        token in lowered
        for token in (
            "rerun",
            "blob",
            "iframe",
            "rrd-blob",
            "rrd",
            "rrd uri",
            "run id",
            "active run",
            "stage",
            "stage id",
        )
    )
    has_success_gate = any(
        token in lowered
        for token in (
            "success",
            "ready",
            "successful",
            "succeeded",
            "passed",
            "green",
            "until success",
            "until ready",
            "until both",
            "both success",
            "both are success",
            "rerun_blob_success",
            "rerun_mount_success",
            "iframe mount success",
            "rerun blob iframe until success",
            "rrd uri non-empty",
            "rrd uri set",
            "rrd uri available",
            "rrd uri populated",
            "rrd uri not empty",
            "consecutive success",
            "success streak",
            "stable success",
        )
    )
    return has_rerun_surface and has_success_gate


def match_chat_intent(user_text: str) -> str | None:
    text = str(user_text or "").strip()
    if not text:
        return None
    lowered = _normalize_intent_text(text)
    if _success_gated_watch_request(lowered):
        return "watch_sim"
    # Keep watch intent precedence over load-franka whenever the user asks to
    # monitor/retry the rerun view (especially with SUCCESS gating language).
    if (
        ("franka" in lowered or "load franka" in lowered)
        and (
            "watch" in lowered
            or "monitor" in lowered
            or "track" in lowered
            or "blob" in lowered
            or "iframe" in lowered
            or "until success" in lowered
        )
    ):
        return "watch_sim"
    for intent, pattern in _INTENT_RULES:
        if pattern.search(text) or pattern.search(lowered):
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
        sim_viz = _sim_viz(state)
        iframe_url = str(sim_viz.get("rerun_iframe_url") or "/rerun/").strip() or "/rerun/"
        stage = str(sim_viz.get("stage") or "idle").strip() or "idle"
        run_id = str(sim_viz.get("run_id") or "").strip() or "none"
        mode = str(sim_viz.get("mode") or "static").strip() or "static"
        rrd_uri = str(sim_viz.get("rrd_uri") or "").strip() or "pending"
        status = format_sim2real_status(state, rerun_ready=rerun_ready)
        return (
            status
            + f"\n- **watch_url**: `{iframe_url}` (embedded iframe) with run/stage badge overlay."
            + f"\n- **watch_stage**: `{stage}` for **run_id** `{run_id}`."
            + f"\n- **watch_mode**: `{mode}`."
            + f"\n- **rrd_uri**: `{rrd_uri}`."
            + "\n- **operator_gate**: `Rerun blob iframe until SUCCESS`."
            + "\n- Start from `GET /api/sim-viz/status` and surface `watch_url`, `watch_stage`, and `run_id`."
            + "\n- Keep the **Rerun** panel open; poll `/api/sim-viz/status` until `rrd_uri` becomes non-empty."
            + "\n- Keep polling until stage transitions beyond `submitted` and a fresh `rrd_updated_at` appears."
            + "\n- Then keep retrying **blob fetch + iframe mount** until both report **SUCCESS**."
            + "\n- Treat watch complete only when `RERUN_BLOB_SUCCESS=SUCCESS` and `RERUN_MOUNT_SUCCESS=SUCCESS`."
            + "\n- Keep the stage/run overlay aligned to the active run by re-reading `run_id` + `stage` from `/api/sim-viz/status` each loop."
            + "\n- If prompted as `Rerun blob iframe until SUCCESS`, follow the same two-signal gate above."
            + "\n- Re-check `/api/sim-viz/status` after each mount loop and keep gating on the active `run_id`."
            + "\n- If either status is not `SUCCESS`, retry mounting `/rerun/` and fetching `/api/sim-viz/rrd-blob`."
            + "\n- Use `/api/sim-viz/rrd` as fallback source if blob fetch has transient failures."
            + "\n- Keep looping until both checks remain `SUCCESS` for the active run/stage badge."
            + "\n- Require **consecutive SUCCESS confirmations** (success streak) before declaring watch complete."
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
