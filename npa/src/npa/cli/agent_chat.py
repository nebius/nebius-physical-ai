"""Grounded chat intent routing for the NPA agent VM backend."""

from __future__ import annotations

import json
import os
import re
from typing import Any

BYOF_ONBOARD_SKILL_PATH = "skills/workflows/byof-onboard/SKILL.md"
OSS_SOLUTION_REGISTRY_ONBOARD_SKILL_PATH = (
    "skills/workflows/oss-solution-registry-onboard/SKILL.md"
)

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
    r".{0,240}\b(?:until|till|when|once|retry|wait)\b.{0,240}\b(?:success|successful|ready|succeeded|passed|green|healthy)\b"
    r"|\b(?:rerun[_ -]?blob[_ -]?success|rerun[_ -]?mount[_ -]?success)\b"
    r"|\bRERUN_(?:BLOB|MOUNT)_SUCCESS\b"
    r"|\brrd[_ -]?uri\b.{0,240}\b(?:non[- ]?empty|set|available|present|populated|not\s+empty)\b"
    r"|\b(?:timeline|sim(?:\s*[-_ ]?viz)|watch(?:\s+the)?\s+sim)\b.{0,240}\b(?:until|till|when|once|retry|wait)\b.{0,240}\b(?:success|successful|ready|succeeded|passed|green|healthy)\b"
    r"|\brun[_ -]?id\b.{0,240}\b(?:scoped|scope|match|matching)\b.{0,240}\b(?:success|ready)\b"
    r"|\b(?:run[_ -]?id|stage|stage[_ -]?id)\b.{0,240}\b(?:scoped|scope|match|matching)\b.{0,240}\b(?:success|ready)\b",
    re.IGNORECASE,
)

_RERUN_SUCCESS_PHRASE_RE = re.compile(
    r"\b(?:rerun|rrd|blob|iframe|mount)\b.{0,300}\b(?:until|till|when|once|retry|wait|keep trying)\b.{0,300}\b(?:success|successful|ready|succeeded|passed|green|healthy)\b"
    r"|\b(?:until|till|when|once)\b.{0,300}\b(?:success|successful|ready|succeeded|passed|green|healthy)\b.{0,300}\b(?:rerun|rrd|blob|iframe|mount)\b"
    r"|\b(?:blob|iframe|mount)\b.{0,300}\b(?:both|all)\b.{0,120}\b(?:success|successful|ready|healthy)\b"
    r"|\b(?:both|all)\b.{0,300}\b(?:blob|iframe|mount)\b.{0,120}\b(?:success|successful|ready|healthy)\b",
    re.IGNORECASE,
)

_NON_STOCK_ARTIFACT_DISCOVERY_RE = re.compile(
    r"\b(?:non[\s-]?stock|customer|custom)\b"
    r".{0,160}\b(?:run|sim\s*[- ]?2\s*[- ]?real|sim2real)\b"
    r".{0,160}\b(?:artifacts?|outputs?|recording|rrd|video|report|logs?|view|load|use)\b",
    re.IGNORECASE,
)

_INTENT_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "start_sim2real",
        re.compile(
            r"\b(?:start|run|launch|execute|kick\s*off|submit)\b"
            r".{0,120}\b(?:sim\s*[- ]?2\s*[- ]?real|sim2real|simulation\s+pipeline|pipeline)\b"
            r"|\b(?:sim\s*[- ]?2\s*[- ]?real|sim2real)\b.{0,120}\b(?:start|launch|execute|kick\s*off|submit)\b",
            re.IGNORECASE,
        ),
    ),
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
            r"|\bwatch(?:\s+the)?\s+sim\b"
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
    (
        "create_vlm_rl_workflow",
        re.compile(
            r"\b(?:create|generate|build|make|draft|compose|write)\b"
            r".{0,120}\b(?:vlm[_\s-]?rl|vlm\s+rl|rl[_\s-]?vlm)\b"
            r"|\b(?:vlm[_\s-]?rl|rl[_\s-]?vlm)\b.{0,120}\b(?:workflow|yaml|spec|loop)\b"
            r"|\b(?:outer|inner)\b.{0,80}\b(?:loop|iteration)\b.{0,120}\b(?:workflow|yaml|spec)\b"
            r"|\b(?:outer\s+loop|inner\s+loop)\b.{0,80}\b(?:gate|decision|promote)\b"
            r"|\b(?:create|generate|build|make)\b.{0,80}\b(?:vlm|critic)\b.{0,80}\b(?:gate|loop|workflow)\b"
            r"|\b(?:policy\s+rollout|heldout\s+eval)\b.{0,80}\b(?:workflow|yaml|spec|loop)\b"
            r"|\b(?:workflow|yaml|spec)\b.{0,120}\b(?:policy\s+rollout|heldout\s+eval|vlm\s+critic|quality\s+gate)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "create_gate_workflow",
        re.compile(
            r"\b(?:create|generate|build|make|draft|compose|write)\b"
            r".{0,120}\b(?:token[_\s-]?factory|tokenfactory)\b"
            r"|\b(?:token[_\s-]?factory|tokenfactory)\b.{0,80}\b(?:workflow|yaml|spec|gate)\b"
            r"|\b(?:quality[_\s-]?gate|cosmos[_\s-]?gate|augment[_\s-]?gate)\b.{0,80}\b(?:workflow|yaml|spec|loop)\b"
            r"|\b(?:create|generate|build|make)\b.{0,80}\b(?:quality|augment)\b.{0,80}\b(?:gate|loop|workflow)\b"
            r"|\breason[_\s-]?scene\b.{0,80}\b(?:workflow|yaml|spec)\b"
            r"|\b(?:scene\s+reasoning|cosmos\s+reason)\b.{0,80}\b(?:workflow|yaml|spec|loop)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "create_workflow",
        re.compile(
            r"\b(?:create|generate|build|make|draft|compose|write)\b"
            r".{0,80}\b(?:2[\s-]?step|two[\s-]?step)\b"
            r".{0,80}\b(?:sim\s*[- ]?2\s*[- ]?real|sim2real)\b"
            r".{0,40}\b(?:workflow|yaml|spec)\b"
            r"|\b(?:create|generate|build|make|draft|compose|write)\b"
            r".{0,80}\b(?:npa[\s.-]?workflow|workflow\s+yaml|workflow\s+spec)\b"
            r".{0,80}\b(?:sim\s*[- ]?2\s*[- ]?real|sim2real)\b"
            r"|\b(?:2[\s-]?step|two[\s-]?step)\b"
            r".{0,80}\b(?:sim\s*[- ]?2\s*[- ]?real|sim2real)\b"
            r".{0,40}\b(?:workflow|yaml|spec)\b"
            r"|\b(?:create|generate|build|make|draft|compose|write)\b"
            r".{0,120}\b(?:leisaac|isaac[\s-]?lab|byof)\b"
            r".{0,120}\b(?:workflow|yaml|spec)\b"
            r"|\b(?:leisaac|isaac[\s-]?lab)\b"
            r".{0,120}\b(?:workflow|yaml|spec)\b"
            r"|\b(?:create|generate|build|make|draft|compose|write)\b"
            r".{0,120}\bgpu\b"
            r".{0,120}\b(?:workflow|yaml|spec)\b"
            r".{0,120}\b(?:multi[\s-]?region|cross[\s-]?region|2(?:\s+different)?\s+regions?|two(?:\s+different)?\s+regions?|multi[\s-]?project|cross[\s-]?project)\b"
            r"|\b(?:multi[\s-]?region|cross[\s-]?region|2(?:\s+different)?\s+regions?|two(?:\s+different)?\s+regions?)\b"
            r".{0,120}\bgpu\b"
            r".{0,120}\b(?:workflow|yaml|spec)\b"
            r"|\b(?:generate|create|draft|write|show)\b.{0,80}\b(?:example|simple|minimal)?\b.{0,120}\bworkflow\b.{0,80}\b(?:yaml|spec)\b"
            r"|\bworkflow\b.{0,80}\b(?:yaml|spec)\b.{0,80}\b(?:example|simple|minimal)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "find_artifacts",
        re.compile(
            r"\b(?:find|discover|list|browse|show|view|open|inspect)\b.{0,120}\b(?:artifacts?|outputs?)\b"
            r"|\bwhat can i view\b"
            r"|\bwhat\b.{0,80}\bartifacts?\b.{0,120}\bview\b"
            r"|\b(?:what|which)\b.{0,80}\b(?:sim\s*[- ]?2\s*[- ]?real|sim2real)?\s*run\b.{0,80}\b(?:view|load|open|use)\b"
            r"|\b(?:non[\s-]?stock|customer|custom)\b.{0,120}\b(?:run|sim\s*[- ]?2\s*[- ]?real|sim2real)\b.{0,120}\b(?:artifacts?|outputs?|recording|rrd|video|report|logs?)\b"
            r"|\bartifact\b.{0,120}\b(?:browser|viewer|preview|download)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "onboard_solution",
        re.compile(
            r"\b(?:onboard|add|integrate|containerize|dockerize|register)\b.{0,140}\b(?:solution|tool|component|repo|repository|open[\s-]?source)\b"
            r"|\b(?:byof|bring your own fork|custom fork)\b.{0,140}\b(?:workflow|image|container|registry|infra|run)\b"
            r"|\b(?:github|repo(?:sitory)?)\b.{0,140}\b(?:workbench|npa|workflow|sky|kubernetes)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "list_recordings",
        re.compile(
            r"\b(?:list|show|view|browse|get|fetch)\b.{0,80}\b(?:recordings?|run\s+history|runs?)\b"
            r"|\b(?:list|show|view|browse|get|fetch)\b.{0,80}\.rrd\b"
            r"|\b(?:recordings?|run\s+history|past\s+runs?|available\s+runs?)\b"
            r"|\b(?:switch|load|open)\b.{0,80}\b(?:recording|run)\b.{0,80}\b(?:from\s+history|another|different|other)\b"
            r"|\b(?:other\s+run|different\s+run|previous\s+run|past\s+run)\b"
            r"|\bavailable\b.{0,40}\.rrd\b",
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
        "infra_backends",
        re.compile(
            r"\b(?:k8s|kubernetes|cluster|clusters|backend|backends|infra|infrastructure)\b"
            r".{0,120}\b(?:present|available|configured|exists?|list|show|query|which|what)\b"
            r"|\b(?:list|show|query|what|which)\b.{0,120}\b(?:k8s|kubernetes|clusters?|backends?|infra)\b"
            r"|\bno\b.{0,80}\b(?:infra|infrastructure|cluster|backend)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "mk8s_provision",
        re.compile(
            r"\b(?:deploy|create|provision|ensure|spin\s*up)\b"
            r".{0,120}\b(?:mk8s|managed\s+kubernetes|k8s|kubernetes)\b"
            r".{0,80}\b(?:cluster|backend|infra|infrastructure)?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "live_infra_loop",
        re.compile(
            r"\b(?:run|submit|launch|test)\b.{0,120}\b(?:live|real)\b.{0,120}\b(?:infra|infrastructure|dev\s*vm)\b"
            r"|\b(?:tmux|cursor[- ]?loop|loop)\b.{0,120}\b(?:live|real)\b.{0,120}\b(?:infra|workflow)\b"
            r"|\b(?:verify|check)\b.{0,120}\b(?:gpu|accelerator)\b.{0,120}\b(?:compat|compatibility)\b"
            r"|\b(?:retry|loop)\b.{0,120}\b(?:failed[_ -]?prechecks|precheck)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "cosmos_capabilities",
        re.compile(
            r"\bcosmos(?:2|3)?\b.{0,120}\b(?:support|supports|capabilit(?:y|ies)|expose|offers|finetun(?:e|ing)|train(?:ing)?|infer(?:ence)?)\b"
            r"|\b(?:support|supports|capabilit(?:y|ies)|expose|offers)\b.{0,120}\bcosmos(?:2|3)?\b"
            r"|\b(?:can|could)\b.{0,80}\bcosmos(?:2|3)?\b.{0,120}\b(?:do|run|train|finetun(?:e|ing)|infer)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "lancedb_capabilities",
        re.compile(
            r"\blancedb\b.{0,120}\b(?:support|supports|capabilit(?:y|ies)|expose|offers|import|backfill|view|query)\b"
            r"|\b(?:support|supports|capabilit(?:y|ies)|expose|offers)\b.{0,120}\blancedb\b",
            re.IGNORECASE,
        ),
    ),
    (
        "component_capabilities",
        re.compile(
            r"\b(?:component|tool|workbench)\b.{0,120}\b(?:support|supports|capabilit(?:y|ies)|expose|offers)\b"
            r"|\bwhat\b.{0,80}\b(?:does|can)\b.{0,80}\b(?:cosmos|lancedb|sonic|isaac(?:\s|-)?lab|lerobot|groot|token(?:\s|-)?factory)\b.{0,80}\b(?:support|do|expose)\b",
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
    (
        "soperator",
        re.compile(
            r"\b(soperator|slurm(?:[- ]on[- ]k(?:ubernetes|8s))?|slurm cluster"
            r"|deploy\s+slurm|slurm\s+deploy)\b",
            re.IGNORECASE,
        ),
    ),
]

INTENT_APIS: dict[str, list[str]] = {
    "watch_sim": ["sim-viz/status", "sim-viz/rrd", "sim-viz/rrd-blob", "workflows/sim2real/status"],
    "find_artifacts": ["artifacts/runs", "artifacts/run/{run_id}", "sim-viz/load-artifact", "sim-viz/status"],
    "create_workflow": ["workflows/draft", "workflows/validate"],
    "create_vlm_rl_workflow": ["workflows/draft", "workflows/validate", "workflows/plan"],
    "create_gate_workflow": ["workflows/draft", "workflows/validate", "workflows/plan"],
    "onboard_solution": ["tools", "workflows/validate", "workflows/plan"],
    "infra_backends": ["infra/k8s", "infra/provision", "workflows/submit"],
    "mk8s_provision": ["infra/mk8s", "infra/mk8s/provision", "infra/k8s"],
    "live_infra_loop": ["infra/k8s", "infra/provision", "workflows/validate", "workflows/plan", "workflows/submit", "tools"],
    "list_recordings": ["sim-viz/recordings", "sim-viz/runs"],
    "sim2real_status": ["sim-viz/status", "workflows/sim2real/status"],
    "sim_assets": ["sim-assets", "sim-assets/selection"],
    "cameras": ["sim-assets/cameras"],
    "cosmos_capabilities": ["tools"],
    "lancedb_capabilities": ["tools"],
    "component_capabilities": ["tools"],
    "tools_catalog": ["tools"],
    "configure_s3": ["tools"],
    "cosmos3": [],
    "soperator": ["infra/soperator/validate", "infra/soperator/deploy", "infra/soperator/status/{name}", "tools"],
    "load_franka": ["sim-viz/load-franka-demo", "sim-viz/status"],
}

_DEFAULT_REGISTRY = "cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw"
_DEFAULT_TOOL_IMAGE_TAGS: dict[str, tuple[str, str]] = {
    "cosmos": ("npa-cosmos", "1.0.9"),
    "lancedb": ("npa-lancedb", "0.30.3"),
    "isaac-lab": ("npa-isaac-lab", "2.3.2.post1"),
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
            for token in ("success", "successful", "succeeded", "passed", "green", "ready", "healthy")
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
            "healthy",
            "until success",
            "until ready",
            "until healthy",
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
    if re.search(r"\b(soperator|slurm(?:[- ]on[- ]k(?:ubernetes|8s))?|slurm cluster|deploy\s+slurm|slurm\s+deploy)\b", text, re.IGNORECASE):
        return "soperator"
    if _NON_STOCK_ARTIFACT_DISCOVERY_RE.search(text) or _NON_STOCK_ARTIFACT_DISCOVERY_RE.search(lowered):
        return "find_artifacts"
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


def _image_for_tool(tool: str) -> str:
    registry = os.environ.get("NPA_REGISTRY", "").strip() or _DEFAULT_REGISTRY
    image_name, tag = _DEFAULT_TOOL_IMAGE_TAGS.get(tool, (f"npa-{tool}", "<tag>"))
    return f"{registry.rstrip('/')}/{image_name}:{tag}"


def format_cosmos_capabilities(tool_refs: list[str]) -> str:
    cosmos_refs = [ref for ref in tool_refs if "cosmos" in ref or "token_factory" in ref]
    sample = ", ".join(sorted(cosmos_refs)[:4]) if cosmos_refs else "workbench.cosmos2.transfer"
    cosmos_image = _image_for_tool("cosmos")
    return "\n".join(
        [
            "**Cosmos component capabilities**:",
            "- **Inference**: Cosmos3 text-to-image workflow (`cosmos3-text-to-image-inference.yaml`).",
            "- **Setup + model staging**: `npa workbench cosmos check|fetch`.",
            "- **Fine-tuning / post-training**: `npa workbench cosmos train` (serverless + runtime options).",
            "- **Pipeline integration**: Cosmos augment stage via `workbench.cosmos2.transfer` and Token Factory reasoning paths.",
            f"- **Registry image default**: `{cosmos_image}` (override via `NPA_REGISTRY` if needed).",
            f"- **Catalog examples**: `{sample}`",
            "- Use run-scoped S3 URIs for artifacts and keep credentials in `~/.npa/credentials.yaml`.",
        ]
    )


def format_lancedb_capabilities(tool_refs: list[str]) -> str:
    lancedb_refs = [ref for ref in tool_refs if ref.startswith("workbench.lancedb.")]
    sample = ", ".join(sorted(lancedb_refs)[:5]) if lancedb_refs else "workbench.lancedb.import_bdd100k"
    lancedb_image = _image_for_tool("lancedb")
    return "\n".join(
        [
            "**LanceDB component capabilities**:",
            "- **Data ingest**: BDD100K import into run-scoped Lance tables.",
            "- **Feature backfill**: CPU + GPU UDF backfills (including CLIP embeddings).",
            "- **Dataset shaping**: materialized view creation for failure-mode slices.",
            "- **Serving path**: endpoint-backed execution for workflows and tooling.",
            f"- **Registry image default**: `{lancedb_image}` (use your real registry, never `<your-registry-id>` placeholders).",
            f"- **Catalog examples**: `{sample}`",
            "- Keep table/URI names in config; avoid embedding project-specific constants in workflow states.",
        ]
    )


def format_component_capabilities(tool_refs: list[str]) -> str:
    return "\n".join(
        [
            "**Workbench component capabilities** (customer-facing building blocks):",
            "- **Cosmos**: setup/fetch, inference, and finetuning/post-training lanes.",
            "- **LanceDB**: ingest, backfill, view/materialization, query workflows.",
            "- **Isaac Lab / RL**: train/eval policy building blocks for simulation pipelines.",
            "- **Token Factory + VLM**: reasoning, augment, scoring, and decision-gate loops.",
            "- Ask for a component by name (for example: `Cosmos capabilities`) to get targeted commands + workflow patterns.",
            f"- **Current toolRef count**: `{len(tool_refs)}`",
        ]
    )


def format_live_infra_loop_guidance() -> str:
    isaac_image = _image_for_tool("isaac-lab")
    lancedb_image = _image_for_tool("lancedb")
    cosmos_image = _image_for_tool("cosmos")
    return "\n".join(
        [
            "**Live infra loop guidance (DEV VM + tmux)**:",
            "- Resolve registry images from your actual Nebius registry (no placeholders):",
            f"  - Isaac Lab: `{isaac_image}`",
            f"  - LanceDB: `{lancedb_image}`",
            f"  - Cosmos: `{cosmos_image}`",
            "- GPU compatibility precheck before each launch:",
            "  1. `sky check`",
            "  2. `sky gpus list`",
            "  3. ensure requested accelerator exists in your active K8s context",
            "- Loop pattern in tmux:",
            "```bash",
            "SESSION=live-infra-loop-$(date -u +%Y%m%dT%H%M%SZ)",
            "tmux new -d -s \"$SESSION\"",
            "tmux send-keys -t \"$SESSION:0.0\" 'set -euo pipefail; ATTEMPT=1; while [ $ATTEMPT -le 5 ]; do echo \"attempt=$ATTEMPT\"; npa/.venv/bin/npa workbench workflow validate-spec <spec.yaml> --json && npa/.venv/bin/npa workbench workflow plan-spec <spec.yaml> --run-id loop-$ATTEMPT --json && npa/.venv/bin/python <runner>.py --image <real-registry-image> --gpu-type <compatible-gpu> && break; ATTEMPT=$((ATTEMPT+1)); sleep $((ATTEMPT*15)); done' C-m",
            "```",
            "- If `FAILED_PRECHECKS` appears: adjust image reference or accelerator and retry in the same loop.",
        ]
    )


def format_infra_backends(state: dict[str, Any]) -> str:
    infra = state.get("infra")
    if not isinstance(infra, dict):
        infra = {}
    configured = infra.get("configured") if isinstance(infra.get("configured"), list) else []
    local_clusters = infra.get("local_clusters") if isinstance(infra.get("local_clusters"), list) else []
    cloud_clusters = infra.get("cloud_clusters") if isinstance(infra.get("cloud_clusters"), list) else []
    has_infra = bool(infra.get("has_infra"))
    project = str(infra.get("project") or "default")
    lines = [
        "**Kubernetes / workflow infra status**:",
        f"- **project**: `{project}`",
        f"- **agent_npa_ready**: `{bool(infra.get('agent_npa_ready'))}`",
    ]
    if configured:
        lines.append("- **configured backends**:")
        for item in configured[:5]:
            if isinstance(item, dict):
                lines.append(
                    "  - "
                    f"`{item.get('cluster_name') or item.get('context') or 'configured'}` "
                    f"source=`{item.get('source', 'project_config')}` "
                    f"kubeconfig=`{item.get('kubeconfig', '')}`"
                )
    if local_clusters:
        lines.append("- **local agent clusters**:")
        for item in local_clusters[:5]:
            if isinstance(item, dict):
                lines.append(
                    "  - "
                    f"`{item.get('cluster_name') or item.get('context') or 'cluster'}` "
                    f"kubeconfig_exists=`{bool(item.get('kubeconfig_exists'))}`"
                )
    if cloud_clusters:
        lines.append("- **Nebius MK8s clusters**:")
        for item in cloud_clusters[:5]:
            if isinstance(item, dict):
                lines.append(
                    "  - "
                    f"`{item.get('name') or item.get('id') or 'cluster'}` "
                    f"id=`{item.get('id', '')}` status=`{item.get('status', '')}`"
                )
    if not has_infra:
        lines.extend(
            [
                "- **No Kubernetes infra is currently specified or available.**",
                "- Options:",
                "  1. Let the agent deploy minimal GPU Kubernetes for the workflow (`POST /api/infra/provision`).",
                "  2. Configure an existing backend in `~/.npa/config.yaml` under `projects.<alias>.kubernetes`.",
                "  3. Submit with explicit `project` / `cluster_name` once you choose a target.",
            ]
        )
    else:
        lines.append("- The agent can use the listed backend or provision another one if requested.")
    return "\n".join(lines)


def format_mk8s_provision() -> str:
    return "\n".join(
        [
            "**Deploy or ensure an mk8s Kubernetes backend with npa:**",
            "- API: `POST /api/infra/mk8s/provision` with `project`, `cluster_name`, and optional `dry_run`.",
            "- It calls `npa provision-if-absent` from the agent VM using staged `~/.npa/config.yaml` and credentials.",
            "- Use `dry_run: true` to verify project, storage, and Terraform actions without changing infra.",
            "- Use `GET /api/infra/mk8s` or `GET /api/infra/k8s` to list configured, cached, and cloud mk8s backends.",
            "- Workflow submit can also provision mk8s when `allow_provision: true` and no backend exists.",
        ]
    )


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


def format_generate_workflow(
    yaml_text: str,
    validation: dict[str, Any],
    *,
    template: str = "two-step",
    plan: dict[str, Any] | None = None,
    runnable: bool | None = None,
) -> str:
    from npa.cli.agent_workflow import format_workflow_chat_reply

    return format_workflow_chat_reply(yaml_text, validation, template=template, plan=plan, runnable=runnable)


def format_soperator_deploy() -> str:
    return "\n".join(
        [
            "**Deploy a soperator (Slurm-on-Kubernetes) cluster** from the agent with npa:",
            "1. `POST /api/infra/soperator/validate` with `spec_yaml` (or `spec`) first.",
            "2. `POST /api/infra/soperator/deploy` using the same `npa.soperator/v0.0.1` spec. "
            "Set `dry_run: true` to validate and return the deploy command without mutating infra.",
            "3. `GET /api/infra/soperator/status/{name}` checks the cluster (runs `npa soperator status --output json`).",
            "4. The operator-machine equivalent remains `npa soperator deploy --spec cluster.yaml --output json`.",
            "- Spec: one or more worker pools "
            "(mixed presets ok) and optional per-pool `docker_cache: true` (IO_M3 image cache).",
            "- Preflight quotas: `compute.instance.count`, `compute.instance.non-gpu.vcpu`, "
            "`compute.disk.count`, and `compute.disk.size.network-ssd-io-m3` (GPU on-demand "
            "quota is often 0 -- use `preemptible: true` for GPU pools).",
            "- **workflow toolRef**: `infra.soperator.deploy` (config.soperator_spec = path to the spec).",
            "- GPU workers must be fabric-capable 8-GPU SXM presets; 1-GPU presets can't cluster.",
            "- See the `soperator` skill for post-deploy fixes and worker-registration gotchas.",
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


def format_onboard_solution() -> str:
    registry = os.environ.get("NPA_REGISTRY", "").strip() or "<resolved-from-~/.npa/config.yaml>"
    byof_skill_path = BYOF_ONBOARD_SKILL_PATH
    registry_skill_path = OSS_SOLUTION_REGISTRY_ONBOARD_SKILL_PATH
    return "\n".join(
        [
            "**Yes — chat can onboard a new OSS solution end-to-end.**",
            f"- **BYOF skill:** `{byof_skill_path}`",
            f"- **Registry skill:** `{registry_skill_path}`",
            "- Flow: **contract** → **containerize** (`--base-profile ubuntu`) → **deploy/test** (`--workload container-verify`).",
            "- Registry/catalog readiness additionally requires reading upstream docs, listing native capabilities by family, testing each accepted claim with `--workload solution-smoke` (named JSON artifact), and running the live Nebius pull path.",
            "- Capability families and current onboarded matrices live in the registry skill and `docs/workbench/oss-solution-catalog.md`.",
            "- Sim stacks (LeIsaac RL/datagen): use `--base-profile isaac-lab` per the skill workload table.",
            "- Generic Ubuntu onboarding (replace `<repo-url>` / `<repo-ref>`):",
            "```bash",
            "npa/.venv/bin/python npa/scripts/run_byof_repo.py \\",
            "  --repo-url <repo-url> \\",
            "  --repo-ref <repo-ref> \\",
            "  --base-profile ubuntu \\",
            "  --registry " + registry + " \\",
            "  --workload container-verify \\",
            "  --cleanup",
            "```",
            "- Registry candidate capability smoke: use `--workload solution-smoke` with `--build-command`, `--smoke-command`, `--solution-name`, `--capability-name`, and `--smoke-artifact-name`.",
            "- Build-only smoke (no SkyPilot submit): add `--skip-run`.",
            "- Live verify: `bash npa/scripts/verify_byof_onboarding_live.sh` with `NPA_BYOF_LIVE_PIPELINE=1`.",
            "- Do not mark a solution registry-ready from a Docker build or generic import check alone.",
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


def format_find_artifacts() -> str:
    return "\n".join(
        [
            "**Artifact finder (generic, no workflow allowlist):**",
            "1. Discover runs: `GET /api/artifacts/runs?prefix=&limit=100`",
            "2. Inspect one run: `GET /api/artifacts/run/{run_id}`",
            "3. Load selected artifact: `POST /api/sim-viz/load-artifact` with `s3_uri` or `run_id` + `key`",
            "- Render hints are additive (`rerun`, `video`, `image`, `json`, `text`, `download`).",
            "- Unknown/new file types are still listed and selectable as `download`.",
            "- Use `GET /api/sim-viz/status` after load to confirm `artifact_render`, `artifact_key`, and `rerun_ready`.",
        ]
    )


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
    if intent == "find_artifacts":
        return format_find_artifacts()
    if intent == "sim_assets":
        return format_sim_assets(state)
    if intent == "cameras":
        return format_cameras(state, default_cameras=default_cameras)
    if intent == "infra_backends":
        return format_infra_backends(state)
    if intent == "mk8s_provision":
        return format_mk8s_provision()
    if intent == "live_infra_loop":
        return format_live_infra_loop_guidance()
    if intent == "cosmos_capabilities":
        return format_cosmos_capabilities(tool_refs)
    if intent == "lancedb_capabilities":
        return format_lancedb_capabilities(tool_refs)
    if intent == "component_capabilities":
        return format_component_capabilities(tool_refs)
    if intent == "tools_catalog":
        return format_tools_catalog(tool_refs)
    if intent == "configure_s3":
        return format_configure_s3()
    if intent == "cosmos3":
        return format_cosmos3_setup()
    if intent == "soperator":
        return format_soperator_deploy()
    if intent == "onboard_solution":
        return format_onboard_solution()
    if intent == "load_franka":
        ready = rerun_ready if rerun_ready is not None else bool(_sim_viz(state).get("rerun_ready"))
        return format_load_franka_status(state, rerun_ready=ready, loaded_now=loaded_franka_now)
    if intent in {"create_workflow", "create_vlm_rl_workflow", "create_gate_workflow"}:
        draft = state.get("workflow_draft", {})
        if not isinstance(draft, dict):
            draft = {}
        yaml_text = str(draft.get("yaml") or "").strip()
        if yaml_text:
            validation = draft.get("validation") if isinstance(draft.get("validation"), dict) else {}
            plan = draft.get("plan") if isinstance(draft.get("plan"), dict) else {}
            runnable = bool(draft.get("runnable"))
            if not validation:
                validation = {
                    "ok": True,
                    "status": "valid",
                    "name": str(draft.get("name") or "unnamed"),
                    "states": draft.get("states") or [],
                }
            template = str(draft.get("template") or "")
            if not template:
                template = "two-step" if intent == "create_workflow" else (
                    "vlm-rl-loop" if intent == "create_vlm_rl_workflow" else "token-factory-gate"
                )
            return format_generate_workflow(yaml_text, validation, template=template, plan=plan, runnable=runnable)
        from npa.cli.agent_workflow import generate_workflow_draft

        generated = generate_workflow_draft(intent=intent, user_text="", tool_refs=frozenset(tool_refs))
        validation = generated["validation"] if isinstance(generated.get("validation"), dict) else {"ok": False}
        plan = generated.get("plan") if isinstance(generated.get("plan"), dict) else {}
        runnable = bool(generated.get("runnable"))
        return format_generate_workflow(
            str(generated.get("yaml") or ""),
            validation,
            template=str(generated["template"]),
            plan=plan,
            runnable=runnable,
        )
    if intent == "list_recordings":
        return (
            "**Run history**: use `GET /api/sim-viz/recordings` to list `.rrd` files or "
            "`GET /api/sim-viz/runs` to list run-scoped history.\n"
            "- Click **Load run data** or select a run from the **Known runs** dropdown to switch the Rerun viewer.\n"
            "- Each entry shows run_id, stage, camera, and last-updated timestamp.\n"
            "- The current active run is highlighted in the Run History panel."
        )
    return format_sim2real_status(state, rerun_ready=rerun_ready)


def apis_for_intent(intent: str) -> list[str]:
    return list(INTENT_APIS.get(intent, []))
