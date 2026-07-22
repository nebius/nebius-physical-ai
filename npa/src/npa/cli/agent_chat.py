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
        "drive_sim2real",
        re.compile(
            r"\b(?:drive|orchestrate|automate|auto[- ]?run)\b.{0,80}\b(?:sim\s*[- ]?2\s*[- ]?real|sim2real)\b"
            r"|\bautonomous(?:ly)?\b.{0,80}\b(?:sim\s*[- ]?2\s*[- ]?real|sim2real)\b"
            r"|\b(?:sim\s*[- ]?2\s*[- ]?real|sim2real)\b.{0,60}\b(?:outer\s+loop)\b.{0,40}\b(?:drive|automate|autonomous|self[- ]?driv\w*)\b"
            r"|\b(?:close|drive)\b.{0,40}\b(?:the\s+)?(?:outer\s+)?loop\b.{0,60}\b(?:sim\s*[- ]?2\s*[- ]?real|sim2real)\b",
            re.IGNORECASE,
        ),
    ),
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
        "create_loop_gate_workflow",
        re.compile(
            r"\b(?:create|generate|build|make|draft|compose|write)\b"
            r".{0,120}\b(?:loop[_\s-]?gate|decision[_\s-]?gate)\b"
            r"|\b(?:loop[_\s-]?gate|decision[_\s-]?gate)\b.{0,80}\b(?:workflow|yaml|spec)\b"
            r"|\b(?:create|generate|build|make|draft)\b.{0,80}\b(?:sim2real|sim\s*[- ]?2\s*[- ]?real)\b"
            r".{0,80}\b(?:loop|gate|decision)\b.{0,80}\b(?:workflow|yaml|spec)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "create_rl_policy_workflow",
        re.compile(
            r"\b(?:create|generate|build|make|draft|compose|write)\b"
            r".{0,120}\b(?:rl[_\s-]?policy|policy[_\s-]?train(?:ing)?|isaac[_\s-]?lab\s+rl)\b"
            r"|\b(?:rl[_\s-]?policy|policy[_\s-]?train(?:ing)?)\b.{0,80}\b(?:workflow|yaml|spec|success\s+gate)\b"
            r"|\b(?:create|generate|draft)\b.{0,80}\b(?:rl|reinforcement\s+learning)\b"
            r".{0,80}\b(?:policy|train(?:ing)?)\b.{0,80}\b(?:workflow|yaml|spec)\b",
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
        "workflow_execute_guidance",
        re.compile(
            r"\b(?:how|where|can)\b.{0,80}\b(?:execute|run|submit)\b.{0,80}\b(?:workflow|yaml|spec|npa\.workflow)\b"
            r"|\b(?:execute|run)\b.{0,80}\b(?:workflow|yaml|spec)\b.{0,80}\b(?:for\s+real|on\s+(?:k8s|kubernetes|cluster)|with\s+sky|via\s+cli)\b"
            r"|\b(?:run-spec|--execute|scheduler[- ]?plan)\b"
            r"|\b(?:difference|diff|vs)\b.{0,80}\b(?:plan[- ]?only|submit|execute)\b",
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
            r"|\bonboard\s+https?://"
            r"|\bonboard\b.{0,160}\b(?:ubuntu|container|registry|deploy\s+smoke)\b"
            r"|\b(?:byof|bring your own fork|custom fork)\b.{0,140}\b(?:workflow|image|container|registry|infra|run)\b"
            r"|\b(?:github\.com|gitlab\.com)/[^\s]+.{0,140}\b(?:container|registry|ubuntu|smoke|deploy)\b"
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
        "sonic_capabilities",
        re.compile(
            r"\bsonic\b.{0,120}\b(?:support|supports|capabilit(?:y|ies)|expose|offers|train|eval|export)\b"
            r"|\b(?:support|supports|capabilit(?:y|ies)|expose|offers)\b.{0,120}\bsonic\b"
            r"|\b(?:can|could)\b.{0,80}\bsonic\b.{0,120}\b(?:do|run|train|eval|export)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "lerobot_capabilities",
        re.compile(
            r"\blerobot\b.{0,120}\b(?:support|supports|capabilit(?:y|ies)|expose|offers|train|eval)\b"
            r"|\b(?:support|supports|capabilit(?:y|ies)|expose|offers)\b.{0,120}\blerobot\b"
            r"|\b(?:can|could)\b.{0,80}\blerobot\b.{0,120}\b(?:do|run|train|eval)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "groot_capabilities",
        re.compile(
            r"\bgroot\b.{0,120}\b(?:support|supports|capabilit(?:y|ies)|expose|offers|train|eval|infer)\b"
            r"|\b(?:support|supports|capabilit(?:y|ies)|expose|offers)\b.{0,120}\bgroot\b"
            r"|\b(?:can|could)\b.{0,80}\bgroot\b.{0,120}\b(?:do|run|train|eval|infer)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "genesis_capabilities",
        re.compile(
            r"\bgenesis\b.{0,120}\b(?:support|supports|capabilit(?:y|ies)|expose|offers|sim|simulate)\b"
            r"|\b(?:support|supports|capabilit(?:y|ies)|expose|offers)\b.{0,120}\bgenesis\b"
            r"|\b(?:can|could)\b.{0,80}\bgenesis\b.{0,120}\b(?:do|run|simulate)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "mjlab_capabilities",
        re.compile(
            r"\bmjlab\b.{0,120}\b(?:support|supports|capabilit(?:y|ies)|expose|offers|eval|locomotion)\b"
            r"|\b(?:support|supports|capabilit(?:y|ies)|expose|offers)\b.{0,120}\bmjlab\b"
            r"|\b(?:can|could)\b.{0,80}\bmjlab\b.{0,120}\b(?:do|run|eval)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "isaac_lab_capabilities",
        re.compile(
            r"\bisaac(?:\s|-)?lab\b.{0,120}\b(?:support|supports|capabilit(?:y|ies)|expose|offers|train|eval|rl)\b"
            r"|\b(?:support|supports|capabilit(?:y|ies)|expose|offers)\b.{0,120}\bisaac(?:\s|-)?lab\b"
            r"|\b(?:can|could)\b.{0,80}\bisaac(?:\s|-)?lab\b.{0,120}\b(?:do|run|train|eval)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "component_capabilities",
        re.compile(
            r"\b(?:component|tool|workbench)\b.{0,120}\b(?:support|supports|capabilit(?:y|ies)|expose|offers)\b"
            r"|\bwhat\b.{0,80}\b(?:does|can)\b.{0,80}\b(?:cosmos|lancedb|sonic|isaac(?:\s|-)?lab|lerobot|groot|token(?:\s|-)?factory|genesis|mjlab)\b.{0,80}\b(?:support|do|expose)\b",
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
    "drive_sim2real": [
        "agent/sim2real/drive",
        "workflows/sim2real/submit",
        "workflows/sim2real/status",
        "workflows/sim2real/runs/{run_id}",
    ],
    "start_sim2real": ["workflows/sim2real/submit"],
    "watch_sim": ["sim-viz/status", "sim-viz/rrd", "sim-viz/rrd-blob", "workflows/sim2real/status"],
    "find_artifacts": ["artifacts/runs", "artifacts/run/{run_id}", "sim-viz/load-artifact", "sim-viz/status"],
    "create_workflow": ["workflows/draft", "workflows/validate", "workflows/plan"],
    "create_vlm_rl_workflow": ["workflows/draft", "workflows/validate", "workflows/plan"],
    "create_gate_workflow": ["workflows/draft", "workflows/validate", "workflows/plan"],
    "create_loop_gate_workflow": ["workflows/draft", "workflows/validate", "workflows/plan"],
    "create_rl_policy_workflow": ["workflows/draft", "workflows/validate", "workflows/plan"],
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
    "sonic_capabilities": ["tools"],
    "lerobot_capabilities": ["tools"],
    "groot_capabilities": ["tools"],
    "genesis_capabilities": ["tools"],
    "mjlab_capabilities": ["tools"],
    "isaac_lab_capabilities": ["tools"],
    "component_capabilities": ["tools"],
    "tools_catalog": ["tools"],
    "configure_s3": ["tools"],
    "cosmos3": ["tools"],
    "soperator": ["infra/soperator/validate", "infra/soperator/deploy", "infra/soperator/status/{name}", "tools"],
    "load_franka": ["sim-viz/load-franka-demo", "sim-viz/status"],
    "workflow_execute_guidance": ["workflows/validate", "workflows/plan", "workflows/submit", "tools"],
}

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
    lines.append("- Open the **Rerun** tab to inspect camera entities in the embedded viewer.")
    return "\n".join(lines)


def format_tools_catalog(tool_refs: list[str], *, sample_size: int = 16) -> str:
    count = len(tool_refs)
    groups: dict[str, list[str]] = {}
    for ref in tool_refs:
        token = str(ref or "")
        if token.startswith("workbench."):
            parts = token.split(".")
            family = parts[1] if len(parts) > 1 else "workbench"
        elif token.startswith("infra."):
            family = "infra"
        else:
            family = "other"
        groups.setdefault(family, []).append(token)
    lines = [
        f"**Workbench tool catalog** ({count} toolRefs — same surface as `npa.workflow` YAML `toolRef`):",
        "- Chat can **draft / validate / plan** YAML for these tools.",
        "- Agent **Submit YAML** returns a **scheduler plan only** (not SkyPilot/K8s execute).",
        "- Real execution: `npa workbench workflow run-spec <spec.yaml> --execute` on the operator machine.",
        "",
        "**Families:**",
    ]
    for family in sorted(groups):
        refs = sorted(groups[family])
        preview = ", ".join(f"`{ref}`" for ref in refs[:3])
        more = f" (+{len(refs) - 3})" if len(refs) > 3 else ""
        lines.append(f"- **{family}** ({len(refs)}): {preview}{more}")
    flat_sample = tool_refs[:sample_size]
    if flat_sample:
        lines.append("")
        lines.append("**Sample toolRefs:**")
        for ref in flat_sample:
            lines.append(f"- `{ref}`")
        if count > sample_size:
            lines.append(f"- … and **{count - sample_size}** more via `GET /api/tools`")
    lines.append("- Invoke tools via `npa workbench <tool> …` or npa.workflow specs on your operator machine.")
    return "\n".join(lines)


def _image_for_tool(tool: str) -> str:
    from npa.deploy.images import primary_container_registry

    registry = primary_container_registry()
    image_name, tag = _DEFAULT_TOOL_IMAGE_TAGS.get(tool, (f"npa-{tool}", "<tag>"))
    return f"{registry.rstrip('/')}/{image_name}:{tag}"


def _format_tool_family_capabilities(name: str, tool_refs: list[str], *, prefixes: tuple[str, ...], bullets: list[str]) -> str:
    matched = [
        ref
        for ref in tool_refs
        if any(token in str(ref).lower() for token in prefixes)
    ]
    sample = ", ".join(f"`{ref}`" for ref in sorted(matched)[:5]) if matched else "_(none registered in this agent catalog)_"
    lines = [f"**{name} capabilities** (CLI + npa.workflow `toolRef`):"]
    lines.extend(f"- {bullet}" for bullet in bullets)
    lines.append(f"- **Catalog matches**: {sample}")
    lines.append("- Draft a workflow in chat, then execute on the operator machine with `npa workbench workflow run-spec … --execute`.")
    return "\n".join(lines)


def format_cosmos_capabilities(tool_refs: list[str]) -> str:
    cosmos_image = _image_for_tool("cosmos")
    return _format_tool_family_capabilities(
        "Cosmos",
        tool_refs,
        prefixes=("cosmos", "token_factory"),
        bullets=[
            "**Inference**: Cosmos3 text-to-image workflow (`cosmos3-text-to-image-inference.yaml`).",
            "**Setup + model staging**: `npa workbench cosmos check|fetch`.",
            "**Fine-tuning / post-training**: `npa workbench cosmos train` (serverless + runtime options).",
            "**Pipeline integration**: Cosmos augment via `workbench.cosmos2.transfer` and Token Factory reasoning paths.",
            f"**Registry image default**: `{cosmos_image}` (override via `NPA_REGISTRY` if needed).",
            "Use run-scoped S3 URIs for artifacts and keep credentials in `~/.npa/credentials.yaml`.",
        ],
    )


def format_lancedb_capabilities(tool_refs: list[str]) -> str:
    lancedb_image = _image_for_tool("lancedb")
    return _format_tool_family_capabilities(
        "LanceDB",
        tool_refs,
        prefixes=("lancedb",),
        bullets=[
            "**Data ingest**: BDD100K import into run-scoped Lance tables.",
            "**Feature backfill**: CPU + GPU UDF backfills (including CLIP embeddings).",
            "**Dataset shaping**: materialized view creation for failure-mode slices.",
            "**Serving path**: endpoint-backed execution for workflows and tooling.",
            f"**Registry image default**: `{lancedb_image}` (use your real registry, never placeholders).",
            "Keep table/URI names in config; avoid embedding project-specific constants in workflow states.",
        ],
    )


def format_sonic_capabilities(tool_refs: list[str]) -> str:
    return _format_tool_family_capabilities(
        "SONIC",
        tool_refs,
        prefixes=("sonic",),
        bullets=[
            "**Train / eval / export** locomotion policies via `npa workbench sonic …`.",
            "**Workflow toolRefs**: `workbench.sonic.train`, `workbench.sonic.eval`, `workbench.sonic.export`.",
            "Use project `gpu_profile` + SkyPilot config on the operator machine for GPU jobs.",
        ],
    )


def format_lerobot_capabilities(tool_refs: list[str]) -> str:
    return _format_tool_family_capabilities(
        "LeRobot",
        tool_refs,
        prefixes=("lerobot",),
        bullets=[
            "**Eval / train** policies via `npa workbench lerobot …`.",
            "**Workflow toolRef**: `workbench.lerobot.eval` (plus serverless train flows).",
            "Keep Hugging Face tokens in `~/.npa/credentials.yaml`.",
        ],
    )


def format_groot_capabilities(tool_refs: list[str]) -> str:
    return _format_tool_family_capabilities(
        "GR00T",
        tool_refs,
        prefixes=("groot",),
        bullets=[
            "**Train / eval / inference** via `npa workbench groot …`.",
            "Use workbench images from your Nebius registry; avoid hardcoded registry IDs.",
        ],
    )


def format_genesis_capabilities(tool_refs: list[str]) -> str:
    return _format_tool_family_capabilities(
        "Genesis",
        tool_refs,
        prefixes=("genesis",),
        bullets=[
            "**Simulation backend** selectable in the agent Rerun Selection panel (`sim_backend=genesis`).",
            "**CLI**: `npa workbench genesis …` for container smokes and workflows.",
        ],
    )


def format_mjlab_capabilities(tool_refs: list[str]) -> str:
    return _format_tool_family_capabilities(
        "MJLab",
        tool_refs,
        prefixes=("mjlab", "retargeting"),
        bullets=[
            "**Locomotion eval** and SONIC checkpoint scoring via `npa workbench mjlab …`.",
            "**Workflow toolRef**: `workbench.mjlab.eval` (+ retargeting helpers).",
        ],
    )


def format_isaac_lab_capabilities(tool_refs: list[str]) -> str:
    isaac_image = _image_for_tool("isaac-lab")
    return _format_tool_family_capabilities(
        "Isaac Lab",
        tool_refs,
        prefixes=("isaac", "rl.policy", "byof"),
        bullets=[
            "**RL train/eval** building blocks for simulation pipelines.",
            f"**Registry image default**: `{isaac_image}`.",
            "**Workflow toolRefs**: `workbench.rl.policy_train`, `workbench.rl.evaluate_policy`, `workbench.byof.repo`.",
            "Ask chat to draft an `rl-policy-success` or BYOF Isaac Lab workflow YAML.",
        ],
    )


def format_component_capabilities(tool_refs: list[str]) -> str:
    return "\n".join(
        [
            "**Workbench component capabilities** (same building blocks as CLI + YAML):",
            "- **Cosmos / Token Factory**: setup, inference, finetune, VLM/reasoning gates.",
            "- **LanceDB**: ingest, backfill, views, query workflows.",
            "- **Isaac Lab / RL / BYOF**: train/eval policy + OSS container onboarding.",
            "- **SONIC / MJLab / Retargeting**: locomotion train/eval/export.",
            "- **LeRobot / GR00T / Genesis**: policy and sim backends.",
            "- Ask by name (`SONIC capabilities`, `what can lerobot do?`) for catalog-backed answers.",
            f"- **Current toolRef count**: `{len(tool_refs)}`",
            "- Chat drafts YAML; operator CLI executes with `run-spec --execute` / SkyPilot submit.",
        ]
    )


def format_workflow_execute_guidance() -> str:
    return "\n".join(
        [
            "**Workflow capability map (chat vs CLI/YAML):**",
            "| Step | Agent chat / UI | Operator CLI / SDK |",
            "|---|---|---|",
            "| Draft YAML | Yes (`create_*` intents + Workflow panel) | Yes (`author-npa-workflow` / hand-edit) |",
            "| Validate | Yes (`POST /api/workflows/validate`) | `npa workbench workflow validate-spec` |",
            "| Plan | Yes (`POST /api/workflows/plan`) | `npa workbench workflow plan-spec` |",
            "| Scheduler plan submit | Yes (`POST /api/workflows/submit` = **plan-only**) | `run-spec --plan-only --scheduler-plan` |",
            "| Execute tool steps on K8s | No (not from agent chat) | `run-spec --execute` / SkyPilot `submit` |",
            "| Direct `npa workbench <tool>` | Guidance only | Full CLI surface |",
            "",
            "**Operator execute example:**",
            "```bash",
            "npa/.venv/bin/npa workbench workflow validate-spec /tmp/spec.yaml --json",
            "npa/.venv/bin/npa workbench workflow plan-spec /tmp/spec.yaml --run-id agent-run --json",
            "npa/.venv/bin/npa workbench workflow run-spec /tmp/spec.yaml --execute --json",
            "```",
            "- Use chat to author/validate YAML, then run the CLI on the operator/dev VM for real workloads.",
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
            "**BYOF / OSS onboarding (operator CLI — chat guides, does not execute builds):**",
            f"- **BYOF skill:** `{byof_skill_path}`",
            f"- **Registry skill:** `{registry_skill_path}`",
            "- **Ladder:** `docs/architecture/oss-onboarding-ladder.md` (Tier 0 BYOF → Tier 1 workflow → Tier 2 first-class tool).",
            "- Flow: **contract** → **containerize** (`--base-profile ubuntu`) → **deploy/test** (`--workload container-verify`).",
            "- Registry/catalog readiness additionally requires reading upstream docs, listing that solution's native capabilities (upstream names), testing each accepted claim with `--workload solution-smoke` (named JSON artifact), and running the live Nebius pull path.",
            "- Per-solution capability matrices live in the registry skill and `docs/workbench/oss-solution-catalog.md`.",
            "- Sim stacks (LeIsaac RL/datagen): use `--base-profile isaac-lab` per the skill workload table.",
            "- Generic Ubuntu onboarding (replace `<repo-url>` / `<repo-ref>`):",
            "```bash",
            "npa workbench byof run \\",
            "  --repo-url <repo-url> \\",
            "  --repo-ref <repo-ref> \\",
            "  --base-profile ubuntu \\",
            "  --registry " + registry + " \\",
            "  --workload container-verify \\",
            "  --cleanup",
            "```",
            "- Equivalent script still works: `npa/scripts/run_byof_repo.py` (same flags).",
            "- Registry candidate capability smoke: use `--workload solution-smoke` with `--build-command`, `--smoke-command`, `--solution-name`, `--capability-name`, and `--smoke-artifact-name`.",
            "- Build-only smoke (no SkyPilot submit): add `--skip-run`.",
            "- Live verify: `bash npa/scripts/verify_byof_onboarding_live.sh` with `NPA_BYOF_LIVE_PIPELINE=1`.",
            "- Chat can also draft a BYOF `npa.workflow` YAML; run it with `npa workbench workflow run-spec … --execute` on the operator machine.",
            "- After container-verify: author a solution workflow (`author-npa-workflow`) or promote to a first-class tool (`contributor-context.md`).",
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


def format_drive_sim2real_guidance(state: dict[str, Any]) -> str:
    """Grounded guidance for the autonomous Sim2Real drive (confirmation-gated)."""
    sim_viz = _sim_viz(state)
    run_id = str(sim_viz.get("run_id") or "").strip() or "none"
    stage = str(sim_viz.get("stage") or "idle").strip() or "idle"
    return "\n".join(
        [
            "**Autonomous Sim2Real drive** (agent-orchestrated outer loop):",
            f"- **active_run_id**: `{run_id}`  **stage**: `{stage}`",
            "- The agent composes: launch sim → run eval → read gate metrics → "
            "diagnose failure mode → adjust config → re-run.",
            "- Each iteration surfaces a **promote_checkpoint** / **loop_back** decision "
            "with the reason (`success_rate` vs `threshold`).",
            "- **GPU-spending** — every launch passes through a confirmation gate. "
            "The drive is *proposed* first; re-send with the returned confirmation token to execute.",
            "- A stage is marked complete only when `workflows/sim2real/status` / "
            "`runs/{run_id}` confirms it — no fabricated run data.",
            "- Drive it: `POST /api/agent/sim2real/drive` with "
            "`{ \"config\": {\"run_id\": ..., \"threshold\": 0.8, \"max_iterations\": 3}, \"confirm_token\": ... }`.",
            "- Read-only observation stays on `GET /api/workflows/sim2real/status`.",
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
    if intent == "drive_sim2real":
        return format_drive_sim2real_guidance(state)
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
    if intent == "sonic_capabilities":
        return format_sonic_capabilities(tool_refs)
    if intent == "lerobot_capabilities":
        return format_lerobot_capabilities(tool_refs)
    if intent == "groot_capabilities":
        return format_groot_capabilities(tool_refs)
    if intent == "genesis_capabilities":
        return format_genesis_capabilities(tool_refs)
    if intent == "mjlab_capabilities":
        return format_mjlab_capabilities(tool_refs)
    if intent == "isaac_lab_capabilities":
        return format_isaac_lab_capabilities(tool_refs)
    if intent == "component_capabilities":
        return format_component_capabilities(tool_refs)
    if intent == "tools_catalog":
        return format_tools_catalog(tool_refs)
    if intent == "workflow_execute_guidance":
        return format_workflow_execute_guidance()
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
    if intent in {
        "create_workflow",
        "create_vlm_rl_workflow",
        "create_gate_workflow",
        "create_loop_gate_workflow",
        "create_rl_policy_workflow",
    }:
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
                if intent == "create_vlm_rl_workflow":
                    template = "vlm-rl-loop"
                elif intent == "create_gate_workflow":
                    template = "token-factory-gate"
                elif intent == "create_loop_gate_workflow":
                    template = "loop-gate"
                elif intent == "create_rl_policy_workflow":
                    template = "rl-policy-success"
                else:
                    template = "two-step"
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
        return format_list_recordings(state)
    return format_sim2real_status(state, rerun_ready=rerun_ready)


def format_list_recordings(state: dict[str, Any]) -> str:
    sim_viz = _sim_viz(state)
    runs = state.get("sim_viz_runs")
    recordings = state.get("sim_viz_recordings")
    available = sim_viz.get("available_run_ids") if isinstance(sim_viz.get("available_run_ids"), list) else []
    lines = ["**Run / recording history** (grounded):"]
    active = str(sim_viz.get("active_run_id") or sim_viz.get("run_id") or "").strip()
    if active:
        lines.append(f"- **active_run_id**: `{active}`")
    if isinstance(runs, list) and runs:
        lines.append(f"- **sim-viz/runs**: `{len(runs)}` entries")
        for row in runs[:8]:
            if not isinstance(row, dict):
                continue
            run_id = str(row.get("run_id") or "").strip()
            if not run_id:
                continue
            stage = str(row.get("stage") or "")
            camera = str(row.get("camera") or "")
            marker = " ← active" if run_id == active else ""
            lines.append(f"  - `{run_id}` stage=`{stage}` camera=`{camera}`{marker}")
    elif available:
        lines.append(f"- **available_run_ids**: `{len(available)}`")
        for run_id in [str(item) for item in available[:8] if str(item).strip()]:
            marker = " ← active" if run_id == active else ""
            lines.append(f"  - `{run_id}`{marker}")
    else:
        lines.append("- No run history in session yet — open the **Rerun** tab and use **Runs & artifacts** → **Discover runs**.")
    if isinstance(recordings, list) and recordings:
        lines.append(f"- **sim-viz/recordings**: `{len(recordings)}` `.rrd` files")
        for row in recordings[:6]:
            if isinstance(row, dict):
                name = str(row.get("name") or row.get("path") or row.get("uri") or "").strip()
                if name:
                    lines.append(f"  - `{name}`")
            else:
                text = str(row or "").strip()
                if text:
                    lines.append(f"  - `{text}`")
    lines.extend(
        [
            "- Switch viewer: paste a run id → **Load run**, or select from **Runs & artifacts** (latest first).",
            "- Prefer `GET /api/sim-viz/runs` + `GET /api/sim-viz/recordings` over guessing run ids.",
        ]
    )
    return "\n".join(lines)


def apis_for_intent(intent: str) -> list[str]:
    return list(INTENT_APIS.get(intent, []))
