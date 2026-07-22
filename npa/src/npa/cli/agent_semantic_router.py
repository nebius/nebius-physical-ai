"""Semantic intent fallthrough for the NPA agent VM backend.

The deterministic regex router (`agent_chat.match_chat_intent`) stays the fast,
zero-token path for high-frequency intents. This module is the *fallthrough*
that only runs when that router returns ``None``: it maps a paraphrase the regex
missed onto a known grounded intent (or onto the Phase-B action loop) using, in
order of preference:

1. a cheap keyword pre-filter (0 tokens),
2. a short-lived cache of prior classifications (0 tokens),
3. one cheap-tier structured model call as a last resort.

This lets the agent retire the brittle regex tail (e.g. the ~40-alternation
``watch_sim`` monster) without regressing the parity-guaranteed grounded intents:
those still match in ``match_chat_intent`` at 0 tokens and never reach here.

All model access is via an injected ``model_call`` so tests spend zero tokens.
The module is embedded verbatim into the agent VM backend by ``agent.py``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

MODE_INTENT = "intent"
MODE_ACTION = "action"
MODE_NONE = "none"

SOURCE_KEYWORD = "keyword"
SOURCE_CACHE = "cache"
SOURCE_MODEL = "model"
SOURCE_NONE = "none"

# Keyword hint groups: each entry is (intent, required_any, surface_any). A turn
# matches when it contains any trigger word AND (if given) any surface word. Only
# intents also present in ``known_intents`` are ever returned, so this map can
# stay broad without inventing intents the backend cannot ground.
_KEYWORD_HINTS: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
    (
        "watch_sim",
        ("watch", "monitor", "keep an eye", "keep tabs", "follow along", "tail", "observe", "live view"),
        ("sim", "simulation", "rerun", "timeline", "rollout", "run", "viewer"),
    ),
    (
        "sim2real_status",
        ("status", "how is", "how's", "progress", "where are we", "state of"),
        ("sim", "sim2real", "pipeline", "run", "workflow", "stage"),
    ),
    (
        "find_artifacts",
        ("find", "browse", "discover", "what can i view", "look at", "show me", "list"),
        ("artifact", "artifacts", "output", "outputs", "recording", "report", "results"),
    ),
    (
        "list_recordings",
        ("history", "past", "previous", "earlier", "prior"),
        ("run", "runs", "recording", "recordings", "rrd"),
    ),
    (
        "tools_catalog",
        ("what can you do", "what tools", "capabilities overview", "list tools", "available tools", "toolref"),
        (),
    ),
    (
        "cosmos_capabilities",
        ("cosmos",),
        ("do", "support", "capab", "run", "train", "infer", "finetune", "generate"),
    ),
    (
        "lancedb_capabilities",
        ("lancedb", "lance db"),
        ("do", "support", "capab", "query", "import", "backfill", "table"),
    ),
    (
        "sonic_capabilities",
        ("sonic",),
        ("do", "support", "capab", "train", "eval", "export", "locomotion"),
    ),
    (
        "lerobot_capabilities",
        ("lerobot", "le robot"),
        ("do", "support", "capab", "train", "eval", "policy"),
    ),
    (
        "groot_capabilities",
        ("groot", "gr00t"),
        ("do", "support", "capab", "train", "eval", "infer"),
    ),
    (
        "genesis_capabilities",
        ("genesis",),
        ("do", "support", "capab", "sim", "simulate"),
    ),
    (
        "isaac_lab_capabilities",
        ("isaac lab", "isaac-lab", "isaaclab"),
        ("do", "support", "capab", "train", "eval", "rl"),
    ),
    (
        "configure_s3",
        ("s3", "bucket", "object storage", "storage endpoint"),
        ("configure", "set up", "setup", "credentials", "connect"),
    ),
    (
        "drive_sim2real",
        ("drive", "autonomous", "autonomously", "orchestrate", "self-driving", "close the loop"),
        ("sim", "sim2real", "loop", "pipeline"),
    ),
]


def _normalize(text: str) -> str:
    lowered = str(text or "").lower().replace("\n", " ")
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _keyword_intent(lowered: str, known_intents: frozenset[str]) -> str | None:
    for intent, triggers, surfaces in _KEYWORD_HINTS:
        if intent not in known_intents:
            continue
        if not any(trigger in lowered for trigger in triggers):
            continue
        if surfaces and not any(surface in lowered for surface in surfaces):
            continue
        return intent
    return None


def _extract_json(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    for candidate in ([fenced.group(1)] if fenced else []) + [raw]:
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    brace = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace:
        try:
            parsed = json.loads(brace.group(0))
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, TypeError):
            return None
    return None


def _sem_message_content(data: Any) -> str:
    # Uniquely named so the shared embedded backend namespace never clobbers (or
    # is clobbered by) agent_actions._message_content, which supports list-form
    # content the classifier does not need.
    if not isinstance(data, dict):
        return ""
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return ""
    content = message.get("content") if isinstance(message, dict) else None
    return content if isinstance(content, str) else ""


def _sem_tokens_from(data: Any) -> int:
    if not isinstance(data, dict):
        return 0
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return 0
    total = usage.get("total_tokens")
    if isinstance(total, bool) or not isinstance(total, (int, float)):
        return 0
    return int(total)


def _model_messages(lowered: str, known_intents: frozenset[str]) -> list[dict[str, str]]:
    intent_list = ", ".join(sorted(known_intents))
    system = (
        "You are the NPA workbench intent classifier. Map the operator turn to "
        "exactly one known intent, or to 'action' when it needs a multi-step tool "
        "loop, or to 'none' when it is an open question.\n"
        "Respond with a SINGLE JSON object: "
        '{\"intent\": \"<one_of_known|action|none>\", \"confidence\": <0..1>}.\n'
        f"Known intents: {intent_list}."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": lowered},
    ]


def _none_result(source: str = SOURCE_NONE) -> dict[str, Any]:
    return {"intent": None, "mode": MODE_NONE, "confidence": 0.0, "tokens": 0, "source": source}


def classify_intent_semantic(
    user_text: str,
    *,
    known_intents: frozenset[str] | set[str] | list[str],
    model_call: Callable[..., Any] | None = None,
    cache: dict[str, dict[str, Any]] | None = None,
    use_model: bool = True,
    tier: str = "cheap",
    min_confidence: float = 0.4,
) -> dict[str, Any]:
    """Classify a regex-missed turn into a known intent / action / none.

    Returns ``{intent, mode, confidence, tokens, source}``. Keyword and cache
    hits cost 0 tokens; only a genuine miss spends one cheap-tier model call.
    Never raises: a model/parse failure degrades to a ``none`` result so the
    caller falls through to its existing cheap-LLM path unchanged.
    """
    known = frozenset(str(i) for i in known_intents)
    lowered = _normalize(user_text)
    if not lowered:
        return _none_result()

    keyword = _keyword_intent(lowered, known)
    if keyword:
        return {
            "intent": keyword,
            "mode": MODE_INTENT,
            "confidence": 0.6,
            "tokens": 0,
            "source": SOURCE_KEYWORD,
        }

    if cache is not None and lowered in cache:
        cached = dict(cache[lowered])
        cached["tokens"] = 0
        cached["source"] = SOURCE_CACHE
        return cached

    if not use_model or model_call is None:
        return _none_result()

    try:
        data = model_call(_model_messages(lowered, known), tier=tier)
    except Exception:  # noqa: BLE001 - degrade to none on any planner failure
        return _none_result()
    tokens = _sem_tokens_from(data)
    parsed = _extract_json(_sem_message_content(data))
    if not isinstance(parsed, dict):
        result = _none_result(SOURCE_MODEL)
        result["tokens"] = tokens
        return result

    raw_intent = str(parsed.get("intent") or "").strip()
    try:
        confidence = float(parsed.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.5
    # A non-finite or below-threshold confidence must not misroute a turn.
    if not (confidence == confidence) or confidence in (float("inf"), float("-inf")):
        confidence = 0.0
    if raw_intent in known and confidence < float(min_confidence):
        result = _none_result(SOURCE_MODEL)
        result["tokens"] = tokens
        result["confidence"] = confidence
        return result
    if raw_intent in known:
        mode = MODE_INTENT
        intent: str | None = raw_intent
    elif raw_intent == MODE_ACTION:
        mode = MODE_ACTION
        intent = None
    else:
        mode = MODE_NONE
        intent = None
    result = {
        "intent": intent,
        "mode": mode,
        "confidence": confidence,
        "tokens": tokens,
        "source": SOURCE_MODEL,
    }
    if cache is not None:
        cache[lowered] = {
            "intent": intent,
            "mode": mode,
            "confidence": confidence,
        }
    return result
