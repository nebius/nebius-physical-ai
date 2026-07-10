"""Cost-aware model routing for the NPA agent chat backend.

The agent answers most operator turns from the deterministic intent router in
``agent_chat`` with zero model calls. When a turn *does* fall through to Nebius
Token Factory, this module keeps the request cheap and appropriately sized:

- classify the turn into a cost tier
  (``cheap`` / ``standard`` / ``reasoning`` / ``vision``),
- order the configured model ladder so the cheapest adequate model is tried
  first and expensive reasoners are reserved for turns that need them,
- select the Token Factory ``-fast`` flavor only for interactive turns that
  have a fast variant (identical output, lower latency, higher price),
- disable hidden reasoning traces on cheap tiers so we do not pay for tokens we
  immediately discard,
- enforce an input-size guardrail so one oversized paste cannot blow the token
  budget,
- summarize per-turn token ``usage`` so cost is observable.

Every function here is pure and side-effect free, so it unit-tests without any
network access. The module source is embedded verbatim into the agent VM
backend by ``agent.py`` (same mechanism as ``agent_chat``), so it must not
import anything that is unavailable on the deployed VM.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

# ── Token Factory model tiers (cheapest-capable first) ───────────────────────
# These are model *families*; the Fast flavor is applied separately by
# ``flavor_variants`` / ``build_model_ladder``.
CHEAP_MODEL = "Qwen/Qwen3-32B"
STANDARD_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
REASONING_MODEL = "nvidia/Cosmos3-Super-Reasoner"
VISION_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"

TIER_CHEAP = "cheap"
TIER_STANDARD = "standard"
TIER_REASONING = "reasoning"
TIER_VISION = "vision"

# Preferred concrete-model order per tier. The resilience ladder falls through
# this list on transient/availability errors, so each tier ends in a broadly
# available fallback.
TIER_MODELS: dict[str, tuple[str, ...]] = {
    TIER_CHEAP: (CHEAP_MODEL, STANDARD_MODEL),
    TIER_STANDARD: (STANDARD_MODEL, CHEAP_MODEL),
    TIER_REASONING: (REASONING_MODEL, STANDARD_MODEL),
    TIER_VISION: (VISION_MODEL, REASONING_MODEL),
}

# Models with a known Token Factory ``-fast`` flavor. Appending ``-fast`` to a
# model without a fast flavor would 404, so only these are expanded.
FAST_CAPABLE = frozenset({CHEAP_MODEL, STANDARD_MODEL})

# Guardrail: cap raw user input so one oversized paste cannot dominate cost.
MAX_INPUT_CHARS = 24000

# Heuristics that justify escalating a free-form turn to the reasoning tier.
_REASONING_RE = re.compile(
    r"\b(?:why|how\s+come|explain|reasoning|reason\s+about|analyz|diagnos|"
    r"debug|root\s*cause|trade[- ]?off|compare|contrast|design|architect|"
    r"plan\s+out|step[- ]by[- ]step|optimi[sz]e|prove|derive|physics|dynamics|"
    r"kinematics|policy\s+collapse|failure\s+mode)\b",
    re.IGNORECASE,
)

# Turns longer than this (chars) are treated as at least ``standard`` tier.
_STANDARD_LENGTH = 800


def has_image_content(messages: Sequence[Any] | None) -> bool:
    """Return True when any message carries image content parts (vision turn)."""
    if not messages:
        return False
    for item in messages:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and str(part.get("type", "")).startswith("image"):
                    return True
        elif isinstance(content, str) and "data:image/" in content:
            return True
    return False


def classify_tier(
    user_text: str,
    *,
    intent: str | None = None,
    messages: Sequence[Any] | None = None,
) -> str:
    """Classify a free-form turn into the cheapest adequate cost tier.

    Grounded intents never reach the model, so this only decides *how much*
    model to spend on genuine open questions: cheap by default, escalating to
    standard for longer/compound turns, reasoning for analytical requests, and
    vision when image content is present.
    """
    if has_image_content(messages):
        return TIER_VISION
    text = str(user_text or "").strip()
    if not text:
        return TIER_CHEAP
    if _REASONING_RE.search(text):
        return TIER_REASONING
    if len(text) >= _STANDARD_LENGTH or text.count("?") >= 3:
        return TIER_STANDARD
    return TIER_CHEAP


def flavor_variants(model: str, *, interactive: bool) -> list[str]:
    """Return the ordered flavor variants to try for a base model.

    Interactive turns prefer the ``-fast`` flavor (sub-second latency) and fall
    back to the base flavor; non-interactive or non-fast-capable models use the
    base flavor only.
    """
    base = str(model or "").strip()
    if not base:
        return []
    if interactive and base in FAST_CAPABLE and not base.endswith("-fast"):
        return [f"{base}-fast", base]
    return [base]


def build_model_ladder(
    tier: str,
    configured: Iterable[str] | None,
    *,
    interactive: bool = True,
    requested_model: str = "",
    allow_tier_defaults: bool = True,
) -> list[str]:
    """Build the ordered list of concrete models to try for a turn.

    Order of precedence:
      1. an explicit ``requested_model`` (user override) always first,
      2. the tier-preferred models,
      3. any remaining configured models as further fallback.

    When ``allow_tier_defaults`` is False (operator set an explicit model
    allowlist via ``NPA_AGENT_LLM_MODELS``), tier preferences are only honored
    where they intersect the allowlist, so we never call a disallowed model.
    Each base model is expanded into its flavor variants.
    """
    configured_list = [str(m).strip() for m in (configured or []) if str(m).strip()]
    tier_prefs = list(TIER_MODELS.get(tier, TIER_MODELS[TIER_STANDARD]))

    bases: list[str] = []

    def _add(model: str) -> None:
        value = str(model or "").strip()
        if value and value not in bases:
            bases.append(value)

    if requested_model:
        _add(requested_model)
    if allow_tier_defaults:
        for model in tier_prefs:
            _add(model)
        for model in configured_list:
            _add(model)
    else:
        for model in tier_prefs:
            if model in configured_list:
                _add(model)
        for model in configured_list:
            _add(model)

    if not bases:
        for model in configured_list:
            _add(model)
    if not bases:
        _add(STANDARD_MODEL)

    ladder: list[str] = []
    for base in bases:
        for variant in flavor_variants(base, interactive=interactive):
            if variant not in ladder:
                ladder.append(variant)
    return ladder


def filter_available(ladder: Sequence[str], available: Iterable[str] | None) -> list[str]:
    """Drop ladder entries the endpoint cannot serve (e.g. missing ``-fast``).

    ``available`` is the set of model IDs the Token Factory key actually
    exposes. When it is empty/unknown we cannot filter safely, so the ladder is
    returned unchanged and the resilience loop falls through on 404s. When
    filtering removes everything (misconfiguration), the original ladder is
    returned so a turn is never stranded with an empty ladder.
    """
    ladder_list = [str(m).strip() for m in (ladder or []) if str(m).strip()]
    available_set = {str(m).strip() for m in (available or []) if str(m).strip()}
    if not available_set:
        return ladder_list
    kept = [m for m in ladder_list if m in available_set]
    return kept or ladder_list


def thinking_enabled(tier: str) -> bool:
    """Only the reasoning tier keeps the (billed) hidden reasoning trace."""
    return tier == TIER_REASONING


def chat_extra(tier: str) -> dict[str, Any]:
    """Extra chat-completion payload fields for a tier.

    Non-reasoning tiers disable the model's hidden thinking so we do not pay
    for a trace we discard. Token Factory / vLLM reads ``chat_template_kwargs``.
    """
    if thinking_enabled(tier):
        return {}
    return {"chat_template_kwargs": {"thinking": False}}


def enforce_input_budget(text: str, *, max_chars: int = MAX_INPUT_CHARS) -> tuple[bool, str]:
    """Cap a single user turn to ``max_chars``.

    Returns ``(within_budget, text_or_trimmed)``. When over budget, the middle
    is dropped (head + tail preserved) so both the instruction and any trailing
    question survive, with a visible truncation marker.
    """
    value = str(text or "")
    if len(value) <= max_chars:
        return True, value
    head_len = int(max_chars * 0.7)
    tail_len = max_chars - head_len
    head = value[:head_len].rstrip()
    tail = value[-tail_len:].lstrip() if tail_len > 0 else ""
    trimmed = head + "\n\n...[input truncated to fit the agent token budget]...\n\n" + tail
    return False, trimmed


def usage_summary(data: Any) -> dict[str, int]:
    """Extract a compact token-usage summary from a chat-completion response."""
    if not isinstance(data, dict):
        return {}
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return {}
    summary: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            summary[key] = int(value)
    return summary
