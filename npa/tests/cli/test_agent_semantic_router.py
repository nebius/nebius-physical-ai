"""Tier-0/1 tests for the semantic intent fallthrough (agent_semantic_router).

The deterministic regex router still owns the parity intents at 0 tokens; these
tests cover the fallthrough that catches paraphrases regex misses, with a mocked
model_call so no real tokens are spent.
"""

from __future__ import annotations

import json

from npa.cli import agent_chat
from npa.cli import agent_semantic_router as S

KNOWN = frozenset(agent_chat.INTENT_APIS.keys())


def _completion(obj: dict, tokens: int = 5) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": json.dumps(obj)}}],
        "usage": {"total_tokens": tokens},
    }


def test_keyword_shortcircuit_costs_zero_tokens():
    calls = {"n": 0}

    def _model(messages, *, tier="cheap"):  # pragma: no cover - must not be called
        calls["n"] += 1
        return _completion({"intent": "none"})

    # "keep an eye on the simulation" is a watch paraphrase the regex may miss.
    result = S.classify_intent_semantic(
        "could you keep an eye on the simulation for me",
        known_intents=KNOWN,
        model_call=_model,
    )
    assert result["intent"] == "watch_sim"
    assert result["mode"] == S.MODE_INTENT
    assert result["tokens"] == 0
    assert result["source"] == S.SOURCE_KEYWORD
    assert calls["n"] == 0


def test_paraphrase_routes_via_model_when_keyword_misses():
    def _model(messages, *, tier="cheap"):
        return _completion({"intent": "sonic_capabilities", "confidence": 0.8}, tokens=9)

    result = S.classify_intent_semantic(
        "tell me about whole-body locomotion training support",
        known_intents=KNOWN,
        model_call=_model,
    )
    assert result["intent"] == "sonic_capabilities"
    assert result["mode"] == S.MODE_INTENT
    assert result["source"] == S.SOURCE_MODEL
    assert result["tokens"] == 9


def test_action_mode_when_model_says_action():
    def _model(messages, *, tier="cheap"):
        return _completion({"intent": "action"})

    result = S.classify_intent_semantic(
        "chain a few checks together and then decide what to do",
        known_intents=KNOWN,
        model_call=_model,
    )
    assert result["mode"] == S.MODE_ACTION
    assert result["intent"] is None


def test_unknown_model_intent_degrades_to_none():
    def _model(messages, *, tier="cheap"):
        return _completion({"intent": "totally_made_up"})

    result = S.classify_intent_semantic(
        "some obscure question", known_intents=KNOWN, model_call=_model
    )
    assert result["mode"] == S.MODE_NONE
    assert result["intent"] is None


def test_model_failure_degrades_to_none_without_raising():
    def _model(messages, *, tier="cheap"):
        raise RuntimeError("token factory down")

    result = S.classify_intent_semantic(
        "an open ended question", known_intents=KNOWN, model_call=_model
    )
    assert result["mode"] == S.MODE_NONE
    assert result["tokens"] == 0


def test_cache_short_circuits_second_call():
    calls = {"n": 0}
    cache: dict = {}

    def _model(messages, *, tier="cheap"):
        calls["n"] += 1
        return _completion({"intent": "cosmos_capabilities", "confidence": 0.7}, tokens=8)

    text = "does it handle world-model generation and diffusion inference"
    first = S.classify_intent_semantic(text, known_intents=KNOWN, model_call=_model, cache=cache)
    second = S.classify_intent_semantic(text, known_intents=KNOWN, model_call=_model, cache=cache)
    assert first["intent"] == "cosmos_capabilities"
    assert second["intent"] == "cosmos_capabilities"
    assert second["tokens"] == 0
    assert second["source"] == S.SOURCE_CACHE
    assert calls["n"] == 1  # model consulted once, cache served the rest


def test_low_confidence_model_intent_is_rejected():
    def _model(messages, *, tier="cheap"):
        return _completion({"intent": "sonic_capabilities", "confidence": 0.1})

    result = S.classify_intent_semantic(
        "an ambiguous phrase", known_intents=KNOWN, model_call=_model, min_confidence=0.4
    )
    assert result["mode"] == S.MODE_NONE
    assert result["intent"] is None


def test_non_finite_confidence_is_rejected():
    def _model(messages, *, tier="cheap"):
        return _completion({"intent": "sonic_capabilities", "confidence": float("nan")})

    result = S.classify_intent_semantic(
        "another phrase", known_intents=KNOWN, model_call=_model
    )
    assert result["mode"] == S.MODE_NONE


def test_use_model_false_stays_zero_tokens():
    result = S.classify_intent_semantic(
        "an open question with no keywords",
        known_intents=KNOWN,
        model_call=lambda *a, **k: _completion({"intent": "none"}),
        use_model=False,
    )
    assert result["mode"] == S.MODE_NONE
    assert result["tokens"] == 0


def test_parity_intents_never_reach_semantic_layer():
    # Every intent the regex router already handles must keep matching there, so
    # the semantic fallthrough is never consulted for them (0-token guarantee).
    samples = {
        "what is the current sim2real status": "sim2real_status",
        "load franka demo": "load_franka",
        "show me the tools catalog": "tools_catalog",
        "what does cosmos support": "cosmos_capabilities",
    }
    for text, _expected in samples.items():
        assert agent_chat.match_chat_intent(text) is not None, text
