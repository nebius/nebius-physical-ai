"""Unit tests for cost-aware model routing (agent_routing).

These are pure-logic tests: no network, no tokens, no GPU imports. They guard
the layer that keeps the NPA agent cheap when a turn falls through the grounded
router to Token Factory.
"""

from __future__ import annotations

from npa.cli import agent_routing as r


# ── classify_tier ────────────────────────────────────────────────────────────


def test_classify_tier_defaults_to_cheap_for_short_simple_turns() -> None:
    assert r.classify_tier("what tools are available?") == r.TIER_CHEAP
    assert r.classify_tier("hi") == r.TIER_CHEAP
    assert r.classify_tier("") == r.TIER_CHEAP


def test_classify_tier_escalates_reasoning_on_analytical_language() -> None:
    for prompt in (
        "why did the policy collapse during rollout?",
        "explain the trade-off between H200 and B300",
        "help me debug this failure mode and find the root cause",
        "compare Genesis and Isaac Lab for this task",
    ):
        assert r.classify_tier(prompt) == r.TIER_REASONING, prompt


def test_classify_tier_escalates_standard_on_length_or_many_questions() -> None:
    assert r.classify_tier("x" * (r._STANDARD_LENGTH + 1)) == r.TIER_STANDARD
    assert r.classify_tier("a? b? c? d?") == r.TIER_STANDARD


def test_classify_tier_vision_when_image_content_present() -> None:
    messages = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}
    ]
    assert r.classify_tier("describe this", messages=messages) == r.TIER_VISION


def test_has_image_content_detects_inline_data_uri_and_parts() -> None:
    assert r.has_image_content([{"role": "user", "content": "see data:image/png;base64,AAA"}])
    assert r.has_image_content([{"role": "user", "content": [{"type": "image"}]}])
    assert not r.has_image_content([{"role": "user", "content": "plain text"}])
    assert not r.has_image_content(None)


# ── flavor selection ─────────────────────────────────────────────────────────


def test_flavor_variants_interactive_prefers_fast_then_base() -> None:
    assert r.flavor_variants(r.CHEAP_MODEL, interactive=True) == [
        f"{r.CHEAP_MODEL}-fast",
        r.CHEAP_MODEL,
    ]


def test_flavor_variants_non_interactive_uses_base_only() -> None:
    assert r.flavor_variants(r.CHEAP_MODEL, interactive=False) == [r.CHEAP_MODEL]


def test_flavor_variants_skips_fast_for_non_fast_capable_models() -> None:
    assert r.flavor_variants(r.REASONING_MODEL, interactive=True) == [r.REASONING_MODEL]
    assert r.flavor_variants(r.VISION_MODEL, interactive=True) == [r.VISION_MODEL]


def test_flavor_variants_does_not_double_suffix_fast() -> None:
    already = f"{r.CHEAP_MODEL}-fast"
    assert r.flavor_variants(already, interactive=True) == [already]


# ── build_model_ladder ───────────────────────────────────────────────────────


def test_build_model_ladder_cheap_tier_orders_cheapest_first() -> None:
    ladder = r.build_model_ladder(
        r.TIER_CHEAP,
        [r.CHEAP_MODEL, r.STANDARD_MODEL, r.REASONING_MODEL],
        interactive=True,
    )
    # Cheap model (fast then base) leads; reasoner is a late fallback.
    assert ladder[0] == f"{r.CHEAP_MODEL}-fast"
    assert ladder[1] == r.CHEAP_MODEL
    assert ladder.index(r.CHEAP_MODEL) < ladder.index(r.STANDARD_MODEL)
    assert ladder.index(r.STANDARD_MODEL) < ladder.index(r.REASONING_MODEL)


def test_build_model_ladder_reasoning_tier_leads_with_reasoner() -> None:
    ladder = r.build_model_ladder(r.TIER_REASONING, [], interactive=True)
    assert ladder[0] == r.REASONING_MODEL


def test_build_model_ladder_requested_model_wins() -> None:
    ladder = r.build_model_ladder(
        r.TIER_CHEAP,
        [r.CHEAP_MODEL],
        interactive=False,
        requested_model="custom/Model",
    )
    assert ladder[0] == "custom/Model"


def test_build_model_ladder_respects_explicit_allowlist() -> None:
    # allow_tier_defaults False => only configured models, tier-preferred first.
    configured = [r.STANDARD_MODEL]
    ladder = r.build_model_ladder(
        r.TIER_CHEAP,
        configured,
        interactive=False,
        allow_tier_defaults=False,
    )
    assert ladder == [r.STANDARD_MODEL]
    assert r.CHEAP_MODEL not in ladder


def test_build_model_ladder_never_empty() -> None:
    assert r.build_model_ladder(r.TIER_CHEAP, [], interactive=False, allow_tier_defaults=False)


def test_build_model_ladder_deduplicates() -> None:
    ladder = r.build_model_ladder(
        r.TIER_CHEAP,
        [r.CHEAP_MODEL, r.CHEAP_MODEL, r.STANDARD_MODEL],
        interactive=True,
    )
    assert len(ladder) == len(set(ladder))


# ── thinking / extra payload ─────────────────────────────────────────────────


def test_thinking_only_enabled_for_reasoning_tier() -> None:
    assert r.thinking_enabled(r.TIER_REASONING) is True
    assert r.thinking_enabled(r.TIER_CHEAP) is False
    assert r.thinking_enabled(r.TIER_STANDARD) is False
    assert r.thinking_enabled(r.TIER_VISION) is False


def test_chat_extra_disables_thinking_off_reasoning_tier() -> None:
    assert r.chat_extra(r.TIER_CHEAP) == {"chat_template_kwargs": {"thinking": False}}
    assert r.chat_extra(r.TIER_STANDARD) == {"chat_template_kwargs": {"thinking": False}}
    assert r.chat_extra(r.TIER_REASONING) == {}


# ── input budget guardrail ───────────────────────────────────────────────────


def test_enforce_input_budget_passes_small_input_unchanged() -> None:
    ok, text = r.enforce_input_budget("short question")
    assert ok is True
    assert text == "short question"


def test_enforce_input_budget_trims_oversized_input() -> None:
    huge = "HEAD" + ("x" * 40000) + "TAILQUESTION?"
    ok, text = r.enforce_input_budget(huge, max_chars=1000)
    assert ok is False
    assert len(text) <= 1000 + 100  # marker overhead
    assert text.startswith("HEAD")
    assert "truncated" in text
    assert text.rstrip().endswith("TAILQUESTION?")


# ── usage summary ────────────────────────────────────────────────────────────


def test_usage_summary_extracts_token_counts() -> None:
    data = {"usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19}}
    assert r.usage_summary(data) == {
        "prompt_tokens": 12,
        "completion_tokens": 7,
        "total_tokens": 19,
    }


def test_usage_summary_handles_missing_or_malformed() -> None:
    assert r.usage_summary({}) == {}
    assert r.usage_summary(None) == {}
    assert r.usage_summary({"usage": "nope"}) == {}
    assert r.usage_summary({"usage": {"prompt_tokens": True}}) == {}
