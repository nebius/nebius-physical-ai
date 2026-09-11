from __future__ import annotations

from npa.agent_backend.access_approval import (
    classify_followup,
    format_later_reply,
    format_open_reply,
    format_plan_reply,
)


def test_agent_approval_conversation_is_explicit_and_resumable() -> None:
    plan = {
        "status": "blocked",
        "counts": {"hf": 2, "ngc": 1},
        "resume_command": "npa configure --prepare-catalog-access",
    }

    assert classify_followup("prepare full catalog access", has_pending_plan=False) == "plan"
    assert classify_followup("yes", has_pending_plan=True) == "open"
    assert classify_followup("done", has_pending_plan=True) == "recheck"
    assert classify_followup("later", has_pending_plan=True) == "later"
    assert classify_followup("yes", has_pending_plan=False) == ""
    reply = format_plan_reply(plan)
    assert "2 HF resource(s)" in reply
    assert "1 NGC artifact(s)" in reply
    assert "will not click" in reply
    assert "npa configure --prepare-catalog-access" in format_later_reply(plan)
    opened = format_open_reply({**plan, "official_urls": ["https://huggingface.co/vendor/repo"]})
    assert "<https://huggingface.co/vendor/repo>" in opened
    assert "has not clicked" in opened


def test_new_approval_plan_requires_unambiguous_provider_or_access_language() -> None:
    positives = (
        "prepare HF access",
        "check NGC approval",
        "audit Hugging Face dataset access",
        "prepare the gated catalog",
        "open the approval pages",
        "review catalog access",
        "approve the NGC artifact",
    )
    for text in positives:
        assert classify_followup(text, has_pending_plan=False) == "plan", text

    negatives = (
        "check my model's dataset status",
        "audit the workflow catalog",
        "review the model metrics",
        "prepare a public dataset",
        "check artifact integrity",
    )
    for text in negatives:
        assert classify_followup(text, has_pending_plan=False) == "", text


def test_pending_approval_plan_keeps_short_followup_behavior() -> None:
    assert classify_followup("yes", has_pending_plan=True) == "open"
    assert classify_followup("done", has_pending_plan=True) == "recheck"
    assert classify_followup("later", has_pending_plan=True) == "later"


def test_agent_ready_reply_does_not_claim_acceptance() -> None:
    reply = format_plan_reply({"status": "ready", "counts": {"hf": 0, "ngc": 0}})
    assert "Ready" in reply
    assert "accepted" not in reply.lower()
