"""Unit tests for Describe-this visual feedback helpers and UI contract."""

from __future__ import annotations

from pathlib import Path

from npa.cli import agent_visual_feedback as vf
from npa.cli.agent import (
    AGENT_UI_VERSION,
    AGENT_VISUAL_FEEDBACK_CONTRACT,
    _embedded_agent_visual_feedback_source,
)

AGENT_MODULE = Path(__file__).resolve().parents[2] / "src" / "npa" / "cli" / "agent.py"


def _embedded_ui_html(source: str) -> str:
    marker = "cat <<'HTML' | sudo tee /opt/npa-agent/ui.html >/dev/null"
    start = source.index(marker)
    end = source.index("\nHTML\n", start)
    return source[start:end]


def test_describe_user_prompt_is_kind_specific() -> None:
    rerun = vf.describe_user_prompt(
        "rerun",
        {"run_id": "agent-run-abc", "camera": "heldout-sim", "capture": "frame"},
    )
    assert vf.DESCRIBE_MARKER in rerun
    assert "heldout-sim" in rerun
    assert "noise/static" in rerun.lower() or "RGB noise" in rerun
    assert "Next actions" in rerun

    video = vf.describe_user_prompt("video", {"run_id": "r1"})
    assert "video viewer" in video
    assert "success/failure" in video.lower() or "success/failure" in vf._KIND_GUIDANCE["video"]

    data = vf.describe_user_prompt("data", {})
    assert "metadata-only" in data or "Data pane" in vf._KIND_GUIDANCE["data"]
    assert "pixels" in data.lower() or "pixels" in vf._KIND_GUIDANCE["data"]


def test_normalize_messages_for_llm_preserves_image_parts() -> None:
    data_url = "data:image/png;base64," + ("A" * 32)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{vf.DESCRIBE_MARKER} Describe this"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]
    llm = vf.normalize_messages_for_llm(messages)
    assert isinstance(llm[0]["content"], list)
    assert llm[0]["content"][1]["type"] == "image_url"
    assert llm[0]["content"][1]["image_url"]["url"].startswith("data:image/")


def test_normalize_messages_for_storage_strips_images() -> None:
    data_url = "data:image/jpeg;base64," + ("B" * 64)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{vf.DESCRIBE_MARKER} hello"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]
    stored = vf.normalize_messages_for_storage(messages, visual_kind="rerun")
    assert isinstance(stored[0]["content"], str)
    assert "data:image/" not in stored[0]["content"]
    assert "omitted" in stored[0]["content"]


def test_oversized_image_is_dropped() -> None:
    huge = "data:image/png;base64," + ("C" * (vf.MAX_IMAGE_DATA_URL_CHARS + 10))
    content = vf.normalize_message_content_for_llm(
        [
            {"type": "text", "text": "see this"},
            {"type": "image_url", "image_url": {"url": huge}},
        ]
    )
    assert content == "see this"


def test_is_visual_feedback_turn_detection() -> None:
    assert vf.is_visual_feedback_turn(user_text=f"{vf.DESCRIBE_MARKER} x")
    assert vf.is_visual_feedback_turn(visual_context={"kind": "rerun"})
    assert vf.is_visual_feedback_turn(
        messages=[
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}],
            }
        ]
    )
    assert not vf.is_visual_feedback_turn(user_text="what is the sim status?")


def test_format_visual_context_block_skips_secrets() -> None:
    block = vf.format_visual_context_block(
        {
            "kind": "rerun",
            "run_id": "agent-run-1",
            "password": "nope",
            "note": "token=abc",
        }
    )
    assert "agent-run-1" in block
    assert "password" not in block
    assert "token=abc" not in block


def test_text_from_content_handles_multimodal() -> None:
    assert (
        vf.text_from_content(
            [{"type": "text", "text": "hello"}, {"type": "image_url", "image_url": {"url": "x"}}]
        )
        == "hello"
    )


def test_embedded_visual_feedback_source_strips_future_import() -> None:
    raw = _embedded_agent_visual_feedback_source()
    assert "def describe_user_prompt(" in raw
    assert "from __future__" not in raw
    assert '"""' not in raw[:40]


def test_ui_and_backend_visual_feedback_contract() -> None:
    source = AGENT_MODULE.read_text(encoding="utf-8")
    ui_html = _embedded_ui_html(source)
    assert f'AGENT_UI_VERSION = "{AGENT_UI_VERSION}"' in source
    for marker in AGENT_VISUAL_FEEDBACK_CONTRACT:
        assert marker in source, f"missing contract marker in agent.py: {marker!r}"
        if marker != "normalize_messages_for_llm":
            assert marker in ui_html or marker in source, marker
    assert 'id="describeVisual"' in ui_html
    assert "async function describeVisual" in ui_html
    assert "async function captureVisualContext" in ui_html
    assert "visual_context" in ui_html
    assert "_AGENT_VISUAL_FEEDBACK_EMBED" in source
    assert ".replace(_AGENT_VISUAL_FEEDBACK_EMBED, agent_visual_feedback_source)" in source
    # Chat path must preserve multimodal content (not str()-coerce list parts).
    assert "normalize_messages_for_llm(raw_messages)" in source
    assert "is_visual_feedback_turn(" in source
    assert "None if visual_turn else _agent_chat_with_tools" in source


def test_build_multimodal_user_content() -> None:
    text_only = vf.build_multimodal_user_content("hi", None)
    assert text_only == "hi"
    multi = vf.build_multimodal_user_content("hi", "data:image/png;base64,abcd")
    assert isinstance(multi, list)
    assert multi[0]["type"] == "text"
    assert multi[1]["type"] == "image_url"
