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
    assert "NOT 'blank'" in rerun or "not blank" in rerun.lower() or "not 'blank'" in rerun
    assert "Next actions" in rerun

    video = vf.describe_user_prompt("video", {"run_id": "r1"})
    assert "video viewer" in video
    assert "success/failure" in video.lower() or "success/failure" in vf._KIND_GUIDANCE["video"]

    data = vf.describe_user_prompt("data", {"text_excerpt": '{"success_rate": 0.4}'})
    assert "success_rate" in data
    assert "pixels" in data.lower() or "pixels" in vf._KIND_GUIDANCE["data"]


def test_metadata_only_grounded_reply_never_invents_pixels() -> None:
    reply = vf.build_metadata_only_visual_reply(
        {
            "kind": "rerun",
            "run_id": "demo-workbench-ui",
            "artifact_key": "checkpoints/sim2real-b/demo-workbench-ui/reports/sim2real.rrd",
            "note": "Isaac Lab GR00T proxy",
        }
    )
    assert "metadata only" in reply.lower()
    assert "inventing pixels" in reply.lower()
    assert "No viewer frame was attached" in reply
    assert "demo-workbench-ui" in reply
    assert "GR00T" in reply or "foundation-policy" in reply or "Isaac" in reply
    assert "RGB noise" not in reply


def test_metadata_only_prompt_forbids_invented_pixels() -> None:
    prompt = vf.describe_user_prompt(
        "rerun",
        {
            "artifact_key": "checkpoints/sim2real-b/demo-workbench-ui/reports/sim2real.rrd",
            "capture": "metadata-only",
            "has_image": False,
            "note": "Isaac Lab GR00T proxy",
        },
    )
    assert "No viewer frame image is attached" in prompt
    assert "Do NOT invent pixels" in prompt


def test_infer_visual_domain_hints_from_metadata_not_uri_allowlist() -> None:
    hints = vf.infer_visual_domain_hints(
        {
            "artifact_key": "checkpoints/sim2real-b/demo-workbench-ui/reports/sim2real.rrd",
            "visualization_note": "Isaac Lab held-out camera with GR00T policy proxy",
            "workflow_name": "sim2real",
        }
    )
    joined = " ".join(hints).lower()
    assert "gr00t" in joined or "foundation-policy" in joined
    assert "isaac" in joined
    prompt = vf.describe_user_prompt(
        "rerun",
        {
            "artifact_key": "checkpoints/sim2real-b/demo-workbench-ui/reports/sim2real.rrd",
            "note": "Isaac Lab + GR00T visualization",
            "capture": "frame",
            "frame_quality": "rendered",
        },
    )
    assert "Domain hints" in prompt
    assert "blank" in prompt.lower()  # guidance warns against false blank calls


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
    assert 'id="describeVisual"' in ui_html
    assert "async function describeVisual" in ui_html
    assert "async function captureVisualContext" in ui_html
    assert "waitForQualityRerunFrame" in ui_html
    assert "frameLooksBlank" in ui_html
    assert "sampleFrameStats" in ui_html
    assert "captureCanvasDataUrl" in ui_html
    assert "ensureRerunCaptureBridge" in ui_html
    assert "grabFromRerunCaptureBridge" in ui_html
    assert "pickBestIframeCanvas" in ui_html
    assert "skipUserAppend" in ui_html
    assert "Describe this — capturing" in ui_html
    # Must not gate async WebGL capture on sync blank checks alone.
    assert "Always attempt async capture" in ui_html or "do not gate" in ui_html.lower() or "MediaStream bridge" in ui_html
    assert "visual_context" in ui_html
    assert "maxChars = 700000" in ui_html
    assert "client_max_body_size 32m" in source
    assert "_AGENT_VISUAL_FEEDBACK_EMBED" in source
    assert ".replace(_AGENT_VISUAL_FEEDBACK_EMBED, agent_visual_feedback_source)" in source
    # Chat path must preserve multimodal content (not str()-coerce list parts).
    assert "normalize_messages_for_llm(raw_messages)" in source
    assert "is_visual_feedback_turn(" in source
    assert "None if visual_turn else _agent_chat_with_tools" in source
    assert "infer_visual_domain_hints" in _embedded_agent_visual_feedback_source()


def test_build_multimodal_user_content() -> None:
    text_only = vf.build_multimodal_user_content("hi", None)
    assert text_only == "hi"
    multi = vf.build_multimodal_user_content("hi", "data:image/png;base64,abcd")
    assert isinstance(multi, list)
    assert multi[0]["type"] == "text"
    assert multi[1]["type"] == "image_url"


def test_frame_looks_blank_from_stats_rejects_uniform_gray() -> None:
    # Cleared WebGL buffers often land as mid-gray with ~0 variance.
    assert vf.frame_looks_blank_from_stats(mean=160.0, variance=2.0, value_range=4.0)
    assert vf.frame_looks_blank_from_stats(mean=3.0, variance=1.0, value_range=2.0)
    assert vf.frame_looks_blank_from_stats(mean=250.0, variance=1.0, value_range=3.0)
    # Skeleton-on-dark-grid style content has high variance/range.
    assert not vf.frame_looks_blank_from_stats(mean=40.0, variance=1200.0, value_range=200.0)
    # Sparse orange/cyan strokes on near-black: mean/variance stay tiny, but vivid pixels count.
    assert not vf.frame_looks_blank_from_stats(
        mean=4.0,
        variance=8.0,
        value_range=210.0,
        vivid=vf.BLANK_VIVID_MIN + 3,
        vivid_ratio=vf.BLANK_VIVID_RATIO_MIN * 2,
    )


def test_blank_detection_constants_are_mirrored_in_ui_source() -> None:
    source = AGENT_MODULE.read_text(encoding="utf-8")
    ui_html = _embedded_ui_html(source)
    for name in (
        "BLANK_VIVID_MIN",
        "BLANK_VIVID_RATIO_MIN",
        "BLANK_LIT_MIN",
        "BLANK_VARIANCE_STRICT",
        "BLANK_RANGE_MIN",
    ):
        value = getattr(vf, name)
        assert f"const {name} = {value}" in ui_html, f"UI missing {name}={value}"
        assert f"{name} = {value}" in Path(vf.__file__).read_text(encoding="utf-8")


def test_g1_trajectory_domain_hint_warns_against_blank_claim() -> None:
    hints = vf.infer_visual_domain_hints(
        {
            "note": "G1 trajectory overlay",
            "artifact_key": "reports/locomotion.rrd",
            "camera": "heldout-sim",
        }
    )
    joined = " ".join(hints).lower()
    assert "skeleton" in joined or "trajectory" in joined or "locomotion" in joined
    assert "blank" in joined or "uniform-gray" in joined or "uniform" in joined
    prompt = vf.describe_user_prompt(
        "rerun",
        {
            "note": "G1 trajectory",
            "capture": "frame",
            "has_image": True,
            "frame_quality": "rendered",
        },
    )
    assert "skeleton" in prompt.lower() or "wireframe" in prompt.lower()
    assert "NOT 'blank'" in prompt or "not blank" in prompt.lower()
