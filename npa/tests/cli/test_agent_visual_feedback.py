"""Unit tests for Describe-this visual feedback helpers and UI contract."""

from __future__ import annotations

from pathlib import Path

from npa.cli import agent_visual_feedback as vf
from npa.cli.agent import (
    AGENT_UI_VERSION,
    AGENT_VISUAL_FEEDBACK_CONTRACT,
    _embedded_agent_provenance_source,
    _embedded_agent_visual_feedback_source,
)

AGENT_MODULE = Path(__file__).resolve().parents[2] / "src" / "npa" / "cli" / "agent.py"
AGENT_CONTRACTS_MODULE = AGENT_MODULE.with_name("agent_contracts.py")


def _offline_meta(**overrides: object) -> dict[str, object]:
    meta: dict[str, object] = {
        "run_id": "groot17-two-gpu-pipeline-20260811t0131z-example-r11",
        "artifact_key": "reports/groot-offline-evaluation.rrd",
        "note": "Offline held-out policy evaluation",
        "artifact_contract_authoritative": True,
        "evaluation_kind": "offline held-out policy evaluation",
        "closed_loop": False,
        "camera": "front",
    }
    meta.update(overrides)
    return meta


def _embedded_ui_html(source: str = "") -> str:
    """Return rendered agent UI HTML (sourced from agent_ui.html)."""
    from npa.cli.agent import rendered_agent_ui_html

    return rendered_agent_ui_html()



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


def test_describe_prompt_includes_pipeline_provenance() -> None:
    prov = (
        "Augment — Cosmos Transfer 2.5 (nvidia/Cosmos-Transfer2.5-2B) [GPU (Nebius K8s)]; "
        "Pseudo-label augmented — Token Factory VLM (Qwen/Qwen2.5-VL-72B-Instruct) [hosted GPU (Token Factory)]"
    )
    prompt = vf.describe_user_prompt(
        "rerun",
        {"run_id": "paidf-1", "capture": "frame", "has_image": True, "provenance": prov},
    )
    assert "Pipeline provenance" in prompt
    assert "Cosmos Transfer 2.5" in prompt
    assert "Token Factory VLM" in prompt
    # The reply structure must ask the model to state where the visual comes from.
    assert "Where it comes from" in prompt


def test_visual_context_block_surfaces_provenance() -> None:
    block = vf.format_visual_context_block(
        {"run_id": "paidf-1", "provenance": "Augment — Cosmos Transfer 2.5 [GPU (Nebius K8s)]"}
    )
    assert "provenance" in block
    assert "Cosmos Transfer 2.5" in block


def test_describe_prompt_includes_grounded_origin() -> None:
    origin = (
        "No separate original input image was stored for run `paidf-1` — the earliest "
        "stored visuals are the Cosmos Transfer 2.5 augment OUTPUTS."
    )
    prompt = vf.describe_user_prompt(
        "rerun",
        {"run_id": "paidf-1", "capture": "frame", "has_image": True, "origin": origin},
    )
    assert "Original input" in prompt
    assert "No separate original input image was stored" in prompt
    # The reply structure must tell the model to use grounded origin, not guess.
    assert "Original input block" in prompt


def test_visual_context_block_surfaces_origin() -> None:
    block = vf.format_visual_context_block(
        {"run_id": "paidf-1", "origin": "No separate original input image was stored for run `paidf-1`."}
    )
    assert "origin" in block
    assert "No separate original input image was stored" in block


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


def test_groot_learning_hint_forbids_rollout_or_sim_inference() -> None:
    hints = vf.infer_visual_domain_hints(_offline_meta())
    assert len(hints) == 1
    hint = hints[0].lower()
    assert "offline held-out" in hint
    assert "not a simulator/robot rollout" in hint
    assert "do not infer synthetic imagery" in hint
    assert "rollout view in sim" not in hint


def test_learning_visual_reply_fails_closed_on_origin_contradictions() -> None:
    meta = _offline_meta(
        has_image=True,
        capture="frame",
        frame_quality="rendered",
        origin="Original visual evidence is a persisted held-out LeRobot video.",
        provenance="Synchronized learning replay — Rerun + MCAP",
    )
    assert vf.learning_visual_reply_needs_correction(
        "This indicates a synthetic simulation; the absence of an original input is expected.",
        meta,
    )
    assert not vf.learning_visual_reply_needs_correction(
        "The low-resolution camera frame is aligned with expert actions.",
        meta,
    )
    assert not vf.learning_visual_reply_needs_correction(
        "This frame is not controlling the robot; it is offline evaluation.", meta
    )
    reply = vf.truthful_learning_visual_reply(meta)
    assert "quality-captured viewer frame" in reply
    assert "will not invent replacement pixel details" in reply
    assert "does not prove motion, task success, or closed-loop control" in reply


def test_operational_two_gpu_offline_context_is_classified_without_a_frame() -> None:
    meta = _offline_meta(capture="metadata-only", has_image=False)

    assert vf.is_offline_groot_learning_context(meta) is True
    assert "not simulator/robot rollout" in vf.learning_visual_fact_block(meta)
    reply = vf.build_metadata_only_visual_reply(meta)
    assert "offline held-out" in reply.lower()
    assert "not a physical-robot or closed-loop robot rollout" in reply.lower()


def test_offline_semantics_require_authoritative_contract_not_a_filename() -> None:
    malicious = {
        "artifact_key": "reports/groot-offline-evaluation.rrd",
        "note": "closed-loop physical robot rollout",
    }
    assert vf.is_offline_groot_learning_context(malicious) is False
    assert vf.learning_visual_fact_block(malicious) == ""


def test_blank_or_unavailable_capture_is_metadata_only() -> None:
    blank = _offline_meta(
        has_image=True,
        capture="frame",
        frame_quality="blank",
        frame_blank=True,
    )
    reply = vf.truthful_learning_visual_reply(blank)
    assert "No quality-captured frame was available" in reply
    assert "did not inspect pixels" in reply


def test_metadata_only_describe_keeps_structured_visual_feedback_path() -> None:
    source = AGENT_MODULE.read_text(encoding="utf-8")
    assert "if origin_reply and not visual_turn and not has_image_content" in source
    ui = _embedded_ui_html(source)
    assert "explicitly an offline held-out policy evaluation" in ui
    assert "do not infer synthetic imagery or task behavior from the GR00T name" in ui
    assert "The camera pixels come from the persisted held-out LeRobot observation videos" in ui
    assert "Never say the original input is absent" in ui
    assert "rerunRecordingActivatedAt" in ui
    assert "window.Cypress ? 0 : 8000" in ui
    fact_block = vf.learning_visual_fact_block(
        _offline_meta(origin="held-out LeRobot observations")
    )
    assert "NON-NEGOTIABLE FACTS FOR THIS LEARNING REPLAY" in fact_block
    assert "The original visual inputs are present" in fact_block
    assert "A single frame does not prove motion" in fact_block


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
    contract_source = source + AGENT_CONTRACTS_MODULE.read_text(encoding="utf-8")
    ui_html = _embedded_ui_html(source)
    assert f'AGENT_UI_VERSION = "{AGENT_UI_VERSION}"' in source
    for marker in AGENT_VISUAL_FEEDBACK_CONTRACT:
        assert marker in contract_source, (
            f"missing visual-feedback contract marker: {marker!r}"
        )
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
    assert (
        "Always attempt async capture" in ui_html
        or "do not gate" in ui_html.lower()
        or "MediaStream bridge" in ui_html
    )
    assert "visual_context" in ui_html
    assert "maxChars = 700000" in ui_html
    assert "client_max_body_size 32m" in contract_source
    assert "_AGENT_VISUAL_FEEDBACK_EMBED" in source
    assert (
        ".replace(_AGENT_VISUAL_FEEDBACK_EMBED, agent_visual_feedback_source)" in source
    )
    # Chat path must preserve multimodal content (not str()-coerce list parts).
    assert "normalize_messages_for_llm(raw_messages)" in source
    assert "is_visual_feedback_turn(" in source
    assert "None if visual_turn else _agent_chat_with_tools" in source
    assert "infer_visual_domain_hints" in _embedded_agent_visual_feedback_source()
    # Grounded original-input resolution is wired end to end (UI → backend).
    assert "prov.origin" in ui_html
    assert "meta.origin" in ui_html
    assert "grounded-provenance" in source
    assert "build_run_origin" in _embedded_agent_provenance_source()


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
