"""Stable UI/backend contracts and source embedding helpers for the agent CLI."""

from __future__ import annotations

from pathlib import Path

from npa.cli.agent_source_embed import embedded_module_source


AGENT_MEDIA_PREVIEW_CONTRACT = (
    "authenticatedPreviewObjectUrl",
    "artifactContentUrl",
    "video.src = contentUrl",
    "Loading video preview…",
    "No RRD/MCAP recording; use the artifacts below",
    'pre.textContent = String(payload.text || "")',
    'data-preview-url="',
    "Keep the Rerun iframe mounted under the media pane",
    'id="renderModeVideo"',
    'id="artifactPreviewHost"',
    'id="viewerPaneMedia"',
    "URL.createObjectURL(blob)",
    '@app.api_route("/artifacts/file/{{filename}}", methods=["GET", "HEAD"])',
    '@app.api_route("/artifacts/content", methods=["GET", "HEAD"])',
    "parse_http_byte_range",
    "X-Content-Type-Options",
    "artifact_media_type(",
)

AGENT_RERUN_NO_BUNDLE_SPLASH_CONTRACT = (
    'id="rerunBundleCover"',
    "waitUntilRerunPastBundleSplash",
    "showRerunBundleCover",
    "hideRerunBundleCover",
    "safeHideRerunBundleCover",
    "Warm Rerun assets before revealing the iframe",
    "Preparing viewer…",
    "Uncover without blocking mount latency",
    "non-blank canvas",
    "swapRerunRecordingInPlace",
    "add_receiver",
)

AGENT_VISUAL_FEEDBACK_CONTRACT = (
    'id="describeVisual"',
    "captureVisualContext",
    "describeVisual",
    "[npa-visual-feedback]",
    "visual_context",
    "normalize_messages_for_llm",
    "infer_visual_domain_hints",
    "frameLooksBlank",
    "sampleFrameStats",
    "captureCanvasDataUrl",
    "ensureRerunCaptureBridge",
    "grabFromRerunCaptureBridge",
    "pickBestIframeCanvas",
    "probeRerunCanvasContent",
    "waitForQualityRerunFrame",
    "skipUserAppend",
    "Describe this — capturing",
    "client_max_body_size 32m",
    "maxChars = 700000",
    "_maybe_origin_reply",
    "build_run_origin",
    "Grounded origin facts for this run",
)

AGENT_FOXGLOVE_CONTRACT = (
    'id="renderModeFoxglove"',
    'id="viewerPaneFoxglove"',
    'id="foxgloveHost"',
    'id="foxgloveStatus"',
    "ensureFoxgloveViewer",
    "setFoxgloveDataSource",
    "refreshFoxgloveViewer",
    "mountFoxgloveViewer",
    "/api/foxglove/config",
    "captureFoxgloveContext",
    "/api/foxglove/convert-run",
    "downloadMcap",
    "openFoxgloveWeb",
    "/api/foxglove/export",
    'id="foxgloveOpenWeb"',
    "mountSelfHostedViewer",
    "self-hosted",
    "cross-origin iframe",
)

AGENT_CHAT_QUEUE_CONTRACT = (
    "chatQueue",
    "enqueueChatJob",
    "processChatQueue",
    "queueChatText",
)

AGENT_VIEWER_CHAT_DRAWER_CONTRACT = (
    "viewer-focus",
    "chat-drawer-open",
    'id="chatDrawerToggle"',
    "openChatDrawer",
    "openFullChatTab",
    "setChatDrawerOpen",
    'id="openFullChatTab"',
    'id="chatDrawerClose"',
    "chat-fab",
    "transform-origin: bottom right",
)

AGENT_STAGES_RUN_PICKER_CONTRACT = (
    'id="stagesRunSelect"',
    'id="stagesRunInput"',
    'id="stagesLoadRun"',
    "stages-run-picker",
    "loadSelectedRun",
    "syncRunChooserFields",
    "filterStagesRunSelect",
    "Search NPA workflow/artifact runs",
)

AGENT_READABLE_COLOR_CONTRACT = (
    "--ink-strong",
    "thinking-ellipsis",
    "Color contrast rules",
)


def _embedded_source(path: Path) -> str:
    return embedded_module_source(path)


def _embedded_agent_workflow_source() -> str:
    source = _embedded_source(Path(__file__).with_name("agent_workflow.py"))
    canonical = (
        Path(__file__).resolve().parents[3]
        / "workflows"
        / "workbench"
        / "npa-workflows"
        / "sim2real.yaml"
    ).read_text(encoding="utf-8")
    marker = '_EMBEDDED_CANONICAL_SIM2REAL_YAML = ""'
    if marker not in source:
        raise RuntimeError(
            "agent workflow source lost its canonical Sim2Real embed marker"
        )
    return source.replace(
        marker,
        f"_EMBEDDED_CANONICAL_SIM2REAL_YAML = {canonical!r}",
        1,
    )


def _embedded_agent_routing_source() -> str:
    return _embedded_source(Path(__file__).with_name("agent_routing.py"))


def _embedded_agent_chat_source() -> str:
    return _embedded_source(Path(__file__).with_name("agent_chat.py"))


def _embedded_agent_recordings_source() -> str:
    return _embedded_source(Path(__file__).with_name("agent_recordings.py"))


def _embedded_agent_stages_source() -> str:
    return _embedded_source(Path(__file__).with_name("agent_stages.py"))


def _embedded_agent_visual_feedback_source() -> str:
    return _embedded_source(Path(__file__).with_name("agent_visual_feedback.py"))


def _embedded_agent_rrd_proxy_source() -> str:
    return _embedded_source(Path(__file__).with_name("agent_rrd_proxy.py"))


def _embedded_agent_state_source() -> str:
    return _embedded_source(Path(__file__).with_name("agent_state.py"))


def _embedded_agent_s3_guard_source() -> str:
    return _embedded_source(Path(__file__).with_name("agent_s3_guard.py"))


def _embedded_agent_artifacts_source() -> str:
    return _embedded_source(
        Path(__file__).resolve().parents[1] / "workflows" / "artifacts.py"
    )


def _embedded_agent_artifact_content_source() -> str:
    return _embedded_source(Path(__file__).with_name("agent_artifact_content.py"))


def _embedded_agent_provenance_source() -> str:
    return _embedded_source(
        Path(__file__).resolve().parents[1] / "workflows" / "data_factory_provenance.py"
    )


def rendered_agent_ui_html() -> str:
    """Render the standalone UI template with agent bootstrap constants."""
    from npa.cli.agent import AGENT_UI_VERSION, DEFAULT_AGENT_USER

    raw = Path(__file__).with_name("agent_ui.html").read_text(encoding="utf-8")
    return raw.replace("{AGENT_UI_VERSION}", AGENT_UI_VERSION).replace(
        "{DEFAULT_AGENT_USER}", DEFAULT_AGENT_USER
    )
