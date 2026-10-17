"""Validate the UI wiring served by the live agent verification gate."""

from __future__ import annotations

from typing import Callable, NoReturn

from npa.cli.agent_contracts import LEISAAC_CONTROL_READINESS_CONTRACT


_UI_MARKERS = (
    'name="viewport" content="width=device-width',
    'id="chatForm"',
    'id="mobileChatAuth"',
    'id="tabMain"',
    'id="tabRerun"',
    'id="stagesPanel"',
    'id="agentAccessPanel"',
    'id="agentAccessProjectSelect"',
    "/api/access",
    "<h3>Stages</h3>",
    'id="stagesRunSelect"',
    'id="stagesLoadRun"',
    "loadSelectedRun",
    "stages-run-picker",
    "filterStagesRunSelect",
    "Search NPA workflow/artifact runs",
    "function sendChat(",
    "function wireUi(",
    "activateMainTab",
    "initNpaAgentUi",
    "mobile-agent",
    "history.replaceState",
    "location.username",
    'name="npa-ui-version" content="{ui_version}"',
    # Media preview contract — keep in sync with AGENT_MEDIA_PREVIEW_CONTRACT
    # (HTML-visible subset; backend route markers are source-tested separately).
    "authenticatedPreviewObjectUrl",
    "Loading video preview…",
    'id="renderModeVideo"',
    'id="artifactPreviewHost"',
    'id="viewerPaneMedia"',
    "URL.createObjectURL(blob)",
    # No user-visible Rerun "Loading application bundle" splash.
    'id="rerunBundleCover"',
    "waitUntilRerunPastBundleSplash",
    "Preparing viewer…",
    "Warm Rerun assets before revealing the iframe",
    "Uncover without blocking mount latency",
    "scheduleRerunBundleUncover",
    "safeHideRerunBundleCover",
    "non-blank canvas",
    "swapRerunRecordingInPlace",
    "add_receiver",
    # Describe-this visual feedback (vision tier).
    'id="describeVisual"',
    "captureVisualContext",
    "describeVisual",
    "[npa-visual-feedback]",
    "visual_context",
    "enqueueChatJob",
    "processChatQueue",
    "queueChatText",
    "viewer-focus",
    'id="chatDrawerToggle"',
    "thinking-ellipsis",
    "waitForQualityRerunFrame",
    "captureCanvasDataUrl",
    "ensureRerunCaptureBridge",
    "pickBestIframeCanvas",
    "sampleFrameStats",
    "skipUserAppend",
    "Describe this — capturing",
    "do not prefetch .rrd bytes",
    'id="openFullChatTab"',
    "openFullChatTab",
    'id="chatDrawerClose"',
    "chat-fab",
    "transform-origin: bottom right",
    # Embedded Foxglove viewer pane.
    'id="renderModeFoxglove"',
    'id="viewerPaneFoxglove"',
    'id="foxgloveHost"',
    "ensureFoxgloveViewer",
    "mountFoxgloveViewer",
    "/api/foxglove/config",
    'id="tenantResourcesPanel"',
    "<h3>Tenant resources</h3>",
    'id="tenantResourcesRefresh"',
    "refreshTenantResources",
    "/api/resources",
    "Accessible / discovered",
    "Configured references",
    # Capability-gated LeIsaac tab and authenticated WebRTC bridge.
    LEISAAC_CONTROL_READINESS_CONTRACT,
    "ensureLeIsaacTab",
    "ensureLeIsaacTab(leisaacCapability)",
    "unavailableLeIsaacStatus",
    "removeLeIsaacTab",
    "refreshLeIsaacCapability",
    "connectLeIsaac",
    "/api/leisaac/status",
    "/api/leisaac/select",
    "/api/leisaac/bundles/reset",
)


def verify_agent_ui(
    ui_html: str, *, ui_version: str, fail: Callable[[str], NoReturn]
) -> None:
    """Require the served UI's controls, version, and authenticated media wiring.

    Args:
        ui_html: HTML returned by the authenticated agent UI.
        ui_version: Expected UI deployment version.
        fail: Callback that reports a failed contract and terminates verification.

    Returns:
        None.

    Raises:
        typer.Exit: Via ``fail`` when a required UI contract is missing.
    """
    for marker in _UI_MARKERS:
        marker = marker.replace("{ui_version}", ui_version)
        if marker not in ui_html:
            fail(f"UI html missing wiring marker: {marker}")
    _verify_viewer_mount(ui_html, fail)
    _verify_authenticated_media(ui_html, fail)


def _verify_viewer_mount(ui_html: str, fail: Callable[[str], NoReturn]) -> None:
    if 'loading="lazy"' in ui_html:
        fail("UI html must not use lazy-loading on the Rerun iframe")
    if ".tab-panel[hidden]" in ui_html:
        fail("UI html must not hide tab panels with display:none via hidden attribute")
    if (
        'Mount the viewer immediately so "Loading application bundle" starts early'
        in ui_html
    ):
        fail(
            "UI must not mount Rerun before bundle warm (exposes Loading application bundle)"
        )
    if "await waitUntilRerunPastBundleSplash(iframe, 45000)" in ui_html:
        fail("UI must not block mount on long splash wait (latency)")
    if "await waitUntilRerunPastBundleSplash(iframe, 120000)" in ui_html:
        fail("UI must not block mount on long splash wait (latency)")
    load_artifact_source = ui_html.split("async function loadArtifact(payload)")[
        1
    ].split("async function refresh()")[0]
    if "swapRerunRecordingInPlace" not in load_artifact_source:
        fail(
            "loadArtifact must soft-swap Rerun recordings instead of always remounting wasm"
        )


def _verify_authenticated_media(ui_html: str, fail: Callable[[str], NoReturn]) -> None:
    # Guard against regressions that put bare authenticated URLs on <video src>
    # (browsers omit Authorization headers for media elements under basic auth).
    if (
        '`<video controls src="${previewUrl}">`' in ui_html
        or '<video controls src="${previewUrl}">' in ui_html
    ):
        fail("UI html must not assign artifact previewUrl directly to <video src>")
    if '`<img alt="artifact image" src="${previewUrl}"' in ui_html:
        fail("UI html must not assign artifact previewUrl directly to <img src>")
