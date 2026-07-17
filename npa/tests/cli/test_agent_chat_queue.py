"""Contract tests for agent chat queuing and viewer chat drawer."""

from __future__ import annotations

from pathlib import Path

from npa.cli.agent import (
    AGENT_CHAT_QUEUE_CONTRACT,
    AGENT_READABLE_COLOR_CONTRACT,
    AGENT_UI_VERSION,
    AGENT_VIEWER_CHAT_DRAWER_CONTRACT,
)

AGENT_MODULE = Path(__file__).resolve().parents[2] / "src" / "npa" / "cli" / "agent.py"


def _embedded_ui_html(source: str) -> str:
    marker = "cat <<'HTML' | sudo tee /opt/npa-agent/ui.html >/dev/null"
    start = source.index(marker)
    end = source.index("\nHTML\n", start)
    return source[start:end]


def test_chat_queue_contract_in_ui() -> None:
    source = AGENT_MODULE.read_text(encoding="utf-8")
    ui = _embedded_ui_html(source)
    assert f'AGENT_UI_VERSION = "{AGENT_UI_VERSION}"' in source
    for marker in AGENT_CHAT_QUEUE_CONTRACT:
        assert marker in ui, marker
    # Busy send must queue instead of dropping.
    send = ui.split("async function sendChat()")[1].split("function setChatInput")[0]
    assert "queueChatText" in send
    assert "if (chatSendInFlight)" not in send
    queue = ui.split("function enqueueChatJob")[1].split("async function activateMainTab")[0]
    assert "chatQueue.push" in queue
    assert "processChatQueue" in queue


def test_viewer_chat_drawer_contract() -> None:
    source = AGENT_MODULE.read_text(encoding="utf-8")
    ui = _embedded_ui_html(source)
    for marker in AGENT_VIEWER_CHAT_DRAWER_CONTRACT:
        assert marker in ui, marker
    describe = ui.split("async function describeVisual()")[1].split("let lastRrdUpdatedAt")[0]
    assert "openChatDrawer" in describe
    assert "queueChatText" in describe
    # Describe stays in viewer-focus instead of forcing Chat tab takeover.
    assert 'activateMainTab("chat"' not in describe
    # From Viewer, Chat tab opens drawer; Full chat expands to Chat tab.
    assert 'next === "chat" && activeMainTab === "rerun"' in ui
    assert "openFullChatTab" in ui


def test_thinking_ellipsis_high_contrast() -> None:
    source = AGENT_MODULE.read_text(encoding="utf-8")
    ui = _embedded_ui_html(source)
    for marker in AGENT_READABLE_COLOR_CONTRACT:
        assert marker in ui, marker
    assert "thinking-ellipsis" in ui
    assert 'aria-label="thinking">...</span>' in ui
    # Old low-contrast lime sparkle/dots must not be the thinking indicator.
    assert "thinking-dots" not in ui
    assert "sparkle" not in ui.split("thinking-ellipsis")[0][-200:]


def test_soft_swap_prefers_quality_without_rrd_prefetch() -> None:
    source = AGENT_MODULE.read_text(encoding="utf-8")
    ui = _embedded_ui_html(source)
    mount = ui.split("async function mountRerunIframe(camera, runId)")[1].split(
        "async function mountRerunIframeUntilSuccess"
    )[0]
    assert "swapRerunRecordingInPlace" in mount
    assert "already-mounted" in mount
    # Do not prefetch .rrd bodies (many runs); soft-swap loads via add_receiver only.
    assert "prefetchRerunRecording" not in ui
    assert 'rel="prefetch" href="/rerun/recordings/sim2real.rrd"' not in ui
    assert "do not prefetch .rrd bytes" in ui
    assert "waitForQualityRerunFrame" in ui
    swap = ui.split("async function swapRerunRecordingInPlace")[1].split(
        "async function mountRerunIframe(camera, runId)"
    )[0]
    assert "await waitForQualityRerunFrame" in swap
    assert "Updating recording" in swap
    assert "add_receiver" in swap
