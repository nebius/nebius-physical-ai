"""Hard contract: users must never see Rerun's Loading application bundle splash."""

from __future__ import annotations

from pathlib import Path

from npa.agent_rerun_bundle_check import (
    FORBIDDEN_UI_MARKERS,
    REQUIRED_UI_MARKERS,
    assert_rerun_ui_eager_load_contract,
)
from npa.cli.agent import AGENT_RERUN_NO_BUNDLE_SPLASH_CONTRACT, AGENT_UI_VERSION

AGENT_MODULE = Path(__file__).resolve().parents[2] / "src" / "npa" / "cli" / "agent.py"
OLD_MOUNT_BEFORE_WARM = 'Mount the viewer immediately so "Loading application bundle" starts early'


def _embedded_ui_html(source: str) -> str:
    """Return the bootstrap-installed ui.html heredoc (not verify-live Python)."""
    marker = "cat <<'HTML' | sudo tee /opt/npa-agent/ui.html >/dev/null"
    start = source.index(marker)
    end = source.index("\nHTML\n", start)
    return source[start:end]


def test_agent_rerun_no_bundle_splash_contract_in_source() -> None:
    source = AGENT_MODULE.read_text(encoding="utf-8")
    ui_html = _embedded_ui_html(source)
    assert f'AGENT_UI_VERSION = "{AGENT_UI_VERSION}"' in source
    for marker in AGENT_RERUN_NO_BUNDLE_SPLASH_CONTRACT:
        assert marker in ui_html, f"missing no-bundle-splash marker in UI: {marker!r}"
    assert 'id="rerunBundleCover"' in ui_html
    assert "waitUntilRerunPastBundleSplash" in ui_html
    assert "showRerunBundleCover" in ui_html
    assert "hideRerunBundleCover" in ui_html
    # Old mount-before-warm strategy that exposed the splash (UI only; verify-live may mention it).
    assert OLD_MOUNT_BEFORE_WARM not in ui_html
    assert assert_rerun_ui_eager_load_contract(ui_html) == []
    # Mount path must warm before navigating the iframe.
    mount_src = ui_html.split("async function mountRerunIframe")[1].split(
        "async function mountRerunIframeUntilSuccess"
    )[0]
    assert "await warmRerunBundle()" in mount_src
    assert "waitUntilRerunPastBundleSplash" in mount_src
    assert "showRerunBundleCover" in mount_src


def test_bundle_check_required_markers_include_cover() -> None:
    assert 'id="rerunBundleCover"' in REQUIRED_UI_MARKERS
    assert "waitUntilRerunPastBundleSplash" in REQUIRED_UI_MARKERS
    assert any("Mount the viewer immediately" in marker for marker in FORBIDDEN_UI_MARKERS)


def test_boot_page_warms_before_mount() -> None:
    source = AGENT_MODULE.read_text(encoding="utf-8")
    ui_html = _embedded_ui_html(source)
    boot = ui_html.split("async function bootPage")[1].split("function startPeriodicRefresh")[0]
    assert "await Promise.all([refreshPromise, artifactsPromise, warmPromise])" in boot
    assert "await ensureFrankaRerunLoaded()" in boot
    # Must not race mount with warm anymore.
    assert "Promise.all([refreshPromise, artifactsPromise, warmPromise, mountPromise])" not in boot
