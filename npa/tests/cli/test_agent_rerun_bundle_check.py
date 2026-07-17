"""Unit tests for Rerun bundle eager-load contract (no live infra)."""

from __future__ import annotations

from pathlib import Path

from npa.agent_rerun_bundle_check import (
    FORBIDDEN_UI_MARKERS,
    REQUIRED_UI_MARKERS,
    assert_rerun_ui_eager_load_contract,
    format_bundle_budget_report,
    BundleBudgetResult,
    TimedFetch,
)
from npa.cli import agent as agent_module


def test_agent_ui_source_satisfies_eager_load_contract() -> None:
    import re

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    ui_start = source.index("cat <<'HTML' | sudo tee /opt/npa-agent/ui.html >/dev/null")
    ui_html = source[ui_start : source.index("\nHTML\n", ui_start)]
    # Bootstrap embeds the UI template; markers must exist in the shipped UI HTML.
    errors = assert_rerun_ui_eager_load_contract(ui_html)
    assert errors == [], errors
    iframe = re.search(r'<iframe id="rerunFrame"[^>]*>', ui_html)
    assert iframe is not None, "missing rerunFrame iframe"
    assert "loading=" not in iframe.group(0), iframe.group(0)
    assert "Remount after display:none" not in ui_html
    assert ".tab-panel[hidden] {{" not in ui_html
    assert ".tab-panel[hidden] {" not in ui_html


def test_assert_rerun_ui_eager_load_contract_detects_lazy_iframe() -> None:
    good = "\n".join(REQUIRED_UI_MARKERS)
    assert assert_rerun_ui_eager_load_contract(good) == []
    bad = good + '\n<iframe id="rerunFrame" allowfullscreen loading="lazy"></iframe>'
    errors = assert_rerun_ui_eager_load_contract(bad)
    assert any("loading=" in err or "lazy" in err for err in errors)


def test_forbidden_markers_include_lazy_and_hidden_panel() -> None:
    assert any("lazy" in marker for marker in FORBIDDEN_UI_MARKERS)
    assert any("tab-panel[hidden]" in marker for marker in FORBIDDEN_UI_MARKERS)
    assert any("Loading application bundle" in marker for marker in FORBIDDEN_UI_MARKERS)
    assert 'id="rerunBundleCover"' in REQUIRED_UI_MARKERS


def test_format_bundle_budget_report_includes_fetches() -> None:
    result = BundleBudgetResult(
        ok=False,
        errors=("re_viewer.js TTFB too slow: 9.00s > 3.00s",),
        fetches=(
            TimedFetch("/rerun/re_viewer.js", 200, 9.0, 9.5, 10000),
        ),
        ui_version="2026071102",
    )
    report = format_bundle_budget_report(result)
    assert "ui_version=2026071102" in report
    assert "ok=false" in report
    assert "re_viewer.js" in report
    assert "error: re_viewer.js TTFB too slow" in report
