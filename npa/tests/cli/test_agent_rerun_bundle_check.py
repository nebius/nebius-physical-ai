"""Unit tests for Rerun bundle eager-load contract (no live infra)."""

from __future__ import annotations

import httpx

from npa.agent_rerun_bundle_check import (
    BundleBudgetResult,
    FORBIDDEN_UI_MARKERS,
    REQUIRED_UI_MARKERS,
    TimedFetch,
    assert_rerun_ui_eager_load_contract,
    format_bundle_budget_report,
    timed_get,
)


def test_agent_ui_source_satisfies_eager_load_contract() -> None:
    import re

    from npa.cli.agent import rendered_agent_ui_html
    ui_html = rendered_agent_ui_html()
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


def test_timed_get_retries_incomplete_chunked_read(monkeypatch) -> None:
    calls = {"count": 0}

    class _Response:
        status_code = 200

        def __init__(self, fail: bool) -> None:
            self.fail = fail

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_bytes(self, _size):
            if self.fail:
                raise httpx.RemoteProtocolError(
                    "peer closed connection without sending complete message body "
                    "(incomplete chunked read)"
                )
            yield b"complete"

    def _stream(*_args, **_kwargs):
        calls["count"] += 1
        return _Response(fail=calls["count"] == 1)

    monkeypatch.setattr("npa.agent_rerun_bundle_check.httpx.stream", _stream)
    monkeypatch.setattr("npa.agent_rerun_bundle_check.time.sleep", lambda _seconds: None)

    result = timed_get("https://example.test/rerun/re_viewer.js", auth=None, verify=False, timeout=1)

    assert calls["count"] == 2
    assert result.status_code == 200
    assert result.nbytes == len(b"complete")
