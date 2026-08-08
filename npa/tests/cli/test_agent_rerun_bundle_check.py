"""Unit tests for Rerun bundle eager-load contract (no live infra)."""

from __future__ import annotations

import httpx

from npa import agent_rerun_bundle_check as bundle_check
from npa.agent_rerun_bundle_check import (
    FORBIDDEN_UI_MARKERS,
    REQUIRED_UI_MARKERS,
    assert_rerun_ui_eager_load_contract,
    format_bundle_budget_report,
    BundleBudgetResult,
    TimedFetch,
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
    assert any(
        "Loading application bundle" in marker for marker in FORBIDDEN_UI_MARKERS
    )
    assert 'id="rerunBundleCover"' in REQUIRED_UI_MARKERS


def test_format_bundle_budget_report_includes_fetches() -> None:
    result = BundleBudgetResult(
        ok=False,
        errors=("re_viewer.js TTFB too slow: 9.00s > 3.00s",),
        fetches=(TimedFetch("/rerun/re_viewer.js", 200, 9.0, 9.5, 10000),),
        ui_version="2026071102",
    )
    report = format_bundle_budget_report(result)
    assert "ui_version=2026071102" in report
    assert "ok=false" in report
    assert "re_viewer.js" in report
    assert "error: re_viewer.js TTFB too slow" in report


def test_timed_get_retries_one_truncated_sidecar_response(monkeypatch) -> None:
    calls = 0

    class _Response:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def iter_bytes(self, _chunk_size: int):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise httpx.RemoteProtocolError("incomplete chunked read")
            yield b"rerun-bundle"

    class _Stream:
        def __enter__(self):
            return _Response()

        def __exit__(self, *_args) -> None:
            return None

    monkeypatch.setattr(
        bundle_check.httpx, "stream", lambda *_args, **_kwargs: _Stream()
    )
    monkeypatch.setattr(bundle_check.time, "sleep", lambda _seconds: None)

    result = timed_get(
        "https://agent.example/rerun/re_viewer.js", auth=None, verify=True, timeout=5
    )

    assert result.status_code == 200
    assert result.nbytes == len(b"rerun-bundle")
    assert calls == 2


def test_timed_get_retries_transient_gateway_during_rerun_restart(monkeypatch) -> None:
    calls = 0

    class _Response:
        def __init__(self) -> None:
            nonlocal calls
            calls += 1
            self.status_code = 502 if calls == 1 else 200
            self.request = httpx.Request(
                "GET", "https://agent.example/rerun/re_viewer_bg.wasm"
            )

        def raise_for_status(self) -> None:
            if self.status_code != 200:
                raise httpx.HTTPStatusError(
                    "restart window",
                    request=self.request,
                    response=httpx.Response(self.status_code, request=self.request),
                )

        def iter_bytes(self, _chunk_size: int):
            yield b"rerun-bundle"

    class _Stream:
        def __enter__(self):
            return _Response()

        def __exit__(self, *_args) -> None:
            return None

    monkeypatch.setattr(
        bundle_check.httpx, "stream", lambda *_args, **_kwargs: _Stream()
    )
    monkeypatch.setattr(bundle_check.time, "sleep", lambda _seconds: None)

    result = timed_get(
        "https://agent.example/rerun/re_viewer_bg.wasm",
        auth=None,
        verify=True,
        timeout=5,
    )

    assert result.status_code == 200
    assert result.nbytes == len(b"rerun-bundle")
    assert calls == 2
