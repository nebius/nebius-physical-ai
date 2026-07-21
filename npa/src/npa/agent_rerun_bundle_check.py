"""Live checks that the agent Rerun viewer bundle loads without a visible splash.

Guards against regressions where Chat/Rerun tabs defer wasm download
(loading=lazy, visibility:hidden) or reveal the iframe before assets are
warmed — which surfaces Rerun's own "Loading application bundle" splash.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx

# UI must keep the iframe eager/paintable while Chat is active, and cover it
# until past Rerun's application-bundle splash.
#
# These are intentional string-match regression guards: the agent UI is an
# embedded HTML/JS blob inside agent.py, so we cannot unit-test behavior
# cheaply. Renaming a helper (e.g. waitUntilRerunPastBundleSplash) will break
# CI — update the marker list in the same change when that happens.
REQUIRED_UI_MARKERS = (
    'id="tabMain"',
    'id="tabRerun"',
    "activateMainTab",
    "tab-panel.is-inactive",
    "opacity: 0",
    'id="rerunBundleCover"',
    "waitUntilRerunPastBundleSplash",
    "safeHideRerunBundleCover",
    "Warm Rerun assets before revealing the iframe",
    "Preparing viewer…",
    "Uncover without blocking mount latency",
    "scheduleRerunBundleUncover",
    "non-blank canvas",
    "swapRerunRecordingInPlace",
    "add_receiver",
)

FORBIDDEN_UI_MARKERS = (
    # Exact iframe anti-pattern (avoid matching verify-live source strings).
    'allowfullscreen loading="lazy"',
    'loading="lazy"></iframe>',
    ".tab-panel[hidden] {",
    "Remount after display:none",
    # Old strategy that mounted before warm and exposed Rerun's splash text.
    'Mount the viewer immediately so "Loading application bundle" starts early',
    # Blocking splash waits that add mount/boot latency.
    "await waitUntilRerunPastBundleSplash(iframe, 45000)",
    "await waitUntilRerunPastBundleSplash(iframe, 120000)",
    # Old inactive-tab CSS that deferred wasm/WebGL init while Chat was showing.
    ".tab-panel.is-inactive {\n        position: absolute;\n        left: 16px;\n        right: 16px;\n        top: 16px;\n        visibility: hidden;",
)


@dataclass(frozen=True)
class TimedFetch:
    path: str
    status_code: int
    ttfb_s: float
    total_s: float
    nbytes: int


@dataclass(frozen=True)
class BundleBudgetResult:
    ok: bool
    errors: tuple[str, ...]
    fetches: tuple[TimedFetch, ...]
    ui_version: str


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return default
    return float(raw)


def bundle_budgets() -> dict[str, float]:
    """Return timing budgets (seconds). Overridable via env for slow links."""
    return {
        "js_ttfb": _env_float("NPA_RERUN_JS_TTFB_MAX_SECONDS", 3.0),
        "js_total": _env_float("NPA_RERUN_JS_MAX_SECONDS", 8.0),
        "wasm_ttfb": _env_float("NPA_RERUN_WASM_TTFB_MAX_SECONDS", 5.0),
        "wasm_total": _env_float("NPA_RERUN_WASM_MAX_SECONDS", 30.0),
        "wasm_cached_total": _env_float("NPA_RERUN_WASM_CACHED_MAX_SECONDS", 12.0),
    }


def assert_rerun_ui_eager_load_contract(ui_html: str) -> list[str]:
    """Return human-readable errors if the UI regresses to slow bundle loading."""
    errors: list[str] = []
    for marker in REQUIRED_UI_MARKERS:
        if marker not in ui_html:
            errors.append(f"UI missing eager-load marker: {marker!r}")
    for marker in FORBIDDEN_UI_MARKERS:
        if marker in ui_html:
            errors.append(f"UI contains slow-load anti-pattern: {marker!r}")
    return errors


def timed_get(
    url: str,
    *,
    auth: tuple[str, str] | None,
    verify: bool,
    timeout: float,
) -> TimedFetch:
    started = time.perf_counter()
    ttfb_s = 0.0
    nbytes = 0
    status = 0
    with httpx.stream("GET", url, auth=auth, verify=verify, timeout=timeout) as resp:
        status = int(resp.status_code)
        resp.raise_for_status()
        for chunk in resp.iter_bytes(64 * 1024):
            if nbytes == 0:
                ttfb_s = time.perf_counter() - started
            nbytes += len(chunk)
    total_s = time.perf_counter() - started
    try:
        path = httpx.URL(url).path
    except Exception:  # noqa: BLE001
        path = url
    return TimedFetch(
        path=path,
        status_code=status,
        ttfb_s=ttfb_s,
        total_s=total_s,
        nbytes=nbytes,
    )


def check_rerun_bundle_load_budget(
    agent_base: str,
    *,
    auth: tuple[str, str],
    verify: bool = True,
    timeout: float = 60.0,
) -> BundleBudgetResult:
    """Fetch live UI + Rerun assets and enforce eager-load timing budgets."""
    base = agent_base.rstrip("/")
    errors: list[str] = []
    fetches: list[TimedFetch] = []
    budgets = bundle_budgets()
    ui_version = ""

    ui_resp = httpx.get(f"{base}/", auth=auth, verify=verify, timeout=min(timeout, 20.0))
    if ui_resp.status_code != 200:
        errors.append(f"UI fetch failed status={ui_resp.status_code}")
        return BundleBudgetResult(False, tuple(errors), tuple(fetches), ui_version)
    ui_html = ui_resp.text
    if 'name="npa-ui-version" content="' in ui_html:
        start = ui_html.index('name="npa-ui-version" content="') + len('name="npa-ui-version" content="')
        end = ui_html.find('"', start)
        ui_version = ui_html[start:end] if end > start else ""
    errors.extend(assert_rerun_ui_eager_load_contract(ui_html))

    js = timed_get(
        f"{base}/rerun/re_viewer.js",
        auth=auth,
        verify=verify,
        timeout=timeout,
    )
    fetches.append(js)
    if js.nbytes < 1024:
        errors.append(f"re_viewer.js too small: {js.nbytes} bytes")
    if js.ttfb_s > budgets["js_ttfb"]:
        errors.append(
            f"re_viewer.js TTFB too slow: {js.ttfb_s:.2f}s > {budgets['js_ttfb']:.2f}s "
            "(bundle request should start immediately)"
        )
    if js.total_s > budgets["js_total"]:
        errors.append(
            f"re_viewer.js download too slow: {js.total_s:.2f}s > {budgets['js_total']:.2f}s"
        )

    wasm = timed_get(
        f"{base}/rerun/re_viewer_bg.wasm",
        auth=auth,
        verify=verify,
        timeout=timeout,
    )
    fetches.append(wasm)
    if wasm.nbytes < 1_000_000:
        errors.append(f"re_viewer_bg.wasm too small: {wasm.nbytes} bytes")
    if wasm.ttfb_s > budgets["wasm_ttfb"]:
        errors.append(
            f"re_viewer_bg.wasm TTFB too slow: {wasm.ttfb_s:.2f}s > {budgets['wasm_ttfb']:.2f}s "
            "(Loading application bundle should not stall before first bytes)"
        )
    if wasm.total_s > budgets["wasm_total"]:
        errors.append(
            f"re_viewer_bg.wasm download too slow: {wasm.total_s:.2f}s > {budgets['wasm_total']:.2f}s"
        )

    # Second fetch should hit browser/proxy/nginx cache paths and not feel like a cold stall.
    wasm2 = timed_get(
        f"{base}/rerun/re_viewer_bg.wasm",
        auth=auth,
        verify=verify,
        timeout=timeout,
    )
    fetches.append(wasm2)
    if wasm2.total_s > budgets["wasm_cached_total"]:
        errors.append(
            f"cached re_viewer_bg.wasm download too slow: {wasm2.total_s:.2f}s > "
            f"{budgets['wasm_cached_total']:.2f}s"
        )

    return BundleBudgetResult(
        ok=not errors,
        errors=tuple(errors),
        fetches=tuple(fetches),
        ui_version=ui_version,
    )


def format_bundle_budget_report(result: BundleBudgetResult) -> str:
    lines = [
        f"ui_version={result.ui_version or 'unknown'}",
        f"ok={str(result.ok).lower()}",
    ]
    for fetch in result.fetches:
        lines.append(
            f"{fetch.path}: status={fetch.status_code} ttfb={fetch.ttfb_s:.3f}s "
            f"total={fetch.total_s:.3f}s bytes={fetch.nbytes}"
        )
    for err in result.errors:
        lines.append(f"error: {err}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry for scripts/verify_agent_rerun_bundle.sh."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Agent public URL, e.g. https://IP/")
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--insecure", action="store_true", help="Skip TLS verify (self-signed).")
    args = parser.parse_args(argv)
    result = check_rerun_bundle_load_budget(
        args.base,
        auth=(args.user, args.password),
        verify=not args.insecure,
    )
    report = format_bundle_budget_report(result)
    print(report)
    if not result.ok:
        print("verify_agent_rerun_bundle: FAILED", file=sys.stderr)
        return 1
    print("verify_agent_rerun_bundle: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
