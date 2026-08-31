"""The agent capability audit must keep working, and its verdict must stay green.

Nothing else runs ``audit_agent_capabilities.py``, and it couples to internals
that move: the private ``_bootstrap_agent_stack`` keyword signature, the
``backend.py`` heredoc shape in the generated setup script, and
``agent_chat``'s ``match_chat_intent`` / ``build_grounded_reply``. Without this
test the script rots silently and the next person to reach for it finds a
traceback instead of an answer.

Running its offline tier here also makes it a whole-surface regression gate:
a route that stops being registered, a route that starts 5xx-ing, an intent
that stops matching, or a duplicate registration that shadows a live handler
all fail here rather than on someone's VM.

Offline and free -- no cluster, no VM, no Token Factory call, no port bound.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "npa" / "scripts" / "audit_agent_capabilities.py"

# The script execs the rendered backend into this module name.
_RENDERED_MODULE = "npa_audit_backend"


def _load():
    spec = importlib.util.spec_from_file_location("audit_agent_capabilities", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations through sys.modules, so register before exec.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def audit(tmp_path):
    """Load the script and undo the import-state it mutates."""
    original_path = list(sys.path)
    module = _load()
    try:
        yield module
    finally:
        sys.modules.pop("audit_agent_capabilities", None)
        sys.modules.pop(_RENDERED_MODULE, None)
        sys.path[:] = original_path


def test_audit_script_renders_and_reports_a_healthy_surface(audit, tmp_path) -> None:
    from fastapi.testclient import TestClient

    body = audit.render_backend_body()
    assert body.strip(), "bootstrap emitted no backend.py body"
    assert "__NPA_AGENT_" not in body, (
        "rendered backend still contains an unsubstituted embed placeholder"
    )

    app, _globals = audit.load_backend_app(body, tmp_path)
    routes = audit.iter_routes(app)
    # Guards against a vacuous pass: an empty routing table has no bad routes
    # and no duplicates, so every assertion below would hold for the wrong
    # reason. The floor is deliberately loose -- this is a smoke bound, not a
    # count to bump on every new route.
    assert len(routes) > 80, f"suspiciously few routes enumerated: {len(routes)}"
    assert ("GET", "/health") in routes

    assert audit.shadowed_routes(app) == [], (
        "a method+path is registered more than once; Starlette serves the first "
        "and every later registration is unreachable"
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        probes = audit.probe_routes(client, routes)
        capabilities = audit.probe_capabilities(client)

    assert probes, "no routes were probed"
    broken = [
        f"{p['method']} {p['path']} -> {p.get('status')} {p.get('error') or p.get('detail') or ''}"
        for p in probes
        if audit.classify_outcome(p) == "error"
    ]
    # `needs_arguments`, `gated`, and `absent_in_sandbox` are correct outcomes
    # for a sandbox with no bucket, no TLS ingress, and no staged recording.
    # `error` means a 5xx or an unhandled exception, which never is.
    assert not broken, "routes failed rather than declining:\n" + "\n".join(broken)

    not_working = [
        f"{c['capability']}: {c.get('status')} {c.get('detail')}"
        for c in capabilities
        if not c.get("works")
    ]
    assert not not_working, "advertised capabilities did not work:\n" + "\n".join(
        not_working
    )


def test_audit_script_confirms_every_chat_intent_still_routes(audit) -> None:
    probes = audit.probe_chat_router()
    assert len(probes) > 30, f"intent probe list shrank unexpectedly: {len(probes)}"

    unmatched = [
        f"{p['expected_intent']}: {p['prompt']!r} matched {p.get('matched_intent')!r}"
        for p in probes
        if not p.get("intent_ok")
    ]
    assert not unmatched, (
        "these prompts stopped reaching their grounded intent, so each now costs "
        "a paid model call for an answer the zero-token layer has:\n"
        + "\n".join(unmatched)
    )

    silent = [
        f"{p['expected_intent']}: {p.get('error', 'empty reply')}"
        for p in probes
        if not p.get("reply_ok")
    ]
    assert not silent, "these intents matched but produced no reply:\n" + "\n".join(
        silent
    )


def test_audit_render_drift_diff_ignores_notation_not_real_drift(audit) -> None:
    """The drift check must not report path-converter spelling as drift.

    Starlette keeps the converter in ``route.path`` while OpenAPI emits the bare
    name, and FastAPI never lists its own doc routes in the document it serves.
    Reporting those made the deployed-tier diff 18 entries of noise, which is
    how a drift check becomes something people ignore.
    """
    assert audit.comparable_route("GET", "/artifacts/run/{run_id:path}") == (
        "GET",
        "/artifacts/run/{run_id}",
    )
    assert audit.comparable_route("GET", "/health") == ("GET", "/health")
    for path in ("/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"):
        assert path in audit._OPENAPI_UNLISTED_PATHS


def test_audit_shadowed_route_detector_can_go_red(audit) -> None:
    """A duplicate detector that cannot fire proves nothing about the backend."""
    from fastapi import FastAPI

    app = FastAPI()

    @app.get("/solo")
    def solo():  # pragma: no cover - registration is the subject
        return {}

    assert audit.shadowed_routes(app) == []

    app.add_api_route("/solo", lambda: {}, methods=["GET"])
    assert audit.shadowed_routes(app) == ["GET /solo (x2)"]
