"""Live provider evidence is opt-in and cannot be replaced by fake classifier replies."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path

import pytest

from npa.agent_backend.model_router import MODEL_CRITERIA
from npa.clients.credentials import load_credentials
from npa.clients.token_factory import resolve_config

pytestmark = pytest.mark.token_factory_e2e


def _require_live():
    if os.environ.get("NPA_JEV_ROUTING_LIVE") != "1":
        pytest.skip(
            "Set NPA_JEV_ROUTING_LIVE=1 for real provider routing/cache evaluation"
        )


def _load_script(name):
    path = Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"jev_live_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "use_jev", [False, True], ids=["token_factory_cache", "jev_routes_and_cache"]
)
def test_live_routing_and_provider_cache_evidence(tmp_path, use_jev):
    _require_live()
    module = _load_script("evaluate_jev_routing")
    report = module.run_evaluation(use_jev=use_jev)
    (tmp_path / "jev-routing-evidence.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    assert report["passed"], report["cache_evidence"]


def _isolate_provider_environment(monkeypatch, use_jev):
    config = resolve_config(require_api_key=True)
    key = os.environ.get("TYPESAFE_API_KEY", "") or load_credentials().tokens.get(
        "TYPESAFE_API_KEY", ""
    )
    if use_jev and not key:
        pytest.fail("Live Jev /chat validation requires TYPESAFE_API_KEY")
    for name in tuple(os.environ):
        if name.startswith("NPA_AGENT_") or name in {
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "NEBIUS_S3_BUCKET",
            "NEBIUS_TENANT_ID",
            "TYPESAFE_API_KEY",
        }:
            monkeypatch.delenv(name)
    monkeypatch.setenv("NEBIUS_TOKEN_FACTORY_KEY", config.api_key)
    monkeypatch.setenv("NEBIUS_TOKEN_FACTORY_BASE_URL", config.base_url)
    monkeypatch.setenv("NPA_AGENT_MODEL_ROUTER", "jev")
    if use_jev:
        monkeypatch.setenv("TYPESAFE_API_KEY", key)


def _agent_cases():
    fast, reasoning = MODEL_CRITERIA
    return [
        (
            fast,
            "Write one original sentence containing a red bird and a blue lake.",
            ("bird", "lake"),
        ),
        (
            reasoning,
            "Prove by induction that the sum of the first n odd positive integers equals n squared. End with PROOF_COMPLETE.",
            ("PROOF_COMPLETE",),
        ),
    ]


def _agent_record(client, index, case):
    expected, prompt, required = case
    response = client.post(
        "/chat",
        json={
            "messages": [{"role": "user", "content": prompt}],
            "session_id": f"jev-live-{index}",
        },
        timeout=None,
    )
    response.raise_for_status()
    data = response.json()
    return {
        "expected_model": expected,
        "model": data.get("model"),
        "provider": data.get("provider"),
        "usage": data.get("usage"),
        "routing": data.get("model_routing") or {},
        "required_words_present": all(
            word in data.get("reply", "") for word in required
        ),
        "reply_sha256": hashlib.sha256(data.get("reply", "").encode()).hexdigest(),
    }


def _assert_agent_record(record, use_jev):
    assert record["model"] == record["expected_model"]
    assert record["provider"] == "token_factory"
    assert record["usage"]["total_tokens"] > 0
    assert record["required_words_present"]
    route = record["routing"]
    assert route["served_model"] == record["model"]
    if use_jev:
        assert route["status"] == "accepted"
        assert route["selected_model"] == record["model"]
    else:
        assert route["reason"] == "missing_credential"


@pytest.mark.parametrize(
    "use_jev",
    [False, True],
    ids=["token_factory_rendered_fallback", "jev_rendered_agent"],
)
def test_live_rendered_agent_routes_and_generates(monkeypatch, tmp_path, use_jev):
    _require_live()
    _isolate_provider_environment(monkeypatch, use_jev)
    audit = _load_script("audit_agent_capabilities")
    audit.materialize_runtime(audit.render_backend_body(), tmp_path)
    with audit.serve_live(tmp_path) as client:
        records = [
            _agent_record(client, index, case)
            for index, case in enumerate(_agent_cases())
        ]
    report = {
        "mode": "jev_rendered_agent" if use_jev else "token_factory_rendered_fallback",
        "records": records,
    }
    (tmp_path / "jev-agent-evidence.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    for record in records:
        _assert_agent_record(record, use_jev)
