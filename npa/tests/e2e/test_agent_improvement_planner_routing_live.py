"""Opt-in zero-token planner routing with real deployed queue and review receipts.

The coordinator supplies an owner-only bundle after verifying the deployment's
improvements.py hash and an action-loop-only queue configuration. The bundle
contains agent_url, deployment, deployed_source_sha256, components, scope,
item_id, ownership, validation_refs and review_ref. The item must already await
review for the clean local candidate before this test records new observations. Receipts come from actual candidate checks and
an independently obtained review; this test neither creates nor attests them.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import uuid

import pytest

from npa.agent_backend import improvements
from npa.agent_backend.actions import run_action_loop
from npa.agent_backend.improvements import _digest, _read_private, _relative_file
from npa.cli.agent_deployment import build_deployment_manifest
from .agent_live_helpers import load_agent_live_context


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.agent_live,
    pytest.mark.skipif(
        os.environ.get("NPA_AGENT_LIVE") != "1"
        or os.environ.get("NPA_INTEGRATION_E2E") != "1"
        or os.environ.get("NPA_AGENT_IMPROVEMENT_LIVE") != "1",
        reason="Enable the approved deployed planner queue lifecycle explicitly.",
    ),
]


def _result(response):
    assert response.status_code == 200, "deployed planner feedback request failed"
    payload = response.json()
    assert payload.get("grounded") is True
    assert payload.get("usage", {}).get("total_tokens") == 0
    assert payload.get("status") == "ready", "dedicated planner queue is not enabled"
    return payload["result"]


def test_deployed_planner_failure_routing_and_review_lifecycle():
    bundle_path = os.environ.get("NPA_AGENT_IMPROVEMENT_PLANNER_LIVE_BUNDLE", "")
    assert bundle_path, "supply the protected coordinator deployment and lifecycle bundle"
    bundle = json.loads(_read_private(Path(bundle_path)))
    ctx = load_agent_live_context()
    assert bundle["agent_url"].rstrip("/") == ctx.api_base, "deployment source proof targets another agent"
    assert bundle["deployed_source_sha256"] == hashlib.sha256(
        Path(improvements.__file__).read_bytes()
    ).hexdigest(), "coordinator must verify the exact candidate module on the deployment"
    assert bundle["components"] == ["action-loop"], "require an approved action-loop-only queue"
    assert bundle["validation_refs"], "real candidate check receipts are required"
    assert bundle["review_ref"], "an independently produced review receipt is required"

    repository = Path(improvements.__file__).resolve().parents[4]
    deployment = bundle["deployment"]
    expected_deployment = build_deployment_manifest(
        project_alias=deployment["project_alias"], name=deployment["deployment_name"],
        workspace_label=deployment["workspace_label"], repo_root=repository,
        bootstrap_timestamp=deployment["bootstrap_timestamp"],
    )
    assert deployment == expected_deployment, "bundle does not identify the clean local candidate"
    response = ctx.get("/api/deployment")
    assert response.status_code == 200, "deployed source identity is unavailable"
    assert response.json() == expected_deployment, "the running deployment differs from the candidate"

    item_id = bundle["item_id"]
    initial = _result(ctx.get(f"/api/agent/improvements/{item_id}"))["item"]
    assert initial["state"] == "ready_for_review", "prepare real checks before recording more observations"
    assert initial["component"] == "action-loop" and initial["kind"] == "tool_error"
    scope = bundle["scope"]
    assert initial["scope"] == scope and scope["component"] == "action-loop"
    assert "npa/src/npa/agent_backend/improvements.py" in scope["files"]
    snapshot = {
        "base_revision": scope["base_revision"],
        "files": [
            {"path": name, "sha256": hashlib.sha256(_relative_file(repository, name).read_bytes()).hexdigest()}
            for name in scope["files"]
        ],
    }
    assert initial["candidate_sha256"] == _digest(snapshot), "receipts describe different candidate source"
    assert initial["owner"] == bundle["ownership"]["owner"]
    assert initial["generation"] == bundle["ownership"]["generation"]

    for failure in ("unavailable", "malformed"):
        calls = []

        def planner(*args, **kwargs):
            calls.append(1)
            if failure == "unavailable":
                raise RuntimeError("synthetic dedicated planner unavailable")
            return {"choices": [{"message": {"content": "invalid action JSON"}}]}

        result = run_action_loop(
            "inspect available evidence", tools={}, model_call=planner,
        )
        assert result["ok"] is False and result["tokens"] == 0
        assert len(calls) == (1 if failure == "unavailable" else 2)
        assert result["steps"][0]["phase"] == "plan"
        payload = {"episode_id": "planner-probe-" + uuid.uuid4().hex, "result": result}
        first = _result(ctx.post("/api/agent/improvements/reconcile", json=payload))
        assert len(first) == 1, "planner failure was dropped by the scoped deployed queue"
        item = first[0]
        assert item["component"] == "action-loop" and item["kind"] == "tool_error"
        assert item["id"] == bundle["item_id"], "receipts must belong to the planner failure item"
        replay = _result(ctx.post("/api/agent/improvements/reconcile", json=payload))
        assert replay[0]["id"] == item["id"]
        assert replay[0]["occurrences"] == item["occurrences"]
        history = _result(ctx.get(f"/api/agent/improvements/{item['id']}"))
        assert any(
            occurrence["evidence"] == result["steps"][0]
            for occurrence in history["occurrences"]
        ), "the deployed store did not retain the real planner failure"

    for reference in bundle["validation_refs"]:
        state = _result(ctx.post(f"/api/agent/improvements/{item_id}/validation", json={
            **bundle["ownership"], "evidence_ref": reference,
        }))
    assert state["state"] == "ready_for_review"
    verified = _result(ctx.post(f"/api/agent/improvements/{item_id}/review", json={
        "evidence_ref": bundle["review_ref"],
    }))
    assert verified["state"] == "verified"
    lessons = _result(ctx.get("/api/agent/improvements/lessons?target=action-loop"))
    assert any(lesson["item_id"] == item_id for lesson in lessons)
    history = _result(ctx.get(f"/api/agent/improvements/{item_id}"))
    assert history["item"]["review"]["identity_provenance"] == "coordinator-attested-external-review"
