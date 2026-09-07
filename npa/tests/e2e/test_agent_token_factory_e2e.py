"""Real Token Factory requests through the agent's rendered loopback backend.

Uses synthetic inputs and isolated local state; never deploys or changes a VM.
Run explicitly with provider credentials and pytest --basetemp outside the repo.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import os
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from npa.clients.token_factory import resolve_config

pytestmark = pytest.mark.token_factory_e2e


def test_live_rendered_agent_text_reasoning_and_vision(monkeypatch, tmp_path: Path) -> None:
    config = resolve_config(require_api_key=False)
    if not config.api_key:
        pytest.skip("Live agent model routing requires Token Factory credentials")

    # Keep the live provider credentials while preventing the isolated runtime
    # from using ambient agent state, storage, or trajectory destinations.
    for name in tuple(os.environ):
        if name.startswith("NPA_AGENT_") or name in {
            "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
            "NEBIUS_S3_BUCKET", "NEBIUS_TENANT_ID",
        }:
            monkeypatch.delenv(name)
    monkeypatch.setenv("NEBIUS_TOKEN_FACTORY_KEY", config.api_key)
    monkeypatch.setenv("NEBIUS_TOKEN_FACTORY_BASE_URL", config.base_url)

    audit_path = Path(__file__).resolve().parents[2] / "scripts" / "audit_agent_capabilities.py"
    spec = importlib.util.spec_from_file_location("agent_model_live_audit", audit_path)
    assert spec is not None and spec.loader is not None
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    audit.materialize_runtime(audit.render_backend_body(), tmp_path)

    scene = Image.new("RGB", (512, 384), "white")
    draw = ImageDraw.Draw(scene)
    draw.rectangle((60, 140, 180, 260), fill="red")
    draw.ellipse((300, 140, 420, 260), fill="blue")
    image_path = tmp_path / "synthetic-shapes.png"
    scene.save(image_path)
    image_url = "data:image/png;base64," + base64.b64encode(image_path.read_bytes()).decode("ascii")
    requests = [
        (
            "cheap",
            "nvidia/Nemotron-3_5-Lightning",
            [{"role": "user", "content": "Write one original sentence containing a red bird and a blue lake."}],
            ("bird", "lake"),
        ),
        (
            "reasoning",
            "MiniMaxAI/MiniMax-M3",
            [{"role": "user", "content": "Explain why a sealed box containing only two red balls and three blue balls must yield a red ball within four draws without replacement. Give the worst case ordering."}],
            ("red", "blue"),
        ),
        (
            "vision",
            "MiniMaxAI/MiniMax-M3",
            [{"role": "user", "content": [
                {"type": "text", "text": "Describe the colors, shapes, and left-right order actually visible in the attached image."},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]}],
            ("red", "blue", "left", "right"),
        ),
    ]
    results = []
    with audit.serve_live(tmp_path) as client:
        workflow = """apiVersion: npa.workflow/v0.0.1
kind: Workflow
metadata:
  name: public-metrics-proof
config:
  metrics_input_uri: file:///tmp/public-metrics.json
  insights_store_uri: file:///tmp/public-insights
initial: record
states:
  record:
    toolRef: workbench.insights.record
    terminal: true
"""
        for operation in ("Validate", "Plan"):
            response = client.post("/chat", json={"messages": [{
                "role": "user",
                "content": f"{operation} this workflow YAML and report its status and toolRefs.\n```yaml\n{workflow}```",
            }]}, timeout=None)
            response.raise_for_status()
            operation_result = response.json()
            assert operation_result["grounded"] is True
            assert operation_result["workflow_validation"]["ok"] is True
            assert "public-metrics-proof" in operation_result["reply"]
            if operation == "Plan":
                assert "workbench.insights.record" in operation_result["reply"]
        for tier, expected_model, messages, required_words in requests:
            response = client.post(
                "/chat", json={"messages": messages, "session_id": f"model-test-{tier}"},
                timeout=None,
            )
            response.raise_for_status()
            data = response.json()
            assert data["model"] == expected_model
            assert data["tier"] == tier
            assert data["provider"] == "token_factory"
            assert data["usage"]["total_tokens"] > 0
            reply = data["reply"].lower()
            assert all(word in reply for word in required_words), data["reply"]
            if tier == "vision":
                assert "square" in reply or "rectangle" in reply
                assert "circle" in reply or "circular" in reply
            results.append({key: data[key] for key in ("model", "tier", "provider", "reply", "usage")})
    (tmp_path / "agent-model-results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
