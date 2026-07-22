"""Rendered-backend compile check for the embedded agent backend.

Renders ``setup_script`` with a mocked SSH client, extracts the ``backend.py``
heredoc body, and ``ast.parse`` + ``compile`` it. This guards the embedded
f-string mechanism: a stray brace or an un-substituted placeholder becomes a
hard failure here instead of a ``SyntaxError`` at agent-VM import time.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace

import pytest


def _render_backend_body(monkeypatch) -> str:
    from npa.cli import agent as agent_module

    captured: dict[str, str] = {}

    class _DummySsh:
        def upload_file(self, local_path: str, remote_path: str) -> None:
            if "npa-agent-bootstrap" in remote_path:
                try:
                    captured["setup_script"] = Path(local_path).read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    pass

        def run_or_raise(self, _command: str) -> None:
            return None

        def run(self, _command: str) -> None:
            return None

    monkeypatch.setattr(agent_module, "SSHClient", lambda config: _DummySsh())
    monkeypatch.setattr(agent_module, "resolve_ssh_config", lambda **_kwargs: SimpleNamespace(ssh={}))

    agent_module._bootstrap_agent_stack(
        host="203.0.113.50",
        ssh_user="ubuntu",
        ssh_key_path="/tmp/key",
        project_alias="smoke",
        project_id="project-id",
        tenant_id="tenant-id",
        region="us-central1",
        auth_user="npa",
        auth_password="password",
        agent_port=8088,
        backend_port=8787,
        rerun_port=9090,
        llm_model="nvidia/Cosmos3-Super-Reasoner",
        llm_models=["nvidia/Cosmos3-Super-Reasoner"],
        tf_api_key="",
        nebius_ai_key="",
        public_https=True,
    )
    setup_script = captured["setup_script"]
    match = re.search(
        r"cat <<'PY' \| sudo tee /opt/npa-agent/backend\.py >/dev/null\n(?P<body>.*?)\nPY\n",
        setup_script,
        flags=re.DOTALL,
    )
    assert match, "bootstrap setup script must emit backend.py heredoc"
    return match.group("body")


def test_rendered_backend_compiles(monkeypatch) -> None:
    body = _render_backend_body(monkeypatch)
    # No embed placeholder should survive substitution.
    assert "__NPA_AGENT_" not in body, "an embed placeholder was not substituted"
    tree = ast.parse(body)
    assert tree is not None
    compile(body, "backend.py", "exec")


def test_rendered_backend_wires_action_loop_and_route(monkeypatch) -> None:
    body = _render_backend_body(monkeypatch)
    # Phase B: agent_actions is embedded and the /agent/act route is wired.
    assert "def run_action_loop" in body
    assert "TOOL_ALLOWLIST" in body
    assert '@app.post("/agent/act")' in body
    # Phase C: sim2real drive orchestration embedded + route wired.
    assert "def drive_sim2real_loop" in body
    assert '@app.post("/agent/sim2real/drive")' in body
    # Phase D: semantic router embedded + wired into the /chat fallthrough.
    assert "def classify_intent_semantic" in body
    assert "def _semantic_route" in body
    # Phase F: quantitative signals + run memory embedded + routes wired.
    assert "def extract_quantitative_signals" in body
    assert "class RunMemory" in body
    assert '@app.get("/agent/memory/compare")' in body
    # Grounded-first is preserved: /chat still exists and is separate.
    assert '@app.post("/chat")' in body


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
