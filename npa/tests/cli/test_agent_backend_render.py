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
    # Phase F: quantitative signals embedded + memory routes wired.
    assert "def extract_quantitative_signals" in body
    assert '@app.get("/agent/memory/compare")' in body
    # Phase G: run memory is SHIPPED (imported), not embedded, in backend.py.
    assert "from agent_backend.memory import RunMemory" in body
    assert "class RunMemory" not in body  # no longer inlined into backend.py
    assert "__NPA_AGENT_MEMORY" not in body
    # Blueprint Phase H: retrieval is SHIPPED + routes wired + allowlisted tool.
    assert "from agent_backend import retrieval as _retrieval" in body
    assert '@app.post("/agent/retrieval/index")' in body
    assert '@app.get("/agent/retrieval/search")' in body
    assert '@app.get("/agent/retrieval/status")' in body
    assert "def _maybe_retrieval_grounded" in body
    assert "retrieval-grounded" in body
    # Blueprint Phase I: observability is SHIPPED + trace routes wired.
    assert "from agent_backend import trace as _agent_tracing" in body
    assert '@app.get("/agent/trace/spans")' in body
    assert '@app.post("/agent/trace/analyze")' in body
    assert "def _record_agent_trace" in body
    # Grounded-first is preserved: /chat still exists and is separate.
    assert '@app.post("/chat")' in body


def test_rendered_backend_ships_retrieval_and_trace_modules(monkeypatch) -> None:
    body = _render_backend_body(monkeypatch)
    # Neither shipped module is inlined into backend.py; both are imported.
    assert "def build_lance_store" not in body
    assert "def analyze_traces" not in body
    assert "__NPA_AGENT_RETRIEVAL_SHIP__" not in body
    assert "__NPA_AGENT_TRACE_SHIP__" not in body


def test_shipped_agent_backend_memory_module_compiles(monkeypatch) -> None:
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
        r"cat <<'PY' \| sudo tee /opt/npa-agent/agent_backend/memory\.py >/dev/null\n(?P<body>.*?)\nPY\n",
        setup_script,
        flags=re.DOTALL,
    )
    assert match, "bootstrap must ship agent_backend/memory.py as an importable file"
    body = match.group("body")
    assert "__NPA_AGENT_MEMORY_SHIP__" not in body, "ship placeholder not substituted"
    compile(body, "agent_backend/memory.py", "exec")
    assert "class RunMemory" in body


def _capture_setup_script(monkeypatch) -> str:
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
    return captured["setup_script"]


@pytest.mark.parametrize(
    ("module", "marker"),
    [
        ("retrieval", "def build_lance_store"),
        ("trace", "def analyze_traces"),
    ],
)
def test_shipped_agent_backend_modules_compile(monkeypatch, module, marker) -> None:
    setup_script = _capture_setup_script(monkeypatch)
    match = re.search(
        rf"cat <<'PY' \| sudo tee /opt/npa-agent/agent_backend/{module}\.py >/dev/null\n(?P<body>.*?)\nPY\n",
        setup_script,
        flags=re.DOTALL,
    )
    assert match, f"bootstrap must ship agent_backend/{module}.py as an importable file"
    body = match.group("body")
    assert "__NPA_AGENT_" not in body, "ship placeholder not substituted"
    compile(body, f"agent_backend/{module}.py", "exec")
    assert marker in body


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
