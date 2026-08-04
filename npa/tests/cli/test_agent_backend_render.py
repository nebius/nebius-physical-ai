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
    # Phase B/G: actions are shipped/imported and the /agent/act route is wired.
    assert "from agent_backend.actions import (" in body
    assert "def run_action_loop" not in body
    # Recording identity guard embedded + used to gate rerun_ready (no stock demo).
    assert "def recording_has_run_entities" in body
    assert "def _served_recording_is_run_specific" in body
    assert '@app.post("/agent/act")' in body
    # Phase C/G: Sim2Real drive orchestration shipped + route wired.
    assert "from agent_backend.sim2real_loop import (" in body
    assert "def drive_sim2real_loop" not in body
    assert '@app.post("/agent/sim2real/drive")' in body
    # Phase D/G: semantic router shipped + wired into the /chat fallthrough.
    assert "from agent_backend.semantic_router import classify_intent_semantic" in body
    assert "def classify_intent_semantic" not in body
    assert "def _semantic_route" in body
    # Phase F: quantitative signals embedded + memory routes wired.
    assert "def extract_quantitative_signals" in body
    assert '@app.get("/agent/memory/compare")' in body
    assert '@app.get("/agent/memory/explain")' in body
    assert '"memory_explain_regression": _tool_memory_explain_regression' in body
    assert 'diagnosis["run_memory"] = memory_evidence' in body
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
    # Insights backbone wiring: read-only tools shipped in the allowlist +
    # executors, and the /chat action branch drives the loop (no boilerplate).
    assert '"insights_query": _tool_insights_query' in body
    assert '"insights_compare": _tool_insights_compare' in body
    assert "def _agent_insights_settings" in body
    assert "run_chat_action_loop," in body
    assert "run_chat_action_loop(" in body
    assert "Use `POST /api/agent/act` with a JSON body carrying your goal" not in body


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
        ("actions", "def run_action_loop"),
        ("semantic_router", "def classify_intent_semantic"),
        ("sim2real_loop", "def drive_sim2real_loop"),
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


def test_rendered_backend_imports_and_registers_foxglove_routes(monkeypatch, tmp_path):
    """Execute the rendered backend for real, not just compile it.

    The Foxglove routes are registered by a *call* into a shipped module
    (`agent_backend.foxglove_routes`), so a name-ordering or wiring mistake is
    invisible to `ast.parse`/`compile` and would only surface as an ImportError
    on the agent VM. Extract the rendered backend plus its shipped modules into a
    temp package and import it.
    """
    pytest.importorskip("fastapi")
    import importlib.util
    import sys

    setup_script = _capture_setup_script(monkeypatch)

    def _extract(remote_path: str) -> str:
        match = re.search(
            r"cat <<'PY' \| sudo tee " + re.escape(remote_path) + r" >/dev/null\n(.*?)\nPY\n",
            setup_script,
            flags=re.DOTALL,
        )
        assert match, f"bootstrap does not write {remote_path}"
        return match.group(1)

    package = tmp_path / "agent_backend"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    for name in (
        "memory",
        "actions",
        "semantic_router",
        "sim2real_loop",
        "retrieval",
        "trace",
        "foxglove",
        "foxglove_routes",
    ):
        (package / f"{name}.py").write_text(
            _extract(f"/opt/npa-agent/agent_backend/{name}.py"), encoding="utf-8"
        )
    backend_path = tmp_path / "backend.py"
    backend_path.write_text(_extract("/opt/npa-agent/backend.py"), encoding="utf-8")

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location("npa_rendered_backend", backend_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop("npa_rendered_backend", None)

    paths = {getattr(route, "path", "") for route in module.app.routes}
    for expected in (
        "/foxglove/config",
        "/foxglove/status",
        "/foxglove/load-artifact",
        "/foxglove/convert-run",
        "/foxglove/live",
    ):
        assert expected in paths, f"rendered backend did not register {expected}"


def test_rendered_backend_loads_real_skill_excerpts(monkeypatch, tmp_path):
    """The skill loader must resolve real SKILL.md files from skills/index.yaml.

    ``index.yaml`` paths are repo-root-relative, so joining them onto the index's
    own directory produced ``skills/skills/skills/...`` and every excerpt came
    back empty — silently disabling skill injection for the whole agent. Execute
    the rendered loader against the real repo tree so that stays fixed.
    """
    pytest.importorskip("fastapi")
    import importlib.util
    import sys

    setup_script = _capture_setup_script(monkeypatch)

    def _extract(remote_path: str) -> str:
        match = re.search(
            r"cat <<'PY' \| sudo tee " + re.escape(remote_path) + r" >/dev/null\n(.*?)\nPY\n",
            setup_script,
            flags=re.DOTALL,
        )
        assert match, f"bootstrap does not write {remote_path}"
        return match.group(1)

    package = tmp_path / "agent_backend"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    for name in (
        "memory",
        "actions",
        "semantic_router",
        "sim2real_loop",
        "retrieval",
        "trace",
        "foxglove",
        "foxglove_routes",
    ):
        (package / f"{name}.py").write_text(
            _extract(f"/opt/npa-agent/agent_backend/{name}.py"), encoding="utf-8"
        )
    backend_path = tmp_path / "backend.py"
    backend_path.write_text(_extract("/opt/npa-agent/backend.py"), encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(tmp_path))
    # _skill_index_candidates() falls back to Path.cwd()/"skills"/"index.yaml".
    monkeypatch.chdir(repo_root)
    spec = importlib.util.spec_from_file_location("npa_rendered_skill_backend", backend_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    try:
        spec.loader.exec_module(module)

        index, root = module._load_skill_index()
        assert index, "skill index did not load"
        assert (root / index["cosmos3-npa-workflow"]).is_file(), (
            f"skill paths do not resolve from root={root}"
        )

        excerpt = module._skill_excerpt("cosmos3-npa-workflow")
        assert excerpt, "cosmos3-npa-workflow excerpt is empty"
        assert "npa.workflow" in excerpt

        # A Cosmos 3 workflow ask must reach for the npa.workflow skill first,
        # not the SkyPilot-oriented one.
        names, context = module._resolve_skill_context(
            user_text="write me a cosmos3 workflow yaml", intent=None
        )
        assert names[0] == "cosmos3-npa-workflow", names
        assert "npa.workflow" in context
    finally:
        sys.modules.pop("npa_rendered_skill_backend", None)


def test_rendered_backend_has_no_mangled_regex_escapes(monkeypatch) -> None:
    """Regex escapes must survive the outer f-string intact.

    ``setup_script`` is one ~6700-line non-raw f-string, so a single-backslash
    escape inside it is interpreted by the OUTER string first. ``\\s`` and ``\\d``
    only warn, but ``\\b`` is a valid Python escape and silently becomes a
    backspace (0x08) -- the emitted word-boundary anchors were real control
    characters, so intent regexes in the deployed backend could never match.
    Assert on the rendered text, since the source reads correctly either way.
    """
    body = _render_backend_body(monkeypatch)

    control = {c for c in body if c in "\x08\x0c\x0b\x07\x00"}
    assert not control, (
        f"rendered backend contains control characters {sorted(map(hex, map(ord, control)))}; "
        "a single-backslash escape leaked through the outer f-string"
    )
    # The word boundaries are present as real two-character regex escapes.
    assert r"\b(?:stage|stages|step|steps)\b" in body
    assert r"\b(agent-run-[A-Za-z0-9_-]+|sim2real-[A-Za-z0-9_.:-]+)\b" in body


def test_agent_module_source_has_no_invalid_escape_sequences() -> None:
    """``agent.py`` must compile without invalid-escape warnings.

    These are ``SyntaxWarning`` on Python >= 3.12 (noise on every import in the
    workflow pods) and ``DeprecationWarning`` below it, which is why they went
    unnoticed. Compiling the source directly catches them on any interpreter.
    """
    import warnings
    from pathlib import Path

    import npa.cli.agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compile(source, "agent.py", "exec")
    offenders = [str(w.message) for w in caught if "invalid escape" in str(w.message)]
    assert not offenders, offenders


def test_rendered_backend_labels_nurec_camera_without_inheriting(monkeypatch) -> None:
    """A NuRec run must not inherit the previous run's camera label.

    ``sim_viz`` state persists across artifact loads, and ``camera`` is seeded
    from it. Loading a reconstruction after a Sim2Real pipeline run therefore
    reported ``camera="heldout-sim"`` while the very same response carried the
    NuRec note explaining there is no held-out simulation camera. Observed live
    on the deployed agent.
    """
    body = _render_backend_body(monkeypatch)

    assert "NEURAL_RECONSTRUCTION_CAMERA_LABEL" in body
    assert 'NEURAL_RECONSTRUCTION_CAMERA_LABEL = "novel-view"' in body
    # The label is applied on the neural-reconstruction branch, not inherited.
    assert "camera = NEURAL_RECONSTRUCTION_CAMERA_LABEL" in body


def test_rendered_backend_allows_head_on_the_rrd_blob_probe(monkeypatch) -> None:
    """The UI HEADs /api/sim-viz/rrd-blob; a GET-only route answers 405.

    The probe failure is caught and ignored, so the viewer still works -- but it
    logged a console error on every single page load, which is exactly how real
    errors get overlooked. Observed live.
    """
    body = _render_backend_body(monkeypatch)

    assert '@app.api_route("/sim-viz/rrd-blob", methods=["GET", "HEAD"])' in body
    assert '@app.get("/sim-viz/rrd-blob")' not in body
