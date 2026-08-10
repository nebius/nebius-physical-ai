"""Rendered-backend compile check for the embedded agent backend.

Renders ``setup_script`` with a mocked SSH client, extracts the ``backend.py``
heredoc body, and ``ast.parse`` + ``compile`` it. This guards the embedded
f-string mechanism: a stray brace or an un-substituted placeholder becomes a
hard failure here instead of a ``SyntaxError`` at agent-VM import time.
"""

from __future__ import annotations

import ast
import builtins
import json
import re
import secrets
import symtable
from pathlib import Path
from types import SimpleNamespace

import pytest

from npa.cli.agent_embed import embedded_python_source


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


def test_embedded_python_source_normalizes_module_and_standalone_block(tmp_path) -> None:
    module = tmp_path / "runtime.py"
    module.write_text(
        '"""Module docs."""\n'
        "from __future__ import annotations\n"
        "kept = True\n"
        "# NPA_EMBED_STANDALONE_START\n"
        "standalone_only = True\n"
        "# NPA_EMBED_STANDALONE_END\n",
        encoding="utf-8",
    )

    source = embedded_python_source(module, strip_standalone=True)

    assert source == "kept = True\n"


def test_rendered_backend_ast_has_no_undefined_globals(monkeypatch) -> None:
    """Catch embedded helpers that reference a global the backend never defines."""
    body = _render_backend_body(monkeypatch)
    ast.parse(body)
    root = symtable.symtable(body, "backend.py", "exec")
    interpreter_globals = {
        "__builtins__",
        "__cached__",
        # Python 3.14's PEP 649 annotation scopes expose this compiler-owned
        # binding through symtable even though it is not a backend dependency.
        "__conditional_annotations__",
        "__file__",
        "__loader__",
        "__name__",
        "__package__",
        "__spec__",
    }
    bound = set(dir(builtins)) | interpreter_globals
    bound.update(
        symbol.get_name()
        for symbol in root.get_symbols()
        if symbol.is_assigned()
        or symbol.is_imported()
        or symbol.is_namespace()
        or symbol.is_parameter()
    )

    unresolved: set[str] = set()

    def scan(table) -> None:
        for symbol in table.get_symbols():
            if symbol.is_referenced() and symbol.is_global() and symbol.get_name() not in bound:
                unresolved.add(symbol.get_name())
        for child in table.get_children():
            scan(child)

    scan(root)
    assert unresolved == set(), f"rendered backend has undefined globals: {sorted(unresolved)}"


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
        "/access",
        "/foxglove/config",
        "/foxglove/status",
        "/foxglove/load-artifact",
        "/foxglove/convert-run",
        "/foxglove/live",
    ):
        assert expected in paths, f"rendered backend did not register {expected}"

    command_secrets = [secrets.token_urlsafe(32) for _index in range(5)]
    public_steps = module._workflow_run_steps(
        [
            {
                "key": "runs/example/npa-workflow/manifest.json",
                "payload": {
                    "steps": [
                        {
                            "state": "publish",
                            "status": "ok",
                            "returncode": 0,
                            "argv": [
                                "tool",
                                f"Authorization: Bearer {command_secrets[0]}",
                                f"--secret-key:{command_secrets[1]}",
                                "--password",
                                command_secrets[2],
                                f"--password={command_secrets[3]}",
                                f"wrapped --token {command_secrets[4]} --keep useful",
                                "--output",
                                "s3://safe-bucket/result.json",
                            ],
                            "outputs": [
                                {
                                    "uri": "https://objects.example/result?token="
                                    + command_secrets[0]
                                }
                            ],
                        }
                    ]
                },
            }
        ]
    )
    assert all(secret not in str(public_steps) for secret in command_secrets)
    assert "<redacted>" in public_steps[0]["command"]
    assert "--keep useful" in public_steps[0]["command"]
    assert "--output s3://safe-bucket/result.json" in public_steps[0]["command"]
    assert public_steps[0]["output_uri"] == "https://objects.example/result"

    report = module.discover_agent_access(
        tenant_id="tenant-test",
        deployment_project_id="project-test",
        fallback_buckets=[],
        list_projects=lambda _tenant: [
            {"metadata": {"id": "project-test", "name": "Project Test"}}
        ],
        list_buckets=lambda _project: [
            {"metadata": {"id": "bucket-resource-test", "name": "bucket-test"}}
        ],
        probe_bucket=lambda _bucket: module.BucketProbe("available", "available"),
        now=lambda: "2026-08-06T23:30:00+00:00",
    )
    monkeypatch.setattr(module, "_agent_access_report", lambda *, refresh=False: report)
    access_payload = module.agent_access(refresh=True)
    assert access_payload["apiVersion"] == "npa.agent.access/v1"
    assert access_payload["identity"]["tenant_id"] == "tenant-test"
    assert access_payload["projects"][0]["id"] == "project-test"

    called: dict[str, object] = {}

    class _RunPage:
        runs = []
        total_runs = 0
        truncated = False
        discovery_complete = True
        source_errors = ()

    def _list_runs(buckets, **kwargs):
        called["buckets"] = list(buckets)
        called["project_map"] = dict(kwargs.get("bucket_projects") or {})
        return _RunPage()

    monkeypatch.setattr(
        module,
        "_agent_s3_client",
        lambda: (object(), {"bucket": "bucket-test", "prefix": ""}),
    )
    monkeypatch.setattr(module, "list_runs_cached_multi", _list_runs)
    scoped = module.artifacts_runs(
        limit=20,
        resource_bucket="bucket-test",
        project_id="project-test",
    )
    assert called["buckets"] == ["bucket-test"]
    assert called["project_map"] == {"bucket-test": "project-test"}
    assert scoped["resource_scope"] == {
        "project_id": "project-test",
        "bucket": "bucket-test",
    }

    indexed_runs = [
        module.RunSummary(
            f"indexed-run-{index}",
            f"2031-01-0{index + 1}T00:00:00Z",
            1,
            False,
            bucket="bucket-test",
            project_id="project-test",
            resolved_prefix=f"category-{index}",
        )
        for index in range(3)
    ]

    class _PagedRunPage:
        runs = indexed_runs
        total_runs = 3
        truncated = False
        discovery_complete = True
        source_errors = ()

    monkeypatch.setattr(module, "list_runs_cached_multi", lambda *_args, **_kwargs: _PagedRunPage())
    first_runs = module.artifacts_runs(limit=2)
    second_runs = module.artifacts_runs(limit=2, cursor=first_runs["next_cursor"])
    assert first_runs["count"] == 2
    assert first_runs["pagination_complete"] is False
    assert second_runs["count"] == 1
    assert second_runs["pagination_complete"] is True
    assert {item["run_id"] for item in [*first_runs["runs"], *second_runs["runs"]]} == {
        "indexed-run-0",
        "indexed-run-1",
        "indexed-run-2",
    }
    assert second_runs["runs"][0]["resolved_prefix"]

    with pytest.raises(module.HTTPException) as exc_info:
        module.artifacts_runs(resource_bucket="caller-bucket", project_id="project-test")
    assert exc_info.value.status_code == 403

    # The exact same access boundary feeds the stage-evidence endpoint. An
    # artifact-only run exposes observed groups, never a Sim2Real template or
    # fabricated execution success/not-run status.
    artifacts = [
        module.Artifact(
            run_id="foreign-run-1",
            key=f"foreign/foreign-run-1/{stage}/output-{index}.bin",
            s3_uri=f"s3://bucket-test/foreign/foreign-run-1/{stage}/output-{index}.bin",
            size=10,
            last_modified="2026-08-07T00:00:00Z",
            render="download",
            inline=False,
        )
        for index, stage in enumerate(("capture", "train", "evaluate"), start=1)
    ]
    first_artifact_page = module.ArtifactListPage(
        artifacts=artifacts[:1],
        truncated=True,
        next_cursor="opaque-page-two",
        page_size=1000,
    )
    source = module.RunSummary(
        "foreign-run-1",
        "2026-08-07T00:00:00Z",
        3,
        False,
        bucket="bucket-test",
        project_id="project-test",
        resolved_prefix="foreign",
    )
    source_search_buckets: list[list[str]] = []

    def _find_selected_source(buckets, **_kwargs):
        source_search_buckets.append(list(buckets))
        return [source], (), True

    monkeypatch.setattr(module, "find_run_sources_across_buckets", _find_selected_source)
    monkeypatch.setattr(
        module,
        "list_artifacts_page",
        lambda *_args, **_kwargs: first_artifact_page,
    )
    first_page = module.artifacts_for_run(
        "foreign-run-1",
        resource_bucket="bucket-test",
    )
    assert source_search_buckets == [["bucket-test"]]
    assert first_page["pagination"] == {
        "contract": "one_native_s3_page",
        "max_objects": 1000,
        "continue_with": ["next_cursor", "resolved_prefix", "resource_bucket", "source_selected"],
    }
    assert first_page["count"] == 1
    assert first_page["truncated"] is True
    assert first_page["next_cursor"] == "opaque-page-two"
    assert first_page["resolved_prefix"] == "foreign"

    # load-run must consume an exact source tuple for duplicate run IDs and
    # persist that tuple with the selected Rerun snapshot. An unqualified
    # duplicate is a 409, never a stale history fallback.
    recording = module.Artifact(
        run_id="foreign-run-1",
        key="foreign/foreign-run-1/reports/sim2real.rrd",
        s3_uri="s3://bucket-test/foreign/foreign-run-1/reports/sim2real.rrd",
        size=128,
        last_modified="2026-08-07T00:00:00Z",
        render="rerun",
        inline=True,
    )
    duplicate_source = module.RunSummary(
        "foreign-run-1",
        "2026-08-07T00:00:00Z",
        1,
        False,
        bucket="bucket-test",
        project_id="project-test",
        resolved_prefix="other",
    )
    with monkeypatch.context() as load_patch:
        state: dict[str, object] = {}
        published = tmp_path / "selected-sim2real.rrd"
        load_patch.setattr(module, "RECORDINGS_DIR", tmp_path / "recordings")
        load_patch.setattr(module, "RECORDING_PATH", published)
        load_patch.setattr(module, "_load_state", lambda: state)
        load_patch.setattr(module, "_save_state", lambda _state: None)
        load_patch.setattr(module, "_record_sim_viz_run", lambda *_args: None)
        load_patch.setattr(module, "_restart_rerun_serve", lambda **_kwargs: False)
        load_patch.setattr(
            module,
            "_publish_rrd_recording",
            lambda path: published.write_bytes(path.read_bytes()),
        )
        load_patch.setattr(
            module,
            "download_s3_uri",
            lambda _uri, path, **_kwargs: (
                path.parent.mkdir(parents=True, exist_ok=True),
                path.write_bytes(b"selected-recording"),
                path,
            )[-1],
        )
        load_patch.setattr(module, "list_artifacts", lambda *_args, **_kwargs: [recording])
        load_patch.setattr(
            module,
            "find_run_sources_across_buckets",
            lambda *_args, **_kwargs: ([source, duplicate_source], (), True),
        )

        loaded = module.sim_viz_load_run(
            {
                "run_id": "foreign-run-1",
                "resource_bucket": "bucket-test",
                "project_id": "project-test",
                "resolved_prefix": "foreign",
                "source_selected": True,
            }
        )
        assert loaded["sim_viz"]["artifact_render"] == "rerun"
        assert loaded["sim_viz"]["artifact_key"].endswith("/reports/sim2real.rrd")
        assert loaded["sim_viz"]["bucket"] == "bucket-test"
        assert loaded["sim_viz"]["resolved_prefix"] == "foreign"

        load_patch.setattr(
            module,
            "find_run_artifacts_across_buckets",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                module.AmbiguousRunSourceError("select an exact source")
            ),
        )
        with pytest.raises(module.HTTPException) as ambiguous_load:
            module.sim_viz_load_run({"run_id": "foreign-run-1"})
        assert ambiguous_load.value.status_code == 409

    cursor_call: dict[str, str] = {}

    def _next_artifact_page(_bucket, _run_id, *, prefix, cursor, **_kwargs):
        cursor_call.update(prefix=prefix, cursor=cursor)
        return module.ArtifactListPage(
            artifacts=artifacts[1:],
            truncated=False,
            next_cursor="",
            page_size=1000,
        )

    monkeypatch.setattr(module, "list_artifacts_page", _next_artifact_page)
    second_page = module.artifacts_for_run(
        "foreign-run-1",
        cursor=first_page["next_cursor"],
        resolved_prefix=first_page["resolved_prefix"],
        resource_bucket=first_page["bucket"],
    )
    assert cursor_call == {"prefix": "foreign", "cursor": "opaque-page-two"}
    assert second_page["count"] == 2
    assert second_page["truncated"] is False

    duplicate = module.RunSummary(
        "foreign-run-1",
        "2026-08-07T00:00:00Z",
        1,
        False,
        bucket="bucket-test",
        project_id="project-test",
        resolved_prefix="other",
    )
    monkeypatch.setattr(
        module,
        "find_run_sources_across_buckets",
        lambda *_args, **_kwargs: ([source, duplicate], (), True),
    )
    ambiguous = module.artifacts_for_run("foreign-run-1")
    assert ambiguous.status_code == 409
    assert b'"code":"ambiguous_run_id"' in ambiguous.body
    assert b'"resolved_prefix":"foreign"' in ambiguous.body
    assert b'"resolved_prefix":"other"' in ambiguous.body

    flat = module.RunSummary(
        "foreign-run-1", "2026-08-07T00:00:00Z", 1, False,
        bucket="bucket-test", project_id="project-test", resolved_prefix="",
    )
    monkeypatch.setattr(
        module,
        "find_run_sources_across_buckets",
        lambda *_args, **_kwargs: ([flat, duplicate], (), True),
    )
    flat_page = module.artifacts_for_run(
        "foreign-run-1", resource_bucket="bucket-test", project_id="project-test",
        source_selected=True,
    )
    assert flat_page["resolved_prefix"] == ""

    monkeypatch.setattr(
        module,
        "find_run_sources_across_buckets",
        lambda *_args, **_kwargs: ([], (), True),
    )
    maintenance = module.artifacts_for_run("codex-maintenance-20310102T030405Z")
    assert maintenance.status_code == 404
    assert b'"code":"run_not_discovered"' in maintenance.body
    assert b"Codex maintenance job IDs" in maintenance.body
    monkeypatch.setattr(
        module,
        "find_run_sources_across_buckets",
        lambda *_args, **_kwargs: ([], ({"code": "artifact_discovery_unavailable"},), False),
    )
    incomplete = module.artifacts_for_run("unseen-run-20310102T030405Z")
    assert incomplete.status_code == 503
    assert b'"code":"artifact_search_incomplete"' in incomplete.body

    monkeypatch.setattr(
        module,
        "find_run_sources_across_buckets",
        lambda *_args, **_kwargs: ([source], (), False),
    )
    incomplete_unique = module.artifacts_for_run("foreign-run-1")
    assert incomplete_unique.status_code == 503
    assert b'"code":"artifact_search_incomplete"' in incomplete_unique.body
    # A fully-qualified source is still safe when unrelated candidates were
    # truncated: the exact bucket + prefix tuple itself was server-discovered.
    exact_incomplete_page = module.artifacts_for_run(
        "foreign-run-1",
        resource_bucket="bucket-test",
        project_id="project-test",
        resolved_prefix="foreign",
        source_selected=True,
    )
    assert exact_incomplete_page["resolved_prefix"] == "foreign"

    monkeypatch.setattr(
        module,
        "_agent_s3_client",
        lambda: (object(), {"bucket": "bucket-test", "prefix": ""}),
    )
    monkeypatch.setattr(
        module,
        "find_run_sources_across_buckets",
        lambda *_args, **_kwargs: ([source], (), True),
    )
    monkeypatch.setattr(module, "list_artifacts", lambda *_args, **_kwargs: artifacts)
    details = module._sim2real_run_details(
        {
            "latest_submit": {},
            "sim_viz": {},
            "sim_viz_runs": {},
            "sim2real_runs": {},
            "workflow_draft": {},
        },
        run_id="foreign-run-1",
        resource_bucket="bucket-test",
        project_id="project-test",
    )
    assert details["project_id"] == "project-test"
    assert details["bucket"] == "bucket-test"
    assert details["stage_summary"]["text"] == (
        "3 observed groups · execution status unavailable"
    )
    assert {stage["status"] for stage in details["stages"]} == {"observed_output"}
    assert details["stage_summary"]["succeeded_count"] == 0
    assert details["stage_summary"]["not_run_count"] == 0
    assert all(stage["evidence_source"] == "artifact_listing" for stage in details["stages"])

    # Stage inspection keeps the selected resource boundary and recursively
    # redacts credential-bearing JSON fields before they reach the browser.
    marker = "credential-value-must-not-escape"
    config_artifact = module.Artifact(
        run_id="foreign-run-1",
        key="foreign/foreign-run-1/configs/manifest.json",
        s3_uri="s3://bucket-test/foreign/foreign-run-1/configs/manifest.json",
        size=128,
        last_modified="2026-08-07T00:00:00Z",
        render="json",
        inline=True,
    )

    class _Body:
        closed = False

        def read(self, size):
            payload = (
                b'{"api_key":"credential-value-must-not-escape",'
                b'"nested":{"password":"credential-value-must-not-escape"},'
                b'"safe":"visible"}'
            )
            return payload[:size]

        def close(self):
            self.closed = True

    class _S3:
        def get_object(self, **_kwargs):
            return {"Body": _Body()}

    monkeypatch.setattr(
        module,
        "_agent_s3_client",
        lambda: (_S3(), {"bucket": "bucket-test", "prefix": ""}),
    )
    monkeypatch.setattr(module, "list_artifacts", lambda *_args, **_kwargs: [config_artifact])
    inspected = module.artifacts_stage(
        "foreign-run-1",
        stage_key="configs",
        resource_bucket="bucket-test",
        project_id="project-test",
    )
    assert inspected["bucket"] == "bucket-test"
    assert inspected["project_id"] == "project-test"
    assert inspected["count"] == 1
    assert marker not in str(inspected)
    assert "visible" in str(inspected)

    # A caller-known bucket and guessed prefix are not sufficient authority.
    # Both cross-project detail surfaces must first match the exact source tuple
    # in bounded server-side run discovery, without attempting object reads.
    monkeypatch.setattr(
        module,
        "list_artifacts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unauthorized artifact listing")
        ),
    )
    for detail_loader in (
        lambda: module.artifacts_stage(
            "foreign-run-1",
            stage_key="configs",
            resource_bucket="bucket-test",
            project_id="project-test",
            resolved_prefix="guessed/private",
            source_selected=True,
        ),
        lambda: module._artifact_backed_run_details(
            {},
            "foreign-run-1",
            resource_bucket="bucket-test",
            project_id="project-test",
            resolved_prefix="guessed/private",
            source_selected=True,
        ),
    ):
        with pytest.raises(module.HTTPException) as unauthorized_source:
            detail_loader()
        assert unauthorized_source.value.status_code == 404

    monkeypatch.setattr(module, "list_artifacts", lambda *_args, **_kwargs: [config_artifact])
    with pytest.raises(module.HTTPException) as detail_exc:
        module.artifacts_stage(
            "foreign-run-1",
            stage_key="configs",
            resource_bucket="caller-bucket",
            project_id="project-test",
        )
    assert detail_exc.value.status_code == 403

    # The run-details/status surface uses the same redacted projection for both
    # workflow_steps and its human-readable logs.
    manifest_payload = json.dumps(
        {
            "workflow": "redaction-regression",
            "steps": [
                {
                    "state": "publish",
                    "status": "ok",
                    "returncode": 0,
                    "argv": [
                        "tool",
                        f"Authorization: Bearer {command_secrets[0]}",
                        f"--secret-key:{command_secrets[1]}",
                        "--password",
                        command_secrets[2],
                        f"--password={command_secrets[3]}",
                    ],
                }
            ],
        }
    ).encode()
    manifest_artifact = module.Artifact(
        run_id="foreign-run-1",
        key="foreign/foreign-run-1/npa-workflow/manifest.json",
        s3_uri="s3://bucket-test/foreign/foreign-run-1/npa-workflow/manifest.json",
        size=len(manifest_payload),
        last_modified="2026-08-07T00:00:00Z",
        render="json",
        inline=True,
    )

    class _ManifestS3:
        def get_object(self, **_kwargs):
            class _ManifestBody:
                def read(self, size):
                    return manifest_payload[:size]

                def close(self):
                    return None

            return {"Body": _ManifestBody()}

    monkeypatch.setattr(
        module,
        "_agent_s3_client",
        lambda: (_ManifestS3(), {"bucket": "bucket-test", "prefix": ""}),
    )
    monkeypatch.setattr(module, "list_artifacts", lambda *_args, **_kwargs: [manifest_artifact])
    redacted_details = module._artifact_backed_run_details(
        {},
        "foreign-run-1",
        resource_bucket="bucket-test",
        project_id="project-test",
    )
    assert redacted_details is not None
    assert all(secret not in str(redacted_details) for secret in command_secrets)
    assert "<redacted>" in str(redacted_details["workflow_steps"])
    assert "<redacted>" in str(redacted_details["logs"])

    invalid_prefixes = (".", "..", "safe/../escape", "safe/%2e%2e/escape")
    for invalid_prefix in invalid_prefixes:
        with pytest.raises(module.HTTPException) as page_exc:
            module.artifacts_for_run(
                "foreign-run-1",
                resolved_prefix=invalid_prefix,
                resource_bucket="bucket-test",
            )
        assert page_exc.value.status_code == 400
        with pytest.raises(module.HTTPException) as stage_exc:
            module.artifacts_stage(
                "foreign-run-1",
                resolved_prefix=invalid_prefix,
                resource_bucket="bucket-test",
                project_id="project-test",
            )
        assert stage_exc.value.status_code == 400
        with pytest.raises(module.HTTPException) as details_exc:
            module._artifact_backed_run_details(
                {},
                "foreign-run-1",
                resolved_prefix=invalid_prefix,
                resource_bucket="bucket-test",
                project_id="project-test",
            )
        assert details_exc.value.status_code == 400

    source_recording = tmp_path / "source.rrd"
    source_recording.write_bytes(b"new-recording")
    published_recording = tmp_path / "published" / "sim2real.rrd"
    published_recording.parent.mkdir()
    published_recording.write_bytes(b"existing-recording")
    monkeypatch.setattr(module, "RECORDING_PATH", published_recording)
    original_replace = Path.replace

    def fail_temp_replace(path, target):
        if path.name.endswith(".tmp"):
            raise OSError("synthetic replace failure")
        return original_replace(path, target)

    with monkeypatch.context() as replace_patch:
        replace_patch.setattr(Path, "replace", fail_temp_replace)
        with pytest.raises(OSError, match="synthetic replace failure"):
            module._publish_rrd_recording(source_recording)
    assert published_recording.read_bytes() == b"existing-recording"
    assert list(published_recording.parent.glob("*.tmp")) == []

    secret = "do-not-return-this-credential"

    def _fail_access(*, refresh=False):
        raise RuntimeError(secret)

    monkeypatch.setattr(module, "_agent_access_report", _fail_access)
    failed = module.agent_access()
    assert failed.status_code == 503
    assert secret.encode() not in failed.body


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
