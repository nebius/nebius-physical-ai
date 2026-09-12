"""Regression tests for the shipped Agent artifact-route policy."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import importlib.util
import re
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from npa.agent_backend import artifact_routes
from npa.agent_backend.artifact_routes import (
    ARTIFACT_ROUTE_POLICY_CONTRACT,
    artifact_run_snapshot_scope,
    build_artifact_run_detail_response,
    build_artifact_run_list_response,
    decide_artifact_run_lookup,
)
from npa.agent_backend.shipping import render_shipped_backend_install


@dataclass(frozen=True)
class _Run:
    run_id: str
    project_id: str = "project-a"
    bucket: str = "artifact-bucket"
    resolved_prefix: str = "workflow"

    @property
    def run_ref(self) -> str:
        return f"ref-{self.project_id}-{self.resolved_prefix}"

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "project_id": self.project_id,
            "bucket": self.bucket,
            "resolved_prefix": self.resolved_prefix,
            "source_prefix": self.resolved_prefix,
            "run_ref": self.run_ref,
            "summary_complete": True,
        }


class _Snapshots:
    def __init__(self) -> None:
        self.items: list[object] = []
        self.metadata: dict[str, object] = {}
        self.context = ""

    def page(self, *, cursor, context, limit, items=None, metadata=None):
        if cursor:
            assert items is None
            assert metadata is None
            assert cursor.startswith("snapshot:")
            assert context == self.context
            offset = int(cursor.rsplit(":", 1)[-1])
        else:
            assert items is not None
            self.items = list(items)
            self.metadata = dict(metadata or {})
            self.context = context
            offset = 0
        end = min(offset + limit, len(self.items))
        next_cursor = f"snapshot:{end}" if end < len(self.items) else ""
        return self.items[offset:end], next_cursor, dict(self.metadata)


def _page(*runs: _Run, complete: bool = True, errors=()):
    return SimpleNamespace(
        runs=list(runs),
        truncated=not complete,
        discovery_complete=complete,
        total_runs=len(runs),
        source_errors=tuple(errors),
    )


def test_cold_search_filters_refreshed_index_before_immutable_pagination() -> None:
    snapshots = _Snapshots()
    first = build_artifact_run_list_response(
        _page(
            _Run("paidf-first"),
            _Run("unrelated-run"),
            _Run("paidf-second"),
        ),
        query="paidf",
        page_size=1,
        cursor="",
        snapshot_context="scope-digest",
        snapshot_page=snapshots.page,
        effective_scope_complete=True,
    )

    assert [item["run_id"] for item in first["runs"]] == ["paidf-first"]
    assert first["observed_run_count"] == 3
    assert first["observed_match_count"] == 2
    assert first["total_runs"] == 2
    assert first["pagination_complete"] is False

    # A continuation never accepts a newly scanned/stale page. It resumes the
    # original filtered snapshot and therefore cannot skip or duplicate rows.
    second = build_artifact_run_list_response(
        None,
        query="paidf",
        page_size=1,
        cursor=first["next_cursor"],
        snapshot_context="scope-digest",
        snapshot_page=snapshots.page,
        effective_scope_complete=True,
    )
    assert [item["run_id"] for item in second["runs"]] == ["paidf-second"]
    assert second["next_cursor"] == ""
    assert second["pagination_complete"] is True


def test_partial_effective_scope_never_reports_a_globally_empty_result() -> None:
    response = build_artifact_run_list_response(
        _page(),
        query="missing",
        page_size=100,
        cursor="",
        snapshot_context="partial-scope",
        snapshot_page=_Snapshots().page,
        effective_scope_complete=False,
    )

    assert response["runs"] == []
    assert response["total_runs"] is None
    assert response["total_runs_scope"] == "unavailable"
    assert response["query_complete"] is False
    assert response["pagination_complete"] is False
    assert response["truncated"] is True
    assert response["source_errors"] == [
        {
            "code": "artifact_search_incomplete",
            "message": (
                "One or more authorized artifact sources could not be "
                "searched completely."
            ),
        }
    ]


def test_snapshot_scope_binds_query_selection_and_complete_source_tuples() -> None:
    scope = artifact_run_snapshot_scope(
        query="paidf",
        prefix=" workflow/ ",
        resource_scope={"project_id": "project-a", "bucket": "bucket-a"},
        source_tuples=(
            ("project-b", "bucket-a", "second"),
            ("project-a", "bucket-a", "first"),
            ("project-a", "bucket-a", "first"),
        ),
    )

    assert scope == {
        "contract": ARTIFACT_ROUTE_POLICY_CONTRACT,
        "query": "paidf",
        "prefix": "workflow",
        "resource_scope": {"project_id": "project-a", "bucket": "bucket-a"},
        "sources": [
            ["project-a", "bucket-a", "first"],
            ["project-b", "bucket-a", "second"],
        ],
    }
    with pytest.raises(ValueError, match="three fields"):
        artifact_run_snapshot_scope(
            query="", prefix="", resource_scope={}, source_tuples=(("only", "two"),)
        )


def test_lookup_preserves_project_in_source_ambiguity() -> None:
    decision = decide_artifact_run_lookup(
        run_id="shared-run",
        matches=(
            _Run("shared-run", project_id="project-a"),
            _Run("shared-run", project_id="project-b"),
        ),
        source_errors=(),
        discovery_complete=True,
        effective_scope_complete=True,
        exact_source_authorized=False,
    )

    assert decision.status_code == 409
    assert decision.selected is None
    assert decision.body is not None
    assert decision.body["error"]["code"] == "ambiguous_run_id"
    assert [item["project_id"] for item in decision.body["sources"]] == [
        "project-a",
        "project-b",
    ]


def test_detail_page_preserves_exact_tuple_and_unknown_artifacts() -> None:
    selected = _Run("durable-run", resolved_prefix="workflow/runs")
    unknown = {
        "key": "workflow/runs/durable-run/reports/result.new-format",
        "render": "download",
        "size": 123,
    }
    response = build_artifact_run_detail_response(
        selected=selected,
        artifact_page={
            "artifacts": [unknown],
            "truncated": True,
            "next_cursor": "native-s3-cursor",
            "page_size": 1000,
        },
        project_id=selected.project_id,
        base_prefix="",
        access={"status": "available"},
        preferred=unknown,
    )

    assert response["artifacts"] == [unknown]
    assert response["preferred"] == unknown
    assert response["resource_bucket"] == selected.bucket
    assert response["project_id"] == selected.project_id
    assert response["resolved_prefix"] == selected.resolved_prefix
    assert response["run_ref"] == selected.run_ref
    assert response["source_selected"] is True
    assert response["pagination"]["continue_with"] == [
        "next_cursor",
        "run_ref",
        "project_id",
        "resource_bucket",
        "resolved_prefix",
        "source_selected",
    ]
    assert response["truncated"] is True
    assert response["next_cursor"] == "native-s3-cursor"


def test_positive_lookup_preserves_incomplete_scope_without_false_absence() -> None:
    run = _Run("durable-run")
    unselected = decide_artifact_run_lookup(
        run_id=run.run_id,
        matches=(run,),
        source_errors=(),
        discovery_complete=True,
        effective_scope_complete=False,
        exact_source_authorized=False,
    )
    selected = decide_artifact_run_lookup(
        run_id=run.run_id,
        matches=(run,),
        source_errors=(),
        discovery_complete=True,
        effective_scope_complete=False,
        exact_source_authorized=True,
    )

    assert unselected.status_code == 200
    assert unselected.body is None
    assert unselected.selected is run
    assert unselected.source_resolution_complete is False
    assert selected.status_code == 200
    assert selected.selected is run
    assert selected.source_resolution_complete is True


def test_lookup_fails_closed_when_server_source_tuple_is_incomplete() -> None:
    decision = decide_artifact_run_lookup(
        run_id="missing-source-run",
        matches=(_Run("missing-source-run", project_id=""),),
        source_errors=(),
        discovery_complete=True,
        effective_scope_complete=True,
        exact_source_authorized=False,
    )

    assert decision.status_code == 503
    assert decision.selected is None
    assert decision.body is not None
    assert decision.body["error"]["code"] == "artifact_search_incomplete"
    assert decision.body["source_errors"][0]["code"] == (
        "artifact_source_identity_incomplete"
    )


@pytest.mark.parametrize(
    ("complete", "status_code", "error_code"),
    [
        (True, 404, "run_not_discovered"),
        (False, 503, "artifact_search_incomplete"),
    ],
)
def test_absence_is_404_only_after_complete_effective_discovery(
    complete: bool, status_code: int, error_code: str
) -> None:
    decision = decide_artifact_run_lookup(
        run_id="missing-run",
        matches=(),
        source_errors=(),
        discovery_complete=complete,
        effective_scope_complete=complete,
        exact_source_authorized=False,
    )
    assert decision.status_code == status_code
    assert decision.body is not None
    assert decision.body["error"]["code"] == error_code


def test_bootstrap_ships_the_shared_policy_without_obsolete_route_copy() -> None:
    script = render_shipped_backend_install()
    source = artifact_routes.__file__
    assert source is not None
    assert "ARTIFACT_ROUTE_POLICY_CONTRACT" in script
    assert "def build_artifact_run_list_response" in script
    assert "def build_artifact_run_detail_response" in script
    assert "def decide_artifact_run_lookup" in script
    assert "cold_seed" not in script
    assert "def register_artifact_routes" not in script
    assert "class ArtifactRouteDeps" not in script


def test_rendered_backend_imports_and_calls_the_shipped_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Render the VM backend and guard its shared-policy wiring end to end."""
    from npa.cli import agent as agent_module

    captured: dict[str, str] = {}

    class _Ssh:
        def upload_private_text(self, content: str, remote_path: str) -> None:
            if "npa-agent-bootstrap" in remote_path:
                captured["setup"] = content

        def run_or_raise(self, _command: str, **_kwargs) -> None:
            return None

        def run(self, _command: str) -> None:
            return None

    monkeypatch.setattr(agent_module, "SSHClient", lambda config: _Ssh())
    monkeypatch.setattr(
        agent_module,
        "resolve_ssh_config",
        lambda **_kwargs: SimpleNamespace(ssh={}),
    )
    monkeypatch.setattr(agent_module, "_stage_agent_npa_source", lambda *_args: None)
    agent_module._bootstrap_agent_stack(
        host="203.0.113.50",
        ssh_user="operator",
        ssh_key_path=str(tmp_path / "synthetic-key"),
        project_alias="synthetic",
        project_id="project-synthetic",
        tenant_id="tenant-synthetic",
        region="region-synthetic",
        auth_user="agent-user",
        auth_password="synthetic-password",
        agent_port=8088,
        backend_port=8787,
        rerun_port=9090,
        llm_model=agent_module.DEFAULT_LLM_MODEL,
        llm_models=agent_module.DEFAULT_LLM_MODELS,
        tf_api_key="",
        nebius_ai_key="",
        public_https=True,
    )
    setup = captured["setup"]

    def _extract(remote_path: str) -> str:
        match = re.search(
            r"cat <<'PY' \| sudo tee "
            + re.escape(remote_path)
            + r" >/dev/null\n(?P<body>.*?)\nPY\n",
            setup,
            flags=re.DOTALL,
        )
        assert match, f"bootstrap does not write {remote_path}"
        return match.group("body")

    backend = _extract("/opt/npa-agent/backend.py")
    shipped = _extract("/opt/npa-agent/agent_backend/artifact_routes.py")
    backend_tree = ast.parse(backend)
    expected = {
        "artifact_run_snapshot_scope",
        "build_artifact_run_detail_response",
        "build_artifact_run_list_response",
        "decide_artifact_run_lookup",
    }
    imported = {
        alias.name
        for node in ast.walk(backend_tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "agent_backend.artifact_routes"
        for alias in node.names
    }
    called = {
        node.func.id
        for node in ast.walk(backend_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    locally_redefined = {
        node.name
        for node in ast.walk(backend_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert expected <= imported
    assert expected <= called
    assert expected.isdisjoint(locally_redefined)

    rendered_module = tmp_path / "rendered_artifact_routes.py"
    rendered_module.write_text(shipped, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(
        "rendered_artifact_routes", rendered_module
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        assert module.ARTIFACT_ROUTE_POLICY_CONTRACT == (
            ARTIFACT_ROUTE_POLICY_CONTRACT
        )
        assert expected <= set(vars(module))
    finally:
        sys.modules.pop(spec.name, None)
