from __future__ import annotations

from pathlib import Path

from npa.clients import config
from npa.orchestration.npa_workflow.artifact_load import (
    discover_final_rerun_artifact,
    load_final_artifact_into_agent,
)


class FakeS3:
    def __init__(self, keys: set[str]) -> None:
        self.keys = keys
        self.s3 = self

    def head_object(self, *, Bucket: str, Key: str):  # noqa: ANN201
        if Key not in self.keys:
            raise KeyError(Key)
        return {"ContentLength": 1}

    def get_paginator(self, name: str):  # noqa: ANN201
        assert name == "list_objects_v2"
        return self

    def paginate(self, *, Bucket: str, Prefix: str):  # noqa: ANN201
        return [{"Contents": [{"Key": key} for key in self.keys if key.startswith(Prefix)]}]


class Response:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def _inventory(artifact: str, *, next_cursor: str = "") -> dict:
    key = artifact.removeprefix("s3://bucket/")
    return {
        "ok": True,
        "run_id": "paidf-1",
        "run_ref": "npa1_paidf_1",
        "project_id": "project-a",
        "bucket": "bucket",
        "resource_bucket": "bucket",
        "resolved_prefix": "physical-ai-data-factory",
        "source_selected": True,
        "artifacts": [] if next_cursor else [{"key": key, "s3_uri": artifact}],
        "next_cursor": next_cursor,
        "truncated": bool(next_cursor),
    }


def _ready_status(artifact: str) -> dict:
    return {
        "run_id": "paidf-1",
        "artifact_uri": artifact,
        "artifact_key": artifact.removeprefix("s3://bucket/"),
        "artifact_render": "rerun",
        "artifact_run_ref": "npa1_paidf_1",
        "project_id": "project-a",
        "bucket": "bucket",
        "resolved_prefix": "physical-ai-data-factory",
        "rerun_ready": True,
    }


def _patch_agent(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    import npa.cli.agent as agent

    auth = tmp_path / "agent.env"
    auth.write_text("AGENT_USER=user\nAGENT_PASSWORD=do-not-print\n", encoding="utf-8")
    monkeypatch.setattr(
        agent,
        "resolve_project_agents",
        lambda _project: {"agent": {"agent_url": "https://agent.invalid"}},
    )
    monkeypatch.setattr(
        agent,
        "_agent_record",
        lambda _project, _name: {
            "agent_url": "https://agent.invalid",
            "auth_secret_path": str(auth),
            "tls_verify": True,
        },
    )
    monkeypatch.setattr(agent, "_record_tls_verify", lambda _record: True)


def test_discovers_exact_nested_paidf_final_artifact() -> None:
    key = "physical-ai-data-factory/paidf-1/reports/sim2real.rrd"
    client = FakeS3({key, "other-run/reports/sim2real.rrd"})

    uri = discover_final_rerun_artifact(
        "s3://bucket/physical-ai-data-factory/paidf-1", client=client
    )

    assert uri == f"s3://bucket/{key}"


def test_load_discovers_exact_source_then_posts_strict_v3_and_persists(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / ".npa/config.yaml")
    _patch_agent(monkeypatch, tmp_path)
    artifact = "s3://bucket/physical-ai-data-factory/paidf-1/reports/sim2real.rrd"
    client = FakeS3({artifact.removeprefix("s3://bucket/")})
    requests: list[tuple[str, str, dict | None]] = []

    def request(method: str, url: str, **kwargs):  # noqa: ANN001, ANN202
        requests.append((method, url, kwargs.get("json")))
        if "/api/artifacts/run/" in url:
            return Response(200, _inventory(artifact))
        if method == "POST":
            return Response(200, {"ok": True, "sim_viz": _ready_status(artifact)})
        if len([item for item in requests if item[1].endswith("/api/sim-viz/status")]) == 1:
            return Response(200, {"artifact_uri": "", "rerun_ready": False})
        return Response(200, _ready_status(artifact))

    result = load_final_artifact_into_agent(
        project="demo",
        run_id="paidf-1",
        run_prefix_uri="s3://bucket/physical-ai-data-factory/paidf-1",
        storage_client=client,
        http_request=request,
    )

    assert result.status == "verified"
    assert result.posted is True
    assert requests[2] == (
        "POST",
        "https://agent.invalid/api/sim-viz/load-artifact",
        {
            "run_id": "paidf-1",
            "run_ref": "npa1_paidf_1",
            "key": "physical-ai-data-factory/paidf-1/reports/sim2real.rrd",
            "project_id": "project-a",
            "resource_bucket": "bucket",
            "resolved_prefix": "physical-ai-data-factory",
            "source_selected": True,
        },
    )
    assert all(
        not payload or "s3_uri" not in payload for _method, _url, payload in requests
    )
    state = (tmp_path / ".npa/workflow-submissions/demo/paidf-1.json").read_text()
    assert artifact in state
    assert "do-not-print" not in state


def test_resume_skips_duplicate_post_when_agent_already_has_artifact(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _patch_agent(monkeypatch, tmp_path)
    artifact = "s3://bucket/physical-ai-data-factory/paidf-1/reports/sim2real.rrd"
    client = FakeS3({artifact.removeprefix("s3://bucket/")})
    methods: list[str] = []

    def request(method: str, _url: str, **_kwargs):  # noqa: ANN202
        methods.append(method)
        if "/api/artifacts/run/" in _url:
            return Response(200, _inventory(artifact))
        return Response(200, _ready_status(artifact))

    result = load_final_artifact_into_agent(
        project="demo",
        run_id="paidf-1",
        run_prefix_uri="s3://bucket/physical-ai-data-factory/paidf-1",
        storage_client=client,
        http_request=request,
    )

    assert result.verified is True
    assert result.posted is False
    assert methods == ["GET", "GET"]


def test_agent_source_ambiguity_fails_closed_without_post(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _patch_agent(monkeypatch, tmp_path)
    artifact = "s3://bucket/physical-ai-data-factory/paidf-1/reports/sim2real.rrd"
    client = FakeS3({artifact.removeprefix("s3://bucket/")})
    methods: list[str] = []

    def request(method: str, _url: str, **_kwargs):  # noqa: ANN202
        methods.append(method)
        return Response(
            409,
            {
                "ok": False,
                "error": {"code": "ambiguous_run_id"},
                "sources": [{}, {}],
            },
        )

    result = load_final_artifact_into_agent(
        project="demo",
        run_id="paidf-1",
        run_prefix_uri="s3://bucket/physical-ai-data-factory/paidf-1",
        storage_client=client,
        http_request=request,
    )

    assert result.status == "partial"
    assert result.posted is False
    assert "HTTP 409" in result.detail
    assert methods == ["GET"]


def test_missing_agent_is_partial_not_workflow_failure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    import npa.cli.agent as agent

    monkeypatch.setattr(agent, "resolve_project_agents", lambda _project: {})
    client = FakeS3(
        {"physical-ai-data-factory/paidf-1/reports/sim2real.rrd"}
    )

    result = load_final_artifact_into_agent(
        project="demo",
        run_id="paidf-1",
        run_prefix_uri="s3://bucket/physical-ai-data-factory/paidf-1",
        storage_client=client,
    )

    assert result.status == "partial"
    assert "workflow succeeded" in result.detail
    assert result.retry_command == (
        "npa workbench workflow load-artifact paidf-1 --project demo"
    )
