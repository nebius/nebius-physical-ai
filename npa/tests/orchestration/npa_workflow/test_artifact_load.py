from __future__ import annotations

from pathlib import Path

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


def test_load_posts_exact_uri_then_verifies_and_persists(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _patch_agent(monkeypatch, tmp_path)
    artifact = "s3://bucket/physical-ai-data-factory/paidf-1/reports/sim2real.rrd"
    client = FakeS3({artifact.removeprefix("s3://bucket/")})
    requests: list[tuple[str, str, dict | None]] = []

    def request(method: str, url: str, **kwargs):  # noqa: ANN001, ANN202
        requests.append((method, url, kwargs.get("json")))
        if method == "POST":
            return Response(200, {"ok": True})
        if len(requests) == 1:
            return Response(200, {"artifact_uri": "", "rerun_ready": False})
        return Response(
            200,
            {"artifact_uri": artifact, "artifact_render": "rerun", "rerun_ready": True},
        )

    result = load_final_artifact_into_agent(
        project="demo",
        run_id="paidf-1",
        run_prefix_uri="s3://bucket/physical-ai-data-factory/paidf-1",
        storage_client=client,
        http_request=request,
    )

    assert result.status == "verified"
    assert result.posted is True
    assert requests[1] == (
        "POST",
        "https://agent.invalid/api/sim-viz/load-artifact",
        {"s3_uri": artifact},
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
        return Response(
            200,
            {"artifact_uri": artifact, "artifact_render": "rerun", "rerun_ready": True},
        )

    result = load_final_artifact_into_agent(
        project="demo",
        run_id="paidf-1",
        run_prefix_uri="s3://bucket/physical-ai-data-factory/paidf-1",
        storage_client=client,
        http_request=request,
    )

    assert result.verified is True
    assert result.posted is False
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
