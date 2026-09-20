from __future__ import annotations

import json
from pathlib import Path

import pytest

from npa.clients import config
from npa.orchestration.npa_workflow.artifact_load import (
    discover_final_rerun_artifact,
    load_final_artifact_into_agent,
)
from npa.orchestration.npa_workflow.submission_state import submission_state_path


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
        return [
            {"Contents": [{"Key": key} for key in self.keys if key.startswith(Prefix)]}
        ]


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
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / ".npa/config.yaml")
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


def test_missing_agent_is_partial_not_workflow_failure(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    import npa.cli.agent as agent

    monkeypatch.setattr(agent, "resolve_project_agents", lambda _project: {})
    client = FakeS3({"physical-ai-data-factory/paidf-1/reports/sim2real.rrd"})

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


def test_storage_failure_detail_is_redacted_before_return_and_persistence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / ".npa/config.yaml")
    caplog.set_level("DEBUG", logger="npa.orchestration.npa_workflow.artifact_load")
    query_secret = "synthetic-list-query"
    assignment_secret = "synthetic-list-assignment"
    bearer_secret = "synthetic-list-bearer"
    plain_credential = "hunter2"

    class FailingListing:
        s3: object

        def __init__(self) -> None:
            self.s3 = self

        def head_object(self, **_kwargs):  # noqa: ANN201
            raise RuntimeError(
                "exact artifact unavailable at "
                f"s3://bucket/exact?signature={query_secret} "
                f"login failed for {plain_credential}"
            )

        def get_paginator(self, _name: str):  # noqa: ANN201
            return self

        def paginate(self, **_kwargs):  # noqa: ANN201
            raise RuntimeError(
                "listing failed at "
                f"s3://bucket/reports?signature={query_secret} "
                f'{{"aws_secret_access_key":"{assignment_secret}"}} '
                f"Bearer {bearer_secret} login failed for {plain_credential}"
            )

    result = load_final_artifact_into_agent(
        project="demo",
        run_id="paidf-storage-redaction",
        run_prefix_uri="s3://bucket/paidf-storage-redaction",
        storage_client=FailingListing(),
        credential_values={"AWS_SECRET_ACCESS_KEY": plain_credential},
    )

    persisted = json.loads(
        submission_state_path("demo", "paidf-storage-redaction").read_text()
    )
    exposed = f"{result.to_dict()}\n{persisted}\n{caplog.text}"
    assert result.status == "partial"
    assert query_secret not in exposed
    assert assignment_secret not in exposed
    assert bearer_secret not in exposed
    assert plain_credential not in exposed


def test_agent_failure_detail_redacts_loaded_auth_before_return_and_persistence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / ".npa/config.yaml")
    _patch_agent(monkeypatch, tmp_path)
    artifact = "s3://bucket/paidf-agent-redaction/reports/sim2real.rrd"
    client = FakeS3({artifact.removeprefix("s3://bucket/")})
    query_secret = "synthetic-agent-query"
    assignment_secret = "synthetic-agent-assignment"
    bearer_secret = "synthetic-agent-bearer"

    def request(*_args, **_kwargs):
        raise RuntimeError(
            "agent failed at "
            f"https://agent.invalid/status?token={query_secret} "
            f"authorization={assignment_secret} "
            f"Bearer {bearer_secret} do-not-print"
        )

    result = load_final_artifact_into_agent(
        project="demo",
        run_id="paidf-agent-redaction",
        run_prefix_uri="s3://bucket/paidf-agent-redaction",
        storage_client=client,
        http_request=request,
    )

    persisted = json.loads(
        submission_state_path("demo", "paidf-agent-redaction").read_text()
    )
    exposed = f"{result.to_dict()}\n{persisted}"
    assert result.status == "partial"
    assert query_secret not in exposed
    assert assignment_secret not in exposed
    assert bearer_secret not in exposed
    assert "do-not-print" not in exposed


@pytest.mark.parametrize("run_prefix_uri", ["", "s3://bucket/paidf-1"])
def test_optional_handoff_keeps_workflow_success_when_receipt_is_corrupt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run_prefix_uri: str,
) -> None:
    from npa.cli.workbench.workflow import _load_paidf_artifact

    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / ".npa/config.yaml")
    secret = "synthetic-receipt-secret"
    path = submission_state_path("demo", "paidf-1")
    path.parent.mkdir(parents=True)
    body = f'{{"aws_secret_access_key":"{secret}",'.encode()
    path.write_bytes(body)
    if run_prefix_uri:
        monkeypatch.setattr(
            "npa.orchestration.npa_workflow.src_staging._storage_client",
            lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("synthetic storage failure")
            ),
        )

    result = _load_paidf_artifact(
        project="demo",
        run_id="paidf-1",
        run_prefix_uri=run_prefix_uri,
    )

    assert result["status"] == "partial"
    assert "receipt_warning" in result
    assert secret not in str(result)
    assert path.read_bytes() == body


def test_optional_handoff_redacts_storage_failure_detail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from npa.cli.workbench.workflow import _load_paidf_artifact

    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / ".npa/config.yaml")
    query_secret = "synthetic-query-value"
    assignment_secret = "synthetic-assignment-value"
    credential_secret = "synthetic-credential-value"
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.src_staging._storage_client",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError(
                "storage failed at "
                f"https://storage.invalid/object?signature={query_secret} "
                f"custom_secret={assignment_secret} {credential_secret}"
            )
        ),
    )
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.submission_state.update_submission_state",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError(f"receipt failed for password {credential_secret}")
        ),
    )

    result = _load_paidf_artifact(
        project="demo",
        run_id="paidf-redaction",
        run_prefix_uri="s3://bucket/paidf-redaction",
        credential_values={"AWS_SECRET_ACCESS_KEY": credential_secret},
    )

    detail = str(result["detail"])
    receipt_warning = str(result["receipt_warning"])
    assert result["status"] == "partial"
    assert query_secret not in detail
    assert assignment_secret not in detail
    assert credential_secret not in detail
    assert "https://storage.invalid/object?<redacted>" in detail
    assert "custom_secret=<redacted>" in detail
    assert credential_secret not in receipt_warning


def test_submission_receipt_warning_redacts_failure_context() -> None:
    from npa.cli.workbench.workflow import _submission_receipt_warning

    query_secret = "synthetic-warning-query"
    assignment_secret = "synthetic-warning-assignment"
    token_secret = "hf_syntheticwarningtoken"
    plain_credential = "hunter2"

    warning = _submission_receipt_warning(
        ValueError(
            "write failed at "
            f"https://storage.invalid/receipt?token={query_secret} "
            f"custom_secret={assignment_secret} {token_secret} "
            f"login failed for password {plain_credential}"
        ),
        secrets=(plain_credential,),
    )

    assert query_secret not in warning
    assert assignment_secret not in warning
    assert token_secret not in warning
    assert plain_credential not in warning
    assert "https://storage.invalid/receipt?<redacted>" in warning
    assert "custom_secret=<redacted>" in warning
