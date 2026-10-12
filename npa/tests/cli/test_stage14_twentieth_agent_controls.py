from __future__ import annotations

import hashlib
import importlib.util
import io
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError

fastapi = pytest.importorskip("fastapi")
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from npa.cli import agent_stage_runtime  # noqa: E402
from npa.agent_backend.shipping import SHIPPED_BACKEND_MODULES  # noqa: E402
from npa.agent_backend.foxglove_routes import (  # noqa: E402
    FoxgloveDeps,
    register_foxglove_routes,
)
from npa.agent_backend.publication_reader import PublicationConflict  # noqa: E402
from npa.workflows.sim2real import publication  # noqa: E402


def _capture_setup_script(monkeypatch: pytest.MonkeyPatch) -> str:
    from npa.cli import agent as agent_module

    captured: dict[str, str] = {}

    class DummySsh:
        def upload_file(self, local_path: str, remote_path: str) -> None:
            if "npa-agent-bootstrap" in remote_path:
                captured["setup_script"] = Path(local_path).read_text(encoding="utf-8")

        def upload_private_text(self, content: str, remote_path: str) -> None:
            if "npa-agent-bootstrap" in remote_path:
                captured["setup_script"] = content

        def run_or_raise(self, _command: str, **_kwargs) -> None:
            return None

        def run(self, _command: str) -> None:
            return None

    monkeypatch.setattr(agent_module, "SSHClient", lambda config: DummySsh())
    monkeypatch.setattr(
        agent_module, "resolve_ssh_config", lambda **_kwargs: SimpleNamespace(ssh={})
    )
    _bootstrap_test_agent(agent_module)
    return captured["setup_script"]


def _bootstrap_test_agent(agent_module) -> None:
    agent_module._bootstrap_agent_stack(
        host="203.0.113.50",
        ssh_user="ubuntu",
        ssh_key_path="unit-test-ssh-key",
        project_alias="smoke",
        project_id="project-id",
        tenant_id="tenant-id",
        region="us-central1",
        auth_user="npa",
        auth_password="password",
        agent_port=8088,
        backend_port=8787,
        rerun_port=9090,
        llm_model=agent_module.DEFAULT_LLM_MODEL,
        llm_models=agent_module.DEFAULT_LLM_MODELS,
        tf_api_key="",
        nebius_ai_key="",
        public_https=True,
    )


def _import_rendered_backend(monkeypatch, tmp_path, *, module_name: str):
    setup_script = _capture_setup_script(monkeypatch)

    def extract(remote_path: str) -> str:
        match = re.search(
            r"cat <<'PY' \| sudo tee "
            + re.escape(remote_path)
            + r" >/dev/null\n(.*?)\nPY\n",
            setup_script,
            flags=re.DOTALL,
        )
        assert match, f"bootstrap does not write {remote_path}"
        return match.group(1)

    package = tmp_path / "agent_backend"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    for name in SHIPPED_BACKEND_MODULES:
        (package / f"{name}.py").write_text(
            extract(f"/opt/npa-agent/agent_backend/{name}.py"), encoding="utf-8"
        )
    backend_path = tmp_path / "backend.py"
    backend_path.write_text(extract("/opt/npa-agent/backend.py"), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location(module_name, backend_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_ROOT = "s3://demo-bucket/sim2real/run-a"
_TRANSACTION_ID = "generation-a"
_LOCK_URI = f"{_ROOT}/reports/.sim2real-publication.json"
_CANONICAL_REPORT = f"{_ROOT}/reports/sim2real-report.json"
_IMMUTABLE_REPORT = (
    f"{_ROOT}/reports/generations/{_TRANSACTION_ID}/sim2real-report.json"
)
_CANONICAL_RRD = f"{_ROOT}/reports/sim2real.rrd"
_IMMUTABLE_RRD = f"{_ROOT}/reports/generations/{_TRANSACTION_ID}/sim2real.rrd"
_CANONICAL_MCAP = f"{_ROOT}/reports/sim2real.mcap"
_CANONICAL_STAGE14 = f"{_ROOT}/components/stage_14.json"
_STAGE14_BYTES = b'{"stage":14}\n'
_IMMUTABLE_STAGE14 = (
    f"{_ROOT}/components/history/stage_14/"
    f"{hashlib.sha256(_STAGE14_BYTES).hexdigest()}.json"
)


def _present(uri: str, immutable_uri: str, payload: bytes) -> dict[str, object]:
    return {
        "uri": uri,
        "state": "present",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "immutable_uri": immutable_uri,
    }


def _complete_journal(
    *,
    report: bytes = b'{"visualization":{"source":"committed"}}\n',
    rrd: bytes = b"committed-rrd",
) -> bytes:
    objects = [
        _present(_CANONICAL_REPORT, _IMMUTABLE_REPORT, report),
        _present(_CANONICAL_RRD, _IMMUTABLE_RRD, rrd),
        {"uri": _CANONICAL_MCAP, "state": "absent"},
        _present(_CANONICAL_STAGE14, _IMMUTABLE_STAGE14, _STAGE14_BYTES),
    ]
    return publication._journal_bytes(
        transaction_id=_TRANSACTION_ID,
        attempt_id="a" * 32,
        state="committed",
        objects=objects,
    )


def _artifact(module, *, bucket: str, prefix: str, run_id: str):
    key = f"{prefix}/{run_id}/input/frame.png"
    return module.Artifact(
        run_id=run_id,
        key=key,
        s3_uri=f"s3://{bucket}/{key}",
        size=5,
        last_modified="2026-10-02T00:00:00Z",
        render="image",
        inline=True,
        namespace=prefix,
        relative_key="input/frame.png",
    )


def test_report_summary_read_rejects_post_head_same_size_replacement() -> None:
    committed = b'{"visualization":{"source":"committed"}}'
    replaced = b'{"visualization":{"source":"tampered!"}}'
    assert len(committed) == len(replaced)

    class S3:
        def get_object(self, *, Bucket: str, Key: str):
            del Bucket, Key
            return {"Body": io.BytesIO(replaced)}

    with pytest.raises(PublicationConflict, match="bytes changed"):
        agent_stage_runtime._read_bounded_json_object(
            S3(),
            "demo-bucket",
            "sim2real/run-a/reports/generations/g/report.json",
            expected_sha256=hashlib.sha256(committed).hexdigest(),
            expected_size=len(committed),
        )


def test_rendered_publication_conflicts_are_409_on_all_consumer_routes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module_name = "npa_rendered_twentieth_consumer_status"
    module = _import_rendered_backend(monkeypatch, tmp_path, module_name=module_name)
    bucket = "bucket-test"
    project = "project-test"
    prefix = "sim2real"
    run_id = "run-a"
    artifact = _artifact(module, bucket=bucket, prefix=prefix, run_id=run_id)

    def conflict(*_args, **_kwargs):
        raise module.PublicationConflict("publishing")

    monkeypatch.setattr(
        module,
        "_agent_artifact_s3_client",
        lambda: (object(), {"bucket": bucket, "prefix": prefix}),
    )
    monkeypatch.setattr(module, "_agent_access_report", lambda: {})
    monkeypatch.setattr(
        module,
        "_load_selected_run_artifacts",
        lambda **_kwargs: (bucket, project, prefix, [artifact]),
    )
    monkeypatch.setattr(module, "_committed_publication_artifacts", conflict)
    kwargs = {
        "resource_bucket": bucket,
        "project_id": project,
        "resolved_prefix": prefix,
        "source_selected": True,
    }
    try:
        for call in (
            lambda: module.artifacts_stage(run_id, **kwargs),
            lambda: module.fiftyone_dataset(run_id, **kwargs),
            lambda: module.artifacts_run_provenance(run_id, **kwargs),
        ):
            with pytest.raises(module.HTTPException) as exc_info:
                call()
            assert exc_info.value.status_code == 409
    finally:
        sys.modules.pop(module_name, None)


def test_rendered_legacy_consumers_recheck_journal_after_reads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module_name = "npa_rendered_twentieth_legacy_fence"
    module = _import_rendered_backend(monkeypatch, tmp_path, module_name=module_name)
    bucket = "bucket-test"
    project = "project-test"
    prefix = "sim2real"
    run_id = "run-a"
    artifact = _artifact(module, bucket=bucket, prefix=prefix, run_id=run_id)

    monkeypatch.setattr(
        module,
        "_agent_artifact_s3_client",
        lambda: (object(), {"bucket": bucket, "prefix": prefix}),
    )
    monkeypatch.setattr(module, "_agent_access_report", lambda: {})
    monkeypatch.setattr(
        module,
        "_load_selected_run_artifacts",
        lambda **_kwargs: (bucket, project, prefix, [artifact]),
    )
    monkeypatch.setattr(
        module,
        "_committed_publication_artifacts",
        lambda *_args, **_kwargs: (
            [artifact],
            None,
            artifact,
            [artifact.key],
            object(),
        ),
    )

    def conflict(*_args, **_kwargs):
        raise module.PublicationConflict("first journal appeared")

    monkeypatch.setattr(
        module,
        "_assert_legacy_publication_snapshot_still_unjournaled",
        conflict,
    )
    kwargs = {
        "resource_bucket": bucket,
        "project_id": project,
        "resolved_prefix": prefix,
        "source_selected": True,
    }
    try:
        for call in (
            lambda: module.artifacts_stage(run_id, **kwargs),
            lambda: module.fiftyone_dataset(run_id, **kwargs),
            lambda: module.artifacts_run_provenance(run_id, **kwargs),
        ):
            with pytest.raises(module.HTTPException) as exc_info:
                call()
            assert exc_info.value.status_code == 409
    finally:
        sys.modules.pop(module_name, None)


def test_rendered_exact_source_storage_failure_cannot_return_stale_history(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module_name = "npa_rendered_twentieth_stale_history"
    module = _import_rendered_backend(monkeypatch, tmp_path, module_name=module_name)
    bucket = "bucket-test"
    project = "project-test"
    prefix = "sim2real"
    run_id = "run-a"
    run_ref = module.encode_run_ref(bucket, prefix, run_id)
    artifact = _artifact(module, bucket=bucket, prefix=prefix, run_id=run_id)
    state = {
        "sim_viz_runs": {
            run_ref: {
                "run_id": run_id,
                "artifact_run_ref": run_ref,
                "project_id": project,
                "bucket": bucket,
                "resolved_prefix": prefix,
                "stage": "stale-history",
                "rrd_uri": "file:///opt/npa-agent/recordings/stale.rrd",
            }
        },
        "sim2real_runs": {},
    }

    monkeypatch.setattr(module, "_load_session_run_if_known", lambda **_kwargs: None)
    monkeypatch.setattr(
        module,
        "_agent_artifact_s3_client",
        lambda: (object(), {"bucket": bucket, "prefix": prefix}),
    )
    monkeypatch.setattr(
        module,
        "_load_selected_run_artifacts",
        lambda **_kwargs: (bucket, project, prefix, [artifact]),
    )
    monkeypatch.setattr(
        module,
        "_committed_publication_artifacts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ClientError(
                {"Error": {"Code": "ServiceUnavailable", "Message": "retry"}},
                "GetObject",
            )
        ),
    )
    monkeypatch.setattr(module, "_load_state", lambda: state)
    monkeypatch.setattr(module, "_save_state", lambda _state: None)
    try:
        with pytest.raises(module.HTTPException) as exc_info:
            module.sim_viz_load_run(
                {
                    "run_id": run_id,
                    "run_ref": run_ref,
                    "resource_bucket": bucket,
                    "project_id": project,
                    "resolved_prefix": prefix,
                    "source_selected": True,
                }
            )
        assert exc_info.value.status_code == 502
    finally:
        sys.modules.pop(module_name, None)


def test_rendered_artifact_content_never_serves_post_verification_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module_name = "npa_rendered_twentieth_byte_binding"
    module = _import_rendered_backend(monkeypatch, tmp_path, module_name=module_name)
    bucket = "demo-bucket"
    prefix = "sim2real"
    run_id = "run-a"
    committed = b"GOOD-COMMITTED-BYTES"
    replacement = b"EVIL-SWAPPED--BYTES!"
    assert len(replacement) == len(committed)
    journal = _complete_journal(rrd=committed)
    journal_key = _LOCK_URI.removeprefix(f"s3://{bucket}/")
    immutable_key = _IMMUTABLE_RRD.removeprefix(f"s3://{bucket}/")

    class Store:
        def get_object(self, *, Bucket: str, Key: str, **_kwargs):
            assert Bucket == bucket
            if Key == journal_key:
                return {"Body": io.BytesIO(journal), "ETag": '"journal"'}
            assert Key == immutable_key
            return {"Body": io.BytesIO(replacement), "ContentLength": len(replacement)}

    artifact = module.Artifact(
        run_id=run_id,
        key=immutable_key,
        s3_uri=_IMMUTABLE_RRD,
        size=len(replacement),
        last_modified="2026-10-02T00:00:00Z",
        render="rerun",
        inline=True,
        namespace=prefix,
        relative_key=immutable_key.removeprefix(f"{prefix}/{run_id}/"),
    )
    monkeypatch.setattr(
        module,
        "_authorized_artifact_content",
        lambda *_args, **_kwargs: (Store(), run_id, bucket, artifact),
    )
    monkeypatch.setattr(module, "_begin_agent_artifact_access", lambda: None)
    monkeypatch.setattr(module, "_end_agent_artifact_access", lambda: None)
    client = TestClient(module.app)
    try:
        response = client.get(
            "/artifacts/content",
            params={
                "run_id": run_id,
                "run_ref": "selected",
                "key": immutable_key,
                "project_id": "project-test",
                "resource_bucket": bucket,
                "resolved_prefix": prefix,
                "source_selected": "true",
            },
        )
        assert response.status_code == 409
        assert replacement not in response.content
    finally:
        sys.modules.pop(module_name, None)


def test_rendered_nonpublication_content_pins_authorized_etag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module_name = "npa_rendered_twentieth_etag_binding"
    module = _import_rendered_backend(monkeypatch, tmp_path, module_name=module_name)
    bucket = "demo-bucket"
    prefix = "sim2real"
    run_id = "run-a"
    artifact = module.Artifact(
        run_id=run_id,
        key=f"{prefix}/{run_id}/input/frame.png",
        s3_uri=f"s3://{bucket}/{prefix}/{run_id}/input/frame.png",
        size=5,
        last_modified="2026-10-02T00:00:00Z",
        render="image",
        inline=True,
        namespace=prefix,
        relative_key="input/frame.png",
        source_etag='"authorized-etag"',
    )

    class Store:
        def get_object(self, **kwargs):
            assert kwargs["IfMatch"] == '"authorized-etag"'
            raise ClientError(
                {
                    "Error": {
                        "Code": "PreconditionFailed",
                        "Message": "object changed",
                    },
                    "ResponseMetadata": {"HTTPStatusCode": 412},
                },
                "GetObject",
            )

    monkeypatch.setattr(
        module,
        "_authorized_artifact_content",
        lambda *_args, **_kwargs: (Store(), run_id, bucket, artifact),
    )
    monkeypatch.setattr(module, "_begin_agent_artifact_access", lambda: None)
    monkeypatch.setattr(module, "_end_agent_artifact_access", lambda: None)
    client = TestClient(module.app)
    try:
        response = client.get(
            "/artifacts/content",
            params={
                "run_id": run_id,
                "run_ref": "selected",
                "key": artifact.key,
                "project_id": "project-test",
                "resource_bucket": bucket,
                "resolved_prefix": prefix,
                "source_selected": "true",
            },
        )
        assert response.status_code == 409
        assert b"object changed" not in response.content
    finally:
        sys.modules.pop(module_name, None)


def test_foxglove_preserves_publication_conflict_status(tmp_path: Path) -> None:
    state = {"sim_viz": {"run_id": "run-a"}}

    def conflict(**_kwargs):
        raise PublicationConflict("publishing")

    app = FastAPI()
    register_foxglove_routes(
        app,
        FoxgloveDeps(
            load_state=lambda: state,
            save_state=lambda _state: None,
            record_run=lambda _state, _viz: None,
            foxglove_config=lambda *_args, **_kwargs: {},
            load_artifact=lambda _body: {"ok": True},
            convert_run=lambda **_kwargs: None,
            now_iso=lambda: "2026-10-02T00:00:00Z",
            validate_run_id=lambda value: value,
            data_dir=tmp_path / "data",
            runs_dir=tmp_path / "runs",
            prepare_canonical_mcap=conflict,
        ),
        HTTPException,
    )

    response = TestClient(app).post(
        "/foxglove/convert-run",
        json={"run_id": "run-a"},
    )

    assert response.status_code == 409
