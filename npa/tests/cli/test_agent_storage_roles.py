"""Exercise separate deployment writes and exact-source artifact reads."""

from __future__ import annotations

import ast
import inspect
import importlib.util
import tempfile
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from npa.cli import agent_artifact_sources as sources
from npa.cli import agent_env_files, agent_storage_runtime as storage
from .test_agent_backend_render import _render_backend_body


DEPLOYMENT = (
    "output-bucket",
    "validation/unique",
    "https://output.example",
    "output-access",
    "output-secret",
    "attached-account",
)
READ = (
    "input-bucket",
    "evaluations/real",
    "https://input.example",
    "input-access",
    "input-secret",
    "attached-account",
)


def _read_record():
    return {
        "storage": {
            "bucket": READ[0],
            "endpoint_url": READ[2],
            "aws_access_key_id": READ[3],
            "aws_secret_access_key": READ[4],
        }
    }


def _stage(monkeypatch, *, read=READ, output=DEPLOYMENT):
    captured = {}
    monkeypatch.setattr(
        agent_env_files,
        "_stage_private_text",
        lambda _ssh, **kwargs: captured.update(kwargs),
    )
    agent_env_files._write_agent_s3_env(
        object(),
        bucket=output[0],
        prefix=output[1],
        endpoint=output[2],
        access_key=output[3],
        secret_key=output[4],
        region="test-region",
        artifact_storage=read,
    )
    for line in captured["content"].splitlines():
        key, value = line.split("=", 1)
        monkeypatch.setenv(key, value)
    return captured


def _functions(source, namespace, *names):
    tree = ast.parse(source)
    selected = []
    for name in names:
        matches = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == name
        ]
        assert len(matches) == 1
        matches[0].decorator_list = []
        selected.append(matches[0])
    with tempfile.TemporaryDirectory(prefix="npa-rendered-storage-test-") as directory:
        path = Path(directory) / "rendered_storage.py"
        path.write_text(ast.unparse(ast.Module(body=selected, type_ignores=[])))
        specification = importlib.util.spec_from_file_location("rendered_storage", path)
        module = importlib.util.module_from_spec(specification)
        module.__dict__.update(
            {key: value for key, value in namespace.items() if not key.startswith("__")}
        )
        specification.loader.exec_module(module)
        return vars(module)


@pytest.mark.parametrize("source_project", ["deployment-project", "read-project"])
def test_exact_read_credentials_do_not_change_deployment_writes(
    monkeypatch, source_project
):
    calls = []

    def credentials(project, **kwargs):
        calls.append((project, kwargs))
        return _read_record()

    monkeypatch.setattr(sources, "project_credential_record", credentials)
    read = sources.resolve_configured_artifact_storage_credentials(
        [{"project_id": source_project, "bucket": READ[0], "resolved_prefix": READ[1]}],
        deployment_project_id="deployment-project",
        current=DEPLOYMENT,
    )
    _stage(monkeypatch, read=read)
    seen = []
    monkeypatch.setattr(
        storage, "build_s3_client", lambda **kwargs: seen.append(kwargs) or object()
    )
    _, reader = storage._agent_s3_client()
    _, writer = storage._agent_output_s3_client()
    assert (reader["bucket"], reader["prefix"]) == READ[:2]
    assert (writer["bucket"], writer["prefix"], writer["access_key"]) == (
        DEPLOYMENT[0],
        DEPLOYMENT[1],
        DEPLOYMENT[3],
    )
    assert seen[1]["aws_secret_access_key"] == DEPLOYMENT[4]
    assert os.environ["AWS_ACCESS_KEY_ID"] == DEPLOYMENT[3]
    assert read[5] == DEPLOYMENT[5]
    if source_project != "deployment-project":
        assert seen[0]["aws_secret_access_key"] == READ[4]
        assert calls == [("read-project", {"migrate_legacy": False})]
    else:
        assert seen[0]["aws_secret_access_key"] == DEPLOYMENT[4]
        assert calls == []


def test_clearing_sources_removes_stale_read_credentials(monkeypatch):
    _stage(monkeypatch)
    _stage(monkeypatch, read=None)
    assert storage._agent_s3_settings() == storage._agent_output_s3_settings()
    assert os.environ["NPA_AGENT_ARTIFACT_S3_ACCESS_KEY_ID"] == ""
    assert os.environ["NPA_AGENT_ARTIFACT_S3_SECRET_ACCESS_KEY"] == ""
    captured = {}
    monkeypatch.setattr(
        agent_env_files,
        "_stage_private_text",
        lambda _ssh, **kwargs: captured.update(kwargs),
    )
    agent_env_files._write_agent_artifact_sources_env(object(), artifact_sources=[])
    assert captured["content"] == "NPA_AGENT_ARTIFACT_SOURCES_B64=\n"


def test_configured_reader_cannot_supply_missing_write_credentials(monkeypatch):
    _stage(monkeypatch, output=("", "", "", "", "", "attached-account"))
    with pytest.raises(HTTPException, match="not configured"):
        storage._agent_output_s3_client()
    assert storage._agent_output_s3_client_optional()[0] is None


def test_invalid_source_identity_never_falls_back_to_output_credential(monkeypatch):
    _stage(monkeypatch)
    monkeypatch.setenv("NPA_AGENT_ARTIFACT_S3_SECRET_ACCESS_KEY", "")
    with pytest.raises(HTTPException, match="not configured"):
        storage._agent_s3_client()


@pytest.mark.parametrize(
    "prefix",
    [
        "",
        "evaluations",
        "evaluations/real",
        "evaluations/real/new",
        "/evaluations/real/",
    ],
)
def test_output_prefix_rejects_read_scope_overlap(prefix):
    with pytest.raises(sources.AgentStorageCredentialError, match="overlap"):
        sources.resolve_agent_output_prefix(
            {},
            requested=prefix,
            bucket=READ[0],
            current_prefix="",
            artifact_sources=[
                {
                    "project_id": "read-project",
                    "bucket": READ[0],
                    "resolved_prefix": READ[1],
                }
            ],
        )


@pytest.mark.parametrize(
    "prefix",
    [
        "a/../b",
        "a//b",
        "a\\b",
        "a/%2e%2e/b",
        "a/%252e%252e/b",
        "a%2fb",
        "s3://bucket/prefix",
        "a\nkey",
    ],
)
def test_output_prefix_rejects_malformed_subtrees(prefix):
    with pytest.raises(sources.AgentStorageCredentialError):
        sources.resolve_agent_output_prefix(
            {}, requested=prefix, bucket=READ[0], current_prefix="", artifact_sources=[]
        )


def test_output_scope_normalizes_and_reuses_saved_prefix_without_source_inference():
    read = [
        {"project_id": "read-project", "bucket": READ[0], "resolved_prefix": READ[1]}
    ]
    kwargs = dict(bucket=READ[0], current_prefix="", artifact_sources=read)
    assert (
        sources.resolve_agent_output_prefix(
            {}, requested=" /validation/new/ ", **kwargs
        )
        == "validation/new"
    )
    assert (
        sources.resolve_agent_output_prefix(
            {"output_prefix": "validation/saved"}, requested="", **kwargs
        )
        == "validation/saved"
    )
    assert (
        sources.resolve_agent_output_prefix(
            {}, requested="evaluations/real-other", **kwargs
        )
        == "evaluations/real-other"
    )


def _writer_namespace(source, writes):
    class Client:
        def __init__(self, credentials):
            self.credentials = credentials

        def put_object(self, **kwargs):
            writes.append((self.credentials, kwargs))

    namespace = {
        "os": os,
        "json": json,
        "Path": Path,
        "HTTPException": HTTPException,
        "build_s3_client": lambda **kwargs: Client(kwargs),
    }
    storage_names = [
        node.name
        for node in ast.parse(Path(storage.__file__).read_text()).body
        if isinstance(node, ast.FunctionDef)
    ]
    namespace = _functions(
        source,
        namespace,
        *storage_names,
        "_join_agent_s3_prefix",
        "_state_s3_settings",
        "_state_s3_client",
        "_save_state_to_s3",
        "_persist_chat_session_to_s3",
        "_upload_output_file",
        "_agent_insights_settings",
    )
    _writer_identity(namespace)
    return namespace


def _writer_identity(namespace):
    namespace.update(
        _state_s3_key=lambda: (
            namespace["_state_s3_settings"]()["prefix"] + "/state.json"
        ),
        _sanitize_chat_session_id=lambda value: value,
        _chat_memory_tenant=lambda: "test-tenant",
        _chat_session_key=lambda session, settings: (
            settings["prefix"] + "/chat/" + session + ".json"
        ),
        _chat_memory_uri=lambda session, settings: (
            "s3://" + settings["bucket"] + "/" + session
        ),
        run_id="fixture-run",
    )


def test_rendered_writers_use_deployment_scope_and_reader_uses_source(
    monkeypatch, tmp_path
):
    source = _render_backend_body(monkeypatch)
    _stage(monkeypatch)
    writes = []
    namespace = _writer_namespace(source, writes)
    artifact = tmp_path / "fixture.json"
    artifact.write_text('{"fixture":true}')
    namespace["_save_state_to_s3"]({"fixture": True})
    namespace["_persist_chat_session_to_s3"]({"id": "fixture-session"})
    namespace["_upload_output_file"](artifact, "reports/fixture.json")
    assert len(writes) == 3
    for credentials, request in writes:
        assert credentials["aws_access_key_id"] == DEPLOYMENT[3]
        assert request["Bucket"] == DEPLOYMENT[0]
        assert request["Key"].startswith(DEPLOYMENT[1] + "/")
    assert (
        namespace["_agent_insights_settings"]()["store_uri"]
        == "s3://output-bucket/validation/unique/insights/store"
    )
    reader, settings = namespace["_agent_s3_client"]()
    assert reader.credentials["aws_access_key_id"] == READ[3]
    assert settings["bucket"] == READ[0]


@pytest.mark.parametrize(
    "override",
    [
        "evaluations/real/state",
        "validation",
        "validation/unique-other/state",
        "validation/unique/../real",
        "validation/unique/%2e%2e/real",
        "validation/unique//state",
        "validation/unique\\state",
    ],
)
def test_state_prefix_override_cannot_escape_dedicated_output(monkeypatch, override):
    source = _render_backend_body(monkeypatch)
    _stage(monkeypatch)
    monkeypatch.setenv("NPA_AGENT_STATE_S3_PREFIX", override)
    writes = []
    namespace = _writer_namespace(source, writes)
    namespace["build_s3_client"] = lambda **_kwargs: pytest.fail("storage accessed")
    with pytest.raises(HTTPException, match="inside the deployment output subtree"):
        namespace["_save_state_to_s3"]({"fixture": True})
    assert writes == []


@pytest.mark.parametrize("override", ["", "/validation/unique/custom-state/"])
def test_state_prefix_override_stays_inside_dedicated_output(monkeypatch, override):
    source = _render_backend_body(monkeypatch)
    _stage(monkeypatch)
    monkeypatch.setenv("NPA_AGENT_STATE_S3_PREFIX", override)
    writes = []
    namespace = _writer_namespace(source, writes)
    namespace["_save_state_to_s3"]({"fixture": True})
    [(credentials, request)] = writes
    assert credentials["aws_access_key_id"] == DEPLOYMENT[3]
    assert request["Bucket"] == DEPLOYMENT[0]
    expected = override.strip("/") or "validation/unique/npa-agent/session-state"
    assert request["Key"] == expected + "/state.json"


def test_state_prefix_without_output_subtree_preserves_legacy_override(monkeypatch):
    _stage(monkeypatch, output=(DEPLOYMENT[0], "", *DEPLOYMENT[2:]))
    assert (
        storage._agent_state_s3_prefix("/legacy/session-state/")
        == "legacy/session-state"
    )
    assert storage._agent_state_s3_prefix("") == "npa-agent/session-state"


def _bootstrap_dependencies(monkeypatch, agent, key, record):
    overrides = {
        "_agent_record": lambda *_: record,
        "_resolve_project_alias": lambda _: "deployment",
        "_resolve_record_public_ip": lambda _: record["public_ip"],
        "_resolve_agent_ssh_key": lambda *_a, **_k: str(key),
        "_load_auth_secret": lambda _: ("synthetic-user", "synthetic-password"),
        "_resolve_deploy_llm_credentials": lambda: ("", "test-model"),
        "_resolve_agent_storage_credentials": lambda *_: DEPLOYMENT,
        "current_operation": lambda: None,
        "_persist_agent_service_account_id": lambda *_: None,
    }
    for name, value in overrides.items():
        monkeypatch.setattr(agent, name, value)
    monkeypatch.setattr(
        sources,
        "project_credential_record",
        lambda *_a, **_k: _read_record(),
    )


def _bootstrap_context(monkeypatch, tmp_path, source_project):
    from npa.cli import agent

    key = tmp_path / "ssh-key"
    key.write_text("synthetic")
    read = [
        {"project_id": source_project, "bucket": READ[0], "resolved_prefix": READ[1]}
    ]
    record = {
        "project_id": "deployment-project",
        "tenant_id": "test-tenant",
        "region": "test-region",
        "public_ip": "203.0.113.10",
        "artifact_sources": read,
    }
    _bootstrap_dependencies(monkeypatch, agent, key, record)
    captured = {}

    def converge(**kwargs):
        captured.update(kwargs["bootstrap_kwargs"])
        return SimpleNamespace(evidence={"state": "healthy"}, primary_error=None)

    monkeypatch.setattr(agent, "converge_remote_agent_setup", converge)
    monkeypatch.setattr(
        agent,
        "_store_agent_record",
        lambda _project, _name, value: captured.update(saved_record=value),
    )
    function = inspect.unwrap(agent.bootstrap_cmd)
    defaults = {
        name: parameter.default.default
        if hasattr(parameter.default, "default")
        else parameter.default
        for name, parameter in inspect.signature(function).parameters.items()
    }
    return agent, function, defaults, captured


@pytest.mark.parametrize("source_project", ["deployment-project", "read-project"])
def test_bootstrap_preserves_write_role_and_persists_dedicated_output(
    monkeypatch, tmp_path, source_project
):
    _agent, function, defaults, captured = _bootstrap_context(
        monkeypatch, tmp_path, source_project
    )
    function(
        **{
            **defaults,
            "project": "deployment",
            "name": "test-agent",
            "output_prefix": "/validation/new/",
        }
    )
    assert (
        captured["s3_bucket"],
        captured["s3_prefix"],
        captured["s3_access_key"],
        captured["s3_secret_key"],
    ) == (DEPLOYMENT[0], "validation/new", DEPLOYMENT[3], DEPLOYMENT[4])
    assert captured["service_account_id"] == DEPLOYMENT[5]
    assert captured["artifact_storage"][:2] == READ[:2]
    assert captured["saved_record"]["output_prefix"] == "validation/new"
    assert "credentials" not in captured["saved_record"]


def test_bootstrap_refresh_keeps_refreshed_write_identity_separate(
    monkeypatch, tmp_path
):
    from npa.clients import nebius

    agent, function, defaults, captured = _bootstrap_context(
        monkeypatch, tmp_path, "read-project"
    )
    refreshed = {
        "s3_bucket": "refreshed-output",
        "s3_prefix": "",
        "s3_endpoint": "https://output.example",
        "nebius_api_key": "refreshed-access",
        "nebius_secret_key": "refreshed-secret",
        "service_account_id": DEPLOYMENT[5],
    }
    monkeypatch.setattr(
        nebius, "bootstrap_agent_environment", lambda *_a, **_k: refreshed
    )
    monkeypatch.setattr(
        agent, "_resolve_deploy_storage_credentials", lambda **_kwargs: refreshed
    )
    monkeypatch.setattr(
        agent, "persist_agent_terraform_credentials", lambda *_a, **_k: None
    )
    monkeypatch.setattr(agent, "write_config", lambda *_a, **_k: None)
    function(
        **{
            **defaults,
            "project": "deployment",
            "refresh_credentials": True,
            "output_prefix": "validation/refreshed",
        }
    )
    assert captured["s3_bucket"] == "refreshed-output"
    assert captured["s3_access_key"] == "refreshed-access"
    assert captured["artifact_storage"][0] == READ[0]
    assert captured["artifact_storage"][3] == READ[3]
    assert captured["service_account_id"] == DEPLOYMENT[5]


def _capture_bootstrap_storage(monkeypatch):
    from npa.cli import agent

    original = agent._bootstrap_agent_stack
    profile = {}
    storage_env = {}

    def bootstrap(**kwargs):
        return original(
            **kwargs,
            s3_bucket=DEPLOYMENT[0],
            s3_prefix=DEPLOYMENT[1],
            s3_endpoint=DEPLOYMENT[2],
            s3_access_key=DEPLOYMENT[3],
            s3_secret_key=DEPLOYMENT[4],
            artifact_storage=READ,
        )

    monkeypatch.setattr(agent, "_bootstrap_agent_stack", bootstrap)
    monkeypatch.setattr(
        agent,
        "_write_agent_operator_profile",
        lambda _ssh, **kwargs: profile.update(kwargs),
    )
    monkeypatch.setattr(
        agent, "_write_agent_s3_env", lambda _ssh, **kwargs: storage_env.update(kwargs)
    )
    source = _render_backend_body(monkeypatch)
    return source, profile, storage_env


def test_remote_operator_profile_and_subprocess_keep_write_credentials(monkeypatch):
    import shutil

    source, profile, storage_env = _capture_bootstrap_storage(monkeypatch)
    assert profile["s3_bucket"] == DEPLOYMENT[0]
    assert profile["s3_prefix"] == DEPLOYMENT[1]
    assert profile["s3_access_key"] == DEPLOYMENT[3]
    assert profile["s3_secret_key"] == DEPLOYMENT[4]
    assert storage_env["artifact_storage"] == READ
    _stage(monkeypatch)
    monkeypatch.setenv("TF_VAR_ssh_public_key", "synthetic")
    namespace = {"os": os, "Path": Path, "shutil": shutil, "json": json}
    namespace = _functions(source, namespace, "_agent_command_env")
    environment = namespace["_agent_command_env"]()
    assert environment["AWS_ACCESS_KEY_ID"] == DEPLOYMENT[3]
    assert environment["AWS_SECRET_ACCESS_KEY"] == DEPLOYMENT[4]
    assert environment["NPA_AGENT_S3_PREFIX"] == DEPLOYMENT[1]
    assert not any(key.startswith("NPA_AGENT_ARTIFACT_S3_") for key in environment)


def test_read_selector_does_not_establish_list_or_read_access(monkeypatch):
    from npa.cli import agent_access_runtime as access
    from npa.cli.agent_access import AccessProbeError

    _stage(monkeypatch)

    class DeniedClient:
        def list_objects_v2(self, **_kwargs):
            raise PermissionError("AccessDenied")

        def put_object(self, **_kwargs):
            raise AssertionError("access checks must never write")

    monkeypatch.setattr(storage, "build_s3_client", lambda **_kwargs: DeniedClient())
    client, settings = storage._agent_s3_client()
    with pytest.raises(AccessProbeError):
        access._agent_probe_bucket(client, settings["bucket"])


def test_bootstrap_empty_source_file_clears_saved_reader(monkeypatch, tmp_path):
    _agent, function, defaults, captured = _bootstrap_context(
        monkeypatch, tmp_path, "read-project"
    )
    source_file = tmp_path / "empty-sources.json"
    source_file.write_text("[]")
    source_file.chmod(0o600)
    function(
        **{
            **defaults,
            "project": "deployment",
            "artifact_source_file": str(source_file),
        }
    )
    assert captured["artifact_storage"] is None
    assert captured["artifact_sources"] == ()
    assert "artifact_sources" not in captured["saved_record"]
    assert captured["s3_access_key"] == DEPLOYMENT[3]


def test_bootstrap_recovery_keeps_explicit_output_prefix(monkeypatch, tmp_path):
    from typer.testing import CliRunner
    from npa.cli import agent

    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(tmp_path / "operations"))
    monkeypatch.setattr(
        agent,
        "resolve_environment",
        lambda _project: SimpleNamespace(
            project_id="example-project",
            tenant_id="example-tenant",
            region="test-region",
        ),
    )
    monkeypatch.setattr(agent, "_agent_record", lambda *_args: {})
    result = CliRunner().invoke(
        agent.app,
        [
            "bootstrap",
            "--project",
            "example",
            "--name",
            "example-agent",
            "--output-prefix",
            "validation/exact",
        ],
    )
    assert result.exit_code == 1
    [journal] = (tmp_path / "operations").glob("*/journal.json")
    argv = json.loads(journal.read_text())["recovery_commands"]["resume_argv"]
    assert argv[argv.index("--output-prefix") + 1] == "validation/exact"


def _configure_owned_viewer(monkeypatch, module, tmp_path):
    monkeypatch.setattr(module, "_record_sim_viz_run", lambda *_args: None)
    monkeypatch.setattr(module, "_restart_rerun_serve", lambda **_kwargs: False)
    monkeypatch.setattr(module, "_rerun_ready_state", lambda **_kwargs: False)
    monkeypatch.setattr(
        module,
        "_ensure_same_run_canonical_recording",
        lambda *_a, **_k: None,
        raising=False,
    )
    for name, value in {
        "RECORDINGS_DIR": tmp_path / "recordings",
        "RECORDING_PATH": tmp_path / "recordings/active.rrd",
        "RRD_PATH": tmp_path / "local.rrd",
    }.items():
        monkeypatch.setattr(module, name, value)


def _owned_storage_client(monkeypatch, module):
    objects, used_credentials = {}, []

    class Client:
        def __init__(self, credentials):
            used_credentials.append(credentials)
            self.credentials = credentials

        def put_object(self, **kwargs):
            objects[kwargs["Bucket"], kwargs["Key"]] = kwargs["Body"]

        def download_file(self, bucket, key, destination):
            assert self.credentials["aws_access_key_id"] == DEPLOYMENT[3]
            Path(destination).write_bytes(objects[bucket, key])

    monkeypatch.setattr(module, "build_s3_client", lambda **kwargs: Client(kwargs))

    return objects, used_credentials


def _owned_output_backend(monkeypatch, tmp_path):
    import sys
    from .test_agent_backend_render import (
        _clear_rendered_agent_backend_modules,
        _import_rendered_backend,
    )

    _clear_rendered_agent_backend_modules()
    module = _import_rendered_backend(
        monkeypatch, tmp_path, module_name="agent_output_storage_test"
    )
    _stage(monkeypatch)
    monkeypatch.setenv("NEBIUS_PROJECT_ID", "deployment-project")
    state = {"sim2real_runs": {"fixture-run": {"artifact_uris": []}}}
    monkeypatch.setattr(module, "_load_state", lambda: state)
    monkeypatch.setattr(module, "_save_state", lambda value: state.update(value))
    _configure_owned_viewer(monkeypatch, module, tmp_path)
    objects, used_credentials = _owned_storage_client(monkeypatch, module)

    def deny_read(**_kwargs):
        raise HTTPException(status_code=403, detail="not a discovered reader artifact")

    monkeypatch.setattr(module, "_authorize_agent_artifact_uri", deny_read)
    monkeypatch.setattr(
        module,
        "_agent_access_report",
        lambda: (_ for _ in ()).throw(
            AssertionError("owned output must not infer a tenant read grant")
        ),
    )
    sys.modules.pop("agent_output_storage_test", None)
    return module, state, objects, used_credentials


def test_actual_rendered_upload_and_load_uses_own_output_credentials(
    monkeypatch, tmp_path
):
    module, state, objects, used_credentials = _owned_output_backend(
        monkeypatch, tmp_path
    )
    # Synthetic unit-test bytes exercise transport; this is not a simulator recording claim.
    fixture = tmp_path / "fixture.rrd"
    fixture.write_bytes(b"RRF2-unit-test-transport")
    namespace = vars(module).copy()
    namespace["run_id"] = "fixture-run"
    namespace = _functions(
        (tmp_path / "backend.py").read_text(), namespace, "_upload_output_file"
    )
    uri = namespace["_upload_output_file"](fixture, "reports/fixture.rrd")
    state["sim2real_runs"]["fixture-run"]["artifact_uris"] = [uri]
    response = module.sim_viz_load_run({"run_id": "fixture-run", "rrd_uri": uri})
    assert response["ok"] is True
    assert module.RECORDING_PATH.read_bytes() == fixture.read_bytes()
    viz = state["sim_viz"]
    assert (viz["bucket"], viz["project_id"], viz["resolved_prefix"]) == (
        DEPLOYMENT[0],
        "deployment-project",
        DEPLOYMENT[1] + "/sim2real-b",
    )
    assert viz["artifact_uri"] == uri
    assert len(objects) == 1
    assert all(item["aws_access_key_id"] == DEPLOYMENT[3] for item in used_credentials)


@pytest.mark.parametrize(
    "mutation",
    ["unknown-run", "unrecorded-uri", "outside-output", "different-reference"],
)
def test_recorded_output_cannot_grant_another_run_or_object(monkeypatch, mutation):
    _stage(monkeypatch)
    monkeypatch.setenv("NEBIUS_PROJECT_ID", "deployment-project")
    uri = "s3://output-bucket/validation/unique/sim2real-b/fixture-run/reports/fixture.rrd"
    state = {"sim2real_runs": {"fixture-run": {"artifact_uris": [uri]}}}
    arguments = dict(state=state, run_id="fixture-run", uri=uri, run_ref="")
    if mutation == "unknown-run":
        arguments["run_id"] = "other-run"
    elif mutation == "unrecorded-uri":
        arguments["uri"] = uri.replace("fixture.rrd", "other.rrd")
    elif mutation == "outside-output":
        arguments["uri"] = "s3://output-bucket/unrelated/private.rrd"
        state["sim2real_runs"]["fixture-run"]["artifact_uris"] = [arguments["uri"]]
    else:
        arguments["run_ref"] = storage.encode_run_ref(
            DEPLOYMENT[0], DEPLOYMENT[1] + "/sim2real-b", "other-run"
        )
    if mutation in {"unknown-run", "unrecorded-uri"}:
        assert storage._recorded_output_source(**arguments) is None
    else:
        with pytest.raises(HTTPException):
            storage._recorded_output_source(**arguments)


def _pending_bootstrap_convergence(monkeypatch, agent):
    from npa.cli.agent_setup_convergence import converge_remote_agent_setup

    events = []
    agent._agent_record("deployment", "test-agent")["setup_state"] = (
        "remote_bootstrap_pending"
    )
    monkeypatch.setattr(
        agent, "converge_remote_agent_setup", converge_remote_agent_setup
    )
    monkeypatch.setattr(
        agent,
        "_bootstrap_agent_stack",
        lambda **kwargs: events.append(("bootstrap", kwargs)),
    )
    monkeypatch.setattr(
        agent,
        "_reconcile_agent_setup",
        lambda **kwargs: events.append(("reconcile", kwargs)) or {"state": "healthy"},
    )
    monkeypatch.setattr(
        agent,
        "_store_agent_record",
        lambda _project, _name, record: events.append(("persist", record)),
    )
    return events


def _storage_override_arguments(tmp_path, override, defaults):
    arguments = {**defaults, "project": "deployment", "name": "test-agent"}
    if override == "output":
        arguments["output_prefix"] = "validation/new"
    elif override == "source":
        source_file = tmp_path / "new-source.json"
        source_file.write_text(
            json.dumps(
                [
                    {
                        "project_id": "read-project",
                        "bucket": READ[0],
                        "resolved_prefix": "new/evidence",
                    }
                ]
            )
        )
        source_file.chmod(0o600)
        arguments["artifact_source_file"] = str(source_file)
    return arguments


@pytest.mark.parametrize("override", ["output", "source", "none"])
def test_explicit_storage_scope_restage_precedes_healthy_pending_adoption(
    monkeypatch, tmp_path, override
):
    agent, function, defaults, _captured = _bootstrap_context(
        monkeypatch, tmp_path, "read-project"
    )
    events = _pending_bootstrap_convergence(monkeypatch, agent)
    arguments = _storage_override_arguments(tmp_path, override, defaults)
    function(**arguments)
    bootstraps = [
        (index, value)
        for index, (phase, value) in enumerate(events)
        if phase == "bootstrap"
    ]
    if override == "none":
        assert bootstraps == []
        assert events[0][0] == "reconcile"
        return
    [index_and_call] = bootstraps
    index, call = index_and_call
    assert call["resume_services"] is False
    final_records = [
        (index, value)
        for index, (phase, value) in enumerate(events)
        if phase == "persist" and value.get("setup_state") == "healthy"
    ]
    assert len(final_records) == 1 and index < final_records[0][0]
    if override == "output":
        assert (
            call["s3_prefix"]
            == final_records[0][1]["output_prefix"]
            == "validation/new"
        )
    else:
        assert call["artifact_storage"][1] == "new/evidence"
        assert (
            final_records[0][1]["artifact_sources"][0]["resolved_prefix"]
            == "new/evidence"
        )


@pytest.mark.parametrize("override", ["output", "source", "none"])
def test_transport_loss_cannot_adopt_old_healthy_agent_with_new_storage_scope(
    monkeypatch, tmp_path, override
):
    from typer import Exit
    from npa.clients.ssh import SSHError

    agent, function, defaults, _captured = _bootstrap_context(
        monkeypatch, tmp_path, "read-project"
    )
    events = _pending_bootstrap_convergence(monkeypatch, agent)
    agent._agent_record("deployment", "test-agent").update(
        setup_state="healthy", output_prefix=DEPLOYMENT[1]
    )

    def transport_loss(**kwargs):
        events.append(("bootstrap", kwargs))
        raise SSHError("synthetic transport interruption before storage staging")

    monkeypatch.setattr(agent, "_bootstrap_agent_stack", transport_loss)
    arguments = _storage_override_arguments(tmp_path, override, defaults)
    if override == "none":
        function(**arguments)
        assert events[-1][1]["setup_state"] == "healthy"
        return
    with pytest.raises(Exit):
        function(**arguments)
    assert [phase for phase, _value in events] == [
        "persist",
        "bootstrap",
        "reconcile",
        "persist",
    ]
    for phase, record in events:
        if phase == "persist":
            assert record["setup_state"] == "remote_bootstrap_pending"
            assert record["output_prefix"] == DEPLOYMENT[1]
            assert record["artifact_sources"][0]["resolved_prefix"] == READ[1]
