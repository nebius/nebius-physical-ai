from __future__ import annotations

from npa.cli.agent import rendered_agent_ui_html

import base64
import json
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer import Exit
from typer.testing import CliRunner

from npa.cli.agent import (
    AGENT_MEDIA_PREVIEW_CONTRACT,
    AGENT_RERUN_NO_BUNDLE_SPLASH_CONTRACT,
    AGENT_UI_VERSION,
    _normalize_llm_models,
    app,
    build_agent_urls,
)

runner = CliRunner()


def test_artifact_only_live_probe_is_read_only_and_state_stable() -> None:
    from npa.cli.agent import _artifact_only_http_probe

    digest = "a" * 64
    payloads = {
        "/api/health": {"ok": True, "state_sha256": digest},
        "/api/session": {"chat_history": []},
        "/api/artifacts/runs?prefix=&limit=100": {"runs": [{"run_id": "run-a"}]},
        "/api/tools": {"tool_refs": ["dataset"]},
        "/api/workflows/sim2real/status": {"latest_submit": {}},
        "/api/infra/k8s": {"ok": True},
    }

    class Response:
        def __init__(self, payload: dict) -> None:
            self.payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self.payload

    class GetOnlyClient:
        paths: list[str] = []

        def get(self, path: str) -> Response:
            self.paths.append(path)
            return Response(payloads[path])

    client = GetOnlyClient()
    result = _artifact_only_http_probe(client)  # type: ignore[arg-type]
    assert result["state_sha256"] == digest
    assert result["run_count"] == 1
    assert client.paths[0] == "/api/health"
    assert client.paths[-1] == "/api/health"


def test_fresh_deploy_refuses_nonempty_remote_state_before_apply(
    monkeypatch, tmp_path
) -> None:
    from npa.cli.agent import DeploymentIdentityError, _apply_agent_terraform

    applied = False
    monkeypatch.setattr(
        "npa.cli.agent.provisioner.prepare_working_dir", lambda *_args, **_kwargs: tmp_path
    )
    monkeypatch.setattr("npa.cli.agent.provisioner.init", lambda **_kwargs: None)
    monkeypatch.setattr(
        "npa.cli.agent.provisioner.state_list",
        lambda _tf_dir: ["nebius_compute_v1_instance.workbench"],
    )

    def apply(**_kwargs):
        nonlocal applied
        applied = True
        return {}

    monkeypatch.setattr("npa.cli.agent.provisioner.apply", apply)
    with pytest.raises(DeploymentIdentityError, match="no matching immutable agent record"):
        _apply_agent_terraform(
            project="project-a",
            name="agent-a",
            merged_vars={},
            env_region="us-central1",
            require_empty_state=True,
        )
    assert applied is False


def _mock_fresh_deploy_until_terraform(monkeypatch, tmp_path) -> tuple[dict, list[str]]:
    """Arrange a record-less deploy that reaches authoritative remote-state inspection."""
    deployment = {
        "deployment_id": "npa-agent-owner",
        "deployment_name": "agent-a",
        "project_alias": "project-a",
        "runtime_namespace": "project-a/agent-a",
        "repository": "org/repo",
        "branch": "codex/owner",
        "commit": "a" * 40,
        "source_tree": "b" * 40,
        "short_commit": "a" * 12,
        "workspace_label": "Workspace",
        "bootstrap_timestamp": "2026-08-10T00:00:00Z",
    }
    creds = {
        "service_account_id": "sa-agent",
        "nebius_api_key": "ak-agent",
        "nebius_secret_key": "sk-agent",
        "s3_bucket": "state-bucket",
        "s3_endpoint": "https://storage.example.invalid",
    }
    mutations: list[str] = []
    monkeypatch.setattr(
        "npa.cli.agent.build_deployment_manifest", lambda **_kwargs: deployment
    )
    monkeypatch.setattr("npa.cli.agent._agent_record", lambda *_args: {})
    monkeypatch.setattr(
        "npa.cli.agent._agent_terraform_state_exists", lambda *_args: False
    )
    monkeypatch.setattr(
        "npa.cli.agent.resolve_environment",
        lambda *_args, **kwargs: SimpleNamespace(
            project_id=kwargs.get("project_id"),
            tenant_id=kwargs.get("tenant_id"),
            region=kwargs.get("region"),
        ),
    )
    monkeypatch.setattr(
        "npa.cli.agent._resolve_deploy_llm_credentials", lambda: ("tf-key", "model")
    )
    monkeypatch.setattr("npa.cli.agent._agent_hard_prereq_results", lambda _path: [])
    monkeypatch.setattr(
        "npa.cli.agent._agent_token_factory_result",
        lambda _key: SimpleNamespace(status="PASS"),
    )
    monkeypatch.setattr(
        "npa.clients.nebius.bootstrap_agent_environment", lambda *_args, **_kwargs: creds
    )
    monkeypatch.setattr(
        "npa.cli.agent._resolve_deploy_storage_credentials", lambda **_kwargs: creds
    )
    monkeypatch.setattr("npa.clients.nebius.get_iam_token", lambda: "iam-token")
    monkeypatch.setattr(
        "npa.cli.agent._ensure_terraform_state_bucket", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        "npa.cli.agent._persist_agent_project_config", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        "npa.cli.agent._store_agent_record",
        lambda *_args, **_kwargs: mutations.append("record"),
    )
    monkeypatch.setattr(
        "npa.cli.agent._destroy_agent_terraform",
        lambda *_args, **_kwargs: mutations.append("destroy"),
    )
    monkeypatch.setattr(
        "npa.cli.agent.provisioner.prepare_working_dir",
        lambda *_args, **_kwargs: tmp_path,
    )
    monkeypatch.setattr("npa.cli.agent.provisioner.init", lambda **_kwargs: None)
    return deployment, mutations


def _call_fresh_deploy() -> None:
    from npa.cli.agent import deploy_cmd

    deploy_cmd(
        project="project-a",
        name="agent-a",
        project_id="project-id",
        tenant_id="tenant-id",
        region="us-central1",
        ssh_user="ubuntu",
        ssh_public_key_path="/unused/key.pub",
        tf_var=[],
        agent_port=8088,
        backend_port=8787,
        rerun_port=9090,
        llm_model="model",
        llm_models=[],
        foxglove_embed_src="",
        foxglove_org_slug="",
        foxglove_live_url="",
        no_public_https=False,
        workspace_label="Workspace",
        stock_demo=False,
    )


def test_fresh_deploy_state_inspection_failure_is_nondestructive(
    monkeypatch, tmp_path
) -> None:
    from npa.deploy.provisioner import ProvisionerError

    _deployment, mutations = _mock_fresh_deploy_until_terraform(monkeypatch, tmp_path)

    def fail_state_list(_tf_dir):
        raise ProvisionerError("backend unavailable")

    monkeypatch.setattr("npa.cli.agent.provisioner.state_list", fail_state_list)
    monkeypatch.setattr(
        "npa.cli.agent.provisioner.apply", lambda **_kwargs: mutations.append("apply")
    )

    with pytest.raises(Exit):
        _call_fresh_deploy()
    assert mutations == []


def test_fresh_deploy_backend_init_failure_is_nondestructive(
    monkeypatch, tmp_path
) -> None:
    from npa.deploy.provisioner import ProvisionerError

    _deployment, mutations = _mock_fresh_deploy_until_terraform(monkeypatch, tmp_path)

    def fail_init(**_kwargs):
        raise ProvisionerError("backend authentication unavailable")

    monkeypatch.setattr("npa.cli.agent.provisioner.init", fail_init)
    monkeypatch.setattr(
        "npa.cli.agent.provisioner.state_list",
        lambda _tf_dir: mutations.append("state-list"),
    )
    monkeypatch.setattr(
        "npa.cli.agent.provisioner.apply", lambda **_kwargs: mutations.append("apply")
    )

    with pytest.raises(Exit):
        _call_fresh_deploy()
    assert mutations == []


def test_fresh_deploy_remote_state_refusal_leaves_no_record(
    monkeypatch, tmp_path
) -> None:
    _deployment, mutations = _mock_fresh_deploy_until_terraform(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "npa.cli.agent.provisioner.state_list",
        lambda _tf_dir: ["nebius_compute_v1_instance.workbench"],
    )
    monkeypatch.setattr(
        "npa.cli.agent.provisioner.apply", lambda **_kwargs: mutations.append("apply")
    )

    with pytest.raises(Exit):
        _call_fresh_deploy()
    assert mutations == []


def test_destroy_refuses_terraform_state_without_agent_record(monkeypatch) -> None:
    destroyed = False
    monkeypatch.setattr("npa.cli.agent._agent_record", lambda *_args: {})
    monkeypatch.setattr("npa.cli.agent._agent_terraform_state_exists", lambda *_args: True)

    def destroy(*_args, **_kwargs):
        nonlocal destroyed
        destroyed = True

    monkeypatch.setattr("npa.cli.agent._destroy_agent_terraform", destroy)
    result = runner.invoke(
        app, ["destroy", "--project", "project-a", "--name", "agent-a"]
    )
    assert result.exit_code == 1
    assert "unknown ownership" in result.output
    assert destroyed is False


def test_deploy_refuses_local_state_without_agent_record(monkeypatch) -> None:
    from npa.cli.agent import deploy_cmd

    deployment = {
        "deployment_id": "npa-agent-owner",
        "deployment_name": "agent-a",
        "project_alias": "project-a",
        "runtime_namespace": "project-a/agent-a",
        "repository": "org/repo",
        "branch": "codex/owner",
        "commit": "a" * 40,
        "source_tree": "b" * 40,
        "short_commit": "a" * 12,
        "workspace_label": "Workspace",
        "bootstrap_timestamp": "2026-08-10T00:00:00Z",
    }
    mutated = False
    monkeypatch.setattr(
        "npa.cli.agent.build_deployment_manifest", lambda **_kwargs: deployment
    )
    monkeypatch.setattr("npa.cli.agent._agent_record", lambda *_args: {})
    monkeypatch.setattr("npa.cli.agent._agent_terraform_state_exists", lambda *_args: True)

    def store(*_args, **_kwargs):
        nonlocal mutated
        mutated = True

    monkeypatch.setattr("npa.cli.agent._store_agent_record", store)
    with pytest.raises(Exit):
        deploy_cmd(
            project="project-a",
            name="agent-a",
            project_id="project-id",
            tenant_id="tenant-id",
            region="us-central1",
            ssh_user="ubuntu",
            ssh_public_key_path="/unused/key.pub",
            tf_var=[],
            agent_port=8088,
            backend_port=8787,
            rerun_port=9090,
            llm_model="model",
            llm_models=[],
            foxglove_embed_src="",
            foxglove_org_slug="",
            foxglove_live_url="",
            no_public_https=False,
            workspace_label="Workspace",
            stock_demo=False,
        )
    assert mutated is False


def test_status_is_unhealthy_on_live_deployment_mismatch(monkeypatch) -> None:
    deployment = {
        "deployment_id": "npa-agent-owner",
        "deployment_name": "agent",
        "project_alias": "project-a",
        "runtime_namespace": "project-a/agent",
        "repository": "org/repo",
        "branch": "codex/owner",
        "commit": "a" * 40,
        "source_tree": "b" * 40,
        "short_commit": "a" * 12,
        "workspace_label": "Workspace",
        "bootstrap_timestamp": "2026-08-10T00:00:00Z",
    }
    record = {
        "agent_url": "https://203.0.113.50/",
        "rerun_url": "https://203.0.113.50/rerun/",
        "auth_secret_path": "/private/auth.env",
        "deployment": deployment,
    }
    live = dict(deployment)
    live["commit"] = "c" * 40
    monkeypatch.setattr("npa.cli.agent._agent_record", lambda *_args: record)
    monkeypatch.setattr("npa.cli.agent._load_auth_secret", lambda _path: ("npa", "secret"))
    monkeypatch.setattr("npa.cli.agent._health", lambda *_args, **_kwargs: (True, 200))
    monkeypatch.setattr("npa.cli.agent.fetch_live_deployment", lambda *_args, **_kwargs: live)
    result = runner.invoke(
        app, ["status", "--project", "project-a", "--name", "agent", "--json"]
    )
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["health"] is False
    assert payload["deployment_matches_record"] is False
    assert "commit" in payload["deployment_error"]


def test_staged_agent_source_is_readable_by_unprivileged_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa.cli import agent as agent_module

    archive = tmp_path / "source.tar.gz"
    archive.write_bytes(b"archive")
    monkeypatch.setattr(
        agent_module, "_create_agent_source_archive", lambda _commit: str(archive)
    )

    class FakeSSH:
        command = ""

        def upload_file(self, local: str, remote: str) -> None:
            assert local == str(archive)
            assert remote.startswith("/tmp/npa-agent-source-")

        def run_or_raise(self, command: str) -> None:
            self.command = command

        def run(self, command: str) -> None:
            assert command.startswith("rm -f /tmp/npa-agent-source-")

    ssh = FakeSSH()
    agent_module._stage_agent_npa_source(ssh, commit="a" * 40)  # type: ignore[arg-type]

    assert "sudo chown -R root:root /opt/npa-agent/npa-src" in ssh.command
    assert "sudo chmod -R a+rX /opt/npa-agent/npa-src" in ssh.command


def _agent_source() -> str:
    """agent.py plus source modules embedded or split out of it.

    Source-scanning assertions include login, nginx, and effective-access policy
    modules that the bootstrap embeds into the generated backend.
    """
    from npa.cli import agent as agent_module
    from npa.cli import agent_access_runtime as agent_access_runtime_module
    from npa.cli import agent_login as agent_login_module
    from npa.cli import agent_site as agent_site_module
    from npa.cli import agent_viewer_runtime as agent_viewer_runtime_module

    return "\n".join(
        Path(module.__file__).read_text(encoding="utf-8")
        for module in (
            agent_module,
            agent_access_runtime_module,
            agent_login_module,
            agent_site_module,
            agent_viewer_runtime_module,
        )
    )


def _agent_ui_bundle() -> str:
    """agent.py + nginx site policy + rendered UI HTML (UI lives in agent_ui.html)."""
    return _agent_source() + "\n" + rendered_agent_ui_html()


def test_build_agent_urls_https_default() -> None:
    urls = build_agent_urls("203.0.113.50")
    assert urls["public_url"] == "https://203.0.113.50/"
    assert urls["agent_url"] == urls["public_url"]
    assert urls["rerun_url"] == "https://203.0.113.50/rerun/"
    assert urls["sim_assets_url"] == "https://203.0.113.50/assets/"
    assert (
        urls["cameras_api_url"] == "https://203.0.113.50/assets/api/sim-assets/cameras"
    )
    assert urls["direct_url"] == "http://203.0.113.50:8088/"


def test_build_agent_urls_http_legacy() -> None:
    urls = build_agent_urls("203.0.113.50", public_https=False)
    assert urls["public_url"] == "http://203.0.113.50:8088/"
    assert urls["agent_url"] == urls["public_url"]
    assert urls["sim_assets_url"] == "http://203.0.113.50:8088/assets/"
    assert (
        urls["cameras_api_url"]
        == "http://203.0.113.50:8088/assets/api/sim-assets/cameras"
    )


def test_ensure_terraform_state_bucket_creates_missing_bucket(monkeypatch) -> None:
    from npa.cli.agent import _ensure_terraform_state_bucket

    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        "npa.clients.nebius.bucket_exists", lambda _project, _bucket: False
    )
    monkeypatch.setattr(
        "npa.clients.nebius.ensure_bucket",
        lambda project, bucket: calls.append((project, bucket)),
    )

    _ensure_terraform_state_bucket(project_id="project-1", bucket_name="bucket-1")

    assert calls == [("project-1", "bucket-1")]


def test_ensure_terraform_state_bucket_skips_existing_bucket(monkeypatch) -> None:
    from npa.cli.agent import _ensure_terraform_state_bucket

    called = False

    monkeypatch.setattr(
        "npa.clients.nebius.bucket_exists", lambda _project, _bucket: True
    )

    def _ensure(project: str, bucket: str) -> None:
        nonlocal called
        _ = (project, bucket)
        called = True

    monkeypatch.setattr("npa.clients.nebius.ensure_bucket", _ensure)

    _ensure_terraform_state_bucket(project_id="project-1", bucket_name="bucket-1")

    assert called is False


def test_apply_agent_terraform_filters_runtime_only_s3_prefix(
    monkeypatch, tmp_path
) -> None:
    from npa.cli.agent import _apply_agent_terraform

    captured: dict[str, str] = {}

    monkeypatch.setattr(
        "npa.cli.agent.provisioner.prepare_working_dir",
        lambda *_args, **_kwargs: tmp_path,
    )
    monkeypatch.setattr("npa.cli.agent.provisioner.init", lambda **_kwargs: None)

    def _apply(*, tf_dir, tf_vars):
        assert tf_dir == tmp_path
        captured.update(tf_vars)
        return {"vm_ip": "203.0.113.50"}

    monkeypatch.setattr("npa.cli.agent.provisioner.apply", _apply)

    _apply_agent_terraform(
        project="fresh",
        name="agent",
        env_region="us-central1",
        merged_vars={
            "s3_bucket": "agent-state",
            "s3_prefix": "runtime/artifacts",
            "s3_endpoint": "https://storage.us-central1.nebius.cloud",
            "nebius_api_key": "ak",
            "nebius_secret_key": "sk",
            "service_account_id": "sa",
        },
    )

    assert captured["s3_bucket"] == "agent-state"
    assert "s3_prefix" not in captured


def test_resolve_deploy_storage_credentials_prefers_bootstrap_when_writable(
    monkeypatch,
) -> None:
    from npa.cli.agent import _resolve_deploy_storage_credentials

    monkeypatch.setattr(
        "npa.cli.agent_artifact_storage._storage_credentials_allow_writes",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        "npa.cli.agent_artifact_storage.load_credentials",
        lambda **_kwargs: SimpleNamespace(
            s3_bucket="",
            s3_endpoint="",
            s3_access_key_id="",
            s3_secret_access_key="",
        ),
    )
    bootstrap = {
        "s3_bucket": "bucket-boot",
        "s3_endpoint": "https://storage.us-central1.nebius.cloud",
        "nebius_api_key": "ak-boot",
        "nebius_secret_key": "sk-boot",
    }

    resolved = _resolve_deploy_storage_credentials(
        region="us-central1", bootstrap_creds=bootstrap
    )

    assert resolved["s3_bucket"] == "bucket-boot"
    assert resolved["nebius_api_key"] == "ak-boot"


def test_resolve_deploy_storage_credentials_prefers_shared_artifact_bucket(
    monkeypatch,
) -> None:
    from npa.cli.agent import _resolve_deploy_storage_credentials

    monkeypatch.setattr(
        "npa.cli.agent_artifact_storage._storage_credentials_allow_writes",
        lambda **kwargs: kwargs["bucket"] == "shared-bucket",
    )
    monkeypatch.setattr(
        "npa.cli.agent_artifact_storage.load_credentials",
        lambda **_kwargs: SimpleNamespace(
            s3_bucket="s3://shared-bucket/checkpoints/",
            s3_endpoint="https://storage.us-central1.nebius.cloud",
            s3_access_key_id="ak-shared",
            s3_secret_access_key="sk-shared",
        ),
    )
    bootstrap = {
        "s3_bucket": "npa-bucket-terraform",
        "s3_endpoint": "https://storage.us-central1.nebius.cloud",
        "nebius_api_key": "ak-boot",
        "nebius_secret_key": "sk-boot",
    }

    resolved = _resolve_deploy_storage_credentials(
        region="us-central1", bootstrap_creds=bootstrap
    )

    assert resolved["s3_bucket"] == "shared-bucket"
    assert resolved["s3_prefix"] == "checkpoints"
    assert resolved["nebius_api_key"] == "ak-shared"


def test_resolve_deploy_storage_credentials_prefers_selected_project_storage(
    monkeypatch,
) -> None:
    from npa.cli.agent import _resolve_deploy_storage_credentials

    monkeypatch.setattr(
        "npa.cli.agent_artifact_storage.resolve_project_storage",
        lambda *_args, **_kwargs: SimpleNamespace(
            checkpoint_bucket="s3://project-bucket/isaac-runs/",
            endpoint_url="https://storage.us-central1.nebius.cloud",
            aws_access_key_id="ak-project",
            aws_secret_access_key="sk-project",
        ),
    )
    monkeypatch.setattr(
        "npa.cli.agent_artifact_storage.load_credentials",
        lambda **_kwargs: SimpleNamespace(
            s3_bucket="s3://shared-bucket/",
            s3_endpoint="https://storage.eu-north1.nebius.cloud",
            s3_access_key_id="ak-shared",
            s3_secret_access_key="sk-shared",
        ),
    )
    monkeypatch.setattr(
        "npa.cli.agent_artifact_storage._storage_credentials_allow_writes",
        lambda **_kwargs: True,
    )
    bootstrap = {
        "service_account_id": "sa-agent",
        "s3_bucket": "bootstrap-bucket",
        "s3_endpoint": "https://storage.us-central1.nebius.cloud",
        "nebius_api_key": "ak-bootstrap",
        "nebius_secret_key": "sk-bootstrap",
    }

    resolved = _resolve_deploy_storage_credentials(
        region="us-central1",
        bootstrap_creds=bootstrap,
        project_alias="target-project",
    )

    assert resolved["service_account_id"] == "sa-agent"
    assert resolved["s3_bucket"] == "project-bucket"
    assert resolved["s3_prefix"] == "isaac-runs"
    assert resolved["nebius_api_key"] == "ak-project"


def test_resolve_explicit_artifact_project_storage_preserves_nested_prefix(
    monkeypatch,
) -> None:
    from npa.cli.agent import _resolve_artifact_project_storage

    monkeypatch.setattr(
        "npa.cli.agent_artifact_storage.resolve_project_storage",
        lambda *_args, **_kwargs: SimpleNamespace(
            checkpoint_bucket="s3://artifact-bucket/nested/shared-root/",
            endpoint_url="https://storage.us-central1.nebius.cloud",
            aws_access_key_id="artifact-ak",
            aws_secret_access_key="artifact-sk",
        ),
    )
    monkeypatch.setattr(
        "npa.cli.agent_artifact_storage._storage_credentials_allow_writes",
        lambda **_kwargs: True,
    )

    resolved = _resolve_artifact_project_storage(
        "artifact-source", region="us-central1"
    )

    assert resolved == (
        "artifact-bucket",
        "nested/shared-root",
        "https://storage.us-central1.nebius.cloud",
        "artifact-ak",
        "artifact-sk",
    )


def test_artifact_prefix_validation_rejects_traversal() -> None:
    from npa.cli.agent import _validate_artifact_prefix

    assert _validate_artifact_prefix("/nested/shared-root/") == "nested/shared-root"
    for value in ("../escape", "nested/../escape", "nested\\escape"):
        with pytest.raises(ValueError):
            _validate_artifact_prefix(value)


def test_resolve_deploy_storage_credentials_falls_back_to_shared(monkeypatch) -> None:
    from npa.cli.agent import _resolve_deploy_storage_credentials

    def _probe(**kwargs):
        return kwargs["bucket"] == "shared-bucket"

    monkeypatch.setattr(
        "npa.cli.agent_artifact_storage._storage_credentials_allow_writes", _probe
    )
    monkeypatch.setattr(
        "npa.cli.agent_artifact_storage.load_credentials",
        lambda **_kwargs: SimpleNamespace(
            s3_bucket="s3://shared-bucket/",
            s3_endpoint="https://storage.us-central1.nebius.cloud",
            s3_access_key_id="ak-shared",
            s3_secret_access_key="sk-shared",
        ),
    )
    bootstrap = {
        "s3_bucket": "bucket-boot",
        "s3_endpoint": "https://storage.us-central1.nebius.cloud",
        "nebius_api_key": "ak-boot",
        "nebius_secret_key": "sk-boot",
    }

    resolved = _resolve_deploy_storage_credentials(
        region="us-central1", bootstrap_creds=bootstrap
    )

    assert resolved["s3_bucket"] == "shared-bucket"
    assert resolved["nebius_api_key"] == "ak-shared"


def test_resolve_deploy_storage_credentials_prefers_saved_project_state(
    monkeypatch,
) -> None:
    from npa.cli.agent import _resolve_deploy_storage_credentials

    class _TfState:
        bucket = "state-bucket"
        endpoint = "https://storage.us-central1.nebius.cloud"
        access_key = "ak-state"
        secret_key = "sk-state"

    def _probe(**kwargs):
        return kwargs["bucket"] == "state-bucket"

    monkeypatch.setattr(
        "npa.cli.agent_artifact_storage._storage_credentials_allow_writes", _probe
    )
    monkeypatch.setattr(
        "npa.cli.agent_artifact_storage.resolve_terraform_state",
        lambda _project: _TfState(),
    )
    bootstrap = {
        "service_account_id": "sa-agent",
        "s3_bucket": "bucket-boot",
        "s3_endpoint": "https://storage.us-central1.nebius.cloud",
        "nebius_api_key": "ak-boot",
        "nebius_secret_key": "sk-boot",
    }

    resolved = _resolve_deploy_storage_credentials(
        region="us-central1",
        bootstrap_creds=bootstrap,
        project_alias="fresh",
    )

    assert resolved["service_account_id"] == "sa-agent"
    assert resolved["s3_bucket"] == "state-bucket"
    assert resolved["nebius_api_key"] == "ak-state"


def test_resolve_deploy_storage_credentials_fails_without_writable_storage(
    monkeypatch,
) -> None:
    from npa.cli.agent import _resolve_deploy_storage_credentials

    monkeypatch.setattr(
        "npa.cli.agent_artifact_storage._storage_credentials_allow_writes",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(
        "npa.cli.agent_artifact_storage.load_credentials",
        lambda **_kwargs: SimpleNamespace(
            s3_bucket="s3://shared-bucket/",
            s3_endpoint="https://storage.us-central1.nebius.cloud",
            s3_access_key_id="ak-shared",
            s3_secret_access_key="sk-shared",
        ),
    )
    bootstrap = {
        "s3_bucket": "bucket-boot",
        "s3_endpoint": "https://storage.us-central1.nebius.cloud",
        "nebius_api_key": "ak-boot",
        "nebius_secret_key": "sk-boot",
    }

    with pytest.raises(Exit):
        _resolve_deploy_storage_credentials(
            region="us-central1", bootstrap_creds=bootstrap
        )


def test_deploy_persists_terraform_state_before_apply(monkeypatch, tmp_path) -> None:
    from npa.cli.agent import deploy_cmd

    events: list[tuple[str, dict]] = []
    creds = {
        "service_account_id": "sa-agent",
        "nebius_api_key": "ak-agent",
        "nebius_secret_key": "sk-agent",
        "s3_bucket": "npa-agent-state",
        "s3_endpoint": "https://storage.us-central1.nebius.cloud",
    }

    def _write_config(payload: dict) -> None:
        events.append(("write_config", payload))

    def _apply_agent_terraform(**kwargs):
        assert any(
            event == "write_config"
            and payload.get("projects", {})
            .get("fresh", {})
            .get("terraform_state", {})
            .get("bucket")
            == "npa-agent-state"
            for event, payload in events
        )
        events.append(("apply", kwargs))
        return {
            "vm_ip": "203.0.113.50",
            "instance_id": "instance-agent",
            "ssh_key_path": str(tmp_path / "id_ed25519"),
        }

    monkeypatch.setattr(
        "npa.cli.agent.resolve_environment",
        lambda *_args, **kwargs: SimpleNamespace(
            project_id=kwargs.get("project_id"),
            tenant_id=kwargs.get("tenant_id"),
            region=kwargs.get("region"),
        ),
    )
    monkeypatch.setattr(
        "npa.clients.nebius.bootstrap_agent_environment",
        lambda *_args, **_kwargs: creds,
    )
    monkeypatch.setattr(
        "npa.cli.agent._resolve_deploy_storage_credentials", lambda **_kwargs: creds
    )
    monkeypatch.setattr("npa.clients.nebius.get_iam_token", lambda: "iam-token")
    monkeypatch.setattr(
        "npa.cli.agent._ensure_terraform_state_bucket", lambda **_kwargs: None
    )
    monkeypatch.setattr("npa.cli.agent._apply_agent_terraform", _apply_agent_terraform)
    monkeypatch.setattr("npa.cli.agent._is_routable_public_ip", lambda _ip: True)
    monkeypatch.setattr(
        "npa.cli.agent._write_auth_secret", lambda **_kwargs: tmp_path / "auth.env"
    )
    monkeypatch.setattr(
        "npa.cli.agent._resolve_deploy_llm_credentials", lambda: ("tf-key", "model-a")
    )
    monkeypatch.setattr("npa.cli.agent._resolve_operator_credentials", lambda: ("", ""))
    monkeypatch.setattr("npa.cli.agent._bootstrap_agent_stack", lambda **_kwargs: None)
    monkeypatch.setattr("npa.cli.agent.ensure_ingress", lambda **_kwargs: None)
    monkeypatch.setattr("npa.cli.agent.write_config", _write_config)

    # Satisfy the fail-fast deploy prerequisites (terraform + SSH key pair) that
    # now run before any cloud side effects.
    (tmp_path / "id_ed25519.pub").write_text("ssh-ed25519 AAAA test\n")
    (tmp_path / "id_ed25519").write_text("-----BEGIN OPENSSH PRIVATE KEY-----\n")
    monkeypatch.setenv("NPA_TERRAFORM_BIN", "/usr/bin/terraform")

    deploy_cmd(
        project="fresh",
        name="agent",
        project_id="project-1",
        tenant_id="tenant-1",
        region="us-central1",
        ssh_user="ubuntu",
        ssh_public_key_path=str(tmp_path / "id_ed25519.pub"),
        tf_var=[],
        agent_port=8088,
        backend_port=8787,
        rerun_port=9090,
        llm_model="model-a",
        llm_models=[],
        no_public_https=False,
    )

    assert [event for event, _payload in events].count("write_config") >= 2
    assert any(event == "apply" for event, _payload in events)


def test_bootstrap_enables_public_https_nginx() -> None:
    source = _agent_source()
    assert "ssl_certificate /etc/nginx/ssl/npa-agent.crt" in source
    assert "DEFAULT_HTTPS_PORT" in source
    assert "Customer URL: use" in source
    assert "--no-public-https" in source


def test_bootstrap_nginx_serves_rerun_recording_to_same_origin_wasm() -> None:
    source = _agent_source()
    assert r"cap-[A-Za-z0-9_-]{{43}}\\.rrd" in source
    assert "location /rerun/recordings/" in source
    assert "alias /opt/npa-agent/recordings/$1;" in source
    recordings_location = source.split('location ~ "^/rerun/recordings/(cap-', 1)[
        1
    ].split("location /rerun/recordings/", 1)[0]
    directives = [
        line.strip()
        for line in recordings_location.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert "auth_basic off;" in directives
    assert not [line for line in directives if "Access-Control" in line]
    assert 'add_header Cross-Origin-Resource-Policy "same-origin" always;' in directives
    denied_location = source.split("location /rerun/recordings/ {{", 1)[1].split(
        "location ~* ^/rerun/", 1
    )[0]
    assert "return 404;" in denied_location
    rerun_viewer_location = source.split("location /rerun/ {{", 1)[1].split(
        "location / {{", 1
    )[0]
    assert "auth_basic off;" in rerun_viewer_location
    rerun_asset_location = source.split("location ~* ^/rerun/", 1)[1].split(
        "location /rerun/ {{", 1
    )[0]
    assert "auth_basic off;" in rerun_asset_location


def test_bootstrap_embeds_lichtblick_viewer() -> None:
    source = _agent_source()
    # nginx: co-serve the MCAP same-origin and proxy the viewer sidecar.
    assert "location /lichtblick/recordings/" in source
    assert "location /lichtblick/ {{" in source
    assert "proxy_pass http://127.0.0.1:{lichtblick_port}/;" in source
    # backend: sim-viz status carries the Lichtblick embed fields.
    assert (
        'LICHTBLICK_RECORDING_HTTP_PATH = "/lichtblick/recordings/sim2real.mcap"'
        in source
    )
    assert "def _lichtblick_iframe_url" in source
    assert '"lichtblick_ready": False,' in source
    assert '"lichtblick_iframe_url": "/lichtblick/",' in source
    assert "def _publish_mcap_recording" in source
    assert 'elif render == "mcap":' in source
    # best-effort viewer sidecar unit.
    assert "npa-lichtblick.service" in source
    # verify() probes the embed plumbing.
    assert "lichtblick embed probe" in source
    # Region-agnostic image acquisition: the sidecar pulls from whichever mirror
    # registry (eu-north1 or us-central1) is reachable, not a locally-built image.
    assert "lichtblick_pull_candidates" in source
    assert "for lb_cand in {lichtblick_pull_candidates}" in source
    assert "npa-lichtblick image acquired from" in source


def test_lichtblick_recordings_grant_no_cross_origin_read() -> None:
    """The MCAP alias is unauthenticated, so it must not be CORS-readable.

    A run's MCAP carries camera frames, VLM critiques and reward signals, and the
    location runs with ``auth_basic off`` (wasm/worker fetches cannot carry basic
    auth). A wildcard ``Access-Control-Allow-Origin`` would let any page a viewer
    visits read those recordings off this host; the embed is same-origin and needs
    no CORS grant at all.
    """

    source = _agent_source()
    recordings_location = source.split("location /lichtblick/recordings/ {{", 1)[
        1
    ].split("location = /lichtblick/ {{", 1)[0]
    # Compare directives only: the block's comment names these headers to explain
    # why they are absent, so a bare substring check would match the prose.
    directives = [
        line.strip()
        for line in recordings_location.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert "auth_basic off;" in directives
    granted = [line for line in directives if "Access-Control" in line]
    assert not granted, f"recordings must grant no CORS access, got {granted}"
    assert 'add_header Cross-Origin-Resource-Policy "same-origin" always;' in directives


def test_ui_pins_lichtblick_recording_fetch_to_the_page_origin() -> None:
    """Because the recordings alias grants no CORS, the viewer's fetch must be
    same-origin even when the backend built ds.url from a configured public
    origin that differs from the origin the page was loaded from."""

    source = _agent_ui_bundle()
    assert "function pinLichtblickDsToSameOrigin" in source
    assert "window.location.origin" in source
    # The iframe URL always flows through the rewrite.
    assert 'return pinLichtblickDsToSameOrigin(url) || "/lichtblick/";' in source


def test_ui_seeds_the_lichtblick_layout_once_rather_than_wiping_every_mount() -> None:
    """The layout wipe evicts a pre-injection layout; it must not run every mount.

    Wiping on each (re)mount also discards a layout the user arranged inside the
    embed, so the wipe is gated on a per-UI-version seed marker.
    """

    source = _agent_ui_bundle()
    assert "function lichtblickNeedsLayoutSeed" in source
    assert "function markLichtblickLayoutSeeded" in source
    # The wipe is reachable only behind the seed check.
    mount = source.split("function mountLichtblickIframe", 1)[1].split(
        "async function ensureLichtblickForActiveRun", 1
    )[0]
    assert "if (lichtblickNeedsLayoutSeed()) {" in mount
    reset_calls = mount.count("resetLichtblickLayoutStorage()")
    assert reset_calls == 1, f"expected one guarded wipe, found {reset_calls}"


def test_bootstrap_injects_lichtblick_default_layout() -> None:
    source = _agent_source()
    # The viewer document is exact-matched so nginx can inject a default layout via
    # the upstream-provided placeholder, so the point cloud + camera show on load.
    assert "location = /lichtblick/ {{" in source
    assert (
        "sub_filter '{lichtblick_layout_placeholder}' '{lichtblick_default_layout}';"
        in source
    )
    assert "def _lichtblick_default_layout_json" in source

    from npa.cli import agent_site as agent_site_module

    layout = json.loads(agent_site_module._lichtblick_default_layout_json())
    panels = layout["configById"]
    three_d = next(v for k, v in panels.items() if k.startswith("3D!"))
    assert three_d["topics"]["/heldout/points"]["visible"] is True
    assert three_d["followTf"] == "sim2real"
    image = next(v for k, v in panels.items() if k.startswith("Image!"))
    assert image["imageMode"]["imageTopic"] == "/camera"


def test_bootstrap_ui_embeds_lichtblick_render_mode() -> None:
    source = _agent_ui_bundle()
    assert 'id="renderModeLichtblick"' in source
    assert 'data-render-mode="lichtblick"' in source
    assert 'id="lichtblickFrame"' in source
    assert 'id="viewerPaneLichtblick"' in source
    assert 'bindClick("openLichtblick"' in source
    assert 'bindClick("loadLichtblickViewer"' in source
    assert "function applyLichtblickSimViz" in source
    assert "function mountLichtblickIframe" in source
    assert "View in Lichtblick" in source


def test_bootstrap_ui_lichtblick_autoloads_run_mcap() -> None:
    # Clicking the Lichtblick tab / reload finds and loads the run's .mcap directly,
    # and the artifact type filter exposes an 'mcap' option so it is discoverable.
    source = _agent_ui_bundle()
    assert "function ensureLichtblickForActiveRun" in source
    assert '<option value="mcap">' in source
    assert "ensureLichtblickForActiveRun()" in source


def test_bootstrap_artifact_file_transcodes_ppm_to_png() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    # .ppm/.bmp/.tiff are transcoded to PNG on serve so the browser can render them.
    assert "needs_image_transcode(safe_name)" in source
    assert 'media_type="image/png"' in source


def test_franka_rerun_fallback_keeps_3d_outside_pinhole_projection() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert "_franka_demo_joint_angles" in source
    assert "frame_count = 90" in source
    assert "world/camera_frustums/{{name}}" in source
    assert 'f"{entity}/frustum"' not in source
    assert 'f"{entity}/origin"' not in source


def test_agent_artifact_discovery_requires_s3_components() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert "list_runs(" in source
    assert "list_artifacts(" in source
    assert "download_s3_uri(" in source
    assert "Use this S3-backed Sim2Real run" in source
    assert "No S3 artifacts found for that run" in source
    assert '"source": "s3"' in source
    assert "local_path.resolve() != target.resolve()" in source
    assert "run artifacts to S3" in source
    assert "def _local_run_summaries" not in source


def test_agent_help_smoke() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "deploy" in result.output
    assert "fresh-setup" in result.output
    assert "bootstrap" in result.output
    assert "verify-live" in result.output


def test_bootstrap_embeds_chat_endpoint() -> None:

    source = _agent_ui_bundle()
    assert '@app.post("/chat")' in source
    assert '@app.get("/session")' in source
    assert '@app.get("/models")' in source
    assert "Workbench Chat" in source
    assert "NEBIUS_TOKEN_FACTORY_KEY" in source
    assert "NPA_AGENT_LLM_MODELS" in source
    assert 'id="chatModel"' in source
    assert "llm.env" in source
    assert "renderInlineMarkdownLite" in source
    assert "showThinkingBubble" in source
    assert "thinking-ellipsis" in source
    assert 'aria-label="thinking">...</span>' in source
    assert "font-family: Inter, system-ui" in source
    assert "font-family: monospace" not in source
    assert "quick-pill" in source
    assert "--brand: #e5ff4f;" in source
    assert "--sidebar: #0d2a3d;" in source
    assert "--thinking-fg:" in source
    assert ".msg-row.user .bubble" in source
    assert "color: var(--brand-ink);" in source
    assert "markdownLiteHtml" in source
    assert "Secure basic-auth session" in source
    assert "enqueueChatJob" in source
    assert "npa workbench byof run" in source or "run_byof_repo.py" in source
    assert "For BYOF solution onboarding" in source
    assert "Always use real registry-qualified images" in source
    assert "`<your-registry-id>` placeholders" in source
    assert "sky gpus list" in source
    bootstrap_split = '        const lines = String(text || "").split(/\\r?\\n/);'
    assert "\r" not in bootstrap_split
    assert "\\r?\\n" in bootstrap_split
    assert "restoreSession" in source
    assert "bootPage()" in source
    assert "ensureFrankaRerunLoaded" in source
    assert "setTimeout(() =>" in source
    assert "startPeriodicRefresh" in source
    assert "fetchWithTimeout" in source
    assert "welcome.html" in source
    assert "login-help.html" in source
    assert "/welcome" in source
    assert "_agent_public_login_form_html" in source
    assert 'id="npa-sign-in"' in source
    assert "Sign in</button>" in source
    assert "encodeURIComponent(user)" in source
    assert 'normalizedPath === "/login-help.html"' in source
    assert 'normalizedPath === "/welcome"' in source
    assert "showRerunPlaceholder" in source
    assert "rerunIframeLoaded" in source
    assert "setChatModels" in source
    assert "selectedChatModel" in source
    assert "startApp()" in source
    assert "function bindClick(" in source
    assert "function wireUi()" in source
    assert "function showToast(" in source
    assert 'id="statusBar"' in source
    assert 'id="toastHost"' in source
    assert "DOMContentLoaded" in source
    assert "initNpaAgentUi" in source
    assert 'id="chatForm"' in source
    assert "mobile-agent" in source
    assert 'name="viewport" content="width=device-width' in source
    assert "mobileChatAuth" in source
    assert "npa_agent_basic_auth" in source
    assert "mobileAuthTokenCache" in source
    assert "verifyMobileChatAuth" in source
    assert 'credentials: useExplicitAuth ? "omit" : "include"' in source
    assert "activeChatSessionId" in source
    assert "/api/chat/sessions" in source
    assert "npa-agent/tenants/" in source
    assert "/deployments/{{deployment_id}}/chat-sessions" in source
    assert "Send failed." in source
    assert "queueChatText" in source
    assert "AGENT_UI_VERSION" in source or "npa-ui-version" in source
    assert 'add_header Cache-Control "no-store, no-cache, must-revalidate"' in source
    assert "@media (max-width: 900px)" in source
    assert "safe-area-inset-bottom" in source
    # Mobile tabs must show their own content: the viewer (layout-rerun) on the
    # Rerun tab and run selection (Stages) on the Main tab — not hidden behind a
    # global "Panels" toggle. Only the desktop YAML panel stays collapsed.
    assert "body.mobile-agent .layout-rerun { grid-template-columns: 1fr; }" in source
    assert "body.mobile-agent .workflow-panel { display: none; }" in source
    assert "body.mobile-agent #panelChat .chat-panel" in source
    assert "body.mobile-agent.mobile-show-panels .layout-rerun" not in source
    # Mobile: don't mount the unsupported Rerun wasm viewer — route to Voxel51.
    assert "function isMobileLayout()" in source
    assert "function showMobileRerunNotice(" in source
    assert "3D Rerun viewer needs a desktop browser" in source
    assert "#panelVoxel.is-inactive { display: none; }" in source
    # Voxel51 auto-loads the latest run so the tab always shows content.
    assert "/api/artifacts/runs?limit=1" in source
    # Multi-bucket discovery is deterministic: only configured buckets are scanned.
    assert "def _agent_s3_buckets(" in source
    assert "accessible_artifact_buckets(_agent_access_report())" in source
    assert "list_runs_cached_multi" in source
    assert "find_run_artifacts_across_buckets" in source
    assert "Never enumerate every credential-readable bucket" in source
    # Modern refresh + iOS/desktop friendliness (cascade override layer):
    assert "Modern refresh (2026)" in source
    assert "-webkit-font-smoothing: antialiased" in source
    assert "env(safe-area-inset-top)" in source  # iOS notch handling on the top bar
    assert "prefers-reduced-motion" in source
    assert "focus-visible" in source
    # iOS: >=16px inputs on small screens so focusing a field never triggers zoom.
    assert "font-size: 16px;" in source
    # Brand tokens are preserved by the refresh layer.
    assert "--brand: #e5ff4f;" in source
    assert "border-bottom: 4px solid var(--brand)" in source


def test_watch_intent_uses_live_sim_viz_status() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert 'elif intent in {"sim2real_status", "watch_sim"}:' in source
    assert "live_status = sim_viz_status()" in source
    assert 'state["sim_viz"] = dict(live_status)' in source


def test_bootstrap_public_login_form() -> None:
    from npa.cli import agent as agent_module

    html = agent_module._agent_public_login_form_html("npa")
    assert 'id="npa-sign-in"' in html
    assert (
        'id="npa-sign-in-btn">Sign in</button>' in html
        or 'type="submit">Sign in</button>' in html
    )
    assert 'value="npa"' in html
    assert "encodeURIComponent(user)" in html
    assert "encodeURIComponent(pass)" in html
    assert "history.replaceState" in html
    assert "persistBasicAuth" in html
    assert (
        'normalizedPath === "/login-help.html"' in html or '"/login-help.html"' in html
    )


def test_bootstrap_ui_button_wiring_patterns() -> None:

    source = _agent_ui_bundle()
    for control_id in (
        "chatActionS3",
        "chatActionCosmos",
        "chatActionWatch",
        "openRerun",
        "workflowStatus",
    ):
        assert f'bindClick("{control_id}"' in source
    # The Selection / Scene-mode section was removed (viewer just shows run
    # artifacts now); its controls must no longer be wired or present.
    for removed_id in ("loadFrankaRerun", "applySelection", "submitWorkflow"):
        assert f'bindClick("{removed_id}"' not in source
    assert 'id="chatForm"' in source
    assert 'chatForm.addEventListener("submit"' in source
    assert 'await apiJson("/api/chat"' in source
    assert 'await apiJson("/api/sim-viz/load-franka-demo"' in source
    assert 'await apiJson("/api/sim-assets/selection"' in source
    # Dead camera-preview UI helper removed (G6); endpoint may still exist server-side.
    assert "setChatBusy(false)" in source
    assert (
        "finally {"
        in source.split("async function processChatQueue")[1].split(
            "function enqueueChatJob"
        )[0]
    )
    assert "queueChatText" in source
    assert "processChatQueue" in source


def test_bootstrap_embeds_cameras_panel() -> None:

    source = _agent_ui_bundle()
    # Cameras panel removed from UI; APIs and stock camera metadata remain.
    assert "cameras-panel" not in source
    assert "cameraCards" not in source
    assert "Preview in Rerun" not in source
    assert '@app.get("/sim-assets/cameras")' in source
    assert '@app.post("/sim-viz/camera-preview")' in source
    assert "world/cameras/" in source
    assert "world/camera_frustums/" in source
    assert 'f"{{frustum_entity}}/frustum"' in source
    assert 'f"{{entity}}/frustum"' not in source
    assert "There is no separate Cameras panel in the UI" in source
    assert "stock_workspace" in source
    assert "stock_ee_mounted" in source
    assert "frustumSvg" in source
    assert 'id="tabMain"' in source
    assert 'id="tabRerun"' in source
    assert "layout-rerun" in source
    assert "activateMainTab" in source
    assert "tab-panel.is-inactive" in source
    assert (
        "defer the Rerun wasm viewer bundle" in source
        or "unload or defer the Rerun wasm" in source
    )
    import re

    iframe = re.search(r'<iframe id="rerunFrame"[^>]*>', source)
    assert iframe is not None
    assert "loading=" not in iframe.group(0)
    ui_html = rendered_agent_ui_html()
    for marker in AGENT_RERUN_NO_BUNDLE_SPLASH_CONTRACT:
        assert marker in ui_html, f"missing no-bundle-splash marker: {marker!r}"
    assert (
        'Mount the viewer immediately so "Loading application bundle" starts early'
        not in ui_html
    )
    assert (
        "rerunIframeLoaded = false"
        not in source.split("async function activateMainTab")[1].split(
            "async function"
        )[0]
    )


def test_ui_renders_tenant_resource_states_and_refresh_control() -> None:
    source = _agent_ui_bundle()
    for marker in (
        'id="tenantResourcesPanel"',
        '<h3>Tenant resources</h3>',
        'id="tenantResourcesRefresh"',
        'id="tenantResourceCategories"',
        "refreshTenantResources",
        'loadJson("/api/resources" + suffix)',
        "Accessible / discovered",
        "Configured references",
        "Discovery succeeded; no resources were returned",
        "resource-status-error",
        "request_error",
    ):
        assert marker in source
    assert 'bindClick("tenantResourcesRefresh"' in source


def test_bootstrap_stock_camera_defaults_match_scene_assets() -> None:
    from npa.cli import agent as agent_module
    from npa.genesis.scene_assets import (
        CAMERA_PLACEMENT_STOCK_EE_MOUNTED,
        CAMERA_PLACEMENT_STOCK_WORKSPACE,
        DEFAULT_CAMERA_NAMES,
    )

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    for name in DEFAULT_CAMERA_NAMES:
        assert f'"name": "{name}"' in source
    assert CAMERA_PLACEMENT_STOCK_WORKSPACE in source
    assert CAMERA_PLACEMENT_STOCK_EE_MOUNTED in source
    assert '"pos": [1.0, 0.0, 0.8]' in source
    assert '"pos": [0.4, 0.0, 0.4]' in source


def test_bootstrap_embeds_franka_rerun_ux() -> None:

    source = _agent_ui_bundle()
    assert "--sidebar: #0d2a3d" in source
    assert "--brand: #e5ff4f" in source
    assert "--surface-blue: #dceeff" in source
    assert "letter-spacing: 0.22em" in source
    assert "border-bottom: 4px solid var(--brand)" in source
    assert '@app.post("/sim-viz/load-franka-demo")' in source
    assert "NPA_AGENT_PRELOAD_STOCK_DEMO" in source
    assert "if not PRELOAD_STOCK_DEMO or not RRD_PATH.is_file()" in source
    assert "_wire_franka_demo" in source
    assert "_generate_franka_demo_rrd" in source
    assert "_log_franka_robot_geometry" in source
    assert "robot/franka/links" in source
    assert "Open in Rerun" in source
    # Selection / Scene-mode section removed from the viewer tab.
    assert 'id="sceneMode"' not in source
    assert 'id="robotPreset">' not in source
    assert "Apply stock selection" not in source
    assert "Load active Sim2Real in Rerun" not in source
    assert '<label class="pill"><input id="propCube"' not in source
    assert (
        'class="panel rerun-panel rerun-stage"' in source
        or 'class="panel rerun-panel rerun-stage"' in source
    )
    assert ".layout-rerun {{" in source or ".layout-rerun {" in source
    assert "cameras-panel" not in source
    assert "rerun-frame-shell" in source
    assert "robotPreset" in source
    assert "rerunPlaceholder" in source
    assert 'id="rerunFrame" title="rerun" src="about:blank"' in source
    assert "theme=dark" in source
    assert "allowfullscreen" in source
    assert "RERUN_RECORDING_PATH" in source
    assert "location.origin + RERUN_RECORDING_PATH" in source
    assert "rrdUrl = await resolveRerunRecordingUrl();" in source
    assert "rrdUrl.startsWith" in source
    assert "location.origin + rrdUrl" in source
    assert "_rerun_iframe_url" in source
    assert "NPA_AGENT_PUBLIC_URL" in source
    assert "cap-[A-Za-z0-9_-]{{43}}" in source
    assert (
        "Prefer the public recording copy; authenticated blob fetch remains the fallback"
        in source
    )
    assert "does not reliably consume parent-created blob URLs" in source
    # Path-only `/rerun/...` is parsed by Rerun as host `rerun` and must not be emitted.
    assert "url=/rerun/recordings/sim2real.rrd" not in source
    assert '"&renderer=webgl&hide_welcome_screen=1&camera="' not in source
    assert 'rel="preload" href="/rerun/re_viewer.js"' in source
    assert "waitForRerunReady" in source
    assert "waitForRerunRenderSettle" in source
    assert "scheduleRerunBundleUncover" in source
    assert "Uncover without blocking mount latency" in source
    assert "swapRerunRecordingInPlace" in source
    assert "handle.add_receiver(recordingUrl, false)" in source
    assert "mountRerunIframe" in source
    assert "mountRerunIframeUntilSuccess" in source
    assert "simViz && (simViz.rerun_ready || simViz.rrd_uri)" in source
    assert "_wait_for_rerun_web_viewer" in source
    apply_selection_source = source.split("async function applySelection")[1].split(
        "async function submitWorkflow"
    )[0]
    assert "await waitForRerunSuccess" in apply_selection_source
    assert 'activeArtifactRender = "rerun"' in apply_selection_source
    fetch_with_timeout_source = source.split("async function fetchWithTimeout")[
        1
    ].split("async function apiJson")[0]
    assert "withMobileAuth" in fetch_with_timeout_source
    api_json_before_fetch = source.split("async function apiJson")[1].split(
        "let resp;"
    )[0]
    assert (
        'throw new Error("Unlock chat with your agent password.");'
        not in api_json_before_fetch
    )
    assert "lastRerunBlobStatus" in source
    assert "lastRerunMountStatus" in source
    assert "mountedRerunRunKey" in source
    assert "already-mounted" in source
    assert "iframe.dataset.rerunRunKey" in source
    assert (
        'rerunIframeLoaded && iframe && !iframe.hidden && iframe.getAttribute("src")'
        in source
    )
    for marker in AGENT_MEDIA_PREVIEW_CONTRACT:
        assert marker in source, f"missing media-preview contract marker: {marker!r}"
    assert "baselineRrdUpdatedAt" in source
    assert "successStreakTarget" in source
    assert "successStreak" in source
    assert "stageAdvanced" in source
    assert "RERUN_MOUNT_SUCCESS" in source
    assert "Rerun iframe mount missing SUCCESS blob/mount state" in source
    # The authenticated blob endpoint is the fallback when the public recording
    # copy is not published yet. (This used to assert `resolveRerunRrdUrl`, a
    # helper that no longer existed — the call sites raised ReferenceError inside
    # a catch, so the substring assertion passed while the fallback was dead.)
    assert "resolveRerunRecordingUrl" in source
    assert "RERUN_BLOB_SUCCESS" in source
    assert "/api/sim-viz/rrd-blob" in source
    assert "resolve_rrd_proxy_target" in source or "rrd_proxy_uri_allowed" in source
    assert "file_uri_path_allowed" in source
    assert "MAX_RRD_PROXY_BYTES" in source
    assert "Refusing to proxy disallowed rrd_uri host" in source
    assert "_AGENT_RRD_PROXY_EMBED" in source
    assert "_STATE_LOCK" in source
    assert "Process-wide lock" in source
    assert "rrdUrl = await resolveRerunRecordingUrl();" in source
    assert "?run_id=" in source
    assert '"/api/sim-viz/status?run_id="' in source
    # Media preview uses authenticated blob URLs; Rerun still avoids parent blob URLs for wasm.
    assert "does not reliably consume parent-created blob URLs" in source
    assert "media_type=artifact_media_type(safe_name)" in source
    assert "apis_used" in source
    assert "format_live_context_block" in source
    assert "match_chat_intent" in source
    assert "renderAssetsSummary" in source
    assert "selectionPayloadFromUi" in source


def test_bootstrap_embeds_run_switching_controls() -> None:

    source = _agent_ui_bundle()
    assert 'id="runIdInput"' in source
    assert 'id="runIdSelect"' in source
    assert 'id="loadRunData"' in source
    assert '@app.post("/sim-viz/load-run")' in source
    assert "available_run_ids" in source
    assert "active_run_id" in source
    assert "_record_sim_viz_run" in source
    assert "_wire_sim2real_run_preview" in source
    assert "Prefer a run-scoped Rerun recording over stale history entries" in source
    assert (
        'preferred and (preferred.render == "rerun" or '
        "(requested_bucket and source_selected))" in source
    )
    assert "held-out simulation camera stream" in source
    assert "reference proxy context" in source
    from npa.cli import agent as agent_module

    stage_runtime = Path(agent_module.__file__).with_name("agent_stage_runtime.py").read_text(
        encoding="utf-8"
    )
    assert "def _artifact_backed_run_details" in stage_runtime
    assert "def _workflow_stage_defs_from_state" in stage_runtime
    assert "artifact presence does not establish execution success" in source
    assert "npa.stage-evidence/v1" in source
    assert "runDetailsRequestId" in source
    assert "runDetailsAbortController" in source
    assert "execution status unavailable" in source
    assert "Never let a sparse update erase richer artifact fields from load-run" in source
    assert "Read-only: do not _record/_save here" in source
    assert (
        "Always use the stock demo run id and clear any prior media-artifact preview"
        in source
    )
    status_src = source.split('@app.get("/sim-viz/status")')[1].split(
        '@app.get("/sim-viz/runs")'
    )[0]
    assert "_save_state(state)" not in status_src
    assert "_record_sim_viz_run(state, payload)" not in status_src
    franka_src = source.split("def _wire_franka_demo")[1].split(
        "def _wire_sim2real_run_preview"
    )[0]
    assert '"run_id": "franka-demo"' in franka_src
    assert '"artifact_render": "rerun"' in franka_src
    submit_source = source.split("def submit_sim2real(payload: dict | None = None):")[
        1
    ].split("cat <<'PY' | sudo tee /opt/npa-agent/bootstrap_rrd.py", 1)[0]
    assert "_wire_sim2real_run_preview" in submit_source
    assert '"sim_viz": sim_viz' in submit_source


def test_bootstrap_embeds_artifact_browser_and_endpoints() -> None:
    from npa.cli import agent as agent_module

    source = _agent_ui_bundle()
    assert 'id="artifactPrefix"' in source
    assert 'id="artifactTypeFilter"' in source
    assert 'id="artifactSort"' in source
    assert 'id="runsArtifactsPanel"' in source
    assert 'id="runIdSelect"' in source
    assert "mergeRunsLatestFirst" in source
    assert "available_runs" in source
    assert 'id="artifactList"' in source
    assert 'id="renderedDataSummary"' in source
    assert '@app.get("/artifacts/runs")' in source
    assert '@app.get("/artifacts/run/{{run_id:path}}")' in source
    assert '@app.post("/sim-viz/load-artifact")' in source
    assert 'npa-artifact-discovery-contract" content="s3-source-qualified-v1' in source
    assert "run_ref" in source
    assert "resolve_run_artifacts" in source
    assert "artifact_run_ref" in source
    assert "served_recording_sha256" in source
    # Every artifact must be directly downloadable: streaming download endpoint
    # + a per-artifact Download button wired to it.
    assert '@app.get("/artifacts/download")' in source
    assert (
        'data-action="download-artifact"' in source
        or "data-action='download-artifact'" in source
    )
    assert "async function downloadArtifact(" in source
    assert "/api/artifacts/download?" in source
    # Clicking a stage describes it and inlines its artifacts/info/configs.
    assert '@app.get("/artifacts/stage/{{run_id:path}}")' in source
    assert "async function showStageDetail(" in source
    assert "/api/artifacts/stage/" in source
    assert 'id="stageDetail"' in source
    assert "data-stage-label=" in source
    # Voxel51 / FiftyOne dataset tab.
    assert '@app.get("/fiftyone/dataset/{{run_id:path}}")' in source
    assert "build_fiftyone_dataset" in source
    assert 'id="tabVoxel"' in source
    assert 'id="panelVoxel"' in source
    assert 'data-tab="voxel51"' in source
    assert "async function loadVoxelDataset(" in source
    assert "/api/fiftyone/dataset/" in source
    assert 'id="voxelGrid"' in source
    # Regression: #panelVoxel must be a SIBLING of #panelRerun, not nested inside
    # it. If nested, panelRerun.is-inactive (opacity:0) makes the whole Voxel tab
    # blank when active. Assert panelRerun is fully closed before panelVoxel opens.
    _rr = source.index('id="panelRerun"')
    _vx = source.index('id="panelVoxel"')
    _between = source[_rr:_vx]
    assert _between.count("<div") == _between.count("</div>"), (
        "#panelVoxel appears nested inside #panelRerun (div-balance off)"
    )
    # Loading/viewing an artifact must NOT post a chat message anymore.
    assert 'Loaded artifact `" + String(simViz.artifact_key' not in source
    assert "Select a run or enter a run_id first" in source
    assert "No S3 artifacts found for <code>" in source
    assert "Runs &amp; artifacts" in source or "Runs & artifacts" in source
    assert "latest first" in source
    assert "updateRenderedDataSummary" in source
    assert "_wait_rerun_web_viewer_healthy" in source
    assert (
        'await mountRerunIframeUntilSuccess(String(simViz.camera || "workspace"), 8, loadedRunId)'
        in source
    )
    assert "EnvironmentFile=-/opt/npa-agent/s3.env" in source
    embedded = agent_module._embedded_agent_artifacts_source()
    assert "list_runs" in embedded
    assert "list_artifacts" in embedded


def test_bootstrap_run_finder_filters_by_name_or_id_not_path() -> None:
    """The run finder must let operators find runs by NAME/ID (client-side
    filter), not by an S3 path/category. Discovery is always generic.
    """
    source = _agent_ui_bundle()
    # Field is a run name/ID finder, not an "Artifact prefix" path.
    assert "Find run (name or ID)" in source
    assert "type part of a run name or ID" in source
    assert "Artifact prefix" not in source
    # It filters the discovered run list client-side (no path prefix to the server).
    assert "function runFilterValue()" in source
    assert "const runFilter = runFilterValue().toLowerCase();" in source
    assert 'runFilterInput.addEventListener("input"' in source
    # Discovery is generic (no ?prefix= path); the old prefix-path helper is gone.
    # The picker follows every bounded server cursor, rather than assuming one
    # oversized response is the whole tenant inventory.
    assert "const ARTIFACT_RUN_LIST_LIMIT = 200;" in source
    assert '"/api/artifacts/runs?limit=" + ARTIFACT_RUN_LIST_LIMIT' in source
    assert 'cursor = String(data.next_cursor || "");' in source
    assert "} while (cursor);" in source
    # Typing in the box also triggers a SERVER-side search so runs beyond the
    # newest page (by name/ID) are findable, not just client-side filtering.
    assert "&q=" in source
    assert "refreshArtifactRuns(value)" in source
    assert "discoverFromPrefix" not in source
    assert "artifactPrefixValue" not in source


def test_bootstrap_artifact_stage_selector_and_clickable_timeline() -> None:
    """The stages/artifact browser must let you choose a workflow-progress step.

    A Stage selector filters the artifact list by pipeline stage, and clicking a
    stage row in the Run Monitor timeline scopes the artifact browser to it. The
    UI lives in agent_ui.html, so assert against the rendered UI bundle.
    """
    source = _agent_ui_bundle()
    # Stage selector in the artifact browser.
    assert 'id="artifactStageFilter"' in source
    assert "function artifactStageFilterValue()" in source
    assert "function deriveArtifactStage(key, runId, wrapper)" in source
    assert "function populateArtifactStageFilter(artifacts, runId)" in source
    assert "populateArtifactStageFilter(artifacts, runId);" in source
    # Deeply-nested runs (<run>/<workflow-name>/<stage>/...) expose real stages.
    assert "function runStageWrapper(artifacts, runId)" in source
    # Stage participates in filtering and re-renders on change.
    assert (
        "if (stageFilter && deriveArtifactStage(item.key, runId, stageWrapper) !== stageFilter) return false;"
        in source
    )
    assert '["artifactStageFilter", "artifactTypeFilter", "artifactSort"]' in source
    # Timeline stage rows are tagged and clickable to drive the stage filter.
    assert "stage_key: stageKey," in source
    assert 'data-stage-key="' in source
    assert ".stage-item[data-stage-key]" in source
    # The filter's stage derivation must match the timeline/backend compound keys
    # (e.g. eval/heldout) so clicking a stage filters instead of yielding nothing.
    assert 'if (first === "eval" && parts[1]) return "eval/" + parts[1];' in source
    # Workflow status resolves the run generically (no path prefix required).
    assert 'const status = await loadJson("/api/workflows/sim2real/status");' in source


def test_data_factory_recording_note_wired_in_apply_loaded_artifact() -> None:
    """A physical-ai-data-factory .rrd must be recognised as its own recording
    type (in the embedded agent bootstrap) so the Rerun viewer shows
    augmented-frame guidance, not the Sim2Real held-out-camera / Franka note —
    both applications write reports/sim2real.rrd.

    These live in the bootstrap template and its embedded viewer runtime, so
    this is a source-text regression guard across the generated backend inputs.
    """
    source = _agent_source()
    # The DF recording detector is defined and keyed on the app id.
    assert "def _is_data_factory_recording(key: str) -> bool:" in source
    assert 'DATA_FACTORY_APP_ID = "physical-ai-data-factory"' in source
    # Path-boundary match (segment), not a bare substring.
    assert '(DATA_FACTORY_APP_ID + "/") in str(key or' in source
    # The DF-specific branch (note + preview_entity) is present and precedes S2R.
    assert "if _is_data_factory_recording(key):" in source
    assert 'sim_viz["preview_entity"] = "augmented"' in source
    assert "Physical AI Data Factory recording loaded." in source
    # The Sim2Real camera label must NOT be applied to DF recordings.
    assert (
        "_is_sim2real_pipeline_recording(key) and not _is_data_factory_recording(key)"
        in source
    )


def test_bootstrap_visualize_run_selector_lists_discovered_runs() -> None:
    """Discovered runs must be choosable to visualize, and the Rerun viewer must
    remain present.

    Regression guard: runs discovered under a custom artifact prefix (e.g.
    physical-ai-data-factory) must be surfaced in the run selector by unioning
    server-side known runs with discovered runs (latest-first), not clobbering.
    """
    source = _agent_ui_bundle()
    # Rerun viewer + run selector still present.
    assert 'id="rerunFrame"' in source
    assert 'id="panelRerun"' in source
    assert 'id="runIdSelect"' in source
    # Generic discovery feeds the discovered-runs set (server-search unions in).
    assert "discoveredArtifactRuns = runs;" in source
    # The run selector is a UNION of known + discovered runs (does not clobber).
    assert "mergeRunsLatestFirst(knownAvailableRuns, discoveredArtifactRuns)" in source
    assert 'fillRunSelectOptionsRich(document.getElementById("runIdSelect")' in source


def test_bootstrap_run_history_uses_source_qualified_index() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert '"sim_viz_runs": []' not in source
    assert "if not isinstance(runs, dict):" in source
    assert "history_key = run_ref or run_id" in source
    assert "runs[history_key] = snapshot" in source
    assert 'state["active_run_id"] = run_id' in source
    assert (
        "Never let a sparse update erase richer artifact fields from load-run" in source
    )


def test_bootstrap_ui_strips_url_credentials() -> None:
    source = _agent_source()
    assert "location.username" in source
    assert "location.password" in source
    assert "history.replaceState" in source
    assert 'location.protocol + "//" + location.host + location.pathname' in source
    assert "_agent_strip_url_credentials_js" in source
    assert "stripUrlCredentials" in source


def test_bootstrap_ui_fetch_uses_credentials_include() -> None:

    source = _agent_ui_bundle()
    assert 'credentials: "include"' in source
    assert 'credentials: "same-origin"' not in source
    assert "setChatBusy(true)" in source
    assert "setChatBusy(false)" in source
    assert "if (btn) btn.disabled = busy;" in source
    assert "if (input) input.disabled = busy;" in source
    assert "JSON.stringify(value)" in source
    assert "JSON.stringify(assets.selection" not in source


def test_default_run_discovery_is_generic_not_hardcoded() -> None:
    """Default (no-prefix) run discovery must scan the bucket generically
    (enumerate category folders under every root from S3), NOT hardcode any
    workflow path, and drop the agent's own infra roots from the listing."""
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    # Generic scan across all roots in configured buckets; no hardcoded workflow
    # prefixes. The no-prefix endpoint calls the multi-bucket cached wrapper.
    assert "list_runs_cached_multi(" in source
    assert "exclude=_discovery_exclude_roots()" in source
    assert "AGENT_DEFAULT_WORKFLOW_PREFIXES" not in source
    # Per-run lookup falls back to a generic cross-category, cross-bucket find.
    runtime = Path(agent_module.__file__).with_name("agent_stage_runtime.py").read_text(
        encoding="utf-8"
    )
    assert "find_run_artifacts_across_buckets(" in runtime


def test_run_details_resolves_run_generically_by_id() -> None:
    """Stage determination must resolve a run generically by id (across all
    categories under the run root) so any run shows real artifact-backed stages
    instead of the generic sim2real 'not_run' template — no path/prefix required.
    """
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).with_name("agent_stage_runtime.py").read_text(
        encoding="utf-8"
    )
    # Backend resolves the run generically across categories (no prefix needed).
    assert "def _artifact_backed_run_details(" in source
    assert "resource_bucket: str = \"\"" in source
    assert "find_run_artifacts_across_buckets(" in source
    # Frontend loads run details / run by id WITHOUT a path prefix.
    ui = _agent_ui_bundle()
    assert '"/api/workflows/sim2real/runs/" + encodeURIComponent(target)' in ui
    assert "body: JSON.stringify({ run_id: targetRunId, run_ref: targetRunRef })" in ui
    assert 'entry.source_type === "artifact_storage"' in ui
    assert "loadArtifactsForSelectedRun(chosen, null, entry, { pendingSelection: true })" in ui
    assert "prefix: artifactPrefixValue()" not in ui
    assert 'params.set("resource_bucket", resourceBucket)' in ui
    assert 'params.set("resolved_prefix", resolvedPrefix)' in ui
    assert 'params.set("source_selected", "1")' in ui
    assert '"stages succeeded"' not in ui


def test_bootstrap_chat_has_scroll_to_bottom_button() -> None:
    """The chat log ships a jump-to-latest arrow wired to scroll to the end."""
    source = _agent_ui_bundle()
    assert 'id="chatScrollBottom"' in source
    assert 'class="chat-log-wrap"' in source
    assert "function scrollChatToBottom(" in source
    assert "function updateChatScrollButton(" in source
    # Wired: scroll listener toggles the arrow; click jumps to the end.
    assert 'chatLogEl.addEventListener("scroll", updateChatScrollButton' in source
    assert "scrollChatToBottom(true)" in source


def test_bootstrap_system_prompt_no_localhost() -> None:

    source = _agent_ui_bundle()
    assert "Never suggest localhost" in source
    assert "/api/sim-viz/load-franka-demo" in source
    assert (
        "localhost:8080"
        not in source.split("_agent_system_prompt")[1].split("return")[0]
    )


def test_resolve_deploy_llm_credentials_reads_credentials(monkeypatch) -> None:
    monkeypatch.setattr(
        "npa.clients.credentials.load_credentials",
        lambda: type("Creds", (), {"token_factory_api_key": "tf-test-key"})(),
    )
    from npa.cli.agent import _resolve_deploy_llm_credentials

    key, model = _resolve_deploy_llm_credentials()
    assert key == "tf-test-key"
    assert model == "nvidia/Cosmos3-Super-Reasoner"


def test_normalize_llm_models_supports_repeated_and_csv_values() -> None:
    models = _normalize_llm_models(
        [
            "nvidia/Cosmos3-Super-Reasoner,meta-llama/Llama-3.3-70B-Instruct",
            "Qwen/Qwen2.5-VL-72B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct",
        ]
    )
    assert models[0] == "nvidia/Cosmos3-Super-Reasoner"
    assert "meta-llama/Llama-3.3-70B-Instruct" in models
    assert "Qwen/Qwen2.5-VL-72B-Instruct" in models


def test_agent_status_json(monkeypatch) -> None:
    deployment = {
        "deployment_id": "npa-agent-test",
        "deployment_name": "agent",
        "project_alias": "us-central1",
        "runtime_namespace": "us-central1/agent",
        "repository": "org/repo",
        "branch": "codex/test",
        "commit": "a" * 40,
        "source_tree": "b" * 40,
        "short_commit": "a" * 12,
        "workspace_label": "Workspace",
        "bootstrap_timestamp": "2026-08-10T00:00:00Z",
    }
    monkeypatch.setattr(
        "npa.cli.agent._agent_record",
        lambda project, name: {
            "public_ip": "8.8.8.8",
            "agent_url": "https://203.0.113.50/",
            "public_url": "https://203.0.113.50/",
            "public_https": True,
            "direct_url": "http://203.0.113.50:8088/",
            "rerun_url": "https://203.0.113.50/rerun/",
            "sim_viz_url": "https://203.0.113.50/rerun/",
            "sim_assets_url": "https://203.0.113.50/assets/",
            "cameras_api_url": "https://203.0.113.50/assets/api/sim-assets/cameras",
            "auth_secret_path": "/tmp/agent-auth",
            "deployment": deployment,
            "llm": {
                "provider": "token_factory",
                "model": "nvidia/Cosmos3-Super-Reasoner",
            },
        },
    )
    monkeypatch.setattr("npa.cli.agent._load_auth_secret", lambda _: ("npa", "secret"))
    monkeypatch.setattr(
        "npa.cli.agent._health",
        lambda *_args, **_kwargs: (True, 200),
    )
    monkeypatch.setattr(
        "npa.cli.agent.fetch_live_deployment", lambda *_args, **_kwargs: deployment
    )

    result = runner.invoke(app, ["status", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["health"] is True
    assert payload["ui_status_code"] == 200
    assert payload["rerun_status_code"] == 200
    assert payload["sim_viz_url"].endswith("/rerun/")
    assert payload["sim_assets_url"].endswith("203.0.113.50/assets/")
    assert payload["cameras_api_url"].endswith("/assets/api/sim-assets/cameras")


def test_verify_live_accepts_non_us_central1_region(monkeypatch) -> None:
    """Route C deploys with --region eu-north1; verify-live must not hard-fail
    non-us-central1 regions (regression for the README Route C failure)."""
    monkeypatch.setattr(
        "npa.cli.agent._agent_record",
        lambda project, name: {
            "public_ip": "8.8.8.8",
            "region": "eu-north1",
            "auth_secret_path": "/tmp/agent-auth",
        },
    )
    monkeypatch.setattr("npa.cli.agent._is_routable_public_ip", lambda _ip: True)

    def _boom(_path: str) -> tuple[str, str]:
        raise ValueError("stop-after-region-gate")

    monkeypatch.setattr("npa.cli.agent._load_auth_secret", _boom)

    result = runner.invoke(app, ["verify-live"])

    assert result.exit_code == 1
    assert "region mismatch" not in result.output
    assert "stop-after-region-gate" in result.output


def test_verify_live_requires_a_recorded_region(monkeypatch) -> None:
    monkeypatch.setattr(
        "npa.cli.agent._agent_record",
        lambda project, name: {
            "public_ip": "8.8.8.8",
            "region": "",
            "auth_secret_path": "/tmp/agent-auth",
        },
    )
    monkeypatch.setattr("npa.cli.agent._is_routable_public_ip", lambda _ip: True)

    result = runner.invoke(app, ["verify-live"])

    assert result.exit_code == 1
    assert "missing its deploy region" in result.output


def test_verify_live_runs_pytests(monkeypatch) -> None:
    workflow_status_timeouts: list[float] = []
    deployment = {
        "deployment_id": "npa-agent-test",
        "deployment_name": "agent",
        "project_alias": "us-central1",
        "runtime_namespace": "us-central1/agent",
        "repository": "nebius/nebius-physical-ai",
        "branch": "codex/test",
        "commit": "a" * 40,
        "source_tree": "b" * 40,
        "short_commit": "a" * 12,
        "workspace_label": "NPA Workbench",
        "bootstrap_timestamp": "2026-08-10T00:00:00Z",
    }

    class _Resp:
        def __init__(
            self, payload: dict[str, object] | str | bytes, *, status_code: int = 200
        ) -> None:
            self.status_code = status_code
            self._payload = payload
            if isinstance(payload, (bytes, str)):
                self.content = (
                    payload.encode("utf-8") if isinstance(payload, str) else payload
                )
                self.text = (
                    payload.decode("utf-8") if isinstance(payload, bytes) else payload
                )
            else:
                self.content = b""
                self.text = ""

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            if isinstance(self._payload, dict):
                return self._payload
            return {"ok": True}

        @property
        def headers(self) -> dict[str, str]:
            if isinstance(self._payload, (bytes, str)):
                return {"content-type": "application/octet-stream"}
            return {"content-type": "application/json"}

    class _Proc:
        def __init__(self, code: int = 0) -> None:
            self.returncode = code

    monkeypatch.setattr(
        "npa.cli.agent._agent_record",
        lambda project, name: {
            "public_ip": "8.8.8.8",
            "region": "us-central1",
            "agent_url": "https://203.0.113.50/",
            "public_url": "https://203.0.113.50/",
            "public_https": True,
            "direct_url": "http://203.0.113.50:8088/",
            "rerun_url": "https://203.0.113.50/rerun/",
            "sim_viz_url": "https://203.0.113.50/rerun/",
            "sim_assets_url": "https://203.0.113.50/assets/",
            "cameras_api_url": "https://203.0.113.50/assets/api/sim-assets/cameras",
            "auth_secret_path": "/tmp/agent-auth",
            "deployment": deployment,
        },
    )
    monkeypatch.setattr("npa.cli.agent._load_auth_secret", lambda _: ("npa", "secret"))
    monkeypatch.setattr("npa.cli.agent._health", lambda *_args, **_kwargs: (True, 200))

    def _fake_http_get(url, *_args, **_kwargs):
        url_s = str(url)
        if url_s.endswith("/api/deployment"):
            return _Resp(deployment)
        if url_s.endswith("/api/tools"):
            return _Resp({"tool_refs": [f"tool.{idx}" for idx in range(19)]})
        if url_s.endswith("/api/sim-assets"):
            return _Resp({"scene_spec": {"schema": "x"}, "robot_spec": {"schema": "y"}})
        if url_s.endswith("/api/sim-assets/cameras"):
            return _Resp(
                {
                    "cameras": [
                        {
                            "name": "workspace",
                            "placement": "stock_workspace",
                            "fov": 60.0,
                        },
                        {"name": "wrist", "placement": "stock_ee_mounted", "fov": 90.0},
                    ],
                    "selected": ["workspace"],
                }
            )
        if url_s.endswith("/api/sim-assets/selection"):
            return _Resp(
                {
                    "scene_spec_uri": "stock://scene/default",
                    "assets_uri": "",
                    "robot_spec_uri": "stock://robot/franka",
                    "cameras_uri": "stock://cameras/default",
                    "robot_preset": "franka",
                    "sim_backend": "isaac",
                }
            )
        if url_s.endswith("/api/session"):
            return _Resp({"chat_history": [], "selection": {}})
        if url_s.endswith("/api/access"):
            return _Resp(
                {
                    "ok": True,
                    "apiVersion": "npa.agent.access/v1",
                    "status": "available",
                    "scope": "single_project",
                    "identity": {
                        "tenant_id": "tenant-id",
                        "deployment_project_id": "project-id",
                        "deployment_project_name": "default",
                    },
                    "capabilities": {},
                    "projects": [],
                    "errors": [],
                    "refreshed_at": "2026-08-06T23:30:00+00:00",
                }
            )
        if url_s.endswith("/api/sim-viz/status"):
            params = _kwargs.get("params") or {}
            run_id = str(params.get("run_id") or "")
            return _Resp(
                {
                    "run_id": run_id or "agent-run-123",
                    "rerun_ready": True,
                    "rrd_uri": "/api/sim-viz/rrd",
                    "stage": "stage_14_rerun_viz" if run_id else "demo",
                }
            )
        if url_s.endswith("/api/sim-viz/rrd") or url_s.endswith(
            "/api/sim-viz/rrd-blob"
        ):
            return _Resp(b"RRD" * 32, status_code=200)
        if url_s.endswith("/api/health"):
            return _Resp({"ok": True})
        if url_s.endswith("/api/infra/k8s"):
            return _Resp({"ok": True, "agent_npa_ready": True})
        if "/api/resources" in url_s:
            return _Resp(
                {
                    "ok": True,
                    "categories": [
                        {
                            "id": "project",
                            "status": "configured",
                            "configured_count": 1,
                            "discovered_count": 0,
                        },
                        {
                            "id": "network",
                            "status": "empty",
                            "configured_count": 0,
                            "discovered_count": 0,
                        },
                    ],
                }
            )
        if url_s.endswith("/api/workflows/sim2real/status"):
            workflow_status_timeouts.append(float(_kwargs["timeout"]))
            return _Resp({"latest_submit": {"run_id": "agent-run-123"}, "sim_viz": {"stage": "demo"}})
        if url_s.endswith("/welcome"):
            return _Resp("<html>NPA Agent is running</html>", status_code=200)
        if url_s.endswith("/healthz"):
            return _Resp('{"ok":true}', status_code=200)
        if "/rerun/" in url_s:
            return _Resp(b"console.log('rerun');", status_code=200)
        if url_s.rstrip("/").endswith(("203.0.113.50", ":8088")):
            html = (
                f'<html><head><meta name="viewport" content="width=device-width, initial-scale=1">'
                f'<meta name="npa-ui-version" content="{AGENT_UI_VERSION}"></head>'
                "<body>"
                '<div id="tabMain"></div><div id="tabRerun"></div>'
                '<div id="agentAccessPanel"></div><button id="agentAccessRefresh"></button>'
                '<select id="agentAccessProjectSelect"></select>'
                '<article class="access-project-detail">No searchable artifact bucket.</article>'
                '<script>function refreshAccess(){ fetch("/api/access"); }</script>'
                '<div id="stagesPanel"><h3>Stages</h3>'
                '<div class="stages-run-picker">'
                '<select id="stagesRunSelect"></select>'
                '<label>Search NPA workflow/artifact runs</label>'
                '<input id="stagesRunInput" />'
                '<button id="stagesLoadRun"></button></div></div>'
                '<div id="tenantResourcesPanel"><h3>Tenant resources</h3>'
                '<button id="tenantResourcesRefresh"></button>'
                'Accessible / discovered; Configured references</div>'
                '<script>function loadSelectedRun(){} function syncRunChooserFields(){} '
                'function filterStagesRunSelect(){} function resolveStagesRunChoice(){}</script>'
                '<div id="renderModeVideo"></div><div id="artifactPreviewHost"></div>'
                '<div id="viewerPaneMedia"></div><div id="rerunBundleCover"></div>'
                '<button id="renderModeFoxglove"></button>'
                '<div id="viewerPaneFoxglove"><div id="foxgloveHost"></div></div>'
                "<script>function ensureFoxgloveViewer(){} function mountFoxgloveViewer(){} "
                'fetch("/api/foxglove/config");</script>'
                '<button id="describeVisual"></button>'
                '<button id="chatDrawerToggle" class="chat-fab"></button>'
                '<button id="chatDrawerClose"></button>'
                '<form id="chatForm"></form><div id="mobileChatAuth"></div>'
                '<script>function wireUi(){} function sendChat(){} function activateMainTab(){} '
                'function authenticatedPreviewObjectUrl(){} function waitUntilRerunPastBundleSplash(){} '
                'function scheduleRerunBundleUncover(){} function swapRerunRecordingInPlace(){} '
                'function safeHideRerunBundleCover(){} function captureVisualContext(){} '
                'function describeVisual(){} function enqueueChatJob(){} function processChatQueue(){} '
                'function queueChatText(){} function waitForQualityRerunFrame(){} '
                'function captureCanvasDataUrl(){} function ensureRerunCaptureBridge(){} '
                'function pickBestIframeCanvas(){} function sampleFrameStats(){} '
                'function openFullChatTab(){} '
                'function refreshTenantResources(){ fetch("/api/resources"); } '
                'do not prefetch .rrd bytes; skipUserAppend; Describe this — capturing; '
                'async function loadArtifact(payload){ await swapRerunRecordingInPlace(); } '
                '<button id="openFullChatTab"></button>'
                "async function refresh(){} "
                "handle.add_receiver(recordingUrl, false); "
                'initNpaAgentUi; mobile-agent; history.replaceState(null, "", ""); '
                "location.username; location.password; "
                "Warm Rerun assets before revealing the iframe; Preparing viewer…; "
                "Uncover without blocking mount latency; non-blank canvas; "
                "viewer-focus; thinking-ellipsis; [npa-visual-feedback]; visual_context; "
                "transform-origin: bottom right; "
                "Loading video preview…; URL.createObjectURL(blob)"
                "</script></body></html>"
            )
            return _Resp(html, status_code=200)
        return _Resp(
            {"ok": True, "tool_ref": "tool.0", "argv_template": ["echo", "ok"]}
        )

    def _fake_http_post(url, *_args, **_kwargs):
        url_s = str(url)
        if url_s.endswith("/api/chat"):
            payload = (_kwargs.get("json") or {}) if isinstance(_kwargs, dict) else {}
            messages = payload.get("messages", []) if isinstance(payload, dict) else []
            last_content = ""
            if isinstance(messages, list) and messages:
                tail = messages[-1]
                if isinstance(tail, dict):
                    last_content = str(tail.get("content") or "")
            if "create 2-step sim2real workflow" in last_content.lower():
                return _Resp(
                    {
                        "ok": True,
                        "grounded": True,
                        "reply": "**Generated npa.workflow/v0.0.1 spec**",
                        "workflow_yaml": "apiVersion: npa.workflow/v0.0.1\nkind: Workflow\nmetadata:\n  name: sim2real-two-step\nstates:\n  augment: {}\n  envgen: {}\n",
                        "apis_used": ["workflows/draft", "workflows/validate"],
                    }
                )
            if (
                "add an open source repo" in last_content.lower()
                or "leisaac" in last_content.lower()
            ):
                from npa.cli.agent_chat import format_onboard_solution

                return _Resp(
                    {
                        "ok": True,
                        "grounded": True,
                        "reply": format_onboard_solution(),
                        "apis_used": ["tools", "workflows/validate", "workflows/plan"],
                    }
                )
            return _Resp(
                {
                    "ok": True,
                    "grounded": True,
                    "reply": "**Sim2Real status**\n- **run_id**: `agent-run-123`\n- **stage**: `demo`",
                    "apis_used": ["sim-viz/status"],
                }
            )
        if url_s.endswith("/api/sim-assets/selection"):
            return _Resp(
                {"ok": True, "selection": {"scene_spec_uri": "stock://scene/default"}}
            )
        if url_s.endswith("/api/workflows/sim2real/submit"):
            return _Resp(
                {
                    "ok": True,
                    "run_id": "agent-run-123",
                    "sim_viz": {
                        "run_id": "agent-run-123",
                        "stage": "stage_14_rerun_viz",
                        "rrd_uri": "/api/sim-viz/rrd",
                        "rerun_ready": True,
                    },
                }
            )
        if url_s.endswith("/api/workflows/submit"):
            return _Resp(
                {
                    "ok": True,
                    "submit_mode": "agent-live-infra-dry-run",
                    "scheduler_plan": {"ok": True},
                    "run_id": "verify-live-agent-infra",
                }
            )
        if url_s.endswith("/api/sim-viz/load-franka-demo"):
            return _Resp(
                {
                    "ok": True,
                    "sim_viz": {"rerun_ready": True, "rrd_uri": "/api/sim-viz/rrd"},
                }
            )
        if url_s.endswith("/api/sim-viz/camera-preview"):
            return _Resp(
                {"ok": True, "entity_path": "world/camera_frustums/workspace/frustum"}
            )
        return _Resp({"ok": True})

    monkeypatch.setattr("npa.cli.agent.httpx.get", _fake_http_get)
    monkeypatch.setattr("npa.cli.agent.httpx.post", _fake_http_post)
    from npa.agent_rerun_bundle_check import BundleBudgetResult

    monkeypatch.setattr(
        "npa.agent_rerun_bundle_check.check_rerun_bundle_load_budget",
        lambda *_args, **_kwargs: BundleBudgetResult(
            ok=True,
            errors=(),
            fetches=(),
            ui_version=AGENT_UI_VERSION,
        ),
    )
    calls: list[list[str]] = []

    def _fake_run(args, **_kwargs):
        calls.append(list(args))
        return _Proc(0)

    monkeypatch.setattr("npa.cli.agent.subprocess.run", _fake_run)

    result = runner.invoke(app, ["verify-live"])
    assert result.exit_code == 0, result.output
    assert "verify-live: ok" in result.output
    assert workflow_status_timeouts == [30.0]
    assert calls == [
        [
            "npa/.venv/bin/python",
            "-m",
            "pytest",
            "npa/tests/smoke/test_agent_smoke.py",
            "npa/tests/smoke/test_agent_chat_smoke.py",
            "-q",
        ],
        [
            "npa/.venv/bin/python",
            "-m",
            "pytest",
            "npa/tests/cli/test_agent.py",
            "npa/tests/cli/test_agent_workflow.py",
            "-q",
        ],
        [
            "npa/.venv/bin/python",
            "-m",
            "pytest",
            "npa/tests/e2e/test_agent_live.py",
            "-q",
        ],
    ]


def _sample_agent_state(*, rerun_ready: bool = True, stage: str = "demo") -> dict:
    return {
        "sim_viz": {
            "run_id": "agent-run-123",
            "stage": stage,
            "camera": "workspace",
            "rerun_ready": rerun_ready,
            "rrd_updated_at": "2025-06-25T12:00:00+00:00",
        },
        "selection": {
            "robot_preset": "franka",
            "sim_backend": "isaac",
            "scene_spec_uri": "stock://scene/default",
            "robot_spec_uri": "stock://robot/franka",
            "cameras_uri": "stock://cameras/default",
            "assets_uri": "",
            "props": ["cube"],
        },
        "latest_submit": {
            "run_id": "agent-run-123",
            "submitted_at": "2025-06-25T11:00:00+00:00",
        },
        "camera_selection": ["workspace"],
    }


def test_match_chat_intent_status_queries() -> None:
    from npa.cli.agent_chat import match_chat_intent

    assert match_chat_intent("what is the current sim2real status") == "sim2real_status"
    assert match_chat_intent("workflow status please") == "sim2real_status"
    assert match_chat_intent("check simViz status now") == "sim2real_status"
    assert match_chat_intent("status for sim_viz run") == "sim2real_status"
    assert match_chat_intent("watch the sim in rerun") == "watch_sim"
    assert match_chat_intent("tail the simulation timeline") == "watch_sim"
    assert (
        match_chat_intent("open the rerun iframe and show latest timeline")
        == "watch_sim"
    )
    assert match_chat_intent("show stage badge overlay for this run") == "watch_sim"
    assert (
        match_chat_intent("poll sim-viz/status and refresh rerun iframe") == "watch_sim"
    )
    assert match_chat_intent("rerun blob iframe until SUCCESS") == "watch_sim"
    assert match_chat_intent("RERUN_BLOB_IFRAME_UNTIL_SUCCESS") == "watch_sim"
    assert match_chat_intent("rerunblobiframeuntilsuccess") == "watch_sim"
    assert match_chat_intent("Rerun blob iframe;\nuntil SUCCESS.") == "watch_sim"
    assert match_chat_intent("rerun blob/iframe until SUCCESS") == "watch_sim"
    assert (
        match_chat_intent("rerun blob + iframe until success, keep retrying mount")
        == "watch_sim"
    )
    assert match_chat_intent("rerun blob iframe till successful mount") == "watch_sim"
    assert match_chat_intent("rerunblobiframetilsuccess") == "watch_sim"
    assert (
        match_chat_intent("rerun blob iframe until successful for run-id scoped checks")
        == "watch_sim"
    )
    assert match_chat_intent("blob+iframe until success") == "watch_sim"
    assert match_chat_intent("blobiframeuntilsuccess") == "watch_sim"
    assert (
        match_chat_intent("until SUCCESS rerun blob iframe for this run") == "watch_sim"
    )
    assert (
        match_chat_intent(
            "keep trying rerun iframe until both blob and mount are success"
        )
        == "watch_sim"
    )
    assert (
        match_chat_intent("wait for RERUN_BLOB_SUCCESS and RERUN_MOUNT_SUCCESS")
        == "watch_sim"
    )
    assert (
        match_chat_intent("watch sim-viz/status until rrd_uri is non-empty")
        == "watch_sim"
    )
    assert match_chat_intent("watch the sim until SUCCESS") == "watch_sim"
    assert match_chat_intent("watch the sim timeline until SUCCESS") == "watch_sim"
    assert (
        match_chat_intent("watch sim-viz timeline until SUCCESS and keep retrying")
        == "watch_sim"
    )
    assert (
        match_chat_intent("watch sim-viz/status until rrd_uri is not empty")
        == "watch_sim"
    )
    assert (
        match_chat_intent("watch sim-viz/status until rrd_uri is populated")
        == "watch_sim"
    )
    assert (
        match_chat_intent("watch rrduri for active runid until SUCCESS") == "watch_sim"
    )
    assert (
        match_chat_intent("keep monitoring rerun until rrd_uri is set") == "watch_sim"
    )
    assert (
        match_chat_intent("watchrrduriuntilsuccess for runid agent-run-123")
        == "watch_sim"
    )
    assert (
        match_chat_intent("rrduriuntilsuccess for runid agent-run-123") == "watch_sim"
    )
    assert (
        match_chat_intent("watchsimuntilsuccess for runid agent-run-123") == "watch_sim"
    )
    assert match_chat_intent("runidrrduriuntilsuccess") == "watch_sim"
    assert match_chat_intent("runid/rrduri SUCCESS for the active run") == "watch_sim"
    assert match_chat_intent("runidrrdurisuccess") == "watch_sim"
    assert (
        match_chat_intent("runidscoped rerun blob iframe until success") == "watch_sim"
    )
    assert (
        match_chat_intent("runid + stage scoped rerun blob iframe until SUCCESS")
        == "watch_sim"
    )
    assert (
        match_chat_intent("runid stage scoped rerun blob iframe until SUCCESS")
        == "watch_sim"
    )
    assert (
        match_chat_intent(
            "rerun blob iframe until SUCCESS with runid and stage matching"
        )
        == "watch_sim"
    )
    assert (
        match_chat_intent("rrdurinonempty until SUCCESS for active runid")
        == "watch_sim"
    )
    assert (
        match_chat_intent("rrdurinotempty until SUCCESS for active runid")
        == "watch_sim"
    )
    assert (
        match_chat_intent(
            "Enhance NPA agent chat intent routing and Rerun blob iframe until SUCCESS. "
            "Branch feat/npa-agent. Bootstrap rtxpro/agent after changes."
        )
        == "watch_sim"
    )
    assert (
        match_chat_intent("load franka in rerun and keep blob iframe until SUCCESS")
        == "watch_sim"
    )
    assert match_chat_intent("load franka in rerun") == "load_franka"
    assert match_chat_intent("show me the sim assets selection") == "sim_assets"
    assert match_chat_intent("list cameras") == "cameras"
    assert match_chat_intent("what tools can workbench do") == "tools_catalog"
    assert match_chat_intent("configure S3 bucket") == "configure_s3"
    assert match_chat_intent("setup cosmos3") == "cosmos3"
    assert match_chat_intent("create 2-step sim2real workflow") == "create_workflow"
    assert (
        match_chat_intent("generate two-step sim2real workflow yaml")
        == "create_workflow"
    )
    assert (
        match_chat_intent("generate an example simple workflow YAML")
        == "create_workflow"
    )
    assert match_chat_intent("camera angle inspector with frustum preview") == "cameras"
    assert (
        match_chat_intent("specify scene robot cameras props selection") == "sim_assets"
    )
    assert match_chat_intent("hello there") is None


def test_build_grounded_status_reply_unpacks_fields() -> None:
    from npa.cli.agent_chat import build_grounded_reply

    state = _sample_agent_state()
    reply = build_grounded_reply("sim2real_status", state, ["tool.a"], rerun_ready=True)
    assert "**run_id**" in reply
    assert "`agent-run-123`" in reply
    assert "**stage**" in reply
    assert "`demo`" in reply
    assert "GET /api" not in reply


def test_build_grounded_watch_sim_reply_mentions_status_polling_and_success() -> None:
    from npa.cli.agent_chat import build_grounded_reply

    state = _sample_agent_state()
    reply = build_grounded_reply("watch_sim", state, ["tool.a"], rerun_ready=True)
    assert "/api/sim-viz/status" in reply
    assert "rrd_uri" in reply
    assert "SUCCESS" in reply
    assert "**watch_stage**" in reply
    assert "**watch_mode**" in reply
    assert "run_id` + `stage`" in reply


def test_format_live_context_block_redacts_secrets() -> None:
    from npa.cli.agent_chat import format_live_context_block

    block = format_live_context_block(_sample_agent_state())
    assert "agent-run-123" in block
    assert "password" not in block.lower()
    assert "credentials" not in block.lower()


def test_apis_for_intent_includes_status_paths() -> None:
    from npa.cli.agent_chat import apis_for_intent

    apis = apis_for_intent("sim2real_status")
    assert "sim-viz/status" in apis
    assert "workflows/sim2real/status" in apis
    watch_apis = apis_for_intent("watch_sim")
    assert "sim-viz/rrd-blob" in watch_apis


def test_bootstrap_embeds_recordings_endpoint() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert '@app.get("/sim-viz/recordings")' in source
    assert "sim_viz_recordings" in source
    assert '"/opt/npa-agent/recordings"' in source
    assert '"recordings"' in source
    assert '"count"' in source
    assert '"size_bytes"' in source
    assert '"updated_at"' in source


def test_bootstrap_chat_copy_yaml_support_present() -> None:

    source = _agent_ui_bundle()
    assert "msg-copy-btn" in source
    assert "extractFencedCode" in source
    assert "copyTextToClipboard" in source


def test_bootstrap_emitted_ui_script_is_valid_javascript(monkeypatch) -> None:
    if not shutil.which("node"):
        return
    from npa.cli import agent as agent_module

    captured: dict[str, str] = {}

    class _DummySsh:
        def upload_file(self, local_path: str, _remote_path: str) -> None:
            try:
                text = Path(local_path).read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return
            if "npa-agent-bootstrap" in _remote_path:
                captured["setup_script"] = text

        def run_or_raise(self, _command: str) -> None:
            return None

        def run(self, _command: str) -> None:
            return None

    monkeypatch.setattr(agent_module, "SSHClient", lambda config: _DummySsh())
    monkeypatch.setattr(
        agent_module, "resolve_ssh_config", lambda **_kwargs: SimpleNamespace(ssh={})
    )

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
        llm_models=[
            "nvidia/Cosmos3-Super-Reasoner",
            "meta-llama/Llama-3.3-70B-Instruct",
        ],
        tf_api_key="",
        nebius_ai_key="",
        public_https=True,
    )

    setup_script = captured["setup_script"]
    shell_proc = subprocess.run(
        ["bash", "-n"],
        input=setup_script,
        text=True,
        capture_output=True,
        check=False,
    )
    assert shell_proc.returncode == 0, shell_proc.stderr
    assert "RERUN_CAPABILITY_NAME_RE" in setup_script
    assert "RERUN_RECORDING_HTTP_PATH" not in setup_script
    assert 'sim_viz["served_recording_sha256"] = hashlib.sha256(' in setup_script
    assert 'sim_viz.pop("served_recording_sha256", None)' in setup_script
    assert "hashlib.sha256(recording_bytes).hexdigest() == bound_sha256" in setup_script
    html_match = re.search(
        r"cat <<'HTML' \| sudo tee /opt/npa-agent/ui\.html >/dev/null\n(?P<html>.*?)\nHTML",
        setup_script,
        flags=re.DOTALL,
    )
    assert html_match, "bootstrap setup script must emit ui.html"
    scripts = re.findall(
        r"<script>(.*?)</script>", html_match.group("html"), flags=re.DOTALL
    )
    assert scripts, "ui.html must include browser JavaScript"
    proc = subprocess.run(
        ["node", "--check", "-"],
        input="\n".join(scripts),
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_bootstrap_recordings_api_in_system_prompt() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert "sim-viz/recordings" in source
    assert "available .rrd recording" in source


def test_bootstrap_uses_unique_remote_setup_script_path() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert "npa-agent-bootstrap-{secrets.token_hex" in source


def test_rrd_publish_uses_request_unique_atomic_temp_path() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    publish = source.split("def _publish_rrd_recording", 1)[1].split(
        "def _safe_artifact_key", 1
    )[0]
    assert "secrets.token_hex(6)" in publish
    assert 'with_suffix(".rrd.tmp")' not in publish


def test_bootstrap_installs_boto3_for_artifact_endpoints() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert "pip install fastapi uvicorn httpx pyyaml boto3" in source


def test_bootstrap_installs_nebius_cli_and_sa_profile() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert "storage.eu-north1.nebius.cloud/cli/install.sh" in source
    assert "--token-file /mnt/cloud-metadata/token" in source
    assert 'nebius_profile = "cursor-sa"' in source
    assert "--profile {nebius_profile}" in source
    assert '"$NEBIUS_BIN" --profile {nebius_profile} iam get-access-token >/dev/null' in source
    assert 'sudo -H "$NEBIUS_BIN" profile create' in source
    assert 'sudo -H "$NEBIUS_BIN" --profile {nebius_profile} iam get-access-token' in source
    assert "nebius CLI binary not found after install" in source
    assert "--parent-id" in source


def test_list_recordings_intent_routing() -> None:
    from npa.cli.agent_chat import apis_for_intent, match_chat_intent

    assert match_chat_intent("list recordings") == "list_recordings"
    assert match_chat_intent("show run history") == "list_recordings"
    assert match_chat_intent("browse available .rrd files") == "list_recordings"
    assert match_chat_intent("switch to a different run recording") == "list_recordings"
    apis = apis_for_intent("list_recordings")
    assert "sim-viz/recordings" in apis
    assert "sim-viz/runs" in apis


def test_list_recordings_grounded_reply() -> None:
    from npa.cli.agent_chat import build_grounded_reply

    state: dict = {}
    reply = build_grounded_reply("list_recordings", state, [])
    assert "recordings" in reply.lower() or "run history" in reply.lower()
    assert "sim-viz/recordings" in reply or "sim-viz/runs" in reply


def test_agent_config_persists_ssh_and_credentials() -> None:
    from npa.cli.agent import AgentConfig

    record = AgentConfig(
        project_alias="rtxpro",
        name="agent",
        project_id="project-1",
        tenant_id="tenant-1",
        region="eu-north1",
        public_ip="203.0.113.50",
        instance_id="instance-1",
        agent_url="https://203.0.113.50/",
        rerun_url="https://203.0.113.50/rerun/",
        sim_viz_url="https://203.0.113.50/rerun/",
        sim_assets_url="https://203.0.113.50/assets/",
        cameras_api_url="https://203.0.113.50/assets/api/sim-assets/cameras",
        auth_user="npa",
        auth_secret_path="/tmp/auth.env",
        llm_provider="token_factory",
        llm_model="nvidia/Cosmos3-Super-Reasoner",
        ssh_key_path="~/.ssh/id_ed25519",
        service_account_id="serviceaccount-abc",
        credentials={
            "service_account_id": "serviceaccount-abc",
            "s3_bucket": "npa-bucket-test",
            "s3_endpoint": "https://storage.eu-north1.nebius.cloud",
            "access_key": "key",
            "secret_key": "secret",
        },
    )
    payload = record.to_dict()
    assert payload["ssh_key_path"] == "~/.ssh/id_ed25519"
    assert payload["service_account_id"] == "serviceaccount-abc"
    assert payload["credentials"]["access_key"] == "key"


def test_resolve_agent_ssh_key_prefers_record_and_cli() -> None:
    from npa.cli.agent import _resolve_agent_ssh_key

    record = {"ssh_key_path": "/record/key"}
    assert _resolve_agent_ssh_key(record, cli_ssh_key="/cli/key") == "/cli/key"
    assert _resolve_agent_ssh_key(record) == "/record/key"


def test_resolve_agent_storage_credentials_prefers_record() -> None:
    from npa.cli.agent import _resolve_agent_storage_credentials

    record = {
        "service_account_id": "serviceaccount-abc",
        "credentials": {
            "service_account_id": "serviceaccount-abc",
            "s3_bucket": "bucket",
            "s3_prefix": "runs",
            "s3_endpoint": "https://storage.eu-north1.nebius.cloud",
            "access_key": "key",
            "secret_key": "secret",
        },
    }
    bucket, prefix, endpoint, access_key, secret_key, sa_id = (
        _resolve_agent_storage_credentials(
            "rtxpro",
            record,
        )
    )
    assert bucket == "bucket"
    assert prefix == "runs"
    assert endpoint.endswith("nebius.cloud")
    assert access_key == "key"
    assert secret_key == "secret"
    assert sa_id == "serviceaccount-abc"


def test_credential_refresh_cannot_replace_recorded_service_account() -> None:
    from npa.cli.agent_access import consistent_agent_service_account_id

    assert (
        consistent_agent_service_account_id("serviceaccount-a", "serviceaccount-a")
        == "serviceaccount-a"
    )
    assert (
        consistent_agent_service_account_id("serviceaccount-a", "")
        == "serviceaccount-a"
    )
    with pytest.raises(ValueError, match="different service account"):
        consistent_agent_service_account_id("serviceaccount-a", "serviceaccount-b")


def test_bootstrap_stages_nebius_env_and_record_ssh_key() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert "EnvironmentFile=-/opt/npa-agent/nebius.env" in source
    assert "_write_agent_nebius_env" in source
    assert "bootstrap_agent_environment" in source
    assert "--refresh-credentials" in source
    assert "--ssh-key" in source
    assert "_resolve_agent_ssh_key" in source
    assert "_creds_from_terraform_state" in source


def test_agent_nebius_env_uses_metadata_profile_without_static_iam_token() -> None:
    from npa.cli import agent as agent_module

    commands: list[str] = []

    class SSH:
        def run_or_raise(self, command: str) -> None:
            commands.append(command)

    agent_module._write_agent_nebius_env(
        SSH(),
        project_alias="agent-project",
        agent_name="agent",
        project_id="project-test",
        tenant_id="tenant-test",
        region="eu-north1",
        service_account_id="serviceaccount-test",
        bucket="bucket-test",
        endpoint="https://storage.example",
        access_key="synthetic-access",
        secret_key="synthetic-secret",
        iam_token="synthetic-stale-token",
    )

    assert len(commands) == 1
    encoded = shlex.split(commands[0])[1]
    env_text = base64.b64decode(encoded).decode("utf-8")
    assert "NEBIUS_PROFILE=cursor-sa" in env_text
    assert "NPA_NEBIUS_CONFIG=/root/.nebius/config.yaml" in env_text
    assert "NPA_NEBIUS_CREDENTIAL_SOURCE=instance_metadata" in env_text
    assert "NEBIUS_IAM_TOKEN" not in env_text
    assert "NPA_NEBIUS_IAM_TOKEN" not in env_text
    assert "TF_VAR_iam_token" not in env_text
    assert "synthetic-stale-token" not in env_text


def test_bootstrap_verifies_attached_identity_with_project_scoped_fallback() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert "attached service-account verification failed" in source
    assert "expected_sa={expected_agent_service_account_id}" in source
    assert "isinstance(value, str) and value == expected" in source
    assert '[[ "$whoami_json" != *"$expected_sa"* ]]' not in source
    assert "iam project list --parent-id \"$expected_tenant\" --all" in source
    assert "iam project get --id \"$expected_project\"" in source
    assert "forcing a broad tenant editors grant" in source
    assert "env -u NEBIUS_IAM_TOKEN -u NPA_NEBIUS_IAM_TOKEN" in source


def test_creds_from_terraform_state(monkeypatch) -> None:
    from npa.cli.agent import _creds_from_terraform_state

    class _Tf:
        bucket = "npa-bucket-test"
        endpoint = "https://storage.us-central1.nebius.cloud"
        access_key = "AKIA"
        secret_key = "SECRET"

    monkeypatch.setattr("npa.cli.agent.resolve_terraform_state", lambda _p: _Tf())
    monkeypatch.setattr(
        "npa.cli.agent._resolve_agent_service_account_id",
        lambda _project, _record: "serviceaccount-abc",
    )
    record = {
        "project_id": "project-1",
        "tenant_id": "tenant-1",
        "region": "us-central1",
    }
    creds = _creds_from_terraform_state("rtxpro", record)
    assert creds is not None
    assert creds["nebius_api_key"] == "AKIA"
    assert creds["s3_bucket"] == "npa-bucket-test"
    assert creds["service_account_id"] == "serviceaccount-abc"


def test_bootstrap_embed_uses_placeholder_for_agent_chat() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert "_AGENT_CHAT_EMBED" in source
    assert ".replace(_AGENT_CHAT_EMBED, agent_chat_source)" in source
    raw = agent_module._embedded_agent_chat_source()
    assert '"onboard_solution"' in raw
    assert "{0,140}" in raw
    rendered = source.split("_AGENT_CHAT_EMBED = ", 1)[0]  # sanity: module loads
    assert rendered


def test_bootstrap_embeds_skill_context_and_api_accounting() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert "_resolve_skill_context" in source
    assert "_skill_index_candidates" in source
    assert "apis_suggested" in source
    assert "skills_used" in source
    assert "_dedupe(apis_used)" in source


def test_bootstrap_embeds_scoped_state_s3_persistence() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert "_state_s3_key" in source
    assert "NPA_AGENT_STATE_S3_PREFIX" in source
    assert "NPA_AGENT_SESSION_SCOPE" in source
    assert "_save_state_to_s3" in source
    assert "_load_state_from_s3" in source


def test_bootstrap_embeds_provider_resilience_fallback() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert "_chat_with_resilience" in source
    assert "_provider_chat" in source
    assert "NPA_AGENT_LLM_PROVIDER" in source
    assert "NPA_AGENT_LLM_PROVIDERS" in source
    assert "default_provider" in source


def test_bootstrap_chat_model_selector_defaults_to_auto_routing() -> None:

    source = _agent_ui_bundle()
    # An explicit Auto option lets the UI post an empty model so the backend
    # applies cost-tier routing instead of pinning the branded reasoner.
    assert "Auto (cost-aware)" in source
    # The old behaviors that defeated cost routing must be gone:
    # 1) selectedChatModel no longer hardcodes the default model as a fallback,
    assert (
        'return String((select && select.value) || "").trim() || "{DEFAULT_LLM_MODEL}"'
        not in source
    )
    # 2) the chat response no longer overwrites the selector (would hijack Auto).
    assert "if (select) select.value = String(data.model);" not in source


def test_bootstrap_embeds_cost_aware_routing() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    # Placeholder is declared, substituted, and consumed by the chat handler.
    assert "_AGENT_ROUTING_EMBED" in source
    assert ".replace(_AGENT_ROUTING_EMBED, agent_routing_source)" in source
    assert "build_model_ladder(" in source
    assert "classify_tier(" in source
    assert "chat_extra(tier)" in source
    assert "enforce_input_budget(" in source
    assert "usage_summary(data)" in source
    # The embedded routing source must actually be inlined (function defs present).
    raw = agent_module._embedded_agent_routing_source()
    assert "def build_model_ladder(" in raw
    assert "def classify_tier(" in raw
    assert "FAST_CAPABLE" in raw


def test_default_llm_models_are_cost_ordered() -> None:
    from npa.cli import agent as agent_module

    models = list(agent_module.DEFAULT_LLM_MODELS)
    # Cheap workhorse leads; branded reasoner is not first.
    assert models[0] == "Qwen/Qwen3-32B"
    assert models[0] != agent_module.DEFAULT_LLM_MODEL
    assert agent_module.DEFAULT_LLM_MODEL in models


def test_deploy_seeds_cost_ordered_ladder_without_explicit_models(
    monkeypatch, tmp_path
) -> None:
    """A bare `npa agent deploy` (no --llm-models) configures the full tier
    ladder on the VM, so routing works without the operator listing models."""
    from npa.cli.agent import deploy_cmd

    captured: dict[str, object] = {}
    creds = {"service_account_id": "sa", "s3_bucket": "b", "s3_endpoint": "e"}

    monkeypatch.setattr(
        "npa.cli.agent.resolve_environment",
        lambda *a, **k: SimpleNamespace(
            project_id=k.get("project_id"),
            tenant_id=k.get("tenant_id"),
            region=k.get("region"),
        ),
    )
    monkeypatch.setattr(
        "npa.clients.nebius.bootstrap_agent_environment", lambda *a, **k: creds
    )
    monkeypatch.setattr("npa.clients.nebius.get_iam_token", lambda: "iam")
    monkeypatch.setattr(
        "npa.cli.agent._resolve_deploy_storage_credentials", lambda **k: creds
    )
    monkeypatch.setattr(
        "npa.cli.agent._ensure_terraform_state_bucket", lambda **k: None
    )
    monkeypatch.setattr("npa.cli.agent._persist_agent_project_config", lambda **k: None)
    monkeypatch.setattr(
        "npa.cli.agent._apply_agent_terraform",
        lambda **k: {
            "vm_ip": "203.0.113.50",
            "instance_id": "i-1",
            "ssh_key_path": "/k",
        },
    )
    monkeypatch.setattr("npa.cli.agent._is_routable_public_ip", lambda _ip: True)
    monkeypatch.setattr(
        "npa.cli.agent._write_auth_secret", lambda **k: tmp_path / "auth.env"
    )
    monkeypatch.setattr(
        "npa.cli.agent._resolve_deploy_llm_credentials",
        lambda: ("tf-key", "nvidia/Cosmos3-Super-Reasoner"),
    )
    monkeypatch.setattr("npa.cli.agent._resolve_operator_credentials", lambda: ("", ""))
    monkeypatch.setattr("npa.cli.agent._bootstrap_agent_stack", lambda **k: None)
    monkeypatch.setattr("npa.cli.agent.ensure_ingress", lambda **k: None)
    monkeypatch.setattr(
        "npa.cli.agent._store_agent_record",
        lambda project, name, rec: captured.update(rec),
    )

    # Satisfy the fail-fast deploy prerequisites (terraform + SSH key pair).
    (tmp_path / "id_ed25519.pub").write_text("ssh-ed25519 AAAA test\n")
    (tmp_path / "id_ed25519").write_text("-----BEGIN OPENSSH PRIVATE KEY-----\n")
    monkeypatch.setenv("NPA_TERRAFORM_BIN", "/usr/bin/terraform")

    deploy_cmd(
        project="agent-live",
        name="agent",
        project_id="project-1",
        tenant_id="tenant-1",
        region="eu-north1",
        ssh_user="ubuntu",
        ssh_public_key_path=str(tmp_path / "id_ed25519.pub"),
        tf_var=[],
        agent_port=8088,
        backend_port=8787,
        rerun_port=9090,
        llm_model="nvidia/Cosmos3-Super-Reasoner",
        llm_models=[],
        no_public_https=True,
    )

    configured = list(captured.get("llm", {}).get("models", []))  # type: ignore[union-attr]
    # All four routing tiers are present without the operator listing them.
    for expected in (
        "Qwen/Qwen3-32B",
        "meta-llama/Llama-3.3-70B-Instruct",
        "nvidia/Cosmos3-Super-Reasoner",
        "Qwen/Qwen2.5-VL-72B-Instruct",
    ):
        assert expected in configured, f"{expected} missing from {configured}"


def test_agent_preflight_all_pass(monkeypatch, tmp_path) -> None:
    from npa.cli import agent as agent_module

    (tmp_path / "id_ed25519.pub").write_text("ssh-ed25519 AAAA test\n")
    (tmp_path / "id_ed25519").write_text("-----BEGIN OPENSSH PRIVATE KEY-----\n")
    monkeypatch.setenv("NPA_TERRAFORM_BIN", "/usr/bin/terraform")
    monkeypatch.setattr(
        agent_module, "_resolve_deploy_llm_credentials", lambda: ("tf-key", "m")
    )

    result = runner.invoke(
        app,
        [
            "preflight",
            "--skip-nebius",
            "--ssh-public-key-path",
            str(tmp_path / "id_ed25519.pub"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "[PASS] terraform" in result.output
    assert "[PASS] ssh_public_key" in result.output
    assert "[PASS] token_factory" in result.output


def test_agent_preflight_fails_on_missing_terraform_and_keys(
    monkeypatch, tmp_path
) -> None:
    from npa.cli import agent as agent_module

    monkeypatch.delenv("NPA_TERRAFORM_BIN", raising=False)
    monkeypatch.setattr(agent_module.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        agent_module, "_resolve_deploy_llm_credentials", lambda: ("", "m")
    )

    result = runner.invoke(
        app,
        [
            "preflight",
            "--skip-nebius",
            "--ssh-public-key-path",
            str(tmp_path / "missing.pub"),
        ],
    )
    assert result.exit_code == 1, result.output
    assert "[FAIL] terraform" in result.output
    assert "[FAIL] ssh_public_key" in result.output
    assert "[WARN] token_factory" in result.output


def test_agent_preflight_json_output(monkeypatch, tmp_path) -> None:
    from npa.cli import agent as agent_module

    (tmp_path / "id_ed25519.pub").write_text("ssh-ed25519 AAAA test\n")
    (tmp_path / "id_ed25519").write_text("priv\n")
    monkeypatch.setenv("NPA_TERRAFORM_BIN", "/usr/bin/terraform")
    monkeypatch.setattr(
        agent_module, "_resolve_deploy_llm_credentials", lambda: ("tf-key", "m")
    )

    result = runner.invoke(
        app,
        [
            "preflight",
            "--skip-nebius",
            "--json",
            "--ssh-public-key-path",
            str(tmp_path / "id_ed25519.pub"),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert {c["name"] for c in payload["checks"]} == {
        "terraform",
        "ssh_public_key",
        "ssh_private_key",
        "token_factory",
    }


def test_agent_preflight_nebius_fail(monkeypatch, tmp_path) -> None:
    from npa.cli import agent as agent_module

    (tmp_path / "id_ed25519.pub").write_text("ssh-ed25519 AAAA test\n")
    (tmp_path / "id_ed25519").write_text("priv\n")
    monkeypatch.setenv("NPA_TERRAFORM_BIN", "/usr/bin/terraform")
    monkeypatch.setattr(
        agent_module, "_resolve_deploy_llm_credentials", lambda: ("tf-key", "m")
    )

    def _boom() -> str:
        raise RuntimeError("no profile")

    monkeypatch.setattr("npa.clients.nebius.get_iam_token", _boom)

    result = runner.invoke(
        app,
        ["preflight", "--ssh-public-key-path", str(tmp_path / "id_ed25519.pub")],
    )
    assert result.exit_code == 1, result.output
    assert "[FAIL] nebius_profile" in result.output


def test_deploy_fails_fast_on_missing_ssh_key(monkeypatch, tmp_path) -> None:
    """Deploy aborts on a missing SSH key BEFORE any cloud IAM side effects."""
    from npa.cli.agent import deploy_cmd

    stored: list[dict] = []
    monkeypatch.setattr(
        "npa.cli.agent.build_deployment_manifest",
        lambda **_kwargs: {
            "deployment_id": "npa-agent-owner",
            "deployment_name": "agent",
            "project_alias": "fresh",
            "runtime_namespace": "fresh/agent",
            "repository": "org/repo",
            "branch": "codex/owner",
            "commit": "a" * 40,
            "source_tree": "b" * 40,
            "short_commit": "a" * 12,
            "workspace_label": "Workspace",
            "bootstrap_timestamp": "2026-08-10T00:00:00Z",
        },
    )
    monkeypatch.setattr("npa.cli.agent._agent_record", lambda *_args: {})
    monkeypatch.setattr(
        "npa.cli.agent._agent_terraform_state_exists", lambda *_args: False
    )
    monkeypatch.setenv("NPA_TERRAFORM_BIN", "/usr/bin/terraform")
    monkeypatch.setattr(
        "npa.cli.agent._store_agent_record",
        lambda _project, _name, record: stored.append(record),
    )
    monkeypatch.setattr(
        "npa.cli.agent.resolve_environment",
        lambda *a, **k: SimpleNamespace(
            project_id=k.get("project_id"),
            tenant_id=k.get("tenant_id"),
            region=k.get("region"),
        ),
    )
    monkeypatch.setattr(
        "npa.cli.agent._resolve_deploy_llm_credentials", lambda: ("tf-key", "m")
    )

    def _must_not_run(*a, **k):
        raise AssertionError("cloud bootstrap must not run when prerequisites fail")

    monkeypatch.setattr("npa.clients.nebius.bootstrap_agent_environment", _must_not_run)

    with pytest.raises(Exit) as exc:
        deploy_cmd(
            project="fresh",
            name="agent",
            project_id="project-1",
            tenant_id="tenant-1",
            region="us-central1",
            ssh_user="ubuntu",
            ssh_public_key_path=str(tmp_path / "missing.pub"),
            tf_var=[],
            agent_port=8088,
            backend_port=8787,
            rerun_port=9090,
            llm_model="model-a",
            llm_models=[],
            no_public_https=False,
        )
    assert exc.value.exit_code == 1
    assert stored == []

    # A failed prerequisite must not reserve the namespace: after fixing the
    # local key, a retry proceeds to the next deployment phase.
    (tmp_path / "missing.pub").write_text("ssh-ed25519 AAAA test\n")
    (tmp_path / "missing").write_text("private\n")
    reached_cloud_bootstrap = False

    def _retry_reaches_cloud(*_args, **_kwargs):
        nonlocal reached_cloud_bootstrap
        reached_cloud_bootstrap = True
        raise RuntimeError("retry sentinel")

    monkeypatch.setattr(
        "npa.clients.nebius.bootstrap_agent_environment", _retry_reaches_cloud
    )
    with pytest.raises(RuntimeError, match="retry sentinel"):
        deploy_cmd(
            project="fresh",
            name="agent",
            project_id="project-1",
            tenant_id="tenant-1",
            region="us-central1",
            ssh_user="ubuntu",
            ssh_public_key_path=str(tmp_path / "missing.pub"),
            tf_var=[],
            agent_port=8088,
            backend_port=8787,
            rerun_port=9090,
            llm_model="model-a",
            llm_models=[],
            no_public_https=False,
        )
    assert reached_cloud_bootstrap is True
    assert stored == []


def test_deploy_fails_fast_on_missing_terraform(monkeypatch, tmp_path) -> None:
    """Deploy aborts on a missing terraform binary BEFORE any cloud side effects."""
    from npa.cli import agent as agent_module
    from npa.cli.agent import deploy_cmd

    (tmp_path / "id_ed25519.pub").write_text("ssh-ed25519 AAAA test\n")
    (tmp_path / "id_ed25519").write_text("priv\n")
    monkeypatch.delenv("NPA_TERRAFORM_BIN", raising=False)
    monkeypatch.setattr(agent_module.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        "npa.cli.agent.resolve_environment",
        lambda *a, **k: SimpleNamespace(
            project_id=k.get("project_id"),
            tenant_id=k.get("tenant_id"),
            region=k.get("region"),
        ),
    )
    monkeypatch.setattr(
        "npa.cli.agent._resolve_deploy_llm_credentials", lambda: ("tf-key", "m")
    )

    def _must_not_run(*a, **k):
        raise AssertionError("cloud bootstrap must not run when terraform is missing")

    monkeypatch.setattr("npa.clients.nebius.bootstrap_agent_environment", _must_not_run)

    with pytest.raises(Exit) as exc:
        deploy_cmd(
            project="fresh",
            name="agent",
            project_id="project-1",
            tenant_id="tenant-1",
            region="us-central1",
            ssh_user="ubuntu",
            ssh_public_key_path=str(tmp_path / "id_ed25519.pub"),
            tf_var=[],
            agent_port=8088,
            backend_port=8787,
            rerun_port=9090,
            llm_model="model-a",
            llm_models=[],
            no_public_https=False,
        )
    assert exc.value.exit_code == 1


def test_deploy_warns_on_missing_token_factory_key(
    monkeypatch, tmp_path, capsys
) -> None:
    """Deploy surfaces the Token Factory 503 warning up front (before Terraform)."""
    from npa.cli.agent import deploy_cmd
    from npa.clients.nebius import NebiusError

    (tmp_path / "id_ed25519.pub").write_text("ssh-ed25519 AAAA test\n")
    (tmp_path / "id_ed25519").write_text("priv\n")
    monkeypatch.setenv("NPA_TERRAFORM_BIN", "/usr/bin/terraform")
    monkeypatch.setattr(
        "npa.cli.agent.resolve_environment",
        lambda *a, **k: SimpleNamespace(
            project_id=k.get("project_id"),
            tenant_id=k.get("tenant_id"),
            region=k.get("region"),
        ),
    )
    # No Token Factory key configured.
    monkeypatch.setattr(
        "npa.cli.agent._resolve_deploy_llm_credentials", lambda: ("", "m")
    )
    # Stop the flow right after the warning, before any real provisioning.
    monkeypatch.setattr(
        "npa.clients.nebius.bootstrap_agent_environment",
        lambda *a, **k: (_ for _ in ()).throw(NebiusError("stop after warning")),
    )

    with pytest.raises(Exit):
        deploy_cmd(
            project="fresh",
            name="agent",
            project_id="project-1",
            tenant_id="tenant-1",
            region="us-central1",
            ssh_user="ubuntu",
            ssh_public_key_path=str(tmp_path / "id_ed25519.pub"),
            tf_var=[],
            agent_port=8088,
            backend_port=8787,
            rerun_port=9090,
            llm_model="model-a",
            llm_models=[],
            no_public_https=False,
        )
    err = capsys.readouterr().err
    assert "503" in err


def test_agent_nebius_auth_result_pass(monkeypatch) -> None:
    from npa.cli import agent as agent_module

    monkeypatch.setattr("npa.clients.nebius.get_iam_token", lambda: "iam-token")
    result = agent_module._agent_nebius_auth_result()
    assert result.status == "PASS"
    assert result.name == "nebius_profile"


def test_resolve_agent_service_account_id_from_nebius(mocker) -> None:
    from npa.cli.agent import _resolve_agent_service_account_id

    mocker.patch(
        "npa.clients.nebius.resolve_service_account_id",
        return_value="serviceaccount-u00s24wzj2wk8z9tqq",
    )
    record = {"project_id": "project-u00zhx4tpr00xh99b28n52"}
    assert (
        _resolve_agent_service_account_id("rtxpro", record)
        == "serviceaccount-u00s24wzj2wk8z9tqq"
    )


def test_run_details_surface_per_stage_workflow_logs() -> None:
    """Run details must surface real per-stage execution (command/returncode/status)
    from the npa.workflow run manifest so operators can view logs of each stage."""
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).with_name("agent_stage_runtime.py").read_text(
        encoding="utf-8"
    )
    assert "def _workflow_run_steps(" in source
    assert "/npa-workflow/manifest.json" in source
    assert '"workflow_steps": workflow_steps' in source
    # Enriched logs include the per-stage command lines.
    assert "workflow_steps = _workflow_run_steps(" in source


def test_artifact_file_transcodes_non_web_images_to_png() -> None:
    """Non-web images (.ppm sim camera frames, .bmp, .tiff) must be transcoded to
    PNG by the artifact file endpoint so they are viewable in the Rerun/Image panes."""
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert "needs_image_transcode(safe_name)" in source
    assert 'format="PNG"' in source
    assert 'media_type="image/png"' in source


def test_stages_tab_run_search_uses_server_search() -> None:
    """The Stages-tab run search must also query the server (not just filter the
    fetched page) so runs older than the newest page are findable there too."""
    source = _agent_ui_bundle()
    assert "stagesSearchTimer" in source
    # Both run-search boxes wire the debounced server search.
    assert source.count("await refreshArtifactRuns(value)") >= 2


def test_ui_script_calls_no_undefined_local_helper() -> None:
    """Catch a helper that was deleted (or renamed) but is still called.

    `node --check` only proves the script *parses*; calling a removed function is
    a runtime ReferenceError that silently breaks a whole handler. Two real cases
    motivated this: a merge dropped `applyViewerChromeForMode` while `refresh()`
    still called it (aborting the refresh loop mid-way), and `resolveRerunRrdUrl`
    had been gone for a while behind a try/catch.

    Scope is deliberately narrow — bare calls to camelCase names, which is what
    this UI's own helpers look like — so prose and member calls do not trip it.
    """
    import re

    script = rendered_agent_ui_html().split("<script>")[-1].split("</script>")[0]
    defined = set(re.findall(r"function\s+([A-Za-z_$][\w$]*)\s*\(", script))
    defined |= set(re.findall(r"(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=", script))
    called = set(
        re.findall(r"(?<![.\w$])([a-z][a-z0-9]*(?:[A-Z][A-Za-z0-9]*)+)\s*\(", script)
    )

    # Browser globals plus names provided by the dynamically imported glue module.
    allowed = {
        "clearInterval",
        "clearTimeout",
        "createImageBitmap",
        "decodeURIComponent",
        "drawImage",
        "encodeURIComponent",
        "isFinite",
        "isNaN",
        "localStorage",
        "mountFoxgloveViewer",
        "mountSelfHostedViewer",
        "parseFloat",
        "parseInt",
        "requestAnimationFrame",
        "setInterval",
        "setTimeout",
        "structuredClone",
    }
    undefined = sorted(called - defined - allowed)
    assert not undefined, f"UI script calls undefined helper(s): {undefined}"


def test_ui_recomputes_the_viewer_cta_once_the_iframe_mounts() -> None:
    """The "no recording yet" banner must be re-evaluated after the mount.

    ``cta.hidden = ready && rerunIframeLoaded``, and ``rerunIframeLoaded`` flips
    to true asynchronously. A status refresh landing before that leaves the
    banner painted above a viewer that has already loaded the recording, because
    nothing recomputes it. Reproduced on a real NuRec run: absent on first load,
    present after a reload.
    """
    from pathlib import Path

    html = (
        Path(__file__).resolve().parents[2] / "src" / "npa" / "cli" / "agent_ui.html"
    ).read_text(encoding="utf-8")

    mount = html.split("rerunIframeLoaded = true;", 1)[1][:600]
    assert "updateSimvizCta(" in mount, (
        "the CTA must be recomputed immediately after the iframe mounts"
    )


def test_ui_treats_a_mounted_viewer_as_proof_a_recording_exists() -> None:
    """Readiness must not depend on the status fetch alone.

    Recomputing the CTA after the mount is not enough: the recompute reads
    ``lastSimVizStatus``, which is still empty while the first sim-viz fetch is
    in flight, so the banner went on claiming "no recording yet" above a viewer
    that had already decoded and rendered one. Measured at 0.5-4s of overlap on
    every page load. A mounted iframe holding a resolved recording URL is the
    more direct evidence, so it has to feed ``ready`` too.
    """
    from pathlib import Path

    html = (
        Path(__file__).resolve().parents[2] / "src" / "npa" / "cli" / "agent_ui.html"
    ).read_text(encoding="utf-8")

    assert (
        "const mountProvesRecording = Boolean(rerunIframeLoaded && lastRerunRecordingUrl);"
        in html
    )
    assert (
        "const ready = Boolean(status.rerun_ready || status.rrd_uri || mountProvesRecording);"
        in html
    ), "a mounted viewer must count towards readiness"


def test_ui_viewer_banner_copy_tracks_readiness_both_ways() -> None:
    """The Rerun banner must assign copy for BOTH states, like its siblings.

    Only the not-ready branch used to set text, so whenever a recording WAS
    ready but the iframe had not mounted yet, the banner kept its "No
    run-specific Rerun recording yet" default and contradicted the run's own
    published recording. Measured against the deployed agent: 593 of 593
    samples over six page loads showed that false claim while
    ``/api/sim-viz/status`` reported ``rerun_ready: true``.
    """
    from pathlib import Path

    html = (
        Path(__file__).resolve().parents[2] / "src" / "npa" / "cli" / "agent_ui.html"
    ).read_text(encoding="utf-8")

    branch = html.split("const mountProvesRecording", 1)[1].split(
        "function setRenderMode", 1
    )[0]
    assert "cta.textContent = ready" in branch, "the ready state needs its own copy"
    assert "No run-specific Rerun recording yet." in branch
    # The ready copy must not itself deny the recording.
    ready_copy = branch.split("cta.textContent = ready", 1)[1].split(":", 1)[0]
    assert "No run-specific" not in ready_copy


def test_ui_does_not_claim_no_recording_before_the_status_arrives() -> None:
    """ "Unknown" must not be reported as "absent".

    ``updateSimvizCta`` runs before the first ``/api/sim-viz/status`` response,
    when ``lastSimVizStatus`` is still null. Treating that as "not ready" made
    the banner assert there was no recording during the first 1-3s of every page
    load, for runs that had published one. Measured on the deployed agent: 43
    false-claim samples across six loads remained after the readiness fix, all
    inside that pre-status window.
    """
    from pathlib import Path

    html = (
        Path(__file__).resolve().parents[2] / "src" / "npa" / "cli" / "agent_ui.html"
    ).read_text(encoding="utf-8")

    assert "const haveStatus = Boolean(simViz || lastSimVizStatus);" in html
    branch = html.split("const mountProvesRecording", 1)[1].split(
        "function setRenderMode", 1
    )[0]
    # The definitive "no recording" claim is gated behind having a status.
    assert "haveStatus" in branch
    claim_idx = branch.index("No run-specific Rerun recording yet.")
    gate_idx = branch.index("haveStatus", branch.index("cta.textContent = ready"))
    assert gate_idx < claim_idx, "the absence claim must be gated on haveStatus"
