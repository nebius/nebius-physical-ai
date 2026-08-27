from __future__ import annotations

from npa.cli import agent as agent_module
from npa.cli.agent import rendered_agent_ui_html

import base64
import json
import subprocess
import re
import shutil
from typing import NoReturn, get_type_hints
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from typer import Exit
from typer.testing import CliRunner
import yaml

from npa.cli.agent import (
    AGENT_MEDIA_PREVIEW_CONTRACT,
    AGENT_RERUN_NO_BUNDLE_SPLASH_CONTRACT,
    AGENT_UI_VERSION,
    _normalize_llm_models,
    app,
    build_agent_urls,
)

runner = CliRunner()

# A few tests evaluate the emitted JavaScript with a real engine, which is the only
# way to check it rather than pattern-match it. `node` is not a suite prerequisite,
# so skip rather than fail -- and skip rather than return early, so a machine
# without it reports the coverage it did not run.
requires_node = pytest.mark.skipif(
    shutil.which("node") is None,
    reason="needs node to evaluate the emitted JavaScript",
)


def test_fail_is_typed_as_non_returning_and_preserves_cli_exit() -> None:
    assert get_type_hints(agent_module._fail)["return"] is NoReturn
    with pytest.raises(Exit) as caught:
        agent_module._fail("focused failure")
    assert caught.value.exit_code == 1


def test_artifact_only_timeout_allows_preserved_run_inventory() -> None:
    from npa.cli.agent import ARTIFACT_ONLY_HTTP_TIMEOUT_SECONDS

    # S3-backed workflow status can legitimately take more than the generic
    # 30-second HTTP default while it inventories a large preserved run.
    assert ARTIFACT_ONLY_HTTP_TIMEOUT_SECONDS >= 60.0


def test_canonical_artifact_only_probe_is_read_only_and_complete() -> None:
    import httpx

    from npa.cli.agent_resources import artifact_only_http_probe

    digest = "a" * 64
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.raw_path.decode())
        payloads = {
            "/api/health": {"state_sha256": digest},
            "/api/session": {"selection": {}},
            "/api/artifacts/runs?prefix=&limit=100": {"runs": [{"run_id": "one"}]},
            "/api/tools": {"tool_refs": ["workbench.foxglove.convert_run"]},
            "/api/workflows/sim2real/status": {"stage": "idle"},
            "/api/infra/k8s": {"has_infra": False},
        }
        return httpx.Response(200, json=payloads[paths[-1]])

    with httpx.Client(
        base_url="https://agent.example",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = artifact_only_http_probe(client)

    assert result["state_sha256"] == digest
    assert result["run_count"] == result["tool_ref_count"] == 1
    assert paths == [
        "/api/health",
        "/api/session",
        "/api/artifacts/runs?prefix=&limit=100",
        "/api/tools",
        "/api/workflows/sim2real/status",
        "/api/infra/k8s",
        "/api/health",
    ]


def test_canonical_artifact_only_probe_rejects_state_mutation() -> None:
    import httpx

    from npa.cli.agent_deployment import DeploymentIdentityError
    from npa.cli.agent_resources import artifact_only_http_probe

    health_digests = iter(("a" * 64, "b" * 64))

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/health":
            return httpx.Response(200, json={"state_sha256": next(health_digests)})
        if path == "/api/artifacts/runs":
            return httpx.Response(200, json={"runs": []})
        if path == "/api/tools":
            return httpx.Response(200, json={"tool_refs": []})
        return httpx.Response(200, json={})

    with httpx.Client(
        base_url="https://agent.example",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(DeploymentIdentityError, match="mutated durable session"):
            artifact_only_http_probe(client)


def test_divergent_agent_helper_modules_are_not_packaged() -> None:
    import importlib.util

    cli_root = Path(agent_module.__file__).parent
    duplicates = ("agent_credentials", "agent_live_verify", "agent_prereqs")
    assert all(not (cli_root / f"{name}.py").exists() for name in duplicates)
    assert all(
        importlib.util.find_spec(f"npa.cli.{name}") is None for name in duplicates
    )


def test_staged_agent_source_is_readable_by_unprivileged_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa.cli import agent as agent_module

    archive = tmp_path / "source.tar.gz"
    archive.write_bytes(b"archive")
    monkeypatch.setattr(
        agent_module, "_create_agent_source_archive", lambda: str(archive)
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
    agent_module._stage_agent_npa_source(ssh)  # type: ignore[arg-type]

    assert "sudo chown -R root:root /opt/npa-agent/npa-src" in ssh.command
    assert "sudo chmod -R a+rX /opt/npa-agent/npa-src" in ssh.command


@pytest.fixture(autouse=True)
def _successful_storage_probe(monkeypatch):
    """Keep unrelated agent tests hermetic; storage failures override this."""

    from npa.clients import storage_validation
    from npa.clients.storage_validation import StorageProbeResult

    monkeypatch.setattr(
        storage_validation,
        "probe_storage_write",
        lambda **_kwargs: StorageProbeResult(
            True,
            "ok",
            "Writable S3 verified with a cleaned write/delete probe.",
            cleanup_attempted=True,
            cleanup_succeeded=True,
        ),
    )
    monkeypatch.setattr(
        storage_validation,
        "probe_terraform_backend",
        lambda **_kwargs: StorageProbeResult(
            True,
            "new_state_prefix_valid",
            "Exact Terraform backend prefix verified.",
            cleanup_attempted=True,
            cleanup_succeeded=True,
        ),
    )
    monkeypatch.setattr(
        "npa.cli.agent.resolve_project_storage",
        lambda *_args, **_kwargs: SimpleNamespace(
            checkpoint_bucket="s3://configured-bucket/artifacts",
            endpoint_url="https://storage.eu-north1.nebius.cloud",
            aws_access_key_id="configured-access-key",
            aws_secret_access_key="configured-secret-key",
        ),
    )
    # Individual capacity tests override this. Unrelated deploy tests stub every
    # cloud dependency and must not consult the developer machine's provider.
    monkeypatch.setattr(
        "npa.cli.agent._agent_check_whole_path_capacity", lambda *args, **kwargs: None
    )
    # Reconciliation is a distinct remote boundary from bootstrap. Tests in
    # this module that stub bootstrap as successful also get exact healthy
    # evidence; failure/adoption matrices live in test_agent_setup_convergence.
    monkeypatch.setattr(
        "npa.cli.agent._reconcile_agent_setup",
        lambda **_kwargs: {
            "state": "healthy",
            "service_fingerprint": "test-service-fingerprint",
            "credential_fingerprint": "test-credential-fingerprint",
            "models_healthy": True,
        },
    )


def _agent_source() -> str:
    """agent.py plus source modules embedded or split out of it.

    Source-scanning assertions include login, nginx, and effective-access policy
    modules that the bootstrap embeds into the generated backend.
    """
    from npa.cli import agent as agent_module
    from npa.cli import agent_access_runtime as agent_access_runtime_module
    from npa.cli import agent_assets
    from npa.cli import agent_env_files
    from npa.cli import agent_login as agent_login_module
    from npa.cli import agent_site as agent_site_module
    from npa.cli import agent_viewer_runtime as agent_viewer_runtime_module

    sources = [
        Path(module.__file__).read_text(encoding="utf-8")
        for module in (
            agent_module,
            agent_access_runtime_module,
            agent_assets,
            agent_env_files,
            agent_login_module,
            agent_site_module,
            agent_viewer_runtime_module,
        )
    ]
    sources.append(
        Path(agent_module.__file__)
        .with_name("agent_artifact_content.py")
        .read_text(encoding="utf-8")
    )
    return "\n".join(sources)


def _agent_ui_bundle() -> str:
    """Agent deploy source plus rendered UI HTML (UI lives in agent_ui.html)."""
    return _agent_source() + "\n" + rendered_agent_ui_html()


def _agent_nginx_site() -> str:
    """Render the nginx policy the bootstrap actually writes."""

    from npa.cli.agent import _nginx_agent_site_body

    return _nginx_agent_site_body(backend_port=8787, rerun_port=9090)


def test_agent_operator_profile_stages_exact_nonsecret_kubernetes_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa.cli import agent_env_files

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
projects:
  demo:
    kubernetes:
      cluster_name: exact-cluster
      context: exact-context
      kubeconfig: /operator/private/kubeconfig
      gpu_profile: rtxpro
      gpu_accelerator: RTXPRO-6000-BLACKWELL-SERVER-EDITION
      untrusted_extension: must-not-copy
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(agent_env_files, "CONFIG_PATH", config_path)

    staged: list[str] = []

    class FakeSSH:
        def upload_private_text(self, content: str, _remote_path: str) -> None:
            staged.append(content)

        def run_or_raise(
            self, *_args: object, **_kwargs: object
        ) -> tuple[int, str, str]:
            return 0, "", ""

        def run(self, *_args: object, **_kwargs: object) -> tuple[int, str, str]:
            return 0, "", ""

    agent_env_files._write_agent_operator_profile(
        FakeSSH(),  # type: ignore[arg-type]
        ssh_user="ubuntu",
        project_alias="demo",
        project_id="project-1",
        tenant_id="tenant-1",
        region="us-central1",
        tf_api_key="",
        s3_bucket="",
        s3_endpoint="",
        s3_access_key="",
        s3_secret_key="",
    )

    configs = [json.loads(value) for value in staged if '"default_project"' in value]
    assert len(configs) == 2
    target = configs[0]["projects"]["demo"]["kubernetes"]
    assert target == {
        "cluster_name": "exact-cluster",
        "context": "exact-context",
        "gpu_profile": "rtxpro",
        "gpu_accelerator": "RTXPRO-6000-BLACKWELL-SERVER-EDITION",
    }


def test_agent_operator_profile_privately_stages_remote_kubeconfig(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa.cli import agent_env_files

    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\nclusters: []\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "projects:\n"
        "  demo:\n"
        "    kubernetes:\n"
        "      cluster_name: exact-cluster\n"
        "      context: exact-context\n"
        f"      kubeconfig: {kubeconfig}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(agent_env_files, "CONFIG_PATH", config_path)

    staged: list[tuple[str, str]] = []

    class FakeSSH:
        def upload_private_text(self, content: str, remote_path: str) -> None:
            staged.append((content, remote_path))

        def run_or_raise(
            self, *_args: object, **_kwargs: object
        ) -> tuple[int, str, str]:
            return 0, "", ""

        def run(self, *_args: object, **_kwargs: object) -> tuple[int, str, str]:
            return 0, "", ""

    agent_env_files._write_agent_operator_profile(
        FakeSSH(),  # type: ignore[arg-type]
        ssh_user="ubuntu",
        project_alias="demo",
        project_id="project-1",
        tenant_id="tenant-1",
        region="us-central1",
        tf_api_key="",
        s3_bucket="",
        s3_endpoint="",
        s3_access_key="",
        s3_secret_key="",
    )

    assert [value for value, _path in staged].count(
        "apiVersion: v1\nclusters: []\n"
    ) == 2
    configs = [
        json.loads(value) for value, _path in staged if '"default_project"' in value
    ]
    assert configs[0]["projects"]["demo"]["kubernetes"]["kubeconfig"] == (
        "/home/ubuntu/.npa/clusters/exact-cluster/kubeconfig"
    )
    assert configs[1]["projects"]["demo"]["kubernetes"]["kubeconfig"] == (
        "/root/.npa/clusters/exact-cluster/kubeconfig"
    )


def test_agent_operator_profile_stages_exact_project_credentials() -> None:
    from npa.cli import agent_env_files

    staged: list[str] = []

    class FakeSSH:
        def upload_private_text(self, content: str, _remote_path: str) -> None:
            staged.append(content)

        def run_or_raise(
            self, *_args: object, **_kwargs: object
        ) -> tuple[int, str, str]:
            return 0, "", ""

        def run(self, *_args: object, **_kwargs: object) -> tuple[int, str, str]:
            return 0, "", ""

    agent_env_files._write_agent_operator_profile(
        FakeSSH(),  # type: ignore[arg-type]
        ssh_user="ubuntu",
        project_alias="demo",
        project_id="project-1",
        tenant_id="tenant-1",
        region="us-central1",
        tf_api_key="token-factory-secret",
        s3_bucket="bucket-1",
        s3_endpoint="https://storage.us-central1.nebius.cloud",
        s3_access_key="access-secret",
        s3_secret_key="storage-secret",
        service_account_id="service-account-1",
    )

    credentials = [
        json.loads(value) for value in staged if '"project_credentials"' in value
    ]
    assert len(credentials) == 2
    root = credentials[0]["project_credentials"]
    assert root["schema_version"] == "npa.project-credentials.v2"
    assert root["current_project_id"] == "project-1"
    exact = root["projects"]["project-1"]
    assert exact["aliases"] == ["demo"]
    assert exact["storage"]["bucket"] == "s3://bucket-1"
    assert exact["nebius"] == {
        "service_account_id": "service-account-1",
        "service_account_project_id": "project-1",
    }


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


def test_agent_access_logs_are_query_free() -> None:
    from npa.cli.agent_site import nginx_agent_site_body

    source = _agent_source()
    site = nginx_agent_site_body(
        backend_port=8787,
        rerun_port=9090,
        ui_version="test",
    )
    log_block = source.split("log_format npa_agent_safe", 1)[1].split("NGINXLOG", 1)[0]

    assert "--no-access-log" in source
    assert "--log-level warning" in source
    assert "npa-agent-access.log npa_agent_safe" in site
    assert "$request_method $uri $server_protocol" in log_block
    assert "$request_uri" not in log_block
    assert '"$request "' not in log_block


def test_customer_url_is_canonical_for_persisted_public_https_ip() -> None:
    from npa.cli.agent import _record_customer_url

    record = {
        "public_ip": "8.8.8.8",
        "public_https": True,
        "public_url": "https://203.0.113.50/",
        "agent_url": "https://203.0.113.50/",
    }

    assert _record_customer_url(record) == "https://8.8.8.8/"


def test_existing_agent_public_ip_resolves_from_provider_state(monkeypatch) -> None:
    from npa.cli.agent import _resolve_record_public_ip

    monkeypatch.setattr(
        "npa.cli.agent.resolve_instance_network_context",
        lambda instance_id: SimpleNamespace(
            instance_id=instance_id,
            public_ip="8.8.8.8/32",
            project_id="project-1",
            security_group_ids=("sg-1",),
        ),
    )

    assert (
        _resolve_record_public_ip(
            {"instance_id": "instance-1", "public_ip": "203.0.113.50"}
        )
        == "8.8.8.8"
    )


def test_existing_agent_public_ip_rejects_non_public_provider_state(
    monkeypatch,
) -> None:
    from npa.cli.agent import _resolve_record_public_ip
    from npa.clients.network import NetworkIngressError

    monkeypatch.setattr(
        "npa.cli.agent.resolve_instance_network_context",
        lambda _instance_id: SimpleNamespace(public_ip="127.0.0.1"),
    )

    with pytest.raises(NetworkIngressError, match="routable public IP"):
        _resolve_record_public_ip({"instance_id": "instance-1"})


def test_ensure_terraform_state_bucket_preserves_missing_configuration(
    monkeypatch,
) -> None:
    from npa.cli.agent import _ensure_terraform_state_bucket
    from npa.clients import nebius

    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        "npa.clients.nebius.bucket_exists", lambda _project, _bucket: False
    )
    monkeypatch.setattr(
        "npa.clients.nebius.ensure_bucket",
        lambda project, bucket: calls.append((project, bucket)),
    )

    with pytest.raises(nebius.NebiusError, match="missing.*preserved configuration"):
        _ensure_terraform_state_bucket(project_id="project-1", bucket_name="bucket-1")

    assert calls == []


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


def test_apply_agent_terraform_retries_without_sa_and_warns(
    monkeypatch, tmp_path, capsys
) -> None:
    """On compute PermissionDenied with an attached SA, retry without it + warn loudly."""
    from npa.cli.agent import _apply_agent_terraform
    from npa.deploy.provisioner import ProvisionerError

    monkeypatch.setattr(
        "npa.cli.agent.provisioner.prepare_working_dir", lambda *_a, **_k: tmp_path
    )
    monkeypatch.setattr("npa.cli.agent.provisioner.init", lambda **_k: None)

    calls: list[dict] = []

    def _apply(*, tf_dir, tf_vars):
        calls.append(dict(tf_vars))
        if len(calls) == 1:
            raise ProvisionerError(
                "Error: service compute: PermissionDenied creating instance"
            )
        return {"vm_ip": "203.0.113.50"}

    monkeypatch.setattr("npa.cli.agent.provisioner.apply", _apply)

    result = _apply_agent_terraform(
        project="fresh",
        name="agent",
        env_region="eu-north1",
        merged_vars={
            "s3_bucket": "agent-state",
            "s3_endpoint": "https://storage.eu-north1.nebius.cloud",
            "nebius_api_key": "ak",
            "nebius_secret_key": "sk",
            "service_account_id": "serviceaccount-abc",
        },
    )

    assert result == {"vm_ip": "203.0.113.50"}
    assert len(calls) == 2
    # First attempt attached the SA; the retry dropped it.
    assert calls[0]["service_account_id"] == "serviceaccount-abc"
    assert calls[1]["service_account_id"] == ""
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "self-mint" in err


def test_apply_failure_preserves_errored_state_and_exact_recovery(
    monkeypatch, tmp_path
) -> None:
    from npa.cli.agent import _apply_agent_terraform
    from npa.deploy.provisioner import BackendBucketMissingError
    from npa.provisioning_journal import ProvisioningOperation, operation_context

    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(tmp_path / "operations"))
    tf_dir = tmp_path / "terraform"
    tf_dir.mkdir()
    (tf_dir / "errored.tfstate").write_text('{"version":4,"resources":[]}')
    monkeypatch.setattr(
        "npa.cli.agent.provisioner.prepare_working_dir",
        lambda *_args, **_kwargs: tf_dir,
    )
    monkeypatch.setattr("npa.cli.agent.provisioner.init", lambda **_kwargs: None)
    monkeypatch.setattr(
        "npa.cli.agent_terraform._ensure_terraform_state_bucket",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "npa.cli.agent.provisioner.apply",
        lambda **_kwargs: (_ for _ in ()).throw(
            BackendBucketMissingError("NoSuchBucket during state upload")
        ),
    )
    operation = ProvisioningOperation.prepare(
        command="npa agent deploy",
        project_alias="demo",
        project_id="project-a",
        tenant_id="tenant-a",
        region="eu-north1",
        resource_type="agent",
        requested_name="agent",
        ownership_source="test",
        resume_command="npa agent deploy --project demo --name agent",
        destroy_command="npa agent destroy --project demo --name agent --yes",
    )

    with operation_context(operation), pytest.raises(BackendBucketMissingError):
        operation.transition("mutating")
        _apply_agent_terraform(
            project="demo",
            name="agent",
            env_region="eu-north1",
            merged_vars={
                "s3_bucket": "state-bucket",
                "s3_endpoint": "https://storage.example",
                "nebius_api_key": "access",
                "nebius_secret_key": "secret",
                "nebius_project_id": "project-a",
                "instance_name": "agent-demo-agent",
            },
        )

    summary = operation.recovery_summary()
    assert summary["phase"] == "recovery-required"
    assert len(summary["local_state"]) == 1
    assert Path(summary["local_state"][0]).is_file()
    assert summary["resume_command"] == "npa agent deploy --project demo --name agent"


def test_retry_restores_preserved_state_before_apply(monkeypatch, tmp_path) -> None:
    from npa.cli.agent import _apply_agent_terraform
    from npa.provisioning_journal import ProvisioningOperation, operation_context

    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(tmp_path / "operations"))
    tf_dir = tmp_path / "terraform"
    tf_dir.mkdir()
    monkeypatch.setattr(
        "npa.cli.agent.provisioner.prepare_working_dir",
        lambda *_args, **_kwargs: tf_dir,
    )
    monkeypatch.setattr("npa.cli.agent.provisioner.init", lambda **_kwargs: None)
    monkeypatch.setattr("npa.cli.agent.provisioner.state_list", lambda _path: [])
    calls: list[str] = []
    monkeypatch.setattr(
        "npa.cli.agent_terraform._ensure_terraform_state_bucket",
        lambda **_kwargs: calls.append("backend"),
    )
    monkeypatch.setattr(
        "npa.cli.agent.provisioner.state_push",
        lambda _state, _path: calls.append("state-push"),
    )
    monkeypatch.setattr(
        "npa.cli.agent.provisioner.apply",
        lambda **_kwargs: calls.append("apply") or {"instance_id": "instance-a"},
    )
    monkeypatch.setattr(
        "npa.cli.agent.provisioner.state_pull",
        lambda _path: b'{"version":4,"resources":[]}',
    )
    operation = ProvisioningOperation.prepare(
        command="npa agent deploy",
        project_alias="demo",
        project_id="project-a",
        tenant_id="tenant-a",
        region="eu-north1",
        resource_type="agent",
        requested_name="agent",
        ownership_source="test",
        resume_command="npa agent deploy --project demo --name agent",
    )
    operation.preserve_state_bytes(b'{"version":4,"resources":[]}', name="errored")
    operation.transition("recovery-required")

    with operation_context(operation):
        _apply_agent_terraform(
            project="demo",
            name="agent",
            env_region="eu-north1",
            merged_vars={
                "s3_bucket": "state-bucket",
                "s3_endpoint": "https://storage.example",
                "nebius_api_key": "access",
                "nebius_secret_key": "secret",
                "nebius_project_id": "project-a",
                "instance_name": "agent-demo-agent",
            },
        )

    assert calls == [
        "backend",
        "backend",
        "state-push",
        "backend",
        "apply",
        "backend",
    ]
    assert operation.read()["phase"] == "state-durable"


def test_backend_loss_at_apply_boundary_blocks_before_remote_mutation(
    monkeypatch, tmp_path
) -> None:
    from npa.cli.agent import _apply_agent_terraform
    from npa.deploy.provisioner import BackendBucketMissingError
    from npa.provisioning_journal import ProvisioningOperation, operation_context

    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(tmp_path / "operations"))
    tf_dir = tmp_path / "terraform"
    tf_dir.mkdir()
    monkeypatch.setattr(
        "npa.cli.agent.provisioner.prepare_working_dir",
        lambda *_args, **_kwargs: tf_dir,
    )
    monkeypatch.setattr("npa.cli.agent.provisioner.init", lambda **_kwargs: None)
    validations = 0

    def validate(**_kwargs) -> None:
        nonlocal validations
        validations += 1
        if validations == 2:
            raise BackendBucketMissingError("backend disappeared before apply")

    monkeypatch.setattr(
        "npa.cli.agent_terraform._ensure_terraform_state_bucket", validate
    )
    monkeypatch.setattr(
        "npa.cli.agent.provisioner.apply",
        lambda **_kwargs: pytest.fail("Terraform apply must not start"),
    )
    operation = ProvisioningOperation.prepare(
        command="npa agent deploy",
        project_alias="demo",
        project_id="project-a",
        resource_type="agent",
        requested_name="agent",
        resume_command="npa agent deploy --project demo --name agent",
    )
    with operation_context(operation), pytest.raises(BackendBucketMissingError):
        _apply_agent_terraform(
            project="demo",
            name="agent",
            env_region="eu-north1",
            merged_vars={
                "s3_bucket": "state-bucket",
                "s3_endpoint": "https://storage.example",
                "nebius_api_key": "access",
                "nebius_secret_key": "secret",
                "nebius_project_id": "project-a",
                "instance_name": "agent-demo-agent",
            },
        )
    assert validations == 2
    assert operation.read()["phase"] == "recovery-required"


def test_write_agent_nebius_env_omits_operator_iam_token(monkeypatch) -> None:
    """The staged VM env must carry S3 keys but NO copied operator IAM token."""
    from npa.cli.agent import _write_agent_nebius_env

    commands: list[str] = []
    staged: list[str] = []

    class _FakeSSH:
        def upload_private_text(self, content: str, _remote_path: str):
            staged.append(content)

        def run_or_raise(self, command: str, **_kwargs):
            commands.append(command)
            return ""

        def run(self, _command: str):
            return ""

    _write_agent_nebius_env(
        _FakeSSH(),
        project_alias="prod",
        agent_name="agent",
        project_id="project-abc",
        tenant_id="tenant-abc",
        region="eu-north1",
        service_account_id="serviceaccount-abc",
        bucket="agent-state",
        endpoint="https://storage.eu-north1.nebius.cloud",
        access_key="ak-id",
        secret_key="sk-val",
    )

    # Exactly one atomic install command; credential bytes travel via private SFTP.
    assert len(commands) == 1
    assert "ak-id" not in commands[0] and "sk-val" not in commands[0]
    staged_env = staged[0]

    # S3 access key stays (HMAC, not replaceable by an SA bearer token).
    assert "AWS_ACCESS_KEY_ID=ak-id" in staged_env
    assert "AWS_SECRET_ACCESS_KEY=sk-val" in staged_env
    assert "NEBIUS_SERVICE_ACCOUNT_ID=serviceaccount-abc" in staged_env

    # No copied operator IAM token / bootstrap profile anywhere.
    for forbidden in (
        "NEBIUS_IAM_TOKEN",
        "NPA_NEBIUS_IAM_TOKEN",
        "TF_VAR_iam_token",
        "NPA_REUSE_IAM_TOKEN",
        "agent-bootstrap",
        "nebius-token",
    ):
        assert forbidden not in staged_env, forbidden
    assert all("nebius-token" not in cmd for cmd in commands)
    assert all("agent-bootstrap" not in cmd for cmd in commands)


def test_agent_operator_profile_scopes_foxglove_token_to_private_credentials() -> None:
    from npa.cli.agent import _write_agent_operator_profile

    staged: list[tuple[str, str]] = []

    class _FakeSSH:
        def upload_private_text(self, content: str, remote_path: str) -> None:
            staged.append((remote_path, content))

        def run_or_raise(self, _command: str, **_kwargs) -> str:
            return ""

        def run(self, _command: str) -> str:
            return ""

    _write_agent_operator_profile(
        _FakeSSH(),
        ssh_user="ubuntu",
        project_alias="prod",
        project_id="project-abc",
        tenant_id="tenant-abc",
        region="eu-north1",
        tf_api_key="tf-unit-secret",
        foxglove_api_token="fox-unit-secret",
        s3_bucket="agent-state",
        s3_endpoint="https://storage.example",
        s3_access_key="access",
        s3_secret_key="secret",
        service_account_id="serviceaccount-agent",
    )

    credential_payloads = [
        content for _path, content in staged if '"FOXGLOVE_API_TOKEN"' in content
    ]
    assert len(credential_payloads) == 2
    assert all(
        '"FOXGLOVE_API_TOKEN": "fox-unit-secret"' in item
        for item in credential_payloads
    )
    for item in credential_payloads:
        payload = json.loads(item)
        project_store = payload["project_credentials"]
        assert project_store["schema_version"] == "npa.project-credentials.v2"
        assert project_store["current_project_id"] == "project-abc"
        project = project_store["projects"]["project-abc"]
        assert project["aliases"] == ["prod"]
        assert project["storage"]["bucket"] == "s3://agent-state"
        assert project["nebius"]["service_account_project_id"] == "project-abc"
        # The legacy compatibility view is retained only alongside an exact
        # selected owner; it must be byte-for-byte the selected project record.
        assert project_store["current_project_id"] == "project-abc"
        assert payload["storage"] == project["storage"]


def test_agent_operator_profile_resolves_foxglove_token_only_into_private_credentials(
    monkeypatch,
) -> None:
    import inspect

    from npa.cli.agent import _write_agent_operator_profile

    staged: list[tuple[str, str]] = []
    commands: list[str] = []

    class _FakeSSH:
        def upload_private_text(self, content: str, remote_path: str) -> None:
            staged.append((remote_path, content))

        def run_or_raise(self, command: str, **_kwargs) -> str:
            commands.append(command)
            return ""

        def run(self, command: str) -> str:
            commands.append(command)
            return ""

    monkeypatch.setattr(
        "npa.clients.credentials.load_credentials",
        lambda: SimpleNamespace(foxglove_api_token="fox-fallback-unit-secret"),
    )
    _write_agent_operator_profile(
        _FakeSSH(),
        ssh_user="ubuntu",
        project_alias="prod",
        project_id="project-abc",
        tenant_id="tenant-abc",
        region="eu-north1",
        tf_api_key="",
        s3_bucket="agent-state",
        s3_endpoint="https://storage.example",
        s3_access_key="access",
        s3_secret_key="secret",
    )

    public_payloads = [
        content for _path, content in staged if '"default_project"' in content
    ]
    credential_payloads = [
        content for _path, content in staged if '"FOXGLOVE_API_TOKEN"' in content
    ]
    assert len(public_payloads) == len(credential_payloads) == 2
    assert all("fox-fallback-unit-secret" not in payload for payload in public_payloads)
    assert all("fox-fallback-unit-secret" in payload for payload in credential_payloads)
    assert all("fox-fallback-unit-secret" not in command for command in commands)
    assert (
        "foxglove_api_token"
        not in inspect.signature(agent_module._bootstrap_agent_stack).parameters
    )


def test_resolve_deploy_storage_credentials_prefers_bootstrap_when_writable(
    monkeypatch,
) -> None:
    from npa.cli.agent import _resolve_deploy_storage_credentials

    monkeypatch.setattr(
        "npa.cli.agent._storage_credentials_allow_writes", lambda **_kwargs: True
    )
    monkeypatch.setattr(
        "npa.cli.agent.resolve_project_storage",
        lambda *_args, **_kwargs: SimpleNamespace(
            checkpoint_bucket="",
            endpoint_url="",
            aws_access_key_id="",
            aws_secret_access_key="",
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
        "npa.cli.agent._storage_credentials_allow_writes",
        lambda **kwargs: kwargs["bucket"] == "shared-bucket",
    )
    monkeypatch.setattr(
        "npa.cli.agent.resolve_project_storage",
        lambda *_args, **_kwargs: SimpleNamespace(
            checkpoint_bucket="s3://shared-bucket/checkpoints/",
            endpoint_url="https://storage.us-central1.nebius.cloud",
            aws_access_key_id="ak-shared",
            aws_secret_access_key="sk-shared",
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
        "npa.cli.agent.resolve_project_storage",
        lambda *_args, **_kwargs: SimpleNamespace(
            checkpoint_bucket="s3://project-bucket/isaac-runs/",
            endpoint_url="https://storage.us-central1.nebius.cloud",
            aws_access_key_id="ak-project",
            aws_secret_access_key="sk-project",
        ),
    )
    monkeypatch.setattr(
        "npa.clients.credentials.load_credentials",
        lambda **_kwargs: SimpleNamespace(
            s3_bucket="s3://shared-bucket/",
            s3_endpoint="https://storage.eu-north1.nebius.cloud",
            s3_access_key_id="ak-shared",
            s3_secret_access_key="sk-shared",
        ),
    )
    monkeypatch.setattr(
        "npa.cli.agent._storage_credentials_allow_writes", lambda **_kwargs: True
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


def test_resolve_deploy_storage_credentials_falls_back_to_shared(monkeypatch) -> None:
    from npa.cli.agent import _resolve_deploy_storage_credentials

    def _probe(**kwargs):
        return kwargs["bucket"] == "shared-bucket"

    monkeypatch.setattr("npa.cli.agent._storage_credentials_allow_writes", _probe)
    monkeypatch.setattr(
        "npa.cli.agent.resolve_project_storage",
        lambda *_args, **_kwargs: SimpleNamespace(
            checkpoint_bucket="s3://shared-bucket/",
            endpoint_url="https://storage.us-central1.nebius.cloud",
            aws_access_key_id="ak-shared",
            aws_secret_access_key="sk-shared",
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


def test_resolve_deploy_storage_credentials_rejects_shared_for_explicit_project(
    monkeypatch,
) -> None:
    from npa.cli.agent import (
        AgentStorageCredentialError,
        _resolve_deploy_storage_credentials,
    )

    monkeypatch.setattr(
        "npa.cli.agent.resolve_project_storage",
        lambda *_args, **_kwargs: SimpleNamespace(
            checkpoint_bucket="s3://project-bucket/",
            endpoint_url="https://storage.us-central1.nebius.cloud",
            aws_access_key_id="ak-project",
            aws_secret_access_key="sk-project",
        ),
    )
    monkeypatch.setattr(
        "npa.cli.agent.resolve_terraform_state",
        lambda _project: SimpleNamespace(
            bucket="state-bucket",
            endpoint="https://storage.us-central1.nebius.cloud",
            access_key="ak-state",
            secret_key="sk-state",
        ),
    )
    monkeypatch.setattr(
        "npa.cli.agent._storage_credentials_allow_writes",
        lambda **kwargs: kwargs["bucket"] == "shared-bucket",
    )

    def _shared_credentials_must_not_be_loaded(**_kwargs):
        raise AssertionError("explicit-project deploy consulted shared credentials")

    monkeypatch.setattr(
        "npa.clients.credentials.load_credentials",
        _shared_credentials_must_not_be_loaded,
    )

    with pytest.raises(AgentStorageCredentialError):
        _resolve_deploy_storage_credentials(
            region="us-central1",
            project_alias="target-project",
            bootstrap_creds={
                "s3_bucket": "bootstrap-bucket",
                "s3_endpoint": "https://storage.us-central1.nebius.cloud",
                "nebius_api_key": "ak-bootstrap",
                "nebius_secret_key": "sk-bootstrap",
            },
        )


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

    monkeypatch.setattr("npa.cli.agent._storage_credentials_allow_writes", _probe)
    monkeypatch.setattr(
        "npa.cli.agent.resolve_terraform_state", lambda _project: _TfState()
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
    from npa.cli.agent import (
        AgentStorageCredentialError,
        _resolve_deploy_storage_credentials,
    )

    monkeypatch.setattr(
        "npa.cli.agent._storage_credentials_allow_writes", lambda **_kwargs: False
    )
    monkeypatch.setattr(
        "npa.cli.agent.resolve_project_storage",
        lambda *_args, **_kwargs: SimpleNamespace(
            checkpoint_bucket="s3://shared-bucket/",
            endpoint_url="https://storage.us-central1.nebius.cloud",
            aws_access_key_id="ak-shared",
            aws_secret_access_key="sk-shared",
        ),
    )
    bootstrap = {
        "s3_bucket": "bucket-boot",
        "s3_endpoint": "https://storage.us-central1.nebius.cloud",
        "nebius_api_key": "ak-boot",
        "nebius_secret_key": "sk-boot",
    }

    with pytest.raises(AgentStorageCredentialError):
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
    monkeypatch.setattr("npa.cli.agent._bootstrap_agent_stack", lambda **_kwargs: None)
    monkeypatch.setattr("npa.cli.agent.ensure_ingress", lambda **_kwargs: None)
    monkeypatch.setattr(
        "npa.cli.agent.remove_npa_ingress_for_instance_ports",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr("npa.cli.agent.write_config", _write_config)

    # Satisfy the fail-fast deploy prerequisites (terraform + SSH key pair) that
    # now run before any cloud side effects.
    (tmp_path / "id_ed25519.pub").write_text(
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAABAgMEBQYHCAkKCwwNDg8QERITFBUWFxgZGhscHR4f test\n"
    )
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


def test_deploy_feedback_names_bounded_phases_and_quiet_period() -> None:
    import inspect

    from npa.cli.agent import deploy_cmd

    source = inspect.getsource(deploy_cmd)
    for phase in ("Phase 1/4", "Phase 2/4", "Phase 3/4", "Phase 4/4 probe"):
        assert phase in source
    assert "quiet for several minutes" in source
    assert "journalctl -u cloud-final -u npa-agent-backend" in source


def test_bootstrap_enables_public_https_nginx() -> None:
    source = _agent_source()
    assert "ssl_certificate /etc/nginx/ssl/npa-agent.crt" in source
    assert "DEFAULT_HTTPS_PORT" in source
    assert "Customer URL: use" in source
    assert "--no-public-https" in source


def test_public_https_keeps_backend_loopback_only() -> None:
    source = _agent_source()
    assert "uvicorn backend:app --host 127.0.0.1 --port {backend_port}" in source
    assert "uvicorn backend:app --host 0.0.0.0 --port {backend_port}" not in source
    assert "proxy_pass http://127.0.0.1:{backend_port}/;" in source


def test_public_ingress_excludes_internal_backend_port() -> None:
    from npa.cli.agent import _agent_extra_ingress_ports

    ports = _agent_extra_ingress_ports(
        agent_port=8088,
        rerun_port=9090,
        public_https=True,
    )
    assert ports == [443, 9090]
    assert 8787 not in ports


def test_existing_agent_bootstrap_fails_closed_when_https_ingress_cannot_be_ensured() -> (
    None
):
    source = _agent_source()
    assert '_fail(f"npa network ensure-ingress failed: {exc}")' in source
    assert "Customer HTTPS on port 443 may be unreachable" not in source


def test_leisaac_signaling_uses_backend_session_auth() -> None:
    source = _agent_source()
    assert "location ^~ /api/leisaac/signal {{" not in source
    for route in ("/api/leisaac/signal", "/api/leisaac/signal/sign_in"):
        location = source.split(f"location = {route} {{{{", 1)[1].split("  }}", 1)[0]
        assert "auth_basic off" in location
        assert "proxy_set_header Upgrade $http_upgrade;" in location
        assert "proxy_set_header Host $http_host;" in location
        assert "proxy_set_header Origin $http_origin;" in location
    general_api = source.split("location /api/ {{", 1)[1].split("  }}", 1)[0]
    assert "auth_basic off" not in general_api
    assert "Server-level Basic auth protects the general /api/ location" in source
    assert "exact signaling WebSocket routes turn it off" in source


def test_leisaac_pod_backhaul_uses_authenticated_public_https_only() -> None:
    source = _agent_source()
    location = source.split("location = /api/leisaac/backhaul {{", 1)[1].split(
        "location /api/ {{", 1
    )[0]
    assert "auth_basic off" not in location
    assert "proxy_pass http://127.0.0.1:{backend_port}/;" in location
    assert "proxy_set_header X-Forwarded-Proto $scheme;" in location


def test_bootstrap_nginx_serves_public_rerun_recording() -> None:
    source = _agent_source()
    assert "location /rerun/recordings/" in source
    assert "auth_basic off" in source
    assert "alias /opt/npa-agent/recordings/" in source
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
    # Region-agnostic image acquisition uses the anonymous public GHCR release.
    assert "lichtblick_pull_candidates" in source
    assert "for lb_cand in {lichtblick_pull_candidates}" in source
    assert "npa-lichtblick image acquired from" in source
    assert "docker login" not in source


def test_lichtblick_recordings_grant_no_cross_origin_read() -> None:
    """The MCAP alias is unauthenticated, so it must not be CORS-readable.

    A run's MCAP carries camera frames, VLM critiques and reward signals, and the
    location runs with ``auth_basic off`` (wasm/worker fetches cannot carry basic
    auth). A wildcard ``Access-Control-Allow-Origin`` would let any page a viewer
    visits read those recordings off this host; the embed is same-origin and needs
    no CORS grant at all.
    """

    source = _agent_nginx_site()
    recordings_location = source.split("location /lichtblick/recordings/ {", 1)[
        1
    ].split("location = /lichtblick/ {", 1)[0]
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
    assert (
        'const pinned = pinLichtblickDsToSameOrigin(url) || "/lichtblick/";' in source
    )
    assert 'viewer.searchParams.set("npa.layout", layoutKind);' in source
    assert 'viewer.searchParams.set("npa.camera"' in source
    assert "learningArtifactContractFor" in source
    assert "contract.matches.mcap" in source


def test_lichtblick_uses_native_static_size_and_range_semantics() -> None:
    """The reader gets size/ranges from nginx without a script import proxy."""

    source = _agent_nginx_site()
    block = source.split("location /lichtblick/recordings/ {", 1)[1].split(
        "location = /lichtblick/ {", 1
    )[0]
    assert "alias /opt/npa-agent/recordings/;" in block
    assert "proxy_pass" not in block
    assert "gzip off;" in block
    assert 'Cache-Control "no-cache, no-transform"' in block
    assert "Accept-Ranges" in block
    assert "Access-Control-Allow-Origin" not in block


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
    assert "if (lichtblickNeedsLayoutSeed(simViz)) {" in mount
    reset_calls = mount.count("resetLichtblickLayoutStorage()")
    assert reset_calls == 1, f"expected one guarded wipe, found {reset_calls}"


def test_bootstrap_injects_lichtblick_default_layout() -> None:
    source = _agent_source()
    nginx = _agent_nginx_site()
    # The viewer document is exact-matched so nginx can inject a default layout via
    # the upstream-provided placeholder, so the point cloud + camera show on load.
    assert "location = /lichtblick/ {" in nginx
    assert (
        "sub_filter '/*LICHTBLICK_SUITE_DEFAULT_LAYOUT_PLACEHOLDER*/' '(()=>{" in nginx
    )
    assert "def _lichtblick_default_layout_json" in source

    from npa.cli import agent_assets
    from npa.cli import agent_site as agent_site_module

    layout = json.loads(agent_assets._lichtblick_default_layout_json())
    panels = layout["configById"]
    three_d = next(v for k, v in panels.items() if k.startswith("3D!"))
    assert three_d["topics"]["/heldout/points"]["visible"] is True
    assert three_d["followTf"] == "sim2real"
    image = next(v for k, v in panels.items() if k.startswith("Image!"))
    assert image["imageMode"]["imageTopic"] == "/camera"
    learning_layout = json.loads(agent_site_module._lichtblick_learning_layout_json())
    learning_image = next(
        v for k, v in learning_layout["configById"].items() if k.startswith("Image!")
    )
    assert learning_layout["layout"].startswith("Image!")
    assert learning_image["imageMode"]["imageTopic"] == "/camera/__NPA_PRIMARY_CAMERA__"
    script = agent_site_module._lichtblick_default_layout_script()
    assert 'query.get("npa.layout")!=="learning"' in script
    assert 'query.get("npa.camera")' in script
    assert 'imageTopic="/camera/"+camera' in script
    assert "window.Worker=function(scriptUrl,options)" in script
    assert 'new URL("/lichtblick/npa-worker.js"' in script
    assert "location = /lichtblick/npa-worker.js {" in nginx
    assert "npa.target" in nginx


def test_lichtblick_nginx_inline_javascript_has_no_nginx_variables_or_controls() -> (
    None
):
    """Inline nginx directive values cannot contain raw controls or bare ``$``."""

    from npa.cli.agent_site import (
        _lichtblick_default_layout_script,
        _lichtblick_worker_script,
    )

    for script in (_lichtblick_default_layout_script(), _lichtblick_worker_script()):
        assert "$" not in script
        assert "\\" not in script
        assert not [char for char in script if ord(char) < 32 or ord(char) == 127]


@requires_node
def test_lichtblick_worker_accepts_only_same_origin_lichtblick_javascript() -> None:
    from npa.cli.agent_site import _lichtblick_worker_script

    worker = _lichtblick_worker_script()
    harness = r"""
const target = process.argv[1];
global.self = {
  location: new URL("https://agent.example/lichtblick/npa-worker.js?npa.size=217423&npa.target=" + encodeURIComponent(target)),
  fetch: async () => new Response(null, {status: 200, headers: {"accept-ranges": "bytes"}}),
};
global.importScripts = (url) => process.stdout.write("IMPORTED=" + url);
eval(Buffer.from(process.argv[2], "base64").toString("utf8"));
"""
    encoded = base64.b64encode(worker.encode()).decode()
    allowed = "/lichtblick/assets/mcap.worker.js"
    result = subprocess.run(
        ["node", "-e", harness, allowed, encoded],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (
        result.stdout
        == "IMPORTED=https://agent.example/lichtblick/assets/mcap.worker.js"
    )
    for target in (
        "//foreign.invalid/worker.js",
        "https://foreign.invalid/worker.js",
        "https://user:password@agent.example/lichtblick/worker.js",
        "javascript:alert(1)",
        "data:text/javascript,alert(1)",
        "/api/private.js",
        "/lichtblick/recordings/run.mcap",
        "/lichtblick/npa-worker.js",
        "/lichtblick/%252e%252e/api/private.js",
        "/lichtblick/worker.js?next=/lichtblick/good.js",
        "/lichtblick/worker.js%3fnext=/lichtblick/good.js",
        "/lichtblick/worker.js#fragment",
        "/lichtblick\\worker.js",
        "/lichtblick/worker.js\x01",
    ):
        rejected = subprocess.run(
            ["node", "-e", harness, target, encoded],
            check=False,
            capture_output=True,
            text=True,
        )
        assert rejected.returncode != 0, target
        assert "invalid Lichtblick worker target" in rejected.stderr


def test_lichtblick_anonymous_routes_do_not_disable_api_authentication() -> None:
    source = _agent_nginx_site()
    api_block = source.split("location /api/ {", 1)[1].split(
        "location /assets/api/", 1
    )[0]
    assert "auth_basic off" not in api_block
    assert "location = /lichtblick/npa-worker.js {" in source
    assert "location /lichtblick/recordings/ {" in source


def test_bootstrap_installs_docker_for_fresh_lichtblick_fallback() -> None:
    source = _agent_source()
    assert "if ! command -v docker >/dev/null 2>&1; then" in source
    assert "apt-get install -y docker.io" in source
    assert "sudo systemctl enable --now docker" in source


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


def test_bootstrap_ui_mcap_cards_bind_exact_provenance_in_page() -> None:
    source = _agent_ui_bundle()
    assert 'data-action="open-foxglove-artifact"' in source
    assert 'data-sd-action="foxglove"' in source
    assert "artifactFoxgloveSelection" in source
    assert "viewFoxgloveArtifact(artifactFoxgloveSelection(btn))" in source
    assert 'String(selected.s3_uri || "").trim()' in source
    embedded_handler = source.split("async function viewFoxgloveArtifact", 1)[1].split(
        "async function openFoxgloveWeb", 1
    )[0]
    assert 'setRenderMode("foxglove")' in embedded_handler
    assert 'apiJson("/api/foxglove/export"' in embedded_handler
    assert "resource_bucket: selected.bucket" in embedded_handler
    assert "project_id: selected.project_id" in embedded_handler
    assert "resolved_prefix: selected.resolved_prefix" in embedded_handler
    assert "await setFoxgloveDataSource(config);" in embedded_handler
    assert (
        "await setFoxgloveDataSource(config, { force: true })" not in embedded_handler
    )
    assert "const responseConfig = data && data.foxglove" in embedded_handler
    assert "setFoxgloveSwitching(Boolean(foxgloveHandle" in embedded_handler
    assert embedded_handler.index(
        "const pinnedBeforeSelection"
    ) < embedded_handler.index("foxglovePinnedArtifactSelection = { ...selected")
    before_export = embedded_handler.split('apiJson("/api/foxglove/export"', 1)[0]
    assert "teardownFoxgloveViewer()" not in before_export
    assert "const readinessPromise = selectedHandle.whenReady()" in embedded_handler
    assert "void readinessPromise.then" in embedded_handler
    assert "this pane is not marked ready" in embedded_handler
    assert "foxgloveHandle.selectLayout" in embedded_handler
    assert "setFoxgloveActiveArtifactState" in embedded_handler
    assert "++foxgloveArtifactOperationSequence" in embedded_handler
    assert "prior.controller.abort()" in embedded_handler
    assert "foxgloveArtifactOperation !== operation" in embedded_handler
    assert "foxglovePinnedArtifactSelection" in embedded_handler
    assert "showFoxgloveArtifactFailure" in embedded_handler
    assert 'id="foxgloveArtifactRetry"' in source
    assert 'button.setAttribute("aria-busy", busy ? "true" : "false")' in source
    assert "window.open" not in embedded_handler
    assert "location.replace" not in embedded_handler
    assert "window.npaAgentArtifacts = Object.freeze" in source
    assert "loadExactSource: loadExactArtifactSource" in source
    exact_source_handler = source.split("async function loadExactArtifactSource", 1)[
        1
    ].split("function learningStagesFromContract", 1)[0]
    # Exact-source inventory now discovers every page before selecting and
    # immediately opens the global preferred recording.
    assert "deferPreferredViewer: false" in exact_source_handler
    assert "loadArtifactsForSelectedRun(runRef" in exact_source_handler
    external_handler = source.split("async function openFoxgloveWeb", 1)[1].split(
        "async function captureFoxgloveContext", 1
    )[0]
    assert 'window.open("about:blank", "_blank")' in external_handler
    assert 'id="viewerPaneFoxglove"' in source
    foxglove_pane = source.split('id="viewerPaneFoxglove"', 1)[1].split(
        'id="viewerPaneMedia"', 1
    )[0]
    assert "Open in Foxglove</button>" in foxglove_pane
    assert source.index("View in Foxglove") < source.index("View in Lichtblick")


def test_bootstrap_artifact_file_transcodes_ppm_to_png() -> None:
    source = _agent_source()
    # .ppm/.bmp/.tiff are transcoded to PNG on serve so the browser can render them.
    assert "needs_image_transcode(safe_name)" in source
    assert 'media_type="image/png"' in source


def test_franka_rerun_fallback_keeps_3d_outside_pinhole_projection() -> None:

    source = _agent_source()
    assert "_franka_demo_joint_angles" in source
    assert "frame_count = 90" in source
    assert "world/camera_frustums/{{name}}" in source
    assert 'f"{entity}/frustum"' not in source
    assert 'f"{entity}/origin"' not in source


def test_agent_artifact_discovery_requires_s3_components() -> None:

    source = _agent_source()
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
    assert "registry placeholders" in source
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
    # Multi-bucket discovery: the agent searches every accessible bucket (never
    # relies on copying a run into one bucket).
    assert "def _agent_s3_buckets(" in source
    assert "accessible_artifact_buckets(_agent_access_report())" in source
    assert "list_runs_cached_multi" in source
    assert "find_run_artifacts_across_buckets" in source
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

    source = _agent_source()
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
    assert 'data-tab="rerun">View</button>' in source
    viewer_tab_markup = source.split('<div class="render-mode-tabs"', 1)[1].split(
        "</div>", 1
    )[0]
    view_tab = viewer_tab_markup.index('id="renderModeRerun"')
    foxglove_tab = viewer_tab_markup.index('id="renderModeFoxglove"')
    lichtblick_tab = viewer_tab_markup.index('id="renderModeLichtblick"')
    assert view_tab < foxglove_tab < lichtblick_tab
    assert 'id="renderModeRerun" data-render-mode="rerun">View</button>' in source
    assert 'role="tab" aria-selected="true" aria-controls="viewerPaneRerun"' in source
    assert 'data-testid="open-foxglove-web"' in source
    assert "View in Foxglove</button>" in source
    assert "Open in Foxglove</button>" in source
    assert 'id="foxgloveVisualizationSummary"' in source
    assert "prepareFoxgloveVisualization" in source
    assert "let foxglovePreparePromise = null" in source
    assert "let foxgloveConfigWarmPromise = null" in source
    assert "void warmFoxgloveConfig()" in source
    assert "foxgloveConfigForActiveRun" in source
    assert "visualization.checked" in source
    ensure_foxglove = source.split("async function ensureFoxgloveViewer", 1)[1].split(
        "function teardownFoxgloveViewer", 1
    )[0]
    assert "config = await prepareFoxgloveVisualization(config)" not in ensure_foxglove
    assert ensure_foxglove.index("mod.mountFoxgloveViewer") < ensure_foxglove.rindex(
        "prepareFoxgloveVisualizationAfterMount(config)"
    )
    assert '"-source-default"' in _agent_source()
    assert 'pane.setAttribute("aria-hidden"' in source
    assert 'btn.setAttribute("aria-selected"' in source
    assert 'event.key === "ArrowRight"' in source
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
        "<h3>Tenant resources</h3>",
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
    from npa.genesis.scene_assets import (
        CAMERA_PLACEMENT_STOCK_EE_MOUNTED,
        CAMERA_PLACEMENT_STOCK_WORKSPACE,
        DEFAULT_CAMERA_NAMES,
    )

    source = _agent_source()
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
    assert "RERUN_CAPABILITY_NAME_RE" in source
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
    assert "local_media_type = artifact_media_type(safe_name)" in source
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

    stage_runtime = (
        Path(agent_module.__file__)
        .with_name("agent_stage_runtime.py")
        .read_text(encoding="utf-8")
    )
    assert "def _artifact_backed_run_details" in stage_runtime
    assert "def _workflow_stage_defs_from_state" in stage_runtime
    assert "artifact presence does not establish execution success" in source
    assert "npa.stage-evidence/v1" in source
    assert "runDetailsRequestId" in source
    assert "runDetailsAbortController" in source
    assert "execution status unavailable" in source
    assert (
        "Never let a sparse update erase richer artifact fields from load-run" in source
    )
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
    # Every artifact must be directly downloadable: streaming download endpoint
    # + a per-artifact Download button wired to it.
    assert '@app.api_route("/artifacts/download", methods=["GET", "HEAD"])' in source
    assert (
        'data-action="download-artifact"' in source
        or "data-action='download-artifact'" in source
    )
    assert "function downloadArtifact(" in source
    assert "/api/artifacts/content?" in source
    # Clicking a stage describes it and inlines its artifacts/info/configs.
    assert '@app.get("/artifacts/stage/{{run_id:path}}")' in source
    assert "async function showStageDetail(" in source
    assert "/api/artifacts/stage/" in source
    assert 'id="stageDetail"' in source
    assert "data-stage-label=" in source
    # Artifact-backed dataset/provenance tab. It must not present its own grid as
    # a Voxel51/FiftyOne imitation, and it must separate source from generated data.
    assert '@app.get("/fiftyone/dataset/{{run_id:path}}")' in source
    assert "build_fiftyone_dataset" in source
    assert 'id="tabVoxel"' in source
    assert 'id="panelVoxel"' in source
    assert 'data-tab="voxel51"' in source
    assert "async function loadVoxelDataset(" in source
    assert "/api/fiftyone/dataset/" in source
    assert 'id="voxelGrid"' in source
    assert "Dataset &amp; provenance" in source
    assert "FiftyOne-style" not in source
    assert "Source input" in source
    assert "Derived conditioning data" in source
    assert "Synthetic / augmented data" in source
    assert "Artifact summary only — FiftyOne did not run" in source
    assert 'id="voxelReview"' in source
    assert "data_role_label" in source
    # Loading by run-relative key resolves a discovered object. An unscoped exact
    # S3 URI receives the structured v2 migration error instead of guessing a run.
    assert "resolve_run_artifacts(" in source
    assert '"contract_version": "npa.agent.load-artifact.v2"' in source
    assert '"code": "run_id_required_for_s3_uri"' in source
    assert 'may_use_default_recording = payload_run in {"", "franka-demo"}' in source
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


def test_direct_run_load_cancels_background_discovery_and_uses_exact_artifacts() -> (
    None
):
    source = _agent_ui_bundle()

    assert "let artifactRunsAbortController = null;" in source
    assert "artifactRunsAbortController.abort();" in source
    assert "Exact run loading takes precedence" in source
    assert (
        "await loadArtifactsForSelectedRun(runRef || runId, null, exactEntry" in source
    )
    assert "if (loaded && activeArtifactInventory.length)" in source
    assert 'const artifactsPromise = refreshArtifactRuns("", {' in source
    assert "singlePage: true," in source
    assert "background: true," in source
    assert "Render the authoritative workflow timeline before attempting" in source
    assert "!context.deferPreferredViewer && !context.suppressPreferredAutoload && preferred" in source
    assert "deferPreferredViewer: true" in source
    assert 'showToast("Run loaded; preferred viewer failed: "' in source
    assert '"#stageList .stage-physical-job"' in source
    assert (
        "if (!physicalStageCount) await loadRunDetails(runId, detailOptions);" in source
    )


def test_artifact_inventory_autopaginates_before_global_preference_and_selection() -> None:
    source = _agent_ui_bundle()
    block = source.split(
        "async function loadArtifactsForSelectedRun", 1
    )[1].split("async function loadExactArtifactSource", 1)[0]

    assert "const seenCursors = new Set();" in block
    assert "while (nextCursor)" in block
    assert "seenCursors.has(nextCursor)" in block
    assert "paginationEmptyPageCount" in block
    assert "paginationDuplicateCount" in block
    assert "Artifact inventory source changed during pagination" in block
    assert "Artifact inventory is truncated but the server returned no continuation cursor" in block
    assert 'continuation.set("project_id", selectedSource.project_id);' in block
    assert 'continuation.set("resource_bucket", selectedSource.bucket);' in block
    assert 'continuation.set("resolved_prefix", selectedSource.resolved_prefix);' in block
    assert 'continuation.set("source_selected", "1");' in block
    assert "const preferred = selectPreferredArtifact(artifacts);" in block
    assert block.index("while (nextCursor)") < block.index("setActiveRunId(runId)")
    assert block.index("selectPreferredArtifact(artifacts)") < block.index(
        "setActiveRunId(runId)"
    )


def test_artifact_backed_training_run_loads_without_rerun_recording() -> None:
    source = _agent_ui_bundle()
    backend = (
        Path(__file__)
        .resolve()
        .parents[2]
        .joinpath("src/npa/cli/agent.py")
        .read_text(encoding="utf-8")
    )

    assert '"output_artifact_count"' in backend
    assert '"preview_status": "no_previewable_recording"' in backend
    assert '"artifacts_available": True' in backend
    status_body = backend.split('@app.get("/sim-viz/status")', 1)[1].split(
        '@app.get("/sim-viz/runs")', 1
    )[0]
    assert 'preview_status == "no_previewable_recording"' in status_body
    assert 'payload["rerun_ready"] = False' in status_body
    assert 'state: "no-preview-artifacts"' in source
    assert 'placeholder.setAttribute("data-state"' in source
    assert "No RRD/MCAP recording; use the artifacts below" in source


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
    assert (
        '["artifactStageFilter", "artifactTypeFilter", "artifactRoleFilter", "artifactSort"]'
        in source
    )
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
        in " ".join(source.split())
    )


def test_groot_learning_recording_activates_real_rrd_and_truthful_note() -> None:
    from npa.cli import agent as agent_module
    from npa.cli import agent_viewer_runtime

    source = Path(agent_viewer_runtime.__file__).read_text(encoding="utf-8")
    branch = source.split('if render == "rerun":', 1)[1].split(
        'elif render == "mcap":', 1
    )[0]
    assert "if is_learning:" in branch
    assert 'sim_viz["preview_entity"] = f"heldout/camera/{camera}"' in branch
    assert "validated primary camera is {camera}" in branch
    assert "Offline held-out GR00T policy evaluation loaded (not a rollout)." in branch
    assert "finite training loss, and provenance" in branch
    assert (
        "Offline held-out GR00T policy evaluation loaded (not a rollout)"
        in agent_module._embedded_agent_viewer_runtime_source()
    )
    assert 'rrd_tmp = RRD_PATH.with_suffix(".rrd.tmp")' in branch
    assert "shutil.copy2(local_path, rrd_tmp)" in branch
    assert branch.index("rrd_tmp.replace(RRD_PATH)") < branch.index(
        "_restart_rerun_serve(force=True)"
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
    assert "discoveredArtifactRuns = [...runs];" in source
    assert '(cursor ? " · loading more…" : "")' in source
    # The run selector is a UNION of known + discovered runs (does not clobber).
    assert "mergeRunsLatestFirst(knownAvailableRuns, discoveredArtifactRuns)" in source
    assert 'fillRunSelectOptionsRich(document.getElementById("runIdSelect")' in source


def test_bootstrap_run_history_uses_run_id_index() -> None:

    source = _agent_source()
    assert '"sim_viz_runs": []' not in source
    assert "if not isinstance(runs, dict):" in source
    assert "runs[history_key] = snapshot" in source
    assert 'state["active_run_id"] = run_id' in source
    assert 'state["active_run_ref"] = run_ref' in source
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

    source = _agent_source()
    # Generic scan across all bucket roots AND every accessible bucket; no
    # hardcoded workflow prefixes. The no-prefix endpoint calls the multi-bucket
    # cached wrapper (which discovers via list_all_runs per bucket under the hood).
    assert "list_runs_cached_multi(" in source
    assert "exclude=_discovery_exclude_roots()" in source
    assert "AGENT_DEFAULT_WORKFLOW_PREFIXES" not in source
    # Per-run lookup falls back to a generic cross-category, cross-bucket find.
    runtime = (
        Path(agent_module.__file__)
        .with_name("agent_stage_runtime.py")
        .read_text(encoding="utf-8")
    )
    assert "find_run_artifacts_across_buckets(" in runtime


def test_run_details_resolves_run_generically_by_id() -> None:
    """Stage determination must resolve a run generically by id (across all
    categories under the run root) so any run shows real artifact-backed stages
    instead of the generic sim2real 'not_run' template — no path/prefix required.
    """

    source = (
        Path(agent_module.__file__)
        .with_name("agent_stage_runtime.py")
        .read_text(encoding="utf-8")
    )
    # Backend resolves the run generically across categories (no prefix needed).
    assert "def _artifact_backed_run_details(" in source
    assert 'resource_bucket: str = ""' in source
    assert "find_run_artifacts_across_buckets(" in source
    # Frontend loads run details / run by id WITHOUT a path prefix.
    ui = _agent_ui_bundle()
    assert '"/api/workflows/sim2real/runs/" + encodeURIComponent(target)' in ui
    assert "body: JSON.stringify({ run_id: targetRunId, run_ref: targetRunRef })" in ui
    assert 'entry.source_type === "artifact_storage"' in ui
    assert (
        "loadArtifactsForSelectedRun(chosen, null, entry, { pendingSelection: true })"
        in ui
    )
    assert "prefix: artifactPrefixValue()" not in ui
    assert 'params.set("resource_bucket", resourceBucket)' in ui
    assert 'params.set("resolved_prefix", resolvedPrefix)' in ui
    assert 'params.set("source_selected", "1")' in ui
    assert '"stages succeeded"' not in ui


def test_artifact_cards_define_runtime_metadata_before_rendering() -> None:
    """Artifact card rendering must not fail on undefined metadata variables."""
    ui = _agent_ui_bundle()

    assert "const learningSummary = data && data.summary && data.summary.learning" in ui
    assert "list.hidden = Boolean(learningSummary);" in ui
    assert 'const s3uri = String(item.s3_uri || "");' in ui
    assert "data-s3-uri=\"' + escapeHtml(s3uri)" in ui


def test_agent_ui_surfaces_physical_managed_job_ids_per_stage() -> None:
    ui = _agent_ui_bundle()

    assert 'const managedJobId = String(stage.job_id || "").trim();' in ui
    assert 'class="stage-physical-job"' in ui
    assert "Physical managed job ID:" in ui


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
    monkeypatch.setattr(
        "npa.cli.agent._agent_record",
        lambda project, name: {
            "public_ip": "8.8.8.8",
            "agent_url": "https://8.8.8.8/",
            "public_url": "https://8.8.8.8/",
            "public_https": True,
            "direct_url": "http://8.8.8.8:8088/",
            "rerun_url": "https://8.8.8.8/rerun/",
            "sim_viz_url": "https://8.8.8.8/rerun/",
            "sim_assets_url": "https://8.8.8.8/assets/",
            "cameras_api_url": "https://8.8.8.8/assets/api/sim-assets/cameras",
            "auth_secret_path": "/tmp/agent-auth",
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
        "npa.cli.agent._basic_auth_protects_endpoint",
        lambda *_args, **_kwargs: (True, 401),
    )

    result = runner.invoke(app, ["status", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["health"] is True
    assert payload["basic_auth_enforced"] is True
    assert payload["unauthenticated_ui_status_code"] == 401
    assert payload["endpoint_disclosure_allowed"] is True
    assert payload["ui_status_code"] == 200
    assert payload["rerun_status_code"] == 200
    assert payload["public_url"] == "https://8.8.8.8/"
    assert payload["sim_viz_url"].endswith("/rerun/")
    assert payload["sim_assets_url"].endswith("8.8.8.8/assets/")
    assert payload["cameras_api_url"].endswith("/assets/api/sim-assets/cameras")
    assert payload["direct_url"] == ""


def test_agent_status_withholds_endpoint_when_basic_auth_is_not_enforced(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "npa.cli.agent._agent_record",
        lambda project, name: {
            "public_ip": "8.8.8.8",
            "agent_url": "https://8.8.8.8/",
            "public_url": "https://8.8.8.8/",
            "public_https": True,
            "direct_url": "http://8.8.8.8:8088/",
            "rerun_url": "https://8.8.8.8/rerun/",
            "sim_viz_url": "https://8.8.8.8/rerun/",
            "sim_assets_url": "https://8.8.8.8/assets/",
            "cameras_api_url": "https://8.8.8.8/assets/api/sim-assets/cameras",
            "auth_secret_path": "/tmp/agent-auth",
            "llm": {},
        },
    )
    monkeypatch.setattr("npa.cli.agent._load_auth_secret", lambda _: ("npa", "secret"))
    monkeypatch.setattr("npa.cli.agent._health", lambda *_args, **_kwargs: (True, 200))
    monkeypatch.setattr(
        "npa.cli.agent._basic_auth_protects_endpoint",
        lambda *_args, **_kwargs: (False, 200),
    )

    result = runner.invoke(app, ["status", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["health"] is False
    assert payload["basic_auth_enforced"] is False
    assert payload["unauthenticated_ui_status_code"] == 200
    assert payload["endpoint_disclosure_allowed"] is False
    for key in (
        "public_ip",
        "public_url",
        "direct_url",
        "ui_url",
        "rerun_url",
        "sim_viz_url",
        "sim_assets_url",
        "cameras_api_url",
    ):
        assert payload[key] == ""


def test_agent_status_not_found_json_is_nonzero(monkeypatch) -> None:
    monkeypatch.setattr("npa.cli.agent._agent_record", lambda _project, _name: {})
    monkeypatch.setattr(
        "npa.agent_status.partial_agent_status",
        lambda project, name: {
            "project": project,
            "name": name,
            "classification": "NOT_FOUND",
        },
    )

    result = runner.invoke(app, ["status", "--project", "demo", "--json"])

    assert result.exit_code == 1
    assert json.loads(result.output)["classification"] == "NOT_FOUND"


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
            "agent_url": "https://8.8.8.8/",
            "public_url": "https://8.8.8.8/",
            "public_https": True,
            "direct_url": "http://8.8.8.8:8088/",
            "rerun_url": "https://8.8.8.8/rerun/",
            "sim_viz_url": "https://8.8.8.8/rerun/",
            "sim_assets_url": "https://8.8.8.8/assets/",
            "cameras_api_url": "https://8.8.8.8/assets/api/sim-assets/cameras",
            "auth_secret_path": "/tmp/agent-auth",
        },
    )
    monkeypatch.setattr("npa.cli.agent._load_auth_secret", lambda _: ("npa", "secret"))
    monkeypatch.setattr("npa.cli.agent._health", lambda *_args, **_kwargs: (True, 200))

    def _fake_http_get(url, *_args, **_kwargs):
        url_s = str(url)
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
        if url_s.endswith("/api/leisaac/status"):
            return _Resp(
                {
                    "available": False,
                    "episodes_available": False,
                    "run_id": "",
                    "reason": "No LeIsaac runtime is registered with this agent.",
                }
            )
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
            return _Resp(
                {
                    "latest_submit": {"run_id": "agent-run-123"},
                    "sim_viz": {"stage": "demo"},
                }
            )
        if url_s.endswith("/welcome"):
            return _Resp("<html>NPA Agent is running</html>", status_code=200)
        if url_s.endswith("/healthz"):
            return _Resp('{"ok":true}', status_code=200)
        if "/rerun/" in url_s:
            return _Resp(b"console.log('rerun');", status_code=200)
        if url_s.rstrip("/").endswith(("8.8.8.8", ":8088")):
            if _kwargs.get("auth") is None:
                return _Resp({"detail": "authentication required"}, status_code=401)
            html = (
                f'<html><head><meta name="viewport" content="width=device-width, initial-scale=1">'
                f'<meta name="npa-ui-version" content="{AGENT_UI_VERSION}">'
                '<meta name="leisaac-control-readiness-contract" '
                'content="LEISAAC_CONTROL_READINESS_CONTRACT"></head>'
                "<body>"
                '<div id="tabMain"></div><div id="tabRerun"></div>'
                '<div id="agentAccessPanel"></div><button id="agentAccessRefresh"></button>'
                '<select id="agentAccessProjectSelect"></select>'
                '<article class="access-project-detail">No searchable artifact bucket.</article>'
                '<script>function refreshAccess(){ fetch("/api/access"); }</script>'
                '<div id="stagesPanel"><h3>Stages</h3>'
                '<div class="stages-run-picker">'
                '<select id="stagesRunSelect"></select>'
                "<label>Search NPA workflow/artifact runs</label>"
                '<input id="stagesRunInput" />'
                '<button id="stagesLoadRun"></button></div></div>'
                '<div id="tenantResourcesPanel"><h3>Tenant resources</h3>'
                '<button id="tenantResourcesRefresh"></button>'
                "Accessible / discovered; Configured references</div>"
                "<script>function loadSelectedRun(){} function syncRunChooserFields(){} "
                "function filterStagesRunSelect(){} function resolveStagesRunChoice(){}</script>"
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
                "<script>function wireUi(){} function sendChat(){} function activateMainTab(){} "
                "function authenticatedPreviewObjectUrl(){} function waitUntilRerunPastBundleSplash(){} "
                "function scheduleRerunBundleUncover(){} function swapRerunRecordingInPlace(){} "
                "function safeHideRerunBundleCover(){} function captureVisualContext(){} "
                "function describeVisual(){} function enqueueChatJob(){} function processChatQueue(){} "
                "function queueChatText(){} function waitForQualityRerunFrame(){} "
                "function captureCanvasDataUrl(){} function ensureRerunCaptureBridge(){} "
                "function pickBestIframeCanvas(){} function sampleFrameStats(){} "
                "function ensureLeIsaacTab(){} function removeLeIsaacTab(){} "
                "function unavailableLeIsaacStatus(){} function refreshLeIsaacCapability(){} "
                "function connectLeIsaac(){} ensureLeIsaacTab(leisaacCapability); "
                "/api/leisaac/status /api/leisaac/select /api/leisaac/bundles/reset "
                "/api/leisaac/client/index.js /api/leisaac/signal "
                "LeIsaac-SO101-LiftCube-v0 "
                "function openFullChatTab(){} "
                'function refreshTenantResources(){ fetch("/api/resources"); } '
                "do not prefetch .rrd bytes; skipUserAppend; Describe this — capturing; "
                "async function loadArtifact(payload){ await swapRerunRecordingInPlace(); } "
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
    import sys as _sys

    expected_py = _sys.executable or "python3"
    assert workflow_status_timeouts == [30.0]
    assert calls == [
        [
            expected_py,
            "-m",
            "pytest",
            "npa/tests/smoke/test_agent_smoke.py",
            "npa/tests/smoke/test_agent_chat_smoke.py",
            "-q",
        ],
        [
            expected_py,
            "-m",
            "pytest",
            "npa/tests/cli/test_agent.py",
            "npa/tests/cli/test_agent_workflow.py",
            "-q",
        ],
        [expected_py, "-m", "pytest", "npa/tests/e2e/test_agent_live.py", "-q"],
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

    source = _agent_source()
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


@requires_node
def test_bootstrap_emitted_ui_script_is_valid_javascript(monkeypatch) -> None:
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

        def upload_private_text(self, content: str, remote_path: str) -> None:
            if "npa-agent-bootstrap" in remote_path:
                captured["setup_script"] = content

        def run_or_raise(self, _command: str, **_kwargs) -> None:
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
        public_https=True,
    )

    setup_script = captured["setup_script"]
    assert "--ws-per-message-deflate false" in setup_script
    assert "--no-ws-per-message-deflate" not in setup_script
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

    source = _agent_source()
    assert "sim-viz/recordings" in source
    assert "available .rrd recording" in source


def test_agent_dry_run_counts_only_its_exact_healthy_project_record() -> None:
    source = _agent_source()

    assert (
        'runtime_agent_name = str(os.environ.get("NPA_AGENT_NAME") or "agent")'
        in source
    )
    assert "record = agents.get(runtime_agent_name)" in source
    assert "project_id != runtime_project_id" in source
    assert (
        "any("
        not in source.split("def _configured_healthy_agent_exists", 1)[1].split(
            "def _agent_command_env", 1
        )[0]
    )


def test_bootstrap_uses_unique_remote_setup_script_path() -> None:

    source = _agent_source()
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

    source = _agent_source()
    assert "pip install fastapi uvicorn httpx pyyaml boto3" in source


def test_bootstrap_installs_nebius_cli_and_sa_profile() -> None:

    source = _agent_source()
    assert "storage.eu-north1.nebius.cloud/cli/install.sh" in source
    assert "--token-file /mnt/cloud-metadata/token" in source
    assert 'nebius_profile = "cursor-sa"' in source
    assert "--profile {nebius_profile}" in source
    assert (
        '"$NEBIUS_BIN" --profile {nebius_profile} iam get-access-token >/dev/null'
        in source
    )
    assert 'sudo -H "$NEBIUS_BIN" profile create' in source
    assert (
        'sudo -H "$NEBIUS_BIN" --profile {nebius_profile} iam get-access-token'
        in source
    )
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

    source = _agent_source()
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
        def upload_private_text(self, content: str, _remote_path: str) -> None:
            commands.append(content)

        def run_or_raise(self, command: str, **_kwargs) -> None:
            commands.append(command)

        def run(self, _command: str) -> None:
            return None

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

    assert len(commands) == 2
    env_text = commands[0]
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
    assert 'iam project list --parent-id "$expected_tenant" --all' in source
    assert 'iam project get --id "$expected_project"' in source
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

    source = _agent_source()
    assert "_AGENT_CHAT_EMBED" in source
    assert ".replace(_AGENT_CHAT_EMBED, agent_chat_source)" in source
    raw = agent_module._embedded_agent_chat_source()
    assert '"onboard_solution"' in raw
    assert "{0,140}" in raw
    rendered = source.split("_AGENT_CHAT_EMBED = ", 1)[0]  # sanity: module loads
    assert rendered


def test_bootstrap_embeds_skill_context_and_api_accounting() -> None:

    source = _agent_source()
    assert "_resolve_skill_context" in source
    assert "_skill_index_candidates" in source
    assert "apis_suggested" in source
    assert "skills_used" in source
    assert "_dedupe(apis_used)" in source


def test_bootstrap_embeds_scoped_state_s3_persistence() -> None:

    source = _agent_source()
    assert "_state_s3_key" in source
    assert "NPA_AGENT_STATE_S3_PREFIX" in source
    assert "NPA_AGENT_SESSION_SCOPE" in source
    assert "_save_state_to_s3" in source
    assert "_load_state_from_s3" in source


def test_bootstrap_embeds_provider_resilience_fallback() -> None:

    source = _agent_source()
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

    source = _agent_source()
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
    monkeypatch.setattr("npa.cli.agent._bootstrap_agent_stack", lambda **k: None)
    monkeypatch.setattr("npa.cli.agent.ensure_ingress", lambda **k: None)
    monkeypatch.setattr(
        "npa.cli.agent.remove_npa_ingress_for_instance_ports",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "npa.cli.agent._store_agent_record",
        lambda project, name, rec: captured.update(rec),
    )

    # Satisfy the fail-fast deploy prerequisites (terraform + SSH key pair).
    (tmp_path / "id_ed25519.pub").write_text(
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAABAgMEBQYHCAkKCwwNDg8QERITFBUWFxgZGhscHR4f test\n"
    )
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

    (tmp_path / "id_ed25519.pub").write_text(
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAABAgMEBQYHCAkKCwwNDg8QERITFBUWFxgZGhscHR4f test\n"
    )
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
    assert "[PASS] cloud_init_yaml" in result.output
    assert "[PASS] token_factory" in result.output


def test_agent_hard_prereqs_fail_closed_on_provider_lock_error(
    monkeypatch, tmp_path
) -> None:
    from npa.terraform_lock import TerraformLockError

    public_key = tmp_path / "id_ed25519.pub"
    public_key.write_text(
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAABAgMEBQYHCAkKCwwNDg8QERITFBUWFxgZGhscHR4f test\n",
        encoding="utf-8",
    )
    (tmp_path / "id_ed25519").write_text("private\n", encoding="utf-8")
    monkeypatch.setenv("NPA_TERRAFORM_BIN", "/usr/bin/terraform")

    def reject_lock(_terraform_dir):
        raise TerraformLockError("provider lock does not cover test platform")

    monkeypatch.setattr("npa.terraform_lock.validate_provider_lock", reject_lock)
    results = agent_module._agent_hard_prereq_results(str(public_key))

    terraform = next(result for result in results if result.name == "terraform")
    assert terraform.status == "FAIL"
    assert "provider-lock compatibility failed" in terraform.summary
    assert terraform.remedy == "provider lock does not cover test platform"


def test_agent_preflight_invokes_exact_deploy_storage_decision(
    monkeypatch, tmp_path
) -> None:
    from npa.cli import agent as agent_module

    (tmp_path / "id_ed25519.pub").write_text(
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAABAgMEBQYHCAkKCwwNDg8QERITFBUWFxgZGhscHR4f test\n"
    )
    (tmp_path / "id_ed25519").write_text("priv\n")
    monkeypatch.setenv("NPA_TERRAFORM_BIN", "/usr/bin/terraform")
    monkeypatch.setattr(
        agent_module, "_resolve_deploy_llm_credentials", lambda: ("tf-key", "m")
    )
    calls: list[dict] = []

    def _resolve(**kwargs):
        calls.append(dict(kwargs))
        return {"s3_bucket": "configured-bucket"}

    monkeypatch.setattr(agent_module, "_resolve_deploy_storage_credentials", _resolve)

    result = runner.invoke(
        app,
        [
            "preflight",
            "--skip-nebius",
            "--project",
            "dev",
            "--ssh-public-key-path",
            str(tmp_path / "id_ed25519.pub"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "region": "",
            "project_alias": "dev",
            "emit_status": False,
        }
    ]
    assert "Deployment credential path selected" in result.output


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

    (tmp_path / "id_ed25519.pub").write_text(
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAABAgMEBQYHCAkKCwwNDg8QERITFBUWFxgZGhscHR4f test\n"
    )
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
        "cloud_init_yaml",
        "ssh_private_key",
        "ssh_egress",
        "writable_s3",
        "token_factory",
    }


def test_agent_preflight_rejects_private_key_passed_as_public_path(
    monkeypatch, tmp_path
) -> None:
    from npa.cli import agent as agent_module

    private = tmp_path / "id_ed25519"
    private.write_text(
        "-----BEGIN OPENSSH PRIVATE KEY-----\nprivate-material\n"
        "-----END OPENSSH PRIVATE KEY-----\n"
    )
    (tmp_path / "id_ed25519.pub").write_text(
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAABAgMEBQYHCAkKCwwNDg8QERITFBUWFxgZGhscHR4f test\n"
    )
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
            str(private),
        ],
    )

    assert result.exit_code == 1, result.output
    assert "[FAIL] ssh_public_key" in result.output
    assert "[FAIL] cloud_init_yaml" in result.output
    assert "public `.pub` file" in result.output
    assert "private-material" not in result.output


def test_agent_cloud_init_quotes_public_key_comments_before_yaml_parse() -> None:
    public_key = "ssh-ed25519 AAAA operator: recovery # comment"

    rendered = agent_module._render_agent_cloud_init("ubuntu", public_key)
    payload = yaml.safe_load(rendered)

    assert payload["users"][0]["ssh_authorized_keys"] == [public_key]


def test_legacy_agent_cloud_init_reproduces_multiline_private_key_yaml_failure() -> (
    None
):
    legacy = """#cloud-config

users:
  - name: ubuntu
    shell: /bin/bash
    sudo: ALL=(ALL) NOPASSWD:ALL
    ssh_authorized_keys:
      - -----BEGIN OPENSSH PRIVATE KEY-----
private-material
-----END OPENSSH PRIVATE KEY-----
"""

    with pytest.raises(yaml.YAMLError):
        yaml.safe_load(legacy)


def test_agent_preflight_fails_when_storage_write_probe_is_forbidden(
    monkeypatch, tmp_path
) -> None:
    from npa.cli import agent as agent_module
    from npa.clients import storage_validation
    from npa.clients.storage_validation import StorageProbeResult

    (tmp_path / "id_ed25519.pub").write_text(
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAABAgMEBQYHCAkKCwwNDg8QERITFBUWFxgZGhscHR4f test\n"
    )
    (tmp_path / "id_ed25519").write_text("priv\n")
    monkeypatch.setenv("NPA_TERRAFORM_BIN", "/usr/bin/terraform")
    monkeypatch.setattr(
        agent_module, "_resolve_deploy_llm_credentials", lambda: ("tf-key", "m")
    )
    monkeypatch.setattr(
        storage_validation,
        "probe_storage_write",
        lambda **_kwargs: StorageProbeResult(
            False,
            "forbidden",
            "S3 write probe was forbidden; the configured access key lacks data-plane permission.",
        ),
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

    assert result.exit_code == 1
    assert "[FAIL] writable_s3" in result.output
    assert "summary:" in result.output
    assert "fail" in result.output


def test_agent_status_read_only_does_not_probe_storage(monkeypatch, tmp_path) -> None:
    from npa.cli import agent as agent_module
    from npa.clients import storage_validation

    monkeypatch.setattr(
        agent_module, "_resolve_project_alias", lambda value: value or "demo"
    )
    monkeypatch.setattr(
        agent_module,
        "_agent_record",
        lambda *_args: {
            "agent_url": "https://agent/",
            "rerun_url": "https://agent/rerun/",
            "auth_secret_path": str(tmp_path / "auth"),
            "public_ip": "203.0.113.50",
        },
    )
    monkeypatch.setattr(agent_module, "_load_auth_secret", lambda _path: ("u", "p"))
    monkeypatch.setattr(agent_module, "_health", lambda *_args, **_kwargs: (True, 200))
    monkeypatch.setattr(
        agent_module,
        "_basic_auth_protects_endpoint",
        lambda *_args, **_kwargs: (True, 401),
    )
    monkeypatch.setattr(
        storage_validation,
        "probe_storage_write",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("read-only status must not write a storage probe")
        ),
    )

    result = runner.invoke(app, ["status", "--project", "demo", "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["health"] is True


def test_agent_preflight_nebius_fail(monkeypatch, tmp_path) -> None:
    from npa.cli import agent as agent_module

    (tmp_path / "id_ed25519.pub").write_text(
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAABAgMEBQYHCAkKCwwNDg8QERITFBUWFxgZGhscHR4f test\n"
    )
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

    monkeypatch.setenv("NPA_TERRAFORM_BIN", "/usr/bin/terraform")
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


def test_deploy_fails_fast_on_missing_terraform(monkeypatch, tmp_path) -> None:
    """Deploy aborts on a missing terraform binary BEFORE any cloud side effects."""
    from npa.cli import agent as agent_module
    from npa.cli.agent import deploy_cmd

    (tmp_path / "id_ed25519.pub").write_text(
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAABAgMEBQYHCAkKCwwNDg8QERITFBUWFxgZGhscHR4f test\n"
    )
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

    (tmp_path / "id_ed25519.pub").write_text(
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAABAgMEBQYHCAkKCwwNDg8QERITFBUWFxgZGhscHR4f test\n"
    )
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

    source = (
        Path(agent_module.__file__)
        .with_name("agent_stage_runtime.py")
        .read_text(encoding="utf-8")
    )
    assert "def _workflow_run_steps(" in source
    assert "/npa-workflow/manifest.json" in source
    assert '"workflow_steps": workflow_steps' in source
    # Enriched logs include the per-stage command lines.
    assert "workflow_steps = _workflow_run_steps(" in source


def test_artifact_file_transcodes_non_web_images_to_png() -> None:
    """Non-web images (.ppm sim camera frames, .bmp, .tiff) must be transcoded to
    PNG by the artifact file endpoint so they are viewable in the Rerun/Image panes."""
    source = _agent_source()
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


def test_coerce_cli_list_handles_unresolved_typer_option() -> None:
    """An unresolved typer.Option default (OptionInfo) must coerce to [].

    deploy_cmd is called programmatically (fresh-setup / `agent setup`); omitting
    a list option leaks an OptionInfo that once crashed `for item in tf_var`.
    """
    import typer

    from npa.cli.agent import _coerce_cli_list

    unresolved = typer.Option([], "--tf-var")
    assert type(unresolved).__name__ == "OptionInfo"  # guard the precondition
    result = _coerce_cli_list(unresolved)
    assert result == []
    list(result)  # must be iterable (the original crash site)


def test_coerce_cli_list_passthrough_and_none() -> None:
    from npa.cli.agent import _coerce_cli_list

    assert _coerce_cli_list(["a", "b"]) == ["a", "b"]
    assert _coerce_cli_list(("x",)) == ["x"]
    assert _coerce_cli_list(None) == []


def test_agent_setup_picks_configured_project(monkeypatch, tmp_path) -> None:
    """`npa agent setup` resolves project_id/tenant_id/region from config, no typing."""
    import yaml

    from npa.clients import config as config_module

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "default_project": "prod",
                "projects": {
                    "prod": {
                        "project_id": "project-prod",
                        "tenant_id": "tenant-a",
                        "region": "eu-north1",
                    },
                    "dev": {
                        "project_id": "project-dev",
                        "tenant_id": "tenant-a",
                        "region": "us-central1",
                    },
                },
            }
        )
    )
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)

    key_file = tmp_path / "id_ed25519.pub"
    key_file.write_text(
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAABAgMEBQYHCAkKCwwNDg8QERITFBUWFxgZGhscHR4f test\n"
    )

    captured: dict = {}

    def _fake_fresh_setup(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("npa.cli.agent.fresh_setup_cmd", _fake_fresh_setup)

    # Explicit --project resolves tenant/project/region from config (no typing).
    result = runner.invoke(
        app,
        ["setup", "--project", "dev", "--ssh-public-key-path", str(key_file)],
    )

    assert result.exit_code == 0, result.output
    assert captured["project"] == "dev"
    assert captured["project_id"] == "project-dev"
    assert captured["tenant_id"] == "tenant-a"
    assert captured["region"] == "us-central1"

    # Interactive: pressing Enter accepts the default_project (prod).
    captured.clear()
    result = runner.invoke(
        app,
        ["setup", "--ssh-public-key-path", str(key_file)],
        input="\n",
    )
    assert result.exit_code == 0, result.output
    assert captured["project"] == "prod"
    assert captured["project_id"] == "project-prod"


def _write_agent_setup_config(tmp_path, monkeypatch):
    """Configure one project alias and return its ssh public-key path."""
    import yaml

    from npa.clients import config as config_module

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "default_project": "dev",
                "projects": {
                    "dev": {
                        "project_id": "project-dev",
                        "tenant_id": "tenant-a",
                        "region": "us-central1",
                    }
                },
            }
        )
    )
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    key_file = tmp_path / "id_ed25519.pub"
    key_file.write_text(
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAABAgMEBQYHCAkKCwwNDg8QERITFBUWFxgZGhscHR4f test\n"
    )
    (tmp_path / "id_ed25519").write_text("-----BEGIN OPENSSH PRIVATE KEY-----\n")
    monkeypatch.setenv("NPA_TERRAFORM_BIN", "/usr/bin/terraform")
    return key_file


def test_agent_setup_passes_concrete_defaults_to_deploy(monkeypatch, tmp_path) -> None:
    """`agent setup` -> `fresh-setup` -> `deploy` must not leak Typer OptionInfo.

    Regression: `setup_cmd` calls `fresh_setup_cmd` as a plain function, so every
    omitted option used to arrive as a `typer.models.OptionInfo` sentinel and
    flow into Terraform vars / nginx ports / boolean flags.
    """
    from npa.cli.agent import (
        DEFAULT_AGENT_PORT,
        DEFAULT_BACKEND_PORT,
        DEFAULT_RERUN_PORT,
    )

    key_file = _write_agent_setup_config(tmp_path, monkeypatch)

    captured: dict = {}
    monkeypatch.setattr(
        "npa.cli.agent.deploy_cmd", lambda **kwargs: captured.update(kwargs)
    )

    result = runner.invoke(
        app,
        ["setup", "--project", "dev", "--ssh-public-key-path", str(key_file)],
    )
    assert result.exit_code == 0, result.output

    leaked = {
        key: value
        for key, value in captured.items()
        if type(value).__name__ in {"OptionInfo", "ArgumentInfo"}
    }
    assert leaked == {}, f"unresolved Typer defaults reached deploy: {sorted(leaked)}"

    assert captured["ssh_user"] == "ubuntu"
    assert captured["agent_port"] == DEFAULT_AGENT_PORT
    assert captured["backend_port"] == DEFAULT_BACKEND_PORT
    assert captured["rerun_port"] == DEFAULT_RERUN_PORT
    assert captured["tf_var"] == []
    assert captured["llm_models"] == []
    assert captured["no_public_https"] is False


def test_agent_fresh_setup_forwards_agent_only(monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr("npa.cli.agent._agent_record", lambda *args, **kwargs: {})
    monkeypatch.setattr("npa.cli.agent._store_project_environment", lambda **kwargs: None)
    monkeypatch.setattr(
        "npa.cli.agent.deploy_cmd", lambda **kwargs: captured.update(kwargs)
    )

    result = runner.invoke(
        app,
        [
            "fresh-setup",
            "--project",
            "dev",
            "--project-id",
            "project-dev",
            "--tenant-id",
            "tenant-a",
            "--region",
            "us-central1",
            "--agent-only",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["agent_only"] is True


def _stub_agent_deploy_cloud_calls(
    monkeypatch, tmp_path, *, credential_sentinels: bool = False
):
    """Stub every cloud side effect in `deploy_cmd`; return captured calls."""
    calls: dict = {}
    access_key = (
        "NPA_PR218_ACCESS_SENTINEL_DO_NOT_PERSIST"
        if credential_sentinels
        else "ak-agent"
    )
    secret_key = (
        "NPA_PR218_SECRET_SENTINEL_DO_NOT_PERSIST"
        if credential_sentinels
        else "sk-agent"
    )
    creds = {
        "service_account_id": "sa-agent",
        "nebius_api_key": access_key,
        "nebius_secret_key": secret_key,
        "s3_bucket": "npa-agent-state",
        "s3_endpoint": "https://storage.us-central1.nebius.cloud",
    }

    def _apply(**kwargs):
        calls["merged_vars"] = dict(kwargs["merged_vars"])
        return {
            "vm_ip": "203.0.113.50",
            "instance_id": "instance-agent",
            "ssh_key_path": str(tmp_path / "id_ed25519"),
        }

    def _bootstrap_environment(*_args, **kwargs):
        calls["bootstrap_environment_kwargs"] = dict(kwargs)
        return creds

    monkeypatch.setattr(
        "npa.clients.nebius.bootstrap_agent_environment", _bootstrap_environment
    )
    monkeypatch.setattr("npa.clients.nebius.get_iam_token", lambda: "iam-token")
    monkeypatch.setattr(
        "npa.clients.nebius.get_project_region", lambda _pid: "us-central1"
    )
    monkeypatch.setattr(
        "npa.cli.agent._resolve_deploy_storage_credentials", lambda **k: creds
    )
    monkeypatch.setattr(
        "npa.cli.agent._agent_check_public_ip_quota", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "npa.cli.agent._ensure_terraform_state_bucket", lambda **k: None
    )
    monkeypatch.setattr("npa.cli.agent._apply_agent_terraform", _apply)
    monkeypatch.setattr("npa.cli.agent._is_routable_public_ip", lambda _ip: True)
    monkeypatch.setattr(
        "npa.cli.agent._write_auth_secret", lambda **k: tmp_path / "auth.env"
    )
    monkeypatch.setattr(
        "npa.cli.agent._resolve_deploy_llm_credentials", lambda: ("tf-key", "model-a")
    )
    monkeypatch.setattr(
        "npa.cli.agent._bootstrap_agent_stack",
        lambda **kwargs: calls.__setitem__("bootstrap", dict(kwargs)),
    )
    monkeypatch.setattr("npa.cli.agent.ensure_ingress", lambda **k: None)
    monkeypatch.setattr(
        "npa.cli.agent.remove_npa_ingress_for_instance_ports",
        lambda *_args, **_kwargs: [],
    )
    return calls


def test_agent_setup_renders_string_terraform_vars(monkeypatch, tmp_path) -> None:
    """The full `agent setup` chain must hand Terraform real strings.

    Regression: `server_port` / `ssh_user` / `extra_ingress_ports` used to be
    rendered from `OptionInfo` objects, producing literal
    "<typer.models.OptionInfo object at 0x...>" Terraform var values.
    """
    key_file = _write_agent_setup_config(tmp_path, monkeypatch)
    calls = _stub_agent_deploy_cloud_calls(monkeypatch, tmp_path)

    result = runner.invoke(
        app,
        ["setup", "--project", "dev", "--ssh-public-key-path", str(key_file)],
    )
    assert result.exit_code == 0, result.output

    merged_vars = calls["merged_vars"]
    assert merged_vars["server_port"] == "8088"
    assert merged_vars["ssh_user"] == "ubuntu"
    assert merged_vars["extra_ingress_ports"] == "[443,9090]"
    assert not any("OptionInfo" in str(value) for value in merged_vars.values()), (
        f"OptionInfo leaked into terraform vars: {merged_vars}"
    )
    assert calls["bootstrap_environment_kwargs"]["reuse_storage_credentials"] == {
        "service_account_id": "sa-agent",
        "nebius_api_key": "ak-agent",
        "nebius_secret_key": "sk-agent",
        "s3_bucket": "npa-agent-state",
        "s3_endpoint": "https://storage.us-central1.nebius.cloud",
    }


def test_agent_deploy_keeps_s3_sentinels_out_of_terraform_and_agent_record(
    monkeypatch, tmp_path
) -> None:
    import yaml

    from npa.cli import agent as agent_module
    from npa.clients import config as config_module

    key_file = _write_agent_setup_config(tmp_path, monkeypatch)
    calls = _stub_agent_deploy_cloud_calls(
        monkeypatch, tmp_path, credential_sentinels=True
    )

    result = runner.invoke(
        app,
        ["setup", "--project", "dev", "--ssh-public-key-path", str(key_file)],
    )

    assert result.exit_code == 0, result.output
    terraform_payload = json.dumps(calls["merged_vars"], sort_keys=True)
    access_sentinel = "NPA_PR218_ACCESS_SENTINEL_DO_NOT_PERSIST"
    secret_sentinel = "NPA_PR218_SECRET_SENTINEL_DO_NOT_PERSIST"
    assert access_sentinel not in terraform_payload
    assert secret_sentinel not in terraform_payload
    assert calls["bootstrap"]["s3_access_key"] == access_sentinel
    assert calls["bootstrap"]["s3_secret_key"] == secret_sentinel
    saved = yaml.safe_load(config_module.CONFIG_PATH.read_text(encoding="utf-8"))
    agent_record = saved["projects"]["dev"]["agents"]["agent"]
    assert "credentials" not in agent_record
    assert access_sentinel not in json.dumps(agent_record)
    assert secret_sentinel not in json.dumps(agent_record)

    template = (
        Path(agent_module.provisioner.__file__).parent
        / "terraform"
        / "cloud_init.yaml.tpl"
    ).read_text(encoding="utf-8")
    protected_write_files = template.split('%{ if workbench_type != "agent" ~}', 1)[
        1
    ].split("%{ endif ~}", 1)[0]
    assert "write_files:" in protected_write_files
    assert "${aws_access_key}" in protected_write_files
    assert "${aws_secret_key}" in protected_write_files


def test_agent_setup_keeps_public_https_enabled(monkeypatch, tmp_path) -> None:
    """`--no-public-https` defaults to False, so `agent setup` keeps HTTPS on.

    Regression: the unresolved `OptionInfo(False)` sentinel is *truthy*, so
    `public_https = not no_public_https` silently evaluated to False and the
    agent deployed without HTTPS.
    """
    key_file = _write_agent_setup_config(tmp_path, monkeypatch)
    calls = _stub_agent_deploy_cloud_calls(monkeypatch, tmp_path)

    result = runner.invoke(
        app,
        ["setup", "--project", "dev", "--ssh-public-key-path", str(key_file)],
    )
    assert result.exit_code == 0, result.output
    assert calls["bootstrap"]["public_https"] is True
    assert calls["bootstrap"]["ssh_user"] == "ubuntu"
    assert calls["bootstrap"]["agent_port"] == 8088
    assert calls["bootstrap"]["backend_port"] == 8787
    assert calls["bootstrap"]["rerun_port"] == 9090


def test_agent_setup_requires_configured_projects(monkeypatch, tmp_path) -> None:
    """With no configured projects, `npa agent setup` points to `npa configure`."""
    from npa.clients import config as config_module

    monkeypatch.setattr(config_module, "CONFIG_PATH", tmp_path / "missing.yaml")
    result = runner.invoke(app, ["setup"])
    assert result.exit_code != 0
    assert "npa configure" in result.output


def _wait_for_cloud_init_body() -> str:
    """Return the real local-exec script Terraform quotes back in its error.

    The body echoes every diagnostic string the script can print, so a hint that
    matches against it classifies every failure the same way. Reading it from the
    shipped Terraform keeps the tests below honest.
    """
    from npa.deploy import provisioner as provisioner_module

    main_tf = (
        Path(provisioner_module.__file__).parent / "terraform" / "main.tf"
    ).read_text(encoding="utf-8")
    body = main_tf[main_tf.index('resource "null_resource" "wait_for_cloud_init"') :]
    return body[: body.index("\n    EOT")]


def _terraform_local_exec_error(output: str) -> str:
    return (
        "terraform apply failed (exit 1):\n"
        "Error: local-exec provisioner error\n"
        "  with null_resource.wait_for_cloud_init,\n"
        f"Error running command '{_wait_for_cloud_init_body()}': exit status 1. "
        f"Output: {output}"
    )


def test_agent_deploy_failure_hint_diagnoses_ssh_unreachable() -> None:
    """A wait_for_cloud_init SSH timeout gets a concise reachability diagnosis."""
    from npa.cli.agent import _agent_deploy_failure_hint

    detail = _terraform_local_exec_error(
        "Waiting for SSH on 203.0.113.50:22 (up to ~4 minutes, progress every 30s)...\n"
        "  still waiting (attempt 4/30): tcp/22 has not opened from this host yet\n"
        "ERROR: SSH to ubuntu@203.0.113.50:22 never succeeded within the boot window.\n"
        "tcp/22 on 203.0.113.50 never opened from this machine, so the VM's SSH port "
        "is unreachable from here."
    )
    hint = _agent_deploy_failure_hint(detail)
    assert "tcp/22 never opened" in hint
    assert "split-tunnel" in hint
    assert "authenticated" not in hint


def test_agent_deploy_failure_hint_separates_a_key_problem_from_the_network() -> None:
    """The wait now distinguishes a closed port from a refused key."""
    from npa.cli.agent import _agent_deploy_failure_hint

    detail = _terraform_local_exec_error(
        "Waiting for SSH on 203.0.113.50:22 (up to ~4 minutes, progress every 30s)...\n"
        "  still waiting (attempt 8/30): tcp/22 is open, SSH not ready yet\n"
        "ERROR: SSH to ubuntu@203.0.113.50:22 never succeeded within the boot window.\n"
        "tcp/22 on 203.0.113.50 opened, but SSH never authenticated, so this is the "
        "key or the sshd config rather than the network."
    )
    hint = _agent_deploy_failure_hint(detail)
    assert "never authenticated" in hint
    assert "--ssh-public-key-path" in hint
    assert "VPN" not in hint


def test_agent_deploy_failure_hint_diagnoses_cloud_init_error() -> None:
    """A cloud-init runcmd failure is distinguished from an SSH timeout."""
    from npa.cli.agent import _agent_deploy_failure_hint

    detail = _terraform_local_exec_error(
        "Waiting for SSH on 203.0.113.50:22...\n"
        "Waiting for cloud-init boot-finished...\n"
        "Polling cloud-init status...\n"
        "cloud-init status: error\n"
        "ERROR: cloud-init finished with status 'error'; the VM bootstrap failed."
    )
    hint = _agent_deploy_failure_hint(detail)
    assert "cloud-init bootstrap failed" in hint
    assert "SSH never became reachable" not in hint


def test_agent_deploy_failure_hint_empty_for_unrelated_errors() -> None:
    from npa.cli.agent import _agent_deploy_failure_hint

    assert _agent_deploy_failure_hint("terraform apply failed: quota exceeded") == ""
    assert _agent_deploy_failure_hint("") == ""


def test_agent_whole_path_blocker_precedes_storage_and_terraform(
    monkeypatch: pytest.MonkeyPatch, mocker, tmp_path: Path
) -> None:
    from npa.cli.agent import deploy_cmd
    from npa.provisioning_preflight import PreflightBlockedError

    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(tmp_path / "operations"))
    monkeypatch.setattr(
        "npa.cli.agent.resolve_environment",
        lambda *args, **kwargs: SimpleNamespace(
            project_id="project-x", tenant_id="tenant-x", region="us-central1"
        ),
    )
    monkeypatch.setattr(
        "npa.clients.nebius.get_project_region", lambda _pid: "us-central1"
    )
    monkeypatch.setattr("npa.cli.agent._agent_record", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        "npa.cli.agent._agent_check_whole_path_capacity",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            PreflightBlockedError("compute.disk.count shortfall=1")
        ),
    )
    storage = mocker.patch("npa.cli.agent._agent_storage_result")
    bootstrap = mocker.patch("npa.clients.nebius.bootstrap_agent_environment")
    terraform = mocker.patch("npa.cli.agent._apply_agent_terraform")

    with pytest.raises(Exit):
        deploy_cmd(
            project="fresh",
            name="agent",
            project_id="project-x",
            tenant_id="tenant-x",
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

    storage.assert_not_called()
    bootstrap.assert_not_called()
    terraform.assert_not_called()
    [journal] = (tmp_path / "operations").glob("*/journal.json")
    assert json.loads(journal.read_text(encoding="utf-8"))["phase"] == "rolled-back"


def test_agent_only_deploy_omits_paidf_capacity_reservation(
    monkeypatch: pytest.MonkeyPatch, mocker, tmp_path: Path
) -> None:
    """The explicit lifecycle-validation mode still gates the agent VM itself."""
    from npa.cli.agent import deploy_cmd

    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(tmp_path / "operations"))
    monkeypatch.setattr(
        "npa.cli.agent.resolve_environment",
        lambda *args, **kwargs: SimpleNamespace(
            project_id="project-x", tenant_id="tenant-x", region="us-central1"
        ),
    )
    monkeypatch.setattr(
        "npa.clients.nebius.get_project_region", lambda _pid: "us-central1"
    )
    monkeypatch.setattr("npa.cli.agent._agent_record", lambda *args, **kwargs: {})
    capacity = mocker.patch("npa.cli.agent._agent_check_whole_path_capacity")
    mocker.patch(
        "npa.cli.agent._agent_hard_prereq_results",
        return_value=[],
    )
    mocker.patch(
        "npa.cli.agent._agent_storage_result",
        return_value=SimpleNamespace(status="PASS"),
    )
    mocker.patch(
        "npa.cli.agent._resolve_deploy_llm_credentials", return_value=("k", "m")
    )
    mocker.patch(
        "npa.clients.nebius.bootstrap_agent_environment",
        return_value={"iam_token": "token-from-bootstrap"},
    )
    mocker.patch(
        "npa.cli.agent._apply_agent_terraform",
        side_effect=RuntimeError("stop after preflight"),
    )

    with pytest.raises(RuntimeError, match="stop after preflight"):
        deploy_cmd(
            project="fresh",
            name="agent",
            project_id="project-x",
            tenant_id="tenant-x",
            region="us-central1",
            ssh_user="ubuntu",
            ssh_public_key_path=str(tmp_path / "id_ed25519.pub"),
            tf_var=[],
            agent_only=True,
            agent_port=8088,
            backend_port=8787,
            rerun_port=9090,
            llm_model="model-a",
            llm_models=[],
            no_public_https=False,
        )

    assert capacity.call_args.kwargs["include_paidf"] is False
    [journal] = (tmp_path / "operations").glob("*/journal.json")
    commands = json.loads(journal.read_text())["recovery_commands"]
    assert "--agent-only" in commands["resume_argv"]


def test_agent_check_public_ip_quota_fails_when_exhausted(monkeypatch) -> None:
    """Deploy aborts early with guidance when the region's public-IP quota is full."""
    from npa.cli.agent import _agent_check_public_ip_quota
    from npa.clients import nebius as nebius_module

    monkeypatch.setattr(nebius_module, "get_project_region", lambda _pid: "us-central1")
    monkeypatch.setattr(
        nebius_module, "get_public_ipv4_quota", lambda _tid, _region: (10, 10)
    )

    with pytest.raises(Exit):
        _agent_check_public_ip_quota("project-x", "tenant-x", "eu-north1")


def test_agent_check_public_ip_quota_passes_with_headroom(monkeypatch) -> None:
    from npa.cli.agent import _agent_check_public_ip_quota
    from npa.clients import nebius as nebius_module

    monkeypatch.setattr(nebius_module, "get_project_region", lambda _pid: "uk-south1")
    monkeypatch.setattr(
        nebius_module, "get_public_ipv4_quota", lambda _tid, _region: (0, 3)
    )

    # Must not raise.
    _agent_check_public_ip_quota("project-x", "tenant-x", "uk-south1")


def test_agent_check_public_ip_quota_noop_when_quota_unknown(monkeypatch) -> None:
    """An unreadable quota never blocks a deploy."""
    from npa.cli.agent import _agent_check_public_ip_quota
    from npa.clients import nebius as nebius_module

    monkeypatch.setattr(nebius_module, "get_project_region", lambda _pid: "us-central1")
    monkeypatch.setattr(
        nebius_module, "get_public_ipv4_quota", lambda _tid, _region: (None, None)
    )

    # Must not raise even though the region resolved.
    _agent_check_public_ip_quota("project-x", "tenant-x", "eu-north1")


def test_agent_check_public_ip_quota_noop_when_region_unresolved(monkeypatch) -> None:
    from npa.cli.agent import _agent_check_public_ip_quota
    from npa.clients import nebius as nebius_module

    monkeypatch.setattr(nebius_module, "get_project_region", lambda _pid: "")

    def _boom(_tid, _region):  # pragma: no cover - must not be reached
        raise AssertionError("quota lookup should be skipped when region is unknown")

    monkeypatch.setattr(nebius_module, "get_public_ipv4_quota", _boom)

    # No region and no fallback -> skip entirely.
    _agent_check_public_ip_quota("project-x", "tenant-x", "")


def test_agent_public_ip_quota_result_fails_when_exhausted(monkeypatch) -> None:
    from npa.cli.agent import _agent_public_ip_quota_result
    from npa.clients import config as config_module
    from npa.clients import nebius as nebius_module

    monkeypatch.setattr(
        config_module,
        "list_projects",
        lambda: {
            "p": {
                "project_id": "project-x",
                "tenant_id": "tenant-x",
                "region": "us-central1",
            }
        },
    )
    monkeypatch.setattr(config_module, "default_project_name", lambda: "p")
    monkeypatch.setattr(nebius_module, "get_project_region", lambda _pid: "us-central1")
    monkeypatch.setattr(nebius_module, "get_public_ipv4_quota", lambda _t, _r: (10, 10))

    result = _agent_public_ip_quota_result()
    assert result.status == "FAIL"
    assert "exhausted" in result.summary.lower()


def test_agent_public_ip_quota_result_passes_with_headroom(monkeypatch) -> None:
    from npa.cli.agent import _agent_public_ip_quota_result
    from npa.clients import config as config_module
    from npa.clients import nebius as nebius_module

    monkeypatch.setattr(
        config_module,
        "list_projects",
        lambda: {
            "p": {
                "project_id": "project-x",
                "tenant_id": "tenant-x",
                "region": "uk-south1",
            }
        },
    )
    monkeypatch.setattr(config_module, "default_project_name", lambda: "p")
    monkeypatch.setattr(nebius_module, "get_project_region", lambda _pid: "uk-south1")
    monkeypatch.setattr(nebius_module, "get_public_ipv4_quota", lambda _t, _r: (0, 3))

    result = _agent_public_ip_quota_result()
    assert result.status == "PASS"


def test_agent_public_ip_quota_result_skips_without_project(monkeypatch) -> None:
    from npa.cli.agent import _agent_public_ip_quota_result
    from npa.clients import config as config_module

    monkeypatch.setattr(config_module, "list_projects", lambda: {})
    monkeypatch.setattr(config_module, "default_project_name", lambda: "")

    result = _agent_public_ip_quota_result()
    assert result.status == "PASS"
    assert "skipped" in result.summary.lower()


def test_agent_check_compute_instance_quota_fails_when_exhausted(monkeypatch) -> None:
    """Deploy aborts early when the region's compute.instance.count is full.

    Regression: preflight/deploy checked only public IPv4, so a `limit 0`
    compute quota let the disk/network/SG create before the VM create failed.
    """
    from npa.cli.agent import _agent_check_compute_instance_quota
    from npa.clients import nebius as nebius_module

    monkeypatch.setattr(nebius_module, "get_project_region", lambda _pid: "us-central1")
    monkeypatch.setattr(
        nebius_module, "get_compute_instance_quota", lambda _t, _r: (0, 0)
    )

    with pytest.raises(Exit):
        _agent_check_compute_instance_quota("project-x", "tenant-x", "eu-north1")


def test_agent_check_compute_instance_quota_skips_a_redeploy(monkeypatch) -> None:
    """`agent_exists` (a re-deploy reusing the VM) never blocks on the quota."""
    from npa.cli.agent import _agent_check_compute_instance_quota
    from npa.clients import nebius as nebius_module

    def _boom(*_a, **_k):  # pragma: no cover - must not be reached
        raise AssertionError("quota lookup should be skipped for an existing agent")

    monkeypatch.setattr(nebius_module, "get_project_region", _boom)

    _agent_check_compute_instance_quota(
        "project-x", "tenant-x", "eu-north1", agent_exists=True
    )


def test_agent_check_compute_instance_quota_noop_when_unreadable(monkeypatch) -> None:
    from npa.cli.agent import _agent_check_compute_instance_quota
    from npa.clients import nebius as nebius_module

    monkeypatch.setattr(nebius_module, "get_project_region", lambda _pid: "us-central1")
    monkeypatch.setattr(
        nebius_module, "get_compute_instance_quota", lambda _t, _r: (None, None)
    )

    _agent_check_compute_instance_quota("project-x", "tenant-x", "eu-north1")


def test_agent_compute_instance_quota_result_fails_on_limit_zero(monkeypatch) -> None:
    from npa.cli.agent import _agent_compute_instance_quota_result
    from npa.clients import config as config_module
    from npa.clients import nebius as nebius_module

    monkeypatch.setattr(
        config_module,
        "list_projects",
        lambda: {
            "p": {
                "project_id": "project-x",
                "tenant_id": "tenant-x",
                "region": "us-central1",
            }
        },
    )
    monkeypatch.setattr(config_module, "default_project_name", lambda: "p")
    monkeypatch.setattr(nebius_module, "get_project_region", lambda _pid: "us-central1")
    monkeypatch.setattr(
        nebius_module, "get_compute_instance_quota", lambda _t, _r: (0, 0)
    )

    result = _agent_compute_instance_quota_result()
    assert result.status == "FAIL"
    assert "compute instance quota" in result.summary.lower()


def test_agent_compute_instance_quota_result_passes_with_headroom(monkeypatch) -> None:
    from npa.cli.agent import _agent_compute_instance_quota_result
    from npa.clients import config as config_module
    from npa.clients import nebius as nebius_module

    monkeypatch.setattr(
        config_module,
        "list_projects",
        lambda: {
            "p": {
                "project_id": "project-x",
                "tenant_id": "tenant-x",
                "region": "uk-south1",
            }
        },
    )
    monkeypatch.setattr(config_module, "default_project_name", lambda: "p")
    monkeypatch.setattr(nebius_module, "get_project_region", lambda _pid: "uk-south1")
    monkeypatch.setattr(
        nebius_module, "get_compute_instance_quota", lambda _t, _r: (0, 3)
    )

    assert _agent_compute_instance_quota_result().status == "PASS"


def test_resolve_project_alias_prefers_explicit(monkeypatch) -> None:
    from npa.cli.agent import _resolve_project_alias

    assert _resolve_project_alias("myproj") == "myproj"


def test_resolve_project_alias_uses_configured_default(monkeypatch) -> None:
    from npa.cli.agent import _resolve_project_alias
    from npa.clients import config as config_module

    monkeypatch.setattr(config_module, "default_project_name", lambda: "workbench-poc")
    assert _resolve_project_alias("") == "workbench-poc"


def test_resolve_project_alias_falls_back_to_static_default(monkeypatch) -> None:
    from npa.cli.agent import _resolve_project_alias, DEFAULT_PROJECT_ALIAS
    from npa.clients import config as config_module

    monkeypatch.setattr(config_module, "default_project_name", lambda: "")
    assert _resolve_project_alias("") == DEFAULT_PROJECT_ALIAS


def test_remove_agent_record_drops_empty_agents_key(monkeypatch, tmp_path) -> None:
    import yaml
    from npa.cli.agent import _remove_agent_record
    from npa.clients import config as config_module

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "projects": {
                    "p": {
                        "project_id": "project-x",
                        "agents": {"agent": {"public_ip": "203.0.113.50"}},
                    }
                }
            }
        )
    )
    monkeypatch.setattr(config_module, "CONFIG_PATH", cfg)

    _remove_agent_record("p", "agent")

    data = yaml.safe_load(cfg.read_text())
    # The empty agents map is dropped entirely, but the project stanza stays.
    assert "agents" not in data["projects"]["p"]
    assert data["projects"]["p"]["project_id"] == "project-x"


def test_cleanup_agent_local_files_removes_auth_env(monkeypatch, tmp_path) -> None:
    from npa.cli import agent as agent_module

    monkeypatch.setattr(agent_module.Path, "home", staticmethod(lambda: tmp_path))
    agent_dir = tmp_path / ".npa" / "agents" / "p" / "agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "auth.env").write_text("AGENT_USER=npa\nAGENT_PASSWORD=x\n")

    agent_module._cleanup_agent_local_files("p", "agent")

    assert not agent_dir.exists()


def _owned_orphan_inventory() -> dict:
    return {
        "items": [
            {
                "metadata": {
                    "id": "instance-orphan",
                    "name": "agent-prod-agent",
                    "labels": {"npa-operation-id": "operation-a"},
                }
            }
        ]
    }


def test_orphan_delete_provider_rejection_is_unresolved(monkeypatch) -> None:
    from npa.cli import agent as agent_module
    from npa.clients import nebius as nebius_module
    from npa.deploy.provisioner import ProvisionerError

    monkeypatch.setattr(
        nebius_module, "_run_json", lambda *args, **kwargs: _owned_orphan_inventory()
    )
    monkeypatch.setattr(
        nebius_module,
        "_run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            nebius_module.NebiusError("provider rejected delete")
        ),
    )

    with pytest.raises(ProvisionerError, match="provider rejected delete"):
        agent_module._cleanup_orphan_agent_instances(
            "project-a", "agent-prod-agent", operation_id="operation-a"
        )


def test_orphan_delete_postcheck_still_present_is_not_reported_deleted(
    monkeypatch,
) -> None:
    from npa.cli import agent as agent_module
    from npa.clients import nebius as nebius_module
    from npa.deploy.provisioner import ProvisionerError

    monkeypatch.setattr(
        nebius_module, "_run_json", lambda *args, **kwargs: _owned_orphan_inventory()
    )
    monkeypatch.setattr(nebius_module, "_run", lambda *args, **kwargs: "")

    with pytest.raises(ProvisionerError, match="still reports it present"):
        agent_module._cleanup_orphan_agent_instances(
            "project-a", "agent-prod-agent", operation_id="operation-a"
        )


def test_orphan_delete_reports_only_verified_absence(monkeypatch, capsys) -> None:
    from npa.cli import agent as agent_module
    from npa.clients import nebius as nebius_module

    inventories = iter([_owned_orphan_inventory(), {"items": []}])
    monkeypatch.setattr(
        nebius_module, "_run_json", lambda *args, **kwargs: next(inventories)
    )
    monkeypatch.setattr(nebius_module, "_run", lambda *args, **kwargs: "")

    agent_module._cleanup_orphan_agent_instances(
        "project-a", "agent-prod-agent", operation_id="operation-a"
    )

    assert (
        "Verified deleted orphan agent instance instance-orphan"
        in capsys.readouterr().out
    )


def test_destroy_terraform_orphan_sweep_runs_after_tf_destroy(
    monkeypatch, tmp_path
) -> None:
    from types import SimpleNamespace
    from npa.cli import agent as agent_module

    calls: list[str] = []
    monkeypatch.setattr(
        agent_module,
        "_resolve_destroy_tf_vars",
        lambda p, n, r: {
            "nebius_region": "eu-north1",
            "instance_name": f"agent-{p}-{n}",
            "nebius_project_id": "project-x",
            "s3_session_token": "backend-only-session",
        },
    )
    monkeypatch.setattr(
        agent_module,
        "_cleanup_agent_ingress",
        lambda *_a, **_k: calls.append("ingress"),
    )
    monkeypatch.setattr(
        agent_module,
        "_cleanup_orphan_agent_instances",
        lambda *_a, **_k: calls.append("orphan"),
    )
    monkeypatch.setattr(
        agent_module,
        "resolve_terraform_state",
        lambda _p: SimpleNamespace(
            bucket="b", access_key="k", secret_key="s", endpoint="e"
        ),
    )
    monkeypatch.setattr(
        agent_module, "_agent_terraform_state_exists", lambda _p, _n: True
    )
    monkeypatch.setattr(
        agent_module.provisioner, "prepare_working_dir", lambda *a, **k: tmp_path
    )
    monkeypatch.setattr(agent_module.provisioner, "init", lambda *a, **k: None)
    monkeypatch.setattr(
        agent_module.provisioner, "destroy", lambda *a, **k: calls.append("tf_destroy")
    )

    agent_module._destroy_agent_terraform("p", "n", record={"instance_id": "i"})

    # Terraform owns the instance (destroy first); the by-name orphan sweep is a
    # post-destroy safety net, not a pre-Terraform delete of the managed VM.
    assert calls == ["ingress", "tf_destroy", "orphan"]


def test_destroy_terraform_no_state_refuses_unguarded_name_reclaim(monkeypatch) -> None:
    from types import SimpleNamespace
    from npa.cli import agent as agent_module
    from npa.deploy.provisioner import ProvisionerError

    calls: list[str] = []
    monkeypatch.setattr(
        agent_module,
        "_resolve_destroy_tf_vars",
        lambda p, n, r: {
            "nebius_region": "eu-north1",
            "instance_name": f"agent-{p}-{n}",
            "nebius_project_id": "project-x",
        },
    )
    monkeypatch.setattr(
        agent_module,
        "_cleanup_agent_ingress",
        lambda *_a, **_k: calls.append("ingress"),
    )
    monkeypatch.setattr(
        agent_module,
        "_cleanup_orphan_agent_instances",
        lambda *_a, **_k: calls.append("orphan"),
    )
    monkeypatch.setattr(
        agent_module,
        "resolve_terraform_state",
        lambda _p: SimpleNamespace(
            bucket="", access_key="", secret_key="", endpoint=""
        ),
    )
    monkeypatch.setattr(
        agent_module, "_agent_terraform_state_exists", lambda _p, _n: False
    )

    def _boom(*_a, **_k):  # pragma: no cover - must not run without state
        raise AssertionError("terraform destroy must not run without state")

    monkeypatch.setattr(agent_module.provisioner, "destroy", _boom)

    with pytest.raises(ProvisionerError, match="refusing an unguarded name-based"):
        agent_module._destroy_agent_terraform("p", "n", record=None)

    assert calls == ["ingress"]


def test_destroy_recovers_empty_journal_backend_from_exact_project_credentials(
    monkeypatch, tmp_path
) -> None:
    from types import SimpleNamespace

    from npa.cli import agent as agent_module
    from npa.provisioning_journal import ProvisioningOperation

    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(tmp_path / "operations"))
    operation = ProvisioningOperation.prepare(
        command="npa agent deploy",
        project_alias="prod",
        project_id="project-x",
        tenant_id="tenant-x",
        region="us-central1",
        resource_type="agent",
        requested_name="agent",
        ownership_source="test",
        resume_command="npa agent deploy --project prod --name agent",
    )
    operation.transition("recovery-required")
    monkeypatch.setattr(
        "npa.clients.credentials.load_credentials",
        lambda **_kwargs: SimpleNamespace(
            s3_project_id="project-x",
            s3_bucket="s3://project-state",
            s3_access_key_id="access",
            s3_secret_access_key="secret",
            s3_endpoint="https://storage.us-central1.nebius.cloud",
        ),
    )
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        agent_module,
        "_resolve_destroy_tf_vars",
        lambda _p, _n, _r, *, backend_override: {
            "nebius_region": "us-central1",
            "instance_name": "agent-prod-agent",
            "nebius_project_id": "project-x",
            "s3_session_token": "",
        },
    )
    monkeypatch.setattr(agent_module, "_cleanup_agent_ingress", lambda *_a: None)
    monkeypatch.setattr(
        agent_module,
        "_cleanup_orphan_agent_instances",
        lambda *_a, **kwargs: observed.setdefault(
            "operation_id", kwargs["operation_id"]
        ),
    )
    monkeypatch.setattr(
        agent_module,
        "resolve_terraform_state",
        lambda _p: SimpleNamespace(
            bucket="", access_key="", secret_key="", endpoint=""
        ),
    )
    monkeypatch.setattr(
        agent_module, "_agent_terraform_state_exists", lambda _p, _n: False
    )
    monkeypatch.setattr(
        agent_module.provisioner,
        "prepare_working_dir",
        lambda *_a, **kwargs: (
            observed.setdefault("bucket", kwargs["bucket"]) and tmp_path
        ),
    )
    monkeypatch.setattr(
        agent_module.provisioner,
        "init",
        lambda **kwargs: observed.setdefault("backend", kwargs["backend_config"]),
    )
    monkeypatch.setattr(agent_module.provisioner, "state_list", lambda _path: [])
    monkeypatch.setattr(
        agent_module.provisioner,
        "destroy",
        lambda **_kwargs: observed.setdefault("destroyed", True),
    )

    agent_module._destroy_agent_terraform(
        "prod",
        "agent",
        operation_id=operation.operation_id,
        project_id="project-x",
    )

    assert observed["bucket"] == "project-state"
    assert observed["operation_id"] == operation.operation_id
    assert observed["destroyed"] is True
    assert operation.read()["phase"] == "destroyed"


def _stub_owned_agent_destroy(monkeypatch, tmp_path):
    from npa.cli import agent as agent_module

    monkeypatch.setattr(
        agent_module,
        "_resolve_destroy_tf_vars",
        lambda p, n, r: {
            "nebius_region": "eu-north1",
            "instance_name": f"agent-{p}-{n}",
            "nebius_project_id": "project-x",
        },
    )
    monkeypatch.setattr(agent_module, "_cleanup_agent_ingress", lambda *_a, **_k: None)
    monkeypatch.setattr(
        agent_module, "_cleanup_orphan_agent_instances", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        agent_module,
        "resolve_terraform_state",
        lambda _p: SimpleNamespace(
            bucket="b", access_key="k", secret_key="s", endpoint="e"
        ),
    )
    monkeypatch.setattr(
        agent_module, "_agent_terraform_state_exists", lambda _p, _n: True
    )
    monkeypatch.setattr(
        agent_module.provisioner, "prepare_working_dir", lambda *a, **k: tmp_path
    )
    monkeypatch.setattr(agent_module.provisioner, "init", lambda *a, **k: None)
    return agent_module


def test_agent_destroy_recovers_default_sg_by_deleting_owned_parent_network(
    monkeypatch, tmp_path
) -> None:
    agent_module = _stub_owned_agent_destroy(monkeypatch, tmp_path)
    destroys: list[dict] = []

    def destroy(**kwargs):
        destroys.append(kwargs)
        if len(destroys) == 1:
            raise agent_module.ProvisionerError(
                "rpc error: FailedPrecondition: cannot delete default security group"
            )

    monkeypatch.setattr(agent_module.provisioner, "destroy", destroy)
    monkeypatch.setattr(
        agent_module.provisioner,
        "state_list",
        lambda _tf_dir: ["nebius_vpc_v1_network.workbench"],
    )
    monkeypatch.setattr(
        agent_module.provisioner,
        "state_resource_id",
        lambda address, **kwargs: "vpcnetwork-owned",
    )
    network_delete = Mock()
    monkeypatch.setattr("npa.clients.network.nebius._run", network_delete)

    agent_module._destroy_agent_terraform("prod", "agent", record={"instance_id": "i"})

    assert len(destroys) == 2
    assert all("s3_session_token" not in call["tf_vars"] for call in destroys)
    network_delete.assert_called_once_with(
        ["vpc", "network", "delete", "--id", "vpcnetwork-owned"]
    )


def test_agent_destroy_preserves_unowned_network_on_default_sg_refusal(
    monkeypatch, tmp_path
) -> None:
    agent_module = _stub_owned_agent_destroy(monkeypatch, tmp_path)
    monkeypatch.setattr(
        agent_module.provisioner,
        "destroy",
        lambda **kwargs: (_ for _ in ()).throw(
            agent_module.ProvisionerError(
                "rpc error: FailedPrecondition: cannot delete default security group"
            )
        ),
    )
    monkeypatch.setattr(agent_module.provisioner, "state_list", lambda _tf_dir: [])
    network_delete = Mock()
    monkeypatch.setattr("npa.clients.network.nebius._run", network_delete)

    with pytest.raises(agent_module.ProvisionerError) as caught:
        agent_module._destroy_agent_terraform(
            "prod", "agent", record={"instance_id": "i"}
        )

    assert "reused/shared network" in str(caught.value)
    assert "npa agent destroy" in str(caught.value)
    network_delete.assert_not_called()


def test_agent_destroy_does_not_mask_genuine_nondefault_sg_failure(
    monkeypatch, tmp_path
) -> None:
    agent_module = _stub_owned_agent_destroy(monkeypatch, tmp_path)
    failures = iter(
        [
            agent_module.ProvisionerError(
                "FailedPrecondition: non-default security group is still in use"
            ),
            agent_module.ProvisionerError("second destroy also failed"),
        ]
    )
    monkeypatch.setattr(
        agent_module.provisioner,
        "destroy",
        lambda **kwargs: (_ for _ in ()).throw(next(failures)),
    )
    monkeypatch.setattr(agent_module.provisioner, "state_list", lambda _tf_dir: [])
    network_delete = Mock()
    monkeypatch.setattr("npa.clients.network.nebius._run", network_delete)

    with pytest.raises(agent_module.ProvisionerError) as caught:
        agent_module._destroy_agent_terraform(
            "prod", "agent", record={"instance_id": "i"}
        )

    assert "non-default security group is still in use" in str(caught.value)
    network_delete.assert_not_called()


def test_agent_destroy_retries_an_already_absent_security_group(
    monkeypatch, tmp_path
) -> None:
    agent_module = _stub_owned_agent_destroy(monkeypatch, tmp_path)
    destroys = 0

    def destroy(**kwargs):
        nonlocal destroys
        destroys += 1
        if destroys == 1:
            raise agent_module.ProvisionerError("NotFound: security group is absent")

    monkeypatch.setattr(agent_module.provisioner, "destroy", destroy)
    monkeypatch.setattr(agent_module.provisioner, "state_list", lambda _tf_dir: [])
    network_delete = Mock()
    monkeypatch.setattr("npa.clients.network.nebius._run", network_delete)

    agent_module._destroy_agent_terraform("prod", "agent", record={"instance_id": "i"})

    assert destroys == 2
    network_delete.assert_not_called()


def test_agent_project_option_defaults_are_consistent() -> None:
    """deploy must resolve --project the same way status/destroy do.

    Regression: deploy/fresh-setup/bootstrap/verify-live defaulted --project to the
    static `us-central1` alias while status/destroy resolved the configured
    default, so a `-p`-less deploy stored the agent where a later `-p`-less status
    could not find it — and destroy then reported success on an empty state while
    the real VM and its public IP kept running.
    """
    from npa.cli import agent as agent_module

    import inspect

    for command in (
        agent_module.deploy_cmd,
        agent_module.fresh_setup_cmd,
        agent_module.bootstrap_cmd,
        agent_module.verify_live_cmd,
        agent_module.status_cmd,
        agent_module.destroy_cmd,
    ):
        option = inspect.signature(command).parameters["project"].default
        default = getattr(option, "default", option)
        assert default == "", f"{command.__name__} pins --project to {default!r}"


def test_resolve_project_alias_prefers_the_only_configured_project(monkeypatch) -> None:
    """`default_project_name()` returns "default" for an unset config, naming nothing."""
    from npa.cli import agent as agent_module
    from npa.clients import config as config_module

    monkeypatch.setattr(config_module, "default_project_name", lambda: "default")
    monkeypatch.setattr(
        config_module, "list_projects", lambda: {"tle-workbench": {"project_id": "p-1"}}
    )

    assert agent_module._resolve_project_alias("") == "tle-workbench"
    assert agent_module._resolve_project_alias("explicit") == "explicit"


def test_resolve_project_alias_uses_the_configured_default_when_present(
    monkeypatch,
) -> None:
    from npa.cli import agent as agent_module
    from npa.clients import config as config_module

    monkeypatch.setattr(config_module, "default_project_name", lambda: "prod")
    monkeypatch.setattr(
        config_module,
        "list_projects",
        lambda: {"prod": {"project_id": "p-1"}, "dev": {"project_id": "p-2"}},
    )

    assert agent_module._resolve_project_alias("") == "prod"


def test_ssh_egress_check_warns_when_outbound_ssh_is_blocked(monkeypatch) -> None:
    """Deploy waits for the new VM's tcp/22 from this host, then rolls it back."""
    from npa.cli.agent_network import PROBE_ENV_VAR, _agent_ssh_egress_result

    monkeypatch.setenv(PROBE_ENV_VAR, "ssh.example:22")

    def _timeout(address, timeout):
        raise TimeoutError("timed out")

    result = _agent_ssh_egress_result(connect=_timeout)

    assert result.status == "WARN"
    assert "tcp/22" in result.summary
    assert "VPN" in result.remedy


def test_ssh_egress_check_passes_when_the_probe_connects(monkeypatch) -> None:
    from npa.cli.agent_network import PROBE_ENV_VAR, _agent_ssh_egress_result

    monkeypatch.setenv(PROBE_ENV_VAR, "ssh.example:2222")
    closed: list[bool] = []

    class _Socket:
        def close(self) -> None:
            closed.append(True)

    seen: list[tuple[str, int]] = []

    def _connect(address, timeout):
        seen.append(address)
        return _Socket()

    result = _agent_ssh_egress_result(connect=_connect)

    assert result.status == "PASS"
    assert seen == [("ssh.example", 2222)]
    assert closed == [True]
    # A generic host that answers proves less than it looks like it does.
    assert "split" in result.summary


def test_ssh_egress_check_prefers_a_recorded_nebius_agent_ip(monkeypatch) -> None:
    """A split tunnel can allow github.com:22 and still drop a fresh cloud IP.

    Regression: the check reported PASS off github.com while the operator's agent
    VM was unreachable on 22/443/8088, and the deploy then burned the boot window
    and rolled the VM back.
    """
    from npa.cli import agent_network
    from npa.cli.agent_network import PROBE_ENV_VAR, _agent_ssh_egress_result
    from npa.clients import config as config_module

    monkeypatch.delenv(PROBE_ENV_VAR, raising=False)
    monkeypatch.setattr(
        config_module,
        "list_projects",
        lambda: {"prod": {"agents": {"agent": {"public_ip": "203.0.113.50"}}}},
    )
    assert agent_network.recorded_agent_ip() == "203.0.113.50"

    seen: list[tuple[str, int]] = []

    def _timeout(address, timeout):
        seen.append(address)
        raise TimeoutError("timed out")

    result = _agent_ssh_egress_result(connect=_timeout)

    assert seen == [("203.0.113.50", 22)]
    assert result.status == "WARN"
    assert "your agent VM" in result.summary


def test_ssh_egress_check_is_quiet_without_dns(monkeypatch) -> None:
    """No DNS says nothing about SSH egress, so it must not warn."""
    import socket

    from npa.cli.agent_network import PROBE_ENV_VAR, _agent_ssh_egress_result

    monkeypatch.setenv(PROBE_ENV_VAR, "ssh.example:22")

    def _no_dns(address, timeout):
        raise socket.gaierror("Name or service not known")

    assert _agent_ssh_egress_result(connect=_no_dns).status == "PASS"


def test_ssh_egress_check_can_be_disabled(monkeypatch) -> None:
    from npa.cli.agent_network import PROBE_ENV_VAR, _agent_ssh_egress_result

    monkeypatch.setenv(PROBE_ENV_VAR, "off")

    def _must_not_run(address, timeout):  # pragma: no cover - must not run
        raise AssertionError("the probe must not open a socket when disabled")

    result = _agent_ssh_egress_result(connect=_must_not_run)
    assert result.status == "PASS"
    assert "skipped" in result.summary


def test_public_ip_quota_gate_skips_an_agent_that_already_has_its_ip(
    monkeypatch,
) -> None:
    """Re-deploying an existing agent reuses its address, so require no headroom."""
    from npa.cli.agent import _agent_check_public_ip_quota
    from npa.clients import nebius as nebius_module

    monkeypatch.setattr(nebius_module, "get_project_region", lambda _pid: "us-central1")
    monkeypatch.setattr(
        nebius_module,
        "get_public_ipv4_quota",
        lambda _tid, _region: (_ for _ in ()).throw(
            AssertionError("quota must not be queried for an existing agent")
        ),
    )

    # Must not raise, and must not even read the quota.
    _agent_check_public_ip_quota(
        "project-x", "tenant-x", "us-central1", agent_exists=True
    )


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


def test_artifact_role_summary_uses_the_declared_role() -> None:
    """Guard the conflict-prone semantic-role/artifact-role UI merge."""

    script = rendered_agent_ui_html().split("<script>")[-1].split("</script>")[0]
    assert "acc[artifactRole] = (acc[artifactRole] || 0) + 1" in script
    assert "acc[role] = (acc[role] || 0) + 1" not in script


def test_boot_rerun_mount_preserves_a_newer_operator_media_preview() -> None:
    """A late boot-time Rerun mount must not replace an explicit replay video."""

    script = rendered_agent_ui_html().split("<script>")[-1].split("</script>")[0]
    assert "const explicitMediaSelected = () => operatorMediaPreviewActive;" in script
    assert "if (explicitMediaSelected())" in script


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
