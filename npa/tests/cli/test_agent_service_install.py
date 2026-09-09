"""Interrupted installers must resume safely without hiding their first failure."""

from __future__ import annotations

import hashlib
import re
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from npa.cli import agent_service_install as subject
from npa.clients.ssh import SSHError


def _install(ssh, script="set -eu\necho install\n", *, resuming=True, stage=None):
    return subject.install_agent_services(
        ssh, setup_script=script, stage_source=stage or Mock(), resuming=resuming
    )


def test_exact_receipt_skips_source_staging_and_installer() -> None:
    script = "set -eu\necho install\n"
    digest = hashlib.sha256(script.encode()).hexdigest()
    ssh = Mock()
    ssh.run.return_value = (0, digest + "\n", "")
    stage = Mock()
    assert _install(ssh, script, stage=stage) is True
    stage.assert_not_called()
    ssh.upload_private_text.assert_not_called()
    ssh.run_or_raise.assert_not_called()


@pytest.mark.parametrize("receipt", [(1, "", "missing"), (0, "other-digest", "")])
def test_missing_or_changed_receipt_runs_complete_install(receipt) -> None:
    ssh = Mock()
    ssh.run.return_value = receipt
    stage = Mock()
    assert _install(ssh, stage=stage) is False
    stage.assert_called_once_with(ssh)
    ssh.run_or_raise.assert_called_once()


def test_explicit_bootstrap_runs_even_with_matching_receipt() -> None:
    ssh = Mock()
    assert _install(ssh, resuming=False) is False
    assert "services-installed.sha256" not in str(ssh.run.call_args_list)
    ssh.run_or_raise.assert_called_once()


def test_unreadable_receipt_does_not_authorize_installation() -> None:
    ssh = Mock()
    ssh.run.side_effect = SSHError("connection lost")
    with pytest.raises(SSHError, match="connection lost"):
        _install(ssh)
    ssh.upload_private_text.assert_not_called()


def test_lost_final_ssh_response_preserves_remote_receipt() -> None:
    ssh = Mock()
    script = "set -eu\necho install\n"
    digest = hashlib.sha256(script.encode()).hexdigest()
    ssh.run.return_value = (1, "", "missing")

    def completed_before_transport_loss(*_args, **_kwargs):
        uploaded, _path = ssh.upload_private_text.call_args.args
        assert uploaded.startswith(script)
        assert digest in uploaded[len(script) :]
        ssh.run.return_value = (0, digest + "\n", "")
        raise SSHError("lost final response")

    ssh.run_or_raise.side_effect = completed_before_transport_loss
    with pytest.raises(SSHError, match="lost final response"):
        _install(ssh, script)
    assert _install(ssh, script) is True
    assert ssh.run_or_raise.call_count == 1


def test_cleanup_transport_error_does_not_hide_install_failure(caplog) -> None:
    ssh = Mock()
    original = SSHError("installer failed")
    ssh.run_or_raise.side_effect = original
    ssh.run.side_effect = SSHError("cleanup failed")
    with pytest.raises(SSHError) as caught:
        _install(ssh, resuming=False)
    assert caught.value is original
    assert "preserving the original error" in caplog.text


def test_cleanup_failure_after_success_remains_an_error() -> None:
    ssh = Mock()
    ssh.run.side_effect = SSHError("cleanup failed")
    with pytest.raises(SSHError, match="cleanup failed"):
        _install(ssh, resuming=False)


@pytest.fixture
def bootstrap_harness(monkeypatch):
    from npa.cli import agent

    manifest = {"bootstrap_timestamp": "first-attempt", "commit": "source-one"}
    ssh = Mock()
    ssh.run.return_value = (1, "", "missing")

    def installed(*_args, **_kwargs):
        script = ssh.upload_private_text.call_args.args[0]
        digest = re.findall(r"builtin printf '%s\\n' ([a-f0-9]{64})", script)[-1]
        ssh.run.return_value = (0, digest, "")

    ssh.run_or_raise.side_effect = installed
    monkeypatch.setattr(agent, "SSHClient", lambda **_kwargs: ssh)
    monkeypatch.setattr(
        agent, "resolve_ssh_config", lambda **_kw: SimpleNamespace(ssh={})
    )
    monkeypatch.setattr(agent, "build_deployment_manifest", Mock(return_value=manifest))
    monkeypatch.setattr(
        agent, "assert_remote_owner_if_present", Mock(return_value=manifest)
    )
    stage = Mock()
    monkeypatch.setattr(agent, "_stage_agent_npa_source", stage)
    llm = Mock()
    monkeypatch.setattr(agent.agent_llm_config, "write_agent_llm_env", llm)
    for writer in (
        "_write_agent_s3_env",
        "_write_agent_artifact_sources_env",
        "_write_agent_operator_profile",
        "_record_remote_setup_ready",
    ):
        monkeypatch.setattr(agent, writer, Mock())
    credentials = Mock(side_effect=[SSHError("staging interrupted"), None])
    monkeypatch.setattr(agent, "_write_agent_nebius_env", credentials)
    monkeypatch.setattr(agent, "verify_remote_deployment", Mock())
    return agent, manifest, stage, credentials, llm


@pytest.mark.parametrize("change_password", [False, True])
def test_bootstrap_retry_restages_credentials_and_honors_changed_settings(
    bootstrap_harness, change_password, tmp_path
) -> None:
    agent, manifest, stage, credentials, llm = bootstrap_harness
    kwargs = dict(
        host="192.0.2.10",
        ssh_user="ubuntu",
        ssh_key_path=str(tmp_path / "id_ed25519"),
        project_alias="test",
        agent_name="agent",
        project_id="project-test",
        tenant_id="tenant-test",
        region="test-region",
        auth_user="npa",
        auth_password="first-password",
        agent_port=8088,
        backend_port=8787,
        rerun_port=9090,
    )
    with pytest.raises(SSHError, match="staging interrupted"):
        agent._bootstrap_agent_stack(**kwargs)
    agent.build_deployment_manifest.return_value = {
        **manifest,
        "bootstrap_timestamp": "retry",
    }
    if change_password:
        kwargs["auth_password"] = "changed-password"
    agent._bootstrap_agent_stack(**kwargs, resume_services=True)
    assert stage.call_count == (2 if change_password else 1)
    assert credentials.call_count == llm.call_count == 2
    assert agent.verify_remote_deployment.call_args.args[1] == manifest


@pytest.mark.parametrize("exit_code", [0, 7])
def test_receipt_is_atomic_private_and_written_only_after_success(
    tmp_path, monkeypatch, exit_code
) -> None:
    import os
    import subprocess

    sudo = tmp_path / "sudo"
    sudo.write_text('#!/bin/sh\nexec "$@"\n')
    sudo.chmod(0o700)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    receipt_script = (
        subject._receipt_script("test-digest")
        .replace("/opt/npa-agent", str(tmp_path))
        .replace("mv -fT", "mv -f")
    )  # BSD mv has no -T; target is a regular file.
    result = subprocess.run(
        ["bash", "-c", f"set -eu\n(exit {exit_code})\n" + receipt_script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == exit_code
    receipt = tmp_path / "services-installed.sha256"
    assert receipt.exists() is (exit_code == 0)
    if exit_code == 0:
        assert receipt.read_text() == "test-digest\n"
        assert receipt.stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob(".services-installed.*"))
