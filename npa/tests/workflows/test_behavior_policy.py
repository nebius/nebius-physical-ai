"""Verify checkpoint identity and lifecycle of the managed BEHAVIOR policy."""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import subprocess
import threading
from unittest.mock import Mock
import zipfile

import pytest

from npa.workflows.behavior_challenge import policy, protocol


@pytest.mark.parametrize("status", [200, 302, 503])
def test_readiness_requires_direct_loopback_health_response(status, monkeypatch):
    paths = []

    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            paths.append(self.path)
            self.send_response(status if self.path == "/healthz" else 200)
            self.send_header("Location", "/redirected")
            self.end_headers()

        def log_message(self, *args):
            pass

    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example.invalid:1")
    with HTTPServer(("127.0.0.1", 0), HealthHandler) as server:
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}
        )
        thread.start()
        try:
            assert policy._healthy(server.server_port) is (status == 200)
            assert paths == ["/healthz"]
        finally:
            server.shutdown()
            thread.join()


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    archive = tmp_path / "checkpoint.zip"
    checkpoint = tmp_path / "checkpoint"
    files = {
        "params/weights": b"synthetic weights",
        "assets/turning_on_radio/norm_stats.json": b"{}",
    }
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, value in files.items():
            path = checkpoint / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
            bundle.writestr(policy.CHECKPOINT_PREFIX + name, value)
    args = argparse.Namespace(
        policy_root=tmp_path,
        policy_python=tmp_path / "python",
        policy_checkpoint=checkpoint,
        policy_archive=archive,
        host="127.0.0.1",
        port=8000,
    )
    plan = {
        "recipe": {
            "tasks": ["turning_on_radio"],
            "policy_checkpoint_sha256": protocol.file_digest(archive),
        },
        "cases": [{"policy_port": None}],
    }
    monkeypatch.setattr(policy, "_verify_source", lambda root: None)
    monkeypatch.setenv("NPA_OPENPI_ACCEPT_GEMMA_TERMS", "YES")
    return args, plan, tmp_path


def test_checkpoint_verification_detects_different_loaded_weights(prepared):
    args, plan, _ = prepared
    (args.policy_checkpoint / "params/weights").write_bytes(b"substitute")
    with pytest.raises(ValueError, match="Loaded checkpoint bytes"):
        policy._verify_checkpoint(
            args.policy_archive,
            args.policy_checkpoint,
            plan["recipe"]["policy_checkpoint_sha256"],
        )


def test_checkpoint_verification_rejects_extra_files(prepared):
    args, plan, _ = prepared
    (args.policy_checkpoint / "model.safetensors").write_bytes(b"alternate model")
    with pytest.raises(ValueError, match="files or radio normalization"):
        policy._verify_checkpoint(
            args.policy_archive,
            args.policy_checkpoint,
            plan["recipe"]["policy_checkpoint_sha256"],
        )


def test_checkpoint_verification_rejects_wrong_archive(prepared):
    args, _, _ = prepared
    with pytest.raises(ValueError, match="archive differs"):
        policy._verify_checkpoint(args.policy_archive, args.policy_checkpoint, "0" * 64)


@pytest.mark.parametrize("value", ["", "NO", "yes"])
def test_terms_refuse_before_checkpoint_access(prepared, monkeypatch, value):
    args, plan, output = prepared
    monkeypatch.setenv("NPA_OPENPI_ACCEPT_GEMMA_TERMS", value)
    monkeypatch.setattr(
        policy, "_verify_checkpoint", lambda *a: pytest.fail("checkpoint access")
    )
    with pytest.raises(ValueError, match="scoped operator acceptance"):
        with policy.managed_policy(args, plan, output):
            pytest.fail("must refuse")


def test_radio_checkpoint_cannot_be_used_for_other_tasks(prepared):
    args, plan, output = prepared
    plan["recipe"]["tasks"] = ["another_task"]
    with pytest.raises(ValueError, match="turning_on_radio only"):
        with policy.managed_policy(args, plan, output):
            pytest.fail("must refuse")


def test_partial_policy_configuration_refuses(prepared):
    args, plan, output = prepared
    args.policy_archive = None
    with pytest.raises(ValueError, match="all four"):
        with policy.managed_policy(args, plan, output):
            pytest.fail("must refuse")


def test_unrelated_ready_endpoint_refuses(prepared, monkeypatch):
    args, plan, output = prepared
    monkeypatch.setattr(policy, "_healthy", lambda port: True)
    monkeypatch.setattr(
        policy.subprocess, "Popen", lambda *a, **k: pytest.fail("spawn")
    )
    with pytest.raises(ValueError, match="unrelated endpoint"):
        with policy.managed_policy(args, plan, output):
            pytest.fail("must refuse")


def test_policy_stops_after_evaluator_failure_and_records_loaded_identity(
    prepared, monkeypatch
):
    args, plan, output = prepared
    process = Mock()
    process.poll.return_value = None
    factory = Mock(return_value=process)
    monkeypatch.setattr(policy.subprocess, "Popen", factory)
    responses = iter([False, True])
    monkeypatch.setattr(policy, "_healthy", lambda port: next(responses))
    with pytest.raises(RuntimeError, match="evaluator failed"):
        with policy.managed_policy(args, plan, output):
            raise RuntimeError("evaluator failed")
    process.terminate.assert_called_once()
    command = factory.call_args.args[0]
    assert command[command.index("--repo-id") + 1] == "turning_on_radio"
    evidence = json.loads((output / "policy-provenance.json").read_text())
    assert (
        evidence["checkpoint_archive_sha256"]
        == plan["recipe"]["policy_checkpoint_sha256"]
    )
    assert evidence["memory_compliance"] == "unverified"
    assert "PYTHONPATH" not in factory.call_args.kwargs["env"]


def test_policy_startup_exit_is_not_readiness():
    process = Mock()
    process.poll.return_value = 1
    with pytest.raises(RuntimeError, match="before readiness"):
        policy._wait_for_policy(process, 8000)


def test_stuck_policy_is_killed_during_cleanup():
    process = Mock()
    process.poll.return_value = None
    process.wait.side_effect = [subprocess.TimeoutExpired("fixture", 30), 0]
    policy._stop_policy(process)
    process.kill.assert_called_once()


def test_checkpoint_rejects_archive_traversal(prepared):
    args, _, _ = prepared
    with zipfile.ZipFile(args.policy_archive, "w") as bundle:
        bundle.writestr(policy.CHECKPOINT_PREFIX + "../../outside", b"bad")
    with pytest.raises(ValueError, match="layout"):
        policy._verify_checkpoint(
            args.policy_archive,
            args.policy_checkpoint,
            protocol.file_digest(args.policy_archive),
        )
