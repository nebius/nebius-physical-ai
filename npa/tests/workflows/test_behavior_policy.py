"""Verify checkpoint identity and lifecycle of the managed BEHAVIOR policy."""

import argparse
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import subprocess
import threading
from unittest.mock import Mock
import zipfile

import pytest

from npa.workflows.behavior_challenge import (
    comet_policy,
    policy,
    policy_server,
    protocol,
)


@dataclass
class _Camera:
    obs_key: str
    dataset_key: str
    resolution: tuple = (240, 240)


@dataclass
class _Robot:
    name: str
    robot_type: str
    observations: dict
    action: tuple = ("unchanged action indices",)
    proprio: tuple = ("unchanged proprioception indices",)


@pytest.fixture
def baseline_robot():
    links = ("zed_link", "left_realsense_link", "right_realsense_link")
    cameras = {
        f"image_{index}": _Camera(
            f"robot::robot:{link}:Camera:0::rgb", f"observation.rgb.{link}_camera_0"
        )
        for index, link in enumerate(links)
    }
    return _Robot("robot", "R1Pro", cameras)


def test_policy_adapter_reads_official_observations_without_changing_values(
    baseline_robot,
):
    original = asdict(baseline_robot)
    adapted = policy_server._evaluator_robot(baseline_robot)
    observation = {
        "robot_r1::proprio": object(),
        "robot_r1::robot_r1:zed_link:Camera:0::rgb": object(),
        "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb": object(),
        "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb": object(),
    }
    assert observation[f"{adapted.name}::proprio"] is observation["robot_r1::proprio"]
    for camera, link in zip(
        adapted.observations.values(),
        ("zed_link", "left_realsense_link", "right_realsense_link"),
        strict=True,
    ):
        assert (
            observation[camera.obs_key]
            is observation[f"robot_r1::robot_r1:{link}:Camera:0::rgb"]
        )
    assert adapted.action == baseline_robot.action
    assert adapted.proprio == baseline_robot.proprio
    assert asdict(baseline_robot) == original
    for key, camera in adapted.observations.items():
        assert camera.dataset_key == baseline_robot.observations[key].dataset_key
        assert camera.resolution == baseline_robot.observations[key].resolution


@pytest.mark.parametrize("field", ["name", "robot_type", "cameras", "key"])
def test_policy_adapter_refuses_unexpected_upstream_contract(baseline_robot, field):
    if field in {"name", "robot_type"}:
        setattr(baseline_robot, field, "unexpected")
    elif field == "cameras":
        baseline_robot.observations.pop("image_2")
    else:
        baseline_robot.observations["image_0"].obs_key = "privileged::state"
    with pytest.raises(ValueError, match="Unexpected baseline"):
        policy_server._evaluator_robot(baseline_robot)


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
    with pytest.raises(ValueError, match="files or normalization"):
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


def test_comet_policy_dispatch_is_native_and_task_bound(prepared, monkeypatch):
    args, plan, output = prepared
    args.policy_kind = "comet12"
    args.policy_execution_variant = "native"
    args.policy_task_name = "turning_on_radio"
    command = ["/runtime/python", "comet_server.py"]
    prepare = Mock(return_value=command)
    monkeypatch.setattr(comet_policy, "prepare_policy", prepare)
    monkeypatch.setattr(policy, "_healthy", lambda port: False)

    assert policy._prepare_policy(args, plan, output) == command
    prepare.assert_called_once_with(args, plan, output)

    args.policy_execution_variant = "final-stage-backtrack"
    with pytest.raises(ValueError, match="Unsupported comet12"):
        with policy.managed_policy(args, plan, output):
            pytest.fail("must refuse")


def test_comet_policy_requires_task_name_and_rejects_it_for_other_kinds(prepared):
    args, plan, output = prepared
    args.policy_kind = "comet12"
    args.policy_execution_variant = "native"
    args.policy_task_name = None
    with pytest.raises(ValueError, match="requires --policy-task-name"):
        with policy.managed_policy(args, plan, output):
            pytest.fail("must refuse")

    args.policy_kind = "official"
    args.policy_task_name = "turning_on_radio"
    with pytest.raises(ValueError, match="requires --policy-kind comet12"):
        with policy.managed_policy(args, plan, output):
            pytest.fail("must refuse")


def test_comet_policy_cannot_silently_skip_without_managed_paths(prepared):
    args, plan, output = prepared
    args.policy_kind = "comet12"
    args.policy_execution_variant = "native"
    args.policy_task_name = "turning_on_radio"
    for field in policy.POLICY_FIELDS:
        setattr(args, field, None)

    with pytest.raises(ValueError, match="requires all four policy paths"):
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
    assert evidence["observation_name_adapter"]["sha256"] == protocol.file_digest(
        output / "policy-server.py"
    )
    assert command[1].endswith("/policy_server.py")
    assert "PYTHONPATH" not in factory.call_args.kwargs["env"]


def test_policy_startup_exit_is_not_readiness():
    process = Mock()
    process.poll.return_value = 1
    with pytest.raises(RuntimeError, match="before readiness"):
        policy._wait_for_policy(process, 8000)


def test_policy_startup_times_out_while_process_remains_alive(monkeypatch):
    process = Mock()
    process.poll.return_value = None
    monkeypatch.setattr(policy, "_healthy", lambda port: False)
    monkeypatch.setattr(policy.time, "monotonic", Mock(side_effect=[10, 12]))
    with pytest.raises(RuntimeError, match="within 2 seconds"):
        policy._wait_for_policy(process, 8000, timeout_seconds=2)


def test_policy_startup_returns_when_process_becomes_ready(monkeypatch):
    process = Mock()
    process.poll.return_value = None
    monkeypatch.setattr(policy, "_healthy", lambda port: True)
    monkeypatch.setattr(policy.time, "monotonic", lambda: 10)
    policy._wait_for_policy(process, 8000, timeout_seconds=2)


def test_policy_startup_rejects_health_response_after_deadline(monkeypatch):
    process = Mock()
    process.poll.return_value = None
    monkeypatch.setattr(policy, "_healthy", lambda port: True)
    monkeypatch.setattr(policy.time, "monotonic", Mock(side_effect=[10, 13]))
    with pytest.raises(RuntimeError, match="within 2 seconds"):
        policy._wait_for_policy(process, 8000, timeout_seconds=2)


def test_policy_startup_sleep_does_not_exceed_remaining_time(monkeypatch):
    process = Mock()
    process.poll.return_value = None
    monkeypatch.setattr(policy, "_healthy", lambda port: False)
    monkeypatch.setattr(policy.time, "monotonic", Mock(side_effect=[10, 11.75, 12]))
    sleep = Mock()
    monkeypatch.setattr(policy.time, "sleep", sleep)
    with pytest.raises(RuntimeError, match="within 2 seconds"):
        policy._wait_for_policy(process, 8000, timeout_seconds=2)
    sleep.assert_called_once_with(0.25)


def test_policy_startup_timeout_stops_managed_process(prepared, monkeypatch):
    args, plan, output = prepared
    process = Mock()
    process.poll.return_value = None
    monkeypatch.setattr(policy.subprocess, "Popen", Mock(return_value=process))
    monkeypatch.setattr(policy, "_healthy", lambda port: False)
    monkeypatch.setattr(policy, "_POLICY_STARTUP_TIMEOUT_SECONDS", 0)
    with pytest.raises(RuntimeError, match="within 0 seconds"):
        with policy.managed_policy(args, plan, output):
            pytest.fail("must not evaluate before policy readiness")
    process.terminate.assert_called_once()


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
