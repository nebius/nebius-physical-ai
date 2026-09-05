# Verify application Ray startup isolation and pinned per-pod preparation failures.
"""Run hermetic bootstrap contracts without starting Ray or contacting model hosts."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest
import yaml


EXAMPLE = Path(__file__).parents[2] / "workflows/workbench/ray-clip-development"


def _run_application_start(tmp_path, *, rank, gpus="1"):
    """Execute the real startup shell with only its service executable replaced."""
    script = (EXAMPLE / "cluster/start.sh").read_text()
    command = "exec /tmp/ray-clip-env/bin/ray"
    assert script.count(command) == 1
    recorder = tmp_path / "record.py"
    recorder.write_text(
        "import json, os, sys\n"
        "print(json.dumps({'argv': sys.argv[1:], "
        "'address': os.environ.get('RAY_ADDRESS'), "
        "'driver_on_workers': os.environ.get('RAY_JOB_ALLOW_DRIVER_ON_WORKER_NODES')}))\n"
    )
    script = script.replace(command, shlex.join([sys.executable, str(recorder)]))
    environment = {
        **os.environ,
        "SKYPILOT_NODE_IPS": "192.0.2.10\n192.0.2.11",
        "SKYPILOT_NODE_RANK": str(rank),
        "SKYPILOT_NUM_GPUS_PER_NODE": gpus,
        "RAY_ADDRESS": "127.0.0.1:6380",
    }
    return subprocess.run(
        ["bash", "-c", script], env=environment, cwd=tmp_path,
        capture_output=True, text=True, check=False,
    )


def _assert_disjoint_service_ports(options: dict, rank: int) -> None:
    """Reject collisions with SkyPilot management ports or application workers."""
    services = [
        "--object-manager-port", "--node-manager-port", "--ray-client-server-port",
        "--dashboard-agent-listen-port", "--dashboard-agent-grpc-port",
        "--runtime-env-agent-port", "--metrics-export-port",
    ]
    if rank == 0:
        services.extend(["--port", "--dashboard-port"])
    # These are the observed SkyPilot 0.12.2 management Ray defaults.
    management_ports = {6380, 8266, 8076, 10001, 52365}
    application_ports = [int(options[service]) for service in services]
    first_worker = int(options["--min-worker-port"])
    last_worker = int(options["--max-worker-port"])
    assert 1 <= first_worker <= last_worker <= 10999
    assert len(application_ports) == len(set(application_ports))
    assert set(application_ports).isdisjoint(management_ports)
    for port in application_ports:
        assert 1 <= port <= 10999
        assert not first_worker <= port <= last_worker


@pytest.mark.parametrize(("rank", "gpus"), [(0, "1"), (1, "1"), (0, "2")])
def test_application_ports_and_rank_are_disjoint_from_management_runtime(tmp_path, rank, gpus):
    """Require explicit application placement, private Jobs access and separate ports.

    Args:
        tmp_path: Pytest-owned command recording directory.
        rank: SkyPilot head or worker index.
        gpus: Allocated GPU count passed by SkyPilot.
    Returns:
        None.
    Raises:
        AssertionError: Bootstrap selects ambiguous or conflicting runtime settings.
    """
    result = _run_application_start(tmp_path, rank=rank, gpus=gpus)
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    arguments = observed["argv"]
    assert arguments[0] == "start"
    assert "--block" in arguments
    assert observed["address"] is None
    assert observed["driver_on_workers"] == "0"
    options = dict(argument.split("=", 1) for argument in arguments if "=" in argument)
    assert options["--node-ip-address"] == f"192.0.2.{10 + rank}"
    assert options["--num-gpus"] == gpus
    if rank == 0:
        assert "--head" in arguments
        assert options["--dashboard-host"] == "127.0.0.1"
        assert options["--include-dashboard"] == "true"
        assert "--address" not in options
    else:
        assert options["--address"] == "192.0.2.10:6381"
        assert "--head" not in arguments
    _assert_disjoint_service_ports(options, rank)
    assert "stop" not in arguments


@pytest.mark.parametrize(("rank", "gpus"), [("", "1"), ("0", "")])
def test_missing_skypilot_identity_fails_before_starting_ray(tmp_path, rank, gpus):
    """Refuse missing placement values before invoking the service executable.

    Args:
        tmp_path: Pytest-owned command recording directory.
        rank: Present or missing SkyPilot node rank.
        gpus: Present or missing SkyPilot GPU allocation.
    Returns:
        None.
    Raises:
        AssertionError: A service starts without required placement identity.
    """
    result = _run_application_start(tmp_path, rank=rank, gpus=gpus)
    assert result.returncode != 0
    assert "required" in result.stderr
    assert not result.stdout


def test_cluster_mounts_only_bootstrap_and_keeps_application_submission_external():
    """Keep source submission with Ray Jobs and cluster setup with SkyPilot.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: The cluster task expands source, credentials or placement scope.
    """
    workflow = yaml.safe_load((EXAMPLE / "cluster.yaml").read_text())
    assert workflow["file_mounts"] == {"/tmp/ray-clip-bootstrap": "./cluster"}
    assert "workdir" not in workflow
    assert workflow["resources"]["image_id"].startswith("docker:docker.io/pytorch/pytorch@sha256:")
    assert workflow["resources"]["use_spot"] is False
    kubernetes = workflow["config"]["kubernetes"]
    assert kubernetes["networking"] == "portforward"
    assert kubernetes["pod_config"]["spec"]["automountServiceAccountToken"] is False
    assert workflow["run"].strip() == "bash /tmp/ray-clip-bootstrap/start.sh"


@pytest.fixture
def preparation():
    """Import setup code without executing any package or model operations.

    Args:
        None.
    Returns:
        The preparation module with its main guard unexecuted.
    Raises:
        ImportError: The preparation source cannot be imported.
    """
    source = EXAMPLE / "cluster/prepare.py"
    specification = importlib.util.spec_from_file_location("clip_preparation_test", source)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_corrupt_download_never_reaches_cuda_or_readiness(preparation, tmp_path, monkeypatch):
    """Stop preparation after a bad model digest before CUDA or readiness output.

    Args:
        preparation: Imported, unexecuted setup module.
        tmp_path: Temporary synthetic model and environment directory.
        monkeypatch: Pytest filesystem and subprocess boundary fixture.
    Returns:
        None.
    Raises:
        AssertionError: Corrupt weights are accepted or later preparation runs.
    """
    from unittest.mock import Mock

    model = tmp_path / "model"
    model.mkdir()
    (model / "pytorch_model.bin").write_bytes(b"invalid synthetic weights")
    monkeypatch.setattr(preparation, "MODEL_DIRECTORY", model)
    monkeypatch.setattr(preparation, "APPLICATION_ENVIRONMENT", tmp_path / "environment")
    operations = Mock()
    inspection = Mock(side_effect=AssertionError("Corrupt weights reached CUDA inspection"))
    monkeypatch.setattr(preparation.subprocess, "run", operations)
    monkeypatch.setattr(preparation.subprocess, "check_output", inspection)
    with pytest.raises(RuntimeError, match="pinned public snapshot"):
        preparation.main()
    commands = [call.args[0] for call in operations.call_args_list]
    assert len(commands) == 3
    inspection.assert_not_called()
    assert commands[0][1:3] == ["-m", "venv"]
    assert "--system-site-packages" in commands[0]
    assert commands[1][1:4] == ["-m", "pip", "--python"]
    assert commands[1][4] == str(tmp_path / "environment/bin/python")
    assert commands[2][0] == commands[1][4]
    assert "snapshot_download" in commands[2][-1]
    assert "token=False" in commands[2][-1]
    assert preparation.MODEL_REVISION in commands[2][-1]


def test_dependency_failure_prevents_model_download(preparation, monkeypatch):
    """Propagate failed environment setup instead of downloading model weights.

    Args:
        preparation: Imported, unexecuted setup module.
        monkeypatch: Pytest subprocess boundary fixture.
    Returns:
        None.
    Raises:
        AssertionError: Preparation continues after a failed environment operation.
    """
    from unittest.mock import Mock

    failure = subprocess.CalledProcessError(7, ["isolated-environment"])
    command = Mock(side_effect=failure)
    monkeypatch.setattr(preparation.subprocess, "run", command)
    with pytest.raises(subprocess.CalledProcessError) as caught:
        preparation.main()
    assert caught.value.returncode == 7
    command.assert_called_once()


def test_preparation_timing_excludes_inspection_from_model_download(preparation, tmp_path, monkeypatch):
    """Keep CUDA inspection and dependency freezing outside model-download timing.

    Args:
        preparation: Imported, unexecuted setup module.
        tmp_path: Temporary receipt directory.
        monkeypatch: Pytest clock, process and receipt-destination fixture.
    Returns:
        None.
    Raises:
        AssertionError: Reported phase boundaries include the wrong operations.
    """
    from types import SimpleNamespace
    from unittest.mock import Mock

    destination = tmp_path / "preparation.json"
    monkeypatch.setattr(preparation, "Path", Mock(return_value=destination))
    monkeypatch.setattr(preparation, "time", SimpleNamespace(time=Mock(return_value=150.0)))
    monkeypatch.setattr(preparation, "_inspect_cuda_environment", Mock(return_value={"test_fixture": True}))
    monkeypatch.setattr(preparation.subprocess, "check_output", Mock(return_value="synthetic==1\n"))
    preparation._write_preparation_receipt(100.0, 120.0, 130.0, "synthetic-digest", "fixture-python")
    receipt = json.loads(destination.read_text())
    assert receipt["dependency_preparation_seconds"] == 20.0
    assert receipt["model_fetch_and_verify_seconds"] == 10.0
    assert receipt["cuda_environment_inspection_seconds"] == 20.0
    assert receipt["dependencies_ready_at_unix"] == 120.0
    assert receipt["model_ready_at_unix"] == 130.0
    assert receipt["finished_at_unix"] == 150.0
    assert receipt["dependency_freeze"] == ["synthetic==1"]
