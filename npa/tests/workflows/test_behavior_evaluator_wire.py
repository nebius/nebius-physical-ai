"""Exercise the version-bound singleton BEHAVIOR evaluator wire adapter."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import msgpack
import numpy as np
import pytest

from npa.workflows.behavior_challenge import (
    comet_policy,
    evaluator_versions,
    evaluator_wire,
    rlc_observations,
    rlc_policy,
)

LEGACY = evaluator_versions.UPSTREAM_COMMITS["3.9.2"]
CURRENT = evaluator_versions.UPSTREAM_COMMITS["3.9.3"]


def _legacy_observation() -> dict:
    value = {evaluator_wire.PROPRIOCEPTION: np.arange(61, dtype=np.float32)}
    for camera, key in zip(evaluator_wire.CAMERAS, evaluator_wire.RGB_KEYS):
        size = 720 if camera == "zed_link" else 480
        value[key] = np.zeros((size, size, 4), dtype=np.uint8)
    value["private_task_state"] = np.array([7], dtype=np.int64)
    return value


def _current_observation() -> dict:
    value = {key: array[None] for key, array in _legacy_observation().items()}
    value["robot_r1::cam_rel_poses"] = np.zeros((1, 21), dtype=np.float32)
    value["task_id"] = np.zeros((1, 1), dtype=np.int64)
    for camera in evaluator_wire.CAMERAS:
        size = 720 if camera == "zed_link" else 480
        key = f"robot_r1::robot_r1:{camera}:Camera:0::depth_linear"
        value[key] = np.zeros((1, size, size), dtype=np.float32)
    return value


def _codec():
    def default(value):
        if not isinstance(value, np.ndarray):
            raise TypeError(type(value))
        return {
            "__ndarray__": True,
            "data": value.tobytes(),
            "dtype": value.dtype.str,
            "shape": value.shape,
        }

    def hook(value):
        if value.get("__ndarray__") is not True:
            return value
        return np.frombuffer(value["data"], dtype=value["dtype"]).reshape(
            value["shape"]
        )

    return SimpleNamespace(
        Packer=lambda: SimpleNamespace(
            pack=lambda value: msgpack.packb(value, default=default)
        ),
        unpackb=lambda value: msgpack.unpackb(value, object_hook=hook),
    )


class _Socket:
    def __init__(self, messages):
        codec = _codec()
        self.messages = [codec.Packer().pack(message) for message in messages]
        self.sent = []

    async def send(self, value):
        self.sent.append(value)

    async def __aiter__(self):
        for message in self.messages:
            yield message


def _server(path: Path, name: str, monkeypatch):
    monkeypatch.setitem(sys.modules, "comet_policy", comet_policy)
    monkeypatch.setitem(sys.modules, "rlc_observations", rlc_observations)
    monkeypatch.setitem(sys.modules, "evaluator_versions", evaluator_versions)
    monkeypatch.setitem(sys.modules, "evaluator_wire", evaluator_wire)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_current_singleton_unbatches_only_policy_inputs_and_rebatches_action():
    wire = evaluator_wire.EvaluatorWire(CURRENT)
    source = _current_observation()

    selected = wire.observation_for_policy(source)
    action = wire.action_for_evaluator(np.arange(23, dtype=np.float32))

    assert set(selected) == set(evaluator_wire.POLICY_OBSERVATIONS)
    assert selected[evaluator_wire.PROPRIOCEPTION].shape == (61,)
    assert selected[evaluator_wire.RGB_KEYS[0]].shape == (720, 720, 4)
    assert "task_id" not in selected and "private_task_state" not in selected
    assert action.shape == (1, 23)


def test_legacy_wire_preserves_unbatched_policy_inputs_and_action():
    wire = evaluator_wire.EvaluatorWire(LEGACY)
    source = _legacy_observation()

    selected = wire.observation_for_policy(source)
    action = np.arange(23, dtype=np.float32)

    assert selected[evaluator_wire.PROPRIOCEPTION] is source["robot_r1::proprio"]
    assert wire.action_for_evaluator(action) is action


@pytest.mark.parametrize("batch_size", [0, 2, 10])
def test_current_wire_rejects_non_singleton_before_policy(batch_size):
    observation = _current_observation()
    observation["task_id"] = np.zeros((batch_size, 1), dtype=np.int64)

    with pytest.raises(ValueError, match="one consistent batch row"):
        evaluator_wire.EvaluatorWire(CURRENT).observation_for_policy(observation)


@pytest.mark.parametrize(
    "message, error",
    [
        ({"reset": False}, "reset"),
        ({"reset": 1}, "reset"),
        ({"reset": np.bool_(True)}, "reset"),
        ({"reset": True, "other": 1}, "reset"),
        ({"__action_chunk_size__": 4}, "chunk"),
    ],
)
def test_reset_and_chunk_messages_fail_closed(message, error):
    wire = evaluator_wire.EvaluatorWire(CURRENT)
    if "reset" in message:
        with pytest.raises(ValueError, match=error):
            wire.is_reset(message)
    else:
        with pytest.raises(ValueError, match=error):
            wire.observation_for_policy(message)


def test_unknown_version_and_cross_version_payloads_reject():
    with pytest.raises(ValueError, match="supported official"):
        evaluator_wire.EvaluatorWire("0" * 40)
    with pytest.raises(ValueError, match="not unbatched"):
        evaluator_wire.EvaluatorWire(LEGACY).observation_for_policy(
            _current_observation()
        )
    with pytest.raises(ValueError, match="batch row"):
        evaluator_wire.EvaluatorWire(CURRENT).observation_for_policy(
            _legacy_observation()
        )


def test_current_wire_rejects_nonnumeric_privileged_leaf():
    observation = _current_observation()
    observation["private_task_state"] = np.array(["secret"])

    with pytest.raises(ValueError, match="numeric array leaf"):
        evaluator_wire.EvaluatorWire(CURRENT).observation_for_policy(observation)


def test_comet_connection_uses_real_msgpack_bytes_and_current_shapes(monkeypatch):
    path = Path(comet_policy.__file__).with_name("comet_server.py")
    server = _server(path, "wire_comet_server", monkeypatch)
    codec = _codec()
    monkeypatch.setitem(
        sys.modules, "openpi_client", SimpleNamespace(msgpack_numpy=codec)
    )
    policy = Mock()
    policy.act.return_value = np.zeros((1, 23), dtype=np.float32)
    socket = _Socket([{"reset": True}, _current_observation()])

    asyncio.run(server._connection(socket, policy, upstream_commit=CURRENT))

    inputs = policy.act.call_args.args[0]
    assert set(inputs) == set(evaluator_wire.POLICY_OBSERVATIONS)
    assert inputs["robot_r1::proprio"].shape == (61,)
    assert _codec().unpackb(socket.sent[-1])["action"].shape == (1, 23)


def test_comet_rejects_n2_before_model_act(monkeypatch):
    path = Path(comet_policy.__file__).with_name("comet_server.py")
    server = _server(path, "wire_comet_n2_server", monkeypatch)
    monkeypatch.setitem(
        sys.modules, "openpi_client", SimpleNamespace(msgpack_numpy=_codec())
    )
    observation = _current_observation()
    observation["task_id"] = np.zeros((2, 1), dtype=np.int64)
    policy = Mock()

    with pytest.raises(ValueError, match="batch row"):
        asyncio.run(server._connection(_Socket([observation]), policy, None, CURRENT))
    policy.act.assert_not_called()


def test_rlc_connection_uses_current_wire_without_changing_policy_shape(monkeypatch):
    path = Path(rlc_policy.__file__).with_name("rlc_server.py")
    server = _server(path, "wire_rlc_server", monkeypatch)
    monkeypatch.setitem(
        sys.modules, "openpi_client", SimpleNamespace(msgpack_numpy=_codec())
    )
    tensor = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.zeros(23)))
    policy = Mock()
    policy.act.return_value = tensor
    socket = _Socket([_current_observation()])

    asyncio.run(server._connection(socket, policy, CURRENT))

    assert policy.act.call_args.args[0]["robot_r1::proprio"].shape == (61,)
    assert _codec().unpackb(socket.sent[-1])["action"].shape == (1, 23)


def test_managed_rlc_command_carries_exact_current_revision(tmp_path):
    args = SimpleNamespace(
        policy_python=Path("/runtime/python"),
        policy_root=Path("/runtime/source"),
        policy_checkpoint=Path("/runtime/checkpoint"),
        port=8000,
    )

    command = rlc_policy._command(args, 1, tmp_path, upstream_commit=CURRENT)

    index = command.index("--upstream-commit")
    assert command[index + 1] == CURRENT
    with pytest.raises(ValueError, match="supported official"):
        rlc_policy._command(args, 1, tmp_path, upstream_commit="f" * 40)


def test_managed_preparation_requires_explicit_revision():
    with pytest.raises(ValueError, match="lacks its evaluator revision"):
        comet_policy._plan_upstream_commit({})
    with pytest.raises(ValueError, match="lacks its evaluator revision"):
        rlc_policy._plan_upstream_commit({})


@pytest.mark.parametrize("kind", ["comet", "rlc"])
def test_staged_server_cli_imports_versioned_wire_portably(tmp_path, kind):
    output = tmp_path / kind
    output.mkdir()
    if kind == "comet":
        comet_policy._stage_adapters(output, comet_policy.COMET12_PROFILE)
        server = output / "comet_server.py"
    else:
        rlc_policy._adapter_files(output, selected=False)
        server = output / "rlc_server.py"

    result = subprocess.run(
        [sys.executable, str(server), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--upstream-commit" in result.stdout
