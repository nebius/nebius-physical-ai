"""Exercise RLC transfer boundaries, published weight identity and episode resets."""

import argparse
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from npa.workflows.behavior_challenge import rlc_observations, rlc_policy


@pytest.fixture
def observation():
    value = {"robot_r1::proprio": np.arange(61, dtype=np.float32)}
    for camera in rlc_observations.CAMERAS:
        size = 720 if camera == "zed_link" else 480
        value[f"robot_r1::robot_r1:{camera}:Camera:0::rgb"] = np.zeros(
            (size, size, 4), dtype=np.uint8
        )
    return value


def test_policy_cannot_receive_task_state_or_instance_metadata(observation):
    observation.update(task_id=[49], instance_id=311, object_pose="forbidden")
    result = rlc_observations.policy_observation(observation)
    assert len(result) == 4
    assert "task_id" not in result
    assert "instance_id" not in result
    assert "object_pose" not in result
    assert result["robot_r1::proprio"] is observation["robot_r1::proprio"]
    for name, value in result.items():
        if name.endswith("::rgb"):
            assert value.shape[-1] == 3
            assert np.shares_memory(value, observation[name])


@pytest.mark.parametrize(
    "state", [np.zeros(256), np.full(61, np.nan), np.zeros((1, 61))]
)
def test_old_or_invalid_proprioception_is_rejected(observation, state):
    observation["robot_r1::proprio"] = state
    with pytest.raises(ValueError, match="61-element"):
        rlc_observations.policy_observation(observation)


@pytest.mark.parametrize(
    "image", [np.zeros((224, 224, 3), dtype=np.uint8), np.zeros((720, 720, 3))]
)
def test_unexpected_camera_contract_is_rejected(observation, image):
    observation["robot_r1::robot_r1:zed_link:Camera:0::rgb"] = image
    with pytest.raises(ValueError, match="camera"):
        rlc_observations.policy_observation(observation)


def test_proprioception_selects_correct_arms_grippers_and_trunk(observation):
    state = observation["robot_r1::proprio"]
    indices = rlc_observations.PROPRIOCEPTION_INDICES["R1Pro"]
    assert state[indices["arm_left_qpos"]].tolist() == list(range(3, 10))
    assert state[indices["arm_right_qpos"]].tolist() == list(range(28, 35))
    assert state[indices["gripper_left_qpos"]].tolist() == [24, 25]
    assert state[indices["gripper_right_qpos"]].tolist() == [49, 50]
    assert state[indices["trunk_qpos"]].tolist() == [53, 54, 55, 56]
    assert state[indices["base_qvel"]].tolist() == [0, 1, 2]


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setitem(sys.modules, "rlc_observations", rlc_observations)
    path = Path(rlc_policy.__file__).with_name("rlc_server.py")
    spec = importlib.util.spec_from_file_location("test_rlc_server", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_adapter_preserves_original_checkout(tmp_path, server, monkeypatch):
    old = "from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES"
    relative = (
        "src/b1k/policies/b1k_policy.py",
        "src/b1k/shared/eval_b1k_wrapper.py",
        "openpi/src/openpi/policies/b1k_policy.py",
    )
    for name in relative:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(old + "\n# frozen model code\n")
    monkeypatch.setattr(sys, "path", sys.path.copy())
    server._policy_source(tmp_path, tmp_path / "overlay")
    for name in relative:
        assert (tmp_path / name).read_text().startswith(old)
        module = name.removeprefix("openpi/").removeprefix("src/")
        patched = (tmp_path / "overlay" / module).read_text()
        assert "from rlc_observations import PROPRIOCEPTION_INDICES" in patched
        assert patched.endswith("# frozen model code\n")


def test_connection_resets_memory_without_sending_reset_response(
    server, observation, monkeypatch
):
    codec = SimpleNamespace(
        Packer=lambda: SimpleNamespace(pack=lambda x: x), unpackb=lambda x: x
    )
    monkeypatch.setitem(
        sys.modules, "openpi_client", SimpleNamespace(msgpack_numpy=codec)
    )
    policy = Mock()
    policy.act.return_value.cpu.return_value.numpy.return_value = np.zeros(23)

    class Socket:
        def __init__(self):
            self.sent = []

        async def send(self, value):
            self.sent.append(value)

        async def __aiter__(self):
            for value in ({"reset": True}, observation, {"reset": True}, observation):
                yield value

    socket = Socket()
    asyncio.run(server._connection(socket, policy))
    assert policy.reset.call_count == 4
    assert policy.act.call_count == 2
    assert len(socket.sent) == 3
    assert all("action" in response for response in socket.sent[1:])


@pytest.mark.parametrize(
    "split,tasks",
    [("report", ["radio"]), ("development", "all"), ("development", ["a", "b"])],
)
def test_unvalidated_transfer_cannot_enter_reporting(split, tasks, tmp_path):
    plan = {"recipe": {"split": split, "tasks": tasks}}
    with pytest.raises(ValueError, match="one development task"):
        rlc_policy.prepare_policy(argparse.Namespace(), plan, tmp_path)


def test_published_file_verifier_rejects_substituted_model(tmp_path, monkeypatch):
    weights = tmp_path / "weights"
    weights.write_bytes(b"different weights")
    identity = {"weights": {"size": 17, "sha256": "0" * 64}}
    manifest = {
        "revision": rlc_policy.MODEL_REVISION,
        "checkpoints": {"test": identity},
    }
    monkeypatch.setattr(rlc_policy.json, "loads", lambda _: manifest)
    with pytest.raises(ValueError, match="bytes differ"):
        rlc_policy._verify_published_files(
            tmp_path,
            "test",
            {"weights": hashlib.sha256(weights.read_bytes()).hexdigest()},
        )


def test_task_mapping_rejects_new_tasks_and_reordered_registry(tmp_path):
    old = tmp_path / "BEHAVIOR-1K/docs/challenge/task_data.json"
    new = tmp_path / "docs/challenge/task_data.json"
    for path, count in ((old, 50), (new, 100)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"tasks": [{"id": str(i)} for i in range(count)]}))
    with pytest.raises(ValueError, match="original 50"):
        rlc_policy._task_checkpoint(tmp_path, tmp_path, "51")
    data = json.loads(new.read_text())
    data["tasks"][0], data["tasks"][1] = data["tasks"][1], data["tasks"][0]
    new.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="original 50"):
        rlc_policy._task_checkpoint(tmp_path, tmp_path, "0")
