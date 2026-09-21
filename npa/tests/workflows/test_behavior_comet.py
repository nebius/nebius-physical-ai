"""Focused contracts for the simulation-independent Comet12 adapter."""

import asyncio
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
from unittest.mock import Mock
import zipfile

import numpy as np
import pytest

from npa.workflows.behavior_challenge import comet_policy
from npa.workflows.behavior_challenge import __main__ as behavior_main


@pytest.fixture
def observation():
    value = {"robot_r1::proprio": np.arange(61, dtype=np.float32)}
    for camera in comet_policy.CAMERAS:
        size = 720 if camera == "zed_link" else 480
        key = f"robot_r1::robot_r1:{camera}:Camera:0::rgb"
        value[key] = np.zeros((size, size, 4), dtype=np.uint8)
    value["robot_r1::robot_r1:zed_link:Camera:0::depth_linear"] = np.ones(
        (720, 720), dtype=np.float32
    )
    value["task_state"] = {"forbidden": True}
    return value


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setitem(sys.modules, "comet_policy", comet_policy)
    path = Path(comet_policy.__file__).with_name("comet_server.py")
    spec = importlib.util.spec_from_file_location("test_comet_server", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git_blob(value: bytes) -> str:
    digest = hashlib.sha1(f"blob {len(value)}\0".encode(), usedforsecurity=False)
    digest.update(value)
    return digest.hexdigest()


def test_frozen_checkpoint_inventory_has_exact_public_identity():
    manifest = comet_policy.load_checkpoint_manifest()

    assert manifest["repository"] == "sunshk/openpi_comet"
    assert manifest["revision"] == comet_policy.MODEL_REVISION
    assert manifest["file_count"] == len(manifest["files"]) == 5948
    assert manifest["total_bytes"] == 12_441_382_544
    assert manifest["lfs_file_count"] == 2436
    assert manifest["lfs_bytes"] == 12_411_371_755
    assert manifest["task_ids"] == list(comet_policy.SUPPORTED_TASK_IDS)


def test_checkpoint_verifier_checks_lfs_git_blob_and_exact_set(tmp_path):
    lfs = b"large weight bytes"
    metadata = b"metadata"
    (tmp_path / "params").mkdir()
    (tmp_path / "params/weight").write_bytes(lfs)
    (tmp_path / "metadata").write_bytes(metadata)
    manifest = {
        "files": {
            "params/weight": {
                "size": len(lfs),
                "sha256": hashlib.sha256(lfs).hexdigest(),
            },
            "metadata": {
                "size": len(metadata),
                "git_blob_sha1": _git_blob(metadata),
            },
        }
    }

    assert (
        comet_policy._verify_checkpoint_files(tmp_path, manifest) == manifest["files"]
    )
    (tmp_path / "metadata").write_bytes(b"substitute")
    with pytest.raises(ValueError, match="size differs"):
        comet_policy._verify_checkpoint_files(tmp_path, manifest)
    (tmp_path / "metadata").write_bytes(metadata)
    (tmp_path / "extra").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="file set differs"):
        comet_policy._verify_checkpoint_files(tmp_path, manifest)


def test_checkpoint_verifier_rejects_symlink(tmp_path):
    target = tmp_path / "target"
    target.write_bytes(b"bytes")
    (tmp_path / "link").symlink_to(target)
    manifest = {
        "files": {
            "target": {"size": 5, "git_blob_sha1": _git_blob(b"bytes")},
            "link": {"size": 5, "git_blob_sha1": _git_blob(b"bytes")},
        }
    }
    with pytest.raises(ValueError, match="regular file"):
        comet_policy._verify_checkpoint_files(tmp_path, manifest)


def test_checkpoint_archive_binds_full_sha_prefix_and_extracted_bytes(
    tmp_path, monkeypatch
):
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "metadata").write_bytes(b"metadata")
    manifest = {
        "files": {
            "metadata": {
                "size": 8,
                "git_blob_sha1": _git_blob(b"metadata"),
            }
        }
    }
    archive = tmp_path / "checkpoint.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(f"{comet_policy.CHECKPOINT_NAME}/metadata", b"metadata")
    expected = hashlib.sha256(archive.read_bytes()).hexdigest()
    monkeypatch.setattr(comet_policy, "load_checkpoint_manifest", lambda: manifest)

    assert (
        comet_policy.verify_checkpoint_archive(archive, root, expected)
        == manifest["files"]
    )
    with pytest.raises(ValueError, match="archive differs"):
        comet_policy.verify_checkpoint_archive(archive, root, "0" * 64)
    (root / "metadata").write_bytes(b"changed!")
    with pytest.raises(ValueError, match="differs from its archive"):
        comet_policy.verify_checkpoint_archive(archive, root, expected)


def test_checkpoint_manifest_rejects_malformed_identity(tmp_path):
    manifest = comet_policy.load_checkpoint_manifest()
    relative = next(iter(manifest["files"]))
    manifest["files"][relative] = {"size": 1, "sha256": "not-a-digest"}
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="digest is malformed"):
        comet_policy.load_checkpoint_manifest(path)


def test_source_verifier_requires_pinned_clean_bytes(tmp_path, monkeypatch):
    source = tmp_path / "source.py"
    source.write_text("pinned\n")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(comet_policy, "SOURCE_FILES", {"source.py": expected})
    replies = iter([comet_policy.SOURCE_COMMIT + "\n", ""])
    monkeypatch.setattr(
        comet_policy.subprocess,
        "check_output",
        lambda *args, **kwargs: next(replies),
    )

    assert comet_policy.verify_source(tmp_path) == {"source.py": expected}

    replies = iter([comet_policy.SOURCE_COMMIT + "\n", "?? shadow.py\n"])
    monkeypatch.setattr(
        comet_policy.subprocess,
        "check_output",
        lambda *args, **kwargs: next(replies),
    )
    with pytest.raises(ValueError, match="clean pinned checkout"):
        comet_policy.verify_source(tmp_path)

    source.write_text("substituted\n")
    replies = iter([comet_policy.SOURCE_COMMIT + "\n", ""])
    monkeypatch.setattr(
        comet_policy.subprocess,
        "check_output",
        lambda *args, **kwargs: next(replies),
    )
    with pytest.raises(ValueError, match="source file bytes differ"):
        comet_policy.verify_source(tmp_path)


def test_observation_mapping_excludes_depth_and_evaluator_metadata(observation):
    result = comet_policy.policy_observation(observation)

    assert set(result) == {
        "robot_r1::proprio",
        *(f"robot_r1::robot_r1:{name}:Camera:0::rgb" for name in comet_policy.CAMERAS),
    }
    assert result["robot_r1::proprio"] is observation["robot_r1::proprio"]
    assert all(
        value.shape[-1] == 3 for key, value in result.items() if key.endswith("::rgb")
    )
    assert "task_state" not in result
    assert not any("depth" in key for key in result)


def test_proprioception_map_matches_current_r1pro_vector(observation):
    state = observation["robot_r1::proprio"]
    indices = comet_policy.PROPRIOCEPTION_INDICES["R1Pro"]

    assert state[indices["base_qvel"]].tolist() == [0, 1, 2]
    assert state[indices["arm_left_qpos"]].tolist() == list(range(3, 10))
    assert state[indices["gripper_left_qpos"]].tolist() == [24, 25]
    assert state[indices["arm_right_qpos"]].tolist() == list(range(28, 35))
    assert state[indices["gripper_right_qpos"]].tolist() == [49, 50]
    assert state[indices["trunk_qpos"]].tolist() == [53, 54, 55, 56]


@pytest.mark.parametrize(
    "state", [np.zeros(256), np.full(61, np.nan), np.zeros((1, 61))]
)
def test_observation_mapping_rejects_invalid_proprioception(observation, state):
    observation["robot_r1::proprio"] = state
    with pytest.raises(ValueError, match="61-element"):
        comet_policy.policy_observation(observation)


@pytest.mark.parametrize(
    "image",
    [np.zeros((224, 224, 3), dtype=np.uint8), np.zeros((720, 720, 3))],
)
def test_observation_mapping_rejects_invalid_camera(observation, image):
    observation["robot_r1::robot_r1:zed_link:Camera:0::rgb"] = image
    with pytest.raises(ValueError, match="camera|uint8"):
        comet_policy.policy_observation(observation)


def test_action_adapter_requires_one_finite_23_vector():
    action = np.arange(23, dtype=np.float32)[None]
    assert np.array_equal(comet_policy.validate_action(action), action[0])

    with pytest.raises(ValueError, match="one 23-element"):
        comet_policy.validate_action(action[0])
    action[0, 4] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        comet_policy.validate_action(action)


def test_task_identity_binds_supported_name_and_index(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "task_mapping.json").write_text(
        json.dumps({"picking_up_trash": {"task_index": 1, "task": "Pick trash."}})
    )
    assert comet_policy.task_identity(tmp_path, 1, "picking_up_trash")["task"]
    with pytest.raises(ValueError, match="name and ID"):
        comet_policy.task_identity(tmp_path, 1, "turning_on_radio")
    with pytest.raises(ValueError, match="does not declare"):
        comet_policy.task_identity(tmp_path, 2, "picking_up_trash")


def test_prepare_policy_stages_only_adapters_and_public_provenance(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    (source / "scripts/task_mapping.json").write_text(
        json.dumps({"picking_up_trash": {"task_index": 1, "task": "Pick trash."}})
    )
    checkpoint = tmp_path / "private-checkpoint"
    checkpoint.mkdir()
    upstream = tmp_path / "upstream"
    (upstream / "docs/challenge").mkdir(parents=True)
    (upstream / "docs/challenge/task_data.json").write_text(
        json.dumps(
            {
                "tasks": [
                    {"id": "turning_on_radio"},
                    {"id": "picking_up_trash"},
                ]
            }
        )
    )
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(comet_policy, "verify_source", lambda root: {})
    monkeypatch.setattr(
        comet_policy,
        "verify_checkpoint_archive",
        lambda archive, root, sha: {"params/private": {}},
    )
    args = SimpleNamespace(
        policy_python=Path("/runtime/python"),
        policy_root=source,
        policy_checkpoint=checkpoint,
        policy_archive=tmp_path / "private-checkpoint.zip",
        policy_task_name="picking_up_trash",
        upstream_root=upstream,
        port=8000,
    )
    plan = {
        "recipe": {
            "split": "development",
            "tasks": ["picking_up_trash"],
            "policy_checkpoint_sha256": "a" * 64,
        },
        "cases": [{"policy_port": 8000}],
    }

    command = comet_policy.prepare_policy(args, plan, output)
    provenance = json.loads((output / "policy-provenance.json").read_text())

    assert command[0:2] == ["/runtime/python", str(output / "comet_server.py")]
    assert command[-6:] == [
        "--task-id",
        "1",
        "--task-name",
        "picking_up_trash",
        "--port",
        "8000",
    ]
    assert provenance["redistribution"] == "private-runtime-checkpoint"
    assert provenance["evaluation_status"] == "not_evaluated"
    assert provenance["checkpoint_file_count"] == 1
    assert set(provenance["adapters"]) == {
        "comet_policy.py",
        "comet_server.py",
        "comet12-checkpoint.json",
    }
    assert not any("private" in name for name in provenance["adapters"])


def test_prepare_policy_rejects_task_or_split_drift(tmp_path):
    args = SimpleNamespace(policy_task_name="picking_up_trash")
    for recipe in (
        {"split": "other", "tasks": ["picking_up_trash"]},
        {"split": "development", "tasks": ["turning_on_radio"]},
        {"split": "development", "tasks": ["a", "b"]},
    ):
        with pytest.raises(ValueError, match="Comet"):
            comet_policy._campaign_task(args, {"recipe": recipe})


def test_prepare_policy_rejects_missing_or_mismatched_case_ports():
    args = SimpleNamespace(port=8000)
    with pytest.raises(ValueError, match="at least one"):
        comet_policy._verify_campaign_ports(args, {"cases": []})
    with pytest.raises(ValueError, match="port must match"):
        comet_policy._verify_campaign_ports(args, {"cases": [{"policy_port": 8001}]})
    comet_policy._verify_campaign_ports(args, {"cases": [{"policy_port": None}]})


def test_campaign_task_rejects_current_registry_reordering(tmp_path):
    source = tmp_path / "source"
    upstream = tmp_path / "upstream"
    (source / "scripts").mkdir(parents=True)
    (upstream / "docs/challenge").mkdir(parents=True)
    (source / "scripts/task_mapping.json").write_text(
        json.dumps({"picking_up_trash": {"task_index": 1, "task": "Pick trash."}})
    )
    (upstream / "docs/challenge/task_data.json").write_text(
        json.dumps(
            {
                "tasks": [
                    {"id": "picking_up_trash"},
                    {"id": "turning_on_radio"},
                ]
            }
        )
    )
    args = SimpleNamespace(
        policy_root=source,
        upstream_root=upstream,
        policy_task_name="picking_up_trash",
    )
    plan = {"recipe": {"split": "development", "tasks": ["picking_up_trash"]}}

    with pytest.raises(ValueError, match="current evaluator registry"):
        comet_policy._campaign_task(args, plan)


def test_source_overlay_patches_only_simulation_import(tmp_path, monkeypatch):
    source = tmp_path / "source"
    policy = source / "src/openpi/policies/b1k_policy.py"
    policy.parent.mkdir(parents=True)
    old = "from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES"
    policy.write_text(old + "\nPINNED = True\n")
    monkeypatch.setattr(comet_policy, "verify_source", lambda root: {})

    overlay = comet_policy.build_source_overlay(source, tmp_path / "overlay")

    assert policy.read_text().startswith(old)
    patched = (overlay / "openpi/policies/b1k_policy.py").read_text()
    assert "from comet_policy import PROPRIOCEPTION_INDICES" in patched
    assert patched.endswith("PINNED = True\n")


def test_loader_uses_pinned_config_and_upstream_wrapper(tmp_path, server, monkeypatch):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/task_mapping.json").write_text(
        json.dumps({"picking_up_trash": {"task_index": 1, "task": "Pick trash."}})
    )
    created = Mock(return_value="model")
    wrapper = Mock(return_value="wrapped")
    config = Mock(return_value="config")
    modules = {
        "openpi": ModuleType("openpi"),
        "openpi.policies": ModuleType("openpi.policies"),
        "openpi.policies.policy_config": SimpleNamespace(create_trained_policy=created),
        "openpi.shared": ModuleType("openpi.shared"),
        "openpi.shared.eval_b1k_wrapper": SimpleNamespace(B1KPolicyWrapper=wrapper),
        "openpi.training": ModuleType("openpi.training"),
        "openpi.training.config": SimpleNamespace(get_config=config),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    args = SimpleNamespace(
        source_root=tmp_path,
        checkpoint=tmp_path / "checkpoint",
        task_id=1,
        task_name="picking_up_trash",
    )
    original = Path.cwd()

    assert server._load_policy(args, tmp_path / "overlay") == "wrapped"
    assert Path.cwd() == original
    config.assert_called_once_with("pi05_b1k-base")
    created.assert_called_once_with("config", args.checkpoint)
    wrapper.assert_called_once_with(
        "model",
        task_name="picking_up_trash",
        control_mode="receeding_horizon",
        max_len=32,
        fine_grained_level=0,
    )


def test_connection_resets_state_and_returns_original_action_shape(
    observation, server, monkeypatch
):
    codec = SimpleNamespace(
        Packer=lambda: SimpleNamespace(pack=lambda value: value),
        unpackb=lambda value: value,
    )
    monkeypatch.setitem(
        sys.modules, "openpi_client", SimpleNamespace(msgpack_numpy=codec)
    )
    policy = Mock()
    policy.act.return_value = np.zeros((1, 23), dtype=np.float32)

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
    assert socket.sent[0] == {"policy": "comet12-2025-transfer"}
    assert all(item["action"].shape == (23,) for item in socket.sent[1:])


def test_server_parser_restricts_checkpoint_tasks(server):
    common = [
        "--source-root",
        "/source",
        "--checkpoint",
        "/checkpoint",
        "--task-name",
        "picking_up_trash",
        "--port",
        "8000",
    ]
    assert server.parser().parse_args([*common, "--task-id", "1"]).task_id == 1
    with pytest.raises(SystemExit):
        server.parser().parse_args([*common, "--task-id", "2"])


def test_managed_cli_exposes_comet_kind_and_task_name():
    value = argparse.ArgumentParser()
    behavior_main._add_policy_arguments(value)

    args = value.parse_args(
        ["--policy-kind", "comet12", "--policy-task-name", "picking_up_trash"]
    )

    assert args.policy_kind == "comet12"
    assert args.policy_task_name == "picking_up_trash"


def test_health_endpoint_is_the_only_readiness_surface(server):
    connection = SimpleNamespace(respond=Mock(return_value="ready"))

    assert server._health(connection, SimpleNamespace(path="/healthz")) == "ready"
    connection.respond.assert_called_once_with(200, "OK\n")
    assert server._health(connection, SimpleNamespace(path="/other")) is None
