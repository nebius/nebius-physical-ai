from __future__ import annotations

import json
import hashlib
import shutil
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest
import yaml

from npa.workflows.sim2real.robot_contract import (
    ROBOT_CONTRACT_SCHEMA,
    RobotContractError,
    assert_robot_contract,
    materialize_robot_contract,
    stock_robot_contract,
)
from npa.workflows.sim2real import (
    isaac_robot_asset,
    isaac_stage_contract,
    workflow_stage,
)
from npa.genesis import robot_assets


URDF = """<?xml version="1.0"?>
<robot name="test_panda">
  <link name="base"/>
  <link name="arm1"><visual><geometry><mesh filename="meshes/arm.stl"/></geometry></visual></link>
  <link name="arm2"/>
  <link name="hand"/>
  <link name="left_finger"/>
  <link name="right_finger"/>
  <joint name="joint1" type="revolute"><parent link="base"/><child link="arm1"/></joint>
  <joint name="joint2" type="revolute"><parent link="arm1"/><child link="arm2"/></joint>
  <joint name="joint3" type="revolute"><parent link="arm2"/><child link="hand"/></joint>
  <joint name="left_joint" type="prismatic"><parent link="hand"/><child link="left_finger"/></joint>
  <joint name="right_joint" type="prismatic"><parent link="hand"/><child link="right_finger"/></joint>
</robot>
"""

ROOT = Path(__file__).resolve().parents[3]


def _doc(**overrides: object) -> dict:
    payload = {
        "schema": "npa.sim2real.robot_spec.v1",
        "robot_source": "byo_urdf",
        "name": "test_panda",
        "robot_uri": "s3://inputs/robot/pkg/robot.urdf",
        "asset_root_uri": "s3://inputs/robot/pkg",
        "base_link": "base",
        "ee_link": "hand",
        "finger_links": ["left_finger", "right_finger"],
        "joint_names": [
            "joint1",
            "joint2",
            "joint3",
            "left_joint",
            "right_joint",
        ],
        "gripper_joint_names": ["left_joint", "right_joint"],
        "n_arm_joints": 3,
        "n_gripper_joints": 2,
        "kp": [100, 100, 100, 20, 20],
        "kv": [10, 10, 10, 2, 2],
        "force_lower": [-10, -10, -10, -2, -2],
        "force_upper": [10, 10, 10, 2, 2],
        "home_qpos": [0, -0.5, 0.5, 0.04, 0.04],
        "gripper_open": 0.04,
        "gripper_close": 0.0,
        "task": {"skill": "lift", "workspace_reach_m": 0.6},
        "license": {"spdx": "BSD-3-Clause"},
        "provenance": {"source": "https://example.invalid/public-robot"},
    }
    payload.update(overrides)
    return payload


class FakeStorage:
    def __init__(self, doc: dict, *, include_mesh: bool = True) -> None:
        self.objects = {
            "s3://inputs/spec.json": json.dumps(doc).encode(),
            "s3://inputs/robot/pkg/robot.urdf": URDF.encode(),
        }
        if include_mesh:
            self.objects["s3://inputs/robot/pkg/meshes/arm.stl"] = b"solid arm\n"
        self.uploads: dict[str, bytes] = {}

    def download_file(self, uri: str, path: str) -> str:
        if uri not in self.objects:
            raise FileNotFoundError(uri)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.objects[uri])
        return str(target)

    def download_directory(self, uri: str, path: str) -> str:
        prefix = uri.rstrip("/") + "/"
        target = Path(path)
        for key, value in self.objects.items():
            if key.startswith(prefix):
                output = target / key.removeprefix(prefix)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(value)
        return str(target)

    def upload_directory(self, path: str, uri: str) -> str:
        source = Path(path)
        for item in source.rglob("*"):
            if item.is_file():
                self.uploads[
                    uri.rstrip("/") + "/" + item.relative_to(source).as_posix()
                ] = item.read_bytes()
        return uri

    def upload_file(self, path: str, uri: str) -> str:
        self.uploads[uri] = Path(path).read_bytes()
        return uri


def _materialize(tmp_path: Path, storage: FakeStorage) -> dict:
    return materialize_robot_contract(
        robot_spec_uri="s3://inputs/spec.json",
        root_uri="s3://private-run/root",
        work_dir=tmp_path,
        client=storage,
    )


def test_stage2_materializes_content_addressed_urdf_contract(tmp_path: Path) -> None:
    storage = FakeStorage(_doc())
    contract = _materialize(tmp_path, storage)

    assert contract["schema"] == ROBOT_CONTRACT_SCHEMA
    assert contract["status"] == "immutable_urdf_validated"
    assert assert_robot_contract(contract) == contract["embodiment_digest"]
    spec = contract["robot_spec"]
    assert spec["robot_uri"].startswith(
        "s3://private-run/root/stage_02_assets/robot/by-sha256/"
    )
    assert spec["urdf_dependencies"] == ["meshes/arm.stl"]
    assert spec["expected_action_dim"] == 4
    assert spec["expected_observation_dim"] == 24
    assert contract["embodiment"]["arm_joint_names"] == [
        "joint1",
        "joint2",
        "joint3",
    ]
    assert contract["content_addressed_uri"] in storage.uploads
    parsed = urlparse(contract["content_addressed_uri"])
    assert contract["contract_digest"] in parsed.path


def test_committed_panda_example_is_a_complete_parseable_robot_spec() -> None:
    example = (
        ROOT
        / "npa"
        / "workflows"
        / "workbench"
        / "npa-workflows"
        / "examples"
        / "robot-spec-panda-urdf.json"
    )
    document = json.loads(example.read_text())
    spec = robot_assets.parse_robot_spec(document)
    assert spec.robot_source == robot_assets.ROBOT_SOURCE_BYO_URDF
    assert spec.base_link == "panda_link0"
    assert spec.ee_link == "panda_hand"
    assert spec.joint_names[-2:] == spec.gripper_joint_names
    assert spec.dof_count == 9
    assert (
        document["provenance"]["commit"] == "c55b102711fc0aebe80c6952d2ce97c38110abba"
    )


def test_canonical_yaml_exposes_robot_spec_uri_in_stage2_argv() -> None:
    workflow = yaml.safe_load(
        (
            ROOT / "npa" / "workflows" / "workbench" / "npa-workflows" / "sim2real.yaml"
        ).read_text()
    )
    assert workflow["config"]["robot_spec_uri"] == ""
    argv = workflow["states"]["stage-02-assets"]["run"]["argv"]
    index = argv.index("--robot-spec-uri")
    assert argv[index + 1] == "{{config.robot_spec_uri}}"
    outputs = workflow["states"]["stage-02-assets"]["outputs"]
    assert any(
        item["schema"] == ROBOT_CONTRACT_SCHEMA
        and item["uri"].endswith("consumed_robot_spec.json")
        for item in outputs
    )


def test_embodiment_digest_depends_on_content_not_request_uri(tmp_path: Path) -> None:
    first_storage = FakeStorage(_doc())
    first = _materialize(tmp_path / "first", first_storage)
    second_storage = FakeStorage(_doc())
    second_storage.objects["s3://inputs/alternate-spec.json"] = (
        second_storage.objects.pop("s3://inputs/spec.json")
    )
    second = materialize_robot_contract(
        robot_spec_uri="s3://inputs/alternate-spec.json",
        root_uri="s3://private-run/root",
        work_dir=tmp_path / "second",
        client=second_storage,
    )
    assert first["embodiment_digest"] == second["embodiment_digest"]


def test_missing_urdf_mesh_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(RobotContractError, match="dependency.*missing"):
        _materialize(tmp_path, FakeStorage(_doc(), include_mesh=False))


def test_invalid_link_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(RobotContractError, match="links absent.*unknown_hand"):
        _materialize(tmp_path, FakeStorage(_doc(ee_link="unknown_hand")))


def test_invalid_ordered_gripper_joints_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(RobotContractError, match="ordered trailing gripper joints"):
        _materialize(
            tmp_path,
            FakeStorage(_doc(gripper_joint_names=["right_joint", "left_joint"])),
        )


def test_gripper_task_rejects_robot_without_gripper(tmp_path: Path) -> None:
    document = _doc(
        joint_names=["joint1", "joint2", "joint3"],
        gripper_joint_names=[],
        finger_links=[],
        n_gripper_joints=0,
        kp=[100, 100, 100],
        kv=[10, 10, 10],
        force_lower=[-10, -10, -10],
        force_upper=[10, 10, 10],
        home_qpos=[0, -0.5, 0.5],
    )
    with pytest.raises(RobotContractError, match="unsupported gripper contract"):
        _materialize(tmp_path, FakeStorage(document))


def test_inaccessible_and_non_s3_robot_specs_are_actionable(tmp_path: Path) -> None:
    storage = FakeStorage(_doc())
    with pytest.raises(RobotContractError, match="must be an accessible s3"):
        materialize_robot_contract(
            robot_spec_uri="https://example.invalid/spec.json",
            root_uri="s3://private-run/root",
            work_dir=tmp_path / "http",
            client=storage,
        )
    with pytest.raises(RobotContractError, match="robot_spec_uri is inaccessible"):
        materialize_robot_contract(
            robot_spec_uri="s3://inputs/missing.json",
            root_uri="s3://private-run/root",
            work_dir=tmp_path / "missing",
            client=storage,
        )


def test_contract_tamper_is_rejected(tmp_path: Path) -> None:
    contract = _materialize(tmp_path, FakeStorage(_doc()))
    contract["robot_spec"]["expected_action_dim"] = 999
    with pytest.raises(RobotContractError, match="contract digest mismatch"):
        assert_robot_contract(contract)


@pytest.mark.parametrize(
    ("stage", "operation"), [(7, "prepare"), (9, "fetch"), (10, "fetch")]
)
def test_canonical_isaac_stages_receive_same_contract_and_dimensions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: int,
    operation: str,
) -> None:
    contract = _materialize(tmp_path, FakeStorage(_doc()))
    root = "s3://private-run/prefix/run-123"

    def fake_read(uri: str, *, directory: Path) -> dict:
        del directory
        if uri.endswith("task-contract.json"):
            return {"task_contract_digest": "task-digest"}
        return contract

    monkeypatch.setattr(isaac_stage_contract, "read_json", fake_read)
    monkeypatch.setattr(isaac_stage_contract, "source_sha", lambda: "a" * 40)
    monkeypatch.setenv("NPA_TASK_IMAGE", "registry.invalid/isaac@sha256:" + "b" * 64)
    args = SimpleNamespace(
        root_uri=root,
        run_id="run-123",
        stage=stage,
        task_id="Isaac-Lift-Cube-Franka-v0",
        capture_fps="10",
        capture_width="64",
        capture_height="64",
        png_compress_level="2",
    )

    env = workflow_stage._common_isaac_env(
        args, split_uri=root + "/envs/train/envs.jsonl"
    )
    dimensions = contract["embodiment"]["dimensions"]
    assert env["NPA_BYO_ROBOT_TASK"] == "1"
    assert env["NPA_SIM2REAL_ROBOT_SPEC_URI"].endswith("consumed_robot_spec.json")
    assert env["NPA_SIM2REAL_EMBODIMENT_DIGEST"] == contract["embodiment_digest"]
    assert env["NPA_SIM2REAL_EXPECTED_ACTION_DIM"] == dimensions["action"]
    assert env["NPA_SIM2REAL_EXPECTED_OBSERVATION_DIM"] == dimensions["observation"]
    assert env["NPA_SIM2REAL_ROBOT_ASSET_OPERATION"] == operation


def test_stock_contract_does_not_change_isaac_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = "s3://private-run/prefix/run-123"

    def fake_read(uri: str, *, directory: Path) -> dict:
        del directory
        if uri.endswith("task-contract.json"):
            return {"task_contract_digest": "task-digest"}
        return stock_robot_contract()

    monkeypatch.setattr(isaac_stage_contract, "read_json", fake_read)
    monkeypatch.setattr(isaac_stage_contract, "source_sha", lambda: "a" * 40)
    monkeypatch.setenv("NPA_TASK_IMAGE", "registry.invalid/isaac@sha256:" + "b" * 64)
    args = SimpleNamespace(
        root_uri=root,
        run_id="run-123",
        stage=7,
        task_id="Isaac-Lift-Cube-Franka-v0",
        capture_fps="10",
        capture_width="64",
        capture_height="64",
        png_compress_level="2",
    )
    env = workflow_stage._common_isaac_env(
        args, split_uri=root + "/envs/train/envs.jsonl"
    )
    assert "NPA_BYO_ROBOT_TASK" not in env
    assert "NPA_SIM2REAL_ROBOT_SPEC_URI" not in env
    assert "NPA_SIM2REAL_ROBOT_ASSET_OPERATION" not in env


def test_resolved_usd_fetch_verifies_manifest_and_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    usd = tmp_path / "published.usd"
    usd.write_bytes(b'#usda 1.0\ndef Xform "robot" {}\n')
    usd_sha = hashlib.sha256(usd.read_bytes()).hexdigest()
    spec = {
        "asset_root_uri": "s3://private-run/source/",
        "source_relative_path": "robot.urdf",
        "source_tree_sha256": "a" * 64,
        "source_sha256": "b" * 64,
        "source_format": "urdf",
        "embodiment_digest": "c" * 64,
        "resolved_usd_uri": "s3://private-run/resolved/robot.usd",
        "resolved_manifest_uri": "s3://private-run/resolved/manifest.json",
        "expected_action_dim": 8,
        "expected_observation_dim": 36,
    }
    manifest = {
        "embodiment_digest": spec["embodiment_digest"],
        "source_tree_sha256": spec["source_tree_sha256"],
        "expected_action_dim": spec["expected_action_dim"],
        "expected_observation_dim": spec["expected_observation_dim"],
        "usd_sha256": usd_sha,
    }
    manifest_path = tmp_path / "published-manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    def fake_download(uri: str, destination: Path) -> None:
        source = manifest_path if uri.endswith("manifest.json") else usd
        shutil.copy2(source, destination)

    monkeypatch.setattr(isaac_robot_asset, "_spec", lambda: spec)
    monkeypatch.setattr(isaac_robot_asset, "_download", fake_download)
    monkeypatch.setenv("NPA_ROBOT_WORK_DIR", str(tmp_path / "runtime"))
    result = isaac_robot_asset.fetch()
    assert result["usd_sha256"] == usd_sha
    assert (
        tmp_path / "runtime" / "resolved" / "robot.usd"
    ).read_bytes() == usd.read_bytes()

    manifest["expected_action_dim"] = 9
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(isaac_robot_asset.IsaacRobotAssetError, match="parity mismatch"):
        isaac_robot_asset.fetch()


def test_isaac_prepare_converts_urdf_and_publishes_digest_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_bytes = b'<robot name="r"><link name="base"/></robot>\n'
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    tree_hasher = hashlib.sha256()
    tree_hasher.update(b"robot.urdf\0" + source_sha.encode() + b"\0")
    spec = {
        "asset_root_uri": "s3://private-run/source/",
        "source_relative_path": "robot.urdf",
        "source_tree_sha256": tree_hasher.hexdigest(),
        "source_sha256": source_sha,
        "source_format": "urdf",
        "embodiment_digest": "c" * 64,
        "resolved_usd_uri": "s3://private-run/resolved/robot.usd",
        "resolved_manifest_uri": "s3://private-run/resolved/manifest.json",
        "expected_action_dim": 2,
        "expected_observation_dim": 16,
        "base_link": "base",
        "ee_link": "tool",
        "finger_links": ["left", "right"],
    }
    uploads: dict[str, bytes] = {}

    def fake_download_tree(uri: str, destination: Path) -> None:
        assert uri == spec["asset_root_uri"]
        destination.mkdir(parents=True)
        (destination / "robot.urdf").write_bytes(source_bytes)

    def fake_upload(path: Path, uri: str) -> None:
        uploads[uri] = path.read_bytes()

    class FakeApp:
        def close(self) -> None:
            # Isaac Kit shutdown may terminate the interpreter.  The immutable
            # handoff must therefore be durable before close begins.
            assert spec["resolved_usd_uri"] in uploads
            assert spec["resolved_manifest_uri"] in uploads

    class FakeLauncher:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["headless"] is True
            self.app = FakeApp()

    class FakeCfg:
        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)

    class FakeConverter:
        def __init__(self, cfg: FakeCfg) -> None:
            self.usd_path = str(Path(cfg.usd_dir) / cfg.usd_file_name)
            Path(self.usd_path).write_bytes(b'#usda 1.0\ndef Xform "robot" {}\n')

    app_module = types.ModuleType("isaaclab.app")
    app_module.AppLauncher = FakeLauncher
    converters_module = types.ModuleType("isaaclab.sim.converters")
    converters_module.UrdfConverter = FakeConverter
    converters_module.UrdfConverterCfg = FakeCfg
    monkeypatch.setitem(sys.modules, "isaaclab.app", app_module)
    monkeypatch.setitem(sys.modules, "isaaclab.sim.converters", converters_module)
    monkeypatch.setattr(isaac_robot_asset, "_spec", lambda: spec)
    monkeypatch.setattr(isaac_robot_asset, "_download_tree", fake_download_tree)
    monkeypatch.setattr(isaac_robot_asset, "_upload", fake_upload)
    monkeypatch.setattr(isaac_robot_asset, "_validate_usd", lambda path, value: None)
    monkeypatch.setenv("NPA_ROBOT_WORK_DIR", str(tmp_path / "runtime"))

    manifest = isaac_robot_asset.prepare()
    assert manifest["converter"] == "isaaclab.sim.converters.UrdfConverter"
    assert (
        manifest["usd_sha256"]
        == hashlib.sha256(uploads[spec["resolved_usd_uri"]]).hexdigest()
    )
    published_manifest = json.loads(uploads[spec["resolved_manifest_uri"]])
    assert published_manifest["embodiment_digest"] == spec["embodiment_digest"]
